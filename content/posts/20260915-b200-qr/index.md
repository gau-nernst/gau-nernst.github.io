+++
date = '2026-09-15T20:00:00+08:00'
title = "World's fastest (panel) QR factorization on B200"
url = 'b200-qr'
+++
GPU MODE recently hosted a series of leaderboards for [linear algebra kernels](https://www.gpumode.com/news/linear-algebra-kernels-age-of-research). I participated in the first one, QR decomposition, and claimed the 2nd place. I think it's a breath of fresh air, a departure from the usual LLM-oriented kernels most people program for nowadays.

As usual, my submission is open-sourced at [gau-nernst/gpu-mode-kernels](https://github.com/gau-nernst/gpu-mode-kernels/tree/fb291890aa1fe35648799bf7b2731056240bcabf/linalg/qr_v2).

{{< toc >}}

## Introduction to QR decomposition

I don't want to touch too much on this as I'm no expert on the topic (in fact, I only learned about QR decomposition, mostly from Codex, during the competition). There are excellent resources online explaining this, such as [the one from Michael](https://ml-mike.com/writing/qr_v2/). Instead, I will focus on my thought process tackling the crux of the challenge: chained dependency.

```python
# input: A[M, N]

# we can't parallelize the outer loop
for n in range(N-1):
    reflector = householder(A[n:, n])

    # we can parallelize the inner loop
    for tail in range(n+1, N):
        A[n:, tail] = reflect(A[N:, tail], reflector)
```

As you can see from the rough outline of the QR algorithm above, computing the Householder for column n requires the reflectors from all previous columns, making it impossible to parallelize the Householder computation. However, the trailing columns update can be parallelized easily as they are independent. One simple strategy is to let 1 warp computes the Householder (computing Householder requires a reduction of that column, hence we want to use a single warp to avoid cross-warp communication), then multiple warps can update the trailing columns at the same time. This is the crux of my approach to (panel) QR decomposition.

In terms of the problem shapes, there are 7 unique shapes, with some shapes are generated in certain special ways.

ID | Batch | n | Case
---|---|---|---
0 | 20 | 32
1 | 40 | 176
2 | 40 | 352
3 | 640 | 512
4 | 60 | 1024
5 | 8 | 2048
6 | 2 | 4096
7 | 640 | 512 | mixed
8 | 60 | 1024 | mixed
9 | 640 | 512 | rankdef
10 | 640 | 512 | clustered
11 | 60 | 1024 | nearrank

## QR32: Single-warp, register-resident kernel

QR32 is the first shape. This is a tiny amount of data: using 1 warp, only 32 registers per thread are required to hold the full matrix. Hence, for the first problem shape, I worked on a fully register-resident kernel i.e. 1 warp solves one 32x32 QR matrix. An advantage is that there is no inter-warp communication and no shared memory usage at all.

```cpp
// pseudocode
// row-major input A[i,j]
// thread layout: element A[i,j] is held by thread/lane i, in its j-th register
float x[32];

// load data
for (int i = 0; i < 32 / 8; i++)
  ldg_v8_f32(x + i * 8, A_ptr + lane_id * 32 + i * 8);

// main loop
#pragma unroll
for (int col = 0; col < 31; col++) {
  // compute Householder
  // this involves computing warp sum (vector norm)
  float v = ...;

  // trailing columns update
  #pragma unroll
  for (int trail = col + 1; trail < 32; trail++) {
    // this involves computing warp sum (dot product)
    x[trail] = ...;
  }
}

// store result
for (int i = 0; i < 32 / 8; i++)
  stg_v8_f32(M_ptr + lane_id * 32 + i * 8, x + i * 8);
```

I want to highlight certain design decisions here:
- We definitely want to use vectorized 256-bit loads on Blackwell. This limits what thread layout (i.e. which elements of a matrix a thread holds) we can use. Since the input data is in row-major, a thread needs to hold a row tile spanning over multiple columns. For a 32x32 matrix and a single warp design we are currently exploring, letting each thread holds an entire row is a natural choice.
- It's important to have `#pragma unroll` directives. Normally I don't find it very useful since nvcc is already quite aggressive at loop unrolling. However, perhaps because the loop count is quite large, nvcc decided not to unroll it.
- Once the loops are fully unrolled, loop iterators are "constants". Hence, using `col` or `trail` to index register array `x[]` will not cause register spilling to local memory (for dynamic indexing) as the register can be addressed directly.

I didn't keep this implementation in the end because it was not faster than my newer designs designed for larger shapes. Regardless, I think it's worth mentioning this as part of the journey.

## QR176: Single-CTA, cooperative warps

For larger problem shapes, 1 warp definitely can't hold the entire matrix in register memory. Hence, we need to design a multi-warp strategy.

As mentioned earlier, 256-bit load is something we definitely want to do. Thus, each warp holds $[N, 8]$ tile of the problem shape, where each lane holds $[\mathrm{ceil}(N/32), 8]$ elements. There will be some rounding effect as N might not divide by 32 (e.g. 176 % 32 = 16). It's a bit annoying to handle but not too bad. The whole problem shape is now partitioned into $[N, 8]$ work tiles.

{{< figure src="panel8.svg" alt="Thread layout" caption="Thread layout: each warp holds $[N,8]$ tile, where each lane loads 8 consecutive FP32 values." >}}

Let's see if we have enough registers to hold the full 176x176 matrix in one CTA's register memory. 176x176 = 30,976. But recall that we need to round up the column size to the next multiple of 32, hence the number of registers is actually 192x176 = 33,792. This is well below the 64k register limit per CTA. Morever, since batch size is 40 for QR176, occupancy is not a concern i.e. we can use 1 SM per matrix and fully utilize an SM's resources for the CTA. Hence, for QR176, we use 176/8 = 22 warps per CTA, where each warp owns a $[N, 8]$ tile.

Up to this point, we still haven't used any shared memory for storing the full matrix at all. This is a deliberate choice: even though shared memory is fast compared to global memory, it still can't beat holding the data in register memory, which eliminates data roundtrip to shared memory, as long as resource limit allows.

Within an $[N, 8]$ tile owned by a warp, the logic is not much different from 1-warp design from QR32. Each warp only has 8 columns now, and each lane holds multiple elements per column ($\mathrm{ceil}(N/32)$ to be exact). The column vector reduction (vector norm in Householder and dot product in trailing column updates) now requires a small within-thread reduction before the within-warp reduction. Again, no cross-warp communication up to this point yet.

```cpp
constexpr ROW_ITEMS = cdiv(176, 32);
float x[ROW_ITEMS][8];

// load data [N, 8]
for (int i = 0; i < ROW_ITEMS; i++) {
  int row = i * 32 + lane_id;
  int col = warp_id * 8;
  if (row < 176)
    ldg_v8_f32(x[i], A_ptr + row * 176 + col);
  else
    // fill x[i] with zeros
}

// main loop over 8 columns
#pragma unroll
for (int i = 0; i < 8; i++) {
  const int col = warp_id * 8 + i;

  // compute Householder
  float v[ROW_ITEMS];
  ...

  // trailing columns update
  #pragma unroll
  for (int trail = i + 1; trail < 8; trail++) {
    ...
  }
}
```

For the first warp, the above code is sufficient. However, for other warps handling subsequent work tiles, they also need to update their owned columns using reflectors from earlier warps. This raises the need for cross-warp communication.

```cpp
constexpr ROW_ITEMS = cdiv(176, 32);
float x[ROW_ITEMS][8];

// load data
...

// NEW: update columns with previous reflectors
for (int col = 0; col = warp_id * 8; col++) {
  ...
}

// compute reflectors for 8 columns
#pragma unroll
for (int i = 0; i < 8; i++) {
  ...
}
```

The above design also establishes an interesting producer-consumer pattern: every warp takes turn to become a producer (computes and publishes reflectors), while the remaining ones are consumers (updates their owned columns). Once a warp has finished its job (i.e. update the columns and compute the reflectors), it can exit early.

{{< figure src="producer_consumer.svg" alt="Producer-Consumer pattern" caption="Producer-Consumer pattern: each warp takes turn to be the producer." >}}

A natural choice for cross-warp communication is via shared memory. Recall that for QR176, we need 176x176x4 = 123,904 bytes (with padding) to hold the whole matrix. This is the same size for holding all of the reflectors (we purposely don't do compact storage to avoid inefficient memory access and complicated, non-uniform logic that may affect codegen), and it's far below Blackwell's shared memory limit of ~227kb. Again, occupancy is not a concern (batch size < number of SMs), so we are free to use all of the available shared memory.

Fitting all of the reflectors in shared memory means that we don't need shared buffer reuse logic. Imagine if we can only hold 16 reflectors in shared memory. For the 17th reflector, its producer must wait for all consumers of the 1st reflector to finish before it can override the buffer slot, adding extra synchronization latency. But if we can hold all of the reflectors at once, producer warps literally "fire and forget: they can publish their reflectors without any delay, signal, then continue their work (self-update the columns and compute next reflectors).

```cpp
constexpr ROW_ITEMS = cdiv(176, 32);
float x[ROW_ITEMS][8];

// NEW: shared memory holding reflectors
extern __shared__ float storage[];
float* reflectors = storage;  // [ROWS, COLS]
float* taus = reflectors + ROWS * COLS;  // [COLS]

// load data to x[ROW_ITEMS][8]
...

// consumer phase: update columns with previous reflectors
// panel is dynamic loop, i is fully unrolled
for (int panel = 0; panel = warp_id; panel++) {
  for (int i = 0; i < 8; i++) {
    const int col = panel * 8 + i;
    __syncthreads();  // wait for reflectors to arrive

    // load reflector from smem
    float v[ROW_ITEMS];
    for (int item = 0; item < ROW_ITEMS; ++item) {
      const int row = item * 32 + lane;
      v[item] = row < ROWS ? reflectors[col * ROWS + row] : 0.0f;
    }

    // update the 8 owned columns
    for (int j = 0; j < 8; j++) {
      ...
    }
  }
}

// producer phase: compute reflectors
#pragma unroll
for (int i = 0; i < 8; i++) {
  const int col = warp_id * 8 + i;

  // compute Householder
  float v[ROW_ITEMS];
  ...

  // NEW: publish to other warps via shared memory
  for (int item = 0; item < ROW_ITEMS; ++item) {
    const int row = item * 32 + lane;
    if (row < ROWS) reflectors[col * ROWS + row] = v[item];
  }
  __syncthreads();

  // trailing columns update
  ...
}
```

Notice we are using `__syncthreads()` for memory synchronization here. You may find it strange that warps don't arrive on the same `__syncthreads()` line of code e.g. when warp1 waits for warp0's reflectors, warp1 enters `__syncthreads()` of the consumer block, while warp0 uses `__syncthreads()` in the producer block. This is perfectly fine and correct. `__syncthreads()` compiles to [`bar.sync 0` in PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar), or sometimes known as [`NamedBarrier` in CUTLASS](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/arch/barrier.h#L287), which is one of the 16 hardware barriers in NVIDIA GPUs. The act of arriving (and waiting) on a particular barrier doesn't concern which part of the code each warp is at.

- Fun annecdote. At the time of the competition, Codex 5.5 kept telling me this was invalid regardless of my reassurance. I think recent models/agents still think this is invalid too.

Overall everything looks good up to this point, but I think we can do better. One limitation of `__syncthreads()`/`bar.sync` is that it is both **arrive** and **wait**. For producer warps, it means they have to wait for all consumer warps to arrive before it can continue execution, even though it doesn't need to. Ideally we want something more compact: producers only need to **arrive**, and consumers only need to **wait**. We already have a solution to this: `mbarrier` commonly used in TMA and tcgen05 code for uni-directional memory synchronization.

- `bar.sync` also has a `bar.sync.arrive` variant. While writing this, I can't remember if I have tried it or encounter any problems with it.

```cpp
// shared memory: reflectors and mbar
extern __shared__ float storage[];
float* reflectors = storage;  // [ROWS, COLS]
float* taus = reflectors + ROWS * COLS;  // [COLS]
const int mbars = __cvta_generic_to_shared(taus + COLS);

// NEW: initialize mbar
// one mbar for each reflector
if (warp == 0 && elect_sync()) {
  for (int i = 0; i < COLS; ++i) {
    mbar_init(mbars + i * 8, 32);
  }
}
__syncthreads();

// consumer phase: update columns with previous reflectors
for (int panel = 0; panel = warp_id; panel++) {
  for (int i = 0; i < 8; i++) {
    const int col = panel * 8 + i;
    mbar_wait(mbars + col * 8, 0);  // NEW: mbarrier.try_wait loop

    // update the 8 owned columns
    ...
  }
}

// producer phase: compute reflectors
#pragma unroll
for (int i = 0; i < 8; i++) {
  const int col = warp_id * 8 + i;

  // compute and publish reflectors
  ...
  mbar_arrive(mbars + col * 8);  // NEW: replace __syncthreads()

  // trailing columns update
  ...
}
```

This design also performs well for QR32, and I didn't notice any regression compared to the previous register-resident kernel. Hence, in the final submission, I removed the register-resident kernel and used this kernel for QR32 as well.

**Comparison with standard Warp specialization** In typical Warp specialized GEMM kernels, each warp assumes a single role throughout its lifetime. Our design above is also warp specialization, but the role is not fixed: each warp takes turn acting as a producer. I also experimented with static warp specialization: 1 warp is responsible to compute reflectors, while N other warps update the trailing columns. This was slower than the final design we have here because of the extra data communications: when the producer warp has to compute reflector for column $i$, it needs to wait for the consumer warps to update that column.

Our northstar so far has always been "avoid communication as much as possible", and it has been working out quite well.

## QR352, QR512, and QR1024: compact WY transform

For the next shapes, we can't hold the full matrix in a threadblock anymore, hence requiring breaking down the matrices into multiple **panels**. This is also an apt time to introduce **Compact WY transform**: we can batch multiple reflectors together and apply them to the trailing columns using matmul operations. Again, this is not my forte and in fact Codex helped me with the high-level linear algebra operations, so I won't delve too much into it.

```python
A  # shape: [N, N]
panel    = A[:, :M]
trailing = A[:, M:]

# compute QR on the first M columns
# V is reflectors, H is panel's final results
V, tau, H = panel_QR(panel)

# compact WY transform
trailing = Q^T @ trailing

# where
Q = I - V @ T @ V^T
T = inv(diag(1/tau) + strictUpper(V^T @ V))

# if trailing is small enough, we can do the same
# QR routine on trailing matrix. otherwise, we need
# to perform another round of panel QR + compact WY
# transform.
```

To make the expression more natural, we can rewrite it as

```python
trailing = trailing - V @ T^T @ V^T @ trailing
T^T = inv(diag(1/tau) + strictLower(V^T @ V))
```

Everything can be implemented as ordinary PyTorch ops, they are basically a few matmuls and a single matrix inversion. I had a mini Triton kernel to compute `diag(1/tau) + strictLower(gram)`, where `gram = V^T @ V`, in a single kernel but I don't think it matters much anyway.

The `panel_QR` routine is the same as the previous QR kernel, with the only exception that the input matrices are not squares (Well, technically QR decomposition is not limited to square matrices). We also need to output the standalone reflectors for compact WY transform. Looking back at our previous kernel, we can see that this is basically free: we are already storing the reflectors in shared memory to communicate from producer to consumers; hence, we only need to issue TMA store to copy them to global memory, which has very low overhead.

```cpp
// consumer phase
for (int panel = 0; panel = warp_id; panel++) {
  ...
}

// producer phase
#pragma unroll
for (int i = 0; i < 8; i++) {
  ...
}

// emit V when it's not the last square QR
if constexpr (ROWS > COLS) {
  v_fp32 += batch * ROWS * COLS;

  __syncwarp();
  asm volatile("fence.proxy.async.shared::cta;");
  constexpr int PANEL_SIZE = 8 * ROWS;
  if (elect_sync()) {
    const int sV_fp32 = __cvta_generic_to_shared(reflectors) + warp * PANEL_SIZE * 4;
    tma_s2g(v_fp32 + warp * PANEL_SIZE, sV_fp32, PANEL_SIZE * 4);
  }
}
```

For the larger QR1024 shape, we have to replace 256-bit global loads (8 FP32 elements) with 128-bit loads (4 FP32 elements) to reduce register pressure i.e. each warp is now responsible for 4-element-wide panel, instead of 8.

### Panel schedule



### Low precision matmul

With the compact WY transform framework in place, we need to make it run faster. The most obvious area to attack is **low precision matmul**: there are 4 GEMMs involved per compact WY transform, and B200 has very bad FP32 TFLOPS compared to FP16/BF16 (70 FP32 TFLOPS vs ~1000 TF32 TFLOPS vs ~2000 FP16/BF16 TFLOPS). Using TF32 was easy, it was a PyTorch flag away, though that was a bit annoying to do fine-grained precision policy for each matmul (they have different sensitivity to the final results, especially tricky for QR512 test cases!).

### Triangular matrix inversion

Matrix inversion has always been the annoyingly slow operator. From my prior knowledge writing [GDN](https://arxiv.org/abs/2412.06464) kernels, I attempted using fully matmul-based matrix inversion methods, such as [Newton-Schulz iterations](https://www.emergentmind.com/topics/newton-schulz-iterations) and [Neumann series](https://en.wikipedia.org/wiki/Neumann_series), but they didn't provide sufficient accuracy and stability for the QR problem. Hence, I was uncertain if I could write a better matrix inversion than what was provided in PyTorch/CuBLAS/CuSOLVER.

However, what we can exploit is the **triangular structure** of the matrix that we are taking the inverse of. For unknown reasons, there are no standard triangular matrix inverse functions (technically there is [`cusolverDnXtrtri`](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxtrtri) but it only supports a single matrix, no batch API), only **triangular solve** i.e. compute $Y = X^{-1} A$. We can do triangular solve against an identity matrix for example, but again it won't be optimal.

The approach to triangular matrix inverse is pretty simple: we partition the matrix into 2x2 smaller tiles, and compute the inverse on 3 of them using [forward substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_substitution) (1 tile is fully empty because of the triangular structure). The diagonal tiles' inverse is final, but off-diagonal tile requires an additional pass using block forward substitution (the same as foward substitution formula, but replace scalar multiplication with matrix multiplication).

TODO: diagram

It was some time ago so I couldn't remember all the details but in the end I only provided 96x96 and 128x128 inverse. One possible reason is that they provide the largest possible panel size under my panel QR design (352x128x4 = 180,224 and 512x96x4 = 196,608 < 228 kB smem limit), and they factor into nice powers of 2 (96 = 64 + 32).

I found that doing repeated, hierarchical 2x2 block forward substitution is faster than 3x3 or 4x4 block forward substitution, even though the latter require less FLOPs. I think it's because 2x2 block requires significantly less memory synchronization, though it can also be skill issue in my part.

TODO: diagram

For the largest off-diagonal inverse, 64x64 off-diagonal tile in 128x128 inverse and 64x32 in 96x96 inverse, I use PyTorch for the matmul since I don't think I can write a better one myself.

## QR1024: 2-CTA, threadblock cluster communication

For QR1024, we are still under-utilizing the GPUs since B200 has 148 SMs but the problem shape only has batch=60. A natural idea is to use more than 1 CTA to process the panel QR, but that would introduce potentially expensive cross-CTA communication. Luckily, I recall there are nifty tools available to threadblock cluster:
- [`st.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async): non-blocking store operation from register to shared memory of a peer CTA.
- [S2S TMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk): TMA from local shared memory to peer CTA's shared memory.

They both report the completion via mbarrier, which is convenient. We can naturally extend our current design to 2-CTA with threadblock cluster. CTA0 additionally sends its reflectors to CTA1, while CTA1 now has an extra stage of receiving CTA0's reflectors, instead of just its own.

```cpp
extern __shared__ float storage[];
float* reflectors = storage;
constexpr int LOCAL_COLS = COLS / 2;
float* taus = reflectors + ROWS * LOCAL_COLS;
const int reflector_addr = __cvta_generic_to_shared(reflectors);
const int tau_addr = reflector_addr + ROWS * LOCAL_COLS * 4;
const int mbars = tau_addr + COLS * 4;

// (NEW) precompute CTA1's smem address
// CTA1 doesn't need this, so we compute it unconditionally.
const int reflector_addr1 = reflector_addr | 0x01000000;
const int tau_addr1 = tau_addr | 0x01000000;

// replace __syncthreads() with cluster barrier to make sure
// one CTA not race before peer CTA init its mbarriers.
if (warp == 0 && elect_sync()) {
  for (int i = 0; i < COLS; ++i) mbar_init(mbars + i * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;");
}
asm volatile("barrier.cluster.arrive.relaxed.aligned;");
asm volatile("barrier.cluster.wait.acquire.aligned;");

// (NEW) consumer: update using reflectors from CTA0
// this is a 0-iter loop on CTA0
for (int panel = 0; panel < rank * NUM_WARPS; panel++) {
  for (int i = 0; i < 8; i++) {
    // 1 warp waits for the arrival of reflectors
    const int col = panel * 8 + i;
    if (warp == 0)
      mbar_wait(mbars + col * 8, 0);
    __syncthreads();

    // update columns
  }
}

// make sure all warps finish using CTA0's reflectors.
// we will reuse smem for the next stage.
__syncthreads();

// consumer: update using local reflectors (same as before)
for (int panel = rank * NUM_WARPS; panel < rank * NUM_WARPS + warp; panel++) {
  // use local coordinates for smem indexing
  const int local_panel = panel - rank * NUM_WARPS;
  ...
}

// producer: compute reflectors
#pragma unroll
for (int i = 0; i < 8; i++) {
  const int col = (rank * NUM_WARPS + warp) * 8 + i;
  const int local_col = warp * 8 + i;

  // compute reflectors and store to smem as usual
  ...

  __syncwarp();
  asm volatile("fence.proxy.async.shared::cta;");
  if (elect_sync()) {
    // signal to other warps in current CTA
    mbar_arrive(mbars + col * 8);

    // (NEW) signal to peer CTA
    if (rank == 0) {
      // magic number to get peer CTA's smem address
      // send reflector with TMA, send tau with st.async
      const int remote_mbar = (mbars + col * 8) | 0x01000000;
      tma_s2s(reflector_addr1 + col * ROWS * 4,
              reflector_addr + local_col * ROWS * 4,
              ROWS * 4, remote_mbar);
      st_async_f32(tau_addr1 + col * 4, tau_value, remote_mbar);
    }
  }

  // update trailing columns
  ...
}
```

The reflector itself is already stored in shared memory for consumer warps within the same CTA, hence we only need to issue the shared-to-shared TMA. For tau, we use `st.async`, which should be faster than normal store (I didn't measure but I hope it's faster in the sense that it's non-blocking so the producer warp can continue its execution).

To get CTA1's shared memory address, we simply set a particular bit, instead of [clearing it](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cute/arch/copy_sm100_tma.hpp#L63) as in the original tcgen05 tutorial.

## QR2048 and QR4096: Multi-CTA, grid-wide coordination

For QR2048 and QR4096, there are only 8 and 2 matrices respectively per kernel invocation, prompting us to use even more CTAs per QR matrix. Threadblock cluster supports more than 2-CTAs, but they are not scalable: The data transfer (`st.async` and S2S TMA) is push-based, meaning that we would need to push to every consuming CTAs, resulting in extra memory traffic. Doing pull-based data transfer would not improve the situation: not only we still have to do signalling (i.e. the consumer CTAs need to know when data is ready), memory traffic is neither reduced.

Instead of using threadblock cluster, we can go back to the more traditional inter-CTA synchronization via global memory (or L2 to be exact). Producer CTAs publish reflectors to global memory, and consumer CTAs can read from global memory to update their owned columns. Synchronization is achieved with GPU-scope release-acquire semantics.

Another issue with large QR matrix shapes is that 1 warp can't hold the full `[rows, 8]` panel in register memory (2048 x 8 / 32 = 512 registers/thread). Hence, we have no choice but to distribute a single column across multiple warps. This means that many subroutines that require a column reduction, such as computing the reflectors and doing column updates, necessiate cross-warp reduction via shared memory.

To keep the code simple, we opt for a straight-forward thread layout: each CTA holds `[rows, 8]` panel, where each thread still loads 8 elements at a time for efficient 256-bit loads. The panel is sharded along the row dimension, where each warp holds `[rows / NUM_WARPS, 8]` tile.

TODO: diagram compare before and after

```cpp
// (NEW) consumer: update reflectors produced by earlier CTAs
for (int k = 0; k)
```

We put some efforts to make sure the memory synchronization has low overheads. The flag is polled with `.relaxed.gpu.L1::no_allocate`. `.relaxed` helps to observe the flag cheaply, while acquire semantics is achieved with the later `fence.acquire.gpu`. `.L1::no_allocate` helps with cache behavior - caching the `false` flag value has no benefit, we are trying to observe the flip as soon as possible! Also only one thread polls the flag to avoid hammering the memory subsystem - ordering for other threads in a CTA is achieved using another `__syncthreads()`.

## Final remarks

I hope what we have gone through in the blogpost feels like a natural progression to the problem.
- We start with a register-resident kernel because we want to avoid communication as much as possible. A single warp handles a matrix.
- Then, we come up with a producer-consumer kernel where warps in a CTA cooperate with each other.
- Next, to better utilize GPU resources, we use 2 CTAs to handle one matrix, which follows the exact same producer-consumer pattern, and utilizing TMA and `st.async` for efficient cross-CTA communication.
- Finally, for even smaller batch sizes, multiple CTAs can coordinate via global memory.

I collected the top 3 submissions and re-ran them on Modal over the problem shapes (timing reported in us).

batch | n | case | 10billiontokens | gau.nernst (me) | dhu.randhar
--|--|--|--|--|--
20 | 32 | | 14.8 | 12.9 | **11.1**
40 | 176 | | 184.2 | **95.7** | 222.8
40 | 352 | | 573.5 | **313** | 546.8
640 | 512 | | **2599** | 3388 | 3153
60 | 1024 | | 2367 | **2094** | 2272
8 | 2048 | | **2088** | 3513 | 2596
2 | 4096 | | **4031** | 8170 | 4905
640 | 512 | mixed | 3537 | **3378** | 4205
60 | 1024 | mixed | 3205 | **2082** | 3064
640 | 512 | rankdef | **2201** | 3370 | 2486
640 | 512 | clustered | **1689** | 3385 | 1871
60 | 1024 | nearrank | **2026** | 2083 | 2482
 | | | geomean | 1175 | 1239 | 1274

Compared to the top submission, I'm quite behind in many cases. However, if you pay close attention, for **mixed cases** (matrices are generated using different methods, so participants can't exploit the same structure across the whole batch), my solution's runtime does not change at all, while others slow down significantly. This is because I made no attempts to detect and exploit structure in the input data at all, though some may argue that tuning the matmul precision in compact WY transform was already "exploiting" the stricter tests of QR512.

For QR176 and QR352, I'm nearly 2 times faster than the best submission, validating the sound design in my kernel. This also made me confident that even though I did not produce the fastest QR kernel, I was sure my panel QR kernel was fastest on the world for B200 at the time of the competition (simply because no one else would write a QR kernel for B200 outside of this competition).

The relatively bad timing of my QR512, after accounting for not using shortcuts, also reveals a weakness that I miss for this shape: occupancy. Throughout this journey, I have conveniently used up all available SM resources, which is fair for SM-limited shapes, but might not be optimal when batch size is larger than number of SMs. It might be useful to be more conservative with resources in order to increase occupancy (number of active CTAs per SM) for QR512.
