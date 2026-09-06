+++
date = '2026-09-06T20:00:00+08:00'
title = 'World's fastest (panel) QR factorization on B200'
url = 'b200-qr'
+++
GPU MODE recently hosted a series of leaderboards for [linear algebra kernels](https://www.gpumode.com/news/linear-algebra-kernels-age-of-research). I participated in the first one, QR decomposition, and claimed the 2nd place. I think it's a breath of fresh air, a departure from the usual LLM-oriented kernels most people program for nowadays.

As usual, my submission is open-sourced at [gau-nernst/gpu-mode-kernels](https://github.com/gau-nernst/gpu-mode-kernels/tree/fb291890aa1fe35648799bf7b2731056240bcabf/linalg/qr_v2).

{{< toc >}}

## Introduction to QR decomposition

I don't want to touch too much on this, as I'm no expert on the topic (in fact, I only learned about QR decomposition, mostly from Codex, during the competition). There are excellent resources online explaining this, such as [the one from Michael](https://ml-mike.com/writing/qr_v2/). Instead, I will focus on my thought process tackling the crux of the challenge: chained dependency.

```python
# input: A[M, N]

# we can't parallelize this
for n in range(N-1):
    reflector = householder(A[n:, n])

    # we can parallelize this
    for tail in range(n+1, N):
        A[n:, tail] = reflect(A[N:, tail], reflector)
```

As you can see from the rough outline of the QR algorithm above, computing the Householder for column n requires the reflectors from all previous columns, making it impossible to parallelize the Householder computation. However, the trailing columns update can be parallelized easily as they are independent. One simple strategy is to let 1 warp computes the Householder (computing Householder requires a reduction of that column, hence we want to use a single warp for this purpose to avoid cross-warp communication), then multiple warps can update the trailing columns at the same time. This is the crux of my approach to (panel) QR decomposition.

Case | Batch | n
---|---|---
0 | 20 | 32
1 | 40 | 176
2 | 40 | 352
3 | 640 | 512
4 | 60 | 1024
5 | 8 | 2048
6 | 2 | 4096
7 | 640 | 512
8 | 60 | 1024
9 | 640 | 512
10 | 640 | 512
11 | 60 | 1024

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

I want to highlight some decisions and important points here:
- We definitely want to use vectorized 256-bit loads on Blackwell. This limits what thread layout (i.e. which elements of a matrix a thread holds) we can use. Since the input data is in row-major, a thread needs to hold a row tile spanning over multiple columns. For a 32x32 matrix and a single warp design we are currently exploring, letting each thread holds an entire row is a natural choice.
- It's important that we have `#pragma unroll` directives. Normally I don't find it very useful since nvcc is already quite aggressive at loop unrolling. However, perhaps because the loop count is quite large, nvcc decided not to unroll it.
- Once the loops are fully unrolled, loop iterators are "constants". Hence, using `col` or `trail` to index register array `x[]` will not cause register spilling to local memory (for dynamic indexing), but each register is addressed directly.

I didn't keep this implementation in the end because it was not faster than my newer solutions designed for larger shapes. Regardless, I think it's worth mentioning this as part of the journey.

## QR176: Single-CTA, cooperative warps

For larger problem shapes, 1 warp definitely can't hold the entire matrix in register memory. Hence, we need to design a multi-warp strategy.

As mentioned earlier, 256-bit load is something we definitely want to do. Thus, each warp holds [N, 8] tile of the problem shape, where each lane holds [ceil(N/32), 8] elements. There will be some rounding effect as N might not divide by 32 (e.g. 176 % 32 = 16). It's a bit annoying to handle but not too bad. The whole problem shape is now partitioned into [N, 8] work tiles.

<some diagram here>

Let's see if we have enough registers to hold the full 176x176 matrix in one CTA's register memory. 176x176 = 30,976. But recall that we need to round up the column size to the next multiple of 32, hence the number of registers is actually 192x176 = 33,792. This is well below the 64k register limit per CTA. Morever, since batch size is 40 for QR176, occupancy is not a concern i.e. we can use 1 SM per matrix and fully utilize an SM's resources for the CTA. Hence, for QR176, we use 176/8 = 22 warps per CTA, where each warp owns a [N, 8] tile.

Up to this point, we still haven't used any shared memory for storing the full matrix at all. This is a deliberate choice: even though shared memory is fast compared to global memory, it still can't beat holding the data in register memory, which eliminates data roundtrip to and from shared memory, as long as resource limit allows.

Within an [N, 8] tile owned by a warp, the logic is not much different from 1-warp design from QR32. Each warp only has 8 columns now, and each lane holds multiple elements per column (ceil(N/32) to be exact). The column vector reduction (vector norm in Householder and dot product in trailing column updates) now requires a small within-thread reduction before the within-warp reduction. Again, no cross-warp communication up to this point yet.

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

For the first warp, the above code is sufficient. However, for other warps handling subsequent work tiles, they also need to update their owned columns using reflectors from earlier warps. This establishes a need for cross-warp communication.

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

The above design also establishes an interesting producer-consumer pattern: every warp takes turn to become a producer (computes and publishes reflectors), while they are all consumers (updates the owned columns with reflectors). Once a warp has finished its job (i.e. update the columns and compute the reflectors), it can exit early.

<diagram>

A natural choice for cross-warp communication is via shared memory. Recall that for QR176, we need 176x176x4 = 123,904 bytes (with padding) to hold the whole matrix. This is the same size for holding all of the reflectors (we purposely don't do compact storage to avoid inefficient memory access and complicated, non-uniform logic that may affect codegen), and it's far below Blackwell's shared memory limit of ~227kb. Again, occupancy is not a concern (batch size < number of SMs), so we are free to use all of the available shared memory.

Fitting all of the reflectors in shared memory means that we don't need some kind of buffer slots reuse logic. Imagine we can only hold 16 reflectors in shared memory. For the 17th reflector, its producer must wait for all consumers of the 1st reflector to finish before it can override the buffer slot, adding extra synchronization latency. But if we can hold all of the reflectors at once, producer warps literally "fire and forget: they can publish their reflectors without any delay, signal, then continue their work (self-update the columns and compute next reflectors).

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

### Side notes

Warp specialization

## QR352, QR512, and QR1024: compact WY update

For the next shape, we can't hold the whole matrix in a threadblock anymore

TODO: schedule design

NOTE: for QR512, occupancy=1 design might not be optimal

## QR1024: 2-CTA, threadblock cluster communication

## QR2048 and QR4096: Multi-CTA, grid-wide coordination

## Final remarks
