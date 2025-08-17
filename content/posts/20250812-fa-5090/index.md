+++
date = '2025-08-14T18:38:59+08:00'
draft = true
title = 'Writing Speed-of-Light Flash Attention for 5090 in CUDA C++'
url = 'fa-5090'
+++
In this post, I will walkthrough how I learned to implement Flash Attention for 5090 in CUDA C++. The main objective is to learn writing attention in CUDA C++, since many features are not available in [Triton](https://triton-lang.org/main/index.html), such as MXFP8 / NVFP4 MMA for sm120. I also feel this is a natural next step after learning about matmul kernels. Lastly, there are [many](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) [excellent](https://www.spatters.ca/mma-matmul) [blogposts](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) on writing fast matmul kernels, but there is none for attention. So I want to take this chance to write up something nicely.

Readers are highly recommended to be familiar with CUDA C++ and how to use Tensor cores on NVIDIA GPUs. Of course you can still read along and clarify with your favourite LLMs along the way. Or you can check out GPU-MODE series ([slides](https://github.com/gpu-mode/lectures), [YouTube](https://www.youtube.com/@GPUMODE)) for basic CUDA C++ knowledge, as well as the excellent matmul blogposts mentioned above, to quickly get up to speed.

You can find the full implementation discussed in this post here: https://github.com/gau-nernst/learn-cuda/tree/7e2d6951c3fb2b0211dca756fb2144126a352013/07_attention. For `bs=1, num_heads=8, len_query=4096, len_kv = 8192`, 5090 @ 400W, compile with CUDA 12.9, I obtained the following benchmark results (theoretical limit of 5090 is 209.5 TFLOPS for BF16)

Kernel                         | TFLOPS | % of SOL
-------------------------------|--------|---------
`F.sdpa()` (Flash Attention)   | 186.73 | 89.13%
`F.sdpa()` (CuDNN)             | 203.61 | 97.19%
`flash-attn`                   | 190.58 | 90.97%
v1 (basic)                     | 142.87 | 68.20%
v2 (shared memory swizzling)   | 181.11 | 86.45%
v3 (2-stage pipelining)        | 189.84 | 90.62%
v4 (`ldmatrix.x4` for K and V) | 194.33 | 92.76%
v5 (better pipelining)         | 197.74 | 94.39%

Do note that although I only use Ampere features in these implementations (sm120 supports `cp.async.bulk` i.e. TMA, but I don't use it here), my implementations might not run performantly on earlier generations of GPUs. Due to improvements in newer hardware, you might need to use more tricks to reach Speed-of-Light on older GPUs e.g. pipeline shared memory to register memory data movements.

{{< toc >}}

## Flash Attention algorithm

Let's start with the reference implementation of attention.

```python
from torch import Tensor

def sdpa(q: Tensor, k: Tensor, v: Tensor):
    # q: [B, Lq, DIM]
    # k: [B, Lk, DIM]
    # v: [B, Lk, DIM]
    D = q.shape[-1]
    scale = D ** -0.5
    attn = (q @ k.transpose(-1, -2)) * scale  # [B, Lq, Lk]
    attn = attn.softmax(dim=-1)
    out = attn @ v  # [B, Lq, DIM]
    return out
```

Technically, if the inputs are BF16, some computations should remain in FP32, especially softmax. However, for brevity, we omit them.

We are implementing the algorithm outlined in the [Flash Attention 2 paper](https://arxiv.org/abs/2307.08691). Each threadblock is responsible for a chunk of Q, and we will iterate along the sequence length of KV. A Python-like outline of the algorithm looks like below (P and V follow Flash Attention notation).

```python
scale = DIM ** -0.5
for b_idx in range(B):
    for tile_Q_idx in range(Lq // BLOCK_Q):
        ### start of each threadblock's kernel
        tile_O = torch.zeros(BLOCK_Q, DIM)
        tile_Q = load_Q(b_idx, tile_Q_idx)  # [BLOCK_Q, DIM]

        for tile_KV_idx in range(Lk // BLOCK_KV):
            # first MMA: S = Q @ K.T
            # (BLOCK_Q, DIM) x (BLOCK_KV, DIM).T -> (BLOCK_Q, BLOCK_KV)
            tile_Q                               # (BLOCK_Q, DIM)
            tile_K = load_K(b_idx, tile_KV_idx)  # (BLOCK_KV, DIM)
            tile_S = tile_Q @ tile_K.T           # (BLOCK_Q, BLOCK_KV)
            tile_S = tile_S * scale

            # online softmax and rescale tile_O
            ...

            # second MMA: O = P @ V
            # (BLOCK_Q, BLOCK_KV) x (BLOCK_KV, DIM) -> (BLOCK_Q, DIM)
            tile_P                               # (BLOCK_Q, BLOCK_KV)
            tile_V = load_V(b_idx, tile_KV_idx)  # (BLOCK_KV, DIM)
            tile_O += tile_P @ tile_V            # (BLOCK_Q, DIM)

        # normalize output and write results
        store_O(b_idx, tile_Q_idx)
        ### end of each threadblock's kernel
```

It's implied `DIM` is small, so that we can hold `tile_Q` in register memory throughout the duration of the kernel. This is the reason pretty much all models nowadays use `head_dim=128`. There are exceptions of course, like [MLA](https://arxiv.org/abs/2405.04434), which uses `head_dim=576` for Q and K, and `head_dim=512` for V. I should study [FlashMLA](https://github.com/deepseek-ai/FlashMLA) some day.

Online softmax is quite tricky to explain, so let's delay the explanation of that part. At the high level, we just need to know that online softmax will transform `tile_S` to `tile_P`, and also rescale `tile_O`.

## Version 1 - Basic implementation

We will follow the typical MMA flow
- Load a 2D tile of data from global memory to shared memory using [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async). This requires Ampere (sm80 and above).
- Load data from shared memory to register memory using [`ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix).
- Call [`mma.m16n8k16`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float) for BF16 matrix multiplication (and accumulate).

I want to focus on implementing the algorithm correctly first, hence I leave out more complicated tricks like shared memory swizzling and pipelining. This reduces the surface area for mistakes, and we will revisit them later. I will go through these briefly since they are not the focus of this blogpost. Readers are welcome to refer to matmuls with Tensor cores articles for more detailed explanations.

### Global to Shared memory data transfer

The following templated function does a 2D tile copy from global memory to shared memory.
- Shape of the 2D tile is specified via `HEIGHT` and `WIDTH`.
- `dst` is shared memory address, `src` is global memory address.
- Global memory `src` is row-major, so `src_stride` specifies how much to move to the next row.
- Shared memory `dst` is also row-major, and will be stored as a contiguous block -> `dst_stride = WIDTH`.

```cpp
#include <cuda_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}
```

{{< figure src="global_to_shared.svg" alt="Global to Shared data transfer" caption="2D tile copy from Global memory to Shared memory." >}}

We will use inline assembly to write `cp.async.cg.shared.global`. This PTX does 16-byte transfer, or 8 BF16 elements (`num_elems = 16 / sizeof(nv_bfloat16)`), for each CUDA thread. To ensure coalesced memory access, consecutive threads will be responsible for consecutive groups of 8xBF16.

{{< figure src="coalesced.svg" alt="Coalesced memory access" caption="Consecutive threads are responsible for consecutive groups of 8xBF16." align="center">}}

Note:
- The loop `for (int iter = 0; iter < num_iters; iter++)` is written this way so that the compiler (`nvcc`) can fully unroll the loop. `num_iters` is known at compile time (guaranteed by `constexpr`). If we mix `tid` in the loop, which is a "dynamic" variable to the compiler, the loop can't be unrolled, even when we know certain constraints about the variable i.e. `tid < TB_SIZE`.
- Data type of shared memory pointer `dst` is `uint32_t`. This is intentional. Pretty much all PTX instructions expect shared memory addresses to be in [shared state space](https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces). We can convert C++ pointers, which are generic addresses, to shared state space addresses with `static_cast<uint32_t>(__cvta_generic_to_shared(ptr))`. This is done outside of `global_to_shared()`.

### Shared memory to Register memory data transfer

When doing global->shared data transfer, we think in terms of threadblock tiles and individual CUDA threads. For shared->register data transfer, since this is to service the later MMA instruction, we think in terms of warp/MMA tiles and warps. Following Flash Attention 2 (section 3.3), we let each warp in a threadblock handle a portion of `tile_Q`, splitting along the Q sequence length dimension. This means that different warps will index into different chunks of `tile_Q`, but they all index to the same `tile_K` and `tile_V` chunks in the KV-sequence-length loop.

{{< figure src="fa_warp_partition.svg" alt="Flash Attention warp partition" caption="Warp partition in Flash Attention 2." align="center">}}

Since we are using `mma.m16n8k16` instruction, each MMA 16x8 output tile (`m16n8`) requires 16x16 A tile (`m16k16`) and 8x16 B tile (`n8k16`). `ldmatrix` can load one, two, or four 8x8 tile(s) of 16-bit elements. Hence,
- A tile `m16k16` requires four 8x8 tiles -> `ldmatrix.x4`
- B tile `n8k16` requires two 8x8 tiles -> `ldmatrix.x2`

Only Q acts as A in an MMA. Both K and V act as B in their MMAs, though K will require transposed `ldmatrix` for correct layout (all tensors use row-major layout in global and shared memory).

To use `ldmatrix`, each thread supplies the address of each row. Threads 0-7 select the 1st 8x8 tile, threads 8-15 select the 2nd 8x8 tile, and so on. The [layout of A](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float) in the official PTX documentation can look confusing. But it's easier (at least for me) to focus on the order of 8x8 tiles.

{{< figure src="ldmatrix.svg" alt="ldmatrix for MMA layout" caption="Order of `ldmatrix` tiles in `mma.m16n8k16`." align="center">}}

With the visualisation above, I hope the following snippet makes sense

```cpp
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

uint32_t Q_smem;
uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];

for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
    const int row = (warp_id * WARP_Q) + (mma_id_q * MMA_M) + (lane_id % 16);
    const int col = (mma_id_d * MMA_K) + (lane_id / 16 * 8);
    const uint32_t addr = Q_smem + (row * DIM + col) * sizeof(nv_bfloat16);
    ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
  }
```

- The two nested loops tile `[MMA_M, MMA_K]` (i.e. `[16, 16]`) over `[WARP_Q, DIM]` in shared memory.
- `(warp_id * WARP_Q)` selects the warp tile. We don't need this for K and V.
- `(mma_id_q * MMA_M)` in `row` and `(mma_id_d * MMA_K)` in `col` selects the MMA tile.
- `(lane_id % 16)` in `row` and `(lane_id / 16 * 8)` in `col` select the correct row address for each thread, following the required Multiplicand A layout (see the figure above).

`ldmatrix_x4()` is a small wrapper around `ldmatrix.sync.aligned.m8n8.x4.b16` PTX for convenience. You can refer to [common.h](https://github.com/gau-nernst/learn-cuda/blob/7e2d6951c3fb2b0211dca756fb2144126a352013/07_attention/common.h) for more details.

K and V can be loaded from shared to register memory similarly. One thing to note is about the row-major / column-major layout when using `ldmatrix`. Regardless of whether `.trans` modifier is used, each thread still provides the row address of each row in 8x8 tiles. `.trans` only changes the **register layout** of `ldmatrix` results.

TOOD: add a figure here.

### Draft version

We have the high-level tile-based design, and know how to load the data for MMA. Calling MMA is simple - just drop `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` PTX in our code. Our draft version looks like this.

```cpp
constexpr int BLOCK_Q = 128;
constexpr int BLOCK_KV = 64;
constexpr int DIM = 128;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * 32;

// mma.m16n8k16
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

__global__
void attention_v1_kernel(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  // basic setup
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  // increment Q, K, V, O based on blockIdx.x
  ...

  // set up shared memory
  // Q_smem is overlapped with (K_smem + V_smem), since we only use Q_smem once
  extern __shared__ uint8_t smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;
  const uint32_t V_smem = K_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);

  // FA2: shard BLOCK_Q among warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // set up register memory
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];       // act as A in MMA
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];     // act as B in MMA
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];  // act as A in MMA
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];     // act as B in MMA
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4];          // act as C/D in MMA

  // Q global->shared [BLOCK_Q, DIM]
  global_to_shared<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Q shared->register. select the correct warp tile
  // Q stays in registers throughout the kernel's lifetime
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id % 16);
      const int col = mma_id_d * MMA_K + (lane_id / 16 * 8);
      const uint32_t addr = Q_smem + (row * DIM + col) * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  __syncthreads();

  // main loop
  const int num_kv_iters = len_kv / BLOCK_KV;
  for (int kv_idx = 0; kv_idx < num_kv_iters; kv_idx++) {
    // accumulator for the 1st MMA. reset to zeros
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};  // act as C/D in MMA

    // load K global->shared->registers [BLOCK_KV, DIM]
    // similar to loading Q, except we use ldmatrix_x2()
    ...

    // 1st MMA: S = Q @ K.T
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                       K_rmem[mma_id_kv][mma_id_d],
                       S_rmem[mma_id_q][mma_id_kv]);

    // online softmax. we will touch on this later
    // also pack S_rmem to P_rmem for the 2nd MMA
    ...

    // load V global->shared->registers [BLOCK_KV, DIM]
    // similar to loading K, except we use ldmatrix_x2_trans()
    ...

    // 2nd MMA: O = P @ V
    // similar to the 1st MMA
    ...

    // increment pointer to the next KV block
    K += BLOCK_KV * DIM;
    V += BLOCK_KV * DIM;
  }

  // write output
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      float *regs = O_rmem[mma_id_q][mma_id_d];
      reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * DIM + col)[0] = __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * DIM + col)[0] = __float22bfloat162_rn({regs[2], regs[3]});
    }
}

// kernel launcher
void attention_v1(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  nv_bfloat16 *O,        // [bs, len_q, DIM]
  int bs,
  int len_q,
  int len_kv) {

  // 1 threadblock for each BLOCK_Q
  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);

  // Q overlap with K+V.
  const int smem_size = max(BLOCK_Q, BLOCK_KV * 2) * DIM * sizeof(nv_bfloat16);

  // use dynamic shared memory so we can allocate more than 48kb if needed.
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  attention_v1_kernel<<<num_blocks, TB_SIZE, smem_size>>>(Q, K, V, O, bs, len_q, len_kv);
  CUDA_CHECK(cudaGetLastError());
}
```

Now, let's tackle online softmax.

### Online softmax - theory

For the original explanation, you can refer to [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) and Flash Attention 2 paper.

We have the following mathematical definition of softmax. For each row with length $L_{kv}$

$$
p_l = \frac{\exp(s_l-m)}{\exp(s_0-m) + \exp(s_1-m) + \dots + \exp(s_{L_{kv}-1}-m)}
$$
$$
l\in[0,L_{kv})
$$
$$
m=\max(s_0,s_1,\dots,s_{L_{kv}-1})
$$

$-m$ is max subtraction to improve numerical stability ($\exp(\cdot)$ can easily explode if its input is large). Let's bring out the denominator normaliser and write the whole row as a vector.

$$
\vec P =
\begin{bmatrix}
p_0 \\\\
\vdots \\\\
p_{L_{kv}-1}
\end{bmatrix}
= \frac{1}{\sum_{l\in[0,L_{kv})}\exp(s_l-m)}
\begin{bmatrix}
\exp(s_0-m) \\\\
\vdots \\\\
\exp(s_{L_{kv}-1}-m)
\end{bmatrix}
$$

In our 2nd matmul `O += P @ V`, each row of P (softmax output) is dot-producted with the corresponding column of V.

$$
o=\vec P \cdot \vec V = \frac{1}{\sum_{l\in[0,L_{kv})}\exp(s_l-m)} \sum_{l\in[0,L_{kv})}\exp(s_l-m) \cdot v_l
$$

The extra dot-product is a blessing in disguise - we no longer need individual elements in a row for the final result. This enables Flash Attention to compute attention in one pass. To see it more clearly, let's consider the iterative process of adding a new element during online computation.

$$
o_{[0,L)} = \frac{1}{\sum_{l\in[0,L)}\exp(s_l-m_{[0,L)})} \sum_{l\in[0,L)}\exp(s_l-m_{[0,L)}) \cdot v_l
$$
$$
m_{[0,L)}=\max(s_0,s_1,\dots,s_{L-1})
$$

I'm abusing the notation here, but I hope I get the idea across. When we add a new element $s_{L+1}$

$$
o_{[0,L+1)} = \frac{1}{\sum_{l\in[0,L+1)}\exp(s_l-m_{[0,L+1)})} \sum_{l\in[0,L+1)}\exp(s_l-m_{[0,L+1)}) \cdot v_l
$$

Look at the normaliser (denominator)

$$
\sum_{l\in[0,L+1)}\exp(s_l-m_{[0,L+1)}) = \colorbox{red}{$\displaystyle\exp(m_{[0,L)}-m_{[0,L+1)})$}\colorbox{orange}{$\displaystyle\sum_{l\in[0,L)}\exp(s_l-m_{[0,L)})$} + \colorbox{lime}{$\displaystyle\exp(s_L-m_{[0,L+1)})$}
$$

The equation means that we only need to $\colorbox{red}{rescale}$ the $\colorbox{orange}{previous normaliser}$ before adding the $\colorbox{lime}{new term}$. The same logic can be applied for the dot product with V (unnormalised output). **This is the key idea of online softmax and Flash Attention**.

Define **attention state**

$$
\begin{bmatrix}
m \\\\
\tilde{o} \\\\
\mathrm{sumexp}
\end{bmatrix}
$$

where $m$ is max of elements seen so far, $\tilde{o}$ is **unnormalised** output, and $\mathrm{sumexp}$ is normaliser. We need $m$ to compute the rescaling factor as seen above.

You can convince yourself that updating the attention state is an **associative** operation - it does not matter which elements are used to update the attention state next.

$$
\begin{bmatrix}
m_1 \\\\
\tilde{o}_1 \\\\
\mathrm{sumexp}_1
\end{bmatrix}
\oplus \begin{bmatrix}
m_2 \\\\
\tilde{o}_2 \\\\
\mathrm{sumexp}_2
\end{bmatrix}
= \begin{bmatrix}
m_3 \\\\
\tilde{o}_3 \\\\
\mathrm{sumexp}_3
\end{bmatrix}
= \begin{bmatrix}
\max(m_1,m_2) \\\\
\exp(m_1-m_3)\tilde{o}_1+\exp(m_2-m_3)\tilde{o}_2 \\\\
\exp(m_1-m_3)\mathrm{sumexp}_1+\exp(m_2-m_3)\mathrm{sumexp}_2
\end{bmatrix}
$$

This associative property enables things like [Flash Decoding](https://pytorch.org/blog/flash-decoding/), a split-K version of attention.

### Online softmax - Implementation

We can now fill in the gap of online softmax in our high-level Python implementation.

```python
# attention state
m = torch.zeros(BLOCK_Q)
tile_O = torch.zeros(BLOCK_Q, DIM)
sumexp = torch.zeros(BLOCK_Q)

for _ in range(Lk // BLOCK_KV):
  # 1st MMA
  tile_S = tile_Q @ tile_K.T  # [BLOCK_Q, BLOCK_KV]
  tile_S = tile_S * scale

  # online softmax
  tile_max = tile_S.amax(dim=-1)  # [BLOCK_Q]
  new_m = torch.maximum(m, tile_max)
  tile_P = torch.exp(tile_S - new_m.unsqueeze(-1))

  # rescale
  scale = torch.exp(m - new_m)
  tile_O *= scale.unsqueeze(-1)
  sumexp *= scale
  sumexp += tile_P.sum(dim=-1)
  m = new_m  # save new max

  # 2nd MMA
  tile_O += tile_P @ tile_V  # [BLOCK_Q, DIM]

# apply normalisation
tile_O /= sumexp.unsqueeze(-1)
```

#### Row max

When translating this to CUDA C++, the most tricky part is to wrap our head around MMA layout. Let's start with `tile_S`.

{{< figure src="https://docs.nvidia.com/cuda/parallel-thread-execution/_images/mma-16816-C-f16.png" alt="MMA m16n8k16 output layout" caption="Thread and register layout of MMA m16n8k16 output. Source: [NVIDIA PTX doc](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)" >}}

Softmax scale applies the same scaling for all elements, so that is trivial. Next, we need to compute row max for the current tile. Remember that we allocate the registers for `tile_S` this way.

```cpp
float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4];
```

`4` means `c0,c1,c2,c3` in the figure above i.e. each thread holds 2 consecutive elements from 2 rows. To do reduction within a row (of an MMA output tile), we do reduction for 2 consecutive elements held by a thread, then reduction within a group of 4 threads i.e. `T0-T3`, `T4-T7`, and so on. However, the row reduction is actually within the whole `tile_S`, hence we also need to loop over `BLOCK_KV / MMA_N`. This can be combined with thread-level reduction before 4-thread-level reduction.

TODO: diagram of warp tile S_rmem containing MMA tiles.

```cpp
// initial attention state
float rowmax[WARP_Q / MMA_M][2];
float rowsumexp[WARP_Q / MMA_M][2] = {};
for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
  rowmax[mma_id_q][0] = -FLT_MAX;
  rowmax[mma_id_q][1] = -FLT_MAX;
}

// main loop
const int num_kv_iters = len_kv / BLOCK_KV;
for (int kv_idx = 0; kv_idx < num_kv_iters; kv_idx++) {
  // tile_S = tile_Q @ tile_K.T
  S_rmem[][] = ...

  // loop over rows
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    // apply softmax scale
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int reg_id = 0; reg_id < 4; reg_id++)
        S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

    // rowmax
    float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
      float *regs = S_rmem[mma_id_q][mma_id_kv];
      this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));  // c0 and c1
      this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));  // c2 and c3
    }

    // butterfly reduction within 4 threads
    this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
    this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
    this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
    this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));
  }

  ...
}
```

In a typical reduction kernel, when there are only 32 active threads left, we can use warp shuffle [`__shfl_down_sync()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions) to copy data from higher lanes to lower lanes, and the final result is stored in thread 0. In this case, since we need the max value to be shared across all threads (in a group of 4), we can use `__shfl_xor_sync()` to avoid a broadcast step.

TODO: figure illustrating butterfly reduction

#### Rescaling

With row max of the new tile, we can compute rescaling factor for (unnormalised) output as well as normaliser (sumexp of each row).

```cpp
// new rowmax
this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

// rescale for previous O
float rescale[2];
rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
  O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
  O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
  O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
  O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
}

// save new rowmax
rowmax[mma_id_q][0] = this_rowmax[0];
rowmax[mma_id_q][1] = this_rowmax[1];
```

We don't rescale `rowsumexp` here because we want to fuse it with addition of the new sumexp term later i.e. FMA - fused multiply add. We can't fuse multiplication with MMA, hence we need to do a single multiplication for `O_rmem[][]`.

#### Pack `tile_S` to `tile_P` (and row sum exp)

For the next part, we will loop over the row dimension again (`BLOCK_KV / MMA_N`), to compute and pack `tile_P` from `tile_S`, as well as doing reduction for sumexp. Recall that we declare registers for `S` and `P` as follows.

```cpp
float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4]      // m16n8
uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];  // m16k16
```

Look up the thread/register layout for MMA multiplicand A and output C/D again in PTX docs. Luckily, the layouts are exactly the same - within an 8x8 tile, the arrangement of elements is identical.

TODO: diagram that maps C/D layout to A layout.

It means that for all threads, every 2 floats in `S_rmem` can be packed as BF16x2 in a single 32-bit register of `P_rmem`, exactly how `mma.m16n8k16` expects for the 2nd MMA. There are no data movements across threads in this case. Note that this is not always true: if we use INT8 or FP8 MMA for the 1st and/or 2nd MMA, we would need to permute data across threads to pack `tile_S` to `tile_P`.

Our code for the last part of online softmax is below.

```cpp
// rowsumexp
float this_rowsumexp[2] = {};
for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
  float *regs = S_rmem[mma_id_q][mma_id_kv];
  regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);  // c0
  regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);  // c1
  regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);  // c2
  regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);  // c3

  this_rowsumexp[0] += regs[0] + regs[1];
  this_rowsumexp[1] += regs[2] + regs[3];

  // pack to P registers for next MMA
  // we need to change from m16n8 to m16k16
  nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
  this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
  this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
}

// butterfly reduction on this_rowsumexp[2]
...

// accumulate to total rowsumexp using FMA
rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
```

After this is the 2nd MMA: load V, then compute `tile_O += tile_P @ tile_V`. This completes our 1st version of Flash Attention. Actually we also need to normalise the output before writing `O_rmem` to global memory, but that part should be pretty straight-forward.

You can find the full code for the 1st version at [attention_v1.cu](https://github.com/gau-nernst/learn-cuda/blob/7e2d6951c3fb2b0211dca756fb2144126a352013/07_attention/attention_v1.cu).

### Benchmark setup

Wow, that's plentiful for the 1st version. Indeed, I spent the most time on version 1 trying to implement Flash Attention correctly. Took me 2 days to realize [`__shfl_xor_sync()`'s mask should be 2 (`0b10`) instead of `0x10` for butterfly reduction](https://github.com/gau-nernst/learn-cuda/commit/8fdb3e6a95a18502c2250571eeeb2179860936c0).

TODO: PyTorch benchmark. Correctness check

That's fine, we still have a few tricks up our sleeves for the next few versions.

## Version 2 - Shared memory swizzling

Nsight. Stall short scoreboard -> shared memory. ...

NVIDIA's shared memory is backed by 32 memory banks. Consecutive 4-byte memory addresses are assigned to consecutive memory banks. This poses a problem when we load data from shared memory to register memory with `ldmatrix` -

## Version 3 - 2-stage pipelining

## Version 4 - `ldmatrix.x4` for K and V

Previously, we use `ldmatrix.x2` for K and V since it naturally fits `n8k16` MMA tile. However, since we are handling a larger tile anyway, we can directly use `ldmatrix.x4` to issue fewer instructions. The trick is to select the appropriate 8x8 tiles and compute the row addresses correctly. There are two options: load `n16k16` tile, or `n8k32` tile.

## Version 5 - better pipelining
