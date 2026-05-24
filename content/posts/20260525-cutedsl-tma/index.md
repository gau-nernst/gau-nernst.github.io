+++
date = '2026-05-25T04:00:00+07:00'
title = 'Using TMA in CuteDSL'
url = 'cutedsl-tma'
+++
Some time ago, I explored using CuteDSL to write some kernels for vLLM to enjoy the JIT benefits (ability to specialize on more parameters, don't need to recompile vLLM from source during development, etc...). I have always known that we can write CUDA C++ way (SIMT style) in CuteDSL: there are `cute.arch.thread_idx()`/`cute.arch.block_idx()` (equivalent of `threadIdx`/`blockIdx` in CUDA C++), PTX (or its NVVM equivalent) can be used freely in device code. However, Tensor Memory Accelerator (TMA) is a special exception. Typically we need to encode tensor descriptors on the host side using CUDA driver API `cuTensorMapEncodeTiled()`, but this is not exposed in CuteDSL!

This article documents my little journey with Codex of figuring out various hoops we need to jump through in order to use TMA in CuteDSL. In the end, it provides a way to mentally map the CUDA C++ approach of using TMA to that in CuteDSL, which might not be the best way to use CuteDSL. After figuring out everything, I found out CuteDSL has an official [TMA tutorial](https://github.com/NVIDIA/cutlass/tree/v4.5.0/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma) (lmao), so that might worth a read as well.

{{< toc >}}

## Recap on using TMA in CUDA C++

Using an example of 2D TMA, first we encode the tensormap on the host side:

```cpp
CUtensorMap tmap;

constexpr uint32_t rank = 2;

// shape in gmem
// first dim is assumed to have stride = 1 elem
uint64_t globalDim[rank] = {N, M};
uint64_t globalStrides[rank-1] = {N * sizeof(nv_bfloat16)};

// shape in smem
uint32_t boxDim[rank]         = {BN, BM};
uint32_t elementStrides[rank] = {1, 1};

auto err = cuTensorMapEncodeTiled(
  &tmap,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
  rank,
  (void *)ptr,
  globalDim,
  globalStrides,
  boxDim,
  elementStrides,
  CU_TENSOR_MAP_INTERLEAVE_NONE,
  CU_TENSOR_MAP_SWIZZLE_NONE,
  CU_TENSOR_MAP_L2_PROMOTION_NONE,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
// check error code
```

On device side (kernel code), we (1) initialize mbarrier, (2) issue TMA, and (3) wait for TMA data to arrive.

```cpp
__global__
void kernel(const __grid_constant__ CUtensorMap tmap) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  // set up smem
  // alignment is required for TMA
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  const int STAGE_SIZE = BM * BN * sizeof(nv_bfloat16);
  const int mbar = smem + STAGE_SIZE;

  // init mbar
  if (warp_id == 0 && elect_sync()) {
    mbarrier_init(mbar, 1);
  }
  __syncthreads();

  // issue TMA
  if (warp_id == 0 && elect_sync()) {
    const int off_m = 0;
    const int off_n = 0;
    tma_2d_g2s(&tmap, off_n, off_m, smem, mbar);
    mbarrier_arrive_expect_tx(mbar, STAGE_SIZE);
  }

  // wait for data to arrive
  mbarrier_wait(mbar, 0);

  // do something with the data
}
```

For more detailed information, you can refer to my previous [tcgen05 blogpost](/tcgen05), or the official documentation [Using TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma).

## 2D TMA without swizzling

We start with 2D TMA without swizzling as it is the simplest: no swizzling magic, no box shapes restriction. A typical CuteDSL kernel looks like this

```python
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync

class Kernel:
    def __init__(self, BM: int, BN: int):
        self.BM = BM
        self.BN = BN
        # other hparams...

    @cute.jit
    def __call__(self, A: cute.Tensor):
        # create the equivalent of host-side tensormap encoding
        A_args = self.prepare_tma(A, ...)
        self.kernel(A_args).launch(grid=grid, block=block)

    @cute.kernel
    def kernel(self, A_args):
        ...
```

Let's look deeper into `prepare_tma()`.

```python
class Kernel:
    @cute.jit
    def prepare_tma(self, tensor: cute.Tensor, BM: cutlass.Constexpr, BN: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        s_layout = cute.make_layout((BM, BN), stride=(BN, 1))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            tma_op, tensor, s_layout, cta_tiler=(BM, BN)
        )
        return tma_atom, tma_tensor, s_layout
```

- `tma_op`: this declares the kind of data movement operation, typically maps to a specific family of PTX instructions. `CopyBulkTensorTileG2SOp` is basically [`cp.async.bulk.tensor.dim.shared::cta.global.tile`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) i.e. TMA copy from global to shared memory. There are others like `CopyBulkTensorTileS2GOp`, `LdMatrix8x8x16bOp`, `StMatrix8x8x16bOp`.
- `s_layout`: shared memory layout. In the example above, we declare a row-major `[BM, BN]` tile.
- `make_tiled_tma_atom()`: given a TMA op, gmem tensor, smem layout, and CTA tiler, this function returns `tma_atom` and `tma_tensor`. You may wonder what these are. For now, just think `tma_atom` as the tensormap descriptor, and ignore `tma_tensor`. It would make more sense when we see how TMA is issued using CuteDSL APIs.
  - You may also wonder why we need to pass `cta_tiler`: it seems to be the same as smem layout, or at least can be inferred from smem layout. I don't have an answer to this, it's just the way it is.

For any gmem tensor, we always have to create `tma_atom`, `tma_tensor`, and `s_layout`, and pass these three objects to the kernel. To improve readability, I pass them as a 3-element Python tuple argument, then unpack them in kernel code. The remaining steps roughly look like this:

```python
class Kernel:
    @cute.kernel
    def kernel(self, A_args: tuple[cute.CopyAtom, cute.Tensor, cute.Layout]):
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM = self.BM
        BN = self.BN

        A_tma_atom, A_tma_tensor, sA_layout = A_args

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, byte_alignment=128)
        mbar = smem.allocate_array(cutlass.Int64, 1)

        # init mbar
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar, 1)
        cute.arch.sync_threads()

        # issue TMA
        if warp_id == 0:
            # doesn't matter if we arrive before or after issuing TMA
            with cute.arch.elect_one():
                STAGE_SIZE = BM * BN * 2
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)

            # select the source tile from gmem
            # NOTE: we call this on the **tma_tensor**, not the original gmem tensor
            src = cute.local_tile(A_tma_tensor, tiler=(BM, BN), coord=(0, 0))

            # what's this? i have no idea
            tAsA, tAgA = cpasync.tma_partition(
                A_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sA, 0, 2),
                gmem_tensor=cute.group_modes(src, 0, 2),
            )

            # issue TMA copy using cute.copy() API
            # NOTE: this is called WITHOUT elect_one()
            cute.copy(A_tma_atom, tAgA, tAsA, tma_bar_ptr=mbar)

        # wait for TMA data to arrive
        cute.arch.mbarrier_wait(mbar, 0)
```

From here you can see why we need all three arguments for doing TMA: `tma_atom` and `tma_tensor` are used by `tma_partition()` and `cute.copy()` to issue TMA, while `s_layout` is used to allocate the smem tensor. Mbarrier initialization and `arrive_expect_tx` is pretty much 1-to-1 mapping to the same concepts in CUDA C++. The new difference is (1) selecting the source tile, and (2) using `tma_partition()` and `cute.copy()`. In CUDA C++, we pass the gmem tensor offsets to the TMA instructions. In CuteDSL, ignoring `tma_partition()` for now, `cute.copy()` takes in source and destination tensors, which are actually **views** into the underlying memory buffers, and copies data from source to destination. The smem destination is just the whole smem tensor itself. The gmem source is a **local tile** of the gmem tensor, matching the smem destination's shape.
- In this naive example, we use `coord=(0, 0)` meaning the first `[BM, BN]` tile of the `[M, N]` gmem tensor. If we tile `[M, N]` gmem tensor over the kernel's grid, we can for example use `coord=(bid_m, bid_n)`, where `bid_n, bid_m, _ = cute.arch.block_idx()`.

You may notice that we take the local tile from `tma_tensor`, instead of the original gmem tensor. And it seems like we can treat the TMA tensor like the original gmem tensor in the context of issuing TMA. This is an apt time to explain what a TMA tensor is.

### Cute's Layout, Tensor, and TMA Tensor

There is an official explanation of [TMA Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html), but it's rather dense, so I hope I can give a simpler explanation. Note that you should refer to Cute's documentation for the formal definitions. Let's start with Layout and Tensor, which we kinda gloss over previously.

A **Layout** consists of **shape** and **stride**, which are both tuples of integers. Shape is, well, shape, or dimensions. Stride is how many elements you need to jump to increment a particular dimension by 1.

Colloquial term | Cute's layout
---|---
Row-major `[BM,BN]` contiguous tile | `Layout(shape=(BM,BN), stride=(BN,1))`
Row-major `[BM,BN]` tile inside a `[M,N]` tensor | `Layout(shape=(BM,BN), stride=(N,1))`

Given a Layout, we can index into it using a **Coordinate**, which returns an **Index**. This index is computed as the dot product between the coordinate and the stride.

Layout | Coordinate | Index
-------|------------|--------
`Layout(shape=(BM,BN), stride=(BN,1))` | `(i,j)` | `i*BN + j`
`Layout(shape=(BM,BN), stride=(N,1))` | `(i,j)` | `i*N + j`

- I hope this concept is intuitive so far. This is the same as computing the linear index/offset of multidimensional arrays given a set of indices.
- Notice that we actually only need the Stride to compute the index.

A **Tensor** consists of an **Iterator** and a **Layout**. For most cases, Iterator is just a memory pointer. Hence, indexing into a Tensor, using a Coordinate, is (1) compute the index using its Layout, and (2) use the index to **dereference** the iterator/pointer and obtain the data. It's a formal way of saying:
- Given a 2D tensor `A[M][N]`
- The value of `A[i][j] = A_ptr[i * stride_am + j * stride_an]`

So far so good, but things get a bit tricky with TMA.
1. TMA layout (shape + stride) might not be the same as gmem layout, due to TMA's `boxDim` restrictions, or you may want to do funny things with TMA.
2. We use a set of **global offsets** to specify the gmem's tile to TMA.

There is no `ptr[index]` for TMA, but rather `tmap[off_x, off_y, ...]` (and that selects a **tile** rather than a single value). So Cute says, "hey, let's extend the definition of stride". Given that we want a **TMA Tensor** to map a gmem's coordinate to TMA offsets, which are both tuples, strides of TMA tensor become "mini-vectors" instead of plain integers.

For example, using the simple 2D TMA example above, where the gmem layout is a simple row-major `Layout(shape=(M,N), stride=(N,1))`, a possible TMA tensor would map gmem tensor coordinate `(i,j)` to TMA offsets `(j,i)`, since TMA's 1st dimension must have stride of 1. This can be achieved using `stride=(vec{0,1}, vec{1,0})`.

$$
\mathrm{TMA\ tensor}[i,j] = i (0, 1) + j (1, 0) = (j, i)
$$

- I'm calling them "mini-vectors" since it's easy to think of them that way. But the formal name for them is **Arithmetic Tuple**. This is to distinguish them from normal tuples, since a Cute's shape or stride can be composed of nested tuples i.e. Arithmetic Tuples can be scaled and added up, but normal tuples act as "structure" for shapes and strides.

You don't need to take my words for it. We can verify it with a self-contained example.

```python
from cutlass import BFloat16, cute
from cutlass.cute.nvgpu import cpasync

@cute.jit
def debug(A: cute.Tensor):
    tma_op = cpasync.CopyBulkTensorTileG2SOp()
    s_layout = cute.make_layout((16, 16), stride=(16, 1))
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (16, 16))
    print(f"{tma_tensor=}")
    print(f"{tma_tensor[2, 3]=}")

# compile the function, doesn't run it
M, N = 1024, 512
A = cute.runtime.make_fake_tensor(BFloat16, (M, N), (N, 1), assumed_align=8)
cute.compile(debug, A)
```

This will print

```
tma_tensor=tensor<(0,0) o (1024,512):(1@1,1@0)>
tma_tensor[2, 3]=(3, 2)
```

- `(0,0)` is a coordinate iterator (not sure if I use the right term) starting at `(0,0)`.
- `(1024,512):(1@1,1@0)` is the layout, with shape `(M,N)` (same as gmem tensor shape), and stride `({0,1}, {1,0})`.

You may notice that we seem to use **Tensor** and **Layout** interchangeably here. Technically the math of `i (0, 1) + j (1, 0) = (j, i)` can be done purely using layout. However, we can't take a local tile of a layout, only of a tensor. Adding the following lines in the above example:

```python
    local_tile = cute.local_tile(tma_tensor, (16, 16), (2, 3))
    print(f"{local_tile=}")
```

```
local_tile=tensor<(48,32) o (16,16):(1@1,1@0)>
```

You can see that it adds an offset to the iterator, and changes the shape. If we index into this local tile, the math becomes:

$$
\mathrm{local\ tile}[i,j] = (48,32) + i (0, 1) + j (1, 0) = (48+j, 32+i)
$$

All in all, we can slice the **TMA tensor** as if it is the original gmem tensor, and pass the resulted slice to TMA operations.

### Example: memcpy

To make sure our understanding of TMA is correct, we can turn our simple example above into a memcpy kernel: gmem->smem with TMA, then smem->gmem using manual indexing.

```python
import cutlass
import torch
from cutlass import cute
from cutlass.cute.nvgpu import cpasync


class Kernel:
    def __init__(self, BM: int, BN: int):
        self.BM = BM
        self.BN = BN
        self.tb_size = 128

    @cute.jit
    def prepare_tma(self, tensor: cute.Tensor, BM: cutlass.Constexpr, BN: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        s_layout = cute.make_layout((BM, BN), stride=(BN, 1))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, tensor, s_layout, cta_tiler=(BM, BN))
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor):
        A_args = self.prepare_tma(A, self.BM, self.BN)
        M, N = A.shape
        grid = (N // self.BN, M // self.BM, 1)
        block = (self.tb_size, 1, 1)
        self.kernel(A_args, B).launch(grid=grid, block=block)

    @cute.kernel
    def kernel(
        self,
        A_args: tuple[cute.CopyAtom, cute.Tensor, cute.Layout],
        B: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_n, bid_m, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM = self.BM
        BN = self.BN
        tb_size = self.tb_size

        A_tma_atom, A_tma_tensor, sA_layout = A_args

        # allocate smem
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, byte_alignment=128)
        mbar = smem.allocate_array(cutlass.Int64, 1)

        # init mbar
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar, 1)
        cute.arch.sync_threads()

        # issue TMA
        if warp_id == 0:
            with cute.arch.elect_one():
                STAGE_SIZE = BM * BN * 2
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)

            # select local tile
            src = cute.local_tile(A_tma_tensor, tiler=(BM, BN), coord=(bid_m, bid_n))
            tAsA, tAgA = cpasync.tma_partition(
                A_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sA, 0, 2),
                gmem_tensor=cute.group_modes(src, 0, 2),
            )
            cute.copy(A_tma_atom, tAgA, tAsA, tma_bar_ptr=mbar)

        # wait for TMA data to arrive
        cute.arch.mbarrier_wait(mbar, 0)

        # copy from A smem to B gmem
        for i in cutlass.range_constexpr(BM * BN // tb_size):
            idx = i * tb_size + tid
            col = idx % BN
            row = idx // BN
            B[bid_m * BM + row, bid_n * BN + col] = sA[row, col]

    @staticmethod
    def compile(BM: int, BN: int):
        M = cute.sym_int()
        N = cute.sym_int(divisibility=8)
        A = cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (M, N),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=8,
        )
        B = cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (M, N),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=8,
        )
        return cute.compile(Kernel(BM, BN), A, B, options="--enable-tvm-ffi")


A = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
B = torch.zeros_like(A)

kernel = Kernel.compile(128, 128)
kernel(A, B)
torch.testing.assert_close(A, B)
```

You can play around with the memcpy kernel above to strengthen understanding. Before we move on, I want to point out one more thing. By `cute.copy()`'s convention, the first mode of source and destination tensors should be covered by the copy atom (i.e. `[BM, BN]` in our case), which corresponds to a single TMA instruction. The remaining modes (typically called "rest modes") indicate the number of repeated TMA atoms / TMA instructions. This is the reason we need to call `cute.group_modes()`, which "collapses" the 2D tensors into a single mode.

We can print the TMA atom to inspect it, using `BM=BN=128` from the example above.

```
Copy Atom
  ThrID:         1:0
  TV Layout Src: (1,16384):(0,1)
  TV Layout Dst: (1,16384):(0,1)
  Value type:    bf16
```

It only shows the information as a generic **Copy Atom**. Nevertheless, we can see it corresponds to `128*128=16384` BF16 elements. Thread-Value (TV) Layout is outside the scope of this article.

## 2D TMA with swizzling

In CUDA C++, we specify swizzle type when encoding the tensormap on host side i.e. `cuTensorMapEncodeTiled()`. In CuteDSL, we specify this via smem layout being passed to `make_tiled_tma_atom()`.
- Remember, the TMA atom object holds the TMA descriptor. Hence, logically, any arguments that we normally pass to `cuTensorMapEncodeTiled()` in CUDA C++ must be passed to or deduced by `make_tiled_tma_atom()`.

```python
class Kernel:
    @cute.jit
    def prepare_tma(self, tensor: cute.Tensor, BM: cutlass.Constexpr, BN: cutlass.Constexpr):
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        s_layout = cute.make_layout((BM, BN), stride=(BN, 1))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)

        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, tensor, s_layout, cta_tiler=(BM, BN))
        return tma_atom, tma_tensor, s_layout
```

First, you can take the following table for granted. How Cute computes swizzling is actually not that important.

TMA swizzle type | Cute swizzle `(B,M,S)`
-----------------|-----------------------
32B              | `(1,4,3)`
64B              | `(2,4,3)`
128B             | `(3,4,3)`

Then, just know that `cute.make_composed_layout()` is how we apply a swizzle on an existing layout.
- There are various online materials explaining Cute swizzle (e.g. [Simon's blog](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/)). In short, `(B,M,S)` means that we keep `M` LSBs untouched, and the next `B` bits will be XOR-ed. The XOR pattern is obtained by right-shifting the bits by `S`.
- All swizzle patterns above are applied on the raw smem address (it's possible to swizzle smem offset instead as well). `M=4` corresponds to 16 bytes, which is the unit of being swizzled in TMA. `S=3` means that we take bits starting at bit `M+S=7` as the XOR pattern, corresponding to 128 bytes, or 32 memory banks.

One of my initial attempts was to modify `prepare_tma()` like above, and re-ran the memcpy example as is. I got hit with:

```
  File "/home/thien/learn-cuda/02e_matmul_sm100/memcpy.py", line 70, in then_block_3
    tAsA, tAgA = cpasync.tma_partition(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/thien/learn-cuda/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/_mlir/dialects/_cute_nvgpu_ops_gen.py", line 3299, in __init__
    super().__init__(self.build_generic(attributes=attributes, operands=operands, successors=_ods_successors, regions=regions, loc=loc, ip=ip))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Operation creation failed
```

Uh oh, what is this. I knew for sure I couldn't debug this on my own, but luckily we have coding agents now. Codex gladly figured out that we had to modify how we allocate the smem tensor as well.

```python
sA = smem.allocate_tensor(
    cutlass.BFloat16,
    sA_layout.outer,
    byte_alignment=128,
    swizzle=sA_layout.inner,
)
```

It's not immediately obvious, but this means that we apply the swizzle **on the smem address** rather than smem offset. It makes sense because our previous swizzle pattern `(3,4,3)` operates on the raw address directly.
- This raises the question on why `make_tiled_tma_atom()` takes in a composed layout, which would mean the swizzling is applied on the **smem offset**. I don't quite understand this.

Rerunning the memcpy example:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

🤡. The previous compile error is gone, but now we got a runtime error. I kinda knew there would be an error if I used 128B swizzling with `[128,128]` BF16 tile, as TMA's restriction says that for 128B swizzling, the innermost box dimension must be at most 128B, or 64 BF16 values. This is funny because CuteDSL doesn't stop the users from creating an invalid TMA atom, and it requires understanding of TMA, which is abstracted away, to know why the error is happening.

The fix is simple: use `BN=64`. Re-running the example again, I expected the assert to fail, since we hadn't updated the indexing logic into the smem tensor to account for swizzling. But the check passed! Turns out, since the smem tensor has swizzling enabled, CuteDSL automatically computes swizzling for us. This would be convenient for a real kernel - we don't need to think so much about swizzling when accessing the data. But for education purpose, let's compute the offset after 128B swizzling manually, to confirm that the smem layout is indeed what we would expect from TMA with 128B swizzling.

```python
# copy from A smem to B gmem
for i in cutlass.range_constexpr(BM * BN // tb_size):
    idx = i * tb_size + tid
    col = idx % BN
    row = idx // BN
    swizzled_col = ((col // 8) ^ (row % 8)) * 8 + (col % 8)
    B[bid_m * BM + row, bid_n * BN + col] = sA.iterator[row * BN + swizzled_col]
```

`(col // 8)` is the column in the units of 16B, which is XOR-ed by the row index. Indexing into `.iterator` removes any automated swizzling applied by CuteDSL. This gives correct results as expected.
- As a practice, you can try using `32B` or `64B` swizzling to understand their shared memory layout.

## (Slightly) more advanced patterns

We have covered the basics of using TMA in CuteDSL. Let's move on some typical patterns I have come across.

### GEMM mainloop

In a GEMM mainloop, we iterate over the K dimension, incrementing by `BLOCK_K` at a time. On the smem side, we have multiple buffers to be able to prefetch multiple smem stages ahead of MMA stage. Let's look at how we prepare the TMA arguments first.

```python
class Kernel:
    @cute.jit
    def prepare_tma(self, A: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)

        # we must put num_stages as the last mode since CuteDSL APIs follow this convention
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)

        # don't need to select 1 stage of s_layout, make_tiled_tma_atom() does it internally
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))
        return tma_atom, tma_tensor, s_layout
```

In the kernel side

```python
for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
    mbar = tma_full_mbar + tma_stage
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, self.stage_size)

    src = cute.local_tile(A_tma_tensor, (BM, BK), (bid_m, iter_k))
    dst = sA[None, None, tma_stage]
    tAsA, tAgA = cpasync.tma_partition(
        A_tma_atom,
        cta_coord=0,
        cta_layout=cute.make_layout(1),
        smem_tensor=cute.group_modes(dst, 0, 2),
        gmem_tensor=cute.group_modes(src, 0, 2),
    )
    cute.copy(A_tma_atom, tAgA, tAsA, tma_bar_ptr=mbar)

    tma_stage = (tma_stage + 1) % self.num_stages
```

Note that in Cute, indexing with `None` means selecting all of that mode i.e. `A[None]` actually means `A[:]` in NumPy/PyTorch's convention. Hence, `sA[None, None, tma_stage]` means selecting the `tma_stage`-th smem buffer.

### "Folding" TMA dimensions

When the innermost dimension of smem tile exceeds 128B, such as loading `(128,128)` BF16 tile, usually we have to "fold" this dimension into a new one. In CUDA C++, we can reinterpret the existing gmem tensor (assuming BF16 data type).
- Original gmem layout: `shape=(M,N), stride=(N,1)`
- Unflatten: `shape=(M,N/64,64), stride=(N,64,1)`
- Permute: `shape=(N/64,M,64), stride=(64,N,1)`
- => smem layout: `shape=(BN/64,BM,64), stride=(BM*64,64,1)`

Hence, a 2D logical tile becomes a 3D TMA. Technically we can do the exact same thing in CuteDSL by creating a new gmem tensor from the gmem pointer.

```python
A  # shape=(M,N), stride=(N,1)
M, N = A.shape
A_new = cute.make_tensor(
    A.iterator,
    layout=cute.make_layout((N//64, M, 64), stride=(64, N, 1)),
)

BM = 128
BN = 128
tma_op = cpasync.CopyBulkTensorTileG2SOp()
sA_layout = cute.make_layout((BN // 64, BM, 64), stride=(BM * 64, 64, 1))
sA_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, sA_layout)
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(tma_op, A_new, sA_layout, (BN // 64, BM, 64))
```

But Cute layout is supposed to be very powerful, so I was looking for a "cleaner" way to express this. After some tinkering, this was what worked for me.

```python
A  # shape=(M,N), stride=(N,1)

BM = 128
BN = 128
tma_op = cpasync.CopyBulkTensorTileG2SOp()
sA_layout = cute.make_layout((BM, (64, BN//64)), stride=(64, (1, BN * 64)))
sA_layout = cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, sA_layout)
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    tma_op,
    cute.logical_divide(A, (None, 64)),  # shape=(M, (64, N/64))
    sA_layout,
    (BM, BN),
)
```

The smem layout is still equivalent to that in the earlier example, but we preserve the top-level rank 2 tensor. This can be useful in many cases, such as when we need to index into the smem tensor - we can treat it as 2D tensor `sA[i,j]` and CuteDSL will figure out the rest. Similarly, when handling the TMA tensor, we can take its 2D local tile as usual.

### Uneven TMA offsets

One typical case for attention is packed varlen sequences i.e. we have N sequences of length $l_1, ..., l_N$, and they are tightly backed together. We can't use `cute.local_tile()` directly as `local_tile()` can only specify offsets being multiples of the given tiler. The solution is rather simple: use `cute.domain_offset()` for arbitrary offsets.

```python
Q  # [total_tokens, dim]

# [BM, dim] from Q[bos + tile_id * BM : bos + (tile_id+1) * BM, :]
src = cute.local_tile(
    cute.domain_offset((bos, 0), Q),
    tiler=(BM, dim),
    coord=(tile_id, 0),
)
```

It sometimes feels rather verbose to express such a simple slicing. Let me know if there are better ways.

## Closing remarks

Using TMA in CuteDSL requires some mental gymnastics to manipulate the tensors' layout as to what CuteDSL API expects. But once those are set up correctly, using TMA is quite smooth-sailing. It's worth noting that each `cute.copy()` statement may issue more than 1 `cp.async.bulk.tensor` PTX instruction if CuteDSL can't figure out your intention from the given gmem tensor and smem layout, hence it's good to check the generated PTX (coding agents are very good at this).
