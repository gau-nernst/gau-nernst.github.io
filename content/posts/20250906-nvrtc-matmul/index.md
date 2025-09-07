+++
date = '2025-09-06T15:25:20+08:00'
title = 'Use NVRTC to explore MMA instruction variants'
url = 'nvrtc-matmul'
+++
Recently I tweeted about realistic Speed-of-Light (SOL) of 5090 and RTX PRO 6000 for some dtypes, and [mobicham asked me about FP8 MMA with FP16 accumulation](https://x.com/mobicham/status/1963540008947617947). I of last year would turn to Triton for this - it's trivial to change the accumulation dtype of [tl.dot()](https://triton-lang.org/main/python-api/generated/triton.language.dot.html). However, I roughly know how to write a fast matmul kernel now, so why not do it myself! In addition, I have been tinkering around with [`torch.cuda._compile_kernel()`](https://github.com/pytorch/pytorch/pull/151484), which compiles CUDA kernels super fast via [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html). This seems ideal for JIT-compiled kernels and enumerating all possible permutations of the MMA instruction.

What are the alternatives for authoring CUDA C++ kernels with PyTorch? The [standard approach](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) is to write everything in C++, expose a Python binding or register a PyTorch custom op, and build it as a native extension for Python. To try all possible variants of MMA, I would need to either make a lot of duplicate code, or deal with C++ template hell. It's also terribly slow to compile a PyTorch C++ extension. [torch.utils.cpp_extension.load/load_inline()](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) can be used to handle the former issue: it JIT-compiles the provided code using the same facility as `CUDAExtension`, so we can do some form of string manipulation in Python to codgen appropriate CUDA C++ code. However, the speed issue is not resolved.

You may ask, if I just want to try FP8 MMA with FP16 accumulation, why not use Triton like I mentioned from the start. Sometimes there are issues with Triton: the Triton version released with PyTorch 2.8.0 still has bugs around using FP8 inputs to `tl.dot()`. It was reportedly fixed in [triton-lang/triton#7409](https://github.com/triton-lang/triton/pull/7409), but perhaps it didn't get included in the latest stable release. And I don't want to build Triton from source just for this. Moreover, previously I wanted to play with INT4 MMA, but Triton did not (and still does not) support it - [triton-lang/triton#675](https://github.com/triton-lang/triton/issues/675).

[Cutlass](https://github.com/NVIDIA/cutlass) is another reasonable option, and they do support FP8 with FP16 accumulation, as well as INT4 MMA for my past interest. If I need a proper kernel for production, I would probably use Cutlass'. But in the mean time, let's find an excuse to build my own templated matmul with NVRTC.

{{< toc >}}

## NVRTC is fast! 🏎️

Before we proceed, I want to emphasize how fast NVRTC is. Consider a simple add kernel

```cpp
// kernel.cu
__global__
void add_kernel(const int *A, const int *B, int *C, int M, int N) {
  // A, B, C are 2D matrices with shape M, N
  // we will use 1 threadblock to handle 1 row
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tb_size = blockDim.x;

  A += bid * N;
  B += bid * N;
  C += bid * N;

  for (int col = tid; col < N; col += tb_size)
    C[col] = A[col] + B[col];
}
```

Let's try integrating this kernel to PyTorch with both `load()`/`load_inline()` and `_compile_kernel()`. For the first option, we need kernel launch and glue code in C++.

```cpp
// host.cu
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

__global__
void add_kernel(const int *A, const int *B, int *C, int M, int N);

at::Tensor add(at::Tensor A, at::Tensor B) {
  const int M = A.size(0);
  const int N = A.size(1);
  auto C = at::empty_like(A);

  auto A_ptr = A.data_ptr<int>();
  auto B_ptr = B.data_ptr<int>();
  auto C_ptr = C.data_ptr<int>();

  const int TB_SIZE = 256;
  add_kernel<<<M, TB_SIZE>>>(A_ptr, B_ptr, C_ptr, M, N);
  return C;
}

TORCH_LIBRARY(my_extension, m) {
  m.def("add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_extension, CUDA, m) {
  m.impl("add", &add);
}
```

To build and call the kernel

```python
import time

import torch
from torch.utils.cpp_extension import load_inline

kernel_src = open("kernel.cu").read()
host_src = open("host.cu").read()

t0 = time.perf_counter()
load_inline(
    "my_extension",
    cpp_sources=[],
    cuda_sources=[kernel_src, host_src],
    is_python_module=False,
    no_implicit_headers=True,
)
duration_load_inline = time.perf_counter() - t0
print(f"{duration_load_inline=:.4f} s")

M, N = 100, 1024
a = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
b = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
torch.testing.assert_close(torch.ops.my_extension.add(a, b), a + b)
```

On the first run, the duration is 9.1913s. In subsequent runs, the duration is reduced to 0.0239s thanks to caching. But do remember that we won't get the benefits of caching if we tweak the kernel e.g. autotuning, change dtypes.

Let's do the same exercise for `_compile_kernel()`.

```python
t0 = time.perf_counter()
kernel = torch.cuda._compile_kernel(kernel_src, "add_kernel")
duration_compile_kernel = time.perf_counter() - t0
print(f"{duration_compile_kernel=:.4f} s")


def add2(A: torch.Tensor, B: torch.Tensor):
    M, N = A.shape
    C = torch.empty_like(A)
    kernel((M, 1, 1), (256, 1, 1), (A, B, C, M, N))
    return C


M, N = 100, 1024
a = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
b = torch.randint(-100, 100, (M, N), device="cuda", dtype=torch.int32)
torch.testing.assert_close(add2(a, b), a + b)
```

Notice that our kernel launch code is now in Python, which will be translated to `ctypes` objects and passed to `cuLaunchKernel()`. The compile duration is 0.0095s, a 1000x speedup! Usually I don't like exaggerated graphs, but let's make one to illustrate the significant speedup.

{{< figure src="cold_compile_time.svg" alt="Cold compile time" caption="Cold compile time of `load_inline()` and `_compile_kernel()` for a simple add kernel." align="center">}}

This is awesome for developing kernels, especially for autotuning kernels params.

## Working with _compile_kernel()

### Development workflow

The [current _compile_kernel()](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1736-L1743) accepts two key arguments for passing the kernel code: `kernel_source` and `header_code`. The former must start with the kernel code (and with C linkage i.e. `extern "C"`), while the latter can contain `#include` statements and other function / variable declarations and definitions. There is no big difference in how the two are treated - they are concatenated later anyway before passing to NVRTC.

Having C linkage means that our kernel can't use C++ template. This is fine since we can declare template parameters as global `constexpr` variables or type aliases in the header code, and replace those definitions at compile time. The kernel content remains exactly the same.

```cpp
// before
template <int BLOCK_M, typename TypeInput>
__global__
void matmul_kernel(const TypeInput *A,
                   const TypeInput *B,
                         float     *C,
                   int M, int N, int K) {
  // do something with BLOCK_M and TypeInput
}

// after. in `kernel_source`
extern "C"
__global__
void matmul_kernel(const TypeInput *A,
                   const TypeInput *B,
                         float     *C,
                   int M, int N, int K) {
  // same as before
}

// after. in `header_code`
// the content can be modified in Python
// this will be placed before `kernel_source`
constexpr int BLOCK_M = 128;
using TypeInput = float;
```

Initially I was trying to write the kernel source and header code as a big multi-line string in Python.

```python
import torch

KERNEL_SOURCE = """
extern "C"
__global__
void matmul_kernel(const TypeInput *A,
                   const TypeInput *B,
                         float     *C,
                   int M, int N, int K) {
  ...
}
"""

HEADER_CODE = """
constexpr int BLOCK_M = 128;
using TypeInput = float;
"""

kernel = torch.cuda._compile_kernel(
  KERNEL_SOURCE,
  kernel_name="matmul_kernel",
  header_code=HEADER_CODE,
)
```

But I quickly ran into several limitations
1. **No syntax highlighting / autocompletion features**: The first issue can be alleviated using [Mark's VSCode extension](https://marketplace.visualstudio.com/items?itemName=msaroufim.pytorch-load-inline-highlighter). However, I got into some weird cases where it also affects syntax highlighting of my Python code in the same file, probably because of the prefix and suffix rules `cuda_` / `_cuda`. And it doesn't help with autocompletion.
2. **Special characters need extra escaping**: Newline characters must be written as `\\n` so that Python won't treat them as actual newline characters in the code string. This means that I can't directly copy-paste code to and from `.cu` files. Not a big deal, but inconvenient.

I explored putting the kernel in a standalone file `.cu` instead. When we need to compile the kernel, Python can open and read the files. It works pretty well! The kernel writing experience is exactly the same as the usual CUDA C++ development.

```bash
cuda_mm
├── __init__.py
└── kernel.cu
```

Recall that `_compile_kernel()` requires `kernel_source` to start with the kernel function. A simple solution is to add a special marker to separate the header code and kernel source.

```cpp
// header section
#include "common.h"

constexpr int BLOCK_M = 128;
using TypeInput = float;

// kernel section
// start of kernel -> this is our special marker
extern "C"
__global__
void matmul_kernel(const TypeInput *A,
                   const TypeInput *B,
                         float     *C,
                   int M, int N, int K) {
  ...
}
```

The header section in `kernel.cu` only exists to ensure syntax highlighting and autocompletion work. When we compile the kernel, we can generate the header dynamically depending on the required parameters. You can see my full example here: https://github.com/gau-nernst/gn-kernels/tree/aa0a6794/gn_kernels/cuda_mm.

### Missing CUDA and C++ headers :(

It's pretty standard to include `<cuda_bf16.h>` and/or `<cuda_fp16.h>` headers to work with BF16/FP16 dtypes. However, trying to compile code using those headers with NVRTC results in errors.

```
E           RuntimeError: Kernel compilation failed:
E           matmul_kernel.cu(2): catastrophic error: cannot open source file "cuda_bf16.h"
E             #include <cuda_bf16.h>
```

It turns out NVRTC doesn't automatically include header search paths like NVCC does. It's straight-forward to fix - include `/usr/local/cuda/include`, which is where my CUDA Toolkit installation stores its headers, in `_compile_kernel()`'s `cuda_include_dirs` argument. We can also use `torch.utils.cpp_extension.include_paths("cuda")` to automatically obtain CUDA include paths.

In my original matmul kernel, I also extensively used `uint32_t`/`int8_t` from `<cstdint>`. This also doesn't work with NVRTC due to missing headers. I could use their original native types e.g. `unsigned int`, `signed char`, which I did, but it doesn't solve the root issue of missing C++ standard library headers. I also need `std::is_same_v` for writing asm PTX for various dtypes (which will be covered later). I did try including those headers from GCC, but they didn't work with NVRTC.

Fortunately, I remembered there was CUDA C++ Standard Library - https://nvidia.github.io/cccl/libcudacxx/standard_api.html, which provides implementations of C++ standard features for both device and host code. Usage is simple: replace the C++ standard headers with their CUDA equivalent, and add `cuda::` prefix as needed.

```diff
- #include <cstdint>
+ #include <cuda/std/cstdint>
+ #include <cuda/std/type_traits>

uint32_t reg;  // unchanged
- std::is_same_v<TypeA, TypeB>;
+ cuda::std::is_same_v<TypeA, TypeB>;
```

## Writing a parameterized matmul kernel

Before I proceed, since explaining a matmul kernel (with Tensor Cores) in depth is not in the scope of this post, I welcome readers to refer to these amazing materials if you are not familiar with the subject.
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
- https://www.spatters.ca/mma-matmul

So far we have discussed about the engineering side of compiling kernels with NVRTC, but have yet touched on how to actually write a matmul kernel that can be adapted to various dtypes. Let's limit our design space to the following parameters:
- Data types:
  - `TypeAB`: dtype of input A and B. We assume they have the same dtype, though MMA does support some mixed precision - INT8 and UINT8, FP8_E4M3 and FP8_E5M2, and all pairs of FP8/FP6/FP4 on SM120.
  - `TypeAcc`: accumulation dtype. This can only be FP32, INT32, or FP16. FP32 is the default accumulation dtype for floating point types, and INT32 is the one for integer types. Additionally, FP16 and FP8 MMA also support FP16 accumulation.
  - `TypeC`: dtype of output C. This can be different from `TypeAcc`, such as when A and B are BF16, and we want the output also be BF16 even though the accumulation dtype is FP32.
- Kernel hyperparameters:
  - `BLOCK_M` / `BLOCK_N` / `BLOCK_K`: Shape of threadblock tile.
  - `NUM_WARP_M` / `NUM_WARP_N`: How we partition a threadblock tile into warp tiles.
  - `NUM_STAGES`: Number of stages for pipelining.
  - `GROUP_M`: Control threadblock tile swizzling to improve L2 cache reuse for large M.

The `TypeAB` and `TypeAcc` will dictate what MMA instruction we use. The kernel hyperparameters are used for autotuning to find the best kernel for a given problem shape (and dtypes). Both A and B are K-major, and C is N-major.

### Support FP16 and BF16 MMA

I used my old matmul kernel from here, https://github.com/gau-nernst/learn-cuda/blob/4038627c/02b_matmul_tensorop/matmul_v6.cu, as the starting point. It's a pretty standard Ampere BF16 matmul kernel using `cp.async` and `ldmatrix`. To extend support for FP16, I replace `nv_bfloat16` with `TypeAB`, and everything should work correctly for both BF16 (`nv_bfloat16` from `<cuda_bf16.h>`) and FP16 (`half` from `<cuda_fp16.h>`), except the MMA instruction and epilogue code. Currently the MMA instruction is hard-coded to BF16.

```cpp
__device__ inline
void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "    // D
               "{%4, %5, %6, %7}, "    // A
               "{%8, %9}, "            // B
               "{%10, %11, %12, %13};" // C
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}
```

Dispatching different MMA instruction depending on `TypeAB` is the perfect use case for C++ template. I can do something like this.

```cpp
template <typename TypeAB>
__device__ inline
void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]);

template<>
__device__ inline
void mma_m16n8k16<nv_bfloat16>(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ...");
}

template<>
__device__ inline
void mma_m16n8k16<half>(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...");
}
```

This is fine if I only need to support FP16 and BF16 MMA. To support FP16 accumulation, as well as INT8 MMA, I can additionally make `TypeAcc` as a template parameter. However, this means that I need to manually list out all MMA variants, at least the ones I care about. Although this is in fact what [Cutlass does](https://github.com/NVIDIA/cutlass/blob/v4.1.0/include/cutlass/arch/mma_sm80.h), I was hoping for a less tedious way.

#### asm string builder

Upon carefully re-reading NVIDIA's [Inline PTX Assembly Guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html), there is a way to manipulate PTX instruction string using templates! Based on their examples, I came up with this.

```cpp
// convert C++ type to PTX string
template <typename T>
struct Type_str;
template<> struct Type_str<half> { static constexpr const char value[] = "f16"; };
template<> struct Type_str<nv_bfloat16> { static constexpr const char value[] = "bf16"; };

template <typename TypeAB>
__device__ inline
void mma_m16n8k16(int A[4], int B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.%14.%14.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
                "C"(Type_str<TypeAB>::value));
}
```

Now our `atype` and `btype` for the MMA instruction is adaptive based on `TypeAB`! It's important that we use `static constexpr const char value[]` type so that the string is known at compile time, including its length (`char *value` is a pointer and its length is unknown to the compiler). Ignoring other potential challenges ahead, it looks like supporting other dtypes like INT8 and FP8 is straight-forward: I just need to add more dtype conversion rules for `Type_str` struct.

#### Epilogue

The final thing we need to take care of is the epilogue: convert FP32 accumulate to BF16/FP16 and write the result to global memory. I took the easy route here: conditional code based on output dtype.

```cpp
const float *acc = C_rmem[mma_id_m][mma_id_n];

if constexpr (cuda::std::is_same_v<TypeC, nv_bfloat16>) {
  reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 0) * N + col)[0] = __float22bfloat162_rn({acc[0], acc[1]});
  reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 8) * N + col)[0] = __float22bfloat162_rn({acc[2], acc[3]});
}
else if constexpr (cuda::std::is_same_v<TypeC, half>) {
  reinterpret_cast<half2 *>(C_gmem + (row + 0) * N + col)[0] = __float22half2_rn({acc[0], acc[1]});
  reinterpret_cast<half2 *>(C_gmem + (row + 8) * N + col)[0] = __float22half2_rn({acc[2], acc[3]});
}
```

This assumes output dtype can only be BF16 or FP16, which is fine for now. I tried looking at the [cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt) instruction to find a generic way to convert dtypes, but it doesn't seem to worth the effort. Only FP32->FP16/BF16/FP8 conversion has a specialized convert-and-pack version, and not all TypeAcc->TypeC are possible, at least directly e.g. INT32->FP8 (though this conversion doesn't quite make sense). In addition, I would need to allocate registers of different dtypes depending on `TypeC`, and the `asm volatile` statement can potentially be different depending on how I unpack and pack a register. So let's assume the user (i.e. me) is sensible and only output FP16 or BF16 for now.

### Support INT8 and FP8 MMA

BF16 and FP16 have the same bit-width, so supporting both of them is simple. INT8 and FP8 have smaller bit-width, hence we need to be very careful about size and address calculations.
- For **global memory addresses**, since we use C++ pointers of their corresponding types, address calculation works as expected.
- For **shared memory addresses**, we operate on the raw `uint32_t` shared state space address. Multiplying offsets in terms of their elements with the bit-width (in bytes) would handle things correctly.

`cp.async` code works out of the box since I already took into account bit-width of element dtype. However, we must pay special care for `ldmatrix` and `mma` instructions, which must match the tile shape required for INT8/FP8 MMA.

Even though FP16/BF16 and INT8/FP8 appear to use different MMA shapes - `m16n8k16` and `m16n8k32` respectively, the shapes have the same physical size!
- The width of FP16/BF16 `m16k16` A tile is `16 * 2 = 32` bytes
- The width of INT8/FP8 `m16k32` A tile is `32 * 1 = 32` bytes

The same observation can be made for B and C tiles. Translating this observation to our code for `ldmatrix`.

```diff
// 16 for BF16/FP16, 32 for INT8/FP8
- constexpr int MMA_K = 16;
+ constexpr int MMA_K = 32 / sizeof(TypeAB);

// set up register memory
// regardless of BF16/FP16 or INT8/FP8, for each MMA tile,
// we still use 4 A registers, 2 B registers, and
// 4 C registers per thread.
int A_rmem[WARP_M / MMA_M][BLOCK_K / MMA_K][4];
int B_rmem[WARP_N / MMA_N][BLOCK_K / MMA_K][2];
- float C_rmem[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};
+ TypeAcc C_rmem[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};

// pre-compute ldmatrix address and swizzling
int A_smem_thread, B_smem_thread;
{
  const int m = (warp_id_m * WARP_M) + (lane_id % 16);
  // column offset is always multiplied by 16,
  // which is the width of ldmatrix tile (16-byte).
  // we only need to multiply sizeof(TypeAB) for row offset,
  // since BLOCK_K is in the unit of elements.
-  const int k = (lane_id / 16) * 8;
-  A_smem_thread = swizzle<BLOCK_K * sizeof(TypeAB)>(A_smem + (m * BLOCK_K + k) * sizeof(TypeAB));
+  const int k = (lane_id / 16) * 16;  // in bytes
+  A_smem_thread = swizzle<BLOCK_K * sizeof(TypeAB)>(A_smem + (m * BLOCK_K * sizeof(TypeAB)) + k);
}
```

Main loop code doesn't require any adjustments, except for adding INT8/FP8 support to our `mma()` PTX wrapper. Since INT8 uses INT32 accumulation, we need `TypeAcc` as an extra template parameter. We also need a way to select the correct MMA shape, which can be done in the same fashion as how we select PTX type based on C++ type.

```cpp
template<> struct Type_str<__nv_fp8_e4m3> { static constexpr const char value[] = "e4m3"; };
template<> struct Type_str<__nv_fp8_e5m2> { static constexpr const char value[] = "e5m2"; };
// NOTE: according to C/C++ spec, sign-ness of char is implementation-defined
template<> struct Type_str<signed char> { static constexpr const char value[] = "s8"; };
template<> struct Type_str<unsigned char> { static constexpr const char value[] = "u8"; };

// MMA shape based of element size of TypeAB
template <int element_size>
struct MMA_shape_str;
template<> struct MMA_shape_str<2> { static constexpr const char value[] = "m16n8k16"; };
template<> struct MMA_shape_str<1> { static constexpr const char value[] = "m16n8k32"; };

template <typename TypeAB, typename TypeAcc>
__device__ inline
void mma(int A[4], int B[2], TypeAcc *C) {
  // m16n8k16 for FP16/BF16
  // m16n8k32 for FP8/INT8
  using shape = MMA_shape_str<sizeof(TypeAB)>;

  if constexpr (cuda::std::is_same_v<TypeAcc, float>)
    asm volatile("mma.sync.aligned.%14.row.col.f32.%15.%15.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
                  "C"(shape::value), "C"(Type_str<TypeAB>::value));

  else if constexpr (cuda::std::is_same_v<TypeAcc, int>)
    asm volatile("mma.sync.aligned.%14.row.col.satfinite.s32.%15.%15.s32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]),
                  "C"(shape::value), "C"(Type_str<TypeAB>::value));
}
```

Sadly we can't quite use a single `asm volatile` statement for everything since integer MMA requires `.satfinite` to clamp the output. Without this specifier, overflow will wrap around, which is something I can't imagine to ever be useful. Though for INT8 MMA, it requires at least K=131072 for overflow to happen (assuming all elements are -128), and for UINT8, at least K=33025 (all elements are 255). Another option is to add another templated struct to insert `.satfinite` conditionally based on `TypeAB`.

#### Epilogue (again!)

Again, let's assume the user is sensible, so only the following combinations are permitted:
- `TypeAcc=float` (can be FP16/BF16/FP8 MMA) -> `TypeC` can only be BF16 or FP16 -> we can use the previous code.
- `TypeAcc=int` (INT8 MMA) -> `TypeC` can only be INT32 -> no dtype conversion needed.

Our new epilogue looks as follows.

```cpp
const TypeAcc *acc = C_rmem[mma_id_m][mma_id_n];

if constexpr (cuda::std::is_same_v<TypeAcc, int> && cuda::std::is_same_v<TypeC, int>) {
  reinterpret_cast<int2 *>(C_gmem + (row + 0) * N + col)[0] = reinterpret_cast<const int2 *>(acc)[0];
  reinterpret_cast<int2 *>(C_gmem + (row + 8) * N + col)[0] = reinterpret_cast<const int2 *>(acc)[1];
}
else if constexpr (cuda::std::is_same_v<TypeAcc, float> && cuda::std::is_same_v<TypeC, nv_bfloat16>) {
  reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 0) * N + col)[0] = __float22bfloat162_rn({acc[0], acc[1]});
  reinterpret_cast<nv_bfloat162 *>(C_gmem + (row + 8) * N + col)[0] = __float22bfloat162_rn({acc[2], acc[3]});
}
else if constexpr (cuda::std::is_same_v<TypeAcc, float> && cuda::std::is_same_v<TypeC, nv_bfloat16>) {
  reinterpret_cast<half2 *>(C_gmem + (row + 0) * N + col)[0] = __float22half2_rn({acc[0], acc[1]});
  reinterpret_cast<half2 *>(C_gmem + (row + 8) * N + col)[0] = __float22half2_rn({acc[2], acc[3]});
}
```

It's getting kinda ugly but oh well, we are not writing a compiler. And not that I have a knowledge to write one.

### Support MMA with FP16 accumulation

This is the trickiest part. Previously, we didn't need to change the number of registers per MMA tile per thread, because they remain the same. But with FP16 accumulation, accumulator only needs 2x 32-bit registers per thread, instead of 4x (for each MMA tile). Well, we can do this.

```diff
- TypeAcc C_rmem[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};
+ TypeAcc C_rmem[WARP_M / MMA_M][WARP_N / MMA_N][sizeof(TypeAcc)] = {};  // 4 for FP32/INT32, 2 for FP16
```

Except using `TypeAcc` for `C_rmem` is no longer valid.

Although this goes against my initial remark against C++ template hell, I think this is still reasonable and readable. Perhaps an alternative would be to build the desired PTX MMA instruction string in Python, and wrap it around something like `mma()` function that can be injected via header code.

### Bonus: Support INT4 MMA

INT4 Tensor Cores were removed since Hopper, so there aren't many reasons to investigate INT4 MMA today.

## Closing remarks

Thanks to [msaroufim](https://x.com/marksaroufim) for bringing this to PyTorch. Though the `ctypes` wrapper of NVRTC is pretty straight-forward, and I can probably implement something similar by myself, it's very convenient when it's a PyTorch built-in, so I can focus on writing kernels. It also helps with distributing kernels
