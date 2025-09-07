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

I explored putting the kernel in a standalone file `.cu` instead. When we need to compile the kernel, Python can open and read the files. It works pretty well! The kernel writing experience is exactly the same as the usual.

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

So far we have discussed about the engineering side of compiling kernels with NVRTC, but have yet touched on how to actually write a matmul kernel that can be adapted to various dtypes.

## Structure of an Ampere matmul

Formally, our kernel computes `C = A @ B`, where A, B, C have shapes [M, K], [K, N], and [M, N] respectively. To keep the code simple, we assume both A and B are K-major.

```python

```

## Closing remarks

Thanks to [msaroufim](https://x.com/marksaroufim) for bringing this to PyTorch. Though the `ctypes` wrapper of NVRTC is pretty straight-forward, and I can probably implement something similar by myself, it's very convenient when it's a PyTorch built-in, so I can focus on writing kernels. It also helps with distributing kernels
