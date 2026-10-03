+++
date = '2025-10-03T15:25:20+08:00'
title = 'Flash Attention faster than Speed-of-Light on 5090 with TMA'
url = 'fa-5090-part2'
+++
In my [previous post](/fa-5090), we only explored Ampere features like `cp.async` and `mma.sync`. It left out one of the most powerful tools on SM120: Tensor Memory Accelerator (TMA)! TMA has many benefits over `cp.async` (memory bandwidth is not one of them - you can still saturate memory bandwidth with the plain `cp.async`) that we will walk through later.

The new TMA-based kernel started out as an adaptation of my old work as I gained more experience with TMA in SM100's [tcgen05 kernels](/tcgen05) later. I was surprised at how such a simple change can bring a decent speedup to an already pretty optimized kernel.

TODO: headline table

TODO: where/how do we discuss the regression from v4->v5

CuteDSL 4.7.0 is used throughout this article. As CuteDSL is a fast developing library, future readers may find some of the code don't work anymore in newer versions.

## Step 0: Migration to CuteDSL

Recently I have been writing most of my kernels exclusively in CuteDSL. There is no perf reason behind it, I'm sure both can reach the same speed for most Tensor-Core-related kernels. There are many quality-of-life benefits over CUDA C++ that many have mentioned before, but to me the biggest advantage is meta-programming or (TODO: better / more precise wordings). For example, I built this concise `mma.sync` PTX wrapper for all dtypes.

```python
CUTE_TO_PTX_DTYPE = {
    cutlass.Float32: "f32",
    cutlass.BFloat16: "bf16",
    cutlass.Float16: "f16",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
    cutlass.Int8: "s8",
    cutlass.Uint8: "u8",
    cutlass.Int32: "s32",
    cutlass.Uint32: "u32",
}

@dsl_user_op
def mma_sync(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, *, loc=None, ip=None):
    a_ty = CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)

    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)
```

It builds a PTX string from the given arguments' dtypes. To achieve the same ergonomics in CUDA C++, I would need to hack around C++ templates hell, which is definitely not a pleasant experience.

Now let's focus on the major differences when transitioning to CuteDSL. My style of writing CuteDSL follows the same style of my CUDA C++ code, without using Cutlass-like APIs much. For TMA, refer to my previous [tcgen05](/tcgen05) post on TMA and [cutedsl-tma](/cutedsl-tma) on its usage in CuteDSL.

TODO: do we need a standalone section here?

TODO: warp-specialization kernel

Note: acutally we can keep this pretty simple. Discuss warp-specialization design. Do we need to talk about CuteDSL at all? Maybe not.

## Bonus: FP8 QK MMA

Previous I thought we need some kind of fine-grained scaling, like per-token scaling or maybe MXFP8 on newer hardware, to preserve attention accuracy. However, many attention backends in vLLM simply use a single static global scale for quantized KV cache, sometimes no scaling at all, and yet, accuracy seems acceptable. (TODO: link) This observation prompted me to explore using FP8 precision for QK MMA without any extra complicated scaling. This is simple because  both Q and K are K-major (head dim) for QK MMA, so from the perspective of `mma.sync` instruction, it still covers the same MMA tile in bytes, just different in the number of repeated tiles.
- Doing FP8 MMA for PV is much more complicated. FP32 MMA output layout (S) is compatible with BF16 MMA input layout (P), but not with FP8 MMA input layout. This requires layout shuffling when packing S into P, or V needs to be reshuffled first (which can be fused with the quantization op).

On 5090, we can also use the MXFP8 MMA instruction to unlock ~660 TFLOPS (at 600W) instead of FP8's 480 TFLOPS.