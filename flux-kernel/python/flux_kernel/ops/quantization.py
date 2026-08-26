# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Quantization kernels owned by FluxServe."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _static_scaled_fp8_quant_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    input_numel: tl.constexpr,
    output_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < output_numel
    values = tl.load(input_ptr + offsets, mask=offsets < input_numel, other=0.0)
    scale = tl.load(scale_ptr).to(tl.float32)
    values = values.to(tl.float32) / scale
    values = tl.maximum(tl.minimum(values, 448.0), -448.0)
    tl.store(output_ptr + offsets, values, mask=output_mask)


def static_scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor,
    padded_rows: int | None = None,
) -> torch.Tensor:
    """Quantize a 2-D CUDA tensor with a precomputed per-tensor scale.

    ``scale`` follows the ModelOpt convention: dequantization is
    ``fp8_value * scale``. Optional row padding is written as FP8 zeros in the
    same kernel, which keeps the operation CUDA Graph capturable.
    """
    if not input.is_cuda or not scale.is_cuda:
        raise ValueError("static_scaled_fp8_quant requires CUDA tensors")
    if input.device != scale.device:
        raise ValueError("input and scale must be on the same CUDA device")
    if input.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "input must use bfloat16 or float16, " f"got {input.dtype}"
        )
    if input.ndim != 2:
        raise ValueError(f"input must be 2-D, got shape {tuple(input.shape)}")
    if not input.is_contiguous():
        input = input.contiguous()
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must be a single float32 value")

    rows, columns = input.shape
    output_rows = rows if padded_rows is None else int(padded_rows)
    if output_rows < rows:
        raise ValueError(
            f"padded_rows must be at least the input row count ({rows}), "
            f"got {output_rows}"
        )

    output = torch.empty(
        (output_rows, columns),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    output_numel = output.numel()
    if output_numel == 0:
        return output

    block_size = 256
    _static_scaled_fp8_quant_kernel[(triton.cdiv(output_numel, block_size),)](
        input,
        scale,
        output,
        input.numel(),
        output_numel,
        BLOCK_SIZE=block_size,
    )
    return output


__all__ = ["static_scaled_fp8_quant"]
