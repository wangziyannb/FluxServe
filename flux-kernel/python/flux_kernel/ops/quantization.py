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


_NVFP4_BLOCK_SIZE = 16


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


@triton.jit
def _encode_e2m1(values):
    magnitude = tl.abs(values)
    code = (
        (magnitude > 0.25).to(tl.uint8)
        + (magnitude > 0.75).to(tl.uint8)
        + (magnitude > 1.25).to(tl.uint8)
        + (magnitude > 1.75).to(tl.uint8)
        + (magnitude > 2.5).to(tl.uint8)
        + (magnitude > 3.5).to(tl.uint8)
        + (magnitude > 5.0).to(tl.uint8)
    )
    # ModelOpt uses round-to-nearest-even at the three odd-code midpoints.
    code += (
        (magnitude == 0.75).to(tl.uint8)
        + (magnitude == 1.75).to(tl.uint8)
        + (magnitude == 3.5).to(tl.uint8)
    )
    sign = (values < 0).to(tl.uint8) << 3
    return code | sign


@triton.jit
def _static_scaled_nvfp4_quant_kernel(
    input_ptr,
    global_scale_ptr,
    output_ptr,
    output_scale_ptr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    scale_columns: tl.constexpr,
    output_scale_columns: tl.constexpr,
    scale_column_blocks: tl.constexpr,
    BLOCKED_SCALE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0)
    valid_row = row < rows
    first_block = tl.program_id(1) * BLOCKS_PER_PROGRAM
    block_offsets = first_block + tl.arange(0, BLOCKS_PER_PROGRAM)[:, None]
    pair_offsets = tl.arange(0, 8)[None, :]
    valid_scale_blocks = (
        first_block + tl.arange(0, BLOCKS_PER_PROGRAM) < scale_columns
    )
    valid_blocks = valid_scale_blocks[:, None]

    element_offsets = block_offsets * 16 + pair_offsets * 2
    row_start = row * columns
    even = tl.load(
        input_ptr + row_start + element_offsets,
        mask=valid_row & valid_blocks,
        other=0.0,
    ).to(tl.float32)
    odd = tl.load(
        input_ptr + row_start + element_offsets + 1,
        mask=valid_row & valid_blocks,
        other=0.0,
    ).to(tl.float32)

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    block_amax = tl.maximum(
        tl.max(tl.abs(even), axis=1),
        tl.max(tl.abs(odd), axis=1),
    )
    block_scale = block_amax / (6.0 * global_scale)
    block_scale = tl.where(block_amax == 0.0, 1.0, block_scale)
    block_scale = tl.maximum(
        tl.minimum(block_scale, 448.0), 0.001953125
    )
    block_scale = block_scale.to(tl.float8e4nv).to(tl.float32)
    stored_block_scale = tl.where(
        valid_row & valid_scale_blocks, block_scale, 0.0
    )

    dequant_scale = block_scale[:, None] * global_scale
    even = tl.maximum(tl.minimum(even / dequant_scale, 6.0), -6.0)
    odd = tl.maximum(tl.minimum(odd / dequant_scale, 6.0), -6.0)
    packed = _encode_e2m1(even) | (_encode_e2m1(odd) << 4)
    packed_offsets = (
        row * (columns // 2)
        + block_offsets * 8
        + pair_offsets
    )
    tl.store(output_ptr + packed_offsets, packed, mask=valid_row & valid_blocks)

    scale_offsets = first_block + tl.arange(0, BLOCKS_PER_PROGRAM)
    scale_mask = scale_offsets < output_scale_columns
    if BLOCKED_SCALE:
        row_block = row // 128
        row_in_block = row % 128
        scale_block = scale_offsets // 4
        scale_in_block = scale_offsets % 4
        scale_output_offsets = (
            (row_block * scale_column_blocks + scale_block) * 512
            + (row_in_block % 32) * 16
            + (row_in_block // 32) * 4
            + scale_in_block
        )
    else:
        scale_output_offsets = row * scale_columns + scale_offsets
    tl.store(
        output_scale_ptr + scale_output_offsets,
        stored_block_scale,
        mask=scale_mask,
    )


def static_scaled_nvfp4_quant(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    blocked_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor to packed NVFP4 with 16-element FP8 scales.

    ``global_scale`` is the serialized ModelOpt scale. Dequantization is
    ``e2m1_value * block_scale * global_scale``. Dense cuBLASLt GEMMs require
    the block scales in their interleaved layout. Natural row-major scales are
    also useful when preparing fused MoE weights.
    """
    if not input.is_cuda or not global_scale.is_cuda:
        raise ValueError("static_scaled_nvfp4_quant requires CUDA tensors")
    if input.device != global_scale.device:
        raise ValueError("input and global_scale must be on the same CUDA device")
    if input.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "input must use bfloat16 or float16, " f"got {input.dtype}"
        )
    if input.ndim != 2:
        raise ValueError(f"input must be 2-D, got shape {tuple(input.shape)}")
    if input.shape[1] % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError("input columns must be divisible by 16 for NVFP4")
    if not input.is_contiguous():
        input = input.contiguous()
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise ValueError("global_scale must be a single float32 value")

    rows, columns = input.shape
    packed = torch.empty(
        (rows, columns // 2), dtype=torch.uint8, device=input.device
    )
    scale_columns = columns // _NVFP4_BLOCK_SIZE
    scale_column_blocks = triton.cdiv(scale_columns, 4)
    if blocked_scale:
        scale_rows = triton.cdiv(rows, 128) * 128
        output_scale_columns = scale_column_blocks * 4
        scale_numel = scale_rows * output_scale_columns
        block_scale = torch.empty(
            scale_numel, dtype=torch.float8_e4m3fn, device=input.device
        )
    else:
        scale_rows = rows
        output_scale_columns = scale_columns
        block_scale = torch.empty(
            (rows, scale_columns),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )

    blocks_per_program = 8
    _static_scaled_nvfp4_quant_kernel[
        (scale_rows, triton.cdiv(output_scale_columns, blocks_per_program))
    ](
        input,
        global_scale,
        packed,
        block_scale,
        rows=rows,
        columns=columns,
        scale_columns=scale_columns,
        output_scale_columns=output_scale_columns,
        scale_column_blocks=scale_column_blocks,
        BLOCKED_SCALE=blocked_scale,
        BLOCKS_PER_PROGRAM=blocks_per_program,
    )
    return packed.view(torch.float4_e2m1fn_x2), block_scale


def interleave_nvfp4_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert row-major NVFP4 scales to the cuBLASLt blocked layout."""
    if not scale.is_cuda:
        raise ValueError("interleave_nvfp4_block_scale requires a CUDA tensor")
    if scale.ndim != 2 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError("scale must be a 2-D float8_e4m3fn tensor")

    rows, columns = scale.shape
    padded_rows = triton.cdiv(rows, 128) * 128
    padded_columns = triton.cdiv(columns, 4) * 4
    if (rows, columns) != (padded_rows, padded_columns):
        padded = torch.zeros(
            (padded_rows, padded_columns), dtype=scale.dtype, device=scale.device
        )
        padded[:rows, :columns] = scale
        scale = padded
    return (
        scale.view(padded_rows // 128, 128, padded_columns // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
        .contiguous()
    )


__all__ = [
    "interleave_nvfp4_block_scale",
    "static_scaled_fp8_quant",
    "static_scaled_nvfp4_quant",
]
