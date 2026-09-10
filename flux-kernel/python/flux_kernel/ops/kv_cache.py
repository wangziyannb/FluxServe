"""FP8 KV encoding/scatter and fixed-capacity graph metadata updates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_scatter(
    K,
    V,
    Slots,
    KC,
    VC,
    KH: tl.constexpr,
    KT: tl.constexpr,
    KD: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    CAPACITY: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    ENCODED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + token)
    valid = (col < HEADS * DIM) & (slot >= 0) & (slot < CAPACITY)
    k = tl.load(K + token * KT + (col // DIM) * KH + col % DIM * KD, valid, 0.0).to(
        tl.float32
    )
    v = tl.load(V + token * VT + (col // DIM) * VH + col % DIM * VD, valid, 0.0).to(
        tl.float32
    )
    if not ENCODED:
        k = tl.minimum(tl.maximum(k / KS, -448.0), 448.0)
        v = tl.minimum(tl.maximum(v / VS, -448.0), 448.0)
    offset = slot * HEADS * DIM + col
    tl.store(KC + offset, k, valid)
    tl.store(VC + offset, v, valid)


def quantize_scatter_kv(k, v, paged_kv, slot_mapping, k_scale, v_scale):
    """Write [heads, tokens, dim] into NHD pages, skipping negative slots.

    Encoding and scatter share one launch; no full float cache is materialized.
    FP8 inputs are already encoded and are copied without rescaling.
    """
    kc, vc = paged_kv
    if kc.dtype != torch.float8_e4m3fn or vc.dtype != kc.dtype:
        raise TypeError("quantize_scatter_kv requires FP8 E4M3 destination pages")
    if k.shape != v.shape or k.dtype != v.dtype or k.ndim != 3:
        raise ValueError(
            "K and V must have matching [heads, tokens, dim] shapes and dtype"
        )
    if not kc.is_contiguous() or not vc.is_contiguous() or kc.shape != vc.shape:
        raise ValueError("KV pages must be contiguous NHD tensors")
    if k.shape[0] != kc.shape[2] or k.shape[2] != kc.shape[3]:
        raise ValueError("KV input geometry does not match the page cache")
    slots = slot_mapping.reshape(-1)
    if slots.numel() != k.shape[1] or not slots.is_contiguous():
        raise ValueError(
            "slot_mapping must contain one contiguous slot per input token"
        )
    if k.shape[1] == 0:
        return
    _quantize_scatter[(k.shape[1], triton.cdiv(k.shape[0] * k.shape[2], 256))](
        k,
        v,
        slots,
        kc,
        vc,
        *k.stride(),
        *v.stride(),
        k.shape[0],
        k.shape[2],
        kc.shape[1],
        kc.shape[0] * kc.shape[1],
        float(k_scale),
        float(v_scale),
        k.dtype == torch.float8_e4m3fn,
        256,
    )


@triton.jit
def _decode_pages(
    Indptr, Indices, Out, PAGES: tl.constexpr, DUMMY: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Indptr + row)
    count = tl.load(Indptr + row + 1) - start
    page = tl.load(Indices + start + j, (j < count) & (j < PAGES), DUMMY)
    tl.store(Out + row * PAGES + j, page, j < PAGES)


@triton.jit
def _decode_mask(
    Indptr,
    Last,
    QOffsets,
    Out,
    MAX_KV: tl.constexpr,
    QLEN: tl.constexpr,
    PAGE: tl.constexpr,
    DIFFUSION_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    byte = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = (tl.load(Indptr + row + 1) - tl.load(Indptr + row) - 1) * PAGE + tl.load(
        Last + row
    )
    q_offset = tl.load(QOffsets + row)
    packed = tl.full((BLOCK,), 0, tl.int32)
    for bit in tl.static_range(8):
        linear = byte * 8 + bit
        qpos = q_offset + linear // MAX_KV
        kpos = linear % MAX_KV
        allowed = (kpos < length) & (qpos // DIFFUSION_BLOCK >= kpos // DIFFUSION_BLOCK)
        packed = packed | (allowed.to(tl.int32) << bit)
    tl.store(
        Out + row * (QLEN * MAX_KV // 8) + byte,
        packed.to(tl.uint8),
        byte < QLEN * MAX_KV // 8,
    )


def update_fp8_decode_metadata(
    indptr,
    indices,
    last,
    q_offsets,
    out_indices,
    out_mask,
    *,
    max_kv_len,
    q_len,
    page_size,
    block_length,
    dummy_page
):
    """Update existing graph buffers without planning, allocation or host reads."""
    if max_kv_len % page_size or (max_kv_len * q_len) % 8:
        raise ValueError("Graph capacity must be page and packed-mask aligned")
    rows = last.numel()
    pages = max_kv_len // page_size
    _decode_pages[(rows, triton.cdiv(pages, 256))](
        indptr,
        indices,
        out_indices,
        pages,
        dummy_page,
        256,
    )
    _decode_mask[(rows, triton.cdiv(q_len * max_kv_len // 8, 256))](
        indptr,
        last,
        q_offsets,
        out_mask,
        max_kv_len,
        q_len,
        page_size,
        block_length,
        256,
    )
