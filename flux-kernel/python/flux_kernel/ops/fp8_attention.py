"""Graph-safe query quantization and page-table expansion for FP8 attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _query_amax(Q, A, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                CHUNKS: tl.constexpr, T: tl.constexpr):
    head, chunk = tl.program_id(0), tl.program_id(1)
    tokens = chunk * T + tl.arange(0, T)
    channels = tl.arange(0, D)
    x = tl.load(Q + tokens[:, None] * H * D + head * D + channels[None, :],
                tokens[:, None] < N, 0).to(tl.float32)
    maximum = tl.max(tl.max(tl.abs(x), 1), 0)
    tl.store(A + head * CHUNKS + chunk, maximum)


@triton.jit
def _query_encode(Q, A, O, S, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  CHUNKS: tl.constexpr, C: tl.constexpr, T: tl.constexpr):
    head, chunk = tl.program_id(0), tl.program_id(1)
    parts = tl.arange(0, C)
    maximum = tl.max(tl.load(A + head * CHUNKS + parts, parts < CHUNKS, 0), 0)
    scale = tl.where(maximum > 0, maximum / 448.0, 1.0)
    if chunk == 0:
        tl.store(S + head, scale)
    tokens = chunk * T + tl.arange(0, T)
    channels = tl.arange(0, D)
    offsets = tokens[:, None] * H * D + head * D + channels[None, :]
    x = tl.load(Q + offsets, tokens[:, None] < N, 0).to(tl.float32)
    encoded = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    tl.store(O + offsets, encoded, tokens[:, None] < N)


def quantize_fp8_query(q):
    """NHD BF16/FP16 -> E4M3, with dynamic FP32 scales per query head.

    Reduction and encoding remain on device, including during graph replay.
    Scales cover this call's tokens; KV calibration scales remain independent.
    """
    if (q.ndim != 3 or not q.is_cuda or not q.is_contiguous()
            or q.dtype not in (torch.bfloat16, torch.float16)
            or q.shape[2] not in (64, 128) or q.shape[0] == 0):
        raise ValueError("FP8 query quantization requires nonempty contiguous CUDA NHD BF16/FP16, head_dim 64/128")
    n, h, d = q.shape
    chunks = triton.cdiv(n, 32)
    partial = torch.empty((h, chunks), device=q.device, dtype=torch.float32)
    scales = torch.empty(h, device=q.device, dtype=torch.float32)
    output = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    _query_amax[(h, chunks)](q, partial, n, h, d, chunks, 32)
    _query_encode[(h, chunks)](q, partial, output, scales, n, h, d, chunks,
                               triton.next_power_of_2(chunks), 32)
    return output, scales


@triton.jit
def _copy_page_prefixes(SRC, SRC_PTR, OWNERS, DST, DST_PTR, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    owner = tl.load(OWNERS + row)
    src_start = tl.load(SRC_PTR + owner)
    dst_start = tl.load(DST_PTR + row)
    count = tl.load(DST_PTR + row + 1) - dst_start
    page = tl.load(SRC + src_start + col, col < count, 0)
    tl.store(DST + dst_start + col, page, col < count)


def copy_page_prefixes(indices, indptr, owners, output, output_indptr, max_pages):
    _copy_page_prefixes[(owners.numel(), triton.cdiv(max_pages, 128))](
        indices, indptr, owners, output, output_indptr, 128,
    )


@triton.jit
def _refresh_fa3_schedule(PTR, LAST, ROWS, WORK_PTR, WORK_LEN,
                          PAGE: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = tl.load(ROWS + i, i < N, 0)
    start = tl.load(PTR + row)
    end = tl.load(PTR + row + 1)
    last = tl.load(LAST + row)
    tl.store(WORK_PTR + i, start, i < N)
    tl.store(WORK_LEN + i, (end - start - 1) * PAGE + last, i < N)


def refresh_fa3_schedule(indptr, last_page_len, schedule, page_size):
    rows, work_ptr, work_len = schedule
    _refresh_fa3_schedule[(triton.cdiv(rows.numel(), 256),)](
        indptr, last_page_len, rows, work_ptr, work_len, page_size, rows.numel(), 256,
    )
