"""Hopper FA3 FP8 QK/PV with BF16 output and block-causal paged KV.

Each diffusion block becomes a virtual query sequence that shares a prefix of
the original request's pages. Non-causal attention within that prefix is exactly
the block-causal mask, without a quadratic custom mask or a BF16 KV copy.
Only page indices are duplicated. Uses the public FA3 paged API plus a guarded
adapter for the pinned 0.6.18 scheduler's copied per-work KV metadata.
"""

from itertools import accumulate

import torch

from fluxserve.backend.layers.kv_quantization import FP8_DTYPE

_OBSERVED_KERNELS = set()


def observed_block_kernels():
    return [dict(backend=b, query_dtype=q, kv_dtype=k) for b, q, k in sorted(_OBSERVED_KERNELS)]


def block_segments(q_lens, kv_lens, q_offsets, kv_offsets, block_length):
    """Return (owner, local query start, query count, visible KV count)."""
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    segments = []
    for row, (n, length, offset, kv_offset) in enumerate(
        zip(q_lens, kv_lens, q_offsets, kv_offsets, strict=True)
    ):
        if n <= 0 or length <= 0 or min(offset, kv_offset) < 0:
            raise ValueError("Native FP8 requires positive lengths and nonnegative offsets")
        start = 0
        while start < n:
            block_end = ((offset + start) // block_length + 1) * block_length
            count = min(n - start, block_end - offset - start)
            visible = min(length, block_end - kv_offset)
            if visible <= 0:
                raise ValueError("Native FP8 query block has no visible KV tokens")
            segments.append((row, start, count, visible))
            start += count
    if not segments:
        raise ValueError("Native FP8 requires a nonempty batch")
    return segments


def validate_native_fp8_config(config, model_config, device):
    """Fail before weight loading; never silently substitute BF16 attention."""
    compute = getattr(config, "attention_compute_dtype", "bf16")
    backend = getattr(config, "flashinfer_kernel_backend", "auto")
    if compute != "fp8" and backend == "auto":
        return
    if (getattr(model_config, "model_type", "") == "diffusion_gemma"
            or "DiffusionGemmaForBlockDiffusion" in (getattr(model_config, "architectures", ()) or ())):
        raise ValueError("Explicit FA2/FA3 block-paged attention supports LLaDA2 only")
    from fluxserve.backend.layers.kv_quantization import resolve_kv_dtype

    kv_dtype = resolve_kv_dtype(model_config, config.kv_cache_dtype)
    if compute == "fp8" and kv_dtype != "fp8_e4m3":
        raise ValueError("FP8 attention compute requires --kv-cache-dtype fp8_e4m3")
    if backend == "fa3" and kv_dtype == "fp8_e4m3" and compute != "fp8":
        raise ValueError("FA3 with FP8 KV requires FP8 attention compute; BF16 Q / FP8 KV is unsupported")
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise ValueError("Explicit block-paged attention currently requires Hopper (SM90), e.g. H100")
    if model_config.hidden_size // model_config.num_attention_heads not in (64, 128):
        raise ValueError("Native FP8 attention supports head dimensions 64 and 128")


def bind_fa3_schedule(wrapper, num_sequences):
    """Bind the pinned PrefillPlanSM90Info ABI, outside CUDA graph capture.

    FA3 ignores live CSR lengths during run(): its scheduler copies KV starts
    and lengths per work item at plan time. Query lengths, work assignment and
    launch geometry stay fixed; only those two KV fields must change on replay.
    Reject an unknown ABI instead of reading arbitrary workspace memory.
    """
    import flashinfer

    info = wrapper._plan_info
    if flashinfer.__version__ != "0.6.18" or len(info) != 9:
        raise RuntimeError("Native FP8 dynamic graphs require the validated FlashInfer 0.6.18 SM90 plan ABI")
    buf = wrapper._int_workspace_buffer
    sm_count = torch.cuda.get_device_properties(buf.device).multi_processor_count

    def view(index, count):
        offset = info[index]
        if offset < 0 or offset % 4 or offset + count * 4 > buf.numel():
            raise RuntimeError("Invalid FlashInfer SM90 schedule workspace offsets")
        return buf[offset:offset + count * 4].view(torch.int32)

    count = int(view(6, sm_count + 1)[-1].item())
    if count <= 0:
        raise RuntimeError("Native FP8 received an empty FA3 schedule")
    rows = view(7, count)
    if not bool(((rows >= 0) & (rows < num_sequences)).all().item()):
        raise RuntimeError("Invalid FlashInfer SM90 schedule request indices")
    return rows, view(2, count), view(4, count)


class BlockPagedWrapper:
    """Shared block/page geometry for FA2 BF16, FA3 BF16 and FA3 FP8."""

    block_paged = True

    def __init__(self, workspace, *, compute_dtype="fp8", backend="fa3"):
        if backend not in {"fa2", "fa3"} or compute_dtype not in {"bf16", "fp8"}:
            raise ValueError("Unsupported block-paged attention backend or compute dtype")
        if compute_dtype == "fp8" and backend != "fa3":
            raise ValueError("Native FP8 compute requires FA3")
        self.workspace = workspace
        self.compute_dtype = compute_dtype
        self.backend = backend
        self.data_dtype = FP8_DTYPE if compute_dtype == "fp8" else torch.bfloat16
        self._scales = {}
        self._schedule = None

    def plan(self, *, metadata, kv_indptr, kv_indices, last_page_len,
             page_size, block_length, num_qo_heads, num_kv_heads, head_dim,
             sm_scale, use_cuda_graph=False):
        import flashinfer

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Native FP8 attention must be planned before graph capture")
        if torch.cuda.get_device_capability(self.workspace.device)[0] != 9:
            raise ValueError("Native FP8 attention requires Hopper (SM90)")
        segments = block_segments(*metadata, block_length)
        self._schedule = None
        device = self.workspace.device
        tensor = lambda values: torch.tensor(values, dtype=torch.int32, device=device)
        self.owners = tensor([x[0] for x in segments])
        self.q_starts = tensor([x[1] for x in segments])
        self.qo_indptr = tensor([0, *accumulate(x[2] for x in segments)])
        pages = [(x[3] + page_size - 1) // page_size for x in segments]
        self.kv_indptr = tensor([0, *accumulate(pages)])
        self.kv_indices = torch.empty(sum(pages), dtype=torch.int32, device=device)
        self.last_page_len = tensor([(x[3] - 1) % page_size + 1 for x in segments])
        self.page_size, self.block_length = page_size, block_length
        self.max_pages = max(pages)
        self.num_kv_heads = num_kv_heads
        self.update_metadata(kv_indptr, kv_indices, last_page_len,
                             tensor(metadata[2]), tensor(metadata[3]))
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", backend=self.backend,
            use_cuda_graph=use_cuda_graph, qo_indptr_buf=self.qo_indptr,
            paged_kv_indptr_buf=self.kv_indptr,
            paged_kv_indices_buf=self.kv_indices,
            paged_kv_last_page_len_buf=self.last_page_len,
        )
        self.wrapper.plan(
            self.qo_indptr, self.kv_indptr, self.kv_indices, self.last_page_len,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            causal=False, q_data_type=self.data_dtype, kv_data_type=self.data_dtype,
            o_data_type=torch.bfloat16, sm_scale=sm_scale, disable_split_kv=True,
        )
        actual = (self.wrapper._backend, self.wrapper._cached_q_data_type,
                  self.wrapper._cached_kv_data_type)
        if actual != (self.backend, self.data_dtype, self.data_dtype):
            raise RuntimeError(f"FlashInfer backend/dtype fallback: {actual}")
        if use_cuda_graph and self.backend == "fa3":
            self._schedule = bind_fa3_schedule(self.wrapper, len(segments))

    def update_metadata(self, indptr, indices, last_page_len, q_offsets, kv_offsets):
        from flux_kernel.ops.fp8_attention import copy_page_prefixes

        owners = self.owners.long()
        lengths = (indptr[1:] - indptr[:-1] - 1) * self.page_size + last_page_len
        block_end = ((q_offsets[owners] + self.q_starts) // self.block_length + 1) * self.block_length
        visible = torch.minimum(lengths[owners], block_end - kv_offsets[owners])
        pages = (visible + self.page_size - 1) // self.page_size
        torch.cumsum(pages, dim=0, dtype=torch.int32, out=self.kv_indptr[1:])
        self.last_page_len.copy_((visible - 1) % self.page_size + 1)
        copy_page_prefixes(indices, indptr, self.owners, self.kv_indices,
                           self.kv_indptr, self.max_pages)
        if self._schedule is not None:
            from flux_kernel.ops.fp8_attention import refresh_fa3_schedule
            refresh_fa3_schedule(self.kv_indptr, self.last_page_len,
                                 self._schedule, self.page_size)

    def run(self, q, paged_kv_cache, *, k_scale=None, v_scale=None, enable_pdl=False):
        from flux_kernel.ops.fp8_attention import quantize_fp8_query

        if any(t.dtype != self.data_dtype for t in paged_kv_cache):
            raise TypeError(f"{self.backend} {self.compute_dtype} attention requires {self.data_dtype} KV pages")
        _OBSERVED_KERNELS.add((self.wrapper._backend, str(self.wrapper._cached_q_data_type),
                               str(self.wrapper._cached_kv_data_type)))
        if self.compute_dtype == "bf16":
            if q.dtype != torch.bfloat16 or k_scale not in (None, 1.0) or v_scale not in (None, 1.0):
                raise TypeError("BF16 attention requires BF16 Q and unscaled BF16 KV")
            return self.wrapper.run(q, paged_kv_cache, enable_pdl=enable_pdl)
        key = (k_scale, v_scale)
        scales = self._scales.get(key)
        if scales is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up all layer scales before native FP8 graph capture")
            scales = tuple(torch.full((self.num_kv_heads,), value, dtype=torch.float32,
                                      device=q.device) for value in key)
            self._scales[key] = scales
        encoded, q_scale = quantize_fp8_query(q)
        output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        # Positional scale tensors are consumed inside the FA3 kernel. Passing
        # q_scale as a keyword would instead multiply a host softmax scalar.
        return self.wrapper.run(encoded, paged_kv_cache, q_scale, *scales,
                                out=output, enable_pdl=enable_pdl)


class NativeFP8PagedWrapper(BlockPagedWrapper):
    """Compatibility name for the native FP8 path."""

    native_fp8 = True
