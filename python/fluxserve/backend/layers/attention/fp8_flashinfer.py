"""FlashInfer adapters for scaled FP8 KV with BF16 or native FP8 compute.

No dependency on flashinfer-dllm's same-dtype block-extend kernels. Workspaces
are shared across layers; graph calls use wrappers planned by the graph runner.
"""

from __future__ import annotations

import torch
from dataclasses import replace

from fluxserve.backend.layers.kv_quantization import FP8_DTYPE


def block_causal_mask(q_lens, kv_lens, q_offsets, kv_offsets, block_length, device):
    masks = []
    for q_len, kv_len, q_offset, kv_offset in zip(
        q_lens, kv_lens, q_offsets, kv_offsets, strict=True
    ):
        q_pos = torch.arange(q_len, device=device) + q_offset
        k_pos = torch.arange(kv_len, device=device) + kv_offset
        masks.append(
            (q_pos[:, None] // block_length >= k_pos[None, :] // block_length).flatten()
        )
    return torch.cat(masks)


class _FA2State:
    def __init__(self, device, paged, native=False, block_backend=None):
        import flashinfer

        self.workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        cls = (
            flashinfer.BatchPrefillWithPagedKVCacheWrapper
            if paged
            else flashinfer.BatchPrefillWithRaggedKVCacheWrapper
        )
        if block_backend:
            from fluxserve.backend.layers.attention.native_fp8 import BlockPagedWrapper
            self.wrapper = BlockPagedWrapper(self.workspace, backend=block_backend,
                                            compute_dtype="fp8" if native else "bf16")
        else:
            self.wrapper = cls(self.workspace, kv_layout="NHD", backend="fa2")
        self.key = None


_STATES = {}


def clear_fp8_flashinfer_states():
    _STATES.clear()


def _state(device, paged, native=False, block_backend=None):
    key = (str(device), paged, native, block_backend)
    if key not in _STATES:
        _STATES[key] = _FA2State(device, paged, native, block_backend)
    return _STATES[key]


def _pack(tensor, lengths):
    return torch.cat(
        [
            tensor[i, :, :length].transpose(0, 1).contiguous()
            for i, length in enumerate(lengths)
        ]
    )


def _unpack(output, q, lengths):
    padded = q.new_zeros(q.shape[0], q.shape[2], q.shape[1], q.shape[3])
    offset = 0
    for i, length in enumerate(lengths):
        padded[i, :length] = output[offset : offset + length]
        offset += length
    return padded.transpose(1, 2).contiguous()


class FP8FlashInferAttention:
    def __init__(self, config, compute_dtype="bf16", *, kv_dtype=FP8_DTYPE, kernel_backend="auto"):
        self.config = config
        self.native = compute_dtype == "fp8"
        self.kv_dtype = kv_dtype
        self.block_backend = ("fa3" if kernel_backend == "auto" else kernel_backend) if (
            self.native or kv_dtype == torch.bfloat16
        ) else None

    def _metadata(self, q, batch, prefill):
        q_lens = (
            tuple(batch.flashinfer_prefill_lens_cpu)
            if prefill
            else (q.shape[2],) * q.shape[0]
        )
        kv_lens = tuple(batch.flashinfer_kv_lens_cpu) or q_lens
        offsets = tuple(batch.flashinfer_q_offsets_cpu) or tuple(
            k - n for k, n in zip(kv_lens, q_lens, strict=True)
        )
        kv_offsets = tuple(batch.flashinfer_kv_offsets_cpu) or (0,) * len(q_lens)
        if not (len(q_lens) == len(kv_lens) == len(offsets) == q.shape[0]):
            raise ValueError("FP8 FlashInfer batch metadata does not match Q")
        return q_lens, kv_lens, offsets, kv_offsets

    def _plan(self, q, batch, paged, metadata, page_size=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FP8 FlashInfer planning must finish before CUDA graph capture"
            )
        q_lens, kv_lens, offsets, kv_offsets = metadata
        block = int(batch.flashinfer_block_length or q.shape[2])
        state = _state(q.device, paged, self.native, self.block_backend)
        key = (
            metadata,
            block,
            page_size,
            q.dtype,
            self.kv_dtype,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.scale,
            tuple(batch.flashinfer_kv_indptr_cpu),
            tuple(batch.flashinfer_paged_kv_indices_cpu),
            tuple(batch.flashinfer_paged_kv_last_page_len_cpu),
        )
        # Scheduler page tables can change even with identical lengths. When CPU
        # copies are absent, replan rather than reusing another request's pages.
        if state.key == key and (not paged or batch.flashinfer_paged_kv_indices_cpu):
            return state.wrapper
        if self.block_backend:
            state.wrapper.plan(
                metadata=metadata, kv_indptr=batch.flashinfer_kv_indptr,
                kv_indices=batch.flashinfer_paged_kv_indices,
                last_page_len=batch.flashinfer_paged_kv_last_page_len,
                page_size=page_size, block_length=block,
                num_qo_heads=self.config.num_heads, num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim, sm_scale=self.config.scale,
            )
            state.key = key
            return state.wrapper
        mask = block_causal_mask(q_lens, kv_lens, offsets, kv_offsets, block, q.device)
        kwargs = dict(
            num_qo_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim_qk=self.config.head_dim,
            custom_mask=mask,
            causal=False,
            q_data_type=q.dtype,
            kv_data_type=FP8_DTYPE,
            sm_scale=self.config.scale,
        )
        if paged:
            state.wrapper.plan(
                batch.flashinfer_qo_indptr,
                batch.flashinfer_kv_indptr,
                batch.flashinfer_paged_kv_indices,
                batch.flashinfer_paged_kv_last_page_len,
                page_size=page_size,
                disable_split_kv=True,
                **kwargs,
            )
        else:
            state.wrapper.plan(
                batch.flashinfer_qo_indptr, batch.flashinfer_kv_indptr, **kwargs
            )
        state.key = key
        return state.wrapper

    def forward_paged(self, q, k, v, past, batch, k_scale, v_scale):
        from flux_kernel.ops.kv_cache import quantize_scatter_kv

        if past is None or any(t.dtype != self.kv_dtype for t in past):
            raise TypeError(f"Paged attention requires {self.kv_dtype} KV pages")
        if batch.flashinfer_slot_mapping is None:
            raise ValueError("FP8 paged attention requires slot_mapping")
        metadata = self._metadata(q, batch, batch.use_flashinfer_paged_prefill)
        lengths = metadata[0]
        slots = batch.flashinfer_slot_mapping.reshape(q.shape[0], -1)
        for i, length in enumerate(lengths):
            if self.kv_dtype == FP8_DTYPE:
                quantize_scatter_kv(
                    k[i, :, :length], v[i, :, :length], past,
                    slots[i, :length], k_scale, v_scale,
                )
            else:
                # Identical BF16 cache writes for the explicit FA2/FA3 comparison.
                page_size = int(past[0].shape[1])
                row_slots = slots[i, :length].long()
                pages, offsets = row_slots // page_size, row_slots % page_size
                past[0][pages, offsets] = k[i, :, :length].transpose(0, 1)
                past[1][pages, offsets] = v[i, :, :length].transpose(0, 1)
        packed_q = _pack(q, lengths)
        if batch.flashinfer_full_prefill_graph:
            output = batch.flashinfer_cuda_graph_runner.run_attention(
                q=packed_q,
                paged_kv_cache=past,
                num_q_heads=self.config.num_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                sm_scale=self.config.scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        elif batch.flashinfer_full_decode_graph:
            return batch.flashinfer_cuda_graph_runner.run_decode_attention(
                packed_q,
                past,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            wrapper = self._plan(q, batch, True, metadata, int(past[0].shape[1]))
            output = wrapper.run(
                packed_q, past, k_scale=k_scale, v_scale=v_scale, enable_pdl=False
            )
        return _unpack(output, q, lengths)

    def forward_ragged(self, q, k, v, batch, k_scale, v_scale):
        prefill = batch.use_flashinfer_prefill
        metadata = self._metadata(q, batch, prefill)
        q_lens, kv_lens, _, _ = metadata
        if prefill:
            packed_k, packed_v = _pack(k, kv_lens), _pack(v, kv_lens)
        else:

            def pack_cache(tensor):
                # Dense runners place the rewritten diffusion block at the end
                # of a capacity bucket; remove the gap before ragged attention.
                parts = []
                for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
                    if kv_len < q_len or kv_len > tensor.shape[2]:
                        raise ValueError("Invalid ragged KV length")
                    parts.append(
                        torch.cat(
                            (tensor[i, :, : kv_len - q_len], tensor[i, :, -q_len:]),
                            dim=1,
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )
                return torch.cat(parts)

            packed_k, packed_v = pack_cache(k), pack_cache(v)
        # The pinned FlashInfer ragged FA2 run accepts k_scale/v_scale but
        # silently ignores them for BF16 Q. A ragged buffer is also an NHD
        # page_size=1 cache (zero-copy views), so use the public paged wrapper
        # whose scale contract is implemented for mixed BF16/FP8.
        page_indices = torch.arange(
            packed_k.shape[0], dtype=torch.int32, device=q.device
        )
        page_batch = replace(
            batch,
            flashinfer_paged_kv_indices=page_indices,
            flashinfer_paged_kv_indices_cpu=tuple(range(packed_k.shape[0])),
            flashinfer_paged_kv_last_page_len=torch.ones(
                len(q_lens), dtype=torch.int32, device=q.device
            ),
            flashinfer_paged_kv_last_page_len_cpu=(1,) * len(q_lens),
        )
        wrapper = self._plan(q, page_batch, True, metadata, page_size=1)
        output = wrapper.run(
            _pack(q, q_lens),
            (packed_k.unsqueeze(1), packed_v.unsqueeze(1)),
            k_scale=k_scale,
            v_scale=v_scale,
            enable_pdl=False,
        )
        return _unpack(output, q, q_lens)
