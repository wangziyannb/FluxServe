# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import Any, Optional

import torch

from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.layers.attention.base import (
    AttentionForwardConfig,
    DenseAttention,
)
from fluxserve.backend.layers.attention.flashinfer import (
    FlashInferPagedAttention,
    FlashInferPagedPrefillAttention,
    FlashInferRaggedAttention,
    FlashInferRaggedPrefillAttention,
)
from fluxserve.backend.layers.kv_quantization import (
    KVQuantizationConfig, cache_bytes, decode_kv, encode_kv,
)


class AttentionForward:
    """Routes one attention call to ragged FlashInfer or dense attention."""

    def __init__(self, config: AttentionForwardConfig):
        self.config = config
        self.kv_quantization = KVQuantizationConfig()
        self.kv_observer = None
        self.attention_compute_dtype = "bf16"
        self.flashinfer_kernel_backend = "auto"
        self.flashinfer_fp8 = None
        self.dense = DenseAttention(config)
        self.flashinfer_ragged_prefill = FlashInferRaggedPrefillAttention(config)
        self.flashinfer_paged_prefill = FlashInferPagedPrefillAttention(config)
        self.flashinfer_paged = FlashInferPagedAttention(config)
        self.flashinfer_ragged = FlashInferRaggedAttention(config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        past_key_values: Any = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if self.flashinfer_kernel_backend != "auto" and (
            forward_batch is None
            or not (forward_batch.use_flashinfer_paged_prefill or forward_batch.use_flashinfer_paged_decode)
            or attention_mask is not None
        ):
            raise ValueError("Explicit FA2/FA3 requires paged block-causal metadata; no dense fallback")
        if self.attention_compute_dtype == "fp8" and (
            self.kv_quantization.dtype != "fp8_e4m3"
            or forward_batch is None
            or not (forward_batch.use_flashinfer_paged_prefill or forward_batch.use_flashinfer_paged_decode)
            or attention_mask is not None
        ):
            raise ValueError("Native FP8 attention requires paged FP8 KV and block-causal metadata; no dense fallback")
        if self.kv_observer is not None:
            # LLaDA2 calls here after QK norm and RoPE, before any cache splice.
            self.kv_observer.observe(self.config.layer_id, k, v)
        if self.kv_quantization.dtype == "fp8_e4m3":
            return self._forward_fp8(
                q, k, v, past_key_values, use_cache, attention_mask, forward_batch,
            )
        if self.flashinfer_kernel_backend != "auto":
            if self.flashinfer_fp8 is None:
                from fluxserve.backend.layers.attention.fp8_flashinfer import FP8FlashInferAttention
                self.flashinfer_fp8 = FP8FlashInferAttention(
                    self.config, "bf16", kv_dtype=torch.bfloat16,
                    kernel_backend=self.flashinfer_kernel_backend,
                )
            out = self.flashinfer_fp8.forward_paged(
                q, k, v, past_key_values, forward_batch, None, None,
            )
            return out, (k, v) if use_cache else None
        if self.flashinfer_paged_prefill.can_run(
            q,
            k,
            past_key_values,
            attention_mask,
            forward_batch,
        ):
            present_key_values = (k, v) if use_cache else None
            return (
                self.flashinfer_paged_prefill.forward(
                    q,
                    k,
                    v,
                    past_key_values,
                    forward_batch,
                ),
                present_key_values,
            )
        if (
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_paged_prefill", False)
        ):
            raise RuntimeError(
                "FlashInfer paged prefill was requested but cannot run."
            )

        if self.flashinfer_ragged_prefill.can_run(q, k, attention_mask, forward_batch):
            present_key_values = (k, v) if use_cache else None
            return (
                self.flashinfer_ragged_prefill.forward(q, k, v, forward_batch),
                present_key_values,
            )
        if (
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_prefill", False)
        ):
            raise RuntimeError(
                "FlashInfer ragged prefill was requested but cannot run."
            )

        if self.flashinfer_paged.can_run(q, past_key_values, attention_mask, forward_batch):
            present_key_values = (k, v) if use_cache else None
            return (
                self.flashinfer_paged.forward(q, k, v, past_key_values, forward_batch),
                present_key_values,
            )
        if (
            forward_batch is not None
            and getattr(forward_batch, "use_flashinfer_paged_decode", False)
        ):
            raise RuntimeError("FlashInfer paged decode was requested but cannot run.")

        k, v = self.dense.splice_cache(k, v, past_key_values)
        present_key_values = (k, v) if use_cache else None

        if self.flashinfer_ragged.can_run(
            q,
            k,
            past_key_values,
            attention_mask,
            forward_batch,
        ):
            return (
                self.flashinfer_ragged.forward(q, k, v, forward_batch),
                present_key_values,
            )

        return self.dense.forward(q, k, v, attention_mask), present_key_values

    def _forward_fp8(self, q, k, v, past, use_cache, mask, batch):
        k_scale, v_scale = self.kv_quantization.scales[self.config.layer_id]
        paged = batch is not None and (
            batch.use_flashinfer_paged_prefill or batch.use_flashinfer_paged_decode
        )
        if paged:
            # The paged path fuses encoding with the scatter into persistent KV.
            if self.flashinfer_fp8 is None:
                from fluxserve.backend.layers.attention.fp8_flashinfer import FP8FlashInferAttention
                self.flashinfer_fp8 = FP8FlashInferAttention(
                    self.config, self.attention_compute_dtype,
                    kernel_backend=self.flashinfer_kernel_backend,
                )
            out = self.flashinfer_fp8.forward_paged(q, k, v, past, batch, k_scale, v_scale)
            return out, (encode_kv(k, k_scale), encode_kv(v, v_scale)) if use_cache else None

        k, v = encode_kv(k, k_scale), encode_kv(v, v_scale)
        if past is not None:
            if past[0].dtype != k.dtype or past[1].dtype != v.dtype:
                raise TypeError("FP8 attention requires an encoded FP8 cache.")
            k, v = self.dense.splice_cache(
                cache_bytes(k), cache_bytes(v), (cache_bytes(past[0]), cache_bytes(past[1])),
            )
            k, v = k.view(self.kv_quantization.torch_dtype), v.view(self.kv_quantization.torch_dtype)
        present = (k, v) if use_cache else None
        ragged = batch is not None and (batch.use_flashinfer_prefill or batch.use_flashinfer_decode)
        if ragged:
            if self.flashinfer_fp8 is None:
                from fluxserve.backend.layers.attention.fp8_flashinfer import FP8FlashInferAttention
                self.flashinfer_fp8 = FP8FlashInferAttention(self.config)
            out = self.flashinfer_fp8.forward_ragged(q, k, v, batch, k_scale, v_scale)
        else:
            # Only this layer is materialized, including the active diffusion block.
            out = self.dense.forward(q, decode_kv(k, k_scale), decode_kv(v, v_scale), mask)
        return out, present
