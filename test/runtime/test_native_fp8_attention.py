"""Independent mask and configuration contracts, without CUDA."""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.layers.attention.native_fp8 import block_segments, validate_native_fp8_config


@pytest.mark.parametrize("q_offset,kv_offset", [(0, 0), (37, 5), (130, 19)])
def test_virtual_sequences_equal_block_causal_mask(q_offset, kv_offset):
    segments = block_segments((171,), (230,), (q_offset,), (kv_offset,), 64)
    represented = torch.zeros(171, 230, dtype=torch.bool)
    for owner, start, count, visible in segments:
        assert owner == 0
        represented[start:start + count, :visible] = True
    expected = ((torch.arange(171) + q_offset)[:, None] // 64
                >= (torch.arange(230) + kv_offset)[None, :] // 64)
    assert torch.equal(represented, expected)


def test_native_fp8_rejects_unsupported_config_before_loading(monkeypatch):
    with pytest.raises(ValueError, match="paged"):
        RunnerConfig(attention_compute_dtype="fp8")
    config = RunnerConfig(attention_compute_dtype="fp8", attention_backend="flashinfer",
                          flashinfer_prefill_mode="paged", flashinfer_cache_mode="paged",
                          kv_cache_layout="paged", kv_cache_dtype="bf16")
    model = SimpleNamespace(hidden_size=4096, num_attention_heads=32)
    with pytest.raises(ValueError, match="kv-cache-dtype"):
        validate_native_fp8_config(config, model, "cuda")
    config.kv_cache_dtype = "fp8"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(ValueError, match="Hopper"):
        validate_native_fp8_config(config, model, "cuda")


def test_native_fp8_cannot_fall_back_to_dense():
    from fluxserve.backend.layers.attention.base import AttentionForwardConfig
    from fluxserve.backend.layers.attention.forward import AttentionForward

    layer = AttentionForward(AttentionForwardConfig(0, 4, 2, 128, 2, 128**-0.5))
    layer.attention_compute_dtype = "fp8"
    q = torch.zeros(1, 4, 64, 128)
    with pytest.raises(ValueError, match="no dense fallback"):
        layer.forward(q, q[:, :2], q[:, :2])


@pytest.mark.parametrize("version,plan", [("0.6.19", [0]*9), ("0.6.18", [0]*10)])
def test_unknown_fa3_schedule_abi_is_rejected(monkeypatch, version, plan):
    import sys
    from fluxserve.backend.layers.attention.native_fp8 import bind_fa3_schedule

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__=version))
    with pytest.raises(RuntimeError, match="SM90 plan ABI"):
        bind_fa3_schedule(SimpleNamespace(_plan_info=plan), 2)


def test_explicit_fa3_rejects_bf16_query_with_fp8_kv():
    config = RunnerConfig(attention_backend='flashinfer', flashinfer_prefill_mode='paged',
                          flashinfer_cache_mode='paged', kv_cache_layout='paged',
                          kv_cache_dtype='fp8', flashinfer_kernel_backend='fa3')
    with pytest.raises(ValueError, match='BF16 Q / FP8 KV'):
        validate_native_fp8_config(config, SimpleNamespace(hidden_size=4096, num_attention_heads=32), 'cuda')
    with pytest.raises(ValueError, match='requires FA3'):
        RunnerConfig(attention_compute_dtype='fp8', flashinfer_kernel_backend='fa2')
