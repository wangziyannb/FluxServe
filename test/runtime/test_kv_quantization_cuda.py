"""Small CUDA numerical tests; no checkpoint or quality/performance claim."""

from dataclasses import replace
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.layers.kv_quantization import (
    FP8_DTYPE,
    KVQuantizationConfig,
    cache_bytes,
    encode_kv,
)
from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.forward import AttentionForward
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, ForwardMode
from fluxserve.backend.managers.kvcache.paged import PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def attention(layer=0):
    obj = AttentionForward(AttentionForwardConfig(layer, 4, 2, 128, 2, 128**-0.5))
    obj.kv_quantization = KVQuantizationConfig(
        "fp8_e4m3", ((0.013, 0.029), (0.037, 0.011)), "test"
    )
    return obj


def ref_encode(x, scale):
    return torch.clamp(x.float() / scale, -448, 448).to(FP8_DTYPE)


def ref_attention(q, k, v, scales, q_offset=0, block=64):
    k = (k.float() * scales[0]).bfloat16().repeat_interleave(2, 1)
    v = (v.float() * scales[1]).bfloat16().repeat_interleave(2, 1)
    mask = (torch.arange(q.shape[2], device=q.device) + q_offset)[
        :, None
    ] // block >= torch.arange(k.shape[2], device=q.device)[None, :] // block
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def test_fused_scatter_cross_page_rewrite_encoded_input_and_dummy():
    from flux_kernel.ops.kv_cache import quantize_scatter_kv

    torch.manual_seed(11)
    cache = PagedKVCache(
        num_layers=2,
        batch_size=2,
        local_kv_heads=2,
        max_length=192,
        head_dim=128,
        page_size=64,
        reserve_dummy_page=1,
        dtype=FP8_DTYPE,
        device="cuda",
    )
    slots = torch.tensor([63, 64, 190, -1, cache.dummy_page_id * 64], device="cuda")
    for layer, scales in enumerate(((0.013, 0.029), (0.037, 0.011))):
        kc, vc = cache.layer_paged_kv(layer)
        for repeat in range(3):
            k, v = [
                torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(
                    0, 1
                )
                for _ in range(2)
            ]
            if repeat == 2:
                k, v = ref_encode(k, scales[0]), ref_encode(v, scales[1])
            quantize_scatter_kv(k, v, (kc, vc), slots, *scales)
            for source, dest, scale in zip((k, v), (kc, vc), scales, strict=True):
                expected = (
                    source if source.dtype == FP8_DTYPE else ref_encode(source, scale)
                )
                for j in (0, 1, 2, 4):
                    slot = int(slots[j])
                    assert torch.equal(
                        cache_bytes(dest[slot // 64, slot % 64]),
                        cache_bytes(expected[:, j]),
                    )
        assert not kc[0, :63].float().any()


@pytest.mark.parametrize("backend", ["sdpa", "flex"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8_e4m3"])
def test_dense_graph_replay_rewrite_and_layer_scales(backend, dtype):
    torch.manual_seed(13)
    obj = attention(1)
    if dtype == "bf16":
        obj.kv_quantization = KVQuantizationConfig()
    q = torch.randn(1, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    if dtype == "fp8_e4m3":
        past = tuple(
            ref_encode(torch.randn(1, 2, 128, 128, device="cuda"), scale)
            for scale in (0.037, 0.011)
        )
    else:
        past = tuple(
            torch.randn(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
    mask = torch.ones(64, 128, device="cuda", dtype=torch.bool)
    if backend == "flex":
        from fluxserve.backend.layers.attention.base import _load_flex_attention

        _, create = _load_flex_attention()
        mask = create(
            lambda b, h, q, k: q + 64 >= (k // 64) * 64,
            B=None,
            H=None,
            Q_LEN=64,
            KV_LEN=128,
            device="cuda",
        )

    def run():
        return obj.forward(
            q, k, v, past_key_values=past, use_cache=True, attention_mask=mask
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, stored = run()
    for _ in range(3):
        q.normal_()
        k.normal_()
        v.normal_()
        for buf, scale in zip(past, (0.037, 0.011), strict=True):
            new = torch.randn(buf.shape, device="cuda", dtype=torch.bfloat16)
            buf.copy_(ref_encode(new, scale) if dtype == "fp8_e4m3" else new)
        expected, expected_cache = run()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-2)
        for actual, target in zip(stored, expected_cache, strict=True):
            assert torch.equal(cache_bytes(actual), cache_bytes(target))
    torch.cuda.synchronize()
    graph.reset()


def paged_batch(cache, q_len, kv_len, q_offset=0, prefill=True, seq=0):
    ptr, indices, last = cache.flashinfer_paged_metadata(
        seq_ids=torch.tensor([seq], device="cuda"),
        lengths=torch.tensor([kv_len], device="cuda"),
    )
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
        use_flashinfer_paged_prefill=prefill,
        use_flashinfer_paged_decode=not prefill,
        flashinfer_prefill_lens_cpu=(q_len,) if prefill else (),
        flashinfer_kv_lens_cpu=(kv_len,),
        flashinfer_kv_lens=torch.tensor([kv_len], dtype=torch.int32, device="cuda"),
        flashinfer_seq_ids=torch.tensor([seq], dtype=torch.long, device="cuda"),
        flashinfer_q_offsets_cpu=(q_offset,),
        flashinfer_q_offsets=torch.tensor([q_offset], dtype=torch.int32, device="cuda"),
        flashinfer_kv_offsets_cpu=(0,),
        flashinfer_kv_offsets=torch.zeros(1, dtype=torch.int32, device="cuda"),
        flashinfer_append_batch_indices=torch.zeros(q_len, dtype=torch.int32, device="cuda"),
        flashinfer_append_positions=torch.arange(q_offset, q_offset + q_len, dtype=torch.int32, device="cuda"),
        flashinfer_qo_indptr=torch.tensor([0, q_len], dtype=torch.int32, device="cuda"),
        flashinfer_qo_indptr_cpu=(0, q_len),
        flashinfer_kv_indptr=ptr,
        flashinfer_kv_indptr_cpu=tuple(ptr.cpu().tolist()),
        flashinfer_paged_kv_indices=indices,
        flashinfer_paged_kv_indices_cpu=tuple(indices.cpu().tolist()),
        flashinfer_paged_kv_last_page_len=last,
        flashinfer_paged_kv_last_page_len_cpu=tuple(last.cpu().tolist()),
        flashinfer_slot_mapping=cache.slot_mapping(
            torch.tensor([seq], device="cuda"),
            torch.arange(q_offset, q_offset + q_len, device="cuda").unsqueeze(0),
        ),
        flashinfer_block_length=64,
        flashinfer_page_size=64,
    )


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("dtype", ["bf16", "fp8_e4m3"])
def test_flashinfer_paged_partial_prompt_and_ragged(layer, dtype):
    pytest.importorskip("flashinfer")
    torch.manual_seed(17)
    obj = attention(layer)
    if dtype == "bf16":
        obj.kv_quantization = KVQuantizationConfig()
    scales = (1.0, 1.0) if dtype == "bf16" else obj.kv_quantization.scales[layer]
    cache = PagedKVCache(
        num_layers=2,
        batch_size=2,
        local_kv_heads=2,
        max_length=192,
        head_dim=128,
        page_size=64,
        dtype=obj.kv_quantization.torch_dtype,
        device="cuda",
    )
    for seq, q_len, kv_len, q_offset, prefill in (
        (0, 100, 100, 0, True),
        (0, 64, 128, 64, False),
        (1, 64, 64, 0, True),
        (0, 64, 128, 64, False),
    ):
        q = torch.randn(1, 4, q_len, 128, device="cuda", dtype=torch.bfloat16)
        k, v = [
            torch.randn(1, 2, q_len, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        batch = paged_batch(cache, q_len, kv_len, q_offset, prefill, seq)
        out, _ = obj.forward(
            q, k, v, past_key_values=cache.layer_paged_kv(layer), forward_batch=batch
        )
        dense = cache.materialize(
            seq_ids=torch.tensor([seq], device="cuda"), length=kv_len
        )[layer]
        ref = ref_attention(q, dense[0], dense[1], scales, q_offset)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        if dtype == "bf16" and prefill and q_len % 64:
            # BF16 ragged prefill accepts only the aligned prefix. The online
            # runner decodes the partial block separately; the 64-token case
            # below exercises its ragged prefill path. Paged accepts both.
            continue
        ragged = replace(
            batch,
            use_flashinfer_paged_prefill=False,
            use_flashinfer_paged_decode=False,
            use_flashinfer_prefill=prefill,
            use_flashinfer_decode=not prefill,
            flashinfer_kv_indptr=torch.tensor(
                [0, kv_len], dtype=torch.int32, device="cuda"
            ),
            flashinfer_kv_indptr_cpu=(0, kv_len),
        )
        result, _ = obj.forward(
            q,
            k,
            v,
            past_key_values=None if prefill else (dense[0], dense[1]),
            forward_batch=ragged,
        )
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("mode", ["decomposed", "padded"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8_e4m3"])
def test_flashinfer_runner_graph_request_switch_lengths_and_cleanup(mode, dtype, monkeypatch, request):
    pytest.importorskip("flashinfer")
    # The full mixed backend/KV/Graph suite triggers an illegal access in native
    # BF16 padded mode, although simpler mode-switch sequences pass. The exact
    # interaction is unresolved (see docs/experiments/flashinfer-runtime.md).
    # Serving fixes its configuration for each process. Test native padded in a
    # fresh process with real capture/replay and cleanup, avoiding contamination.
    child_flag = "FLUXSERVE_TEST_NATIVE_GRAPH_CHILD"
    if dtype == "bf16" and mode == "padded" and os.environ.get(child_flag) != "1":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             f"{__file__}::{request.node.name}"],
            env={**os.environ, child_flag: "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stdout
        return
    from fluxserve.backend.execution import flashinfer_cuda_graph_runner as graph_module
    from fluxserve.backend.layers.attention.fp8_flashinfer import (
        clear_fp8_flashinfer_states,
    )

    monkeypatch.setattr(graph_module, "get_attention_tp_size", lambda: 1)

    class ToyModel:
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=512,
        )

        def __init__(self):
            self.layers = [attention(0), attention(1)]
            if dtype == "bf16":
                for layer in self.layers:
                    layer.kv_quantization = KVQuantizationConfig()
            self.channels = torch.arange(512, device="cuda").float().view(1, 1, 512)

        def __call__(
            self, ids, positions, past, use_cache, attention_mask, forward_batch
        ):
            hidden = torch.sin(
                ids[..., None].float() * 0.01
                + positions[..., None] * 0.02
                + self.channels * 0.01
            ).bfloat16()
            for i, layer in enumerate(self.layers):
                q = hidden.view(ids.shape[0], ids.shape[1], 4, 128).transpose(1, 2)
                k, v = q[:, :2], q[:, 2:] * 0.7
                out, _ = layer.forward(
                    q,
                    k,
                    v,
                    past_key_values=past[i],
                    use_cache=False,
                    forward_batch=forward_batch,
                )
                hidden = out.transpose(1, 2).reshape_as(hidden) + hidden
            return hidden, None

    model = ToyModel()
    cache = PagedKVCache(
        num_layers=2,
        batch_size=2,
        local_kv_heads=2,
        max_length=256,
        head_dim=128,
        page_size=64,
        reserve_dummy_page=4,
        dtype=model.layers[0].kv_quantization.torch_dtype,
        device="cuda",
    )
    runner = SimpleNamespace(
        past_key_values=cache,
        model=SimpleNamespace(model=model),
        block_length=64,
        max_length=256,
        decoder=SimpleNamespace(mask_id=7),
        runner_config=SimpleNamespace(decode_cuda_graph_mode=mode),
        kv_quantization=model.layers[0].kv_quantization,
        tp_group=SimpleNamespace(barrier=lambda: None),
    )
    graphs = graph_module.FlashInferCudaGraphRunner(
        "cuda",
        (64, 128),
        num_layers=2,
        decode_capture_batch_sizes=(2,) if mode == "padded" else (1,),
    )
    past = [cache.layer_paged_kv(i) for i in range(2)]
    try:
        graphs.capture_decode_batch_sizes(runner)
        # Replay one prefill bucket with shorter, longer, and switched requests.
        for seq, length in ((0, 100), (1, 127), (0, 75)):
            ids = torch.arange(length, device="cuda").unsqueeze(0) + seq * 100
            positions = torch.arange(length, device="cuda").unsqueeze(0)
            batch = paged_batch(cache, length, length, prefill=True, seq=seq)
            model(ids, positions, past, False, None, batch)
            expected = cache.materialize(
                seq_ids=torch.tensor([seq], device="cuda"), length=length
            ).clone()
            cache_bytes(
                cache.data[
                    :,
                    :,
                    seq
                    * cache.pages_per_sequence : (seq + 1)
                    * cache.pages_per_sequence,
                ]
            ).zero_()
            graphs.replay(
                runner=runner,
                input_ids=ids,
                position_ids=positions,
                forward_batch=batch,
            )
            actual = cache.materialize(
                seq_ids=torch.tensor([seq], device="cuda"), length=length
            )
            # Compare dequantized cache values: intermediate BF16 attention can
            # straddle an FP8 rounding boundary in eager vs padded graph kernels.
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=1e-2, atol=1e-2
            )
        # Change lengths and page rows on the same captured decode graph.
        for seq, offset in ((0, 64), (1, 128), (0, 64), (1, 192)):
            ids = torch.arange(64, device="cuda").unsqueeze(0) + seq * 100
            positions = torch.arange(offset, offset + 64, device="cuda").unsqueeze(0)
            batch = paged_batch(cache, 64, offset + 64, offset, False, seq)
            expected, _ = model(ids, positions, past, False, None, batch)
            actual = graphs.replay_decode(
                runner=runner,
                input_ids=ids,
                position_ids=positions,
                forward_batch=batch,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        assert graphs.replay_count == 3 and graphs.decode_replay_count == 4
        assert graphs.capture_count == 1 and graphs.decode_capture_count == 1
    finally:
        graphs.shutdown_llada2(log=False)
        clear_fp8_flashinfer_states()
    assert (
        not graphs._graphs and not graphs._decode_graphs and graphs._workspace is None
    )


def test_dense_runner_fp8_commit_reorders_requests():
    from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner

    runner = BlockDiffusionRunner.__new__(BlockDiffusionRunner)
    runner.device = "cuda"
    runner.block_length = 4
    runner.past_key_values = torch.zeros(
        2, 2, 3, 2, 12, 8, dtype=FP8_DTYPE, device="cuda"
    )
    data = encode_kv(torch.randn(2, 2, 2, 2, 8, 8, device="cuda"), 0.037)
    runner._write_prefill_kv_cache(
        global_idx=torch.tensor(2, device="cuda"),
        local_idx=1,
        sample_len=5,
        prefilling_kv=data,
    )
    assert torch.equal(
        cache_bytes(runner.past_key_values[:, :, 2, :, :5]),
        cache_bytes(data[:, :, 1, :, :5]),
    )
    output = SimpleNamespace(past_key_values=list(data.flatten(0, 1).unbind()))
    runner._update_finished_kv_cache(
        output,
        torch.tensor([2, 0], device="cuda"),
        torch.tensor([4, 0, 8], device="cuda"),
        torch.tensor([True, True], device="cuda"),
        8,
        2,
    )
    assert torch.equal(
        cache_bytes(runner.past_key_values[:, :, 2, :, 8:12]),
        cache_bytes(data[:, :, 0, :, 4:8]),
    )
    assert torch.equal(
        cache_bytes(runner.past_key_values[:, :, 0, :, 4:8]),
        cache_bytes(data[:, :, 1, :, 4:8]),
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp8_e4m3"])
def test_sdpa_prefill_graph(dtype):
    obj = attention(1)
    if dtype == "bf16":
        obj.kv_quantization = KVQuantizationConfig()
    q = torch.randn(1, 4, 100, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(1, 2, 100, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    pos = torch.arange(100, device="cuda")
    mask = pos[:, None] // 64 >= pos[None, :] // 64

    def run():
        return obj.forward(q, k, v, use_cache=True, attention_mask=mask)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, stored = run()
    for _ in range(2):
        q.normal_()
        k.normal_()
        v.normal_()
        expected, expected_cache = run()
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        for left, right in zip(stored, expected_cache, strict=True):
            assert torch.equal(cache_bytes(left), cache_bytes(right))
    torch.cuda.synchronize()
    graph.reset()
