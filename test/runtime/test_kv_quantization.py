import copy
import json
from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.layers.kv_quantization import (
    FP8_DTYPE,
    KVCalibrationObserver,
    KVQuantizationConfig,
    cache_bytes,
    checkpoint_kv_dtype,
    decode_kv,
    encode_kv,
    extract_checkpoint_kv_scales,
    model_identity,
    resolve_kv_dtype,
)


def model_config(**kwargs):
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        _name_or_path="test/model",
        _commit_hash="revision",
        quant_config=None,
        **kwargs,
    )


def scale_payload(config):
    return {
        "version": 1,
        "kv_cache_dtype": "fp8_e4m3",
        **model_identity(config),
        "layers": {
            "0": {"k_scale": 0.02, "v_scale": 0.03},
            "1": {"k_scale": 0.04, "v_scale": 0.05},
        },
    }


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "declaration",
    [
        {"kv_cache_quant_algo": "FP8"},
        {
            "kv_cache_scheme": {
                "num_bits": 8,
                "type": "float",
                "dynamic": False,
                "strategy": "tensor",
            }
        },
    ],
)
def test_dtype_auto_and_overrides(nested, declaration):
    metadata = {"quantization": declaration} if nested else declaration
    cfg = model_config(quantization_config=metadata)
    assert resolve_kv_dtype(cfg) == "fp8_e4m3"
    assert KVQuantizationConfig.load(cfg, "bf16").torch_dtype == torch.bfloat16
    assert resolve_kv_dtype(model_config()) == "bf16"
    assert resolve_kv_dtype(model_config(), "fp8") == "fp8_e4m3"


@pytest.mark.parametrize(
    "scheme",
    [
        {"kv_cache_quant_algo": "NVFP4"},
        {"kv_cache_quant_algo": "FP8_DYNAMIC"},
        {"kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": True}},
        {"kv_cache_scheme": {"num_bits": 8, "type": "float", "strategy": "channel"}},
        {"kv_cache_scheme": {"num_bits": 8, "type": "int"}},
    ],
)
def test_reject_unsupported_declarations(scheme):
    with pytest.raises(ValueError, match="Unsupported KV-cache"):
        checkpoint_kv_dtype(scheme)


def test_scale_file_precedence_and_identity(tmp_path):
    config = model_config()
    payload = scale_payload(config)
    path = tmp_path / "kv.json"
    path.write_text(json.dumps(payload))
    kv = KVQuantizationConfig.load(
        config, "fp8", path, {0: {"k_scale": 1, "v_scale": 1}}
    )
    assert kv.scales == ((0.02, 0.03), (0.04, 0.05))
    assert kv.torch_dtype == FP8_DTYPE
    for key in ("checkpoint", "weight_format", "model"):
        bad = copy.deepcopy(payload)
        bad[key] = "mismatch"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="model mismatch"):
            KVQuantizationConfig.load(config, "fp8", path)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), True, [1, 2]])
def test_reject_invalid_scales(scale):
    with pytest.raises(ValueError, match="positive finite"):
        KVQuantizationConfig.load(
            model_config(),
            "fp8",
            checkpoint_scales={0: {"k_scale": scale, "v_scale": 1}},
        )


def test_missing_scale_requires_calibration():
    with pytest.raises(ValueError, match="calibrate_kv_cache"):
        KVQuantizationConfig.load(
            model_config(), "fp8", checkpoint_scales={0: {"k_scale": 1}}
        )


@pytest.mark.parametrize("projection", [True, False])
@pytest.mark.parametrize("attention", ["self_attn", "attention"])
def test_extract_only_modelopt_kv_scales(projection, attention):
    state = {
        f"model.layers.0.{attention}.{kind + '_proj.' if projection else ''}{kind}_scale": torch.tensor(
            [value]
        )
        for kind, value in (("k", 0.25), ("v", 0.5))
    }
    state["model.layers.0.self_attn.k_proj.weight_scale"] = torch.tensor(9.0)
    state["model.layers.0.self_attn.k_proj.input_scale"] = torch.tensor(8.0)
    assert extract_checkpoint_kv_scales(state) == {0: {"k_scale": 0.25, "v_scale": 0.5}}
    assert len(state) == 2


def test_calibration_zero_and_tp_reduction(monkeypatch):
    observer = KVCalibrationObserver(2, "cpu")
    observer.observe(0, torch.tensor([-224.0, 1.0]), torch.tensor([112.0]))
    observer.observe(0, torch.tensor([3.0]), torch.tensor([-56.0]))
    observer.observe(1, torch.zeros(3), torch.zeros(3))
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reduce(tensor, op, group):
        calls.append((op, group))
        if op == torch.distributed.ReduceOp.MAX:
            tensor[0, 0] = 448  # another TP rank saw the largest K

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    payload = observer.finish(model_config(), {"warmup_excluded": True}, group="tp")
    assert payload["layers"]["0"]["k_scale"] == 1
    assert payload["layers"]["0"]["v_scale"] == 0.25
    assert payload["layers"]["1"]["k_scale"] == payload["layers"]["1"]["v_scale"] == 1
    assert calls == [
        (torch.distributed.ReduceOp.MAX, "tp"),
        (torch.distributed.ReduceOp.MIN, "tp"),
    ]


def test_calibration_rejects_unseen_and_nonfinite():
    observer = KVCalibrationObserver(2, "cpu")
    with pytest.raises(ValueError, match="every attention layer"):
        observer.finish(model_config(), {})
    observer.observe(0, torch.tensor([float("inf")]), torch.ones(1))
    observer.observe(1, torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="positive finite"):
        observer.finish(model_config(), {})


def test_codec_clipping_and_no_double_scale():
    x = torch.tensor([-1e5, -0.13, 0.0, 0.11, 1e5])
    encoded = encode_kv(x, 0.01)
    expected = torch.clamp(x.float() / 0.01, -448.0, 448.0).to(FP8_DTYPE)
    assert torch.equal(cache_bytes(encoded), cache_bytes(expected))
    assert encode_kv(encoded, 0.01) is encoded
    assert torch.equal(decode_kv(encoded, 0.01), (expected.float() * 0.01).bfloat16())


def test_cli_and_calibration_mode(monkeypatch):
    from fluxserve.cli import build_parser
    from fluxserve import bench_offline

    parser = build_parser()
    serve = parser.parse_args(["serve", "--model", "model", "--kv-cache-dtype", "fp8"])
    assert serve.kv_cache_dtype == "fp8"
    bench = parser.parse_args(
        ["bench_offline", "--model", "model", "--dataset", "data"]
    )
    assert bench.kv_cache_dtype == "auto" and bench.kv_cache_scales is None
    args = parser.parse_args(
        [
            "calibrate_kv_cache",
            "--model",
            "model",
            "--dataset",
            "data",
            "--output",
            "scales.json",
            "--use-cuda-graph",
        ]
    )
    monkeypatch.setattr(bench_offline, "bench_offline", lambda a: None)
    bench_offline.calibrate_kv_cache(args)
    assert args.num_samples == 128
    assert args.attention_backend == "sdpa" and args.kv_cache_dtype == "bf16"
    assert not args.use_cuda_graph and args.calibrate_kv


def test_paged_encoded_storage_reorder_rewrite_and_reuse():
    from fluxserve.backend.managers.kvcache.paged import PagedKVCache

    kwargs = dict(
        num_layers=2,
        batch_size=2,
        local_kv_heads=2,
        max_length=12,
        head_dim=8,
        page_size=4,
        reserve_dummy_page=1,
        device="cpu",
    )
    fp8 = PagedKVCache(dtype=FP8_DTYPE, **kwargs)
    bf16 = PagedKVCache(dtype=torch.bfloat16, **kwargs)
    assert fp8.data.nbytes * 2 == bf16.data.nbytes
    torch.manual_seed(123)
    for scale in (0.02, 0.04):
        encoded = encode_kv(torch.randn(2, 2, 2, 9, 8), scale)
        fp8.write_range(seq_id=1, start=2, kv=encoded)
        out = fp8.materialize(seq_ids=torch.tensor([1, 0]), length=12)
        assert torch.equal(cache_bytes(out[:, :, 0, :, 2:11]), cache_bytes(encoded))
        assert not out[:, :, 1].float().any()
    fp8.page_table[0].copy_(fp8.page_table[1])
    new = encode_kv(torch.randn(2, 2, 2, 5, 8), 0.03)
    fp8.write_range(seq_id=0, start=0, kv=new)
    assert torch.equal(
        cache_bytes(fp8.materialize(seq_ids=torch.tensor([0]), length=5)[:, :, 0]),
        cache_bytes(new),
    )
    assert not fp8.data[:, :, fp8.dummy_page_id].float().any()
    with pytest.raises(TypeError, match="encoded"):
        fp8.write_range(seq_id=0, start=0, kv=new.bfloat16())


def test_dense_attention_current_block_uses_same_encoded_reference():
    from fluxserve.backend.layers.attention.base import AttentionForwardConfig
    from fluxserve.backend.layers.attention.forward import AttentionForward

    torch.manual_seed(7)
    attention = AttentionForward(AttentionForwardConfig(1, 4, 2, 8, 2, 8**-0.5))
    attention.kv_quantization = KVQuantizationConfig(
        "fp8_e4m3", ((1.0, 1.0), (0.02, 0.03)), "test"
    )
    q = torch.randn(2, 4, 3, 8).bfloat16()
    k, v = torch.randn(2, 2, 3, 8).bfloat16(), torch.randn(2, 2, 3, 8).bfloat16()
    past = (
        encode_kv(torch.randn(2, 2, 8, 8), 0.02),
        encode_kv(torch.randn(2, 2, 8, 8), 0.03),
    )
    result, present = attention.forward(q, k, v, past_key_values=past, use_cache=True)
    expected = []
    for old, current, scale in zip(past, (k, v), (0.02, 0.03), strict=True):
        data = old.float().clone()
        data[:, :, -3:] = (
            torch.clamp(current.float() / scale, -448, 448).to(FP8_DTYPE).float()
        )
        expected.append((data * scale).bfloat16())
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, expected[0].repeat_interleave(2, 1), expected[1].repeat_interleave(2, 1)
    )
    torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
    assert all(t.dtype == FP8_DTYPE for t in present)


def _calibration_rank(rank, rendezvous, output_dir):
    from pathlib import Path

    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        observer = KVCalibrationObserver(2, "cpu")
        for layer in range(2):
            observer.observe(
                layer,
                torch.tensor([224.0 * (rank + 1)]),
                torch.tensor([112.0 * (2 - rank)]),
            )
        payload = observer.finish(
            model_config(), {"tp_size": 2}, group=torch.distributed.group.WORLD
        )
        Path(output_dir, f"rank{rank}.json").write_text(json.dumps(payload))
    finally:
        torch.distributed.destroy_process_group()


def test_calibration_real_two_rank_max(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for CPU distributed calibration test")
    torch.multiprocessing.spawn(
        _calibration_rank,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )
    first = json.loads((tmp_path / "rank0.json").read_text())
    assert first == json.loads((tmp_path / "rank1.json").read_text())
    for row in first["layers"].values():
        assert row["k_scale"] == 1 and row["v_scale"] == 0.5


def test_weight_and_kv_metadata_are_independent():
    from fluxserve.cli import _resolve_quant_config

    for weight, expected in (
        ("FP8", "modelopt_fp8"),
        ("NVFP4", "modelopt_nvfp4"),
        (None, None),
    ):
        for nested in (False, True):
            quant = {"kv_cache_quant_algo": "FP8", "quant_algo": weight}
            config = model_config(
                quantization_config={"quantization": quant} if nested else quant
            )
            method = _resolve_quant_config(config)
            assert (method.get_name() if method is not None else None) == expected
            assert resolve_kv_dtype(config) == "fp8_e4m3"


def test_generic_graph_key_contains_scales():
    from fluxserve.backend.execution.cuda_graph_runner import CudaGraphRunner

    graph = CudaGraphRunner.__new__(CudaGraphRunner)
    graph.kv_signature = ("fp8_e4m3", ((0.02, 0.03),))
    first = graph._graph_key(1, True, 64, 128)
    graph.kv_signature = ("fp8_e4m3", ((0.04, 0.03),))
    assert first != graph._graph_key(1, True, 64, 128)


def test_generic_graph_inputs_clear_inactive_rows_and_tail():
    from fluxserve.backend.execution.cuda_graph_runner import CudaGraphRunner

    graph = CudaGraphRunner.__new__(CudaGraphRunner)
    graph.capture_bs = [2]
    graph.model_runner = SimpleNamespace(runner_config=SimpleNamespace(mask_id=7))
    graph.input_ids = torch.full((8,), 99, dtype=torch.long)
    graph.position_ids = torch.full((8,), 99, dtype=torch.long)
    graph.past_key_values = torch.ones(1, 2, 2, 1, 8, 4).to(FP8_DTYPE)
    graph.attention_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    cache = encode_kv(torch.randn(1, 2, 1, 1, 4, 4), 0.02)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    graph.replay_prepare(
        torch.arange(4).unsqueeze(0),
        torch.arange(4).unsqueeze(0),
        cache,
        True,
        4,
        mask,
        8,
    )
    assert graph.input_ids.tolist() == [0, 1, 2, 3, 7, 7, 7, 7]
    assert graph.position_ids.tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    assert not graph.past_key_values[:, :, 1].float().any()
    assert not graph.past_key_values[:, :, :, :, 4:].float().any()
    assert torch.equal(
        cache_bytes(graph.past_key_values[:, :, :1, :, :4]), cache_bytes(cache)
    )
    assert (
        not graph.attention_mask[1, :4].any()
        and not graph.attention_mask[0, :4, 4:].any()
    )
