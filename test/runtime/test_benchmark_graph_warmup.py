import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fluxserve.bench_offline import (
    _flashinfer_capture_counts, _write_memory_and_graph_metrics, warmup_runner,
)
from fluxserve.backend.execution.flashinfer_cuda_graph_runner import FlashInferCudaGraphRunner
from fluxserve.backend.execution.runners.flashinfer_diffusion import FlashInferDiffusionRunner


def acceptance_module():
    path = Path(__file__).resolve().parents[1] / "benchmark/fluxserve/kv_cache_acceptance.py"
    spec = importlib.util.spec_from_file_location("kv_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_warmup_keeps_formal_batch_cache_and_metrics_detect_recapture(dtype, monkeypatch, tmp_path):
    # Exercise real warmup, generate, allocation and graph invalidation on CPU.
    # Replace only GPU forward/capture work with entries keyed by storage address.
    for name in ("synchronize", "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *_: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr("fluxserve.backend.execution.runners.flashinfer_diffusion.get_attention_tp_size", lambda: 1)
    graph = FlashInferCudaGraphRunner.__new__(FlashInferCudaGraphRunner)
    graph.model_family = "llada2"
    graph.supports_llada2_graphs = True
    graph.supports_diffusion_gemma_graphs = False
    graph.capture_sizes = (64,)
    graph._graphs, graph._decode_graphs = {}, {}
    graph.capture_count = graph.decode_capture_count = 0
    graph.invalidation_count = graph.llada2_invalidation_count = 0
    graph._log_callback = lambda *_: None
    graph.record_capture_memory = lambda *_: None
    graph.stats = lambda: {"decode_replay_count": 10}
    runner = FlashInferDiffusionRunner.__new__(FlashInferDiffusionRunner)
    runner.device = "cpu"
    runner.block_length = 64
    runner.max_length = 128
    runner.prefill_lengths = [64]
    runner.num_forwards = 0
    runner.kv_cache_dtype = dtype
    runner.decoder = SimpleNamespace(mask_id=7, eos_id=9)
    runner.model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        num_hidden_layers=2, num_key_value_heads=1, num_attention_heads=2, hidden_size=16,
    )))
    runner.runner_config = SimpleNamespace(
        cache="", gen_length=512, mini_batch_size=4, attention_backend="flashinfer",
        kv_cache_layout="paged", flashinfer_cache_mode="paged", flashinfer_prefill_mode="paged",
        page_size=64, enable_decode_cuda_graph=True, supported_batch_sizes=(1, 2, 4),
        cuda_graph_capture_sizes=(64,),
    )
    runner.flashinfer_graph_runner = graph
    runner.preprocess_inputs = lambda ids: (128, 64, 2)

    def capture_prefill(*_):
        key = (64, runner.past_key_values.data.data_ptr())
        if key not in graph._graphs:
            graph._graphs[key] = object()
            graph.capture_count += 1

    def capture_decode(runner, batch_sizes):
        for size in batch_sizes:
            key = (size, runner.past_key_values.data.data_ptr())
            if key not in graph._decode_graphs:
                graph._decode_graphs[key] = object()
                graph.decode_capture_count += 1

    runner._prefill_batches = capture_prefill
    runner._decode_batches = lambda x, *_: capture_decode(runner, (min(4, x.data.shape[0]),))
    graph.capture_decode_batch_sizes = capture_decode
    args = SimpleNamespace(
        use_cuda_graph=True, use_prefill_cuda_graph=False, use_decode_cuda_graph=False,
        attention_backend="flashinfer", flashinfer_prefill_mode="paged",
        flashinfer_cache_mode="paged", kv_cache_layout="paged",
        batch_size=8, mini_batch_size=4, block_length=64,
        output_dir=str(tmp_path), exp_name="result",
    )
    warmup_runner(runner, args, "cpu", SimpleNamespace(info=lambda *_: None))
    cache = runner.past_key_values
    assert cache.batch_size == 8
    assert runner.runner_config.gen_length == 512
    before = _flashinfer_capture_counts(runner)
    assert before == {"prefill": 1, "decode": 3, "gemma_decode": 0, "invalidations": 0}
    for batch in (8, 2, 8):
        runner.generate(torch.full((batch, 64), 3, dtype=torch.long))
        assert runner.past_key_values is cache
        assert _flashinfer_capture_counts(runner) == before

    # Use the actual metrics writer and acceptance gate together. A positive
    # replay count must not hide captures occurring inside the measured phase.
    metrics_args = (args, runner, SimpleNamespace(total_forward=10, total_token=100, total_time=1), 0, 0, 1)
    _write_memory_and_graph_metrics(*metrics_args, capture_counts_before=before)
    metrics = json.loads((tmp_path / "result_metrics.json").read_text())
    assert metrics["ranks"][0]["flashinfer_graph_before_generation"] == before
    validate = acceptance_module().validate_graph_measurement
    validate(metrics, label="test", backend="flashinfer")
    graph.decode_capture_count += 1
    _write_memory_and_graph_metrics(*metrics_args, capture_counts_before=before)
    metrics = json.loads((tmp_path / "result_metrics.json").read_text())
    assert metrics["ranks"][0]["flashinfer_graph_during_generation"]["decode"] == 1
    with pytest.raises(RuntimeError, match="during timed generation"):
        validate(metrics, label="test", backend="flashinfer")


@pytest.mark.parametrize("field", ["prefill", "decode", "gemma_decode", "invalidations", "missing"])
def test_acceptance_rejects_bad_capture_counters_on_any_rank(field):
    valid = {"prefill": 0, "decode": 0, "gemma_decode": 0, "invalidations": 0}
    bad = dict(valid)
    if field == "missing":
        bad.pop("prefill")
    else:
        bad[field] = 1
    metrics = {"ranks": [{
        "rank": rank, "generic_graph_replays": 0,
        "flashinfer_graph": {"decode_replay_count": 10},
        "flashinfer_graph_during_generation": counts,
    } for rank, counts in enumerate((valid, bad))]}
    with pytest.raises(RuntimeError, match="Missing timed|during timed"):
        acceptance_module().validate_graph_measurement(metrics, label="test", backend="flashinfer")
