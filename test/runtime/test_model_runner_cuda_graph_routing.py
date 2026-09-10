from types import SimpleNamespace
from unittest.mock import Mock

import torch

import fluxserve.backend.execution.cuda_graph_runner as cuda_graph_module
from fluxserve.backend.execution.cuda_graph_runner import CudaGraphRunner
from fluxserve.backend.execution.runners.base import ModelRunner


class FakeGraphRunner:
    def __init__(self):
        self.can_run_calls = 0
        self.replay_calls = 0

    def can_run(self, *args, **kwargs):
        self.can_run_calls += 1
        return True

    def replay(self, *args, **kwargs):
        self.replay_calls += 1
        return "graph"


def make_runner(*, prefill: bool, decode: bool):
    runner = ModelRunner.__new__(ModelRunner)
    runner.runner_config = SimpleNamespace(
        enable_prefill_cuda_graph=prefill,
        enable_decode_cuda_graph=decode,
    )
    runner.enable_cuda_graph = prefill or decode
    runner.graph_runner = FakeGraphRunner()
    runner.forward_normal = lambda *args, **kwargs: "eager"
    return runner


def test_flex_block_mask_bypasses_generic_cuda_graph_runner():
    runner = make_runner(prefill=True, decode=True)
    block_mask = type("BlockMask", (), {})()

    result = runner.forward(
        input_ids=torch.ones((1, 64), dtype=torch.long),
        position_ids=torch.arange(64).unsqueeze(0),
        attention_mask=block_mask,
        use_cache=True,
    )

    assert result == "eager"
    assert runner.graph_runner.can_run_calls == 0
    assert runner.graph_runner.replay_calls == 0


def test_decode_only_cuda_graph_does_not_replay_prefill():
    runner = make_runner(prefill=False, decode=True)

    result = runner.forward(
        input_ids=torch.ones((1, 64), dtype=torch.long),
        position_ids=torch.arange(64).unsqueeze(0),
        use_cache=True,
    )

    assert result == "eager"
    assert runner.graph_runner.can_run_calls == 0


def test_decode_cuda_graph_still_replays_tensor_mask():
    runner = make_runner(prefill=False, decode=True)
    cache = torch.zeros((1, 2, 1, 1, 128, 1))

    result = runner.forward(
        input_ids=torch.ones((1, 64), dtype=torch.long),
        position_ids=torch.arange(64).unsqueeze(0),
        past_key_values=cache,
        attention_mask=torch.ones((1, 64, 128), dtype=torch.bool),
        use_cache=True,
    )

    assert result == "graph"
    assert runner.graph_runner.replay_calls == 1


def test_cuda_graph_runner_skips_disabled_prefill_capture(monkeypatch):
    monkeypatch.setattr(cuda_graph_module, "get_attention_tp_size", lambda: 1)
    model_runner = SimpleNamespace(
        supported_batch_sizes=[1],
        device="cpu",
        block_length=64,
        prefill_lengths=[128],
        cache_lengths=[128],
        decoding_lengths=[],
        max_length=128,
        enable_compile=False,
        enable_cuda_graph=False,
        runner_config=SimpleNamespace(
            enable_prefill_cuda_graph=False,
            enable_decode_cuda_graph=True,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=1,
                num_key_value_heads=2,
                num_attention_heads=2,
                hidden_size=16,
            )
        ),
    )

    graph_runner = CudaGraphRunner(model_runner)

    assert graph_runner.prefill_lengths == []
    assert graph_runner.cache_lengths == [128]


def test_model_runner_releases_generic_cuda_graphs_before_distributed_teardown():
    runner = ModelRunner.__new__(ModelRunner)
    runner.tp_group = SimpleNamespace(device_group="tp-group")
    graph_runner = Mock()
    runner.graph_runner = graph_runner

    runner.shutdown_cuda_graphs(log=False)

    graph_runner.shutdown.assert_called_once_with("tp-group", log=False)
    assert runner.graph_runner is None


def test_cuda_graph_runner_shutdown_releases_graph_state(monkeypatch):
    events = []
    device_module = SimpleNamespace(
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
    )
    runner = CudaGraphRunner.__new__(CudaGraphRunner)
    runner.device_module = device_module
    runner.graphs = {"graph": object()}
    runner.output_buffers = {"output": object()}
    runner.input_ids = object()
    runner.position_ids = object()
    runner.past_key_values = object()
    runner.attention_mask = object()
    runner.stream = object()

    barrier = Mock(side_effect=lambda **kwargs: events.append(("barrier", kwargs)))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(cuda_graph_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        cuda_graph_module,
        "set_graph_pool_id",
        lambda value: events.append(("allocator_pool", value)),
    )
    cuda_graph_module.set_global_graph_memory_pool("pool")

    runner.shutdown("tp-group", log=False)
    runner.shutdown("tp-group", log=False)

    assert events == [
        "synchronize",
        ("barrier", {"group": "tp-group"}),
        ("allocator_pool", None),
        "gc",
        "empty_cache",
        ("barrier", {"group": "tp-group"}),
    ]
    assert runner.graphs == {}
    assert runner.output_buffers == {}
    assert runner.input_ids is None
    assert runner.position_ids is None
    assert runner.past_key_values is None
    assert runner.attention_mask is None
    assert runner.stream is None
    assert cuda_graph_module.get_global_graph_memory_pool() is None
