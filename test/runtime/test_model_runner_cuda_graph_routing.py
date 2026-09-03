from types import SimpleNamespace

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
