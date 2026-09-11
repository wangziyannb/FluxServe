"""Guard against asynchronous GPU timing and rank-zero-only throughput."""

import torch
import pytest

from fluxserve import bench_offline as bench


@pytest.mark.parametrize("distributed", [False, True])
def test_generation_timing_includes_gpu_completion_and_excludes_reduction(monkeypatch, distributed):
    clock = [0.0]
    monkeypatch.setattr(bench.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: distributed)

    def barrier():
        clock[0] += 20.0  # Startup skew must not enter generation time.

    def synchronize(_device):
        clock[0] += 3.0  # Represent outstanding GPU work.

    def all_reduce(value, op):
        assert op == torch.distributed.ReduceOp.MAX
        assert value.item() == 5.0
        value.fill_(7.0)  # Another rank completed later than this rank.
        clock[0] += 100.0  # Metric collection must not enter generation time.

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    start = bench._start_generation_timing("cpu")
    clock[0] += 2.0
    local, maximum = bench._finish_generation_timing(start, "cpu")
    assert local == 5.0
    assert maximum == (7.0 if distributed else 5.0)
