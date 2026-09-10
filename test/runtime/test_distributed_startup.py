import asyncio
from datetime import timedelta
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch.distributed as dist

from fluxserve.backend.distributed.launch import DistributedContext
from fluxserve.backend.engine import distributed_executor as module


def startup_worker(rank, directory, failing_ranks, started, release, ready):
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/store", rank=rank,
        world_size=2, timeout=timedelta(seconds=6),
    )
    events = []
    original_broadcast = module._broadcast_command

    def broadcast(command):
        events.append(command["kind"])
        original_broadcast(command)

    module._broadcast_command = broadcast

    async def startup():
        started[rank].set()
        if rank == 1:
            assert release.wait(10)
        if rank in failing_ranks:
            raise RuntimeError(f"injected failure {rank}")
        return {"ready": rank}

    async def shutdown():
        # Graph cleanup itself contains collectives. Verify every rank reaches
        # the same cleanup phase, including ranks whose warmup succeeded.
        dist.barrier()
        events.append("cleanup")
        dist.barrier()

    executor = module.DistributedGenerationExecutor(
        SimpleNamespace(startup=startup, shutdown=shutdown),
        DistributedContext(rank=rank, local_rank=rank, world_size=2, backend="gloo"),
    )

    async def run():
        if rank == 0:
            try:
                await executor.startup()
                ready.set()
                events.append("ready")
            finally:
                await executor.shutdown_workers()
        else:
            await executor.run_worker_loop()

    try:
        asyncio.run(run())
    except Exception as error:
        events.append(str(error))
    finally:
        dist.destroy_process_group()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(events))


@pytest.mark.parametrize("failing_ranks", [(), (0,), (1,), (0, 1)])
def test_startup_outcomes_and_cleanup_agree_across_real_ranks(tmp_path, failing_ranks):
    context = mp.get_context("spawn")
    started = [context.Event(), context.Event()]
    release, ready = context.Event(), context.Event()
    processes = [context.Process(
        target=startup_worker,
        args=(rank, str(tmp_path), failing_ranks, started, release, ready),
    ) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        assert all(event.wait(15) for event in started)
        assert not ready.wait(0.1), "rank 0 became ready before rank 1 finished"
        release.set()
        for process in processes:
            process.join(15)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
        results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
        assert all(events.count("cleanup") == 1 for events in results)
        if failing_ranks:
            assert not ready.is_set()
            assert "shutdown" not in results[0]
            assert results[0][-1] == results[1][-1]
            for rank in failing_ranks:
                assert f"rank {rank}: RuntimeError: injected failure {rank}" in results[0][-1]
        else:
            assert ready.is_set()
            assert results[0] == ["startup", "ready", "shutdown", "cleanup"]
            assert results[1] == ["cleanup"]
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)


def test_failed_startup_exchange_does_not_send_shutdown_on_broken_group(monkeypatch):
    broadcast = Mock()
    monkeypatch.setattr(module, "_broadcast_command", broadcast)
    monkeypatch.setattr(dist, "all_gather_object", Mock(side_effect=RuntimeError("broken group")))
    base = SimpleNamespace(startup=AsyncMock(return_value={}), shutdown=AsyncMock())
    executor = module.DistributedGenerationExecutor(base, DistributedContext(world_size=2))

    async def run():
        with pytest.raises(RuntimeError, match="broken group"):
            await executor.startup()
        await executor.shutdown_workers()

    asyncio.run(run())
    broadcast.assert_called_once_with({"kind": "startup"})
    base.shutdown.assert_not_called()
