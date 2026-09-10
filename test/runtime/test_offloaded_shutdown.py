import asyncio
import threading
from types import SimpleNamespace

import pytest

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.utils.server_args import ServerArgs


@pytest.mark.parametrize("fail_inference", [False, True])
def test_shutdown_drains_worker_even_after_repeated_cancellation(fail_inference):
    async def run():
        entered, release, finished = (threading.Event() for _ in range(3))
        events = []

        async def inference():
            entered.set()
            assert release.wait(5)
            events.append("inference_finished")
            finished.set()
            if fail_inference:
                raise RuntimeError("injected inference error")

        async def cleanup():
            assert finished.is_set(), "cleanup raced with the inference thread"
            events.append("cleanup")

        executor = SimpleNamespace(offload_execution=True, shutdown_workers=cleanup)
        engine = AsyncLLM(ServerArgs(), executor, tokenizer=SimpleNamespace())
        engine._task = asyncio.create_task(engine._execute(inference))
        assert await asyncio.to_thread(entered.wait, 5)
        shutdown = asyncio.create_task(engine.shutdown())
        try:
            await asyncio.sleep(0)
            engine._task.cancel()  # A second cancellation must still drain.
            await asyncio.sleep(0)
            assert not shutdown.done()
            assert events == []
        finally:
            release.set()
        await asyncio.wait_for(shutdown, 5)
        assert events == ["inference_finished", "cleanup"]
        assert engine._task.cancelled()
        with pytest.raises(RuntimeError, match="shutting down"):
            await engine.start()

    asyncio.run(run())


def test_request_release_is_serialized_with_offloaded_inference():
    async def run():
        entered, release, finished = (threading.Event() for _ in range(3))
        released = []

        async def inference():
            entered.set()
            assert release.wait(5)
            finished.set()

        async def release_requests(ids):
            assert finished.is_set()
            released.extend(ids)

        executor = SimpleNamespace(offload_execution=True, release_requests=release_requests)
        engine = AsyncLLM(ServerArgs(), executor, tokenizer=SimpleNamespace())
        work = asyncio.create_task(engine._execute(inference))
        assert await asyncio.to_thread(entered.wait, 5)
        freeing = asyncio.create_task(engine._release_executor_requests(["request"]))
        try:
            await asyncio.sleep(0)
            assert not freeing.done()
            assert released == []
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(work, freeing), 5)
        assert released == ["request"]
        engine._closed = True
        await engine._release_executor_requests(["late-disconnect"])
        assert released == ["request"]

    asyncio.run(run())
