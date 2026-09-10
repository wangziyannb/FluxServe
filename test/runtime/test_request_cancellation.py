import asyncio
import threading
from types import SimpleNamespace

import anyio
import pytest

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.executor import ExecutionResult
from fluxserve.backend.engine.io_struct import GenerateReqInput
from fluxserve.backend.utils.server_args import ServerArgs


@pytest.mark.parametrize("completion", ["success", "error", "shutdown", "release_error"])
def test_anyio_disconnect_finishes_state_and_preserves_resource_release(completion, caplog):
    async def run():
        entered, finish_inference = threading.Event(), threading.Event()
        released = threading.Event()
        events, scopes = [], []

        async def inference(states):
            entered.set()
            assert finish_inference.wait(5)
            events.append("inference_finished")
            if completion == "error":
                raise RuntimeError("injected inference failure")
            return [ExecutionResult(rid=states[0].rid, token_ids=[7], text="7", finish_reason="stop")]

        async def release_requests(ids):
            assert events == ["inference_finished"]
            assert ids == ["cancelled-request"]
            events.append("release")
            released.set()
            if completion == "release_error":
                raise RuntimeError("injected release failure")

        async def cleanup():
            assert events == ["inference_finished", "release"]
            events.append("shutdown")

        engine = AsyncLLM(ServerArgs(), SimpleNamespace(
            offload_execution=True, execute_batch=inference,
            release_requests=release_requests, shutdown_workers=cleanup,
        ), tokenizer=SimpleNamespace())

        async def consume():
            # StreamingResponse uses an AnyIO cancel scope: cancellation is
            # delivered again at each await, including cleanup's lock wait.
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                async for _ in engine.generate_request(GenerateReqInput(
                    input_ids=[1], rid="cancelled-request", stream=True,
                )):
                    pytest.fail("disconnected request received inference output")

        client = asyncio.create_task(consume())
        shutdown = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            state = engine._states["cancelled-request"]
            scopes[0].cancel()
            await asyncio.wait_for(client, 1)
            assert "cancelled-request" not in engine.scheduler._active
            assert "cancelled-request" not in engine._states
            assert state.finished_reason == "abort"
            assert engine.metrics.aborted_requests == 1
            assert engine.metrics.running_requests == 0
            assert events == []  # Inference still owns the execution lock.

            if completion == "shutdown":
                shutdown = asyncio.create_task(engine.shutdown())
                await asyncio.sleep(0)
                assert not shutdown.done()
            finish_inference.set()
            assert await asyncio.to_thread(released.wait, 5)
            if shutdown is None:
                shutdown = asyncio.create_task(engine.shutdown())
            await asyncio.wait_for(shutdown, 5)
            assert events == ["inference_finished", "release", "shutdown"]
            assert not engine._release_tasks
            assert state.finished_reason == "abort"
            assert state.queue.qsize() == 1
            assert state.queue.get_nowait().finish_reason == "abort"
            assert engine.metrics.successful_requests == 0
            assert engine.metrics.failed_requests == 0
            if completion == "release_error":
                assert "executor request release failed" in caplog.text
                assert "injected release failure" in caplog.text
        finally:
            finish_inference.set()
            if not client.done():
                client.cancel()
            await asyncio.gather(client, return_exceptions=True)
            if shutdown is None:
                await engine.shutdown()
            else:
                await asyncio.gather(shutdown, return_exceptions=True)

    asyncio.run(run())
