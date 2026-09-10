import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fluxserve.backend.engine.executor import BlockDiffusionExecutor
from fluxserve.backend.execution.warmup import (
    prepare_scheduler_warmup_cache, warmup_online_runner, warmup_shapes,
)


def load_entrypoint():
    path = Path(__file__).resolve().parents[2] / "docker/entrypoint.py"
    spec = importlib.util.spec_from_file_location("deployment_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_runner():
    return SimpleNamespace(
        block_length=4, max_length=16, prefilling_limit=8, device="cpu",
        server_args=SimpleNamespace(max_num_seqs=4),
        runner_config=SimpleNamespace(mini_batch_size=2, gen_length=8, attention_backend="sdpa"),
        decoder=SimpleNamespace(mask_id=0, eos_id=1, eos_ids=(1,)),
        early_stop=True, enable_cuda_graph=True, num_forwards=5,
        past_key_values=object(), flashinfer_graph_runner=object(),
    )


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_restores_configuration_cache_and_rng_on_success_or_failure(fail):
    runner = make_runner()
    old_cache = runner.past_key_values
    old_graph = runner.flashinfer_graph_runner
    rng = torch.get_rng_state().clone()
    shapes = []

    def generate(inputs):
        shapes.append(tuple(inputs.shape))
        assert not runner.enable_cuda_graph and not runner.early_stop
        assert runner.flashinfer_graph_runner is None
        assert runner.runner_config.gen_length == 4
        assert not (inputs == runner.decoder.mask_id).any()
        torch.rand(3)
        runner.past_key_values = object()
        runner.num_forwards += 2
        if fail:
            raise RuntimeError("compile failed")

    runner.generate = generate
    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            warmup_online_runner(runner)
    else:
        warmup_online_runner(runner)
        assert shapes == [(1, 9), (2, 9)]
    assert runner.past_key_values is old_cache
    assert runner.flashinfer_graph_runner is old_graph
    assert runner.enable_cuda_graph and runner.early_stop
    assert runner.num_forwards == 5
    assert runner.runner_config.gen_length == 8
    assert torch.equal(torch.get_rng_state(), rng)


def test_warmup_small_context_and_prompt_lengths():
    runner = make_runner()
    runner.max_length = 4
    assert warmup_shapes(runner) == [(1, 0), (2, 0)]
    runner.requires_prompt_lengths = True
    runner.generate = Mock()
    warmup_online_runner(runner)
    assert runner.generate.call_args.kwargs["prompt_lengths"] == [1, 1]


def test_executor_warms_up_before_graphs_and_propagates_failures():
    events = []
    runner = SimpleNamespace(
        prepare_online_warmup=lambda: events.append("warmup"),
        prepare_online_cuda_graphs=lambda: events.append("graphs"),
    )
    asyncio.run(BlockDiffusionExecutor(runner, None).startup())
    assert events == ["warmup", "graphs"]
    events.clear()
    asyncio.run(BlockDiffusionExecutor(runner, None, startup_warmup=False).startup())
    assert events == ["graphs"]
    runner.prepare_online_warmup = Mock(side_effect=RuntimeError("compile failed"))
    events.clear()
    with pytest.raises(RuntimeError, match="compile failed"):
        asyncio.run(BlockDiffusionExecutor(runner, None).startup())
    assert events == []


def test_compiler_caches_are_persistent_and_isolated_by_image(tmp_path):
    module = load_entrypoint()
    manifest = tmp_path / "build-info.json"
    manifest.write_text('"image-one"')
    (tmp_path / "tvm-ffi").mkdir()
    (tmp_path / "tvm-ffi/bridge.so").write_bytes(b"prebuilt bridge")
    first = {"FLUXSERVE_CACHE_DIR": str(tmp_path / "cache")}
    module.prepare_cache(first, manifest)
    assert first["USER"].startswith("fluxserve-")
    assert first["LOGNAME"] == first["USER"]
    assert Path(first["TVM_FFI_CACHE_DIR"]).is_dir()
    assert (Path(first["TVM_FFI_CACHE_DIR"]) / "bridge.so").read_bytes() == b"prebuilt bridge"
    marker = Path(first["TRITON_CACHE_DIR"]) / "compiled-kernel"
    marker.touch()
    second = {"FLUXSERVE_CACHE_DIR": str(tmp_path / "cache")}
    module.prepare_cache(second, manifest)
    assert first == second and marker.exists()
    manifest.write_text('"image-two"')
    third = {"FLUXSERVE_CACHE_DIR": str(tmp_path / "cache")}
    module.prepare_cache(third, manifest)
    assert third["TRITON_CACHE_DIR"] != first["TRITON_CACHE_DIR"]
    assert third["HF_HOME"] == first["HF_HOME"]


def test_cache_overrides_and_offline_readonly_hf(tmp_path, monkeypatch):
    module = load_entrypoint()
    env = {
        "FLUXSERVE_CACHE_DIR": str(tmp_path),
        "TRITON_CACHE_DIR": str(tmp_path / "custom-triton"),
        "HF_HOME": "/mounted-readonly-hf", "HF_HUB_OFFLINE": "1",
    }
    module.prepare_cache(env)
    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "custom-triton")
    assert env["HF_HOME"] == "/mounted-readonly-hf"
    monkeypatch.setattr(module.tempfile, "TemporaryFile", Mock(side_effect=PermissionError("read only")))
    with pytest.raises(RuntimeError, match="TRITON_CACHE_DIR must be writable"):
        module.prepare_cache(env)


def test_entrypoint_executes_without_shell_and_preserves_arguments(monkeypatch, tmp_path):
    module = load_entrypoint()
    assert module.command_argv(["serve", "--model", "model name"]) == ["fluxserve", "serve", "--model", "model name"]
    assert module.command_argv(["bash", "-lc", "echo test"]) == ["bash", "-lc", "echo test"]
    monkeypatch.setenv("FLUXSERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLUX_KERNEL_REQUIRE_PREBUILT", "1")
    monkeypatch.setattr(module.sys, "argv", ["entrypoint.py", "serve", "--model", "$(literal)"])
    monkeypatch.setattr(module.subprocess, "run", Mock())
    monkeypatch.setattr(module.os, "execvp", Mock())
    module.main()
    assert module.subprocess.run.call_args.kwargs["check"] is True
    module.os.execvp.assert_called_once_with("fluxserve", ["fluxserve", "serve", "--model", "$(literal)"])


def test_http_startup_does_not_start_engine_after_warmup_failure():
    from unittest.mock import AsyncMock
    from fluxserve.backend.entrypoints.http_server import create_app

    engine = SimpleNamespace(
        startup_executor=AsyncMock(side_effect=RuntimeError("warmup failed")),
        start=AsyncMock(),
    )
    app = create_app(engine)
    with pytest.raises(RuntimeError, match="warmup failed"):
        asyncio.run(app.router.on_startup[0]())
    engine.start.assert_not_called()


def test_http_shutdown_stops_distributed_workers_before_returning(monkeypatch):
    from unittest.mock import AsyncMock
    from fluxserve.backend.distributed.launch import DistributedContext
    from fluxserve.backend.engine import distributed_executor as module
    from fluxserve.backend.engine.async_llm import AsyncLLM
    from fluxserve.backend.entrypoints.http_server import create_app

    broadcast = Mock()
    monkeypatch.setattr(module, "_broadcast_command", broadcast)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    base = SimpleNamespace(shutdown=AsyncMock())
    executor = module.DistributedGenerationExecutor(base, DistributedContext(world_size=4))
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.executor = executor
    engine._task = None
    engine._new_request_event = asyncio.Event()
    app = create_app(engine)

    async def stop():
        await app.router.on_shutdown[0]()
        broadcast.assert_called_once_with({"kind": "shutdown"})
        base.shutdown.assert_awaited_once()
        # The CLI fallback must not issue a second collective to exited ranks.
        await executor.shutdown_workers()

    asyncio.run(stop())
    broadcast.assert_called_once()
    base.shutdown.assert_awaited_once()


def test_require_prebuilt_never_invokes_compiler(tmp_path, monkeypatch):
    from flux_kernel.cuda import build

    root = tmp_path / "kernel"
    (root / "csrc").mkdir(parents=True)
    (root / "csrc/example.cu").write_text("// example")
    monkeypatch.setenv("FLUX_KERNEL_REQUIRE_PREBUILT", "1")
    monkeypatch.setattr(build, "resolve_cuda_arches", Mock(return_value=("90a",)))
    compiler = Mock()
    monkeypatch.setattr(build.subprocess, "check_call", compiler)
    with pytest.raises(RuntimeError, match="Precompiled Flux Kernel example is missing or incompatible"):
        build.build_cuda_library(root, "example", force=False, verbose=False)
    compiler.assert_not_called()
    assert not (root / "objs").exists()


def test_paged_warmup_uses_final_pool_capacity_and_respects_admission():
    from fluxserve.backend.managers.kvcache.paged import PagedKVCache

    runner = make_runner()
    runner.server_args.scheduler_num_device_pages = 7
    runner.runner_config.page_size = 4
    cache = PagedKVCache(
        num_layers=2, batch_size=4, local_kv_heads=1, max_length=16,
        head_dim=8, page_size=4, num_pages=7, reserve_dummy_page=1,
        dtype=torch.bfloat16, device="cpu",
    )
    def ensure(*, num_device_pages):
        assert num_device_pages == 7
        runner.past_key_values = cache
    runner.ensure_paged_kv_cache = ensure
    actual, batch, prompt = prepare_scheduler_warmup_cache(runner, 4, 9, 4)
    assert actual is cache and actual.uses_external_page_table
    assert (batch, prompt) == (2, 8)
    assert cache.page_table[:2].tolist() == [[1, 2, 3, 0], [4, 5, 6, 0]]


def test_warmup_cli_flags_are_exclusive():
    from fluxserve.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["serve", "--model", "model", "--warmup-only"]).warmup_only
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--model", "model", "--warmup-only", "--skip-startup-warmup"])
