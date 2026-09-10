"""Prepare writable compiler/model caches, then replace PID 1 with the command."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SUBCOMMANDS = {"serve", "bench", "bench_offline", "calibrate_kv_cache", "env"}


def prepare_cache(environ, manifest=Path(__file__).resolve().parents[1] / "build-info.json"):
    # PyTorch calls getpass.getuser() even with an explicit Inductor cache.
    # Arbitrary Docker --user UIDs need not have an /etc/passwd entry.
    environ.setdefault("USER", f"fluxserve-{os.getuid()}")
    environ.setdefault("LOGNAME", environ["USER"])
    root = Path(environ.get("FLUXSERVE_CACHE_DIR", "/var/cache/fluxserve"))
    root = root / f"uid-{os.getuid()}"
    runtime = {"python": sys.version}
    for package in ("torch", "triton", "flashinfer-python", "apache-tvm-ffi", "nvidia-cutlass-dsl", "sgl-kernel"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = None
    identity = manifest.read_bytes() if manifest.exists() else b"development"
    fingerprint = hashlib.sha256(identity + json.dumps(runtime, sort_keys=True).encode()).hexdigest()[:16]
    compiler_root = root / fingerprint
    defaults = {
        "TRITON_CACHE_DIR": compiler_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": compiler_root / "torchinductor",
        "TORCH_EXTENSIONS_DIR": compiler_root / "torch-extensions",
        "FLASHINFER_WORKSPACE_BASE": compiler_root / "flashinfer",
        "TVM_FFI_CACHE_DIR": compiler_root / "tvm-ffi",
        "CUDA_CACHE_PATH": compiler_root / "cuda",
        "HF_MODULES_CACHE": compiler_root / "hf-modules",
        "XDG_CACHE_HOME": root / "xdg",
        "HF_HOME": root / "huggingface",
    }
    for name, default in defaults.items():
        path = Path(environ.setdefault(name, str(default))).expanduser()
        # An explicitly mounted, read-only HF cache is valid in offline mode.
        readonly_hf = name == "HF_HOME" and environ.get("HF_HUB_OFFLINE") == "1"
        if not readonly_hf:
            path.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.TemporaryFile(dir=path):
                    pass
            except OSError as exc:
                raise RuntimeError(f"{name} must be writable: {path}") from exc
        environ[name] = str(path)
    # Seed the versioned writable cache with the image's CPU/CUDA FP8 DLPack
    # bridges. Atomic replacement also supports concurrent container starts.
    for source in (manifest.parent / "tvm-ffi").glob("*.so"):
        target = Path(environ["TVM_FFI_CACHE_DIR"]) / source.name
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with source.open("rb") as library:
                    shutil.copyfileobj(library, temporary)
            try:
                temporary_path.chmod(0o644)
                temporary_path.replace(target)
            finally:
                temporary_path.unlink(missing_ok=True)
    print(f"FluxServe compiler cache: {environ['TRITON_CACHE_DIR']}", flush=True)


def command_argv(args):
    if not args or args[0] in SUBCOMMANDS or args[0].startswith("-"):
        return ["fluxserve", *(args or ["--help"])]
    return args


def main():
    prepare_cache(os.environ)
    command = command_argv(sys.argv[1:])
    if command[0] == "fluxserve" and os.environ.get("FLUX_KERNEL_REQUIRE_PREBUILT") == "1":
        subprocess.run([sys.executable, "-m", "flux_kernel.precompile", "--check"], check=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
