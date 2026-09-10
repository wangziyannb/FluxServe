"""Build the pinned TVM FFI Torch FP8 bridge without initializing a GPU."""

import ctypes
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "tvm-ffi"
    output.mkdir(exist_ok=True)
    package = Path(importlib.util.find_spec("tvm_ffi").origin).parent
    build_script = package / "utils/_build_optional_torch_c_dlpack.py"
    major, minor = torch.__version__.split(".")[:2]
    libraries = {}
    # The loader selects its filename using cuda.is_available(). Build both so
    # CPU environment checks and GPU serving can use the same read-only image.
    for device in ("cpu", "cuda"):
        name = f"libtorch_c_dlpack_addon_torch{major}{minor}-{device}.so"
        command = [sys.executable, str(build_script), "--output-dir", str(output), "--libname", name]
        if device == "cuda":
            command.append("--build-with-cuda")
        subprocess.run(command, check=True)
        library = ctypes.CDLL(str(output / name))
        api = library.TorchDLPackExchangeAPIPtr
        api.restype = ctypes.c_uint64
        api.argtypes = []
        if not api():
            raise RuntimeError(f"Invalid Torch DLPack API: {name}")
        libraries[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    os.environ["TVM_FFI_CACHE_DIR"] = str(output)
    from tvm_ffi import from_dlpack
    view = from_dlpack(torch.empty(1, dtype=torch.float8_e4m3fn))
    print(f"TVM FFI FP8 bridge check: {view.dtype}", flush=True)
    manifest = root / "build-info.json"
    metadata = json.loads(manifest.read_text())
    metadata["tvm_ffi"] = {"version": version("apache-tvm-ffi"), "addons_sha256": libraries}
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
