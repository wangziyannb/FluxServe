"""Build/check native extensions without initializing a CUDA device."""

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Never compile; fail if a binary is stale or missing.")
    parser.add_argument("--output", type=Path, help="Write a build manifest.")
    args = parser.parse_args()
    if args.check:
        os.environ["FLUX_KERNEL_REQUIRE_PREBUILT"] = "1"

    import torch
    import flux_kernel
    from flux_kernel.cuda.activation import build_activation
    from flux_kernel.cuda.moe import build_moe
    from flux_kernel.cuda.rope import build_rope
    from flux_kernel.cuda.rmsnorm.build import build_rmsnorm_fused_parallel

    libraries = [
        build_rmsnorm_fused_parallel(), build_activation(), build_moe(), build_rope()
    ]
    digest = hashlib.sha256()
    packages = [flux_kernel]
    if importlib.util.find_spec("fluxserve") is not None:
        import fluxserve
        packages.append(fluxserve)
    for package in packages:
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*")):
            if path.suffix in {".py", ".cu", ".h", ".cuh"}:
                digest.update(f"{package.__name__}/{path.relative_to(root)}\0".encode())
                digest.update(path.read_bytes())
    manifest = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "architectures": os.environ.get("FLUX_KERNEL_CUDA_ARCH", os.environ.get("TORCH_CUDA_ARCH_LIST")),
        "source_sha256": digest.hexdigest(),
        "flashinfer_revision": os.environ.get("FLUXSERVE_FLASHINFER_REV", "unknown"),
        "flashinfer": version("flashinfer-python"),
        "triton": version("triton"),
        "libraries": {str(path): path.with_suffix(".build").read_text() for path in libraries},
    }
    text = json.dumps(manifest, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
