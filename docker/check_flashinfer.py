"""Validate the FlashInfer APIs used by FluxServe without running CUDA kernels."""

import argparse
import inspect
import json
from pathlib import Path


def require_parameters(function, names, label):
    parameters = inspect.signature(function).parameters
    missing = sorted(set(names) - parameters.keys())
    if missing:
        raise RuntimeError(f"Incompatible FlashInfer: {label} lacks {', '.join(missing)}")


def check_apis(flashinfer, cutlass_fused_moe, activation_type, expected_revision=None):
    revision = flashinfer.__git_commit__
    if expected_revision is not None and revision != expected_revision:
        raise RuntimeError(
            f"FlashInfer revision mismatch: expected {expected_revision}, got {revision}"
        )
    for layout in ("Paged", "Ragged"):
        name = f"BatchPrefillWith{layout}KVCacheWrapper"
        wrapper = getattr(flashinfer, name)
        require_parameters(wrapper.__init__, (
            "kv_layout", "backend", "use_cuda_graph", "block_extend", "block_size",
            "q_offsets_buf", "kv_offsets_buf",
        ), f"{name}.__init__")
        require_parameters(wrapper.plan, (
            "q_offsets", "kv_offsets", "q_data_type", "kv_data_type", "sm_scale",
            "custom_mask", "packed_custom_mask",
        ), f"{name}.plan")
        require_parameters(wrapper.run, ("k_scale", "v_scale"), f"{name}.run")
    for name in ("segment_packbits", "append_paged_kv_cache", "get_batch_indices_positions"):
        if not callable(getattr(flashinfer, name, None)):
            raise RuntimeError(f"Incompatible FlashInfer: missing {name}")
    require_parameters(cutlass_fused_moe, (
        "input", "token_selected_experts", "token_final_scales",
        "fc1_expert_weights", "fc2_expert_weights", "output_dtype", "quant_scales",
        "output", "tp_size", "tp_rank", "ep_size", "ep_rank", "tune_max_num_tokens",
        "enable_pdl", "activation_type",
    ), "cutlass_fused_moe")
    if not hasattr(activation_type, "Swiglu"):
        raise RuntimeError("Incompatible FlashInfer: ActivationType.Swiglu is missing")
    return {
        "version": flashinfer.__version__,
        "revision": revision,
        "paged_block_extend": True,
        "ragged_block_extend": True,
        "fp8_kv_scale_api": True,
        "nvfp4_moe_api": True,
        "validation": "imports_and_signatures_only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    import torch
    import flashinfer
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    result = check_apis(flashinfer, cutlass_fused_moe, ActivationType, args.expected_revision)
    if torch.cuda.is_initialized():
        raise RuntimeError("FlashInfer API build check unexpectedly initialized CUDA")
    print("FlashInfer API check: " + json.dumps(result, sort_keys=True), flush=True)
    if args.manifest is not None:
        metadata = json.loads(args.manifest.read_text())
        metadata["flashinfer_api_check"] = result
        args.manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
