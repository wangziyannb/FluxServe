"""Static, per-layer FP8 KV storage, independent of weight quantization.

Scales are dequantization multipliers (amax / 448), not inverse scales.
Only the attention boundary encodes floating-point inputs. Cache movement must
copy encoded bytes without converting or applying scales a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def add_kv_cache_arguments(parser):
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "bf16", "fp8_e4m3", "fp8"),
        default="auto",
        help="KV storage format, independent of weight quantization.",
    )
    parser.add_argument(
        "--kv-cache-scales",
        help="Calibrated per-layer K/V scale JSON (overrides checkpoint scales).",
    )


def checkpoint_kv_dtype(metadata) -> str:
    if not metadata:
        return "bf16"
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint quantization_config must be a mapping.")
    quant = metadata.get("quantization", metadata)
    algo = quant.get("kv_cache_quant_algo")
    scheme = quant.get("kv_cache_scheme")
    if algo in (None, "") and scheme in (None, {}):
        return "bf16"
    if algo not in (None, "") and str(algo).upper() not in ("FP8", "FP8_E4M3"):
        raise ValueError(
            f"Unsupported KV-cache quantization algorithm: {algo!r}; only static per-tensor FP8 E4M3 is supported."
        )
    if scheme not in (None, {}):
        if not isinstance(scheme, dict) or (
            scheme.get("num_bits") != 8
            or scheme.get("type") != "float"
            or scheme.get("dynamic", False) is not False
            or scheme.get("strategy", "tensor") not in ("tensor", "per_tensor")
            or scheme.get("group_size") not in (None, 0)
            or scheme.get("block_structure") is not None
            or scheme.get("symmetric", True) is not True
            or scheme.get("axis") is not None
            or str(scheme.get("dtype", "fp8_e4m3")).lower()
            not in ("fp8", "fp8_e4m3", "float8_e4m3fn")
        ):
            raise ValueError(
                "Unsupported KV-cache scheme; only static per-tensor FP8 E4M3 is supported."
            )
    return "fp8_e4m3"


def resolve_kv_dtype(config, requested="auto") -> str:
    declared = checkpoint_kv_dtype(getattr(config, "quantization_config", None))
    requested = "fp8_e4m3" if requested == "fp8" else requested
    if requested not in ("auto", "bf16", "fp8_e4m3"):
        raise ValueError(f"Unsupported KV-cache dtype: {requested!r}")
    selected = declared if requested == "auto" else requested
    if selected != "bf16" and (
        getattr(config, "model_type", "") == "diffusion_gemma"
        or "DiffusionGemmaForBlockDiffusion"
        in (getattr(config, "architectures", ()) or ())
    ):
        raise ValueError(
            "FP8 KV cache is supported for LLaDA2 only; DiffusionGemma KV quantization is not implemented."
        )
    return selected


def positive_scale(value, label):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must be a per-tensor scalar.")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a positive finite scalar.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite scalar.")
    float32_value = torch.tensor(value, dtype=torch.float32).item()
    if not math.isfinite(float32_value) or float32_value <= 0:
        raise ValueError(f"{label} must be positive and finite in float32.")
    return value


_SCALE_KEY = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:self_attn|attention)\.(?:(k_proj|v_proj)\.)?([kv])_scale$"
)


def extract_checkpoint_kv_scales(state_dict):
    """Remove only explicit ModelOpt KV scales before linear/expert weight loading."""
    scales = {}
    for name in list(state_dict):
        match = _SCALE_KEY.search(name)
        if match is None:
            continue
        layer, projection, kind = match.groups()
        if projection is not None and projection != f"{kind}_proj":
            raise ValueError(f"Mismatched KV scale name: {name}")
        value = state_dict.pop(name)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        row = scales.setdefault(int(layer), {})
        key = f"{kind}_scale"
        if key in row and not torch.equal(
            torch.as_tensor(row[key]), torch.as_tensor(value)
        ):
            raise ValueError(
                f"Conflicting checkpoint KV scales for layer {layer} {key}."
            )
        row[key] = value
    return scales


def model_identity(config):
    quant = getattr(config, "quant_config", None)
    return {
        "checkpoint": {
            "name": str(getattr(config, "_name_or_path", "")),
            "revision": getattr(config, "_commit_hash", None),
        },
        "weight_format": quant.get_name() if quant is not None else "bf16",
        "model": {
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            ),
        },
    }


@dataclass(frozen=True)
class KVQuantizationConfig:
    dtype: str = "bf16"
    scales: tuple[tuple[float, float], ...] = ()
    source: str = "none"

    @property
    def torch_dtype(self):
        return FP8_DTYPE if self.dtype == "fp8_e4m3" else torch.bfloat16

    @property
    def signature(self):
        return self.dtype, self.scales

    @classmethod
    def load(cls, config, requested="auto", scales_file=None, checkpoint_scales=None):
        dtype = resolve_kv_dtype(config, requested)
        if dtype == "bf16":
            return cls()
        scales = checkpoint_scales or {}
        source = "checkpoint"
        if scales_file:
            payload = json.loads(Path(scales_file).read_text())
            if not isinstance(payload, dict):
                raise ValueError("KV scale file must contain a JSON object.")
            if (
                payload.get("version") != 1
                or payload.get("kv_cache_dtype") != "fp8_e4m3"
            ):
                raise ValueError("Unsupported KV scale file version or dtype.")
            for field, expected in model_identity(config).items():
                if payload.get(field) != expected:
                    raise ValueError(
                        f"KV scale file model mismatch: {field}; expected {expected!r}, got {payload.get(field)!r}."
                    )
            scales = payload.get("layers", {})
            source = str(scales_file)
        if not isinstance(scales, dict):
            raise ValueError(
                "KV scales must be a mapping of layer numbers to K/V scales."
            )
        result = []
        for layer in range(config.num_hidden_layers):
            row = scales.get(str(layer), scales.get(layer, {}))
            if not isinstance(row, dict) or not all(
                key in row for key in ("k_scale", "v_scale")
            ):
                raise ValueError(
                    f"Missing K/V scales for layer {layer}. Run fluxserve calibrate_kv_cache --model ... --dataset ... --output scales.json, then use --kv-cache-scales scales.json."
                )
            result.append(
                tuple(
                    positive_scale(row[key], f"layer {layer} {key}")
                    for key in ("k_scale", "v_scale")
                )
            )
        return cls(dtype, tuple(result), source)


def encode_kv(x: torch.Tensor, scale: float) -> torch.Tensor:
    if x.dtype == FP8_DTYPE:
        return x
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def decode_kv(x: torch.Tensor, scale: float) -> torch.Tensor:
    if x.dtype != FP8_DTYPE:
        raise TypeError("Expected encoded FP8 KV cache.")
    return (x.float() * scale).to(torch.bfloat16)


def cache_bytes(x):
    """Byte view for cache indexing on PyTorch builds without FP8 index kernels."""
    return x.view(torch.uint8) if x.dtype == FP8_DTYPE else x


class KVCalibrationObserver:
    def __init__(self, num_layers, device):
        self.amax = torch.zeros((num_layers, 2), dtype=torch.float32, device=device)
        self.seen = torch.zeros(num_layers, dtype=torch.int32, device=device)

    @torch.no_grad()
    def observe(self, layer, k, v):
        maxima = torch.stack(
            (k.detach().float().abs().amax(), v.detach().float().abs().amax())
        )
        self.amax[layer].copy_(torch.maximum(self.amax[layer], maxima))
        self.seen[layer] = 1

    def finish(self, config, parameters, group=None):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                self.amax, op=torch.distributed.ReduceOp.MAX, group=group
            )
            torch.distributed.all_reduce(
                self.seen, op=torch.distributed.ReduceOp.MIN, group=group
            )
        if not bool(self.seen.all()):
            raise ValueError(
                "Calibration did not observe every attention layer on every rank."
            )
        values = self.amax.cpu().tolist()
        layers = {}
        for i, (k, v) in enumerate(values):
            layers[str(i)] = {
                "k_scale": positive_scale(
                    k / FP8_MAX if k else 1.0, f"layer {i} k_scale"
                ),
                "v_scale": positive_scale(
                    v / FP8_MAX if v else 1.0, f"layer {i} v_scale"
                ),
                "k_amax": k,
                "v_amax": v,
            }
        return {
            "version": 1,
            "kv_cache_dtype": "fp8_e4m3",
            **model_identity(config),
            "calibration": parameters,
            "layers": layers,
        }


def configure_kv_attention(model, config, observer=None):
    for module in model.modules():
        attention = getattr(module, "attention_forward", None)
        if attention is not None:
            attention.kv_quantization = config
            attention.kv_observer = observer
