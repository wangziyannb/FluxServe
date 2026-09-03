from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.layers.linear import ReplicatedLinear
from fluxserve.backend.layers.quantization.modelopt_fp8 import ModelOptFp8Config
from fluxserve.backend.layers.quantization.modelopt_nvfp4 import ModelOptNvfp4Config
from fluxserve.backend.layers.quantization.unquant import UnquantizedLinearMethod
from fluxserve.backend.model_loader.loader import (
    DefaultModelLoader,
    _shard_merged_column_parts,
)
from fluxserve.backend.models.llada2 import _shard_qkv_rows
from fluxserve.cli import (
    _resolve_quant_config,
    _validate_quantization_capability,
    build_parser,
)

STATIC_FP8_CONFIG = {
    "producer": {"name": "modelopt", "version": "test"},
    "quant_method": "modelopt",
    "quant_algo": "FP8",
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"type": "float", "num_bits": 8, "dynamic": False},
            "input_activations": {
                "type": "float",
                "num_bits": 8,
                "dynamic": False,
            },
        }
    },
}

STATIC_NVFP4_CONFIG = {
    "producer": {"name": "modelopt", "version": "test"},
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "type": "float",
                "num_bits": 4,
                "dynamic": False,
                "group_size": 16,
            },
            "input_activations": {
                "type": "float",
                "num_bits": 4,
                "dynamic": False,
                "group_size": 16,
            },
        }
    },
}


def test_auto_detects_modelopt_static_fp8():
    model_config = SimpleNamespace(quantization_config=STATIC_FP8_CONFIG)

    quant_config = _resolve_quant_config(model_config)

    assert quant_config.get_name() == "modelopt_fp8"
    assert quant_config.exclude_modules == ["lm_head"]


def test_auto_detects_modelopt_static_nvfp4():
    model_config = SimpleNamespace(quantization_config=STATIC_NVFP4_CONFIG)

    quant_config = _resolve_quant_config(model_config)

    assert quant_config.get_name() == "modelopt_nvfp4"
    assert quant_config.exclude_modules == ["lm_head"]


@pytest.mark.parametrize(
    "quantization_config",
    [
        {**STATIC_NVFP4_CONFIG, "quant_algo": "NVFP4_AWQ"},
        {
            **STATIC_FP8_CONFIG,
            "config_groups": {
                "group_0": {
                    "weights": {"type": "float", "num_bits": 8, "dynamic": True},
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                        "dynamic": False,
                    },
                }
            },
        },
        {"quant_method": "unknown", "quant_algo": "FP8"},
    ],
)
def test_rejects_unsupported_quantization(quantization_config):
    model_config = SimpleNamespace(quantization_config=quantization_config)

    with pytest.raises(ValueError, match="Unsupported checkpoint quantization format"):
        _resolve_quant_config(model_config)


def test_explicit_fp8_requires_checkpoint_metadata():
    model_config = SimpleNamespace(quantization_config=None)

    with pytest.raises(ValueError, match="requires matching serialized ModelOpt"):
        _resolve_quant_config(model_config, "modelopt_fp8")


def test_explicit_nvfp4_requires_checkpoint_metadata():
    model_config = SimpleNamespace(quantization_config=None)

    with pytest.raises(ValueError, match="requires matching serialized ModelOpt"):
        _resolve_quant_config(model_config, "modelopt_nvfp4")


def test_quantization_cli_choices():
    parser = build_parser()

    serve = parser.parse_args(
        ["serve", "--model", "test", "--quantization", "modelopt_fp8"]
    )
    offline = parser.parse_args(
        [
            "bench_offline",
            "--model",
            "test",
            "--dataset",
            "test",
            "--quantization",
            "modelopt_fp8",
        ]
    )

    assert serve.quantization == "modelopt_fp8"
    assert offline.quantization == "modelopt_fp8"

    nvfp4 = parser.parse_args(
        ["serve", "--model", "test", "--quantization", "modelopt_nvfp4"]
    )
    assert nvfp4.quantization == "modelopt_nvfp4"


def test_modelopt_import_does_not_use_sgl_kernel(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sgl_kernel" or name.startswith("sgl_kernel."):
            raise AssertionError(f"unexpected import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module(
        "fluxserve.backend.layers.quantization.modelopt_fp8"
    )
    importlib.reload(module)
    module = importlib.import_module(
        "fluxserve.backend.layers.quantization.modelopt_nvfp4"
    )
    importlib.reload(module)


@pytest.mark.parametrize("config_class", [ModelOptFp8Config, ModelOptNvfp4Config])
def test_modelopt_ignore_globs_create_unquantized_linears(config_class):
    config = config_class(["model.layers.0*", "model.layers.1.attention*"])

    layer_zero = ReplicatedLinear(
        16,
        16,
        bias=False,
        quant_config=config,
        prefix="model.layers.0.self_attn.query_key_value",
    )
    attention = ReplicatedLinear(
        16,
        16,
        bias=False,
        quant_config=config,
        prefix="model.layers.1.attention.query_key_value",
    )

    assert isinstance(layer_zero.quant_method, UnquantizedLinearMethod)
    assert isinstance(attention.quant_method, UnquantizedLinearMethod)


def test_nvfp4_merged_column_scales_shard_each_logical_weight():
    gate = torch.arange(16).reshape(8, 2)
    up = torch.arange(100, 116).reshape(8, 2)

    rank_zero = _shard_merged_column_parts([gate, up], rank=0, world_size=2)
    rank_one = _shard_merged_column_parts([gate, up], rank=1, world_size=2)

    assert torch.equal(rank_zero, torch.cat([gate[:4], up[:4]]))
    assert torch.equal(rank_one, torch.cat([gate[4:], up[4:]]))


def test_excluded_moe_layer_loads_unquantized_weights_without_scales():
    captured = {}
    experts = SimpleNamespace(quant_method=UnquantizedLinearMethod())
    model = SimpleNamespace(
        quant_config=SimpleNamespace(get_name=lambda: "modelopt_nvfp4"),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    mlp=SimpleNamespace(experts=experts),
                )
            ]
        ),
        apply_state_dicts=lambda state: captured.update(state),
        named_parameters=lambda: (),
        named_buffers=lambda: (),
    )
    gate = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    up = torch.arange(8, 16, dtype=torch.bfloat16).reshape(2, 4)
    down = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    state_dict = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": gate,
        "model.layers.0.mlp.experts.0.up_proj.weight": up,
        "model.layers.0.mlp.experts.0.down_proj.weight": down,
        "model.word_embeddings.weight": torch.zeros(2, 4),
    }

    DefaultModelLoader()._update_state_dict_for_fusemoe_quant(
        model,
        state_dict,
        num_layers=1,
        dtype=torch.bfloat16,
        per_gpu_expert_mapping=[torch.tensor([0])],
        per_gpu_inverse_mapping=[torch.tensor([0])],
        device="cpu",
    )

    prefix = "model.layers.0.mlp.experts"
    assert torch.equal(
        captured[f"{prefix}.w13_weight"],
        torch.stack([torch.cat([gate, up])]),
    )
    assert torch.equal(captured[f"{prefix}.w2_weight"], torch.stack([down]))
    assert not any(key.startswith(prefix) and "scale" in key for key in captured)


@pytest.mark.parametrize(
    ("tp_rank", "tp_size", "expected_rows"),
    [
        (0, 2, [0, 1, 2, 3, 8, 9, 12, 13]),
        (1, 2, [4, 5, 6, 7, 10, 11, 14, 15]),
        (1, 4, [2, 3, 8, 9, 12, 13]),
        (3, 4, [6, 7, 10, 11, 14, 15]),
    ],
)
def test_nvfp4_qkv_scales_follow_gqa_weight_sharding(
    tp_rank, tp_size, expected_rows
):
    scale = torch.arange(16).reshape(16, 1)

    shard = _shard_qkv_rows(
        scale,
        hidden_size=8,
        total_num_heads=4,
        total_kv_heads=2,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )

    assert shard[:, 0].tolist() == expected_rows


def test_nvfp4_rejects_hopper_before_weight_loading(monkeypatch):
    config = ModelOptNvfp4Config()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    with pytest.raises(RuntimeError, match="requires CUDA compute capability 10.0"):
        _validate_quantization_capability(config, "cuda:0")


@pytest.mark.parametrize(
    ("config", "capability"),
    [
        (ModelOptFp8Config(), (8, 9)),
        (ModelOptNvfp4Config(), (10, 0)),
        (ModelOptNvfp4Config(), (12, 0)),
    ],
)
def test_modelopt_accepts_supported_gpu_capability(monkeypatch, config, capability):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: capability
    )

    _validate_quantization_capability(config, "0")
