from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import pytest

from fluxserve.cli import _resolve_quant_config, build_parser

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
