# Adapted from vLLM's ModelOpt quantization integration.
# Copyright (c) 2026 FLUX-OSS

"""ModelOpt serialized static per-tensor FP8 inference support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from flux_kernel import static_scaled_fp8_quant
from torch.nn.parameter import Parameter

from fluxserve.backend.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from fluxserve.backend.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from fluxserve.backend.layers.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from fluxserve.backend.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from fluxserve.backend.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)
_UNLOADED_SCALE = torch.finfo(torch.float32).min


def _validate_static_float8_scheme(name: str, scheme: Dict[str, Any]) -> None:
    if scheme.get("type") != "float" or scheme.get("num_bits") != 8:
        raise ValueError(
            f"ModelOpt FP8 requires {name} to use 8-bit float quantization."
        )
    if scheme.get("dynamic") is not False:
        raise ValueError(
            f"ModelOpt FP8 only supports serialized static {name}; "
            "dynamic quantization is not supported."
        )
    if scheme.get("axis") not in (None, "per_tensor"):
        raise ValueError("ModelOpt FP8 only supports per-tensor scaling.")


def _validate_loaded_scale(layer: torch.nn.Module, scale_name: str) -> None:
    scale = getattr(layer, scale_name, None)
    if scale is None:
        raise ValueError(f"ModelOpt FP8 layer is missing {scale_name}.")
    if torch.any(scale == _UNLOADED_SCALE):
        raise ValueError(
            f"ModelOpt FP8 layer {layer.__class__.__name__} did not load all "
            f"required {scale_name} values."
        )
    if not torch.all(torch.isfinite(scale)) or torch.any(scale <= 0):
        raise ValueError(f"ModelOpt FP8 {scale_name} must contain positive values.")


def _requantize_weight(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    dequantized = (weight.to(torch.float16) * source_scale).to(torch.float16)
    return static_scaled_fp8_quant(dequantized, target_scale)


class ModelOptFp8Config(QuantizationConfig):
    """Configuration for ModelOpt serialized static per-tensor FP8."""

    def __init__(self, exclude_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self.exclude_modules = exclude_modules or []

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        if not isinstance(config, dict):
            raise ValueError("ModelOpt FP8 quantization config must be a mapping.")

        nested = config.get("quantization")
        if isinstance(nested, dict):
            quant = nested
            exclude_modules = quant.get("exclude_modules", [])
        else:
            quant = config
            exclude_modules = quant.get("ignore", [])

        quant_algo = str(quant.get("quant_algo", "")).upper()
        quant_method = str(config.get("quant_method", "modelopt")).lower()
        producer = config.get("producer", {})
        producer_name = (
            str(producer.get("name", "modelopt")).lower()
            if isinstance(producer, dict)
            else ""
        )
        if quant_method not in ("", "modelopt"):
            raise ValueError(
                "ModelOpt FP8 requires quant_method='modelopt'; "
                f"got {quant_method!r}."
            )
        if producer_name not in ("", "modelopt"):
            raise ValueError(
                "ModelOpt FP8 requires a ModelOpt-produced checkpoint; "
                f"got producer {producer_name!r}."
            )
        if quant_algo != "FP8":
            raise ValueError(
                "ModelOpt FP8 only supports quant_algo='FP8'; "
                f"got {quant_algo or '<missing>'!r}."
            )

        kv_algo = quant.get("kv_cache_quant_algo")
        kv_scheme = quant.get("kv_cache_scheme")
        if kv_algo not in (None, "") or kv_scheme not in (None, {}):
            raise ValueError("ModelOpt FP8 KV-cache quantization is not supported.")

        config_groups = quant.get("config_groups")
        if config_groups is not None:
            if not isinstance(config_groups, dict) or not config_groups:
                raise ValueError(
                    "ModelOpt FP8 config_groups must be a non-empty mapping."
                )
            for group in config_groups.values():
                if not isinstance(group, dict):
                    raise ValueError(
                        "Each ModelOpt FP8 config group must be a mapping."
                    )
                _validate_static_float8_scheme("weights", group.get("weights", {}))
                _validate_static_float8_scheme(
                    "input activations", group.get("input_activations", {})
                )

        if not isinstance(exclude_modules, list):
            raise ValueError("ModelOpt FP8 ignore/exclude_modules must be a list.")

        logger.info("Using ModelOpt serialized static per-tensor FP8 weights.")
        return cls(exclude_modules=list(exclude_modules))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if user_quant not in (None, "auto", "modelopt_fp8"):
            return None
        try:
            cls.from_config(hf_quant_cfg)
        except ValueError:
            return None
        return "modelopt_fp8"

    def _is_excluded(self, prefix: str) -> bool:
        normalized = prefix.removeprefix("language_model.")
        return any(
            module in prefix or module in normalized for module in self.exclude_modules
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from fluxserve.backend.layers.linear import LinearBase
        from fluxserve.backend.layers.moe.fused_moe_triton import FusedMoE

        if self._is_excluded(prefix):
            return None
        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return ModelOptFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Dense FP8 linear using a static activation scale and scaled_mm."""

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        for scale_name in ("weight_scale", "input_scale"):
            layer.register_parameter(
                scale_name,
                PerTensorScaleParameter(
                    data=torch.full(
                        (len(output_partition_sizes),),
                        _UNLOADED_SCALE,
                        dtype=torch.float32,
                    ),
                    weight_loader=weight_loader,
                ),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _validate_loaded_scale(layer, "weight_scale")
        _validate_loaded_scale(layer, "input_scale")

        combined_scale = layer.weight_scale.max()
        offset = 0
        for width, source_scale in zip(layer.logical_widths, layer.weight_scale):
            shard = layer.weight.data.narrow(0, offset, width)
            shard.copy_(_requantize_weight(shard, source_scale, combined_scale))
            offset += width

        # scaled_mm expects the RHS in column-major layout.
        layer.weight = Parameter(layer.weight.data.t(), requires_grad=False)
        layer.weight_scale = Parameter(combined_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        padded_rows = 17 if x_2d.shape[0] < 17 else None
        quantized_input = static_scaled_fp8_quant(
            x_2d, layer.input_scale, padded_rows=padded_rows
        )
        output = torch._scaled_mm(
            quantized_input,
            layer.weight,
            scale_a=layer.input_scale,
            scale_b=layer.weight_scale,
            out_dtype=x.dtype,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = output[: x_2d.shape[0]]
        output = output.reshape(*original_shape[:-1], layer.output_size_per_partition)
        if bias is not None:
            output = output + bias
        return output


class ModelOptFp8MoEMethod(FusedMoEMethodBase):
    """ModelOpt static per-tensor FP8 for the existing Triton MoE GEMM."""

    def __init__(self, quant_config: ModelOptFp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        from fluxserve.backend.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.register_parameter(
            "w13_weight",
            ModelWeightParameter(
                data=torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=2,
                output_dim=1,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "w2_weight",
            ModelWeightParameter(
                data=torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=2,
                output_dim=1,
                weight_loader=weight_loader,
            ),
        )

        extra_weight_attrs["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value
        layer.register_parameter(
            "w13_weight_scale",
            PerTensorScaleParameter(
                data=torch.full((num_experts, 2), _UNLOADED_SCALE, dtype=torch.float32),
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "w2_weight_scale",
            PerTensorScaleParameter(
                data=torch.full((num_experts,), _UNLOADED_SCALE, dtype=torch.float32),
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "w13_input_scale",
            PerTensorScaleParameter(
                data=torch.full((num_experts,), _UNLOADED_SCALE, dtype=torch.float32),
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "w2_input_scale",
            PerTensorScaleParameter(
                data=torch.full((num_experts,), _UNLOADED_SCALE, dtype=torch.float32),
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for scale_name in (
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_input_scale",
            "w2_input_scale",
        ):
            _validate_loaded_scale(layer, scale_name)

        combined_scales = layer.w13_weight_scale.max(dim=1).values
        intermediate_size = layer.w13_weight.shape[1] // 2
        for expert_id in range(layer.w13_weight.shape[0]):
            for shard_id in range(2):
                start = shard_id * intermediate_size
                shard = layer.w13_weight.data[
                    expert_id, start : start + intermediate_size
                ]
                shard.copy_(
                    _requantize_weight(
                        shard,
                        layer.w13_weight_scale[expert_id, shard_id],
                        combined_scales[expert_id],
                    )
                )

        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w13_weight_scale = Parameter(combined_scales, requires_grad=False)
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        layer.w13_input_scale = Parameter(
            layer.w13_input_scale.max(), requires_grad=False
        )
        layer.w2_input_scale = Parameter(
            layer.w2_input_scale.max(), requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        del layer
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8_w8a8=True,
            per_channel_quant=False,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)


__all__ = [
    "ModelOptFp8Config",
    "ModelOptFp8LinearMethod",
    "ModelOptFp8MoEMethod",
]
