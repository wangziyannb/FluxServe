# Copyright (c) 2026 FLUX-OSS

"""ModelOpt serialized static NVFP4 inference support."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from flux_kernel import (
    interleave_nvfp4_block_scale,
    static_scaled_nvfp4_quant,
)
from torch.nn.parameter import Parameter

from fluxserve.backend.layers.moe import MoeRunnerConfig
from fluxserve.backend.layers.parameter import (
    BlockQuantScaleParameter,
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
_NVFP4_BLOCK_SIZE = 16


def _validate_nvfp4_scheme(name: str, scheme: Dict[str, Any]) -> None:
    if scheme.get("type") != "float" or scheme.get("num_bits") != 4:
        raise ValueError(
            f"ModelOpt NVFP4 requires {name} to use 4-bit float quantization."
        )
    if scheme.get("dynamic") is not False:
        raise ValueError(
            f"ModelOpt NVFP4 only supports serialized static {name}; "
            "dynamic quantization is not supported."
        )
    if scheme.get("group_size") != _NVFP4_BLOCK_SIZE:
        raise ValueError("ModelOpt NVFP4 requires group_size=16.")


def _validate_global_scale(layer: torch.nn.Module, scale_name: str) -> None:
    scale = getattr(layer, scale_name, None)
    if scale is None:
        raise ValueError(f"ModelOpt NVFP4 layer is missing {scale_name}.")
    scale_float = scale.float()
    if torch.any(scale_float == _UNLOADED_SCALE):
        raise ValueError(
            f"ModelOpt NVFP4 layer {layer.__class__.__name__} did not load all "
            f"required {scale_name} values."
        )
    if not torch.all(torch.isfinite(scale_float)) or torch.any(scale_float <= 0):
        raise ValueError(f"ModelOpt NVFP4 {scale_name} must contain positive values.")


def _validate_block_scale(layer: torch.nn.Module, scale_name: str) -> None:
    scale = getattr(layer, scale_name, None)
    if scale is None or scale.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"ModelOpt NVFP4 {scale_name} must use torch.float8_e4m3fn."
        )
    scale_float = scale.float()
    if not torch.all(torch.isfinite(scale_float)) or torch.any(scale_float < 0):
        raise ValueError(
            f"ModelOpt NVFP4 {scale_name} must contain non-negative values."
        )


def _rescale_block_scales(
    scales: torch.Tensor,
    ratios: torch.Tensor,
) -> torch.Tensor:
    rescaled = scales.float() * ratios.float()
    rescaled = torch.where(
        rescaled == 0,
        rescaled,
        rescaled.clamp(min=2**-9, max=448.0),
    )
    return rescaled.to(torch.float8_e4m3fn)


def _swizzle_moe_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert linear ModelOpt block scales to FlashInfer CUTLASS layout."""
    if scale.ndim != 3 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 MoE block scales must be rank-3 FP8 tensors.")

    experts, rows, columns = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    if padded_rows != rows or padded_columns != columns:
        padded = torch.zeros(
            (experts, padded_rows, padded_columns),
            dtype=scale.dtype,
            device=scale.device,
        )
        padded[:, :rows, :columns].copy_(scale)
        scale = padded

    return (
        scale.reshape(
            experts,
            padded_rows // 128,
            4,
            32,
            padded_columns // 4,
            4,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, padded_rows, padded_columns)
    )


class ModelOptNvfp4Config(QuantizationConfig):
    """Configuration for ModelOpt serialized static NVFP4 W4A4."""

    weight_block_size = (1, _NVFP4_BLOCK_SIZE)

    def __init__(self, exclude_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self.exclude_modules = exclude_modules or []

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptNvfp4Config":
        if not isinstance(config, dict):
            raise ValueError("ModelOpt NVFP4 quantization config must be a mapping.")

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
                "ModelOpt NVFP4 requires quant_method='modelopt'; "
                f"got {quant_method!r}."
            )
        if producer_name not in ("", "modelopt"):
            raise ValueError(
                "ModelOpt NVFP4 requires a ModelOpt-produced checkpoint; "
                f"got producer {producer_name!r}."
            )
        if quant_algo != "NVFP4":
            raise ValueError(
                "ModelOpt NVFP4 only supports quant_algo='NVFP4'; "
                f"got {quant_algo or '<missing>'!r}."
            )

        kv_algo = quant.get("kv_cache_quant_algo")
        kv_scheme = quant.get("kv_cache_scheme")
        if kv_algo not in (None, "") or kv_scheme not in (None, {}):
            raise ValueError("ModelOpt NVFP4 KV-cache quantization is not supported.")

        config_groups = quant.get("config_groups")
        if config_groups is not None:
            if not isinstance(config_groups, dict) or not config_groups:
                raise ValueError(
                    "ModelOpt NVFP4 config_groups must be a non-empty mapping."
                )
            for group in config_groups.values():
                if not isinstance(group, dict):
                    raise ValueError(
                        "Each ModelOpt NVFP4 config group must be a mapping."
                    )
                _validate_nvfp4_scheme("weights", group.get("weights", {}))
                _validate_nvfp4_scheme(
                    "input activations", group.get("input_activations", {})
                )

        group_size = quant.get("group_size")
        if group_size is not None and group_size != _NVFP4_BLOCK_SIZE:
            raise ValueError("ModelOpt NVFP4 requires group_size=16.")
        if not isinstance(exclude_modules, list):
            raise ValueError("ModelOpt NVFP4 ignore/exclude_modules must be a list.")

        logger.info("Using ModelOpt serialized static NVFP4 W4A4 weights.")
        return cls(exclude_modules=list(exclude_modules))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if user_quant not in (None, "auto", "modelopt_nvfp4"):
            return None
        try:
            cls.from_config(hf_quant_cfg)
        except ValueError:
            return None
        return "modelopt_nvfp4"

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
            return ModelOptNvfp4LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return ModelOptNvfp4MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelOptNvfp4LinearMethod(LinearMethodBase):
    """Dense ModelOpt NVFP4 linear backed by block-scaled cuBLASLt."""

    def __init__(self, quant_config: ModelOptNvfp4Config) -> None:
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
        if input_size_per_partition % _NVFP4_BLOCK_SIZE != 0:
            raise ValueError("ModelOpt NVFP4 linear input size must be divisible by 16.")

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
                    input_size_per_partition // 2,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            BlockQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // _NVFP4_BLOCK_SIZE,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        for scale_name in ("weight_scale_2", "input_scale"):
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
        _validate_block_scale(layer, "weight_scale")
        _validate_global_scale(layer, "weight_scale_2")
        _validate_global_scale(layer, "input_scale")

        combined_weight_scale_2 = layer.weight_scale_2.max()
        combined_input_scale = layer.input_scale.max()
        offset = 0
        weight_scale = layer.weight_scale.data
        for width, source_scale_2 in zip(
            layer.logical_widths, layer.weight_scale_2
        ):
            rows = weight_scale.narrow(0, offset, width)
            rows.copy_(
                _rescale_block_scales(
                    rows,
                    source_scale_2 / combined_weight_scale_2,
                )
            )
            offset += width

        packed_weight = layer.weight.data.view(torch.float4_e2m1fn_x2)
        layer.weight = Parameter(packed_weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(
            interleave_nvfp4_block_scale(weight_scale), requires_grad=False
        )
        layer.weight_scale_2 = Parameter(
            combined_weight_scale_2, requires_grad=False
        )
        layer.input_scale = Parameter(combined_input_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        quantized_input, input_block_scale = static_scaled_nvfp4_quant(
            x_2d, layer.input_scale, blocked_scale=True
        )
        output = torch._scaled_mm(
            quantized_input,
            layer.weight,
            scale_a=input_block_scale,
            scale_b=layer.weight_scale,
            out_dtype=x.dtype,
        )
        if isinstance(output, tuple):
            output = output[0]
        output.mul_(layer.input_scale * layer.weight_scale_2)
        output = output.reshape(*original_shape[:-1], layer.output_size_per_partition)
        if bias is not None:
            output = output + bias
        return output


class ModelOptNvfp4MoEMethod(FusedMoEMethodBase):
    """ModelOpt NVFP4 backed by FlashInfer's Blackwell CUTLASS MoE kernel."""

    def __init__(self, quant_config: ModelOptNvfp4Config) -> None:
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
        if (
            hidden_size % _NVFP4_BLOCK_SIZE != 0
            or intermediate_size_per_partition % _NVFP4_BLOCK_SIZE != 0
        ):
            raise ValueError("ModelOpt NVFP4 MoE dimensions must be divisible by 16.")

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
                    hidden_size // 2,
                    dtype=torch.uint8,
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
                    intermediate_size_per_partition // 2,
                    dtype=torch.uint8,
                ),
                input_dim=2,
                output_dim=1,
                weight_loader=weight_loader,
            ),
        )

        extra_weight_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter(
            "w13_weight_scale",
            BlockQuantScaleParameter(
                data=torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // _NVFP4_BLOCK_SIZE,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=2,
                output_dim=1,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "w2_weight_scale",
            BlockQuantScaleParameter(
                data=torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // _NVFP4_BLOCK_SIZE,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=2,
                output_dim=1,
                weight_loader=weight_loader,
            ),
        )
        for name, shape in (
            ("w13_weight_scale_2", (num_experts, 2)),
            ("w2_weight_scale_2", (num_experts,)),
            ("w13_input_scale", (num_experts, 2)),
            ("w2_input_scale", (num_experts,)),
        ):
            layer.register_parameter(
                name,
                PerTensorScaleParameter(
                    data=torch.full(shape, _UNLOADED_SCALE, dtype=torch.float32),
                    weight_loader=weight_loader,
                ),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for scale_name in ("w13_weight_scale", "w2_weight_scale"):
            _validate_block_scale(layer, scale_name)
        for scale_name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            _validate_global_scale(layer, scale_name)

        gate_scale_2 = layer.w13_weight_scale_2[:, 1]
        up_scale_2 = layer.w13_weight_scale_2[:, 0]
        if not torch.equal(gate_scale_2, up_scale_2):
            raise ValueError(
                "FlashInfer CUTLASS NVFP4 MoE requires identical gate/up "
                "weight_scale_2 values."
            )

        w13_input_scale = layer.w13_input_scale.max()
        w2_input_scale = layer.w2_input_scale.max()
        layer.w13_weight = Parameter(
            layer.w13_weight.data.contiguous(), requires_grad=False
        )
        layer.w2_weight = Parameter(
            layer.w2_weight.data.contiguous(), requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            _swizzle_moe_block_scale(layer.w13_weight_scale.data),
            requires_grad=False,
        )
        layer.w2_weight_scale = Parameter(
            _swizzle_moe_block_scale(layer.w2_weight_scale.data),
            requires_grad=False,
        )
        layer.w13_input_scale = Parameter(w13_input_scale, requires_grad=False)
        layer.w2_input_scale = Parameter(w2_input_scale, requires_grad=False)
        layer.w13_dequant_scale = Parameter(
            (w13_input_scale * up_scale_2).float().contiguous(),
            requires_grad=False,
        )
        layer.w2_dequant_scale = Parameter(
            (w2_input_scale * layer.w2_weight_scale_2).float().contiguous(),
            requires_grad=False,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        del layer
        if moe_runner_config.activation != "silu":
            raise ValueError("ModelOpt NVFP4 MoE currently supports SiLU only.")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError(
                "ModelOpt NVFP4 MoE does not support applying router weights "
                "before GEMM1."
            )
        if moe_runner_config.no_combine:
            raise ValueError("ModelOpt NVFP4 MoE requires fused expert combine.")
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        # Individual SM120 CUTLASS translation units can consume several GiB
        # while FlashInfer performs its one-time JIT build.
        os.environ.setdefault("MAX_JOBS", "4")
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from fluxserve.backend.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from fluxserve.backend.layers.moe.topk import TopKOutputChecker

        if not TopKOutputChecker.format_is_standard(dispatch_output.topk_output):
            raise ValueError("ModelOpt NVFP4 MoE requires standard top-k output.")

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights
        tune_tokens = max(1, 1 << (hidden_states.shape[0] - 1).bit_length())
        output_buffer = torch.empty_like(hidden_states)
        output = cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=layer.w13_weight.view(torch.long),
            fc2_expert_weights=layer.w2_weight.view(torch.long),
            output_dtype=hidden_states.dtype,
            quant_scales=[
                layer.w13_input_scale.reciprocal(),
                layer.w13_weight_scale.view(torch.int32),
                layer.w13_dequant_scale,
                layer.w2_input_scale.reciprocal(),
                layer.w2_weight_scale.view(torch.int32),
                layer.w2_dequant_scale,
            ],
            output=output_buffer,
            tp_size=layer.moe_tp_size,
            tp_rank=layer.moe_tp_rank,
            ep_size=layer.moe_ep_size,
            ep_rank=layer.moe_ep_rank,
            tune_max_num_tokens=tune_tokens,
            enable_pdl=False,
            activation_type=ActivationType.Swiglu,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
        if routed_scaling_factor is not None:
            output.mul_(routed_scaling_factor)
        return StandardCombineInput(hidden_states=output)


__all__ = [
    "ModelOptNvfp4Config",
    "ModelOptNvfp4LinearMethod",
    "ModelOptNvfp4MoEMethod",
]
