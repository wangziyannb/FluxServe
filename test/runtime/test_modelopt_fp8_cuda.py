from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from flux_kernel import static_scaled_fp8_quant
from fluxserve.backend.layers.moe.fused_moe_triton import override_config
from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts_impl,
)
from fluxserve.backend.layers.quantization.modelopt_fp8 import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)


def _requires_cuda():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@torch.no_grad()
def test_fp8_linear_matches_reference_and_cuda_graph():
    if not _requires_cuda():
        return

    torch.manual_seed(7)
    input_ = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(48, 64, device="cuda", dtype=torch.float16)
    input_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor(0.015, device="cuda", dtype=torch.float32)

    layer = nn.Module()
    layer.logical_widths = [48]
    layer.output_size_per_partition = 48
    layer.weight = nn.Parameter(
        static_scaled_fp8_quant(weight, weight_scale), requires_grad=False
    )
    layer.weight_scale = nn.Parameter(weight_scale.reshape(1), requires_grad=False)
    layer.input_scale = nn.Parameter(input_scale.reshape(1), requires_grad=False)
    method = ModelOptFp8LinearMethod(ModelOptFp8Config())
    method.process_weights_after_loading(layer)

    eager = method.apply(layer, input_)
    quantized_input = static_scaled_fp8_quant(input_, input_scale)
    reference = (
        quantized_input.float()
        .mul(input_scale)
        .matmul(layer.weight.float().mul(layer.weight_scale))
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(eager, reference, rtol=0.08, atol=0.5)

    static_input = input_.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = method.apply(layer, static_input)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager, graph_output)


@torch.no_grad()
def test_small_fp8_moe_matches_dequantized_reference():
    if not _requires_cuda():
        return

    torch.manual_seed(11)
    tokens, experts, hidden, intermediate = 4, 2, 64, 64
    input_ = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(
        experts, 2 * intermediate, hidden, device="cuda", dtype=torch.float16
    )
    w2 = torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.float16)
    a13_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    a2_scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)
    w13_scale = torch.full((experts,), 0.015, device="cuda")
    w2_scale = torch.full((experts,), 0.015, device="cuda")
    qw13 = torch.stack(
        [static_scaled_fp8_quant(w13[e], w13_scale[e]) for e in range(experts)]
    )
    qw2 = torch.stack(
        [static_scaled_fp8_quant(w2[e], w2_scale[e]) for e in range(experts)]
    )
    topk_ids = torch.tensor([[0], [1], [0], [1]], device="cuda")
    topk_weights = torch.ones(tokens, 1, device="cuda", dtype=torch.float32)

    with override_config(
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }
    ):
        actual = fused_experts_impl(
            input_,
            qw13,
            qw2,
            topk_weights,
            topk_ids,
            use_fp8_w8a8=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            a1_scale=a13_scale,
            a2_scale=a2_scale,
        )

    qinput = static_scaled_fp8_quant(input_, a13_scale)
    expected = torch.empty_like(input_)
    for token, expert in enumerate(topk_ids.flatten().tolist()):
        first = (
            qinput[token].float().mul(a13_scale)
            @ qw13[expert].float().mul(w13_scale[expert]).t()
        ).to(torch.bfloat16)
        gate, up = first.chunk(2)
        activated = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        qactivated = static_scaled_fp8_quant(activated.reshape(1, -1), a2_scale)[0]
        expected[token] = (
            qactivated.float().mul(a2_scale)
            @ qw2[expert].float().mul(w2_scale[expert]).t()
        ).to(torch.bfloat16)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.12, atol=2.0)
