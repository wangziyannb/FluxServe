import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fluxserve.backend.layers.attention import utils


def checker():
    path = Path(__file__).resolve().parents[2] / "docker/check_flashinfer.py"
    spec = importlib.util.spec_from_file_location("flashinfer_build_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_with_parameters(names):
    function = Mock()
    function.__signature__ = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names.split()
    ])
    return function


def api():
    def wrapper():
        return SimpleNamespace(
            __init__=function_with_parameters(
                "kv_layout backend use_cuda_graph block_extend block_size q_offsets_buf kv_offsets_buf"
            ),
            plan=function_with_parameters(
                "q_offsets kv_offsets q_data_type kv_data_type sm_scale custom_mask packed_custom_mask"
            ),
            run=function_with_parameters("k_scale v_scale"),
        )
    return SimpleNamespace(
        __git_commit__="tested-revision", __version__="0.6.18",
        BatchPrefillWithPagedKVCacheWrapper=wrapper(),
        BatchPrefillWithRaggedKVCacheWrapper=wrapper(),
        segment_packbits=Mock(), append_paged_kv_cache=Mock(), get_batch_indices_positions=Mock(),
    ), function_with_parameters(
        "input token_selected_experts token_final_scales fc1_expert_weights fc2_expert_weights "
        "output_dtype quant_scales output tp_size tp_rank ep_size ep_rank tune_max_num_tokens "
        "enable_pdl activation_type"
    ), SimpleNamespace(Swiglu=object())


def test_build_contract_checks_all_paths_without_running_kernels():
    flashinfer, moe, activation = api()
    result = checker().check_apis(flashinfer, moe, activation, "tested-revision")
    assert result["paged_block_extend"] and result["ragged_block_extend"]
    assert result["fp8_kv_scale_api"] and result["nvfp4_moe_api"]
    assert result["validation"] == "imports_and_signatures_only"
    moe.assert_not_called()
    flashinfer.BatchPrefillWithPagedKVCacheWrapper.__init__.assert_not_called()


@pytest.mark.parametrize("missing", ["block_extend", "q_offsets_buf", "kv_offsets", "v_scale", "quant_scales"])
def test_build_contract_rejects_missing_attention_or_moe_api(missing):
    flashinfer, moe, activation = api()
    functions = [moe]
    for wrapper in (flashinfer.BatchPrefillWithPagedKVCacheWrapper, flashinfer.BatchPrefillWithRaggedKVCacheWrapper):
        functions.extend((wrapper.__init__, wrapper.plan, wrapper.run))
    for function in functions:
        params = function.__signature__.parameters
        if missing in params:
            function.__signature__ = inspect.Signature([p for name, p in params.items() if name != missing])
    with pytest.raises(RuntimeError, match=f"lacks {missing}"):
        checker().check_apis(flashinfer, moe, activation)


def test_build_contract_rejects_different_revision():
    with pytest.raises(RuntimeError, match="revision mismatch"):
        checker().check_apis(*api(), expected_revision="another-revision")


def test_ragged_adapter_uses_public_offsets_and_replans_changed_scale(monkeypatch):
    flashinfer, _, _ = api()
    instance = flashinfer.BatchPrefillWithRaggedKVCacheWrapper
    constructor = Mock()

    class Wrapper:
        def __init__(self, workspace, *, kv_layout, block_extend, block_size, backend):
            constructor(workspace, kv_layout=kv_layout, block_extend=block_extend,
                        block_size=block_size, backend=backend)

        plan = staticmethod(instance.plan)
        run = staticmethod(instance.run)

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=Wrapper))
    adapter = utils.BatchBlockExtendRaggedOffsetWrapper(torch.empty(1), dllm_block_size=64)
    assert constructor.call_args.kwargs == {"kv_layout": "NHD", "block_extend": True, "block_size": 64, "backend": "auto"}
    q_offsets = torch.tensor([64], dtype=torch.int32)
    kv_offsets = torch.tensor([0], dtype=torch.int32)
    kwargs = dict(
        qo_indptr=torch.tensor([0, 64], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 128], dtype=torch.int32),
        q_offsets=q_offsets, kv_offsets=kv_offsets,
        num_qo_heads=4, num_kv_heads=2, head_dim=128,
        q_data_type=torch.bfloat16, sm_scale=0.1,
    )
    adapter.plan(**kwargs)
    planned = instance.plan.call_args.kwargs
    assert planned["head_dim_qk"] == 128 and "head_dim" not in planned
    assert planned["kv_data_type"] == torch.bfloat16
    assert planned["q_offsets"] is q_offsets and planned["kv_offsets"] is kv_offsets
    assert planned["sm_scale"] == 0.1 and not planned["causal"]
    adapter.plan(**kwargs)
    assert instance.plan.call_count == 1
    adapter.plan(**{**kwargs, "sm_scale": 0.2})
    assert instance.plan.call_count == 2
    assert instance.plan.call_args.kwargs["sm_scale"] == 0.2
