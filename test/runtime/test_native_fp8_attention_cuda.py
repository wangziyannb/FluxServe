"""Hopper FP8 compute accuracy, dynamic queries, masks and captured schedules."""

import pytest
import torch

from flux_kernel.ops.fp8_attention import quantize_fp8_query
from fluxserve.backend.layers.attention.native_fp8 import NativeFP8PagedWrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper required",
)
FP8 = torch.float8_e4m3fn


@pytest.mark.parametrize("n,h,d", [(1, 8, 128), (171, 16, 128), (2048, 8, 64)])
def test_query_quantization_dynamic_scales_and_graph(n, h, d):
    torch.manual_seed(9)
    q = torch.randn(n, h, d, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):
        quantize_fp8_query(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        encoded, scales = quantize_fp8_query(q)
    for multiplier in (1.0, 0.0, 19.0):
        q.normal_().mul_(multiplier)
        q[:, 0].zero_()
        graph.replay()
        maximum = q.float().abs().amax((0, 2))
        reference_scales = torch.where(maximum > 0, maximum / 448, 1)
        reference = (q.float() / reference_scales[None, :, None]).clamp(-448, 448).to(FP8)
        torch.testing.assert_close(scales, reference_scales, rtol=1e-6, atol=0)
        assert torch.equal(encoded.view(torch.uint8), reference.view(torch.uint8))
    graph.reset()


def reference(q, cache, indptr, indices, metadata, scales, page=64):
    q_lens, kv_lens, offsets, kv_offsets = metadata
    maximum = q.float().abs().amax((0, 2))
    qs = torch.where(maximum > 0, maximum / 448, 1)
    qr = (q.float() / qs[None, :, None]).to(FP8).float() * qs[None, :, None]
    outputs = []
    cursor = 0
    for row, n in enumerate(q_lens):
        kv = []
        for array, scale in zip(cache, scales):
            pages = array.float()[indices[indptr[row]:indptr[row + 1]].long()]
            kv.append((pages.flatten(0, 1)[:kv_lens[row]] * scale).repeat_interleave(q.shape[1] // array.shape[2], 1))
        mask = ((torch.arange(n, device=q.device) + offsets[row])[:, None] // 64
                >= (torch.arange(kv_lens[row], device=q.device) + kv_offsets[row])[None, :] // 64)
        outputs.append(torch.nn.functional.scaled_dot_product_attention(
            qr[cursor:cursor+n].transpose(0, 1), kv[0].transpose(0, 1),
            kv[1].transpose(0, 1), attn_mask=mask).transpose(0, 1))
        cursor += n
    return torch.cat(outputs)


def assert_fp8_close(actual, expected):
    # FA3 also rounds softmax probabilities for the FP8 PV tensor-core product.
    # Bound both aggregate relative error and worst absolute error; this is an
    # operator test, not a claim of checkpoint quality equivalence.
    error = actual.float() - expected
    assert error.norm() / expected.norm().clamp_min(1e-8) < 0.04
    assert error.abs().max() < 0.08


@pytest.mark.parametrize("heads,kv_heads,d", [(32, 4, 128), (16, 2, 128), (8, 1, 128), (8, 1, 64)])
def test_fp8_paged_prefill_offsets_partial_pages_and_gqa(heads, kv_heads, d):
    torch.manual_seed(13)
    tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    scales = (0.013, 0.027)
    cache = tuple((torch.randn(9, 64, kv_heads, d, device="cuda") / s).clamp(-448, 448).to(FP8) for s in scales)
    indptr, indices, last = tensor([0, 3, 6]), tensor([6, 2, 4, 0, 7, 1]), tensor([21, 9])
    metadata = ((100, 71), (149, 137), (37, 66), (0, 5))
    q = torch.randn(171, heads, d, dtype=torch.bfloat16, device="cuda")
    wrapper = NativeFP8PagedWrapper(torch.empty(128*1024*1024, dtype=torch.uint8, device="cuda"))
    wrapper.plan(metadata=metadata, kv_indptr=indptr, kv_indices=indices, last_page_len=last,
                 page_size=64, block_length=64, num_qo_heads=heads, num_kv_heads=kv_heads,
                 head_dim=d, sm_scale=d**-0.5)
    output = wrapper.run(q, cache, k_scale=scales[0], v_scale=scales[1])
    assert output.dtype == torch.bfloat16
    assert_fp8_close(output, reference(q, cache, indptr, indices, metadata, scales))


@pytest.mark.parametrize("d", [64, 128])
def test_fp8_graph_updates_lengths_pages_scales_without_replanning(monkeypatch, d):
    torch.manual_seed(15)
    tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    scales = (0.019, 0.031)
    cache = tuple((torch.randn(12, 64, 1, d, device="cuda") / s).clamp(-448, 448).to(FP8) for s in scales)
    indptr, indices, last = tensor([0, 4, 8]), tensor(list(range(8))), tensor([64, 64])
    q = torch.randn(128, 8, d, dtype=torch.bfloat16, device="cuda")
    wrapper = NativeFP8PagedWrapper(torch.empty(128*1024*1024, dtype=torch.uint8, device="cuda"))
    wrapper.plan(metadata=((64, 64), (256, 256), (192, 192), (0, 0)),
                 kv_indptr=indptr, kv_indices=indices, last_page_len=last,
                 page_size=64, block_length=64, num_qo_heads=8, num_kv_heads=1,
                 head_dim=d, sm_scale=d**-0.5, use_cuda_graph=True)
    for _ in range(2):
        wrapper.run(q, cache, k_scale=scales[0], v_scale=scales[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = wrapper.run(q, cache, k_scale=scales[0], v_scale=scales[1])
    monkeypatch.setattr(wrapper.wrapper, "plan", lambda *a, **kw: pytest.fail("Replanned during replay"))
    for lengths, starts in [((64, 129), (0, 5)), ((256, 128), (7, 1)), ((128, 256), (2, 6))]:
        counts = [(n+63)//64 for n in lengths]
        ptr = tensor([0, counts[0], sum(counts)])
        idx = tensor([i for start, count in zip(starts, counts) for i in range(start, start+count)])
        tail = tensor([(n-1)%64+1 for n in lengths])
        offsets = tuple(((n-1)//64)*64 for n in lengths)
        wrapper.update_metadata(ptr, idx, tail, tensor(offsets), tensor([0, 0]))
        q.normal_().mul_(1.3)
        graph.replay()
        metadata = ((64, 64), lengths, offsets, (0, 0))
        assert_fp8_close(output, reference(q, cache, ptr, idx, metadata, scales))
    graph.reset()


@pytest.mark.parametrize('backend', ['fa2', 'fa3'])
def test_bf16_backends_partial_pages_and_dynamic_graph(backend, monkeypatch, request):
    # FlashInfer backend changes in one process can hang after FP8 Graph tests.
    # Use the same fixed-backend process boundary as the full-model benchmark.
    import os
    import subprocess
    import sys
    flag = 'FLUXSERVE_TEST_BF16_BLOCK_CHILD'
    if os.environ.get(flag) != '1':
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'{__file__}::{request.node.name}'],
            env={**os.environ, flag: '1'}, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stdout
        return
    from fluxserve.backend.layers.attention.native_fp8 import BlockPagedWrapper
    torch.manual_seed(16)
    tensor = lambda x: torch.tensor(x, device='cuda', dtype=torch.int32)
    cache = tuple(torch.randn(12, 64, 1, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    indptr, indices, last = tensor([0, 4, 8]), tensor(list(range(8))), tensor([64, 64])
    q = torch.randn(128, 8, 128, dtype=torch.bfloat16, device='cuda')
    wrapper = BlockPagedWrapper(torch.empty(128*1024*1024, dtype=torch.uint8, device='cuda'),
                                backend=backend, compute_dtype='bf16')
    wrapper.plan(metadata=((64,64), (256,256), (192,192), (0,0)),
                 kv_indptr=indptr, kv_indices=indices, last_page_len=last,
                 page_size=64, block_length=64, num_qo_heads=8, num_kv_heads=1,
                 head_dim=128, sm_scale=128**-0.5, use_cuda_graph=True)
    for _ in range(2): wrapper.run(q, cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output = wrapper.run(q, cache)
    monkeypatch.setattr(wrapper.wrapper, 'plan', lambda *a, **kw: pytest.fail('Replanned during replay'))
    for lengths, starts in [((64,129),(0,5)), ((256,128),(7,1)), ((128,256),(2,6))]:
        counts = [(n+63)//64 for n in lengths]
        ptr = tensor([0,counts[0],sum(counts)])
        idx = tensor([i for start,count in zip(starts,counts) for i in range(start,start+count)])
        tail = tensor([(n-1)%64+1 for n in lengths])
        offsets = tuple(((n-1)//64)*64 for n in lengths)
        wrapper.update_metadata(ptr,idx,tail,tensor(offsets),tensor([0,0]))
        q.normal_()
        graph.replay()
        outputs=[]
        for row,n in enumerate(lengths):
            kv=[array.float()[idx[ptr[row]:ptr[row+1]].long()].flatten(0,1)[:n].repeat_interleave(8,1) for array in cache]
            mask=((torch.arange(64,device='cuda')+offsets[row])[:,None]//64 >= torch.arange(n,device='cuda')[None,:]//64)
            outputs.append(torch.nn.functional.scaled_dot_product_attention(
                q[row*64:(row+1)*64].float().transpose(0,1),kv[0].transpose(0,1),kv[1].transpose(0,1),attn_mask=mask).transpose(0,1))
        torch.testing.assert_close(output.float(),torch.cat(outputs),rtol=0.01,atol=0.003)
    graph.reset()
