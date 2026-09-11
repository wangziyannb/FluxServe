import torch,flashinfer,json
from pathlib import Path
torch.manual_seed(2)
d='cuda';fp8=torch.float8_e4m3fn
q=torch.randn(128,8,128,device=d,dtype=torch.bfloat16)
k=torch.randn(12,64,1,128,device=d,dtype=torch.bfloat16)
v=torch.randn_like(k)
qs=q.float().abs().amax((0,2)).clamp_min(1e-6)/448
ks=torch.full((1,),0.02,device=d);vs=torch.full((1,),0.017,device=d)
q8=(q.float()/qs[None,:,None]).to(fp8)
k8=(k.float()/ks).clamp(-448,448).to(fp8);v8=(v.float()/vs).clamp(-448,448).to(fp8)
qi=torch.tensor([0,64,128],device=d,dtype=torch.int32)
ki=torch.tensor([0,4,8],device=d,dtype=torch.int32)
idx=torch.arange(8,device=d,dtype=torch.int32)
last=torch.tensor([64,64],device=d,dtype=torch.int32)
w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128*1024*1024,device=d,dtype=torch.uint8),kv_layout='NHD',backend='fa3',use_cuda_graph=True,qo_indptr_buf=qi,paged_kv_indptr_buf=ki,paged_kv_indices_buf=idx,paged_kv_last_page_len_buf=last)
w.plan(qi,ki,idx,last,8,1,128,64,q_data_type=fp8,kv_data_type=fp8,o_data_type=torch.bfloat16,causal=False)
from fluxserve.backend.layers.attention.native_fp8 import bind_fa3_schedule
from flux_kernel.ops.fp8_attention import refresh_fa3_schedule
schedule=bind_fa3_schedule(w,2)
def run():return w.run(q8,(k8,v8),qs,ks,vs,enable_pdl=False)
for _ in range(2):o=run()
torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):out=run()
for lengths in [(256,256),(64,129),(128,64)]:
    pages=[(x+63)//64 for x in lengths]
    ki.copy_(torch.tensor([0,pages[0],sum(pages)],device=d,dtype=torch.int32))
    idx[:sum(pages)].copy_(torch.tensor(list(range(pages[0]))+list(range(4,4+pages[1])),device=d,dtype=torch.int32))
    last.copy_(torch.tensor([(x-1)%64+1 for x in lengths],device=d,dtype=torch.int32))
    refresh_fa3_schedule(ki,last,schedule,64)
    g.replay();torch.cuda.synchronize()
    refs=[]
    for b,n in enumerate(lengths):
        kref=(k8[b*4:b*4+pages[b]].float()*ks).reshape(-1,1,128)[:n].repeat_interleave(8,1)
        vref=(v8[b*4:b*4+pages[b]].float()*vs).reshape(-1,1,128)[:n].repeat_interleave(8,1)
        qr=q8[b*64:(b+1)*64].float()*qs[None,:,None]
        refs.append(torch.nn.functional.scaled_dot_product_attention(qr.transpose(0,1),kref.transpose(0,1),vref.transpose(0,1)).transpose(0,1))
    ref=torch.cat(refs)
    print(json.dumps({'lengths':lengths,'rmse':(out.float()-ref).square().mean().sqrt().item(),'max_error':(out.float()-ref).abs().max().item()}),flush=True)
    torch.testing.assert_close(out.float(),ref,rtol=.05,atol=.02)
print('FP8 FA3 dynamic CSR replay passed',flush=True)
