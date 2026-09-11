"""Isolate attention (including Q quantization), without model/decode changes."""
import json
from pathlib import Path
import statistics

import flashinfer
import torch
from fluxserve.backend.layers.attention.native_fp8 import NativeFP8PagedWrapper

torch.manual_seed(42)
rows=[]
device='cuda'
def tensor(x):return torch.tensor(x,device=device,dtype=torch.int32)

for heads,kv_heads in [(16,2),(8,1)]:
    for length in [1024,2048,4096]:
        b,n,d=4,64,128
        pages=length//64
        q=torch.randn(b*n,heads,d,device=device,dtype=torch.bfloat16)
        scales=(0.019,0.031)
        kv=tuple((torch.randn(b*pages,64,kv_heads,d,device=device)/s).clamp(-448,448).to(torch.float8_e4m3fn) for s in scales)
        qi,ki=tensor([i*n for i in range(b+1)]),tensor([i*pages for i in range(b+1)])
        indices,last=tensor(list(range(b*pages))),tensor([64]*b)
        workspace=torch.empty(128*1024*1024,device=device,dtype=torch.uint8)
        legacy=flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace,backend='fa2',kv_layout='NHD')
        legacy.plan(qi,ki,indices,last,heads,kv_heads,d,64,
                    custom_mask=torch.ones(b*n*length,device=device,dtype=torch.bool),
                    q_data_type=torch.bfloat16,kv_data_type=torch.float8_e4m3fn,
                    disable_split_kv=True)
        native=NativeFP8PagedWrapper(workspace)
        native.plan(metadata=((n,)*b,(length,)*b,(length-n,)*b,(0,)*b),
                    kv_indptr=ki,kv_indices=indices,last_page_len=last,page_size=64,block_length=64,
                    num_qo_heads=heads,num_kv_heads=kv_heads,head_dim=d,sm_scale=d**-0.5,
                    use_cuda_graph=True)
        result=dict(query_heads=heads,kv_heads=kv_heads,batch=b,q_length=n,kv_length=length)
        for name,wrapper in [('bf16_compute',legacy),('fp8_compute',native)]:
            def run():return wrapper.run(q,kv,k_scale=scales[0],v_scale=scales[1],enable_pdl=False)
            for _ in range(3):run()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=run()
            for _ in range(10):graph.replay()
            times=[]
            for _ in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):graph.replay()
                end.record();end.synchronize()
                times.append(start.elapsed_time(end)*1000/100)
            result[name+'_us']=statistics.median(times)
            result[name+'_samples_us']=times
            graph.reset()
        result['speedup']=result['bf16_compute_us']/result['fp8_compute_us']
        rows.append(result);print(json.dumps(result),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
