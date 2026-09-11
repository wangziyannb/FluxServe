"""Compare paired GSM8K outcomes and verify complete experimental artifacts."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(line) for line in p.read_text().splitlines()]

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    manifest=read(OUT/'manifest.json')
    expected=[r['id'] for r in rows(OUT/'data/test.jsonl')]
    results={}
    for case in manifest['cases']:
        path=OUT/'measurements'/case['name']
        if not (path/'result.json').exists():continue
        result=read(path/'result.json')
        if result['status']!='passed':continue
        predictions=rows(path/'predictions.jsonl')
        assert [r['id'] for r in predictions]==expected
        result['predictions']=predictions
        results[case['name']]=result
    def find(w,kv,cal,n,g):
        return next((r for r in results.values() if (r['case']['weights'],r['case']['kv'],r['case']['calibration'],r['case']['gpus'],r['case']['graph'])==(w,kv,cal,n,g)),None)
    comparisons=[]
    def compare(label,baseline,candidate):
        if baseline is None or candidate is None:return
        b,c=baseline['metrics'],candidate['metrics']
        pairs=list(zip(baseline['predictions'],candidate['predictions'],strict=True))
        gained=[a['id'] for a,z in pairs if not a['correct'] and z['correct']]
        lost=[a['id'] for a,z in pairs if a['correct'] and not z['correct']]
        changed=[a['id'] for a,z in pairs if a['answer']!=z['answer']]
        discordant=len(gained)+len(lost)
        # Exact two-sided binomial McNemar test; descriptive, not adjusted for multiplicity.
        p=min(1.0,2*sum(math.comb(discordant,k) for k in range(min(len(gained),len(lost))+1))/2**discordant) if discordant else 1.0
        comparisons.append(dict(comparison=label,baseline=baseline['case']['name'],candidate=candidate['case']['name'],
                                accuracy_delta_pp=100*(c['accuracy']-b['accuracy']),tps_ratio=c['tps']/b['tps'],
                                nfe_ratio=c['nfe']/b['nfe'],tokens_ratio=c['generated_tokens']/b['generated_tokens'],
                                gained=len(gained),lost=len(lost),changed_outputs=len(changed),
                                mcnemar_exact_p_unadjusted=p,gained_ids=gained,lost_ids=lost,changed_output_ids=changed))
    for g in (False,True):
        compare('weight_quantization',find('bf16','bf16','none',4,g),find('fp8','bf16','none',4,g))
        for n in (2,4):
            compare('calibration_data_gsm8k_vs_ultrachat',find('fp8','fp8_e4m3','ultrachat',n,g),find('fp8','fp8_e4m3','gsm8k-train',n,g))
            for cal in ('ultrachat','gsm8k-train'):
                compare('kv_quantization',find('fp8','bf16','none',n,g),find('fp8','fp8_e4m3',cal,n,g))
        for cal in ('ultrachat','gsm8k-train'):
            compare('joint_quantization',find('bf16','bf16','none',4,g),find('fp8','fp8_e4m3',cal,4,g))
            compare('gpu_scaling',find('fp8','fp8_e4m3',cal,2,g),find('fp8','fp8_e4m3',cal,4,g))
        compare('gpu_scaling',find('fp8','bf16','none',2,g),find('fp8','bf16','none',4,g))
    for case in manifest['cases']:
        if case['graph']:continue
        c=case
        compare('cuda_graph',find(c['weights'],c['kv'],c['calibration'],c['gpus'],False),find(c['weights'],c['kv'],c['calibration'],c['gpus'],True))
    for c in comparisons:
        if c['comparison']=='gpu_scaling':c['scaling_efficiency']=c['tps_ratio']/2
    save(OUT/'comparisons.json',comparisons)
    if comparisons:
        csv_rows=[{k:v for k,v in c.items() if not k.endswith('_ids')} for c in comparisons]
        with (OUT/'comparisons.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in csv_rows for k in r)))
            writer.writeheader();writer.writerows(csv_rows)
    scale_rows=[]
    a_path,b_path=OUT/'scales-ultrachat.json',OUT/'scales-gsm8k-train.json'
    if a_path.exists() and b_path.exists():
        a,b=read(a_path),read(b_path)
        for i in range(32):
            for kind in ('k','v'):
                u=a['layers'][str(i)][kind+'_scale'];s=b['layers'][str(i)][kind+'_scale']
                scale_rows.append(dict(layer=i,kind=kind,ultrachat=u,gsm8k_train=s,ratio=s/u))
        save(OUT/'scale-comparison.json',scale_rows)
    save(OUT/'analysis-status.json',dict(completed=len(results),expected=len(manifest['cases']),comparisons=len(comparisons)))
    if len(results)==len(manifest['cases']):
        intervals=[]
        total_tokens=0
        for name,result in results.items():
            directory=OUT/'measurements'/name
            command=read(directory/'command.json')
            process=read(directory/'process.json')
            metrics=result['metrics']
            assert process['returncode']==0
            assert not read(directory/'before.json')['processes'].strip()
            assert not read(directory/'after.json')['processes'].strip()
            assert len(process['observed_host_pids'])==result['case']['gpus']
            assert metrics['samples']==1319
            assert metrics['correct']==sum(r['correct'] for r in result['predictions'])
            assert math.isclose(metrics['tps'],metrics['generated_tokens']/metrics['generation_seconds'])
            batches=read(directory/'batches.json')
            assert len(batches)==165 and batches[-1]['start_idx']==1312
            assert sum(len(r['token_counts']) for r in batches)==1319
            assert sum(r['nfe'] for r in batches)==metrics['nfe']
            tokens=rows(directory/'completion-tokens.jsonl')
            assert [r['id'] for r in tokens]==expected
            for rank in metrics['ranks']:
                assert metrics['generation_seconds']+1e-6>=rank['local_generation_seconds']
                if result['case']['graph']:
                    assert rank['flashinfer_graph']['decode_replay_count']>0
                    assert all(v==0 for v in rank['flashinfer_graph_during_generation'].values())
            intervals.append((command['started'],process['finished'],name))
            total_tokens+=metrics['generated_tokens']
        intervals.sort()
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        save(OUT/'verification.json',dict(status='passed',complete_cases=len(results),test_questions_per_case=1319,
                                          total_scored_answers=len(results)*1319,total_tokens=total_tokens,
                                          no_overlapping_gpu_runs=True,original_request_order=True,
                                          timing='synchronized max rank per batch; graph captures/invalidation zero during generation',
                                          comparisons=len(comparisons),measurements_per_case=1))
    print(f'{len(results)}/{len(manifest["cases"])} complete; {len(comparisons)} paired comparisons')

if __name__=='__main__':main()
