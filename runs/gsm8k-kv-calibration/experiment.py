"""Serial full-model calibration and GSM8K accuracy/throughput experiment."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PYTHON = REPO / '.venv/bin/python'
EVAL_PYTHON = OUT / '.eval-venv/bin/python'
spec = importlib.util.spec_from_file_location('throughput_common', REPO / 'runs/quant-throughput/evaluate.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
save, read, digest, stamp = common.save, common.read, common.digest, common.stamp
CASES = [dict(weights=w, kv=kv, calibration=cal, gpus=n, graph=g,
              name=f"{w}-{kv}-{cal}-{n}gpu-{'graph' if g else 'eager'}")
         for w,kv,cal,counts in [('bf16','bf16','none',[4]), ('fp8','bf16','none',[2,4]),
                                ('fp8','fp8_e4m3','ultrachat',[2,4]), ('fp8','fp8_e4m3','gsm8k-train',[2,4])]
         for n in counts for g in (False,True)]

def command(case, directory, dataset, calibration=False):
    models = read(REPO / 'runs/quant-throughput/models.json')
    argv = [str(PYTHON), str(OUT / 'bench_entry.py'), 'calibrate_kv_cache' if calibration else 'bench_offline',
            '--model', models[case['weights']]['path'], '--quantization', 'modelopt_fp8' if case['weights']=='fp8' else 'auto',
            '--kv-cache-dtype', case['kv'], '--dataset', str(dataset), '--dataset-format','openai',
            '--batch-size','8','--mini-batch-size','4','--gen-len','2048','--block-length','64',
            '--prefilling-limit','128','--threshold','0.95','--low-threshold','0.3','--parallel-decoding','threshold',
            '--disable-sorting','--tp-size',str(case['gpus']),'--ep-size',str(case['gpus']),'--dp-size','1','--pp-size','1',
            '--output-dir',str(directory),'--exp-name','run','--log-file','run.log']
    if calibration:
        argv += ['--num-samples','512','--output',str(OUT / f"scales-{case['calibration']}.json")]
    else:
        argv += ['--attention-backend','flashinfer','--flashinfer-prefill-mode','paged','--flashinfer-cache-mode','paged',
                 '--kv-cache-layout','paged','--page-size','64','--flashinfer-decode-batch-mode','max_batch',
                 '--cuda-graph-capture-sizes','64','128','256','512','1024']
        if case['kv']=='fp8_e4m3':
            argv += ['--kv-cache-scales',str(OUT / f"scales-{case['calibration']}.json")]
        if case['graph']:
            argv += ['--use-cuda-graph']
    return argv

def prepare():
    data = read(OUT / 'data-manifest.json')
    assert data['evaluation']['samples']==1319 and data['calibration_samples']==512
    files = subprocess.check_output(['git','ls-files','python','flux-kernel/python','flux-scheduler'],cwd=REPO,text=True).splitlines()
    sources = {p:digest(REPO / p) for p in files if (REPO / p).is_file()}
    for name in ['bench_entry.py','prepare.py']:
        sources[str((OUT/name).relative_to(REPO))] = digest(OUT / name)
    manifest = dict(created=stamp(), commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                    source_hashes=sources, data=data, models=read(REPO/'runs/quant-throughput/models.json'),
                    environment={k:v for k,v in common.environment(4).items() if k in common.environment_keys()},
                    cases=CASES, seed=42, repeat_count=1, timing='sum of synchronized maximum rank generation time per batch',
                    output_difference_policy='allowed; report per-problem accuracy, lengths and NFE',
                    inference='original bench_offline with prepared input IDs and output-only instrumentation',
                    graph_decode_mode='decomposed', question_count=1319,
                    infeasible=[dict(weights=w,kv=kv,gpus=n,status='memory_infeasible',basis='checkpoint_weight_capacity')
                                for w,kv,counts in [('bf16','bf16',[1,2]),('fp8','bf16',[1]),('fp8','fp8_e4m3',[1])] for n in counts])
    save(OUT / 'manifest.json',manifest)
    (OUT / 'source.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    print('Prepared 14 configurations; two training-only calibrations, 1319 test questions each',flush=True)

def check():
    manifest = read(OUT/'manifest.json')
    for p,expected in manifest['source_hashes'].items():
        assert digest(REPO/p)==expected, f'Source changed: {p}'
    for p,expected in manifest['data']['files'].items():
        assert digest(OUT/p)==expected, f'Data changed: {p}'

def calibrate():
    from transformers import AutoConfig
    from fluxserve.cli import _resolve_quant_config
    from fluxserve.backend.layers.kv_quantization import KVQuantizationConfig
    for cal in ('ultrachat','gsm8k-train'):
        check()
        directory = OUT/'calibration'/cal
        scales = OUT/f'scales-{cal}.json'
        if (directory/'validation.json').exists():
            assert digest(scales)==read(directory/'validation.json')['scale_sha256']
            continue
        case = dict(weights='fp8',kv='bf16',gpus=4,graph=False,calibration=cal)
        dataset = OUT/'data'/f'calibration-{cal}.jsonl'
        print(f'{stamp()} Calibrating {cal}: 512 samples',flush=True)
        common.execute(command(case,directory,dataset,True),directory,4,timeout=14400)
        config = AutoConfig.from_pretrained(read(OUT/'manifest.json')['models']['fp8']['path'],trust_remote_code=True,local_files_only=True)
        config.quant_config = _resolve_quant_config(config,'modelopt_fp8')
        quant = KVQuantizationConfig.load(config,'fp8_e4m3',str(scales))
        payload = read(scales)
        assert len(quant.scales)==32 and payload['calibration']['num_samples']==512
        assert payload['calibration']['dataset_sha256']==digest(dataset)
        expected_ids=[json.loads(line)['id'] for line in dataset.read_text().splitlines()]
        assert payload['calibration']['sample_ids']==expected_ids
        save(directory/'validation.json',dict(status='passed',scale_sha256=digest(scales),layers=32,
                                             calibration_dataset_sha256=digest(dataset),samples=512))
        print(f'{stamp()} Calibration validated: {cal}',flush=True)

def validate(case,directory,dataset):
    metrics = read(directory/'run_metrics.json')
    assert metrics['kv_cache_dtype']==case['kv']
    assert metrics['weight_format']==('modelopt_fp8' if case['weights']=='fp8' else 'bf16')
    assert metrics['num_hidden_layers']==32 and len(metrics['ranks'])==case['gpus']
    assert metrics['nfe']>0 and metrics['generation_seconds']>0 and math.isfinite(metrics['tps'])
    if case['kv']=='fp8_e4m3':
        scales=OUT/f"scales-{case['calibration']}.json"
        assert metrics['kv_scale_source']==str(scales)
        assert digest(scales)==read(OUT/'calibration'/case['calibration']/'validation.json')['scale_sha256']
    for rank in metrics['ranks']:
        assert (rank['parameter_bytes_by_dtype'].get('torch.float8_e4m3fn',0)>0)==(case['weights']=='fp8')
        assert rank['kv_data_bytes']>0 and rank['local_generation_seconds']>0
        stats=rank['flashinfer_graph']
        replays=rank['generic_graph_replays']+stats.get('decode_replay_count',0)+stats.get('replay_count',0)
        assert (replays>0)==case['graph']
        if case['graph']:
            assert all(v==0 for v in rank['flashinfer_graph_during_generation'].values()), 'Timed capture/invalidation'
    inputs=[json.loads(line) for line in dataset.read_text().splitlines()]
    answers=[json.loads(line) for line in (directory/f'run_{dataset.stem}_threshold_0.95.jsonl').read_text().splitlines()]
    assert [r['id'] for r in answers]==[r['id'] for r in inputs]
    assert all(a['prompt']==b['prompt'] for a,b in zip(answers,inputs))
    assert sum(r['generated_length'] for r in answers)==metrics['generated_tokens']
    batches=read(directory/'batches.json')
    assert sum(b['nfe'] for b in batches)==metrics['nfe']
    assert sum(b['tokens'] for b in batches)==metrics['generated_tokens']
    assert math.isclose(sum(b['seconds'] for b in batches),metrics['generation_seconds'])
    assert len(batches)==math.ceil(len(inputs)/8)
    tokens=[json.loads(line) for line in (directory/'completion-tokens.jsonl').read_text().splitlines()]
    assert [r['id'] for r in tokens]==[r['id'] for r in inputs]
    metrics.update(samples=len(inputs),eos_missing=sum(not r['eos_found'] for r in tokens),
                   outputs_with_masks=sum(r['masks_before_eos']>0 for r in tokens),
                   output_sha256=digest(directory/f'run_{dataset.stem}_threshold_0.95.jsonl'),
                   tokens_per_forward=metrics['generated_tokens']/metrics['nfe'])
    metrics.update(read(directory/'process.json'))
    return metrics

def run_case(case,stage):
    check()
    directory=OUT/stage/case['name']
    if (directory/'result.json').exists():
        return read(directory/'result.json')
    dataset=OUT/'data'/('smoke.jsonl' if stage=='smoke' else 'test.jsonl')
    result=dict(case=case,started=stamp())
    print(f"{stamp()} Starting {stage}: {case['name']}",flush=True)
    try:
        common.execute(command(case,directory,dataset),directory,case['gpus'],timeout=14400)
        metrics=validate(case,directory,dataset)
        subprocess.run([str(EVAL_PYTHON),str(OUT/'score.py'),str(directory),str(dataset)],check=True,
                       stdout=(directory/'scoring.log').open('w'),stderr=subprocess.STDOUT)
        metrics.update(read(directory/'accuracy.json'))
        result.update(status='passed',metrics=metrics)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    result['finished']=stamp()
    save(directory/'result.json',result)
    print(f"{stamp()} Finished {case['name']}: {result['status']} "+str(result.get('error', {k:result['metrics'][k] for k in ['accuracy','tps','nfe']} if 'metrics' in result else '')),flush=True)
    return result

def report():
    groups=[]
    for case in CASES:
        path=OUT/'measurements'/case['name']/'result.json'
        if not path.exists():
            groups.append({**case,'status':'pending'})
            continue
        r=read(path)
        group={**case,'status':r['status']}
        if r['status']=='passed':
            m=r['metrics']
            group.update({k:m[k] for k in ['accuracy','correct','samples','tps','generated_tokens','nfe','generation_seconds','eos_missing','tokens_per_forward']})
            group['per_gpu_tps']=m['tps']/case['gpus']
            group['peak_device_gib']=max(m['peak_device_used_bytes'].values())/2**30
        else:
            group['error']=r.get('error')
        groups.append(group)
    save(OUT/'summary.json',dict(groups=groups,repeat_count=1,updated=stamp()))
    fields=list(dict.fromkeys(k for row in groups for k in row))
    with (OUT/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(groups)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','calibrate','smoke','measure','report','all'])
    parser.add_argument('--case')
    args=parser.parse_args()
    if args.stage in ('prepare','all'):prepare()
    if args.stage in ('calibrate','all'):calibrate()
    cases=[c for c in CASES if args.case is None or c['name']==args.case]
    if args.stage in ('smoke','all'):
        # Validate one eager and one graph path for each precision/scale/GPU combination.
        for case in cases:run_case(case,'smoke')
    if args.stage in ('measure','all'):
        # Interleave eager and graph order across adjacent groups.
        ordered=[]
        for i in range(0,len(cases),2):ordered.extend(cases[i:i+2] if i%4==0 else cases[i:i+2][::-1])
        for case in ordered:
            trial=read(OUT/'smoke'/case['name']/'result.json')
            if trial['status']=='passed':run_case(case,'measurements')
            report()
    if args.stage in ('report','all'):report()

if __name__=='__main__':main()
