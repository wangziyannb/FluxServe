"""Serial, resumable full GSM8K matrix with explicit attention precision/backend."""
import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'runs/gsm8k-kv-calibration'
spec = importlib.util.spec_from_file_location('previous_gsm', OLD / 'experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
save, read, digest, stamp = exp.save, exp.read, exp.digest, exp.stamp
SCALE_FP8 = OLD / 'scales-ultrachat.json'
SCALE_BF16 = OUT / 'scales-ultrachat-bf16-model.json'
CASES = []
for weights, kv, compute, backend, counts in [
    ('fp8', 'fp8_e4m3', 'fp8', 'fa3', [2, 4]),
    ('bf16', 'bf16', 'bf16', 'fa3', [4]),
    ('bf16', 'fp8_e4m3', 'fp8', 'fa3', [4]),
    ('bf16', 'bf16', 'bf16', 'fa2', [4]),
]:
    for count in counts:
        for graph in (False, True):
            CASES.append(dict(weights=weights, kv=kv, attention_compute=compute,
                              kernel_backend=backend, gpus=count, graph=graph,
                              calibration='ultrachat' if kv == 'fp8_e4m3' else 'none',
                              name=f'{weights}-kv-{kv}-attn-{compute}-{backend}-{count}gpu-{"graph" if graph else "eager"}'))


def scale_for(case):
    return SCALE_FP8 if case['weights'] == 'fp8' else SCALE_BF16


def command(case, directory, dataset, calibration=False):
    cmd = exp.command(case, directory, dataset, calibration)
    if calibration:
        cmd[cmd.index('--output') + 1] = str(SCALE_BF16)
    else:
        cmd += ['--attention-compute-dtype', case['attention_compute'],
                '--flashinfer-kernel-backend', case['kernel_backend']]
        if case['kv'] == 'fp8_e4m3':
            cmd[cmd.index('--kv-cache-scales') + 1] = str(scale_for(case))
    return cmd


def prepare():
    if (OUT / 'manifest.json').exists():
        raise RuntimeError('Manifest already exists; preserve the recorded experiment identity')
    data = read(OLD / 'data-manifest.json')
    paths = [p for folder in ('python/fluxserve', 'flux-kernel/python/flux_kernel') for p in (ROOT / folder).rglob('*.py')]
    paths += [Path(__file__), OLD / 'bench_entry.py', OLD / 'experiment.py', OLD / 'score.py',
              ROOT / 'runs/quant-throughput/evaluate.py']
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    inputs = {str((OLD / p).relative_to(ROOT)): data['files'][p] for p in (
        'data/test.jsonl', 'data/smoke.jsonl', 'data/gold.jsonl', 'data/fewshot.json',
        'data/calibration-ultrachat.jsonl')}
    inputs[str(SCALE_FP8.relative_to(ROOT))] = read(OLD / 'calibration/ultrachat/validation.json')['scale_sha256']
    models = read(ROOT / 'runs/quant-throughput/models.json')
    for model in models.values():
        path = Path(model['path'])
        for p in path.iterdir():
            if p.suffix in ('.json', '.py', '.jinja', '.txt'):
                inputs[str(p)] = digest(p)
    payload = dict(created=stamp(), commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   cases=CASES, sources=sources, inputs=inputs, models=models,
                   evaluation=data['evaluation'], dataset_revision=data['sources']['gsm8k-test']['revision'],
                   calibration_source=data['sources']['ultrachat'], calibration_samples=512,
                   calibration_method=data['calibration_method'], calibration_selection=data['calibration_selection'],
                   gsm8k_calibration_used=False, seed=42, repeat_count=1,
                   timing='synchronized maximum rank generation time per batch; prefill and decode included',
                   output_differences='allowed; report tokens, NFE, answer/token disagreement and accuracy',
                   precision_label='weights / KV storage / QK+PV compute; FP8 compute retains higher precision softmax/accumulation and BF16 output',
                   environment={k:v for k,v in exp.common.environment(4).items() if k in exp.common.environment_keys()},
                   capacity_exclusions=['BF16 weights on 1 or 2 H100 80GB', 'FP8 weights on 1 H100 80GB'],
                   comparisons=['4-GPU FA3 full FP8 versus BF16 weights/KV/attention',
                                '4-GPU BF16 weights/KV/attention FA3 versus FA2, identical block/page adapter',
                                'full FP8 2 versus 4 GPUs', 'Graph versus eager within each combination'],
                   unsupported='FA3 BF16 Q with FP8 KV; BF16 weights + FP8 KV uses FP8 attention')
    save(OUT / 'manifest.json', payload)
    (OUT / 'source.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True))
    with tarfile.open(OUT / 'source-snapshot.tar.gz', 'w:gz') as archive:
        for relative in sources:
            archive.add(ROOT / relative, arcname=relative, recursive=False)
    check()
    report()


def check():
    manifest = read(OUT / 'manifest.json')
    for section in ('sources', 'inputs'):
        for p, expected in manifest[section].items():
            assert digest(ROOT / p) == expected, f'{section} changed: {p}'


def calibrate_bf16():
    check()
    directory = OUT / 'calibration-bf16-ultrachat'
    validation = directory / 'validation.json'
    if validation.exists():
        assert digest(SCALE_BF16) == read(validation)['scale_sha256']
        return
    if (directory / 'process.json').exists():
        raise RuntimeError('Prior calibration attempt exists; review before retrying')
    case = dict(weights='bf16', kv='bf16', calibration='ultrachat', gpus=4, graph=False)
    dataset = OLD / 'data/calibration-ultrachat.jsonl'
    save(OUT / 'status.json', dict(state='calibrating', case='bf16-model-ultrachat', updated=stamp(), pid=os.getpid()))
    print(stamp(), 'Calibrating BF16 model on UltraChat 512', flush=True)
    exp.common.execute(command(case, directory, dataset, True), directory, 4, timeout=21600)
    from transformers import AutoConfig
    from fluxserve.cli import _resolve_quant_config
    from fluxserve.backend.layers.kv_quantization import KVQuantizationConfig
    config = AutoConfig.from_pretrained(read(OUT / 'manifest.json')['models']['bf16']['path'],
                                       trust_remote_code=True, local_files_only=True)
    config.quant_config = _resolve_quant_config(config, 'auto')
    quant = KVQuantizationConfig.load(config, 'fp8_e4m3', str(SCALE_BF16))
    payload = read(SCALE_BF16)
    assert len(quant.scales) == len(payload['layers']) == 32
    assert payload['calibration']['num_samples'] == 512
    assert payload['calibration']['dataset_sha256'] == digest(dataset)
    assert payload['calibration']['sample_ids'] == [json.loads(line)['id'] for line in dataset.read_text().splitlines()]
    save(validation, dict(status='passed', model='bf16', samples=512, layers=32,
                          dataset_sha256=digest(dataset), scale_sha256=digest(SCALE_BF16)))
    check()


def validate(case, directory, dataset):
    metrics = read(directory / 'run_metrics.json')
    assert metrics['timing'] == 'synchronized_perf_counter_max_rank_per_batch'
    assert metrics['weight_format'] == ('modelopt_fp8' if case['weights'] == 'fp8' else 'bf16')
    assert metrics['kv_cache_dtype'] == case['kv']
    assert metrics['attention_compute_dtype'] == case['attention_compute']
    assert metrics['flashinfer_kernel_backend'] == case['kernel_backend']
    assert metrics['attention_kernel'] == f"flashinfer-{case['kernel_backend']}-{case['attention_compute']}"
    assert metrics['num_hidden_layers'] == 32 and len(metrics['ranks']) == case['gpus']
    assert metrics['nfe'] > 0 and metrics['generation_seconds'] > 0 and math.isfinite(metrics['tps'])
    dtype = 'torch.float8_e4m3fn' if case['attention_compute'] == 'fp8' else 'torch.bfloat16'
    expected_kernel = dict(backend=case['kernel_backend'], query_dtype=dtype, kv_dtype=dtype)
    for rank in metrics['ranks']:
        assert rank['observed_block_attention_kernels'] == [expected_kernel]
        assert (rank['parameter_bytes_by_dtype'].get('torch.float8_e4m3fn', 0) > 0) == (case['weights'] == 'fp8')
        assert rank['kv_data_bytes'] > 0 and rank['local_generation_seconds'] > 0
        stats = rank['flashinfer_graph']
        replays = rank['generic_graph_replays'] + stats.get('decode_replay_count', 0) + stats.get('replay_count', 0)
        assert (replays > 0) == case['graph']
        if case['graph']:
            assert all(v == 0 for v in rank['flashinfer_graph_during_generation'].values()), 'Timed capture/invalidation'
    if case['kv'] == 'fp8_e4m3':
        scale = scale_for(case)
        assert metrics['kv_scale_source'] == str(scale)
        validation = (OLD / 'calibration/ultrachat/validation.json' if case['weights'] == 'fp8'
                      else OUT / 'calibration-bf16-ultrachat/validation.json')
        assert digest(scale) == read(validation)['scale_sha256']
        metrics['kv_scale_sha256'] = digest(scale)
    inputs = [json.loads(s) for s in dataset.read_text().splitlines()]
    answers_path = directory / f'run_{dataset.stem}_threshold_0.95.jsonl'
    answers = [json.loads(s) for s in answers_path.read_text().splitlines()]
    tokens = [json.loads(s) for s in (directory / 'completion-tokens.jsonl').read_text().splitlines()]
    assert [a['id'] for a in inputs] == [a['id'] for a in answers] == [a['id'] for a in tokens]
    assert all(a['prompt'] == b['prompt'] for a,b in zip(answers,inputs))
    assert sum(a['generated_length'] for a in answers) == metrics['generated_tokens']
    batches = read(directory / 'batches.json')
    assert len(batches) == math.ceil(len(inputs) / 8)
    assert sum(b['nfe'] for b in batches) == metrics['nfe']
    assert sum(b['tokens'] for b in batches) == metrics['generated_tokens']
    assert math.isclose(sum(b['seconds'] for b in batches), metrics['generation_seconds'])
    metrics.update(samples=len(inputs), eos_missing=sum(not r['eos_found'] for r in tokens),
                   outputs_with_masks=sum(r['masks_before_eos'] > 0 for r in tokens),
                   output_sha256=digest(answers_path), tokens_per_forward=metrics['generated_tokens']/metrics['nfe'],
                   forwards_per_second=metrics['nfe']/metrics['generation_seconds'])
    metrics.update(read(directory / 'process.json'))
    assert metrics['returncode'] == 0
    assert (directory / 'after.json').exists()
    return metrics


def run_case(case, stage):
    check()
    directory = OUT / stage / case['name']
    result_file = directory / 'result.json'
    if result_file.exists():
        result = read(result_file)
        if result['status'] != 'passed':
            raise RuntimeError(f"Review failed attempt before retry: {directory}")
        return result
    if (directory / 'command.json').exists():
        raise RuntimeError(f'Review interrupted attempt before retry: {directory}')
    dataset = OLD / 'data' / ('smoke.jsonl' if stage == 'smoke' else 'test.jsonl')
    result = dict(case=case, stage=stage, started=stamp())
    save(OUT / 'status.json', dict(state='running', stage=stage, case=case, updated=stamp(), pid=os.getpid()))
    print(stamp(), 'Starting', stage, case['name'], flush=True)
    try:
        exp.common.execute(command(case, directory, dataset), directory, case['gpus'], timeout=21600)
        check()
        metrics = validate(case, directory, dataset)
        with (directory / 'scoring.log').open('w') as log:
            subprocess.run([str(exp.EVAL_PYTHON), str(OLD / 'score.py'), str(directory), str(dataset)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        metrics.update(read(directory / 'accuracy.json'))
        result.update(status='passed', metrics=metrics)
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    result['finished'] = stamp()
    save(result_file, result)
    report()
    print(stamp(), 'Finished', stage, case['name'], result['status'],
          {k:result['metrics'][k] for k in ('accuracy','tps','generated_tokens','nfe')} if result['status']=='passed' else result['error'], flush=True)
    if result['status'] != 'passed':
        raise RuntimeError(result['error'])
    return result


def report():
    groups=[]
    for case in CASES:
        path=OUT/'measurements'/case['name']/'result.json'
        row={**case,'status':'pending'}
        if path.exists():
            result=read(path)
            row['status']=result['status']
            if result['status']=='passed':
                m=result['metrics']
                row.update({k:m[k] for k in ('accuracy','correct','samples','tps','generated_tokens','nfe','generation_seconds','eos_missing','tokens_per_forward','forwards_per_second')})
                row.update(per_gpu_tps=m['tps']/case['gpus'],peak_device_gib=max(m['peak_device_used_bytes'].values())/2**30)
            else: row['error']=result['error']
        groups.append(row)
    save(OUT/'summary.json',dict(updated=stamp(),repeat_count=1,groups=groups))
    with (OUT/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in groups for k in row)))
        writer.writeheader();writer.writerows(groups)
    ratios=[]
    def compare(label,a,b):
        if a['status']!='passed' or b['status']!='passed':return
        value=dict(comparison=label,numerator=a['name'],denominator=b['name'],
                   tps_ratio=a['tps']/b['tps'],token_ratio=a['generated_tokens']/b['generated_tokens'],
                   nfe_ratio=a['nfe']/b['nfe'],forwards_per_second_ratio=a['forwards_per_second']/b['forwards_per_second'],
                   accuracy_delta_pp=100*(a['accuracy']-b['accuracy']))
        if label=='2-to-4-gpu':value['scaling_efficiency']=value['tps_ratio']/2
        a_tokens=[json.loads(s) for s in (OUT/'measurements'/a['name']/'completion-tokens.jsonl').read_text().splitlines()]
        b_tokens=[json.loads(s) for s in (OUT/'measurements'/b['name']/'completion-tokens.jsonl').read_text().splitlines()]
        value['different_token_sequences']=sum(x['token_ids']!=y['token_ids'] for x,y in zip(a_tokens,b_tokens))
        value['different_completion_lengths']=sum(len(x['token_ids'])!=len(y['token_ids']) for x,y in zip(a_tokens,b_tokens))
        ratios.append(value)
    def find(**kw):return next(r for r in groups if all(r[k]==v for k,v in kw.items()))
    for graph in (False,True):
        full=find(weights='fp8',gpus=4,graph=graph)
        bf3=find(weights='bf16',kv='bf16',kernel_backend='fa3',graph=graph)
        bf2=find(weights='bf16',kv='bf16',kernel_backend='fa2',graph=graph)
        mix=find(weights='bf16',kv='fp8_e4m3',graph=graph)
        compare('full-fp8-versus-full-bf16-fa3',full,bf3)
        compare('fa3-versus-fa2-bf16-identical-adapter',bf3,bf2)
        compare('kv-and-attention-fp8-fa3',mix,bf3)
        compare('fp8-weights-with-fp8-kv-and-attention-fa3',full,mix)
        compare('2-to-4-gpu',full,find(weights='fp8',gpus=2,graph=graph))
    for row in groups:
        if row['graph']:
            compare('graph-versus-eager',row,find(**{k:row[k] for k in ('weights','kv','attention_compute','kernel_backend','gpus')},graph=False))
    save(OUT/'comparisons.json',ratios)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','smoke','run','report'])
    parser.add_argument('--case')
    args=parser.parse_args()
    with (OUT/'experiment.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='prepare':prepare();return
        if args.stage=='report':report();return
        selected=[c for c in CASES if args.case is None or c['name']==args.case]
        assert selected, 'Unknown case'
        try:
            for case in selected:
                if case['weights']=='bf16' and case['kv']=='fp8_e4m3':calibrate_bf16()
                run_case(case,'smoke')
                if args.stage=='run':run_case(case,'measurements')
            report()
            save(OUT/'status.json',dict(state='complete',stage=args.stage,selected_cases=len(selected),updated=stamp(),pid=os.getpid()))
        except Exception as exc:
            save(OUT/'status.json',dict(state='failed',error=str(exc),updated=stamp(),pid=os.getpid()))
            raise

if __name__=='__main__':main()
