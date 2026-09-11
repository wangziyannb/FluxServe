"""Small full-model acceptance set; does not replace full GSM8K evaluation."""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path('/workspace/FluxServe')
OUT = ROOT / 'runs/fp8-attention'
spec = importlib.util.spec_from_file_location('gsm_experiment', ROOT / 'runs/gsm8k-kv-calibration/experiment.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
dataset = exp.OUT / 'data/smoke.jsonl'

paths = sorted(p for folder in ['python/fluxserve', 'flux-kernel/python/flux_kernel']
               for p in (ROOT/folder).rglob('*.py'))
sources = {str(p.relative_to(ROOT)): exp.digest(p) for p in paths}
exp.save(OUT/'source-hashes.json', sources)
summary = []
for count, compute, graph in [(2, 'fp8', False), (2, 'fp8', True), (2, 'bf16', True),
                               (4, 'fp8', True), (4, 'bf16', True)]:
    name = f'{count}gpu-{compute}-compute-{"graph" if graph else "eager"}'
    case = dict(weights='fp8', kv='fp8_e4m3', calibration='ultrachat', gpus=count, graph=graph, name=name)
    directory = OUT / name
    assert all(exp.digest(ROOT/p) == digest for p, digest in sources.items()), 'Runtime sources changed'
    if (directory/'result.json').exists():
        summary.append(exp.read(directory/'result.json'))
        continue
    print(exp.stamp(), 'Starting', name, flush=True)
    command = exp.command(case, directory, dataset) + ['--attention-compute-dtype', compute]
    exp.common.execute(command, directory, count, timeout=2400)
    metrics = exp.validate(case, directory, dataset)
    assert metrics['attention_compute_dtype'] == compute
    if compute == 'fp8':
        assert metrics['attention_kernel'] == 'flashinfer-fa3-fp8'
    with (directory/'scoring.log').open('w') as log:
        subprocess.run([str(exp.EVAL_PYTHON), str(exp.OUT/'score.py'), str(directory), str(dataset)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics.update(exp.read(directory/'accuracy.json'))
    result = dict(case=case, compute=compute, status='passed', metrics=metrics)
    exp.save(directory/'result.json', result)
    summary.append(result)
    exp.save(OUT/'summary.json', summary)
    print(exp.stamp(), 'Passed', name, {k:metrics[k] for k in ['tps','nfe','generated_tokens','correct','samples']}, flush=True)
assert all(exp.digest(ROOT/p) == digest for p, digest in sources.items())
print('All full-model acceptance runs passed; source hashes unchanged.', flush=True)
