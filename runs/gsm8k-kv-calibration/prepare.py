"""Pin source datasets and build immutable tokenized experiment inputs."""
import concurrent.futures
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from evalscope.benchmarks.gsm8k.gsm8k_adapter import PROMPT_TEMPLATE, FEWSHOT_TEMPLATE

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
MODELS = json.loads((REPO / 'runs/quant-throughput/models.json').read_text())
DATASETS = {
    'ultrachat': dict(repo='HuggingFaceH4/ultrachat_200k', revision='8049631c405ae6576f93f445c6b8166f76f5505a', split='train_sft', file='data/train_sft-00000-of-00003-a3ecf92756993583.parquet'),
    'gsm8k-train': dict(repo='openai/gsm8k', revision='740312add88f781978c0658806c59bc2815b9866', split='train', file='main/train-00000-of-00001.parquet'),
    'gsm8k-test': dict(repo='openai/gsm8k', revision='740312add88f781978c0658806c59bc2815b9866', split='test', file='main/test-00000-of-00001.parquet'),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def download(item):
    name, meta = item
    path = OUT / 'data' / f'{name}.parquet'
    url = f"https://huggingface.co/datasets/{meta['repo']}/resolve/{meta['revision']}/{meta['file']}"
    if not path.exists():
        print('Downloading', name, flush=True)
        temp = path.with_suffix('.tmp')
        with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as target:
            while chunk := response.read(4 * 1024 * 1024):
                target.write(chunk)
        temp.replace(path)
    return name, {**meta, 'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def main():
    (OUT / 'data').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        sources = dict(pool.map(download, DATASETS.items()))
    train = pq.read_table(sources['gsm8k-train']['path']).to_pylist()
    test = pq.read_table(sources['gsm8k-test']['path']).to_pylist()
    assert len(train) == 7473 and len(test) == 1319
    ultra = next(pq.ParquetFile(sources['ultrachat']['path']).iter_batches(batch_size=512)).to_pylist()
    assert len(ultra) == 512
    tokenizers = {k: AutoTokenizer.from_pretrained(v['path'], local_files_only=True, trust_remote_code=True) for k, v in MODELS.items()}
    tokenizer = tokenizers['fp8']
    def row(key, messages, question, limit=None, generation_prompt=True):
        texts = {k: t.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt) for k, t in tokenizers.items()}
        ids = {k: t(texts[k], add_special_tokens=False)['input_ids'] for k, t in tokenizers.items()}
        assert ids['fp8'] == ids['bf16'], f'Tokenizer mismatch: {key}'
        original_length = len(ids['fp8'])
        tokens = ids['fp8'][:limit] if limit else ids['fp8']
        return dict(id=key, input_ids=tokens, prompt=tokenizer.decode(tokens, skip_special_tokens=False),
                    question=question, untruncated_input_length=original_length, input_length=len(tokens),
                    input_truncated=original_length > len(tokens))
    fewshot = []
    for record in train[:4]:
        reasoning, answer = record['answer'].rsplit('####', 1)
        fewshot.append(record['question'] + '\n\nReasoning:\n' + reasoning.strip() + '\n\nANSWER: \\boxed{' + answer.strip() + '}')
    prefix = '\n\n'.join(fewshot)
    evaluation = []
    gold = []
    for i, record in enumerate(test):
        prompt = FEWSHOT_TEMPLATE.format(fewshot=prefix, question=record['question'])
        evaluation.append(row(f'gsm8k/test/{i}', [{'role': 'user', 'content': prompt}], record['question']))
        gold.append(dict(id=f'gsm8k/test/{i}', question=record['question'], answer=record['answer'], target=record['answer'].rsplit('####', 1)[1].strip()))
    order = np.random.default_rng(42).permutation(512).tolist()
    calibration = {}
    calibration['ultrachat'] = [row(f'ultrachat/train_sft/{i}', ultra[i]['messages'], ultra[i]['messages'][0]['content'], 2048, False) for i in order]
    calibration['gsm8k-train'] = [row(f'gsm8k/train/{i}', [{'role':'user', 'content':PROMPT_TEMPLATE.format(question=train[i]['question'])}], train[i]['question'], 2048) for i in order]
    test_questions = {r['question'].strip() for r in test}
    assert not test_questions.intersection(r['question'].strip() for r in train[:512])
    overlaps = [(i, q) for i, record in enumerate(ultra) for q in test_questions if any(q in message['content'] for message in record['messages'])]
    assert not overlaps, f'UltraChat exact test question overlap: {overlaps}'
    jsonl(OUT / 'data/test.jsonl', evaluation)
    jsonl(OUT / 'data/gold.jsonl', gold)
    # A separate 16-item operational smoke test exercises varying prompt lengths.
    smoke_ids = sorted(set([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,max(range(len(evaluation)),key=lambda i:evaluation[i]['input_length'])]))
    jsonl(OUT / 'data/smoke.jsonl', [evaluation[i] for i in smoke_ids])
    for name, rows in calibration.items():
        jsonl(OUT / 'data' / f'calibration-{name}.jsonl', rows)
    save(OUT / 'data/fewshot.json', dict(train_indices=list(range(4)), records=train[:4], text=prefix))
    files = [p for p in (OUT / 'data').glob('*.json*')]
    manifest = dict(sources=sources, tokenizers_identical=True, calibration_samples=512,
                    calibration_selection='first 512, then numpy default_rng(42) permutation; matches datasets.shuffle(seed=42)',
                    calibration_max_input_tokens=2048, calibration_generation_length=2048,
                    calibration_method='FluxServe SDPA eager BF16 KV; observes prefill and all diffusion iterations',
                    ultrachat_processing='full conversation with model chat template, no generation prompt; right truncate at 2048 tokens',
                    gsm8k_calibration_processing='question only plus EvalScope zero-shot CoT instruction; model chat template; no gold answer input',
                    evaluation=dict(samples=1319, fewshot_train_indices=list(range(4)), few_shot=4, gen_len=2048,
                                    template='pinned EvalScope GSM8K FEWSHOT_TEMPLATE, plus model chat template',
                                    no_input_truncation=True, batch_size=8, mini_batch_size=4),
                    exact_test_question_overlap=dict(gsm8k_train=0,ultrachat=0),
                    input_stats={name:dict(min=min(r['input_length'] for r in rows),max=max(r['input_length'] for r in rows),
                                          mean=float(np.mean([r['input_length'] for r in rows])),truncated=sum(r['input_truncated'] for r in rows))
                                 for name,rows in {**calibration,'test':evaluation}.items()},
                    files={str(p.relative_to(OUT)):sha(p) for p in files},
                    evalscope_commit='acd09b44384d53174768bb1063f675420f76fae9',
                    versions={k:importlib.metadata.version(k) for k in ['evalscope','datasets','pyarrow','transformers','numpy']})
    save(OUT / 'data-manifest.json', manifest)
    print(json.dumps(manifest['input_stats'],indent=2),flush=True)

if __name__ == '__main__':
    main()
