"""Use reviewed token IDs with the existing benchmark and inference algorithm."""
import json
from pathlib import Path
import random
import numpy as np
import torch
import fluxserve.bench_offline as benchmark

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
_batches = []
_record_batch = benchmark.record_batch_performance_metrics
_write_results = benchmark._write_results

def record_batch(*args, **kwargs):
    result = _record_batch(*args, **kwargs)
    _batches.append(dict(start_idx=args[3], seconds=result.sample_time, nfe=result.nfe,
                         tokens=result.batch_token_number, token_counts=result.batch_token_numbers,
                         tps=result.tps))
    return result

def write_results(args, batch_info, all_input_ids, prompts, questions, ids, tokenizer, dataset_name, start, stop, logger, eos_ids):
    _write_results(args, batch_info, all_input_ids, prompts, questions, ids, tokenizer, dataset_name, start, stop, logger, eos_ids)
    outputs = batch_info.original_order(batch_info.outputs)
    mask_id = tokenizer.convert_tokens_to_ids('<|mask|>')
    with (Path(args.output_dir) / 'completion-tokens.jsonl').open('w') as stream:
        for i, out in enumerate(outputs):
            tokens = out[0, all_input_ids[i].shape[1]:].tolist()
            first_eos = next((j for j,t in enumerate(tokens) if t in eos_ids), None)
            effective = tokens[:first_eos + 1] if first_eos is not None else tokens
            stream.write(json.dumps(dict(id=ids[i], token_ids=effective, eos_found=first_eos is not None,
                                        canvas_tokens=len(tokens), prompt_tokens=all_input_ids[i].shape[1],
                                        masks_before_eos=effective.count(mask_id))) + '\n')
    (Path(args.output_dir) / 'batches.json').write_text(json.dumps(_batches, indent=2) + '\n')

def load_prepared_inputs(dataset, tokenizer, dataset_format='auto', **kwargs):
    with open(dataset) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert all(r['input_ids'] and len(r['input_ids']) == r['input_length'] for r in rows)
    return ([torch.tensor(r['input_ids'], dtype=torch.long).unsqueeze(0) for r in rows],
            [r['prompt'] for r in rows], [r['question'] for r in rows], [r['id'] for r in rows])

# This module is imported by multiprocessing spawn in each rank as well.
benchmark.load_inputs = load_prepared_inputs
benchmark.record_batch_performance_metrics = record_batch
benchmark._write_results = write_results

if __name__ == '__main__':
    from fluxserve.cli import main
    main()
