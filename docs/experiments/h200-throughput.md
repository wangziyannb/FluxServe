# FluxServe LLaDA2.0-flash H200 Throughput Experiment

## 1. Goal

Measure three independent effects on one H200 node:

1. ModelOpt FP8 versus BF16 at the same GPU count.
2. Strong scaling from 1/2/4/8 H200 GPUs with a fixed global batch.
3. CUDA Graph versus eager execution for the same model and GPU count.

Do not combine results from different model depths. Do not include checkpoint
loading, JIT compilation, or CUDA Graph capture time in throughput.

### Provenance

The base workload comes from Kenneth Zhao's upstream `dev-fp8` commit
`fd1c489`: HumanEval, batch 8, generation length 512, block length 64,
threshold 0.95, threshold decoding, disabled sorting, and SDPA. That upstream
run used all HumanEval records and `cache=prefix`.

This document follows the stricter local A/B protocol instead: only the first
eight records are used, eager and Graph are repeated and counterbalanced, and
the current mainline dense/no-prefix cache path is used. These differences must
be stated when results are reported.

## 2. Code And Models

- FluxServe branch: `quantization`
- Minimum tested revision: `700c2e2`
- BF16: `inclusionAI/LLaDA2.0-flash`
- FP8: `thnkinbtfly/llada2.0-flash-fp8`
- Dataset: `data/humaneval.jsonl`

H200 is SM90. ModelOpt FP8 is supported. ModelOpt NVFP4 is not part of this
experiment because the current NVFP4 backend requires Blackwell (SM100+).

The full BF16 checkpoint is about 192 GiB on disk, so it cannot fit on one
141 GB H200. The full FP8 checkpoint is about 98 GiB and should be tested on
one H200 first. Therefore:

| GPUs | BF16 full model | FP8 full model | Valid FP8/BF16 comparison |
| ---: | :---: | :---: | :---: |
| 1 | No, expected OOM | Expected to fit; verify first | No |
| 2 | Yes | Yes | Yes |
| 4 | Yes | Yes | Yes |
| 8 | Yes | Yes | Yes |

If FP8 CUDA Graph capture exceeds memory on one H200, record that mode as OOM.
Do not reduce its batch size and place the result in the same comparison table.

## 3. Fixed Workload

Use the first eight HumanEval records in their original order:

```bash
mkdir -p /data/fluxserve-h200-throughput
sed -n '1,8p' data/humaneval.jsonl \
  > /data/fluxserve-h200-throughput/humaneval-first8.jsonl
```

Keep these settings identical for every measured case:

| Setting | Value |
| --- | --- |
| Task | HumanEval code generation, records 0-7 |
| Global batch size | 8 |
| Mini-batch size | 4 |
| Maximum generation length | 512 |
| Block length | 64 |
| Prefill limit | 128 |
| Threshold | 0.95 |
| Low threshold | 0.3 |
| Parallel decoding | `threshold` |
| Attention backend | SDPA |
| KV cache layout | dense |
| Sorting | disabled/original dataset order |
| DP | 1 |
| PP | 1 |
| CUDA Graph capture sizes | 64, 128, 256, 512, 1024 |

TPS means effective completion tokens divided by generation wall time. Prompt
tokens, checkpoint loading, warmup, kernel compilation, and Graph capture are
excluded.

## 4. Parallelism Matrix

Use model tensor parallelism and expert parallelism across the same set of
GPUs. Keep data parallelism disabled so that the execution model remains the
same and CUDA Graph stays available.

| H200 count | `CUDA_VISIBLE_DEVICES` | `--tp-size` | `--ep-size` | `--dp-size` |
| ---: | --- | ---: | ---: | ---: |
| 1 | `0` | 1 | 1 | 1 |
| 2 | `0,1` | 2 | 2 | 1 |
| 4 | `0,1,2,3` | 4 | 4 | 1 |
| 8 | `0,1,2,3,4,5,6,7` | 8 | 8 | 1 |

Before the two-GPU run, use `nvidia-smi topo -m` and select an NVLink-connected
pair. Run all models serially and stop unrelated GPU services.

## 5. Environment Record

Save this once with the result set:

```bash
git rev-parse HEAD
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv
python -c 'import torch, transformers, triton; print(torch.__version__, transformers.__version__, triton.__version__)'
```

Build and use the Dockerfile from the tested revision. It contains SM90 code:

```bash
docker build -f docker/Dockerfile.flux-cu129 -t fluxserve:h200 .
```

Use the same image, driver, clocks, power settings, and MoE kernel config for
all cases. Complete any Triton JIT work before recording measurements.

## 6. Benchmark Command

Set one case at a time. Example for two H200 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export N=2
export FORMAT=fp8
export MODEL=thnkinbtfly/llada2.0-flash-fp8
export QUANTIZATION=modelopt_fp8
export MODE=eager
export REP=1

GRAPH_ARGS=()
if [ "$MODE" = graph ]; then
  GRAPH_ARGS=(--use-cuda-graph)
fi

fluxserve bench_offline \
  --model "$MODEL" \
  --quantization "$QUANTIZATION" \
  --dataset /data/fluxserve-h200-throughput/humaneval-first8.jsonl \
  --dataset-format openai \
  --batch-size 8 \
  --mini-batch-size 4 \
  --gen-len 512 \
  --block-length 64 \
  --prefilling-limit 128 \
  --threshold 0.95 \
  --low-threshold 0.3 \
  --parallel-decoding threshold \
  --attention-backend sdpa \
  --flashinfer-prefill-mode dense \
  --flashinfer-cache-mode dense \
  --kv-cache-layout dense \
  --tp-size "$N" \
  --dp-size 1 \
  --ep-size "$N" \
  --pp-size 1 \
  --cuda-graph-capture-sizes 64 128 256 512 1024 \
  --disable-sorting \
  --exp-name "${FORMAT}-${N}gpu-${MODE}-${REP}" \
  --output-dir /data/fluxserve-h200-throughput/results \
  --log-file "/data/fluxserve-h200-throughput/results/${FORMAT}-${N}gpu-${MODE}-${REP}.log" \
  "${GRAPH_ARGS[@]}"
```

For BF16, change only:

```bash
export FORMAT=bf16
export MODEL=inclusionAI/LLaDA2.0-flash
export QUANTIZATION=auto
```

For each GPU count, change `CUDA_VISIBLE_DEVICES`, `N`, and the output name
according to the parallelism table. The FluxServe CLI launches its local NCCL
workers; do not wrap this command in `torchrun` or `mpirun`.

## 7. Run Order

For every valid `(format, GPU count)` pair:

1. Run one unrecorded eager invocation to populate caches and verify loading.
2. Run one unrecorded Graph invocation to finish capture and verify replay.
3. Execute five measured eager runs and five measured Graph runs.
4. Counterbalance the measured mode order as:

```text
E, G, G, E, G, E, E, G, G, E
```

Use a fresh process for each invocation. FluxServe performs warmup before its
timed generation loop. Never run BF16 and FP8 concurrently.

## 8. Correctness And Acceptance Checks

- `/health` is not relevant to `bench_offline`; successful worker exit is.
- All eight request IDs must remain `HumanEval/0` through `HumanEval/7`.
- Token count and NFE must be recorded for every run.
- Graph runs must actually replay CUDA Graphs rather than silently running eager.
- Graph and eager output must be identical within the same model/checkpoint.
- BF16 and FP8 output equality is not required.
- Record peak allocated GPU memory for every rank.
- Reject runs with another compute process on any selected GPU.
- Report median TPS and MAD; retain all five raw measurements.

## 9. Result Tables

### Absolute Throughput

| Format | Mode | 1x H200 | 2x H200 | 4x H200 | 8x H200 |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | eager | N/A |  |  |  |
| BF16 | Graph | N/A |  |  |  |
| FP8 | eager |  |  |  |  |
| FP8 | Graph |  |  |  |  |

Each cell should contain `median TPS (MAD)`.

### Quantization Gain At Equal GPU Count

| Mode | 2x H200 | 4x H200 | 8x H200 |
| --- | ---: | ---: | ---: |
| eager FP8/BF16 |  |  |  |
| Graph FP8/BF16 |  |  |  |

Do not report a one-GPU FP8/BF16 ratio because full BF16 is not runnable there.

### Strong Scaling

For FP8, use one H200 as the baseline:

```text
speedup(N) = TPS(N) / TPS(1)
efficiency(N) = speedup(N) / N
```

For BF16, use two H200 GPUs as the baseline:

```text
speedup(N) = TPS(N) / TPS(2)
efficiency(N) = speedup(N) / (N / 2)
```

Report eager and Graph scaling separately. Also report per-GPU TPS so that
communication overhead is visible.

## 10. Interpretation Rules

- Compare FP8 versus BF16 only at the same GPU count and with the full model.
- Compare Graph versus eager only within the same format and GPU count.
- Keep global batch 8 fixed; this is a strong-scaling experiment.
- A separate capacity/weak-scaling experiment may increase batch with GPU
  count, but its results must not be mixed into these tables.
- Report the single-H200 BF16 case as memory-infeasible, not as zero TPS.
- If TP/EP scaling loses efficiency, inspect NCCL time and MoE all-to-all before
  attributing the loss to quantization kernels.
