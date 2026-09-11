"""Fixed-load three-format H100 experiment. All artifacts stay beside this file."""
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PYTHON = REPO / ".venv/bin/python"
SCALES = OUT / "kv-scales.json"
DATASET = OUT / "humaneval-first8.jsonl"
SOURCES = [REPO / "python/fluxserve/bench_offline.py",
           REPO / "python/fluxserve/backend/execution/runners/base.py"]
CASES = [dict(weights=w, kv=kv, gpus=n, graph=g,
              name=f"{w}-{kv}-{n}gpu-{'graph' if g else 'eager'}")
         for w, kv, counts in [("bf16", "bf16", [4]), ("fp8", "bf16", [2, 4]),
                                ("fp8", "fp8_e4m3", [2, 4])]
         for n in counts for g in [False, True]]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def environment(count):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, range(count))),
               CUDA_HOME="/usr/local/cuda", FLUX_KERNEL_CUDA_ARCH="90a",
               TORCH_CUDA_ARCH_LIST="9.0", MAX_JOBS="4", CMAKE_BUILD_PARALLEL_LEVEL="4",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="4", PYTHONUNBUFFERED="1",
               FLUXSERVE_FLASHINFER_REV="5ce4d077c33bbe167386cf5487a4a5bb4bcafbfd",
               FLASHINFER_WORKSPACE_BASE=str(REPO / ".cache/fi018-cu128/flashinfer"))
    for key, folder in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                        ("TORCH_EXTENSIONS_DIR", "torch-extensions"), ("TVM_FFI_CACHE_DIR", "tvm-ffi"),
                        ("CUDA_CACHE_PATH", "cuda")]:
        env[key] = str(REPO / ".cache/cu128" / folder)
    return env


def smi(fields, kind="gpu"):
    return subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}",
                                    "--format=csv,noheader,nounits"], text=True)


def snapshot():
    return dict(time=stamp(), inventory=smi("index,uuid,memory.used,utilization.gpu,clocks.sm,power.draw"),
                processes=smi("gpu_uuid,pid,used_memory", "compute-apps"))


def idle(count):
    snap = snapshot()
    uuids = {line.split(", ")[1] for line in snap["inventory"].splitlines()
             if int(line.split(", ")[0]) < count}
    if len(uuids) != count:
        raise RuntimeError("Requested GPUs are unavailable")
    if any(line.split(", ")[0] in uuids for line in snap["processes"].splitlines()):
        raise RuntimeError("Selected GPUs have other compute processes")
    return snap


def execute(command, directory, count, timeout=7200):
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / "before.json", idle(count))
    env = environment(count)
    save(directory / "command.json", dict(argv=command, started=stamp(),
          environment={k: v for k, v in env.items() if k in environment_keys()}))
    peaks = {}
    observed_pids = {}
    ownership_method = "process_group"
    with (directory / "launcher.log").open("w") as log, (directory / "telemetry.jsonl").open("w") as telemetry:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                snap = snapshot()
                telemetry.write(json.dumps(snap) + "\n")
                telemetry.flush()
                uuids = set()
                for line in snap["inventory"].splitlines():
                    index, uuid, used, *_ = line.split(", ")
                    if int(index) < count:
                        uuids.add(uuid)
                        peaks[index] = max(peaks.get(index, 0), int(used) * 1024**2)
                active = {}
                for line in snap["processes"].splitlines():
                    uuid, pid, _ = line.split(", ")
                    if uuid not in uuids:
                        continue
                    active.setdefault(uuid, set()).add(pid)
                    if len(active[uuid]) > 1:
                        raise RuntimeError(f"Multiple GPU processes on {uuid}; expected one worker per GPU")
                    if uuid in observed_pids and observed_pids[uuid] != pid:
                        raise RuntimeError(f"GPU process changed on {uuid}; result is contaminated")
                    observed_pids[uuid] = pid
                    try:
                        group = os.getpgid(int(pid))
                    except ProcessLookupError:
                        # NVML exposes host PIDs while /proc exposes container PIDs.
                        # Require idle boundaries plus a stable, unique PID on each
                        # GPU throughout this one-worker-per-GPU invocation. Any
                        # additional or replacement PID invalidates the run.
                        ownership_method = "idle_boundaries_and_stable_exclusive_host_pid_per_gpu"
                        continue
                    if group != proc.pid:
                        raise RuntimeError(f"External GPU process {pid} contaminated this run")
                if time.monotonic() > deadline:
                    raise TimeoutError("Run exceeded timeout, including teardown")
                time.sleep(0.5)
            if proc.returncode:
                raise RuntimeError(f"Process exited with code {proc.returncode}; see launcher.log")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
            save(directory / "process.json", dict(returncode=proc.returncode, finished=stamp(),
                 peak_device_used_bytes=peaks, gpu_process_ownership_method=ownership_method,
                 observed_host_pids=observed_pids))
    # Worker cleanup must have completed before another model starts.
    save(directory / "after.json", idle(count))
    return peaks


def environment_keys():
    return {"CUDA_VISIBLE_DEVICES", "CUDA_HOME", "FLUX_KERNEL_CUDA_ARCH", "TORCH_CUDA_ARCH_LIST",
            "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS", "PYTHONUNBUFFERED",
            "FLUXSERVE_FLASHINFER_REV", "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH"}


def prepare():
    from transformers import AutoConfig, AutoTokenizer
    from fluxserve.bench_offline import load_inputs
    models = read(OUT / "models.json")
    if set(models) != {"bf16", "fp8"}:
        raise RuntimeError("Both pinned downloads must finish before preparation")
    DATASET.write_text("\n".join((REPO / "data/humaneval.jsonl").read_text().splitlines()[:8]) + "\n")
    identities, token_ids, dimensions = {}, {}, {}
    for kind, model in models.items():
        path = Path(model["path"])
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        inputs, _, _, ids = load_inputs(DATASET, tokenizer, "openai")
        token_ids[kind] = [t.tolist()[0] for t in inputs]
        assert ids == [f"HumanEval/{i}" for i in range(8)]
        dimensions[kind] = {key: getattr(config, key) for key in
                            ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_experts"]}
        index = read(path / "model.safetensors.index.json")
        shards = sorted(set(index["weight_map"].values()))
        assert all((path / s).is_file() for s in shards)
        identities[kind] = {**model, "dimensions": dimensions[kind], "weight_metadata": index["metadata"],
                            "shards": {s: (path / s).stat().st_size for s in shards},
                            "metadata_sha256": {p.name: digest(p) for p in path.iterdir()
                                                 if p.suffix in {".json", ".py", ".jinja", ".txt"}}}
    assert dimensions["bf16"] == dimensions["fp8"] and dimensions["bf16"]["num_hidden_layers"] == 32
    assert token_ids["bf16"] == token_ids["fp8"], "Tokenizer/input mismatch"
    save(OUT / "inputs.json", {"ids": ids, "token_ids": token_ids["bf16"]})
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in SOURCES}
    (OUT / "source.patch").write_text(subprocess.check_output(["git", "diff"], cwd=REPO, text=True))
    manifest = dict(created=stamp(), commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    models=identities, source_hashes=source_hashes, dataset_sha256=digest(DATASET),
                    calibration_dataset_sha256=digest(REPO / "data/gsm8k.jsonl"),
                    experiment_driver_sha256=digest(Path(__file__)),
                    benchmark_environment={k: v for k, v in environment(4).items() if k in environment_keys()},
                    versions={p: importlib.metadata.version(p) for p in ["torch", "flashinfer-python", "transformers", "triton", "nvidia-cutlass-dsl"]},
                    topology=subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
                    hardware=smi("index,uuid,name,memory.total,driver_version"), cases=CASES,
                    infeasible=[dict(weights=w, kv=kv, gpus=n, status="memory_infeasible", basis="checkpoint_weight_capacity")
                                for w, kv, counts in [("bf16", "bf16", [1, 2]), ("fp8", "bf16", [1]), ("fp8", "fp8_e4m3", [1])]
                                for n in counts])
    save(OUT / "manifest.json", manifest)
    print("Prepared pinned models, identical tokenized inputs, and environment manifest", flush=True)


def check_sources():
    for relative, expected in read(OUT / "manifest.json")["source_hashes"].items():
        if digest(REPO / relative) != expected:
            raise RuntimeError("Runtime source changed; invalidate prior measurements and prepare a new experiment")


def command(case, directory, calibrate=False):
    models = read(OUT / "models.json")
    args = [str(PYTHON), "-m", "fluxserve.cli", "calibrate_kv_cache" if calibrate else "bench_offline",
            "--model", models[case["weights"]]["path"], "--quantization", "modelopt_fp8" if case["weights"] == "fp8" else "auto",
            "--kv-cache-dtype", case["kv"], "--dataset", str(REPO / "data/gsm8k.jsonl" if calibrate else DATASET),
            "--dataset-format", "openai", "--batch-size", "8", "--mini-batch-size", "4", "--gen-len", "512",
            "--block-length", "64", "--prefilling-limit", "128", "--threshold", "0.95", "--low-threshold", "0.3",
            "--parallel-decoding", "threshold", "--disable-sorting", "--tp-size", str(case["gpus"]),
            "--ep-size", str(case["gpus"]), "--dp-size", "1", "--pp-size", "1", "--output-dir", str(directory),
            "--exp-name", "run", "--log-file", "run.log"]
    if calibrate:
        args += ["--num-samples", "128", "--output", str(SCALES)]
    else:
        args += ["--attention-backend", "flashinfer", "--flashinfer-prefill-mode", "paged", "--flashinfer-cache-mode", "paged",
                 "--kv-cache-layout", "paged", "--page-size", "64", "--flashinfer-decode-batch-mode", "max_batch",
                 "--cuda-graph-capture-sizes", "64", "128", "256", "512", "1024"]
        if case["kv"] == "fp8_e4m3":
            args += ["--kv-cache-scales", str(SCALES)]
        if case["graph"]:
            args += ["--use-cuda-graph"]
    return args


def calibrate():
    check_sources()
    if not SCALES.exists():
        print("Starting four-GPU KV calibration on GSM8K first128", flush=True)
        case = dict(weights="fp8", kv="bf16", gpus=4, graph=False)
        execute(command(case, OUT / "calibration", True), OUT / "calibration", 4)
    from transformers import AutoConfig
    from fluxserve.cli import _resolve_quant_config
    from fluxserve.backend.layers.kv_quantization import KVQuantizationConfig
    config = AutoConfig.from_pretrained(read(OUT / "models.json")["fp8"]["path"], trust_remote_code=True, local_files_only=True)
    config.quant_config = _resolve_quant_config(config, "modelopt_fp8")
    validated = KVQuantizationConfig.load(config, "fp8_e4m3", str(SCALES))
    assert len(validated.scales) == 32
    payload = read(SCALES)
    assert payload["calibration"]["num_samples"] == 128
    assert payload["calibration"]["dataset_sha256"] == digest(REPO / "data/gsm8k.jsonl")
    save(OUT / "calibration-validation.json", dict(status="passed", sha256=digest(SCALES), layers=len(validated.scales)))
    print("KV calibration validated", flush=True)


def validate(case, directory):
    metrics = read(directory / "run_metrics.json")
    assert metrics["timing"] == "synchronized_perf_counter_max_rank_per_batch"
    assert metrics["kv_cache_dtype"] == case["kv"]
    assert metrics["weight_format"] == ("modelopt_fp8" if case["weights"] == "fp8" else "bf16")
    assert metrics["num_hidden_layers"] == 32
    assert metrics["nfe"] > 0 and metrics["generated_tokens"] > 0 and metrics["generation_seconds"] > 0
    assert math.isfinite(metrics["tps"])
    assert len(metrics["ranks"]) == case["gpus"]
    if case["kv"] == "fp8_e4m3":
        assert metrics["kv_scale_source"] == str(SCALES)
        assert digest(SCALES) == read(OUT / "calibration-validation.json")["sha256"]
    for rank in metrics["ranks"]:
        fp8_bytes = rank["parameter_bytes_by_dtype"].get("torch.float8_e4m3fn", 0)
        assert (fp8_bytes > 0) == (case["weights"] == "fp8")
        assert rank["local_generation_seconds"] > 0 and rank["kv_data_bytes"] > 0
        graph = rank["flashinfer_graph"]
        replays = rank["generic_graph_replays"] + graph.get("decode_replay_count", 0) + graph.get("replay_count", 0)
        if case["graph"]:
            assert replays > 0, "Graph silently fell back to eager"
            during = rank["flashinfer_graph_during_generation"]
            assert all(during[k] == 0 for k in ["prefill", "decode", "gemma_decode", "invalidations"]), "Timed graph capture/invalidation"
        else:
            assert replays == 0
    paths = list(directory.glob("run_humaneval-first8_*.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert [r["id"] for r in rows] == [f"HumanEval/{i}" for i in range(8)]
    assert sum(r["generated_length"] for r in rows) == metrics["generated_tokens"]
    signature = [(r["id"], r["answer"], r["generated_length"]) for r in rows]
    metrics.update(output_sha256=hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
                   generated_lengths=[r["generated_length"] for r in rows])
    metrics.update(read(directory / "process.json"))
    return metrics


def run_case(case, stage, repeat=None):
    check_sources()
    directory = OUT / stage / case["name"]
    if repeat is not None:
        directory /= str(repeat)
    result_file = directory / "result.json"
    if result_file.exists():
        return read(result_file)
    print(f"{stamp()} Starting {stage} {case['name']} repeat={repeat}", flush=True)
    result = dict(case=case, repeat=repeat, started=stamp())
    try:
        execute(command(case, directory), directory, case["gpus"])
        metrics = validate(case, directory)
        if stage == "measurements":
            trial = read(OUT / "trials" / case["name"] / "result.json")
            metrics["output_matches_trial"] = metrics["output_sha256"] == trial["metrics"]["output_sha256"]
        result.update(status="passed", metrics=metrics)
    except Exception as exc:
        log = (directory / "launcher.log").read_text(errors="replace") if (directory / "launcher.log").exists() else ""
        status = "oom" if any(s in log for s in ["CUDA out of memory", "torch.OutOfMemoryError"]) else "failed"
        result.update(status=status, error=f"{type(exc).__name__}: {exc}")
    result["finished"] = stamp()
    save(result_file, result)
    print(f"Finished {case['name']}: {result['status']} " + str(result.get("error", result.get("metrics", {}).get("tps", ""))), flush=True)
    return result


def trials():
    for case in CASES:
        run_case(case, "trials")
    pairs = {}
    for eager, graph in zip(CASES[::2], CASES[1::2]):
        a = read(OUT / "trials" / eager["name"] / "result.json")
        b = read(OUT / "trials" / graph["name"] / "result.json")
        status = "unavailable"
        if a["status"] == b["status"] == "passed":
            status = "passed" if a["metrics"]["output_sha256"] == b["metrics"]["output_sha256"] else "output_mismatch"
        pairs[eager["name"].removesuffix("-eager")] = status
    save(OUT / "pair-validation.json", pairs)


def measure():
    # User explicitly chose throughput priority after reviewing trial differences.
    # Output differences are reported, not used to cherry-pick measurements.
    eligible = [c for c in CASES if read(OUT / "trials" / c["name"] / "result.json")["status"] == "passed"]
    for repeat in range(1, 6):
        for case in eligible if repeat % 2 else list(reversed(eligible)):
            run_case(case, "measurements", repeat)
            report()


def report():
    groups, raw = {}, []
    for case in CASES:
        runs = [read(p) for p in sorted((OUT / "measurements" / case["name"]).glob("*/result.json"))]
        good = [r["metrics"] for r in runs if r["status"] == "passed"]
        trial_file = OUT / "trials" / case["name"] / "result.json"
        trial = read(trial_file) if trial_file.exists() else {}
        group = {**case, "valid_runs": len(good), "status": "complete" if len(good) == 5 else "incomplete", "trial_status": trial.get("status", "pending")}
        if good:
            values = [r["tps"] for r in good]
            median = statistics.median(values)
            group.update(median_tps=median, mad_tps=statistics.median(abs(v - median) for v in values),
                         per_gpu_tps=median / case["gpus"], tokens=[r["generated_tokens"] for r in good],
                         nfe=[r["nfe"] for r in good], seconds=[r["generation_seconds"] for r in good],
                         tokens_per_forward=[r["generated_tokens"] / r["nfe"] for r in good],
                         forwards_per_second=[r["nfe"] / r["generation_seconds"] for r in good],
                         unique_output_count=len({r["output_sha256"] for r in good}),
                         output_matches_trial=[r["output_matches_trial"] for r in good],
                         peak_device_used_bytes={str(i): max(r["peak_device_used_bytes"].get(str(i), 0) for r in good) for i in range(case["gpus"])},
                         peak_allocated_bytes=max(rank["peak_allocated_bytes"] for r in good for rank in r["ranks"]))
        groups[case["name"]] = group
        for r in runs:
            metrics = r.get("metrics", {})
            raw.append({**case, "repeat": r["repeat"], "status": r["status"],
                        **{k: r.get("metrics", {}).get(k) for k in ["tps", "generated_tokens", "nfe", "generation_seconds", "output_sha256"]},
                        "tokens_per_forward": metrics.get("generated_tokens", 0) / metrics["nfe"] if metrics.get("nfe") else None,
                        "forwards_per_second": metrics.get("nfe", 0) / metrics["generation_seconds"] if metrics.get("generation_seconds") else None,
                        "output_matches_trial": metrics.get("output_matches_trial"),
                        "error": r.get("error", "")})
    ratios = []
    def ratio(label, numerator, denominator, efficiency_divisor=None):
        a, b = groups[numerator], groups[denominator]
        row = dict(label=label, numerator=numerator, denominator=denominator)
        if a["status"] == b["status"] == "complete":
            row.update(status="complete", speedup=a["median_tps"] / b["median_tps"])
            row.update(numerator_tokens=a["tokens"], denominator_tokens=b["tokens"],
                       numerator_nfe=a["nfe"], denominator_nfe=b["nfe"])
            if efficiency_divisor:
                row["efficiency"] = row["speedup"] / efficiency_divisor
        else:
            row["status"] = "unavailable"
        ratios.append(row)
    for mode in ["eager", "graph"]:
        ratio("model_quantization", f"fp8-bf16-4gpu-{mode}", f"bf16-bf16-4gpu-{mode}")
        ratio("joint_quantization", f"fp8-fp8_e4m3-4gpu-{mode}", f"bf16-bf16-4gpu-{mode}")
        for n in [2, 4]:
            ratio("kv_quantization", f"fp8-fp8_e4m3-{n}gpu-{mode}", f"fp8-bf16-{n}gpu-{mode}")
        for kv in ["bf16", "fp8_e4m3"]:
            ratio("scaling_2_to_4", f"fp8-{kv}-4gpu-{mode}", f"fp8-{kv}-2gpu-{mode}", 2)
    for case in CASES:
        if case["graph"]:
            ratio("cuda_graph", case["name"], case["name"].removesuffix("-graph") + "-eager")
    differences = []
    for comparison in ratios:
        for repeat in range(1, 6):
            left_dir = OUT / "measurements" / comparison["numerator"] / str(repeat)
            right_dir = OUT / "measurements" / comparison["denominator"] / str(repeat)
            left_files = list(left_dir.glob("run_humaneval-first8_*.jsonl"))
            right_files = list(right_dir.glob("run_humaneval-first8_*.jsonl"))
            if len(left_files) != 1 or len(right_files) != 1:
                continue
            left = [json.loads(line) for line in left_files[0].read_text().splitlines()]
            right = [json.loads(line) for line in right_files[0].read_text().splitlines()]
            differences.append(dict(label=comparison["label"], numerator=comparison["numerator"],
                denominator=comparison["denominator"], repeat=repeat,
                changed_answer_ids=[a["id"] for a, b in zip(left, right, strict=True) if a["answer"] != b["answer"]],
                changed_length_ids=[a["id"] for a, b in zip(left, right, strict=True) if a["generated_length"] != b["generated_length"]],
                numerator_lengths=[r["generated_length"] for r in left],
                denominator_lengths=[r["generated_length"] for r in right]))
    save(OUT / "output-differences.json", differences)
    save(OUT / "summary.json", dict(updated=stamp(), groups=groups, ratios=ratios,
         infeasible=read(OUT / "manifest.json")["infeasible"], valid_measurements=sum(g["valid_runs"] for g in groups.values()),
         expected_measurements=50, pair_validation=read(OUT / "pair-validation.json") if (OUT / "pair-validation.json").exists() else {}))
    summary = read(OUT / "summary.json")
    summary["acceptance_policy"] = "Throughput priority, user-authorized output differences; runtime, timing and graph validity gates remain required."
    summary["interpretation"] = "Eager/Graph may change outputs, NFE, batching and attention kernels (BF16 eager native FA3 versus decomposed Graph FA2); ratios measure implementation throughput, not isolated launch overhead or quality equivalence."
    save(OUT / "summary.json", summary)
    unavailable = [{**row, "graph": graph, "valid_runs": 0}
                   for row in read(OUT / "manifest.json")["infeasible"] for graph in [False, True]]
    for filename, rows in [("summary.csv", list(groups.values()) + unavailable), ("raw.csv", raw), ("ratios.csv", ratios)]:
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            with (OUT / filename).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "calibrate", "trials", "measure", "report", "all"])
    stage = parser.parse_args().stage
    if stage == "all":
        for function in [prepare, calibrate, trials, measure, report]:
            function()
    else:
        globals()[stage]()
