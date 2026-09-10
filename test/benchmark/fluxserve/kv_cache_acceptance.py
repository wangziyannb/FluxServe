"""Serial LLaDA2 KV acceptance runs. Requires idle GPUs and a working runtime.

Use --stage check to record hardware availability without loading a model.
Quality uses the same pinned EvalScope revision as test/ci/eval/1N1G/gsm8k.sh.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

EVALSCOPE_COMMIT = "acd09b44384d53174768bb1063f675420f76fae9"
REPO = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True
        ).strip()

    inventory = query(
        ["--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu"]
    )
    processes = query(["--query-compute-apps=gpu_uuid,pid,used_memory"])
    return {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inventory": inventory,
        "processes": processes,
    }


def require_idle(gpus, output):
    snapshot = gpu_snapshot()
    rows = [line.split(", ") for line in snapshot["inventory"].splitlines()]
    selected = {row[1] for row in rows if row[0] in gpus}
    if len(selected) != len(gpus):
        raise RuntimeError(f"Requested GPUs are unavailable: {gpus}")
    busy = [
        line
        for line in snapshot["processes"].splitlines()
        if line.split(", ")[0] in selected
    ]
    write_json(
        output / "availability.json",
        {
            **snapshot,
            "selected_gpus": gpus,
            "status": "unverified_busy" if busy else "idle",
        },
    )
    if busy:
        raise RuntimeError(
            "Selected GPUs have running compute processes; full experiments were not started."
        )


def runtime_command(args, command, extra):
    return [
        args.python,
        "-m",
        "fluxserve.cli",
        command,
        "--model",
        args.model,
        "--quantization",
        "modelopt_fp8",
        *extra,
    ]


def execute(command, path, env, allowed_groups=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path.with_suffix(".command.json"),
        {"command": command, "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
    )
    peaks = {}
    samples = []
    selected = set(env["CUDA_VISIBLE_DEVICES"].split(","))
    with path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 7200
            while process.poll() is None:
                snapshot = gpu_snapshot()
                uuids = set()
                for line in snapshot["inventory"].splitlines():
                    index, uuid, _, _, used, _ = line.split(", ")
                    if index in selected:
                        uuids.add(uuid)
                        peaks[index] = max(peaks.get(index, 0), int(used) * 1024**2)
                samples.append(snapshot)
                for line in snapshot["processes"].splitlines():
                    uuid, pid, _ = line.split(", ")
                    if uuid not in uuids:
                        continue
                    try:
                        group = os.getpgid(int(pid))
                    except ProcessLookupError:
                        continue
                    if group not in (process.pid, *allowed_groups):
                        raise RuntimeError(
                            f"Unrelated GPU process {pid} appeared during the run; result is not accepted"
                        )
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Experiment timed out, including graph teardown; result is not accepted"
                    )
                time.sleep(0.5)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            write_json(path.with_suffix(".telemetry.json"), samples)
    return {"peak_device_used_bytes": peaks}


def calibrate(args):
    require_idle(args.gpus, args.output)
    command = runtime_command(
        args,
        "calibrate_kv_cache",
        [
            "--dataset",
            str(REPO / "data/gsm8k.jsonl"),
            "--num-samples",
            "128",
            "--output",
            str(args.scales),
            "--tp-size",
            str(len(args.gpus)),
            "--ep-size",
            str(len(args.gpus)),
            "--batch-size",
            "8",
            "--mini-batch-size",
            "4",
            "--gen-len",
            "512",
            "--block-length",
            "64",
            "--threshold",
            "0.95",
            "--parallel-decoding",
            "threshold",
            "--output-dir",
            str(args.output / "calibration"),
        ],
    )
    execute(command, args.output / "calibration.log", args.env)


def median_mad(values):
    median = statistics.median(values)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
    }


def performance(args):
    first8 = args.output / "humaneval-first8.jsonl"
    first8.write_text(
        "\n".join((REPO / "data/humaneval.jsonl").read_text().splitlines()[:8]) + "\n"
    )
    groups = {}
    for backend in ("sdpa", "flashinfer"):
        for count, graph in ((1, False), (4, False), (4, True)):
            if len(args.gpus) < count:
                groups[f"{backend}-{count}-{'graph' if graph else 'eager'}"] = {
                    "status": "unverified_hardware"
                }
                continue
            for repeat in range(5):
                # Alternate dtype order to reduce systematic ordering effects.
                for dtype in (
                    ("bf16", "fp8_e4m3") if repeat % 2 == 0 else ("fp8_e4m3", "bf16")
                ):
                    gpus = args.gpus[:count]
                    require_idle(gpus, args.output)
                    label = f"{backend}-{count}-{'graph' if graph else 'eager'}-{dtype}"
                    directory = args.output / "performance" / label / str(repeat)
                    command = runtime_command(
                        args,
                        "bench_offline",
                        [
                            "--dataset",
                            str(first8),
                            "--batch-size",
                            "8",
                            "--mini-batch-size",
                            "4",
                            "--gen-len",
                            "512",
                            "--block-length",
                            "64",
                            "--threshold",
                            "0.95",
                            "--parallel-decoding",
                            "threshold",
                            "--attention-backend",
                            backend,
                            "--tp-size",
                            str(count),
                            "--ep-size",
                            str(count),
                            "--kv-cache-dtype",
                            dtype,
                            "--output-dir",
                            str(directory),
                            "--exp-name",
                            "run",
                            "--log-file",
                            "run.log",
                        ],
                    )
                    if dtype == "fp8_e4m3":
                        command += ["--kv-cache-scales", str(args.scales)]
                    if graph:
                        command += ["--use-cuda-graph"]
                    telemetry = execute(
                        command,
                        directory / "launcher.log",
                        {**args.env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)},
                    )
                    metrics = json.loads((directory / "run_metrics.json").read_text())
                    metrics.update(telemetry)
                    if graph:
                        for rank in metrics["ranks"]:
                            replays = rank["generic_graph_replays"] + rank[
                                "flashinfer_graph"
                            ].get("decode_replay_count", 0)
                            if replays <= 0:
                                raise RuntimeError(
                                    f"No generation graph replay recorded for {label} rank {rank['rank']}"
                                )
                    rows = groups.setdefault(label, {"status": "running", "runs": []})[
                        "runs"
                    ]
                    rows.append(metrics)
                    entry = groups[label]
                    for field in ("tps", "nfe"):
                        entry[field] = median_mad([row[field] for row in rows])
                    entry["max_gpu_peak_device_used_bytes"] = median_mad(
                        [max(row["peak_device_used_bytes"].values()) for row in rows]
                    )
                    for field in (
                        "kv_data_bytes",
                        "graph_input_kv_bytes",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                    ):
                        entry[f"max_rank_{field}"] = median_mad(
                            [max(rank[field] for rank in row["ranks"]) for row in rows]
                        )
                        entry[f"sum_ranks_{field}"] = median_mad(
                            [sum(rank[field] for rank in row["ranks"]) for row in rows]
                        )
                    entry["status"] = "measured" if len(rows) == 5 else "running"
                    write_json(args.output / "performance.json", groups)


def quality(args):
    # Require the pinned installation; never silently use a different evaluator.
    check = "import importlib.metadata,json; d=importlib.metadata.distribution('evalscope'); print(json.loads(d.read_text('direct_url.json'))['vcs_info']['commit_id'])"
    revision = subprocess.check_output(
        [args.evalscope_python, "-c", check], text=True
    ).strip()
    if revision != EVALSCOPE_COMMIT:
        raise RuntimeError(
            f"EvalScope must be installed from commit {EVALSCOPE_COMMIT}"
        )
    scores = {}
    for dtype in ("bf16", "fp8_e4m3"):
        require_idle(args.gpus, args.output)
        directory = args.output / "quality" / dtype
        directory.mkdir(parents=True, exist_ok=True)
        command = runtime_command(
            args,
            "serve",
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--attention-backend",
                "sdpa",
                "--tp-size",
                str(len(args.gpus)),
                "--ep-size",
                str(len(args.gpus)),
                "--kv-cache-dtype",
                dtype,
                "--threshold",
                "0.95",
                "--block-length",
                "64",
                "--max-model-len",
                "4096",
                "--max-new-tokens",
                "512",
            ],
        )
        if dtype == "fp8_e4m3":
            command += ["--kv-cache-scales", str(args.scales)]
        with (directory / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=args.env,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 1200
                while True:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            "Quality server failed readiness; see server.log"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{args.port}/health", timeout=2
                        ):
                            break
                    except OSError:
                        time.sleep(2)
                evaluate = [
                    args.evalscope_python,
                    "-m",
                    "evalscope.cli.cli",
                    "eval",
                    "--model",
                    args.model,
                    "--model-id",
                    f"kv-{dtype}",
                    "--eval-type",
                    "openai_api",
                    "--api-url",
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    "--api-key",
                    "EMPTY",
                    "--datasets",
                    "humaneval",
                    "--eval-batch-size",
                    "4",
                    "--judge-strategy",
                    "rule",
                    "--generation-config",
                    '{"max_tokens":512,"temperature":0.0,"n":1}',
                    "--work-dir",
                    str(directory),
                    "--no-timestamp",
                    "--no-collect-perf",
                ]
                execute(
                    evaluate,
                    directory / "evalscope.log",
                    args.env,
                    allowed_groups=(server.pid,),
                )
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
                    raise RuntimeError(
                        "Server teardown timed out; quality run is not accepted"
                    )
        reports = list((directory / "reports").rglob("humaneval.json"))
        if len(reports) != 1:
            raise RuntimeError("Expected exactly one full HumanEval report")
        report = json.loads(reports[0].read_text())
        if report.get("num") != 164:
            raise RuntimeError("HumanEval acceptance requires all 164 problems")
        scores[dtype] = float(report["score"])
    drop = scores["bf16"] - scores["fp8_e4m3"]
    write_json(
        args.output / "quality.json",
        {
            "evalscope_commit": revision,
            "num": 164,
            "pass_at_1": scores,
            "drop": drop,
            "status": "passed" if drop <= 0.02 else "failed",
        },
    )
    if drop > 0.02:
        raise RuntimeError(f"FP8 KV pass@1 drop {drop:.4f} exceeds 0.02")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("check", "calibrate", "quality", "performance", "all"),
        default="check",
    )
    parser.add_argument("--model", default="thnkinbtfly/llada2.0-flash-fp8")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/kv-cache-acceptance")
    )
    parser.add_argument("--scales", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--evalscope-python", default="/tmp/evalscope-venv/bin/python")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    args.gpus = args.gpus.split(",")
    args.output = args.output_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.scales = (args.scales or args.output / "scales.json").resolve()
    args.env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(args.gpus)}
    write_json(
        args.output / "source.json",
        {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "diff": subprocess.check_output(["git", "diff"], cwd=REPO, text=True),
            "model": args.model,
            "evalscope_commit": EVALSCOPE_COMMIT,
            "source_sha256": {
                name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                for name in subprocess.check_output(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=REPO,
                    text=True,
                ).splitlines()
                if (REPO / name).is_file()
            },
        },
    )
    try:
        require_idle(args.gpus, args.output)
        if args.stage in ("calibrate", "all"):
            calibrate(args)
        if args.stage in ("quality", "all"):
            quality(args)
        if args.stage in ("performance", "all"):
            performance(args)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        write_json(
            args.output / "incomplete.json", {"stage": args.stage, "reason": str(exc)}
        )
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
