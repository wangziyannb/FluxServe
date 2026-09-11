"""Final independent artifact and measurement acceptance checks."""
import csv
import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


manifest = read(OUT / "manifest.json")
summary = read(OUT / "summary.json")
assert summary["valid_measurements"] == summary["expected_measurements"] == 50
assert len(summary["groups"]) == 10
assert manifest["acceptance_policy"]["output_equality_required"] is False
for relative, expected in manifest["source_hashes"].items():
    assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected
for package, expected in manifest["versions"].items():
    assert importlib.metadata.version(package) == expected
assert hashlib.sha256((OUT / "humaneval-first8.jsonl").read_bytes()).hexdigest() == manifest["dataset_sha256"]
assert hashlib.sha256((OUT / "kv-scales.json").read_bytes()).hexdigest() == read(OUT / "calibration-validation.json")["sha256"]

intervals = []
prefill_replays = []
for name, group in summary["groups"].items():
    assert group["status"] == "complete" and group["valid_runs"] == 5
    assert sorted(p.name for p in (OUT / "measurements" / name).iterdir()) == ["1", "2", "3", "4", "5"]
    values = []
    for repeat in range(1, 6):
        directory = OUT / "measurements" / name / str(repeat)
        result = read(directory / "result.json")
        assert result["status"] == "passed"
        metrics = result["metrics"]
        close(metrics["tps"], metrics["generated_tokens"] / metrics["generation_seconds"])
        close(metrics["generation_seconds"], max(r["local_generation_seconds"] for r in metrics["ranks"]))
        assert len(metrics["ranks"]) == group["gpus"]
        assert metrics["weight_format"] == ("modelopt_fp8" if group["weights"] == "fp8" else "bf16")
        assert metrics["kv_cache_dtype"] == group["kv"] and metrics["num_hidden_layers"] == 32
        if group["graph"]:
            for rank in metrics["ranks"]:
                assert rank["flashinfer_graph"]["decode_replay_count"] > 0
                assert all(v == 0 for v in rank["flashinfer_graph_during_generation"].values())
                prefill_replays.append(rank["flashinfer_graph"]["prefill_replay_count"])
        argv = read(directory / "command.json")["argv"]
        for flag in ["--tp-size", "--ep-size"]:
            assert argv[argv.index(flag) + 1] == str(group["gpus"])
        assert argv[argv.index("--batch-size") + 1] == "8"
        assert argv[argv.index("--mini-batch-size") + 1] == "4"
        assert argv[argv.index("--model") + 1] == manifest["models"][group["weights"]]["path"]
        assert ("--use-cuda-graph" in argv) == group["graph"]
        assert read(directory / "process.json")["returncode"] == 0
        rows = [json.loads(line) for line in next(directory.glob("run_humaneval-first8_*.jsonl")).read_text().splitlines()]
        assert [r["id"] for r in rows] == [f"HumanEval/{i}" for i in range(8)]
        assert sum(r["generated_length"] for r in rows) == metrics["generated_tokens"]
        signature = [(r["id"], r["answer"], r["generated_length"]) for r in rows]
        assert hashlib.sha256(json.dumps(signature).encode()).hexdigest() == metrics["output_sha256"]
        assert (directory / "telemetry.jsonl").stat().st_size > 0
        intervals.append((datetime.datetime.fromisoformat(result["started"]), datetime.datetime.fromisoformat(result["finished"])))
        values.append(metrics["tps"])
    median = statistics.median(values)
    close(group["median_tps"], median)
    close(group["mad_tps"], statistics.median(abs(v - median) for v in values))
    close(group["per_gpu_tps"], median / group["gpus"])

intervals.sort()
assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
assert len(summary["ratios"]) == 17
for row in summary["ratios"]:
    assert row["status"] == "complete"
    expected = summary["groups"][row["numerator"]]["median_tps"] / summary["groups"][row["denominator"]]["median_tps"]
    close(row["speedup"], expected)
    if row["label"] == "scaling_2_to_4":
        close(row["efficiency"], expected / 2)
with (OUT / "summary.csv").open() as handle:
    rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert all(not row["median_tps"] for row in rows if row["status"] == "memory_infeasible")
with (OUT / "raw.csv").open() as handle:
    assert len(list(csv.DictReader(handle))) == 50
assert len(read(OUT / "output-differences.json")) == 85
verification = dict(status="passed", valid_measurements=50, runtime_cases=10,
    ratio_checks=17, output_comparisons=85, measurement_intervals_nonoverlapping=True,
    source_and_environment_unchanged=True, output_equality_required=False,
    graph_scope="decode; prefill remains eager on this batch-8/mini-batch-4 workload" if not any(prefill_replays) else "prefill and decode",
    token_count_policy="Prompt excluded; first EOS included; otherwise count non-mask completion tokens.")
(OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))
