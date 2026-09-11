"""Verify published records and summary arithmetic without GPU inference."""

import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/fa3-gsm8k"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), (
        actual, expected
    )


def verify_archive(directory):
    manifest = read_json(directory / "records-manifest.json")
    path = directory / manifest["archive"]
    assert path.stat().st_size == manifest["archive_bytes"]
    assert sha(path.read_bytes()) == manifest["archive_sha256"]
    metadata = manifest["members"]
    assert len(metadata) == manifest["member_count"]
    assert sum(m["bytes"] for m in metadata.values()) == manifest["uncompressed_bytes"]
    results = {}
    with tarfile.open(path, "r:gz") as archive:
        assert len(archive.getmembers()) == len(metadata)
        assert set(archive.getnames()) == set(metadata)
        for member in archive.getmembers():
            assert member.isfile()
            assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            data = archive.extractfile(member).read()
            assert len(data) == metadata[member.name]["bytes"]
            assert sha(data) == metadata[member.name]["sha256"]
            if member.name.endswith("/result.json"):
                results[member.name] = json.loads(data)
    return results


def main():
    results = verify_archive(OUT)
    verify_archive(REPO / "runs/fp8-attention")
    summary = list(csv.DictReader((OUT / "summary.csv").open()))
    assert len(summary) == 10
    for row in summary:
        formal = results[f"measurements/{row['name']}/result.json"]
        smoke = results[f"smoke/{row['name']}/result.json"]
        assert row["status"] == formal["status"] == smoke["status"] == "passed"
        metrics = formal["metrics"]
        assert metrics["samples"] == 1319 and smoke["metrics"]["samples"] == 16
        assert metrics["returncode"] == 0
        for key in ("tps", "accuracy", "correct", "samples", "generated_tokens", "nfe", "generation_seconds"):
            close(row[key], metrics[key])
        close(row["tps"], metrics["generated_tokens"] / metrics["generation_seconds"])
        close(row["accuracy"], metrics["correct"] / metrics["samples"])
        close(row["forwards_per_second"], metrics["nfe"] / metrics["generation_seconds"])
        close(row["per_gpu_tps"], metrics["tps"] / int(row["gpus"]))
        close(row["peak_device_gib"], max(metrics["peak_device_used_bytes"].values()) / 2**30)

    manifest = read_json(OUT / "manifest.json")
    preflight = read_json(OUT / "preflight-validation.json")
    snapshot = OUT / "source-snapshot.tar.gz"
    assert sha(snapshot.read_bytes()) == preflight["source_snapshot_sha256"]
    with tarfile.open(snapshot, "r:gz") as archive:
        for path, digest in manifest["sources"].items():
            assert sha(archive.extractfile(path).read()) == digest, path

    audit = read_json(OUT / "completion-audit.json")
    assert audit["status"] == "passed" and audit["all_runs_passed"]
    assert audit["formal_runs"] == audit["smoke_runs"] == 10
    assert audit["total_formal_answers"] == sum(int(r["samples"]) for r in summary)
    assert audit["total_generated_tokens"] == sum(int(r["generated_tokens"]) for r in summary)
    assert sha((OUT / "scales-ultrachat-bf16-model.json").read_bytes()) == audit["calibration_bf16"]["scale_sha256"]

    sources = {
        "current": {r["name"]: r for r in summary},
        "previous": {
            r["name"]: r
            for r in csv.DictReader((REPO / "runs/gsm8k-kv-calibration/summary.csv").open())
            if r["calibration"] != "gsm8k-train"
        },
    }
    combined = list(csv.DictReader((OUT / "combined-matrix.csv").open()))
    merged = read_json(OUT / "combined-matrix.json")
    expected = {(phase, name) for phase, records in sources.items() for name in records}
    assert len(combined) == len(merged["rows"]) == len(expected) == 20
    for records in (combined, merged["rows"]):
        assert {(r["phase"], r["name"]) for r in records} == expected
        for row in records:
            source = sources[row["phase"]][row["name"]]
            for key in ("tps", "accuracy", "correct", "samples", "generated_tokens", "nfe", "generation_seconds"):
                close(row[key], source[key])
    by_name = {r["name"]: r for r in merged["rows"]}
    for comparison in merged["comparisons"]:
        numerator = by_name[comparison["numerator"]]
        denominator = by_name[comparison["denominator"]]
        for target, source in (("tps_ratio", "tps"), ("tokens_ratio", "generated_tokens"), ("nfe_ratio", "nfe"), ("forwards_per_second_ratio", "forwards_per_second")):
            close(comparison[target], numerator[source] / denominator[source])
        close(comparison["accuracy_delta_pp"], 100 * (numerator["accuracy"] - denominator["accuracy"]))
    print("PASS: both archives and all member hashes; 10 full + 10 smoke runs; source snapshot; scale; 20 combined rows and comparison arithmetic.")


if __name__ == "__main__":
    main()
