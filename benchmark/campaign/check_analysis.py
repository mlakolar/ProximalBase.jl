#!/usr/bin/env python3
"""Analytic controls and corrupted-record checks for the analysis utilities."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from summarize import summarize
from validate import validate


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def fixture(folder, factor):
    folder.mkdir()
    for rep in range(1, 11):
        prefix = folder / f"julia-1.13.2-rep{rep:02}"
        row = dict(id="analytic/control", family="control", name="control", dtype="Float64",
                   shape="1", layout="dense", phase="warm", expected_zero="true", rep=rep,
                   mean_ns=factor*rep, median_ns=factor*rep, min_ns=factor*rep/2,
                   max_ns=factor*rep*1.5, mean_bytes=0, min_bytes=0, max_bytes=0,
                   samples=3, gc_share=0)
        write_rows(Path(str(prefix) + "-metrics.tsv"), [row])
        write_rows(Path(str(prefix) + "-samples.tsv"),
                   [dict(id=row["id"], rep=rep, sample=i, time_ns=factor*rep*t)
                    for i, t in enumerate((0.5, 1, 1.5), 1)])
        write_rows(Path(str(prefix) + "-allocations.tsv"),
                   [dict(id=row["id"], rep=rep, sample=i, bytes=0) for i in range(1, 11)])
        Path(str(prefix) + "-catalog.tsv").write_text("id\nanalytic/control\n")
        Path(str(prefix) + "-errors.tsv").write_text("id\texception\n")
        Path(str(prefix) + "-metadata.toml").write_text(
            f'julia="1.13.2"\nrep={rep}\ncpu="control"\nblas="control"\n'
            'blas_threads=1\njulia_threads=1\nbenchmarktools="control"\n'
            'seed_scheme="control"\nsamples_target=3\nseconds_limit=0.1\n'
            'profile_seconds=0.2\nmode="full"\ncase_filter=""\ncases=1\nfailures=0\n'
            f'harness_sha="{"a"*40}"\nsource_sha="{"b"*40}"\n')


with tempfile.TemporaryDirectory(prefix="pb-analysis-control-") as temporary:
    root = Path(temporary)
    fixture(root / "before", 2)
    fixture(root / "after", 1)
    assert validate(root / "before", versions=["1.13.2"])["valid"]
    assert summarize(root / "before", root / "summary", 10, ["1.13.2"])
    with (root / "summary" / "summary.csv").open() as stream:
        row = next(csv.DictReader(stream))
    expected_sd = math.sqrt(110/3)  # SD of 2,4,...,20
    expected_half = 2.262 * expected_sd / math.sqrt(10)
    assert float(row["mean_ns"]) == 11
    assert math.isclose(float(row["sd_between_runs_ns"]), expected_sd)
    assert math.isclose(float(row["ci95_low_ns"]), 11-expected_half)
    assert math.isclose(float(row["ci95_high_ns"]), 11+expected_half)
    command = [sys.executable, str(Path(__file__).with_name("compare.py")),
               str(root / "before"), str(root / "after"), str(root / "comparison")]
    subprocess.run(command, check=True)
    with (root / "comparison" / "comparison.csv").open() as stream:
        row = next(csv.DictReader(stream))
    for key in ("paired_geometric_speedup", "ci95_low", "ci95_high"):
        assert math.isclose(float(row[key]), 2, rel_tol=1e-12)
    # A missing repetition cannot silently become a complete baseline.
    missing = root / "before" / "julia-1.13.2-rep10-metrics.tsv"
    saved = missing.read_text()
    missing.unlink()
    assert not validate(root / "before", versions=["1.13.2"])["valid"]
    assert not summarize(root / "before", root / "incomplete", 10, ["1.13.2"])
    assert len(json.loads((root / "incomplete" / "status.json").read_text())["incomplete"]) == 1
    missing.write_text(saved)
    # Ten rows with a wrong repetition ID are also incomplete.
    with missing.open(newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    row["rep"] = 11
    write_rows(missing, [row])
    assert not summarize(root / "before", root / "wrong-rep", 10, ["1.13.2"])
    missing.write_text(saved)
    # Retained raw observations must agree with their reported statistics.
    raw = root / "before" / "julia-1.13.2-rep01-samples.tsv"
    saved = raw.read_text()
    raw.write_text(saved.replace("1.0", "9.0"))
    assert not validate(root / "before", versions=["1.13.2"])["valid"]
    raw.write_text(saved)
    # Changing instrumentation invalidates an automatic speedup comparison.
    for path in (root / "after").glob("*-metadata.toml"):
        path.write_text(path.read_text().replace("a"*40, "c"*40))
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode and "configuration differs" in rejected.stderr
print("ANALYSIS_CONTROLS_PASSED mean,SD,t-interval,paired-ratio,missing/wrong-repetition,raw-corruption,settings-mismatch")
