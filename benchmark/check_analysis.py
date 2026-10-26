#!/usr/bin/env python3
"""Analytic controls and corrupted-record checks for the analysis utilities."""
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile

from compare import compare
from summarize import summarize
from validate import validate


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def fixture(folder, factor, runs=10, allocated=0, case="analytic/control"):
    folder.mkdir()
    for rep in range(1, runs+1):
        prefix = str(folder / f"julia-1.13.2-rep{rep:02}")
        row = dict(id=case, family="control", name="control", dtype="Float64",
                   shape="1", layout="dense", phase="warm", expected_zero="true", rep=rep,
                   mean_ns=factor*rep, median_ns=factor*rep, min_ns=factor*rep/2,
                   max_ns=factor*rep*1.5, mean_bytes=allocated, min_bytes=allocated,
                   max_bytes=allocated, samples=3, gc_share=0)
        write_rows(Path(prefix+"-metrics.tsv"), [row])
        write_rows(Path(prefix+"-samples.tsv"),
                   [dict(id=case, rep=rep, sample=i, time_ns=factor*rep*t)
                    for i, t in enumerate((0.5, 1, 1.5), 1)])
        write_rows(Path(prefix+"-allocations.tsv"),
                   [dict(id=case, rep=rep, sample=i, bytes=allocated) for i in range(1, 11)])
        Path(prefix+"-catalog.tsv").write_text(f"id\n{case}\n")
        Path(prefix+"-errors.tsv").write_text("id\texception\n")
        Path(prefix+"-metadata.toml").write_text(
            f'julia="1.13.2"\nrep={rep}\ncpu="control"\ncpu_model="control"\nblas="control"\n'
            'blas_threads=1\njulia_threads=1\nbenchmarktools="control"\n'
            'seed_scheme="control"\nsamples_target=3\nseconds_limit=0.1\n'
            'profile_seconds=0\nsuite="quick"\ngroups="vectors"\ncases=1\nfailures=0\n'
            f'harness_hash="{"a"*40}"\nsource_hash="{"b"*40}"\n')


def reject(before, after, output, message, expected=10):
    try:
        compare(before, after, output, expected)
    except ValueError as error:
        assert message in str(error), error
        assert (output/"errors.json").exists()
    else:
        raise AssertionError("Invalid comparison accepted")


with tempfile.TemporaryDirectory(prefix="pb-analysis-control-") as temporary:
    root = Path(temporary)
    # Actual-count t intervals and allocation differences at routine and extended counts.
    for runs, critical in ((1, None), (3, 4.303), (5, 2.776), (10, 2.262)):
        before, after = root/f"before{runs}", root/f"after{runs}"
        fixture(before, 2, runs, allocated=64)
        fixture(after, 1, runs, allocated=16)
        assert validate(before, runs)["valid"]
        assert summarize(before, root/f"summary{runs}", runs)
        with (root/f"summary{runs}"/"summary.csv").open() as stream:
            row = next(csv.DictReader(stream))
        expected_mean = runs+1
        assert float(row["mean_ns"]) == expected_mean
        if runs == 1:
            assert math.isnan(float(row["ci95_low_ns"]))
        else:
            expected_sd = statistics.stdev(range(2, 2*runs+1, 2))
            half = critical*expected_sd/math.sqrt(runs)
            assert math.isclose(float(row["sd_between_runs_ns"]), expected_sd)
            assert math.isclose(float(row["ci95_low_ns"]), max(0, expected_mean-half))
            assert math.isclose(float(row["ci95_high_ns"]), expected_mean+half)
        row = compare(before, after, root/f"comparison{runs}", runs)[0]
        assert math.isclose(row["paired_geometric_speedup"], 2)
        assert row["mean_bytes_change"] == -48
        assert row["before_max_bytes"] == 64 and row["after_max_bytes"] == 16
        assert row["unexpected_allocation"]
        if runs == 1:
            assert math.isnan(row["ci95_low"]) and "exploratory" in row["timing_assessment"]
        else:
            assert math.isclose(row["ci95_low"], 2) and math.isclose(row["ci95_high"], 2)
    before, after = root/"before10", root/"after10"
    missing = before/"julia-1.13.2-rep10-metrics.tsv"
    saved = missing.read_text(); missing.unlink()
    assert not validate(before, 10)["valid"]
    assert not summarize(before, root/"incomplete", 10)
    reject(before, after, root/"missing-comparison", "Invalid results")
    assert len(json.loads((root/"incomplete"/"status.json").read_text())["incomplete"]) == 1
    missing.write_text(saved)
    with missing.open(newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    row["rep"] = 11; write_rows(missing, [row])
    assert not summarize(before, root/"wrong-rep", 10)
    missing.write_text(saved)
    raw = before/"julia-1.13.2-rep01-samples.tsv"
    saved = raw.read_text(); raw.write_text(saved.replace("1.0", "9.0"))
    assert not validate(before, 10)["valid"]
    assert not summarize(before, root/"raw-corruption", 10)
    raw.write_text(saved)
    # Consistent new case catalogs are still incompatible with the starting revision.
    fixture(root/"other-cases", 1, case="analytic/different")
    reject(before, root/"other-cases", root/"unmatched", "Case coverage differs")
    assert json.loads((root/"unmatched"/"coverage.json").read_text())["before_only"]
    for path in after.glob("*-metadata.toml"):
        path.write_text(path.read_text().replace("a"*40, "c"*40))
    reject(before, after, root/"settings-mismatch", "configuration differs")
    assert not validate(root/"empty")["valid"]
print("ANALYSIS_CONTROLS_PASSED repetitions=1,3,5,10; means,SD,t-intervals,paired-ratios,allocations,missing/wrong-repetitions,raw-corruption,unmatched-cases,settings-mismatch")
