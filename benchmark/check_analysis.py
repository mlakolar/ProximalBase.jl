#!/usr/bin/env python3
"""Analytic controls and corrupted-record checks for the analysis utilities."""
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from compare import compare
from summarize import summarize
from validate import calibration_metadata, timer_limited, validate


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def fixture(folder, factor, runs=10, allocated=0, case="analytic/control", evals=1000):
    folder.mkdir()
    for rep in range(1, runs+1):
        prefix = str(folder / f"julia-1.13.2-rep{rep:02}")
        row = dict(id=case, family="control", name="control", dtype="Float64",
                   shape="1", layout="dense", phase="warm", expected_zero="true", rep=rep,
                   mean_ns=factor*rep, median_ns=factor*rep, min_ns=factor*rep/2,
                   max_ns=factor*rep*1.5, mean_bytes=allocated, min_bytes=allocated,
                   max_bytes=allocated, samples=3, gc_share=0, evals=evals)
        write_rows(Path(prefix+"-metrics.tsv"), [row])
        write_rows(Path(prefix+"-samples.tsv"),
                   [dict(id=case, rep=rep, sample=i, time_ns=factor*rep*t)
                    for i, t in enumerate((0.5, 1, 1.5), 1)])
        write_rows(Path(prefix+"-allocations.tsv"),
                   [dict(id=case, rep=rep, sample=i, bytes=allocated) for i in range(1, 11)])
        write_rows(Path(prefix+"-catalog.tsv"), [dict(id=case, family="control", dtype="Float64",
                   shape="1", layout="dense", phase="warm", zero_alloc="true", evals_policy="auto")])
        write_rows(Path(prefix+"-calibration.tsv"),
                   [dict(phase=phase, sample=i, time_ns=30) for phase in ("start", "end")
                    for i in range(1, 101)])
        Path(prefix+"-errors.tsv").write_text("id\texception\n")
        Path(prefix+"-metadata.toml").write_text(
            f'julia="1.13.2"\nrep={rep}\ncpu="control"\ncpu_model="control"\nblas="control"\n'
            'blas_threads=1\njulia_threads=1\nbenchmarktools="control"\n'
            'seed_scheme="control"\nsamples_target=3\nseconds_limit=0.1\n'
            'calibration_samples=1000\ncalibration_seconds=1.0\ntune_seconds=0.1\n'
            'timer_start_median_ns=30.0\ntimer_start_min_ns=30.0\ntimer_start_step_ns=nan\n'
            'timer_end_median_ns=30.0\ntimer_end_min_ns=30.0\ntimer_end_step_ns=nan\ntimer_floor_ns=30.0\n'
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
    assert row["timing_assessment"] == "faster"
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
    # Strict batch-time rule, including its boundary and unusable floors.
    for evals, median, limited in ((1, 120, True), (1, 450, False), (10, 100, False),
                                    (200, 60, False), (1, 300, False)):
        assert timer_limited(dict(evals=evals, median_ns=median), dict(timer_floor_ns=30)) == limited
    for values in ([30.0]*100, [0.001, 41.0, 41.0, 83.0]*25):
        calibrated = calibration_metadata(dict(start=values, end=values))
        assert calibrated["timer_floor_ns"] == (30 if values[0] == 30 else 41)
        if values[0] == 30:
            assert math.isnan(calibrated["timer_start_step_ns"])
        else:
            assert calibrated["timer_start_step_ns"] == 41
    skewed = calibration_metadata(dict(start=[0.001]*100 + [41.0, 83.0],
                                        end=[0.001]*100 + [41.0, 83.0]))
    assert skewed["timer_start_median_ns"] == 0.001 and skewed["timer_floor_ns"] == 41
    example = calibration_metadata(dict(start=[20, 30, 40]*100, end=[40]*100))
    assert example["timer_floor_ns"] == 40
    for values in ([30.0]*99, [0.0]*100, [0.001]*100):
        calibrated = calibration_metadata(dict(start=values, end=values))
        assert math.isnan(calibrated["timer_floor_ns"])
        for evals in (1, 2, 100):
            assert timer_limited(dict(evals=evals, median_ns=1), calibrated) == (evals == 1)

    def metric_edit(folder, rep, **changes):
        path = folder/f"julia-1.13.2-rep{rep:02d}-metrics.tsv"
        with path.open(newline="") as stream:
            row = next(csv.DictReader(stream, delimiter="\t"))
        row.update(changes)
        write_rows(path, [row])

    limited_before, limited_after = root/"limited-before", root/"limited-after"
    fixture(limited_before, 120, 3)
    fixture(limited_after, 60, 3, allocated=16)
    for rep in (1, 3):
        metric_edit(limited_before, rep, evals=1)
    # K=20 limits repetition 3 as well as repetition 1.
    metric_edit(limited_after, 2, evals=1)
    row = compare(limited_before, limited_after, root/"limited-comparison", 3, 20)[0]
    assert row["timer_limited_by"] == "before:rep01,rep03;after:rep02"
    assert row["timer_limited"] and row["timer_limit_multiple"] == 20
    assert row["before_evals"] == "1/1000" and row["after_evals"] == "1/1000"
    assert row["timing_assessment"] == "timer-limited"
    assert row["paired_geometric_speedup"] == 2 and row["unexpected_allocation"]
    report = (root/"limited-comparison"/"REPORT.md").read_text()
    assert "timer-limited; unexpected allocation" in report and "ratio gates" in report
    # Only the before side is limited at the default K.
    metric_edit(limited_after, 2, evals=1000)
    row = compare(limited_before, limited_after, root/"one-side", 3)[0]
    assert row["timer_limited_by"] == "before:rep01"
    row = compare(limited_before, limited_after, root/"override", 3, 1)[0]
    assert not row["timer_limited"] and row["timer_limited_by"] == ""
    for multiple, expected_limit in ((10, "True"), (1, "False")):
        output = root/f"limit-summary{multiple}"
        assert summarize(limited_before, output, 3, timer_limit_multiple=multiple)
        with (output/"summary.csv").open() as stream:
            assert next(csv.DictReader(stream))["timer_limited"] == expected_limit
        assert "timer_floor_ns range 30–30 ns; unusable calibration in 0 processes" in (output/"REPORT.md").read_text()
    process_meta = limited_before/"julia-1.13.2-rep03-metadata.toml"
    saved_meta = process_meta.read_text()
    process_calibration = limited_before/"julia-1.13.2-rep03-calibration.tsv"
    saved_process_calibration = process_calibration.read_text()
    process_meta.write_text(saved_meta.replace("=30.0", "=40.0"))
    write_rows(process_calibration, [dict(phase=phase, sample=i, time_ns=40)
                                     for phase in ("start", "end") for i in range(1, 101)])
    row = compare(limited_before, limited_after, root/"process-floors", 3)[0]
    assert row["timer_limited_by"] == "before:rep01,rep03"
    assert summarize(limited_before, root/"floor-range", 3)
    assert "timer_floor_ns range 30–40 ns" in (root/"floor-range"/"REPORT.md").read_text()
    process_meta.write_text(saved_meta)
    process_calibration.write_text(saved_process_calibration)
    for script, args, filename in (("compare.py", [limited_before, limited_after], "comparison.csv"),
                                   ("summarize.py", [limited_before], "summary.csv")):
        output = root/f"cli-{script}"
        command = [sys.executable, str(Path(__file__).with_name(script)),
                   *(str(arg) for arg in args), str(output), "--expected-runs", "3",
                   "--timer-limit-multiple", "1"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        with (output/filename).open() as stream:
            assert next(csv.DictReader(stream))["timer_limited"] == "False"
        command[-1] = "nan"
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0 and "timer limit multiple" in failed.stderr
    fixture(root/"single-limited", 120, 1, evals=1)
    row = compare(root/"single-limited", root/"single-limited", root/"single-limit-report")[0]
    assert row["timing_assessment"] == "single run; exploratory; timer-limited"
    for multiple in (0, -1, float("nan"), float("inf")):
        for function, args in ((compare, (limited_before, limited_after, root/"invalid-k")),
                               (summarize, (limited_before, root/"invalid-k-summary"))):
            try:
                function(*args, timer_limit_multiple=multiple)
            except ValueError as error:
                assert "timer limit multiple" in str(error)
            else:
                raise AssertionError("invalid K accepted")
    fixture(root/"corrupt", 120, 1)
    folder = root/"corrupt"
    path = folder/"julia-1.13.2-rep01-metrics.tsv"
    saved = path.read_text()
    for changes in (dict(median_ns=999), dict(evals=0), dict(evals=-1), dict(evals="1.5")):
        metric_edit(folder, 1, **changes)
        assert not validate(folder)["valid"]
        path.write_text(saved)
    path = folder/"julia-1.13.2-rep01-metadata.toml"
    saved = path.read_text()
    for key in ("timer_floor_ns", "timer_start_median_ns", "timer_start_min_ns", "timer_end_median_ns", "timer_end_min_ns"):
        path.write_text(saved.replace(key+"=30.0", key+"=31.0"))
        assert not validate(folder)["valid"]
    for key in ("timer_start_step_ns", "timer_end_step_ns"):
        path.write_text(saved.replace(key+"=nan", key+"=0.0"))
        assert not validate(folder)["valid"]
    path.write_text(saved)
    catalog = folder/"julia-1.13.2-rep01-catalog.tsv"
    saved_catalog = catalog.read_text()
    catalog.write_text(saved_catalog.replace("auto", "unknown"))
    assert not validate(folder)["valid"]
    catalog.write_text(saved_catalog.replace("auto", "fixed"))
    assert validate(folder)["valid"]
    catalog.write_text(saved_catalog)
    calibration = folder/"julia-1.13.2-rep01-calibration.tsv"
    saved_calibration = calibration.read_text()
    calibration.unlink()
    assert not validate(folder)["valid"]
    calibration.write_text(saved_calibration)
    for old, new in (("start\t1\t30", "start\t2\t30"), ("start\t1\t30", "other\t1\t30"),
                     ("start\t1\t30", "start\t1\tnan"), ("start\t1\t30", "start\t1\t-1"),
                     ("start\t1\t30", "start\t0\t30")):
        calibration.write_text(saved_calibration.replace(old, new, 1))
        assert not validate(folder)["valid"]
    calibration.write_text(saved_calibration)
    for values in ([30.0]*99, [0.0]*100, [0.001, 41.0, 41.0, 83.0]*25):
        write_rows(calibration, [dict(phase=phase, sample=i, time_ns=value)
                                for phase in ("start", "end") for i, value in enumerate(values, 1)])
        calibrated = calibration_metadata(dict(start=values, end=values))
        lines = [line for line in saved.splitlines() if not line.startswith("timer_")]
        path.write_text("\n".join(lines + [f"{key}={value}" for key, value in calibrated.items()])+"\n")
        assert validate(folder)["valid"]
        metric_edit(folder, 1, evals=1)
        output = root/f"calibration-summary{len(values)}-{values[0]}"
        assert summarize(folder, output)
        if math.isnan(calibrated["timer_floor_ns"]):
            assert "unusable calibration in 1 processes" in (output/"REPORT.md").read_text()
        with (output/"summary.csv").open() as stream:
            assert next(csv.DictReader(stream))["timer_limited"] == "True"
        metric_edit(folder, 1, evals=1000)
print("ANALYSIS_CONTROLS_PASSED repetitions=1,3,5,10; means,SD,t-intervals,paired-ratios,allocations,missing/wrong-repetitions,raw-corruption,unmatched-cases,settings-mismatch; timer-rule,boundary,mixed-evals,limited-sides,K-overrides,invalid-K,process-floors,unusable/constant/stepped-calibration,missing-calibration,corrupt-median/evals/floor,eligible-2x,limited-ratio/allocation-warning")
