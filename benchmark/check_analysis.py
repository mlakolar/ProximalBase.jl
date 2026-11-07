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

from compare import compare, parse_reps
from summarize import summarize
from validate import calibration_metadata, timer_limited, validate


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def fixture(folder, factor, runs=10, allocated=0, case="analytic/control", evals=1000, version="1.13.2",
            multipliers=None):
    """`multipliers[rep-1]` scales the times of a repetition (noise in paired ratios); a list
    `allocated` gives the bytes of each repetition."""
    folder.mkdir(exist_ok=True)   # several versions may share a directory
    per_rep = allocated if isinstance(allocated, (list, tuple)) else [allocated]*runs
    for rep, allocated in zip(range(1, runs+1), per_rep):
        prefix = str(folder / f"julia-{version}-rep{rep:02}")
        scale = factor*rep*(1 if multipliers is None else multipliers[rep-1])
        row = dict(id=case, family="control", name="control", dtype="Float64",
                   shape="1", layout="dense", phase="warm", expected_zero="true", rep=rep,
                   mean_ns=scale, median_ns=scale, min_ns=scale/2,
                   max_ns=scale*1.5, mean_bytes=allocated, min_bytes=allocated,
                   max_bytes=allocated, samples=3, gc_share=0, evals=evals)
        write_rows(Path(prefix+"-metrics.tsv"), [row])
        write_rows(Path(prefix+"-samples.tsv"),
                   [dict(id=case, rep=rep, sample=i, time_ns=scale*t)
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
            f'julia="{version}"\nrep={rep}\ncpu="control"\ncpu_model="control"\nblas="control"\n'
            'blas_threads=1\njulia_threads=1\nbenchmarktools="control"\n'
            'seed_scheme="control"\nsamples_target=3\nseconds_limit=0.1\n'
            'calibration_samples=1000\ncalibration_seconds=1.0\ntune_seconds=0.1\n'
            'timer_start_median_ns=30.0\ntimer_start_min_ns=30.0\ntimer_start_step_ns=30.0\n'
            'timer_end_median_ns=30.0\ntimer_end_min_ns=30.0\ntimer_end_step_ns=30.0\ntimer_floor_ns=30.0\n'
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
    # check_bounds and os (#439): two known values must agree; a record without a key is unknown,
    # never reported as matching.
    def recorded_fixture(name, factor, **settings):
        folder = root/name
        fixture(folder, factor, 3)
        for path in folder.glob("*-metadata.toml"):
            path.write_text(path.read_text() + "".join(f'{k}="{v}"\n' for k, v in settings.items()))
        return folder

    def statuses(name, version="1.13.2"):
        by_version = json.loads((root/name/"settings.json").read_text())["by_version"]
        return {key: entry["status"] for key, entry in by_version[version].items()}

    agreed = dict(check_bounds="default", os="Linux x86_64")
    recorded_before = recorded_fixture("recorded-before", 2, **agreed)
    recorded_after = recorded_fixture("recorded-after", 1, **agreed)
    checked = validate(recorded_before, 3)
    assert checked["valid"] and checked["unrecorded_settings"] == []
    row = compare(recorded_before, recorded_after, root/"recorded-match", 3)[0]
    assert math.isclose(row["paired_geometric_speedup"], 2) and row["timing_assessment"] == "faster"
    assert statuses("recorded-match") == dict(check_bounds="match", os="match")
    report = (root/"recorded-match"/"REPORT.md").read_text()
    assert "Recorded settings, Julia 1.13.2: `check_bounds` default on both sides; `os` Linux x86_64 on both sides." in report
    assert "verified" not in report
    for key, value in (("check_bounds", "yes"), ("check_bounds", "no"), ("os", "Darwin arm64")):
        conflicting = recorded_fixture(f"recorded-{key}-{value.split()[0]}", 1, **{**agreed, key: value})
        assert validate(conflicting, 3)["valid"]
        reject(recorded_before, conflicting, root/f"conflict-{key}-{value.split()[0]}",
               f"Julia 1.13.2 {key}: before {agreed[key]}; after {value}", expected=3)
        reject(conflicting, recorded_before, root/f"conflict-reverse-{key}-{value.split()[0]}",
               f"Julia 1.13.2 {key}: before {value}; after {agreed[key]}", expected=3)
    unrecorded = recorded_fixture("recorded-none", 1)
    checked = validate(unrecorded, 3)
    assert checked["valid"] and checked["unrecorded_settings"] == ["check_bounds", "os"]
    row = compare(unrecorded, recorded_after, root/"unknown-before", 3)[0]
    assert math.isclose(row["paired_geometric_speedup"], 1)
    assert statuses("unknown-before") == dict(check_bounds="unknown_before", os="unknown_before")
    report = (root/"unknown-before"/"REPORT.md").read_text()
    assert "`check_bounds` unknown on the before side (default on the after side)" in report
    assert "`check_bounds` and `os` could not be verified to match" in report
    compare(recorded_after, unrecorded, root/"unknown-after", 3)
    assert statuses("unknown-after") == dict(check_bounds="unknown_after", os="unknown_after")
    compare(unrecorded, unrecorded, root/"unknown-both", 3)
    assert statuses("unknown-both") == dict(check_bounds="unknown_both", os="unknown_both")
    assert "`check_bounds` unknown on both sides" in (root/"unknown-both"/"REPORT.md").read_text()
    # One key recorded on both sides, the other on one side only: one match, one unknown.
    partial = recorded_fixture("recorded-partial", 1, check_bounds="default")
    compare(recorded_before, partial, root/"partial", 3)
    assert statuses("partial") == dict(check_bounds="match", os="unknown_after")
    assert "`os` could not be verified to match" in (root/"partial"/"REPORT.md").read_text()
    # Within one directory a setting may not change, appear or disappear between repetitions.
    meta = recorded_after/"julia-1.13.2-rep02-metadata.toml"
    saved = meta.read_text()
    for edited in (saved.replace('check_bounds="default"', 'check_bounds="yes"'),
                   saved.replace('os="Linux x86_64"', 'os="Darwin arm64"'),
                   saved.replace('check_bounds="default"\n', ""),
                   saved.replace('check_bounds="default"', 'check_bounds="maybe"'),
                   saved.replace('os="Linux x86_64"', 'os=" "')):
        meta.write_text(edited)
        assert not validate(recorded_after, 3)["valid"]
        reject(recorded_before, recorded_after, root/f"edited-{len(list(root.iterdir()))}", "Invalid results", expected=3)
    meta.write_text(saved)
    assert validate(recorded_after, 3)["valid"]
    # A literal os = "<unrecorded>" is a recorded value, never a missing key (review of #471).
    literal = recorded_fixture("recorded-literal", 1, check_bounds="default", os="<unrecorded>")
    checked = validate(literal, 3)
    assert checked["valid"] and checked["unrecorded_settings"] == []
    literal_meta = literal/"julia-1.13.2-rep02-metadata.toml"
    saved = literal_meta.read_text()
    literal_meta.write_text(saved.replace('os="<unrecorded>"\n', ""))
    assert not validate(literal, 3)["valid"]          # present in rep01 and rep03, absent in rep02
    literal_meta.write_text(saved)
    literal_too = recorded_fixture("recorded-literal-too", 2, check_bounds="default", os="<unrecorded>")
    compare(literal_too, literal, root/"literal-match", 3)
    assert statuses("literal-match") == dict(check_bounds="match", os="match")
    assert "`os` <unrecorded> on both sides" in (root/"literal-match"/"REPORT.md").read_text()
    compare(literal_too, partial, root/"literal-vs-missing", 3)
    assert statuses("literal-vs-missing") == dict(check_bounds="match", os="unknown_after")
    # Several Julia versions (review of #471): settings are compared within each version only.
    V1, V2 = "1.10.0", "1.11.0"

    def versions_fixture(name, factor, per_version):
        folder = root/name
        for version, settings in per_version.items():
            fixture(folder, factor, 3, version=version)
            for path in folder.glob(f"julia-{version}-rep*-metadata.toml"):
                path.write_text(path.read_text() + "".join(f'{k}="{v}"\n' for k, v in settings.items()))
        checked = validate(folder, 3)
        assert checked["valid"] and checked["versions"] == [V1, V2], checked["issues"]
        return folder

    def settings_of(name):
        return json.loads((root/name/"settings.json").read_text())["by_version"]

    linux, darwin = "Linux x86_64", "Darwin arm64"
    # Swapped known values: the value sets of both directories agree, every version disagrees.
    for key, (a, b) in (("check_bounds", ("default", "yes")), ("os", (linux, darwin))):
        swap_before = versions_fixture(f"swap-{key}-before", 2, {V1: {key: a}, V2: {key: b}})
        swap_after = versions_fixture(f"swap-{key}-after", 1, {V1: {key: b}, V2: {key: a}})
        out = root/f"swap-{key}"
        reject(swap_before, swap_after, out, f"Julia {V1} {key}: before {a}; after {b}", expected=3)
        error = json.loads((out/"errors.json").read_text())["error"]
        assert f"Julia {V2} {key}: before {b}; after {a}" in error
        assert not (out/"settings.json").exists() and not (out/"comparison.csv").exists()
    # Matching values per version, different between versions: accepted, a match in each version.
    per_version = {V1: dict(check_bounds="default", os=linux), V2: dict(check_bounds="yes", os=darwin)}
    rows = compare(versions_fixture("multi-before", 2, per_version),
                   versions_fixture("multi-after", 1, per_version), root/"multi-match", 3)
    assert [r["julia"] for r in rows] == [f"julia-{V1}", f"julia-{V2}"]
    assert all(math.isclose(r["paired_geometric_speedup"], 2) for r in rows)
    assert settings_of("multi-match") == {
        V1: dict(check_bounds=dict(status="match", before="default", after="default"),
                 os=dict(status="match", before=linux, after=linux)),
        V2: dict(check_bounds=dict(status="match", before="yes", after="yes"),
                 os=dict(status="match", before=darwin, after=darwin))}
    report = (root/"multi-match"/"REPORT.md").read_text()
    assert f"Recorded settings, Julia {V1}: `check_bounds` default on both sides; `os` {linux} on both sides." in report
    assert f"Recorded settings, Julia {V2}: `check_bounds` yes on both sides; `os` {darwin} on both sides." in report
    assert "verified" not in report
    # Missing on different sides in different versions: two unknowns, never a cross-version
    # comparison of default (1.10, before) with yes (1.11, after).
    compare(versions_fixture("crossed-before", 2, {V1: dict(check_bounds="default", os=linux), V2: dict(os=linux)}),
            versions_fixture("crossed-after", 1, {V1: dict(os=linux), V2: dict(check_bounds="yes", os=linux)}),
            root/"crossed", 3)
    crossed = settings_of("crossed")
    assert crossed[V1]["check_bounds"] == dict(status="unknown_after", before="default", after=None)
    assert crossed[V2]["check_bounds"] == dict(status="unknown_before", before=None, after="yes")
    assert crossed[V1]["os"]["status"] == crossed[V2]["os"]["status"] == "match"
    report = (root/"crossed"/"REPORT.md").read_text()
    assert f"Julia {V1}: `check_bounds` unknown on the after side (default on the before side)" in report
    assert f"Julia {V2}: `check_bounds` unknown on the before side (yes on the after side)" in report
    # One version fully matching, the other partly unknown: the unknowns stay with their version.
    compare(versions_fixture("mixed-before", 2, {V1: dict(check_bounds="default", os=linux), V2: dict(check_bounds="no")}),
            versions_fixture("mixed-after", 1, {V1: dict(check_bounds="default", os=linux), V2: {}}),
            root/"mixed", 3)
    assert statuses("mixed", V1) == dict(check_bounds="match", os="match")
    assert statuses("mixed", V2) == dict(check_bounds="unknown_after", os="unknown_both")
    lines = (root/"mixed"/"REPORT.md").read_text().splitlines()
    first = next(l for l in lines if l.startswith(f"Recorded settings, Julia {V1}:"))
    second = next(l for l in lines if l.startswith(f"Recorded settings, Julia {V2}:"))
    assert "verified" not in first
    assert "`check_bounds` and `os` could not be verified to match" in second
    # A known mismatch in one version rejects even when the other version is unknown.
    reject(versions_fixture("one-bad-before", 2, {V1: dict(check_bounds="default"), V2: {}}),
           versions_fixture("one-bad-after", 1, {V1: dict(check_bounds="no"), V2: dict(check_bounds="no")}),
           root/"one-bad", f"Julia {V1} check_bounds: before default; after no", expected=3)
    # Three-outcome rule (#462): one-sided 95% bounds on slowdown = after/before, from the paired
    # log ratios; pass below 1.05, investigate at >= 1.05 with the lower bound above 1.
    def outcome_of(name, slowdown, multipliers=None, runs=5, primary=None, evals=1000, factor=100, allocated=0,
                   reps=None):
        before, after = root/f"{name}-before", root/f"{name}-after"
        fixture(before, factor, runs, evals=evals)
        fixture(after, factor*slowdown, runs, evals=evals, multipliers=multipliers, allocated=allocated)
        assert validate(before, runs)["valid"] and validate(after, runs)["valid"]
        results = compare(before, after, root/name, runs, primary=primary, reps=reps)
        with (root/name/"outcomes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(results) == 1
        return rows[0]

    noise = [1.01, 0.99, 1.0, 1.02, 0.98]
    x = [math.log(1.0*m) for m in noise]                          # log(after/before) at slowdown 1
    margin = 2.132*statistics.stdev(x)/math.sqrt(5)
    row = outcome_of("rule-pass", 1.0, noise)
    assert row["timing_outcome"] == "pass" and row["role"] == "diagnostic" and row["reason"] == ""
    assert math.isclose(float(row["slowdown"]), math.exp(statistics.mean(x)))
    assert math.isclose(float(row["one_sided95_high"]), math.exp(statistics.mean(x)+margin))
    assert math.isclose(float(row["one_sided95_low"]), math.exp(statistics.mean(x)-margin))
    assert outcome_of("rule-investigate", 1.10, noise)["timing_outcome"] == "investigate"
    wide = [1.10, 0.90, 1.0, 1.15, 0.88]
    assert outcome_of("rule-inconclusive", 1.03, wide)["timing_outcome"] == "inconclusive"
    # Not significant is not a pass: a 4% slowdown with wide noise stays inconclusive.
    row = outcome_of("rule-not-significant", 1.04, wide)
    assert row["timing_outcome"] == "inconclusive" and float(row["one_sided95_low"]) < 1 < float(row["one_sided95_high"])
    # Boundaries without noise: 1.04 passes; exactly 1.05 is investigate, never pass.
    assert outcome_of("rule-1.04", 1.04)["timing_outcome"] == "pass"
    row = outcome_of("rule-1.05", 1.05)
    assert row["timing_outcome"] == "investigate" and math.isclose(float(row["one_sided95_high"]), 1.05)
    # A faster revision passes; a large slowdown with noise is investigate.
    assert outcome_of("rule-faster", 0.8, wide)["timing_outcome"] == "pass"
    assert outcome_of("rule-slower", 1.5, wide)["timing_outcome"] == "investigate"
    # A single run or a timer-limited case is inconclusive, with the reason.
    # Their observed slowdown is kept as a diagnostic; the bounds are unavailable.
    row = outcome_of("rule-single", 1.2, runs=1)
    assert (row["timing_outcome"], row["reason"]) == ("inconclusive", "single run")
    assert math.isclose(float(row["slowdown"]), 1.2)
    assert row["one_sided95_low"] == row["one_sided95_high"] == "nan"
    row = outcome_of("rule-limited", 1.0, noise, evals=1, factor=1)
    assert (row["timing_outcome"], row["reason"]) == ("inconclusive", "timer-limited")
    assert math.isclose(float(row["slowdown"]), math.exp(statistics.mean(x)))
    assert row["one_sided95_low"] == row["one_sided95_high"] == "nan"
    row = outcome_of("rule-limited-primary", 1.0, noise, evals=1, factor=1, primary={"analytic/control"})
    assert "| inconclusive; timer-limited | no |" in (root/"rule-limited-primary"/"REPORT.md").read_text()
    assert f"| {math.exp(statistics.mean(x)):.3f} [unavailable] |" in (root/"rule-limited-primary"/"REPORT.md").read_text()
    # Named primary cases: the role column and the report table; comparison.csv is unchanged.
    row = outcome_of("rule-primary", 1.0, noise, primary={"analytic/control"})
    assert row["role"] == "primary" and row["timing_outcome"] == "pass" and row["unexpected_allocation"] == "False"
    report = (root/"rule-primary"/"REPORT.md").read_text()
    assert "timing** outcomes, not an overall performance approval" in report
    assert "| pass | no |" in report and "Unexpected allocation in" not in report
    assert "Primary cases: 1 (pass 1, investigate 0, inconclusive 0)." in report
    assert "| `analytic/control` | julia-1.13.2 |" in report and "Diagnostic cases: 0" in report
    assert (root/"rule-primary"/"comparison.csv").read_text() == (root/"rule-pass"/"comparison.csv").read_text()
    assert "Primary cases: 0" in (root/"rule-pass"/"REPORT.md").read_text()
    with (root/"rule-pass"/"comparison.csv").open() as stream:
        assert next(csv.reader(stream)) == [
            "julia", "id", "paired_runs", "timer_limited", "timer_limited_by", "timer_limit_multiple",
            "before_evals", "after_evals", "before_mean_ns", "after_mean_ns", "paired_geometric_speedup",
            "ci95_low", "ci95_high", "timing_assessment", "before_mean_bytes", "after_mean_bytes",
            "mean_bytes_change", "before_max_bytes", "after_max_bytes", "unexpected_allocation"]
    # Timing passes, but the after side allocates in a case expected not to: the timing outcome is
    # unchanged, and the allocation is flagged beside it and above the primary table.
    row = outcome_of("rule-allocating", 1.0, noise, primary={"analytic/control"}, allocated=16)
    assert row["timing_outcome"] == "pass" and row["unexpected_allocation"] == "True"
    report = (root/"rule-allocating"/"REPORT.md").read_text()
    assert "**Unexpected allocation in 1 primary case(s)**, whatever their timing outcome: `analytic/control` (julia-1.13.2)." in report
    assert "| pass | **yes** |" in report
    assert report.index("**Unexpected allocation in") < report.index("| Primary case |")
    with (root/"rule-allocating"/"comparison.csv").open() as stream:
        assert next(csv.DictReader(stream))["unexpected_allocation"] == "True"
    # A named primary case that the comparison lacks is an error, not a silent omission.
    try:
        compare(root/"rule-pass-before", root/"rule-pass-after", root/"rule-typo", 5, primary={"analytic/typo"})
    except ValueError as error:
        assert "Primary cases not in the comparison: analytic/typo" in str(error)
        assert (root/"rule-typo"/"errors.json").exists() and not (root/"rule-typo"/"outcomes.csv").exists()
    else:
        raise AssertionError("an unknown primary case was accepted")
    # The command line reads the primary list from a file (comments and blank lines ignored).
    listing = root/"primary.txt"
    listing.write_text("# primary cases\n\nanalytic/control\n")
    subprocess.run([sys.executable, str(Path(__file__).with_name("compare.py")), str(root/"rule-pass-before"),
                    str(root/"rule-pass-after"), str(root/"rule-cli"), "--expected-runs", "5", "--primary", str(listing)],
                   check=True, capture_output=True)
    with (root/"rule-cli"/"outcomes.csv").open() as stream:
        assert next(csv.DictReader(stream))["role"] == "primary"
    assert not validate(root/"empty")["valid"]
    # Strict batch-time rule, including its boundary and unusable floors.
    for evals, median, limited in ((1, 120, True), (1, 450, False), (10, 100, False),
                                    (200, 60, False), (1, 300, False)):
        assert timer_limited(dict(evals=evals, median_ns=median), dict(timer_floor_ns=30)) == limited
    for values, step in (([0.001, 41.0, 42.0, 83.0], 41), ([30.0], 30),
                         ([25.0, 31.0, 30.0], 25), ([0.001], float("nan")),
                         ([0.4, 0.001], float("nan"))):
        calibrated = calibration_metadata(dict(start=values*100, end=values*100))
        for phase in ("start", "end"):
            observed = calibrated[f"timer_{phase}_step_ns"]
            assert math.isnan(observed) if math.isnan(step) else observed == step
        if values == [30.0]:
            assert calibrated["timer_floor_ns"] == 30
        if math.isnan(step):
            assert math.isnan(calibrated["timer_floor_ns"])
    skewed = calibration_metadata(dict(start=[0.001]*766 + [41.0]*78 + [42.0]*155,
                                       end=[0.001]*766 + [41.0]*78 + [42.0]*155))
    assert skewed["timer_start_median_ns"] == skewed["timer_end_median_ns"] == 0.001
    assert skewed["timer_start_step_ns"] == skewed["timer_end_step_ns"] == 41
    assert skewed["timer_floor_ns"] == 41
    for median in (83, 125):
        assert timer_limited(dict(evals=1, median_ns=median), skewed)
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
        path.write_text(saved.replace(key+"=30.0", key+"=0.0"))
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
    checked = validate(folder)
    assert not checked["valid"] and "missing calibration file" in "\n".join(checked["issues"])
    calibration.write_text(saved_calibration)
    for old, new in (("start\t1\t30", "start\t2\t30"), ("start\t1\t30", "other\t1\t30"),
                     ("start\t1\t30", "start\t1\tnan"), ("start\t1\t30", "start\t1\t-1"),
                     ("start\t1\t30", "start\t0\t30")):
        calibration.write_text(saved_calibration.replace(old, new, 1))
        assert not validate(folder)["valid"]
    calibration.write_text(saved_calibration)
    for values in ([30.0]*99, [0.0]*100, [0.001, 41.0, 42.0, 83.0]*25):
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
    # Older records remain inspectable, but cannot support a comparison.
    legacy = root/"legacy"
    fixture(legacy, 120, 1)
    prefix = legacy/"julia-1.13.2-rep01"
    meta_path = Path(str(prefix)+"-metadata.toml")
    saved_meta = meta_path.read_text()
    meta_path.write_text("\n".join(line for line in saved_meta.splitlines()
                                   if not line.startswith(("timer_", "calibration_", "tune_seconds")))+"\n")
    Path(str(prefix)+"-calibration.tsv").unlink()
    metrics = Path(str(prefix)+"-metrics.tsv")
    with metrics.open(newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    del row["evals"]
    write_rows(metrics, [row])
    checked = validate(legacy)
    assert not checked["valid"]
    messages = "\n".join(checked["issues"])
    assert "missing metadata key calibration_samples (results without timer calibration)" in messages
    assert "missing calibration file" in messages
    assert not summarize(legacy, root/"legacy-summary")
    status = json.loads((root/"legacy-summary"/"status.json").read_text())
    assert status["record_validation"] == checked
    with (root/"legacy-summary"/"summary.csv").open() as stream:
        assert next(csv.DictReader(stream))["timer_limited"] == "True"
    assert "unusable calibration in 1 processes" in (root/"legacy-summary"/"REPORT.md").read_text()
    reject(legacy, legacy, root/"legacy-comparison", "missing metadata key calibration_samples", expected=1)
    for evals, limited in ((1, True), (2, False)):
        assert timer_limited(dict(evals=evals, median_ns=120), {}) == limited
        metric_edit(legacy, 1, evals=evals)
        output = root/f"legacy-evals{evals}"
        assert not summarize(legacy, output)
        with (output/"summary.csv").open() as stream:
            assert next(csv.DictReader(stream))["timer_limited"] == str(limited)
    assert timer_limited(dict(median_ns=120), dict(timer_floor_ns=30))
    # A chosen subset of repetitions (--reps): each directory is validated whole, and only the
    # chosen repetitions enter the pairing, the intervals, the outcomes and the flags.
    script = str(Path(__file__).with_name("compare.py"))
    T975, T95 = {3: 3.182, 4: 2.776}, {3: 2.353, 4: 2.132}
    noisy = [1.01, 0.99, 1.0, 1.02, 0.98, 1.10, 0.90, 1.0, 1.15, 0.88]
    reps_before, reps_after = root/"reps-before", root/"reps-after"
    fixture(reps_before, 100, 10)
    fixture(reps_after, 100, 10, multipliers=noisy)

    def reps_outputs(name):
        rows = {}
        for filename in ("comparison.csv", "outcomes.csv"):
            with (root/name/filename).open() as stream:
                rows[filename] = list(csv.DictReader(stream))
        return (rows["comparison.csv"], rows["outcomes.csv"], (root/name/"REPORT.md").read_text().splitlines(),
                json.loads((root/name/"coverage.json").read_text()))

    def check_subset(name, reps, decision):
        compared, outcomes, report, coverage = reps_outputs(name)
        row, out = compared[0], outcomes[0]
        n = len(reps)
        ratios = [-math.log(noisy[r-1]) for r in reps]
        mu, half = statistics.mean(ratios), T975[n-1]*statistics.stdev(ratios)/math.sqrt(n)
        assert row["paired_runs"] == out["paired_runs"] == str(n)
        assert math.isclose(float(row["paired_geometric_speedup"]), math.exp(mu))
        assert math.isclose(float(row["ci95_low"]), math.exp(mu-half))
        assert math.isclose(float(row["ci95_high"]), math.exp(mu+half))
        assert math.isclose(float(row["before_mean_ns"]), statistics.mean(100*r for r in reps))
        assert math.isclose(float(row["after_mean_ns"]), statistics.mean(100*r*noisy[r-1] for r in reps))
        margin = T95[n-1]*statistics.stdev(ratios)/math.sqrt(n)
        assert math.isclose(float(out["slowdown"]), math.exp(-mu))
        assert math.isclose(float(out["one_sided95_low"]), math.exp(-mu-margin))
        assert math.isclose(float(out["one_sided95_high"]), math.exp(-mu+margin))
        assert out["timing_outcome"] == decision and out["reason"] == ""
        assert coverage["repetitions"] == reps and coverage["shared_measurements"] == n
        return report

    compare(reps_before, reps_after, root/"reps-1-5", 10, reps=[1, 2, 3, 4, 5])
    report = check_subset("reps-1-5", [1, 2, 3, 4, 5], "pass")
    assert report[2] == "Repetitions compared: 1-5 (5 of the 10 per Julia version in each directory)."
    compare(reps_before, reps_after, root/"reps-6-10", 10, reps=parse_reps(["6-10"]))
    report = check_subset("reps-6-10", [6, 7, 8, 9, 10], "inconclusive")
    assert report[2] == "Repetitions compared: 6-10 (5 of the 10 per Julia version in each directory)."
    compare(reps_before, reps_after, root/"reps-gap", 10, reps=parse_reps(["7", "1-3"]))
    report = check_subset("reps-gap", [1, 2, 3, 7], "pass")
    assert report[2] == "Repetitions compared: 1-3, 7 (4 of the 10 per Julia version in each directory)."
    compare(reps_before, reps_after, root/"reps-all", 10)
    compared, outcomes, report, coverage = reps_outputs("reps-all")
    assert report[2] == "Repetitions compared: 1-10 (all 10 per Julia version in each directory)."
    assert coverage["repetitions"] == list(range(1, 11)) and compared[0]["paired_runs"] == "10"
    assert any("A case is limited if any compared repetition on either side is limited." in l for l in report)
    compare(reps_before, reps_after, root/"reps-range", 10, reps=range(1, 11))
    for filename in ("REPORT.md", "comparison.csv", "outcomes.csv", "coverage.json", "settings.json"):
        assert (root/"reps-range"/filename).read_bytes() == (root/"reps-all"/filename).read_bytes()
    report = (root/"single-limit-report"/"REPORT.md").read_text().splitlines()
    assert report[2] == "Repetitions compared: 1 (all 1 per Julia version in each directory)."
    assert json.loads((root/"single-limit-report"/"coverage.json").read_text())["repetitions"] == [1]
    # One chosen repetition out of ten is a single run: exploratory, no interval or bounds.
    row = compare(reps_before, reps_after, root/"reps-single", 10, reps=[8])[0]
    assert row["paired_runs"] == 1 and row["timing_assessment"] == "single run; exploratory"
    assert math.isnan(row["ci95_low"]) and math.isnan(row["ci95_high"])
    compared, outcomes, report, coverage = reps_outputs("reps-single")
    assert report[2] == "Repetitions compared: 8 (1 of the 10 per Julia version in each directory)."
    assert f"| {1/noisy[7]:.3f} [unavailable] |" in "\n".join(report)
    assert (outcomes[0]["timing_outcome"], outcomes[0]["reason"], outcomes[0]["paired_runs"]) == ("inconclusive", "single run", "1")
    assert outcomes[0]["one_sided95_low"] == outcomes[0]["one_sided95_high"] == "nan"
    # Timer-limited repetitions inside and outside the subset: only the chosen ones count.
    limited_before, limited_after = root/"reps-limited-before", root/"reps-limited-after"
    fixture(limited_before, 10, 10)
    fixture(limited_after, 5, 10)
    for rep in (2, 7):
        metric_edit(limited_before, rep, evals=1)
    assert validate(limited_before, 10)["valid"]
    row = compare(limited_before, limited_after, root/"reps-limited-all", 10)[0]
    assert row["timer_limited_by"] == "before:rep02,rep07"
    row = compare(limited_before, limited_after, root/"reps-limited-in", 10, reps=[6, 7, 8, 9, 10])[0]
    assert row["timer_limited"] and row["timer_limited_by"] == "before:rep07"
    assert row["before_evals"] == "1/1000" and row["after_evals"] == "1000"
    assert row["timing_assessment"] == "timer-limited" and row["paired_runs"] == 5
    row = compare(limited_before, limited_after, root/"reps-limited-out", 10, reps=[8, 9, 10])[0]
    assert not row["timer_limited"] and row["timer_limited_by"] == ""
    assert row["before_evals"] == "1000" and row["timing_assessment"] == "faster"
    row = compare(limited_before, limited_after, root/"reps-limited-single", 10, reps=[7])[0]
    assert row["timing_assessment"] == "single run; exploratory; timer-limited"
    outcomes = reps_outputs("reps-limited-single")[1]
    assert (outcomes[0]["timing_outcome"], outcomes[0]["reason"]) == ("inconclusive", "timer-limited")
    # An allocation in one after repetition flags the case only when that repetition is chosen.
    alloc_before, alloc_after = root/"reps-alloc-before", root/"reps-alloc-after"
    fixture(alloc_before, 100, 10)
    fixture(alloc_after, 100, 10, allocated=[0]*9 + [16])
    assert validate(alloc_after, 10)["valid"]
    row = compare(alloc_before, alloc_after, root/"reps-alloc-out", 10, reps=[1, 2, 3, 4, 5])[0]
    assert not row["unexpected_allocation"]
    assert row["after_mean_bytes"] == row["after_max_bytes"] == row["mean_bytes_change"] == 0
    row = compare(alloc_before, alloc_after, root/"reps-alloc-in", 10, reps=[6, 7, 8, 9, 10])[0]
    assert row["unexpected_allocation"] and row["after_max_bytes"] == 16
    assert math.isclose(row["after_mean_bytes"], 3.2) and math.isclose(row["mean_bytes_change"], 3.2)
    assert reps_outputs("reps-alloc-in")[1][0]["unexpected_allocation"] == "True"
    # Two Julia versions: the subset applies to each.
    versions_before, versions_after = root/"reps-versions-before", root/"reps-versions-after"
    for version in (V1, V2):
        fixture(versions_before, 2, 10, version=version)
    fixture(versions_after, 1, 10, version=V1)
    fixture(versions_after, 1, 10, version=V2, multipliers=noisy)
    rows = compare(versions_before, versions_after, root/"reps-versions", 10, reps=[6, 7, 8, 9, 10])
    assert [(r["julia"], r["paired_runs"]) for r in rows] == [(f"julia-{V1}", 5), (f"julia-{V2}", 5)]
    assert math.isclose(rows[0]["paired_geometric_speedup"], 2)
    assert math.isclose(rows[1]["paired_geometric_speedup"],
                        2/math.exp(statistics.mean(math.log(m) for m in noisy[5:])))
    compared, outcomes, report, coverage = reps_outputs("reps-versions")
    assert [o["paired_runs"] for o in outcomes] == ["5", "5"] and coverage["shared_measurements"] == 10
    assert report[2] == "Repetitions compared: 6-10 (5 of the 10 per Julia version in each directory)."
    # Named primary cases with a subset: the role, the table and the outcome follow the subset.
    row = outcome_of("reps-primary-first", 1.0, noisy, runs=10, primary={"analytic/control"}, reps=[1, 2, 3, 4, 5])
    assert (row["role"], row["timing_outcome"], row["paired_runs"]) == ("primary", "pass", "5")
    report = (root/"reps-primary-first"/"REPORT.md").read_text()
    assert "Primary cases: 1 (pass 1, investigate 0, inconclusive 0)." in report and "| pass | no |" in report
    row = outcome_of("reps-primary-last", 1.0, noisy, runs=10, primary={"analytic/control"}, reps=[6, 7, 8, 9, 10])
    assert (row["role"], row["timing_outcome"]) == ("primary", "inconclusive")
    assert "Primary cases: 1 (pass 0, investigate 0, inconclusive 1)." in (root/"reps-primary-last"/"REPORT.md").read_text()
    # Invalid selections are rejected before the output directory is created.
    for reps, message, expected in (([11], "repetitions [11] are outside 1..10 (--expected-runs 10)", 10),
                                    ([0, 12], "repetitions [0, 12] are outside 1..10", 10),
                                    ([6], "repetitions [6] are outside 1..5 (--expected-runs 5)", 5),
                                    ([6, 6], "repetitions [6] are selected more than once", 10),
                                    (parse_reps(["1-5", "3-7"]), "repetitions [3, 4, 5] are selected more than once", 10),
                                    ([], "selects no repetition", 10),
                                    ([True], "must be integers", 10), ([6.0], "must be integers", 10),
                                    (["6"], "must be integers", 10), ("6-10", "parse strings with parse_reps", 10),
                                    ([11], "expected runs must be between 1 and 10", 11),
                                    ([1], "expected runs must be between 1 and 10", 0)):
        output = root/f"reps-invalid-{len(list(root.iterdir()))}"
        try:
            compare(reps_before, reps_after, output, expected, reps=reps)
        except ValueError as error:
            assert message in str(error), error
            assert not output.exists()
        else:
            raise AssertionError(f"invalid repetitions accepted: {reps!r}")
    for token, message in (("7-6", "reversed range '7-6'"), ("-1", "malformed"), ("1--3", "malformed"),
                           ("6_0", "malformed"), ("٣", "malformed"), ("", "malformed"), (" 6", "malformed"),
                           ("+6", "malformed"), ("11", "repetition 11 is above 10, the largest --expected-runs"),
                           ("1-99999999", "repetition 99999999 is above 10"),
                           ("9"*5000, "--reps:"), ("1-" + "9"*5000, "--reps:")):
        try:
            parse_reps(["1", token])
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f"malformed repetition accepted: {token!r}")
    assert parse_reps(["10", "6-9"]) == [10, 6, 7, 8, 9] and parse_reps(["6-6"]) == [6]
    # The command line: errors exit 2 with the message; equivalent forms give identical files.
    def cli(name, *reps, expected=10):
        command = [sys.executable, script, str(reps_before), str(reps_after), str(root/name),
                   "--expected-runs", str(expected)]
        return subprocess.run(command + [arg for spec in reps for arg in ("--reps", *spec)],
                              capture_output=True, text=True, timeout=60)

    for reps, message, expected in (([["6"]], "outside 1..5 (--expected-runs 5)", 5),
                                    ([["11"]], "above 10, the largest --expected-runs", 10),
                                    ([["6"], ["6"]], "more than once", 10), ([["1-5", "3-7"]], "more than once", 10),
                                    ([["7-6"]], "reversed range", 10), ([["-1"]], "malformed", 10),
                                    ([["6_0"]], "malformed", 10), ([[""]], "malformed", 10),
                                    ([["1-99999999"]], "above 10", 10),
                                    ([["1-" + "9"*5000]], "--reps:", 10)):
        name = f"reps-cli-invalid-{len(list(root.iterdir()))}"
        failed = cli(name, *reps, expected=expected)
        assert failed.returncode == 2 and message in failed.stderr, (reps, failed.stderr)
        assert not (root/name).exists()
    for name, reps in (("reps-cli-range", [["6-10"]]), ("reps-cli-list", [["6", "7", "8", "9", "10"]]),
                       ("reps-cli-unsorted", [["10", "6-9"]]), ("reps-cli-repeated", [["6-8"], ["9", "10"]]),
                       ("reps-cli-full", [["1-10"]]), ("reps-cli-none", [])):
        assert cli(name, *reps).returncode == 0
    for filename in ("REPORT.md", "comparison.csv", "outcomes.csv", "coverage.json", "settings.json"):
        for name in ("reps-cli-list", "reps-cli-unsorted", "reps-cli-repeated", "reps-6-10"):
            assert (root/name/filename).read_bytes() == (root/"reps-cli-range"/filename).read_bytes(), (name, filename)
        assert (root/"reps-cli-full"/filename).read_bytes() == (root/"reps-cli-none"/filename).read_bytes()
        assert (root/"reps-cli-none"/filename).read_bytes() == (root/"reps-all"/filename).read_bytes()

# The baseline tools get their own interpreter: this script imports the modern validate.py, and the
# baseline checks must not (they use the frozen historical rule).
historical = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve().parent/"baselines/tools/check_historical.py")],
                            capture_output=True, text=True, timeout=120)
assert historical.returncode == 0, f"check_historical.py exited with {historical.returncode}\n{historical.stdout}\n{historical.stderr}"
print(historical.stdout.strip().splitlines()[-1])

print("ANALYSIS_CONTROLS_PASSED repetitions=1,3,5,10; means,SD,t-intervals,paired-ratios,allocations,missing/wrong-repetitions,raw-corruption,unmatched-cases,settings-mismatch,check_bounds/os-match/conflict/unknown/partial/within-run/literal-sentinel/per-version-swap/match/crossed-unknown/mixed; three-outcome pass/investigate/inconclusive/boundaries/single-run/timer-limited/diagnostic-slowdown/primary/allocation-flag/unknown-primary/cli; timer-rule,boundary,mixed-evals,limited-sides,K-overrides,invalid-K,process-floors,unusable/constant/stepped-calibration,missing-calibration,legacy-records,quantized-tick,corrupt-median/evals/floor,eligible-2x,limited-ratio/allocation-warning; reps subsets/intervals/bounds/non-contiguous/header/coverage/full-identity/single-run/limited-in-out/single-limited/allocation-in-out/two-versions/primary/api-errors/parse-errors/cli-errors/cli-forms")
