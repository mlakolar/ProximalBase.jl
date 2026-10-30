#!/usr/bin/env python3
"""Aggregate independent run means; never select the fastest run."""
import argparse
import csv
import glob
import json
import math
from pathlib import Path
import statistics as st
import tomllib
from validate import check_timer_limit_multiple, discover_versions, timer_limited, validate

# Two-sided 95% Student-t critical values, for 2--10 independent runs.
T975 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}


def mean_interval(values):
    if len(values) == 1:
        return float("nan")
    return T975[len(values)-1] * st.stdev(values) / math.sqrt(len(values))

def read_rows(pattern):
    rows = []
    for filename in sorted(glob.glob(pattern)):
        with open(filename, newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                row["_file"] = filename
                rows.append(row)
    return rows

def summarize(root, output, expected=1, versions=None, timer_limit_multiple=10):
    check_timer_limit_multiple(timer_limit_multiple)
    root, output = Path(root), Path(output)
    versions = discover_versions(root) if versions is None else versions
    if not 1 <= expected <= 10:
        raise ValueError("expected runs must be between 1 and 10")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(str(root / "*-metrics.tsv"))
    errors = read_rows(str(root / "*-errors.tsv"))
    metadata = {}
    floors = {}
    for path in sorted(root.glob("*-metadata.toml")):
        with path.open("rb") as stream:
            meta = tomllib.load(stream)
        metadata[path.name.removesuffix("-metadata.toml")] = meta
        floors.setdefault(path.name.split("-rep")[0], []).append(float(meta.get("timer_floor_ns", float("nan"))))
    groups = {}
    catalogs = {}
    for row in read_rows(str(root / "*-catalog.tsv")):
        version = Path(row["_file"]).name.split("-rep")[0]
        catalogs.setdefault(version, set()).add(row["id"])
    for row in rows:
        version = Path(row["_file"]).name.split("-rep")[0]
        groups.setdefault((version, row["id"]), []).append(row)
    results = []
    duplicate_groups = []
    for (version, case_id), group in sorted(groups.items()):
        reps = [int(row["rep"]) for row in group]
        if len(set(reps)) != len(reps):
            duplicate_groups.append((version, case_id))
        values = [float(row["mean_ns"]) for row in group]
        average = st.mean(values)
        sd = st.stdev(values) if len(values)>1 else float("nan")
        interval = mean_interval(values) if len(values)<=10 else float("nan")
        first = group[0]
        result = {key:first[key] for key in ("id","family","name","dtype","shape","layout","phase","expected_zero")}
        result.update(julia=version, runs=len(values), complete=len(reps)==expected and set(reps)==set(range(1,expected+1)),
            timer_limited=any(timer_limited(r, metadata.get(Path(r["_file"]).name.removesuffix("-metrics.tsv"), {}),
                                            timer_limit_multiple) for r in group),
            mean_ns=average, sd_between_runs_ns=sd, ci95_low_ns=max(0,average-interval) if math.isfinite(interval) else float("nan"),
            ci95_high_ns=average+interval, cv=sd/average if average else 0,
            median_run_median_ns=st.median(float(r["median_ns"]) for r in group),
            min_run_mean_ns=min(values), max_run_mean_ns=max(values),
            mean_bytes=st.mean(float(r["mean_bytes"]) for r in group),
            max_bytes=max(float(r["max_bytes"]) for r in group),
            min_samples=min(int(r["samples"]) for r in group),
            max_gc_share=max(float(r["gc_share"]) for r in group))
        results.append(result)
    if not results:
        raise RuntimeError("No valid benchmark measurements")
    with open(output/"summary.csv", "w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(results)
    incomplete = [{"julia":version,"id":case_id,"valid_runs":len(groups.get((version,case_id),[]))}
                  for version, ids in catalogs.items() for case_id in sorted(ids)
                  if (len(groups.get((version,case_id),[]))!=expected or
                      {int(r["rep"]) for r in groups.get((version,case_id),[])}!=set(range(1,expected+1)))]
    unexpected = [r for r in results if r["expected_zero"]=="true" and r["max_bytes"]>0]
    variable = sorted([r for r in results if r["cv"]>0.15],key=lambda r:r["cv"],reverse=True)
    missing_versions=sorted(set("julia-"+v for v in versions)-set(catalogs))
    status={"measurement_rows":len(rows),"case_version_groups":len(results),
        "expected_runs":expected,"versions":sorted(catalogs),"missing_versions":missing_versions,
        "cases_per_version":{v:len(ids) for v,ids in catalogs.items()},
        "errors":errors,"incomplete":incomplete,"duplicate_groups":duplicate_groups,
        "unexpected_allocation_groups":len(unexpected),"high_variability_groups":len(variable)}
    checked = validate(root, expected, versions)
    status["record_validation"] = checked
    (output/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    calibration_lines = []
    for version, values in sorted(floors.items()):
        usable = [value for value in values if math.isfinite(value)]
        span = f"{min(usable):g}–{max(usable):g}" if usable else "NaN–NaN"
        calibration_lines.append(f"{version}: timer_floor_ns range {span} ns; unusable calibration in {len(values)-len(usable)} processes.")
    lines=["# ProximalBase benchmark results", "",
        f"{len(rows):,} valid case/run measurements; {len(results):,} case/version groups. Target: {expected} independent runs per case.", "",
        "The primary time is the arithmetic mean of run means. Each run uses a fresh Julia process and a keyed seeded input draw. The standard deviation and Student-t interval use independent runs, not individual samples. A single run has no uncertainty estimate (NaN in the interval columns). Input variability and machine noise are both included. No adverse run is discarded.", "",
        "Within each run, BenchmarkTools collects up to the configured sample/time limit. Compilation, validation and prepare callbacks are excluded. Cold cases include construction and the first proximal call. CPU profiles include prepare callbacks; allocation profiles isolate the measured operation. These CPU measurements do not cover GPU hardware or every parameter combination.", "",
        f"Validation errors: {len(errors)}. Incomplete case/version groups: {len(incomplete)}. Duplicate groups: {len(duplicate_groups)}. Record validation passed: {checked['valid']}.", "",
        *calibration_lines, "",
        f"Timer-limited measurements have median_ns * evals < {timer_limit_multiple:g} * timer_floor_ns (strict); a missing or NaN floor limits only evals == 1. Missing evals is timer-limited. Any limited repetition limits the case/version. Ratio gates must filter on timer_limited.", "",
        "## Unexpected warmed allocations", "",
        "These are candidates for inspection, not automatically confirmed implementation defects.", "",
        "| Case | Julia | Mean bytes | Maximum bytes | Runs |", "|---|---|---:|---:|---:|"]
    for r in sorted(unexpected,key=lambda r:r["max_bytes"],reverse=True)[:40]:
        lines.append(f"| `{r['id']}` | {r['julia']} | {r['mean_bytes']:.1f} | {r['max_bytes']:.0f} | {r['runs']} |")
    if not unexpected: lines.append("| None observed | | 0 | 0 | |")
    lines += ["", "## Variable cases", "",
        "CV is the between-run standard deviation divided by the mean. Large CV can reflect input-dependent work, short timing resolution, or shared-node noise.", "",
        "| Case | Julia | Mean ± SD (µs) | CV | Runs |", "|---|---|---:|---:|---:|"]
    for r in variable[:25]:
        lines.append(f"| `{r['id']}` | {r['julia']} | {r['mean_ns']/1000:.3f} ± {r['sd_between_runs_ns']/1000:.3f} | {r['cv']:.1%} | {r['runs']} |")
    lines += ["", "## Slowest measured calls", "",
        "Absolute cost is not evidence of a defect; decompositions and initialization are expected to be expensive.", "",
        "| Case | Julia | Mean (ms) | 95% interval (ms) | Runs |", "|---|---|---:|---:|---:|"]
    for r in sorted(results,key=lambda r:r["mean_ns"],reverse=True)[:25]:
        lines.append(f"| `{r['id']}` | {r['julia']} | {r['mean_ns']/1e6:.3f} | [{r['ci95_low_ns']/1e6:.3f}, {r['ci95_high_ns']/1e6:.3f}] | {r['runs']} |")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in status.items() if k not in ("errors","incomplete","duplicate_groups")},indent=2))
    return checked["valid"] and not errors and not incomplete and not duplicate_groups and not missing_versions

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results");parser.add_argument("output")
    parser.add_argument("--expected-runs",type=int,choices=range(1,11),default=1)
    parser.add_argument("--versions",nargs="+")
    parser.add_argument("--timer-limit-multiple",type=float,default=10)
    args=parser.parse_args()
    raise SystemExit(0 if summarize(args.results,args.output,args.expected_runs,args.versions,args.timer_limit_multiple) else 2)
