#!/usr/bin/env python3
"""Check coverage, metadata and consistency with retained raw observations."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tomllib

# Source may change in a comparison; instrumentation and measurement settings may not.
MATCH_SETTINGS = ("julia", "cpu", "cpu_model", "blas", "blas_threads", "julia_threads",
                  "benchmarktools", "seed_scheme", "samples_target", "seconds_limit",
                  "profile_seconds", "harness_hash", "suite", "groups",
                  "calibration_samples", "calibration_seconds", "tune_seconds")


def discover_versions(root):
    return sorted({p.name.removeprefix("julia-").split("-rep")[0]
                   for p in Path(root).glob("*-metadata.toml")})


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def calibration_metadata(phases):
    result = {}
    for phase, values in phases.items():
        rounded = sorted({round(value) for value in values})
        result[f"timer_{phase}_median_ns"] = statistics.median(values)
        result[f"timer_{phase}_min_ns"] = min(values)
        result[f"timer_{phase}_step_ns"] = min(
            (b-a for a, b in zip(rounded, rounded[1:])), default=float("nan"))
    steps = [result[f"timer_{phase}_step_ns"] for phase in ("start", "end")]
    medians = [result[f"timer_{phase}_median_ns"] for phase in ("start", "end")]
    unusable = (any(len(values) < 100 for values in phases.values()) or
                all(math.isnan(step) for step in steps) and all(value < 1 for value in medians))
    result["timer_floor_ns"] = (float("nan") if unusable else
                               max(medians + [step for step in steps if not math.isnan(step)]))
    return result


def check_timer_limit_multiple(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timer limit multiple must be positive and finite")


def timer_limited(row, meta, multiple=10):
    check_timer_limit_multiple(multiple)
    floor = float(meta["timer_floor_ns"])
    evals = int(row["evals"])
    return evals == 1 if math.isnan(floor) else float(row["median_ns"]) * evals < multiple * floor


def validate(root, expected=1, versions=None):
    root = Path(root)
    versions = discover_versions(root) if versions is None else versions
    issues, configurations, catalogs = [], {}, {}
    if not 1 <= expected <= 10:
        issues.append("expected runs must be between 1 and 10")
    if not versions:
        issues.append("no process metadata found")
    counts = dict(processes=0, measurements=0, timing_samples=0, allocation_calls=0)
    wanted = {f"julia-{v}-rep{r:02d}" for v in versions for r in range(1, expected + 1)}
    found = {p.name.removesuffix("-metadata.toml") for p in root.glob("*-metadata.toml")}
    if found != wanted:
        issues.append(f"process coverage: missing={sorted(wanted-found)}, extra={sorted(found-wanted)}")
    for prefix in sorted(wanted & found):
        try:
            with (root / (prefix + "-metadata.toml")).open("rb") as stream:
                meta = tomllib.load(stream)
            version, repetition = prefix.removeprefix("julia-").split("-rep")
            if str(meta["julia"]) != version or int(meta["rep"]) != int(repetition):
                raise ValueError("metadata version/repetition disagrees with filename")
            if meta["failures"] or meta["suite"] not in ("quick", "extended"):
                raise ValueError("failed process or unknown suite")
            if meta["julia_threads"] != 1 or meta["blas_threads"] != 1:
                raise ValueError("expected one Julia and BLAS thread")
            if meta.get("expected_package_root") and meta.get("source_path") != meta["expected_package_root"]:
                raise ValueError("loaded package root differs from the requested source")
            for key in ("source_hash", "harness_hash"):
                if len(meta[key]) != 40 or any(c not in "0123456789abcdef" for c in meta[key]):
                    raise ValueError(f"missing content provenance: {key}")
            config = tuple(str(meta[key]) for key in MATCH_SETTINGS + ("source_hash",))
            if version in configurations and configurations[version] != config:
                raise ValueError("settings changed within a Julia version")
            configurations[version] = config
            phases = {"start": {}, "end": {}}
            for row in records(root / (prefix + "-calibration.tsv")):
                phase = row["phase"]
                index, value = int(row["sample"]), float(row["time_ns"])
                if phase not in phases or index in phases[phase] or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid calibration observation")
                phases[phase][index] = value
            for values in phases.values():
                if not values or set(values) != set(range(1, len(values)+1)):
                    raise ValueError("calibration sample coverage mismatch")
            calibrated = calibration_metadata({phase: list(values.values()) for phase, values in phases.items()})
            for key, value in calibrated.items():
                recorded = float(meta[key])
                if not (math.isnan(recorded) and math.isnan(value) or
                        math.isclose(recorded, value, rel_tol=1e-12, abs_tol=1e-7)):
                    raise ValueError(f"{key} disagrees with raw calibration")
            catalog_rows = records(root / (prefix + "-catalog.tsv"))
            if any(r["evals_policy"] not in ("auto", "fixed") for r in catalog_rows):
                raise ValueError("unknown evals_policy in catalog")
            catalog = {r["id"] for r in catalog_rows}
            if not catalog or len(catalog) != len(catalog_rows) or len(catalog) != meta["cases"]:
                raise ValueError("catalog duplicates or case-count mismatch")
            if version in catalogs and catalogs[version] != catalog:
                raise ValueError("case catalog changed between repetitions")
            catalogs[version] = catalog
            if records(root / (prefix + "-errors.tsv")):
                raise ValueError("case errors retained")
            metrics = records(root / (prefix + "-metrics.tsv"))
            if len(metrics) != len(catalog) or {r["id"] for r in metrics} != catalog:
                raise ValueError("metrics do not cover the catalog exactly once")
            raw = {}
            for suffix, value in (("samples", "time_ns"), ("allocations", "bytes")):
                grouped = {}
                for row in records(root / (prefix + f"-{suffix}.tsv")):
                    if int(row["rep"]) != int(repetition) or row["id"] not in catalog:
                        raise ValueError(f"foreign observation in {suffix}")
                    values = grouped.setdefault(row["id"], {})
                    index = int(row["sample"])
                    observation = float(row[value])
                    if index in values or not math.isfinite(observation) or observation < 0:
                        raise ValueError(f"duplicate or invalid observation in {suffix}")
                    values[index] = observation
                raw[suffix] = grouped
            for row in metrics:
                case = row["id"]
                if int(row["rep"]) != int(repetition):
                    raise ValueError(f"wrong repetition in {case}")
                if int(row["evals"]) <= 0:
                    raise ValueError(f"evals must be a positive integer in {case}")
                nsamples = int(row["samples"])
                times = raw["samples"].get(case, {})
                allocs = raw["allocations"].get(case, {})
                if set(times) != set(range(1, nsamples + 1)) or nsamples < 1:
                    raise ValueError(f"timing sample coverage mismatch in {case}")
                if set(allocs) != set(range(1, 11)):
                    raise ValueError(f"allocation sample coverage mismatch in {case}")
                if not math.isclose(float(row["median_ns"]), statistics.median(times.values()),
                                    rel_tol=1e-12, abs_tol=1e-7):
                    raise ValueError(f"median_ns disagrees with raw data in {case}")
                for values, keys in ((list(times.values()), ("mean_ns", "min_ns", "max_ns")),
                                     (list(allocs.values()), ("mean_bytes", "min_bytes", "max_bytes"))):
                    for key, observed in zip(keys, (statistics.mean(values), min(values), max(values))):
                        if not math.isclose(float(row[key]), observed, rel_tol=1e-12, abs_tol=1e-7):
                            raise ValueError(f"{key} disagrees with raw data in {case}")
                counts["timing_samples"] += nsamples
                counts["allocation_calls"] += len(allocs)
            counts["processes"] += 1
            counts["measurements"] += len(metrics)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"{prefix}: {error}")
    return dict(valid=not issues, expected_runs=expected, versions=list(versions),
                cases_per_version={v: len(c) for v, c in catalogs.items()}, **counts, issues=issues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("output")
    parser.add_argument("--expected-runs", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--versions", nargs="+")
    args = parser.parse_args()
    result = validate(args.results, args.expected_runs, args.versions)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 2)
