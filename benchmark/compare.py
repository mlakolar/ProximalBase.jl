#!/usr/bin/env python3
"""Compare matching revisions under the same benchmark definitions and settings."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tomllib

from summarize import mean_interval
from validate import MATCH_SETTINGS, check_timer_limit_multiple, timer_limited, validate


def load(root, expected):
    checked = validate(root, expected)
    if not checked["valid"]:
        raise ValueError(f"Invalid results {root}: {checked['issues']}")
    rows, configs, metadata = {}, set(), {}
    for path in sorted(Path(root).glob("*-metadata.toml")):
        with path.open("rb") as stream:
            meta = tomllib.load(stream)
        metadata[path.name.removesuffix("-metadata.toml")] = meta
        configs.add(tuple(str(meta[key]) for key in MATCH_SETTINGS))
    for path in sorted(Path(root).glob("*-metrics.tsv")):
        version = path.name.split("-rep")[0]
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                row["_metadata"] = metadata[path.name.removesuffix("-metrics.tsv")]
                rows[(version, row["id"], int(row["rep"]))] = row
    return rows, configs


def compare(before_root, after_root, output, expected=1, timer_limit_multiple=10):
    check_timer_limit_multiple(timer_limit_multiple)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh comparison directory to avoid stale reports")
    output.mkdir(parents=True, exist_ok=True)
    try:
        before, old_configs = load(before_root, expected)
        after, new_configs = load(after_root, expected)
        coverage = dict(shared_measurements=len(before.keys() & after.keys()),
                        before_only=sorted(before.keys()-after.keys()),
                        after_only=sorted(after.keys()-before.keys()))
        (output/"coverage.json").write_text(json.dumps(coverage, indent=2)+"\n")
        if old_configs != new_configs:
            raise ValueError("Benchmark configuration differs; match hardware, suite, groups, instrumentation and measurement settings")
        if coverage["before_only"] or coverage["after_only"]:
            raise ValueError("Case coverage differs; inspect coverage.json and rerun the same cases")
        groups = {}
        for key in sorted(before):
            groups.setdefault(key[:2], []).append((before[key], after[key]))
        results = []
        for (version, case), pairs in groups.items():
            old_times = [float(old["mean_ns"]) for old, _ in pairs]
            new_times = [float(new["mean_ns"]) for _, new in pairs]
            if min(old_times+new_times) <= 0:
                raise ValueError(f"Nonpositive timing in {case}")
            ratios = [math.log(old/new) for old, new in zip(old_times, new_times)]
            mu, half = statistics.mean(ratios), mean_interval(ratios)
            old_bytes = statistics.mean(float(old["mean_bytes"]) for old, _ in pairs)
            new_bytes = statistics.mean(float(new["mean_bytes"]) for _, new in pairs)
            old_max = max(float(old["max_bytes"]) for old, _ in pairs)
            new_max = max(float(new["max_bytes"]) for _, new in pairs)
            unexpected = any(new["expected_zero"] == "true" and float(new["max_bytes"]) > 0
                             for _, new in pairs)
            limited_by = []
            for side, index in (("before", 0), ("after", 1)):
                reps = [f"rep{int(pair[index]['rep']):02d}" for pair in pairs
                        if timer_limited(pair[index], pair[index]["_metadata"], timer_limit_multiple)]
                if reps:
                    limited_by.append(side + ":" + ",".join(reps))
            limited = bool(limited_by)
            assessment = ("single run; exploratory" if expected == 1 else
                          "inconclusive" if mu-half <= 0 <= mu+half else
                          "faster" if mu > 0 else "slower")
            if limited:
                assessment = "single run; exploratory; timer-limited" if expected == 1 else "timer-limited"
            results.append(dict(julia=version, id=case, paired_runs=expected,
                timer_limited=limited, timer_limited_by=";".join(limited_by),
                timer_limit_multiple=timer_limit_multiple,
                before_evals="/".join(str(v) for v in sorted({int(old["evals"]) for old, _ in pairs})),
                after_evals="/".join(str(v) for v in sorted({int(new["evals"]) for _, new in pairs})),
                before_mean_ns=statistics.mean(old_times), after_mean_ns=statistics.mean(new_times),
                paired_geometric_speedup=math.exp(mu), ci95_low=math.exp(mu-half),
                ci95_high=math.exp(mu+half), timing_assessment=assessment,
                before_mean_bytes=old_bytes, after_mean_bytes=new_bytes,
                mean_bytes_change=new_bytes-old_bytes,
                before_max_bytes=old_max, after_max_bytes=new_max,
                unexpected_allocation=unexpected))
        with (output/"comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(results)
        lines = ["# ProximalBase revision comparison", "",
                 "Ratios above one favor the proposed revision. Intervals use paired log ratios across fresh process/input draws; they do not correct for correlated machine noise or multiple comparisons. Interleave revisions when confirming small differences.", "",
                 "Single-run ratios are exploratory and have no confidence interval. Allocation counts measure Julia heap bytes, not native allocations or peak memory.", "",
                 f"A measurement is timer-limited when median_ns * evals < {timer_limit_multiple:g} * timer_floor_ns (strict). With an unusable calibration (NaN floor), only evals == 1 is limited. A case is limited if any repetition on either side is limited. Timer-limited cases are not to be used for ratio gates.", "",
                 "| Case | Julia | Before / after (µs) | Speedup [95% interval] | Mean bytes before / after | Maximum bytes before / after | Assessment |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for row in results:
            interval = ("unavailable" if expected == 1 else
                        f"{row['ci95_low']:.3f}, {row['ci95_high']:.3f}")
            note = row["timing_assessment"]
            if row["unexpected_allocation"]:
                note += "; unexpected allocation"
            lines.append(f"| `{row['id']}` | {row['julia']} | {row['before_mean_ns']/1000:.3f} / {row['after_mean_ns']/1000:.3f} | {row['paired_geometric_speedup']:.3f} [{interval}] | {row['before_mean_bytes']:.1f} / {row['after_mean_bytes']:.1f} | {row['before_max_bytes']:.0f} / {row['after_max_bytes']:.0f} | {note} |")
        (output/"REPORT.md").write_text("\n".join(lines)+"\n")
        return results
    except (ValueError, OSError, KeyError) as error:
        (output/"errors.json").write_text(json.dumps({"error": str(error)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before"); parser.add_argument("after"); parser.add_argument("output")
    parser.add_argument("--expected-runs", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--timer-limit-multiple", type=float, default=10)
    args = parser.parse_args()
    try:
        compare(args.before, args.after, args.output, args.expected_runs, args.timer_limit_multiple)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")
