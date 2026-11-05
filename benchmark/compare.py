#!/usr/bin/env python3
"""Compare matching revisions under the same benchmark definitions and settings."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import tomllib

from summarize import mean_interval
from validate import (MATCH_SETTINGS, RECORDED_SETTINGS, check_timer_limit_multiple,
                      recorded_setting, timer_limited, validate)


# The three-outcome rule of #462 on rho = after/before, with one-sided 95% bounds from the paired
# log ratios: pass when the upper bound is below 1.05; investigate when rho-hat >= 1.05 and the
# lower bound is above 1 (confirm independently); otherwise inconclusive. A case never passes
# because nothing was significant. These are timing outcomes only: allocations, correctness and
# complexity are judged separately (outcomes.csv carries unexpected_allocation beside them).
SLOWDOWN_LIMIT = 1.05
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}


def outcome(ratios, limited):
    """(timing outcome, slowdown, lower, upper, reason) for paired log ratios log(before/after).

    A timer-limited case or a single run is inconclusive without bounds, but keeps its observed
    slowdown as a diagnostic."""
    x = [-r for r in ratios]   # log(after/before)
    mean = statistics.mean(x)
    slowdown = math.exp(mean)
    if limited:
        return "inconclusive", slowdown, math.nan, math.nan, "timer-limited"
    if len(x) < 2:
        return "inconclusive", slowdown, math.nan, math.nan, "single run"
    margin = T95[len(x)-1]*statistics.stdev(x)/math.sqrt(len(x))
    lower, upper = math.exp(mean-margin), math.exp(mean+margin)
    if upper < SLOWDOWN_LIMIT:
        return "pass", slowdown, lower, upper, ""
    if slowdown >= SLOWDOWN_LIMIT and lower > 1:
        return "investigate", slowdown, lower, upper, ""
    return "inconclusive", slowdown, lower, upper, ""


def read_primary(path):
    """Case IDs, one per line; blank lines and lines starting with # are ignored."""
    lines = (line.strip() for line in Path(path).read_text().splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def parse_reps(tokens):
    """Repetition numbers from --reps tokens, each an integer k or an inclusive range a-b.

    Numbers above 10 (the largest --expected-runs) are rejected before a range is expanded.
    Bounds, duplicates and emptiness are checked by select_reps."""
    reps = []
    for token in tokens:
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", token, re.ASCII):
            raise ValueError(f"--reps: malformed repetition {token!r}; use k or a-b")
        first, _, last = token.partition("-")
        first, last = int(first), int(last or first)
        if max(first, last) > 10:
            raise ValueError(f"--reps: repetition {max(first, last)} is above 10, the largest --expected-runs")
        if first > last:
            raise ValueError(f"--reps: reversed range {token!r}")
        reps += range(first, last+1)
    return reps


def select_reps(reps, expected):
    """The sorted repetitions to compare: all of 1..expected when reps is None.

    reps holds plain integers, each in 1..expected and named once; strings go through parse_reps."""
    if reps is None:
        return list(range(1, expected+1))
    if not 1 <= expected <= 10:
        raise ValueError("expected runs must be between 1 and 10")
    if isinstance(reps, str):
        raise ValueError("reps must be integers; parse strings with parse_reps")
    reps = list(reps)
    if any(type(r) is not int for r in reps):
        raise ValueError(f"reps must be integers: {reps}")
    if not reps:
        raise ValueError("--reps selects no repetition")
    outside = sorted({r for r in reps if not 1 <= r <= expected})
    if outside:
        raise ValueError(f"repetitions {outside} are outside 1..{expected} (--expected-runs {expected})")
    repeated = sorted({r for r in reps if reps.count(r) > 1})
    if repeated:
        raise ValueError(f"repetitions {repeated} are selected more than once in --reps")
    return sorted(reps)


def reps_note(selection, expected):
    """The REPORT.md line naming the compared repetitions; consecutive ones as a range."""
    spans, start = [], selection[0]
    for previous, current in zip(selection, selection[1:] + [None]):
        if current != previous + 1:
            spans.append(str(start) if start == previous else f"{start}-{previous}")
            start = current
    share = f"all {expected}" if len(selection) == expected else f"{len(selection)} of the {expected}"
    return f"Repetitions compared: {', '.join(spans)} ({share} per Julia version in each directory)."


def load(root, expected):
    checked = validate(root, expected)
    if not checked["valid"]:
        raise ValueError(f"Invalid results {root}: {checked['issues']}")
    rows, configs, metadata = {}, set(), {}
    recorded = {}   # recorded[julia version][key] = value, or None when the record lacks the key
    for path in sorted(Path(root).glob("*-metadata.toml")):
        with path.open("rb") as stream:
            meta = tomllib.load(stream)
        metadata[path.name.removesuffix("-metadata.toml")] = meta
        configs.add(tuple(str(meta[key]) for key in MATCH_SETTINGS))
        settings = {key: recorded_setting(meta, key) for key in RECORDED_SETTINGS}
        version = str(meta["julia"])
        if recorded.setdefault(version, settings) != settings:   # validate() rejects this already
            raise ValueError(f"Recorded settings changed within Julia {version} in {root}")
    for path in sorted(Path(root).glob("*-metrics.tsv")):
        version = path.name.split("-rep")[0]
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                row["_metadata"] = metadata[path.name.removesuffix("-metrics.tsv")]
                rows[(version, row["id"], int(row["rep"]))] = row
    return rows, configs, recorded


def recorded_status(before, after):
    """Compare RECORDED_SETTINGS within each Julia version present on both sides.

    `before` and `after` map a version to {key: value or None}. Values of different versions are
    never compared. Two known values of a version that differ reject the comparison (every such
    mismatch is named). A side without the key (a record older than it) leaves the setting
    unknown: the comparison goes on, and its status names the unknown side instead of claiming a
    match. Returns {version: {key: {"status", "before", "after"}}}, with None for a missing value.
    """
    status, mismatches = {}, []
    for version in sorted(before.keys() & after.keys()):
        status[version] = {}
        for key in RECORDED_SETTINGS:
            old, new = before[version][key], after[version][key]
            if old is not None and new is not None and old != new:
                mismatches.append(f"Julia {version} {key}: before {old}; after {new}")
            code = ("match" if old is not None and new is not None else
                    "unknown_both" if old is None and new is None else
                    "unknown_before" if old is None else "unknown_after")
            status[version][key] = dict(status=code, before=old, after=new)
    if mismatches:
        raise ValueError("Benchmark configuration differs in recorded settings: " + "; ".join(mismatches))
    return status


def recorded_note(by_version):
    """One REPORT.md paragraph per Julia version."""
    notes = []
    for version, status in by_version.items():
        parts = []
        for key, entry in status.items():
            if entry["status"] == "match":
                parts.append(f"`{key}` {entry['before']} on both sides")
            elif entry["status"] == "unknown_both":
                parts.append(f"`{key}` unknown on both sides")
            else:
                side = entry["status"].removeprefix("unknown_")
                other = "after" if side == "before" else "before"
                parts.append(f"`{key}` unknown on the {side} side ({entry[other]} on the {other} side)")
        unverified = [key for key, entry in status.items() if entry["status"] != "match"]
        note = f"Recorded settings, Julia {version}: " + "; ".join(parts) + "."
        if unverified:
            note += (" A record without a key predates it; " + " and ".join(f"`{k}`" for k in unverified) +
                     " could not be verified to match.")
        notes += [note, ""]
    return notes


def compare(before_root, after_root, output, expected=1, timer_limit_multiple=10, primary=None, reps=None):
    """Pair the repetitions `reps` (default all of 1..expected) of two validated directories.

    Each directory is validated whole; only the chosen repetitions enter the comparison."""
    check_timer_limit_multiple(timer_limit_multiple)
    selection = select_reps(reps, expected)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh comparison directory to avoid stale reports")
    output.mkdir(parents=True, exist_ok=True)
    try:
        before, old_configs, old_recorded = load(before_root, expected)
        after, new_configs, new_recorded = load(after_root, expected)
        chosen = set(selection)
        before = {key: row for key, row in before.items() if key[2] in chosen}
        after = {key: row for key, row in after.items() if key[2] in chosen}
        runs = len(selection)
        coverage = dict(repetitions=selection, shared_measurements=len(before.keys() & after.keys()),
                        before_only=sorted(before.keys()-after.keys()),
                        after_only=sorted(after.keys()-before.keys()))
        (output/"coverage.json").write_text(json.dumps(coverage, indent=2)+"\n")
        if old_configs != new_configs:
            raise ValueError("Benchmark configuration differs; match hardware, suite, groups, instrumentation and measurement settings")
        settings = recorded_status(old_recorded, new_recorded)
        (output/"settings.json").write_text(json.dumps(dict(by_version=settings), indent=2)+"\n")
        if coverage["before_only"] or coverage["after_only"]:
            raise ValueError("Case coverage differs; inspect coverage.json and rerun the same cases")
        primary = set() if primary is None else set(primary)
        unknown = sorted(primary - {case for _, case, _ in before})
        if unknown:
            raise ValueError(f"Primary cases not in the comparison: {', '.join(unknown)}")
        groups = {}
        for key in sorted(before):
            groups.setdefault(key[:2], []).append((before[key], after[key]))
        results, outcomes = [], []
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
                limited_reps = [f"rep{int(pair[index]['rep']):02d}" for pair in pairs
                                if timer_limited(pair[index], pair[index]["_metadata"], timer_limit_multiple)]
                if limited_reps:
                    limited_by.append(side + ":" + ",".join(limited_reps))
            limited = bool(limited_by)
            assessment = ("single run; exploratory" if runs == 1 else
                          "inconclusive" if mu-half <= 0 <= mu+half else
                          "faster" if mu > 0 else "slower")
            if limited:
                assessment = "single run; exploratory; timer-limited" if runs == 1 else "timer-limited"
            results.append(dict(julia=version, id=case, paired_runs=runs,
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
            decision, slowdown, lower, upper, reason = outcome(ratios, limited)
            outcomes.append(dict(julia=version, id=case, role="primary" if case in primary else "diagnostic",
                paired_runs=runs, slowdown=slowdown, one_sided95_low=lower, one_sided95_high=upper,
                timing_outcome=decision, reason=reason, unexpected_allocation=unexpected))
        with (output/"comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(results)
        lines = ["# ProximalBase revision comparison", "", reps_note(selection, expected), "",
                 "Ratios above one favor the proposed revision. Intervals use paired log ratios across fresh process/input draws; they do not correct for correlated machine noise or multiple comparisons. Interleave revisions when confirming small differences.", "",
                 "Single-run ratios are exploratory and have no confidence interval. Allocation counts measure Julia heap bytes, not native allocations or peak memory.", "",
                 *recorded_note(settings),
                 f"A measurement is timer-limited when median_ns * evals < {timer_limit_multiple:g} * timer_floor_ns (strict). With an unusable calibration (NaN floor), only evals == 1 is limited. A case is limited if any compared repetition on either side is limited. Timer-limited cases are not to be used for ratio gates.", "",
                 "| Case | Julia | Before / after (µs) | Speedup [95% interval] | Mean bytes before / after | Maximum bytes before / after | Assessment |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for row in results:
            interval = ("unavailable" if runs == 1 else
                        f"{row['ci95_low']:.3f}, {row['ci95_high']:.3f}")
            note = row["timing_assessment"]
            if row["unexpected_allocation"]:
                note += "; unexpected allocation"
            lines.append(f"| `{row['id']}` | {row['julia']} | {row['before_mean_ns']/1000:.3f} / {row['after_mean_ns']/1000:.3f} | {row['paired_geometric_speedup']:.3f} [{interval}] | {row['before_mean_bytes']:.1f} / {row['after_mean_bytes']:.1f} | {row['before_max_bytes']:.0f} / {row['after_max_bytes']:.0f} | {note} |")
        with (output/"outcomes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(outcomes)
        lines += ["", "## Timing outcomes (three-outcome rule)", "",
                  f"These are **timing** outcomes, not an overall performance approval: allocations, correctness and complexity are judged separately. Slowdown is after/before (the inverse of the speedup) with one-sided 95% bounds. **pass**: upper bound below {SLOWDOWN_LIMIT}; **investigate**: slowdown at least {SLOWDOWN_LIMIT} and lower bound above 1 (confirm with independent measurements); **inconclusive**: neither, a single run, or timer-limited (the observed slowdown is shown without bounds). Primary cases are the ones named for this comparison; the others are diagnostic.", ""]
        for role in ("primary", "diagnostic"):
            rows = [o for o in outcomes if o["role"] == role]
            counts = {name: sum(o["timing_outcome"] == name for o in rows) for name in ("pass", "investigate", "inconclusive")}
            lines.append(f"{role.capitalize()} cases: {len(rows)} (" + ", ".join(f"{k} {v}" for k, v in counts.items()) + ").")
        named = [o for o in outcomes if o["role"] == "primary"]
        allocating = [o for o in named if o["unexpected_allocation"]]
        if allocating:
            lines += ["", f"**Unexpected allocation in {len(allocating)} primary case(s)**, whatever their timing outcome: " +
                      ", ".join(f"`{o['id']}` ({o['julia']})" for o in allocating) + "."]
        if named:
            lines += ["", "| Primary case | Julia | Slowdown [one-sided 95% bounds] | Timing outcome | Unexpected allocation |",
                      "|---|---|---:|---|---|"]
            for o in named:
                bounds = (f"{o['slowdown']:.3f} [unavailable]" if math.isnan(o["one_sided95_low"]) else
                          f"{o['slowdown']:.3f} [{o['one_sided95_low']:.3f}, {o['one_sided95_high']:.3f}]")
                lines.append(f"| `{o['id']}` | {o['julia']} | {bounds} | {o['timing_outcome']}{'; ' + o['reason'] if o['reason'] else ''} | "
                             f"{'**yes**' if o['unexpected_allocation'] else 'no'} |")
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
    parser.add_argument("--primary", help="file of primary case IDs, one per line (# comments)")
    parser.add_argument("--reps", nargs="+", action="extend", metavar="SPEC",
                        help="repetitions to compare, each k or a-b (default: all); all repetitions are still validated")
    args = parser.parse_args()
    try:
        compare(args.before, args.after, args.output, args.expected_runs, args.timer_limit_multiple,
                None if args.primary is None else read_primary(args.primary),
                reps=None if args.reps is None else parse_reps(args.reps))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")
