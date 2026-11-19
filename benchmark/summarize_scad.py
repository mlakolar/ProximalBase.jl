#!/usr/bin/env python3
"""Summarize paired SCAD benchmark runs. Usage: summarize_scad.py DIR [DIR ...]."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def spread(values):
    return f"{statistics.median(values):.3f} [{percentile(values, .1):.3f},{percentile(values, .9):.3f}]"


def analyze(directory):
    cases = defaultdict(lambda: defaultdict(dict))
    with (directory / "raw.csv").open() as stream:
        for row in csv.DictReader(stream):
            pairs = cases[(row["case"], row["comparison"])][row["round"]]
            if row["role"] in pairs:
                raise ValueError(f"Duplicate role for {row['case']} round {row['round']}")
            pairs[row["role"]] = row

    summaries = {}
    print(f"\n{directory}")
    print("case | reference | C/ref median [p10,p90] | A/A median [p10,p90] | "
          "C/ref minima | C ns | bytes ref/C/AA | allocs ref/C/AA | C first/ref first")
    for (case, comparison), rounds in sorted(cases.items()):
        cp, aa, cm, ns, cfirst, rfirst = [], [], [], [], [], []
        memory = defaultdict(set)
        allocs = defaultdict(set)
        for roles in rounds.values():
            if roles.keys() != {"reference", "candidate", "control"}:
                raise ValueError(f"Unpaired benchmark data: {case}")
            r, c, a = (roles[key] for key in ("reference", "candidate", "control"))
            ratio = float(c["median_ns"]) / float(r["median_ns"])
            cp.append(ratio)
            aa.append(float(a["median_ns"]) / float(c["median_ns"]))
            cm.append(float(c["min_ns"]) / float(r["min_ns"]))
            ns.append(float(c["median_ns"]))
            (cfirst if int(c["order"]) < int(r["order"]) else rfirst).append(ratio)
            for row in roles.values():
                memory[row["role"]].add(int(row["memory"]))
                allocs[row["role"]].add(int(row["allocs"]))
        count = len(cp)
        key = ("reference", "candidate", "control")
        maxima = lambda values: "/".join(str(max(values[label])) for label in key)
        order_effect = "/".join(f"{statistics.median(xs):.3f}" if xs else "–" for xs in (cfirst, rfirst))
        summaries[case] = (comparison, statistics.median(cp), statistics.median(aa),
                           statistics.median(cm), count)
        print(f"{case} | {comparison} | {spread(cp)} | {spread(aa)} | "
              f"{statistics.median(cm):.3f} | {statistics.median(ns):.1f} | "
              f"{maxima(memory)} | {maxima(allocs)} | {order_effect}")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    results = [analyze(d) for d in args.directories]
    all_cases = set.union(*(set(result) for result in results))
    print("\nRepeatable archive slowdown above 5% across the supplied runs "
          "(paired median and paired minimum ratios):")
    flagged = []
    for case in sorted(all_cases):
        if not all(case in result for result in results):
            continue
        records = [result[case] for result in results]
        if not all(k == "archive" and cr > 1.05 and cm > 1.05 and n >= 5
                   for k, cr, aa, cm, n in records):
            continue
        flagged.append(case)
        print(f"{case}: C/archive " + ", ".join(f"{r[1]:.3f}" for r in records)
              + " | A/A " + ", ".join(f"{r[2]:.3f}" for r in records))
    print(f"{len(flagged)} flagged out of {len(all_cases)} cases "
          "(including contextual cases in the total).")


if __name__ == "__main__":
    main()
