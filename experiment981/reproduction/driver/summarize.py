#!/usr/bin/env python3
"""Summarize warm comparisons and reference error from the experiment TSV files."""

import csv
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median, quantiles
import sys


def read_tsv(path):
    with path.open() as inp:
        return list(csv.DictReader(inp, delimiter="\t"))


def paired(timing):
    profiles = defaultdict(lambda: defaultdict(dict))
    types = defaultdict(dict)
    cold = defaultdict(dict)
    for row in read_tsv(timing):
        key = row["julia"], row["type"], row["case"], int(row["n"])
        side = int(row["side"])
        if row["phase"] == "warm":
            profiles[key][side][int(row["rep"])] = float(row["ns_per_call"])
            types[key][side] = row["label"], int(row["bytes_per_call"])
        elif row["phase"] == "first":
            cold[key][side] = float(row["ns_per_call"]) * 1e-6
    print(timing.name)
    print("  Cold first-call ms by type and side", dict(cold))
    summary = defaultdict(list)
    all_cases = []
    for key, sides in profiles.items():
        assert set(sides) == {1, 2}
        left, right = sides[1], sides[2]
        assert len(left) >= 5 and set(left) == set(right)
        ratios = [right[i] / left[i] for i in sorted(left)]
        row = median(ratios), key, median(left.values()), median(right.values()), types[key]
        summary[key[2]].append(row)
        all_cases.append(row)
    ordinary = [row for row in all_cases if not row[1][2].startswith("adversarial")]
    print(f"  {len(all_cases)} paired cells, ordinary median right/left {median(r[0] for r in ordinary):.4g}")
    if "A-A" in timing.name:
        noise = [max(row[0], 1/row[0])-1 for row in all_cases]
        print(f"  A/A symmetric absolute ratio deviation median {median(noise):.4g}, p90 {quantiles(noise,n=10)[-1]:.4g}, max {max(noise):.4g}")
    for case, rows in sorted(summary.items()):
        print(f"  {case:18} median right/left {median(r[0] for r in rows):.4g}, range {min(r[0] for r in rows):.4g}–{max(r[0] for r in rows):.4g}")
        for r in rows:
            if r[1][-1] in (32, 4096, 1_000_000):
                print(f"    {r[1][1]:8} n={r[1][-1]:7d}: {r[2]:.5g} → {r[3]:.5g} ns, ratio {r[0]:.4g}, bytes {r[4][1][1]} → {r[4][2][1]}")
    print("  Extremes")
    for row in (min(ordinary), max(ordinary)):
        print("   ", row)


def accuracy(result):
    rows = read_tsv(result)
    counts = Counter(row["status"] for row in rows)
    print(result.name, dict(counts))
    for number_type in ("Float32", "Float64"):
        data = []
        for r in rows:
            if r["type"] != number_type or r["abs_error"] in ("", "Inf", "NaN"):
                continue
            data.append(r)
        if not data:
            continue
        maximum = max(data, key=lambda r: Decimal(r["abs_error"]))
        normalized = max(data, key=lambda r: Decimal(r["normalized_error"]))
        ordinary = [r for r in data if "subnormal" not in r["case"]]
        ordinary_max = max(ordinary, key=lambda r: Decimal(r["normalized_error"]))
        print(f'  {number_type} max abs {Decimal(maximum["abs_error"]):.5g} ({maximum["case"]}), max normalized {Decimal(normalized["normalized_error"]):.5g} ({normalized["case"]}), excluding subnormals {Decimal(ordinary_max["normalized_error"]):.5g} ({ordinary_max["case"]}, {ordinary_max["path"]})')
        for r in data:
            if r["status"] in ("material", "fail"):
                print(f'    {r["case"]}, {r["path"]}, n={r["n"]}, got {r["observed"]}, reference {Decimal(r["reference"]):.10g}')


for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if "timing-" in path.name or "pilot-" in path.name:
        paired(path)
    else:
        accuracy(path)
