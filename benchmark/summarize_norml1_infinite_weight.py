#!/usr/bin/env python3
"""Summarize raw.csv of norml1_infinite_weight.jl: paired per-round ratios Y/X and XC/X.

usage: summarize_norml1_infinite_weight.py DIR [DIR ...]   (several directories pool their rounds)

Directories are pooled only when they have the same X and Y source trees (the X: and Y: lines of their
provenance.txt). A run with the roles reversed gives the inverse ratios and must be summarized on its own:
pooling it with a forward run would cancel a consistent slowdown, and the A/A column cannot show that.
"""
import csv, statistics, sys, collections

def roles(d):
    found = {}
    try:
        for line in open(f"{d}/provenance.txt"):
            for key in ("X", "Y"):
                if line.startswith(key + ": "):
                    found[key] = line[len(key) + 2:].strip()
    except OSError:
        pass
    return (found.get("X"), found.get("Y"))

if len(sys.argv) < 2:
    sys.exit(__doc__)
assignments = {d: roles(d) for d in sys.argv[1:]}
if len(set(assignments.values())) > 1 or any(None in a for a in assignments.values()):
    sys.exit("refusing to pool runs without a common X/Y assignment (read from provenance.txt); "
             "summarize reversed-role or unlabeled runs separately:\n" +
             "\n".join(f"  {d}: X={x} Y={y}" for d, (x, y) in assignments.items()))

rows = collections.defaultdict(lambda: collections.defaultdict(dict))  # case -> (dir,round) -> role -> (med, min, allocs)
for d in sys.argv[1:]:
    with open(f"{d}/raw.csv") as f:
        for r in csv.DictReader(f):
            rows[r["case"]][(d, r["round"])][r["role"]] = (float(r["median_ns"]), float(r["min_ns"]), int(r["allocs"]))

def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

print("case | Y/X median [p10,p90] | XC/X median [p10,p90] | min-ratio Y/X | allocs X/Y | flag")
worst = 0.0
for case in sorted(rows):
    yx, cx, ymin = [], [], []
    allocs = set()
    for key, roles in rows[case].items():
        if {"X", "Y", "XC"} <= roles.keys():
            yx.append(roles["Y"][0] / roles["X"][0]); cx.append(roles["XC"][0] / roles["X"][0])
            ymin.append(roles["Y"][1] / roles["X"][1]); allocs.add((roles["X"][2], roles["Y"][2]))
    my, mc = statistics.median(yx), statistics.median(cx)
    flag = "SLOW" if my > 1.05 else ("fast" if my < 0.95 else "")
    worst = max(worst, my)
    print(f"{case} | {my:.3f} [{pct(yx,.1):.3f},{pct(yx,.9):.3f}] | {mc:.3f} [{pct(cx,.1):.3f},{pct(cx,.9):.3f}] | {statistics.median(ymin):.3f} | {sorted(allocs)} | {flag}")
print(f"worst median Y/X over {len(rows)} cases: {worst:.3f}")
