#!/usr/bin/env python3
"""Summarize benchmark/groupscad/run.jl results.

    python3 benchmark/groupscad/summarize.py DIR            # one process: X/R and the A/A ratio XC/X per case
    python3 benchmark/groupscad/summarize.py CAND LEGACY    # two processes: median X/R of CAND over median X/R of LEGACY

Per case and round the ratios come from ratios.csv; the summary is the median over rounds, with the minimum and maximum.
With two directories, a case is compared only when the outputs of X in the two processes agree (outputs/*.bin): nonfinite
entries and values must be equal (NaN with NaN, infinities of the same sign); finite vector entries must agree within
16 (m + 8) kappa eps(T) times the largest finite magnitude of the candidate vector and be zero in both or in neither;
finite values must agree within 16 (m + K) eps(T) relative; kappa is (a - 1)/(a - 1 - gamma) for the step below a - 1 and 1 otherwise. The outputs compared are those of the first input of each case. Cases whose ratio exceeds 1.05 are
marked with '!'. The memory and allocations of X are those of its last row in raw.csv (the last round).
"""
import array
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

EPS = {"Float32": 2.0 ** -23, "Float64": 2.0 ** -52}
KAPPA = {"below": 2.0, "at": 1.0, "above": 1.0}
TOTAL = 4096


def ratios(directory):
    out = defaultdict(lambda: ([], []))
    with open(os.path.join(directory, "ratios.csv")) as f:
        for row in csv.DictReader(f):
            out[row["case"]][0].append(float(row["x_to_reference"]))
            out[row["case"]][1].append(float(row["control_to_x"]))
    return out


def memory(directory):
    out = {}
    with open(os.path.join(directory, "raw.csv")) as f:
        for row in csv.DictReader(f):
            if row["role"] == "X":
                out[row["case"]] = (int(row["memory"]), int(row["allocs"]))
    return out


def outputs(directory, case):
    T = case.split("/")[1]
    code = "f" if T == "Float32" else "d"
    data = array.array(code)
    path = os.path.join(directory, "outputs", case.replace("/", "_") + ".bin")
    with open(path, "rb") as f:
        data.frombytes(f.read())
    return data


def agree(case, a, b):
    parts = case.split("/")
    T, m, step, op = parts[1], int(parts[2][1:]), parts[4], parts[6]
    if len(a) != len(b):
        return False
    eps, K = EPS[T], TOTAL // m
    scalars = 1 if op in ("value", "value_and_prox") else 0
    for i in range(scalars):
        x, y = a[i], b[i]
        if not (math.isfinite(x) and math.isfinite(y)):
            if not ((math.isnan(x) and math.isnan(y)) or x == y):
                return False
            continue
        if abs(x - y) > 16 * (m + K) * eps * abs(x):
            return False
    rest_a, rest_b = a[scalars:], b[scalars:]
    if not rest_a:
        return True
    for x, y in zip(rest_a, rest_b):
        if not (math.isfinite(x) and math.isfinite(y)):
            if not ((math.isnan(x) and math.isnan(y)) or x == y):
                return False
    finite = [abs(v) for v in rest_a if math.isfinite(v)]
    bound = 16 * (m + 8) * KAPPA[step] * eps * max(finite, default=0.0)
    return all(abs(x - y) <= bound and (x == 0) == (y == 0)
               for x, y in zip(rest_a, rest_b) if math.isfinite(x) and math.isfinite(y))


def summary(values):
    return statistics.median(values), min(values), max(values)


def main(args):
    if len(args) == 1:
        r, mem = ratios(args[0]), memory(args[0])
        print("case,x_to_reference_median,min,max,control_to_x_median,min,max,x_memory,x_allocs")
        for case in sorted(r):
            xr, aa = r[case]
            print(case, *("%.4f" % v for v in summary(xr) + summary(aa)), *mem.get(case, ("", "")), sep=",")
        return
    if len(args) != 2:
        sys.exit(__doc__)
    cand, legacy = ratios(args[0]), ratios(args[1])
    mem_c, mem_l = memory(args[0]), memory(args[1])
    print("case,agree,ratio_of_medians,min,max,candidate_aa_median,legacy_aa_median,candidate_memory,legacy_memory,flag")
    for case in sorted(set(cand) & set(legacy)):
        ok = agree(case, outputs(args[0], case), outputs(args[1], case))
        if not ok:
            print(case, "no", "", "", "", "", "", "", "", "", sep=",")
            continue
        # the processes are independent: the ratio of the medians, with the range of the candidate's rounds over the
        # legacy median
        base = statistics.median(legacy[case][0])
        med = statistics.median(cand[case][0]) / base
        lo, hi = min(cand[case][0]) / base, max(cand[case][0]) / base
        print(case, "yes", "%.4f" % med, "%.4f" % lo, "%.4f" % hi,
              "%.4f" % statistics.median(cand[case][1]), "%.4f" % statistics.median(legacy[case][1]),
              mem_c.get(case, ("",))[0], mem_l.get(case, ("",))[0], "!" if med > 1.05 else "", sep=",")


if __name__ == "__main__":
    main(sys.argv[1:])
