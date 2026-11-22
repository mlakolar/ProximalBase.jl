#!/usr/bin/env python3
"""Summarize benchmark/groupscad/run.jl results.

    python3 benchmark/groupscad/summarize.py DIR            # one process: X/R and the A/A ratio XC/X per case
    python3 benchmark/groupscad/summarize.py CAND LEGACY    # two processes: median X/R of CAND over median X/R of LEGACY

Per case and round the ratios come from ratios.csv; the summary is the median over rounds, with the minimum and maximum.
With two directories, a case appears only when both directories have it. When both have outputs_all/ and declare the same number
of inputs in verification.csv, the case is compared on every input, and a file with a different number of records, the wrong
record width or an inconsistent first record gives agree = incompatible. Older directories (without outputs_all/ or without the
declared count) are compared on the first input only, and the column "compared" says so. Outputs agree when nonfinite
entries and values are equal (NaN with NaN, infinities of the same sign); finite vector entries agree within
16 (m + 8) kappa eps(T) times the largest finite magnitude of the candidate vector and are zero in both or in neither;
finite values agree within 16 (m + K) eps(T) relative; kappa is (a - 1)/(a - 1 - gamma) for the step below a - 1 and 1 otherwise.
In the two-directory output, ratio_of_medians is the median over the recorded rounds of the candidate's X/R divided by the
median X/R of the legacy directory, and min and max are the candidate's smallest and largest recorded round X/R divided by that
legacy median; rounds_candidate and rounds_legacy are the numbers of recorded rounds (a round dropped by run.jl for a cell is
not counted). Cases whose ratio exceeds 1.05 are marked with '!'. The memory and allocations of X are those of its last row in
raw.csv (the last recorded round).
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


def outputs_all(directory, case):
    path = os.path.join(directory, "outputs_all", case.replace("/", "_") + ".bin")
    if not os.path.exists(path):
        return None
    code = "f" if case.split("/")[1] == "Float32" else "d"
    data = array.array(code)
    with open(path, "rb") as f:
        data.frombytes(f.read())
    return data


def declared_inputs(directory):
    """The number of inputs of each case that run.jl declares in verification.csv (None for a directory without the column)."""
    path = os.path.join(directory, "verification.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or "inputs" not in rows[0]:
        return None
    return {row["case"]: int(row["inputs"]) for row in rows}


def record_width(case):
    """The number of values that one input contributes to the saved outputs: the value, the point, or both."""
    op = case.split("/")[6]
    return {"value": 1, "value_and_prox": 1 + TOTAL}.get(op, TOTAL)


def agree_all(case, cand, legacy):
    """(status, compared, reason) of the saved outputs of a case in two directories.

    Both directories with outputs_all/ and the same declared number of inputs n >= 1: every input is compared, after the
    files are checked to have exactly n records of the expected width, a first-input file of one record, and a first record of
    outputs_all/ equal to it; a violation gives the status "incompatible". Otherwise (an older directory without outputs_all/
    or without the declared count) only the first input is compared, and "compared" says so.
    """
    w = record_width(case)
    first_a, first_b = outputs(cand, case), outputs(legacy, case)
    if len(first_a) != w or len(first_b) != w:
        return "incompatible", "first", "the first-input file has %d and %d values, expected %d" % (len(first_a), len(first_b), w)
    all_a, all_b = outputs_all(cand, case), outputs_all(legacy, case)
    n_a, n_b = declared_inputs(cand), declared_inputs(legacy)
    if all_a is None or all_b is None or n_a is None or n_b is None:
        return ("yes" if agree(case, first_a, first_b) else "no"), "first (an older directory)", ""
    na, nb = n_a.get(case), n_b.get(case)
    if na is None or nb is None or na != nb or na < 1:
        return "incompatible", "all", "declared inputs %s and %s" % (na, nb)
    for name, data in (("candidate", all_a), ("legacy", all_b)):
        if len(data) != na * w:
            return "incompatible", "all", "%s outputs_all has %d values, expected %d inputs of %d" % (name, len(data), na, w)
    if all_a[:w].tobytes() != first_a.tobytes() or all_b[:w].tobytes() != first_b.tobytes():
        return "incompatible", "all", "the first record of outputs_all differs from the first-input file"
    ok = all(agree(case, all_a[i * w:(i + 1) * w], all_b[i * w:(i + 1) * w]) for i in range(na))
    return ("yes" if ok else "no"), "all (%d inputs)" % na, ""


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
    print("case,agree,ratio_of_medians,min,max,candidate_aa_median,legacy_aa_median,candidate_memory,legacy_memory,flag,"
          "compared,rounds_candidate,rounds_legacy,reason")
    for case in sorted(set(cand) & set(legacy)):
        status, compared, reason = agree_all(case, args[0], args[1])
        rounds = (len(cand[case][0]), len(legacy[case][0]))
        if status != "yes":
            print(case, status, "", "", "", "", "", "", "", "", compared, *rounds, reason, sep=",")
            continue
        # the processes are independent: the ratio of the medians, with the range of the candidate's rounds over the
        # legacy median
        base = statistics.median(legacy[case][0])
        med = statistics.median(cand[case][0]) / base
        lo, hi = min(cand[case][0]) / base, max(cand[case][0]) / base
        print(case, "yes", "%.4f" % med, "%.4f" % lo, "%.4f" % hi,
              "%.4f" % statistics.median(cand[case][1]), "%.4f" % statistics.median(legacy[case][1]),
              mem_c.get(case, ("",))[0], mem_l.get(case, ("",))[0], "!" if med > 1.05 else "", compared, *rounds, "", sep=",")


if __name__ == "__main__":
    main(sys.argv[1:])
