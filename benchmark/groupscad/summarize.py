#!/usr/bin/env python3
"""Summarize benchmark/groupscad/run.jl results.

    python3 benchmark/groupscad/summarize.py DIR            # one process: X/R and the A/A ratio XC/X per case
    python3 benchmark/groupscad/summarize.py CAND LEGACY    # two processes: median X/R of CAND over median X/R of LEGACY

Per case and round the ratios come from ratios.csv; the summary is the median over rounds, with the minimum and maximum.
With two directories, a case appears only when both directories have it. When both have outputs_all/ and declare the same number
of inputs in verification.csv, the case is compared on every input; for such a current-format directory the first-input file and
the outputs_all file of the case must be present with exactly the byte length of one and of n records (the first record of
outputs_all equal to the first-input file), and any violation, also a directory with only one of the two markers, gives
agree = incompatible. Only a recognizable older directory (neither declared inputs nor outputs_all/) is compared on the first
input, and the column "compared" says so. Outputs agree when nonfinite
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


def _record_file(path, case, expected):
    """(values, reason): the values of a binary record file, or None and the reason when the file is missing or its byte length
    is not exactly that of `expected` values."""
    code = "f" if case.split("/")[1] == "Float32" else "d"
    data = array.array(code)
    if not os.path.isfile(path):
        return None, "%s is missing" % path
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != expected * data.itemsize:
        return None, "%s has %d bytes, expected %d" % (path, len(raw), expected * data.itemsize)
    data.frombytes(raw)
    return data, ""


def declared_inputs(directory):
    """The number of inputs of each case that run.jl declares in verification.csv (None for a directory without the file or
    without the column)."""
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


def _load(directory, case):
    """The saved outputs of a case in one directory: (kind, first, all, n, reason). The kind is "current" for a directory in the
    format of run.jl (verification.csv with the column "inputs", and outputs_all/), "legacy" for one with neither of the two, and
    "incompatible" when a current directory lacks a required record or one has the wrong byte length, or when it has only one of
    the two markers. A current directory needs, for the case, the declared count n >= 1, a first-input file of exactly one record
    and an outputs_all file of exactly n records, the first of which equals the first-input file."""
    w = record_width(case)
    first, reason = _record_file(os.path.join(directory, "outputs", case.replace("/", "_") + ".bin"), case, w)
    if first is None:
        return "incompatible", None, None, None, reason
    declared = declared_inputs(directory)
    has_all = os.path.isdir(os.path.join(directory, "outputs_all"))
    if declared is None and not has_all:
        return "legacy", first, None, None, ""
    if declared is None or not has_all:
        return "incompatible", first, None, None, "only one of the declared inputs and outputs_all/ is present in %s" % directory
    n = declared.get(case)
    if n is None or n < 1:
        return "incompatible", first, None, n, "the case has no declared count of inputs (%s) in %s" % (n, directory)
    all_inputs, reason = _record_file(os.path.join(directory, "outputs_all", case.replace("/", "_") + ".bin"), case, n * w)
    if all_inputs is None:
        return "incompatible", first, None, n, reason
    if all_inputs[:w].tobytes() != first.tobytes():
        return "incompatible", first, None, n, "the first record of outputs_all differs from the first-input file in %s" % directory
    return "current", first, all_inputs, n, ""


def agree_all(case, cand, legacy):
    """(status, compared, reason) of the saved outputs of a case in two directories.

    Both directories in the current format with the same declared number of inputs n >= 1: every input is compared. A current
    directory is checked for the presence and the exact byte length of the first-input file and of the outputs_all file of the
    case (see _load); a missing or truncated record, or a first record that differs from the first-input file, gives the status
    "incompatible", never an exception and never a first-input comparison. A recognizable older directory (neither declared
    inputs nor outputs_all/) is compared on the first input only against any directory whose records are valid, and "compared"
    says so.
    """
    w = record_width(case)
    ka, first_a, all_a, na, ra = _load(cand, case)
    kb, first_b, all_b, nb, rb = _load(legacy, case)
    for name, kind, reason in (("candidate", ka, ra), ("legacy", kb, rb)):
        if kind == "incompatible":
            return "incompatible", "none", "%s: %s" % (name, reason)
    if ka == "legacy" or kb == "legacy":
        return ("yes" if agree(case, first_a, first_b) else "no"), "first (an older directory)", ""
    if na != nb:
        return "incompatible", "all", "declared inputs %s and %s" % (na, nb)
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
