#!/usr/bin/env python3
"""Summarizer of the wide least-squares study (#1103); Python standard library only.

    python3 -I summarize.py <output dir>

It reads from the output directory (searched recursively) the files written by harness.jl
(timings-*.tsv, admm-untimed-*.tsv, reference-*.tsv, outputs-*/outputs-index.tsv) and by accuracy.jl
(accuracy-*.tsv), rebuilds the tables from the raw rows, applies the pre-registered criteria of the plan and
prints markdown tables. It writes tables.md and verdict.json into the output directory. A timing file without
its closing `# complete: yes` line is reported and left out. Process files of different BLAS thread counts belong
in different directories; the notes list header fields that differ between the processes read.

Criteria as implemented (the plan is the reference; see the README):
  * Per cell (case, measurement, param) and variant V in {C_B, C_D, C_W}: each process gives the per-round paired
    ratios V/S, S'/S and V'/V (rounds paired within the process), and their medians. A/A spread = the largest
    |median - 1| of S'/S and V'/V over the processes of the cell.
  * faster: every per-process median of V/S is below 0.95 and 1 - max(medians) exceeds the A/A spread;
    slower: every median is above 1.05 and min(medians) - 1 exceeds the A/A spread; otherwise inconclusive.
    The plan asks for six processes (three per role order); a cell with fewer is marked partial.
  * An output passes accuracy when eta <= 1e3 eps(T). A cell is left out of the speed criteria of a variant when
    the variant fails accuracy at its step (the step of a warm cell; step 1 for first use; any step for penalty).
  * Recommendations (A), (B), (C) per variant, over the cells and outputs of all Julia versions present (see recommend()).
"""
import collections
import glob
import json
import math
import os
import statistics
import sys

EPS = {"Float64": 2.0 ** -52, "Float32": 2.0 ** -23}
ETA_FACTOR = 1e3
S_MARGIN = 10.0
FASTER, SLOWER = 0.95, 1.05
VARIANTS = ["C_B", "C_D", "C_W"]
PRIMARY = ["S", "C_B", "C_D", "C_W"]
TIME_MEASUREMENTS = ["first_use", "warm", "penalty", "admm_total"]
PROCESSES_WANTED = 6
GAP_SLACK = 1e-8
SECTION_11_1 = "1500x5000"


def read_tsv(path):
    header, cols, rows = {}, None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                k, _, v = line[1:].partition(":")
                header[k.strip()] = v.strip()
            elif cols is None:
                cols = line.split("\t")
            else:
                f = line.split("\t")
                if len(f) != len(cols):
                    raise ValueError("%s: malformed row %r" % (path, line))
                rows.append(dict(zip(cols, f)))
    return header, rows


def num(s):
    try:
        return float(s)
    except ValueError:
        return math.nan


def finite(x):
    return x == x and abs(x) != math.inf


def median(xs):
    return statistics.median(xs) if xs else math.nan


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if isinstance(x, float) and abs(x) != math.inf and (abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0)):
        return "%.*e" % (digits - 1, x)
    if isinstance(x, float):
        return "%.*g" % (digits + 1, x)
    return str(x)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |")
    return "\n".join(out)


class Process:
    def __init__(self, key, header):
        self.key = key                      # (julia, order, proc)
        self.header = header
        self.groups = set()
        self.data = collections.defaultdict(lambda: collections.defaultdict(dict))   # cell -> role -> round -> value
        self.extra = collections.defaultdict(lambda: collections.defaultdict(dict))  # cell -> role -> round -> (gc, bytes, allocs)
        self.files = []


def load_timings(root, notes):
    procs = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "timings-*.tsv"), recursive=True)):
        header, rows = read_tsv(path)
        if header.get("complete") != "yes":
            notes.append("left out (no closing `# complete: yes`): %s" % os.path.basename(path))
            continue
        key = (header.get("julia", "?"), header.get("order", "?"), header.get("proc", "?"))
        p = procs.get(key)
        if p is None:
            p = procs[key] = Process(key, header)
        p.files.append(os.path.basename(path))
        p.groups.update(header.get("groups", "").split(","))
        for r in rows:
            cell = (r["case"], r["measurement"], r["param"])
            rnd = int(r["round"])
            if rnd in p.data[cell][r["role"]]:
                notes.append("duplicate row in process %s: %s %s round %d" % (key, cell, r["role"], rnd))
            p.data[cell][r["role"]][rnd] = num(r["value"])
            p.extra[cell][r["role"]][rnd] = (num(r["gc_s"]), num(r["bytes"]), num(r["allocs"]))
    return procs


def paired(p, cell, num_role, den_role):
    a, b = p.data[cell].get(num_role, {}), p.data[cell].get(den_role, {})
    return [a[r] / b[r] for r in sorted(set(a) & set(b)) if finite(a[r]) and finite(b[r]) and b[r] > 0]


def judge(procs, cell, variant):
    """The judgement of one cell for one variant over the given processes."""
    meds, aa, lo, hi, rounds = [], [], [], [], 0
    for p in procs:
        rs = paired(p, cell, variant, "S")
        if not rs:
            continue
        meds.append(median(rs))
        lo.append(min(rs))
        hi.append(max(rs))
        rounds = max(rounds, len(rs))
        for a, b in (("S'", "S"), (variant + "'", variant)):
            q = paired(p, cell, a, b)
            if q:
                aa.append(abs(median(q) - 1))
    if not meds:
        return None
    spread = max(aa) if aa else 0.0
    if all(m < FASTER for m in meds) and 1 - max(meds) > spread:
        verdict = "faster"
    elif all(m > SLOWER for m in meds) and min(meds) - 1 > spread:
        verdict = "slower"
    else:
        verdict = "inconclusive"
    return {"n": len(meds), "meds": meds, "center": median(meds), "lo": min(lo), "hi": max(hi), "lo_med": min(meds),
            "hi_med": max(meds), "spread": spread, "verdict": verdict, "partial": len(meds) < PROCESSES_WANTED,
            "rounds": rounds}


def case_info(case):
    """(eltype tag, design, response, size) of a case id such as f64/gauss/noisy/500x5000."""
    f = case.split("/")
    return f[0], f[1] if len(f) > 1 else "", f[2] if len(f) > 2 else "", f[-1]


def param_gamma(param):
    if param.startswith("g="):
        try:
            return float(param[2:].split("/")[0])
        except ValueError:
            return None
    return None


def load_accuracy(root, notes):
    """rows[(julia, case, role)] -> list of dicts."""
    acc = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "**", "accuracy-*.tsv"), recursive=True)):
        header, rows = read_tsv(path)
        if header.get("complete") != "yes":
            notes.append("left out (no closing `# complete: yes`): %s" % os.path.basename(path))
            continue
        if header.get("inputs_match_outputs") == "false":
            notes.append("accuracy file %s was evaluated against inputs that differ from those of the outputs" % os.path.basename(path))
        for r in rows:
            r["eta"], r["forward_error"] = num(r["eta"]), num(r["forward_error"])
            r["gamma"], r["gamma_index"], r["vindex"] = float(r["gamma"]), int(r["gamma_index"]), int(r["vindex"])
            r["passed"] = r["eta"] <= ETA_FACTOR * EPS[r["eltype"]] and r["status"] == "ok"
            r["ratio"] = r["eta"] / EPS[r["eltype"]]
            acc[(header.get("julia", "?"), r["case"], r["role"])].append(r)
    return acc


def load_reference(root):
    ref = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "reference-*.tsv"), recursive=True)):
        kv = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.rstrip("\n").partition("\t")
                kv[k] = v
        if "case" in kv:
            ref[kv["case"]] = kv
    return ref


def cell_accuracy_ok(acc, julia, case, variant, param, measurement):
    """False when the variant fails accuracy at the step(s) of the cell."""
    rows = acc.get((julia, case, variant), [])
    if not rows:
        return True
    if measurement == "warm":
        g = param_gamma(param)
        rows = [r for r in rows if g is not None and abs(r["gamma"] - g) <= 1e-9 * abs(g)]
    elif measurement == "first_use":
        rows = [r for r in rows if r["gamma"] == 1.0]
    elif measurement != "penalty":
        return True
    return all(r["passed"] for r in rows)


def main(root):
    notes = []
    procs = load_timings(root, notes)
    acc = load_accuracy(root, notes)
    ref = load_reference(root)
    untimed = []
    for path in sorted(glob.glob(os.path.join(root, "**", "admm-untimed-*.tsv"), recursive=True)):
        header, rows = read_tsv(path)
        for r in rows:
            r["julia"] = os.path.basename(path)[len("admm-untimed-"):-len(".tsv")]
        untimed += rows
    index_rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "outputs-index.tsv"), recursive=True)):
        header, rows = read_tsv(path)
        for r in rows:
            r["julia"] = header.get("julia", "?")
        index_rows += rows
    out = []

    def section(title, body=None):
        out.append("\n## " + title + "\n")
        if body:
            out.append(body)

    versions = sorted({k[0] for k in procs})
    by_version = {v: [p for k, p in sorted(procs.items()) if k[0] == v] for v in versions}

    # ---- processes
    rows = []
    for k, p in sorted(procs.items()):
        h = p.header
        rows.append([k[0], k[1], k[2], ",".join(sorted(p.groups)), h.get("rounds"), h.get("blas_threads"), h.get("cpu_model"),
                     h.get("loadavg_start"), h.get("wall_seconds", "-")])
    section("Processes", md_table(["julia", "order", "proc", "groups", "rounds", "blas threads", "cpu", "load at start", "wall s"], rows))
    for what in ("source_sha", "harness_sha", "inputs_index_sha256", "cpu_model", "blas_config", "openblas", "l3_cache",
                 "blas_threads", "check_bounds", "rounds"):
        vals = {p.header.get(what) for p in procs.values()}
        if len(vals) > 1:
            notes.append("processes differ in %s: %s" % (what, sorted(map(str, vals))))
    if procs:
        out.append("\nsource_sha: %s; harness_sha: %s; inputs_index_sha256: %s; BLAS: %s; OpenBLAS: %s; L3: %s\n" % tuple(
            next(iter({p.header.get(w, "?") for p in procs.values()})) for w in
            ("source_sha", "harness_sha", "inputs_index_sha256", "blas_config", "openblas", "l3_cache")))

    # ---- timing judgements
    verdicts = {}      # (julia, case, measurement, param, variant) -> judgement
    cells = collections.defaultdict(set)
    for v in versions:
        for p in by_version[v]:
            for cell in p.data:
                if cell[1] in TIME_MEASUREMENTS:
                    cells[v].add(cell)
    for v in versions:
        for cell in sorted(cells[v]):
            for var in VARIANTS:
                j = judge(by_version[v], cell, var)
                if j:
                    j["accuracy_ok"] = cell_accuracy_ok(acc, v, cell[0], var, cell[2], cell[1])
                    verdicts[(v, cell[0], cell[1], cell[2], var)] = j

    # A/A noise floor
    rows = []
    for v in versions:
        for meas in TIME_MEASUREMENTS:
            devs = collections.defaultdict(list)
            for cell in cells[v]:
                if cell[1] != meas:
                    continue
                for p in by_version[v]:
                    for a, b in (("S'", "S"), ("C_B'", "C_B"), ("C_D'", "C_D"), ("C_W'", "C_W")):
                        q = paired(p, cell, a, b)
                        if q:
                            devs[a + "/" + b].append(abs(median(q) - 1))
            for pair, d in sorted(devs.items()):
                rows.append([v, meas, pair, len(d), median(d), max(d)])
    section("A/A controls: deviation of the per-process median of the copy over its role from 1",
            md_table(["julia", "measurement", "pair", "process-cells", "median |dev|", "max |dev|"], rows))

    rows = []
    for v in versions:
        for meas in TIME_MEASUREMENTS:
            for var in VARIANTS:
                for tag in ("f64", "f32"):
                    js = [j for (vv, case, m, par, vr), j in verdicts.items()
                          if vv == v and m == meas and vr == var and case_info(case)[0] == tag]
                    if not js:
                        continue
                    cnt = collections.Counter(j["verdict"] for j in js if j["accuracy_ok"])
                    rows.append([v, meas, var, tag, len(js), cnt["faster"], cnt["slower"], cnt["inconclusive"],
                                 sum(not j["accuracy_ok"] for j in js), sum(j["partial"] for j in js),
                                 median([j["center"] for j in js])])
    section("Timing verdicts per measurement and variant (cells; the accuracy-failed cells are left out of the counts)",
            md_table(["julia", "measurement", "variant", "type", "cells", "faster", "slower", "inconclusive", "accuracy-failed",
                      "partial", "median ratio V/S"], rows))

    rows = []
    for (v, case, meas, par, var), j in sorted(verdicts.items()):
        rows.append([v, meas, case, par, var, j["n"], j["center"], j["lo_med"], j["hi_med"], j["lo"], j["hi"], j["spread"],
                     j["verdict"] + ("" if j["accuracy_ok"] else " (accuracy failed)") + (" (partial)" if j["partial"] else "")])
    section("Timing detail: ratios V/S per cell (median of the per-process medians; range of the per-process medians; "
            "minimum and maximum of the per-round ratios; A/A spread)",
            md_table(["julia", "measurement", "case", "param", "variant", "processes", "median", "min med", "max med",
                      "min round", "max round", "A/A spread", "verdict"], rows))

    # ---- first-use breakdown and absolute times
    rows = []
    for v in versions:
        agg = collections.defaultdict(list)
        for p in by_version[v]:
            for (case, meas, par), roles in p.data.items():
                if meas == "first_use_part":
                    for role, d in roles.items():
                        agg[(case, role, par)] += [x for x in d.values() if finite(x)]
        for (case, role, par), xs in sorted(agg.items()):
            rows.append([v, case, role, par, median(xs)])
    section("First-use breakdown: median seconds of each setup step (primary roles)",
            md_table(["julia", "case", "role", "step", "median s"], rows))
    rows = []
    for v in versions:
        agg = collections.defaultdict(list)
        for p in by_version[v]:
            for (case, meas, par), roles in p.data.items():
                if meas in ("first_use", "penalty") or (meas == "warm" and par.startswith("g=1/")):
                    for role, d in roles.items():
                        if role in PRIMARY:
                            agg[(case, meas, par, role)] += [x for x in d.values() if finite(x)]
        for (case, meas, par, role), xs in sorted(agg.items()):
            rows.append([v, case, meas, par, role, median(xs)])
    section("Median seconds per call of the primary roles (first use, warm at step 1, penalty patterns)",
            md_table(["julia", "case", "measurement", "param", "role", "median s"], rows))

    # ---- allocations
    rows, nonzero = [], []
    for v in versions:
        agg = collections.defaultdict(list)
        for p in by_version[v]:
            for (case, meas, par), roles in p.extra.items():
                if meas == "alloc":
                    for role, d in roles.items():
                        for (gc, b, a) in d.values():
                            agg[(role, par)].append((b, a, case))
        for (role, par), xs in sorted(agg.items()):
            bad = sorted({(c, b, a) for b, a, c in xs if b != 0 or a != 0})
            rows.append([v, role, par, len(xs), max(b for b, _, _ in xs), max(a for _, a, _ in xs), len(bad)])
            nonzero += [[v, role, par, c, b, a] for c, b, a in bad][:5]
    section("Warmed allocations per call (maximum over cases; the number of cases with a nonzero count)",
            md_table(["julia", "role", "call", "samples", "max bytes", "max allocs", "cases nonzero"], rows))
    if nonzero:
        out.append("\nFirst nonzero cases of each role and call:\n")
        out.append(md_table(["julia", "role", "call", "case", "bytes", "allocs"], nonzero))

    # ---- accuracy
    acc_versions = sorted({k[0] for k in acc})
    rows = []
    for v in acc_versions:
        for kind in ("main", "control"):
            for role in PRIMARY:
                rs = [r for (vv, case, rl), lst in acc.items() if vv == v and rl == role for r in lst if r["kind"] == kind]
                if rs:
                    rows.append([v, kind, role, len(rs), sum(not r["passed"] for r in rs), max(r["ratio"] for r in rs),
                                 median([r["ratio"] for r in rs])])
    section("Accuracy: outputs, failures (eta > 1e3 eps) and eta/eps per role",
            md_table(["julia", "kind", "role", "outputs", "failures", "max eta/eps", "median eta/eps"], rows))

    def ratio_to_s(rs_v, rs_s):
        s_by = {(r["gamma_index"], r["vindex"]): r["eta"] for r in rs_s}
        q = [r["eta"] / s_by[(r["gamma_index"], r["vindex"])] for r in rs_v
             if (r["gamma_index"], r["vindex"]) in s_by and s_by[(r["gamma_index"], r["vindex"])] > 0 and finite(r["eta"])]
        return q

    rows, margin_bad = [], []
    for v in acc_versions:
        for case in sorted({k[1] for k in acc if k[0] == v}):
            rs_s = acc.get((v, case, "S"), [])
            if not rs_s or rs_s[0]["kind"] != "main":
                continue
            smax = max(r["ratio"] for r in rs_s)
            if smax > ETA_FACTOR / S_MARGIN:
                margin_bad.append([v, case, smax])
            row = [v, case, smax]
            for var in VARIANTS:
                rv = acc.get((v, case, var), [])
                q = ratio_to_s(rv, rs_s)
                row += [max((r["ratio"] for r in rv), default=math.nan), sum(not r["passed"] for r in rv), max(q, default=math.nan)]
            rows.append(row)
    heads = ["julia", "case", "S max eta/eps"]
    for var in VARIANTS:
        heads += [var + " max eta/eps", var + " failures", var + " max eta/eta_S"]
    section("Accuracy per main case (all outputs: steps by inputs)", md_table(heads, rows))
    section("Margin of S under the threshold (the plan asks for a factor %g in every main case)" % S_MARGIN,
            md_table(["julia", "case", "S max eta/eps"], margin_bad) if margin_bad else "S has at least a factor %g margin in every main case evaluated." % S_MARGIN)

    controls = []
    reduces = collections.Counter()
    s_fails = collections.Counter()
    for v in acc_versions:
        for case in sorted({k[1] for k in acc if k[0] == v}):
            rs_s = acc.get((v, case, "S"), [])
            if not rs_s or rs_s[0]["kind"] != "control":
                continue
            r_s = rs_s[0]
            row = [v, case, r_s["ratio"], "fail" if not r_s["passed"] else "pass"]
            for var in VARIANTS:
                rv = acc.get((v, case, var), [])
                if rv:
                    row += [rv[0]["ratio"], "fail" if not rv[0]["passed"] else "pass"]
                    if not r_s["passed"]:
                        reduces[(v, var)] += rv[0]["eta"] < r_s["eta"]
                else:
                    row += [math.nan, "-"]
            if not r_s["passed"]:
                s_fails[v] += 1
            controls.append(row)
    heads = ["julia", "control", "S eta/eps", "S"]
    for var in VARIANTS:
        heads += [var + " eta/eps", var]
    section("Accuracy controls from #1078 (eta/eps and pass/fail)", md_table(heads, controls))
    rows = [[v, var, reduces[(v, var)], s_fails[v]] for v in acc_versions for var in VARIANTS]
    section("#1078: controls where S fails and the variant has a smaller eta than S (it does not by itself pass)",
            md_table(["julia", "variant", "controls with eta_V < eta_S", "controls where S fails"], rows))
    fwd = []
    for v in acc_versions:
        for role in PRIMARY:
            xs = [r["forward_error"] for (vv, case, rl), lst in acc.items() if vv == v and rl == role for r in lst
                  if finite(r["forward_error"])]
            if xs:
                fwd.append([v, role, len(xs), median(xs), max(xs)])
    section("Forward error against the 128-bit dual solution (n <= 500; diagnostic)",
            md_table(["julia", "role", "outputs", "median", "max"], fwd))
    copies_bad = [[r["julia"], r["case"], r["role"], r["status"], r["copy_identical"]] for r in index_rows
                  if r["status"] != "ok" or r["copy_identical"] != "true"]
    section("Outputs that are marked as an error or that differ from the A/A copy of their role",
            md_table(["julia", "case", "role", "status", "copy identical"], copies_bad[:40]) if copies_bad else "none")

    # ---- ADMM
    admm_rows, admm_ref_rows = [], []
    for c, kv in sorted(ref.items()):
        admm_ref_rows.append([c, kv.get("julia"), kv.get("iterations"), kv.get("converged"), kv.get("stationarity"), kv.get("objective")])
    section("Reference optimum (S-ADMM at rho = 1 to stationarity 1e-12 max(1, |X'y|inf))",
            md_table(["case", "julia", "iterations", "converged", "stationarity", "objective"], admm_ref_rows))
    kstar = {}
    for r in untimed:
        if r["record"] in ("k_boyd", "k_tight"):
            kstar[(r["julia"], r["case"], r["rho"], r["role"], r["record"])] = int(float(r["k"]))
    rows = []
    for (j, case, rho, role, rec), k in sorted(kstar.items()):
        rows.append([j, case, rho, role, rec, k if k else "cap"])
    section("ADMM iteration counts of the untimed pass (cap = not reached within the cap)",
            md_table(["julia", "case", "rho", "role", "level", "iterations"], rows))
    admm_cells = sorted({(v, c) for v in versions for c in cells[v] if c[1] == "admm_total"})
    rows = []
    for v, cell in admm_cells:
        case, _, par = cell
        for var in VARIANTS:
            j = verdicts.get((v, case, "admm_total", par, var))
            if not j:
                continue
            fac = judge(by_version[v], (case, "admm_factor", par), var)
            itr = judge(by_version[v], (case, "admm_iter", par), var)
            rows.append([v, case, par, var, j["center"], j["verdict"], fac["center"] if fac else None, itr["center"] if itr else None,
                         j["n"]])
    section("ADMM: total time ratio V/S per instance, rho and stopping level (and the ratios of the factorization and of the iterations)",
            md_table(["julia", "case", "param", "variant", "total V/S", "verdict", "factor V/S", "iterations V/S", "processes"], rows))
    gap_rows = []
    gaps = {}
    for v in versions:
        for p in by_version[v]:
            for (case, meas, par), roles in p.data.items():
                if meas in ("admm_gap", "admm_objective", "admm_stationarity"):
                    for role, d in roles.items():
                        gaps.setdefault((v, case, par, role, meas), []).extend(x for x in d.values() if finite(x))
    for (v, case, par, role, meas), xs in sorted(gaps.items()):
        if meas == "admm_gap" and role in PRIMARY:
            st = gaps.get((v, case, par, role, "admm_stationarity"), [])
            gap_rows.append([v, case, par, role, len(xs), max(xs), median(st) if st else None])
    section("ADMM: objective gap to the reference (maximum over runs) and median final stationarity",
            md_table(["julia", "case", "param", "role", "runs", "max gap", "median stationarity"], gap_rows))
    cap_rows = []
    for v in versions:
        for p in by_version[v]:
            for (case, meas, par), roles in p.data.items():
                if meas == "admm_capped":
                    for role, d in roles.items():
                        if any(x == 1 for x in d.values()):
                            cap_rows.append((v, case, par, role))
    section("ADMM cells where a role reached the iteration cap (timed runs skipped, inconclusive)",
            md_table(["julia", "case", "param", "role"], sorted(set(cap_rows))) if cap_rows else "none")

    # ---- criteria
    verdict = recommend(versions, by_version, verdicts, acc, acc_versions, ref, notes)
    section("Criteria and recommendation", format_verdict(verdict))
    section("Notes", "\n".join("- " + n for n in notes) if notes else "none")

    text = "# Summary of %s\n%s\n" % (os.path.basename(os.path.abspath(root)), "\n".join(out))
    print(text)
    with open(os.path.join(root, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(root, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)
    return 0


def instance_11_1(verdicts, versions):
    """The Boyd 11.1 instance: the f64 case with n = 1500, p = 5000 that has ADMM cells (else the largest instance)."""
    cases = sorted({k[1] for k in verdicts if k[2] == "admm_total"})
    pick = [c for c in cases if c.endswith(SECTION_11_1)]
    if pick:
        return pick[0], True
    if not cases:
        return None, False

    def size(c):
        n, _, p = case_info(c)[3].partition("x")
        return int(n) * int(p)
    return max(cases, key=size), False


def recommend(versions, by_version, verdicts, acc, acc_versions, ref, notes):
    """Applies the pre-registered criteria of the plan to each variant over the Julia versions present."""
    result = {"variants": {}, "provisional": [], "versions": versions}
    inst, exact = instance_11_1(verdicts, versions)
    result["section_11_1_instance"] = inst
    if inst is None:
        result["provisional"].append("no ADMM cells")
    elif not exact:
        result["provisional"].append("the instance %s stands in for the 1500x5000 instance of section 11.1" % inst)
    for var in VARIANTS:
        c = {}
        # (A1) passes eta in every case where S passes
        a1, why1 = True, []
        if not acc_versions:
            a1, why1 = None, ["no accuracy data"]
        for v in acc_versions:
            s_pass = {}
            for (vv, case, role), rows in acc.items():
                if vv == v and role == "S":
                    for r in rows:
                        s_pass[(case, r["gamma_index"], r["vindex"])] = r["passed"]
            for (vv, case, role), rows in acc.items():
                if vv == v and role == var:
                    for r in rows:
                        if s_pass.get((case, r["gamma_index"], r["vindex"])) and not r["passed"]:
                            a1 = False
                            why1.append("%s fails where S passes: %s step %d input %d (eta/eps %.3g) on %s" % (
                                var, case, r["gamma_index"], r["vindex"], r["ratio"], v))
        c["A1_passes_eta_where_S_passes"] = {"value": a1, "why": why1[:10], "count": len(why1)}
        # (A2) not slower on warmed prox in any primary case (float64, one BLAS thread)
        slower, cells_n = [], 0
        for (v, case, meas, par, vr), j in verdicts.items():
            if vr == var and meas == "warm" and case_info(case)[0] == "f64" and j["accuracy_ok"]:
                cells_n += 1
                if j["verdict"] == "slower":
                    slower.append("%s %s %s" % (v, case, par))
        c["A2_not_slower_on_warm_prox"] = {"value": (not slower) if cells_n else None, "cells": cells_n, "slower": slower[:10],
                                           "count": len(slower)}
        # ADMM at the 11.1 instance, rho = 1
        admm = {}
        for v in versions:
            for level in ("boyd", "tight"):
                par = "rho=1.0,level=%s" % level
                j = verdicts.get((v, inst, "admm_total", par, var)) if inst else None
                gap_ok = None
                if j:
                    xs = {}
                    for role in ("S", var):
                        g = []
                        for p in by_version[v]:
                            g += [x for x in p.data[(inst, "admm_gap", par)].get(role, {}).values() if finite(x)]
                        xs[role] = max(g) if g else None
                    pstar = num(ref.get(inst, {}).get("objective", "nan"))
                    if xs["S"] is not None and xs[var] is not None and finite(pstar):
                        gap_ok = xs[var] <= xs["S"] + GAP_SLACK * max(1.0, abs(pstar))
                admm[(v, level)] = {"verdict": j["verdict"] if j else None, "gap_ok": gap_ok,
                                    "partial": j["partial"] if j else None}
        faster_both = bool(admm) and all(a["verdict"] == "faster" for a in admm.values())
        gaps_ok = bool(admm) and all(a["gap_ok"] for a in admm.values())
        c["A3_admm_faster_at_both_levels"] = {"value": faster_both if admm and all(a["verdict"] for a in admm.values()) else None,
                                              "detail": {"%s/%s" % k: a for k, a in admm.items()}}
        c["A3_objective_gap_within_S_plus_slack"] = {"value": gaps_ok if admm and all(a["gap_ok"] is not None for a in admm.values()) else None}
        # (B1) passes eta at the 11.1 inputs at rho = 1, i.e. step 1 of the instance
        b1 = None
        if inst and acc_versions:
            b1 = True
            for v in acc_versions:
                rows = [r for r in acc.get((v, inst, var), []) if r["gamma"] == 1.0]
                if not rows:
                    b1 = None if b1 else b1
                elif not all(r["passed"] for r in rows):
                    b1 = False
        c["B1_passes_eta_at_the_instance_at_rho_1"] = {"value": b1}
        # (B3) not slower at the tight target (any instance and rho)
        tight_slower = [(v, case, par) for (v, case, meas, par, vr), j in verdicts.items()
                        if vr == var and meas == "admm_total" and par.endswith("level=tight") and j["verdict"] == "slower"]
        tight_cells = [1 for (v, case, meas, par, vr) in verdicts if vr == var and meas == "admm_total" and par.endswith("level=tight")]
        c["B3_not_slower_at_the_tight_target"] = {"value": (not tight_slower) if tight_cells else None, "slower": tight_slower[:10]}
        A = [c["A1_passes_eta_where_S_passes"]["value"], c["A2_not_slower_on_warm_prox"]["value"],
             c["A3_admm_faster_at_both_levels"]["value"], c["A3_objective_gap_within_S_plus_slack"]["value"]]
        B = [b1, c["A3_admm_faster_at_both_levels"]["value"], c["B3_not_slower_at_the_tight_target"]["value"]]
        unknown = [name for name, val in c.items() if val.get("value") is None]
        if all(x is True for x in A):
            rec = "A"
        elif any(x is False for x in A):
            rec = "B" if all(x is True for x in B) else ("C" if any(x is False for x in B) else "undetermined")
        else:
            rec = "undetermined"
        partial = [k for k, j in verdicts.items() if k[4] == var and j["partial"] and (k[2] == "admm_total" or k[2] == "warm")]
        result["variants"][var] = {"criteria": {k: v for k, v in c.items()}, "recommendation": rec, "unknown_criteria": unknown,
                                   "partial_cells": len(partial)}
    return result


def format_verdict(verdict):
    lines = []
    if verdict["provisional"]:
        lines.append("Provisional: " + "; ".join(verdict["provisional"]) + ".\n")
    lines.append("Instance of section 11.1: %s\n" % verdict.get("section_11_1_instance"))
    rows = []
    for var, d in verdict["variants"].items():
        for name, val in d["criteria"].items():
            rows.append([var, name, str(val.get("value")), ""])
        rows.append([var, "**recommendation**", "(%s)" % d["recommendation"],
                     "unknown: " + ", ".join(d["unknown_criteria"]) if d["unknown_criteria"] else ""])
        if d["partial_cells"]:
            rows.append([var, "cells with fewer than %d processes (warm, admm)" % PROCESSES_WANTED, str(d["partial_cells"]), ""])
    lines.append(md_table(["variant", "criterion", "value", "note"], rows))
    lines.append("\n(A) propose a PB fixed-penalty backend: A1, A2, A3 all true. (B) solver-side factorization only: not (A), "
                 "and B1, A3 (faster at both levels) and B3 true. (C) no change otherwise. The recommendation does not close "
                 "#1078; see the table of controls for the residuals.")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit("usage: python3 -I summarize.py <output dir>")
    sys.exit(main(sys.argv[1]))
