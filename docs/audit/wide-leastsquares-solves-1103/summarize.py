#!/usr/bin/env python3
"""Summarizer of the wide least-squares study (#1103); Python standard library only.

    python3 -I summarize.py <output dir> [<inputs dir or inputs-index.tsv>]

It reads from the output directory (searched recursively) the files written by harness.jl (timings-*.tsv,
admm-untimed-*.tsv, reference-*.tsv, outputs-<julia>/outputs-index.tsv) and by accuracy.jl (accuracy-*.tsv), together
with the input manifest inputs-index.tsv (found in the output directory, or given as the second argument), rebuilds the
tables from the raw rows, applies the pre-registered criteria of the plan and prints markdown tables. It writes
tables.md, per-process-ratios.tsv and verdict.json into the output directory.

The campaign design is fixed in this file and in the manifest: the Julia versions 1.10.13 and 1.13.4, the three
processes of each of the two role orders (S_first, C_first) with one BLAS thread (processes with other thread counts
are diagnostics and take no part in the criteria), 21 rounds (5 ADMM repeats), the A/A controls S' and C', and every
timed case, step, input kind and ADMM instance of the manifest. A cell, accuracy key or artifact that is absent,
incomplete, truncated (no closing `# complete: yes` line), or made from another manifest or identity is `unknown`: it
never satisfies a criterion and it makes the verdict provisional.

Criteria as implemented (the plan is the reference; see the README):
  * Per cell (case, measurement, parameter) and variant V in {C_B, C_D, C_W}: each of the six processes gives the paired
    ratios V/S, S'/S and V'/V of every round, and their medians. A/A spread = the largest |median - 1| of S'/S and
    V'/V over the six processes. faster: all six medians of V/S are below 0.95 and 1 - max(medians) exceeds the
    spread; slower: all six are above 1.05 and min(medians) - 1 exceeds the spread; otherwise inconclusive. A cell that
    lacks any process, round or A/A pair is unknown.
  * An output passes accuracy when eta <= 1e3 eps(T), decided by accuracy.jl in BigFloat. The accuracy evidence of a
    cell is the steps and inputs that its measurement uses (first use: step 1 and the random inputs; warm: its step
    and kind of inputs; penalty: the steps of its schedule and the random inputs; ADMM: the step 1/rho and the
    ADMM iterates). A cell whose variant (or S) fails there is excluded from the speed criteria; missing evidence
    makes it unknown. Exclusion is reported apart from the timing label.
  * Recommendations (A), (B), (C) per variant (see recommend()).
"""
import collections
import glob
import hashlib
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
EXPECTED_VERSIONS = ("1.10.13", "1.13.4")
REFERENCE_JULIA = "1.13.4"
EXPECTED_ORDERS = ("S_first", "C_first")
EXPECTED_PROCS = ("1", "2", "3")
EXPECTED_ROUNDS = 21
EXPECTED_REPEATS = 5
PRIMARY_BLAS = "1"
ADMM_RHOS = (0.1, 1.0, 10.0)
ADMM_LEVELS = ("boyd", "tight")
GAP_SLACK = 1e-8
INSTANCE_11_1 = (1500, 5000)
ALT2 = (1.0, 1e3)
NEWSTEP = {"Float64": (1e-3, 1e3, 1.0, 1e6), "Float32": (1e-3, 1e3, 1.0)}
TIMING_COLUMNS = ["case", "measurement", "param", "round", "position", "role", "value", "gc_s", "bytes", "allocs"]
IDENT_FIELDS = ("source_sha", "harness_sha", "inputs_index_sha256")
SAME_FIELDS = ("julia", "order", "blas_threads", "cpu_vendor", "cpu_model", "blas_config", "openblas")
MAX_LISTED = 40


class Skipped(Exception):
    """A file that cannot be used: interrupted, malformed or of another format."""


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def finite(x):
    return x == x and abs(x) != math.inf


def median(xs):
    xs = list(xs)
    return statistics.median(xs) if xs else math.nan


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float) and abs(x) != math.inf and (abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0)):
        return "%.*e" % (digits - 1, x)
    if isinstance(x, float):
        return "%.*g" % (digits + 1, x)
    return str(x)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c if isinstance(c, str) else fmt(c) for c in r) + " |")
    return "\n".join(out)


def listing(rows, headers, what):
    """A table of at most MAX_LISTED rows, with a line that says how many rows were left out."""
    rows = list(rows)
    out = md_table(headers, rows[:MAX_LISTED])
    if len(rows) > MAX_LISTED:
        out += "\n\n(%d further %s not listed)" % (len(rows) - MAX_LISTED, what)
    return out


def gk(g, k):
    """The key of a step and an input index: the step rounded to 12 digits (the steps are written with repr)."""
    return (float("%.12g" % g), k)


# ---------------------------------------------------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------------------------------------------------

def read_tsv(path):
    """(header, columns, rows) of a tab-separated file of the study; Skipped for an interrupted or malformed file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = [l for l in fh.read().split("\n") if l != ""]
    except OSError as e:
        raise Skipped("unreadable: %s" % e)
    if not lines or lines[-1] != "# complete: yes":
        raise Skipped("no closing `# complete: yes` line (interrupted write)")
    header, cols, rows = {}, None, []
    for line in lines:
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            header[k.strip()] = v.strip()
        elif cols is None:
            cols = line.split("\t")
        else:
            f = line.split("\t")
            if len(f) != len(cols):
                raise Skipped("malformed row with %d fields instead of %d" % (len(f), len(cols)))
            rows.append(f)
    if cols is None:
        raise Skipped("no column line")
    return header, cols, rows


def read_kv(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = [l for l in fh.read().split("\n") if l != ""]
    except OSError as e:
        raise Skipped("unreadable: %s" % e)
    if not lines or lines[-1] != "complete\tyes":
        raise Skipped("no closing `complete yes` line (interrupted write)")
    kv = {}
    for l in lines[:-1]:
        k, sep, v = l.partition("\t")
        if not sep:
            raise Skipped("malformed line")
        kv[k] = v
    return kv


class Notes:
    """The reasons that make the verdict provisional, and the other notes."""

    def __init__(self):
        self.provisional = []
        self.notes = []

    def prov(self, text):
        if text not in self.provisional:
            self.provisional.append(text)

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def skip(self, name, why):
        self.prov("file %s left out: %s" % (name, why))


# ---------------------------------------------------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------------------------------------------------

def parse_floats(s):
    return [float(x) for x in s.split(",")] if s else []


class Manifest:
    def __init__(self, path):
        header, cols, rows = read_tsv(path)
        if header.get("format") != "pb1103-inputs-2":
            raise Skipped("format %s, expected pb1103-inputs-2" % header.get("format", "none"))
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()
        self.path, self.header = path, header
        self.cases = collections.OrderedDict()
        for f in rows:
            r = dict(zip(cols, f))
            c = self.cases.get(r["case"])
            if c is None:
                c = self.cases[r["case"]] = {
                    "id": r["case"], "kind": r["kind"], "eltype": r["eltype"], "design": r["design"],
                    "response": r["response"], "n": int(r["n"]), "p": int(r["p"]), "timed": r["timed"] == "true",
                    "lasso": r["lasso"] == "true", "gammas": parse_floats(r["gammas"]),
                    "acc_gammas": parse_floats(r["acc_gammas"]), "lambda_max": num(r["lambda_max"]),
                    "sigma_max": num(r["sigma_max"]), "arrays": {}}
            shape = [int(x) for x in r["shape"].split("x")]
            c["arrays"][r["array"]] = {"file": r["file"], "shape": shape, "sha256": r["sha256"]}
        for c in self.cases.values():
            if not all(k in c["arrays"] for k in ("X", "y", "V")):
                raise Skipped("case %s lacks one of X, y, V" % c["id"])
            c["nv"] = c["arrays"]["V"]["shape"][1]


def find_manifest(root, arg):
    if arg:
        p = os.path.join(arg, "inputs-index.tsv") if os.path.isdir(arg) else arg
        return [p]
    return sorted(glob.glob(os.path.join(root, "**", "inputs-index.tsv"), recursive=True))


# ---------------------------------------------------------------------------------------------------------------------
# Cells and the timing rows
# ---------------------------------------------------------------------------------------------------------------------

def canon(meas, param):
    """The canonical parameter of a cell: the steps and penalties as numbers, so that spellings do not matter."""
    try:
        if meas == "first_use":
            return ("g", float(param[2:]))
        if meas == "warm":
            g, kind = param[2:].split("/")
            return ("g", float(g), kind)
        if meas.startswith("admm_") and param.startswith("rho="):
            r, l = param.split(",")
            return ("rho", float(r[4:]), l[6:])
    except (ValueError, IndexError):
        pass
    return param


def expected_cells(man):
    """The timed cells of the campaign: (case, measurement, canonical parameter) for every timed case."""
    cells = []
    for c in man.cases.values():
        if c["kind"] != "main" or not c["timed"]:
            continue
        cells.append((c["id"], "first_use", ("g", 1.0)))
        for g in c["gammas"]:
            for kind in ("rand", "admm"):
                cells.append((c["id"], "warm", ("g", g, kind)))
        for s in ("alt2", "newstep"):
            cells.append((c["id"], "penalty", s))
        if c["lasso"]:
            for rho in ADMM_RHOS:
                for lvl in ADMM_LEVELS:
                    cells.append((c["id"], "admm_total", ("rho", rho, lvl)))
    return cells


def needed_keys(case, meas, cn):
    """The (step, input) keys whose accuracy the timing of a cell depends on."""
    if meas == "first_use":
        return [gk(1.0, k) for k in range(1, 5)]
    if meas == "warm":
        ks = range(1, 5) if cn[2] == "rand" else range(5, 9)
        return [gk(cn[1], k) for k in ks]
    if meas == "penalty":
        steps = ALT2 if cn == "alt2" else NEWSTEP[case["eltype"]]
        return [gk(g, k) for g in steps for k in range(1, 5)]
    if meas == "admm_total":
        return [gk(1.0 / cn[1], k) for k in range(5, 9)]
    return []


class Proc:
    """One timing process (Julia version, role order, process id, BLAS threads), possibly split over several files."""

    def __init__(self, key):
        self.key = key
        self.files = []
        self.headers = []
        self.groups = set()
        self.data = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.extra = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.plans, self.loads, self.workingset, self.errors = {}, collections.defaultdict(float), {}, []
        self.duplicates = 0


def load_timings(root, notes):
    procs = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "timings-*.tsv"), recursive=True)):
        name = os.path.basename(path)
        try:
            header, cols, rows = read_tsv(path)
            if header.get("format") != "pb1103-timings-2" or cols != TIMING_COLUMNS:
                raise Skipped("not a pb1103-timings-2 file")
        except Skipped as e:
            notes.skip(name, e)
            continue
        key = (header.get("julia", "?"), header.get("order", "?"), header.get("proc", "?"), header.get("blas_threads", "?"))
        p = procs.get(key)
        if p is None:
            p = procs[key] = Proc(key)
        for f in SAME_FIELDS + IDENT_FIELDS:
            if p.headers and p.headers[0].get(f) != header.get(f):
                notes.prov("process %s/%s/%s (BLAS threads %s): the file %s differs from the first file in %s (%s, %s)" % (
                    key[0], key[1], key[2], key[3], name, f, p.headers[0].get(f), header.get(f)))
        p.files.append(name)
        p.headers.append(header)
        p.groups.update(header.get("groups", "").split(","))
        for f in rows:
            case, meas, param, rnd, _pos, role, val, gc, nbytes, nalloc = f
            rnd = int(rnd)
            if meas == "error":
                p.errors.append((case, param, role, rnd))
                continue
            if meas == "batch_plan":
                p.plans[(case, param)] = num(val)
                continue
            if meas == "loadavg_1m":
                p.loads[(case, param)] = max(p.loads[(case, param)], num(val))
                continue
            if meas == "workingset":
                p.workingset[(case, role)] = max(p.workingset.get((case, role), 0.0), num(val))
                continue
            cell = (case, meas, canon(meas, param))
            slot = p.data[cell][role]
            if rnd in slot:
                p.duplicates += 1
            slot[rnd] = num(val)
            if meas in ("first_use", "alloc"):
                p.extra[cell][role][rnd] = (num(gc), num(nbytes), num(nalloc))
    for p in procs.values():
        if p.duplicates:
            notes.prov("process %s: %d duplicate rows" % (p.key, p.duplicates))
        if p.errors:
            notes.prov("process %s: %d recorded errors (first: %s)" % (p.key, len(p.errors), p.errors[0]))
    return procs


def ratios(d, num_role, den_role, rounds):
    a, b = d.get(num_role, {}), d.get(den_role, {})
    out = {}
    for r in rounds:
        x, y = a.get(r), b.get(r)
        if x is not None and y is not None and finite(x) and finite(y) and y > 0:
            out[r] = x / y
    return out


def expected_rounds(meas):
    return list(range(1, (EXPECTED_REPEATS if meas.startswith("admm_") else EXPECTED_ROUNDS) + 1))


def judge(primary, version, cell, variant):
    """The timing judgement of one cell for one variant over the six expected processes of a Julia version.

    Returns a dict with the label (faster, slower, inconclusive, or unknown when any process, round or A/A pair is
    missing), the numbers that are available, and the per-process medians, minima and maxima."""
    meas = cell[1]
    rounds = expected_rounds(meas)
    gaps, per, meds, aa, allr = [], [], [], [], []
    for o in EXPECTED_ORDERS:
        for pid in EXPECTED_PROCS:
            proc = primary.get((version, o, pid))
            if proc is None:
                gaps.append("process %s/%s absent" % (o, pid))
                continue
            d = proc.data.get(cell, {})
            rr = {"V": ratios(d, variant, "S", rounds), "S": ratios(d, "S'", "S", rounds),
                  "C": ratios(d, variant + "'", variant, rounds)}
            for name, lab in (("V", "%s/S" % variant), ("S", "S'/S"), ("C", "%s'/%s" % (variant, variant))):
                if len(rr[name]) != len(rounds):
                    gaps.append("%s/%s %s: %d of %d rounds" % (o, pid, lab, len(rr[name]), len(rounds)))
            if rr["V"]:
                vals = list(rr["V"].values())
                per.append((o, pid, median(vals), min(vals), max(vals)))
                meds.append(per[-1][2])
                allr.extend(vals)
            for name in ("S", "C"):
                if rr[name]:
                    aa.append(abs(median(rr[name].values()) - 1))
    spread = max(aa) if aa else math.nan
    res = {"complete": not gaps, "gaps": gaps, "per": per, "meds": meds, "spread": spread, "n": len(meds),
           "center": median(meds), "lo_med": min(meds) if meds else math.nan, "hi_med": max(meds) if meds else math.nan,
           "lo": min(allr) if allr else math.nan, "hi": max(allr) if allr else math.nan}
    if gaps:
        res["label"] = "unknown"
    elif all(m < FASTER for m in meds) and 1 - max(meds) > spread:
        res["label"] = "faster"
    elif all(m > SLOWER for m in meds) and min(meds) - 1 > spread:
        res["label"] = "slower"
    else:
        res["label"] = "inconclusive"
    return res


# ---------------------------------------------------------------------------------------------------------------------
# The campaign: the artifacts that are accepted, the cells and the criteria
# ---------------------------------------------------------------------------------------------------------------------

def ident(h):
    return tuple(h.get(f, "?") for f in IDENT_FIELDS)


def tri_all(vals):
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


class Campaign:
    def __init__(self, root, manifest_arg, notes):
        self.root, self.notes = root, notes
        self.man = None
        self.cells = []
        self.instance = None
        self._cell, self._acc = {}, {}
        self.load_manifest(manifest_arg)
        self.load_processes()
        self.load_accuracy()
        self.load_aux()
        self.pick_instance()
        self.check_design()

    # ---- the manifest
    def load_manifest(self, arg):
        paths = find_manifest(self.root, arg)
        if not paths:
            self.notes.prov("no inputs-index.tsv: the expected design is unknown")
            return
        mans = []
        for p in paths:
            try:
                mans.append(Manifest(p))
            except Skipped as e:
                self.notes.skip(p, e)
        if not mans:
            return
        if len({m.sha256 for m in mans}) > 1:
            self.notes.prov("%d different inputs-index.tsv files; the first is used" % len({m.sha256 for m in mans}))
        self.man = mans[0]
        self.cells = expected_cells(self.man)

    # ---- the timing processes
    def load_processes(self):
        procs = load_timings(self.root, self.notes)
        self.procs = {}
        for key, p in procs.items():
            digests = {h.get("inputs_index_sha256") for h in p.headers}
            if self.man is not None and digests != {self.man.sha256}:
                self.notes.prov("process %s left out: made from another input manifest (%s)" % (key, sorted(digests)))
                continue
            self.procs[key] = p
        self.primary = {(k[0], k[1], k[2]): p for k, p in self.procs.items() if k[3] == PRIMARY_BLAS}
        self.diag = {k: p for k, p in self.procs.items() if k[3] != PRIMARY_BLAS}
        idents = collections.Counter(ident(h) for p in self.primary.values() for h in p.headers)
        if len(idents) > 1:
            self.notes.prov("the primary timing files differ in (source, harness, inputs) identity: %s" % sorted(idents))
        self.ident = idents.most_common(1)[0][0] if idents else None
        if self.ident and "unset" in self.ident:
            self.notes.prov("the timing files carry an unset source or harness identity")

    def same_ident(self, h, what):
        if self.ident is None:
            return True
        if ident(h)[:2] != self.ident[:2]:
            self.notes.prov("%s left out: source/harness identity %s differs from that of the timing files %s" % (
                what, ident(h)[:2], self.ident[:2]))
            return False
        return True

    # ---- accuracy
    def load_accuracy(self):
        self.acc = collections.defaultdict(dict)
        self.acc_versions, self.acc_forward_max = set(), {}
        for path in sorted(glob.glob(os.path.join(self.root, "**", "accuracy-*.tsv"), recursive=True)):
            name = os.path.basename(path)
            try:
                header, cols, rows = read_tsv(path)
                if header.get("format") != "pb1103-accuracy-2":
                    raise Skipped("not a pb1103-accuracy-2 file")
            except Skipped as e:
                self.notes.skip(name, e)
                continue
            if self.man is not None and header.get("inputs_index_sha256") != self.man.sha256:
                self.notes.prov("%s left out: evaluated against another input manifest" % name)
                continue
            if not self.same_ident(header, name):
                continue
            v = header.get("julia", "?")
            if v in self.acc_versions:
                self.notes.prov("two accuracy files for Julia %s" % v)
            self.acc_versions.add(v)
            self.acc_forward_max[v] = header.get("forward_max_n", "?")
            for f in rows:
                r = dict(zip(cols, f))
                self.acc[(v, r["case"], r["role"])][gk(float(r["gamma"]), int(r["vindex"]))] = {
                    "eta": num(r["eta"]), "fe": num(r["forward_error"]), "pass": r["pass"] == "true",
                    "status": r["status"], "copy": r["copy_identical"], "eltype": r["eltype"], "kind": r["kind"],
                    "gamma": float(r["gamma"]), "vindex": int(r["vindex"])}

    # ---- outputs index, untimed files, references
    def load_aux(self):
        self.outputs, self.untimed, self.refs = {}, {}, []
        for path in sorted(glob.glob(os.path.join(self.root, "**", "outputs-index.tsv"), recursive=True)):
            try:
                header, cols, rows = read_tsv(path)
            except Skipped as e:
                self.notes.skip(os.path.relpath(path, self.root), e)
                continue
            if self.man is not None and header.get("inputs_index_sha256") != self.man.sha256:
                self.notes.prov("%s left out: made from another input manifest" % os.path.relpath(path, self.root))
                continue
            if not self.same_ident(header, os.path.relpath(path, self.root)):
                continue
            self.outputs[header.get("julia", "?")] = [dict(zip(cols, f)) for f in rows]
        for path in sorted(glob.glob(os.path.join(self.root, "**", "admm-untimed-*.tsv"), recursive=True)):
            name = os.path.basename(path)
            try:
                header, cols, rows = read_tsv(path)
            except Skipped as e:
                self.notes.skip(name, e)
                continue
            if self.man is not None and header.get("inputs_index_sha256") != self.man.sha256:
                self.notes.prov("%s left out: made from another input manifest" % name)
                continue
            if not self.same_ident(header, name):
                continue
            self.untimed.setdefault((header.get("julia", "?"), header.get("case", "?")), []).extend(dict(zip(cols, f)) for f in rows)
        for path in sorted(glob.glob(os.path.join(self.root, "**", "reference-*.tsv"), recursive=True)):
            try:
                kv = read_kv(path)
            except Skipped as e:
                self.notes.skip(os.path.basename(path), e)
                continue
            kv["_file"] = os.path.basename(path)
            self.refs.append(kv)

    # ---- the instance of section 11.1 and the design
    def pick_instance(self):
        if self.man is None:
            return
        lasso = [c for c in self.man.cases.values() if c["lasso"] and c["kind"] == "main"]
        exact = [c for c in lasso if (c["n"], c["p"]) == INSTANCE_11_1]
        if exact:
            self.instance = exact[0]["id"]
        elif lasso:
            self.instance = max(lasso, key=lambda c: c["n"] * c["p"])["id"]
            self.notes.prov("the instance %s stands in for the 1500x5000 instance of section 11.1" % self.instance)
        else:
            self.notes.prov("the manifest holds no lasso instance")
        if not any(c["kind"] == "control" for c in self.man.cases.values()):
            self.notes.prov("the manifest holds no #1078 controls")

    def check_design(self):
        for v in EXPECTED_VERSIONS:
            for o in EXPECTED_ORDERS:
                for pid in EXPECTED_PROCS:
                    if (v, o, pid) not in self.primary:
                        self.notes.prov("Julia %s: the process %s/%s with one BLAS thread is absent" % (v, o, pid))
            if v not in self.acc_versions:
                self.notes.prov("Julia %s: no accepted accuracy file" % v)
        for k in self.primary:
            if k[0] not in EXPECTED_VERSIONS or k[1] not in EXPECTED_ORDERS or k[2] not in EXPECTED_PROCS:
                self.notes.prov("unexpected process %s with one BLAS thread takes no part in the criteria" % (k,))
        if self.diag:
            self.notes.note("diagnostic processes (BLAS threads other than %s): %s" % (PRIMARY_BLAS, sorted(self.diag)))

    # ---- accuracy evidence
    def acc_status(self, version, case_id, role, need):
        rows = self.acc.get((version, case_id, role))
        if not rows:
            return "unknown"
        missing = sum(1 for k in need if k not in rows)
        failed = sum(1 for k in need if k in rows and not rows[k]["pass"])
        return "fail" if failed else ("unknown" if missing else "pass")

    # ---- the cells
    def cell(self, version, cell, variant):
        key = (version, cell, variant)
        if key in self._cell:
            return self._cell[key]
        case = self.man.cases[cell[0]]
        meas = cell[1]
        j = judge(self.primary, version, cell, variant)
        if meas == "admm_total":
            j = self.admm_adjust(version, cell, variant, j)
        need = needed_keys(case, meas, cell[2])
        aV = self.acc_status(version, case["id"], variant, need)
        aS = self.acc_status(version, case["id"], "S", need)
        if aV == "fail":
            elig = "excluded (%s fails accuracy)" % variant
        elif aS == "fail":
            elig = "excluded (S fails accuracy)"
        elif aV == "pass" and aS == "pass":
            elig = "eligible"
        else:
            elig = "unknown (accuracy evidence incomplete)"
        label = j["label"] if elig == "eligible" else ("excluded" if elig.startswith("excluded") else "unknown")
        rec = dict(j, eligibility=elig, timing_label=j["label"], label=label, acc_V=aV, acc_S=aS)
        self._cell[key] = rec
        return rec

    def admm_adjust(self, version, cell, variant, j):
        """Reaching the target and the end point of the timed runs of an ADMM cell."""
        case, _, cn = cell
        capped, invalid = set(), 0
        for o in EXPECTED_ORDERS:
            for pid in EXPECTED_PROCS:
                proc = self.primary.get((version, o, pid))
                if proc is None:
                    continue
                reach = proc.data.get((case, "admm_reached", cn), {})
                for role in ("S", variant):
                    if reach.get(role, {}).get(0) == 0:
                        capped.add(role)
                valid = proc.data.get((case, "admm_valid", cn), {})
                for role in ("S", "S'", variant, variant + "'"):
                    invalid += sum(1 for val in valid.get(role, {}).values() if val != 1)
        j = dict(j)
        if capped:
            j.update(label="inconclusive", capped=sorted(capped), complete=True)
            j["gaps"] = ["the target was not reached within the cap by %s (a lower bound)" % ", ".join(sorted(capped))]
        elif invalid and j["label"] != "unknown":
            j.update(label="unknown", complete=False)
            j["gaps"] = j["gaps"] + ["%d timed runs ended outside the target" % invalid]
        return j

    # ---- reference optimum
    def reference(self):
        """(objective, reason): the objective of the pinned reference of the instance when it is usable."""
        if self.instance is None or self.man is None:
            return None, "no instance"
        c = self.man.cases[self.instance]
        cands = [kv for kv in self.refs if kv.get("case") == self.instance]
        if not cands:
            return None, "no reference file for %s" % self.instance
        why = []
        for kv in cands:
            bad = []
            if kv.get("x_sha256") != c["arrays"]["X"]["sha256"] or kv.get("y_sha256") != c["arrays"]["y"]["sha256"]:
                bad.append("other arrays")
            if kv.get("inputs_index_sha256") != self.man.sha256:
                bad.append("other manifest")
            if kv.get("converged") != "true":
                bad.append("not converged")
            if kv.get("julia") != REFERENCE_JULIA:
                bad.append("Julia %s, not the pinned %s" % (kv.get("julia"), REFERENCE_JULIA))
            if self.ident and (kv.get("source_sha"), kv.get("harness_sha")) != self.ident[:2]:
                bad.append("other source or harness identity")
            if not bad:
                return num(kv.get("objective")), "ok"
            why.append("%s: %s" % (kv["_file"], ", ".join(bad)))
        return None, "; ".join(why)

    def objective_runs(self, version, cell, role):
        """All timed objectives of a role in an ADMM cell over the expected processes and repeats, and whether all exist."""
        vals, complete = [], True
        for o in EXPECTED_ORDERS:
            for pid in EXPECTED_PROCS:
                proc = self.primary.get((version, o, pid))
                if proc is None:
                    complete = False
                    continue
                d = proc.data.get((cell[0], "admm_objective", cell[2]), {}).get(role, {})
                got = [d[r] for r in expected_rounds("admm_total") if r in d and finite(d[r])]
                complete = complete and len(got) == EXPECTED_REPEATS
                vals.extend(got)
        return vals, complete

    # ---- the criteria
    def not_slower(self, variant, cells):
        recs = [self.cell(v, c, variant) for v in EXPECTED_VERSIONS for c in cells]
        if not recs:
            return None, "no cells"
        n_slower = sum(1 for r in recs if r["label"] == "slower")
        n_unknown = sum(1 for r in recs if r["label"] == "unknown")
        n_excl = sum(1 for r in recs if r["label"] == "excluded")
        info = "%d cells: %d slower, %d unknown, %d excluded for accuracy" % (len(recs), n_slower, n_unknown, n_excl)
        if n_slower:
            return False, info
        return (None if n_unknown else True), info

    def all_faster(self, variant, cells):
        recs = [self.cell(v, c, variant) for v in EXPECTED_VERSIONS for c in cells]
        if not recs:
            return None, "no cells"
        n_def = sum(1 for r in recs if r["label"] in ("slower", "inconclusive", "excluded"))
        n_unknown = sum(1 for r in recs if r["label"] == "unknown")
        info = "%d cells: %d faster, %d slower or inconclusive or excluded, %d unknown" % (
            len(recs), sum(1 for r in recs if r["label"] == "faster"), n_def, n_unknown)
        if n_def:
            return False, info
        return (None if n_unknown else True), info

    def accuracy_everywhere(self, variant):
        """A1: V passes at every (case, step, input) of the manifest where S passes, in both Julia versions."""
        missing, fails = 0, []
        for v in EXPECTED_VERSIONS:
            for c in self.man.cases.values():
                rs, rv = self.acc.get((v, c["id"], "S"), {}), self.acc.get((v, c["id"], variant), {})
                for g in c["acc_gammas"]:
                    for k in range(1, c["nv"] + 1):
                        s, r = rs.get(gk(g, k)), rv.get(gk(g, k))
                        if s is None or r is None:
                            missing += 1
                        elif s["pass"] and not r["pass"]:
                            fails.append((v, c["id"], g, k))
        info = "%d keys without evidence, %d where %s fails and S passes" % (missing, len(fails), variant)
        if fails:
            return False, info + " (first: %s)" % (fails[0],)
        return (None if missing else True), info

    def instance_accuracy(self, variant):
        """B1: V passes at the step 1 and all 8 inputs of the instance of section 11.1 in both Julia versions."""
        if self.instance is None:
            return None, "no instance"
        need = [gk(1.0, k) for k in range(1, 9)]
        st = [self.acc_status(v, self.instance, variant, need) for v in EXPECTED_VERSIONS]
        info = "status per Julia version: %s" % ", ".join(st)
        if "fail" in st:
            return False, info
        return (True if all(s == "pass" for s in st) else None), info

    def gap_criterion(self, variant):
        """A3, second part: the objective gap of V is within that of S plus 1e-8 max(1, |p*|) at rho = 1, both levels."""
        pstar, why = self.reference()
        if pstar is None or not finite(pstar):
            return None, "reference unusable: %s" % why
        ok, missing, detail = True, False, []
        for v in EXPECTED_VERSIONS:
            for lvl in ADMM_LEVELS:
                cell = (self.instance, "admm_total", ("rho", 1.0, lvl))
                s, cs = self.objective_runs(v, cell, "S")
                x, cx = self.objective_runs(v, cell, variant)
                if not (cs and cx):
                    missing = True
                    continue
                gS, gV = max(s) - pstar, max(x) - pstar
                detail.append("%s/%s: gap S %.3g, %s %.3g" % (v, lvl, gS, variant, gV))
                if gV > gS + GAP_SLACK * max(1.0, abs(pstar)):
                    ok = False
        info = "; ".join(detail)
        if not ok:
            return False, info
        return (None if missing else True), info

    def completeness(self):
        """The reasons from the cells and accuracy keys of the fixed design that lack evidence."""
        if self.man is None:
            return
        bad, first, unknown_acc = collections.Counter(), {}, 0
        for v in EXPECTED_VERSIONS:
            for c in self.cells:
                for var in VARIANTS:
                    r = self.cell(v, c, var)
                    if not r["complete"]:
                        bad[c[1]] += 1
                        first.setdefault(c[1], "%s %s %s %s: %s" % (v, c[0], param_text(c[2]), var, r["gaps"][0] if r["gaps"] else "?"))
                    if r["eligibility"].startswith("unknown"):
                        unknown_acc += 1
        for meas, n in sorted(bad.items()):
            self.notes.prov("%d (version, cell, variant) judgements of %s lack evidence (first: %s)" % (n, meas, first[meas]))
        if unknown_acc:
            self.notes.prov("%d (version, cell, variant) judgements lack accuracy evidence at the steps and inputs they use" % unknown_acc)
        missing = 0
        for v in EXPECTED_VERSIONS:
            for c in self.man.cases.values():
                for role in PRIMARY:
                    rows = self.acc.get((v, c["id"], role), {})
                    missing += sum(1 for g in c["acc_gammas"] for k in range(1, c["nv"] + 1) if gk(g, k) not in rows)
        if missing:
            self.notes.prov("%d accuracy keys (version, case, role, step, input) have no evidence" % missing)

    def recommend(self):
        out = {"variants": {}, "instance": self.instance, "provisional": list(self.notes.provisional)}
        if self.man is None:
            for var in VARIANTS:
                out["variants"][var] = {"criteria": {}, "recommendation": "undetermined", "unknown": ["no manifest"]}
            return out
        warm = [c for c in self.cells if c[1] == "warm" and self.man.cases[c[0]]["eltype"] == "Float64"]
        inst = self.instance
        i11 = [(inst, "admm_total", ("rho", 1.0, l)) for l in ADMM_LEVELS] if inst else []
        tight = [c for c in self.cells if c[1] == "admm_total" and c[2][2] == "tight"]
        for var in VARIANTS:
            crit = {}
            crit["A1 passes eta wherever S passes"] = self.accuracy_everywhere(var)
            crit["A2 not slower on the warmed prox (Float64 cells)"] = self.not_slower(var, warm)
            crit["A3 ADMM total faster at rho = 1, both levels"] = self.all_faster(var, i11)
            crit["A3 objective gap within S plus slack"] = self.gap_criterion(var)
            crit["B1 passes eta at the instance, rho = 1"] = self.instance_accuracy(var)
            crit["B3 not slower at the tight target (all instances and rho)"] = self.not_slower(var, tight)
            vals = {k: v[0] for k, v in crit.items()}
            A = [vals[k] for k in crit if k.startswith("A")]
            B = [vals["B1 passes eta at the instance, rho = 1"], vals["A3 ADMM total faster at rho = 1, both levels"],
                 vals["B3 not slower at the tight target (all instances and rho)"]]
            if all(a is True for a in A):
                rec = "A"
            elif any(a is False for a in A):
                rec = "B" if all(b is True for b in B) else ("C" if any(b is False for b in B) else "undetermined")
            else:
                rec = "undetermined"
            out["variants"][var] = {"criteria": {k: {"value": v[0], "info": v[1]} for k, v in crit.items()},
                                    "recommendation": rec, "unknown": [k for k, v in crit.items() if v[0] is None]}
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------------

MEASUREMENTS = ("first_use", "warm", "penalty", "admm_total")


def l3_bytes(text):
    try:
        t = text.strip().upper()
        mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}.get(t[-1], 1)
        return float(t[:-1] if t[-1] in "KMG" else t) * mult
    except (ValueError, IndexError, AttributeError):
        return math.nan


def param_text(cn):
    if isinstance(cn, tuple):
        if cn[0] == "g" and len(cn) == 2:
            return "g=%g" % cn[1]
        if cn[0] == "g":
            return "g=%g/%s" % (cn[1], cn[2])
        if cn[0] == "rho":
            return "rho=%g,level=%s" % (cn[1], cn[2])
    return str(cn)


def table_processes(camp):
    rows = []
    for k, p in sorted(camp.procs.items()):
        h = p.headers[0]
        rows.append([k[0], k[1], k[2], k[3], ",".join(sorted(p.groups)), h.get("rounds", "?"), h.get("cpu_vendor", "?"),
                     h.get("cpu_model", "?"), h.get("loadavg_start", "?"), ", ".join(sorted({x.get("wall_seconds", "-") for x in p.headers}))])
    return md_table(["julia", "order", "proc", "BLAS threads", "groups", "rounds", "CPU vendor", "CPU model", "load at start", "wall s"], rows)


def table_aa(camp):
    rows = []
    for v in EXPECTED_VERSIONS:
        for meas in MEASUREMENTS:
            devs = collections.defaultdict(list)
            for cell in camp.cells:
                if cell[1] != meas:
                    continue
                rounds = expected_rounds(meas)
                for o in EXPECTED_ORDERS:
                    for pid in EXPECTED_PROCS:
                        proc = camp.primary.get((v, o, pid))
                        if proc is None:
                            continue
                        d = proc.data.get(cell, {})
                        for a, b in (("S'", "S"), ("C_B'", "C_B"), ("C_D'", "C_D"), ("C_W'", "C_W")):
                            r = ratios(d, a, b, rounds)
                            if r:
                                devs["%s/%s" % (a, b)].append(abs(median(r.values()) - 1))
            for pair, d in sorted(devs.items()):
                rows.append([v, meas, pair, len(d), median(d), max(d)])
    return md_table(["julia", "measurement", "pair", "process-cells", "median |dev|", "max |dev|"], rows)


def table_verdicts(camp):
    rows = []
    for v in EXPECTED_VERSIONS:
        for meas in MEASUREMENTS:
            for var in VARIANTS:
                for tp in ("Float64", "Float32"):
                    cs = [c for c in camp.cells if c[1] == meas and camp.man.cases[c[0]]["eltype"] == tp]
                    if not cs:
                        continue
                    recs = [camp.cell(v, c, var) for c in cs]
                    cnt = collections.Counter(r["label"] for r in recs)
                    rows.append([v, meas, var, tp, len(recs), sum(1 for r in recs if r["complete"]), cnt["faster"], cnt["slower"],
                                 cnt["inconclusive"], cnt["unknown"], cnt["excluded"], median([r["center"] for r in recs])])
    return md_table(["julia", "measurement", "variant", "type", "cells", "complete", "faster", "slower", "inconclusive",
                     "unknown", "excluded (accuracy)", "median ratio V/S"], rows)


def table_detail(camp, root):
    rows, per_rows = [], []
    for v in EXPECTED_VERSIONS:
        for cell in camp.cells:
            for var in VARIANTS:
                r = camp.cell(v, cell, var)
                rows.append([v, cell[1], cell[0], param_text(cell[2]), var, r["n"], r["center"], r["lo_med"], r["hi_med"],
                             r["lo"], r["hi"], r["spread"], r["label"] + ("" if r["complete"] else " (incomplete: %d gaps)" % len(r["gaps"])),
                             r["eligibility"]])
                for (o, pid, med, lo, hi) in r["per"]:
                    per_rows.append([v, cell[0], cell[1], param_text(cell[2]), var, o, pid, repr(med), repr(lo), repr(hi)])
    with open(os.path.join(root, "per-process-ratios.tsv"), "w", encoding="utf-8") as fh:
        fh.write("julia\tcase\tmeasurement\tparam\tvariant\torder\tproc\tmedian\tmin\tmax\n")
        for r in per_rows:
            fh.write("\t".join(r) + "\n")
    return md_table(["julia", "measurement", "case", "param", "variant", "processes", "median of medians", "min process median",
                     "max process median", "min round", "max round", "A/A spread", "label", "accuracy eligibility"], rows)


def pooled(camp, version, cell, role, field=0):
    """The values of a role in a cell over the primary processes of a version and all rounds (field: value or extras)."""
    vals = []
    for (v, o, pid), p in camp.primary.items():
        if v != version:
            continue
        if field == 0:
            vals.extend(x for x in p.data.get(cell, {}).get(role, {}).values() if finite(x))
        else:
            vals.extend(x[field - 1] for x in p.extra.get(cell, {}).get(role, {}).values() if finite(x[field - 1]))
    return vals


def table_first_use(camp):
    rows, parts, gc = [], [], []
    for v in EXPECTED_VERSIONS:
        for c in camp.cells:
            if c[1] != "first_use":
                continue
            for role in PRIMARY:
                t = pooled(camp, v, c, role)
                if t:
                    rows.append([v, c[0], role, median(t), median(pooled(camp, v, c, role, 1)), median(pooled(camp, v, c, role, 2)),
                                 median(pooled(camp, v, c, role, 3)), len(t)])
        names = sorted({(cell[0], cell[2]) for p in camp.primary.values() for cell in p.data if cell[1] == "first_use_part"})
        for case, step in names:
            for role in PRIMARY:
                x = pooled(camp, v, (case, "first_use_part", step), role)
                if x:
                    parts.append([v, case, role, step, median(x)])
    a = md_table(["julia", "case", "role", "median s", "median GC s", "median bytes", "median allocations", "samples"], rows)
    b = md_table(["julia", "case", "role", "setup step", "median s"], parts)
    return a, b


def table_per_call(camp):
    rows = []
    for v in EXPECTED_VERSIONS:
        for c in camp.cells:
            if c[1] == "first_use" or c[1] == "penalty" or (c[1] == "warm" and c[2][1] == 1.0 and c[2][2] == "rand"):
                vals = [(role, pooled(camp, v, c, role)) for role in PRIMARY]
                if any(x for _, x in vals):
                    rows.append([v, c[0], c[1], param_text(c[2])] + [median(x) for _, x in vals])
    return md_table(["julia", "case", "measurement", "param"] + PRIMARY, rows)


def table_plans(camp):
    rows = {}
    for (v, o, pid), p in camp.primary.items():
        for (case, param), t in p.plans.items():
            rows.setdefault((v, case, param), []).append(t)
    out = []
    for (v, case, param), ts in sorted(rows.items()):
        kind, _, rest = param.partition("|")
        out.append([v, case, kind, rest.replace("|", ", "), median(ts), len(ts)])
    return md_table(["julia", "case", "group", "calls per batch and batches", "slowest call s (median over processes)", "processes"], out)


def table_alloc(camp):
    rows, nonzero = [], []
    for v in EXPECTED_VERSIONS:
        agg = collections.defaultdict(lambda: [0.0, 0.0, set(), 0])
        for (vv, o, pid), p in camp.primary.items():
            if vv != v:
                continue
            for cell, roles in p.extra.items():
                if cell[1] != "alloc":
                    continue
                for role, d in roles.items():
                    for (_gc, b, a) in d.values():
                        s = agg[(role, cell[2])]
                        s[0], s[1], s[3] = max(s[0], b), max(s[1], a), s[3] + 1
                        if b != 0 or a != 0:
                            s[2].add(cell[0])
        for (role, call), s in sorted(agg.items()):
            rows.append([v, role, call, s[3], s[0], s[1], len(s[2])])
            nonzero.extend([v, role, call, c] for c in sorted(s[2]))
    a = md_table(["julia", "role", "call", "samples", "max bytes per call", "max allocations per call", "cases with a nonzero count"], rows)
    return a + ("\n\nCases with a nonzero count:\n\n" + listing(nonzero, ["julia", "role", "call", "case"], "cases") if nonzero else "")


def table_workingset(camp):
    rows = []
    for v in EXPECTED_VERSIONS:
        cases = {}
        for (vv, o, pid), p in camp.primary.items():
            if vv != v:
                continue
            l3 = l3_bytes(p.headers[0].get("l3_cache", ""))
            for (case, role), b in p.workingset.items():
                e = cases.setdefault((case, role), [0.0, l3])
                e[0] = max(e[0], b)
        loads = {}
        for (vv, o, pid), p in camp.primary.items():
            if vv == v:
                for (case, grp), la in p.loads.items():
                    loads[case] = max(loads.get(case, 0.0), la)
        for (case, role), (b, l3) in sorted(cases.items()):
            if role in PRIMARY:
                rows.append([v, case, role, b, l3, b / l3 if finite(l3) and l3 > 0 else math.nan, loads.get(case, math.nan)])
    return md_table(["julia", "case", "role", "reachable bytes", "L3 bytes", "ratio", "max load average (1 min) during the case"], rows)


def table_domain(camp):
    """rho/sigma_max^2 of every (case, step) and the accuracy of each role there; the domain per design."""
    rows, dom = [], collections.defaultdict(lambda: [0, 0, 0.0, math.inf])
    for c in camp.man.cases.values():
        if c["kind"] != "main":
            continue
        for g in c["acc_gammas"]:
            ratio = (1.0 / g) / (c["sigma_max"] ** 2)
            row = [c["id"], c["design"], c["eltype"], g, ratio]
            for role in PRIMARY:
                worst, nfail, nrows = 0.0, 0, 0
                for v in EXPECTED_VERSIONS:
                    for k in range(1, c["nv"] + 1):
                        r = camp.acc.get((v, c["id"], role), {}).get(gk(g, k))
                        if r is not None:
                            nrows += 1
                            worst = max(worst, r["eta"] / EPS[c["eltype"]])
                            nfail += 0 if r["pass"] else 1
                row.append("-" if nrows == 0 else "%s (%s)" % (fmt(worst), "fail" if nfail else "pass"))
                if nrows and role != "S":
                    d = dom[(c["design"], c["eltype"], role)]
                    d[0] += 1
                    d[1] += 1 if nfail else 0
                    d[2] = max(d[2], ratio) if nfail else d[2]
                    d[3] = min(d[3], ratio)
            rows.append(row)
    a = md_table(["case", "design", "type", "step gamma", "rho / sigma_max^2"] + ["%s max eta/eps" % r for r in PRIMARY], rows)
    drows = [[d, t, r, v[0], v[1], v[2] if v[1] else None, v[3] if v[0] else None] for (d, t, r), v in sorted(dom.items())]
    b = md_table(["design", "type", "variant", "(case, step) pairs with evidence", "pairs that fail", "largest rho/sigma_max^2 of a failing pair",
                  "smallest rho/sigma_max^2 evaluated"], drows)
    return a, b


def table_accuracy(camp):
    out = []
    rows, margin, ctl_rows, fwd = [], [], [], []
    for v in sorted(camp.acc_versions):
        for kind in ("main", "control"):
            for role in PRIMARY:
                rs = [r for (vv, case, rl), d in camp.acc.items() if vv == v and rl == role for r in d.values() if r["kind"] == kind]
                if rs:
                    rows.append([v, kind, role, len(rs), sum(1 for r in rs if not r["pass"]),
                                 max(r["eta"] / EPS[r["eltype"]] for r in rs), median([r["eta"] / EPS[r["eltype"]] for r in rs])])
        for case in camp.man.cases.values():
            ss = camp.acc.get((v, case["id"], "S"), {})
            if case["kind"] == "main" and ss:
                smax = max(r["eta"] / EPS[case["eltype"]] for r in ss.values())
                if smax > ETA_FACTOR / S_MARGIN:
                    margin.append([v, case["id"], smax])
        for role in PRIMARY:
            xs = [r["fe"] for (vv, case, rl), d in camp.acc.items() if vv == v and rl == role for r in d.values() if finite(r["fe"])]
            if xs:
                fwd.append([v, role, camp.acc_forward_max.get(v, "?"), len(xs), median(xs), max(xs)])
    out.append(("Accuracy: outputs, failures (eta > 1e3 eps, or a recorded failure) and eta/eps per role",
                md_table(["julia", "kind", "role", "outputs", "failures", "max eta/eps", "median eta/eps"], rows)))
    pc = []
    for v in sorted(camp.acc_versions):
        for case in camp.man.cases.values():
            if case["kind"] != "main":
                continue
            ss = camp.acc.get((v, case["id"], "S"), {})
            if not ss:
                continue
            row = [v, case["id"], max(r["eta"] / EPS[case["eltype"]] for r in ss.values())]
            for var in VARIANTS:
                rv = camp.acc.get((v, case["id"], var), {})
                if not rv:
                    row += [math.nan, "-", math.nan]
                    continue
                ratio = [rv[k]["eta"] / ss[k]["eta"] for k in rv if k in ss and ss[k]["eta"] > 0 and finite(rv[k]["eta"])]
                row += [max(r["eta"] / EPS[case["eltype"]] for r in rv.values()), sum(1 for r in rv.values() if not r["pass"]),
                        max(ratio) if ratio else math.nan]
            pc.append(row)
    heads = ["julia", "case", "S max eta/eps"]
    for var in VARIANTS:
        heads += ["%s max eta/eps" % var, "%s failures" % var, "%s max eta/eta_S" % var]
    out.append(("Accuracy per main case (all steps by inputs)", listing(pc, heads, "cases")))
    out.append(("Margin of S under the threshold: main cases where S exceeds eta/eps = %g (a factor %g below 1e3)" % (ETA_FACTOR / S_MARGIN, S_MARGIN),
                md_table(["julia", "case", "S max eta/eps"], margin) if margin else
                "No main case with accuracy evidence has an S output above eta/eps = %g (cases without evidence are not covered)." % (ETA_FACTOR / S_MARGIN)))
    reduces = collections.Counter()
    sfail = collections.Counter()
    for v in sorted(camp.acc_versions):
        for case in camp.man.cases.values():
            if case["kind"] != "control":
                continue
            rs = camp.acc.get((v, case["id"], "S"), {})
            if not rs:
                continue
            r_s = next(iter(rs.values()))
            row = [v, case["id"], r_s["eta"] / EPS[case["eltype"]], "pass" if r_s["pass"] else "fail"]
            if not r_s["pass"]:
                sfail[v] += 1
            for var in VARIANTS:
                rv = camp.acc.get((v, case["id"], var), {})
                if rv:
                    r = next(iter(rv.values()))
                    row += [r["eta"] / EPS[case["eltype"]], "pass" if r["pass"] else "fail"]
                    if not r_s["pass"] and r["eta"] < r_s["eta"]:
                        reduces[(v, var)] += 1
                else:
                    row += [math.nan, "-"]
            ctl_rows.append(row)
    heads = ["julia", "control", "S eta/eps", "S"]
    for var in VARIANTS:
        heads += ["%s eta/eps" % var, var]
    out.append(("Accuracy controls from #1078 (eta/eps and pass/fail)", listing(ctl_rows, heads, "controls")))
    out.append(("#1078: controls where S fails and the variant has a smaller eta than S (that does not by itself pass)",
                md_table(["julia", "variant", "controls with eta_V < eta_S", "controls where S fails"],
                         [[v, var, reduces[(v, var)], sfail[v]] for v in sorted(camp.acc_versions) for var in VARIANTS])))
    out.append(("Forward error against the 128-bit dual solution (cases with n up to the recorded limit; a diagnostic, relative to the norm of the solution, absolute when it is 0)",
                md_table(["julia", "role", "n limit", "outputs", "median", "max"], fwd)))
    return out


def table_outputs(camp):
    if not camp.outputs:
        return "Unavailable: no outputs-index.tsv was accepted."
    bad = []
    for v, rows in sorted(camp.outputs.items()):
        for r in rows:
            if r["nfail"] != "0" or r["copy_identical"] != "true":
                bad.append([v, r["case"], r["role"], r["nfail"], r["copy_identical"]])
    if not bad:
        return "Observed in %d Julia version(s): no output with a recorded failure, and every A/A copy bitwise identical to its role." % len(camp.outputs)
    return listing(bad, ["julia", "case", "role", "recorded failures", "copy bitwise identical"], "outputs")


def table_admm(camp):
    out = []
    pstar, why = camp.reference()
    rrows = []
    for kv in camp.refs:
        rrows.append([kv.get("case"), kv.get("julia"), kv.get("iterations"), kv.get("converged"), kv.get("stationarity"),
                      kv.get("objective"), kv.get("cap"), kv.get("_file")])
    out.append(("Reference optima (S-ADMM at rho = 1 to stationarity 1e-12 max(1, |X'y|inf)); usable for the instance of section 11.1: %s" %
                ("yes, objective %s" % fmt(pstar) if pstar is not None else "no (%s)" % why),
                md_table(["case", "julia", "iterations", "converged", "stationarity", "objective", "cap", "file"], rrows)))
    lv = []
    traj = []
    for (v, case), rows in sorted(camp.untimed.items()):
        for r in rows:
            if r["record"] == "level":
                lv.append([v, case, r["rho"], r["role"], r["level"], r["reached"], r["k"], r["cap"], num(r["r_norm"]), num(r["s_norm"]),
                           num(r["stationarity"]), num(r["objective"])])
            elif r["record"] == "trajectory" and r["role"] == "S":
                traj.append([v, case, r["rho"], r["k"], num(r["r_norm"]), num(r["s_norm"]), num(r["stationarity"]), num(r["objective"])])
            elif r["record"] == "error":
                lv.append([v, case, r["rho"], r["role"], "error", "false", "0", r["cap"], math.nan, math.nan, math.nan, math.nan])
    out.append(("ADMM stopping records of the untimed pass: whether the target was reached, the iteration count k (the cap run when not reached) and the measures at k",
                listing(lv, ["julia", "case", "rho", "role", "level", "reached", "k", "cap", "r", "s", "stationarity", "objective"], "records")))
    out.append(("ADMM trajectory of S at the checkpoints", listing(traj, ["julia", "case", "rho", "k", "r", "s", "stationarity", "objective"], "rows")))
    rows, orows = [], []
    for v in EXPECTED_VERSIONS:
        for cell in [c for c in camp.cells if c[1] == "admm_total"]:
            for var in VARIANTS:
                r = camp.cell(v, cell, var)
                f = judge(camp.primary, v, (cell[0], "admm_factor", cell[2]), var)
                it = judge(camp.primary, v, (cell[0], "admm_iter", cell[2]), var)
                rows.append([v, cell[0], param_text(cell[2]), var, r["center"], r["label"] + ("" if r["complete"] else " (incomplete)"), f["center"], it["center"], r["n"],
                             "; ".join(r.get("gaps", [])[:2])])
    out.append(("ADMM: total time ratio V/S per instance, rho and stopping level (and the ratios of the factorization and of the iterations)",
                listing(rows, ["julia", "case", "param", "variant", "total V/S", "label", "factor V/S", "iterations V/S", "processes", "gaps"], "rows")))
    for v in EXPECTED_VERSIONS:
        for cell in [c for c in camp.cells if c[1] == "admm_total"]:
            for role in PRIMARY:
                objs, _ = camp.objective_runs(v, cell, role)
                if objs:
                    st = pooled(camp, v, (cell[0], "admm_stationarity", cell[2]), role)
                    orows.append([v, cell[0], param_text(cell[2]), role, len(objs), median(objs), (max(objs) - pstar) if pstar is not None else math.nan,
                                  median(st)])
    out.append(("ADMM: the objective of the timed runs (median), its largest gap to the usable reference (nan without one) and the median final stationarity",
                listing(orows, ["julia", "case", "param", "role", "runs", "median objective", "max gap", "median stationarity"], "rows")))
    return out


def table_diag(camp):
    rows = []
    for key, p in sorted(camp.diag.items()):
        for cell, roles in sorted(p.data.items(), key=lambda x: str(x[0])):
            if cell[1] not in ("first_use", "warm", "penalty"):
                continue
            for var in VARIANTS:
                r = ratios(roles, var, "S", expected_rounds(cell[1]))
                if r:
                    rows.append([key[0], key[1], key[2], key[3], cell[0], cell[1], param_text(cell[2]), var, median(r.values()), min(r.values()), max(r.values())])
    return listing(rows, ["julia", "order", "proc", "BLAS threads", "case", "measurement", "param", "variant", "median V/S", "min", "max"], "rows")


def format_verdict(v):
    lines = []
    prov = v["provisional"]
    if prov:
        lines.append("**Provisional.** The evidence is incomplete or inconsistent:\n")
        lines.extend("- " + t for t in prov[:MAX_LISTED])
        if len(prov) > MAX_LISTED:
            lines.append("- (%d further reasons not listed)" % (len(prov) - MAX_LISTED))
        lines.append("")
    lines.append("Instance of section 11.1: %s\n" % v["instance"])
    rows = []
    for var, d in v["variants"].items():
        for name, c in d["criteria"].items():
            rows.append([var, name, {True: "True", False: "False", None: "unknown"}[c["value"]], c["info"]])
        rows.append([var, "**recommendation**", "(%s)%s" % (d["recommendation"], " provisional" if prov else ""),
                     "unknown criteria: %s" % ", ".join(d["unknown"]) if d["unknown"] else ""])
    lines.append(md_table(["variant", "criterion", "value", "evidence"], rows))
    lines.append("\n(A) propose a PB fixed-penalty backend: A1, A2 and A3 all true. (B) solver-side factorization only: not (A) because a "
                 "criterion of (A) is false, and B1, the ADMM speed criterion and B3 true. (C) no change when a criterion of (B) is false. "
                 "A criterion with missing evidence is unknown and decides neither way. The recommendation does not close #1078; see the "
                 "table of controls for the residuals.")
    return "\n".join(lines)


def main(argv):
    root = argv[1]
    notes = Notes()
    camp = Campaign(root, argv[2] if len(argv) > 2 else None, notes)
    out = []

    def section(title, body):
        out.append("\n## %s\n\n%s\n" % (title, body))

    section("Processes", table_processes(camp))
    if camp.man is None:
        section("Design", "No manifest: no table of cells can be built.")
    else:
        section("A/A controls: deviation of the per-process median of the copy over its role from 1", table_aa(camp))
        section("Timing labels per measurement and variant (cells; excluded = fails accuracy at the steps and inputs it uses)", table_verdicts(camp))
        section("Timing detail: ratios V/S per cell (median of the per-process medians; range of the per-process medians; minimum and "
                "maximum of the per-round ratios; A/A spread); the per-process values are in per-process-ratios.tsv",
                table_detail(camp, root))
        a, b = table_first_use(camp)
        section("First use: median seconds, GC seconds, bytes and allocations of the primary roles", a)
        section("First-use breakdown: median seconds of each setup step (primary roles)", b)
        section("Median seconds per call of the primary roles (first use, warm at step 1 on random inputs, penalty schedules)", table_per_call(camp))
        section("Batch plans: calls per batch and batches, from the calibration of each case", table_plans(camp))
        section("Warmed allocations per call (maximum over cases; cases with a nonzero count)", table_alloc(camp))
        section("Working set (reachable bytes of each object) against the L3 size, and the load average during each case", table_workingset(camp))
        a, b = table_domain(camp)
        section("rho / sigma_max^2 of each case and step, with the accuracy of each role there (worst over both Julia versions and the 8 inputs)", a)
        section("Accuracy domain per design: the failing pairs of each variant and the range of rho / sigma_max^2", b)
        for title, body in table_accuracy(camp):
            section(title, body)
        section("Outputs with a recorded failure or whose A/A copy differs bitwise", table_outputs(camp))
        for title, body in table_admm(camp):
            section(title, body)
        if camp.diag:
            section("Diagnostic processes (BLAS threads other than %s): ratios V/S per cell; they take no part in the criteria" % PRIMARY_BLAS, table_diag(camp))
    camp.completeness()
    verdict = camp.recommend()
    for var, d in verdict["variants"].items():
        for k in d["unknown"]:
            notes.prov("%s: the criterion %s is unknown" % (var, k))
    verdict["provisional"] = list(notes.provisional)
    verdict["is_provisional"] = bool(notes.provisional)
    section("Criteria and recommendation", format_verdict(verdict))
    section("Notes", "\n".join("- " + n for n in notes.notes) if notes.notes else "none")
    text = "# Summary of %s\n%s\n" % (os.path.basename(os.path.abspath(root)), "\n".join(out))
    print(text)
    with open(os.path.join(root, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(root, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or not os.path.isdir(sys.argv[1]):
        sys.exit("usage: python3 -I summarize.py <output dir> [<inputs dir or inputs-index.tsv>]")
    sys.exit(main(sys.argv))
