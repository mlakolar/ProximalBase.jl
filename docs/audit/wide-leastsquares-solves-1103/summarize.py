#!/usr/bin/env python3
"""Summarizer of the wide least-squares study (#1103); Python standard library only.

Verdict mode (the default) judges one registered campaign under one pinned identity:

    python3 -I summarize.py <campaign dir> --source-sha <sha> --harness-sha <sha> --manifest <campaign dir>/inputs-index.tsv

Tables only, a diagnostic for partial or other data (no criteria, no recommendation, not evidence):

    python3 -I summarize.py <dir> --tables-only [--manifest <inputs-index.tsv>] [--source-sha <sha>] [--harness-sha <sha>]

Gathering a campaign from the producer tree into a new directory, with its checksum list (kept with the raw results):

    python3 -I summarize.py --gather <producer dir> <new campaign dir> [--inputs <inputs-index.tsv>]

--write-checksums <dir> rewrites the checksum list of a directory over its ALLOWLIST files (used by the tests to
re-list edited copies); a campaign is gathered with --gather.

The campaign directory is searched recursively for the files that match ALLOWLIST: the manifest, the files of harness.jl
(timings-*.tsv, admm-untimed-*.tsv, reference-*.tsv with their z binaries, outputs-index.tsv with outputs-failures.tsv)
and of accuracy.jl (accuracy-*.tsv).

Scope: the checks guard against incomplete, mixed, misplaced, truncated or corrupted evidence, and against values that
contradict another value of the same record or of a related file. The authenticity of the values rests on the pinned
harness and the checksum list; an edit that keeps the files consistent with each other is outside these checks.

Verdict mode requires: the full profile of generate.jl (the case and control inventory below, checked against the
manifest gathered into the campaign directory), both Julia versions 1.10.13 and 1.13.4, the three processes of each role
order with one BLAS thread on one CPU kind (one BLAS configuration per Julia version), 21 rounds with all eight roles,
5 ADMM repeats with the roles of the families that reached each level, one untimed file per Julia version and lasso
instance, one converged reference per lasso instance, per Julia version one outputs index, one failures index and one
accuracy file of the full manifest, and the checksum list. Before computing anything it runs a preflight in three
classes: (1) the manifest, the pinned identity and the checksum list; (2) each file on its own (format, closing line,
no repeated header key, identity, conditions, parseable fields, values in their domain and consistent within the record,
no duplicate rows, no recorded `error` row) and the files read against the checksum list (the same paths, and the
SHA-256 of the bytes that were parsed); (3) the inventory and the consistency across files (missing or extra files and
rows, the role order of each round against the file's order, batch plans against their rule, the ADMM counts against
the untimed records, the ADMM end points against their targets, CPU and BLAS metadata, the accuracy rows against the
outputs and failures). At the first class with a problem it prints the problems it found in that class (a file with
another format or other columns is not checked further), exits with status 2 and writes no verdict (it removes a
verdict.json, tables.md and per-process-ratios.tsv left in the directory by an earlier run). When the preflight passes,
it applies the criteria of the plan as Boolean logic and writes tables.md, per-process-ratios.tsv and verdict.json into
the campaign directory.

Criteria as implemented (the plan is the reference; see the README):
  * Per cell (case, measurement, parameter) and variant V in {C_B, C_D, C_W}: each of the six processes of a Julia
    version gives the paired per-round ratios V/S, S'/S and V'/V, and their medians. A/A spread = the largest
    |median - 1| of S'/S and V'/V over the six processes. faster: all six medians of V/S are below 0.95 and
    1 - max(medians) exceeds the spread; slower: all six are above 1.05 and min(medians) - 1 exceeds the spread;
    otherwise inconclusive.
  * ADMM cells: when S or V is capped at the level (the untimed run went through the cap without meeting the target),
    the cell is `capped`, a lower bound, which counts as inconclusive.
  * An output passes accuracy when eta <= 1e3 eps(T), decided by accuracy.jl in BigFloat. The accuracy evidence of a
    cell is a set of (step, input) pairs: first use, step 1 and the four random inputs; warm, its step and its four
    inputs; penalty, the pairs that one cycle of the schedule executes (recorded by the harness in the rows
    penalty_pairs and checked against the same rule here); ADMM, the step 1/rho with the four stored ADMM vectors,
    registered test vectors sampled from an S-ADMM run at rho = 1, which the timed runs do not take as inputs. A cell whose variant or S fails at one of them is `excluded`:
    its timing is reported and it is not faster for any criterion.
  * Recommendation per variant (recommend()): (A) when A1, A2, A3 speed and A3 gap are true; (B) when (A) does not
    hold and B1, A3 speed and B3 are true; (C) when neither (A) nor (B) holds.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import sys

# ---------------------------------------------------------------------------------------------------------------------
# The registered campaign
# ---------------------------------------------------------------------------------------------------------------------

VERSIONS = ("1.10.13", "1.13.4")
REFERENCE_JULIA = "1.13.4"
ORDERS = ("S_first", "C_first")
PROCS = ("1", "2", "3")
ROUNDS, REPEATS, BATCHES = 21, 5, 3
ADMM_CAP, REFERENCE_CAP = 20000, 100000
ADMM_TIGHT, REFERENCE_TOL = 1e-8, 1e-12
BATCH_CALL_CAP = 3000
CHECK_BOUNDS_NO = "2"
TIMED_GROUPS = ("first_use", "warm", "penalty", "alloc", "admm")
ROLE_ORDER = {"S_first": ["S", "S'", "C_B", "C_B'", "C_D", "C_D'", "C_W", "C_W'"],
              "C_first": ["C_B", "C_B'", "C_D", "C_D'", "C_W", "C_W'", "S", "S'"]}
ROLES = ROLE_ORDER["S_first"]
PRIMARY = ["S", "C_B", "C_D", "C_W"]
VARIANTS = ["C_B", "C_D", "C_W"]
PARTS = {"S": ("copy_check", "svd", "V", "UtY_s2"), "C_B": ("check_alloc", "gram", "rhs", "factor"),
         "C_D": ("check_alloc", "gram", "factor"), "C_W": ("check_alloc", "gram", "factor", "whiten")}
SIZES = [(100, 1000), (500, 5000), (1000, 2000), (1000, 1100), (200, 20000), (1500, 5000)]
LASSO_SIZES = [(1500, 5000), (500, 5000), (200, 20000)]
DESIGNS = (("gauss", ("noisy",)), ("rankhalf", ("consistent", "inconsistent")), ("duprows", ("consistent", "inconsistent")))
TAG = {"Float64": "f64", "Float32": "f32"}
GRID = {"Float64": (1e-3, 1.0, 1e3, 1e6), "Float32": (1e-3, 1.0, 1e3)}
ADMM_STEPS = (0.1, 10.0)
SCHEDULES = {"alt2": {"Float64": (1.0, 1e3), "Float32": (1.0, 1e3)},
             "newstep": {"Float64": (1e-3, 1e3, 1.0, 1e6), "Float32": (1e-3, 1e3, 1.0)}}
NV_MAIN, NV_RAND = 8, 4
ADMM_RHOS = (0.1, 1.0, 10.0)
ADMM_LEVELS = ("boyd", "tight")
ADMM_TIMED = ("admm_total", "admm_factor", "admm_iter", "admm_objective", "admm_stationarity", "admm_valid", "admm_r", "admm_s",
              "admm_eps_pri", "admm_eps_dual")
# relative slack for comparisons of values written with 10 significant digits
SLACK = 1e-9
INSTANCE_11_1 = "f64/gauss/noisy/1500x5000"
CONTROL_DESIGNS = (("axis1x2", 1, 2), ("axis1x2n", 1, 2), ("axis1x3", 1, 3), ("diag2x4", 2, 4), ("diag2x4z", 2, 4),
                   ("rot1x2", 1, 2), ("rot2x4", 2, 4), ("rot2x4z", 2, 4), ("sq2x2", 2, 2), ("tall3x2", 3, 2))
# the steps 1/rho of the controls, rho = eps(T)/8, eps(T), 32 eps(T) (powers of two, exact in Float64)
CONTROL_STEPS = {"Float64": (("eps8", 2.0 ** 55), ("eps", 2.0 ** 52), ("eps32", 2.0 ** 47)),
                 "Float32": (("eps8", 2.0 ** 26), ("eps", 2.0 ** 23), ("eps32", 2.0 ** 18))}

EPS = {"Float64": 2.0 ** -52, "Float32": 2.0 ** -23}
ETA_FACTOR, S_MARGIN = 1e3, 10.0
FASTER, SLOWER = 0.95, 1.05
GAP_SLACK = 1e-8
MAX_LISTED = 40

FORMATS = {"manifest": "pb1103-inputs-2", "timings": "pb1103-timings-3", "untimed": "pb1103-admm-untimed-3",
           "reference": "pb1103-reference-1", "outputs": "pb1103-outputs-2", "failures": "pb1103-outputs-failures-1",
           "accuracy": "pb1103-accuracy-4"}
MANIFEST_COLUMNS = ["case", "kind", "eltype", "design", "response", "n", "p", "seed", "timed", "lasso", "gammas",
                    "acc_gammas", "lambda_max", "sigma_max", "array", "file", "shape", "sha256", "note"]
TIMING_COLUMNS = ["case", "measurement", "param", "round", "position", "role", "value", "gc_s", "bytes", "allocs"]
UNTIMED_MEASURES = ["r_norm", "s_norm", "eps_pri", "eps_dual", "stationarity", "objective"]
UNTIMED_COLUMNS = ["case", "rho", "role", "record", "level", "status", "k", "cap"] + UNTIMED_MEASURES + ["note"]
ADMM_CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
OUTPUT_COLUMNS = ["case", "role", "file", "eltype", "shape", "sha256", "nfail", "copy_identical", "acc_gammas", "nv"]
FAILURE_COLUMNS = ["case", "role", "gamma_index", "vindex", "message"]
ACCURACY_COLUMNS = ["case", "eltype", "kind", "role", "gamma_index", "gamma", "vindex", "eta", "forward_error", "pass",
                    "status", "copy_identical"]
GROUP_OF = {"first_use": "first_use", "first_use_part": "first_use", "warm": "warm", "penalty": "penalty",
            "penalty_pairs": "penalty", "alloc": "alloc", "admm_iters": "admm", "admm_reached": "admm"}
GROUP_OF.update({m: "admm" for m in ADMM_TIMED})
CONTEXT = ("batch_plan", "loadavg_1m", "workingset")
SUMMARY_FILES = ("verdict.json", "tables.md", "per-process-ratios.tsv")
# The gathered campaign: the file-name patterns of the files that verdict mode reads (searched recursively), and the
# checksum list written over them at gather time (--write-checksums).
ALLOWLIST = ("inputs-index.tsv", "timings-*.tsv", "admm-untimed-*.tsv", "reference-*.tsv", "reference-*-z.f64",
             "outputs-index.tsv", "outputs-failures.tsv", "accuracy-*.tsv")
CHECKSUM_FILE = "sha256-list.tsv"
CHECKSUM_FORMAT = "pb1103-checksums-1"


def registered_cases():
    """The case inventory of the full profile of generate.jl, in its order: id -> fields."""
    cases = collections.OrderedDict()
    for (n, p) in SIZES:
        for design, responses in DESIGNS:
            for t in ("Float64", "Float32"):
                for resp in responses:
                    lasso = t == "Float64" and design == "gauss" and (n, p) in LASSO_SIZES
                    acc = tuple(sorted(set(GRID[t]) | set(ADMM_STEPS))) if lasso else GRID[t]
                    cid = "%s/%s/%s/%dx%d" % (TAG[t], design, resp, n, p)
                    cases[cid] = dict(kind="main", eltype=t, design=design, response=resp, n=n, p=p,
                                      timed=resp != "inconsistent", lasso=lasso, gammas=GRID[t], acc_gammas=acc, nv=NV_MAIN)
    for t in ("Float64", "Float32"):
        for sgn in ("pos", "neg"):
            for rtag, g in CONTROL_STEPS[t]:
                for name, n, p in CONTROL_DESIGNS:
                    cid = "c1078/%s/%s/%s/%s" % (TAG[t], name, sgn, rtag)
                    cases[cid] = dict(kind="control", eltype=t, design=name, response="control", n=n, p=p, timed=False,
                                      lasso=False, gammas=(g,), acc_gammas=(g,), nv=1)
    return cases


def executed_pairs(steps, K=NV_RAND):
    """The (step, input index) pairs of one cycle of a penalty batch: call i uses steps[mod1(i, G)] and input mod1(i, K),
    for i = 1 .. lcm(G, K) (schedule_call of harness.jl)."""
    G = len(steps)
    L = G * K // math.gcd(G, K)
    return [(steps[(i - 1) % G], (i - 1) % K + 1) for i in range(1, L + 1)]


def round_indices(n, r):
    """The role indices (0-based) by position in round r: the process order in odd rounds, reversed in even ones."""
    return list(range(n)) if r % 2 == 1 else list(range(n - 1, -1, -1))


def case_file(cid):
    return cid.replace("/", "_").replace("=", "-").replace("+", "p")


def family(role):
    return role.rstrip("'")


def f32(x):
    """x rounded to the nearest Float32 (round to nearest even)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def penalty_of(g, eltype):
    """rho = inv(T(g)) of the prototypes and LeastSquaresLoss. For Float32 the quotient is formed in Float64 and
    rounded once more; for division, a 53-bit intermediate rounded to 24 bits gives the correctly rounded quotient."""
    return 1.0 / g if eltype == "Float64" else f32(1.0 / f32(g))


def gk(g, k):
    """The key of a step and an input index: the step rounded to 12 significant digits."""
    return (float("%.12g" % g), k)


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def finite(x):
    return isinstance(x, float) and x == x and abs(x) != math.inf


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


def to_int(s):
    return int(s)


def to_float(s):
    return float(s)


# The SHA-256 of each file this script has read, keyed by its absolute path: each digest is of the bytes that were
# parsed (the file is read once), and the checksum list is compared with this record.
DIGESTS = {}


def read_bytes(path):
    with open(path, "rb") as fh:
        data = fh.read()
    DIGESTS[os.path.abspath(path)] = hashlib.sha256(data).hexdigest()
    return data


def sha256_file(path):
    read_bytes(path)
    return DIGESTS[os.path.abspath(path)]


class Skipped(Exception):
    """A file that cannot be read as a file of the study."""


def read_tsv(path):
    """(header, columns, rows) of a tab-separated file of the study. A `# key: value` line anywhere is a header line; a
    repeated header key makes the file unusable."""
    try:
        lines = [l for l in read_bytes(path).decode("utf-8", errors="replace").split("\n") if l != ""]
    except OSError as e:
        raise Skipped("unreadable: %s" % e)
    if not lines or lines[-1] != "# complete: yes":
        raise Skipped("no closing `# complete: yes` line (interrupted write)")
    header, cols, rows = {}, None, []
    for i, line in enumerate(lines):
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            if k.strip() in header:
                raise Skipped("header key %r repeated" % k.strip())
            header[k.strip()] = v.strip()
        elif cols is None:
            cols = line.split("\t")
        else:
            f = line.split("\t")
            if len(f) != len(cols):
                raise Skipped("line %d has %d fields instead of %d" % (i + 1, len(f), len(cols)))
            rows.append(f)
    if cols is None:
        raise Skipped("no column line")
    return header, cols, rows


def read_kv(path):
    try:
        lines = [l for l in read_bytes(path).decode("utf-8", errors="replace").split("\n") if l != ""]
    except OSError as e:
        raise Skipped("unreadable: %s" % e)
    if not lines or lines[-1] != "complete\tyes":
        raise Skipped("no closing `complete yes` line (interrupted write)")
    kv = {}
    for i, l in enumerate(lines[:-1]):
        k, sep, v = l.partition("\t")
        if not sep:
            raise Skipped("line %d is not `key<TAB>value`" % (i + 1))
        if k in kv:
            raise Skipped("key %s repeated" % k)
        kv[k] = v
    return kv


class Problems:
    """Problems grouped by (where, what): the number of occurrences and the first detail."""

    def __init__(self):
        self.items = collections.OrderedDict()

    def add(self, where, what, detail=""):
        e = self.items.get((where, what))
        if e is None:
            self.items[(where, what)] = [1, detail]
        else:
            e[0] += 1

    def files(self):
        return {w for (w, _) in self.items}

    def __bool__(self):
        return bool(self.items)

    def __len__(self):
        return len(self.items)

    def lines(self):
        out = []
        for (where, what), (n, detail) in self.items.items():
            out.append("%s: %s%s%s" % (where, what, " (%d occurrences)" % n if n > 1 else "", "; first: %s" % detail if detail else ""))
        return out


# ---------------------------------------------------------------------------------------------------------------------
# The checksum list
# ---------------------------------------------------------------------------------------------------------------------

def campaign_files(root):
    """The files under root that match ALLOWLIST, as paths relative to root with / separators."""
    out = set()
    for pat in ALLOWLIST:
        for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
            if os.path.isfile(p):
                out.add(os.path.relpath(p, root).replace(os.sep, "/"))
    return sorted(out)


def write_checksums(root):
    lines = ["# format: %s" % CHECKSUM_FORMAT, "path\tsha256"]
    files = campaign_files(root)
    for f in files:
        with open(os.path.join(root, f), "rb") as fh:
            lines.append("%s\t%s" % (f, hashlib.sha256(fh.read()).hexdigest()))
    lines.append("# complete: yes")
    tmp = os.path.join(root, CHECKSUM_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(root, CHECKSUM_FILE))      # a new file, not a rewrite of the old one
    return len(files)


# The destination of each kind of file in a gathered campaign: kind -> (file-name pattern, destination directory). The
# outputs indices and accuracy files go into the directory named like their producer directory (outputs-<julia>).
GATHER_KINDS = (("timings", "timings-*.tsv", "timings"), ("untimed", "admm-untimed-*.tsv", "reference"),
                ("reference", "reference-*.tsv", "reference"), ("z", "reference-*-z.f64", "reference"),
                ("outputs index", "outputs-index.tsv", None), ("failures index", "outputs-failures.tsv", None),
                ("accuracy", "accuracy-*.tsv", None))


class GatherError(Exception):
    pass


def gather_plan(src, dst, inputs):
    """The source -> destination mapping of a gather, built before anything is copied. It refuses a destination that
    exists and is not empty, a manifest other than `inputs` in the producer tree (or no single manifest when `inputs` is
    not given), two sources with the same destination, an outputs file outside a directory named outputs-<julia>, and a
    kind of file that the tree does not hold."""
    if os.path.exists(dst) and (not os.path.isdir(dst) or os.listdir(dst)):
        raise GatherError("the destination %s exists and is not an empty directory" % dst)
    manifests = sorted(p for p in glob.glob(os.path.join(src, "**", "inputs-index.tsv"), recursive=True) if os.path.isfile(p))
    if inputs:
        if not os.path.isfile(inputs):
            raise GatherError("--inputs %s is not a file" % inputs)
        others = [p for p in manifests if os.path.abspath(p) != os.path.abspath(inputs)]
        if others:
            raise GatherError("the producer tree holds another manifest: %s" % ", ".join(others))
        man = inputs
    elif len(manifests) != 1:
        raise GatherError("give --inputs, or keep exactly one inputs-index.tsv in the producer tree (found %d)" % len(manifests))
    else:
        man = manifests[0]
    plan = {os.path.join(dst, "inputs-index.tsv"): man}
    problems = []
    for kind, pat, sub in GATHER_KINDS:
        found = sorted(p for p in glob.glob(os.path.join(src, "**", pat), recursive=True) if os.path.isfile(p))
        if kind == "reference":
            found = [p for p in found if not p.endswith("-z.f64")]
        if not found:
            problems.append("no %s file (%s) in the producer tree" % (kind, pat))
        for p in found:
            d = sub
            if d is None:
                d = os.path.basename(os.path.dirname(p))
                if not d.startswith("outputs-"):
                    problems.append("%s is not in a directory named outputs-<julia>" % p)
                    continue
            target = os.path.join(dst, d, os.path.basename(p))
            if target in plan:
                problems.append("%s and %s would both be copied to %s" % (plan[target], p, os.path.relpath(target, dst)))
                continue
            plan[target] = p
    if problems:
        raise GatherError("; ".join(problems))
    return plan


def gather(src, dst, inputs):
    """Copies the campaign files of the producer tree src into the new directory dst by gather_plan, stopping at the
    first failed copy, and writes the checksum list from the copied bytes after every copy succeeded."""
    plan = gather_plan(src, dst, inputs)
    os.makedirs(dst, exist_ok=True)
    for target, source in sorted(plan.items()):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            raise GatherError("%s exists already" % target)
        shutil.copyfile(source, target)
    copied = campaign_files(dst)
    if len(copied) != len(plan):
        raise GatherError("the destination holds %d campaign files after copying %d" % (len(copied), len(plan)))
    return write_checksums(dst)


def read_checksums(root, P):
    """path -> SHA-256 from the checksum list of root, or None after recording why it cannot be used."""
    path = os.path.join(root, CHECKSUM_FILE)
    if not os.path.isfile(path):
        P.add(CHECKSUM_FILE, "absent: write it at gather time with --write-checksums")
        return None
    try:
        header, cols, rows = read_tsv(path)
    except Skipped as e:
        P.add(CHECKSUM_FILE, str(e))
        return None
    finally:
        DIGESTS.pop(os.path.abspath(path), None)
    if header.get("format") != CHECKSUM_FORMAT or cols != ["path", "sha256"]:
        P.add(CHECKSUM_FILE, "not a %s file" % CHECKSUM_FORMAT)
        return None
    listed = {}
    for f in rows:
        if f[0] in listed:
            P.add(CHECKSUM_FILE, "path listed twice", f[0])
        listed[f[0]] = f[1]
    return listed


def check_checksums(root, listed, P):
    """The files read under root (DIGESTS) against the list: the same set of paths and the same SHA-256 for each."""
    top = os.path.abspath(root) + os.sep
    read = {os.path.relpath(p, root).replace(os.sep, "/"): d for p, d in DIGESTS.items() if p.startswith(top)}
    for f in sorted(set(listed) - set(read)):
        P.add(CHECKSUM_FILE, "listed files that verdict mode did not read", f)
    for f in sorted(set(read) - set(listed)):
        P.add(CHECKSUM_FILE, "files read that the list does not hold", f)
    for f in sorted(set(read) & set(listed)):
        if read[f] != listed[f]:
            P.add(f, "SHA-256 differs from the checksum list")


# ---------------------------------------------------------------------------------------------------------------------
# Class 1: the manifest and the pinned identity
# ---------------------------------------------------------------------------------------------------------------------

class Manifest:
    def __init__(self, path, header, cases, sha):
        self.path, self.header, self.cases, self.sha256 = path, header, cases, sha


def load_manifest(path, P):
    """The manifest at path, or None after recording why it cannot be read."""
    where = os.path.basename(path)
    try:
        header, cols, rows = read_tsv(path)
    except Skipped as e:
        P.add(where, str(e))
        return None
    if header.get("format") != FORMATS["manifest"]:
        P.add(where, "format %s, expected %s" % (header.get("format", "<absent>"), FORMATS["manifest"]))
        return None
    if cols != MANIFEST_COLUMNS:
        P.add(where, "unexpected columns")
        return None
    cases = collections.OrderedDict()
    meta_cols = [c for c in MANIFEST_COLUMNS if c not in ("array", "file", "shape", "sha256", "note")]
    first_meta = {}
    for i, f in enumerate(rows):
        r = dict(zip(cols, f))
        meta = tuple(r[c] for c in meta_cols)
        if first_meta.setdefault(r["case"], meta) != meta:
            P.add(where, "the rows of a case disagree in their case fields", r["case"])
        try:
            c = cases.get(r["case"])
            if c is None:
                c = cases[r["case"]] = dict(
                    id=r["case"], kind=r["kind"], eltype=r["eltype"], design=r["design"], response=r["response"],
                    n=to_int(r["n"]), p=to_int(r["p"]), seed=to_int(r["seed"]), timed=r["timed"] == "true",
                    lasso=r["lasso"] == "true", gammas=tuple(to_float(x) for x in r["gammas"].split(",")),
                    acc_gammas=tuple(to_float(x) for x in r["acc_gammas"].split(",")),
                    lambda_max=to_float(r["lambda_max"]) if r["lambda_max"] else math.nan,
                    sigma_max=to_float(r["sigma_max"]), arrays={})
            if r["array"] in c["arrays"]:
                P.add(where, "array repeated", "%s %s" % (r["case"], r["array"]))
            c["arrays"][r["array"]] = {"file": r["file"], "shape": [to_int(x) for x in r["shape"].split("x")],
                                       "sha256": r["sha256"]}
        except (ValueError, KeyError) as e:
            P.add(where, "unparseable row", "line of %s: %s" % (r.get("case", "?"), e))
    for c in list(cases.values()):
        if sorted(c["arrays"]) != ["V", "X", "y"]:
            P.add(where, "case does not have exactly the arrays X, y and V", "%s: %s" % (c["id"], sorted(c["arrays"])))
            del cases[c["id"]]
            continue
        c["nv"] = c["arrays"]["V"]["shape"][-1]
    return Manifest(path, header, cases, DIGESTS[os.path.abspath(path)])


def check_manifest_registered(man, pins, P):
    """The manifest is the full profile of generate.jl under the pinned identity."""
    where = os.path.basename(man.path)
    h = man.header
    for key, want in (("profile", "full"), ("filter", ""), ("julia", REFERENCE_JULIA), ("source_sha", pins["source_sha"]),
                      ("harness_sha", pins["harness_sha"])):
        if h.get(key) != want:
            P.add(where, "header %s is %r, expected %r" % (key, h.get(key, "<absent>"), want))
    reg = registered_cases()
    missing = [c for c in reg if c not in man.cases]
    extra = [c for c in man.cases if c not in reg]
    if missing:
        P.add(where, "%d registered cases absent" % len(missing), ", ".join(missing[:5]))
    if extra:
        P.add(where, "%d cases outside the registered inventory" % len(extra), ", ".join(extra[:5]))
    for cid, want in reg.items():
        c = man.cases.get(cid)
        if c is None or "nv" not in c:
            continue
        for key in ("kind", "eltype", "design", "response", "n", "p", "timed", "lasso", "nv"):
            if c[key] != want[key]:
                P.add(where, "case field %s differs from the registered design" % key, "%s: %r, expected %r" % (cid, c[key], want[key]))
        for key in ("gammas", "acc_gammas"):
            if tuple(c[key]) != tuple(want[key]):
                P.add(where, "case field %s differs from the registered design" % key, "%s: %r, expected %r" % (cid, c[key], want[key]))
        if c["arrays"]["X"]["shape"] != [c["n"], c["p"]] or c["arrays"]["y"]["shape"] != [c["n"]] or \
                c["arrays"]["V"]["shape"] != [c["p"], c["nv"]]:
            P.add(where, "array shapes disagree with n, p and the inputs", cid)
        if not (finite(c["sigma_max"]) and c["sigma_max"] > 0):
            P.add(where, "sigma_max not positive and finite", cid)
        if c["kind"] == "main" and not (finite(c["lambda_max"]) and c["lambda_max"] > 0):
            P.add(where, "lambda_max not positive and finite", cid)


# ---------------------------------------------------------------------------------------------------------------------
# Class 2: each file on its own
# ---------------------------------------------------------------------------------------------------------------------

def check_identity(header, man, pins, where, P, keys=("source_sha", "harness_sha")):
    for key in keys:
        if pins.get(key) and header.get(key) != pins[key]:
            P.add(where, "%s is %s, the pinned campaign has %s" % (key, header.get(key, "<absent>"), pins[key]))
    if man is not None and header.get("inputs_index_sha256") != man.sha256:
        P.add(where, "made from the manifest %s, not from the pinned %s" % (header.get("inputs_index_sha256", "<absent>"), man.sha256))


def canon(meas, param):
    """(canonical parameter, parsed extra) of a timing row; ValueError for a malformed parameter."""
    if meas == "first_use":
        if not param.startswith("g="):
            raise ValueError(param)
        return float(param[2:]), None
    if meas == "warm":
        g, kind = param[2:].split("/") if param.startswith("g=") else (None, None)
        if kind not in ("rand", "admm"):
            raise ValueError(param)
        return (float(g), kind), None
    if meas in ("penalty", "first_use_part", "alloc", "loadavg_1m", "workingset"):
        return param, None
    if meas == "batch_plan":
        parts = param.split("|")
        if len(parts) != 5 or [x.split("=", 1)[0] for x in parts[1:]] != ["calls", "batches", "cap", "tmax"]:
            raise ValueError(param)
        kv = dict(x.split("=", 1) for x in parts[1:])
        plan = {"calls": int(kv["calls"]), "batches": int(kv["batches"]), "cap": int(kv["cap"]), "tmax": float(kv["tmax"])}
        return parts[0], plan
    if meas == "penalty_pairs":
        name, _, rest = param.partition("|")
        pairs = []
        for x in rest.split(","):
            g, _, k = x.partition("@")
            pairs.append((float(g), int(k)))
        return name, tuple(pairs)
    if meas in GROUP_OF and GROUP_OF[meas] == "admm":
        r, l = param.split(",")
        if not (r.startswith("rho=") and l.startswith("level=")):
            raise ValueError(param)
        return (float(r[4:]), l[6:]), None
    raise ValueError("measurement %s" % meas)


class TimingFile:
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(path)
        self.header, self.groups, self.rows, self.untimed_ids = {}, [], {}, {}
        self.plans, self.pairs, self.loads, self.workingset = {}, {}, {}, {}


def parse_timing(path, man, pins, verdict, P):
    tf = TimingFile(path)
    where = tf.name
    try:
        header, cols, rows = read_tsv(path)
    except Skipped as e:
        P.add(where, str(e))
        return None
    tf.header = header
    if header.get("format") != FORMATS["timings"] or cols != TIMING_COLUMNS:
        P.add(where, "not a %s file" % FORMATS["timings"], "format %s" % header.get("format", "<absent>"))
        return None
    check_identity(header, man, pins, where, P)
    if header.get("order") not in ORDERS:
        P.add(where, "order %s" % header.get("order", "<absent>"))
    if verdict:
        conds = (("julia", VERSIONS), ("proc", PROCS), ("rounds", (str(ROUNDS),)), ("batches", (str(BATCHES),)),
                 ("admm_repeats", (str(REPEATS),)), ("admm_cap", (str(ADMM_CAP),)), ("threads", ("1",)),
                 ("blas_threads", ("1",)), ("check_bounds", (CHECK_BOUNDS_NO,)))
        for key, allowed in conds:
            if header.get(key) not in allowed:
                P.add(where, "condition %s is %s, the registered campaign has %s" % (key, header.get(key, "<absent>"), " or ".join(allowed)))
    tf.groups = header.get("groups", "").split(",")
    if not tf.groups or any(g not in TIMED_GROUPS for g in tf.groups) or len(set(tf.groups)) != len(tf.groups):
        P.add(where, "groups %s" % header.get("groups", "<absent>"))
    for k, v in header.items():
        if k.startswith("untimed_file "):
            parts = v.split(" ")
            if len(parts) != 2:
                P.add(where, "malformed untimed_file line", k)
            tf.untimed_ids[k[len("untimed_file "):].strip()] = tuple(parts)
    for i, f in enumerate(rows):
        case, meas, param, rnd, pos, role, val, gc, nbytes, nalloc = f
        line = "%s %s %s round %s %s" % (case, meas, param, rnd, role)
        if meas == "error":
            P.add(where, "recorded error rows (the cells must be rerun)", line)
            continue
        if man is not None and case not in man.cases:
            P.add(where, "rows of a case outside the manifest", line)
            continue
        if role not in ROLES and role != "-":
            P.add(where, "rows with an unknown role", line)
            continue
        try:
            rnd, pos = to_int(rnd), to_int(pos)
            vals = (to_float(val), to_float(gc), to_float(nbytes), to_float(nalloc))
            cp, extra = canon(meas, param)
        except (ValueError, KeyError) as e:
            P.add(where, "unparseable field", "%s (%s)" % (line, e))
            continue
        grp = GROUP_OF.get(meas)
        if meas in ("loadavg_1m", "workingset"):
            store, key = (tf.loads, (case, cp)) if meas == "loadavg_1m" else (tf.workingset, (case, cp, role))
            if key in store:
                P.add(where, "duplicate rows", line)
            if cp not in TIMED_GROUPS or cp not in tf.groups or not (finite(vals[0]) and vals[0] >= 0):
                P.add(where, "context rows of an undeclared group or with a value out of range", line)
            store[key] = vals[0]
            continue
        if meas == "batch_plan":
            grp = "warm" if cp == "warm" else ("penalty" if cp in ("penalty:alt2", "penalty:newstep") else None)
            if grp is None:
                P.add(where, "batch_plan of an unknown group", line)
                continue
            if (case, cp) in tf.plans:
                P.add(where, "duplicate rows", line)
            tf.plans[(case, cp)] = (extra, vals[0])
        elif meas == "penalty_pairs":
            if (case, cp) in tf.pairs:
                P.add(where, "duplicate rows", line)
            tf.pairs[(case, cp)] = extra
        if grp not in tf.groups:
            P.add(where, "rows of a group the file does not declare", line)
            continue
        if meas in ("batch_plan", "penalty_pairs"):
            continue
        key = (case, meas, cp, rnd, role)
        if key in tf.rows:
            P.add(where, "duplicate rows", line)
            continue
        tf.rows[key] = (pos,) + vals
    return tf


class Untimed:
    def __init__(self, path, version, case, sha):
        self.path, self.name, self.version, self.case, self.sha = path, os.path.basename(path), version, case, sha
        self.levels, self.traj = {}, []


def status_agrees(level, status, m, target):
    """Whether the measures m (r, s, eps_pri, eps_dual, stationarity, objective) of a level record agree with its status:
    reached needs the target met (boyd: r <= eps_pri and s <= eps_dual; tight: stationarity <= target), capped needs it
    missed, each up to the relative slack of the 10-digit output."""
    hi, lo = 1 + SLACK, 1 - SLACK
    if level == "boyd":
        met = m[0] <= m[2] * hi and m[1] <= m[3] * hi
        missed = m[0] >= m[2] * lo or m[1] >= m[3] * lo
    else:
        met, missed = m[4] <= target * hi, m[4] > target * lo
    return met if status == "reached" else missed


def parse_untimed(path, man, pins, verdict, P):
    """An untimed file: identity and stopping parameters (the scale is max(1, lambda_max) of the manifest), one level row
    per (rho, primary role, level) whose measures agree with its status, finite nonnegative measures, and trajectory
    rows that are the checkpoints up to the last iteration run, that iteration and the reached iterations, without
    repeats, with each level row equal to the trajectory row at its k."""
    where = os.path.basename(path)
    try:
        header, cols, rows = read_tsv(path)
    except Skipped as e:
        P.add(where, str(e))
        return None
    if header.get("format") != FORMATS["untimed"] or cols != UNTIMED_COLUMNS:
        P.add(where, "not a %s file" % FORMATS["untimed"], "format %s" % header.get("format", "<absent>"))
        return None
    check_identity(header, man, pins, where, P)
    case, version = header.get("case", "?"), header.get("julia", "?")
    if verdict and version not in VERSIONS:
        P.add(where, "julia %s" % version)
    c = man.cases.get(case) if man is not None else None
    try:
        cap = to_int(header.get("admm_cap", ""))
        tight = to_float(header.get("tight_tol", ""))
        hscale = to_float(header.get("scale", ""))
    except ValueError:
        P.add(where, "unparseable admm_cap, tight_tol or scale")
        return None
    scale = hscale
    if man is not None:
        if c is None or not c["lasso"]:
            P.add(where, "case %s is not a lasso instance of the manifest" % case)
            return None
        for a, key in (("X", "x_sha256"), ("y", "y_sha256")):
            if header.get(key) != c["arrays"][a]["sha256"]:
                P.add(where, "%s differs from the manifest" % key)
        scale = max(1.0, c["lambda_max"])
        if hscale != scale:
            P.add(where, "scale %r differs from max(1, lambda_max) = %r of the manifest" % (hscale, scale))
    if verdict and cap != ADMM_CAP:
        P.add(where, "admm_cap %d, the registered campaign has %d" % (cap, ADMM_CAP))
    if tight != ADMM_TIGHT:
        P.add(where, "tight_tol %r, the registered campaign has %r" % (tight, ADMM_TIGHT))
    target = ADMM_TIGHT * scale
    u = Untimed(path, version, case, DIGESTS[os.path.abspath(path)])
    u.cap = cap
    texts, traj = {}, collections.defaultdict(dict)
    for f in rows:
        r = dict(zip(cols, f))
        line = "%s rho=%s %s %s %s k=%s" % (r["case"], r["rho"], r["role"], r["record"], r["level"], r["k"])
        if r["case"] != case:
            P.add(where, "rows of another case than the header", line)
            continue
        if r["record"] == "error":
            P.add(where, "recorded error rows (the untimed run must be rerun)", "%s: %s" % (line, r["note"]))
            continue
        try:
            rho = to_float(r["rho"])
            k = to_int(r["k"])
            meas = tuple(to_float(r[x]) for x in UNTIMED_MEASURES)
        except ValueError as e:
            P.add(where, "unparseable field", "%s (%s)" % (line, e))
            continue
        if rho not in ADMM_RHOS or r["role"] not in PRIMARY:
            P.add(where, "rows outside the registered penalties and primary roles", line)
            continue
        if not all(finite(x) and x >= 0 for x in meas) or not 1 <= k <= cap or r["cap"] != str(cap):
            P.add(where, "rows with a measure that is not finite and nonnegative, k out of 1..cap or another cap", line)
            continue
        mtext = tuple(r[x] for x in UNTIMED_MEASURES)
        if r["record"] == "trajectory":
            if k in traj[(rho, r["role"])]:
                P.add(where, "duplicate trajectory rows", line)
            traj[(rho, r["role"])][k] = mtext
            u.traj.append((rho, r["role"], k) + meas)
            continue
        if r["record"] != "level" or r["level"] not in ADMM_LEVELS:
            P.add(where, "rows with an unknown record or level", line)
            continue
        st = r["status"]
        if not (st == "reached" or (st == "capped" and k == cap)):
            P.add(where, "level rows with an invalid status or k", "%s: status %s" % (line, st))
            continue
        if not status_agrees(r["level"], st, meas, target):
            P.add(where, "level rows whose measures contradict their status", "%s: status %s, %s" % (line, st, mtext))
            continue
        key = (rho, r["role"], r["level"])
        if key in u.levels:
            P.add(where, "duplicate level rows", line)
            continue
        texts[key] = mtext
        u.levels[key] = {"status": st, "k": k, "r": meas[0], "s": meas[1], "eps_pri": meas[2], "eps_dual": meas[3],
                         "stationarity": meas[4], "objective": meas[5]}
    for rho in ADMM_RHOS:
        for role in PRIMARY:
            lv = [u.levels.get((rho, role, l)) for l in ADMM_LEVELS]
            if None in lv:
                P.add(where, "level records absent", "rho=%s %s" % (rho, role))
                continue
            ran = max(x["k"] for x in lv) if all(x["status"] == "reached" for x in lv) else cap
            want = {k for k in ADMM_CHECKPOINTS if k <= ran} | {ran} | {x["k"] for x in lv if x["status"] == "reached"}
            got = traj.get((rho, role), {})
            if set(got) != want:
                P.add(where, "trajectory rows other than the checkpoints, the last iteration and the reached iterations",
                      "rho=%s %s: %s, expected %s" % (rho, role, sorted(got), sorted(want)))
                continue
            for l, x in zip(ADMM_LEVELS, lv):
                if got.get(x["k"]) != texts[(rho, role, l)]:
                    P.add(where, "level records that differ from the trajectory row at their k", "rho=%s %s %s" % (rho, role, l))
    return u


def parse_reference(path, man, pins, P):
    """A reference file: identity, the instance and its digests, rho, tol and cap of the registered reference, scale,
    lambda and lambda_max from the manifest, finite nonnegative measures, convergence (stationarity <= tol * scale and
    an iteration count on the check cadence of every 10 iterations or the cap), and the z file of p Float64 values with
    the recorded SHA-256."""
    where = os.path.basename(path)
    try:
        kv = read_kv(path)
    except Skipped as e:
        P.add(where, str(e))
        return None
    if kv.get("format") != FORMATS["reference"]:
        P.add(where, "not a %s file" % FORMATS["reference"])
        return None
    check_identity(kv, man, pins, where, P)
    case = kv.get("case", "?")
    if kv.get("julia") != REFERENCE_JULIA:
        P.add(where, "generated on Julia %s, not the pinned %s" % (kv.get("julia"), REFERENCE_JULIA))
    c = man.cases.get(case) if man is not None else None
    try:
        it, cap = to_int(kv.get("iterations", "")), to_int(kv.get("cap", ""))
        rho, tol = to_float(kv.get("rho", "")), to_float(kv.get("tol", ""))
        stat, obj, scale = to_float(kv.get("stationarity", "")), to_float(kv.get("objective", "")), to_float(kv.get("scale", ""))
        lam, lmax = to_float(kv.get("lambda", "")), to_float(kv.get("lambda_max", ""))
    except ValueError as e:
        P.add(where, "unparseable field", str(e))
        return None
    if man is not None:
        if c is None or not c["lasso"]:
            P.add(where, "case %s is not a lasso instance of the manifest" % case)
            return None
        x, y = c["arrays"]["X"]["sha256"], c["arrays"]["y"]["sha256"]
        if kv.get("x_sha256") != x or kv.get("y_sha256") != y:
            P.add(where, "X or y digest differs from the manifest")
        if where != "reference-%s-%s-%s.tsv" % (case_file(case), x[:12], y[:12]):
            P.add(where, "file name does not match the instance and its digests")
        want_scale = max(1.0, c["lambda_max"])
        if scale != want_scale or lmax != c["lambda_max"] or abs(lam - 0.1 * c["lambda_max"]) > 1e-15 * abs(lam):
            P.add(where, "scale, lambda or lambda_max differ from the manifest",
                  "scale %r (manifest %r), lambda %r, lambda_max %r (manifest %r)" % (scale, want_scale, lam, lmax, c["lambda_max"]))
        scale = want_scale
    if (rho, tol, cap) != (1.0, REFERENCE_TOL, REFERENCE_CAP):
        P.add(where, "rho, tol or cap differ from the registered 1, %g, %d" % (REFERENCE_TOL, REFERENCE_CAP))
    if not (finite(stat) and stat >= 0 and finite(obj) and obj >= 0 and 1 <= it <= cap):
        P.add(where, "measures not finite and nonnegative, or iterations out of 1..cap")
    elif kv.get("converged") != "true":
        P.add(where, "the reference did not converge", "%d iterations, stationarity %g" % (it, stat))
    elif not (stat <= tol * scale and (it % 10 == 0 or it == cap)):
        P.add(where, "a converged reference with its stationarity above tol * scale or iterations off the check cadence",
              "%d iterations, stationarity %g, target %g" % (it, stat, tol * scale))
    zfile = os.path.join(os.path.dirname(path), kv.get("z_file", ""))
    if not kv.get("z_file") or not os.path.isfile(zfile):
        P.add(where, "the z file is missing")
    else:
        data = read_bytes(zfile)
        if DIGESTS[os.path.abspath(zfile)] != kv.get("z_sha256"):
            P.add(where, "the SHA-256 of the z file differs")
        if c is not None and len(data) != 8 * c["p"]:
            P.add(where, "the z file has %d bytes, %d Float64 values need %d" % (len(data), c["p"], 8 * c["p"]))
    kv["_file"], kv["_objective"] = where, obj
    return kv


class Outputs:
    def __init__(self, dirpath, header, rows, failures, sha):
        self.dir, self.header, self.rows, self.failures, self.sha256 = dirpath, header, rows, failures, sha
        self.version = header.get("julia", "?")


def parse_outputs(ipath, man, pins, verdict, P):
    d = os.path.dirname(ipath)
    where = os.path.join(os.path.basename(d), "outputs-index.tsv")
    fpath = os.path.join(d, "outputs-failures.tsv")
    fwhere = os.path.join(os.path.basename(d), "outputs-failures.tsv")
    try:
        header, cols, rows = read_tsv(ipath)
    except Skipped as e:
        P.add(where, str(e))
        return None
    try:
        fheader, fcols, frows = read_tsv(fpath)
    except Skipped as e:
        P.add(fwhere, str(e))
        return None
    if header.get("format") != FORMATS["outputs"] or cols != OUTPUT_COLUMNS:
        P.add(where, "not a %s file" % FORMATS["outputs"])
        return None
    if fheader.get("format") != FORMATS["failures"] or fcols != FAILURE_COLUMNS:
        P.add(fwhere, "not a %s file" % FORMATS["failures"])
        return None
    check_identity(header, man, pins, where, P)
    for key in ("source_sha", "harness_sha", "inputs_index_sha256", "julia"):
        if fheader.get(key) != header.get(key):
            P.add(fwhere, "%s differs from outputs-index.tsv" % key)
    if verdict and header.get("julia") not in VERSIONS:
        P.add(where, "julia %s" % header.get("julia"))
    index = {}
    for f in rows:
        r = dict(zip(cols, f))
        line = "%s %s" % (r["case"], r["role"])
        c = man.cases.get(r["case"]) if man is not None else None
        if man is not None and c is None:
            P.add(where, "rows of a case outside the manifest", line)
            continue
        if r["role"] not in PRIMARY:
            P.add(where, "rows of a role that is not primary", line)
            continue
        if (r["case"], r["role"]) in index:
            P.add(where, "duplicate rows", line)
            continue
        try:
            r["nfail"] = to_int(r["nfail"])
            nv = to_int(r["nv"])
            steps = tuple(to_float(x) for x in r["acc_gammas"].split(","))
        except ValueError as e:
            P.add(where, "unparseable field", "%s (%s)" % (line, e))
            continue
        if r["nfail"] < 0 or r["copy_identical"] not in ("true", "false"):
            P.add(where, "nfail or copy_identical out of range", line)
        if c is not None and (r["eltype"] != c["eltype"] or nv != c["nv"] or steps != tuple(c["acc_gammas"]) or
                              r["shape"] != "%dx%d" % (c["p"], c["nv"] * len(c["acc_gammas"]))):
            P.add(where, "element type, inputs, steps or shape differ from the manifest", line)
        index[(r["case"], r["role"])] = r
    failures, counts = {}, collections.Counter()
    for f in frows:
        r = dict(zip(fcols, f))
        line = "%s %s %s %s" % (r["case"], r["role"], r["gamma_index"], r["vindex"])
        ir = index.get((r["case"], r["role"]))
        if ir is None:
            P.add(fwhere, "failure rows without a row in outputs-index.tsv", line)
            continue
        try:
            gi, k = to_int(r["gamma_index"]), to_int(r["vindex"])
        except ValueError:
            P.add(fwhere, "unparseable field", line)
            continue
        c = man.cases.get(r["case"]) if man is not None else None
        if c is not None and not (1 <= gi <= len(c["acc_gammas"]) and 1 <= k <= c["nv"]):
            P.add(fwhere, "failure keys out of range", line)
            continue
        key = (r["case"], r["role"], gi, k)
        if key in failures:
            P.add(fwhere, "duplicate failure keys", line)
            continue
        failures[key] = r["message"]
        counts[(r["case"], r["role"])] += 1
    for key, r in index.items():
        if r["nfail"] != counts[key]:
            P.add(where, "nfail differs from the number of failure rows", "%s %s: %d against %d" % (key + (r["nfail"], counts[key])))
    return Outputs(d, header, index, failures, DIGESTS[os.path.abspath(ipath)])


class Accuracy:
    def __init__(self, path, header):
        self.path, self.name, self.header = path, os.path.basename(path), header
        self.version = header.get("julia", "?")
        self.rows = {}


def parse_accuracy(path, man, pins, verdict, P):
    where = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
    try:
        header, cols, rows = read_tsv(path)
    except Skipped as e:
        P.add(where, str(e))
        return None
    if header.get("format") != FORMATS["accuracy"] or cols != ACCURACY_COLUMNS:
        P.add(where, "not a %s file" % FORMATS["accuracy"], "format %s" % header.get("format", "<absent>"))
        return None
    check_identity(header, man, pins, where, P)
    if verdict:
        for key in ("julia", "evaluator_julia"):
            if header.get(key) not in VERSIONS:
                P.add(where, "%s %s" % (key, header.get(key, "<absent>")))
        if header.get("filter") != "":
            P.add(where, "evaluated with the filter %r; the campaign needs every case" % header.get("filter"))
    if header.get("eta_precision_bits") != "256" or header.get("forward_precision_bits") != "128":
        P.add(where, "evaluator precisions other than 256 and 128 bits")
    try:
        fmax = to_int(header.get("forward_max_n", ""))
    except ValueError:
        P.add(where, "unparseable forward_max_n")
        return None
    a = Accuracy(path, header)
    for f in rows:
        r = dict(zip(cols, f))
        line = "%s %s %s %s" % (r["case"], r["role"], r["gamma_index"], r["vindex"])
        c = man.cases.get(r["case"]) if man is not None else None
        if man is not None and c is None:
            P.add(where, "rows of a case outside the manifest", line)
            continue
        if r["role"] not in PRIMARY:
            P.add(where, "rows of a role that is not primary", line)
            continue
        try:
            gi, k, g = to_int(r["gamma_index"]), to_int(r["vindex"]), to_float(r["gamma"])
            eta, fe = to_float(r["eta"]), to_float(r["forward_error"])
        except ValueError as e:
            P.add(where, "unparseable field", "%s (%s)" % (line, e))
            continue
        st, ps = r["status"], r["pass"]
        if ps not in ("true", "false") or not (st in ("ok", "nonfinite") or st.startswith("error: ")):
            P.add(where, "pass or status out of range", line)
            continue
        if (ps == "true" and (st != "ok" or not finite(eta))) or (st != "ok" and (ps == "true" or eta != math.inf)):
            P.add(where, "pass, status and eta disagree", line)
            continue
        if st == "ok" and (r["eltype"] not in EPS or not (finite(eta) or eta == math.inf) or eta < 0):
            P.add(where, "eta of a status-ok output negative or not a number, or an unknown element type", line)
            continue
        if st == "ok":
            # the pass flag was decided in BigFloat; the written Float64 eta decides it too unless it lands on the threshold
            thr = ETA_FACTOR * EPS[r["eltype"]]
            if (eta < thr and ps != "true") or (eta > thr and ps != "false"):
                P.add(where, "pass flag contradicts eta and the threshold 1e3 eps(T)", "%s: eta %r, pass %s" % (line, eta, ps))
                continue
        if c is not None:
            if c["n"] > fmax:
                fe_ok = fe != fe
            else:
                fe_ok = (finite(fe) and fe >= 0) if st == "ok" and finite(eta) else (fe == math.inf or (finite(fe) and fe >= 0))
            if not fe_ok or r["copy_identical"] not in ("true", "false"):
                P.add(where, "forward_error or copy_identical out of its domain", "%s: %r" % (line, fe))
                continue
        if c is not None and (r["eltype"] != c["eltype"] or r["kind"] != c["kind"] or not 1 <= gi <= len(c["acc_gammas"])
                              or g != c["acc_gammas"][gi - 1] or not 1 <= k <= c["nv"]):
            P.add(where, "element type, kind, step or input index differ from the manifest", line)
            continue
        key = (r["case"], r["role"], gi, k)
        if key in a.rows:
            P.add(where, "duplicate rows", line)
            continue
        a.rows[key] = {"eta": eta, "fe": fe, "pass": ps == "true", "status": st, "copy": r["copy_identical"],
                       "eltype": r["eltype"], "kind": r["kind"], "gamma": g, "vindex": k}
    return a


# ---------------------------------------------------------------------------------------------------------------------
# The campaign: loading in classes and the inventory (class 3)
# ---------------------------------------------------------------------------------------------------------------------

def plan_rule(tmax, cycle, min_calls, target):
    """(calls per batch, batches) of plan_batches in harness.jl for the slowest calibrated time per call tmax."""
    if tmax >= 0.1:
        return cycle, 1
    cap = cycle * max(1, BATCH_CALL_CAP // cycle)
    calls = max(min_calls, math.ceil(target / tmax) if tmax > 0 else min_calls)
    return min(max(cycle * -(-calls // cycle), cycle), cap), BATCHES


class Proc:
    """One timing process (Julia version, role order, process id, BLAS threads), possibly split over several files."""

    def __init__(self, key):
        self.key, self.files = key, []
        self.rows, self.plans, self.pairs, self.loads, self.workingset, self.untimed_ids = {}, {}, {}, {}, {}, {}
        self.group_files = collections.defaultdict(list)
        self._data = None

    @property
    def order(self):
        return self.key[1]

    def data(self):
        """cell (case, measurement, parameter) -> role -> round -> value, built once from the rows."""
        if self._data is None:
            d = collections.defaultdict(lambda: collections.defaultdict(dict))
            for (case, meas, cp, rnd, role), v in self.rows.items():
                d[(case, meas, cp)][role][rnd] = v[1]
            self._data = d
        return self._data


class Campaign:
    def __init__(self, root, man, pins, verdict):
        self.root, self.man, self.pins, self.verdict = root, man, pins, verdict
        self.procs, self.untimed, self.refs, self.outputs, self.acc_files = {}, {}, {}, {}, {}
        self.acc, self.notes = {}, []

    def glob(self, pattern):
        return sorted(glob.glob(os.path.join(self.root, "**", pattern), recursive=True))

    def load_files(self, P):
        """Class 2: parse every file. Returns the parsed objects (a file with a problem is marked in P)."""
        man, pins, verdict = self.man, self.pins, self.verdict
        parsed = {"timings": [], "untimed": [], "references": [], "outputs": [], "accuracy": []}
        for path in self.glob("inputs-index.tsv"):
            if man is not None and os.path.abspath(path) != os.path.abspath(man.path) and sha256_file(path) != man.sha256:
                P.add(os.path.relpath(path, self.root), "another manifest in the campaign directory")
        for path in self.glob("timings-*.tsv"):
            parsed["timings"].append((path, parse_timing(path, man, pins, verdict, P)))
        for path in self.glob("admm-untimed-*.tsv"):
            parsed["untimed"].append((path, parse_untimed(path, man, pins, verdict, P)))
        for path in self.glob("reference-*.tsv"):
            parsed["references"].append((path, parse_reference(path, man, pins, P)))
        for path in self.glob("outputs-index.tsv"):
            parsed["outputs"].append((path, parse_outputs(path, man, pins, verdict, P)))
        for path in self.glob("outputs-failures.tsv"):
            if not os.path.isfile(os.path.join(os.path.dirname(path), "outputs-index.tsv")):
                P.add(os.path.relpath(path, self.root), "failures index without an outputs index in its directory")
        for path in self.glob("accuracy-*.tsv"):
            parsed["accuracy"].append((path, parse_accuracy(path, man, pins, verdict, P)))
        return parsed

    def accept(self, parsed, bad):
        """Builds the campaign from the parsed files, leaving out those named in `bad` (tables-only) and noting it."""
        def ok(obj, where):
            if obj is None or where in bad:
                self.notes.append("left out: %s" % where)
                return False
            return True
        for path, tf in parsed["timings"]:
            if not ok(tf, os.path.basename(path)):
                continue
            h = tf.header
            key = (h.get("julia", "?"), h.get("order", "?"), h.get("proc", "?"), h.get("blas_threads", "?"))
            p = self.procs.setdefault(key, Proc(key))
            p.files.append(tf)
            for g in tf.groups:
                p.group_files[g].append(tf.name)
            for k, v in tf.rows.items():
                if k in p.rows:
                    p.dup = getattr(p, "dup", []) + [(tf.name, k)]
                p.rows[k] = v
            for attr in ("plans", "pairs", "loads", "untimed_ids", "workingset"):
                mine = getattr(p, attr)
                for k, v in getattr(tf, attr).items():
                    if k in mine:
                        p.dup_context = getattr(p, "dup_context", []) + [(tf.name, (attr, k))]
                    mine[k] = v
        for path, u in parsed["untimed"]:
            if ok(u, os.path.basename(path)):
                self.untimed.setdefault((u.version, u.case), []).append(u)
        for path, kv in parsed["references"]:
            if ok(kv, os.path.basename(path)):
                self.refs.setdefault(kv["case"], []).append(kv)
        for path, o in parsed["outputs"]:
            d = os.path.basename(os.path.dirname(path))
            if os.path.join(d, "outputs-failures.tsv") in bad:
                bad = bad | {os.path.join(d, "outputs-index.tsv")}
            if ok(o, os.path.join(d, "outputs-index.tsv")):
                self.outputs.setdefault(o.version, []).append(o)
        for path, a in parsed["accuracy"]:
            if ok(a, os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))):
                self.acc_files.setdefault(a.version, []).append(a)
        for v, files in self.acc_files.items():
            for a in files:
                for (case, role, gi, k), r in a.rows.items():
                    c = self.man.cases.get(case) if self.man is not None else None
                    g = c["acc_gammas"][gi - 1] if c is not None else r["gamma"]
                    self.acc.setdefault((v, case, role), {})[gk(g, k)] = r

    # ---- class 3
    def inventory(self, P):
        man = self.man
        lasso = [c for c in man.cases.values() if c["kind"] == "main" and c["lasso"]]
        timed = [c for c in man.cases.values() if c["kind"] == "main" and c["timed"]]
        expected = {(v, o, p, "1") for v in VERSIONS for o in ORDERS for p in PROCS}
        for key in sorted(self.procs):
            if key not in expected:
                P.add("process %s/%s/%s (BLAS threads %s)" % key, "not part of the registered campaign",
                      ", ".join(f.name for f in self.procs[key].files))
        for key in sorted(expected):
            where = "process %s/%s/%s" % key[:3]
            p = self.procs.get(key)
            if p is None:
                P.add(where, "absent (no timing file)")
                continue
            for g in TIMED_GROUPS:
                n = len(p.group_files.get(g, []))
                if n != 1:
                    P.add(where, "group %s is in %d files (one is required)" % (g, n), ", ".join(p.group_files.get(g, [])))
            for name, k in getattr(p, "dup", []):
                P.add(where, "rows repeated across its files", "%s %s" % (name, k))
            for name, k in getattr(p, "dup_context", []):
                P.add(where, "context rows repeated across its files", "%s %s" % (name, k))
            self.check_process_rows(p, timed, where, P)
            self.check_process_admm(p, lasso, where, P)
        for v in VERSIONS:
            for c in lasso:
                n = len(self.untimed.get((v, c["id"]), []))
                if n != 1:
                    P.add("untimed files", "%d files for Julia %s and %s (one is required)" % (n, v, c["id"]))
        for (v, case), us in self.untimed.items():
            if v not in VERSIONS:
                P.add("untimed files", "file of Julia %s" % v, us[0].name)
        for c in lasso:
            n = len(self.refs.get(c["id"], []))
            if n != 1:
                P.add("references", "%d reference files for %s (one is required)" % (n, c["id"]))
        for v in VERSIONS:
            outs, accs = self.outputs.get(v, []), self.acc_files.get(v, [])
            if len(outs) != 1:
                P.add("outputs", "%d outputs indices for Julia %s (one is required)" % (len(outs), v))
            if len(accs) != 1:
                P.add("accuracy", "%d accuracy files for Julia %s (one is required)" % (len(accs), v))
            if len(outs) == 1:
                self.check_outputs_coverage(v, outs[0], P)
            if len(outs) == 1 and len(accs) == 1:
                self.check_accuracy_coverage(v, outs[0], accs[0], P)
        for v in set(self.outputs) | set(self.acc_files):
            if v not in VERSIONS:
                P.add("outputs and accuracy", "files of Julia %s" % v)
        self.check_hardware(P)

    def check_hardware(self, P):
        """One CPU kind across the primary timing files (populated vendor and model), and one BLAS configuration and
        OpenBLAS build per Julia version."""
        files = [f for k, p in sorted(self.procs.items()) if k[3] == "1" for f in p.files]
        for key in ("cpu_vendor", "cpu_model"):
            vals = collections.Counter(f.header.get(key, "") for f in files)
            if any(v in ("", "unknown") for v in vals) or len(vals) > 1:
                P.add("timing files", "%s missing or not the same in every primary timing file" % key, ", ".join(
                    "%r in %d" % (v, n) for v, n in sorted(vals.items())))
        for v in VERSIONS:
            for key in ("blas_config", "openblas"):
                vals = collections.Counter(f.header.get(key, "") for f in files if f.header.get("julia") == v)
                if len(vals) > 1 or "" in vals:
                    P.add("timing files", "%s missing or not the same in the primary timing files of Julia %s" % (key, v),
                          ", ".join("%r in %d" % (x, n) for x, n in sorted(vals.items())))

    def check_process_rows(self, p, timed, where, P):
        names = ROLE_ORDER[p.order]
        prim = [r for r in names if "'" not in r]
        exp, plans, pairs = {}, set(), set()
        for c in timed:
            cid = c["id"]
            for rnd in range(1, ROUNDS + 1):
                for pos, idx in enumerate(round_indices(len(names), rnd), 1):
                    role = names[idx]
                    exp[(cid, "first_use", 1.0, rnd, role)] = pos
                    for g in c["gammas"]:
                        for kind in ("rand", "admm"):
                            exp[(cid, "warm", (g, kind), rnd, role)] = pos
                    for s in SCHEDULES:
                        exp[(cid, "penalty", s, rnd, role)] = pos
                for pos, idx in enumerate(round_indices(len(prim), rnd), 1):
                    for part in PARTS[prim[idx]]:
                        exp[(cid, "first_use_part", part, rnd, prim[idx])] = pos
            for pos, role in enumerate(names, 1):
                for call in ("prox", "refactor_prox"):
                    exp[(cid, "alloc", call, 1, role)] = pos
            plans.add((cid, "warm"))
            plan = p.plans.get((cid, "warm"))
            if plan is None or not (plan[0]["cap"] == BATCH_CALL_CAP and finite(plan[0]["tmax"]) and plan[0]["tmax"] >= 0 and
                                    (plan[0]["calls"], plan[0]["batches"]) == plan_rule(plan[0]["tmax"], NV_RAND, 30, 0.02)):
                P.add(where, "batch_plan of warm absent or not the registered rule for its tmax", "%s %s" % (cid, plan))
            for s in SCHEDULES:
                steps = SCHEDULES[s][c["eltype"]]
                want = tuple(executed_pairs(steps))
                if p.pairs.get((cid, s)) != want:
                    P.add(where, "penalty_pairs absent or different from the registered schedule",
                          "%s %s: %s, expected %s" % (cid, s, p.pairs.get((cid, s)), want))
                pairs.add((cid, s))
                plans.add((cid, "penalty:" + s))
                plan = p.plans.get((cid, "penalty:" + s))
                cyc = len(want)
                if plan is None or not (plan[0]["cap"] == BATCH_CALL_CAP and finite(plan[0]["tmax"]) and plan[0]["tmax"] >= 0 and
                                        (plan[0]["calls"], plan[0]["batches"]) == plan_rule(plan[0]["tmax"], cyc, 1, 0.05)):
                    P.add(where, "batch_plan of penalty absent or not the registered rule for its tmax", "%s %s %s" % (cid, s, plan))
        for k in sorted(set(p.plans) - plans):
            P.add(where, "batch_plan rows outside the registered design", str(k))
        for k in sorted(set(p.pairs) - pairs):
            P.add(where, "penalty_pairs rows outside the registered design", str(k))
        for key, pos in exp.items():
            v = p.rows.get(key)
            if v is None:
                P.add(where, "rows absent", "%s %s %s round %d %s" % key)
                continue
            if v[0] != pos:
                P.add(where, "role order of the round differs from the order of the process",
                      "%s %s %s round %d %s at position %d, expected %d" % (key + (v[0], pos)))
            meas = key[1]
            if meas == "alloc":
                if not (finite(v[3]) and v[3] >= 0 and finite(v[4]) and v[4] >= 0):
                    P.add(where, "alloc rows with bytes or allocations out of range", "%s %s %s round %d %s" % key)
            elif not (finite(v[1]) and (v[1] > 0 or (meas == "first_use_part" and v[1] >= 0))):
                P.add(where, "times that are not finite and positive", "%s %s %s round %d %s" % key)
        for key in p.rows:
            if GROUP_OF.get(key[1]) != "admm" and key not in exp:
                P.add(where, "rows outside the registered design", "%s %s %s round %d %s" % key)

    def check_process_admm(self, p, lasso, where, P):
        names = ROLE_ORDER[p.order]
        version = p.key[0]
        exp, unchecked = set(), set()
        for c in lasso:
            cid = c["id"]
            us = self.untimed.get((version, cid), [])
            if len(us) != 1:
                P.add(where, "ADMM rows cannot be checked without one untimed file", "%s (Julia %s)" % (cid, version))
                unchecked.add(cid)
                continue
            u = us[0]
            ident = p.untimed_ids.get(cid)
            if ident != (u.name, u.sha):
                P.add(where, "the untimed file recorded for the timed runs is not the untimed file of the campaign",
                      "%s: recorded %s, campaign %s" % (cid, ident, (u.name, u.sha)))
            for rho in ADMM_RHOS:
                for lvl in ADMM_LEVELS:
                    cp = (rho, lvl)
                    for role in PRIMARY:
                        rec = u.levels.get((rho, role, lvl))
                        if rec is None:
                            continue
                        for meas, want in (("admm_iters", float(rec["k"])), ("admm_reached", 1.0 if rec["status"] == "reached" else 0.0)):
                            key = (cid, meas, cp, 0, role)
                            exp.add(key)
                            v = p.rows.get(key)
                            if v is None or v[1] != want or v[0] != 0:
                                P.add(where, "%s absent, not at position 0 or different from the untimed record" % meas,
                                      "%s rho=%s %s %s: %s, untimed %s" % (cid, rho, lvl, role, None if v is None else v[1], want))
                    active = [r for r in names if u.levels.get((rho, family(r), lvl), {}).get("status") == "reached"]
                    target = ADMM_TIGHT * max(1.0, c["lambda_max"])
                    for rep in range(1, REPEATS + 1):
                        for pos, idx in enumerate(round_indices(len(active), rep), 1):
                            role = active[idx]
                            run = {m: p.rows.get((cid, m, cp, rep, role)) for m in ADMM_TIMED}
                            if all(x is not None and finite(x[1]) and x[1] >= 0 for x in run.values()):
                                val = {m: x[1] for m, x in run.items()}
                                hi = 1 + SLACK
                                if lvl == "tight" and val["admm_valid"] == 1.0 and val["admm_stationarity"] > target * hi:
                                    P.add(where, "tight runs marked valid with a stationarity above the target",
                                          "%s rho=%s rep %d %s: %g > %g" % (cid, rho, rep, role, val["admm_stationarity"], target))
                                if lvl == "boyd" and val["admm_valid"] == 1.0 and not (
                                        val["admm_r"] <= val["admm_eps_pri"] * hi and val["admm_s"] <= val["admm_eps_dual"] * hi):
                                    P.add(where, "Boyd runs marked valid whose r and s miss their tolerances",
                                          "%s rho=%s rep %d %s" % (cid, rho, rep, role))
                                if abs(val["admm_total"] - val["admm_factor"] - val["admm_iter"]) > 1e-8 * val["admm_total"]:
                                    P.add(where, "ADMM total time differs from factorization plus iterations",
                                          "%s rho=%s %s rep %d %s" % (cid, rho, lvl, rep, role))
                            for meas in ADMM_TIMED:
                                key = (cid, meas, cp, rep, role)
                                exp.add(key)
                                v = p.rows.get(key)
                                if v is None:
                                    P.add(where, "timed ADMM rows absent", "%s %s rho=%s %s rep %d %s" % (cid, meas, rho, lvl, rep, role))
                                    continue
                                if v[0] != pos:
                                    P.add(where, "role order of the ADMM repeat differs from the order of the process",
                                          "%s rho=%s %s rep %d %s at position %d, expected %d" % (cid, rho, lvl, rep, role, v[0], pos))
                                if meas == "admm_valid" and v[1] != 1.0:
                                    P.add(where, "timed ADMM runs whose end point misses the target of the level",
                                          "%s rho=%s %s rep %d %s" % (cid, rho, lvl, rep, role))
                                elif not finite(v[1]) or v[1] < 0 or (meas in ("admm_total", "admm_factor", "admm_iter") and v[1] <= 0):
                                    P.add(where, "timed ADMM values out of range", "%s %s rho=%s %s rep %d %s" % (cid, meas, rho, lvl, rep, role))
        for key in p.rows:
            if GROUP_OF.get(key[1]) == "admm" and key not in exp and key[0] not in unchecked:
                P.add(where, "ADMM rows outside the registered design or for a capped family", "%s %s %s round %d %s" % key)

    def check_outputs_coverage(self, v, o, P):
        where = "outputs of Julia %s" % v
        for cid in self.man.cases:
            for role in PRIMARY:
                if (cid, role) not in o.rows:
                    P.add(where, "outputs-index.tsv lacks rows", "%s %s" % (cid, role))

    def check_accuracy_coverage(self, v, o, a, P):
        where = a.name + " (Julia %s)" % v
        if a.header.get("outputs_index_sha256") != o.sha256:
            P.add(where, "evaluated from another outputs index than that of the campaign")
        for cid, c in self.man.cases.items():
            for role in PRIMARY:
                for gi in range(1, len(c["acc_gammas"]) + 1):
                    for k in range(1, c["nv"] + 1):
                        r = a.rows.get((cid, role, gi, k))
                        if r is None:
                            P.add(where, "accuracy rows absent", "%s %s step %d input %d" % (cid, role, gi, k))
                            continue
                        failed = (cid, role, gi, k) in o.failures
                        if failed != r["status"].startswith("error: "):
                            P.add(where, "status disagrees with the failures index", "%s %s step %d input %d" % (cid, role, gi, k))
                        ir = o.rows.get((cid, role))
                        if ir is not None and r["copy"] != ir["copy_identical"]:
                            P.add(where, "copy_identical disagrees with the outputs index", "%s %s step %d input %d" % (cid, role, gi, k))
        extra = [k for k in a.rows if k[0] not in self.man.cases]
        if extra:
            P.add(where, "rows outside the manifest", str(extra[0]))


# ---------------------------------------------------------------------------------------------------------------------
# Timing judgement and criteria
# ---------------------------------------------------------------------------------------------------------------------

def ratios(d, num_role, den_role):
    a, b = d.get(num_role, {}), d.get(den_role, {})
    return {r: a[r] / b[r] for r in a if r in b and finite(a[r]) and finite(b[r]) and b[r] > 0}


def per_process(camp, version, cell, variant):
    """Per primary process of a version: (order, proc, ratios V/S, S'/S, V'/V) for the rows that are present."""
    out = []
    for o in ORDERS:
        for pid in PROCS:
            p = camp.procs.get((version, o, pid, "1"))
            if p is None:
                continue
            d = p.data().get(cell, {})
            out.append((o, pid, ratios(d, variant, "S"), ratios(d, "S'", "S"), ratios(d, variant + "'", variant)))
    return out


def judge(camp, version, cell, variant):
    """The timing judgement of one cell for one variant over the processes of a version (all six in verdict mode)."""
    per, meds, aa, allr = [], [], [], []
    for o, pid, rv, rs, rc in per_process(camp, version, cell, variant):
        if rv:
            vals = list(rv.values())
            per.append((o, pid, median(vals), min(vals), max(vals)))
            meds.append(per[-1][2])
            allr.extend(vals)
        for x in (rs, rc):
            if x:
                aa.append(abs(median(x.values()) - 1))
    spread = max(aa) if aa else math.nan
    res = {"per": per, "meds": meds, "spread": spread, "n": len(meds), "center": median(meds),
           "lo_med": min(meds) if meds else math.nan, "hi_med": max(meds) if meds else math.nan,
           "lo": min(allr) if allr else math.nan, "hi": max(allr) if allr else math.nan}
    if meds and all(m < FASTER for m in meds) and 1 - max(meds) > spread:
        res["timing"] = "faster"
    elif meds and all(m > SLOWER for m in meds) and min(meds) - 1 > spread:
        res["timing"] = "slower"
    else:
        res["timing"] = "inconclusive"
    return res


def expected_cells(man):
    """The timed cells: (case, measurement, canonical parameter) for every timed main case."""
    cells = []
    for c in man.cases.values():
        if c["kind"] != "main" or not c["timed"]:
            continue
        cells.append((c["id"], "first_use", 1.0))
        for g in c["gammas"]:
            for kind in ("rand", "admm"):
                cells.append((c["id"], "warm", (g, kind)))
        for s in SCHEDULES:
            cells.append((c["id"], "penalty", s))
        if c["lasso"]:
            for rho in ADMM_RHOS:
                for lvl in ADMM_LEVELS:
                    cells.append((c["id"], "admm_total", (rho, lvl)))
    return cells


def needed_keys(case, meas, cp):
    """The (step, input) keys of the accuracy evidence of a cell (for ADMM, the stored ADMM vectors at the step 1/rho)."""
    if meas == "first_use":
        return [gk(1.0, k) for k in range(1, NV_RAND + 1)]
    if meas == "warm":
        ks = range(1, NV_RAND + 1) if cp[1] == "rand" else range(NV_RAND + 1, NV_MAIN + 1)
        return [gk(cp[0], k) for k in ks]
    if meas == "penalty":
        return [gk(g, k) for g, k in executed_pairs(SCHEDULES[cp][case["eltype"]])]
    if meas == "admm_total":
        return [gk(1.0 / cp[0], k) for k in range(NV_RAND + 1, NV_MAIN + 1)]
    return []


class Verdict:
    """The criteria on a campaign that passed the preflight: every value is True or False."""

    def __init__(self, camp):
        self.camp, self.man = camp, camp.man
        self.cells = expected_cells(self.man)
        self._cell = {}

    def acc_pass(self, version, cid, role, need):
        rows = self.camp.acc[(version, cid, role)]
        return all(rows[k]["pass"] for k in need)

    def status(self, version, cid, rho, role, lvl):
        return self.camp.untimed[(version, cid)][0].levels[(rho, role, lvl)]["status"]

    def cell(self, version, cell, variant):
        key = (version, cell, variant)
        if key in self._cell:
            return self._cell[key]
        cid, meas, cp = cell
        j = judge(self.camp, version, cell, variant)
        need = needed_keys(self.man.cases[cid], meas, cp)
        okV, okS = self.acc_pass(version, cid, variant, need), self.acc_pass(version, cid, "S", need)
        capped = []
        if meas == "admm_total":
            capped = [r for r in ("S", variant) if self.status(version, cid, cp[0], r, cp[1]) == "capped"]
        if not okV or not okS:
            label = "excluded"
        elif capped:
            label = "capped"
        else:
            label = j["timing"]
        rec = dict(j, label=label, capped=capped, accuracy="pass" if okV and okS else
                   ("%s fails" % (variant if not okV else "S")))
        self._cell[key] = rec
        return rec

    def not_slower(self, variant, cells):
        recs = [self.cell(v, c, variant) for v in VERSIONS for c in cells]
        cnt = collections.Counter(r["label"] for r in recs)
        info = "%d cells: %s" % (len(recs), ", ".join("%d %s" % (n, k) for k, n in sorted(cnt.items())))
        return cnt["slower"] == 0, info

    def all_faster(self, variant, cells):
        recs = [self.cell(v, c, variant) for v in VERSIONS for c in cells]
        cnt = collections.Counter(r["label"] for r in recs)
        info = "%d cells: %s" % (len(recs), ", ".join("%d %s" % (n, k) for k, n in sorted(cnt.items())))
        return cnt["faster"] == len(recs), info

    def accuracy_everywhere(self, variant):
        """A1: V passes at every (case, step, input) of the manifest where S passes, in both Julia versions."""
        fails = []
        for v in VERSIONS:
            for cid in self.man.cases:
                rs, rv = self.camp.acc[(v, cid, "S")], self.camp.acc[(v, cid, variant)]
                for k, s in rs.items():
                    if s["pass"] and not rv[k]["pass"]:
                        fails.append((v, cid) + k)
        info = "%d keys where %s fails and S passes" % (len(fails), variant)
        return not fails, info + (" (first: %s)" % (fails[0],) if fails else "")

    def instance_accuracy(self, variant):
        """B1: V passes at the step 1 and all 8 inputs of the instance of section 11.1 in both Julia versions."""
        need = [gk(1.0, k) for k in range(1, NV_MAIN + 1)]
        st = [self.acc_pass(v, INSTANCE_11_1, variant, need) for v in VERSIONS]
        return all(st), "passes per Julia version: %s" % ", ".join(str(x) for x in st)

    def reference(self, cid):
        return self.camp.refs[cid][0]["_objective"]

    def objectives(self, version, cell, role):
        vals = []
        for o in ORDERS:
            for pid in PROCS:
                d = self.camp.procs[(version, o, pid, "1")].data().get((cell[0], "admm_objective", cell[2]), {})
                vals.extend(d.get(role, {}).values())
        return vals

    def gap_criterion(self, variant):
        """A3, second part: at rho = 1 and both levels on the section 11.1 instance, the largest timed objective gap of V
        to its reference is within that of S plus 1e-8 max(1, |p*|); False when S or V is capped at a level (no timed
        objective of the capped role; the speed part of A3 is then False too)."""
        pstar = self.reference(INSTANCE_11_1)
        ok, detail = True, []
        for v in VERSIONS:
            for lvl in ADMM_LEVELS:
                cell = (INSTANCE_11_1, "admm_total", (1.0, lvl))
                capped = [r for r in ("S", variant) if self.status(v, INSTANCE_11_1, 1.0, r, lvl) == "capped"]
                if capped:
                    ok = False
                    detail.append("%s/%s: %s capped" % (v, lvl, ", ".join(capped)))
                    continue
                gS = max(self.objectives(v, cell, "S")) - pstar
                gV = max(self.objectives(v, cell, variant)) - pstar
                detail.append("%s/%s: gap S %.3g, %s %.3g" % (v, lvl, gS, variant, gV))
                if gV > gS + GAP_SLACK * max(1.0, abs(pstar)):
                    ok = False
        return ok, "; ".join(detail)

    def recommend(self):
        warm = [c for c in self.cells if c[1] == "warm" and self.man.cases[c[0]]["eltype"] == "Float64"]
        i11 = [(INSTANCE_11_1, "admm_total", (1.0, l)) for l in ADMM_LEVELS]
        tight = [c for c in self.cells if c[1] == "admm_total" and c[2][1] == "tight"]
        out = {}
        for var in VARIANTS:
            crit = collections.OrderedDict()
            crit["A1 passes eta wherever S passes"] = self.accuracy_everywhere(var)
            crit["A2 not slower on the warmed prox (Float64 cells)"] = self.not_slower(var, warm)
            crit["A3 ADMM total faster at rho = 1, both levels (section 11.1 instance)"] = self.all_faster(var, i11)
            crit["A3 objective gap within S plus slack"] = self.gap_criterion(var)
            crit["B1 passes eta at the section 11.1 inputs, step 1"] = self.instance_accuracy(var)
            crit["B3 not slower at the tight target (all instances and rho)"] = self.not_slower(var, tight)
            val = {k: v[0] for k, v in crit.items()}
            names = list(crit)
            A = all(val[k] for k in names[:4])
            B = (not A) and val[names[4]] and val[names[2]] and val[names[5]]
            rec = "A" if A else ("B" if B else "C")
            out[var] = {"criteria": {k: {"value": v[0], "info": v[1]} for k, v in crit.items()}, "recommendation": rec}
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


def param_text(cp):
    if isinstance(cp, float):
        return "g=%g" % cp
    if isinstance(cp, tuple) and isinstance(cp[1], str) and cp[1] in ("rand", "admm"):
        return "g=%g/%s" % cp
    if isinstance(cp, tuple):
        return "rho=%g,level=%s" % cp
    return str(cp)


def primary_procs(camp, version):
    return [p for k, p in sorted(camp.procs.items()) if k[0] == version and k[3] == "1"]


def table_cells(camp):
    if camp.man is not None:
        return expected_cells(camp.man)
    seen = set()
    for p in camp.procs.values():
        for (case, meas, cp, _r, _role) in p.rows:
            if meas in MEASUREMENTS:
                seen.add((case, meas, cp))
    return sorted(seen, key=str)


def versions_of(camp):
    return sorted({k[0] for k in camp.procs} | set(camp.acc_files) | {v for (v, _c) in camp.untimed}) or list(VERSIONS)


def table_processes(camp):
    rows = []
    for k, p in sorted(camp.procs.items()):
        h = p.files[0].header
        rows.append([k[0], k[1], k[2], k[3], ", ".join(sorted(g for g in p.group_files)), h.get("rounds", "?"), h.get("cpu_vendor", "?"),
                     h.get("cpu_model", "?"), h.get("loadavg_start", "?"), ", ".join(sorted({f.header.get("wall_seconds", "-") for f in p.files}))])
    return md_table(["julia", "order", "proc", "BLAS threads", "groups", "rounds", "CPU vendor", "CPU model", "load at start", "wall s"], rows)


def table_aa(camp, cells):
    rows = []
    for v in versions_of(camp):
        for meas in MEASUREMENTS:
            devs = collections.defaultdict(list)
            for cell in cells:
                if cell[1] != meas:
                    continue
                for p in primary_procs(camp, v):
                    d = p.data().get(cell, {})
                    for a, b in (("S'", "S"), ("C_B'", "C_B"), ("C_D'", "C_D"), ("C_W'", "C_W")):
                        r = ratios(d, a, b)
                        if r:
                            devs["%s/%s" % (a, b)].append(abs(median(r.values()) - 1))
            for pair, d in sorted(devs.items()):
                rows.append([v, meas, pair, len(d), median(d), max(d)])
    return md_table(["julia", "measurement", "pair", "process-cells", "median |dev|", "max |dev|"], rows)


def table_labels(ver):
    rows = []
    for v in VERSIONS:
        for meas in MEASUREMENTS:
            for var in VARIANTS:
                for tp in ("Float64", "Float32"):
                    cs = [c for c in ver.cells if c[1] == meas and ver.man.cases[c[0]]["eltype"] == tp]
                    if not cs:
                        continue
                    recs = [ver.cell(v, c, var) for c in cs]
                    cnt = collections.Counter(r["label"] for r in recs)
                    rows.append([v, meas, var, tp, len(recs), cnt["faster"], cnt["slower"], cnt["inconclusive"], cnt["capped"],
                                 cnt["excluded"], median([r["center"] for r in recs])])
    return md_table(["julia", "measurement", "variant", "type", "cells", "faster", "slower", "inconclusive", "capped (ADMM)",
                     "excluded (accuracy)", "median ratio V/S"], rows)


def table_detail(camp, cells, ver, outdir):
    rows, per_rows = [], []
    for v in versions_of(camp):
        for cell in cells:
            for var in VARIANTS:
                r = ver.cell(v, cell, var) if ver is not None else judge(camp, v, cell, var)
                if ver is None and not r["per"]:
                    continue
                row = [v, cell[1], cell[0], param_text(cell[2]), var, r["n"], r["center"], r["lo_med"], r["hi_med"], r["lo"], r["hi"], r["spread"]]
                if ver is not None:
                    row += [r["label"], r["accuracy"]]
                rows.append(row)
                for (o, pid, med, lo, hi) in r["per"]:
                    per_rows.append([v, cell[0], cell[1], param_text(cell[2]), var, o, pid, repr(med), repr(lo), repr(hi)])
    if outdir is not None:
        with open(os.path.join(outdir, "per-process-ratios.tsv"), "w", encoding="utf-8") as fh:
            fh.write("julia\tcase\tmeasurement\tparam\tvariant\torder\tproc\tmedian\tmin\tmax\n")
            for r in per_rows:
                fh.write("\t".join(r) + "\n")
    heads = ["julia", "measurement", "case", "param", "variant", "processes", "median of medians", "min process median",
             "max process median", "min round", "max round", "A/A spread"]
    if ver is not None:
        heads += ["label", "accuracy at the executed pairs"]
    return md_table(heads, rows) if ver is not None else listing(rows, heads, "rows")


def pooled(camp, version, cell, role, field=1):
    """The values of a role in a cell over the primary processes of a version and all rounds (field 1: value, 2: GC
    seconds, 3: bytes, 4: allocations)."""
    vals = []
    cid, meas, cp = cell
    for p in primary_procs(camp, version):
        for rnd in range(0, ROUNDS + 1):
            v = p.rows.get((cid, meas, cp, rnd, role))
            if v is not None and finite(v[field]):
                vals.append(v[field])
    return vals


def table_first_use(camp, cells):
    rows, parts = [], []
    for v in versions_of(camp):
        for c in cells:
            if c[1] != "first_use":
                continue
            for role in PRIMARY:
                t = pooled(camp, v, c, role)
                if t:
                    rows.append([v, c[0], role, median(t), median(pooled(camp, v, c, role, 2)), median(pooled(camp, v, c, role, 3)),
                                 median(pooled(camp, v, c, role, 4)), len(t)])
                    for part in PARTS[role]:
                        x = pooled(camp, v, (c[0], "first_use_part", part), role)
                        if x:
                            parts.append([v, c[0], role, part, median(x)])
    a = md_table(["julia", "case", "role", "median s", "median GC s", "median bytes", "median allocations", "samples"], rows)
    b = md_table(["julia", "case", "role", "setup step", "median s"], parts)
    return a, b


def table_per_call(camp, cells):
    rows = []
    for v in versions_of(camp):
        for c in cells:
            if c[1] in ("first_use", "penalty") or (c[1] == "warm" and c[2] == (1.0, "rand")):
                vals = [pooled(camp, v, c, role) for role in PRIMARY]
                if any(vals):
                    rows.append([v, c[0], c[1], param_text(c[2])] + [median(x) for x in vals])
    return md_table(["julia", "case", "measurement", "param"] + PRIMARY, rows)


def table_plans(camp):
    rows = {}
    for (v, o, pid, b), p in camp.procs.items():
        if b != "1":
            continue
        for (case, grp), (plan, t) in p.plans.items():
            rows.setdefault((v, case, grp, plan["calls"], plan["batches"], plan["cap"]), []).append(t)
    out = [[v, case, grp, str(calls), str(nb), str(cap), median(ts), len(ts)] for (v, case, grp, calls, nb, cap), ts in sorted(rows.items())]
    return md_table(["julia", "case", "group", "calls per batch", "batches", "call cap", "slowest calibrated call s (median over processes)",
                     "processes"], out)


def table_alloc(camp):
    rows, nonzero = [], []
    for v in versions_of(camp):
        agg = collections.defaultdict(lambda: [0.0, 0.0, set(), 0])
        for p in primary_procs(camp, v):
            for (case, meas, cp, rnd, role), x in p.rows.items():
                if meas != "alloc":
                    continue
                s = agg[(role, cp)]
                s[0], s[1], s[3] = max(s[0], x[3]), max(s[1], x[4]), s[3] + 1
                if x[3] != 0 or x[4] != 0:
                    s[2].add(case)
        for (role, call), s in sorted(agg.items()):
            rows.append([v, role, call, s[3], s[0], s[1], len(s[2])])
            nonzero.extend([v, role, call, c] for c in sorted(s[2]))
    a = md_table(["julia", "role", "call", "samples", "max bytes per call", "max allocations per call", "cases with a nonzero count"], rows)
    return a + ("\n\nCases with a nonzero count:\n\n" + listing(nonzero, ["julia", "role", "call", "case"], "cases") if nonzero else "")


def table_workingset(camp):
    rows = []
    for v in versions_of(camp):
        cases, loads = {}, {}
        for p in primary_procs(camp, v):
            l3 = l3_bytes(p.files[0].header.get("l3_cache", ""))
            for (case, _grp, role), b in p.workingset.items():
                e = cases.setdefault((case, role), [0.0, l3])
                e[0] = max(e[0], b)
            for (case, grp), la in p.loads.items():
                loads[case] = max(loads.get(case, 0.0), la)
        for (case, role), (b, l3) in sorted(cases.items()):
            if role in PRIMARY:
                rows.append([v, case, role, b, l3, b / l3 if finite(l3) and l3 > 0 else math.nan, loads.get(case, math.nan)])
    return md_table(["julia", "case", "role", "reachable bytes", "L3 bytes", "ratio", "max load average (1 min) during the case"], rows)


def table_domain(camp, complete):
    """rho/sigma_max^2 (rho = inv(T(gamma)) rounded to the element type) of each main (case, step), with the largest
    eta/eps of each role there over both versions and the inputs, and the domain per design."""
    if camp.man is None:
        return "No manifest.", "No manifest."
    rows, dom = [], collections.defaultdict(lambda: [0, 0, 0.0, math.inf])
    for c in camp.man.cases.values():
        if c["kind"] != "main":
            continue
        for g in c["acc_gammas"]:
            ratio = penalty_of(g, c["eltype"]) / (c["sigma_max"] ** 2)
            row = [c["id"], c["design"], c["eltype"], g, ratio]
            for role in PRIMARY:
                worst, nfail, nrows = 0.0, 0, 0
                for v in VERSIONS:
                    for k in range(1, c["nv"] + 1):
                        r = camp.acc.get((v, c["id"], role), {}).get(gk(g, k))
                        if r is not None:
                            nrows += 1
                            worst = max(worst, r["eta"] / EPS[c["eltype"]])
                            nfail += 0 if r["pass"] else 1
                full = nrows == len(VERSIONS) * c["nv"]
                row.append("-" if nrows == 0 else "%s (%s%s)" % (fmt(worst), "fail" if nfail else "pass", "" if full else ", %d of %d" % (nrows, len(VERSIONS) * c["nv"])))
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
    if not complete:
        b += "\n\n(Observed values from the accepted files only; a pair counts when any of its outputs has evidence.)"
    return a, b


def table_accuracy(camp):
    out = []
    rows, margin, ctl_rows, fwd = [], [], [], []
    vers = sorted(camp.acc_files)
    cases = camp.man.cases.values() if camp.man is not None else []
    for v in vers:
        for kind in ("main", "control"):
            for role in PRIMARY:
                rs = [r for (vv, case, rl), d in camp.acc.items() if vv == v and rl == role for r in d.values() if r["kind"] == kind]
                if rs:
                    rows.append([v, kind, role, len(rs), sum(1 for r in rs if not r["pass"]),
                                 max(r["eta"] / EPS[r["eltype"]] for r in rs), median([r["eta"] / EPS[r["eltype"]] for r in rs])])
        for case in cases:
            ss = camp.acc.get((v, case["id"], "S"), {})
            if case["kind"] == "main" and ss:
                smax = max(r["eta"] / EPS[case["eltype"]] for r in ss.values())
                if smax > ETA_FACTOR / S_MARGIN:
                    margin.append([v, case["id"], smax])
        for role in PRIMARY:
            xs = [r["fe"] for (vv, case, rl), d in camp.acc.items() if vv == v and rl == role for r in d.values() if finite(r["fe"])]
            if xs:
                fwd.append([v, role, camp.acc_files[v][0].header.get("forward_max_n", "?"), len(xs), median(xs), max(xs)])
    out.append(("Accuracy: outputs, failures (eta > 1e3 eps, a nonfinite output or a recorded failure) and eta/eps per role",
                md_table(["julia", "kind", "role", "outputs", "failures", "max eta/eps", "median eta/eps"], rows)))
    pc = []
    for v in vers:
        for case in cases:
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
    out.append(("Accuracy per main case (all steps by inputs; eta/eta_S is the largest ratio over the (step, input) keys "
                "where S has a positive eta and the variant a finite one; other keys are left out of that column)",
                listing(pc, heads, "cases")))
    out.append(("Margin of S under the threshold: main cases where S exceeds eta/eps = %g (a factor %g below 1e3)" % (ETA_FACTOR / S_MARGIN, S_MARGIN),
                md_table(["julia", "case", "S max eta/eps"], margin) if margin else
                "No main case with accuracy evidence has an S output above eta/eps = %g." % (ETA_FACTOR / S_MARGIN)))
    reduces, sfail = collections.Counter(), collections.Counter()
    for v in vers:
        for case in cases:
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
                         [[v, var, reduces[(v, var)], sfail[v]] for v in vers for var in VARIANTS])))
    out.append(("Forward error against the 128-bit dual solution (cases with n up to the recorded limit; a diagnostic, relative to "
                "the norm of the solution, absolute when it is 0; outputs with a nonfinite entry or a recorded failure are left out)",
                md_table(["julia", "role", "n limit", "outputs", "median", "max"], fwd)))
    return out


def table_outputs(camp):
    if not camp.outputs:
        return "Unavailable: no outputs index was accepted."
    bad, compared = [], 0
    for v, outs in sorted(camp.outputs.items()):
        for o in outs:
            for (case, role), r in sorted(o.rows.items()):
                compared += 1
                if r["nfail"] != 0 or r["copy_identical"] != "true":
                    bad.append([v, case, role, r["nfail"], r["copy_identical"]])
    if compared == 0:
        return "Unavailable: the accepted outputs indices have no rows."
    if not bad:
        return ("In the %d accepted (Julia version, case, role) outputs: no recorded failure, and the A/A copy bitwise identical "
                "to its role in each." % compared)
    return listing(bad, ["julia", "case", "role", "recorded failures", "copy bitwise identical"], "outputs")


def table_admm(camp, cells, ver):
    out = []
    rrows = []
    for case, kvs in sorted(camp.refs.items()):
        for kv in kvs:
            rrows.append([case, kv.get("julia"), kv.get("iterations"), kv.get("converged"), kv.get("stationarity"),
                          kv.get("objective"), kv.get("cap"), kv.get("_file")])
    out.append(("Reference optima (S-ADMM at rho = 1 to stationarity 1e-12 max(1, |X'y|inf)), one per instance",
                md_table(["case", "julia", "iterations", "converged", "stationarity", "objective", "cap", "file"], rrows)))
    lv, traj = [], []
    for (v, case), us in sorted(camp.untimed.items()):
        for u in us:
            for (rho, role, lvl), r in sorted(u.levels.items()):
                lv.append([v, case, rho, role, lvl, r["status"], r["k"], u.cap, r["r"], r["s"], r["stationarity"], r["objective"]])
            for t in u.traj:
                if t[1] == "S":
                    traj.append([v, case, t[0], t[2], t[3], t[4], t[7], t[8]])
    out.append(("ADMM stopping records of the untimed pass: the status (reached, or capped after the cap of its file, which is "
                "%d in verdict mode), the iteration count k and the measures at k" % ADMM_CAP,
                listing(lv, ["julia", "case", "rho", "role", "level", "status", "k", "cap", "r", "s", "stationarity", "objective"], "records")))
    out.append(("ADMM trajectory of S at the checkpoints (the other primary roles are in the untimed files)",
                listing(traj, ["julia", "case", "rho", "k", "r", "s", "stationarity", "objective"], "rows")))
    rows, orows = [], []
    acells = [c for c in cells if c[1] == "admm_total"]
    for v in versions_of(camp):
        for cell in acells:
            for var in VARIANTS:
                r = ver.cell(v, cell, var) if ver is not None else judge(camp, v, cell, var)
                if ver is None and not r["per"]:
                    continue
                f = judge(camp, v, (cell[0], "admm_factor", cell[2]), var)
                it = judge(camp, v, (cell[0], "admm_iter", cell[2]), var)
                row = [v, cell[0], param_text(cell[2]), var, r["center"], f["center"], it["center"], r["n"]]
                if ver is not None:
                    row.append(r["label"] + (" (%s capped)" % ", ".join(r["capped"]) if r["capped"] else ""))
                rows.append(row)
    heads = ["julia", "case", "param", "variant", "total V/S", "factor V/S", "iterations V/S", "processes"] + (["label"] if ver is not None else [])
    out.append(("ADMM: total time ratio V/S per instance, rho and stopping level (and the ratios of the factorization and of the iterations)",
                listing(rows, heads, "rows")))
    for v in versions_of(camp):
        for cell in acells:
            refs = camp.refs.get(cell[0], [])
            pstar = refs[0]["_objective"] if len(refs) == 1 else None
            for role in PRIMARY:
                objs = pooled(camp, v, (cell[0], "admm_objective", cell[2]), role)
                if objs:
                    st = pooled(camp, v, (cell[0], "admm_stationarity", cell[2]), role)
                    orows.append([v, cell[0], param_text(cell[2]), role, len(objs), median(objs),
                                  (max(objs) - pstar) if pstar is not None else math.nan, median(st)])
    out.append(("ADMM: the objective of the timed runs (median), its largest gap to the reference of the same instance (- without "
                "exactly one accepted reference) and the median final stationarity",
                listing(orows, ["julia", "case", "param", "role", "runs", "median objective", "max gap", "median stationarity"], "rows")))
    return out


def table_diag(camp):
    rows = []
    for key, p in sorted(camp.procs.items()):
        if key[3] == "1":
            continue
        for cell, roles in sorted(p.data().items(), key=lambda x: str(x[0])):
            if cell[1] not in ("first_use", "warm", "penalty"):
                continue
            for var in VARIANTS:
                r = ratios(roles, var, "S")
                if r:
                    rows.append([key[0], key[1], key[2], key[3], cell[0], cell[1], param_text(cell[2]), var, median(r.values()),
                                 min(r.values()), max(r.values())])
    return listing(rows, ["julia", "order", "proc", "BLAS threads", "case", "measurement", "param", "variant", "median V/S", "min", "max"], "rows")


def format_verdict(recs, pins, man):
    lines = ["Campaign: source %s, harness %s, manifest %s.\n" % (pins["source_sha"], pins["harness_sha"], man.sha256)]
    rows = []
    for var, d in recs.items():
        for name, c in d["criteria"].items():
            rows.append([var, name, str(c["value"]), c["info"]])
        rows.append([var, "**recommendation**", "(%s)" % d["recommendation"], ""])
    lines.append(md_table(["variant", "criterion", "value", "evidence"], rows))
    lines.append("\n(A) propose a PB fixed-penalty backend: A1, A2 and both parts of A3 true. (B) solver-side factorization only: "
                 "(A) does not hold, and B1, the A3 speed criterion and B3 are true. (C) no change: both (A) and (B) are ruled out, "
                 "that is, a criterion of (A) is false and a criterion of (B) is false. The recommendation does not close #1078; "
                 "see the table of controls for the residuals.")
    return "\n".join(lines)


def build_tables(camp, ver, outdir):
    out = []

    def section(title, body):
        out.append("\n## %s\n\n%s\n" % (title, body))

    cells = table_cells(camp)
    section("Processes", table_processes(camp))
    section("A/A controls: deviation of the per-process median of the copy over its role from 1", table_aa(camp, cells))
    if ver is not None:
        section("Timing labels per measurement and variant (excluded = the variant or S fails accuracy at the executed pairs; "
                "capped = S or the variant reached the ADMM cap, a lower bound)", table_labels(ver))
    section("Timing detail: ratios V/S per cell (median of the per-process medians; range of the per-process medians; minimum and "
            "maximum of the per-round ratios; A/A spread)%s" % ("; the per-process values are in per-process-ratios.tsv" if outdir else ""),
            table_detail(camp, cells, ver, outdir))
    a, b = table_first_use(camp, cells)
    section("First use: median seconds, GC seconds, bytes and allocations of the primary roles", a)
    section("First-use breakdown: median seconds of each setup step (primary roles)", b)
    section("Median seconds per call of the primary roles (first use, warm at step 1 on random inputs, penalty schedules)", table_per_call(camp, cells))
    section("Batch plans from the calibration of each case", table_plans(camp))
    section("Warmed allocations per call (maximum over cases; cases with a nonzero count)", table_alloc(camp))
    section("Working set (reachable bytes of each object) against the L3 size, and the load average during each case", table_workingset(camp))
    a, b = table_domain(camp, ver is not None)
    section("rho / sigma_max^2 of each main case and step (rho = inv(T(gamma)) in the element type), with the largest eta/eps of "
            "each role there over both Julia versions and the 8 inputs (the count of evaluated outputs when it is not 16)", a)
    section("Accuracy domain per design: the failing pairs of each variant and the range of rho / sigma_max^2", b)
    for title, body in table_accuracy(camp):
        section(title, body)
    section("Outputs with a recorded failure or whose A/A copy differs bitwise", table_outputs(camp))
    for title, body in table_admm(camp, cells, ver):
        section(title, body)
    if any(k[3] != "1" for k in camp.procs):
        section("Processes with BLAS threads other than 1: ratios V/S per cell", table_diag(camp))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------

def refuse(root, title, P):
    removed = []
    for name in SUMMARY_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(name)
    msg = ["summarize.py: the campaign is refused (%s): %d problem%s" % (title, len(P), "" if len(P) == 1 else "s")]
    msg.extend("  - " + line for line in P.lines())
    if removed:
        msg.append("  (removed the earlier %s)" % ", ".join(removed))
    msg.append("No verdict was written. A partial or diagnostic summary is available with --tables-only.")
    text = "\n".join(msg)
    print(text, file=sys.stderr)
    return 2


def run_verdict(args):
    root = args.dir
    pins = {"source_sha": args.source_sha, "harness_sha": args.harness_sha}
    P = Problems()
    for key, val in pins.items():
        if not val or not all(ch in "0123456789abcdef" for ch in val) or not 7 <= len(val) <= 40:
            P.add("pins", "--%s must be 7 to 40 lowercase hex digits" % key.replace("_", "-"), repr(val))
    if not args.manifest or not os.path.isfile(args.manifest):
        P.add("pins", "--manifest must name the inputs-index.tsv of the campaign", repr(args.manifest))
    elif not os.path.abspath(args.manifest).startswith(os.path.abspath(root) + os.sep):
        P.add("pins", "--manifest must be the inputs-index.tsv gathered into the campaign directory", args.manifest)
    man = load_manifest(args.manifest, P) if not P else None
    if man is not None:
        check_manifest_registered(man, pins, P)
    listed = read_checksums(root, P)
    if P:
        return refuse(root, "class 1: the manifest, the pinned identity and the checksum list", P)
    camp = Campaign(root, man, pins, verdict=True)
    parsed = camp.load_files(P)
    check_checksums(root, listed, P)
    if P:
        return refuse(root, "class 2: the files on their own", P)
    camp.accept(parsed, set())
    camp.inventory(P)
    if P:
        return refuse(root, "class 3: the inventory and the consistency across files", P)
    ver = Verdict(camp)
    recs = ver.recommend()
    out = build_tables(camp, ver, root)
    out.append("\n## Criteria and recommendation\n\n%s\n" % format_verdict(recs, pins, man))
    text = "# Summary of the campaign in %s\n%s\n" % (os.path.basename(os.path.abspath(root)), "\n".join(out))
    print(text)
    with open(os.path.join(root, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    verdict = {"source_sha": pins["source_sha"], "harness_sha": pins["harness_sha"], "manifest_sha256": man.sha256,
               "variants": recs}
    with open(os.path.join(root, "verdict.json"), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)
    return 0


def run_tables_only(args):
    root = args.dir
    pins = {"source_sha": args.source_sha, "harness_sha": args.harness_sha}
    P = Problems()
    paths = [args.manifest] if args.manifest else sorted(glob.glob(os.path.join(root, "**", "inputs-index.tsv"), recursive=True))
    man = None
    for path in paths:
        man = load_manifest(path, P)
        if man is not None:
            break
    camp = Campaign(root, man, pins, verdict=False)
    parsed = camp.load_files(P)
    camp.accept(parsed, P.files())
    gaps = Problems()
    if man is not None:
        check_manifest_registered(man, pins if all(pins.values()) else dict(pins, source_sha=man.header.get("source_sha"),
                                                                              harness_sha=man.header.get("harness_sha")), gaps)
        try:
            camp.inventory(gaps)
        except (KeyError, IndexError, TypeError) as e:
            gaps.add("inventory", "not checked further", repr(e))
    out = build_tables(camp, None, None)
    head = ["# PARTIAL tables of %s (--tables-only: no criteria, no recommendation; not evidence)\n" % os.path.basename(os.path.abspath(root)),
            "Manifest: %s." % (("%s (%s)" % (os.path.relpath(man.path, root) if man.path.startswith(root) else man.path, man.sha256)) if man else "none"),
            "Identities of the accepted timing files: %s.\n" % (sorted({(f.header.get("source_sha"), f.header.get("harness_sha"))
                                                                    for p in camp.procs.values() for f in p.files}) or "none")]
    if P:
        head.append("Files left out (problems on their own):\n")
        head.extend("- " + l for l in P.lines()[:MAX_LISTED * 5])
        head.append("")
    if gaps:
        head.append("Gaps against the registered campaign (a verdict needs none):\n")
        lines = gaps.lines()
        head.extend("- " + l for l in lines[:MAX_LISTED])
        if len(lines) > MAX_LISTED:
            head.append("- (%d further gaps not listed)" % (len(lines) - MAX_LISTED))
        head.append("")
    text = "\n".join(head) + "\n".join(out) + "\n"
    print(text)
    with open(os.path.join(root, "tables-partial.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description="Summarizer of the wide least-squares study (#1103).")
    ap.add_argument("dir", nargs="?", help="the campaign directory (searched recursively)")
    ap.add_argument("--gather", nargs=2, metavar=("PRODUCER_DIR", "DEST_DIR"),
                    help="copy the campaign files of the producer tree into a new directory and write %s" % CHECKSUM_FILE)
    ap.add_argument("--inputs", default="", help="with --gather: the inputs-index.tsv of the campaign")
    ap.add_argument("--source-sha", default="", help="the pinned source commit")
    ap.add_argument("--harness-sha", default="", help="the pinned harness commit")
    ap.add_argument("--manifest", default="", help="the inputs-index.tsv of the campaign")
    ap.add_argument("--tables-only", action="store_true", help="print tables from the available files; no criteria or verdict")
    ap.add_argument("--write-checksums", action="store_true",
                    help="rewrite %s over the files of a directory (ALLOWLIST) and exit" % CHECKSUM_FILE)
    args = ap.parse_args(argv[1:])
    if args.gather:
        try:
            n = gather(args.gather[0], args.gather[1], args.inputs)
        except (GatherError, OSError) as e:
            print("summarize.py --gather refused: %s" % e, file=sys.stderr)
            return 2
        print("gathered %d files into %s and wrote %s" % (n, args.gather[1], CHECKSUM_FILE))
        return 0
    if not args.dir or not os.path.isdir(args.dir):
        ap.error("%s is not a directory" % args.dir)
    if args.write_checksums:
        n = write_checksums(args.dir)
        print("wrote %s with %d files" % (os.path.join(args.dir, CHECKSUM_FILE), n))
        return 0
    return run_tables_only(args) if args.tables_only else run_verdict(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
