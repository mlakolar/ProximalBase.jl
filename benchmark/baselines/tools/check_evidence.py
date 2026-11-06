#!/usr/bin/env python3
"""Checks of the archived evidence of the iterate-family 0.5 baseline.

usage: python3 check_evidence.py EXTRACTION_DIR [BASELINE_DIR]

EXTRACTION_DIR is the extracted archive (the directory that holds base-groups, aa-layouts, cmp-groups,
...); BASELINE_DIR holds the committed iterate-family-0.5.csv (default: this repository's). Three
groups of checks, in this order; a failed structure check skips the others, which would otherwise
fail on missing data:

  structure  the twelve result directories hold exactly repetitions 1-3, every case once, every file
             with the expected row count, case set and repetition, the error files empty, and every value
             the later checks read well formed (so that bad data is a named error, not a crash);
  identity   the twelve metadata files record the same harness, source, Julia, settings and kind;
  aa/counts  the A/A comparison (speedup, 95% interval, means, timer-limited, assessment) is recomputed
             from the raw means of the base and A/A repetitions, the committed csv columns aa_ratio,
             aa_half_width, gate5 and timer_limited are recomputed, and the counts of the baseline are
             recomputed and compared with pinned values and with the csv.

The timer-limited rule is the frozen historical/validate.py, loaded by its path. The arithmetic is that
of the historical compare.py. Archived floats are compared within ULPS units in the last place of the
larger value: the archive was computed on macOS, and the exp and log of another platform's libm may
differ by one unit. Every figure the baseline publishes is rounded to four decimals.

A stage that raises anyway is reported as "<stage>: could not complete: <ExceptionType>: <message>" (the
errors found so far are kept, the dependent stages skipped): a defect of the checks, not a finding.

Standard library only; Python 3.11 or later. Each error names the check, the file, the case, the repetition
and the field where they apply.
"""
import collections, csv, importlib.util, json, math, os, statistics, sys, tomllib
from types import SimpleNamespace

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "historical", "validate.py")
BASELINES = os.path.dirname(HERE)
CSV_NAME = "iterate-family-0.5.csv"

ROLES, KINDS, REPS = ("base", "aa"), ("groups", "layouts"), (1, 2, 3)
JULIA_TAG = "julia-1.13.3"
N_CASES = {"groups": 3455, "layouts": 264}
T975_2 = 4.303            # two-sided 95% Student t for two degrees of freedom, as summarize.py has it
ULPS = 4
GATE_HALF = 0.05
PIN = {"harness_hash": "844fa5330b9065ab33fc7bb9ef7605dbe8560e16",
       "source_sha": "def81ec9d021c39ed1727cfeb9598357291dd60d",
       "source_hash": "a1583f070615bd3ea38705eacda4401ca1c6bafd", "julia": "1.13.3"}
PIN_GROUPS = {"groups": "atom,owner,products,sparse", "layouts": "vectors"}
# Present and equal in all twelve metadata files: the sixteen keys the generator checks, and more.
SAME_KEYS = ("julia", "cpu_model", "cpu", "os", "check_bounds", "blas", "blas_threads", "julia_threads",
             "benchmarktools", "harness_hash", "source_hash", "source_sha", "source_dirty", "samples_target",
             "seconds_limit", "seed_scheme", "profile_seconds", "calibration_samples", "calibration_seconds",
             "tune_seconds")
# Counts of the baseline, recomputed from the evidence (the numbers of verify_csv3.py and the provenance record).
COUNTS = {"cases": 3719, "timer_limited_base": 94, "timer_limited_aa": 94, "timer_limited_aa_only": 0,
          "timer_limited": 94, "not_timer_limited": 3625, "gate5": 1092, "other": 2533, "varying_work": 104,
          "aa_conclusive": 104, "diagnostic": 16, "diagnostic_flipped": 2}


def is_diagnostic(row):
    """The two first-activation layouts that the generator excludes from gate5 by decision."""
    return (row["family"] == "sparse" and row["name"] == "first_activation"
            and row["layout"] in ("absent_distinct_batch", "absent_distinct_batch_x16"))


def frozen():
    """The historical timer rule, loaded by path so that the name `validate` never resolves."""
    spec = importlib.util.spec_from_file_location("frozen_historical_validate", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aa_stats(base, aa):
    """The historical compare.py arithmetic for paired means of one case (base before, A/A after)."""
    ratios = [math.log(old / new) for old, new in zip(base, aa)]
    mu, half = statistics.mean(ratios), T975_2 * statistics.stdev(ratios) / math.sqrt(len(ratios))
    return dict(mu=mu, half=half, paired_geometric_speedup=math.exp(mu), ci95_low=math.exp(mu - half),
                ci95_high=math.exp(mu + half), before_mean_ns=statistics.mean(base), after_mean_ns=statistics.mean(aa))


def assessment(mu, half, limited):
    return "timer-limited" if limited else "inconclusive" if mu - half <= 0 <= mu + half else "faster" if mu > 0 else "slower"


def ulps(a, b):
    m = max(abs(a), abs(b))
    return abs(a - b) / math.ulp(m) if m else 0.0


class Errors:
    """Error messages; after LIMIT of one (check, field) the rest is only counted."""
    LIMIT = 5

    def __init__(self):
        self.items, self.seen, self.dropped = [], collections.Counter(), collections.Counter()
        self.structure = self.crashed = False

    def add(self, check, field, message):
        if check == "structure":
            self.structure = True
        self.seen[check, field] += 1
        if self.seen[check, field] <= self.LIMIT:
            self.items.append(f"{check}: {message}")
        else:
            self.dropped[check, field] += 1

    def list(self):
        return self.items + [f"{c}: {n} more errors about {f}" for (c, f), n in sorted(self.dropped.items())]


def read_table(root, rel, errs, columns, delimiter):
    """The rows of a csv or tsv file, or None (with an error) if unreadable or columns are missing."""
    try:
        with open(os.path.join(root, rel), newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        errs.add("structure", rel, f"{rel} cannot be read: {type(exc).__name__}: {exc}")
        return None
    missing = [c for c in columns if c not in (reader.fieldnames or [])]
    if missing:
        errs.add("structure", rel, f"{rel} has no column {', '.join(missing)}")
        return None
    return rows


def numbers(rows, rel, errs, fields, positive=()):
    """Every field of every row is a finite number (positive for `positive`)."""
    for row in rows:
        for f in fields:
            try:
                v = float(row[f])
                good = math.isfinite(v) and (v > 0 or f not in positive)
            except (TypeError, ValueError):
                good = False
            if not good:
                errs.add("structure", f"{rel} {f}", f"{rel}, case {row.get('id')}: field {f} is {row.get(f)!r}, "
                         f"not a finite{' positive' if f in positive else ''} number")


def integers(rows, rel, errs, fields):
    """Every field of every row is a positive integer in the form int() accepts (so not 1.5, 1e3 or an empty string)."""
    for row in rows:
        for f in fields:
            try:
                good = int(row[f]) > 0
            except (TypeError, ValueError):
                good = False
            if not good:
                errs.add("structure", f"{rel} {f}", f"{rel}, case {row.get('id')}: field {f} is {row.get(f)!r}, "
                         f"not a positive integer")


def case_set(rows, rel, errs, expected):
    """Row count and uniqueness of ids; returns the id set."""
    ids = [r.get("id") for r in rows]
    if len(ids) != expected:
        errs.add("structure", f"{rel} rows", f"{rel} has {len(ids)} rows, expected {expected}")
    for i, n in collections.Counter(ids).items():
        if n > 1:
            errs.add("structure", f"{rel} duplicate", f"{rel}: case {i} appears {n} times")
    return set(ids)


def same_cases(have, want, rel, against, errs):
    for what, ids in (("missing", want - have), ("unexpected", have - want)):
        if ids:
            errs.add("structure", f"{rel} cases", f"{rel}: {len(ids)} {what} cases compared with {against}, "
                     f"e.g. {sorted(ids, key=str)[0]}")


def guarded(stage):
    """Turn an unexpected exception of a stage into a named error, so that no traceback hides the findings so far.
    The message names the exception type: it is a defect of the checks or of unforeseen data, not a finding."""
    def wrap(fn):
        def run(*args):
            errs = next(a for a in args if isinstance(a, Errors))
            try:
                return fn(*args)
            except Exception as exc:
                errs.add(stage, "could not complete", f"could not complete: {type(exc).__name__}: {exc}")
                errs.crashed = True
                return None
        run.__name__, run.__doc__ = fn.__name__, fn.__doc__
        return run
    return wrap


@guarded("structure")
def check_structure(root, baseline_dir, errs):
    """Load the evidence; return it, or None if a structure check failed (the errors are in errs)."""
    ev = SimpleNamespace(root=root, runs={}, cmp={}, coverage={}, csv={}, meta={})
    # comparison files and the committed csv first: they define the case sets
    cmp_ids = {}
    for k in KINDS:
        rel = f"cmp-{k}/comparison.csv"
        rows = read_table(root, rel, errs, ("julia", "id", "paired_runs", "timer_limited", "timer_limit_multiple",
                                            "before_mean_ns", "after_mean_ns", "paired_geometric_speedup", "ci95_low",
                                            "ci95_high", "timing_assessment"), ",")
        if rows is not None:
            numbers(rows, rel, errs, ("before_mean_ns", "after_mean_ns", "paired_geometric_speedup", "ci95_low",
                                      "ci95_high"), positive=("ci95_low", "ci95_high"))
            cmp_ids[k] = case_set(rows, rel, errs, N_CASES[k])
            ev.cmp[k] = {r["id"]: r for r in rows}
        rel = f"cmp-{k}/coverage.json"
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                cov = json.load(fh)
        except (OSError, ValueError) as exc:         # includes JSON and UTF-8 decoding errors
            errs.add("structure", rel, f"{rel} cannot be read: {type(exc).__name__}: {exc}")
        else:
            bad = [f for f in ("before_only", "after_only") if not isinstance(cov, dict) or not isinstance(cov.get(f), list)]
            if not isinstance(cov, dict) or isinstance(cov.get("shared_measurements"), bool) or not isinstance(cov.get("shared_measurements"), int):
                bad.append("shared_measurements")
            for f in bad:
                errs.add("structure", rel + f, f"{rel}: field {f} is {cov.get(f) if isinstance(cov, dict) else cov!r}, "
                         f"expected {'an integer' if f == 'shared_measurements' else 'a list'} in a JSON object")
            ev.coverage[k] = cov
    rows = read_table(baseline_dir, CSV_NAME, errs, ("id", "family", "name", "layout", "work_varies", "timer_limited",
                                                     "aa_ratio", "aa_half_width", "gate5"), ",")
    csv_ids = None
    if rows is not None:
        csv_ids = case_set(rows, CSV_NAME, errs, sum(N_CASES.values()))
        ev.csv = {r["id"]: r for r in rows}
    if len(cmp_ids) == 2 and csv_ids is not None:
        union = cmp_ids["groups"] | cmp_ids["layouts"]
        same_cases(union, csv_ids, "the comparison files", CSV_NAME, errs)
        same_cases(csv_ids, union, CSV_NAME, "the comparison files", errs)
    # the twelve result directories
    for role in ROLES:
        for k in KINDS:
            d = f"{role}-{k}"
            names = {f"{JULIA_TAG}-rep{r:02d}-{part}" for r in REPS for part in ("metadata.toml", "metrics.tsv", "errors.tsv")}
            try:
                found = set(os.listdir(os.path.join(root, d)))
            except OSError:
                found = set()
            for what, files in (("missing", names - found), ("unexpected", found - names)):
                for name in sorted(files):
                    errs.add("structure", d + what, f"{d}/{name} is {what} (repetitions 1-3 and nothing else are expected)")
            runs = ev.runs[role, k] = {}
            for r in REPS:
                prefix = f"{JULIA_TAG}-rep{r:02d}-"
                stem, meta = f"{d}/{prefix}", None
                if prefix + "metadata.toml" in found:
                    try:
                        with open(os.path.join(root, stem + "metadata.toml"), "rb") as fh:
                            meta = tomllib.load(fh)
                    except (OSError, ValueError) as exc:      # includes TOMLDecodeError and UnicodeDecodeError
                        errs.add("structure", stem, f"{stem}metadata.toml cannot be read: {type(exc).__name__}: {exc}")
                rows = (read_table(root, stem + "metrics.tsv", errs, ("id", "rep", "work", "evals", "mean_ns", "median_ns"), "\t")
                        if prefix + "metrics.tsv" in found else None)
                if (prefix + "errors.tsv" in found
                        and (er := read_table(root, stem + "errors.tsv", errs, ("id", "exception", "stack"), "\t")) is not None
                        and er):
                    errs.add("structure", stem + "errors", f"{stem}errors.tsv has {len(er)} error rows, expected none")
                if rows is None:
                    continue
                rel = stem + "metrics.tsv"
                ids = case_set(rows, rel, errs, N_CASES[k])
                numbers(rows, rel, errs, ("work", "median_ns", "mean_ns"), positive=("mean_ns",))
                integers(rows, rel, errs, ("evals",))
                if k in cmp_ids:
                    same_cases(ids, cmp_ids[k], rel, f"cmp-{k}/comparison.csv", errs)
                for row in rows:
                    if row.get("rep") != str(r):
                        errs.add("structure", rel + " rep", f"{rel}, case {row.get('id')}: field rep is {row.get('rep')!r}, "
                                 f"expected {r} (repetition {r})")
                if meta is not None:
                    ev.meta[stem + "metadata.toml"] = (role, k, r, meta)
                    floor = meta.get("timer_floor_ns")      # read by the timer rule of every dependent check
                    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
                        errs.add("structure", stem + "floor", f"{stem}metadata.toml: field timer_floor_ns is {floor!r}, "
                                 f"expected a finite positive number")
                    if meta.get("cases") != len(rows):
                        errs.add("structure", stem + "cases", f"{stem}metadata.toml: field cases is {meta.get('cases')!r}, "
                                 f"{rel} has {len(rows)} rows")
                runs[r] = SimpleNamespace(meta=meta, rows={x.get("id"): x for x in rows})
    return None if errs.structure else ev


@guarded("identity")
def check_identity(ev, errs):
    """The metadata files agree on the run, and the comparison files on their settings."""
    ref = None
    for rel, (role, k, r, meta) in ev.meta.items():
        for key, want in (*PIN.items(), ("groups", PIN_GROUPS[k]), ("suite", "extended")):
            if meta.get(key) != want:
                errs.add("identity", key, f"{key} in {rel} is {meta.get(key)!r}, expected {want!r}")
        if meta.get("source_dirty") is not False:
            errs.add("identity", "source_dirty", f"source_dirty in {rel} is {meta.get('source_dirty')!r}, expected false")
        if isinstance(meta.get("failures"), bool) or meta.get("failures") != 0:
            errs.add("identity", "failures", f"failures in {rel} is {meta.get('failures')!r}, expected 0")
        if meta.get("rep") != r:
            errs.add("identity", "rep", f"rep in {rel} is {meta.get('rep')!r}, expected {r}")
        ref = ref or (rel, meta)
        for key in SAME_KEYS:
            if key not in meta:
                errs.add("identity", key, f"{key} is missing in {rel}")
            elif meta[key] != ref[1].get(key):
                errs.add("identity", key, f"{key} in {rel} is {meta[key]!r}, expected {ref[1].get(key)!r} as in {ref[0]}")
    for k in KINDS:
        for i, row in ev.cmp.get(k, {}).items():
            for field, want in (("julia", JULIA_TAG), ("paired_runs", "3"), ("timer_limit_multiple", "10")):
                if row[field] != want:
                    errs.add("identity", f"cmp {field}", f"cmp-{k}/comparison.csv, case {i}: field {field} is {row[field]!r}, expected {want!r}")
        cov, n = ev.coverage[k], len(ev.cmp[k]) * len(REPS)
        if cov["before_only"] or cov["after_only"] or cov["shared_measurements"] != n:
            errs.add("identity", "coverage", f"cmp-{k}/coverage.json: before_only {cov['before_only'][:2]}, after_only "
                     f"{cov['after_only'][:2]}, shared_measurements {cov['shared_measurements']!r}, expected none and {n}")


@guarded("aa/counts")
def check_aa_counts(ev, errs, info):
    """Recompute the A/A comparison, the csv columns and the counts; info collects report lines."""
    rule = frozen()
    bit, worst = collections.Counter(), collections.defaultdict(float)
    n = collections.Counter(); derived = {}
    for k in KINDS:
        cmp_rel = f"cmp-{k}/comparison.csv"
        for i, arch in ev.cmp[k].items():
            runs = {role: [ev.runs[role, k][r] for r in REPS] for role in ROLES}
            rows = {role: [x.rows[i] for x in runs[role]] for role in ROLES}
            limit = {role: [rule.timer_limited(row, x.meta, 10) for row, x in zip(rows[role], runs[role])] for role in ROLES}
            s = aa_stats([float(x["mean_ns"]) for x in rows["base"]], [float(x["mean_ns"]) for x in rows["aa"]])
            limited = any(limit["base"]) or any(limit["aa"])
            for field in ("paired_geometric_speedup", "ci95_low", "ci95_high", "before_mean_ns", "after_mean_ns"):
                a, d = float(arch[field]), ulps(float(arch[field]), s[field])
                bit[field] += a == s[field]; worst[field] = max(worst[field], d)
                if d > ULPS:
                    errs.add("aa", field, f"{field} of {i} in {cmp_rel} is {arch[field]}, recomputed {s[field]!r} "
                             f"from the base and A/A mean_ns of repetitions 1-3 ({d:.3g} ulps, bound {ULPS})")
            if arch["timer_limited"] != str(limited):
                errs.add("aa", "timer_limited", f"timer_limited of {i} in {cmp_rel} is {arch['timer_limited']}, "
                         f"recomputed {limited} by the frozen rule (base {limit['base']}, A/A {limit['aa']} for repetitions 1-3)")
            want = assessment(s["mu"], s["half"], limited)
            if arch["timing_assessment"] != want:
                errs.add("aa", "timing_assessment", f"timing_assessment of {i} in {cmp_rel} is "
                         f"{arch['timing_assessment']!r}, recomputed {want!r}")
            derived[i] = SimpleNamespace(s=s, limit=limit, limited=limited, rows=rows["base"],
                                         half=math.sqrt(float(arch["ci95_high"]) / float(arch["ci95_low"])) - 1,
                                         g=float(arch["paired_geometric_speedup"]))
    total = len(derived)
    info.append(f"aa: {total} cases; bit-equal " + ", ".join(f"{f} {bit[f]}" for f in worst)
                + f"; largest distance {max(worst.values()):.3g} ulps (bound {ULPS})")
    # the committed csv
    flipped = diagnostic = 0
    for i, c in derived.items():
        row = ev.csv[i]
        diag = is_diagnostic(row)
        gate_before = not c.limited and c.half <= GATE_HALF
        diagnostic += diag; flipped += diag and gate_before
        c.gate5 = gate_before and not diag
        c.varying = len({float(x["work"]) for x in c.rows}) > 1
        for field, want in (("aa_ratio", f"{c.g:.4f}"), ("aa_half_width", f"{c.half:.4f}"),
                            ("gate5", str(c.gate5).lower()), ("timer_limited", str(any(c.limit["base"])).lower()),
                            ("work_varies", str(c.varying).lower())):
            if row[field] != want:
                errs.add("csv", field, f"{field} of {i} in {CSV_NAME} is {row[field]!r}, recomputed {want!r}")
    # counts
    live = [c for c in derived.values() if not c.limited]
    got = {"cases": total, "timer_limited_base": sum(any(c.limit["base"]) for c in derived.values()),
           "timer_limited_aa": sum(any(c.limit["aa"]) for c in derived.values()),
           "timer_limited_aa_only": sum(any(c.limit["aa"]) and not any(c.limit["base"]) for c in derived.values()),
           "timer_limited": total - len(live), "not_timer_limited": len(live), "gate5": sum(c.gate5 for c in live),
           "varying_work": sum(c.varying for c in derived.values()), "diagnostic": diagnostic, "diagnostic_flipped": flipped,
           "aa_conclusive": sum(c.s["mu"] - c.s["half"] > 0 or c.s["mu"] + c.s["half"] < 0 for c in live)}
    got["other"] = len(live) - got["gate5"]
    from_csv = {"gate5": sum(r["gate5"] == "true" for r in ev.csv.values()),
                "timer_limited_base": sum(r["timer_limited"] == "true" for r in ev.csv.values()),
                "varying_work": sum(r["work_varies"] == "true" for r in ev.csv.values()), "cases": len(ev.csv)}
    for key, want in COUNTS.items():
        if got[key] != want:
            errs.add("counts", key, f"{key} is {got[key]} from the evidence, expected {want}")
        if key in from_csv and from_csv[key] != got[key]:
            errs.add("counts", key + " csv", f"{key} is {from_csv[key]} in {CSV_NAME}, {got[key]} from the evidence")
    if sum(len(ev.cmp[k]) for k in KINDS) != sum(N_CASES.values()):
        errs.add("counts", "cases", "counts: the comparison files hold other than 3455 + 264 cases")
    info.append("counts: " + ", ".join(f"{k} {v}" for k, v in got.items()))


def check(root, baseline_dir=BASELINES, info=None):
    """All checks; the list of error messages (empty if the evidence passes)."""
    errs, info = Errors(), [] if info is None else info
    ev = check_structure(root, baseline_dir, errs)
    if ev is not None:
        check_identity(ev, errs)
        if not errs.crashed:
            check_aa_counts(ev, errs, info)
    return errs.list()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__.split("\n\n")[0])
    lines = []
    problems = check(sys.argv[1], *sys.argv[2:], info=lines)
    print("\n".join(lines + problems))
    sys.exit(1 if problems else 0)
