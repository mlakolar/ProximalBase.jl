#!/usr/bin/env python3
"""Generate benchmark/baselines/iterate-family-0.5.{md,csv} from the repetitions' files.

Every figure comes from the *-metrics.tsv and *-metadata.toml files of the result directories;
nothing is typed. The identity of the run is checked first (one harness hash, the pinned source,
a clean tree, the same Julia/CPU/OS/settings everywhere, the repetitions complete, no failures)
and the script stops if it does not hold.

usage: make_baseline.py --groups DIR --layouts DIR --reps 3 --benchmark-dir REPO/benchmark
         --harness-commit SHA --source-commit SHA --out-md FILE --out-csv FILE [--crosscheck FILE]
"""
import argparse, collections, csv, glob, math, os, re, statistics as st, sys, tomllib

ap = argparse.ArgumentParser()
ap.add_argument("--groups", required=True); ap.add_argument("--layouts", required=True)
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--benchmark-dir", required=True)
ap.add_argument("--harness-commit", required=True); ap.add_argument("--source-commit", required=True)
ap.add_argument("--out-md", required=True); ap.add_argument("--out-csv", required=True)
ap.add_argument("--crosscheck"); ap.add_argument("--spread-limit", type=float, default=0.05)
ap.add_argument("--crosscheck-end", help="second independent-loop file (after the repetitions)")
ap.add_argument("--restore-probe", help="output of restore_probe.jl (same loop code in another process)")
ap.add_argument("--aa-compare", nargs="*", default=[], help="comparison.csv of compare.py for the A/A control (set base before, set aa after)")
ap.add_argument("--replicate-json", help="same-input replicate statistics (replicate2.py)")
ap.add_argument("--check", action="store_true", help="only check the identity and print the summary")
args = ap.parse_args()
sys.path.insert(0, args.benchmark_dir)
from validate import timer_limited          # the repository's own definition

def tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

# ---- load and check identity ----------------------------------------------------------------
rows_by_id = collections.defaultdict(list)       # id -> [(rep, row, meta)]
metas = []
for label, d in (("groups", args.groups), ("layouts", args.layouts)):
    files = sorted(glob.glob(os.path.join(d, "*-metadata.toml")))
    if len(files) != args.reps:
        sys.exit(f"{label}: {len(files)} metadata files, expected {args.reps}")
    for mf in files:
        meta = tomllib.load(open(mf, "rb")); meta["_label"] = label; metas.append(meta)
        if meta["failures"] != 0: sys.exit(f"{mf}: failures = {meta['failures']}")
        if meta["suite"] != "extended": sys.exit(f"{mf}: suite {meta['suite']}")
        for r in tsv(mf.replace("-metadata.toml", "-metrics.tsv")):
            rows_by_id[r["id"]].append((int(r["rep"]), r, meta))
        errs = tsv(mf.replace("-metadata.toml", "-errors.tsv"))
        if errs: sys.exit(f"{mf}: {len(errs)} error rows")
def same(key, allow_missing=False):
    vals = {str(m.get(key)) for m in metas}
    if len(vals) != 1: sys.exit(f"metadata key {key} differs: {sorted(vals)}")
    return vals.pop()
ident = {k: same(k) for k in ("julia", "cpu_model", "cpu", "os", "check_bounds", "blas", "blas_threads",
                              "julia_threads", "benchmarktools", "harness_hash", "source_hash", "source_sha",
                              "source_dirty", "samples_target", "seconds_limit", "seed_scheme")}
if ident["source_sha"] != args.source_commit: sys.exit(f"source_sha {ident['source_sha']} != {args.source_commit}")
if ident["source_dirty"].lower() != "false": sys.exit("the source tree was dirty")
if ident["julia_threads"] != "1" or ident["blas_threads"] != "1": sys.exit("threads are not 1")
floors = [m["timer_floor_ns"] for m in metas]
groups_meta = {m["_label"]: m["groups"] for m in metas}
for cid, lst in rows_by_id.items():
    if sorted(rep for rep, _, _ in lst) != list(range(1, args.reps + 1)): sys.exit(f"{cid}: repetitions {[r for r,_,_ in lst]}")
print(f"{len(rows_by_id)} cases; harness_hash {ident['harness_hash']}; julia {ident['julia']}; {ident['cpu_model']}; {ident['os']}")

# ---- per-case figures ------------------------------------------------------------------------
cases = []
for cid, lst in rows_by_id.items():
    lst.sort(key=lambda t: t[0])
    first = lst[0][1]
    meds = [float(r["median_ns"]) for _, r, _ in lst]
    mins = [float(r["min_ns"]) for _, r, _ in lst]
    evals = sorted({int(r["evals"]) for _, r, _ in lst})
    limited = any(timer_limited(r, m, 10) for _, r, m in lst)
    works = [float(r["work"]) for _, r, _ in lst]
    work = float(st.median(works))
    ratios = [m / w for m, w in zip(meds, works) if w > 0]
    med = st.median(meds)
    cases.append(dict(id=cid, family=first["family"], name=first["name"], dtype=first["dtype"], shape=first["shape"],
                      layout=first["layout"], phase=first["phase"], work=int(round(work)), work_varies=len(set(works)) > 1, expected_zero=first["expected_zero"],
                      evals="/".join(map(str, evals)), median_ns=med, min_ns=min(mins),
                      spread=max(meds) / min(meds) - 1, per_work_ns=(st.median(ratios) if ratios else float("nan")),
                      max_bytes=max(float(r["max_bytes"]) for _, r, _ in lst), timer_limited=limited))
cases.sort(key=lambda c: c["id"])
aa_rows = {}
for path in args.aa_compare:
    for r in csv.DictReader(open(path)): aa_rows[r["id"]] = r
for c in cases:
    r = aa_rows.get(c["id"])
    c["aa"] = None; c["gate5"] = None
    if r is not None:
        g, lo, hi = float(r["paired_geometric_speedup"]), float(r["ci95_low"]), float(r["ci95_high"])
        half = math.sqrt(hi / lo) - 1 if lo > 0 and hi > 0 else float("inf")
        c["aa"] = dict(g=g, half=half, assess=r["timing_assessment"])
        c["gate5"] = (not c["timer_limited"]) and r["timer_limited"] != "True" and half <= 0.05
        # diagnostic by decision: each of the two layouts missed the 20% cross-check with its independent loop on one of the two machines tried
        if c["family"] == "sparse" and c["name"] == "first_activation" and c["layout"] in ("absent_distinct_batch", "absent_distinct_batch_x16"): c["gate5"] = False
if args.aa_compare:
    missing = [c["id"] for c in cases if c["aa"] is None]
    if missing: sys.exit(f"{len(missing)} cases without an A/A row, e.g. {missing[0]}")
if args.check:
    sys.exit(0)

# ---- CSV -------------------------------------------------------------------------------------
fields = ["id", "family", "name", "dtype", "shape", "layout", "phase", "work", "work_varies", "expected_zero", "evals",
          "median_us", "min_us", "spread", "per_work_ns", "max_bytes", "timer_limited"] + (["aa_ratio", "aa_half_width", "gate5"] if args.aa_compare else [])
with open(args.out_csv, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(fields)
    for c in cases:
        w.writerow([c["id"], c["family"], c["name"], c["dtype"], c["shape"], c["layout"], c["phase"], c["work"],
                    str(c["work_varies"]).lower(), c["expected_zero"], c["evals"], f"{c['median_ns']/1000:.4g}", f"{c['min_ns']/1000:.4g}",
                    f"{c['spread']:.3f}", "" if math.isnan(c["per_work_ns"]) else f"{c['per_work_ns']:.4g}",
                    f"{c['max_bytes']:.0f}", str(c["timer_limited"]).lower()]
                   + ([f"{c['aa']['g']:.4f}", f"{c['aa']['half']:.4f}", str(c["gate5"]).lower()] if args.aa_compare else []))

# ---- helpers for the markdown ---------------------------------------------------------------
def us(ns):
    v = ns / 1000
    return f"{v:.3g}" if v < 1000 else f"{v:,.0f}"
def rng(vals, fmt=lambda v: f"{v:.3g}"):
    lo, hi = min(vals), max(vals)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}–{fmt(hi)}"

# family-level rows: (family, name, layout), all dtypes and shapes
fam = collections.defaultdict(list)
for c in cases: fam[(c["family"], c["name"], c["layout"])].append(c)
fam_lines = ["| family | case | layout | cases | median µs (range) | per work unit ns (range) | max spread | max bytes | timer-limited |" + (" can gate 5% |" if args.aa_compare else ""),
             "|:--|:--|:--|--:|--:|--:|--:|--:|--:|" + ("--:|" if args.aa_compare else "")]
for (f, n, l), lst in sorted(fam.items()):
    pw = [c["per_work_ns"] for c in lst if not math.isnan(c["per_work_ns"]) and c["per_work_ns"] > 0]
    fam_lines.append(f"| {f} | `{n}` | `{l}` | {len(lst)} | {rng([c['median_ns']/1000 for c in lst])} | "
                     f"{rng(pw) if pw else ''} | {max(c['spread'] for c in lst):.1%} | {max(c['max_bytes'] for c in lst):.0f} | "
                     f"{sum(c['timer_limited'] for c in lst)} |" + (f" {sum(bool(c['gate5']) for c in lst)} |" if args.aa_compare else ""))

# scaling: exponents between fixtures, by case name
def parse_shape(c):
    s, f = c["shape"], c["family"]
    m = re.fullmatch(r"n(\d+)_s(\d+)(?:_m\d+)?", s)
    if f == "sparse" and m: return ("vec", int(m[1]), int(m[2]))
    m = re.fullmatch(r"p(\d+)_g(\d+)_(contiguous|strided)_act(\d+)", s)
    if f == "atom" and m: return ("atom", int(m[1]), (int(m[2]), m[3], int(m[4])))
    m = re.fullmatch(r"p(\d+)_s(\d+)", s)
    if f == "owner" and m: return ("owner", int(m[1]), int(m[2]))
    return None
FLAT, LIN, QUAD = "flat", "linear", "quadratic"
# expectations (acceptance guide §6 target table, the 0.5.x behaviour): (axis size n/p, axis s/active)
EXPECT = {
    "vec": {"read_x[i]": (FLAT, FLAT), "write_x[i]_active": (FLAT, FLAT), "isactive": (FLAT, FLAT), "first_activation": (FLAT, FLAT),
            "sweep_active_update": (FLAT, LIN), "sweep_activecoordinates": (FLAT, LIN), "sweep_activevalues": (FLAT, LIN),
            "sweep_full": (LIN, None), "sweep_via_coordinates": (LIN, None), "sweep_via_x": (LIN, None),
            "dot_vx": (FLAT, LIN), "dot_x": (FLAT, LIN), "dot_via_coordinates": (FLAT, LIN), "dot_xy": (FLAT, LIN),
            "dropzeros!": (FLAT, LIN), "rmul!_finite": (FLAT, LIN), "rmul!_inf": (FLAT, LIN), "fill!_coordinates_zero": (FLAT, LIN),
            "fill!_zero": (FLAT, LIN), "construct": (LIN, None), "copy": (LIN, None), "copyto!_dense": (LIN, None),
            "broadcast_self": (LIN, None), "replace_cheap": (LIN, None), "replace_staged": (LIN, None),
            "copyto!_state": (FLAT, LIN), "copyto!_same_kind": (FLAT, LIN), "broadcast_assign": (FLAT, LIN),
            "fill!_zero_dropzeros": (FLAT, LIN)},
    "atom": {"read_x[i]": (FLAT, FLAT), "write_x[i]_active": (FLAT, FLAT), "isactive": (FLAT, FLAT), "first_activation": (FLAT, FLAT),
             "sweep_active_update": (FLAT, LIN), "sweep_activecoordinates": (FLAT, LIN), "atom_norm": (FLAT, FLAT), "atom_dot": (FLAT, FLAT),
             "atomindices_loop": (FLAT, LIN), "atomindices_view": (FLAT, LIN), "dot_vx": (FLAT, LIN), "dropzeros!": (FLAT, LIN),
             "rmul!_finite": (LIN, None), "rmul!_inf": (LIN, None), "construct": (LIN, None), "copy": (LIN, None),
             "copyto!_dense": (LIN, None), "broadcast_self": (LIN, None), "replace_cheap": (LIN, None), "replace_staged": (LIN, None),
             "fill!_zero": (FLAT, LIN), "fill!_zero_dropzeros": (FLAT, LIN), "copyto!_state": (FLAT, LIN), "copyto!_same_kind": (FLAT, LIN),
             "broadcast_assign": (FLAT, LIN), "_nvalues": (FLAT, LIN), "_values!": (FLAT, LIN), "_setvalues!": (FLAT, LIN), "_restartdot": (FLAT, LIN)},
    "owner": {"read_X[i,j]": (FLAT, FLAT), "write_X[i,j]_active": (FLAT, FLAT), "first_activation": (FLAT, FLAT),
              "sweep_activecoordinates": (FLAT, LIN), "sweep_prox_coordinate!": (FLAT, LIN), "sweep_block_prox_write": (FLAT, LIN), "construct": (QUAD, None), "Matrix": (QUAD, None),
              "fill!_coordinates_zero": (QUAD, FLAT), "fill!_coordinates_zero_dropzeros": (QUAD, None), "fill!_zero": (FLAT, LIN),
              "_values!": (FLAT, LIN), "_setvalues!": (FLAT, LIN), "_restartdot": (FLAT, LIN)},
}
def exps(kind):
    """exponents of the per-call median against the size axis (n, p) and the count axis (s, active), by case."""
    out = collections.defaultdict(lambda: ([], []))
    by = collections.defaultdict(dict)
    for c in cases:
        ps = parse_shape(c)
        if ps is None or ps[0] != kind: continue
        key = (c["name"], c["layout"], c["dtype"])
        if kind == "vec":    by[key][(ps[1], ps[2])] = c["median_ns"]
        elif kind == "atom": by[key][(ps[1], ps[2])] = c["median_ns"]
        else:                by[key][(ps[1], ps[2])] = c["median_ns"]
    for key, d in by.items():
        if kind == "vec":    pairs_n, pairs_s = ((10_000, 100), (1_000_000, 100)), ((1_000_000, 1_000), (1_000_000, 10_000))
        elif kind == "atom":
            pairs_n, pairs_s = None, None
        else:                pairs_n, pairs_s = ((500, 100), (4000, 100)), ((1000, 1000), (1000, 100_000))
        if kind == "atom":
            ns = {}
            for (p, shp), t in d.items(): ns[(p, shp)] = t
            for (p, shp), t in ns.items():
                g, kd, a = shp
                if p == 80_000 and a == 10 and (800_000, (g, kd, 1)) in ns:      # the same number of active groups
                    out[key][0].append(math.log(ns[(800_000, (g, kd, 1))] / t) / math.log(10))
                if a == 1 and (p, (g, kd, 10)) in ns:
                    out[key][1].append(math.log(ns[(p, (g, kd, 10))] / t) / math.log(10))
            continue
        if all(k in d for k in pairs_n):
            out[key][0].append(math.log(d[pairs_n[1]] / d[pairs_n[0]]) / math.log(pairs_n[1][0] / pairs_n[0][0]))
        if all(k in d for k in pairs_s):
            out[key][1].append(math.log(d[pairs_s[1]] / d[pairs_s[0]]) / math.log(pairs_s[1][1] / pairs_s[0][1]))
    return out
def verdict(e, expected):
    if expected is None or e is None: return ""
    return {FLAT: abs(e) <= 0.4, LIN: 0.6 <= e <= 1.5, QUAD: 1.6 <= e <= 2.6}[expected]
scal_lines = ["| group | case | size axis: exponent (range) | expected | count axis: exponent (range) | expected | outside the target |",
              "|:--|:--|:--|:--|:--|:--|:--|"]
deviations = []
for kind, label in (("vec", "sparse (n, s)"), ("atom", "atom (p, active groups)"), ("owner", "owner (p, s)")):
    e = exps(kind)
    byname = collections.defaultdict(lambda: ([], []))
    for (name, layout, dt), (en, es) in e.items():
        byname[name][0].extend(en); byname[name][1].extend(es)
    for name in sorted(byname):
        en, es = byname[name]
        exp = EXPECT[kind].get(name, (None, None))
        bad = []
        if exp[0] and en and not all(verdict(x, exp[0]) for x in en): bad.append("size")
        if exp[1] and es and not all(verdict(x, exp[1]) for x in es): bad.append("count")
        if bad: deviations.append((label, name, bad))
        scal_lines.append(f"| {label} | `{name}` | {rng(en, lambda v: f'{v:.2f}') if en else '–'} | {exp[0] or 'no stated target'} | "
                          f"{rng(es, lambda v: f'{v:.2f}') if es else '–'} | {exp[1] or '–'} | {', '.join(bad) if bad else ''} |")

# not gateable
loud = collections.defaultdict(list)
for c in cases:
    if c["spread"] > args.spread_limit: loud[(c["family"], c["name"], c["layout"])].append(c)
loud_lines = ["| family | case | layout | cases above the limit | of | worst spread |", "|:--|:--|:--|--:|--:|--:|"]
for k, lst in sorted(loud.items(), key=lambda kv: -max(c["spread"] for c in kv[1])):
    loud_lines.append(f"| {k[0]} | `{k[1]}` | `{k[2]}` | {len(lst)} | {len(fam[k])} | {max(c['spread'] for c in lst):.0%} |")

# A/A control: what can gate a 5% slowdown
gate_text = ""; gate_family = []; gate_detail = []
if args.aa_compare:
    byfam = collections.defaultdict(collections.Counter)
    for c in cases:
        k = byfam[c["family"]]; k["cases"] += 1
        if c["timer_limited"]: k["limited"] += 1
        elif c["gate5"]: k["gate"] += 1
        else: k["nogate"] += 1
        if not c["timer_limited"] and c["aa"]["assess"] in ("faster", "slower"): k["alarm"] += 1
    tot = sum(byfam.values(), collections.Counter())
    gate_family = ["| family | cases | timer-limited | can gate 5% | cannot gate 5% | A/A false alarms |", "|:--|--:|--:|--:|--:|--:|"]
    for f_, k in sorted(byfam.items()) + [("all", tot)]:
        gate_family.append(f"| {f_} | {k['cases']} | {k['limited']} | {k['gate']} | {k['nogate']} | {k['alarm']} |")
    nl = tot["cases"] - tot["limited"]
    gate_text = (f"{tot['gate']} of {tot['cases']} cases can gate a 5% slowdown ({tot['limited']} are timer-limited, {tot['nogate']} have an A/A interval wider "
                 f"than ±5%); the A/A comparison flagged {tot['alarm']} of the {nl} cases that are not timer-limited as faster or slower ({tot['alarm']/nl:.1%}; "
                 f"about 5% is expected from a 95% interval).")
    nog = collections.defaultdict(list)
    for c in cases:
        if not c["timer_limited"] and not c["gate5"]: nog[(c["family"], c["name"], c["layout"])].append(c)
    gate_detail = ["| family | case | layout | cannot gate | of | worst A/A half-width |", "|:--|:--|:--|--:|--:|--:|"]
    for k, lst in sorted(nog.items(), key=lambda kv: -max(c["aa"]["half"] for c in kv[1])):
        w_ = max(c["aa"]["half"] for c in lst)
        gate_detail.append(f"| {k[0]} | `{k[1]}` | `{k[2]}` | {len(lst)} | {len(fam[k])} | {'unbounded' if math.isinf(w_) else f'{w_:.0%}'} |")

# timer-limited
tl = collections.defaultdict(list)
for c in cases:
    if c["timer_limited"]: tl[(c["family"], c["name"], c["layout"])].append(c)
tl_lines = ["| family | case | layout | timer-limited | of | median µs (range of the limited ones) |", "|:--|:--|:--|--:|--:|--:|"]
for k, lst in sorted(tl.items()):
    tl_lines.append(f"| {k[0]} | `{k[1]}` | `{k[2]}` | {len(lst)} | {len(fam[k])} | {rng([c['median_ns']/1000 for c in lst])} |")

# allocations
flagged = [c for c in cases if c["expected_zero"] == "true" and c["max_bytes"] > 0]
fl = collections.defaultdict(list)
for c in flagged: fl[(c["family"], c["name"], c["layout"])].append(c)
fl_lines = ["| family | case | layout | cases | of | bytes per call (range) |", "|:--|:--|:--|--:|--:|--:|"]
for k, lst in sorted(fl.items()):
    fl_lines.append(f"| {k[0]} | `{k[1]}` | `{k[2]}` | {len(lst)} | {len(fam[k])} | {rng([c['max_bytes'] for c in lst], lambda v: f'{v:.0f}')} |")
watch = ("fill!_zero", "fill!_zero_dropzeros", "broadcast_assign", "copyto!_state", "copyto!_same_kind", "copyto!_dense",
         "rmul!_finite", "rmul!_inf", "copyto!_unrelated", "broadcast_self", "copyto!_convert", "copyto!_activevalues")
wb = collections.defaultdict(list)
for c in cases:
    if c["name"] in watch and c["expected_zero"] != "true": wb[(c["family"], c["name"])].append(c["max_bytes"])
wb_lines = ["| family | case | cases | max bytes per call |", "|:--|:--|--:|--:|"]
for k, v in sorted(wb.items()): wb_lines.append(f"| {k[0]} | `{k[1]}` | {len(v)} | {max(v):.0f} |")

# cross-check
cross_lines = []
if args.crosscheck:
    cc = {(r["operation"], r["fixture"]): (float(r["min_ns_per_op"]), float(r["median_ns_per_op"])) for r in tsv(args.crosscheck)}
    def find(name, layout_prefix, shape, dtype="Float64"):
        for c in cases:
            if c["family"] == "sparse" and c["name"] == name and c["layout"].startswith(layout_prefix) and c["shape"] == shape and c["dtype"] == dtype:
                return c
    cce = ({(r["operation"], r["fixture"]): (float(r["min_ns_per_op"]), float(r["median_ns_per_op"])) for r in tsv(args.crosscheck_end)}
           if args.crosscheck_end else {})
    cross_lines = ["| operation | fixture | harness case / operations per call | harness median ns | loop before ns | ratio | loop after ns | ratio |",
                   "|:--|:--|:--|--:|--:|--:|--:|--:|"]
    cross_off = []
    for (op, case, lay, shape, per, fx) in (
            ("first_activation_per_coordinate", "first_activation", "absent_distinct_batch", "n1000000_s1000", 256, "Float64 n=1000000 s=1000"),
            ("first_activation_per_coordinate_x16", "first_activation", "absent_distinct_batch_x16", "n1000000_s1000", 4096, "Float64 n=1000000 s=1000"),
            ("sweep_activecoordinates_per_active", "sweep_activecoordinates", "vector", "n1000000_s10000", 10_000, "Float64 n=1000000 s=10000"),
            ("dot_v_x_per_active", "dot_vx", "z0", "n1000000_s10000", 10_000, "Float64 n=1000000 s=10000")):
        c = find(case, lay, shape)
        if c is None or (op, fx) not in cc: cross_lines.append(f"| {op} | {fx} | missing | | | |"); continue
        h = c["median_ns"] / per; i = cc[(op, fx)][1]
        e = cce.get((op, fx), (None, None))[1]
        cross_lines.append(f"| {op} | {fx} | `{c['id']}` / {per} | {h:.3g} | {i:.3g} | {h/i:.2f} | "
                           + (f"{e:.3g} | {h/e:.2f} |" if e else " | |"))
        if abs(h / i - 1) > 0.2 or (e and abs(h / e - 1) > 0.2): cross_off.append((op, h / i, h / e if e else None, [abs(h - i) * per] + ([abs(h - e) * per] if e else [])))
        if op.endswith("_x16"): x16 = dict(h=h, i=i, e=e, per=per)

import json as _json
cross_off = globals().get("cross_off", []); x16 = globals().get("x16")
cross_note = ""
if args.crosscheck and args.crosscheck_end:
    drift = sorted(((max(cce[k][1] / cc[k][1], cc[k][1] / cce[k][1]), k) for k in cc if k in cce), reverse=True)
    dmax, dk = drift[0]
    parts = []
    if cross_off:
        def _dir(r0, r1):
            rs = [r for r in (r0, r1) if r]
            return "the harness figure is the higher one" if all(r > 1 for r in rs) else ("the harness figure is the lower one" if all(r < 1 for r in rs) else "the harness figure is on both sides")
        rows = "; ".join(f"`{op}` {r0:.2f}" + (f" and {r1:.2f}" if r1 else "") + f" ({_dir(r0, r1)})" for op, r0, r1, _ in cross_off)
        parts.append(f"Rows more than 20% apart (ratio before and after): {rows}.")
    else:
        parts.append("No row is more than 20% apart.")
    if dmax > 1.2:
        k0 = cc[dk][1]; k1 = cce[dk][1]
        parts.append(f"The loop itself is not reproducible across processes for some operations: its own two runs (before and after the repetitions, separate processes) "
                     f"differ by up to {dmax:.2f}× (`{dk[0]}`, {dk[1]}: {k0:.3g} and {k1:.3g} ns per operation).")
    if args.restore_probe and x16:
        pr = {}
        for ln in open(args.restore_probe):
            m = re.match(r"nb=(\d+) restore=(\w+) median ns/coord = ([0-9.]+)", ln.strip())
            if m: pr[(int(m.group(1)), m.group(2))] = float(m.group(3))
        if (16, "dropzeros") in pr and (16, "copyto") in pr:
            pd_, pc_ = pr[(16, "dropzeros")], pr[(16, "copyto")]
            parts.append(f"The restore probe runs the loop's own code for 4096 coordinates (restore by zeroing and `dropzeros!`) in a third process of the same job and gets "
                         f"{pd_:.3g} ns per coordinate, against {x16['i']:.3g} and {x16['e']:.3g} in the two runs of the loop; restoring with `copyto!` instead gives {pc_:.3g}. "
                         f"The harness figure for that row, {x16['h']:.3g}, is {abs(x16['h']/pd_-1):.0%} and {abs(x16['h']/pc_-1):.0%} away from the two probe values.")
    parts.append("On an Intel Xeon Silver 4116 node, measured earlier on this branch with the same timed operations, the first-activation ratios were "
                 "0.77 and 0.74 for 256 coordinates and 1.04 and 1.03 for 4096; the cause of the differences is not established. Because each of the two layouts "
                 "missed the 20% criterion on one of the two machines, the `first_activation` rows of `absent_distinct_batch` and `absent_distinct_batch_x16` of the "
                 "sparse family are **diagnostic**: they do not gate.")
    if not [op for op, *_ in cross_off if not op.startswith("first_activation")]:
        parts.append("The `sweep_activecoordinates` and `dot_vx` rows agree within 20% on this machine.")
    cross_note = "\n" + " ".join(parts) + "\n"
replicate_text = ""
if args.replicate_json:
    rp = _json.load(open(args.replicate_json))
    replicate_text = (f" The noise with the inputs held fixed was measured on a single earlier repetition 1 of the same sources on the same "
                      f"machine type (the canary run, harness before the review fixes; `atomindices_loop` left out because its operation "
                      f"changed): of {rp['n']} cases, {rp['within5']:.0%} agreed within 5%, {rp['within10']:.0%} within 10% and "
                      f"{rp['within20']:.0%} within 20% (median deviation {rp['median_abs_dev']:.1%}).")

n_work_varies = sum(c["work_varies"] for c in cases)
if args.aa_compare:
    gate_rule = (" What decides is the A/A control: the same source was measured a second time with the same harness and the same repetition "
                 "numbers (the set `aa`, in separate processes, in alternating order) and the two sets were compared with `compare.py` exactly as a revision "
                 "would be. A case can gate a 5% slowdown when it is not timer-limited and the half-width of its A/A 95% interval for the ratio, "
                 "√(high/low) − 1, is at most 5%, so that a true 5% slowdown would put the whole interval of a comparison below 1. " + gate_text)
    gate_section = ("## What can gate a 5% slowdown (the A/A control)\n\n" + gate_text + " The interval is that of `compare.py` for three paired "
                    "repetitions (a Student interval with two degrees of freedom), so a pass is evidence from one control, not a guarantee; the rows "
                    "of the family table below carry the number of cases that can gate. Cases not listed here and not timer-limited can gate at every fixture. The sparse `first_activation` rows of the layouts `absent_distinct_batch` (256 coordinates per call) and `absent_distinct_batch_x16` (4096) are diagnostic and never gate (see the cross-check below).\n\n"
                    + chr(10).join(gate_family) + "\n\nCases, by family, case and layout, that are neither timer-limited nor able to gate:\n\n" + chr(10).join(gate_detail) + "\n")
else:
    gate_rule = " " + replicate_text
    gate_section = ("## Cases that cannot gate a 5% slowdown (spread above " + f"{args.spread_limit:.0%}" + ")\n\n" + chr(10).join(loud_lines) + "\n")
# ---- the markdown ----------------------------------------------------------------------------
n_main = sum(1 for c in cases if c["family"] != "group_layouts"); n_lay = len(cases) - n_main
timerfloor = f"{min(floors):g}–{max(floors):g} ns"
md = f"""# The iterate family, 0.5.x baseline (issue #420)

The pre-change measurement that the performance gate of the iterate-family design requires (FT24,
numerics C3): every 0.6 change on a hot path reports its cases against this baseline, in the same
session, with `benchmark/compare.py`. The cases are the `sparse` and `atom` groups, the strong-zero,
clear and momentum cases of the `owner` group, the existing `products` group and the existing
`group_layouts` cases (FD-11). The source is the audited 0.5.x commit, so that no FD is in the measured code.

## Identity of the measurement

| | |
|:--|:--|
| Comparison point | ProximalBase 0.5.x at `{args.source_commit}`, the audited commit (the owner baseline's source too), with package source hash `{ident['source_hash']}` (`src/`, `ext/`, `Project.toml`), `source_dirty = {ident['source_dirty'].lower()}` |
| Harness | `benchmark/run.jl` and `benchmark/cases.jl` at commit `{args.harness_commit}` of this branch, hash `{ident['harness_hash']}` (`harness_hash` in each metadata file) |
| Julia | {ident['julia']}, `JULIA_NUM_THREADS=1`, {ident['blas_threads']} BLAS thread ({ident['blas']}), `--check-bounds={ident['check_bounds']}`, BenchmarkTools {ident['benchmarktools']} |
| CPU and OS | {ident['cpu_model']} (`{ident['cpu']}`), {ident['os']} |
| Settings | extended suite, `PB_REPETITION=1,2,3` in fresh processes, {ident['samples_target']} samples of at most {ident['seconds_limit']} s; timer floor {timerfloor} |
| Cases | {len(cases)}: {n_main} in the groups run (`{groups_meta['groups']}`), {n_lay} in the layouts run (`{groups_meta['layouts']}` with `PB_CASES='^group_layouts/'`) |
| Limits | one machine and one operating system (the rows above); the owner baseline `symmetric-owner-0.5.md` was measured on a different machine, so its rows are not comparable row by row; three repetitions; no end-to-end solver sweep; no `p = 8·10⁶` |

```sh
julia benchmark/setup.jl /absolute/path/to/pinned benchmark/results/env-pinned
for r in 1 2 3; do
  PB_SUITE=extended PB_GROUPS={groups_meta['groups']} PB_REPETITION=$r \\
    PB_OUTPUT=benchmark/results/groups PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/pinned \\
    JULIA_NUM_THREADS=1 julia --project=benchmark/results/env-pinned benchmark/run.jl
  PB_SUITE=extended PB_GROUPS=vectors PB_CASES='^group_layouts/' PB_REPETITION=$r \\
    PB_OUTPUT=benchmark/results/layouts PB_EXPECTED_PACKAGE_ROOT=/absolute/path/to/pinned \\
    JULIA_NUM_THREADS=1 julia --project=benchmark/results/env-pinned benchmark/run.jl
done
python3 benchmark/validate.py benchmark/results/groups benchmark/results/groups/validation.json --expected-runs 3
python3 benchmark/validate.py benchmark/results/layouts benchmark/results/layouts/validation.json --expected-runs 3
```

Any edit of `benchmark/run.jl` or `benchmark/cases.jl` changes the harness hash: measure the pinned
source again with the new harness, in the same session as the revision you compare, before using
these numbers as a gate.

## How to read the tables

`median` is the median over the three repetitions of each repetition's median sample time; `min` is
the smallest sample over all repetitions. Times are in microseconds **per call**; `work` is the number of
operations in one call, and the per-work figure is the median divided by it. `spread` is the largest
repetition median over the smallest minus one. Each repetition draws its own inputs (`PB_REPETITION` is a seed), so the
spread mixes the machine's noise with the effect of the inputs: it describes the baseline and does not decide whether a case can gate, because
`compare.py` pairs the same repetition numbers on both sides and the inputs cancel in the pair.{gate_rule} In {n_work_varies} cases the input draw changes the number of
operations in a call between repetitions (all of them in the `products` group, where the draw decides how many entries of an operand are active): `work` is then the median over the
repetitions, the per-work figure the median of the per-repetition ratios, `work_varies` is `true` in the CSV, and the spread includes the change of work. A case with `evals = 1` carries page-fault and GC
noise, so read its `min`.
`bytes` is the largest `@allocated` of one warm call; it is reliable as "zero or not" and not as a size for
large allocations (see #401). A row is timer-limited when `median · evals < 10 · timer floor`: it cannot gate.
The full table, one line per case, is in `iterate-family-0.5.csv` (the same directory); the tables below
group the cases by family, case and layout, over both element types and every fixture.

Spellings that differ between 0.5.x and 0.6 are chosen when the case is built: on 0.5.x, `atomindices(x, k)` is
`x.groups[k]`, `samegroups` is `_samelayout`, and `coordinates(x)` is `x` itself for a `SparseIterate`, so the
`via_coordinates` and `via_x` rows coincide. See the README of this directory for the rows that need a note.

## Cases by family

{chr(10).join(fam_lines)}

## Scaling

The exponent is the slope of log(median per call) against log(size) between two fixtures, so that the work per call
changes only along the axis in question. The size axis is `n` (10⁴ to 10⁶ at `s = 100`), `p` at the same number of
active groups (8·10⁴ with 10% of the groups active against 8·10⁵ with 1%) or the owner's `p` (500 to 4000 at `s = 100`);
the count axis is `s` (10³ to 10⁴ at `n = 10⁶`), the share of active groups (1% to 10% at equal `p`) or the owner's `s`
(10³ to 10⁵ at `p = 1000`). The target is the acceptance guide's §6 table, read for the 0.5.x code, with room for the
cache hierarchy and fixed overheads: flat is |exponent| ≤ 0.4, linear 0.6 to 1.5, quadratic 1.6 to 2.6. A column
outside its target is a statement about what 0.5.x does today (most often a cache effect over a factor of 10 or 100),
not a defect. A curve that is flat over a factor of 100 is a regression alarm and not a proof of O(1). Exceptions that are
known: `AtomIterate` has no `rmul!` method (an O(p) loop), the owner's `fill!(coordinates(X), 0)` is an O(q) loop, and
`coordinates(x) === x` on a `SparseIterate`.

{chr(10).join(scal_lines)}

{gate_section}
## Timer-limited cases

{chr(10).join(tl_lines)}

## Allocation

Cases flagged `zero_alloc` (the operations FT23 lists) that allocate on 0.5.x:

{chr(10).join(fl_lines) if flagged else "None."}

Measured bytes per call of operations whose target is 0 B in the acceptance guide's §6 table but that FT23 does not list
(not flagged; `construct`, `copy` and the staged replace allocate by design):

{chr(10).join(wb_lines)}
"""
if cross_lines:
    md += f"""
## Cross-check with independent loops

Four operations were timed again by a plain `time_ns` loop outside the harness (median of many calls, one thread,
same machine, once before the repetitions and once after them). The ratio is the harness median per operation over
the loop's median.

{chr(10).join(cross_lines)}
{cross_note}"""
open(args.out_md, "w").write(md)
print("wrote", args.out_md, args.out_csv, "|", len(loud), "loud families |", len(tl), "timer-limited families |",
      len(flagged), "flagged cases that allocate | deviations:", deviations)
