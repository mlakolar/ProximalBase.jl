#!/usr/bin/env python3
"""Digest of benchmark/ridge910.jl runs made by slurm/native.sbatch in both role orders (roles are named: base,
the candidates, control = the base compiled again). For every candidate: the cases whose median per-round ratio
to base is above 1.05 (below 0.95) in both orders, and the ratios of the cells named on the command line.
  python3 bench/ridge910_digest.py <results/nativeN> <stratum prefix, e.g. xeon2640v4-julia1.13.4-FINAL> [regex of cases to list]"""
import csv, sys, statistics as st, os, re
root, prefix = sys.argv[1:3]
show = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
def load(order):
    d = os.path.join(root, f"{prefix}-ridge910-{order}")
    if not os.path.isfile(os.path.join(d, "DONE")): return None
    rows = {}
    for r in csv.DictReader(open(os.path.join(d, "raw.csv"))):
        rows.setdefault((r["case"], r["round"]), {})[r["role"]] = float(r["min_ns"])
    out = {}
    for (case, _), v in rows.items():
        for role, t in v.items():
            if role != "base": out.setdefault(case, {}).setdefault(role, []).append(t / v["base"])
        out[case].setdefault("_ns", []).append(v["base"])
    return {c: {k: st.median(x) for k, x in o.items()} for c, o in out.items()}
f, r = load("fwd"), load("rev")
if f is None or r is None:
    print(f"{prefix} ridge910: incomplete (fwd {'ok' if f else 'missing'}, rev {'ok' if r else 'missing'})"); sys.exit()
roles = [k for k in next(iter(f.values())) if not k.startswith("_")]
cases = [c for c in f if c in r]
print(f"{prefix} ridge910: {len(cases)} cases; ratio to base, forward / reversed order")
for role in roles:
    slow = [(c, f[c][role], r[c][role]) for c in cases if f[c][role] > 1.05 and r[c][role] > 1.05]
    fast = [(c, f[c][role], r[c][role]) for c in cases if f[c][role] < 0.95 and r[c][role] < 0.95]
    one = [(c, f[c][role], r[c][role]) for c in cases if (f[c][role] > 1.05) != (r[c][role] > 1.05)]
    print(f"  {role}: {len(slow)} slower in both orders, {len(fast)} faster in both, {len(one)} above 1.05 in one order only")
    for c, a, b in slow: print(f"    SLOWER {c}: {a:.3f} / {b:.3f} (control {f[c]['control']:.3f} / {r[c]['control']:.3f}; base {f[c]['_ns']:.4g} ns)")
    for c, a, b in one: print(f"    one order {c}: {a:.3f} / {b:.3f} (control {f[c]['control']:.3f} / {r[c]['control']:.3f})")
    if show:
        for c in cases:
            if show.search(c): print(f"    {c}: {a_fmt(f[c][role])} / {a_fmt(r[c][role])}  base {f[c]['_ns']:.4g} ns" if False else f"    {c}: {f[c][role]:.3f} / {r[c][role]:.3f}  base {f[c]['_ns']:.4g} ns")
