#!/usr/bin/env python3
"""Both-orders digest of ProximalBase's own paired harnesses (benchmark/*.jl) run by slurm/native.sbatch.
For every case: the median over rounds of candidate/base in the forward run (base is role X) and in the reversed
run (candidate is role X), with the A/A control XC/X of each run. A case is called slower (faster) when the ratio
is above 1.05 (below 0.95) in both orders.
  python3 bench/native_digest.py <results/nativeN> <stratum prefix, e.g. xeon2640v4-julia1.13.4-FINAL2> <harness>"""
import csv, sys, statistics as st, os
root, prefix, harness = sys.argv[1:4]
def load(order):
    d = os.path.join(root, f"{prefix}-{harness}-{order}")
    if not os.path.isfile(os.path.join(d, "DONE")): return None
    rows = {}
    for r in csv.DictReader(open(os.path.join(d, "raw.csv"))):
        rows.setdefault((r["case"], r["round"]), {})[r["role"]] = float(r["median_ns"])
    out = {}
    for (case, _), v in rows.items():
        if not all(k in v for k in ("X", "Y", "XC")): continue
        o = out.setdefault(case, {"yx": [], "aa": [], "x": []})
        o["yx"].append(v["Y"] / v["X"]); o["aa"].append(v["XC"] / v["X"]); o["x"].append(v["X"])
    return {c: (st.median(o["yx"]), st.median(o["aa"]), st.median(o["x"])) for c, o in out.items()}
f, r = load("fwd"), load("rev")
if f is None or r is None:
    print(f"{prefix} {harness}: incomplete (fwd {'ok' if f else 'missing'}, rev {'ok' if r else 'missing'})"); sys.exit()
slow, fast, n = [], 0, 0
for c in sorted(f):
    if c not in r: continue
    n += 1
    a, b = f[c][0], 1 / r[c][0]            # candidate/base in each order
    if a > 1.05 and b > 1.05: slow.append((c, a, b, f[c][1], r[c][1], f[c][2]))
    elif a < 0.95 and b < 0.95: fast += 1
print(f"{prefix} {harness}: {n} cases; faster in both orders {fast}; slower in both orders {len(slow)}")
for c, a, b, aa1, aa2, x in slow:
    print(f"  SLOWER {c}: candidate/base {a:.3f} (forward), {b:.3f} (reversed); A/A {aa1:.3f}, {aa2:.3f}; base {x:.1f} ns")
if "--all" in sys.argv:
    for c in sorted(f):
        if c in r: print(f"  {c}: {f[c][0]:.3f} / {1/r[c][0]:.3f}  (A/A {f[c][1]:.3f} / {r[c][1]:.3f})")
