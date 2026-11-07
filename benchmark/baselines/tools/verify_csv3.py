# Independent recomputation of every figure of the baseline csv from the raw metrics and metadata files (no use of compare.py output).
import csv, sys, math, statistics, collections, tomllib
E, CSV = sys.argv[1], sys.argv[2]
T975 = 4.302652729911275   # Student t, 0.975, 2 degrees of freedom
def load(d):
    R = collections.defaultdict(dict); floor = {}
    for r in (1, 2, 3):
        meta = tomllib.load(open(f"{E}/{d}/julia-1.13.3-rep0{r}-metadata.toml", "rb")); floor[r] = float(meta["timer_floor_ns"])
        for row in csv.DictReader(open(f"{E}/{d}/julia-1.13.3-rep0{r}-metrics.tsv"), delimiter="\t"): R[row["id"]][r] = row
    return R, floor
B = collections.defaultdict(dict); AA = collections.defaultdict(dict); FB = {}; FA = {}
for kind in ("groups", "layouts"):
    rb, fb = load(f"base-{kind}"); ra, fa = load(f"aa-{kind}")
    for i, v in rb.items(): B[i] = v
    for i, v in ra.items(): AA[i] = v
    FB[kind] = fb; FA[kind] = fa
kind_of = {}
for kind in ("groups", "layouts"):
    for i in load(f"base-{kind}")[0]: kind_of[i] = kind
rows = list(csv.DictReader(open(CSV))); bad = collections.Counter(); worst = collections.defaultdict(float)
nchk = collections.Counter(); false_alarm = 0; ngate = 0; ntl = 0; nnt = 0
for r in rows:
    i = r["id"]; L = [B[i][k] for k in (1, 2, 3)]; M = [AA[i][k] for k in (1, 2, 3)]; k = kind_of[i]
    meds = [float(x["median_ns"]) for x in L]; works = [float(x["work"]) for x in L]
    ratios = [m / w for m, w in zip(meds, works) if w > 0]
    exp = {"median_us": statistics.median(meds) / 1000, "min_us": min(float(x["min_ns"]) for x in L) / 1000,
           "spread": max(meds) / min(meds) - 1, "max_bytes": max(float(x["max_bytes"]) for x in L), "work": round(statistics.median(works))}
    if ratios: exp["per_work_ns"] = statistics.median(ratios)
    # A/A control, recomputed from the raw per-repetition means
    lr = [math.log(float(a["mean_ns"]) / float(b["mean_ns"])) for a, b in zip(L, M)]
    mu = statistics.mean(lr); half_log = T975 * statistics.stdev(lr) / math.sqrt(3)
    exp["aa_ratio"] = math.exp(mu); exp["aa_half_width"] = math.exp(half_log) - 1
    tl = any(float(x["median_ns"]) * int(x["evals"]) < 10 * FB[k][int(x["rep"])] for x in L) or any(float(x["median_ns"]) * int(x["evals"]) < 10 * FA[k][int(x["rep"])] for x in M)
    forced = r["family"] == "sparse" and r["name"] == "first_activation" and r["layout"] in ("absent_distinct_batch", "absent_distinct_batch_x16")
    gate = (not tl) and exp["aa_half_width"] <= 0.05 and not forced
    if (r["timer_limited"] == "true") != any(float(x["median_ns"]) * int(x["evals"]) < 10 * FB[k][int(x["rep"])] for x in L): bad["timer_limited(base)"] += 1
    if (r["gate5"] == "true") != gate: bad["gate5"] += 1
    ntl += tl; nnt += (not tl); ngate += gate
    if not tl and (mu - half_log > 0 or mu + half_log < 0): false_alarm += 1
    for c, v in exp.items():
        g = float(r[c]); tol = 6e-4 * max(abs(v), 1e-9) + (5e-4 if c in ("spread", "aa_half_width", "aa_ratio") else 0)
        worst[c] = max(worst[c], abs(g - v) / max(abs(v), 1e-9))
        if abs(g - v) > tol: bad[c] += 1
    if (r["work_varies"] == "true") != (len(set(works)) > 1): bad["work_varies"] += 1
    if r["evals"] != "/".join(map(str, sorted({int(x["evals"]) for x in L}))): bad["evals"] += 1
    for c in ("family", "name", "dtype", "shape", "layout", "phase", "expected_zero"):
        if L[0][c] != r[c]: bad[c] += 1
print("rows", len(rows), "| mismatches:", dict(bad) or "none")
print("worst relative error:", {c: f"{v:.1e}" for c, v in worst.items()})
print(f"timer-limited (either side) {ntl}; not timer-limited {nnt}; can gate {ngate} (csv {sum(r['gate5']=='true' for r in rows)}); A/A conclusive among not timer-limited {false_alarm} ({false_alarm/nnt:.1%})")
