#!/usr/bin/env python3
"""Summarize screen901.jl runs (orchestrator-owned).

  python3 summarize901.py --num A --den base [--also ref] [--csv out.csv] RUN_DIR [RUN_DIR ...]

Each RUN_DIR is one process (one role order). For every case and run the paired ratio num/den is the median
over rounds of ns[num]/ns[den] in the same round; the A/A ratio is the median of ns[X#aa]/ns[X], X the run's
first role. A case is "slower" only if its ratio exceeds the threshold in every run that holds it
(both orders), and "beyond A/A" if it also exceeds (1 + that run's |A/A - 1|) in every run.
"""
import argparse, csv, math, os, statistics, sys
from collections import defaultdict


def load(run):
    roles = None
    prov = {}
    with open(os.path.join(run, "provenance.txt")) as f:
        for line in f:
            k, _, v = line.strip().partition(" ")
            if k == "roles":
                roles = v.split(",")
            if k in ("host", "cpu", "slurm_job", "affinity", "elapsed_s"):
                prov[k] = v
            if k == "tree":
                prov.setdefault("trees", []).append(v)
    ns = defaultdict(lambda: defaultdict(dict))  # case -> round -> role -> ns
    with open(os.path.join(run, "raw.csv")) as f:
        for r in csv.DictReader(f):
            ns[r["case"]][int(r["round"])][r["role"]] = float(r["ns"])
    alloc = defaultdict(dict)
    if os.path.exists(os.path.join(run, "alloc.csv")):
        with open(os.path.join(run, "alloc.csv")) as f:
            for r in csv.DictReader(f):
                alloc[r["case"]][r["role"]] = int(r["bytes"])
    return roles, prov, ns, alloc


def paired(ns_case, a, b):
    rs = [rd[a] / rd[b] for rd in ns_case.values() if a in rd and b in rd and rd[b] > 0]
    return statistics.median(rs) if rs else None


def med(ns_case, a):
    v = [rd[a] for rd in ns_case.values() if a in rd]
    return statistics.median(v) if v else None


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def category(case):
    p = case.split("/")
    if p[0] == "Group":   # Group/T/m/layout/regime/op
        return f"Group/m{p[2][1:]}/{p[4]}"
    if p[0] == "Norm":
        return f"Norm/{p[3]}"
    return p[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", required=True)
    ap.add_argument("--den", default="base")
    ap.add_argument("--also", action="append", default=[], help="other roles to report against --den")
    ap.add_argument("--csv")
    ap.add_argument("--threshold", type=float, default=1.05)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    runs = []
    for r in a.runs:
        roles, prov, ns, alloc = load(r)
        runs.append(dict(dir=r, roles=roles, prov=prov, ns=ns, alloc=alloc, X=roles[0]))
    print("runs:")
    for r in runs:
        print(f"  {r['dir']}: X={r['X']} roles={','.join(r['roles'])} host={r['prov'].get('host')} "
              f"cpu={r['prov'].get('cpu')} job={r['prov'].get('slurm_job')} core={r['prov'].get('affinity')} "
              f"elapsed={r['prov'].get('elapsed_s')}s")

    cases = sorted(set().union(*[set(r["ns"]) for r in runs]))
    rows = []
    aa_all = defaultdict(list)
    for c in cases:
        row = dict(case=c, cat=category(c), ratios=[], aa=[], also={k: [] for k in a.also},
                   ns_den=None, ns_num=None, alloc_num=None, alloc_den=None)
        for i, r in enumerate(runs):
            if c not in r["ns"]:
                continue
            q = paired(r["ns"][c], a.num, a.den)
            aa = paired(r["ns"][c], r["X"] + "#aa", r["X"])
            if q is not None:
                row["ratios"].append(q)
                row["aa"].append(aa)
                aa_all[i].append(aa)
            for k in a.also:
                v = paired(r["ns"][c], k, a.den)
                if v is not None:
                    row["also"][k].append(v)
            if row["ns_den"] is None:
                row["ns_den"], row["ns_num"] = med(r["ns"][c], a.den), med(r["ns"][c], a.num)
                row["alloc_num"] = r["alloc"].get(c, {}).get(a.num)
                row["alloc_den"] = r["alloc"].get(c, {}).get(a.den)
        if row["ratios"]:
            rows.append(row)

    print("\nA/A per run (|X#aa/X - 1|): median, p90, max")
    for i, v in aa_all.items():
        d = sorted(abs(x - 1) for x in v if x)
        if d:
            print(f"  run {i}: {statistics.median(d):.4f}  {d[int(0.9 * (len(d) - 1))]:.4f}  {d[-1]:.4f}  (n={len(d)})")

    nrun = max(len(r["ratios"]) for r in rows)
    slow = [r for r in rows if len(r["ratios"]) == nrun and min(r["ratios"]) > a.threshold]
    beyond = [r for r in slow if all(q > 1 + abs((aa or 1) - 1) for q, aa in zip(r["ratios"], r["aa"]))]
    slow2 = [r for r in rows if len(r["ratios"]) == nrun and min(r["ratios"]) > 1.02]
    alloc_reg = [r for r in rows if (r["alloc_num"] or 0) > (r["alloc_den"] or 0)]
    print(f"\n{a.num}/{a.den}: cases {len(rows)}, runs per case {nrun}; slower in every run: "
          f">1.02: {len(slow2)}, >{a.threshold}: {len(slow)} ({len(beyond)} beyond A/A); "
          f"allocation regressions: {len(alloc_reg)}")
    allr = [statistics.median(r["ratios"]) for r in rows]
    print(f"geomean {gmean(allr):.4f}, min {min(allr):.3f}, max {max(allr):.3f}")
    for k in a.also:
        v = [statistics.median(r["also"][k]) for r in rows if r["also"][k]]
        if v:
            print(f"  {k}/{a.den} on the same cases: geomean {gmean(v):.4f}, max {max(v):.3f}")

    print("\ngeomean by category (num/den" + "".join(f" | {k}/{a.den}" for k in a.also) + "), worst:")
    bycat = defaultdict(list)
    for r in rows:
        bycat[r["cat"]].append(r)
    for cat in sorted(bycat):
        rs = bycat[cat]
        g = gmean([statistics.median(r["ratios"]) for r in rs])
        w = max(rs, key=lambda r: min(r["ratios"]))
        extra = "".join(f" | {gmean([statistics.median(r['also'][k]) for r in rs if r['also'][k]]):.3f}" for k in a.also)
        print(f"  {cat:32s} {g:.3f}{extra}   worst {w['case']} {'/'.join(f'{q:.3f}' for q in w['ratios'])}")

    print(f"\ntop {a.top} by min ratio over runs:")
    print(f"  {'case':52s} {'ratios (runs)':>20s} {'A/A':>14s} {'den ns':>9s} {'num ns':>9s}" +
          "".join(f" {k + '/' + a.den:>12s}" for k in a.also))
    for r in sorted(rows, key=lambda r: -min(r["ratios"]))[: a.top]:
        print(f"  {r['case']:52s} {'/'.join(f'{q:.3f}' for q in r['ratios']):>20s} "
              f"{'/'.join(f'{q:.3f}' for q in r['aa'] if q):>14s} {r['ns_den']:9.1f} {r['ns_num']:9.1f}" +
              "".join(f" {'/'.join(f'{q:.3f}' for q in r['also'][k]):>12s}" for k in a.also))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "category", "ratios", "aa", "den_ns", "num_ns", "alloc_den", "alloc_num"] + [f"{k}/{a.den}" for k in a.also])
            for r in rows:
                w.writerow([r["case"], r["cat"], ";".join(f"{q:.4f}" for q in r["ratios"]),
                            ";".join(f"{q:.4f}" for q in r["aa"] if q), f"{r['ns_den']:.2f}", f"{r['ns_num']:.2f}",
                            r["alloc_den"], r["alloc_num"]] + [";".join(f"{q:.4f}" for q in r["also"][k]) for k in a.also])


if __name__ == "__main__":
    main()
