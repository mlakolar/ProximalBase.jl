# Every case that the decision rule calls slower (per-process median ratio > 1.05 in every process of a stratum, the
# A/A control within [0.98, 1.02] in every process), per tree and stratum, split by whether the tree changes code that
# the case runs ("reach"). The cases outside the reach measure the false-positive rate of the rule.
#   python3 bench/slowcells.py <results dir of a campaign> [--list]
import sys, csv, glob, os, re, statistics as st
root = sys.argv[1]; full = "--list" in sys.argv
REACH = {"A3": r"^(l1/(prox|prox_w|prox_g|prox_gw|valprox|valprox_w|block_loop)|enet/)", "V": r"^(l1|l2|group|tinygroup)/(value|valprox)",
         "D3": r"^(l2|group|tinygroup)/", "B925": r"^l1/(prox_g|prox_gw)/", "FINAL": r"^(l1|l2|group|tinygroup|enet)/"}
strata = {}
for d in sorted(glob.glob(root + "/*-p*")):
    if os.path.exists(d + "/DONE"): strata.setdefault(re.sub(r"-p\d+$", "", os.path.basename(d)), []).append(d)
for s, ds in sorted(strata.items()):
    per = {}; ref = None; bad = set()
    for di, d in enumerate(ds):
        for r in csv.DictReader(open(d + "/meta.csv")):
            ref = r["ref"]
            if r["status"] != "ok": bad.add((r["case"], r["impl"]))
        cur = {}
        for r in csv.reader(open(d + "/raw.csv")):
            if r[0] == "case": continue
            cur.setdefault((r[0], r[2]), {})[r[1]] = float(r[4])
        acc = {}
        for (c, _), row in cur.items():
            if ref in row:
                for im, v in row.items(): acc.setdefault((c, im), []).append(v / row[ref])
        for k, v in acc.items(): per.setdefault(k, []).append(st.median(v))
    ctl = ref + "C"
    for tr in sorted(set(k[1] for k in per) - {ref, ctl}):
        rx = re.compile(REACH.get(tr, "."))
        inn, out, nin, nout = [], [], 0, 0
        for (c, im), v in sorted(per.items()):
            if im != tr or (c, im) in bad: continue
            k = per.get((c, ctl), [1.0]); ok = min(k) >= 0.98 and max(k) <= 1.02
            r = bool(rx.search(c)); nin += r; nout += not r
            if min(v) > 1.05 and ok: (inn if r else out).append("%s %.2f-%.2f" % (c, min(v), max(v)))
        print("%s %s (%d procs): in reach %d of %d, outside %d of %d" % (s, tr, len(ds), len(inn), nin, len(out), nout))
        if full:
            for x in inn: print("    in  ", x)
            for x in out: print("    out ", x)
