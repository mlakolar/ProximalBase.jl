# Digest of bench/alloc_probe.jl (ordinary allocations): per CPU kind, (1) where the output lands relative to the
# input, per length and allocation pattern: the share of trials with (pointer(y) - pointer(x)) mod 4096 in the band
# 16..256 and the share whose arrays are on huge pages; (2) per function and length, the candidate/reference ratio of
# the trials (median over rounds of one call): its quantiles, the share of trials above 1.05, the same split by band,
# the reference and candidate times, and the range of the A/A control.
#   python3 bench/alloc_digest.py <directory with *-alloc-p*.csv> [--trials]
import sys, csv, glob, os, re, statistics as st, collections
root = sys.argv[1]
by = {}
for f in sorted(glob.glob(root + "/*-alloc-p*.csv")):
    kind = os.path.basename(f).split("-")[0]
    proc = os.path.basename(f)
    for r in csv.DictReader(open(f)):
        k = (proc, r["T"], int(r["n"]), r["pattern"], int(r["trial"]))
        d = by.setdefault(kind, {}).setdefault(k, {"yoff": int(r["yoff"]), "huge": (int(r["xhuge_kb"]) > 0 and int(r["yhuge_kb"]) > 0) if "xhuge_kb" in r else (int(r["xhuge_pct"]) > 0 and int(r["yhuge_pct"]) > 0)})
        d[(r["func"], r["impl"])] = float(r["med_ns"])
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
inband = lambda d: 16 <= d["yoff"] <= 256
for kind, tr in sorted(by.items()):
    procs = len(set(k[0] for k in tr))
    print("\n## %s (%d processes, %d trials)\n" % (kind, procs, len(tr)))
    print("Placement: share of trials with the output 16 to 256 bytes after the input modulo 4096 (both element types), and on huge pages\n")
    pats = sorted(set(k[3] for k in tr)); ns = sorted(set(k[2] for k in tr))
    print("| n | " + " | ".join(pats) + " | all | huge pages |\n|--:|" + "--:|" * (len(pats) + 2))
    for n in ns:
        cells = []
        for p in pats:
            v = [inband(d) for k, d in tr.items() if k[2] == n and k[3] == p]
            cells.append("%d%%" % round(100 * sum(v) / len(v)))
        v = [inband(d) for k, d in tr.items() if k[2] == n]; h = [d["huge"] for k, d in tr.items() if k[2] == n]
        print("| %d | %s | %d%% | %d%% |" % (n, " | ".join(cells), round(100 * sum(v) / len(v)), round(100 * sum(h) / len(h))))
    for func in ("l1", "enet"):
        for T in ("Float32", "Float64"):
            print("\n%s, %s: candidate/reference per trial\n" % (func, T))
            print("| n | ref ns (median) | cand ns (median) | ratio min | median | 90% | max | trials > 1.05 | in band: median, max (trials) | outside: median, max (trials) | control min–max |\n|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|:--|")
            for n in ns:
                ds = [d for k, d in tr.items() if k[1] == T and k[2] == n and (func, "ref") in d]
                if not ds: continue
                rt = [d[(func, "cand")] / d[(func, "ref")] for d in ds]
                ct = [d[(func, "refC")] / d[(func, "ref")] for d in ds]
                ib = [d[(func, "cand")] / d[(func, "ref")] for d in ds if inband(d)]; ob = [d[(func, "cand")] / d[(func, "ref")] for d in ds if not inband(d)]
                fmt = lambda v: "%.2f, %.2f (%d)" % (st.median(v), max(v), len(v)) if v else "none"
                print("| %d | %.0f | %.0f | %.2f | %.2f | %.2f | %.2f | %d%% | %s | %s | %.2f–%.2f |" % (
                    n, st.median(d[(func, "ref")] for d in ds), st.median(d[(func, "cand")] for d in ds), min(rt), st.median(rt), q(rt, 0.9), max(rt),
                    round(100 * sum(x > 1.05 for x in rt) / len(rt)), fmt(ib), fmt(ob), min(ct), max(ct)))
    if "--trials" in sys.argv:
        print("\nTrials with a ratio above 1.05:\n")
        for k, d in sorted(tr.items()):
            for func in ("l1", "enet"):
                if (func, "ref") in d and d[(func, "cand")] / d[(func, "ref")] > 1.05:
                    print("  %s %s n=%d %s trial %d: offset %d, %.0f -> %.0f ns (%.2f), control %.2f" % (k[1], func, k[2], k[3], k[4], d["yoff"], d[(func, "ref")],
                          d[(func, "cand")], d[(func, "cand")] / d[(func, "ref")], d[(func, "refC")] / d[(func, "ref")]))
