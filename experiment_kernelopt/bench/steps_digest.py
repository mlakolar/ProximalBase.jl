# Digest of a campaign of the suite "steps" (NormL1 with a step per entry): per stratum (CPU kind and Julia version)
# and operation, one row per length with the reference time and, for every tree and for the A/A control, the range
# over the processes of the per-process median of the paired ratios tree/reference. A tree is called slower in a cell
# when that median is above 1.05 in every process and the control is within [0.98, 1.02] in every process; faster
# when it is below 0.95 in every process with the same control condition. A cell is not classified ("invalid") when an
# output differed from the reference or a call allocated, when the tree or the control is missing from a process, or
# when the stratum has fewer than two processes. With --summary only the counts are printed.
#   python3 bench/steps_digest.py <results dir of the campaign> [--summary] [--ops=prox_g,prox_gw]
import sys, csv, glob, os, re, statistics as st
root = sys.argv[1]; summary = "--summary" in sys.argv
ops_opt = [a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--ops=")]
strata = {}
for d in sorted(glob.glob(root + "/*-p*")):
    if os.path.exists(d + "/DONE"): strata.setdefault(re.sub(r"-p\d+$", "", os.path.basename(d)), []).append(d)
for s, ds in sorted(strata.items()):
    per = {}; refns = {}; ref = None; bad = set()
    for d in ds:
        for r in csv.DictReader(open(d + "/meta.csv")):
            ref = r["ref"]
            if r["status"] != "ok" or r["alloc_bytes"] != "0": bad.add((r["case"], r["impl"]))
        cur = {}
        for r in csv.reader(open(d + "/raw.csv")):
            if r[0] == "case": continue
            cur.setdefault((r[0], r[2]), {})[r[1]] = float(r[4])
        acc = {}; accref = {}
        for (c, _), row in cur.items():
            if ref in row:
                accref.setdefault(c, []).append(row[ref])
                for im, v in row.items(): acc.setdefault((c, im), []).append(v / row[ref])
        for k, v in acc.items(): per.setdefault(k, []).append(st.median(v))
        for c, v in accref.items(): refns.setdefault(c, []).append(st.median(v))
    ctl = ref + "C"
    trees = sorted(set(k[1] for k in per) - {ref, ctl})
    cases = sorted(set(k[0] for k in per), key=lambda c: (c.split("/")[1], c.split("/")[2], int(c.split("/")[3][1:])))
    ops = ops_opt[0] if ops_opt else sorted(set(c.split("/")[1] for c in cases))
    print("\n## %s (%d processes; reference %s; status or allocation problems: %d)" % (s, len(ds), ref, len(bad)))
    count = {}
    for op in ops:
        for T in ("Float64", "Float32"):
            rows = [c for c in cases if c.split("/")[1] == op and c.split("/")[2] == T]
            if not rows: continue
            if not summary:
                print("\n%s, %s\n\n| n | %s ns | %s | control |\n|--:|--:|%s:--|" % (op, T, ref, " | ".join(trees), ":--|" * len(trees)))
            for c in rows:
                n = int(c.split("/")[3][1:]); k = per.get((c, ctl), [])
                complete = len(ds) >= 2 and len(k) == len(ds) and (c, ctl) not in bad and (c, ref) not in bad
                ok = complete and min(k) >= 0.98 and max(k) <= 1.02
                cells = []
                for tr in trees:
                    v = per.get((c, tr), [])
                    valid = complete and len(v) == len(ds) and (c, tr) not in bad
                    mark = ""
                    if not valid: mark = " (invalid)"
                    elif ok and min(v) > 1.05: mark = " **slower**"
                    elif ok and max(v) < 0.95: mark = " faster"
                    elif not ok and (min(v) > 1.05 or max(v) < 0.95): mark = " (control)"
                    key = (tr, "1-15" if n < 16 else "16+", op)
                    cnt = count.setdefault(key, [0, 0, 0, 0, 0]); cnt[0] += 1
                    cnt[1] += mark == " **slower**"; cnt[2] += mark == " faster"; cnt[3] += mark == " (control)"; cnt[4] += mark == " (invalid)"
                    cells.append(("%.3f–%.3f%s" % (min(v), max(v), mark)) if v else "missing (invalid)")
                if not summary:
                    print("| %d | %.1f | %s | %s |" % (n, st.median(refns[c]), " | ".join(cells), ("%.3f–%.3f" % (min(k), max(k))) if k else "missing"))
    print("\nCounts (cells, slower, faster, undecided for the control, invalid), both element types:\n")
    print("| tree | lengths | operation | cells | slower | faster | control unclean | invalid |\n|:--|:--|:--|--:|--:|--:|--:|--:|")
    for (tr, rng, op), c in sorted(count.items()):
        print("| %s | %s | %s | %d | %d | %d | %d | %d |" % (tr, rng, op, c[0], c[1], c[2], c[3], c[4]))
