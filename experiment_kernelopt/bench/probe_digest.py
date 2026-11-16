# Digest of bench/layout_probe.jl outputs: per file (node, Julia version, process) and element type, one line per
# layout with the reference time of one call (median over rounds, microseconds), the candidate/reference ratio and
# the control/reference ratio, for NormL1 and ElasticNet.
#   python3 bench/probe_digest.py <results/probe dir> [regex on the file name] [n]
import sys, csv, glob, os, re
root = sys.argv[1]; rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else "."); n = sys.argv[3] if len(sys.argv) > 3 else "1048576"
for f in sorted(glob.glob(root + "/*.csv")):
    if not rx.search(os.path.basename(f)): continue
    rows = [r for r in csv.DictReader(open(f)) if r["n"] == n]
    if not rows: continue
    print("==", os.path.basename(f))
    for T in ("Float32", "Float64"):
        lay = []
        for r in rows:
            if r["T"] == T and r["layout"] not in lay: lay.append(r["layout"])
        print("  %s  %-16s %5s %5s %6s %6s | %8s %6s %6s | %8s %6s %6s" % (T, "layout", "x%4k", "y%4k", "xhugeK", "yhugeK", "l1 ref", "cand", "ctl", "enet ref", "cand", "ctl"))
        for l in lay:
            d = {r["f"]: r for r in rows if r["T"] == T and r["layout"] == l}
            a = d["l1"]; s = "  %s  %-16s %5s %5s %6s %6s" % (T, l, a["xoff4k"], a["yoff4k"], a["xhugekb"], a["yhugekb"])
            for fn in ("l1", "enet"):
                r = d[fn]; ref = float(r["ref_med"])
                s += " | %8.1f %6.2f %6.2f" % (ref / 1e3, float(r["cand_med"]) / ref, float(r["refC_med"]) / ref)
            print(s)
