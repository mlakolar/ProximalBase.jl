# Digest of bench/offset_probe.jl outputs: for one element type, size and backing, one row per output offset and one
# column per implementation: the median time of one call relative to the same implementation at offset 0 (the first
# row gives the times at offset 0 in microseconds), and cand/ref at each offset.
#   python3 bench/offset_digest.py <results/probe dir> <regex on the file name> [T] [n] [backing]
import sys, csv, glob, os, re
root = sys.argv[1]; rx = re.compile(sys.argv[2]); T = sys.argv[3] if len(sys.argv) > 3 else "Float32"
n = sys.argv[4] if len(sys.argv) > 4 else "1048576"; backing = sys.argv[5] if len(sys.argv) > 5 else "huge"
for f in sorted(glob.glob(root + "/*.csv")):
    if not rx.search(os.path.basename(f)): continue
    rows = [r for r in csv.DictReader(open(f)) if r["T"] == T and r["n"] == n and r["backing"] == backing]
    if not rows: continue
    impls = []; offs = []
    for r in rows:
        if r["impl"] not in impls: impls.append(r["impl"])
        if int(r["yoff"]) not in offs: offs.append(int(r["yoff"]))
    t = {(int(r["yoff"]), r["impl"]): float(r["med_ns"]) for r in rows}
    print("== %s  %s n=%s %s" % (os.path.basename(f), T, n, backing))
    print("  %6s " % "yoff" + " ".join("%13s" % i for i in impls) + "   l1 cand/ref  enet cand/ref")
    print("  %6s " % "us@0" + " ".join("%13.2f" % (t[(0, i)] / 1e3) for i in impls))
    for o in offs:
        print("  %6d " % o + " ".join("%13.2f" % (t[(o, i)] / t[(0, i)]) for i in impls) +
              "   %11.2f  %13.2f" % (t[(o, "l1/cand")] / t[(o, "l1/ref")], t[(o, "enet/cand")] / t[(o, "enet/ref")]))
