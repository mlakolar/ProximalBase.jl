# Digest of bench/sparse_probe.jl outputs: one row per element type and operation, one column per K (the number of
# value sets cycled through): the reference time of one call in microseconds, copy excluded, and candidate/reference.
#   python3 bench/sparse_digest.py <results/probe dir> [regex on the file name]
import sys, csv, glob, os, re
root = sys.argv[1]; rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
for f in sorted(glob.glob(root + "/*.csv")):
    if not rx.search(os.path.basename(f)): continue
    rows = list(csv.DictReader(open(f)))
    if not rows: continue
    Ks = sorted(set(int(r["K"]) for r in rows))
    print("==", os.path.basename(f)); print("  %-16s " % "" + "  ".join("K=%-3d ref us cand ctl" % k for k in Ks))
    seen = []
    for r in rows:
        key = (r["T"], r["op"])
        if key in seen: continue
        seen.append(key); s = "  %-8s %-7s " % key
        for k in Ks:
            d = {q["impl"]: float(q["med_ns"]) - float(q["copy_ns"]) for q in rows if (q["T"], q["op"]) == key and int(q["K"]) == k}
            if "ref" not in d: s += " " * 24; continue
            s += "%9.2f %5.2f %4.2f   " % (d["ref"] / 1e3, d["cand"] / d["ref"], d["refC"] / d["ref"])
        print(s)
