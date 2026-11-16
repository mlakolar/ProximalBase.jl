# For every stratum, element type and size: over the output offsets 16..256 of bench/offset_probe.jl, the median over
# processes of candidate/reference per offset, then the largest such ratio with its offset, and the number of offsets
# whose ratio is above 1.05, for NormL1 and ElasticNet.
#   python3 bench/offset_band.py <results/probe dir> [more dirs...]
import sys, csv, glob, os, re, statistics as st
fams = ("l1", "enet"); names = {"l1": "NormL1", "enet": "ElasticNet", "l1g": "NormL1 with a step array"}
dirs = []
for a in sys.argv[1:]:
    if a.startswith("--families="): fams = tuple(a.split("=")[1].split(","))
    else: dirs.append(a)
files = [f for d in dirs for f in sorted(glob.glob(d + "/*.csv"))]
acc = {}
for f in files:
    m = re.match(r"(.*?)-[a-z]\d\d-\d\d-(julia[\d.]+)-", os.path.basename(f)); s = m.group(1) + "-" + m.group(2)
    t = {(r["T"], r["n"], int(r["yoff"]), r["impl"]): float(r["med_ns"]) for r in csv.DictReader(open(f)) if r["backing"] == "huge"}
    for (T, n, o, im), v in t.items():
        if im.endswith("/cand") and (16 <= o <= 256 or o == 0):
            fam = im.split("/")[0]; acc.setdefault((s, T, n, fam), {}).setdefault(o, []).append(v / t[(T, n, o, fam + "/ref")])
print("| stratum | type | entries | " + " | ".join("%s: largest candidate/master (offset), offsets above 1.05, value at offset 0" % names[f] for f in fams) + " |"); print("|:--|:--|--:|" + ":--|" * len(fams))
for s in sorted(set(k[0] for k in acc)):
    for T in ("Float32", "Float64"):
        for n in ("1048576", "16384", "1024"):
            cells = []
            for fam in fams:
                d = acc.get((s, T, n, fam))
                if not d: cells.append(""); continue
                med = {o: st.median(v) for o, v in d.items() if o != 0}; o = max(med, key=med.get)
                cells.append("%.2f (+%d), %d of %d, %.2f" % (med[o], o, sum(v > 1.05 for v in med.values()), len(med), st.median(d[0])))
            if cells[0]: print("| %s | %s | %s | %s |" % (s, T, n, " | ".join(cells)))
