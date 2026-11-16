# Summary of bench/offset_probe.jl outputs over the files of a stratum (CPU kind and Julia version): for each element
# type and size, the range over the output offsets 16..256 bytes (the band) and over the processes of (a) the time of
# an implementation relative to its own time at offset 0 and (b) candidate/reference, and candidate/reference
# outside the band (offsets 0 and 512 and above).
#   python3 bench/offset_summary.py <results/probe dir> [more dirs...]
import sys, csv, glob, os, re
files = [f for d in sys.argv[1:] for f in sorted(glob.glob(d + "/*.csv"))]
strata = {}
for f in files:
    m = re.match(r"(.*?)-[a-z]\d\d-\d\d-(julia[\d.]+)-", os.path.basename(f))
    strata.setdefault(m.group(1) + "-" + m.group(2), []).append(f)
band = lambda o: 16 <= o <= 256
out = lambda o: o == 0 or o >= 512
def rng(v): return "%.2f–%.2f" % (min(v), max(v)) if v else ""
print("| stratum (processes) | type | entries | master NormL1, own | A3 NormL1, own | A3/master NormL1 in band | outside | master ElasticNet, own | A3 ElasticNet, own | A3/master ElasticNet in band | outside |")
print("|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|")
for s, fs in sorted(strata.items()):
    for T in ("Float32", "Float64"):
        for n in ("1048576", "16384", "1024"):
            acc = {k: [] for k in ("l1ref", "l1cand", "l1r", "l1o", "enref", "encand", "enr", "eno")}
            np = 0
            for f in fs:
                rows = [r for r in csv.DictReader(open(f)) if r["T"] == T and r["n"] == n and r["backing"] == "huge"]
                t = {(int(r["yoff"]), r["impl"]): float(r["med_ns"]) for r in rows}
                offs = sorted(set(o for o, _ in t))
                if not offs or (4064, "enet/cand") not in t: continue
                np += 1
                for o in offs:
                    for fam, a, b in (("l1", "l1", "l1"), ("enet", "en", "en")):
                        ref, cand = t[(o, fam + "/ref")], t[(o, fam + "/cand")]
                        if band(o):
                            acc[a + "ref"].append(ref / t[(0, fam + "/ref")]); acc[a + "cand"].append(cand / t[(0, fam + "/cand")]); acc[a + "r"].append(cand / ref)
                        elif out(o): acc[a + "o"].append(cand / ref)
            if np == 0: continue
            print("| %s (%d) | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (s, np, T, n, rng(acc["l1ref"]), rng(acc["l1cand"]), rng(acc["l1r"]), rng(acc["l1o"]),
                  rng(acc["enref"]), rng(acc["encand"]), rng(acc["enr"]), rng(acc["eno"])))
