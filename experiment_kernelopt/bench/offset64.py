# candidate/reference at the output offset +64 bytes (and at offset 0), per stratum, type and size, range over the
# processes of bench/offset_probe.jl (probe2, probe5) and, for 2^20 and 16384 entries, of bench/layout_probe.jl (probe1).
#   python3 bench/offset64.py <results/m3>
import sys, csv, glob, os, re
root = sys.argv[1]
acc = {}
def add(s, T, n, fam, off, v): acc.setdefault((s, T, n, fam, off), []).append(v)
def stratum(f):
    m = re.match(r"(.*?)-[a-z]\d\d-\d\d-(julia[\d.]+)-", os.path.basename(f)); return m.group(1) + "-" + m.group(2)
for d in ("probe2", "probe5"):
    for f in sorted(glob.glob("%s/%s/*.csv" % (root, d))):
        t = {(r["T"], r["n"], int(r["yoff"]), r["impl"]): float(r["med_ns"]) for r in csv.DictReader(open(f)) if r["backing"] == "huge"}
        for (T, n, o, im) in list(t):
            if im.endswith("/cand") and o in (0, 64):
                fam = im.split("/")[0]; add(stratum(f), T, n, fam, o, t[(T, n, o, im)] / t[(T, n, o, fam + "/ref")])
for f in sorted(glob.glob("%s/probe1/*.csv" % root)):
    for r in csv.DictReader(open(f)):
        if r["layout"] in ("huge/x0/y64", "huge/x0/y0"):
            add(stratum(f), r["T"], r["n"], r["f"], 64 if r["layout"].endswith("y64") else 0, float(r["cand_med"]) / float(r["ref_med"]))
def rng(v): return "%.2f–%.2f (%d)" % (min(v), max(v), len(v)) if v else ""
print("| stratum | type | entries | NormL1 at +64 | NormL1 at 0 | ElasticNet at +64 | ElasticNet at 0 |"); print("|:--|:--|--:|:--|:--|:--|:--|")
for s in sorted(set(k[0] for k in acc)):
    for T in ("Float32", "Float64"):
        for n in ("1048576", "16384", "1024"):
            g = lambda fam, o: rng(acc.get((s, T, n, fam, o), []))
            if (s, T, n, "l1", 64) in acc: print("| %s | %s | %s | %s | %s | %s | %s |" % (s, T, n, g("l1", 64), g("l1", 0), g("enet", 64), g("enet", 0)))

# grouped rows for the report (REPORT.md section 8.7)
if "--report" in sys.argv:
    names = {"epyc7513": "Zen 3", "xeon2640v4": "Broadwell", "xeon4116": "Skylake-SP"}
    B, F32, F64 = ("Float32", "Float64"), ("Float32",), ("Float64",)
    M, K16, K1, ALL = ("1048576",), ("16384",), ("1024",), ("1048576", "16384", "1024")
    groups = [("epyc7513", v, T, n) for v in ("1.13.4", "1.10.13") for (T, n) in ((F32, M), (F64, M), (F32, K16), (F64, K16), (B, K1))]
    groups += [("xeon2640v4", "1.13.4", B, ALL), ("xeon2640v4", "1.10.13", B, ALL), ("xeon4116", "1.13.4", B, M), ("xeon4116", "1.13.4", B, K16 + K1), ("xeon4116", "1.10.13", B, ALL)]
    lab = {M: "2^20", K16: "16,384", K1: "1024", ALL: "1024 to 2^20", K16 + K1: "1024 and 16,384"}
    def r(kind, v, Ts, ns, fam, o):
        x = [y for T in Ts for n in ns for y in acc.get(("%s-julia%s" % (kind, v), T, n, fam, o), [])]
        return "%.2f–%.2f" % (min(x), max(x)) if min(x) != max(x) and "%.2f" % min(x) != "%.2f" % max(x) else "%.2f" % min(x)
    print("\n   | CPU, Julia | type | entries | NormL1 at +64 | at 0 | ElasticNet at +64 | at 0 |\n   |:--|:--|--:|:--|:--|:--|:--|")
    for (kind, v, Ts, ns) in groups:
        print("   | %s, %s | %s | %s | %s | %s | %s | %s |" % (names[kind], v, "both" if len(Ts) == 2 else Ts[0], lab[ns], r(kind, v, Ts, ns, "l1", 64), r(kind, v, Ts, ns, "l1", 0),
              r(kind, v, Ts, ns, "enet", 64), r(kind, v, Ts, ns, "enet", 0)))
