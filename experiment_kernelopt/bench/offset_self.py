# Largest slowdown of a loop against itself at output offset 0, over the offsets 16..256 bytes and the processes of
# bench/offset_probe.jl, per CPU kind, for one Julia version, at 2^20 / 16,384 / 1024 entries (both element types).
#   python3 bench/offset_self.py <results/probe5> [julia version, default 1.13.4]
import sys, csv, glob, os, re
v = sys.argv[2] if len(sys.argv) > 2 else "1.13.4"
acc = {}
for f in sorted(glob.glob(sys.argv[1] + "/*julia%s-*.csv" % v)):
    kind = os.path.basename(f).split("-")[0]
    t = {(r["T"], r["n"], int(r["yoff"]), r["impl"]): float(r["med_ns"]) for r in csv.DictReader(open(f)) if r["backing"] == "huge"}
    for (T, n, o, im), x in t.items():
        if 16 <= o <= 256: acc.setdefault((kind, n, im), []).append(x / t[(T, n, 0, im)])
rows = [("master, NormL1 `prox!`", "l1/ref"), ("A3, NormL1 `prox!`", "l1/cand"), ("master, ElasticNet `prox!`", "enet/ref"), ("A3, ElasticNet `prox!`", "enet/cand"),
        ("master, SqrNormL2 `prox!` (one multiplication per entry)", "ridge/ref"), ("master, NormL2 `prox!`", "l2/ref"), ("`y .= a .* x`", "scale"), ("`copyto!`", "copy")]
print("   | | Zen 3 | Broadwell | Skylake-SP |\n   |:--|:--|:--|:--|")
for name, im in rows:
    print("   | %s | %s |" % (name, " | ".join(" / ".join("%.2f" % max(acc[(k, n, im)]) for n in ("1048576", "16384", "1024")) for k in ("epyc7513", "xeon2640v4", "xeon4116"))))
