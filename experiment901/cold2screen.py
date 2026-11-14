#!/usr/bin/env python3
"""Convert raw.csv of l2_cold_paths_x.jl / l2_norms.jl (roles X, Y, XC) to the screen901 format, so that
summarize901.py can pool campaign runs:  cold2screen.py <run dir> <label of X> <label of Y> <out dir>
Several shard directories of one process order may be given as a comma-separated <run dir> list."""
import csv, os, sys
runs, lx, ly, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
name = {"X": lx, "XC": lx + "#aa", "Y": ly, "A": lx, "Acontrol": lx + "#aa", "B": ly}
with open(os.path.join(out, "raw.csv"), "w", newline="") as f, open(os.path.join(out, "alloc.csv"), "w", newline="") as g:
    w = csv.writer(f); w.writerow(["case", "role", "round", "ns"])
    a = csv.writer(g); a.writerow(["case", "role", "bytes"]); seen = set()
    for r in runs:
        for row in csv.DictReader(open(os.path.join(r, "raw.csv"))):
            role = row.get("role", row.get("side"))
            w.writerow([row["case"], name[role], row["round"], row["median_ns"]])
            k = (row["case"], role)
            if k not in seen and "memory" in row:
                seen.add(k); a.writerow([row["case"], name[role], row["memory"]])
prov = open(os.path.join(runs[0], "provenance.txt")).read()
with open(os.path.join(out, "provenance.txt"), "w") as f:
    f.write(prov)
    f.write(f"roles {lx},{lx}#aa,{ly}\n")
    for r in runs:
        p = os.path.join(r, "job_meta.txt")
        if os.path.exists(p):
            f.write(open(p).read())
