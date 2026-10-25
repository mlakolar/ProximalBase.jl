"""Summarize successful paired samples; retain and count all failed runs."""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "timings.csv").open()))
groups = defaultdict(list)
for row in rows:
    groups[tuple(row[k] for k in ("n", "p", "precision", "seed", "scenario", "tolerance", "check_every"))].append(row)
lines = ["# Common-accuracy solver timings", "", "Times include stopping checks. Failed samples are excluded from speedups and counted below.", "Variability is the range of repetition times; seed-specific medians remain separate.", "CPU/GPU is a ratio of medians, not a claim of a universal crossover.", "", "| n × p | type | seed | scenario | tolerance | check | CPU median [min,max] ms | GPU median [min,max] ms | CPU/GPU | failures / total |", "|---|---|---|---|---|---|---|---|---|---|"]
failures = 0
for key, group in groups.items():
    times = {b: [float(r["seconds"]) * 1000 for r in group if r["backend"] == b and r["status"] == "success"] for b in ("cpu", "gpu")}
    failed = sum(r["status"] != "success" for r in group)
    failures += failed
    def fmt(values):
        return f"{st.median(values):.3f} [{min(values):.3f},{max(values):.3f}]" if values else "no success"
    # Compare only if every planned paired repetition succeeded.
    ratio = f'{st.median(times["cpu"]) / st.median(times["gpu"]):.3f}' if not failed and times["cpu"] and times["gpu"] else "NA"
    n, p, precision, seed, scenario, tol, frequency = key
    lines.append(f"| {n} × {p} | {precision} | {seed} | {scenario} | {tol} | {frequency} | {fmt(times['cpu'])} | {fmt(times['gpu'])} | {ratio} | {failed}/{len(group)} |")
lines.extend(["", f"Recorded samples: {len(rows)}. Failed samples: {failures}.", "", "Setup and first-call costs are recorded separately in setup-*.toml. First-call time includes compilation and execution; it is not pure compilation time.", "The transfer scenario includes workspace construction, upload, solve, and final coefficient download. Resident verification downloads are outside the clock.", "See environment.txt and the saved manifest for hardware and versions. Profiling and explicit-GC diagnostics are separate runs.", ""])
(root / "summary.md").write_text("\n".join(lines))
print(f"{len(rows)} samples, {failures} failures; wrote {root / 'summary.md'}")
