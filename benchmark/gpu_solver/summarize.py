"""Summarize successful paired samples; retain and count all failed runs."""
from validate import read_design, validate
from analysis import group_samples, paired_summary
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = validate(root)
_, repetitions, _ = read_design(root)
groups = group_samples(rows)
lines = ["# Common-accuracy solver timings", "", "Times include stopping checks. Failed samples are excluded from speedups and counted below.", "Variability is the range of repetition times; seed-specific medians remain separate.", "CPU/GPU is a ratio of medians, not a claim of a universal crossover.", "", "| n × p | type | seed | scenario | tolerance | check | CPU median [min,max] ms | GPU median [min,max] ms | CPU/GPU | failures / total |", "|---|---|---|---|---|---|---|---|---|---|"]
failures = 0
for key, group in groups.items():
    paired = paired_summary(group, repetitions)
    times, failed = paired["times"], paired["failed"]
    failures += failed
    def fmt(values):
        return f"{st.median(values):.3f} [{min(values):.3f},{max(values):.3f}]" if values else "no success"
    # Compare only if every planned paired repetition succeeded.
    ratio = f'{paired["ratio"]:.3f}' if paired["ratio"] is not None else "NA"
    n, p, precision, seed, scenario, tol, frequency = key
    lines.append(f"| {n} × {p} | {precision} | {seed} | {scenario} | {tol} | {frequency} | {fmt(times['cpu'])} | {fmt(times['gpu'])} | {ratio} | {failed}/{len(group)} |")
lines.extend(["", f"Recorded samples: {len(rows)}. Failed samples: {failures}.", "", "Setup and first-call costs are recorded separately in setup-*.toml. First-call time includes compilation and execution; it is not pure compilation time.", "The transfer scenario includes workspace construction, upload, solve, and final coefficient download. Resident verification downloads are outside the clock.", "See environment.txt and the saved manifest for hardware and versions. Profiling and explicit-GC diagnostics are separate runs.", ""])
(root / "summary.md").write_text("\n".join(lines))
print(f"{len(rows)} samples, {failures} failures; wrote {root / 'summary.md'}")
