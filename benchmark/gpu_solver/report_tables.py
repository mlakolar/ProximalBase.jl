"""Recreate compact tables from retained timings and diagnostic logs (Python 3)."""
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from validate import read_design, validate
from analysis import group_samples, paired_summary

root = Path(sys.argv[1])
rows = validate(root / "full")
_, repetitions, _ = read_design(root / "full")
groups = defaultdict(list)
for key, group in group_samples(rows).items():
    n, p, precision, seed, scenario, tolerance, frequency = key
    if float(tolerance) == 1e-4 and int(frequency) == 1:
        groups[(n, p, precision, scenario)].append(paired_summary(group, repetitions))
print(f"\nTimes to residual 1e-4, checked every update. Planned repetitions per backend and seed: {repetitions}. "
      "Ranges span seed-specific medians of successful samples. Failed samples are counted; "
      "CPU/GPU ratios require every planned pair in every seed to succeed.\n")
print("| Shape | Type | Scenario | CPU ms | GPU ms | CPU/GPU | failures / total |")
print("|---|---|---|---|---|---|---|")
def span(values):
    return f"{min(values):.3f}–{max(values):.3f}" if values else "no success"

for key, values in groups.items():
    spans = [span([st.median(v["times"][backend]) for v in values if v["times"][backend]])
             for backend in ("cpu", "gpu")]
    ratios = [v["ratio"] for v in values]
    spans.append(span(ratios) if all(r is not None for r in ratios) else "NA")
    spans.append(f'{sum(v["failed"] for v in values)}/{sum(v["total"] for v in values)}')
    n, p, precision, scenario = key
    print(f"| {n} × {p} | {precision} | {scenario} | " + " | ".join(spans) + " |")
print("\nMaximum independent residuals:")
def diagnostic_max(values):
    return max(values) if all(math.isfinite(v) for v in values) else "nonfinite"

for tol in sorted({row["tolerance"] for row in rows}, key=float, reverse=True):
    print(tol, diagnostic_max([float(row["residual"]) for row in rows if row["tolerance"] == tol]))
print("Maximum absolute objective gap:", diagnostic_max([abs(float(row["objective_gap"])) for row in rows]))
print("\nOriginal diagnostic runs (separate from benchmark samples):\n")
print("| Shape | Check | Instrument/GC policy | Median ms | Min–max ms |")
print("|---|---|---|---|---|")
for shape in (400, 10000):
    samples = defaultdict(list)
    frequency = None
    for line in (root / f"profile-{shape}.log").read_text().splitlines():
        if line.startswith("ANOMALY,"):
            _, freq, _, _, instrument, seconds, _ = line.split(",")
            samples[(freq, instrument)].append(float(seconds) * 1000)
        elif line.startswith("CUDA_TIMED,"):
            frequency = re.search(r"frequency=(\d+)", line).group(1)
        elif line.startswith("(value =") and frequency is not None:
            samples[(frequency, "cuda_timed")].append(float(re.search(r", time = ([\d.e+-]+)", line).group(1)) * 1000)
            frequency = None
    for (freq, instrument), values in samples.items():
        print(f"| {shape} | {freq} | {instrument} | {st.median(values):.3f} | {min(values):.3f}–{max(values):.3f} |")
