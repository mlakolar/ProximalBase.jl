"""Recreate compact tables from retained timings and diagnostic logs (Python 3)."""
import csv
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from validate import validate

root = Path(sys.argv[1])
rows = validate(root / "full")
paired = defaultdict(lambda: defaultdict(list))
for row in rows:
    key = tuple(row[k] for k in ("n", "p", "precision", "tolerance", "check_every", "scenario", "seed"))
    if row["status"] == "success":
        paired[key][row["backend"]].append(float(row["seconds"]) * 1000)
groups = defaultdict(list)
for key, backends in paired.items():
    if key[3] == "0.0001" and key[4] == "1" and all(backends[b] for b in ("cpu", "gpu")):
        cpu, gpu = (st.median(backends[b]) for b in ("cpu", "gpu"))
        groups[key[:3] + (key[5],)].append((cpu, gpu, cpu / gpu))
print("\nTimes to residual 1e-4, checked every update. Each range spans the three seed medians; each median uses five repetitions. CPU/GPU ratios are paired within seed.\n")
print("| Shape | Type | Scenario | CPU ms | GPU ms | CPU/GPU |")
print("|---|---|---|---|---|---|")
for key, values in groups.items():
    spans = [f"{min(v[i] for v in values):.3f}–{max(v[i] for v in values):.3f}" for i in range(3)]
    n, p, precision, scenario = key
    print(f"| {n} × {p} | {precision} | {scenario} | " + " | ".join(spans) + " |")
print("\nMaximum independent residuals:")
for tol in sorted({row["tolerance"] for row in rows}, key=float, reverse=True):
    print(tol, max(float(row["residual"]) for row in rows if row["tolerance"] == tol))
print("Maximum absolute objective gap:", max(abs(float(row["objective_gap"])) for row in rows))
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
