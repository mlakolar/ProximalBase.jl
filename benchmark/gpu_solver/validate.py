"""Validate saved design coverage and numerical claims without rerunning solves."""
import csv
import itertools
import math
import re
import sys
from pathlib import Path


def validate(root):
    root = Path(root)
    environment = (root / "environment.txt").read_text()
    mode = re.search(r"^mode: (small|full)$", environment, re.M).group(1)
    repetitions = int(re.search(r"^repetitions: (\d+)$", environment, re.M).group(1))
    source = re.search(r"^source: (\w+)$", environment, re.M).group(1)
    expected = set()
    shapes = ((400, 40),) if mode == "small" else ((400, 40), (4000, 400), (10000, 1000))
    seeds = (103,) if mode == "small" else (103, 104, 105)
    for (n, p), seed, precision in itertools.product(shapes, seeds, ("Float32", "Float64")):
        tolerances = (1e-4, 1e-6) if mode == "full" and precision == "Float64" else (1e-4,)
        for tolerance, frequency, scenario, repetition, backend in itertools.product(
                tolerances, (1, 10), ("resident", "transfers"), range(1, repetitions + 1), ("cpu", "gpu")):
            expected.add((n, p, seed, precision, tolerance, frequency, scenario, repetition, backend))
    rows = list(csv.DictReader((root / "timings.csv").open()))
    observed = set()
    for r in rows:
        key = (int(r["n"]), int(r["p"]), int(r["seed"]), r["precision"],
               float(r["tolerance"]), int(r["check_every"]), r["scenario"],
               int(r["repetition"]), r["backend"])
        if key in observed:
            raise ValueError(f"Duplicate sample: {key}")
        observed.add(key)
        if r["source"] != source:
            raise ValueError("Source differs from environment receipt")
        if not math.isfinite(float(r["seconds"])) or float(r["seconds"]) <= 0:
            raise ValueError("Invalid timing")
        if not 0 <= int(r["iterations"]) <= int(r["maxiter"]):
            raise ValueError("Invalid iteration count")
        if float(r["reference_residual"]) > 1e-10:
            raise ValueError("Reference missed its target")
        if r["status"] == "success":
            if not 0 <= float(r["residual"]) <= float(r["tolerance"]):
                raise ValueError("Successful sample missed its independent target")
            if not abs(float(r["objective_gap"])) <= float(r["objective_limit"]):
                raise ValueError("Successful sample failed objective comparison")
        elif r["status"] not in {"iteration_cap", "nonfinite", "residual_failure", "objective_failure"}:
            raise ValueError("Unknown termination status")
    if observed != expected:
        raise ValueError(f"Incomplete design: {len(expected-observed)} missing, {len(observed-expected)} unexpected")
    print(f"Validated {len(rows)} unique samples and numerical success claims ({mode}).")
    return rows


if __name__ == "__main__":
    validate(sys.argv[1])
