"""Synthetic receipt and failure-preservation controls; no GPU is required."""
import csv
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis import group_samples, paired_summary
from validate import validate

HERE = Path(__file__).resolve().parent
FIELDS = ("n", "p", "seed", "precision", "tolerance", "check_every", "scenario",
          "repetition", "backend", "source", "seconds", "iterations", "maxiter",
          "residual", "reference_residual", "objective_gap", "objective_limit", "status")


def fixture(root, repetitions=5):
    root.mkdir()
    (root / "environment.txt").write_text(
        f"mode: full\nrepetitions: {repetitions}\nsource: abc123\n")
    rows = []
    for (n, p), seed, precision in itertools.product(
            ((400, 40), (4000, 400), (10000, 1000)), (103, 104, 105), ("Float32", "Float64")):
        tolerances = (1e-4, 1e-6) if precision == "Float64" else (1e-4,)
        for tol, frequency, scenario, rep, backend in itertools.product(
                tolerances, (1, 10), ("resident", "transfers"),
                range(1, repetitions + 1), ("cpu", "gpu")):
            rows.append(dict(n=n, p=p, seed=seed, precision=precision, tolerance=tol,
                             check_every=frequency, scenario=scenario, repetition=rep,
                             backend=backend, source="abc123", seconds=0.002 if backend == "cpu" else 0.001,
                             iterations=10, maxiter=100, residual=0, reference_residual=0,
                             objective_gap=0, objective_limit=1e-6, status="success"))
    write_rows(root, rows)
    return rows


def write_rows(root, rows):
    with (root / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_report(name, root):
    result = subprocess.run([sys.executable, str(HERE / name), str(root)],
                            capture_output=True, text=True, check=True)
    return result.stdout


class AnalysisControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pb-gpu-analysis-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for shape in (400, 10000):
            (self.root / f"profile-{shape}.log").write_text("")

    def test_complete_and_nondefault_repetition_counts(self):
        for repetitions in (5, 7):
            full = self.root / "full"
            if full.exists():
                full.rename(self.root / "previous")
            fixture(full, repetitions)
            self.assertEqual(len(validate(full)), 216 * repetitions)
            output = run_report("report_tables.py", self.root)
            self.assertIn(f"Planned repetitions per backend and seed: {repetitions}", output)
            row = next(line for line in output.splitlines() if line.startswith("| 400 × 40 | Float32 | resident |"))
            self.assertEqual(row.split("|")[6].strip(), "2.000–2.000")
            self.assertEqual(row.split("|")[7].strip(), f"0/{6 * repetitions}")

    def test_failed_pair_suppresses_both_reporters(self):
        full = self.root / "full"
        rows = fixture(full)
        rows[0].update(status="iteration_cap", iterations=100, residual=1)
        write_rows(full, rows)
        output = run_report("report_tables.py", self.root)
        row = next(line for line in output.splitlines() if line.startswith("| 400 × 40 | Float32 | resident |"))
        self.assertEqual(row.split("|")[6].strip(), "NA")
        self.assertEqual(row.split("|")[7].strip(), "1/30")
        run_report("summarize.py", full)
        summary = (full / "summary.md").read_text()
        self.assertIn("| NA | 1/10 |", summary)
        self.assertIn("Failed samples: 1.", summary)

    def test_empty_design_and_too_few_repetitions(self):
        for repetitions in (0, 1, 4):
            root = self.root / str(repetitions)
            fixture(root, repetitions)
            with self.assertRaisesRegex(ValueError, "five"):
                validate(root)

    def test_nonfinite_failure_remains_visible(self):
        full = self.root / "full"
        rows = fixture(full)
        rows[0].update(status="nonfinite", residual="nan", objective_gap="nan")
        write_rows(full, rows)
        output = run_report("report_tables.py", self.root)
        self.assertIn("nonfinite", output)
        row = next(line for line in output.splitlines() if line.startswith("| 400 × 40 | Float32 | resident |"))
        self.assertEqual(row.split("|")[6].strip(), "NA")
        self.assertEqual(row.split("|")[7].strip(), "1/30")

    def test_invalid_reference_and_objective_claims(self):
        full = self.root / "full"
        rows = fixture(full)
        for value in ("nan", "inf", -1, 1e-5):
            rows[0]["reference_residual"] = value
            write_rows(full, rows)
            with self.assertRaisesRegex(ValueError, "Reference"):
                validate(full)
        rows[0]["reference_residual"] = 0
        for value in ("nan", "inf", -1):
            rows[0]["objective_limit"] = value
            write_rows(full, rows)
            with self.assertRaisesRegex(ValueError, "objective"):
                validate(full)

    def test_missing_duplicate_and_unpaired_samples(self):
        full = self.root / "full"
        rows = fixture(full)
        write_rows(full, rows[1:])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate(full)
        write_rows(full, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate(full)
        group = next(iter(group_samples(rows).values()))
        self.assertIsNone(paired_summary(group[1:], 5)["ratio"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
