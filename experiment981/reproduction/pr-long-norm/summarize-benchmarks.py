import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def group(records, *fields):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[f] for f in fields)].append(r)
    return groups


def compare(rows, reps):
    times = {}
    for row in rows:
        key = row["right"], int(row["rep"]), row["side"]
        assert key not in times
        elapsed = int(row["ns"]) / int(row.get("calls", "1"))
        assert elapsed > 0 and math.isfinite(elapsed), row
        times[key] = elapsed
    assert len(times) == 4 * reps
    sequence = range(1, reps + 1)
    reference = [times["pairwise", i, "1"] for i in sequence]
    patched = [times["pairwise", i, "2"] for i in sequence]
    ratios = [b / a for a, b in zip(reference, patched)]
    aa = [times["AA", i, "2"] / times["AA", i, "1"] for i in sequence]
    ab_q = quantiles(ratios, n=4, method="inclusive")
    aa_q = quantiles(aa, n=4, method="inclusive")
    return {
        "pr_ns": median(reference), "fixed_ns": median(patched), "paired_ratio": median(ratios),
        "paired_q1": ab_q[0], "paired_q3": ab_q[2],
        "aa": median(aa), "aa_q1": aa_q[0], "aa_q3": aa_q[2],
    }


def summarize_solver(folder, reps, mode):
    cells = group(read(folder / "solves.tsv"), "julia", "type", "problem", "stride")
    alloc = group(read(folder / "allocations.tsv"), "julia", "type", "problem", "stride")
    assert len(cells) == 8 and cells.keys() == alloc.keys()
    for key, rows in sorted(cells.items()):
        stats = compare(rows, reps)
        records = {r["label"]: r for r in alloc[key]}
        assert set(records) == {"PR", "AA", "pairwise"}
        endpoints = {}
        for r in rows:
            build = "PR" if r["side"] == "1" else r["right"]
            identified = (r["steps"], r["scores"], r["reference"], r["reported"])
            assert math.isfinite(float(r["reference"])) and math.isfinite(float(r["reported"]))
            assert endpoints.setdefault(build, identified) == identified
        for build, (steps, scores, _, _) in endpoints.items():
            assert steps == records[build]["steps"] and scores == records[build]["scores"]
        assert endpoints["PR"] == endpoints["AA"]
        assert records["AA"]["same_solution"] == "true"
        record = dict(zip(("julia", "type", "problem", "stride"), key), mode=mode, **stats)
        for name, build in (("pr", "PR"), ("fixed", "pairwise")):
            table = records[build]
            for field in ("bytes", "steps", "scores"):
                record[name + "_" + field] = int(table[field])
            record[name + "_reference"] = endpoints[build][2]
        record["same_solution"] = records["pairwise"]["same_solution"]
        yield record


def summarize_scores(folder, reps, mode):
    cells = group(read(folder / "scores.tsv"), "julia", "type", "case", "n")
    assert len(cells) == 56
    for key, rows in sorted(cells.items(), key=lambda item: (*item[0][:-1], int(item[0][-1]))):
        metrics = compare(rows, reps)
        pr = [r for r in rows if r["side"] == "1"]
        fixed = [r for r in rows if r["side"] == "2" and r["right"] == "pairwise"]
        assert len(pr) == 2 * reps and len(fixed) == reps
        [reference] = {r["reference"] for r in rows}
        [calls] = {int(r["calls"]) for r in rows}
        [pr_bytes] = {int(r["bytes"]) for r in pr}
        [fixed_bytes] = {int(r["bytes"]) for r in fixed}
        [original] = {r["observed"] for r in pr}
        [patched] = {r["observed"] for r in fixed}
        if int(key[-1]) <= 1024:
            assert original == patched
        record = dict(zip(("julia", "type", "case", "n"), key), mode=mode, **metrics)
        record.update(pr_bytes=pr_bytes, fixed_bytes=fixed_bytes, calls=calls,
                      observed_pr=original, observed_fixed=patched, reference=reference)
        yield record


def write(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("folders", type=Path, nargs="+")
    opts = parser.parse_args()
    solve_rows = []
    score_rows = []
    for folder in opts.folders:
        version, mode = folder.name.split("-")
        solver = list(summarize_solver(folder, opts.reps, mode))
        scores = list(summarize_scores(folder, opts.reps, mode))
        assert {r["julia"] for r in solver + scores} == {version}
        solve_rows.extend(solver)
        score_rows.extend(scores)

        def ratios(rows):
            values = [r["paired_ratio"] for r in rows]
            return f"median {median(values):.4f}x, range {min(values):.4f}–{max(values):.4f}x"

        print(folder.name,
              "solver", ratios(solver),
              "short scores", ratios([r for r in scores if int(r["n"]) <= 1024]),
              "long scores", ratios([r for r in scores if int(r["n"]) > 1024]))
    opts.output.mkdir(parents=True, exist_ok=True)
    write(opts.output / "solver-summary.tsv", solve_rows)
    write(opts.output / "score-summary.tsv", score_rows)


if __name__ == "__main__":
    main()
