import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles


def rows(folder, filename):
    with (folder / filename).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def case(row):
    return row["problem"], row["type"], int(row["stride"])


def pairs(records, divisor=lambda row: 1, side=lambda row: row["side"]):
    measured = defaultdict(dict)
    for row in records:
        rep = int(row["rep"])
        label = side(row)
        assert label not in measured[rep]
        measured[rep][label] = int(row["ns"]) / divisor(row)
    for rep, values in measured.items():
        assert set(values) == {"1", "2"}, (rep, values)
    return measured


def grouped(records, key):
    result = defaultdict(list)
    for row in records:
        result[key(row)].append(row)
    return result


def measure_cell(solves, allocations, events, replay, projection, solver_reps, profile_reps):
    sizes = {row["label"]: row for row in allocations}
    steps = int(sizes["C"]["steps"])
    calls = int(sizes["C"]["scores"])
    assert len(events) == calls
    assert [int(row["state"]) for row in events] == list(range(1, calls + 1))
    for label, row in sizes.items():
        assert row["same_solution"] == "true", (label, row)
        assert int(row["steps"]) == steps and int(row["scores"]) == calls
    for row in solves:
        assert int(row["steps"]) == steps and int(row["scores"]) == calls
        assert math.isfinite(float(row["reference"]))
        assert float(row["reference"]) == float(solves[0]["reference"])

    comparison, control = (pairs([r for r in solves if r["right"] == label]) for label in ("C", "AA"))
    assert set(comparison) == set(control) == set(range(1, solver_reps + 1))
    pr_ms = median(v["1"] for v in comparison.values()) / 1e6
    c_ms = median(v["2"] for v in comparison.values()) / 1e6
    ab_ratios = [v["2"] / v["1"] for v in comparison.values()]
    aa_ratios = [v["2"] / v["1"] for v in control.values()]
    ab_quartiles = quantiles(ab_ratios, n=4, method="inclusive")
    aa_quartiles = quantiles(aa_ratios, n=4, method="inclusive")
    result = {
        "steps": steps, "calls": calls, "reference": solves[0]["reference"],
        "pr_ms": pr_ms, "c_ms": c_ms, "paired_ratio": median(ab_ratios),
        "paired_q1": ab_quartiles[0], "paired_q3": ab_quartiles[2],
        "aa_ratio": median(aa_ratios), "aa_q1": aa_quartiles[0], "aa_q3": aa_quartiles[2],
        "pr_bytes": int(sizes["PR"]["bytes"]), "c_bytes": int(sizes["C"]["bytes"]),
    }

    by_route = grouped(events, lambda row: row["route"])
    assert set(by_route) <= {"fast", "sort", "exact"}
    by_route["sorted"] = by_route["sort"] + by_route["exact"]
    by_route["all"] = events
    assert all(int(row["c_bytes"]) > int(row["pr_bytes"]) for row in by_route["sorted"])
    assert all(int(row["c_bytes"]) == int(row["pr_bytes"]) for row in by_route["fast"])
    [wrapper] = {int(row["pr_bytes"]) for row in events}
    assert result["pr_bytes"] == int(sizes["AA"]["bytes"]) == 0
    result["event_wrapper_bytes"] = wrapper
    by_replay = grouped(replay, lambda row: row["route"])
    result["sorted_at_zero"] = sum(r["zero_point"] == "true" for r in by_route["sorted"])
    for route in ("all", "fast", "sorted", "sort", "exact"):
        route_events = by_route[route]
        n = len(route_events)
        result[f"{route}_n"] = n
        result[f"{route}_bytes"] = sum(int(row["c_bytes"]) - int(row["pr_bytes"]) for row in route_events)
        if not n:
            assert not by_replay[route]
            for measure in ("c_us", "pr_us", "delta_ms", "replay_bytes"):
                result[f"{route}_{measure}"] = 0
            continue
        records = by_replay[route]
        assert all(int(row["calls"]) == n for row in records)
        matched = pairs(records, lambda row: int(row["sweeps"]), lambda row: "1" if row["label"] == "PR" else "2")
        assert set(matched) == set(range(1, profile_reps + 1))
        result[f"{route}_c_us"] = median(v["2"] for v in matched.values()) / (1000 * n)
        result[f"{route}_pr_us"] = median(v["1"] for v in matched.values()) / (1000 * n)
        result[f"{route}_delta_ms"] = median(v["2"] - v["1"] for v in matched.values()) / 1e6
        batch_bytes = [
            int(median(int(row["bytes_per_sweep"]) for row in records if row["label"] == label))
            for label in ("PR", "C")
        ]
        result[f"{route}_replay_bytes"] = batch_bytes[1] - batch_bytes[0]
    sorted_count = result["sorted_n"]
    if sorted_count:
        assert len(projection) == profile_reps
        assert all(int(r["calls"]) == sorted_count for r in projection)
        result["projection_us"] = median(int(r["ns"]) for r in projection) / (1000 * sorted_count)
        result["projection_bytes"] = int(median(int(r["bytes"]) for r in projection))
    else:
        assert not projection
        result["projection_us"] = 0
        result["projection_bytes"] = 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-reps", type=int, default=12)
    parser.add_argument("--profile-reps", type=int, default=15)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("folders", nargs="+", type=Path)
    options = parser.parse_args()
    data = []
    for folder in options.folders:
        files = ("solves.tsv", "allocations.tsv", "route-events.tsv", "route-replays.tsv", "sorted-projection.tsv")
        tables = [grouped(rows(folder, file), case) for file in files]
        seen = set(tables[0])
        assert all(set(table) == seen for table in tables[1:4]) and len(seen) == 8
        for key in sorted(seen):
            all_rows = [table[key] for table in tables]
            [version] = {r["julia"] for table in all_rows for r in table}
            row = dict(zip(("julia", "problem", "type", "stride"), (version, *key)))
            row.update(measure_cell(*all_rows, options.solver_reps, options.profile_reps))
            data.append(row)
            print(version, *key, f"{row['pr_ms']:.3f}/{row['c_ms']:.3f} ms",
                  f"{row['paired_ratio']:.3f}x (A/A {row['aa_ratio']:.3f}x)",
                  f"sorted {row['sorted_n']}/{row['calls']} exact {row['exact_n']}",
                  f"{row['c_bytes']/1024:.1f} KiB/solve",
                  f"sorted {row['sorted_pr_us']:.2f}/{row['sorted_c_us']:.2f} us/call")
    with options.output.open("w") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    main()
