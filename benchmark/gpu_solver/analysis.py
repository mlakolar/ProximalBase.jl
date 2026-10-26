"""Shared grouping and complete-pair acceptance for the GPU reports."""
from collections import defaultdict
import statistics as st

GROUP_KEYS = ("n", "p", "precision", "seed", "scenario", "tolerance", "check_every")


def group_samples(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)
    return groups


def paired_summary(group, repetitions):
    records = {b: [r for r in group if r["backend"] == b] for b in ("cpu", "gpu")}
    times = {b: [float(r["seconds"]) * 1000 for r in records[b]
                 if r["status"] == "success"] for b in records}
    failed = sum(r["status"] != "success" for r in group)
    expected = set(range(1, repetitions + 1))
    complete = not failed and all(
        len(records[b]) == repetitions and
        {int(r["repetition"]) for r in records[b]} == expected for b in records)
    ratio = st.median(times["cpu"]) / st.median(times["gpu"]) if complete else None
    return dict(times=times, failed=failed, total=len(group), ratio=ratio)
