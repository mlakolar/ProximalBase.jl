import csv
from collections import Counter
from decimal import Decimal
from pathlib import Path
import sys


def records(file):
    with file.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def verify(folder, version, mode):
    baseline = folder / f"PR-{version}-{mode}"
    fixed = folder / f"pairwise-{version}-{mode}"

    def artifact(prefix, suffix):
        return prefix.with_name(prefix.name + "-" + suffix)

    for build, checks in ((baseline, 11372), (fixed, 11390)):
        assert artifact(build, "package.exit").read_text().strip() == "0"
        assert str(checks) in artifact(build, "package.log").read_text()

    before = records(artifact(baseline, "norm-oracle.tsv"))
    after = records(artifact(fixed, "norm-oracle.tsv"))
    assert len(before) == len(after) == 28
    for a, b in zip(before, after):
        for field in ("type", "case", "n", "exponent", "reference"):
            assert a[field] == b[field], (version, mode, field, a, b)
        assert b["status"] == "pass", b

    before = records(artifact(baseline, "stress.tsv"))
    after = records(artifact(fixed, "stress.tsv"))
    assert len(before) == len(after) == 18
    for a, b in zip(before, after):
        assert a["case"] == b["case"] and a["type"] == b["type"], (a, b)
        if b["case"].startswith(("published_", "reconstructed_")):
            assert b["status"] == "pass", b
        else:
            assert b["status"] == a["status"], (a, b)

    before = records(artifact(baseline, "oracle.tsv"))
    after = records(artifact(fixed, "oracle.tsv"))
    assert len(before) == len(after) == 185
    grades = {"pass": 0, "documented_range": 0, "roundoff_at_zero": 1, "accuracy": 1, "material": 2, "fail": 3}
    for a, b in zip(before, after):
        for field in ("type", "case", "path", "n", "exact_squared", "reference"):
            assert a[field] == b[field], (version, mode, field, a, b)
        assert grades[b["status"]] <= grades[a["status"]], (a, b)
        if a["status"] == "documented_range":
            assert b["status"] == "documented_range", (a, b)
        elif b["frozen_bound_ratio"]:
            assert Decimal(b["frozen_bound_ratio"]) <= 1, b

    assert "observed = 8192.0f0, expected = 8192.0f0" in artifact(fixed, "longsum.log").read_text()
    old_counts = Counter(r["status"] for r in before)
    new_counts = Counter(r["status"] for r in after)
    print(version, mode, "11390 package checks and 28 independent norm checks pass;",
          "baseline full oracle", dict(old_counts), "paired", dict(new_counts))


folder = Path(sys.argv[1])
for version in ("1.10.13", "1.13.4"):
    for mode in ("default", "O0"):
        verify(folder, version, mode)
