#!/usr/bin/env python3
"""Compare scalar evidence for the original vector fixtures in the PS #508 archive."""
import argparse
import collections
import csv
import gzip
import itertools
import math
import struct
from pathlib import Path


def reader(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def bits(number):
    return struct.pack("!d", float(number))


def prior_radius(row):
    radius = float(row["r"])
    if row["type"] == "Float32":
        return struct.unpack("!f", struct.pack("!f", radius))[0]
    return radius


def compare(old, new, version, bounds):
    previous = (row for row in reader(old) if row["shape"] == "1")
    historical_groups = {row["group"] for row in reader(old) if row["shape"] == "1"}
    replay = (row for row in reader(new) if row["design"] == "A"
              and row["route"] != "input" and row["group"] in historical_groups
              and not (row["group"] == "contract" and row["case"] == "empty"))
    counts = collections.Counter()
    maximum = 0.0
    for index, (old_row, new_row) in enumerate(itertools.zip_longest(previous, replay), 1):
        assert old_row and new_row, (index, old_row, new_row)
        assert tuple(old_row[k] for k in ("type", "n", "group", "case", "api")) == tuple(
            new_row[k] for k in ("type", "n", "group", "case", "route")
        ), (index, old_row, new_row)
        assert bits(prior_radius(old_row)) == bits(new_row["r"]), (index, old_row, new_row)
        counts["vector_cases"] += 1
        earlier, current = float(old_row["mass_minus_r"]), float(new_row["margin"])
        counts["stored_margin_matches"] += bits(earlier) == bits(current)
        counts["exact_feasibility_matches"] += old_row["output_true_feasible"] == str(current <= 0).lower()
        counts["pb_feasibility_matches"] += old_row["output_accept"] == new_row["pb_feasible"]
        maximum = max(maximum, abs(earlier - current))
    ordered = ("vector_cases", "stored_margin_matches", "exact_feasibility_matches", "pb_feasibility_matches")
    return (version, bounds, *(counts[key] for key in ordered), maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    arguments = parser.parse_args()
    print("version\tbounds\tvector_cases\tstored_margin_matches\texact_feasibility_matches\tpb_feasibility_matches\tmax_abs_margin_difference")
    for version in ("1.10.13", "1.13.4"):
        for bounds in ("auto", "yes"):
            cell = f"{version}-{bounds}"
            old, = (arguments.old / cell).glob("*/pb-projections.tsv*")
            recorded_pin = arguments.new / f"ps508-pin-{cell}.tsv.gz"
            replayed = recorded_pin if recorded_pin.exists() else arguments.new / f"master-{cell}.tsv.gz"
            print(*compare(old, replayed, version, bounds), sep="\t")


if __name__ == "__main__":
    main()
