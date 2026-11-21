#!/usr/bin/env python3
"""Verify stored replay vectors and exact dyadic sign and cap comparisons."""
import argparse
import collections
import csv
from fractions import Fraction
import gzip
import hashlib
import io
from pathlib import Path
import struct


SCALE = 1 << 1074


def read_table(file):
    with gzip.open(file, "rt", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def exact_scaled(value):
    integer, power_of_two = value.as_integer_ratio()
    return integer << (1074 - (power_of_two.bit_length() - 1))


def check_comparison(state, margin, threshold):
    assert state in ("le", "gt", "cross", "unavailable"), state
    assert state != "le" or margin <= threshold
    assert state != "gt" or margin > threshold


def verify_replay(table, vector_path):
    rows = collections.Counter()
    with gzip.open(vector_path, "rb") as source:
        data = source.read()
    if vector_path.name.endswith(".hex.gz"):
        data = bytes.fromhex(data.decode("ascii"))
    with io.BytesIO(data) as vectors:
        for row in read_table(table):
            encoding = {"Float32": "<f", "Float64": "<d"}[row["type"]]
            assert vectors.tell() == int(row["byte_offset"])
            size = struct.calcsize(encoding) * int(row["n"])
            data = vectors.read(size)
            assert len(data) == size and hashlib.sha256(data).hexdigest() == row["vector_sha256"]
            radius = exact_scaled(float(row["radius"]))
            margin = sum(exact_scaled(abs(x)) for x, in struct.iter_unpack(encoding, data)) - radius
            assert float(Fraction(margin, SCALE)) == float(row["margin"])
            caps = (exact_scaled(float(row["cap20"])), exact_scaled(float(row["cap40"])))
            for label, cap in zip(("20", "40"), caps):
                assert str(margin <= cap).lower() == row[f"exact_pass{label}"]
            for design in "ABC":
                for suffix, threshold in zip(("sign", "cap20", "cap40"), (0, *caps)):
                    check_comparison(row[f"{design}_{suffix}"], margin, threshold)
            rows[row["kind"]] += 1
        assert not vectors.read(1)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("vectors", type=Path)
    args = parser.parse_args()
    for kind, count in sorted(verify_replay(args.table, args.vectors).items()):
        print(kind, count)


if __name__ == "__main__":
    main()
