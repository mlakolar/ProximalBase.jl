"""Parse retained CUDA.jl 5.11.3 printed traces; durations are microseconds.

This parser follows the displayed headers, including optional host thread/task
columns. It summarizes diagnostic traces, never the benchmark timing samples.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def microseconds(text):
    value, unit = text.split()
    return float(value) * {"ns": 0.001, "µs": 1, "ms": 1000, "s": 1e6}[unit]


def summarize(root):
    result = {}
    for shape in (400, 10000):
        sections = {}
        current = side = header = None
        for line in (root / f"profile-{shape}.log").read_text().splitlines():
            if line.startswith(("OPERATION:", "FULL SOLVE PROFILE:")):
                current = line
                sections[current] = {s: defaultdict(lambda: [0, 0.0]) for s in ("host", "device")}
                side = None
            elif current and line.startswith("Profiler ran for "):
                sections[current]["wall"] = line
            elif current and line.startswith("Host-side activity:"):
                side = "host"
                sections[current]["host_summary"] = line
            elif current and line.startswith("Device-side activity:"):
                side = "device"
                sections[current]["device_summary"] = line
            elif current and side and line.startswith("│"):
                cells = [cell.strip() for cell in line.split("│")[1:-1]]
                if cells and cells[0] == "ID":
                    header = cells
                    continue
                if len(cells) < 4 or not cells[0].isdigit():
                    continue
                name = cells[header.index("Name")]
                if side == "device" and not name.startswith("[copy"):
                    name = ("gemv" if "gemv" in name else "reduction" if "mapreduce" in name
                            else "broadcast" if "broadcast" in name else "other kernel")
                entry = sections[current][side][name]
                entry[0] += 1
                entry[1] += microseconds(cells[header.index("Time")])
        result[str(shape)] = sections
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
