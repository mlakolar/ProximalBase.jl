#!/usr/bin/env python3
"""Summarize the output of kkt_local.jl: medians per tree, paired ratios per round and their ranges.

usage: python3 benchmark/summarize_kkt_local.py OUT.tsv FILE.out [FILE.out ...]

Timing output (RAW, VAL, SCAN and SKIP lines). For each row of each file the table OUT.tsv has the Julia version, the
group, the label, the calls per pass, the median ns per call of each tree, the largest warmed allocation (bytes), the value
flag (whether the trees that ran agree with base / whether they are expected to), the number of results compared, whether
the four checked trees agree, the share of positive scores of a scan, the trees left out of the row after a SKIP line and,
for each pair a/b of

    head/base  base/pre  head/pre  basectl/base  headctl/head

the median, minimum and maximum over the rounds of the ratio of the time of a to the time of b in the same round (the
trees of a round run back to back). A pair is left empty when a tree is missing from the file; base/pre and head/pre
need a history tree named pre. For the rows of group E the calls per pass are the number of inactive coordinates, and the
ns per call of its three rows are ns per inactive coordinate (the gradient row makes no kkt_violation call).

The column head_vs_controls compares head/base with the two controls of the row (basectl/base and headctl/head, two
compilations of the same source). With m1 and m2 the medians of the controls, lo is the smallest of m1, m2, 1/m1 and
1/m2, and hi = 1/lo. The column is below when the median of head/base is under lo, above when it is over hi and within
otherwise. It is empty when head/base is missing, and it uses the controls that are present.

Compiled-code output (IR lines, PB_MODE=ir). The table OUT.ir.tsv (OUT without a final .tsv, followed by .ir.tsv) has
the group, the label and, for each tree, the number of lines of the optimized code of the loop, the number of Julia
callees followed, the number of calls in the loop text whose target names kkt_violation, and same, differs or unresolved
relative to base. It is written when a file has IR lines. The number of rows per state is printed for each file. Outputs
written before the callees were followed count the calls named by Julia symbols in that column.

It uses the Python standard library.
"""
import re
import sys
from statistics import median

PAIRS = [("head", "base"), ("base", "pre"), ("head", "pre"), ("basectl", "base"), ("headctl", "head")]


def parse(path):
    env, rows, vals, skips, irs = [], {}, {}, [], []
    scans, skipped, stops = {}, {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("ENV "):
                env.append(line[4:])
            elif line.startswith("RAW\t"):
                _, group, label, name, ncalls, alloc, raw = line.split("\t")
                rows.setdefault((group, label), {})[name] = (int(ncalls), int(alloc), [float(x) for x in raw.split(",")])
            elif line.startswith("VAL\t"):
                f = line.split("\t")
                # f[5] is the number of results compared and f[6] the agreement of the four checked trees in the
                # current layout (8 fields); older outputs hold the values of the trees in f[5]
                vals[(f[1], f[2])] = (f[3], f[4], f[5] if len(f) >= 8 else "", f[6] if len(f) >= 7 else "")
            elif line.startswith("SCAN\t"):
                f = line.split("\t")
                scans[(f[1], f[2])] = f[3]
            elif line.startswith("SKIP\t"):
                skips.append(line)
                f = line.split("\t")
                m = re.match(r"\[([^\]]+)\]", f[3]) if len(f) > 3 else None
                if m:
                    skipped.setdefault((f[1], f[2]), []).append(m.group(1))
            elif line.startswith(("VALUES DIFFER\t", "CASE FAILED\t", "CHECK")):
                stops.append(line)
            elif line.startswith("IR\t"):
                _, group, label, desc = line.split("\t")
                irs.append((group, label, desc))
    return env, rows, vals, skips, irs, scans, skipped, stops


def ratio(a, b):
    r = [x / y for x, y in zip(a, b)]
    return median(r), min(r), max(r)


def head_vs_controls(parts):
    if "head" not in parts or "base" not in parts:
        return ""
    controls = []
    if "basectl" in parts:
        controls.append(ratio(parts["basectl"][2], parts["base"][2])[0])
    if "headctl" in parts:
        controls.append(ratio(parts["headctl"][2], parts["head"][2])[0])
    if not controls:
        return ""
    lo = min(controls + [1 / c for c in controls])
    hi = 1 / lo
    m = ratio(parts["head"][2], parts["base"][2])[0]
    return "below" if m < lo else "above" if m > hi else "within"


def parse_ir(desc):
    """name=lines:calls:kkt_calls:same|differs;... -> {name: (lines, calls, kkt_calls, state)}"""
    out = {}
    for item in desc.split(";"):
        name, _, rest = item.partition("=")
        f = rest.split(":")
        if len(f) == 4:
            out[name] = tuple(f)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, files = sys.argv[1], sys.argv[2:]
    cols = ["file", "julia", "group", "label", "calls", "trees", "ns_per_call", "alloc_max", "values", "compared", "core_values",
            "scan_share", "skipped"]
    for a, b in PAIRS:
        cols += [f"{a}/{b}", f"{a}/{b}_min", f"{a}/{b}_max"]
    cols += ["head_vs_controls", "rounds"]
    ircols, irrows, irtrees = ["file", "julia", "group", "label"], [], []
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for path in files:
            env, rows, vals, skips, irs, scans, skipped, stops = parse(path)
            m = re.search(r"julia=(\S+)", " ".join(env))
            julia = m.group(1) if m else "?"
            print(f"## {path}")
            for e in env:
                print("  ", e)
            for s in skips + stops:
                print("  ", s)
            fname = path.split("/")[-1]
            for (group, label), parts in rows.items():
                med = {n: median(v[2]) for n, v in parts.items()}
                rec = {"file": fname, "julia": julia, "group": group, "label": label,
                       "calls": next(iter(parts.values()))[0], "trees": ",".join(parts),
                       "ns_per_call": ";".join(f"{n}={med[n]:.3f}" for n in parts),
                       "alloc_max": max(v[1] for v in parts.values()),
                       "values": "/".join(vals.get((group, label), ("?", "?", "", ""))[:2]),
                       "compared": vals.get((group, label), ("?", "?", "", ""))[2],
                       "core_values": vals.get((group, label), ("?", "?", "", ""))[3],
                       "scan_share": scans.get((group, label), ""),
                       "skipped": ",".join(skipped.get((group, label), [])),
                       "rounds": len(next(iter(parts.values()))[2])}
                for a, b in PAIRS:
                    if a in parts and b in parts:
                        r = ratio(parts[a][2], parts[b][2])
                        rec[f"{a}/{b}"], rec[f"{a}/{b}_min"], rec[f"{a}/{b}_max"] = (f"{x:.3f}" for x in r)
                rec["head_vs_controls"] = head_vs_controls(parts)
                fh.write("\t".join(str(rec.get(c, "")) for c in cols) + "\n")
            for group, label, desc in irs:
                d = parse_ir(desc)
                for n in d:
                    if n not in irtrees:
                        irtrees.append(n)
                irrows.append((fname, julia, group, label, d))
            if irs:
                states = {}
                for _, _, _, _, d in irrows[-len(irs):]:
                    for n, t in d.items():
                        states[t[3]] = states.get(t[3], 0) + 1
                print("   IR rows:", len(irs), "; tree states:", ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
    print(f"wrote {out}")
    if irrows:
        irout = (out[:-4] if out.endswith(".tsv") else out) + ".ir.tsv"
        for n in irtrees:
            ircols += [f"{n}_lines", f"{n}_callees", f"{n}_kkt_calls", f"{n}_vs_base"]
        with open(irout, "w") as fh:
            fh.write("\t".join(ircols) + "\n")
            for fname, julia, group, label, d in irrows:
                cells = [fname, julia, group, label]
                for n in irtrees:
                    cells += list(d.get(n, ("", "", "", "")))
                fh.write("\t".join(cells) + "\n")
        print(f"wrote {irout}")


if __name__ == "__main__":
    main()
