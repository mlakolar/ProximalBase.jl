# Digest of a campaign of the suite "a3" (or of any suite whose case ids are family/operation/type/n<length>/...): per
# stratum (CPU kind and Julia version, its processes pooled)
#   1. per tree, geometric means of the per-case ratio tree/reference (median over processes of the per-process median
#      of the paired ratios) by operation and size class (short: up to 64 entries, medium: 128 to 8192, large: above);
#   2. per tree, the cells that are slower in every process (per-process median ratio above 1.05, the A/A control
#      within [0.98, 1.02] in every process), worst first, with the reference and tree times of one call;
#   3. with --tables, per family, operation and element type one row per length: the reference time and for every
#      tree the range of the per-process ratios, and for prox_out the offsets of the output of the processes
#      (pointer(y) - pointer(x)) mod 4096, from the "a3 layout" lines of run.log.
# A process is used only if it finished (DONE, no CHECKFAIL). Within a process a (case, implementation) is valid when
# meta.csv has it with status ok and raw.csv has it in exactly --rounds (default 21) distinct rounds with finite
# positive times; a cell is classified, and enters a geometric mean, only when the tree, the reference and the control
# are valid in every process of the stratum and the stratum has at least two processes.
#   python3 bench/matrix_digest.py <results dir of the campaign> [--tables] [--trees=A3,A3c] [--ops=prox_out,prox_in] [--rounds=21]
import sys, csv, glob, os, re, math, statistics as st
root = sys.argv[1]; tables = "--tables" in sys.argv
def opt(name):
    v = [a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--" + name + "=")]
    return v[0] if v else None
want_trees, want_ops = opt("trees"), opt("ops")
ROUNDS = int(opt("rounds")[0]) if opt("rounds") else 21
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
def cls(n): return "short" if n <= 64 else ("medium" if n <= 8192 else "large")
strata = {}; unused = {}
for d in sorted(glob.glob(root + "/*-p*")):
    s = re.sub(r"-p\d+$", "", os.path.basename(d))
    if os.path.exists(d + "/DONE") and not os.path.exists(d + "/CHECKFAIL") and os.path.exists(d + "/meta.csv") and os.path.exists(d + "/raw.csv"):
        strata.setdefault(s, []).append(d)
    else: unused.setdefault(s, []).append(os.path.basename(d)); strata.setdefault(s, [])
for s, ds in sorted(strata.items()):
    per = {}; ns_ = {}; ref = None; bad = set(); layout = {}; backing = {}
    for d in ds:
        ok = set()
        for r in csv.DictReader(open(d + "/meta.csv")):
            ref = r["ref"]
            if r["status"] == "ok": ok.add((r["case"], r["impl"]))
            else: bad.add((r["case"], r["impl"]))
        if os.path.exists(d + "/run.log"):
            for l in open(d + "/run.log"):
                m = re.match(r"a3 layout (\w+) n=(\d+): y (\d+)", l)
                if m:
                    layout.setdefault((m.group(1), int(m.group(2))), []).append(int(m.group(3)))
                    h = re.search(r"hugepct x (-?\d+) y (-?\d+)", l)
                    if h: backing.setdefault((m.group(1), int(m.group(2))), []).append("%s/%s" % (h.group(1), h.group(2)))
        cur = {}; seen = {}; invalid = set()
        for r in csv.reader(open(d + "/raw.csv")):
            if r[0] == "case": continue
            v = float(r[4]); k = (r[0], r[1])
            if r[2] in seen.setdefault(k, set()) or not (v > 0 and math.isfinite(v)): invalid.add(k)
            seen[k].add(r[2])
            cur.setdefault((r[0], r[2]), {})[r[1]] = v
        valid = {k for k, rounds in seen.items() if len(rounds) == ROUNDS and k not in invalid and k in ok}
        for k in seen:
            if k not in valid: bad.add(k)
        acc = {}; accn = {}
        for (c, _), row in cur.items():
            if (c, ref) in valid:
                for im, v in row.items():
                    if (c, im) not in valid: continue
                    acc.setdefault((c, im), []).append(v / row[ref]); accn.setdefault((c, im), []).append(v)
        for k, v in acc.items(): per.setdefault(k, []).append(st.median(v))
        for k, v in accn.items(): ns_.setdefault(k, []).append(st.median(v))
    ctl = ref + "C"
    trees = [t for t in sorted(set(k[1] for k in per) - {ref, ctl, "copy"}) if not want_trees or t in want_trees]
    key = lambda c: (c.split("/")[0], c.split("/")[1], c.split("/")[2], int(c.split("/")[3][1:]))
    cases = sorted(set(k[0] for k in per), key=key)
    if want_ops: cases = [c for c in cases if c.split("/")[1] in want_ops]
    print("\n## %s (%d processes used%s; reference %s; invalid (case, implementation) pairs: %d)" % (s, len(ds),
          ("; NOT used: " + " ".join(unused[s])) if s in unused else "", ref, len(bad)))
    if not ds: continue
    # a cell is classified only if the tree and the reference are valid in every one of at least two processes, and no
    # process of the stratum was left out
    full = lambda c, t: len(ds) >= 2 and s not in unused and len(per.get((c, t), [])) == len(ds) and (c, t) not in bad and (c, ref) not in bad
    ctlok = lambda c: full(c, ctl) and min(per[(c, ctl)]) >= 0.98 and max(per[(c, ctl)]) <= 1.02
    ncomplete = sum(1 for c in cases for t in trees if full(c, t) and full(c, ctl)); ncells = len(cases) * len(trees)
    print("\ncells with complete, valid data: %d of %d" % (ncomplete, ncells))
    # 1. geometric means
    ops = sorted(set((key(c)[0], key(c)[1]) for c in cases))
    print("\nGeometric means of tree/reference (cases), both element types:\n")
    print("| family | operation | size | cases | " + " | ".join(trees) + " | control |\n|:--|:--|:--|--:|" + "--:|" * (len(trees) + 1))
    for fam, op in ops:
        for k in ("short", "medium", "large"):
            call = [c for c in cases if key(c)[:2] == (fam, op) and cls(key(c)[3]) == k]
            if not call: continue
            cs = [c for c in call if full(c, ctl) and all(full(c, t) for t in trees)]     # the same valid cases for every tree
            if not cs: print("| %s | %s | %s | 0 of %d | no valid case |" % (fam, op, k, len(call))); continue
            cells = ["%.2f [%.2f–%.2f]" % (gm([st.median(per[(c, t)]) for c in cs]), min(st.median(per[(c, t)]) for c in cs), max(st.median(per[(c, t)]) for c in cs)) for t in trees]
            print("| %s | %s | %s | %s | %s | %.2f |" % (fam, op, k, len(cs) if len(cs) == len(call) else "%d of %d" % (len(cs), len(call)), " | ".join(cells), gm([st.median(per[(c, ctl)]) for c in cs])))
    # 2. reproducible regressions and gains
    for t in trees:
        cs_t = [c for c in cases if full(c, t)]
        slow = sorted(((min(per[(c, t)]), max(per[(c, t)]), c) for c in cs_t if ctlok(c) and min(per[(c, t)]) > 1.05), reverse=True)
        fast = [c for c in cs_t if ctlok(c) and max(per[(c, t)]) < 0.95]
        unc = [c for c in cs_t if not ctlok(c) and (min(per[(c, t)]) > 1.05 or max(per[(c, t)]) < 0.95)]
        mixed = sorted(((max(per[(c, t)]), min(per[(c, t)]), c) for c in cs_t if ctlok(c) and max(per[(c, t)]) > 1.05 and min(per[(c, t)]) <= 1.05), reverse=True)
        print("\n%s: %d of %d cells slower in every process, %d faster in every process, %d above 1.05 in some processes only, %d undecided (control unclean)" % (t, len(slow), len(cases), len(fast), len(mixed), len(unc)))
        for lo, hi, c in slow[:40]:
            print("  slower  %-44s %.3f–%.3f   %.1f -> %.1f ns" % (c, lo, hi, st.median(ns_[(c, ref)]), st.median(ns_[(c, t)])))
        for hi, lo, c in mixed[:25]:
            T, n = key(c)[2], key(c)[3]
            print("  some    %-44s %.3f–%.3f   %.1f -> %.1f ns   output offsets %s" % (c, lo, hi, st.median(ns_[(c, ref)]), st.median(ns_[(c, t)]), layout.get((T, n), "")))
    # 3. tables
    if tables:
        for fam, op in ops:
            for T in ("Float64", "Float32"):
                rows = [c for c in cases if key(c)[:3] == (fam, op, T)]
                if not rows: continue
                print("\n%s/%s, %s\n\n| n | %s ns | %s | control |%s\n|--:|--:|%s:--|%s" % (fam, op, T, ref, " | ".join(trees), " output offsets |" if op == "prox_out" else "", ":--|" * len(trees), ":--|" if op == "prox_out" else ""))
                for c in rows:
                    n = key(c)[3]; k = per.get((c, ctl), []); cells = []
                    for t in trees:
                        v = per.get((c, t), [])
                        if not full(c, t) or not full(c, ctl): mark = " (invalid)"
                        elif ctlok(c) and min(v) > 1.05: mark = " **slower**"
                        elif ctlok(c) and max(v) < 0.95: mark = " faster"
                        elif not ctlok(c) and (min(v) > 1.05 or max(v) < 0.95): mark = " (control)"
                        else: mark = ""
                        cells.append(("%.3f–%.3f%s" % (min(v), max(v), mark)) if v else "missing")
                    print("| %d | %.1f | %s | %s |%s" % (n, st.median(ns_[(c, ref)]), " | ".join(cells), ("%.3f–%.3f" % (min(k), max(k))) if k else "missing", (" %s%s |" % (" ".join(map(str, layout.get((T, n), []))), ("; huge x/y % " + " ".join(backing[(T, n)])) if (T, n) in backing else "")) if op == "prox_out" else ""))
