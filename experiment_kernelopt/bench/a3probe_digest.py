# Digest of a probe directory written by slurm/a3probes.sbatch (the offset sweep and the ordinary-allocation probe),
# per CPU kind and candidate tree:
#   1. offset sweep (arrays forced onto huge pages at chosen offsets, no page shift): for each element type, function
#      and length, the times at offset 0, the median over processes of candidate/reference at each output offset; the
#      largest in the band 16..256 bytes with its offset, the number of band offsets above 1.05, the largest outside
#      the band, the range of the A/A control, and the backing that was obtained (smallest huge-page percentage of
#      either array over the rows; "?" for files that do not record it);
#   2. page shifts: candidate/reference at the offsets 64, 96, 128 and 192 for every number of added 4 KiB pages;
#   3. ordinary allocations: candidate/reference per trial, by the kind of pattern (ordinary or advised), by whether the
#      output lies in the band, and by backing: "huge" when every 2 MiB page that lies wholly inside each array is a
#      huge page (an array of 4 MiB or more that does not start at a multiple of 2 MiB cannot reach 100%: its two end
#      pages are shared; below 4 MiB "huge" means 100%), "small" both 0%, "mixed" otherwise, "?" when the file has only
#      the huge-page kB of whole mappings (the first version of the probe) or the share is unknown.
#   python3 bench/a3probe_digest.py <results/probe dir> [--band=48,192]
import sys, csv, glob, os, re, statistics as st
d = sys.argv[1]
blo, bhi = 16, 256
for a in sys.argv[2:]:
    if a.startswith("--band="): blo, bhi = map(int, a.split("=")[1].split(","))
inband = lambda o: blo <= o <= bhi
def groups():
    g = {}
    for f in sorted(glob.glob(d + "/*-offset-p*.csv") + glob.glob(d + "/*-alloc-p*.csv")):
        m = re.match(r"(.*?)-[a-z]\d\d-\d\d-julia[\d.]+-(.*)-(offset|alloc)-p\d+\.csv$", os.path.basename(f))
        g.setdefault((m.group(1), m.group(2)), {}).setdefault(m.group(3), []).append(f)
    return g
med = st.median
for (kind, cand), fs in sorted(groups().items()):
    files = fs.get("offset", [])
    if files:
        print("\n## %s, candidate %s: offset sweep on huge pages (%d processes)\n" % (kind, cand, len(files)))
        acc = {}; ctl = {}; absns = {}; back = {}
        for f in files:
            rows = [r for r in csv.DictReader(open(f)) if r["backing"] == "huge"]
            t = {(r["T"], int(r["n"]), int(r.get("ypages") or 0), int(r["yoff"]), r["impl"]): float(r["med_ns"]) for r in rows}
            for r in rows:
                if "xhuge_pct" in r and r["xhuge_pct"] not in (None, ""):
                    k = (r["T"], int(r["n"])); back[k] = min(back.get(k, 100), int(r["xhuge_pct"]), int(r["yhuge_pct"]))
            for (T, n, pg, o, im), v in t.items():
                fam, _, who = im.partition("/")
                if who != "cand": continue
                ref = t[(T, n, pg, o, fam + "/ref")]
                acc.setdefault((T, n, fam, pg, o), []).append(v / ref)
                ctl.setdefault((T, n, fam), []).append(t[(T, n, pg, o, fam + "/refC")] / ref)
                if o == 0 and pg == 0: absns.setdefault((T, n, fam), []).append((ref, v))
        print("| type | entries | function | master ns at offset 0 | candidate ns at offset 0 | ratio at 0 | largest in band (offset) | band offsets > 1.05 | largest outside band (offset) | control min–max | huge % |")
        print("|:--|--:|:--|--:|--:|--:|:--|:--|:--|:--|--:|")
        for (T, n, fam) in sorted(absns, key=lambda k: (k[0], k[2], k[1])):
            offs = sorted(o for (T2, n2, f2, pg, o) in acc if (T2, n2, f2, pg) == (T, n, fam, 0))
            m = {o: med(acc[(T, n, fam, 0, o)]) for o in offs}
            band = [o for o in offs if inband(o)]; outb = [o for o in offs if not inband(o)]
            ob = max(band, key=lambda o: m[o]); oo = max(outb, key=lambda o: m[o])
            print("| %s | %d | %s | %.0f | %.0f | %.2f | %.2f (+%d) | %d of %d | %.2f (+%d) | %.2f–%.2f | %s |" % (T, n, fam, med(a for a, _ in absns[(T, n, fam)]),
                  med(b for _, b in absns[(T, n, fam)]), m[0], m[ob], ob, sum(m[o] > 1.05 for o in band), len(band), m[oo], oo,
                  min(ctl[(T, n, fam)]), max(ctl[(T, n, fam)]), back.get((T, n), "?")))
        shifts = sorted({k[3] for k in acc})
        if len(shifts) > 1:
            print("\n### %s, candidate %s: the same offsets with whole pages added (median over processes of candidate/reference)\n" % (kind, cand))
            print("| type | entries | function | offset | " + " | ".join("+%d pages" % p for p in shifts) + " |\n|:--|--:|:--|--:|" + "--:|" * len(shifts))
            for (T, n, fam) in sorted({k[:3] for k in acc if k[3] != 0}, key=lambda k: (k[0], k[2], k[1])):
                for o in (0, 64, 96, 128, 192):
                    if (T, n, fam, shifts[-1], o) not in acc: continue
                    print("| %s | %d | %s | %d | %s |" % (T, n, fam, o, " | ".join(("%.2f" % med(acc[(T, n, fam, p, o)])) if (T, n, fam, p, o) in acc else "" for p in shifts)))
    files = fs.get("alloc", [])
    if files:
        print("\n## %s, candidate %s: ordinary allocations (%d processes)\n" % (kind, cand, len(files)))
        tr = {}
        for p, f in enumerate(files):
            for r in csv.DictReader(open(f)):
                k = (p, r["T"], int(r["n"]), r["pattern"], r["trial"])
                if k not in tr:
                    if "xhuge_pct" in r:
                        hs = [int(r["xhuge_pct"]), int(r["yhuge_pct"])] + [int(r[c]) for c in ("xhuge_after", "yhuge_after") if r.get(c) not in (None, "")]
                        size = int(r["n"]) * (4 if r["T"] == "Float32" else 8); W = -(-size // (2 << 20)) + 1
                        need = 100 if size < (4 << 20) else (100 * (W - 2)) // W
                        b = "?" if min(hs) < 0 else ("huge" if min(hs) >= need else ("small" if max(hs) == 0 else "mixed"))
                    else: b = "?"
                    tr[k] = {"yoff": int(r["yoff"]), "back": b}
                tr[k][r["func"] + "/" + r["impl"]] = float(r["med_ns"])
        classes = [(adv, band, b) for adv in (False, True) for band in (True, False) for b in ("huge", "mixed", "small", "?")]
        f2 = lambda v, g: ("%.2f" % g(v)) if v else ""
        for fam in ("l1", "enet"):
            print("\n%s: candidate/reference per trial; cells are trials, median, maximum\n" % fam)
            head = []
            for adv, band, b in classes:
                if any(k[3].endswith("_thp") == adv and inband(e["yoff"]) == band and e["back"] == b for k, e in tr.items()):
                    head.append((adv, band, b))
            print("| type | entries | " + " | ".join("%s, %s, %s" % ("advised" if a else "ordinary", "band" if bd else "outside", b) for a, bd, b in head) + " | ordinary trials > 1.00 | > 1.05 | control min–max |")
            print("|:--|--:|" + ":--|" * len(head) + "--:|--:|:--|")
            for (T, n) in sorted({(k[1], k[2]) for k in tr}):
                cells = []; allr = []; ctlr = []
                for adv, band, b in head:
                    v = [e[fam + "/cand"] / e[fam + "/ref"] for k, e in tr.items() if (k[1], k[2]) == (T, n) and k[3].endswith("_thp") == adv and inband(e["yoff"]) == band and e["back"] == b]
                    cells.append("%d, %s, %s" % (len(v), f2(v, med), f2(v, max)) if v else "")
                for k, e in tr.items():
                    if (k[1], k[2]) != (T, n): continue
                    ctlr.append(e[fam + "/refC"] / e[fam + "/ref"])
                    if not k[3].endswith("_thp"): allr.append(e[fam + "/cand"] / e[fam + "/ref"])
                print("| %s | %d | %s | %d | %d | %.2f–%.2f |" % (T, n, " | ".join(cells), sum(v > 1.0 for v in allr), sum(v > 1.05 for v in allr), min(ctlr), max(ctlr)))
