# Combine vb-base-r*.txt and vb-head-r*.txt (separate processes, alternating order) into one table.
import sys, glob, re, collections
def load(prefix):
    d = collections.defaultdict(list)
    for p in sorted(glob.glob(prefix + "-r*.txt")):
        for line in open(p, encoding="utf8"):
            m = re.match(r"BENCH (\S+) (\S+) n=(\d+) (\S+) alloc=(\d+) ns_min=(\S+) ns_max=(\S+)", line)
            if m:
                T, case, n, call, alloc, lo, hi = m.groups()
                d[(T, case, int(n), call)].append((int(alloc), float(lo), float(hi)))
    return d
b, h = load("vb-base"), load("vb-head")
print("%-8s %-26s %6s %-16s %9s %9s %12s %12s %7s  %s" % ("T", "case", "n", "call", "alloc b", "alloc h", "ns base", "ns head", "ratio", "spread (min..max over all reps) base | head"))
flag = []
for key in sorted(b, key=lambda k: (k[0], k[2], k[1], k[3])):
    if key not in h: continue
    ab = sorted({x[0] for x in b[key]}); ah = sorted({x[0] for x in h[key]})
    nb = min(x[1] for x in b[key]); nh = min(x[1] for x in h[key])
    sb = (nb, max(x[2] for x in b[key])); sh = (nh, max(x[2] for x in h[key]))
    r = nh / nb
    mark = "" if 0.97 <= r <= 1.03 else "  <-- OUT OF BAND"
    if mark or ab != ah: flag.append((key, r, ab, ah, sb, sh))
    print("%-8s %-26s %6d %-16s %9s %9s %12.2f %12.2f %7.3f  %.2f..%.2f | %.2f..%.2f%s" % (
        key[0], key[1], key[2], key[3], "/".join(map(str, ab)), "/".join(map(str, ah)), nb, nh, r, sb[0], sb[1], sh[0], sh[1], mark))
print("processes: base %d, head %d" % (len(next(iter(b.values()))), len(next(iter(h.values())))))
print("flagged:", len(flag))
for key, r, ab, ah, sb, sh in flag:
    print("FLAG", key, "ratio %.3f" % r, "alloc", ab, ah, "spread base %.2f..%.2f head %.2f..%.2f" % (sb + sh))
