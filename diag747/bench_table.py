# Combine vb-base-r*.txt and vb-head-r*.txt (separate processes, alternating order) into one table.
import sys, glob, re, collections
sys.stdout.reconfigure(encoding="utf-8")
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

# ---- the LLVM IR blocks of the first process of each side, names and labels normalized
import re as _re
def ir_blocks(path):
    out, cur, name = {}, None, None
    for line in open(path, encoding="utf8"):
        if line.startswith("#IR-BEGIN "): name = line[10:].strip(); cur = []
        elif line.startswith("#IR-END "): out[name] = "".join(cur); cur = None
        elif cur is not None: cur.append(line)
    return out
def norm(s):
    s = _re.sub(r"\b(julia|j)_([^\s\"(@]*?)_[0-9]+", r"\1_\2_N", s)         # compiled function names with a counter
    s = _re.sub(r"jl_global#[0-9]+", "jl_global#N", s)
    s = _re.sub(r"\b[0-9]{9,}\b", "N", s)                                # type tags and addresses of a process
    return _re.sub(r"\bL[0-9]+", "L", s)
print("\n== IR: base against head (the first process of each, names and block labels normalized)")
ba, ha = ir_blocks(sorted(glob.glob("vb-base-r*.txt"))[0]), ir_blocks(sorted(glob.glob("vb-head-r*.txt"))[0])
nsame = 0
for k in sorted(ba):
    same = k in ha and norm(ba[k]) == norm(ha[k])
    nsame += same
    print("%-48s %s  (%d lines)" % (k, "identical" if same else "DIFFERENT", ba[k].count("\n")))
    if not same and k in ha:
        import difflib
        for l in list(difflib.unified_diff(norm(ba[k]).splitlines(), norm(ha[k]).splitlines(), lineterm="", n=0))[:12]: print("      ", l[:150])
print("IR blocks identical: %d of %d" % (nsame, len(ba)))
