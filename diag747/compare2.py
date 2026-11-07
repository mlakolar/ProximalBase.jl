# Compare the differential outputs of the six runs (issue #747, option C).
#   W-base, W-head : Windows, default exponent range           L-base, L-head : Linux, default range
#   Lw-base, Lw-head: Linux with MPFR's exponent range set to that of Windows (the oracle: Base's shift is exact there)
# Usage: python3 -I compare2.py DIR     (DIR holds diff-<run>.txt)
import sys, re, os, collections, math
from fractions import Fraction
from decimal import Decimal, getcontext
import decimal
sys.stdout.reconfigure(encoding="utf-8")
c = getcontext(); c.prec = 200; c.Emax = decimal.MAX_EMAX; c.Emin = decimal.MIN_EMIN

D = sys.argv[1]
def load(name):
    d = {}
    with open(os.path.join(D, "diff-%s.txt" % name), encoding="utf8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition(" => ")
            d[k] = v
    return d
TYPE = re.compile(r"\b(BigFloat\d+|Float32|Float64)\b")
def typ(k):
    m = TYPE.search(k)
    return m.group(1) if m else "?"
def tgroup(t): return "BigFloat" if t.startswith("BigFloat") else t
def family(k):
    t = k.split()
    if t[0] == "broad": return "broad:" + t[1]
    if t[0].startswith("fam="): return t[0][4:]
    if " _power_factors " in k or " _scaled_product " in k or " _scaled_product3 " in k or " _fused_equal_term " in k: return "helpers"
    return "NormL1Fused-core"
NUM = re.compile(r"-?(?:Inf|NaN)|-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?")
def kind(v):
    if v.startswith("ERR"): return "ERR"
    toks = NUM.findall(v)
    if any(t.lstrip("-") in ("Inf", "NaN") for t in toks): return "nonfinite"
    return "finite"

def same(a, b, an, bn):
    keys = set(a) | set(b)
    diff = [k for k in keys if a.get(k) != b.get(k)]
    by = collections.Counter(family(k) for k in diff)
    print("%s vs %s: %d keys, %d differ%s" % (an, bn, len(keys), len(diff), "" if not diff else "  " + str(dict(by.most_common(8)))))
    for k in sorted(diff)[:5]: print("   ", k, "|", a.get(k, "-")[:60], "|", b.get(k, "-")[:60])
    return len(diff)

print("== (1) Linux head against Linux base (the change is a no-op where Clong is as wide as Int)")
Lb, Lh = load("L-base"), load("L-head")
n1 = same(Lb, Lh, "L-base", "L-head")
Lwb, Lwh = load("Lw-base"), load("Lw-head")
n1b = same(Lwb, Lwh, "Lw-base", "Lw-head")

Wb, Wh = load("W-base"), load("W-head")
print("\n== (2) the oracle: Windows head against Linux head with the exponent range of Windows")
byfam = collections.defaultdict(lambda: [0, 0, []])    # BigFloat lines: total, mismatching
nfl = [0, 0]
for k, v in Wh.items():
    t = typ(k)
    if t.startswith("BigFloat"):
        e = byfam[family(k)]; e[0] += 1
        if v != Lwh.get(k):
            e[1] += 1
            if len(e[2]) < 3: e[2].append((k, v[:70], Lwh.get(k, "-")[:70]))
    else:
        nfl[0] += 1
        if v != Lb.get(k) and v != Lh.get(k): nfl[1] += 1
print("%-34s %9s %9s" % ("family (BigFloat lines)", "lines", "mismatch"))
tot = mm = 0
for f in sorted(byfam):
    tot += byfam[f][0]; mm += byfam[f][1]
    if byfam[f][1]: print("%-34s %9d %9d" % (f, byfam[f][0], byfam[f][1]))
print("BigFloat lines in total %d, W-head differs from the oracle in %d" % (tot, mm))
for f in sorted(byfam):
    for x in byfam[f][2]: print("   MISMATCH", f, x)
print("Float32/Float64 lines of W-head differing from both L-base and L-head (informational, libm): %d of %d" % (nfl[1], nfl[0]))
nfloat_wbase_head = [k for k in Wh if not typ(k).startswith("BigFloat") and Wb.get(k) != Wh[k]]
print("Float32/Float64: W-base against W-head must be identical: %d lines differ" % len(nfloat_wbase_head))
for k in nfloat_wbase_head[:5]: print("   ", k, Wb.get(k, "-")[:60], Wh[k][:60])

print("\n== (3) Windows base against Windows head: the changed lines")
chg = collections.defaultdict(lambda: collections.Counter())
unexpl = []
bad_classes = []
for k, vb in Wb.items():
    vh = Wh.get(k)
    if vh == vb or not typ(k).startswith("BigFloat"): continue
    f = family(k)
    orc = Lwh.get(k)
    ex = vh == orc
    kb, kh = kind(vb), kind(vh)
    cls = {("ERR", "ERR"): "ERR->ERR", ("ERR", "finite"): "error->value", ("ERR", "nonfinite"): "error->nonfinite"}.get((kb, kh))
    if cls is None:
        cls = "%s->%s" % (kb, kh)
    chg[f][cls] += 1
    chg[f]["changed"] += 1
    chg[f]["explained" if ex else "UNEXPLAINED"] += 1
    if not ex and len(unexpl) < 40: unexpl.append((k, vb[:50], vh[:50], (orc or "-")[:50]))
    if kb in ("finite",) and kh in ("nonfinite", "ERR") : chg[f]["FINITE->NONFINITE/ERR"] += 1; bad_classes.append(k)
    if kb == "nonfinite" and kh == "ERR": bad_classes.append(k)
print("%-34s %8s %9s  classes" % ("family", "changed", "explained"))
tc = te = 0
for f in sorted(chg):
    d = chg[f]; tc += d["changed"]; te += d["explained"]
    cl = {k: v for k, v in d.items() if k not in ("changed", "explained", "UNEXPLAINED")}
    print("%-34s %8d %9d  %s%s" % (f, d["changed"], d["explained"], cl, "" if not d["UNEXPLAINED"] else "  UNEXPLAINED %d" % d["UNEXPLAINED"]))
print("changed BigFloat lines %d, explained by the oracle %d, unexplained %d" % (tc, te, tc - te))
print("value->Inf / finite->non-finite / value->error changes: %d" % len(bad_classes))
for k in bad_classes[:10]: print("   ", k, "|", Wb[k][:50], "|", Wh[k][:50])
for u in unexpl[:10]: print("   UNEXPLAINED", u)

print("\n== (4) how wrong was master on Windows: Windows base against Linux base with the range of Windows")
wrong = collections.defaultdict(lambda: [0, 0])
for k, v in Wb.items():
    if typ(k).startswith("BigFloat"):
        e = wrong[family(k)]; e[0] += 1
        if v != Lwb.get(k): e[1] += 1
print("%-34s %9s %9s" % ("family", "lines", "W-base != Lw-base"))
for f in sorted(wrong):
    if wrong[f][1]: print("%-34s %9d %9d" % (f, wrong[f][0], wrong[f][1]))
print("(the same lines that W-head fixes are those changed in (3); a family listed here but unchanged in (3) is a master difference that the shadow did not remove)")
still = collections.Counter()
for k, v in Wb.items():
    if typ(k).startswith("BigFloat") and v != Lwb.get(k) and Wh.get(k) == v: still[family(k)] += 1
print("lines where W-base differs from Lw-base and W-head is unchanged (the shadow did not repair them): %d  %s" % (sum(still.values()), dict(still.most_common(10))))

# ---- element-level regressions: an element equal to the oracle (the same code on a platform without the defect) at base and
# different from it at head (the head made a correct element wrong), and the head2 variant (head with Int exponent sums in
# _exceptional_product) against the oracle
print("\n== (3a) element-level: lines with an element correct at base and wrong at head, against the oracle")
def elem_regressions(Wb_, Wx_, label):
    reg = collections.Counter(); ex = {}
    for k, vb in Wb_.items():
        if not typ(k).startswith("BigFloat"): continue
        vh = Wx_.get(k); vo = Lwh.get(k)
        if vh is None or vh == vb or vb.startswith("ERR") or vh.startswith("ERR"): continue
        tb, th, to = NUM.findall(vb), NUM.findall(vh), NUM.findall(vo)
        if not (len(tb) == len(th) == len(to)): continue
        if any(tb[i] == to[i] and th[i] != to[i] for i in range(len(tb))):
            reg[family(k)] += 1; ex.setdefault(family(k), []).append((k, vb[:50], vh[:50], vo[:50]))
    print("%s: %d lines  %s" % (label, sum(reg.values()), dict(reg)))
    for f in sorted(ex):
        for e in ex[f][:2]: print("    ", f, e)
elem_regressions(Wb, Wh, "head")
if os.path.exists(os.path.join(D, "diff-W-head2.txt")):
    Wh2 = load("W-head2")
    print("\n== (5) the variant head2: head with the exponent sums of _exceptional_product in Int (Windows), against the oracle")
    mm2 = collections.Counter(); tot2 = 0
    for k, v in Wh2.items():
        if typ(k).startswith("BigFloat"):
            tot2 += 1
            if v != Lwh.get(k): mm2[family(k)] += 1
    print("BigFloat lines %d, head2 differs from the oracle in %d  %s" % (tot2, sum(mm2.values()), dict(mm2.most_common(12))))
    fl2 = [k for k in Wh2 if not typ(k).startswith("BigFloat") and Wh2[k] != Wb.get(k)]
    print("Float32/Float64 lines differing between W-base and W-head2: %d" % len(fl2))
    elem_regressions(Wb, Wh2, "head2")
    ch2 = sum(1 for k in Wh2 if Wh2[k] != Wh.get(k))
    print("lines of head2 that differ from head: %d" % ch2)

# ---- independent closed forms for a sample of the changed lines (NormL1, NormL2, IndSOC at clean margins)
print("\n== (3b) independent closed forms for the changed lines of NormL1, NormL2 and IndSOC (positive scalars, scaled arithmetic)")
EMAX = 2**30 - 1
def mags(p):
    one = Fraction(1); r = 1 - Fraction(1, 2**p)
    h = EMAX // 4
    return [(one, 0), (Fraction(3, 2), 0), (one, h), (one, -h), (r, EMAX), (r / 2, EMAX), (one, -EMAX + 8)]
def val_scaled(lst):
    """sum of (mant, exp) terms as (Fraction, exp), dropping terms 4500 binades below the largest"""
    m = max(e for _, e in lst); return sum((a * Fraction(2) ** (e - m) for a, e in lst if e - m > -4500), Fraction(0)), m
def dec(fr): return Decimal(fr.numerator) / Decimal(fr.denominator)
def parse_x(k, p):
    m = re.search(r"x=(-?\d+),(-?\d+),(-?\d+)", k)
    if not m: return None
    P = mags(p)
    return [(abs(int(g)) and P[abs(int(g)) - 1], int(g) < 0) for g in m.groups()]
def check_value(k, vh, p):
    xs = parse_x(k, p)
    if xs is None: return None
    t = k.split(); name = t[1]
    ents = [a for a, s in xs]
    try: hv = Decimal(vh)
    except Exception: return None
    if name == "NormL1":
        s, m = val_scaled(ents); ref_ln = (dec(s) * Decimal("0.5")).ln() / Decimal(2).ln() + m
    elif name == "NormL2":
        s, m = val_scaled([(a * a, 2 * e) for a, e in ents])
        ref_ln = (dec(s).sqrt() * Decimal("0.5")).ln() / Decimal(2).ln() + Decimal(m) / 2
    else:
        return None
    return abs(Decimal(2) ** (hv.ln() / Decimal(2).ln() - ref_ln) - 1) <= 2 * Decimal(2) ** (1 - p) + Decimal("0.06")
chk = collections.Counter()
for k, vb in Wb.items():
    vh = Wh.get(k)
    if vh == vb or not typ(k).startswith("BigFloat") or not k.endswith(" value"): continue
    f = family(k)
    if f not in ("broad:NormL1", "broad:NormL2"): continue
    p = int(typ(k)[8:])
    if kind(vh) != "finite": chk[(f, "head non-finite")] += 1; continue
    r = check_value(k, vh, p)
    chk[(f, "checked-in-envelope" if r else ("skipped" if r is None else "OUT-OF-ENVELOPE"))] += 1
print(dict(chk))
print("== done")
