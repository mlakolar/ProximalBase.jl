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
F32 = re.compile(r"(?<=[0-9])f(?=-?[0-9])")
def norm32(v):
    """Julia prints Float32 as 1.2f38 and Inf32/NaN32; read them as 1.2e38, Inf, NaN"""
    if v.startswith("ERR"): return v
    return F32.sub("e", v).replace("Inf32", "Inf").replace("NaN32", "NaN")
def load(name):
    d = {}
    with open(os.path.join(D, "diff-%s.txt" % name), encoding="utf8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition(" => ")
            d[k] = norm32(v)
    return d
TYPE = re.compile(r"\b(BigFloat\d+|Float32|Float64)\b")
def typ(k):
    m = TYPE.search(k)
    return m.group(1) if m else "?"
def tgroup(t): return "BigFloat" if t.startswith("BigFloat") else t
def family(k):
    t = k.split()
    if t[0] == "broad": return "broad:" + t[1]
    if t[0].startswith("spectral:") or t[0].startswith("logdet:"): return t[0]
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
lapack_info = collections.Counter(); lapack_tot = collections.Counter()
for k, v in Wh.items():
    t = typ(k)
    if t.startswith("BigFloat") and family(k).startswith("spectral:lapack:"):
        lapack_tot[family(k)] += 1
        if v != Lwh.get(k): lapack_info[family(k)] += 1
        continue
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
print("spectral:lapack:* lines (matrices whose LAPACK output may depend on the platform; informational): %d lines, %d differ from the oracle  %s"
      % (sum(lapack_tot.values()), sum(lapack_info.values()), dict(lapack_info.most_common(6))))
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

# ---- independent references for the spectral lines (exact spectra of the diagonal matrices; Decimal at 120 digits)
print("\n== (3c) spectral families: independent references for the lines of W-head and for the changed lines")
c.prec = 120
WE = 2**30 - 1
def D2(k): return Decimal(2) ** k
TS = {"Float32": (24, 127), "Float64": (53, 1023)}
def Tconst(T):
    nb, ex = TS[T]
    M = (1 - D2(-nb)) * D2(ex + 1)
    fm = D2(-ex + 1) if False else D2(-(ex - 1))      # floatmin = 2^(1 - emax)
    return nb, ex, M, fm
def spec_sigma(T, mid):
    nb, ex, M, fm = Tconst(T)
    return {"d1": [M, M / 2], "d2": [M, M], "d3": [M, fm], "d4": [Decimal(3), Decimal(1)], "d5": [M, M / 2],
            "s1": [M, M / 2], "s2": [M, M], "s3": [Decimal(3), Decimal(1)], "s4": [M, Decimal(0)],
            "p6": [M, M / 2], "f7": [2 * M, Decimal(0)], "s5": [2 * M, Decimal(0)]}.get(mid)
def spec_level(T, lid):
    nb, ex, M, fm = Tconst(T)
    W = WE - 8
    return {"a": Decimal("0.5"), "b": Decimal("1.5") * D2(-ex), "c": D2(-W), "d": D2(W), "e": Decimal("1.5") * D2(-ex - 20)}[lid]
def spec_value_ref(fam, T, mid, lid, scales):
    sg = spec_sigma(T, mid)
    if sg is None: return None
    lam = spec_level(T, lid)
    sa = sum(sg); s2 = sum(x * x for x in sg)
    h = (WE - 8) // 2 + 5
    a, b = scales if scales else (None, None)
    if fam == "SV-NormL1": return lam * sa
    if fam == "SV-NormL2": return lam * s2.sqrt()
    if fam == "SV-NormLinf": return lam * max(sg)
    if fam == "SV-SqrNormL2": return lam / 2 * s2
    if fam == "SV-ElasticNet": return lam * sa + lam / 2 * s2
    if fam == "SV-SumLargest": return lam * max(sg)
    if fam == "SV-NormL1w": return lam * (sg[0] + 2 * sg[1]) if False else None
    if fam in ("SV-Scaled-NormL1",): return lam * sa * a
    if fam in ("SV-Nested-NormL1",): return lam * sa * a * b
    if fam in ("SV-Nested-NormL2",): return lam * s2.sqrt() * a * b
    if fam in ("NuclearNorm", "SV-NormL1Nonneg", "Sp-NormL1"): return lam * sa
    if fam == "SV-Constrained-Nonneg": return lam * sa
    if fam == "Sp-Nonneg": return lam * sa if mid in ("s2", "s3", "s4") else None
    if fam == "SV-CubeNormL2": return lam / 3 * s2.sqrt() ** 3
    if fam == "SV-SLOPE": return lam * (max(sg) + min(sg) / 2)
    if fam == "Sp-NormL2": return lam * s2.sqrt()
    if fam == "Sp-SqrNormL2": return lam / 2 * s2
    if fam in ("Sp-SumLargest", "Sp-Conj-IndBallL1"): return lam * max(sg)
    if fam == "Spectral-Nested-NormL1": return lam * sa * a * b
    if fam == "SV-Conj-IndBallL1": return lam * max(sg)
    if fam == "SV-Conj-IndBallL2": return lam * s2.sqrt()
    if fam == "SV-Conj-SqrNormL2": return s2 / (2 * lam)
    return None
def spec_parse(k):
    t = k.split()
    fam = t[0].split(":")[-1]
    T = dict(x.split("=") for x in t[2:5] if "=" in x)
    mid = T["M"]; mat = T["mat"]; lv = T["lvl"]
    scales = None
    if "/" in lv:
        lv, sid = lv.split("/")
        h = (WE - 8) // 2 + 5
        scales = (D2(h), D2(h)) if sid == "n1" else (D2(h), D2(-h))
    return fam, mat, mid, lv, scales, t[-1]
def spec_ok(printed, ref, mat, p):
    """'in p-bit envelope' (|v - ref| <= 2^(2-p) |ref|, p < 200; vacuous at p = 2), 'in matrix-precision budget only'
    (within 64 ulp of the matrix type but outside the p-bit envelope), 'overflow' (Inf accepted above floatmax), or 'OUT'"""
    nb, ex, M, fm = Tconst(mat)
    eps = D2(1 - nb)
    if printed in ("Inf", "NaN") or printed.startswith("ERR"): pv = None
    else: pv = Decimal(printed)
    b = 64 * 2 * eps * abs(ref) + fm
    if abs(ref) + b >= M:
        if printed == "Inf": return "overflow"
        if abs(ref) - b > M: return "OUT"
    if pv is None: return "OUT"
    dv = abs(pv - ref)
    if p < 200 and dv <= D2(2 - p) * abs(ref) + fm: return "vacuous p=2 envelope" if p == 2 else "in p-bit envelope"
    return "in matrix-precision budget only" if dv <= b else "OUT"
def psplit(p): return "p=2 (envelope vacuous)" if p == 2 else ("p=3..199" if p < 200 else "p>=200")
chk = collections.Counter(); bad = []
chg_chk = collections.Counter(); chk_p = collections.Counter(); chg_p = collections.Counter()
for k, vh in Wh.items():
    if not k.startswith("spectral:") or not k.endswith(" value") or "lapack" in k.split()[0]: continue
    fam, mat, mid, lv, scales, ent = spec_parse(k)
    ref = spec_value_ref(fam, mat, mid, lv, scales)
    if ref is None:
        chk["no closed form"] += 1; continue
    p = int(typ(k)[8:])
    res = spec_ok(vh, ref, mat, p)
    chk[res] += 1; chk_p[(psplit(p), res)] += 1
    if res == "OUT" and len(bad) < 10: bad.append((k, vh[:30], str(ref)[:30]))
    if Wb.get(k) != vh:
        chg_chk["changed, " + res] += 1; chg_p[(psplit(p), res)] += 1

def spec_prox_ref(fam, T, mid, lid):
    """the entries [a11, a21, a12, a22] of the prox at step 1 for the diagonal matrices, or None"""
    nb, ex, M, fm = Tconst(T)
    dg = {"d1": (M, M / 2), "d2": (M, M), "d3": (M, fm), "d4": (Decimal(3), Decimal(1)), "d5": (-M, M / 2),
          "s1": (M, -M / 2), "s2": (M, M), "s3": (Decimal(3), Decimal(1)), "s4": (M, Decimal(0))}.get(mid)
    if dg is None: return None
    lam = spec_level(T, lid)
    soft = lambda d: (1 if d >= 0 else -1) * max(abs(d) - lam, Decimal(0))
    if fam in ("SV-NormL1", "Sp-NormL1", "NuclearNorm"): out = [soft(dg[0]), soft(dg[1])]
    elif fam in ("SV-NormL1Nonneg", "Sp-NormL1Nonneg"): out = [(1 if d >= 0 else -1) * max(abs(d) - lam, Decimal(0)) if fam == "SV-NormL1Nonneg" else max(d - lam, Decimal(0)) for d in dg]
    else: return None
    return [out[0], Decimal(0), Decimal(0), out[1]]
nprox = collections.Counter(); pbad = []
for k, vh in Wh.items():
    if not k.startswith("spectral:") or "lapack" in k.split()[0]: continue
    if not (k.endswith(" prox") or k.endswith(" vprox")): continue
    fam, mat, mid, lv, scales, ent = spec_parse(k)
    ref = spec_prox_ref(fam, mat, mid, lv)
    if ref is None: continue
    toks = NUM.findall(vh)
    toks = toks[-4:]
    if len(toks) != 4 or any(t.lstrip("-") in ("Inf", "NaN") for t in toks):
        nprox["non-finite output"] += 1
        if len(pbad) < 6: pbad.append((k, vh[:50]))
        continue
    nb, ex, M, fm = Tconst(mat)
    eps = D2(1 - nb)
    ok = all(abs(Decimal(t) - r) <= 64 * 2 * eps * abs(r) + fm for t, r in zip(toks, ref))
    nprox["in budget" if ok else "OUT"] += 1
    if not ok and len(pbad) < 6: pbad.append((k, vh[:50], [str(r)[:12] for r in ref]))
    if fam.endswith("Nonneg") and any(Decimal(t) < 0 and fam.startswith("Sp") for t in toks): nprox["infeasible (negative eigenvalue)"] += 1
print("W-head prox/vprox outputs on diagonal matrices against the soft-threshold of the exact spectrum:", dict(nprox))
for b_ in pbad: print("   PROX", b_)

print("W-head spectral value lines with an independent reference:", dict(chk))
print("of which changed from W-base:", dict(chg_chk))
print("by precision (p = 2: the p-bit envelope 2^(2-p)|ref| is vacuous, so those lines are only checked by the oracle comparison):")
for (pg, res), n in sorted(chk_p.items()): print("   %-24s %-34s all %-5d changed %d" % (pg, res, n, chg_p.get((pg, res), 0)))
for b_ in bad: print("   OUT", b_)

# ---- scalar finite-to-Inf spectral lines: a genuine overflow (the exact value is above floatmax) against an overflow-edge allowance
# (the exact value is at most floatmax, e.g. exactly floatmax, and the overflow budget of the references accepts Inf)
print("\n== (3e) spectral value lines that are Inf at the head: genuine overflow or overflow-edge allowance")
inf_cls = collections.Counter(); inf_ex = []
chg_cls = collections.Counter()
for k, vh in Wh.items():
    if not k.startswith("spectral:") or not k.endswith(" value") or vh != "Inf": continue
    fam, mat, mid, lv, scales, ent = spec_parse(k)
    ref = spec_value_ref(fam, mat, mid, lv, scales)
    lap = "lapack" in k.split()[0]
    if ref is None: inf_cls[("no closed form", lap)] += 1; continue
    nb, ex, M, fm = Tconst(mat)
    kind_ = "genuine overflow (exact value > floatmax)" if abs(ref) > M else "overflow-edge allowance (exact value <= floatmax)"
    inf_cls[(kind_, lap)] += 1
    if abs(ref) <= M and len(inf_ex) < 4: inf_ex.append((k, str(ref)[:24]))
    vb = Wb.get(k)
    if vb is not None and vb != "Inf" and not vb.startswith("ERR"): chg_cls[(kind_, lap, "finite at base")] += 1
for (kk, lap), n in sorted(inf_cls.items(), key=str): print("  %-52s %-18s %d" % (kk, "informational" if lap else "strict", n))
print("  of which finite at W-base:", {("%s, %s" % (a[0][:22], "lapack" if a[1] else "strict")): v for a, v in chg_cls.items()})
for e in inf_ex: print("   EDGE", e)

# ---- LogDetLoss at a BigFloat X: the value against the exact formula
print("\n== (3d) logdet:LogDetLoss: BigFloat X = [1 t; t 1], S Float64, w Float64: the value against w (<S, X> - log det X)")
c.prec = 160
def logdet_parse(k):
    t = k.split()
    kv = dict(x.split("=") for x in t[2:] if "=" in x)
    return kv["S"], int(kv["e"]), kv["w"], t[-1]
def logdet_ref(sid, e, wid):
    w = {"w0": Decimal(1), "w30": D2(30), "w100": D2(100), "w1000": D2(1000)}[wid]
    tt = D2(e)
    dot = {"s1": tt, "s2": Decimal(2), "s3": tt / 2}[sid]
    ld = -(tt * tt + tt ** 4 / 2) if e < -300 else (1 - tt * tt).ln()
    return w, dot, ld, w * (dot - ld)
eps64 = D2(-52)
minpos = D2(-WE - 1)
ldc = collections.Counter(); ldchg = collections.Counter(); ldbad = []
for k, vh in Wh.items():
    if not k.startswith("logdet:") or not k.endswith(" value"): continue
    sid, e, wid, _ = logdet_parse(k)
    p = int(typ(k)[8:])
    try: pv = Decimal(vh)
    except Exception: pv = None
    w, dot, ld, ref = logdet_ref(sid, e, wid)
    envA = 2 * D2(1 - p) * abs(ref) + minpos
    envB = 64 * eps64 * w * (abs(dot) + abs(ld)) + minpos
    # a result below the smallest BigFloat is 0, and one above the range is Inf
    top = (1 - D2(-p)) * D2(WE + 1)
    if abs(ref) > top: res = "overflow" if vh in ("Inf", "-Inf") else "OUT"
    elif pv is None: res = "OUT"
    else:
        dv = abs(pv - ref)
        res = "in p-bit envelope" if dv <= envA else ("in matrix-precision budget only" if dv <= envB else "OUT")
    ldc[res] += 1
    changed = Wb.get(k) != vh
    if changed: ldchg[res] += 1
    if res == "OUT" and len(ldbad) < 8: ldbad.append((k, vh[:30], str(ref)[:30]))
print("W-head value lines:", dict(ldc))
print("of which changed from W-base:", dict(ldchg))
for b_ in ldbad: print("   OUT", b_)

# ---- rejection at the entry point against coverage
print("\n== (6) families rejected at the entry point (a constructor or an input builder that throws), against families that were evaluated")
for nm, D_ in (("W-head", Wh),):
    mk = collections.defaultdict(collections.Counter)
    allerr = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    harness = []
    for k, v in D_.items():
        if k.startswith("harness "): harness.append((k, v)); continue
        if k.endswith(" make") and v.startswith("ERR"): mk[family(k)][(typ(k), v[4:])] += 1
        elif k.endswith(" input") and v.startswith("ERR"): mk[family(k) + " (input builder)"][(typ(k), v[4:])] += 1
        elif k.endswith(" value") and (k.startswith("broad ") or k.startswith("spectral:") or k.startswith("logdet:")):
            e = allerr[family(k)]; e[0] += 1
            if v.startswith("ERR"): e[1] += 1; e[2][v[4:]] += 1
    print("%s: constructors or input builders that throw ('rejected at the entry point'):" % nm)
    for f, c in sorted(mk.items()): print("   %-44s %s" % (f, dict(c)))
    if not mk: print("   none")
    rej = [(f, e) for f, e in sorted(allerr.items()) if e[0] and e[1] == e[0]]
    print("%s: families built but whose every `value` line is an error (the method rejects the argument):" % nm)
    for f, e in rej: print("   %-44s %d lines  %s" % (f, e[0], dict(e[2])))
    if not rej: print("   none")
    part = [(f, e) for f, e in sorted(allerr.items()) if e[0] and 0 < e[1] < e[0]]
    print("%s: families with some `value` lines erroring (coverage with partial rejection): %d families; the first ones: %s"
          % (nm, len(part), [(f, "%d/%d" % (e[1], e[0])) for f, e in part[:12]]))
    print("%s: harness group failures (an error outside the row guards): %s" % (nm, harness if harness else "none"))

print("== done")
