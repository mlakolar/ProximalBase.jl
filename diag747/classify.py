# Sort the changed lines of the base-versus-head differential by family and type.
import sys, re, collections
def load(path):
    d = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition(" => ")
            d[k] = v
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
def family(k):
    t = k.split()
    if t[0].startswith("fam="):
        return t[0][4:], t[1]
    lab = t[0]
    body = k
    if " _power_factors_exact " in body or " _scaled_product_exact " in body:
        fam = "helpers-exact (new)"
    elif " _power_factors " in body or " _scaled_product " in body or " _scaled_product3 " in body:
        fam = "helpers-original"
    else:
        fam = "NormL1Fused"
    return fam, lab
def tgroup(lab):
    return "BigFloat" if lab.startswith("BigFloat") else lab
print("keys: base %d head %d same-keys %s" % (len(a), len(b), set(a) == set(b)))
tot, chg, ex = collections.Counter(), collections.Counter(), {}
for k in a:
    fam, lab = family(k)
    key = (fam, tgroup(lab))
    tot[key] += 1
    if a[k] != b.get(k):
        chg[key] += 1
        ex.setdefault(key, []).append(k)
print("%-26s %-9s %9s %9s" % ("family", "type", "lines", "changed"))
for key in sorted(tot):
    print("%-26s %-9s %9d %9d" % (key[0], key[1], tot[key], chg[key]))
print("changed in total:", sum(chg.values()))
for key in sorted(ex):
    if key[0] not in ("NormL1Fused", "helpers-exact (new)", "scaledNF"):
        print("UNEXPECTED change", key, ex[key][:5])
