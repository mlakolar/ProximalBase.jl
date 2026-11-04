"""Merge extraction (extract/chunk-*.json) and verification (verify/P*.json)
into the audit index TSVs. Usage: python3 -I build_index.py <audit dir> <out dir>"""
import json, glob, os, sys, re
from collections import Counter, defaultdict

A, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
meta = {m["number"]: m for m in json.load(open(f"{A}/items-meta.json"))}
items, cands = {}, {}
for f in sorted(glob.glob(f"{A}/extract/chunk-*.json")):
    d = json.load(open(f))
    for i in d["items"]:
        items[i["number"]] = i
    for c in d["candidates"]:
        cands[c["id"]] = c
clusters, cmap = {}, {}
for f in sorted(glob.glob(f"{A}/verify/P[0-9].json")):
    d = json.load(open(f))
    for c in d["clusters"]:
        clusters[c["id"]] = c
    cmap.update(d["candidate_map"])
# final overrides by the coordinator (cluster id -> dict of fields), if present
ov_path = f"{A}/verify/overrides.json"
if os.path.exists(ov_path):
    for cid, fields in json.load(open(ov_path)).items():
        clusters[cid].update(fields)

missing = sorted(set(cands) - set(cmap))
unknown = sorted(v for v in set(cmap.values()) if v not in clusters)
print("candidates", len(cands), "mapped", len(set(cands) & set(cmap)), "unmapped", len(missing), missing[:10])
print("map targets missing", unknown[:10])

def clean(s):
    return re.sub(r"[\t\r\n]+", " ", str(s or "")).strip()

ORDER = ["CONFIRMED", "UNVERIFIED", "APPROVED_LIMITATION", "INTENTIONALLY_UNSUPPORTED",
         "FIXED", "OBSOLETE", "CONFORMS", "DUPLICATE", "OUT_OF_SCOPE"]

def item_disposition(n):
    it = items[n]
    cids = it.get("candidate_ids", [])
    if not cids:
        if it["residual"] == "infra-only" or it["item_hint"] == "process-infra":
            return "no package residual (infra/process)", []
        if it["item_hint"].startswith("unmerged"):
            return "no residual (unmerged/abandoned)", []
        return "no residual reported", []
    cl = sorted({cmap.get(c) for c in cids if cmap.get(c)})
    classes = Counter(clusters[c]["classification"] for c in cl if c in clusters)
    lims = sorted({clusters[c].get("lim_id") for c in cl if c in clusters and clusters[c].get("lim_id")})
    parts = [f"{k}:{classes[k]}" for k in ORDER if classes.get(k)]
    disp = "residuals " + ",".join(parts)
    if lims:
        disp += " -> " + ",".join(lims)
    return disp, cl

rows = []
for n in sorted(meta):
    m, it = meta[n], items[n]
    disp, cl = item_disposition(n)
    later = sorted({str(x) for c in it.get("candidate_ids", []) for x in (cands[c].get("possible_fix_refs") or [])})
    related = sorted(set(m["links"]) | set(m["xrefs"]))
    rows.append([n, m["kind"], clean(m["title"]), m["url"], (m["closed"] or "")[:10], clean(m["reason"]),
                 " ".join(f"#{r}" for r in related[:25]),
                 "yes" if it.get("candidate_ids") else "no",
                 clean("; ".join(later))[:300],
                 clean(it.get("what_it_did"))[:200],
                 disp, " ".join(cl)])
hdr = ["number", "kind", "title", "url", "closed", "closing_reason", "related", "residual_reported",
       "later_fix_refs_named", "what_it_did", "audit_disposition", "clusters"]
with open(f"{OUT}/items.tsv", "w") as f:
    f.write("\t".join(hdr) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")

with open(f"{OUT}/clusters.tsv", "w") as f:
    h = ["cluster", "classification", "lim_id", "title", "contract_rule", "owner_issue", "source_items",
         "candidates", "tests_and_fixes", "probe", "observed_1.13", "observed_1.10", "reason"]
    f.write("\t".join(h) + "\n")
    for cid in sorted(clusters, key=lambda k: (k.split("-")[0], int(re.sub(r"\D", "", k.split("-")[1]) or 0))):
        c = clusters[cid]
        ev = c.get("evidence", {}) or {}
        evs = "; ".join(x for x in [", ".join(ev.get("test_refs", []) or []), ", ".join(ev.get("fix_refs", []) or [])] if x)
        # the verifier's probe and what it printed, without the scratch paths
        obs = [clean(ev.get(k)).replace(f"{A}/verify/", "")[:400] for k in ("reproducer", "output_113", "output_110")]
        f.write("\t".join(clean(x) for x in [cid, c["classification"], c.get("lim_id", ""), c["title"],
                c.get("contract_rule", ""), c.get("owner_issue") or "", " ".join(f"#{s}" for s in c.get("source_items", [])),
                " ".join(c["candidate_ids"]), evs[:300], *obs, c.get("classification_reason", "")[:400]]) + "\n")

cc = Counter(c["classification"] for c in clusters.values())
cand_cc = Counter(clusters[cmap[c]]["classification"] for c in cands if c in cmap and cmap[c] in clusters)
print("clusters", len(clusters), dict(cc))
print("candidates by final class", dict(cand_cc))
print("items with residual", sum(1 for i in items.values() if i.get("candidate_ids")))
json.dump({"clusters": dict(cc), "candidates": dict(cand_cc), "n_clusters": len(clusters),
           "n_candidates": len(cands), "n_items": len(items)}, open(f"{OUT}/counts.json", "w"), indent=1)
