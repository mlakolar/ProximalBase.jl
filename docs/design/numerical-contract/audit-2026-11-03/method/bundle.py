import json, os, sys, re
from collections import defaultdict
D = sys.argv[1]; os.chdir(D)
os.makedirs("items", exist_ok=True)
prs = {p["number"]: p for page in json.load(open("closed-pr-pages.json")) for p in page}
iss = {i["number"]: i for page in json.load(open("closed-issue-pages.json")) for i in page if "pull_request" not in i}
issue_api = {i["number"]: i for page in json.load(open("closed-issue-pages.json")) for i in page}
com = defaultdict(list)
for p in json.load(open("issue-comments-pages.json")):
    for c in p:
        n = int(c["issue_url"].rsplit("/", 1)[1]); com[n].append(c)
rc = defaultdict(list)
for p in json.load(open("review-comments-pages.json")):
    for c in p:
        n = int(c["pull_request_url"].rsplit("/", 1)[1]); rc[n].append(c)
prg = {int(k): v for k, v in json.load(open("pr-graphql.json")).items()}
isg = {int(k): v for k, v in json.load(open("issue-graphql.json")).items()}
resolved = {}
for n, v in prg.items():
    for t in v["reviewThreads"]["nodes"]:
        if t["comments"]["nodes"]:
            resolved[t["comments"]["nodes"][0]["databaseId"]] = (t["isResolved"], t["isOutdated"])
meta = []
def refs(v):
    s = set()
    for t in v.get("timelineItems", {}).get("nodes", []):
        for key in ("source", "subject", "canonical"):
            if t.get(key) and t[key].get("number"): s.add(t[key]["number"])
    return s
for n in sorted(set(prs) | set(iss)):
    kind = "PR" if n in prs else "issue"
    it = prs.get(n) or iss[n]
    ev = []
    for c in com[n]:
        ev.append((c["created_at"], f"### comment by {c['user']['login']} {c['created_at']} (id {c['id']})\n{c['body'] or ''}"))
    if kind == "PR":
        for r in prg[n]["reviews"]["nodes"]:
            if (r["body"] or "").strip():
                ev.append((r["submittedAt"] or "", f"### review {r['state']} by {(r['author'] or {}).get('login')} {r['submittedAt']}\n{r['body']}"))
        for c in rc[n]:
            st = resolved.get(c["id"])
            tag = "" if st is None else f" [thread resolved={st[0]} outdated={st[1]}]"
            ev.append((c["created_at"], f"### inline review comment by {c['user']['login']} {c['created_at']} on {c.get('path')}:{c.get('line') or c.get('original_line')} (discussion_r{c['id']}){tag}\n{c['body'] or ''}"))
    ev.sort(key=lambda x: x[0])
    if kind == "PR":
        closed = it["merged_at"] or it["closed_at"]; reason = "merged" if it["merged_at"] else "closed-unmerged"
        links = sorted(x["number"] for x in prg[n]["closingIssuesReferences"]["nodes"])
        xr = sorted(refs(prg[n]) - {n})
    else:
        closed = it["closed_at"]; reason = it.get("state_reason") or ""
        links = sorted(x["number"] for x in isg[n]["closedByPullRequestsReferences"]["nodes"])
        closers = []
        for t in isg[n]["timelineItems"]["nodes"]:
            if t["__typename"] == "ClosedEvent" and t.get("closer"):
                c = t["closer"]; closers.append(f"PR#{c['number']}" if c.get("number") else f"commit {c.get('oid','')[:10]}")
        xr = sorted(refs(isg[n]) - {n})
        reason = reason + (" by " + ",".join(closers) if closers else "")
    head = f"# {kind} #{n}: {it['title']}\nURL: {it['html_url']}\nState: closed ({reason}) at {closed}; created {it['created_at']} by {it['user']['login']}\nLinked (closing refs): {links}\nCross-referenced by: {xr}\nLabels: {[l['name'] for l in it.get('labels', [])]}\n\n## Body\n{it.get('body') or ''}\n\n## Discussion ({len(ev)} entries)\n"
    text = head + "\n\n".join(e[1] for e in ev) + "\n"
    open(f"items/{n:04d}.md", "w").write(text)
    meta.append(dict(number=n, kind=kind, title=it["title"], url=it["html_url"], closed=closed, reason=reason, links=links, xrefs=xr, size=len(text), n_entries=len(ev)))
json.dump(meta, open("items-meta.json", "w"), indent=0)
tot = sum(m["size"] for m in meta)
print("items", len(meta), "total chars", tot)
big = sorted(meta, key=lambda m: -m["size"])[:15]
for m in big: print(m["number"], m["kind"], m["size"], m["title"][:70])
