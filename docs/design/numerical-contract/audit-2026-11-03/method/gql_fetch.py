import json, subprocess, sys, os
D = sys.argv[1]
os.chdir(D)
prs = [p for page in json.load(open("closed-pr-pages.json")) for p in page]
iss = [i for page in json.load(open("closed-issue-pages.json")) for i in page if "pull_request" not in i]

def gql(q):
    out = subprocess.run(["gh", "api", "graphql", "-f", "query=" + q], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr); raise SystemExit(1)
    return json.loads(out.stdout)

PRF = """number title
 reviews(first:100){nodes{author{login} state submittedAt body}}
 reviewThreads(first:100){nodes{isResolved isOutdated path line comments(first:1){nodes{databaseId}}}}
 closingIssuesReferences(first:20){nodes{number}}
 timelineItems(first:100, itemTypes:[CROSS_REFERENCED_EVENT, CONNECTED_EVENT]){nodes{__typename
   ... on CrossReferencedEvent{source{__typename ... on Issue{number} ... on PullRequest{number}}}
   ... on ConnectedEvent{subject{__typename ... on Issue{number} ... on PullRequest{number}}}}}"""
ISF = """number title stateReason
 closedByPullRequestsReferences(first:10, includeClosedPrs:true){nodes{number}}
 timelineItems(first:100, itemTypes:[CLOSED_EVENT, CROSS_REFERENCED_EVENT, MARKED_AS_DUPLICATE_EVENT, CONNECTED_EVENT]){nodes{__typename
   ... on ClosedEvent{createdAt stateReason closer{__typename ... on PullRequest{number} ... on Commit{oid}}}
   ... on CrossReferencedEvent{source{__typename ... on Issue{number} ... on PullRequest{number}}}
   ... on MarkedAsDuplicateEvent{canonical{__typename ... on Issue{number} ... on PullRequest{number}}}
   ... on ConnectedEvent{subject{__typename ... on Issue{number} ... on PullRequest{number}}}}}"""

def batch(nums, kind, fields, size):
    res = {}
    for k in range(0, len(nums), size):
        chunk = nums[k:k+size]
        body = " ".join(f"n{n}: {kind}(number:{n}){{ {fields} }}" for n in chunk)
        q = f'query {{ repository(owner:"mlakolar", name:"ProximalBase.jl") {{ {body} }} }}'
        data = gql(q)["data"]["repository"]
        for key, v in data.items():
            res[int(key[1:])] = v
        print(kind, k + len(chunk), file=sys.stderr)
    return res

pr_extra = batch(sorted(p["number"] for p in prs), "pullRequest", PRF, 15)
json.dump(pr_extra, open("pr-graphql.json", "w"))
is_extra = batch(sorted(i["number"] for i in iss), "issue", ISF, 25)
json.dump(is_extra, open("issue-graphql.json", "w"))
# sanity: any connection truncated?
for n, v in pr_extra.items():
    for f in ("reviews", "reviewThreads", "timelineItems"):
        if len(v[f]["nodes"]) >= 100: print("TRUNCATED PR", n, f)
for n, v in is_extra.items():
    if len(v["timelineItems"]["nodes"]) >= 100: print("TRUNCATED issue", n)
print("done", len(pr_extra), len(is_extra))
