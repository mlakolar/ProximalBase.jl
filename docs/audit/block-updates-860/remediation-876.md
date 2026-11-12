# Historical exposure and proposed remediation (#876)

This assessment prepares cleanup; **no historical ref mutation, PR closure or
Support request is authorized or performed**. The clean replacement PR #877
supersedes #867 for integration. It does not purge the original Git objects.

## Verified scope

The clean branch starts at master
`a5010d20446b0fcd518e5ec6f0194d55bd227f21`. An ancestry check for contaminated
commit `cc447af990dfe96e5dd0d6fdcf68891b877f3a01` returns exit status 1 for master
and the replacement. The replacement's `src/` is identical to that master.
The three offending blob IDs below are absent from its reachable object set:

| Historical file | Git blob ID |
|:--|:--|
| `raw-measurements.tar.gz` | `9e75587cb3a25a616391ff85cb1a4100bc4bf82c` |
| `sources-and-codegen.tar.gz` | `f7952ef790bb9ff2d043c1bdb3769c59b61f372a` |
| `validation-logs.tar.gz` | `9a2a3cadf9c4df0d1962145915336dd2f2429057` |

The files were added under `docs/audit/block-updates-860/artifacts/` in the
contaminated commit, then removed in
`81e973c6b0c0ce4277c92348185ff24d21244dd5`. Tree deletion did not erase history.
The prior assessment found environment identifiers, archive ownership/timestamps
and runtime addresses; credential/private-key scans found no matches. Actual
identifying values and unsanitized contents are intentionally omitted here.

A read-only inventory checked 460 advertised refs in the main repository and
three in `tkelman/ProximalBase.jl`, fetching missing objects for ancestry tests.
Exactly these three checked refs retain the contaminated commit:

| Repository ref | Observed tip |
|:--|:--|
| `refs/heads/perf/860-block-updates` | `78af86978cadc473cdf4392fa2a7818c0ba137ef` |
| `refs/pull/867/head` | `78af86978cadc473cdf4392fa2a7818c0ba137ef` |
| `refs/pull/867/merge` | `02b6ca06fb119a5cf9d6b75afd9a6e2fefaa399b` |

No checked tag, master ref, other advertised main-repository ref or checked
fork ref contains it. The audit release tag points to clean `b77f4a4e`.
The fork listing also names `zxj5470-mirrors/ProximalBase.jl`; its Git and REST
ref endpoints returned 404 even with authenticated access. Its contents and
refs therefore remain unverified, not established clean.

The original commit page and historical raw-archive URL returned HTTP 200 to
anonymous HEAD requests. The same commit-page URL under `tkelman/ProximalBase.jl`
also returned 200 despite its clean advertised refs; no page/archive body was
retrieved for these checks. HTTP status alone does not distinguish a retained
network object, cache or rendered page content. Unadvertised refs, inaccessible
forks, caches and other clones are outside the ancestry inventory's guarantees.
Ref tips can change; refresh the inventory before approved cleanup.

## Least-disruptive proposal: approval required

1. Merge the independently reviewed, green clean replacement PR #877.
2. With separate approval, close #867 as superseded and close #860 as investigated,
   explicitly recording that the underlying performance regression is not fixed.
3. With separate approval, retire only `refs/heads/perf/860-block-updates`.
   Prefer deletion after integration; replacing its tip with the clean branch
   is an alternative if the branch name must survive. Do not rewrite master,
   unrelated branches or tags. Preserve the original evidence privately.
4. With separate approval, ask GitHub Support to assess retained PR refs and
   cached objects. Keep #876 open until remaining exposure is resolved or the
   maintainer explicitly accepts it. Recheck refs and public accessibility.

The following are **proposed commands, not executed**. Revalidate the observed
old tip first; the explicit lease rejects a changed branch. PR integration
requires the final reviewed SHA from #877's handoff.

```sh
# After explicit integration approval:
gh pr merge 877 --repo mlakolar/ProximalBase.jl --squash --match-head-commit REVIEWED_SHA
# After explicit approval to supersede the original PR:
gh pr close 867 --repo mlakolar/ProximalBase.jl --comment 'Superseded by #877; historical privacy cleanup remains tracked in #876.'
# Preferred targeted branch retirement, after explicit deletion approval:
git push --force-with-lease=refs/heads/perf/860-block-updates:78af86978cadc473cdf4392fa2a7818c0ba137ef origin :refs/heads/perf/860-block-updates
# Alternative ONLY if replacing that branch is explicitly approved:
git push --force-with-lease=refs/heads/perf/860-block-updates:78af86978cadc473cdf4392fa2a7818c0ba137ef origin refs/heads/audit/860-876-clean-final:refs/heads/perf/860-block-updates
```

Execute one branch alternative, not both. Confirm the replacement tip and its
ancestry before any approved replacement. The new branch/PR, master, unrelated
refs and approved sanitized release must not be removed by this proposal.
The commands do not attempt to push or delete GitHub-managed `refs/pull/*`.

## Support requirements and limits

[GitHub's removal documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
states that PR refs are read-only to pushes. Branch changes or PR closure can
leave cached commits, PR refs and copies elsewhere accessible. Support assesses
sensitive-data eligibility; this metadata-only case has no detected credentials,
and assistance is not guaranteed. A proposed request should identify the
repository, contaminated commit/blobs, affected PR #867 and refs above, completed
ref cleanup and unresolved fork visibility. Ask whether dereferencing can preserve
conversation history. Do not send identifying values or raw archives by default.
Only GitHub can assess server-side retention; local absence is not proof of purge.
