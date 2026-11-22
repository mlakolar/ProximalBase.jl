# Repository hygiene and performance evidence

Keep git focused on maintained Julia/Python source, tests, benchmark **drivers**
and small curated documentation. Agents may produce extensive diagnostics in
their local worktrees; those outputs must not be committed to the package.

The required CI gate runs `scripts/check_repo_hygiene.py` in the
change-classification job on the **resulting changed tree** (not all historical
files). The check inspects the blobs, at the checked-out commit, of the paths
that `git diff` reports as added, copied, modified or renamed between the
checked-out commit and the classifier's base:

- For a pull request, the base is the first parent of the tested merge commit.
- For a push to master with a usable `before` commit, the base is `before`.
- For a push without a usable `before` commit (all zeros, missing or not
  fetchable) and for a manual run, the base is the first parent of the
  checked-out commit, if that commit has one.

The check rejects:

- Any newly added or modified tracked file larger than **1 MiB**, or more than
  **5 MiB combined changed file contents** in one pull request/push.
- Compressed archives, binary benchmark datasets and logs.
- Generated output under `benchmark/**/{results,artifacts,evidence,...}/`
  or `docs/audit/**/{results,artifacts,evidence,...}/`, including per-sample
  CSVs and machine-generated receipts.

Already-tracked artifacts are grandfathered unless modified. This prevents
legacy reports from breaking CI; it **does not erase them from git history**.
A legitimate oversized fixture requires a reviewable exact-path, SHA-locked,
reasoned exception in `scripts/check_repo_hygiene.py`, not a blanket waiver.

For performance investigations, commit reusable benchmarks and a concise
Markdown result summary. Preserve necessary raw evidence outside the package
repository (e.g., a durable release attachment) and link it with its SHA256 in
the issue/PR. GitHub Actions artifacts are convenient for short-term review
but have retention limits. Prefer squash merges after temporary artifacts
have been removed from the PR's final tree.

`scripts/audit_repository_history.py` inventories every reachable git blob,
including blobs only in older commits. Its read-only workflow runs monthly,
manually, and when the auditor changes. JSON reports are CI artifacts, never
tracked results. Historical size cleanup, if needed, requires a separate
maintainer decision: no automatic git history rewriting or force-pushing.
