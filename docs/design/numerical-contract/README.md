# Numerical contract for 0.6: design record

This folder records how ProximalBase's numerical contract for the 0.6 release
was designed: the drafts, the reviews, and the decisions with their reasons.
It is outside `docs/src`, so Documenter does not build it. The agreed
contract will become `docs/src/numerics.md`, linked from
`docs/src/architecture.md`, and the Precision section of
`docs/src/interface.md` will become a short summary of it.

The contract covers what the package promises about the numbers it returns:
types in mixed precision, overflow and underflow, accuracy, decisions taken
on rounded quantities (thresholds, membership, ranks, searches), the meaning
of NaN, `Inf`, zero and infinite parameters and steps, agreement between
oracles, GPU arrays, and the cost the package accepts for its guarantees.
0.6 is a breaking release, so earlier decisions (#238, #283, #343 and
others) may be reversed here.

## Files, in the order they were written

| file | author | content |
|:---|:---|:---|
| `01-chatgpt-readers-edition.md` | ChatGPT | "Reader's edition, Draft 2": the first full draft, a narrative guide with acceptance cases V1–V9 and S1–S8 |
| `02-claude-review-of-draft2.md` | Claude | review of Draft 2: what to keep, what conflicts with settled decisions, what it leaves out; acceptance cases run at master `3f90a29` |
| `03-claude-draft3.md` | Claude | Draft 3: rules with identifiers and status, merging Draft 2 with the precision design of #366 |
| `04-chatgpt-comment.md` | ChatGPT | comparison of Draft 2 with #366; finds that Draft 3's rule T3 contradicts its case V9, and proposes a middle option for mixed precision |
| `05-decisions.md` | Claude | every decision written out: background, example, options with pros and cons, precedent in other packages, recommendation |
| `06-claude-draft4.md` | Claude | Draft 4: Draft 3 revised for ChatGPT's comparison, with every recommendation of `05-decisions.md` adopted |
| `07-chatgpt-review-of-draft4.md` | ChatGPT | review of Draft 4: approves the converted model (V8, V9 = 0); asks for V10, a defensible rank-projection guarantee, and a decision on `LogDetLoss` |
| `08-claude-draft5.md` | Claude | Draft 5: Draft 4 revised for the review and the maintainer's question about bitwise results |
| `09-chatgpt-review-of-draft5.md` | ChatGPT | review of Draft 5: the wide-argument, narrow-output case, one policy per kind of decision, A5 and O4 bounds, a fixture for decision 39 |
| `10-impact-on-open-issues.md` | Claude (subagent), checked | every open issue and PR classified against the decisions; stale references; gaps in the contract text |
| `11-claude-draft6.md` | Claude | **the current draft**: Draft 5 revised for the review and the screen; decisions 40–47 added to `05` |
| `bench-q1-threshold.jl`, `bench-d6-quadratic.jl` | Claude | measurements for decisions 3 and 6 |

## Process

1. The maintainer answers the decisions in `05-decisions.md` in comments on
   the pull request (#385), where all discussion of the design takes place.
2. Each answer is recorded as a commit to `05-decisions.md` (the "Decided"
   line of that decision), and the draft is revised to match.
3. When every decision is answered, the agreed contract is written to
   `docs/src/numerics.md`, with the interface page's summary and the
   architecture link, in the same pull request. That pull request changes
   documentation only.
4. Implementation follows in separate pull requests, tracked in #366, which
   becomes the tracker for the whole contract. It starts with a reconciliation
   table (oracle, problem after conversion, promised result, acceptance case,
   status) before any patch.

Whether this folder stays after the docs page is written is decided at merge.
`docs/audit/release-0.5.4` is the precedent for keeping such a record.
