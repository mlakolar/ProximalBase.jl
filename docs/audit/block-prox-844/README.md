# #844 implementation and PB-only performance validation

PR #848 implements pure `prox_block[!]` and the approved PB 0.6 names
`BlockIterate`, `blockview` and `blockindices`, with no compatibility aliases.
The mathematical layer computes compact numerical results. Public iterate
writers maintain values/activity; algorithms own trials, residuals and policy.
Independent numerical output is a caller precondition, including when public
`activevalues` accessors expose storage-backed views.

The current performance follow-up corrects writer benchmark setup, removes
the custom inactive writer, and evaluates already-active fast paths. Those writer trials are now deferred
after confirmed ordinary-path regressions. The remaining candidate optimizes
small validated GroupNormL2 CPU calls and preserves original checked wrapper
routing; it is evaluated independently. No new
view type, public writer, unchecked operation or #426 protection is introduced.
See [the investigation](block-update-performance.md).

## Benchmark correction

Earlier `inactive_nonzero` probes did not reliably begin inactive:
`fill!(x,0)` preserves activity. The driver now clears zero-valued active groups
with `dropzeros!` outside timing and validates starting values, activity and
order after repeated writes. Earlier inactive-nonzero timing claims are
superseded; inactive-zero and active-zero regressions remain valid evidence.

## Evidence and reproducibility

[Archive manifest](measurements/archive.json) records checksums and the immutable
source of raw evidence removed from the review tree. The durable archive is
[the audit branch](https://github.com/mlakolar/ProximalBase.jl/tree/audit/pb848-block-updates-20261111),
initially pinned at `b6d38cedd746f775c28571816f18a87c18797fff`. It retains the full
historical audit, rejected trials, raw timings and validation records; it has
no CI artifact expiry. The working PR retains benchmark/reproduction scripts,
compact outcomes and checksums. This reduces the review diff without rewriting
existing Git commits or claiming that old Git objects have been removed.

The [original full audit history](https://github.com/mlakolar/ProximalBase.jl/blob/b6d38cedd746f775c28571816f18a87c18797fff/docs/audit/block-prox-844/README.md)
records the initial rename and earlier PB/PS investigation. Its old names,
pins and measurements are historical evidence, not current downstream claims.

## Gates and scope

PB merges require PB correctness, applicable PB tests/CI, documentation,
performance acceptance and independent maintainer review. Under
[PB #527](https://github.com/mlakolar/ProximalBase.jl/issues/527), downstream
migration and exact-pin green integration are release requirements, not gates
for each breaking PB master merge. PS #238 owns its future naming migration;
PS #230 and PB #527 retain wider release integration. No downstream gate is
waived and no ProximalSolvers source, pins, tests, docs, CI or PR #237 is changed.

The corrected mathematical candidate's full matrices and independent
confirmations are complete; compact results and immutable raw evidence are
linked from the investigation. Writer optimization is deferred. The original
bulk/whole-solver performance gate and independent review remain pending.
The PR stays draft and unmerged. No Claude is used.
