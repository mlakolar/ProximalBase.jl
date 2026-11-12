# Minimal supported test inventory

Every test runs in a fresh Julia process with bounds checks. This prevents
custom fixture methods from contaminating ambiguity and dispatch checks in
other files. `tiers.toml` is authoritative and rejects undeclared, missing and
duplicate Julia files before any tests execute. All supported tests currently
belong to quick; standard/thorough/full include that inventory.

The preserved storage suites cover scalar/block/symmetric indexing, active
zeros, copies, views, aliases, arbitrary block layouts, writes and products.
`ridge_precision` preserves the frozen independent BigFloat ridge tests and
excludes only constrained-ridge sections. `minimal_math` uses independent dense
and BigFloat references across Float32/64.
`numerical_kernels` tests the shared overflow/underflow kernels of numerical_kernels.jl
(bounded products, halved residuals, squared residuals) against BigFloat
references in Float32/64, including genuine NaN/Inf inputs and exact zeros.
`numerical_recovery` audits the live recovery paths themselves, branch by branch
(`docs/audit/numerical-kernels-file/recovery-coverage.md` lists each one and the coverage evidence):
the `_ExpSum` accumulator and the signed/overflowing/underflowing `QuadraticFunction` values, nested and
single-factor `Scaled` recovery (value parts, level folds, partial folds, array-step routes, exact zeros and
infinite factors), `NormL1` weights and thresholds (and the unweighted entry type, #684), the ridge prox
weights and owners, the SVD prox, the `_pow2_weighted` sums, `norm`/`norm_diff`, the sparse and symmetric three-factor products and the symmetric
part of an asymmetric argument, each against an independent BigFloat evaluation of the formula in Float32 and
Float64. Known defects are recorded with `@test_broken` against the correct reference and flip when fixed.
`symmetric_parameters` covers symmetric per-entry parameters as a model precondition: valid
symmetric models (`NormL1`, `SqrNormL2`, `LinearFunction`, nested `Scaled` and a custom separable
function) against dense and structurally symmetric references in Float32/64, the explicit `check_symmetric_parameters` (stored and unstored pairs, nested `Scaled`,
NaN and signed zeros, malformed metadata), the structural refusals that the oracles keep, and
instrumented parameter arrays that count reads to show the oracles perform no mirrored-pair scan.
`core_functions` exercises retained
interface, lasso building blocks and custom scalar/vector block extensions.
`dropzeros_compaction` checks the shared `dropzeros!` loop of `SparseIterate` and
`SymmetricSparseIterate` against an independent copy of the previous loop (count, live
values and IDs in order, reverse map) and a dense reference, with later use and allocations.
`block_replacement` checks whole-iterate copies against independent value/activity
references, layout ownership and cache preservation, conversions and preflight errors.
`block_value_updates` checks buffered writes against independent dense values,
including activation, active zeros, aliases, conversion/partial failures, custom
callbacks, offset axes and live block views after whole-iterate replacement.
`cache_and_blocks` protects cache reset after referenced data mutations and
block-coordinate KKT semantics. `ci_contract` protects required Linux gates and
inventory rejection. The local helper contains only allocation measurement and
frozen storage bookkeeping checks, with no packaged Testing extension.

This is the complete minimal suite, not the historical package full suite.
The removed historical catalog is preserved at the frozen baseline and
legacy/full. No retained failure is skipped to accommodate the reduction.

## CI sharding

The full declared inventory is split across three Linux/current-Julia runners.
The shards are computed from the same sorted inventory as the unsharded local
run, round-robin, so each declared test runs exactly once across the three
shards. Every shard validates the complete inventory before selecting files,
and each file still runs in a fresh Julia process with bounds checks enabled.

Locally, run all tests with `julia scripts/test.jl`, or reproduce a CI shard
with `julia scripts/test.jl --shard=1/3` (similarly `2/3` and `3/3`).
The inventory/partition contract is checked by `ci_contract`. All three
shards, the frozen differential check, and the strict documentation build are
required by the existing PR verdict and merge gate. PR builds may warm
independent Julia package caches for the three validation job types.
