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
`numerical_kernels` tests the shared overflow/underflow kernels of utils.jl
(bounded products, halved residuals, squared residuals) against BigFloat
references in Float32/64, including genuine NaN/Inf inputs and exact zeros. `core_functions` exercises retained
interface, lasso building blocks and custom scalar/vector block extensions.
`block_replacement` checks whole-owner copies against independent value/activity
references, layout ownership and cache preservation, conversions and preflight errors.
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
