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
and BigFloat references across Float32/64. `core_functions` exercises retained
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

`block_value_updates` checks buffered block writes against a plain dense vector,
including active zeros, position/order invariants, prefixes, conversion failures,
overlapping sources, and contiguous, strided, arbitrary and reversed groups.
