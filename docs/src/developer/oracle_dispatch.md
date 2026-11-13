# Function traits, dispatch and coordinate oracles

## Capability declarations and method dispatch

The first four traits default false on `AbstractFunction` types, with instance
forwarding. `iszeropreserving` defaults false on instances. `SeparableFunction`
sets separability/proximability true and supplies generic values/prox from two
scalar hooks. It does not supply gradients, KKT, restrict or bounds.

Traits inform algorithms; concrete Julia methods implement calls. A retained
`SqrNormL2` has `isseparable`=true and scalar `prox_block` but no `value_entry` or
array-step prox. A custom block-only hook has an allocating wrapper but no
full-array prox. Symmetric-iterate methods are explicit, since the symmetric iterate is not an array.
Avoid broad method declarations that make applicability appear wider than the
implementation's actual supported domain.

## Generic defaults and public boundaries

The general `gradient!` fallback checks output/input, calls the fused gradient
and returns out. Implement `value_and_gradient!` if relying on it. Allocating
variants allocate a dense gradient. The default fused prox computes `prox!` then
the value at out; its scalar return is not a `(value,out)` tuple. There is no
allocating value_and_prox binding.

Custom public overrides replace the checked boundary. Validate real types,
shape, parameter extents and aliases yourself. Generic `check_parameters` defaults to
nothing; for a parameterized extension implement it. Symmetry of per-entry
parameters is a precondition of the model: no public oracle, generic or built-in,
compares mirrored parameter entries, so an override has nothing to add, and the
caller or solver runs `check_symmetric_parameters` once per model. For a custom
`SeparableFunction`, `entry_parameters` must be declared so that explicit check
can validate it. Built-ins have private routes that skip the public preflight
once it has run; those private helpers are not extension targets. Generic custom
dispatch calls public methods rather than requiring an `_isbuiltin` escape.

Array steps are a distinct method family: generic separable and zero functions
accept entry steps; `Scaled` forwards when its leaf supports them. Symmetric-iterate array
steps are rejected before writes. No trait automatically supplies this dispatch.

`ElasticNet` overrides the public scalar-step `prox!` and `value_and_prox!` only to
refuse a negative or `NaN` step before writes, then takes the checked separable route;
its private routes (used by `Scaled`) and `prox_block` do not check the step. Its
scalar-step array kernels form the ridge factor of the step once per call, as
`SqrNormL2` does, and apply it in the separable loops after the soft threshold of
each entry. For an argument of a concrete element type, the number computed for each
entry equals `prox_block(f, x[i], i, γ)` bit for bit, before its conversion to the
element type of the destination and its storage (a new `SparseIterate` destination
does not store a zero result, so that entry reads as `+0`). An argument of another
element type, and an array of steps, run `prox_block` per entry. A `Scaled`
`ElasticNet` takes the value recovery of every function first; where that gives no finite
value, or `0` at a point where the function is not exactly `0`, for a finite positive
factor, it adds the scaled values of the ℓ₁ part and of the ridge part, each recovered as
its own family recovers it and rounded on its own, and the sum replaces the first result
only if it is finite after conversion to the type of that result. The folds keep an
infinite level, so that the other level can fold; a fold that overflows is not used.

## Pure hooks and storage kernels

Raw scalar `value_entry`/`prox_block` use **global entry indices**. Generic sparse
kernels pass stored coordinate IDs for `SparseIterate`, global members for blocks,
and converted full matrix indices for symmetric-iterate variables. Symmetric values count
both mirrored entry terms. A declared zero-preserving function can skip inactive
zeros. No kernel reads parameter entries to compare mirrored pairs.

Compact-vector `prox_block!` is a public extension hook with no retained built-in
implementation. The default allocating method supplies an independent floating
Vector and delegates. Its index denotes a custom block and its vectors follow
that block's order. The hook author validates buffer shapes/types/overlap;
no generic full-array lifting, wrapper forwarding or KKT follows from it.

The solver constructs trials, verifies group layouts, calls the mathematical
hook in independent scratch, writes public storage, then updates residuals and
its own caches. Neither scalar nor vector hook silently activates a coordinate.
For symmetric scalar updates it validates parameters first and converts the
independent ID to the full-entry index; no symmetric iterate is available in the raw hook.

## KKT and restrictions

Retained scalar screens are explicitly implemented for `NormL1`, `ElasticNet`, `SqrNormL2`,
`ZeroFunction` and `LinearFunction`; `Scaled` forwards. Four-argument dispatch adds
coordinate semantics and checks, or maximum-over-group for `BlockIterate`.
It does not turn a quadratic/least-squares loss into a scalar penalty screen.
KKT means a signed screen at a coordinate assumed zero, not a general solver
residual. restrict slices supported parameters; no default exists for custom
functions. A restricted quadratic keeps its constant and assumes omitted
variables zero; vector-response least squares keeps Y and selects columns.

See [the extension tutorial](../extending.md), [symmetric parameters](../symmetric_parameters.md)
and [current pure block contract](contracts/block-prox-0.6.md).
