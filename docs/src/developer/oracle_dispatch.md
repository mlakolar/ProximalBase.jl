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

`NormL1Nonneg` overrides the public scalar-step `prox!` and `value_and_prox!` the same way and
for the same reason, and takes the generic separable loops after the check; an array of steps is
not read by it. Its value reaches the `NormL1` loops through the hook `_l1_value` and its stored
`NormL1`, and a dense array without weights has a method of its own (`_value_dense`).

`IndNonnegative` overrides the two public scalar-step methods in the same way, with a check of its own, and has no
loop of its own: the generic separable loops compute its value and projection, and `value_and_prox!` ends in
`_value_and_prox_sep!`, which evaluates the value at the stored output also for an indicator.

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

Compact-vector `prox_block!` is a public hook implemented by `GroupNormL2`, `GroupNormLinf` and
available to custom functions. The default allocating method supplies an independent
floating `Vector` and delegates. For `GroupNormL2`, the index is its group number
and the compact vectors follow that group's order; custom methods define their
own layout. `GroupNormL2` validates buffer shapes, real one-based storage,
steps and overlap; a custom hook must do the same before writing. A custom
hook alone supplies no full-array lifting, wrapper forwarding or KKT screen.

The solver constructs trials, verifies group layouts, calls the mathematical
hook in independent scratch, writes public storage, then updates residuals and
its own caches. Neither scalar nor vector hook silently activates a coordinate.
For symmetric scalar updates it validates parameters first and converts the
independent ID to the full-entry index; no symmetric iterate is available in the raw hook.

## KKT and restrictions

Retained scalar screens are explicitly implemented for `NormL1`, `NormL1Nonneg`, `ElasticNet`, `SqrNormL2`,
`ZeroFunction` and `LinearFunction`; `GroupNormL2` has block scores, `GroupNormLinf` the margin at a zero group, `NormL2` a full-point form and `IndNonnegative` the three-argument screen only; `Scaled` forwards
to them (below). The three-argument form is the signed
screen of a coordinate assumed zero. For those six functions, the four-argument form on an
array, a `SparseIterate`, a `BlockIterate` (maximum over the group) or a `SymmetricSparseIterate`
evaluates the actual coordinate through one entry kernel per function, and the full-point form
`kkt_violation(f,v,x)` makes a pass over an array point, possibly followed by a pass by entry for non-finite values (for `NormL1Nonneg` also where the natural score of an entry is infinite or `NaN`, through its `_kkt_flag` method); it is not provided for a
`SymmetricSparseIterate` (`ArgumentError`). For a function defined outside the package, the public
four-argument methods read `x`: the three-argument value at a zero coordinate or group, and an
`ArgumentError` at a nonzero one; a method of the function itself is used as defined, and there
is no generic full-point method. Under `Scaled` the method of such a function is not called: the
scaled three-argument value at a zero coordinate or group, and an `ArgumentError` at a nonzero one,
unless the methods for `Scaled{<:TheirType}` are defined. A built-in function without a method, such as `SqrtLeastSquaresLoss`,
throws `MethodError`; under `Scaled` the `MethodError` is for the call on the `Scaled` function.
`GroupNormL2` supplies a native block score instead (one block kernel, `_l2_score`), checking the
selected ordered layout and the full gradient length; its array-point form takes an entry index,
its full-point form checks the whole layout of a `BlockIterate`, and its three-argument form throws
because it is not entrywise separable. `GroupNormLinf` has its own methods of `_kkt_violation` (the margin of the dual norm at a zero group, with the layout checks of `GroupNormL2`) and is not in
`_kkt_class`, so that `Scaled` over it has no method. `NormL2` has the full-point form, the kernel with all of `x`
as one block, and its other forms throw `ArgumentError`. `Scaled` scores a wrapper with a finite
positive factor by evaluating the score of its function on `v/α` and multiplying it by `α`; a nested chain whose product of factors
is not a normal number is scored through the function with the product applied to the gradient
(`LinearFunction`, `SqrNormL2`) or folded into the level (`NormL1`, `NormL1Nonneg`, `ElasticNet`, `GroupNormL2`, `NormL2`),
and refuses the nonzero and full-point forms where there is no fold, as it does for an infinite
factor (the [scaling guide](@ref scaled-kkt)). Dispatch does not
turn a quadratic/least-squares loss into a scalar penalty screen. KKT is the raw first-order score
of the supplied `v`, not a general solver residual: scaling, tolerances and convergence decisions
are the caller's. restrict slices supported parameters; no default exists for custom
functions. A restricted quadratic keeps its constant and assumes omitted
variables zero; vector-response least squares keeps Y and selects columns.

See [the extension tutorial](../extending.md), [symmetric parameters](../symmetric_parameters.md)
and [current pure block contract](contracts/block-prox-0.6.md).
