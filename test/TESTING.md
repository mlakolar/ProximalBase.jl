# Minimal supported test inventory

Every test runs in a fresh Julia process with bounds checks. This prevents
custom fixture methods from contaminating ambiguity and dispatch checks in
other files. `tiers.toml` is authoritative and rejects undeclared, missing and
duplicate Julia files before any tests execute. All supported tests currently
belong to quick; standard/thorough/full include that inventory.

The preserved storage suites cover scalar/block/symmetric indexing, active
zeros, copies, views, aliases, arbitrary block layouts, writes and products.
`iterate_queries` checks `SparseCoordinates` sorting, display and array
indexing against plain vectors for vector and matrix owners, including
exception type parity and preservation of the working set on reads.
`symmetric_iterate_interface` also checks finite and infinite `Scaled` gradients
in Float32/64, including active-zero membership, IEEE activation of implicit
zeros, retained finite exponent products, and validation before writes.
`ridge_precision` preserves the frozen independent BigFloat ridge tests and
excludes only constrained-ridge sections. `minimal_math` uses independent dense
and BigFloat references across Float32/64.
`numerical_kernels` tests the shared overflow/underflow kernels of numerical_kernels.jl
(bounded products, halved residuals, squared residuals, two-term exponent sums) against BigFloat
references in Float32/64, including genuine NaN/Inf inputs and exact zeros.
`quadratic_domain` tests `QuadraticFunction` for a symmetric `A` that need not be positive
semidefinite: values, gradients and the proximal point against analytic and BigFloat references,
the step domain of the proximal operator (accepted and rejected steps at the boundary, outputs
preserved on `DomainError`, `Scaled` and nested wrappers, `reset_cache!` after a change of
definiteness), the empty matrix, the traits and warm-call allocations, in Float32/64.
`numerical_recovery` audits the live recovery paths themselves, branch by branch
(`docs/audit/numerical-kernels-file/recovery-coverage.md` lists each one and the coverage evidence):
the `_ExpSum` accumulator and the signed/overflowing/underflowing `QuadraticFunction` values, nested and
single-factor `Scaled` recovery (value parts, level folds, partial folds, array-step routes, exact zeros and
infinite factors), `NormL1` weights and thresholds (and the unweighted entry type, #684), `NormL1` values
whose terms round below floatmin on every representation (#373), the ridge prox
weights and owners, the SVD prox, the recomputed-dot shift, `norm`/`norm_diff`, the sparse and symmetric three-factor products and the symmetric
part of an asymmetric argument, each against an independent BigFloat evaluation of the formula in Float32 and
Float64. Known defects are recorded with `@test_broken` against the correct reference and flip when fixed.
`symmetric_parameters` covers symmetric per-entry parameters as a model precondition: valid
symmetric models (`NormL1`, `SqrNormL2`, `LinearFunction`, nested `Scaled` and a custom separable
function) against dense and structurally symmetric references in Float32/64, the explicit `check_symmetric_parameters` (stored and unstored pairs, nested `Scaled`,
NaN and signed zeros, malformed metadata), the structural refusals that the oracles keep, and
instrumented parameter arrays that count reads to show the oracles perform no mirrored-pair scan.
`elasticnet` and `elasticnet_extremes` cover `ElasticNet` (#908): the constructor, traits, analytic cases (at steps
`γ ≥ 0` and one element type, a zero `λ2` is `NormL1` bit for bit, and with a finite `λ2` an entry with a zero weight,
or with a zero `λ1` and a finite weight, is the `SqrNormL2` ridge bit for bit, except that `-0` gives `+0`; an
infinite weight fixes finite inputs at `+0` and an infinite `λ2` follows its own rule), values whose two parts round
below the subnormal spacing on their own, the refusal of negative
and NaN scalar steps before writes, dense, sparse, block and symmetric representations, the KKT screen and
restriction, agreement with the archived implementation within 4 ulp, warmed allocations, the endpoint truth
table of the plan, Float32/Float64 extremes against 4096-bit BigFloat (proximal points within 4 ulp of
`shrink(v, t̂)/(1 + γλ2)` with the kernel's threshold, values within `(n + 2) eps`), `Scaled` folds, chains,
infinite and zero factors, an infinite level beside a finite one, levels that need different folds, and read
counts that show values and proximal points on sparse and symmetric iterates read the weights of the stored
entries only.
`norml1nonneg` and `norml1nonneg_extremes` cover `NormL1Nonneg` (#947) in Float32/64: the constructor and traits,
analytic cases, proximal points against a golden-section search of the constrained objective and against `max(v - t̂, 0)`
in 4096-bit BigFloat with the kernel's threshold (within 4 ulp), the refusal of negative and `NaN` scalar steps before
writes, dense, sparse, block and symmetric representations with read counts, `value_and_prox!` against the value at the
stored output, restriction and inferred types, the archived implementation (proximal points bit for bit where the
infinite-level rule did not change them, the changed rows asserted as changes), warmed allocations, the endpoint truth
table of the plan, values against BigFloat on both sides of the loop switches and at each early return of the
unweighted method (`NaN` before `Inf`, within `(n + 2) eps`), terms below `floatmin` (#373), the point-aware KKT scores
and the full-point measure (an infinite entry at the first, a middle and the last position, array against view against
the entry form), and `Scaled` folds, chains, infinite and zero factors and the hard route of an infinite level.

`indnonnegative` covers `IndNonnegative` (#948) in Float32/64 against references written with comparisons, not the
kernels: the truth table over the infinities, extremes, subnormals, signed zeros and `NaN` (value `NaN`, `Inf` or `0`,
projection `max(v, 0)` with `-0` and `-Inf` giving `+0`), integer inputs, the admitted and refused scalar steps (refusal
before writes, also with an aliased argument and on iterates), the routes that do not check the step (an array of
steps, `prox_block`, `Scaled`), dense, view, reshape, `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate`
arguments, a counting element type that shows the value, and the proximal operator between iterates of the same kind, read stored entries only, validation
before writes (size, complex entries, offset axes), the value returned by `value_and_prox!` equal to the value at the
stored output (`NaN` included), `Scaled` with finite, huge and infinite factors and nested chains, the screen and the
`MethodError` of the forms with a point, traits, inference and warmed allocations.
`kkt_point` tests the three forms of `kkt_violation` (#970) for `NormL1`, `ElasticNet`, `ZeroFunction`, `LinearFunction`
and `SqrNormL2` in Float32/64: the worked example of the contract, the signed margin at a zero entry against the
nonnegative mismatch at a nonzero one, `ElasticNet(λ1, 0)` against `NormL1(λ1)` bit for bit, mixed element and
parameter types against BigFloat within `k eps(W) (|v| + |λ2 x| + c)`, the full-point value against `max(0, max score)`,
the two loops of ProximalSolvers (bit for bit, unweighted), a point from a proximal step, the cancellation case, empty
arrays, matrices, `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate`, NaN and infinite entries of `v` and `x`
(the precedence rules, at the first, a middle and the last position of a vector long enough to take the loop of natural scores and the pass by
entry), infinite coefficients, levels and weights, signed zeros and subnormals, validation of `GroupNormL2` and `Scaled`
at a nonzero point (their scores are in `kkt_block`), functions defined outside the package, no mutation, ranges as
inputs, inference and zero allocations.
`kkt_block` tests the point-aware forms of `kkt_violation` (#970) for `GroupNormL2`, `NormL2` and `Scaled` in Float32/64.
Blocks are checked against a BigFloat evaluation of the definition (the signed screen `‖v_G‖₂ - c` at a zero block, whose
positive part is the distance from `-v_G` to the subdifferential, and that distance at a nonzero block, with `c` the exact
product of level and weight) within `(n + 8) u (‖v_G‖₂ + c) + 4 n nextfloat(zero(W))`, with `u = eps(W)/2`. The
excluded band is `(1-100eps(W))*floatmax(W) < abs(ref) <= (1+100eps(W))*floatmax(W)`; references beyond it must overflow with
the correct sign. The helper is itself tested on wrong values. The magnitudes run from subnormal to `floatmax` (squares that overflow and underflow), the block
sizes from 1 to 64, and the coefficients zero, tiny, large and infinite: the group example of the contract, the full-point value against the maximum of the local scores and the entries
in no group, `SparseIterate` and `BlockIterate` points, NaN and infinite entries and their precedence, signed zeros and
subnormals, validation and layout, `NormL2` (the full-point form, and the errors of the other forms), `Scaled` over each
leaf against the BigFloat definition of the scaled function and its stated domain, chains of wrappers whose factors do not
multiply to a normal number (the product applied to the gradient of `LinearFunction` and `SqrNormL2`, folded into
the level of the others where it gives a normal one, an `ArgumentError` where it does not), infinite factors (also at an entry in no group), leaves without a
score, functions defined outside the package under `Scaled` (with a screen alone, with four-argument and full-point methods of
their own, and with methods for `Scaled{<:T}`), out-of-range indices through the refusing routes, mixed element types and result
types, `@inferred` (also of the class of a `Scaled`), zero allocations of the dense forms and of the cold paths of the block
kernel, entries converted to the working type before they are classified or measured (integer extremes, `UInt128`, `Float16`
with an integer point, `Rational` and `BigInt` points), the merging of an infinite factor by the constructors, the refusals of hand-built stacks with an infinite factor (nonzero points of
each point type and the full-point form), and malformed parameters of a symmetric model under `Scaled`.
`kkt_linf` tests the full-point form of `kkt_violation` (#970) for `NormLinf` and for `Scaled` over it in Float32/64: the Euclidean
distance from `-v` to the subdifferential of `λ max|xᵢ|`, against a BigFloat evaluation with sort-based levels of the ball and the
simplex within `(2n + √n + 5) u (‖v‖₂ + λ) + 4 n nextfloat(zero(W))` (six families of data, `n` from 1 to 200), the worked examples,
a face with one entry far below the level and a bulk of equal entries (`n` up to 2,000,001, an exact reference in closed form, with
measured guards that a shift of the level by the smallest face entry exceeds in vectorized and in scalar summation) and the same shape with random entries,
the zero point against the norm of the proximal point, stationary points built from the proximal operator, the steepest descent
rate along sampled directions and along the direction of the nearest subgradient, faces of one, several and all entries
with mixed signs, both branches of the face (each checked on its inputs), a tie broken by one ulp, `x = 0` inside, on and
outside the ball, `λ = 0` and `λ = Inf`, signed zeros, subnormals, `typemin(Int)`, mixed element types, BigFloat, Float16 at small sizes and the refusal of the lengths for which `n + 1` is not finite in Float16, the precedence
of `NaN` and infinite entries, ranges (a residual far below the level, a level near `floatmax`, sums that overflow, stationary
points with a zero residual), matrices, `SparseIterate` and `BlockIterate` points, the refusal of the three-argument and four-argument forms and of a
`SymmetricSparseIterate`, no mutation of `v`, `x` or the buffer of the object (also a buffer populated by a proximal call), zero allocations for same-type `Float32` and `Float64` arrays with a level of a hardware float type, and inference, `Scaled` (one
factor, the forms of the leaf, an infinite factor, retained chains that fold and that do not, a symmetric point, whose message does not advise the
four-argument form), a head above large ties (up to `10^6` entries) with a level near them, where the first level of all the entries rounds near the ties:
at the zero point, on a face of all the entries and on a face with entries off it, through `Scaled` (one factor, and a chain that folds) and
on the scaled path, a grid of such clusters around the breakpoint, a two-entry input, the closed form of a face inside its range and faces of
nearly equal entries at its boundary, against exact distances in BigFloat, and the absence of ambiguities.
`kkt_smooth` tests full-point `kkt_violation` for `LeastSquaresLoss` and
`QuadraticFunction` in Float32/64. It checks analytic and BigFloat references,
bit parity with the gradient oracle, vector and multiresponse layouts, sparse
and structured products, exceptional and invalid inputs, scratch aliases,
cold and warm allocations, subsequent value, gradient and prox calls, and
the `Scaled` and custom dispatch routes.
`groupnorml2` checks `GroupNormL2` against independent BigFloat evaluations of
its value, group soft thresholding and group KKT screen in Float32/64: analytical
and threshold cases, singleton groups against weighted `NormL1`, noncontiguous
groups, in-place and aliased buffers, preflight failures that leave outputs
unchanged (invalid steps included), exceptional parameters and values, norms and
thresholds that overflow or underflow, values whose terms round below `floatmin`,
direct and fused values swept across `floatmax` (outside a band of 32 eps around the
overflow threshold, the class and accuracy of the BigFloat value; within it, `floatmax`
or `Inf`, #904) and subnormal values, zero-weight and infinite-weight groups near
`floatmax`, and the separation of the pure compact `prox_block!` from `BlockIterate`
storage.
`groupnormlinf` checks `GroupNormLinf` against a sort-based ℓ1-ball reference in BigFloat, written in the file, in Float32/64: one group
equal to `NormLinf` and singleton groups equal to `NormL1` (the proximal outputs bit for bit, the values within a tolerance), irregular range and permuted
partitions with entries in no group, a group above 1024 candidates (Michelot's pass), zero and infinite weights, the compact `prox_block!`
against the array operator, the endpoint rows and their combinations (a NaN or infinite group beside ordinary ones), radii that overflow or
underflow in two of their factors, sums above `floatmax/2`, the recomputed values, the fused value against the value at the stored output,
`BlockIterate` and `SparseIterate` arguments and outputs, the use of the object's buffer, `Scaled`, the margin at a zero group and the
refusals at a nonzero group, validation before writes and zero warmed allocations (arrays, a `Matrix`, irregular partitions, in place, and sweeps of
`prox_block!` over groups of different sizes).
`norml2` checks `NormL2` against BigFloat: values on arrays and on the three
iterates (stored zeros, empty iterates, symmetric off-diagonal multiplicity, huge and
subnormal entries), proximal operators, invalid steps, the symmetric-iterate
refusals, its equality with one group of `GroupNormL2` in ordinary and
exceptional cases, direct values across `floatmax` and in the subnormal range on
arrays and on sparse and symmetric iterates, fused values on arrays, and a Float32
sum of 2^25 + 1024 squares. Most checks compare with BigFloat values; the NaN, Inf
and zero cases use their analytical results. `sparse_value_cost` ports the counting element type of the
historical suite: the `NormL2` value of a 100 × 100 sparse or symmetric iterate with a
few stored entries creates no zero of that type, on the ordinary and the recovery
paths (#808).
`leastsquares_predictor` checks the least-squares oracles at a caller-supplied predictor or residual
(`value_from_predictor`, `value_and_gradient_from_predictor!`, `value_from_residual`, #915) and the accessor
`leastsquares_data` in Float32/64, raw and `Scaled`, against 256-bit BigFloat references formed from the exact
bits of X, Y, z (or r) and the scale as supplied: ordinary, exact-fit, near-fit, cancelling, underflowing,
overflowing, subnormal and one-large-residual data, extreme and subnormal scales, Float64 scales with Float32
data, `Int` scales above 2^24 and the integer that Float64 does not hold (refused with the other unsupported
scales), the #609 inputs that `Scaled(f, α)(β)` misses, the overflow threshold (the boundary region where either `floatmax` or `Inf` is accepted, with a value below it that must be finite and one beyond it that must be `Inf`), non-finite entries, the documented range limits of the
gradient and of an overflowing `z - Y`, agreement of `f(β)`, `value_and_gradient!` and the oracles on the same
predictor (for the raw loss each value within γ_n of the reference of the same rounded residual, because the order of
additions can differ between compiled methods, and the gradients bit for bit; for a `Scaled` loss each value against
the exact-difference reference with `value_ok` and the gradients checked componentwise by
`abs(gs[j] - gl[j]) <= 4 * eps(T) * abs(gl[j]) + floatmin(T)`),
the products counted with a strided matrix type (none for the value oracles, one by Xᵀ for the combined one, and
the products of the existing methods), the values of a `CountMat` design and a plain matrix each compared to
α Σ dᵢ² / 2 of its own residual within γ_n (α = 1 for the raw loss; the order of additions can differ between
compiled methods) with the residual and the gradient α Xᵀd bit for bit, and integer data (partial sums exact in any
order) that compare value, residual and gradient of both designs bit for bit with the exact values, the scratch of the loss left unread and unwritten (also as a view that is
the predictor), dimension, scale and overlap errors that leave `g`, `d` and the scratch unchanged, the accepted and
refused layouts of `Base.mightalias`, empty designs, tasks that share one loss with distinct buffers (one thread
here), the domain refusals (`MethodError`), public but unexported names, and warmed zero allocation and inference.
It also checks the ordinary path at its acceptance bounds (sums near `n*floatmin/eps` and near the bound of the
exponent test, which is also checked directly; on every input the values of the oracles satisfy `value_ok` against
the reference (the finite error bound for `abs(ref) * (1 + e) <= floatmax(T)` with `e = (n + 8) * eps(T)` and
the subnormal allowance, and the class rules otherwise, including an accurate finite value or `+Inf` close to the
overflow threshold even just below it), have type `T` and
are `NaN`, `+Inf` or nonnegative as the entries require, and on the inputs with at most two nonzero entries, among
them 64 entries with one nonzero entry at both bounds, they agree under `isequal` with the kernel called directly;
`f(β)` and `value_and_gradient!` of the unscaled loss at a finite `r` agree under `isequal` with the oracles on those
inputs, and are checked against the reference on the others), offset axes (an offset view of the predictor, the residual or the
response refused with `DimensionMismatch` where the axes differ, offset outputs and a row- or column-offset design
refused by the combined oracle before any write, and equal
offset axes accepted), the accuracy of the sums of squares of a constant vector of length 10^6 through the three
oracles, `f(β)` and `value_and_gradient!`, and the classification at the top of the range of Float32 values from
sums of 2^16 to 2^20 + 1 squares with Float64, `Int` and Float32 scales (the oracles must give `Inf` where the
exact value overflows and an accurate finite value just below `floatmax`).
`sqrt_leastsquares` checks `SqrtLeastSquaresLoss` (#960): construction (types and element-type promotion,
dimension, complex and matrix-response refusals, one-based axes of `Y`, `X` and the scratch through the positional
constructor, empty losses, `Scaled(f, 0)`), the traits, the worked examples, `f(β)` for dense, `SparseIterate` and
`BlockIterate` `β` and the value oracles at `X * β` and at the residuals of either sign, each checked against the
reference within `(n/2 + 4) eps(T)` (the two signs of one residual bit for bit), and mixed-type, sparse, integer and `Float16` data, the value oracles `value_from_predictor` and `value_from_residual` and the accessor
`leastsquares_data` in Float32/64, raw and `Scaled`, against 256-bit BigFloat references formed from the exact bits of
`Y`, `z` (or `r`) and the factor as supplied, within `(n/2 + 4) eps(T)`: ordinary, exact-fit, near-fit, cancelling,
underflowing and overflowing sums of squares (the norm finite where the sum of squares is not), subnormal,
one-large-residual and near-`floatmax` data, extreme and subnormal factors, the edges of the two ordinary-path tests
(sums near `n*floatmin/eps`, values near `floatmax/2` and `floatmax`; `f(r)` of the unscaled loss bit for bit with
the oracle where at most two entries are nonzero), the overflow threshold of the value with its
boundary region, vectors of `2^17` and `2^17 + 1` entries and of 10^6 equal entries, non-finite entries (`NaN`, and
`Inf` of the same and of opposite signs in `z` and `Y`), the domain (lengths, offset axes, element types, non-strided
arrays, a nested `Scaled`, the factors that are refused also at an exact fit and with `Inf`), the design and the scratch
left unread and unwritten (a counting matrix, a design of `NaN`, a sentinel in the scratch, a view of the scratch as the
predictor, arguments unchanged bitwise), the oracles that the loss does not provide (`MethodError`, `lipschitz`, and
`reset_cache!`), and warmed zero allocation and inference, also above `2^17` entries.
`linear_shapes` checks the `LinearFunction` shape contract, `size(c) == size(x)` (the
p × p matrix of a symmetric iterate): every value, gradient and proximal oracle, allocating
and in place, `Scaled` included, on dense, sparse, wrapped and iterate arguments in Float32/64,
against BigFloat references; same-length pairs of other shapes are refused with an unchanged output.
`linear_sparse_value` checks the `LinearFunction` value at a matrix `SparseIterate` (#810): the strong-zero rule
(a position that is not stored, or is stored as `+0` or `-0`, adds nothing, even against a `NaN` or infinite
coefficient; a nonzero entry keeps ordinary arithmetic), the recovery of an overflowing dot with finite participating
entries, the `Diagonal` coefficient array, cancellation, underflow and the overflow boundary against BigFloat
references computed from the stored entries in Float32/64, warmed allocations, the oracles that reach the value, the
shape refusals and unchanged results for dense, vector and symmetric arguments.
`sparse_value_cost` also restores the deterministic guard for that value: a counted `zero` and a coefficient array that
records its reads show, in the cases tested, that no zero is formed for an unstored entry and that the
coefficients are read at the nonzero stored entries (at the nonzero stored diagonal entries for a `Diagonal`) and nowhere
else, on the ordinary and the recovery paths, with a control that catches a dense pass.
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

`mcp` checks the minimax concave penalty and its global proximal operator
against a `BigFloat` evaluation of the full objective on each region and a
separate numerical search. It covers `Float32` and `Float64` ties and adjacent
inputs, parameter and step endpoints, overflow and subnormals, stored-output
fused values, and pre-write validation for arrays and iterates.

This is the complete minimal suite, not the historical package full suite.
The removed historical catalog is preserved at the frozen baseline and
legacy/full. No retained failure is skipped to accommodate the reduction.

## CI sharding

The full declared inventory is split across three Linux/current-Julia runners.
The runner assigns selected files longest-processing-time first. It considers
files in descending weight order, breaking ties by file name, and assigns each
file to the shard with the smallest cumulative weight, breaking ties by shard
index. Within each shard, files run in their original sorted selection order.
Each declared test is assigned to exactly one of the three full CI shards.
The runner validates the complete inventory and weight table before selection,
and runs each selected file in a fresh Julia process with bounds checks enabled.

The `[weights]` entries in `tiers.toml` estimate test-file subprocess wall
time in seconds. The initial snapshot uses medians of three measurements,
rounded to one decimal second, from these Julia 1.13.4 GitHub Actions runs of
the 50-file inventory.
[44576422943](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44576422943/attempts/1),
[44578529805](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44578529805/attempts/1),
and [44578535495](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44578535495/attempts/1).

The 51-file snapshot used provisional weights of 30 seconds for
`logisticloss` and 140 seconds for the expanded `numerical_recovery`.
Both use a single observation rounded up to the next five seconds. The
observed wall times were 27.848 and 135.210 seconds, respectively, on Julia
1.13.4. [51-file calibration run](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44592628040/attempts/1)

The 53-file snapshot used provisional weights of 80 seconds for
`kkt_smooth`, 15 seconds for `scad`, and 145 seconds for the expanded
`numerical_recovery`. Each rounded one observation
up to the next five seconds. The measured times were 78.011, 13.374,
and 142.804 seconds on Julia 1.13.4. [53-file calibration run](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44604834492/attempts/1)

The 54-file snapshot uses provisional weights of 35 seconds for
`groupnormlinf`, 85 for `kkt_smooth`, 15 for `scad`, and 150 for
`numerical_recovery`. Each rounds one observation from the successor
checkout up to the next five seconds. The measured times were 33.898,
84.136, 14.302, and 149.517 seconds on Julia 1.13.4.
[54-file calibration run](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44615346170/attempts/1)

The 55-file snapshot adds a provisional 15-second weight for `mcp`. This
rounds the slower of two passing Julia 1.13.4 observations, 10.808 and
14.887 seconds, up to the next five seconds. The second file passed before
a newer master push cancelled its workflow.
[Upstream PR run](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44631092792/attempts/1)
and [master shard log](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44633118421/job/135944670781).

After a significant inventory or runtime change, collect fresh timings from
representative successful CI runs with a matching inventory and Julia version.
Update the affected weights and record the timing sources. If a new test has
no representative timings yet, assign an explicit conservative provisional
weight and record its justification.

Locally, run all tests with `julia scripts/test.jl`, or reproduce a CI shard
with `julia scripts/test.jl --shard=1/3` (similarly `2/3` and `3/3`).

`PB_TEST_FILE_TIME` records the parent runner's wall-clock time in seconds for
each test-file subprocess, including startup and compilation. Its `status` is
`pass` for a zero exit, `fail` for a nonzero exit, or `error` if running the
subprocess throws. `PB_TEST_SHARD_TIME` records the total wall-clock seconds
for the selected files, including runner overhead, and uses `shard=all` for
unsharded runs. Find both records in the GitHub Actions test job logs and the
`package-validation-logs-linux-current-julia-shard-*` artifacts.

The inventory/partition contract is checked by `ci_contract`. All three
shards, the frozen differential check, and the strict documentation build are
required by the existing PR verdict and merge gate. PR builds may warm
independent Julia package caches for the three validation job types.

`logisticloss` checks homogeneous Float32 and Float64 logistic regression with
dense, strided-view and CSC designs. It covers binary and fractional responses,
weights, offsets, validation and aliases, nonfinite predictors, ordinary scaling,
warmed allocations and product counts. Independent BigFloat and finite-difference
references check the value and both gradient spaces, including weighted tails.

## SCAD coverage

`scad` checks the globally minimizing scalar prox against independent BigFloat objectives
and a bounded numerical search. It covers Float32/64, steps on both sides of the curvature
transition, exact ties, parameter and input endpoints, dense and iterate oracles, stored zeros
and aliasing, per-entry step reads, pre-write refusals and supported Scaled paths.

## NormL0OffDiag coverage

`norml0offdiag` compares the proximal map with an exact `Rational{BigInt}` minimization over the
keep and drop patterns of the pairs: exhaustively for `p = 2, 3` over values below, at and above the
threshold with both signs and the four zero pairs, and on seeded samples for `p = 4`. It checks the
diagonal, symmetry, ties, the value and the fused value, agreement with the all-entry `NormL0` off
the diagonal for `p` up to 64 with mixed level and step types, the contrast on the diagonal, the
refusals with a sentinel output, strided views and reshapes, default steps, scaling and warmed
allocations on both kernel routes.
