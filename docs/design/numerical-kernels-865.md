# Shared numerical kernels (issue #865)

Status: implementation pilot. Scope is the minimal core after #850 and the
dead-code removal of `refactor/dead-code-865`. Targets are homogeneous
`Float64` and `Float32` on the CPU. Cross-precision behavior that is already
documented or tested is kept, not extended. Nothing here is public API.

The helpers that this note places in `src/utils.jl` now live in
`src/numerical_kernels.jl`; the move changed no code.

**Historical record.** The inventory, tables and decisions below describe the
code as #865 found and left it. The current implementation differs as follows:

- The correctly rounded coefficient chain (`_scaled_coefficient`,
  `_weight_coefficient3`, `_recoverable_factors`, the typed four-argument and
  untyped three-argument `_weight_coefficient`) has been removed.
- It had no caller: the KKT coefficient uses the two-factor
  `_weight_coefficient` only.
- Rows and bullets that mention it are marked *(removed)*.
- The ℓ2 kernels of `NormL2` and `GroupNormL2` (#900) are not in `src/numerical_kernels.jl`: they are
  dedicated code in `norml2.jl` and `groupnorml2.jl`, which share with the other families only the
  kernels that section 7 lists.

## 1. Inventory of recovery helpers and their live callers

Callers were taken from the current sources, not the archived pre-#850 tree.
"Ordinary path" is the code that runs for finite, well-scaled data.

| Mechanism | Helpers (src/utils.jl unless noted) | Live callers | Ordinary path | Guarantee |
|---|---|---|---|---|
| Bounded product of three factors | `_positive_product`, `_exceptional_product`, `_weight_product`, `_threshold_product`, `_scaled_product(a,b,c,k)` | NormL1 value and prox (`_positive_product`, `_weight_product`, `_threshold_product`). `_exceptional_product` is their cold path. The ℓ2 shrinkage forms its ordinary threshold inline (section 7). | `(a*b)*c`, plus one range test on `a*b`, which is loop-invariant for a scalar step | O7/R3: a product of finite factors is not lost to an intermediate over- or underflow |
| Weighted exponent-scaled product | `_scaled_product(w,v,k)` | `_pow2_weighted` (Linear, LeastSquares), `_expsum_value` (Quadratic), `_scaled_square_value` (SqrNormL2, LeastSquares), the ridge gradient fallback, the NormL1 converted-level value | cold paths only | R1/R3: w·v·2^k rounded once at the end |
| Retained-exponent product | `_scaled_product_retained` (2 and 3 factors) | `Scaled` value parts (`_apply_parts`, scaled.jl) | cold | #767: Int exponent sums that may wrap |
| Correctly rounded coefficient *(removed: had no caller)* | `_scaled_coefficient`, `_weight_coefficient3` | none (the KKT coefficient uses the two-factor `_weight_coefficient`) | cold | one correctly rounded product, also when subnormal |
| Squared residuals in power-of-two units | `_square_value`, `_exceptional_square_value`, `_scaled_square_value`, `_square_parts`, `_square_exponent`, `_square_scaled_sum` (a method for `y === nothing`), `_residual_exponent` (the code `_EXPONENT_NONFINITE` for a residual that is not finite, `_EXPONENT_ZERO`), `_square_exponent` (a method for `y === nothing` that also tells NaN, `_EXPONENT_NAN`, from Inf, `_EXPONENT_INF`), `_scaled_residual`, `_scaled_norm` | SqrNormL2 value and value parts. LeastSquaresLoss recovery (`_exceptional_square_value`, and `_square_exponent`/`_square_scaled_sum` through `_leastsquares_squares`). The cold path of the NormL2 value (`_l2_value_cold` calls `_residual_squares`) | `_sumabs2`/`_sumabs2_diff` and one comparison | R1: an overflowing or underflowing square does not hide a representable value |
| Overflowing residual a − b of finite data | inline `a/2 - b/2` in `_residual_exponent`, `_scaled_residual`, `_ridge_gradient_overflow` (sqrnorml2.jl) | SqrNormL2 value and gradient, LeastSquaresLoss recovery | none | R2: the difference is formed before scaling, so cancellation keeps its units |
| Ordinary sum-of-squares trigger | inline `s >= n * (floatmin/eps)` | `_square_value`, `_ridge_value_wide_center` (sqrnorml2.jl), `_leastsquares_value` (leastsquares.jl) | one comparison | R1: no square underflowed by more than eps of the sum |
| Overflow-safe units of a recomputed dot | inline `max(0, bound(maxabs a) + bound(maxabs b) + ⌈log₂ n⌉ + margin - emax)` | `_linear_value_recover` (margin 2), `_leastsquares_recover` (margin 3, plus a Y term) | none | R1: no partial sum of the recomputation overflows |
| Exponent-ordered sum of parts | `_pow2_weighted` (generic, and a method for two terms), `_pow2_sum` | LinearFunction, LeastSquaresLoss recoveries | none | R1 |
| Exact-product accumulator | `_ExpSum`, `_expsum_*` | QuadraticFunction recovery. The recomputed `GroupNormL2` value (`_expsum_start`, `_expsum_product`) | none | R1 (#520) |
| Index-set ℓ2 norm (section 7) | Dedicated kernels of norml2.jl: `_l2_allzero`, `_l2_scan`, `_l2_scaled_norm`, `_l2_shrink!`, `_l2_shrink_cold!`, `_l2_term`, `_l2_fused_term`, `_l2_term_cold`, `_l2_value`, `_l2_value_cold`, `_l2_parts_value`, `_l2_cold_type`; `_group_screen`, `_group_screen_cold`, `_group_value_recomputed` (groupnorml2.jl) | NormL2 and GroupNormL2 | `_indexed_sumabs2`, one range test, no extra pass | R1/R3, NR1/NR2, D1-D3, Option C |
| Sum kept at the ordinary result | `_sum_kept` | NormL1 value, GroupNormL2 value | one comparison | #373 |
| Underflow risk of products | `_underflow_threshold`, `_min_nonzero_exponent`, `_quadratic_products_normal` | QuadraticFunction only | one pass for max\|x\| | R1 (#520) |
| Ridge prox weights | `_RidgeOrdinary`, `_RidgeScaled`, `_ridge_*` (sqrnorml2.jl) | SqrNormL2 prox and prox_block | precomputed weights, vectorized loop | R3 |
| Ridge center rounding | `_ridge_fits`, `_ridge_round_big`, `_ridge_round_center32`, `_ridge_all_fit` | SqrNormL2 value, value parts, gradient, prox | free for a center stored in the model precision | T3/R3 |

### Mechanisms that are genuinely shared

* **Squared residuals.** One recomputation backs both SqrNormL2 and
  LeastSquaresLoss. On master the sequence "exponent of the largest residual,
  then typemax/typemin cases, then scaled compensated sum, then the 2k−1 half
  exponent" was written out three times: in `_scaled_square_value`,
  `_square_parts` and `_leastsquares_squares`.
* **Halved residual.** Three sites form an overflowing `a − b` of finite data as
  `a/2 − b/2` one binade up: the exponent and scaled-residual helpers, and the
  ridge gradient.
* **Bounded product.** `_exceptional_product(a,b,c,p)` repeated the body of the
  then-uncalled `_scaled_product(a,b,c,k)` with `k = 0`, line for line. The
  two-factor `_scaled_product(w,v,k)` and `_scaled_product_retained(w,v,k)` each
  re-derived the three-factor kernel with `c = 1`.
* **Sum-of-squares trigger.** The same predicate appears in three places in two
  families.
* **Units of a recomputed dot.** The same shift formula appears in
  LinearFunction and LeastSquaresLoss.

### Mechanisms left function-specific

* `_ExpSum` and the quadratic underflow test have one family, QuadraticFunction.
  The issue says not to redesign `_ExpSum`.
* `_pow2_weighted` already is the shared kernel for Linear and LeastSquares; it
  is unchanged.
* The ridge prox weights (`_RidgeOrdinary`/`_RidgeScaled`) and the
  `_ProjectionScale` machinery are specific to one family, out of scope.
* `_scaled_coefficient` gave correct rounding, a stronger guarantee than the
  bounded-product kernel. Merging the two would weaken one of them.
  *(removed: the helper had no caller and was deleted rather than merged.)*
* `norm_diff`/`_norm_diff2` uses a `hypot` fallback that could be replaced by
  the squared-residual kernel. That would change the public function's
  exceptional-path results in the last bits, so it is recorded as a follow-up
  (section 6), not done here.

## 2. Behavior table

The same rules hold for every kernel and for both consumer families.

| Case | Rule | Where it is enforced |
|---|---|---|
| Homogeneous ordinary finite data | The ordinary arithmetic result is returned bitwise, with no extra pass, allocation or call | the triggers (`_squares_normal`, the `a*b` range test, `isfinite` of the dot) |
| Finite inputs, overflowing difference, product or sum, representable result | Recomputed in power-of-two units and rounded once at the end. The ordinary Inf is never returned | `_halved_difference`, `_residual_squares`, `_scaled_product`, `_product_sum_shift` + `_pow2_sum` |
| Underflow restored by a later factor (λ, a weight, a scale) | The significands are combined first and the exponent is applied once, so the later factor finds the value | `_scaled_product`, `_square_parts`, `_scaled_square_value` |
| Genuine NaN/Inf input (data, center, level) | The ordinary IEEE result is kept. The kernels return `nothing` or the ordinary product, and never "recover" a value from a nonfinite input | `_residual_exponent` (typemax), `_residual_squares` (`nothing`), the finiteness gates of the recoveries |
| Exact zero residuals or zero factors | Zero without a recomputation. The three-factor kernel gives `+0` (a zero factor wins over an infinite one, as on master). The weighted form gives the signed `w·v` | `_residual_squares` (`(0, 0)`), `_scaled_product` |
| Aliased output (`prox!(x, f, x, γ)`, gradient over a center) | Any decision that needs a full pass reads only the parameters before the output is written. The kernels do not write | unchanged consumer code |

## 3. Minimal internal API

Each primitive below has at least two live call sites in different code paths,
and at least one of them is used by two function families.

```julia
# utils.jl: bounded-product kernel (three significands, one final ldexp)
_scaled_product(a, b, c, k)          # the kernel
_scaled_product(w, v, k)             # = kernel with c = 1, zero rule w*v
_exceptional_product(a, b, c)        # = kernel with k = 0, @noinline cold path
_scaled_product_retained(w, v, k)    # = retained kernel with c = 1

# utils.jl: squared residuals
_halved_difference(a, b, R) -> (d, h)        # a - b, or a/2 - b/2 with h = 1
_residual_squares(x, y, R) -> (s, k) | nothing   # Σ μᵢ(xᵢ - yᵢ)² = s 2^(2k)
_squares_normal(s, n, T) -> Bool             # keep the ordinary sum (R1)

# utils.jl: recomputed dot
_product_sum_shift(R, maxabs_a, maxabs_b, n, margin) -> k
```

These replace inline copies; they add no new layer. No helper takes a policy
argument, and none dispatches abstractly in a per-entry loop.

## 4. Consumers migrated

* SqrNormL2 uses `_halved_difference` (gradient), `_squares_normal` (Float64
  center at a Float32 argument), and `_residual_squares` through
  `_scaled_square_value`/`_square_parts`. Its two duplicated center-rounding
  dispatches (value and value parts) become one, `_ridge_value_center`. The two
  identical center rounders become one, `_ridge_round`.
* LeastSquaresLoss uses `_squares_normal`, `_residual_squares` (replacing the
  local `_leastsquares_squares` body), `_scaled_square_value` (directly, instead
  of through `_exceptional_square_value` with λ = 1), and `_product_sum_shift`.
* LinearFunction uses `_product_sum_shift`.

Behavior change (needs maintainer acceptance): the SqrNormL2 gradient at an
infinite level with a nonzero subnormal residual. Master's cold path always
halved the residual, which rounds such a residual to zero, and returned
`Inf·0 = NaN`. With `_halved_difference` the residual is halved only where
`x - y` overflows, so the gradient is the IEEE `λ(x - y) = ±Inf`. This is the
only difference on the characterization grid.

## 5. Verification

* `test/numerical_kernels.jl`: kernel-level tests in Float32 and Float64 against
  BigFloat references, covering every row of the behavior table, plus
  consumer-level regression tests for SqrNormL2 and LeastSquaresLoss.
* `docs/audit/numerical-kernels-865/characterize.jl` loads the baseline and the
  candidate source trees in one process. It runs the same grid of 148,528
  ordinary and exceptional calls on each and compares every result bit for bit.
  The grid covers SqrNormL2 value, gradient, value_and_gradient, prox!,
  prox_block and Scaled value parts, on dense, sparse, block and symmetric
  arguments, centered and uncentered, with wide and BigFloat centers. It also
  covers LeastSquaresLoss value, gradient and value_and_gradient on dense,
  sparse and multi-response arguments, plus LinearFunction, NormL1 and
  `norm_diff`. Mutating each kernel in a scratch copy is detected by this
  harness, which shows the grid reaches the cold paths. Results are in
  `docs/audit/numerical-kernels-865/README.md`.
* Paired benchmarks: `docs/audit/numerical-kernels-865/bench.jl`.

## 6. Limitations and follow-ups (not done here)

* `norm_diff` and `_norm2_scaled` (SymmetricSparseIterate norm) use `hypot`
  loops on the exceptional path. Both could use `_residual_squares` instead,
  which is O(n) without `hypot` and more accurate. Doing so would change
  exceptional-path results in the last bits of a public function, so it needs
  maintainer approval.
* `_scaled_product_retained` (3 factors) and `_scaled_product` (3 factors)
  differ only in how BigFloat exponents are summed (Int wraparound vs Int128).
  They are kept separate because #767 depends on the wraparound.
* `_ridge_fits(Float32, ::Float64)` tests the unrounded center against the
  Float32 range, while `_level_fits` tests the rounded value. They disagree
  within half an ulp of `floatmax(Float32)` and `floatmin(Float32)`, and there
  the conservative test computes with the unrounded center. This is existing
  behavior, kept bitwise. It is recorded as a possible contract clarification.
* The QuadraticFunction recovery (`_ExpSum`) is a third, distinct mechanism (an
  exact-product sum, not scaled squares). Migrating it was not justified by
  shared code.

## 7. The ℓ2 kernels of NormL2 and GroupNormL2 (#900): dedicated paths

The ℓ2 shrinkage, values and group screen are dedicated code in `src/functions/norml2.jl` (for one index
set, the whole array for `NormL2`, one group for `GroupNormL2`) and `src/functions/groupnorml2.jl` (the group
value recomputation and the group screen with its cold path). Their outputs are bit-identical to `76964085`
in every type and rounding mode. A consolidation of these kernels into `numerical_kernels.jl`, shared with
the other families, was measured against this design; this file records the dedicated one.

**Shared with other families** (the only dependencies of the ℓ2 kernels on `numerical_kernels.jl`)

* `_indexed_sumabs2(x, G, T)`: the vectorized sum of squares of the entries of `x[G]`, each rounded to `T`;
  the first pass of the ordinary shrinkage, value and screen, and the sum of squares of the stored values of a
  `SparseIterate` (`_sumabs2` of `SqrNormL2`).
* `_sum_kept`: the test that the ordinary sum of the terms of a value is kept (`NormL1` values, the `GroupNormL2`
  value).
* `_working_type`, `_squares_normal`, `_level_fits`, `_scaled_product`, `_power_factors`, `_scale_value`,
  `_compensated_add`, `_frexp`, `_kkt_coefficient`, `_ExpSum` with `_expsum_start` and `_expsum_product`, and,
  for the `NormL2` value, `_sumabs2`, `_all_zero`, `_residual_squares` and `_square_term_count`.
* The kernels that the other families use and that stay in `numerical_kernels.jl`: the `y = nothing` methods of
  `_square_exponent` and `_square_scaled_sum` with the classification codes `_EXPONENT_NAN`, `_EXPONENT_INF`,
  `_EXPONENT_ZERO` (`SqrNormL2`, `norm`), `_scaled_norm`, `_threshold_product` (`NormL1`), and the method of
  `_pow2_weighted(R, w, t1, t2)` for two terms, described next.

**The two-term `_pow2_weighted`** (LinearFunction, LeastSquaresLoss) has the operations of the generic method
(the larger term first, aligned addition, renormalization, the final product). For the IEEE types only
(`R <: Base.IEEEFloat`: Julia rounds them to nearest only, and their `frexp` is exact, so the larger term is
first), a term more than `2 precision(R)` binades below the sum is not shifted: it is below half an ulp of the
sum, which round-to-nearest then leaves unchanged, and the shift can round to a subnormal number or to zero,
the slow case that is avoided. Every other type, BigFloat included, does the generic method's sum in every
rounding mode: under RoundUp, RoundDown and RoundToZero a term below half an ulp can still move the result,
depending on its sign and the mode. The generic method remains the definition. The two agree where each
nonzero term has an exponent sum `exponent(m) + e` above `typemin(Int)`, the value from which the selection of
the generic method starts, and the tests compare them there bit for bit, for BigFloat at two precisions under
all four rounding modes. At that sum the generic method selects no term and the method for two terms takes the
single nonzero term, so the two differ there: for BigFloat `(floatmin(BigFloat), -2^62)` with a zero term, and
for the IEEE types `(T(0.75), typemin(Int) + 1)` with a zero term and the weight `T(Inf)` (`Inf` from the
method for two terms, `NaN` from the generic one).

**Dedicated to the ℓ2 norms**

| Function | Role |
|---|---|
| `_l2_allzero(x, G, T)` | whether every entry rounded to `T` is a zero of either sign (the destination type decides, as in the sums of squares in `T`) |
| `_l2_scan`, `_l2_scaled_norm` | the cold pass: NaN, Inf and the largest exponent of a group, then the compensated sum of its entries scaled by `2^-e` |
| `_l2_shrink!`, `_l2_shrink_cold!` | the ordinary shrinkage with its single normal-range test and the cold path decided by the parameters and the values |
| `_l2_term`, `_l2_fused_term`, `_l2_term_cold` | the term `λw‖x[G]‖₂` of a value, the fused variant and the cold term |
| `_l2_value`, `_l2_value_cold` | the `NormL2` value on arrays and on the three iterates |
| `_l2_parts_value`, `_l2_cold_type`, `_l2_ordinary` | `m 2^E` rounded once, the type of the cold recomputations (at least Float64), the ordinary range test |
| `_group_value_checked`, `_group_value_recomputed` | the recomputed `GroupNormL2` value, with `_ExpSum` |
| `_group_screen`, `_group_screen_cold` | `‖g[G]‖₂ - λ0 w` and its cold path |

**Recovery algorithms that exist twice.** The dedicated paths copy these algorithms of the shared kernels,
all on cold paths only:

* the scaled norm of a group: `_l2_scan` + `_l2_scaled_norm` (one pass for the class and the largest exponent, one
  for the compensated sum of entries scaled by `2^-e`) against `_square_exponent` + `_square_scaled_sum`
  (`y === nothing` methods), which serve `SqrNormL2` and `norm`;
* the product of the level, the weight and the norm rounded once to `T`: `_l2_parts_value` and the frexp products in
  `_l2_term_cold` and `_group_screen_cold` against `_scaled_product` and `_expsum_value`;
* the difference of the scaled norm and the coefficient in the group screen, aligned by hand in the units of the
  larger exponent, against `_pow2_sum`;
* the threshold `γλw` of the ordinary shrinkage, formed inline against `_threshold_product` (three lines);
* the `NormL2` value cold path, which calls `_residual_squares` as `SqrNormL2` does (one algorithm).

**Exceptions to bit identity.** None for the ℓ2 norms. The two BigFloat differences that the maintainer ruled
on for the shared kernel (D-X: exponent sums beyond `Int64`; D-MPFR: a directed rounding mode at MPFR's
exponent limits) do not arise here, because the dedicated paths are those of `76964085`; with them the
BigFloat exponent sums of the group value and the screen wrap at `Int64` as before
(`GroupNormL2(λ0 = w = 2^(2^62-2))` at `x = [2^(2^62-2)]` has value `0`, and `test/groupnorml2.jl` records it as a known defect with
`@test_broken`).
