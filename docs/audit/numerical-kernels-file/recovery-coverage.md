# Numerical-recovery coverage audit

Scope: the live overflow, underflow, cancellation and exceptional-recovery
machinery in `src/` at `origin/master` a5010d20 (before the planned move of the
numerical helpers from `src/utils.jl` into `src/numerical_kernels.jl`). This
commit is test-only: nothing under `src/` changed, and `test/numerical_recovery.jl`
passes on the unchanged tree, so it is also the behavioral baseline for that move.

Locations in the tables are those of a5010d20. After the move, every helper listed
under `utils.jl` lives in `src/numerical_kernels.jl`, except the scaled-value
protocol hooks (`_value_parts`, `_zero_value_exponent`, `_EXACT_ZERO`, the alias
hooks), the stored-value scans (`_stored_maxabs`, `_all_finite`), `_tril_column_isqrt`
and `_model_precision`, which stay in `src/utils.jl`.

The tests are in `test/numerical_recovery.jl` (declared in `test/tiers.toml`,
quick tier, and described in `test/TESTING.md`). Every case runs in Float32 and
Float64 against a BigFloat evaluation of the mathematical formula on the exact
data (never by calling package code in BigFloat), with ulp tolerances stated in
the file header and justified per algorithm. The suite is not a claim of
exhaustive coverage: see "Remaining gaps".

## Method

Coverage was measured per test file, in a fresh process with bounds checks, with
Julia's line-coverage instrumentation restricted to `src/`:

```sh
export PATH=/tmp/julia-1.13.3/bin:$PATH
cd ProximalBase-recovery-tests
for t in active_handle block_copy block_replacement cache_and_blocks core_functions iterate_invariants \
         iterate_queries iterate_writes matrixview minimal_math numerical_kernels real_iterates ridge_precision \
         sparse_iterate_basics_grids sparse_iterate_basics sparse_iterate symmetric_iterate_indexing \
         symmetric_iterate_interface symmetric_products; do
  find src -name '*.cov' -delete
  julia --project=. --startup-file=no --check-bounds=yes --code-coverage=@$PWD/src \
    -e 'include(ARGS[1]); include(ARGS[2]); include(ARGS[3])' test/tiers.jl test/testutils.jl test/$t.jl
  # collect src/**/*.cov for $t outside the tree, then delete them
done
# after: the same for test/numerical_recovery.jl, merged with the runs above
```

(`ci_contract` does not execute `src/` and was omitted.) The per-line counts of
the `.cov` files were summed over runs, and the lines of each recovery function
were read (`src/utils.jl`, `src/functions/*.jl`, `src/array_steps.jl`,
`src/sparse_iterate.jl`, `src/symmetric_parameters.jl`). All `.cov` files were deleted
afterwards (`*.jl.cov` is also git-ignored); none is committed.

Caveats of line coverage that the tests deal with explicitly:

- A line holding several branches (`a && return x`, `c ? p : q`, one-line methods)
  counts as executed when any branch ran. For those the tests assert a
  distinguishing observable (for example an exact `+0.0`, `Inf`, `NaN`, or a
  value that only the recovery can produce), not just that the function was
  called. The table below says "partial" where the evidence is at line level only.
- A method that Julia never compiled has no counter at all. Such methods are
  listed as "missing" (never executed) below; several are reachable only through the
  public oracles with unusual arguments (owners, array steps, nested factors).
- Trivial one-line methods that are always inlined (traits, `_ridge_fits` for equal
  types) show no counter of their own and are not counted as gaps.

## Inventory

Status: **covered** means every branch in the row is now executed with a test
that asserts a result distinguishing that branch; **remaining gap** says what is
not and why. "Existing coverage" is tested / partial / missing, with the tests
that execute the lines (from the coverage runs above, lines with a positive count).

| Recovery mechanism | Location (file:function) | Existing coverage (tested / partial / missing, with the evidence: which existing test hits which branch, per the coverage run) | New tests (testset name in numerical_recovery.jl) | Status (covered / remaining gap + why) |
|:--|:--|:--|:--|:--|
| Bounded three-factor product, exceptional branch (`ab` outside the normal range), zero factor gives +0 | `utils.jl`: `_positive_product`, `_exceptional_product` | tested: ordinary branch by core_functions, minimal_math, matrixview; the exceptional call (line 29/32, `_scaled_product(a,b,c,0)`) by numerical_kernels "bounded-product kernel" | "NormL1 values: levels and weights outside the range, infinite and zero parameters", "NormL1 prox thresholds and the weight kernels" | covered |
| `_scaled_product` (2 and 3 factors), exponents applied by one final `ldexp`; zero and non-finite factors | `utils.jl`: `_scaled_product` | tested: numerical_kernels (random overflow/underflow factors, zero beats Inf, NaN/Inf ordinary), core_functions, minimal_math, ridge_precision | used throughout (value recoveries); "power-of-two sums (_pow2_weighted) and the recomputed-dot shift" | covered |
| Retained-exponent product (Int sum, modulo 2^64, #767) | `utils.jl`: `_scaled_product_retained` | tested: numerical_kernels "retained exponent across Int wraparound (#767)" and the random agreement loop; minimal_math | "Scaled values: overflow, underflow and exact zeros through the factor", "Scaled single factors, zero and infinite scales, aliases and nested wrappers" (through `_apply_parts`) | covered |
| Weighted product for thresholds `γλw` (infinite weight gives Inf even at zero step), typed rounding once, `Bool` weight | `utils.jl`: `_weight_product` (3-arg and typed) | partial: core_functions and minimal_math execute the lines (Inf weights, zero level); the mixed-precision "formed in the promotion, rounded once" form and the `Bool` weight had no asserting test | "NormL1 prox thresholds and the weight kernels" | covered |
| Coefficient `s*w` with `0*Inf` resolved by the weight; KKT coefficient | `utils.jl`: `_weight_coefficient(s,w)`, `_weight_coefficient(U,λ,w)`, `_kkt_coefficient` | partial: line 47 (`ifelse(isnan(p) & !isnan(s), oftype(p,w), p)`, the zero-level/infinite-weight and infinite-level/zero-weight case) never executed (cache_and_blocks, core_functions, minimal_math run only the ordinary lines); untyped `_kkt_coefficient(::Type, λ, w)` not executed | "NormL1 prox thresholds and the weight kernels" (also `kkt_violation` at infinite, zero and overflowing coefficients) | covered |
| Correctly rounded 128-bit three-factor coefficient, subnormal and signed-zero results | `utils.jl`: `_weight_coefficient3`, `_scaled_coefficient`, `_recoverable_factors`, `_weight_coefficient(U,γ,λ,w)` | missing: never compiled by any existing test, and **no caller exists in `src/`** (only the 3-argument KKT form is used; the 4-argument chain is dead code referenced only by `docs/design/numerical-kernels-865.md`) | "NormL1 prox thresholds and the weight kernels": 3000 random triples against the correctly rounded BigFloat product (`===`), subnormal, overflow, signed underflow, zero/NaN/Inf, `Float32` from `Float64` factors | covered as a baseline of the helper itself; remaining gap: it is unreachable from a public oracle (dead code, see "Remaining gaps") |
| Zero-times-infinite value guard (`_weighted_value`) | `utils.jl`: `_weighted_value` | missing | "NormL1 prox thresholds and the weight kernels" | covered |
| Compensated sum of scaled values and dots (`_scaled_value_sum`, `_scaled_dot`) | `utils.jl`: `_scaled_value_sum`, `_scaled_dot`, `_compensated_add` | partial: `_scaled_dot` (dense) by numerical_kernels and core_functions; `_scaled_value_sum` missing (only called by `_converted_l1_value`) | "NormL1 values: levels and weights outside the range, infinite and zero parameters", "LinearFunction: overflowing dots, owners and iterates" | covered |
| Scaled dot over stored entries of a sparse iterate, symmetric owner (multiplicity), block iterate | `sparse_iterate.jl`: `_scaled_dot(::SparseIterate)`, `_scaled_dot(::SymmetricSparseIterate)`; `block_iterate.jl`: `_scaled_dot(::BlockIterate)` | missing (never compiled). The owner and block methods have no live caller in `src/` (`_linear_scaled_dot` has its own owner method) | "LinearFunction: overflowing dots, owners and iterates" (direct `_scaled_dot` calls on the three iterates, and the live `_linear_scaled_dot` owner/sparse/view routes) | covered directly; owner/block `_scaled_dot` are unreachable from a public oracle |
| Ordered power-of-two sum with cancellation before alignment, exact zero sum, non-finite terms, BigFloat Int128 frames | `utils.jl`: `_pow2_weighted`, `_pow2_sum` | partial: all 25 lines executed by numerical_kernels (through LeastSquares/Linear recoveries); no test of exact cancellation before a small term, non-finite terms, zero-term skipping or the BigFloat frame path | "power-of-two sums (_pow2_weighted) and the recomputed-dot shift" | covered |
| Units of a recomputed dot (`_product_sum_shift`, `_bound_exponent`, `_log2_ceil`, `_max_exponent`) | `utils.jl` | tested: numerical_kernels "recomputed-dot units" (random n, margin) | "power-of-two sums (_pow2_weighted) and the recomputed-dot shift" (zero/Inf/NaN bounds, exact `_log2_ceil`) | covered |
| Value-parts protocol, exact-zero exponent, `_EXACT_ZERO`, `_upper_exponent`, `_underflow_exponent` | `utils.jl`: `_parts`, `_parts_product`, `_value_parts`, `_zero_value_exponent`, `_upper_exponent`, `_underflow_exponent`; `scaled.jl`: `_parts_wanted`, `_zero_value_exponent(::Scaled)` | partial: `_parts`, `_parts_product` by numerical_kernels; `_upper_exponent` and `_underflow_exponent` compiled but never executed; `_parts_wanted` 7/12 lines (the exact-zero and infinite-value exits never run); `_zero_value_exponent(::Scaled)` missing | "Scaled values: overflow, underflow and exact zeros through the factor", "Scaled single factors, zero and infinite scales, aliases and nested wrappers" | covered (see S3 below for the documented normal-range limit of the zero exit) |
| `_ExpSum` accumulator: frame change, compensated add, first-term frame, scaled product/term/dot, final conversion | `utils.jl`: `_ExpSum`, `_expsum_start`, `_expsum_add`, `_expsum_product`, `_expsum_term`, `_expsum_dot`, `_expsum_value` | missing: compiled by the precompile workload only, 0 executions in every existing test | "exponent-sum accumulator (_ExpSum)" (frame rescale, cancellation compensation, Inf/0/subnormal conversion to Float32/Float64, random sums of terms spanning 1800 binades) | covered |
| `_ExpSum` quadratic forms over dense, `SparseMatrixCSC` and `Diagonal` matrices | `utils.jl`: `_expsum_quadratic` (3 methods) | missing | "exponent-sum accumulator (_ExpSum)", "quadratic value: overflow, cancellation and underflow recovery" | covered |
| Quadratic value recovery: signed forms whose dot overflows and is cancelled by `b'x`, Ax overflowing to Inf-Inf, subnormal products with rounded-once result, `c`, `b`, sparse/diagonal `A`, sparse-iterate `x` | `quadratic.jl`: `_quadratic_checked`, `_quadratic_value_recover`, `value_and_gradient!` | missing: `_quadratic_value_recover` 0/9; the checked branch (`!isfinite(v) || risk`) is run only on ordinary values by core_functions and symmetric_products | "quadratic value: overflow, cancellation and underflow recovery" (exact cancellation `v = c`, null-space NaN, true overflow stays ±Inf, ordinary dot loses bits in 1+ cases while the recovery is correctly rounded), "quadratic recovery trigger and guards" | covered |
| Underflow trigger and the products-normal test (null-space argument keeps its exact zero) | `quadratic.jl`: `_quadratic_underflow_risk`, `_quadratic_products_normal`; `utils.jl`: `_underflow_threshold`, `_min_nonzero_exponent` (bit-pattern and generic), `_stored_minexp` | partial: threshold and risk evaluated on ordinary calls (core_functions, symmetric_products); `_quadratic_products_normal` 0/8, `_min_nonzero_exponent` 0 | "quadratic recovery trigger and guards" (risk true/false at zero x, threshold overflow to Inf, min exponent with 0/NaN/Inf/subnormal/Diagonal/sparse/BigFloat/transposed, guards for NaN/Inf/non-finite data and signed zeros) | covered |
| `QuadraticFunction` eigendecomposition prox (first use caches Q and Λ), aliasing, fused value | `quadratic.jl`: `_eigen!`, `prox!`, `value_and_prox!` | partial: all lines run (cache_and_blocks, core_functions), accuracy not compared with an independent solve | "quadratic oracles: fused and standalone, eigendecomposition prox" (BigFloat solve of (I+γA)z = x-γb at γ ∈ {1e-3, 1, 50}, cache reuse, `reset_cache!`, alias, `value_and_prox!` value at the stored point, gradient bound n eps `|A||x|`) | covered |
| Squared-residual kernel: halved difference, exponent of the largest residual, compensated scaled sum, `(zero,0)` for exact zero, `nothing` for non-finite | `utils.jl`: `_halved_difference`, `_residual_exponent`, `_scaled_residual`, `_residual_squares`, `_squares_normal` | tested: numerical_kernels ("halved difference", "squared-residual kernel", trigger test), also core_functions, minimal_math, ridge_precision | not repeated | covered by existing tests |
| Squared value `λ/2 Σ(x-y)²` exceptional branches: non-finite or zero λ, λ below floatmin, exact zero, integer entries, block iterate | `utils.jl`: `_square_value`, `_exceptional_square_value`, `_scaled_square_value`, `_square_parts` | tested at line level (100%): numerical_kernels, core_functions, minimal_math, ridge_precision; the one-line exits (λ not finite, `iszero(λ) && isfinite(s)`, the `DenseArray` exact-zero exit) and integer/Block arguments had no asserting test | "squared-residual consumers: the exceptional branches of the value" | covered |
| `norm_diff` hypot fallback (sum of squares overflows/underflows, equal arrays, NaN, Inf), integer widening, `p = Inf`, owner overloads, argument errors | `utils.jl`: `norm_diff`, `_norm_diff2`, `_norm_difference`; `sparse_iterate.jl`: `norm_diff` owner methods | missing: `norm_diff` and `_norm_diff2` never compiled by any existing test | "norms: the symmetric owner multiplicity and the norm_diff scaled fallback" | covered |
| `_norm2_scaled` for an owner: off-diagonal multiplicity 2, NaN/Inf coordinates, p = 1, 2, 3, Inf; block iterate | `sqrnorml2.jl`: `_norm2_scaled`; `sparse_iterate.jl`: `norm(::SymmetricSparseIterate)` | missing: owner loop never compiled (2/11 lines, array method only) | "norms: the symmetric owner multiplicity and the norm_diff scaled fallback" | covered |
| `_ProjectionScale`, `_projection_accumulate`, `_scaled_chain`, retained exponent, `_apply_scale`, `_scale_inverse` | `scaled.jl` | missing: 0 of the accumulate and chain lines; `_apply_scale(::_ProjectionScale)` 0/5 | "Scaled values: ...", "Scaled oracles on factor chains: ...", "Scaled single factors, zero and infinite scales, aliases and nested wrappers" | covered (the `Real,Real` product accumulation and the non-IEEE `_LevelProbe` window are executed through rational/BigFloat factors) |
| Scaled value recovery: fold into a normal level, partial fold with shifted window (and recentring near the range edge), fold into a subnormal level, Inf*0 at an infinite factor, parts | `scaled.jl`: `_folded_level`, `_fold_level`, `_value_fold_scale`, `_level_window`, `_shifted_fold_value`, `_scaled_value_partial`, `_scaled_value_recover`, `_recover_from_parts`, `_scaled_value_by`, `_scaled_nan_value`, `_exactly_zero_at(::NormL1)` | partial: `_scaled_value_recover` 8/14 lines and `_parts_wanted` 7/12, all through `Scaled(SqrNormL2, α)` (numerical_kernels, core_functions, minimal_math); `_folded_level`, `_level_window`, `_shifted_fold_value`, `_scaled_value_partial` (0/18), `_scaled_nan_value`, `_exactly_zero_at(::NormL1)` never executed | "Scaled values: overflow, underflow and exact zeros through the factor" (NormL1 fold, weights forcing the partial fold, window recentring, factor above one with zero/subnormal leaf, exact zero stays +0, infinite factor, NaN/Inf data, wrapper in wrapper), "Scaled single factors, ..." | covered; see S3 and S4 for documented limits |
| Nested chains: value, gradient, fused value-and-gradient, scalar-step prox, `value_and_prox!`, `prox_block`, KKT, lipschitz | `scaled.jl`: `_value_chain`, `_gradient_chain!`, `_value_and_gradient_chain!`, `_scale!`, `_nested_step`, `_prox_chain!`, `_value_and_prox_chain!`, `_prox_block_nested`, `_nested_kkt`, `lipschitz`, `_value_parts_scaled(::Scaled)` | missing (0 executions; `_step_exceptional` and `_scaled_chain` only on ordinary calls, cache_and_blocks, minimal_math) | "Scaled oracles on factor chains: gradient, prox, fused and block forms" (SqrNormL2 and NormL1 leaves, symmetric-owner gradient, single-factor step out of range, infinite level, KKT with and without folded level) | covered except the recorded defect D1 and limits S1, S2 |
| Scaled array steps: factor in or outside the band, `:folded`, `:shifted`, `:rounded` routes, stored infinite level, infinite factor, `_scaled_value_overflow` | `scaled_array.jl`: `_array_route`, `_wide_scale`, `_array_fold`, `_shifted_fold`, `_prox_chain!`, `_value_and_prox_chain!`, `_scaled_value_overflow`; `array_steps.jl` | missing: no existing test passes an array step (only the `Scaled` method tables compile) | "Scaled array steps: folded, shifted and rounded routes", "Scaled single factors, ..." | covered for NormL1, a custom separable function and `ZeroFunction`; `SqrNormL2` has no array-step method (MethodError), so no recovery there |
| Zero scale simplification and its refusals | `scaled.jl`: `_zero_scaled`, `_finite_zero`, `_finite_parameter` | missing (0 executions) | "Scaled single factors, zero and infinite scales, aliases and nested wrappers" | covered |
| Ridge gradient `λ(x-y)`: halved difference when `x-y` overflows, repair pass, wide level (outside the model range), wide centers | `sqrnorml2.jl`: `_ridge_gradient!`, `_ridge_gradient_dense!`, `_ridge_gradient_repair!`, `_ridge_gradient_overflow`, `_ridge_gradient_wide` | tested: dense forms by ridge_precision, numerical_kernels, core_functions; owner output forms (`_ridge_gradient!(::SymmetricSparseIterate)`, `_ridge_gradient_coordinates!`, `_ridge_gradient_wide_coordinates!`) and the dense-from-owner method missing | "SqrNormL2 oracles: ridge weights, owners, wide levels and centers" | covered |
| Ridge prox weights `1/(1+t)`, `t/(1+t)`: ordinary and scaled, centers, coordinate form, aliasing, owner output, wide centers (Float64 in Float32 model, BigFloat in Float64) | `sqrnorml2.jl`: `_ridge_needs_scaling`, `_ridge_ordinary`, `_ridge_scaled`, `_ridge_apply_*`, `_ridge_prox!`, `prox_block`, `_ridge_all_fit` | partial: dense, coordinate and wide-IEEE-center forms by ridge_precision (frozen BigFloat references), numerical_kernels, minimal_math; owner outputs (`_ridge_prox!(::SymmetricSparseIterate, …)`), generic non-IEEE center (`_ridge_apply_wide`), generic `_ridge_all_fit` missing | "SqrNormL2 oracles: ridge weights, owners, wide levels and centers" (five regimes of `γλ`, with and without centers, owner and dense outputs, in-place) | covered |
| LeastSquares value recovery (zero residual, finite squares, overflowing `Xβ` with scaled recomputation) | `leastsquares.jl`: `_leastsquares_recover`, `_leastsquares_own`, `_leastsquares_big`, `_leastsquares_squares` | tested: numerical_kernels (five cases and NaN/Inf), minimal_math | "LeastSquaresLoss: SVD prox, multi-response data and iterate arguments" (multi-response, sparse/block β, fused vs standalone) | covered |
| LeastSquares SVD prox (square V with γ <= 1 and > 1, rectangular V, empty design) | `leastsquares.jl`: `prox!`, `_svd_cache!` | partial: `prox!` 15/23 and `_svd_cache!` 11/18 (cache_and_blocks, core_functions); the γ > 1 branch, rectangular V and the empty-design cache never ran | "LeastSquaresLoss: SVD prox, multi-response data and iterate arguments" (BigFloat solve of `(X'X + I/γ)β = X'Y + v/γ` for n > p, n = p, p > n, γ from subnormal to 1e30, multi-response) | covered |
| Linear dot recovery (overflowing dot, `d` in its own units) | `linear.jl`: `_linear_value`, `_linear_value_recover`, `_linear_scaled_dot` | partial: dense and `SparseIterate` by numerical_kernels; owner method and the copy-and-scale route missing; `_owner_dot` (now `_symmetric_matrix_dot`, #884) and `_checked_value_dot` (symmetric_parameters.jl) never executed | "LinearFunction: overflowing dots, owners and iterates" | covered |
| `LinearFunction` prox shift and owner output | `linear.jl`: `_prox_unchecked!` | partial: 6/9 (shape-mismatch loop and owner methods missing) | "LinearFunction: overflowing dots, owners and iterates" | covered |
| NormL1 values and thresholds: weights/levels outside the range, infinite/zero weight and level, level outside the argument's precision | `norml1.jl`: `value_entry`, `_l1_prox_threshold`, `_value_dense`, `_l1_value_sum`, `_converted_l1_value` | partial: core_functions, minimal_math, matrixview run the ordinary and exceptional product branches; `_converted_l1_value` and `_scaled_value_sum` never executed | "NormL1 values: ...", "NormL1 prox thresholds and the weight kernels" (random thresholds, sparse iterate, in place, fused) | covered; defect D2 recorded |
| Sparse/symmetric three-factor products: exceptional `a*x` branch (`_triple`), overflowing partial sums with later cancellation (`_rescue_entry!`), genuine NaN/Inf, empty owner | `sparse_iterate.jl`: `_add_coefficient!`, `_triple`, `_rescue_entry!`, `_rescue_nonfinite!`, `_logical_coordinates` | partial: ordinary branch of `_add_coefficient!` (19/27 lines) and the finite scan of `_rescue_nonfinite!` by sparse_iterate, sparse_iterate_basics(_grids); `_triple` 0/8 and `_rescue_entry!` 0/26 never executed | "sparse and symmetric matrix products: overflowing terms, cancellation and non-finite factors" | covered |
| Symmetric part of an asymmetric argument: halves-first mean, tiny-α product term by term | `sparse_iterate.jl`: `_SymmetricPart`, `_symmetric_mean`, `_scaled_symmetric_mul!`, `_product3` | missing (never compiled) | "symmetric part of an asymmetric argument: the mean and the tiny-α product" | covered |
| Fused vs standalone oracles | `value_and_gradient!`, `value_and_prox!`, `prox_block` vs `prox!` of the functions above | partial: ridge_precision/numerical_kernels for SqrNormL2 and LeastSquares only | every function-level testset above (Quadratic, Scaled chains and array steps, NormL1, SqrNormL2 incl. owners, LeastSquares, Linear) | covered |

### Not recovery mechanisms but cold paths found in `src/`

| Cold path | Location | Existing coverage | Status |
|:--|:--|:--|:--|
| Integer `isqrt` column of `tril_cartesian` for `p >= 2^24` | `utils.jl`: `_tril_column_isqrt` | compiled, never executed (0/1) | remaining gap: needs a `p x p` owner with `p >= 16 777 216` (not constructible in a test); the float path is exact below that |
| `opnorm2` Lanczos and the fixed-step upper bounds | `utils.jl`: `_largest_eigenvalue`, `_opnorm2`, `_opnorm_upper`, `_opnorm2_upper` | not compiled by any existing test; `lipschitz` of the function types also not | remaining gap: numerical estimate, not an overflow/underflow recovery; left to a separate audit |
| `_scale!` for tuple blocks, device-array routes (`_is_device_array`, copy-and-scale dot on device arrays) | `scaled.jl`, `linear.jl` | never executed | remaining gap: tuple/device arguments need extension types not available in the suite |
| Dimension-mismatch throws (`_size_mismatch`, `_parameter_length_mismatch`, `_difference_size_mismatch`) | `utils.jl` | partially executed elsewhere; new tests exercise `_difference_size_mismatch` via `norm_diff` | covered for the numerical ones |

## Executed recovery lines, before and after

Counts are executable lines (those with a counter in the union of the before and
after runs) of the functions of each mechanism, executed at least once.
"Before" is the 19 existing numerical/storage test files; "after" adds
`numerical_recovery`. Line counts overstate the progress of single-line
branches (see Method), so the table is a floor for "never executed" rather than
a measure of branch coverage.

| Mechanism (functions in the rows above) | executable lines | executed before | executed after | existing tests that executed them |
|:--|--:|--:|--:|:--|
| bounded products and weights | 76 | 45 (59%) | 76 (100%) | cache_and_blocks, core_functions, matrixview, minimal_math, numerical_kernels, ridge_precision |
| scaled sums, dots and units | 60 | 37 (61%) | 60 (100%) | minimal_math, numerical_kernels, ridge_precision |
| `_ExpSum` and the quadratic recovery | 149 | 51 (34%) | 148 (99%) | cache_and_blocks, core_functions, symmetric_products |
| value-parts protocol | 8 | 4 (50%) | 8 (100%) | numerical_kernels |
| squared-residual kernel | 75 | 75 (100%) | 75 (100%) | core_functions, minimal_math, numerical_kernels, ridge_precision |
| `norm_diff` hypot fallback | 34 | 0 (0%) | 34 (100%) | none |
| Scaled recovery (values, chains, folds, parts) | 312 | 72 (23%) | 308 (98%) | cache_and_blocks, core_functions, minimal_math, numerical_kernels |
| Scaled array-step routes | 74 | 3 (4%) | 73 (98%) | core_functions |
| SqrNormL2 (value, gradient, ridge prox) | 244 | 195 (79%) | 242 (99%) | core_functions, minimal_math, numerical_kernels, ridge_precision |
| LeastSquaresLoss (value recovery, SVD prox) | 101 | 86 (85%) | 101 (100%) | cache_and_blocks, core_functions, minimal_math, numerical_kernels |
| LinearFunction (dot recovery) | 87 | 45 (51%) | 87 (100%) | core_functions, numerical_kernels |
| NormL1 (weights, thresholds, converted level) | 55 | 47 (85%) | 55 (100%) | cache_and_blocks, core_functions, matrixview, minimal_math |
| sparse/symmetric products (`_triple`, `_rescue_entry!`) | 80 | 28 (35%) | 80 (100%) | sparse_iterate, sparse_iterate_basics, sparse_iterate_basics_grids |
| symmetric part (`_symmetric_mean`, tiny alpha) | 23 | 0 (0%) | 23 (100%) | none |
| owner norm (`_norm2_scaled`) | 25 | 3 (12%) | 25 (100%) | core_functions, minimal_math, numerical_kernels, ridge_precision, sparse_iterate_basics |
| **total** | **1403** | **691 (49%)** | **1395 (99%)** | |

The eight lines still unexecuted after the new file are function header lines of
multi-method definitions whose bodies run (`_min_nonzero_exponent`, `_shifted_fold_value`,
`_exactly_zero_at`, `_array_fold`, the two inlined `_ridge_scaled` wrapper lines),
the `Base.IndexStyle` trait of the lazy `_ScaledArray`, and `_ExpScaledArray(::AbstractArray, ::Real)`,
which is never called because the product of a chain is always a `_ProjectionScale`.

## Defects found

Each is pinned with `@test_broken` against the independent reference in the
testset "recorded defects ..." (5 broken assertions per type; they flip to
an "Unexpected Pass" error when fixed). Both are **suspected** defects, not
certain: the numerical page promises "practical safeguards, not exhaustive
range recovery", but in each case a sibling route of the same function recovers
the value, which is why they are not merely listed as limitations.

**D1 (suspected): scalar-step prox of a nested `Scaled` chain over a weighted
`NormL1` loses the threshold when the level cannot hold the factors and the
weights compensate.** `_nested_step` folds the factors and the step significand
into the level `λ0` only (`_fold_level_step`); if `α1 α2 γ λ0` is not a normal
number (a zero, a subnormal level that has lost bits, or an overflow) the function is
left with the step rounded (`step = α1 α2 γ`, 0 or Inf) and the threshold is
wrong, even though `α1 α2 γ λ0 w` is representable. The *value* of the same chain
recovers it through the partial fold (`_scaled_value_partial`), and the
*array-step* route recovers it through `_shifted_fold`. `prox!`, `value_and_prox!`
and `prox_block` take the scalar-step route.

```julia
using ProximalBase
T = Float32; E = 127; p = E ÷ 2 + 20            # same shape in Float64 with E = 1023
f  = NormL1(T(1.3) * 2f0^20, T[2f0^(E - 10)])
s  = Scaled(Scaled(f, 2f0^-p), 2f0^-p)           # α1 α2 = 2^-(E+39): not a normal number
γ  = 2f0^-30
x  = T[3 * 2f0^-50]                              # threshold t = α1 α2 γ λ0 w = 2.2551405e-18
prox!(zeros(T, 1), s, x, γ)                      # Float32: [2.6645353e-15] (x unchanged)   expected [2.66228e-15]
prox!(zeros(T, 1), s, x, T[γ])                   # array step: [2.66228e-15]  (correct)
s(T[1])                                          # value: 2.4214387e-9 (correct)
# Float64 (same construction): current 2.6623668547554047e-15, expected 2.662280118581606e-15
#   (the level 1.3 * 2^-1072 is subnormal with two bits: a 4% error in the threshold)
# Overflow side: NormL1(1.3 * 2^(E-20), [2^(-(E-10)-20)]) under two factors 2^p, γ = 2^-30, x = 3 t:
#   Float64: current 0.0, expected 1.1143689514737182e302     (Float32: 0.0 vs 2.1093705e32)
```

**D2 (suspected): a weighted `NormL1` value is NaN at an infinite entry when
`λ0 * w` underflows**, while the same level without weights gives `Inf`
and the exact product of positive finite factors with `+Inf` is `+Inf`.
`_positive_product(a, b, Inf)` falls to `_exceptional_product` with `a*b = 0`, and
`_scaled_product(a, b, c, 0)` returns the ordinary `(a*b)*c = 0*Inf = NaN` for the
non-finite factor. (A zero factor against an infinite one is documented as NaN in
other places, but here no factor is zero: the zero comes from underflow of a
partial product.)

```julia
T = Float64; E = 1023                            # Float32 with E = 127 is the same
fu = NormL1(2.0^-(E ÷ 2), [2.0^-(E - 5)])
fu([Inf])                                        # NaN        expected Inf
NormL1(2.0^-(E ÷ 2))([Inf])                      # Inf (unweighted)
ProximalBase._positive_product(2.0^-(E ÷ 2), 2.0^-(E - 5), Inf)   # NaN, expected Inf
```

### Limits that are probably by design (recorded, not asserted)

These are not pinned with `@test_broken` because the source comments describe
them as intended, or the contract only promises recovery where a mechanism
exists. They are listed so that the move to `numerical_kernels.jl` does not
silently change them, and because a maintainer may want to promote one to a
defect.

- **S1. `SqrNormL2` (and any function without `_fold_level`) under `Scaled` with a
  step product outside the range.** `_nested_step` hands the function the step
  rounded (`"A function without levels to fold gets the step as rounded"`). With
  `Scaled(SqrNormL2(2^-(E-20)), 2^(E-20))` and `γ = 2^(E-20)` the product
  `α γ` is `Inf` while `α γ λ = 2^(E-20)` is representable:
  `prox` gives `0.0` (Float64 and Float32), expected `3/(1 + 2^(E-20)) = 3.4997e-302`
  (Float64) / `1.8489e-32` (Float32). The same happens for two-factor chains over
  `SqrNormL2`. For `NormL1` the level absorbs the product (and D1 below is the case
  where it cannot).
- **S2. Gradients of `Scaled` are the leaf gradient times the factor.** If the leaf
  gradient overflows or underflows before the factor is applied, `Scaled` returns
  `Inf` or a flushed zero although the product is representable
  (`gradient!` of `Scaled(SqrNormL2(2^(E-5)), 2^-(E-5))` at `2^(E-2)`: `[Inf]`,
  expected `2.247e307` in Float64, `4.2535296e37` in Float32). The fused value is
  recovered, the gradient is not; it applies equally to a single factor and to a
  chain. The documented recovery of "gradient differences" is for the unwrapped
  `SqrNormL2`.
- **S3. A leaf value that underflowed to exactly 0 under a factor above one is not
  recomputed when the product would be subnormal.** `_parts_wanted` uses
  `_zero_value_exponent`: the recovery runs only if the scaled value could be a
  normal number. `Scaled(SqrNormL2(1.0), 2.0^40)([1.3 * 2.0^-540])` is `0.0` where the
  representable value is `7.17e-314` (Float32: `Scaled(SqrNormL2(1f0), 2f0^20)([1.3f0 * 2f0^-77])`
  is `0.0f0`, expected `3.88e-41`). A *subnormal* leaf value is recovered.
- **S4. `Scaled(SqrNormL2(λ), Inf)` at the zero vector is `NaN`**, while
  `Scaled(NormL1(λ), Inf)` at the zero vector is `0`. `_scaled_nan_value` returns 0
  only where `_exactly_zero_at` decides it from the parameters, and that is
  implemented for indicators and `NormL1` only (the code comment says "otherwise what
  the parts give": `Inf * 0`).

## Follow-up: the QuadraticFunction proximal operator

The eigendecomposition proximal operator had no overflow recovery: `γλ`, `γb`,
`x - γb` or a product with the eigenbasis that overflows gave `0`, `±Inf` or `NaN`
at representable points (for example `QuadraticFunction(Diagonal([2.0^816]))`
with step `2.0^816` at `[2.0^970]`: `0.0` against `2^-662`; `SqrNormL2` gave the
right value). The follow-up that adds the recomputation also adds the testset
"quadratic prox: intermediate overflow at a representable proximal point"
(one-dimensional and dense cases against BigFloat, the ridge agreement, bitwise
ordinary path, exact zeros, NaN/Inf, an infinite step, aliasing, the fused oracle).

The recomputation takes the numerator `Q'(x - γb)` with `γ` applied before the
projection, as on the ordinary path: the BLAS product where `x - γb` and the
product are finite, and otherwise each entry summed from significands and
exponents (`_ExpSum`). After the projection it only divides by `1 + γλ ≥ 1` and
rescales exactly, so no projected term is rounded where a later factor could
restore it. An earlier revision rounded `Q'b` to the working type and multiplied
it by `γ` afterwards, which lost a projected coefficient below the subnormal
range that `γ` would have restored (for `A = 2^300 [1 -2; -2 4]`, `b = [0, δ]`, `x = 0`, `γ = 2^800`, the
point `-(γδ/5)[2, 1]` came out as `[0, -0]`). The testset covers that rank-one
reproducer in Float32 and Float64, a cancelling subnormal `b`, and random dense,
sparse and diagonal 2×2 forms with a zero eigenvalue beside an overflowing `γλ`
(both numerator routes), each coordinate against the BigFloat solution of `(I + γA) z = x - γb`.
`spectral_accuracy.jl` has the corresponding study.

## Remaining gaps

This is an audit of the paths reachable from the library as it is today, not a proof
that every branch is exercised; the points below are known holes.

- Line coverage cannot see branches on one line. The tests assert a distinguishing
  value for the one-line exits that were found, but no branch-level instrument (for
  example a mutation run) was used, so an untested one-line exit may remain.
- Dead code in `src/`, tested directly as a baseline but unreachable from a public
  oracle: `_weight_coefficient(::Type, γ, λ, w)`, `_weight_coefficient3`,
  `_scaled_coefficient` (correct rounding), `_recoverable_factors`; the owner and block
  methods of `_scaled_dot`. If the planned move drops them, the corresponding checks
  can go with them. (Done in the follow-up that removes duplicated and unreachable
  helpers: the helpers and their baseline checks are gone, and the owner and block
  checks now go through the live routes, `_linear_scaled_dot`.)
- The remaining deep branches of nested wrappers are reached only through the public
  types that exist in the package (`NormL1`, `SqrNormL2`, `ZeroFunction`, plus a custom
  `SeparableFunction` in the test): the `Scaled` machinery for custom leaf functions
  with `_has_value_parts` or `_fold_level` methods is exercised through those only.
  `_scale!` for tuples of blocks, device (GPU) arrays and `_is_device_array`
  routes are untested.
- Mixed `Float32`/`Float64` is documented as "not a guaranteed mode"; the mixed cases
  that the SqrNormL2 docstring describes (a wider level or center against a narrower
  model) are covered, other mixes are not.
- `_tril_column_isqrt` (p >= 2^24) is not constructible in a test; `opnorm2` and the fixed
  step bounds (`_largest_eigenvalue`, `_opnorm2`, `_opnorm_upper`) are not recovery paths
  and have no tests at all today.
- The accuracy of the three-factor products is stated in the docs as "within one
  neighboring value"; the tests check `<= 2` ulp of the correctly rounded exact term for
  `_triple` and a relative-to-`Σ|terms|` bound for sums in storage order, which is what the
  algorithm can promise; they do not test the exact ulp-neighbour statement at
  the overflow and half-denorm-min boundaries.
- Random cases use fixed seeds (one `Xoshiro` per test function); a change of the
  RNG stream in a future Julia can change which random cases run, not the properties.

## Results of this commit

- `test/numerical_recovery.jl`: 7656 passing and 5 broken assertions in Float64,
  7825 passing and 5 broken in Float32 (recorded defects D1, D2); about 55 s on one
  core including compilation (23 s + 21 s of testsets).
- Full suite, `julia scripts/test.jl`: 21 test files, all pass, in 3 m 48 s on one core
  (broken counts live inside `numerical_recovery`: 10 in total, the recorded defects).
