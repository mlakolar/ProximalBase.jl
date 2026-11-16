# External implementations: source-level inventory for the kernel study

Compiled 2026-11-15. PB = ProximalBase.jl at `50db4b1e` (`/path/to/pbopt/base`, read only).
Third-party sources were cloned (depth 1) or downloaded into
`/path/to/pbopt/harness/experiment_kernelopt/external/src-*/`. They were only read, never
built or run. The only code that was run is my own small snippets under the session scratchpad: codegen inspection
and a few-second timing on the workstation (i5-13600K, Raptor Lake, AVX2, no AVX-512). These are sanity checks, not
benchmarks.

## Versions read

| source | where | version / commit | license | PB (MIT) may |
|:--|:--|:--|:--|:--|
| ProximalOperators.jl | `src-ProximalOperators` | v0.17.0 (Project.toml), `92c25478cf76` (2026-10-18) | MIT | copy code (keep the notice) |
| Julia LinearAlgebra and Base, 1.13.4 | `~/.julia/juliaup/julia-1.13.4+0.x64.linux.gnu/share/julia/{base,stdlib/v1.13/LinearAlgebra/src}` | 1.13.4 | MIT | copy |
| Julia LinearAlgebra and Base, 1.10.13 | `~/.julia/juliaup/julia-1.10.13+0.x64.linux.gnu/share/julia/{base,stdlib/v1.10/LinearAlgebra/src}` | 1.10.13 | MIT | copy |
| OpenBLAS bundled with 1.13.4 | `src-OpenBLAS-0.3.30` | OpenBLAS_jll 0.3.30+1, tag v0.3.30 `993fad6aebbc` | BSD-3 | copy with notice (irrelevant: x87 asm) |
| OpenBLAS bundled with 1.10.13 | `src-OpenBLAS-0.3.23` | OpenBLAS_jll 0.3.23+5, tag v0.3.23 `394a9fbafe90` | BSD-3 | as above |
| Reference LAPACK (new dnrm2/dlassq) | `src-lapack` | master `49b93082cc00` (CMake says 3.12.1) | modified BSD | copy with notice |
| Reference LAPACK (old dnrm2.f) | `src-lapack-3.9.0/dnrm2.f` | raw file from tag v3.9.0 | modified BSD | copy with notice |
| glmnet (CRAN mirror) | `src-glmnet` | 5.2, `37f9d27dc161` | **GPL-2** | **ideas only** |
| grpreg (CRAN mirror) | `src-grpreg` | 3.6.0, `a1482e9d03f8` | **GPL-3** | **ideas only** |
| Lasso.jl | `src-Lasso` | 0.7.6, `877fe85c00f3` | MIT | copy |
| skglm | `src-skglm` | 0.6dev, `c73abe71b897` | BSD-3 | copy with notice |
| celer | `src-celer` | 0.7.5dev0, `eef3bb5f239c` | BSD-3 | copy with notice |

The CRAN glmnet 5.2 source no longer has the Fortran `glmnet5dpclean.m`. The algorithms are in the C++ headers
`src/glmnetpp/include/glmnetpp_bits/` (Eigen-based).

---

## 1. ProximalOperators.jl (PO)

### 1.1 `NormL1`: `src/functions/normL1.jl`

What it computes: the prox of `γ λ‖·‖₁` for a scalar λ, or of `γ Σ λᵢ|xᵢ|` for an array λ, with a scalar or array γ.
There are 8 `prox!` methods (scalar or array λ) × (scalar or array γ) × (real or complex x).
**Every `prox!` returns the value `f(y)` at the prox point**, not `y`.

Real x, scalar λ and γ: the value is fused into the same `@simd` loop.
```julia
gl = gamma * f.lambda
@inbounds @simd for i in eachindex(x)
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    n1y += y[i] > 0 ? y[i] : -y[i]
end
return f.lambda * n1y
```
Array λ: the same update with `gl = gamma * f.lambda[i]` per entry, then `return sum(f.lambda .* abs.(y))`. That
is a second pass, and the broadcast inside `sum` **allocates a temporary array** on every call. Complex x uses
`sign(x)*(abs(x) <= gl ? 0 : abs(x)-gl)`. `gradient!` is `y .= λ .* sign.(x)`, which returns `f(x)`.

- **Numerics.** For |x| < gl the result is `x + (-x) = +0`, including x = −0.0 (`-0.0 + 0.0 = +0.0`). At gl = 0,
  x = −0.0 takes the first branch: `-0.0 + 0.0 = +0`. A NaN x fails both comparisons, giving `NaN + (-NaN) = NaN`.
  x = ±Inf with finite gl gives ±Inf. gl = Inf with finite x gives +0. x = Inf with gl = Inf gives `Inf - Inf = NaN`.
  A negative λ is rejected at construction. Infinite or NaN parameters get no special handling.
- **Cost.** Two compares, a negation, two selects and an add per entry, plus a compare, a select and an add for the
  fused L1 sum. One pass for scalar λ; two passes and an allocation for array λ.
- **SIMD.** The scalar-λ real loop vectorizes (select-based, `@simd` reduction). On Julia 1.13 it compiles to about
  7.5 vector ops per 4 doubles (my codegen check of the same expression), and to about 8 on 1.10.

### 1.2 `NormL2`: `normL2.jl`
```julia
normx = norm(x)
scale = max(0, 1 - f.lambda * gamma / normx)
for i in eachindex(x); y[i] = scale*x[i]; end
return f.lambda * scale * normx
```
- **Returns** `λ‖y‖ = λ·scale·‖x‖`, computed without a second pass, the same trick as PB's `t*n`.
- **Norm.** `LinearAlgebra.norm` (§2): BLAS `nrm2` for a strided BlasFloat vector of length ≥ 32 (the x87 kernel on
  x86_64, §3), `generic_norm2` below that.
- **Numerics.**
  - Zero x with λγ > 0: `λγ/0 = Inf`, `max(0, -Inf) = 0`, `y = 0.0*x`. A −0.0 entry stays −0.0 (PB writes +0).
  - Zero x with λγ = 0: `0/0 = NaN`, and `max(0, NaN) = NaN` (Julia's max propagates NaN), so **y = NaN**. This is
    a PO bug at λ = 0 or γ = 0.
  - Overflow and underflow are handled only inside `norm`.
- **SIMD.** The scaling loop has no `@inbounds`, so y is bounds-checked. Not a tuned loop.

### 1.3 `NormL21`: `normL21.jl` (the sum of the column or row norms of a matrix)
```julia
for j in axes(X, 2)
    nslice = R(0)
    for i in axes(X, 1); nslice += abs(X[i, j])^2; end
    nslice = sqrt(nslice)
    scal = 1 - gl / nslice
    scal = scal <= 0 ? R(0) : scal
    for i in axes(X, 1); Y[i, j] = scal * X[i, j]; end
    n21X += scal * nslice
end
return f.lambda * n21X
```
- **Groups.** Only whole columns (dim = 1) or rows (dim = 2), not arbitrary index groups.
- **Per group.** One sqrt and one division. Two passes over the group, one for the sum of squares and one for the
  scaling. Returns the value from `scal*nslice`, again without a third pass.
- **Numerics.**
  - Unscaled sum of squares. A group whose squares overflow gets `nslice = Inf`, `scal = 1`, and is copied (the
    returned value is Inf).
  - A group whose squares underflow to 0 gets `gl/0 = Inf` and is zeroed. That is wrong when gl is also tiny: for
    x = 1e-200 and gl = 1e-201 the result should be 0.9e-200.
  - gl = 0 on a zero group gives `0/0` and NaN output (the same bug as NormL2).
  - A zeroed group is `0*X`, so −0 entries keep their sign.
- **SIMD.** No `@inbounds` and no `@simd`, so the sum is a sequential, latency-bound chain (about 4 cycles per
  entry). dim = 2 walks rows of a column-major array, which is strided.

### 1.4 `SqrNormL2`: `sqrNormL2.jl`
```julia
gl = gamma * f.lambda
for k in eachindex(x)
    y[k] = x[k] / (1 + gl)
    sqny += abs2(y[k])
end
return f.lambda / R(2) * sqny
```
- **Cost.** One **division per entry** (`1 + gl` is invariant, but the division is not hoisted to a reciprocal
  without fast-math), plus the fused sum of squares for the returned value. No `@simd` or `@inbounds`, so the
  reduction is sequential. The array-λ and array-γ variants do `x[k] / (1 + gamma*λ[k])`.
- **Numerics.** None special. `1 + gl` overflows for gl near floatmax, and `x/(1+gl)` loses x when gl is huge.
  PB's convex-weight formulation exists to avoid exactly that.

### 1.5 `ElasticNet`: `elasticNet.jl`

`f(x) = μ‖x‖₁ + (λ/2)‖x‖²`.
```julia
gm = gamma * f.mu; gl = gamma * f.lambda
for i in eachindex(x)
    y[i] = (x[i] + (x[i] <= -gm ? gm : (x[i] >= gm ? -gm : -x[i])))/(1 + gl)
    sqnorm2x += abs2(y[i]); norm1x += abs(y[i])
end
return f.mu * norm1x + f.lambda / R(2) * sqnorm2x
```
- **Algorithm.** Soft thresholding, then one division per entry. Two fused reductions for the returned value. No
  `@simd`, so the reductions are sequential.
- **Zeros.** Same signed-zero behavior as NormL1 (+0 below the threshold, since `+0/(1+gl) = +0`).

### 1.6 Least squares: `leastSquares.jl`, `leastSquaresDirect.jl`, `leastSquaresIterative.jl`

`LeastSquares(A, b, λ; iterative=false)` represents `f(x) = (λ/2)‖Ax − b‖²`.

**Direct** (`LeastSquaresDirect`):
- **At construction.** Eagerly forms the Gram matrix `S = λ A'A` if m ≥ n ("Tall"), else `S = λ AA'` ("Fat"), and
  `λ A'b`. That is O(mn²) or O(m²n) even if `prox!` is never called. It caches `gamma = -1` and `fact`.
- **`prox!`.** `if gamma != f.gamma; factor_step!(f, gamma); end`, then `f.fact = cholesky(f.S + I/gamma)`. So it
  **refactorizes on every change of γ**: O(k³/3) with k = min(m, n), plus an **allocation** of the k×k matrix
  `S + I/γ` each time. Sparse A uses `ldlt(S; shift=1/γ)` (CHOLMOD).
- **Solve.** `q = λA'b + x/γ`, then:
  - Tall: two `LAPACK.trtrs!` (triangular solves) on q. About n² multiply-adds in total.
  - Fat, by Woodbury: `y = γ (q − λA'(fact \ (A q)))`. That is `mul!(res, A, q)`, two `trtrs!`,
    `mul!(y, A', res)`, then `y .*= -λ; y .+= q; y .*= γ`. About 2mn + m² multiply-adds and three extra passes
    over y.
  - The fat form **subtracts terms of size γλA'b** at a large γ. This is the cancellation that PB's
    `prox!` comment explicitly avoids.
- **Returns.** `prox!` always ends with `mul!(f.res, f.A, y); f.res .-= f.b; return (λ/2)*norm(f.res)^2`: **one extra
  product with A (mn multiply-adds) plus a residual pass and a `norm` (BLAS nrm2, x87) per prox**, just to return
  the value.
- **`gradient!`.** `res = Ax − b`, `y = A'res`, `y .*= λ` (always, even for λ = 1). Returns `(λ/2) dot(res, res)`
  (BLAS ddot).
- **`f(x)`.** `res = Ax − b`, then `(λ/2) norm(res)^2`. `norm` then `^2` takes a sqrt and squares it again, and
  goes through x87 nrm2 for m ≥ 32.

**Iterative** (`LeastSquaresIterative`): CG (IterativeSolvers.jl) on `λA'A + I/γ` (Tall) or `λAA' + I/γ` (Fat),
warm-started at x (Tall). It declares `is_proximable = false` because the prox is inexact. Same returned value
(one extra product with A).

**Calculus wrappers** that touch these kernels:
- `Regularize(f, ρ, a)` is the elastic-net-style composition. `prox!` calls `prox!(y, f, gr2 .* (x .+ gr .* a), gr2 .* γ)`,
  which allocates a temporary for the scaled argument on every call, and returns
  `v + ρ/2*norm(y .- a)^2`, which allocates again and calls `norm`.
- `normdiff2(x, y)` (`utilities/normdiff.jl`) is a plain sequential loop `s += abs2(x[i]-y[i])` with no `@simd`.

**Summary for PO.** Its prox kernels are straightforward. They always compute the value at the prox point, which
costs an extra pass (L1, L2, ridge) or an extra matrix–vector product (least squares). Several loops lack
`@simd`/`@inbounds`, and several exceptional cases are wrong (0/0 at λ = 0 or γ = 0, underflow zeroing). The only
idea of interest is the select-based soft threshold, and an even cheaper select form exists (§7, idea 1).

---

## 2. Julia's `norm`, `sum(abs2, x)`, `max`/`min`: 1.13.4 versus 1.10.13

### 2.1 `LinearAlgebra.norm` for p = 2 (`generic.jl`, `dense.jl`)

- **Dispatch.** `norm(itr)` → `norm2(itr)`. For `Union{Array{T},StridedVector{T}}` with `T<:BlasFloat` (`dense.jl`,
  identical in both versions):
  ```julia
  const NRM2_CUTOFF = 32
  norm2(x::Union{Array{T},StridedVector{T}}) where {T<:BlasFloat} =
      length(x) < NRM2_CUTOFF ? generic_norm2(x) : BLAS.nrm2(x)
  ```
  So a strided Float32 or Float64 vector (or any dense Array, including matrices, as a Frobenius norm) of length
  **≥ 32 goes to BLAS `nrm2`**, and shorter ones go to `generic_norm2`.
- **1.13 changes.** `norm(itr)` gained a one-argument method (#56330, to reduce latency) and a
  `norm_recursive_check` that iterates the first element. `norm(itr, p)` is `@constprop :aggressive`. These are
  negligible.
- **`generic_norm2`** (same algorithm in both; 1.10 uses `while iterate`, 1.13 uses `Iterators.rest`):
  ```julia
  maxabs = normInf(x)                       # pass 1: mapreduce(norm, max, x)
  (ismissing(maxabs) || iszero(maxabs) || isinf(maxabs)) && return maxabs
  if isfinite(length(x)*maxabs*maxabs) && !iszero(maxabs*maxabs)   # no scaling
      sum::promote_type(Float64, T) = norm_sqr(v); for v in rest; sum += norm_sqr(v); end
      return convert(T, sqrt(sum))
  else
      sum = abs2(norm(v)/maxabs); for v in rest; sum += (norm(v)/maxabs)^2; end
      return convert(T, maxabs*sqrt(sum))
  end
  ```
  - **Passes.** Always **two** (the max, then the sum).
  - **Sum.** A sequential loop without `@simd` (latency-bound). Float32 accumulates in Float64.
  - **Scaling path.** One **division per entry**.
  - **The scaling test is crude.** `n·maxabs²` finite and `maxabs²` nonzero. It does not detect underflow of the
    small entries when the largest one is moderate.
  - **NaN.** Gives NaN through `maxabs`.
  - **Inf.** Returned directly (`isinf(maxabs)`), even if a NaN is also present (the max may return Inf first).
  - **Zero.** Returns `maxabs = 0`, with no 0/0.
- **`norm_sqr(x::Number) = abs2(x)`.**
- **Timing sanity check** (Float64, workstation, warm, ns per entry; one run each, not a benchmark):

  | n | `BLAS.nrm2` | AVX2 `@simd` loop `s += x*x` | `sum(abs2, x)` | `norm(x)` |
  |:--|:--|:--|:--|:--|
  | 16 | 0.38 | 0.07 | 0.26 (1.13) / 0.51 (1.10) | **0.97 (1.13) / 1.55 (1.10)** (`generic_norm2`) |
  | 1000 | 0.21 | 0.04 | 0.05 to 0.06 | 0.21 (= nrm2) |
  | 100 000 | 0.20 | 0.05 | 0.065 | 0.20 (= nrm2) |

  So **`norm(x)` of a Float64 vector costs 4 to 5× an AVX2 sum of squares while the data is in cache**, and 14× at
  n = 16.

### 2.2 `sum(abs2, x)` and the pairwise `mapreduce` (`base/reduce.jl`)

The pairwise kernel is **identical in both versions**:
- `sum(f, a) = mapreduce(f, add_sum, a)`. `_mapreduce` on IndexLinear calls `mapreduce_impl(f, op, A, ifirst, ilast,
  pairwise_blocksize(f, op))`.
- `mapreduce_impl` (the generic one, `@noinline`): a block of `ilast − ifirst < blksize` entries is reduced
  sequentially with `@simd for i = ifirst+2:ilast; v = op(v, f(A[i])); end`. Larger ranges are split at the middle
  and recursed.
- `pairwise_blocksize(f, op) = 1024`. There is also `pairwise_blocksize(::typeof(abs2), ::typeof(+)) = 4096` in both
  versions, but **`sum(abs2, x)` does not use it**, because its op is `add_sum`, not `+`. PB's
  `_ls_sumabs2_diff_pairwise` (a closure f with `+`) does not use it either. So PB's comment on `_LS_ONE_LOOP_MAX`
  ("blocks of at most 1024 entries … on Julia 1.10 and 1.13") is correct. Only a literal `mapreduce(abs2, +, x)`
  would use 4096.
- **Codegen.** LLVM vectorizes the `@simd` block loop with 4 accumulators × 4 lanes (AVX2, Float64), so each lane
  sums at most 64 terms of a block.

**Difference: max and min reductions.**
- 1.10 has a specialized `mapreduce_impl` for `op::Union{typeof(max), typeof(min)}`:
  - It uses 4 explicit accumulators over chunks of 256, with
    `_fast(max, x, y) = ifelse(isnan(x), x, ifelse(x > y, x, y))`.
  - A final fix-up pass (`isbadzero`) rescans the array when the result is a "bad" zero (−0 for max), so that
    `maximum([0.0, -0.0]) === 0.0`.
- **1.13 removed it** (no `_fast` and no `isbadzero` in `base/`). `maximum` and `mapreduce(f, max, x)` take the
  generic pairwise `@simd` path with `max = max_float` (the LLVM `maximum` intrinsic).
- This changes `generic_norm2`'s first pass and anything built on `maximum`. It does **not** touch PB's
  `_maximum_abs`, which reduces bit patterns as integers.

### 2.3 `max`/`min` for Float64 (`base/math.jl`)

- **1.10.13** (math.jl:894–912). Non-aarch64 x86 has `has_native_fminmax = false`:
  ```julia
  function max(x::T, y::T) where {T<:Union{Float32,Float64}}
      diff = x - y
      argmax = ifelse(signbit(diff), y, x)
      anynan = isnan(x)|isnan(y)
      return ifelse(anynan, diff, argmax)
  end
  ```
  `min` is the same with the select reversed. These are IEEE-2019 `maximum`/`minimum` semantics (NaN propagates,
  −0 < +0), built by hand from a subtraction, a sign test and NaN tests. The LLVM IR is
  `fsub, select, fcmp ord, select`.
- **1.13.4** (math.jl:852–858):
  ```julia
  max(x::T, y::T) where {T<:IEEEFloat} = max_float(x, y)   # → call double @llvm.maximum.f64
  min(x::T, y::T) where {T<:IEEEFloat} = min_float(x, y)
  ```
  The semantics are the same. The x86 backend lowers `llvm.maximum` to `vmaxpd` plus NaN and signed-zero fix-ups,
  unless operands are known.
- **`clamp(x, lo, hi)`** is select-based in both versions: `x > hi ? hi : x < lo ? lo : x` (1.13 `intfuncs.jl:1403`,
  1.10 `math.jl:98` as `ifelse`). It does not call max or min.

**Codegen of PB's `shrink`** (`max(v - c, 0) + min(v + c, 0)`) in an `@inbounds @simd` loop over Float64 arrays with a
scalar c (my snippet; the main vector body unrolled ×4, non-memory ymm ops per 4 doubles):

| expression | Julia 1.13.4 | Julia 1.10.13 |
|:--|:--|:--|
| PB `max(v-c,0) + min(v+c,0)` | **7**: `vsubpd, vmaxpd, vcmpunordpd, vblendvpd, vaddpd, vminpd, vaddpd` | **11**: sub, add, `vcmpordpd`, `vpcmpgtq`, `vpand`, `vblendvpd`, `vcmpordpd`, `vpcmpgtq`, `vandpd`, `vandnpd`, add |
| `max(v-c,-0.0) + min(v+c,0)` | 5 | 11.5 |
| PO form `v + (v <= -c ? c : (v >= c ? -c : -v))` | about 7.5 | about 8 |
| `v - clamp(v, -c, c)` (selects) | **3**: `vmaxpd, vminpd, vsubpd` | **3** |

- **Why 1.13 needs a fix-up for max.** `max(x, +0.0)` needs the NaN fix-up (`vcmpunordpd + vblendvpd`; `vblendvpd`
  ymm is several uops on Intel). `min(x, +0.0)` maps to a single `vminpd(0, x)`, because the x86 operand order
  then gives the IEEE result. `max(x, -0.0)` would also be a single `vmaxpd`.
- **Clamp needs no fix-up.** `vmaxpd(-c, v)` returns v when v is NaN, so NaN propagates.
- **Same results.** On a grid of ±0, ±0.5, ±1, ±2.5, ±Inf, ±1e308, ±5e-324 with c ∈ {0, 5e-324, 1, 1e308, Inf},
  all four forms are **bitwise identical** to PB's on non-NaN inputs (checked with `isequal`) and NaN for NaN.

---

## 3. OpenBLAS `nrm2`, `dot`, `gemv` on x86_64

**Versions.**
- Julia 1.13.4 bundles `libopenblas64_.0.3.30.so` (OpenBLAS_jll 0.3.30+1).
- Julia 1.10.13 bundles `libopenblas64_.0.3.23.so` (OpenBLAS_jll 0.3.23+5).
- Both are ILP64 DYNAMIC_ARCH builds behind libblastrampoline.

**Kernel selection** (`kernel/x86_64/KERNEL*`, the same in both tags):
- **No x86_64 target overrides `DNRM2KERNEL` or `SNRM2KERNEL`.** Only `KERNEL` (the defaults) and `KERNEL.generic`
  mention NRM2. `KERNEL.SKYLAKEX` includes `KERNEL.HASWELL`, `KERNEL.COOPERLAKE` includes `KERNEL.SKYLAKEX`, and
  `KERNEL.ZEN` defines none. So for HASWELL, ZEN, SKYLAKEX and COOPERLAKE:
  - `DNRM2KERNEL = nrm2.S` (x87)
  - `SNRM2KERNEL = nrm2_sse.S`
- `DDOTKERNEL = ddot.c`, with `ddot_microk_haswell-2.c` (HASWELL, ZEN) or `ddot_microk_skylakex-2.c` (SKYLAKEX,
  COOPERLAKE, SAPPHIRERAPIDS). These are AVX2/FMA or AVX-512 microkernels.
- `DGEMVNKERNEL = dgemv_n_4.c` (microkernel `dgemv_n_microk_haswell-4.c` or `_skylakex-4.c`) and
  `DGEMVTKERNEL = dgemv_t_4.c` (`dgemv_t_microk_haswell-4.c`).

**`dnrm2`: `kernel/x86_64/nrm2.S` (identical in both tags except one line).**
- Four **x87** accumulators (`fldz` ×4). The loop is unrolled by 8:
  `FLD 0*SIZE(X); fmul %st(0),%st; … faddp %st,%st(7); faddp %st,%st(5); …`. Strided input has a separate loop.
- The final steps are `faddp` ×3, `fsqrt`, and `FST (%rsp); MOVSD (%rsp),%xmm0`, so the result is rounded to double
  once.
- 0.3.23 tests `INCX` with `jle` and 0.3.30 with `je`. Julia never passes a nonpositive stride here (a reversed view
  gave the right norm on both).

**Algorithm and numerics.**
- A **plain, unscaled sum of squares in 80-bit extended precision** (64-bit significand, 15-bit exponent).
- No finite double can overflow or underflow when squared there (2^±2048 against a range of 2^±16382). So the result
  is overflow- and underflow-safe without scaling, and more accurate than a double sum. A check gave
  `nrm2([1; fill(2^-27, 40)]) = 1.000000000000001`, which is the correctly rounded value, while a double loop gives
  `1.0000000000000007`.
- Inf and NaN follow x87 IEEE rules (Inf² = Inf, NaN propagates).

**Cost.** Scalar x87, one `fmul` and one `fadd` per element, at best about 1 element per cycle. Measured
0.20 ns/elem, against 0.04 to 0.05 ns/elem for a 4×4-accumulator AVX2 loop: **4 to 5× slower in cache**. When the
data streams from DRAM the gap narrows towards the bandwidth limit.

**Interface** (`interface/nrm2.c`).
- n ≤ 0 returns 0.
- 0.3.30 adds shortcuts: n == 1 returns `|x[1]|`, and incx == 0 returns `sqrt(n)|x[1]|`.
- Negative incx is rebased.

**`snrm2`: `nrm2_sse.S`.** Converts float to double (`cvtps2pd`) and accumulates squares in double with 128-bit
SSE2 (`mulpd`/`addpd`), using four 2-lane accumulators, 8 floats per iteration. Then `sqrtsd` and `cvtsd2ss`. Also
unscaled and range-safe (Float32 squares fit in double). The cost is roughly 2 to 4× an AVX2 Float32 loop, which is
also more accurate.

**Surprise: CPU detection on the workstation.**
- OpenBLAS 0.3.23 (Julia 1.10) reports **core "Prescott"** on this Raptor Lake i5-13600K.
- `driver/others/dynamic.c` in 0.3.23 knows Alder Lake (extended model 9, models 7 and 10) but not Raptor Lake
  (0xB7 is extended model 11), so it falls back to the generic Prescott (SSE3) kernels for **all** of BLAS,
  including dgemv and ddot.
- OpenBLAS 0.3.30 (Julia 1.13) reports "Haswell".
- Consequence: **local Julia 1.10 versus 1.13 comparisons of anything calling BLAS (least-squares gradients and
  prox) are confounded on this machine**. The cluster CPUs (E5-2640 v4, EPYC) are detected normally by both
  versions. `nrm2` is the x87 kernel on every target either way.

---

## 4. Reference LAPACK: `dnrm2.f90` / `dlassq.f90` (≥ 3.10), and the old `dnrm2.f`

### 4.1 New `BLAS/SRC/dnrm2.f90` (Anderson 2017, "Blue's algorithm" with three accumulators)

**Thresholds** (from `radix`, `minexponent`, `maxexponent`, `digits`):

| constant | Float64 | Float32 |
|:--|:--|:--|
| tsml | 2^−511 | 2^−63 |
| tbig | 2^486 | 2^52 |
| ssml | 2^537 | 2^75 |
| sbig | 2^−538 | 2^−76 |

```fortran
do i = 1, n
   ax = abs(x(ix))
   if (ax > tbig) then
      abig = abig + (ax*sbig)**2; notbig = .false.
   else if (ax < tsml) then
      if (notbig) asml = asml + (ax*ssml)**2
   else
      amed = amed + ax**2
   end if
end do
! combine: abig dominant → abig + (amed*sbig)*sbig, scl = 1/sbig
!          asml and amed → ymax**2*(1 + (ymin/ymax)**2) of the two square roots
DNRM2 = scl*sqrt(sumsq)
```

- **Mid accumulator.** Squares of ax ∈ [2^−511, 2^486] lie in [2^−1022, 2^972]. They are always normal and never
  overflow for n < 2^52.
- **Big and small accumulators.** Big entries are scaled down by 2^−538 and small ones up by 2^537, so neither range
  can over- or underflow.
- **Skipping.** Once a big entry is seen, small ones are skipped (`notbig`), since they cannot matter.
- **Handles:**
  - overflow of the sum of squares of finite entries (up to the final `scl*sqrt`, which overflows only when the norm
    itself is unrepresentable);
  - underflow of squares of tiny entries, including subnormals;
  - mixtures of the two.
- **NaN.** NaN falls into `amed` (both comparisons are false), and the `amed /= amed` and `LA_ISNAN` tests in the
  combination keep it, so the result is NaN.
- **Inf.** Goes to `abig = Inf`, giving Inf, unless there is a NaN, which gives NaN.
- **Zero.** All entries land in `asml = 0`, so `sumsq = amed = 0` and the result is 0. There is no division by zero.
- **Cost.** One pass, one `abs`, 1 to 2 compares, one multiply-add (plus one more multiply off the mid range) per
  entry, and data-dependent branches. **Not SIMD as written.** A masked SIMD version needs about 3 accumulators,
  2 compares and 2 or 3 blends per vector, roughly 2 to 3× a plain AVX2 sum of squares.

### 4.2 `SRC/dlassq.f90`

Same three accumulators, with thresholds from `la_constants.f90`. It also takes an incoming `(scale, sumsq)` pair,
which it classifies into one of the accumulators by `scale*sqrt(sumsq)`. A NaN `scale` or `sumsq` returns
immediately. A zero scale or sumsq resets the pair to (1, 0).

### 4.3 Old `dnrm2.f` (LAPACK ≤ 3.9: Hammarling's one-pass scale and ssq)
```fortran
DO IX = 1, 1+(N-1)*INCX, INCX
  IF (X(IX).NE.ZERO) THEN
    ABSXI = ABS(X(IX))
    IF (SCALE.LT.ABSXI) THEN
      SSQ = ONE + SSQ*(SCALE/ABSXI)**2;  SCALE = ABSXI
    ELSE
      SSQ = SSQ + (ABSXI/SCALE)**2
    END IF
  END IF
END DO
NORM = SCALE*SQRT(SSQ)
```
- **Cost.** **One division per nonzero entry** (two operations when the scale grows) and branchy. Latency-bound
  through `SSQ` and `SCALE`. Several times a plain sum.
- **Numerics.**
  - Range-safe for finite input.
  - **Two Infs give NaN**: the second does `SSQ + (Inf/Inf)^2`.
  - NaN propagates unless the scale is later reset.
  - Zeros are skipped.
  - n == 1 returns `ABS(X(1))`.

---

## 5. glmnet 5.2 (C++ glmnetpp, GPL-2: ideas only)

**Elastic-net coordinate update** (`elnet_point/internal/base.hpp:50-68`), static `update_beta`:
```cpp
auto u = gk + a_copy * x_var;              // partial-residual correlation + x_k'x_k * old beta
auto v = std::abs(u) - penalty * l1_regul; // vp_k * lambda * alpha
a = 0.0;
if (v > 0.0) a = std::max(a_min, std::min(a_max, std::copysign(v,u) / (x_var + penalty * l2_regul)));
```
This is `β_k ← clamp(S(g_k + xv_k β_k, vp_k λα) / (xv_k + vp_k λ(1−α)), lower_k, upper_k)`, with
`xv_k = Σ w_i x_ik²` (1 for standardized columns).
- **Signed zeros.** Below the threshold the result is **+0**. Above it, `copysign(v, u)` with v > 0, so it is
  never zero.
- **NaN.** A NaN u gives `v > 0.0` false, so **a = 0: NaN is silently mapped to zero**.
- **Cost.** One division per coordinate update (the denominator is not cached as a reciprocal), one compare
  branch, and box clamps. Scalar code, since coordinate descent is inherently sequential.

**Naive versus covariance updating.**
- **Naive** (`gaussian_naive.hpp`, `gaussian_base.hpp:376`). `gk = compute_grad(r, X.col(k))`, an O(n) dot with the
  current residual. After a change d, `r -= d * x_k`, an O(n) axpy (Eigen, vectorized).
- **Covariance** (`gaussian_cov.hpp`, `gaussian_base.hpp:238-289`).
  - Keeps `g_j = x_j'r` for all p and a cache `c_` (p × nx) of Gram columns `x_j'x_k` for the active k only.
  - A new active k fills its column with `X.col(j).dot(X.col(k))`, reusing symmetric entries already cached.
  - After a change d of β_k it does `g_j -= c_(j, active_idx(k)) * d` for all j, which is O(p) per update and
    independent of n.
  - The R^2 bookkeeping is `rsq += d*(2*gk - d*xv)`.

**Grouped multi-response (mgaussian) update** (`gaussian_base.hpp:631-655`):
```cpp
g_next = g_curr + xvk * ak;                 // vector of nr responses
auto gkn = g_next.norm();                   // Eigen: sqrt(squaredNorm()), unscaled
auto u = 1.0 - l1_regul * vpk / gkn;
if (u <= 0.0) ak.setZero();
else { ak = (u/(xvk + l2_regul * vpk)) * g_next; chkbnds(...); }
```
- **Per group.** Group soft thresholding with the ridge denominator folded into **one scalar factor**. One sqrt and
  two divisions per group, then one scaling pass. `chkbnds` handles box constraints iteratively and rarely.
- **Numerics.**
  - The norm is unscaled, so it can overflow or underflow.
  - A zero gradient group with a positive penalty gives `l1*vpk/0 = Inf`, `u = -Inf`, and the group is zeroed (+0).
  - **A zero group with `l1*vpk == 0` gives 0/0 = NaN, then `u <= 0` is false, so ak = NaN.**
  - A NaN gradient makes u NaN, so `u <= 0` is false and ak becomes NaN (it propagates here, unlike the scalar
    case).

---

## 6. Other group-lasso and proximal libraries (short entries)

### 6.1 grpreg 3.6.0 (`src/grpreg_init.c`, `src/gdfit_gaussian.c`, GPL-3: ideas only)
```c
double norm(double *x, int p) { double s = 0; for (j...) s = s + pow(x[j],2); return sqrt(s); }
double S(double z, double l) { if (z > l) return z-l; if (z < -l) return z+l; return 0; }
// gd_gaussian, with groups orthonormalized beforehand (X_g'X_g/n = I):
z[j] = crossprod(x, r, n, j)/n + a[j];          // K dots of length n
len = S(norm(z, K[g]), lam1) / (1+lam2);         // grLasso; F/Fs for MCP/SCAD
if (len != 0 || a[K1[g]] != 0)                   // skip the residual update for a group that stays zero
  for (j in group) { b[j] = len * z[j] / z_norm; shift = b[j]-a[j]; r -= x_j * shift; }
```
- **Algorithm.** Because the groups are orthonormalized, the group update is closed form: block soft thresholding of
  `z` with a scalar `len/‖z‖`.
- **Per group.** One sqrt and **one division per entry** (`z[j]/z_norm`, not a reciprocal). The norm is a
  sequential unscaled `pow(x,2)` sum.
- **Skipping.** **The residual update is skipped** for a group that was zero and stays zero.
- **Numerics.**
  - S maps NaN to 0.
  - **A zero `z` with a previously nonzero group gives `0*0/0` = NaN.**

### 6.2 Lasso.jl 0.7.6 (`src/coordinate_descent.jl`, MIT)
```julia
S(z, γ) = abs(z) <= γ ? zero(z) : ifelse(z > 0, z - γ, z + γ)
newcoef = S(v, λωj*α)/(Xssq[icoef] + λωj*(1 - α))      # naive; covariance mode uses an XtX cache
```
- **Update.** The same elastic-net update as glmnet: soft threshold, then one division per coordinate update. The
  covariance variant (`Xty`, `getXtX`) mirrors glmnet's.
- **Numerics.**
  - |z| ≤ γ gives +0, including z = −0 at γ = 0.
  - NaN propagates: `abs(NaN) <= γ` is false and `NaN + γ` is NaN.
- **SIMD.** One branch and one select. Scalar use only.

### 6.3 skglm 0.6dev (`skglm/utils/prox_funcs.py`, BSD-3)
```python
def ST(x, u, positive=False):  x > u → x-u; x < -u → x+u; else 0.
def ST_vec(x, u): return np.sign(x) * np.maximum(0., np.abs(x) - u)   # gives -0.0 for negative |x|<u
def BST(x, u):
    norm_x = norm(x)
    # Equality includes x=0, u=0: the other branch would divide by zero.
    if norm_x <= u: return np.zeros_like(x)
    else: return (1 - u / norm_x) * x
BST_vec: scaling = np.maximum(1 - u / norm(x.reshape(-1, g), axis=1), 0)   # equal-size groups only
```
- **Zero test.** `norm_x <= u` is chosen explicitly to avoid 0/0 at x = 0, u = 0.
- **Signed zeros.** `ST_vec` uses sign × max, so a negative entry below the threshold becomes **−0.0**. Scalar ST
  gives +0.
- **Norm.** numba's `np.linalg.norm`, unscaled. Not checked here whether numba calls BLAS for it.

### 6.4 celer 0.7.5dev0 (`celer/cython_utils.pyx`, `celer/multitask_fast.pyx`, BSD-3)
```cython
cdef inline floating ST(floating x, floating u) nogil:
    if x > u: return x - u
    elif x < - u: return x + u
    else: return 0
cdef void BST(int n_tasks, floating * x, floating u) nogil:
    norm_x = fnrm2(&n_tasks, x, &inc)         # BLAS nrm2
    if norm_x < u: x[:] = 0
    else: tmp = 1. - u / norm_x; fscal(&n_tasks, &tmp, x, &inc)
```
- **Norm.** Block soft thresholding via BLAS `nrm2` and `scal`. With an OpenBLAS-backed SciPy on x86_64 this is the
  x87 kernel of §3 (my inference).
- **0/0.** `norm_x < u` is strict, so `norm_x = u = 0` gives `1 - 0/0 = NaN`.
- **Signed zeros.** `norm_x == u > 0` scales by 0, keeping −0 signs.

---

## 7. Comparison with PB

### A. Soft thresholding and weighted L1 (`shrink`, `_threshold_product`, `_l1_prox_threshold`, norml1.jl, separable.jl)

**What PB does.**
- `shrink(v, c) = max(v - c, zero(T)) + min(v + c, zero(T))` (numerical_kernels.jl:21).
- `prox!` on a dense array is `_prox_sep_dense!`, an `@inbounds` loop (no `@simd`, none needed without a reduction)
  of `prox_block(f, x[i], i, γ)` → `_l1_prox_threshold(v, γ, λ0, w_i, S)`:
  - `_threshold_product` forms `ab = γλ0` (loop-invariant) and `c = ab*w_i`, plus a normal-range flag on ab
    (invariant; the comment notes LLVM unswitches it).
  - The ordinary path is `shrink(v, c)`. Unweighted: `w = true`.
- `value_and_prox!` is prox, then a separate value pass (`_value_sep`), on purpose (separable.jl:203-207).
- The value of unweighted dense arrays is `λ0 * sum(abs, x)` (pairwise), with post-hoc checks.

**External approaches.**

| | operation per entry | ops per 4 doubles (1.13 / 1.10) | signed zero below threshold | NaN |
|:--|:--|:--|:--|:--|
| PB `shrink` | max + min + 3 add/sub | 7 / 11 | +0 | NaN |
| PO NormL1 | 2 compares, 2 selects, negation, add (+ fused L1 sum) | about 7.5 / 8 | +0 | NaN |
| glmnet, grpreg, celer, skglm `ST` | branchy `\|z\| − λ`, copysign or ±λ | scalar | +0 | glmnet, grpreg, celer: **0**; skglm ST: 0 |
| skglm `ST_vec` | sign × max | — | **−0** | NaN |
| Lasso.jl | branch + select | scalar | +0 | NaN |
| `v - clamp(v, -c, c)` (none of the libraries uses exactly this form) | max, min, sub | **3 / 3** | +0 | NaN |

**PB is already as good as or better than every library** on semantics: NaN propagates, −0 maps to +0, and the
threshold is formed once. The *formula*, though, costs 2.3× (1.13) to 3.7× (1.10) the vector ops of the clamp form,
which gives identical bits. PO's fused value is extra work that PB's `prox!` does not do.

### B. Euclidean norm and group-L2 shrinkage (`_l2_shrink!`, `_indexed_sumabs2`, `_l2_term`, `_l2_value`)

**What PB does.**
- **Norm.** One `@simd` sum of squares in the working precision (`_indexed_sumabs2`), then `sqrt`. The ordinary
  result is kept when `isfinite(s) & s ≥ n·floatmin/eps` (`_squares_normal`). Everything else goes to a
  `@noinline` cold path:
  - an exponent pass (`_maximum_abs` on bit patterns, which vectorizes);
  - a compensated sum of scaled squares in at least Float64.
- **Shrinkage.** The threshold `c = γλw` comes from `_threshold_product`, with a range test. Then: one sqrt, one
  compare `n > c`, one division `c/n`, and a scaling pass `y = t*x`. The fused value is `t*n`, with no third pass.
- **Exits.** The zero group gives +0, through a bitwise OR test (`_l2_zero_group`) only when s == 0. The
  parameter-decided exits (w = 0, Inf, λ = 0, γ = 0 or Inf) are out of line.
- **`NormL2(x)` value.** `_l2_value` uses `_sumabs2 = sum(abs2, x)` (pairwise, 1024-entry SIMD blocks). PB never
  calls `LinearAlgebra.norm` in these kernels. It defines `norm` methods for `SparseIterate` and `BlockIterate`
  views that forward to `norm` on the stored values, and therefore to x87 `nrm2` for ≥ 32 stored Float64 entries.
  The Lanczos `_largest_eigenvalue` uses `norm(v)`, which is negligible next to its matrix–vector products.

**External approaches.**

| | ordinary-path passes | over/underflow safety | cost relative to PB's ordinary norm |
|:--|:--|:--|:--|
| PB | 1 SIMD pass + post-test | cold rescaled recompute | 1× |
| Julia `norm`, n ≥ 32 → OpenBLAS x87 `nrm2` | 1 scalar x87 pass | intrinsic (80-bit) | **4 to 5× in cache** |
| Julia `generic_norm2`, n < 32 | 2 passes (max, then sequential sum; division per entry if scaling) | crude test | about 14× at n = 16 |
| LAPACK 3.10+ Blue `dnrm2` | 1 branchy scalar pass, 3 accumulators | full | about 2 to 3× if made SIMD; more as written |
| old LAPACK `dnrm2.f` | 1 pass, a division per entry | yes, except 2×Inf → NaN | several × |
| PO NormL2, NormL21; glmnet `g.norm()`; grpreg `norm` | 1 pass, unscaled (PO NormL21 and grpreg sequential) | none | 1× (Eigen) to 4× (sequential) |

**Group shrinkage arithmetic.** PB, PO and glmnet all use one sqrt and one division per group plus a scaling pass.
grpreg divides per entry. skglm and celer are the same as PB, apart from their 0/0 conventions.

**PB already does the cheapest thing** on the ordinary path, and is the only implementation here with both SIMD speed
and full range safety. Nothing to import. The Blue three-accumulator pass would make the cold path one pass instead
of two, but cold-path speed is irrelevant, and rulings R13 and R20 forbid new recovery work.

**Small groups.** PB's `@simd` group-sum loop (Float64, AVX2) vectorizes with VF = 4 and interleave 4, so the main
vector body needs ≥ 16 entries.
- On **Julia 1.13** LLVM adds a VF = 4 epilogue loop, so groups of 4 to 15 are partly vectorized. Groups given as
  `Vector{Int}` use **`vgatherqpd`** gathers.
- On **Julia 1.10** there is no epilogue loop and no gather, so **groups shorter than 16 are summed by a
  scalar, sequential chain**.

(Checked by `code_llvm` of a copy of `_indexed_sumabs2` on both versions.) Eigen's `squaredNorm` (glmnet) uses
packet-sized vectors with a scalar tail, so it vectorizes from 4 entries.

### C. Least squares (`_ls_value`, `_leastsquares_residual_gradient!`, `prox!` via cached thin SVD, leastsquares_predictor.jl)

**What PB does.**
- **Value.** `_simd_sumabs2` / `_ls_simd_sumabs2_diff`: one 16-lane loop up to 2^17 entries, the pairwise
  reduction above that. The bit-level range test `_ls_ordinary_sum` decides; everything else goes to one cold call.
- **Gradient.** `d .= z .- Y` (one pass), then `mul!(g, X', d)`, which is OpenBLAS `dgemv_t` (`dgemv_t_4.c` with the
  Haswell microkernel). The value is `_ls_value(f, d, nothing)`, a separate pass over d.
- **`prox!`.**
  - Lazily computes the thin SVD on first use and caches V, s², `diag(s)U'Y`.
  - Then each call costs `w = V'v` (gemv), r divisions, and `β = V w` (tall) or `β = v + V w` (fat).
  - **Any γ costs the same, with no refactorization.** The forms avoid subtracting γX'Y-sized terms.
  - **`prox!` returns β**, not a value.

**External approaches.**

PO Direct, per call at a fixed γ, k = min(n, p):

| shape | multiply-adds | extra |
|:--|:--|:--|
| tall | two `trtrs` = p² | + np for the returned value, an x87 `nrm2` |
| fat | 2np + n² | + np for the value; 3 extra passes over β; cancellation at large γ |

At each change of γ, PO does a Cholesky O(k³/3) and allocates a k×k matrix. It builds the Gram matrix eagerly at
construction.

PB, per call:

| shape | multiply-adds |
|:--|:--|
| tall | 2p² (two gemv over the p×p matrix V) |
| fat | 2np (the minimum for identity plus a rank-n operator) |

γ changes are free.

So:
- **Fat case.** PB is cheaper than PO (no n², no refactorization, no value product) and more accurate.
- **Tall case at a repeated γ.** PO's solve does **half the multiply-adds and half the memory traffic** of PB's two
  gemv. That is the one place where an external design beats PB (§8, idea 2).
- **PO's `gradient!`.** Always multiplies by λ, and `f(x)` uses `norm(res)^2` (x87 nrm2, then sqrt then square).
  PB is cheaper.

### D. Ridge and elastic net (sqrnorml2.jl; there is no ElasticNet type on master)

**What PB does.**
- The ridge prox is the convex combination `a x + b c` with `a = 1/(1+t)` and `b = t/(1+t)`, formed **once per call**
  (`_ridge_weights_t`, with a `t ≤ 1` branch for accuracy). The center-free case is a `z .= a .* x` broadcast:
  **one multiply per entry**.
- Exceptional steps and levels go to a scaled path decided once per call.
- An elastic net is assembled from `NormL1` and `SqrNormL2` by composition (the #909 ElasticNet is not on master;
  ruling R2 forbids an ElasticNet-specific ridge).

**External approaches.**

| | ridge part | elastic net |
|:--|:--|:--|
| PO SqrNormL2, ElasticNet | **one division per entry**, sequential fused value | `S(x, γμ)/(1+γλ)`, same |
| glmnet, Lasso.jl | — | `S(u, λα)/(xv + λ(1−α))`, one division per coordinate update |
| glmnet mgaussian | — | ridge folded into the group factor `u/(xv + λ₂vp)` |

**PB is already cheaper** than PO (a multiply by a precomputed weight instead of a division, and no value pass). In
coordinate descent the denominator `xv_k + λ₂` is fixed per coordinate, so caching its reciprocal (or `a_k`) would
remove the per-update division in glmnet and Lasso.jl. PB has no coordinate-descent loop of its own (that is
ProximalSolvers' business).

---

## 8. Ideas worth an experiment (ranked)

Every idea is subject to the standing gate in `prior_findings.md`:
- no reproducible slowdown above 5% (and none above A/A noise for #901-scope code) on any existing ordinary path;
- measured on Julia 1.10 **and** 1.13, on more than one CPU type, with a noise-floor tree.

**1. Soft threshold as `v - clamp(v, -c, c)`** (select form: `v - ifelse(v > c, c, ifelse(v < -c, -c, v))`, which is
what `Base.clamp` is in both versions).
- **Where.** `shrink` and therefore every NormL1 `prox!`/`prox_block`, the ElasticNet composition, and any PS loop
  that calls `shrink`. The study harness already has `st_clamp` and `st_clampsel` in `kernels/kernels.jl`.
- **Mechanism.** 3 vector ops per 4 doubles instead of 7 (1.13) or 11 (1.10): no NaN or signed-zero fix-ups, no
  `vblendvpd`. Expect a gain only when the loop is compute-bound (arrays in L1/L2, or the weighted loop with a
  per-entry `γλ0·w` multiply). DRAM-resident arrays are bound by load and store traffic either way. The gain should
  be larger on 1.10.
- **Lighter alternative.** `max(v - c, -zero(T)) + min(v + c, zero(T))` drops to 5 ops on 1.13 but does not help
  1.10.
- **Numerical risk.**
  - Bitwise identical to PB on all non-NaN inputs tested, including ±0, ±Inf, subnormal and ±1e308 values and
    thresholds 0, 5e-324, 1e308 and Inf. The proof sketch: |v| ≤ c gives `v − v = +0`; v > c gives `v − c`; v < −c
    gives `v + c`, the same roundings as PB.
  - A NaN v propagates (`vmaxpd(-c, v)` returns v). The NaN payload and sign may differ.
  - A NaN c only reaches `shrink` off the ordinary path. `clamp` with a NaN c returns v, so the result is +0, while
    PB's gives NaN. **Check `_l1_prox_threshold`'s exceptional path** (`_weight_product` can return Inf or NaN
    thresholds), or keep the old formula there.
- **Falsifiable claim.** `prox!(y, NormL1(λ), x, γ)` on a Float64 n = 1000 array (in L1) runs at least 1.3× faster on
  both Julia versions, and the weighted version at least 1.15× faster. Otherwise, drop the idea.

**2. Least-squares `prox!` at a repeated step, tall X (n ≥ p): cache a one-product operator.**
- **The change.** On the second call with the same γ, build `H_γ = V diag(ρ/(s²+ρ)) V'` (symmetric p×p) and
  `c_γ = V diag(1/(s²+ρ)) diag(s)U'Y` from the cached SVD, at O(p³) with gemm. Then `β = c_γ + H_γ v` is **one gemv**.
- **Alternative.** PO's design, a Cholesky of `X'X + ρI` with two triangular solves. Same multiply-adds, but trsv is
  slower per multiply-add than gemv in OpenBLAS.
- **Mechanism.** p² instead of 2p² multiply-adds per prox, and V is read once instead of twice. This halves the
  memory traffic when p×p doesn't fit in cache. It pays back after about p/2 calls at one γ (ADMM with a fixed ρ).
  **No gain in the fat case**, where PB's 2np is already the minimum.
- **Numerical risk.** The eigenvalues of H lie in (0, 1]. Neither form subtracts γX'Y-sized terms, so accuracy should
  match the two-gemv form to within a few ulps per entry. Bitwise results change (allowed per R28 for LS outputs).
- **Costs.** p² extra memory, a second code path, a γ-equality test per call, and the extra build time on the first
  repeat.
- **Check first** that PS actually calls `prox!` of LeastSquaresLoss repeatedly at one step.

**3. Fused prox and value for `value_and_prox!` of NormL1** (PO's fused loop `y = …; s += |y|`, which vectorizes).
- **Mechanism.** One pass instead of two, saving a reread of y. This matters for DRAM-resident n; expect at most about
  1.3× there and nothing in cache.
- **Numerical risk.** The sum order changes from pairwise `sum(abs, ·)` to 16 sequential lanes, with error growth
  O(n/16·eps) instead of O(log n·eps). Keep the existing `_sum_kept` and overflow post-tests. The analogous LS
  decision (R27) chose pairwise accuracy for n > 2^17, so a length cap would be needed.
- **Caution.** PB's comment says the fused form "does not vectorize". That holds for the generic `value_entry`, not
  for the plain NormL1 expression.
- **Expected value:** low. Only worth trying if `value_and_prox!` of large L1 problems matters to PS.

**4. Short groups on Julia 1.10: a sum with 4 lanes and 1 interleave** (in the spirit of Eigen's packet reduction in
glmnet).
- **The change.** For |G| < 16, sum with one 4-wide accumulator (or 2×2 lanes) instead of the `@simd` loop, which
  1.10 runs scalar for those lengths.
- **Mechanism.** It shortens the dependency chain for groups of 4 to 15 entries. Expect a gain on 1.10 only, for
  group-lasso workloads with small groups. 1.13 already has the epilogue loop.
- **Risk.** The summation order changes the last bits of the norm and hence of t. It must not regress 1.13 or
  groups ≥ 16. R5 forbids branching on the Julia version, so only a length test is acceptable.
- **Expected value:** small to moderate. The per-group fixed costs (threshold product, range tests, sqrt, division,
  exits) may dominate at these sizes.

**5. Avoid `LinearAlgebra.norm` (x87 `nrm2`) on hot paths of the consumers** (PS convergence tests, and PB's
`norm(::SparseIterate)` and `norm(::BlockIterate view)`).
- **Mechanism.** 4 to 5× faster in cache by using PB's own `_l2_value`-style ordinary path (SIMD sum + post-test +
  cold recompute).
- **Risk.** None numerically if the PB kernel is used, since it is already range-safe. PB's own oracles do not call
  `norm` in their kernels, so this is mostly a note for ProximalSolvers.

**6. (Not recommended.)**
- **Blue/Anderson single-pass cold path for norms.** One pass instead of two, but cold-path speed does not matter and
  R13 and R20 forbid it.
- **PO-style refactorization at every γ.** Strictly worse than the SVD when γ changes.
- **Returning values from `prox!`** (PO's API). Extra work.
- **Per-entry divisions** (PO ridge, grpreg). Slower.
- **glmnet and grpreg NaN-to-zero soft threshold.** Silently hides NaN; PB's contract propagates it.

**Own observations** (not suggested by external code; listed for completeness, unranked):
- **Skip the sqrt and division for zeroed groups.** In `_l2_shrink!`, test `s ≤ c²` (one multiply per group, with a
  range guard on c²) before `sqrt` and `c/n`. This saves about 10 cycles of sqrt and division throughput per zeroed
  group, which is most groups in sparse group-lasso. The risk is at the boundary, where `fl(c²)` can zero a group
  that `sqrt(s) > c` would shrink to about eps·x: different bits, numerically benign.
- **Avoid gathers on 1.13 for `Vector{Int}` groups.** Gathers are slow on Broadwell and on GDS-mitigated
  Skylake–Ice Lake parts. A scalar-load 4-accumulator loop might win there. Measure on the cluster CPUs.

---

## 9. Licenses that restrict reuse

- **glmnet (GPL-2) and grpreg (GPL-3)**: algorithms may be described and reimplemented independently, but no code
  may be copied into MIT ProximalBase. All glmnet and grpreg excerpts above are quoted for analysis only.
- **Code may be adopted** with the copyright notice from ProximalOperators.jl, Lasso.jl and Julia (MIT), and from
  OpenBLAS, LAPACK, skglm and celer (BSD-style). Nothing proposed above needs copied code: the clamp form is a
  one-liner, and the cached operator is an algebraic identity.
