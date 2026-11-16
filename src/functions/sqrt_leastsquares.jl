####################################
#
#  Square-root least squares  ‖Y - Xβ‖₂
#
####################################

"""
    SqrtLeastSquaresLoss(Y, X)

The square-root least-squares loss `f(β) = ‖Y - Xβ‖₂` for a vector response `Y` and a design matrix `X`, with no
`1/√n` factor: the normalized square-root lasso objective `(1/√n) ‖Y - Xβ‖₂ + λ ‖β‖₁` has the data term
`Scaled(f, inv(sqrt(n)))`, and a penalty `λ_normalized` on that scale corresponds to `λ_unnormalized = √n λ_normalized`
on the unnormalized loss. PB defines no sum of functions; the objective is written here as a formula.

The loss is convex and is not smooth: it is not differentiable where `Y = Xβ`. It has no gradient and no proximal
operator in this package (`issmooth(f)` and `isproximable(f)` are `false`, `isconvex(f)` is `true`); `gradient!`,
`value_and_gradient!`, `prox!`, `value_and_prox!`, `prox_block`, `kkt_violation` and `restrict` are a `MethodError`
and `lipschitz(f)` is an `ArgumentError`. It stores the same data as [`LeastSquaresLoss`](@ref), which an
algorithm for the square-root loss reads through [`ProximalBase.leastsquares_data`](@ref) (the accessor is shared, and
the two losses are told apart by their types), and it takes the caller's predictor `z = Xβ` or residual through the
value oracles [`ProximalBase.value_from_predictor`](@ref) and [`ProximalBase.value_from_residual`](@ref), for the loss
and for one `Scaled` of it. There is no `ProximalBase.value_and_gradient_from_predictor!` for it.

`Y` is a vector with real entries and `X` a matrix with real entries and `size(X, 1) == length(Y)` (a
`DimensionMismatch` otherwise); a matrix `Y` is a `MethodError` and complex entries are an `ArgumentError`. `Y` and `X`
must have axes that start at 1 (an `ArgumentError` otherwise). `β` of `f(β)` is a vector that `mul!` with `X` accepts: a
dense vector, a `SparseIterate` or a `BlockIterate`.

`f(β)` forms `Xβ` in the scratch `f.tmp` and is not safe to call from two tasks on one `f`. An entry of `Xβ` that
overflows is not formed again: the value is then the norm of the computed `Xβ - Y`, typically `Inf` or `NaN`. The
norm of the computed `Xβ - Y` is evaluated by the kernel of the oracles, not as the square root of a sum of squares:
where the squares overflow or underflow in `T = float(promote_type(eltype(Y), eltype(X)))`, the norm is recomputed in
power-of-two units. The accuracy statement is for finite entries that `T` represents exactly, `n` of them with
`n ≤ 2^40`, and either `(n + 3) eps(T) ≤ 1/16` or strided `Float32` or `Float64` vectors of one element type. An
entry that `T` does not represent (an `Int` `Y` above `maxintfloat(T)`, a `Rational` `Y`) is converted to `T` inside the
sum, and that rounding is outside the bound. With `e = (n/2 + 4) eps(T)` and `v` the exact norm of the computed
`Xβ - Y`, the value in `T` is within `e |v|` of `v`, plus `2 nextfloat(zero(T))` where `|v| < floatmin(T)`. The
product `Xβ` itself carries the rounding of the matrix-vector product. `f(β)` returns its value converted to
`float(eltype(β))`: the bound holds for `eltype(β) == T`, and a narrower `β` adds one rounding to its type. With
`thr = floatmax(T) + ulp/2` the round-to-nearest overflow threshold, a `v` within the relative error `e` of `thr`, below
it as well as above it, gives `Inf` or a finite value within the bound (`floatmax(T)` is then possible), and a `v`
well above `thr` gives `Inf`. `Scaled(f, Inf)(β)` is `Inf * f(β)`, which is `NaN` at an exact fit; the value oracles
refuse the factor `Inf` with an `ArgumentError`.
"""
struct SqrtLeastSquaresLoss{T<:AbstractFloat, Ty<:AbstractVector, Tx<:AbstractMatrix, A<:AbstractVector{T}} <: AbstractFunction
  Y::Ty
  X::Tx
  tmp::A                  # Xβ for f(β), as in LeastSquaresLoss; the value oracles neither read nor write it

  function SqrtLeastSquaresLoss{T, Ty, Tx, A}(Y, X, tmp) where {T<:AbstractFloat, Ty<:AbstractVector,
      Tx<:AbstractMatrix, A<:AbstractVector{T}}
    _check_one_based(Y, X, tmp)
    all(a -> eltype(a) <: Real, (Y, X, tmp)) || _real_error(:SqrtLeastSquaresLoss)
    size(X, 1) == length(Y) || throw(DimensionMismatch("X has $(size(X, 1)) rows and Y has $(length(Y))"))
    axes(tmp) == axes(Y) || throw(DimensionMismatch("tmp has axes $(axes(tmp)) and Y has axes $(axes(Y))"))
    new{T, Ty, Tx, A}(Y, X, tmp)
  end
end
SqrtLeastSquaresLoss(Y::Ty, X::Tx, tmp::A) where {T<:AbstractFloat, Ty<:AbstractVector, Tx<:AbstractMatrix,
    A<:AbstractVector{T}} = SqrtLeastSquaresLoss{T, Ty, Tx, A}(Y, X, tmp)

function SqrtLeastSquaresLoss(Y::AbstractVector, X::AbstractMatrix)
  eltype(Y) <: Real || _real_error(:SqrtLeastSquaresLoss)
  eltype(X) <: Real || _real_error(:SqrtLeastSquaresLoss)
  size(X, 1) == length(Y) || throw(DimensionMismatch("X has $(size(X, 1)) rows and Y has $(length(Y))"))
  T = float(promote_type(eltype(Y), eltype(X)))
  SqrtLeastSquaresLoss(Y, X, _zeros_like(Y, T, size(Y)))
end

issmooth(::Type{<:SqrtLeastSquaresLoss}) = false
isproximable(::Type{<:SqrtLeastSquaresLoss}) = false
isconvex(::Type{<:SqrtLeastSquaresLoss}) = true
_argument_type(::Type{<:SqrtLeastSquaresLoss}) = AbstractVector

function (f::SqrtLeastSquaresLoss{T})(β::AbstractVector) where {T}
  _check_one_based(β)
  _checkreal(f, β)
  mul!(f.tmp, f.X, β)
  _value_at(_sqrt_ls_value(one(T), f.tmp, f.Y, f.Y), β)
end

# The value α ‖x - y‖₂ (α ‖x‖₂ for y === nothing) of f(β) and of the oracles, α a positive finite Float32 or Float64
# (one(T) for the loss), x the predictor Xβ or z, or the residual, and y nothing or the response Y with the axes of x.
# The fourth argument src is what the cold call reads the second operand from: y itself (an array or nothing), or the
# loss or Scaled that the oracle was called with (_sqrt_ls_response). The result has the type T of the sum s of squares:
# the common element type of x and y. The sum is formed in one vectorizing loop up to _LS_ONE_LOOP_MAX entries of a
# strided Float32 or Float64 vector x with y === nothing or a strided y of the same element type, and by Julia's
# pairwise reduction above (the sums of the least-squares oracles, which allocate nothing). An argument that those two
# methods do not take, such as the Float64 scratch and the Float32 response of one loss, takes _sumabs2_diff and _sumabs2.
@inline _sqrt_ls_value(α, x, y, src) = _sqrt_ls_result(α, x, src, _sqrt_ls_sum(x, y))
@inline _sqrt_ls_sum(x, y) = _sumabs2_diff(x, y)
@inline _sqrt_ls_sum(x, ::Nothing) = _sumabs2(x)
@inline function _sqrt_ls_sum(x::StridedVector{T}, y::StridedVector{T}) where {T<:Union{Float32, Float64}}
  axes(x) == axes(y) || _ls_axes_error("the predictor", axes(x, 1), axes(y, 1))
  length(x) > _LS_ONE_LOOP_MAX ? _sqrt_ls_pairwise(x, y) : _ls_simd_sumabs2_diff(x, y)
end
@inline function _sqrt_ls_sum(x::StridedVector{T}, ::Nothing) where {T<:Union{Float32, Float64}}
  length(x) > _LS_ONE_LOOP_MAX ? _sqrt_ls_pairwise(x, nothing) : _simd_sumabs2(x)
end
# the pairwise reductions of the least-squares oracles, out of line so that they do not enlarge the ordinary path
@noinline _sqrt_ls_pairwise(x, y) = _ls_sumabs2_diff_pairwise(x, y)
@noinline _sqrt_ls_pairwise(x, ::Nothing) = sum(abs2, x)

# The ordinary path keeps α √ŝ, with ŝ the computed sum and n = length(x), when ŝ ≥ n floatmin(T)/eps(T)
# (_squares_normal) and the product p = α fl(√ŝ) is below floatmax(T)/2 (_l2_ordinary): the two tests of NormL2, and a
# NaN fails the first. An infinite ŝ gives an infinite p, which fails the second (α is positive and finite), so that a
# kept ŝ is finite. The product is formed in the promoted type P of α and T, and the result is converted to T. Every
# other sum, one that is NaN, overflowed or underflowed included, goes to the one call _sqrt_ls_cold, which recomputes
# the norm from the squares in power-of-two units (_scaled_norm).
#
# Accuracy. Let the entries be finite Float32 or Float64 values of type T, d_i = x_i - y_i (x_i for y === nothing) the
# exact differences, s = Σ d_i², v = α √s the exact value, u = eps(T)/2, e = (n/2 + 4) eps(T) = (n + 8) u,
# γ_m = m u/(1 - m u), and thr = floatmax(T) + ulp(floatmax(T))/2, the round-to-nearest overflow threshold. The factor
# α is a Float32 or Float64, which converts exactly to P and to R = _recovery_type(T, float(typeof(α))). Claim: if
# n ≤ 2^40 and either (n + 3) u ≤ 1/32 or x and y are strided Float32 or Float64 vectors of one element type, then the
# result v̂ satisfies
#   |v̂ - v| ≤ e |v| + 2 nextfloat(zero(T)) [|v| < floatmin(T)],
# v̂ is finite where |v| (1 + e) ≤ floatmax(T), and v̂ is not finite where |v| ≥ thr (1 + e); between the two it is
# finite within the bound or not finite. (A strided pair of one type with n ≤ 2^17 has (n + 3) u ≤ 1/32.)
#
# (1) The ordinary path, with ŝ finite and n floatmin(T) ≤ eps(T) ŝ. A difference of two floats is exact where it is
# below floatmin(T), so each computed difference has relative error ≤ u (it is finite: ŝ is). A square has relative
# error ≤ u, or, where it is below floatmin(T), absolute error ≤ u floatmin(T); these n absolute errors total
# ≤ n u floatmin(T) ≤ 2u² ŝ. Any order of n - 1 additions of nonnegative terms, which covers the loop, its vector
# lanes and the reduction of the lanes, gives each term at most n - 1 factors 1 + ε, |ε| ≤ u (an addition whose result
# is below floatmin(T) is exact). So each term carries at most 2 + 1 + (n - 1) = n + 2 factors, which gives
# ŝ = s (1 + θ₁) + ζ with |θ₁| ≤ γ_{n+2} and |ζ| ≤ 2.1u² ŝ, hence ŝ = s (1 + θ) with |θ| ≤ γ_{n+3}, because
# 2.1u² (1 + γ_{n+2})/(1 - 2.1u²) < u ≤ γ_{n+3} - γ_{n+2}. For a residual one factor fewer suffices. For the pairwise
# reduction a term passes at most 1023 additions inside its block of at most 1024 entries and at most ⌈log2 n⌉ in
# the tree above the blocks (src/functions/leastsquares.jl), so m = 1027 + ⌈log2 n⌉ ≤ 1089 (n ≤ 2^62) replaces n + 3.
# Let t = m u ≤ 1/32. Then |θ| ≤ γ_m ≤ 32t/31, and |√(1 + θ) - 1| ≤ (|θ|/2)(1 + |θ|) ≤ 0.54t. The rounded square
# root, the rounded product and the conversion to T (absent for P = T; for P wider than T the product has a relative
# error below u, and the conversion rounds a value below floatmax(T)/2) are three more factors 1 + ε with |ε| ≤ u.
# Hence v̂ = v (1 + ω) with |ω| ≤ 0.54t + (1 + u)³ - 1 + 0.54t · 3.0001u ≤ 0.54t + 3.1u, and the value before the last
# rounding has |ω_pre| ≤ 0.54t + 2.1u. For m = n + 3 this gives |ω| ≤ 0.54 n u + 4.8u ≤ e, and for m ≤ 1089,
# |ω| ≤ 592u ≤ e (n > 2^17). Where the last rounding falls below floatmin(T) (a small α), its error is at most
# nextfloat(zero(T))/2, which is u floatmin(T) ≤ u |v| if |v| ≥ floatmin(T), and the allowance otherwise.
#
# (2) The cold call, which also covers the sums that (1) does not keep. Where a residual is not finite, _scaled_norm
# returns the ordinary IEEE value convert(R, α) √ŝ: NaN for a NaN entry or an Inf - Inf residual, otherwise Inf.
# Otherwise it forms the differences in R (from the halved entries where a difference overflows in R), scales them
# by 2^-k, 2^k the power of two at or below the largest one, so that the largest scaled difference is in [1, 2) and
# the sum m of their squares is in [1, 4n], and adds the squares by Sum2 of Ogita, Rump and Oishi
# (_compensated_add). With u_R = eps(R)/2 ≤ 2^-53: a difference and a square each have relative error ≤ u_R, the
# compensated sum has relative error ≤ u_R + γ²_{n-1}, and the squares and scaled differences that underflow in R change
# the sum m ≥ 1 by at most n 2^-1073. For n ≤ 2^40, (n u_R)² ≤ 2^-13 n u_R ≤ 2^-26. So the computed m̂ = m (1 + θ_R) has
# |θ_R| ≤ 4.01u_R + 1.01 (n u_R)², then √m̂ has relative error ≤ 2.01u_R + 0.51 (n u_R)² and the rounded root adds u_R, and
# α √m̂ 2^k is formed by _scaled_product with one more rounding of relative error u_R (the power of two is exact). The
# value v_R before that last rounding has |ω_R,pre| ≤ 3.1u_R + 0.0001 n u_R, and v_R has |ω_R| ≤ 4.1u_R + 0.0001 n u_R,
# with an absolute error ≤ nextfloat(zero(R))/2 more where it is below floatmin(R). For R = T = Float64 this is ≤ e. For
# T = Float32 (R = Float64), 4.1u_R + 0.6 · 2^-26 < 0.2u, and converting v_R to T adds a rounding of relative error u (or
# absolute error ≤ nextfloat(zero(T))/2): |ω| ≤ 1.3u ≤ e. A product that overflows in R is Inf, which converts to Inf. An
# exactly zero residual (α positive and finite) gives +0.
#
# (3) The ranges. Let q be the value before the last rounding to nearest to T (the rounded product in (1), the final
# rounding in _scaled_product for R = T, the conversion otherwise); |q - v| ≤ ω_pre |v|. In (1) with (n + 3) u ≤ 1/32
# and e = t + 5u, e - ω_pre - e ω_pre ≥ 0.46t + 2.9u - 0.032 (0.54t + 2.1u) > 0; with m ≤ 1089, e ≥ 2^17 u and
# ω_pre ≤ 592u, e (1 - ω_pre) - ω_pre > 0; in (2), with ω_pre ≤ 3.1u + 0.0001 n u, e (1 - ω_pre) - ω_pre > 0 too. So where |v| (1 + e) ≤
# floatmax(T), q ≤ |v| (1 + ω_pre) ≤ floatmax(T) and the rounded result is finite, and where |v| ≥ thr (1 + e),
# q ≥ |v| (1 - ω_pre) ≥ thr rounds to Inf. A finite result lies within |ω| |v| of v by the bounds above. For n = 0 the
# sum is +0 and so is the value.
@inline function _sqrt_ls_result(α, x, src, s::T) where {T}
  v = α * sqrt(s)
  (_squares_normal(s, length(x), T) & _l2_ordinary(v, T)) && return convert(T, v)
  _sqrt_ls_cold(α, x, src, s)
end

# α ‖x - y‖₂ from the squared-residual kernel in R = _recovery_type(T, float(typeof(α))), converted to T. The second
# operand is src: nothing, the response array itself, or the loss (or its Scaled) whose response Y is loaded here, so that
# the ordinary path passes to the call no array that it loaded from a field of the loss, which would need a GC root in
# the caller.
@noinline function _sqrt_ls_cold(α, x, src, s::T) where {T}
  R = _recovery_type(T, float(typeof(α)))
  convert(T, _scaled_norm(x, _sqrt_ls_response(src), convert(R, s), convert(R, α)))
end
@inline _sqrt_ls_response(src) = src
@inline _sqrt_ls_response(src::SqrtLeastSquaresLoss) = src.Y
