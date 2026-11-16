# Standalone candidate kernels of the study. Nothing here is copied from another library: the variants
# named after a library (po_*, blue_*) are written from the description of its algorithm, to measure the
# strategy next to ProximalBase's. This file is included twice (modules K and KC) so that one kernel has
# an independently compiled A/A control.
using LinearAlgebra

####################################
#  A. Soft thresholding
####################################

# ProximalBase's shrink: Julia's max and min (NaN and signed-zero aware)
@inline st_maxmin(v::T, c::T) where {T} = max(v - c, zero(T)) + min(v + c, zero(T))
# the same sum with the two-operand selects of the x86 max/min instructions: the first operand is returned
# when it is NaN, so a NaN v propagates (v ± c is NaN)
@inline _selmax(a, b) = ifelse(b > a, b, a)
@inline _selmin(a, b) = ifelse(b < a, b, a)
@inline st_selmaxmin(v::T, c::T) where {T} = _selmax(v - c, zero(T)) + _selmin(v + c, zero(T))
# v minus its projection onto [-c, c]
@inline st_clamp(v::T, c::T) where {T} = v - clamp(v, -c, c)
# the same written with selects (it loses a NaN threshold as clamp does: both comparisons are false)
@inline st_clampsel(v::T, c::T) where {T} = v - ifelse(v > c, c, ifelse(v < -c, -c, v))
# v minus its projection with the selects ordered so that a NaN in v or c propagates, plus +0, which turns the -0
# of v = -0 at a zero threshold into +0: the values of st_maxmin for every v and every c that is not negative
@inline function st_clampz(v::T, c::T) where {T}
  lo = -c
  m = ifelse(v > lo, v, lo)
  (v - ifelse(m < c, m, c)) + zero(T)
end
# sign(v) max(|v| - c, 0): gives -0.0 for a negative v below the threshold
@inline st_copysign(v::T, c::T) where {T} = copysign(max(abs(v) - c, zero(T)), v)
@inline st_copysignsel(v::T, c::T) where {T} = copysign(_selmax(abs(v) - c, zero(T)), v)
# three-way select
@inline st_ifelse(v::T, c::T) where {T} = ifelse(v > c, v - c, ifelse(v < -c, v + c, zero(T)))
# the form of ProximalOperators.jl's NormL1 prox loop (a correction added to v)
@inline st_po(v::T, c::T) where {T} = v + (v <= -c ? c : (v >= c ? -c : -v))

# y = shrink(x, c) with a scalar threshold
function l1prox!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, c::T) where {F, T}
  @inbounds @simd for i in eachindex(y, x)
    y[i] = st(x[i], c)
  end
  y
end
# the same without @simd (what LLVM's own vectorizer makes of the plain loop)
function l1prox_plain!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, c::T) where {F, T}
  @inbounds for i in eachindex(y, x)
    y[i] = st(x[i], c)
  end
  y
end
# a threshold per entry: c0 w[i] (weights), with c0 = γλ
function l1prox_w!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, c0::T, w::AbstractArray{T}) where {F, T}
  @inbounds @simd for i in eachindex(y, x, w)
    y[i] = st(x[i], c0 * w[i])
  end
  y
end
# a step per entry and weights: (γ[i] λ) w[i]
function l1prox_gw!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, γ::AbstractArray{T}, λ::T, w::AbstractArray{T}) where {F, T}
  @inbounds @simd for i in eachindex(y, x, w, γ)
    y[i] = st(x[i], (γ[i] * λ) * w[i])
  end
  y
end
function l1prox_g!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, γ::AbstractArray{T}, λ::T) where {F, T}
  @inbounds @simd for i in eachindex(y, x, γ)
    y[i] = st(x[i], γ[i] * λ)
  end
  y
end

# Σ |x[i]|
function sumabs_simd(x::AbstractArray{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(x)
    s += abs(x[i])
  end
  s
end
sumabs_base(x) = sum(abs, x)
sumabs_blas(x) = BLAS.asum(x)
# Σ (λ w[i]) |x[i]|, the coefficient formed first as ProximalBase does
function wsumabs_simd(x::AbstractArray{T}, λ::T, w::AbstractArray{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(x, w)
    s += (λ * w[i]) * abs(x[i])
  end
  s
end

# prox and the value λ Σ |y[i]| at the result in one pass
function l1valprox_fused!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, c::T, λ::T) where {F, T}
  s = zero(T)
  @inbounds @simd for i in eachindex(y, x)
    r = st(x[i], c)
    y[i] = r
    s += abs(r)
  end
  λ * s
end
# two passes: the prox loop, then the value loop
l1valprox_two!(st::F, y, x, c, λ) where {F} = (l1prox!(st, y, x, c); λ * sumabs_simd(y))
function l1valprox_w_fused!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, γ::T, λ::T, w::AbstractArray{T}) where {F, T}
  s = zero(T)
  c0 = γ * λ
  @inbounds @simd for i in eachindex(y, x, w)
    r = st(x[i], c0 * w[i])
    y[i] = r
    s += (λ * w[i]) * abs(r)
  end
  s
end
l1valprox_w_two!(st::F, y, x, γ, λ, w) where {F} = (l1prox_w!(st, y, x, γ * λ, w); wsumabs_simd(y, λ, w))

####################################
#  D. Elastic net: soft thresholding, then the ridge weight
####################################

# prox of γ(λ1‖⋅‖₁ + λ2‖⋅‖²/2): shrink(x, c)·a with c = γλ1 and a = 1/(1 + γλ2) formed once
function enprox!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, c::T, a::T) where {F, T}
  @inbounds @simd for i in eachindex(y, x)
    y[i] = a * st(x[i], c)
  end
  y
end
function enprox_w!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, c0::T, a::T, w::AbstractArray{T}) where {F, T}
  @inbounds @simd for i in eachindex(y, x, w)
    y[i] = a * st(x[i], c0 * w[i])
  end
  y
end
# a step per entry: one division per entry for the ridge weight
function enprox_g!(st::F, y::AbstractArray{T}, x::AbstractArray{T}, γ::AbstractArray{T}, λ1::T, λ2::T) where {F, T}
  @inbounds @simd for i in eachindex(y, x, γ)
    y[i] = st(x[i], γ[i] * λ1) / (one(T) + γ[i] * λ2)
  end
  y
end
# λ1 Σ|xᵢ| + λ2 Σ xᵢ²/2 in one pass
function envalue(x::AbstractArray{T}, λ1::T, λ2::T) where {T}
  s1 = zero(T); s2 = zero(T)
  @inbounds @simd for i in eachindex(x)
    s1 += abs(x[i]); s2 += abs2(x[i])
  end
  λ1 * s1 + λ2 * s2 / 2
end

####################################
#  B. Euclidean norm and its shrinkage
####################################

function sumsq_simd(x::AbstractArray{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(x)
    s += abs2(x[i])
  end
  s
end
function sumsq_plain(x::AbstractArray{T}) where {T}
  s = zero(T)
  @inbounds for i in eachindex(x)
    s += abs2(x[i])
  end
  s
end
norm_simd(x) = sqrt(sumsq_simd(x))
norm_plain(x) = sqrt(sumsq_plain(x))
norm_base(x) = sqrt(sum(abs2, x))
norm_la(x) = norm(x)
norm_blas(x) = BLAS.nrm2(x)
norm_dot(x) = sqrt(BLAS.dot(x, x))

# Two passes with a scale (the strategy of LinearAlgebra.generic_norm2 when the plain sum is not safe, here
# always): the largest magnitude, then the scaled sum of squares
function norm_scaled(x::AbstractArray{T}) where {T}
  m = zero(T)
  @inbounds @simd for i in eachindex(x)
    m = max(m, abs(x[i]))
  end
  (iszero(m) || !isfinite(m)) && return m
  s = zero(T)
  @inbounds @simd for i in eachindex(x)
    s += abs2(x[i] / m)                  # a division per entry: 1/m overflows for a subnormal m
  end
  m * sqrt(s)
end

# One pass with three accumulators (Blue 1978; Anderson 2017, the algorithm of LAPACK 3.10's dnrm2/dlassq):
# big entries are scaled down, small ones scaled up, the others summed as they are. Written from the
# published algorithm, with constants computed for the type (Julia's exponent(floatmin(T)) is one less than Fortran's
# MINEXPONENT, so the scaling of the small accumulator is one binade more conservative than the reference's).
@inline function _blue_constants(::Type{T}) where {T<:Union{Float32, Float64}}
  emin, emax, t = exponent(floatmin(T)), exponent(floatmax(T)) + 1, precision(T)
  tsml = T(2)^cld(emin, 2)                 # below it a square may underflow
  tbig = T(2)^fld(emax - t + 1, 2)         # above it a square may overflow
  ssml = T(2)^(-fld(emin - t, 2))
  sbig = T(2)^(-cld(emax + t - 1, 2))
  (tsml, tbig, ssml, sbig)
end
function norm_blue(x::AbstractArray{T}) where {T<:Union{Float32, Float64}}
  tsml, tbig, ssml, sbig = _blue_constants(T)
  asml = amed = abig = zero(T)
  notbig = true
  @inbounds for i in eachindex(x)
    ax = abs(x[i])
    if ax > tbig
      abig += abs2(ax * sbig); notbig = false
    elseif ax < tsml
      notbig && (asml += abs2(ax * ssml))
    else
      amed += abs2(ax)
    end
  end
  if abig > 0
    (amed > 0 || isnan(amed)) && (abig += (amed * sbig) * sbig)
    return sqrt(abig) / sbig
  elseif asml > 0
    if amed > 0 || isnan(amed)
      amed = sqrt(amed); asml = sqrt(asml) / ssml
      ymin, ymax = minmax(asml, amed)
      return ymax * sqrt(one(T) + abs2(ymin / ymax))
    end
    return sqrt(asml) / ssml
  end
  sqrt(amed)
end

# prox of c‖⋅‖₂ (c = γλ), returning the norm of the result (0 for a group set to zero): the plain sum of
# squares, one sqrt, one division
function l2prox!(y::AbstractArray{T}, x::AbstractArray{T}, c::T) where {T}
  n = sqrt(sumsq_simd(x))
  if n > c
    t = one(T) - c / n
    @inbounds @simd for i in eachindex(y, x)
      y[i] = t * x[i]
    end
    return t * n
  end
  fill!(y, zero(T))
  zero(T)
end
# the strategy of ProximalOperators.jl's NormL2: LinearAlgebra.norm, then a broadcast scale
function l2prox_norm!(y::AbstractArray{T}, x::AbstractArray{T}, c::T) where {T}
  n = norm(x)
  t = max(zero(T), one(T) - c / n)
  y .= t .* x
  t * n
end
function l2prox_nrm2!(y::AbstractArray{T}, x::AbstractArray{T}, c::T) where {T}
  n = BLAS.nrm2(x)
  t = max(zero(T), one(T) - c / n)
  y .= t .* x
  t * n
end

####################################
#  B. Group shrinkage
####################################

@inline function sumsq_idx(x::AbstractArray{T}, G) where {T}
  s = zero(T)
  @inbounds @simd for i in G
    s += abs2(x[i])
  end
  s
end
@inline function sumsq_idx_plain(x::AbstractArray{T}, G) where {T}
  s = zero(T)
  @inbounds for i in G
    s += abs2(x[i])
  end
  s
end
@inline function scale_idx!(y, x, G, t)
  @inbounds @simd for i in G
    y[i] = t * x[i]
  end
  nothing
end
@inline function scale_idx_plain!(y, x, G, t)
  @inbounds for i in G
    y[i] = t * x[i]
  end
  nothing
end
@inline function zero_idx!(y::AbstractArray{T}, G) where {T}
  @inbounds for i in G
    y[i] = zero(T)
  end
  nothing
end

# every group: sum of squares, sqrt, compare, scale or zero. `c0 = γλ`, the threshold of group k is c0 w[k].
function group_prox!(y::AbstractArray{T}, x::AbstractArray{T}, groups, w::Vector{T}, c0::T) where {T}
  @inbounds for k in eachindex(groups)
    G = groups[k]
    c = c0 * w[k]
    n = sqrt(sumsq_idx(x, G))
    if n > c
      scale_idx!(y, x, G, one(T) - c / n)
    else
      zero_idx!(y, G)
    end
  end
  y
end
# without @simd in the group loops
function group_prox_plain!(y::AbstractArray{T}, x::AbstractArray{T}, groups, w::Vector{T}, c0::T) where {T}
  @inbounds for k in eachindex(groups)
    G = groups[k]
    c = c0 * w[k]
    n = sqrt(sumsq_idx_plain(x, G))
    if n > c
      scale_idx_plain!(y, x, G, one(T) - c / n)
    else
      zero_idx!(y, G)
    end
  end
  y
end
# The zero exit before the square root: a sum of squares below c² (1 - 4 eps) has a root below c whatever
# the rounding of c², of the sum and of the root, so the group is zero without sqrt or division.
function group_prox_early!(y::AbstractArray{T}, x::AbstractArray{T}, groups, w::Vector{T}, c0::T) where {T}
  δ = one(T) - 4 * eps(T)
  @inbounds for k in eachindex(groups)
    G = groups[k]
    c = c0 * w[k]
    s = sumsq_idx(x, G)
    if s < (c * c) * δ
      zero_idx!(y, G)
    else
      n = sqrt(s)
      if n > c
        scale_idx!(y, x, G, one(T) - c / n)
      else
        zero_idx!(y, G)
      end
    end
  end
  y
end
# the fused value λ Σ w[k] ‖y[G]‖ with the prox (value at the stored result to a few rounding errors: t n)
function group_valprox!(y::AbstractArray{T}, x::AbstractArray{T}, groups, w::Vector{T}, γ::T, λ::T) where {T}
  v = zero(T)
  c0 = γ * λ
  @inbounds for k in eachindex(groups)
    G = groups[k]
    c = c0 * w[k]
    n = sqrt(sumsq_idx(x, G))
    if n > c
      t = one(T) - c / n
      scale_idx!(y, x, G, t)
      v += (λ * w[k]) * (t * n)
    else
      zero_idx!(y, G)
    end
  end
  v
end
function group_value(x::AbstractArray{T}, groups, w::Vector{T}, λ::T) where {T}
  v = zero(T)
  @inbounds for k in eachindex(groups)
    v += (λ * w[k]) * sqrt(sumsq_idx(x, groups[k]))
  end
  v
end
# groups as the columns of a matrix (the layout of ProximalOperators.jl's NormL21), norms by LinearAlgebra
function group_prox_cols_norm!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, w::Vector{T}, c0::T) where {T}
  @inbounds for k in axes(X, 2)
    xk = view(X, :, k)
    n = norm(xk)
    t = max(zero(T), one(T) - c0 * w[k] / n)
    view(Y, :, k) .= t .* xk
  end
  Y
end

# one compact block, as a block coordinate descent step calls it
@inline function block_prox!(out::AbstractVector{T}, u::AbstractVector{T}, c::T) where {T}
  n = sqrt(sumsq_simd(u))
  if n > c
    t = one(T) - c / n
    @inbounds @simd for i in eachindex(out, u)
      out[i] = t * u[i]
    end
  else
    fill!(out, zero(T))
  end
  out
end

####################################
#  C. Least squares
####################################

# ‖r‖²/2
halfsumsq_simd(r) = sumsq_simd(r) / 2
halfsumsq_base(r) = sum(abs2, r) / 2
halfsumsq_dot(r) = BLAS.dot(r, r) / 2
halfsumsq_nrm2(r) = abs2(BLAS.nrm2(r)) / 2
# ‖z - Y‖²/2
function halfsumsq_diff_simd(z::AbstractVector{T}, Y::AbstractVector{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(z, Y)
    s += abs2(z[i] - Y[i])
  end
  s / 2
end

# value and gradient at β: r = Xβ - Y, g = X'r, value ‖r‖²/2. Three passes over r after the product.
function ls_valgrad!(g::AbstractVector{T}, r::AbstractVector{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}, β::AbstractVector{T}) where {T}
  mul!(r, X, β)
  r .-= Y
  mul!(g, transpose(X), r)
  sumsq_simd(r) / 2
end
# the residual and its sum of squares in one pass
function ls_valgrad_fused!(g::AbstractVector{T}, r::AbstractVector{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}, β::AbstractVector{T}) where {T}
  mul!(r, X, β)
  s = zero(T)
  @inbounds @simd for i in eachindex(r, Y)
    d = r[i] - Y[i]
    r[i] = d
    s += abs2(d)
  end
  mul!(g, transpose(X), r)
  s / 2
end
# the residual by gemv itself: r = Y, then r = X β - r (one pass fewer over r before the second product)
function ls_valgrad_gemv!(g::AbstractVector{T}, r::AbstractVector{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}, β::AbstractVector{T}) where {T}
  copyto!(r, Y)
  mul!(r, X, β, one(T), -one(T))
  mul!(g, transpose(X), r)
  sumsq_simd(r) / 2
end
# from a predictor z held by the caller: d = z - Y, g = X'd
function ls_valgrad_pred!(g::AbstractVector{T}, d::AbstractVector{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}, z::AbstractVector{T}) where {T}
  d .= z .- Y
  mul!(g, transpose(X), d)
  sumsq_simd(d) / 2
end
function ls_valgrad_pred_fused!(g::AbstractVector{T}, d::AbstractVector{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}, z::AbstractVector{T}) where {T}
  s = zero(T)
  @inbounds @simd for i in eachindex(d, z, Y)
    e = z[i] - Y[i]
    d[i] = e
    s += abs2(e)
  end
  mul!(g, transpose(X), d)
  s / 2
end

# The least-squares proximal operator argmin ‖Xβ - Y‖²/2 + ‖β - v‖²/(2γ) by a cached Cholesky factor of
# X'X + I/γ (p ≤ n) or of XX' + I/γ with the Woodbury identity (p > n); refactorized when γ changes. This
# is the strategy of ProximalOperators.jl's LeastSquaresDirect, written from its description.
mutable struct CholLS{T}
  X::Matrix{T}
  XtY::Vector{T}
  G::Matrix{T}            # X'X (thin) or XX' (fat)
  F::Matrix{T}            # the factor's storage
  γ::T
  q::Vector{T}
  t::Vector{T}
  fat::Bool
end
function CholLS(X::Matrix{T}, Y::Vector{T}) where {T}
  n, p = size(X)
  fat = p > n
  G = fat ? X * X' : X' * X
  CholLS{T}(X, X' * Y, G, similar(G), T(NaN), zeros(T, p), zeros(T, fat ? n : 0), fat)
end
function chol_factor!(f::CholLS{T}, γ::T) where {T}
  if γ != f.γ
    copyto!(f.F, f.G)
    ρ = inv(γ)
    @inbounds for i in axes(f.F, 1)
      f.F[i, i] += ρ
    end
    _, info = LAPACK.potrf!('U', f.F)
    info == 0 || throw(LinearAlgebra.PosDefException(info))     # the rounded Gram matrix plus I/γ is not positive definite
    f.γ = γ
  end
  f
end
function chol_prox!(β::Vector{T}, f::CholLS{T}, v::Vector{T}, γ::T) where {T}
  chol_factor!(f, γ)
  ρ = inv(γ)
  q = f.q
  @inbounds @simd for i in eachindex(q, v)
    q[i] = f.XtY[i] + ρ * v[i]
  end
  if f.fat
    # (X'X + ρI)⁻¹ q = (q - X'(XX' + ρI)⁻¹ X q)/ρ
    mul!(f.t, f.X, q)
    LAPACK.potrs!('U', f.F, f.t)
    mul!(β, transpose(f.X), f.t)
    @inbounds @simd for i in eachindex(β, q)
      β[i] = (q[i] - β[i]) * γ
    end
  else
    copyto!(β, q)
    LAPACK.potrs!('U', f.F, β)
  end
  β
end
