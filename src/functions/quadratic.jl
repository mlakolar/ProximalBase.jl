####################################
#  quadratic
####################################

"""
    QuadraticFunction(A, b = nothing, c = 0)

The quadratic `f(x) = x'Ax / 2 + b'x + c` for a symmetric positive semidefinite
`A`, with `b = 0` when omitted. Smooth, with gradient `Ax + b`, and proximable:
the first proximal operator computes the eigendecomposition `A = QΛQ'` once,
after which `prox_{γf}(x) = Q (I + γΛ)⁻¹ Q'(x - γb)` costs two products with
`Q` for any positive finite `γ`. A product `γλ`, `γb` or `x - γb` that overflows
does not change a representable proximal point: those calls recompute it in
power-of-two units (two extra vectors are allocated), and ordinary calls run as before.

The constructor checks real types, squareness and the length of b, but does
not verify symmetry or positive semidefiniteness. Those are caller assumptions.
Values/gradients use A directly; the cached eigenbasis uses Symmetric(Matrix(A))
with its default upper triangle. Use vectors of compatible length. Data are
retained by reference; after same-shape A changes call reset_cache!.
"""
mutable struct QuadraticFunction{T<:AbstractFloat, M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} <: AbstractFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}        # A x, and scratch for the proximal operator
  Λ::Vector{T}          # A = Q Diagonal(Λ) Q', computed by the first prox
  Q::Matrix{T}
  w::Vector{T}

  function QuadraticFunction{T, M, V}(A, b, c, tmp, Λ, Q, w) where {T<:AbstractFloat, M<:AbstractMatrix,
      V<:Union{Nothing, AbstractVector}}
    all(a -> a === nothing || eltype(a) <: Real,
        (A, b, tmp, Λ, Q, w)) || _real_error(:QuadraticFunction)
    new{T, M, V}(A, b, c, tmp, Λ, Q, w)
  end
end
QuadraticFunction(A::M, b::V, c::T, tmp::Vector{T}, Λ::Vector{T}, Q::Matrix{T}, w::Vector{T}) where {T<:AbstractFloat,
    M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} =
  QuadraticFunction{T, M, V}(A, b, c, tmp, Λ, Q, w)

function QuadraticFunction(A::AbstractMatrix, b::Union{Nothing, AbstractVector} = nothing, c::Real = 0)
  eltype(A) <: Real || _real_error(:QuadraticFunction)
  b === nothing || eltype(b) <: Real || _real_error(:QuadraticFunction)
  n = checksquare(A)
  b === nothing || length(b) == n || throw(DimensionMismatch("A is $n × $n and b has length $(length(b))"))
  T = float(promote_type(eltype(A), b === nothing ? eltype(A) : eltype(b), typeof(c)))
  QuadraticFunction{T, typeof(A), typeof(b)}(A, b, T(c), zeros(T, n), T[], Matrix{T}(undef, 0, 0), zeros(T, n))
end

issmooth(::Type{<:QuadraticFunction}) = true
_argument_type(::Type{<:QuadraticFunction}) = AbstractVector
isproximable(::Type{<:QuadraticFunction}) = true
isconvex(::Type{<:QuadraticFunction}) = true

function (f::QuadraticFunction)(x::AbstractVector)
  _checkreal(f, x)
  m = _mul_maxabs!(f.tmp, f.A, x)
  v = dot(x, f.tmp) / 2 + f.c
  v = f.b === nothing ? v : v + dot(f.b, x)
  _value_at(_quadratic_checked(f, x, v, eltype(f.tmp), m), x)
end

function value_and_gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(value_and_gradient!, f, g, x)
  if eltype(g) !== eltype(x)
    _gradient_unchecked!(g, f, x)
    return _value_unchecked(f, x)
  end
  m = _mul_maxabs!(g, f.A, x)
  v = dot(g, x) / 2 + f.c
  if f.b !== nothing
    g .+= f.b
    v += dot(f.b, x)
  end
  _value_at(_quadratic_checked(f, x, v, eltype(g), m), x)
end

# y = A x, and max|x| for the trigger below. For a diagonal A the two are one pass: y[i] = d[i] x[i] is the
# product that mul! rounds into y, so y is the same, and the maximum is read where x[i] is already loaded.
@inline function _mul_maxabs!(y, A, x)
  mul!(y, A, x)
  _stored_maxabs(x)
end
@inline function _mul_maxabs!(y::StridedVector{P}, A::Diagonal{P}, x::StridedVector{P}) where {P<:Union{Float32, Float64}}
  U = _int_type(P)
  d = A.diag
  (length(y) == length(d) == length(x)) || return invoke(_mul_maxabs!, Tuple{Any, Any, Any}, y, A, x)
  m = zero(U)
  @inbounds @simd for i in eachindex(y)
    xi = x[i]
    y[i] = d[i] * xi
    m = max(m, reinterpret(U, xi) & typemax(U))
  end
  reinterpret(P, m)
end

# The ordinary value, or its recomputation from the exact products: after an overflow (a non-finite
# value of finite data), or when the products A x rounded into the buffer type P, and the final dots,
# could have lost more than eps |v| (R1, #520). m is max|x|, which the product A x has already read.
@inline function _quadratic_checked(f::QuadraticFunction, x, v, ::Type{P}, m) where {P}
  if !isfinite(v) || (_quadratic_underflow_risk(f, x, v, P, m) && !_quadratic_products_normal(f, x, v, P))
    _quadratic_value_recover(f, x, v)
  else
    v
  end
end

# The threshold of _quadratic_underflow_risk is a worst case: it assumes that products may be subnormal.
# When none can be, the ordinary value carries only relative errors, and the value is as accurate as the
# ordinary computation (R1), however small it is, an exact zero of a null-space argument included.
# Every nonzero product of two operands is normal if the sum of the exponents of their smallest nonzero
# entries is at least exponent(floatmin): |a b| >= 2^(ea + eb) (Julia's exponent has a significand in
# [1, 2)); a product with a zero is exact; and a sum with a subnormal result is exact. The products are
# those of A x, of x and A x in the dot, and of b and x. The test is made on the operands and on
# A x, which is recomputed into f.tmp (the buffer may hold A x + b, and an ordinary call has not
# modified it; the cost is that of one product, on a path that is rare). It needs the whole computation
# in the type P (a wider argument or buffer is rounded elsewhere). Data need not be tested for being
# finite: a finite value has none that is not (an Inf makes a product Inf or NaN).
@noinline function _quadratic_products_normal(f::QuadraticFunction, x, v, ::Type{P}) where {P}
  (_underflow_checked(P) && eltype(x) === P && eltype(f.tmp) === P && typeof(v) === P &&
   eltype(f.A) <: Union{Float32, Float64} && (f.b === nothing || eltype(f.b) <: Union{Float32, Float64})) ||
    return false
  emin = exponent(floatmin(P))
  ex = _stored_minexp(x)
  _stored_minexp(f.A) + ex >= emin || return false
  f.b === nothing || _stored_minexp(f.b) + ex >= emin || return false
  mul!(f.tmp, f.A, x)
  _min_nonzero_exponent(f.tmp) + ex >= emin
end

# Rounding a product into P changes an entry of A x by at most denorm_min(P)/2 per term, and that
# error is multiplied by an entry of x; the products x[i] (A x)[i] and b[i] x[i] of the dots lose up to
# denorm_min(P)/2 each. |v| < floatmin(P) (n² max|x| / 4 + n) (see _underflow_threshold) is the condition
# under which that can exceed eps |v|. A well-scaled value passes with
# one pass over x. An exact zero of x, or any entry of A, does not fire it.
@inline function _quadratic_underflow_risk(f::QuadraticFunction, x, v, ::Type{P}, m) where {P}
  (_underflow_checked(P) && _underflow_checked(typeof(v))) || return false
  n = length(x)
  iszero(m) && return false
  abs(v) < _underflow_threshold(P, n * n / 4, m, n)
end

# ½ x'Ax + b'x + c summed from the exact products, each a significand and an exponent (_ExpSum), so that
# no A[i,j] x[j] is rounded into a type that can lose it. The cost is that of A x with a few
# bit operations per entry (exact zeros are skipped), and nothing is allocated.
@noinline function _quadratic_value_recover(f::QuadraticFunction, x::AbstractVector, v)
  R = typeof(v)
  _all_finite(x) || return v
  (_all_finite(f.A) && _all_finite(f.b) && isfinite(f.c)) || return v
  acc = _expsum_start(promote_type(R, Float64))
  acc = _expsum_quadratic(acc, f.A, x, -1)
  f.b === nothing || (acc = _expsum_dot(acc, f.b, x, 0))
  acc = _expsum_product(acc, f.c, one(R))
  _expsum_value(R, acc)
end

function gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(gradient!, f, g, x)
  mul!(g, f.A, x)
  f.b === nothing || (g .+= f.b)
  g
end

function _eigen!(f::QuadraticFunction{T}) where {T}
  if isempty(f.Λ)
    E = eigen(Symmetric(Matrix{T}(f.A)))
    f.Λ, f.Q = E.values, E.vectors
  end
  f
end

function prox!(y::AbstractVector, f::QuadraticFunction{T}, x::AbstractVector, γ::Real) where {T}
  _checkwrite(prox!, f, y, x)
  _eigen!(f)
  γ = convert(T, γ)
  t = f.tmp
  f.b === nothing ? copyto!(t, x) : (t .= x .- γ .* f.b)
  mul!(f.w, transpose(f.Q), t)
  if !_quadratic_prox_divide!(f.w, f.Λ, γ) && _quadratic_prox_recoverable(f, x, γ)
    return _quadratic_prox_recover!(y, f, x, γ)
  end
  mul!(t, f.Q, f.w)
  copyto!(y, t)
end

# w ./= 1 .+ γ .* Λ, and whether the ordinary result stands: Σ|wᵢ| and Σ(1 + γλᵢ) are finite. That
# holds only if every entry is (an overflow of γb, x - γb, Q'(x - γb) or γλ makes one Inf or NaN), and
# it bounds every partial sum of the product Q w, whose entries are at most 1 in magnitude. The sums
# vectorize beside the division, so the ordinary path makes no other pass; a sum that overflows on
# finite entries only sends a near-floatmax point to the recomputation, which gives it too.
@inline function _quadratic_prox_divide!(w::Vector{T}, Λ::Vector{T}, γ::T) where {T<:Union{Float32, Float64}}
  sw = sd = zero(T)
  @inbounds @simd for i in eachindex(w, Λ)
    d = 1 + γ * Λ[i]
    wi = w[i] / d
    w[i] = wi
    sw += abs(wi)
    sd += d
  end
  isfinite(sw) & isfinite(sd)
end
function _quadratic_prox_divide!(w, Λ, γ)
  w ./= 1 .+ γ .* Λ
  true
end
# Only finite data and a positive finite step in Float32 or Float64 are recomputed: a genuine NaN or
# Inf, or a step outside the domain, keeps the ordinary result.
@inline _quadratic_prox_recoverable(f::QuadraticFunction{T}, x, γ) where {T} =
  _underflow_checked(T) && isfinite(γ) && γ > 0 && _all_finite(x) && _all_finite(f.b) && _all_finite(f.Λ)

# 1 + γλ as m 2^e with |m| in [1/2, 1), also where γλ overflows (1 is then below its last bit)
@inline function _quadratic_prox_denominator(γ::T, λ::T) where {T}
  s = γ * λ
  isfinite(s) && return _frexp(one(T) + s)
  mγ, eγ = _frexp(γ)
  mλ, eλ = _frexp(λ)
  m, e = _frexp(mγ * mλ)
  (m, eγ + eλ + e)
end
# The proximal point Q z, zᵢ = (wᵢ - γuᵢ)/(1 + γλᵢ) with w = Q'x and u = Q'b, recomputed where an
# intermediate left the range while the point may be representable (R3, as for the ridge proximal
# point; the one-dimensional QuadraticFunction(Diagonal([λ])) is SqrNormL2(λ)). No step scales the
# whole vector by its largest entry, which would round its small entries away:
# - x and b are scaled only as far as keeps their products with Q' finite (|Qᵢⱼ| <= 1), zero
#   unless they are near floatmax, and then by a few binades;
# - each zᵢ is kept as its own significand and exponent: the numerator wᵢ - γuᵢ is summed exactly
#   (_ExpSum), so that a cancellation between its terms is not amplified, and divided once;
# - Q z is the BLAS product in a unit that scales z up (never down), so that its entries are normal
#   numbers and none rounds coarser than its own spacing; where z would have to be scaled down (some
#   zᵢ within a few binades of floatmax or beyond), each entry of Q z is summed from the parts and
#   rounded once instead (O(n²) scalar terms), so that no small entry is rounded away.
# This path allocates two vectors (Q'b and the exponents of z).
@noinline function _quadratic_prox_recover!(y, f::QuadraticFunction{T}, x, γ::T) where {T}
  Q, Λ, w, t, b = f.Q, f.Λ, f.w, f.tmp, f.b
  n = length(w)
  sx = _product_sum_shift(T, _stored_maxabs(x), one(T), n, 2)
  t .= _scale_value.(x, Ref(_power_factors(T, -sx)))
  mul!(w, transpose(Q), t)
  u = zero(w)
  sb = 0
  if b !== nothing
    sb = _product_sum_shift(T, _stored_maxabs(b), one(T), n, 2)
    t .= _scale_value.(b, Ref(_power_factors(T, -sb)))
    mul!(u, transpose(Q), t)
  end
  ez = Vector{Int}(undef, n)
  E = typemin(Int)
  @inbounds for i in eachindex(w)
    md, ed = _quadratic_prox_denominator(γ, Λ[i])
    mw, ew = _frexp(w[i])
    acc = _expsum_add(_expsum_start(T), mw, ew + sx)
    acc = _expsum_product(acc, -γ, u[i], sb)          # γuᵢ from the significands (u may be subnormal)
    mn, en = _frexp(acc.s + acc.c)
    m = mn / md
    w[i], ez[i] = m, acc.ex + en - ed
    iszero(m) || (E = max(E, exponent(m) + ez[i]))
  end
  E == typemin(Int) && return fill!(y, zero(eltype(y)))
  # |zᵢ| < 2^(E+1): in the unit 2^K the largest zᵢ is near the top of the range and the sums of Q z
  # stay below 2^(emax - 2). For K <= 0 (z is scaled up, or not at all) no entry of z rounds coarser
  # than its own spacing, and Q z is the BLAS product of normal numbers, scaled back with one rounding.
  K = E + 4 + _log2_ceil(n) - _max_exponent(T)
  if K <= 0
    @inbounds for i in eachindex(w)
      w[i] = _scaled_product(w[i], one(T), ez[i] - K)
    end
    mul!(t, Q, w)
    return y .= _scaled_product.(t, one(T), K)
  end
  @inbounds for i in eachindex(y)
    acc = _expsum_start(T)
    for j in 1:n
      acc = _expsum_product(acc, Q[i, j], w[j], ez[j])
    end
    y[i] = _expsum_value(T, acc)
  end
  y
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, f::QuadraticFunction{T}, y::AbstractVector, x::AbstractVector) where {T}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  length(x) == length(f.tmp) || throw(DimensionMismatch())
  true
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::QuadraticFunction, g::AbstractVector, x::AbstractVector)
  _checkreal(f, x)
  _checkreal(f, g)
  _checkgradient_output(g, x)
  _checksize(g, x)
  # Preserve the dimension error of the product, which checks before writing.
  length(x) == size(f.A, 2) || mul!(g, f.A, x)
end

restrict(f::QuadraticFunction, S::AbstractVector{<:Integer}) =
  QuadraticFunction(f.A[S, S], f.b === nothing ? nothing : f.b[S], f.c)


lipschitz(f::QuadraticFunction) = _opnorm_upper(f.A)
