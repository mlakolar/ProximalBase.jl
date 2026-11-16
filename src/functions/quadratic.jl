####################################
#  quadratic
####################################

"""
    QuadraticFunction(A, b = nothing, c = 0)

Quadratic function ``f(x) = x^T A x/2 + b^T x + c``, with gradient `A*x+b`;
omitted `b` is zero. Arguments are real, one-based vectors. `A` must be square and
**symmetric**, and `b` must have matching length. Symmetry is a caller precondition:
the constructor neither checks nor repairs it. Indefinite symmetric `A` is allowed.

The proximal point solves `(I + γ*A)*y = x - γ*b` using a cached eigendecomposition.
For a negative computed minimum eigenvalue, the prox rejects a step whose rounded
`γ*(-λmin)` is not less than one, raising `DomainError` before writing the output.
This uses the effective step after scaling and conversion to the function's type.
Near the spectral boundary, computed and exact positive definiteness can disagree.

# Examples
```jldoctest
julia> f = QuadraticFunction([2.0 0.0; 0.0 4.0], [1.0, -2.0]);

julia> f([1.0, 1.0])
2.0

julia> gradient(f, [1.0, 1.0])
2-element Vector{Float64}:
 3.0
 2.0
```

# Edge cases
- [`isconvex`](@ref) returns `false`: positive semidefiniteness is not certified.
  Solvers that require convexity must validate it, including after data changes.
- `A` and `b` are retained by reference. Call [`reset_cache!`](@ref) after same-shape
  changes; reconstruct after dimension changes. The cached eigenbasis reads `A`'s
  upper triangle, while values and gradients use `A` directly.

See [quadratic operations and step domain](@ref Quadratic-function) for exact
boundary qualifications, caching, alias rules and numerical limits.
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
    _check_one_based(A, b, tmp, Λ, Q, w)
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
isconvex(::Type{<:QuadraticFunction}) = false

function (f::QuadraticFunction)(x::AbstractVector)
  _check_one_based(x)
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
# could have lost more than eps |v| (finite-input value accuracy). m is max|x|, which the product A x has already read.
@inline function _quadratic_checked(f::QuadraticFunction, x, v, ::Type{P}, m) where {P}
  if !isfinite(v) || (_quadratic_underflow_risk(f, x, v, P, m) && !_quadratic_products_normal(f, x, v, P))
    _quadratic_value_recover(f, x, v)
  else
    v
  end
end

# The threshold of _quadratic_underflow_risk is a worst case: it assumes that products may be subnormal.
# When none can be, the ordinary value carries only relative errors, and the value is as accurate as the
# ordinary computation, however small it is, an exact zero of a null-space argument included.
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

# With λ the smallest cached eigenvalue (the decomposition is ascending), the kernel divides by
# 1 + γ Λ. For a finite step γ ≥ 0 and finite cached eigenvalues, the smallest computed denominator is
# 1 - fl(γ (-λ)), which is positive when fl(γ (-λ)) < 1, so the check accepts exactly when every
# computed denominator is positive. A negative λ with a step that fails the comparison (an overflowing
# product and a NaN step included) throws; λ ≥ 0, an empty matrix, and a negative step are not
# checked. For a step γ = Inf and a nonnegative Λ that contains 0, the check accepts and the
# denominators are [NaN, ...]: the step behavior there is unchanged.
@inline function _check_quadratic_step(f::QuadraticFunction, γ)
  isempty(f.Λ) && return nothing
  λ = @inbounds f.Λ[1]
  (λ < 0 && !(γ * -λ < 1)) && _quadratic_step_error(γ, λ)
  nothing
end

@noinline _quadratic_step_error(γ, λ) =
  throw(DomainError(γ, "the proximal operator of a QuadraticFunction needs γ |λmin| < 1 when its smallest " *
                       "eigenvalue λmin is negative (so that 1 + γ λmin > 0); got γ = $γ and λmin = $λ"))

function prox!(y::AbstractVector, f::QuadraticFunction{T}, x::AbstractVector, γ::Real) where {T}
  _checkwrite(prox!, f, y, x)
  _eigen!(f)
  γ = convert(T, γ)
  _check_quadratic_step(f, γ)
  t = f.tmp
  f.b === nothing ? copyto!(t, x) : (t .= x .- γ .* f.b)
  mul!(f.w, transpose(f.Q), t)
  f.w ./= 1 .+ γ .* f.Λ
  mul!(t, f.Q, f.w)
  copyto!(y, t)
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, f::QuadraticFunction{T}, y::AbstractVector, x::AbstractVector) where {T}
  _check_one_based(y, x)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  length(x) == length(f.tmp) || throw(DimensionMismatch())
  true
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::QuadraticFunction, g::AbstractVector, x::AbstractVector)
  _check_one_based(g, x)
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
