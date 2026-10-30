####################################
#
#  Smooth functions that are also proximable
#
####################################

####################################
#  zero
####################################

"""
    ZeroFunction()

The zero function `f(x) = 0`: smooth, with gradient zero, and proximable, with
the identity as proximal operator. The penalty of an unpenalized problem.
"""
struct ZeroFunction <: AbstractFunction end

issmooth(::Type{ZeroFunction}) = true
isproximable(::Type{ZeroFunction}) = true
isseparable(::Type{ZeroFunction}) = true
isconvex(::Type{ZeroFunction}) = true
iszeropreserving(::ZeroFunction) = true
_checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)},
            ::ZeroFunction, y::AbstractArray, x::AbstractArray) = ((op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x); _checksize(y, x); true)

(::ZeroFunction)(x::AbstractArray) = zero(_valtype(x))

function prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  y === x || copyto!(y, x)
  y
end
value_and_prox!(y::AbstractArray, f::ZeroFunction, x::AbstractArray, γ::Real) =
  (_prox_unchecked!(y, f, x, γ); zero(_valtype(y)))

function value_and_gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  fill!(g, zero(eltype(g)))
  zero(_valtype(x))
end
gradient!(g::AbstractArray, f::ZeroFunction, x::AbstractArray) = (_value_and_gradient_unchecked!(g, f, x); g)

prox_entry(::ZeroFunction, v::Real, ::Integer, ::Real) = v
kkt_violation(::ZeroFunction, g::AbstractArray, i::Integer) = abs(g[i])
restrict(f::ZeroFunction, ::AbstractVector{<:Integer}) = f

####################################
#  linear
####################################

"""
    LinearFunction(c, d = 0)

The linear function `f(x) = ⟨c, x⟩ + d`, with gradient `c`. Its proximal operator
is the shift `x - γc`.
"""
struct LinearFunction{C<:AbstractArray, D<:Real} <: AbstractFunction
  c::C
  d::D
end
LinearFunction(c::AbstractArray) = LinearFunction(c, zero(float(eltype(c))))

issmooth(::Type{<:LinearFunction}) = true
isproximable(::Type{<:LinearFunction}) = true
isseparable(::Type{<:LinearFunction}) = true
isconvex(::Type{<:LinearFunction}) = true

(f::LinearFunction)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_unchecked(f, x))
_value_unchecked(f::LinearFunction, x::AbstractArray) = (_checkvalue(f, x); _value_at(dot(f.c, x) + f.d, x))

# Generic dot only needs equal lengths; specialized matrix methods check shapes.
_dot_checks_size(::Type{C}, ::Type{X}) where {C, X} =
  (C <: Diagonal && X <: AbstractMatrix) ||
  (C <: AbstractMatrix && X <: Diagonal) ||
  (C <: StridedMatrix && X <: SymmetricSparseIterate) ||
  (C <: SymmetricSparseIterate && X <: StridedMatrix) ||
  (C <: _SparseDotCSC && X <: Union{_SparseDotCSC, _SparseDotMatrix}) ||
  (C <: _SparseDotMatrix && X <: _SparseDotCSC) ||
  (_SparseWrappedColumnDotChecksSize &&
   ((C <: _SparseDotWrappedCSC && X <: _SparseDotCSC) ||
    (C <: _SparseDotCSC && X <: _SparseDotWrappedCSC)))
_dot_checks_size(::Type{<:Adjoint{<:Union{Real, Complex}, C}}, ::Type{<:Adjoint{<:Union{Real, Complex}, X}}) where {C, X} = _dot_checks_size(C, X)
_dot_checks_size(::Type{<:Transpose{<:Any, C}}, ::Type{<:Transpose{<:Any, X}}) where {C, X} = _dot_checks_size(C, X)
_dot_checks_size(::Type{<:Adjoint{<:Any, C}}, ::Type{<:Transpose{<:Any, X}}) where {C, X} =
  _SparseMixedDotChecksSize && C <: _SparseDotCSC && X <: _SparseDotCSC
_dot_checks_size(::Type{<:Transpose{<:Any, C}}, ::Type{<:Adjoint{<:Any, X}}) where {C, X} =
  _SparseMixedDotChecksSize && C <: _SparseDotCSC && X <: _SparseDotCSC
_checkdot(c::C, x::X) where {C, X} = (_dot_checks_size(C, X) ? _checksize(c, x) : _checkparam(c, x))
_checkvalue(f::LinearFunction, x) = (_checkreal(f, x); _checkdot(f.c, x))

function _checkwrite(op::O, f::LinearFunction, y::AbstractArray, x::AbstractArray) where {O}
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  _checkreal(f, x)
  op === value_and_prox! && (_checkreal(f, y); _checkdot(f.c, y))
  op === value_and_gradient! && _checkdot(f.c, x)
  # c is written into y: copied by the gradient, subtracted by the prox
  _check_parameter_axes(y, f.c)
  true
end

prox!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real) =
  (_check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  if size(f.c) == size(x)
    y .= x .- γ .* _checked_parameter(y, f.c)
  else
    # dot accepts equal-length arrays even when their shapes differ.
    @inbounds for i in eachindex(IndexLinear(), y, x)
      y[i] = x[i] - γ * f.c[i]
    end
  end
  y
end

value_and_gradient!(g::AbstractArray, f::LinearFunction, x::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::AbstractArray, f::LinearFunction, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)
  _copy_parameter!(g, f.c)
  v
end
gradient!(g::AbstractArray, f::LinearFunction, x::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
_gradient_unchecked!(g::AbstractArray, f::LinearFunction, x::AbstractArray) =
  (_checkwrite(gradient!, f, g, x); _copy_parameter!(g, f.c))

Base.@propagate_inbounds prox_entry(f::LinearFunction, v::Real, i::Integer, γ::Real) = v - γ * f.c[i]
_checkparams(f::LinearFunction, x) = _checkparam(f.c, x)
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = abs(g[i] + f.c[i])
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)

####################################
#  squared Euclidean norm
####################################

"""
    SqrNormL2(λ = 1)
    SqrNormL2(λ, y)

The squared Euclidean norm `f(x) = (λ/2) ‖x - y‖²`, with `y = 0` when omitted:
the ridge penalty, and for `λ = 1` the loss `‖x - y‖² / 2`. Smooth, with
gradient `λ(x - y)`, separable, and proximable, with proximal operator
`(x + γλy) / (1 + γλ)`. For a matrix `x`, `‖⋅‖` is the Frobenius norm.
"""
struct SqrNormL2{T<:Real, Y<:Union{Nothing, AbstractArray}} <: AbstractFunction
  λ::T
  y::Y

  function SqrNormL2{T, Y}(λ::T, y::Y) where {T<:Real, Y<:Union{Nothing, AbstractArray}}
    _checknonneg("λ", λ)
    new{T, Y}(λ, y)
  end
end
SqrNormL2(λ::Real = 1, y::Union{Nothing, AbstractArray} = nothing) = SqrNormL2{typeof(λ), typeof(y)}(λ, y)

issmooth(::Type{<:SqrNormL2}) = true
isproximable(::Type{<:SqrNormL2}) = true
isseparable(::Type{<:SqrNormL2}) = true
isconvex(::Type{<:SqrNormL2}) = true
iszeropreserving(f::SqrNormL2) = f.y === nothing && isfinite(f.λ)

# Σ xᵢ² over the entries of x, in O(nnz) for the sparse iterates
_sumabs2(x::AbstractArray) = eltype(x) <: Integer ? sum(v -> abs2(float(v)), x) : sum(abs2, x)
function _sumabs2(x::SparseIterate)
  s = zero(float(eltype(x)))
  @inbounds @simd for k in 1:x.nnz
    s += abs2(float(x.nzval[k]))
  end
  s
end
function _sumabs2(x::SymmetricSparseIterate{T}) where {T}
  s = zero(float(T))
  @inbounds for k = 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    s += (r == c ? 1 : 2) * abs2(float(x.nzval[k]))
  end
  s
end
_sumabs2(x::AtomIterate) = _sumabs2(x.storage)

# Keep the vectorized sum-of-squares path for ordinary magnitudes; use a
# scaled norm when squaring overflows or loses the subnormal entries.
function _norm2(x::AbstractArray)
  eltype(x) <: AbstractFloat || return _norm2_scaled(x)
  s = _sumabs2(x)
  isfinite(s) && s >= floatmin(typeof(s)) && return sqrt(s)
  iszero(s) && x isa DenseArray && iszero(sum(abs, x)) && return s
  _norm2_scaled(x)
end
_norm2_scaled(x::AbstractArray) = norm(x)
_norm2_scaled(x::AtomIterate) = norm(x.storage)
function _norm2_scaled(x::SymmetricSparseIterate{T}) where {T}
  n = zero(float(T))
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(x.p, x.nzval2ind[k])
    isnan(x.nzval[k]) && return oftype(n, NaN)
    n = hypot(n, x.nzval[k])
    r == c || (n = hypot(n, x.nzval[k]))
  end
  n
end

(f::SqrNormL2)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_unchecked(f, x))
function _value_unchecked(f::SqrNormL2, x::AbstractArray)
  _checkvalue(f, x)
  T = _valtype(x)
  iszero(f.λ) && return zero(T)
  s = f.y === nothing ? _sumabs2(x) : _sumabs2_diff(x, f.y)
  _value_at(convert(T, f.λ) * s / 2, x)
end

_checkvalue(f::SqrNormL2, x) = (_checkparams(f, x); f.y === nothing || _checksize(f.y, x); true)

value_and_gradient!(g::AbstractArray, f::SqrNormL2, x::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::AbstractArray, f::SqrNormL2, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  λ = convert(float(eltype(g)), f.λ)
  f.y === nothing ? (g .= λ .* x) : (g .= λ .* (x .- _checked_parameter(g, f.y)))
  _value_unchecked(f, x)
end
gradient!(g::AbstractArray, f::SqrNormL2, x::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::AbstractArray, f::SqrNormL2, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  λ = convert(float(eltype(g)), f.λ)
  f.y === nothing ? (g .= λ .* x) : (g .= λ .* (x .- _checked_parameter(g, f.y)))
end

prox!(z::AbstractArray, f::SqrNormL2, x::AbstractArray, γ::Real) =
  (_check_oracle_arrays(prox!, f, z, x, γ); _prox_unchecked!(z, f, x, γ))
function _prox_unchecked!(z::AbstractArray, f::SqrNormL2, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, z, x)
  T = float(eltype(z))
  a, b = _ridge_weights(T(γ), T(f.λ))
  f.y === nothing ? (z .= a .* x) : (z .= _ridge_blend.(x, _checked_parameter(z, f.y), a, b))
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}, f::SqrNormL2, y::AbstractArray, x::AbstractArray)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  f.y === nothing || (_checksize(f.y, x); _check_parameter_axes(y, f.y))
  true
end

# The convex weights 1/(1+γλ), γλ/(1+γλ), without overflowing γλ or
# the numerator x + γλy of a centered proximal operator.
@inline function _ridge_weights(γ::T, λ::T) where {T}
  t = γ * λ
  if t <= one(T)
    d = one(T) + t
    return inv(d), t / d
  end
  ρ = inv(γ) / λ
  d = one(T) + ρ
  ρ / d, inv(d)
end
# Interpolate same-sign endpoints from the endpoint with larger weight,
# so rounding in two weighted products cannot overflow near floatmax.
@inline function _ridge_blend(x, y, a::T, b::T) where {T}
  x, y = T(x), T(y)
  if signbit(x) == signbit(y)
    return b <= a ? muladd(b, y - x, x) : muladd(a, x - y, y)
  end
  a * x + b * y
end
function prox_entry(f::SqrNormL2{<:Real, Nothing}, v::Real, ::Integer, γ::Real)
  T = typeof(float(v))
  a, _ = _ridge_weights(T(γ), T(f.λ))
  a * v
end
Base.@propagate_inbounds function prox_entry(f::SqrNormL2, v::Real, i::Integer, γ::Real)
  T = typeof(float(v))
  a, b = _ridge_weights(T(γ), T(f.λ))
  _ridge_blend(T(v), T(f.y[i]), a, b)
end
_checkparams(f::SqrNormL2, x) = _checkparam(f.y, x)

kkt_violation(::SqrNormL2{<:Real, Nothing}, g::AbstractArray, i::Integer) = abs(g[i])
kkt_violation(f::SqrNormL2, g::AbstractArray, i::Integer) = abs(g[i] - f.λ * f.y[i])

restrict(f::SqrNormL2{<:Real, Nothing}, ::AbstractVector{<:Integer}) = f
restrict(f::SqrNormL2, S::AbstractVector{<:Integer}) = SqrNormL2(f.λ, f.y[S])

####################################
#  quadratic
####################################

"""
    QuadraticFunction(A, b = nothing, c = 0)

The quadratic `f(x) = x'Ax / 2 + b'x + c` for a symmetric positive semidefinite
`A`, with `b = 0` when omitted. Smooth, with gradient `Ax + b`, and proximable:
the first proximal operator computes the eigendecomposition `A = QΛQ'` once,
after which `prox_{γf}(x) = Q (I + γΛ)⁻¹ Q'(x - γb)` costs two products with
`Q` for any `γ`.
"""
mutable struct QuadraticFunction{T<:AbstractFloat, M<:AbstractMatrix, V<:Union{Nothing, AbstractVector}} <: AbstractFunction
  A::M
  b::V
  c::T
  tmp::Vector{T}        # A x, and scratch for the proximal operator
  Λ::Vector{T}          # A = Q Diagonal(Λ) Q', computed by the first prox
  Q::Matrix{T}
  w::Vector{T}
end

function QuadraticFunction(A::AbstractMatrix, b::Union{Nothing, AbstractVector} = nothing, c::Real = 0)
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
  mul!(f.tmp, f.A, x)
  v = dot(x, f.tmp) / 2 + f.c
  _value_at(f.b === nothing ? v : v + dot(f.b, x), x)
end

function value_and_gradient!(g::AbstractVector, f::QuadraticFunction, x::AbstractVector)
  _checkwrite(value_and_gradient!, f, g, x)
  if eltype(g) !== eltype(x)
    _gradient_unchecked!(g, f, x)
    return _value_unchecked(f, x)
  end
  mul!(g, f.A, x)
  v = dot(g, x) / 2 + f.c
  if f.b !== nothing
    g .+= f.b
    v += dot(f.b, x)
  end
  _value_at(v, x)
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
  f.w ./= 1 .+ γ .* f.Λ
  mul!(t, f.Q, f.w)
  copyto!(y, t)
end

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!)}, f::QuadraticFunction{T}, y::AbstractVector, x::AbstractVector) where {T}
  _checksize(y, x)
  length(x) == length(f.tmp) || throw(DimensionMismatch())
  _checkreal(f, x)
  op === value_and_prox! && _checkreal(f, y)
  true
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::QuadraticFunction, g::AbstractVector, x::AbstractVector)
  _checkgradient_output(g, x)
  _checksize(g, x)
  # Preserve the dimension error of the product, which checks before writing.
  length(x) == size(f.A, 2) || mul!(g, f.A, x)
  _checkreal(f, x)
end

restrict(f::QuadraticFunction, S::AbstractVector{<:Integer}) =
  QuadraticFunction(f.A[S, S], f.b === nothing ? nothing : f.b[S], f.c)

####################################
#  Kronecker quadratic
####################################

"""
    KronQuadratic(A, B, C = nothing)

The quadratic `f(Θ) = ⟨Θ, AΘB⟩ / 2 + ⟨C, Θ⟩` of a `p × q` matrix `Θ`, for
symmetric positive semidefinite `A` (`p × p`) and `B` (`q × q`), with `C = 0`
when omitted: `vec(Θ)'(B ⊗ A) vec(Θ) / 2 + vec(C)'vec(Θ)`, without forming the
`pq × pq` matrix `B ⊗ A`. Smooth, with gradient `AΘB + C`; `size(f)` is
`(p, q)`. `A` and `B` are stored as dense matrices; one that is symmetric only up
to rounding, such as an inverse computed by `inv`, is symmetrized.

`Θ` may be a `p × q` matrix, the vector `vec(Θ)`, a [`SparseIterate`](@ref) of
either shape or, when `p = q`, a [`SymmetricSparseIterate`](@ref). On a
`SymmetricSparseIterate` the gradient is the one over symmetric matrices,
`(AΘB + BΘA + C + C')/2`, so that a solver that steps on one triangle moves
correctly. `C` must be exactly symmetric for a value or second-order oracle at
a symmetric iterate, and for a gradient written into one. The value and the
gradient use buffers that `f` owns and allocate nothing; do not share `f`
between threads.

Examples:
- the direct difference (D-trace) loss `tr(ΣxΔΣyΔ)/2 - tr(Δ(Σy - Σx))` of a
  symmetric `Δ` (CovSel) is `KronQuadratic(Σx, Σy, Σx - Σy)`;
- the loss `tr(ΣxΘΣyΘ')/2 - Θ[a, b]` (DiffPrecTest), whose minimizer is the
  column `(b - 1)p + a` of `(Σy ⊗ Σx)⁻¹`, is `KronQuadratic(Σx, Σy, -E)`, with
  `E` the indicator of the entry `(a, b)`; DiffPrecTest's reduced loss on
  `svec(Θ)` is this function on a symmetric `Θ`, in coordinates that scale the
  off-diagonal entries by `√2`. For a symmetric iterate, use `-(E + E')/2`
  as the linear term.
"""
struct KronQuadratic{T<:AbstractFloat, MC<:Union{Nothing, Matrix{T}}} <: AbstractFunction
  A::Matrix{T}
  B::Matrix{T}
  C::MC
  D::Matrix{T}          # Θ as a dense p × q matrix
  AD::Matrix{T}         # A Θ
  W::Matrix{T}          # A Θ B, and the gradient
end

function KronQuadratic(A::AbstractMatrix, B::AbstractMatrix, C::Union{Nothing, AbstractMatrix} = nothing)
  p, q = checksquare(A), checksquare(B)
  C === nothing || size(C) == (p, q) ||
    throw(DimensionMismatch("A is $p × $p and B is $q × $q, so C must be $p × $q, not $(size(C, 1)) × $(size(C, 2))"))
  T = float(promote_type(eltype(A), eltype(B), C === nothing ? Bool : eltype(C)))
  # dense copies, so that the products never index a Symmetric or sparse matrix
  Ad, Bd = Matrix{T}(A), Matrix{T}(B)
  Cd = C === nothing ? nothing : Matrix{T}(C)
  all(isfinite, Ad) && all(isfinite, Bd) && (Cd === nothing || all(isfinite, Cd)) ||
    throw(ArgumentError("A, B and C must be finite"))
  KronQuadratic{T, typeof(Cd)}(_symmetrized!(Ad, "A"), _symmetrized!(Bd, "B"), Cd, zeros(T, p, q), zeros(T, p, q), zeros(T, p, q))
end

# M, which must be symmetric; one that is symmetric up to rounding, such as an
# inverse computed by inv, becomes (M + M')/2
function _symmetrized!(M::Matrix{T}, name) where {T}
  issymmetric(M) && return M
  δ = maximum(abs(M[i, j] - M[j, i]) for j in axes(M, 2) for i in j+1:size(M, 1))
  δ <= sqrt(eps(T)) * maximum(abs, M) ||
    throw(ArgumentError("$name must be symmetric (one that is symmetric up to rounding is accepted); wrap it in Symmetric to use one of its triangles"))
  for j in axes(M, 2), i in j+1:size(M, 1)
    M[i, j] = M[j, i] = (M[i, j] + M[j, i]) / 2
  end
  M
end

issmooth(::Type{<:KronQuadratic}) = true
isconvex(::Type{<:KronQuadratic}) = true

Base.size(f::KronQuadratic) = size(f.D)

# Θ is vec(Θ) or a p × q matrix
_checkshape(f::KronQuadratic, Θ::AbstractVector) = length(Θ) == length(f.D) ||
  throw(DimensionMismatch("Θ has length $(length(Θ)) and f takes $(size(f.D, 1)) × $(size(f.D, 2)) matrices"))
_checkshape(f::KronQuadratic, Θ::AbstractArray) = size(Θ) == size(f.D) ||
  throw(DimensionMismatch("Θ has size $(size(Θ)) and f takes $(size(f.D, 1)) × $(size(f.D, 2)) matrices"))

# Θ as a dense matrix in f.D
function _densify!(f::KronQuadratic, Θ::AbstractArray)
  _checkshape(f, Θ)
  copyto!(f.D, Θ)
end
function _densify!(f::KronQuadratic, Θ::SparseIterate)
  _checkshape(f, Θ)
  D = fill!(f.D, 0)
  @inbounds for k = 1:nnz(Θ)
    D[Θ.nzval2ind[k]] = Θ.nzval[k]
  end
  D
end
function _densify!(f::KronQuadratic, Θ::SymmetricSparseIterate)
  _checkshape(f, Θ)
  D = fill!(f.D, 0)
  @inbounds for k = 1:nnz(Θ)
    r, c = tril_cartesian(Θ.p, Θ.nzval2ind[k])
    D[r, c] = D[c, r] = Θ.nzval[k]
  end
  D
end

_checkvalue(f::KronQuadratic, Θ) = (_checkreal(f, Θ); _checkshape(f, Θ))

(f::KronQuadratic)(Θ::AbstractArray) =
  (_check_value_arrays(f, Θ); _value_unchecked(f, Θ))
function _value_unchecked(f::KronQuadratic, Θ::AbstractArray)
  _checkreal(f, Θ)
  D = _densify!(f, Θ)
  mul!(f.AD, f.A, D)
  mul!(f.W, f.AD, f.B)
  v = dot(D, f.W) / 2
  _value_at(f.C === nothing ? v : v + dot(f.C, D), Θ)
end

value_and_gradient!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, G, Θ); _value_and_gradient_unchecked!(G, f, Θ))
function _value_and_gradient_unchecked!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray)
  _checkwrite(value_and_gradient!, f, G, Θ)
  D = _densify!(f, Θ)
  W = f.W
  mul!(f.AD, f.A, D)
  mul!(W, f.AD, f.B)
  v = dot(D, W) / 2
  if f.C !== nothing
    v += dot(f.C, D)
    W .+= f.C
  end
  Θ isa SymmetricSparseIterate && _symmetrize!(W)
  copyto!(G, W)
  _value_at(v, Θ)
end

gradient!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, G, Θ); _gradient_unchecked!(G, f, Θ))
_gradient_unchecked!(G::AbstractArray, f::KronQuadratic, Θ::AbstractArray) =
  (_checkwrite(gradient!, f, G, Θ); _value_and_gradient_unchecked!(G, f, Θ); G)

_checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::KronQuadratic, G::AbstractArray, Θ::AbstractArray) =
  (_checkgradient_output(G, Θ); _checkshape(f, G); _checkshape(f, Θ); _checkreal(f, Θ))

# (W + W')/2 in place
function _symmetrize!(W::Matrix)
  p = checksquare(W)
  @inbounds for c = 1:p, r = c+1:p
    W[r, c] = W[c, r] = (W[r, c] + W[c, r]) / 2
  end
  W
end

####################################
#  Lipschitz constants of the gradients
####################################

lipschitz(::ZeroFunction) = 0.0
lipschitz(f::LinearFunction) = zero(float(eltype(f.c)))
lipschitz(f::SqrNormL2) = float(f.λ)
lipschitz(f::QuadraticFunction) = _opnorm_upper(f.A)
lipschitz(f::KronQuadratic) = _opnorm_upper(f.A) * _opnorm_upper(f.B)       # also over symmetric Θ
