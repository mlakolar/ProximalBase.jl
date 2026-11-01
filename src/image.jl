####################################
#
#  Infimal composition with a linear map
#
####################################

struct _OrthogonalImage{T<:Real}
  μ::T
  latent::Scratch
  inner::Scratch
  scaled::Scratch
  residual::Scratch
  wide_latent::Scratch
  wide_inner::Scratch
end

struct _ImageValueCache{T<:AbstractFloat}
  U::Matrix{T}
  V::Matrix{T}
  s::Vector{T}
  N::Matrix{T}
  factor::Cholesky{T, Matrix{T}}
  coefficients::Vector{T}
  nullwork::Vector{T}
end

mutable struct _QuadraticImage{T<:AbstractFloat, M<:AbstractMatrix}
  gram::M
  factor::Any
  γ::T
  valuecache::Union{Nothing, _ImageValueCache{T}}
  latent::Vector{T}
  work::Vector{T}
  ambient::Vector{T}
end

"""
    LinearImage(f, L, μ; check = true)
    LinearImage(f::QuadraticFunction, L)

The image `g(y) = inf{f(x) : Lx = y}` of a function under a linear map `L`,
on real vectors. Its proximal operator is `Lx`, where `x` minimizes
`f(x) + ‖Lx - y‖²/(2γ)`. Two cases are supported:

- With `L'L = μI`, `μ > 0`, and a convex proximable `f`, the latent point is
  `prox_{(γ/μ)f}(L'y/μ)`. A matrix is checked unless `check = false`; an
  operator needs `size`, `eltype`, `mul!` and its adjoint, and is trusted to
  satisfy the identity. The value is `f(L'y/μ)` on the range of `L`, and
  `Inf` outside it, with tolerance `√eps(T) max(1, ‖y‖)` in the floating
  data type `T`. A standalone value requires a standalone value of `f`;
  `value_and_prox!` uses the inner proximal value even when none is available.
- For `f = QuadraticFunction(Q, q, c)` and a matrix `L`, `Q` must be
  symmetric positive semidefinite and positive definite on `null(L)`.
  Symmetry is checked; positive semidefiniteness is assumed. The latent
  point solves `(γQ + L'L)x = L'y - γq`. A Cholesky factorization is cached
  for the last step. A very small or large step can make this matrix
  numerically singular even when the assumptions hold. Pivots at the
  rounding scale of the assembled matrix are rejected: a squared pivot
  must exceed `64n eps(T) maxⱼ |Gⱼⱼ|` for `G = γQ + L'L`. The standalone value
  uses a rank-revealing SVD of `L` and a solve on its null space, cached on
  first use. Singular values at most `max(size(L)...) eps(T) σ₁` are dropped,
  as in [`IndAffine`](@ref); infeasible arguments have value `Inf`.

Quadratic factorizations and buffers use the promoted floating type of `Q`
and `L`; scalar values are rounded to the argument's type, as everywhere in
the package (see the Precision section of the manual). Sparse and structured band
matrices (`Diagonal`, `Bidiagonal`, `Tridiagonal`, `SymTridiagonal`) use sparse
factorizations, which compute in `Float64` on Julia 1.10 and allocate during
solves. Other matrices are converted to dense matrices for factorization.
The standalone quadratic value densifies sparse data. Orthogonal-image
buffers promote the floating types of `L` and the argument.
Outputs keep their element type. An orthogonal image widens its inner workspace
when the transformed step falls outside its normal range; exceptional `BigFloat`
computations may allocate. Dense `prox!` and `value_and_prox!` allocate nothing
after warming the step and buffers in the ordinary range when the inner
oracles allocate nothing.

Convex and proximable, but neither separability nor smoothness is claimed.
Nonfinite arguments have value `Inf`; proximal points propagate `NaN`.
The function owns its buffers; do not share it between threads. Data are
stored by reference. Call [`reset_cache!`](@ref) on the outermost wrapper
after changing `L`, `Q`, `q` or `c`, or construct a new function. Sizes must
stay unchanged.
"""
struct LinearImage{F<:AbstractFunction, M, C} <: AbstractFunction
  f::F
  L::M
  cache::C

  function LinearImage{F, M, C}(f::F, L::M, cache::C) where {F<:AbstractFunction, M, C}
    eltype(L) <: Real || _real_error(:LinearImage)
    new{F, M, C}(f, L, cache)
  end
end
LinearImage(f::F, L::M, cache::C) where {F<:AbstractFunction, M, C} = LinearImage{F, M, C}(f, L, cache)

function _check_image_data(L)
  L isa Real && throw(ArgumentError("L must be a matrix or a linear operator"))
  eltype(L) <: Real || _real_error(:LinearImage)
  L isa AbstractMatrix && !all(isfinite, _stored_values(L)) &&
    throw(ArgumentError("L must have finite data"))
  nothing
end

function LinearImage(f::AbstractFunction, L, μ::Real; check::Bool = true)
  _check_image_data(L)
  isfinite(μ) && μ > 0 || throw(ArgumentError("μ must be finite and positive, got $μ"))
  isconvex(f) && isproximable(f) ||
    throw(ArgumentError("LinearImage with L'L = μI requires a convex proximable function"))
  if check && L isa AbstractMatrix
    T = float(eltype(L))
    # Normalize before forming the Gram matrix, so μ need not fit in T.
    A = T.(L ./ sqrt(μ))
    norm(A' * A - I) <= _tol(T) * max(1, size(L, 2)) ||
      throw(ArgumentError("L'L must be μI; pass check = false to skip this test"))
  end
  cache = _OrthogonalImage(μ, Scratch(), Scratch(), Scratch(), Scratch(), Scratch(), Scratch())
  LinearImage(f, L, cache)
end

_image_sparse(A::AbstractMatrix) = issparse(A) ||
  A isa Union{Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal}
_image_sparse(A::Union{Symmetric, Hermitian}) = _image_sparse(parent(A))

function _check_image_quadratic(f::QuadraticFunction, L)
  n = size(L, 2)
  size(f.A) == (n, n) && (f.b === nothing || length(f.b) == n) ||
    throw(DimensionMismatch("Q and q must match the $(n) columns of L"))
  eltype(f.A) <: Real && (f.b === nothing || eltype(f.b) <: Real) ||
    _real_error(:LinearImage)
  all(isfinite, _stored_values(f.A)) &&
    (f.b === nothing || all(isfinite, f.b)) && isfinite(f.c) ||
    throw(ArgumentError("Q, q and c must be finite"))
  issymmetric(f.A) || throw(ArgumentError("Q must be symmetric positive semidefinite"))
  nothing
end

function LinearImage(f::QuadraticFunction, L::AbstractMatrix)
  _check_image_data(L)
  _check_image_quadratic(f, L)
  T = float(promote_type(eltype(f.A), eltype(L)))
  gram = _image_sparse(f.A) || _image_sparse(L) ? spzeros(T, 0, 0) : zeros(T, 0, 0)
  cache = _QuadraticImage(gram, nothing, T(NaN), nothing,
    zeros(T, size(L, 2)), zeros(T, size(L, 2)), zeros(T, size(L, 1)))
  LinearImage(f, L, cache)
end

function LinearImage(f::AbstractFunction, L)
  L isa AbstractArray && (eltype(L) <: Real || _real_error(:LinearImage))
  throw(ArgumentError(
    "LinearImage supports QuadraticFunction with a matrix L, or a convex proximable f with L'L = μI (supply μ); use the equivalent QuadraticFunction for LeastSquaresLoss"))
end

const _OrthogonalLinearImage = LinearImage{<:AbstractFunction, M, <:_OrthogonalImage} where {M}
const _QuadraticLinearImage = LinearImage{<:QuadraticFunction, <:AbstractMatrix, <:_QuadraticImage}

isproximable(::Type{<:LinearImage}) = true
isconvex(::Type{<:LinearImage{F}}) where {F} = isconvex(F)
_isindicator(::Type{<:LinearImage{F, M, <:_OrthogonalImage}}) where {F, M} = _isindicator(F)

_image_type(h::_OrthogonalLinearImage, x) = float(promote_type(eltype(h.L), eltype(x)))
_image_type(h::LinearImage{F, M, <:_QuadraticImage{T}}, x) where {F, M, T} = T
_image_latent(h::_OrthogonalLinearImage, x) =
  _array_buffer!(h.cache.latent, _image_type(h, x), (size(h.L, 2),))
_image_latent(h::_QuadraticLinearImage, x) = h.cache.latent
_image_inner(h::_OrthogonalLinearImage, x) =
  _array_buffer!(h.cache.inner, _image_type(h, x), (size(h.L, 2),))

function _checkparams(h::LinearImage, x::AbstractArray)
  x isa AbstractVector && length(x) == size(h.L, 1) ||
    throw(DimensionMismatch("LinearImage needs a vector of length $(size(h.L, 1))"))
  z = _image_latent(h, x)
  length(z) == size(h.L, 2) || throw(DimensionMismatch("data sizes changed; construct a new LinearImage"))
  _checkparams(h.f, z)
  if h.f isa QuadraticFunction
    size(h.f.A) == (length(z), length(z)) || throw(DimensionMismatch("Q must match the columns of L"))
    _checkparam(h.f.b, z)
  end
  true
end

function _checkwrite(op::O, h::LinearImage, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(h, y); _checkreal(h, x)
  _checksize(y, x)
  _checkparams(h, x)
  y isa AbstractVector || throw(DimensionMismatch("LinearImage needs a vector output"))
  z = _image_latent(h, x)
  p = h.cache isa _OrthogonalImage ? _image_inner(h, x) : z
  _checkwrite(op, h.f, p, z)
  true
end

# Keep a scale's exponent separate until it meets both multiplicands.
function _image_inverse_scale(v, ::Type{T}) where {T}
  m, e = frexp(float(v))
  convert(T, inv(m)), -e
end
@inline function _image_product(a, b, scale::Tuple{T, Int}) where {T}
  ma, ea = frexp(convert(T, a))
  mb, eb = frexp(convert(T, b))
  ldexp((ma * mb) * scale[1], ea + eb + scale[2])
end

function _image_scaled_dot(a, b, scale::Tuple{T, Int}) where {T}
  v = zero(T)
  @inbounds for i in eachindex(a, b)
    v += _image_product(a[i], b[i], scale)
  end
  v
end

function _image_scaled_mul!(z, A::AbstractMatrix, x, scale)
  @inbounds for i in axes(A, 1)
    v = zero(eltype(z))
    for j in axes(A, 2)
      v += _image_product(A[i, j], x[j], scale)
    end
    z[i] = v
  end
  z
end
function _image_scaled_mul!(z, A, x, scale)
  mul!(z, A, x)
  @inbounds for i in eachindex(z)
    z[i] = _image_product(z[i], one(eltype(z)), scale)
  end
  z
end

function _image_orthogonal_coordinates!(z, h::_OrthogonalLinearImage, x)
  T = eltype(z)
  if h.L isa AbstractMatrix
    _image_scaled_mul!(z, h.L', x, _image_inverse_scale(h.cache.μ, T))
  else
    # Operators expose products only: put one square-root scale on each side.
    scale = _image_inverse_scale(sqrt(h.cache.μ), T)
    scaled = _array_buffer!(h.cache.scaled, T, size(x))
    @inbounds for i in eachindex(scaled, x)
      scaled[i] = _image_product(x[i], one(T), scale)
    end
    _image_scaled_mul!(z, h.L', scaled, scale)
  end
  z
end

# Only the membership comparison is normalized; latent coordinates retain
# small entries that can still make a representable contribution to f.
function _image_in_range!(residual, scaled, L, z, x, ::Type{T}) where {T}
  S = eltype(residual)
  k = max(0, _magnitude_exponent(_maximum_abs(x)))
  scale = (one(S), -k)
  _image_scaled_mul!(residual, L, z, scale)
  @inbounds for i in eachindex(scaled, residual, x)
    scaled[i] = ldexp(convert(S, x[i]), -k)
    residual[i] -= scaled[i]
  end
  _norm2(residual) <= _tol(T) * max(ldexp(one(S), -k), _norm2(scaled))
end

function (h::_OrthogonalLinearImage)(x::AbstractArray)
  _checkreal(h, x)
  _checkparams(h, x)
  T = _image_type(h, x)
  isfinite(_maximum_abs(x)) || return _valtype(x)(Inf)
  z = _image_orthogonal_coordinates!(_image_latent(h, x), h, x)
  scaled = _array_buffer!(h.cache.scaled, T, size(x))
  residual = _array_buffer!(h.cache.residual, T, size(x))
  _image_in_range!(residual, scaled, h.L, z, x, float(eltype(h.L))) || return _valtype(x)(Inf)
  _value_at(h.f(z), x)
end

function _image_prox!(y, h::_OrthogonalLinearImage, x, γ, withvalue::Bool)
  T = _image_type(h, x)
  step = γ / h.cache.μ
  # Inner oracles commonly convert the step to their workspace's type.
  # Widen exceptional steps before that conversion combines them with data.
  if !(floatmin(T) <= step <= floatmax(T))
    W = T === Float32 && floatmin(Float64) <= step <= floatmax(Float64) ? Float64 : BigFloat
    return convert(T, _image_wide_prox!(y, h, x, γ, W, withvalue))
  end
  z = _image_orthogonal_coordinates!(_image_latent(h, x), h, x)
  p = _image_inner(h, x)
  v = withvalue ? value_and_prox!(p, h.f, z, step) : (prox!(p, h.f, z, step); zero(eltype(z)))
  mul!(y, h.L, p)
  convert(eltype(z), v)
end

function _image_wide_prox!(y, h, x, γ, ::Type{T}, withvalue) where {T}
  z = _array_buffer!(h.cache.wide_latent, T, (size(h.L, 2),))
  p = _array_buffer!(h.cache.wide_inner, T, size(z))
  _image_orthogonal_coordinates!(z, h, x)
  step = T(γ) / T(h.cache.μ)
  v = _image_wide_inner!(p, h.f, z, step, withvalue)
  mul!(y, h.L, p)
  v
end

_image_wide_inner!(p, f, z, step, withvalue) =
  withvalue ? value_and_prox!(p, f, z, step) : (prox!(p, f, z, step); zero(eltype(p)))

function _image_wide_inner!(p, f::QuadraticFunction, z, step, withvalue)
  # This inner oracle owns caches in its data's precision, so widening its
  # destination alone would still narrow the transformed step.
  T = eltype(p)
  A = Matrix{T}(f.A)
  G = step .* A
  @inbounds for i in axes(G, 1)
    G[i, i] += one(T)
  end
  f.b === nothing ? copyto!(p, z) : (p .= z .- step .* f.b)
  ldiv!(cholesky!(Symmetric(G)), p)
  withvalue || return zero(T)
  mul!(z, A, p)
  v = _image_scaled_dot(p, z, (T(0.5), 0)) + T(f.c)
  f.b === nothing ? v : v + dot(f.b, p)
end

# A successful Cholesky can retain a rounding-level pivot of a singular matrix.
_image_factor_diagonal(factor::Cholesky) = diag(factor.U)
_image_factor_diagonal(factor) = diag(factor)
function _image_diagonal_scale(A)
  scale = zero(float(eltype(A)))
  @inbounds for i in axes(A, 1)
    scale = max(scale, abs(A[i, i]))
  end
  scale
end
function _image_factor_ok(factor, ::Type{T}, scale, n) where {T}
  issuccess(factor) || return false
  d = _image_factor_diagonal(factor)
  isempty(d) && return true
  isfinite(scale) && scale > 0 && all(isfinite, d) || return false
  # Gram/projection products and Cholesky accumulate errors on the n*eps
  # scale. A safety factor of 64 allows for their combined rounding variation,
  # rather than treating a backend's rounding-level positive pivot as rank.
  # Compare square roots in relative units to avoid overflow or underflow.
  minimum(d) / sqrt(scale) > sqrt(T(64) * max(1, n) * eps(T))
end

function _image_factor!(h::_QuadraticLinearImage, γ)
  cache = h.cache
  T = eltype(cache.latent)
  step = convert(T, γ)
  message = "LinearImage requires Q positive semidefinite and positive definite on null(L); the step may also be too small or large for γQ + L'L to be numerically positive definite"
  isfinite(step) && step > 0 || throw(ArgumentError(message))
  cache.factor !== nothing && cache.γ == step && return cache
  M = typeof(cache.gram)
  if cache.factor === nothing
    A = convert(M, h.L)
    cache.gram = A' * A
  end
  A = step * convert(M, h.f.A) + cache.gram
  all(isfinite, _stored_values(A)) || throw(ArgumentError(message))
  factor = try
    cholesky(Symmetric(A); check = false)
  catch e
    e isa Union{PosDefException, ZeroPivotException, LAPACKException} || rethrow()
    throw(ArgumentError(message))
  end
  _image_factor_ok(factor, T, _image_diagonal_scale(A), size(A, 1)) || throw(ArgumentError(message))
  cache.factor, cache.γ = factor, step
  cache
end

# The dense factor's concrete type keeps the hot solve free of allocations.
_image_solve!(c::_QuadraticImage{T, Matrix{T}}) where {T} =
  ldiv!(c.factor::Cholesky{T, Matrix{T}}, c.latent)
_image_solve!(c::_QuadraticImage) = copyto!(c.latent, c.factor \ c.latent)

function _image_quadratic_value(h::_QuadraticLinearImage, z)
  c = h.cache
  T = eltype(z)
  mul!(c.work, h.f.A, z)
  v = _image_scaled_dot(z, c.work, (T(0.5), 0)) + convert(T, h.f.c)
  h.f.b === nothing ? v : v + convert(T, dot(h.f.b, z))
end

function _image_prox!(y, h::_QuadraticLinearImage, x, γ, withvalue::Bool)
  c = _image_factor!(h, γ)
  z = mul!(c.latent, h.L', x)
  h.f.b === nothing || (z .-= c.γ .* h.f.b)
  k = all(isfinite, z) ? 0 : _image_scaled_rhs!(h, x)
  _image_solve!(c)
  if !iszero(k)
    @inbounds for i in eachindex(z)
      z[i] = ldexp(z[i], k)
    end
  end
  mul!(y, h.L, z)
  withvalue ? _image_quadratic_value(h, z) : zero(eltype(z))
end

function _image_scaled_rhs!(h::_QuadraticLinearImage, x)
  c = h.cache
  T = eltype(c.latent)
  ml, mx = _maximum_abs(h.L), _maximum_abs(x)
  eq = h.f.b === nothing ? 0 :
    _magnitude_exponent(c.γ) + _magnitude_exponent(_maximum_abs(h.f.b)) + 2
  e = max(eq, _magnitude_exponent(ml) + _magnitude_exponent(mx) + 2 +
    ndigits(max(1, length(x)); base = 2))
  k = max(0, e - exponent(floatmax(T)) + 2)
  scale = (one(T), -k)
  _image_scaled_mul!(c.latent, h.L', x, scale)
  if h.f.b !== nothing
    @inbounds for i in eachindex(c.latent, h.f.b)
      c.latent[i] -= _image_product(c.γ, h.f.b[i], scale)
    end
  end
  k
end

function prox!(y::AbstractArray, h::LinearImage, x::AbstractArray, γ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, h, y, x)
  isfinite(γ) && γ > 0 || throw(ArgumentError("γ must be finite and positive"))
  if isfinite(_maximum_abs(x))
    _image_prox!(y, h, x, γ, false)
  else
    fill!(y, NaN)
  end
  y
end

function value_and_prox!(y::AbstractArray, h::LinearImage, x::AbstractArray, γ::Real)
  x = _symmetric_argument(y, x)
  _checkwrite(value_and_prox!, h, y, x)
  isfinite(γ) && γ > 0 || throw(ArgumentError("γ must be finite and positive"))
  if !isfinite(_maximum_abs(x))
    fill!(y, NaN)
    return _valtype(y)(NaN)
  end
  _value_at(_image_prox!(y, h, x, γ, true), y)
end

function _image_value_cache!(h::_QuadraticLinearImage)
  c = h.cache
  c.valuecache === nothing || return c.valuecache
  T = eltype(c.latent)
  m, n = size(h.L)
  F = svd(Matrix{T}(h.L); full = m < n)
  tol = max(m, n) * eps(T) * (isempty(F.S) ? zero(T) : F.S[1])
  k = count(>(tol), F.S)
  U, V, N = F.U[:, 1:k], F.V[:, 1:k], F.V[:, (k + 1):n]
  A = N' * Matrix{T}(h.f.A) * N
  # Projection can leave rounding noise on an exact null direction; retain
  # Q's scale even when the entire reduced matrix is at that noise level.
  scale = max(_image_diagonal_scale(h.f.A), _image_diagonal_scale(A))
  R = cholesky(Symmetric(A); check = false)
  _image_factor_ok(R, T, scale, n) ||
    throw(ArgumentError("Q must be positive definite on null(L) numerically"))
  c.valuecache = _ImageValueCache(U, V, F.S[1:k], N, R, zeros(T, k), zeros(T, n - k))
end

function (h::_QuadraticLinearImage)(x::AbstractArray)
  _checkreal(h, x)
  _checkparams(h, x)
  c = h.cache
  T = eltype(c.latent)
  isfinite(_maximum_abs(x)) || return _valtype(x)(Inf)
  v = _image_value_cache!(h)
  @inbounds for j in eachindex(v.coefficients, v.s)
    scale = _image_inverse_scale(v.s[j], T)
    a = zero(T)
    for i in eachindex(x)
      a += _image_product(v.U[i, j], x[i], scale)
    end
    v.coefficients[j] = a
  end
  mul!(c.latent, v.V, v.coefficients)
  # Feasibility is checked independently of the constrained minimization.
  k = max(0, _magnitude_exponent(_maximum_abs(x)))
  _image_scaled_mul!(c.ambient, h.L, c.latent, (one(T), -k))
  nx = zero(T)
  @inbounds for i in eachindex(x, c.ambient)
    a = ldexp(convert(T, x[i]), -k)
    nx = hypot(nx, a)
    c.ambient[i] -= a
  end
  _norm2(c.ambient) <= _tol(T) * max(ldexp(one(T), -k), nx) || return _valtype(x)(Inf)
  if !isempty(v.nullwork)
    mul!(c.work, h.f.A, c.latent)
    h.f.b === nothing || (c.work .+= h.f.b)
    mul!(v.nullwork, v.N', c.work)
    ldiv!(v.factor, v.nullwork)
    mul!(c.latent, v.N, v.nullwork, -one(T), one(T))
  end
  _value_at(_image_quadratic_value(h, c.latent), x)
end

function _checkstep(op::O, h::LinearImage, y::AbstractArray, x::AbstractArray, γ::Real) where {O}
  isfinite(γ) && γ > 0 || throw(ArgumentError("γ must be finite and positive"))
  true
end
