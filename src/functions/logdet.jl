####################################
#
#  Gaussian negative log-likelihood  w (tr(SX) - log det X)
#
####################################

"""
    LogDetLoss(S, w = 1)

The Gaussian negative log-likelihood `f(X) = w (tr(SX) - log det X)` for finite
`w > 0`, of a
precision matrix `X`, for a symmetric `S` such as a sample covariance, and
`Inf` when `X` is not positive definite, has a nonfinite entry, or fails the
symmetry check `abs(X[i,j] - X[j,i]) ≤ sqrt(eps(T)) * max(1, abs(X[i,j]),
abs(X[j,i]))`, with `T = eltype(X)`. The loss of the graphical lasso.

It is smooth on the positive definite matrices, with gradient `w(S - X⁻¹)`,
which [`value_and_gradient!`](@ref) computes from one Cholesky factorization;
the value never throws, so that a line search can reject a step that leaves the
cone. It is also proximable: with `μ = γw` and the eigendecomposition
`X - μS = U Diagonal(t) U'`, `prox_{γf}(X) = U Diagonal(z) U'`, where
`zᵢ = (tᵢ + √(tᵢ² + 4μ))/2`, evaluated without overflow or cancellation.
[`value_and_prox!`](@ref) returns the value from the `zᵢ`.
At a finite input whose half spectrum or mapped roots overflow, the output is
computed in units scaled by a power of two. Its error relative to `‖X‖` matches
an ordinary decomposition; components below `eps*‖X‖` are not resolved, so their
log-determinant is not accurate. Entries exceeding `floatmax` within the
reconstruction's rounding bound saturate. The log-determinant uses the mapped
roots and the trace uses the stored output. That output can round to a singular
matrix, so the returned value can be finite even when `f(Y)` is `Inf`.
For a nonsymmetric proximal input, `X` is replaced by its symmetric part:
the skew part contributes only a constant to the proximal objective.
Both use workspaces that `f` owns, sized by `S`, and
allocate nothing; do not share `f` between threads. The workspace has the data's
precision (`Float32` or `Float64`); outputs of any precision are written by rounding,
and values are rounded to the type of the argument (see the Precision section
of the manual).
When an intermediate sum overflows although the weighted value is representable,
the value is computed in scaled units.
"""
struct LogDetLoss{T<:_BlasReal} <: AbstractFunction
  S::Matrix{T}
  w::T
  eig::SymEigenWorkspace{T}     # the proximal operator
  C::Matrix{T}                  # Cholesky factors and inverses
  out::Scratch                  # the Hessian product, for an output BLAS cannot write

  function LogDetLoss{T}(S::Matrix{T}, w::T) where {T<:_BlasReal}
    isfinite(w) && w > 0 || throw(ArgumentError("w must be finite and positive, got $w"))
    p = checksquare(S)
    new{T}(S, w, SymEigenWorkspace(T, p), zeros(T, p, p), Scratch())
  end
end

function LogDetLoss(S::AbstractMatrix, w::Real = 1)
  T = float(promote_type(eltype(S), typeof(w)))
  T <: _BlasReal || throw(ArgumentError("LogDetLoss needs Float32 or Float64 entries, got $T"))
  LogDetLoss{T}(Matrix{T}(Symmetric(S)), convert(T, w))
end

issmooth(::Type{<:LogDetLoss}) = true
_argument_type(::Type{<:LogDetLoss}) = AbstractMatrix
isproximable(::Type{<:LogDetLoss}) = true
isconvex(::Type{<:LogDetLoss}) = true

function _checkwrite(op::Union{typeof(prox!), typeof(value_and_prox!), typeof(gradient!), typeof(value_and_gradient!)}, f::LogDetLoss{T}, Y::AbstractMatrix, X::AbstractMatrix) where {T}
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(Y, X)
  _checksize(Y, X)
  size(X) == size(f.S) || throw(DimensionMismatch("X has size $(size(X)) and S $(size(f.S))"))
  _checkreal(f, X)
  op === value_and_prox! && _checkreal(f, Y)
  true
end

# log det X from the Cholesky factor of X in f.C, or nothing if X is not
# positive definite
function _logdet_chol!(f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  size(X) == size(f.S) || throw(DimensionMismatch("X has size $(size(X)) and S $(size(f.S))"))
  _checkreal(f, X)
  all(isfinite, X) && _issymmetric_tol(X) || return nothing
  C = f.C
  copyto!(C, X)
  _, info = _potrf!('U', C)
  info == 0 || return nothing
  ld = zero(T)
  @inbounds for i in axes(C, 1)
    ld += log(C[i, i])
  end
  2ld
end

function (f::LogDetLoss{T})(X::AbstractMatrix) where {T}
  ld = _logdet_chol!(f, X)
  ld === nothing ? _valtype(X)(Inf) : _value_at(_logdet_value(f, X, ld), X)
end

@inline function _logdet_value(f::LogDetLoss, X::AbstractMatrix, ld)
  v = f.w * (dot(f.S, X) - ld)
  isfinite(v) || !isfinite(ld) || !(all(isfinite, f.S) && all(isfinite, X)) ? v :
    oftype(v, _scaled_logdet_value(f, X, ld))
end

# The products that overflow are summed in units of 2^(ks + kx) and the others
# in their own units, so that huge products that cancel keep the moderate ones
# and the log-determinant; the two parts are combined exactly when the dot is
# representable.
@noinline function _scaled_logdet_value(f::LogDetLoss{T}, X::AbstractMatrix, ld) where {T}
  R = float(promote_type(T, eltype(X)))
  ks = _magnitude_exponent(_maximum_abs(f.S))
  kx = _magnitude_exponent(_maximum_abs(X))
  k = ks + kx
  big, small = _split_dot(f.S, X, _power_factors(R, -ks), _power_factors(R, -kx))
  w, l = convert(R, f.w), convert(R, ld)
  d = ldexp(big, k) + small
  isfinite(d) && return w * (d - l)
  _scaled_product(w, big + _scale_value(small - l, _power_factors(R, -k)), k)
end

function value_and_gradient!(G::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  _checkwrite(value_and_gradient!, f, G, X)
  ld = _logdet_chol!(f, X)
  if ld === nothing
    fill!(G, T(NaN))
    return _valtype(X)(Inf)
  end
  v = _logdet_value(f, X, ld)
  C, S, w = f.C, f.S, f.w
  _potri!('U', C)             # the upper triangle of X⁻¹
  @inbounds for j in axes(C, 2), i in 1:j
    g = w * (S[i, j] - C[i, j])
    G[i, j] = g
    G[j, i] = g
  end
  _value_at(v, X)
end

function value_and_prox!(Y::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix, γ::Real) where {T}
  _checkwrite(value_and_prox!, f, Y, X)
  root_step = sqrt(convert(T, γ)) * sqrt(f.w)
  A = f.eig.A
  # The domain is symmetric: the skew part contributes only a constant
  # to the proximal objective, so use the symmetric part of any input.
  logdet_Y = if root_step > one(T)
    # Half of X/(γw) - S; divide twice to retain the contribution of X even
    # when 1/(γw) underflows. No large step multiplies the covariance.
    A .= (((X ./ 2 .+ transpose(X) ./ 2) ./ root_step) ./ root_step) ./ 2 .- f.S ./ 2
    _logdet_shift_overflow(f, X) && return _scaled_logdet_prox!(Y, f, X, root_step)
    t, U = _eigen!(f.eig)
    _logdet_root_overflow(t, f, X, root_step) && return _scaled_logdet_prox!(Y, f, X, root_step)
    _rebuild_sym_out!(Y, U, t, s -> _logdet_scaled_root_half(s, root_step), f.C; logs = true)
  else
    # Half of X - γwS, without forming a possibly underflowed γw.
    A .= X ./ 4 .+ transpose(X) ./ 4 .- ((f.S ./ 2 .* root_step) .* root_step)
    _logdet_shift_overflow(f, X) && return _scaled_logdet_prox!(Y, f, X, root_step)
    t, U = _eigen!(f.eig)
    _logdet_root_overflow(t, f, X, root_step) && return _scaled_logdet_prox!(Y, f, X, root_step)
    _rebuild_sym_out!(Y, U, t, s -> _logbarrier_root_half(s, root_step), f.C; logs = true)
  end
  _value_at(_logdet_value(f, Y, logdet_Y), Y)
end

# Check before reconstruction, while an aliased input is still intact.
_logdet_finite(f::LogDetLoss{T}, X) where {T} =
  isfinite(convert(T, _maximum_abs(X))) && all(isfinite, f.S)
_logdet_shift_overflow(f, X) = !all(isfinite, f.eig.A) && _logdet_finite(f, X)

function _logdet_root_overflow(t, f::LogDetLoss{T}, X, r) where {T}
  bad = false
  @inbounds for h in t
    z = r > one(T) ? _logdet_scaled_root_half(h, r) : _logbarrier_root_half(h, r)
    bad |= !isfinite(h) || !isfinite(z)
  end
  bad && _logdet_finite(f, X)
end

# Carry each root's exponent separately so that small roots retain their logs.
@inline function _logdet_root_parts(h::T, e::Int, r::T) where {T}
  iszero(h) && return frexp(r)
  a, b = frexp(h)
  c, d = frexp(r)
  # The large-step half matrix was divided by the step.
  if r > one(T)
    a *= c * c
    b += 2d
  end
  b += e
  q = max(b, d)
  x, y = ldexp(a, b - q), ldexp(c, d - q)
  z, k = if h >= 0
    frexp(x + hypot(x, y))
  else
    # Use the unscaled step mantissa even when y underflows.
    frexp(c * (c / (hypot(x, y) - x)))
  end
  z, k + (h >= 0 ? q : 2d - q)
end

@noinline function _scaled_logdet_prox!(Y, f::LogDetLoss{T}, X, r::T) where {T}
  A = f.eig.A
  c, d = frexp(r)
  ex = _magnitude_exponent(convert(T, _maximum_abs(X)))
  es = _magnitude_exponent(_maximum_abs(f.S))
  large = r > one(T)
  e = large ? max(ex - 1 - 2d, es - 1) : max(ex - 1, es - 1 + 2d)
  @inbounds for j in axes(A, 2), i in axes(A, 1)
    x, xt = convert(T, X[i, j]), convert(T, X[j, i])
    if large
      A[i, j] = (ldexp(x, -e - 2 - 2d) / c / c +
                 ldexp(xt, -e - 2 - 2d) / c / c) - ldexp(f.S[i, j], -e - 1)
    else
      A[i, j] = ldexp(x, -e - 2) + ldexp(xt, -e - 2) -
                (ldexp(f.S[i, j], -e - 1 + 2d) * c) * c
    end
  end
  # Normalize the converted half matrix rather than the unshifted input.
  a = _magnitude_exponent(_maximum_abs(A))
  @inbounds for i in eachindex(A)
    A[i] = ldexp(A[i], -a)
  end
  e += a
  t, U = _eigen!(f.eig)
  k = typemin(Int)
  ld = zero(T)
  @inbounds for h in t
    m, b = _logdet_root_parts(h, e, r)
    if m > 0
      ld += log(m) + b * log(T(2))
      k = max(k, b)
    end
  end
  k == typemin(Int) && (k = 0)
  root = let e = e, k = k, r = r
    h -> begin
      m, b = _logdet_root_parts(h, e, r)
      ldexp(m, b - k)
    end
  end
  σ = zero(T)
  @inbounds for h in t
    σ = max(σ, root(h))
  end
  _rebuild_sym!(f.C, U, t, root)
  _spectral_rescale!(Y, f.C, k, size(A, 1), σ)
  _value_at(_logdet_value(f, Y, ld), Y)
end

# The positive root of z² - 2half*z - root_step² = 0. Keeping the square root
# of the step avoids overflow and underflow of the step product itself.
@inline function _logbarrier_root_half(half::T, root_step::T) where {T}
  r = hypot(half, root_step)
  half >= 0 && return half + r
  denominator = r - half
  # The positive root remains finite when half < 0, even if hypot itself
  # overflows. Recompute the half denominator from scaled inputs in that case.
  if isinf(denominator)
    scaled_denominator = hypot(half / 2, root_step / 2) - half / 2
    return (root_step / 2) * (root_step / scaled_denominator)
  end
  root_step * (root_step / denominator)
end

# The same root when half = (X/(γw) - S)/2 and root_step = √(γw) > 1.
@inline function _logdet_scaled_root_half(half::T, root_step::T) where {T}
  r = hypot(half, inv(root_step))
  half >= 0 && return ((half + r) * root_step) * root_step
  denominator = r - half
  isinf(denominator) ? (one(T) / 2) / (r / 2 - half / 2) : inv(denominator)
end
prox!(Y::AbstractMatrix, f::LogDetLoss, X::AbstractMatrix, γ::Real) =
  (_checkwrite(prox!, f, Y, X); value_and_prox!(Y, f, X, γ); Y)

# the gradient S - X⁻¹ is not Lipschitz on the positive definite matrices
lipschitz(::LogDetLoss{T}) where {T} = T(Inf)
