####################################
#
#  Second-order oracles: Hessian-vector products and quadratic models
#
####################################

"""
    hashessian(f)

Whether the smooth function `f` provides second-order oracles:
[`hessian_vec!`](@ref), the product of its Hessian with a direction, and
[`quadratic_model`](@ref), its second-order model at a point as another
function. Newton–CG, trust-region and proximal Newton methods need them; a
solver checks this trait as it checks [`issmooth`](@ref).
"""
hashessian(::Type{<:AbstractFunction}) = false
hashessian(f::AbstractFunction) = hashessian(typeof(f))

"""
    hessian_vec!(Hv, f, x, v)

Store the product `∇²f(x) v` of the Hessian of `f` at `x` with the direction
`v` in `Hv` and return `Hv`, for a function with [`hashessian`](@ref). `Hv`
must not overlap `x` or `v`, as for `mul!`. Built-in methods, the wrappers
around any function included, throw an `ArgumentError` before writing when
`Hv` is `x` or `v`, or a block of `Hv` is the corresponding block of `x` or
`v`. Other sharing of memory is not supported and need not be detected. `x`
and `v` may be the same array. `Hv` is a dense array of the shape of `x`; `x`
and `v` may be sparse iterates. For a [`SeparableSum`](@ref) or a [`Sum`](@ref)
of such functions, `x`, `v` and `Hv` are tuples of blocks with the same
structure. The product costs about as much as a gradient: two products with `X`
for [`LeastSquaresLoss`](@ref) and [`GLMLoss`](@ref), `w X⁻¹VX⁻¹` from one
Cholesky factorization for [`LogDetLoss`](@ref) (`NaN`s when `x` is not
positive definite), and it allocates nothing. On a
[`SymmetricSparseIterate`](@ref) the product of [`KronQuadratic`](@ref) is the
one over symmetric matrices, as its gradient is. Several products at the same
`x` are cheaper through [`quadratic_model`](@ref), which does the work that
depends on `x` once.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
function hessian_vec! end

"""
    hessian_vec(f, x, v)

Allocating version of [`hessian_vec!`](@ref).
"""
hessian_vec(f::AbstractFunction, x::AbstractArray, v::AbstractArray) = hessian_vec!(_gradient_buffer(x), f, x, v)
hessian_vec(f::AbstractFunction, x::Tuple, v::Tuple) = hessian_vec!(_gradient_buffer(x), f, x, v)

"""
    quadratic_model(f, x)
    quadratic_model!(q, f, x)

The second-order model of the smooth function `f` at `x`, as a smooth function
`q` of ProximalBase whose gradient at `x` is `∇f(x)` and whose Hessian is
`∇²f(x)`:

    q(z) = f(x) + ⟨∇f(x), z - x⟩ + ⟨z - x, ∇²f(x)(z - x)⟩ / 2   (up to a constant)

For `GLMLoss`, the curvature floor described below makes this a regularized
curvature model where the floor is active, rather than the exact Taylor
Hessian. It retains the gradient at the model point.

A proximal Newton method minimizes `q + g` with any solver and penalty `g`,
then searches along the step on `f + g`. `quadratic_model!` updates a model
returned by `quadratic_model(f, x₀)` to the model at `x`, in place and without
allocating for finite data (see the `GLMLoss` case below), and returns it. A model of a [`SeparableSum`](@ref) must have as
many blocks as `f`, and a model of a [`Sum`](@ref) as many parts; otherwise a
`DimensionMismatch` is thrown. Each sum is checked when the update reaches it,
before any of its own blocks change; a mismatch in a nested sum can therefore
be found after earlier sibling blocks were updated.

| `f` | `q` |
|:----|:----|
| [`QuadraticFunction`](@ref), [`KronQuadratic`](@ref), [`LeastSquaresLoss`](@ref), [`SqrNormL2`](@ref), [`LinearFunction`](@ref), [`ZeroFunction`](@ref) | `f` itself |
| [`GLMLoss`](@ref) | the weighted least-squares loss of IRLS, `GLMLoss(ProximalBase.Squared(), z, X; weights = w ⊙ ℓ'', offset)` with the working responses `zᵢ = ηᵢ - ℓ'ᵢ / ℓ''ᵢ` |
| [`LogDetLoss`](@ref)`(S, w)` | `KronQuadratic(wW, W, w(S - 2W))` with `W = x⁻¹` |
| [`Scaled`](@ref), [`Tilt`](@ref), [`Translate`](@ref), [`Sum`](@ref), [`SeparableSum`](@ref), [`Regularize`](@ref), [`Precompose`](@ref) | the same construction around the models of the parts |

For a `GLMLoss`, `ℓ''ᵢ` is floored at `√eps`, as in IRLS, so that the working
responses stay finite where the curvature vanishes (a Huber loss beyond `δ`);
the gradient of `q` at `x` is still `∇f(x)`, up to the rounding of `ηᵢ - zᵢ` (which loses `ℓ'ᵢ/ℓ''ᵢ`
when `|ηᵢ|` exceeds it by more than `1/eps`, as for a Poisson loss at `η = -1e30`). The coefficients `wᵢℓ''ᵢ`
are the *effective* ones: a Poisson observation whose `exp(ηᵢ)` overflows still
has a finite coefficient in `Float32` or `Float64` when `wᵢ exp(ηᵢ)` is representable (weight `1/8` at
`η = 710` gives `exp(710)/8` and the working response `709`), and a coefficient
or working response of a finite observation that is not representable in the
working precision (or a nonzero weight whose coefficient rounds to zero) throws
a [`PrecisionError`](@ref), which is never clamped. `quadratic_model!` computes
the whole model before it writes, so after a `PrecisionError` every field of `q`
is as it was and `q` can be updated again. `q` must own its arrays, as `quadratic_model`
makes it: `q.y` and `q.weights` must be two separate arrays, and none of the arrays of `f` (`y`,
the weights, the offset and its buffers); an identical array is an `ArgumentError`, a view or
reshape of one is not detected. Non-finite data propagate as
`NaN`/`Inf` without an error in the update of a model; `quadratic_model` builds a new
loss and so checks the offset again, which must be finite, as in the constructor: an
offset made non-finite after the construction is an `ArgumentError` there. Allocation
and global state, in four cases. (i) An ordinary update in `Float32` or `Float64` allocates nothing as
long as the staged coefficients and responses are finite and no coefficient vanished.
(ii) A recovery that does not need the certified tier below (the exponent split of a Poisson
`exp(ηᵢ)` that overflows, a response below the curvature floor, the correction of a
subnormal coefficient) allocates nothing either. A response below the curvature floor whose
terms leave the range (`|ηᵢ|` and `yᵢ/√eps` near `floatmax`) is formed in double-word
arithmetic and rounded once, for `Float32` and `Float64` responses; a response of another
type (integer, rational) is first converted to the working float type, as `e - y` is. (iii) A
Poisson coefficient `wᵢ exp(ηᵢ)` within about 0.1% of `floatmax` or past it, but not clearly
past it, is rounded once, correctly, by a certified computation in `BigFloat`
at an explicit precision (the result does not depend on the current `BigFloat` precision),
and allocates; it is `Inf`, and so a [`PrecisionError`](@ref), exactly when the exact value
reaches `floatmax + ulp/2`. Otherwise (a non-finite observation, a predictor that overflowed
in `Xβ + offset` even for finite `X`, `β` and offset, or a vanished coefficient) the update
checks the staged values in a new vector of the length of `y`, and allocates, whether
it then throws or returns. A `BigFloat` response is computed in the current `BigFloat`
precision and allocates. (iv) No production path reads or changes the process-global `BigFloat`
precision or rounding mode, except that the arithmetic of a `BigFloat` response uses the current
precision, as above. The all-or-nothing update holds for one `GLMLoss`:
the models of a [`Sum`](@ref) or [`SeparableSum`](@ref) of such losses are
updated one part after another. For `LogDetLoss`, `q` is
written in the new point `z` rather than in the step `z - x`: minimizing
`q(z) + λ‖z‖₁` over symmetric `z` is the Newton step of the QUIC algorithm for
the graphical lasso (Hsieh, Sustik, Dhillon and Ravikumar, 2014), and
coordinate descent on a [`SymmetricSparseIterate`](@ref) solves it. `x` must
then be positive definite.

See [Symmetric parameters](@ref) for which arrays must be symmetric and which calls check them.
"""
function quadratic_model end
function quadratic_model! end
@doc (@doc quadratic_model) quadratic_model!

@noinline _no_hessian(f) =
  throw(ArgumentError(issmooth(f) ? "no Hessian is known for $(nameof(typeof(f)))" :
                                    "$(nameof(typeof(f))) is not smooth"))

hessian_vec!(Hv, f::AbstractFunction, x, v) =
  (_check_second_order_arrays(f, x, Hv, v); _no_hessian(f))
_hessian_vec_unchecked!(Hv, f::AbstractFunction, x, v) =
  _isbuiltin(f) ? _no_hessian(f) : hessian_vec!(Hv, f, x, v)
quadratic_model(f::AbstractFunction, x) =
  (_check_second_order_arrays(f, x); _no_hessian(f))
_quadratic_model_unchecked(f::AbstractFunction, x) =
  _isbuiltin(f) ? _no_hessian(f) : quadratic_model(f, x)
quadratic_model!(q, f::AbstractFunction, x) =
  (_check_second_order_arrays(f, x); _no_hessian(f))
_quadratic_model_unchecked!(q, f::AbstractFunction, x) =
  _isbuiltin(f) ? _no_hessian(f) : quadratic_model!(q, f, x)

_checkdirection(v, x) = size(v) == size(x) ||
  throw(DimensionMismatch("the direction has size $(size(v)) and the point $(size(x))"))
function _checkdirection(v::Tuple, x::Tuple)
  length(v) == length(x) || throw(DimensionMismatch("the direction and the point must have the same number of blocks"))
  _checkdirection_blocks(v, x)
end

# Hv must not be x or v; blocks pair up to the shortest tuple, and the size checks report a
# length mismatch. Tuples are never compared whole: () === () for any two empty tuples.
_checkhessian_output(Hv, x, v) =
  (Hv === x && throw(ArgumentError("the Hessian product must not overwrite x"));
   Hv === v && throw(ArgumentError("the Hessian product must not overwrite v")); nothing)
@generated _checkhessian_output(Hv::Tuple, x::Tuple, v::Tuple) =
  _blockwise(k -> :(_checkhessian_output(Hv[$k], x[$k], v[$k])), min(fieldcount(Hv), fieldcount(x), fieldcount(v)), nothing)

@generated _checkdirection_blocks(v::Tuple, x::Tuple) =
  _blockwise(k -> :(_checkdirection(v[$k], x[$k])), min(fieldcount(v), fieldcount(x)), true)

####################################
#  quadratics: the model is the function
####################################

for F in (:ZeroFunction, :LinearFunction, :SqrNormL2, :QuadraticFunction, :KronQuadratic, :LeastSquaresLoss)
  @eval begin
    hashessian(::Type{<:$F}) = true
    quadratic_model(f::$F, x) =
      (_check_second_order_arrays(f, x); _quadratic_model_unchecked(f, x))
    _quadratic_model_unchecked(f::$F, x) = (_checkreal(f, x); f)
    quadratic_model!(q::$F, f::$F, x) =
      (_check_second_order_arrays(f, x); _quadratic_model_unchecked!(q, f, x))
    _quadratic_model_unchecked!(q::$F, f::$F, x) = (_checkreal(f, x); q)
  end
end

hessian_vec!(Hv::AbstractArray, f::Union{ZeroFunction, LinearFunction}, x::AbstractArray, v::AbstractArray) =
  (_check_second_order_arrays(f, x, Hv, v); _hessian_vec_unchecked!(Hv, f, x, v))
function _hessian_vec_unchecked!(Hv::AbstractArray, f::Union{ZeroFunction, LinearFunction}, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(f, x)
  fill!(Hv, zero(eltype(Hv)))
end

hessian_vec!(Hv::AbstractArray, f::SqrNormL2, x::AbstractArray, v::AbstractArray) =
  (_check_second_order_arrays(f, x, Hv, v); _hessian_vec_unchecked!(Hv, f, x, v))
function _hessian_vec_unchecked!(Hv::AbstractArray, f::SqrNormL2, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(f, x)
  copyto!(Hv, v)
  Hv .*= convert(eltype(Hv), f.λ)
end

function hessian_vec!(Hv::AbstractVector, f::QuadraticFunction, x::AbstractVector, v::AbstractVector)
  _checkreal(f, Hv)
  _checkreal(f, x)
  _checkreal(f, v)
  _checkhessian_output(Hv, x, v)
  _checksize(Hv, x)
  _checkdirection(v, x)
  mul!(Hv, f.A, v)
end

function hessian_vec!(Hv::AbstractVecOrMat, f::LeastSquaresLoss, β::AbstractVecOrMat, v::AbstractVecOrMat)
  _checkreal(f, Hv)
  _checkreal(f, β)
  _checkreal(f, v)
  _checkhessian_output(Hv, β, v)
  _checksize(Hv, β)
  _checkdirection(v, β)
  mul!(f.tmp, f.X, v)
  mul!(Hv, transpose(f.X), f.tmp)
end

# A V B, symmetrized when the variable is a symmetric matrix
hessian_vec!(Hv::AbstractArray, f::KronQuadratic, Θ::AbstractArray, V::AbstractArray) =
  (_check_second_order_arrays(f, Θ, Hv, V); _hessian_vec_unchecked!(Hv, f, Θ, V))
function _hessian_vec_unchecked!(Hv::AbstractArray, f::KronQuadratic, Θ::AbstractArray, V::AbstractArray)
  _checkshape(f, Hv)
  _checkshape(f, Θ)
  D = _densify!(f, V)
  mul!(f.AD, f.A, D)
  mul!(f.W, f.AD, f.B)
  (Θ isa SymmetricSparseIterate || Hv isa SymmetricSparseIterate) && _symmetrize!(f.W)
  copyto!(Hv, f.W)
end

####################################
#  generalized linear models
####################################

hashessian(::Type{<:GLMLoss}) = true

# X' Diagonal(w ⊙ ℓ''(η)) X v, with η = Xβ + offset; the buffer η holds Xv once
# the weights are in d
function hessian_vec!(Hv::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector, v::AbstractVector) where {L, T}
  _checkreal(f, Hv)
  _checkreal(f, β)
  _checkreal(f, v)
  _checkhessian_output(Hv, β, v)
  _checksize(Hv, β)
  _checkdirection(v, β)
  η = _linear_predictor!(f, β)
  d = f.d
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    d[i] = iszero(w) ? zero(T) : w * convert(T, deriv2(f.loss, η[i], f.y[i]))
  end
  mul!(η, f.X, v)
  η .= ifelse.(iszero.(d), zero(T), η .* d)
  mul!(Hv, transpose(f.X), η)
end

function quadratic_model(f::GLMLoss{L, T}, β::AbstractVector) where {L, T}
  _checkreal(f, β)
  n = length(f.y)
  _quadratic_model_unchecked!(GLMLoss(Squared(), zeros(T, n), f.X; weights = zeros(T, n), offset = f.offset), f, β)
end

# The IRLS model: weights vᵢ = wᵢℓ''ᵢ and working responses zᵢ = ηᵢ - ℓ'ᵢ/ℓ''ᵢ, with ℓ''
# floored so that the responses stay finite; the model's gradient at β is X'(wᵢℓ'ᵢ), the
# gradient of f, whatever the floor.
#
# The pair (vᵢ, zᵢ) of every observation is staged in the buffers of f, zᵢ over ηᵢ
# (read first) and vᵢ in d, and validated; q is written only once all of them are valid,
# so that a failure at one observation leaves every field of q as it was (O6). The
# buffers hold no result of f: gradient! and hessian_vec! overwrite them.
function quadratic_model!(q::GLMLoss{Squared, T, Vector{T}, <:Any, Vector{T}}, f::GLMLoss{L, T},
                          β::AbstractVector) where {L, T}
  _checkreal(f, β)
  q.X === f.X && q.offset === f.offset ||
    throw(ArgumentError("the model must come from quadratic_model of the same loss"))
  n = length(f.y)
  length(q.y) == n && length(q.weights) == n ||
    throw(DimensionMismatch("the model has $(length(q.y)) observations and the loss $n"))
  # q owns its output arrays: neither may be the other, nor an array of the loss (the buffers η
  # and d that stage the update, its data y, weights and offset, which the commit would
  # overwrite); identity only, a view or a reshape is not detected (O7)
  qy, qw = q.y, q.weights
  (qy === qw || qy === f.η || qy === f.d || qy === f.y || qy === f.weights || qy === f.offset ||
   qw === f.η || qw === f.d || qw === f.y || qw === f.weights || qw === f.offset) &&
    throw(ArgumentError("q.y and q.weights must be separate arrays, and not arrays of the loss"))
  η = _linear_predictor!(f, β)
  z, v = η, f.d
  floor = sqrt(eps(T))
  vanished = false                       # a nonzero weight with a coefficient that rounded to zero
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    if iszero(w)
      v[i] = z[i] = zero(T)
      continue
    end
    vi, zi = _irls_pair(f.loss, w, η[i], f.y[i], floor)
    vanished |= iszero(vi)
    v[i], z[i] = vi, zi
  end
  # a finite observation must give finite, nonzero coefficients: an infinite or vanishing
  # weight, or a non-finite response, is not the IRLS model of f; non-finite data propagate
  # as before. One pass over the staged values; only a suspicious result is looked at again.
  finite = true
  @inbounds @simd for i in eachindex(v)
    finite &= isfinite(v[i]) & isfinite(z[i])
  end
  (finite & !vanished) || _irls_check(f, β, v, z)
  @inbounds @simd for i in eachindex(v)
    qw[i] = v[i]
    qy[i] = z[i]
  end
  q
end

# The staged pair (v, z) of f is not finite, or a coefficient vanished. A predictor, response or
# weight that is itself not finite explains it (the result propagates, as before); otherwise
# the first observation with finite data throws. The predictors are recomputed in a new array,
# since f.η holds z; this path allocates (a successful update with non-finite data too).
@noinline function _irls_check(f::GLMLoss{L, T}, β, v, z) where {L, T}
  η = similar(f.η)
  mul!(η, f.X, β)
  f.offset === nothing || (η .+= f.offset)
  for i in eachindex(v)
    w = convert(T, _entry(f.weights, i))
    iszero(w) && continue
    representable = isfinite(v[i]) & isfinite(z[i]) & !iszero(v[i])
    finite_data = isfinite(η[i]) & isfinite(f.y[i]) & isfinite(w)
    (representable | !finite_data) && continue   # a non-finite predictor, response or weight propagates
    throw(PrecisionError("the quadratic model of a GLMLoss has no representable IRLS coefficient and " *
      "working response in $T at observation $i (linear predictor $(η[i]), weight $w, " *
      "coefficient $(v[i]), working response $(z[i])); the model is unchanged"))
  end
  nothing
end

# (wℓ'', η - ℓ'/ℓ'') with ℓ'' floored at `floor`
@inline function _irls_pair(ℓ::ScalarLoss, w::T, η::T, y, floor::T) where {T}
  d1 = convert(T, deriv(ℓ, η, y))
  d2 = max(convert(T, deriv2(ℓ, η, y)), floor)
  w * d2, η - d1 / d2
end

# Poisson: ℓ' = exp(η) - y, ℓ'' = exp(η), one exponential for both. There are three tiers.
# (1) The ordinary expressions are evaluated first and returned when the coefficient is
# below 0.999 floatmax in modulus (`_nearmax`) and the response is finite: w exp(η) is then
# far from the overflow threshold, which an error of a few ulps cannot reach.
# (2) Where exp(η) overflows at a finite η, or where the coefficient or the response of a
# finite observation leaves the range although exp(η) is above the floor (exp(η) - y
# overflows for a large negative y, say), they are recomputed: w exp(η) = w m 2^j with the
# power of two held outside the range until the final rounding, and η - 1 + y exp(-η), the
# limit of η - (exp(η) - y)/exp(η), as the response.
# (3) A coefficient of either tier within 0.1% of floatmax or past it is not trusted, because
# `w * exp(η)` and `w m 2^j` are approximations (a few ulps, 2^-50 relative at most for Float64)
# that can fall on the wrong side of the overflow threshold floatmax + ulp/2; there the
# coefficient is the correctly rounded w exp(η), Inf if that overflows (which the caller
# rejects), from `_poisson_upper`. Below the floor the coefficient is w √eps and
# only the response can leave the range: η - (exp(η) - y)/√eps is a sum of two terms that
# may each be out of range while the sum is not (η = -floatmax and y/√eps just above it),
# so it is formed in double-word arithmetic and rounded once (`_floored_response`), and the
# caller rejects it only if that rounded sum is not finite.
# A non-finite weight, response or predictor is not recovered: it propagates as before.
@inline function _irls_pair(::Poisson, w::T, η::T, y, floor::T) where {T}
  e = exp(η)
  if isfinite(e)
    d2 = max(e, floor)
    v, z = w * d2, η - convert(T, e - y) / d2
    if !((abs(v) < _nearmax(T)) & isfinite(z)) && isfinite(y) & isfinite(w)
      if isfinite(v) & isfinite(z)         # v within 0.1% of floatmax: certify v, z is ordinary
        e > floor && (v = _poisson_upper(w, η, v))
        return v, z
      end
      e > floor && return _poisson_recovered(w, η, y)
      isfinite(z) || !isfinite(η) || (z = _floored_response(η, e, y, d2))
    end
    return v, z
  end
  isfinite(η) & isfinite(y) & isfinite(w) && return _poisson_recovered(w, η, y)
  d2 = max(e, floor)                     # non-finite data propagate as before
  w * d2, η - convert(T, e - y) / d2
end

# η - (e - y)/d2 for a finite η, e and y, rounded once to T: the quotient (y/2)/d2 is held as a
# sum qh + ql of two floats (ql is the exact remainder, by a fused multiply-add), added to η/2
# by an error-free sum, and the sum is rounded once to T, through a round-to-odd value in the
# working precision when that differs from T (so that no tie of the working precision decides
# the rounding to T). The halves keep every term in range and the doubling at the end is exact,
# so the result is Inf exactly when the sum, not one of its terms, is out of range (up to ties of
# a relative width 2^-100). For Float32 and Float64 responses nothing here allocates or reads global
# state (no BigFloat precision or rounding mode); a BigFloat response allocates and is computed in the
# current BigFloat precision (T7).
# The working precision R is that of the promotion of T and the response, as `e - y` is: a response
# of another type (an integer, a rational) is converted to it first, as in the ordinary expression
# (numerics T3), and a BigFloat response is computed at the BigFloat precision current (T7) and
# then delivered in T; the once-rounding statement is for Float32 and Float64 responses (S1).
@noinline function _floored_response(η::T, e::T, y, d2::T) where {T}
  R = promote_type(T, typeof(y))
  half = R(0.5)
  ηh, yh, eh, dh = convert(R, η) * half, convert(R, y) * half, convert(R, e) * half, convert(R, d2)
  qh = yh / dh                           # (y/2)/d2
  isfinite(qh) || return convert(T, qh)  # |y/d2| > 2 floatmax: the sum exceeds floatmax
  ql = fma(-qh, dh, yh) / dh             # qh + ql = (y/2)/d2 to about twice the precision
  s = ηh + qh                            # s + sl = ηh + qh exactly, unless s overflowed
  isfinite(s) || return convert(T, 2 * s)
  bv = s - ηh
  sl = (ηh - (s - bv)) + (qh - bv)
  t = sl + (ql - eh / dh)
  hi = s + t
  lo = t - (hi - s)                      # hi + lo = s + t exactly (|s| ≥ |t|)
  isfinite(hi) || return convert(T, 2 * hi)
  R === T || (hi = _round_odd(hi, lo))
  convert(T, 2 * hi)
end

# hi + lo, with hi = RN(hi + lo), rounded to odd: hi where lo = 0 or the last bit of hi is odd,
# else the neighbor of hi toward lo. Rounding that value to a type with at most p - 2 bits is the
# rounding of hi + lo (no double rounding).
@inline function _round_odd(hi::R, lo::R) where {R<:Base.IEEEFloat}
  (iszero(lo) || isodd(reinterpret(Base.uinttype(R), hi))) && return hi
  lo > 0 ? nextfloat(hi) : prevfloat(hi)
end
# R not an IEEE type (BigFloat): the sum is rounded in the precision of R, and then to T (T7)
@inline _round_odd(hi, lo) = hi

@noinline function _poisson_recovered(w::T, η::T, y) where {T}
  m, j = _exp_split(η)
  R = promote_type(T, typeof(y))          # y exp(-η) in the precision of the data, as e - y is
  v = _scaled_product(w, m, j)
  # a coefficient in the subnormal range, or rounded to floatmin from below it, is rounded once as well (ldexp rounds a second time
  # there, and before Julia 1.11 truncates); there the power j is small enough to be a float
  if T <: Base.IEEEFloat && isfinite(v) && abs(v) <= floatmin(T) && !iszero(w) && !iszero(m) &&
     exponent(floatmin(T)) <= j <= exponent(floatmax(T))
    v = _scaled_coefficient(w, m, ldexp(one(T), j), v)
  end
  abs(v) < _nearmax(T) || (v = _poisson_upper(w, η, v))
  v, (η - one(T)) + convert(T, y * exp(-convert(R, η)))
end

# 0.999 floatmax: a coefficient below it in modulus is not close enough to the overflow threshold
# for the error of `w * exp(η)` or of the exponent split to change its class. Other types
# keep the ordinary result: abs(v) < Inf is isfinite(v) (and fails for NaN, as isfinite does).
@inline _nearmax(::Type{T}) where {T<:Union{Float32, Float64}} = T(0.999) * floatmax(T)
@inline _nearmax(::Type{T}) where {T} = T(Inf)

using Base.MPFR: libmpfr, MPFRRoundingMode, MPFRRoundDown, MPFRRoundUp

# w exp(η) rounded once to T, to nearest with ties to even and Inf for a value at or past
# floatmax + ulp/2: the coefficient of a finite w and η near or past the overflow threshold. It is
# computed in two stages.
# Screen: L = log2|w| + η log2(e), in Float64, is within 1e-11 of the exact log2|w exp(η)|
# (|η| ≤ _exp_bound ≈ 1456 so |η log2(e)| ≤ 2100, two roundings of relative size 2^-53, and log2 of
# a double within a few ulps of a value of modulus at most 1075), and log2(floatmax + ulp/2) is
# below emax + 1. L > emax + 1 + 2^-9 therefore means that the exact value exceeds the threshold
# by more than 0.13%, and the result is ±Inf. For η > _exp_bound(T) (where the exponent split gives
# Inf) L exceeds emax + 2 even for the smallest subnormal weight, so the screen returns Inf there too.
# Certification (Ziv): at the explicit precision p, lo = RD(RD(exp(η)) |w|) and hi = RU(RU(exp(η)) |w|)
# (MPFR directed rounding, from `mpfr_exp` and `mpfr_mul` called at the destination precision) enclose
# |w| exp(η). Rounding to nearest (`T(x, RoundNearest)`, an explicit mode) is monotone, so if lo and hi
# round to the same T they are the correctly rounded value, the class (finite or Inf) included; otherwise
# p is doubled. For η ≠ 0 the value is transcendental, so never a rounding boundary of T, and an unbounded
# loop would end; for η = 0 the product is exact. The loop stops at 65536 bits: going past it would need
# |w| exp(η) within about 2^-65000 (relative) of a rounding boundary of T. No such input is known (the
# closest tested, w = floatmax and η near 2^-54, is about 2^-107 away), but none is proved impossible, and
# for one the call throws PrecisionError instead of returning an uncertified value. Nothing here sets or reads the BigFloat precision or rounding
# mode, and nothing takes a lock; the call allocates (BigFloat).
# A zero weight or a NaN coefficient (0·Inf from an overflowed exponent, propagated as it was) is returned as is.
@noinline function _poisson_upper(w::T, η::T, v::T) where {T<:Union{Float32, Float64}}
  (iszero(w) | isnan(v)) && return v
  aw = abs(w)
  L = log2(Float64(aw)) + Float64(η) * 1.4426950408889634
  L > exponent(floatmax(T)) + 1 + 2.0^-9 && return copysign(T(Inf), w)
  p = 2 * precision(T) + 32
  while p <= 65536
    lo = T(_wexp_directed(Float64(aw), Float64(η), p, MPFRRoundDown), RoundNearest)
    hi = T(_wexp_directed(Float64(aw), Float64(η), p, MPFRRoundUp), RoundNearest)
    lo == hi && return copysign(lo, w)
    p *= 2
  end
  throw(PrecisionError("w exp(η) could not be rounded to $T at 65536 bits (weight $w, linear predictor $η)"))
end
@inline _poisson_upper(w::T, η::T, v::T) where {T} = v

# |w| exp(η) rounded toward r (MPFRRoundDown or MPFRRoundUp) at p bits, for finite w and η and
# p ≥ 53: exp(η) rounded toward r, then multiplied by |w| and rounded toward r again. The inputs
# are converted exactly, and no global precision or rounding mode is read.
function _wexp_directed(w::Float64, η::Float64, p::Int, r::MPFRRoundingMode)
  x = BigFloat(η, MPFRRoundDown; precision = p)
  b = BigFloat(w, MPFRRoundDown; precision = p)
  z = BigFloat(; precision = p)
  ccall((:mpfr_exp, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode), z, x, r)
  ccall((:mpfr_mul, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode), z, z, b, r)
  z
end

####################################
#  the Gaussian log-likelihood
####################################

hashessian(::Type{<:LogDetLoss}) = true

# w X⁻¹ V X⁻¹, from the Cholesky factor of X; the eigenvalue workspace, which
# only the proximal operator uses, holds X⁻¹ V, and the product goes through
# f.out when BLAS cannot write Hv
function hessian_vec!(Hv::StridedMatrix{T}, f::LogDetLoss{T}, X::AbstractMatrix, V::AbstractMatrix) where {T}
  _checkreal(f, Hv)
  _checkreal(f, X)
  _checkreal(f, V)
  _checkhessian_output(Hv, X, V)
  _checksize(Hv, X)
  _checkdirection(V, X)
  if _logdet_chol!(f, X) === nothing
    fill!(Hv, T(NaN))
    return Hv
  end
  C, M = f.C, f.eig.A
  _potri!('U', C)                                         # the upper triangle of X⁻¹
  B = _blas_ready(Hv) ? Hv : _array_buffer!(f.out, T, size(Hv))
  copyto!(B, V)
  BLAS.symm!('L', 'U', one(T), C, B, zero(T), M)
  BLAS.symm!('R', 'U', f.w, C, M, zero(T), B)
  B === Hv || copyto!(Hv, B)
  Hv
end
@noinline _throw_logdet_output(::LogDetLoss{T}) where {T} =
  throw(ArgumentError("hessian_vec! of a LogDetLoss writes a dense Matrix{$T}"))
hessian_vec!(Hv::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix, V::AbstractMatrix) where {T} =
  (_checkreal(f, X); _checkreal(f, V); _checkreal(f, Hv);
   _checkhessian_output(Hv, X, V);
   _throw_logdet_output(f))

function quadratic_model(f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  _checkreal(f, X)
  p = size(f.S, 1)
  _quadratic_model_unchecked!(KronQuadratic(Matrix{T}(I, p, p), Matrix{T}(I, p, p), zeros(T, p, p)), f, X)
end

# KronQuadratic(wW, W, w(S - 2W)), W = X⁻¹: w(tr(WZWZ)/2 + tr((S - 2W)Z)) is the
# second-order expansion of w(tr(SZ) - log det Z) at X, since WXW = W; the
# matrices are written exactly symmetric, from the upper triangle of W
quadratic_model!(q::KronQuadratic{T, Matrix{T}}, f::LogDetLoss{T}, X::AbstractMatrix) where {T} =
  (_check_second_order_arrays(f, X); _quadratic_model_unchecked!(q, f, X))
function _quadratic_model_unchecked!(q::KronQuadratic{T, Matrix{T}}, f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  size(q) == size(f.S) || throw(DimensionMismatch("the model is for $(size(q)) matrices and S is $(size(f.S))"))
  _logdet_chol!(f, X) === nothing &&
    throw(ArgumentError("X is not positive definite: LogDetLoss has no quadratic model there"))
  C, S, w = f.C, f.S, f.w
  _potri!('U', C)
  A, B, Cq = q.A, q.B, q.C
  @inbounds for j in axes(C, 2), i in 1:j
    Wij = C[i, j]
    A[i, j] = A[j, i] = w * Wij
    B[i, j] = B[j, i] = Wij
    Cq[i, j] = Cq[j, i] = w * (S[i, j] - 2Wij)
  end
  q
end

####################################
#  functions built from functions
####################################

for F in (:Scaled, :Tilt, :Translate)
  @eval hashessian(::Type{<:$F{G}}) where {G} = hashessian(G)
end
hashessian(::Type{<:Sum{F}}) where {F} = all(hashessian, fieldtypes(F))
hashessian(::Type{SeparableSum{F}}) where {F} = all(hashessian, fieldtypes(F))

hessian_vec!(Hv, s::Scaled, x, v) =
  (_check_second_order_arrays(s, x, Hv, v); _hessian_vec_unchecked!(Hv, s, x, v))
_hessian_vec_unchecked!(Hv, s::Scaled, x, v) = _chained(s) ?
  (_checkparams(s, x); _hessian_vec_unchecked!(Hv, _scaled_chain(s)[1], x, v); _scale!(Hv, _scaled_chain(s)[2])) :
  (_checkparams(s, x); _hessian_vec_unchecked!(Hv, s.f, x, v); _scale!(Hv, s.α))
hessian_vec!(Hv, t::Tilt, x, v) =
  (_check_second_order_arrays(t, x, Hv, v); _hessian_vec_unchecked!(Hv, t, x, v))
_hessian_vec_unchecked!(Hv, t::Tilt, x, v) = (_checkparams(t, x); _hessian_vec_unchecked!(Hv, t.f, x, v))
hessian_vec!(Hv, t::Translate, x::AbstractArray, v) =
  (_check_second_order_arrays(t, x, Hv, v); _hessian_vec_unchecked!(Hv, t, x, v))
_hessian_vec_unchecked!(Hv, t::Translate, x::AbstractArray, v) = _hessian_vec_unchecked!(Hv, t.f, _shifted(t, x), v)

# the remaining functions' products added into Hv through the buffer b
_add_hessian_vecs!(Hv, b, x, v, ::Tuple{}) = Hv
function _add_hessian_vecs!(Hv, b, x, v, fs::Tuple)
  _hessian_vec_unchecked!(b, first(fs), x, v)
  _accumulate!(Hv, b)
  _add_hessian_vecs!(Hv, b, x, v, Base.tail(fs))
end

hessian_vec!(Hv::Union{AbstractArray, Tuple}, s::Sum, x::Union{AbstractArray, Tuple}, v::Union{AbstractArray, Tuple}) =
  (_check_second_order_arrays(s, x, Hv, v); _hessian_vec_unchecked!(Hv, s, x, v))
function _hessian_vec_unchecked!(Hv::Union{AbstractArray, Tuple}, s::Sum, x::Union{AbstractArray, Tuple}, v::Union{AbstractArray, Tuple})
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(s, x)
  _checkreal(s, x)
  length(s.fs) > 1 && _check_sum_output(Hv)
  _hessian_vec_unchecked!(Hv, first(s.fs), x, v)
  length(s.fs) == 1 && return Hv
  b = _sum_buffer!(s.scratch, Hv, x)
  _add_hessian_vecs!(Hv, b, x, v, Base.tail(s.fs))
end

hessian_vec!(Hv::Tuple, s::SeparableSum, x::Tuple, v::Tuple) =
  (_check_second_order_arrays(s, x, Hv, v); _hessian_vec_unchecked!(Hv, s, x, v))
_hessian_vec_unchecked!(Hv::Tuple, s::SeparableSum, x::Tuple, v::Tuple) =
  (_checksize(Hv, x); _checkdirection(v, x); _checkparams(s, x); _checkreal(s, x);
   _hessian_blocks!(s.fs, Hv, x, v); Hv)

quadratic_model(s::Scaled, x) =
  (_check_second_order_arrays(s, x); _quadratic_model_unchecked(s, x))
_quadratic_model_unchecked(s::Scaled, x) = Scaled(_quadratic_model_unchecked(s.f, x), s.α)
quadratic_model(t::Tilt, x) =
  (_check_second_order_arrays(t, x); _quadratic_model_unchecked(t, x))
_quadratic_model_unchecked(t::Tilt, x) = (_checkreal(t, x); Tilt(_quadratic_model_unchecked(t.f, x), t.c, t.d))
quadratic_model(t::Translate, x::AbstractArray) =
  (_check_second_order_arrays(t, x); _quadratic_model_unchecked(t, x))
_quadratic_model_unchecked(t::Translate, x::AbstractArray) = Translate(_quadratic_model_unchecked(t.f, _shifted(t, x)), t.b)
quadratic_model(s::Sum, x) =
  (_check_second_order_arrays(s, x); _quadratic_model_unchecked(s, x))
_quadratic_model_unchecked(s::Sum, x) = Sum(map(f -> _quadratic_model_unchecked(f, x), s.fs)...)
quadratic_model(s::SeparableSum, x::Tuple) =
  (_check_second_order_arrays(s, x); _quadratic_model_unchecked(s, x))
_quadratic_model_unchecked(s::SeparableSum, x::Tuple) =
  (_checkblocks(s, x); SeparableSum(_quadratic_blocks(s.fs, x)))

quadratic_model!(q::Scaled, s::Scaled, x) =
  (_check_second_order_arrays(s, x); _quadratic_model_unchecked!(q, s, x))
_quadratic_model_unchecked!(q::Scaled, s::Scaled, x) = (_quadratic_model_unchecked!(q.f, s.f, x); q)
quadratic_model!(q::Tilt, t::Tilt, x) =
  (_check_second_order_arrays(t, x); _quadratic_model_unchecked!(q, t, x))
_quadratic_model_unchecked!(q::Tilt, t::Tilt, x) = (_checkreal(t, x); _quadratic_model_unchecked!(q.f, t.f, x); q)
quadratic_model!(q::Translate, t::Translate, x::AbstractArray) =
  (_check_second_order_arrays(t, x); _quadratic_model_unchecked!(q, t, x))
_quadratic_model_unchecked!(q::Translate, t::Translate, x::AbstractArray) = (_quadratic_model_unchecked!(q.f, t.f, _shifted(t, x)); q)
quadratic_model!(q::Sum, s::Sum, x) =
  (_check_second_order_arrays(s, x); _quadratic_model_unchecked!(q, s, x))
_quadratic_model_unchecked!(q::Sum, s::Sum, x) =
  (length(q.fs) == length(s.fs) || throw(DimensionMismatch("the model has $(length(q.fs)) parts and the function $(length(s.fs))"));
   _model_parts!(q.fs, s.fs, x); q)
quadratic_model!(q::SeparableSum, s::SeparableSum, x::Tuple) =
  (_check_second_order_arrays(s, x); _quadratic_model_unchecked!(q, s, x))
_quadratic_model_unchecked!(q::SeparableSum, s::SeparableSum, x::Tuple) =
  (_checkblocks(s, x); _checkmodelblocks(q, s); _quadratic_blocks!(q.fs, s.fs, x); q)
function _checkmodelblocks(q::SeparableSum, s::SeparableSum)
  length(q.fs) == length(s.fs) || throw(DimensionMismatch("the model has $(length(q.fs)) blocks and the function $(length(s.fs))"))
end

_model_parts!(::Tuple{}, ::Tuple{}, x) = nothing
function _model_parts!(qs::Tuple, fs::Tuple, x)
  _quadratic_model_unchecked!(first(qs), first(fs), x)
  _model_parts!(Base.tail(qs), Base.tail(fs), x)
end

@generated _hessian_blocks!(fs::Tuple, Hv::Tuple, x::Tuple, v::Tuple) =
  _blockwise(k -> :(_hessian_vec_unchecked!(Hv[$k], fs[$k], x[$k], v[$k])), min(fieldcount(fs), fieldcount(Hv), fieldcount(x), fieldcount(v)), nothing)
@generated _quadratic_blocks(fs::Tuple, x::Tuple) =
  _blocktuple(k -> :(_quadratic_model_unchecked(fs[$k], x[$k])), min(fieldcount(fs), fieldcount(x)))
@generated _quadratic_blocks!(qs::Tuple, fs::Tuple, x::Tuple) =
  _blockwise(k -> :(_quadratic_model_unchecked!(qs[$k], fs[$k], x[$k])), min(fieldcount(qs), fieldcount(fs), fieldcount(x)), nothing)
