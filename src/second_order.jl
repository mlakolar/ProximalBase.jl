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
`v` in `Hv` and return `Hv`, for a function with [`hashessian`](@ref). Use an
output that does not overlap `x` or `v`, as for `mul!`. `Hv` is
a dense array of the shape of `x`; `x` and `v` may be sparse iterates. The
product costs about as much as a gradient: two products with `X` for
[`LeastSquaresLoss`](@ref) and [`GLMLoss`](@ref), `w X⁻¹VX⁻¹` from one Cholesky
factorization for [`LogDetLoss`](@ref) (`NaN`s when `x` is not positive
definite), and it allocates nothing. On a [`SymmetricSparseIterate`](@ref) the
product of [`KronQuadratic`](@ref) is the one over symmetric matrices, as its
gradient is. Several products at the same `x` are cheaper through
[`quadratic_model`](@ref), which does the work that depends on `x` once.

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
allocating, and returns it. A model of a [`SeparableSum`](@ref) must have as
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
the gradient of `q` at `x` is still `∇f(x)` exactly. The coefficients `wᵢℓ''ᵢ`
are the *effective* ones: a Poisson observation whose `exp(ηᵢ)` overflows still
has a finite coefficient when `wᵢ exp(ηᵢ)` is representable (weight `1/8` at
`η = 710` gives `exp(710)/8` and the working response `709`), and a coefficient
or working response of a finite observation that is not representable in the
working precision (or a nonzero weight whose coefficient rounds to zero) throws
a [`PrecisionError`](@ref), which is never clamped. `quadratic_model!` computes
the whole model before it writes, so after a `PrecisionError` every field of `q`
is as it was and `q` can be updated again. Non-finite data propagate as
`NaN`/`Inf` without an error. The all-or-nothing update holds for one `GLMLoss`:
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
  _checksize(Hv, x)
  _checkdirection(v, x)
  mul!(Hv, f.A, v)
end

function hessian_vec!(Hv::AbstractVecOrMat, f::LeastSquaresLoss, β::AbstractVecOrMat, v::AbstractVecOrMat)
  _checkreal(f, Hv)
  _checkreal(f, β)
  _checkreal(f, v)
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
  η = _linear_predictor!(f, β)
  z, v = η, f.d
  floor = sqrt(eps(T))
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    if iszero(w)
      v[i] = z[i] = zero(T)
      continue
    end
    ηi, yi = η[i], f.y[i]
    vi, zi = _irls_pair(f.loss, w, ηi, yi, floor)
    # a finite observation must give finite, nonzero coefficients: an infinite or
    # vanishing weight, or a non-finite response, is not the IRLS model of f;
    # non-finite data propagate as before
    (isfinite(vi) & isfinite(zi) & !iszero(vi)) || !(isfinite(ηi) & isfinite(yi)) ||
      _irls_unrepresentable(i, ηi, w, vi, zi)
    v[i], z[i] = vi, zi
  end
  qw, qy = q.weights, q.y
  @inbounds @simd for i in eachindex(v)
    qw[i] = v[i]
    qy[i] = z[i]
  end
  q
end

@noinline _irls_unrepresentable(i, η::T, w, v, z) where {T} =
  throw(PrecisionError("the quadratic model of a GLMLoss has no representable IRLS coefficient and " *
    "working response in $T at observation $i (linear predictor $η, weight $w, " *
    "coefficient $v, working response $z); the model is unchanged"))

# (wℓ'', η - ℓ'/ℓ'') with ℓ'' floored at `floor`
@inline function _irls_pair(ℓ::ScalarLoss, w::T, η::T, y, floor::T) where {T}
  d1 = convert(T, deriv(ℓ, η, y))
  d2 = max(convert(T, deriv2(ℓ, η, y)), floor)
  w * d2, η - d1 / d2
end

# Poisson: ℓ' = exp(η) - y, ℓ'' = exp(η), one exponential for both. Where exp(η) overflows
# at a finite η, the coefficient w exp(η) = w m 2^j is formed with the power of two held
# outside the range until the final rounding, so that it is returned whenever it is
# representable (Inf otherwise, which the caller rejects), and the response is
# η - 1 + y exp(-η), the limit of η - (exp(η) - y)/exp(η) without the Inf/Inf.
@inline function _irls_pair(::Poisson, w::T, η::T, y, floor::T) where {T}
  e = exp(η)
  if isfinite(e) || !isfinite(η)
    d2 = max(e, floor)
    return w * d2, η - convert(T, e - y) / d2
  end
  m, j = _exp_pow2(η)
  _scaled_product(w, m, j), (η - one(T)) + convert(T, y * exp(-η))
end

# exp(x) = m 2^j for a finite x, with j = round(x/log 2) and x - j log 2 in two parts, the
# high one with trailing zeros so that j⋅hi is exact (fdlibm); m ≈ exp of a number in
# [-log(2)/2, log(2)/2], to about one ulp, which exp(x + log w) would not give. The same
# arithmetic as the exponent-recovery helper of the GLM value oracles (#358, part B);
# one of them should replace the other where the two meet.
@inline _exp_ln2_parts(::Type{Float64}) = (6.93147180369123816490e-01, 1.90821492927058770002e-10)
@inline _exp_ln2_parts(::Type{Float32}) = (6.9314575195f-01, 1.4286067653f-06)
@inline function _exp_pow2(x::T) where {T<:Union{Float32, Float64}}
  hi, lo = _exp_ln2_parts(T)
  j = round(x * T(1.4426950408889634))
  exp((x - j * hi) - j * lo), unsafe_trunc(Int, j)
end
@inline _exp_pow2(x::AbstractFloat) = frexp(exp(x))

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
hessian_vec!(Hv::AbstractMatrix, f::LogDetLoss{T}, X::AbstractMatrix, V::AbstractMatrix) where {T} =
  (_checkreal(f, X); _checkreal(f, V); _checkreal(f, Hv);
   throw(ArgumentError("hessian_vec! of a LogDetLoss writes a dense Matrix{$T}")))

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
_hessian_vec_unchecked!(Hv, s::Scaled, x, v) = (_checkparams(s, x); _hessian_vec_unchecked!(Hv, s.f, x, v); _scale!(Hv, s.α))
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
  Hv .+= b
  _add_hessian_vecs!(Hv, b, x, v, Base.tail(fs))
end

hessian_vec!(Hv::AbstractArray, s::Sum, x::AbstractArray, v::AbstractArray) =
  (_check_second_order_arrays(s, x, Hv, v); _hessian_vec_unchecked!(Hv, s, x, v))
function _hessian_vec_unchecked!(Hv::AbstractArray, s::Sum, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(s, x)
  _checkreal(s, x)
  _hessian_vec_unchecked!(Hv, first(s.fs), x, v)
  length(s.fs) == 1 && return Hv
  b = _array_buffer!(s.scratch, eltype(Hv), size(Hv))
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
