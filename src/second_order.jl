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
allocating, and returns it.

| `f` | `q` |
|:----|:----|
| [`QuadraticFunction`](@ref), [`KronQuadratic`](@ref), [`LeastSquaresLoss`](@ref), [`SqrNormL2`](@ref), [`LinearFunction`](@ref), [`ZeroFunction`](@ref) | `f` itself |
| [`GLMLoss`](@ref) | the weighted least-squares loss of IRLS, `GLMLoss(ProximalBase.Squared(), z, X; weights = w ⊙ ℓ'', offset)` with the working responses `zᵢ = ηᵢ - ℓ'ᵢ / ℓ''ᵢ` |
| [`LogDetLoss`](@ref)`(S, w)` | `KronQuadratic(wW, W, w(S - 2W))` with `W = x⁻¹` |
| [`Scaled`](@ref), [`Tilt`](@ref), [`Translate`](@ref), [`Sum`](@ref), [`SeparableSum`](@ref), [`Regularize`](@ref), [`Precompose`](@ref) | the same construction around the models of the parts |

For a `GLMLoss`, `ℓ''ᵢ` is floored at `√eps`, as in IRLS, so that the working
responses stay finite where the curvature vanishes (a Huber loss beyond `δ`);
the gradient of `q` at `x` is still `∇f(x)` exactly. For `LogDetLoss`, `q` is
written in the new point `z` rather than in the step `z - x`: minimizing
`q(z) + λ‖z‖₁` over symmetric `z` is the Newton step of the QUIC algorithm for
the graphical lasso (Hsieh, Sustik, Dhillon and Ravikumar, 2014), and
coordinate descent on a [`SymmetricSparseIterate`](@ref) solves it. `x` must
then be positive definite.
"""
function quadratic_model end
function quadratic_model! end
@doc (@doc quadratic_model) quadratic_model!

@noinline _no_hessian(f) =
  throw(ArgumentError(issmooth(f) ? "no Hessian is known for $(nameof(typeof(f)))" :
                                    "$(nameof(typeof(f))) is not smooth"))

hessian_vec!(Hv, f::AbstractFunction, x, v) = _no_hessian(f)
quadratic_model(f::AbstractFunction, x) = _no_hessian(f)
quadratic_model!(q, f::AbstractFunction, x) = _no_hessian(f)

_checkdirection(v, x) = size(v) == size(x) ||
  throw(DimensionMismatch("the direction has size $(size(v)) and the point $(size(x))"))
function _checkdirection(v::Tuple, x::Tuple)
  length(v) == length(x) || throw(DimensionMismatch("the direction and the point must have the same number of blocks"))
  map(_checkdirection, v, x)
  true
end

####################################
#  quadratics: the model is the function
####################################

for F in (:ZeroFunction, :LinearFunction, :SqrNormL2, :QuadraticFunction, :KronQuadratic, :LeastSquaresLoss)
  @eval begin
    hashessian(::Type{<:$F}) = true
    quadratic_model(f::$F, x) = f
    quadratic_model!(q::$F, ::$F, x) = q
  end
end

function hessian_vec!(Hv::AbstractArray, f::Union{ZeroFunction, LinearFunction}, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(f, x)
  fill!(Hv, zero(eltype(Hv)))
end

function hessian_vec!(Hv::AbstractArray, f::SqrNormL2, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(f, x)
  copyto!(Hv, v)
  Hv .*= convert(eltype(Hv), f.λ)
end

function hessian_vec!(Hv::AbstractVector, f::QuadraticFunction, x::AbstractVector, v::AbstractVector)
  _checksize(Hv, x)
  _checkdirection(v, x)
  mul!(Hv, f.A, v)
end

function hessian_vec!(Hv::AbstractVecOrMat, f::LeastSquaresLoss, β::AbstractVecOrMat, v::AbstractVecOrMat)
  _checksize(Hv, β)
  _checkdirection(v, β)
  mul!(f.tmp, f.X, v)
  mul!(Hv, transpose(f.X), f.tmp)
end

# A V B, symmetrized when the variable is a symmetric matrix
function hessian_vec!(Hv::AbstractArray, f::KronQuadratic, Θ::AbstractArray, V::AbstractArray)
  _checkshape(f, Hv)
  _checkshape(f, Θ)
  D = _densify!(f, V)
  mul!(f.AD, f.A, D)
  mul!(f.W, f.AD, f.B)
  Θ isa SymmetricSparseIterate && _symmetrize!(f.W)
  copyto!(Hv, f.W)
end

####################################
#  generalized linear models
####################################

hashessian(::Type{<:GLMLoss}) = true

# X' Diagonal(w ⊙ ℓ''(η)) X v, with η = Xβ + offset; the buffer η holds Xv once
# the weights are in d
function hessian_vec!(Hv::AbstractVector, f::GLMLoss{L, T}, β::AbstractVector, v::AbstractVector) where {L, T}
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
  n = length(f.y)
  quadratic_model!(GLMLoss(Squared(), zeros(T, n), f.X; weights = zeros(T, n), offset = f.offset), f, β)
end

# the IRLS model: weights wᵢℓ''ᵢ and working responses ηᵢ - ℓ'ᵢ/ℓ''ᵢ, with ℓ''
# floored so that the responses stay finite; the model's gradient at β is
# X'(wᵢℓ'ᵢ), the gradient of f, whatever the floor
function quadratic_model!(q::GLMLoss{Squared, T, Vector{T}, <:Any, Vector{T}}, f::GLMLoss{L, T},
                          β::AbstractVector) where {L, T}
  q.X === f.X && q.offset === f.offset ||
    throw(ArgumentError("the model must come from quadratic_model of the same loss"))
  η = _linear_predictor!(f, β)
  z, v = q.y, q.weights
  floor = sqrt(eps(T))
  @inbounds for i in eachindex(η)
    w = convert(T, _entry(f.weights, i))
    if iszero(w)
      v[i] = z[i] = zero(T)
      continue
    end
    d1 = convert(T, deriv(f.loss, η[i], f.y[i]))
    d2 = max(convert(T, deriv2(f.loss, η[i], f.y[i])), floor)
    v[i] = w * d2
    z[i] = η[i] - d1 / d2
  end
  q
end

####################################
#  the Gaussian log-likelihood
####################################

hashessian(::Type{<:LogDetLoss}) = true

# w X⁻¹ V X⁻¹, from the Cholesky factor of X; the eigenvalue workspace, which
# only the proximal operator uses, holds X⁻¹ V
function hessian_vec!(Hv::StridedMatrix{T}, f::LogDetLoss{T}, X::AbstractMatrix, V::AbstractMatrix) where {T}
  _checksize(Hv, X)
  _checkdirection(V, X)
  if _logdet_chol!(f, X) === nothing
    fill!(Hv, T(NaN))
    return Hv
  end
  C, M = f.C, f.eig.A
  LAPACK.potri!('U', C)                                   # the upper triangle of X⁻¹
  copyto!(Hv, V)
  BLAS.symm!('L', 'U', one(T), C, Hv, zero(T), M)
  BLAS.symm!('R', 'U', f.w, C, M, zero(T), Hv)
end
hessian_vec!(::AbstractMatrix, ::LogDetLoss{T}, ::AbstractMatrix, ::AbstractMatrix) where {T} =
  throw(ArgumentError("hessian_vec! of a LogDetLoss writes a dense Matrix{$T}"))

function quadratic_model(f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  p = size(f.S, 1)
  quadratic_model!(KronQuadratic(Matrix{T}(I, p, p), Matrix{T}(I, p, p), zeros(T, p, p)), f, X)
end

# KronQuadratic(wW, W, w(S - 2W)), W = X⁻¹: w(tr(WZWZ)/2 + tr((S - 2W)Z)) is the
# second-order expansion of w(tr(SZ) - log det Z) at X, since WXW = W; the
# matrices are written exactly symmetric, from the upper triangle of W
function quadratic_model!(q::KronQuadratic{T, Matrix{T}}, f::LogDetLoss{T}, X::AbstractMatrix) where {T}
  size(q) == size(f.S) || throw(DimensionMismatch("the model is for $(size(q)) matrices and S is $(size(f.S))"))
  _logdet_chol!(f, X) === nothing &&
    throw(ArgumentError("X is not positive definite: LogDetLoss has no quadratic model there"))
  C, S, w = f.C, f.S, f.w
  LAPACK.potri!('U', C)
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

hessian_vec!(Hv, s::Scaled, x, v) = (_checkparams(s, x); hessian_vec!(Hv, s.f, x, v); _scale!(Hv, s.α))
hessian_vec!(Hv, t::Tilt, x, v) = (_checkparams(t, x); hessian_vec!(Hv, t.f, x, v))
hessian_vec!(Hv, t::Translate, x::AbstractArray, v) = hessian_vec!(Hv, t.f, _shifted(t, x), v)

# the remaining functions' products added into Hv through the buffer b
_add_hessian_vecs!(Hv, b, x, v, ::Tuple{}) = Hv
function _add_hessian_vecs!(Hv, b, x, v, fs::Tuple)
  hessian_vec!(b, first(fs), x, v)
  Hv .+= b
  _add_hessian_vecs!(Hv, b, x, v, Base.tail(fs))
end

function hessian_vec!(Hv::AbstractArray, s::Sum, x::AbstractArray, v::AbstractArray)
  _checksize(Hv, x)
  _checkdirection(v, x)
  _checkparams(s, x)
  hessian_vec!(Hv, first(s.fs), x, v)
  length(s.fs) == 1 && return Hv
  b = _array_buffer!(s.scratch, eltype(Hv), size(Hv))
  _add_hessian_vecs!(Hv, b, x, v, Base.tail(s.fs))
end

hessian_vec!(Hv::Tuple, s::SeparableSum, x::Tuple, v::Tuple) =
  (_checksize(Hv, x); _checkdirection(v, x); _checkparams(s, x);
   map((f, Hk, xk, vk) -> hessian_vec!(Hk, f, xk, vk), s.fs, Hv, x, v); Hv)

quadratic_model(s::Scaled, x) = Scaled(quadratic_model(s.f, x), s.α)
quadratic_model(t::Tilt, x) = Tilt(quadratic_model(t.f, x), t.c, t.d)
quadratic_model(t::Translate, x::AbstractArray) = Translate(quadratic_model(t.f, _shifted(t, x)), t.b)
quadratic_model(s::Sum, x) = Sum(map(f -> quadratic_model(f, x), s.fs)...)
quadratic_model(s::SeparableSum, x::Tuple) =
  (_checkblocks(s, x); SeparableSum(map((f, xk) -> quadratic_model(f, xk), s.fs, x)))

quadratic_model!(q::Scaled, s::Scaled, x) = (quadratic_model!(q.f, s.f, x); q)
quadratic_model!(q::Tilt, t::Tilt, x) = (quadratic_model!(q.f, t.f, x); q)
quadratic_model!(q::Translate, t::Translate, x::AbstractArray) = (quadratic_model!(q.f, t.f, _shifted(t, x)); q)
quadratic_model!(q::Sum, s::Sum, x) = (_model_parts!(q.fs, s.fs, x); q)
quadratic_model!(q::SeparableSum, s::SeparableSum, x::Tuple) =
  (_checkblocks(s, x); map((qk, f, xk) -> quadratic_model!(qk, f, xk), q.fs, s.fs, x); q)

_model_parts!(::Tuple{}, ::Tuple{}, x) = nothing
function _model_parts!(qs::Tuple, fs::Tuple, x)
  quadratic_model!(first(qs), first(fs), x)
  _model_parts!(Base.tail(qs), Base.tail(fs), x)
end
