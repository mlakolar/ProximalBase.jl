####################################
#
#  Functions of the singular values
#
####################################

"""
    SingularValue(f)

The function `F(X) = f(σ(X))` of the singular values `σ(X)` of a matrix, for a
vector function `f` with a proximal operator that is absolutely symmetric:
invariant under permutations and sign changes of its argument. Its proximal
operator acts on the singular values,

    prox_{γF}(X) = U Diagonal(prox_{γf}(σ)) Vᵀ,   X = U Diagonal(σ) Vᵀ (thin SVD),

(Lewis, 1995), and [`value_and_prox!`](@ref) returns `f` at the new singular
values, without a second decomposition. For a nonconvex separable `f` whose
scalar proximal operator is nondecreasing on `[0, ∞)`, such as [`MCP`](@ref),
[`SCAD`](@ref) or [`NormL0`](@ref), the same formula gives a global minimizer
(generalized singular value thresholding; Lu, Zhu, Xu, Yan and Lin, 2015).

| `SingularValue(f)` | `F(X)` |
|:-------------------|:-------|
| `SingularValue(NormL1(λ))` | the nuclear norm `λ‖X‖_*`, also [`NuclearNorm`](@ref) |
| `SingularValue(NormLinf(λ))` | the spectral norm `λ‖X‖₂` |
| `SingularValue(IndBallL1(r))` | the indicator of the nuclear-norm ball `‖X‖_* ≤ r` |
| `SingularValue(IndBallLinf(r))` | the indicator of the spectral-norm ball `‖X‖₂ ≤ r` |
| `SingularValue(IndBallL0(r))` | the indicator of the matrices of rank at most `r`, also [`IndBallRank`](@ref) |
| `SingularValue(ElasticNet(λ1, λ2))` | the matrix elastic net `λ1‖X‖_* + (λ2/2)‖X‖_F²` |
| `SingularValue(Conjugate(Constrained(IndBox(0, λ), IndHalfspace(1, λ * k))))` | the Ky Fan `k`-norm `λ(σ₁ + … + σₖ)` |
| `SingularValue(MCP(λ, a))`, `SCAD`, `NormL0` | nonconvex penalties of the singular values |

The Ky Fan norm is the support function of `{w : 0 ≤ wᵢ ≤ λ, Σᵢ wᵢ ≤ λk}` at the
singular values. It needs the halfspace: the conjugate of the capped simplex,
with the hyperplane `Σᵢ wᵢ = k`, is the sum of the `k` largest entries, which is
not absolutely symmetric.

`isconvex` and the indicator property come from `f`. In the value `F(X)`,
singular values within a relative `√ε` of the largest count as zero, as they do
for `IndBallRank`: the value at a proximal point of low rank then agrees with
the one [`value_and_prox!`](@ref) returns, also for a discontinuous `f` such as
`NormL0`. The value of `Conjugate(SingularValue(f))` is `f*(σ(Y))`, known when
that of `Conjugate(f)` is, as for the nuclear-norm ball and the spectral norm,
which are conjugate to each other.

$_SPECTRAL_NOTE
"""
mutable struct SingularValue{F<:AbstractFunction} <: AbstractFunction
  f::F
  ws::Any          # the SVD workspace for the size and element type of the input
  t::Any           # the proximal point of the singular values, or a rounded copy

  function SingularValue{F}(f::F) where {F<:AbstractFunction}
    isproximable(f) || throw(ArgumentError("SingularValue needs a proximable f, but $(nameof(F)) is not"))
    new{F}(f, nothing, nothing)
  end
end
SingularValue(f::AbstractFunction) = SingularValue{typeof(f)}(f)

isproximable(::Type{<:SingularValue}) = true
isconvex(::Type{<:SingularValue{F}}) where {F} = isconvex(F)
_isindicator(::Type{<:SingularValue{F}}) where {F} = _isindicator(F)

# a vector of length k and element type T, kept between calls
function _vector_buffer(cache, ::Type{T}, k::Integer) where {T}
  if cache isa Vector{T}
    length(cache) == k && return cache
  end
  Vector{T}(undef, k)
end

# the singular values of X, in descending order, owned by the workspace of F
function _singular_values!(F::SingularValue, X::AbstractMatrix)
  T = float(eltype(X))
  ws = _svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  s
end

# the value, with the singular values within rounding of zero set to zero
function (F::SingularValue)(X::AbstractMatrix)
  s = _singular_values!(F, X)
  T = eltype(s)
  t = _vector_buffer(F.t, T, length(s))
  F.t = t
  c = isempty(s) ? zero(T) : _tol(T) * s[1]       # relative, so that F scales with X
  @inbounds for j in eachindex(s)
    t[j] = s[j] <= c ? zero(T) : s[j]
  end
  convert(T, F.f(t))
end

function value_and_prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real)
  _checksize(Y, X)
  T = float(eltype(Y))
  ws = _svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = _svd!(ws)
  t = _vector_buffer(F.t, T, length(s))
  F.t = t
  v = value_and_prox!(t, F.f, s, γ)
  # scale the singular vectors up to the last nonzero singular value
  k = 0
  @inbounds for j in eachindex(t)
    iszero(t[j]) || (k = j)
  end
  @inbounds for j in 1:k
    tj = t[j]
    for i in axes(U, 1)
      U[i, j] *= tj
    end
  end
  _mul_lowrank!(Y, U, Vt, k, ws)
  convert(T, v)
end
prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real) = (value_and_prox!(Y, F, X, γ); Y)

# the conjugate of f ∘ σ is f* ∘ σ (Lewis, 1995)
_conjugate_value(F::SingularValue, Y::AbstractMatrix) = _conjugate_value(F.f, _singular_values!(F, Y))
