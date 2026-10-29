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

`isconvex` and the indicator property come from `f`. The domain of `F` is the
finite matrices: a matrix with a non-finite entry has value `Inf`, also for a
bounded `f` such as `NormL0`. In the value `F(X)`, singular values within a
relative `√ε` of the largest count as zero, as they do for `IndBallRank`: the
value at a proximal point of low rank then agrees with the one
[`value_and_prox!`](@ref) returns, also for a discontinuous `f` such as
`NormL0`. The value of `Conjugate(SingularValue(f))` is `f*` at the masked
singular values of `Y`, with the same domain, known when that of `Conjugate(f)`
is, as for the nuclear-norm ball and the spectral norm, which are conjugate to
each other.

The zero mask stays correct when the largest singular value overflows, and
when the threshold `√ε σ₁` is not a normal number: the mask then comes from the
singular values of `X` scaled to unit range. At an overflowing spectrum,
membership in `IndBallL1`, `IndBallL2`, `IndBallLinf` and `IndBox` is decided in
the scaled units, with the tolerance of the original ones. Whether another `f`
evaluates an overflowing spectrum correctly depends on `f`.

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
_argument_type(::Type{<:SingularValue}) = AbstractMatrix
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
  _checkblas(F, T)
  ws = _svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  s
end

# The domain is the finite matrices: a non-finite entry gives Inf. Singular
# values within rounding of zero count as zero. An overflowing spectrum (k > 0;
# a subnormal one has k < 0) decides membership in scaled units.
function (F::SingularValue)(X::AbstractMatrix)
  T = float(eltype(X))
  _checkblas(F, T)
  all(isfinite, X) || return T(Inf)
  t, scaled, k = _masked_singular_values!(F, X)
  k > 0 && _scales_membership(F.f) && return _scaled_membership(F.f, scaled, k)
  convert(T, F.f(t))
end

# The singular values with those within a relative tolerance of the largest set
# to zero, in the buffer F.t. When the largest one overflows, or the threshold
# is not a normal number, the mask comes from the singular values of X scaled
# by 2^-k, which are returned, masked, with k.
function _masked_singular_values!(F::SingularValue, X::AbstractMatrix)
  T = float(eltype(X))
  s = _singular_values!(F, X)
  t = _vector_buffer(F.t, T, length(s))
  F.t = t
  if !isempty(s) && (!_finite_value(s[1]) || (s[1] > zero(T) && _tol(T) * s[1] <= floatmin(T)))
    return _scaled_singular_mask!(F, X, s, t)
  end
  c = isempty(s) ? zero(T) : _tol(T) * s[1]       # relative, so that F scales with X
  @inbounds for j in eachindex(s)
    t[j] = s[j] <= c ? zero(T) : s[j]
  end
  t, nothing, 0
end

@noinline function _scaled_singular_mask!(F::SingularValue, X::AbstractMatrix, s::Vector{T}, t::Vector{T}) where {T}
  copyto!(t, s)
  ws = _svd_workspace(F.ws, T, size(X)...)
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  c = _tol(T) * s[1]
  @inbounds for j in eachindex(s)
    if s[j] <= c
      t[j] = zero(T)
      s[j] = zero(T)
    end
  end
  t, s, k
end

function value_and_prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real)
  _checkwrite(value_and_prox!, F, Y, X)
  T = float(eltype(Y))
  _checkblas(F, T)
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

function _checkspectrum(F::SingularValue, ::Type{T}, dims::Tuple) where {T}
  ws = F.ws
  k = min(dims...)
  T <: _BlasReal && ws isa SVDWorkspace{T} && (ws.m, ws.n) == dims ? ws.s : Vector{T}(undef, k)
end
function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, F::SingularValue, Y::AbstractMatrix, X::AbstractMatrix)
  _checksize(Y, X)
  T = float(eltype(Y))
  _checkworkspace(_svd_workspace, F.ws, T, size(X)...)
  s = _checkspectrum(F, T, size(X))
  t = _checkbuffer(F.t, T, size(s))
  _checkwrite(value_and_prox!, F.f, t, s)
end
_checkvalue(F::SingularValue, X::AbstractMatrix) =
  _checkvalue(F.f, _checkbuffer(F.t, float(eltype(X)), (min(size(X)...),)))

# the conjugate of f ∘ σ is f* ∘ σ (Lewis, 1995)
_checkconjugate(F::SingularValue, Y::AbstractMatrix) = _checkconjugate(F.f, _checkspectrum(F, float(eltype(Y)), size(Y)))
# with the zero mask of the value, and the same domain
function _conjugate_value(F::SingularValue, Y::AbstractMatrix)
  _checkconjugate(F, Y)
  _checkblas(F, float(eltype(Y)))
  all(isfinite, Y) || return float(eltype(Y))(Inf)
  t, = _masked_singular_values!(F, Y)
  _conjugate_value(F.f, t)
end
