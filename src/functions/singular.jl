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
bounded `f` such as `NormL0`. When a decomposition is needed, a non-finite
input gives an all-`NaN` proximal output; `value_and_prox!` returns `Inf`.
Only functions that count nonzeros, such as `NormL0` and `IndBallL0`, treat
singular values at most `√ε σ₁` as zero. Other functions see the computed
spectrum. Membership and domain tests use `√ε·max(1, |bound|, σ₁)`; sum
constraints keep their own scaled tolerance. The zero mask is resolved in
scaled units when the largest singular value overflows or its threshold is
not normal. See [Functions of a spectrum](@ref).

`value_and_prox!` returns `f` at the exact proximal spectrum. Re-evaluation at
the returned matrix agrees only when retained entries exceed the numerical-zero
threshold. The value of `Conjugate(SingularValue(f))` is `f*` at the computed
singular values of `Y`, with the same membership tolerance and finite domain.
Whether another `f` evaluates an overflowing spectrum correctly depends on `f`.

At a finite input whose spectrum overflows, the proximal output and the value
returned by `value_and_prox!` are computed in scaled units for unweighted
`NormL1`, `NormL1Nonneg`, `NormL2`, `NormLinf`, `SumLargest`,
`ElasticNet(λ, 0)`, `IndNonnegative`, `IndBallL0`, `ZeroFunction`, and boxes
whose bounds are all zero or infinite. Other functions keep the ordinary
result, which may be infinite, including `SqrNormL2`, bounded balls, weighted
penalties and compositions. This rule does not change the standalone value.

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
function _singular_values!(F::SingularValue, X::AbstractMatrix; copied::Bool = false)
  T = float(eltype(X))
  _checkblas(F, T)
  ws = _svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copied || copyto!(ws.A, X)
  s, = _svd!(ws; vectors = false)
  s
end

# The domain is the finite matrices: a non-finite entry gives Inf. Singular
# values within rounding of zero count as zero for counting functions. An
# overflowing spectrum decides membership in scaled units.
function (F::SingularValue)(X::AbstractMatrix)
  _checkreal(F, X)
  T = float(eltype(X))
  _checkblas(F, T)
  if _is_device_array(X)
    ws = _svd_workspace(F.ws, T, size(X)...)
    F.ws = ws
    copyto!(ws.A, X)
    all(isfinite, ws.A) || return T(Inf)
  else
    all(isfinite, X) || return T(Inf)
  end
  t, scaled, k = _masked_singular_values!(F, X; copied = _is_device_array(X))
  _decided_on_spectrum(F.f) && return _spectrum_value(F.f, scaled === nothing ? t : scaled, k)
  _counts_nonzeros(F.f) && _scale_invariant(F.f) && scaled !== nothing && return convert(T, F.f(scaled))
  convert(T, F.f(t))
end

# Only counting functions mask numerical zeros. At extreme scales, return
# both the original units and the resolved spectrum of X scaled by 2^-k.
function _masked_singular_values!(F::SingularValue, X::AbstractMatrix; copied::Bool = false, mask::Bool = _counts_nonzeros(F.f))
  T = float(eltype(X))
  s = _singular_values!(F, X; copied)
  !mask && (isempty(s) || _finite_value(s[1])) && return s, nothing, 0
  t = _vector_buffer(F.t, T, length(s))
  F.t = t
  if !isempty(s) && (!_finite_value(s[1]) || (s[1] > zero(T) && _tol(T) * s[1] <= floatmin(T)))
    return _scaled_singular_mask!(F, X, s, t, mask)
  end
  c = isempty(s) ? zero(T) : _tol(T) * s[1]       # relative, so that F scales with X
  @inbounds for j in eachindex(s)
    t[j] = s[j] <= c ? zero(T) : s[j]
  end
  t, nothing, 0
end

@noinline function _scaled_singular_mask!(F::SingularValue, X::AbstractMatrix, s::Vector{T}, t::Vector{T}, mask::Bool) where {T}
  copyto!(t, s)
  ws = _svd_workspace(F.ws, T, size(X)...)
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, _is_device_array(X) ? ws.A : X)
  s, = _svd!(ws; vectors = false)
  c = _tol(T) * s[1]
  @inbounds for j in eachindex(s)
    if mask && s[j] <= c
      t[j] = zero(T)
      s[j] = zero(T)
    end
  end
  t, s, k
end

function value_and_prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real)
  X = _symmetric_argument(Y, X)
  _checkwrite(value_and_prox!, F, Y, X)
  T = float(eltype(Y))
  if !all(isfinite, X)
    fill!(Y, T(NaN))
    return T(Inf)
  end
  ws = _svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = _svd!(ws)
  t = _vector_buffer(F.t, T, length(s))
  F.t = t
  if _scales_spectral_prox(F.f) && _spectral_overflow(s, X, ws)
    return _scaled_wrapper_svd_prox!(Y, F, X, γ, ws, t)
  end
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
prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real) = (X = _symmetric_argument(Y, X); value_and_prox!(Y, F, X, γ); Y)

function _checkspectrum(F::SingularValue, ::Type{T}, dims::Tuple) where {T}
  ws = F.ws
  k = min(dims...)
  T <: _BlasReal && ws isa SVDWorkspace{T} && (ws.m, ws.n) == dims ? ws.s : Vector{T}(undef, k)
end
function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, F::SingularValue, Y::AbstractMatrix, X::AbstractMatrix)
  _checkreal(F, X)
  _checkreal(F, Y)
  _checksize(Y, X)
  T = float(eltype(Y))
  _checkblas(F, T)
  s = _checkspectrum(F, T, size(X))
  t = _checkbuffer(F.t, T, size(s))
  _checkwrite(value_and_prox!, F.f, t, s)
end
_checkvalue(F::SingularValue, X::AbstractMatrix) = (_checkreal(F, X); _checkvalue(F.f, _checkbuffer(F.t, float(eltype(X)), (min(size(X)...),))))

# the conjugate of f ∘ σ is f* ∘ σ (Lewis, 1995)
_checkconjugate(F::SingularValue, Y::AbstractMatrix) = (_checkreal(F, Y); _checkconjugate(F.f, _checkspectrum(F, float(eltype(Y)), size(Y))))
# The conjugate has its own counting trait and the same domain.
function _conjugate_value(F::SingularValue, Y::AbstractMatrix)
  _checkconjugate(F, Y)
  _checkblas(F, float(eltype(Y)))
  all(isfinite, Y) || return float(eltype(Y))(Inf)
  f = Conjugate(F.f)
  t, scaled, k = _masked_singular_values!(F, Y; mask = _counts_nonzeros(f))
  _decided_on_spectrum(f) && return _spectrum_value(f, scaled === nothing ? t : scaled, k)
  _conjugate_value(F.f, t)
end

@noinline function _scaled_wrapper_svd_prox!(Y, F::SingularValue, X, γ, ws::SVDWorkspace{T}, t) where {T}
  # The overflow check has restored the device input in ws.A.
  A = _is_device_array(X) ? ws.A : X
  e = _magnitude_exponent(convert(T, _maximum_abs(A)))
  _scaled_spectral_input!(ws.A, A)
  s, U, Vt = _svd!(ws)
  σ = _maximum_abs(s)
  v = _scaled_spectrum_prox!(t, F.f, s, γ, e)
  # SumLargest can increase the magnitude of negative components.
  a = _maximum_abs(t)
  σ = isfinite(a) ? max(σ, a) : σ
  k = 0
  @inbounds for j in eachindex(t)
    iszero(t[j]) || (k = j)
  end
  @inbounds for j in 1:k, i in axes(U, 1)
    U[i, j] *= t[j]
  end
  B = Y isa StridedMatrix{T} && _blas_ready(Y) ? Y : ws.A
  _mul_lowrank!(B, U, Vt, k, ws)
  _spectral_rescale!(Y, B, e, max(size(X)...), σ)
  convert(T, v)
end
