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

`isconvex` and the indicator property come from `f`. At a matrix with a NaN
entry, a function that counts nonzeros, such as `NormL0` or `IndBallL0`, has
value NaN (E2), also when another entry is infinite. One that takes the same
value at every spectrum, as `NormL0(0)` and `IndBallL0(r)` with
`r ≥ minimum(size(X))` do, has value `0` at NaN entries alone and `Inf` at an
infinite entry. Otherwise an infinite entry gives `Inf`, which for a bounded `f`
such as `NormL0` is not yet the limit that E3 asks for, and the functions that
do not count nonzeros give `Inf` at a NaN entry too. When a
decomposition is needed, a non-finite input gives an all-`NaN` proximal output;
`value_and_prox!` returns the value of a counting function at that output, NaN,
or `0` when the function is constant, and `Inf` for the other functions.
Only functions that count nonzeros, such as `NormL0` and `IndBallL0`, treat
singular values at most `√ε σ₁` as zero. Other functions see the computed
spectrum. Membership and domain tests use `√ε·max(1, |bound|, σ₁)`; sum
constraints keep their own scaled tolerance. The zero mask is resolved in
scaled units when the largest singular value overflows or its threshold is
not normal. See [Functions of a spectrum](@ref).

`value_and_prox!` returns `f` at the proximal spectrum. For a function that
counts nonzeros it returns the value of `SingularValue(f)` at the stored matrix:
the entries of the proximal spectrum at most `√ε` times the largest count as
zero, as in the value at `Y`. When that decision is not certain, because an
entry lies within the rounding of the zero threshold, the threshold is not
normal, or the spectrum or the output is not finite, the value is the one at
the stored matrix itself, from a second decomposition. Re-evaluation at `Y`
then returns the same value. For the other functions re-evaluation agrees up to
rounding. The value of `Conjugate(SingularValue(f))` is `f*` at the computed
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

# A non-finite entry gives Inf, or for a counting function the value of
# _count_nonfinite_value. Singular values within rounding of zero count as zero
# for counting functions. An overflowing spectrum decides membership in scaled
# units.
function (F::SingularValue)(X::AbstractMatrix)
  _checkreal(F, X)
  T = float(eltype(X))
  _checkblas(F, T)
  if _is_device_array(X)
    ws = _svd_workspace(F.ws, T, size(X)...)
    F.ws = ws
    copyto!(ws.A, X)
    all(isfinite, ws.A) || return _singular_nonfinite_value(F.f, ws.A, T)
  else
    all(isfinite, X) || return _singular_nonfinite_value(F.f, X, T)
  end
  t, scaled, k = _masked_singular_values!(F, X; copied = _is_device_array(X))
  _decided_on_spectrum(F.f) && return _spectrum_value(F.f, scaled === nothing ? t : scaled, k)
  _counts_nonzeros(F.f) && _scale_invariant(F.f) && scaled !== nothing && return convert(T, F.f(scaled))
  convert(T, F.f(t))
end

# Whether a function that counts nonzeros takes the same value, zero, at every
# spectrum of length k (E4 first): the zero level of NormL0, and IndBallL0 with
# room for k nonzeros. IndBallL0, scaled and box-constrained ones are in
# constrained_nonconvex.jl, after their types.
_count_constant(f, k::Int) = false
_count_constant(f::NormL0, k::Int) = iszero(f.λ)

# The value at a matrix A with a non-finite entry. For a counting function, a NaN
# entry gives NaN (E2) unless the function is constant, and an infinite entry
# gives Inf; NaN takes precedence when the value depends on it. A constant one is
# Inf at an infinite entry and zero at NaN entries alone. Other functions give Inf.
function _count_nonfinite_value(constant::Bool, A::AbstractArray, ::Type{T}) where {T}
  constant && return any(isinf, A) ? T(Inf) : zero(T)
  any(isnan, A) ? T(NaN) : T(Inf)
end
_singular_nonfinite_value(f, A::AbstractMatrix, ::Type{T}) where {T} =
  _counts_nonzeros(f) ? _count_nonfinite_value(_count_constant(f, min(size(A)...)), A, T) : T(Inf)
# The value at an all-NaN output (O2): zero for a constant counting function, NaN
# for another counting function, and Inf for the other functions.
_singular_nan_output_value(f, k::Int, ::Type{T}) where {T} =
  _counts_nonzeros(f) ? (_count_constant(f, k) ? zero(T) : T(NaN)) : T(Inf)

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

# The value of a function that counts nonzeros at the stored output Y = U Diagonal(t) Vᵀ
# (O2), from the proximal spectrum t, the largest magnitude σ of the spectrum before the
# proximal step, and the larger dimension p. The standalone value reads the singular
# (or eigen-) values of Y, whose zero mask differs from the one of t only through the
# rounding of the reconstruction and of the decomposition of Y. Returns nothing when t
# cannot decide the value and the standalone value at Y must be used:
#   - the spectrum or the output is not finite or may overflow (|Yᵢⱼ| ≤ max|t|);
#   - the zero threshold c = √ε·max|t| is not normal, where the standalone mask is resolved
#     in scaled units, and rounding of Y onto the subnormal grid can change its rank;
#   - a nonzero entry of t lies within δ = 4·p·ε·max|t| of c. The singular values of Y differ
#     from |t| by the error of the reconstruction (O(ε·max|t|)) plus the backward error of
#     the decomposition of Y (O(p·ε·max|t|)); 4·p·ε is the bound of _spectral_rescale!.
#     Measured on random matrices (100 draws for sizes up to 300, 3 at 1000, Float32 and
#     Float64), the singular values moved by at most 21ε·max|t|.
# An entry that is exactly zero is not tested: it is a direction that Y does not have, and the
# value the standalone computation gives there is rounding noise. In Gaussian matrices of
# size up to 1000 (Float32 and Float64, at least 3 draws each) it was at most 1.6ε·max|t|,
# against c = 2.9e3·ε·max|t| in Float32 and 6.7e7·ε·max|t| in Float64. Testing it would send
# every call with a zero in t to the second decomposition once δ reaches c (Float32, p ≥ 723),
# which took 1.5 times as long at p = 1000. This rests on those measurements, for dimensions p
# well below 1/√ε (2.9e3 in Float32, 6.7e7 in Float64, where 4·p·ε reaches the threshold only
# at a quarter of that); it is not a certificate for arbitrary dimensions.
# Otherwise the value is the standalone evaluation on t with the entries at most c set to
# zero; t is overwritten, no longer needed after the reconstruction.
function _stored_count_value!(f, t::AbstractVector{T}, σ::T, p::Int) where {T}
  m = _maximum_abs(t)
  max(m, σ) <= floatmax(T) / 2 || return nothing
  if m > zero(T)
    c = _tol(T) * m
    c <= floatmin(T) && return nothing
    δ = 4 * p * eps(T) * m
    near = false
    @inbounds for j in eachindex(t)
      a = abs(t[j])
      near |= !iszero(a) & (abs(a - c) <= δ)
      a <= c && (t[j] = zero(T))
    end
    near && return nothing
  end
  convert(T, _decided_on_spectrum(f) ? _spectrum_value(f, t, 0) : f(t))
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

# The proximal point of the singular values receives the step, with arrays of the kernel's
# types (see Spectral); the spectrum is a placeholder input, whose values the checks must not read.
function _checkstep(op::O, F::SingularValue, Y::AbstractMatrix, X::AbstractMatrix, γ::Real) where {O}
  t, s = _singular_check_arrays(F, Y, X)
  _checkstep(value_and_prox!, F.f, t, _StepInput(s), γ)
end

function value_and_prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real)
  X = _symmetric_argument(Y, X)
  _checkwrite(value_and_prox!, F, Y, X)
  _checkstep(value_and_prox!, F, Y, X, γ)
  T = float(eltype(Y))
  if !all(isfinite, X)
    fill!(Y, T(NaN))
    return _singular_nan_output_value(F.f, min(size(X)...), T)
  end
  ws = _svd_workspace(F.ws, T, size(X)...)
  F.ws = ws
  copyto!(ws.A, X)
  s, U, Vt = _svd!(ws)
  t = _vector_buffer(F.t, T, length(s))
  F.t = t
  if _scales_spectral_prox(F.f) && _spectral_overflow(s, X, ws)
    v = _scaled_wrapper_svd_prox!(Y, F, X, γ, ws, t)
    # The output is rescaled to the original units, where an entry can exceed the range of T:
    # a function that counts nonzeros has the value at the stored matrix.
    return _counts_nonzeros(F.f) ? F(Y) : v
  end
  σ = _counts_nonzeros(F.f) ? _maximum_abs(s) : zero(T)
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
  if _counts_nonzeros(F.f)
    w = _stored_count_value!(F.f, t, σ, max(size(X)...))
    w === nothing && return F(Y)
    return w
  end
  convert(T, v)
end
prox!(Y::AbstractMatrix, F::SingularValue, X::AbstractMatrix, γ::Real) = (X = _symmetric_argument(Y, X); value_and_prox!(Y, F, X, γ); Y)

function _checkspectrum(F::SingularValue, ::Type{T}, dims::Tuple) where {T}
  ws = F.ws
  k = min(dims...)
  T <: _BlasReal && ws isa SVDWorkspace{T} && (ws.m, ws.n) == dims ? ws.s : Vector{T}(undef, k)
end
# The arrays that the proximal point of the singular values will see: dense vectors of the
# working type, as the kernel's, whatever their contents.
function _singular_check_arrays(F::SingularValue, Y::AbstractMatrix, X::AbstractMatrix)
  T = float(eltype(Y))
  s = _checkspectrum(F, T, size(X))
  _checkbuffer(F.t, T, size(s)), s
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, F::SingularValue, Y::AbstractMatrix, X::AbstractMatrix)
  _checkreal(F, X)
  _checkreal(F, Y)
  _checksize(Y, X)
  _checkblas(F, float(eltype(Y)))
  t, s = _singular_check_arrays(F, Y, X)
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
