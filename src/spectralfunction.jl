####################################
#
#  Functions of the eigenvalues of a symmetric matrix
#
####################################

"""
    Spectral(f)

The function `F(X) = f(λ(X))` of a symmetric matrix `X` through its eigenvalues
`λ(X)`, for a proximable function `f` of a vector that does not depend on the
order of the entries. Its proximal operator acts on the eigenvalues: for
`X = V diag(λ) V'`, `prox_{γF}(X) = V diag(prox_{γf}(λ)) V'` (Lewis, 1996), and
`F` is convex when `f` is. For example:

- `Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))` is the indicator of
  the Fantope `{X : 0 ⪯ X ⪯ I, tr(X) = k}`, the convex hull of the projections of
  rank `k`, as in sparse principal component analysis (Vu, Cho, Lei and Rohe,
  2013);
- `Spectral(SumLargest(k))` is the
  sum of the `k` largest eigenvalues, the support function of the Fantope;
- `Spectral(IndBox(l, u))` is the indicator of the matrices with eigenvalues in
  `[l, u]`;
- `Spectral(IndBinary(0, 1))` is the indicator of the orthogonal projection
  matrices, whose eigenvalues are zero or one;
- `Spectral(IndSimplex(1))` is the indicator of the density matrices,
  `{X ⪰ 0 : tr(X) = 1}`;
- `Spectral(NormL1(λ))` is `λ` times the trace norm, the nuclear norm of a
  symmetric matrix.

`Spectral(IndNonnegative())` is [`IndPSD`](@ref)`()`, and
`Spectral(NormL1Nonneg(λ))` is [`TracePSD`](@ref)`(λ)`. The proximal operator
reads the upper triangle of its input, which should be symmetric. The domain of
`F` is the finite symmetric matrices: the value is `Inf` at a matrix with a
non-finite entry, also for a bounded `f`, and at a matrix that is not
symmetric. When a decomposition is needed, any non-finite entry, including one in the
lower triangle, gives an all-`NaN` proximal output; `value_and_prox!` returns
zero for an indicator and `NaN` otherwise. Membership in `IndBox`, `IndBallLinf` and `IndNonnegative` is
decided with the tolerance `√ε·max(1, |bound|, ‖λ‖∞)`, relative to the
spectrum, since computed eigenvalues carry errors of order `ε‖λ‖∞`: a
positive semidefinite matrix of low rank and any scale lies in
`Spectral(IndBox(0, Inf))`, as it does in `IndPSD()`. When an extreme
eigenvalue overflows, membership in `IndBallL1`, `IndBallL2`, `IndBallLinf`,
`IndBox` and `IndNonnegative` is decided on the eigenvalues of the matrix
scaled to unit range, with the tolerance of the original units. A
function with weights, bounds or normals
that differ between entries depends on their order and is rejected.

At a finite input whose spectrum overflows, the proximal output and the value
returned by `value_and_prox!` are computed in scaled units for unweighted
`NormL1`, `NormL1Nonneg`, `NormL2`, `NormLinf`, `SumLargest`,
`ElasticNet(λ, 0)`, `IndNonnegative`, `IndBallL0`, `ZeroFunction`, and boxes
whose bounds are all zero or infinite. Other functions keep the ordinary
result, which may be infinite, including `SqrNormL2`, bounded balls, weighted
penalties and compositions. This rule does not change the standalone value.

$_SPECTRAL_NOTE
"""
mutable struct Spectral{F<:AbstractFunction} <: AbstractFunction
  f::F
  eig::Any         # the eigendecomposition
  z::Scratch       # the eigenvalues of the result
  out::Scratch     # the result, for an output BLAS cannot write

  function Spectral{F}(f::F) where {F<:AbstractFunction}
    isproximable(f) || throw(ArgumentError("Spectral needs a proximable function, and $(nameof(typeof(f))) is not"))
    _ispermutationinvariant(f) ||
      throw(ArgumentError("Spectral needs a function that does not depend on the order of the entries, unlike this $(nameof(typeof(f)))"))
    new{F}(f, nothing, Scratch(), Scratch())
  end
end
Spectral(f::AbstractFunction) = Spectral{typeof(f)}(f)

isproximable(::Type{<:Spectral}) = true
_argument_type(::Type{<:Spectral}) = AbstractMatrix
isconvex(::Type{<:Spectral{F}}) where {F} = isconvex(F)
_isindicator(::Type{<:Spectral{F}}) where {F} = _isindicator(F)
_conjugate_isindicator(::Type{<:Spectral{F}}) where {F} = _conjugate_isindicator(F)

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, s::Spectral, Y::AbstractMatrix, X::AbstractMatrix)
  _checkreal(s, X)
  _checkreal(s, Y)
  _checksize(Y, X)
  n = checksquare(X)
  T = float(eltype(Y))
  ws = s.eig
  _checkblas(s, T)
  λ = T <: _BlasReal && ws isa SymEigenWorkspace{T} && ws.n == n ? ws.w : Vector{T}(undef, n)
  z = _checkbuffer(s.z, T, (n,))
  _checkwrite(value_and_prox!, s.f, z, λ)
end

# The domain is the finite matrices: a non-finite entry gives Inf, before the
# symmetry test and any decomposition.
function (s::Spectral)(X::AbstractMatrix)
  _checkreal(s, X)
  T = float(eltype(X))
  _checkblas(s, T)
  n = checksquare(X)
  if !_is_device_array(X)
    all(isfinite, X) || return T(Inf)
    _issymmetric_tol(X) || return T(Inf)
  end
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  copyto!(ws.A, X)
  !_is_device_array(X) || all(isfinite, ws.A) || return T(Inf)
  !_is_device_array(X) || _issymmetric_tol(ws.A) || return T(Inf)
  λ = _eigvals!(ws)
  if !isempty(λ) && !(_finite_value(λ[1]) && _finite_value(λ[end])) && _scales_membership(s.f)
    return _scaled_spectral_membership!(s, ws, X)
  end
  _relative_membership(s.f) && return _scaled_membership(s.f, λ, 0)
  convert(T, s.f(λ))
end

# the balls and boxes at an overflowing spectrum, on the eigenvalues of X/2^k
@noinline function _scaled_spectral_membership!(s::Spectral, ws, X::AbstractMatrix)
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, _is_device_array(X) ? ws.A : X)
  _scaled_membership(s.f, _eigvals!(ws), k)
end

function value_and_prox!(Y::AbstractMatrix, s::Spectral, X::AbstractMatrix, γ::Real)
  _checkwrite(value_and_prox!, s, Y, X)
  T = float(eltype(Y))
  if !all(isfinite, X)
    fill!(Y, T(NaN))
    return _isindicator(s) ? zero(T) : T(NaN)
  end
  n = checksquare(X)
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  copyto!(ws.A, X)
  λ, V = _eigen!(ws)
  # the eigenvalues of the result, exactly as many as those of X
  z = _buffer!(s.z, T, n)
  length(z) == n || resize!(z, n)
  if _scales_spectral_prox(s.f) && _spectral_overflow(λ, X, ws, s.out)
    return _scaled_wrapper_eigen_prox!(Y, s, X, γ, ws, z)
  end
  v = value_and_prox!(z, s.f, λ, γ)
  _rebuild_sym_signed!(Y, V, z, _blas_ready(Y) ? Y : _array_buffer!(s.out, T, (n, n)))
  convert(T, v)
end
prox!(Y::AbstractMatrix, s::Spectral, X::AbstractMatrix, γ::Real) =
  (value_and_prox!(Y, s, X, γ); Y)

# Y = Σⱼ zⱼ vⱼvⱼ' for the eigenvectors vⱼ in the columns of V, through B, which
# is Y when BLAS can write Y and an n × n buffer otherwise; destroys V. The proximal
# operator of a function that does not depend on the order of the entries keeps
# the ascending order of the eigenvalues, so the negative zⱼ come first and the
# positive ones last: two products with syrk, of the columns scaled by √|zⱼ|, in
# O(n²k) for k nonzero zⱼ. Otherwise, rank-one updates.
function _rebuild_sym_signed!(Y::AbstractMatrix, V::Matrix{T}, z::AbstractVector{T}, B::StridedMatrix{T}) where {T<:_BlasReal}
  n = size(V, 1)
  size(B) == (n, n) || throw(DimensionMismatch("the output must be $n × $n"))
  if issorted(z)
    neg = 0                                   # z[1:neg] < 0 < z[pos:n]
    while neg < n && z[neg + 1] < 0
      neg += 1
    end
    pos = n + 1
    while pos > neg + 1 && z[pos - 1] > 0
      pos -= 1
    end
    @inbounds for j in Iterators.flatten((1:neg, pos:n))
      view(V, :, j) .*= sqrt(abs(z[j]))
    end
    β = zero(T)
    if pos <= n
      BLAS.syrk!('U', 'N', one(T), view(V, :, pos:n), β, B)
      β = one(T)
    end
    if neg >= 1
      BLAS.syrk!('U', 'N', -one(T), view(V, :, 1:neg), β, B)
      β = one(T)
    end
    iszero(β) && fill!(B, zero(T))
  else
    fill!(B, zero(T))
    @inbounds for j in 1:n
      iszero(z[j]) || BLAS.syr!('U', z[j], view(V, :, j), B)
    end
  end
  _copy_upper_to_lower!(B)
  B === Y || copyto!(Y, B)
  Y
end

# whether f(x) does not depend on the order of the entries of x: no weights,
# bounds, centers or normals that vary between the entries
_ispermutationinvariant(::AbstractFunction) = false
_ispermutationinvariant(::Union{ZeroFunction, NormL0, MCP, SCAD, IndNonnegative, NormL2, CubeNormL2, NormLinf,
                                IndBallL0, IndBallL1, IndBallL2, IndSimplex, SumLargest}) = true
_ispermutationinvariant(::Union{NormL1{<:Real, Nothing}, NormL1Nonneg{<:Real, Nothing},
                                ElasticNet{<:Real, Nothing}, SqrNormL2{<:Real, Nothing}}) = true
_ispermutationinvariant(::IndBox{<:Real, <:Real}) = true
_ispermutationinvariant(::IndBinary{<:Real, <:Real}) = true
_ispermutationinvariant(::Union{IndHyperplane{<:Real}, IndHalfspace{<:Real}, IndHyperslab{<:Real}}) = true
_ispermutationinvariant(s::Union{Scaled, Conjugate}) = _ispermutationinvariant(s.f)
_ispermutationinvariant(c::BoxConstrained) = _ispermutationinvariant(c.f) && _ispermutationinvariant(c.box)
_ispermutationinvariant(c::LinearConstrained) = _ispermutationinvariant(c.f) && _ispermutationinvariant(c.set)

# Scaling is a capability of these proximal operations, separate from admission.
_scales_spectral_prox(::AbstractFunction) = false
_scales_spectral_prox(::Union{IndNonnegative, IndBallL0, ZeroFunction, NormL2, NormLinf, SumLargest,
                             NormL1{<:Real, Nothing}, NormL1Nonneg{<:Real, Nothing}}) = true
_scales_spectral_prox(f::ElasticNet{<:Real, Nothing}) = iszero(f.λ2)
_cone_bound(x::Real) = iszero(x) || isinf(x)
_cone_bound(x::AbstractArray) = all(_cone_bound, x)
_scales_spectral_prox(f::IndBox) = _cone_bound(f.l) && _cone_bound(f.u)

function _scaled_spectrum_prox!(z, f::Union{IndNonnegative, IndBallL0, ZeroFunction, IndBox}, x, γ, e)
  value_and_prox!(z, f, x, γ)
end

_spectral_level(f::Union{NormL1, NormL1Nonneg}) = f.λ0
_spectral_level(f::ElasticNet) = f.λ1
_spectral_level(f::Union{NormL2, NormLinf, SumLargest}) = f.λ

_spectral_unit(f::Union{NormL1, ElasticNet}, ::Type{T}) where {T} = NormL1(one(T))
_spectral_unit(f::NormL1Nonneg, ::Type{T}) where {T} = NormL1Nonneg(one(T))
_spectral_unit(f::NormL2, ::Type{T}) where {T} = NormL2(one(T))

function _scaled_spectrum_prox!(z::Vector{T}, f::Union{NormL1, NormL1Nonneg, ElasticNet, NormL2}, x, γ, e) where {T}
  λ = convert(T, _spectral_level(f))
  c = iszero(λ) ? zero(T) : _scaled_product(convert(T, γ), λ, -e)
  v = value_and_prox!(z, _spectral_unit(f, T), x, c)
  iszero(λ) ? zero(T) : _scaled_product(λ, v, e)
end

function _scaled_spectrum_prox!(z::Vector{T}, f::NormLinf, x, γ, e) where {T}
  λ = convert(T, f.λ)
  c = iszero(λ) ? zero(T) : _scaled_product(convert(T, γ), λ, -e)
  if norm(x, 1) <= c
    fill!(z, zero(T))
    return zero(T)
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, T, length(x)), x, eachindex(IndexLinear(), x), c, abs)
  z .= sign.(x) .* min.(abs.(x), τ)
  iszero(λ) ? zero(T) : _scaled_product(λ, τ, e)
end

function _scaled_spectrum_prox!(z::Vector{T}, f::SumLargest, x, γ, e) where {T}
  _sumlargest_prox!(z, f, x, γ, Val(false), -e)
  v = _buffer!(f.sums, T, length(z))
  copyto!(v, z)
  f.k < length(v) && _kth_largest!(view(v, 1:length(v)), f.k)
  iszero(f.λ) ? zero(T) : _scaled_product(convert(T, f.λ), _sumlargest_sum(v, f.k, one(T)), e)
end

@noinline function _scaled_wrapper_eigen_prox!(Y, s::Spectral, X, γ, ws::SymEigenWorkspace{T}, z) where {T}
  # The overflow check keeps the device input separate from the eigenvectors.
  A = _is_device_array(X) ? _array_buffer!(s.out, T, size(X)) : X
  e = _magnitude_exponent(convert(T, _maximum_abs(A)))
  _scaled_spectral_input!(ws.A, A)
  λ, V = _eigen!(ws)
  σ = _maximum_abs(λ)
  v = _scaled_spectrum_prox!(z, s.f, λ, γ, e)
  # SumLargest can increase the magnitude of negative components.
  a = _maximum_abs(z)
  σ = isfinite(a) ? max(σ, a) : σ
  n = size(X, 1)
  B = Y isa StridedMatrix{T} && _blas_ready(Y) ? Y : _array_buffer!(s.out, T, (n, n))
  _rebuild_sym_signed!(B, V, z, B)
  _spectral_rescale!(Y, B, e, n, σ)
  convert(T, v)
end
