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
`F` is the finite symmetric matrices: the value is `NaN` at a matrix with a
`NaN` entry, and `Inf` at a matrix with an infinite entry, also for a bounded
`f`, and at a matrix that is not symmetric. For an `f` that is constant on all
vectors (the zero-level functions and `IndBallL0` listed under
[`SingularValue`](@ref)), `F` is the indicator of the symmetric matrices, and
its standalone value depends on a `NaN` entry only through symmetry: it is
`NaN` at an off-diagonal `NaN` of a matrix that is not symmetric by type
(`Symmetric`, `Hermitian`, `Diagonal`, `SymTridiagonal` or a
`SymmetricSparseIterate`, or a transpose or adjoint of one), also with an
infinite entry (E2: the `NaN` takes precedence), `Inf` at an infinite entry
otherwise, and otherwise `0` or `Inf` as the finite remainder is symmetric or
not. `value_and_prox!` keeps the
general rule. When a decomposition is needed, any
non-finite entry, including one in the lower triangle, gives an all-`NaN`
proximal output; `value_and_prox!` returns the same value. Only functions that count nonzeros mask eigenvalues with magnitude at
most `√ε·‖λ‖∞`. Other functions see the computed spectrum. Membership and
domain tests use `√ε·max(1, |bound|, ‖λ‖∞)`; sums keep their own scaled
tolerance. These decisions use scaled units when the spectrum overflows.
See [Functions of a spectrum](@ref). For a function that counts nonzeros,
`value_and_prox!` returns the value of `Spectral(f)` at the stored matrix: the
entries of the proximal spectrum at most `√ε` times the largest magnitude count
as zero, and when that decision is not certain (an entry within the rounding of
the threshold, a threshold that is not normal, a spectrum or output that is not
finite) the value is the one at the stored matrix itself, from a second
decomposition. A function with weights, bounds or normals
that differ between entries depends on their order and is rejected.

At a finite input whose spectrum overflows, the proximal output and the value
returned by `value_and_prox!` are computed in scaled units for unweighted
`NormL1`, `NormL1Nonneg`, `NormL2`, `NormLinf`, `SumLargest`,
`ElasticNet(λ, 0)`, `IndNonnegative`, `IndBallL0`, `ZeroFunction`, and boxes
whose bounds are all zero or infinite. Other functions keep the ordinary
result, which may be infinite, including `SqrNormL2`, bounded balls, weighted
penalties and compositions. For `IndBallL0` the value is that of the stored
matrix, whose entries in the original units can exceed the range of the type.

The standalone value at a finite input whose spectrum overflows is the value
at the spectrum of `X/2ᵏ`, scaled by `2^(dk)` for a function of degree `d`,
with the exponent applied last. This covers `NormL1`, `NormL2`, `NormLinf`,
`SumLargest`, `NormL1Nonneg`, `ElasticNet`, uncentered `SqrNormL2`,
`CubeNormL2`, the support functions of balls, boxes and simplices, the
conjugate of uncentered `SqrNormL2`, the conjugates of scaled `NormL1`,
`NormL2` and `NormLinf`, and `Scaled` and `Constrained` forms of these. The
support functions of a box with a linear constraint, such as that of the
capped simplex, keep the ordinary result (#648). The support functions of
boxes and simplices, which have no level that a scale folds into, keep the
ordinary result at a level outside the normal range of the precision of `X`.
Other functions, such as `NormLq`, centered or translated functions and
functions of the user, keep the ordinary result, which may be infinite. See
[Functions of a spectrum](@ref).

$_SPECTRAL_NOTE
"""
mutable struct Spectral{F<:AbstractFunction} <: AbstractFunction
  f::F
  eig::Any         # the eigendecomposition
  top::Any         # the workspace of the smallest eigenpairs, for the linear minimization oracle
  z::Scratch       # the eigenvalues of the result
  out::Scratch     # the result, for an output BLAS cannot write

  function Spectral{F}(f::F) where {F<:AbstractFunction}
    isproximable(f) || throw(ArgumentError("Spectral needs a proximable function, and $(nameof(typeof(f))) is not"))
    _ispermutationinvariant(f) ||
      throw(ArgumentError("Spectral needs a function that does not depend on the order of the entries, unlike this $(nameof(typeof(f)))"))
    new{F}(f, nothing, nothing, Scratch(), Scratch())
  end
end
Spectral(f::AbstractFunction) = Spectral{typeof(f)}(f)

isproximable(::Type{<:Spectral}) = true
_argument_type(::Type{<:Spectral}) = AbstractMatrix
isconvex(::Type{<:Spectral{F}}) where {F} = isconvex(F)
_isindicator(::Type{<:Spectral{F}}) where {F} = _isindicator(F)
_conjugate_isindicator(::Type{<:Spectral{F}}) where {F} = _conjugate_isindicator(F)

# The arrays that the proximal point of the eigenvalues will see: dense vectors of the
# working type, as the kernel's, whatever their contents.
function _spectral_check_arrays(s::Spectral, Y::AbstractMatrix, X::AbstractMatrix)
  n = checksquare(X)
  T = float(eltype(Y))
  ws = s.eig
  λ = T <: _BlasReal && ws isa SymEigenWorkspace{T} && ws.n == n ? ws.w : Vector{T}(undef, n)
  _checkbuffer(s.z, T, (n,)), λ
end

function _checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, s::Spectral, Y::AbstractMatrix, X::AbstractMatrix)
  _checkreal(s, X)
  _checkreal(s, Y)
  _checksize(Y, X)
  checksquare(X)
  _checkblas(s, float(eltype(Y)))
  z, λ = _spectral_check_arrays(s, Y, X)
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
    all(isfinite, X) || return _spectral_nonfinite_value(s, X, T)
    _issymmetric_tol(X) || return T(Inf)
  end
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  copyto!(ws.A, X)
  !_is_device_array(X) || all(isfinite, ws.A) || return _spectral_nonfinite_value(s, ws.A, T)
  !_is_device_array(X) || _issymmetric_tol(ws.A) || return T(Inf)
  λ = _eigvals!(ws)
  if _counts_nonzeros(s.f)
    if !isempty(λ)
      m = max(abs(λ[1]), abs(λ[end]))
      if !isfinite(m) || (m > zero(T) && _tol(T) * m <= floatmin(T))
        return _scaled_spectral_mask!(s, ws, X)
      end
    end
    _mask_eigenvalues!(λ)
  end
  if !isempty(λ) && !(_finite_value(λ[1]) && _finite_value(λ[end])) && _decided_on_spectrum(s.f)
    return _scaled_spectral_membership!(s, ws, X)
  end
  _decided_on_spectrum(s.f) && return _spectrum_value(s.f, λ, 0)
  v = convert(T, s.f(λ))
  # An overflowing spectrum: the value at the eigenvalues of X/2^k, when f has a recovery.
  isempty(λ) || (_finite_value(λ[1]) && _finite_value(λ[end])) || !_spectrum_scalable(s.f) ? v :
    _spectral_overflow_value!(s, ws, X, v)
end

# Functions of a spectrum whose value does not depend on the entries (E2, E4), the one place where
# SingularValue, Spectral, NuclearNorm's wrappers and the counting functions are classified.
# Explicit methods only, with the default "not established": no inner function is evaluated to
# infer constancy. The kind of g on spectra of length p, which are nonnegative when nonneg (singular
# values), is :zero when g is the zero function there (a zero level and no domain constraint, E4),
# :whole when g has the value of the indicator of all of ℝᵖ (0 at NaN entries alone, Inf at an
# infinite entry), and nothing otherwise. Weighted and centered functions, infinite levels and other
# functions with domain constraints have no method. These are called only after a finiteness test
# failed.
_constant_kind(g, p, nonneg::Bool) = nothing
_constant_kind(::ZeroFunction, p, nonneg::Bool) = :zero
_constant_kind(g::NormL1{<:Real, Nothing}, p, nonneg::Bool) = iszero(g.λ0) ? :zero : nothing
_constant_kind(g::Union{NormL2, NormLinf, CubeNormL2, SqrNormL2{<:Real, Nothing}}, p, nonneg::Bool) =
  iszero(g.λ) ? :zero : nothing
# SumLargest(k, λ) needs at least k entries: on a shorter spectrum it is not constant (N7)
_constant_kind(g::SumLargest, p, nonneg::Bool) = iszero(g.λ) && g.k <= p ? :zero : nothing
_constant_kind(g::ElasticNet{<:Real, Nothing}, p, nonneg::Bool) = iszero(g.λ1) && iszero(g.λ2) ? :zero : nothing
# The counting functions (#614): IndBallL0 with room for p nonzeros is the indicator of every
# matrix; NormL0, whose only parameter is its level, is the zero function at a zero level.
_constant_kind(g::IndBallL0, p, nonneg::Bool) = g.s >= p ? :whole : nothing
_constant_kind(g::NormL0, p, nonneg::Bool) = iszero(g.λ) ? :zero : nothing
# A counting function under a positive finite scale keeps its kind. On a box that holds every
# singular value (l ≤ 0, u = Inf), on a nonnegative spectrum, a constant counting function is
# constant too, and the box makes it the indicator of a set of real matrices: Inf at an infinite
# entry, as for the indicator of every matrix (maintainer decision on PR #384, item 2).
_constant_kind(f::Scaled, p, nonneg::Bool) =
  _counts_nonzeros(f.f) && isfinite(f.α) ? _constant_kind(f.f, p, nonneg) : nothing
_constant_kind(c::BoxConstrained, p, nonneg::Bool) =
  nonneg && _counts_nonzeros(c.f) && all(<=(0), c.box.l) && all(==(Inf), c.box.u) &&
  _constant_kind(c.f, p, nonneg) !== nothing ? :whole : nothing

# The standalone value at a non-finite X. For an inner function that is constant on ℝⁿ (_constant_kind),
# Spectral(g) is the indicator of the symmetric matrices, its domain, which E4 keeps:
# - in a matrix that is not symmetric by type, an off-diagonal NaN decides symmetry, so the value
#   depends on it (E2) and is NaN;
# - an infinite entry gives Inf, as an indicator of a set of real matrices (E3);
# - otherwise the NaN entries are on the diagonal, or in a matrix symmetric by type, and do not
#   decide membership (E2): the finite rest does.
# Otherwise the value is NaN at a NaN entry and Inf at an infinite one. The fused call reads the
# symmetric part and is not classified.
function _spectral_nonfinite_value(s::Spectral, X::AbstractMatrix, ::Type{T}) where {T}
  # The structure of f is not checked at a non-finite input (E1 there is #686).
  _constant_kind(s.f, size(X, 1), false) === nothing && return _nonfinite_value(X, T)
  bytype = _symmetric_by_type(X)
  !bytype && _offdiagonal_nan(X) && return T(NaN)
  any(isinf, X) && return T(Inf)
  (bytype || _issymmetric_tol(X)) ? zero(T) : T(Inf)
end
function _offdiagonal_nan(X::AbstractMatrix)
  @inbounds for j in axes(X, 2), i in axes(X, 1)
    i != j && isnan(X[i, j]) && return true
  end
  false
end

# Keep the signs of eigenvalues that are separated from numerical zero.
function _mask_eigenvalues!(λ::Vector{T}) where {T}
  c = isempty(λ) ? zero(T) : _tol(T) * max(abs(λ[1]), abs(λ[end]))
  @inbounds for j in eachindex(λ)
    abs(λ[j]) <= c && (λ[j] = zero(T))
  end
  λ
end

@noinline function _scaled_spectral_mask!(s::Spectral, ws, X::AbstractMatrix)
  invariant = _scale_invariant(s.f)
  decided = _decided_on_spectrum(s.f)
  # Bounds that do not scale with X need the spectrum in the original units.
  t = invariant || decided ? nothing : _buffer!(s.z, eltype(ws.w), length(ws.w))
  if !invariant && !decided
    length(t) == length(ws.w) || resize!(t, length(ws.w))
    copyto!(t, ws.w)
  end
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, _is_device_array(X) ? ws.A : X)
  λ = _mask_eigenvalues!(_eigvals!(ws))
  decided && return convert(eltype(λ), _spectrum_value(s.f, λ, k))
  invariant && return convert(eltype(λ), s.f(λ))
  @inbounds for j in eachindex(λ)
    iszero(λ[j]) && (t[j] = zero(eltype(λ)))
  end
  convert(eltype(λ), _spectrum_value(s.f, t, 0))
end

# Decide membership on the eigenvalues of X/2^k when original units overflow.
@noinline function _scaled_spectral_membership!(s::Spectral, ws, X::AbstractMatrix)
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, _is_device_array(X) ? ws.A : X)
  _spectrum_value(s.f, _eigvals!(ws), k)
end

# The proximal point of the eigenvalues receives the step, with arrays of the kernel's types,
# so that a function that picks a branch by storage (the conjugate of NormLinf) picks the kernel's.
# The eigenvalue buffer holds stale values, so it is passed as a placeholder input, whose
# values the checks must not read; it still shows the storage.
function _checkstep(op::O, s::Spectral, Y::AbstractMatrix, X::AbstractMatrix, γ::Real) where {O}
  z, λ = _spectral_check_arrays(s, Y, X)
  _checkstep(value_and_prox!, s.f, z, _StepInput(λ), γ)
end

function value_and_prox!(Y::AbstractMatrix, s::Spectral, X::AbstractMatrix, γ::Real)
  X = _spectral_argument(X)
  _checkwrite(value_and_prox!, s, Y, X)
  _checkstep(value_and_prox!, s, Y, X, γ)
  T = float(eltype(Y))
  if !all(isfinite, X)
    v = _nonfinite_value(X, T)                    # before Y, which may be X, is written
    fill!(Y, T(NaN))
    return v
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
    v = _scaled_wrapper_eigen_prox!(Y, s, X, γ, ws, z)
    # The output is rescaled to the original units, where an entry can exceed the range of T:
    # a function that counts nonzeros has the value at the stored matrix.
    return _counts_nonzeros(s.f) ? s(Y) : v
  end
  σ = _counts_nonzeros(s.f) ? _maximum_abs(λ) : zero(T)
  v = value_and_prox!(z, s.f, λ, γ)
  _rebuild_sym_signed!(Y, V, z, _blas_ready(Y) ? Y : _array_buffer!(s.out, T, (n, n)))
  if _counts_nonzeros(s.f)
    w = _stored_count_value!(s.f, z, σ, n)
    w === nothing && return s(Y)
    return w
  end
  convert(T, v)
end
prox!(Y::AbstractMatrix, s::Spectral, X::AbstractMatrix, γ::Real) =
  (X = _spectral_argument(X); value_and_prox!(Y, s, X, γ); Y)

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

####################################
#  values at an overflowing spectrum
####################################

# A positively homogeneous f of degree d has f(2ᵏ s) = 2^(dk) f(s): at the spectrum s of X 2⁻ᵏ,
# the value of F at X is the value of Scaled(f, 2^(dk)) at s, which the recovery of Scaled
# forms with the exponent applied last (an exponent-only _ProjectionScale through
# _scaled_value_by, so that a level the factor folds into is used at its own scale).
_homogeneity_degree(::AbstractFunction) = nothing
_homogeneity_degree(::Union{NormL1, NormL2, NormLinf, SumLargest, NormL1Nonneg, SLOPE}) = 1
_homogeneity_degree(::CubeNormL2) = 3
# support functions are positively homogeneous of degree one. Not the box with a linear
# constraint (the Ky Fan set): its kernel has no recovery here, and evaluating it at the spectrum
# of X/2^k would carry its known defects into overflowing values (#648, #642, #647).
_support_degree(::AbstractFunction) = nothing
_support_degree(::Union{IndBallL1, IndBallL2, IndSimplex, IndBox}) = 1
_support_degree(f::Scaled) = _support_degree(f.f)

_scaled_leaf(f) = f
_scaled_leaf(f::Scaled) = _scaled_leaf(f.f)
# Conjugates whose recovery has its own method: the dual-norm balls and the squared norm.
_conjugate_scalable(::AbstractFunction) = false
_conjugate_scalable(::Union{NormL1{<:Real, Nothing}, NormL2, NormLinf, SqrNormL2{<:Real, Nothing}}) = true
# Whether the value at an overflowing spectrum has a recovery, from the type: an unsupported
# function, such as a function of the user's, keeps the ordinary result and costs no second decomposition.
_spectrum_scalable(f::AbstractFunction) = _homogeneity_degree(f) !== nothing
_spectrum_scalable(::Union{ElasticNet, SqrNormL2{<:Real, Nothing}}) = true
_spectrum_scalable(f::Scaled) = _spectrum_scalable(f.f)
_spectrum_scalable(c::Conjugate) = _conjugate_spectrum_scalable(c.f)
_conjugate_spectrum_scalable(g) = _support_degree(g) !== nothing || _conjugate_scalable(_scaled_leaf(g))

_unit_scale(::Type{T}) where {T} = _ProjectionScale(one(float(T)) / 2, 1)

# The value b⋅f(2ᵏ s) at the spectrum s of X 2⁻ᵏ, where k > 0 because the spectrum of X overflowed
# and b is the product of the scales around f; nothing when f has no recovery (the caller keeps its
# own result). A zero or infinite scale keeps the ordinary result, so that exact zeros stay exact.
function _spectrum_scaled(f::AbstractFunction, s::AbstractVector, k::Int, b::_ProjectionScale)
  d = _homogeneity_degree(f)
  d === nothing && return nothing
  _spectrum_scaled_by(f, s, _ProjectionScale(b.m, b.e + d * k))
end
function _spectrum_scaled_by(f, s::AbstractVector{T}, a::_ProjectionScale) where {T}
  v = f(s)
  _spectrum_weighted(f) && return _weighted_spectrum_scaled(f, s, a, v)
  # A function that has neither a level to fold the factor into nor value parts (`_value_parts`), at
  # a level outside the normal range of T, has lost its value to underflow at the spectrum of
  # X/2^k: it keeps the ordinary result rather than a wrong finite one. (With normal levels and no
  # weights, a value at that spectrum, whose largest entry is at least one, is small only through
  # cancellation.) A function with value parts is recomputed by the recovery of Scaled.
  _scaled_small(v) && !_has_value_parts(typeof(f)) && !_spectrum_levels_normal(f, T) && return nothing
  _scaled_value_by(f, a, s, v)
end
# A weighted function (a nonzero level and weights) is a sum of products of the level, a weight
# and an entry of the spectrum. Such a product can lose its value to underflow although each
# factor is normal, so a small value at the spectrum of X/2^k is no bound on the value at X. Only
# a fold carries the factor into those products, and only into a normal level: each term is then a
# product rounded once (`_positive_product`), and the sum of these nonnegative terms is accurate
# when it is a normal number or infinite, or when every term is an exact zero. A normal value at
# the spectrum is accurate and scales with one rounding, also below the normal range, and an
# infinite one stays infinite under a scale above one. Otherwise the function keeps the ordinary
# result.
_spectrum_weighted(f) = false
_spectrum_weighted(f::Union{NormL1, NormL1Nonneg}) = f.w !== nothing && !iszero(f.λ0)
_spectrum_weighted(f::ElasticNet) = f.w !== nothing && !iszero(f.λ1)
@noinline function _weighted_spectrum_scaled(f, s::AbstractVector, a::_ProjectionScale, v)
  r = _apply_scale(a, v)
  _scaled_small(v) || ((isfinite(v) || !_scale_below_one(a)) && return r)
  g = _fold_level(f, _NormalScale(a))
  g === nothing && return nothing
  u = convert(typeof(r), _value_unchecked(g, s))
  !_scaled_small(u) || (iszero(u) && _weighted_terms_zero(f.w, s)) ? u : nothing
end
function _weighted_terms_zero(w, s::AbstractVector)
  @inbounds for i in eachindex(s)
    iszero(s[i]) || iszero(_entry(w, i)) || return false
  end
  true
end
# whether the levels that no fold takes are zero, infinite or normal numbers of T
_normal_level(::Type{T}, λ) where {T} = iszero(λ) || !isfinite(λ) || abs(convert(T, λ)) >= floatmin(T)
_spectrum_levels_normal(f, ::Type) = true
_spectrum_levels_normal(f::Union{SumLargest, CubeNormL2}, ::Type{T}) where {T} = _normal_level(T, f.λ)
_spectrum_levels_normal(f::SLOPE, ::Type{T}) where {T} = all(λ -> _normal_level(T, λ), f.λ)
# (called on the leaf of a scale chain: a positive scale does not change an indicator)
_support_levels_normal(f, ::Type) = true
_support_levels_normal(f::IndSimplex, ::Type{T}) where {T} = _normal_level(T, f.r)
_support_levels_normal(f::IndBox, ::Type{T}) where {T} =
  all(l -> _normal_level(T, l), f.l) && all(u -> _normal_level(T, u), f.u)
function _spectrum_scaled(f::Scaled, s::AbstractVector, k::Int, b::_ProjectionScale)
  g, c = _scaled_chain(f, b)
  c isa _ProjectionScale ? _spectrum_scaled(g, s, k, c) : nothing
end
# The two levels have degrees one and two: the parts are recovered separately and added.
function _spectrum_scaled(f::ElasticNet, s::AbstractVector, k::Int, b::_ProjectionScale)
  iszero(f.λ2) && return _spectrum_scaled_by(f, s, _ProjectionScale(b.m, b.e + k))
  iszero(f.λ1) && return _spectrum_scaled_by(f, s, _ProjectionScale(b.m, b.e + 2k))
  v1 = _spectrum_scaled_by(ElasticNet(f.λ1, zero(f.λ2), f.w), s, _ProjectionScale(b.m, b.e + k))
  v2 = _spectrum_scaled_by(ElasticNet(zero(f.λ1), f.λ2), s, _ProjectionScale(b.m, b.e + 2k))
  v1 === nothing || v2 === nothing ? nothing : v1 + v2
end
# (λ/2)‖x‖² is ElasticNet(0, λ), whose level the factor folds into.
function _spectrum_scaled(f::SqrNormL2{<:Real, Nothing}, s::AbstractVector, k::Int, b::_ProjectionScale)
  (iszero(f.λ) || !isfinite(f.λ)) && return nothing
  _spectrum_scaled(ElasticNet(zero(f.λ), f.λ), s, k, b)
end
# The conjugate of f0 is evaluated through the value of the conjugate of f0 (_conjugate_value), so
# that no Conjugate is formed unless a level is folded.
_spectrum_scaled(c::Conjugate, s::AbstractVector, k::Int, b::_ProjectionScale) = _conjugate_scaled(c.f, s, k, b)
function _conjugate_scaled(f0, s::AbstractVector{T}, k::Int, b::_ProjectionScale) where {T}
  g, a = _scaled_chain(f0, _unit_scale(T))
  a isa _ProjectionScale || return nothing
  _conjugate_spectrum_scaled(g, f0, s, k, b, a)
end
# A support function: degree one, the recovery of Scaled at the value of the conjugate. Every
# support function here is that of a set, whose indicator a positive scale does not change, so the
# value and the fold are those of the leaf g of the scale chain.
function _conjugate_spectrum_scaled(g::AbstractFunction, f0, s, k, b, _)
  T = eltype(s)
  d = _support_degree(g)
  d === nothing && return nothing
  a = _ProjectionScale(b.m, b.e + d * k)
  v = _conjugate_value(g, s)
  _scaled_small(v) && !_support_levels_normal(g, T) && return nothing
  r = _apply_scale(a, v)
  # the trigger of _scaled_value_by, whose recovery folds into the radius of a ball; only that
  # fold forms the Conjugate
  if g isa Union{IndBallL1, IndBallL2} &&
     ((!isfinite(r) && _scale_below_one(a)) || (_scaled_small(r) && !iszero(v)) || (_scaled_small(v) && _scale_above_one(a)))
    return _scaled_value_recover(Conjugate(g), a, s, v, r)
  end
  r
end
# The conjugate of λ times a norm under the scales a is the indicator of the dual ball of radius
# aλ, decided in the units of the spectrum, without forming aλ (the outer scales do not change an
# indicator). Only this branch, at an overflowing spectrum, decides on the spectrum: ordinary
# values keep the conjugate's own test.
_conjugate_spectrum_scaled(g::NormL1{<:Real, Nothing}, f0, s, k, b, a) = _conjugate_ball_scaled(s, g.λ0, a, k, Inf)
_conjugate_spectrum_scaled(g::NormL2, f0, s, k, b, a) = _conjugate_ball_scaled(s, g.λ, a, k, 2)
_conjugate_spectrum_scaled(g::NormLinf, f0, s, k, b, a) = _conjugate_ball_scaled(s, g.λ, a, k, 1)
function _conjugate_ball_scaled(s::AbstractVector{T}, λ, a::_ProjectionScale, k::Int, p) where {T}
  isfinite(λ) || return _ind(isinf(λ) && λ > 0, T)
  S = float(promote_type(T, typeof(λ), typeof(a.m)))
  iszero(λ) && return _spectrum_ball_scaled(s, zero(S), 0, k, p)
  mλ, eλ = frexp(S(λ))
  _spectrum_ball_scaled(s, S(a.m) * mλ, _ball_exponent(a.e + Int(eλ)), k, p)
end
# The exponent of a ball's bound reaches only shifts by ldexp, which saturates. An exponent beyond Int (a
# BigFloat chain) is brought to a magnitude of 3 * 2^61, beyond the range of every floating-point type here
# (at most 2^62) by more than the spectrum's own exponents, so that the shift saturates as it would unclamped.
@inline _ball_exponent(e::Int) = e
@inline _ball_exponent(e::Integer) = Int(clamp(e, -3 * 2^61, 3 * 2^61))
# (aλ/2 ‖⋅‖²)*(y) = ‖y‖²/(2aλ) is ElasticNet(0, 1/m) with the factor 2^-e, for aλ = m 2^e.
function _conjugate_spectrum_scaled(g::SqrNormL2{<:Real, Nothing}, f0, s, k, b, a)
  (g.λ > 0 && isfinite(g.λ)) || return nothing
  mλ, eλ = frexp(float(g.λ))
  m, e = _frexp(a.m * mλ)
  _spectrum_scaled(ElasticNet(zero(m), inv(m)), s, k, _ProjectionScale(b.m, b.e - a.e - Int(eλ) - Int(e)))
end
# Decided functions: the domain in the units of the spectrum first, then the value.
function _spectrum_scaled(f::NormL1Nonneg, s::AbstractVector{T}, k::Int, b::_ProjectionScale) where {T}
  _checkparams(f, s)
  iszero(_spectrum_value(IndNonnegative(), s, k)) || return T(Inf)
  # Negative rounding contributes zero; the scaled spectrum is private to the workspace.
  @inbounds for i in eachindex(s)
    s[i] = max(s[i], zero(T))
  end
  _spectrum_scaled_by(f, s, _ProjectionScale(b.m, b.e + k))
end
function _spectrum_scaled(c::BoxConstrained, s::AbstractVector{T}, k::Int, b::_ProjectionScale) where {T}
  _checkparams(c, s)
  iszero(_spectrum_value(c.box, s, k)) || return T(Inf)
  _spectrum_scaled(c.f, s, k, b)
end
function _spectrum_scaled(c::LinearConstrained, s::AbstractVector{T}, k::Int, b::_ProjectionScale) where {T}
  _checkparams(c, s)
  iszero(_spectrum_value(c.set, s, k)) || return T(Inf)
  _spectrum_scaled(c.f, s, k, b)
end

# The ordinary value v at an overflowing spectrum, or the recovered value.
@noinline function _spectrum_overflow_value(f, s::AbstractVector, k::Int, v)
  _spectrum_scalable(f) || return v
  r = _spectrum_scaled(f, s, k, _unit_scale(eltype(s)))
  r === nothing ? v : convert(typeof(v), r)
end

# The eigenvalues of X/2^k, as for decisions, for the recovery of a continuous value.
@noinline function _spectral_overflow_value!(s::Spectral, ws, X::AbstractMatrix, v)
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, _is_device_array(X) ? ws.A : X)
  _spectrum_overflow_value(s.f, _eigvals!(ws), k, v)
end
