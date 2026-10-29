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
reads the upper triangle of its input, which should be symmetric, and the value
is `Inf` at a matrix that is not. A function with weights, bounds or normals
that differ between entries depends on their order and is rejected.

$_SPECTRAL_NOTE
"""
mutable struct Spectral{F<:AbstractFunction} <: AbstractFunction
  f::F
  eig::Any         # the eigendecomposition
  z::Scratch       # the eigenvalues of the result
  out::Scratch     # the result, for an output that is not strided

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
  _checksize(Y, X)
  n = checksquare(X)
  T = float(eltype(Y))
  ws = s.eig
  λ = T <: _BlasReal && ws isa SymEigenWorkspace{T} && ws.n == n ? ws.w : Vector{T}(undef, n)
  z = _checkbuffer(s.z, T, (n,))
  _checkwrite(value_and_prox!, s.f, z, λ)
end

# The domain is the finite matrices: a non-finite entry gives Inf, before the
# symmetry test and any decomposition.
function (s::Spectral)(X::AbstractMatrix)
  T = float(eltype(X))
  n = checksquare(X)
  all(isfinite, X) || return T(Inf)
  _issymmetric_tol(X) || return T(Inf)
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  copyto!(ws.A, X)
  λ = _eigvals!(ws)
  if !isempty(λ) && !(_finite_value(λ[1]) && _finite_value(λ[end])) && _scales_membership(s.f)
    return _scaled_spectral_membership!(s, ws, X)
  end
  convert(T, s.f(λ))
end

# the balls and boxes at an overflowing spectrum, on the eigenvalues of X/2^k
@noinline function _scaled_spectral_membership!(s::Spectral, ws, X::AbstractMatrix)
  copyto!(ws.A, X)
  k = _magnitude_exponent(_maximum_abs(ws.A))
  _scaled_spectral_input!(ws.A, X)
  _scaled_membership(s.f, _eigvals!(ws), k)
end

function value_and_prox!(Y::AbstractMatrix, s::Spectral, X::AbstractMatrix, γ::Real)
  _checkwrite(value_and_prox!, s, Y, X)
  T = float(eltype(Y))
  n = checksquare(X)
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  copyto!(ws.A, X)
  λ, V = _eigen!(ws)
  # the eigenvalues of the result, exactly as many as those of X
  z = _buffer!(s.z, T, n)
  length(z) == n || resize!(z, n)
  v = value_and_prox!(z, s.f, λ, γ)
  _rebuild_sym_signed!(Y, V, z, Y isa StridedMatrix ? Y : _array_buffer!(s.out, T, (n, n)))
  convert(T, v)
end
prox!(Y::AbstractMatrix, s::Spectral, X::AbstractMatrix, γ::Real) =
  (_checkwrite(prox!, s, Y, X); value_and_prox!(Y, s, X, γ); Y)

# Y = Σⱼ zⱼ vⱼvⱼ' for the eigenvectors vⱼ in the columns of V, through B, which
# is Y when Y is strided and an n × n buffer otherwise; destroys V. The proximal
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
