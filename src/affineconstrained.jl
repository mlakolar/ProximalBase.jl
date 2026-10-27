####################################
#
#  A quadratic on an affine set
#
####################################

"""
    Constrained(f, IndAffine(A, b))

The quadratic `f`, a [`LeastSquaresLoss`](@ref) with a vector response or a
[`QuadraticFunction`](@ref), on the affine set `{x : Ax = b}`, and `Inf` outside
it. Its proximal operator is the equality-constrained ridge step

    x = M(c - Aᵀλ),   (AMAᵀ)λ = AMc - b,

with `M = (H + I/γ)⁻¹` for the Hessian `H` of `f` and `c = v/γ - ∇f(0)`: for the
least-squares loss `‖Y - Xβ‖²/2`, `H = X'X` and `c = X'Y + v/γ`.

`M` is applied through the decomposition that `f` keeps for its own proximal
operator, the thin SVD of `X` or the eigendecomposition of the quadratic's
matrix, computed on first use. The `m × m` matrix `AMAᵀ` depends on `γ`, and
the function keeps its Cholesky factor for the last `γ`. A call with the same
`γ` costs the unconstrained step plus two products with an `m × n` matrix, and
a new `γ` adds the factorization of `AMAᵀ`; neither allocates. A rank-deficient
`A` takes the equivalent constraint of full row rank from the thin SVD that
`IndAffine` caches. Do not share one function object between threads.

With ADMM, `f = Constrained(LeastSquaresLoss(Y, X), IndAffine(A, b))` and
`g = NormL1(λ)` give the lasso under equality constraints, such as log-contrast
regression with several subcompositions, `Aβ = 0` with a row of `A` for each.
"""
Constrained(f::Union{LeastSquaresLoss, QuadraticFunction}, C::IndAffine) = AffineConstrained(f, C)
Constrained(f::AbstractFunction, ::IndAffine) =
  throw(ArgumentError("Constrained on an affine set takes LeastSquaresLoss or QuadraticFunction, not $(nameof(typeof(f)))"))

# f on {x : Ax = b}. With H = V Diagonal(s2) V' from f and the constraint Cx = d,
# C of full row rank, M = V Diagonal(e) V' + γ(I - VV') for e = 1/(s2 + 1/γ),
# and CMCᵀ = W Diagonal(e) Wᵀ + γN for W = CV and N = C(I - VV')Cᵀ, both
# positive semidefinite. When V is square, N = 0 and M = V Diagonal(e) V'.
mutable struct AffineConstrained{T<:AbstractFloat, F<:Union{LeastSquaresLoss, QuadraticFunction}, S<:IndAffine} <: AbstractFunction
  f::F
  set::S
  # computed on first use
  W::Matrix{T}                  # CV
  N::Matrix{T}                  # C(I - VV')Cᵀ, empty when V is square
  Ws::Matrix{T}                 # W Diagonal(√e)
  chol::Cholesky{T, Matrix{T}}  # of CMCᵀ for the step γ
  γ::T                          # NaN before the first factorization
  t::Vector{T}                  # V'z
  u::Vector{T}
  r::Vector{T}                  # CMz - d, then the multiplier
  output::Scratch               # dense output for a non-strided destination
end

function AffineConstrained(f::Union{LeastSquaresLoss, QuadraticFunction}, set::IndAffine)
  f isa LeastSquaresLoss && !(f.Y isa AbstractVector) &&
    throw(ArgumentError("Constrained on an affine set takes a LeastSquaresLoss with a vector response"))
  n = f isa LeastSquaresLoss ? size(f.X, 2) : size(f.A, 1)
  size(set.A, 2) == n || throw(DimensionMismatch("the loss has $n coefficients and A has $(size(set.A, 2)) columns"))
  T = eltype(f.tmp)
  e = Matrix{T}(undef, 0, 0)
  AffineConstrained{T, typeof(f), typeof(set)}(f, set, e, e, e, Cholesky(e, 'U', 0), T(NaN), T[], T[], T[], Scratch())
end

isproximable(::Type{<:AffineConstrained}) = true
isconvex(::Type{<:AffineConstrained}) = true

(c::AffineConstrained)(x::AbstractVector) = (T = float(eltype(x)); iszero(c.set(x)) ? convert(T, c.f(x)) : T(Inf))

# the constraint Cx = d, of full row rank: Vᵀx = c on the rank-deficient path
_constraint(s::IndAffine) = (s.A, s.b)
_constraint(s::IndAffine{T, M, Tuple{Matrix{T}, Vector{T}}}) where {T, M} = (s.fact[1]', s.fact[2])

# H = V Diagonal(s2) V', from the decomposition f keeps for its proximal operator
_hessian_decomposition!(f::LeastSquaresLoss) = (_svd_cache!(f); (f.V, f.s2))
_hessian_decomposition!(f::QuadraticFunction) = (_eigen!(f); (f.Q, f.Λ))

# z = v/γ - ∇f(0)
function _ridge_rhs!(z, f::LeastSquaresLoss, v, ρ)
  rhs = _xty_cache!(f)
  z .= ρ .* v .+ rhs
end
_ridge_rhs!(z, f::QuadraticFunction, v, ρ) = f.b === nothing ? (z .= ρ .* v) : (z .= ρ .* v .- f.b)

function _affine_cache!(c::AffineConstrained{T}) where {T}
  isnan(c.γ) || return c
  V, _ = _hessian_decomposition!(c.f)
  C, _ = _constraint(c.set)
  n, k = size(V)
  c.W = Matrix{T}(C * V)
  if k < n
    R = Matrix{T}(C) - c.W * V'     # the rows of C orthogonal to the range of V
    c.N = R * R'
  end
  c.Ws = similar(c.W)
  m = size(c.W, 1)
  c.chol = Cholesky(zeros(T, m, m), 'U', 0)
  c.t, c.u, c.r = zeros(T, k), zeros(T, k), zeros(T, m)
  c
end

# the Cholesky factor of CMCᵀ = W Diagonal(e) Wᵀ + γN in place
function _affine_factor!(c::AffineConstrained{T}, γ::T) where {T}
  _, s2 = _hessian_decomposition!(c.f)
  ρ = inv(γ)
  W, Ws, G = c.W, c.Ws, c.chol.factors
  @inbounds for j in axes(W, 2)
    sj = sqrt(inv(s2[j] + ρ))
    for i in axes(W, 1)
      Ws[i, j] = W[i, j] * sj
    end
  end
  mul!(G, Ws, Ws')
  isempty(c.N) || (G .+= γ .* c.N)
  c.chol = cholesky!(Symmetric(G, :U); check = false)
  issuccess(c.chol) || throw(PosDefException(c.chol.info))
  c.γ = γ
  return c
end

function prox!(y::AbstractVector, c::AffineConstrained{T}, x::AbstractVector, γ::Real) where {T}
  _checksize(y, x)
  _checkaffine(c.set, x)
  y isa StridedVector || return copyto!(y, prox!(_array_buffer!(c.output, T, size(y)), c, x, γ))
  _affine_cache!(c)
  s = convert(T, γ)
  c.γ == s || _affine_factor!(c, s)
  V, s2 = _hessian_decomposition!(c.f)
  C, d = _constraint(c.set)
  W, t, u, r = c.W, c.t, c.u, c.r
  ρ = inv(s)
  _ridge_rhs!(y, c.f, x, ρ)                 # z
  mul!(t, V', y)
  u .= t ./ (s2 .+ ρ)
  # r = CMz - d, with CMz = W Diagonal(e) V'z + γ(Cz - WV'z)
  mul!(r, W, u)
  if !isempty(c.N)
    mul!(r, C, y, s, one(T))
    mul!(r, W, t, -s, one(T))
  end
  r .-= d
  ldiv!(c.chol, r)                          # the multiplier λ
  mul!(t, W', r, -one(T), one(T))           # V'(z - Cᵀλ)
  if isempty(c.N)
    u .= t ./ (s2 .+ ρ)
    mul!(y, V, u)
  else
    # M(z - Cᵀλ) = γ(z - Cᵀλ) + V Diagonal(e - γ) V'(z - Cᵀλ)
    mul!(y, C', r, -one(T), one(T))
    @inbounds for i in eachindex(t, u)
      u[i] = -s2[i] * t[i] / (ρ * (s2[i] + ρ))
    end
    y .*= s
    mul!(y, V, u, one(T), one(T))
  end
  return y
end

# the proximal point lies on the set, which the value does not check again
function value_and_prox!(y::AbstractVector, c::AffineConstrained{T}, x::AbstractVector, γ::Real) where {T}
  prox!(y, c, x, γ)
  # Reuse the dense destination for the value too: structured matrix products
  # with sparse iterates can allocate. Copy back any rounding in y's element type.
  point = y isa StridedVector ? y : copyto!(_array_buffer!(c.output, T, size(y)), y)
  convert(float(eltype(y)), c.f(point))
end
