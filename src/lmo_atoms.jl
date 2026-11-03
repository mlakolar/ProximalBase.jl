####################################
#
#  Compact atoms of the linear minimization oracles (see lmo_atom)
#
####################################

"""
    VertexAtom{T,N}

A signed coordinate vertex `value * e_index` of an array of size `dims`, the result of
[`lmo_atom`](@ref) for [`IndSimplex`](@ref) and [`IndBallL1`](@ref). The fields are
`dims::NTuple{N,Int}`, the size of the array, `index::Int`, the linear (column-major) index
of the nonzero entry, and `value::T`; the zero atom has `index == 0` and `value == 0`. It
stores O(1) numbers whatever the size of the array.

It supports `size`, `eltype`, `dot(g, a)`, `LinearAlgebra.axpy!(α, a, x)` for a dense `x`, and
`Array(a)`.
"""
struct VertexAtom{T<:AbstractFloat, N}
  dims::NTuple{N, Int}
  index::Int
  value::T
end

"""
    LowRankAtom{T}

A matrix `U * V'` in factored form, the result of [`lmo_atom`](@ref) for the nuclear-norm
ball, the density matrices and the Fantope. The fields are `U::Matrix{T}`, `m × k`, and
`V::Matrix{T}`, `n × k`, with `k = 0` for the zero atom. A symmetric atom has `V === U`, so the
factor is stored once and `Array(a)` is exactly symmetric. Storage is `(m + n) k` numbers.

It supports `size`, `eltype`, `dot(g, a)` for any `g` that can multiply a matrix (a sparse
matrix, or an operator that defines `*`), `LinearAlgebra.axpy!(α, a, X)` for a dense `X`, and
`Array(a)`. A factored iterate `X = Σⱼ cⱼ Uⱼ Vⱼ'` takes `(1 - γ) X + γ a` by scaling its own
coefficients and appending the columns of `a`.
"""
struct LowRankAtom{T<:AbstractFloat}
  U::Matrix{T}
  V::Matrix{T}
  function LowRankAtom{T}(U::Matrix{T}, V::Matrix{T}) where {T<:AbstractFloat}
    size(U, 2) == size(V, 2) || throw(DimensionMismatch("the factors have $(size(U, 2)) and $(size(V, 2)) columns"))
    new{T}(U, V)
  end
end
LowRankAtom(U::Matrix{T}, V::Matrix{T}) where {T<:AbstractFloat} = LowRankAtom{T}(U, V)

const _Atom = Union{VertexAtom, LowRankAtom}

Base.size(a::VertexAtom) = a.dims
Base.size(a::LowRankAtom) = (size(a.U, 1), size(a.V, 1))
Base.size(a::_Atom, d::Integer) = d <= length(size(a)) ? size(a)[d] : 1
Base.eltype(::Type{<:VertexAtom{T}}) where {T} = T
Base.eltype(::Type{<:LowRankAtom{T}}) where {T} = T
Base.ndims(a::_Atom) = length(size(a))
Base.length(a::_Atom) = prod(size(a))
Base.:(==)(a::VertexAtom, b::VertexAtom) = a.dims == b.dims && a.index == b.index && a.value == b.value
Base.isequal(a::VertexAtom, b::VertexAtom) = a.dims == b.dims && a.index == b.index && isequal(a.value, b.value)
Base.hash(a::VertexAtom, h::UInt) = hash(a.value, hash(a.index, hash(a.dims, h)))

function Base.show(io::IO, a::VertexAtom)
  print(io, "VertexAtom(", a.index == 0 ? "zero" : string(a.value, " at ", a.index), ", size ", a.dims, ")")
end
function Base.show(io::IO, a::LowRankAtom)
  print(io, "LowRankAtom(", size(a, 1), " × ", size(a, 2), ", ", size(a.U, 2), " factor",
        size(a.U, 2) == 1 ? "" : "s", a.U === a.V ? ", symmetric" : "", ")")
end

# a dense array of the atom
function Base.Array(a::VertexAtom{T}) where {T}
  x = zeros(T, a.dims)
  a.index > 0 && (@inbounds x[a.index] = a.value)
  x
end
Base.Array{S}(a::VertexAtom) where {S} = convert(Array{S}, Array(a))
Base.Array(a::LowRankAtom{T}) where {T} = size(a.U, 2) == 0 ? zeros(T, size(a)) : a.U * a.V'
Base.Array{S}(a::LowRankAtom) where {S} = convert(Array{S}, Array(a))
Base.Matrix(a::LowRankAtom) = Array(a)

# ⟨g, a⟩
function LinearAlgebra.dot(g::AbstractArray, a::VertexAtom)
  size(g) == a.dims || throw(DimensionMismatch("g has size $(size(g)), the atom $(a.dims)"))
  a.index == 0 ? zero(promote_type(eltype(g), eltype(a))) : @inbounds g[a.index] * a.value
end
function LinearAlgebra.dot(g::AbstractMatrix, a::LowRankAtom)
  size(g) == size(a) || throw(DimensionMismatch("g has size $(size(g)), the atom $(size(a))"))
  size(a.U, 2) == 0 && return zero(promote_type(eltype(g), eltype(a)))
  dot(a.U, g * a.V)
end
LinearAlgebra.dot(a::_Atom, g::AbstractArray) = dot(g, a)

# x ← x + α a, on a dense array
function LinearAlgebra.axpy!(α::Number, a::VertexAtom, x::AbstractArray)
  size(x) == a.dims || throw(DimensionMismatch("x has size $(size(x)), the atom $(a.dims)"))
  a.index > 0 && (@inbounds x[a.index] += α * a.value)
  x
end
function LinearAlgebra.axpy!(α::Number, a::LowRankAtom, x::AbstractMatrix)
  size(x) == size(a) || throw(DimensionMismatch("x has size $(size(x)), the atom $(size(a))"))
  size(a.U, 2) == 0 && return x
  mul!(x, a.U, a.V', α, true)
  x
end

####################################
#  the entry point
####################################

"""
    lmo_atom(C, g; method = :full, rtol = nothing, maxiter = nothing, fallback = true)

A minimizer of `s ↦ dot(g, s)` over the compact set `C`, as a compact atom instead of a
dense array: the result of [`lmo!`](@ref) without the `m × n` matrix. Supported sets:

| `C` | atom | minimizer |
|:----|:-----|:----------|
| [`IndSimplex`](@ref)`(r)`, [`IndBallL1`](@ref)`(r)` | [`VertexAtom`](@ref) | the vertex of `lmo!`, with the same selection and ties |
| `SingularValue(IndBallL1(r))`, the nuclear-norm ball `‖S‖_* ≤ r` | [`LowRankAtom`](@ref) | `-r u vᵀ` for a leading singular pair `(u, v)` of `g` |
| `Spectral(IndSimplex(r))`, the density matrices of trace `r` | [`LowRankAtom`](@ref) | `r v vᵀ` for an eigenvector `v` of the smallest eigenvalue of `g` |
| `Spectral(IndSimplex(r; equality = false))`, the PSD matrices of trace at most `r` | [`LowRankAtom`](@ref) | `0` if the smallest eigenvalue is nonnegative, otherwise as above |
| the Fantope `Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))` | [`LowRankAtom`](@ref) | `Q Qᵀ` for the `k` eigenvectors of the `k` smallest eigenvalues of `g` |

The atom owns its arrays, never a view of a workspace: it stays valid after later calls, also
with other sizes and element types. Its element type is `float(eltype(g))`.

**Matrix sets.** `g` is a real `Matrix`, or any `AbstractMatrix` that supports `mul!` when
`method = :krylov`: a sparse matrix, for one. Device arrays are not supported. For the spectral
sets `g` must be square, and the oracle minimizes against its symmetric part `(g + gᵀ)/2`, which
has the same pairing `⟨g, S⟩` with every symmetric `S` as `g`; the upper triangle alone is never
read. The matrix is finite: a `NaN` or infinite entry is an `ArgumentError`, because no atom
represents an undefined direction (`lmo!` writes `NaN` instead). The zero gradient, and a zero
radius, give the zero atom for the nuclear-norm ball and the inequality simplex, and the first
coordinate vector (`Q` the first `k` columns of the identity) for the sets of fixed trace.
The leading vectors are not unique when the leading value is repeated or `g` has a tie at the
`k`-th eigenvalue; any minimizer is returned, with the sign of the largest entry of `v` made
positive. A `k = 0` Fantope gives the zero atom and `k = n` the identity, without a
decomposition.

**Exactness.** `method = :full`, the default, uses LAPACK on a dense copy of `g`, so a sparse `g`
is densified: `gesdd` for the nuclear ball; `syevr` for the smallest eigenpairs, or `syevd` for
more than half of them. The matrix is scaled by a power of two first, so extreme scales do not
overflow. `method = :krylov` (the nuclear-norm ball and the density matrices) runs Lanczos
bidiagonalization or Lanczos with full reorthogonalization on products with `g`, from a fixed
starting vector, using `O(nnz)` work per step and no dense copy. It stops when the residual of
the pair is at most `rtol` times its value (for the eigenproblem, times the larger of the extreme Ritz values in magnitude; `rtol` defaults to `√ε`, `maxiter` to `150` steps, at
most `min(size(g))`), and otherwise falls back to the full decomposition, or throws an
`ErrorException` when `fallback = false`. The rank-`k` Fantope has no Krylov method: a
single-vector method does not find all of a repeated eigenvalue's eigenspace.

**What is certified.** The Krylov result is *residual-checked, not certified*: the small residual
of one Ritz pair does not show that no larger singular value or smaller eigenvalue was missed.
The value `dot(g, atom)` of the atom is evaluated directly from the atom, up to rounding, and is an upper bound on
`minₛ ⟨g, s⟩`, and `-dot(g, atom)` is a lower bound on the support function of `C` at `-g`
(`r σ₁(g)` for the nuclear-norm ball). The starting vector is fixed, so a miss is reproducible: a gradient whose leading direction is orthogonal to it
(a structured or block-diagonal gradient, for one) is missed with a small residual, and the atom is then suboptimal. A Frank–Wolfe gap `⟨∇f(x), x - s⟩` formed with an
approximate atom is therefore a lower bound on the gap of the exact oracle, and *not* an upper
bound on `f(x) - f*`. Only the default path gives the oracle's answer up to the rounding of the
decomposition.

Construct one set object per task: the full path caches its workspace in the set, as `prox!`
does. The Krylov path allocates local arrays only, `(m + n)` times the number of steps.
"""
function lmo_atom(C::AbstractFunction, g::AbstractArray; kwargs...)
  haslmo(C) || _lmo_unsupported(C)
  throw(ArgumentError("lmo_atom is defined for IndSimplex, IndBallL1, the nuclear-norm ball, " *
                      "the density matrices and the Fantope; use lmo! for $(typeof(C))"))
end

function _atom_options(method, rtol, maxiter, fallback, krylov::Bool)
  method in (:full, :krylov) || throw(ArgumentError("method must be :full or :krylov, got $(repr(method))"))
  if method === :krylov
    krylov || throw(ArgumentError("method = :krylov is defined for the nuclear-norm ball and the density matrices only"))
  else
    rtol === nothing && maxiter === nothing || throw(ArgumentError("rtol and maxiter apply to method = :krylov only"))
  end
  rtol === nothing || (rtol isa Real && rtol > 0 && isfinite(rtol)) || throw(ArgumentError("rtol must be positive and finite, got $rtol"))
  maxiter === nothing || (maxiter isa Integer && maxiter >= 1) || throw(ArgumentError("maxiter must be a positive integer, got $maxiter"))
  nothing
end

####################################
#  vertices of the simplex and the l1 ball
####################################

function lmo_atom(C::Union{IndSimplex, IndBallL1}, g::AbstractArray; method::Symbol = :full, rtol = nothing,
                  maxiter = nothing, fallback::Bool = true)
  _checkreal(C, g)
  _atom_options(method, rtol, maxiter, fallback, false)
  haslmo(C) || _lmo_unsupported(C)
  T = float(eltype(g))
  _lmo_check!(C, g, T)
  i, v, nan = _lmo_vertex(C, g, T)
  nan && throw(ArgumentError("g contains NaN: no atom represents an undefined direction (lmo! writes NaN)"))
  VertexAtom{T, ndims(g)}(size(g), i, v)
end

####################################
#  the sets of matrices
####################################

haslmo(F::SingularValue{<:IndBallL1}) = haslmo(F.f)
haslmo(s::Spectral{<:IndSimplex}) = haslmo(s.f)
haslmo(s::Spectral{<:_CappedSimplex}) = haslmo(s.f)

const _NuclearBall = SingularValue{<:IndBallL1}
const _Spectrahedron = Spectral{<:IndSimplex}
const _Fantope = Spectral{<:_CappedSimplex}

# the arrays whose entries bound the entries of a matrix: the stored entries of a sparse one
_stored_values(g::AbstractMatrix) = g
_stored_values(g::SparseArrays.AbstractSparseMatrixCSC) = nonzeros(g)
_stored_values(g::Union{Symmetric, Hermitian, Adjoint, Transpose}) = _stored_values(parent(g))

# the checks that precede any computation or write: type, finiteness and the scale of g
function _atom_matrix(C, g::AbstractMatrix)
  _checkreal(C, g)
  T = float(eltype(g))
  _checkblas(C, T)
  _is_device_array(g) && throw(ArgumentError("lmo_atom and lmo! of $(nameof(typeof(C))) do not support device arrays: " *
                                              "copy the matrix to the host explicitly"))
  T
end

# the largest |entry| of g, NaN or Inf when an entry is not finite
function _max_abs(g, ::Type{T}) where {T}
  mx = zero(T)
  @inbounds for x in _stored_values(g)
    a = abs(convert(T, x))
    isnan(a) && return a
    mx = max(mx, a)
  end
  mx
end

# the power of two 2^e with which g is scaled, to entries of magnitude in [0.5, 1)
_scale_exponent(mx::AbstractFloat) = exponent(mx) + 1

mutable struct _Seed
  seed::UInt64
end

function _start_vector(::Type{T}, n::Integer) where {T}
  v = Matrix{T}(undef, n, 1)
  _randomize!(v, 1:1, _Seed(0x9e3779b97f4a7c15))
  w = norm(v)
  v ./= w
  vec(v)
end

# x ← 2^-e y without an overflow or an underflow of the exponent
@inline function _unscale!(x::AbstractVector{T}, y::AbstractVector{T}, e::Int) where {T}
  @inbounds for i in eachindex(x, y)
    x[i] = ldexp(y[i], -e)
  end
  x
end

# the operators of the bidiagonalization: p = G x, q = Gᵀ x, for the matrix scaled by 2^-e
struct _ScaledOp{T, M}
  G::M
  e::Int
  xs::Vector{T}
end
function _mul_matrix!(p::Vector{T}, op::_ScaledOp{T}, x::AbstractVector{T}) where {T}
  mul!(p, op.G, _unscale!(resize!(op.xs, length(x)), x, op.e))
end
function _mul_adjoint!(q::Vector{T}, op::_ScaledOp{T}, x::AbstractVector{T}) where {T}
  mul!(q, op.G', _unscale!(resize!(op.xs, length(x)), x, op.e))
end

# w ← (G + Gᵀ)/2 x for the matrix scaled by 2^-e; a Symmetric G needs one product
function _mul_symmetric_part!(w::Vector{T}, w2::Vector{T}, op::_ScaledOp{T}, x::AbstractVector{T}) where {T}
  xs = _unscale!(resize!(op.xs, length(x)), x, op.e)
  if op.G isa Union{Symmetric, Hermitian}
    mul!(w, op.G, xs)
  else
    mul!(w, op.G, xs)
    mul!(w2, op.G', xs)
    @inbounds for i in eachindex(w)
      w[i] = (w[i] + w2[i]) / 2
    end
  end
  w
end

# p ← p - Q (Qᵀ p), twice, for the first k columns of Q (a column-major Matrix)
function _reorth!(p::Vector{T}, Q::Matrix{T}, k::Int, c::Vector{T}) where {T}
  k == 0 && return p
  Qk = view(Q, :, 1:k)
  ck = view(c, 1:k)
  for _ in 1:2
    mul!(ck, Qk', p)
    mul!(p, Qk, ck, -one(T), one(T))
  end
  p
end

# The leading singular triplet of G / 2^e by Golub–Kahan–Lanczos bidiagonalization with full
# reorthogonalization: G V = U B, Gᵀ U = V Bᵀ + βⱼ vⱼ₊₁ eⱼᵀ, B upper bidiagonal. For the left
# singular vector y of B with value σ, (σ, U y, V z) satisfies G (V z) = σ (U y) exactly and
# ‖Gᵀ(U y) - σ (V z)‖ = βⱼ |yⱼ|, which is the residual tested. Returns (σ, u, v, residual,
# steps, converged), with σ in the scaled units.
function _gkl_top(G, ::Type{T}, e::Int, rtol::T, maxiter::Int) where {T}
  m, n = size(G)
  K = min(maxiter, m, n)
  V = Matrix{T}(undef, n, K + 1)
  U = Matrix{T}(undef, m, K)
  α, β, c = zeros(T, K), zeros(T, K), Vector{T}(undef, K + 1)
  p, q = Vector{T}(undef, m), Vector{T}(undef, n)
  op = _ScaledOp{T, typeof(G)}(G, e, Vector{T}(undef, max(m, n)))
  V[:, 1] = _start_vector(T, n)
  σ, y, z, res, steps, ok = zero(T), T[], T[], T(Inf), 0, false
  for j in 1:K
    _mul_matrix!(p, op, view(V, :, j))
    j > 1 && axpy!(-β[j - 1], view(U, :, j - 1), p)
    _reorth!(p, U, j - 1, c)
    aj = norm(p)
    α[j] = aj
    steps = j
    if iszero(aj)
      # G v_j is in the span of the previous directions: the subspace is invariant
      break
    end
    U[:, j] = p ./ aj
    _mul_adjoint!(q, op, view(U, :, j))
    axpy!(-aj, view(V, :, j), q)
    _reorth!(q, V, j, c)
    bj = norm(q)
    β[j] = bj
    check = j == K || iszero(bj) || (j >= 3 && j % 4 == 0)
    if check
      B = zeros(T, j, j)
      for i in 1:j
        B[i, i] = α[i]
        i < j && (B[i, i + 1] = β[i])
      end
      F = svd!(B)
      σ = F.S[1]
      y, z = F.U[:, 1], F.Vt[1, :]
      res = bj * abs(y[j])
      if iszero(bj) || res <= rtol * σ
        ok = true
        break
      end
    end
    j < K && (V[:, j + 1] = q ./ bj)
  end
  if iszero(α[steps]) && steps >= 1
    # G v_j is in the span of u₁…u_{j-1}, so (V_j, U_{j-1}) is an invariant pair: G V_j = U_{j-1} M and
    # Gᵀ U_{j-1} = V_j Mᵀ with the (j-1) × j matrix M = [B_{j-1}  β_{j-1} e_{j-1}], whose singular
    # triplets are then exact triplets of G (not those of B_{j-1}, unless β_{j-1} = 0)
    j = steps
    if j >= 2
      M = zeros(T, j - 1, j)
      for i in 1:j - 1
        M[i, i] = α[i]
        M[i, i + 1] = β[i]
      end
      F = svd!(M)
      σ, y, z, res, ok = F.S[1], F.U[:, 1], F.Vt[1, :], zero(T), true
      steps = j
    else
      # G v₁ = 0: the starting vector is in the null space; the full decomposition decides
      return zero(T), T[], T[], zero(T), 1, false
    end
  end
  isempty(y) && return zero(T), T[], T[], T(Inf), steps, false
  u = view(U, :, 1:length(y)) * y
  v = view(V, :, 1:length(z)) * z
  σ, u, v, res, steps, ok
end

# The smallest eigenvalue of the symmetric part of G / 2^e, by Lanczos with full reorthogonalization.
# For the smallest Ritz value θ with the eigenvector y of the tridiagonal matrix, (θ, V y) has the
# residual βⱼ |yⱼ|, tested against the larger of the extreme Ritz values in magnitude.
function _lanczos_min(G, ::Type{T}, e::Int, rtol::T, maxiter::Int) where {T}
  n = size(G, 1)
  K = min(maxiter, n)
  V = Matrix{T}(undef, n, K + 1)
  α, β, c = zeros(T, K), zeros(T, K), Vector{T}(undef, K + 1)
  w, w2 = Vector{T}(undef, n), Vector{T}(undef, n)
  op = _ScaledOp{T, typeof(G)}(G, e, Vector{T}(undef, n))
  V[:, 1] = _start_vector(T, n)
  θ, y, res, steps, ok = zero(T), T[], T(Inf), 0, false
  for j in 1:K
    _mul_symmetric_part!(w, w2, op, view(V, :, j))
    aj = dot(view(V, :, j), w)
    α[j] = aj
    steps = j
    axpy!(-aj, view(V, :, j), w)
    j > 1 && axpy!(-β[j - 1], view(V, :, j - 1), w)
    _reorth!(w, V, j, c)
    bj = norm(w)
    β[j] = bj
    check = j == K || iszero(bj) || (j >= 3 && j % 4 == 0)
    if check
      Tj = SymTridiagonal(α[1:j], β[1:j - 1])
      F = eigen(Tj, 1:1)
      θ = F.values[1]
      y = F.vectors[:, 1]
      ρ = max(abs(θ), abs(eigmax(Tj)))
      res = bj * abs(y[j])
      if iszero(bj) || res <= rtol * ρ
        ok = true
        break
      end
    end
    j < K && (V[:, j + 1] = w ./ bj)
  end
  isempty(y) && return zero(T), T[], T(Inf), steps, false
  v = view(V, :, 1:length(y)) * y
  θ, v, res, steps, ok
end

# the sign convention: the first entry of largest magnitude is positive
function _positive_pivot!(x::AbstractVector, y::Union{Nothing, AbstractVector} = nothing)
  isempty(x) && return x
  i = 1
  @inbounds for k in 2:length(x)
    abs(x[k]) > abs(x[i]) && (i = k)
  end
  if x[i] < 0
    x .= .-x
    y === nothing || (y .= .-y)
  end
  x
end

# the full decompositions, of g / 2^e, into vectors owned by the caller
function _nuclear_full(F::SingularValue, g::AbstractMatrix, ::Type{T}, e::Int) where {T}
  ws = _svd_workspace(F.ws, T, size(g)...)
  F.ws = ws
  @inbounds for j in axes(g, 2), i in axes(g, 1)
    ws.A[i, j] = ldexp(convert(T, g[i, j]), -e)
  end
  _, U, Vt = _svd!(ws)
  u, v = U[:, 1], Vt[1, :]
  v isa Vector || (v = collect(v))
  _positive_pivot!(v, u)
  u, v
end

# dense copy of the symmetric part of g / 2^e into the upper triangle of A
function _symmetric_part!(A::Matrix{T}, g::AbstractMatrix, e::Int, negate::Bool) where {T}
  n = size(A, 1)
  sg = negate ? -one(T) : one(T)
  @inbounds for j in 1:n
    for i in 1:j
      A[i, j] = sg * (ldexp(convert(T, g[i, j]), -e - 1) + ldexp(convert(T, g[j, i]), -e - 1))   # scale, then add: no overflow
    end
  end
  A
end
# a sparse g is read through a dense copy: random access to a sparse matrix is slow
_dense_for_read(g::AbstractMatrix) = g
_dense_for_read(g::SparseArrays.AbstractSparseMatrixCSC) = Matrix(g)
_dense_for_read(g::Union{Symmetric, Hermitian}) = parent(g) isa SparseArrays.AbstractSparseMatrixCSC ? Matrix(g) : g

# the k smallest eigenpairs of the symmetric part of g / 2^e, ascending: values and vectors
function _smallest_eigenpairs(s::Spectral, g::AbstractMatrix, ::Type{T}, e::Int, k::Int) where {T}
  n = size(g, 1)
  h = _dense_for_read(g)
  if 2k <= n
    ws = _eigtop_workspace(s.top, T, n, k)
    s.top = ws
    _symmetric_part!(ws.A, h, e, true)           # the k largest eigenvalues of -g are the k smallest of g
    λ, Z = _eigen_top!(ws)
    # ascending in -g means descending in g: reverse into ascending order of g
    vals = T[-λ[k + 1 - i] for i in 1:k]
    vecs = Matrix{T}(undef, n, k)
    @inbounds for j in 1:k, i in 1:n
      vecs[i, j] = Z[i, k + 1 - j]
    end
    return vals, vecs
  end
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  _symmetric_part!(ws.A, h, e, false)
  λ, V = _eigen!(ws)
  copy(λ[1:k]), V[:, 1:k]
end

####################################
#  the atoms of the sets of matrices
####################################

# the Krylov iteration scales the vectors by 2^-e, which must neither overflow nor leave the normal range
_krylov_scale_ok(::Type{T}, e::Int) where {T} = abs(e) <= exponent(floatmax(T)) - 29

_nonfinite_atom(strict::Bool) = strict ?
  throw(ArgumentError("g must be finite: no atom represents an undefined direction (lmo! writes NaN)")) : nothing

_zero_atom(::Type{T}, m::Integer, n::Integer) where {T} = LowRankAtom(zeros(T, m, 0), zeros(T, n, 0))

# nothing for a non-finite g unless strict
function _nuclear_atom(F::SingularValue, g::AbstractMatrix, method, rtol, maxiter, fallback, strict::Bool = true)
  T = _atom_matrix(F, g)
  _atom_options(method, rtol, maxiter, fallback, true)
  haslmo(F) || _lmo_unsupported(F)
  _lmo_check_fits(T, "the radius", F.f.r)
  m, n = size(g)
  r = convert(T, F.f.r)
  (iszero(r) || m == 0 || n == 0) && return _zero_atom(T, m, n)
  mx = _max_abs(g, T)
  isfinite(mx) || return _nonfinite_atom(strict)
  iszero(mx) && return _zero_atom(T, m, n)
  e = _scale_exponent(mx)
  local u, v
  done = false
  if method === :krylov
    tol = rtol === nothing ? sqrt(eps(T)) : convert(T, rtol)
    ok = false
    if _krylov_scale_ok(T, e)
      _, uk, vk, _, _, ok = _gkl_top(g, T, e, tol, maxiter === nothing ? 150 : Int(maxiter))
    end
    if ok
      u, v, done = uk, vk, true
      _positive_pivot!(v, u)
    else
      fallback || error("the Lanczos iteration did not reach rtol = $tol (fallback = false)")
    end
  end
  done || ((u, v) = _nuclear_full(F, g, T, e))
  # S = -r u vᵀ; the coefficient is in the left factor, which then has norm r
  U = reshape(u, m, 1)
  U .*= -r
  LowRankAtom(U, reshape(v, n, 1))
end

function lmo_atom(F::_NuclearBall, g::AbstractMatrix; method::Symbol = :full, rtol = nothing, maxiter = nothing,
                  fallback::Bool = true)
  _nuclear_atom(F, g, method, rtol, maxiter, fallback)
end

# the smallest eigenpair of g by the Krylov method, or by the full decomposition
function _spectrahedron_atom(s::Spectral, g::AbstractMatrix, method, rtol, maxiter, fallback, strict::Bool = true)
  T = _atom_matrix(s, g)
  _atom_options(method, rtol, maxiter, fallback, true)
  haslmo(s) || _lmo_unsupported(s)
  n = checksquare(g)
  f = s.f
  _lmo_check_fits(T, "the radius", f.r)
  r = convert(T, f.r)
  if iszero(r) || n == 0
    f.equality && n == 0 && f.r > _tol(T) && throw(ArgumentError("the density matrices of size 0 have no point of trace $(f.r)"))
    return _zero_atom(T, n, n)
  end
  mx = _max_abs(g, T)
  isfinite(mx) || return _nonfinite_atom(strict)
  if iszero(mx)
    f.equality || return _zero_atom(T, n, n)
    v = zeros(T, n, 1)
    v[1] = sqrt(r)
    return LowRankAtom(v, v)
  end
  e = _scale_exponent(mx)
  local λ, v
  done = false
  if method === :krylov
    tol = rtol === nothing ? sqrt(eps(T)) : convert(T, rtol)
    ok = false
    if _krylov_scale_ok(T, e)
      θ, vk, _, _, ok = _lanczos_min(g, T, e, tol, maxiter === nothing ? 150 : Int(maxiter))
    end
    if ok
      λ, v, done = θ, vk, true
    else
      fallback || error("the Lanczos iteration did not reach rtol = $tol (fallback = false)")
    end
  end
  if !done
    vals, vecs = _smallest_eigenpairs(s, g, T, e, 1)
    λ, v = vals[1], vecs[:, 1]
  end
  !f.equality && λ >= 0 && return _zero_atom(T, n, n)
  _positive_pivot!(v)
  v .*= sqrt(r)
  F = reshape(v, n, 1)
  LowRankAtom(F, F)
end

function lmo_atom(s::_Spectrahedron, g::AbstractMatrix; method::Symbol = :full, rtol = nothing, maxiter = nothing,
                  fallback::Bool = true)
  _spectrahedron_atom(s, g, method, rtol, maxiter, fallback)
end

function _fantope_atom(s::Spectral, g::AbstractMatrix, method, rtol, maxiter, fallback, strict::Bool = true)
  T = _atom_matrix(s, g)
  _atom_options(method, rtol, maxiter, fallback, false)
  _checkparams(s.f, g)
  haslmo(s) || _lmo_unsupported(s)
  n = checksquare(g)
  k = Int(s.f.set.b)
  k <= n || throw(ArgumentError("the Fantope with trace $k has no point of size $n"))
  # the minimizer does not depend on g when k is 0 or n, as for the capped simplex
  k == 0 && return _zero_atom(T, n, n)
  if k == n
    Q = Matrix{T}(I, n, n)
    return LowRankAtom(Q, Q)
  end
  mx = _max_abs(g, T)
  isfinite(mx) || return _nonfinite_atom(strict)
  Q = zeros(T, n, k)
  if iszero(mx)
    for j in 1:k
      Q[j, j] = one(T)
    end
  else
    _, Q = _smallest_eigenpairs(s, g, T, _scale_exponent(mx), k)
    for j in 1:k
      _positive_pivot!(view(Q, :, j))
    end
  end
  LowRankAtom(Q, Q)
end

function lmo_atom(s::_Fantope, g::AbstractMatrix; method::Symbol = :full, rtol = nothing, maxiter = nothing,
                  fallback::Bool = true)
  _fantope_atom(s, g, method, rtol, maxiter, fallback)
end

####################################
#  dense output
####################################

const _MatrixLMOSet = Union{_NuclearBall, _Spectrahedron, _Fantope}

_matrix_atom(C::_NuclearBall, g) = _nuclear_atom(C, g, :full, nothing, nothing, true, false)
_matrix_atom(C::_Spectrahedron, g) = _spectrahedron_atom(C, g, :full, nothing, nothing, true, false)
_matrix_atom(C::_Fantope, g) = _fantope_atom(C, g, :full, nothing, nothing, true, false)

# The parameters of the set must fit the output before anything is computed
_check_output_radius(C::Union{_NuclearBall, _Spectrahedron}, ::Type{T}) where {T} = _lmo_check_fits(T, "the radius", C.f.r)
_check_output_radius(C::_Fantope, ::Type{T}) where {T} = nothing

function lmo!(s::AbstractMatrix, C::_MatrixLMOSet, g::AbstractMatrix)
  _checkreal(C, g)
  _checkreal(C, s)
  s isa Union{Array, SubArray{<:Any, <:Any, <:Array}} && eltype(s) <: AbstractFloat ||
    throw(ArgumentError("the output of lmo! must be an Array, or a view of one, of floating-point numbers; got $(typeof(s))"))
  _checksize(s, g)
  haslmo(C) || _lmo_unsupported(C)
  _check_output_radius(C, eltype(s))
  a = _matrix_atom(C, g)               # every error is raised here, before s is written
  if a === nothing
    fill!(s, eltype(s)(NaN))
  else
    _materialize!(s, a)
  end
  s
end

function _materialize!(s::AbstractMatrix, a::LowRankAtom)
  T = eltype(s)
  size(a.U, 2) == 0 && return fill!(s, zero(T))
  if s isa StridedMatrix{T} && _blas_ready(s) && eltype(a) === T
    a.U === a.V ? mul!(s, a.U, a.U') : mul!(s, a.U, a.V')
  else
    copyto!(s, Array(a))
  end
  s
end

####################################
#  conjugate values, the support functions
####################################

# The domain is the finite matrices, as for the value of Spectral and SingularValue: a non-finite entry
# gives Inf. The conjugate of f ∘ λ is f* ∘ λ (Lewis, 1996) on the symmetric matrices, where F is finite, so
# F*(Y) = F*((Y + Yᵀ)/2): the values are those of the eigenvalues of the symmetric part of Y.
# r λ_max for the density matrices, r max(λ_max, 0) for the PSD matrices of trace at most r, and the
# sum of the k largest eigenvalues for the Fantope, without an atom or a dense matrix.
_has_conjugate_value(s::Union{_Spectrahedron, _Fantope}) = haslmo(s)
_support_fallback(::Union{_Spectrahedron, _Fantope}) = false

function _checkconjugate(s::Union{_Spectrahedron, _Fantope}, Y::AbstractMatrix)
  _checkreal(s, Y)
  _checkblas(s, float(eltype(Y)))
  checksquare(Y)
  s isa _Fantope && (_checkparams(s.f, Y); Int(s.f.set.b) <= size(Y, 1) ||
    throw(ArgumentError("the Fantope with trace $(Int(s.f.set.b)) has no point of size $(size(Y, 1))")))
  true
end

function _conjugate_value(s::Union{_Spectrahedron, _Fantope}, Y::AbstractMatrix)
  _checkconjugate(s, Y)
  T = float(eltype(Y))
  haslmo(s) || _lmo_unsupported(s)
  n = size(Y, 1)
  is_f = s isa _Fantope
  k = is_f ? Int(s.f.set.b) : 1
  if is_f ? k == 0 : (iszero(s.f.r) || n == 0)
    return zero(T)
  end
  _is_device_array(Y) && throw(ArgumentError("the conjugate value of $(nameof(typeof(s))) does not support device arrays"))
  mx = _max_abs(Y, T)
  isfinite(mx) || return T(Inf)
  iszero(mx) && return zero(T)
  e = _scale_exponent(mx)
  ws = _eig_workspace(s.eig, T, n)
  s.eig = ws
  _symmetric_part!(ws.A, _dense_for_read(Y), e, false)
  λ = _eigvals!(ws)
  if is_f
    t = zero(T)
    @inbounds for i in n - k + 1:n
      t += λ[i]
    end
    return convert(T, ldexp(t, e))
  end
  top = max(λ[n], s.f.equality ? λ[n] : zero(T))
  iszero(top) && return zero(T)
  convert(T, _scaled_product(convert(T, s.f.r), top, e))
end
