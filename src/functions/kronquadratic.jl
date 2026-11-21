####################################
#  Kronecker quadratic
####################################

"""
    KronQuadratic(A, B, C = nothing)

The quadratic `q(X) = ⟨X, A X B⟩ / 2 + ⟨C, X⟩` of a `p × q` matrix `X`, with the
Frobenius inner product `⟨U, V⟩ = tr(U'V)` and `C = 0` when it is omitted. It is
`vec(X)' (B ⊗ A) vec(X) / 2 + vec(C)' vec(X)` without the `pq × pq` matrix `B ⊗ A`.

`A` is `p × p` and `B` is `q × q`, both nonempty, finite and exactly symmetric. `C` is
a finite `p × q` matrix. `A`, `B` and `C` have one element type, `Float32` or
`Float64`; `Symmetric`, `Hermitian` and `Diagonal` wrappers of that type are
accepted. Other element types, mixed ones included, throw `ArgumentError`:
convert every mismatching argument to one of the two precisions first, for example
`Float64.(C)` when `A` and `B` are `Float64` matrices. The constructor stores dense copies,
so later changes to the caller's arrays do not change `f`.
[`ProximalBase.kronquadratic_data`](@ref) returns the stored copies.

The gradient depends on the argument:
- an `AbstractMatrix` `X` of size `(p, q)` has the gradient `A X B + C`;
- a [`SymmetricSparseIterate`](@ref) `X`, for `p == q`, has the gradient
  `(A X B + B X A) / 2 + C`, the derivative over symmetric matrices with the
  Frobenius inner product. When `C` is supplied, these calls require it to be
  exactly symmetric and throw `ArgumentError` before writes otherwise.
The value is the same formula for both. A dense symmetric matrix, a `Symmetric`
wrapper and `matrixview(X)` are `AbstractMatrix` arguments and get `A X B + C`.

The argument and a gradient destination have the element type `T` of the data.
`gradient` returns a `Matrix{T}` and `value_and_gradient` returns `(v, G)` with
`v::T` and `G::Matrix{T}`. `gradient!` and `value_and_gradient!` write into a
writable strided `p × q` matrix `G`, row-strided views included, that must be
independent of the argument and of the matrices `kronquadratic_data(f)` returns. As
for the other built-in gradients, `G === X` throws `ArgumentError` and other
overlapping views are unsupported; the oracles also throw `ArgumentError` before
writes when `Base.mightalias` reports an overlap of `G` with the argument, `A`, `B`
or `C`. There is no proximal operator.

`f` is convex when `A` and `B` are both positive semidefinite or both negative
semidefinite. [`isconvex`](@ref) returns `false` because the constructor does
not check semidefiniteness. The exact product `‖A‖₂ ‖B‖₂` of the spectral
norms is a Lipschitz constant of the gradient; [`lipschitz`](@ref) returns the
product of the two computed norms in working precision, which can round below the
exact product and is subject to the ordinary range limits (it can overflow to
`Inf`, or be `NaN` when one norm overflows and the other is zero).

The oracles use scratch matrices stored in `f`: do not call them concurrently on one
object.

# Examples
```jldoctest
julia> f = KronQuadratic([2.0;;], [3.0;;], [1.0;;]);

julia> f([5.0;;])
80.0

julia> gradient(f, [5.0;;])
1×1 Matrix{Float64}:
 31.0
```

See [Kronecker quadratic](@ref Kronecker-quadratic) for the symmetric gradient,
the D-trace loss and the coordinate-descent direction model.
"""
struct KronQuadratic{T<:Union{Float32, Float64}, MC<:Union{Nothing, Matrix{T}}} <: AbstractFunction
  A::Matrix{T}
  B::Matrix{T}
  C::MC
  csym::Bool          # whether C is nothing, or square and exactly symmetric
  X::Matrix{T}        # scratch: the copy of the argument on the copy routes
  AX::Matrix{T}       # scratch: A X
  W::Matrix{T}        # scratch: A X B (plus C before the copy into a destination off the direct route), or B X

  function KronQuadratic(A::AbstractMatrix, B::AbstractMatrix, C::Union{Nothing, AbstractMatrix} = nothing)
    T = _kron_eltype(A, B, C)
    _check_one_based(A, B, C)
    p, q = checksquare(A), checksquare(B)
    p >= 1 && q >= 1 || throw(ArgumentError("KronQuadratic requires nonempty A and B, got $p × $p and $q × $q"))
    C === nothing || size(C) == (p, q) ||
      throw(DimensionMismatch("A is $p × $p and B is $q × $q, so C must be $p × $q, not $(size(C, 1)) × $(size(C, 2))"))
    Ad, Bd = Matrix{T}(A), Matrix{T}(B)
    Cd = C === nothing ? nothing : Matrix{T}(C)
    all(isfinite, Ad) || throw(ArgumentError("KronQuadratic requires finite entries in A"))
    all(isfinite, Bd) || throw(ArgumentError("KronQuadratic requires finite entries in B"))
    Cd === nothing || all(isfinite, Cd) || throw(ArgumentError("KronQuadratic requires finite entries in C"))
    _check_symmetric(Ad, "the KronQuadratic matrix A", "A")
    _check_symmetric(Bd, "the KronQuadratic matrix B", "B")
    csym = Cd === nothing || (p == q && _issymmetric_exact(Cd, _symequal))
    new{T, typeof(Cd)}(Ad, Bd, Cd, csym, zeros(T, p, q), zeros(T, p, q), zeros(T, p, q))
  end
end

# The common element type of A, B and C, which must be Float32 or Float64. The message names the arrays to
# convert to the first supported type among them (Float64 when there is none).
function _kron_eltype(A, B, C)
  TA, TB = eltype(A), eltype(B)
  TC = C === nothing ? TA : eltype(C)
  (TA === Float32 || TA === Float64) && TA === TB === TC && return TA
  _kron_eltype_error(TA, TB, C === nothing ? nothing : TC)
end
@noinline function _kron_eltype_error(TA, TB, TC)
  names = TC === nothing ? ("A", "B") : ("A", "B", "C")
  types = TC === nothing ? (TA, TB) : (TA, TB, TC)
  i = findfirst(t -> t === Float32 || t === Float64, types)
  S = i === nothing ? Float64 : types[i]
  fix = join(("$S.($n)" for (n, t) in zip(names, types) if t !== S), ", ", " and ")
  got = join(("eltype($n) = $t" for (n, t) in zip(names, types)), ", ", " and ")
  throw(ArgumentError("KronQuadratic requires $(join(names, ", ", " and ")) with one element type, " *
                      "Float32 or Float64, but $got; convert first, for example with $fix"))
end

issmooth(::Type{<:KronQuadratic}) = true
# Convex for semidefinite A and B of the same sign, which the constructor does not check.
isconvex(::Type{<:KronQuadratic}) = false
Base.size(f::KronQuadratic) = (size(f.A, 1), size(f.B, 1))
lipschitz(f::KronQuadratic) = _opnorm_upper(f.A) * _opnorm_upper(f.B)

"""
    ProximalBase.kronquadratic_data(f::KronQuadratic)

Return `(; A, B, C)`: the matrices that `f` stores, not copies, with `C === nothing`
when the linear coefficient was omitted or passed as `nothing`. Treat them as read-only: they are ordinary
mutable matrices, and a change to them changes `f` without the checks of the constructor.

# Examples
```jldoctest
julia> d = ProximalBase.kronquadratic_data(KronQuadratic([2.0;;], [3.0;;]));

julia> (d.A, d.B, d.C)
([2.0;;], [3.0;;], nothing)
```
"""
kronquadratic_data(f::KronQuadratic) = (A = f.A, B = f.B, C = f.C)

_argument_type(::Type{<:KronQuadratic}) = Union{AbstractMatrix, SymmetricSparseIterate}
check_parameters(f::KronQuadratic, X::Union{AbstractMatrix, SymmetricSparseIterate}) = (_kron_checksize(f, X, "X"); nothing)

####################################
#  Checks before writes
####################################

_kron_checksize(f::KronQuadratic, X, name) = size(X) == size(f) || _kron_size_error(f, X, name)
@noinline _kron_size_error(f, X, name) =
  throw(DimensionMismatch("KronQuadratic takes $(size(f)[1]) × $(size(f)[2]) matrices, and $name is $(size(X, 1)) × $(size(X, 2))"))
@noinline _kron_type_error(what, S, T, name) =
  throw(ArgumentError("KronQuadratic $what must have the element type $T of its data, not $S; convert first, for example with $T.($name)"))

function _kron_check_input(f::KronQuadratic{T}, X::AbstractMatrix) where {T}
  eltype(X) === T || _kron_type_error("input", eltype(X), T, "X")
  _check_one_based(X)
  _kron_checksize(f, X, "X")
  nothing
end

function _kron_check_input(f::KronQuadratic{T}, X::SymmetricSparseIterate) where {T}
  eltype(X) === T || _kron_type_error("input", eltype(X), T, "X")
  p, q = size(f)
  p == q || throw(DimensionMismatch("a SymmetricSparseIterate argument requires a square KronQuadratic, and this one takes $p × $q matrices"))
  _kron_checksize(f, X, "X")
  f.csym || throw(ArgumentError("KronQuadratic at a SymmetricSparseIterate requires an exactly symmetric C; " *
                                "construct it with a symmetric C, for example (C + C')/2"))
  nothing
end

function _kron_check_output(G, f::KronQuadratic{T}, X) where {T}
  G isa StridedMatrix || throw(ArgumentError("KronQuadratic writes its gradient into a writable strided matrix, not a $(typeof(G))"))
  eltype(G) === T || _kron_type_error("gradient output", eltype(G), T, "G")
  _check_one_based(G)
  _kron_checksize(f, G, "G")
  _checkgradient_output(G, X)
  Base.mightalias(G, X) && throw(ArgumentError("the gradient must not overwrite x"))
  (Base.mightalias(G, f.A) || Base.mightalias(G, f.B) || (f.C !== nothing && Base.mightalias(G, f.C))) &&
    throw(ArgumentError("KronQuadratic gradient output must be independent of A, B and C"))
  nothing
end

####################################
#  Matrix arguments: the gradient A X B + C
####################################

# The direct route takes a strided matrix with unit first stride and a column stride of at least max(1, rows), which
# mul! can pass to BLAS for eligible sizes. An accepted matrix argument that fails this test is copied into f.X
# first, and an accepted destination that fails it receives a copy of the result from f.W.
@inline _kron_blas(X) = false
@inline _kron_blas(X::StridedMatrix) = stride(X, 1) == 1 && stride(X, 2) >= max(1, size(X, 1))

# The value at X, and with G !== nothing the gradient A X B + C written into G.
function _kron_dense!(G, f::KronQuadratic, X::AbstractMatrix)
  _kron_blas(X) && return _kron_dense_blas!(G, f, X)
  copyto!(f.X, X)
  _kron_dense_blas!(G, f, f.X)
end

function _kron_dense_blas!(G, f::KronQuadratic, X)
  mul!(f.AX, f.A, X)
  if G === nothing
    mul!(f.W, f.AX, f.B)
    return _kron_dense_value(f, X, f.W)
  elseif _kron_blas(G)
    mul!(G, f.AX, f.B)
    v = _kron_dense_value(f, X, G)
    f.C === nothing || (G .+= f.C)
    return v
  else
    W = f.W
    mul!(W, f.AX, f.B)
    v = _kron_dense_value(f, X, W)
    f.C === nothing || (W .+= f.C)
    copyto!(G, W)
    return v
  end
end

# ⟨X, W⟩/2 + ⟨C, X⟩ for W = A X B
@inline function _kron_dense_value(f::KronQuadratic, X, W)
  v = dot(X, W) / 2
  f.C === nothing ? v : v + dot(f.C, X)
end

(f::KronQuadratic)(X::AbstractMatrix) = (_kron_check_input(f, X); _kron_dense!(nothing, f, X))

function value_and_gradient!(G, f::KronQuadratic, X::AbstractMatrix)
  _kron_check_input(f, X)
  _kron_check_output(G, f, X)
  _kron_dense!(G, f, X)
end
gradient!(G, f::KronQuadratic, X::AbstractMatrix) = (value_and_gradient!(G, f, X); G)

####################################
#  Symmetric iterates: the gradient (A X B + B X A)/2 + C
####################################

# The products M X with the matrix that a symmetric iterate X represents take one of two routes: column by column
# over the active coordinates (_kron_left!), or a matrix product on a dense copy. Value-only and gradient calls take
# the dense route when 32n > p² for n nonzero active values. In exclusive sweeps on an Intel Xeon E5-2640 v4 with
# Julia 1.10.13 and 1.13.4, the accumulating route of a value-only call ran at 0.04–0.82 of a straightforward dense
# implementation below this cutoff, with the crossover near n ≈ p²/28 (p = 800) and p²/23 (p = 200); that of a
# gradient call crossed near p²/30 (p = 800) and p²/21 (p = 200). See benchmark/README.md, "Measured performance".
# Counting the nonzero values needs no coordinate conversion.
function _kron_dense_route(X::SymmetricSparseIterate, p::Int)
  vals = activevalues(X)
  n = 0
  @inbounds @simd for a in eachindex(vals)
    n += !iszero(vals[a])
  end
  32n > p * p
end

# axpy y[:, j] += v M[:, i]
@inline function _kron_axpy!(Y, v, M, i, j, p)
  @inbounds @simd for k in 1:p
    Y[k, j] = muladd(v, M[k, i], Y[k, j])
  end
end

# Y = M X for a p × p matrix M, column by column over the active coordinates: a nonzero value v at (r, c)
# adds v M[:, r] to column c of Y and, off the diagonal, v M[:, c] to column r. With a second pair (Z, N)
# the same for Z = N X. A zero coefficient, stored or absent, contributes nothing.
function _kron_left!(Y::Matrix{T}, M::Matrix{T}, Z, N, X::SymmetricSparseIterate{T}) where {T}
  fill!(Y, zero(T))
  Z === nothing || fill!(Z, zero(T))
  p = size(M, 1)
  ids, vals = activecoordinates(X), activevalues(X)
  @inbounds for a in eachindex(vals)
    v = vals[a]
    iszero(v) && continue
    r, c = tril_cartesian(p, ids[a])
    _kron_axpy!(Y, v, M, r, c, p)
    Z === nothing || _kron_axpy!(Z, v, N, r, c, p)
    if r != c
      _kron_axpy!(Y, v, M, c, r, p)
      Z === nothing || _kron_axpy!(Z, v, N, c, r, p)
    end
  end
  Y
end

# The matrix that X represents, written into D.
function _kron_densify!(D::Matrix{T}, X::SymmetricSparseIterate{T}) where {T}
  fill!(D, zero(T))
  p = size(D, 1)
  ids, vals = activecoordinates(X), activevalues(X)
  @inbounds for a in eachindex(vals)
    r, c = tril_cartesian(p, ids[a])
    v = vals[a]
    D[r, c] = v
    D[c, r] = v
  end
  D
end

# ⟨W, X⟩/2 + ⟨C, X⟩ over the stored entries of X in one pass: an off-diagonal coordinate counts at both of its
# positions, and a zero coefficient contributes nothing.
function _kron_stored_value(W::Matrix{T}, C, X::SymmetricSparseIterate{T}) where {T}
  p = size(W, 1)
  ids, vals = activecoordinates(X), activevalues(X)
  s, t = zero(T), zero(T)
  @inbounds for a in eachindex(vals)
    v = vals[a]
    iszero(v) && continue
    r, c = tril_cartesian(p, ids[a])
    if r == c
      s = muladd(W[r, r], v, s)
      C === nothing || (t = muladd(C[r, r], v, t))
    else
      s = muladd(W[r, c] + W[c, r], v, s)
      C === nothing || (t = muladd(C[r, c] + C[c, r], v, t))
    end
  end
  s / 2 + t
end

# Σᵢⱼ P[i, j] Q[j, i] = ⟨P, Qᵀ⟩ of two p × p matrices, in tiles of up to 32 × 32 entries, each summed on its
# own before it is added to the total.
function _kron_transposed_dot(P::Matrix{T}, Q::Matrix{T}) where {T}
  p = size(P, 1)
  s = zero(T)
  @inbounds for jb in 1:32:p, ib in 1:32:p
    t = zero(T)
    for j in jb:min(jb + 31, p), i in ib:min(ib + 31, p)
      t = muladd(P[i, j], Q[j, i], t)
    end
    s += t
  end
  s
end

# ⟨D, W⟩/2 + ⟨C, D⟩ for the dense copy D of the argument
@inline _kron_dense_copy_value(f::KronQuadratic, D, W) =
  (v = dot(D, W) / 2; f.C === nothing ? v : v + dot(f.C, D))

# The value ⟨X, A X B⟩/2 + ⟨C, X⟩. The accumulating route forms A X and B X by accumulation and avoids the further
# product (A X) B and any call of mul!: with X, A and B symmetric,
# ⟨X, A X B⟩ = tr(A X B X) = Σᵢⱼ (A X)[i, j] (B X)[j, i].
function _kron_symmetric_value(f::KronQuadratic{T}, X::SymmetricSparseIterate{T}) where {T}
  p = size(f.A, 1)
  if _kron_dense_route(X, p)
    D = _kron_densify!(f.X, X)
    mul!(f.AX, f.A, D)
    mul!(f.W, f.AX, f.B)
    return _kron_dense_copy_value(f, D, f.W)
  end
  _kron_left!(f.AX, f.A, f.W, f.B, X)
  v = _kron_transposed_dot(f.AX, f.W) / 2
  f.C === nothing ? v : v + dot(f.C, X)
end

# The value, and the gradient W/2 + Wᵀ/2 + C for W = A X B written into G, in column order over all entries. The two
# entries of a mirrored pair add the same two halves, and C, when present, is exactly symmetric here
# (_kron_check_input), so they receive the same value; halving first keeps the loop free of branches.
function _kron_symmetric!(G::StridedMatrix{T}, f::KronQuadratic{T}, X::SymmetricSparseIterate{T}) where {T}
  p = size(f.A, 1)
  AX, W, C = f.AX, f.W, f.C
  if _kron_dense_route(X, p)
    D = _kron_densify!(f.X, X)
    mul!(AX, f.A, D)
    mul!(W, AX, f.B)
    v = _kron_dense_copy_value(f, D, W)
  else
    _kron_left!(AX, f.A, nothing, nothing, X)
    mul!(W, AX, f.B)
    v = _kron_stored_value(W, C, X)
  end
  @inbounds for j in 1:p
    for i in 1:p
      g = W[i, j] / 2 + W[j, i] / 2
      G[i, j] = C === nothing ? g : g + C[i, j]
    end
  end
  v
end

(f::KronQuadratic)(X::SymmetricSparseIterate) = (_kron_check_input(f, X); _kron_symmetric_value(f, X))

function value_and_gradient!(G, f::KronQuadratic, X::SymmetricSparseIterate)
  _kron_check_input(f, X)
  _kron_check_output(G, f, X)
  _kron_symmetric!(G, f, X)
end
gradient!(G, f::KronQuadratic, X::SymmetricSparseIterate) = (value_and_gradient!(G, f, X); G)

# The allocating oracles at a symmetric iterate (the generic ones take arrays): a dense Matrix{T} gradient.
gradient(f::KronQuadratic, X::SymmetricSparseIterate) = gradient!(_gradient_buffer(X), f, X)
function value_and_gradient(f::KronQuadratic, X::SymmetricSparseIterate)
  G = _gradient_buffer(X)
  v = value_and_gradient!(G, f, X)
  v, G
end

# The wrappers (Scaled) reach these oracles through the internal entry points.
_gradient_unchecked!(G, f::KronQuadratic, X) = gradient!(G, f, X)
_value_and_gradient_unchecked!(G, f::KronQuadratic, X) = value_and_gradient!(G, f, X)
