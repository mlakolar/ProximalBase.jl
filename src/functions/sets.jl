####################################
#
#  Indicator functions of sets
#
####################################

# An indicator function is 0 on its set and Inf elsewhere. Its value accepts a
# point within a relative tolerance √eps of the set, so that the result of the
# projection is feasible, and value_and_prox! returns 0 without checking.

####################################
#  ℓ0 ball
####################################

"""
    IndBallL0(s; diagonal = true)

The indicator of the arrays with at most `s` nonzero entries, nonconvex. Its
proximal operator keeps the `s` entries of largest magnitude (exactly `s` when
there are ties) and sets the others to zero: a selection in O(n) expected time
(Floyd and Rivest's algorithm), without allocating, and in O(nnz) on a
[`SparseIterate`](@ref).

For a symmetric matrix, a `Symmetric` or a [`SymmetricSparseIterate`](@ref),
`s` counts the entries of one triangle, so that an entry and its transpose are
kept or dropped together, and the result is symmetric; the output may be a
dense matrix or a `SymmetricSparseIterate`. With `diagonal = false`, the
diagonal of a matrix does not count and is kept, as for the off-diagonal
sparsity of a precision matrix:

```julia
prox!(S, IndBallL0(s; diagonal = false), Symmetric(Θ, :L))   # S::SymmetricSparseIterate
```
"""
struct IndBallL0 <: AbstractFunction
  s::Int
  diagonal::Bool      # whether the diagonal of a matrix counts
  scratch::Scratch

  function IndBallL0(s::Integer; diagonal::Bool = true)
    s >= 0 || throw(ArgumentError("s must be nonnegative, got $s"))
    new(Int(s), diagonal, Scratch())
  end
end

isproximable(::Type{IndBallL0}) = true

# whether entry i of x is one that counts
@inline _counts(f::IndBallL0, x::AbstractArray, i::Integer) = true
@inline function _counts(f::IndBallL0, x::AbstractMatrix, i::Integer)
  f.diagonal && return true
  m = size(x, 1)
  mod1(i, m) != cld(i, m)
end

function _count_nonzero(f::IndBallL0, x::AbstractArray)
  k = 0
  @inbounds for i in eachindex(IndexLinear(), x)
    k += (!iszero(x[i]) & _counts(f, x, i))
  end
  k
end
function _count_nonzero(f::IndBallL0, x::SparseIterate)
  f.diagonal || ndims(x) == 1 || return invoke(_count_nonzero, Tuple{IndBallL0, AbstractArray}, f, x)
  count(!iszero, view(x.nzval, 1:x.nnz))
end

_ind_value(ok::Bool, x) = ok ? zero(float(eltype(x))) : float(eltype(x))(Inf)
(f::IndBallL0)(x::AbstractArray) = _ind_value(_count_nonzero(f, x) <= f.s, x)

# the entries of the lower triangle of a symmetric matrix, the diagonal unless
# it does not count
function (f::IndBallL0)(x::Union{Symmetric, SymmetricSparseIterate})
  p = checksquare(x)
  k = 0
  @inbounds for c in 1:p, r in (f.diagonal ? c : c + 1):p
    k += !iszero(x[r, c])
  end
  _ind_value(k <= f.s, x)
end

function prox!(y::AbstractArray, f::IndBallL0, x::AbstractArray, ::Real)
  _checksize(y, x)
  if _count_nonzero(f, x) <= f.s
    y === x || copyto!(y, x)
    return y
  end
  n = length(x)
  buf = _buffer!(f.scratch, float(eltype(x)), n)
  k = 0
  @inbounds for i in eachindex(IndexLinear(), x)
    _counts(f, x, i) || continue
    k += 1
    buf[k] = abs(x[i])
  end
  τ = f.s == 0 ? oftype(float(zero(eltype(x))), Inf) : _kth_largest!(view(buf, 1:k), f.s)
  ties = f.s
  @inbounds for i in eachindex(IndexLinear(), x)
    ties -= (_counts(f, x, i) & (abs(x[i]) > τ))
  end
  @inbounds for i in eachindex(IndexLinear(), y, x)
    v = x[i]
    if !_counts(f, x, i)
      y[i] = v
    else
      a = abs(v)
      keep = a > τ || (a == τ && ties > 0)
      keep && a == τ && (ties -= 1)
      y[i] = keep ? v : zero(eltype(y))
    end
  end
  y
end

# the stored entries of a vector hold every nonzero
function prox!(y::SparseIterate{<:Any, 1}, f::IndBallL0, x::SparseIterate{<:Any, 1}, γ::Real)
  _checksize(y, x)
  nz = _count_nonzero(f, x)
  (nz <= f.s || f.s == 0) && return invoke(prox!, Tuple{AbstractArray, IndBallL0, AbstractArray, Real}, y, f, x, γ)
  buf = _buffer!(f.scratch, float(eltype(x)), x.nnz)
  @inbounds for k in 1:x.nnz
    buf[k] = abs(x.nzval[k])
  end
  τ = _kth_largest!(view(buf, 1:x.nnz), f.s)
  ties = f.s - count(v -> abs(v) > τ, view(x.nzval, 1:x.nnz))
  y === x || copyto!(y, x)
  @inbounds for k in 1:y.nnz          # ties: the first stored entries are kept
    a = abs(y.nzval[k])
    if a < τ || (a == τ && ties <= 0)
      y.nzval[k] = zero(eltype(y))
    elseif a == τ
      ties -= 1
    end
  end
  y
end

# a symmetric matrix: the entries of the lower triangle, each counted once and
# written to both triangles of a dense output
function prox!(y::AbstractMatrix, f::IndBallL0, x::Union{Symmetric, SymmetricSparseIterate}, ::Real)
  _checksize(y, x)
  p = checksquare(x)
  T = float(eltype(x))
  first_row(c) = f.diagonal ? c : c + 1
  buf = _buffer!(f.scratch, T, div(p * (p + 1), 2))
  k = 0
  nz = 0
  @inbounds for c in 1:p, r in first_row(c):p
    a = abs(x[r, c])
    k += 1
    buf[k] = a
    nz += !iszero(a)
  end
  τ = nz <= f.s ? -one(T) : f.s == 0 ? T(Inf) : _kth_largest!(view(buf, 1:k), f.s)
  ties = f.s
  if nz > f.s
    @inbounds for c in 1:p, r in first_row(c):p
      ties -= abs(x[r, c]) > τ
    end
  end
  @inbounds for c in 1:p, r in c:p
    v = x[r, c]
    if !f.diagonal && r == c
      keep = true
    else
      a = abs(v)
      keep = a > τ || (a == τ && ties > 0)
      keep && a == τ && (ties -= 1)
    end
    _set_symmetric!(y, keep ? v : zero(v), r, c)
  end
  y
end

@inline _set_symmetric!(y::SymmetricSparseIterate, v, r, c) = _setstored!(y, v, tril_index(y.p, r, c))
@inline function _set_symmetric!(y::AbstractMatrix, v, r, c)
  @inbounds y[r, c] = v
  @inbounds y[c, r] = v
  y
end

value_and_prox!(y::AbstractArray, f::IndBallL0, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); zero(float(eltype(y))))

####################################
#  ℓ1 ball
####################################

"""
    IndBallL1(r = 1)

The indicator of the ℓ1 ball `{x : Σᵢ |xᵢ| ≤ r}`. Its proximal operator is the
projection, soft thresholding at the level that puts the result on the
boundary, found in O(n) expected time by Condat's algorithm.
"""
struct IndBallL1{T<:Real} <: AbstractFunction
  r::T
  scratch::Scratch

  IndBallL1{T}(r::T) where {T<:Real} = (_checknonneg("r", r); new{T}(r, Scratch()))
end
IndBallL1(r::Real = 1) = IndBallL1{typeof(r)}(r)

isproximable(::Type{<:IndBallL1}) = true
isconvex(::Type{<:IndBallL1}) = true

function (f::IndBallL1)(x::AbstractArray)
  T = float(eltype(x))
  norm(x, 1) <= f.r + _tol(T) * max(one(T), f.r) ? zero(T) : T(Inf)
end

function prox!(y::AbstractArray, f::IndBallL1, x::AbstractArray, ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  r = convert(T, f.r)
  if norm(x, 1) <= r
    y === x || copyto!(y, x)
    return y
  end
  if iszero(r)
    fill!(y, zero(eltype(y)))
    return y
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, T, length(x)), x, eachindex(IndexLinear(), x), r, abs)
  y .= shrink.(x, τ)
end
value_and_prox!(y::AbstractArray, f::IndBallL1, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); zero(float(eltype(y))))

####################################
#  ℓ2 ball
####################################

"""
    IndBallL2(r = 1)

The indicator of the Euclidean ball `{x : ‖x‖₂ ≤ r}` (the Frobenius ball for
matrices). Its proximal operator scales `x` onto the ball when it is outside.
"""
struct IndBallL2{T<:Real} <: AbstractFunction
  r::T

  IndBallL2{T}(r::T) where {T<:Real} = (_checknonneg("r", r); new{T}(r))
end
IndBallL2(r::Real = 1) = IndBallL2{typeof(r)}(r)

isproximable(::Type{<:IndBallL2}) = true
isconvex(::Type{<:IndBallL2}) = true

function (f::IndBallL2)(x::AbstractArray)
  T = float(eltype(x))
  sqrt(_sumabs2(x)) <= f.r + _tol(T) * max(one(T), f.r) ? zero(T) : T(Inf)
end

function prox!(y::AbstractArray, f::IndBallL2, x::AbstractArray, ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  nx = sqrt(convert(T, _sumabs2(x)))
  r = convert(T, f.r)
  if nx <= r
    y === x || copyto!(y, x)
  else
    y .= (r / nx) .* x
  end
  y
end
value_and_prox!(y::AbstractArray, f::IndBallL2, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); zero(float(eltype(y))))

####################################
#  simplex
####################################

"""
    IndSimplex(r = 1; equality = true)

The indicator of the simplex `{x ≥ 0 : Σᵢ xᵢ = r}`, or with `equality = false`
of `{x ≥ 0 : Σᵢ xᵢ ≤ r}`. Its proximal operator is the projection,
`max.(x .- τ, 0)` for the level `τ` found in O(n) expected time by Condat's
algorithm.
"""
struct IndSimplex{T<:Real} <: AbstractFunction
  r::T
  equality::Bool
  scratch::Scratch

  IndSimplex{T}(r::T, equality::Bool) where {T<:Real} = (_checknonneg("r", r); new{T}(r, equality, Scratch()))
end
IndSimplex(r::Real = 1; equality::Bool = true) = IndSimplex{typeof(r)}(r, equality)

isproximable(::Type{<:IndSimplex}) = true
isconvex(::Type{<:IndSimplex}) = true

function (f::IndSimplex)(x::AbstractArray)
  T = float(eltype(x))
  t = _tol(T)
  isempty(x) && return (f.equality && f.r > t) ? T(Inf) : zero(T)
  minimum(x) >= -t || return T(Inf)
  s = convert(T, sum(x))
  slack = t * max(one(T), f.r, abs(s))
  ok = f.equality ? abs(s - f.r) <= slack : s <= f.r + slack
  ok ? zero(T) : T(Inf)
end

function prox!(y::AbstractArray, f::IndSimplex, x::AbstractArray, ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  r = convert(T, f.r)
  if !f.equality
    s = zero(T)
    for v in x
      s += max(v, zero(v))
    end
    if s <= r
      y .= max.(x, zero(T))
      return y
    end
  end
  if iszero(r)
    fill!(y, zero(eltype(y)))
    return y
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, T, length(x)), x, eachindex(IndexLinear(), x), r)
  y .= max.(x .- τ, zero(T))
end
value_and_prox!(y::AbstractArray, f::IndSimplex, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); zero(float(eltype(y))))

####################################
#  hyperplane and halfspace
####################################

"""
    IndHyperplane(a, b)

The indicator of the hyperplane `{x : ⟨a, x⟩ = b}`, `a ≠ 0`. Its proximal
operator is the projection `x - ((⟨a, x⟩ - b) / ‖a‖²) a`.
"""
struct IndHyperplane{A<:AbstractArray, T<:Real} <: AbstractFunction
  a::A
  b::T
  a2::T          # ‖a‖²

  function IndHyperplane{A, T}(a::A, b::T) where {A<:AbstractArray, T<:Real}
    a2 = convert(T, sum(abs2, a))
    a2 > 0 || throw(ArgumentError("a must be nonzero"))
    new{A, T}(a, b, a2)
  end
end
IndHyperplane(a::AbstractArray, b::Real) = (T = float(promote_type(eltype(a), typeof(b))); IndHyperplane{typeof(a), T}(a, convert(T, b)))

"""
    IndHalfspace(a, b)

The indicator of the halfspace `{x : ⟨a, x⟩ ≤ b}`, `a ≠ 0`. Its proximal operator
is the projection, `x - (max(⟨a, x⟩ - b, 0) / ‖a‖²) a`.
"""
struct IndHalfspace{A<:AbstractArray, T<:Real} <: AbstractFunction
  a::A
  b::T
  a2::T

  function IndHalfspace{A, T}(a::A, b::T) where {A<:AbstractArray, T<:Real}
    a2 = convert(T, sum(abs2, a))
    a2 > 0 || throw(ArgumentError("a must be nonzero"))
    new{A, T}(a, b, a2)
  end
end
IndHalfspace(a::AbstractArray, b::Real) = (T = float(promote_type(eltype(a), typeof(b))); IndHalfspace{typeof(a), T}(a, convert(T, b)))

for F in (:IndHyperplane, :IndHalfspace)
  @eval isproximable(::Type{<:$F}) = true
  @eval isconvex(::Type{<:$F}) = true
  @eval value_and_prox!(y::AbstractArray, f::$F, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); zero(float(eltype(y))))
end

function (f::IndHyperplane)(x::AbstractArray)
  T = float(eltype(x))
  abs(dot(f.a, x) - f.b) <= _tol(T) * max(one(T), abs(f.b), sqrt(f.a2 * _sumabs2(x))) ? zero(T) : T(Inf)
end
function (f::IndHalfspace)(x::AbstractArray)
  T = float(eltype(x))
  dot(f.a, x) - f.b <= _tol(T) * max(one(T), abs(f.b), sqrt(f.a2 * _sumabs2(x))) ? zero(T) : T(Inf)
end

function prox!(y::AbstractArray, f::IndHyperplane, x::AbstractArray, ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  t = convert(T, (dot(f.a, x) - f.b) / f.a2)
  y .= x .- t .* f.a
end
function prox!(y::AbstractArray, f::IndHalfspace, x::AbstractArray, ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  t = convert(T, max(dot(f.a, x) - f.b, zero(f.b)) / f.a2)
  y .= x .- t .* f.a
end
