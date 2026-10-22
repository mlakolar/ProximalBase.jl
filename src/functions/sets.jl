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
    IndBallL0(s)

The indicator of the vectors with at most `s` nonzero entries, nonconvex. Its
proximal operator keeps the `s` entries of largest magnitude (exactly `s` when
there are ties) and sets the others to zero: a quickselect in O(n) expected
time, without allocating, and in O(nnz) on a [`SparseIterate`](@ref). On a [`SymmetricSparseIterate`](@ref), `s` counts its
coordinates, the entries of the lower triangle, so that an entry and its
transpose are kept or dropped together.
"""
struct IndBallL0 <: AbstractFunction
  s::Int
  scratch::Scratch

  IndBallL0(s::Integer) = (s >= 0 || throw(ArgumentError("s must be nonnegative, got $s")); new(Int(s), Scratch()))
end

isproximable(::Type{IndBallL0}) = true

_count_nonzero(x::AbstractArray) = count(!iszero, x)
_count_nonzero(x::SparseIterate) = count(!iszero, view(x.nzval, 1:x.nnz))

(f::IndBallL0)(x::AbstractArray) = _count_nonzero(x) <= f.s ? zero(float(eltype(x))) : float(eltype(x))(Inf)
(f::IndBallL0)(x::SymmetricSparseIterate) = f(view(x.nzval, 1:x.nnz))

function prox!(y::AbstractArray, f::IndBallL0, x::AbstractArray, ::Real)
  _checksize(y, x)
  if _count_nonzero(x) <= f.s
    y === x || copyto!(y, x)
    return y
  end
  if f.s == 0
    fill!(y, zero(eltype(y)))
    return y
  end
  n = length(x)
  buf = _buffer!(f.scratch, float(eltype(x)), n)
  k = 0
  @inbounds for i in eachindex(IndexLinear(), x)
    k += 1
    buf[k] = abs(x[i])
  end
  τ = _kth_largest!(view(buf, 1:n), f.s)
  _keep_largest!(y, x, τ, f.s - count(v -> abs(v) > τ, x))
end

# y = x where |x| > τ, and where |x| = τ for the first `ties` such entries; 0 elsewhere
function _keep_largest!(y::AbstractArray, x::AbstractArray, τ, ties::Integer)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    v = x[i]
    a = abs(v)
    if a > τ || (a == τ && ties > 0)
      a > τ || (ties -= 1)
      y[i] = v
    else
      y[i] = zero(eltype(y))
    end
  end
  y
end

# the stored entries hold every nonzero
function prox!(y::SparseIterate, f::IndBallL0, x::SparseIterate, γ::Real)
  _checksize(y, x)
  nz = _count_nonzero(x)
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

# over the coordinates
function prox!(y::SymmetricSparseIterate, f::IndBallL0, x::SymmetricSparseIterate, γ::Real)
  _checksize(y, x)
  prox!(coordinates(y), f, coordinates(x), γ)
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
  s = zero(T)
  for v in x
    v >= -t || return T(Inf)
    s += v
  end
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
