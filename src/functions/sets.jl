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
dense matrix or a `SymmetricSparseIterate`. The result is the projection in the
Frobenius norm: an off-diagonal entry stands for a pair, whose removal costs
twice its square, so it competes with the diagonal by `√2|xᵢⱼ|`. With
`diagonal = false`, the diagonal of a matrix does not count and is kept, as for
the off-diagonal sparsity of a precision matrix:

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
_isindicator(::Type{IndBallL0}) = true

# whether entry i of x is one that counts
@inline _counts(f::IndBallL0, x::AbstractArray, i::Integer) = true
@inline function _counts(f::IndBallL0, x::AbstractMatrix, i::Integer)
  f.diagonal && return true
  m = size(x, 1)
  mod1(i, m) != cld(i, m)
end

# whether every entry of x counts, as unless x is a matrix with a free diagonal
_allcount(f::IndBallL0, x::AbstractArray) = f.diagonal || ndims(x) != 2

function _count_nonzero(f::IndBallL0, x::AbstractArray)
  _allcount(f, x) && return count(!iszero, x)
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

(f::IndBallL0)(x::AbstractArray) = _ind(_count_nonzero(f, x) <= f.s, float(eltype(x)))

# the entries of the lower triangle of a symmetric matrix, the diagonal unless
# it does not count
function (f::IndBallL0)(x::Union{Symmetric, SymmetricSparseIterate})
  p = checksquare(x)
  k = 0
  @inbounds for c in 1:p, r in (f.diagonal ? c : c + 1):p
    k += !iszero(x[r, c])
  end
  _ind(k <= f.s, float(eltype(x)))
end

function prox!(y::AbstractArray, f::IndBallL0, x::AbstractArray, ::Real)
  _checksize(y, x)
  if _count_nonzero(f, x) <= f.s
    y === x || copyto!(y, x)
    return y
  end
  _allcount(f, x) ? _keep_largest_all!(y, f, x) : _keep_largest_counted!(y, f, x)
end

# every entry counts: loops without a test per entry, which vectorize
function _keep_largest_all!(y::AbstractArray, f::IndBallL0, x::AbstractArray)
  n = length(x)
  buf = _buffer!(f.scratch, float(eltype(x)), n)
  @inbounds for (k, i) in enumerate(eachindex(IndexLinear(), x))
    buf[k] = abs(x[i])
  end
  τ = f.s == 0 ? oftype(float(zero(eltype(x))), Inf) : _kth_largest!(view(buf, 1:n), f.s)
  # keep the entries at or above τ in one branch-free pass, then drop the ties at τ
  # beyond the s entries, which is rare, so that this works in place
  above, equal = 0, 0
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    a = abs(x[i])
    above += a > τ
    equal += a == τ
  end
  @inbounds @simd for i in eachindex(IndexLinear(), y, x)
    v = x[i]
    y[i] = ifelse(abs(v) >= τ, v, zero(v))
  end
  equal > f.s - above ? _drop_extra_ties!(y, τ, f.s - above, i -> true) : y
end

# zero the entries of y equal to τ in magnitude after the first `keep` of those
# for which counts(i) holds
function _drop_extra_ties!(y::AbstractArray, τ, keep::Integer, counts::F) where {F}
  isfinite(τ) || return y
  @inbounds for i in eachindex(IndexLinear(), y)
    if counts(i) && abs(y[i]) == τ
      keep > 0 ? (keep -= 1) : (y[i] = zero(eltype(y)))
    end
  end
  y
end

# a matrix whose diagonal is free: it does not count and is kept
function _keep_largest_counted!(y::AbstractArray, f::IndBallL0, x::AbstractArray)
  n = length(x)
  buf = _buffer!(f.scratch, float(eltype(x)), n)
  k = 0
  @inbounds for i in eachindex(IndexLinear(), x)
    _counts(f, x, i) || continue
    k += 1
    buf[k] = abs(x[i])
  end
  τ = f.s == 0 ? oftype(float(zero(eltype(x))), Inf) : _kth_largest!(view(buf, 1:k), f.s)
  above, equal = 0, 0
  @inbounds for i in eachindex(IndexLinear(), x)
    c, a = _counts(f, x, i), abs(x[i])
    above += c & (a > τ)
    equal += c & (a == τ)
  end
  @inbounds for i in eachindex(IndexLinear(), y, x)
    v = x[i]
    y[i] = ifelse(!_counts(f, x, i) | (abs(v) >= τ), v, zero(v))
  end
  equal > f.s - above ? _drop_extra_ties!(y, τ, f.s - above, i -> _counts(f, x, i)) : y
end

# the stored entries of a vector hold every nonzero
function prox!(y::SparseIterate{<:Any, 1}, f::IndBallL0, x::SparseIterate{<:Any, 1}, γ::Real)
  _checksize(y, x)
  nz = _count_nonzero(f, x)
  if nz <= f.s
    y === x || copyto!(y, x)
    return y
  elseif f.s == 0
    @inbounds for k in 1:y.nnz          # every entry is zero; y keeps its stored entries
      y.nzval[k] = zero(eltype(y))
    end
    return y
  end
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
  w = sqrt(T(2))
  buf = _buffer!(f.scratch, T, div(p * (p + 1), 2))
  k = 0
  nz = 0
  @inbounds for c in 1:p, r in first_row(c):p
    a = _pairweight(f, r, c, w) * abs(x[r, c])
    k += 1
    buf[k] = a
    nz += !iszero(a)
  end
  τ = nz <= f.s ? -one(T) : f.s == 0 ? T(Inf) : _kth_largest!(view(buf, 1:k), f.s)
  ties = f.s
  if nz > f.s
    @inbounds for c in 1:p, r in first_row(c):p
      ties -= _pairweight(f, r, c, w) * abs(x[r, c]) > τ
    end
  end
  @inbounds for c in 1:p, r in c:p
    v = x[r, c]
    if !f.diagonal && r == c
      keep = true
    else
      a = _pairweight(f, r, c, w) * abs(v)
      keep = a > τ || (a == τ && ties > 0)
      keep && a == τ && (ties -= 1)
    end
    _set_symmetric!(y, keep ? v : zero(v), r, c)
  end
  y
end

# the weight of entry (r, c) in the selection: when the diagonal counts, an
# off-diagonal entry stands for the pair (r, c), (c, r), whose removal costs
# twice its square in the Frobenius norm; with a free diagonal every counted
# entry is off the diagonal, and the weight is 1
@inline _pairweight(f::IndBallL0, r, c, w) = f.diagonal && r != c ? w : one(w)

@inline _set_symmetric!(y::SymmetricSparseIterate, v, r, c) = _setstored!(y, v, tril_index(y.p, r, c))
@inline function _set_symmetric!(y::AbstractMatrix, v, r, c)
  @inbounds y[r, c] = v
  @inbounds y[c, r] = v
  y
end


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
_isindicator(::Type{<:IndBallL1}) = true
isconvex(::Type{<:IndBallL1}) = true

function (f::IndBallL1)(x::AbstractArray)
  T = float(eltype(x))
  eltype(x) <: Integer && return _ball_scaled_value(f, x, T)
  n = norm(x, 1)
  slack = f.r + _tol(T) * max(one(T), f.r)
  # A non-finite norm already fails the comparison with a finite bound.
  _finite_bound(slack) && return _ind(n <= slack, T)
  _ball_scaled_value(f, x, T)
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
_isindicator(::Type{<:IndBallL2}) = true
isconvex(::Type{<:IndBallL2}) = true

function (f::IndBallL2)(x::AbstractArray)
  T = float(eltype(x))
  eltype(x) <: Integer && return _ball_scaled_value(f, x, T)
  n = _norm2(x)
  slack = f.r + _tol(T) * max(one(T), f.r)
  # A non-finite norm already fails the comparison with a finite bound.
  _finite_bound(slack) && return _ind(n <= slack, T)
  _ball_scaled_value(f, x, T)
end

@noinline function _ball_scaled_value(f, x, ::Type{T}) where {T}
  m = _maximum_abs(x)
  isfinite(m) || return T(Inf)
  S = float(promote_type(typeof(m), typeof(f.r)))
  isfinite(f.r) || return _ind(isinf(f.r), T)
  k = _comparison_exponent(_magnitude_exponent(m), _magnitude_exponent(float(f.r)))
  s = _power_factors(S, -k)
  r = ldexp(S(f.r), -k)
  n = f isa IndBallL1 ? _scaled_sum(x, s, abs) : sqrt(_scaled_sumabs2(x, s))
  slack = _tol(T) * max(ldexp(one(S), -k), r)
  _ind(n <= r + slack, T)
end

function prox!(y::AbstractArray, f::IndBallL2, x::AbstractArray, ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  nx = convert(T, _norm2(x))
  r = convert(T, f.r)
  if nx <= r
    y === x || copyto!(y, x)
  else
    _scale_l2!(y, x, r, nx)
  end
  y
end

####################################
#  ℓ2 sphere
####################################

"""
    IndSphereL2(r = 1)

The indicator of the Euclidean sphere `{x : ‖x‖₂ = r}` (the Frobenius sphere for
matrices), for `r > 0`, nonconvex. Its proximal operator scales `x` onto the
sphere, `r x / ‖x‖₂`; every point of the sphere is nearest to `x = 0`, which goes
to `r` times the first unit vector. With a norm, [`Constrained`](@ref)`(f,
IndSphereL2(r))` is the step of sparse principal component analysis.
"""
struct IndSphereL2{T<:Real} <: AbstractFunction
  r::T

  IndSphereL2{T}(r::T) where {T<:Real} = (r > 0 || throw(ArgumentError("r must be positive, got $r")); new{T}(r))
end
IndSphereL2(r::Real = 1) = IndSphereL2{typeof(r)}(r)

isproximable(::Type{<:IndSphereL2}) = true
_isindicator(::Type{<:IndSphereL2}) = true

function (f::IndSphereL2)(x::AbstractArray)
  T = float(eltype(x))
  r = convert(T, f.r)
  abs(convert(T, _norm2(x)) - r) <= _tol(T) * max(one(T), r) ? zero(T) : T(Inf)
end

function prox!(y::AbstractArray, f::IndSphereL2, x::AbstractArray, ::Real)
  _checksize(y, x)
  isempty(x) && throw(ArgumentError("the sphere has no point of size $(size(x))"))
  T = float(eltype(y))
  nx = convert(T, _norm2(x))
  r = convert(T, f.r)
  if iszero(nx)
    fill!(y, zero(eltype(y)))
    y[firstindex(y)] = r
  else
    _scale_l2!(y, x, r, nx)
  end
  y
end

# Keep the single broadcast at ordinary scales. Otherwise divide by a finite
# magnitude before taking the norm, so neither the norm nor r/n must fit.
function _scale_l2!(y, x, r, n)
  t = r / n
  if isfinite(n) && n >= floatmin(typeof(n)) && isfinite(t) && t >= floatmin(typeof(t))
    y .= t .* x
  else
    m = maximum(abs, x)
    y .= x ./ m
    y .*= r / _norm2(y)
  end
  t
end

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
_isindicator(::Type{<:IndSimplex}) = true
isconvex(::Type{<:IndSimplex}) = true

function (f::IndSimplex)(x::AbstractArray)
  T = float(eltype(x))
  t = _tol(T)
  isempty(x) && return (f.equality && f.r > t) ? T(Inf) : zero(T)
  minimum(x) >= -t || return T(Inf)
  s = convert(T, eltype(x) <: Integer ? sum(float, x) : sum(x))
  slack = t * max(one(T), f.r, abs(s))
  # A finite slack implies a finite sum; overflowed deviations already reject.
  # An overflowed upper bound contains every finite sum.
  _finite_bound(slack) || return _simplex_scaled_value(f, x, T)
  ok = f.equality ? abs(s - f.r) <= slack : s <= f.r + slack
  ok ? zero(T) : T(Inf)
end

@noinline function _simplex_scaled_value(f, x, ::Type{T}) where {T}
  m = _maximum_abs(x)
  isfinite(m) || return T(Inf)
  S = float(promote_type(T, typeof(f.r)))
  isfinite(f.r) || return _ind(!f.equality && isinf(f.r), T)
  k = _comparison_exponent(_magnitude_exponent(m), _magnitude_exponent(float(f.r)))
  s = _scaled_sum(x, _power_factors(S, -k))
  r = ldexp(S(f.r), -k)
  slack = _tol(T) * max(ldexp(one(S), -k), r, abs(s))
  _ind(f.equality ? abs(s - r) <= slack : s <= r + slack, T)
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

####################################
#  hyperplane, halfspace and hyperslab
####################################

"""
    IndHyperplane(a, b)

The indicator of the hyperplane `{x : ⟨a, x⟩ = b}`, `a ≠ 0`. Its proximal
operator is the projection `x - ((⟨a, x⟩ - b) / ‖a‖²) a`. A scalar `a` stands
for an array of `a`s of the size of `x`: `IndHyperplane(1, 0)` is the set of
arrays whose entries sum to zero, of any size. An array `a` is stored without
copying; construct a new function after changing `a` in place.

Every finite nonzero normal is accepted, including subnormal and integer
entries; `b` must be finite. The normal and `b` are scaled by exact powers of
two when needed to keep the squared norm in range. The membership value is
reliable for all finite inputs, within its floating-point tolerance. The
projection is accurate when the scaled `b`, the scaled residual and the output
are representable; inputs near `floatmax` are outside this projection guarantee.
Accuracy is limited by the less precise of the function's floating-point type
and the output type.
"""
struct IndHyperplane{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction
  a::A
  b::T
  s::NTuple{2, T} # the scale, applied as two successive powers of two
  q::T           # ‖s·a‖², or (s·a)² for a scalar a

  function IndHyperplane{A, T}(a::A, b::T) where {A<:Union{Real, AbstractArray}, T<:Real}
    s, q = _normal_cache(a, b)
    new{A, T}(a, b, s, q)
  end
end
IndHyperplane(a::Union{Real, AbstractArray}, b::Real) =
  (T = float(promote_type(eltype(a), typeof(b))); IndHyperplane{typeof(a), T}(a, convert(T, b)))

"""
    IndHalfspace(a, b)

The indicator of the halfspace `{x : ⟨a, x⟩ ≤ b}`, `a ≠ 0`. Its proximal operator
is the projection, `x - (max(⟨a, x⟩ - b, 0) / ‖a‖²) a`. A scalar `a` stands for
an array of `a`s of the size of `x`, as in [`IndHyperplane`](@ref). As there,
construct a new function after changing an array `a` in place.

Every finite nonzero normal is accepted, including subnormal and integer
entries; `b` must be finite. As for [`IndHyperplane`](@ref), powers of two
rescale the normal and `b` when needed. The membership value is reliable for all
finite inputs, within its floating-point tolerance. The projection is accurate
when the scaled `b`, the scaled residual and the output are representable;
inputs near `floatmax` are outside this projection guarantee. Accuracy is
limited by the less precise of the function's floating-point type and the
output type.
"""
struct IndHalfspace{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction
  a::A
  b::T
  s::NTuple{2, T}
  q::T

  function IndHalfspace{A, T}(a::A, b::T) where {A<:Union{Real, AbstractArray}, T<:Real}
    s, q = _normal_cache(a, b)
    new{A, T}(a, b, s, q)
  end
end
IndHalfspace(a::Union{Real, AbstractArray}, b::Real) =
  (T = float(promote_type(eltype(a), typeof(b))); IndHalfspace{typeof(a), T}(a, convert(T, b)))

"""
    IndHyperslab(a, l, u)

The indicator of the hyperslab `{x : l ≤ ⟨a, x⟩ ≤ u}`, `a ≠ 0`. Its proximal
operator is `x` inside the slab, and `x - ((⟨a, x⟩ - b) / ‖a‖²) a` otherwise,
where `b` is the violated bound. A scalar `a` stands for an array of `a`s of
the size of `x`. An array `a` is stored without copying; construct a new
function after changing it in place.

The bounds satisfy `l ≤ u`, `l < Inf` and `u > -Inf`. Equal bounds give a
hyperplane, one infinite bound a halfspace, and both infinite the whole space.
Every finite nonzero normal is accepted, including subnormal and integer
entries. As for [`IndHyperplane`](@ref), powers of two rescale the normal and
each finite bound when needed. Membership checks each bound independently,
within `sqrt(eps(T)) * max(1, abs(b), ‖a‖ ‖x‖)` in the original units, where
`T = float(eltype(x))`. For scaled normals the tolerance uses the less precise
of `T` and the function's floating-point type. The membership value is reliable
for all finite inputs. Nonfinite inputs are rejected even for the
whole space. An empty input has a projection only when `l ≤ 0 ≤ u`.

The projection is accurate when the scaled active bound, residual and output
are representable; inputs near `floatmax` are outside this guarantee. Accuracy
is limited by the less precise of the function's floating-point type and the
output type. An array of positive steps `γ` gives the projection in the metric
of `Diagonal(1 ./ γ)`, with the correction along `γ .* a`. This uses the
hyperplane's formula and also requires its weighted squared norm and correction
to be representable; extreme steps may overflow.
"""
struct IndHyperslab{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction
  a::A
  l::T
  u::T
  s::NTuple{2, T}
  q::T

  function IndHyperslab{A, T}(a::A, l::T, u::T) where {A<:Union{Real, AbstractArray}, T<:Real}
    l <= u && l < Inf && u > -Inf || throw(ArgumentError("bounds must satisfy l ≤ u, l < Inf and u > -Inf"))
    s, q = _normal_cache(a, zero(T))
    new{A, T}(a, l, u, s, q)
  end
end
IndHyperslab(a::Union{Real, AbstractArray}, l::Real, u::Real) =
  (T = float(promote_type(eltype(a), typeof(l), typeof(u))); IndHyperslab{typeof(a), T}(a, convert(T, l), convert(T, u)))

# Leave moderate normals unchanged; otherwise bring their maximum into [1, 2).
# Two factors also cover a reciprocal beyond floatmax for a subnormal maximum.
function _normal_cache(a, b::T) where {T}
  isfinite(b) || throw(ArgumentError("b must be finite"))
  m = zero(T)
  for v in a
    u = convert(T, v)
    isfinite(u) || throw(ArgumentError("a must have finite entries"))
    m = max(m, abs(u))
  end
  m > 0 || throw(ArgumentError("a must be nonzero"))
  e = min(exponent(floatmax(T)), exponent(floatmax(float(eltype(a))))) ÷ 4
  if ldexp(one(T), -e) <= m <= ldexp(one(T), e)
    s = (one(T), one(T))
    q = eltype(a) <: AbstractFloat ? convert(T, sum(abs2, a)) : sum(v -> abs2(convert(T, v)), a)
  else
    k = -exponent(m)
    k1 = min(k, exponent(floatmax(T)))
    s = (ldexp(one(T), k1), ldexp(one(T), k - k1))
    q = sum(v -> abs2(_scale_normal(v, s)), a)
  end
  s, q
end

@inline _scale_normal(v, s::NTuple{2, T}) where {T} = (convert(T, v) * s[1]) * s[2]
@inline _isscaled(f::Union{IndHyperplane, IndHalfspace, IndHyperslab}) = !isone(f.s[1])

# A view of the normalized coefficients; live buffers copy its values once.
struct _ScaledNormal{T, N, A<:AbstractArray} <: AbstractArray{T, N}
  a::A
  s::NTuple{2, T}
end
_ScaledNormal(a::AbstractArray, s::NTuple{2, T}) where {T} = _ScaledNormal{T, ndims(a), typeof(a)}(a, s)
Base.size(a::_ScaledNormal) = size(a.a)
Base.axes(a::_ScaledNormal) = axes(a.a)
Base.IndexStyle(::Type{<:_ScaledNormal{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(a::_ScaledNormal, I::Vararg{Int, N}) where {N} = _scale_normal(a.a[I...], a.s)

_normalized_normal(a::Real, s) = _scale_normal(a, s)
_normalized_normal(a::AbstractArray, s) = _ScaledNormal(a, s)
@inline _scaled_normal(f::Union{IndHyperplane, IndHalfspace, IndHyperslab}) = _isscaled(f) ? _normalized_normal(f.a, f.s) : f.a
@inline _scaled_offset(f, b) = _isscaled(f) ? _scale_normal(b, f.s) : b
@inline _scaled_offset(f::Union{IndHyperplane, IndHalfspace}) = _scaled_offset(f, f.b)

for F in (:IndHyperplane, :IndHalfspace, :IndHyperslab)
  @eval isproximable(::Type{<:$F}) = true
  @eval _isindicator(::Type{<:$F}) = true
  @eval isconvex(::Type{<:$F}) = true
end
_checkparams(f::Union{IndHyperplane, IndHalfspace}, x) = (_checkparam(f.a, x); true)

function _checkparams(f::IndHyperslab, x)
  _checkparam(f.a, x)
  f.a isa AbstractArray && Base.Broadcast.check_broadcast_axes(axes(x), f.a)
  true
end

# ⟨a, x⟩ and ‖a‖², for an array a or a scalar one
_dot_normal(a::Real, x::AbstractArray) = a * sum(x)
_dot_normal(a::AbstractArray, x::AbstractArray) = dot(a, x)
_normal_sqnorm(f::Union{IndHyperplane, IndHalfspace, IndHyperslab}, x::AbstractArray) = f.a isa Real ? f.q * length(x) : f.q
function _dot_normal(a::_ScaledNormal, x::AbstractArray)
  _checkparam(a, x)
  T = float(promote_type(eltype(a), eltype(x)))
  g = zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), a, x)
    g += a[i] * x[i]
  end
  g
end

function _stored_dot_normal(a, x)
  T = float(promote_type(eltype(a), eltype(x)))
  _scaled_dot(a, x, (one(T), one(T)))
end
_dot_normal(a::Real, x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}) = _stored_dot_normal(a, x)
_dot_normal(a::AbstractArray, x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}) = _stored_dot_normal(a, x)
_dot_normal(a::_ScaledNormal, x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}) =
  _stored_dot_normal(a, x)

# Keep the absolute floor in the original units, without forming the scale.
function _scaled_normal_slack(f, x, ::Type{T}, b = f.b) where {T}
  tol = max(_tol(T), _tol(typeof(f.q)))
  max((tol * f.s[1]) * f.s[2], tol * abs(_scaled_offset(f, b)),
      (tol * sqrt(_normal_sqnorm(f, x))) * _norm2_scaled(x))
end

function (f::IndHyperplane)(x::AbstractArray)
  T = float(eltype(x))
  eltype(x) <: Integer && return _linear_scaled_value(f, x, T)
  _isscaled(f) && return _scaled_normal_value(f, x, T)
  slack = _tol(T) * max(one(T), abs(f.b), sqrt(_normal_sqnorm(f, x) * _sumabs2(x)))
  g = _dot_normal(f.a, x) - f.b
  _finite_bound(slack) && _ordinary_residual_finite(f, x, g, typeof(slack)) ||
    return _linear_scaled_value(f, x, T)
  abs(g) <= slack ? zero(T) : T(Inf)
end
function (f::IndHalfspace)(x::AbstractArray)
  _halfspace_value(f, x, f.b)
end

function _halfspace_value(f, x, b, lower::Bool = false)
  T = float(eltype(x))
  eltype(x) <: Integer && return _linear_scaled_value(f, x, T, b, lower)
  _isscaled(f) && return _scaled_normal_value(f, x, T, b, lower)
  slack = _tol(T) * max(one(T), abs(b), sqrt(_normal_sqnorm(f, x) * _sumabs2(x)))
  g = _dot_normal(f.a, x) - b
  _finite_bound(slack) && _ordinary_residual_finite(f, x, g, typeof(slack)) ||
    return _linear_scaled_value(f, x, T, b, lower)
  _ind(lower ? -g <= slack : g <= slack, T)
end

function (f::IndHyperslab)(x::AbstractArray)
  _checkparams(f, x)
  T = float(eltype(x))
  isfinite(f.l) && !iszero(_halfspace_value(f, x, f.l, true)) && return T(Inf)
  isfinite(f.u) && return _halfspace_value(f, x, f.u)
  _ind(isfinite(_maximum_abs(x)), T)
end

# At matching precisions, the finite squared-norm bound precludes residual overflow.
@inline function _ordinary_residual_finite(f, x, g, ::Type{S}) where {S}
  P = float(promote_type(eltype(f.a), eltype(x)))
  (P === S && S <: Union{Float32, Float64}) || _finite_value(g)
end

@noinline function _scaled_normal_value(f, x, ::Type{T}, b = f.b, lower::Bool = false) where {T}
  g = _dot_normal(_normalized_normal(f.a, f.s), x) - _scaled_offset(f, b)
  slack = _scaled_normal_slack(f, x, T, b)
  _finite_value(g) && _finite_bound(slack) &&
    return _ind(f isa IndHyperplane ? abs(g) <= slack : (lower ? -g <= slack : g <= slack), T)
  _linear_scaled_value(f, x, T, b, lower)
end

# Include the normal's scale in the offset exponent before forming either one.
@noinline function _linear_scaled_value(f, x, ::Type{T}, b = f.b, lower::Bool = false) where {T}
  m = _maximum_abs(x)
  isfinite(m) || return T(Inf)
  S = promote_type(T, typeof(f.q))
  e = exponent(f.s[1]) + exponent(f.s[2])
  k = _comparison_exponent(_magnitude_exponent(m), e, _magnitude_exponent(abs(b)) + e)
  s = _power_factors(S, -k)
  b = ldexp(S(b), e - k)
  a = _normalized_normal(f.a, f.s)
  g = _scaled_dot(a, x, s) - b
  nx = sqrt(_scaled_sumabs2(x, s))
  q = f.a isa Real ? sqrt(S(f.q)) * sqrt(S(length(x))) : sqrt(S(f.q))
  tol = _isscaled(f) ? max(_tol(T), _tol(typeof(f.q))) : _tol(T)
  slack = tol * max(ldexp(one(S), e - k), abs(b), q * nx)
  _ind(f isa IndHyperplane ? abs(g) <= slack : (lower ? -g <= slack : g <= slack), T)
end

function _checkwrite(op::O, f::Union{IndHyperplane, IndHalfspace, IndHyperslab}, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checksize(y, x)
  op === value_and_prox! && _checkparams(f, x)
  if f.a isa AbstractArray
    length(f.a) == length(x) || _dot_normal(_scaled_normal(f), x)
    Base.Broadcast.check_broadcast_axes(axes(y), x, f.a)
  end
  true
end

function _hyperslab_input(f, x)
  isfinite(_maximum_abs(x)) || throw(ArgumentError("a hyperslab projection needs finite inputs"))
  isempty(x) && !(f.l <= 0 <= f.u) && throw(ArgumentError("the hyperslab contains no empty array"))
  nothing
end

# Choose the active bound from the residual, independently of membership slack.
function _hyperslab_residual(f, x)
  a = _scaled_normal(f)
  # Floating-point products avoid wraparound for integer inputs and normals.
  s = eltype(x) <: Integer ? _stored_dot_normal(a, x) : _dot_normal(a, x)
  if isfinite(f.l)
    g = s - _scaled_offset(f, f.l)
    g < 0 && return g
  end
  if isfinite(f.u)
    g = s - _scaled_offset(f, f.u)
    g > 0 && return g
  end
  zero(s)
end

function prox!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _hyperslab_input(f, x)
  isempty(x) && return y
  g = _hyperslab_residual(f, x)
  iszero(g) && return (y === x ? y : copyto!(y, x))
  t = g / _normal_sqnorm(f, x)
  y .= x .- t .* _scaled_normal(f)
end

function prox!(y::AbstractArray, f::IndHyperplane, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _isscaled(f) && return _scaled_normal_prox!(y, f, x)
  T = float(eltype(y))
  t = convert(T, (_dot_normal(f.a, x) - f.b) / _normal_sqnorm(f, x))
  y .= x .- t .* f.a
end
function prox!(y::AbstractArray, f::IndHalfspace, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _isscaled(f) && return _scaled_normal_prox!(y, f, x)
  T = float(eltype(y))
  t = convert(T, max(_dot_normal(f.a, x) - f.b, zero(f.b)) / _normal_sqnorm(f, x))
  y .= x .- t .* f.a
end

function _scaled_normal_prox!(y, f, x)
  T = float(eltype(y))
  a = _normalized_normal(f.a, f.s)
  g = _dot_normal(a, x) - _scaled_offset(f)
  f isa IndHalfspace && (g = max(g, zero(g)))
  t = convert(T, g / _normal_sqnorm(f, x))
  y .= x .- t .* a
end
