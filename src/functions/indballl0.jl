"""
    IndBallL0(s; diagonal = true)

Indicator of the set with at most `s` counted nonzero entries. The budget is a
nonnegative integer, saturated at `typemax(Int)`. The value is `0` for a feasible
argument and `Inf` otherwise. A counted NaN gives NaN at any budget. Counted
infinities are nonzero, and either sign of zero counts as zero.

The input representation determines the projection.

- A dense `Vector`, a dense `Matrix` (including rectangular matrices), or a
  [`SparseIterate`](@ref) vector or matrix has independent entries. The
  projection retains up to `s` counted nonzeros of largest magnitude. Ties go
  to the earlier logical column-major coordinate.
- A `Symmetric` wrapper over a dense matrix or a [`SymmetricSparseIterate`](@ref)
  has symmetric entries. A nonzero diagonal entry costs one and a nonzero
  off-diagonal pair costs two. With counted diagonals, the projection compares
  the feasible splits between diagonals and pairs to minimize squared
  Frobenius error. With free diagonals it retains up to `s ÷ 2` pairs of largest
  magnitude. Ties within each class go to the earlier diagonal or strict
  lower-triangle coordinate in column-major order. Equal savings across
  feasible splits favor fewer pairs.

With `diagonal = false`, matrix diagonals are free and are copied verbatim.
The flag has no effect on vectors. For a rectangular matrix the diagonal
has length `min(size(X)...)`.

```jldoctest
julia> X = [5.0 -3.0; -3.0 7.0]; f = IndBallL0(1; diagonal = false);

julia> Tuple(prox(f, X))
(5.0, -3.0, 0.0, 7.0)

julia> Tuple(prox(f, Symmetric(X, :L)))
(5.0, 0.0, 0.0, 7.0)

julia> Tuple(prox(IndBallL0(2), Symmetric([1.0 7.0; 7.0 2.0], :L)))
(0.0, 7.0, 7.0, 0.0)
```

Input and output elements must have the same type, `Float32` or `Float64`.
Dense output from `Symmetric` is an ordinary matrix, with both entries of
each pair written from the chosen triangle. It may be the wrapper's parent.
The other inputs use a same-kind output and support input/output identity.
For the two iterate kinds, a distinct output receives the input working set
and its slot order. Projection zeros dropped values without deactivating
their slots. Other overlaps, mixed owner/dense calls, and wrapper outputs
such as `Symmetric` destinations are outside the supported interface.

Projection requires finite counted entries and a finite nonnegative real
scalar step `γ`. A zero step also projects. Shape and step errors are checked
before writing. The projection copies selected entries without arithmetic and
zeros the remaining counted nonzeros. [`value_and_prox!`](@ref) returns the
typed zero value of that stored result. One selection buffer belongs to the
function object. Use a separate object for each concurrent projection.

A single `Scaled(f, α)` forwards `α * γ` on a positive step, which must remain
finite. A zero step stays zero, and nested indicator scaling uses the original
step when the factors cannot be folded into one wrapper.
"""
struct IndBallL0 <: AbstractFunction
  s::Int
  diagonal::Bool
  scratch::Scratch

  function IndBallL0(s::Integer; diagonal::Bool = true)
    s >= 0 || throw(ArgumentError("s must be nonnegative, got $s"))
    new(s > typemax(Int) ? typemax(Int) : Int(s), diagonal, Scratch())
  end
end

isconvex(::Type{IndBallL0}) = false
isproximable(::Type{IndBallL0}) = true
_isindicator(::Type{IndBallL0}) = true

# The buffer is not part of the function.
Base.:(==)(f::IndBallL0, g::IndBallL0) = f.s == g.s && f.diagonal == g.diagonal
Base.hash(f::IndBallL0, h::UInt) = hash(f.diagonal, hash(f.s, hash(:IndBallL0, h)))
Base.show(io::IO, f::IndBallL0) = print(io, "IndBallL0(", f.s, "; diagonal = ", f.diagonal, ")")

const _L0Float = Union{Float32, Float64}
const _L0Dense{T} = Union{Vector{T}, Matrix{T}}
const _L0Symmetric{T} = Symmetric{T, Matrix{T}}
const _L0Sparse{T} = Union{SparseIterate{T, 1}, SparseIterate{T, 2}}

@noinline _l0_unsupported(what, T) = throw(ArgumentError(
  "IndBallL0 supports dense vectors and matrices, Symmetric of a dense matrix, SparseIterate " *
  "vectors and matrices, and SymmetricSparseIterate, with homogeneous Float32 or Float64 and a " *
  "scalar step. Got $what of type $T"))

@noinline _l0_step_error(γ) = throw(ArgumentError(
  "the step of IndBallL0 must be finite and nonnegative, got $γ"))
@noinline _l0_nonfinite() = throw(ArgumentError("IndBallL0 requires finite counted entries for projection"))

(f::IndBallL0)(X) = _l0_unsupported("the argument", typeof(X))
prox!(Y, f::IndBallL0, X, γ) = _l0_unsupported("the output, input or step", typeof.((Y, X, γ)))
prox!(Y::SymmetricSparseIterate, f::IndBallL0, X, γ::AbstractArray) = _l0_unsupported("the step", typeof(γ))
function prox!(Y, f::IndBallL0, X, γ::Real)
  _l0_project!(Y, f, X, γ)
  Y
end
function value_and_prox!(Y, f::IndBallL0, X, γ::Real)
  _l0_project!(Y, f, X, γ)
  zero(eltype(Y))
end
value_and_prox!(Y, f::IndBallL0, X, γ::AbstractArray) = _l0_unsupported("the step", typeof(γ))
value_and_prox!(Y::SymmetricSparseIterate, f::IndBallL0, X, γ::AbstractArray) = _l0_unsupported("the step", typeof(γ))
_value_and_prox_unchecked!(Y, f::IndBallL0, X, γ::Real) = value_and_prox!(Y, f, X, γ)

_l0_project!(Y, f, X, γ) = _l0_unsupported("the output or input", typeof.((Y, X)))
@inline function _l0_preflight(Y, X, γ)
  size(Y) == size(X) || throw(DimensionMismatch("the output has size $(size(Y)) and the input $(size(X))"))
  (isfinite(γ) && γ >= 0) || _l0_step_error(γ)
  nothing
end

_l0_free_diagonal(f, x::AbstractVector) = nothing
function _l0_free_diagonal(f, x::AbstractMatrix)
  f.diagonal && return nothing
  stride = size(x, 1) + 1
  stride, 1 + (min(size(x)...) - 1) * stride
end
@inline _l0_free(::Nothing, id) = false
@inline _l0_free(diag::Tuple, id) = id <= diag[2] && rem(id - 1, diag[1]) == 0

@inline _l0_invalid(v, ::Val{:value}) = isnan(v)
@inline _l0_invalid(v, ::Val{:prox}) = !isfinite(v)

# Count and classify together. A free diagonal does not affect either result.
function _l0_entry_stats(values, ids, diag, mode)
  n, bad = 0, false
  @inbounds @simd for a in eachindex(values)
    counted = !_l0_free(diag, ids[a])
    v = values[a]
    n += counted & !iszero(v)
    bad |= counted & _l0_invalid(v, mode)
  end
  n, bad
end

@inline _l0_value(::Type{T}, f, n, nan) where {T} = nan ? T(NaN) : n <= f.s ? zero(T) : T(Inf)
function (f::IndBallL0)(x::_L0Dense{T}) where {T<:_L0Float}
  n, nan = _l0_entry_stats(x, eachindex(x), _l0_free_diagonal(f, x), Val(:value))
  _l0_value(T, f, n, nan)
end
function (f::IndBallL0)(x::_L0Sparse{T}) where {T<:_L0Float}
  n, nan = _l0_entry_stats(activevalues(x), activecoordinates(x), _l0_free_diagonal(f, x), Val(:value))
  _l0_value(T, f, n, nan)
end

function _l0_project!(y::Array{T, N}, f::IndBallL0, x::Array{T, N}, γ) where {T<:_L0Float, N}
  N in (1, 2) || _l0_unsupported("the argument", typeof(x))
  _l0_preflight(y, x, γ)
  ids = eachindex(x)
  diag = _l0_free_diagonal(f, x)
  n, bad = _l0_entry_stats(x, ids, diag, Val(:prox))
  bad && _l0_nonfinite()
  if n <= f.s
    y === x || copyto!(y, x)
    return
  end
  τ, quota = T(Inf), 0
  if f.s > 0
    # Nonzero magnitudes suffice, since this branch has more than s of them.
    b = _array_buffer!(f.scratch, T, (n,))
    j = 0
    @inbounds for i in ids
      v = x[i]
      if !_l0_free(diag, i) && !iszero(v)
        b[j += 1] = abs(v)
      end
    end
    τ = _l0_kth_largest!(b, f.s)
    quota = f.s - _l0_count_above(b, τ)
  end
  @inbounds for i in ids
    v = x[i]
    a = abs(v)
    free = _l0_free(diag, i)
    tied = !free && a == τ && quota > 0
    quota -= tied
    y[i] = (free || a > τ || tied) ? v : zero(T)
  end
  nothing
end

# Rank stored entries by magnitude then by logical coordinate. Inserting the
# coordinate into the selection key avoids sorting ties or moving active slots.
@inline _l0_key(v, id) = (abs(v), -id)

function _l0_project!(y::SparseIterate{T, N}, f::IndBallL0, x::SparseIterate{T, N}, γ) where {T<:_L0Float, N}
  N in (1, 2) || _l0_unsupported("the argument", typeof(x))
  _l0_preflight(y, x, γ)
  values, ids = activevalues(x), activecoordinates(x)
  diag = _l0_free_diagonal(f, x)
  n, bad = _l0_entry_stats(values, ids, diag, Val(:prox))
  bad && _l0_nonfinite()
  y === x || copyto!(y, x)
  n <= f.s && return
  τ = (T(Inf), 0)
  if f.s > 0
    b = _array_buffer!(f.scratch, Tuple{T, Int}, (n,))
    j = 0
    @inbounds for a in eachindex(values)
      id, v = ids[a], values[a]
      if !_l0_free(diag, id) && !iszero(v)
        b[j += 1] = _l0_key(v, id)
      end
    end
    τ = _l0_kth_largest!(b, f.s)
  end
  out = activevalues(y)
  @inbounds for a in eachindex(out)
    if !_l0_free(diag, ids[a]) && isless(_l0_key(out[a], ids[a]), τ)
      out[a] = zero(T)
    end
  end
  nothing
end

# the strict lower magnitudes in column-major order
function _l0_fill!(b, X, p)
  i = 0
  @inbounds for c in 1:p, r in c+1:p
    i += 1
    b[i] = abs(X[r, c])
  end
  b
end

function _l0_count_above(b, τ)
  a = 0
  @inbounds for i in eachindex(b)
    a += b[i] > τ
  end
  a
end

# The k-th largest of the keys in b, 1 ≤ k ≤ length(b), by an in-place
# quickselect with a three-way partition and the median of three entries at
# positions from a local xorshift generator (seeded by length(b), so the result and
# the permutation of b are the same for the same input). The entries equal to the pivot leave
# the range of the later rounds. The order of b changes.
@inline function _l0_xorshift(x::UInt64)
  x ⊻= x << 13
  x ⊻= x >> 7
  x ⊻= x << 17
  x
end

@inline _l0_lt(a, b) = isless(a, b)
@inline _l0_lt(a::AbstractFloat, b::AbstractFloat) = a < b

function _l0_kth_largest!(b::AbstractVector, k::Int)
  lo, hi = 1, length(b)
  state = UInt64(length(b)) | 0x9e3779b97f4a7c15
  @inbounds while lo < hi
    width = UInt64(hi - lo + 1)
    state = _l0_xorshift(state)
    x = b[lo + Int(state % width)]
    state = _l0_xorshift(state)
    y = b[lo + Int(state % width)]
    state = _l0_xorshift(state)
    z = b[lo + Int(state % width)]
    pivot = max(min(x, y), min(max(x, y), z))
    # [lo, lt) above the pivot, [lt, i) equal to it, (gt, hi] below it
    i, lt, gt = lo, lo, hi
    while i <= gt
      v = b[i]
      if _l0_lt(pivot, v)
        b[i] = b[lt]
        b[lt] = v
        lt += 1
        i += 1
      elseif _l0_lt(v, pivot)
        b[i] = b[gt]
        b[gt] = v
        gt -= 1
      else
        i += 1
      end
    end
    if k < lt
      hi = lt - 1
    elseif k > gt
      lo = gt + 1
    else
      return pivot
    end
  end
  @inbounds b[k]
end

# Return the number of nonzero diagonals and pairs separately. The same visit
# detects counted NaNs for value and rejects nonfinite counted data for prox.
function _l0_symmetric_stats(x::AbstractMatrix, diagonal, mode)
  d, e, bad = 0, 0, false
  p = size(x, 1)
  @inbounds for c in 1:p
    if diagonal
      v = x[c, c]
      d += !iszero(v)
      bad |= _l0_invalid(v, mode)
    end
    @simd for r in c+1:p
      v = x[r, c]
      e += !iszero(v)
      bad |= _l0_invalid(v, mode)
    end
  end
  d, e, bad
end

@inline function _l0_isdiag(p, id)
  r, c = tril_cartesian(p, id)
  r == c
end

function _l0_symmetric_stats(x::SymmetricSparseIterate, diagonal, mode)
  d, e, bad = 0, 0, false
  vals, ids = activevalues(x), activecoordinates(x)
  p = size(x, 1)
  @inbounds for a in eachindex(vals)
    v = vals[a]
    iszero(v) && continue
    if _l0_isdiag(p, ids[a])
      diagonal || continue
      d += 1
    else
      e += 1
    end
    bad |= _l0_invalid(v, mode)
  end
  d, e, bad
end

# Even if the owner represents a very large logical matrix, its full-entry
# count is tested without multiplying the number of stored pairs.
@inline _l0_feasible(s, d, e) = d <= s && e <= (s - d) ÷ 2

# Make the trusted triangle accessible as lower-triangle entries. In particular,
# an upper-triangle source does not read the unselected half of the parent.
@inline _l0_lower(x::Symmetric) = x.uplo == 'L' ? parent(x) : transpose(parent(x))

function (f::IndBallL0)(x::Union{_L0Symmetric{T}, SymmetricSparseIterate{T}}) where {T<:_L0Float}
  _l0_symmetric_value(T, f, x isa Symmetric ? _l0_lower(x) : x)
end
function _l0_symmetric_value(::Type{T}, f, x) where {T}
  d, e, nan = _l0_symmetric_stats(x, f.diagonal, Val(:value))
  nan ? T(NaN) : _l0_feasible(f.s, d, e) ? zero(T) : T(Inf)
end

prox(f::IndBallL0, x::SymmetricSparseIterate{T}, γ::Real = 1) where {T<:_L0Float} =
  prox!(similar(x), f, x, γ)
prox(f::IndBallL0, x::SymmetricSparseIterate, γ::Real = 1) =
  _l0_unsupported("the argument", typeof(x))
prox(f::IndBallL0, x::SymmetricSparseIterate, γ::AbstractArray) =
  _l0_unsupported("the step", typeof(γ))

function _l0_project!(y::Matrix{T}, f::IndBallL0, x::_L0Symmetric{T}, γ) where {T<:_L0Float}
  _l0_preflight(y, x, γ)
  _l0_project_symmetric!(y, f, _l0_lower(x))
end

function _l0_project_symmetric!(y::Matrix{T}, f, x) where {T}
  d, e, bad = _l0_symmetric_stats(x, f.diagonal, Val(:prox))
  bad && _l0_nonfinite()
  keepall = (zero(T), 0)
  dropall = (T(Inf), 0)
  diag, pairs = keepall, keepall
  p = size(x, 1)
  if !_l0_feasible(f.s, d, e)
    if !f.diagonal
      k = f.s ÷ 2
      pairs = dropall
      if k > 0
        b = _array_buffer!(f.scratch, T, (p * (p - 1) ÷ 2,))
        _l0_fill!(b, x, p)
        τ = _l0_kth_largest!(b, k)
        pairs = (τ, k - _l0_count_above(b, τ))
      end
    elseif f.s == 0
      diag, pairs = dropall, dropall
    else
      b = _array_buffer!(f.scratch, T, (d + e,))
      bd, bp = view(b, 1:d), view(b, d+1:d+e)
      id, ip = 0, 0
      @inbounds for c in 1:p
        v = x[c, c]
        iszero(v) || (bd[id += 1] = abs(v))
        for r in c+1:p
          v = x[r, c]
          iszero(v) || (bp[ip += 1] = abs(v))
        end
      end
      kd, kp = _l0_best_split!(bd, bp, f.s)
      diag = _l0_threshold(bd, kd, keepall, dropall)
      pairs = _l0_threshold(bp, kp, keepall, dropall)
    end
  end
  _l0_write_symmetric!(y, x, diag, pairs, p)
end

function _l0_threshold(b, k, keepall, dropall)
  k == length(b) && return keepall
  k == 0 && return dropall
  @inbounds τ = b[k]
  (τ, k - _l0_count_above(b, τ))
end

# A threshold of zero copies even signed zeros. A strictly positive threshold
# writes positive zero to dropped entries. Both writes for a pair follow the
# read of its single source slot, so y may be the parent of the input.
function _l0_write_symmetric!(y, x, (τd, td), (τp, tp), p)
  @inbounds for c in 1:p
    v = x[c, c]
    a = abs(v)
    tie = a == τd && td > 0
    td -= tie
    y[c, c] = (iszero(τd) || a > τd || tie) ? v : zero(v)
    for r in c+1:p
      v = x[r, c]
      a = abs(v)
      tie = a == τp && tp > 0
      tp -= tie
      w = (iszero(τp) || a > τp || tie) ? v : zero(v)
      y[r, c] = y[c, r] = w
    end
  end
  nothing
end

function _l0_project!(y::SymmetricSparseIterate{T}, f::IndBallL0, x::SymmetricSparseIterate{T}, γ) where {T<:_L0Float}
  _l0_preflight(y, x, γ)
  d, e, bad = _l0_symmetric_stats(x, f.diagonal, Val(:prox))
  bad && _l0_nonfinite()
  y === x || copyto!(y, x)
  _l0_feasible(f.s, d, e) && return

  keepall = (zero(T), 0)
  dropall = (T(Inf), 0)
  diag = f.diagonal ? dropall : keepall
  pairs = dropall
  vals, ids = activevalues(y), activecoordinates(y)
  p = size(y, 1)
  if f.s > (f.diagonal ? 0 : 1)
    b = _array_buffer!(f.scratch, Tuple{T, Int}, (d + e,))
    bd, bp = view(b, 1:d), view(b, d+1:d+e)
    id, ip = 0, 0
    @inbounds for a in eachindex(vals)
      v = vals[a]
      iszero(v) && continue
      k = ids[a]
      if _l0_isdiag(p, k)
        f.diagonal && (bd[id += 1] = _l0_key(v, k))
      else
        bp[ip += 1] = _l0_key(v, k)
      end
    end
    if f.diagonal
      kd, kp = _l0_best_split!(bd, bp, f.s)
      diag = _l0_stored_threshold(bd, kd, keepall, dropall)
      pairs = _l0_stored_threshold(bp, kp, keepall, dropall)
    else
      pairs = _l0_kth_largest!(bp, f.s ÷ 2)
    end
  end
  @inbounds for a in eachindex(vals)
    k = ids[a]
    τ = _l0_isdiag(p, k) ? diag : pairs
    if !iszero(first(τ)) && isless(_l0_key(vals[a], k), τ)
      vals[a] = zero(T)
    end
  end
  nothing
end

function _l0_stored_threshold(b, k, keepall, dropall)
  k == length(b) && return keepall
  k == 0 && return dropall
  @inbounds b[k]
end

_l0_magnitude(v::AbstractFloat) = v
_l0_magnitude(v::Tuple) = first(v)

function _l0_sort_prefix!(b, k)
  k == 0 && return
  k < length(b) && _l0_kth_largest!(b, k)
  sort!(view(b, 1:k); rev = true, alg = QuickSort)
  nothing
end

# For each e, the best support uses the largest e pairs and largest
# min(D, s - 2e) diagonals. Increasing e saves the square of the next pair
# twice and loses the squares of up to two diagonals. Marginal savings are
# nonincreasing: the next pair is no larger, and dropped diagonals get larger
# as their number goes from zero to one to two. The first nonpositive
# marginal therefore gives the smallest maximizing e.
function _l0_best_split!(bd, bp, s)
  D = length(bd)
  Kd, Kp = min(D, s), min(length(bp), s ÷ 2)
  _l0_sort_prefix!(bd, Kd)
  _l0_sort_prefix!(bp, Kp)
  e = min(Kp, max(0, (s - D) ÷ 2))
  d = min(D, s - 2e)
  @inbounds while e < Kp
    nextd = min(D, s - 2(e + 1))
    pair = _l0_magnitude(bp[e + 1])
    a = _l0_magnitude(bd[nextd + 1])
    b = d - nextd == 2 ? _l0_magnitude(bd[d]) : zero(a)
    _l0_pair_wins(pair, a, b) || break
    d, e = nextd, e + 1
  end
  d, e
end

# Extract a binary significand square in 106 bits from either supported float.
@inline function _l0_square_parts(x)
  m, e = frexp(Float64(x))
  sig = UInt128(ldexp(m, 53))
  sig * sig, e
end

# Compare 2pair² to a² + b² (a ≥ b) without forming a rounded or out-of-range
# floating square. In the mixed ordering b < pair < a, a two-exponent gap
# between a and pair gives a > 2pair. Otherwise a² aligns exactly in at most
# 108 bits. After subtraction the remaining integer must strictly exceed
# b² in the same units; taking the floor of b² suffices for that comparison.
@inline function _l0_pair_wins(pair, a, b)
  pair >= a && return pair > b
  pair <= b && return false
  qa, ea = _l0_square_parts(a)
  qp, ep = _l0_square_parts(pair)
  ea - ep >= 2 && return false
  twice = qp << 1
  loss = qa << (2 * (ea - ep))
  twice <= loss && return false
  iszero(b) && return true
  qb, eb = _l0_square_parts(b)
  twice - loss > (qb >> (2 * (ep - eb)))
end
