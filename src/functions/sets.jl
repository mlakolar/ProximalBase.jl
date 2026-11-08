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
proximal operator keeps the `s` entries of largest magnitude (exactly `s` of the
nonzero ones when there are more, ties included) and sets the others to zero: a selection in O(n) expected time
(Floyd and Rivest's algorithm), reusing a scratch buffer, and in O(nnz) on a
[`SparseIterate`](@ref). The projection does not depend on the step `γ`, which
may be any scalar in `[0, Inf]`; a negative or NaN step throws an `ArgumentError`
before anything is written.

For a symmetric matrix, a `Symmetric` or a [`SymmetricSparseIterate`](@ref), `s` counts the entries of
the matrix itself: with `d` nonzero diagonal entries and `e` nonzero off-diagonal pairs (an entry and its
transpose, `xᵢⱼ = xⱼᵢ`), the set is `d + 2e ≤ s`. A counted diagonal entry costs one unit of `s` and a pair
two. With `diagonal = false` the diagonal does not count and is kept, and the set is `2e ≤ s`, as for the
off-diagonal sparsity of a precision matrix:

```julia
prox!(S, IndBallL0(s; diagonal = false), Symmetric(Θ, :L))   # S::SymmetricSparseIterate
```

Explicit and implicit zeros, of either sign, are zero, so the count is that of a dense matrix and of a
`SymmetricSparseIterate` alike; the value at an owner reads its stored coefficients only. The output is
symmetric and may be a dense matrix or a `SymmetricSparseIterate`.

The result is the projection in the Frobenius norm. Keeping a nonzero diagonal entry costs one unit of `s`
and saves `xᵢᵢ²` of the squared error, and keeping a nonzero pair costs two units and saves `2xᵢⱼ²`. The
projection keeps the `d` diagonal entries of largest magnitude and the `e` pairs of largest magnitude, for
the `e` with `d = min(D, s - 2e)` at which the saving is largest (`D` is the number of nonzero counted
diagonal entries); with `diagonal = false` it keeps the `min(E, fld(s, 2))` nonzero pairs of largest magnitude
(`E` is the number of nonzero pairs; a larger budget leaves the matrix unchanged). Within
diagonal entries, and within pairs, a tie in magnitude is broken for the entry earlier in the lower
triangle read by columns, and between two choices of `e` with equal saving the one with the smaller `e` is
taken. The ranking inside the diagonal entries, and inside the pairs, is exact for `Float32` and `Float64`
entries, whose magnitudes the working type represents exactly; other element types are ranked after
conversion to the working float type (`Float64` for integers and rationals, the type itself for other floats). The
number of pairs kept is chosen by comparing the savings in floating point, with the squares formed from
magnitudes scaled by a power of two so that the largest finite one is at most 1 (they cannot overflow;
`Float32` sums in `Float64`): the choice is a maximizer of the saved squared error up to the rounding of the
sums of squares in that comparison (of the converted magnitudes for the other element types), so that a
difference below that rounding can be decided either way. An infinite entry is a limit of finite magnitudes: it
is kept in preference to any finite one, the choices compared first by the weight of the infinite entries
they keep (one for a diagonal entry, two for a pair) and then by the finite saving. The cost is O(p²) for
the selection plus a sort of the kept candidates; for `Float32` or `Float64` entries of a fixed size the call
does not allocate after a warm-up call (the scratch buffer is reused), while larger problems and other element
types such as `BigFloat` can allocate.

A NaN entry is undefined (E2 in the numerical contract). Let `m` be the largest number of entries that
count: `p²` for a symmetric `p × p` matrix (`p(p - 1)` with `diagonal = false`, where the diagonal does not
count); for any other array, the number of its entries that count (its length, less the diagonal of a matrix
with `diagonal = false`). For `0 < s < m` a NaN in a counted entry makes every counted
output NaN, and so is the value `value_and_prox!` returns for it; a free diagonal is copied. For `s ≥ m`
the projection is the identity, NaN included, and its value is `0` (`s` is compared with `m` without
overflow for any `p`). For `s = 0` the counted block of the projection is zero (a free diagonal is kept, NaN included) and its fused
value is `0`.
The value of the function at an argument with a NaN in a counted entry is NaN for every `s < m`, since the
argument may or may not be in the set. When the NaN block is written into an output of integer element type,
the `InexactError` is thrown before anything is written.
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
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndBallL0, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); true)
_isindicator(::Type{IndBallL0}) = true
_counts_nonzeros(::IndBallL0) = true
_scale_invariant(::IndBallL0) = true

# whether entry i of x is one that counts
@inline _counts(f::IndBallL0, x::AbstractArray, i::Integer) = true
@inline function _counts(f::IndBallL0, x::AbstractMatrix, i::Integer)
  f.diagonal && return true
  m = size(x, 1)
  mod1(i, m) != cld(i, m)
end

# whether every entry of x counts, as unless x is a matrix with a free diagonal
_allcount(f::IndBallL0, x::AbstractArray) = f.diagonal || ndims(x) != 2

# the number of counted entries that are not zero (a NaN is not zero), and whether a counted entry is NaN,
# in the one pass over x that the count takes anyway
function _count_nan(f::IndBallL0, x::AbstractArray)
  _allcount(f, x) && return _count_nan_all(x)
  nz, nan = 0, false
  @inbounds for i in eachindex(IndexLinear(), x)
    v, c = x[i], _counts(f, x, i)
    nz += (!iszero(v) & c)
    nan |= (isnan(v) & c)
  end
  nz, nan
end
function _count_nan(f::IndBallL0, x::SparseIterate)
  f.diagonal || ndims(x) == 1 || return invoke(_count_nan, Tuple{IndBallL0, AbstractArray}, f, x)
  _count_nan_values(view(x.nzval, 1:x.nnz))
end

# Every entry counts: the counts of the nonzero entries and of the NaN ones, from the stored values of a
# standard sparse array (its other entries are zero) and from all entries of any other array.
_count_nan_all(x::AbstractArray) = _count_nan_values(x)
_count_nan_all(x::SparseVector) = _count_nan_values(nonzeros(x))
_count_nan_all(x::SparseMatrixCSC) = _count_nan_values(view(nonzeros(x), 1:nnz(x)))

# The IEEE types are counted with a counter as narrow as the entries (a 32-bit counter for a 32-bit entry), so
# that the masks of the comparison are added without widening; an array with more than 2^30 entries is counted in
# blocks, so that the counter cannot overflow. NaN is found by carrying the sum of the entries, which is NaN
# whenever an entry is NaN and costs one addition per entry, less than a second comparison; the sum of finite and
# infinite entries can also be NaN (+Inf and -Inf, or an overflow), and then the entries are searched for a NaN,
# a cold path.
function _count_nan_values(x::AbstractArray{T}) where {T<:Base.IEEEFloat}
  length(x) > 2^30 && return _count_nan_blocks(x)
  a = zero(sizeof(T) <= 4 ? Int32 : Int64)
  z = zero(T)
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    v = x[i]
    a += !iszero(v)
    z += v
  end
  Int(a), isnan(z) && _l0_anynan(x)
end
@noinline function _count_nan_blocks(x::AbstractArray{T}) where {T<:Base.IEEEFloat}
  nz, z = 0, zero(T)
  r = eachindex(IndexLinear(), x)
  lo = first(r)
  while lo <= last(r)
    hi = min(lo + (2^30 - 1), last(r))
    a = zero(Int64)
    @inbounds @simd for i in lo:hi
      v = x[i]
      a += !iszero(v)
      z += v
    end
    nz += a
    lo = hi + 1
  end
  nz, isnan(z) && _l0_anynan(x)
end
@noinline _l0_anynan(x) = any(isnan, x)
function _count_nan_values(x::AbstractArray)
  nz, nan = 0, false
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    v = x[i]
    nz += !iszero(v)
    nan |= isnan(v)
  end
  nz, nan
end

# the number of entries that count: one per coordinate, a free diagonal excluded
_ncounted(f::IndBallL0, x::AbstractArray) = _allcount(f, x) ? length(x) : length(x) - min(size(x, 1), size(x, 2))

# The largest full count of a symmetric p × p matrix: every diagonal entry counts one and every off-diagonal
# pair two, so p² with a counted diagonal and p(p - 1) without. In Int128, so that p² cannot overflow Int.
_ncounted_symmetric(f::IndBallL0, p::Integer) = (q = Int128(p); f.diagonal ? q * q : q * (q - 1))

# E2: with 0 < s < m counted entries, a NaN in one decides whether the point is in the set and which
# entries are kept, so every counted output and the value depend on it; s ≥ m is the identity and
# s = 0 the zero map
@inline _nan_block_applies(f::IndBallL0, nan::Bool, m::Integer) = nan && 0 < f.s < m

function (f::IndBallL0)(x::AbstractArray)
  _checkreal(f, x)
  T = _valtype(x)
  f.s >= _ncounted(f, x) && return zero(T)     # the whole space: in the set whatever the entries are, NaN included
  nz, nan = _count_nan(f, x)
  nan ? convert(T, NaN) : _ind(nz <= f.s, T)
end

# A symmetric matrix counts its full logical entries: a nonzero diagonal entry counts one (unless the
# diagonal is free) and a nonzero off-diagonal pair counts two. The entries are read from the lower triangle.
(f::IndBallL0)(x::Symmetric) = _symmetric_l0_value(f, x)
(f::IndBallL0)(x::SymmetricSparseIterate) = _symmetric_l0_value(f, x)
function _symmetric_l0_value(f::IndBallL0, x)
  _checkreal(f, x)
  p = checksquare(x)
  T = _valtype(x)
  f.s >= _ncounted_symmetric(f, p) && return zero(T)     # the whole space
  k, nan = 0, false
  @inbounds for c in 1:p
    if f.diagonal
      v = x[c, c]
      k += !iszero(v)
      nan |= isnan(v)
    end
    for r in c + 1:p
      v = x[r, c]
      k += 2 * !iszero(v)
      nan |= isnan(v)
    end
  end
  nan ? convert(T, NaN) : _ind(k <= f.s, T)
end
# the stored coefficients hold every nonzero entry: no implicit zero is read
function _symmetric_l0_value(f::IndBallL0, x::SymmetricSparseIterate)
  _checkreal(f, x)
  T = _valtype(x)
  f.s >= _ncounted_symmetric(f, x.p) && return zero(T)   # the whole space
  k, nan = 0, false
  @inbounds for j in 1:x.nnz
    v = x.nzval[j]
    r, c = tril_cartesian(x.p, x.nzval2ind[j])
    if r == c
      f.diagonal || continue
      k += !iszero(v)
    else
      k += 2 * !iszero(v)
    end
    nan |= isnan(v)
  end
  nan ? convert(T, NaN) : _ind(k <= f.s, T)
end

function prox!(y::AbstractArray, f::IndBallL0, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  _direct_prox!(y, f, x)
  y
end

# The projection into y; true when it stored the NaN block (E2 above), whose value is NaN (O2).
function _value_and_prox_body!(y, f::IndBallL0, x, γ::Real = 1)
  _checkwrite(value_and_prox!, f, y, x)      # the shapes before the step, as the generic body does
  _checkstep_nonneg(γ)
  _direct_prox!(y, f, x) ? convert(_valtype(y), NaN) : zero(_valtype(y))
end

function _direct_prox!(y::AbstractArray, f::IndBallL0, x::AbstractArray)
  m = _ncounted(f, x)
  if f.s >= m                           # the whole space: the identity, NaN included, without a count
    y === x || copyto!(y, x)
    return false
  end
  nz, nan = _count_nan(f, x)
  if _nan_block_applies(f, nan, m)
    return _nan_counted!(y, f, x)
  elseif nz <= f.s
    y === x || copyto!(y, x)
    return false
  end
  _allcount(f, x) ? _keep_largest_all!(y, f, x) : _keep_largest_counted!(y, f, x)
  false
end

# NaN in every counted entry; a free diagonal is copied. The NaN is converted before anything is written.
function _nan_counted!(y::AbstractArray, f::IndBallL0, x::AbstractArray)
  _allcount(f, x) && return _nan_block!(y)
  nanv = convert(eltype(y), NaN)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = _counts(f, x, i) ? nanv : x[i]
  end
  true
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
  _checkstep_nonneg(γ)
  _checkwrite(prox!, f, y, x)
  _direct_prox!(y, f, x)
  y
end
function _direct_prox!(y::SparseIterate{<:Any, 1}, f::IndBallL0, x::SparseIterate{<:Any, 1})
  if f.s >= length(x)                   # the whole space: the identity, NaN included
    y === x || copyto!(y, x)
    return false
  end
  nz, nan = _count_nan(f, x)
  if _nan_block_applies(f, nan, length(x))
    return _nan_block!(y)               # every coordinate of the vector is NaN, and stored
  elseif nz <= f.s
    y === x || copyto!(y, x)
    return false
  elseif f.s == 0
    @inbounds for k in 1:y.nnz          # every entry is zero; y keeps its stored entries
      y.nzval[k] = zero(eltype(y))
    end
    return false
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
  false
end

# A symmetric matrix, written to both triangles of a dense output and to the lower triangle of an owner.
# The set counts full entries: a nonzero diagonal entry costs 1 of the budget s and a nonzero off-diagonal
# pair 2. Keeping a diagonal entry saves x_ii² of the squared error and keeping a pair saves 2x_ij², so the
# projection chooses the d diagonal entries and e pairs with d + 2e ≤ s that save the most: the d largest
# diagonal entries and the e largest pairs by magnitude, for the e at which the total is largest.
function prox!(y::AbstractMatrix, f::IndBallL0, x::Symmetric, γ::Real)
  _checkstep_nonneg(γ)
  _checkwrite(prox!, f, y, x)
  _direct_prox!(y, f, x)
  y
end
function _direct_prox!(y::AbstractMatrix, f::IndBallL0, x::Symmetric)
  p = checksquare(x)
  T = float(eltype(x))
  A = T === Float32 ? Float64 : T           # the type of the sums of squares
  s = f.s
  m = _ncounted_symmetric(f, p)
  if s >= m                                 # the whole space: the identity, NaN included
    _l0_write_identity!(y, x, p)
    return false
  end
  # the counts of the nonzero diagonal entries and pairs, a NaN in a counted entry, and the largest finite magnitude
  D, E, nan, M = 0, 0, false, zero(A)
  @inbounds for c in 1:p
    if f.diagonal
      v = x[c, c]
      D += !iszero(v)
      nan |= isnan(v)
      a = abs(A(v))
      (isfinite(a) & (a > M)) && (M = a)
    end
    for r in c + 1:p
      v = x[r, c]
      E += !iszero(v)
      nan |= isnan(v)
      a = abs(A(v))
      (isfinite(a) & (a > M)) && (M = a)
    end
  end
  if nan && 0 < s < m                       # E2: every coordinate that counts is NaN; a free diagonal is kept
    nanv = convert(eltype(y), NaN)
    @inbounds for c in 1:p, r in c:p
      _set_symmetric!(y, !f.diagonal && r == c ? x[r, c] : nanv, r, c)
    end
    return true
  end
  if D + 2E <= s                            # in the set
    _l0_write_identity!(y, x, p)
    return false
  end
  if !f.diagonal                            # the diagonal is kept; the fld(s, 2) pairs of largest magnitude
    kp = s ÷ 2
    τ, ties = zero(A), 0
    if kp > 0
      buf = _buffer!(f.scratch, A, E)
      _l0_gather_pairs!(buf, x, p, A)
      τ = _kth_largest!(view(buf, 1:E), kp)
      ties = kp - _l0_count_above(buf, E, τ)
    end
    _l0_write_selected!(y, f, x, p, A, 0, τ, 0, kp, τ, ties)
    return false
  end
  Kd, Kp = min(D, s), min(E, s ÷ 2)
  sh = M > 0 ? exponent(M) + 1 : 0          # the squares are formed from magnitudes at most 1, so they cannot overflow
  buf = _buffer!(f.scratch, A, D + E + Kd + 1)
  bd, bp, pre = view(buf, 1:D), view(buf, D + 1:D + E), view(buf, D + E + 1:D + E + Kd + 1)
  if Kd > 0
    _l0_gather_diagonal!(bd, x, p, A)
    Kd < D && _kth_largest!(bd, Kd)
    sort!(view(bd, 1:Kd); rev = true, alg = QuickSort)
  end
  if Kp > 0
    _l0_gather_pairs!(bp, x, p, A)
    Kp < E && _kth_largest!(bp, Kp)
    sort!(view(bp, 1:Kp); rev = true, alg = QuickSort)
  end
  # prefix sums of the squares of the diagonal entries, an infinite one standing for a weight and no square
  pre[1] = zero(A)
  Id = 0
  @inbounds for i in 1:Kd
    a = bd[i]
    Id += isinf(a)
    pre[i + 1] = pre[i] + _l0_square(a, sh)
  end
  Ie = 0
  @inbounds for i in 1:Kp
    Ie += isinf(bp[i])
  end
  # the candidates e = 0, ..., Kp pairs with d = min(D, s - 2e) diagonal entries, compared by the weight of the
  # infinite entries kept and then by the saving; the smaller e wins a tie
  beste, bestw, bestsav = 0, -1, zero(A)
  Q = zero(A)
  @inbounds for e in 0:Kp
    e > 0 && (Q += _l0_square(bp[e], sh))
    d = min(D, s - 2e)
    w = min(d, Id) + 2 * min(e, Ie)
    sav = pre[d + 1] + 2 * Q
    if w > bestw || (w == bestw && sav > bestsav)
      beste, bestw, bestsav = e, w, sav
    end
  end
  e = beste
  d = min(D, s - 2e)
  τd = d > 0 ? bd[d] : zero(A)
  τp = e > 0 ? bp[e] : zero(A)
  td = d - _l0_count_above(bd, d, τd)
  tp = e - _l0_count_above(bp, e, τp)
  _l0_write_selected!(y, f, x, p, A, d, τd, td, e, τp, tp)
  false
end

# the magnitude of v in the type A of the sums: the ranking inside a class uses it unscaled, so that no
# entry is lost to underflow; only the squares in the sums are formed from magnitudes scaled by 2^-sh
@inline _l0_mag(::Type{A}, v) where {A} = abs(A(v))
@inline _l0_square(a, sh::Int) = isinf(a) ? zero(a) : (b = ldexp(a, -sh); b * b)

# the magnitudes of the nonzero diagonal entries, and of the nonzero pairs, in the order of the
# lower triangle by columns
function _l0_gather_diagonal!(buf, x, p::Int, ::Type{A}) where {A}
  k = 0
  @inbounds for c in 1:p
    v = x[c, c]
    iszero(v) || (buf[k += 1] = _l0_mag(A, v))
  end
  k
end
function _l0_gather_pairs!(buf, x, p::Int, ::Type{A}) where {A}
  k = 0
  @inbounds for c in 1:p, r in c + 1:p
    v = x[r, c]
    iszero(v) || (buf[k += 1] = _l0_mag(A, v))
  end
  k
end

# the number of the first n entries of buf that are larger than τ
function _l0_count_above(buf, n::Int, τ)
  k = 0
  @inbounds for i in 1:n
    k += buf[i] > τ
  end
  k
end

# Write the selection: of the nonzero diagonal entries the d of largest magnitude, and of the nonzero pairs the
# e of largest magnitude, the entries above the threshold and the first `ties` of those at it, in the order of
# the lower triangle by columns. A diagonal entry that does not count is kept.
function _l0_write_selected!(y, f::IndBallL0, x, p::Int, ::Type{A}, d::Int, τd, td::Int, e::Int, τp, tp::Int) where {A}
  @inbounds for c in 1:p, r in c:p
    v = x[r, c]
    keep = iszero(v)
    if r == c
      if !f.diagonal
        keep = true
      elseif !keep && d > 0
        a = _l0_mag(A, v)
        if a > τd
          keep = true
        elseif a == τd && td > 0
          keep = true
          td -= 1
        end
      end
    elseif !keep && e > 0
      a = _l0_mag(A, v)
      if a > τp
        keep = true
      elseif a == τp && tp > 0
        keep = true
        tp -= 1
      end
    end
    _set_symmetric!(y, keep ? v : zero(v), r, c)
  end
  y
end

function _l0_write_identity!(y, x, p::Int)
  @inbounds for c in 1:p, r in c:p
    _set_symmetric!(y, x[r, c], r, c)
  end
  y
end

@inline _set_symmetric!(y::_OwnerMatrix, v, r, c) = _setstored!(y.x, v, tril_index(y.x.p, r, c))
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

The projection does not depend on the step `γ`, which may be any scalar in `[0, Inf]`; a negative or NaN step throws an `ArgumentError` before anything is written.

At an argument with a NaN or infinite entry and a finite radius `r > 0`, one infinite entry `±Inf` gives `±r` there and zero elsewhere (the limit of the projection along finite arguments); two or more infinite entries, whose relative growth decides the direction, and any NaN give an all-NaN result. In a `SymmetricSparseIterate` a mirrored off-diagonal pair is one entry counted twice: its limit is `±r/2` for the ℓ1 ball and `±r/√2` otherwise. For `r = 0` the result is zero for every argument, NaN included. An entry that is finite but overflows a sum, or a narrower output type, is not an infinite entry. `value_and_prox!` returns `NaN` when it stores an all-NaN result and `0` otherwise. On device arrays the ℓ1 ball and the simplex still use their device kernels, which do not yet classify these arguments.
"""
struct IndBallL1{T<:Real} <: AbstractFunction
  r::T
  scratch::Scratch

  IndBallL1{T}(r::T) where {T<:Real} = (_checknonneg("r", r); new{T}(r, Scratch()))
end
IndBallL1(r::Real = 1) = IndBallL1{typeof(r)}(r)

isproximable(::Type{<:IndBallL1}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndBallL1, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); true)
_isindicator(::Type{<:IndBallL1}) = true
isconvex(::Type{<:IndBallL1}) = true

function (f::IndBallL1)(x::AbstractArray)
  _checkreal(f, x)
  T = float(eltype(x))
  eltype(x) <: Integer && return _ball_scaled_value(f, x, T)
  n = norm(x, 1)
  slack = f.r + _tol(T) * max(one(T), f.r)
  # A non-finite norm already fails the comparison with a finite bound.
  _finite_bound(slack) && return _ind(n <= slack, T)
  _ball_scaled_value(f, x, T)
end

function prox!(y::AbstractArray, f::IndBallL1, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  _direct_prox!(y, f, x)
  y
end

# The projection; true when it stored the NaN block (see "NaN and infinite arguments" below).
function _direct_prox!(y::AbstractArray, f::IndBallL1, x::AbstractArray)
  T = float(eltype(y))
  r = convert(T, f.r)
  n = norm(x, 1)
  if n <= r
    y === x || copyto!(y, x)
    return false
  end
  if !isfinite(n)
    nan = _ball_nonfinite!(y, f, x)
    nan === nothing || return nan
  end
  if iszero(r)
    fill!(y, zero(eltype(y)))
    return false
  end
  τ = _simplex_threshold!(_buffer!(f.scratch, T, length(x)), x, eachindex(IndexLinear(), x), r, abs)
  if _lossy_threshold(τ, r, length(x))
    _project_shifted!(y, f, x)
  else
    y .= shrink.(x, τ)
  end
  false
end
####################################
#  ℓ2 ball
####################################

"""
    IndBallL2(r = 1)

The indicator of the Euclidean ball `{x : ‖x‖₂ ≤ r}` (the Frobenius ball for
matrices). Its proximal operator scales `x` onto the ball when it is outside.

The projection does not depend on the step `γ`, which may be any scalar in `[0, Inf]`; a negative or NaN step throws an `ArgumentError` before anything is written.

At an argument with a NaN or infinite entry and a finite radius `r > 0`, one infinite entry `±Inf` gives `±r` there and zero elsewhere (the limit of the projection along finite arguments); two or more infinite entries, whose relative growth decides the direction, and any NaN give an all-NaN result. In a `SymmetricSparseIterate` a mirrored off-diagonal pair is one entry counted twice: its limit is `±r/2` for the ℓ1 ball and `±r/√2` otherwise. For `r = 0` the result is zero for every argument, NaN included. An entry that is finite but overflows a sum, or a narrower output type, is not an infinite entry. `value_and_prox!` returns `NaN` when it stores an all-NaN result and `0` otherwise.
"""
struct IndBallL2{T<:Real} <: AbstractFunction
  r::T

  IndBallL2{T}(r::T) where {T<:Real} = (_checknonneg("r", r); new{T}(r))
end
IndBallL2(r::Real = 1) = IndBallL2{typeof(r)}(r)

isproximable(::Type{<:IndBallL2}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndBallL2, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); true)
_isindicator(::Type{<:IndBallL2}) = true
isconvex(::Type{<:IndBallL2}) = true

function (f::IndBallL2)(x::AbstractArray)
  _checkreal(f, x)
  T = float(eltype(x))
  eltype(x) <: Integer && return _ball_scaled_value(f, x, T)
  n = _norm2(x)
  slack = f.r + _tol(T) * max(one(T), f.r)
  # A non-finite norm already fails the comparison with a finite bound.
  _finite_bound(slack) && return _ind(n <= slack, T)
  _ball_scaled_value(f, x, T)
end

# A sorted spectrum gives its scale without another reduction.
_spectrum_scale(v::AbstractVector{T}) where {T} = isempty(v) ? zero(T) : max(abs(v[1]), abs(v[end]))
_decided_on_spectrum(::AbstractFunction) = false
_decided_on_spectrum(::Union{IndBallL1, IndBallL2, IndBox, IndNonnegative, IndBinary, NormL1Nonneg}) = true
# At an overflowing spectrum (k > 0), a function with a recovery takes the value in the units of
# the spectrum (see _spectrum_scaled); k < 0 is a tiny spectrum of a function that counts nonzeros.
function _spectrum_value(f::AbstractFunction, v, k::Int)
  (k == 0 || _scale_invariant(f)) && return f(v)
  if k > 0
    r = _spectrum_scaled(f, v, k, _unit_scale(eltype(v)))
    r === nothing || return r
  end
  f(ldexp.(v, k))
end

# Scale bounds before narrowing, keeping the absolute floor in original units.
function _spectrum_ball(v::AbstractVector{T}, bound, k::Int, p) where {T}
  isfinite(bound) || return _ind(isinf(bound) && bound > 0, T)
  S = float(promote_type(T, typeof(bound)))
  _spectrum_ball_scaled(v, S(bound), 0, k, p)
end
# The bound mb 2^eb, finite, which is not formed when it is outside the range.
function _spectrum_ball_scaled(v::AbstractVector{T}, mb::S, eb::Int, k::Int, p) where {T, S}
  α, r = ldexp(one(S), -k), ldexp(mb, eb - k)
  # A bound beyond the range of the scaled units contains the (finite) spectrum.
  _finite_value(r) || return zero(T)
  m = S(_spectrum_scale(v))
  n = p == 1 ? sum(abs, v) : p == 2 ? _norm2(v) : m
  slack = r + _tol(T) * max(α, abs(r), m)
  _finite_value(n) && _finite_bound(slack) && return _ind(n <= slack, T)
  # A norm or the bound plus slack may overflow even at a finite spectrum.
  e = _comparison_exponent(_magnitude_exponent(m), _magnitude_exponent(abs(r)))
  factors = _power_factors(S, -e)
  n = p == 1 ? _scaled_sum(v, factors, abs) :
    p == 2 ? sqrt(_scaled_sumabs2(v, factors)) : ldexp(m, -e)
  r = ldexp(mb, eb - k - e)
  slack = _tol(T) * max(ldexp(one(S), -k - e), abs(r), ldexp(m, -e))
  _ind(n <= r + slack, T)
end
_spectrum_value(f::IndBallL1, v, k::Int) = _spectrum_ball(v, f.r, k, 1)
_spectrum_value(f::IndBallL2, v, k::Int) = _spectrum_ball(v, f.r, k, 2)
function _spectrum_value(f::Union{IndBox, IndBinary}, v::AbstractVector{T}, k::Int) where {T}
  _checkparams(f, v)
  S = float(promote_type(T, eltype(f.l), eltype(f.u)))
  floor, t = max(ldexp(one(S), -k), S(_spectrum_scale(v))), S(_tol(T))
  @inbounds for i in eachindex(v)
    l, u = ldexp(S(_entry(f.l, i)), -k), ldexp(S(_entry(f.u, i)), -k)
    ok = f isa IndBox ? _inbox(v[i], l, u, t, floor) :
      abs(v[i] - l) <= t * max(floor, abs(l)) || abs(v[i] - u) <= t * max(floor, abs(u))
    ok || return T(Inf)
  end
  zero(T)
end
_spectrum_value(::IndNonnegative, v::AbstractVector{T}, k::Int) where {T} =
  _ind(isempty(v) || minimum(v) >= -_tol(T) * max(ldexp(one(T), -k), _spectrum_scale(v)), T)
function _spectrum_value(f::NormL1Nonneg, v::AbstractVector{T}, k::Int) where {T}
  _checkparams(f, v)
  # Preserve the ordinary reduction when its domain test already accepts.
  k == 0 && all(x -> x >= -_tol(T), v) && return f(v)
  # An overflowing spectrum: the domain, then the value in its units, when it is recovered.
  if k > 0
    r = _spectrum_scaled(f, v, k, _unit_scale(T))
    r === nothing || return convert(T, r)
  end
  iszero(_spectrum_value(IndNonnegative(), v, k)) || return T(Inf)
  # Only the domain decision changes; negative rounding contributes zero.
  s = zero(T)
  @inbounds @simd for i in eachindex(v)
    s += convert(T, value_entry(f, ldexp(max(v[i], zero(T)), k), i))
  end
  s
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

function prox!(y::AbstractArray, f::IndBallL2, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  _direct_prox!(y, f, x)
  y
end

function _direct_prox!(y::AbstractArray, f::IndBallL2, x::AbstractArray)
  T = float(eltype(y))
  nx = convert(T, _norm2(x))
  r = convert(T, f.r)
  if nx <= r
    y === x || copyto!(y, x)
  else
    if !isfinite(nx)
      nan = _ball_nonfinite!(y, f, x)
      nan === nothing || return nan
    end
    _scale_l2!(y, x, r, nx)
  end
  false
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

The projection does not depend on the step `γ`, which may be any scalar in `[0, Inf]`; a negative or NaN step throws an `ArgumentError` before anything is written.

At an argument with a NaN or infinite entry and a finite radius `r > 0`, one infinite entry `±Inf` gives `±r` there and zero elsewhere (the limit of the projection along finite arguments); two or more infinite entries, whose relative growth decides the direction, and any NaN give an all-NaN result. In a `SymmetricSparseIterate` a mirrored off-diagonal pair is one entry counted twice: its limit is `±r/2` for the ℓ1 ball and `±r/√2` otherwise. An entry that is finite but overflows a sum, or a narrower output type, is not an infinite entry. `value_and_prox!` returns `NaN` when it stores an all-NaN result and `0` otherwise.
"""
struct IndSphereL2{T<:Real} <: AbstractFunction
  r::T

  IndSphereL2{T}(r::T) where {T<:Real} = (r > 0 || throw(ArgumentError("r must be positive, got $r")); new{T}(r))
end
IndSphereL2(r::Real = 1) = IndSphereL2{typeof(r)}(r)

isproximable(::Type{<:IndSphereL2}) = true
_isindicator(::Type{<:IndSphereL2}) = true

function (f::IndSphereL2)(x::AbstractArray)
  _checkreal(f, x)
  T = float(eltype(x))
  r = convert(T, f.r)
  abs(convert(T, _norm2(x)) - r) <= _tol(T) * max(one(T), r) ? zero(T) : T(Inf)
end

function prox!(y::AbstractArray, f::IndSphereL2, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  x = _symmetric_argument(y, x)
  _direct_prox!(y, f, x)
  y
end

function _direct_prox!(y::AbstractArray, f::IndSphereL2, x::AbstractArray)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  isempty(x) && throw(ArgumentError("the sphere has no point of size $(size(x))"))
  T = float(eltype(y))
  nx = convert(T, _norm2(x))
  r = convert(T, f.r)
  if !isfinite(nx)
    nan = _ball_nonfinite!(y, f, x)
    nan === nothing || return nan
  end
  # a norm that underflows in a narrower destination is not the zero point
  if iszero(nx) && all(iszero, x)
    fill!(y, zero(eltype(y)))
    y[firstindex(y)] = r
  else
    _scale_l2!(y, x, r, nx)
  end
  false
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

The projection does not depend on the step `γ`, which may be any scalar in `[0, Inf]`; a negative or NaN step throws an `ArgumentError` before anything is written.

At an argument with a NaN or infinite entry and a finite radius `r > 0`: one `+Inf` entry takes `r` and every other entry is zero, whatever the `-Inf` entries (in a `SymmetricSparseIterate`, `r/2` in each mirrored entry of an off-diagonal one); two or more `+Inf` entries and any NaN give an all-NaN result. A `-Inf` entry is zero and the finite entries are projected as usual; when every entry is `-Inf`, the inequality simplex is zero and the equality simplex is all NaN. The equality simplex of one entry is the point `r` for every argument, NaN included, and `r = 0` gives zero. An entry that is finite but overflows a sum, or a narrower output type, is not an infinite entry. `value_and_prox!` returns `NaN` when it stores an all-NaN result and `0` otherwise. On device arrays the simplex still uses its device kernel, which does not yet classify these arguments.
"""
struct IndSimplex{T<:Real} <: AbstractFunction
  r::T
  equality::Bool
  scratch::Scratch

  IndSimplex{T}(r::T, equality::Bool) where {T<:Real} = (_checknonneg("r", r); new{T}(r, equality, Scratch()))
end
IndSimplex(r::Real = 1; equality::Bool = true) = IndSimplex{typeof(r)}(r, equality)

isproximable(::Type{<:IndSimplex}) = true
_checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::IndSimplex, y::AbstractArray, x::AbstractArray) =
  (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); true)
_isindicator(::Type{<:IndSimplex}) = true
isconvex(::Type{<:IndSimplex}) = true

function (f::IndSimplex)(x::AbstractArray)
  _checkreal(f, x)
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

_decided_on_spectrum(::IndSimplex) = true
function _spectrum_value(f::IndSimplex, v::AbstractVector{T}, k::Int) where {T}
  iszero(_spectrum_value(IndNonnegative(), v, k)) || return T(Inf)
  _spectrum_sum(v, f.r, k, f.equality)
end

# Sums retain their own relative tolerance, including cancellation.
function _spectrum_sum(v::AbstractVector{T}, bound, k::Int, equality::Bool) where {T}
  isfinite(bound) || return _ind(!equality && isinf(bound), T)
  S = float(promote_type(T, typeof(bound)))
  r, α = ldexp(S(bound), -k), ldexp(one(S), -k)
  s = sum(v)
  slack = _tol(T) * max(α, abs(r), abs(s))
  if !_finite_bound(slack)
    e = _comparison_exponent(_magnitude_exponent(S(_spectrum_scale(v))), _magnitude_exponent(abs(r)))
    s = _scaled_sum(v, _power_factors(S, -e))
    r, α = ldexp(S(bound), -k - e), ldexp(one(S), -k - e)
    slack = _tol(T) * max(α, abs(r), abs(s))
  end
  _ind(equality ? abs(s - r) <= slack : s <= r + slack, T)
end

function prox!(y::AbstractArray, f::IndSimplex, x::AbstractArray, γ::Real)
  _checkstep_nonneg(γ)
  x = _symmetric_argument(y, x)
  _checkwrite(prox!, f, y, x)
  _direct_prox!(y, f, x)
  y
end

function _direct_prox!(y::AbstractArray, f::IndSimplex, x::AbstractArray)
  T = float(eltype(y))
  r = convert(T, f.r)
  if !f.equality
    s = zero(T)
    for v in x
      s += max(v, zero(v))
    end
    if s <= r
      y .= max.(x, zero(T))
      return false
    end
    if !isfinite(s)
      nan = _simplex_nonfinite!(y, f, x)
      nan === nothing || return nan
    end
  end
  if iszero(r)
    fill!(y, zero(eltype(y)))
    return false
  end
  buf, idx = _buffer!(f.scratch, T, length(x)), eachindex(IndexLinear(), x)
  # Condat's pass discards a NaN, leaving the level finite while every output depends on the
  # entry (E2); the strict pass gives a NaN level instead, which is classified below. The
  # inequality simplex reaches its NaN through the positive-part sum above.
  τ = f.equality ? _simplex_threshold_strict!(buf, x, idx, r) : _simplex_threshold!(buf, x, idx, r)
  if _lossy_threshold(τ, r, length(x))
    nan = _simplex_nonfinite!(y, f, x)
    nan === nothing || return nan
    _project_shifted!(y, f, x)
  else
    y .= max.(x .- τ, zero(T))
  end
  false
end

####################################
#  NaN and infinite arguments of the direct projections (#469, decision 48)
####################################

# A direct projection returns the proximal limit at an infinite argument when the limit is
# determined and the NaN block when it is not; see numerics.md E2, E3, O2 and M2. Nothing here
# runs on a finite argument unless a reduction that the projection computes anyway (a norm, a
# positive sum, the threshold) is not finite, and the first step then is to read the original
# argument: a reduction that overflowed on finite data, or an entry that only overflows in a
# narrower destination, is not an infinite argument and goes on to the finite computation.

const _DirectSet = Union{IndBallL1, IndBallL2, IndSphereL2, IndSimplex}

# O2: the value is the one at the stored point, NaN when a NaN block was stored. These and
# IndBallL0 (above) are the indicators whose generic shortcut (zero without looking) is replaced.
function _value_and_prox_body!(y, f::_DirectSet, x, γ::Real = 1)
  _checkstep_nonneg(γ)
  _checkwrite(value_and_prox!, f, y, x)
  _direct_prox!(y, f, x) ? convert(_valtype(y), NaN) : zero(_valtype(y))
end

_nan_block!(y) = (fill!(y, convert(eltype(y), NaN)); true)

# The independent coordinates of the output and what they hold: a mirrored off-diagonal pair
# of a SymmetricSparseIterate is one coordinate, counted twice in the norm (M2); every entry
# of any other output is one. `at` is the position (a linear index, or the row) of the first
# coordinate that satisfies `target` (any infinity for a ball, +Inf for the simplex) and `col`
# its column in the symmetric case (0 otherwise).
struct _Classes
  nan::Int
  pos::Int
  neg::Int
  n::Int
  at::Int
  col::Int
end

function _classify(y::AbstractArray, x::AbstractArray, target)
  nan, pos, neg = count(isnan, x), count(==(Inf), x), count(==(-Inf), x)
  i = pos + neg == 0 ? nothing : findfirst(target, x)
  _Classes(nan, pos, neg, length(x), i === nothing ? 0 : LinearIndices(x)[i], 0)
end
function _classify(y::_OwnerMatrix, x::AbstractArray, target)
  p = size(x, 1)
  nan = pos = neg = at = col = 0
  for c in 1:p, r in c:p
    v = x[r, c]
    if isnan(v)
      nan += 1
    elseif isinf(v)
      v > 0 ? (pos += 1) : (neg += 1)
      at == 0 && target(v) && ((at, col) = (r, c))
    end
  end
  _Classes(nan, pos, neg, p * (p + 1) ÷ 2, at, col)
end

# the weight of an off-diagonal coordinate of a symmetric output in the limit: the pair
# satisfies 2|a| = r (ℓ1, simplex) or 2a² = r² (ℓ2)
_pairweight(::Union{IndBallL1, IndSimplex}) = 1 / 2
_pairweight(::Union{IndBallL2, IndSphereL2}) = 1 / sqrt(2)

# store the vector with `v` at the classified coordinate and zero elsewhere
function _vertex!(y::AbstractArray, f, c::_Classes, v)
  vv = convert(eltype(y), v)
  y .= ifelse.(LinearIndices(y) .== c.at, vv, zero(vv))
  false
end
function _vertex!(y::_OwnerMatrix, f, c::_Classes, v)
  fill!(y, zero(eltype(y)))
  _set_symmetric!(y, convert(eltype(y), c.at == c.col ? v : v * _pairweight(f)), c.at, c.col)
  false
end

# Ball and sphere with a positive radius: one infinite coordinate selects ±r e_k, several have
# no limit that does not depend on their relative growth, and a NaN reaches every output.
# Zero radius is the singleton {0}. Returns nothing when x has no NaN or infinite entry.
function _ball_nonfinite!(y, f, x)
  isfinite(f.r) || return nothing
  c = _classify(y, x, isinf)
  c.nan + c.pos + c.neg == 0 && return nothing
  iszero(f.r) && (fill!(y, zero(eltype(y))); return false)
  (c.nan > 0 || c.pos + c.neg > 1) && return _nan_block!(y)
  _vertex!(y, f, c, c.pos == 1 ? float(f.r) : -float(f.r))
end

# Simplex: NaN reaches every output except where the output does not depend on the input (a
# zero radius, the equality simplex of one coordinate). Two +Inf have no limit; one selects
# r e_k, whatever the -Inf entries. With no +Inf, a -Inf entry takes 0 and the finite entries
# the reduced simplex (nothing here: the finite computation does it); when every entry is
# -Inf the inequality simplex is 0 and the equality simplex has no limit.
function _simplex_nonfinite!(y, f::IndSimplex, x)
  isfinite(f.r) || return nothing
  c = _classify(y, x, ==(Inf))
  c.nan + c.pos + c.neg == 0 && return nothing
  iszero(f.r) && (fill!(y, zero(eltype(y))); return false)
  f.equality && c.n == 1 && (fill!(y, convert(eltype(y), f.r)); return false)
  (c.nan > 0 || c.pos > 1) && return _nan_block!(y)
  c.pos == 1 && return _vertex!(y, f, c, float(f.r))
  c.neg == c.n || return nothing
  f.equality ? _nan_block!(y) : (fill!(y, zero(eltype(y))); false)
end

# The level τ carries the radius r in x - τ. When one unit of rounding of τ is already larger
# than r (or τ is not a number: an overflowed sum, or an entry outside the destination's range),
# x - τ cannot resolve r and the result is wrong. Ordinary arguments never come near this, so
# the finite computation is unchanged for them; only the extremes take the shifted recovery.
@inline _lossy_threshold(τ, r, n) = !isfinite(τ) || abs(τ) * eps(typeof(τ)) > r

# The projection onto {z ≥ 0 : Σ z = r} does not change when every entry moves by the same
# amount. Shifting by the largest entry, in the wider of the argument's and the destination's
# types, gives z ≤ 0 with a level in [-r, 0): no sum overflows, and x - τ is evaluated as
# (x - max) - level, which keeps the radius next to entries of any size. For the ℓ1 ball the
# magnitudes are projected and the signs restored; entries that are -Inf go to zero.
@noinline function _project_shifted!(y, f::Union{IndBallL1, IndSimplex}, x)
  S = float(promote_type(eltype(x), eltype(y)))
  u = f isa IndBallL1 ? abs.(x) : x
  M = maximum(v -> convert(S, v), u)
  z = vec(Array(convert.(S, u) .- M))
  τ = _simplex_threshold!(_buffer!(f.scratch, S, length(z)), z, eachindex(IndexLinear(), z), convert(S, f.r))
  if f isa IndBallL1
    y .= sign.(x) .* max.((abs.(convert.(S, x)) .- M) .- τ, zero(S))
  else
    y .= max.((convert.(S, x) .- M) .- τ, zero(S))
  end
  y
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
reliable for all finite inputs, within its floating-point tolerance.

Every finite argument whose projection is representable in the output type is
projected, also when the dot product `⟨a, x⟩`, the residual `⟨a, x⟩ - b` or the
correction of the formula would overflow in the working precision: the
projection is then formed in exact powers of two. An entry of the projection
beyond the largest finite value is `±Inf`; it is never clamped. An argument with
a NaN or infinite entry follows the formula in IEEE arithmetic. An array of
positive finite steps `γ` gives the projection in the metric of
`Diagonal(1 ./ γ)`, with the same guarantee; steps whose weighted squared norm
`Σᵢ γᵢaᵢ²`, or whose products with the normal, underflow are outside it, and so
are arguments of `Float16` entries with more than about `2^11` entries.
Accuracy is limited by the less precise of the function's floating-point type
and the output type.
"""
struct IndHyperplane{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction
  a::A
  b::T
  s::NTuple{2, T} # the scale, applied as two successive powers of two
  q::T           # ‖s·a‖², or (s·a)² for a scalar a

  function IndHyperplane{A, T}(a::A, b::T) where {A<:Union{Real, AbstractArray}, T<:Real}
    eltype(a) <: Real || _real_error(:IndHyperplane)
    s, q = _normal_cache(a, b)
    new{A, T}(a, b, s, q)
  end
end
IndHyperplane(a::Union{Real, AbstractArray}, b::Real) =
  (eltype(a) <: Real || _real_error(:IndHyperplane); T = float(promote_type(eltype(a), typeof(b))); IndHyperplane{typeof(a), T}(a, convert(T, b)))

"""
    IndHalfspace(a, b)

The indicator of the halfspace `{x : ⟨a, x⟩ ≤ b}`, `a ≠ 0`. Its proximal operator
is the projection, `x - (max(⟨a, x⟩ - b, 0) / ‖a‖²) a`. A scalar `a` stands for
an array of `a`s of the size of `x`, as in [`IndHyperplane`](@ref). As there,
construct a new function after changing an array `a` in place.

Every finite nonzero normal is accepted, including subnormal and integer
entries; `b` must be finite. As for [`IndHyperplane`](@ref), powers of two
rescale the normal and `b` when needed. The membership value is reliable for all
finite inputs, within its floating-point tolerance. As for
[`IndHyperplane`](@ref), every finite argument whose projection is representable
in the output type is projected, also when the dot product, the residual or the
correction of the formula would overflow; an entry of the projection beyond the
largest finite value is `±Inf`, and an argument with a NaN or infinite entry
follows the formula in IEEE arithmetic. Accuracy is limited by the less precise
of the function's floating-point type and the output type.
"""
struct IndHalfspace{A<:Union{Real, AbstractArray}, T<:Real} <: AbstractFunction
  a::A
  b::T
  s::NTuple{2, T}
  q::T

  function IndHalfspace{A, T}(a::A, b::T) where {A<:Union{Real, AbstractArray}, T<:Real}
    eltype(a) <: Real || _real_error(:IndHalfspace)
    s, q = _normal_cache(a, b)
    new{A, T}(a, b, s, q)
  end
end
IndHalfspace(a::Union{Real, AbstractArray}, b::Real) =
  (eltype(a) <: Real || _real_error(:IndHalfspace); T = float(promote_type(eltype(a), typeof(b))); IndHalfspace{typeof(a), T}(a, convert(T, b)))

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
for all finite inputs. Nonfinite inputs follow the projection formula
using IEEE arithmetic; the whole space copies its input. An empty input has
a projection only when `l ≤ 0 ≤ u`.

The projection is accurate when the scaled active bound, residual and output
are representable; inputs near `floatmax` are outside this guarantee. Accuracy
is limited by the less precise of the function's floating-point type and the
output type. An array of positive finite steps `γ` gives the projection in the
metric of `Diagonal(1 ./ γ)`, with the correction along `γ .* a`. This uses the
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
    eltype(a) <: Real || _real_error(:IndHyperslab)
    l <= u && l < Inf && u > -Inf || throw(ArgumentError("bounds must satisfy l ≤ u, l < Inf and u > -Inf"))
    s, q = _normal_cache(a, zero(T))
    new{A, T}(a, l, u, s, q)
  end
end
IndHyperslab(a::Union{Real, AbstractArray}, l::Real, u::Real) =
  (eltype(a) <: Real || _real_error(:IndHyperslab); T = float(promote_type(eltype(a), typeof(l), typeof(u))); IndHyperslab{typeof(a), T}(a, convert(T, l), convert(T, u)))

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

# the axes of an array, or of the matrix that an owner represents
_matrix_axes(x) = axes(x)
_matrix_axes(x::SymmetricSparseIterate) = _owner_axes(x)
function _checkparams(f::IndHyperslab, x)
  _checkparam(f.a, x)
  f.a isa AbstractArray && Base.Broadcast.check_broadcast_axes(_matrix_axes(x), f.a)
  true
end

# ⟨a, x⟩ and ‖a‖², for an array a or a scalar one
_dot_normal(a::Real, x::AbstractArray) = a * sum(x)
_dot_normal(a::AbstractArray, x::AbstractArray) = dot(a, x)
_normal_sqnorm(f::Union{IndHyperplane, IndHalfspace, IndHyperslab}, x::_ArrayOrOwner) = f.a isa Real ? f.q * _nentries(x) : f.q
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

(f::IndHyperplane)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::IndHyperplane, x::_ArrayOrOwner) = _normal_value(f, x, Val(false))
function _normal_value(f::IndHyperplane, x::_ArrayOrOwner, checked)
  T = float(eltype(x))
  eltype(x) <: Integer && return _linear_scaled_value(f, x, T, f.b, false, checked)
  _isscaled(f) && return _scaled_normal_value(f, x, T, f.b, false, checked)
  slack = _tol(T) * max(one(T), abs(f.b), sqrt(_normal_sqnorm(f, x) * _sumabs2(x)))
  g = _value_dot_normal(f.a, x, checked) - f.b
  _finite_bound(slack) && _ordinary_residual_finite(f, x, g, typeof(slack)) ||
    return _linear_scaled_value(f, x, T, f.b, false, checked)
  abs(g) <= slack ? zero(T) : T(Inf)
end
(f::IndHalfspace)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::IndHalfspace, x::_ArrayOrOwner) = _normal_value(f, x, Val(false))
function _normal_value(f::IndHalfspace, x::_ArrayOrOwner, checked)
  _halfspace_value(f, x, f.b, false, checked)
end

function _halfspace_value(f, x, b, lower::Bool = false, checked = Val(false))
  T = float(eltype(x))
  eltype(x) <: Integer && return _linear_scaled_value(f, x, T, b, lower, checked)
  _isscaled(f) && return _scaled_normal_value(f, x, T, b, lower, checked)
  slack = _tol(T) * max(one(T), abs(b), sqrt(_normal_sqnorm(f, x) * _sumabs2(x)))
  g = _value_dot_normal(f.a, x, checked) - b
  _finite_bound(slack) && _ordinary_residual_finite(f, x, g, typeof(slack)) ||
    return _linear_scaled_value(f, x, T, b, lower, checked)
  _ind(lower ? -g <= slack : g <= slack, T)
end

(f::IndHyperslab)(x::AbstractArray) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::IndHyperslab, x::_ArrayOrOwner) = _normal_value(f, x, Val(false))
function _normal_value(f::IndHyperslab, x::_ArrayOrOwner, checked)
  _checkparams(f, x)
  T = float(eltype(x))
  isfinite(f.l) && !iszero(_halfspace_value(f, x, f.l, true, checked)) && return T(Inf)
  isfinite(f.u) && return _halfspace_value(f, x, f.u, false, checked)
  _ind(isfinite(_maximum_abs(x)), T)
end

# At matching precisions, the finite squared-norm bound precludes residual overflow.
@inline function _ordinary_residual_finite(f, x, g, ::Type{S}) where {S}
  P = float(promote_type(eltype(f.a), eltype(x)))
  (P === S && S <: Union{Float32, Float64}) || _finite_value(g)
end

@noinline function _scaled_normal_value(f, x, ::Type{T}, b = f.b, lower::Bool = false, checked = Val(false)) where {T}
  g = _value_dot_normal(_normalized_normal(f.a, f.s), x, checked) - _scaled_offset(f, b)
  slack = _scaled_normal_slack(f, x, T, b)
  _finite_value(g) && _finite_bound(slack) &&
    return _ind(f isa IndHyperplane ? abs(g) <= slack : (lower ? -g <= slack : g <= slack), T)
  _linear_scaled_value(f, x, T, b, lower, checked)
end

# Include the normal's scale in the offset exponent before forming either one.
@noinline function _linear_scaled_value(f, x, ::Type{T}, b = f.b, lower::Bool = false, checked = Val(false)) where {T}
  m = _maximum_abs(x)
  isfinite(m) || return T(Inf)
  S = promote_type(T, typeof(f.q))
  e = exponent(f.s[1]) + exponent(f.s[2])
  k = _comparison_exponent(_magnitude_exponent(m), e, _magnitude_exponent(abs(b)) + e)
  s = _power_factors(S, -k)
  b = ldexp(S(b), e - k)
  a = _normalized_normal(f.a, f.s)
  g = _value_scaled_dot(a, x, s, checked) - b
  nx = sqrt(_scaled_sumabs2(x, s))
  q = f.a isa Real ? sqrt(S(f.q)) * sqrt(S(_nentries(x))) : sqrt(S(f.q))
  tol = _isscaled(f) ? max(_tol(T), _tol(typeof(f.q))) : _tol(T)
  slack = tol * max(ldexp(one(S), e - k), abs(b), q * nx)
  _ind(f isa IndHyperplane ? abs(g) <= slack : (lower ? -g <= slack : g <= slack), T)
end

function _checkwrite(op::O, f::Union{IndHyperplane, IndHalfspace, IndHyperslab}, y::AbstractArray, x::AbstractArray) where {O<:Union{typeof(prox!), typeof(value_and_prox!)}}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  op === value_and_prox! && _checkparams(f, x)
  if f.a isa AbstractArray
    length(f.a) == length(x) || _dot_normal(_scaled_normal(f), x)
    Base.Broadcast.check_broadcast_axes(axes(y), x, f.a)
  end
  f isa IndHyperslab && isempty(x) && !(f.l <= 0 <= f.u) &&
    throw(ArgumentError("the hyperslab contains no empty array"))
  true
end

# Choose the active bound from the residual, independently of membership slack.
# Branch before passing the normal: the scaled and unscaled normals have
# different types, and a Union of them allocates on Julia 1.10 and under coverage.
_hyperslab_residual(f, x) = _isscaled(f) ?
  _hyperslab_residual(f, x, _normalized_normal(f.a, f.s)) : _hyperslab_residual(f, x, f.a)
function _hyperslab_residual(f, x, a)
  # Floating-point products avoid wraparound for integer inputs and normals.
  s = eltype(x) <: Integer ? _stored_dot_normal(a, x) : _dot_normal(a, x)
  if isfinite(f.l)
    g = s - _scaled_offset(f, f.l)
    (isnan(g) || g < 0) && return g
  end
  if isfinite(f.u)
    g = s - _scaled_offset(f, f.u)
    (isnan(g) || g > 0) && return g
  end
  zero(s)
end

prox!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndHyperslab, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  isempty(x) && return y
  g = _hyperslab_residual(f, x)
  iszero(g) && return (y === x ? y : copyto!(y, x))
  t = g / _normal_sqnorm(f, x)
  _isscaled(f) ? _hyperslab_output!(y, x, _normalized_normal(f.a, f.s), t) : _hyperslab_output!(y, x, f.a, t)
end
_hyperslab_output!(y, x, a, t) = (y .= x .- t .* _checked_parameter(y, a))

# The coefficient of the correction, g / d with g = ⟨a, x⟩ - b and d = ‖a‖² or its metric
# version, formed as in IndHyperslab. Unlike IndHyperslab, it is then widened to a wider
# output type, which does not recompute it in that type (#362). It is never rounded below
# the type of g / d or the output's.
@inline function _model_coefficient(y, g, d)
  t = g / d
  convert(promote_type(float(eltype(y)), typeof(t)), t)
end

prox!(y::AbstractArray, f::IndHyperplane, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndHyperplane, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _linear_set_prox!(y, f, x)
end
prox!(y::AbstractArray, f::IndHalfspace, x::AbstractArray, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::IndHalfspace, x::AbstractArray, ::Real)
  _checkwrite(prox!, f, y, x)
  _linear_set_prox!(y, f, x)
end

# Branch before passing the normal: the scaled and unscaled normals have different types.
@inline _linear_set_prox!(y, f, x) = _isscaled(f) ?
  _linear_set_prox!(y, f, x, _normalized_normal(f.a, f.s), _scaled_offset(f)) :
  _linear_set_prox!(y, f, x, f.a, f.b)
# The working normal a and offset b: those of f, or the normalized ones.
@inline function _linear_set_prox!(y, f, x, a, b)
  g = _dot_normal(a, x) - b                  # the residual; a halfspace clamps it only below
  d = _normal_sqnorm(f, x)
  t = _model_coefficient(y, f isa IndHalfspace ? max(g, zero(g)) : g, d)
  if !_scalar_step_safe(f, a, g, t) && _extreme_projection!(y, f, x, a, d, nothing, typeof(t))
    return y
  end
  y .= x .- t .* _checked_parameter(y, a)
end

####################################
#  projections at overflowing intermediates
####################################

# The formula x - t a, with t = g / d and g = ⟨a, x⟩ - b, overflows in an intermediate result
# for finite arguments whose projection is representable (#563). The ordinary arithmetic stays as
# it was; a scalar test of the values it already has decides whether any of its intermediate
# results can have overflowed, and only then is the projection recomputed in powers of two
# (_extreme_projection!). Write S for the type of t and B = floatmax(S) / 2. The comparisons below
# are written `v <= B` or `isfinite(v)`, so that a NaN fails them and fires the recomputation.
#
# Two facts bound the arithmetic. (F1) An overflow is absorbing: once a term, a partial sum or
# the subtraction of b overflows, g is ±Inf or NaN (Inf - Inf), whatever the order of the sum, the
# lanes of the dot product or a fused multiply-add. So a finite g means that none of them
# overflowed. For a halfspace it is tested before max(g, 0), which would turn -Inf into 0; for a
# hyperplane the test of t covers it, as g / d is then infinite or NaN. (F2) If
# |t| R ≤ B for an upper bound R of the magnitudes of the factors that t multiplies, then every
# such product is at most B (1 + 2^-p)^2 < floatmax(S): the factor 2 absorbs the roundings in
# t, R and the products.
#
# Scalar step. The order of evaluation is the dot product, g, the clamp, t = g / d, the products
# t αᵢ and the differences xᵢ - t αᵢ, with α the working normal (scaled when f is). d is
# f.q = Σ αᵢ² or f.q n for a scalar normal; with e = emax / 4, _normal_cache keeps max |αᵢ| in
# [2^-e, 2^e], or in [1, 2) when scaled, so d is normal and finite and t = g / d overflows only
# through g. By (F1) a finite g covers the dot product and the offset. |t| R ≤ B covers t (an
# infinite or NaN t fails it) and, by (F2), every t αᵢ, with R = |α| for a scalar normal and
# R = √q ≥ max |αᵢ| (1 - 2^-p) for an array normal, since q ≥ αᵢ² up to the rounding of the sum.
# The difference xᵢ - t αᵢ of two finite numbers overflows only when the exact projection does not
# fit.
# Where it fires on an argument whose formula was finite: g or a partial sum within floatmax of
# overflow, or |t| R > floatmax / 2 (an array normal: R up to √n times max |αᵢ|).
@inline _half_floatmax(::Type{S}) where {S<:AbstractFloat} = floatmax(S) / 2
@inline _normal_bound(f, a::Real, ::Type{S}) where {S} = abs(convert(S, float(a)))
@inline _normal_bound(f, a::AbstractArray, ::Type{S}) where {S} = sqrt(convert(S, f.q))
# A hyperplane's g is NaN or infinite exactly when its t is, which fails the second test.
@inline _raw_finite(::IndHyperplane, g) = true
@inline _raw_finite(::IndHalfspace, g) = _finite_value(g)
@inline function _scalar_step_safe(f, a, g, t::S) where {S}
  _raw_finite(f, g) && abs(t) * _normal_bound(f, a, S) <= _half_floatmax(S)
end

# The projection x - (g / d) c with c = a, or γ .* a in a diagonal metric, formed in powers of
# two, for an argument whose entries are all finite. The write is the projection's only effect;
# false means that nothing was written and the formula's own result stands: a NaN or infinite
# entry in x (IEEE propagation, as before), or a d below floatmin, where the coefficient
# underflows (#631, kept as it was).
#
# k is chosen from the exponents of the largest entry and of the offset. With α = a 2^e the
# working normal (e = 0 unless it is scaled), G = ⟨α, x 2^-k⟩ - b 2^(e-k) is ĝ 2^-k for the
# residual ĝ. Each x 2^-k is below 2 in magnitude and |αᵢ| ≤ 2^(emax / 4) (below 2 when scaled),
# and |b 2^(e-k)| < 2, so |G| < n 2^(emax / 4 + 2) + 2, which is below floatmax for every n of a
# Float32 or Float64 array and, for Float16 (emax / 4 = 3), for n < 2^11; the offset is formed with
# the exponent e - k directly, never as b 2^e. A halfspace is active exactly when G > 0.
# d = D 2^w with D in [1, 2) where d is the scalar step's, or a scalar normal's with finite d. For
# steps γ and an array normal, D is formed here in S, from the exponent w of the largest term
# γᵢαᵢ² (αᵢ widened to S before it is squared) and the sum of the terms with 2^-w applied to the
# significands, so that D is in [1, 8n) whatever d overflowed or lost to the narrower type of its
# own terms. For a scalar normal with d = Inf, D is q times the sum of γᵢ 2^-w with 2^w the
# exponent of the largest step, and D ≥ 1/2 as d ≥ floatmax. So D ≥ 1/2, and t″ = G / D is finite
# with |G| as above. Then t = t″ 2^J with J = k - w, and the correction cᵢ = t″ γᵢ αᵢ 2^J is
# formed from the significands of the three factors with the power of two last, so that no factor
# or partial product over- or underflows; xᵢ itself is never scaled and an αᵢ = 0 leaves it
# unchanged. Where cᵢ itself exceeds floatmax, a representable yᵢ = xᵢ - cᵢ needs |cᵢ| ≤
# 2 floatmax, and yᵢ = 2 (xᵢ / 2 - cᵢ / 2) is formed; beyond that, the result is ±Inf.
@noinline function _extreme_projection!(y, f, x, a, d, γ, ::Type{S}) where {S}
  isfinite(d) && !(d >= floatmin(typeof(d))) && return false
  m = _maximum_abs(x)
  isfinite(m) || return false
  e = exponent(f.s[1]) + exponent(f.s[2])
  k = _comparison_exponent(_magnitude_exponent(m), _magnitude_exponent(abs(f.b)) + e)
  G = _scaled_dot(a, x, _power_factors(S, -k)) - ldexp(convert(S, f.b), e - k)
  if f isa IndHalfspace ? !(G > 0) : iszero(G)
    y === x || (y .= x)
    return true
  end
  D, w = _extreme_denominator(f, a, d, γ, S)
  _extreme_output!(y, _ExtremeOutput(G / D, k - w), x, a, γ)
  true
end

# d = D 2^w, for the scalar step and for a scalar normal in a diagonal metric ...
function _extreme_denominator(f, a, d, ::Nothing, ::Type{S}) where {S}
  w = exponent(d)
  ldexp(convert(S, d), -w), w
end
function _extreme_denominator(f, a::Real, d, γ::AbstractArray, ::Type{S}) where {S}
  if isfinite(d)
    w = exponent(d)
    return ldexp(convert(S, d), -w), w
  end
  w = exponent(_maximum_abs(γ))
  convert(S, f.q) * _scaled_sum(γ, _power_factors(S, -w)), w
end
# ... and for steps with an array normal, from the largest term of the sum.
function _extreme_denominator(f, a::AbstractArray, d, γ::AbstractArray, ::Type{S}) where {S}
  w = typemin(Int)
  @inbounds for i in eachindex(IndexLinear(), γ)
    α, g = convert(S, float(a[i])), convert(S, γ[i])
    iszero(α) || (w = max(w, exponent(g) + 2exponent(α)))
  end
  D = zero(S)
  @inbounds for i in eachindex(IndexLinear(), γ)
    α = convert(S, float(a[i]))
    D += _scaled_product(convert(S, γ[i]), α, α, -w)
  end
  D, w
end

struct _ExtremeOutput{S<:AbstractFloat}
  t::S
  J::Int
end
@inline function _scaled_correction(o::_ExtremeOutput{S}, xi, γi, ai) where {S}
  xs, γs, α = convert(S, xi), convert(S, γi), convert(S, float(ai))
  c = _scaled_product(o.t, γs, α, o.J)
  isfinite(c) && return xs - c
  ldexp(xs / 2 - _scaled_product(o.t, γs, α, o.J - 1), 1)
end
@inline (o::_ExtremeOutput{S})(xi, ai) where {S} = _scaled_correction(o, xi, one(S), ai)
@inline (o::_ExtremeOutput)(xi, ai, γi) = _scaled_correction(o, xi, γi, ai)
_extreme_output!(y, o, x, a, ::Nothing) = (y .= o.(x, _checked_parameter(y, a)))
_extreme_output!(y, o, x, a, γ) = (y .= o.(x, _checked_parameter(y, a), _checked_parameter(y, γ)))

_decided_on_spectrum(::Union{IndHyperplane, IndHalfspace, IndHyperslab}) = true
function _spectrum_value(f::Union{IndHyperplane, IndHalfspace, IndHyperslab}, v::AbstractVector{T}, k::Int) where {T}
  k == 0 && return f(v)
  _checkparams(f, v)
  if f isa IndHyperslab
    isfinite(f.l) && !iszero(_spectrum_linear(f, v, k, f.l, true)) && return T(Inf)
    isfinite(f.u) && return _spectrum_linear(f, v, k, f.u, false)
    return zero(T)
  end
  _spectrum_linear(f, v, k, f.b, false)
end
function _spectrum_linear(f, v::AbstractVector{T}, k, b, lower) where {T}
  S = promote_type(T, typeof(f.q))
  e = exponent(f.s[1]) + exponent(f.s[2])
  j = _comparison_exponent(_magnitude_exponent(S(_spectrum_scale(v))),
    _magnitude_exponent(abs(b)) + e - k)
  factors = _power_factors(S, -j)
  offset = ldexp(S(b), e - k - j)
  g = _scaled_dot(_normalized_normal(f.a, f.s), v, factors) - offset
  q = f.a isa Real ? sqrt(S(f.q)) * sqrt(S(length(v))) : sqrt(S(f.q))
  tol = _isscaled(f) ? max(_tol(T), _tol(typeof(f.q))) : _tol(T)
  slack = tol * max(ldexp(one(S), e - k - j), abs(offset), q * sqrt(_scaled_sumabs2(v, factors)))
  _ind(f isa IndHyperplane ? abs(g) <= slack : (lower ? -g <= slack : g <= slack), T)
end

# a step that the indicator projections reject, found before any block of a sum is written
_checkstep(op::O, ::Union{IndBallL0, IndBallL1, IndBallL2, IndSphereL2, IndSimplex}, y::AbstractArray, x::AbstractArray, γ::Real) where {O} =
  _checkstep_nonneg(γ)

####################################
#  a SymmetricSparseIterate
####################################

# The direct projections, the ℓ0 ball and the hyperplanes, halfspaces and hyperslabs, with the owner of a
# symmetric matrix as the destination or as the point. The matrix of the owner is the argument or the output:
# a mirrored pair is one coordinate counted twice in the norm and in ⟨a, X⟩. The ℓ0 ball counts the
# entries of the matrix (a pair twice), read from the lower triangle, so an owner or a Symmetric argument
# reaches it as a Symmetric one.
# The value of the four direct projections at an owner is not defined; the value of the ℓ0 ball, and of
# a hyperplane, halfspace or hyperslab, is.
const _OwnerSets = Union{IndBallL0, IndBallL1, IndBallL2, IndSphereL2, IndSimplex, IndHyperplane, IndHalfspace, IndHyperslab}
_dense_set_point(f, x::SymmetricSparseIterate) = matrixview(x)
_dense_set_point(f::IndBallL0, x::SymmetricSparseIterate) = Symmetric(matrixview(x), :L)

prox!(y::SymmetricSparseIterate, f::_OwnerSets, x::_ArrayOrOwner, γ::Real) = _owner_prox!(y, f, x, γ)
value_and_prox!(y::SymmetricSparseIterate, f::_OwnerSets, x::_ArrayOrOwner, γ::Real) = _owner_value_and_prox!(y, f, x, γ)
# a dense output reads the parameters at every entry of the matrix of the owner: they are checked first
prox!(y::AbstractArray, f::_OwnerSets, x::SymmetricSparseIterate, γ::Real) =
  (_check_entry_arrays(f, x); prox!(y, f, _dense_set_point(f, x), γ))
value_and_prox!(y::AbstractArray, f::_OwnerSets, x::SymmetricSparseIterate, γ::Real) =
  (_check_entry_arrays(f, x); value_and_prox!(y, f, _dense_set_point(f, x), γ))
prox(f::_OwnerSets, x::SymmetricSparseIterate, γ::Real = 1) = prox!(similar(x, float(eltype(x))), f, x, γ)

(f::IndHyperplane)(x::SymmetricSparseIterate) = (_check_value_arrays(f, x); _value_checked(f, x))
(f::IndHalfspace)(x::SymmetricSparseIterate) = (_check_value_arrays(f, x); _value_checked(f, x))
(f::IndHyperslab)(x::SymmetricSparseIterate) = (_check_value_arrays(f, x); _value_checked(f, x))
