"""
    IndBallL0(s; diagonal = false)

Indicator of the exactly symmetric `p × p` matrices (`p ≥ 1`) with at most `s`
nonzero off-diagonal entries, `#{(i, j) : i ≠ j, Xᵢⱼ ≠ 0} ≤ s`. Each nonzero
off-diagonal pair `{i, j}` counts two entries. The diagonal is not counted and
may be anything. The set is nonconvex, so `isconvex` is `false`. The budget `s`
is a nonnegative integer; one above `typemax(Int)` is stored as `typemax(Int)`,
and `IndBallL0(s)` allows `s ÷ 2` nonzero pairs.

Only `diagonal = false` is accepted. The archived function counted the diagonal
by default (`diagonal = true`); that form and its default are refused with an
`ArgumentError`, so a call such as `IndBallL0(4)` fails instead of changing
meaning. A negative `s` throws an `ArgumentError` too.

The arguments are a `Matrix{T}` or a `Symmetric{T, Matrix{T}}` with
`T` equal to `Float32` or `Float64`. The output of the proximal operator is a
`Matrix{T}` of the same size and element type. Other arguments throw an
`ArgumentError` that names the supported types: vectors, other element types
(`Int`, `BigFloat`), a `Hermitian`, views, sparse matrices, a
[`SymmetricSparseIterate`](@ref), mixed `Float32` and `Float64`, and array
steps. A non-square matrix, or an output of another size, throws a
`DimensionMismatch`. A `0 × 0` matrix throws an `ArgumentError`. A `Matrix`
that is not exactly symmetric throws an `ArgumentError` from the symmetry check
of the package, in the value and in the proximal operator; pass
`Symmetric(X, :L)` to use its lower triangle. A `Symmetric(A, :U)` argument
reads the upper triangle of `A`.

# Value

The value is `0` when `2 * #{i > j : Xᵢⱼ ≠ 0} ≤ s` and `Inf` otherwise. A `NaN`
in a strict lower entry gives `NaN`. An infinite entry counts as nonzero, and
zeros of either sign count as zero. The diagonal is not read for the count.

# Projection

`prox!(Y, f, X, γ)` writes the Euclidean (Frobenius) projection of `X` onto the
set. Let `n = p(p - 1) ÷ 2`, `k = min(s ÷ 2, n)`, and let `E` be the number of
nonzero strict lower entries of `X`. If `E ≤ k`, every pair is kept. Otherwise
the `k` strict lower entries of largest magnitude are kept and every other
off-diagonal entry becomes `+0`, a `-0` input included. Keeping the pair `{i, j}`
of an exactly symmetric `X` saves `2 Xᵢⱼ²` of the squared error, so the `k`
largest magnitudes are optimal. Ties in magnitude go to the entry earlier in the
strict lower triangle read by columns, `(2, 1), (3, 1), …, (p, 1), (3, 2), …`,
so that the output does not depend on the selection algorithm. Magnitudes are
compared with `abs`, without squaring, and are ranked correctly across the full
exponent range of `Float32` and `Float64`.

For the matrix `[5 1 -3; 1 7 2; -3 2 9]` and `s = 2` or `s = 3`, `k = 1` and the
pair `(3, 1)` with value `-3` is kept: the projection is `[5 0 -3; 0 7 0; -3 0 9]`.
For `s = 6` the matrix is returned unchanged.

```julia
f = IndBallL0(2; diagonal = false)
Y = prox(f, [5.0 1 -3; 1 7 2; -3 2 9])   # [5 0 -3; 0 7 0; -3 0 9]
f(Y)                                     # 0.0
```

Every pair is written from the logical lower entry `X[i, j]`, `i > j`, into both
`Y[i, j]` and `Y[j, i]`, and each diagonal entry is copied. The output is
symmetric, also for an accepted input pair `(+0.0, -0.0)`: both slots receive the
lower entry. Exact input and output identity `Y === X` and, for a `Symmetric` input,
`Y === parent(X)` are supported. Other overlaps are not supported.

# Steps and finite input

The step `γ` is a scalar, finite and nonnegative; the projection does not depend on
it, and `γ = 0` projects too. A negative, `NaN` or infinite step throws an
`ArgumentError` before anything is written. One consequence is that `Scaled(f, α)`
passes `α * γ` to `f`, so `Scaled(f, Inf)` at a positive step, or a finite `α` whose
product `α * γ` overflows, throws an `ArgumentError` before writing (`Scaled(f, 0)`
is `f`).

The proximal operator refuses an input with a nonfinite entry on the diagonal or in
the strict lower triangle: it throws an `ArgumentError` before anything is written.
The check and every other check of the arguments precede the first write.

[`value_and_prox!`](@ref) returns `f(Y)` read from the stored output.

# Cost and buffer

The projection takes `O(p²)` operations on typical input when `E > k`, by an
in-place selection of the `k`-th largest magnitude, and `O(p²)` when `E ≤ k`. The
selection pivot rule has a worst case of `O(p⁴)`. The object keeps one buffer of `n`
magnitudes of the element type. A repeated call with the same size and element type
does not allocate. Use a separate `IndBallL0(f.s; diagonal = false)` in each task whose
calls may run at the same time, since the buffer belongs to the object.

The proximal operator, `value_and_prox!` and the value have no `SymmetricSparseIterate`
form. `kkt_violation`, `restrict`, `prox_block` and `lipschitz` have no method for
this function.
"""
struct IndBallL0 <: AbstractFunction
  s::Int
  scratch::Scratch

  function IndBallL0(s::Integer; diagonal::Bool = true)
    diagonal && throw(ArgumentError(
      "IndBallL0 supports only the budget on off-diagonal entries; use IndBallL0(s; diagonal = false)"))
    s >= 0 || throw(ArgumentError("s must be nonnegative, got $s"))
    new(s > typemax(Int) ? typemax(Int) : Int(s), Scratch())
  end
end

isconvex(::Type{IndBallL0}) = false
isproximable(::Type{IndBallL0}) = true
_isindicator(::Type{IndBallL0}) = true

# The buffer is not part of the function.
Base.:(==)(f::IndBallL0, g::IndBallL0) = f.s == g.s
Base.hash(f::IndBallL0, h::UInt) = hash(f.s, hash(:IndBallL0, h))
Base.show(io::IO, f::IndBallL0) = print(io, "IndBallL0(", f.s, "; diagonal = false)")

const _L0Float = Union{Float32, Float64}
const _L0Input{T} = Union{Matrix{T}, Symmetric{T, Matrix{T}}}

@noinline _l0_unsupported(what, T) = throw(ArgumentError(
  "IndBallL0 supports a Matrix or Symmetric of Float32 or Float64 as the argument, a Matrix of the same " *
  "element type as the output, and a scalar step; got $what of type $T"))

@noinline _l0_step_error(γ) = throw(ArgumentError(
  "the step of IndBallL0 must be finite and nonnegative, got $γ"))
@noinline _l0_nonfinite(i, j, v) = throw(ArgumentError(
  "IndBallL0 projects finite entries only, but entry ($i, $j) is $v"))
@noinline _l0_empty() = throw(ArgumentError("IndBallL0 needs a matrix with at least one row and column"))

const _L0Subject = "the argument of IndBallL0"

# the dimension p of a square matrix with p ≥ 1
@inline function _l0_dimension(X)
  p = checksquare(X)
  p >= 1 || _l0_empty()
  p
end

# the unsupported arguments of the value, the proximal operator and the fused call
(f::IndBallL0)(X) = _l0_unsupported("the argument", typeof(X))
prox!(Y, f::IndBallL0, X, γ) = _l0_unsupported("the output, input or step", typeof.((Y, X, γ)))
value_and_prox!(Y, f::IndBallL0, X, γ::Real) = _l0_unsupported("the output, input or step", typeof.((Y, X, γ)))
value_and_prox!(Y, f::IndBallL0, X, γ::AbstractArray) = _l0_unsupported("the step", typeof(γ))
value_and_prox!(Y::SymmetricSparseIterate, f::IndBallL0, X, γ::AbstractArray) = _l0_unsupported("the step", typeof(γ))
prox!(Y::SymmetricSparseIterate, f::IndBallL0, X, γ::AbstractArray) = _l0_unsupported("the output", typeof(Y))
prox(f::IndBallL0, X::SymmetricSparseIterate, γ::Real = 1) = _l0_unsupported("the argument", typeof(X))
prox(f::IndBallL0, X::SymmetricSparseIterate, γ::AbstractArray) = _l0_unsupported("the argument", typeof(X))

function (f::IndBallL0)(X::_L0Input{T}) where {T<:_L0Float}
  p = _l0_dimension(X)
  _check_symmetric(X, _L0Subject, "X")
  E = 0
  @inbounds for c in 1:p, r in c+1:p
    v = X[r, c]
    isnan(v) && return T(NaN)
    E += !iszero(v)
  end
  E <= f.s ÷ 2 ? zero(T) : T(Inf)
end

function prox!(Y::Matrix{T}, f::IndBallL0, X::_L0Input{T}, γ::Real) where {T<:_L0Float}
  p = _l0_dimension(X)
  size(Y) == size(X) || throw(DimensionMismatch("the output has size $(size(Y)) and the input $(size(X))"))
  (isfinite(γ) && γ >= 0) || _l0_step_error(γ)
  _check_symmetric(X, _L0Subject, "X")
  n = p * (p - 1) ÷ 2
  k = min(f.s ÷ 2, n)
  E = _l0_count(X, p)
  if E <= k
    _l0_write!(Y, X, p, zero(T), 0, true)
  else
    b = _array_buffer!(f.scratch, T, (n,))
    _l0_fill!(b, X, p)
    τ = _l0_kth_largest!(b, k)
    _l0_write!(Y, X, p, τ, k - _l0_count_above(b, τ), false)
  end
  Y
end

# The value of the stored output, which is feasible by construction.
function value_and_prox!(Y::Matrix{T}, f::IndBallL0, X::_L0Input{T}, γ::Real) where {T<:_L0Float}
  prox!(Y, f, X, γ)
  f(Y)
end
_value_and_prox_unchecked!(Y::Matrix{T}, f::IndBallL0, X::_L0Input{T}, γ::Real) where {T<:_L0Float} =
  value_and_prox!(Y, f, X, γ)

# the number of nonzero strict lower entries; every diagonal and strict lower
# entry must be finite
function _l0_count(X, p)
  E = 0
  @inbounds for c in 1:p
    v = X[c, c]
    isfinite(v) || _l0_nonfinite(c, c, v)
    for r in c+1:p
      v = X[r, c]
      isfinite(v) || _l0_nonfinite(r, c, v)
      E += !iszero(v)
    end
  end
  E
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

# The k-th largest of the finite magnitudes b, 1 ≤ k ≤ length(b), by an in-place
# quickselect with a three-way partition and a median-of-three pivot. A value
# equal to the pivot ends the search, so repeated magnitudes cost no extra rounds.
# The order of b is not kept. For k = 0 nothing is kept: the threshold is infinite.
function _l0_kth_largest!(b::Vector{T}, k::Int) where {T}
  k >= 1 || return T(Inf)
  lo, hi = 1, length(b)
  @inbounds while lo < hi
    mid = lo + (hi - lo) >> 1
    x, y, z = b[lo], b[mid], b[hi]
    pivot = max(min(x, y), min(max(x, y), z))
    # [lo, lt) above the pivot, [lt, i) equal to it, (gt, hi] below it
    i, lt, gt = lo, lo, hi
    while i <= gt
      v = b[i]
      if v > pivot
        b[i] = b[lt]
        b[lt] = v
        lt += 1
        i += 1
      elseif v < pivot
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

# Write each pair from its logical lower entry v = X[r, c] into both slots, and
# each diagonal entry from itself. With keepall every pair is kept. Otherwise an
# entry is kept when |v| > τ, or when |v| == τ and a place of the quota is left
# (the quota counts the tied entries to keep, in column-major order); other
# off-diagonal entries become +0. Each read precedes the writes to its two slots
# and no later read touches them, so Y may be X or parent(X).
function _l0_write!(Y, X, p, τ::T, quota::Int, keepall::Bool) where {T}
  @inbounds for c in 1:p
    Y[c, c] = X[c, c]
    for r in c+1:p
      v = X[r, c]
      m = abs(v)
      keep = keepall
      if !keep
        if m > τ
          keep = true
        elseif m == τ && quota > 0
          keep = true
          quota -= 1
        end
      end
      w = keep ? v : zero(T)
      Y[r, c] = w
      Y[c, r] = w
    end
  end
  Y
end
