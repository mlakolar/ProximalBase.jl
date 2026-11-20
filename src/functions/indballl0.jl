"""
    IndBallL0(s; diagonal = false)

Indicator of the symmetric `p × p` matrices (`p ≥ 1`) with at most `s` nonzero
off-diagonal entries, `#{(i, j) : i ≠ j, Xᵢⱼ ≠ 0} ≤ s`. Each nonzero off-diagonal
pair `{i, j}` counts two entries, so `s` allows `s ÷ 2` nonzero pairs. The diagonal
is not counted: the value does not read it, and the projection copies it (finite
entries only). Symmetric means `Xᵢⱼ == Xⱼᵢ` as the symmetry check of the package
reads it: `0.0` and `-0.0` are equal, and `NaN` matches `NaN`. The set is nonconvex
for `1 ≤ s ÷ 2 < p(p - 1)/2`; `isconvex` is `false` for every budget. The budget `s`
is a nonnegative integer; one above `typemax(Int)` is stored as `typemax(Int)`.

Only `diagonal = false` is accepted. The archived function counted the diagonal by
default; `diagonal = true`, and so a call such as `IndBallL0(4)`, throws an
`ArgumentError`. A negative `s` throws an `ArgumentError` too.

The arguments are a `Matrix{T}` or a `Symmetric{T, Matrix{T}}` with `T` equal to
`Float32` or `Float64`, and the output of the proximal operator is a `Matrix{T}` of
the same size. Other array arguments, outputs and array steps (a vector, a view, an
`Adjoint`, a `Hermitian`, a sparse matrix, a [`SymmetricSparseIterate`](@ref), another
element type, mixed `Float32` and `Float64`) throw an `ArgumentError` that names the
supported types; through `prox`, an array with offset axes gets the one-based-indexing
`ArgumentError` instead. A scalar argument, or a step that is neither a real number nor
an array, can give a `MethodError`. A non-square matrix, or an output of another size,
throws a `DimensionMismatch`, and a `0 × 0` matrix an `ArgumentError`. A `Matrix` that
is not symmetric in the sense above throws an `ArgumentError` in the value and in the
proximal operator; pass `Symmetric(X, :L)` to use its lower triangle. A
`Symmetric(A, :U)` argument reads the upper triangle of `A`.

# Value

The value is `0` when `2 * #{i > j : Xᵢⱼ ≠ 0} ≤ s` and `Inf` otherwise. A `NaN` in a
strict lower entry gives `NaN`. An infinite strict lower entry counts as nonzero, and
zeros of either sign count as zero.

# Projection

`prox!(Y, f, X, γ)` writes the Euclidean (Frobenius) projection of `X` onto the set.
Let `n = p(p - 1) ÷ 2`, `k = min(s ÷ 2, n)`, and let `E` be the number of nonzero
strict lower entries of `X`. If `E ≤ k`, every pair is kept. Otherwise the `k` strict
lower entries of largest magnitude are kept and every other off-diagonal entry becomes
`+0`, a `-0` input included. For a finite symmetric `X`, keeping the pair `{i, j}`
saves `2 Xᵢⱼ²` of the squared error, so the `k` largest magnitudes give the minimum
error. Ties in magnitude go to the entry earlier in the strict lower triangle read by
columns, `(2, 1), (3, 1), …, (p, 1), (3, 2), …`. Magnitudes are compared as `abs`
values, without squaring.

For the matrix `[5 1 -3; 1 7 2; -3 2 9]` and `s = 2` or `s = 3`, `k = 1` and the pair
`(3, 1)` with value `-3` is kept: the projection is `[5 0 -3; 0 7 0; -3 0 9]`. For
`s = 6` the matrix is returned unchanged.

```julia
f = IndBallL0(2; diagonal = false)
Y = prox(f, [5.0 1 -3; 1 7 2; -3 2 9])   # [5 0 -3; 0 7 0; -3 0 9]
f(Y)                                     # 0.0
```

Each kept pair is written from its logical lower entry `X[i, j]`, `i > j`, into both
`Y[i, j]` and `Y[j, i]`, a dropped pair is `+0` in both, and each diagonal entry is
copied. The output is symmetric, also for an accepted input pair `(+0.0, -0.0)`.
`Y === X` and, for a `Symmetric` input, `Y === parent(X)` are supported. Other
overlaps are not supported. [`value_and_prox!`](@ref) returns `f(Y)` read from the
stored output.

# Steps and finite input

The step `γ` is a scalar that must be finite and nonnegative; the projection does not
depend on it, and `γ = 0` projects too. A negative, `NaN` or infinite step throws an
`ArgumentError` before anything is written. A single `Scaled(f, α)` passes `α * γ`
to `f`, so `Scaled(f, Inf)` at a positive step, or a finite `α` whose product `α * γ`
overflows, throws an `ArgumentError` before writing (`Scaled(f, 0)` is `f`). A nested
chain of `Scaled` wrappers whose factor product overflows passes `γ` to `f`, so it
projects at a finite `γ ≥ 0`.

The proximal operator throws an `ArgumentError` before anything is written for an
input with a nonfinite entry on the diagonal or in the strict lower triangle. The
checks of the shapes, the step, the symmetry and the finiteness of the entries precede
the first write.

# Cost and buffer

The projection takes `O(p²)` operations when `E ≤ k`. When `E > k`, the `k`-th largest
magnitude is found by an in-place quickselect with pivot positions drawn from a
pseudo-random sequence seeded by `n`; its expected cost is `O(p²)` and its worst case is
quadratic in the number of pairs. The object keeps one buffer of `n` magnitudes of the
element type, allocated at the first call that needs it and again when the size or the
element type changes. A repeated call on the same object with the same size and element
type does not allocate. Use a separate `IndBallL0(f.s; diagonal = false)` in each task
whose calls may run at the same time, since the buffer belongs to the object.

`kkt_violation`, `restrict`, `prox_block` and `lipschitz` are not supported:
`lipschitz` throws an `ArgumentError`, and `restrict`, `prox_block` and
`kkt_violation` a `MethodError`.
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

# The value of the stored output, which is feasible for the finite input that prox! accepts.
function value_and_prox!(Y::Matrix{T}, f::IndBallL0, X::_L0Input{T}, γ::Real) where {T<:_L0Float}
  prox!(Y, f, X, γ)
  f(Y)
end
_value_and_prox_unchecked!(Y::Matrix{T}, f::IndBallL0, X::_L0Input{T}, γ::Real) where {T<:_L0Float} =
  value_and_prox!(Y, f, X, γ)

# the number of nonzero strict lower entries; throws if a diagonal or strict lower
# entry is not finite
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

# The k-th largest of the finite magnitudes b, 0 ≤ k ≤ length(b), by an in-place
# quickselect with a three-way partition and the median of three entries at
# positions from a local xorshift generator (seeded by length(b), so the result and
# the permutation of b are the same for the same input). The entries equal to the pivot leave
# the range of the later rounds. The order of b is not kept. For k = 0 nothing is
# kept: the threshold is infinite.
@inline function _l0_xorshift(x::UInt64)
  x ⊻= x << 13
  x ⊻= x >> 7
  x ⊻= x << 17
  x
end

function _l0_kth_largest!(b::Vector{T}, k::Int) where {T}
  k >= 1 || return T(Inf)
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
# and no later read touches them, so Y may be X (a Matrix) or parent(X) (a Symmetric).
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
