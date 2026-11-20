"""
    NormL0(λ; diagonal = false)

The nonconvex penalty `g(X) = λ ⋅ #{(i, j) : i ≠ j, Xᵢⱼ ≠ 0}` on a square
matrix: each nonzero off-diagonal entry costs `λ` and the diagonal costs
nothing. For a symmetric `X`, a pair `Xᵢⱼ = Xⱼᵢ ≠ 0` with `i < j` contributes `2λ`. The level `λ`
is a finite real number that is not negative; for a real `λ` that is not finite
or is negative, the constructor throws `ArgumentError`. Both signed zeros count
as zero.

**Inputs.** `X` and the output `Y` are one-based strided `p × p` matrices
(`p = 0` is allowed) with the same element type, `Float32` or `Float64`, that
Base can both read and write. Distinct entries of `Y`, and of `X` when
`Y === X`, occupy distinct memory: a destination that repeats a storage
position, such as a zero-stride view, is unsupported and may go undetected, as
in `Base.mightalias`. `X` has finite entries. Symmetry of `X` under `==` is
the caller's precondition for the statements below about the symmetric
problem; it is not checked, and asymmetry alone causes no refusal. A pair
`(+0.0, -0.0)` is symmetric. The step `γ` is a finite real number that is
positive.

**Value.** With `k` the number of nonzero entries off the diagonal, `g(X)` has
the element type `T` of `X`. It is `zero(T)` when `k = 0` or `λ = 0`.
Otherwise it is `convert(T, R(λ) * R(k))` with
`R = promote_type(T, float(typeof(λ)))`, so a `Float64` level is not rounded to
`Float32` before the product. The result can overflow to `Inf` when the product
rounds beyond `floatmax(T)`.

**Proximal map.** For `V` that meets the input conditions and is symmetric under
`==`, the output `Y` of
`prox!(Y, g, V, γ)` minimizes `g(Z) + ‖Z - V‖²_F / (2γ)` over symmetric `Z`,
when `λ` and `γ` convert exactly to a `Float32` or `Float64` working type of
[`NormL0`](@ref). `NormL0` converts the entry, `λ` and `γ` to that working
type and compares the square of the entry with `2γλ`. Other levels and steps,
for example a `Rational` or an `Int` above `2^53`, are rounded to that type
first. With a `BigFloat` working type the converted entry, its square and the
product `2γλ` can each be rounded at the current precision. The decision below
is made with the converted and rounded values:
- The diagonal is copied bit for bit, including `-0.0`.
- Each entry `(i, j)` with `i ≠ j` is the `(i, j)` entry of
  `prox(NormL0(λ), V, γ)`, computed from `Vᵢⱼ` alone and not mirrored from
  `Vⱼᵢ`. For `λ > 0` the entry is kept when the scalar comparison of
  [`NormL0`](@ref) on the converted operands finds its square above `2γλ`,
  and is positive zero otherwise, so a tie of the converted operands gives
  zero. A mathematical tie can be kept: a pair of ones at `λ = 1//3` and
  `γ = 3//2` is kept. At `λ = 0` the map is the identity, and a `(+0.0, -0.0)` pair
  stays as it is.
- A pair `i < j` costs `2λ·1{x ≠ 0} + 2(x - v)²/(2γ)`. Dividing by two gives
  the scalar problem of `NormL0`, so the threshold `2γλ` is unchanged. The
  diagonal term `(x - v)²/(2γ)` has the minimizer `x = v`.
- `Y` is symmetric under `==` when `V` is.

For an asymmetric `X` the same entrywise rule is applied to each entry, so
neither triangle is selected. The result need not be the proximal point over
symmetric matrices, and the value counts every nonzero off-diagonal entry of
the given matrix.

A direct call of `value_and_prox!` returns the value at the stored `Y`.
`Y === X` is supported under the storage precondition above. `prox` returns `similar(X, T)`, a `Matrix` for Base arrays.

**Refusals.** Every refusal of a direct call happens before any write, so `X`
and `Y` are left unchanged. A call in which exactly one of the following
conditions fails throws `ArgumentError`:
- offset axes;
- an argument of one of these storage kinds: a vector, a tuple, a sparse or
  other non-strided matrix (for example `Symmetric`, `Adjoint` or
  [`matrixview`](@ref)), a [`SparseIterate`](@ref), a
  [`SymmetricSparseIterate`](@ref) or a [`BlockIterate`](@ref);
- an array step;
- an element type other than `Float32` and `Float64`, or different element
  types of `X` and `Y`;
- a reinterpreted array that Base cannot both read and write, such as a
  `Float64` reinterpretation of a parent whose element type has padding;
- a different `Y` for which `Base.mightalias(Y, X)` is true;
- a real scalar step that is nonpositive or non-finite;
- a non-finite entry of `X`.
A call in which the only failing condition is a non-square `X`, or a `Y` of
another size, throws `DimensionMismatch`. When several conditions fail, which
exception is thrown is not specified. Arguments outside these families, for
example a number as `X` or a step that is neither real nor an array, can give
`MethodError`. `prox_block`, `prox_block!`, `restrict` and `kkt_violation` are
not provided: called on this function directly with the argument counts of
their generic forms, they throw `ArgumentError`.

**Through `Scaled`.** For a scale `α` that `Scaled` accepts, the step passed to
this function, the value and the fused value of `Scaled(NormL0(λ; diagonal =
false), α)` follow the rules of `Scaled`, including its rounding in the
exceptional range (see [Scaling a function](@ref scaling-guide)). They can
differ from the ordinary products `α * γ` and `α * g(X)`, the fused value need
not equal the value of `Scaled` at the stored output, and a step that is not
finite and positive after this computation is refused. Through `Scaled`, the
unsupported arguments above are refused before any write, with an exception
from the preprocessing of `Scaled`, the checks of this function or dispatch,
for example `ArgumentError`, `DimensionMismatch`, `MethodError` or
`OverflowError`, and `Scaled` may read such arguments before the refusal.

Not provided: positive definiteness of the output, steps `0` and `Inf`, and
non-finite input. See the [numerical contract](@ref spec-norml0offdiag).
"""
NormL0(::Real)

# The function of NormL0(λ; diagonal = false). It is not exported; the keyword constructor in norml0.jl returns it.
struct _NormL0OffDiag{T<:Real} <: AbstractFunction
  λ::T

  function _NormL0OffDiag{T}(λ::T) where {T<:Real}
    isfinite(λ) || throw(ArgumentError("NormL0(λ; diagonal = false) requires a finite λ, got $λ"))
    _checknonneg("λ", λ)
    new{T}(λ)
  end
end

isproximable(::Type{<:_NormL0OffDiag}) = true

Base.show(io::IO, f::_NormL0OffDiag) = (print(io, "NormL0("); show(io, f.λ); print(io, "; diagonal = false)"))

@noinline _l0od_error(what) = throw(ArgumentError("NormL0(λ; diagonal = false) $what"))

# The checks on the matrix X of the value oracle; the proximal oracles use _l0od_check_pair. Storage and
# element type are checked before the shape.
function _l0od_check_matrix(X)
  _check_one_based(X)
  X isa StridedMatrix || _l0od_error("requires a strided matrix, got $(typeof(X))")
  _l0od_check_float(X)
  _l0od_check_access(X)
  checksquare(X)
  nothing
end
function _l0od_check_float(X)
  eltype(X) === Float32 || eltype(X) === Float64 ||
    _l0od_error("requires Float32 or Float64 entries, got $(eltype(X))")
  nothing
end

# A reinterpreted array whose parent has padding can be writable and not readable, or the reverse
# (Base.PaddingError). The check uses Base's own tests, through SubArray and ReshapedArray wrappers, before any write.
_l0od_accessible(A) = true
_l0od_accessible(A::Union{SubArray, Base.ReshapedArray}) = _l0od_accessible(parent(A))
function _l0od_accessible(A::Base.ReinterpretArray)
  try
    Base.check_readable(A)
    Base.check_writable(A)
  catch e
    e isa Base.PaddingError || rethrow()
    return false
  end
  _l0od_accessible(parent(A))
end
_l0od_check_access(A) =
  (_l0od_accessible(A) || _l0od_error("requires an array that Base can read and write, got $(typeof(A))"); nothing)

# A column sum has the integer width of the entries.
_l0od_zero(::Type{Float32}) = Int32(0)
_l0od_zero(::Type{Float64}) = Int64(0)

# A pass over the columns of a strided matrix other than a Float32 or Float64 Matrix, in column-major index
# order, counting the entries that are not finite. The error is raised by a separate @noinline helper.
# Symmetry of X is not read.
function _l0od_scan(X::StridedMatrix)
  p = size(X, 1)
  bad = 0
  @inbounds for j in 1:p
    s = _l0od_zero(eltype(X))
    @simd for i in 1:p
      s += !isfinite(X[i, j])
    end
    bad += s
  end
  bad == 0 || _l0od_nonfinite_error()
  nothing
end

@noinline _l0od_nonfinite_error() = _l0od_error("requires finite entries in X")

# A Matrix is read in one pass in linear index order. x - x is +0.0 for a finite x and NaN otherwise, so
# the sum is +0.0 exactly when every entry is finite, in any order of the additions.
function _l0od_scan(X::Matrix{T}) where {T<:Union{Float32, Float64}}
  s = zero(T)
  @inbounds @simd for k in eachindex(X)
    s += X[k] - X[k]
  end
  s == 0 || _l0od_nonfinite_error()
  nothing
end

# The number of nonzero entries off the diagonal.
function _l0od_count(X::StridedMatrix)
  p = size(X, 1)
  k = 0
  @inbounds for j in 1:p
    s = _l0od_zero(eltype(X))
    @simd for i in 1:p
      s += !iszero(X[i, j])
    end
    k += s - !iszero(X[j, j])
  end
  k
end

function _l0od_check_step(γ)
  (isfinite(γ) && γ > 0) || _l0od_error("requires a finite positive step, got $γ")
  nothing
end

function _l0od_check_pair(Y, X)
  _check_one_based(Y, X)
  (X isa StridedMatrix && Y isa StridedMatrix) ||
    _l0od_error("requires strided matrices, got $(typeof(Y)) and $(typeof(X))")
  eltype(Y) === eltype(X) || _l0od_error("requires the same element type in X and Y")
  _l0od_check_float(X)
  _l0od_check_access(X)
  _l0od_check_access(Y)
  checksquare(X)
  _checksize(Y, X)
  Y !== X && Base.mightalias(Y, X) && _l0od_error("requires Y to be X or independent of it")
  nothing
end

@inline _l0od_value(::Type{T}, λ, k::Integer) where {T} =
  (k == 0 || iszero(λ)) ? zero(T) : (R = promote_type(T, float(typeof(λ))); convert(T, R(λ) * R(k)))

function (f::_NormL0OffDiag)(X::AbstractMatrix)
  _l0od_check_matrix(X)
  _l0od_scan(X)
  _l0od_value(eltype(X), f.λ, _l0od_count(X))
end

# Both rules decide one entry from its own value. The cutoff of NormL0 is used
# in the same cases as in its dense kernel: equal element, level and step types
# and at least 64 entries. The union result is handed to a loop that does not
# depend on it.
@inline _l0od_rule(f::_NormL0OffDiag, X, γ) = NormL0(f.λ)
@inline function _l0od_rule(f::_NormL0OffDiag{T}, X::AbstractMatrix{T}, γ::T) where {T<:Union{Float32, Float64}}
  g = NormL0(f.λ)
  length(X) < 64 ? g : _l0_cutoff(g, γ)
end

# Writes the output; the diagonal entry of each column is restored after the column is written.
# The loops use @simd without ivdep. On Julia 1.10, a reinterpreted Float32 matrix over Float64 or
# ComplexF32 storage reads and writes entries through shared memory words, and the loop with ivdep gave
# wrong entries for such an input or output. The tests run with --check-bounds=yes, which hides this.
function _l0od_loop!(Y::StridedMatrix, g, X::StridedMatrix, γ)
  p = size(X, 1)
  @inbounds for j in 1:p
    d = X[j, j]
    @simd for i in 1:p
      Y[i, j] = prox_block(g, X[i, j], 1, γ)
    end
    Y[j, j] = d
  end
  Y
end

# The single-array loop for Y === X, so that no second array is read or written.
function _l0od_inplace!(X::StridedMatrix, g, γ)
  p = size(X, 1)
  @inbounds for j in 1:p
    d = X[j, j]
    @simd for i in 1:p
      X[i, j] = prox_block(g, X[i, j], 1, γ)
    end
    X[j, j] = d
  end
  X
end

function _l0od_prox!(Y, f::_NormL0OffDiag, X, γ)
  _l0od_check_pair(Y, X)
  _l0od_check_step(γ)
  _l0od_scan(X)
  g = _l0od_rule(f, X, γ)
  Y === X ? _l0od_inplace!(X, g, γ) : _l0od_loop!(Y, g, X, γ)
  nothing
end

function prox!(Y::AbstractMatrix, f::_NormL0OffDiag, X::AbstractMatrix, γ::Real)
  _l0od_prox!(Y, f, X, γ)
  Y
end

function value_and_prox!(Y::AbstractMatrix, f::_NormL0OffDiag, X::AbstractMatrix, γ::Real)
  _l0od_prox!(Y, f, X, γ)
  _l0od_value(eltype(Y), f.λ, _l0od_count(Y))
end

# Arguments without a method above: the value, prox! and value_and_prox! of other argument types, and the
# block, restriction and screening operations. The methods with typed arguments repeat the signatures of
# generic methods in interface.jl, block_prox.jl and symmetric_parameters.jl, so that no call is ambiguous.
@noinline _l0od_unsupported(what, T) = throw(ArgumentError(
  "NormL0(λ; diagonal = false) supports strided square matrices with Float32 or Float64 entries and a real " *
  "step; got $what of type $T"))
@noinline _l0od_not_provided(op) = throw(ArgumentError("NormL0(λ; diagonal = false) does not provide $op"))

(f::_NormL0OffDiag)(X) = _l0od_unsupported("the argument", typeof(X))
prox!(Y, f::_NormL0OffDiag, X, γ) = _l0od_unsupported("the output, input or step", typeof.((Y, X, γ)))
prox!(Y::SymmetricSparseIterate, f::_NormL0OffDiag, X, γ::AbstractArray) =
  _l0od_unsupported("the output, input or step", typeof.((Y, X, γ)))
value_and_prox!(Y, f::_NormL0OffDiag, X, γ) = _l0od_unsupported("the output, input or step", typeof.((Y, X, γ)))
value_and_prox!(Y, f::_NormL0OffDiag, X, γ::Real) = _l0od_unsupported("the output, input or step", typeof.((Y, X, γ)))
value_and_prox!(Y::SymmetricSparseIterate, f::_NormL0OffDiag, X, γ::AbstractArray) =
  _l0od_unsupported("the output, input or step", typeof.((Y, X, γ)))
prox(f::_NormL0OffDiag, X::SymmetricSparseIterate, γ::Real = 1) = _l0od_unsupported("the input", typeof(X))
# The allocating oracle checks the storage and element type of X before it allocates the output. The
# Symmetric and Hermitian method matches the generic method of interface.jl for those wrappers.
function _l0od_prox_alloc(f, X, γ)
  _l0od_check_matrix(X)
  prox!(similar(X, eltype(X)), f, X, γ)
end
prox(f::_NormL0OffDiag, X::AbstractArray, γ::Real = 1) = _l0od_prox_alloc(f, X, γ)
prox(f::_NormL0OffDiag, X::Union{Symmetric, Hermitian}, γ::Real = 1) = _l0od_prox_alloc(f, X, γ)
prox(f::_NormL0OffDiag, X::AbstractArray, γ::AbstractArray) = _l0od_unsupported("the step", typeof(γ))
prox(f::_NormL0OffDiag, X::Tuple, γ::Real = 1) = _l0od_unsupported("the input", typeof(X))
prox(f::_NormL0OffDiag, X::Tuple, γ::AbstractArray) = _l0od_unsupported("the input", typeof(X))

prox_block(f::_NormL0OffDiag, v, k, γ) = _l0od_not_provided(:prox_block)
prox_block(f::_NormL0OffDiag, u::AbstractVector, k::Integer, γ::Real) = _l0od_not_provided(:prox_block)
prox_block!(out, f::_NormL0OffDiag, u, k, γ) = _l0od_not_provided(:prox_block!)
restrict(f::_NormL0OffDiag, S) = _l0od_not_provided(:restrict)
kkt_violation(f::_NormL0OffDiag, v, i) = _l0od_not_provided(:kkt_violation)
kkt_violation(f::_NormL0OffDiag, v, x, k) = _l0od_not_provided(:kkt_violation)
kkt_violation(f::_NormL0OffDiag, v::AbstractArray, x::AbstractArray, k::Integer) = _l0od_not_provided(:kkt_violation)
kkt_violation(f::_NormL0OffDiag, v::AbstractArray, x::BlockIterate, k::Integer) = _l0od_not_provided(:kkt_violation)
kkt_violation(f::_NormL0OffDiag, v::AbstractArray, x::SymmetricSparseIterate, k::Integer) =
  _l0od_not_provided(:kkt_violation)
