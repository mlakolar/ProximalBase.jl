####################################
#  linear
####################################

"""
    LinearFunction(c, d = 0)

The linear function `f(x) = ⟨c, x⟩ + d`, with gradient `c`. Its proximal operator
is the shift `x - γc`.

The coefficients and the argument have the same dimensions, `size(c) == size(x)`;
on a `SymmetricSparseIterate`, `c` is `p × p`. Every oracle throws a
`DimensionMismatch` for any other pair, even one with as many entries, before it writes.

At a matrix `SparseIterate` the value is the sum over its stored entries whose value is
nonzero, and reads `c` at those entries only. A `Diagonal` `c` is read at the nonzero stored
diagonal entries, in `O(p)`; the other entries of `x` are ignored, as in the diagonal-only
`dot` of `Diagonal`. An entry that is not stored, or is stored as `+0` or `-0`, adds nothing,
even if the coefficient there is `NaN`, `Inf` or `-Inf`; the constant `d` is added after the
sum. A nonzero entry follows ordinary arithmetic, also when it or its coefficient is not
finite. The order of the summation is not specified (the reduction is vectorized, so its
partial sums follow neither the storage order nor the index order), and it can change the IEEE
class of the result when an entry is not finite. For example, with `c = ones(2, 2)`, `d = 0`
and `h = 2^exponent(floatmax(T))`, storing `x[2] = h`, `x[3] = h` and then `x[1] = -Inf`, a
serial sum in the storage order overflows to `+Inf` before it adds `-Inf` and gives `NaN`,
while the dense `dot` gives `-Inf`.
When `d`, the nonzero entries of `x` and the coefficients at them are finite and the exact
value lies inside the range of the type by the accuracy of the sum, the value is finite,
also if products or partial sums overflow.

```julia
x = SparseIterate(2, 2); x[1, 1] = 2.0
f = LinearFunction([1.0 NaN; 1.0 1.0], 0.25)    # NaN at an entry of x that is not stored
f(x)                                            # 2.25
f(Matrix(x))                                    # NaN, as dot(f.c, Matrix(x)) + f.d
```

This rule is for the value at a matrix `SparseIterate`. For a dense `c` that is not a
`Diagonal`, `dot(c, x)` is the generic `dot`, which visits all positions; for a `Diagonal` it is
the diagonal-only `dot` of `Diagonal`. A vector `SparseIterate` and a `SymmetricSparseIterate`
keep their own rules.
"""
struct LinearFunction{C<:AbstractArray, D<:Real} <: AbstractFunction
  c::C
  d::D
  function LinearFunction{C, D}(c, d) where {C<:AbstractArray, D<:Real}
    eltype(c) <: Real || _real_error(:LinearFunction)
    new{C, D}(c, d)
  end
end
LinearFunction(c::AbstractArray, d::Real) = LinearFunction{typeof(c), typeof(d)}(c, d)

LinearFunction(c::AbstractArray) = (eltype(c) <: Real || _real_error(:LinearFunction); LinearFunction(c, zero(float(eltype(c)))))

issmooth(::Type{<:LinearFunction}) = true
isproximable(::Type{<:LinearFunction}) = true
isseparable(::Type{<:LinearFunction}) = true
isconvex(::Type{<:LinearFunction}) = true

(f::LinearFunction)(x::_ArrayOrSymmetricIterate) =
  (_check_value_arrays(f, x); _value_checked(f, x))
_value_unchecked(f::LinearFunction, x::AbstractArray) = (_checkvalue(f, x); _value_at(_linear_value(f, x, dot(f.c, x) + f.d), x))
# the matrix that the symmetric iterate represents, whose off-diagonal entries count at both positions
_value_unchecked(f::LinearFunction, x::SymmetricSparseIterate) =
  (_checkvalue(f, x); _value_at(_linear_value(f, x, _symmetric_matrix_dot(f.c, x) + f.d), x))

# A dot that overflows, alone or with d, at a representable value is summed
# again with x in units of 2^k, chosen so that no partial sum can overflow; d
# keeps its own units, so that a small d survives a dot that cancels (R1). A matrix
# SparseIterate takes the exponent-separated recomputation below instead.
@inline _linear_value(f::LinearFunction, x, v) = isfinite(v) ? v : _linear_value_recover(f.c, f.d, x, v)
@noinline function _linear_value_recover(c, d, x, v)
  (_all_finite(x) && _all_finite(c) && isfinite(d)) || return v
  R = typeof(v)
  k = _product_sum_shift(R, _stored_maxabs(c), _stored_maxabs(x), _nentries(x), 2)
  p = convert(R, _linear_scaled_dot(c, x, _power_factors(R, -k)))
  _pow2_sum(R, (p, k), (convert(R, d), 0))
end
# Σ cᵢ (xᵢ 2^-k), over the stored entries of x where it has them; other array
# types (and device arrays) take the scaled copy of x with their own dot
_linear_scaled_dot(c, x::Union{DenseArray, SparseIterate}, s) =
  _is_device_array(x) || _is_device_array(c) ? _linear_scaled_copy_dot(c, x, s) : _scaled_dot(c, x, s)
# a symmetric iterate counts an off-diagonal entry for both of its positions:
# each position is a separate scaled product, so the doubled coefficient is
# never formed
function _linear_scaled_dot(c, x::SymmetricSparseIterate, s)
  v = zero(s[1])
  @inbounds for k in 1:x.nnz
    r, j = tril_cartesian(x.p, x.nzval2ind[k])
    q = float(_entry(c, (j - 1) * x.p + r)) * _scale_value(x.nzval[k], s)
    v += q
    r == j || (v += q)
  end
  v
end
_linear_scaled_dot(c, x, s) = _linear_scaled_copy_dot(c, x, s)
_linear_scaled_copy_dot(c, x, s) = dot(c, x .* s[1] .* s[2])

# A matrix SparseIterate (D1): an entry of x that is not stored, or is stored as a zero, adds
# nothing to the value, also if the coefficient there is NaN or Inf. The value of x is read from
# the storage (nzval, ind2nzval), not through getindex: a scan of the logical matrix that visits
# the implicit entries forms a zero at each of them. size(c) == size(x) is checked first.
function _value_unchecked(f::LinearFunction, x::SparseIterate{<:Any, 2})
  _checkvalue(f, x)
  _value_at(_linear_value(f, x, _linear_stored_dot(f.c, x) + f.d), x)
end

# The sum over the stored entries, traversed by their storage index (the reduction is @simd, so its
# order is not specified). The main loop has no branch on the value of an entry (a data-dependent
# branch on stored zeros that are interleaved with nonzeros can mispredict; the search for the first
# nonzero entry below does branch): a zero entry takes the coefficient index of the first stored
# nonzero entry, i0, and its product is replaced by a zero by a select. So c is read at the indices of
# nonzero stored entries and, for a zero entry, again at i0 (the coefficient at the zero entry's own
# index is not read), and a NaN or Inf coefficient there cannot reach the sum. Without a nonzero
# stored entry the sum is 0. In benchmark/linear_sparse_value.jl at this change (AMD Zen 3, Julia
# 1.13.3, Float32 and Float64, m = 128 and 1024), fully stored iterates in sequential or shuffled
# storage order with 0, 50% and 90% stored zeros took 0.45 to 0.94 times the time of the base's
# generic dot(c, x) over the matrix SparseIterate. In quick paired probes, a traversal in column
# order through ind2nzval was slower than this loop in every cell measured (iterates of 2% to 100%
# stored entries).
function _linear_stored_dot(c, x::SparseIterate{<:Any, 2})
  T = promote_type(eltype(c), eltype(x))
  s, z = zero(T), zero(T)
  k0 = _first_nonzero(x)
  k0 == 0 && return s
  i0 = x.nzval2ind[k0]
  @inbounds @simd for k in 1:x.nnz
    v = x.nzval[k]
    skip = iszero(v)
    t = _entry(c, ifelse(skip, i0, x.nzval2ind[k])) * v
    s += ifelse(skip, z, t)
  end
  s
end
# the position in nzval of the first nonzero stored entry, 0 if there is none
function _first_nonzero(x::SparseIterate{<:Any, 2})
  @inbounds for k in 1:x.nnz
    iszero(x.nzval[k]) || return k
  end
  0
end

# The recomputation for a matrix SparseIterate, when the ordinary value is not finite. The
# participating entries are the stored entries of x that are nonzero. If d and those entries of
# x and of c are finite, the products are summed as exponent-separated terms (_ExpSum), a product
# from the significands and the sum of the exponents of its factors, so that neither factor is
# scaled before the other acts, and the sum is combined with d by `_linear_recovered_value`
# below. Zero entries are skipped before c is read, so c is read at participating entries only.
# Otherwise the ordinary value v stands: an Inf or a NaN in the data, not in a product, is a
# genuine result. A coefficient at a zero of x is not read, so it does not stop the recomputation.
@noinline function _linear_value_recover(c, d, x::SparseIterate{<:Any, 2}, v)
  isfinite(d) || return v
  R = typeof(v)
  acc = _expsum_start(promote_type(R, Float64))
  @inbounds for k in 1:x.nnz
    xk = x.nzval[k]
    iszero(xk) && continue
    ck = _entry(c, x.nzval2ind[k])
    (isfinite(xk) && isfinite(ck)) || return v
    acc = _expsum_product(acc, ck, xk)
  end
  _linear_recovered_value(R, acc, d)
end

# The value of a recovery: the dot kept by `_expsum_product` as the compensated pair (s, c) in the
# frame 2^ex, combined with d. d is not added to the pair itself: after an exactly cancelling dot
# the pair can hold nonzero opposite parts s = -c (the compensation of absorbed terms), and a small
# d added to it is absorbed in their rounding. The pair is collapsed first, and the dot is rebased
# to its own exponent, so that d meets the dot's value and not its largest product. A is the type
# of the accumulator, at least Float64, and R the type of the value.
#   Range. Every term added since the last restart of the pair is m 2^e with |m| < 1 and e <= ex
# (a pair that reaches exactly (0, 0) restarts in the frame of its next term, which can be a
# smaller one), so in the units of the frame each term is below 1, and |s|, |c| and m = s + c are
# below 2n + 4 for n terms (|c| collects the roundings of the partial sums, which are of the order
# of eps(A) times them), with n < 2^63. The collapse is made in A, where it cannot overflow, and the
# dot alone is not converted to R.
#   Steps. (1) m = fl_A(s + c): one rounding, |m - (s + c)| <= eps(A)/2 |m|, and m == 0 only if
# s + c == 0 exactly. The remainder r = (s + c) - m is computed exactly (a TwoSum of s and c), and
# |r| <= eps(A)/2 |m|. (2) m = mm 2^em with 1/2 <= |mm| < 1, and the dot goes into a new
# accumulator in the frame ex + em: only an exponent changes, so this step is exact. (3) d enters
# by `_expsum_product` as a term of its own, then r as one more term in the frame ex: the sum
# is taken in the frame of the largest of them, and when d cancels the dot, r is what remains of
# the pair. (4) `_expsum_value` rounds the pair to A and converts that to R. Overflow follows
# this rounded value: a value just above floatmax(R) rounds to floatmax(R), and the result is Inf
# only past floatmax(R) plus half an ulp; for Float32 data (A = Float64, R = Float32) it is a
# double rounding, which can differ from the correctly rounded result at that threshold. The
# caller then converts the value to float(eltype(x)): for a Float32 iterate with Float64
# coefficients (R = Float64) that is one more rounding, and one more point of overflow.
#   Dropped terms. `_expsum_add` loses a term, or a part of the sum, below 2^-1074 of the frame it
# adds to (the subnormal limit of A). With E the largest frame exponent the pair ever held, the
# loss is at most n 2^(E-1074) in all, at most n 2^-1072 times the largest |product|; it can be a
# whole term, when the pair restarts in a frame more than 1074 binades below an earlier one.
#   Error of the pair. A product of Float64 or mixed factors is rounded once (eps(A)/2 |p|); of
# Float32 factors it is exact in A. The compensation c is a recursive (Neumaier, Sum2) sum of exact
# errors, so the error of the pair grows like n^2 eps(A)^2 / 4 times the sum of the magnitudes,
# not n eps(A)^2. With the drops above, for n < 2^50 (any n for Float32 data) the value is within
# 4 n eps(R) (sum |cᵢ xᵢ| + |d|) of the exact one whenever it is finite. That bound can exceed
# floatmax(R), and then it constrains nothing: R1 does not guarantee a finite value there, and
# the result can be an infinity for a finite exact value (see #907).
#   Guaranteed. R1 where the bound is below the range, as above. If the pair sums to zero, the
# value is R(d) when d is representable in R (d is a term of its own, converted once), and +0 if
# d is zero. A dot of value delta beside d, when the products are exact in A (for example Float32
# data, or significands of at most 26 bits), delta is above n 2^(E-1074) / eps(R), and
# n^2 eps(A)^2 / 4 times the sum of the magnitudes is below eps(R) |delta|, is returned with an
# error of at most a few eps(R) (|delta| + |d|).
#   Not guaranteed. (1) That terms which cancel exactly in real arithmetic cancel: Float64 and
# mixed products are rounded, so the pair can keep a residual and a small d is lost against it.
# (2) Terms more than 1074 binades below the frame are dropped, and after a restart this can
# leave one whole term as the result. (3) That the pair sums to zero for an exactly cancelling
# dot: the compensation is a rounded sum, and it fails for terms of very different sizes (it
# depends on the spread of the exponents and on the number of terms, and not on the spread
# alone); the value then has the absolute error of R1. Relative accuracy when d cancels the dot
# is limited by the error of the pair and the drops above only, since the collapse remainder is
# kept. (4) A finite value when the products lie far beyond the range (about 2^1077 for Float64,
# 2^150 for Float32 data): the error of the pair, in the units of the products, can exceed the
# range, and the result is then an infinity.
function _linear_recovered_value(::Type{R}, acc::_ExpSum{A}, d) where {R, A}
  s, c = acc.s, acc.c
  m = s + c
  collapsed = _expsum_start(A)
  if !iszero(m)
    mm, em = _frexp(m)
    collapsed = _expsum_add(collapsed, mm, _exponent_sum(A, acc.ex, em))
  end
  collapsed = _expsum_product(collapsed, d, one(R))
  b = m - s
  r = (s - (m - b)) + (c - b)           # the remainder (s + c) - m, exactly
  if !iszero(r)
    mr, er = _frexp(r)
    collapsed = _expsum_add(collapsed, mr, _exponent_sum(A, acc.ex, er))
  end
  _expsum_value(R, collapsed)
end

# A Diagonal coefficient at a matrix SparseIterate: dot(c, x) for a Diagonal c reads the
# diagonal of x only, in O(p), and ignores the other entries of x, as f(Matrix(x)) does. The
# value keeps that, and reads c.diag[j] only where x has a nonzero stored diagonal entry.
function _value_unchecked(f::LinearFunction{<:Diagonal}, x::SparseIterate{<:Any, 2})
  _checkvalue(f, x)
  _value_at(_linear_value(f, x, _linear_diagonal_dot(f.c.diag, x) + f.d), x)
end
# The diagonal positions are visited in increasing index, with a branch on an absent or zero entry.
# A branch-free select like that of the general matrix was slower than this loop in quick paired
# probes (Float32 and Float64, p = 128 and 1024, AMD Zen 3, Julia 1.13.3). In
# benchmark/linear_sparse_value.jl at this change, fully stored Diagonal cells took 0.61 to 0.94
# times the base's dot(c, x) for Float32, and the Float64 cells were inconclusive: the forward
# and reversed-role runs disagreed.
function _linear_diagonal_dot(dg, x::SparseIterate{<:Any, 2})
  s = zero(promote_type(eltype(dg), eltype(x)))
  @inbounds for j in 1:size(x, 1)
    pos = x.ind2nzval[j, j]
    pos == 0 && continue
    v = x.nzval[pos]
    iszero(v) && continue
    s += dg[j] * v
  end
  s
end
# The recomputation for a Diagonal coefficient: the participating entries are the nonzero stored
# diagonal entries of x; the other entries of x are not read, so a NaN there does not stop it. As
# for the general method, the products are exponent-separated terms and the sum is combined with d
# by `_linear_recovered_value`.
@noinline function _linear_value_recover(c::Diagonal, d, x::SparseIterate{<:Any, 2}, v)
  isfinite(d) || return v
  R = typeof(v)
  acc = _expsum_start(promote_type(R, Float64))
  @inbounds for j in 1:size(x, 1)
    pos = x.ind2nzval[j, j]
    pos == 0 && continue
    xj = x.nzval[pos]
    iszero(xj) && continue
    cj = c.diag[j]
    (isfinite(xj) && isfinite(cj)) || return v
    acc = _expsum_product(acc, cj, xj)
  end
  _linear_recovered_value(R, acc, d)
end

# c and x have the same dimensions, size(c) == size(x), the size of a symmetric
# iterate being that of the p × p matrix it represents. Every oracle applies this
# rule before it reads or writes, whatever dot would accept for the array types.
_checkparams(f::LinearFunction, x) = size(f.c) == size(x) || _coefficient_size_mismatch(size(f.c), size(x))
@noinline _coefficient_size_mismatch(sc, sx) =
  throw(DimensionMismatch("the coefficients c have size $sc and the argument has size $sx"))

function _checkwrite(op::O, f::LinearFunction, y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate) where {O}
  _checkreal(f, x)
  _checkreal(f, y)
  (op === gradient! || op === value_and_gradient!) && _checkgradient_output(y, x)
  _checksize(y, x)
  _checkparams(f, x)
  # c is written into y: copied by the gradient, subtracted by the prox
  _check_parameter_axes(y, f.c)
  true
end

prox!(y::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate, γ::Real) =
  (x = _symmetric_argument(y, x); _check_oracle_arrays(prox!, f, y, x, γ); _prox_unchecked!(y, f, x, γ))
function _prox_unchecked!(y::AbstractArray, f::LinearFunction, x::AbstractArray, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  y .= x .- γ .* _assumed_symmetric(y, f.c)
  y
end
# a symmetric iterate point and a dense output: the entries of the matrix that the symmetric iterate represents
_prox_unchecked!(y::AbstractArray, f::LinearFunction, x::SymmetricSparseIterate, γ::Real) =
  _prox_unchecked!(y, f, matrixview(x), γ)
# a symmetric iterate output: each coordinate once, from the lower entry of the symmetric argument
function _prox_unchecked!(y::SymmetricSparseIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate, γ::Real)
  _checkwrite(prox!, f, y, x)
  γ = convert(float(eltype(y)), γ)
  p = y.p
  @inbounds for c in 1:p, r in c:p
    _setstored!(y, x[r, c] - γ * f.c[(c - 1) * p + r], tril_index(p, r, c))
  end
  y
end

value_and_gradient!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
function _value_and_gradient_unchecked!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)
  _copy_parameter!(g, f.c)
  v
end
gradient!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
_gradient_unchecked!(g::_ArrayOrSymmetricIterate, f::LinearFunction, x::_ArrayOrSymmetricIterate) =
  (_checkwrite(gradient!, f, g, x); _copy_parameter!(g, f.c))

Base.@propagate_inbounds prox_block(f::LinearFunction, v::Real, i::Integer, γ::Real) = v - γ * f.c[i]
kkt_violation(f::LinearFunction, g::AbstractArray, i::Integer) = (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] + f.c[i]))
restrict(f::LinearFunction, S::AbstractVector{<:Integer}) = LinearFunction(f.c[S], f.d)


lipschitz(f::LinearFunction) = zero(float(eltype(f.c)))
