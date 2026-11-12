####################################
#  ℓ1 norm
####################################

"""
    NormL1(λ0 = 1)
    NormL1(λ0, w)

The weighted ℓ1 norm `f(x) = λ0 Σᵢ wᵢ |xᵢ|`, with all `wᵢ = 1` when `w` is
omitted. Its proximal operator is entry-wise soft thresholding
(`shrink`). Separable, so `prox_block` computes any scalar
coordinate, and on a sparse iterate the value and the proximal operator touch
only the stored entries. On a `SymmetricSparseIterate` the weights are a
`p × p` matrix (or anything indexed by the linear indices of one), as for a
dense matrix. When symmetry is required, mirrored weights must agree at each
coordinate the operation reads. Wrap a matrix in `Symmetric(W, :L)` to use its
lower triangle.
Values follow the overflow rule in Overflow and underflow of values; terms that
round below `floatmin` are recomputed where they could change the value by more
than ordinary rounding (issue #373).
An infinite weight maps a **finite input value** to zero at every level and
step, including a zero level or step (the value is `Inf` away from zero).
For inputs of `NaN` or `±Inf` at finite level there is **no fixed-zero
guarantee**: ordinary soft thresholding may return `NaN` (issue #853).
A zero weight remains unpenalized. An infinite overall level `λ0` at any
nonnegative step instead fixes every positive-weight entry at `+0`, even for
non-finite inputs (historical decision 49). At `γ = 0` this is the limit
`γ → 0+` of the proximal map of the fixed function, the projection onto the
closure of its domain, not soft thresholding by `0⋅Inf` (issue #665):
`prox(NormL1(Inf, [1, 0, 2]), [3.0, 4.0, 5.0], 0.0) == [0.0, 4.0, 0.0]`.
Zero-weight entries stay free. A threshold that overflows to `Inf` from
finite level, step and weight factors is not an infinite model parameter;
its behavior for `±Inf` inputs is a deferred limitation (issue #866).
The value is `0` at the zero vector and remains `NaN` at a `NaN` entry.

See Symmetric parameters for which arrays must be symmetric and which calls check them.
"""
struct NormL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ0::T
  w::W

  function NormL1{T, W}(λ0::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    w === nothing || eltype(w) <: Real || _real_error(:NormL1)
    _checknonneg("λ0", λ0)
    _checknonneg("the weights", w)
    new{T, W}(λ0, w)
  end
end
NormL1(λ0::Real = 1, w::Union{Nothing, AbstractArray} = nothing) = NormL1{typeof(λ0), typeof(w)}(λ0, w)

isconvex(::Type{<:NormL1}) = true
iszeropreserving(::NormL1) = true
_checkparams(f::NormL1, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::NormL1, v::Real, i::Integer)
  a = abs(float(v))
  # an absent weight is one in the entry's type: the Bool `true` of `_entry` would widen to Float64 (#684)
  w = f.w === nothing ? one(a) : _entry(f.w, i)
  p = _positive_product(f.λ0, w, a)
  ifelse(isinf(w) & !iszero(a), oftype(p, Inf) * a, p)
end
Base.@propagate_inbounds prox_block(f::NormL1, v::T, i::Integer, γ::Real) where {T<:Real} =
  first(_l1_prox_threshold(v, γ, f.λ0, _entry(f.w, i), float(T)))

# Whether the level and the step, as given, fix a penalized entry (w > 0) at zero whatever it holds,
# NaN and ±Inf included (E4 before E2): an infinite level at a nonnegative step, where the domain of
# the entry is {0}. At γ = 0 this is the limit γ → 0+ with the function fixed (#665), the projection
# onto the closure of the domain, not the identity that the threshold 0⋅Inf would suggest. Never
# decided from the threshold γλ0w, which overflows for finite parameters (R3) and is 0⋅Inf at a
# zero step. Only the exceptional branch below reads it: 0⋅Inf is NaN, outside the ordinary range.
@inline _l1_fixing(γ, λ) = (γ >= 0) & isinf(λ)
@inline _l1_fixed(γ, λ, w) = (w > 0) & _l1_fixing(γ, λ)

# The scalar proximal operator and its threshold γλ0w, as _weight_product forms it. In the ordinary
# range of γλ0, where the step and the level are finite and positive, no entry is fixed, and the loop
# keeps the bare soft thresholding.
Base.@propagate_inbounds function _l1_prox_threshold(v, γ, λ, w, ::Type{S}) where {S}
  R = _working_type(S, γ, λ, w)
  ab = convert(R, γ) * convert(R, λ)
  if floatmin(R) <= ab <= floatmax(R)
    t = convert(S, ab * convert(R, w))
    return shrink(v, t), t
  end
  t = _weight_product(S, γ, λ, w)
  r = shrink(v, t)
  ifelse(_l1_fixed(γ, λ, w), zero(r), r), t
end

kkt_violation(f::NormL1, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]) - _kkt_coefficient(eltype(g), f.λ0, _entry(f.w, i)))
restrict(f::NormL1, S::AbstractVector{<:Integer}) = NormL1(f.λ0, _restrict_entries(f.w, S))

# without weights, Base's vectorized sum (a sum with init takes a slow sequential path)
function (f::NormL1{<:Real, Nothing})(x::DenseArray{<:AbstractFloat})
  T = float(eltype(x))
  iszero(f.λ0) && return zero(T)
  _level_fits(T, f.λ0) || return _converted_l1_value(f.λ0, x)
  s = sum(abs, x)
  # an infinite level fixes every entry at zero (E4): the value at the zero vector is 0, not Inf⋅0
  isinf(f.λ0) && iszero(s) && return zero(T)
  v = convert(T, f.λ0) * s
  isfinite(v) || !isfinite(f.λ0) || !all(isfinite, x) ? v : _value_dense(f, x)
end

# A level outside the argument's precision is applied before final conversion.
@noinline function _converted_l1_value(λ, x)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)))
  m = _maximum_abs(x)
  isfinite(m) || return convert(T, convert(R, λ) * sum(abs, x))
  e = _magnitude_exponent(m)
  s = _scaled_value_sum(x, _power_factors(R, -e), abs)
  convert(T, _scaled_product(convert(R, λ), s, e))
end

@inline _l1_value(::NormL1, v, a) = a * abs(v)

@inline function _l1_value_sum(f, x::Array{T}, allow_zero::Bool) where {T<:Union{Float32,Float64}}
  s = zero(T)
  ordinary = true
  @inbounds @simd for i in eachindex(IndexLinear(), x)
    w = f.w[i]
    a = f.λ0 * w
    # Test the factors: a rounded zero may conceal recoverable underflow.
    z = allow_zero & (iszero(f.λ0) | iszero(w))
    s += _l1_value(f, x[i], a)
    ordinary &= z | (floatmin(T) <= a <= floatmax(T))
  end
  s, ordinary
end

# Preserve the original ordinary loop; the checks of its result are scalar tests after it. Every pass
# checks intermediates, not just the final sum.
function _value_dense(
    f::NormL1{T,<:Array{T}},
    x::Array{T}) where {T<:Union{Float32,Float64}}
  s, ordinary = _l1_value_sum(f, x, false)
  ordinary && _l1_sum_kept(s, length(x)) && return s
  _l1_exceptional_value(f, x, s, ordinary)
end

# The cold paths of the loop above: a coefficient out of range or zero is retried with SIMD exempting zero
# factors, and 0*Inf/NaN takes the entrywise zero-factor and domain priorities of value_entry; then the sum
# is checked for products that rounded below floatmin.
@noinline function _l1_exceptional_value(f, x, s, ordinary)
  if !ordinary
    s, ordinary = _l1_value_sum(f, x, true)
    ordinary && !isnan(s) || (s = invoke(_value_dense, Tuple{Any,AbstractArray}, f, x))
  end
  _l1_underflow_checked(f, x, s, length(x))
end

# Products that round below floatmin (#373). The value is a sum of n terms λ0 wᵢ |xᵢ| ≥ 0, each rounded
# once or twice (the coefficient, then the product with |xᵢ|); a coefficient out of the normal range is
# formed by value_entry with one rounding, so the ordinary flag and these exact products leave only the
# rounding of a term into the subnormal range (or to zero). That rounding loses at most δ/2 =
# eps(T) floatmin(T)/2 per term, n eps floatmin / 2 in all, which is at most eps/2 of a sum s ≥ n floatmin.
# A sum below that bound (a zero sum included) may have lost more than ordinary rounding, even a normal one:
# 512 terms 2^-135 + 2^-150 in Float32 each round to 2^-135 and the sum loses 256 units of 2^-126. Sums of
# subnormal partial sums are exact, and the relative errors of normal terms and of the summation are those
# of every ordinary value. The test is false for NaN, so an ordinary NaN (or Inf) is returned as it was.
@inline _l1_sum_kept(s, n) = !(s < n * floatmin(typeof(s)))

# The value s of f at x from its ordinary loop, a sum of at most n rounded terms, recomputed where those
# terms may have lost more than ordinary rounding (above); only Float32 and Float64 values are checked.
@inline function _l1_underflow_checked(f::NormL1, x, s, n::Integer)
  _underflow_checked(typeof(s)) || return s
  _l1_sum_kept(s, n) ? s : _l1_underflow_value(f, x, s)
end

# Σ wᵢ |xᵢ| is accumulated with every term's exponent (_ExpSum, products of significands that are exact for
# Float32 data in Float64) and λ0 is applied to its significand and exponent before the one conversion to
# the type of the value: a sum rounded to that type first would underflow (or overflow) where λ0 brings it
# back. The recomputation covers finite positive levels and a finite s. Then every term that value_entry
# makes +0 has a zero factor, which the accumulator skips (a zero weight with a NaN or ±Inf entry, an
# infinite weight at a zero entry), and every other term has finite factors: an infinite weight at a
# nonzero entry, or a positive weight at a NaN or ±Inf entry, makes s Inf or NaN. A zero level (an exact
# zero, or the Inf of an infinite weight) and an infinite level (its domain rule) keep s, and so does an
# exact zero, which a scan of the stored values (`_all_zero`) decides before the recomputation. The result is
# in the type of s also where the accumulator type is known only at run time (abstract factor arrays), so that
# the callers stay inferred.
@noinline function _l1_underflow_value(f::NormL1, x, s::T)::T where {T}
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return s
  iszero(s) && _all_zero(_stored_values(x)) && return s
  R = promote_type(T, Float64, _factor_type(λ), _l1_factor_type(f.w), _l1_factor_type(_stored_values(x)))
  _expsum_value(T, _l1_weighted_abs(_expsum_start(R), f.w, x), λ)
end

# The floating type that holds every entry of a factor array without narrowing it: from the element type, or,
# for an abstract one (Real[...], AbstractFloat[...]), from the entries themselves, so that the accumulator is
# concrete and a wider entry (a BigFloat among Float64s) keeps its range.
_l1_factor_type(::Nothing) = Float16
_l1_factor_type(a::AbstractArray) =
  isconcretetype(eltype(a)) ? float(eltype(a)) : mapreduce(v -> typeof(float(v)), promote_type, a; init = Float16)

# Σ wᵢ |xᵢ| (wᵢ = 1 without weights) over the entries that the value reads, as the generic separable
# kernels index them: every entry of a dense array, the stored entries of a sparse iterate, the active
# groups of a block iterate, and each stored coordinate of a symmetric iterate at both of its positions.
function _l1_weighted_abs(acc, w, x::AbstractArray)
  @inbounds for i in eachindex(IndexLinear(), x)
    acc = _expsum_product(acc, _entry(w, i), abs(float(x[i])))
  end
  acc
end
function _l1_weighted_abs(acc, w, x::SparseIterate)
  @inbounds for k in 1:x.nnz
    acc = _expsum_product(acc, _entry(w, x.nzval2ind[k]), abs(float(x.nzval[k])))
  end
  acc
end
function _l1_weighted_abs(acc, w, x::BlockIterate)
  @inbounds for a in 1:x.nactive, i in x.groups[x.active[a]]
    acc = _expsum_product(acc, _entry(w, i), abs(float(x.storage[i])))
  end
  acc
end
function _l1_weighted_abs(acc, w, x::SymmetricSparseIterate)
  p = x.p
  @inbounds for k in 1:x.nnz
    r, c = tril_cartesian(p, x.nzval2ind[k])
    v = abs(float(x.nzval[k]))
    acc = _expsum_product(acc, _entry(w, (c - 1) * p + r), v)
    r == c || (acc = _expsum_product(acc, _entry(w, (r - 1) * p + c), v))
  end
  acc
end

# The other representations keep the generic loops of value_entry and add the same check of their sum:
# dense arrays other than the loop above (views, mixed precision, unweighted arrays that are not dense),
# n entries; sparse iterates, nnz terms; block iterates, at most the length of the storage; symmetric
# iterates, at most two terms per stored coordinate.
_value_dense(f::NormL1, x::AbstractArray) =
  _l1_underflow_checked(f, x, invoke(_value_dense, Tuple{Any, AbstractArray}, f, x), length(x))
_value_sep(f::NormL1, x::SparseIterate) =
  _l1_underflow_checked(f, x, invoke(_value_sep, Tuple{Any, SparseIterate}, f, x), x.nnz)
_value_sep(f::NormL1, x::BlockIterate) =
  _l1_underflow_checked(f, x, invoke(_value_sep, Tuple{Any, BlockIterate}, f, x), length(x.storage))
_value_sep(f::NormL1, x::SymmetricSparseIterate) =
  _l1_underflow_checked(f, x, invoke(_value_sep, Tuple{Any, SymmetricSparseIterate}, f, x), 2 * x.nnz)
