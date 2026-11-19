####################################
#  ℓ1 norm
####################################

"""
    NormL1(λ0 = 1)
    NormL1(λ0, w)

Weighted ℓ₁ penalty `f(x) = λ₀ Σᵢ wᵢ |xᵢ|`, with all weights equal to
one when `w` is omitted. The level and weights must be nonnegative and not `NaN`; these are checked
at construction. Weights are retained by reference and paired with entries
by linear index; later array mutations are not revalidated.

[`prox`](@ref) applies entrywise soft thresholding ([`shrink`](@ref)) with
threshold `γλ₀wᵢ`. The function is separable and zero-preserving; value
and same-kind sparse-iterate prox calls visit only stored entries.

# Examples
```jldoctest
julia> f = NormL1(0.5, [1.0, 2.0]);

julia> f([2.0, -1.0])
2.0

julia> prox(f, [2.0, -1.0], 1.0)
2-element Vector{Float64}:
 1.5
 0.0
```

# Edge cases
- Use real, one-based arrays and one weight per entry.
- On a [`SymmetricSparseIterate`](@ref), weights have `p²` entries in full-matrix
  linear order, interpreted as an exactly symmetric `p × p` array. Oracles assume this; validate with
  [`ProximalBase.check_symmetric_parameters`](@ref).
- A zero weight leaves an entry unpenalized. An infinite weight fixes a **finite**
  input at zero, even at zero level or step, but can return `NaN` for non-finite
  inputs at finite level.
- An infinite overall level fixes positive-weight entries at `+0`, even for
  non-finite inputs and zero step; zero-weight entries remain free.

See [ℓ₁ endpoint rules](@ref l1-endpoints), the
[numerical contract](@ref numerics) and [`prox!`](@ref) for step and alias rules.
"""
struct NormL1{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  λ0::T
  w::W

  function NormL1{T, W}(λ0::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    _check_one_based(w)
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
# NaN and ±Inf included (numerics.md, "NaN, infinities and zeros"): an infinite level at a nonnegative
# step, where the domain of the entry is {0}. At γ = 0 this is the limit γ → 0+ with the function fixed
# (#665), the projection onto the closure of the domain, not the identity that the threshold 0⋅Inf would
# suggest. Never decided from the threshold γλ0w, which overflows for finite parameters and is 0⋅Inf at a
# zero step. Only the exceptional branch below reads it: 0⋅Inf is NaN, outside the ordinary range.
@inline _l1_fixing(γ, λ) = (γ >= 0) & isinf(λ)
@inline _l1_fixed(γ, λ, w) = (w > 0) & _l1_fixing(γ, λ)

# The scalar proximal operator and its threshold γλ0w, as _weight_product forms it. In the ordinary
# range of γλ0, where the step and the level are finite and positive, no entry is fixed, and the loop
# keeps the bare soft thresholding, `soft`: _shrink_nonneg, or shrink in the loop over a long array
# (_l1_prox_long!). The two agree for every threshold that is not negative.
Base.@propagate_inbounds function _l1_prox_threshold(v, γ, λ, w, ::Type{S}, soft::K = _shrink_nonneg) where {S, K}
  t, normal = _threshold_product(S, γ, λ, w)
  if normal
    return soft(v, t), t          # γλ0 > 0 and w ≥ 0: the threshold is not negative
  end
  t = _weight_product(S, γ, λ, w)
  r = shrink(v, t)
  ifelse(_l1_fixed(γ, λ, w), zero(r), r), t
end

# Array steps in the loop over every entry (separable.jl), as for ElasticNet (elasticnet.jl, #910). With a step
# array the range test of γᵢλ0 (_threshold_product) is a test per entry, which keeps the loop scalar. When the
# element types of x and γ are concrete and every step takes the ordinary branch of _l1_prox_threshold, which a pass
# over the steps in blocks of 256 decides (it returns false after the first block that holds a step outside it), the
# loop calls that branch alone, and LLVM can vectorize it. Otherwise it calls prox_block(f, …). Either loop stores
# convert(eltype(y), prox_block(f, xᵢ, i, γᵢ)) in entry i, bit for bit.
struct _L1OrdinarySteps{F<:NormL1} <: SeparableFunction
  f::F
end
Base.@propagate_inbounds prox_block(g::_L1OrdinarySteps, v::T, i::Integer, γ::Real) where {T<:Real} =
  shrink(v, first(_threshold_product(float(T), γ, g.f.λ0, _entry(g.f.w, i))))
iszeropreserving(::_L1OrdinarySteps) = true
_checkparams(g::_L1OrdinarySteps, x) = _checkparams(g.f, x)

# Arrays of fewer than _L1_STEP_PASS_MIN entries skip the pass over the steps and call the generic loop of
# separable.jl, which makes the checks: for a few entries the pass costs more than the ordinary loop saves. This
# method is only the test of the length, inlined into its caller, so that a short array pays one comparison before the
# call of the generic loop; the checks, the pass and the loop of the ordinary branch are in _l1_prox_steps!, out of
# line. The length of x is read before the checks: an array type whose length throws, or changes between calls, is
# outside what the arrays here support.
const _L1_STEP_PASS_MIN = 16
@inline _prox_sep_dense!(y::AbstractArray, f::NormL1, x::AbstractArray, γ::AbstractArray) =
  length(x) < _L1_STEP_PASS_MIN ? _prox_entries!(y, f, x, γ) : _l1_prox_steps!(y, f, x, γ)
@noinline function _l1_prox_steps!(y, f::NormL1, x, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  isconcretetype(eltype(x)) && isconcretetype(eltype(γ)) &&
    _l1_ordinary_steps(f, γ, float(eltype(x)), eachindex(IndexLinear(), y, x)) &&
    return _prox_entries!(y, _L1OrdinarySteps(f), x, γ)
  _prox_entries!(y, f, x, γ)
end
function _l1_ordinary_steps(f::NormL1, γ, ::Type{S}, inds) where {S}
  for b in first(inds):256:last(inds)
    ok = true
    @inbounds for i in b:(b + min(255, last(inds) - b))   # b + 255 can overflow at indices near typemax(Int)
      ok &= last(_threshold_product(S, _entry(γ, i), f.λ0, _entry(f.w, i)))
    end
    ok || return false
  end
  true
end

# A scalar step on a host array (an Array, or a view or a reshape of one) of more than _L1_SELECT_BYTES bytes takes
# shrink on the ordinary branch, and one of up to that size _shrink_nonneg. In a loop over arrays that do not fit the
# first-level cache together, the selects of _shrink_nonneg are slower than shrink for some placements of the output
# relative to the argument (by up to 1.4 times on AMD Zen 3 with huge pages, at 20 KiB per array and more), and
# faster for the others; up to 16 KiB per array they were faster there at every placement that was measured (on an
# Intel Skylake-SP core one placement read 1.06 times at 12 KiB). The two give the same values, signed zeros
# included (numerical_kernels.jl). The length test is inlined and the long loop is a separate method, so that the
# loop over a short array is reached without another call. The iterates and prox_block keep _shrink_nonneg, as does
# the generic loop of an array of steps; the loop of ordinary steps above calls shrink.
const _L1_SELECT_BYTES = 16384
# An output that is the argument has no placement relative to it and keeps _shrink_nonneg at every length, as does
# an array whose element type has no size of its own (an abstract type, BigFloat) or a size of zero.
@inline _l1_select_short(y, x) = y === x || _l1_select_short(x, eltype(x))
@inline _l1_select_short(x, ::Type{T}) where {T} =
  !isbitstype(T) || sizeof(T) == 0 || length(x) <= _L1_SELECT_BYTES ÷ sizeof(T)
struct _L1Shrink{F<:NormL1} <: SeparableFunction
  f::F
end
Base.@propagate_inbounds prox_block(g::_L1Shrink, v::T, i::Integer, γ::Real) where {T<:Real} =
  first(_l1_prox_threshold(v, γ, g.f.λ0, _entry(g.f.w, i), float(T), shrink))
_checkwrite(p::Union{typeof(prox!), typeof(value_and_prox!)}, g::_L1Shrink, y::_ArrayOrSymmetricIterate, x::_ArrayOrSymmetricIterate) =
  _checkwrite(p, g.f, y, x)
@inline _prox_sep!(y::_HostArray, f::NormL1, x::_HostArray, γ::Real) =
  _l1_select_short(y, x) ? _prox_sep_dense!(y, f, x, γ) : _l1_prox_long!(y, f, x, γ)
@noinline _l1_prox_long!(y, f::NormL1, x, γ) = _prox_sep_dense!(y, _L1Shrink(f), x, γ)

kkt_violation(f::NormL1, g::AbstractArray, i::Integer) =
  (_check_one_based(g); eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i]) - _kkt_coefficient(eltype(g), f.λ0, _entry(f.w, i)))

# The point-aware score (see kkt_violation): the coefficient cᵢ = λ0 wᵢ formed by _kkt_coefficient(W, ...) and rounded
# once to the floating working type W (the screen above does the same with the element type of g when that type is
# floating, and otherwise keeps the coefficient in the types of the level and the weight), then |vᵢ| - cᵢ at a zero
# entry and |vᵢ + cᵢ sign(xᵢ)| at a nonzero one. An unweighted function forms the coefficient once per call.
@inline _kkt_state(f::NormL1{<:Real, Nothing}, ::Type{W}) where {W} = _kkt_coefficient(W, f.λ0, true)
@inline _kkt_state(::NormL1, ::Type) = nothing
# The coefficient of entry i within a call: the value that _kkt_state computed once for an unweighted function is
# returned as it is, so that it is not formed again for each entry; a weighted function forms λ0 wᵢ here with
# _kkt_coefficient.
Base.@propagate_inbounds _kkt_entry_coefficient(f::NormL1, ::Nothing, ::Type{W}, i) where {W} =
  _kkt_coefficient(W, f.λ0, _entry(f.w, i))
@inline _kkt_entry_coefficient(::NormL1, c, ::Type, i) = c
Base.@propagate_inbounds function _kkt_natural(f::NormL1, s, ::Type{W}, v::W, x::W, i) where {W}
  c = _kkt_entry_coefficient(f, s, W, i)
  ifelse(iszero(x), abs(v) - c, abs(v + c * sign(x)))
end
@_kkt_entrywise NormL1

restrict(f::NormL1, S::AbstractVector{<:Integer}) = NormL1(f.λ0, _restrict_entries(f.w, S))

# without weights: one loop for a few hundred entries (_value_sumabs) and Base's vectorized sum otherwise (a sum with
# init takes a slow sequential path)
function (f::NormL1{<:Real, Nothing})(x::DenseArray{<:AbstractFloat})
  _check_one_based(x)
  T = float(eltype(x))
  iszero(f.λ0) && return zero(T)
  _level_fits(T, f.λ0) || return _converted_l1_value(f.λ0, x)
  s = _value_sumabs(x)
  # an infinite level fixes every entry at zero: the value at the zero vector is 0, not Inf⋅0
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
  ordinary && _sum_kept(s, length(x)) && return s
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

# The value s of f at x from its ordinary loop, a sum of at most n rounded terms, recomputed where those
# terms may have lost more than ordinary rounding (#373, see _sum_kept in numerical_kernels.jl); only
# Float32 and Float64 values are checked.
@inline function _l1_underflow_checked(f::NormL1, x, s, n::Integer)
  _underflow_checked(typeof(s)) || return s
  _sum_kept(s, n) ? s : _l1_underflow_value(f, x, s)
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
@inline _l1_underflow_value(f::NormL1, x, s::T) where {T} = first(_l1_underflow_result(f, x, s))::T

# The flag records the exact-zero test for callers that will scale the result.
@noinline function _l1_underflow_result(f::NormL1, x, s::T)::Tuple{T, Bool} where {T}
  λ = f.λ0
  (λ > 0 && isfinite(λ)) || return s, false
  iszero(s) && _all_zero(_stored_values(x)) && return s, true
  R = promote_type(T, Float64, _factor_type(λ), _l1_factor_type(f.w), _l1_factor_type(_stored_values(x)))
  v = _expsum_value(T, _l1_weighted_abs(_expsum_start(R), f.w, x), λ)::T
  v, false
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
