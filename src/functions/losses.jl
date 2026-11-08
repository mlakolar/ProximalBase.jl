####################################
#
#  Losses of residuals and margins: the check, hinge and ε-insensitive losses
#
####################################

"""
    CheckLoss(τ)
    CheckLoss(τ, w)

The check loss of quantile regression, `f(x) = Σᵢ wᵢ ρ_τ(xᵢ)` with
`ρ_τ(u) = max(τu, (τ - 1)u)` for `τ ∈ (0, 1)`, and all `wᵢ = 1` when `w` is
omitted. Its proximal operator moves an entry above `γτwᵢ` down by `γτwᵢ`, one
below `-γ(1 - τ)wᵢ` up by `γ(1 - τ)wᵢ`, and sets those in between to zero.

It is a function of residuals. Since `ρ_τ(-u) = ρ_{1-τ}(u)`, the loss
`Σᵢ ρ_τ(yᵢ - zᵢ)` of predictions `z` of responses `y` is
`Translate(CheckLoss(1 - τ), y)`, which a splitting method such as ADMM applies
to `z = Xβ`. Its conjugate is the indicator of the box `[wᵢ(τ - 1), wᵢτ]`, the
dual feasible set of quantile regression. For coordinate descent and Newton
steps, [`ProximalBase.SmoothCheck`](@ref) and
[`ProximalBase.ConvolutionCheck`](@ref) are smooth versions of the loss of `Xβ`
for [`GLMLoss`](@ref).
"""
struct CheckLoss{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  τ::T
  w::W

  function CheckLoss{T, W}(τ::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    w === nothing || eltype(w) <: Real || _real_error(:CheckLoss)
    0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1), got $τ"))
    _checknonneg("the weights", w)
    new{T, W}(τ, w)
  end
end
CheckLoss(τ::Real, w::Union{Nothing, AbstractArray} = nothing) = CheckLoss{typeof(τ), typeof(w)}(τ, w)

isconvex(::Type{<:CheckLoss}) = true
iszeropreserving(::CheckLoss) = true
_checkparams(f::CheckLoss, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::CheckLoss, v::T, i::Integer) where {T<:Real}
  τ = convert(float(T), f.τ)
  w = _entry(f.w, i)
  φ = v >= 0 ? τ * v : (τ - 1) * v
  p = w * φ
  iszero(w) ? zero(p) : _weighted_value(p, φ)          # a zero weight leaves the entry free, NaN included
end
Base.@propagate_inbounds function _value_entry_ordinary(f::CheckLoss, v::T, i::Integer) where {T<:Real}
  τ = convert(float(T), f.τ)
  _entry(f.w, i) * (v >= 0 ? τ * v : (τ - 1) * v)
end
_has_ordinary_value(::Type{<:CheckLoss}) = true
Base.@propagate_inbounds function prox_entry(f::CheckLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = _entry(f.w, i)
  u = S(v)
  # the thresholds γwτ and γw(1-τ), each formed in the promotion of S and the factors, rounded once
  # the thresholds γτwᵢ and γ(1-τ)wᵢ as for the other weighted penalties: the step and the level first
  # (invariant in a loop over entries), then the weight; formed in the promotion of S and the factors
  # and rounded once, with the range recovery and the zero and infinite weights of _weight_coefficient
  hi = _weight_coefficient(S, γ, f.τ, w)
  lo = _weight_coefficient(S, γ, _one_minus(_working_type(S, γ, w, f.τ), f.τ), w)
  # a NaN u is repaired in one select: it stays NaN (E2) unless the map is constant, that is for the step
  # γ == Inf with a positive weight, or a hard weight w == Inf (a zero weight is free: E4 first). This is read
  # from the parameters and not from the derived thresholds, which also overflow to Inf for finite
  # parameters. A NaN step (unchecked) keeps the old zero. Weights are nonnegative, so w < Inf is also false
  # for a NaN weight.
  r = ifelse(u > hi, u - hi, ifelse(u < -lo, u + lo, zero(S)))
  ifelse((u != u) & (γ == γ) & ((γ == Inf) ? (w == 0) : (w < Inf)), u, r)
end

# wτ and w(1-τ) of the KKT test in the gradient's element type T, each rounded once; the complement is
# formed in the promotion of T and the factors. An integer gradient keeps the untyped values.
@inline function _check_kkt_levels(::Type{T}, w, τ) where {T<:AbstractFloat}
  R = _working_type(T, w, τ)
  _typed_product(T, w, τ), _typed_product(T, w, _one_minus(R, τ))
end
@inline _check_kkt_levels(::Type, w, τ) = w * τ, w * (1 - τ)

# from zero, the objective decreases upward at rate -gᵢ - wᵢτ and downward at
# rate gᵢ - wᵢ(1 - τ)
function kkt_violation(f::CheckLoss, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  w = _entry(f.w, i)
  hi, lo = _check_kkt_levels(eltype(g), w, f.τ)
  max(-g[i] - hi, g[i] - lo)
end
restrict(f::CheckLoss, S::AbstractVector{<:Integer}) = CheckLoss(f.τ, _restrict_entries(f.w, S))

"""
    HingeLoss()
    HingeLoss(y)
    HingeLoss(y, w)

The hinge loss of support vector machines, `f(x) = Σᵢ wᵢ max(0, 1 - yᵢxᵢ)`, for
nonzero labels `yᵢ`, usually `±1`, and weights `wᵢ`, all 1 when omitted. Without
labels, `x` holds the margins. For the label `1`, its proximal operator moves an
entry below `1 - γwᵢ` up by `γwᵢ`, puts one in between at 1, and keeps one above
1; a label `yᵢ` acts on the margin `yᵢxᵢ`, with the step `γwᵢyᵢ²`. Its conjugate
is `Σᵢ sᵢ/yᵢ` on the box `-wᵢ ≤ sᵢ/yᵢ ≤ 0`, and `Inf` off it.
"""
struct HingeLoss{Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  y::Y
  w::W

  function HingeLoss{Y, W}(y::Y, w::W) where {Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}}
    y === nothing || eltype(y) <: Real || _real_error(:HingeLoss)
    w === nothing || eltype(w) <: Real || _real_error(:HingeLoss)
    y === nothing || all(!iszero, y) || throw(ArgumentError("the labels must be nonzero"))
    _checknonneg("the weights", w)
    new{Y, W}(y, w)
  end
end
HingeLoss(y::Union{Nothing, AbstractArray} = nothing, w::Union{Nothing, AbstractArray} = nothing) =
  HingeLoss{typeof(y), typeof(w)}(y, w)

isconvex(::Type{<:HingeLoss}) = true
_checkparams(f::HingeLoss, x) = (_checkparam(f.y, x); _checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::HingeLoss, v::T, i::Integer) where {T<:Real}
  w = _entry(f.w, i)
  φ = max(1 - _entry(f.y, i) * v, zero(float(T)))
  p = w * φ
  iszero(w) ? zero(p) : _weighted_value(p, φ)          # a zero weight leaves the entry free, NaN included
end
Base.@propagate_inbounds _value_entry_ordinary(f::HingeLoss, v::T, i::Integer) where {T<:Real} =
  _entry(f.w, i) * max(1 - _entry(f.y, i) * v, zero(float(T)))
_has_ordinary_value(::Type{<:HingeLoss}) = true
# The step γw|y|² formed in the promotion R of S and the factors, rounded once to S. The ordinary
# arithmetic, without a call, is (γw·|y|)·|y| in R, and it is the step while γw is a normal number
# (the test is returned with it). Otherwise the recovery of _hinge_step_exceptional decides, zero and
# infinite weights included, on a separate call. For a scalar step and unweighted entries γw is
# invariant, so the test is unswitched out of a loop; an array loop (below) keeps the arithmetic
# vectorized by testing all entries together.
@inline _normal_range(ab::R) where {R<:AbstractFloat} = (floatmin(R) <= ab) & (ab <= floatmax(R))
# The same test for the IEEE types as one unsigned comparison: the bit patterns of the positive normal
# numbers are exactly [bits(floatmin), bits(floatmax)]; zero, subnormals, negatives (sign bit), Inf and
# NaN fall outside, the subtraction wrapping around (checked against the test above for every Float16 and
# Float32 pattern and for random Float64 ones).
_bits(::Type{Float16}) = UInt16
_bits(::Type{Float32}) = UInt32
_bits(::Type{Float64}) = UInt64
@inline function _normal_range(ab::R) where {R<:Union{Float16, Float32, Float64}}
  U = _bits(R)
  lo = reinterpret(U, floatmin(R))
  reinterpret(U, ab) - lo <= reinterpret(U, floatmax(R)) - lo
end
# The label enters with its sign: (ab·y)·y is (ab·|y|)·|y| bit for bit, and saves the abs.
@inline function _hinge_step_ordinary(::Type{S}, γ, y::S, w) where {S<:AbstractFloat}
  R = _working_type(S, γ, w)
  ab = convert(R, γ) * convert(R, w)
  convert(S, (ab * convert(R, y)) * convert(R, y)), _normal_range(ab)
end
@inline function _hinge_step(::Type{S}, γ, y::S, w) where {S<:AbstractFloat}
  s, ok = _hinge_step_ordinary(S, γ, y, w)
  ok ? s : _hinge_step_exceptional(S, γ, abs(y), w)
end
@noinline function _hinge_step_exceptional(::Type{S}, γ, ay, w) where {S<:AbstractFloat}
  R = _working_type(S, γ, w)
  g, v, y = convert(R, γ), convert(R, w), convert(R, ay)
  if !_recoverable_factors(g, v, y)
    isfinite(y) & !iszero(y) || return _weight_coefficient(S, γ, ay, w * ay)   # not a label
    # zero, infinite and NaN parameters: an infinite weight gives Inf and a zero weight 0 at every
    # step, otherwise a zero step gives 0 and an infinite step Inf
    (isnan(g) | isnan(v)) && return S(NaN)
    isinf(v) && return S(Inf)
    iszero(v) && return zero(S)
    return iszero(g) ? zero(S) : S(Inf)
  end
  # significands in [1/2, 1) have no range problem; the combined exponent goes into the exact scaled
  # product, which rounds once (_ldexp_round), so that no factor is rounded early where it is
  # subnormal. The significand products add a rounding of 2⁻ᵖ before it.
  mg, eg = frexp(g)
  mv, ev = frexp(v)
  my, ey = frexp(y)
  convert(S, _ldexp_round((mg * mv) * (my * my), eg + ev + 2ey))
end

# the proximal operator of γwᵢ max(0, 1 - u) at the margin u = yᵢv, mapped back
# a NaN margin fails u < 1 and stays NaN (E2)
@inline _hinge_apply(v::S, y::S, s::S) where {S<:AbstractFloat} = (u = y * v; ifelse(u < 1, ifelse(u < 1 - s, u + s, one(S)), u) / y)
Base.@propagate_inbounds function prox_entry(f::HingeLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  y = convert(S, _entry(f.y, i))
  _hinge_apply(S(v), y, _hinge_step(S, γ, y, _entry(f.w, i)))
end

# The array loop. The ordinary arithmetic of every entry is a branch-free expression, so that the loop
# vectorizes, and the entries that need range recovery are found by a flag (C2: recovery only on a
# reliable trigger). Without aliasing the flag is gathered while y is written; if it is set, only the
# flagged entries are recomputed by prox_entry from the untouched x. With y === x the coefficients are
# classified first, read-only, without computing a step; if one is not ordinary the entries are
# visited by prox_entry in turn (each is read before it is written).
Base.@propagate_inbounds function _hinge_ordinary(f::HingeLoss, v::T, i::Integer, γ) where {T<:Real}
  S = float(T)
  y = convert(S, _entry(f.y, i))
  s, ok = _hinge_step_ordinary(S, γ, y, _entry(f.w, i))
  _hinge_apply(S(v), y, s), ok
end
# the destination may share memory with a parameter (the step array of a wrapper, a view of the weights)
_shares_memory(y, ::Union{Nothing, Real}) = false
# a lazy wrapper of an array (a scaled or reciprocal step array) may not report the memory it reads, so
# any parameter that is not strided memory counts as possibly shared: its entries are then read one by one
_shares_memory(y, p::AbstractArray) = y === p || Base.mightalias(y, p) || !(p isa StridedArray || p isa AbstractRange)
_aliases_parameters(y, f::HingeLoss, γ) = _shares_memory(y, f.y) || _shares_memory(y, f.w) || _shares_memory(y, γ)
@inline function _hinge_range_ok(::Type{S}, γ, w) where {S<:AbstractFloat}
  R = _working_type(S, γ, w)
  _normal_range(convert(R, γ) * convert(R, w))
end
# element types of the data and the parameters; an abstract one (Vector{Real}) has no single working type
# for the block test, so each entry is promoted by prox_entry
_concrete_eltype(::Nothing) = true
_concrete_eltype(::Real) = true
_concrete_eltype(a::AbstractArray) = isconcretetype(eltype(a))
function _prox_sep_dense!(y::AbstractArray, f::HingeLoss, x::AbstractArray, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  S = float(eltype(x))
  _aliases_parameters(y, f, γ) && return _prox_sep_recover!(y, f, x, γ)   # entry by entry: each parameter is read before it is written
  (_concrete_eltype(x) & _concrete_eltype(f.y) & _concrete_eltype(f.w) & _concrete_eltype(γ)) || return _prox_sep_recover!(y, f, x, γ)
  y === x && return _hinge_blocks!(y, nothing, f, γ, S)   # nothing: in place, the loop names the one array once
  eltype(y) <: AbstractFloat || return _hinge_blocks!(y, x, f, γ, S)   # an ordinary value is converted only where it is the result
  # The ordinary arithmetic of every entry is a branch-free expression, so that the loop vectorizes, and the
  # entries that need range recovery are found by a flag (C2: recovery only on a reliable trigger); the
  # parameters are not y here, so the recovery can read them again.
  T = eltype(y)
  ok = true
  @inbounds for i in eachindex(IndexLinear(), y, x)
    r, oki = _hinge_ordinary(f, x[i], i, _entry(γ, i))
    y[i] = convert(T, r)
    ok &= oki
  end
  ok && return y
  _hinge_repair!(y, f, x, γ)
end
@noinline function _hinge_repair!(y, f, x, γ)
  T = eltype(y)
  S = float(eltype(x))
  # blocks: the test of a block is a branch-free reduction (vectorized), the entries of a block that fails it
  # are visited one by one, so that one exceptional entry costs one block and not the array
  for blk in Iterators.partition(eachindex(IndexLinear(), y, x), 128)
    ok = true
    @inbounds for i in blk
      ok &= _hinge_range_ok(S, _entry(γ, i), _entry(f.w, i))
    end
    ok && continue
    @inbounds for i in blk
      _hinge_range_ok(S, _entry(γ, i), _entry(f.w, i)) || (y[i] = convert(T, prox_entry(f, x[i], i, _entry(γ, i))))
    end
  end
  y
end
# Blocks of 128 entries. The test of a block is a read-only, branch-free reduction over its coefficients
# (vectorized); a block whose coefficients all have a normal product γw is written by the ordinary
# loop (vectorized), any other block entry by entry through prox_entry (the recovery). Every entry reads
# its value and all its parameters before its own slot of y is written, and no other slot, so a
# destination that shares memory with a parameter array (a wrapper's step array, a view) stays exact.
# The ordinary value is converted only where it is the result. With y === x the loop reads the
# destination: under two names the compiler's overlap test between the stores to y and the loads from
# x sent this loop to its scalar version (measured 3x slower).
_hinge_source(y, ::Nothing) = y
_hinge_source(y, x) = x
function _hinge_blocks!(y, x, f, γ, ::Type{S}) where {S}
  T = eltype(y)
  src = _hinge_source(y, x)
  for blk in Iterators.partition(eachindex(IndexLinear(), y, src), 128)
    ok = true
    @inbounds for i in blk
      ok &= _hinge_range_ok(S, _entry(γ, i), _entry(f.w, i))
    end
    if ok
      @inbounds for i in blk
        y[i] = convert(T, _hinge_ordinary(f, src[i], i, _entry(γ, i))[1])
      end
    else
      @inbounds for i in blk
        y[i] = convert(T, prox_entry(f, src[i], i, _entry(γ, i)))
      end
    end
  end
  y
end
@noinline function _prox_sep_recover!(y, f, x, γ)
  T = eltype(y)
  @inbounds for i in eachindex(IndexLinear(), y, x)
    y[i] = convert(T, prox_entry(f, x[i], i, _entry(γ, i)))
  end
  y
end

# CheckLoss array loop: the design of HingeLoss's (a flag in the fused out-of-place loop, blocks in place or for
# an integer output or possibly shared parameters), with its own functions so that the HingeLoss loops stay as
# measured. The thresholds γτwᵢ and γ(1-τ)wᵢ are products of a loop-invariant step-and-level part, formed in the
# promotion R of the model precision and every factor, and the weight; they are the ordinary coefficients of
# _weight_coefficient exactly where the step-and-level parts are normal numbers in R (the weight, whatever it is,
# then needs no recovery), so the range test involves γ and τ only.
_aliases_parameters(y, f::CheckLoss, γ) = _shares_memory(y, f.w) || _shares_memory(y, γ)
@inline function _check_range_ok(::Type{S}, f::CheckLoss, w, γ) where {S<:AbstractFloat}
  R = _working_type(S, γ, w, f.τ)
  g = convert(R, γ)
  _normal_range(g * convert(R, f.τ)) & _normal_range(g * _one_minus(R, f.τ))
end
Base.@propagate_inbounds function _check_ordinary(f::CheckLoss, v::T, i::Integer, γ) where {T<:Real}
  S = float(T)
  w = _entry(f.w, i)
  u = S(v)
  R = _working_type(S, γ, w, f.τ)
  g, wr = convert(R, γ), convert(R, w)
  a, b = g * convert(R, f.τ), g * _one_minus(R, f.τ)
  hi, lo = convert(S, a * wr), convert(S, b * wr)
  r = ifelse(u > hi, u - hi, ifelse(u < -lo, u + lo, zero(S)))
  # the NaN rule of prox_entry
  ifelse((u != u) & (γ == γ) & ((γ == Inf) ? (w == 0) : (w < Inf)), u, r), _normal_range(a) & _normal_range(b)
end
function _prox_sep_dense!(y::AbstractArray, f::CheckLoss, x::AbstractArray, γ)
  _checkwrite(prox!, f, y, x)
  _checkparam(γ, x)
  S = float(eltype(x))
  _aliases_parameters(y, f, γ) && return _prox_sep_recover!(y, f, x, γ)   # entry by entry: each parameter is read before it is written
  (_concrete_eltype(x) & _concrete_eltype(f.w) & _concrete_eltype(γ)) || return _prox_sep_recover!(y, f, x, γ)
  y === x && return _check_blocks!(y, nothing, f, γ, S)   # nothing: in place, the loop names the one array once
  eltype(y) <: AbstractFloat || return _check_blocks!(y, x, f, γ, S)
  T = eltype(y)
  ok = true
  @inbounds for i in eachindex(IndexLinear(), y, x)
    r, oki = _check_ordinary(f, x[i], i, _entry(γ, i))
    y[i] = convert(T, r)
    ok &= oki
  end
  ok && return y
  _check_repair!(y, f, x, γ)
end
@noinline function _check_repair!(y, f, x, γ)
  T = eltype(y)
  S = float(eltype(x))
  for blk in Iterators.partition(eachindex(IndexLinear(), y, x), 128)
    ok = true
    @inbounds for i in blk
      ok &= _check_range_ok(S, f, _entry(f.w, i), _entry(γ, i))
    end
    ok && continue
    @inbounds for i in blk
      _check_range_ok(S, f, _entry(f.w, i), _entry(γ, i)) || (y[i] = convert(T, prox_entry(f, x[i], i, _entry(γ, i))))
    end
  end
  y
end
function _check_blocks!(y, x, f, γ, ::Type{S}) where {S}
  T = eltype(y)
  src = _hinge_source(y, x)
  for blk in Iterators.partition(eachindex(IndexLinear(), y, src), 128)
    ok = true
    @inbounds for i in blk
      ok &= _check_range_ok(S, f, _entry(f.w, i), _entry(γ, i))
    end
    if ok
      @inbounds for i in blk
        y[i] = convert(T, _check_ordinary(f, src[i], i, _entry(γ, i))[1])
      end
    else
      @inbounds for i in blk
        y[i] = convert(T, prox_entry(f, src[i], i, _entry(γ, i)))
      end
    end
  end
  y
end

# f is differentiable at zero, with derivative -wᵢyᵢ
kkt_violation(f::HingeLoss, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); abs(g[i] - _kkt_product(eltype(g), _entry(f.w, i), _entry(f.y, i))))
restrict(f::HingeLoss, S::AbstractVector{<:Integer}) = HingeLoss(_restrict_entries(f.y, S), _restrict_entries(f.w, S))

"""
    SqrHingeLoss()
    SqrHingeLoss(y)
    SqrHingeLoss(y, w)
    SqrHingeLoss(nothing, w)

The squared hinge loss `f(x) = Σᵢ wᵢ max(0, 1 - yᵢxᵢ)²`, for finite nonzero
labels `yᵢ`, usually `±1`, and finite nonnegative weights `wᵢ`, all 1 when
omitted. Without labels, `x` holds the margins; `SqrHingeLoss(nothing, w)`
weights those margins. A zero weight gives the zero function in that entry.

For the label `1`, its proximal operator keeps `v ≥ 1` and otherwise gives
`(v + 2γwᵢ)/(1 + 2γwᵢ)`. A label acts on the margin `yᵢv`, with the step
`γwᵢyᵢ²`. Steps are positive finite reals, including a step per entry.
The loss is convex and smooth, with gradient
`gᵢ = -2wᵢyᵢ max(0, 1 - yᵢxᵢ)` and Lipschitz constant `2 maxᵢ wᵢyᵢ²`.
Its conjugate is `Σᵢ (sᵢ/yᵢ + sᵢ²/(4wᵢyᵢ²))` on `sᵢ/yᵢ ≤ 0`, with a
zero-weight term the indicator of `sᵢ = 0`.

Use this separable loss in splitting methods. [`ProximalBase.SquaredHinge`](@ref)
is the scalar loss of a linear predictor for [`GLMLoss`](@ref).
"""
struct SqrHingeLoss{Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  y::Y
  w::W

  function SqrHingeLoss{Y, W}(y::Y, w::W) where {Y<:Union{Nothing, AbstractArray}, W<:Union{Nothing, AbstractArray}}
    y === nothing || eltype(y) <: Real || _real_error(:SqrHingeLoss)
    w === nothing || eltype(w) <: Real || _real_error(:SqrHingeLoss)
    y === nothing || all(t -> t isa Real && isfinite(t) && !iszero(t), y) ||
      throw(ArgumentError("the labels must be finite and nonzero"))
    w === nothing || all(t -> t isa Real && isfinite(t) && t >= 0, w) ||
      throw(ArgumentError("the weights must be finite and nonnegative"))
    new{Y, W}(y, w)
  end
end
SqrHingeLoss(y::Union{Nothing, AbstractArray} = nothing, w::Union{Nothing, AbstractArray} = nothing) =
  SqrHingeLoss{typeof(y), typeof(w)}(y, w)

isconvex(::Type{<:SqrHingeLoss}) = true
issmooth(::Type{<:SqrHingeLoss}) = true
_checkparams(f::SqrHingeLoss, x) = (_checkparam(f.y, x); _checkparam(f.w, x); true)
restrict(f::SqrHingeLoss, S::AbstractVector{<:Integer}) = SqrHingeLoss(_restrict_entries(f.y, S), _restrict_entries(f.w, S))

# Keep an overflowing active residual as r ⋅ 2^e until the weight is applied.
@inline function _sqrhinge_residual(y::T, v::T) where {T}
  r = max(one(T) - y * v, zero(T))
  isfinite(r) && return r, 0
  my, ey = _frexp(y)
  mv, ev = _frexp(v)
  e = ey + ev
  ldexp(one(T), -e) - my * mv, e
end

@inline function _sqrhinge_value(w::T, r::T, e::Integer) where {T}
  e == 0 && return _positive_product(w, r, r)
  mw, ew = _frexp(w)
  mr, er = _frexp(r)
  ldexp(mw * mr * mr, _exponent_sum(T, ew, er, er, e, e))
end

# The factor 2 must also survive when w|y|r is below the subnormal range. S selects the exponent sum: the margin
# exponent of a GLM predictor can already be a wrapped Int, whose Int sum is right modulo 2^64 (see _weighted_deriv).
@inline _sqrhinge_slope(y::T, w::T, r::T, e::Integer) where {T} = _sqrhinge_slope(T, y, w, r, e)
@inline function _sqrhinge_slope(::Type{S}, y::T, w::T, r::T, e::Integer) where {S, T}
  (iszero(w) || iszero(r)) && return zero(T)
  p = _positive_product(w, abs(y), r)
  e == 0 && floatmin(T) <= p <= floatmax(T) && return 2p
  mw, ew = _frexp(w)
  my, ey = _frexp(abs(y))
  mr, er = _frexp(r)
  ldexp(2 * mw * my * mr, _exponent_sum(S, ew, ey, er, e))
end

Base.@propagate_inbounds function value_entry(f::SqrHingeLoss, v::Real, i::Integer)
  u, y, w = promote(float(v), float(_entry(f.y, i)), float(_entry(f.w, i)))
  iszero(w) && return zero(u)
  r, e = _sqrhinge_residual(y, u)
  _sqrhinge_value(w, r, e)
end

# Scale the numerator and denominator separately when a blend weight or its
# endpoint is outside the floating-point range, but their product need not be.
@noinline function _sqrhinge_prox(v::T, y::T, w::T, γ::T) where {T}
  mv, ev = _frexp(v)
  my, ey = _frexp(y)
  mw, ew = _frexp(w)
  mg, eg = _frexp(γ)
  et = _exponent_sum(T, eg, ew, ey, ey)
  ed = max(0, et)
  d = ldexp(one(T), -ed) + ldexp(2 * mg * mw * my^2, et - ed)
  en = _exponent_sum(T, eg, ew, ey)
  ep = max(ev, en)
  n = ldexp(mv, ev - ep) + ldexp(2 * mg * mw * my, en - ep)
  ldexp(n / d, ep - ed)
end

Base.@propagate_inbounds function prox_entry(f::SqrHingeLoss, v::Real, i::Integer, γ::Real)
  S = float(typeof(v))
  u, y, w, step = promote(float(v), float(_entry(f.y, i)), float(_entry(f.w, i)), float(γ))
  (iszero(w) || iszero(step)) && return S(v)
  c = inv(y)
  (y > 0 ? u >= c : u <= c) && return S(v)
  h = _positive_product(step, w, abs(y))
  s = h * abs(y)
  if floatmin(typeof(u)) <= h <= floatmax(typeof(u)) &&
     floatmin(typeof(u)) <= s <= floatmax(typeof(u)) && isfinite(c)
    a, b = _ridge_weights(s, oftype(s, 2))
    return S(_ridge_blend(u, c, a, b))
  end
  S(_sqrhinge_prox(u, y, w, step))
end

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::SqrHingeLoss, g::_ArrayOrOwner, x::_ArrayOrOwner)
  _checkreal(f, x)
  _checkreal(f, g)
  _checkgradient_output(g, x)
  _checksize(g, x)
  _checkparams(f, x)
  true
end

Base.@propagate_inbounds function _sqrhinge_gradient(f::SqrHingeLoss, v::Real, i::Integer)
  u, y, w = promote(float(v), float(_entry(f.y, i)), float(_entry(f.w, i)))
  iszero(w) && return zero(u)
  r, e = _sqrhinge_residual(y, u)
  -copysign(_sqrhinge_slope(y, w, r, e), y)
end

gradient!(g::_ArrayOrOwner, f::SqrHingeLoss, x::_ArrayOrOwner) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::AbstractArray, f::SqrHingeLoss, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  @inbounds for i in eachindex(IndexLinear(), g, x)
    g[i] = _sqrhinge_gradient(f, x[i], i)
  end
  g
end
# an owner point with a dense output: the entries of the matrix that the owner represents
_gradient_unchecked!(g::AbstractArray, f::SqrHingeLoss, x::SymmetricSparseIterate) =
  _gradient_unchecked!(g, f, matrixview(x))
# an owner output: the symmetric part of the entrywise gradient
function _gradient_unchecked!(g::SymmetricSparseIterate, f::SqrHingeLoss, x::_ArrayOrOwner)
  _checkwrite(gradient!, f, g, x)
  _symmetric_coordinates!(g, (v, j) -> _sqrhinge_gradient(f, v, j), x)
end

value_and_gradient!(g::_ArrayOrOwner, f::SqrHingeLoss, x::_ArrayOrOwner) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
_value_and_gradient_unchecked!(g::AbstractArray, f::SqrHingeLoss, x::SymmetricSparseIterate) =
  _value_and_gradient_unchecked!(g, f, matrixview(x))
function _value_and_gradient_unchecked!(g::SymmetricSparseIterate, f::SqrHingeLoss, x::_ArrayOrOwner)
  _checkwrite(value_and_gradient!, f, g, x)
  v = _value_unchecked(f, x)
  _symmetric_coordinates!(g, (u, j) -> _sqrhinge_gradient(f, u, j), x)
  v
end
function _value_and_gradient_unchecked!(g::AbstractArray, f::SqrHingeLoss, x::AbstractArray)
  _checkwrite(value_and_gradient!, f, g, x)
  T = float(eltype(x))
  v = zero(T)
  @inbounds for i in eachindex(IndexLinear(), g, x)
    u, y, w = promote(float(x[i]), float(_entry(f.y, i)), float(_entry(f.w, i)))
    r, e = iszero(w) ? (zero(u), 0) : _sqrhinge_residual(y, u)
    v += convert(T, _sqrhinge_value(w, r, e))
    g[i] = -copysign(_sqrhinge_slope(y, w, r, e), y)
  end
  v
end

# f is differentiable at zero, with derivative -2wᵢyᵢ.
function kkt_violation(f::SqrHingeLoss, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  y, w = promote(float(_entry(f.y, i)), float(_entry(f.w, i)))
  abs(g[i] - copysign(_sqrhinge_slope(y, w, one(y), 0), y))
end

function lipschitz(f::SqrHingeLoss)
  a = f.y === nothing ? f.w : f.y
  a === nothing && return 2.0
  _checkparams(f, a)
  T = float(promote_type(f.y === nothing ? Bool : eltype(f.y), f.w === nothing ? Bool : eltype(f.w)))
  L = zero(T)
  @inbounds for i in eachindex(IndexLinear(), a)
    y, w = T(_entry(f.y, i)), T(_entry(f.w, i))
    L = max(L, _sqrhinge_slope(y, w, abs(y), 0))
  end
  L
end

"""
    EpsInsensitiveLoss(ε)
    EpsInsensitiveLoss(ε, w)

The ε-insensitive loss of support vector regression,
`f(x) = Σᵢ wᵢ max(0, |xᵢ| - ε)` for `ε ≥ 0`, with all `wᵢ = 1` when `w` is
omitted: residuals within `ε` cost nothing. Its proximal operator keeps an entry
with `|v| ≤ ε`, puts one with `ε < |v| ≤ ε + γwᵢ` at `±ε`, and moves a larger one
towards zero by `γwᵢ`. As for [`CheckLoss`](@ref), the loss of predictions `z`
of `y` is `Translate(EpsInsensitiveLoss(ε), y)`. Its conjugate is `ε Σᵢ |sᵢ|` on
the box `|sᵢ| ≤ wᵢ`.
"""
struct EpsInsensitiveLoss{T<:Real, W<:Union{Nothing, AbstractArray}} <: SeparableFunction
  ε::T
  w::W

  function EpsInsensitiveLoss{T, W}(ε::T, w::W) where {T<:Real, W<:Union{Nothing, AbstractArray}}
    w === nothing || eltype(w) <: Real || _real_error(:EpsInsensitiveLoss)
    _checknonneg("ε", ε)
    _checknonneg("the weights", w)
    new{T, W}(ε, w)
  end
end
EpsInsensitiveLoss(ε::Real, w::Union{Nothing, AbstractArray} = nothing) = EpsInsensitiveLoss{typeof(ε), typeof(w)}(ε, w)

isconvex(::Type{<:EpsInsensitiveLoss}) = true
iszeropreserving(::EpsInsensitiveLoss) = true
_checkparams(f::EpsInsensitiveLoss, x) = (_checkparam(f.w, x); true)

Base.@propagate_inbounds function value_entry(f::EpsInsensitiveLoss, v::T, i::Integer) where {T<:Real}
  w = _entry(f.w, i)
  φ = max(abs(v) - convert(float(T), f.ε), zero(float(T)))
  # An infinite ε makes the loss zero, whatever the entry.
  isinf(f.ε) && return zero(w * φ)
  # Where φ is NaN (a NaN entry, or an infinite one where a finite ε rounds to Inf in the entry's
  # type), a zero weight leaves the entry free; other weights give NaN, and other entries take
  # the expression below unchanged.
  isnan(φ) && return iszero(w) ? zero(w * φ) : w * φ
  _weighted_value(w * φ, φ)
end
Base.@propagate_inbounds _value_entry_ordinary(f::EpsInsensitiveLoss, v::T, i::Integer) where {T<:Real} =
  _entry(f.w, i) * max(abs(v) - convert(float(T), f.ε), zero(float(T)))
_has_ordinary_value(::Type{<:EpsInsensitiveLoss}) = true
Base.@propagate_inbounds function prox_entry(f::EpsInsensitiveLoss, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  w = _entry(f.w, i)
  s, ε, u = _weight_coefficient(S, γ, w), convert(S, f.ε), S(v)
  a = abs(u)
  a <= ε ? u : (a <= ε + s ? copysign(ε, u) : u - copysign(s, u))
end

# zero is inside the flat part when ε > 0; with ε = 0 the loss is wᵢ|xᵢ|
kkt_violation(f::EpsInsensitiveLoss, g::AbstractArray, i::Integer) =
  (eltype(g) <: Real || _real_error(:kkt_violation); f.ε > 0 ? abs(g[i]) : abs(g[i]) - _entry(f.w, i))
restrict(f::EpsInsensitiveLoss, S::AbstractVector{<:Integer}) = EpsInsensitiveLoss(f.ε, _restrict_entries(f.w, S))

# the values of their conjugates, for Conjugate, within the tolerance of the
# indicator functions (_within is in calculus.jl)
function _conjugate_value(f::CheckLoss, s::AbstractArray)
  _checkconjugate(f, s)
  T = float(eltype(s))
  τ = convert(T, f.τ)
  @inbounds for i in eachindex(IndexLinear(), s)
    w = _entry(f.w, i)
    (_within(s[i], w * τ, T) && _within(-s[i], w * (1 - τ), T)) || return T(Inf)
  end
  zero(T)
end
function _conjugate_value(f::HingeLoss, s::AbstractArray)
  _checkconjugate(f, s)
  T = float(eltype(s))
  v = zero(T)
  @inbounds for i in eachindex(IndexLinear(), s)
    q = s[i] / _entry(f.y, i)
    (_within(q, zero(T), T) && _within(-q, convert(T, _entry(f.w, i)), T)) || return T(Inf)
    v += convert(T, q)
  end
  v
end
function _conjugate_value(f::SqrHingeLoss, s::AbstractArray)
  _checkconjugate(f, s)
  T = float(eltype(s))
  v = zero(T)
  @inbounds for i in eachindex(IndexLinear(), s)
    u, y, w = promote(float(s[i]), float(_entry(f.y, i)), float(_entry(f.w, i)))
    if iszero(w)
      _within(abs(u), zero(u), T) || return T(Inf)
    else
      _within(u / y, zero(u), T) || return T(Inf)
      # Add the linear and quadratic terms at one scale before rounding.
      ms, es = _frexp(u)
      my, ey = _frexp(y)
      mw, ew = _frexp(w)
      U = typeof(u)
      q, eq = ms / my, _exponent_sum(U, es, -ey)
      eq2 = _exponent_sum(U, eq, eq, -ew)
      e = max(eq, eq2)
      z = ldexp(q, eq - e) + ldexp(q^2 / (4mw), eq2 - e)
      v += convert(T, ldexp(z, e))
    end
  end
  v
end
function _conjugate_value(f::EpsInsensitiveLoss, s::AbstractArray)
  _checkconjugate(f, s)
  T = float(eltype(s))
  v = zero(T)
  @inbounds for i in eachindex(IndexLinear(), s)
    a = abs(s[i])
    _within(a, convert(T, _entry(f.w, i)), T) || return T(Inf)
    v += a
  end
  iszero(f.ε) || iszero(v) ? zero(T) : convert(T, f.ε) * v
end

####################################
#
#  Smooth losses of a linear predictor: the squared hinge and the smoothed
#  check losses, with bounded curvature
#
####################################

"""
    ProximalBase.SquaredHinge()

The squared hinge loss `ℓ(η, y) = max(0, 1 - yη)²` for a label `y ∈ {-1, 1}`,
the loss of the L2-loss support vector machine. Smooth, with curvature 2, so
coordinate descent and gradient methods minimize a [`GLMLoss`](@ref) with it
directly.
For a separable loss with a proximal operator, use [`SqrHingeLoss`](@ref).
"""
struct SquaredHinge <: ScalarLoss end
@inline function value_and_deriv(::SquaredHinge, η::Real, y::Real)
  m = max(1 - y * float(η), zero(float(η)))
  (abs2(m), -2 * y * m)
end
(ℓ::SquaredHinge)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::SquaredHinge, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
deriv2(::SquaredHinge, η::Real, y::Real) = (t = float(η); y * t < 1 ? 2 * abs2(y) * one(t) : zero(t))
curvature(::SquaredHinge) = 2.0
# The margin m = max(1 - yη, 0) of η = ηs 2^k as r 2^e, by the helpers of
# SqrHingeLoss, which keep an overflowing yη apart. An active margin is at least
# about ulp(1)/2, so m² does not underflow; the derivative -2ym does when |y| is
# far below 1.
_lost_deriv(::SquaredHinge, v, η, y) = _below_normal(v) & !iszero(y) & (y * η < 1)
_pure_everywhere(::SquaredHinge) = true   # src/functions/glm.jl
@inline function _sqrhinge_margin(y::T, ηs::T, k::Int) where {T}
  k == 0 && return _sqrhinge_residual(y, ηs)
  (iszero(y) || y * ηs > 0) && return (iszero(y) ? one(T) : zero(T)), 0
  my, ey = _frexp(y)
  mv, ev = _frexp(ηs)
  e = ey + ev + k
  ldexp(one(T), -e) - my * mv, e
end
function _weighted_pair(::SquaredHinge, w::T, ηs::T, k::Int, y::T) where {T}
  r, e = _sqrhinge_margin(y, ηs, k)
  _pow2_pair(w, r, r, 2e)
end
# The margin exponent e of _sqrhinge_margin is an Int that can already have wrapped. Its Int total with the three
# exponents of the slope equals the true total modulo 2^64, so it is the true total when that fits Int64. A wide sum
# would take the wrapped e as the true one, so this caller keeps the Int sum.
_weighted_deriv(::SquaredHinge, w::T, ηs::T, k::Int, y::T) where {T} =
  -copysign(_sqrhinge_slope(Int, y, w, _sqrhinge_margin(y, ηs, k)...), y)

"""
    ProximalBase.SmoothCheck(τ, h)

The check loss of quantile regression smoothed at the scale `h > 0`: the Moreau
envelope `min_s ρ_τ(s) + (s - u)²/(2h)` of `ρ_τ(u) = max(τu, (τ - 1)u)` at the
residual `u = y - η`. It is `τu - hτ²/2` for `u ≥ hτ`,
`(τ - 1)u - h(1 - τ)²/2` for `u ≤ -h(1 - τ)`, and `u²/(2h)` in between: an
asymmetric Huber loss, within `h/2` of `ρ_τ`, with curvature `1/h`. With it,
coordinate descent and proximal Newton fit penalized quantile regression
through [`GLMLoss`](@ref), without a linear program.
"""
struct SmoothCheck{T<:Real} <: ScalarLoss
  τ::T
  h::T

  function SmoothCheck{T}(τ::T, h::T) where {T<:Real}
    0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1), got $τ"))
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    new{T}(τ, h)
  end
end
SmoothCheck(τ::Real, h::Real) = (p = promote(τ, h); SmoothCheck{typeof(p[1])}(p...))
# In forms that avoid a material loss of accuracy in their intermediates: the
# branch is chosen by q = u/h against τ and -(1 - τ), not u against hτ (hτ can
# underflow and misplace the branch, at an exact fit with a subnormal h for
# instance); the linear branches take the small factor τ (or 1 - τ) out,
# τ(u - hτ/2) rather than τu - hτ²/2 (τ² can underflow before the product with a
# large h); the middle one is q u/2 rather than u²/(2h) (u² can underflow, or 2h
# overflow). In the middle |q| < 1, and where the value is normal q is at worst
# one bit below floatmin. A value or derivative below floatmin has lost accuracy
# (the recovery recomputes it where a weight lifts it).
@inline function value_and_deriv(ℓ::SmoothCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  σ = 1 - τ
  q = u / h
  q >= τ && return (τ * (u - h * τ / 2), -τ)
  q <= -σ && return (σ * (-u - h * σ / 2), σ)
  (q * u / 2, -q)
end
(ℓ::SmoothCheck)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::SmoothCheck, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
function deriv2(ℓ::SmoothCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  -(1 - τ) < u / h < τ ? inv(h) : zero(u)   # the branches of value_and_deriv
end
curvature(ℓ::SmoothCheck) = inv(float(ℓ.h))
# The residual u = y - η as m 2^e; the branches as in value_and_deriv, the linear
# ones as τ(u - hτ/2) and (1 - τ)(-u - h(1 - τ)/2) in units of 2^e, the middle one
# as m²/(2 mₕ) in units of 2^(2e - eₕ).
# with the forms of value_and_deriv a normal value is accurate; a value below
# floatmin has lost accuracy, which only a weight w > 1 can lift (the default)
_lost_value(::SmoothCheck, v, η, y) = _below_normal(v) & (η != y)
_lost_deriv(::SmoothCheck, v, η, y) = _below_normal(v) & (η != y)
# linear beyond a bounded zone of u = y - η, NaN in it; ℓ' bounded by max(τ, 1 - τ)
# (src/functions/glm.jl, `_value_traps` and `_deriv_in_value`)
_value_traps(::SmoothCheck) = true
_deriv_in_value(::SmoothCheck) = true
_pure_everywhere(::SmoothCheck) = true
# the branch from q = u/h = (m/mₕ) 2^(e - eₕ) against τ and -(1 - τ), compared
# in significand and exponent, so that neither q nor hτ underflows on the way
@inline function _smoothcheck_parts(ℓ::SmoothCheck, ηs::T, k::Int, y::T) where {T}
  m, e = _residual_pow2(ηs, k, y)
  τ, h = convert(T, ℓ.τ), convert(T, ℓ.h)
  mh, eh = _frexp(h)
  qm, qe = -m / mh, e - eh
  zone = _pow2_cmp(qm, qe, τ) >= 0 ? 1 : _pow2_cmp(qm, qe, -(1 - τ)) <= 0 ? -1 : 0
  -m, e, τ, h, zone
end
# the sign of a 2^ea - b, for finite a and b
@inline function _pow2_cmp(a::T, ea::Int, b::T) where {T}
  (iszero(a) || iszero(b) || signbit(a) != signbit(b)) && return cmp(a, b)
  ma, xa = _frexp(a)
  mb, xb = _frexp(b)
  xa += ea
  xa == xb && return cmp(ma, mb)
  s = xa > xb ? 1 : -1
  signbit(a) ? -s : s
end
# c t/2 in units of 2^e, for positive c and t, without forming c t in T (it can
# underflow before the units restore it)
@inline _half_product_units(c::T, t::T, e::Int) where {T} =
  (fc = _frexp(c); ft = _frexp(t); ldexp(fc[1] * ft[1], fc[2] + ft[2] - 1 - e))
function _weighted_pair(ℓ::SmoothCheck, w::T, ηs::T, k::Int, y::T) where {T}
  m, e, τ, h, zone = _smoothcheck_parts(ℓ, ηs, k, y)
  zone > 0 && return _pow2_pair(w, τ, m - _half_product_units(h, τ, e), e)
  zone < 0 && return _pow2_pair(w, 1 - τ, -m - _half_product_units(h, 1 - τ, e), e)
  mh, eh = _frexp(h)
  _pow2_pair(w, m * m / (2mh), 2e - eh)
end
function _weighted_deriv(ℓ::SmoothCheck, w::T, ηs::T, k::Int, y::T) where {T}
  m, e, τ, h, zone = _smoothcheck_parts(ℓ, ηs, k, y)
  zone > 0 && return w * -τ
  zone < 0 && return w * (1 - τ)
  mh, eh = _frexp(h)
  _scaled_product(w, -m / mh, e - eh)
end

"""
    ProximalBase.ConvolutionCheck(τ, h; kernel = :epanechnikov)

The check loss of quantile regression convolved with a kernel of bandwidth
`h > 0`, the smoothing of the conquer estimator (He, Pan, Tan and Zhou, 2023):
`∫ ρ_τ(u - t) K(t/h)/h dt` at the residual `u = y - η`. The kernels are
`:uniform`, `K = 1/2` on `[-1, 1]`, and `:epanechnikov`,
`K(t) = 3(1 - t²)/4` on `[-1, 1]`. The loss equals `ρ_τ(u)` for `|u| ≥ h`
and is a polynomial in between, with second derivative `K(u/h)/h`, so its
curvature is `1/(2h)` or `3/(4h)`. Unlike [`ProximalBase.SmoothCheck`](@ref),
it is centered: its derivative is `τ - 1/2` at `u = 0`.
"""
struct ConvolutionCheck{T<:Real, K} <: ScalarLoss
  τ::T
  h::T

  function ConvolutionCheck{T, K}(τ::T, h::T) where {T<:Real, K}
    K in (:uniform, :epanechnikov) || throw(ArgumentError("the kernel must be :uniform or :epanechnikov, got :$K"))
    0 < τ < 1 || throw(ArgumentError("τ must be in (0, 1), got $τ"))
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    new{T, K}(τ, h)
  end
end
ConvolutionCheck(τ::Real, h::Real; kernel::Symbol = :epanechnikov) =
  (p = promote(τ, h); ConvolutionCheck{typeof(p[1]), kernel}(p...))

# ρ_τ(u) = (τ - 1/2)u + |u|/2, and the convolution of |u| with the kernel at
# bandwidth h is h m(u/h) for |u| < h, with m(s) = ∫|s - t|K(t)dt: m, m' and
# m'' = 2K on [-1, 1]
@inline _kernel_abs(::ConvolutionCheck{<:Any, :uniform}, s) = ((1 + s^2) / 2, s, one(s))
@inline _kernel_abs(::ConvolutionCheck{<:Any, :epanechnikov}, s) =
  (3 * (1 + 2s^2) / 8 - s^4 / 8, 3s / 2 - s^3 / 2, 3 * (1 - s^2) / 2)

@inline function value_and_deriv(ℓ::ConvolutionCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  u >= h && return (τ * u, -τ)
  u <= -h && return ((τ - 1) * u, 1 - τ)
  m, dm, _ = _kernel_abs(ℓ, u / h)
  ((τ - one(τ) / 2) * u + h * m / 2, -(τ - one(τ) / 2) - dm / 2)
end
(ℓ::ConvolutionCheck)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::ConvolutionCheck, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
function deriv2(ℓ::ConvolutionCheck, η::Real, y::Real)
  u = float(y - η)
  h = convert(typeof(u), ℓ.h)
  abs(u) < h ? _kernel_abs(ℓ, u / h)[3] / (2h) : zero(u)
end
curvature(ℓ::ConvolutionCheck{<:Any, :uniform}) = inv(2 * float(ℓ.h))
curvature(ℓ::ConvolutionCheck{<:Any, :epanechnikov}) = 3 / (4 * float(ℓ.h))
# The residual u = y - η as m 2^e: the linear branches c u in units of 2^e, the
# kernel zone (τ - 1/2)u + h m(u/h)/2 as two terms added in power-of-two units.
# The derivative -(τ - 1/2) - m'(s)/2 is bounded, but s = u/h is subnormal for a
# subnormal residual: there m'(s) = s (uniform) or 3s/2 (Epanechnikov, s³ negligible),
# kept as (u/h) in units of 2^(e - eₕ).
_lost_value(::ConvolutionCheck, v, η, y) = _below_normal(v)
_lost_deriv(::ConvolutionCheck, v, η, y) = _below_normal(v)
# linear for |u| ≥ h, NaN in the kernel zone; ℓ' bounded by 1/2 + |τ - 1/2|
_value_traps(::ConvolutionCheck) = true
_deriv_in_value(::ConvolutionCheck) = true
_pure_everywhere(::ConvolutionCheck) = true
_kernel_slope0(::ConvolutionCheck{<:Any, :uniform}) = 1
_kernel_slope0(::ConvolutionCheck{<:Any, :epanechnikov}) = 3 // 2
function _weighted_deriv(ℓ::ConvolutionCheck, w::T, ηs::T, k::Int, y::T) where {T}
  m, e = _residual_pow2(ηs, k, y)
  m = -m
  τ, h = convert(T, ℓ.τ), convert(T, ℓ.h)
  u = ldexp(m, e)
  u >= h && return w * -τ
  u <= -h && return w * (1 - τ)
  mh, eh = _frexp(h)
  ms, es = m / mh, e - eh
  s = ldexp(ms, es)
  b, eb = abs(s) >= floatmin(T) ? _frexp(-_kernel_abs(ℓ, s)[2] / 2) : (-convert(T, _kernel_slope0(ℓ)) * ms / 2, es)
  a, ea = _pow2_add2(_frexp(-(τ - one(τ) / 2))..., b, eb)
  _scaled_product(w, a, ea)
end
function _weighted_pair(ℓ::ConvolutionCheck, w::T, ηs::T, k::Int, y::T) where {T}
  m, e = _residual_pow2(ηs, k, y)
  m = -m
  τ, h = convert(T, ℓ.τ), convert(T, ℓ.h)
  u = ldexp(m, e)
  u >= h && return _pow2_pair(w, τ, m, e)
  u <= -h && return _pow2_pair(w, τ - 1, m, e)
  ma, ea = _product_pow2(τ - one(τ) / 2, m)
  mh, eh = _frexp(h)
  a, b = _pow2_add2(ma, ea + e, mh * _kernel_abs(ℓ, u / h)[1] / 2, eh)
  _pow2_pair(w, a, b)
end

####################################
#  values as a significand and an exponent
####################################

# The scaled-value mechanism (`_sep_value_parts`, separable.jl): the terms w φ with φ as parts,
# so that neither a margin y v, a square nor a product with a weight need be representable. A
# term at a non-finite entry or label is the ordinary one.
_has_value_parts(::Type{<:Union{HingeLoss, SqrHingeLoss, EpsInsensitiveLoss, CheckLoss}}) = true
# Exact-zero protection, a proof per operation where a term can round into the subnormal range of the
# value type T (each such rounding loses less than δ/2, δ the smallest subnormal of T).
# - Hinge and squared hinge without weights, with labels that are absent, integers or Float16/32/64
#   (decided by the type): the term is computed in the promotion S of T and the labels' type, and
#   max(1 - t, 0), t the margin y v rounded in S, is 0 or at least eps(S)/2 >= 2^-54, a normal
#   number in every argument type (1 - t for a float t < 1); its square is at least 2^-108, also
#   normal. A margin that rounds to 1 or more is a zero within the accuracy of the ordinary
#   computation, measured against the magnitudes of 1 and y v (R1), and the parts round the same
#   margin. Labels of a wider type (BigFloat, and BigInt, whose margin is formed in BigFloat) can
#   give a term below the normal range of T, which converting it to T loses: the bound below
#   applies instead. Fixed-width integer and Bool labels promote with T to T or to Float64.
# - ε-insensitive without weights, with an ε of a type that converts exactly into every argument
#   type (Bool, 8- and 16-bit integers, Float16, Float32): |v| - ε that is subnormal is an exact
#   difference, and a sum whose result is subnormal is exact. Another ε is rounded to T first; an ε
#   that keeps its range in T is rounded with a relative error, so a zero term is within the
#   accuracy of the ordinary computation (measured against ε), and only the last roundings lose:
#   the bound below; an ε that does not keep its range (a subnormal or zero in T) has no bound.
# In those trait cases a zero or subnormal value is exact, and Scaled skips the recovery for it.
const _HingeExactLabels = Union{Nothing, AbstractArray{<:Union{Bool, Base.BitInteger, Float16, Float32, Float64}}}
const _ExactEps = Union{Bool, Int8, UInt8, Int16, UInt16, Float16, Float32}
_zero_value_exact(::Union{HingeLoss{<:_HingeExactLabels, Nothing}, SqrHingeLoss{<:_HingeExactLabels, Nothing},
                          EpsInsensitiveLoss{<:_ExactEps, Nothing}}) = true
# Otherwise the term w φ (or φ, or its conversion to T) is the last product, which loses less than
# δ/2: |f(x)| < n δ. The check loss rounds τ to T (which must keep its range there) and τ v first,
# which loses less than δ/2 where it is subnormal and is then multiplied by a weight: one pass over
# the weights bounds them (the unweighted loss multiplies by 1).
function _zero_value_exponent(f::Union{HingeLoss, SqrHingeLoss, EpsInsensitiveLoss}, x::AbstractArray)
  _zero_value_exact(f) && return _EXACT_ZERO
  if f isa EpsInsensitiveLoss
    T = _valtype(x)
    # an ε that converts exactly to T makes |v| - ε exact where it is subnormal, as for the trait
    f.w === nothing && convert(T, f.ε) == f.ε && return _EXACT_ZERO
    _level_fits(T, f.ε) || return nothing
  end
  _underflow_exponent(_valtype(x), length(x))
end
function _zero_value_exponent(f::CheckLoss, x::AbstractArray)
  _level_fits(_valtype(x), f.τ) || return nothing
  f.w === nothing && return _underflow_exponent(_valtype(x), length(x))
  we = _weight_exponents(f.w)
  we === nothing ? nothing : _underflow_exponent(_valtype(x), length(x), 1 + max(0, we[2]))
end
function _value_parts(f::Union{HingeLoss, SqrHingeLoss, EpsInsensitiveLoss}, x::AbstractArray, v)
  iszero(v) && _zero_value_exact(f) && return nothing
  _sep_value_parts(f, x, _sep_parts_type(x, f isa EpsInsensitiveLoss ? f.ε : f.y, f.w))
end
_value_parts(f::CheckLoss, x::AbstractArray, v) = _sep_value_parts(f, x, _sep_parts_type(x, f.τ, f.w))
_parameter_alias(f::Union{HingeLoss, SqrHingeLoss}, out) = _mightalias_parameter(out, f.y) || _mightalias_parameter(out, f.w)
_parameter_alias(f::Union{EpsInsensitiveLoss, CheckLoss}, out) = _mightalias_parameter(out, f.w)

# a term computed as the ordinary one, for a non-finite entry or parameter
@inline _ordinary_term_add(acc::_ExpSum, f, v, i) = _term_add(acc, value_entry(f, v, i), 0)

Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::HingeLoss, v, i) where {R}
  y, u, w = convert(R, _entry(f.y, i)), convert(R, v), _entry(f.w, i)
  (isfinite(y) && isfinite(u)) || return _ordinary_term_add(acc, f, v, i)
  r, e = _sqrhinge_residual(y, u)
  _weighted_term_add(acc, w, _parts(r, e))
end
Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::SqrHingeLoss, v, i) where {R}
  y, u, w = convert(R, _entry(f.y, i)), convert(R, v), _entry(f.w, i)
  isfinite(u) || return _ordinary_term_add(acc, f, v, i)
  r, e = _sqrhinge_residual(y, u)
  mr, er = _frexp(r)
  _weighted_term_add(acc, w, _parts(mr * mr, 2 * (er + e)))
end
Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::EpsInsensitiveLoss, v, i) where {R}
  u = convert(R, v)
  isfinite(u) || return _ordinary_term_add(acc, f, v, i)
  _weighted_term_add(acc, _entry(f.w, i), _frexp(max(abs(u) - convert(R, f.ε), zero(R))))
end
Base.@propagate_inbounds function _entry_terms(acc::_ExpSum{R}, f::CheckLoss, v, i) where {R}
  u = convert(R, v)
  isfinite(u) || return _ordinary_term_add(acc, f, v, i)
  τ = convert(R, f.τ)
  _weighted_term_add(acc, _entry(f.w, i), _parts_product(_frexp(u >= 0 ? τ : τ - 1), _frexp(u)))
end
