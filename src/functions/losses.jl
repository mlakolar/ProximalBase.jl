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
  _weighted_value(w * φ, φ)
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
  u > hi ? u - hi : (u < -lo ? u + lo : zero(S))
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
  _weighted_value(w * φ, φ)
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
@inline _hinge_apply(v::S, y::S, s::S) where {S<:AbstractFloat} = (u = y * v; ifelse(u >= 1, u, ifelse(u < 1 - s, u + s, one(S))) / y)
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
_aliases_parameters(y, f::HingeLoss, γ) = y === f.y || y === f.w || y === γ   # γ may be the step array
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
  _hinge_blocks!(y, y === x ? nothing : x, f, γ, S)   # nothing: in place, the loop names the one array once
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
  my, ey = frexp(y)
  mv, ev = frexp(v)
  e = ey + ev
  ldexp(one(T), -e) - my * mv, e
end

@inline function _sqrhinge_value(w::T, r::T, e::Int) where {T}
  e == 0 && return _positive_product(w, r, r)
  mw, ew = frexp(w)
  mr, er = frexp(r)
  ldexp(mw * mr * mr, ew + 2 * (er + e))
end

# The factor 2 must also survive when w|y|r is below the subnormal range.
@inline function _sqrhinge_slope(y::T, w::T, r::T, e::Int) where {T}
  (iszero(w) || iszero(r)) && return zero(T)
  p = _positive_product(w, abs(y), r)
  e == 0 && floatmin(T) <= p <= floatmax(T) && return 2p
  mw, ew = frexp(w)
  my, ey = frexp(abs(y))
  mr, er = frexp(r)
  ldexp(2 * mw * my * mr, ew + ey + er + e)
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
  mv, ev = frexp(v)
  my, ey = frexp(y)
  mw, ew = frexp(w)
  mg, eg = frexp(γ)
  et = eg + ew + 2ey
  ed = max(0, et)
  d = ldexp(one(T), -ed) + ldexp(2 * mg * mw * my^2, et - ed)
  en = eg + ew + ey
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

function _checkwrite(::Union{typeof(gradient!), typeof(value_and_gradient!)}, f::SqrHingeLoss, g::AbstractArray, x::AbstractArray)
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

gradient!(g::AbstractArray, f::SqrHingeLoss, x::AbstractArray) =
  (_check_oracle_arrays(gradient!, f, g, x); _gradient_unchecked!(g, f, x))
function _gradient_unchecked!(g::AbstractArray, f::SqrHingeLoss, x::AbstractArray)
  _checkwrite(gradient!, f, g, x)
  @inbounds for i in eachindex(IndexLinear(), g, x)
    g[i] = _sqrhinge_gradient(f, x[i], i)
  end
  g
end

value_and_gradient!(g::AbstractArray, f::SqrHingeLoss, x::AbstractArray) =
  (_check_oracle_arrays(value_and_gradient!, f, g, x); _value_and_gradient_unchecked!(g, f, x))
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
      ms, es = frexp(u)
      my, ey = frexp(y)
      mw, ew = frexp(w)
      q, eq = ms / my, es - ey
      e = max(eq, 2eq - ew)
      z = ldexp(q, eq - e) + ldexp(q^2 / (4mw), 2eq - ew - e)
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
  iszero(f.ε) ? zero(T) : convert(T, f.ε) * v
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
@inline function value_and_deriv(ℓ::SmoothCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  u >= h * τ && return (τ * u - h * τ^2 / 2, -τ)
  u <= -h * (1 - τ) && return ((τ - 1) * u - h * (1 - τ)^2 / 2, 1 - τ)
  (u^2 / (2h), -u / h)
end
(ℓ::SmoothCheck)(η::Real, y::Real) = value_and_deriv(ℓ, η, y)[1]
deriv(ℓ::SmoothCheck, η::Real, y::Real) = value_and_deriv(ℓ, η, y)[2]
function deriv2(ℓ::SmoothCheck, η::Real, y::Real)
  u = float(y - η)
  τ, h = convert(typeof(u), ℓ.τ), convert(typeof(u), ℓ.h)
  -h * (1 - τ) < u < h * τ ? inv(h) : zero(u)
end
curvature(ℓ::SmoothCheck) = inv(float(ℓ.h))

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
