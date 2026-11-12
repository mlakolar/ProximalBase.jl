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
Values follow the overflow rule in Overflow and underflow of values.
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

# Preserve the original ordinary loop; retry zeros with SIMD before resorting
# to compensation. Every pass checks intermediates, not just the final sum.
function _value_dense(
    f::NormL1{T,<:Array{T}},
    x::Array{T}) where {T<:Union{Float32,Float64}}
  s, ordinary = _l1_value_sum(f, x, false)
  ordinary && return s
  s, ordinary = _l1_value_sum(f, x, true)
  # 0*Inf/NaN needs the entrywise zero-factor and domain priorities.
  ordinary && !isnan(s) ? s : invoke(_value_dense, Tuple{Any,AbstractArray}, f, x)
end
