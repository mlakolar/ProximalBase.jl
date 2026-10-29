####################################
#
#  Local linear approximation (reweighted ℓ1) of nonconvex penalties
#
####################################

# A penalty Σᵢ pᵢ(|xᵢ|) with each pᵢ concave and nondecreasing on [0, ∞) lies
# below its tangent at x̃: pᵢ(|t|) ≤ pᵢ(|x̃ᵢ|) + pᵢ′(|x̃ᵢ|)(|t| - |x̃ᵢ|). The
# next stage of LLA minimizes the loss plus the weighted ℓ1 norm Σᵢ wᵢ|xᵢ|
# with wᵢ = pᵢ′(|x̃ᵢ|), a convex problem. The slopes are taken from the right,
# which at a kink is a supergradient of the concave pᵢ, so the majorization
# holds there too.

"""
    lla_weight(f, v, i)

The slope `pᵢ′(|v|)` of the nonconvex penalty `f(x) = Σᵢ pᵢ(|xᵢ|)` at `|v|`,
taken from the right: the weight of entry `i` in the local linear approximation
(LLA) of `f` at a point whose entry `i` is `v`. `i` is a linear index, as in
[`value_entry`](@ref), and the entry weights of `f` are included. For
[`GroupMCP`](@ref) and [`GroupSCAD`](@ref), `i` is a group and `v` its
Euclidean norm. [`lla_weights!`](@ref) gives the weights of a whole point.

| `f` | `pᵢ′(t)` for `t > 0` | `pᵢ′(0)` |
|:----|:---------------------|:---------|
| [`CappedL1`](@ref)`(λ, a, w)` | `λwᵢ` for `t < a`, `0` for `t ≥ a` | `λwᵢ` |
| [`LogPenalty`](@ref)`(λ, ε, w)` | `λwᵢ / (ε + t)` | `λwᵢ / ε` |
| [`MCP`](@ref)`(λ, a)` | `max(λ - t/a, 0)` | `λ` |
| [`SCAD`](@ref)`(λ, a)` | `λ` for `t ≤ λ`, `max(aλ - t, 0) / (a - 1)` beyond | `λ` |
| [`GroupMCP`](@ref), [`GroupSCAD`](@ref) | the slope of the MCP or SCAD at level `λwₖ` | `λwₖ` |
| [`NormL1`](@ref)`(λ0, w)`, [`GroupNormL2`](@ref) | `λ0wᵢ` (`λ0wₖ`): the norm is its own linearization | `λ0wᵢ` |

At zero the slope is `pᵢ′(0+)`, so the first stage of LLA from `x = 0` is the
lasso. For the capped ℓ1 penalty at `|v| = a` the weight is `0`, the rule of
the multi-stage convex relaxation of T. Zhang (2010): `λwᵢ` for `|v| < a` and `0`
otherwise. The slopes of the MCP and SCAD are continuous away from zero.
An infinite entry weight gives the LLA weight `Inf` at every point, so the next
stage keeps that entry at zero, as the penalty does.

[`NormLq`](@ref) has an infinite slope at zero, so every stage would keep the
zero entries at zero; it throws an `ArgumentError`, as do functions that are
not of this form.
"""
function lla_weight end

"""
    lla_weights!(w, f, x)
    lla_weights(f, x)

Store in `w` the weights of the local linear approximation (LLA) of the
nonconvex penalty `f` at `x`, and return `w`. The next stage of LLA, the
weighted lasso that majorizes `f` at `x` up to a constant, has the penalty

- `NormL1(1, w)` for a separable `f` ([`CappedL1`](@ref), [`LogPenalty`](@ref),
  [`MCP`](@ref), [`SCAD`](@ref)): `w` has the size of `x`, and
  `w[i] = `[`lla_weight`](@ref)`(f, x[i], i)`, the slope `pᵢ′(|xᵢ|)` from the
  right;
- `GroupNormL2(1, f.groups, w)` for [`GroupMCP`](@ref) and [`GroupSCAD`](@ref):
  `w` has one entry per group, `w[k] = pₖ′(‖x[Gₖ]‖₂)`.

The weights include the level `λ` and the entry or group weights of `f`; to
keep the level apart, divide them by `λ` and use `NormL1(λ, w)`. `x` may be a
[`SparseIterate`](@ref), a [`SymmetricSparseIterate`](@ref) (with a `p × p`
array `w`) or an [`AtomIterate`](@ref). `lla_weights!` allocates nothing;
`lla_weights(f, x)` allocates `w`.

Each LLA stage is convex, so its minimizer, λmax, the strong rule and safe
screening are those of a weighted lasso. Alternating stages and weights is a
majorization-minimization algorithm: each stage solved exactly decreases the
loss plus `f`.

# Example
```julia
f = MCP(0.5, 3)
w = lla_weights(f, [0.0, 0.6, 2.0])     # [0.5, 0.3, 0.0]
NormL1(1, w)                            # the penalty of the next stage
```
"""
function lla_weights! end

@doc (@doc lla_weights!) lla_weights

@noinline _no_lla(f) =
  throw(ArgumentError("$(nameof(typeof(f))) has no LLA weights; they are defined for the penalties Σᵢ pᵢ(|xᵢ|) with pᵢ concave on [0, ∞): CappedL1, LogPenalty, MCP, SCAD, GroupMCP and GroupSCAD (and NormL1 and GroupNormL2, their own linearizations)"))
@noinline _no_lla(::NormLq) =
  throw(ArgumentError("NormLq has an infinite slope at zero, so its LLA weights at zero are infinite and every stage keeps the zero entries at zero; use LogPenalty, whose slope at zero is finite, or the proximal operator of NormLq"))

lla_weight(f::AbstractFunction, ::Real, ::Integer) = _no_lla(f)
lla_weights!(::AbstractArray, f::AbstractFunction, ::AbstractArray) = _no_lla(f)

####################################
#  separable penalties
####################################

Base.@propagate_inbounds function lla_weight(f::NormL1, v::T, i::Integer) where {T<:Real}
  S = float(T)
  return _weight_coefficient(S(f.λ0), S(_entry(f.w, i)))
end

Base.@propagate_inbounds function lla_weight(f::CappedL1, v::T, i::Integer) where {T<:Real}
  S = float(T)
  w = S(_entry(f.w, i))
  c = _weight_coefficient(S(f.λ), w)
  return ifelse((abs(S(v)) < S(f.a)) | isinf(w), c, zero(S))
end

Base.@propagate_inbounds function lla_weight(f::LogPenalty, v::T, i::Integer) where {T<:Real}
  S = float(T)
  return _weight_coefficient(S(f.λ), S(_entry(f.w, i))) / (S(f.ε) + abs(S(v)))
end

function lla_weight(f::MCP, v::T, ::Integer) where {T<:Real}
  S = float(T)
  return max(S(f.λ) - abs(S(v)) / S(f.a), zero(S))
end

function lla_weight(f::SCAD, v::T, ::Integer) where {T<:Real}
  S = float(T)
  λ, a, t = S(f.λ), S(f.a), abs(S(v))
  t <= λ && return λ
  return max(a * λ - t, zero(S)) / (a - 1)
end

lla_weight(f::NormLq, ::Real, ::Integer) = _no_lla(f)
lla_weights!(::AbstractArray, f::NormLq, ::AbstractArray) = _no_lla(f)

# entry by entry over the dense array; a sparse iterate reads its entries in O(1)
function lla_weights!(w::AbstractArray, f::SeparableFunction, x::AbstractArray)
  _checksize(w, x)
  _checkparams(f, x)
  T = eltype(w)
  @inbounds for i in eachindex(IndexLinear(), w, x)
    w[i] = convert(T, lla_weight(f, x[i], i))
  end
  return w
end

lla_weights(f::AbstractFunction, x::AbstractArray) = lla_weights!(_gradient_buffer(x), f, x)

####################################
#  group penalties
####################################

lla_weight(f::GroupNormL2, v::T, k::Integer) where {T<:Real} = float(T)(_weight_coefficient(f.λ0, f.w[k]))
lla_weight(f::_RadialGroupPenalty, v::T, k::Integer) where {T<:Real} =
  isinf(_weight_coefficient(f.λ0, f.w[k])) ? float(T)(Inf) : lla_weight(_group_scalar(f, float(T), k), v, 1)

function lla_weights!(w::AbstractArray, f::Union{GroupNormL2, _RadialGroupPenalty}, x::AbstractArray)
  length(w) == length(f.groups) ||
    throw(DimensionMismatch("the weights have length $(length(w)) and there are $(length(f.groups)) groups"))
  _checkgroups(f, x)
  T = eltype(w)
  for k in eachindex(f.groups)
    w[k] = convert(T, lla_weight(f, _group_norm(f, x, f.groups[k]), k))
  end
  return w
end

lla_weights(f::Union{GroupNormL2, _RadialGroupPenalty}, x::AbstractArray) =
  lla_weights!(Vector{float(eltype(x))}(undef, length(f.groups)), f, x)
