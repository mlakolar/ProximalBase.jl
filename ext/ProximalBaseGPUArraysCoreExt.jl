module ProximalBaseGPUArraysCoreExt

# The separable functions, the norms and balls, the group functions and the
# losses with a design matrix on GPU arrays. The dense paths loop over linear
# indices and read and write one entry at a time, which a GPU array forbids;
# here they are broadcasts and reductions, which run as kernels. A function
# object enters a kernel through Ref(f), so it must be a bits type there: the
# built-in functions that hold parameter arrays (weights, bounds, a center)
# broadcast over those arrays instead, and never enter a kernel themselves.

using ProximalBase
using ProximalBase: NormL1, NormL1Nonneg, ElasticNet, IndBox, IndNonnegative, SqrNormL2, NormL2,
                    NormLinf, IndBallL1, IndSimplex,
                    GroupNormL2, GroupNormLinf, SparseGroupNormL2, IndBallGroupL0, AbstractFunction,
                    LeastSquaresLoss, GLMLoss, Regularize, BoxConstrained,
                    prox_entry, value_entry, shrink, _checksize, _checkparams, _checkgroups, _tol, _ind,
                    _kth_largest!, _level_fits, _level_product, _scaled_product, _positive_product,
                    _weight_coefficient, _weight_product, _weighted_value, _value_at, _valtype, _checkreal,
                    _reshapeparam
using GPUArraysCore: AbstractGPUArray, AnyGPUArray
using LinearAlgebra: mul!
using Base.Broadcast: broadcasted, instantiate
import Adapt

const GPUArray = AbstractGPUArray

# similar allocates the parent array kind even for a wrapped device input.
_device_buffer(x, ::Type{T}) where {T} = similar(x, T, size(x))
_device_buffer_type(::Type{X}, ::Type{T}) where {X, T} = Base.promote_op(_device_buffer, X, Type{T})
function ProximalBase._checkbuffer(s::ProximalBase.Scratch, ::Type{T}, x::AnyGPUArray) where {T}
  B = _device_buffer_type(typeof(x), T)
  b = s.buf
  b isa B && size(b) == size(x) && return b
  _device_buffer(x, T)::B
end
function ProximalBase._array_buffer!(s::ProximalBase.Scratch, ::Type{T}, x::AnyGPUArray) where {T}
  b = ProximalBase._checkbuffer(s, T, x)
  s.buf = b
  b
end
ProximalBase._tuple_buffer_type(::Type{X}) where {T, N, X<:AnyGPUArray{T, N}} =
  _device_buffer_type(X, float(T))
ProximalBase._gradient_buffer(x::AnyGPUArray) = _device_buffer(x, float(eltype(x)))

# Device storage can be dense without being accessible to host BLAS.
ProximalBase._is_device_array(::GPUArray) = true
ProximalBase._blas_ready(::GPUArray{<:Any, 2}) = false

# the sum of the lazily broadcast f.(args...), as one reduction on the device
_sum(::Type{T}, f::F, args...) where {T, F} = mapreduce(identity, +, instantiate(broadcasted(f, args...)); init = zero(T))

# whether p(args...) holds for every entry, as one reduction on the device
_all(p::P, args...) where {P} = mapreduce(identity, &, instantiate(broadcasted(p, args...)); init = true)

# a parameter array, or `true` (a scalar 1 in a broadcast) when there is none
_param(::Nothing) = true
_param(w) = w

####################################
#  Any separable function that is a bits type, through its scalar proxes
####################################

function ProximalBase._prox_sep!(y::AnyGPUArray, f, x::AnyGPUArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkparams(f, x)
  y .= prox_entry.(Ref(f), x, LinearIndices(x), γ)
  y
end

# Reciprocal steps stay a lazy device broadcast, without indexing their wrapper.
_steps(γ::AbstractArray, x) = _reshapeparam(γ, x)
_steps(γ::ProximalBase._ReciprocalArray, x) = broadcasted(inv, _steps(γ.γ, x))
function ProximalBase._prox_sep!(y::AnyGPUArray, f, x::AnyGPUArray, γ::AbstractArray)
  _checkreal(f, γ)
  ProximalBase._checkwrite(prox!, f, y, x)
  ProximalBase._checkparam(γ, x)
  y .= prox_entry.(Ref(f), x, LinearIndices(x), _steps(γ, x))
  y
end
# Parameter arrays cannot be captured in the kernel's function object.
_enet_prox(v::T, γ, λ1, λ2, w) where {T} =
  shrink(v, convert(float(T), _weight_coefficient(γ, λ1, w))) / (1 + convert(float(T), γ * λ2))
function ProximalBase._prox_sep!(y::AnyGPUArray, f::ElasticNet, x::AnyGPUArray, γ::AbstractArray)
  _checkreal(f, γ)
  ProximalBase._checkwrite(prox!, f, y, x)
  ProximalBase._checkparam(γ, x)
  y .= _enet_prox.(x, _steps(γ, x), _device_level(float(eltype(y)), f.λ1), _device_level(float(eltype(y)), f.λ2), _reshapeparam(_param(f.w), x))
  y
end

_term(f, v, i, ::Type{T}) where {T} = convert(T, value_entry(f, v, i))
ProximalBase._value_dense(f, x::GPUArray) =
  (_checkreal(f, x); T = _valtype(x); _sum(T, _term, Ref(f), x, LinearIndices(x), T))

####################################
#  The built-in functions with parameter arrays
####################################

# the step and level in the element type of the output, so that a Float32
# array is not computed in Float64
_scalar(y, a) = convert(float(eltype(y)), a)

# Keep a wider isbits level until its weight or step has been applied.
function _device_level(::Type{T}, λ) where {T}
  converted = convert(T, λ)
  _level_preserved(T, λ) && converted == λ && return converted
  R = promote_type(T, float(typeof(λ)))
  convert(isbitstype(R) ? R : promote_type(T, Float64), λ)
end

function ProximalBase._prox_sep!(y::AnyGPUArray, f::NormL1, x::AnyGPUArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkparams(f, x)
  y .= shrink.(x, _WeightLevel(_group_levels(float(eltype(y)), γ, f.λ0)...).(_reshapeparam(_param(f.w), x)))
end
# Keep the ordinary reduction; recover exceptional sums in power-of-two units.
function ProximalBase._value_dense(f::NormL1, x::GPUArray)
  _checkreal(f, x)
  T = float(eltype(x))
  f.w === nothing && iszero(f.λ0) && return zero(T)
  _level_fits(T, f.λ0) || return _scaled_l1_value(f.λ0, x, _reshapeparam(_param(f.w), x))
  v = _sum(T, _WeightValue(_scalar(x, f.λ0)), instantiate(broadcasted(abs, x)), _reshapeparam(_param(f.w), x))
  isfinite(v) || iszero(f.λ0) || !isfinite(f.λ0) || !_all(isfinite, x) ? v : _scaled_l1_value(f.λ0, x, _reshapeparam(_param(f.w), x))
end

# The device accumulates in T; only the host combines an unconverted level.
ProximalBase._converted_l1_value(λ, x::GPUArray) = _scaled_l1_value(λ, x, true)
@noinline function _scaled_l1_value(λ, x, w)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)))
  m = mapreduce(abs, max, x; init = zero(T))
  isfinite(m) || return convert(T, convert(R, λ) * _sum(T, _WeightValue(one(T)), instantiate(broadcasted(abs, x)), w))
  k = ProximalBase._magnitude_exponent(m)
  scales = ProximalBase._power_factors(T, -k)
  n = _sum(T, _ScaledWeightValue(_WeightValue(one(T)), scales), x, w)
  convert(T, _scaled_product(convert(R, λ), convert(R, n), k))
end

function ProximalBase._prox_sep!(y::AnyGPUArray, f::NormL1Nonneg, x::AnyGPUArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkparams(f, x)
  y .= max.(x .- _WeightLevel(_group_levels(float(eltype(y)), γ, f.λ0)...).(_reshapeparam(_param(f.w), x)), 0)
end
function ProximalBase._value_dense(f::NormL1Nonneg, x::GPUArray)
  _checkreal(f, x)
  T = float(eltype(x))
  t = _tol(T)
  _all(v -> v >= -t, x) || return T(Inf)
  f.w === nothing && iszero(f.λ0) && return zero(T)
  _sum(T, _WeightValue(_scalar(x, f.λ0)), instantiate(broadcasted(max, x, zero(T))), _reshapeparam(_param(f.w), x))
end

function ProximalBase._prox_sep!(y::AnyGPUArray, f::ElasticNet, x::AnyGPUArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkparams(f, x)
  y .= shrink.(x, _weight_coefficient.(_scalar(y, γ * f.λ1), _reshapeparam(_param(f.w), x))) ./ (1 + _scalar(y, γ * f.λ2))
end
function ProximalBase._value_dense(f::ElasticNet, x::GPUArray)
  _checkreal(f, x)
  T = float(eltype(x))
  λ1 = _scalar(x, f.λ1)
  # the weights in the element type; a level that is not positive and finite
  # enters every coefficient
  l1 = f.w === nothing && iszero(f.λ1) ? zero(T) :
       (λ1 > 0) & isfinite(λ1) ? λ1 * _sum(T, _EnetValue(one(T)), instantiate(broadcasted(abs, x)), _reshapeparam(_param(f.w), x)) :
       _sum(T, _EnetValue(λ1), instantiate(broadcasted(abs, x)), _reshapeparam(_param(f.w), x))
  if isfinite(f.λ1) && !iszero(f.λ1) && (!_level_fits(T, f.λ1) || !isfinite(l1))
    l1 = _scaled_l1_value(f.λ1, x, _reshapeparam(_param(f.w), x))
  end
  q = iszero(f.λ2) ? zero(T) : _sum(T, abs2, x)
  l2 = iszero(f.λ2) ? zero(T) : _scalar(x, f.λ2) * q / 2
  if !iszero(f.λ2) && (!_level_fits(T, f.λ2) || !isfinite(l2) || iszero(l2) || issubnormal(l2) || issubnormal(q))
    l2 = _scaled_square_value(f.λ2, x)
  end
  # NaN entries propagate even when both levels vanish.
  f.w === nothing && iszero(f.λ1) && iszero(f.λ2) && return _all(!isnan, x) ? zero(T) : T(NaN)
  l1 + l2
end

function ProximalBase._prox_sep!(y::AnyGPUArray, f::IndBox, x::AnyGPUArray, ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkparams(f, x)
  y .= clamp.(x, _reshapeparam(f.l, x), _reshapeparam(f.u, x))
end
# whether l ≤ v ≤ u within the tolerance t of the indicator functions, as on
# the CPU; a functor rather than a closure, so that Julia 1.10 infers its type
struct _InBox{T}
  t::T
end
(b::_InBox)(v, l, u) = ProximalBase._inbox(v, l, u, b.t)

function ProximalBase._value_dense(f::IndBox, x::GPUArray)
  _checkreal(f, x)
  T = float(eltype(x))
  _all(_InBox(_tol(T)), x, _reshapeparam(f.l, x), _reshapeparam(f.u, x)) ? zero(T) : T(Inf)
end

ProximalBase._prox_sep!(y::AnyGPUArray, f::IndNonnegative, x::AnyGPUArray, ::Real) = (_checkreal(f, x); _checkreal(f, y); _checksize(y, x); y .= max.(x, 0))
function ProximalBase._value_dense(f::IndNonnegative, x::GPUArray)
  _checkreal(f, x)
  T = float(eltype(x))
  t = _tol(T)
  _all(v -> v >= -t, x) ? zero(T) : T(Inf)
end

# the value of SqrNormL2 with a center: Σ (xᵢ - yᵢ)² as one reduction
function ProximalBase._sumabs2_diff(x::GPUArray, y::GPUArray)
  size(x) == size(y) || throw(DimensionMismatch("the arrays have sizes $(size(x)) and $(size(y))"))
  T = float(promote_type(eltype(x), eltype(y)))
  _sum(T, (a, b) -> abs2(a - b), x, y)
end

function ProximalBase._norm_diff2(x::GPUArray, y::GPUArray)
  s = ProximalBase._sumabs2_diff(x, y)
  isfinite(s) && s >= floatmin(typeof(s)) && return sqrt(s)
  T = typeof(s)
  m = mapreduce(identity, max, instantiate(broadcasted((a, b) -> abs(a - b), x, y)); init = zero(T))
  (iszero(m) || !isfinite(m)) && return m
  m * sqrt(_sum(T, (a, b) -> abs2((a - b) / m), x, y))
end


# Conjugate values use the CPU tolerances and support-function conventions.
struct _InConjugateL1{T, L}
  λ::L
end
(p::_InConjugateL1{T})(v, w) where {T} = ProximalBase._within(abs(v), _weight_coefficient(p.λ, w), T)
struct _BoxSupport{T} end
(::_BoxSupport{T})(v, l, u) where {T} = convert(T, v >= 0 ? (iszero(v) ? zero(T) : v * u) : v * l)
function ProximalBase._conjugate_value(f::NormL1, y::AnyGPUArray)
  ProximalBase._checkconjugate(f, y)
  T = float(eltype(y))
  λ = _device_level(T, f.λ0)
  _ind(_all(_InConjugateL1{T, typeof(λ)}(λ), y, _reshapeparam(_param(f.w), y)), T)
end
function ProximalBase._conjugate_value(f::IndNonnegative, y::AnyGPUArray)
  ProximalBase._checkconjugate(f, y)
  T = float(eltype(y))
  t = _tol(T)
  _ind(_all(v -> v <= t, y), T)
end
function ProximalBase._conjugate_value(f::IndBox, y::AnyGPUArray)
  ProximalBase._checkconjugate(f, y)
  T = float(eltype(y))
  _sum(T, _BoxSupport{T}(), y, _reshapeparam(_param(f.l), y), _reshapeparam(_param(f.u), y))
end

####################################
#  Norms, balls and the simplex
####################################

# NormL2 and IndBallL2 use the generic methods and GPUArrays' scaled norm.
# NormL2 also needs device reductions for an overflowing norm or weighted value.
# Spectral decompositions already run on the host; retain that boundary for
# the exceptional trace reduction too.
ProximalBase._scaled_trace_value(f::ProximalBase.TracePSD, X::GPUArray{<:Any, 2}) =
  (_checkreal(f, X); ProximalBase._scaled_trace_value(f, Array(X)))

# The ℓ∞ norm, the ℓ1 ball and the simplex clip or shrink the entries at the level τ with
# Σᵢ max(g(xᵢ) - τ, 0) = r, for g = abs or the identity, which the CPU finds by
# a selection. Michelot's algorithm finds it with reductions alone: τ is the
# level that makes the entries above the last level sum to r, so it only
# grows, and the entries at or below it drop out until none does, after at
# most n passes and in practice a few.

function ProximalBase._scaled_norm2_value(λ, x::GPUArray)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)))
  m = mapreduce(abs, max, x; init = zero(T))
  isfinite(m) || return convert(T, convert(R, λ) * m)
  k = ProximalBase._magnitude_exponent(m)
  s = ProximalBase._power_factors(T, -k)
  n = sqrt(_sum(T, v -> abs2(ProximalBase._scale_value(v, s)), x))
  convert(T, _scaled_product(convert(R, λ), convert(R, n), k))
end

# Apply the squared norm's power and half-weight on the host.
@noinline function _scaled_square_value(λ, x)
  T = float(eltype(x))
  R = promote_type(T, float(typeof(λ)))
  m = mapreduce(abs, max, x; init = zero(T))
  isfinite(m) || return convert(T, _positive_product(convert(R, λ), convert(R, m), convert(R, m)))
  k = ProximalBase._magnitude_exponent(m)
  s = ProximalBase._power_factors(T, -k)
  n = _sum(T, v -> abs2(ProximalBase._scale_value(v, s)), x)
  convert(T, _scaled_product(convert(R, λ), convert(R, n) / 2, 2k))
end

function ProximalBase._norm2_shrink!(f::NormL2, y::GPUArray, x::GPUArray, c)
  _checkreal(f, x)
  _checkreal(f, y)
  T = float(eltype(x))
  scale = mapreduce(abs, max, x; init = zero(T))
  if isfinite(scale) && scale > one(T)
    inverse = inv(scale)
    nx = sqrt(_sum(T, v -> abs2(v * inverse), x))
    threshold = c / scale
    if nx > threshold
      t = one(nx) - threshold / nx
      y .= t .* (x .* inverse)
      y .*= scale
      return (nx - threshold) * scale
    end
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  y .= (one(T) - c / scale) .* x
  scale - c
end

# g(v) when it exceeds τ, and 0 otherwise; whether it exceeds τ
struct _Above{G, T}
  g::G
  τ::T
end
(a::_Above)(v) = (u = a.g(v); u > a.τ ? u : zero(u))
struct _CountAbove{G, T}
  g::G
  τ::T
end
(a::_CountAbove)(v) = Int(a.g(v) > a.τ)

function _level(x::GPUArray, r::T, g::G) where {T, G}
  c = length(x)
  τ = (_sum(T, _Above(g, T(-Inf)), x) - r) / c
  while true
    m = _sum(Int, _CountAbove(g, τ), x)
    (m >= c || m == 0) && return τ                   # no entry dropped out
    c = m
    τ = (_sum(T, _Above(g, τ), x) - r) / c
  end
end

(f::NormLinf)(x::GPUArray) = (_checkreal(f, x); ProximalBase._weighted_normlinf(f.λ, x))
function ProximalBase._weighted_normlinf(λ::Real, x::GPUArray)
  T = float(eltype(x))
  n = mapreduce(abs, max, x; init = zero(T))
  _level_fits(T, λ) && return convert(T, λ) * n
  R = promote_type(T, float(typeof(λ)))
  convert(T, convert(R, λ) * convert(R, n))
end

function ProximalBase.value_and_prox!(y::GPUArray, f::NormLinf, x::GPUArray, γ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  T = float(eltype(y))
  if iszero(f.λ)
    y === x || copyto!(y, x)
    return zero(T)
  end
  c = _level_product_device(T, γ, f.λ)
  if _sum(T, abs, x) <= c
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  τ = _level(x, c, abs)
  y .= sign.(x) .* min.(abs.(x), τ)
  return _level_preserved(T, f.λ) ? convert(T, convert(T, f.λ) * τ) : f(y)
end

function (f::IndBallL1)(x::GPUArray)
  _checkreal(f, x)
  T = float(eltype(x))
  _sum(T, abs, x) <= f.r + _tol(T) * max(one(T), f.r) ? zero(T) : T(Inf)
end

function ProximalBase.prox!(y::GPUArray, f::IndBallL1, x::GPUArray, ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  T = float(eltype(y))
  r = convert(T, f.r)
  if _sum(T, abs, x) <= r
    y === x || copyto!(y, x)
    return y
  end
  iszero(r) && return fill!(y, zero(eltype(y)))
  y .= shrink.(x, _level(x, r, abs))
end

function ProximalBase.prox!(y::GPUArray, f::IndSimplex, x::GPUArray, ::Real)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  T = float(eltype(y))
  r = convert(T, f.r)
  if !f.equality && _sum(T, v -> max(v, zero(v)), x) <= r
    y .= max.(x, zero(T))
    return y
  end
  iszero(r) && return fill!(y, zero(eltype(y)))
  y .= max.(x .- _level(x, r, identity), zero(T))
end

####################################
#  Group norms: segmented reductions over the groups
####################################

# A group norm reduces over the entries of each group and broadcasts a number
# per group (a factor or a level) back to its entries. On the device, the
# groups whose sizes lie in one class (1, 2, 3–4, 5–8, …) are the columns of an
# index matrix P, padded with the group's first index and masked by the
# group's size; the gather x[P] and one reduction along its columns give the
# norms of all the groups of the class. A vector of group ids carries the
# numbers back to the entries, by one gather and one broadcast. The padding
# at most doubles a class, and the number of passes grows with the number of
# size classes, the logarithm of the largest group, not with the number of
# groups. The layout is built on the host from the groups: once by
# adapt(CuArray, f), or at each call on a function left on the host.

# the groups of one size class: column j of P holds the group at position
# pos[j] of the order of the classes
struct _GroupBlock{M, R, W}
  P::M                  # m × K_c indices
  len::R                # 1 × K_c: the sizes of the groups
  w::W                  # 1 × K_c: their weights, or nothing
  pos::UnitRange{Int}
end

struct _GroupLayout{B, V}
  blocks::Vector{B}
  gid::V                # gid[i] = the position of the group of index i, K + 1 if it is in none
  pos::Vector{Int}      # pos[k] = the position of group k (on the host)
end

_sizeclass(n::Int) = n <= 1 ? 0 : ndigits(n - 1; base = 2)       # ⌈log₂ n⌉

# dev moves a host array to the device
function _layout(groups, group_of::Vector{Int}, w, dev::D) where {D}
  K = length(groups)
  sizes = [length(G) for G in groups]
  class = _sizeclass.(sizes)
  order = sortperm(class; alg = MergeSort)        # stable: the groups of a class in their order
  pos = invperm(order)
  starts = [j for j in 1:K if j == 1 || class[order[j]] != class[order[j-1]]]
  blocks = map(eachindex(starts)) do c
    r = starts[c]:(c == length(starts) ? K : starts[c+1] - 1)
    ks = order[r]
    m = maximum(sizes[ks])
    P = Matrix{Int}(undef, m, length(ks))
    for (j, k) in enumerate(ks)
      G = groups[k]
      P[:, j] .= first(G)
      P[1:length(G), j] .= G
    end
    _GroupBlock(dev(P), dev(reshape(sizes[ks], 1, :)), w === nothing ? nothing : dev(reshape(w[ks], 1, :)), r)
  end
  gid = [k == 0 ? K + 1 : pos[k] for k in group_of]
  _GroupLayout(blocks, dev(gid), pos)
end

_weights(f::IndBallGroupL0) = nothing
_weights(f) = f.w

const _GroupFunction = Union{GroupNormL2, GroupNormLinf, SparseGroupNormL2, IndBallGroupL0}

# A group function with the layout of its groups on the device, which
# adapt(CuArray, f) returns. On host arrays it is the function itself.
struct DeviceGroups{F<:_GroupFunction, L<:_GroupLayout} <: AbstractFunction
  f::F
  layout::L
end
DeviceGroups(f::_GroupFunction, dev) = DeviceGroups(f, _layout(f.groups, f.group_of, _weights(f), dev))

ProximalBase.isproximable(::Type{<:DeviceGroups{F}}) where {F} = ProximalBase.isproximable(F)
ProximalBase.isconvex(::Type{<:DeviceGroups{F}}) where {F} = ProximalBase.isconvex(F)
ProximalBase._isindicator(::Type{<:DeviceGroups{F}}) where {F} = ProximalBase._isindicator(F)
function ProximalBase._fold_level(d::DeviceGroups, α)
  f = ProximalBase._fold_level(d.f, α)
  f === nothing ? nothing : DeviceGroups(f, d.layout)
end

function ProximalBase._zero_scaled(d::DeviceGroups)
  f = ProximalBase._zero_scaled(d.f)
  f isa ZeroFunction ? f : DeviceGroups(f, d.layout)
end

(d::DeviceGroups)(x::AbstractArray) = (_checkreal(d, x); d.f(x))
ProximalBase.prox!(y::AbstractArray, d::DeviceGroups, x::AbstractArray, γ::Real) = (_checkreal(d, x); _checkreal(d, y); prox!(y, d.f, x, γ))
ProximalBase.value_and_prox!(y::AbstractArray, d::DeviceGroups, x::AbstractArray, γ::Real) = (_checkreal(d, x); _checkreal(d, y); value_and_prox!(y, d.f, x, γ))

(d::DeviceGroups)(x::GPUArray) = (_checkreal(d, x); _value(d.f, d.layout, x))
ProximalBase.value_and_prox!(y::GPUArray, d::DeviceGroups, x::GPUArray, γ::Real) = (_checkreal(d, x); _checkreal(d, y); _group_prox!(y, d.f, d.layout, x, γ, Val(true)))
ProximalBase.prox!(y::GPUArray, d::DeviceGroups, x::GPUArray, γ::Real) = (_checkreal(d, x); _checkreal(d, y); _group_prox!(y, d.f, d.layout, x, γ, Val(false)); y)

# a function left on the host builds the layout on the device of x at each call
_on_device(f, x) = DeviceGroups(f, A -> copyto!(similar(x, eltype(A), size(A)), A))
for F in (:GroupNormL2, :GroupNormLinf, :SparseGroupNormL2, :IndBallGroupL0)
  @eval begin
    (f::$F)(x::GPUArray) = (_checkreal(f, x); _on_device(f, x)(x))
    ProximalBase.prox!(y::GPUArray, f::$F, x::GPUArray, γ::Real) = (_checkreal(f, x); _checkreal(f, y); prox!(y, _on_device(f, x), x, γ))
    ProximalBase.value_and_prox!(y::GPUArray, f::$F, x::GPUArray, γ::Real) = (_checkreal(f, x); _checkreal(f, y); value_and_prox!(y, _on_device(f, x), x, γ))
  end
end

# h(x[i], a...) where j ≤ len (an entry of the group), and z on the padding
struct _Masked{H, T}
  h::H
  z::T
end
(m::_Masked{H, T})(v, j, len, a...) where {H, T} = j <= len ? convert(T, m.h(v, a...)) : m.z

# the gather of the columns of block b from the vector xv
_gather(xv, b::_GroupBlock) = xv[b.P]

# op over h(x[i], a...) for i in each group of block b, from its gather X, as a 1 × K_c row
_segreduce(h::H, op::O, z, X, b::_GroupBlock, a...) where {H, O} =
  mapreduce(identity, op, instantiate(broadcasted(_Masked(h, z), X, 1:size(X, 1), b.len, a...)); dims = 1, init = z)

# y[i] ← op(y[i], a[gid[i]], gid[i]) for the indices up to the largest in a group
function _scatter_back!(op::O, yv, lay::_GroupLayout, a) where {O}
  yl = view(yv, 1:length(lay.gid))
  yl .= op.(yl, a[lay.gid], lay.gid)
  return yv
end

# Ordinary group thresholds, norms and factors use the output type T, with
# step and level converted on the host. Exceptional weights retain their
# stored precision until the complete product is rounded.
struct _Level{T}
  step::T
  λ::T
end
# Only a stored infinite weight fixes a group. Retry finite weights that do
# not fit the kernel type before rounding the complete product.
@inline function _group_threshold(step::T, λ::T, w) where {T}
  isinf(w) && return T(Inf)
  _level_fits(T, w) && return _positive_product(step, λ, convert(T, w))
  R = promote_type(T, float(typeof(w)))
  convert(T, _positive_product(convert(R, step), convert(R, λ), convert(R, w)))
end
(l::_Level{T})(w) where {T} = _group_threshold(l.step, l.λ, w)

# Exact conversions retain subnormal factors for the kernel's product recovery.
_level_preserved(::Type{T}, a) where {T} = _level_fits(T, a) || convert(T, a) == a
_level_product_device(::Type{T}, γ, λ) where {T} =
  _level_preserved(T, γ) && _level_preserved(T, λ) ? convert(T, γ) * convert(T, λ) : _level_product(T, γ, λ)

# Combine converted step/level factors before entering a kernel.
function _group_levels(::Type{T}, γ, λ) where {T}
  _level_preserved(T, γ) && _level_preserved(T, λ) && return (convert(T, γ), convert(T, λ))
  R = promote_type(T, float(typeof(γ)), float(typeof(λ)))
  c = convert(R, γ) * convert(R, λ)
  _level_fits(T, c) && return (one(T), convert(T, c))
  m, e = frexp(c)
  a = ldexp(one(R), fld(e, 2))
  b = ldexp(m, e - fld(e, 2))
  _level_fits(T, a) && _level_fits(T, b) ? (convert(T, a), convert(T, b)) : (one(T), convert(T, c))
end

# the threshold and value of an entry of a weighted ℓ1 norm, with infinite and
# zero weights as in prox_entry and value_entry
struct _WeightLevel{T}
  step::T
  λ::T
end
(l::_WeightLevel{T})(w) where {T} = _weight_product(l.step, l.λ, convert(T, w))
struct _WeightValue{T}
  λ::T
end
(l::_WeightValue{T})(v, w) where {T} = (w = convert(T, w); p = _positive_product(l.λ, w, v); ifelse(isinf(w) & !iszero(v), oftype(p, Inf) * v, p))
# A functor keeps the scaled weighted reduction inferred on Julia 1.10.
struct _ScaledWeightValue{T, S}
  weight::_WeightValue{T}
  scales::S
end
(l::_ScaledWeightValue)(v, w) = l.weight(abs(ProximalBase._scale_value(v, l.scales)), w)
# the ℓ1 term of ElasticNet at |v|, as its value_entry; the level is one when
# it multiplies the sum instead
struct _EnetValue{T}
  λ::T
end
(l::_EnetValue{T})(a, w) where {T} = (w = convert(T, w); _weighted_value(_weight_coefficient(l.λ, w) * a, a))

# λwₖ ⋅ v and group shrinkage at γλwₖ: its factor and value at the proximal point
struct _Weighted{T}
  λ::T
end
function (l::_Weighted{T})(v, w) where {T}
  isinf(w) && return convert(T, _weighted_value(w * v, v))
  _level_fits(T, w) && return _positive_product(l.λ, convert(T, w), v)
  R = promote_type(T, float(typeof(w)))
  convert(T, _positive_product(convert(R, l.λ), convert(R, w), convert(R, v)))
end
struct _L2Factor{T}
  step::T
  λ::T
end
(l::_L2Factor{T})(n, w) where {T} = iszero(w) ? one(T) :
  (c = _group_threshold(l.step, l.λ, w); n > c ? one(T) - c / n : zero(T))
struct _L2Value{T}
  step::T
  λ::T
end
(l::_L2Value{T})(n, w) where {T} = iszero(w) ? zero(T) :
  (c = _group_threshold(l.step, l.λ, w); n > c ? _Weighted(l.λ)(n - c, w) : zero(T))

# Scale each group before squaring. A direct sum of squares can overflow or
# underflow even when its norm, and hence its proximal factor, is representable.
struct _AbsMapped{H}
  h::H
end
(a::_AbsMapped)(v) = abs(a.h(v))
struct _ScaledNormSquare{H}
  h::H
end
(a::_ScaledNormSquare)(v, m) = iszero(m) ? zero(m) : abs2(a.h(v) / m)
function _group_norm_parts(X, b::_GroupBlock, ::Type{T}, h::H = identity) where {T, H}
  m = _segreduce(_AbsMapped(h), max, zero(T), X, b)
  s = _segreduce(_ScaledNormSquare(h), +, zero(T), X, b, m)
  m, s
end
function _group_norms(X, b::_GroupBlock, ::Type{T}, h::H = identity) where {T, H}
  m, s = _group_norm_parts(X, b, T, h)
  m .* sqrt.(s)
end

# An overflowing finite norm still has a representable shrink factor.
function (l::_L2Factor{T})(n, w, m, s) where {T}
  isfinite(n) && return l(n, w)
  iszero(w) && return one(T)
  c = _group_threshold(l.step, l.λ, w)
  isfinite(m) && isfinite(c) ? max(one(T) - (c / m) / sqrt(s), zero(T)) : l(n, w)
end

# |v| for an index in one of the K groups
struct _AbsIn{T}
  K::Int
end
(a::_AbsIn{T})(v, g) where {T} = g <= a.K ? convert(T, abs(v)) : zero(T)

# λΣₖ wₖ ‖x[Gₖ]‖₂ and the ℓ1 norm of the entries in groups
function _l2_part(::Type{T}, xv, lay::_GroupLayout, λ) where {T}
  v = zero(T)
  for b in lay.blocks
    X = _gather(xv, b)
    q = _level_fits(T, λ) ? _sum(T, _Weighted(convert(T, λ)), _group_norms(X, b, T), b.w) : T(Inf)
    v += isfinite(q) ? q : _scaled_group_sum(T, λ, X, b, false)
  end
  return v
end
# Keep each group's norm as a significand and power until weighting it.
struct _ScaledGroupValue{T}
  λ::T
  k::Int
end
function (f::_ScaledGroupValue{T})(m, s, w) where {T}
  isinf(w) && return _Weighted(f.λ)(m, w)
  (iszero(m) || !isfinite(m)) && return _Weighted(f.λ)(m, w)
  a, e = frexp(m)
  _level_fits(T, w) && return _scaled_product(f.λ, convert(T, w), a * sqrt(s), e - f.k)
  R = promote_type(T, float(typeof(w)))
  convert(T, _scaled_product(convert(R, f.λ), convert(R, w), convert(R, a) * sqrt(convert(R, s)), e - f.k))
end
function _group_exponent(m::T, w) where {T}
  iszero(m) || iszero(w) || !isfinite(m) || !isfinite(w) ? 0 :
    ProximalBase._magnitude_exponent(m) + ProximalBase._magnitude_exponent(w)
end

@noinline function _scaled_group_sum(::Type{T}, λ, X, b, linf) where {T}
  m = _segreduce(abs, max, zero(T), X, b)
  s = linf ? one(T) : _segreduce(_ScaledNormSquare(identity), +, zero(T), X, b, m)
  if _level_fits(T, λ)
    return _sum(T, _ScaledGroupValue(convert(T, λ), 0), m, s, b.w)
  end
  k = mapreduce(identity, max, instantiate(broadcasted(_group_exponent, m, b.w)); init = typemin(Int))
  q = _sum(T, _ScaledGroupValue(one(T), k), m, s, b.w)
  R = promote_type(T, float(typeof(λ)))
  convert(T, _scaled_product(convert(R, λ), convert(R, q), k))
end

@noinline function _scaled_grouped_l1(::Type{T}, λ, xv, lay) where {T}
  K = length(lay.pos)
  z = _AbsIn{T}(K).(view(xv, 1:length(lay.gid)), lay.gid)
  _scaled_l1_value(λ, z, true)
end
function _weighted_grouped_l1(::Type{T}, λ, xv, lay) where {T}
  iszero(λ) && return zero(T)
  v = _level_fits(T, λ) ? convert(T, λ) * _grouped_l1(T, xv, lay) : T(Inf)
  isfinite(v) ? v : _scaled_grouped_l1(T, λ, xv, lay)
end

_grouped_l1(::Type{T}, xv, lay::_GroupLayout) where {T} =
  _sum(T, _AbsIn{T}(length(lay.pos)), view(xv, 1:length(lay.gid)), lay.gid)

# the factors of the groups in block order, then the entry for the indices in no group
function _group_buffer(yv, ::Type{T}, lay::_GroupLayout, free) where {T}
  a = similar(yv, T, length(lay.pos) + 1)
  view(a, length(a):length(a)) .= free
  return a
end

# Reduce the layout weights on their device without short-circuiting.
_has_infinite_group_weight(lay::_GroupLayout) = mapreduce(b -> _any_infinite_group_weight(b.w), |, lay.blocks; init = false)
_any_infinite_group_weight(w) = mapreduce(isinf, |, w; init = false)

# GroupNormL2: block soft thresholding
_value(f::GroupNormL2, lay, x) =
  (_checkreal(f, x); _checkgroups(f, x); T = float(eltype(x)); (iszero(f.λ0) && !_has_infinite_group_weight(lay)) ? zero(T) : _l2_part(T, vec(x), lay, f.λ0))

_scale(v, t, _) = iszero(t) ? zero(v) : t * v

# Value reductions return host scalars and can synchronize the device. Compile
# them out for prox!, while sharing the actual proximal calculation.
function _group_prox!(y, f::GroupNormL2, lay, x, γ::Real, ::Val{V}) where {V}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkgroups(f, x)
  S = float(eltype(y))
  y === x || copyto!(y, x)
  (iszero(f.λ0) && !_has_infinite_group_weight(lay)) && return zero(S)
  yv = vec(y)
  fac = _group_buffer(yv, S, lay, one(S))
  step, λ = _group_levels(S, γ, f.λ0)
  v = zero(S)
  for b in lay.blocks
    m, q = _group_norm_parts(_gather(yv, b), b, S)
    s = m .* sqrt.(q)
    view(fac, b.pos) .= _L2Factor(step, λ).(vec(s), vec(b.w), vec(m), vec(q))
    V && (v += _sum(S, _L2Value(step, λ), s, b.w))
  end
  _scatter_back!(_scale, yv, lay, fac)
  return V && (!_level_preserved(S, γ) || !_level_preserved(S, f.λ0) || !isfinite(v)) ? _l2_part(S, yv, lay, f.λ0) : v
end

# SparseGroupNormL2: soft thresholding, then block soft thresholding
function _value(f::SparseGroupNormL2, lay, x)
  _checkreal(f, x)
  _checkgroups(f, x)
  T = float(eltype(x))
  xv = vec(x)
  return ((iszero(f.λ0) && !_has_infinite_group_weight(lay)) ? zero(T) : _l2_part(T, xv, lay, f.λ0)) +
         (iszero(f.λ1) ? zero(T) : _weighted_grouped_l1(T, f.λ1, xv, lay))
end

struct _Shrink{T}
  c::T
end
(s::_Shrink{T})(v) where {T} = shrink(convert(T, v), s.c)
struct _SparseScale{T}
  c::T
  K::Int
end
(s::_SparseScale{T})(v, t, g) where {T} = g <= s.K ? (iszero(t) ? zero(T) : t * shrink(convert(T, v), s.c)) : v

function _group_prox!(y, f::SparseGroupNormL2, lay, x, γ::Real, ::Val{V}) where {V}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkgroups(f, x)
  S = float(eltype(y))
  y === x || copyto!(y, x)
  zero_group = iszero(f.λ0) && !_has_infinite_group_weight(lay)
  zero_group && iszero(f.λ1) && return zero(S)
  yv = vec(y)
  c1 = _level_preserved(S, γ) && _level_preserved(S, f.λ1) ? convert(S, γ * f.λ1) : _level_product_device(S, γ, f.λ1)
  if zero_group
    yl = view(yv, 1:length(lay.gid))
    yl .= _SparseScale(c1, length(lay.pos)).(yl, one(S), lay.gid)
    return V ? _weighted_grouped_l1(S, f.λ1, yv, lay) : zero(S)
  end
  fac = _group_buffer(yv, S, lay, one(S))
  step, λ = _group_levels(S, γ, f.λ0)
  v2 = zero(S)
  for b in lay.blocks
    m, q = _group_norm_parts(_gather(yv, b), b, S, _Shrink(c1))
    s = m .* sqrt.(q)
    view(fac, b.pos) .= _L2Factor(step, λ).(vec(s), vec(b.w), vec(m), vec(q))
    V && (v2 += _sum(S, _L2Value(step, λ), s, b.w))
  end
  _scatter_back!(_SparseScale(c1, length(lay.pos)), yv, lay, fac)
  if V && (!_level_preserved(S, γ) || !_level_preserved(S, f.λ0) || !isfinite(v2))
    v2 = _l2_part(S, yv, lay, f.λ0)
  end
  return v2 + (V && !iszero(f.λ1) ? _weighted_grouped_l1(S, f.λ1, yv, lay) : zero(S))
end

# GroupNormLinf: each group clipped at a level, by Moreau's identity with the
# projection onto an ℓ1 ball, whose level Michelot's algorithm finds for all
# the groups of a block at once, as _level does for one vector
function _value(f::GroupNormLinf, lay, x)
  _checkreal(f, x)
  _checkgroups(f, x)
  T = float(eltype(x))
  (iszero(f.λ0) && !_has_infinite_group_weight(lay)) && return zero(T)
  xv = vec(x)
  v = zero(T)
  for b in lay.blocks
    X = _gather(xv, b)
    q = _level_fits(T, f.λ0) ? _sum(T, _Weighted(convert(T, f.λ0)), _segreduce(abs, max, zero(T), X, b), b.w) : T(Inf)
    v += isfinite(q) ? q : _scaled_group_sum(T, f.λ0, X, b, true)
  end
  return v
end

_sum_above(v, τ) = (a = abs(v); a > τ ? a : zero(a))
_count_above(v, τ) = abs(v) > τ
_clip(v, τ, _) = sign(v) * min(abs(v), τ)

# the levels of the groups of block b, from its gather X, with thresholds c (a
# 1 × K_c row): 0 for a group whose ℓ1 norm is at most its threshold
function _linf_levels(X, b::_GroupBlock, c)
  S = eltype(c)
  s = _segreduce(abs, +, zero(S), X, b)
  done = s .<= c
  τ = ifelse.(done, zero(S), (s .- c) ./ b.len)
  n = copy(b.len)                                 # the entries above the level
  while true
    m = _segreduce(_count_above, +, 0, X, b, τ)
    done .|= (m .>= n) .| (m .== 0)               # no entry dropped out
    _all(identity, done) && return τ
    a = _segreduce(_sum_above, +, zero(S), X, b, τ)
    τ .= ifelse.(done, τ, (a .- c) ./ m)
    n .= ifelse.(done, n, m)
  end
end

function _group_prox!(y, f::GroupNormLinf, lay, x, γ::Real, ::Val{V}) where {V}
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkgroups(f, x)
  S = float(eltype(y))
  y === x || copyto!(y, x)
  (iszero(f.λ0) && !_has_infinite_group_weight(lay)) && return zero(S)
  yv = vec(y)
  levels = _group_buffer(yv, S, lay, S(Inf))
  step, λ = _group_levels(S, γ, f.λ0)
  v = zero(S)
  for b in lay.blocks
    τ = _linf_levels(_gather(yv, b), b, _Level(step, λ).(b.w))
    view(levels, b.pos) .= vec(τ)
    V && (v += _sum(S, _Weighted(λ), τ, b.w))
  end
  _scatter_back!(_clip, yv, lay, levels)
  return V && (!_level_preserved(S, γ) || !_level_preserved(S, f.λ0) || !isfinite(v)) ? _value(f, lay, y) : v
end

# IndBallGroupL0: the number of nonzero groups, and of nonzero entries in a group
function _value(f::IndBallGroupL0, lay, x)
  _checkreal(f, x)
  _checkgroups(f, x)
  xv = vec(x)
  nz, ok = 0, true
  for b in lay.blocks
    c = _segreduce(!iszero, +, 0, _gather(xv, b), b)
    nz += _sum(Int, >(0), c)
    ok &= _all(<=(f.k), c)
  end
  return _ind(ok && nz <= f.s, float(eltype(x)))
end

_keep(v, keep, _) = keep ? v : zero(v)

# The projection computes the norms of the groups on the device and selects
# the s largest on the host, among the K norms, as the CPU does, the earlier
# groups first among ties. With a bound k on the entries of a group, it would
# need a selection inside each group, which runs on the CPU only.
function _group_prox!(y, f::IndBallGroupL0, lay, x, ::Real, ::Val)
  _checkreal(f, x)
  _checkreal(f, y)
  _checksize(y, x)
  _checkgroups(f, x)
  f.k == typemax(Int) ||
    throw(ArgumentError("IndBallGroupL0 with a bound k on the entries of a group projects on the CPU only"))
  S = float(eltype(y))
  y === x || copyto!(y, x)
  yv = vec(y)
  K = length(lay.pos)
  norms = similar(yv, S, K)
  for b in lay.blocks
    view(norms, b.pos) .= vec(_group_norms(_gather(yv, b), b, S))
  end
  keep = _keep_groups(Array(norms), lay.pos, f.s)
  _scatter_back!(_keep, yv, lay, copyto!(similar(yv, Bool, K + 1), keep))
  return zero(S)
end

# which groups the projection keeps, in block order, then true for the indices
# in no group: those of the s largest norms, the earlier groups first among
# ties. The scaled norm is zero exactly for a zero group: a nonzero finite
# group has norm at least its largest magnitude, and NaN/Inf norms are nonzero.
function _keep_groups(norms::Vector{T}, pos::Vector{Int}, s::Int) where {T}
  K = length(pos)
  keep = fill(true, K + 1)
  count(!iszero, norms) <= s && return keep
  q = norms[pos]                                  # in the order of the groups
  τ = s == 0 ? T(Inf) : _kth_largest!(copy(q), s)
  ties = s - count(>(τ), q)
  for k in 1:K
    if q[k] > τ
      continue
    elseif q[k] == τ && ties > 0
      ties -= 1
    else
      keep[pos[k]] = false
    end
  end
  return keep
end

####################################
#  Losses with a design matrix
####################################

# LeastSquaresLoss and GLMLoss keep their buffers next to the response, so that
# adapt(CuArray, f) puts X, y and the buffers on the device. Their products by
# X and X' are mul!, which the device arrays implement; the passes over the
# observations, which loop on the CPU, are reductions and broadcasts here.

ProximalBase._zeros_like(a::GPUArray, ::Type{T}, dims) where {T} = fill!(similar(a, T, dims), zero(T))


# The losses with their buffers on the device. The parameters keep the bounds
# of the types, so that the methods here are more specific than those of the
# CPU.
const _DeviceLeastSquares{T} = LeastSquaresLoss{T, <:AbstractVecOrMat, <:AbstractMatrix, N, <:GPUArray{T, N}} where {N}
const _DeviceGLM{L, T} = GLMLoss{L, T, <:AbstractVector, <:AbstractMatrix, <:Union{Nothing, AbstractVector},
                                 <:Union{Nothing, AbstractVector}, <:GPUArray{T, 1}}

function (f::_DeviceLeastSquares{T})(β::AbstractVecOrMat) where {T}
  _checkreal(f, β)
  mul!(f.tmp, f.X, β)
  return _value_at(convert(T, ProximalBase._sumabs2_diff(f.Y, f.tmp)) / 2, β)
end

# value_and_gradient!, gradient! and hessian_vec! are products and broadcasts,
# and run as they are; the proximal operator needs the SVD of X
_reject_device_leastsquares() =
  throw(ArgumentError("the proximal operator of LeastSquaresLoss uses the SVD of X, by LAPACK, and runs on the CPU only"))
ProximalBase.prox!(y::AbstractVecOrMat, f::_DeviceLeastSquares, x::AbstractVecOrMat, ::Real) =
  (_checkreal(f, x); _checkreal(f, y); _reject_device_leastsquares())
ProximalBase._checkwrite(::Union{typeof(prox!), typeof(value_and_prox!)}, f::_DeviceLeastSquares,
                        y::AbstractVecOrMat, x::AbstractVecOrMat) =
  (_checkreal(f, x); _checkreal(f, y); _reject_device_leastsquares())

# wᵢ ℓ(ηᵢ, yᵢ) and wᵢ ℓ'(ηᵢ, yᵢ) in type T, as the loops of the CPU compute them
struct _ObsValue{T, L}
  loss::L
end
_ObsValue{T}(loss::L) where {T, L} = _ObsValue{T, L}(loss)
function (o::_ObsValue{T})(η, y, w) where {T}
  weight = convert(T, w)
  iszero(weight) ? zero(T) : weight * convert(T, o.loss(η, y))
end
struct _ObsDeriv{T, L}
  loss::L
end
_ObsDeriv{T}(loss::L) where {T, L} = _ObsDeriv{T, L}(loss)
function (o::_ObsDeriv{T})(η, y, w) where {T}
  weight = convert(T, w)
  iszero(weight) ? zero(T) : weight * convert(T, ProximalBase.deriv(o.loss, η, y))
end

function (f::_DeviceGLM{L, T})(β::AbstractVector) where {L, T}
  _checkreal(f, β)
  η = ProximalBase._linear_predictor!(f, β)
  return _value_at(_sum(T, _ObsValue{T}(f.loss), η, f.y, _param(f.weights)), β)
end

function ProximalBase.value_and_gradient!(g::AbstractVector, f::_DeviceGLM{L, T}, β::AbstractVector) where {L, T}
  _checkreal(f, β)
  _checkreal(f, g)
  ProximalBase._checkgradient_output(g, β)
  _checksize(g, β)
  η = ProximalBase._linear_predictor!(f, β)
  w = _param(f.weights)
  f.d .= _ObsDeriv{T}(f.loss).(η, f.y, w)
  s = _sum(T, _ObsValue{T}(f.loss), η, f.y, w)
  mul!(g, transpose(f.X), f.d)
  return _value_at(s, β)
end

function ProximalBase.gradient!(g::AbstractVector, f::_DeviceGLM{L, T}, β::AbstractVector) where {L, T}
  _checkreal(f, β)
  _checkreal(f, g)
  ProximalBase._checkgradient_output(g, β)
  _checksize(g, β)
  η = ProximalBase._linear_predictor!(f, β)
  f.d .= _ObsDeriv{T}(f.loss).(η, f.y, _param(f.weights))
  return mul!(g, transpose(f.X), f.d)
end

####################################
#  Moving a function object to the device: adapt(CuArray, f) and the like
####################################

# These rebuild the function on the host with its parameter arrays converted,
# so that its constructor checks them there.
Adapt.adapt_structure(to, f::NormL1) = NormL1(f.λ0, Adapt.adapt(to, f.w))
Adapt.adapt_structure(to, f::NormL1Nonneg) = NormL1Nonneg(f.λ0, Adapt.adapt(to, f.w))
Adapt.adapt_structure(to, f::ElasticNet) = ElasticNet(f.λ1, f.λ2, Adapt.adapt(to, f.w))
Adapt.adapt_structure(to, f::IndBox) = IndBox(Adapt.adapt(to, f.l), Adapt.adapt(to, f.u))
Adapt.adapt_structure(to, f::SqrNormL2) = SqrNormL2(f.λ, Adapt.adapt(to, f.y))
# Without the host buffer of its proximal operator, a regularized bits type is a
# bits type that enters a kernel.
function Adapt.adapt_structure(to, r::Regularize)
  f, c = Adapt.adapt(to, r.f), Adapt.adapt(to, r.c)
  Regularize{typeof(f), typeof(r.ρ), typeof(c), Nothing}(f, r.ρ, c, nothing)
end
# Back on the host, it gets a buffer again.
Adapt.adapt_structure(to::Type{<:Array}, r::Regularize) = Regularize(Adapt.adapt(to, r.f), r.ρ, Adapt.adapt(to, r.c))
Adapt.adapt_structure(to, c::BoxConstrained) = BoxConstrained(Adapt.adapt(to, c.f), Adapt.adapt(to, c.box))

# The group functions keep their groups on the host and get the layout of the
# groups on the device; adapt(Array, ⋅) gives back the function itself.
Adapt.adapt_structure(to, f::_GroupFunction) = DeviceGroups(f, A -> Adapt.adapt(to, A))
Adapt.adapt_structure(to, d::DeviceGroups) = Adapt.adapt_structure(to, d.f)
Adapt.adapt_structure(::Type{<:Array}, f::_GroupFunction) = f
Adapt.adapt_structure(::Type{<:Array}, d::DeviceGroups) = d.f

# The losses move their data and get new buffers next to it; the SVD of
# LeastSquaresLoss is computed again, on the CPU, when it is needed.
Adapt.adapt_structure(to, f::LeastSquaresLoss) = LeastSquaresLoss(Adapt.adapt(to, f.Y), Adapt.adapt(to, f.X))
Adapt.adapt_structure(to, f::GLMLoss) =
  GLMLoss(f.loss, Adapt.adapt(to, f.y), Adapt.adapt(to, f.X); weights = Adapt.adapt(to, f.weights), offset = Adapt.adapt(to, f.offset))

# Normalize scalar parameters on the host; exceptional exponents stay separate.
function _device_projection_scale(::Type{T}, a::Real) where {T}
  c = _device_level(T, a)
  if isfinite(a) && !iszero(a) && (iszero(c) || isinf(c))
    m, e = frexp(float(a))
    return ProximalBase._ProjectionScale(_device_level(T, m), Int(e))
  end
  c
end
_device_projection_scale(::Type{T}, a::ProximalBase._ProjectionScale) where {T} =
  ProximalBase._ProjectionScale(_device_level(T, a.m), a.e)
_device_projection_check(v, l, u, w) = clamp(v,
  -ProximalBase._projection_coefficient(l, 1, w), ProximalBase._projection_coefficient(u, 1, w))

# Direct dual projections broadcast parameters separately from the function.
for (F, op, level) in ((NormL1, ProximalBase._projection_clip, :λ0),
                      (NormL1Nonneg, ProximalBase._projection_upper, :λ0),
                      (ProximalBase.CheckLoss, ProximalBase._projection_check, :τ))
  @eval function ProximalBase._conjugate_projection!(z::AnyGPUArray, f::$F, x::AnyGPUArray, γ, α = 1)
    T = float(eltype(x))
    w = _reshapeparam(_param(f.w), x)
    if f isa ProximalBase.CheckLoss
      l = _device_projection_scale(T, ProximalBase._projection_accumulate(α, 1 - f.τ))
      u = _device_projection_scale(T, ProximalBase._projection_accumulate(α, f.τ))
      z .= _device_projection_check.(x, l, u, w)
    else
      a = _device_projection_scale(T, ProximalBase._projection_accumulate(α, f.$level))
      z .= $op.(x, a, one(T), w)
    end
  end
end
Adapt.adapt_structure(to, f::ProximalBase.CheckLoss) = ProximalBase.CheckLoss(f.τ, Adapt.adapt(to, f.w))
Adapt.adapt_structure(to, f::ProximalBase.SeparableSum) = ProximalBase.SeparableSum(map(g -> Adapt.adapt(to, g), f.fs))
Adapt.adapt_structure(to, f::ProximalBase.Conjugate) = ProximalBase.Conjugate(Adapt.adapt(to, f.f))
Adapt.adapt_structure(to, f::ProximalBase.Scaled) = ProximalBase.Scaled(Adapt.adapt(to, f.f), f.α)
Adapt.adapt_structure(to, f::ProximalBase.Translate) = ProximalBase.Translate(Adapt.adapt(to, f.f), Adapt.adapt(to, f.b))

ProximalBase._maximum_abs(x::AnyGPUArray) = mapreduce(abs, max, x; init = zero(float(eltype(x))))

function ProximalBase._project_l2!(y::AnyGPUArray, x::AnyGPUArray, λ, α = 1, w = 1)
  isinf(w) && return copyto!(y, x)
  (iszero(λ) || iszero(w)) && return fill!(y, zero(eltype(y)))
  T = float(eltype(x))
  α = _device_projection_scale(T, ProximalBase._projection_accumulate(α, λ))
  λ = one(T)
  c = ProximalBase._projection_product(α, λ, w)
  # Device reductions avoid scalar indexing even for wrapped array views.
  m = ProximalBase._maximum_abs(x)
  iszero(m) && return copyto!(y, x)
  n = sqrt(_sum(typeof(m), abs2, x))
  t = min(one(n), c / n)
  if isfinite(n) && n >= floatmin(typeof(n)) && isfinite(c) && t >= floatmin(typeof(t))
    y .= t .* x
  else
    ProximalBase._project_l2_scaled!(y, x, λ, α, w, c, n)
  end
  y
end

ProximalBase._conjugate_isindicator(::Type{<:DeviceGroups{F}}) where {F} = ProximalBase._conjugate_isindicator(F)
ProximalBase._direct_conjugate(::Type{<:DeviceGroups{GroupNormL2{T, G}}}) where {T, G} = true
ProximalBase._checkparams(d::DeviceGroups, x) = _checkparams(d.f, x)

# Store a significand and exponent per group, so even an underflowing factor
# can be applied to the original coordinates in the scatter kernel.
function _projection_parts(m::T, q, λ, α, w)::Tuple{T, Int} where {T}
  isinf(w) && return (one(m), 0)
  (iszero(λ) || iszero(α) || iszero(w)) && return (zero(m), 0)
  (iszero(m) || isinf(λ) || isinf(α)) && return (one(m), 0)
  a, e = frexp(m)
  r1, e1 = ProximalBase._projection_scale_parts(α)
  r2, e2 = frexp(float(λ))
  r3, e3 = frexp(float(w))
  r, k = frexp(((r1 * r2) * r3) / (a * sqrt(q)))
  k += e1 + e2 + e3 - e
  k > 1 || (k == 1 && r >= 0.5) ? (one(r), 0) : (r, k)
end
_projection_scatter(v, p, _) = ProximalBase._projection_scale(v, p[1], p[2])
function ProximalBase._conjugate_projection!(z::AnyGPUArray, d::DeviceGroups{<:GroupNormL2}, x::AnyGPUArray, γ, α = 1)
  lay, f = d.layout, d.f
  T = float(eltype(x))
  (iszero(f.λ0) && !_has_infinite_group_weight(lay)) && return fill!(z, zero(eltype(z)))
  α = _device_projection_scale(T, ProximalBase._projection_accumulate(α, f.λ0))
  parts = _group_buffer(vec(x), Tuple{T, Int}, lay, Ref((zero(T), 0)))
  for b in lay.blocks
    m, q = _group_norm_parts(_gather(vec(x), b), b, T)
    view(parts, b.pos) .= vec(_projection_parts.(m, q, one(T), α, b.w))
  end
  n = length(lay.gid)
  view(vec(z), 1:n) .= _projection_scatter.(view(vec(x), 1:n), parts[lay.gid], lay.gid)
  view(vec(z), (length(lay.gid) + 1):length(z)) .= zero(eltype(z))
  z
end
ProximalBase._conjugate_projection!(z::AnyGPUArray, f::GroupNormL2, x::AnyGPUArray, γ, α = 1) =
  ProximalBase._conjugate_projection!(z, _on_device(f, x), x, γ, α)

end
