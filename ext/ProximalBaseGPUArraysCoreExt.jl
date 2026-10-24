module ProximalBaseGPUArraysCoreExt

# The separable functions on GPU arrays. The dense paths loop over linear
# indices and read and write one entry at a time, which a GPU array forbids;
# here they are broadcasts and reductions, which run as kernels. A function
# object enters a kernel through Ref(f), so it must be a bits type there:
# the built-in functions that hold parameter arrays (weights, bounds, a center)
# broadcast over those arrays instead, and never enter a kernel themselves.

using ProximalBase
using ProximalBase: NormL1, NormL1Nonneg, ElasticNet, IndBox, IndNonnegative, SqrNormL2,
                    prox_entry, value_entry, shrink, _checksize, _checkparams, _tol
using GPUArraysCore: AbstractGPUArray
using Base.Broadcast: broadcasted, instantiate
import Adapt

const GPUArray = AbstractGPUArray

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

function ProximalBase._prox_sep!(y::GPUArray, f, x::GPUArray, γ::Real)
  _checksize(y, x)
  _checkparams(f, x)
  y .= prox_entry.(Ref(f), x, LinearIndices(x), γ)
  y
end

_term(f, v, i, ::Type{T}) where {T} = convert(T, value_entry(f, v, i))
ProximalBase._value_dense(f, x::GPUArray) =
  (T = float(eltype(x)); _sum(T, _term, Ref(f), x, LinearIndices(x), T))

####################################
#  The built-in functions with parameter arrays
####################################

# the step and level in the element type of the output, so that a Float32
# array is not computed in Float64
_scalar(y, a) = convert(float(eltype(y)), a)

function ProximalBase._prox_sep!(y::GPUArray, f::NormL1, x::GPUArray, γ::Real)
  _checksize(y, x)
  _checkparams(f, x)
  y .= shrink.(x, _scalar(y, γ * f.λ0) .* _param(f.w))
end
ProximalBase._value_dense(f::NormL1, x::GPUArray) =
  (T = float(eltype(x)); _scalar(x, f.λ0) * _sum(T, (w, v) -> w * abs(v), _param(f.w), x))

function ProximalBase._prox_sep!(y::GPUArray, f::NormL1Nonneg, x::GPUArray, γ::Real)
  _checksize(y, x)
  _checkparams(f, x)
  y .= max.(x .- _scalar(y, γ * f.λ0) .* _param(f.w), 0)
end
function ProximalBase._value_dense(f::NormL1Nonneg, x::GPUArray)
  T = float(eltype(x))
  t = _tol(T)
  _all(v -> v >= -t, x) || return T(Inf)
  _scalar(x, f.λ0) * _sum(T, (w, v) -> w * max(v, zero(v)), _param(f.w), x)
end

function ProximalBase._prox_sep!(y::GPUArray, f::ElasticNet, x::GPUArray, γ::Real)
  _checksize(y, x)
  _checkparams(f, x)
  y .= shrink.(x, _scalar(y, γ * f.λ1) .* _param(f.w)) ./ (1 + _scalar(y, γ * f.λ2))
end
function ProximalBase._value_dense(f::ElasticNet, x::GPUArray)
  T = float(eltype(x))
  _scalar(x, f.λ1) * _sum(T, (w, v) -> w * abs(v), _param(f.w), x) + _scalar(x, f.λ2) * _sum(T, abs2, x) / 2
end

function ProximalBase._prox_sep!(y::GPUArray, f::IndBox, x::GPUArray, ::Real)
  _checksize(y, x)
  _checkparams(f, x)
  y .= clamp.(x, f.l, f.u)
end
# whether l ≤ v ≤ u within the tolerance t of the indicator functions, as on
# the CPU; a functor rather than a closure, so that Julia 1.10 infers its type
struct _InBox{T}
  t::T
end
(b::_InBox)(v, l, u) = (v >= l - b.t * max(one(b.t), abs(l))) & (v <= u + b.t * max(one(b.t), abs(u)))

function ProximalBase._value_dense(f::IndBox, x::GPUArray)
  T = float(eltype(x))
  _all(_InBox(_tol(T)), x, f.l, f.u) ? zero(T) : T(Inf)
end

ProximalBase._prox_sep!(y::GPUArray, ::IndNonnegative, x::GPUArray, ::Real) = (_checksize(y, x); y .= max.(x, 0))
function ProximalBase._value_dense(::IndNonnegative, x::GPUArray)
  T = float(eltype(x))
  t = _tol(T)
  _all(v -> v >= -t, x) ? zero(T) : T(Inf)
end

# the value of SqrNormL2 with a center: Σ (xᵢ - yᵢ)² as one reduction
function ProximalBase._sumabs2_diff(x::GPUArray, y::GPUArray)
  size(x) == size(y) || throw(DimensionMismatch("the arrays have sizes $(size(x)) and $(size(y))"))
  _sum(float(promote_type(eltype(x), eltype(y))), (a, b) -> abs2(a - b), x, y)
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

end
