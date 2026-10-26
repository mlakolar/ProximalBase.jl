module ProximalBaseGPUArraysCoreExt

# The separable functions, the norms and balls, the group functions and the
# losses with a design matrix on GPU arrays. The dense paths loop over linear
# indices and read and write one entry at a time, which a GPU array forbids;
# here they are broadcasts and reductions, which run as kernels. A function
# object enters a kernel through Ref(f), so it must be a bits type there: the
# built-in functions that hold parameter arrays (weights, bounds, a center)
# broadcast over those arrays instead, and never enter a kernel themselves.

using ProximalBase
using ProximalBase: NormL1, NormL1Nonneg, ElasticNet, IndBox, IndNonnegative, SqrNormL2,
                    NormLinf, IndBallL1, IndSimplex,
                    GroupNormL2, GroupNormLinf, SparseGroupNormL2, IndBallGroupL0, AbstractFunction,
                    LeastSquaresLoss, GLMLoss,
                    prox_entry, value_entry, shrink, _checksize, _checkparams, _checkgroups, _tol, _ind,
                    _kth_largest!
using GPUArraysCore: AbstractGPUArray
using LinearAlgebra: mul!
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
#  Norms, balls and the simplex
####################################

# NormL2 and IndBallL2 need only the sum of squares, a reduction, and a
# broadcast, and run on the generic methods. The ℓ∞ norm, the ℓ1 ball and the
# simplex clip or shrink the entries at the level τ with
# Σᵢ max(g(xᵢ) - τ, 0) = r, for g = abs or the identity, which the CPU finds by
# a selection. Michelot's algorithm finds it with reductions alone: τ is the
# level that makes the entries above the last level sum to r, so it only
# grows, and the entries at or below it drop out until none does, after at
# most n passes and in practice a few.

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

(f::NormLinf)(x::GPUArray) = (T = float(eltype(x)); convert(T, f.λ) * mapreduce(abs, max, x; init = zero(T)))

function ProximalBase.value_and_prox!(y::GPUArray, f::NormLinf, x::GPUArray, γ::Real)
  _checksize(y, x)
  T = float(eltype(y))
  c = convert(T, γ) * convert(T, f.λ)
  if _sum(T, abs, x) <= c
    fill!(y, zero(eltype(y)))
    return zero(T)
  end
  τ = _level(x, c, abs)
  y .= sign.(x) .* min.(abs.(x), τ)
  return convert(T, f.λ) * τ
end

function (f::IndBallL1)(x::GPUArray)
  T = float(eltype(x))
  _sum(T, abs, x) <= f.r + _tol(T) * max(one(T), f.r) ? zero(T) : T(Inf)
end

function ProximalBase.prox!(y::GPUArray, f::IndBallL1, x::GPUArray, ::Real)
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

(d::DeviceGroups)(x::AbstractArray) = d.f(x)
ProximalBase.prox!(y::AbstractArray, d::DeviceGroups, x::AbstractArray, γ::Real) = prox!(y, d.f, x, γ)
ProximalBase.value_and_prox!(y::AbstractArray, d::DeviceGroups, x::AbstractArray, γ::Real) = value_and_prox!(y, d.f, x, γ)

(d::DeviceGroups)(x::GPUArray) = _value(d.f, d.layout, x)
ProximalBase.value_and_prox!(y::GPUArray, d::DeviceGroups, x::GPUArray, γ::Real) = _value_and_prox!(y, d.f, d.layout, x, γ)
ProximalBase.prox!(y::GPUArray, d::DeviceGroups, x::GPUArray, γ::Real) = (_value_and_prox!(y, d.f, d.layout, x, γ); y)

# a function left on the host builds the layout on the device of x at each call
_on_device(f, x) = DeviceGroups(f, A -> copyto!(similar(x, eltype(A), size(A)), A))
for F in (:GroupNormL2, :GroupNormLinf, :SparseGroupNormL2, :IndBallGroupL0)
  @eval begin
    (f::$F)(x::GPUArray) = _on_device(f, x)(x)
    ProximalBase.prox!(y::GPUArray, f::$F, x::GPUArray, γ::Real) = prox!(y, _on_device(f, x), x, γ)
    ProximalBase.value_and_prox!(y::GPUArray, f::$F, x::GPUArray, γ::Real) = value_and_prox!(y, _on_device(f, x), x, γ)
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

# The thresholds c wₖ of the groups, the norms and the factors are computed in
# the element type T of the output, with c converted on the host, so that a
# Float32 array is not computed in Float64 inside a kernel.
struct _Level{T}
  c::T
end
(l::_Level{T})(w) where {T} = l.c * convert(T, w)

# wₖ ⋅ v, wₖ √s and the shrinkage of a group of norm √s at c wₖ: its factor and new weighted norm
struct _Weighted{T} end
(::_Weighted{T})(v, w) where {T} = convert(T, w) * v
struct _WSqrt{T} end
(::_WSqrt{T})(s, w) where {T} = convert(T, w) * sqrt(s)
struct _L2Factor{T}
  c::T
end
(l::_L2Factor{T})(s, w) where {T} = (c = l.c * convert(T, w); n = sqrt(s); n > c ? one(T) - c / n : zero(T))
struct _L2Value{T}
  c::T
end
(l::_L2Value{T})(s, w) where {T} = (c = l.c * convert(T, w); n = sqrt(s); n > c ? convert(T, w) * (n - c) : zero(T))

# |v| for an index in one of the K groups
struct _AbsIn{T}
  K::Int
end
(a::_AbsIn{T})(v, g) where {T} = g <= a.K ? convert(T, abs(v)) : zero(T)

# Σₖ wₖ ‖x[Gₖ]‖₂ and the ℓ1 norm of the entries in groups
function _l2_part(::Type{T}, xv, lay::_GroupLayout) where {T}
  v = zero(T)
  for b in lay.blocks
    v += _sum(T, _WSqrt{T}(), _segreduce(abs2, +, zero(T), _gather(xv, b), b), b.w)
  end
  return v
end
_grouped_l1(::Type{T}, xv, lay::_GroupLayout) where {T} =
  _sum(T, _AbsIn{T}(length(lay.pos)), view(xv, 1:length(lay.gid)), lay.gid)

# the factors of the groups in block order, then the entry for the indices in no group
function _group_buffer(yv, ::Type{T}, lay::_GroupLayout, free) where {T}
  a = similar(yv, T, length(lay.pos) + 1)
  view(a, length(a):length(a)) .= free
  return a
end

# GroupNormL2: block soft thresholding
_value(f::GroupNormL2, lay, x) =
  (_checkgroups(f, x); T = float(eltype(x)); convert(T, f.λ0) * _l2_part(T, vec(x), lay))

_scale(v, t, _) = t * v

function _value_and_prox!(y, f::GroupNormL2, lay, x, γ::Real)
  _checksize(y, x)
  _checkgroups(f, x)
  S = float(eltype(y))
  y === x || copyto!(y, x)
  yv = vec(y)
  fac = _group_buffer(yv, S, lay, one(S))
  c = convert(S, γ * f.λ0)
  v = zero(S)
  for b in lay.blocks
    s = _segreduce(abs2, +, zero(S), _gather(yv, b), b)
    view(fac, b.pos) .= _L2Factor(c).(vec(s), vec(b.w))
    v += _sum(S, _L2Value(c), s, b.w)
  end
  _scatter_back!(_scale, yv, lay, fac)
  return convert(S, f.λ0) * v
end

# SparseGroupNormL2: soft thresholding, then block soft thresholding
function _value(f::SparseGroupNormL2, lay, x)
  _checkgroups(f, x)
  T = float(eltype(x))
  xv = vec(x)
  return convert(T, f.λ0) * _l2_part(T, xv, lay) + convert(T, f.λ1) * _grouped_l1(T, xv, lay)
end

struct _ShrinkSq{T}
  c::T
end
(s::_ShrinkSq{T})(v) where {T} = abs2(shrink(convert(T, v), s.c))
struct _SparseScale{T}
  c::T
  K::Int
end
(s::_SparseScale{T})(v, t, g) where {T} = g <= s.K ? t * shrink(convert(T, v), s.c) : v

function _value_and_prox!(y, f::SparseGroupNormL2, lay, x, γ::Real)
  _checksize(y, x)
  _checkgroups(f, x)
  S = float(eltype(y))
  y === x || copyto!(y, x)
  yv = vec(y)
  fac = _group_buffer(yv, S, lay, one(S))
  c1 = convert(S, γ * f.λ1)
  c2 = convert(S, γ * f.λ0)
  v2 = zero(S)
  for b in lay.blocks
    s = _segreduce(_ShrinkSq(c1), +, zero(S), _gather(yv, b), b)
    view(fac, b.pos) .= _L2Factor(c2).(vec(s), vec(b.w))
    v2 += _sum(S, _L2Value(c2), s, b.w)
  end
  _scatter_back!(_SparseScale(c1, length(lay.pos)), yv, lay, fac)
  return convert(S, f.λ0) * v2 + convert(S, f.λ1) * _grouped_l1(S, yv, lay)
end

# GroupNormLinf: each group clipped at a level, by Moreau's identity with the
# projection onto an ℓ1 ball, whose level Michelot's algorithm finds for all
# the groups of a block at once, as _level does for one vector
function _value(f::GroupNormLinf, lay, x)
  _checkgroups(f, x)
  T = float(eltype(x))
  xv = vec(x)
  v = zero(T)
  for b in lay.blocks
    v += _sum(T, _Weighted{T}(), _segreduce(abs, max, zero(T), _gather(xv, b), b), b.w)
  end
  return convert(T, f.λ0) * v
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

function _value_and_prox!(y, f::GroupNormLinf, lay, x, γ::Real)
  _checksize(y, x)
  _checkgroups(f, x)
  S = float(eltype(y))
  y === x || copyto!(y, x)
  yv = vec(y)
  levels = _group_buffer(yv, S, lay, S(Inf))
  c = convert(S, γ * f.λ0)
  v = zero(S)
  for b in lay.blocks
    τ = _linf_levels(_gather(yv, b), b, _Level(c).(b.w))
    view(levels, b.pos) .= vec(τ)
    v += _sum(S, _Weighted{S}(), τ, b.w)
  end
  _scatter_back!(_clip, yv, lay, levels)
  return convert(S, f.λ0) * v
end

# IndBallGroupL0: the number of nonzero groups, and of nonzero entries in a group
function _value(f::IndBallGroupL0, lay, x)
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
function _value_and_prox!(y, f::IndBallGroupL0, lay, x, ::Real)
  _checksize(y, x)
  _checkgroups(f, x)
  f.k == typemax(Int) ||
    throw(ArgumentError("IndBallGroupL0 with a bound k on the entries of a group projects on the CPU only"))
  S = float(eltype(y))
  y === x || copyto!(y, x)
  yv = vec(y)
  K = length(lay.pos)
  sq = similar(yv, S, K)
  nz = similar(yv, Bool, K)
  for b in lay.blocks
    X = _gather(yv, b)
    view(sq, b.pos) .= vec(_segreduce(abs2, +, zero(S), X, b))
    view(nz, b.pos) .= vec(_segreduce(!iszero, |, false, X, b))
  end
  keep = _keep_groups(Array(sq), Array(nz), lay.pos, f.s)
  _scatter_back!(_keep, yv, lay, copyto!(similar(yv, Bool, K + 1), keep))
  return zero(S)
end

# which groups the projection keeps, in block order, then true for the indices
# in no group: those of the s largest squared norms sq, the earlier groups
# first among ties; nz tells the nonzero groups (a sum of squares may underflow)
function _keep_groups(sq::Vector{T}, nz::Vector{Bool}, pos::Vector{Int}, s::Int) where {T}
  K = length(pos)
  keep = fill(true, K + 1)
  count(nz) <= s && return keep
  q = sq[pos]                                     # in the order of the groups
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

# the gradient at a device array is a device array
ProximalBase._gradient_buffer(x::GPUArray) = similar(x, float(eltype(x)))

# The losses with their buffers on the device. The parameters keep the bounds
# of the types, so that the methods here are more specific than those of the
# CPU.
const _DeviceLeastSquares{T} = LeastSquaresLoss{T, <:AbstractVecOrMat, <:AbstractMatrix, N, <:GPUArray{T, N}} where {N}
const _DeviceGLM{L, T} = GLMLoss{L, T, <:AbstractVector, <:AbstractMatrix, <:Union{Nothing, AbstractVector},
                                 <:Union{Nothing, AbstractVector}, <:GPUArray{T, 1}}

function (f::_DeviceLeastSquares{T})(β::AbstractVecOrMat) where {T}
  mul!(f.tmp, f.X, β)
  return convert(T, ProximalBase._sumabs2_diff(f.Y, f.tmp)) / 2
end

# value_and_gradient!, gradient! and hessian_vec! are products and broadcasts,
# and run as they are; the proximal operator needs the SVD of X
ProximalBase.prox!(::AbstractVecOrMat, ::_DeviceLeastSquares, ::AbstractVecOrMat, ::Real) =
  throw(ArgumentError("the proximal operator of LeastSquaresLoss uses the SVD of X, by LAPACK, and runs on the CPU only"))

# wᵢ ℓ(ηᵢ, yᵢ) and wᵢ ℓ'(ηᵢ, yᵢ) in type T, as the loops of the CPU compute them
struct _ObsValue{T, L}
  loss::L
end
_ObsValue{T}(loss::L) where {T, L} = _ObsValue{T, L}(loss)
(o::_ObsValue{T})(η, y, w) where {T} = iszero(w) ? zero(T) : convert(T, w) * convert(T, o.loss(η, y))
struct _ObsDeriv{T, L}
  loss::L
end
_ObsDeriv{T}(loss::L) where {T, L} = _ObsDeriv{T, L}(loss)
(o::_ObsDeriv{T})(η, y, w) where {T} = iszero(w) ? zero(T) : convert(T, w) * convert(T, ProximalBase.deriv(o.loss, η, y))

function (f::_DeviceGLM{L, T})(β::AbstractVector) where {L, T}
  η = ProximalBase._linear_predictor!(f, β)
  return _sum(T, _ObsValue{T}(f.loss), η, f.y, _param(f.weights))
end

function ProximalBase.value_and_gradient!(g::AbstractVector, f::_DeviceGLM{L, T}, β::AbstractVector) where {L, T}
  _checksize(g, β)
  η = ProximalBase._linear_predictor!(f, β)
  w = _param(f.weights)
  f.d .= _ObsDeriv{T}(f.loss).(η, f.y, w)
  s = _sum(T, _ObsValue{T}(f.loss), η, f.y, w)
  mul!(g, transpose(f.X), f.d)
  return s
end

function ProximalBase.gradient!(g::AbstractVector, f::_DeviceGLM{L, T}, β::AbstractVector) where {L, T}
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

end
