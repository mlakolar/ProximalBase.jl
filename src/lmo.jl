####################################
#
#  Linear minimization oracles of the compact vector sets (see lmo!)
#
####################################

const _LMOSet = Union{IndSimplex, IndBallL1, IndBallL2, IndBox}

haslmo(f::Union{IndSimplex, IndBallL1, IndBallL2}) = isfinite(f.r)
haslmo(f::IndBox) = _finite_bounds(f.l) && _finite_bounds(f.u)
_finite_bounds(b::Real) = isfinite(b)
_finite_bounds(b::AbstractArray) = all(isfinite, b)

function lmo!(s::AbstractArray, f::_LMOSet, g::AbstractArray)
  haslmo(f) || _lmo_unsupported(f)
  _checkreal(f, g)
  _checkreal(f, s)
  s isa Union{Array, SubArray{<:Any, <:Any, <:Array}} && eltype(s) <: AbstractFloat ||
    throw(ArgumentError("the output of lmo! must be an Array, or a view of one, of floating-point numbers; got $(typeof(s))"))
  _checksize(s, g)
  T = eltype(s)
  _lmo_check!(f, g, T)
  _lmo_body!(s, f, g, T)
  s
end

# parameters, before any write
_lmo_fits(::Type{T}, v::Real) where {T} = isfinite(convert(T, v))
_lmo_fits(::Type{T}, v::AbstractArray) where {T} =
  promote_type(eltype(v), T) === T || all(x -> isfinite(convert(T, x)), v)
_lmo_check_fits(::Type{T}, name, v) where {T} =
  _lmo_fits(T, v) || throw(ArgumentError("$name is not representable in $T"))
_lmo_check!(f::Union{IndBallL1, IndBallL2}, g, ::Type{T}) where {T} = _lmo_check_fits(T, "the radius", f.r)
function _lmo_check!(f::IndSimplex, g, ::Type{T}) where {T}
  _lmo_check_fits(T, "the radius", f.r)
  f.equality && isempty(g) && f.r > _tol(T) && throw(ArgumentError("the simplex has no point of size $(size(g))"))
  nothing
end
function _lmo_check!(f::IndBox, g, ::Type{T}) where {T}
  _checkparams(f, g)
  _lmo_check_fits(T, "a lower bound", f.l)
  _lmo_check_fits(T, "an upper bound", f.u)
  nothing
end

# first index of the extreme entry, and whether some entry is NaN
function _lmo_extreme(g, by, better)
  nan = false
  bi = 1
  best = by(g[1])
  @inbounds for k in 1:length(g)
    v = g[k]
    nan |= isnan(v)
    a = by(v)
    if better(a, best)
      best, bi = a, k
    end
  end
  bi, best, nan
end

function _lmo_body!(s, f::IndSimplex, g, ::Type{T}) where {T}
  r = convert(T, f.r)
  if iszero(r) || isempty(g)
    fill!(s, zero(T))
    return
  end
  i, m, nan = _lmo_extreme(g, identity, <)
  if nan
    fill!(s, T(NaN))
  elseif !f.equality && m >= 0
    fill!(s, zero(T))
  else
    fill!(s, zero(T))
    @inbounds s[i] = r
  end
end

function _lmo_body!(s, f::IndBallL1, g, ::Type{T}) where {T}
  r = convert(T, f.r)
  if iszero(r) || isempty(g)
    fill!(s, zero(T))
    return
  end
  i, m, nan = _lmo_extreme(g, v -> abs(float(v)), >)
  gi = @inbounds g[i]
  if nan
    fill!(s, T(NaN))
  elseif iszero(m)
    fill!(s, zero(T))
  else
    fill!(s, zero(T))
    @inbounds s[i] = gi < 0 ? r : -r
  end
end

function _lmo_body!(s, f::IndBallL2, g, ::Type{T}) where {T}
  r = convert(T, f.r)
  if iszero(r) || isempty(g)
    fill!(s, zero(T))
    return
  end
  n = norm(g)
  if isfinite(n)
    iszero(n) && return fill!(s, zero(T))
    t = convert(typeof(n), r) / n
    if n >= floatmin(typeof(n)) && isfinite(t) && t >= floatmin(typeof(t))
      @inbounds @simd for k in 1:length(g)
        s[k] = convert(T, zero(t) - t * g[k])
      end
      return
    end
    return _lmo_l2_scaled!(s, g, r)
  end
  # a NaN entry, an infinite entry, or a finite vector whose norm overflows
  any(isnan, g) && return fill!(s, T(NaN))
  ninf = count(isinf, g)
  if ninf == 1
    i = findfirst(isinf, g)
    gi = g[i]
    fill!(s, zero(T))
    s[i] = gi < 0 ? r : -r
  elseif ninf > 1
    fill!(s, T(NaN))
  else
    _lmo_l2_scaled!(s, g, r)
  end
end

# divide by the largest magnitude first, so that neither the norm nor r/‖g‖ must fit
function _lmo_l2_scaled!(s, g, r::T) where {T}
  m = maximum(v -> abs(float(v)), g)
  @inbounds for k in 1:length(g)
    s[k] = convert(T, g[k] / m)
  end
  c = r / norm(s)
  @inbounds for k in 1:length(g)
    s[k] = zero(T) - c * s[k]
  end
  s
end

function _lmo_body!(s, f::IndBox, g, ::Type{T}) where {T}
  @inbounds @simd for k in 1:length(g)
    lo, hi, v = convert(T, _entry(f.l, k)), convert(T, _entry(f.u, k)), g[k]
    s[k] = ifelse(lo == hi, lo, ifelse(v > 0, lo, ifelse(v < 0, hi, ifelse(isnan(v), T(NaN), lo)))) + zero(T)
  end
end
