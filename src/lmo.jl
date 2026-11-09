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
  # structure (types, sizes, parameter lengths), then the set, then the parameters (E1, O6)
  _checkreal(f, g)
  _checkreal(f, s)
  s isa Union{Array, SubArray{<:Any, <:Any, <:Array}} && eltype(s) <: AbstractFloat ||
    throw(ArgumentError("the output of lmo! must be an Array, or a view of one, of floating-point numbers; got $(typeof(s))"))
  _checksize(s, g)
  f isa IndBox && _checkparams(f, g)
  haslmo(f) || _lmo_unsupported(f)
  T = eltype(s)
  _lmo_check!(f, g, T)
  _lmo_body!(s, f, g, T)
  s
end

# the owner of a symmetric matrix is not an output of lmo!
lmo!(s::SymmetricSparseIterate, f::_LMOSet, g::AbstractArray) = throw(ArgumentError(
  "the output of lmo! must be an Array, or a view of one, of floating-point numbers; got $(typeof(s))"))

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

# The vertex that the oracle of a simplex or an ℓ1 ball selects, as (index, value, nan): the first
# index of the extreme entry and the signed radius, index 0 and value 0 for the origin, and nan when
# an entry of g is NaN. lmo! writes it into an array and lmo_atom keeps it as it is.
function _lmo_vertex(f::IndSimplex, g, ::Type{T}) where {T}
  r = convert(T, f.r)
  (iszero(r) || isempty(g)) && return 0, zero(T), false
  i, m, nan = _lmo_extreme(g, identity, <)
  nan && return 0, T(NaN), true
  !f.equality && m >= 0 && return 0, zero(T), false
  i, r, false
end

function _lmo_vertex(f::IndBallL1, g, ::Type{T}) where {T}
  r = convert(T, f.r)
  (iszero(r) || isempty(g)) && return 0, zero(T), false
  i, m, nan = _lmo_extreme(g, v -> abs(float(v)), >)
  nan && return 0, T(NaN), true
  iszero(m) && return 0, zero(T), false
  gi = @inbounds g[i]
  i, gi < 0 ? r : -r, false
end

function _lmo_body!(s, f::Union{IndSimplex, IndBallL1}, g, ::Type{T}) where {T}
  i, v, nan = _lmo_vertex(f, g, T)
  fill!(s, nan ? T(NaN) : zero(T))
  i > 0 && (@inbounds s[i] = v)
  s
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
    # arithmetic in a type that holds both the norm and the output
    R = promote_type(typeof(n), T)
    rr = convert(R, r)
    t = rr / convert(R, n)
    if n >= floatmin(typeof(n)) && isfinite(t) && t >= floatmin(R)
      # |gₖ| ≤ ‖g‖, so the exact entry is at most r; rounding of t and of the product
      # can only push it past r, where it is clamped (an overflow to Inf included)
      @inbounds @simd for k in 1:length(g)
        s[k] = convert(T, max(-rr, min(rr, zero(R) - t * convert(R, g[k]))))
      end
      return
    end
    return _lmo_l2_scaled!(s, g, rr)
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
    _lmo_l2_scaled!(s, g, convert(promote_type(float(eltype(g)), T), r))
  end
end

# The norm overflows or r/‖g‖ leaves the normal range. With m = max |gₖ|, ‖g‖ = m ν for
# ν = ‖g/m‖ ∈ [1, √n], so c = r/ν ≤ r is representable, and the entry is gₖ (c/m) or,
# when c/m is not a normal number, (gₖ/m) c. Both are at most r; clamp the rounding.
function _lmo_l2_scaled!(s, g, rr::R) where {R}
  T = eltype(s)
  m = convert(R, maximum(v -> abs(float(v)), g))
  ν2 = zero(R)
  @inbounds for k in 1:length(g)
    ν2 += abs2(convert(R, g[k]) / m)
  end
  c = rr / sqrt(ν2)
  q = c / m
  direct = isfinite(q) && q >= floatmin(R)
  @inbounds for k in 1:length(g)
    v = convert(R, g[k])
    x = direct ? v * q : (v / m) * c
    s[k] = convert(T, max(-rr, min(rr, zero(R) - x)))
  end
  s
end

function _lmo_body!(s, f::IndBox, g, ::Type{T}) where {T}
  @inbounds @simd for k in 1:length(g)
    lo, hi, v = convert(T, _entry(f.l, k)), convert(T, _entry(f.u, k)), g[k]
    s[k] = ifelse(lo == hi, lo, ifelse(v > 0, lo, ifelse(v < 0, hi, ifelse(isnan(v), T(NaN), lo)))) + zero(T)
  end
end
