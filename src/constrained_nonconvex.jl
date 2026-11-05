####################################
#
#  Nonconvex penalties on a box
#
####################################

# MCP, SCAD and the ℓ0 penalty on a box. The scalar proximal problem
#
#   minimize  fᵢ(t) + (t - v)²/(2γ)  over  l ≤ t ≤ u
#
# is piecewise quadratic, convex or concave on each piece between the kinks of
# fᵢ. When it is convex everywhere (MCP for γ < a, SCAD for γ < a - 1), its
# minimizer on the interval is the clamped unconstrained one, as for a convex f.
# Otherwise it is the best of the candidates: the ends of the interval, the
# kinks inside it, and the vertices of the convex pieces clamped to their piece,
# all clamped to the interval. The minimum of each piece on the interval is
# among them, since a convex piece is smallest at its clamped vertex and a
# concave one at an end.
_has_box_prox(::Union{MCP, SCAD, NormL0}) = true

# Choose before promotion, since promotion can round or overflow an endpoint.
_pick_bound(a, b, ::typeof(max)) = a > b ? a : b > a ? b : max(a, b)
_pick_bound(a, b, ::typeof(min)) = a < b ? a : b < a ? b : min(a, b)
_bound_overflows(a, b, pick, ::Type{T}) where {T} =
  isfinite(_pick_bound(a, b, pick)) && isinf(convert(T, _pick_bound(a, b, pick)))
_convert_bound(a, b, pick, ::Type{T}) where {T} = convert(T, _pick_bound(a, b, pick))

function _intersect_boxes(A::IndBox, B::IndBox)
  bounds = (A.l, A.u, B.l, B.u)
  shape = nothing
  for p in bounds
    if p isa AbstractArray
      if shape === nothing
        shape = p
      else
        _checkparam(p, shape)
      end
    end
  end
  l1, u1, l2, u2 = map(p -> p isa AbstractArray ? vec(p) : p, bounds)
  # Test the exact intersection before rounded endpoints can conceal an empty box.
  all((l1 .<= u2) .& (l2 .<= u1)) ||
    throw(ArgumentError("the lower bounds must not exceed the upper bounds"))
  T = promote_type(map(eltype, bounds)...)
  if any(_bound_overflows.(l1, l2, max, T)) || any(_bound_overflows.(u1, u2, min, T))
    T = promote_type(map(p -> float(eltype(p)), bounds)...)
  end
  l = _convert_bound.(l1, l2, max, T)
  u = _convert_bound.(u1, u2, min, T)
  l = l isa AbstractArray ? reshape(l, size(shape)) : l
  u = u isa AbstractArray ? reshape(u, size(shape)) : u
  return IndBox(l, u)
end

# Intersect in linear order and keep finite endpoints finite; clamping the
# inner proximal operator would not give the constrained one.
Constrained(c::BoxConstrained{<:Union{MCP, SCAD, NormL0}}, C::IndBox) =
  BoxConstrained(c.f, _intersect_boxes(c.box, C))
Constrained(c::BoxConstrained{<:Union{MCP, SCAD, NormL0}}, ::IndNonnegative) = Constrained(c, IndBox(0.0, Inf))

Base.@propagate_inbounds function prox_entry(c::BoxConstrained{<:MCP}, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  f, s = c.f, convert(S, γ)
  l, u = convert(S, _entry(c.box.l, i)), convert(S, _entry(c.box.u, i))
  s < f.a && return _box_convex_entry(c, v, i, γ, l, u)
  x, b = convert(S, v), convert(S, f.a) * convert(S, f.λ)
  isnan(x) && return _box_nan_entry(c, x, l, u)
  # concave (linear for γ = a) between the kinks 0 and ±aλ, flat beyond, where
  # the vertex is x
  _best_candidate(c, x, i, s, l, u, (b, -b, max(x, b), min(x, -b)))
end

Base.@propagate_inbounds function prox_entry(c::BoxConstrained{<:SCAD}, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  f, s = c.f, convert(S, γ)
  l, u = convert(S, _entry(c.box.l, i)), convert(S, _entry(c.box.u, i))
  s < f.a - 1 && return _box_convex_entry(c, v, i, γ, l, u)
  x, λ = convert(S, v), convert(S, f.λ)
  isnan(x) && return _box_nan_entry(c, x, l, u)
  b = convert(S, f.a) * λ
  # λ|t| up to λ, with the vertices x ∓ sλ; concave (or linear) between λ and aλ;
  # flat beyond aλ, with the vertex x
  _best_candidate(c, x, i, s, l, u,
                  (λ, -λ, b, -b, clamp(x - s * λ, zero(S), λ), clamp(x + s * λ, -λ, zero(S)), max(x, b), min(x, -b)))
end

# zero, when the box holds it, against the point of the box nearest to x, where
# the penalty is λ
Base.@propagate_inbounds function prox_entry(c::BoxConstrained{<:NormL0}, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  l, u = convert(S, _entry(c.box.l, i)), convert(S, _entry(c.box.u, i))
  x = convert(S, v)
  isnan(x) && return _box_nan_entry(c, x, l, u)
  _best_candidate(c, x, i, convert(S, γ), l, u, (x,))
end

# A NaN entry (E2), after the maps that the parameters fix (E4): a box of one
# point and an infinite level give the clamped zero at every entry, whatever
# linear term a wrapper adds, so at NaN too. Otherwise the entry depends on x and
# stays NaN, also at γ = Inf, where a tilted box can put the minimizer at a
# corner; a zero level is the clamp, which keeps NaN. No comparison of the
# objectives would keep it. The level is read as stored, not converted.
@inline _box_constant(c::BoxConstrained, l, u) = (l == u) | isinf(c.f.λ)
@inline _box_nan_entry(c::BoxConstrained, x::S, l::S, u::S) where {S} =
  _box_constant(c, l, u) ? clamp(zero(S), l, u) : x
# The convex branch (finite γ), where the clamped proximal entry is already NaN
# at a NaN entry, except for the maps that the parameters fix. A select, not a
# branch, so that the ordinary loop keeps its code.
Base.@propagate_inbounds function _box_convex_entry(c::BoxConstrained, v, i, γ, l::S, u::S) where {S}
  r = clamp(prox_entry(c.f, v, i, γ), l, u)
  ifelse(isnan(v) & _box_constant(c, l, u), clamp(zero(S), l, u), r)
end

# the candidate with the smallest objective, each clamped to [l, u], with the
# clamped zero and the finite ends of the interval; ties go toward zero, as in
# the proximal operators of the penalties
Base.@propagate_inbounds function _best_candidate(c::BoxConstrained, x::S, i::Integer, s::S, l::S, u::S, candidates) where {S}
  h(t) = value_entry(c.f, t, i) + (t - x)^2 / (2s)
  best = clamp(zero(S), l, u)
  hb = h(best)
  for t in (l, u, candidates...)
    isfinite(t) || continue
    t = clamp(t, l, u)
    ht = h(t)
    if ht < hb || (ht == hb && abs(t) < abs(best))
      best, hb = t, ht
    end
  end
  best
end

# The value outside the box or with a NaN entry (E2, E4 first), as a sum over the
# entries in which NaN takes precedence over Inf: NaN at a NaN entry that the
# value depends on, zero at a NaN entry of a coordinate without bounds when the
# level is zero (the zero function there), and the membership of the box
# otherwise. Called only where the box value is not zero, so the ordinary path
# pays nothing.
struct _BoxNaNTerm{T}
  free::Bool      # a zero level: an unbounded coordinate does not depend on its entry
  t::T            # the membership tolerance
end
@inline function (b::_BoxNaNTerm{T})(v, l, u) where {T}
  isnan(v) && return (b.free & (l == -Inf) & (u == Inf)) ? zero(T) : T(NaN)
  _inbox(v, l, u, b.t) ? zero(T) : T(Inf)
end
@noinline function _box_nan_value(c::BoxConstrained{<:Union{MCP, SCAD, NormL0}}, x::AbstractArray, ::Type{T}) where {T}
  b = _BoxNaNTerm{T}(iszero(c.f.λ), _tol(T))
  if _is_device_array(x)
    s = sum(b.(x, _reshapeparam(c.box.l, x), _reshapeparam(c.box.u, x)); init = zero(T))
  else
    s = zero(T)
    for i in eachindex(IndexLinear(), x)
      s += b(x[i], _entry(c.box.l, i), _entry(c.box.u, i))
    end
  end
  iszero(s) ? convert(T, _value_unchecked(c.f, x)) : s
end

# The value at the stored output (O2). The output is in the box, or NaN where it
# depends on a NaN entry, which the penalty turns into NaN, except at a zero
# level: the zero function ignores the NaN that the box still depends on.
function _value_and_prox_box!(y, c::BoxConstrained{<:Union{MCP, SCAD, NormL0}}, x, γ)
  _checkwrite(value_and_prox!, c, y, x, γ)
  _prox_sep!(y, c, x, γ)
  T = float(eltype(y))
  iszero(c.f.λ) ? convert(T, _value_unchecked(c, y)) : convert(T, _value_unchecked(c.f, y))
end

# Counting functions under SingularValue (singular.jl) that are constant on the
# spectra of length k: IndBallL0 with room for k nonzeros, a scaled one when its
# function is, and one on a box when its function is and the box holds every
# singular value.
_count_constant(f::IndBallL0, k::Int) = f.s >= k
_count_constant(f::Scaled, k::Int) = _count_constant(f.f, k)
_count_constant(c::BoxConstrained, k::Int) =
  _count_constant(c.f, k) && all(<=(0), c.box.l) && all(==(Inf), c.box.u)

# At zero, MCP and SCAD have the kink of λ|t|: leaving zero pays when |gᵢ| > λ,
# in the direction of -gᵢ, which the box must allow.
function kkt_violation(c::BoxConstrained{<:Union{MCP, SCAD}}, g::AbstractArray, i::Integer)
  eltype(g) <: Real || _real_error(:kkt_violation)
  v = kkt_violation(c.f, g, i)
  v > 0 || return v
  (g[i] < 0 ? _entry(c.box.u, i) > 0 : _entry(c.box.l, i) < 0) ? v : zero(v)
end
