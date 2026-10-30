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
  s < f.a && return clamp(prox_entry(f, v, i, γ), l, u)
  x, b = convert(S, v), convert(S, f.a) * convert(S, f.λ)
  # concave (linear for γ = a) between the kinks 0 and ±aλ, flat beyond, where
  # the vertex is x
  _best_candidate(c, x, i, s, l, u, (b, -b, max(x, b), min(x, -b)))
end

Base.@propagate_inbounds function prox_entry(c::BoxConstrained{<:SCAD}, v::T, i::Integer, γ::Real) where {T<:Real}
  S = float(T)
  f, s = c.f, convert(S, γ)
  l, u = convert(S, _entry(c.box.l, i)), convert(S, _entry(c.box.u, i))
  s < f.a - 1 && return clamp(prox_entry(f, v, i, γ), l, u)
  x, λ = convert(S, v), convert(S, f.λ)
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
  _best_candidate(c, x, i, convert(S, γ), l, u, (x,))
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

# At zero, MCP and SCAD have the kink of λ|t|: leaving zero pays when |gᵢ| > λ,
# in the direction of -gᵢ, which the box must allow.
function kkt_violation(c::BoxConstrained{<:Union{MCP, SCAD}}, g::AbstractArray, i::Integer)
  v = kkt_violation(c.f, g, i)
  v > 0 || return v
  (g[i] < 0 ? _entry(c.box.u, i) > 0 : _entry(c.box.l, i) < 0) ? v : zero(v)
end
