module SCADReferenceTest
# Independent BigFloat reference for the proximal operator and the value of the smoothly clipped absolute deviation
# penalty (SCAD), for finite λ > 0, a > 2 and γ > 0 (the endpoint conventions belong to the callers),
#
#     p(t) = λt                                   0 ≤ t ≤ λ,
#     p(t) = λ² + (t - λ)(λ + (aλ - t)/(a - 1))/2  λ < t ≤ aλ,
#     p(t) = (a + 1)λ²/2                           t > aλ,
#
#     prox_{γp}(v) = argmin_u p(|u|) + (u - v)²/(2γ).
#
# `scad_reference` minimizes the FULL objective over the three pieces of p. For v ≥ 0 only u ≥ 0 can win, and on each
# piece the objective is a quadratic in u, so its minimum over u ≥ 0 lies at an endpoint of a piece or at the vertex of a
# piece (clamped to it) that has positive curvature; the three pieces are [0, λ], [λ, aλ] and [aλ, max(v, aλ)]. The candidate
# with the smallest objective wins and v < 0 gives the negative of the answer for -v.
#
# Conventions
#   * Exact tie (two candidates whose objectives agree) -> the candidate of minimum magnitude; a zero result is +0.
#   * `bits = nothing` (the default) works at the current BigFloat precision; an integer sets it for the call. The
#     objectives are compared at that precision. Candidates whose objectives agree to within 2^-(bits - 40) of their size
#     are recomputed at 4096 bits (only then, and only below 4096 bits: exact ties and near-ties of enormous dynamic
#     range); candidates that still agree to that tolerance are a tie.
#   * `scad_reference` returns the minimizer and the minimum of the objective.
#
# `scad_search` is a bounded numerical search of the same objective (a grid followed by ternary refinement) that does not
# use the vertex formulas; it exists to check the analytic reference. `scad_objective` is the full objective.
#
# Used by the scalar SCAD test and the GroupSCAD test.

export scad_penalty, scad_objective, scad_reference, scad_search

const SCAD_REFERENCE_WIDE_BITS = 4096

function scad_penalty(λ, a, v)
  t = abs(v)
  iszero(λ) && return zero(t)
  isnan(t) && return t
  iszero(t) && return zero(t)
  isinf(λ) && return λ
  (t <= λ || isinf(a)) && return λ * t
  t >= a * λ && return (a + 1) * λ^2 / 2
  # Integrate the linearly decreasing slope from λ to t by its trapezoid.
  λ^2 + (t - λ) * (λ + (a * λ - t) / (a - 1)) / 2
end

function scad_objective(λ, a, t, γ, u)
  L, A, V, G, U = BigFloat.((λ, a, t, γ, u))
  scad_penalty(L, A, U) + (U - V)^2 / (2G)
end

# Enumerate endpoints and interior vertices of the objective polynomials at the current precision, independently of the
# step regimes and candidate comparisons in the proximal implementation. Returns (u, objective, certain) for the
# magnitude u ≥ 0 of minimum size among the candidates whose objective is within 2^-(precision - 40) of the smallest, and
# whether no other candidate is that close.
function _scad_minimizer(λ, a, v, γ)
  L, A, V, G = BigFloat.((λ, a, v, γ))
  t = abs(V)
  c2 = inv(2G)
  c1 = -t / G
  pieces = ((zero(L), L, c2, c1 + L),
            (L, A * L, c2 - inv(2(A - 1)), c1 + A * L / (A - 1)),
            (A * L, max(t, A * L), c2, c1))
  cost(u) = scad_penalty(L, A, u) + (u - t)^2 / (2G)
  cands = BigFloat[zero(L)]
  for (lo, hi, q, b) in pieces
    vertex = q > 0 ? clamp(-b / (2q), lo, hi) : lo
    append!(cands, (lo, vertex, hi))
  end
  sort!(cands)
  keep = BigFloat[cands[1]]
  for k in 2:length(cands)
    cands[k] == keep[end] || push!(keep, cands[k])
  end
  F = BigFloat[cost(u) for u in keep]
  Fmin = minimum(F)
  near = [i for i in eachindex(keep) if F[i] - Fmin <= ldexp(max(abs(F[i]), abs(Fmin)), -(precision(BigFloat) - 40))]
  i = first(near)                       # `keep` is sorted: the first is of minimum magnitude
  keep[i], Fmin, length(near) == 1
end

"""
    scad_reference(λ, a, v, γ; bits = nothing) -> (u, objective)

Proximal point `u` of the SCAD penalty with parameters `λ`, `a` at `v` and step `γ`, and the minimum of the objective,
by minimizing the full objective over the pieces of the penalty (finite `λ > 0`, `a > 2`, `γ > 0`). See the header of this
file for the conventions (ties, precision).
"""
function scad_reference(λ, a, v, γ; bits::Union{Nothing, Integer} = nothing)
  bits === nothing || bits >= 64 || throw(ArgumentError("at least 64 bits are required, got $bits"))
  run(f, b) = b === nothing ? f() : setprecision(f, BigFloat, b)
  u, h, certain = run(() -> _scad_minimizer(λ, a, v, γ), bits)
  if !certain && something(bits, precision(BigFloat)) < SCAD_REFERENCE_WIDE_BITS
    u, h, _ = run(() -> _scad_minimizer(λ, a, v, γ), SCAD_REFERENCE_WIDE_BITS)
  end
  V = BigFloat(v)
  (iszero(u) ? u : copysign(u, V)), h
end

# Search the full feasible magnitude range, then refine the best grid cell
# without using the stationary-point formulas.
function scad_search(λ, a, t, γ)
  objective(u) = scad_objective(λ, a, t, γ, u)
  grid = range(zero(t), abs(t); length = 1025)
  i = argmin(map(objective, grid))
  lo, hi = grid[max(1, i - 1)], grid[min(length(grid), i + 1)]
  best = objective(grid[i])
  for _ in 1:80
    l, r = (2lo + hi) / 3, (lo + 2hi) / 3
    hl, hr = objective(l), objective(r)
    best = min(best, hl, hr)
    if hl <= hr
      hi = r
    else
      lo = l
    end
  end
  best
end

end # module
