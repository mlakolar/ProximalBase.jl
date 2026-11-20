module MCPReferenceTest
# Independent BigFloat reference for the proximal operator and the value of the minimax concave penalty (MCP),
#
#     p(u) = λ|u| - u²/(2a)   for |u| ≤ aλ,        p(u) = aλ²/2   otherwise,        λ, a > 0,
#
#     prox_{γp}(v) = argmin_u p(u) + (u - v)²/(2γ).
#
# The reference minimizes the FULL objective over the two regions of p. For v ≥ 0 only u ≥ 0 can win (p is even),
# and the objective is a quadratic on each region, so its minimum over u ≥ 0 lies at one of
#
#     0 and aλ                     the endpoints of the inner region [0, aλ],
#     a(v - γλ)/(a - γ)            the stationary point of the inner region, when γ ≠ a and it lies in [0, aλ]
#                                  (a minimum for γ < a, a maximum for γ > a; the feasibility is decided by exact
#                                  comparisons, γ < a: γλ ≤ v ≤ aλ; γ > a: aλ ≤ v ≤ γλ),
#     v                            the stationary point of the outer region, when v ≥ aλ,
#
# the candidate with the smallest objective wins, and v < 0 gives the negative of the answer for -v.
#
# Conventions
#   * Exact tie (two candidates whose objectives agree) -> the candidate of minimum magnitude; a zero result is +0.
#   * The objectives are compared at 256 bits. Candidates whose objectives agree to within 2^-216 of their size are
#     recomputed at 4096 bits (only then: exact ties and near-ties of enormous dynamic range); candidates that still
#     agree to within 2^-4056 are a tie.
#   * Endpoints, in precedence order (Float, Int and BigFloat inputs are converted exactly):
#       λ = 0                  value 0, prox = identity (NaN, ±Inf and -0 included), whatever a and γ are;
#       λ = Inf                value 0 at ±0, NaN at NaN, Inf elsewhere; prox = constant +0, also for γ = 0 and NaN v;
#       finite λ > 0, γ = 0    prox = identity (-0, NaN, ±Inf included);
#       finite λ > 0, γ = Inf  prox = constant +0, NaN included;
#       finite steps           NaN and ±Inf pass (prox(NaN) = NaN, prox(±Inf) = ±Inf);
#       finite a, v = ±Inf     value aλ²/2 (the value saturates); NaN gives NaN;
#       a = Inf                p(u) = λ|u|: value λ|v|, prox = soft threshold sign(v) max(|v| - γλ, 0) with +0 at ties.
#   * Invalid parameters (NaN, λ < 0, a ≤ 0, γ < 0) throw ArgumentError; so does bits < 256.
#
# `mcp_reference_search` is a bounded numerical search of the same objective (a grid on each region followed by golden
# section refinement) that does not use the stationary formula; it returns a point within about 2^-100 of the scale
# of the minimizer and exists to check the analytic reference. `mcp_reference_objective` is the full objective.
#
# Used by the quick MCP test and by the extended reference sweep.

export mcp_reference, mcp_reference_value, mcp_reference_search, mcp_reference_objective

const MCP_REFERENCE_BITS = 256
const MCP_REFERENCE_WIDE_BITS = 4096

function _check_bits(bits)
    bits >= MCP_REFERENCE_BITS || throw(ArgumentError("at least $MCP_REFERENCE_BITS bits are required, got $bits"))
    return bits
end
function _check_parameters(λ, a)
    (isnan(λ) || λ < 0) && throw(ArgumentError("λ must be nonnegative, got $λ"))
    (isnan(a) || a <= 0) && throw(ArgumentError("a must be positive, got $a"))
    return nothing
end
function _check_step(γ)
    (isnan(γ) || γ < 0) && throw(ArgumentError("γ must be nonnegative, got $γ"))
    return nothing
end

# the result for a value u ≥ 0 computed for |v|, with the sign of v and +0 for a zero result
_signed(u::BigFloat, v) = iszero(u) ? BigFloat(0) : (v < 0 ? -u : u)

# Endpoints of the prox in precedence order; `nothing` when the minimization decides. Called with the working
# precision set.
function _prox_endpoint(λ, v, γ)
    iszero(λ) && return BigFloat(v)
    isinf(λ) && return BigFloat(0)
    iszero(γ) && return BigFloat(v)
    isinf(γ) && return BigFloat(0)
    (isnan(v) || isinf(v)) && return BigFloat(v)
    return nothing
end

# full objective p(|u|) + (u - v)²/(2γ) of BigFloat arguments (a may be Inf: p = λ|u|)
function _objective(u::BigFloat, λ::BigFloat, a::BigFloat, v::BigFloat, γ::BigFloat)
    m = abs(u)
    p = isinf(a) ? λ * m : (m <= a * λ ? λ * m - m * m / (2 * a) : a * λ * λ / 2)
    d = u - v
    return p + d * d / (2 * γ)
end

# Minimizer over u ≥ 0 of the objective for v ≥ 0 (the argument is |v|) with finite λ, a, γ > 0, at the working
# precision `bits`. Returns (u, certain); `certain` is false when another candidate has an objective within
# 2^-(bits - 40) of the smallest one (then u is the smallest such candidate).
function _positive_minimizer(λ, a, v, γ, bits)
    lb, ab, gb, vb = BigFloat(λ), BigFloat(a), BigFloat(γ), abs(BigFloat(v))
    c = ab * lb
    cands = BigFloat[BigFloat(0), c]
    if gb < ab
        if vb >= gb * lb && vb <= c
            push!(cands, clamp(ab * (vb - gb * lb) / (ab - gb), BigFloat(0), c))
        end
    elseif gb > ab
        if vb >= c && vb <= gb * lb
            push!(cands, clamp(ab * (vb - gb * lb) / (ab - gb), BigFloat(0), c))
        end
    end
    vb >= c && push!(cands, vb)
    sort!(cands)
    keep = BigFloat[cands[1]]
    for k in 2:length(cands)
        cands[k] == keep[end] || push!(keep, cands[k])
    end
    F = BigFloat[_objective(u, lb, ab, vb, gb) for u in keep]
    Fmin = minimum(F)
    near = [i for i in eachindex(keep) if F[i] - Fmin <= ldexp(max(abs(F[i]), abs(Fmin)), -(bits - 40))]
    return keep[first(near)], length(near) == 1        # `keep` is sorted: the first is of minimum magnitude
end

"""
    mcp_reference(λ, a, v, γ; bits = 256) -> BigFloat

Proximal point of the minimax concave penalty with parameters `λ`, `a` at `v` and step `γ`, by minimizing the full
objective over the two regions of the penalty. See the header of this file for the conventions (ties, endpoints).
"""
function mcp_reference(λ::Real, a::Real, v::Real, γ::Real; bits::Integer = MCP_REFERENCE_BITS)
    _check_parameters(λ, a); _check_step(γ); _check_bits(bits)
    return setprecision(BigFloat, bits) do
        r = _prox_endpoint(λ, v, γ)
        r === nothing || return r
        if isinf(a)                                      # p(u) = λ|u|: soft threshold
            m, t = abs(BigFloat(v)), BigFloat(γ) * BigFloat(λ)
            return _signed(m > t ? m - t : BigFloat(0), v)
        end
        u, certain = setprecision(() -> _positive_minimizer(λ, a, v, γ, bits), BigFloat, bits)
        if !certain && bits < MCP_REFERENCE_WIDE_BITS
            wide = MCP_REFERENCE_WIDE_BITS
            u, _ = setprecision(() -> _positive_minimizer(λ, a, v, γ, wide), BigFloat, wide)
        end
        return _signed(u, v)
    end
end

"""
    mcp_reference_value(λ, a, v; bits = 256) -> BigFloat

Value of the minimax concave penalty at `v`: 0 for `λ = 0`; 0, NaN or Inf for `λ = Inf` (at ±0, at NaN, elsewhere);
otherwise NaN at NaN, `λ|v|` for `a = Inf`, `aλ²/2` at ±Inf and `p(v)` for finite `a`.
"""
function mcp_reference_value(λ::Real, a::Real, v::Real; bits::Integer = MCP_REFERENCE_BITS)
    _check_parameters(λ, a); _check_bits(bits)
    return setprecision(BigFloat, bits) do
        iszero(λ) && return BigFloat(0)
        if isinf(λ)
            isnan(v) && return BigFloat(NaN)
            return iszero(v) ? BigFloat(0) : BigFloat(Inf)
        end
        isnan(v) && return BigFloat(NaN)
        lb = BigFloat(λ)
        isinf(a) && return lb * abs(BigFloat(v))
        ab = BigFloat(a)
        isinf(v) && return ab * lb * lb / 2
        m = abs(BigFloat(v))
        return m <= ab * lb ? lb * m - m * m / (2 * ab) : ab * lb * lb / 2
    end
end

"""
    mcp_reference_objective(λ, a, v, γ, u; bits = 256) -> BigFloat

The full objective `p(u) + (u - v)²/(2γ)` for `λ ≥ 0`, `a > 0` (`Inf` allowed), finite `v`, `u` and finite `γ > 0`.
"""
function mcp_reference_objective(λ::Real, a::Real, v::Real, γ::Real, u::Real; bits::Integer = MCP_REFERENCE_BITS)
    _check_parameters(λ, a); _check_bits(bits)
    (isfinite(λ) && isfinite(v) && isfinite(u) && isfinite(γ) && γ > 0) ||
        throw(ArgumentError("finite λ, v, u and a finite positive γ are required"))
    return setprecision(BigFloat, bits) do
        _objective(BigFloat(u), BigFloat(λ), BigFloat(a), BigFloat(v), BigFloat(γ))
    end
end

# golden-section search of a function on [lo, hi]; returns the midpoint of the final bracket
function _golden(obj, lo::BigFloat, hi::BigFloat, iters::Integer)
    φ = (sqrt(BigFloat(5)) - 1) / 2
    x1 = hi - φ * (hi - lo); x2 = lo + φ * (hi - lo)
    f1 = obj(x1); f2 = obj(x2)
    for _ in 1:iters
        if f1 <= f2
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - φ * (hi - lo); f1 = obj(x1)
        else
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + φ * (hi - lo); f2 = obj(x2)
        end
    end
    return (lo + hi) / 2
end

# candidates of one region [lo, hi]: the best grid point and the golden-section refinement of its bracket
function _search_region(obj, lo::BigFloat, hi::BigFloat, grid::Integer, bits::Integer)
    lo == hi && return BigFloat[lo]
    pts = BigFloat[lo + (hi - lo) * k / grid for k in 0:grid]
    pts[end] = hi
    vals = BigFloat[obj(p) for p in pts]
    k = argmin(vals)                                      # the first minimum: the smallest point on ties
    refined = _golden(obj, pts[max(k - 1, 1)], pts[min(k + 1, length(pts))], ceil(Int, 1.45 * bits))
    return BigFloat[pts[k], refined]
end

"""
    mcp_reference_search(λ, a, v, γ; bits = 256, grid = 64) -> BigFloat

Bounded numerical search of the same minimization: a grid of `grid + 1` points and a golden-section refinement on each
region of the penalty (at most `1.45 bits` steps), among the candidates the smallest objective, ties to the smallest
magnitude and +0. It does not use the stationary formula of `mcp_reference`, and the endpoints are the same.
"""
function mcp_reference_search(λ::Real, a::Real, v::Real, γ::Real; bits::Integer = MCP_REFERENCE_BITS, grid::Integer = 64)
    _check_parameters(λ, a); _check_step(γ); _check_bits(bits)
    grid >= 2 || throw(ArgumentError("grid must be at least 2"))
    return setprecision(BigFloat, bits) do
        r = _prox_endpoint(λ, v, γ)
        r === nothing || return r
        lb, ab, gb, x = BigFloat(λ), BigFloat(a), BigFloat(γ), abs(BigFloat(v))
        obj(u) = _objective(u, lb, ab, x, gb)
        zero_ = BigFloat(0)
        if isinf(ab)
            regions = [(zero_, x)]
        else
            c = ab * lb
            regions = x > c ? [(zero_, c), (c, x)] : [(zero_, x)]
        end
        cands = BigFloat[]
        for (lo, hi) in regions
            append!(cands, _search_region(obj, lo, hi, grid, bits))
        end
        sort!(cands)
        F = BigFloat[obj(u) for u in cands]
        Fmin = minimum(F)
        i = findfirst(j -> F[j] - Fmin <= ldexp(max(abs(F[j]), abs(Fmin)), -(bits - 40)), eachindex(cands))
        return _signed(cands[i], v)
    end
end

end # module
