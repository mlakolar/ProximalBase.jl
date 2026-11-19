# All finite inputs are binary rationals. Use exact arithmetic through the
# mass equation and squared distance. Only the final square root is rounded.
const Q = Rational{BigInt}
dyadic(x) = Q(x)

function mass(groups, t)
    total = zero(Q)
    for (u, k) in groups
        u > t && (total += k * (u - t))
    end
    return total
end

# Monotone bisection on the piecewise linear mass. The endpoints always
# enclose a root. Test the chord intersection in exact rational arithmetic.
# Once the bracket lies within one linear segment this test terminates,
# including when the true root is itself a breakpoint (by testing knots).
function certified_level(groups, c)
    lo = minimum(first, groups) - c
    hi = maximum(first, groups)
    for (u, _) in groups
        mass(groups, u) == c && return u
    end
    while true
        L, H = mass(groups, lo), mass(groups, hi)
        @assert L >= c > H
        t = lo + (L - c) * (hi - lo) / (L - H)
        mass(groups, t) == c && return t
        mid = (lo + hi) / 2
        if mass(groups, mid) >= c
            lo = mid
        else
            hi = mid
        end
    end
end

function grouped_inputs(v, x, ::Type{W}) where W
    m = maximum(a -> abs(W(a)), x; init=zero(W))
    # Count in hardware arithmetic; all subsequent sums are exact and weighted.
    groups = Dict{W,Int}()
    outside = Dict{W,Int}()
    for i in eachindex(v, x)
        vi, xi = W(v[i]), W(x[i])
        if abs(xi) == m
            u = iszero(m) ? abs(vi) : -sign(xi) * vi
            groups[u] = get(groups, u, 0) + 1
        else
            a = abs(vi)
            outside[a] = get(outside, a, 0) + 1
        end
    end
    return iszero(m), [(dyadic(u), k) for (u,k) in groups],
           sum((dyadic(a)^2*k for (a,k) in outside); init=zero(Q))
end

function exact_score(v, x, c)
    W = float(promote_type(eltype(v), eltype(x)))
    atzero, groups, off2 = grouped_inputs(v, x, W)
    v2 = off2 + sum((k * u^2 for (u,k) in groups); init=zero(Q))
    isinf(c) && return (atzero ? zero(Q) : Inf, v2)
    (iszero(c) || isempty(v)) && return v2, v2
    if atzero && mass(groups, zero(Q)) <= c
        return zero(Q), v2
    end
    t = certified_level(groups, c)
    return off2 + sum(k*min(u,t)^2 for (u,k) in groups), v2
end
