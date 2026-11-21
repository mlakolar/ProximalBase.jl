# Experimental output-value sharing. Public methods are loaded verbatim.
function value_with_native_screen!(y::Vector{T}, f::ProximalBase.IndBallL1{T}, x::Vector{T}, step::Real) where {T<:BinaryFloat}
    ProximalBase.prox!(y, f, x, step)
    r = f.r
    (iszero(r) || isinf(r)) && return (f(y), nothing)
    s, k, _ = ProximalBase._ball_l1_norm(y, T)
    v = if isnan(s)
        T(NaN)
    else
        isfinite(s) && ProximalBase._ball_l1_inside(r, s, k) ? zero(T) : T(Inf)
    end
    bounds = k == 0 && supported_radius(r) ? cheap_from_sum(s, r, length(y)) : nothing
    v, bounds
end
