include("oracle.jl")
using Random, Test

# Independently enumerate every nonempty support K. The closest point of its
# affine hull assigns p_i = u_i + (c - sum_K u)/|K| on K. Keep it only when
# p is nonnegative, then minimize squared distance over all feasible points.
# The minimizing face of the simplex necessarily contributes its minimizer.
function enumerated(v,x,c)
    G = dyadic.(v)
    m = maximum(abs,x)
    J = findall(a -> abs(a) == m, x)
    iszero(m) && sum(abs,G) <= c && return zero(Q)
    u = iszero(m) ? abs.(G) : [-Int(sign(x[i]))*G[i] for i in J]
    off = sum((G[i]^2 for i in eachindex(G) if i ∉ J);init=zero(Q))
    best = nothing
    for mask in 1:(1<<length(u))-1
        active = findall(i -> isodd(mask>>(i-1)),eachindex(u))
        shift = (c-sum(u[active]))/length(active)
        p = zeros(Q,length(u))
        p[active] = u[active] .+ shift
        minimum(p) < 0 && continue
        obj = sum(abs2,u-p) + off
        if best === nothing || obj < best
            best = obj
        end
    end
    return best
end

@testset "exact squared distances against exhaustive feasible supports" begin
    rng = MersenneTwister(981)
    for T in (Float32,Float64), n in 1:7, rep in 1:16
        v = T.(rand(rng,-11:11,n)) ./ 8
        x = isodd(rep) ? zeros(T,n) : T.(rand(rng,-1:1,n))
        c = dyadic(T(rand(rng,0:21))/8)
        @test exact_score(v,x,c)[1] == enumerated(v,x,c)
    end
    # The round-3 case has only the head active. Its squared distance is
    # (head - c)^2 + m*tie^2. Check the published rounded decimal as well.
    c = dyadic(0.1f0)
    v = [1f0;fill(0.009486833f0,100_000)]
    r = exact_score(v,zero(v),c)[1]
    @test r == (1-c)^2 + 100_000*dyadic(v[end])^2
    @test abs(sqrt(BigFloat(r)) - big"3.1320918539449983") < big"5e-17"
end
