# Run from an instantiated candidate project. The optional argument is the
# number of ordinary samples per type. The wide sweep uses 4096-bit arithmetic.
using ProximalBase, Random, LinearAlgebra
BLAS.set_num_threads(1)

function penalty(l, a, u)
    u <= l && return l * u
    u >= a * l && return (a + 1) * l^2 / 2
    (2a * l * u - u^2 - l^2) / (2(a - 1))
end

function check(l::T, a::T, t::T, step::T) where T
    z = prox_block(SCAD(l, a), t, 1, step)
    isfinite(z) && zero(T) <= z <= t || error("invalid point: " * repr((l,a,t,step,z)))
    L, A, V, G, Z = BigFloat.((l, a, t, step, z))
    H(u) = penalty(L, A, u) + (u-V)^2 / (2G)
    best = zero(V)
    cost = H(best)
    for (lo, hi, quadratic, linear) in (
        (zero(V), L, 1/(2G), L-V/G),
        (L, A*L, 1/(2G)-1/(2(A-1)), A*L/(A-1)-V/G),
        (A*L, max(A*L,V), 1/(2G), -V/G))
        for u in (lo, hi, quadratic > 0 ? clamp(-linear/(2quadratic), lo, hi) : lo)
            h = H(u)
            if h < cost || (h == cost && u < best)
                best, cost = u, h
            end
        end
    end
    floor_spacing = BigFloat(nextfloat(zero(T)))
    rounding = abs(H(best+floor_spacing)-cost)
    got = H(Z)
    got <= cost * (1 + 64eps(T)) + rounding ||
        error("objective: " * repr((l,a,t,step,z,T(best),Float64(got/cost))))
    actual = value_entry(SCAD(l,a), t, 1)
    expected = T(penalty(L,A,V))
    isapprox(actual, expected; rtol=24eps(T), atol=2nextfloat(zero(T))) ||
        error("penalty: " * repr((l,a,t,step,actual,expected)))
    cost == 0 ? zero(T) : T(max(0,(got-cost-rounding)/cost))
end

function input(rng, ::Type{T}, spread) where T
    lo, hi = spread ? (exponent(nextfloat(zero(T))), exponent(floatmax(T))-1) : (-20,20)
    l = max(nextfloat(zero(T)), ldexp(one(T)+rand(rng,T),rand(rng,lo:hi)))
    a = rand(rng,Bool) ? T(2)+T(14)*rand(rng,T) :
        ldexp(one(T)+rand(rng,T),rand(rng,2:(spread ? hi : 20)))
    g = if rand(rng,Bool)
        rand(rng,(prevfloat(a-1),a-1,nextfloat(a-1),a+1,T(0.5)*a,2a))
    else
        max(nextfloat(zero(T)), ldexp(one(T)+rand(rng,T),rand(rng,lo:hi)))
    end
    t = if rand(rng,Bool)
        # Probe a few ulps from the joins and both objective switches.
        k = rand(rng,(one(T),a,g,g+1,(a/2+g/2)+T(0.5),sqrt(g)*sqrt(a+1)))
        v = k*l
        rand(rng,(prevfloat(v),v,nextfloat(v)))
    else
        ldexp(one(T)+rand(rng,T),rand(rng,lo:hi))
    end
    l, a, t, g
end

seed = 953
rng = Xoshiro(seed)
n = parse(Int, get(ARGS,1,"200000"))
println("Julia $VERSION; $(Sys.CPU_NAME); Julia threads $(Threads.nthreads()); BLAS threads $(BLAS.get_num_threads()); seed $seed")
for (precision, spread, count) in ((256,false,n),(4096,true,max(200,n÷10))), T in (Float32,Float64)
    setprecision(BigFloat,precision) do
        err = zero(T)
        accepted = 0
        while accepted < count
            l,a,t,step = input(rng,T,spread)
            all(isfinite,(l,a,t,step)) && t >= 0 && step > 0 || continue
            err = max(err,check(l,a,t,step))
            accepted += 1
        end
        println("$T, $precision bits, $accepted cases, max excess relative objective after rounding allowance: $err")
    end
end
