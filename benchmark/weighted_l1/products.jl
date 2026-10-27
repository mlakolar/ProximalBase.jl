# Original PR #128 random-exponent probe, retained verbatim in its RNG and
# reference arithmetic so before/after maxima can be compared.
using ProximalBase, Random, Printf
setprecision(BigFloat, 512)
ulps(x::T, ref) where {T} = Float64(abs(BigFloat(x) - ref) / BigFloat(eps(T(max(abs(ref), floatmin(T))))))
for T in (Float32, Float64)
    rng = Xoshiro(3)
    E = T === Float32 ? 150 : 1100
    worst, mism = 0.0, 0
    for _ in 1:200_000
        a = (T(rand(rng)) + T(0.5)) * T(2)^rand(rng, -E:E) * (rand(rng) < 0.02 ? T(0) : T(1))
        b, c = ((T(rand(rng)) + T(0.5)) * T(2)^rand(rng, -E:E) for _ in 1:2)
        (isfinite(a) && isfinite(b) && isfinite(c)) || continue
        ref = BigFloat(a) * BigFloat(b) * BigFloat(c)
        p = ProximalBase._positive_product(a, b, c)
        r = T(ref)
        if isfinite(r) && (iszero(r) || abs(r) >= floatmin(T))
            iszero(r) ? (p == 0 || (mism += 1)) : (worst = max(worst, ulps(p, ref)))
        elseif !isfinite(r)
            isinf(p) || (mism += 1)
        end
    end
    @printf("%s %.8f ulp; %d mismatches\n", T, worst, mism)
    @assert mism == 0
    @assert worst <= 1.5
end
