function rejected(PB, ::Type{T}, n) where {T}
    n >= 4 || return nothing
    rng = Xoshiro(1138 + n)
    setprecision(BigFloat,2304) do
        for i in 1:360
            x = if i <= 16
                y = zeros(T,n)
                y[1] = one(T)
                y[2:4] .= eps(T) * T(0.5 + i/128)
                y
            else
                rand(rng, T, n) .* T.(exp2.(rand(rng, -20:4, n))) .* rand(rng, T[-1,1], n)
            end
            S = sum(v -> abs(BigFloat(v)), x)
            r = T(S)
            BigFloat(r) < S && (r = nextfloat(r))
            !iszero(PB.IndBallL1(r)(x)) && return (x,r)
        end
        nothing
    end
end

function micro_fixtures(emit, PB, quick)
    for T in (Float32, Float64), n in (quick ? (2, 2048) : (2,4,16,256,2048))
        signs = [isodd(i) ? -one(T) : one(T) for i in 1:n]
        dyadic = signs ./ T(n)
        deep = dyadic ./ 4
        deep[end] = -zero(T)
        emit("deep-interior", deep, one(T))
        emit("one-ulp-interior", dyadic, nextfloat(one(T)))
        emit("exact-boundary", dyadic, one(T))
        rounded_copy = zeros(T,n)
        rounded_copy[1], rounded_copy[end] = -one(T), eps(T)/4
        n > 2 && (rounded_copy[2] = -zero(T))
        emit("rounded-infeasible-copy", rounded_copy, one(T))
        emit("active-tied", signs, T(0.001 * n))
        emit("active-exact", signs, T(n)/4)
        fixture = rejected(PB,T,n)
        fixture !== nothing && emit("rounded-feasible-rejection", fixture...)
    end
end
