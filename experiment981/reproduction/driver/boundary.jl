using ProximalBase
using Printf
using Test

function probe(T, label, v, x, c, ref)
    scale = sqrt(sum(z -> abs2(BigFloat(z)), v)) + BigFloat(c)
    got = try
        kkt_violation(NormLinf(c), v, x)
    catch ex
        string(nameof(typeof(ex)))
    end
    err = got isa Real ? abs(BigFloat(got) - ref) : BigFloat(Inf)
    @printf "%s\t%s\t%d\t%s\t%.16g\t%.10g\t%.10g\n" T label length(x) string(got) ref err (err / scale)
end

println("type\tcase\tn\tobserved\treference\tabs_error\tnormalized_error")
for T in (Float32, Float64)
    high = ldexp(one(T), precision(T) + 3)
    tail = ldexp(one(T), -20)
    v = [high; one(T); tail; zeros(T, 4093)]
    ref = sqrt(BigFloat(0.5) + BigFloat(tail)^2)
    probe(T, "cancellation_ball", v, zero(v), high, ref)
    probe(T, "cancellation_face", v, fill(-one(T), length(v)), high, ref)

    v = [high; fill(T(9), 256)]
    radius = high + T(2304)
    probe(T, "exact_stationarity_ball", v, zero(v), radius, BigFloat(0))
    probe(T, "exact_stationarity_face", v, fill(-one(T), length(v)), radius, BigFloat(0))
end

v, x = setprecision(BigFloat, 256) do
    BigFloat[-1, -2], BigFloat[1, nextfloat(BigFloat(1))]
end
setprecision(BigFloat, 128) do
    @assert abs(convert(BigFloat, x[1])) < abs(convert(BigFloat, x[2]))
    probe(BigFloat, "mixed_stored_precision_unique_face", v, x, BigFloat(2), BigFloat(1))
end
for T in (Float32, Float64)
    # Scaled-prefix replay on C, power-of-two routing on the other candidates.
    let t = floatmax(T)
        v = T[-t/2, -t/2, -t/2]
        x = ones(T, length(v))
        c = t/4
        outcome = try
            @inferred kkt_violation(NormLinf(c), v, x)
            "inferred"
        catch ex
            sprint(showerror, ex)
        end
        println(stderr, "inference in overflow-range face for ", T, " ", outcome)
    end
end
