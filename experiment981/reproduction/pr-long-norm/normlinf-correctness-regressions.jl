# Run with the ProximalBase checkout under test as the active project.
# Expected values come from analytic projections of the stored inputs.
# Float32 and Float64 are checked by default. Pass --generic to add two
# compatibility checks outside the production scope.
using ProximalBase

function check(score, name, expected; rtol=zero(expected))
    observed = try
        score()
    catch err
        err isa InterruptException && rethrow()
        replace(sprint(showerror, err), '\n' => ' ')
    end
    passed = observed isa typeof(expected) && isfinite(observed) &&
        isapprox(observed, expected; rtol, atol=zero(expected))
    println(join((name, typeof(expected), observed, expected, passed ? "pass" : "fail"), '\t'))
    return Int(!passed)
end

function check_ties(name, tie::Float32, n_ties, radius)
    v = [1f0; fill(tie, n_ties)]
    z = zeros(Float32, length(v))
    face = fill(-1f0, length(v))
    # Only the first entry receives projected mass.
    exact = setprecision(BigFloat, 160) do
        head_residual = 1 - BigFloat(radius)
        @assert head_residual > BigFloat(tie)
        Float32(sqrt(head_residual^2 + n_ties * BigFloat(tie)^2))
    end
    # This tolerance is for these two well-conditioned reference cases.
    rtol = 32 * eps(Float32)
    return check(() -> kkt_violation(NormLinf(radius), v, z), name * "_ball", exact; rtol) +
        check(() -> kkt_violation(NormLinf(radius), v, face), name * "_face", exact; rtol) +
        check(() -> kkt_violation(Scaled(NormLinf(radius / 2), 2f0), v, face),
              name * "_scaled_face", exact; rtol)
end

function check_stationarity(::Type{T}) where {T}
    high = ldexp(one(T), precision(T) + 3)
    tail = ldexp(one(T), -20)
    v = [high; one(T); tail; zeros(T, 4093)]
    face = fill(-one(T), length(v))
    # The exact threshold is 1/2. The first two entries have equal residuals.
    exact = T(sqrt(BigFloat(1) / 2 + BigFloat(tail)^2))
    rtol = 8 * eps(T)
    failures = check(() -> kkt_violation(NormLinf(high), v, zero(v)),
                     "nonstationary_cancellation_ball", exact; rtol)
    failures += check(() -> kkt_violation(NormLinf(high), v, face),
                      "nonstationary_cancellation_face", exact; rtol)

    v = [high; fill(T(9), 256)]
    level = high + T(2304)
    # The exact mass of the positive entries equals the radius.
    failures += check(() -> kkt_violation(NormLinf(level), v, zero(v)),
                      "exact_stationarity_ball", zero(T))
    failures += check(() -> kkt_violation(NormLinf(level), v, fill(-one(T), length(v))),
                      "exact_stationarity_face", zero(T))
    return failures
end

function check_subnormal(::Type{T}) where {T}
    spacing = nextfloat(zero(T))
    v = zeros(T, 4096)
    face = fill(-one(T), length(v))
    excess = 2047 * spacing
    # The common residual is the excess divided by 4096. Its norm is
    # 31.984375 spacings, which rounds to 32 in T.
    expected = 32 * spacing
    failures = check(() -> kkt_violation(NormLinf(excess), v, face),
                     "all_subnormal_face", expected)
    v[1] = floatmin(T)
    level = v[1] + excess
    failures += check(() -> kkt_violation(NormLinf(level), v, face),
                      "mixed_scale_subnormal_face", expected)
    return failures
end

function check_precision()
    x = setprecision(BigFloat, 256) do
        [BigFloat(1), nextfloat(BigFloat(1))]
    end
    failures = setprecision(BigFloat, 128) do
        v = BigFloat[-1, -2]
        check(() -> kkt_violation(NormLinf(BigFloat(2)), v, x),
              "bigfloat_unique_face", BigFloat(1))
    end
    v = fill(Float16(200), 30_000)
    expected = Float16(200 * sqrt(BigFloat(length(v))))
    failures += check(() -> kkt_violation(NormLinf(Float16(0)), v, zero(v)),
                      "float16_zero_radius_cold_norm", expected)
    return failures
end

function main()
    println("case\ttype\tobserved\texpected\tstatus")
    failures = check_ties("published_100001", 0.009486833f0, 100_000, 0.1f0)
    failures += check_ties("reconstructed_1000001", 0.0002515878878f0, 1_000_000, 0.1f0)
    for T in (Float32, Float64)
        failures += check_stationarity(T)
        failures += check_subnormal(T)
    end
    return failures
end

ARGS in (String[], ["--generic"]) || error("usage: normlinf-correctness-regressions.jl [--generic]")
failures = main()
if "--generic" in ARGS
    failures += check_precision()
end
flush(stdout)
iszero(failures) || exit(1)
