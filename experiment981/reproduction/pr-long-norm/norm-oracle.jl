using ProximalBase
include("norm-cases.jl")

println("julia\ttype\tcase\tn\texponent\treference\tobserved\tabs_error\trelative_error\tstatus")
for T in (Float32, Float64)
    n = T === Float32 ? 1 << 22 : 1 << 20
    k = T === Float32 ? 100 : 650
    ordinary = ((n0, path, 0) for n0 in (1025, 100001, n)
        for path in ("zero_penalty", "zero_outside", "singleton_face", "signed_face"))
    cold = ((n, "zero_penalty", exp) for exp in (k, -k))
    for (n0, path, shift) in Iterators.flatten((ordinary, cold))
        v, x, c, exact = norm_case(T, n0, path; exponent = shift)
        observed = kkt_violation(NormLinf(c), v, x)
        ae = abs(BigFloat(observed) - exact)
        ne = ae / exact
        ok = observed isa T && ne <= 8 * eps(T)
        println(join((VERSION, T, path, n0, shift, exact, observed, ae, ne, ok ? "pass" : "fail"), '\t'))
    end
end
