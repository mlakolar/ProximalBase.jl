using ProximalBase
using Printf

println("name\tshape\tn\tradius\tstored_tie\tvalue\texact\tabs_error\tnormalized_error")
function check(label, tie, count, radius; head_first = true)
    magnitudes = head_first ? [1f0; fill(tie, count)] : [fill(tie, count); 1f0]
    at_zero = zeros(Float32, count + 1)
    at_face = ones(Float32, count + 1)
    exact, scale = setprecision(BigFloat, 160) do
        t = BigFloat(tie)
        c = BigFloat(radius)
        @assert 1 - c > t
        sqrt((1 - c)^2 + count * t^2), sqrt(1 + count * t^2) + c
    end
    for (shape, v, x) in (("zero", magnitudes, at_zero), ("face", -magnitudes, at_face))
        measured = kkt_violation(NormLinf(radius), v, x)
        error = abs(BigFloat(measured) - exact)
        @printf "%s\t%s\t%d\t%.10g\t%.10g\t%.12g\t%.16g\t%.8g\t%.8g\n" label shape count+1 radius tie measured exact error (error / scale)
    end
end

check("round3_published", 0.009486833f0, 100_000, .1f0)
let m = 1_000_000, radius = 0.1f0, rho = 0.06
    tie = Float32((1 - Float64(radius)) / (1 + rho * m^2 * (eps(Float32) / 2)))
    check("round3_million_reconstruction_head_first", tie, m, radius)
    check("round3_million_reconstruction_head_last", tie, m, radius; head_first = false)
end
check("issue1003_case_4", 0.0027806521f0, 1_000_000, 0.5f0; head_first = false)
