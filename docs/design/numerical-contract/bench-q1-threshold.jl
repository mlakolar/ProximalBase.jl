using LinearAlgebra
# Q1 options for soft thresholding a Float32 vector with a Float64 threshold t.
@inline st(x, t) = copysign(max(abs(x) - t, zero(x)), x)
function a_converted!(y::Vector{Float32}, x::Vector{Float32}, t::Float64)
    t32 = Float32(t)                       # (a) round the threshold once
    @inbounds @simd for i in eachindex(x)
        y[i] = st(x[i], t32)
    end
    y
end
function b_stored!(y::Vector{Float32}, x::Vector{Float32}, t::Float64)
    @inbounds @simd for i in eachindex(x)   # (b) every entry in Float64
        y[i] = Float32(st(Float64(x[i]), t))
    end
    y
end
function c_twoword!(y::Vector{Float32}, x::Vector{Float32}, t::Float64)
    th = Float32(t); tl = Float32(t - th)  # (c) threshold as th + tl, Float32 arithmetic
    @inbounds @simd for i in eachindex(x)
        a = abs(x[i])
        d = (a - th) - tl
        y[i] = copysign(max(d, 0f0), x[i])
    end
    y
end
function timeit(f!, y, x, t; reps)
    best = Inf
    for _ in 1:15
        s = time_ns()
        for _ in 1:reps
            Base.donotdelete(f!(y, x, t))
        end
        best = min(best, (time_ns() - s) / reps)
    end
    best
end
t = 1.0 - 2.0^-30
x1 = Float32[1, -1, 2, 0.5]
for f! in (a_converted!, b_stored!, c_twoword!)
    println(rpad(string(f!), 14), f!(similar(x1), x1, t))
end
for n in (1_000, 100_000, 10_000_000)
    x = randn(Float32, n); y = similar(x)
    reps = max(1, 10^8 ÷ n)
    ta = timeit(a_converted!, y, x, 0.1; reps)
    tb = timeit(b_stored!, y, x, 0.1; reps)
    tc = timeit(c_twoword!, y, x, 0.1; reps)
    println("n = $n: (a) $(round(ta, sigdigits=3)) ns, (b) $(round(tb / ta, digits=2))×, (c) $(round(tc / ta, digits=2))×")
end
println("isapprox default rtol at Float32: ", Base.rtoldefault(Float32), "  sqrt(eps) = ", sqrt(eps(Float32)))
println(Sys.CPU_NAME, "  Julia ", VERSION)
