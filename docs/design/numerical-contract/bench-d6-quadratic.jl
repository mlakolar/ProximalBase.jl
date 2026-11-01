using ProximalBase, LinearAlgebra, Random
Random.seed!(1)
n = 2000
B = randn(Float32, n, n); A = B' * B / n + I       # stored Float32 SPD matrix
x = randn(n)                                        # Float64 argument
f = QuadraticFunction(A)
exact = setprecision(BigFloat, 256) do
    Ab = BigFloat.(A); xb = BigFloat.(x); (dot(xb, Ab * xb) / 2, Ab * xb)
end
v = f(x); g = gradient!(zeros(n), f, x)
relv(v) = Float64(abs(v - exact[1]) / abs(exact[1]))
println("value: type ", typeof(v), ", relative error ", relv(v))
println("gradient: relative error ", Float64(norm(g - exact[2]) / norm(exact[2])))
w = dot(x, A * x) / 2                               # widened: Float32 data, Float64 arithmetic
println("widened value (Float64 arithmetic on stored A): relative error ", relv(w))
println("eps(Float32) = ", eps(Float32), "   eps(Float64) = ", eps(Float64))
t(fn) = (fn(); minimum(@elapsed(fn()) for _ in 1:20) * 1e6)
x32 = Float32.(x); A64 = Float64.(A)
println("time f(x) today: ", round(t(() -> f(x)), digits = 1), " μs")
println("time Float32 data, Float64 arithmetic (generic A*x): ", round(t(() -> dot(x, A * x)), digits = 1), " μs")
println("time all-Float64 BLAS: ", round(t(() -> dot(x, A64 * x)), digits = 1), " μs")
println("time all-Float32 BLAS: ", round(t(() -> dot(x32, A * x32)), digits = 1), " μs")
