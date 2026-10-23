# Robust regression with the Huber loss, and sparse Poisson regression with an
# exposure offset: two losses of the form Σᵢ ℓ(ηᵢ, yᵢ), η = Xβ + offset, that
# `GLMLoss` provides with one product with X per value and gradient.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(6)

# FISTA with a backtracking line search (see 05_logistic_regression.jl)
function fista_backtracking(f, g, p; L = 1.0, iters = 3000)
    β, βold, z, gz, βnew, d = (zeros(p) for _ in 1:6)
    t = 1.0
    for _ in 1:iters
        fz = value_and_gradient!(gz, f, z)
        while true
            βnew .= z .- gz ./ L
            prox!(βnew, g, βnew, 1 / L)
            d .= βnew .- z
            f(βnew) - fz - dot(gz, d) <= L / 2 * dot(d, d) + 1e-12 * max(1, abs(fz)) && break
            L *= 2
        end
        copyto!(βold, β)
        copyto!(β, βnew)
        tnew = (1 + sqrt(1 + 4t^2)) / 2
        z .= β .+ ((t - 1) / tnew) .* (β .- βold)
        t = tnew
    end
    β
end

# --- Huber regression with gross outliers ----------------------------------
# h_δ(r) = r²/2 for |r| ≤ δ and δ(|r| - δ/2) beyond: quadratic for small
# residuals, linear for large ones, so an outlier pulls the fit with a bounded
# force. There is no penalty here: `ZeroFunction()` has the identity as its
# proximal operator, which turns FISTA into accelerated gradient descent.
n, p = 300, 4
X = [ones(n) randn(rng, n, p - 1)]
β_true = [1.0, 2.0, -1.0, 0.5]
y = X * β_true + 0.5 * randn(rng, n)
bad = randperm(rng, n)[1:30]                              # 10% of the responses,
y[bad] .+= 15 .* sign.(X[bad, 2])                         # shifted along the second column

β_ls = X \ y                                              # least squares
β_huber = fista_backtracking(HuberLoss(y, X, 1.0), ZeroFunction(), p)

println("least squares ", round.(β_ls, digits = 3), ", error ", round(norm(β_ls - β_true), digits = 3))
println("Huber         ", round.(β_huber, digits = 3), ", error ", round(norm(β_huber - β_true), digits = 3))
@assert norm(gradient(HuberLoss(y, X, 1.0), β_huber)) < 1e-6
@assert norm(β_huber - β_true) < norm(β_ls - β_true) / 4

# --- sparse Poisson regression with an exposure offset ----------------------
# Counts yᵢ ~ Poisson(eᵢ exp(xᵢ'β)) with exposure eᵢ: the offset log eᵢ enters the
# linear predictor with a fixed coefficient 1. The loss is Σᵢ exp(ηᵢ) - yᵢηᵢ.
function poisson(rng, μ)                                  # Knuth's algorithm, small μ
    k, q, s, u = 0, exp(-μ), exp(-μ), rand(rng)
    while u > s
        k += 1
        q *= μ / k
        s += q
    end
    k
end

n, p = 500, 20
X = randn(rng, n, p) ./ 2
β_true = [0.8; -0.6; 0.5; zeros(p - 3)]
exposure = 1 .+ 4 .* rand(rng, n)
y = Float64[poisson(rng, exposure[i] * exp(dot(X[i, :], β_true))) for i in 1:n]

f = PoissonLoss(y, X; offset = log.(exposure))
λmax = norm(gradient(f, zeros(p)), Inf)                   # the smallest λ with β = 0
λ = 0.1 * λmax
β = fista_backtracking(f, NormL1(λ), p)

println("Poisson, λ = ", round(λ, digits = 2), ": nonzero ", findall(!iszero, β),
        ", β[1:3] = ", round.(β[1:3], digits = 3), " (truth ", β_true[1:3], ")")
grad = gradient(f, β)
@assert all(abs(grad[j] + λ * sign(β[j])) < 1e-6 for j in 1:p if !iszero(β[j]))
@assert all(abs(grad[j]) <= λ + 1e-6 for j in 1:p if iszero(β[j]))
@assert all(!iszero, β[1:3]) && norm(β - β_true) < 0.3
