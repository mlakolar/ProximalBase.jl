# Sparse logistic regression with an unpenalized intercept, by FISTA with a
# backtracking line search.
#
#     minimize  Σᵢ log(1 + exp(ηᵢ)) - yᵢηᵢ + λ Σⱼ wⱼ |βⱼ|,   η = Xβ, yᵢ ∈ {0, 1}
#
# The first column of X is the intercept, and the weight w₁ = 0 leaves it
# unpenalized. `value_and_gradient!` gives the loss and its gradient in one pass.
# The gradient of the logistic loss is Lipschitz with constant ‖X‖²/4 (the
# curvature bound 1/4 of the scalar loss), but a line search usually finds a
# larger step.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(5)
n, p = 400, 40
X = [ones(n) randn(rng, n, p)]
β_true = [0.5; 2.0; -2.0; 1.5; zeros(p - 3)]
y = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-X * β_true)))

f = LogisticLoss(y, X)
λ = 15.0
w = [0.0; ones(p)]                                       # no penalty on the intercept
g = NormL1(λ, w)
# (the ℓ1 penalty shrinks the selected coefficients toward zero; MCP or SCAD,
# as in 03_nonconvex_penalties.jl, would not)

# FISTA with backtracking: increase L until the quadratic model at z bounds f.
# Near the solution the step d is tiny and both sides of the test agree to
# rounding error, so the test allows a slack relative to |f(z)|.
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
    β, L
end

β, L = fista_backtracking(f, g, p + 1)
Lbound = ProximalBase.curvature(ProximalBase.Logistic()) * opnorm(X)^2
accuracy = sum((X * β .> 0) .== (y .== 1)) / n

println("intercept ", round(β[1], digits = 3), ", coefficients ", round.(β[2:5], digits = 3),
        " (truth ", β_true[2:5], ")")
println("nonzero coefficients: ", findall(!iszero, β[2:end]))
println("step found by the line search 1/", round(L, digits = 1), ", from the bound 1/", round(Lbound, digits = 1))
println("training accuracy ", round(accuracy, digits = 3))

# optimality: ∇f(β)₁ = 0 for the intercept, ∇f(β)ⱼ = -λ sign(βⱼ) for a nonzero
# coefficient, and |∇f(β)ⱼ| ≤ λ for a zero one
grad = gradient(f, β)
@assert abs(grad[1]) < 1e-6
@assert all(abs(grad[j] + λ * sign(β[j])) < 1e-6 for j in 2:p+1 if !iszero(β[j]))
@assert all(abs(grad[j]) <= λ + 1e-6 for j in 2:p+1 if iszero(β[j]))
@assert all(!iszero, β[2:4])
@assert L <= 2Lbound && accuracy > 0.8
