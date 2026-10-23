# The lasso three ways: proximal gradient descent (ISTA), its accelerated version
# (FISTA), and ADMM, all built from the same two functions.
#
#     minimize  f(β) + g(β),   f(β) = ‖y - Xβ‖² / 2,   g(β) = λ ‖β‖₁
#
# `LeastSquaresLoss` is smooth and proximable, so one object gives the gradient
# that (F)ISTA needs and the proximal operator that ADMM needs; its proximal
# operator comes from a thin SVD of X computed once.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
n, p = 200, 50
X = randn(rng, n, p)
β_true = [3.0; -2.0; 1.5; zeros(p - 3)]
y = X * β_true + 0.5 * randn(rng, n)
λ = 20.0

f = LeastSquaresLoss(y, X)
g = NormL1(λ)
L = opnorm(X)^2                    # Lipschitz constant of ∇f

# ISTA: β ← prox_{g/L}(β - ∇f(β)/L). `value_and_prox!` returns g at the new point,
# so tracking the objective costs no extra pass.
function ista(f, g, L, p; iters = 2000)
    β, grad = zeros(p), zeros(p)
    obj = Float64[]
    for _ in 1:iters
        gradient!(grad, f, β)
        β .-= grad ./ L
        gβ = value_and_prox!(β, g, β, 1 / L)
        push!(obj, f(β) + gβ)
    end
    β, obj
end

# FISTA: the same step taken at an extrapolated point z
function fista(f, g, L, p; iters = 500)
    β, βold, z, grad = zeros(p), zeros(p), zeros(p), zeros(p)
    t = 1.0
    for _ in 1:iters
        gradient!(grad, f, z)
        copyto!(βold, β)
        β .= z .- grad ./ L
        prox!(β, g, β, 1 / L)
        tnew = (1 + sqrt(1 + 4t^2)) / 2
        z .= β .+ ((t - 1) / tnew) .* (β .- βold)
        t = tnew
    end
    β
end

# ADMM on the split β = z: β ← prox_{f/ρ}(z - u), z ← prox_{g/ρ}(β + u), u ← u + β - z
function admm(f, g, p; ρ = 200.0, iters = 1000)
    β, z, u, v = zeros(p), zeros(p), zeros(p), zeros(p)
    for _ in 1:iters
        v .= z .- u
        prox!(β, f, v, 1 / ρ)
        v .= β .+ u
        prox!(z, g, v, 1 / ρ)
        u .+= β .- z
    end
    z
end

# the largest violation of the optimality conditions of the lasso:
# ∇f(β)ᵢ = -λ sign(βᵢ) where βᵢ ≠ 0, and |∇f(β)ᵢ| ≤ λ where βᵢ = 0
function kkt(f, λ, β)
    grad = gradient(f, β)
    maximum(i -> iszero(β[i]) ? max(abs(grad[i]) - λ, 0.0) : abs(grad[i] + λ * sign(β[i])), eachindex(β))
end

β_ista, obj = ista(f, g, L, p)
β_fista = fista(f, g, L, p)
β_admm = admm(f, g, p)

for (name, β) in (("ISTA", β_ista), ("FISTA", β_fista), ("ADMM", β_admm))
    println(rpad(name, 6), "objective ", round(f(β) + g(β), digits = 6), ", nonzeros ",
            count(!iszero, β), ", KKT violation ", round(kkt(f, λ, β), sigdigits = 2))
end
println("first coefficients: ", round.(β_fista[1:4], digits = 3), " (truth ", β_true[1:4], ")")

@assert all(diff(obj) .<= 1e-9)                          # ISTA decreases the objective
@assert kkt(f, λ, β_ista) < 1e-6 && kkt(f, λ, β_fista) < 1e-6 && kkt(f, λ, β_admm) < 1e-6
@assert norm(β_ista - β_fista) < 1e-6 && norm(β_admm - β_fista) < 1e-6
@assert findall(!iszero, β_fista) == 1:3                 # the support of β_true
