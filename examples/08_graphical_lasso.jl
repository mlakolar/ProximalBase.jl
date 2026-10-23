# The graphical lasso by ADMM.
#
#     minimize  tr(SΩ) - log det Ω + λ Σᵢ≠ⱼ |Ωᵢⱼ|
#
# over precision matrices Ω, for a sample covariance S. ADMM splits Ω = Z:
#     Ω ← prox_{f/ρ}(Z - U)     f = tr(S⋅) - log det, by an eigendecomposition
#     Z ← prox_{g/ρ}(Ω + U)     g = the weighted ℓ1 norm, by soft thresholding
#     U ← U + Ω - Z
# `LogDetLoss` computes its proximal operator in a workspace it owns and is `Inf`
# outside the positive definite matrices; the weights of `NormL1` leave the
# diagonal unpenalized.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(8)
p, n = 30, 400
Ω0 = Matrix(SymTridiagonal(fill(1.0, p), fill(0.4, p - 1)))   # a chain graph
A = randn(rng, n, p) / cholesky(Ω0).U'                         # rows from N(0, Ω0⁻¹)
S = Symmetric(A'A / n)

λ = 0.2
f = LogDetLoss(S)
g = NormL1(λ, ones(p, p) - I)
@assert f(-Matrix(1.0I, p, p)) == Inf                          # outside the domain

function glasso_admm(f, g, p; ρ = 1.0, tol = 1e-8, maxiter = 10_000)
    Ω, Z, U = Matrix(1.0I, p, p), Matrix(1.0I, p, p), zeros(p, p)
    V, Zold = zeros(p, p), zeros(p, p)
    obj = Float64[]
    for iter in 1:maxiter
        V .= Z .- U
        prox!(Ω, f, V, 1 / ρ)
        copyto!(Zold, Z)
        V .= Ω .+ U
        gZ = value_and_prox!(Z, g, V, 1 / ρ)                    # g(Z), from the thresholding
        U .+= Ω .- Z
        push!(obj, f(Z) + gZ)
        primal, dual = norm(Ω - Z), ρ * norm(Z - Zold)          # the residuals of ADMM
        primal < tol * p && dual < tol * p && return Z, obj, iter
    end
    error("no convergence")
end

Ω̂, obj, iters = glasso_admm(f, g, p)

edges(M) = Set((i, j) for j in 1:p for i in j+1:p if !iszero(M[i, j]))
found, truth = edges(Ω̂), edges(Ω0)
println("ADMM converged in ", iters, " iterations; objective ", round(obj[end], digits = 6))
println("edges: ", length(found), " found, ", length(found ∩ truth), " of the ", length(truth),
        " true ones, ", length(setdiff(found, truth)), " others")

# optimality: S - Ω⁻¹ + λ Γ = 0, with Γᵢⱼ = sign(Ωᵢⱼ) off the diagonal where
# Ωᵢⱼ ≠ 0, |Γᵢⱼ| ≤ 1 where Ωᵢⱼ = 0, and no penalty on the diagonal
G = S - inv(Ω̂)
for j in 1:p, i in 1:p
    if i == j
        @assert abs(G[i, i]) < 1e-6
    elseif iszero(Ω̂[i, j])
        @assert abs(G[i, j]) <= λ + 1e-6
    else
        @assert abs(G[i, j] + λ * sign(Ω̂[i, j])) < 1e-6
    end
end
@assert isposdef(Ω̂) && truth ⊆ found
