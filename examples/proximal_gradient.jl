# Proximal-gradient iteration
#
# *Assumes: gradient descent and convexity · You will learn: a complete proximal-gradient iteration and its optimality check.*
#
# This loop is educational caller code. ProximalBase supplies gradient and prox.
using ProximalBase, LinearAlgebra
function illustrate_pg()
    X = [1.0 0; 0 2; 1 0; 0 2]
    y = [2.0, -2, 2, -2]
    n = length(y)
    f = (1 / n) * LeastSquaresLoss(y, X)
    g = NormL1(0.2)
    γ = 1 / lipschitz(f)
    β = zeros(2)
    grad = similar(β)
    old_objective = f(β) + g(β)
    for _ in 1:120
        gradient!(grad, f, β)
        β .-= γ .* grad
        prox!(β, g, β, γ)
        objective = f(β) + g(β)
        @assert objective <= old_objective + 1e-13
        old_objective = objective
    end
    ## The columns are orthogonal, so the exact lasso solution is available.
    a = vec(sum(abs2, X; dims=1)) / n
    b = X' * y / n
    reference = sign.(b) .* max.(abs.(b) .- 0.2, 0) ./ a
    @assert β ≈ reference
    ## Compute the gradient mapping explicitly: no solver helper is needed.
    G = (β - prox(g, β - γ * gradient(f, β), γ)) / γ
    @assert norm(G, Inf) < 1e-10
    @assert maximum(abs.(gradient(f, β) + 0.2 .* sign.(β))) < 1e-10
    (coefficients = β, mapping_norm = norm(G, Inf))
end
illustrate_pg()
