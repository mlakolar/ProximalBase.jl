# Coordinate-level operations
#
# *Assumes: partial derivatives and vector indexing · You will learn: coordinate-wise updates, residual tracking and KKT screens.*
# For the terminology behind [working-set screening](../glossary.md#Working-set).
using ProximalBase, LinearAlgebra
function illustrate_coordinates()
    X = [1.0 0; 0 2; 1 1]
    y = [2.0, -1, 1]
    n = length(y)
    f = (1 / n) * LeastSquaresLoss(y, X)
    g = NormL1(0.2)
    β = SparseIterate(2)
    residual = copy(y)  # y-Xβ
    a = vec(sum(abs2, X; dims=1)) / n
    for _ in 1:100
        for j in 1:2
            old = β[j]
            trial = old + dot(view(X, :, j), residual) / (n * a[j])
            new = prox_block(g, trial, j, 1 / a[j])
            β[j] = new
            residual .-= view(X, :, j) .* (new - old)
        end
    end
    @assert residual ≈ y - X * β
    grad = gradient(f, β)
    for j in 1:2
        if iszero(β[j])
            @assert kkt_violation(g, grad, β, j) <= 1e-10
        else
            @assert abs(grad[j] + 0.2 * sign(β[j])) < 1e-10
        end
    end
    ## The three-argument screen tests a coordinate hypothetically at zero.
    @assert kkt_violation(g, [0.1, -0.7], 2) ≈ 0.5
    @assert value_entry(g, -2.0, 1) ≈ 0.4
    (coefficients = Vector(β), residual = residual)
end
illustrate_coordinates()
