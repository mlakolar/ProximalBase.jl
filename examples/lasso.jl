# Least-squares and lasso primitives
#
# *Assumes: gradients, least squares and basic Julia · You will learn: the mathematical loss/penalty scaling and corresponding operations.*
#
# For n observations, use f(β) = ||Xβ-y||²/(2n) and g(β) = λ||β||₁.
# LeastSquaresLoss itself divides by two, not by 2n.
using ProximalBase, LinearAlgebra
X = [1.0 2; 3 1; 0 2; 2 -1]
y = [1.0, 2, 3, -0.5]
n = length(y)
f = Scaled(LeastSquaresLoss(y, X), 1 / n)
g = NormL1(0.2)
β = [0.3, -0.4]
r = X * β - y
@assert f(β) ≈ sum(abs2, r) / (2 * n)
@assert g(β) ≈ 0.2 * sum(abs, β)
@assert gradient(f, β) ≈ X' * r / n
@assert prox(g, β, 0.1) ≈ sign.(β) .* max.(abs.(β) .- 0.02, 0)
# --- Fused and mutating calls ---
out = similar(β)
@assert value_and_gradient!(out, f, β) ≈ f(β)
@assert out ≈ X' * r / n
@assert value_and_prox!(out, g, β, 0.1) ≈ g(out)
@assert out ≈ [0.28, -0.38]
# The loss also has a proximal map. Check it against a direct linear solve.
γ = 0.3
loss = LeastSquaresLoss(y, X)
@assert prox(loss, β, γ) ≈ (I + γ * X' * X) \ (β + γ * X' * y)
# Matrix responses use the Frobenius norm and a coefficient column per response.
Y = hcat(y, -y)
B = hcat(β, -β)
multi = LeastSquaresLoss(Y, X)
@assert multi(B) ≈ sum(abs2, X * B - Y) / 2
@assert gradient(multi, B) ≈ X' * (X * B - Y)
(value = f(β), penalty = g(β), gradient = gradient(f, β))
