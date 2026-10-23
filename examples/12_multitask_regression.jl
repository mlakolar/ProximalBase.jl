# Multitask regression: k regressions on the same predictors that share their
# relevant predictors, by penalizing the rows of the p × k coefficient matrix.
#
#     minimize  ‖Y - XB‖² / 2 + λ Σᵢ √k ‖B[i, :]‖₂
#
# `LeastSquaresLoss(Y, X)` takes a matrix Y and gives the gradient X'(XB - Y) for
# a matrix B. The rows of a p × k matrix are the groups [i:p:p*k for i in 1:p]
# of its linear indices, so `GroupNormL2` with these groups selects whole rows,
# the predictors shared by all tasks.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(12)
n, p, k = 100, 40, 5
X = randn(rng, n, p)
relevant = [3, 11, 27]
B_true = zeros(p, k)
B_true[relevant, :] = 2 .* randn(rng, 3, k)
Y = X * B_true + 0.5 * randn(rng, n, k)

f = LeastSquaresLoss(Y, X)
rows = [i:p:p*k for i in 1:p]                          # the rows of B as linear indices
λmax = maximum(norm((X' * Y)[i, :]) / sqrt(k) for i in 1:p)
λ = 0.2 * λmax
g = GroupNormL2(λ, rows)                               # weights √k, the group sizes

L = opnorm(X)^2
B, Bold, Z, G = zeros(p, k), zeros(p, k), zeros(p, k), zeros(p, k)
t = 1.0
for _ in 1:2000
    gradient!(G, f, Z)
    copyto!(Bold, B)
    B .= Z .- G ./ L
    prox!(B, g, B, 1 / L)                             # shrink each row as a block
    tnew = (1 + sqrt(1 + 4t^2)) / 2
    Z .= B .+ ((t - 1) / tnew) .* (B .- Bold)
    global t = tnew
end

selected = [i for i in 1:p if !iszero(B[i, :])]
println("λ = ", round(λ, digits = 2), ": selected rows ", selected, " (truth ", relevant, ")")
println("error ", round(norm(B - B_true) / norm(B_true), digits = 3), " relative to ‖B_true‖")

# optimality, row by row: ∇f(B)[i, :] = -λ√k B[i, :] / ‖B[i, :]‖ for a nonzero
# row, and ‖∇f(B)[i, :]‖ ≤ λ√k for a zero row
gradient!(G, f, B)
for i in 1:p
    if iszero(B[i, :])
        @assert norm(G[i, :]) <= λ * sqrt(k) + 1e-6
    else
        @assert norm(G[i, :] + λ * sqrt(k) * B[i, :] / norm(B[i, :])) < 1e-6
    end
end
@assert selected == relevant
