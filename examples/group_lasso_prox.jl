# Group-lasso proximal building blocks
#
# *Assumes: the lasso building blocks and grouped variables · You will learn: the group penalty, its group soft thresholding and pure compact block updates.*
# `GroupNormL2(λ, groups, w)` is ``g(β)=λ\sum_k w_k\|β_{G_k}\|_2``. This page checks the penalty and its
# [proximal operator](../glossary.md#Proximal-operator) against their formulas and takes one caller-written
# block step. It is not a solver: group-lasso coordinate descent belongs to a solver package.
using ProximalBase, LinearAlgebra
# ---- The penalty ----
# Noncontiguous, arbitrarily ordered groups; the default weights are √|Gₖ|.
groups = [[1, 4, 2], [5, 3], [6]]
λ = 0.4
g = GroupNormL2(λ, groups)
w = [sqrt(3), sqrt(2), 1.0]                  # the default weights, for the checks below
β = [1.0, -2.0, 0.25, 2.0, 0.0, -1.5]
@assert g(β) ≈ λ * (sqrt(3) * norm(β[[1, 4, 2]]) + sqrt(2) * norm(β[[5, 3]]) + abs(β[6]))
# A group of one index is the ℓ₁ term with the same weight.
@assert GroupNormL2(λ, [[i] for i in 1:6])(β) ≈ NormL1(λ)(β)
# ---- Group soft thresholding ----
# Each group shrinks towards zero in Euclidean norm: max(1 - γλwₖ/‖βₖ‖, 0) βₖ.
γ = 0.5
p = prox(g, β, γ)
for (k, G) in enumerate(groups)
    c = γ * λ * w[k]
    @assert p[G] ≈ max(1 - c / norm(β[G]), 0) * β[G]
end
@assert p[[5, 3]] == [0, 0]   # ‖(0, 0.25)‖ = 0.25 is below γλ√2 ≈ 0.28
y = similar(β)
@assert value_and_prox!(y, g, β, γ) ≈ g(y) && y == p
# ---- Pure compact block operations ----
# `prox_block!` reads a compact trial for one group, in the order of its indices, and writes only its
# output buffer: it does not activate or update an iterate.
x = BlockIterate(6, g)                       # an iterate with the groups of g
@assert collect(blockindices(x, 1)) == [1, 4, 2] && nactive(x) == 0
u = [3.0, 4.0, 12.0]                         # trial for group 1, ‖u‖ = 13
out = similar(u)
prox_block!(out, g, u, 1, 1.0)
@assert out ≈ (1 - λ * sqrt(3) / 13) * u && nactive(x) == 0
# The caller writes the result; only this write activates the group.
copyto!(blockview(x, 1), out)
@assert collect(activecoordinates(x)) == [1] && Vector(x)[[1, 4, 2]] == out
# A zero result written to an inactive group leaves it inactive.
copyto!(blockview(x, 2), prox_block(g, [0.1, -0.1], 2, 1.0))
@assert nactive(x) == 1
# ---- Optimality screen and one block step ----
# For least squares f(β) = ‖y - Xβ‖²/(2n), the zero group k is optimal when ‖∇f[Gₖ]‖ ≤ λwₖ, and
# kkt_violation returns ‖∇f[Gₖ]‖ - λwₖ.
X = [1.0 0.0 2.0 1.0 0.0 1.0;
     0.0 1.0 1.0 0.0 2.0 0.0;
     1.0 1.0 0.0 2.0 1.0 1.0;
     2.0 0.0 1.0 1.0 0.0 0.0]
yobs = [3.0, -1.0, 2.0, 4.0]
n = size(X, 1)
f = (1 / n) * LeastSquaresLoss(yobs, X)
z = BlockIterate(6, g)
grad = gradient(f, Vector(z))
for k in 1:3
    @assert kkt_violation(g, grad, z, k) ≈ norm(grad[groups[k]]) - λ * w[k]
end
# A caller-written block proximal-gradient step for group 1 with the step 1/Lₖ,
# Lₖ = ‖X[:, G₁]‖₂²/n, decreases f + g.
G = collect(blockindices(z, 1))
L = opnorm(X[:, G])^2 / n
trial = Vector(blockview(z, 1)) - grad[G] / L
step = prox_block(g, trial, 1, 1 / L)
before = f(Vector(z)) + g(z)
copyto!(blockview(z, 1), step)
@assert f(Vector(z)) + g(z) < before
(before, f(Vector(z)) + g(z))
