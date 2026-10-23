# The group lasso, by proximal gradient descent and by block coordinate descent.
#
#     minimize  ‖y - Xβ‖² / 2 + λ Σₖ √|Gₖ| ‖β[Gₖ]‖₂
#
# `GroupNormL2(λ, groups)` is the penalty, with weights √|Gₖ| by default; its
# proximal operator shrinks each group toward zero as a block. On an ordinary
# vector it serves FISTA. On an `AtomIterate` that shares its groups, the
# coordinates are the groups: `prox_coordinate!` applies the proximal step to one
# group, and the active groups are the ones that are nonzero.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(4)
groups = [1:3, 4:8, 9:10, 11:20, 21:24, 25:30]          # groups of unequal sizes
n, p = 150, 30
X = randn(rng, n, p)
β_true = zeros(p)
β_true[4:8] .= [1.0, -1.0, 0.5, 0.5, -0.5]
β_true[21:24] .= 2.0
y = X * β_true + 0.5 * randn(rng, n)

f = LeastSquaresLoss(y, X)
g = GroupNormL2(30.0, groups)

# FISTA on a plain vector
function fista(f, g, L, p; iters = 3000)
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
β_pg = fista(f, g, opnorm(X)^2, p)

# block coordinate descent on an AtomIterate: for group k, a gradient step with
# step 1/Lₖ, Lₖ = ‖X[:, Gₖ]‖², then the group's proximal step
function block_cd(X, y, g; sweeps = 2000)
    β = AtomIterate(size(X, 2), g)                     # shares the groups of g
    r = copy(y)                                        # r = y - Xβ
    Lk = [opnorm(X[:, G])^2 for G in β.groups]
    for _ in 1:sweeps, k in 1:ncoordinates(β)
        G = β.groups[k]
        old = Vector(atom(β, k))
        atom(β, k) .+= view(X, :, G)' * r ./ Lk[k]
        prox_coordinate!(β, g, k, 1 / Lk[k])
        mul!(r, view(X, :, G), old .- atom(β, k), 1, 1)
    end
    dropzeros!(β)                                      # forget groups that ended at zero
end
β_cd = block_cd(X, y, g)

println("active groups: ", activecoordinates(β_cd), " of ", length(groups))
for (k, G) in enumerate(groups)
    println("group ", k, " ", rpad(string(G), 7), " ‖β[G]‖ = ", rpad(round(norm(β_pg[G]), digits = 4), 8),
            "truth ", round(norm(β_true[G]), digits = 4))
end

# optimality: ∇f(β)[G] = -λ wₖ β[G] / ‖β[G]‖ for a nonzero group, and
# ‖∇f(β)[G]‖ ≤ λ wₖ for a zero group, which `kkt_violation` measures
grad = gradient(f, β_pg)
for (k, G) in enumerate(groups)
    w = 30.0 * sqrt(length(G))
    if iszero(β_pg[G])
        @assert norm(grad[G]) <= w + 1e-6
        @assert kkt_violation(g, grad, β_cd, k) <= 1e-6
    else
        @assert norm(grad[G] + w * β_pg[G] / norm(β_pg[G])) < 1e-6
    end
end
@assert norm(β_pg - Vector(β_cd)) < 1e-6
@assert Set(activecoordinates(β_cd)) == Set([2, 5])
@assert all(iszero, β_pg[[1:3; 9:20; 25:30]])
