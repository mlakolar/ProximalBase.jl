# Separable penalties with constraints.
#
# Constrained(f, C) is a convex separable f restricted to a box, a hyperplane, a
# halfspace or a simplex C, with its proximal operator. On a box it clamps the
# proximal operator of f; on a hyperplane ⟨a, x⟩ = b it is prox_{γf}(x - γνa)
# for the multiplier ν that meets the constraint. Three problems:
#
# - regression on compositional data, whose coefficients sum to zero, with the
#   lasso penalty on that hyperplane;
# - the lasso with coefficients in [0, 1], by coordinate descent, since a
#   separable penalty on a box stays separable;
# - a soft selection of the k largest of n scores, by the projection onto the
#   capped simplex {z : 0 ≤ z ≤ 1, Σᵢ zᵢ = k}.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(16)

function fista(f, g, p; iters = 5000)
    L = lipschitz(f)
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

centered(A) = A .- sum(A; dims = 1) ./ size(A, 1)

# --- the lasso on compositional data -----------------------------------------
# Each row of W holds the proportions of p parts, positive and summing to one,
# as in microbiome or household-budget data. The response depends on the log
# proportions, y = log.(W) β + ε with Σⱼ βⱼ = 0 (Lin, Shi, Feng and Li, 2014).
# The constraint makes the fit invariant to the scale of each row: scaling row i
# by cᵢ adds log cᵢ to each of its predictors, and so log cᵢ Σⱼ βⱼ = 0 to its fit.
# An intercept, unpenalized, is removed by centering.
n, p = 100, 30
W = exp.(randn(rng, n, p))
W ./= sum(W; dims = 2)
β_true = [1.0; -0.8; 0.6; -0.5; -0.3; zeros(p - 5)]   # sums to zero
y = log.(W) * β_true .+ 0.2 .* randn(rng, n)

λ = 0.07
f = (1 / n) * LeastSquaresLoss(centered(y), centered(log.(W)))
g = Constrained(NormL1(λ), IndHyperplane(1, 0))
β = fista(f, g, p)
println("zero-sum lasso: support ", findall(!iszero, β), ", Σβ = ", sum(β))
@assert abs(sum(β)) < 1e-12 && norm(gradient_mapping(f, g, β, 1 / lipschitz(f))) < 1e-9
@assert findall(!iszero, β) ⊇ 1:5

# the optimality conditions: ∇f(β)ⱼ + ν + λ sign(βⱼ) = 0 where βⱼ ≠ 0, and
# |∇f(β)ⱼ + ν| ≤ λ where βⱼ = 0, for one multiplier ν
grad = gradient(f, β)
S = findall(!iszero, β)
ν = -grad[S[1]] - λ * sign(β[S[1]])
@assert all(abs(grad[j] + ν + λ * sign(β[j])) < 1e-8 for j in S)
@assert all(abs(grad[j] + ν) <= λ + 1e-8 for j in 1:p if iszero(β[j]))

# scaling the rows of W leaves the estimate unchanged, up to the accuracy of
# FISTA; the estimate of the lasso without the constraint changes
c = exp.(2 .* randn(rng, n))
fc = (1 / n) * LeastSquaresLoss(centered(y), centered(log.(c .* W)))
βc = fista(fc, g, p)
β1, β1c = fista(f, NormL1(λ), p), fista(fc, NormL1(λ), p)
println("rows rescaled: zero-sum lasso moves by ", norm(βc - β), ", the lasso by ", norm(β1c - β1))
@assert norm(βc - β) < 1e-6 && norm(β1c - β1) > 1e-3

# --- the lasso on [0, 1] by coordinate descent -------------------------------
# NormL1 on the box 0 ≤ β ≤ 1 is separable: prox_entry clamps the soft
# thresholding of each coordinate, so coordinate descent with exact residuals
# solves the problem one coefficient at a time.
X = randn(rng, n, p)
β_true = [2.0; 1.5; 0.5; zeros(p - 3)]                  # the first two exceed the bound
y = X * β_true + 0.3 * randn(rng, n)
f = (1 / n) * LeastSquaresLoss(y, X)
g = Constrained(NormL1(0.05), IndBox(0, 1))

β = SparseIterate(p)
r = copy(y)                                             # the residual y - Xβ
a = [sum(abs2, view(X, :, j)) / n for j in 1:p]          # the curvature of f along coordinate j
for sweep in 1:1000
    change = 0.0
    for j in 1:p
        old = β[j]
        v = old + dot(view(X, :, j), r) / (n * a[j])     # the minimizer of f along j
        new = prox_entry(g, v, j, 1 / a[j])             # soft thresholding, then the box
        if new != old
            β[j] = new
            r .-= (new - old) .* view(X, :, j)
            change = max(change, abs(new - old))
        end
    end
    change < 1e-13 && break
end
βf = fista(f, g, p)
println("lasso on [0, 1]: β[1:4] = ", round.(Array(β)[1:4], digits = 4), ", ", count(!iszero, β), " nonzero, ",
        "FISTA differs by ", norm(Array(β) - βf))
@assert all(0 .<= β .<= 1) && β[1] == β[2] == 1          # the first two sit on the bound
@assert norm(Array(β) - βf) < 1e-8
@assert norm(gradient_mapping(f, g, Array(β), 1 / lipschitz(f))) < 1e-9

# --- a soft selection of the k largest scores --------------------------------
# The projection of t s onto the capped simplex weights the items between 0
# and 1, with total weight k. For small t it spreads the weight; as t grows it
# becomes the indicator of the k largest scores.
k = 3
s = randn(rng, 10)
C = Constrained(IndBox(0, 1), IndHyperplane(1, k))
topk = sortperm(s; rev = true)[1:k]
for t in (0.5, 2.0, 1000.0)
    z = prox(C, t .* s)
    println("t = ", t, ": weights ", round.(z, digits = 3))
    @assert all(0 .<= z .<= 1) && sum(z) ≈ k
end
z = prox(C, 1000 .* s)
@assert z[topk] == ones(k) && count(!iszero, z) == k
