# Constrained least squares by projected gradient descent.
#
#     minimize  ‖y - Xβ‖² / 2   subject to  β ∈ C
#
# The proximal operator of the indicator function of a set is the projection
# onto it, so projected gradient descent is proximal gradient descent with an
# indicator. Four sets: the nonnegative orthant (nonnegative least squares), the
# simplex (weights that sum to one), an ℓ1 ball (the constrained lasso), and the
# vectors with at most s nonzeros, a nonconvex set, for which projected gradient
# descent is iterative hard thresholding.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(7)
n, p = 100, 30
X = randn(rng, n, p)
L = opnorm(X)^2

function projected_fista(f, g, L, p; iters = 3000)
    β, βold, z, grad = zeros(p), zeros(p), zeros(p), zeros(p)
    t = 1.0
    for _ in 1:iters
        gradient!(grad, f, z)
        copyto!(βold, β)
        β .= z .- grad ./ L
        prox!(β, g, β, 1 / L)                         # the projection onto C
        tnew = (1 + sqrt(1 + 4t^2)) / 2
        z .= β .+ ((t - 1) / tnew) .* (β .- βold)
        t = tnew
    end
    β
end

# β is a fixed point of the projected gradient step at the solution
residual(f, g, β, L) = norm(β - prox(g, β - gradient(f, β) / L, 1 / L))

# --- nonnegative least squares ----------------------------------------------
β_true = [2.0; 0.0; 1.0; zeros(p - 3)]
f = LeastSquaresLoss(X * β_true + 0.3 * randn(rng, n), X)
β = projected_fista(f, IndNonnegative(), L, p)
grad = gradient(f, β)
println("nonnegative: ", count(>(0), β), " positive coefficients of ", p, " (no sparsity penalty), ",
        "β[1:3] = ", round.(β[1:3], digits = 3), ", min β = ", minimum(β))
@assert all(>=(0), β) && residual(f, IndNonnegative(), β, L) < 1e-8
@assert all(grad .>= -1e-6) && all(abs(grad[j]) < 1e-6 for j in 1:p if β[j] > 0)   # complementary slackness

# --- the simplex: weights that sum to one -----------------------------------
w_true = [0.5; 0.3; 0.2; zeros(p - 3)]
f = LeastSquaresLoss(X * w_true + 0.1 * randn(rng, n), X)
C = IndSimplex(1.0)
w = projected_fista(f, C, L, p)
println("simplex: weights ", round.(w[1:4], digits = 3), ", sum ", sum(w))
@assert all(>=(0), w) && sum(w) ≈ 1 && residual(f, C, w, L) < 1e-8
grad = gradient(f, w)                                   # the multiplier ν: ∇f(w)ⱼ = ν where wⱼ > 0,
ν = grad[argmax(w)]                                     # and ∇f(w)ⱼ ≥ ν where wⱼ = 0
@assert all(abs(grad[j] - ν) < 1e-6 for j in 1:p if w[j] > 0) && all(grad .>= ν - 1e-6)

# --- an ℓ1 ball: the constrained form of the lasso --------------------------
β_true = [2.0; -1.5; 1.0; zeros(p - 3)]
f = LeastSquaresLoss(X * β_true + 0.3 * randn(rng, n), X)
r = 3.5                                                 # below ‖β_true‖₁ = 4.5
β = projected_fista(f, IndBallL1(r), L, p)
println("ℓ1 ball: ‖β‖₁ = ", round(norm(β, 1), digits = 6), ", nonzero ", findall(!iszero, β))
@assert norm(β, 1) <= r + 1e-9 && residual(f, IndBallL1(r), β, L) < 1e-8
@assert norm(β, 1) ≈ r                                  # the constraint is active

# --- at most s nonzeros: iterative hard thresholding ------------------------
# The set is not convex, so plain projected gradient steps (no extrapolation)
# from zero, which converge to a fixed point; with these well-conditioned
# columns it is the true support.
s = 3
g = IndBallL0(s)
β = zeros(p)
grad = similar(β)
for _ in 1:500
    gradient!(grad, f, β)
    β .-= grad ./ L
    prox!(β, g, β, 1 / L)                               # keep the s largest entries
end
println("ℓ0 ball: nonzero ", findall(!iszero, β), ", β = ", round.(β[1:3], digits = 3))
@assert count(!iszero, β) == s && findall(!iszero, β) == 1:3
@assert residual(f, g, β, L) < 1e-8
