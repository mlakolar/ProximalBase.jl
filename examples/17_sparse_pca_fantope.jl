# Sparse principal component analysis by Fantope projection and selection.
#
#     maximize  ⟨S, X⟩ - λ ‖X‖₁   subject to  X ∈ Fᵏ = {X : 0 ⪯ X ⪯ I, tr(X) = k}
#
# The Fantope Fᵏ is the convex hull of the projections onto k-dimensional
# subspaces, so without the penalty the maximizer is the projection onto the
# leading k eigenvectors of the sample covariance S. The ℓ1 penalty makes the
# estimate of the subspace sparse (Vu, Cho, Lei and Rohe, 2013).
#
# The indicator of Fᵏ is Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k))):
# its projection projects the eigenvalues onto the capped simplex. ADMM splits
# the problem as h(X) + g(Y) with X = Y, for h = Tilt(indicator of Fᵏ, -S), whose
# proximal operator is the projection of X + S/ρ, and g = λ‖⋅‖₁.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(17)

# --- data: a spiked covariance with a sparse leading eigenvector ---------------
p, n, k = 50, 200, 1
v = zeros(p)
v[1:5] .= [0.6, 0.5, 0.4, 0.35, 0.3]
v ./= norm(v)
Σ = I + 2.0 * v * v'                             # the leading eigenvector is v
X = randn(rng, n, p) * cholesky(Σ).U              # rows from N(0, Σ)
S = Symmetric(X' * X / n)

# --- plain PCA ------------------------------------------------------------------
u = eigen(S).vectors[:, end]
println("PCA: ", count(x -> abs(x) > 1e-3, u), " nonzero loadings, |⟨u, v⟩| = ", round(abs(dot(u, v)), digits = 3))

# --- Fantope projection and selection by ADMM -----------------------------------
λ = 0.15
fantope = Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))
h = Tilt(fantope, -Matrix(S))                    # -⟨S, X⟩ on the Fantope
g = NormL1(λ)
ρ = 1.0
Xk, Y, U = zeros(p, p), zeros(p, p), zeros(p, p)
iters = 0
for it in 1:2000
    global iters = it
    prox!(Xk, h, Y .- U, 1 / ρ)                  # the projection of Y - U + S/ρ onto Fᵏ
    Yold = copy(Y)
    prox!(Y, g, Xk .+ U, 1 / ρ)                  # soft thresholding
    U .+= Xk .- Y
    r, s = norm(Xk - Y), ρ * norm(Y - Yold)      # the primal and dual residuals
    max(r, s) < 1e-9 && break
end
println("ADMM: ", iters, " iterations, ‖X - Y‖ = ", norm(Xk - Y))

μ = eigvals(Symmetric(Xk))
@assert all(-1e-9 .<= μ .<= 1 + 1e-9) && tr(Xk) ≈ k        # X is in the Fantope
@assert norm(Xk - Y) < 1e-8

# the leading eigenvector of the estimate: sparse, and nearer to v than PCA's
w = eigen(Symmetric(Y)).vectors[:, end]
support = findall(x -> abs(x) > 1e-6, w)
println("Fantope projection and selection: loadings on ", support, ", |⟨w, v⟩| = ", round(abs(dot(w, v)), digits = 3))
@assert support == 1:5
@assert abs(dot(w, v)) > abs(dot(u, v))

# the optimality conditions, through the scaled dual variable U: ρU is a
# subgradient of λ‖⋅‖₁ at Y, and S - ρU is normal to Fᵏ at X, so that the
# projection of X + S - ρU onto Fᵏ is X
@assert all(abs(ρ * U[i] - λ * sign(Y[i])) < 1e-6 for i in eachindex(Y) if !iszero(Y[i]))
@assert all(abs(ρ * U[i]) <= λ + 1e-6 for i in eachindex(Y) if iszero(Y[i]))
@assert norm(prox(fantope, Xk .+ (Matrix(S) .- ρ .* U)) - Xk) < 1e-6
