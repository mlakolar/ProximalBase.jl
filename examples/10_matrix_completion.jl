# Matrix completion: recover a low-rank matrix from a subset Ω of its entries.
#
# Soft-impute is proximal gradient descent on
#     ½ ‖P_Ω(Z - M)‖² + λ ‖Z‖_*,
# whose gradient P_Ω(Z - M) has Lipschitz constant 1. The step fills the missing
# entries with the current guess and shrinks the singular values of the result
# by λ, which is the proximal operator of `NuclearNorm(λ)`. Decreasing λ along a
# path, with warm starts, converges quickly. Hard-impute instead projects onto
# the matrices of rank at most r, `IndBallRank(r)`: a nonconvex set, whose
# projection keeps the r largest singular values.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(10)
m, n, r = 60, 40, 3
M = randn(rng, m, r) * randn(rng, r, n)                 # the matrix, of rank 3
observed = rand(rng, m, n) .< 0.4                       # 40% of the entries
println("observed ", count(observed), " of ", m * n, " entries; a rank 3 matrix has ",
        r * (m + n - r), " degrees of freedom")

missing_error(Z) = norm((Z - M)[.!observed]) / norm(M[.!observed])

# --- soft-impute along a path of λ ------------------------------------------
Z = zeros(m, n)
V = similar(Z)
for λ in exp.(range(log(50.0), log(0.01), length = 30))
    g = NuclearNorm(λ)
    for _ in 1:100
        V .= ifelse.(observed, M, Z)                    # Z - P_Ω(Z - M)
        prox!(Z, g, V, 1.0)                             # shrink the singular values by λ
    end
end
# the singular values beyond the third decay slowly toward zero; the numerical
# rank is 3 long before they reach it
σ = svdvals(Z)
println("soft-impute: σ = ", round.(σ[1:5], sigdigits = 3), ", relative error on the missing entries ",
        round(missing_error(Z), sigdigits = 2))
@assert σ[r+1] < 1e-2 * σ[r] && missing_error(Z) < 5e-3

# --- hard-impute: projection onto rank ≤ 3 ----------------------------------
Zh = zeros(m, n)
proj = IndBallRank(r)
for _ in 1:300
    V .= ifelse.(observed, M, Zh)
    prox!(Zh, proj, V, 1.0)
end
println("hard-impute: rank ", rank(Zh), ", relative error on the missing entries ",
        round(missing_error(Zh), sigdigits = 2))
@assert rank(Zh) == r && missing_error(Zh) < 1e-6
