# Projections onto sets of covariance matrices.
#
# - the nearest positive semidefinite matrix: `IndPSD()`;
# - the nearest correlation matrix, by Dykstra's alternating projections between
#   the positive semidefinite cone and the matrices with unit diagonal;
# - the nearest positive semidefinite matrix of rank r: `IndPSD(rank = r)`;
# - eigenvalue shrinkage: `TracePSD(μ)`, the trace on the PSD cone, whose
#   proximal operator subtracts μ from the eigenvalues and clips at zero;
# - hard thresholding of a covariance: `IndBallL0(s; diagonal = false)` keeps the
#   s largest off-diagonal entries of a symmetric matrix, and its diagonal.
#
# The spectral functions call LAPACK through workspaces that they own.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(9)
p = 10

# a correlation matrix, then an indefinite estimate of it, as pairwise estimates
# from incomplete data can be
B = randn(rng, 30, p)
R = Symmetric(B'B) |> M -> Symmetric(M ./ sqrt.(diag(M) * diag(M)'))
E = 0.25 * randn(rng, p, p)
R̃ = Symmetric(R + (E + E') / 2 - Diagonal(diag(E)))           # unit diagonal kept
@assert eigmin(R̃) < 0

# --- the nearest positive semidefinite matrix -------------------------------
P = prox(IndPSD(), R̃, 1.0)
Λ, V = eigen(R̃)
@assert P ≈ V * Diagonal(max.(Λ, 0)) * V' && eigmin(Symmetric(P)) > -1e-10
println("nearest PSD matrix: smallest eigenvalue ", round(eigmin(R̃), digits = 4), " → ",
        round(eigmin(Symmetric(P)), sigdigits = 2), ", distance ", round(norm(P - R̃), digits = 4))

# --- the nearest correlation matrix (Higham, 2002) --------------------------
# Dykstra's algorithm corrects the projection onto the cone with the increment
# ΔS of the previous round, which plain alternating projections would omit.
function nearest_correlation(A; tol = 1e-10, maxiter = 10_000)
    psd = IndPSD()
    Y, X, ΔS, Rk = Matrix(A), zeros(size(A)), zeros(size(A)), zeros(size(A))
    for iter in 1:maxiter
        Rk .= Y .- ΔS
        prox!(X, psd, Rk, 1.0)                                    # onto the PSD cone
        ΔS .= X .- Rk
        Y .= X
        Y[diagind(Y)] .= 1                                        # onto unit diagonals
        norm(Y - X) < tol && return Y, iter
    end
    error("no convergence")
end
C, iters = nearest_correlation(R̃)
println("nearest correlation matrix: ", iters, " rounds, smallest eigenvalue ",
        round(eigmin(Symmetric(C)), sigdigits = 2), ", distance ", round(norm(C - R̃), digits = 4))
@assert all(diag(C) .== 1) && eigmin(Symmetric(C)) > -1e-8
@assert norm(C - R̃) >= norm(P - R̃) - 1e-10                    # a smaller set than the cone

# --- the nearest PSD matrix of rank 2 ---------------------------------------
P2 = prox(IndPSD(rank = 2), R̃, 1.0)
λ2 = eigvals(Symmetric(P2))
println("rank 2: eigenvalues ", round.(λ2[end-2:end], digits = 4), " (the two largest of R̃: ",
        round.(Λ[end-1:end], digits = 4), ")")
@assert all(abs.(λ2[1:end-2]) .< 1e-10) && λ2[end-1:end] ≈ Λ[end-1:end]

# --- eigenvalue shrinkage ---------------------------------------------------
μ = 0.5
T = prox(TracePSD(μ), R̃, 1.0)
@assert eigvals(Symmetric(T)) ≈ sort(max.(Λ .- μ, 0))
println("TracePSD(", μ, "): rank ", count(>(1e-10), eigvals(Symmetric(T))), " of ", p)

# --- hard thresholding a sample covariance ----------------------------------
# Keep the diagonal and the s largest off-diagonal pairs (i, j), j < i, in a
# SymmetricSparseIterate, which stores only the lower triangle.
s = 6
Ŝ = Symmetric(B'B / 30)
Θ = SymmetricSparseIterate(p)
prox!(Θ, IndBallL0(s; diagonal = false), Symmetric(Matrix(Ŝ), :L), 1.0)
offdiag = sort([abs(Ŝ[i, j]) for j in 1:p for i in j+1:p], rev = true)
kept = [(i, j) for j in 1:p for i in j+1:p if !iszero(Θ[i, j])]
println("hard thresholding: ", nnz(Θ), " stored entries (", p, " diagonal, ", length(kept), " off-diagonal)")
@assert length(kept) == s && nnz(Θ) == p + s && all(Θ[i, i] == Ŝ[i, i] for i in 1:p)
@assert minimum(abs(Ŝ[i, j]) for (i, j) in kept) >= offdiag[s] && Θ == Θ'
