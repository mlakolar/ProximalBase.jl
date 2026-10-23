# A differential network: estimate Δ = Ωx - Ωy, the difference of two precision
# matrices, directly from the two sample covariances, without estimating Ωx or
# Ωy. The D-trace loss (Zhao, Cai and Li, 2014)
#
#     tr(Σx Δ Σy Δ) / 2 - tr(Δ (Σy - Σx))
#
# is minimized by Ωx - Ωy at the population covariances. It is
# `KronQuadratic(Σx, Σy, Σx - Σy)`, the quadratic ⟨Δ, ΣxΔΣy⟩/2 + ⟨C, Δ⟩ computed
# without forming the p² × p² matrix Σy ⊗ Σx. The estimate is sparse and
# symmetric, so the iterate is a `SymmetricSparseIterate`, on which the gradient
# is the one over symmetric matrices. Proximal gradient descent with the ℓ1 norm
# keeps only the entries that thresholding leaves nonzero.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(13)
p, n = 20, 2000
Ωx = Matrix(SymTridiagonal(fill(1.0, p), fill(0.3, p - 1)))   # a chain graph
Ωy = copy(Ωx)
Ωy[5, 2] = Ωy[2, 5] = 0.3                                      # an edge appears
Ωy[11, 10] = Ωy[10, 11] = 0.0                                  # and one disappears
@assert isposdef(Ωy)
sample(Ω) = randn(rng, n, p) / cholesky(Ω).U'                  # rows from N(0, Ω⁻¹)
X, Y = sample(Ωx), sample(Ωy)
Σx, Σy = X'X / n, Y'Y / n

f = KronQuadratic(Σx, Σy, Σx - Σy)
λ = 0.15
g = NormL1(λ)
L = opnorm(Σx) * opnorm(Σy)                                    # Lipschitz constant of ∇f

Δ = SymmetricSparseIterate(p)
G = zeros(p, p)
for _ in 1:1500
    gradient!(G, f, Δ)                     # (ΣxΔΣy + ΣyΔΣx)/2 + Σx - Σy
    Δ .= Δ .- G ./ L                       # each entry of the lower triangle once
    prox!(Δ, g, Δ, 1 / L)                  # soft thresholding of the stored entries
    dropzeros!(Δ)                          # forget the entries it set to zero
end

# the coordinates of Δ are the entries of its lower triangle; the stored ones are
# the active coordinates, and tril_cartesian gives their (row, column)
edges = sort([tril_cartesian(p, k) for k in activecoordinates(Δ)])
println("stored entries: ", nnz(Δ), " of ", ncoordinates(Δ), " coordinates")
for (r, c) in edges
    println("Δ[", r, ", ", c, "] = ", round(Δ[r, c], digits = 3), "   truth ", Ωx[r, c] - Ωy[r, c])
end

# optimality over symmetric matrices: G + λ sign(Δ) = 0 where Δ ≠ 0, and |G| ≤ λ
# where Δ = 0, with G the gradient over symmetric matrices
gradient!(G, f, Δ)
@assert all(abs(G[i, j] + λ * sign(Δ[i, j])) < 1e-6 for j in 1:p, i in 1:p if !iszero(Δ[i, j]))
@assert all(abs(G[i, j]) <= λ + 1e-6 for j in 1:p, i in 1:p if iszero(Δ[i, j]))
@assert edges == [(5, 2), (11, 10)] && Δ[5, 2] < 0 < Δ[11, 10] && Δ[2, 5] == Δ[5, 2]
