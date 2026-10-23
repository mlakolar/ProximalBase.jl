# Sparse plus low-rank: split a matrix M into a low-rank part L and a sparse part
# S, as in robust principal component analysis.
#
#     minimize  ½ ‖M - S - L‖² + λ₁ ‖S‖₁ + λ₂ ‖L‖_*
#
# The variable is the pair (S, L). `SeparableSum(NormL1(λ₁), NuclearNorm(λ₂))` is
# the penalty on the pair: its proximal operator acts on a tuple of blocks, soft
# thresholding S and shrinking the singular values of L. The smooth part has
# the same gradient S + L - M in both blocks, with Lipschitz constant 2.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(11)
m, n, r = 50, 40, 2
L0 = randn(rng, m, r) * randn(rng, r, n)                       # low rank
S0 = zeros(m, n)
spikes = randperm(rng, m * n)[1:100]                           # 5% of the entries
S0[spikes] .= 8 .* sign.(randn(rng, 100))
M = L0 + S0 + 0.1 * randn(rng, m, n)

λ₁, λ₂ = 0.5, 2.5
h = SeparableSum(NormL1(λ₁), NuclearNorm(λ₂))

# FISTA on the pair (S, L)
S, L = zeros(m, n), zeros(m, n)
Sold, Lold, ZS, ZL, G = (zeros(m, n) for _ in 1:5)
t = 1.0
obj = Float64[]
for _ in 1:500
    G .= ZS .+ ZL .- M                                         # the gradient in both blocks
    copyto!(Sold, S)
    copyto!(Lold, L)
    S .= ZS .- G ./ 2
    L .= ZL .- G ./ 2
    hSL = value_and_prox!((S, L), h, (S, L), 1 / 2)           # both blocks, and h(S, L)
    push!(obj, norm(M - S - L)^2 / 2 + hSL)
    tnew = (1 + sqrt(1 + 4t^2)) / 2
    ZS .= S .+ ((t - 1) / tnew) .* (S .- Sold)
    ZL .= L .+ ((t - 1) / tnew) .* (L .- Lold)
    global t = tnew
end

σ = svdvals(L)
found, truth = Set(findall(!iszero, vec(S))), Set(spikes)          # linear indices
println("objective ", round(obj[end], digits = 4), "; L has rank ", count(>(1e-8), σ),
        ", relative error ", round(norm(L - L0) / norm(L0), digits = 4))
println("S has ", length(found), " nonzeros: ", length(found ∩ truth), " of the ", length(truth),
        " spikes, ", length(setdiff(found, truth)), " others")

# optimality: with G = S + L - M, -G is a subgradient of λ₁‖⋅‖₁ at S and of
# λ₂‖⋅‖_* at L; the first condition is entrywise
G .= S .+ L .- M
@assert all(abs(G[k] + λ₁ * sign(S[k])) < 1e-6 for k in eachindex(S) if !iszero(S[k]))
@assert all(abs(G[k]) <= λ₁ + 1e-6 for k in eachindex(S) if iszero(S[k]))
@assert opnorm(G) <= λ₂ + 1e-6                                 # ‖G‖₂ ≤ λ₂ for the nuclear norm
@assert count(>(1e-8), σ) == r && truth ⊆ found && length(found) <= 1.1 * length(truth)
@assert norm(L - L0) / norm(L0) < 0.1           # the nuclear norm shrinks σᵢ by about λ₂
