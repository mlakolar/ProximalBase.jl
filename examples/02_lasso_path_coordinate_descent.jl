# A lasso path by coordinate descent, with warm starts and an active set.
#
#     minimize  ‖y - Xβ‖² / 2 + λ ‖β‖₁   for a decreasing sequence of λ
#
# One coordinate step minimizes over βⱼ alone: with the residual r = y - Xβ and
# aⱼ = ‖Xⱼ‖², the new value is the proximal operator of the penalty, with step
# 1/aⱼ, at the point βⱼ + Xⱼ'r / aⱼ. `prox_entry` computes it without writing β,
# so a coordinate that stays at zero is never written, and a `SparseIterate`
# stores only the coordinates that have been nonzero: its stored entries are the
# active set. `kkt_violation` tells whether a coordinate at zero should join.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(2)
n, p = 100, 300
X = randn(rng, n, p)
β_true = zeros(p)
β_true[[5, 50, 120, 200, 260]] = [2.0, -1.5, 1.0, -1.0, 0.8]
y = X * β_true + 0.3 * randn(rng, n)

a = [sum(abs2, view(X, :, j)) for j in 1:p]          # coordinate curvatures ‖Xⱼ‖²

# one pass over the coordinates js; returns the largest change of a coordinate
function sweep!(β, r, X, a, g, js)
    δ = 0.0
    for j in js
        old = β[j]
        z = old + dot(view(X, :, j), r) / a[j]
        new = prox_entry(g, z, j, 1 / a[j])
        if new != old                                  # write only what changes
            β[j] = new
            axpy!(old - new, view(X, :, j), r)         # keep r = y - Xβ
            δ = max(δ, abs(new - old) * sqrt(a[j]))
        end
    end
    δ
end

# solve for one λ from the current β: sweep the active set to convergence, then
# check every coordinate; repeat while some coordinate at zero violates optimality
function solve!(β, r, X, a, g; tol = 1e-10)
    grad = similar(r, size(X, 2))
    while true
        while sweep!(β, r, X, a, g, collect(activecoordinates(β))) > tol end
        mul!(grad, X', r)
        grad .*= -1                                    # ∇f(β) = -X'r
        violators = [j for j in 1:size(X, 2) if !isactive(β, j) && kkt_violation(g, grad, j) > 1e-9]
        isempty(violators) && return β
        sweep!(β, r, X, a, g, violators)
    end
end

λmax = norm(X' * y, Inf)                               # the smallest λ with β = 0
λs = λmax .* exp.(range(0, log(0.02), length = 25))

β = SparseIterate(p)
r = copy(y)
path = Vector{Float64}[]
for λ in λs
    solve!(β, r, X, a, NormL1(λ))
    push!(path, Vector(β))
    println("λ = ", rpad(round(λ, digits = 2), 8), " nonzeros ", rpad(count(!iszero, β), 4),
            " stored ", nnz(β))
end

# the optimality conditions at every λ of the path
f = LeastSquaresLoss(y, X)
for (λ, βλ) in zip(λs, path)
    grad = gradient(f, βλ)
    @assert all(abs(grad[i] + λ * sign(βλ[i])) < 1e-6 for i in findall(!iszero, βλ))
    @assert all(abs(grad[i]) <= λ + 1e-6 for i in findall(iszero, βλ))
end
@assert iszero(path[1])                                 # λmax gives β = 0
@assert Set(findall(!iszero, path[12])) == Set([5, 50, 120, 200, 260])
@assert nnz(β) < p                                      # most coordinates were never stored
