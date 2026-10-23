# A working-set solver for the lasso with many more predictors than observations.
#
#     minimize  ‖y - Xβ‖² / 2 + λ ‖β‖₁,   X of size 200 × 5000
#
# Most coefficients are zero at the solution, so the solver works on a small set
# S of coordinates: it solves the problem restricted to S, then checks the
# optimality of every coordinate outside S and adds the worst violators.
# `restrict(f, S)` and `restrict(g, S)` are the loss and the penalty of β[S] (for
# the loss, the columns S of X), and `kkt_violation(g, ∇f(β), j)` measures how
# much a coordinate at zero violates optimality: positive when moving it away
# from zero would decrease the objective.

using ProximalBase, LinearAlgebra, Random

rng = Xoshiro(14)
n, p = 200, 5000
X = randn(rng, n, p)
truth = sort(randperm(rng, p)[1:10])
β_true = zeros(p)
β_true[truth] = 2 .* sign.(randn(rng, 10))
y = X * β_true + 0.5 * randn(rng, n)

f = LeastSquaresLoss(y, X)
λ = 0.2 * norm(X' * y, Inf)
g = NormL1(λ)

kkt(f, λ, β) = (grad = gradient(f, β);
    maximum(i -> iszero(β[i]) ? max(abs(grad[i]) - λ, 0.0) : abs(grad[i] + λ * sign(β[i])), eachindex(β)))

# FISTA from β0, stopping when the optimality conditions hold to tol
function fista(f, g, λ, β0; tol = 1e-9, maxiter = 100_000)
    L = opnorm(f.X)^2
    β, βold, z, grad = copy(β0), copy(β0), copy(β0), similar(β0)
    t = 1.0
    for iter in 1:maxiter
        gradient!(grad, f, z)
        copyto!(βold, β)
        β .= z .- grad ./ L
        prox!(β, g, β, 1 / L)
        tnew = (1 + sqrt(1 + 4t^2)) / 2
        z .= β .+ ((t - 1) / tnew) .* (β .- βold)
        t = tnew
        iter % 25 == 0 && kkt(f, λ, β) < tol && return β, iter
    end
    error("no convergence")
end

function working_set(f, g, λ, p; add = 10)
    β, S = zeros(p), Int[]
    sizes = Int[]
    while true
        grad = gradient(f, β)
        viol = [(kkt_violation(g, grad, j), j) for j in 1:p if !(j in S)]
        filter!(v -> v[1] > 1e-9, viol)
        isempty(viol) && return β, sizes
        append!(S, last.(sort!(viol, rev = true)[1:min(add, end)]))   # the worst violators
        βS, _ = fista(restrict(f, S), restrict(g, S), λ, β[S])      # the small problem
        β[S] = βS
        push!(sizes, length(S))
    end
end

working_set(f, g, λ, p)                            # compile both before timing them
fista(f, g, λ, zeros(p))
t_ws = @elapsed β_ws, sizes = working_set(f, g, λ, p)
t_full = @elapsed β_full, iters = fista(f, g, λ, zeros(p))

println("working sets of sizes ", sizes, "; ", count(!iszero, β_ws), " nonzero coefficients")
println("true support found: ", truth ⊆ findall(!iszero, β_ws))
println("time: working set ", round(t_ws, digits = 3), " s, FISTA on all ", p, " coordinates ",
        round(t_full, digits = 3), " s (", iters, " iterations)")

@assert kkt(f, λ, β_ws) < 1e-8
@assert abs(f(β_ws) + g(β_ws) - f(β_full) - g(β_full)) < 1e-8 * (f(β_full) + g(β_full))
@assert last(sizes) < 100 && truth ⊆ findall(!iszero, β_ws)
