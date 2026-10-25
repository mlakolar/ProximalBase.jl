module GPUSolverBenchmark

using LinearAlgebra, Random, ProximalBase

"""Generate once in Float64; both backends receive the same rounded inputs."""
function problem(n, p, seed, ::Type{T}) where {T}
    rng = MersenneTwister(seed)
    X = randn(rng, n, p) / sqrt(n)
    truth = zeros(p)
    s = max(1, ceil(Int, 0.05p))
    truth[1:s] .= ifelse.(rand(rng, Bool, s), 1.0, -1.0)
    y = X * truth + 0.1randn(rng, n)
    X, y = T.(X), T.(y)
    X64, y64 = Float64.(X), Float64.(y)
    correlation = norm(X64' * y64, Inf)
    lambda = T(0.1correlation)
    gamma = T(inv(1.01opnorm(X64)^2))
    return (; X, y, lambda, gamma, scale = max(1.0, correlation, Float64(lambda)))
end

function workspace(data, convert_array = identity)
    loss = LeastSquaresLoss(convert_array(data.y), convert_array(data.X))
    beta = convert_array(zeros(eltype(data.X), size(data.X, 2)))
    return (; loss, penalty = NormL1(data.lambda), beta,
            gradient = similar(beta), temporary = similar(beta), candidate = similar(beta))
end

function mapping!(w, gamma)
    gradient!(w.gradient, w.loss, w.beta)
    @. w.temporary = w.beta - gamma * w.gradient
    prox!(w.candidate, w.penalty, w.temporary, gamma)
    return nothing
end

"""Check at the returned iterate, including iteration zero and the cap.

`iterations` counts accepted updates. Mapping evaluations for stopping are included
in solve time. A check does not accept the candidate when the current iterate passes.
"""
function solve!(w, gamma, scale; tolerance = 1e-4, check_every = 1, maxiter = 20_000)
    check_every > 0 || throw(ArgumentError("check_every must be positive"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    checks = 0
    residual = Inf
    for iteration in 0:maxiter
        mapping!(w, gamma)
        if iteration % check_every == 0 || iteration == maxiter
            checks += 1
            @. w.temporary = (w.beta - w.candidate) / gamma
            residual = norm(w.temporary, Inf) / scale
            isfinite(residual) || return (; iterations = iteration, checks, residual, status = "nonfinite")
            residual <= tolerance && return (; iterations = iteration, checks, residual, status = "converged")
        end
        iteration == maxiter && break
        copyto!(w.beta, w.candidate)
    end
    return (; iterations = maxiter, checks, residual, status = "iteration_cap")
end

"""Independent Float64 residual and objective on the rounded input data."""
function verify(data, beta)
    X, y, b = Float64.(data.X), Float64.(data.y), Float64.(beta)
    lambda, gamma = Float64(data.lambda), Float64(data.gamma)
    e = X * b - y
    z = b - gamma * (X' * e)
    candidate = sign.(z) .* max.(abs.(z) .- gamma * lambda, 0.0)
    residual = norm((b - candidate) / gamma, Inf) / data.scale
    objective = dot(e, e) / 2 + lambda * sum(abs, b)
    return (; residual, objective)
end

function reference(data; maxiter = 100_000)
    rounded = merge(data, (; X = Float64.(data.X), y = Float64.(data.y),
                           lambda = Float64(data.lambda), gamma = Float64(data.gamma)))
    w = workspace(rounded)
    result = solve!(w, rounded.gamma, rounded.scale; tolerance = 1e-10, maxiter)
    result.status == "converged" || error("Tighter CPU reference failed: $result")
    return verify(data, w.beta)
end

# A solver stopping claim is accepted only after the independent check and a
# comparison to the tighter reference. Preserve the objective gap, including its sign.
function assessment(result, independent, ref, tolerance, ::Type{T}) where {T}
    gap = independent.objective - ref.objective
    objective_limit = 10tolerance * max(1.0, abs(ref.objective))
    status = result.status
    if status == "converged"
        status = !isfinite(independent.residual) || independent.residual > tolerance ?
                 "residual_failure" : !isfinite(gap) || abs(gap) > objective_limit ?
                 "objective_failure" : "success"
    end
    return (; status, gap, objective_limit)
end

end
