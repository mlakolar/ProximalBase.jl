using LinearAlgebra, Random, Statistics
include("common.jl")
BLAS.set_num_threads(1)

struct DenseLoss{T}
    A::Matrix{T}
    target::Vector{T}
    ridge::T
end

struct DiagonalLoss{T}
    weights::Vector{T}
    target::Vector{T}
end

struct Workspace{T}
    x::Vector{T}
    g::Vector{T}
    next::Vector{T}
    residual::Vector{T}
end

function gradient!(v, f::DenseLoss, x, residual)
    mul!(residual, f.A, x)
    @. residual -= f.target
    mul!(v, transpose(f.A), residual)
    @. v += f.ridge * x
    return v
end

function gradient!(v, f::DiagonalLoss, x, residual)
    @. v = f.weights * (x - f.target)
    return v
end

workspace(f::DenseLoss{T}) where {T} =
    Workspace(zeros(T, size(f.A, 2)), zeros(T, size(f.A, 2)), zeros(T, size(f.A, 2)), zeros(T, size(f.A, 1)))
workspace(f::DiagonalLoss{T}) where {T} =
    Workspace(zeros(T, length(f.weights)), zeros(T, length(f.weights)), zeros(T, length(f.weights)), T[])

# The data, step size, and tolerance are identical for both checkouts in a cell.
function problem(::Type{T}, name) where {T}
    rng = MersenneTwister(981)
    if name == "dense"
        A = T.(randn(rng, 1024, 512) ./ sqrt(1024))
        target = A * T.(randn(rng, 512)) + T(0.1) .* T.(randn(rng, 1024))
        loss = DenseLoss(A, target, T(0.2))
        step = T(0.8) / (opnorm(A)^2 + loss.ridge)
        fraction = T(0.15)
    else
        n = 4096
        weights = [one(T) + T(mod(i, 13)) / 4 for i in 1:n]
        loss = DiagonalLoss(weights, T.(randn(rng, n)))
        step = T(0.9) / maximum(weights)
        fraction = T(0.2)
    end
    w = workspace(loss)
    gradient!(w.g, loss, w.x, w.residual)
    c = fraction * sum(abs, w.g)
    tolerance = (T === Float32 ? T(1e-5) : T(1e-9)) * norm(w.g)
    return loss, step, c, tolerance
end

no_capture(v, x) = nothing

@noinline function solve!(w, loss, f, step, tolerance, stride, prox!, score, capture)
    fill!(w.x, zero(eltype(w.x)))
    evaluations = 0
    for iter in 0:800
        gradient!(w.g, loss, w.x, w.residual)
        if iter % stride == 0 || iter == 800
            evaluations += 1
            capture(w.g, w.x)
            r = score(f, w.g, w.x)
            if r <= tolerance || iter == 800
                return (steps=iter, scores=evaluations, residual=r)
            end
        end
        @. w.next = w.x - step * w.g
        prox!(w.x, f, w.next, step)
    end
    error("unreachable")
end

# Independent high-precision mass bisection used only to check terminal iterates.
function reference_score(v, x, c)
    setprecision(BigFloat, 192) do
        m = maximum(abs, x)
        matched = BigFloat[]
        off2 = BigFloat(0)
        for i in eachindex(x, v)
            if iszero(m)
                push!(matched, abs(BigFloat(v[i])))
            elseif abs(x[i]) == m
                push!(matched, signbit(x[i]) ? BigFloat(v[i]) : -BigFloat(v[i]))
            else
                off2 += BigFloat(v[i])^2
            end
        end
        r = BigFloat(c)
        iszero(m) && sum(matched) <= r && return BigFloat(0)
        lo = minimum(matched) - r
        hi = maximum(matched)
        for _ in 1:156
            mid = (lo + hi) / 2
            mass = sum(u -> max(u - mid, 0), matched)
            if mass > r
                lo = mid
            else
                hi = mid
            end
        end
        threshold = (lo + hi) / 2
        sqrt(off2 + sum(u -> min(u, threshold)^2, matched))
    end
end

function warm_endpoint(args, c, tolerance)
    result = solve!(args...)
    w = first(args)
    reference = reference_score(w.g, w.x, c)
    @assert result.steps < 800 "solve hit iteration cap"
    @assert reference <= BigFloat(tolerance) * big"1.01" (reference, tolerance, result)
    return result, reference, copy(w.x)
end
