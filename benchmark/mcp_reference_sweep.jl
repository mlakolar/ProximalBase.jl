# Run in the candidate's package environment with one Julia thread.
# PB_MCP_SWEEP=large PB_MCP_SEED=952 julia --project=. --threads=1 benchmark/mcp_reference_sweep.jl
using ProximalBase, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "test", "mcp_reference.jl"))
using .MCPReferenceTest

Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)

function magnitude(rng, ::Type{T}) where {T}
    e = rand(rng, exponent(nextfloat(zero(T))):exponent(floatmax(T)))
    T(ldexp(1 + BigFloat(rand(rng, T)), e))
end

function case_parameters(rng, ::Type{T}, group) where {T}
    λ, a, γ, t = ntuple(_ -> magnitude(rng, T), 4)
    if group in (:firm_lower, :firm_upper)
        γ = rand(rng, Bool) ? prevfloat(a) : a / T(2)
        iszero(γ) && return nothing
        t = T(BigFloat(λ) * BigFloat(group === :firm_lower ? γ : a))
    elseif group === :hard_switch
        γ = rand(rng, Bool) ? max(a, γ) : a
        t = T(BigFloat(λ) * sqrt(BigFloat(a) * BigFloat(γ)))
    end
    (λ, a, γ, min(t, floatmax(T)))
end

function nearby(t::T) where {T}
    edge = [max(zero(T), nextfloat(t, k)) for k in -2:2]
    filter!(isfinite, edge)
    [edge; -edge]
end

function close_value(x::T, reference; check_sign=false) where {T}
    rounded = T(reference)
    isequal(x, rounded) && return true
    check_sign && signbit(x) != signbit(reference) && return false
    M = BigFloat(floatmax(T))
    if isinf(x) || isinf(rounded)
        return !isnan(x) && abs(abs(reference) - M) <= 4BigFloat(eps(floatmax(T)))
    end
    abs(BigFloat(x) - reference) <= 4BigFloat(eps(abs(rounded)))
end

function close_point(x, want, v::T, exact_branch) where {T}
    if exact_branch || iszero(want) || want == BigFloat(v)
        return x === T(want)
    end
    close_value(x, want; check_sign=true)
end

function compare_case(λ, a, γ, t::T) where {T}
    x = nearby(t)
    f = MCP(λ, a)
    y = prox(f, x, γ)
    for i in eachindex(x)
        v, got = x[i], y[i]
        want = mcp_reference(λ, a, v, γ)
        value = value_entry(f, v, i)
        want_value = mcp_reference_value(λ, a, v)
        got === prox_block(f, v, i, γ) && close_point(got, want, v, γ >= a) ||
            error("prox T=$T λ=$λ a=$a γ=$γ v=$v got=$got reference=$(T(want))")
        close_value(value, want_value) ||
            error("value T=$T λ=$λ a=$a v=$v got=$value reference=$(T(want_value))")
    end
    observed_value = value_and_prox!(x, f, x, γ)
    isequal(x, y) && isequal(observed_value, f(y)) ||
        error("fused T=$T λ=$λ a=$a γ=$γ t=$t")
    length(x)
end

function main()
    schedule = get(ENV, "PB_MCP_SWEEP", "large")
    n = schedule == "smoke" ? 64 : schedule == "large" ? 16384 : parse(Int, schedule)
    n > 0 || error("a positive sweep count is required")
    seed = parse(Int, get(ENV, "PB_MCP_SEED", "952"))
    rng = MersenneTwister(seed)
    count = 0
    setprecision(BigFloat, 256) do
        for T in (Float32, Float64), group in (:all_ranges, :firm_lower, :firm_upper, :hard_switch)
            completed = 0
            for _ in 1:n
                parameters = case_parameters(rng, T, group)
                parameters === nothing && continue
                completed += compare_case(parameters...)
            end
            println("MCP reference Julia=$VERSION type=$T group=$group seed=$seed points=$completed")
            count += completed
        end
    end
    summary = "MCP_REFERENCE_SWEEP schedule=$schedule seed=$seed count=$count result=PASS julia=$VERSION"
    println(summary)
    output = get(ENV, "PB_MCP_OUTPUT", "")
    isempty(output) || write(output, summary * "\n")
end

main()
