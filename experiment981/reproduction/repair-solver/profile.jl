include("solver.jl")

# The route checks use a disposable copy. Every timed and allocation-measured
# call below runs on a separately loaded, unmodified checkout.
function instrument(candidate, target)
    cp(joinpath(candidate, "src"), joinpath(target, "src"))
    path = joinpath(target, "src", "functions", "normlinf.jl")
    source = read(path, String)
    sort_sig = "function _linf_sorted_distance!(y::Vector{W}, c::W, ball) where {W<:Union{Float32, Float64}}"
    exact_sig = "function _linf_exact_sorted_prefix(y, c)"
    for sig in (sort_sig, exact_sig)
        @assert length(findall(sig, source)) == 1 sig
    end
    source = replace(source, sort_sig => sort_sig * "\n  _bench_sorts[] += 1",
                    exact_sig => exact_sig * "\n  _bench_exact[] += 1")
    write(path, "const _bench_sorts = Ref(0)\nconst _bench_exact = Ref(0)\n" * source)
    load_checkout(target, :RouteBench)
end

@noinline function replay(score, f, states, repeats, ::Type{T}) where {T}
    result = zero(T)
    for _ in 1:repeats, (v, x) in states
        result += score(f, v, x)
    end
    result
end

@noinline score_bytes(score, f, state) = @allocated score(f, state...)
@noinline replay_bytes(score, f, states, T) = @allocated replay(score, f, states, 1, T)

function route_pair(io, prefix, pr, c_repaired, f_pr, f_c, states, T, label, reps)
    repeats = max(1, cld(64, length(states)))
    funcs, penalties = (pr.kkt_violation, c_repaired.kkt_violation), (f_pr, f_c)
    for (score, f) in zip(funcs, penalties)
        replay(score, f, states, repeats, T)
    end
    bytes = ntuple(i -> replay_bytes(funcs[i], penalties[i], states, T), 2)
    first = isodd(length(states)) ? (1, 2) : (2, 1)
    for rep in 1:reps
        for (position, side) in enumerate(isodd(rep) ? first : reverse(first))
            start = time_ns()
            result = replay(funcs[side], penalties[side], states, repeats, T)
            elapsed = time_ns() - start
            @assert isfinite(result)
            tsv(io, prefix..., label, length(states), repeats, rep, position, ("PR", "C")[side], elapsed, bytes[side])
        end
    end
end

function projection_input(v::Vector{T}, x) where {T}
    m = maximum(abs, x)
    if iszero(m)
        return abs.(v), true
    end
    y = T[x[i] > 0 ? -v[i] : v[i] for i in eachindex(x) if abs(x[i]) == m]
    return y, false
end

@noinline function projection_batch(project!, c, inputs)
    total = zero(c)
    for (y, ball) in inputs
        total += project!(y, c, ball)
    end
    total
end

@noinline projection_bytes(project!, c, inputs) = @allocated projection_batch(project!, c, inputs)

function time_projection(io, prefix, pb, c, states, reps)
    inputs = map(s -> projection_input(s...), states)
    project! = pb._linf_sorted_distance!
    projection_batch(project!, c, deepcopy(inputs))
    bytes = projection_bytes(project!, c, deepcopy(inputs))
    for rep in 1:reps
        # Sorting is destructive. Restoring unsorted inputs is outside the timer.
        batch = deepcopy(inputs)
        start = time_ns()
        result = projection_batch(project!, c, batch)
        elapsed = time_ns() - start
        @assert isfinite(result)
        tsv(io, prefix..., length(inputs), rep, elapsed, bytes)
    end
end

function profile_case(event_io, time_io, proj_io, pr, repaired, traced, name, ::Type{T}, stride, reps) where {T}
    loss, step, c, tol = problem(T, name)
    f_pr, f_c, f_trace = pr.NormLinf(c), repaired.NormLinf(c), traced.NormLinf(c)
    states = Tuple{Vector{T},Vector{T}}[]
    routes = Symbol[]
    capture(v, x) = (push!(states, (copy(v), copy(x))); nothing)
    trace_score = traced.kkt_violation
    sorts, exact = traced._bench_sorts, traced._bench_exact
    function tracked(f, v, x)
        a, b = sorts[], exact[]
        value = trace_score(f, v, x)
        sorted, precise = sorts[] - a, exact[] - b
        @assert 0 <= precise <= sorted <= 1
        push!(routes, precise == 1 ? :exact : sorted == 1 ? :sort : :fast)
        value
    end
    tracer = workspace(loss)
    result = solve!(tracer, loss, f_trace, step, tol, stride, traced.prox!, tracked, capture)
    normal = workspace(loss)
    expected = solve!(normal, loss, f_c, step, tol, stride, repaired.prox!, repaired.kkt_violation, no_capture)
    @assert result == expected && tracer.x == normal.x && length(states) == length(routes) == result.scores

    a, b = pr.kkt_violation, repaired.kkt_violation
    # Warm both entry points and the allocation wrapper before emitting counts.
    replay(a, f_pr, states, 1, T)
    replay(b, f_c, states, 1, T)
    score_bytes(a, f_pr, first(states))
    score_bytes(b, f_c, first(states))
    prefix = (VERSION, name, T, stride)
    for (i, (state, route)) in enumerate(zip(states, routes))
        pr_bytes = score_bytes(a, f_pr, state)
        repaired_bytes = score_bytes(b, f_c, state)
        tsv(event_io, prefix..., i, route, all(iszero, state[2]), pr_bytes, repaired_bytes)
    end

    partitions = ("all" => trues(length(states)),
                  "sorted" => (routes .!= :fast), "fast" => (routes .== :fast),
                  "sort" => (routes .== :sort), "exact" => (routes .== :exact))
    for (label, members) in partitions
        selected = states[members]
        isempty(selected) && continue
        route_pair(time_io, prefix, pr, repaired, f_pr, f_c, selected, T, label, reps)
    end
    selected = states[routes .!= :fast]
    isempty(selected) || time_projection(proj_io, prefix, repaired, c, selected, reps)
end

function main(pr, repaired, traced, folder, reps)
    open(joinpath(folder, "route-events.tsv"), "w") do event_io
        open(joinpath(folder, "route-replays.tsv"), "w") do time_io
            open(joinpath(folder, "sorted-projection.tsv"), "w") do proj_io
                cell = ("julia", "problem", "type", "stride")
                tsv(event_io, cell..., "state", "route", "zero_point", "pr_bytes", "c_bytes")
                tsv(time_io, cell..., "route", "calls", "sweeps", "rep", "position", "label", "ns", "bytes_per_sweep")
                tsv(proj_io, cell..., "calls", "rep", "ns", "bytes")
                for (name, T, stride) in Iterators.product(("dense", "diagonal"), (Float32, Float64), (1, 10))
                    profile_case(event_io, time_io, proj_io, pr, repaired, traced, name, T, stride, reps)
                    foreach(flush, (event_io, time_io, proj_io))
                    println(stderr, "profile ", VERSION, " ", name, " ", T, " stride=", stride)
                end
            end
        end
    end
end

pr, candidate, folder, reps = ARGS
mktempdir(folder) do copy_dir
    Base.invokelatest(main, load_checkout(pr, :PRReplay), load_checkout(candidate, :CandidateReplay),
                      instrument(candidate, copy_dir), folder, parse(Int, reps))
end
