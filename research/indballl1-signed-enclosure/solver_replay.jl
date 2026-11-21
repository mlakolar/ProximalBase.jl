using ProximalBase, ProximalSolvers, LinearAlgebra, Random, SHA
const PB = ProximalBase
const PS = ProximalSolvers
include("enclosures.jl")
include("ps508_fixtures.jl")
include("timing_support.jl")
BLAS.set_num_threads(1)

struct Coefficients{T}
    kind::Symbol
    active::Bool
    indicator::T
    x::Vector{T}
end

mutable struct Probe{Mode,T}
    ball::PB.IndBallL1{T}
    cap::Float64
    score::Float64
    in_stop::Bool
    trace::Vector{Coefficients{T}}
end
Probe(::Val{Mode}, r::T, cap) where {Mode,T} =
    Probe{Mode,T}(PB.IndBallL1(r), cap, 0.0, false, Coefficients{T}[])

(::Probe{M,T})(x) where {M,T} = zero(T)

@inline score_bounds(b::Nothing, cap) = -1.0
@inline score_bounds(b::Margin, cap) = Float64(b[2] <= cap) + (b[2] - b[1]) / 2

function PB.value_and_prox!(y, g::Probe{Mode,T}, x, step) where {Mode,T}
    v = PB.value_and_prox!(y, g.ball, x, step)
    g.score += iszero(v)
    if Mode === :trace
        kind = g.in_stop ? :stationarity_projection : :trial
        push!(g.trace, Coefficients(kind, !isequal(y, x), v, copy(y)))
    elseif Mode === :legacy
        lo, hi, _ = screened_enclosure(y, g.ball.r, g.cap)
        g.score += score_bounds((lo, hi), g.cap)
    elseif Mode === :cap || Mode === :cap_stop
        g.score += score_bounds(screen_then_signed(y, g.ball.r, g.cap), g.cap)
    elseif Mode === :sign
        g.score += score_bounds(screen_then_signed(y, g.ball.r), 0.0)
    elseif Mode === :accurate
        g.score += score_bounds(compensated(y, g.ball.r), g.cap)
    end
    # Benchmark a fixed smooth-monitor workload. Checks feed the result sink.
    zero(T)
end

function PS._stationary!(state, f, g::Probe{Mode}, opts::PS.ProximalOptions) where {Mode}
    if Mode === :trace
        push!(g.trace, Coefficients(:stop_point, false, g.ball(state.x), copy(state.x)))
        g.in_stop = true
    end
    result = invoke(PS._stationary!, Tuple{Any,Any,Any,PS.ProximalOptions}, state, f, g, opts)
    if Mode === :trace
        g.in_stop = false
    end
    result
end

struct SolveRun{F,X,G,M,O}
    f::F
    start::X
    x::X
    g::G
    method::M
    opts::O
end
SolveRun(f, start, g, method, opts) = SolveRun(f, start, similar(start), g, method, opts)

function once(run::SolveRun)
    copyto!(run.x, run.start)
    g = run.g
    g.score = 0.0
    trial_only(run.f, run.x, g, run.method, run.opts)
end

@inline modelmode(::Probe{Mode}) where {Mode} = Mode

function (run::SolveRun)(count)
    total = 0.0
    g = run.g
    for _ in 1:count
        result = once(run)
        if modelmode(g) === :cap_stop
            g.score += score_bounds(compensated(result.minimizer, g.ball.r), g.cap)
        end
        total += result.iterations + result.prox_calls + Float64(first(result.minimizer)) + g.score
    end
    total
end

function make_dense_active(T, n, p)
    rng = Xoshiro(1138000 + n + p)
    A = randn(rng, T, n, p) / T(16sqrt(n))
    for j in 1:p
        A[1,j] = (one(T) + T(mod(j, 17)) / 256) / sqrt(T(p))
    end
    y = zeros(T, n)
    y[1] = T(4)
    A, y, T(0.45) * sqrt(T(p))
end

function methods()
    (("PGD", PS.ProximalGradientDescent(ls=PS.LineSearches.BackTracking(α=1.0))),
     ("APG", PS.AccProxGradDescent(α=1.0)))
end

function fixed_scenarios(visit, quick)
    for T in (Float32, Float64), (name, method) in methods()
        for (n, p) in (quick ? ((16, 2),) : ((16, 2), (64, 64), (64, 2048)))
            for geometry in (:interior, :active)
                A, y, r = synthetic(T, n, p, geometry)
                visit("ps508-fixed", string(geometry), name, method, A, y, r)
            end
        end
        if !quick
            for p in (4, 16, 256, 2048)
                A, y, r = make_dense_active(T, 64, p)
                visit("dense-fixed", "dense-active", name, method, A, y, r)
            end
        end
    end
end

@inline compare_bounds(b, q) = b === nothing ? "unavailable" : b[2] <= q ? "le" : b[1] > q ? "gt" : "cross"

function verify_bounds(b, d)
    b === nothing || BigFloat(b[1]) <= d <= BigFloat(b[2]) || error("replay containment failure: $b vs $d")
end

counts(result) = (result.iterations, result.prox_calls, result.grad_calls, result.backtrack_steps)

function write_trial(io, vectors, meta, event, r, cap, tight, trial)
    d = sum(v -> abs(BigFloat(v)), trial.x) - BigFloat(r)
    A = wide_screen(trial.x, r)
    C = compensated(trial.x, r)
    residual = signed_residual(trial.x, r)
    verify_bounds(A, d)
    verify_bounds(C, d)
    verify_bounds(residual, d)
    legacy_lo, legacy_hi, legacy_refine = screened_enclosure(trial.x, r, cap)
    verify_bounds((legacy_lo, legacy_hi), d)
    sign = compare_bounds(A, 0.0)
    c20 = compare_bounds(A, cap)
    c40 = compare_bounds(A, tight)
    bsign = sign == "cross" ? residual : A
    b20 = c20 == "cross" ? residual : A
    b40 = c40 == "cross" ? residual : A
    offset = position(vectors)
    raw = reinterpret(UInt8, trial.x)
    write(vectors, raw)
    row = (meta..., event, trial.kind, trial.active, iszero(trial.indicator),
           Float64(d), cap, tight, d <= BigFloat(cap), d <= BigFloat(tight),
           sign, c20, c40, compare_bounds(bsign, 0.0), compare_bounds(b20, cap),
           compare_bounds(b40, tight), compare_bounds(C, 0.0), compare_bounds(C, cap),
           compare_bounds(C, tight), legacy_refine, legacy_hi <= cap, offset, bytes2hex(sha256(raw)))
    println(io, join(row, '\t'))
end

function trace_run(io, vectors, family, name, label, method, A, y, r, start, alpha, opts)
    f, cap, _, _ = loss_and_budgets(A, y, r, alpha, 2.0^-20, 1e-5)
    _, tight, _, _ = loss_and_budgets(A, y, r, alpha, 2.0^-40, 1e-10)
    g = Probe(Val(:trace), r, cap)
    run = SolveRun(f, start, g, method, opts)
    result = once(run)
    @assert count(t -> t.kind !== :stop_point, g.trace) == result.prox_calls
    meta = (family, name, label, eltype(start), size(A, 1), length(start), Float64(r), counts(result)...)
    for (i, row) in pairs(g.trace)
        write_trial(io, vectors, meta, i, r, cap, tight, row)
    end
    # Record the returned position, separately from step and stop proposals.
    write_trial(io, vectors, meta, length(g.trace)+1, r, cap, tight,
                Coefficients(:return, false, g.ball(result.minimizer), result.minimizer))
end

function traces(path, quick)
    open(joinpath(path, "replay.tsv"), "w") do io
        println(io, "family\tcase\tmethod\ttype\tobservations\tn\tradius\titerations\tprox_calls\tgrad_calls\tbacktracks\tevent\tkind\tactive\tpb_feasible\tmargin\tcap20\tcap40\texact_pass20\texact_pass40\tA_sign\tA_cap20\tA_cap40\tB_sign\tB_cap20\tB_cap40\tC_sign\tC_cap20\tC_cap40\tlegacy_refine20\tlegacy_pass20\tbyte_offset\tvector_sha256")
        open(joinpath(path, "replay-vectors.bin"), "w") do vectors
            setprecision(BigFloat, 2304) do
                fixed_scenarios(quick) do family, name, label, method, A, y, r
                    T = eltype(y)
                    trace_run(io, vectors, family, name, label, method, A, y, r,
                              zeros(T, size(A, 2)), one(T), fixed_solve_options())
                end
                if !quick
                    for T in (Float32, Float64)
                        visit = function (name, A, y, r, x0, alpha, orthogonal)
                            # The archived B-policy preflight requires exact feasible starts.
                            sum(v -> abs(BigFloat(v)), x0) <= BigFloat(r) || return
                            for (label, method) in methods()
                                trace_run(io, vectors, "ps508-accuracy-recipe", name, label, method, A, y, r,
                                          x0, alpha, opt_for_study(320))
                            end
                        end
                        fixtures(T, visit)
                    end
                end
            end
        end
    end
end

function solver_pairs(path, rounds, quick)
    open(joinpath(path, "solve-pairs.tsv"), "w") do io
        println(io, "family\tcase\tmethod\ttype\tobservations\tn\tpair\titerations\tprox_calls\tgrad_calls\tbacktracks\tstrict_matched\t", TIMING_COLUMNS)
        fixed_scenarios(quick) do family, name, label, method, A, y, r
            T = eltype(y)
            opts = fixed_solve_options()
            start = zeros(T, size(A, 2))
            f, cap, _, _ = loss_and_budgets(A, y, r, one(T), 2.0^-20, 1e-5)
            make(mode) = SolveRun(f, start, Probe(Val(mode), r, cap), method, opts)
            baseline, control = make(:base), make(:independent_base)
            observed = once(baseline)
            output = copy(observed.minimizer)
            strict_matched = false
            if family == "ps508-fixed"
                strict = trial_only(f, copy(start), PB.IndBallL1(r), method, opts)
                strict_matched = counts(strict) == counts(observed) && isequal(strict.minimizer, output)
                @assert strict_matched
            end
            @assert counts(once(control)) == counts(observed) && isequal(control.x, output)
            for mode in (:legacy, :cap, :sign, :accurate, :cap_stop)
                candidate = make(mode)
                check = once(candidate)
                @assert counts(check) == counts(observed) && isequal(check.minimizer, output)
                meta = (family, name, label, T, size(A, 1), length(start), mode, counts(observed)..., strict_matched)
                paired(io, meta, baseline, candidate, control, rounds; batchlimit=1_500_000)
            end
            flush(io)
        end
    end
end

@assert Threads.nthreads() == BLAS.get_num_threads() == 1
@assert !Base.get_zero_subnormals()
println("Julia=", VERSION, " BLAS=", BLAS.get_num_threads(), " CPU=", Sys.CPU_NAME,
        " PB=", pkgdir(PB), " PS=", pkgdir(PS))
path, rounds, flavor = ARGS
mkpath(path)
quick = flavor == "canary"
traces(path, quick)
solver_pairs(path, parse(Int, rounds), quick)
