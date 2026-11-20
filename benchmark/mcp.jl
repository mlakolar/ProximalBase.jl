# Paired benchmark of MCP (#952) between the archived source tree (eff21b6d), the candidate and a second
# compilation of the candidate (the A/A control), Float32 and Float64, at n = 8, 1024 and 65536, in four
# ordinary regimes of the step γ against the shape a: firm (γ = a/2), near_firm (γ = 7a/8), near_hard
# (γ = 9a/8) and hard (γ = 3a/2). Per regime: a loop of the scalar prox_block (scalar step and array of
# steps), dense prox! (scalar step and array of steps), the dense value, and value_and_prox! (scalar step and
# array of steps); and, at n = 65536, SparseIterate, BlockIterate and SymmetricSparseIterate (p = 256) value,
# prox! and value_and_prox! with a scalar step.
#
#   PB_ARCHIVE=<root of the archived tree> PB_CANDIDATE=<root of the candidate> PB_OUTPUT=<dir> \
#   PB_ROUNDS=7 julia --threads=1 --check-bounds=no --project=benchmark benchmark/mcp.jl
#
# The three trees are loaded with Base.include into their own modules, as benchmark/indnonnegative.jl does: by
# default in the order archive, candidate, control, and with PB_REVERSE=1 (a fresh process) in the order
# candidate, archive, control, with the candidate also taking the place of the archive in the timing order.
# Each round times the three roles of every case in alternating order (the reverse order in even rounds), and
# keeps per role the minimum over the samples of its trial; at least 5 rounds are required for timing
# (PB_ROUNDS, default 7). The paired ratios of a round are candidate/archive (C/A) and control/candidate (A/A).
# raw.csv has every timing, ratios.csv the raw paired ratios of every round, summary.txt their median, minimum
# and maximum per case with the warmed allocations of the archive and candidate roles, and marks medians above
# 1.05. The roles share one set of plain input and output arrays (the iterates, whose types belong to each tree,
# are built per role), so that the placement of the arrays does not differ between the roles.
#
# The parameters are λ = 3/4 and a = 3 and a step γ per regime: finite values for which the archive and the
# candidate are mathematically equivalent. For γ < a the proximal point of γ·MCP is 0 for |v| <= γλ, the firm
# value sign(v)(|v| - γλ)/(1 - γ/a) for γλ < |v| <= aλ and v above aλ; for γ >= a it is 0 for
# |v| <= λ√(aγ) and v above (the zero wins the ties). The steps of the array-step cases stay on the side of a of
# their regime. Every input magnitude lies at least 2^-6 relative to every threshold of the scalar step and of
# its own array step, so that no input is near a threshold, where two implementations may legitimately decide
# differently; the inputs of the iterates are entries of the dense inputs. Random signs, magnitudes in the three
# pieces of the prox (two for the hard regimes), one seed per element type and regime shared by the roles.
#
# Before timing, every case of the three roles is run once and compared (a) with a closed form evaluated in
# Float64 (the proximal point and the value at it), (b) between the archive and the candidate and (c) between the
# control and the candidate, entry by entry within 16 eps(T) |v| / (1 - γ/a) for the firm regimes and 16 eps(T)
# |v| for the hard ones (values: 8 eps(T) sqrt(n) times that amplification, relative), with the zero pattern
# exactly equal. The tolerances admit a different evaluation order or an fma, not a different formula. The fused
# value of each role is also compared with the value at its own output. The script stops at a failure and
# reports the largest deviation of each comparison in units of its tolerance, and the number of cases that are
# bitwise equal. PB_CHECK_ONLY=1 stops after that and the allocation table. PB_FILTER selects cases by a
# regular expression on their ids ("array_step" is the one that needs array-step support in the archive).
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates, Printf
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)
archive_root = abspath(ENV["PB_ARCHIVE"]); candidate_root = abspath(ENV["PB_CANDIDATE"])
reversed = get(ENV, "PB_REVERSE", "0") == "1"
rounds = parse(Int, get(ENV, "PB_ROUNDS", "7"))
check_only = get(ENV, "PB_CHECK_ONLY", "0") == "1"
check_only || rounds >= 5 || error("PB_ROUNDS = $rounds: at least 5 alternating rounds are required")
module Archive end
module Candidate end
module CandidateControl end
for (m, root) in (reversed ? ((Candidate, candidate_root), (Archive, archive_root)) : ((Archive, archive_root), (Candidate, candidate_root)))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
end
Base.include(CandidateControl, joinpath(candidate_root, "src", "ProximalBase.jl"))
for (name, m) in (("archive", Archive), ("candidate", Candidate))
    isdefined(getfield(m, :ProximalBase), :MCP) || error("the $name tree has no MCP: ", name == "archive" ? archive_root : candidate_root)
end

const SIZES = (8, 1024, 65536)
const REGIMES = (:firm, :near_firm, :near_hard, :hard)
const LAMBDA = 0.75
const SHAPE = 3.0
# the step of each regime and the width of the array of steps around it (all exact in Float32)
const STEP = Dict(:firm => (1.5, 0.5), :near_firm => (2.625, 0.125), :near_hard => (3.375, 0.125), :hard => (4.5, 1.0))
const MARGIN = 2.0^-6
const SINK_LENGTH = 1
firmside(reg) = reg in (:firm, :near_firm)

# the closed forms, in Float64
function mcp_prox(v, γ)
    m = abs(v)
    if γ < SHAPE
        m <= γ * LAMBDA ? 0.0 : m <= SHAPE * LAMBDA ? copysign((m - γ * LAMBDA) / (1 - γ / SHAPE), v) : float(v)
    else
        m <= LAMBDA * sqrt(SHAPE * γ) ? 0.0 : float(v)
    end
end
mcp_value(v) = (m = abs(Float64(v)); m <= SHAPE * LAMBDA ? LAMBDA * m - m^2 / (2 * SHAPE) : SHAPE * LAMBDA^2 / 2)
thresholds(reg, γ) = firmside(reg) ? (γ * LAMBDA, SHAPE * LAMBDA) : (LAMBDA * sqrt(SHAPE * γ),)
# the factor by which the subtraction and the division of the firm piece amplify a rounding error of the input
amplification(reg) = firmside(reg) ? 1 / (1 - (STEP[reg][1] + STEP[reg][2] / 2) / SHAPE) : 1.0

# a magnitude in a piece of the prox, at least MARGIN (relative) from the thresholds of every step of γs
function drawmag(rng, reg, γs)
    g0, w = STEP[reg]
    lo, hi = g0 - w / 2, g0 + w / 2
    ranges = if firmside(reg)
        ((0.02 * LAMBDA, lo * LAMBDA), (hi * LAMBDA, SHAPE * LAMBDA), (SHAPE * LAMBDA, 2 * SHAPE * LAMBDA))
    else
        tlo, thi = LAMBDA * sqrt(SHAPE * lo), LAMBDA * sqrt(SHAPE * hi)
        ((0.02 * LAMBDA, tlo), (thi, 2.5 * thi))
    end
    while true
        l, h = ranges[rand(rng, 1:length(ranges))]
        m = l + (h - l) * rand(rng)
        all(γ -> all(t -> abs(m - t) > MARGIN * t, thresholds(reg, γ)), γs) && return m
    end
end

# the plain arrays of one element type and regime, shared by the roles, with the tolerances and the closed forms
function makedata(::Type{T}, reg) where {T}
    rng = MersenneTwister(952 + 10 * (T === Float32) + findfirst(==(reg), collect(REGIMES)))
    g0, w = STEP[reg]
    γ = T(g0)
    scale = 16 * eps(T) * amplification(reg)
    vscale(n) = 8 * eps(T) * max(1.0, sqrt(n)) * amplification(reg)
    tolof(xd) = scale .* abs.(Float64.(xd))
    D = Dict{Any, Any}(:sink => zeros(Float64, SINK_LENGTH))
    for n in SIZES
        steps = T.(g0 .+ w .* (rand(rng, n) .- 0.5))
        x = Vector{T}(undef, n)
        for i in 1:n
            x[i] = (rand(rng) < 0.5 ? -one(T) : one(T)) * T(drawmag(rng, reg, (g0, Float64(steps[i]))))
        end
        ys = mcp_prox.(Float64.(x), Float64(γ)); ya = mcp_prox.(Float64.(x), Float64.(steps))
        D[n] = (; x, steps, out = similar(x), tol = tolof(x), vtol = vscale(n), γ,
                ref_s = ys, ref_a = ya, vref_x = sum(mcp_value, x), vref_s = sum(mcp_value, ys), vref_a = sum(mcp_value, ya))
    end
    n = 65536; x = D[n].x
    p = 256
    xsp = zeros(T, n); stored = randperm(rng, n)[1:(n ÷ 10)]; xsp[stored] .= x[stored]
    groups = [collect(k:64:n) for k in 1:64]
    xbl = copy(x); for k in 2:2:64; xbl[groups[k]] .= zero(T); end         # half of the groups active
    sym = Tuple{Int, Int, T}[]; M = zeros(T, p, p)
    for j in 1:p, i in j:p
        rand(rng) < 0.1 || continue
        v = x[(j - 1) * p + i]; push!(sym, (i, j, v)); M[i, j] = v; M[j, i] = v
    end
    iter(xd) = (; xd, tol = tolof(xd), ref = mcp_prox.(Float64.(xd), Float64(γ)), vref_x = sum(mcp_value, xd),
                vref_y = sum(mcp_value, mcp_prox.(Float64.(xd), Float64(γ))))
    D[:iterates] = (; groups, xsp, xbl, p, sym, denseout = zeros(T, p, p), vtol = vscale(n), γ,
                    sparse = iter(xsp), block = iter(xbl), symmetric = iter(M))
    D
end

# The timed operation and the check of each kind of case. The arrays and the oracles of the role (prox_block,
# prox!, value_and_prox! of its tree) are arguments, so that each closure holds the arrays of its own size and
# calls the oracle without a lookup in the module of the role. The timed operations store their result in a
# shared array, so that they cannot be removed.
function prox_block_loop(pb::P, f, x, γ) where {P}
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += pb(f, x[i], i, γ)
    end
    s
end
function prox_block_loop_steps(pb::P, f, x, γ) where {P}
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += pb(f, x[i], i, γ[i])
    end
    s
end
loop_case(pb, sink, f, x, γ) = (() -> (sink[1] = prox_block_loop(pb, f, x, γ); nothing), () -> [pb(f, x[i], i, γ) for i in eachindex(x)])
loop_steps_case(pb, sink, f, x, γ) = (() -> (sink[1] = prox_block_loop_steps(pb, f, x, γ); nothing), () -> [pb(f, x[i], i, γ[i]) for i in eachindex(x)])
prox_case(px, out, f, x, γ, snap = copy) = (() -> (px(out, f, x, γ); nothing), () -> (px(out, f, x, γ); snap(out)))
value_case(sink, f, x) = (() -> (sink[1] = f(x); nothing), () -> f(x))
vprox_case(vp, sink, out, f, x, γ, snap = copy) =
    (() -> (sink[1] = vp(out, f, x, γ); nothing), () -> (v = vp(out, f, x, γ); y = snap(out); (v, y, f(out))))

function makeprobes(PB, ::Type{T}, reg, D) where {T}
    probes = NamedTuple[]
    add(id, n, kind, (op, check), tol, vtol, ref) =
        push!(probes, (id = "$(T)/$(reg)/$id", n = n, kind = kind, op = op, check = check, tol = tol, vtol = vtol, ref = ref))
    pb, px, vp = PB.prox_block, PB.prox!, PB.value_and_prox!
    f = PB.MCP(T(LAMBDA), T(SHAPE))
    sink = D[:sink]
    for n in SIZES
        d = D[n]; γ = d.γ
        add("prox_block_loop/scalar_step/n$n", n, :prox, loop_case(pb, sink, f, d.x, γ), d.tol, d.vtol, d.ref_s)
        add("prox_block_loop/array_step/n$n", n, :prox, loop_steps_case(pb, sink, f, d.x, d.steps), d.tol, d.vtol, d.ref_a)
        add("prox!/scalar_step/n$n", n, :prox, prox_case(px, d.out, f, d.x, γ), d.tol, d.vtol, d.ref_s)
        add("prox!/array_step/n$n", n, :prox, prox_case(px, d.out, f, d.x, d.steps), d.tol, d.vtol, d.ref_a)
        add("value/n$n", n, :value, value_case(sink, f, d.x), nothing, d.vtol, d.vref_x)
        add("value_and_prox!/scalar_step/n$n", n, :vprox, vprox_case(vp, sink, d.out, f, d.x, γ), d.tol, d.vtol, (d.vref_s, d.ref_s, d.vref_s))
        add("value_and_prox!/array_step/n$n", n, :vprox, vprox_case(vp, sink, d.out, f, d.x, d.steps), d.tol, d.vtol, (d.vref_a, d.ref_a, d.vref_a))
    end
    I = D[:iterates]; γ = I.γ
    n = length(I.xsp)
    sp = PB.SparseIterate(I.xsp); spout = copy(sp)
    bl = PB.BlockIterate(copy(I.xbl), I.groups); blout = copy(bl)
    S = PB.SymmetricSparseIterate(T, I.p)
    for (i, j, v) in I.sym
        S[i, j] = v
    end
    Sout = copy(S)
    for (name, x, out, it, snap, m) in (("sparse", sp, spout, I.sparse, Vector, n), ("block", bl, blout, I.block, Vector, n),
                                        ("symmetric", S, Sout, I.symmetric, Matrix, I.p^2))
        id = name == "symmetric" ? "p$(I.p)" : "n$n"
        add("$name/value/$id", m, :value, value_case(sink, f, x), nothing, I.vtol, it.vref_x)
        add("$name/prox!/$id", m, :prox, prox_case(px, out, f, x, γ, snap), it.tol, I.vtol, it.ref)
        add("$name/value_and_prox!/$id", m, :vprox, vprox_case(vp, sink, out, f, x, γ, snap), it.tol, I.vtol, (it.vref_y, it.ref, it.vref_y))
    end
    probes
end

const data = Dict(T => Dict(reg => makedata(T, reg) for reg in REGIMES) for T in (Float32, Float64))
const roles = ("archive" => Archive, "candidate" => Candidate, "control" => CandidateControl)
const suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), T, reg, data[T][reg]) for T in (Float32, Float64) for reg in REGIMES]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
isempty(suites["archive"]) && error("PB_FILTER selects no case: ", repr(get(ENV, "PB_FILTER", "")))
@assert getfield.(suites["archive"], :id) == getfield.(suites["candidate"], :id) == getfield.(suites["control"], :id)

# the largest deviation of two results in units of the tolerance of the case: 0 when they are identical, above 1
# when they differ by more than the tolerance, the zero pattern differs, or an entry is not finite
function deviation(a, c, p)
    if a isa Tuple
        length(a) == length(c) || return Inf
        return maximum(deviation(ai, ci, p) for (ai, ci) in zip(a, c))
    elseif a isa AbstractArray
        size(a) == size(c) || return Inf
        (all(isfinite, a) && all(isfinite, c)) || return Inf
        iszero.(a) == iszero.(c) || return Inf
        r = 0.0
        for k in eachindex(a)
            d = abs(Float64(a[k]) - Float64(c[k]))
            d == 0 || (r = max(r, d / p.tol[k]))
        end
        return r
    else
        (isfinite(a) && isfinite(c)) || return Inf
        d = abs(Float64(a) - Float64(c))
        return d == 0 ? 0.0 : d / (p.vtol * max(abs(Float64(a)), abs(Float64(c))))
    end
end

function warm_alloc(op)
    op(); op()
    a = typemax(Int)
    for _ in 1:5
        a = min(a, @allocated op())
    end
    a
end

# the roles share their output arrays, so each result is read right after its own call
dev = Dict(k => 0.0 for k in (:oracle_archive, :oracle_candidate, :archive_candidate, :control_candidate, :fused_archive, :fused_candidate))
bitwise = Dict(:archive_candidate => 0, :control_candidate => 0)
alloc = Dict((r, i) => 0 for r in first.(roles) for i in eachindex(suites["archive"]))
for (i, (a, c, cc)) in enumerate(zip(suites["archive"], suites["candidate"], suites["control"]))
    ra = (a.op(); a.check()); rc = (c.op(); c.check()); rcc = (cc.op(); cc.check())
    for (key, d) in ((:oracle_archive, deviation(ra, a.ref, a)), (:oracle_candidate, deviation(rc, a.ref, a)),
                     (:archive_candidate, deviation(ra, rc, a)), (:control_candidate, deviation(rcc, rc, a)))
        dev[key] = max(dev[key], d)
        d <= 1 || error("$(key) deviates by $d tolerances: $(a.id)")
    end
    if a.kind == :vprox       # the fused value is the value at the output of the role
        for (key, r) in ((:fused_archive, ra), (:fused_candidate, rc))
            d = deviation(r[1], r[3], a)
            dev[key] = max(dev[key], d)
            d <= 1 || error("$(key): the fused value differs from the value at the output by $d tolerances: $(a.id)")
        end
    end
    bitwise[:archive_candidate] += isequal(ra, rc); bitwise[:control_candidate] += isequal(rcc, rc)
    for (r, p) in (("archive", a), ("candidate", c), ("control", cc))
        alloc[(r, i)] = warm_alloc(p.op)
    end
end
ncases = length(suites["archive"])
println("outputs agree for $ncases cases (reversed roles: $reversed)")
for k in (:oracle_archive, :oracle_candidate, :archive_candidate, :control_candidate, :fused_archive, :fused_candidate)
    @printf("  largest deviation %-18s %.3g tolerances\n", k, dev[k])
end
println("  bitwise equal: archive/candidate $(bitwise[:archive_candidate])/$ncases, control/candidate $(bitwise[:control_candidate])/$ncases")
for (i, p) in enumerate(suites["archive"])
    alloc[("archive", i)] == alloc[("candidate", i)] == alloc[("control", i)] ||
        println("  warmed bytes differ: ", p.id, "  archive ", alloc[("archive", i)], "  candidate ", alloc[("candidate", i)], "  control ", alloc[("control", i)])
end
allocating_archive = count(i -> alloc[("archive", i)] > 0, 1:ncases)
allocating_candidate = count(i -> alloc[("candidate", i)] > 0, 1:ncases)
println("  warmed allocating cases: archive $allocating_archive, candidate $allocating_candidate of $ncases")
check_only && exit()

function bench(p)
    op = p.op
    b = @benchmarkable $op() evals = max(1, 2^16 ÷ p.n) samples = 10_000 seconds = 0.04 gctrial = false
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-mcp-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds, "\nReversed roles: ", reversed, "\nRounds: ", rounds)
    println(io, "lambda: ", LAMBDA, "\nshape a: ", SHAPE, "\nsteps (center, width): ", STEP, "\nmargin (relative): ", MARGIN)
    println(io, "archive: ", archive_root, "\ncandidate: ", candidate_root)
end
open(joinpath(outdir, "allocations.csv"), "w") do io
    println(io, "case,archive_bytes,candidate_bytes,control_bytes")
    for (i, p) in enumerate(suites["archive"])
        println(io, p.id, ',', alloc[("archive", i)], ',', alloc[("candidate", i)], ',', alloc[("control", i)])
    end
end
for r in ("archive", "candidate", "control"), p in suites[r]     # compile and warm: one benchmark trial per role and case
    bench(p)
end
order = reversed ? ("candidate", "archive", "control") : ("archive", "candidate", "control")
ids = eachindex(suites["archive"])
mins = Dict((r, i) => Float64[] for r in first.(roles) for i in ids)
mem = Dict((r, i) => 0 for r in first.(roles) for i in ids)
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,min_ns,median_ns,memory,allocs")
    for round in 1:rounds
        for i in (isodd(round) ? ids : reverse(ids))
            for (k, r) in enumerate(isodd(round) ? order : reverse(order))
                t = bench(suites[r][i])
                m = minimum(t)
                push!(mins[(r, i)], m.time)
                mem[(r, i)] = max(mem[(r, i)], m.memory)
                println(io, suites[r][i].id, ',', round, ',', k, ',', r, ',', m.time, ',', median(t).time, ',', m.memory, ',', m.allocs)
            end
        end
        flush(io)
        println("round ", round, "/", rounds); flush(stdout)
    end
end
open(joinpath(outdir, "ratios.csv"), "w") do io
    println(io, "case,round,candidate_over_archive,control_over_candidate")
    for i in ids, round in 1:rounds
        println(io, suites["archive"][i].id, ',', round, ',', mins[("candidate", i)][round] / mins[("archive", i)][round], ',',
                mins[("control", i)][round] / mins[("candidate", i)][round])
    end
end
open(joinpath(outdir, "summary.txt"), "w") do io
    for out in (io, stdout)
        @printf(out, "%-62s %8s %8s %8s %8s %8s %8s %10s %8s\n", "case", "C/A med", "C/A min", "C/A max", "A/A med", "A/A min", "A/A max", "C ns", "bytes A/C")
        for i in ids
            ca = mins[("candidate", i)] ./ mins[("archive", i)]
            aa = mins[("control", i)] ./ mins[("candidate", i)]
            @printf(out, "%-62s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.1f %4d/%d%s\n", suites["archive"][i].id,
                    median(ca), minimum(ca), maximum(ca), median(aa), minimum(aa), maximum(aa),
                    median(mins[("candidate", i)]), mem[("archive", i)], mem[("candidate", i)], median(ca) > 1.05 ? "  > 1.05" : "")
        end
    end
end
