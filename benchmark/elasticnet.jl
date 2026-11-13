# Paired benchmark of ElasticNet (#908) between the archived source tree (eff21b6d), the candidate and
# a second compilation of the candidate (the A/A control), Float32 and Float64: a loop of prox_block,
# dense prox! (unweighted and weighted, a zero λ1 or λ2, per-entry steps), dense values (unweighted,
# weighted, and the two regimes where one part dominates), value_and_prox!, SparseIterate, BlockIterate
# and SymmetricSparseIterate values and prox!, and the prox! of a Scaled ElasticNet.
#
#   PB_ARCHIVE=<root of the archived tree> PB_CANDIDATE=<root of the candidate> PB_OUTPUT=<dir> \
#   PB_ROUNDS=21 julia --threads=1 --check-bounds=no --project=benchmark benchmark/elasticnet.jl
#
# The three trees are loaded with Base.include into their own modules, as benchmark/run.jl does: by default
# in the order archive, candidate, control, and with PB_REVERSE=1 in the order candidate, archive, control,
# with the candidate also taking the place of the archive in the timing order. Each round times the three
# roles of every case in alternating order (the reverse order in even rounds), and keeps per role the
# minimum over the samples of its trial. The paired ratios of a round are candidate/archive and
# control/candidate (A/A); the summary reports their median, minimum and maximum over the rounds, with the
# warmed allocations of each role. The roles share one set of plain input and output arrays (the iterates,
# whose types belong to each tree, are built per role), so that the placement of the arrays does not differ
# between the roles. PB_CHECK_ONLY=1 only compares the outputs: proximal points within 4 units in the last
# place of the archived ones, values within 2(n + 2) eps relative to the archived value (each value is within
# (n + 2) eps of the exact one), and the control identical to the candidate. PB_FILTER selects cases by a
# regular expression on their ids.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates, Printf
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)
archive_root = abspath(ENV["PB_ARCHIVE"]); candidate_root = abspath(ENV["PB_CANDIDATE"])
reversed = get(ENV, "PB_REVERSE", "0") == "1"
module Archive end
module Candidate end
module CandidateControl end
for (m, root) in (reversed ? ((Candidate, candidate_root), (Archive, archive_root)) : ((Archive, archive_root), (Candidate, candidate_root)))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
end
Base.include(CandidateControl, joinpath(candidate_root, "src", "ProximalBase.jl"))

const SIZES = (1024, 65536)
# the plain arrays of one element type, shared by the roles
function makedata(::Type{T}) where {T}
    rng = MersenneTwister(908)
    D = Dict{Any, Any}()
    for n in SIZES
        x = T(2) .* randn(rng, T, n)
        w = T(0.5) .+ rand(rng, T, n)
        steps = T(0.5) .+ rand(rng, T, n)
        signs = ifelse.(rand(rng, n) .< 0.5, -one(T), one(T))
        # the ℓ1 terms below floatmin and the ridge part dominant; the squares below floatmin and the ℓ1 part dominant
        xr = signs .* (T === Float64 ? 1e-120 : 1f-15) .* (one(T) .+ rand(rng, T, n))
        xl = signs .* (T === Float64 ? 1e-170 : 1f-22) .* (one(T) .+ rand(rng, T, n))
        D[n] = (; x, w, steps, xr, xl, out = similar(x))
    end
    n = 65536
    xsp = zeros(T, n); stored = randperm(rng, n)[1:(n ÷ 10)]; xsp[stored] .= T(2) .* randn(rng, T, length(stored))
    groups = [collect(k:64:n) for k in 1:64]
    xbl = copy(D[n].x); for k in 2:2:64; xbl[groups[k]] .= zero(T); end        # half of the groups active
    p = 256
    W = T(0.5) .+ rand(rng, T, p, p); W = (W + W') / 2
    sym = [(i, j, T(2) * randn(rng, T)) for j in 1:p for i in j:p if rand(rng) < 0.1]
    D[:iterates] = (; xsp, groups, xbl, p, W, sym, wn = D[n].w, denseout = zeros(T, p, p))
    D
end

# The timed operation and the check of each kind of case. The arrays and the oracles of the role (prox_block,
# prox!, value_and_prox! of its tree) are arguments, so that every closure holds the arrays of its own size and
# calls the oracle without a lookup in the module of the role.
function prox_block_loop(pb::P, f, x, γ) where {P}
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += pb(f, x[i], i, γ)
    end
    s
end
loop_case(pb, f, x, γ) = (() -> prox_block_loop(pb, f, x, γ), () -> [pb(f, x[i], i, γ) for i in eachindex(x)])
prox_case(px, out, f, x, γ) = (() -> (px(out, f, x, γ); nothing), () -> (px(out, f, x, γ); copy(out)))
value_case(out, f, x) = (() -> (out[1] = f(x); nothing), () -> f(x))
vprox_case(vp, out, f, x, γ) = (() -> (out[1] = vp(out, f, x, γ); nothing), () -> (v = vp(out, f, x, γ); (v, copy(out))))

function makeprobes(PB, ::Type{T}, D) where {T}
    probes = NamedTuple[]
    add(id, n, kind, (op, check)) = push!(probes, (id = "$(T)/$id", n = n, kind = kind, op = op, check = check))
    λ1, λ2, γ = T(0.3), T(0.5), T(0.7)
    pb, px, vp = PB.prox_block, PB.prox!, PB.value_and_prox!
    for n in SIZES
        d = D[n]
        fu, fw = PB.ElasticNet(λ1, λ2), PB.ElasticNet(λ1, λ2, d.w)
        f0, f2 = PB.ElasticNet(zero(T), λ2, d.w), PB.ElasticNet(λ1, zero(T), d.w)
        fr, fl = PB.ElasticNet(T === Float64 ? T(1e-200) : T(1f-25), one(T), d.w), PB.ElasticNet(one(T), one(T), d.w)
        fs = PB.Scaled(fw, T(2))
        n == 1024 && add("prox_block_loop/unweighted/n$n", n, :prox, loop_case(pb, fu, d.x, γ))
        n == 1024 && add("prox_block_loop/weighted/n$n", n, :prox, loop_case(pb, fw, d.x, γ))
        add("prox!/unweighted/n$n", n, :prox, prox_case(px, d.out, fu, d.x, γ))
        add("prox!/weighted/n$n", n, :prox, prox_case(px, d.out, fw, d.x, γ))
        add("prox!/lambda1_zero/n$n", n, :prox, prox_case(px, d.out, f0, d.x, γ))
        add("prox!/lambda2_zero/n$n", n, :prox, prox_case(px, d.out, f2, d.x, γ))
        add("prox!/array_step_weighted/n$n", n, :prox, prox_case(px, d.out, fw, d.x, d.steps))
        add("value/unweighted/n$n", n, :value, value_case(d.out, fu, d.x))
        add("value/weighted/n$n", n, :value, value_case(d.out, fw, d.x))
        add("value/ridge_dominant/n$n", n, :value, value_case(d.out, fr, d.xr))
        add("value/l1_dominant/n$n", n, :value, value_case(d.out, fl, d.xl))
        add("value_and_prox!/weighted/n$n", n, :vprox, vprox_case(vp, d.out, fw, d.x, γ))
        add("scaled_prox!/weighted/n$n", n, :prox, prox_case(px, d.out, fs, d.x, γ))
    end
    I = D[:iterates]
    n = length(I.xsp)
    fw = PB.ElasticNet(λ1, λ2, I.wn)
    sp = PB.SparseIterate(I.xsp); spout = copy(sp)
    add("sparse/value/n$n", n, :value, value_case(I.denseout, fw, sp))
    add("sparse/prox!/n$n", n, :prox, (() -> (px(spout, fw, sp, γ); nothing), () -> (px(spout, fw, sp, γ); Vector(spout))))
    bl = PB.BlockIterate(copy(I.xbl), I.groups); blout = copy(bl)
    add("block/value/n$n", n, :value, value_case(I.denseout, fw, bl))
    add("block/prox!/n$n", n, :prox, (() -> (px(blout, fw, bl, γ); nothing), () -> (px(blout, fw, bl, γ); Vector(blout))))
    S = PB.SymmetricSparseIterate(T, I.p)
    for (i, j, v) in I.sym
        S[i, j] = v
    end
    Sout = copy(S); fsym = PB.ElasticNet(λ1, λ2, I.W)
    add("symmetric/value/p$(I.p)", I.p^2, :value, value_case(I.denseout, fsym, S))
    add("symmetric/prox!/p$(I.p)", I.p^2, :prox, (() -> (px(Sout, fsym, S, γ); nothing), () -> (px(Sout, fsym, S, γ); Matrix(Sout))))
    probes
end

const data = Dict(T => makedata(T) for T in (Float32, Float64))
const roles = ("archive" => Archive, "candidate" => Candidate, "control" => CandidateControl)
const suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), T, data[T]) for T in (Float32, Float64)]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["archive"], :id) == getfield.(suites["candidate"], :id) == getfield.(suites["control"], :id)

within_ulps(a::AbstractFloat, b::AbstractFloat, k) = isequal(a, b) || abs(a - b) <= k * eps(max(abs(a), abs(b)))
within_ulps(a::AbstractArray, b::AbstractArray, k) = size(a) == size(b) && all(within_ulps(u, v, k) for (u, v) in zip(a, b))
within_value(a, b, n) = isequal(a, b) || abs(a - b) <= 2 * (n + 2) * eps(typeof(a)) * abs(b)
function agrees(kind, n, c, a)
    kind === :prox && return within_ulps(c, a, 4)
    kind === :value && return within_value(c, a, n)
    within_value(c[1], a[1], n) && within_ulps(c[2], a[2], 4)
end
# the roles share their output arrays, so each result is read right after its own call
for (a, c, cc) in zip(suites["archive"], suites["candidate"], suites["control"])
    ra = (a.op(); a.check()); rc = (c.op(); c.check()); rcc = (cc.op(); cc.check())
    isequal(rc, rcc) || error("A/A output mismatch: $(a.id)")
    agrees(a.kind, a.n, rc, ra) || error("candidate and archive disagree: $(a.id)")
end
println("outputs agree for $(length(suites["archive"])) cases (reversed roles: $reversed)")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    b = @benchmarkable $op() evals = max(1, 2^16 ÷ p.n) samples = 10_000 seconds = 0.04 gctrial = false
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-elasticnet-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds, "\nReversed roles: ", reversed)
    println(io, "archive: ", archive_root, "\ncandidate: ", candidate_root)
end
for r in ("archive", "candidate", "control"), p in suites[r]     # compile and warm every specialization
    bench(p)
end
rounds = parse(Int, get(ENV, "PB_ROUNDS", "21"))
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
open(joinpath(outdir, "summary.txt"), "w") do io
    for out in (io, stdout)
        @printf(out, "%-48s %8s %8s %8s %8s %8s %8s %10s %8s\n", "case", "C/A med", "C/A min", "C/A max", "A/A med", "A/A min", "A/A max", "C ns", "bytes A/C")
        for i in ids
            ca = mins[("candidate", i)] ./ mins[("archive", i)]
            aa = mins[("control", i)] ./ mins[("candidate", i)]
            @printf(out, "%-48s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.1f %4d/%d%s\n", suites["archive"][i].id,
                    median(ca), minimum(ca), maximum(ca), median(aa), minimum(aa), maximum(aa),
                    median(mins[("candidate", i)]), mem[("archive", i)], mem[("candidate", i)], median(ca) > 1.05 ? "  > 1.05" : "")
        end
    end
end
