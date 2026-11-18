# Paired benchmark of NormL1Nonneg (#947) between the archived source tree (eff21b6d), the candidate and
# a second compilation of the candidate (the A/A control), Float32 and Float64, at n = 8, 1024, 2048 and 65536
# (both sides of the loop switches of the dense value, 16 to 1024 entries, and of the soft threshold, 16 KiB per
# array): a loop of prox_block, dense prox! (unweighted and weighted, a zero level, per-entry steps), dense
# values at a feasible point (unweighted, weighted, a zero level), value_and_prox!, SparseIterate,
# BlockIterate and SymmetricSparseIterate values and prox!, and the prox! and value_and_prox! of a Scaled
# NormL1Nonneg. Cases whose id starts with `info/` time inputs that take the entrywise fallback (a negative
# entry, a NaN entry) once, so that a cliff is visible; they are not part of the comparison of the plan.
#
#   PB_ARCHIVE=<root of the archived tree> PB_CANDIDATE=<root of the candidate> PB_OUTPUT=<dir> \
#   PB_ROUNDS=21 julia --threads=1 --check-bounds=no --project=benchmark benchmark/norml1nonneg.jl
#
# The three trees are loaded with Base.include into their own modules, as benchmark/run.jl does: by default
# in the order archive, candidate, control, and with PB_REVERSE=1 in the order candidate, archive, control,
# with the candidate also taking the place of the archive in the timing order. Each round times the three
# roles of every case in alternating order (the reverse order in even rounds), and keeps per role the
# minimum over the samples of its trial. The paired ratios of a round are candidate/archive and
# control/candidate (A/A); the summary reports their median, minimum and maximum over the rounds, with the
# warmed allocations of each role. The roles share one set of plain input and output arrays (the iterates,
# whose types belong to each tree, are built per role), so that the placement of the arrays does not differ
# between the roles; a dense call takes its argument in turn from four arrays, so that consecutive calls do
# not repeat their input. PB_CHECK_ONLY=1 only compares the outputs: proximal points within 4 units in the last
# place of the archived ones, values within 2(n + 2) eps relative to the archived value at feasible points (each
# value is within (n + 2) eps of the exact one), and the control identical to the candidate; the `info/` cells
# are not compared with the archive. PB_FILTER selects cases by a regular expression on their ids. With
# PB_VS_NORML1=1 the archive role is replaced by NormL1 of the candidate tree (the same code, not the archive):
# the cells are then timed against NormL1 as information, and only the values at feasible points are compared.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates, Printf
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)
archive_root = abspath(ENV["PB_ARCHIVE"]); candidate_root = abspath(ENV["PB_CANDIDATE"])
reversed = get(ENV, "PB_REVERSE", "0") == "1"
vs_norml1 = get(ENV, "PB_VS_NORML1", "0") == "1"
module Archive end
module Candidate end
module CandidateControl end
for (m, root) in (reversed ? ((Candidate, candidate_root), (Archive, archive_root)) : ((Archive, archive_root), (Candidate, candidate_root)))
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
end
Base.include(CandidateControl, joinpath(candidate_root, "src", "ProximalBase.jl"))

const SIZES = (8, 1024, 2048, 65536)
const COPIES = 4
# the plain arrays of one element type, shared by the roles: COPIES arrays of each kind per size, so that
# consecutive calls of a dense case take different inputs
function makedata(::Type{T}) where {T}
    rng = MersenneTwister(947)
    D = Dict{Any, Any}()
    for n in SIZES
        xs = [T(2) .* randn(rng, T, n) for _ in 1:COPIES]
        xps = [abs.(x) for x in xs]
        xneg = [(y = copy(x); y[end] = -one(T); y) for x in xps]
        xnan = [(y = copy(x); y[end] = T(NaN); y) for x in xps]
        w = T(0.5) .+ rand(rng, T, n)
        steps = T(0.5) .+ rand(rng, T, n)
        D[n] = (; xs, xps, xneg, xnan, w, steps, out = similar(xs[1]), res = zeros(T, 1))
    end
    n = 65536
    xsp = zeros(T, n); stored = randperm(rng, n)[1:(n ÷ 10)]; xsp[stored] .= T(2) .* randn(rng, T, length(stored))
    groups = [collect(k:64:n) for k in 1:64]
    xbl = copy(D[n].xs[1]); for k in 2:2:64; xbl[groups[k]] .= zero(T); end        # half of the groups active
    p = 256
    W = T(0.5) .+ rand(rng, T, p, p); W = (W + W') / 2
    sym = [(i, j, T(2) * randn(rng, T)) for j in 1:p for i in j:p if rand(rng) < 0.1]
    D[:iterates] = (; xsp, groups, xbl, p, W, sym, wn = D[n].w, denseout = zeros(T, p, p))
    D
end

# The timed operation and the check of each kind of case. The arrays and the oracles of the role (prox_block,
# prox!, value_and_prox! of its tree) are arguments, so that every closure holds the arrays of its own size and
# calls the oracle without a lookup in the module of the role. A dense operation takes its argument in turn from
# the COPIES arrays; its check reads the first.
function prox_block_loop(pb::P, f, x, γ) where {P}
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += pb(f, x[i], i, γ)
    end
    s
end
turns(xs) = (k = Ref(0); () -> (k[] = k[] == length(xs) ? 1 : k[] + 1; xs[k[]]))
loop_case(pb, f, xs, γ) = (let nxt = turns(xs); () -> prox_block_loop(pb, f, nxt(), γ) end,
                           () -> [pb(f, xs[1][i], i, γ) for i in eachindex(xs[1])])
prox_case(px, out, f, xs, γ) = (let nxt = turns(xs); () -> (px(out, f, nxt(), γ); nothing) end,
                                () -> (px(out, f, xs[1], γ); copy(out)))
value_case(res, f, xs) = (let nxt = turns(xs); () -> (res[1] = f(nxt()); nothing) end, () -> f(xs[1]))
vprox_case(vp, res, out, f, xs, γ) = (let nxt = turns(xs); () -> (res[1] = vp(out, f, nxt(), γ); nothing) end,
                                      () -> (v = vp(out, f, xs[1], γ); (v, copy(out))))

# the archive role builds NormL1Nonneg of the archived tree; with PB_VS_NORML1 it builds NormL1 of the candidate
function makeprobes(PB, ctor, ::Type{T}, D) where {T}
    probes = NamedTuple[]
    add(id, n, kind, (op, check)) = push!(probes, (id = "$(T)/$id", n = n, kind = kind, op = op, check = check))
    λ0, γ = T(0.3), T(0.7)
    pb, px, vp = PB.prox_block, PB.prox!, PB.value_and_prox!
    for n in SIZES
        d = D[n]
        fu, fw, f0, f0u = ctor(λ0), ctor(λ0, d.w), ctor(zero(T), d.w), ctor(zero(T))
        fs = PB.Scaled(fw, T(2))
        n == 1024 && add("prox_block_loop/unweighted/n$n", n, :prox, loop_case(pb, fu, d.xs, γ))
        n == 1024 && add("prox_block_loop/weighted/n$n", n, :prox, loop_case(pb, fw, d.xs, γ))
        add("prox!/unweighted/n$n", n, :prox, prox_case(px, d.out, fu, d.xs, γ))
        add("prox!/weighted/n$n", n, :prox, prox_case(px, d.out, fw, d.xs, γ))
        add("prox!/zero_level_weighted/n$n", n, :prox, prox_case(px, d.out, f0, d.xs, γ))
        add("prox!/zero_level_unweighted/n$n", n, :prox, prox_case(px, d.out, f0u, d.xs, γ))
        add("prox!/array_step_weighted/n$n", n, :prox, prox_case(px, d.out, fw, d.xs, d.steps))
        add("value/unweighted/n$n", n, :value, value_case(d.res, fu, d.xps))
        add("value/weighted/n$n", n, :value, value_case(d.res, fw, d.xps))
        add("value/zero_level_weighted/n$n", n, :value, value_case(d.res, f0, d.xps))
        add("value/zero_level_unweighted/n$n", n, :value, value_case(d.res, f0u, d.xps))
        add("value_and_prox!/unweighted/n$n", n, :vprox, vprox_case(vp, d.res, d.out, fu, d.xs, γ))
        add("value_and_prox!/weighted/n$n", n, :vprox, vprox_case(vp, d.res, d.out, fw, d.xs, γ))
        add("scaled_prox!/weighted/n$n", n, :prox, prox_case(px, d.out, fs, d.xs, γ))
        add("scaled_value_and_prox!/weighted/n$n", n, :vprox, vprox_case(vp, d.res, d.out, fs, d.xs, γ))
        # inputs that leave the feasible set: the entrywise fallback of the value (timed once, not compared)
        if n == 1024 || n == 65536
            add("info/value/unweighted_negative_entry/n$n", n, :info, value_case(d.res, fu, d.xneg))
            add("info/value/weighted_negative_entry/n$n", n, :info, value_case(d.res, fw, d.xneg))
            add("info/value/unweighted_nan_entry/n$n", n, :info, value_case(d.res, fu, d.xnan))
            add("info/value/weighted_nan_entry/n$n", n, :info, value_case(d.res, fw, d.xnan))
        end
    end
    I = D[:iterates]
    n = length(I.xsp)
    fw = ctor(λ0, I.wn)
    spx = PB.SparseIterate(I.xsp); spp = PB.SparseIterate(abs.(I.xsp)); spout = copy(spx)
    add("sparse/value/n$n", n, :value, value_case(I.denseout, fw, [spp]))
    add("sparse/prox!/n$n", n, :prox, (() -> (px(spout, fw, spx, γ); nothing), () -> (px(spout, fw, spx, γ); Vector(spout))))
    blx = PB.BlockIterate(copy(I.xbl), I.groups); blp = PB.BlockIterate(abs.(I.xbl), I.groups); blout = copy(blx)
    add("block/value/n$n", n, :value, value_case(I.denseout, fw, [blp]))
    add("block/prox!/n$n", n, :prox, (() -> (px(blout, fw, blx, γ); nothing), () -> (px(blout, fw, blx, γ); Vector(blout))))
    S = PB.SymmetricSparseIterate(T, I.p); Sp = PB.SymmetricSparseIterate(T, I.p)
    for (i, j, v) in I.sym
        S[i, j] = v
        Sp[i, j] = abs(v)
    end
    Sout = copy(S); fsym = ctor(λ0, I.W)
    add("symmetric/value/p$(I.p)", I.p^2, :value, value_case(I.denseout, fsym, [Sp]))
    add("symmetric/prox!/p$(I.p)", I.p^2, :prox, (() -> (px(Sout, fsym, S, γ); nothing), () -> (px(Sout, fsym, S, γ); Matrix(Sout))))
    probes
end

const data = Dict(T => makedata(T) for T in (Float32, Float64))
const roles = ("archive" => (Archive, :NormL1Nonneg), "candidate" => (Candidate, :NormL1Nonneg), "control" => (CandidateControl, :NormL1Nonneg))
const role_trees = vs_norml1 ? ("archive" => (Candidate, :NormL1), roles[2], roles[3]) : roles
const suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), getfield(getfield(m, :ProximalBase), c), T, data[T])
                               for T in (Float32, Float64)]...) for (r, (m, c)) in role_trees)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["archive"], :id) == getfield.(suites["candidate"], :id) == getfield.(suites["control"], :id)

within_ulps(a::AbstractFloat, b::AbstractFloat, k) = isequal(a, b) || abs(a - b) <= k * eps(max(abs(a), abs(b)))
within_ulps(a::AbstractArray, b::AbstractArray, k) = size(a) == size(b) && all(within_ulps(u, v, k) for (u, v) in zip(a, b))
within_value(a, b, n) = isequal(a, b) || abs(a - b) <= 2 * (n + 2) * eps(typeof(a)) * abs(b)
function agrees(kind, n, c, a)
    kind === :info && return true
    vs_norml1 && kind !== :value && return true
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
println("outputs agree for $(length(suites["archive"])) cases (reversed roles: $reversed, against NormL1: $vs_norml1)")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    b = @benchmarkable $op() evals = max(1, 2^16 ÷ p.n) samples = 10_000 seconds = 0.04 gctrial = false
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-norml1nonneg-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds, "\nReversed roles: ", reversed, "\nAgainst NormL1: ", vs_norml1)
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
        @printf(out, "%-54s %8s %8s %8s %8s %8s %8s %10s %8s\n", "case", "C/A med", "C/A min", "C/A max", "A/A med", "A/A min", "A/A max", "C ns", "bytes A/C")
        for i in ids
            ca = mins[("candidate", i)] ./ mins[("archive", i)]
            aa = mins[("control", i)] ./ mins[("candidate", i)]
            @printf(out, "%-54s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %10.1f %4d/%d%s\n", suites["archive"][i].id,
                    median(ca), minimum(ca), maximum(ca), median(aa), minimum(aa), maximum(aa),
                    median(mins[("candidate", i)]), mem[("archive", i)], mem[("candidate", i)], median(ca) > 1.05 ? "  > 1.05" : "")
        end
    end
end
