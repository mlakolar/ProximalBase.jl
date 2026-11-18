# Paired benchmark of IndNonnegative (#948) between the archived source tree (eff21b6d), the candidate and
# a second compilation of the candidate (the A/A control), Float32 and Float64: a loop of prox_block, dense
# prox! with a scalar step and with an array of steps, dense values at a feasible input and at an input with one
# negative entry (the last), value_and_prox! (a scalar step and an array of steps) with its comparator cells,
# SparseIterate, BlockIterate and SymmetricSparseIterate values and prox!, and the prox! of a Scaled
# IndNonnegative, at n = 1024 and 65536.
#
#   PB_ARCHIVE=<root of the archived tree> PB_CANDIDATE=<root of the candidate> PB_OUTPUT=<dir> \
#   PB_ROUNDS=21 julia --threads=1 --check-bounds=no --project=benchmark benchmark/indnonnegative.jl
#
# The three trees are loaded with Base.include into their own modules, as benchmark/run.jl does: by default
# in the order archive, candidate, control, and with PB_REVERSE=1 in the order candidate, archive, control,
# with the candidate also taking the place of the archive in the timing order. Each round times the three
# roles of every case in alternating order (the reverse order in even rounds), and keeps per role the
# minimum over the samples of its trial. The paired ratios of a round are candidate/archive and
# control/candidate (A/A); the summary reports their median, minimum and maximum over the rounds, with the
# warmed allocations of each role, and marks medians above 1.05. The roles share one set of plain input and
# output arrays (the iterates, whose types belong to each tree, are built per role), so that the placement of
# the arrays does not differ between the roles.
#
# The cases whose id starts with "comparator/" have a different first role. The archive does not take the
# check of the output that the candidate's value_and_prox! makes (its fused value is 0 without evaluating the
# value at the output), so the timing of value_and_prox! is judged against the candidate's own prox! followed
# by f(y), not against the archive. In these cells the "archive" role runs, in the candidate's tree, prox! and
# then f(y) with the output stored in the shared array, and the "candidate" and "control" roles run
# value_and_prox! of their trees: the column C/A of such a cell is the ratio of the candidate's
# value_and_prox! to its prox! followed by f(y). The ratio of value_and_prox! to the archive is the C/A column
# of the cells "value_and_prox!/...", reported and not judged.
#
# The inputs have no NaN, and their negative entries lie far below the tolerance of the archived value
# (sqrt(eps(T)): v >= -sqrt(eps(T)) is feasible there), so the outputs and values of the archive and the
# candidate are equal. The three differences from the archive are not exercised by the timed inputs: the
# archived value accepts a negative entry within its tolerance, its value at a NaN entry is Inf (the
# candidate's is NaN), and its fused value is always 0 (the candidate's is the value at the stored output).
# PB_CHECK_ONLY=1 only compares the outputs, bit for bit (isequal, which tells -0.0 from +0.0): proximal
# points, values, the fused values and the proximal points of value_and_prox!, the comparator cells against
# the candidate's value_and_prox!, and the control against the candidate. PB_FILTER selects cases by a
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
# the plain arrays of one element type, shared by the roles: entries of magnitude at least 0.01, above the
# tolerance of the archived value for Float32 (sqrt(eps) = 3.5e-4)
function makedata(::Type{T}) where {T}
    rng = MersenneTwister(948)
    mags(n) = T(0.01) .+ T(2) .* abs.(randn(rng, T, n))
    signs(n) = ifelse.(rand(rng, n) .< 0.5, -one(T), one(T))
    D = Dict{Any, Any}()
    for n in SIZES
        x = signs(n) .* mags(n)
        steps = T(0.5) .+ rand(rng, T, n)
        xpos = abs.(x)
        xneg = copy(xpos); xneg[end] = -one(T)
        D[n] = (; x, steps, xpos, xneg, out = similar(x))
    end
    n = 65536
    xsp = zeros(T, n); stored = randperm(rng, n)[1:(n ÷ 10)]; xsp[stored] .= signs(length(stored)) .* mags(length(stored))
    groups = [collect(k:64:n) for k in 1:64]
    xbl = copy(D[n].x); for k in 2:2:64; xbl[groups[k]] .= zero(T); end        # half of the groups active
    p = 256
    sym = [(i, j, (rand(rng) < 0.5 ? -one(T) : one(T)) * (T(0.01) + T(2) * abs(randn(rng, T)))) for j in 1:p for i in j:p if rand(rng) < 0.1]
    D[:iterates] = (; xsp, groups, xbl, p, sym, denseout = zeros(T, p, p))
    D
end

# The timed operation and the check of each kind of case. The arrays and the oracles of the role (prox_block,
# prox!, value_and_prox! of its tree) are arguments, so that each closure holds the arrays of its own size and
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
# the comparator of value_and_prox!: prox! and then the value at the output, stored as vprox_case stores it
cmp_case(px, out, f, x, γ) = (() -> (px(out, f, x, γ); out[1] = f(out); nothing), () -> (px(out, f, x, γ); v = f(out); (v, copy(out))))

function makeprobes(PB, role, ::Type{T}, D) where {T}
    probes = NamedTuple[]
    add(id, n, kind, (op, check)) = push!(probes, (id = "$(T)/$id", n = n, kind = kind, op = op, check = check))
    γ = T(0.7)
    pb, px, vp = PB.prox_block, PB.prox!, PB.value_and_prox!
    f = PB.IndNonnegative()
    # the comparator runs in the candidate's tree, in the role "archive"
    CPB = getfield(Candidate, :ProximalBase)
    cf, cpx = CPB.IndNonnegative(), CPB.prox!
    fs = PB.Scaled(f, T(2))
    for n in SIZES
        d = D[n]
        n == 1024 && add("prox_block_loop/n$n", n, :prox, loop_case(pb, f, d.x, γ))
        add("prox!/scalar_step/n$n", n, :prox, prox_case(px, d.out, f, d.x, γ))
        add("prox!/array_step/n$n", n, :prox, prox_case(px, d.out, f, d.x, d.steps))
        add("value/feasible/n$n", n, :value, value_case(d.out, f, d.xpos))
        add("value/one_negative_last/n$n", n, :value, value_case(d.out, f, d.xneg))
        add("value_and_prox!/scalar_step/n$n", n, :vprox, vprox_case(vp, d.out, f, d.x, γ))
        add("value_and_prox!/array_step/n$n", n, :vprox, vprox_case(vp, d.out, f, d.x, d.steps))
        if role == "archive"
            add("comparator/value_and_prox!/scalar_step/n$n", n, :vprox, cmp_case(cpx, d.out, cf, d.x, γ))
            add("comparator/value_and_prox!/array_step/n$n", n, :vprox, cmp_case(cpx, d.out, cf, d.x, d.steps))
        else
            add("comparator/value_and_prox!/scalar_step/n$n", n, :vprox, vprox_case(vp, d.out, f, d.x, γ))
            add("comparator/value_and_prox!/array_step/n$n", n, :vprox, vprox_case(vp, d.out, f, d.x, d.steps))
        end
        add("scaled_prox!/n$n", n, :prox, prox_case(px, d.out, fs, d.x, γ))
    end
    I = D[:iterates]
    n = length(I.xsp)
    sp = PB.SparseIterate(I.xsp); spout = copy(sp)
    add("sparse/value/n$n", n, :value, value_case(I.denseout, f, sp))
    add("sparse/prox!/n$n", n, :prox, (() -> (px(spout, f, sp, γ); nothing), () -> (px(spout, f, sp, γ); Vector(spout))))
    bl = PB.BlockIterate(copy(I.xbl), I.groups); blout = copy(bl)
    add("block/value/n$n", n, :value, value_case(I.denseout, f, bl))
    add("block/prox!/n$n", n, :prox, (() -> (px(blout, f, bl, γ); nothing), () -> (px(blout, f, bl, γ); Vector(blout))))
    S = PB.SymmetricSparseIterate(T, I.p)
    for (i, j, v) in I.sym
        S[i, j] = v
    end
    Sout = copy(S)
    add("symmetric/value/p$(I.p)", I.p^2, :value, value_case(I.denseout, f, S))
    add("symmetric/prox!/p$(I.p)", I.p^2, :prox, (() -> (px(Sout, f, S, γ); nothing), () -> (px(Sout, f, S, γ); Matrix(Sout))))
    probes
end

const data = Dict(T => makedata(T) for T in (Float32, Float64))
const roles = ("archive" => Archive, "candidate" => Candidate, "control" => CandidateControl)
const suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), r, T, data[T]) for T in (Float32, Float64)]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["archive"], :id) == getfield.(suites["candidate"], :id) == getfield.(suites["control"], :id)

# the outputs are compared bit for bit: the inputs avoid the three intended differences from the archive
# (the tolerance of its value, its value at a NaN entry and its fused value)
# the roles share their output arrays, so each result is read right after its own call
for (a, c, cc) in zip(suites["archive"], suites["candidate"], suites["control"])
    ra = (a.op(); a.check()); rc = (c.op(); c.check()); rcc = (cc.op(); cc.check())
    isequal(rc, rcc) || error("A/A output mismatch: $(a.id)")
    isequal(rc, ra) || error("candidate and archive disagree: $(a.id)")
end
println("outputs agree for $(length(suites["archive"])) cases (reversed roles: $reversed)")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    b = @benchmarkable $op() evals = max(1, 2^16 ÷ p.n) samples = 10_000 seconds = 0.04 gctrial = false
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-indnonnegative-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds, "\nReversed roles: ", reversed)
    println(io, "archive: ", archive_root, "\ncandidate: ", candidate_root)
end
for r in ("archive", "candidate", "control"), p in suites[r]     # compile and warm: each probe of each role runs once
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
