# Paired benchmark for #910, 76 cases in Float32 and Float64 at n = 1024 and 65536:
# - ElasticNet with an array of steps: dense prox! (weighted, unweighted, and with every 64th step outside the
#   ridge band), value_and_prox!, and a SparseIterate (n = 65536, n ÷ 10 entries set);
# - ElasticNet with a scalar step: dense prox!, a loop of prox_block (n = 1024 only), and the SparseIterate;
# - SqrNormL2 without a center, with a center of the data type and (Float32 data) with a Float64 center: loops of
#   prox_block with an array of steps (without a center, also with the mixed steps above) and, except with the
#   Float64 center, with a scalar step; scalar prox!; value_and_prox! with the center of the data type; and the
#   SparseIterate without a center;
# - neighbours outside #910 (SqrNormL2 value and gradient!, NormL1 prox! with an array of steps), controls for
#   code placement.
# Roles: a base tree, one or more candidate trees and a second compilation of the base (the A/A control).
#
#   PB_BASE=<root of the base tree> PB_CANDIDATES="A=<root>,B=<root>" PB_OUTPUT=<dir> PB_ROUNDS=21 \
#   julia --threads=1 --check-bounds=no --project=benchmark benchmark/ridge910.jl
#
# Every tree is loaded with Base.include into its own module, as in benchmark/elasticnet.jl. Each round times every
# role of every case in rotating order (the order shifts by one role each round, and the case order reverses
# in even rounds), keeping per role the minimum over the samples of its trial. The paired ratios of a round
# are candidate/base for each candidate and control/base (A/A); the summary gives their median, minimum and
# maximum over the rounds, the bytes allocated per evaluation after a warm-up (the largest over the rounds), and
# marks a case where the median of a candidate's ratio exceeds 1.05. PB_REVERSE=1 loads the candidates before the
# base, which also puts them first in the role order of the first round; the control is loaded last in both orders.
# PB_ROUNDS defaults to 21, PB_OUTPUT to /tmp/pb-ridge910-bench.
# The roles share their plain input and output arrays; each role builds its own SparseIterate. Before timing, every
# candidate output is compared with the base output: the number of cases that are bit-identical is reported, and a
# case whose outputs differ by more than 4 ulp, or differ at all in a non-finite entry, stops the run. The control
# must equal the base. PB_CHECK_ONLY=1 stops after that check; PB_FILTER selects cases by a regular expression on
# their ids.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates, Printf
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)
base_root = abspath(ENV["PB_BASE"])
cand_spec = [Pair(split(s, "=")[1], abspath(split(s, "=")[2])) for s in split(ENV["PB_CANDIDATES"], ",")]
reversed = get(ENV, "PB_REVERSE", "0") == "1"
module_for(name) = (m = Module(Symbol("PB_", name)); m)
roles = Pair{String, Module}[]
loads = vcat(["base" => base_root], cand_spec)
reversed && (loads = vcat(cand_spec, ["base" => base_root]))
for (name, root) in loads
    m = module_for(name)
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    push!(roles, name => m)
end
let m = module_for("control")
    Base.include(m, joinpath(base_root, "src", "ProximalBase.jl"))
    push!(roles, "control" => m)
end
const ROLE_NAMES = first.(roles)
const CANDS = first.(cand_spec)

const SIZES = (1024, 65536)
function makedata(::Type{T}) where {T}
    rng = MersenneTwister(910)
    D = Dict{Any, Any}()
    for n in SIZES
        x = T(2) .* randn(rng, T, n)
        w = T(0.5) .+ rand(rng, T, n)
        c = randn(rng, T, n)
        c64 = randn(rng, Float64, n)
        steps = T(0.5) .+ rand(rng, T, n)
        # every 64th step makes γλ (λ = 0.5) leave the ordinary band, finite, so that the scaled weights apply:
        # 2^1022 above the Float64 band top 2^1021, 2^126 above the Float32 band top 2^125
        mixed = copy(steps); mixed[1:64:end] .= T === Float64 ? T(2.0^1023) : T(2f0^127)
        D[n] = (; x, w, c, c64, steps, mixed, out = similar(x))
    end
    n = 65536
    xsp = zeros(T, n); stored = randperm(rng, n)[1:(n ÷ 10)]; xsp[stored] .= T(2) .* randn(rng, T, length(stored))
    D[:sparse] = (; xsp, steps = D[n].steps)
    D
end

function prox_block_loop(pb::P, f, x, γ) where {P}
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += pb(f, x[i], i, γ)
    end
    s
end
loop_case(pb, f, x, γ) = (() -> prox_block_loop(pb, f, x, γ), () -> [pb(f, x[i], i, γ) for i in eachindex(x)])
# one step per entry through prox_block: the only per-entry route of SqrNormL2, which has no array-step prox!
function prox_block_steps(pb::P, f, x, γ) where {P}
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s += pb(f, x[i], i, γ[i])
    end
    s
end
steps_case(pb, f, x, γ) = (() -> prox_block_steps(pb, f, x, γ), () -> [pb(f, x[i], i, γ[i]) for i in eachindex(x)])
prox_case(px, out, f, x, γ) = (() -> (px(out, f, x, γ); nothing), () -> (px(out, f, x, γ); copy(out)))
vprox_case(vp, out, f, x, γ) = (() -> (out[1] = vp(out, f, x, γ); nothing), () -> (v = vp(out, f, x, γ); vcat(v, out)))
value_case(out, f, x) = (() -> (out[1] = f(x); nothing), () -> [f(x)])
grad_case(gr, out, f, x) = (() -> (gr(out, f, x); nothing), () -> (gr(out, f, x); copy(out)))

function makeprobes(PB, ::Type{T}, D) where {T}
    probes = NamedTuple[]
    add(id, n, (op, check)) = push!(probes, (id = "$(T)/$id", n = n, op = op, check = check))
    λ, λ1, γ = T(0.5), T(0.3), T(0.7)
    pb, px, vp, gr = PB.prox_block, PB.prox!, PB.value_and_prox!, PB.gradient!
    for n in SIZES
        d = D[n]
        r0, rc = PB.SqrNormL2(λ), PB.SqrNormL2(λ, d.c)
        en, enu = PB.ElasticNet(λ1, λ, d.w), PB.ElasticNet(λ1, λ)
        # SqrNormL2 without a center
        add("ridge/prox!/scalar/n$n", n, prox_case(px, d.out, r0, d.x, γ))
        add("ridge/prox_block_loop/steps/n$n", n, steps_case(pb, r0, d.x, d.steps))
        add("ridge/prox_block_loop/steps_mixed/n$n", n, steps_case(pb, r0, d.x, d.mixed))
        add("ridge/prox_block_loop/scalar/n$n", n, loop_case(pb, r0, d.x, γ))
        # SqrNormL2 with a center of the data type, and (Float32 data) a Float64 center
        add("ridge_c/prox!/scalar/n$n", n, prox_case(px, d.out, rc, d.x, γ))
        add("ridge_c/prox_block_loop/steps/n$n", n, steps_case(pb, rc, d.x, d.steps))
        add("ridge_c/prox_block_loop/scalar/n$n", n, loop_case(pb, rc, d.x, γ))
        if T === Float32
            rw = PB.SqrNormL2(Float64(λ), d.c64)
            add("ridge_c64/prox!/scalar/n$n", n, prox_case(px, d.out, rw, d.x, γ))
            add("ridge_c64/prox_block_loop/steps/n$n", n, steps_case(pb, rw, d.x, d.steps))
        end
        add("ridge/value_and_prox!/scalar/n$n", n, vprox_case(vp, d.out, rc, d.x, γ))
        # ElasticNet
        add("enet/prox!/scalar/n$n", n, prox_case(px, d.out, en, d.x, γ))
        add("enet/prox!/array/n$n", n, prox_case(px, d.out, en, d.x, d.steps))
        add("enet/prox!/array_unweighted/n$n", n, prox_case(px, d.out, enu, d.x, d.steps))
        add("enet/prox!/array_mixed/n$n", n, prox_case(px, d.out, en, d.x, d.mixed))
        n == 1024 && add("enet/prox_block_loop/scalar/n$n", n, loop_case(pb, en, d.x, γ))
        add("enet/value_and_prox!/array/n$n", n, vprox_case(vp, d.out, en, d.x, d.steps))
        # unchanged neighbours (controls for code placement)
        add("ridge/value/n$n", n, value_case(d.out, rc, d.x))
        add("ridge/gradient!/n$n", n, grad_case(gr, d.out, rc, d.x))
        add("norml1/prox!/array/n$n", n, prox_case(px, d.out, PB.NormL1(λ1, d.w), d.x, d.steps))
    end
    S = D[:sparse]
    n = length(S.xsp)
    sp = PB.SparseIterate(S.xsp); spout = copy(sp)
    r0 = PB.SqrNormL2(λ)
    en = PB.ElasticNet(λ1, λ)
    add("enet/sparse_prox!/array/n$n", n, (() -> (px(spout, en, sp, S.steps); nothing), () -> (px(spout, en, sp, S.steps); Vector(spout))))
    add("enet/sparse_prox!/scalar/n$n", n, (() -> (px(spout, en, sp, γ); nothing), () -> (px(spout, en, sp, γ); Vector(spout))))
    add("ridge/sparse_prox!/scalar/n$n", n, (() -> (px(spout, r0, sp, γ); nothing), () -> (px(spout, r0, sp, γ); Vector(spout))))
    probes
end

const data = Dict(T => makedata(T) for T in (Float32, Float64))
const suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), T, data[T]) for T in (Float32, Float64)]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert allequal([getfield.(suites[r], :id) for r in ROLE_NAMES])

ulps(a::T, b::T) where {T<:AbstractFloat} = isequal(a, b) ? 0.0 : (isfinite(a) && isfinite(b)) ? abs(Float64(a) - Float64(b)) / eps(max(abs(a), abs(b), floatmin(T))) : Inf
maxulps(a::AbstractArray, b::AbstractArray) = maximum(ulps(u, v) for (u, v) in zip(a, b); init = 0.0)
identical = Dict(c => 0 for c in CANDS)
worst = Dict(c => 0.0 for c in CANDS)
for i in eachindex(suites["base"])
    outs = Dict(r => (suites[r][i].op(); suites[r][i].check()) for r in ROLE_NAMES)
    id = suites["base"][i].id
    isequal(outs["control"], outs["base"]) || error("A/A output mismatch: $id")
    for c in CANDS
        u = maxulps(outs[c], outs["base"])
        u <= 4 || error("candidate $c and base differ by $u ulp (or in a non-finite entry): $id")
        identical[c] += isequal(outs[c], outs["base"])
        worst[c] = max(worst[c], u)
    end
end
for c in CANDS
    println("candidate $c: outputs within 4 ulp of the base in $(length(suites["base"])) cases; bit-identical in $(identical[c]); largest difference $(worst[c]) ulp (reversed roles: $reversed)")
end
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    b = @benchmarkable $op() evals = max(1, 2^16 ÷ p.n) samples = 10_000 seconds = 0.04 gctrial = false
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-ridge910-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds, "\nReversed roles: ", reversed)
    println(io, "base: ", base_root)
    for (c, r) in cand_spec
        println(io, "candidate ", c, ": ", r)
    end
end
for r in ROLE_NAMES, p in suites[r]
    bench(p)
end
rounds = parse(Int, get(ENV, "PB_ROUNDS", "21"))
ids = eachindex(suites["base"])
mins = Dict((r, i) => Float64[] for r in ROLE_NAMES for i in ids)
mem = Dict((r, i) => 0 for r in ROLE_NAMES for i in ids)
nr = length(ROLE_NAMES)
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,min_ns,median_ns,memory,allocs")
    for round in 1:rounds
        order = circshift(collect(ROLE_NAMES), round - 1)
        for i in (isodd(round) ? ids : reverse(ids))
            for (k, r) in enumerate(order)
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
        @printf(out, "%-40s %10s", "case", "base ns")
        for c in vcat(CANDS, "control")
            @printf(out, " %22s", "$c/base med [min,max]")
        end
        @printf(out, " %s\n", "bytes")
        for i in ids
            @printf(out, "%-40s %10.1f", suites["base"][i].id, median(mins[("base", i)]))
            flag = false
            for c in vcat(CANDS, "control")
                q = mins[(c, i)] ./ mins[("base", i)]
                @printf(out, " %8.3f [%5.3f,%5.3f]", median(q), minimum(q), maximum(q))
                c != "control" && median(q) > 1.05 && (flag = true)
            end
            @printf(out, " %s%s\n", join([string(mem[(r, i)]) for r in ROLE_NAMES], "/"), flag ? "  > 1.05" : "")
        end
    end
end
