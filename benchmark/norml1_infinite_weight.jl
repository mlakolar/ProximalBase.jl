# Paired benchmark of the NormL1 proximal hot loops (weighted and unweighted, scalar and per-entry
# steps, dense, sparse and block storage, Float32/Float64). Written for the performance gate of
# issue #853 and kept for #866 and any numerical change that touches these loops.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=21 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/norml1_infinite_weight.jl
#
# Each root contains a src/ folder. Role X is compiled twice (X and the independent A/A control XC); role Y
# is the one under test. Each round times X, Y and XC in alternating order and the summary
# (benchmark/summarize_norml1_infinite_weight.py DIR...) uses the per-round ratios Y/X and XC/X; it pools
# only runs with the same X and Y trees. Run it a second time with the roles reversed (summarized on its
# own, the ratios are inverted) and once more in a fresh process. PB_CHECK_ONLY=1 only verifies that
# the outputs agree across the roles; PB_FILTER selects cases by a regular expression on their ids.
# All workloads use finite trials, where unchanged kernels agree exactly, so a numerical change that
# alters them is reported as an output mismatch before any timing.
using BenchmarkTools, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

function makeprobes(PB, ::Type{T}, n::Int) where {T}
    rng = Random.MersenneTwister(853 + n)
    x = T(2) .* randn(rng, T, n)
    w = T(0.5) .+ rand(rng, T, n)                      # positive finite weights
    winf = copy(w); winf[1:16:end] .= T(Inf)           # one infinite weight in sixteen
    wzero = copy(w); wzero[1:8:end] .= zero(T)         # one zero weight in eight
    γs = T(0.5) .+ rand(rng, T, n)                     # per-entry steps
    γ = T(0.7); λ = T(0.3)
    out = similar(x)
    probes = NamedTuple[]
    add(id, op, check) = push!(probes, (id = "$(T)/n$(n)/$id", op = op, check = check))
    f0 = PB.NormL1(λ)
    f1 = PB.NormL1(λ, w)
    fi = PB.NormL1(λ, winf)
    fz = PB.NormL1(λ, wzero)
    add("unweighted/scalar", () -> (PB.prox!(out, f0, x, γ); nothing), () -> copy(out))
    add("unweighted/array", () -> (PB.prox!(out, f0, x, γs); nothing), () -> copy(out))
    add("weighted/scalar", () -> (PB.prox!(out, f1, x, γ); nothing), () -> copy(out))
    add("weighted/array", () -> (PB.prox!(out, f1, x, γs); nothing), () -> copy(out))
    add("weighted_inf/scalar", () -> (PB.prox!(out, fi, x, γ); nothing), () -> copy(out))
    add("weighted_inf/array", () -> (PB.prox!(out, fi, x, γs); nothing), () -> copy(out))
    add("weighted_zero/scalar", () -> (PB.prox!(out, fz, x, γ); nothing), () -> copy(out))
    add("weighted/value_and_prox", () -> (PB.value_and_prox!(out, f1, x, γ); nothing), () -> (PB.value_and_prox!(out, f1, x, γ), copy(out)))
    add("weighted/alias", () -> (copyto!(out, x); PB.prox!(out, f1, out, γ); nothing), () -> copy(out))
    add("weighted/prox_block_loop", () -> begin
        s = zero(T)
        @inbounds for i in 1:n
            s += PB.prox_block(f1, x[i], i, γ)
        end
        out[1] = s; nothing
    end, () -> out[1])
    # stored-entry loops (they keep the per-entry kernel): 1/8 of the entries stored
    xsp = copy(x); xsp[mod.(1:n, 8) .!= 1] .= zero(T)
    sp = PB.SparseIterate(xsp); spout = copy(sp)
    add("sparse_weighted/scalar", () -> (PB.prox!(spout, f1, sp, γ); nothing), () -> Vector(spout))
    add("sparse_weighted/array", () -> (PB.prox!(spout, f1, sp, γs); nothing), () -> Vector(spout))
    groups = [collect(1:2:n), collect(2:2:n)]
    bl = PB.BlockIterate(copy(x), groups); blout = copy(bl)
    add("block_weighted/scalar", () -> (PB.prox!(blout, f1, bl, γ); nothing), () -> Vector(blout))
    add("block_weighted/array", () -> (PB.prox!(blout, f1, bl, γs); nothing), () -> Vector(blout))
    probes
end
sizes = (1024, 65536)
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), T, n) for T in (Float32, Float64) for n in sizes]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["X"], :id) == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)
for (a, b, c) in zip(suites["X"], suites["Y"], suites["XC"])
    a.op(); b.op(); c.op()
    isequal(a.check(), b.check()) && isequal(a.check(), c.check()) || error("output mismatch: $(a.id)")
end
println("outputs identical across roles for $(length(suites["X"])) probes")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p, n)
    op = p.op
    evals = max(1, 2^18 ÷ n)
    b = @benchmarkable $op() evals = evals samples = 120 seconds = 0.2
    for _ in 1:20
        op()
    end
    run(b)
end
sizeof_of(id) = parse(Int, match(r"/n(\d+)/", id).captures[1])
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb853-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds)
    println(io, "X: ", first_root, "\nY: ", second_root)
end
for r in ("X", "Y", "XC"), p in suites[r]          # compile and warm every specialization
    bench(p, sizeof_of(p.id))
end
rounds = parse(Int, get(ENV, "PB_ROUNDS", "15"))
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,median_ns,min_ns,memory,allocs")
    ids = eachindex(suites["X"])
    for round in 1:rounds
        for i in (isodd(round) ? ids : reverse(ids))
            order = isodd(round) ? ("X", "Y", "XC") : ("XC", "Y", "X")
            for (k, r) in enumerate(order)
                p = suites[r][i]
                t = bench(p, sizeof_of(p.id))
                m = median(t)
                println(io, p.id, ',', round, ',', k, ',', r, ',', m.time, ',', minimum(t).time, ',', m.memory, ',', m.allocs)
            end
        end
        flush(io)
        println("round ", round, "/", rounds); flush(stdout)
    end
end
