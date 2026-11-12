# Paired benchmark of the NormL1 value paths (dense weighted SIMD loop, mixed zero weights, small but ordinary
# values, the zero vector, unweighted, views, sparse, block and symmetric iterates, value_and_prox!) and of the
# recomputation of sums whose products round below floatmin, for Float32/Float64. Written for the performance
# gate of issue #373; the design and the summary are those of norml1_infinite_weight.jl:
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/norml1_value_underflow.jl
#   python3 benchmark/summarize_norml1_infinite_weight.py <dir>
#
# Role X is compiled twice (X and the independent A/A control XC). Each round times X, Y and XC in alternating
# order. Run it again with the roles reversed (summarized on its own) and in a fresh process. PB_CHECK_ONLY=1
# only compares the outputs; PB_FILTER selects cases by a regular expression on their ids; PB_SIZES overrides
# the sizes (comma separated). Probes named `underflow` are the recomputed sums, whose values a fix of #373
# changes: their outputs are printed, not required to agree.
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
    rng = Random.MersenneTwister(373 + n)
    e0 = exponent(floatmin(T))
    x = T(2) .* randn(rng, T, n)
    w = T(0.5) .+ rand(rng, T, n)                       # positive finite weights
    wzero = copy(w); wzero[1:8:end] .= zero(T)          # one zero weight in eight
    xsmall = ldexp.(x, e0 + 40)                         # small, but every product normal: the loop is kept
    xzero = zeros(T, n)
    wu = fill(floatmin(T), n)                           # the #373 data: every product rounds below floatmin
    xu = fill(T(ldexp(big(1.0), -9) + ldexp(big(1.0), -precision(T))), n)
    λ = T(0.3); γ = T(0.7)
    out = similar(x)
    probes = NamedTuple[]
    add(id, op, check) = push!(probes, (id = "$(T)/n$(n)/$id", op = op, check = check))
    val(f, y) = (() -> (out[1] = f(y); nothing), () -> f(y))
    f1, fz, f0, fu = PB.NormL1(λ, w), PB.NormL1(λ, wzero), PB.NormL1(λ), PB.NormL1(one(T), wu)
    add("dense/weighted", val(f1, x)...)
    add("dense/weighted_zero", val(fz, x)...)
    add("dense/weighted_small", val(f1, xsmall)...)
    add("dense/weighted_zero_vector", val(f1, xzero)...)
    add("dense/unweighted", val(f0, x)...)
    add("dense/view_weighted", val(f1, view(x, :))...)
    add("dense/underflow", val(fu, xu)...)
    add("dense/value_and_prox", () -> (PB.value_and_prox!(out, f1, x, γ); nothing), () -> (PB.value_and_prox!(out, f1, x, γ), copy(out)))
    add("dense/value_and_prox_zero", () -> (PB.value_and_prox!(out, fz, x, γ); nothing), () -> (PB.value_and_prox!(out, fz, x, γ), copy(out)))
    # stored-entry loops: 1/8 of the entries stored
    xsp = copy(x); xsp[mod.(1:n, 8) .!= 1] .= zero(T)
    sp = PB.SparseIterate(xsp); spout = copy(sp)
    add("sparse/weighted", val(f1, sp)...)
    add("sparse/unweighted", val(f0, sp)...)
    add("sparse/value_and_prox", () -> (PB.value_and_prox!(spout, f1, sp, γ); nothing), () -> (PB.value_and_prox!(spout, f1, sp, γ), Vector(spout)))
    add("sparse/underflow", val(fu, PB.SparseIterate(xu))...)
    groups = [collect(1:2:n), collect(2:2:n)]
    bl = PB.BlockIterate(copy(x), groups); blout = copy(bl)
    add("block/weighted", val(f1, bl)...)
    add("block/value_and_prox", () -> (PB.value_and_prox!(blout, f1, bl, γ); nothing), () -> (PB.value_and_prox!(blout, f1, bl, γ), Vector(blout)))
    # a symmetric iterate of the same number of matrix entries, one lower-triangle coordinate in eight stored
    p = isqrt(n)
    W = T(0.5) .+ rand(rng, T, p, p); W = (W + W') / 2
    S = PB.SymmetricSparseIterate(T, p)
    k = 0
    for c in 1:p, r in c:p
        (k += 1) % 8 == 1 && (S[r, c] = T(2) * randn(rng, T))
    end
    Sout = copy(S)
    fs = PB.NormL1(λ, W)
    add("symmetric/weighted", val(fs, S)...)
    add("symmetric/value_and_prox", () -> (PB.value_and_prox!(Sout, fs, S, γ); nothing), () -> (PB.value_and_prox!(Sout, fs, S, γ), Matrix(PB.matrixview(Sout))))
    Su = PB.SymmetricSparseIterate(T, p)
    for c in 1:p, r in c:p
        Su[r, c] = xu[1]
    end
    add("symmetric/underflow", val(PB.NormL1(one(T), fill(floatmin(T), p, p)), Su)...)
    probes
end
sizes = parse.(Int, split(get(ENV, "PB_SIZES", "100,1024,65536,100000"), ','))
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), T, n) for T in (Float32, Float64) for n in sizes]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["X"], :id) == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)
for (a, b, c) in zip(suites["X"], suites["Y"], suites["XC"])
    a.op(); b.op(); c.op()
    isequal(a.check(), c.check()) || error("A/A output mismatch: $(a.id)")
    if occursin("underflow", a.id)
        println("$(a.id): X = $(a.check()), Y = $(b.check())")
    else
        isequal(a.check(), b.check()) || error("output mismatch: $(a.id)")
    end
end
println("outputs identical across roles for the ordinary probes ($(length(suites["X"])) probes)")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p, n)
    op = p.op
    evals = max(1, 2^18 ÷ n)
    b = @benchmarkable $op() evals = evals samples = 120 seconds = 0.15
    for _ in 1:20
        op()
    end
    run(b)
end
sizeof_of(id) = parse(Int, match(r"/n(\d+)/", id).captures[1])
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb373-bench")
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
