# Paired benchmark of GroupNormL2 (issue #900): the compact block proximal operator for group sizes 2 to
# 1024, repeated sweeps over all groups, and the array prox!, value_and_prox! and value, Float32/Float64,
# contiguous (range) and noncontiguous (permuted index vector) layouts, with dense, partially zero and
# zero groups.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/groupnorml2.jl
#
# Each root contains a src/ folder that defines GroupNormL2 with the pure prox_block! (the frozen full tree
# legacy/full at eff21b6d7d33c622313122f46c6ee71693c101ee does). Role X is compiled twice (X and the
# independent A/A control XC); role Y is the one under test. Each round times X, Y and XC in alternating
# order; benchmark/summarize_norml1_infinite_weight.py DIR... reports the per-round ratios Y/X and XC/X.
# Run it again with the roles reversed (summarized on its own) and in a fresh process. Inputs and outputs
# are shared by the three roles, so that array placement does not differ between them. PB_CHECK_ONLY=1
# only compares the outputs: XC must equal X exactly, and Y must agree with X to a few rounding errors
# (the two implementations may round the fused value differently); PB_FILTER selects cases by id.
using BenchmarkTools, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

const TOTAL = 4096          # entries of the array cases
const SIZES = (2, 5, 10, 32, 128, 1024)

# Data shared by the roles: per (T, m, layout, pattern) the groups, the argument, an output and the
# compact trials of every group with their outputs.
function makedata(::Type{T}, m::Int, layout::Symbol, pattern::Symbol) where {T}
    rng = Random.MersenneTwister(900 + m + (layout === :perm) + 2 * (pattern === :sparse))
    K = TOTAL ÷ m
    p = K * m
    order = layout === :perm ? randperm(rng, p) : collect(1:p)
    groups = layout === :perm ? [order[(k-1)*m+1:k*m] for k in 1:K] : [(k-1)*m+1:k*m for k in 1:K]
    x = randn(rng, T, p)
    for (k, G) in enumerate(groups)
        if pattern === :sparse && k % 4 != 1
            x[G] .= 0                                # three groups in four are zero
        elseif k % 3 == 0
            x[G] .*= T(0.05)                         # below the threshold: shrunk to zero
        elseif pattern === :partial
            x[G[1:2:end]] .= 0                       # half of the entries are zero
        end
    end
    U = [x[G] for G in groups]
    (; groups, x, y = similar(x), U, outs = [similar(u) for u in U], λ = T(0.6), γ = T(0.8),
       w = T(0.5) .+ rand(rng, T, K))
end

function makeprobes(PB, data)
    probes = NamedTuple[]
    for ((T, m, layout, pattern), d) in data
        id = "$(T)/m$(m)/$(layout)/$(pattern)"
        f = PB.GroupNormL2(d.λ, d.groups, d.w)
        x, y, U, outs, γ = d.x, d.y, d.U, d.outs, d.γ
        K = length(U)
        add(name, op, check) = push!(probes, (id = "$id/$name", op = op, check = check, n = length(x)))
        if layout === :range                         # the compact kernel does not see the layout
            k1 = findfirst(u -> any(!iszero, u), U)
            u, out = U[k1], outs[k1]
            add("prox_block1", () -> (PB.prox_block!(out, f, u, k1, γ); nothing), () -> copy(out))
            kz = findfirst(u -> all(iszero, u), U)
            if kz !== nothing
                uz, oz = U[kz], outs[kz]
                add("prox_block_zero", () -> (PB.prox_block!(oz, f, uz, kz, γ); nothing), () -> copy(oz))
            end
            add("block_sweep", () -> begin
                for k in 1:K
                    @inbounds PB.prox_block!(outs[k], f, U[k], k, γ)
                end
                nothing
            end, () -> reduce(vcat, outs))
        end
        add("prox", () -> (PB.prox!(y, f, x, γ); nothing), () -> copy(y))
        add("value_and_prox", () -> (PB.value_and_prox!(y, f, x, γ); nothing), () -> (PB.value_and_prox!(y, f, x, γ), copy(y)))
        add("value", () -> (y[1] = f(x); nothing), () -> f(x))
    end
    probes
end

keys_ = [(T, m, layout, pattern) for T in (Float32, Float64) for m in SIZES
         for layout in (:range, :perm) for pattern in (:dense, :partial, :sparse)]
data = [(k, makedata(k...)) for k in keys_]
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
suites = Dict(r => makeprobes(getfield(mod, :ProximalBase), data) for (r, mod) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["X"], :id) == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)

agree(a::Tuple, b::Tuple) = all(agree(u, v) for (u, v) in zip(a, b))
agree(a::Number, b::Number) = isequal(a, b) || abs(a - b) <= 8eps(typeof(a)) * abs(a)
function agree(a, b)
    isequal(a, b) && return true
    T = eltype(a)
    scale = maximum(abs, a; init = zero(T))
    all(i -> abs(a[i] - b[i]) <= 8eps(T) * max(scale, abs(a[i])), eachindex(a, b))
end
for (a, b, c) in zip(suites["X"], suites["Y"], suites["XC"])
    a.op(); ra = a.check()
    c.op(); rc = c.check()
    b.op(); rb = b.check()
    isequal(ra, rc) || error("A/A output mismatch: $(a.id)")
    agree(ra, rb) || error("output mismatch beyond rounding: $(a.id)")
end
println("outputs agree across roles for $(length(suites["X"])) probes")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    evals = max(1, 2^16 ÷ p.n)
    b = @benchmarkable $op() evals = evals samples = 100 seconds = 0.15
    for _ in 1:20
        op()
    end
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb900-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds)
    println(io, "X: ", first_root, "\nY: ", second_root)
end
for r in ("X", "Y", "XC"), p in suites[r]          # compile and warm every specialization
    bench(p)
end
rounds = parse(Int, get(ENV, "PB_ROUNDS", "11"))
open(joinpath(outdir, "raw.csv"), "w") do io
    println(io, "case,round,order,role,median_ns,min_ns,memory,allocs")
    ids = eachindex(suites["X"])
    for round in 1:rounds
        for i in (isodd(round) ? ids : reverse(ids))
            order = isodd(round) ? ("X", "Y", "XC") : ("XC", "Y", "X")
            for (k, r) in enumerate(order)
                p = suites[r][i]
                t = bench(p)
                m = median(t)
                println(io, p.id, ',', round, ',', k, ',', r, ',', m.time, ',', minimum(t).time, ',', m.memory, ',', m.allocs)
            end
        end
        flush(io)
        println("round ", round, "/", rounds); flush(stdout)
    end
end
