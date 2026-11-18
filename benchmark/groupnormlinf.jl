# Paired benchmark of GroupNormLinf (issue #956): the array prox!, the in-place prox!(x, f, x, γ), value_and_prox! and
# the value, and repeated sweeps of the compact prox_block! over all groups (an output that is not the input, and in
# place), Float32/Float64, group sizes 4, 32, 512 and 4096 over 4096 entries (1024, 128, 8 and 1 groups), contiguous
# (range) and noncontiguous (permuted index vector) layouts. The weights give group k a radius that is the fraction
# FRACTIONS[mod1(k, 4)] of its ℓ1 norm, 0.1, 0.4, 0.8 and 1.2 in turn. With groups of 4, 32 and 512 entries (1024, 128 and 8
# groups) three groups in four are clipped and one is zero (fraction 1.2); with 4096 entries there is one group, with
# fraction 0.1, which is clipped, and no group is zero. The in-place cases (prox_inplace, block_sweep_inplace) destroy
# their argument: it is restored in the BenchmarkTools setup, outside the timing, and each timed sample is one call
# (evals = 1). The roles are used through exported names, so that they can be any trees that define GroupNormLinf with
# these methods.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/groupnormlinf.jl
#
# Each root contains a src/ folder that defines GroupNormLinf (the frozen full tree legacy/full at
# eff21b6d7d33c622313122f46c6ee71693c101ee does, and so does a tree with this restoration). Role X is compiled twice (X and
# the independent A/A control XC); role Y is the one under test. Each round times X, Y and XC in alternating order;
# benchmark/summarize_norml1_infinite_weight.py DIR... reports the per-round ratios Y/X and XC/X, and reads the raw.csv and
# provenance.txt of this script as it does for l2_norms.jl. Run it again with the roles reversed (summarized on its own) and
# in a fresh process. Inputs and outputs are shared by the three roles, so that array placement does not differ between them.
# PB_CHECK_ONLY=1 compares the outputs without timing: XC must be `isequal` to X, and Y must agree with X to a few rounding errors of
# the level, 8 (m + K) eps(T) times the largest entry for groups of m entries and K groups (the two implementations may
# round the level and the fused value differently); PB_EXACT=1 with PB_CHECK_ONLY=1 requires Y to equal X bit for bit
# instead (`===` per array element and per scalar, so signed zeros and NaN bit patterns count, and XC must equal X the same
# way); PB_FILTER selects cases by id (a regular expression).
using BenchmarkTools, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

const TOTAL = 4096          # entries of the array cases: (TOTAL ÷ m) groups of size m
const SIZES = (4, 32, 512, 4096)
const FRACTIONS = (0.1, 0.4, 0.8, 1.2)

# Data shared by the roles: per (T, m, layout) the groups, the argument and its restore copy, an output, and the compact
# trials of each group with their outputs and their working copies.
function makedata(::Type{T}, m::Int, layout::Symbol) where {T}
    rng = Random.MersenneTwister(956 + m + (layout === :perm))
    K = TOTAL ÷ m
    p = K * m
    order = layout === :perm ? randperm(rng, p) : collect(1:p)
    groups = layout === :perm ? [order[(k-1)*m+1:k*m] for k in 1:K] : [(k-1)*m+1:k*m for k in 1:K]
    x = randn(rng, T, p)
    λ0, γ = T(0.6), T(0.8)
    w = T[T(FRACTIONS[mod1(k, length(FRACTIONS))] * Float64(sum(abs, x[G])) / Float64(γ * λ0)) for (k, G) in enumerate(groups)]
    U = [x[G] for G in groups]
    (; groups, x, x0 = copy(x), xi = copy(x), y = similar(x), U, outs = [similar(u) for u in U],
       W = [similar(u) for u in U], λ0, γ, w)
end

function makeprobes(PB, data)
    probes = NamedTuple[]
    for ((T, m, layout), d) in data
        id = "GroupNormLinf/$(T)/m$(m)/$(layout)"
        f = PB.GroupNormLinf(d.λ0, d.groups, d.w)
        x, x0, xi, y, U, outs, W, γ = d.x, d.x0, d.xi, d.y, d.U, d.outs, d.W, d.γ
        K = length(U)
        # setup restores the argument of a destructive op outside the timing; check restores it too, then runs op
        add(name, op, check; setup = nothing) =
            push!(probes, (id = "$id/$name", op = op, check = check, setup = setup, n = length(x), m = m, K = K))
        add("prox", () -> (PB.prox!(y, f, x, γ); nothing), () -> copy(y))
        restore_x() = (copyto!(xi, x0); nothing)
        prox_inplace() = (PB.prox!(xi, f, xi, γ); nothing)
        add("prox_inplace", prox_inplace, () -> (restore_x(); prox_inplace(); copy(xi)); setup = restore_x)
        add("value_and_prox", () -> (PB.value_and_prox!(y, f, x, γ); nothing),
            () -> (PB.value_and_prox!(y, f, x, γ), copy(y)))
        add("value", () -> (y[1] = f(x); nothing), () -> f(x))
        add("block_sweep", () -> begin
            for k in 1:K
                @inbounds PB.prox_block!(outs[k], f, U[k], k, γ)
            end
            nothing
        end, () -> reduce(vcat, outs))
        restore_blocks() = (for k in 1:K; @inbounds copyto!(W[k], U[k]); end; nothing)
        sweep_inplace() = (for k in 1:K; @inbounds PB.prox_block!(W[k], f, W[k], k, γ); end; nothing)
        add("block_sweep_inplace", sweep_inplace, () -> (restore_blocks(); sweep_inplace(); reduce(vcat, W));
            setup = restore_blocks)
    end
    probes
end

keys_ = [(T, m, layout) for T in (Float32, Float64) for m in SIZES for layout in (:range, :perm)]
data = [(k, makedata(k...)) for k in keys_]
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
suites = Dict(r => makeprobes(getfield(mod, :ProximalBase), data) for (r, mod) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["X"], :id) == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)

# agreement to a few rounding errors of the level: 8 (m + K) eps(T) times the largest entry (arrays), and the same relative to
# the value (scalars, whose sum of K terms each has the error of its level)
agree(a::Tuple, b::Tuple, tf) = all(agree(u, v, tf) for (u, v) in zip(a, b))
agree(a::Number, b::Number, tf) = isequal(a, b) || abs(a - b) <= tf * eps(typeof(a)) * abs(a)
function agree(a, b, tf)
    isequal(a, b) && return true
    T = eltype(a)
    scale = maximum(abs, a; init = zero(T))
    all(i -> abs(a[i] - b[i]) <= tf * eps(T) * max(scale, abs(a[i])), eachindex(a, b))
end
# bit-for-bit equality: tuples elementwise, arrays by size and `===` per element, scalars by `===`
exact_equal(a::Tuple, b::Tuple) = length(a) == length(b) && all(exact_equal(u, v) for (u, v) in zip(a, b))
exact_equal(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(i -> a[i] === b[i], eachindex(a, b))
exact_equal(a, b) = a === b
exact = get(ENV, "PB_EXACT", "0") == "1"
for (a, b, c) in zip(suites["X"], suites["Y"], suites["XC"])
    a.op(); ra = a.check()
    c.op(); rc = c.check()
    b.op(); rb = b.check()
    (exact ? exact_equal(ra, rc) : isequal(ra, rc)) || error("A/A output mismatch: $(a.id)")
    if exact
        exact_equal(ra, rb) || error("output differs bit for bit: $(a.id)")
    else
        agree(ra, rb, 8 * (a.m + a.K)) || error("output mismatch beyond rounding: $(a.id)")
    end
end
println(exact ? "outputs identical bit for bit across roles for $(length(suites["X"])) probes" :
                "outputs agree across roles for $(length(suites["X"])) probes")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

# A non-destructive op is timed with evals = max(1, 2^16 ÷ n) calls per sample and 100 samples. A destructive op (one with
# a setup) is restored by the setup before each timed call, outside the timing, with evals = 1, and takes as many samples as
# the other cases take timed calls (100 evals), within the same time budget.
function bench(p)
    op = p.op
    evals = max(1, 2^16 ÷ p.n)
    if p.setup === nothing
        b = @benchmarkable $op() evals = evals samples = 100 seconds = 0.15
        for _ in 1:20
            op()
        end
    else
        setup = p.setup
        b = @benchmarkable $op() setup = ($setup()) evals = 1 samples = 100 * evals seconds = 0.15
        for _ in 1:20
            setup()
            op()
        end
    end
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb956-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds)
    println(io, "X: ", first_root, "\nY: ", second_root)
end
for r in ("X", "Y", "XC"), p in suites[r]          # compile and warm each probe of the three roles
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
