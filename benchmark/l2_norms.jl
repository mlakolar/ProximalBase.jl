# Paired benchmark of GroupNormL2 and NormL2 (issue #900). GroupNormL2: the compact block proximal
# operator for group sizes 2 to 1024, repeated sweeps over all groups, and the array prox!,
# value_and_prox! and value, Float32/Float64, contiguous (range) and noncontiguous (permuted index
# vector) layouts, with dense, partially zero and zero groups. NormL2: the dense prox!, value_and_prox!
# and value at 1024 and 65536 entries, and the value of a 200 × 200 matrix SparseIterate and of a
# SymmetricSparseIterate from 400 random assignments (about 400 stored coordinates), on the ordinary path and on the recovery of a sum of
# squares that overflows (#808).
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=15 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/l2_norms.jl
#
# Each root contains a src/ folder that defines GroupNormL2 with the pure prox_block! and NormL2 (the
# frozen full tree legacy/full at eff21b6d7d33c622313122f46c6ee71693c101ee does). Role X is compiled twice (X and the
# independent A/A control XC); role Y is the one under test. Each round times X, Y and XC in alternating
# order; benchmark/summarize_norml1_infinite_weight.py DIR... reports the per-round ratios Y/X and XC/X.
# Run it again with the roles reversed (summarized on its own) and in a fresh process. Inputs and outputs
# are shared by the three roles, so that array placement does not differ between them (the iterates of
# the NormL2 value cases are each role's own type, built from the same entries). PB_CHECK_ONLY=1
# only compares the outputs: XC must equal X exactly, and Y must agree with X to a few rounding errors
# (the two implementations may round the fused value differently); PB_EXACT=1 with PB_CHECK_ONLY=1 requires
# Y to equal X bit for bit instead (`===` per array element and per scalar, so signed zeros and NaN bit
# patterns count, and XC must equal X the same way); PB_FILTER selects cases by id.
using BenchmarkTools, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

const TOTAL = 4096          # target entries of the array cases: (TOTAL ÷ m) groups of size m
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
       w = T(0.5) .+ rand(rng, T, K), g = randn(rng, T, p))
end

function makeprobes(PB, data)
    probes = NamedTuple[]
    for ((T, m, layout, pattern), d) in data
        id = "$(T)/m$(m)/$(layout)/$(pattern)"
        f = PB.GroupNormL2(d.λ, d.groups, d.w)
        x, y, U, outs, γ = d.x, d.y, d.U, d.outs, d.γ
        K = length(U)
        add(name, op, check) = push!(probes, (id = "$id/$name", op = op, check = check, n = length(x)))
        T = eltype(x)
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
        if pattern !== :partial
            # a block-coordinate sweep as a solver runs it: for every group, the trial u = x[G] - s g[G]
            # from the iterate, the pure prox_block! into scratch, and the write copyto!(blockview(x, k), out);
            # the iterate is restored from x0 by a whole-iterate copy before each sweep
            xb = PB.BlockIterate(copy(x), d.groups)
            x0 = copy(xb)
            g = d.g
            Uu, Ou = [similar(u) for u in U], [similar(u) for u in U]   # own scratch: U stays as the other probes use it
            add("block_update_sweep", () -> begin
                copyto!(xb, x0)
                for k in 1:K
                    G = PB.blockindices(xb, k)
                    u, out = Uu[k], Ou[k]
                    @inbounds for a in eachindex(u)
                        i = G[a]
                        u[a] = xb[i] - T(0.5) * g[i]
                    end
                    PB.prox_block!(out, f, u, k, γ)
                    copyto!(PB.blockview(xb, k), out)
                end
                nothing
            end, () -> Vector(xb))
        end
        add("prox", () -> (PB.prox!(y, f, x, γ); nothing), () -> copy(y))
        add("value_and_prox", () -> (PB.value_and_prox!(y, f, x, γ); nothing), () -> (PB.value_and_prox!(y, f, x, γ), copy(y)))
        add("value", () -> (y[1] = f(x); nothing), () -> f(x))
    end
    probes
end

function makel2probes(PB, ::Type{T}, dense) where {T}
    probes = NamedTuple[]
    f = PB.NormL2(T(0.3))
    for (n, x, y) in dense
        add(name, op, check) = push!(probes, (id = "NormL2/$(T)/n$(n)/$name", op = op, check = check, n = n))
        add("prox", () -> (PB.prox!(y, f, x, T(0.8)); nothing), () -> copy(y))
        add("value_and_prox", () -> (PB.value_and_prox!(y, f, x, T(0.8)); nothing), () -> (PB.value_and_prox!(y, f, x, T(0.8)), copy(y)))
        add("value", () -> (y[1] = f(x); nothing), () -> f(x))
    end
    p, nstored = 200, 400
    rng = Random.MersenneTwister(808)
    entries = [(rand(rng, 1:p), rand(rng, 1:p), randn(rng)) for _ in 1:nstored]
    huge = T(2)^(exponent(floatmax(T)) - 2)
    for (kind, scale, λ) in ((:ordinary, one(T), T(0.3)), (:recovery, huge, T(2)^-(exponent(floatmax(T)) ÷ 2)))
        g = PB.NormL2(λ)
        S = PB.SymmetricSparseIterate(T, p)
        M = PB.SparseIterate(T, p, p)
        for (r, c, v) in entries
            S[max(r, c), min(r, c)] = scale * T(v)
            M[r, c] = scale * T(v)
        end
        out = zeros(T, 1)
        push!(probes, (id = "NormL2/$(T)/symmetric/$kind/value", op = () -> (out[1] = g(S); nothing), check = () -> g(S), n = 2nstored))
        push!(probes, (id = "NormL2/$(T)/sparse_matrix/$kind/value", op = () -> (out[1] = g(M); nothing), check = () -> g(M), n = 2nstored))
    end
    probes
end
l2dense = Dict(T => [(n, T(2) .* randn(Random.MersenneTwister(n), T, n), zeros(T, n)) for n in (1024, 65536)] for T in (Float32, Float64))

keys_ = [(T, m, layout, pattern) for T in (Float32, Float64) for m in SIZES
         for layout in (:range, :perm) for pattern in (:dense, :partial, :sparse)]
data = [(k, makedata(k...)) for k in keys_]
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
suites = Dict(r => vcat(makeprobes(getfield(mod, :ProximalBase), data),
                        [makel2probes(getfield(mod, :ProximalBase), T, l2dense[T]) for T in (Float32, Float64)]...)
              for (r, mod) in roles)
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
        agree(ra, rb) || error("output mismatch beyond rounding: $(a.id)")
    end
end
println(exact ? "outputs identical bit for bit across roles for $(length(suites["X"])) probes" :
                "outputs agree across roles for $(length(suites["X"])) probes")
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
