# Paired benchmark of the LinearFunction oracles on arguments of matching shape (dense vectors and
# matrices, sparse CSC coefficients or arguments, SparseIterate, BlockIterate, SymmetricSparseIterate,
# Scaled, Float32/Float64, from 16 entries, where the O(1) shape check weighs most, up to 65536).
# Written for the performance gate of the LinearFunction shape contract, which replaced the dispatch
# prediction of `dot` by `size(c) == size(x)`.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=21 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/linear_shape.jl
#
# The protocol and the raw.csv format are those of norml1_infinite_weight.jl: role X is compiled twice
# (X and the independent A/A control XC), role Y is the one under test, each round times X, Y and XC in
# alternating order, and benchmark/summarize_norml1_infinite_weight.py DIR... reports the per-round
# ratios Y/X and XC/X. Run it again with the roles reversed (summarized on its own) and in a fresh
# process. PB_CHECK_ONLY=1 only verifies that the outputs agree across the roles; PB_FILTER selects
# cases by a regular expression on their ids. Every case has a valid shape, where the old and the new
# checks accept the same pairs and the kernels are unchanged, so any output mismatch is reported first.
#
# Unlike norml1_infinite_weight.jl, the roles share one set of input and output arrays (only the iterates,
# whose types belong to each tree, are built per role): these kernels stream through arrays that fit in
# L2, where the relative placement of x, c and the output (4K aliasing) moves a role by up to ±40% with
# identical machine code, which separate arrays per role would attribute to the source.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

# the arrays of one size and element type, shared by the roles
function makedata(::Type{T}, n::Int) where {T}
    rng = Random.MersenneTwister(894 + n)
    m = isqrt(n)                                       # n is a square: the matrix cases are m × m
    c = randn(rng, T, n); x = randn(rng, T, n)
    C = randn(rng, T, m, m); X = randn(rng, T, m, m)
    SC = sprandn(rng, T, m, m, 0.125); SX = sprandn(rng, T, m, m, 0.125)
    xsp = copy(x); xsp[mod.(1:n, 8) .!= 1] .= zero(T)
    sym = [(i, j, randn(rng, T)) for j in 1:m for i in j:m if mod(i + j, 8) == 0]
    (; m, c, x, out = similar(x), C, X, Out = similar(X), SC, SX, SCa = SC', Cs = C + C', xsp, sym)
end

function makeprobes(PB, ::Type{T}, n::Int, D) where {T}
    (; m, c, x, out, C, X, Out, SC, SX, SCa, Cs, xsp, sym) = D
    γ = T(0.7); d = T(0.25)
    probes = NamedTuple[]
    add(id, op, check) = push!(probes, (id = "$(T)/n$(n)/$id", op = op, check = check))
    # dense vectors
    f = PB.LinearFunction(c, d); s = T(3) * f
    add("vector/value", () -> (out[1] = f(x); nothing), () -> out[1])
    add("vector/gradient!", () -> (PB.gradient!(out, f, x); nothing), () -> copy(out))
    add("vector/value_and_gradient!", () -> (out[1] = PB.value_and_gradient!(out, f, x); nothing), () -> copy(out))
    add("vector/prox!", () -> (PB.prox!(out, f, x, γ); nothing), () -> copy(out))
    add("vector/value_and_prox!", () -> (out[1] = PB.value_and_prox!(out, f, x, γ); nothing), () -> copy(out))
    add("vector/scaled_value", () -> (out[1] = s(x); nothing), () -> out[1])
    add("vector/scaled_prox!", () -> (PB.prox!(out, s, x, γ); nothing), () -> copy(out))
    # dense matrices
    fm = PB.LinearFunction(C, d)
    add("matrix/value", () -> (Out[1] = fm(X); nothing), () -> Out[1])
    add("matrix/prox!", () -> (PB.prox!(Out, fm, X, γ); nothing), () -> copy(Out))
    add("matrix/value_and_prox!", () -> (Out[1] = PB.value_and_prox!(Out, fm, X, γ); nothing), () -> copy(Out))
    add("matrix/value_and_gradient!", () -> (Out[1] = PB.value_and_gradient!(Out, fm, X); nothing), () -> copy(Out))
    # sparse coefficients or arguments (one entry in eight stored): the sparse dot methods
    fs = PB.LinearFunction(SC, d); fa = PB.LinearFunction(SCa, d)
    add("sparse_c_dense_x/value", () -> (Out[1] = fs(X); nothing), () -> Out[1])
    add("dense_c_sparse_x/value", () -> (Out[1] = fm(SX); nothing), () -> Out[1])
    add("sparse_c_sparse_x/value", () -> (Out[1] = fs(SX); nothing), () -> Out[1])
    add("sparse_c_dense_x/prox!", () -> (PB.prox!(Out, fs, X, γ); nothing), () -> copy(Out))
    add("adjoint_sparse_c_sparse_x/value", () -> (Out[1] = fa(SX); nothing), () -> Out[1])
    # iterates, built per role: the stored-entry dot of a SparseIterate, a BlockIterate, and a symmetric iterate
    sp = PB.SparseIterate(xsp); spout = copy(sp)
    add("sparse_iterate/value", () -> (out[1] = f(sp); nothing), () -> out[1])
    add("sparse_iterate/prox!", () -> (PB.prox!(out, f, sp, γ); nothing), () -> copy(out))
    add("sparse_iterate/prox!_iterate_output", () -> (PB.prox!(spout, f, sp, γ); nothing), () -> Vector(spout))
    bl = PB.BlockIterate(copy(x), [collect(1:2:n), collect(2:2:n)])
    add("block_iterate/value", () -> (out[1] = f(bl); nothing), () -> out[1])
    si = PB.SymmetricSparseIterate(T, m)
    for (i, j, v) in sym
        si[i, j] = v
    end
    symout = similar(si)
    fsym = PB.LinearFunction(Cs, d)
    add("symmetric/value", () -> (Out[1] = fsym(si); nothing), () -> Out[1])
    add("symmetric/prox!", () -> (PB.prox!(symout, fsym, si, γ); nothing), () -> Matrix(symout))
    add("symmetric/value_and_gradient!", () -> (Out[1] = PB.value_and_gradient!(symout, fsym, si); nothing), () -> (Out[1], Matrix(symout)))
    add("symmetric/prox!_dense_output", () -> (PB.prox!(Out, fsym, si, γ); nothing), () -> copy(Out))
    probes
end
sizes = (16, 1024, 65536)
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
data = Dict((T, n) => makedata(T, n) for T in (Float32, Float64) for n in sizes)
suites = Dict(r => vcat([makeprobes(getfield(m, :ProximalBase), T, n, data[(T, n)]) for T in (Float32, Float64) for n in sizes]...) for (r, m) in roles)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
for r in keys(suites)
    filter!(p -> occursin(filter_re, p.id), suites[r])
end
@assert getfield.(suites["X"], :id) == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)
# the roles share their output arrays, so each result is read right after its own call
for (a, b, c) in zip(suites["X"], suites["Y"], suites["XC"])
    ra = (a.op(); a.check()); rb = (b.op(); b.check()); rc = (c.op(); c.check())
    isequal(ra, rb) && isequal(ra, rc) || error("output mismatch: $(a.id)")
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
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-linear-shape-bench")
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
