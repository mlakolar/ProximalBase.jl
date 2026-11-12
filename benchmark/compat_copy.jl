# Paired benchmark of the two-argument copyto! of SparseIterate and BlockIterate from the sources that
# compat.jl and its copyto! methods concern (dense arrays, the ordinary hot path; CSC matrices, their views,
# adjoints and transposes; transposed sparse vectors; orthogonal factors; CHOLMOD.Dense of the element type
# of the destination and of another one), Float32/Float64, 16 to 65536 entries.
# Written for the performance gate of the compat.jl simplification.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=21 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/compat_copy.jl
#
# The protocol and the raw.csv format are those of norml1_infinite_weight.jl and linear_shape.jl (A/A
# control XC, alternating order, outputs compared across roles before any timing); summarize with
# benchmark/summarize_norml1_infinite_weight.py DIR... and run it again with the roles reversed.
# As in linear_shape.jl, the roles share the source arrays; the destinations are iterates of each tree.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))
const Dense = isdefined(SparseArrays, :CHOLMOD) ? SparseArrays.CHOLMOD.Dense : nothing

# the sources of one size and element type, shared by the roles
function makedata(::Type{T}, n::Int) where {T}
    rng = Random.MersenneTwister(895 + n)
    m = isqrt(n)                                       # n is a square: the matrix cases are m × m
    O = T === Float64 ? Float32 : Float64              # another element type
    v = randn(rng, T, n); M = randn(rng, T, m, m)
    S = sprandn(rng, T, m, m, 0.125); sv = sprandn(rng, T, n, 0.125)
    D = Dense === nothing ? nothing :
        (vs = Dense(reshape(v, n, 1)), ms = Dense(M), vo = Dense(reshape(O.(v), n, 1)), mo = Dense(O.(M)))
    (; m, v, M, S, Sv = view(S, :, 1:m), Sa = S', St = transpose(S), sva = sv', Q = qr(M).Q, D)
end

function makeprobes(PB, ::Type{T}, n::Int, data) where {T}
    (; m, v, M, S, Sv, Sa, St, sva, Q, D) = data
    probes = NamedTuple[]
    add(id, op, check) = push!(probes, (id = "$(T)/n$(n)/$id", op = op, check = check))
    copycase(id, x, y) = add(id, () -> (copyto!(x, y); nothing), () -> (Array(x), collect(PB.activecoordinates(x))))
    xv = PB.SparseIterate(T, n); xm = PB.SparseIterate(T, m, m); xr = PB.SparseIterate(T, 1, n)
    xb = PB.BlockIterate(T, n, max(1, n ÷ 4))
    copycase("vector_from_dense", xv, v)
    copycase("matrix_from_dense", xm, M)
    copycase("block_from_dense", xb, v)
    copycase("matrix_from_csc", xm, S)
    copycase("vector_from_csc", xv, S)
    copycase("matrix_from_csc_view", xm, Sv)
    copycase("matrix_from_csc_adjoint", xm, Sa)
    copycase("matrix_from_csc_transpose", xm, St)
    copycase("row_from_sparsevec_adjoint", xr, sva)
    copycase("matrix_from_qr_q", xm, Q)
    if D !== nothing
        copycase("vector_from_cholmod_same_type", xv, D.vs)
        copycase("matrix_from_cholmod_same_type", xm, D.ms)
        copycase("block_from_cholmod_same_type", xb, D.vs)
        copycase("vector_from_cholmod_other_type", xv, D.vo)
        copycase("matrix_from_cholmod_other_type", xm, D.mo)
        copycase("block_from_cholmod_other_type", xb, D.vo)
    end
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
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-compat-copy-bench")
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
