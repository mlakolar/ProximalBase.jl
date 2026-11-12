# Paired benchmark of the LinearFunction value at a matrix SparseIterate (#810), which sums the stored nonzero
# entries of the iterate in the storage order and reads the coefficients only at them, where the earlier value
# was the generic dot over all m × m positions. Float32/Float64, m = 16, 128, 1024.
#
#   PB_FIRST=<root of role X> PB_SECOND=<root of role Y> PB_OUTPUT=<dir> PB_ROUNDS=21 \
#     julia --threads=1 --check-bounds=no --project=benchmark benchmark/linear_sparse_value.jl
#
# The protocol and the raw.csv format are those of linear_shape.jl and norml1_infinite_weight.jl: role X is
# compiled twice (X and the independent A/A control XC), role Y is the one under test, each round times X, Y and
# XC in alternating order, and benchmark/summarize_norml1_infinite_weight.py DIR... reports the per-round ratios
# Y/X and XC/X. Run it again with the roles reversed (summarized on its own) and in a fresh process.
# PB_FILTER selects cases by a regular expression on their ids, and only the selected cases are built.
# PB_CHECK_ONLY=1 checks the outputs and exits.
#
# The output check differs from linear_shape.jl, where the kernels are unchanged: the new value sums in the storage
# order of the iterate, so the roles agree to the accuracy of the sum, not bit for bit. A value of finite data is
# compared across the roles with isapprox (rtol = 64 eps(T), or atol = 8 eps(T) sqrt(n) Σ|cᵢxᵢ|, n the number of
# stored entries, which covers the rounding of a sum of many terms), a value that is not finite with isequal. The
# exceptional cells (an overflowing dot with finite coefficients) are compared with a BigFloat value of each role's
# own output instead, and a difference across the roles there is reported, not an error.
#
# As in linear_shape.jl the roles share their input and output arrays (only the iterates, whose types belong to
# each tree, are built per role): these kernels stream through arrays that fit in L2, where the relative placement
# of x, c and the output moves a role by up to ±40% with identical machine code. A sample is repeated (evals) until
# it takes about a microsecond or more. The cells that run code the change does not touch (the dense matrix value,
# the vector SparseIterate value, the symmetric value and the matrix-iterate gradient! and prox!) are controls.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
first_root = abspath(ENV["PB_FIRST"]); second_root = abspath(ENV["PB_SECOND"])
module First end
module Second end
module FirstControl end
Base.include(First, joinpath(first_root, "src", "ProximalBase.jl"))
Base.include(Second, joinpath(second_root, "src", "ProximalBase.jl"))
Base.include(FirstControl, joinpath(first_root, "src", "ProximalBase.jl"))

# the entries of an iterate as linear indices and values, in the order they are activated, and the tolerance of the
# output check; a zero value is a stored zero. Shared by the roles.
struct Entries{T}
    idx::Vector{Int}
    val::Vector{T}
    atol::Float64
end

# the arrays of one element type and size m, shared by the roles; the entries are made when a cell asks for them
mutable struct Data{T}
    m::Int
    C::Matrix{T}                 # coefficients
    X::Matrix{T}                 # a dense argument
    Out::Matrix{T}               # a dense output
    ones::Matrix{T}              # the coefficients of the exceptional cells
    dg::Vector{T}                # the diagonal of a Diagonal coefficient array
    cv::Vector{T}                # a vector of coefficients, for the vector control
    out::Vector{T}               # a sink for a value
    sym::Vector{Tuple{Int, Int, T}}
    Cs::Matrix{T}
    rng::Random.MersenneTwister
    entries::Dict{String, Entries{T}}
end
function makedata(::Type{T}, m::Int) where {T}
    rng = Random.MersenneTwister(810 + m + (T === Float32 ? 1 : 0))
    C = randn(rng, T, m, m)
    sym = [(i, j, randn(rng, T)) for j in 1:m for i in j:m if mod(i + j, 8) == 0]
    Data{T}(m, C, randn(rng, T, m, m), similar(C), ones(T, m, m), randn(rng, T, m), randn(rng, T, m * m), zeros(T, 1),
            sym, C + C', rng, Dict{String, Entries{T}}())
end

# the tolerance 8 eps(T) sqrt(n) Σ|cᵢ xᵢ| of the sum of n stored entries
sumtol(::Type{T}, c, idx, val) where {T} =
    8 * Float64(eps(T)) * sqrt(max(length(idx), 1)) * sum(abs(Float64(c[i]) * Float64(v)) for (i, v) in zip(idx, val); init = 1.0)

# k entries at random positions (the first k of a random order, sorted), the values random of both signs; a
# fraction z of them zero; in column-major order, or shuffled
function entries!(D::Data{T}, key, k, z, shuffled) where {T}
    get!(D.entries, key) do
        rng = D.rng; n = D.m^2
        idx = k >= n ? collect(1:n) : sort!(randperm(rng, n)[1:k])
        val = randn(rng, T, k)
        z > 0 && (val[randperm(rng, k)[1:round(Int, z * k)]] .= zero(T))
        if shuffled
            p = randperm(rng, k)
            idx, val = idx[p], val[p]
        end
        Entries{T}(idx, val, sumtol(T, D.C, idx, val))
    end
end
# an iterate of the role's package with the entries, in order; a zero is stored as an active zero
function iterate_of(PB, ::Type{T}, m, e::Entries) where {T}
    x = PB.SparseIterate(T, m, m)
    for (i, v) in zip(e.idx, e.val)
        iszero(v) ? PB.activate!(x, i) : (x[i] = v)
    end
    x
end
# the overflowing dot, with an odd number k of entries (k - 1 if k is even): the first (k + 1) / 2 in column-major
# order are +B and the others -B, B = floatmax / 16, so that the partial sums overflow and the value is B
function exceptional!(D::Data{T}, key, k) where {T}
    get!(D.entries, key) do
        n = D.m^2
        idx = k >= n ? collect(1:n) : sort!(randperm(D.rng, n)[1:k])
        isodd(length(idx)) || pop!(idx)
        k = length(idx)
        B = floatmax(T) / 16
        val = [i <= (k + 1) ÷ 2 ? B : -B for i in 1:k]
        Entries{T}(idx, val, 0.0)
    end
end

# n = m², the size that the case ids carry. A probe is (id, work, build, atol, ref): work counts the entries
# that a call visits, build makes the role's operation and the reading of its result, atol is the tolerance of the
# check, and ref is a BigFloat value with its bound for an exceptional cell (nothing otherwise).
function makeprobes(PB, ::Type{T}, m::Int, D::Data{T}, wanted) where {T}
    n = m^2
    (; C, X, Out, ones, dg, cv, out, sym, Cs) = D
    γ = T(0.7); d = T(0.25)
    probes = NamedTuple[]
    add(id, work, build, atol = 0.0, ref = nothing) = begin
        full = "$(T)/n$(n)/$id"
        wanted(full) && push!(probes, (id = full, work = work, build = build, atol = atol, ref = ref))
    end
    value_probe(id, f, mk_x, e, work) = add(id, work, () -> (x = mk_x(); (() -> (out[1] = f(x); nothing), () -> out[1])), e.atol)
    # NB: the thunks run after this function returns, so each captures a variable that is assigned once
    # the BigFloat value and its R1 bound 4 k eps(T) (Σ|cᵢxᵢ| + |d|), over the k stored entries
    big_ref(e, c, dd) = (sum(big(c[i]) * big(v) for (i, v) in zip(e.idx, e.val)) + big(dd),
        4 * length(e.idx) * big(eps(T)) * (sum(abs(big(c[i]) * big(v)) for (i, v) in zip(e.idx, e.val)) + abs(big(dd))))

    # the cells of the iterate values: label, number of stored entries, fraction of them stored as zeros, shuffled order
    k2 = 2; k1 = max(2, round(Int, 0.01 * n)); k10 = round(Int, 0.1 * n); k50 = round(Int, 0.5 * n)
    f = PB.LinearFunction(C, d)
    cases = Tuple{String, Int, Float64, Bool}[("d2_seq", k2, 0.0, false), ("d1pct_seq", k1, 0.0, false),
        ("d10pct_seq", k10, 0.0, false), ("d10pct_shuf", k10, 0.0, true), ("d50pct_seq", k50, 0.0, false),
        ("d100pct_seq", n, 0.0, false), ("d100pct_shuf", n, 0.0, true),
        ("d100pct_z50_seq", n, 0.5, false), ("d100pct_z90_seq", n, 0.9, false)]
    m == 128 && push!(cases, ("d100pct_z50_shuf", n, 0.5, true), ("d1pct_shuf", k1, 0.0, true))
    for (label, k, z, shuf) in cases
        wanted("$(T)/n$(n)/sparse_value/$label") || continue
        ev = entries!(D, label, k, z, shuf)
        value_probe("sparse_value/$label", f, () -> iterate_of(PB, T, m, ev), ev, length(ev.idx))
    end
    # an overflowing dot with finite coefficients: the base recovers too
    if m >= 128
        fo = PB.LinearFunction(ones, d)
        for (label, k) in (("d1pct", max(3, round(Int, 0.01 * n))), ("d100pct", n))
            id = "exceptional/$label"
            wanted("$(T)/n$(n)/$id") || continue
            eo = exceptional!(D, "exc_" * label, k)
            ref = big_ref(eo, ones, d)
            add(id, length(eo.idx), () -> (x = iterate_of(PB, T, m, eo); (() -> (out[1] = fo(x); nothing), () -> out[1])), 0.0, ref)
        end
    end
    # a Diagonal coefficient array: the diagonal of the iterate is read
    fd = PB.LinearFunction(Diagonal(dg), d)
    for (label, key) in (("d2", "d2_seq"), ("d100pct", "d100pct_seq"))
        wanted("$(T)/n$(n)/diagonal/$label") || continue
        ed = key == "d2_seq" ? diagonal_entries!(D) : entries!(D, key, n, 0.0, false)
        atol = key == "d2_seq" ? ed.atol : 8 * Float64(eps(T)) * sqrt(m) * (1.0 + sum(abs, dg) * 4)
        add("diagonal/$label", m, () -> (x = iterate_of(PB, T, m, ed); (() -> (out[1] = fd(x); nothing), () -> out[1])), atol)
    end
    # the other oracles that reach the value, at m = 128, 10% stored
    if m == 128
        e10 = entries!(D, "d10pct_seq", k10, 0.0, false)
        atol = 8 * Float64(eps(T)) * sqrt(n) * 6 * (1.0 + sum(abs, C))
        add("value_and_gradient!/dense_output", length(e10.idx),
            () -> (x = iterate_of(PB, T, m, e10); (() -> (out[1] = PB.value_and_gradient!(Out, f, x); nothing), () -> (out[1], copy(Out)))), atol)
        add("value_and_prox!/dense_output", length(e10.idx),
            () -> (x = iterate_of(PB, T, m, e10); (() -> (out[1] = PB.value_and_prox!(Out, f, x, γ); nothing), () -> (out[1], copy(Out)))), atol)
        add("value_and_prox!/iterate_output", n,
            () -> (x = iterate_of(PB, T, m, e10); y = PB.SparseIterate(T, m, m);
                   (() -> (out[1] = PB.value_and_prox!(y, f, x, γ); nothing), () -> (out[1], Matrix(y)))), atol)
        # controls: the code of these paths is unchanged
        add("control/gradient!_iterate_input", length(e10.idx),
            () -> (x = iterate_of(PB, T, m, e10); (() -> (PB.gradient!(Out, f, x); nothing), () -> copy(Out))))
        add("control/prox!_iterate_input", length(e10.idx),
            () -> (x = iterate_of(PB, T, m, e10); (() -> (PB.prox!(Out, f, x, γ); nothing), () -> copy(Out))))
    end
    # controls: dense matrix value, vector SparseIterate value, symmetric value
    add("control/dense_value", n, () -> (() -> (out[1] = f(X); nothing), () -> out[1]), 8 * Float64(eps(T)) * sqrt(n) * (1.0 + sum(abs, C) * 4))
    if m == 128 || m == 1024
        xv = randn(D.rng, T, n); xv[mod.(1:n, 10) .!= 1] .= zero(T)
        fv = PB.LinearFunction(cv, d)
        atolv = 8 * Float64(eps(T)) * sqrt(n) * (1.0 + sum(abs, cv) * 4)
        add("control/vector_iterate_value", n ÷ 10, () -> (x = PB.SparseIterate(xv); (() -> (out[1] = fv(x); nothing), () -> out[1])), atolv)
        fsym = PB.LinearFunction(Cs, d)
        add("control/symmetric_value", length(sym),
            () -> (si = PB.SymmetricSparseIterate(T, m); for (i, j, v) in sym; si[i, j] = v; end;
                   (() -> (out[1] = fsym(si); nothing), () -> out[1])), atolv)
    end
    probes
end
# the two entries (1, 1) and (2, 2) of the diagonal
function diagonal_entries!(D::Data{T}) where {T}
    get!(D.entries, "diag2") do
        idx = [1, D.m + 2]; val = T[1.5, -0.75]
        Entries{T}(idx, val, 8 * Float64(eps(T)) * 4 * (1.0 + sum(abs, D.dg[1:2])))
    end
end

sizes = (16, 128, 1024)
roles = Dict("X" => First, "Y" => Second, "XC" => FirstControl)
filter_re = Regex(get(ENV, "PB_FILTER", ""))
wanted(id) = occursin(filter_re, id)
data = Dict((T, m) => makedata(T, m) for T in (Float32, Float64) for m in sizes)
# the cells of each role, built in the same order: id, work, the operation and the reading of its result
function build_suite(role)
    PB = getfield(roles[role], :ProximalBase)
    suite = NamedTuple[]
    for T in (Float32, Float64), m in sizes
        for q in makeprobes(PB, T, m, data[(T, m)], wanted)
            op, check = q.build()
            push!(suite, (id = q.id, work = q.work, op = op, check = check, atol = q.atol, ref = q.ref))
        end
    end
    suite
end
suites = Dict(r => build_suite(r) for r in keys(roles))
@assert getfield.(suites["X"], :id) == getfield.(suites["Y"], :id) == getfield.(suites["XC"], :id)
isempty(suites["X"]) && error("PB_FILTER selects no case")

# the value of a role is a BigFloat value within the R1 bound, or a result of the given kind
function agree(a, b, atol, T)
    if a isa Number && b isa Number
        return isequal(a, b) || (isfinite(a) && isfinite(b) && isapprox(a, b; rtol = 64 * eps(T), atol = atol))
    elseif a isa Tuple
        return length(a) == length(b) && all(agree(u, v, atol, T) for (u, v) in zip(a, b))
    else
        return size(a) == size(b) && all(agree(u, v, atol, T) for (u, v) in zip(a, b))
    end
end
elT(id) = startswith(id, "Float32") ? Float32 : Float64
# the roles share their output arrays, so each result is read right after its own call
let ndiff = 0
    for (a, b, c) in zip(suites["X"], suites["Y"], suites["XC"])
        T = elT(a.id)
        ra = (a.op(); a.check()); rb = (b.op(); b.check()); rc = (c.op(); c.check())
        if a.ref === nothing
            agree(ra, rb, a.atol, T) && agree(ra, rc, a.atol, T) || error("output mismatch: $(a.id): X $(ra) Y $(rb) XC $(rc)")
        else
            # an overflowing dot: each role against the BigFloat value, within the R1 bound
            ref, bound = a.ref
            for (r, v) in (("X", ra), ("Y", rb), ("XC", rc))
                isfinite(v) && abs(big(v) - ref) <= bound ||
                    error("exceptional value of $r out of the bound: $(a.id): $v against $(Float64(ref))")
            end
            isequal(ra, rb) || (ndiff += 1; println("exceptional cell $(a.id): X $(ra) Y $(rb), the reference $(Float64(ref))"))
        end
    end
    println("outputs agree across the roles for $(length(suites["X"])) probes ($(ndiff) exceptional cells differ across the roles)")
end
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function bench(p)
    op = p.op
    evals = max(1, 2^16 ÷ max(p.work, 8))
    b = @benchmarkable $op() evals = evals samples = 120 seconds = 0.2
    for _ in 1:20
        op()
    end
    run(b)
end
outdir = get(ENV, "PB_OUTPUT", "/tmp/pb-linear-sparse-value-bench")
mkpath(outdir)
open(joinpath(outdir, "provenance.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nJulia: ", VERSION, "\nCPU: ", Sys.CPU_NAME, "\nThreads: ", Threads.nthreads())
    println(io, "Bounds option: ", Base.JLOptions().check_bounds)
    println(io, "X: ", first_root, "\nY: ", second_root)
end
for r in ("X", "Y", "XC"), p in suites[r]          # compile and warm every specialization
    bench(p)
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
                t = bench(p)
                m = median(t)
                println(io, p.id, ',', round, ',', k, ',', r, ',', m.time, ',', minimum(t).time, ',', m.memory, ',', m.allocs)
            end
        end
        flush(io)
        println("round ", round, "/", rounds); flush(stdout)
    end
end
