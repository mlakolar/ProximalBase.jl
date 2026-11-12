# One process of the #834 dropzeros! campaign: load one source tree, then check or time every case.
# usage: julia worker.jl SOURCE ROLE OUTPUT.csv SEED MODE(check|time) [CASE_REGEX]
using BenchmarkTools, Random, Statistics, SHA, LinearAlgebra
BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("one Julia thread required")
const SOURCE, ROLE, OUTPUT = ARGS[1], ARGS[2], ARGS[3]
const SEED = parse(Int, ARGS[4])
const MODE = ARGS[5]
const FILTER = length(ARGS) >= 6 ? Regex(ARGS[6]) : r""
module Src end
Base.include(Src, joinpath(SOURCE, "src", "ProximalBase.jl"))
const PB = Src.ProximalBase
using SparseArrays: dropzeros!

# The worker writes the raw storage only to restore a state outside the timed region. Both trees
# must have the storage layout this relies on.
for S in (PB.SparseIterate, PB.SymmetricSparseIterate)
    (:nzval, :nzval2ind, :ind2nzval, :nnz) ⊆ fieldnames(S) || error("unexpected layout of $S")
end

const SIZES = (64, 1024, 16384)          # active coordinates: small, medium, large
const ZERO_FRACTIONS = (0.0, 0.1, 0.5, 0.9, 1.0)
const NPATTERNS = 32                     # cycled, so successive calls see different zero patterns
const BATCH_COORDS = 4096                # a sample runs max(1, 4096 ÷ s) calls on separate iterates

ncoords(s) = 8s                          # one coordinate in eight is active
function make_iterate(kind, ::Type{T}, q) where {T}
    kind === :sparse && return PB.SparseIterate(T, q)
    p = ceil(Int, (sqrt(8q + 1) - 1) / 2)
    PB.SymmetricSparseIterate(T, p)
end

struct Pattern{T}
    ids::Vector{Int}
    vals::Vector{T}
end

# s distinct coordinates in random order; exactly round(f s) slots hold ±0, the rest are nonzero
function make_pattern(rng, ::Type{T}, q, s, f) where {T}
    ids = randperm(rng, q)[1:s]
    vals = T[(rand(rng, Bool) ? 1 : -1) * (T(0.5) + rand(rng, T)) for _ = 1:s]
    for k in randperm(rng, s)[1:round(Int, f * s)]
        vals[k] = rand(rng, Bool) ? zero(T) : -zero(T)
    end
    Pattern(ids, vals)
end

# restore x to pattern P in O(nactive): unmap the live coordinates, then write the pattern
function restore!(x, P)
    @inbounds for t = 1:x.nnz
        x.ind2nzval[x.nzval2ind[t]] = 0
    end
    s = length(P.ids)
    @inbounds for t = 1:s
        x.nzval[t] = P.vals[t]
        x.nzval2ind[t] = P.ids[t]
        x.ind2nzval[P.ids[t]] = t
    end
    x.nnz = s
    nothing
end

# Op is :drop, :fill_drop or :reset, a type parameter so that the timed call dispatches statically
mutable struct Case{X, T, Op}
    id::String
    xs::Vector{X}
    patterns::Vector{Pattern{T}}
    next::Int
end

function setup!(c::Case)
    for x in c.xs
        restore!(x, c.patterns[c.next])
        c.next = c.next % length(c.patterns) + 1
    end
    nothing
end

opof(::Case{X, T, Op}) where {X, T, Op} = Op
function kernel!(c::Case{X, T, :drop}) where {X, T}
    for x in c.xs
        dropzeros!(x)
    end
    nothing
end
function kernel!(c::Case{X, T, :fill_drop}) where {X, T}
    for x in c.xs
        fill!(PB.coordinates(x), 0)
        dropzeros!(x)
    end
    nothing
end
kernel!(c::Case{X, T, :reset}) where {X, T} = setup!(c)    # the restoration alone, timed as a diagnostic

function build_cases()
    cases = Case[]
    for kind in (:sparse, :symmetric), T in (Float32, Float64), s in SIZES
        q = ncoords(s)
        K = max(1, BATCH_COORDS ÷ s)
        ops = [("zeros$(round(Int, 100f))", :drop, f) for f in ZERO_FRACTIONS]
        push!(ops, ("fill_drop", :fill_drop, 0.0), ("reset", :reset, 0.5))
        for (name, op, f) in ops
            id = "$kind/$T/s$s/$name"
            occursin(FILTER, id) || continue
            rng = Xoshiro(hash((SEED, id)))
            xs = [make_iterate(kind, T, q) for _ = 1:K]
            qq = PB.ncoordinates(xs[1])
            patterns = [make_pattern(rng, T, qq, s, f) for _ = 1:NPATTERNS]
            push!(cases, Case{eltype(xs), T, op}(id, xs, patterns, 1))
        end
    end
    cases
end

# the loop dropzeros! used on SymmetricSparseIterate before #834, as the reference of the live state
function reference(P::Pattern, q)
    nz, idx, pos = copy(P.vals), copy(P.ids), zeros(Int, q)
    for (t, i) in enumerate(idx); pos[i] = t; end
    n, i = length(nz), 1
    while i <= n
        if iszero(nz[i])
            pos[idx[i]] = 0
            if i != n
                nz[i] = nz[n]; pos[idx[n]] = i; idx[i] = idx[n]
            end
            n -= 1
        else
            i += 1
        end
    end
    (n, nz[1:n], idx[1:n], pos)
end

function check(c::Case)
    opof(c) === :reset && return true
    ok = true
    for rep = 1:2NPATTERNS ÷ length(c.xs)
        starts = [mod1(c.next + k - 1, NPATTERNS) for k = 1:length(c.xs)]
        setup!(c); kernel!(c)
        for (x, j) in zip(c.xs, starts)
            P = c.patterns[j]
            opof(c) === :fill_drop && (P = Pattern(P.ids, zero(P.vals)))
            n, v, ids, pos = reference(P, PB.ncoordinates(x))
            ok &= x.nnz == n && isequal(x.nzval[1:n], v) && x.nzval2ind[1:n] == ids && x.ind2nzval == pos
        end
    end
    ok
end

function measure(c::Case)
    b = @benchmarkable kernel!($c) setup=(setup!($c)) evals=1 samples=4000 seconds=0.25
    for _ = 1:5
        setup!(c); kernel!(c)
    end
    t = run(b)
    K = length(c.xs)
    # allocation of one warmed batch, measured directly as well
    setup!(c); kernel!(c); setup!(c)
    direct = @allocated kernel!(c)
    (median(t.times) / K, minimum(t.times) / K, mean(t.times) / K, quantile(t.times, 0.9) / K,
     t.allocs, t.memory, direct, length(t.times), K)
end

cases = build_cases()
open(OUTPUT, "w") do io
    println(io, "case,status,median_ns,min_ns,mean_ns,p90_ns,allocs,memory,direct_bytes,samples,batch")
    if MODE == "check"
        for c in cases
            println(io, c.id, ',', check(c) ? "pass" : "FAIL", ",,,,,,,,,", length(c.xs))
        end
    else
        foreach(c -> (setup!(c); kernel!(c)), cases)    # compile every specialization first
        for c in (isodd(SEED) ? cases : reverse(cases))
            r = measure(c)
            println(io, c.id, ",pass,", join(r, ','))
            flush(io)
        end
    end
end
