# One process of the #834 dropzeros! campaign: load one source tree, then check or time every case.
# usage: julia worker.jl SOURCE ROLE OUTPUT.csv SEED MODE(check|time|fingerprint) [CASE_REGEX]
# Every case's generator is seeded from SHA-256 of the campaign seed and the case ID, never from
# Julia's hash, and every output row carries a SHA-256 fingerprint of the case's fixtures (coordinate
# IDs and the exact bits of the values), which run.py requires to match across the roles of a round.
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

# The worker writes the raw storage only to restore a state, outside the timed region except in the
# reset cases, which time the restoration itself. Both trees
# must have the storage layout this relies on.
for S in (PB.SparseIterate, PB.SymmetricSparseIterate)
    (:nzval, :nzval2ind, :ind2nzval, :nnz) ⊆ fieldnames(S) || error("unexpected layout of $S")
end

const SIZES = (64, 1024, 16384)          # active coordinates: small, medium, large
const ZERO_FRACTIONS = (0.0, 0.1, 0.5, 0.9, 1.0)
const NPATTERNS = 32                     # cycled: varying working sets and values (and zero masks unless 0% or 100%)
const BATCH_COORDS = 4096                # a sample runs max(1, 4096 ÷ s) calls on separate iterates
const MAINTAIN_SIZES = (1024, 16384)     # sizes of the working-set maintenance workload

# the generator seed of a case: the first 8 bytes (big-endian) of SHA-256("dropzeros834|SEED|id")
case_seed(seed, id) = foldl((a, b) -> (a << 8) | b, UInt64.(sha256("dropzeros834|$seed|$id")[1:8]); init=UInt64(0))

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

# restore x to pattern P in O(x.nnz + length(P.ids)): unmap the live coordinates, then write P
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

# The writes of one cycle of the maintenance workload on pattern j: new values of the working-set
# slots in order (10% are ±0), then nonzero values for inactive coordinates, which activate them.
struct Writes{T}
    slotvals::Vector{T}
    newids::Vector{Int}
    newvals::Vector{T}
end

# Op is :drop, :fill_drop, :reset or :maintain, a type parameter so that the timed call
# dispatches statically
mutable struct Case{X, T, Op}
    id::String
    seed::UInt64
    xs::Vector{X}
    patterns::Vector{Pattern{T}}
    writes::Vector{Writes{T}}            # :maintain only
    current::Vector{Int}                 # the pattern each iterate was restored to
    next::Int
    dropns::UInt64                       # :maintain only: time inside dropzeros!, and its calls
    dropcalls::Int
end

function setup!(c::Case)
    for (k, x) in enumerate(c.xs)
        restore!(x, c.patterns[c.next])
        c.current[k] = c.next
        c.next = c.next % length(c.patterns) + 1
    end
    nothing
end

# SHA-256 of the fixtures: for every pattern the bytes of its coordinate IDs (Int64) and of its
# values (the exact bits, little-endian), then the same for the workload writes
ENDIAN_BOM == 0x04030201 && Int === Int64 || error("the fingerprint assumes a little-endian 64-bit host")
function fingerprint(c::Case)
    ctx = SHA.SHA256_CTX()
    for P in c.patterns
        SHA.update!(ctx, reinterpret(UInt8, P.ids)); SHA.update!(ctx, reinterpret(UInt8, P.vals))
    end
    for W in c.writes
        SHA.update!(ctx, reinterpret(UInt8, W.slotvals)); SHA.update!(ctx, reinterpret(UInt8, W.newids))
        SHA.update!(ctx, reinterpret(UInt8, W.newvals))
    end
    bytes2hex(SHA.digest!(ctx))
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
# one cycle of working-set maintenance through the public API: write every active coordinate (some
# become zero), activate new ones by writing nonzeros, then dropzeros!, whose time is also accumulated
function kernel!(c::Case{X, T, :maintain}) where {X, T}
    for (k, x) in enumerate(c.xs)
        W = c.writes[c.current[k]]
        coords = PB.coordinates(x)
        active = PB.activecoordinates(x)
        for t in eachindex(W.slotvals)
            coords[active[t]] = W.slotvals[t]
        end
        for t in eachindex(W.newids)
            coords[W.newids[t]] = W.newvals[t]
        end
        t0 = time_ns()
        dropzeros!(x)
        c.dropns += time_ns() - t0
        c.dropcalls += 1
    end
    nothing
end

function make_writes(rng, ::Type{T}, P::Pattern{T}, q) where {T}
    s = length(P.ids)
    slotvals = T[(rand(rng, Bool) ? 1 : -1) * (T(0.5) + rand(rng, T)) for _ = 1:s]
    for k in randperm(rng, s)[1:round(Int, 0.1s)]
        slotvals[k] = rand(rng, Bool) ? zero(T) : -zero(T)
    end
    inactive = setdiff(randperm(rng, q), P.ids)
    newids = inactive[1:round(Int, 0.1s)]
    newvals = T[(rand(rng, Bool) ? 1 : -1) * (T(0.5) + rand(rng, T)) for _ in newids]
    Writes(slotvals, newids, newvals)
end

function build_cases()
    cases = Case[]
    for kind in (:sparse, :symmetric), T in (Float32, Float64), s in SIZES
        q = ncoords(s)
        K = max(1, BATCH_COORDS ÷ s)
        ops = [("zeros$(round(Int, 100f))", :drop, f) for f in ZERO_FRACTIONS]
        push!(ops, ("fill_drop", :fill_drop, 0.0), ("reset", :reset, 0.5))
        s in MAINTAIN_SIZES && push!(ops, ("maintain10", :maintain, 0.0))
        for (name, op, f) in ops
            id = "$kind/$T/s$s/$name"
            occursin(FILTER, id) || continue
            seed = case_seed(SEED, id)
            rng = Xoshiro(seed)
            xs = [make_iterate(kind, T, q) for _ = 1:K]
            qq = PB.ncoordinates(xs[1])
            patterns = [make_pattern(rng, T, qq, s, f) for _ = 1:NPATTERNS]
            writes = op === :maintain ? [make_writes(rng, T, P, qq) for P in patterns] : Writes{T}[]
            push!(cases, Case{eltype(xs), T, op}(id, seed, xs, patterns, writes, zeros(Int, K), 1, 0, 0))
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
            if opof(c) === :maintain
                W = c.writes[j]
                P = Pattern(vcat(P.ids, W.newids), vcat(W.slotvals, W.newvals))
            end
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
    c.dropns, c.dropcalls = 0, 0
    t = run(b)
    dropmean = c.dropcalls == 0 ? "" : c.dropns / c.dropcalls
    K = length(c.xs)
    # allocation of one warmed batch, measured directly as well
    setup!(c); kernel!(c); setup!(c)
    direct = @allocated kernel!(c)
    (median(t.times) / K, minimum(t.times) / K, mean(t.times) / K, quantile(t.times, 0.9) / K,
     t.allocs, t.memory, direct, length(t.times), K, dropmean)
end

cases = build_cases()
open(OUTPUT, "w") do io
    println(io, "case,seed,fingerprint,status,median_ns,min_ns,mean_ns,p90_ns,allocs,memory,direct_bytes,samples,batch,drop_mean_ns")
    head(c) = string(c.id, ',', c.seed, ',', fingerprint(c), ',')
    if MODE == "fingerprint"
        foreach(c -> println(io, head(c), "pass,,,,,,,,,", length(c.xs), ','), cases)
    elseif MODE == "check"
        for c in cases
            status = opof(c) === :reset ? "not_checked" : check(c) ? "pass" : "FAIL"
            println(io, head(c), status, ",,,,,,,,,", length(c.xs), ',')
        end
    else
        foreach(c -> (setup!(c); kernel!(c)), cases)    # compile every specialization first
        for c in (isodd(SEED) ? cases : reverse(cases))
            r = measure(c)
            println(io, head(c), "pass,", join(r, ','))
            flush(io)
        end
    end
end
