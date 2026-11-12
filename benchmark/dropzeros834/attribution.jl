# Bounded attribution of a flagged cell (#834): base and head loaded into one process and timed on
# the same fixtures, interleaved, which removes the differences between separate processes (their
# code and data still sit at different addresses); plus the native code of dropzeros! of both
# trees. Optional hardware counters with LinuxPerf.jl. ROUNDS = 0 skips the timing.
# usage: julia --project=ENV attribution.jl BASE_SRC HEAD_SRC OUTDIR ORDER(base-first|head-first) ROUNDS
using BenchmarkTools, Random, Statistics, SHA, InteractiveUtils, Printf
const BASE, HEAD, OUT, ORDER = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
const ROUNDS = parse(Int, ARGS[5])
mkpath(OUT)
module SrcBase end
module SrcHead end
for (m, src) in (ORDER == "base-first" ? ((SrcBase, BASE), (SrcHead, HEAD)) : ((SrcHead, HEAD), (SrcBase, BASE)))
    Base.include(m, joinpath(src, "src", "ProximalBase.jl"))
end
const TREES = (base = SrcBase.ProximalBase, head = SrcHead.ProximalBase)
using SparseArrays: dropzeros!
const HAVE_PERF = Base.find_package("LinuxPerf") !== nothing
HAVE_PERF && @eval using LinuxPerf

# fixtures exactly as in worker.jl (same seed derivation, generator calls and restoration)
case_seed(seed, id) = foldl((a, b) -> (a << 8) | b, UInt64.(sha256("dropzeros834|$seed|$id")[1:8]); init=UInt64(0))
make_iterate(PB, kind, ::Type{T}, q) where {T} =
    kind === :sparse ? PB.SparseIterate(T, q) : PB.SymmetricSparseIterate(T, ceil(Int, (sqrt(8q + 1) - 1) / 2))
function make_pattern(rng, ::Type{T}, q, s, f) where {T}
    ids = randperm(rng, q)[1:s]
    vals = T[(rand(rng, Bool) ? 1 : -1) * (T(0.5) + rand(rng, T)) for _ = 1:s]
    for k in randperm(rng, s)[1:round(Int, f * s)]
        vals[k] = rand(rng, Bool) ? zero(T) : -zero(T)
    end
    (ids, vals)
end
function restore!(x, (ids, vals))
    @inbounds for t = 1:x.nnz
        x.ind2nzval[x.nzval2ind[t]] = 0
    end
    @inbounds for t in eachindex(ids)
        x.nzval[t] = vals[t]; x.nzval2ind[t] = ids[t]; x.ind2nzval[ids[t]] = t
    end
    x.nnz = length(ids)
    nothing
end
mutable struct Cell{X, P}
    x::X
    patterns::P
    next::Int
end
setup!(c::Cell) = (restore!(c.x, c.patterns[c.next]); c.next = c.next % length(c.patterns) + 1; nothing)
kernel!(c::Cell) = (dropzeros!(c.x); nothing)

const CELLS = [(:symmetric, Float32, 16384, 0.1), (:symmetric, Float64, 16384, 0.1),
               (:sparse, Float32, 16384, 0.1), (:sparse, Float64, 16384, 0.1)]

open(joinpath(OUT, "timing-$ORDER.csv"), "w") do io
    println(io, "cell,round,tree,median_ns,min_ns")
    for (kind, T, s, f) in CELLS
        id = "$kind/$T/s$s/zeros$(round(Int, 100f))"
        cells = map(TREES) do PB
            x = make_iterate(PB, kind, T, 8s)
            rng = Xoshiro(case_seed(834001, id))     # same fixtures for both trees
            Cell(x, [make_pattern(rng, T, PB.ncoordinates(x), s, f) for _ = 1:32], 1)
        end
        @assert cells.base.patterns == cells.head.patterns
        bench(c) = (b = @benchmarkable kernel!($c) setup=(setup!($c)) evals=1 samples=4000 seconds=0.3; run(b))
        foreach(c -> (for _ = 1:5; setup!(c); kernel!(c); end), cells)
        for r = 1:ROUNDS
            for tree in (isodd(r) ? (:base, :head) : (:head, :base))
                t = bench(cells[tree])
                println(io, id, ',', r, ',', tree, ',', median(t.times), ',', minimum(t.times))
            end
            flush(io)
        end
        if HAVE_PERF
            # Counters of the call: a phase with the counters enabled after each restoration and disabled
            # after the call, and a control phase with an empty enable/disable window followed by the
            # same call with the counters disabled. Both phases start from the same primed state at
            # pattern 1 and run 2048 = 64 * 32 iterations, so every restoration has the same input
            # state and every pattern the same multiplicity. The control estimates the cost of the
            # window itself; call - control is the estimate per call. enabled/running are recorded.
            pb = LinuxPerf.make_bench([LinuxPerf.EventType(:hw, :cycles),
                                       [LinuxPerf.EventType(:hw, :instructions), LinuxPerf.EventType(:hw, :branches),
                                        LinuxPerf.EventType(:hw, :branch_mispredicts)],
                                       [LinuxPerf.EventType(:cache, :L1_data, :read, :miss)]])
            N = 64 * length(first(cells).patterns)
            totals(cnt) = Dict(string(k.event) => (Float64(k.value), Int(k.enabled), Int(k.running)) for k in cnt.counters)
            function prime!(c)                       # the state the loop leaves: pattern 32, compacted
                c.next = length(c.patterns); setup!(c); kernel!(c); c.next = 1
            end
            path = joinpath(OUT, "counters-$ORDER.csv")
            open(path, isfile(path) ? "a" : "w") do cio
                filesize(path) == 0 && println(cio, "cell,tree,event,call,empty,net_per_call,call_enabled,call_running,empty_enabled,empty_running")
                for tree in (:base, :head)
                    c = cells[tree]
                    prime!(c); LinuxPerf.reset!(pb)
                    for _ = 1:N
                        setup!(c); LinuxPerf.enable!(pb); kernel!(c); LinuxPerf.disable!(pb)
                    end
                    call = totals(LinuxPerf.counters(pb))
                    prime!(c); LinuxPerf.reset!(pb)
                    for _ = 1:N
                        setup!(c); LinuxPerf.enable!(pb); LinuxPerf.disable!(pb); kernel!(c)
                    end
                    empty = totals(LinuxPerf.counters(pb))
                    for e in sort(collect(keys(call)))
                        (cv, ce, cr), (ev, ee, er) = call[e], empty[e]
                        println(cio, id, ',', tree, ',', e, ',', cv, ',', ev, ',', (cv - ev) / N, ',', ce, ',', cr, ',', ee, ',', er)
                    end
                end
            end
        end
    end
end

for (name, PB) in pairs(TREES), (kind, S) in ((:sparse, PB.SparseIterate), (:symmetric, PB.SymmetricSparseIterate)),
        T in (Float32, Float64)
    sig = S === PB.SparseIterate ? S{T, 1} : S{T}
    asm = sprint(io -> code_native(io, dropzeros!, (sig,); debuginfo=:none, syntax=:intel))
    write(joinpath(OUT, "native-$name-$kind-$T.s"), asm)
end
