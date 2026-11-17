# Ordinary allocations: where Julia places an output array relative to its input, and what the two soft-threshold
# loops cost there. For every element type, length and allocation pattern, a number of trials; in each trial the two
# arrays are allocated as a caller would allocate them, their addresses are recorded, and prox!(y, f, x, γ) of NormL1
# and of ElasticNet is timed for the reference tree, the candidate tree and the reference compiled a second time, on
# those two arrays, the implementations alternating over the rounds in both orders.
#   julia --threads=1 --check-bounds=no bench/alloc_probe.jl <reference root> <candidate root> <output csv> [rounds] [trials]
# Patterns:
#   similar      x = Vector{T}(undef, n); y = similar(x)
#   gap          the same with one other allocation of a random size (16 bytes to 64 KiB) between the two
#   prox         y = prox(f, x, γ): the array that an allocating call returns
#   later        y allocated after 1 to 8 unrelated arrays of the same length were allocated, some of them no longer
#                referenced
#   similar_thp  as similar, and before either array is written the pages of both are advised MADV_HUGEPAGE: arrays at
#   later_thp    the addresses that Julia chose, on huge pages wherever the kernel can provide them; as later
# Odd trials keep the arrays of the earlier trials alive, even trials free them and collect first, so that the heap
# differs between trials. Row: element type, n, pattern, trial, (pointer(y) - pointer(x)) mod 4096, the distance in
# bytes, pointer(x) mod 4096, for x and for y the percentage of the 2 MiB pages that hold the array that are mapped
# as huge pages (bench/hugepct.jl; -1 unknown) after the first calls and again after the timing, the function, the implementation, and the minimum and the median over the rounds of one call (ns).
using Random, Printf, Statistics
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(abspath(root), "src", "ProximalBase.jl"))
  Base.invokelatest(getfield, m, :ProximalBase)   # the binding is defined in a later world than this call
end
const REF = loadtree("ref", ARGS[1]); const CAND = loadtree("cand", ARGS[2]); const REFC = loadtree("refC", ARGS[1])
const ROUNDS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 7
const TRIALS = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 16
include(joinpath(@__DIR__, "hugepct.jl"))
@noinline run1(op::F) where {F} = op()
function timeone(op::F, evals) where {F}
  best = typemax(UInt64)
  for _ in 1:3
    t0 = time_ns()
    for _ in 1:evals; run1(op); end
    best = min(best, time_ns() - t0)
  end
  best / evals
end
# Every helper has its own local names (a name shared with the body would be one boxed variable for all closures).
function impls(::Type{T}, y, x) where {T}
  γ = T(0.8)
  l1(M) = (fl = M.NormL1(T(0.6)); pl = M.prox!; () -> pl(y, fl, x, γ))
  en(M) = (fe = M.ElasticNet(T(0.6), T(0.5)); pe = M.prox!; () -> pe(y, fe, x, γ))
  Pair{String, Any}["l1/ref" => l1(REF), "l1/cand" => l1(CAND), "l1/refC" => l1(REFC),
                    "enet/ref" => en(REF), "enet/cand" => en(CAND), "enet/refC" => en(REFC)]
end
function allocate(::Type{T}, n, pattern, rng, junk) where {T}
  x = Vector{T}(undef, n)
  thp = endswith(pattern, "_thp"); thp && (pattern = pattern[1:end-4])
  pattern == "prox" && randn!(rng, x)
  if pattern == "similar"
    y = similar(x)
  elseif pattern == "gap"
    push!(junk, Vector{UInt8}(undef, 16 * rand(rng, 1:4096)))
    y = similar(x)
  elseif pattern == "prox"
    y = REF.prox(REF.NormL1(T(0.6)), x, T(0.8))
  else
    k = rand(rng, 1:8)
    tmp = [Vector{T}(undef, n) for _ in 1:k]
    for j in 1:2:k; push!(junk, tmp[j]); end
    tmp = nothing
    y = Vector{T}(undef, n)
  end
  if thp
    hugeadvise(UInt(pointer(x)), sizeof(x)); hugeadvise(UInt(pointer(y)), sizeof(y))
  end
  pattern == "prox" || randn!(rng, x)
  fill!(y, zero(T))
  x, y
end
open(ARGS[3], "w") do io
  println(io, "T,n,pattern,trial,yoff,dist,xoff,xhuge_pct,yhuge_pct,func,impl,min_ns,med_ns,xhuge_after,yhuge_after")
  seed = parse(Int, get(ENV, "KO_ALLOC_SEED", string(getpid())))
  println("seed ", seed, "; rounds ", ROUNDS, "; trials ", TRIALS)
  rng = MersenneTwister(seed)
  # the advised patterns last: the advice stays on the memory that the allocator reuses after the arrays are freed
  for patterns in (("similar", "gap", "prox", "later"), ("similar_thp", "later_thp")), T in (Float32, Float64),
      n in (1024, 4096, 8192, 16384, 65536, 100_000, 262144, 1_000_000, 1048576)
    for pattern in patterns
      kept = Any[]; junk = Any[]
      for trial in 1:TRIALS
        if iseven(trial)
          empty!(kept); empty!(junk); GC.gc()
        end
        x, y = allocate(T, n, pattern, rng, junk)
        push!(kept, x); push!(kept, y)
        px, py = UInt(pointer(x)), UInt(pointer(y))
        ims = impls(T, y, x)
        for (_, op) in ims; run1(op); end
        hx, hy = hugepct(px, n * sizeof(T)), hugepct(py, n * sizeof(T))   # after the first touch of both arrays
        evals = [max(1, round(Int, 1e6 / max(1.0, timeone(op, 1)))) for (_, op) in ims]
        t = [Float64[] for _ in ims]
        for r in 1:ROUNDS
          order = circshift(collect(1:length(ims)), (r - 1) ÷ 2); isodd(r) || reverse!(order)
          for k in order; push!(t[k], timeone(ims[k].second, evals[k])); end
        end
        hx2, hy2 = hugepct(px, n * sizeof(T)), hugepct(py, n * sizeof(T))
        for (k, (name, _)) in enumerate(ims)
          func, impl = split(name, '/')
          @printf(io, "%s,%d,%s,%d,%d,%d,%d,%d,%d,%s,%s,%.2f,%.2f,%d,%d\n", T, n, pattern, trial, mod(py - px, UInt(4096)), Int(py) - Int(px),
                  mod(px, UInt(4096)), hx, hy, func, impl, minimum(t[k]), median(t[k]), hx2, hy2)
        end
        flush(io)
      end
    end
  end
end
println("done")
