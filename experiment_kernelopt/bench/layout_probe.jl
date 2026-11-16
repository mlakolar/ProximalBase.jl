# Memory-layout probe for the million-entry proximal loops: the same two arrays for both trees, with the page backing
# (transparent huge pages requested or refused by madvise before the first touch) and the page offsets of the input
# and the output under control, and also arrays as Julia allocates them, with their addresses and the huge-page
# coverage of their mappings (from /proc/self/smaps).
#   julia --threads=1 --check-bounds=no bench/layout_probe.jl <reference root> <candidate root> <output csv> [rounds]
# Each row: element type, function, n, layout, offsets of x and y (bytes mod 4096), huge-page kB of the mappings of
# x and y, and the minimum over rounds and the median over rounds of the time of one call for each tree (ns), the
# trees alternating within a round, in both orders.
using Random, Printf, Statistics
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(abspath(root), "src", "ProximalBase.jl"))
  getfield(m, :ProximalBase)
end
const REF = loadtree("ref", ARGS[1]); const CAND = loadtree("cand", ARGS[2]); const REFC = loadtree("refC", ARGS[1])
const ROUNDS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 15
const HUGE = 2 << 20
const MADV_HUGEPAGE = 14; const MADV_NOHUGEPAGE = 15

# a zero-filled-on-touch region of `bytes` (a multiple of 2 MiB) aligned to 2 MiB, with the given advice
function region(bytes, advice)
  p = Ref{Ptr{Cvoid}}(C_NULL)
  ccall(:posix_memalign, Cint, (Ref{Ptr{Cvoid}}, Csize_t, Csize_t), p, HUGE, bytes) == 0 || error("posix_memalign")
  advice == 0 || ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), p[], bytes, advice) == 0 || error("madvise")
  p[]
end
# AnonHugePages (kB) summed over the mappings that intersect [a, a + len)
function hugekb(a::UInt, len)
  kb = 0; inside = false
  for l in eachline("/proc/self/smaps")
    m = match(r"^([0-9a-f]+)-([0-9a-f]+) ", l)
    if m !== nothing
      lo, hi = parse(UInt, m.captures[1]; base = 16), parse(UInt, m.captures[2]; base = 16)
      inside = lo < a + len && a < hi
    elseif inside && startswith(l, "AnonHugePages:")
      kb += parse(Int, split(l)[2])
    end
  end
  kb
end
@noinline call(M, f::F, y, x, γ) where {F} = M.prox!(y, f, x, γ)
function timeone(M, f, y, x, γ, evals)
  best = typemax(UInt64)
  for _ in 1:3
    t0 = time_ns()
    for _ in 1:evals; call(M, f, y, x, γ); end
    best = min(best, time_ns() - t0)
  end
  best / evals
end
function measure(io, T, fname, n, layout, x, y)
  γ = T(0.8)
  mods = (REF, CAND, REFC)
  fs = map(M -> fname == "l1" ? M.NormL1(T(0.6)) : M.ElasticNet(T(0.6), T(0.5)), mods)
  for k in 1:3; call(mods[k], fs[k], y, x, γ); end
  evals = max(1, round(Int, 2e6 / max(1.0, timeone(mods[1], fs[1], y, x, γ, 1))))
  t = [Float64[] for _ in 1:3]
  for r in 1:ROUNDS
    order = circshift([1, 2, 3], (r - 1) ÷ 2); isodd(r) || reverse!(order)
    for k in order; push!(t[k], timeone(mods[k], fs[k], y, x, γ, evals)); end
  end
  px, py = UInt(pointer(x)), UInt(pointer(y))
  @printf(io, "%s,%s,%d,%s,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", T, fname, n, layout, px % 4096, py % 4096, px % HUGE, py % HUGE,
          hugekb(px, sizeof(x)), hugekb(py, sizeof(y)), minimum(t[1]), median(t[1]), minimum(t[2]), median(t[2]), minimum(t[3]), median(t[3]))
  flush(io)
end

open(ARGS[3], "w") do io
  println(io, "T,f,n,layout,xoff4k,yoff4k,xoff2m,yoff2m,xhugekb,yhugekb,ref_min,ref_med,cand_min,cand_med,refC_min,refC_med")
  for T in (Float32, Float64), n in (1 << 20, 1 << 14)
    bytes = n * sizeof(T); reg = (bytes ÷ HUGE + 2) * HUGE
    src = randn(MersenneTwister(1000 + n), T, n)
    # controlled: huge pages or 4 KiB pages for both, and the offset of y relative to x
    for (hname, adv) in (("huge", MADV_HUGEPAGE), ("small", MADV_NOHUGEPAGE)), (xo, yo) in ((0, 0), (64, 64), (0, 64), (0, 256), (0, 1024), (0, 2048), (0, 3072), (0, 4032), (64, 0), (2048, 0))
      n == 1 << 14 && !((xo, yo) in ((0, 0), (64, 64), (0, 64), (0, 2048))) && continue
      px, py = region(reg, adv), region(reg, adv)
      x = unsafe_wrap(Array, Ptr{T}(px + xo), n); y = unsafe_wrap(Array, Ptr{T}(py + yo), n)
      copyto!(x, src); fill!(y, zero(T))
      for fname in ("l1", "enet"); measure(io, T, fname, n, "$hname/x$xo/y$yo", x, y); end
      Libc.free(px); Libc.free(py)
    end
    # mixed backing: one array on huge pages and the other on 4 KiB pages
    for (hx, hy) in ((MADV_HUGEPAGE, MADV_NOHUGEPAGE), (MADV_NOHUGEPAGE, MADV_HUGEPAGE))
      px, py = region(reg, hx), region(reg, hy)
      x = unsafe_wrap(Array, Ptr{T}(px + 64), n); y = unsafe_wrap(Array, Ptr{T}(py + 64), n)
      copyto!(x, src); fill!(y, zero(T))
      for fname in ("l1", "enet"); measure(io, T, fname, n, hx == MADV_HUGEPAGE ? "xhuge-ysmall" : "xsmall-yhuge", x, y); end
      Libc.free(px); Libc.free(py)
    end
    # as Julia allocates them, eight times, with other allocations in between as in a longer process
    keep = Any[]
    for rep in 1:8
      x = copy(src); y = similar(x); fill!(y, zero(T))
      for fname in ("l1", "enet"); measure(io, T, fname, n, "julia$rep", x, y); end
      push!(keep, rand(T, rand(MersenneTwister(rep), 1:4) * n ÷ 3))
      isodd(rep) && (empty!(keep); GC.gc())
    end
  end
end
println("done")
