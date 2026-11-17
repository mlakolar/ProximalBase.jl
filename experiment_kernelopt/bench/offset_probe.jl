# Offset sweep for the soft-threshold loops: the time of one call as a function of the page offset of the output
# relative to the input, (pointer(y) - pointer(x)) mod 4096, for the library loops of two trees, the reference tree
# compiled a second time, and standalone kernels (kernels/kernels.jl). Both arrays are views of nothing: they are
# true Arrays wrapped around 2 MiB-aligned regions, on transparent huge pages (madvise) unless the layout says small.
#   julia --threads=1 --check-bounds=no bench/offset_probe.jl <reference root> <candidate root> <output csv> [rounds]
# Row: element type, n, backing, y offset (bytes), implementation, minimum and median over rounds of one call (ns).
# KO_OFFSET_SIZES (lengths, comma separated; default 1048576,16384,1024) and KO_OFFSET_IMPLS (a regex on the names of
# the implementations; default all) restrict a run, for sweeps over more lengths with the library loops only.
# KO_OFFSET_PAGES (default 0) lists numbers of whole 4 KiB pages added to the output offsets 0, 64, 96, 128 and 192 for
# the lengths of KO_OFFSET_PAGE_SIZES: the offset modulo 4096 is unchanged, and the two arrays no longer have the same
# position within their 2 MiB pages. The last columns of a row are that number of pages and the percentages of the
# 2 MiB pages of x and of y that are mapped as huge pages (bench/hugepct.jl; -1 unknown), measured after the timing.
using Random, Printf, Statistics
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(abspath(root), "src", "ProximalBase.jl"))
  Base.invokelatest(getfield, m, :ProximalBase)   # the binding is defined in a later world than this call
end
const REF = loadtree("ref", ARGS[1]); const CAND = loadtree("cand", ARGS[2]); const REFC = loadtree("refC", ARGS[1])
include(joinpath(@__DIR__, "hugepct.jl"))
module K; include(joinpath(@__DIR__, "..", "kernels", "kernels.jl")); end
const ROUNDS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 11
const HUGE = 2 << 20
println("sizes ", get(ENV, "KO_OFFSET_SIZES", "default"), "; pages ", get(ENV, "KO_OFFSET_PAGES", "0"), "; page sizes ", get(ENV, "KO_OFFSET_PAGE_SIZES", "default"), "; implementations ", get(ENV, "KO_OFFSET_IMPLS", "all"))
const SIZES = Tuple(parse(Int, v) for v in split(get(ENV, "KO_OFFSET_SIZES", "1048576,16384,1024"), ','))
const IMPL_RX = Regex(get(ENV, "KO_OFFSET_IMPLS", "."))
const PAGES = Tuple(parse(Int, v) for v in split(get(ENV, "KO_OFFSET_PAGES", "0"), ','))
const PAGE_SIZES = Tuple(parse(Int, v) for v in split(get(ENV, "KO_OFFSET_PAGE_SIZES", "16384,1048576"), ','))
function region(bytes, advice)
  p = Ref{Ptr{Cvoid}}(C_NULL)
  ccall(:posix_memalign, Cint, (Ref{Ptr{Cvoid}}, Csize_t, Csize_t), p, HUGE, bytes) == 0 || error("posix_memalign")
  ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), p[], bytes, advice) == 0 || error("madvise")
  p[]
end
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
# Every helper has its own local names: a name assigned in a helper and also in the body of impls would be one shared
# (boxed) variable, and all closures would call whatever was assigned last.
function impls(::Type{T}, y, x, g) where {T}
  γ = T(0.8); c = T(0.8) * T(0.6); a = inv(one(T) + T(0.8) * T(0.5))
  l1(M) = (f = M.NormL1(T(0.6)); p = M.prox!; () -> p(y, f, x, γ))
  en(M) = (f = M.ElasticNet(T(0.6), T(0.5)); p = M.prox!; () -> p(y, f, x, γ))
  rd(M) = (fr = M.SqrNormL2(T(0.5)); pr = M.prox!; () -> pr(y, fr, x, γ))
  l2(M) = (f2 = M.NormL2(T(0.6)); p2 = M.prox!; () -> p2(y, f2, x, γ))
  kl(st) = (kf = K.l1prox!; () -> kf(st, y, x, c))
  ke(st) = (ef = K.enprox!; () -> ef(st, y, x, c, a))
  # a step per entry (the array g, in a third region at offset 0): the loop that candidate B925 vectorizes
  lg(M) = (fg = M.NormL1(T(0.6)); pg = M.prox!; () -> pg(y, fg, x, g))
  Pair{String, Any}["l1/ref" => l1(REF), "l1/cand" => l1(CAND), "l1/refC" => l1(REFC),
                    "enet/ref" => en(REF), "enet/cand" => en(CAND), "enet/refC" => en(REFC),
                    "l1/k_maxmin" => kl(K.st_maxmin), "l1/k_clampz" => kl(K.st_clampz), "l1/k_clamp" => kl(K.st_clamp), "l1/k_selmaxmin" => kl(K.st_selmaxmin),
                    "enet/k_maxmin" => ke(K.st_maxmin), "enet/k_clampz" => ke(K.st_clampz),
                    "copy" => () -> copyto!(y, x),
                    # loops that master already vectorizes: the ridge prox (one multiplication per entry), the NormL2 prox
                    # (a sum of squares, then one multiplication per entry) and a plain broadcast
                    "ridge/ref" => rd(REF), "l2/ref" => l2(REF), "scale" => () -> (y .= a .* x; nothing),
                    "l1g/ref" => lg(REF), "l1g/cand" => lg(CAND), "l1g/refC" => lg(REFC)]
end
open(ARGS[3], "w") do io
  println(io, "T,n,backing,yoff,impl,min_ns,med_ns,ypages,xhuge_pct,yhuge_pct")
  for T in (Float32, Float64), n in SIZES
    bytes = n * sizeof(T); reg = (bytes ÷ HUGE + 2) * HUGE
    src = randn(MersenneTwister(1000 + n), T, n)
    for (bname, adv) in (("huge", 14), ("small", 15))
      bname == "small" && n != 1 << 20 && continue
      offs = bname == "huge" ? (0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, 1024, 2048, 4000, 4032, 4064) : (0, 64, 128)
      px, py, pg = region(reg, adv), region(reg, adv), region(reg, adv)
      x = unsafe_wrap(Array, Ptr{T}(px + 4096), n); copyto!(x, src)
      g = unsafe_wrap(Array, Ptr{T}(pg + 4096), n); copyto!(g, T(0.5) .+ rand(MersenneTwister(7 + n), T, n))
      for pg4k in PAGES, yo in offs
        yo % sizeof(T) == 0 || continue
        pg4k == 0 || (bname == "huge" && n in PAGE_SIZES && yo in (0, 64, 96, 128, 192)) || continue
        0 <= pg4k && 4096 + yo + 4096 * pg4k + bytes <= reg || error("the shifted output leaves its region")
        y = unsafe_wrap(Array, Ptr{T}(py + 4096 + yo + 4096 * pg4k), n); fill!(y, zero(T))
        ims = filter(p -> occursin(IMPL_RX, p.first), impls(T, y, x, g))
        for (_, op) in ims; run1(op); end
        evals = [max(1, round(Int, 1e6 / max(1.0, timeone(op, 1)))) for (_, op) in ims]
        t = [Float64[] for _ in ims]
        for r in 1:ROUNDS
          order = circshift(collect(1:length(ims)), (r - 1) ÷ 2); isodd(r) || reverse!(order)
          for k in order; push!(t[k], timeone(ims[k].second, evals[k])); end
        end
        hx, hy = hugepct(x), hugepct(y)       # the backing obtained, after the timing
        for (k, (name, _)) in enumerate(ims)
          @printf(io, "%s,%d,%s,%d,%s,%.2f,%.2f,%d,%d,%d\n", T, n, bname, yo, name, minimum(t[k]), median(t[k]), pg4k, hx, hy)
        end
        flush(io)
      end
      Libc.free(px); Libc.free(py); Libc.free(pg)
    end
  end
end
println("done")
