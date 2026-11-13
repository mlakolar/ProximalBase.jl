# In-process diagnostics for #872 (exploratory; the headline comparison is run.py).
# usage: julia -O2 --check-bounds=no --project=benchmark/audit852 diag.jl SOURCES_DIR OUTDIR [quick]
# Loads trees A, B, C, C2, D and CD (NAMES below), each as its own module, builds identical fixtures in each, and
# times public copyto! with the candidates in rotated order, plus a decomposition of
# master's _copy_active! and the cost of comparing independently built layouts.
using BenchmarkTools, Random, Statistics, InteractiveUtils, Printf
const SRCDIR, OUT = abspath(ARGS[1]), abspath(ARGS[2])
const QUICK = length(ARGS) > 2
mkpath(OUT)
const NAMES = ["A", "B", "C", "C2", "D", "CD"]
const MODS = Dict{String,Module}()
for name in NAMES
  m = Module(Symbol("Cand", name))
  Core.eval(m, :(using Base))
  Base.include(m, joinpath(SRCDIR, name, "src", "ProximalBase.jl"))
  MODS[name] = getfield(m, :ProximalBase)
end

function group_layout(n, b, layout, rng)
  K = n ÷ b
  layout == "contiguous" && return [(k-1)*b+1:k*b for k in 1:K]
  layout == "strided" && return [k:K:n for k in 1:K]
  perm = randperm(rng, n)
  [perm[(k-1)*b+1:k*b] for k in 1:K]
end

function restore!(d, t)
  copyto!(d.storage, t.storage); copyto!(d.active, t.active); copyto!(d.position, t.position)
  d.nactive = t.nactive; nothing
end

# decomposition kernels on master's fields (each does one part of _copy_active!)
function k_values!(d, s)
  @inbounds for a in 1:s.nactive, i in s.groups[s.active[a]]
    d.storage[i] = s.storage[i]
  end
  d
end
function k_clear!(d::AbstractVector{T}, s) where {T}
  @inbounds for a in 1:d.nactive
    k = d.active[a]
    if s.position[k] == 0
      z = zero(T)
      for i in d.groups[k]; d.storage[i] = z; end
    end
  end
  d
end
function k_ids!(d, s)
  @inbounds for a in 1:s.nactive; d.active[a] = s.active[a]; end
  d
end
function k_pos!(d, s)
  @inbounds for a in 1:d.nactive; d.position[d.active[a]] = 0; end
  @inbounds for a in 1:s.nactive; d.position[s.active[a]] = a; end
  d
end
k_dense_values!(d, s) = (copyto!(d.storage, s.storage); d)
k_bulk_pos!(d, s) = (copyto!(d.position, s.position); d)
k_layout_eq(d, s) = d.groups == s.groups

function timeit(f, prep; samples = QUICK ? 50 : 300)
  for _ in 1:2; prep(); f(); end
  t = @benchmark $f() setup = ($prep()) evals = 1 samples = samples seconds = 1
  median(t).time, minimum(t).time, median(t).allocs
end

# function barrier: the closures capture arguments only, so nothing is boxed
bench(kf::F, dest, src, tmpl) where {F} = timeit(() -> kf(dest, src), () -> restore!(dest, tmpl))

function fixture(PB, T, n, b, layout, frac, dstate, sharing, seed)
  rng = Xoshiro(seed); gs = group_layout(n, b, layout, rng); K = length(gs)
  s = max(1, round(Int, K * frac / 100))
  ks = randperm(rng, K)[1:s]
  src = PB.BlockIterate(T, n, gs)
  for (a, k) in enumerate(ks), (j, i) in enumerate(gs[k]); src[i] = T((mod(a + j, 15) + 1) / 16); end
  dest = sharing == "shared" ? similar(src) : PB.BlockIterate(T, n, gs)
  sharing == "cached" && PB.samegroups(dest, src)    # one validated comparison, remembered
  destonly = setdiff(1:K, ks)[1:min(s, K - s)]
  if dstate == "populated"
    for k in reverse(ks), i in gs[k]; dest[i] = T(3/4); end
    for k in destonly; dest[first(gs[k])] = T(1/4); end
  elseif dstate == "destonly"
    for k in destonly; dest[first(gs[k])] = T(1/4); end
  end
  src, dest, copy(dest), s, K
end

io = open(joinpath(OUT, "diag.csv"), "w")
println(io, "T,n,b,layout,frac,s,K,dest,sharing,kernel,rep_medians_ns,median_ns,min_ns,allocs")
grid = QUICK ? [(T, n, b, l, f) for T in (Float64,), n in (8192,), b in (1, 64), l in ("contiguous", "arbitrary"), f in (10, 100)] :
       [(T, n, b, l, f) for T in (Float32, Float64), n in (8192, 65536), b in (1, 8, 64),
        l in ("contiguous", "strided", "arbitrary"), f in (1, 10, 50, 100)]
for (ci, (T, n, b, layout, frac)) in enumerate(grid), dstate in ("empty", "populated", "destonly"), sharing in ("shared", "independent", "cached")
  frac == 100 && dstate == "destonly" && continue
  seed = 872 + ci
  fx = Dict(name => fixture(MODS[name], T, n, b, layout, frac, dstate, sharing, seed) for name in NAMES)
  s, K = fx["A"][4], fx["A"][5]
  res = Dict{String,Vector{Tuple{Float64,Float64,Int}}}()
  for rep in 1:3
    order = circshift(NAMES, rep + ci)
    for name in order
      push!(get!(res, name, []), bench(copyto!, fx[name][2], fx[name][1], fx[name][3]))
    end
  end
  # master decomposition and layout costs
  PB = MODS["A"]
  kern = ["values" => k_values!, "clear" => k_clear!, "ids" => k_ids!, "positions" => k_pos!,
          "copy_active" => PB._copy_active!, "dense_values" => k_dense_values!,
          "bulk_positions" => k_bulk_pos!, "samelayout" => PB._samelayout]
  sharing == "independent" && push!(kern, "groups_eq" => k_layout_eq)
  for (kname, kf) in kern, rep in 1:3
    push!(get!(res, "A:" * kname, []), bench(kf, fx["A"][2], fx["A"][1], fx["A"][3]))
  end
  for (kname, rs) in sort!(collect(res); by = first)
    meds = [r[1] for r in rs]
    println(io, join((T, n, b, layout, frac, s, K, dstate, sharing, kname, join(Int.(round.(meds)), ';'),
                      median(meds), minimum(r[2] for r in rs), maximum(r[3] for r in rs)), ','))
  end
  flush(io)
end
close(io)

# generated code of the replacement kernel per candidate and layout type
open(joinpath(OUT, "codegen.txt"), "w") do cg
  for name in NAMES, (T, gsf) in ((Float64, n -> [k:k for k in 1:n]), (Float64, n -> [k:n:n for k in 1:n]),
                                  (Float64, n -> [[k] for k in 1:n]), (Float32, n -> [(k-1)*8+1:k*8 for k in 1:n÷8]))
    PB = MODS[name]
    x = PB.BlockIterate(T, 64, gsf(64))
    tt = Tuple{typeof(x), typeof(x)}
    nat = sprint(io -> code_native(io, PB._copy_active!, tt; debuginfo = :none, syntax = :intel))
    llc = sprint(io -> code_llvm(io, copyto!, tt; debuginfo = :none))
    ninstr = count(l -> occursin(r"^\s+[a-z]", l), split(nat, '\n'))
    calls = count(l -> occursin("call", l), split(llc, '\n'))
    println(cg, name, " ", typeof(x), " native_instructions(_copy_active!)=", ninstr,
            " llvm_lines(copyto!)=", count(==('\n'), llc), " llvm_calls(copyto!)=", calls)
  end
end
println("diag done")
