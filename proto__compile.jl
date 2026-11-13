# PB #911: compile/precompile cost through the real PB → PS package boundary (disposable).
# For each variant V, pkgs/bnd/V/{ProximalBase,ProximalSolvers} are developed into a scratch
# environment with a private first depot, so the two packages compile into caches of their own.
# Measured in fresh child processes:
#   precompile wall time of ProximalBase, then of ProximalSolvers (REPS_PRE repetitions, caches removed
#   between them), cache-file sizes, load time, time of the first and second solve of each
#   (solver, T) pair, and the methods/specializations that the PB-side code adds.
# Run (cwd = worktree root): julia --startup-file=no <this file> [VARIANTS=K,B1,B2,B3,C] [REPS_PRE=3] [REPS_TTFX=5]
import Pkg
Pkg.offline(true)
using Printf
const RUNDIR = normpath(joinpath(@__DIR__, ".."))
const WTROOT = normpath(joinpath(RUNDIR, "..", "..", ".."))
const ARGOPTS = Dict{String, String}(String(first(kv)) => String(last(kv)) for kv in (split(a, '='; limit = 2) for a in ARGS if occursin('=', a)))
argopt(name, default) = get(ARGOPTS, name, get(ENV, name, default))
const VARIANTS = String.(split(argopt("VARIANTS", "K,B1,B2,B3,H,C"), ','))
const REPS_PRE = parse(Int, argopt("REPS_PRE", "3"))
const REPS_TTFX = parse(Int, argopt("REPS_TTFX", "5"))
const WORK = joinpath(dirname(WTROOT), "bnd-" * string(VERSION))
println("VERSION = ", VERSION, "  cpu = ", Sys.cpu_info()[1].model, "  SLURM_JOB_ID = ", get(ENV, "SLURM_JOB_ID", "-"))
println("DEPOT_PATH = ", DEPOT_PATH)
flush(stdout)
const JL = Base.julia_cmd()
const BASE_DEPOTS = join(DEPOT_PATH, Sys.iswindows() ? ";" : ":")

function child(code; env, depot)
  cmd = `$JL --startup-file=no --project=$env -e $code`
  cmd = addenv(cmd, "JULIA_DEPOT_PATH" => depot * ":" * BASE_DEPOTS, "JULIA_PKG_OFFLINE" => "true",
               "OPENBLAS_NUM_THREADS" => "1", "JULIA_NUM_THREADS" => "1")
  read(cmd, String)
end
const SETUP = """
import Pkg; Pkg.offline(true)
"""
const TTFX = raw"""
using LinearAlgebra, Random; BLAS.set_num_threads(1)
t_load = @elapsed (using ProximalSolvers, ProximalBase)
PS = ProximalSolvers; PB = ProximalBase
res = String[]
for (sname, mk) in (("cd", () -> PS.CDSolver(order = :cyclic)), ("pg", () -> PS.ProximalGradientDescent()), ("apg", () -> PS.AccProxGradDescent())),
    T in (Float64, Float32)
  rng = Xoshiro(5); X = randn(rng, T, 57, 13); y = randn(rng, T, 57)
  f = (one(T) / T(57)) * PB.LeastSquaresLoss(y, X); g = PB.NormL1(T(0.01))
  t1 = @elapsed PS.solve!(zeros(T, 13), f, g, mk())
  t2 = @elapsed PS.solve!(zeros(T, 13), f, g, mk())
  fr = PB.LeastSquaresLoss(y, X)
  t3 = @elapsed PS.solve!(zeros(T, 13), fr, g, mk())
  push!(res, string(sname, ":", T, "=", t1, "/", t2, "/", t3))
end
println("TTFX load=", t_load, " ", join(res, " "))
"""
const COUNT = raw"""
using ProximalBase, ProximalSolvers
PB = ProximalBase
let nm = 0, ns = 0, fnames = String[]
for s in names(PB; all = true)
  isdefined(PB, s) || continue
  f = getfield(PB, s)
  f isa Function || continue
  for m in methods(f)
    endswith(string(m.file), "leastsquares_image.jl") || continue
    nm += 1
    push!(fnames, string(s))
    ns += length(collect(Base.specializations(m)))
  end
end
println("COUNT functions=", length(unique(fnames)), " methods=", nm, " specializations_after_load=", ns, " names=", join(sort(unique(fnames)), ";"))
end
"""
dirsize(d) = isdir(d) ? sum((filesize(joinpath(r, f)) for (r, _, fs) in walkdir(d) for f in fs); init = 0) : 0
for v in VARIANTS
  env = joinpath(WORK, "env-" * v); depot = joinpath(WORK, "depot-" * v)
  rm(env; force = true, recursive = true); rm(depot; force = true, recursive = true)
  mkpath(env); mkpath(depot)
  pb = joinpath(RUNDIR, "pkgs", "bnd", v, "ProximalBase"); ps = joinpath(RUNDIR, "pkgs", "bnd", v, "ProximalSolvers")
  out = child(SETUP * """
    Pkg.develop([Pkg.PackageSpec(path = $(repr(pb))), Pkg.PackageSpec(path = $(repr(ps)))]; io = devnull)
    for (u, d) in Pkg.dependencies(); d.is_direct_dep && println("DEP ", d.name, " ", d.source); end
    """; env, depot = depot)
  print(replace(out, r"^"m => "[$v] "))
  comp = joinpath(depot, "compiled", "v$(VERSION.major).$(VERSION.minor)")
  for rep in 1:REPS_PRE
    rm(joinpath(comp, "ProximalBase"); force = true, recursive = true)
    rm(joinpath(comp, "ProximalSolvers"); force = true, recursive = true)
    o1 = child(SETUP * """
      t = @elapsed Base.compilecache(Base.identify_package("ProximalBase"))
      println("PRE_PB ", t)
      t = @elapsed Base.compilecache(Base.identify_package("ProximalSolvers"))
      println("PRE_PS ", t)
      """; env, depot)
    tpb = parse(Float64, match(r"PRE_PB (\S+)", o1)[1]); tps = parse(Float64, match(r"PRE_PS (\S+)", o1)[1])
    @printf "[%s] precompile rep %d: ProximalBase %.2f s, ProximalSolvers %.2f s; cache bytes PB=%d PS=%d\n" v rep tpb tps dirsize(joinpath(comp, "ProximalBase")) dirsize(joinpath(comp, "ProximalSolvers"))
    println("CSV,precompile,$VERSION,$v,$rep,$tpb,$tps,", dirsize(joinpath(comp, "ProximalBase")), ",", dirsize(joinpath(comp, "ProximalSolvers")))
    flush(stdout)
  end
  for rep in 1:REPS_TTFX
    o = child(TTFX; env, depot)
    line = strip(something(match(r"TTFX .*", o), (match = "TTFX missing: " * o,)).match)
    println("[$v] rep $rep $line")
    println("CSV,ttfx,$VERSION,$v,$rep,", replace(line, " " => ","))
    flush(stdout)
  end
  v == "K" || (o = child(COUNT; env, depot); println("[$v] ", strip(o)); println("CSV,count,$VERSION,$v,", replace(strip(o), " " => ",")))
  flush(stdout)
end
