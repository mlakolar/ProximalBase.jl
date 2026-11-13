# PB #911: PB's own test suite on the disposable refactored copy (pkgs/pb-H) and, as the baseline, on the
# unmodified PB master export (pkgs/pb-master), each in a scratch environment. Disposable.
import Pkg
Pkg.offline(true)
const RUNDIR = normpath(joinpath(@__DIR__, ".."))
const WTROOT = normpath(joinpath(RUNDIR, "..", "..", ".."))
println("VERSION = ", VERSION, "  cpu = ", Sys.cpu_info()[1].model)
which = length(ARGS) >= 1 ? ARGS[1] : "pb-H,pb-master"
for copyname in split(which, ',')
  pb = joinpath(RUNDIR, "pkgs", copyname)
  env = joinpath(dirname(WTROOT), "pbtest-env-$(copyname)-$(VERSION)")
  rm(env; force = true, recursive = true); mkpath(env)
  Pkg.activate(env; io = devnull)
  Pkg.develop(path = pb; io = devnull)
  for p in ("Test", "Random", "Statistics", "TOML", "LinearAlgebra", "SparseArrays")
    try
      Pkg.add(p; io = devnull)
    catch e
      println("could not add $p offline: ", sprint(showerror, e))
    end
  end
  println("=== PB test suite on $copyname ($(strip(read(joinpath(pb, "TREE"), String))))")
  flush(stdout)
  t = @elapsed ok = success(pipeline(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env $(joinpath(pb, "test", "runtests.jl"))`,
                                            "OPENBLAS_NUM_THREADS" => "1"); stdout, stderr))
  println("=== $copyname: test suite ", ok ? "PASSED" : "FAILED", " in ", round(t; digits = 1), " s")
  flush(stdout)
end
