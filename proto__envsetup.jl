# Disposable environment setup for PB #911 prototype runs (included first by every script).
# PB_ENV=master (default): a scratch environment outside the mirrored worktree with the exported
# PB master tree developed; PB_ENV=pin: the PS worktree's own environment (PB at the declared pin).
import Pkg
# OFFLINE=false (as a NAME=value argument) lets a first run download the few registered dependencies
Pkg.offline(!any(==("OFFLINE=false"), ARGS))
const RUNDIR = normpath(joinpath(@__DIR__, ".."))
const WTROOT = normpath(joinpath(RUNDIR, "..", "..", ".."))
isfile(joinpath(WTROOT, "Project.toml")) || error("cannot find the worktree root from $(@__DIR__)")
const ARGOPTS = Dict{String, String}(String(first(kv)) => String(last(kv)) for kv in (split(a, '='; limit = 2) for a in ARGS if occursin('=', a)))
argopt(name, default) = get(ARGOPTS, name, get(ENV, name, default))
const PB_ENV = argopt("PB_ENV", "master")
if PB_ENV == "master"
  envdir = joinpath(dirname(WTROOT), "env-pbmaster-" * string(VERSION))     # outside the rsync mirror
  mkpath(envdir)
  Pkg.activate(envdir; io = devnull)
  pbpath = joinpath(RUNDIR, "pkgs", "pb-master")
  deps = Pkg.dependencies()
  have = any(d -> d.name == "ProximalBase" && d.source == pbpath, values(deps))
  if !have
    Pkg.develop(path = pbpath; io = devnull)
  end
  names = Set(d.name for d in values(Pkg.dependencies()) if d.is_direct_dep)
  for p in ("PrecompileTools", "LinearAlgebra", "Random", "SparseArrays", "Printf", "SHA")
    p in names || Pkg.add(p; io = devnull)
  end
elseif PB_ENV == "pin"
  Pkg.activate(WTROOT; io = devnull)
  VERSION >= v"1.11" && Pkg.instantiate(io = devnull)
else
  error("PB_ENV must be master or pin")
end
using LinearAlgebra
import ProximalBase
BLAS.set_num_threads(1)
function print_provenance()
  println("VERSION = ", VERSION)
  println("PB_ENV = ", PB_ENV, "  active project = ", Base.active_project())
  println("pathof(ProximalBase) = ", pathof(ProximalBase))
  for (_, info) in Pkg.dependencies()
    info.name == "ProximalBase" &&
      println("ProximalBase: version=", info.version, " source=", info.source, " git_revision=", info.git_revision,
              " tree_hash=", info.tree_hash)
  end
  pbsrc = dirname(dirname(pathof(ProximalBase)))
  if isfile(joinpath(pbsrc, "TREE"))
    println("PB exported tree marker: ", strip(read(joinpath(pbsrc, "TREE"), String)))
  end
  println("BLAS.get_config() = ", BLAS.get_config())
  println("BLAS.get_num_threads() = ", BLAS.get_num_threads(), "  Threads.nthreads() = ", Threads.nthreads())
  println("Sys.cpu_info()[1].model = ", Sys.cpu_info()[1].model, "  hostname-class = ", get(ENV, "SLURM_JOB_PARTITION", "?"))
  println("SLURM_JOB_ID = ", get(ENV, "SLURM_JOB_ID", "-"))
  flush(stdout)
end
