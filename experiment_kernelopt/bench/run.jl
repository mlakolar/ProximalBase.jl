# Driver of the kernel study.
#
#   KO_TREES="base=/path/to/tree[;name=/path/to/other]" KO_OUTPUT=<dir> \
#   KO_SUITES=l1,l2,group,ls KO_TYPES=Float64,Float32 KO_ROUNDS=21 KO_BATCH_MS=2 KO_SAMPLES=3 [KO_FILTER=regex] [KO_CHECK_ONLY=1] \
#     julia --threads=1 --check-bounds=no --project=<env> bench/run.jl
#
# Every tree is a ProximalBase source root (with src/). The first tree is the reference of the ratios and is
# compiled a second time as "<name>C", the A/A control of the library code; the standalone kernels have their
# own compiled copy (module KC).
using LinearAlgebra, Random, Printf, Dates
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(parse(Int, get(ENV, "KO_BLAS_THREADS", "1")))

include(joinpath(@__DIR__, "harness.jl"))
module K;  include(joinpath(@__DIR__, "..", "kernels", "kernels.jl")); end
module KC; include(joinpath(@__DIR__, "..", "kernels", "kernels.jl")); end
import ProximalOperators
const PO = ProximalOperators

function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
  Base.invokelatest(getfield, m, :ProximalBase)   # the binding is defined in a later world than this call
end
# a synced copy without git metadata carries its commit in the file TREE_SHA
treesha(root) = isfile(joinpath(root, "TREE_SHA")) ? strip(read(joinpath(root, "TREE_SHA"), String)) : try
  strip(read(`git -C $root rev-parse HEAD`, String)) * (isempty(strip(read(`git -C $root status --porcelain --untracked-files=no`, String))) ? "" : "+dirty")
catch
  "unknown"
end
# a tree older than the least-squares oracles (PR #918) has no predictor or residual functions: its cases are skipped
opt(M, name) = isdefined(M, name) ? getfield(M, name) : nothing
pbapi(M) = (prox! = M.prox!, prox_block = M.prox_block, value_and_prox! = M.value_and_prox!, prox_block! = M.prox_block!, NormL1 = M.NormL1, NormL2 = M.NormL2,
            GroupNormL2 = M.GroupNormL2, ElasticNet = opt(M, :ElasticNet), LeastSquaresLoss = M.LeastSquaresLoss, value_and_gradient! = M.value_and_gradient!,
            gradient! = M.gradient!, value_from_residual = opt(M, Symbol("value_from_residual")), value_from_predictor = opt(M, Symbol("value_from_predictor")),
            value_and_gradient_from_predictor! = opt(M, Symbol("value_and_gradient_from_predictor!")), mod = M)

specs = [split(s, '=') for s in split(ENV["KO_TREES"], ';') if !isempty(s)]
# KO_REF names the reference tree of the ratios (the first tree by default). The trees are compiled in the order
# of KO_TREES, so that fresh processes with another order place the code differently; the reference is always
# first in the list of implementations, followed by its second compilation "<ref>C", the A/A control.
refname = get(ENV, "KO_REF", String(specs[1][1]))
trees = Tuple{String, String, String}[]
loaded = Dict{String, Any}()
for (name, root) in specs
  root = abspath(String(root))
  push!(trees, (String(name), root, treesha(root)))
  loaded[String(name)] = Base.invokelatest(pbapi, loadtree(name, root))
end
refroot = only(t[2] for t in trees if t[1] == refname)
control = (refname * "C", Base.invokelatest(pbapi, loadtree(refname * "C", refroot)))
# The implementations of a case are in the order of KO_TREES, which is also the order of their first calls (and
# so of the compilation of their kernels); the control is first with KO_CONTROL_FIRST=1 and last otherwise. The
# reference of the ratios is found by name, wherever it is.
pbs = Tuple{String, Any}[(name, loaded[name]) for (name, _, _) in trees]
get(ENV, "KO_CONTROL_FIRST", "0") == "1" ? pushfirst!(pbs, control) : push!(pbs, control)
const REFNAME = refname

include(joinpath(@__DIR__, "suites.jl"))
include(joinpath(@__DIR__, "solvers.jl"))

const TYPES = Dict("Float64" => Float64, "Float32" => Float32)
suites = split(get(ENV, "KO_SUITES", "l1,l2,group,ls"), ',')
types = [TYPES[t] for t in split(get(ENV, "KO_TYPES", "Float64,Float32"), ',')]
cases = Harness.Case[]
for T in types
  "l1" in suites && append!(cases, Base.invokelatest(cases_l1, pbs, K, KC, PO, T))
  "l2" in suites && append!(cases, Base.invokelatest(cases_l2, pbs, K, KC, PO, T))
  "group" in suites && append!(cases, Base.invokelatest(cases_group, pbs, K, KC, PO, T))
  "groupbig" in suites && append!(cases, Base.invokelatest(cases_group, pbs, K, KC, PO, T; total = 2^20, sizes = (5, 32)))
  "ls" in suites && append!(cases, Base.invokelatest(cases_ls, pbs, K, KC, PO, T))
  "enet" in suites && append!(cases, Base.invokelatest(cases_enet, pbs, K, KC, PO, T))
  "solver" in suites && append!(cases, Base.invokelatest(cases_solver, pbs, T))
  "steps" in suites && append!(cases, Base.invokelatest(cases_steps, pbs, T))
  "a3" in suites && append!(cases, Base.invokelatest(cases_a3, pbs, T))
  if "tiny" in suites
    append!(cases, Base.invokelatest(cases_l1, pbs, K, KC, PO, T; sizes = TINY_SIZES))
    append!(cases, Base.invokelatest(cases_l2, pbs, K, KC, PO, T; sizes = TINY_SIZES))
    append!(cases, Base.invokelatest(cases_zero_groups, pbs, T))
    append!(cases, Base.invokelatest(cases_enet, pbs, K, KC, PO, T; sizes = TINY_SIZES))
  end
end
# KO_TREES_ONLY=1 keeps the source trees only (the formal tree-against-tree comparisons): the standalone kernels and
# the external libraries are then neither compiled nor timed
if get(ENV, "KO_TREES_ONLY", "0") == "1"
  names = Set(first.(pbs))
  for c in cases
    filter!(i -> i.name in names, c.impls)
  end
end
if haskey(ENV, "KO_FILTER")
  rx = Regex(ENV["KO_FILTER"])
  filter!(c -> occursin(rx, c.id), cases)
end
outdir = ENV["KO_OUTPUT"]
mkpath(outdir)
open(io -> Harness.provenance(io, trees), joinpath(outdir, "provenance.txt"), "w")
@printf(stderr, "%d cases, %d implementations\n", length(cases), sum(c -> length(c.impls), cases))
nfail = Base.invokelatest(Harness.run, cases, outdir; rounds = parse(Int, get(ENV, "KO_ROUNDS", "21")),
                          batch_ms = parse(Float64, get(ENV, "KO_BATCH_MS", "2")), samples = parse(Int, get(ENV, "KO_SAMPLES", "3")),
                          check_only = get(ENV, "KO_CHECK_ONLY", "0") == "1")
@printf(stderr, "done: %d output differences\n", nfail)
# a difference is a failed run: the timings of a wrong implementation are not results
nfail == 0 || (touch(joinpath(outdir, "CHECKFAIL")); exit(3))
