# Paired benchmark of KronQuadratic (#1059) between the archived source tree (eff21b6d), the candidate and a
# second compilation of the candidate (the A/A control), Float32 and Float64: the value and value_and_gradient!
# on dense p × q matrices, p = q ∈ {50, 200, 800} and 200 × 100, and on SymmetricSparseIterate arguments at
# p = 200 with 200, 2000 and 20100 active coordinates (20100 is every coordinate).
#
#   PB_ARCHIVE=<root of the archived tree> PB_CANDIDATE=<root of the candidate> PB_OUTPUT=<dir> \
#   PB_ROUNDS=11 julia --threads=1 --project=benchmark benchmark/kronquadratic.jl
#
# The trees are loaded with Base.include into their own modules, as benchmark/run.jl does, in the order archive,
# candidate, control (PB_REVERSE=1: candidate, archive, control). An archive that does not load, or that has no
# KronQuadratic, is reported and its cells are timed without it. Each round times the roles of every case in
# alternating order (reversed in even rounds) and keeps per role the median and the minimum over the samples of
# its trial. A case cycles through COPIES distinct arguments, so that consecutive calls do not repeat their input;
# the roles share the dense arguments, the data and the output, and build their own iterates from the same
# coordinates and values. The summary gives per case the median over the rounds of the ratios candidate/archive
# and control/candidate (A/A) of the trial minima and medians, and the warmed allocations of each role.
# PB_QUICK=1 runs small sizes for one round; PB_FILTER selects cases by a regular expression on their ids;
# PB_CHECK_ONLY=1 compares the outputs of the roles and exits before timing.
using BenchmarkTools, LinearAlgebra, Random, Statistics, Dates, Printf
Threads.nthreads() == 1 || error("one Julia thread required")
BLAS.set_num_threads(1)
candidate_root = abspath(ENV["PB_CANDIDATE"])
archive_root = haskey(ENV, "PB_ARCHIVE") ? abspath(ENV["PB_ARCHIVE"]) : nothing
reversed = get(ENV, "PB_REVERSE", "0") == "1"
quick = get(ENV, "PB_QUICK", "0") == "1"
module Archive end
module Candidate end
module CandidateControl end

function load(m, root)
  try
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    isdefined(m.ProximalBase, :KronQuadratic) || return "no KronQuadratic in $root"
    nothing
  catch err
    sprint(showerror, err)[1:min(end, 2000)]
  end
end
archive_error = archive_root === nothing ? "PB_ARCHIVE not set" : nothing
if reversed
  load(Candidate, candidate_root) === nothing || error("the candidate does not load")
  archive_error === nothing && (archive_error = load(Archive, archive_root))
else
  archive_error === nothing && (archive_error = load(Archive, archive_root))
  load(Candidate, candidate_root) === nothing || error("the candidate does not load")
end
load(CandidateControl, candidate_root) === nothing || error("the control does not load")
archive_error === nothing || println("archive skipped: ", archive_error)
const ROLES = archive_error === nothing ? (:archive, :candidate, :control) : (:candidate, :control)
const MODULES = Dict(:archive => Archive, :candidate => Candidate, :control => CandidateControl)
pb(role) = getfield(MODULES[role], :ProximalBase)

const COPIES = 8
const DENSE = quick ? ((24, 24), (40, 20)) : ((50, 50), (200, 200), (800, 800), (200, 100))
const SYMMETRIC_P = quick ? 40 : 200
const SYMMETRIC_N = quick ? (40, 200, 820) : (200, 2000, 20100)

symmetric_data(rng, ::Type{T}, p) where {T} = (M = randn(rng, T, p, p) / sqrt(T(p)); Matrix(Symmetric(M * M' + I)))

# one case: per role a function that runs the next call of the cycle and returns its value, and a check
struct Case
  id::String
  calls::Dict{Symbol, Function}
  results::Dict{Symbol, Function}
end

function dense_cases(::Type{T}, rng) where {T}
  cases = Case[]
  for (p, q) in DENSE
    A, B, C = symmetric_data(rng, T, p), symmetric_data(rng, T, q), randn(rng, T, p, q)
    Xs = [randn(rng, T, p, q) for _ in 1:COPIES]
    G = zeros(T, p, q)
    fs = Dict(r => pb(r).KronQuadratic(A, B, C) for r in ROLES)
    for op in (:value, :value_and_gradient!)
      calls, results = Dict{Symbol, Function}(), Dict{Symbol, Function}()
      for r in ROLES
        f, vg, i = fs[r], pb(r).value_and_gradient!, Ref(0)
        calls[r] = op === :value ? (() -> (i[] = mod1(i[] + 1, COPIES); f(Xs[i[]]))) :
                                   (() -> (i[] = mod1(i[] + 1, COPIES); vg(G, f, Xs[i[]])))
        results[r] = op === :value ? (() -> f(Xs[1])) : (() -> (v = vg(G, f, Xs[1]); (v, copy(G))))
      end
      push!(cases, Case("$T/dense/$(op)/$(p)x$(q)", calls, results))
    end
  end
  cases
end

function symmetric_cases(::Type{T}, rng) where {T}
  cases = Case[]
  p = SYMMETRIC_P
  n = p * (p + 1) ÷ 2
  A, B, C = symmetric_data(rng, T, p), symmetric_data(rng, T, p), symmetric_data(rng, T, p)
  G = zeros(T, p, p)
  fs = Dict(r => pb(r).KronQuadratic(A, B, C) for r in ROLES)
  for nactive in SYMMETRIC_N
    entries = [(randperm(rng, n)[1:nactive], randn(rng, T, nactive)) for _ in 1:COPIES]
    function iterate_of(r, (ks, vs))
      X = pb(r).SymmetricSparseIterate(T, p)
      for (k, v) in zip(ks, vs)
        i, j = pb(r).tril_cartesian(p, k)
        X[i, j] = v
      end
      X
    end
    Xs = Dict(r => [iterate_of(r, e) for e in entries] for r in ROLES)
    for op in (:value, :value_and_gradient!)
      calls, results = Dict{Symbol, Function}(), Dict{Symbol, Function}()
      for r in ROLES
        f, X, vg, i = fs[r], Xs[r], pb(r).value_and_gradient!, Ref(0)
        calls[r] = op === :value ? (() -> (i[] = mod1(i[] + 1, COPIES); f(X[i[]]))) :
                                   (() -> (i[] = mod1(i[] + 1, COPIES); vg(G, f, X[i[]])))
        results[r] = op === :value ? (() -> f(X[1])) : (() -> (v = vg(G, f, X[1]); (v, copy(G))))
      end
      push!(cases, Case("$T/symmetric/$(op)/p$(p)/nactive$(nactive)", calls, results))
    end
  end
  cases
end

cases = vcat(dense_cases(Float32, MersenneTwister(1059)), dense_cases(Float64, MersenneTwister(1059)),
             symmetric_cases(Float32, MersenneTwister(1060)), symmetric_cases(Float64, MersenneTwister(1060)))
if haskey(ENV, "PB_FILTER")
  selection = Regex(ENV["PB_FILTER"])
  filter!(c -> occursin(selection, c.id), cases)
end

# outputs of the roles: the control equals the candidate; the archive agrees within a working-precision tolerance
close_enough(a::Number, b::Number, T) = isapprox(a, b; rtol = 1000eps(T))
close_enough(a::Tuple, b::Tuple, T) = close_enough(a[1], b[1], T) && isapprox(a[2], b[2]; rtol = 1000eps(T))
for c in cases
  T = startswith(c.id, "Float32") ? Float32 : Float64
  rc = c.results[:candidate]()
  isequal(rc, c.results[:control]()) || error("A/A mismatch: $(c.id)")
  :archive in ROLES && (close_enough(c.results[:archive](), rc, T) || error("archive mismatch: $(c.id)"))
end
println("output checks pass: $(length(cases)) cases, roles $(join(ROLES, ", "))")
get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()

function trial(call)
  b = @benchmarkable $call() evals = 1 samples = (quick ? 20 : 200) seconds = (quick ? 0.05 : 0.5)
  t = run(b)
  (median(t.times), minimum(t.times))
end
warmed_allocs(call) = (call(); call(); @allocated call())

rounds = quick ? 1 : parse(Int, get(ENV, "PB_ROUNDS", "11"))
order = reversed && :archive in ROLES ? (:candidate, :archive, :control) : ROLES
raw = IOBuffer()
println(raw, "case\tround\trole\tmedian_ns\tmin_ns")
allocs = Dict((c.id, r) => warmed_allocs(c.calls[r]) for c in cases for r in ROLES)
foreach(c -> foreach(r -> trial(c.calls[r]), ROLES), cases)       # compile and warm every timed call
times = Dict{Tuple{String, Symbol}, Vector{Tuple{Float64, Float64}}}()
for round in 1:rounds, c in cases
  for r in (isodd(round) ? order : reverse(order))
    m, n = trial(c.calls[r])
    push!(get!(times, (c.id, r), Tuple{Float64, Float64}[]), (m, n))
    @printf(raw, "%s\t%d\t%s\t%.0f\t%.0f\n", c.id, round, r, m, n)
  end
end

ratio(a, b, k) = median([x[k] / y[k] for (x, y) in zip(times[a], times[b])])
summary = IOBuffer()
println(summary, "# julia $(VERSION); BLAS $(BLAS.get_config()); BLAS threads $(BLAS.get_num_threads()); ",
        "rounds $rounds; copies $COPIES; $(reversed ? "reversed" : "forward") order; $(now(UTC)) UTC")
archive_error === nothing || println(summary, "# archive skipped: ", replace(archive_error, '\n' => ' '))
println(summary, "case\tcandidate_median_ns\tcandidate_min_ns\tC/A_min\tC/A_median\tA/A_min\tA/A_median\t",
        "allocs_archive\tallocs_candidate\tallocs_control")
for c in cases
  cand = times[(c.id, :candidate)]
  ca = :archive in ROLES ? (ratio((c.id, :candidate), (c.id, :archive), 2), ratio((c.id, :candidate), (c.id, :archive), 1)) : (NaN, NaN)
  aa = (ratio((c.id, :control), (c.id, :candidate), 2), ratio((c.id, :control), (c.id, :candidate), 1))
  aarch = :archive in ROLES ? string(allocs[(c.id, :archive)]) : "-"
  @printf(summary, "%s\t%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%d\t%d\n", c.id, median(first.(cand)), median(last.(cand)),
          ca[1], ca[2], aa[1], aa[2], aarch, allocs[(c.id, :candidate)], allocs[(c.id, :control)])
end
print(String(take!(copy(summary))))
if haskey(ENV, "PB_OUTPUT")
  mkpath(ENV["PB_OUTPUT"])
  write(joinpath(ENV["PB_OUTPUT"], "raw.tsv"), take!(raw))
  write(joinpath(ENV["PB_OUTPUT"], "summary.tsv"), take!(summary))
end
