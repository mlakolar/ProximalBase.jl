# Paired benchmark of KronQuadratic (#1059) between the archived source tree (eff21b6d), the candidate, a second
# compilation of the candidate (the A/A control) and a straightforward dense implementation written in this script
# (the dense role), Float32 and Float64.
#
# Default cells: the value and value_and_gradient! on dense p × q matrices, p = q ∈ {50, 200, 800} and 200 × 100, and
# on SymmetricSparseIterate arguments at p = 200 with 200, 2000 and 20100 active coordinates (20100 is every
# coordinate). PB_SWEEP=1 runs the symmetric sweep instead: p ∈ {50, 200, 800} and nonzero counts from p/2 to every
# coordinate (sweep_counts), one cell at a time; PB_ROUNDS_LARGE sets the rounds at p = 800.
#
#   PB_ARCHIVE=<root of the archived tree> PB_CANDIDATE=<root of the candidate> PB_OUTPUT=<dir> \
#   PB_ROUNDS=11 julia --threads=1 --project=benchmark benchmark/kronquadratic.jl
#
# The trees are loaded with Base.include into their own modules, as benchmark/run.jl does, in the order archive,
# candidate, control (PB_REVERSE=1: candidate, archive, control, also the timing order of odd rounds). An archive
# that does not load, or that does not bind ProximalBase to a module that defines KronQuadratic, is reported and its
# cells are timed without it. The dense role copies a symmetric iterate into a dense matrix through the public
# accessors of the candidate tree, forms A D and (A D) B with mul!, and takes the value dot(D, W)/2 + dot(C, D) and the
# gradient (W + Wᵀ)/2 + C; for a matrix argument it forms A X and (A X) B. Each round times the roles of each case in
# alternating order (reversed in even rounds) and keeps per role the median and the minimum over the samples of its
# trial. A case cycles through COPIES distinct arguments, so that consecutive calls do not repeat their input; the
# roles share the dense arguments, the data and the output, and build their own iterates from the same coordinates
# and values (the dense role reads the candidate's). The summary gives per case the route of the candidate (matrix,
# accumulate or dense), the median over the rounds of the ratios candidate/dense, candidate/archive and
# control/candidate (A/A) of the trial minima and medians, and the warmed allocations of each role.
# PB_QUICK=1 runs small sizes for one round; PB_FILTER selects cases by a regular expression on their ids;
# PB_CHECK_ONLY=1 compares the outputs of the roles at the first argument of each case and exits before timing.
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

# The include only; the bindings it creates are read at top level, in the latest world.
function load(m, root)
  try
    Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
    nothing
  catch err
    first(sprint(showerror, err), 2000)
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
# whether a loaded tree defines the module ProximalBase with KronQuadratic
has_kron(m) = isdefined(m, :ProximalBase) && getfield(m, :ProximalBase) isa Module &&
  isdefined(getfield(m, :ProximalBase), :KronQuadratic)
archive_error === nothing && !has_kron(Archive) && (archive_error = "no ProximalBase.KronQuadratic in $archive_root")
has_kron(Candidate) || error("the candidate has no ProximalBase.KronQuadratic")
has_kron(CandidateControl) || error("the control has no ProximalBase.KronQuadratic")
archive_error === nothing || println("archive skipped: ", archive_error)
const PB_ROLES = archive_error === nothing ? (:archive, :candidate, :control) : (:candidate, :control)
# The dense role is written in this script: it copies a symmetric iterate into a dense matrix and forms A D and
# (A D) B with mul!, with buffers allocated once per case. It reads iterates of the candidate tree.
const ROLES = (PB_ROLES..., :dense)
const MODULES = Dict(:archive => Archive, :candidate => Candidate, :control => CandidateControl, :dense => Candidate)
pb(role) = getfield(MODULES[role], :ProximalBase)

sweep = get(ENV, "PB_SWEEP", "0") == "1"
const COPIES = 8
const DENSE = quick ? ((24, 24), (40, 20)) : ((50, 50), (200, 200), (800, 800), (200, 100))
const SYMMETRIC_P = quick ? 40 : 200
const SYMMETRIC_N = quick ? (40, 200, 820) : (200, 2000, 20100)
const SWEEP_P = quick ? (24, 40) : (50, 200, 800)

symmetric_data(rng, ::Type{T}, p) where {T} = (M = randn(rng, T, p, p) / sqrt(T(p)); Matrix(Symmetric(M * M' + I)))

#  the dense role
struct DenseKron{T}
  A::Matrix{T}
  B::Matrix{T}
  C::Matrix{T}
  D::Matrix{T}
  AX::Matrix{T}
  W::Matrix{T}
end
DenseKron(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}) where {T} =
  DenseKron(A, B, C, zeros(T, size(C)), zeros(T, size(C)), zeros(T, size(C)))
function dense_value(d::DenseKron, X::AbstractMatrix)
  mul!(d.AX, d.A, X); mul!(d.W, d.AX, d.B)
  dot(X, d.W) / 2 + dot(d.C, X)
end
function dense_value_and_gradient!(G, d::DenseKron, X::AbstractMatrix)
  mul!(d.AX, d.A, X); mul!(G, d.AX, d.B)
  v = dot(X, G) / 2 + dot(d.C, X)
  G .+= d.C
  v
end
# the matrix of a symmetric iterate through the public accessors of its tree (passed as ac, av, tc)
function densify!(D, X, ac::F1, av::F2, tc::F3) where {F1, F2, F3}
  fill!(D, 0)
  p = size(D, 1)
  ids, vals = ac(X), av(X)
  @inbounds for a in eachindex(vals)
    r, c = tc(p, ids[a])
    v = vals[a]
    D[r, c] = v
    D[c, r] = v
  end
  D
end
function dense_symmetric_value(d::DenseKron, X, acc)
  densify!(d.D, X, acc...)
  mul!(d.AX, d.A, d.D); mul!(d.W, d.AX, d.B)
  dot(d.D, d.W) / 2 + dot(d.C, d.D)
end
function dense_symmetric_value_and_gradient!(G, d::DenseKron, X, acc)
  v = dense_symmetric_value(d, X, acc)
  W, C = d.W, d.C
  @inbounds for j in axes(G, 2), i in axes(G, 1)
    G[i, j] = (W[i, j] + W[j, i]) / 2 + C[i, j]
  end
  v
end

# the calls of the dense role, built in their own functions so that each closure captures its own counter
function dense_matrix_call(op, d, Xs, G)
  i = Ref(0)
  op === :value ? (() -> (i[] = mod1(i[] + 1, COPIES); dense_value(d, Xs[i[]]))) :
                  (() -> (i[] = mod1(i[] + 1, COPIES); dense_value_and_gradient!(G, d, Xs[i[]])))
end
function dense_symmetric_call(op, d, Xs, G, acc)
  i = Ref(0)
  op === :value ? (() -> (i[] = mod1(i[] + 1, COPIES); dense_symmetric_value(d, Xs[i[]], acc))) :
                  (() -> (i[] = mod1(i[] + 1, COPIES); dense_symmetric_value_and_gradient!(G, d, Xs[i[]], acc)))
end
dense_symmetric_result(op, d, Xs, G, acc) = op === :value ? (() -> dense_symmetric_value(d, Xs[1], acc)) :
  (() -> (v = dense_symmetric_value_and_gradient!(G, d, Xs[1], acc); (v, copy(G))))

# one case: per role a function that runs the next call of the cycle and returns its value, and a check
struct Case
  id::String
  route::String
  calls::Dict{Symbol, Function}
  results::Dict{Symbol, Function}
end

function dense_cases(::Type{T}, rng) where {T}
  cases = Case[]
  for (p, q) in DENSE
    A, B, C = symmetric_data(rng, T, p), symmetric_data(rng, T, q), randn(rng, T, p, q)
    Xs = [randn(rng, T, p, q) for _ in 1:COPIES]
    G = zeros(T, p, q)
    fs = Dict{Symbol, Any}(r => pb(r).KronQuadratic(A, B, C) for r in PB_ROLES)
    d = DenseKron(A, B, C)
    for op in (:value, :value_and_gradient!)
      calls, results = Dict{Symbol, Function}(), Dict{Symbol, Function}()
      for r in PB_ROLES
        f, vg, i = fs[r], pb(r).value_and_gradient!, Ref(0)
        calls[r] = op === :value ? (() -> (i[] = mod1(i[] + 1, COPIES); f(Xs[i[]]))) :
                                   (() -> (i[] = mod1(i[] + 1, COPIES); vg(G, f, Xs[i[]])))
        results[r] = op === :value ? (() -> f(Xs[1])) : (() -> (v = vg(G, f, Xs[1]); (v, copy(G))))
      end
      calls[:dense] = dense_matrix_call(op, d, Xs, G)
      results[:dense] = op === :value ? (() -> dense_value(d, Xs[1])) :
                                        (() -> (v = dense_value_and_gradient!(G, d, Xs[1]); (v, copy(G))))
      push!(cases, Case("$T/dense/$(op)/$(p)x$(q)", "matrix", calls, results))
    end
  end
  cases
end

# the value and value_and_gradient! cases of one symmetric cell: COPIES iterates with n nonzero active values
function symmetric_pair(::Type{T}, rng, p, n, A, B, C, label) where {T}
  q = p * (p + 1) ÷ 2
  G = zeros(T, p, p)
  fs = Dict{Symbol, Any}(r => pb(r).KronQuadratic(A, B, C) for r in PB_ROLES)
  d = DenseKron(A, B, C)
  entries = [(randperm(rng, q)[1:n], randn(rng, T, n)) for _ in 1:COPIES]
  function iterate_of(r, (ks, vs))
    X = pb(r).SymmetricSparseIterate(T, p)
    for (k, v) in zip(ks, vs)
      i, j = pb(r).tril_cartesian(p, k)
      X[i, j] = v
    end
    X
  end
  Xs = Dict{Symbol, Any}(r => [iterate_of(r, e) for e in entries] for r in PB_ROLES)
  Xs[:dense] = Xs[:candidate]
  # the labels count active coordinates; a sampled value of zero would leave its coordinate inactive, which this checks
  all(all(X -> pb(r).nactive(X) == n, Xs[r]) for r in PB_ROLES) || error("an iterate has fewer than $n active coordinates")
  P = pb(:candidate)
  acc = (P.activecoordinates, P.activevalues, P.tril_cartesian)
  pair = Case[]
  for op in (:value, :value_and_gradient!)
    calls, results = Dict{Symbol, Function}(), Dict{Symbol, Function}()
    for r in PB_ROLES
      f, X, vg, i = fs[r], Xs[r], pb(r).value_and_gradient!, Ref(0)
      calls[r] = op === :value ? (() -> (i[] = mod1(i[] + 1, COPIES); f(X[i[]]))) :
                                 (() -> (i[] = mod1(i[] + 1, COPIES); vg(G, f, X[i[]])))
      results[r] = op === :value ? (() -> f(X[1])) : (() -> (v = vg(G, f, X[1]); (v, copy(G))))
    end
    calls[:dense] = dense_symmetric_call(op, d, Xs[:dense], G, acc)
    results[:dense] = dense_symmetric_result(op, d, Xs[:dense], G, acc)
    # the route the candidate takes for this call (the cutoff of value-only calls is k = 32, of gradient calls 16)
    route = P._kron_dense_route(Xs[:candidate][1], p, op === :value ? 32 : 16) ? "dense" : "accumulate"
    push!(pair, Case("$T/symmetric/$(op)/p$(p)/$label", route, calls, results))
  end
  pair
end

function symmetric_cases(::Type{T}, rng) where {T}
  p = SYMMETRIC_P
  A, B, C = symmetric_data(rng, T, p), symmetric_data(rng, T, p), symmetric_data(rng, T, p)
  reduce(vcat, [symmetric_pair(T, rng, p, n, A, B, C, "nactive$n") for n in SYMMETRIC_N])
end

# the nonzero counts of the sweep at p: from p/2 to every coordinate, with counts on both sides of the cutoffs of the
# candidate (2n = p²/16 for value-only calls and 2n = p²/8 for gradient calls)
function sweep_counts(p)
  q = p * (p + 1) ÷ 2
  base = (p / 2, p, 2p, 5p, 10p, 0.02q, 0.05q, 0.1q, 0.25q, q, 0.9p^2 / 32, 1.1p^2 / 32, 0.9p^2 / 16, 1.1p^2 / 16)
  sort!(unique(clamp.(round.(Int, base), 1, q)))
end

# outputs of the roles at the first argument of each case: the control equals the candidate (isequal); the archive and
# the dense role agree within a working-precision tolerance (isapprox, a norm-relative test for the gradient)
close_enough(a::Number, b::Number, T) = isapprox(a, b; rtol = 1000eps(T))
close_enough(a::Tuple, b::Tuple, T) = close_enough(a[1], b[1], T) && isapprox(a[2], b[2]; rtol = 1000eps(T))
function check(c::Case)
  T = startswith(c.id, "Float32") ? Float32 : Float64
  rc = c.results[:candidate]()
  isequal(rc, c.results[:control]()) || error("A/A mismatch: $(c.id)")
  :archive in ROLES && (close_enough(c.results[:archive](), rc, T) || error("archive mismatch: $(c.id)"))
  close_enough(c.results[:dense](), rc, T) || error("dense-role mismatch: $(c.id)")
end

function trial(call, large)
  b = @benchmarkable $call() evals = 1 samples = (quick ? 20 : 200) seconds = (quick ? 0.05 : large ? 1.0 : 0.5)
  t = run(b)
  (median(t.times), minimum(t.times))
end
warmed_allocs(call) = (call(); call(); @allocated call())

rounds = quick ? 1 : parse(Int, get(ENV, "PB_ROUNDS", "11"))
rounds_large = quick ? 1 : parse(Int, get(ENV, "PB_ROUNDS_LARGE", string(rounds)))
order = reversed && :archive in ROLES ? (:candidate, :archive, :control, :dense) : ROLES
raw = IOBuffer()
println(raw, "case\tround\trole\tmedian_ns\tmin_ns")
summary = IOBuffer()
println(summary, "# julia $(VERSION); BLAS $(BLAS.get_config()); BLAS threads $(BLAS.get_num_threads()); CPU $(Sys.CPU_NAME); ",
        "rounds $rounds (p ≥ 800: $rounds_large); copies $COPIES; $(reversed ? "reversed" : "forward") order; ",
        "$(sweep ? "sweep" : "default cells"); $(now(UTC)) UTC")
archive_error === nothing || println(summary, "# archive skipped: ", replace(archive_error, '\n' => ' '))
println(summary, "case\troute\tcandidate_median_ns\tcandidate_min_ns\tC/D_min\tC/D_median\tC/A_min\tC/A_median\t",
        "A/A_min\tA/A_median\tallocs_archive\tallocs_candidate\tallocs_control\tallocs_dense")

# Times the cases (each round times the roles of each case in alternating order) and appends their summary rows.
function run_cases(cases, rounds, large)
  allocs = Dict((c.id, r) => warmed_allocs(c.calls[r]) for c in cases for r in ROLES)
  foreach(c -> foreach(r -> trial(c.calls[r], large), ROLES), cases)       # compile and warm every timed call
  times = Dict{Tuple{String, Symbol}, Vector{Tuple{Float64, Float64}}}()
  for round in 1:rounds, c in cases
    for r in (isodd(round) ? order : reverse(order))
      m, n = trial(c.calls[r], large)
      push!(get!(times, (c.id, r), Tuple{Float64, Float64}[]), (m, n))
      @printf(raw, "%s\t%d\t%s\t%.0f\t%.0f\n", c.id, round, r, m, n)
    end
  end
  ratio(a, b, k) = median([x[k] / y[k] for (x, y) in zip(times[(a, b[1])], times[(a, b[2])])])
  rows = IOBuffer()
  for c in cases
    cand = times[(c.id, :candidate)]
    cd = (ratio(c.id, (:candidate, :dense), 2), ratio(c.id, (:candidate, :dense), 1))
    ca = :archive in ROLES ? (ratio(c.id, (:candidate, :archive), 2), ratio(c.id, (:candidate, :archive), 1)) : (NaN, NaN)
    aa = (ratio(c.id, (:control, :candidate), 2), ratio(c.id, (:control, :candidate), 1))
    aarch = :archive in ROLES ? string(allocs[(c.id, :archive)]) : "-"
    @printf(rows, "%s\t%s\t%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%d\t%d\t%d\n", c.id, c.route,
            median(first.(cand)), median(last.(cand)), cd..., ca..., aa..., aarch,
            allocs[(c.id, :candidate)], allocs[(c.id, :control)], allocs[(c.id, :dense)])
  end
  text = String(take!(rows))
  print(text); flush(stdout)
  print(summary, text)
  nothing
end
selected(c) = !haskey(ENV, "PB_FILTER") || occursin(Regex(ENV["PB_FILTER"]), c.id)
print(String(take!(copy(summary))))       # the header; each run_cases call prints its rows

if sweep
  # one (T, p, n) cell at a time, so that the iterates of a cell are freed before the next one is built
  ncases = 0
  for T in (Float32, Float64), p in SWEEP_P
    rng = MersenneTwister(1061 + p)
    A, B, C = symmetric_data(rng, T, p), symmetric_data(rng, T, p), symmetric_data(rng, T, p)
    for n in sweep_counts(p)
      cell = filter(selected, symmetric_pair(T, rng, p, n, A, B, C, "n$n"))
      isempty(cell) && continue
      foreach(check, cell)
      get(ENV, "PB_CHECK_ONLY", "0") == "1" && continue
      global ncases += length(cell)
      run_cases(cell, p >= 800 ? rounds_large : rounds, p >= 800)
      GC.gc()
    end
  end
  println("sweep: $ncases cases, roles $(join(ROLES, ", "))")
else
  cases = filter(selected, vcat(dense_cases(Float32, MersenneTwister(1059)), dense_cases(Float64, MersenneTwister(1059)),
                                symmetric_cases(Float32, MersenneTwister(1060)), symmetric_cases(Float64, MersenneTwister(1060))))
  foreach(check, cases)
  println("output checks pass: $(length(cases)) cases, roles $(join(ROLES, ", "))")
  get(ENV, "PB_CHECK_ONLY", "0") == "1" && exit()
  run_cases(cases, rounds, false)
end
if haskey(ENV, "PB_OUTPUT")
  mkpath(ENV["PB_OUTPUT"])
  write(joinpath(ENV["PB_OUTPUT"], "raw.tsv"), take!(raw))
  write(joinpath(ENV["PB_OUTPUT"], "summary.tsv"), take!(summary))
end
