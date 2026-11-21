# Paired standalone comparison for PoissonLoss. See benchmark/README.md for usage.
using BenchmarkTools, LinearAlgebra, Printf, Random, Statistics

Threads.nthreads() == 1 || error("use one Julia thread")
BLAS.set_num_threads(1)
const SEED = 1130
const ROUNDS = parse(Int, get(ENV, "PB_ROUNDS", "7"))
const CANDIDATE = abspath(get(ENV, "PB_CANDIDATE", joinpath(@__DIR__, "..")))
const OUTPUT = abspath(get(ENV, "PB_OUTPUT", "poisson-benchmark"))
const ORDER_SEED = parse(Int, get(ENV, "PB_ORDER_SEED", string(SEED)))
const FIRST_CALL = get(ENV, "PB_FIRST_CALL", "0") == "1"
const FILTER = Regex(get(ENV, "PB_FILTER", ""))
ROUNDS >= 5 || error("at least five paired rounds required")

module Candidate end
Base.include(Candidate, joinpath(CANDIDATE, "src", "ProximalBase.jl"))

# Finite-data reference with preallocated buffers, the same products and the
# same scalar term as the candidate. It omits the argument checks of the
# candidate. Compile it independently in each role.
const REFERENCE = raw"""
using LinearAlgebra
function scan!(d, y, η, ::Val{V}) where {V}
  s = 0.0
  @inbounds for i in eachindex(η)
    e = exp(η[i])
    V && (s += ifelse(iszero(y[i]) | (e == Inf), e, e - y[i] * η[i]))
    d === nothing || (d[i] = e - y[i])
  end
  s
end

function ordinary!(g, tmp, X, y, β, value)
  mul!(tmp, X, β)
  s = scan!(g === nothing ? nothing : tmp, y, tmp, value)
  g === nothing || mul!(g, transpose(X), tmp)
  s
end

function cached_fused!(g, d, X, y, η)
  s = scan!(d, y, η, Val(true))
  mul!(g, transpose(X), d)
  s
end

calls(X, y, β, η, tmp, g, d) = (
  value=() -> ordinary!(nothing, tmp, X, y, β, Val(true)),
  gradient=() -> (ordinary!(g, tmp, X, y, β, Val(false)); g),
  fused=() -> ordinary!(g, tmp, X, y, β, Val(true)),
  cached_value=() -> scan!(nothing, y, η, Val(true)),
  cached_fused=() -> cached_fused!(g, d, X, y, η),
)
"""
module Reference end
module ReferenceControl end
Base.include_string(Reference, REFERENCE, "poisson-reference")
Base.include_string(ReferenceControl, REFERENCE, "poisson-reference-control")

function production_calls(PB, f, β, η, g, d)
  gradient!, fused! = PB.gradient!, PB.value_and_gradient!
  cached_value, cached_fused! = PB.value_from_predictor, PB.value_and_gradient_from_predictor!
  (value=() -> f(β),
   gradient=() -> gradient!(g, f, β),
   fused=() -> fused!(g, f, β),
   cached_value=() -> cached_value(f, η),
   cached_fused=() -> cached_fused!(g, d, f, η))
end

function warm_trial(call, evals)
  bench = @benchmarkable $call() evals=evals samples=35 seconds=0.4
  run(bench)
end

function measure!(io, id, calls, outputs, evals, rng)
  occursin(FILTER, id) || return
  roles = ("reference", "candidate", "control")
  if FIRST_CALL
    for j in randperm(rng, 3)
      stats = @timed Base.invokelatest(calls[j])
      println(io, join((id, roles[j], stats.time, stats.bytes), ','))
    end
    return
  end
  results = map(call -> call(), calls)
  @assert results[1] == results[3] id
  @assert isapprox(results[1], results[2]; rtol=128eps(Float64)) id
  for (ref, candidate, control) in outputs
    @assert ref == control id
    @assert isapprox(ref, candidate; rtol=128eps(Float64)) id
  end
  foreach(call -> warm_trial(call, evals), calls)
  order = randperm(rng, 3)
  for round in 1:ROUNDS
    # Rotate roles and reverse the rotation on alternate groups of three.
    traversal = circshift(order, (round - 1) % 3)
    isodd((round - 1) ÷ 3) && reverse!(traversal)
    for j in traversal
      t = warm_trial(calls[j], evals)
      println(io, join((id, round, roles[j], minimum(t).time, median(t).time,
                       t.memory, t.allocs), ','))
    end
  end
  flush(io)
end

function poisson_cases!(io, rng, order_rng, shape, X::Matrix{Float64})
  n, p = size(X)
  y = Float64.(rand(rng, 0:5, n))
  β = randn(rng, p) ./ sqrt(p)
  η = X * β
  PB = Candidate.ProximalBase
  f = PB.PoissonLoss(y, X)
  gs = ntuple(_ -> similar(β), 3)
  ds = ntuple(_ -> similar(y), 3)
  ref = Reference.calls(X, y, β, η, similar(y), gs[1], ds[1])
  cand = production_calls(PB, f, β, η, gs[2], ds[2])
  ctl = ReferenceControl.calls(X, y, β, η, similar(y), gs[3], ds[3])
  for op in keys(ref)
    id = "poisson/$op/Float64/$shape"
    outputs = op === :cached_fused ? (gs, ds) : op in (:gradient, :fused) ? (gs,) : ()
    measure!(io, id, (ref[op], cand[op], ctl[op]), outputs, n * p <= 10^4 ? 30 : 1, order_rng)
  end
end

mkpath(OUTPUT)
open(joinpath(OUTPUT, "provenance.txt"), "w") do io
  println(io, "candidate ", readchomp(`git -C $CANDIDATE rev-parse HEAD`))
  println(io, "Julia ", VERSION, " | ", Sys.CPU_NAME, " | ", BLAS.get_config())
  println(io, "Julia threads ", Threads.nthreads(), " | BLAS threads ", BLAS.get_num_threads())
  println(io, "seed ", SEED, " | order seed ", ORDER_SEED, " | rounds ", ROUNDS, " | filter ", FILTER, " | first call ", FIRST_CALL)
end
open(joinpath(OUTPUT, FIRST_CALL ? "first-call.csv" : "raw.csv"), "w") do io
  println(io, FIRST_CALL ? "case,role,seconds,bytes" : "case,round,role,min_ns,median_ns,bytes,allocs")
  rng = MersenneTwister(SEED)
  order_rng = MersenneTwister(ORDER_SEED)
  for (shape, n, p) in (("n200_p20", 200, 20), ("n2000_p100", 2000, 100),
                        ("n10000_p50", 10000, 50), ("n500_p2000", 500, 2000))
    poisson_cases!(io, rng, order_rng, shape, randn(rng, n, p))
  end
end
