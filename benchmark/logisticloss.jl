# Paired standalone comparison for LogisticLoss and source-to-source comparison
# for the existing least-squares oracles. See benchmark/README.md for usage.
using BenchmarkTools, LinearAlgebra, Printf, Random, SparseArrays, Statistics

Threads.nthreads() == 1 || error("use one Julia thread")
BLAS.set_num_threads(1)
const SEED = 1029
const ROUNDS = parse(Int, get(ENV, "PB_ROUNDS", "7"))
const BASE = abspath(ENV["PB_BASELINE"])
const CANDIDATE = abspath(get(ENV, "PB_CANDIDATE", joinpath(@__DIR__, "..")))
const OUTPUT = abspath(get(ENV, "PB_OUTPUT", "logistic-benchmark"))
const ORDER_SEED = parse(Int, get(ENV, "PB_ORDER_SEED", string(SEED)))
const FIRST_CALL = get(ENV, "PB_FIRST_CALL", "0") == "1"
const FILTER = Regex(get(ENV, "PB_FILTER", ""))
ROUNDS >= 5 || error("at least five paired rounds required")

module Baseline end
module BaselineControl end
module Candidate end
Base.include(Baseline, joinpath(BASE, "src", "ProximalBase.jl"))
Base.include(BaselineControl, joinpath(BASE, "src", "ProximalBase.jl"))
Base.include(Candidate, joinpath(CANDIDATE, "src", "ProximalBase.jl"))

# Ordinary finite-data baseline with preallocated buffers and the same products.
# Compile it independently in each role. The baseline omits API preflight and
# the candidate's exceptional weighted-tail handling.
const REFERENCE = raw"""
using LinearAlgebra
function scan!(d, y, w, η, ::Val{V}) where {V}
  s = zero(eltype(y))
  @inbounds for i in eachindex(η)
    wi = w === nothing ? one(s) : w[i]
    if iszero(wi)
      d === nothing || (d[i] = zero(s))
      continue
    end
    t = η[i]
    e = exp(-abs(t))
    c = t > 0 ? one(s) - y[i] : y[i]
    V && (s += wi * (c * abs(t) + log1p(e)))
    if d !== nothing
      q = e / (one(s) + e)
      d[i] = wi * (t > 0 ? c - q : q - c)
    end
  end
  s
end

function ordinary!(g, tmp, X, y, w, offset, β, value)
  mul!(tmp, X, β)
  offset === nothing || (tmp .+= offset)
  s = scan!(g === nothing ? nothing : tmp, y, w, tmp, value)
  g === nothing || mul!(g, transpose(X), tmp)
  s
end

function cached_fused!(g, d, X, y, w, η)
  s = scan!(d, y, w, η, Val(true))
  mul!(g, transpose(X), d)
  s
end

calls(X, y, w, offset, β, η, tmp, g, d) = (
  value=() -> ordinary!(nothing, tmp, X, y, w, offset, β, Val(true)),
  gradient=() -> (ordinary!(g, tmp, X, y, w, offset, β, Val(false)); g),
  fused=() -> ordinary!(g, tmp, X, y, w, offset, β, Val(true)),
  cached_value=() -> scan!(nothing, y, w, η, Val(true)),
  cached_fused=() -> cached_fused!(g, d, X, y, w, η),
)
"""
module Reference end
module ReferenceControl end
Base.include_string(Reference, REFERENCE, "logistic-reference")
Base.include_string(ReferenceControl, REFERENCE, "logistic-reference-control")

function production_calls(PB, f, β, η, g, d)
  (value=() -> f(β),
   gradient=() -> PB.gradient!(g, f, β),
   fused=() -> PB.value_and_gradient!(g, f, β),
   cached_value=() -> PB.value_from_predictor(f, η),
   cached_fused=() -> PB.value_and_gradient_from_predictor!(g, d, f, η))
end

function warm_trial(call, evals)
  bench = @benchmarkable $call() evals=$evals samples=35 seconds=0.4
  run(bench)
end

function measure!(io, id, calls, outputs, ::Type{T}, evals, rng) where {T}
  occursin(FILTER, id) || return
  roles = ("reference", "candidate", "control")
  if FIRST_CALL
    for j in randperm(rng, 3)
      stats = @timed calls[j]()
      println(io, join((id, roles[j], stats.time, stats.bytes), ','))
    end
    return
  end
  results = map(call -> call(), calls)
  @assert results[1] == results[3] id
  @assert isapprox(results[1], results[2]; rtol=128eps(T)) id
  for (ref, candidate, control) in outputs
    @assert ref == control id
    @assert isapprox(ref, candidate; rtol=128eps(T)) id
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

function logistic_cases!(io, rng, order_rng, shape, X::AbstractMatrix{T}) where {T}
  n, p = size(X)
  y = T.(rand(rng, (0.0, 0.25, 0.5, 1.0), n))
  β = randn(rng, T, p) ./ sqrt(T(p))
  w = T(2) .* rand(rng, T, n)
  w[1:19:end] .= zero(T)
  offset = randn(rng, T, n) ./ T(4)
  PB = Candidate.ProximalBase
  for (kind, W, O) in (("raw", nothing, nothing), ("weights", w, nothing),
                       ("offset", nothing, offset), ("both", w, offset))
    η = X * β
    O === nothing || (η .+= O)
    f = PB.LogisticLoss(y, X; weights=W, offset=O)
    gs = ntuple(_ -> similar(β), 3)
    ds = ntuple(_ -> similar(y), 3)
    ref = Reference.calls(X, y, W, O, β, η, similar(y), gs[1], ds[1])
    cand = production_calls(PB, f, β, η, gs[2], ds[2])
    ctl = ReferenceControl.calls(X, y, W, O, β, η, similar(y), gs[3], ds[3])
    for op in keys(ref)
      id = "logistic/$op/$T/$kind/$shape"
      outputs = op === :cached_fused ? (gs, ds) : op in (:gradient, :fused) ? (gs,) : ()
      measure!(io, id, (ref[op], cand[op], ctl[op]), outputs, T, shape == "small" ? 30 : 1, order_rng)
    end
  end
end

function leastsquares_cases!(io, rng, order_rng, shape, X::Matrix{T}) where {T}
  n, p = size(X)
  y, β = randn(rng, T, n), randn(rng, T, p)
  η = X * β
  modules = (Baseline.ProximalBase, Candidate.ProximalBase, BaselineControl.ProximalBase)
  gs = ntuple(_ -> similar(β), 3)
  ds = ntuple(_ -> similar(y), 3)
  oracles = ntuple(i -> production_calls(modules[i], modules[i].LeastSquaresLoss(y, X),
                                        β, η, gs[i], ds[i]), 3)
  for op in keys(first(oracles))
    id = "leastsquares/$op/$T/raw/$shape"
    outputs = op === :cached_fused ? (gs, ds) : op in (:gradient, :fused) ? (gs,) : ()
    measure!(io, id, map(calls -> calls[op], oracles), outputs, T, shape == "small" ? 30 : 1, order_rng)
  end
end

mkpath(OUTPUT)
open(joinpath(OUTPUT, "provenance.txt"), "w") do io
  for (label, root) in (("master", BASE), ("candidate", CANDIDATE))
    println(io, label, " ", readchomp(`git -C $root rev-parse HEAD`))
  end
  println(io, "Julia ", VERSION, " | ", Sys.CPU_NAME, " | ", BLAS.get_config())
  println(io, "Julia threads ", Threads.nthreads(), " | BLAS threads ", BLAS.get_num_threads())
  println(io, "seed ", SEED, " | order seed ", ORDER_SEED, " | rounds ", ROUNDS, " | filter ", FILTER, " | first call ", FIRST_CALL)
end
open(joinpath(OUTPUT, FIRST_CALL ? "first-call.csv" : "raw.csv"), "w") do io
  println(io, FIRST_CALL ? "case,role,seconds,bytes" : "case,round,role,min_ns,median_ns,bytes,allocs")
  rng = MersenneTwister(SEED)
  order_rng = MersenneTwister(ORDER_SEED)
  for T in (Float32, Float64)
    for (shape, n, p) in (("small", 32, 8), ("tall", 32768, 32), ("wide", 256, 4096))
      X = randn(rng, T, n, p)
      logistic_cases!(io, rng, order_rng, shape, X)
      FIRST_CALL || leastsquares_cases!(io, rng, order_rng, shape, X)
    end
    X = sprandn(rng, T, 32768, 256, 0.01)
    logistic_cases!(io, rng, order_rng, "csc", X)
  end
end
