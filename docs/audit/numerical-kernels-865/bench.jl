# Paired benchmarks for issue #865: ordinary and exceptional paths of the migrated families.
#
#   julia --project=benchmark --threads=1 -O2 --check-bounds=no \
#     docs/audit/numerical-kernels-865/bench.jl BASELINE_TREE CANDIDATE_TREE [ROUNDS] [FILTER]
#
# Both source trees are loaded as modules in one process, see the same data, and are measured in
# interleaved rounds (A B, then B A, ...). Each measurement is the minimum time over a BenchmarkTools
# sample; the reported time is the median over rounds, and the ratio is the median of the per-round
# candidate/baseline ratios. Allocations are bytes per call after warm-up.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Printf
BLAS.set_num_threads(1)

const TREES = abspath.(ARGS[1:2])
const ROUNDS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 7
const FILTER = length(ARGS) >= 4 ? Regex(ARGS[4]) : r""
module Baseline end
module Candidate end
Base.include(Baseline, joinpath(TREES[1], "src", "ProximalBase.jl"))
Base.include(Candidate, joinpath(TREES[2], "src", "ProximalBase.jl"))

# prox_block is passed as a function value: a module-qualified call through a module held in a
# variable would dispatch dynamically and allocate
function block_loop(pb::P, f, x, γ) where {P}
  s = zero(eltype(x))
  @inbounds for i in eachindex(x)
    s += pb(f, x[i], i, γ)
  end
  s
end

# Plain arrays are shared by both trees, so that the two see the same memory: a separate allocation
# per tree changes the relative placement of input and output arrays, and streaming loops then differ
# by up to 1.8x from cache-set aliasing alone, whatever the code. Only module-specific objects
# (function objects, iterates) are built per tree.
function shared(::Type{T}) where {T}
  rng = Xoshiro(865)
  n = 10_000
  E = exponent(floatmax(T))
  h = E ÷ 2 + 4
  x = randn(rng, T, n)
  c = randn(rng, T, n)
  X = randn(rng, T, 500, 200)
  β = randn(rng, T, 200)
  top = fill(ldexp(T(0.75), E + 1), n)
  (; n, E, h, x, c, c64 = randn(rng, Float64, n), g = similar(x), z = similar(x),
     sym = [(rand(rng, 1:200), rand(rng, 1:200), randn(rng, T)) for _ in 1:1000],
     big_x = x .* ldexp(one(T), h),                                   # squares overflow
     tiny_x = x .* ldexp(one(T), exponent(floatmin(T)) ÷ 2 - 4),      # squares underflow
     top, negtop = -top,                                              # x - c overflows
     X, β, Y = X * β .+ randn(rng, T, 500), gβ = similar(β),
     Xo = hcat(fill(ldexp(T(0.75), h), 500), fill(ldexp(T(0.75), h), 500), randn(rng, T, 500)),
     Yo = randn(rng, T, 500),
     βo = T[ldexp(T(0.75), h), -ldexp(T(0.75), h), 1],                # partial products of Xβ overflow
     cbig = fill(floatmax(T) / 4, n), alt = T[isodd(i) ? 1 : -1 for i in 1:n],
     absc = abs.(c), wx = abs.(c) .* ldexp(one(T), -E + 4))
end

function cases(PB, D)
  (; n, E, h, x, c, c64, g, z, big_x, tiny_x, top, negtop, X, β, Y, gβ, Xo, Yo, βo, cbig, alt, absc, wx) = D
  T = eltype(x)
  xs = PB.SparseIterate(T, n)
  for i in 1:10:n
    xs[i] = x[i]
  end
  xm = PB.SymmetricSparseIterate(T, 200)
  for (r, cc, v) in D.sym
    xm[max(r, cc), min(r, cc)] = v
  end
  f0, fc = PB.SqrNormL2(T(0.7)), PB.SqrNormL2(T(0.7), c)
  fbig = PB.SqrNormL2(ldexp(one(T), -E + 2))
  ftiny = PB.SqrNormL2(ldexp(one(T), E - 4))
  ftop = PB.SqrNormL2(ldexp(one(T), -E + 2), negtop)
  ls = PB.LeastSquaresLoss(Y, X)
  lsbig = PB.LeastSquaresLoss(Y .* ldexp(one(T), h), X)               # residual squares overflow
  lso = PB.LeastSquaresLoss(Yo, Xo)
  lin = PB.LinearFunction(c, T(0.5))
  linbig = PB.LinearFunction(cbig, T(0.5))
  l1 = PB.NormL1(T(0.3))
  l1w = PB.NormL1(T(0.3), absc)
  γhuge = ldexp(one(T), E - 1)
  l1x = PB.NormL1(ldexp(one(T), E - 1), wx)                            # γλ overflows, the weight restores it
  [
    "ridge/value/dense" => () -> f0(x),
    "ridge/value/centered" => () -> fc(x),
    "ridge/value/sparse" => () -> f0(xs),
    "ridge/value/symmetric" => () -> f0(xm),
    "ridge/gradient/dense" => () -> PB.gradient!(g, f0, x),
    "ridge/gradient/centered" => () -> PB.gradient!(g, fc, x),
    "ridge/value_and_gradient/centered" => () -> PB.value_and_gradient!(g, fc, x),
    "ridge/prox/dense" => () -> PB.prox!(z, f0, x, T(0.3)),
    "ridge/prox/centered" => () -> PB.prox!(z, fc, x, T(0.3)),
    "ridge/prox_block/dense" => () -> block_loop(PB.prox_block, f0, x, T(0.3)),
    "ridge/prox_block/centered" => () -> block_loop(PB.prox_block, fc, x, T(0.3)),
    (T === Float32 ? ["ridge/value/wide-center" => () -> PB.SqrNormL2(0.7f0, c64)(x)] : Pair{String, Any}[])...,
    "ls/value" => () -> ls(β),
    "ls/gradient" => () -> PB.gradient!(gβ, ls, β),
    "ls/value_and_gradient" => () -> PB.value_and_gradient!(gβ, ls, β),
    "linear/value" => () -> lin(x),
    "l1/value" => () -> l1(x),
    "l1/prox/weighted" => () -> PB.prox!(z, l1w, x, T(0.2)),
    "exc/ridge/value/overflowing-squares" => () -> fbig(big_x),
    "exc/ridge/value/underflowing-squares" => () -> ftiny(tiny_x),
    "exc/ridge/value_parts/scaled" => () -> PB.Scaled(fbig, T(2))(big_x),
    "exc/ridge/gradient/overflowing-residual" => () -> PB.gradient!(g, ftop, top),
    "exc/ridge/value/overflowing-residual" => () -> ftop(top),
    "exc/ls/value/overflowing-squares" => () -> lsbig(β),
    "exc/ls/value/overflowing-Xβ" => () -> lso(βo),
    "exc/linear/value/overflowing-dot" => () -> linbig(alt),
    "exc/l1/prox/scaled-product" => () -> PB.prox!(z, l1x, x, γhuge),
  ]
end

measure(f) = (b = @benchmarkable $f() evals = 1 samples = 200 seconds = 0.25; tune!(b); minimum(run(b)).time)
allocs(f) = (f(); f(); @allocated f())

rows = []
for T in (Float64, Float32)
  D = shared(T)
  A, B = cases(Baseline.ProximalBase, D), cases(Candidate.ProximalBase, D)
  for ((name, fa), (nb, fb)) in zip(A, B)
    @assert name == nb
    occursin(FILTER, "$T/$name") || continue
    isequal(fa(), fb()) || @warn "results differ" T name
    ta, tb, ratios = Float64[], Float64[], Float64[]
    for k in 1:ROUNDS
      if isodd(k)
        a = measure(fa); b = measure(fb)
      else
        b = measure(fb); a = measure(fa)
      end
      push!(ta, a); push!(tb, b); push!(ratios, b / a)
    end
    push!(rows, (T, name, median(ta), median(tb), median(ratios), minimum(ratios), maximum(ratios), allocs(fa), allocs(fb)))
    r = rows[end]
    @printf("%-8s %-42s %11.1f ns %11.1f ns  ratio %.3f [%.3f, %.3f]  alloc %d / %d\n", r...)
    flush(stdout)
  end
end
