# Paired benchmarks for the numerical-kernels follow-up PRs: norms, QuadraticFunction, the spectral
# proximal operators and the sparse triple product, on ordinary and exceptional inputs.
#
#   taskset -c 12 julia --project=benchmark --threads=1 -O2 --check-bounds=no \
#     docs/audit/numerical-kernels-file/bench.jl BASELINE_TREE CANDIDATE_TREE [ROUNDS] [FILTER]
#
# The method is that of docs/audit/numerical-kernels-865/bench.jl: both trees are loaded as modules
# in one process and share the same plain data arrays (only function objects and iterates are built
# per tree); rounds alternate the order (A B, B A, ...); each measurement is the minimum over a
# BenchmarkTools sample; the table gives the median time over rounds, the median and range of the
# per-round candidate/baseline ratios, and the bytes allocated per call after warm-up. Run it with the
# baseline tree in both positions for an A/A control.
using BenchmarkTools, LinearAlgebra, SparseArrays, Random, Statistics, Printf
BLAS.set_num_threads(1)

const TREES = abspath.(ARGS[1:2])
const ROUNDS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 7
const FILTER = length(ARGS) >= 4 ? Regex(ARGS[4]) : r""
module Baseline end
module Candidate end
Base.include(Baseline, joinpath(TREES[1], "src", "ProximalBase.jl"))
Base.include(Candidate, joinpath(TREES[2], "src", "ProximalBase.jl"))
const PBS = (Baseline.ProximalBase, Candidate.ProximalBase)

function symmetric(PB, ::Type{T}, p, vals) where {T}
  S = PB.SymmetricSparseIterate(T, p)
  k = 0
  for c in 1:p, r in c:p
    k += 1
    k > length(vals) && break
    S[r, c] = vals[k]
  end
  S
end

# name => (PB -> zero-argument closure); the closure is benchmarked
function cases(::Type{T}) where {T}
  rng = Xoshiro(1111)
  E = exponent(floatmax(T)); e0 = exponent(floatmin(T))
  n = 10_000
  a, b = randn(rng, T, n), randn(rng, T, n)
  ah, bh = ldexp.(a, E ÷ 2 + 4), ldexp.(b, E ÷ 2 + 4)               # the squares overflow
  at, bt = ldexp.(a, e0 ÷ 2 - 4), ldexp.(b, e0 ÷ 2 - 4)             # the squares underflow
  p = 60
  sv = randn(rng, T, p * (p + 1) ÷ 2)
  m = 50
  Q = Matrix(qr(randn(rng, T, m, m)).Q)
  Λ = abs.(randn(rng, T, m)) .+ T(0.1)
  A = Symmetric(Q * Diagonal(Λ) * Q') |> Matrix
  Ah = ldexp.(A, E - 8)
  x, bq = randn(rng, T, m), randn(rng, T, m)
  X, Y = randn(rng, T, 80, m), randn(rng, T, 80)
  Xh = ldexp.(X, E ÷ 2 - 2)
  As, Bs = Symmetric(randn(rng, T, 40, 40)), Symmetric(randn(rng, T, 40, 40))
  big2 = ldexp(T(0.75), E ÷ 2 + 2)
  Ab, Bb = Symmetric(fill(big2, 40, 40)), Symmetric(fill(big2, 40, 40))
  γs, γh = T(0.5), ldexp(one(T), E - 4)
  Pair{String, Function}[
    "norm_diff/ordinary"              => PB -> (() -> PB.norm_diff(a, b)),
    "norm_diff/equal"                 => PB -> (() -> PB.norm_diff(a, a)),
    "exc/norm_diff/overflowing"       => PB -> (() -> PB.norm_diff(ah, bh)),
    "exc/norm_diff/underflowing"      => PB -> (() -> PB.norm_diff(at, bt)),
    "symmetric_norm/ordinary"         => PB -> (S = symmetric(PB, T, p, sv); () -> norm(S)),
    "exc/symmetric_norm/overflowing"  => PB -> (S = symmetric(PB, T, p, ldexp.(sv, E ÷ 2 + 4)); () -> norm(S)),
    "exc/symmetric_norm/underflowing" => PB -> (S = symmetric(PB, T, p, ldexp.(sv, e0 ÷ 2 - 4)); () -> norm(S)),
    "ridge/value/symmetric"           => PB -> (S = symmetric(PB, T, p, sv); f = PB.SqrNormL2(one(T)); () -> f(S)),
    "quadratic/value"                 => PB -> (f = PB.QuadraticFunction(A, bq, one(T)); () -> f(x)),
    "quadratic/value_and_gradient"    => PB -> (f = PB.QuadraticFunction(A, bq); g = similar(x); () -> PB.value_and_gradient!(g, f, x)),
    "quadratic/value/diagonal"        => PB -> (f = PB.QuadraticFunction(Diagonal(Λ)); () -> f(x)),
    "quadratic/prox"                  => PB -> (f = PB.QuadraticFunction(A, bq); y = similar(x); PB.prox!(y, f, x, γs); () -> PB.prox!(y, f, x, γs)),
    "exc/quadratic/value/underflow"   => PB -> (f = PB.QuadraticFunction(A); xt = ldexp.(x, e0 ÷ 2 + 2); () -> f(xt)),
    "exc/quadratic/prox/large-step"   => PB -> (f = PB.QuadraticFunction(Ah, bq); y = similar(x); xh = ldexp.(x, E - 2); PB.prox!(y, f, xh, γh); () -> PB.prox!(y, f, xh, γh)),
    "ls/prox"                         => PB -> (f = PB.LeastSquaresLoss(Y, X); y = similar(x); PB.prox!(y, f, x, γs); () -> PB.prox!(y, f, x, γs)),
    "ls/prox/large-step"              => PB -> (f = PB.LeastSquaresLoss(Y, X); y = similar(x); PB.prox!(y, f, x, T(4)); () -> PB.prox!(y, f, x, T(4))),
    "exc/ls/prox/large-design"        => PB -> (f = PB.LeastSquaresLoss(Y, Xh); y = similar(x); PB.prox!(y, f, x, γh); () -> PB.prox!(y, f, x, γh)),
    "triple/ordinary"                 => PB -> (Z = PB.SparseIterate(T, 40, 40); for k in 1:40:1600; Z[k] = T(k); end; () -> As * Z * Bs),
    "exc/triple/rescue"               => PB -> (Z = PB.SparseIterate(T, 40, 40); Z[1, 1] = big2; Z[2, 2] = big2; Z[1, 2] = -big2; () -> Ab * Z * Bb),
  ]
end

function measure(f)
  f(); f()
  b = @benchmark $f() samples = 200 evals = 1 seconds = 1
  minimum(b).time, minimum(b).memory
end

for T in (Float64, Float32)
  cs = filter(c -> occursin(FILTER, c.first), cases(T))
  for (name, mk) in cs
    fs = map(mk, PBS)
    times = ([Float64[] for _ in 1:2]...,)
    mem = (0, 0)
    for r in 1:ROUNDS
      order = isodd(r) ? (1, 2) : (2, 1)
      for i in order
        t, m = measure(fs[i])
        push!(times[i], t)
        mem = Base.setindex(mem, m, i)
      end
    end
    ratios = times[2] ./ times[1]
    @printf("%-8s %-36s %12.1f ns %12.1f ns  ratio %.3f [%.3f, %.3f]  alloc %d / %d\n", T, name,
            median(times[1]), median(times[2]), median(ratios), minimum(ratios), maximum(ratios), mem[1], mem[2])
  end
end
