# The calls that dependents of ProximalBase make, timed in 0.4 or 0.5: run it in
# an environment with either version (see README.md).

using BenchmarkTools, LinearAlgebra, Random, SparseArrays, Printf
using ProximalBase
Random.seed!(1)
v = pkgversion(ProximalBase)
old = v < v"0.5"
row(name, t, b) = @printf("| %-46s | %9.1f μs | %8d bytes |\n", name, 1e6t, b)
function m(f, args...)
  f(args...)
  (@belapsed $f($(args)...)), (@ballocated $f($(args)...))
end
println("ProximalBase $v")
x = randn(100_000); y = similar(x)
gr = [(10k-9):(10k) for k in 1:10_000]
M = randn(200, 100); Y = similar(M)
A = randn(400, 200); S = Symmetric(A'A / 400); P = Matrix(I + 0.1 * Symmetric(randn(200, 200))); Q = similar(P)
X, yy = randn(500, 200), randn(500); β = randn(200); g = similar(β)
if old
  f1, fg, fn, fl, ls = ProxL1(0.3), ProxGroupL2(0.3, gr), ProxNuclear(1.0), ProxGaussLikelihood(S), LeastSquaresLoss(yy, X)
  pv(f, out, x) = (prox!(f, out, x, 0.5); value(f, out))
  row("ℓ1 prox, n = 10⁵", m((f, out, x) -> prox!(f, out, x, 0.5), f1, y, x)...)
  row("group ℓ2 prox, 10⁴ groups", m((f, out, x) -> prox!(f, out, x, 0.5), fg, y, x)...)
  row("nuclear norm: prox and value, 200 × 100", m(pv, fn, Y, M)...)
  row("Gaussian likelihood: prox and value, 200 × 200", m(pv, fl, Q, P)...)
  row("least squares: value and gradient, 500 × 200", m((f, g, b) -> value_and_gradient!(f, g, b), ls, g, β)...)
  xs = SparseIterate(sprandn(100_000, 0.01)); ys = similar(xs)
  row("ℓ1 prox, SparseIterate 1% dense", m((f, out, x) -> prox!(f, out, x, 0.5), f1, ys, xs)...)
else
  f1, fg, fn, fl, ls = NormL1(0.3), GroupNormL2(0.3, gr), NuclearNorm(1.0), LogDetLoss(S), LeastSquaresLoss(yy, X)
  row("ℓ1 prox, n = 10⁵", m((out, f, x) -> prox!(out, f, x, 0.5), y, f1, x)...)
  row("group ℓ2 prox, 10⁴ groups", m((out, f, x) -> prox!(out, f, x, 0.5), y, fg, x)...)
  row("nuclear norm: value_and_prox!, 200 × 100", m((out, f, x) -> value_and_prox!(out, f, x, 0.5), Y, fn, M)...)
  row("Gaussian likelihood: value_and_prox!, 200 × 200", m((out, f, x) -> value_and_prox!(out, f, x, 0.5), Q, fl, P)...)
  row("least squares: value and gradient, 500 × 200", m((g, f, b) -> value_and_gradient!(g, f, b), g, ls, β)...)
  xs = SparseIterate(sprandn(100_000, 0.01)); ys = similar(xs)
  row("ℓ1 prox, SparseIterate 1% dense", m((out, f, x) -> prox!(out, f, x, 0.5), ys, f1, xs)...)
end
