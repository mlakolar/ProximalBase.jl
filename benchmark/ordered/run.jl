# The functions of the order of the entries against other Julia packages:
# total-variation denoising against ProximalOperators.jl (also Condat's
# algorithm), isotonic regression against NearlyIsotonicRegression.jl's pool
# adjacent violators, and SLOPE regression, by FISTA with the proximal operator
# of SLOPE, against SLOPE.jl, the interface to libslope. Run from the
# repository root:
#
#   julia --project=benchmark/ordered -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark/ordered benchmark/ordered/run.jl
#
# The proximal operators are called on a fixed input after a warm-up, and the
# times are medians from BenchmarkTools; the regressions report the best of
# three solves.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB
import ProximalOperators as PO
import NearlyIsotonicRegression as NIR
import SLOPE as SL

Random.seed!(1)
BLAS.set_num_threads(1)

t_pb(f, y, x) = (PB.prox!(y, f, x, 1.0); @belapsed PB.prox!($y, $f, $x, 1.0))
t_po(f, y, x) = (PO.prox!(y, f, x, 1.0); @belapsed PO.prox!($y, $f, $x, 1.0))
iso!(y, x) = (copyto!(y, x); NIR.iso!(y))
t_iso(y, x) = (iso!(y, x); @belapsed iso!($y, $x))

row(name, tb, other, to, d) =
  @printf("| %-34s | %9.1f | %-26s | %9.1f | %5.2f× | %.0e |\n", name, 1e6tb, other, 1e6to, to / tb, d)

println("| case | ProximalBase (μs) | other | other (μs) | ratio | largest difference |")
println("|:-----|---:|:---|---:|---:|---:|")
for n in (1_000, 100_000)
  s = n == 1_000 ? "10³" : "10⁵"
  for (kind, x) in (("noise", randn(n)), ("10 steps and noise", repeat(randn(10); inner = n ÷ 10) .+ 0.3 .* randn(n)))
    y1, y2 = similar(x), similar(x)
    tb, to = t_pb(PB.TotalVariation1D(1.0), y1, x), t_po(PO.TotalVariation1D(1.0), y2, x)
    row("TV, $kind, n = $s", tb, "ProximalOperators", to, maximum(abs.(y1 .- y2)))
  end
  x = cumsum(randn(n)) ./ sqrt(n) .+ randn(n)            # a trend and noise
  y1, y2 = similar(x), similar(x)
  tb, to = t_pb(PB.IndIsotonic(), y1, x), t_iso(y2, x)
  row("isotonic, n = $s", tb, "NearlyIsotonicRegression", to, maximum(abs.(y1 .- y2)))
  x = randn(n)
  tb, to = t_pb(PB.SLOPE(sort(rand(n); rev = true)), y1, x), t_pb(PB.NormL1(0.5), y2, x)
  row("SLOPE, n = $s", tb, "NormL1 of ProximalBase", to, NaN)
end

# minimize ‖y - Xβ‖²/(2n) + α Σᵢ λᵢ|β|₍ᵢ₎, without an intercept or scaling
function fista!(β, f, g; tol = 1e-9, maxiter = 100_000)
  L = PB.lipschitz(f)
  βold, z, grad = similar(β), copy(β), similar(β)
  t = 1.0
  for k in 1:maxiter
    PB.gradient!(grad, f, z)
    copyto!(βold, β)
    β .= z .- grad ./ L
    PB.prox!(β, g, β, 1 / L)
    norm(β .- βold) <= tol * max(1, norm(β)) && return k
    tnew = (1 + sqrt(1 + 4t^2)) / 2
    z .= β .+ ((t - 1) / tnew) .* (β .- βold)
    t = tnew
  end
  maxiter
end
best3(g) = (g(); minimum(@elapsed(g()) for _ in 1:3))

println()
println("| SLOPE regression | FISTA with SLOPE (ms) | iterations | SLOPE.jl (ms) | ratio | largest difference, 1/n loss | 1/1 loss |")
println("|:-----|---:|---:|---:|---:|---:|---:|")
for (n, p) in ((200, 500), (1000, 2000))
  X = randn(n, p)
  y = X * [3 .* randn(10); zeros(p - 10)] + randn(n)
  λ = collect(range(2.0, 0.5; length = p))
  α = 0.3 * maximum(abs.(X' * y)) / n / λ[1]
  sl() = SL.slope(X, y; α = α, λ = λ, fit_intercept = false, centering = :none, scaling = :none, tol = 1e-10)
  βsl = Vector(SL.coef(sl(); index = 1, simplify = true))
  tsl = best3(sl)
  f, g = (1 / n) * PB.LeastSquaresLoss(y, X), PB.SLOPE(α .* λ)
  β = zeros(p)
  iters = fista!(β, f, g)
  tpb = best3(() -> (fill!(β, 0); fista!(β, f, g)))
  β1 = zeros(p)
  fista!(β1, PB.LeastSquaresLoss(y, X), g)               # in case the loss has no 1/n
  @printf("| n = %d, p = %d | %.1f | %d | %.1f | %.2f× | %.0e | %.0e |\n", n, p, 1e3tpb, iters, 1e3tsl, tsl / tpb,
          maximum(abs.(β .- βsl)), maximum(abs.(β1 .- βsl)))
end
