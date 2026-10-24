# The proximal operators of the functions of the order of the entries against
# other Julia packages, where they exist: total-variation denoising against
# ProximalOperators.jl (also Condat's algorithm), and isotonic regression
# against NearlyIsotonicRegression.jl's pool adjacent violators. No package
# exposes the proximal operator of SLOPE, so its time is compared with that of
# the sort it needs, which bounds it from below. Run from the repository root:
#
#   julia --project=benchmark/ordered -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark/ordered benchmark/ordered/run.jl
#
# The operators are called on a fixed input after a warm-up, and the times are
# medians from BenchmarkTools. Solvers that use these operators are compared in
# ProximalSolvers.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB
import ProximalOperators as PO
import NearlyIsotonicRegression as NIR

Random.seed!(1)

t_pb(f, y, x) = (PB.prox!(y, f, x, 1.0); @belapsed PB.prox!($y, $f, $x, 1.0))
t_po(f, y, x) = (PO.prox!(y, f, x, 1.0); @belapsed PO.prox!($y, $f, $x, 1.0))
iso!(y, x) = (copyto!(y, x); NIR.iso!(y))
t_iso(y, x) = (iso!(y, x); @belapsed iso!($y, $x))
# the magnitudes and their order, as the proximal operator of SLOPE computes them
t_sort(f, x) = (PB._sortperm_abs!(f, x, Float64); @belapsed PB._sortperm_abs!($f, $x, Float64))

row(name, tb, other, to, d) =
  @printf("| %-34s | %9.1f | %-26s | %9.1f | %5.2f× | %s |\n", name, 1e6tb, other, 1e6to, to / tb, d)

println("| case | ProximalBase (μs) | other | other (μs) | ratio | largest difference |")
println("|:-----|---:|:---|---:|---:|---:|")
for n in (1_000, 100_000)
  s = n == 1_000 ? "10³" : "10⁵"
  for (kind, x) in (("noise", randn(n)), ("10 steps and noise", repeat(randn(10); inner = n ÷ 10) .+ 0.3 .* randn(n)))
    y1, y2 = similar(x), similar(x)
    tb, to = t_pb(PB.TotalVariation1D(1.0), y1, x), t_po(PO.TotalVariation1D(1.0), y2, x)
    row("TV, $kind, n = $s", tb, "ProximalOperators", to, @sprintf("%.0e", maximum(abs.(y1 .- y2))))
  end
  x = cumsum(randn(n)) ./ sqrt(n) .+ randn(n)            # a trend and noise
  y1, y2 = similar(x), similar(x)
  tb, to = t_pb(PB.IndIsotonic(), y1, x), t_iso(y2, x)
  row("isotonic, n = $s", tb, "NearlyIsotonicRegression", to, @sprintf("%.0e", maximum(abs.(y1 .- y2))))
  x = randn(n)
  f = PB.SLOPE(sort(rand(n); rev = true))
  tb, to = t_pb(f, y1, x), t_sort(f, x)
  row("SLOPE, n = $s", tb, "its sort alone", to, "")
end
