# Time of the proximal operators of Constrained against the specialized
# algorithms for the same sets: the projection onto the simplex of ProximalBase
# (Condat's algorithm) and of ProximalOperators.jl, and the projection onto the
# capped simplex by sorting the breakpoints. Run from the repository root:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark benchmark/constrained.jl
#
# Each case calls prox!(y, f, x, γ) on a fixed input after a warm-up and
# reports the median time. "passes" is the time of Constrained divided by that
# of one pass of the proximal operator of f alone.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB
import ProximalOperators as PO

Random.seed!(1)

# The projection onto {z : 0 ≤ z ≤ u, Σᵢ zᵢ = k}, z = clamp.(x .- τ, 0, u): the
# sum is piecewise linear in τ with breakpoints xᵢ - u and xᵢ, so a scan over
# the sorted breakpoints finds the piece that holds τ. O(n log n).
struct CappedSimplexSort
  k::Float64
  u::Float64
  xs::Vector{Float64}
end
CappedSimplexSort(k::Real, u::Real, n::Integer) = CappedSimplexSort(Float64(k), Float64(u), zeros(n))

function project!(z, c::CappedSimplexSort, x)
  n, k, u = length(x), c.k, c.u
  xs = c.xs
  copyto!(xs, x)
  sort!(xs)
  # g(τ) = Σᵢ clamp(xᵢ - τ, 0, u) from τ = xs[1] - u, where g = nu, upward
  i, j = 1, 1                   # the next breakpoints xs[i] - u and xs[j]
  τ, g, slope = xs[1] - u, n * u, 0
  while true
    t = i <= n && xs[i] - u <= xs[j] ? xs[i] - u : xs[j]
    gt = g + slope * (t - τ)
    if gt <= k                  # the level is on this piece
      τ = τ + (g - k) / (-slope)
      break
    end
    τ, g = t, gt
    if i <= n && xs[i] - u == t
      slope -= 1                # entry i leaves the cap u
      i += 1
    else
      slope += 1                # entry j reaches 0
      j += 1
    end
  end
  z .= clamp.(x .- τ, 0, u)
end

time_prox(f::PB.AbstractFunction, y, x, γ) = (PB.prox!(y, f, x, γ); @belapsed PB.prox!($y, $f, $x, $γ))
time_prox(f, y, x, γ) = (PO.prox!(y, f, x, γ); @belapsed PO.prox!($y, $f, $x, $γ))
time_prox(c::CappedSimplexSort, y, x, γ) = (project!(y, c, x); @belapsed project!($y, $c, $x))

function row(name, c, pass, others, x; γ = 0.5)
  y = similar(x)
  tc = time_prox(c, y, x, γ)
  tp = time_prox(pass, y, x, γ)
  cells = [@sprintf("%.1f μs (%.2f×)", 1e6t, t / tc) for t in (time_prox(o, y, x, γ) for o in others)]
  @printf("| %-38s | %10.1f | %5.1f | %s |\n", name, 1e6tc, tc / tp, join(cells, ", "))
end

println("| case | Constrained (μs) | passes | specialized (time, relative to Constrained) |")
println("|:-----|---:|---:|:---|")
for n in (1_000, 100_000)
  x = randn(n)
  s = n == 1_000 ? "10³" : "10⁵"
  row("simplex r = 1, n = $s", PB.Constrained(PB.IndNonnegative(), PB.IndHyperplane(1, 1)), PB.IndNonnegative(),
      (PB.IndSimplex(1.0), PO.IndSimplex(1.0)), x)
  row("simplex r = n/10, n = $s", PB.Constrained(PB.IndNonnegative(), PB.IndHyperplane(1, n / 10)), PB.IndNonnegative(),
      (PB.IndSimplex(n / 10), PO.IndSimplex(n / 10)), x)
  for k in (10, n ÷ 2)
    row("capped simplex k = $(k == 10 ? "10" : "n/2"), n = $s", PB.Constrained(PB.IndBox(0, 1), PB.IndHyperplane(1, k)),
        PB.IndBox(0, 1), (CappedSimplexSort(k, 1.0, n),), x)
  end
  row("zero-sum lasso λ = 0.5, n = $s", PB.Constrained(PB.NormL1(0.5), PB.IndHyperplane(1, 0)), PB.NormL1(0.5), (), x)
  row("ElasticNet on the simplex, n = $s", PB.Constrained(PB.ElasticNet(0.1, 1.0), PB.IndSimplex(1.0)),
      PB.ElasticNet(0.1, 1.0), (), x)
  row("NormL1 on [-1, 1], n = $s", PB.Constrained(PB.NormL1(0.3), PB.IndBox(-1, 1)), PB.NormL1(0.3), (), x)
end
