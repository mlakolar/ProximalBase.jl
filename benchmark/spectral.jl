# Time of the proximal operators of Spectral(f) against the specialized
# functions of the eigenvalues of ProximalBase and ProximalOperators.jl, and of
# the projection onto the Fantope against the same computation written with
# LinearAlgebra's eigen, which allocates. Run from the repository root:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark benchmark/spectral.jl
#
# Each case calls prox!(Y, f, X, γ) on a fixed symmetric X after a warm-up and
# reports the median time and the bytes allocated per call.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB
import ProximalOperators as PO

Random.seed!(1)
BLAS.set_num_threads(1)

# the projection onto the Fantope {0 ⪯ X ⪯ I, tr X = k} with eigen, and the
# capped simplex by bisection on the level of clamp.(λ .- τ, 0, 1)
struct FantopeByEigen
  k::Float64
end
function fantope_by_eigen!(Y, f::FantopeByEigen, X)
  λ, V = eigen(Symmetric(X))
  lo, hi = minimum(λ) - 1, maximum(λ)
  for _ in 1:100
    τ = (lo + hi) / 2
    sum(t -> clamp(t - τ, 0, 1), λ) > f.k ? (lo = τ) : (hi = τ)
  end
  z = clamp.(λ .- (lo + hi) / 2, 0, 1)
  mul!(Y, V * Diagonal(z), V')
end

call!(f::PB.AbstractFunction, Y, X, γ) = PB.prox!(Y, f, X, γ)
call!(f::FantopeByEigen, Y, X, γ) = fantope_by_eigen!(Y, f, X)
call!(f, Y, X, γ) = PO.prox!(Y, f, X, γ)

function measure(f, X; γ = 0.5)
  Y = similar(X)
  call!(f, Y, X, γ)
  t = @belapsed call!($f, $Y, $X, $γ)
  t, @allocated call!(f, Y, X, γ)
end

function row(name, s, others, X)
  t, b = measure(s, X)
  cells = [@sprintf("%.0f μs (%.2f×), %d bytes", 1e6to, to / t, bo) for (to, bo) in (measure(o, X) for o in others)]
  @printf("| %-30s | %8.0f | %7d | %s |\n", name, 1e6t, b, join(cells, "; "))
end

println("| case | Spectral (μs) | bytes | others (time, relative to Spectral; bytes) |")
println("|:-----|---:|---:|:---|")
for n in (50, 200)
  B = randn(n, n)
  A = (B + B') / 2
  row("PSD cone, $n × $n", PB.Spectral(PB.IndNonnegative()), (PB.IndPSD(), PO.IndPSD()), A)
  row("trace norm, $n × $n", PB.Spectral(PB.NormL1(0.3)), (PB.NuclearNorm(0.3), PO.NuclearNorm(0.3)), A)
  row("trace on the cone, $n × $n", PB.Spectral(PB.NormL1Nonneg(0.3)), (PB.TracePSD(0.3),), A)
  row("Fantope k = 5, $n × $n", PB.Spectral(PB.Constrained(PB.IndBox(0, 1), PB.IndHyperplane(1, 5))),
      (FantopeByEigen(5),), A)
  row("eigenvalues in [-1, 1], $n × $n", PB.Spectral(PB.IndBox(-1, 1)), (), A)
end
