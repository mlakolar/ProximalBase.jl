# Time and allocations of the proximal operators of ProximalBase against
# ProximalOperators.jl, the general Julia library. Run from the repository root:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark benchmark/compare.jl
#
# Each case calls prox!(y, f, x, γ) on a fixed input after a warm-up, as a
# solver does, and reports the median time and the bytes allocated per call.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB
import ProximalOperators as PO

Random.seed!(1)

function measure(f, y, x, γ)
  PB_or_PO_prox!(f, y, x, γ)
  t = @belapsed PB_or_PO_prox!($f, $y, $x, $γ)
  b = @allocated PB_or_PO_prox!(f, y, x, γ)
  t, b
end
PB_or_PO_prox!(f::PB.AbstractFunction, y, x, γ) = PB.prox!(y, f, x, γ)
PB_or_PO_prox!(f, y, x, γ) = PO.prox!(y, f, x, γ)

function case(name, fb, fo, x, γ = 0.5)
  tb, bb = measure(fb, similar(x), x, γ)
  to, bo = measure(fo, similar(x), x, γ)
  @printf("| %-34s | %10.1f | %10.1f | %5.2f× | %9d | %9d |\n", name, 1e6tb, 1e6to, to / tb, bb, bo)
end

println("| case | ProximalBase (μs) | ProximalOperators (μs) | speedup | PB bytes | PO bytes |")
println("|:-----|---:|---:|---:|---:|---:|")
n = 100_000
x = randn(n)
case("NormL1, n = 10⁵", PB.NormL1(0.3), PO.NormL1(0.3), x)
w = rand(n)
case("weighted NormL1, n = 10⁵", PB.NormL1(1.0, 0.3 .* w), PO.NormL1(0.3 .* w), x)
case("NormL2, n = 10⁵", PB.NormL2(0.3), PO.NormL2(0.3), x)
case("ElasticNet, n = 10⁵", PB.ElasticNet(0.3, 0.2), PO.ElasticNet(0.3, 0.2), x)
case("NormL0, n = 10⁵", PB.NormL0(0.3), PO.NormL0(0.3), x)
case("NormLinf, n = 10⁵", PB.NormLinf(1.0), PO.NormLinf(1.0), x)
case("IndBallL1(10), n = 10⁵", PB.IndBallL1(10.0), PO.IndBallL1(10.0), x)
case("IndSimplex(1), n = 10⁵", PB.IndSimplex(1.0), PO.IndSimplex(1.0), x)
case("IndBallL0(1000), n = 10⁵", PB.IndBallL0(1000), PO.IndBallL0(1000), x)
case("IndBox(-1, 1), n = 10⁵", PB.IndBox(-1.0, 1.0), PO.IndBox(-1.0, 1.0), x)
X = randn(10, 10_000)
case("group ℓ2, 10⁴ groups of 10", PB.GroupNormL2(0.3, [(10k-9):(10k) for k in 1:10_000], ones(10_000)),
     PO.NormL21(0.3, 1), X)
M = randn(200, 100)
case("NuclearNorm, 200 × 100", PB.NuclearNorm(1.0), PO.NuclearNorm(1.0), M)
case("IndBallRank(5), 200 × 100", PB.IndBallRank(5), PO.IndBallRank(5), M)
A = randn(200, 200); A = (A + A') / 2
case("IndPSD, 200 × 200", PB.IndPSD(), PO.IndPSD(), A)
Xd, yd = randn(500, 200), randn(500)
case("LeastSquares prox, 500 × 200", PB.LeastSquaresLoss(yd, Xd), PO.LeastSquares(Xd, yd), randn(200))
# a step that changes at every call, as in ADMM with an adaptive ρ: PO refactors
struct Alternating{F}; f::F; γs::Vector{Float64}; k::Base.RefValue{Int}; end
alt(f) = Alternating(f, [0.5, 0.7], Ref(0))
function PB_or_PO_prox!(a::Alternating, y, x, γ)
  a.k[] += 1
  PB_or_PO_prox!(a.f, y, x, a.γs[1 + a.k[] % 2])
end
case("LeastSquares prox, γ changing", alt(PB.LeastSquaresLoss(yd, Xd)), alt(PO.LeastSquares(Xd, yd)), randn(200))
