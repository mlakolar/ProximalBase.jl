# The nearest correlation matrix by IndIntersection (Dykstra's algorithm, with
# and without a warm start) against NearestCorrelationMatrix.jl, whose Newton
# method (Qi and Sun, 2006) converges quadratically. Run from the repository root:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=benchmark benchmark/intersection.jl
#
# Each case is the median time of a call on a fixed input after a warm-up. "warm"
# is a call on an input perturbed by 1e-4 from the last one, started from the
# corrections of the last call, as in the next iteration of a solver.

using BenchmarkTools, LinearAlgebra, Random, Printf
import ProximalBase as PB
import NearestCorrelationMatrix as NCM

Random.seed!(1)

function indefinite_correlation(p)
  B = randn(2p, p)
  C = B'B
  d = sqrt.(diag(C))
  E = 0.3 * randn(p, p) / sqrt(p) * 4
  R = C ./ (d * d') + (E + E') / 2
  R[diagind(R)] .= 1
  Symmetric(R)
end

function unit_diagonal(p)
  L, U = fill(-Inf, p, p), fill(Inf, p, p)
  L[diagind(L)] .= 1
  U[diagind(U)] .= 1
  PB.IndBox(L, U)
end

# a call on the other of two nearby inputs, from the corrections of the last call
alternate!(Y, f, inputs, k) = (k[] = k[] % 2 + 1; PB.prox!(Y, f, inputs[k[]], 1.0))

println("| p | Dykstra, cold (ms) | rounds | Dykstra, warm (ms) | rounds | NCM Newton (ms) | NCM alternating projections (ms) | ‖Dykstra − Newton‖ |")
println("|---:|---:|---:|---:|---:|---:|---:|---:|")
for p in (50, 200, 500)
  R = indefinite_correlation(p)
  Rm = Matrix(R)
  Y = similar(Rm)
  cold = PB.IndIntersection(PB.IndPSD(), unit_diagonal(p); warmstart = false, maxiter = 100_000)
  tc = @belapsed PB.prox!($Y, $cold, $Rm, 1.0)
  rc = cold.iterations
  warm = PB.IndIntersection(PB.IndPSD(), unit_diagonal(p); maxiter = 100_000)
  PB.prox!(Y, warm, Rm, 1.0)
  Δ = 1e-4 * Matrix(Symmetric(randn(p, p)))
  inputs = (Rm .+ Δ, Rm .- Δ)
  tw = @belapsed alternate!($Y, $warm, $inputs, $(Ref(0)))
  rw = warm.iterations
  tn = @belapsed NCM.nearest_cor($Rm, NCM.Newton())
  ta = @belapsed NCM.nearest_cor($Rm, NCM.AlternatingProjections())
  Cd = PB.prox(cold, Rm)
  Cn = NCM.nearest_cor(Rm, NCM.Newton())
  @printf("| %d | %.2f | %d | %.2f | %d | %.2f | %.2f | %.1e |\n", p, 1e3tc, rc, 1e3tw, rw, 1e3tn, 1e3ta, norm(Cd - Cn))
end
