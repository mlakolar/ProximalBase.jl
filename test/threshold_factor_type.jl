module ThresholdFactorTypeTest
using Test, ProximalBase
using ..TestUtils

# Thresholds γλ1w, γλ2 and the coefficients of the nonconvex penalties are formed in the promotion
# of the destination and the factors' types and rounded once (numerics T3), as in threshold_precision.jl.

big32 = 2f0^80
x = [2.0^161, 1.0]
ones32, ones64 = ones(Float32, 2), ones(2)

@testset "Float32 factors, Float64 argument: the product overflows Float32" begin
  @test prox(ElasticNet(big32, 0f0, ones32), x, big32) == prox(ElasticNet(2.0^80, 0.0, ones64), x, 2.0^80) == [2.0^160, 0.0]
  @test prox(ElasticNet(big32, 0f0), x, big32) == [2.0^160, 0.0]                        # unweighted
  @test prox(ElasticNet(0f0, big32, ones32), [1.0, 1.0], big32) == prox(ElasticNet(0.0, 2.0^80, ones64), [1.0, 1.0], 2.0^80)
  @test prox(SparseGroupNormL2(big32, 0f0, [1:2], ones(Float32, 1)), x, big32) ==
        prox(SparseGroupNormL2(2.0^80, 0.0, [1:2], ones(1)), x, 2.0^80) == [2.0^160, 0.0]
  # zero and infinite weights through the typed path
  @test prox(ElasticNet(big32, 0f0, Float32[0, Inf]), x, big32) == [2.0^161, 0.0]
  y = zeros(2); prox!(y, ElasticNet(big32, 0f0, ones32), x, fill(big32, 2)); @test y == [2.0^160, 0.0]
  yg = zeros(2); prox!(yg, SparseGroupNormL2(big32, 0f0, [1:2], ones(Float32, 1)), x, fill(big32, 1)); @test yg == [2.0^160, 0.0]
  # the group classification of huge Float32 factors
  g = GroupNormL2(big32, [1:2], fill(big32, 1))
  @test prox(g, x, 1f0) == prox(GroupNormL2(2.0^80, [1:2], fill(2.0^80, 1)), x, 1.0)
end

@testset "Float64 factors, Float32 argument: one rounding to the destination" begin
  v = Float32[1.0000001f0, 3f0]
  for (f, g) in ((CappedL1(0.3, 1.0), CappedL1(0.3, 1.0)), (LogPenalty(0.3, 1.0), LogPenalty(0.3, 1.0)))
    y = prox(f, v, 0.1)
    @test eltype(y) == Float32
    @test y == Float32.(prox(g, Float64.(v), 0.1)) || all(isapprox.(y, prox(g, Float64.(v), 0.1); rtol = 2f-7))
  end
  # threshold γλ1 = 0.1f0*0.3 formed in Float64 then rounded: 0.03 (not the product of rounded factors)
  e = ElasticNet(0.3, 0.0, ones(2))
  @test prox(e, Float32[1, 1], 0.1) == Float32[shrink(1f0, Float32(0.1 * 0.3)), shrink(1f0, Float32(0.1 * 0.3))]
end

@testset "same-type Float32 stays Float32 and does not allocate" begin
  e, v = ElasticNet(0.5f0, 0.25f0), Float32[1, 2, 3]
  @test prox(e, v, 0.5f0) isa Vector{Float32}
  @test @inferred(prox_entry(e, 1f0, 1, 0.5f0)) isa Float32
  prox_entry(e, 1f0, 1, 0.5f0); @test (@allocated prox_entry(e, 1f0, 1, 0.5f0)) == 0
  # the unweighted weight `true` does not widen the working type
  @test ProximalBase._working_type(Float32, 0.5f0, 0.5f0, true) === Float32
end
end
