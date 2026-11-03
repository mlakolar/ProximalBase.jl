module CoefficientFactorTypeTest
using Test, ProximalBase
using ..TestUtils

# Issue #519, losses, KKT diagnostics and LLA weights: a combined coefficient (γwτ, γw(1-τ), γwy²,
# λw, w y) is formed in the promotion of the model precision and the factors' types and rounded once
# (numerics T3, R3). Expected values are exact powers of two, worked out by hand, or BigFloat
# (512 bits) evaluations of the formula, not the helpers under test.

big32 = 2f0^80
px(f, x, γ) = (y = zeros(eltype(x), length(x)); prox!(y, f, x, γ); y)
rnd(::Type{T}, x::BigFloat) where {T} = T(x)

@testset "EpsInsensitiveLoss: the threshold γw overflows Float32 but not Float64" begin
  f = EpsInsensitiveLoss(0.0, Float32[big32])
  @test px(f, [2.0^161], big32) == [2.0^160]
  @test px(f, [-2.0^161], big32) == [-2.0^160]
  @test prox_entry(f, 2.0^161, 1, big32) == 2.0^160
  y = zeros(1); v = value_and_prox!(y, f, [2.0^161], big32)
  @test y == [2.0^160] && v == 2.0^240                      # the value is that of the stored output (O2)
  # the unweighted loss and a flat part
  g = EpsInsensitiveLoss(2.0^100)
  @test px(g, [2.0^101], big32) == [2.0^101 - 2.0^80]
  # underflow of γw: 2^-160 is a normal Float64
  h = EpsInsensitiveLoss(0.0, Float32[2f0^-80])
  @test px(h, [1.0], 2f0^-80) == [1.0 - 2.0^-160] == [1.0]
  @test px(h, [2.0^-150], 2f0^-80) == [2.0^-150 - 2.0^-160]
  # zero and infinite weights
  @test px(EpsInsensitiveLoss(0.0, Float32[0, Inf]), [3.0, 3.0], big32) == [3.0, 0.0]
end

@testset "CheckLoss: γwτ and γw(1-τ) are formed once" begin
  f = CheckLoss(0.5, Float32[big32])
  @test px(f, [2.0^161], big32) == [3 * 2.0^159]
  @test px(f, [-2.0^161], big32) == [-3 * 2.0^159]
  @test px(f, [2.0^158], big32) == [0.0]
  y = zeros(1); value_and_prox!(y, f, [2.0^161], big32); @test y == [3 * 2.0^159]
  # 1 - τ is formed in τ's own type before the threshold is rounded to Float32
  g = CheckLoss(1 - 2.0^-40)
  @test px(g, Float32[-2], 2.0^40) == Float32[-1]
  @test px(g, Float32[-2], 2.0^40) == px(g, Float64[-2], 2.0^40)
  # one rounding of γwτ and of γw(1-τ) to Float32, against a BigFloat product
  τ, γ, w = 0.7, 0.1, 3.0
  h = CheckLoss(τ, [w])
  for u in (1.0000002f0, 2f0, -2f0, 0.5f0, -0.5f0)
    hi = Float32(big(γ) * big(w) * big(τ))
    lo = Float32(big(γ) * big(w) * (1 - big(τ)))
    want = u > hi ? u - hi : (u < -lo ? u + lo : 0f0)
    @test prox_entry(h, u, 1, γ) === want
  end
  # zero and infinite weights
  @test px(CheckLoss(0.5, Float32[0, Inf]), [3.0, 3.0], big32) == [3.0, 0.0]
end

@testset "HingeLoss: the step γwy² is one product" begin
  y = 2f0^-80
  f = HingeLoss(Float32[y], Float32[big32])
  # γwy² = 1 is finite although γw overflows and y² underflows Float32: the margin u = -1 moves to 0
  @test px(f, Float32[-2f0^80], big32) == Float32[0]
  @test px(f, Float32[-2f0^80], big32) == px(HingeLoss(Float64[y], Float64[big32]), Float64[-2.0^80], Float64(big32))
  # Float64: y² overflows alone, γw underflows alone
  g = HingeLoss([2.0^600], [2.0^-600])
  @test px(g, [-2 * 2.0^-600], 2.0^-600) == [-2.0^-600]
  h = HingeLoss([2.0^-600], [2.0^600])
  @test px(h, [-2 * 2.0^600], 2.0^600) == [-2.0^600]
  # ordinary labels ±1 and weights
  @test px(HingeLoss([1.0, -1.0], [1.0, 1.0]), [-1.0, 1.0], 0.5) == [-0.5, 0.5]
  # zero and infinite weights
  @test px(HingeLoss(nothing, Float32[0, Inf]), [-1.0, -1.0], big32) == [-1.0, 1.0]
end

@testset "KKT violations form the coefficient in the gradient's precision" begin
  g = [2.0^160]
  kkt = ProximalBase.kkt_violation
  for f in (NormL1(big32, Float32[big32]), ElasticNet(big32, 1f0, Float32[big32]),
            CappedL1(big32, 1f0, Float32[big32]), Berhu(big32, 1f0, Float32[big32]))
    @test kkt(f, g, 1) == 0.0
    @test kkt(f, [2.0^161], 1) == 2.0^160
    @test kkt(f, [2.0^159], 1) == -2.0^159
  end
  @test kkt(NormL1Nonneg(big32, Float32[big32]), [-2.0^160], 1) == 0.0
  @test kkt(LogPenalty(big32, 1f0, Float32[big32]), g, 1) == 0.0        # λw/ε = 2^160
  @test kkt(LogPenalty(1f0, 2f0^-80, Float32[2f0^80]), [2.0^160], 1) == 0.0   # λw/ε overflows Float32 only in a quotient formed early
  @test kkt(GroupNormL2(big32, [1:1], Float32[big32]), g, [0.0], 1) == 0.0
  @test kkt(GroupMCP(big32, [1:1], Float32[big32]), g, [0.0], 1) == 0.0
  @test kkt(SparseGroupNormL2(big32, big32, [1:1], Float32[big32]), g, [0.0], 1) == 0.0
  # Float64 factors, Float32 gradient: one rounding of λw
  λ, w = 0.1, 3.0
  @test kkt(NormL1(λ, [w]), Float32[1], 1) === 1f0 - Float32(big(λ) * big(w))
  # zero and infinite weights
  @test kkt(NormL1(big32, Float32[0]), [3.0], 1) == 3.0
  @test kkt(NormL1(big32, Float32[Inf]), [3.0], 1) == -Inf
  # losses: w y and w τ
  @test kkt(HingeLoss(Float32[big32], Float32[big32]), g, 1) == 0.0
  @test kkt(HingeLoss(Float32[0.3f0], Float32[0.1f0]), [Float64(0.1f0) * Float64(0.3f0)], 1) == 0.0
  @test kkt(CheckLoss(0.3f0, Float32[0.1f0]), [-(Float64(0.1f0) * Float64(0.3f0))], 1) == 0.0
  @test kkt(CheckLoss(1f-8, [1.0]), [1 - Float64(1f-8)], 1) == 0.0
end

@testset "LLA weights" begin
  # GroupNormL2: λ0w = 2^160 overflows Float32; 2^-160 underflows it
  @test lla_weights(GroupNormL2(big32, [1:1], Float32[big32]), [1.0]) == [2.0^160]
  @test lla_weights(GroupNormL2(2f0^-80, [1:1], Float32[2f0^-80]), [1.0]) == [2.0^-160]
  @test lla_weight(GroupNormL2(big32, [1:1], Float32[big32]), 1.0, 1) == 2.0^160
  # the radial penalties: a finite coefficient is not classified infinite
  @test lla_weights(GroupSCAD(big32, [1:1], Float32[big32]), [1.0]) == [2.0^160]
  @test lla_weights(GroupMCP(big32, [1:1], Float32[big32]), [2.0^159]) == [2.0^160 - 2.0^159 / 3] ||
        isapprox(lla_weights(GroupMCP(big32, [1:1], Float32[big32]), [2.0^159])[1], 2.0^160 - 2.0^159 / 3; rtol = 1e-15)
  # an infinite weight is still infinite, also with a zero level
  @test lla_weights(GroupMCP(2f0, [1:1], Float32[Inf]), [1.0]) == [Inf]
  @test lla_weights(GroupMCP(0f0, [1:1], Float32[Inf]), [1.0]) == [Inf]
  # a wider destination keeps the model precision: not narrowed to the argument's type
  @test lla_weights!(zeros(1), NormL1(0.1, [1.0]), Float32[0]) == [0.1]
  @test lla_weights!(zeros(1), NormL1(big32, Float32[big32]), Float32[0]) == [2.0^160]
  @test lla_weights!(zeros(1), MCP(2.0^200, 3.0), Float32[0]) == [2.0^200]
  @test lla_weights!(zeros(1), GroupNormL2(0.1, [1:1], [1.0]), Float32[0]) == [0.1]
  @test lla_weights!(zeros(Float32, 1), NormL1(0.1, [1.0]), [0.0]) == Float32[0.1]   # narrower destination: one rounding
end

end
