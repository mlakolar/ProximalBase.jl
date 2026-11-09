module CoefficientFactorTypeTest
using Test, ProximalBase, Random
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
  # a step beyond floatmax is Inf (the margin goes to 1, result 1/y), not a finite wrong value
  @test ProximalBase._hinge_step(Float64, 2.0^600, 2.0^500, 2.0^600) == Inf
  @test ProximalBase._hinge_step(Float64, 2.0^1023, 2.0^3, 2.0^1023) == Inf
  @test px(HingeLoss([2.0^500], [2.0^600]), [-2.0^400], 2.0^600) == [2.0^-500]
  # a zero step is the identity although w|y| overflows; an infinite step with a tiny weight is the fixed point 1/y
  @test px(HingeLoss([-3.0], [floatmax(Float64)]), [1.5], 0.0) == [1.5]
  @test px(HingeLoss([-3f0], [floatmax(Float32)]), [1.5f0], 0f0) == [1.5f0]
  @test px(HingeLoss([2.0^-30], [5e-324]), [0.9], Inf) == [2.0^30]
  # integer labels and weights
  @test px(HingeLoss([1, -1], [1, 1]), [-1.0, 1.0], 0.5) == [-0.5, 0.5]
  # a subnormal weight with a label below one: the step is not rounded in two stages. v = 0 gives s/y
  # with s = γw|y|², the exact product rounded once (BigFloat), within one ulp of s here
  for (y, w, γ) in ((-0.6069132140372439, 2.327553e-318, 144380.9785781732), (0.3, 5e-324 * 7, 2.0^40), (-0.9, 1e-320, 3.5e9))
    s = Float64(big(γ) * big(w) * big(y)^2)
    got = prox_entry(HingeLoss([y], [w]), 0.0, 1, γ)
    @test abs(got * y - s) <= 2eps(s)
  end
  y32, w32, γ32 = -0.6f0, 3f-44, 3f4
  s32 = Float32(big(γ32) * big(w32) * big(y32)^2)
  @test abs(prox_entry(HingeLoss(Float32[y32], Float32[w32]), 0f0, 1, γ32) * y32 - s32) <= 2eps(s32)
end

@testset "HingeLoss array loop: entries that need recovery among ordinary ones" begin
  # one entry whose γw overflows Float32 but whose step γwy² = 1 is finite, one with a zero weight, one
  # infinite weight; the others are ordinary. prox! must equal the entrywise prox_entry, aliased or not.
  n = 40
  rng = Random.MersenneTwister(555)
  for T in (Float32, Float64)
    b = T == Float32 ? T(2)^80 : T(2)^600
    x = randn(rng, T, n)
    w = rand(rng, T, n) .+ T(0.5)
    lab = T.(rand(rng, (-1, 1), n))
    w[7] = b; lab[7] = inv(b)                       # γw = b² overflows, γwy² = 1 at γ = b
    w[11] = zero(T); w[12] = T(Inf)
    γ = fill(T(0.7), n); γ[7] = b
    f = HingeLoss(lab, w)
    expected = [prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    y = similar(x); @test prox!(y, f, x, γ) === y; @test y == expected
    z = copy(x); @test prox!(z, f, z, γ) === z; @test z == expected
    @test value_and_prox!(similar(x), f, x, γ) isa Real
    # ordinary entries only: the vectorized path gives the same bits as prox_entry
    g = HingeLoss(lab, rand(rng, T, n) .+ T(0.5))
    e2 = [prox_entry(g, x[i], i, T(0.7)) for i in 1:n]
    y2 = similar(x); prox!(y2, g, x, T(0.7)); @test y2 == e2
    z2 = copy(x); prox!(z2, g, z2, T(0.7)); @test z2 == e2
  end
end

@testset "CheckLoss: Float16, and the thresholds γτwᵢ and γ(1-τ)wᵢ rounded once" begin
  f = CheckLoss(Float16(0.75))
  @test prox_entry(f, Float16(4), 1, 2.0) === Float16(2.5)
  @test prox_entry(f, Float16(-4), 1, 2.0) === Float16(-3.5)
  @test prox_entry(CheckLoss(Float16(1 - 2.0^-10)), Float16(-2), 1, 2.0^10) === Float16(-1)
  # the exact product (BigFloat) rounded once to the destination; the Float64 factors' promotion makes the
  # intermediate rounding of γτ harmless for these inputs
  τ, γ, w = 0.7, 0.1, 3.0
  for u in (1.0000002f0, 2f0, -2f0)
    hi = Float32(big(γ) * big(w) * big(τ))
    @test prox_entry(CheckLoss(τ, [w]), u, 1, γ) === (u > hi ? u - hi : (u < -Float32(big(γ) * big(w) * (1 - big(τ))) ? u + Float32(big(γ) * big(w) * (1 - big(τ))) : 0f0))
  end
end

@testset "CheckLoss: 1-τ in the promotion of all the factors" begin
  # τ = 0.1f0 (Float32), γ = 3.0 (Float64), Float32 input: the complement is formed in Float64, with γ,
  # and the threshold γ(1-τ)w is rounded once to Float32 (BigFloat oracle, exact products)
  τ, γ = 0.1f0, 3.0
  f = CheckLoss(τ)
  lo = Float32(big(γ) * (1 - big(τ)))
  hi = Float32(big(γ) * big(τ))
  @test lo != Float32(big(γ) * Float64(1f0 - τ))                       # rounding 1-τ to Float32 first gives another threshold
  for u in (-4f0, -3f0, -2.7f0, -2.6f0, 2f0, 0.5f0)
    want = u > hi ? u - hi : (u < -lo ? u + lo : 0f0)
    @test prox_entry(f, u, 1, γ) === want
  end
  y = zeros(Float32, 3); prox!(y, f, Float32[-4, 0, 4], γ); @test y == Float32[-4 + lo, 0, 4 - hi]
  # weighted, Float64 weight and step, Float32 τ
  g = CheckLoss(τ, [2.0])
  lo2, hi2 = Float32(big(γ) * (1 - big(τ)) * 2), Float32(big(γ) * big(τ) * 2)
  @test prox_entry(g, -8f0, 1, γ) === -8f0 + lo2
  @test prox_entry(g, 8f0, 1, γ) === 8f0 - hi2
  # the KKT test, with a Float32 gradient and Float64 weight and Float32 τ: w(1-τ) formed in Float64
  kkt = ProximalBase.kkt_violation
  w = 3.0
  lo3, hi3 = Float32(big(w) * (1 - big(τ))), Float32(big(w) * big(τ))
  @test lo3 != Float32(big(w) * Float64(1f0 - τ))
  h = CheckLoss(τ, [w])
  @test kkt(h, Float32[2lo3], 1) === lo3                                # g - w(1-τ), g = 2·lo3 exactly
  @test kkt(h, Float32[-2hi3], 1) === hi3                               # -g - wτ
  lo64 = Float64(big(w) * (1 - big(τ)))
  @test kkt(CheckLoss(τ, [w]), [2lo64], 1) == lo64                       # Float64 gradient: w(1-τ) rounded once to Float64
  # integer gradients keep the baseline types
  @test kkt(CheckLoss(0.5, [2]), [4], 1) === 3.0
end

@testset "LogPenalty: the quotient by ε restores the range of λw" begin
  kkt = ProximalBase.kkt_violation
  p800, m800 = 2.0^800, 2.0^-800
  f = LogPenalty(p800, p800, [p800])                       # λw overflows Float64, λw/ε = 2^800
  @test kkt(f, [2 * p800], 1) == p800
  @test lla_weight(f, 0.0, 1) == p800
  @test lla_weight(f, p800, 1) == p800 / 2                 # λw/(ε + |v|) = 2^800/2
  @test lla_weights(LogPenalty(p800, p800, [p800, p800]), [0.0, p800]) == [p800, p800 / 2]
  g = LogPenalty(m800, m800, [m800])                       # λw underflows to 0, λw/ε = 2^-800
  @test kkt(g, [2 * m800], 1) == m800
  @test lla_weight(g, 0.0, 1) == m800
  @test lla_weight(g, m800, 1) == m800 / 2
  # λw/ε is a different power: 2^900·2^900/2^1000 = 2^800 and 2^-900·2^-900/2^-1000 = 2^-800
  @test kkt(LogPenalty(2.0^900, 2.0^1000, [2.0^900]), [2 * p800], 1) == p800
  @test kkt(LogPenalty(2.0^-900, 2.0^-1000, [2.0^-900]), [2 * m800], 1) == m800
  # Float32 factors and gradient: 2^80·2^80 overflows Float32, the quotient by 2^80 is 2^80
  b = 2f0^80
  @test kkt(LogPenalty(b, b, Float32[b]), Float32[2b], 1) === b
  @test lla_weight(LogPenalty(b, b, Float32[b]), 0f0, 1) === b
  @test lla_weight(LogPenalty(2f0^-80, 2f0^-80, Float32[2f0^-80]), 0f0, 1) === 2f0^-80
  # Float32 factors, Float64 gradient and argument: the product is formed in Float64
  @test kkt(LogPenalty(b, 1f0, Float32[b]), [2.0^161], 1) == 2.0^161 - 2.0^160
  # ordinary values keep the ordinary quotient
  @test lla_weight(LogPenalty(0.5, 0.25, [2.0]), 0.75, 1) == 1.0
  @test kkt(LogPenalty(0.5, 0.25, [2.0]), [3.0], 1) == 3.0 - 4.0
  # zero, infinite and NaN: the weight decides as before
  @test lla_weight(LogPenalty(p800, p800, [Inf]), 0.0, 1) == Inf
  @test lla_weight(LogPenalty(0.0, 1.0, [Inf]), 1.0, 1) == Inf
  @test lla_weight(LogPenalty(p800, p800, [0.0]), 0.0, 1) == 0.0
  @test kkt(LogPenalty(p800, p800, [0.0]), [3.0], 1) == 3.0
  @test kkt(LogPenalty(0.0, 1.0, [Inf]), [3.0], 1) == -Inf
  @test isnan(lla_weight(LogPenalty(1.0, 1.0, [1.0]), NaN, 1))
  # an infinite |v| gives a zero weight, not NaN
  @test lla_weight(LogPenalty(p800, p800, [p800]), Inf, 1) == 0.0
end

@testset "_prox_slope agrees with prox_entry at the threshold" begin
  f = NormL1(0.1)
  r, γ = 0.3f0, 3f0
  @test prox_entry(f, r, 1, γ) == 0f0
  @test ProximalBase._prox_slope(f, r, γ) == 0f0
  @test ProximalBase._prox_slope(NormL1Nonneg(0.1), r, γ) == 0f0
  @test ProximalBase._prox_slope(NormL1(big32), 2f0^100, 1f0) == 1f0     # the threshold 2^80 stays below r
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
  # the breakpoint decision of CappedL1 is made in the wider type (documented)
  c = CappedL1(1.0, 1 + 1e-9)
  @test lla_weights!(zeros(1), c, Float32[1]) == [1.0]
  @test lla_weights!(zeros(Float32, 1), c, Float32[1]) == Float32[0.0]
end

end
