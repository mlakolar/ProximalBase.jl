module ThresholdFactorTypeTest
using Test, ProximalBase, Random
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

@testset "guards use the factors, not their rounding to the destination" begin
  # γλ = 1 although γ = 1e-9 underflows and λ = 1e9 overflows Float16
  h = Float16(3)
  for (f, g) in ((Berhu(1e9, 1.0), Berhu(1e9, 1.0)), (NormLq(1e9, 0.5), NormLq(1e9, 0.5)), (LogPenalty(1e9, 1.0), LogPenalty(1e9, 1.0)))
    y = prox_block(f, h, 1, 1e-9)
    @test y isa Float16
    @test isapprox(Float64(y), prox_block(g, 3.0, 1, 1e-9); rtol = 2e-3)
  end
  # a finite weight that overflows the destination is not an infinite weight
  @test isapprox(Float64(prox_block(NormLq(1.0, 0.5, [1e9]), h, 1, 1e-9)), prox_block(NormLq(1.0, 0.5, [1e9]), 3.0, 1, 1e-9); rtol = 2e-3)
  @test prox_block(NormLq(1.0, 0.5, [Inf]), h, 1, 1.0) == 0
end

# Range recovery (R2/R3), with expected values worked out from the formulas, not from another implementation:
# the soft threshold is γλ1w, shrink(x, t) = x - t, and the ridge weight is 1/(1 + γλ2).
@testset "ElasticNet: the ridge weight γλ2 and the threshold γλ1w recover their range" begin
  # γλ2 = 2^160 overflows Float32; 2^100/(1 + 2^160) = 2^-60 is a normal Float32
  f = ElasticNet(0.0, 2.0^80); x32 = Float32[2.0^100, -2.0^100, 0, 2.0^100 * 3]
  want = Float32[2.0^-60, -2.0^-60, 0, 3 * 2.0^-60]
  @test prox(f, x32, 2.0^80) == want
  @test prox_block(f, x32[1], 1, 2.0^80) === want[1]
  y = similar(x32); @test prox!(y, f, x32, fill(2.0^80, 4)) === y; @test y == want
  y = similar(x32); @test prox!(y, f, x32, 2.0^80) === y; @test y == want
  yv = similar(x32); val = value_and_prox!(yv, f, x32, 2.0^80); @test yv == want
  @test isapprox(val, sum(abs2, Float64.(want)) * 2.0^79; rtol = 1e-6)            # (λ2/2)‖y‖²
  # the slopes: 1/(1 + γλ2) = 2^-140 (a subnormal Float32) where |v| exceeds the threshold
  g = ElasticNet(0.0, 2.0^70)
  @test ProximalBase._prox_block_slope(g, 2f0^100, 1, 2.0^70) == (2f0^-40, 2f0^-140)
  @test ProximalBase._prox_slope(ElasticNet(0f0, 2f0^70), 2f0^100, 2f0^70) == 2f0^-140
  @test ProximalBase._prox_slope(ElasticNet(2f0^10, 2f0^70), 2f0^5, 2f0^70) == 0f0       # below the threshold
  # γλ2 = 2^1200 in Float64: 2^1000/(1 + 2^1200) = 2^-200
  @test prox(ElasticNet(0.0, 2.0^600), [2.0^1000], 2.0^600) == [2.0^-200]
  # γλ1 overflows, the weight restores it: threshold 2^800·2^800·2^-1000 = 2^600
  e = ElasticNet(2.0^800, 0.0, [2.0^-1000])
  @test prox(e, [2.0^601], 2.0^800) == [2.0^600]
  @test prox_block(e, 2.0^601, 1, 2.0^800) == 2.0^600
  y = zeros(1); prox!(y, e, [2.0^601], [2.0^800]); @test y == [2.0^600]
  @test ProximalBase._prox_block_slope(e, 2.0^601, 1, 2.0^800) == (2.0^600, 1.0)
  # γλ1 underflows, the weight restores it: 2^-800·2^-800·2^1000 = 2^-600; x = 2^-599 gives 2^-600
  u = ElasticNet(2.0^-800, 0.0, [2.0^1000])
  @test prox(u, [2.0^-599], 2.0^-800) == [2.0^-600]
  @test prox(u, [-2.0^-599], 2.0^-800) == [-2.0^-600]
  @test prox(u, [2.0^-601], 2.0^-800) == [0.0]                                          # below the threshold
  # zero and infinite weights, and a zero or infinite level, keep their meaning on the typed path
  c = ProximalBase._weight_coefficient
  @test c(Float64, 2.0^800, 2.0^800, 0.0) === 0.0
  @test c(Float64, 2.0^800, 2.0^800, Inf) === Inf
  @test c(Float64, 0.0, 2.0^800, Inf) === Inf
  @test c(Float64, 2.0^800, 2.0^-1000, 2.0^800) === 2.0^600
  @test isnan(c(Float64, 0.0, Inf, 1.0)) && isnan(c(Float64, NaN, 1.0, 1.0)) && isnan(c(Float64, 1.0, 1.0, NaN))
  @test c(Float64, Inf, 2.0, 0.0) === 0.0 && c(Float64, Inf, 2.0, 1.0) === Inf
  @test c(Float64, 2.0^-800, 2.0^-800, 2.0^1000) === 2.0^-600
  # the ordinary path keeps its result: division by 1 + γλ2
  @test prox(ElasticNet(0.5, 0.25), [3.0], 0.5) == [(3.0 - 0.25) / (1 + 0.125)]
end

# γλw where the product is subnormal is rounded once, to nearest even, whatever the Julia version
# (ldexp truncates into the subnormal range up to Julia 1.10); the reference is exact in BigFloat.
@testset "γλw recovers and rounds correctly into the subnormal range" begin
  rng = MersenneTwister(2503)
  exact(T, γ, λ, w) = setprecision(BigFloat, 512) do
    T(BigFloat(γ) * BigFloat(λ) * BigFloat(w))
  end
  c = ProximalBase._weight_coefficient
  # (T, range of the exponent of γλ, range of the exponent of γ, range of the exponent of w)
  for (T, st, sg, sw) in ((Float64, -1100:-1030, -700:-400, 0:70), (Float32, -160:-135, -90:-60, 0:25))
    tested = subnormal = 0
    for trial in 1:4000
      eγ = rand(rng, sg); eλ = rand(rng, st) - eγ
      γ = (1 + rand(rng, T)) * T(2)^eγ
      λ = (1 + rand(rng, T)) * T(2)^eλ
      w = (1 + rand(rng, T)) * T(2)^rand(rng, sw)
      s = γ * λ
      (floatmin(T) <= s <= floatmax(T)) && continue          # the recovery applies only to such a γλ
      want = exact(T, γ, λ, w); tested += 1; (0 < want < floatmin(T)) && (subnormal += 1)
      @test c(T, γ, λ, w) === want
    end
    @test tested > 3000 && subnormal > 500
  end
  # the minimal subnormal times a half: 2^-1075·w rounds to 2^-1074 for w > 1 and to 0 for w < 1
  @test c(Float64, 2.0^-1074, 0.5, 1.5) === 2.0^-1074 && c(Float64, 2.0^-1074, 0.5, 0.75) === 0.0
  @test c(Float64, 2.0^-1074, 0.5, 1.0) === 0.0                                           # a tie goes to even (zero)
  @test c(Float32, 2f0^-149, 0.5f0, 1.5f0) === 2f0^-149 && c(Float32, 2f0^-149, 0.5f0, 0.75f0) === 0f0
end

@testset "NormL0: |v|² > 2γλ is compared where |v|² or 2γλ leaves the range" begin
  f = NormL0(2.0^100)
  @test prox(f, Float32[2.0^110], 2.0^100) == Float32[2.0^110]                        # 2^220 > 2^201
  @test prox(f, Float32[2.0^100], 2.0^100) == Float32[0]                              # 2^200 < 2^201
  @test prox_block(f, 2f0^110, 1, 2.0^100) === 2f0^110
  y = similar(Float32[1, 2]); prox!(y, f, Float32[2.0^110, 2.0^100], 2.0^100); @test y == Float32[2.0^110, 0]
  # Float32 factors and argument: 2γλ = 2^201 and |v|² = 2^220 overflow Float32 itself
  @test prox(NormL0(2f0^100), Float32[2f0^110, 2f0^100], 2f0^100) == Float32[2f0^110, 0]
  # Float64: 2γλ = 2^1200 and |v|² overflow Float64; the ties are zero, as in the ordinary path
  g = NormL0(2.0^499)
  @test prox(g, [2.0^600], 2.0^700) == [0.0]                                          # |v|² = 2γλ = 2^1200
  @test prox(g, [nextfloat(2.0^600)], 2.0^700) == [nextfloat(2.0^600)]
  @test prox(g, [prevfloat(2.0^600)], 2.0^700) == [0.0]
  # underflow: |v|² = 2^-1200 = 2γλ with γ = 2^-700, λ = 2^-501
  h = NormL0(2.0^-501)
  @test prox(h, [2.0^-600], 2.0^-700) == [0.0]
  @test prox(h, [nextfloat(2.0^-600)], 2.0^-700) == [nextfloat(2.0^-600)]
  # ordinary tie and non-finite values
  @test prox(NormL0(2.0), [2.0], 1.0) == [0.0] && prox(NormL0(2.0), [nextfloat(2.0)], 1.0) == [nextfloat(2.0)]
  # Inf is kept, and NaN is kept as NaN where the map depends on the argument (E2, #560)
  @test isequal(prox(NormL0(1.0), [Inf, NaN, 0.0], 1.0), [Inf, NaN, 0.0])
  @test prox(NormL0(1.0), [1.0], 0.0) == [1.0]                                        # γ = 0: identity
end

# NormL0 keeps v when |v|² > 2γλ exactly. The reference is an exact comparison in BigFloat (every product
# of three floats is exact in 512 bits); an exact tie gives zero. The exceptional path (2γλ outside the
# normal range) must agree with it for every near tie; the ordinary path rounds as before.
exact_keep(x, γ, λ) = setprecision(BigFloat, 512) do
  BigFloat(x)^2 > 2 * BigFloat(γ) * BigFloat(λ)
end
@testset "NormL0: exact comparison near a tie where 2γλ leaves the range" begin
  # the two counterexamples of the review: x² - 2γλ = (A - x)² > 0
  for e in (600, -600)
    A = 2.0^e; x = prevfloat(A); γ = prevfloat(x); λ = A / 2
    @test exact_keep(x, γ, λ)
    @test prox(NormL0(λ), [x], γ) == [x]
    @test prox(NormL0(λ), [-x], γ) == [-x]
    @test prox_block(NormL0(λ), x, 1, γ) === x
  end
  # exact ties (γ = x, λ = x/2 gives 2γλ = x² exactly), the value just above and just below
  for (T, es) in ((Float64, (600, 900, -600, -1000)), (Float32, (70, 100, -70, -100)))
    for e in es, m in (T(1), T(1.5), T(1.2345678))
      x = m * T(2)^e; γ = x; λ = x / 2
      @test !exact_keep(x, γ, λ)
      f = NormL0(λ)
      @test prox(f, [x, -x], γ) == [0, 0]                                                # a tie is zero
      @test prox(f, [nextfloat(x), -nextfloat(x)], γ) == [nextfloat(x), -nextfloat(x)]   # above is kept
      @test prox(f, [prevfloat(x)], γ) == [0]                                            # below is zero
      y = [x, nextfloat(x), prevfloat(x), -nextfloat(x)]
      @test prox(f, y, fill(γ, 4)) == [0, nextfloat(x), 0, -nextfloat(x)]                # prox!, vector steps
      z = copy(y); prox!(z, f, z, γ); @test z == [0, nextfloat(x), 0, -nextfloat(x)]    # in place
      w = similar(y); @test prox!(w, f, y, γ) === w; @test w == z
      @test prox_block(f, nextfloat(x), 1, γ) === nextfloat(x) && prox_block(f, x, 1, γ) === T(0)
    end
  end
  # a sweep of near ties at overflow and underflow scales, against the exact reference
  rng = MersenneTwister(503)
  for (T, hi, lo) in ((Float64, 513:1000, -1020:-512), (Float32, 65:120, -120:-64))
    kept = dropped = 0
    for trial in 1:3000
      e = rand(rng, (hi, lo)[rand(rng, 1:2)]); x = (1 + rand(rng, T)) * T(2)^e
      γ = (1 + rand(rng, T)) * T(2)^rand(rng, (e - 5):(e + 5))
      λ0 = setprecision(BigFloat, 512) do
        T(BigFloat(x)^2 / (2 * BigFloat(γ)))
      end
      λ = (λ0, nextfloat(λ0), prevfloat(λ0))[rand(rng, 1:3)]
      (isfinite(λ) && floatmin(T) <= λ) || continue
      sgn = rand(rng, Bool) ? x : -x
      want = exact_keep(x, γ, λ) ? sgn : zero(T)
      want == 0 ? (dropped += 1) : (kept += 1)
      f = NormL0(λ)
      @test prox_block(f, sgn, 1, γ) === want
      @test prox(f, [sgn], γ) == [want]
      z = [sgn]; prox!(z, f, z, γ); @test z == [want]
    end
    @test kept > 100 && dropped > 100                                                    # both outcomes occur
  end
  # mixed precision: Float32 factors with a Float64 argument, and Float64 factors with a Float32 argument
  @test prox(NormL0(floatmax(Float32)), [2.0^600], floatmax(Float32)) == [2.0^600]       # 2^1200 > 2^257
  @test prox(NormL0(2f0^-100), [2.0^-600], 2f0^-100) == [0.0]                            # 2^-1200 < 2^-199
  @test prox(NormL0(2.0^100), Float32[2f0^100, 1.5f0 * 2f0^100, -2f0^110], 2.0^100) ==
        Float32[0, 1.5f0 * 2f0^100, -2f0^110]                      # 2^200 < 2^201 < 2.25·2^200 < 2^220
  # zero and infinite steps and levels, NaN: the earlier contracts
  @test prox(NormL0(2.0^499), [2.0^600], 0.0) == [2.0^600]                              # γ = 0: identity
  @test prox(NormL0(0.0), [2.0^-600, 0.0], 2.0^700) == [2.0^-600, 0.0]                  # λ = 0: identity, zero stays zero
  @test prox(NormL0(Inf), [2.0^600], 2.0^700) == [0.0] && prox(NormL0(2.0^499), [2.0^600], Inf) == [0.0]
  @test isequal(prox(NormL0(2.0^499), [NaN, Inf], 2.0^700), [NaN, Inf])               # finite 2γλ outside the normal range: NaN stays NaN
end

@testset "LLA weights test the raw weight for infinity" begin
  lw = ProximalBase.lla_weight
  c = CappedL1(1e-80, 1.0, [1e80]); l = LogPenalty(1e-80, 1.0, [1e80])
  # the weight 1e80 is finite: Float32(1e80) is Inf, the coefficient γλw = 1 is formed in Float64
  @test lw(c, 3f0, 1) === 0f0 && lw(c, 0.5f0, 1) === 1f0
  @test lw(l, 3f0, 1) === 0.25f0 && lw(l, 0f0, 1) === 1f0
  # an infinite weight fixes the entry at every value
  ci = CappedL1(1e-80, 1.0, [Inf]); li = LogPenalty(1e-80, 1.0, [Inf])
  @test lw(ci, 3f0, 1) === Inf32 && lw(li, 3f0, 1) === Inf32 && isnan(lw(li, NaN32, 1))
  # a zero weight; a NaN weight (the constructors reject it, so the array is changed afterwards) and
  # a NaN argument propagate as before
  @test lw(CappedL1(1.0, 1.0, [0.0]), 0.5f0, 1) === 0f0 && lw(LogPenalty(1.0, 1.0, [0.0]), 0.5f0, 1) === 0f0
  wn = [1.0]; cn = CappedL1(1.0, 1.0, wn); ln = LogPenalty(1.0, 1.0, wn); wn[1] = NaN
  @test isnan(lw(cn, 0.5f0, 1)) && isnan(lw(ln, 0.5f0, 1))
  @test isnan(lw(LogPenalty(1.0, 1.0, [2.0]), NaN32, 1))
  # lla_weights! gives the same entries
  w = zeros(Float32, 2)
  lla_weights!(w, CappedL1(1e-80, 1.0, [1e80, 1e80]), Float32[3, 0.5]); @test w == Float32[0, 1]
  lla_weights!(w, LogPenalty(1e-80, 1.0, [1e80, 1e80]), Float32[3, 0.5]); @test w == Float32[0.25, 1 / 1.5]
  lla_weights!(w, CappedL1(1e-80, 1.0, [Inf, 1e80]), Float32[3, 3]); @test w == Float32[Inf, 0]
end

@testset "same-type Float32 stays Float32 and does not allocate" begin
  e, v = ElasticNet(0.5f0, 0.25f0), Float32[1, 2, 3]
  @test prox(e, v, 0.5f0) isa Vector{Float32}
  @test @inferred(prox_block(e, 1f0, 1, 0.5f0)) isa Float32
  prox_block(e, 1f0, 1, 0.5f0); @test (@allocated prox_block(e, 1f0, 1, 0.5f0)) == 0
  # the unweighted weight `true` does not widen the working type
  @test ProximalBase._working_type(Float32, 0.5f0, 0.5f0, true) === Float32
end
end
