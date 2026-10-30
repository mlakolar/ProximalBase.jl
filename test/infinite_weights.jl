module InfiniteWeightsTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PB = ProximalBase

# an infinite weight fixes the entry at zero at every level and step; a zero
# weight leaves the entry unpenalized at every level, including an infinite one

# the penalties, as constructors of a level and a weight array
penalties(T, λ, w) = (
  NormL1 = NormL1(T(λ), w),
  NormL1Nonneg = NormL1Nonneg(T(λ), w),
  ElasticNet = ElasticNet(T(λ), T(0.5), w),
  CappedL1 = CappedL1(T(λ), T(1), w),
  LogPenalty = LogPenalty(T(λ), T(0.5), w),
  NormLq = NormLq(T(λ), T(0.5), w),
  Berhu = Berhu(T(λ), T(1), w),
)

# the value of entry 1 alone at 0.5, for weight 1 and level 1
entry1(::Type{T}) where {T} = (
  NormL1 = T(0.5),
  NormL1Nonneg = T(0.5),
  ElasticNet = T(0.5 + 0.5 * 0.25 / 2),
  CappedL1 = T(0.5),
  LogPenalty = log(T(2)),
  NormLq = sqrt(T(0.5)),
  Berhu = T(0.5),
)

@testset "infinite weight, penalties" begin
  for T in (Float32, Float64), λ in (0, 1)
    fs = penalties(T, λ, T[1, Inf])
    for name in keys(fs)
      f = fs[name]
      nonneg = name === :NormL1Nonneg
      @testset "$name, $T, λ = $λ" begin
        @test PB.value_entry(f, T(0), 2) === T(0)
        @test PB.value_entry(f, T(0.5), 2) === T(Inf)
        if nonneg
          @test PB.value_entry(f, T(-0.5), 2) === T(Inf)
          @test PB.value_entry(f, T(NaN), 2) === T(Inf)
        else
          @test PB.value_entry(f, T(-0.5), 2) === T(Inf)
          @test isnan(PB.value_entry(f, T(NaN), 2))
        end
        for v in T[1, -1, floatmax(T), -floatmax(T)], γ in T[0, 0.5, 1]
          @test iszero(PB.prox_entry(f, v, 2, γ))
        end
        for γ in T[0, 0.5, 1]
          @test isnan(PB.prox_entry(f, T(NaN), 2, γ))
        end
        @test PB.kkt_violation(f, T[0.5, 0.5], 2) === T(-Inf)
        # arrays: entry 2 is fixed at zero
        if λ == 1
          @test f(T[0.5, 0]) ≈ entry1(T)[name]
        else
          @test f(T[0.5, 0]) ≈ (name === :ElasticNet ? T(0.5 * 0.25 / 2) : T(0))
        end
        @test f(T[0.5, 0.2]) === T(Inf)
        @test f(SparseIterate(sparsevec(T[0.5, 0]))) ≈ f(T[0.5, 0])
        @test f(SparseIterate(sparsevec(T[0.5, 0.2]))) === T(Inf)
        for γ in (T(1), T[0.5, 2])
          y = prox(f, T[1, 3], γ)
          @test y[2] === T(0)
        end
        y = zeros(T, 2)
        v = value_and_prox!(y, f, T[1, 3], T(1))
        @test isfinite(v)
        @test y[2] == 0
      end
    end
  end
end

@testset "zero scaling with infinite weights" begin
  for T in (Float32, Float64), w in (T[1, Inf], T[0, Inf])
    for f in (NormL1(one(T), w), NormL1Nonneg(one(T), w), ElasticNet(one(T), one(T), w))
      nonneg = f isa NormL1Nonneg
      family = nonneg ? NormL1Nonneg : f isa ElasticNet ? ElasticNet : NormL1
      for z in (Scaled(f, zero(T)), Scaled{typeof(f), T}(f, zero(T)),
                zero(T) * f, Scaled(Scaled(f, T(2)), zero(T)))
        @test z isa family
        @test z.w === w
        if z isa ElasticNet
          @test z.λ1 === zero(T) && z.λ2 === zero(T)
        else
          @test z.λ0 === zero(T)
        end
        # entry 2 is fixed; entry 1 is free, with an orthant constraint
        @test z(T[1, 0]) === zero(T)
        @test z(T[1, 2]) === T(Inf)
        for huge in (T(1e30), floatmax(T))
          @test z(T[huge, 0]) === zero(T)
        end
        @test z(T[-1, 0]) === (nonneg ? T(Inf) : zero(T))
        for x in (T[1, 3], T[-1, 3]), γ in (T(1), T[0.5, 2], zero(T))
          expected = T[nonneg ? max(x[1], 0) : x[1], 0]
          @test prox(z, x, γ) == expected
          y = similar(x)
          @test value_and_prox!(y, z, x, γ) === zero(T)
          @test y == expected
        end
        # the fixed entry is unstored in the sparse vector
        xsp = sparsevec([1], T[2], 2)
        @test nnz(xsp) == 1
        @test z(xsp) === zero(T)
        @test prox(z, xsp, one(T)) == T[2, 0]
        xi = SparseIterate(sparsevec(T[2, 3]))
        @test PB.prox_coordinate!(xi, z, 2, one(T)) === zero(T)
        @test xi[2] === zero(T)
        g = T[0.5, -0.5]
        expected = nonneg ? -g[1] : abs(g[1])
        @test PB.kkt_violation(z, g, 1) === expected
        @test PB.kkt_violation(z, g, 2) === T(-Inf)
        for x in (T[2, 0], xsp)
          @test PB.kkt_violation(z, g, x, 1) === expected
          @test PB.kkt_violation(z, g, x, 2) === T(-Inf)
        end
        for (indices, weights) in (([2], T[Inf]), ([1], T[w[1]]), ([2, 1], T[Inf, w[1]]))
          r = PB.restrict(z, indices)
          @test r isa family
          @test r.w == weights
          if r isa ElasticNet
            @test r.λ1 === zero(T) && r.λ2 === zero(T)
          else
            @test r.λ0 === zero(T)
          end
        end
        for y in (T[2, 3], T[-2, -3])
          polar = T[nonneg ? min(y[1], 0) : 0, y[2]]
          @test prox(Conjugate(z), y, one(T)) == polar
        end
        if z isa NormL1
          @test Conjugate(z)(T[0, 3]) === zero(T)
          @test Conjugate(z)(T[1, 0]) === T(Inf)
        end
      end
    end
  end

  for T in (Float32, Float64)
    # finite weights retain the existing simplifications
    for w in (nothing, T[1, 2], T[0, 1])
      @test Scaled(NormL1(one(T), w), zero(T)) isa ZeroFunction
      @test Scaled(ElasticNet(one(T), one(T), w), zero(T)) isa ZeroFunction
      @test Scaled(NormL1Nonneg(one(T), w), zero(T)) isa IndNonnegative
    end
    # non-finite levels retain the exact error
    w, groups = T[1, Inf], [[1], [2]]
    for f in (NormL1(T(Inf), w), NormL1Nonneg(T(Inf), w),
              ElasticNet(one(T), T(Inf), w), ElasticNet(T(Inf), one(T), w))
      @test_throws ArgumentError("zero scaling of $(typeof(f)) has no supported domain-preserving representation") Scaled(f, zero(T))
    end
    # a zero quadratic level must not multiply an overflowing square
    huge = floatmax(T) / 4
    for f in (ElasticNet(one(T), zero(T)), ElasticNet(one(T), zero(T), T[1, 2]))
      @test PB._has_ordinary_value(typeof(f))
      @test PB.value_entry(f, huge, 1) == huge
      @test f(T[huge, 0]) == huge
      @test isnan(f(T[NaN, 0]))
    end
  end
end

@testset "zero scaling of shaped separable penalties" begin
  for T in (Float32, Float64), F in (CappedL1, LogPenalty, NormLq, Berhu)
    shape = T(0.5)
    for w in (nothing, T[1, 0, 2, 3])
      f = F(one(T), shape, w)
      for z in (Scaled(f, zero(T)), Scaled{typeof(f), T}(f, zero(T)),
                zero(T) * f, f * zero(T), Scaled(Scaled(f, T(2)), zero(T)))
        @test z isa ZeroFunction
      end
    end
    @test_throws ArgumentError Scaled(F(T(Inf), shape), zero(T))
    if F === NormLq
      @test_throws ArgumentError F(one(T), T(Inf))
    else
      @test_throws ArgumentError Scaled(F(one(T), T(Inf)), zero(T))
    end
    w = T[1, Inf, 0, 2]
    f = F(one(T), shape, w)
    for z in (Scaled(f, zero(T)), Scaled{typeof(f), T}(f, zero(T)),
              zero(T) * f, f * zero(T), Scaled(Scaled(f, T(2)), zero(T)))
      @test z isa F
      @test z.λ === zero(T)
      @test getfield(z, 2) === shape
      @test z.w === w
      @test z(T[0.5, 0, 3, 1]) === zero(T)
      @test z(T[0.5, 2, 3, 1]) === T(Inf)
      g = T[0.5, -2, 3, -1]
      for i in 1:4
        @test value_entry(z, zero(T), i) === zero(T)
        @test value_entry(z, T(3), i) === (i == 2 ? T(Inf) : zero(T))
        @test isnan(value_entry(z, T(NaN), i))
        @test kkt_violation(z, g, i) === (i == 2 ? T(-Inf) : abs(g[i]))
        @test allocs(value_entry, z, T(3), i) == 0
        @test allocs(prox_entry, z, T(3), i, one(T)) == 0
        @test allocs(kkt_violation, z, g, i) == 0
      end
      for x in (T[0.5, 2, 3, -1], T[floatmax(T)/4, 2, -0.0, 1]),
          γ in (zero(T), one(T), T[0, 0.5, 2, 1])
        expected = copy(x)
        expected[2] = zero(T)
        @test isequal(prox(z, x, γ), expected)
        y = similar(x)
        @test value_and_prox!(y, z, x, γ) === zero(T)
        @test isequal(y, expected)
        @test allocs(prox!, y, z, x, γ) == 0
        @test allocs(value_and_prox!, y, z, x, γ) == 0
      end
      xi = SparseIterate(T[0.5, 2, 3, 1])
      for i in 1:4
        @test PB.prox_coordinate!(xi, z, i, one(T)) === (i == 2 ? zero(T) : T[0.5, 2, 3, 1][i])
      end
      for indices in ([2], [1, 3, 4], [2, 1])
        r = PB.restrict(z, indices)
        @test r isa F
        @test r.λ === zero(T)
        @test getfield(r, 2) === shape
        @test r.w == w[indices]
      end
      @test allocs(z, T[0.5, 0, 3, 1]) == 0
    end
  end
end

@testset "ordinary positive-level bit patterns" begin
  snapshots32 = (
    (0x3f0d9168, 0x3e8f5c29, 0x4019999a, 0x3feb8520),
    (0x3fb4bfd1, 0x3f7c01ab, 0x40148472, 0x3fa3d70b),
    (0x3faed84c, 0x3f5e17a5, 0x4014e68f, 0xff800000),
    (0x4068ded3, 0x405765fe, 0x3fd4279b, 0x3feb8520),
  )
  snapshots64 = (
    (0x3fe1b22d0e560418, 0x3fd1eb851eb851eb, 0x4003333333333333, 0x3ffd70a3d70a3d70),
    (0x3ff697fa16694f54, 0x3fef803556aee2c3, 0x4002908e3f946050, 0x3ff47ae147ae147b),
    (0x3ff5db096feba538, 0x3febc2f496ad75c0, 0x40029cd1f4415716, 0xfff0000000000000),
    (0x400d1bda5119ce07, 0x400aecbfb15b573e, 0x3ffa84f3454dca41, 0x3ffd70a3d70a3d70),
  )
  for (T, U, snapshots) in ((Float32, UInt32, snapshots32), (Float64, UInt64, snapshots64))
    for (F, expected) in zip((CappedL1, LogPenalty, NormLq, Berhu), snapshots)
      f = F(T(0.7), T(0.5), T[1.3, 0.8])
      x = T[0.3, 2.4]
      values = (f(x), value_entry(f, x[2], 2), prox_entry(f, x[2], 2, T(0.4)), kkt_violation(f, x, 2))
      @test reinterpret.(U, values) == expected
      @test value_entry(f, x[2], 2) === PB._value_entry_ordinary(f, x[2], 2)
      if F === LogPenalty
        @test lla_weight(f, x[2], 2) === f.λ * f.w[2] / (f.ε + abs(x[2]))
      end
    end
  end
end

@testset "extreme fixed and unpenalized entries" begin
  for T in (Float32, Float64)
    tiny, huge = nextfloat(zero(T)), floatmax(T)
    for λ in (zero(T), one(T))
      f = LogPenalty(λ, huge, T[Inf])
      @test value_entry(f, tiny, 1) === T(Inf)
      @test f(T[tiny]) === T(Inf)
      @test value_entry(f, zero(T), 1) === zero(T)
      @test lla_weight(f, huge, 1) === T(Inf)
      @test allocs(lla_weight, f, huge, 1) == 0
    end
    for f in (Berhu(zero(T), T(8)), Berhu(one(T), T(8), T[0]), Berhu(one(T), T(8)))
      steps = iszero(f.λ) || f.w !== nothing ? (zero(T), one(T)) : (zero(T),)
      for v in (T(9.3), huge / 4, -huge / 4, -zero(T)), γ in steps
        @test isequal(prox_entry(f, v, 1, γ), v)
      end
    end
    for λ in (zero(T), one(T)), w in (zero(T), one(T))
      f = NormLq(λ, T(0.5), T[w])
      expected = iszero(λ) || iszero(w) ? T(2) : T(-Inf)
      @test kkt_violation(f, T[-2], 1) === expected
    end
    @test kkt_violation(NormLq(zero(T), T(0.5)), T[-2], 1) === kkt_violation(ZeroFunction(), T[-2], 1)
    @test iszero(tiny * tiny)
    @test kkt_violation(NormLq(tiny, T(0.5), T[tiny]), T[2], 1) === T(-Inf)
    for F in (CappedL1, LogPenalty)
      f = F(zero(T), T(0.5), T[1, Inf, 0, 2])
      @test lla_weights(f, T[huge, huge, huge, huge]) == T[0, Inf, 0, 0]
      @test allocs(lla_weights!, zeros(T, 4), f, ones(T, 4)) == 0
    end
  end
end

@testset "infinite weight, lla weights" begin
  for T in (Float32, Float64), λ in (0, 1)
    fs = penalties(T, λ, T[1, Inf])
    for name in (:NormL1, :CappedL1, :LogPenalty)
      f = fs[name]
      for v in T[0, 0.5, 2]
        @test PB.lla_weight(f, v, 2) === T(Inf)
      end
      w = zeros(T, 2)
      PB.lla_weights!(w, f, T[0, 0])
      @test w[1] ≈ (name === :LogPenalty ? T(λ / 0.5) : T(λ))
      @test w[2] === T(Inf)
    end
  end
end

@testset "infinite weight, losses" begin
  for T in (Float32, Float64)
    w = T[1, Inf]
    for f in (CheckLoss(T(0.3), w), EpsInsensitiveLoss(T(0.1), w))
      @test PB.value_entry(f, T(0), 2) === T(0)
      @test PB.value_entry(f, T(0.5), 2) === T(Inf)
      @test PB.value_entry(f, T(-0.5), 2) === T(Inf)
    end
    f = CheckLoss(T(0.3), w)
    for γ in T[0, 0.5, 1], v in T[1, -1, Inf, -Inf]
      @test iszero(PB.prox_entry(f, v, 2, γ))
    end
    @test PB.kkt_violation(f, T[0.5, 0.5], 2) === T(-Inf)

    f = EpsInsensitiveLoss(T(0.1), w)
    @test PB.value_entry(f, T(0.05), 2) === T(0)
    @test PB.value_entry(f, T(0.1), 2) === T(0)
    for γ in T[0, 0.5, 1]
      @test PB.prox_entry(f, T(1), 2, γ) ≈ T(0.1)
      @test PB.prox_entry(f, T(-1), 2, γ) ≈ T(-0.1)
      @test PB.prox_entry(f, T(0.05), 2, γ) ≈ T(0.05)
      @test PB.prox_entry(f, T(Inf), 2, γ) ≈ T(0.1)
    end
    @test PB.kkt_violation(EpsInsensitiveLoss(T(0), w), T[0.5, 0.5], 2) === T(-Inf)

    f = HingeLoss(nothing, w)
    @test PB.value_entry(f, T(2), 2) === T(0)
    @test PB.value_entry(f, T(1), 2) === T(0)
    @test PB.value_entry(f, T(0.5), 2) === T(Inf)
    for γ in T[0, 0.5, 1]
      @test PB.prox_entry(f, T(0.3), 2, γ) ≈ T(1)
      @test PB.prox_entry(f, T(2), 2, γ) ≈ T(2)
    end
    f = HingeLoss(T[1, -1], w)
    @test PB.value_entry(f, T(-2), 2) === T(0)
    @test PB.value_entry(f, T(0), 2) === T(Inf)
    @test PB.prox_entry(f, T(0.3), 2, T(1)) ≈ T(-1)
  end
end

@testset "zero weight, penalties" begin
  for T in (Float32, Float64), λ in (0, 1, Inf)
    fs = penalties(T, λ, T[1, 0])
    for name in keys(fs)
      f = fs[name]
      nonneg = name === :NormL1Nonneg
      enet = name === :ElasticNet
      @testset "$name, $T, λ = $λ" begin
        for v in T[0.5, -0.5, Inf]
          expected = enet ? T(0.5) * v^2 / 2 : T(0)
          val = PB.value_entry(f, v, 2)
          if nonneg && v < 0
            @test val === T(Inf)
          else
            @test val == expected
          end
        end
        for v in T[0.3, 1, -2]
          expected = nonneg ? max(v, 0) : enet ? v / T(1.25) : v
          @test PB.prox_entry(f, v, 2, T(0.5)) ≈ expected
        end
        @test PB.kkt_violation(f, T[0.5, 0.5], 2) ≈ (nonneg ? T(-0.5) : T(0.5))
        if name in (:NormL1, :CappedL1, :LogPenalty)
          @test PB.lla_weight(f, T(0.3), 2) === T(0)
        end
      end
    end
  end
end

@testset "zero weight, losses" begin
  for T in (Float32, Float64), w in (T[1, 0],)
    for f in (CheckLoss(T(0.3), w), EpsInsensitiveLoss(T(0.1), w), HingeLoss(nothing, w), HingeLoss(T[1, -1], w))
      for v in T[0.5, -0.5, Inf]
        @test PB.value_entry(f, v, 2) === T(0)
      end
      for v in T[0.5, -0.5, 3]
        @test PB.prox_entry(f, v, 2, T(0.5)) ≈ v
      end
    end
  end
end

@testset "weight one at a zero level, penalties" begin
  for T in (Float32, Float64)
    fs = penalties(T, 0, T[1, 1])
    for name in keys(fs)
      f = fs[name]
      nonneg = name === :NormL1Nonneg
      enet = name === :ElasticNet
      @test PB.value_entry(f, T(0.5), 2) ≈ (enet ? T(0.5 * 0.25 / 2) : T(0))
      @test PB.prox_entry(f, T(0.7), 2, T(1)) ≈ (enet ? T(0.7 / 1.5) : T(0.7))
      @test PB.kkt_violation(f, T[0.5, 0.5], 2) ≈ (nonneg ? T(-0.5) : T(0.5))
      if name in (:NormL1, :CappedL1, :LogPenalty)
        @test PB.lla_weight(f, T(0.3), 2) === T(0)
      end
    end
  end
end

@testset "symmetric iterates" begin
  for T in (Float32, Float64)
    W = T[0 Inf; Inf 0]
    f = CappedL1(T(0.1), T(1), W)
    X = SymmetricSparseIterate(T[1 2; 2 1])
    @test f(X) === T(Inf)
    y = prox(f, X, 1)
    @test y[1, 2] === T(0) && y[2, 1] === T(0)
    @test y[1, 1] === T(1) && y[2, 2] === T(1)
    @test f(y) === T(0)

    g = NormL1(T(0), T[1 Inf; Inf 1])
    @test g(SymmetricSparseIterate(T[1 0; 0 1])) === T(0)
    @test g(SymmetricSparseIterate(T[1 2; 2 1])) === T(Inf)
  end
end

@testset "neighbouring functions" begin
  for T in (Float32, Float64)
    # box constraint
    c = Constrained(ElasticNet(T(0), T(0.5), T[1, Inf]), IndBox(-1, 1))
    y = prox(c, T[0.5, 0.5], 1)
    @test y ≈ T[1 / 3, 0]
    @test y[2] === T(0)

    # linear constraint, the second call takes the warm-started search
    c = Constrained(NormL1(T(0), T[1, Inf, 1]), IndHyperplane(ones(T, 3), T(1)))
    x = T[0.2, 0.7, 0.4]
    for _ in 1:2
      y = prox(c, x, 1)
      @test y ≈ T[0.4, 0, 0.6]
      @test y[2] === T(0)
    end

    # conjugate
    cj = Conjugate(NormL1(T(0), T[1, Inf]))
    @test cj(T[0, 0]) === T(0)
    @test cj(T[0, 5]) === T(0)
    @test cj(T[1, 0]) === T(Inf)

    # smoothing
    ℓ = PB.Smoothed(NormL1(T(0), T[Inf]), T(0.5))
    @test PB.deriv(ℓ, T(0.3), T(0)) ≈ T(0.6)
    @test PB.deriv2(PB.Smoothed(NormL1(T(Inf), T[0]), T(0.5)), T(0.3), T(0)) === T(0)
    @test PB.deriv2(PB.Smoothed(ElasticNet(T(Inf), T(0.7), T[0]), T(0.5)), T(0.3), T(0)) ≈ T((1 - 1 / (1 + 0.5 * 0.7)) / 0.5)

    # sphere
    y = zeros(T, 3)
    v = value_and_prox!(y, Constrained(NormL1(T(0), T[1, Inf, 1]), IndSphereL2(T(2))), T[0.3, 5, -0.2], 1)
    @test y ≈ T[0.3, 0, -0.2] .* (2 / sqrt(T(0.13)))
    @test y[2] == 0
    @test v == 0
    @test_throws ArgumentError value_and_prox!(zeros(T, 2), Constrained(NormL1(T(1), T[Inf, Inf]), IndSphereL2(T(1))), T[1, 1], 1)

    # conjugate of the epsilon-insensitive loss
    @test Conjugate(EpsInsensitiveLoss(T(0), T[1, Inf]))(T[0.5, Inf]) === T(0)
  end
end

@testset "infinite levels and weights, kkt of the losses, types" begin
  for T in (Float32, Float64)
    # an infinite level
    fs = penalties(T, Inf, T[1, Inf])
    for name in keys(fs)
      f = fs[name]
      @test PB.value_entry(f, T(0), 2) === T(0)
      @test PB.value_entry(f, T(0.5), 2) === T(Inf)
      @test iszero(PB.prox_entry(f, T(1), 2, T(1)))
    end

    # a zero level with an infinite argument
    for f in (CappedL1(T(0), T(1), T[1, 1]), LogPenalty(T(0), T(0.5), T[1, 1]),
              NormLq(T(0), T(0.5), T[1, 1]), Berhu(T(0), T(1), T[1, 1]))
      @test PB.value_entry(f, T(Inf), 2) === T(0)
    end
    @test PB.value_entry(ElasticNet(T(0), T(0.5), T[1, 1]), T(Inf), 2) === T(Inf)

    # kkt of the losses
    @test PB.kkt_violation(HingeLoss(nothing, T[1, Inf]), T[0.5, 0.5], 2) === T(Inf)
    @test PB.kkt_violation(EpsInsensitiveLoss(T(0.1), T[1, Inf]), T[0.5, -0.5], 2) ≈ T(0.5)

    # dense sums agree with the entrywise values
    w = T[1, Inf, 0, 2]
    ps = penalties(T, 1, w)
    cases = [
      (ps.NormL1, T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (ps.NormL1Nonneg, T[0.5, 0, 3, 1], T[0.5, 0.1, 3, 1]),
      (ps.ElasticNet, T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (ps.CappedL1, T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (ps.LogPenalty, T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (ps.NormLq, T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (ps.Berhu, T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (CheckLoss(T(0.3), w), T[0.5, 0, 3, -1], T[0.5, 0.1, 3, -1]),
      (EpsInsensitiveLoss(T(0.1), w), T[0.5, 0.05, 3, -1], T[0.5, 0.2, 3, -1]),
      (HingeLoss(nothing, w), T[0.5, 2, 3, -1], T[0.5, 0.5, 3, -1]),
    ]
    for (f, xfinite, xinf) in cases
      @test isfinite(f(xfinite))
      @test f(xfinite) ≈ sum(PB.value_entry(f, xfinite[i], i) for i in 1:4)
      @test f(xinf) === T(Inf)
    end
  end

  # integer levels keep integer kkt values
  @test PB.kkt_violation(ElasticNet(2, 1), [3], 1) === 1
  @test PB.kkt_violation(NormL1(2), [3], 1) === 1

  # BigFloat
  f = NormL1(big(0), BigFloat[1, Inf])
  @test iszero(PB.prox_entry(f, big(1), 2, big(1)))
  @test PB.value_entry(f, big(0.5), 2) == Inf
  @test PB.kkt_violation(f, BigFloat[1, 1], 2) == -Inf
end

end # module
