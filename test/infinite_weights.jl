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
        for v in T[1, -1, Inf, -Inf], γ in T[0, 0.5, 1]
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
        for v in T[0.5, -0.5, Inf, NaN]
          expected = enet ? T(0.5) * v^2 / 2 : T(0)
          val = PB.value_entry(f, v, 2)
          if nonneg && (v < 0 || isnan(v))
            @test val === T(Inf)
          elseif isnan(expected)
            @test isnan(val)
          else
            @test val == expected
          end
        end
        for v in T[0.3, 1, -2]
          expected = nonneg ? max(v, 0) : enet ? v / T(1.25) : v
          @test PB.prox_entry(f, v, 2, T(0.5)) ≈ expected
        end
        # the slope of NormLq at zero is infinite, whatever its weight
        @test PB.kkt_violation(f, T[0.5, 0.5], 2) ≈ (name === :NormLq ? T(-Inf) : nonneg ? T(-0.5) : T(0.5))
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
      for v in T[0.5, -0.5, Inf, NaN]
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
      @test PB.kkt_violation(f, T[0.5, 0.5], 2) ≈ (name === :NormLq ? T(-Inf) : nonneg ? T(-0.5) : T(0.5))
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

end # module
