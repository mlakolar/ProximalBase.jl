module SpectralToleranceTest

using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils

const PB = ProximalBase

@testset "membership on a computed spectrum" begin
  rng = MersenneTwister(283)
  for T in (Float32, Float64)
    τ = sqrt(eps(T))
    Q = Matrix(qr(randn(rng, T, 8, 8)).Q)
    P = Q[:, 1:5] * Q[:, 1:5]'
    P = (P + P') / 2
    scales = (one(T), T(1e6), T(1e10), T === Float32 ? T(2)^100 : T(2)^500)
    for c in scales, wrap in (Spectral, SingularValue)
      X = c .* P
      for f in (IndNonnegative(), IndBox(0, c), IndBallLinf(c), IndBinary(0, c),
                IndSimplex(5c), IndSimplex(5c; equality = false),
                IndBallL1(5c), IndBallL2(sqrt(T(5)) * c),
                Conjugate(NormL1(c)), Conjugate(NormL2(sqrt(T(5)) * c)),
                Conjugate(NormLinf(5c)), Conjugate(SumLargest(5, c)),
                IndHyperplane(one(T), 5c), IndHalfspace(one(T), 5c),
                IndHyperslab(one(T), 4c, 5c))
        @test wrap(f)(X) == zero(T)
      end
      @test wrap(NormL1Nonneg(one(T)))(X) ≈ 5c rtol = 20eps(T)
      @test wrap(Constrained(NormL1(one(T)), IndBox(0, c)))(X) ≈ 5c rtol = 20eps(T)
      @test wrap(Constrained(NormL1(one(T)), IndHyperplane(one(T), 5c)))(X) ≈ 5c rtol = 20eps(T)
      @test wrap(ZeroFunction())(X) == zero(T)
      @test wrap(Conjugate(ZeroFunction()))(zeros(T, 8, 8)) == zero(T)
      # Residual 100√ε; tolerance √ε since the spectrum's maximum is below 1.
      @test wrap(Conjugate(ZeroFunction()))(Matrix(Diagonal(T[100τ, 0]))) == T(Inf)

      # Residual δ = 100√ε c/(1 - 100√ε); tolerance √ε(c + δ).
      δ = 100τ * c / (1 - 100τ)
      Y = Matrix(Diagonal(T[c + δ, 0]))
      for f in (IndBox(0, c), IndBallLinf(c), IndBinary(0, c),
                IndBallL1(c), IndBallL2(c), Conjugate(NormL1(c)),
                Conjugate(NormL2(c)), Conjugate(NormLinf(c)),
                IndSimplex(c), IndSimplex(c; equality = false),
                Constrained(NormL1(one(T)), IndBox(0, c)))
        @test wrap(f)(Y) == T(Inf)
      end
      # Linear residual d = 100√ε √2(c + d); tolerance √ε √2(c + d).
      for f in (IndHyperplane(one(T), c), IndHalfspace(one(T), c),
                IndHyperslab(one(T), 0, c),
                Constrained(NormL1(one(T)), IndHyperplane(one(T), c)))
        # With one nonzero, the normal term is √2 times that entry.
        d = 100τ * sqrt(T(2)) * c / (1 - 100τ * sqrt(T(2)))
        @test wrap(f)(Matrix(Diagonal(T[c + d, 0]))) == T(Inf)
      end
      # Residual 100√ε c; the binary lower-bound tolerance is √ε c.
      @test wrap(IndBinary(0, c))(Matrix(Diagonal(T[c, 100τ * c]))) == T(Inf)
      if wrap === Spectral
        # Negativity residual 100√ε c; tolerance √ε c.
        Z = Matrix(Diagonal(T[c, -100τ * c]))
        for f in (IndNonnegative(), NormL1Nonneg(one(T)),
                  Constrained(NormL1(one(T)), IndBox(0, Inf)))
          @test wrap(f)(Z) == T(Inf)
        end
        # Sum stays c. Negativity exceeds 100 times √ε times the new maximum.
        Z = Matrix(Diagonal(T[c + δ, -δ]))
        @test wrap(IndSimplex(c))(Z) == T(Inf)
        @test wrap(Conjugate(SumLargest(1, c)))(Z) == T(Inf)
        # Box is satisfied; sum residual 100√ε c, tolerance √ε c.
        @test wrap(Conjugate(SumLargest(1, c)))(Matrix(Diagonal(T[c, 100τ * c]))) == T(Inf)
        @test wrap(Conjugate(IndNonnegative()))(-X) == zero(T)
        # Positive residual 100√ε c; tolerance √ε c.
        @test wrap(Conjugate(IndNonnegative()))(Matrix(Diagonal(T[-c, 100τ * c]))) == T(Inf)
      else
        @test wrap(Conjugate(IndNonnegative()))(zeros(T, 8, 8)) == zero(T)
        # A singular value 100√ε is outside the nonpositive orthant by 100 tolerances.
        @test wrap(Conjugate(IndNonnegative()))(reshape(T[100τ], 1, 1)) == T(Inf)
      end
      for f in (NormL1(c), NormL2(sqrt(T(5)) * c), NormLinf(5c))
        @test Conjugate(SingularValue(f))(X) == zero(T)
      end
      for f in (NormL1(c), NormL2(c), NormLinf(c))
        # Dual norm residual δ; tolerance √ε(c + δ).
        @test Conjugate(SingularValue(f))(Y) == T(Inf)
      end
    end
  end
end

@testset "scaled units and forwarding" begin
  for T in (Float32, Float64)
    M = floatmax(T)
    X = fill(M, 2, 2)
    N = -X
    for wrap in (Spectral, SingularValue)
      @test wrap(IndBinary(0, M))(X) == T(Inf)
      @test wrap(IndSimplex(M))(X) == T(Inf)
      @test wrap(IndSimplex(T(Inf); equality = false))(X) == zero(T)
      @test wrap(Conjugate(NormL1(M)))(X) == T(Inf)
      @test wrap(Conjugate(NormL2(M)))(X) == T(Inf)
      @test wrap(Conjugate(NormLinf(M)))(X) == T(Inf)
      @test wrap(Conjugate(SumLargest(1, M)))(X) == T(Inf)
      @test wrap(Conjugate(ZeroFunction()))(X) == T(Inf)
      @test wrap(Constrained(NormL0(1), IndBox(0, M)))(X) == T(Inf)
      @test wrap(Scaled(IndBox(0, M), T(2)))(X) == T(Inf)
      # Subnormal negative entries are within the absolute domain tolerance.
      u = nextfloat(zero(T))
      @test wrap(Constrained(NormL0(1), IndBox(0, Inf)))(-u .* T[1 1; 1 2]) == T(2)
      @test wrap(Constrained(NormL0(1), IndBox(0, 1)))(-u .* T[1 1; 1 2]) == T(2)
      for f in (ZeroFunction(), IndNonnegative(), NormL1(M), NormL2(M), NormLinf(M), SumLargest(1, M))
        @test Conjugate(SingularValue(f))(X) == T(Inf)
        @test Conjugate(SingularValue(f))(zeros(T, 2, 2)) ==
          (f isa SumLargest ? T(Inf) : zero(T))
      end
      # Preserve the value type through a scalar multiplier of wider precision.
      @test wrap(Scaled(NormL1Nonneg(one(T)), 2.0))(Matrix{T}(I, 2, 2)) === T(4)
      # Norm residual is just above the old tolerance √ε, within √ε times the spectrum.
      x = nextfloat(one(T) + sqrt(eps(T)))
      for f in (IndBallL1(one(T)), IndBallL2(one(T)),
                Conjugate(NormL1(one(T))), Conjugate(NormL2(one(T))), Conjugate(NormLinf(one(T))))
        @test f(T[x]) == T(Inf)
        @test wrap(f)(reshape(T[x], 1, 1)) == zero(T)
      end
    end
    @test Spectral(Conjugate(IndNonnegative()))(N) == zero(T)
    @test Spectral(Conjugate(IndNonnegative()))(X) == T(Inf)
    # Overflowing positive and negative eigenvalues cancel; the normal tolerance is retained.
    Z = T[M M 0 0; M M 0 0; 0 0 -M -M; 0 0 -M -M]
    for f in (IndHyperplane(one(T), 0), IndHalfspace(one(T), 0), IndHyperslab(one(T), 0, 0))
      @test Spectral(f)(Z) == zero(T)
    end
  end
end

@testset "ordinary spectral values retain their bits" begin
  rng = MersenneTwister(28302)
  for T in (Float32, Float64)
    B = T === Float32 ? UInt32 : UInt64
    A = randn(rng, T, 16, 3)
    P = A * A'
    P = (P + P') / 2
    Q = Matrix(qr(randn(rng, T, 16, 16)).Q)
    H = Q * Diagonal(collect(range(T(0.5), T(2); length = 16))) * Q'
    H = (H + H') / 2
    f = NormL1Nonneg(0.5)
    functions = (f, Scaled(f, 1.3),
      Constrained(f, IndBox(0, Inf)),
      Constrained(Scaled(f, 1.3), IndBox(0, Inf)),
      Scaled(Constrained(f, IndBox(0, Inf)), 1.3),
      Constrained(f, IndHalfspace(one(T), T(1e35))))
    for c in (T(1e-30), one(T), T(1e10)), g in functions
      X = c .* P
      F = SingularValue(g)
      # Use the same SVD workspace and routine as the wrapper's value call.
      v = copy(PB._singular_values!(F, X))
      @test all(x -> x >= -PB._tol(T), v)
      @test isfinite(g(v))
      @test reinterpret(B, F(X)) == reinterpret(B, g(v))
      X = c .* H
      S = Spectral(g)
      actual = S(X)
      v = copy(S.eig.w)
      # Every eigenvalue is separated from zero by at least a quarter of c.
      @test minimum(v) > c / 4
      @test isfinite(g(v))
      @test reinterpret(B, actual) == reinterpret(B, g(v))
    end
    # Entries accepted by the ordinary negativity test also keep their bits.
    v = T[-PB._tol(T) / 2, 1, 2, 3]
    @test reinterpret(B, PB._spectrum_value(f, v, 0)) == reinterpret(B, f(v))
    # A relaxed domain must still reach the clamped value through compositions.
    X = Matrix(Diagonal(T[1e10, -1]))
    @test Spectral(Scaled(f, 1.3))(X) ≈ T(6.5e9)
    @test Spectral(Constrained(f, IndBox(0, Inf)))(X) == T(5e9)
    @test Spectral(Constrained(f, IndHalfspace(one(T), T(2e10))))(X) == T(5e9)
  end
end

@testset "fixed spectral tolerance regressions" begin
  rng = MersenneTwister(283)
  Q = Matrix(qr(randn(rng, 9, 9)).Q)
  P = Q[:, 1:5] * Q[:, 1:5]'
  P = (P + P') / 2
  @test Spectral(IndBinary(0, 1e10))(1e10P) == 0
  @test Spectral(IndSimplex(5e10))(1e10P) == 0
  @test Spectral(Conjugate(IndNonnegative()))(Matrix(Diagonal([-1e10, 1e-6]))) == 0
  X = Matrix(Diagonal([1e10, -1.0]))
  @test Spectral(NormL1Nonneg(1))(X) == 1e10
  @test Spectral(NormL1Nonneg(1))(X) ≈ TracePSD(1)(X)
  @test SingularValue(IndBallL2(1e201))(reshape([1e200], 1, 1)) == 0
  M = floatmax(Float64)
  @test Spectral(IndBallL1(M))(Matrix(Diagonal(fill(M, 2)))) == Inf
  @test Conjugate(SingularValue(NormLinf(1)))(Matrix(Diagonal([1; fill(1e-9, 49)]))) == Inf
  # Residual 2^-10; cancellation tolerance is approximately 32.
  X = Matrix(Diagonal([1, -1 + 2.0^-40]))
  for f in (IndHyperplane(2.0^30, 0), IndHalfspace(2.0^30, 0),
            IndHyperslab(2.0^30, 0, 0))
    @test Spectral(f)(X) == 0
  end
  # Vector values retain the standalone tolerance.
  @test Conjugate(IndNonnegative())([-1e10, 1e-6]) == Inf
  @test NormL1Nonneg(1)([1e10, -1.0]) == Inf
  F = SingularValue(NormL0(1))
  Y = zeros(3, 3)
  @test value_and_prox!(Y, F, Matrix(Diagonal([2.0^60, 2, 0])), 1) == 2
  @test F(Y) == 1
end

end
