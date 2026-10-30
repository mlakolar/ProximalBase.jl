module SpectralBoxToleranceTest

using Test
using ProximalBase
using LinearAlgebra
using Random

# a symmetric matrix with eigenvalues λ in a random orthonormal basis
function with_spectrum(rng, λ::Vector{T}) where {T}
  Q = Matrix(qr(randn(rng, T, length(λ), length(λ))).Q)
  X = Q * Diagonal(λ) * Q'
  (X + X') / 2
end

allocations(f, X) = (f(X); @allocated f(X))

@testset "box membership relative to the spectrum" begin
  for T in (Float32, Float64)
    @testset "$T: rank-deficient PSD matrices at every scale" begin
      rng = Xoshiro(240)
      M = floatmax(T)
      for s in (one(T), T(1e10), T(1e30), M / 64)
        for _ in 1:20
          B = randn(rng, T, 4, 2)
          X = B * B'
          X = s * (X / maximum(abs, X))
          X = (X + X') / 2
          @test IndPSD()(X) === zero(T)
          @test Spectral(IndBox(zero(T), T(Inf)))(X) === zero(T)
          @test Spectral(IndNonnegative())(X) === zero(T)
          @test SingularValue(IndBox(zero(T), T(Inf)))(X) === zero(T)
        end
      end
    end

    @testset "$T: an overflowing spectrum" begin
      rng = Xoshiro(2400)
      for _ in 1:20
        B = randn(rng, T, 4, 2)
        X = B * B'
        X = (floatmax(T) / 2) * (X / maximum(abs, X))
        X = (X + X') / 2
        @test IndPSD()(X) === zero(T)
        @test Spectral(IndBox(zero(T), T(Inf)))(X) === zero(T)
        @test Spectral(IndNonnegative())(X) === zero(T)
        @test Spectral(IndNonnegative())(-X) === T(Inf)
      end
    end

    @testset "$T: an overflowing spectrum, singular values" begin
      M = floatmax(T)
      L = M / 4
      t = sqrt(eps(T))
      X(d) = T[M M 0; M -M 0; 0 0 d]
      # singular values √2·M overflow; 3√ε·L < √ε·√2·M < 8√ε·L = 2√ε·M
      @test SingularValue(IndBox(L, T(Inf)))(X(L * (1 - 3t))) === zero(T)
      @test SingularValue(IndBox(L, T(Inf)))(X(L * (1 - 8t))) === T(Inf)
    end

    @testset "$T: a small eigenvalue next to a large one" begin
      rng = Xoshiro(24)
      big = T === Float64 ? T(1e10) : T(1e5)
      for _ in 1:20
        X = with_spectrum(rng, T[big, 1, 0, 0])
        @test Spectral(IndBox(zero(T), big))(X) === zero(T)
        @test Spectral(IndBox(T(-1), T(Inf)))(X) === zero(T)
        @test SingularValue(IndBox(zero(T), big))(X) === zero(T)
      end
    end

    @testset "$T: matrices outside stay outside" begin
      rng = Xoshiro(2401)
      for s in (one(T), T(1e10), floatmax(T) / 2)
        for _ in 1:10
          # an eigenvalue of -1e-3 relative to the largest one
          X = with_spectrum(rng, T[1, T(1e-1), T(-1e-3), 0])
          X = s * (X / maximum(abs, X))
          X = (X + X') / 2
          @test IndPSD()(X) === T(Inf)
          @test Spectral(IndBox(zero(T), T(Inf)))(X) === T(Inf)
          @test Spectral(IndNonnegative())(X) === T(Inf)
        end
      end
      for _ in 1:10
        # above the upper bound, or below the lower one, by 1e-3 relative
        X = with_spectrum(rng, T[2 + T(2e-3), 1, T(1.5), T(1.2)])
        @test Spectral(IndBox(one(T), T(2)))(X) === T(Inf)
        @test SingularValue(IndBox(one(T), T(2)))(X) === T(Inf)
        X = with_spectrum(rng, T[2, 1 - T(2e-3), T(1.5), T(1.2)])
        @test Spectral(IndBox(one(T), T(2)))(X) === T(Inf)
        @test SingularValue(IndBox(zero(T), T(2)))(X) === zero(T)
      end
    end

    @testset "$T: the tolerance at the smaller bound scales with the spectrum" begin
      # eigenvalues 2 and 1 - 1.5√ε lie in [1, 2] within √ε·max(1, |bound|, ‖λ‖∞) = 2√ε
      t = sqrt(eps(T))
      λ = T[2, 1 - T(1.5) * t]
      X = Matrix(Diagonal(λ))
      @test Spectral(IndBox(one(T), T(2)))(X) === zero(T)
      @test Spectral(IndBox(one(T), T(2)))(Matrix(Diagonal(T[2, 1 - 3t]))) === T(Inf)
      @test SingularValue(IndBox(1, 2))(Matrix(Diagonal(T[2, 1 - T(1.5) * t]))) === zero(T)
      @test SingularValue(IndBox(1, 2))(Matrix(Diagonal(T[2, 1 - 3t]))) === T(Inf)
      @test Spectral(IndNonnegative())(Matrix(Diagonal(T[2, -T(1.5) * t]))) === zero(T)
      @test Spectral(IndNonnegative())(Matrix(Diagonal(T[2, -3t]))) === T(Inf)
    end

    @testset "$T: empty and ordinary inputs" begin
      @test Spectral(IndNonnegative())(zeros(T, 0, 0)) === zero(T)
      @test Spectral(IndBox(zero(T), one(T)))(zeros(T, 0, 0)) === zero(T)
      @test Spectral(IndBox(-one(T), one(T)))(T[0 1; 1 0]) === zero(T)
      @test Spectral(IndBallLinf(one(T)))(T[0 2; 2 0]) === T(Inf)
    end

    @testset "$T: no allocations after warm-up" begin
      X = T[2 1; 1 2]
      for f in (Spectral(IndBox(zero(T), T(Inf))), Spectral(IndNonnegative()), SingularValue(IndBox(zero(T), T(Inf))))
        @test allocations(f, X) == 0
      end
    end
  end
end

end
