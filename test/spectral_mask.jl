module SpectralMaskTest

using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils

const PB = ProximalBase

@testset "the spectral counting trait" begin
  @test !PB._counts_nonzeros(NormL1(1))
  @test !PB._counts_nonzeros(Conjugate(NormLinf(1)))
  @test PB._counts_nonzeros(NormL0(1))
  @test PB._counts_nonzeros(IndBallL0(2))
  @test PB._counts_nonzeros(Scaled(NormL0(1), 2))
  @test PB._counts_nonzeros(Constrained(NormL0(1), IndBox(0, Inf)))
  @test !PB._counts_nonzeros(Constrained(NormL1(1), IndHalfspace(1, 2)))
end

@testset "scale invariance of counting functions" begin
  @test !PB._scale_invariant(NormL1(1))
  @test PB._scale_invariant(NormL0(1))
  @test PB._scale_invariant(IndBallL0(2))
  @test PB._scale_invariant(Scaled(NormL0(1), 2))
  @test PB._scale_invariant(Constrained(NormL0(1), IndBox(0, Inf)))
  @test PB._scale_invariant(Constrained(NormL0(1), IndBox(-Inf, 0)))
  @test !PB._scale_invariant(Constrained(NormL0(1), IndBox(1, 3)))
  @test !PB._scale_invariant(Scaled(Constrained(NormL0(1), IndBox(1, 3)), 2))
  @test !PB._scale_invariant(Constrained(NormL1(1), IndHalfspace(1, 2)))
  for T in (Float32, Float64), wrapper in (SingularValue, Spectral)
    d, M = nextfloat(zero(T)), floatmax(T)
    f = wrapper(Constrained(NormL0(1), IndBox(1, 3)))
    @test f(d .* Matrix{T}(I, 2, 2)) == Inf
    @test f(Matrix(Diagonal(T[1.5, 2.5]))) == 2
    @test f(fill(M, 2, 2)) == Inf
    g = wrapper(Constrained(NormL0(1), IndBox(0, Inf)))
    @test g(d .* Matrix{T}(I, 2, 2)) == 2
    # Scaling resolves the smaller eigenvalue rounded to zero in original units.
    @test g(d .* T[1 1; 1 2]) == 2
    @test g(fill(M, 2, 2)) == 1
    h = wrapper(Constrained(NormL0(1), IndBox(0, 3)))
    @test h(d .* Matrix{T}(I, 3, 3)) == 3
    @test h(d .* Matrix{T}(I, 2, 2)) == 2
  end
end

@testset "mask regression cases" begin
  Random.seed!(1)
  Q = Matrix(qr(randn(50, 50)).Q)
  X = Matrix(Symmetric(Q * Diagonal([1; fill(1e-9, 49)]) * Q'))
  X = (X + X') / 2
  v = SingularValue(NormL1(1))(X)
  # Compare at the rounded input; Q changes its spectrum across platforms.
  @test abs(v - NuclearNorm(1)(X)) <= 4eps(Float64) * sum(svdvals(X))
  @test Spectral(NormL0(1))(X) == 1
  @test Conjugate(SingularValue(NormLinf(1)))(X) == Inf
  Random.seed!(2)
  U = Matrix(qr(randn(40, 40)).Q)[:, 1:5]
  P = U * U'
  P = (P + P') / 2
  for c in (1, 1e6, 1e10)
    @test Spectral(IndBallL0(5))(c * P) == 0
  end
  @test Spectral(Constrained(NormL0(1), IndBox(0, Inf)))(Diagonal([1.0, 1e-9])) == 1
  @test SingularValue(NormL0(1))(nextfloat(0.0) .* [1.0 1; 1 2]) == 2
  @test IndBallRank(1)(Diagonal([1e-10, 1e-11])) == Inf
  @test IndPSD(rank = 1)(Diagonal([1e-10, 1e-11])) == Inf
  # The absolute tolerance in the PSD sign test stays unchanged.
  @test IndPSD(rank = 1)(Diagonal([-1e-10, 1e-10])) == 0
  for c in (1.0, 1e-10)
    @test IndBallRank(1)(c .* Diagonal([1.0, 1e-9])) == 0
  end
end

@testset "unmasked continuous spectra" begin
  rng = MersenneTwister(283)
  for T in (Float32, Float64)
    A = randn(rng, T, 9, 7)
    L = randn(rng, T, 9, 2) * randn(rng, T, 2, 7)
    Q = Matrix(qr(randn(rng, T, 7, 7)).Q)
    G = Q * Diagonal(T[1, 0.1, 0.01, 1e-4, 1e-6, 1e-8, 0]) * Q'
    for X in (A, L, G), λ in (T(0.3), T(2))
      @test abs(SingularValue(NormL1(λ))(X) - NuclearNorm(λ)(X)) <=
        4abs(λ) * eps(T) * sum(svdvals(X))
    end
  end
end

@testset "relative counts at every scale" begin
  # The positive eigenvalues are (3 ± √5)/2; the third is exactly zero.
  # Their ratio is above 0.1, far from √ε even after subnormal rounding.
  X = [1.0 1 0; 1 2 0; 0 0 0]
  for k in (-1070, -600, 0, 600, 1000)
    Z = 2.0^k .* X
    @test all(isfinite, Z) && !iszero(Z)
    for f in (NormL0(1), IndBallL0(2), IndBallL0(1),
              Constrained(NormL0(1), IndBox(0, Inf)), Scaled(NormL0(1), 2))
      expected = f isa IndBallL0 ? (f.s == 2 ? 0 : Inf) : (f isa Scaled ? 4 : 2)
      @test Spectral(f)(Z) == SingularValue(f)(Z) == expected
    end
    for r in (1, 2)
      expected = r == 2 ? 0 : Inf
      @test IndBallRank(r)(Z) == SingularValue(IndBallL0(r))(Z) == expected
      @test IndPSD(rank = r)(Z) == expected
    end
  end
  # The largest magnitude may be the negative end of the sorted spectrum.
  @test Spectral(NormL0(1))(Diagonal([-1.0, -1e-9, 0.0])) == 1
  @test Spectral(Constrained(NormL0(1), IndBox(0, Inf)))(Diagonal([-1.0, 1e-9])) == Inf
  for T in (Float32, Float64)
    M, u = floatmax(T), nextfloat(zero(T))
    for Z in (fill(M, 2, 2), T[1 1; 1 2] .* u)
      expected = Z[1, 1] == M ? 1 : 2
      @test Spectral(NormL0(1))(Z) == SingularValue(NormL0(1))(Z) == expected
      @test IndBallRank(1)(Z) == IndPSD(rank = 1)(Z) == (expected == 1 ? 0 : Inf)
    end
  end
end

end
