module SpectralTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# reference implementations through LinearAlgebra
function nuclear_prox(X, c)
  F = svd(X)
  F.U * Diagonal(max.(F.S .- c, 0)) * F.Vt
end
function rank_proj(X, r)
  F = svd(X)
  F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
end
function psd_proj(X, f = t -> max(t, 0))
  E = eigen(Symmetric(X))
  E.vectors * Diagonal(f.(E.values)) * E.vectors'
end
function psd_rank_proj(X, r)
  E = eigen(Symmetric(X))
  n = size(X, 1)
  V = E.vectors[:, n-r+1:n]
  V * Diagonal(max.(E.values[n-r+1:n], 0)) * V'
end
symmetric(n) = (A = randn(n, n); (A + A') / 2)

@testset "NuclearNorm" begin
  for (m, n) in ((12, 8), (8, 12))
    X = randn(m, n)
    f = NuclearNorm(0.5)
    @test f(X) ≈ 0.5 * sum(svdvals(X))
    for γ in (0.1, 1., 10.)
      @test prox(f, X, γ) ≈ nuclear_prox(X, 0.5γ)
    end
    Z = check_prox(f, X, 0.8)
    @test value_and_prox!(similar(X), f, X, 0.8) ≈ f(Z)
  end
  # the old test: a positive semidefinite input
  y = randn(30, 10)
  y = y' * y
  @test prox(NuclearNorm(0.1), y) ≈ nuclear_prox(y, 0.1)
  # the workspace is kept while the size and type stay, and replaced otherwise
  f = NuclearNorm(1.)
  f(randn(5, 3))
  ws = f.ws
  f(randn(5, 3))
  @test f.ws === ws
  Y = randn(4, 4)
  @test f(Y) ≈ sum(svdvals(Y))                             # another size works
  @test f.ws !== ws
  X32 = randn(Float32, 6, 4)
  @test f(X32) isa Float32
  @test prox(f, X32, 0.5f0) ≈ nuclear_prox(X32, 0.5f0) rtol = 1e-4
  # outputs that are not strided, and inputs that are sparse iterates
  X = randn(6, 4)
  Ys = SparseIterate(6, 4)
  @test Matrix(prox!(Ys, NuclearNorm(0.3), X, 1.)) ≈ nuclear_prox(X, 0.3)
  @test NuclearNorm(0.3)(SparseIterate(X)) ≈ 0.3 * sum(svdvals(X))
  @test prox(NuclearNorm(100.), X) == zeros(6, 4)          # every singular value thresholded
  @test_throws ArgumentError NuclearNorm(-1.)
end

@testset "IndBallRank" begin
  X = randn(10, 7)
  f = IndBallRank(3)
  Z = prox(f, X)
  @test Z ≈ rank_proj(X, 3)
  @test rank(Z) == 3
  @test f(Z) == 0 && f(X) == Inf
  @test prox(IndBallRank(7), X) == X                       # no constraint
  @test IndBallRank(8)(X) == 0
  check_prox(f, X, 1.)
  @test value_and_prox!(similar(X), f, X, 1.) == 0
  @test_throws ArgumentError IndBallRank(0)
end

@testset "IndPSD" begin
  n = 9
  X = symmetric(n)
  f = IndPSD()
  Z = prox(f, X)
  @test Z ≈ psd_proj(X)
  @test Z == Z'
  @test f(Z) == 0 && f(X) == Inf
  @test f(randn(n, n) + 20I) == Inf                        # not symmetric
  @test prox(f, Matrix(4.0I, n, n)) ≈ 4I
  check_prox(f, X, 1.; symmetric = true)
  @test value_and_prox!(similar(X), f, X, 1.) == 0
  # the rank constraint: the largest eigenpairs only
  for r in (1, 3)
    fr = IndPSD(rank = r)
    Zr = prox(fr, X)
    @test Zr ≈ psd_rank_proj(X, r)
    @test rank(Zr; rtol = 1e-8) <= r
    @test fr(Zr) == 0
    check_prox(fr, X, 1.; symmetric = true)
  end
  @test prox(IndPSD(rank = n), X) ≈ psd_proj(X)
  P = randn(n, 2); P = P * P'
  @test IndPSD(rank = 2)(P) == 0 && IndPSD(rank = 1)(P) == Inf
  @test_throws ArgumentError IndPSD(rank = 0)
  # a SymmetricSparseIterate input, and an output that is not strided
  s = SymmetricSparseIterate(X)
  @test prox(f, Matrix(s)) ≈ prox!(similar(X), f, s, 1.)
  @test Matrix(prox!(SymmetricSparseIterate(n), f, X, 1.)) ≈ psd_proj(X)
  @test prox(f, Float32.(X)) ≈ psd_proj(X) rtol = 1e-4
end

@testset "TracePSD" begin
  n = 8
  X = symmetric(n) + 0.5I
  f = TracePSD(0.3)
  for γ in (0.5, 2.)
    @test prox(f, X, γ) ≈ psd_proj(X, t -> max(t - 0.3γ, 0))
  end
  Z = check_prox(f, X, 0.7; symmetric = true)
  @test f(Z) ≈ 0.3 * tr(Z)
  @test value_and_prox!(similar(X), f, X, 0.7) ≈ f(Z)
  @test f(X) == Inf || isposdef(X)
  @test f(Matrix(2.0I, n, n)) ≈ 0.3 * 2n
end

end
