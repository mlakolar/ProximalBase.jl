module UtilsTest

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays
using Statistics


@testset "shrink" begin

  @test shrink(3.0, 1.0) == 2.0
  @test shrink(-2.0, 1.0) == -1.0
  @test shrink(0.5, 1.0) == 0.

  Random.seed!(1)
  p = 10
  x = randn(p)
  c = norm(x) + 0.1
  @test shrinkL2!(similar(x), x, c) == zeros(p)
  c = c - 0.15
  @test shrinkL2!(similar(x), x, c) == max(1. - c / norm(x), 0,) * x

  x = randn(p, 2*p)
  c = norm(x) + 0.1
  @test shrinkL2!(similar(x), x, c) == zeros(p, 2*p)
  c = c - 0.15
  @test shrinkL2!(similar(x), x, c) == max(1. - c / norm(x), 0,) * x
end



@testset "multiply" begin

  Random.seed!(1)
  n, m = 100, 100
  A = randn(n, m)
  b = randn(m)
  x = A * b
  xt = A' * b
  for i=1:n
    @test A_mul_B_row(A, b, i) ≈ x[i]
    @test At_mul_B_row(A, b, i) ≈ xt[i]
  end

  bs = sprand(m, 0.3)
  x = A * bs
  xt = A' * bs
  bi = SparseIterate(bs)
  xi = A * bi
  xti = A' * bi
  for i=1:n
    @test A_mul_B_row(A, bs, i) ≈ x[i]
    @test At_mul_B_row(A, bs, i) ≈ xt[i]

    @test A_mul_B_row(A, bi, i) ≈ x[i]
    @test At_mul_B_row(A, bi, i) ≈ xt[i]

    @test xi[i] ≈ x[i]
    @test xti[i] ≈ xt[i]
  end

  # symmetric A * X * B
  X = randn(100, 50)
  Σx = Symmetric(cov(X))
  Y = randn(100, 50)
  Σy = Symmetric(cov(Y))
  Δ = sprand(50, 50, 0.1)
  Δ = (Δ + Δ') / 2.

  A = Σx * Δ * Σy
  Δs = SymmetricSparseIterate(Δ)
  @test A_mul_X_mul_B(Σx, Δs, Σy) ≈ A

  # symmetric A * U * U' * B
  X = randn(100, 50)
  Σx = Symmetric(cov(X))
  Y = randn(100, 50)
  Σy = Symmetric(cov(Y))
  U = randn(50, 5)

  A = (Σx * U)*(U' * Σy)
  @test A_mul_UUt_mul_B(Σx, U, Σy) ≈ A
end

end
