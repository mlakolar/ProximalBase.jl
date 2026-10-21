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

  # the same matrix with only its lower triangle stored; the other entries are NaN
  lowerstored(S) = Symmetric(tril(Matrix(S)) + triu(fill(NaN, size(S)), 1), :L)

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
  @test A_mul_X_mul_B(lowerstored(Σx), Δs, lowerstored(Σy)) ≈ A
  @test [A_mul_X_mul_B_rc(Σx, Δs, Σy, r, c) for r=1:50, c=1:50] ≈ A

  # very sparse X: the products use only the rows and columns of X that hold entries
  Δ = sparse([3, 7, 20, 7], [3, 2, 11, 7], [1.5, -2., 3., 0.5], 50, 50)
  Δs = SymmetricSparseIterate(Δ + Δ')
  A = Σx * Matrix(Δ + Δ') * Σy
  @test A_mul_X_mul_B(Σx, Δs, Σy) ≈ A
  @test A_mul_X_mul_B(lowerstored(Σx), Δs, lowerstored(Σy)) ≈ A
  Δs[11, 20] = 0.                      # stored zero, the only entry in rows 11 and 20
  @test A_mul_X_mul_B(Σx, Δs, Σy) ≈ Σx * Matrix(Δs) * Σy
  @test A_mul_X_mul_B(Σx, SymmetricSparseIterate(50), Σy) == zeros(50, 50)
  @test_throws DimensionMismatch A_mul_X_mul_B(Σx, SymmetricSparseIterate(10), Σy)

  # symmetric A * U * U' * B
  X = randn(100, 50)
  Σx = Symmetric(cov(X))
  Y = randn(100, 50)
  Σy = Symmetric(cov(Y))
  U = randn(50, 5)

  A = (Σx * U)*(U' * Σy)
  @test A_mul_UUt_mul_B(Σx, U, Σy) ≈ A
  @test A_mul_UUt_mul_B(lowerstored(Σx), U, lowerstored(Σy)) ≈ A
  @test [A_mul_UUt_mul_B_rc(Σx, U, Σy, r, c) for r=1:50, c=1:50] ≈ A

  # symmetric A * U * V' * B
  V = randn(50, 5)
  A = (Σx * U)*(V' * Σy)
  @test ProximalBase.A_mul_UVt_mul_B(Σx, U, V, Σy) ≈ A
  @test ProximalBase.A_mul_UVt_mul_B(lowerstored(Σx), U, V, lowerstored(Σy)) ≈ A
  @test [ProximalBase.A_mul_UVt_mul_B_rc(Σx, U, V, Σy, r, c) for r=1:50, c=1:50] ≈ A


  # asymmetric A * X * B
  X = randn(100, 50)
  Σx = Symmetric(cov(X))
  Y = randn(100, 50)
  Σy = Symmetric(cov(Y))
  Δ = sprand(50, 50, 0.1)
  A = Σx * Δ * Σy

  Δs = SparseIterate( Δ[:] )
  R = zeros(50, 50)
  for col=1:50
      for row=1:50
          R[row, col] = A_mul_X_mul_B_rc(Σx, Δs, Σy, row, col)
      end
  end

  @test R ≈ A
  @test A_mul_X_mul_B(Σx, Δs, Σy) ≈ A
  @test A_mul_X_mul_B(lowerstored(Σx), Δs, lowerstored(Σy)) ≈ A

  # very sparse X, including one full column and one full row, which take
  # the two multiplication orders
  for Δ in (sparse([3, 7, 20], [2, 2, 11], [1.5, -2., 3.], 50, 50),
            sparse(1:50, fill(4, 50), randn(50), 50, 50),
            sparse(fill(4, 50), 1:50, randn(50), 50, 50))
    @test A_mul_X_mul_B(Σx, SparseIterate(Δ[:]), Σy) ≈ Σx * Matrix(Δ) * Σy
  end
  @test A_mul_X_mul_B(Σx, SparseIterate(2500), Σy) == zeros(50, 50)
  @test_throws DimensionMismatch A_mul_X_mul_B(Σx, SparseIterate(100), Σy)
end

@testset "regressions" begin
  # At_mul_B_row with SparseIterate on a non-square matrix
  A = randn(3, 5)
  b = SparseIterate(randn(3))
  @test At_mul_B_row(A, b, 5) ≈ dot(A[:, 5], Vector(b))

  # vec2tril with SparseIterate
  x = SparseIterate(6)
  x[2] = 1.0
  x[6] = 2.0
  @test Matrix(ProximalBase.vec2tril(x, 3)) == [0. 0. 0.; 1. 0. 0.; 0. 0. 2.]

  # norm_diff
  A, B = randn(4, 3), randn(4, 3)
  @test norm_diff(A, B) ≈ norm(A - B)
  @test norm_diff(A, B, Inf) ≈ maximum(abs, A - B)

  # Vector of an empty SparseIterate
  @test Vector(SparseIterate(0)) == Float64[]

  @test_throws ArgumentError norm_diff(A, B, 1)
end

@testset "lower triangular helpers" begin
  p = 4
  M = sprandn(p, p, 0.5)
  L = tril(M)
  v = sparsevec([L[r, c] for c=1:p for r=c:p])
  @test ProximalBase.vec2tril(v, p) == L

  S = ProximalBase.tril2symmetric(L)
  @test S == L + L' - Diagonal(diag(L))
  @test issymmetric(S)
end

end
