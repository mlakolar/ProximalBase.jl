module DeprecatedTest

# The interface of 0.4 still works, with deprecation warnings, until 0.6.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

Random.seed!(1)

@testset "deprecated" begin
  x = randn(6)
  out = similar(x)

  @testset "argument order and values" begin
    g = NormL1(0.5)
    @test (@test_deprecated ProximalBase.value(g, x)) == g(x)
    @test (@test_deprecated prox!(g, out, x, 0.3)) == prox(g, x, 0.3)
    @test (@test_deprecated prox!(g, out, x)) == prox(g, x)
    xs = SparseIterate(x)
    @test (@test_deprecated cdprox!(g, xs, 2, 0.3)) == shrink(x[2], 0.15)
    f = SqrNormL2()
    gr = similar(x)
    @test (@test_deprecated gradient!(f, gr, x)) == x
    @test (@test_deprecated value_and_gradient!(f, gr, x)) ≈ norm(x)^2 / 2
    fused = NormL1Fused(0.3, 0.5)
    @test (@test_deprecated ProximalBase.value(fused, 1., -2.)) ≈ 0.3 * 3 + 0.5 * 3
    pair = (zeros(3), zeros(3))
    @test (@test_deprecated prox!(fused, pair, (randn(3), randn(3)), 0.4)) === pair
  end

  @testset "types" begin
    # the old names are the new types
    @test ProximalBase.ProxL1 === NormL1
    @test ProximalBase.ProxZero === ZeroFunction
    @test ProximalBase.ProxL2 === NormL2
    @test ProximalBase.ProxGroupL2 === GroupNormL2
    @test ProximalBase.ProxNuclear === NuclearNorm
    @test ProximalBase.ProxL1Fused === NormL1Fused
    @test ProximalBase.ProxGaussLikelihood === LogDetLoss
    @test ProximalBase.DifferentiableFunction === AbstractFunction
    @test ProximalBase.ProximableFunction === AbstractFunction
    # with a changed convention: ProxL2Sq(λ) was λ‖x‖²
    f = @test_deprecated ProxL2Sq(0.5)
    @test f(x) ≈ 0.5 * norm(x)^2
    f = @test_deprecated L2Loss(x)
    @test f(2x) ≈ norm(x)^2 / 2
    f = @test_deprecated L2Loss()
    @test f(x) ≈ norm(x)^2 / 2
    # the weights were the field λ
    w = rand(6)
    @test (@test_deprecated NormL1(0.5, w).λ) === w
    @test (@test_deprecated GroupNormL2(0.5, [1:3, 4:6]).λ) == sqrt.([3., 3.])
    @test NormL1(0.5, w).λ0 == 0.5
  end

  @testset "iterates and utilities" begin
    xs = SparseIterate([0., 1., 0., 2.])
    @test (@test_deprecated numCoordinates(xs)) == 4
    @test (@test_deprecated numActive(xs)) == 2
    @test (@test_deprecated activeCoordinate(xs, 2)) == 4
    @test (@test_deprecated isActive(xs, 2))
    @test (@test_deprecated ind2subLowerTriangular(4, 3)) == tril_cartesian(4, 3)
    @test (@test_deprecated sub2indLowerTriangular(4, 3, 2)) == tril_index(4, 3, 2)
    @test (@test_deprecated proxL1Fused(0.3, -1.2, 0.2, 0.5, 0.7)) == prox(NormL1Fused(0.2, 0.5), (0.3, -1.2), 0.7)
    @test (@test_deprecated proxL1Fused(0.3, -1.2, 0.2, 0.5)) == prox(NormL1Fused(0.2, 0.5), (0.3, -1.2))
    @test (@test_deprecated shrinkL2!(similar(x), x, 0.4)) ≈ prox(NormL2(0.4), x)

    A = randn(10, 6)
    b, r = randn(6), randn(10)
    @test (@test_deprecated A_mul_B_row(A, b, 3)) ≈ (A * b)[3]
    @test (@test_deprecated At_mul_B_row(A, r, 4)) ≈ (A' * r)[4]
    G = [2, 5]
    @test (@test_deprecated At_mul_B_group!(zeros(2), A, r, G)) ≈ A[:, G]' * r
    @test (@test_deprecated A_mul_B_group!(zeros(10), A, [1., 2.], G)) ≈ A[:, G] * [1., 2.]
    @test (@test_deprecated A_mul_B_group!(copy(r), A, [1., 2.], G, -1, 1)) ≈ r - A[:, G] * [1., 2.]

    Σx, Σy = Symmetric(randn(5, 5)), Symmetric(randn(5, 5))
    Δ = SymmetricSparseIterate(5); Δ[3, 1] = 1.; Δ[4, 4] = -2.
    P = Σx * Matrix(Δ) * Σy
    @test (@test_deprecated A_mul_X_mul_B(Σx, Δ, Σy)) ≈ P
    @test (@test_deprecated A_mul_X_mul_B_rc(Σx, Δ, Σy, 2, 3)) ≈ P[2, 3]
    D = randn(5, 5)
    Dv = SparseIterate(vec(D))
    @test (@test_deprecated A_mul_X_mul_B(Σx, Dv, Σy)) ≈ Σx * D * Σy
    @test (@test_deprecated A_mul_X_mul_B_rc(Σx, Dv, Σy, 4, 1)) ≈ (Σx * D * Σy)[4, 1]
    U = randn(5, 2)
    @test (@test_deprecated A_mul_UUt_mul_B(Σx, U, Σy)) ≈ Σx * U * U' * Σy
    @test (@test_deprecated A_mul_UUt_mul_B_rc(Σx, U, Σy, 1, 5)) ≈ (Σx * U * U' * Σy)[1, 5]
  end
end

end
