module CalculusTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

@testset "Scaled" begin
  x = randn(8)
  f = NormL1(0.5)
  s = Scaled(f, 3.)
  @test 3 * f isa Scaled && f * 3 isa Scaled
  @test (3 * f)(x) ≈ s(x) ≈ 1.5 * sum(abs, x)
  @test prox(s, x, 0.4) ≈ prox(f, x, 1.2)
  @test 2 * s isa Scaled && (2 * s).α == 6. && (2 * s).f === f      # scaling twice multiplies
  check_prox(s, x, 0.4)
  @test value_and_prox!(similar(x), s, x, 0.4) ≈ s(prox(s, x, 0.4))
  # a scaled loss: the gradient scales
  X, y = randn(20, 8), randn(20)
  ls = (1 / 20) * LeastSquaresLoss(y, X)
  @test ls(x) ≈ norm(y - X * x)^2 / 40
  check_gradient(ls, x)
  @test issmooth(ls) && isproximable(ls)
  @test prox(ls, x, 2.) ≈ prox(LeastSquaresLoss(y, X), x, 0.1)
  # kkt_violation and restrict
  grad = randn(8)
  @test kkt_violation(s, grad, 3) ≈ abs(grad[3]) - 1.5
  @test kkt_violation(0 * f, grad, 3) == abs(grad[3])
  @test restrict(s, [2, 5])(x[[2, 5]]) ≈ s([0, x[2], 0, 0, x[5], 0, 0, 0])
  @test cdprox(s, 2., 1, 0.4) ≈ shrink(2., 0.6)
  @test_throws ArgumentError Scaled(f, -1.)
end

@testset "Tilt" begin
  x = randn(8)
  c = randn(8)
  for f in (NormL1(0.4), SqrNormL2(1.5), MCP(0.3))
    t = Tilt(f, c, 2.)
    @test t(x) ≈ f(x) + dot(c, x) + 2
    @test prox(t, x, 0.6) ≈ prox(f, x - 0.6c, 0.6)
    check_prox(t, x, 0.6)
    @test [cdprox(t, x[i], i, 0.6) for i in 1:8] ≈ prox(t, x, 0.6)
  end
  t = Tilt(SqrNormL2(1.5), c)
  check_gradient(t, x)
  @test gradient(t, x) ≈ 1.5x + c
  # at zero, the tilt shifts the gradient
  grad = randn(8)
  @test kkt_violation(Tilt(NormL1(0.4), c), grad, 2) ≈ abs(grad[2] + c[2]) - 0.4
  @test restrict(t, [1, 3])(x[[1, 3]]) ≈ t([x[1], 0, x[3], 0, 0, 0, 0, 0])
end

@testset "Translate" begin
  x = randn(8)
  b = randn(8)
  f = NormL1(0.4)
  t = Translate(f, b)
  @test t(x) ≈ 0.4 * sum(abs, x - b)
  @test prox(t, x, 0.7) ≈ b + prox(f, x - b, 0.7)
  check_prox(t, x, 0.7)
  @test [cdprox(t, x[i], i, 0.7) for i in 1:8] ≈ prox(t, x, 0.7)
  # the loss ‖x - b‖²/2 of a translated squared norm
  q = Translate(SqrNormL2(), b)
  @test q(x) ≈ norm(x - b)^2 / 2
  check_gradient(q, x)
  check_prox(q, x, 0.7)
  @test value_and_prox!(similar(x), t, x, 0.7) ≈ t(prox(t, x, 0.7))
  @test_throws DimensionMismatch t(randn(3))
  M = randn(3, 4)
  @test Translate(NuclearNorm(1.), M)(M) ≈ 0
end

@testset "Conjugate" begin
  # Moreau: x = prox_{γf}(x) + γ prox_{f*/γ}(x/γ), for vectors and matrices
  vector_functions = [NormL1(0.6), NormL1(0.6, rand(10)), NormL2(0.6), NormLinf(0.6), IndBallL1(1.5),
                      IndBallL2(1.5), IndBox(-1., 2.), IndNonnegative(), SqrNormL2(2.), SqrNormL2(2., randn(10)),
                      ZeroFunction(), LinearFunction(randn(10), 0.4), GroupNormL2(0.6, [1:4, 5:10]),
                      GroupNormLinf(0.6, [1:4, 5:10]), IndSimplex(1.)]
  matrix_functions = [NormL1(0.6), NormL2(0.6), NormLinf(0.6), IndBallL1(1.5), IndBox(-1., 2.),
                      SqrNormL2(2.), NuclearNorm(0.6), IndBallL2(1.5)]
  for (fs, x) in ((vector_functions, randn(10)), (matrix_functions, randn(4, 5))), f in fs, γ in (0.3, 2.)
    c = Conjugate(f)
    @test prox(f, x, γ) + γ * prox(c, x ./ γ, 1 / γ) ≈ x
    @test prox(c, x, γ) ≈ x - γ * prox(f, x ./ γ, 1 / γ)
  end
  # the values: dual balls and support functions
  y = [0.3, -0.5, 0.1]
  @test Conjugate(NormL1(0.6))(y) == 0 && Conjugate(NormL1(0.4))(y) == Inf
  @test Conjugate(NormL2(0.6))(y) == 0 && Conjugate(NormL2(0.5))(y) == Inf
  @test Conjugate(NormLinf(0.9))(y) == 0 && Conjugate(NormLinf(0.8))(y) == Inf
  @test Conjugate(IndBallL1(2.))(y) ≈ 2 * 0.5
  @test Conjugate(IndBallL2(2.))(y) ≈ 2 * norm(y)
  @test Conjugate(IndBox([-1., -1., 0.], [2., 1., 1.]))(y) ≈ 0.3 * 2 + 0.5 * 1 + 0.1 * 1
  @test Conjugate(IndNonnegative())(-abs.(y)) == 0 && Conjugate(IndNonnegative())(y) == Inf
  @test Conjugate(SqrNormL2(2.))(y) ≈ norm(y)^2 / 4
  @test Conjugate(ZeroFunction())(zeros(3)) == 0 && Conjugate(ZeroFunction())(y) == Inf
  c = randn(3)
  @test Conjugate(LinearFunction(c, 1.))(c) == -1 && Conjugate(LinearFunction(c))(y) == Inf
  # the conjugate of the l1 norm is the indicator of the l∞ ball: the projection
  @test prox(Conjugate(NormL1(0.4)), y) ≈ clamp.(y, -0.4, 0.4)
  for f in (NormL1(0.6), NormL2(0.6), NormLinf(0.6), IndBallL1(1.5), IndBox(-1., 2.), SqrNormL2(2.))
    check_prox(Conjugate(f), 2 .* randn(10), 0.8)
  end
  @test_throws ArgumentError Conjugate(GroupNormL2(0.5, [1:2]))(randn(2))
  # separable conjugates have the scalar proximal operators
  cf = Conjugate(NormL1(0.6))
  @test isseparable(cf)
  @test [cdprox(cf, y[i], i, 0.8) for i in 1:3] ≈ prox(cf, y, 0.8)
end

@testset "SeparableSum" begin
  S, L = randn(5, 5), Matrix(Symmetric(randn(5, 5)))
  f = SeparableSum(NormL1(0.3), TracePSD(0.2))
  @test f((S, L)) == NormL1(0.3)(S) + TracePSD(0.2)(L)
  Z = prox(f, (S, L), 0.7)
  @test Z[1] ≈ prox(NormL1(0.3), S, 0.7) && Z[2] ≈ prox(TracePSD(0.2), L, 0.7)
  out = (similar(S), similar(L))
  @test prox!(out, f, (S, L), 0.7) === out
  @test value_and_prox!(out, f, (S, L), 0.7) ≈ f(Z)
  @test isproximable(f) && isconvex(f) && !issmooth(f)
  # smooth blocks
  X, y = randn(10, 4), randn(10)
  g = SeparableSum(SqrNormL2(2.), LeastSquaresLoss(y, X))
  a, b = randn(3), randn(4)
  G = (similar(a), similar(b))
  @test value_and_gradient!(G, g, (a, b)) ≈ g((a, b))
  @test G[1] ≈ 2a && G[2] ≈ gradient(LeastSquaresLoss(y, X), b)
  @test gradient!(G, g, (a, b)) === G
  @test_throws DimensionMismatch f((S,))
end

@testset "Sum" begin
  X, y = randn(20, 6), randn(20)
  f = Sum(LeastSquaresLoss(y, X), SqrNormL2(0.5), LinearFunction(randn(6)))
  x = randn(6)
  @test f(x) ≈ sum(g -> g(x), f.fs)
  check_gradient(f, x)
  @test issmooth(f) && isconvex(f) && !isproximable(f)
  @test Sum(SqrNormL2())(x) ≈ norm(x)^2 / 2
  g = similar(x)
  value_and_gradient!(g, Sum(SqrNormL2()), x)
  @test g ≈ x
end

end
