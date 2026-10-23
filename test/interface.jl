module InterfaceTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# a function defined outside ProximalBase, with only a value and prox!
struct Box2 <: AbstractFunction end
ProximalBase.isproximable(::Type{Box2}) = true
(::Box2)(x) = all(abs.(x) .<= 2) ? 0. : Inf
ProximalBase.prox!(y::AbstractArray, ::Box2, x::AbstractArray, ::Real) = (y .= clamp.(x, -2, 2))

@testset "traits" begin
  X, y = randn(10, 4), randn(10)
  S = Matrix(Symmetric(randn(3, 3))) + 4I
  #                                     smooth proximable separable convex
  table = [
    (ZeroFunction(),                     true,  true,  true,  true),
    (LinearFunction(randn(4)),           true,  true,  true,  true),
    (SqrNormL2(2.),                      true,  true,  true,  true),
    (QuadraticFunction(X'X, randn(4)),   true,  true,  false, true),
    (LeastSquaresLoss(y, X),             true,  true,  false, true),
    (LogisticLoss(rand(0:1, 10), X),     true,  false, false, true),
    (PoissonLoss(rand(0:3, 10), X),      true,  false, false, true),
    (HuberLoss(y, X, 1.),                true,  false, false, true),
    (LogDetLoss(S),                      true,  true,  false, true),
    (NormL1(0.5),                        false, true,  true,  true),
    (NormL1Nonneg(0.5),                  false, true,  true,  true),
    (ElasticNet(0.5, 0.1),               false, true,  true,  true),
    (IndBox(-1, 1),                      false, true,  true,  true),
    (IndNonnegative(),                   false, true,  true,  true),
    (NormL0(0.5),                        false, true,  true,  false),
    (MCP(0.5),                           false, true,  true,  false),
    (SCAD(0.5),                          false, true,  true,  false),
    (NormL2(0.5),                        false, true,  false, true),
    (NormLinf(0.5),                      false, true,  false, true),
    (GroupNormL2(0.5, [1:2, 3:4]),       false, true,  false, true),
    (GroupNormLinf(0.5, [1:2, 3:4]),     false, true,  false, true),
    (NormL1Fused(0.5, 0.2),              false, true,  false, true),
    (NuclearNorm(0.5),                   false, true,  false, true),
    (TracePSD(0.5),                      false, true,  false, true),
    (IndPSD(),                           false, true,  false, true),
    (IndPSD(rank = 2),                   false, true,  false, false),
    (IndBallRank(2),                     false, true,  false, false),
    (IndBallL0(2),                       false, true,  false, false),
    (IndBallL1(1.),                      false, true,  false, true),
    (IndBallL2(1.),                      false, true,  false, true),
    (IndSimplex(1.),                     false, true,  false, true),
    (IndHyperplane(ones(3), 1.),         false, true,  false, true),
    (IndHalfspace(ones(3), 1.),          false, true,  false, true),
  ]
  for (f, smooth, proximable, separable, convex) in table
    @testset "$(nameof(typeof(f)))" begin
      check_traits(f; smooth, proximable, separable, convex)
      @test f isa AbstractFunction
    end
  end
  # the calculus passes the traits on
  check_traits(2 * NormL1(1.); proximable = true, separable = true, convex = true)
  check_traits(Tilt(SqrNormL2(), ones(3)); smooth = true, proximable = true, separable = true, convex = true)
  check_traits(Translate(MCP(1.), ones(3)); proximable = true, separable = true)
  check_traits(Conjugate(NormL2(1.)); proximable = true, convex = true)
  check_traits(SeparableSum(NormL1(1.), SqrNormL2()); proximable = true, convex = true)
  check_traits(Sum(SqrNormL2(), LeastSquaresLoss(y, X)); smooth = true, convex = true)
  check_traits(Sum(SqrNormL2(), MCP(1.)))
end

@testset "generic interface" begin
  x = randn(6)
  f = NormL1(0.5)
  @test f(x) == 0.5 * sum(abs, x)
  @test prox(f, x) == prox(f, x, 1) == prox(f, x, 1.0)
  @test prox(f, x, 2) == prox(f, x, 2.0)                  # any real step
  @test prox(f, Float32.(x), 0.5) isa Vector{Float32}      # computed in the element type of x
  @test prox(f, Float32.(x), 0.5) ≈ prox(f, x, 0.5)
  y = similar(x)
  @test value_and_prox!(y, f, x) ≈ f(prox(f, x))
  # prox of a sparse iterate is a sparse iterate, gradients are dense
  xs = SparseIterate(sprandn(8, 0.4))
  @test prox(f, xs, 0.3) isa SparseIterate{Float64, 1}
  @test Vector(prox(f, xs, 0.3)) == prox(f, Vector(xs), 0.3)
  g = gradient(SqrNormL2(), xs)
  @test g isa Vector{Float64} && g == Vector(xs)
  v, g = value_and_gradient(SqrNormL2(), xs)
  @test v ≈ sum(abs2, xs) / 2
  # a generic function: value_and_prox! falls back to prox! then the value
  @test value_and_prox!(similar(x), Box2(), 3 .* x) == 0.
  @test prox(Box2(), [3., -4., 1.]) == [2., -2., 1.]
end

@testset "prox_entry and prox_coordinate!" begin
  γ = 0.7
  separable = [ZeroFunction(), LinearFunction(randn(10)), SqrNormL2(2.), SqrNormL2(1., randn(10)),
               NormL1(0.3), NormL1(0.3, rand(10)), NormL1Nonneg(0.4, rand(10)), ElasticNet(0.3, 0.5),
               IndBox(-0.5, 0.8), IndBox(-rand(10), rand(10)), IndNonnegative(), NormL0(0.2),
               MCP(0.4, 2.5), SCAD(0.4, 3.7), 2 * NormL1(0.3), Tilt(NormL1(0.3), randn(10)),
               Translate(NormL1(0.3), randn(10)), Conjugate(NormL1(0.3))]
  for f in separable
    @testset "$(nameof(typeof(f)))" begin
      x = randn(10)
      z = prox(f, x, γ)
      # the scalar operators are the entries of the proximal operator
      @test [prox_entry(f, x[i], i, γ) for i in 1:10] ≈ z
      # on a SparseIterate, coordinate k is entry k
      xs = SparseIterate(x)
      for k in 1:10
        @test prox_coordinate!(xs, f, k, γ) ≈ z[k]
      end
      @test Vector(xs) ≈ z
      # on an AtomIterate, coordinate k is group k, entry by entry
      xa = AtomIterate(x, [1:3, 4:4, [5, 9], [6, 7, 8, 10]])
      for k in 1:4
        @test prox_coordinate!(xa, f, k, γ) ≈ z[xa.groups[k]]
      end
      @test Vector(xa) ≈ z
    end
  end

  # a coordinate that stays zero is not stored, one that becomes nonzero is
  xs = SparseIterate(5)
  prox_coordinate!(xs, NormL1(1.), 2, 1.)
  @test nnz(xs) == 0
  prox_coordinate!(xs, IndBox(1, 2), 2, 1.)
  @test nnz(xs) == 1 && xs[2] == 1.
  xa = AtomIterate(4, [1:2, 3:4])
  prox_coordinate!(xa, NormL1(1.), 1, 1.)
  @test nactive(xa) == 0
  prox_coordinate!(xa, IndBox(1, 2), 2, 1.)
  @test activecoordinates(xa) == [2] && Vector(xa) == [0., 0., 1., 1.]

  # on a SymmetricSparseIterate, coordinate k is entry (r, c) of the lower triangle, and
  # the per-entry parameters are those of the dense matrix
  p = 5
  D = randn(p, p); D = D + D'
  W = rand(p, p); W = W + W'
  f = NormL1(0.3, W)
  x = SymmetricSparseIterate(D)
  for k in 1:ncoordinates(x)
    r, c = tril_cartesian(p, k)
    @test prox_coordinate!(x, f, k, γ) ≈ shrink(D[r, c], γ * 0.3 * W[r, c])
  end
  @test Matrix(x) ≈ prox(f, D, γ)

  # functions that are not separable
  @test_throws ArgumentError prox_coordinate!(SparseIterate(3), NormL2(1.), 1, 1.)
  @test_throws ArgumentError prox_coordinate!(SymmetricSparseIterate(3), NuclearNorm(1.), 1, 1.)
  @test_throws ArgumentError prox_coordinate!(AtomIterate(4, 2), IndBallL1(1.), 1, 1.)
end

@testset "kkt_violation" begin
  grad = [0.3, -1.2, 0.8, 0.1, -0.5]
  xk = [0., 1., 0., 0., 0.]
  for f in (NormL1(0.5), NormL1(0.5, rand(5)), ZeroFunction(), ElasticNet(0.4, 1.), MCP(0.6), SCAD(0.6),
            NormL1Nonneg(0.2), IndNonnegative(), IndBox(-1., 1.), SqrNormL2(), LinearFunction(randn(5))),
      i in (1, 3, 5)
    # for the separable penalties, the rest of x does not matter
    @test kkt_violation(f, grad, xk, i) == kkt_violation(f, grad, i)
    # at x = 0, a coordinate violates optimality exactly when a proximal gradient step
    # with a small step size moves it away from zero
    if !issmooth(f)
      step = prox(f, -1e-3 .* grad, 1e-3)
      @test (kkt_violation(f, grad, i) > 0) == !iszero(step[i])
    end
  end
  @test kkt_violation(NormL1(0.5), grad, 2) ≈ 0.7
  @test kkt_violation(NormL1Nonneg(0.2), grad, 2) ≈ 1.0      # moving up decreases the objective
  @test kkt_violation(NormL1Nonneg(0.2), grad, 1) ≈ -0.5
  # on a SymmetricSparseIterate, coordinate k is entry (r, c) of the gradient
  p = 3
  G = randn(p, p); G = G + G'
  xs = SymmetricSparseIterate(p)
  for k in 1:ncoordinates(xs)
    r, c = tril_cartesian(p, k)
    @test kkt_violation(NormL1(0.2), G, xs, k) ≈ abs(G[r, c]) - 0.2
  end
end

@testset "restrict" begin
  x = randn(10)
  S = [7, 2, 9]
  z = zeros(10)
  z[S] = x[S]                 # supported on S
  w = rand(10)
  for f in (NormL1(0.5), NormL1(0.5, w), ZeroFunction(), NormL1Nonneg(0.3, w), ElasticNet(0.2, 0.4, w),
            IndBox(-w, w), MCP(0.3), SCAD(0.3), NormL0(0.1), 2 * NormL1(0.5, w))
    fS = restrict(f, S)
    @test fS(x[S]) ≈ f(z)
    @test prox(fS, x[S], 0.7) ≈ prox(f, z, 0.7)[S]
  end
  X, y = randn(20, 10), randn(20)
  for f in (SqrNormL2(2.), QuadraticFunction(X'X, randn(10), 1.), LeastSquaresLoss(y, X),
            LogisticLoss(rand(0:1, 20), X), PoissonLoss(rand(0:2, 20), X), HuberLoss(y, X, 0.5),
            LinearFunction(randn(10), 2.))
    fS = restrict(f, S)
    @test fS(z[S]) ≈ f(z)
    @test gradient(fS, z[S]) ≈ gradient(f, z)[S]
  end
  # SqrNormL2 with a center drops the constant of the center outside S
  c = randn(10)
  fS = restrict(SqrNormL2(2., c), S)
  @test fS(z[S]) ≈ SqrNormL2(2., c)(z) - sum(abs2, c[setdiff(1:10, S)])
end

end
