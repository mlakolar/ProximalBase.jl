module  ProximalFunctionTest

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays
import JuMP, Ipopt

Random.seed!(1)

@testset "prox zero" begin
  f = ProxZero()

  p = 10
  x = randn(p)
  z = zeros(p)

  @test prox!(f, z, x) == x
  @test prox(f, x) == z
  @test value(f, x) == 0.

  xs = SparseIterate(x)
  @test cdprox!(f, xs, 3, 1.) == x[3]
  @test Vector(xs) == x

  @test restrict(f, [2, 5]) === f
  grad = [0.5, -2., 0.]
  @test kkt_violation(f, grad, 2) == 2.
  @test kkt_violation(f, grad, 3) == 0.
end

@testset "proximal_l1" begin
  @testset "shrink to zero" begin
    x = randn(10)
    lambda = maximum(abs, x) + 0.1
    g = ProxL1(lambda)

    hat_x = randn(10)
    prox!(g, hat_x, x)
    @test maximum(abs, hat_x) == 0.

    @test value(g, x) ≈ lambda * sum(abs, x)
  end

  @testset "shrink towards zero" begin
    x = [1.0, 1.4, -3.2]
    lambda = 0.1
    g = ProxL1(lambda)

    hat_x = randn(3)
    prox!(g, hat_x, x)
    @test  hat_x ≈ [0.9, 1.3, -3.1]

    @test value(g, x) ≈ lambda * sum(abs, x)
    @test value(g, hat_x) ≈ lambda * sum(abs, hat_x)
  end

  @testset "shrink with vector" begin
    x = [1.0, 1.4, -3.2]
    lambda = [0.1, 1.5, 1]
    g = ProxL1(1., lambda)

    hat_x = randn(3)
    prox!(g, hat_x, x)
    @test hat_x ≈ [0.9, 0., -2.2]
    @test prox(g, x) ≈ [0.9, 0., -2.2]

    @test value(g, x) ≈ dot(lambda, abs.(x))
    @test value(g, hat_x) ≈ dot(lambda, abs.(hat_x))

    # the scaled weights are not materialized
    prox_allocs(g, out, x) = @allocated prox!(g, out, x, 0.5)
    prox_allocs(g, hat_x, x)
    @test prox_allocs(g, hat_x, x) == 0
  end


  @testset "constructors" begin
    @test ProxL1(2., nothing) == ProxL1(2.)
    @test typeof(ProxL1(2., nothing)) == ProxL1{Float64, Nothing}
  end

  @testset "cdprox!" begin
    x = randn(10)
    λ0 = 0.5

    xs = SparseIterate(x)
    g = ProxL1(λ0)
    for k=1:10
      @test cdprox!(g, xs, k, 2.) ≈ shrink(x[k], 2. * λ0)
    end
    @test Vector(xs) ≈ prox(g, x, 2.)

    λ = rand(10)
    xs = SparseIterate(x)
    g = ProxL1(λ0, λ)
    for k=1:10
      @test cdprox!(g, xs, k, 2.) ≈ shrink(x[k], 2. * λ0 * λ[k])
    end
    @test Vector(xs) ≈ prox(g, x, 2.)
    @test_throws DimensionMismatch cdprox!(ProxL1(λ0, rand(5)), xs, 1, 1.)

    # coordinates of SymmetricSparseIterate index the lower triangle
    M = sprandn(4, 4, 0.8)
    M = (M + M') / 2
    xs = SymmetricSparseIterate(M)
    g = ProxL1(λ0)
    for k=1:numCoordinates(xs)
      v = xs[k]
      @test cdprox!(g, xs, k, 1.) ≈ shrink(v, λ0)
    end
    @test Matrix(xs) ≈ prox(g, Matrix(M))
  end

  @testset "restrict" begin
    x = randn(10)
    S = [7, 2, 9]
    z = zeros(10)
    z[S] = x[S]                 # supported on S

    g = ProxL1(0.5)
    @test restrict(g, S) === g

    λ = rand(10)
    g = ProxL1(0.5, λ)
    gS = restrict(g, S)
    @test gS.λ0 == 0.5
    @test gS.λ == λ[S]
    @test value(gS, x[S]) ≈ value(g, z)
    @test prox(gS, x[S], 2.) ≈ prox(g, x, 2.)[S]
  end

  @testset "kkt_violation" begin
    grad = [0.3, -1.2, 0.8]
    g = ProxL1(0.5)
    @test kkt_violation(g, grad, 1) ≈ -0.2
    @test kkt_violation(g, grad, 2) ≈ 0.7

    g = ProxL1(0.5, [1., 3., 1.])
    @test kkt_violation(g, grad, 2) ≈ -0.3
    @test kkt_violation(g, grad, 3) ≈ 0.3

    # a coordinate at zero violates optimality exactly when a proximal gradient
    # step moves it away from zero
    for g in (ProxL1(0.5), ProxL1(0.5, [1., 3., 1.])), i in 1:3
      @test (kkt_violation(g, grad, i) > 0) == !iszero(prox(g, -grad)[i])
    end
  end
end


@testset "proximal_l1_fused" begin

    @testset "random" begin

    m = JuMP.Model(JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

    JuMP.@variable(m, z1)
    JuMP.@variable(m, z2)
    JuMP.@variable(m, t[1:3] >= 0)
    JuMP.@constraint(m, z1 <= t[1] )
    JuMP.@constraint(m, -t[1] <= z1 )
    JuMP.@constraint(m, z2 <= t[2] )
    JuMP.@constraint(m, -t[2] <= z2 )
    JuMP.@constraint(m, z1 - z2 <= t[3])
    JuMP.@constraint(m, -t[3] <= z1 - z2 )

    for i=1:1000
      x1 = randn()
      x2 = randn()
      λ1 = rand()
      λ2 = rand()

      JuMP.@objective(m, Min, ((x1-z1)^2+(x2-z2)^2)/2. + λ1 * (t[1]+t[2]) + λ2 * t[3])
      JuMP.optimize!(m)

      zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, λ1, λ2)

      @test abs(JuMP.value.(z1) - zp1) + abs(JuMP.value.(z2) - zp2)  ≈ 0. atol=2e-4
    end

    m = JuMP.Model(JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    JuMP.@variable(m, z1)
    JuMP.@variable(m, z2)
    JuMP.@variable(m, t >= 0)
    JuMP.@constraint(m, z1 - z2 <= t)
    JuMP.@constraint(m, -t <= z1 - z2 )

    for i=1:1000
      x1 = randn()
      x2 = randn()
      λ = rand()

      JuMP.@objective(m, Min, ((x1-z1)^2+(x2-z2)^2)/2. + λ * t)
      JuMP.optimize!(m)

      zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, 0., λ)

      @test abs(JuMP.value.(z1) - zp1) + abs(JuMP.value.(z2) - zp2)  ≈ 0. atol=1e-4
    end
  end # random testset

  @testset "nonrandom" begin
      x1 = randn()
      x2 = randn()
      λ1 = rand()

      zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, λ1, 0.)
      @test ProximalBase.shrink(x1, λ1) ≈ zp1
      @test ProximalBase.shrink(x2, λ1) ≈ zp2

      λ1 = 1000.
      zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, λ1, 0.)
      @test ProximalBase.shrink(x1, λ1) ≈ 0.
      @test ProximalBase.shrink(x2, λ1) ≈ 0.
  end

  @testset "vector prox" begin
    g = ProxL1Fused(0.3, 0.7)
    x1, x2 = randn(20), randn(20)
    out = (zeros(20), zeros(20))
    prox!(g, out, (x1, x2), 0.5)
    for i=1:20
      @test (out[1][i], out[2][i]) == ProximalBase.proxL1Fused(x1[i], x2[i], 0.3, 0.7, 0.5)
    end

    @test value(g, 1., -2.) ≈ 0.3 * 3. + 0.7 * 3.
  end

  @testset "Float32" begin
    # x and y within 2λ2γ of each other are fused to their average before shrinking
    z = ProximalBase.proxL1Fused(1f0, 1.1f0, 0.1f0, 0.5f0)
    @test z isa Tuple{Float32, Float32}
    @test all(z .≈ 0.95f0)

    x1, x2 = randn(Float32, 20), randn(Float32, 20)
    out = (zeros(Float32, 20), zeros(Float32, 20))
    prox!(ProxL1Fused(0.3f0, 0.7f0), out, (x1, x2), 0.5f0)
    for i=1:20
      @test (out[1][i], out[2][i]) == ProximalBase.proxL1Fused(x1[i], x2[i], 0.3f0, 0.7f0, 0.5f0)
    end
  end
end # proximal_l1


@testset "proximal_l2" begin

  @testset "shrink to zero" begin
    p = 10
    x = randn(p)
    hat_x = randn(p)

    lambda = norm(x) + 0.1
    g = ProxL2(lambda)
    # check the norm
    @test value(g, x) ≈ lambda * norm(x)
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @test value(g, x) ≈ lambda * sqrt(ln)

    # check shrinkage
    prox!(g, hat_x, x, 1.)
    @test hat_x ≈ zeros(Float64, p)
    @test norm(hat_x) ≈ 0.
  end

  @testset "shrink not to zero" begin
    p = 2
    x = [1., 2.]
    hat_x = randn(p)

    lambda = 1.
    # check the norm
    g = ProxL2(lambda)

    @test value(g, x) ≈ lambda * norm(x)
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @test value(g, x) ≈ lambda * sqrt(ln)

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @test hat_x ≈ (1. - 1. / sqrt(5.))*x
    @test norm(hat_x) ≈ sqrt(5.) - 1.
  end

  @testset "constructors" begin
    @test ProxL2(2., nothing) == ProxL2(2.)
    @test typeof(ProxL2(2., nothing)) == ProxL2{Float64, Nothing}
  end

end

@testset "proximal_l2sq" begin

  @testset "prox operator" begin
    x = [1., 2.]
    lambda = 1.

    # norm value
    g = ProxL2Sq(0.)
    @test value(g, x) == 0.

    g = ProxL2Sq(1.)
    @test value(g, x) == sum(abs2, x)

    g = ProxL2Sq(2.)
    @test value(g, x) ≈ 2. * norm(x)^2

    # prox
    hat_x = similar(x)

    g = ProxL2Sq(0.)
    @test prox!(g, hat_x, x) ≈ x

    g = ProxL2Sq(1.)
    @test prox!(g, hat_x, x) ≈ x / 3.

    g = ProxL2Sq(2.)
    @test prox!(g, hat_x, x) ≈ x / 5.


  end

end


@testset "proximal_nuclear" begin

  y = randn(30, 10)
  y = y' * y

  U, S, V = svd(y)
  for k=1:10
    S[k] = max(S[k] - 0.1, 0.)
  end
  tx = U * (Diagonal(S)*V')

  x = randn(10, 10)
  g = ProxNuclear(0.1)
  prox!(g, x, y)

  @test x ≈ tx

  X = randn(5, 3)
  @test value(ProxNuclear(0.5), X) ≈ 0.5 * sum(svdvals(X))
end


@testset "proximal_l1l2" begin

  @testset "shrink to zero" begin
    p = 10
    x = AtomIterate(p, 2)
    hat_x = AtomIterate(p, 2)

    x .= randn(p)
    hat_x .= randn(p)

    lambda = norm(x) + 0.1

    # check the norm
    g = ProxL2(lambda, [1., 1.])
    @test value(g, x) ≈ lambda * norm(x[1:5]) + lambda * norm(x[6:10])

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @test hat_x.storage ≈ zeros(Float64, p)
    @test norm(hat_x) ≈ 0.
  end

  @testset "shrink not to zero" begin
    p = 10
    x = AtomIterate(p, 2)
    hat_x = AtomIterate(p, 2)

    x .= randn(p)
    hat_x .= randn(p)

    lambda = min(norm(x[1:5]), norm(x[6:10])) - 0.2

    # check the norm
    g = ProxL2(lambda, [1., 1.])
    @test value(g, x) ≈ lambda * norm(x[1:5]) + lambda * norm(x[6:10])

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @test hat_x.storage[1:5] ≈  (1. - lambda/norm(x[1:5])) * x[1:5]
    @test hat_x.storage[6:10] ≈ (1. - lambda/norm(x[6:10])) * x[6:10]
  end

end


@testset "proximal_logdet" begin
  p = 10

  out = zeros(p,p)

  # identity
  Σ = Symmetric(Matrix{Float64}(I, p, p))
  V = Matrix{Float64}(I, p, p)
  g = ProxGaussLikelihood(Σ)
  @test prox(g, V) ≈ Matrix{Float64}(I, p, p)
  @test prox!(g, out, V) ≈ Matrix{Float64}(I, p, p)
  @test value(g, V) ≈ p

  γ = 0.5
  ρ = 2.
  sol = Matrix{Float64}(I, p, p) * (-(1. - ρ)+sqrt((1. - ρ)^2 + 4. * ρ)) / (2. * ρ)
  @test prox!(g, out, V, γ) ≈ sol

  # general S: Z = prox satisfies the optimality condition S - Z⁻¹ + (Z - X)/γ = 0
  p = 20
  A = randn(2p, p)
  S = Symmetric(A'A / 2p)
  B = randn(2p, p)
  X = B'B / 2p + I
  g = ProxGaussLikelihood(S)
  Z = prox(g, X, 0.7)
  @test S - inv(Z) + (Z - X) / 0.7 ≈ zeros(p, p) atol = 1e-8
  @test value(g, X) ≈ tr(S * X) - logdet(X)
end

end
