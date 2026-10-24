module ConstrainedTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# the projection onto the capped simplex {0 ≤ x ≤ u, Σᵢ xᵢ = k}, by bisection on
# the level τ of clamp.(x .- τ, 0, u)
function capped_simplex_proj(x, k, u = 1.0)
  lo, hi = minimum(x) - u, maximum(x)
  for _ in 1:200
    τ = (lo + hi) / 2
    sum(clamp.(x .- τ, 0, u)) > k ? (lo = τ) : (hi = τ)
  end
  clamp.(x .- (lo + hi) / 2, 0, u)
end

@testset "a separable function on a box" begin
  n = 8
  x = 2 .* randn(n)
  w = rand(n)
  l, u = -rand(n), rand(n)
  fs = (NormL1(0.3), NormL1(0.3, w), ElasticNet(0.2, 0.5), NormL1Nonneg(0.2), SqrNormL2(2.),
        SqrNormL2(1., randn(n)), ZeroFunction(), LinearFunction(randn(n)), Scaled(NormL1(1.), 0.4),
        IndBox(-0.3, 0.6))
  for f in fs, C in (IndBox(-0.5, 1.), IndBox(l, u), IndNonnegative(), IndBallLinf(0.7))
    @testset "$(nameof(typeof(f))) on $(nameof(typeof(C)))" begin
      c = Constrained(f, C)
      check_traits(c; proximable = true, separable = true, convex = true)
      for γ in (0.4, 1.7)
        p = check_prox(c, x, γ)
        @test p ≈ clamp.(prox(f, x, γ), c.box.l, c.box.u)
        @test all(i -> prox_entry(c, x[i], i, γ) ≈ p[i], 1:n)
      end
    end
  end

  # against the minimizer of the scalar problems over a grid of the interval
  c = Constrained(NormL1(0.3, w), IndBox(l, u))
  p = prox(c, x, 0.9)
  for i in 1:n
    t, = brute_force_prox(s -> l[i] <= s <= u[i] ? 0.3 * w[i] * abs(s) : Inf, x[i], 0.9; lo = l[i], hi = u[i])
    @test p[i] ≈ t atol = 1e-6
  end

  c = Constrained(NormL1(0.3), IndBox(-0.5, 1.))
  @test c([0.2, -0.5, 1.]) ≈ 0.3 * 1.7
  @test c([0.2, -0.6, 1.]) == Inf
  @test Constrained(IndBox(0, 2), IndBox(1, 3))([1.5, 1.]) == 0
  @test value_and_prox!(similar(x), Constrained(IndBox(0, 2), IndBox(1, 3)), x, 1.) == 0
  @test prox(Constrained(IndBox(0, 2), IndBox(1, 3)), x) ≈ clamp.(x, 1, 2)

  # a box under a box, and a scaled penalty
  @test prox(Constrained(Constrained(NormL1(0.3), IndBox(-1, 1)), IndBox(0, 2)), x, 0.5) ≈ clamp.(shrink.(x, 0.15), 0, 1)
  @test prox(Constrained(0.5 * NormL1(0.6), IndNonnegative()), x, 0.5) ≈ max.(shrink.(x, 0.15), 0)
end

@testset "coordinate steps on a box" begin
  c = Constrained(NormL1(0.3), IndBox(0, 1))
  v = [1.5, -0.2, 0., 0.4, 0., 2.]
  xs = SparseIterate(sparse(v))
  for k in 1:6
    @test prox_coordinate!(xs, c, k, 0.5) == clamp(shrink(v[k], 0.15), 0, 1)
    @test xs[k] == clamp(shrink(v[k], 0.15), 0, 1)
  end
  S = SymmetricSparseIterate(4)
  S[2, 1] = 0.7
  @test prox_coordinate!(S, c, tril_index(4, 2, 1), 1.) ≈ 0.4
  @test S[1, 2] ≈ 0.4
  xa = AtomIterate([1.5, -0.2, 0.3, 2.], [1:2, 3:4])
  @test prox_coordinate!(xa, c, 2, 1.) ≈ [0., 1.]
  @test isactive(xa, 2)

  @test iszeropreserving(Constrained(NormL1(1.), IndBox(-1, 1)))
  @test iszeropreserving(Constrained(Scaled(NormL1(1.), 2.), IndNonnegative()))
  @test iszeropreserving(Constrained(SqrNormL2(1.), IndNonnegative()))
  @test !iszeropreserving(Constrained(NormL1(1.), IndBox(0.5, 1)))
  @test !iszeropreserving(Constrained(LinearFunction(ones(3)), IndBox(-1, 1)))
  @test !iszeropreserving(Constrained(SqrNormL2(1., ones(3)), IndBox(-1, 1)))
  # on a sparse iterate: the stored entries only, or all of them when zero moves
  xs = SparseIterate(sparse([0., 2., 0., -1.]))
  ys = prox(Constrained(NormL1(0.5), IndBox(-1, 1)), xs, 1.)
  @test ys isa SparseIterate && Array(ys) ≈ [0., 1., 0., -0.5]
  @test Array(prox(Constrained(NormL1(0.5), IndBox(0.2, 1)), xs, 1.)) ≈ [0.2, 1., 0.2, 0.2]
  @test Constrained(NormL1(0.5), IndBox(-1, 1))(ys) ≈ 0.75
  @test Constrained(NormL1(0.5), IndBox(-1, 1))(xs) == Inf             # xs[2] = 2 is outside

  # the direction that improves on zero must be open
  c = Constrained(NormL1(1.), IndBox(0, 2))
  g = [-3., 3., 0.5, -0.5]
  @test kkt_violation(c, g, 1) ≈ 2.
  @test kkt_violation(c, g, 2) <= 0
  @test kkt_violation(c, g, 3) <= 0 && kkt_violation(c, g, 4) <= 0
  @test kkt_violation(c, g, xs, 1) ≈ 2.
  # for a linear term, the direction is that of gᵢ + cᵢ
  @test kkt_violation(Constrained(LinearFunction([-2.]), IndBox(-1, 0)), [0.5], 1) <= 0
  @test kkt_violation(Constrained(LinearFunction([-2.]), IndBox(-1, 1)), [0.5], 1) ≈ 1.5

  w, l, u = rand(8), -rand(8), rand(8)
  c = Constrained(NormL1(0.3, w), IndBox(l, u))
  x = randn(8)
  S = [2, 5, 7]
  @test prox(restrict(c, S), x[S], 0.6) ≈ prox(c, x, 0.6)[S]
end

@testset "a separable function on a hyperplane or a halfspace" begin
  n = 10
  x = randn(n)
  @testset "simplices" begin
    for r in (0.5, 1., 3.), γ in (0.3, 2.)
      Δ = prox(IndSimplex(r), x)
      @test prox(Constrained(IndNonnegative(), IndHyperplane(1, r)), x, γ) ≈ Δ
      @test prox(Constrained(ZeroFunction(), IndSimplex(r)), x, γ) ≈ Δ
      @test prox(Constrained(IndNonnegative(), IndHalfspace(1, r)), x, γ) ≈ prox(IndSimplex(r; equality = false), x)
    end
    for k in (1, 3.5, 7), u in (1., 0.8)
      @test prox(Constrained(IndBox(0, u), IndHyperplane(1, k)), 3 .* x) ≈ capped_simplex_proj(3 .* x, k, u)
    end
  end

  @testset "closed forms" begin
    a, b = randn(n), 0.7
    @test prox(Constrained(ZeroFunction(), IndHyperplane(a, b)), x, 0.5) ≈ prox(IndHyperplane(a, b), x)
    # ridge on a hyperplane: z = (x - γνa) / (1 + γλ)
    λ, γ = 2., 0.6
    ν = (dot(a, x) - b * (1 + γ * λ)) / (γ * dot(a, a))
    c = Constrained(SqrNormL2(λ), IndHyperplane(a, b))
    @test prox(c, x, γ) ≈ (x .- γ * ν .* a) ./ (1 + γ * λ)
    check_prox(c, x, γ)
    # an inactive halfspace leaves the proximal point of f
    @test prox(Constrained(NormL1(0.3), IndHalfspace(1, 100)), x, 0.5) == prox(NormL1(0.3), x, 0.5)
  end

  @testset "the lasso penalty on coefficients that sum to zero" begin
    λ, γ = 0.4, 0.8
    c = Constrained(NormL1(λ), IndHyperplane(1, 0))
    check_traits(c; proximable = true, convex = true)
    z = check_prox(c, x, γ)
    @test abs(sum(z)) <= 1e-12 * norm(x, 1)
    # (x - z)/γ - ν ∈ λ ∂‖z‖₁ for one multiplier ν
    nz = findall(!iszero, z)
    ν = (x[nz[1]] - z[nz[1]]) / γ - λ * sign(z[nz[1]])
    @test all(i -> abs((x[i] - z[i]) / γ - ν - λ * sign(z[i])) <= 1e-10, nz)
    @test all(i -> abs(x[i] / γ - ν) <= λ + 1e-10, findall(iszero, z))
    @test c([1., -1., 0.]) ≈ 0.8
    @test c([1., -0.5, 0.]) == Inf
    # on a matrix, the entries sum to zero
    Z = prox(c, randn(3, 4), γ)
    @test size(Z) == (3, 4) && abs(sum(Z)) <= 1e-12
  end

  @testset "general normals" begin
    a = randn(n)
    a[3] = 0.
    for f in (Constrained(NormL1(0.2), IndBox(-1, 1)), ElasticNet(0.1, 0.3), NormL1Nonneg(0.1, rand(n)))
      for (H, b) in ((IndHyperplane, 0.3), (IndHalfspace, -0.4))
        c = Constrained(f, H(a, b))
        z = check_prox(c, 2 .* x, 0.9)
        H === IndHyperplane ? (@test dot(a, z) ≈ b) : (@test dot(a, z) <= b + 1e-12)
        @test z[3] ≈ prox_entry(f, 2x[3], 3, 0.9)           # aᵢ = 0: independent of ν
      end
    end
    # a scalar normal is an array of it
    for H in (IndHyperplane, IndHalfspace)
      @test prox(Constrained(NormL1(0.3), H(2., 1.5)), x, 0.7) ≈ prox(Constrained(NormL1(0.3), H(fill(2., n), 1.5)), x, 0.7)
    end
    # a box after the linear constraint goes under it
    c1 = Constrained(Constrained(NormL1(0.2), IndHyperplane(1, 1)), IndBox(-1, 1))
    c2 = Constrained(Constrained(NormL1(0.2), IndBox(-1, 1)), IndHyperplane(1, 1))
    @test typeof(c1) == typeof(c2) && prox(c1, x, 0.5) == prox(c2, x, 0.5)
    @test Constrained(Constrained(NormL1(0.2), IndHyperplane(1, 1)), IndNonnegative()) isa typeof(
      Constrained(Constrained(NormL1(0.2), IndNonnegative()), IndHyperplane(1, 1)))
  end

  @testset "errors" begin
    @test_throws ArgumentError prox(Constrained(IndBox(0, 1), IndHyperplane(1, n + 1)), x)
    @test_throws ArgumentError prox(Constrained(IndNonnegative(), IndHalfspace(1, -1)), x)
    @test_throws ArgumentError Constrained(MCP(1.), IndBox(0, 1))
    @test_throws ArgumentError Constrained(NormL0(1.), IndNonnegative())
    @test_throws ArgumentError Constrained(NormL2(1.), IndBox(0, 1))
    @test_throws ArgumentError Constrained(SqrNormL2(1.), IndBallL2(1.))
    @test_throws ArgumentError Constrained(Constrained(NormL1(1.), IndHyperplane(1, 0)), IndHyperplane(1, 1))
    @test_throws DimensionMismatch prox(Constrained(NormL1(1.), IndHyperplane(ones(3), 0)), x)
  end

  @testset "Float32" begin
    x32 = Float32.(x)
    for c in (Constrained(NormL1(0.3), IndBox(0, 1)), Constrained(NormL1(0.3), IndHyperplane(1, 0)),
              Constrained(IndBox(0, 1), IndHyperplane(1, 3)), Constrained(ElasticNet(0.1, 0.2), IndSimplex(2.)))
      z = prox(c, x32, 0.7f0)
      @test eltype(z) == Float32
      @test z ≈ prox(c, x, 0.7) atol = 1e-5
      @test isfinite(c(z))
    end
  end
end

end
