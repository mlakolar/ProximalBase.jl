module SeparableTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

@testset "NormL1" begin
  @testset "shrink" begin
    x = randn(10)
    λ = maximum(abs, x) + 0.1
    @test prox(NormL1(λ), x) == zeros(10)
    @test NormL1(λ)(x) ≈ λ * sum(abs, x)
    x = [1.0, 1.4, -3.2]
    @test prox(NormL1(0.1), x) ≈ [0.9, 1.3, -3.1]
    w = [0.1, 1.5, 1]
    f = NormL1(1., w)
    @test prox(f, x) ≈ [0.9, 0., -2.2]
    @test f(x) ≈ dot(w, abs.(x))
    @test NormL1(2., nothing) == NormL1(2.)
    @test typeof(NormL1(2.)) == NormL1{Float64, Nothing}
    @test_throws ArgumentError NormL1(-1.)
    @test_throws ArgumentError NormL1(1., [-1., 1.])
  end
  for f in (NormL1(0.4), NormL1(0.4, rand(8)))
    x = randn(8)
    check_prox(f, x, 0.7)
    @test value_and_prox!(similar(x), f, x, 0.7) ≈ f(prox(f, x, 0.7))
  end
  @testset "Float32" begin
    x = randn(Float32, 10)
    @test NormL1(0.5)(x) isa Float32
    @test prox(NormL1(0.5), x, 0.3) isa Vector{Float32}
    @test NormL1(0.5, rand(10))(x) isa Float32
    @test prox(NormL1(0.5f0, rand(Float32, 10)), x, 0.3f0) isa Vector{Float32}
  end
end

@testset "sparse iterates" begin
  # the value and the proximal operator touch the stored entries of a SparseIterate,
  # and give the results of the dense array
  for f in (NormL1(0.3), NormL1(0.3, rand(50)), NormL1Nonneg(0.2), ElasticNet(0.3, 0.4), NormL0(0.05),
            MCP(0.3), SCAD(0.3), IndNonnegative(), IndBox(-0.5, 0.5))
    xs = SparseIterate(sprandn(50, 0.2))
    x = Vector(xs)
    @test f(xs) ≈ f(x)
    ys = prox(f, xs, 0.6)
    @test Vector(ys) ≈ prox(f, x, 0.6)
    # in place, the stored entries stay stored
    zs = copy(xs)
    prox!(zs, f, zs, 0.6)
    @test Vector(zs) ≈ prox(f, x, 0.6)
    @test nnz(zs) == nnz(xs)
    @test value_and_prox!(similar(xs), f, xs, 0.6) ≈ f(prox(f, x, 0.6))
    # into another iterate, its stored entries are kept, as for a dense array
    out = SparseIterate(50)
    out[1] = 3.
    prox!(out, f, xs, 0.6)
    @test Vector(out) ≈ prox(f, x, 0.6)
    @test isactive(out, 1)
  end
  # a box that excludes zero visits every entry
  f = IndBox(0.5, 1.)
  xs = SparseIterate(sprandn(20, 0.2))
  @test Vector(prox(f, xs, 1.)) == clamp.(Vector(xs), 0.5, 1.)
  @test f(xs) == Inf
  # on a SymmetricSparseIterate: the dense symmetric matrix, each coordinate once
  p = 6
  D = Matrix(Symmetric(sprandn(p, p, 0.4))) + Diagonal(randn(p))
  W = rand(p, p); W = W + W'
  for f in (NormL1(0.2), NormL1(0.2, W), MCP(0.2), IndBox(-0.3, 0.3), SqrNormL2(2.), IndBox(0.1, 1.))
    s = SymmetricSparseIterate(D)
    @test f(s) ≈ f(D)
    @test Matrix(prox(f, s, 0.5)) ≈ prox(f, D, 0.5)
    prox!(s, f, s, 0.5)                                    # in place
    @test Matrix(s) ≈ prox(f, D, 0.5)
  end
  # on an AtomIterate: the active groups for a function that preserves zeros
  xa = AtomIterate([0., 0., 1., -2., 0., 0.], [1:2, 3:4, 5:6])
  @test NormL1(0.5)(xa) ≈ 1.5
  @test MCP(1.)(xa) ≈ MCP(1.)(Vector(xa))
end

@testset "parameter arrays must fit" begin
  # weights and bounds of the wrong length raise an error instead of reading out of bounds
  for f in (NormL1(1., [1., 2.]), NormL1Nonneg(1., [1., 2.]), ElasticNet(1., 1., [1., 2.]), IndBox([-1., -1.], 1.),
            IndBox(-1., [1., 1.]), LinearFunction([1., 2.]), SqrNormL2(1., [1., 2.]), Tilt(NormL1(1.), [1., 2.]),
            Translate(NormL1(1.), [1., 2.]), 2 * NormL1(1., [1., 2.]))
    x = ones(6)
    @test_throws DimensionMismatch prox(f, x, 0.1)
    @test_throws DimensionMismatch cdprox!(SparseIterate(x), f, 3, 0.1)
    @test_throws DimensionMismatch cdprox!(AtomIterate(x, 3), f, 2, 0.1)
    f isa Union{LinearFunction, SqrNormL2, Tilt, Translate} || @test_throws DimensionMismatch f(x)
  end
  # a direct call of the scalar operator checks the index
  @test_throws BoundsError cdprox(NormL1(1., [1., 2.]), 1.0, 5, 0.1)
  # on a SymmetricSparseIterate the weights follow the p × p matrix: a vector over the
  # lower triangle, the convention of 0.4, is rejected
  p = 4
  S = SymmetricSparseIterate(Matrix(1.0I, p, p))
  wtri = rand(p * (p + 1) ÷ 2)
  @test_throws DimensionMismatch cdprox!(S, NormL1(0.1, wtri), ncoordinates(S), 0.5)
  @test_throws DimensionMismatch prox(NormL1(0.1, wtri), S, 0.5)
  @test_throws DimensionMismatch NormL1(0.1, wtri)(S)
  @test cdprox!(S, NormL1(0.1, ones(p, p)), ncoordinates(S), 0.5) ≈ 1 - 0.05
end

@testset "SymmetricSparseIterate into another iterate" begin
  p = 5
  D = Matrix(Symmetric(sprandn(p, p, 0.4))) + I
  s = SymmetricSparseIterate(D)
  y = SymmetricSparseIterate(p)
  y[p, 1] = 3.                     # stored in y only: becomes a stored zero
  prox!(y, NormL1(0.2), s, 1.)
  @test Matrix(y) ≈ prox(NormL1(0.2), D, 1.)
  @test isactive(y, tril_index(p, p, 1)) && y[p, 1] == 0
end

@testset "NormL1Nonneg" begin
  x = [1.0, -0.4, 2.5, 0.1]
  f = NormL1Nonneg(0.5)
  @test prox(f, x) ≈ [0.5, 0., 2., 0.]
  @test f([1., 2.]) == 1.5 && f([1., -1.]) == Inf
  w = rand(4)
  check_prox(NormL1Nonneg(0.3, w), randn(4), 0.9)
end

@testset "ElasticNet" begin
  x = randn(8)
  f = ElasticNet(0.3, 0.5)
  @test f(x) ≈ 0.3 * sum(abs, x) + 0.25 * sum(abs2, x)
  @test prox(f, x, 0.7) ≈ shrink.(x, 0.21) ./ 1.35
  check_prox(f, x, 0.7)
  check_prox(ElasticNet(0.3, 0.5, rand(8)), x, 0.7)
  # the limits: the lasso and ridge
  @test prox(ElasticNet(0.3, 0.), x, 0.7) ≈ prox(NormL1(0.3), x, 0.7)
  @test prox(ElasticNet(0., 0.5), x, 0.7) ≈ prox(SqrNormL2(0.5), x, 0.7)
end

@testset "boxes" begin
  x = 3 .* randn(10)
  l, u = -rand(10), rand(10)
  for f in (IndBox(-1., 2.), IndBox(l, u), IndBallLinf(0.8), IndNonnegative(), IndBox(-Inf, 0.5))
    z = check_prox(f, x, 1.)
    @test f(z) == 0
    @test value_and_prox!(similar(x), f, x, 1.) == 0
  end
  @test prox(IndBox(l, u), x) == clamp.(x, l, u)
  @test prox(IndNonnegative(), x) == max.(x, 0)
  @test IndBallLinf(0.8) isa IndBox
  @test IndBox(0, 1)([0.5, 2.]) == Inf
  @test IndBox(0, 1)([1 + 1e-12, 0.]) == 0                # the tolerance of rounding
  @test IndNonnegative()([-1e-12, 1.]) == 0 && IndNonnegative()([-1e-3, 1.]) == Inf
  @test_throws ArgumentError IndBox(1, 0)
  @test_throws ArgumentError IndBallLinf(-1.)
end

@testset "NormL0" begin
  x = [0.5, -2., 0.05, 1.2]
  f = NormL0(0.5)
  @test f(x) == 2.
  @test prox(f, x, 1.) == [0., -2., 0., 1.2]       # kept when x² > 2γλ = 1
  for v in (-3., -1.1, -0.9, 0.3, 1.0, 2.5), γ in (0.3, 1., 4.)
    t, _ = brute_force_prox(u -> 0.5 * (u != 0), v, γ)
    @test cdprox(f, v, 1, γ) ≈ t atol = 1e-6
  end
  check_prox(f, randn(10), 0.8)
end

@testset "MCP" begin
  λ, a = 0.6, 2.5
  f = MCP(λ, a)
  pen(t) = abs(t) <= a * λ ? λ * abs(t) - t^2 / (2a) : a * λ^2 / 2
  x = 3 .* randn(20)
  @test f(x) ≈ sum(pen, x)
  # the proximal operator is the global minimizer, also for steps γ ≥ a that make the
  # problem nonconvex
  for v in (-3., -1.5, -0.7, -0.2, 0., 0.35, 0.65, 1.0, 1.5, 2.2, 4.), γ in (0.3, 1., 2.4, 2.5, 3., 10.)
    t, ht = brute_force_prox(pen, v, γ)
    z = cdprox(f, v, 1, γ)
    @test pen(z) + (z - v)^2 / (2γ) <= ht + 1e-9
  end
  # firm thresholding for γ < a
  @test cdprox(f, 0.5, 1, 1.) == 0.
  @test cdprox(f, 1.0, 1, 1.) ≈ (1.0 - 0.6) / (1 - 1 / 2.5)
  @test cdprox(f, 2.0, 1, 1.) == 2.0
  check_prox(f, x, 0.8)
  @test_throws ArgumentError MCP(0.5, 0.)
end

@testset "SCAD" begin
  λ, a = 0.6, 3.7
  f = SCAD(λ, a)
  function pen(t)
    s = abs(t)
    s <= λ ? λ * s : s <= a * λ ? (2a * λ * s - s^2 - λ^2) / (2 * (a - 1)) : λ^2 * (a + 1) / 2
  end
  x = 3 .* randn(20)
  @test f(x) ≈ sum(pen, x)
  for v in (-4., -2., -1.1, -0.7, 0., 0.3, 0.9, 1.3, 2.1, 3., 5.), γ in (0.3, 1., 2.6, 2.7, 3., 10.)
    t, ht = brute_force_prox(pen, v, γ)
    z = cdprox(f, v, 1, γ)
    @test pen(z) + (z - v)^2 / (2γ) <= ht + 1e-9
  end
  # the SCAD thresholding rule for γ = 1: soft thresholding up to 2λ
  @test cdprox(f, 1.0, 1, 1.) ≈ 0.4
  @test cdprox(f, 1.5, 1, 1.) ≈ ((a - 1) * 1.5 - a * λ) / (a - 2)
  @test cdprox(f, 3.0, 1, 1.) == 3.0
  check_prox(f, x, 0.8)
  @test_throws ArgumentError SCAD(0.5, 2.)
end

end
