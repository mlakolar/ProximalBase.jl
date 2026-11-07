module ConstrainedBasicsTest

# Constrained on small inputs: a separable function on a box and on a hyperplane or a
# halfspace, against closed forms and the optimality conditions, coordinate steps and errors.
# The function x constraint grids, the multiplier searches and the stress cases are in constrained.jl.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PB = ProximalBase

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
  fs = (NormL1(0.3), SqrNormL2(1., randn(n)))
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

@testset "zero-preserving wrappers on a box" begin
  fs = (Scaled(NormL1(1.), 2.),)
  for f in fs, xs in (SparseIterate([0., 3., 0., -2., 0.]), SparseIterate(5))
    c = Constrained(f, IndBox(-1., 1.))
    @test iszeropreserving(c)
    x = Vector(xs)
    @test c(xs) == c(x)
    for step in (0.7, fill(0.7, 5))
      expected = prox(c, x, step)
      out = SparseIterate([2., 0., 0., 0., 0.])
      @test prox!(out, c, xs, step) === out
      @test Vector(out) ≈ expected
      @test value_and_prox!(out, c, xs, step) ≈ c(expected)
      @test Vector(out) ≈ expected
      @test c(out) ≈ c(expected)
      @test isactive(out, 1) && out[1] == 0
      alias = copy(xs)
      @test prox!(alias, c, alias, step) === alias
      @test Vector(alias) ≈ expected
      alias = copy(xs)
      @test value_and_prox!(alias, c, alias, step) ≈ c(expected)
      @test Vector(alias) ≈ expected
      @test nnz(alias) == nnz(xs)
    end
  end
  @test !iszeropreserving(Constrained(Scaled(NormL1(1.), Inf), IndBox(-1., 1.)))
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
    for f in (Constrained(NormL1(0.2), IndBox(-1, 1)),)
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
    @test_throws ArgumentError Constrained(MCP(1.), IndHyperplane(1, 0))
    @test_throws ArgumentError Constrained(NormL0(1.), IndSimplex(1.))
    @test_throws ArgumentError Constrained(NormL2(1.), IndBox(0, 1))
    @test_throws ArgumentError Constrained(SqrNormL2(1.), IndBallL2(1.))
    @test_throws ArgumentError Constrained(Constrained(NormL1(1.), IndHyperplane(1, 0)), IndHyperplane(1, 1))
    @test_throws DimensionMismatch prox(Constrained(NormL1(1.), IndHyperplane(ones(3), 0)), x)
  end

  @testset "Float32" begin
    x32 = Float32.(x)
    for c in (Constrained(NormL1(0.3), IndBox(0, 1)), Constrained(NormL1(0.3), IndHyperplane(1, 0)))
      z = prox(c, x32, 0.7f0)
      @test eltype(z) == Float32
      @test z ≈ prox(c, x, 0.7) atol = 1e-5
      @test isfinite(c(z))
    end
  end
end

@testset "boxes intersecting closed and open scalar domains" begin
  for T in (Float64,)
    # Entropy has a closed domain at zero; the positive log barrier can instead
    # be constrained to an interval contained in its open domain.
    @test prox(Constrained(NegEntropy(one(T)), IndBox(zero(T), zero(T))), T[1]) == T[0]
    c = Constrained(NegLog(one(T)), IndBox(one(T), T(2)))
    x, out = T[-3, 0, 5], zeros(T, 3)
    expected = clamp.((x .+ sqrt.(x.^2 .+ 4)) ./ 2, one(T), T(2))
    @test prox(c, x) ≈ expected
    @test value_and_prox!(out, c, x, one(T)) ≈ -sum(log, expected)
    @test out ≈ expected
    @test all(>(zero(T)), out)
  end
end

# The value of the conjugate of ElasticNet(λ1, λ2, w) at y, from the explicit formula in BigFloat: for
# λ2 > 0, Σᵢ max(|yᵢ| - λ1 wᵢ, 0)² / (2λ2); for λ2 = 0, the indicator of |yᵢ| ≤ λ1 wᵢ, within the
# relative tolerance sqrt(eps) that the package gives to the values of indicators.
function elastic_conjugate_reference(λ1, λ2, w, y)
  S = BigFloat
  level(i) = S(λ1) * (w === nothing ? one(S) : S(w[i]))
  if λ2 > 0
    sum((max(abs(S(y[i])) - level(i), 0)^2 for i in eachindex(y)); init = zero(S)) / (2 * S(λ2))
  else
    tol = sqrt(S(eps(float(eltype(y)))))
    all(i -> abs(S(y[i])) - level(i) <= tol * max(1, level(i)), eachindex(y)) ? zero(S) : S(Inf)
  end
end

function check_elastic_value(v, T, ref)
  @test v isa T
  if isinf(ref)
    @test v == T(Inf)
  elseif iszero(ref)
    @test v == 0
  else
    @test isapprox(v, T(ref); rtol = 1000 * eps(T))
  end
end

a_value_and_prox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)

@testset "value_and_prox! of the conjugate of an elastic net on a box" begin
  f = Conjugate(ElasticNet(1.0, 0.5))
  x = [5.0, -5.0, 0.5]

  @testset "the value is taken at the stored output" begin
    # clamped at ±2: the value at [2, -2, 0.5] is (1² + 1² + 0) / (2 0.5) = 2, while the
    # unclamped proximal point [3, -3, 0.5] has the value 8
    c = Constrained(f, IndBox(-2.0, 2.0))
    y = zeros(3)
    @test value_and_prox!(y, c, x, 0.5) === 2.0
    @test y == [2.0, -2.0, 0.5]
    @test y == prox(c, x, 0.5)
    @test value_and_prox!(zeros(3), Constrained(f, IndBox(-10.0, 10.0)), x, 0.5) === 8.0
    # the original box: the stored output is inside the domain, where the value is zero
    y = zeros(3)
    @test value_and_prox!(y, Constrained(f, IndBox(-1.0, 1.0)), [1.0, 2.0, -0.5], 0.5) === 0.0
    @test y == [1.0, 1.0, -0.5]
    z = copy(x)
    @test value_and_prox!(z, c, z, 0.5) === 2.0
    @test z == [2.0, -2.0, 0.5]
    @test value_and_prox!(zeros(3), c, x, [0.5, 0.5, 0.5]) === 2.0
    @test value_and_prox!(zeros(3), Constrained(f, IndNonnegative()), x, 0.5) === 4.0
    @test value_and_prox!(Float64[], c, Float64[], 0.5) === 0.0
    empty = Constrained(Conjugate(ElasticNet(1.0, 0.5, Float64[])), IndBox(-2.0, 2.0))
    @test value_and_prox!(Float64[], empty, Float64[], 0.5) === 0.0
  end

  @testset "standalone values stay unavailable" begin
    c = Constrained(f, IndBox(-2.0, 2.0))
    @test_throws ArgumentError f([1.0, 1.0, 0.0])
    @test_throws ArgumentError c([1.0, 1.0, 0.0])
    @test !PB._has_value(c)
    @test !PB._has_value(f)
    @test !PB._has_conjugate_value(ElasticNet(1.0, 0.5))
  end

  @testset "against the explicit formula" begin
    onbound = Set{Int}()
    for T in (Float32, Float64)
      xin = T[5, -5, 0.5, 1.2, -0.05, 0, 3]
      n = length(xin)
      weights = (nothing, T[1, 0, 2, 0.5, 3, 1, 0])
      lo, hi = -T[2, 1, 0.5, 3, 0.1, 1, 2], T[2, 1, 0.5, 3, 0.2, 1, 2]
      boxes = ((IndBox(T(-10), T(10)), fill(T(-10), n), fill(T(10), n)),
               (IndBox(T(-1), T(1)), fill(T(-1), n), fill(T(1), n)),
               (IndBox(lo, hi), lo, hi),
               (IndNonnegative(), zeros(T, n), fill(T(Inf), n)),
               (IndBox(T(0.25), T(0.25)), fill(T(0.25), n), fill(T(0.25), n)))
      steps = (T(0.5), T[0.5, 0.25, 1, 2, 0.75, 0.5, 1])
      # λ1 and the weights give exact levels, so that the reference does not depend on a rounded level
      for w in weights, (λ1, λ2) in ((T(0.25), T(0.7)), (T(0), T(0.7)), (T(0.25), T(0)), (T(0), T(0)), (T(0.25), T(1e-6))),
          (box, l, u) in boxes, γ in steps, alias in (false, true)
        c = Constrained(Conjugate(ElasticNet(λ1, λ2, w)), box)
        expected = prox!(similar(xin), c, xin, γ)
        y = alias ? copy(xin) : similar(xin)
        v = a_value_and_prox!(y, c, alias ? y : xin, γ)
        @test y == expected
        alias || @test xin == T[5, -5, 0.5, 1.2, -0.05, 0, 3]
        check_elastic_value(v, T, elastic_conjugate_reference(λ1, λ2, w, y))
        λ2 > 0 && push!(onbound, count(i -> y[i] == l[i] || y[i] == u[i], 1:n))
      end
    end
    # no entry, some entries and every entry sit on a bound
    @test 0 in onbound && 7 in onbound && any(k -> 0 < k < 7, onbound)
  end

  @testset "levels, bounds and the domain of the indicator" begin
    for T in (Float32, Float64)
      w = T[1, 2, 4]
      level = T(0.5) .* w                                       # 0.5, 1, 2 exactly
      big = fill(T(1e3), 3)
      # the stored output is the bound: exactly at the level and one float on each side
      for (y0, λ2) in ((level, T(0.5)),
                       (T[level[1], nextfloat(level[2]), prevfloat(level[3])], T(0.5)),
                       (T[prevfloat(level[1]), level[2], nextfloat(level[3])], T(1)),
                       (-T[level[1], nextfloat(level[2]), prevfloat(level[3])], T(2)))
        c = Constrained(Conjugate(ElasticNet(T(0.5), λ2, w)), IndBox(y0, y0))
        y = zeros(T, 3)
        v = value_and_prox!(y, c, big, T(1))
        @test y == y0
        check_elastic_value(v, T, elastic_conjugate_reference(T(0.5), λ2, w, y))
      end
      # λ2 = 0: the indicator of |yᵢ| ≤ λ1 wᵢ, with the tolerance of the NormL1 conjugate
      indicator = Conjugate(NormL1(T(0.5), w))
      for (y0, expected) in ((level, T(0)),
                             (T[level[1], nextfloat(level[2]), prevfloat(level[3])], T(0)),
                             (-nextfloat.(level), T(0)),
                             (level .* T(1 + 1e-3), T(Inf)),
                             (T[level[1], level[2], 3], T(Inf)),
                             (T[0, 0, 0], T(0)))
        c = Constrained(Conjugate(ElasticNet(T(0.5), T(0), w)), IndBox(y0, y0))
        y = zeros(T, 3)
        v = value_and_prox!(y, c, big, T(1))
        @test y == y0
        @test v === expected
        @test v === indicator(y)
      end
      # the box misses the domain of the indicator: the output is clamped there, and its value is +Inf;
      # with λ2 > 0 the value at the clamped output is finite
      c = Constrained(Conjugate(ElasticNet(T(0.5), T(0), w)), IndBox(T(5), T(6)))
      y = zeros(T, 3)
      @test value_and_prox!(y, c, T[0, 1, -1], T(1)) === T(Inf)
      @test y == fill(T(5), 3)
      @test y == prox(c, T[0, 1, -1], T(1))
      c = Constrained(Conjugate(ElasticNet(T(0.5), T(0.5), w)), IndBox(T(5), T(6)))
      v = value_and_prox!(y, c, T[0, 1, -1], T(1))
      @test y == fill(T(5), 3)
      @test v ≈ 45.25
    end
  end

  @testset "dense, sparse and symmetric arguments" begin
    c = Constrained(Conjugate(ElasticNet(0.25, 0.7, [1.0, 2.0, 0.0, 0.5, 3.0])), IndBox(-1.0, 1.0))
    xs = SparseIterate(sparse([0.0, 3.0, 0.0, -2.0, 0.0]))
    for γ in (0.7, [0.7, 0.5, 1.0, 0.7, 0.9])
      yd = zeros(5)
      vd = value_and_prox!(yd, c, Vector(xs), γ)
      out = SparseIterate(5)
      v = value_and_prox!(out, c, xs, γ)
      @test v == vd && Vector(out) == yd
      @test isapprox(v, Float64(elastic_conjugate_reference(0.25, 0.7, [1.0, 2.0, 0.0, 0.5, 3.0], yd)); rtol = 1e-13)
      alias = copy(xs)
      @test value_and_prox!(alias, c, alias, γ) == vd
      @test Vector(alias) == yd
    end
    # a zero stays inside the box, and a box that excludes it moves every entry
    for box in (IndNonnegative(), IndBox(0.2, 1.0)), f in (ElasticNet(0.25, 0.7), ElasticNet(0.25, 0.0))
      c = Constrained(Conjugate(f), box)
      yd = zeros(5)
      vd = value_and_prox!(yd, c, Vector(xs), 0.7)
      out = SparseIterate(5)
      @test value_and_prox!(out, c, xs, 0.7) == vd
      @test Vector(out) == yd
      @test isapprox(vd, Float64(elastic_conjugate_reference(0.25, f.λ2, nothing, yd)); rtol = 1e-13)
    end
    # the symmetric matrix: every entry, off-diagonals included
    M = [2.0 0.5 -3.0; 0.5 -1.0 4.0; -3.0 4.0 0.1]
    W = [1.0 2.0 0.0; 2.0 0.5 1.0; 0.0 1.0 3.0]
    for w in (nothing, W), (λ1, λ2) in ((0.25, 0.7), (0.25, 0.0)), γ in (0.7, fill(0.7, 3, 3))
      c = Constrained(Conjugate(ElasticNet(λ1, λ2, w)), IndBox(-1.0, 1.0))
      Yd = zeros(3, 3)
      vd = value_and_prox!(Yd, c, M, γ)
      S = SymmetricSparseIterate(3)
      v = value_and_prox!(S, c, M, γ)
      @test v == vd && Matrix(S) == Yd
      @test Yd == Yd'
      @test isapprox(v, Float64(elastic_conjugate_reference(λ1, λ2, w, Yd)); rtol = 1e-13)
      alias = SymmetricSparseIterate(M)
      @test value_and_prox!(alias, c, alias, γ) == vd
      @test Matrix(alias) == Yd
    end
  end

  @testset "blocks of a separable sum" begin
    c = Constrained(f, IndBox(-2.0, 2.0))
    s = SeparableSum(SqrNormL2(2.0), c)
    point = ([3.0], x)
    dest = ([7.0], zeros(3))
    ref1, ref2 = zeros(1), zeros(3)
    expected = value_and_prox!(ref1, SqrNormL2(2.0), [3.0], 0.5) + value_and_prox!(ref2, c, x, 0.5)
    @test value_and_prox!(dest, s, point, 0.5) == expected
    @test dest == (ref1, ref2)
    @test point == ([3.0], [5.0, -5.0, 0.5])
  end

  @testset "allocations" begin
    x64 = [5.0, -5.0, 0.5]
    steps = (0.5, [0.5, 0.25, 1.0])
    for T in (Float32, Float64), γ in steps
      xt = T.(x64)
      step = γ isa Real ? T(γ) : T.(γ)
      for g in (ElasticNet(T(1), T(0.5)), ElasticNet(T(1), T(0.5), T[1, 0, 2]),
                ElasticNet(T(1), T(0)), ElasticNet(T(1), T(0), T[1, 0, 2]), ElasticNet(T(0), T(0.5)))
        for box in (IndBox(T(-2), T(2)), IndBox(fill(T(-2), 3), fill(T(2), 3)), IndNonnegative())
          c = Constrained(Conjugate(g), box)
          y = zeros(T, 3)
          @test allocs(value_and_prox!, y, c, xt, step) == 0
          z = copy(xt)
          @test allocs(value_and_prox!, z, c, z, step) == 0
          @test allocs(prox!, y, c, xt, step) == 0
        end
      end
    end
  end

  @testset "controls: values that were available" begin
    xc = [5.0, -5.0, 0.5, 1.2]
    for g in (NormL1(0.3), NormL1(0.3, [1.0, 0.0, 2.0, 0.5]), NormL1Nonneg(0.3), ElasticNet(0.3, 0.4),
              ElasticNet(0.3, 0.4, [1.0, 0.0, 2.0, 0.5]), SqrNormL2(0.7), SqrNormL2(0.7, [1.0, 0.0, 2.0, 0.5]),
              IndBox(-1.5, 1.5), Conjugate(NormL1(0.3)), Conjugate(SqrNormL2(0.7)), Conjugate(IndBox(-1.0, 1.0))),
        box in (IndBox(-1.0, 1.0), IndNonnegative()), γ in (0.6, [0.6, 0.3, 1.0, 0.8])
      c = Constrained(g, box)
      y = zeros(4)
      v = value_and_prox!(y, c, xc, γ)
      @test y == prox(c, xc, γ)
      @test v ≈ c(y)
      @test PB._has_value(c)
      @test allocs(value_and_prox!, y, c, xc, γ) == 0
    end
  end
end

end
