module ConstrainedNonconvexTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# the objective of the scalar proximal problem on [l, u], Inf outside
function box_objective(f, l, u, x, s)
  t -> (l <= t <= u ? value_entry(f, t, 1) : Inf) + (t - x)^2 / (2s)
end

@testset "MCP, SCAD and NormL0 on a box" begin
  fs = (MCP(0.5, 3.), MCP(0.6, 0.8), SCAD(0.5, 3.7), SCAD(0.4, 2.2), NormL0(0.3), NormL0(1.))
  boxes = ((-1., 0.5), (0.2, 2.), (-2., -0.1), (0., Inf), (-Inf, 0.), (-0.3, 0.3), (1., 1.))
  for f in fs, (l, u) in boxes, s in (0.4, 1., 2.5, 5.)
    c = Constrained(f, IndBox(l, u))
    for x in (4 .* randn(20)..., 0., l, u, -l, -u)
      isfinite(x) || continue
      p = prox_entry(c, x, 1, s)
      @test l <= p <= u
      h = box_objective(f, l, u, x, s)
      lo, hi = isfinite(l) ? l : min(x, 0.) - 10, isfinite(u) ? u : max(x, 0.) + 10
      t, ht = brute_force_prox(t -> (l <= t <= u ? value_entry(f, t, 1) : Inf), x, s; lo, hi)
      @test h(p) <= ht + 1e-9 * max(1, abs(ht))
    end
  end

  # the counterexample of the issue: clamping hard thresholding is not the prox
  c = Constrained(NormL0(1.), IndBox(-1., 0.5))
  @test clamp(prox_entry(NormL0(1.), 2., 1, 1.), -1., 0.5) == 0.5
  @test prox_entry(c, 2., 1, 1.) == 0.

  # in the convex regime, the clamp of the unconstrained proximal operator
  x = 3 .* randn(30)
  for (f, s) in ((MCP(0.5, 3.), 1.), (MCP(0.5, 3.), 2.9), (SCAD(0.5, 3.7), 1.), (SCAD(0.5, 3.7), 2.6))
    c = Constrained(f, IndBox(-1., 2.))
    @test prox(c, x, s) == clamp.(prox(f, x, s), -1., 2.)
  end

  # arrays of bounds, the value, coordinate steps and sparse iterates
  n = 12
  l, u = -rand(n), rand(n)
  for f in (MCP(0.3, 1.5), SCAD(0.3, 2.5), NormL0(0.2))
    c = Constrained(f, IndBox(l, u))
    check_traits(c; proximable = true, separable = true)
    v = 2 .* randn(n)
    p = check_prox(c, v, 2.)
    @test all(i -> prox_entry(c, v[i], i, 2.) == p[i], 1:n)
    @test value_and_prox!(similar(v), c, v, 2.) ≈ f(p)
    @test c(p) ≈ f(p) && c(fill(5., n)) == Inf
    xs = SparseIterate(sparse(v .* (rand(n) .< 0.5)))
    ys = prox(c, xs, 2.)
    @test ys isa SparseIterate && Array(ys) ≈ prox(c, Array(xs), 2.)
    @test iszeropreserving(c)
    for k in 1:n
      w = xs[k]
      @test prox_coordinate!(xs, c, k, 2.) == prox_entry(c, w, k, 2.)
    end
    S = [2, 5, 9]
    @test prox(restrict(c, S), v[S], 2.) ≈ p[S]
  end
  @test !iszeropreserving(Constrained(MCP(1.), IndBox(0.5, 1.)))

  # the direction that leaves zero must be open
  for f in (MCP(1., 3.), SCAD(1., 3.7))
    c = Constrained(f, IndBox(0., 2.))
    g = [-3., 3., 0.5, -0.5]
    @test kkt_violation(c, g, 1) ≈ 2.
    @test kkt_violation(c, g, 2) <= 0
    @test kkt_violation(c, g, 3) <= 0 && kkt_violation(c, g, 4) <= 0
    @test kkt_violation(Constrained(f, IndBox(-2., 2.)), g, 2) ≈ 2.
    # a violation means that a proximal gradient step from zero leaves it
    for i in 1:4
      moves = prox_entry(c, -g[i], i, 1.) != 0
      @test moves == (kkt_violation(c, g, i) > 0)
    end
  end

  # still only convex functions on a hyperplane, and only these nonconvex ones on a box
  @test_throws ArgumentError Constrained(MCP(1.), IndHyperplane(1, 0))
  @test_throws ArgumentError Constrained(SCAD(1.), IndHalfspace(1, 0))
  @test_throws ArgumentError Constrained(NormL0(1.), IndSimplex(1.))
  @test_throws ArgumentError Constrained(Translate(MCP(1.), ones(3)), IndBox(0, 1))
  @test !isconvex(Constrained(MCP(1.), IndBox(0, 1)))
  @test isconvex(Constrained(NormL1(1.), IndBox(0, 1)))
  @test isconvex(Constrained(Constrained(NormL1(1.), IndBox(-1, 1)), IndBox(0, 2)))
  @test !isconvex(Constrained(Constrained(MCP(1.), IndBox(-1, 1)), IndBox(0, 2)))
  # a box on a box: the penalty on the intersection
  c2 = Constrained(Constrained(MCP(1., 1.5), IndBox(-1, 1)), IndBox(0, 2))
  x = 3 .* randn(10)
  @test prox(c2, x, 4.) == prox(Constrained(MCP(1., 1.5), IndBox(0, 1)), x, 4.)
  @test prox(Constrained(Constrained(NormL0(1.), IndBox(-1, 1)), IndNonnegative()), x, 2.) ==
        prox(Constrained(NormL0(1.), IndBox(0, 1)), x, 2.)
end

intersect(f, a, b) = Constrained(Constrained(f, a), b)

@testset "exact linear box intersections" begin
  for f in (MCP(1.), SCAD(1.), NormL0(1.))
    for (a, b, expected) in (
        (IndBox(100_000, Float16(Inf)), IndBox(Float16(0), Float16(Inf)), (100_000., Inf)),
        (IndBox(Float16(-Inf), -100_000), IndBox(Float16(-Inf), Float16(0)), (-Inf, -100_000.)))
      for (first, second) in ((a, b), (b, a))
        c = intersect(f, first, second)
        @test (c.box.l, c.box.u) === expected
        exact = Constrained(f, IndBox(expected...))
        @test isequal(prox(c, [-200_000., 0., 200_000.]), prox(exact, [-200_000., 0., 200_000.]))
      end
    end
    k = 2^53
    @test_throws ArgumentError intersect(f, IndBox(k, k), IndBox(k + 1, k + 1))
    @test_throws ArgumentError intersect(f, IndBox(k + 1, k + 1), IndBox(Float64(k), Float64(k)))
    c = intersect(f, IndBox(k + 1, k + 1), IndBox(k + 1, k + 1))
    @test c.box.l === c.box.u === k + 1
    # Finite rounding alone does not widen the promoted type.
    c = intersect(f, IndBox(2049, 3000), IndBox(Float16(0), Float16(Inf)))
    @test c.box.l === Float16(2049)
    @test c.box.u === Float16(3000)
    @test_throws ArgumentError intersect(f, IndBox(2049, 2049), IndBox(Float16(2048), Float16(2048)))

    a, b = IndBox([0 1], [2 3]), IndBox([0, 1], [5, 5])
    for (first, second, shape) in ((a, b, (1, 2)), (b, a, (2,)))
      box = intersect(f, first, second).box
      @test vec(box.l) == [0, 1]
      @test vec(box.u) == [2, 3]
      @test size(box.l) == size(box.u) == shape
    end
    box = intersect(f, IndBox([0 10], 20), IndBox(-1, [5, 15])).box
    @test vec(box.l) == [0, 10]
    @test vec(box.u) == [5, 15]
    @test size(box.l) == size(box.u) == (1, 2)
    @test_throws DimensionMismatch intersect(f, IndBox([0, 1], 3), IndBox(0, [3, 4, 5]))
    @test_throws DimensionMismatch intersect(f, IndBox([0, 1], 3), IndBox([0], 4))
    box = intersect(f, IndBox(zeros(0, 2), 3), IndBox(0, Int[])).box
    @test isempty(box.l) && isempty(box.u)
    @test size(box.l) == size(box.u) == (0, 2)

    for T in (Int, Float16, Float32, Float64, Rational{Int}, BigFloat)
      a, b = IndBox(T(0), T(3)), IndBox(T(1), T(2))
      box = intersect(f, a, b).box
      @test box.l === T(1) || (T == BigFloat && box.l == T(1))
      @test box.u == T(2)
      @test typeof(box.l) == typeof(box.u) == T
    end
    for T in (Float16, Float32, Float64), a in (T(0), -T(0)), b in (T(0), -T(0))
      box = intersect(f, IndBox(a, a), IndBox(b, b)).box
      @test isequal(box.l, max(a, b))
      @test isequal(box.u, min(a, b))
    end
    for (a, b, T) in ((IndBox(1//3, 5//3), IndBox(0.5, 2.), Float64),
                      (IndBox(big"0.25", big"1.5"), IndBox(0, 2), BigFloat))
      box = intersect(f, a, b).box
      @test box.l == max(a.l, b.l) && box.u == min(a.u, b.u)
      @test typeof(box.l) == typeof(box.u) == T
    end
    for (a, b) in ((IndBox(-1., 3.), IndBox(0., 2.)),
                   (IndBox(Float32[-1, 0], Float32[2, 3]), IndBox(0f0, 2f0)))
      c = intersect(f, a, b)
      base = Constrained(f, IndBox(max.(a.l, b.l), min.(a.u, b.u)))
      @test isequal(c.box.l, base.box.l) && isequal(c.box.u, base.box.u)
      x = Float32[-2, 3]
      @test isequal(c(x), base(x))
      @test isequal(prox(c, x), prox(base, x))
    end
    c = intersect(f, IndBox(1, 2), IndBox(1, 2))
    c = Constrained(c, IndBox(1, 2))
    base = Constrained(f, IndBox(1, 2))
    t = sqrt(eps(Float32))
    for x in (Float32[1 - t, 2 + 2t], Float32[prevfloat(1 - t), nextfloat(2 + 2t)], Float32[0, 3])
      @test isequal(c(x), base(x))
      @test isequal(f(x), c.f(x))
      @test isequal(prox_entry(c, x[1], 1, 1f0), prox_entry(base, x[1], 1, 1f0))
      y, z = similar(x), similar(x)
      @test isequal(prox!(y, c, x, 1f0), prox!(z, base, x, 1f0))
      @test isequal(value_and_prox!(y, c, x, 1f0), value_and_prox!(z, base, x, 1f0))
      @test isequal(y, z)
    end
    @test intersect(f, IndBallLinf(2), IndBox(1, 3)).box.l === 1
  end
end

@testset "allocations, inference and Float32" begin
  n = 40
  v = 3 .* randn(n)
  y = similar(v)
  for c in (Constrained(MCP(0.4, 1.2), IndBox(-1., 2.)), Constrained(SCAD(0.4, 2.3), IndNonnegative()),
            Constrained(NormL0(0.5), IndBox(-rand(n), rand(n))),
            intersect(MCP(0.4, 1.2), IndBox(-1, 3), IndBox(0, 2)),
            intersect(SCAD(0.4, 2.3), IndBox(-1., 3.), IndBox(0., 2.)),
            intersect(NormL0(0.5), IndBox(-rand(n), rand(n)), IndBox(-0.5, 0.5)))
    @test allocs((y, x) -> prox!(y, c, x, 3.), y, v) == 0
    @test allocs((y, x) -> value_and_prox!(y, c, x, 3.), y, v) == 0
    @inferred prox_entry(c, v[1], 1, 3.)
    @inferred prox!(y, c, v, 3.)
    @inferred value_and_prox!(y, c, v, 3.)
    @inferred c(v)
    z = prox(c, Float32.(v), 3f0)
    @test eltype(z) == Float32
    @test z ≈ prox(c, v, 3.) atol = 1e-5
  end
end

end
