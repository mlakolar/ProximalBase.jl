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

@testset "allocations, inference and Float32" begin
  n = 40
  v = 3 .* randn(n)
  y = similar(v)
  for c in (Constrained(MCP(0.4, 1.2), IndBox(-1., 2.)), Constrained(SCAD(0.4, 2.3), IndNonnegative()),
            Constrained(NormL0(0.5), IndBox(-rand(n), rand(n))))
    @test allocs((y, x) -> prox!(y, c, x, 3.), y, v) == 0
    @test allocs((y, x) -> value_and_prox!(y, c, x, 3.), y, v) == 0
    @inferred prox_entry(c, v[1], 1, 3.)
    z = prox(c, Float32.(v), 3f0)
    @test eltype(z) == Float32
    @test z ≈ prox(c, v, 3.) atol = 1e-5
  end
end

end
