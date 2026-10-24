module BallConstrainedTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

# the objective of the proximal problem of f on a set, γf(z) + ‖z - v‖²/2, Inf
# outside the set
obj(c, z, v, γ) = γ * c(z) + sum(abs2, z .- v) / 2

# random points on the sphere ‖z‖ = r, and points of the sphere near z
sphere_points(z, r, m) = [r .* w ./ norm(w) for w in (randn(size(z)...) for _ in 1:m)]
near_points(z, r, m) = [r .* w ./ norm(w) for w in (z .+ δ .* randn(size(z)...) for δ in (1e-6, 1e-3, 1e-1) for _ in 1:m)]

# the best point of the ball (or the sphere) supported on at most s entries, over
# every support: the projection of v onto the ball on the support
function l0_reference(v, s, r; sphere = false)
  n = length(v)
  best, bz = Inf, zero(v)
  for mask in 0:(2^n - 1)
    count_ones(mask) <= s || continue
    z = [(mask >> (i - 1)) & 1 == 1 ? v[i] : 0.0 for i in 1:n]
    nz = norm(z)
    if sphere
      iszero(nz) && continue
      z .*= r / nz
    elseif nz > r
      z .*= r / nz
    end
    d = sum(abs2, z .- v)
    d < best && ((best, bz) = (d, z))
  end
  bz
end

@testset "IndSphereL2" begin
  f = IndSphereL2(2.)
  @test prox(f, [3., 4.]) ≈ [1.2, 1.6]
  @test prox(f, zeros(3)) == [2., 0., 0.]
  @test f([1.2, 1.6]) == 0 && f([1., 1.]) == Inf && f(zeros(2)) == Inf
  @test f([1.2, 1.6 + 1e-12]) == 0                    # the tolerance of rounding
  M = randn(3, 4)
  @test prox(f, M) ≈ 2 .* M ./ norm(M)
  z = randn(5)
  w = copy(z)
  @test prox!(w, f, w, 1.) ≈ prox(f, z)               # in place
  @test value_and_prox!(similar(z), f, z, 1.) == 0
  check_traits(f; proximable = true)
  @test_throws ArgumentError IndSphereL2(0.)
  @test_throws ArgumentError IndSphereL2(-1.)
  @test_throws ArgumentError prox(f, Float64[])
  @test eltype(prox(IndSphereL2(1f0), Float32[3, 4])) == Float32
  @test allocs((y, x) -> prox!(y, f, x, 1.), similar(z), z) == 0
end

@testset "a norm on an ℓ2 ball" begin
  n = 8
  v = 2 .* randn(n)
  fs = (NormL1(0.3), NormL1(0.3, rand(n)), NormL1Nonneg(0.4), NormL2(0.5), NormLinf(0.2),
        GroupNormL2(0.4, [1:3, 4:5, [6, 8]]), GroupNormLinf(0.3, [1:4, 5:8]), Scaled(NormL1(1.), 0.3),
        ZeroFunction(), IndNonnegative())
  for f in fs, r in (0.5, 2., 50.), γ in (0.4, 1.5)
    @testset "$(nameof(typeof(f))), r = $r" begin
      c = Constrained(f, IndBallL2(r))
      check_traits(c; proximable = true, convex = true)
      p = check_prox(c, v, γ)
      # the proximal point of f, scaled onto the ball
      q = prox(f, v, γ)
      @test p ≈ (norm(q) > r ? (r / norm(q)) .* q : q)
      @test norm(p) <= r * (1 + 1e-12)
      # no point of the ball does better
      for w in sphere_points(p, r, 50), t in (0.3, 1.)
        @test obj(c, t .* w, v, γ) >= obj(c, p, v, γ) - 1e-10
      end
    end
  end

  # matrices: the nuclear norm and the trace on the Frobenius ball
  M = randn(6, 4)
  for (f, X) in ((NuclearNorm(0.4), M), (TracePSD(0.3), Matrix(Symmetric(randn(5, 5)))), (IndPSD(), Matrix(Symmetric(randn(5, 5)))))
    c = Constrained(f, IndBallL2(1.))
    p = check_prox(c, X, 0.7; symmetric = !(f isa NuclearNorm))
    q = prox(f, X, 0.7)
    @test p ≈ (norm(q) > 1 ? q ./ norm(q) : q)
    @test value_and_prox!(similar(X), c, X, 0.7) ≈ c(p)
  end

  # the nonconvex cones: the s largest entries, or the rank-r truncation, scaled
  for s in 0:4, _ in 1:10
    v6 = randn(6)
    c = Constrained(IndBallL0(s), IndBallL2(0.8))
    check_traits(c; proximable = true)
    @test prox(c, v6) ≈ l0_reference(v6, s, 0.8)
  end
  c = Constrained(IndBallRank(2), IndBallL2(1.))
  p = prox(c, M)
  @test rank(p) <= 2 && norm(p) <= 1 + 1e-12
  for _ in 1:200
    W = randn(6, 2) * randn(2, 4)
    W .*= rand() / norm(W)
    @test sum(abs2, W .- M) >= sum(abs2, p .- M) - 1e-10
  end

  # the value scales with the point
  c = Constrained(NormL1(0.5), IndBallL2(1.))
  p = prox(c, v)
  @test value_and_prox!(similar(v), c, v, 1.) ≈ 0.5 * norm(p, 1)
  @test c(p) ≈ 0.5 * norm(p, 1) && c(2 .* p) == Inf

  # an indicator stays an indicator, and its value is 0
  for C in (IndBallL2(1.), IndSphereL2(1.))
    @test ProximalBase._isindicator(typeof(Constrained(IndBallL0(2), C)))
    @test !ProximalBase._isindicator(typeof(Constrained(NormL1(1.), C)))
  end
  @test_throws ArgumentError Constrained(SqrNormL2(1.), IndBallL2(1.))
  @test_throws ArgumentError Constrained(MCP(1.), IndBallL2(1.))
  @test_throws ArgumentError Constrained(IndBox(-1, 1), IndBallL2(1.))
end

@testset "a norm on an ℓ2 sphere" begin
  n = 7
  @testset "NormL1" begin
    w = rand(n)
    for f in (NormL1(0.4), NormL1(0.4, w)), r in (0.5, 3.), γ in (0.5, 2.), scale in (0.1, 3.)
      v = scale .* randn(n)
      c = Constrained(f, IndSphereL2(r))
      check_traits(c; proximable = true)
      p = check_prox(c, v, γ)
      @test norm(p) ≈ r
      cands = [sphere_points(p, r, 200); near_points(p, r, 20)]
      for i in 1:n, σ in (-1, 1)                # the points on one axis
        e = zeros(n); e[i] = σ * r
        push!(cands, e)
      end
      @test all(z -> obj(c, z, v, γ) >= obj(c, p, v, γ) - 1e-9, cands)
    end
    # every entry within its threshold: all the mass on the entry of largest margin
    c = Constrained(NormL1(1., [1., 2., 1., 4.]), IndSphereL2(2.))
    v = [0.5, -1.9, -0.8, 3.]
    @test prox(c, v) == [0., -2., 0., 0.]                 # margins -0.5, -0.1, -0.2, -1
    @test prox(c, v) == prox!(copy(v), c, copy(v), 1.)    # in place
    @test value_and_prox!(similar(v), c, v, 1.) ≈ 4.
    @test prox(c, zeros(4)) == [2., 0., 0., 0.]
    @test_throws ArgumentError prox(Constrained(NormL1(1., fill(Inf, 3)), IndSphereL2(1.)), ones(3))
  end

  @testset "GroupNormL2" begin
    groups = [1:3, 4:5, [6, 7]]
    for f in (GroupNormL2(0.5, groups), GroupNormL2(0.5, [1:3, [6, 7]])), γ in (0.5, 2.), scale in (0.2, 3.)
      v = scale .* randn(n)
      c = Constrained(f, IndSphereL2(1.5))
      p = check_prox(c, v, γ)
      @test norm(p) ≈ 1.5
      cands = [sphere_points(p, 1.5, 300); near_points(p, 1.5, 20)]
      for G in groups                            # the points on one group
        z = zeros(n); z[G] .= v[G]; push!(cands, 1.5 .* z ./ norm(z))
      end
      @test all(z -> obj(c, z, v, γ) >= obj(c, p, v, γ) - 1e-9, cands)
    end
    # every group within its threshold: all the mass on the group of largest margin
    f = GroupNormL2(1., [1:2, 3:4], [1., 0.6])
    c = Constrained(f, IndSphereL2(2.))
    v = [0.3, 0.4, -0.3, 0.]                      # margins 0.5 - 1 and 0.3 - 0.6
    @test prox(c, v) ≈ [0., 0., -2., 0.]
    @test prox!(copy(v), c, copy(v), 1.) ≈ prox(c, v)
    @test value_and_prox!(similar(v), c, v, 1.) ≈ 1.2
    @test prox(c, zeros(4)) == [0., 0., 2., 0.]   # the group of the smaller threshold
    # an index in no group is free on the sphere
    c = Constrained(GroupNormL2(1., [1:2, [4]]), IndSphereL2(2.))
    @test prox(c, [0.1, 0.2, 0., -0.3]) == [0., 0., 2., 0.]
    @test value_and_prox!(zeros(4), c, [0.1, 0.2, 0., -0.3], 1.) == 0
  end

  @testset "IndBallL0" begin
    for s in 1:4, _ in 1:10
      v = randn(6)
      c = Constrained(IndBallL0(s), IndSphereL2(1.3))
      p = prox(c, v)
      @test p ≈ l0_reference(v, s, 1.3; sphere = true)
      @test prox!(copy(v), c, copy(v), 1.) ≈ p
    end
    c = Constrained(IndBallL0(2), IndSphereL2(1.))
    @test prox(c, zeros(3)) == [1., 0., 0.]
    @test value_and_prox!(zeros(3), c, [1., 2., 3.], 1.) == 0
    # with a free diagonal, the diagonal stays and does not count
    M = randn(4, 4)
    P = prox(Constrained(IndBallL0(2; diagonal = false), IndSphereL2(1.)), M)
    Q = prox(IndBallL0(2; diagonal = false), M)
    @test P ≈ Q ./ norm(Q)
    @test prox(Constrained(IndBallL0(0; diagonal = false), IndSphereL2(1.)), M) ≈ Diagonal(diag(M)) ./ norm(diag(M))
    @test_throws ArgumentError prox(Constrained(IndBallL0(0), IndSphereL2(1.)), randn(3))
  end

  @test_throws ArgumentError Constrained(NormL2(1.), IndSphereL2(1.))
  @test_throws ArgumentError Constrained(GroupNormLinf(1., [1:2]), IndSphereL2(1.))
  @test_throws ArgumentError Constrained(SqrNormL2(1.), IndSphereL2(1.))
end

@testset "allocations, inference and Float32" begin
  n = 50
  v = randn(n)
  y = similar(v)
  cs = (Constrained(NormL1(0.3), IndBallL2(1.)), Constrained(GroupNormL2(0.3, [1:10, 11:30]), IndBallL2(1.)),
        Constrained(IndBallL0(5), IndBallL2(1.)), Constrained(NormL1(0.3), IndSphereL2(1.)),
        Constrained(NormL1(10.), IndSphereL2(1.)), Constrained(GroupNormL2(0.3, [1:10, 11:30]), IndSphereL2(1.)),
        Constrained(GroupNormL2(10., [1:25, 26:50]), IndSphereL2(1.)), Constrained(IndBallL0(5), IndSphereL2(1.)))
  for c in cs
    @test allocs((y, x) -> value_and_prox!(y, c, x, 0.7), y, v) == 0
    @inferred value_and_prox!(y, c, v, 0.7)
    @inferred prox!(y, c, v, 0.7)
    z = prox(c, Float32.(v), 0.7f0)
    @test eltype(z) == Float32
    @test z ≈ prox(c, v, 0.7) atol = 1e-5
  end
  M = randn(8, 5)
  c = Constrained(NuclearNorm(0.3), IndBallL2(1.))
  @test allocs((Y, X) -> value_and_prox!(Y, c, X, 1.), similar(M), M) == 0
end

end
