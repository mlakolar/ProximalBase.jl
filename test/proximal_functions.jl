module  ProximalFunctionTest

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

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
  @test prox_coordinate!(f, xs, 3, 1.) == x[3]
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

  @testset "prox_coordinate!" begin
    x = randn(10)
    λ0 = 0.5

    xs = SparseIterate(x)
    g = ProxL1(λ0)
    for k=1:10
      @test prox_coordinate!(g, xs, k, 2.) ≈ shrink(x[k], 2. * λ0)
    end
    @test Vector(xs) ≈ prox(g, x, 2.)

    λ = rand(10)
    xs = SparseIterate(x)
    g = ProxL1(λ0, λ)
    for k=1:10
      @test prox_coordinate!(g, xs, k, 2.) ≈ shrink(x[k], 2. * λ0 * λ[k])
    end
    @test Vector(xs) ≈ prox(g, x, 2.)
    @test_throws DimensionMismatch prox_coordinate!(ProxL1(λ0, rand(5)), xs, 1, 1.)

    # coordinates of SymmetricSparseIterate index the lower triangle
    M = sprandn(4, 4, 0.8)
    M = (M + M') / 2
    xs = SymmetricSparseIterate(M)
    g = ProxL1(λ0)
    for k=1:numCoordinates(xs)
      v = xs[k]
      @test prox_coordinate!(g, xs, k, 1.) ≈ shrink(v, λ0)
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

  @testset "optimality" begin
    # (u, v) minimizes ((u - x)² + (v - y)²) / 2γ + λ1 (|u| + |v|) + λ2 |u - v| if and
    # only if 0 is a subgradient there: (x - u)/γ - t ∈ λ1 ∂|u| and
    # (y - v)/γ + t ∈ λ1 ∂|v| for some t ∈ λ2 ∂|u - v|, where ∂|z| = [-1, 1] at z = 0
    ∂abs(z) = z > 0 ? (1., 1.) : z < 0 ? (-1., -1.) : (-1., 1.)
    function optimal(u, v, x, y, λ1, λ2, γ; tol = 1e-10)
      (al, ah), (bl, bh), (sl, sh) = ∂abs(u), ∂abs(v), ∂abs(u - v)
      gu, gv = (x - u) / γ, (y - v) / γ
      # the interval of t allowed by each condition must intersect the others
      max(λ2 * sl, gu - λ1 * ah - tol, λ1 * bl - gv - tol) <=
        min(λ2 * sh, gu - λ1 * al + tol, λ1 * bh - gv + tol)
    end

    # λ1 = 0 only fuses, λ2 = 0 only shrinks
    for i = 1:1000, (λ1, λ2) in ((rand(), rand()), (0., rand()), (rand(), 0.))
      x, y, γ = randn(), randn(), 0.1 + 2 * rand()
      u, v = ProximalBase.proxL1Fused(x, y, λ1, λ2, γ)
      @test optimal(u, v, x, y, λ1, λ2, γ)
      @test !optimal(u + 1e-3, v, x, y, λ1, λ2, γ)     # the check rejects other points
    end
    @test ProximalBase.proxL1Fused(0.3, -1.2, 0.2, 0.5) == ProximalBase.proxL1Fused(0.3, -1.2, 0.2, 0.5, 1.)
  end

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
    @test ProxL2(2.) isa ProxL2{Float64}
    @test ProxL2(2f0) isa ProxL2{Float32}
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


@testset "proximal_group_l2" begin
  # groups of unequal sizes, one not contiguous; index 9 is in no group
  groups = [1:3, [4, 7], 5:5, [8, 6]]
  x = [1., -2., 0.5, 0.1, 3., -0.4, -0.2, 0.3, 2.]
  λ0, γ = 0.4, 0.7
  g = ProxGroupL2(λ0, groups)
  w = sqrt.(length.(groups))

  @testset "value and prox" begin
    @test g.λ == w
    @test g.groups isa Vector{Vector{Int}}      # the mixed groups are stored as vectors
    @test value(g, x) ≈ λ0 * sum(w[k] * norm(x[groups[k]]) for k in 1:4)
    @test value(g, SparseIterate(x)) ≈ value(g, x)

    z = prox(g, x, γ)
    for k in 1:4
      G = groups[k]
      @test z[G] ≈ max(1 - γ * λ0 * w[k] / norm(x[G]), 0) * x[G]
    end
    @test z[[4, 7]] == [0., 0.]       # ‖x[[4, 7]]‖ < γ λ0 √2
    @test z[9] == x[9]

    # z = prox_{γg}(x) exactly when (x - z) / γ is a subgradient of g at z
    for k in 1:4
      G = groups[k]
      if iszero(z[G])
        @test norm(x[G]) / γ <= λ0 * w[k]
      else
        @test (x[G] - z[G]) / γ ≈ λ0 * w[k] * z[G] / norm(z[G])
      end
    end

    out = randn(9)
    @test prox!(g, out, x, γ) === out
    @test out == z
    y = copy(x)
    @test prox!(g, y, y, γ) == z      # in place

    prox_allocs(g, out, x) = @allocated prox!(g, out, x, 0.5)
    value_allocs(g, x) = @allocated value(g, x)
    prox_allocs(g, out, x)
    value_allocs(g, x)
    @test prox_allocs(g, out, x) == 0
    @test value_allocs(g, x) == 0
  end

  @testset "special cases" begin
    v = randn(10)

    # groups of one index: the ℓ1 norm
    singletons = [[i] for i in 1:10]
    @test prox(ProxGroupL2(0.3, singletons), v, γ) ≈ prox(ProxL1(0.3), v, γ)
    @test value(ProxGroupL2(0.3, singletons), v) ≈ value(ProxL1(0.3), v)
    λ = rand(10)
    @test prox(ProxGroupL2(0.3, singletons, λ), v, γ) ≈ prox(ProxL1(0.3, λ), v, γ)

    # one group: the Euclidean norm
    @test prox(ProxGroupL2(0.3, [1:10], [1.]), v, γ) ≈ prox(ProxL2(0.3), v, γ)

    # groups of linear indices of a matrix, here its rows
    M = randn(3, 4)
    Z = prox(ProxGroupL2(0.3, [i:3:12 for i in 1:3], ones(3)), M, γ)
    for i in 1:3
      @test Z[i, :] ≈ max(1 - γ * 0.3 / norm(M[i, :]), 0) * M[i, :]
    end

    # a zero weight leaves its group unpenalized; no groups, nothing
    @test prox(ProxGroupL2(λ0, groups, [1., 1., 0., 1.]), x, γ)[5] == x[5]
    g0 = ProxGroupL2(λ0, UnitRange{Int}[])
    @test value(g0, x) == 0.
    @test prox(g0, x, γ) == x
  end

  @testset "constructors" begin
    @test_throws ArgumentError ProxGroupL2(λ0, [1:3, 3:4])        # overlapping
    @test_throws ArgumentError ProxGroupL2(λ0, [[1, 2, 1]])        # repeated index
    @test_throws ArgumentError ProxGroupL2(λ0, [1:3, Int[]])       # empty
    @test_throws ArgumentError ProxGroupL2(λ0, [0:2])
    @test_throws ArgumentError ProxGroupL2(-1., [1:3])
    @test_throws ArgumentError ProxGroupL2(λ0, [1:3], [-1.])
    @test_throws DimensionMismatch ProxGroupL2(λ0, [1:3], [1., 2.])

    @test_throws BoundsError value(g, randn(7))                    # index 8 is in a group
    @test_throws BoundsError prox(g, randn(7))
    @test_throws DimensionMismatch prox!(g, zeros(8), x, γ)

    # the groups are copied
    gs = [1:2, 3:4]
    gc = ProxGroupL2(1., gs)
    gs[1] = 1:1
    @test gc.groups == [1:2, 3:4]
  end

  @testset "on an AtomIterate" begin
    # the same penalty, with index 9 in a group of weight 0, so that the groups partition 1:9
    gp = ProxGroupL2(λ0, [groups; [[9]]], [w; 0.])
    @test value(gp, x) ≈ value(g, x)
    @test prox(gp, x, γ) ≈ prox(g, x, γ)

    xa = AtomIterate(x, gp)
    @test xa.groups === gp.groups && xa.group_of === gp.group_of
    @test xa == x && numActive(xa) == 5
    @test eltype(AtomIterate(9, gp)) == Float64 && numActive(AtomIterate(9, gp)) == 0
    @test eltype(AtomIterate(Float32, 9, gp)) == Float32
    @test_throws ArgumentError AtomIterate(x, g)           # index 9 is in no group of g
    @test_throws ArgumentError AtomIterate(10, gp)

    # value and prox! visit the active groups when the groups are shared, and take the
    # methods for arrays otherwise
    @test value(gp, xa) ≈ value(gp, x)
    other = AtomIterate(x, [groups; [[9]]])
    @test value(gp, other) ≈ value(gp, x)
    out = similar(xa)
    @test prox!(gp, out, xa, γ) === out
    @test out ≈ prox(gp, x, γ)
    @test out.active[1:numActive(out)] == xa.active[1:numActive(xa)]
    out2 = AtomIterate(9, 3)
    prox!(gp, out2, other, γ)
    @test out2 ≈ prox(gp, x, γ)

    # prox_coordinate! shrinks one group in place, as prox! shrinks all of them
    z = prox(gp, x, γ)
    ya = copy(xa)
    for k in 1:5
      @test prox_coordinate!(gp, ya, k, γ) == z[gp.groups[k]]
    end
    @test ya == z
    # a group written through atom becomes active if prox_coordinate! leaves it nonzero
    yb = AtomIterate(9, gp)
    atom(yb, 1) .= [3., 4., 0.]
    prox_coordinate!(gp, yb, 1, γ)
    @test isActive(yb, 1)
    @test yb[1:3] ≈ (1 - γ * λ0 * w[1] / 5) * [3., 4., 0.]
    prox_coordinate!(gp, yb, 2, γ)
    @test !isActive(yb, 2)
    @test_throws DimensionMismatch prox_coordinate!(ProxGroupL2(λ0, [1:9]), yb, 1, γ)

    # kkt_violation takes a group: at zero, a group violates optimality exactly when a
    # proximal gradient step moves it away from zero
    grad = [0.3, -1.2, 0.8, 0.1, -0.5, 0.2, 0.6, -0.1, 0.4]
    x0 = AtomIterate(9, gp)
    step = prox(gp, -grad)
    for k in 1:5
      G = gp.groups[k]
      @test kkt_violation(gp, grad, x0, k) ≈ norm(grad[G]) - λ0 * gp.λ[k]
      @test (kkt_violation(gp, grad, x0, k) > 0) == !iszero(step[G])
      # for a separable penalty, the worst index of the group
      @test kkt_violation(ProxL1(0.5), grad, x0, k) == maximum(i -> kkt_violation(ProxL1(0.5), grad, i), G)
    end

    g32 = ProxGroupL2(0.4f0, [groups; [[9]]], Float32[w; 0])
    x32 = AtomIterate(Float32.(x), g32)
    @test eltype(prox_coordinate!(g32, x32, 1, 0.7f0)) == Float32
    @test value(g32, x32) isa Float32

    cd_allocs(g, x, k, γ) = @allocated prox_coordinate!(g, x, k, γ)
    value_allocs(g, x) = @allocated value(g, x)
    cd_allocs(gp, ya, 1, 1e-12); value_allocs(gp, ya)
    @test cd_allocs(gp, ya, 1, 1e-12) == 0
    @test value_allocs(gp, ya) == 0
  end

  @testset "Float32" begin
    g32 = ProxGroupL2(0.4f0, groups)
    @test g32 isa ProxGroupL2{Float32}
    z32 = prox(g32, Float32.(x), 0.7f0)
    @test z32 isa Vector{Float32}
    @test z32 ≈ Float32.(prox(g, x, γ))
    @test value(g32, Float32.(x)) isa Float32
  end

  @testset "restrict" begin
    gw = ProxGroupL2(λ0, groups, [1., 2., 3., 4.])
    S = [7, 2, 9, 1, 4]       # part of group 1, all of group 2, and index 9
    z = zeros(9)
    z[S] = x[S]               # supported on S

    gS = restrict(gw, S)
    @test gS.groups == [[2, 4], [1, 5]]       # positions in S
    @test gS.λ == [1., 2.]
    @test gS.λ0 == λ0
    @test value(gS, x[S]) ≈ value(gw, z)
    @test prox(gS, x[S], γ) ≈ prox(gw, z, γ)[S]

    # no group meets S
    gE = restrict(gw, [9])
    @test value(gE, [5.]) == 0.
    @test prox(gE, [5.]) == [5.]
  end

  @testset "kkt_violation" begin
    grad = [0.3, -1.2, 0.8, 0.1, -0.5, 0.2, 0.6, -0.1, 0.4]
    xk = zeros(9)
    xk[4], xk[5] = 1., 2.     # groups 2 and 3 are nonzero, groups 1 and 4 zero

    @test kkt_violation(g, grad, xk, 1) ≈ norm(grad[1:3]) - λ0 * √3
    @test kkt_violation(g, grad, xk, 3) == kkt_violation(g, grad, xk, 1)
    @test kkt_violation(g, grad, xk, 7) == abs(grad[7])      # x[7] = 0 in a nonzero group
    @test kkt_violation(g, grad, xk, 9) == abs(grad[9])      # in no group
    @test kkt_violation(g, grad, SparseIterate(xk), 7) == abs(grad[7])
    @test_throws MethodError kkt_violation(g, grad, 1)

    # at x = 0, an index violates optimality exactly when a proximal gradient step
    # moves it away from zero
    for i in 1:9
      @test (kkt_violation(g, grad, zeros(9), i) > 0) == !iszero(prox(g, -grad)[i])
    end

    # for the separable penalties, the rest of x does not matter
    for gs in (ProxL1(0.5), ProxL1(0.5, rand(9)), ProxZero()), i in 1:9
      @test kkt_violation(gs, grad, xk, i) == kkt_violation(gs, grad, i)
    end
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
