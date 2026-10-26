module NormsTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

@testset "NormL2" begin
  x = randn(10)
  λ = norm(x) + 0.1
  f = NormL2(λ)
  @test f(x) ≈ λ * norm(x)
  @test prox(f, x) == zeros(10)
  x = [1., 2.]
  @test prox(NormL2(1.), x) ≈ (1 - 1 / sqrt(5)) * x
  @test norm(prox(NormL2(1.), x)) ≈ sqrt(5) - 1
  @test NormL2(2.) isa NormL2{Float64} && NormL2(2f0) isa NormL2{Float32}
  for x in (randn(10), randn(4, 5))
    check_prox(NormL2(0.6), x, 0.9)
  end
  xs = SparseIterate(sprandn(30, 0.2))
  @test NormL2(1.)(xs) ≈ norm(Vector(xs))
  @test Vector(prox(NormL2(0.3), xs)) ≈ prox(NormL2(0.3), Vector(xs))
  D = randn(5, 5); D = D + D'
  s = SymmetricSparseIterate(D)
  @test NormL2(1.)(s) ≈ norm(D)
  prox!(s, NormL2(0.5), s, 1.)                       # in place, each entry once
  @test Matrix(s) ≈ prox(NormL2(0.5), D)
end

@testset "NormLinf" begin
  x = [3., -1., 0.5, 2.]
  f = NormLinf(1.)
  @test f(x) == 3.
  # by Moreau: x minus the projection onto the l1 ball of radius γλ
  z = prox(f, x, 2.)
  @test z ≈ x - prox(IndBallL1(2.), x)
  @test z ≈ [1.5, -1., 0.5, 1.5]
  @test prox(f, x, 10.) == zeros(4)                  # ‖x‖₁ ≤ γλ
  for x in (randn(12), randn(3, 4))
    check_prox(NormLinf(0.7), x, 1.3)
  end
  @test value_and_prox!(similar(x), f, x, 2.) ≈ f(z)
  @test NormLinf(0.5)(Float32.(x)) isa Float32
end

@testset "group norms" begin
  # groups of unequal sizes, one not contiguous; index 9 is in no group
  groups = [1:3, [4, 7], 5:5, [8, 6]]
  x = [1., -2., 0.5, 0.1, 3., -0.4, -0.2, 0.3, 2.]
  λ0, γ = 0.4, 0.7

  @testset "GroupNormL2 value and prox" begin
    f = GroupNormL2(λ0, groups)
    w = sqrt.(length.(groups))
    @test f.w == w
    @test f.groups isa Vector{Vector{Int}}      # the mixed groups are stored as vectors
    @test f(x) ≈ λ0 * sum(w[k] * norm(x[groups[k]]) for k in 1:4)
    @test f(SparseIterate(x)) ≈ f(x)
    z = prox(f, x, γ)
    for k in 1:4
      G = groups[k]
      @test z[G] ≈ max(1 - γ * λ0 * w[k] / norm(x[G]), 0) * x[G]
    end
    @test z[[4, 7]] == [0., 0.]       # ‖x[[4, 7]]‖ < γ λ0 √2
    @test z[9] == x[9]
    check_prox(f, x, γ)
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(z)
  end

  @testset "GroupNormLinf value and prox" begin
    f = GroupNormLinf(λ0, groups)
    @test f.w == ones(4)
    @test f(x) ≈ λ0 * sum(norm(x[G], Inf) for G in groups)
    z = prox(f, x, γ)
    for k in 1:4
      G = groups[k]
      @test z[G] ≈ prox(NormLinf(λ0), x[G], γ)
    end
    @test z[9] == x[9]
    check_prox(f, x, γ)
    check_prox(GroupNormLinf(0.5, [1:4, 5:9], [1., 2.]), randn(9), 0.6)
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(z)
  end

  @testset "special cases" begin
    v = randn(10)
    singletons = [[i] for i in 1:10]
    # groups of one index: the ℓ1 norm
    @test prox(GroupNormL2(0.3, singletons), v, γ) ≈ prox(NormL1(0.3), v, γ)
    @test GroupNormL2(0.3, singletons)(v) ≈ NormL1(0.3)(v)
    @test prox(GroupNormLinf(0.3, singletons), v, γ) ≈ prox(NormL1(0.3), v, γ)
    w = rand(10)
    @test prox(GroupNormL2(0.3, singletons, w), v, γ) ≈ prox(NormL1(0.3, w), v, γ)
    # one group: the Euclidean and ℓ∞ norms
    @test prox(GroupNormL2(0.3, [1:10], [1.]), v, γ) ≈ prox(NormL2(0.3), v, γ)
    @test prox(GroupNormLinf(0.3, [1:10]), v, γ) ≈ prox(NormLinf(0.3), v, γ)
    # groups of linear indices of a matrix, here its rows
    M = randn(3, 4)
    Z = prox(GroupNormL2(0.3, [i:3:12 for i in 1:3], ones(3)), M, γ)
    for i in 1:3
      @test Z[i, :] ≈ max(1 - γ * 0.3 / norm(M[i, :]), 0) * M[i, :]
    end
    # a zero weight leaves its group unpenalized; no groups, nothing
    @test prox(GroupNormL2(λ0, groups, [1., 1., 0., 1.]), x, γ)[5] == x[5]
    f0 = GroupNormL2(λ0, UnitRange{Int}[])
    @test f0(x) == 0.
    @test prox(f0, x, γ) == x
  end

  @testset "constructors" begin
    for F in (GroupNormL2, GroupNormLinf)
      @test_throws ArgumentError F(λ0, [1:3, 3:4])        # overlapping
      @test_throws ArgumentError F(λ0, [[1, 2, 1]])        # repeated index
      @test_throws ArgumentError F(λ0, [1:3, Int[]])       # empty
      @test_throws ArgumentError F(λ0, [0:2])
      @test_throws ArgumentError F(-1., [1:3])
      @test_throws ArgumentError F(λ0, [1:3], [-1.])
      @test_throws DimensionMismatch F(λ0, [1:3], [1., 2.])
      f = F(λ0, groups)
      @test_throws BoundsError f(randn(7))                 # index 8 is in a group
      @test_throws BoundsError prox(f, randn(7))
      @test_throws DimensionMismatch prox!(zeros(8), f, x, γ)
    end
    # the groups are copied
    gs = [1:2, 3:4]
    gc = GroupNormL2(1., gs)
    gs[1] = 1:1
    @test gc.groups == [1:2, 3:4]
    @test GroupNormL2(0.4f0, groups) isa GroupNormL2{Float32}
  end

  @testset "on an AtomIterate" begin
    for F in (GroupNormL2, GroupNormLinf)
      # with index 9 in a group of weight 0, so that the groups partition 1:9
      f = F(λ0, groups)
      fp = F(λ0, [groups; [[9]]], [f.w; 0.])
      @test fp(x) ≈ f(x)
      @test prox(fp, x, γ) ≈ prox(f, x, γ)

      xa = AtomIterate(x, fp)
      @test xa.groups === fp.groups && xa.group_of === fp.group_of
      @test xa == x && nactive(xa) == 5
      @test eltype(AtomIterate(9, fp)) == Float64 && nactive(AtomIterate(9, fp)) == 0
      @test eltype(AtomIterate(Float32, 9, fp)) == Float32
      @test_throws ArgumentError AtomIterate(x, f)           # index 9 is in no group of f
      @test_throws ArgumentError AtomIterate(10, fp)

      # the value and prox! visit the active groups when the groups are shared, and take
      # the methods for arrays otherwise
      @test fp(xa) ≈ fp(x)
      other = AtomIterate(x, [groups; [[9]]])
      @test fp(other) ≈ fp(x)
      out = similar(xa)
      @test prox!(out, fp, xa, γ) === out
      @test out ≈ prox(fp, x, γ)
      @test activecoordinates(out) == activecoordinates(xa)
      @test value_and_prox!(similar(xa), fp, xa, γ) ≈ fp(prox(fp, x, γ))
      out2 = AtomIterate(9, 3)
      prox!(out2, fp, other, γ)
      @test out2 ≈ prox(fp, x, γ)

      # prox_coordinate! shrinks one group in place, as prox! shrinks all of them
      z = prox(fp, x, γ)
      ya = copy(xa)
      for k in 1:5
        @test prox_coordinate!(ya, fp, k, γ) ≈ z[fp.groups[k]]
      end
      @test ya ≈ z
      # a group written through atom becomes active if prox_coordinate! leaves it nonzero
      yb = AtomIterate(9, fp)
      atom(yb, 1) .= [3., 4., 0.]
      prox_coordinate!(yb, fp, 1, γ)
      @test isactive(yb, 1)
      prox_coordinate!(yb, fp, 2, γ)
      @test !isactive(yb, 2)
      @test_throws DimensionMismatch prox_coordinate!(yb, F(λ0, [1:9]), 1, γ)
      # as many groups, but other ones
      fo = F(1.0, [1:2, 3:4])
      ao = AtomIterate([1., 2., 3., 4.], [[1, 3], [2, 4]])
      @test_throws DimensionMismatch prox_coordinate!(ao, fo, 1, 0.1)
      @test_throws DimensionMismatch kkt_violation(fo, randn(4), ao, 1)
      @test prox_coordinate!(AtomIterate([1., 2., 3., 4.], [1:2, 3:4]), fo, 1, 0.1) isa AbstractVector   # equal, not shared

      # kkt_violation takes a group: at zero, a group violates optimality exactly when a
      # proximal gradient step moves it away from zero
      grad = [0.3, -1.2, 0.8, 0.1, -0.5, 0.2, 0.6, -0.1, 0.4]
      x0 = AtomIterate(9, fp)
      step = prox(fp, -grad)
      dual = F === GroupNormL2 ? norm : (v -> norm(v, 1))
      for k in 1:5
        G = fp.groups[k]
        @test kkt_violation(fp, grad, x0, k) ≈ dual(grad[G]) - λ0 * fp.w[k]
        @test (kkt_violation(fp, grad, x0, k) > 0) == !iszero(step[G])
        # for a separable penalty, the worst index of the group
        @test kkt_violation(NormL1(0.5), grad, x0, k) == maximum(i -> kkt_violation(NormL1(0.5), grad, i), G)
      end
    end
    g32 = GroupNormL2(0.4f0, [groups; [[9]]], Float32[sqrt.(length.(groups)); 0])
    x32 = AtomIterate(Float32.(x), g32)
    @test eltype(prox_coordinate!(x32, g32, 1, 0.7f0)) == Float32
    @test g32(x32) isa Float32
  end

  @testset "restrict and kkt_violation" begin
    for F in (GroupNormL2, GroupNormLinf)
      fw = F(λ0, groups, [1., 2., 3., 4.])
      S = [7, 2, 9, 1, 4]       # part of group 1, all of group 2, and index 9
      z = zeros(9)
      z[S] = x[S]               # supported on S
      fS = restrict(fw, S)
      @test fS.groups == [[2, 4], [1, 5]]       # positions in S
      @test fS.w == [1., 2.]
      @test fS.λ0 == λ0
      @test fS(x[S]) ≈ fw(z)
      @test prox(fS, x[S], γ) ≈ prox(fw, z, γ)[S]
      fE = restrict(fw, [9])                     # no group meets S
      @test fE([5.]) == 0.
      @test prox(fE, [5.]) == [5.]

      f = F(λ0, groups)
      grad = [0.3, -1.2, 0.8, 0.1, -0.5, 0.2, 0.6, -0.1, 0.4]
      xk = zeros(9)
      xk[4], xk[5] = 1., 2.     # groups 2 and 3 are nonzero, groups 1 and 4 zero
      dual = F === GroupNormL2 ? norm : (v -> norm(v, 1))
      @test kkt_violation(f, grad, xk, 1) ≈ dual(grad[1:3]) - λ0 * f.w[1]
      @test kkt_violation(f, grad, xk, 3) == kkt_violation(f, grad, xk, 1)
      @test kkt_violation(f, grad, xk, 7) == abs(grad[7])      # x[7] = 0 in a nonzero group
      @test kkt_violation(f, grad, xk, 9) == abs(grad[9])      # in no group
      @test kkt_violation(f, grad, SparseIterate(xk), 7) == abs(grad[7])
      @test_throws MethodError kkt_violation(f, grad, 1)
      @test_throws ArgumentError kkt_violation(f, grad, SymmetricSparseIterate(3), 1)
      for i in 1:9
        @test (kkt_violation(f, grad, zeros(9), i) > 0) == !iszero(prox(f, -grad)[i])
      end
    end
  end
end

@testset "NormL1Fused" begin
  @testset "optimality" begin
    # (u, v) minimizes ((u - x)² + (v - y)²) / 2γ + λ1 (|u| + |v|) + λ2 |u - v| if and
    # only if 0 is a subgradient there
    ∂abs(z) = z > 0 ? (1., 1.) : z < 0 ? (-1., -1.) : (-1., 1.)
    function optimal(u, v, x, y, λ1, λ2, γ; tol = 1e-10)
      (al, ah), (bl, bh), (sl, sh) = ∂abs(u), ∂abs(v), ∂abs(u - v)
      gu, gv = (x - u) / γ, (y - v) / γ
      max(λ2 * sl, gu - λ1 * ah - tol, λ1 * bl - gv - tol) <=
        min(λ2 * sh, gu - λ1 * al + tol, λ1 * bh - gv + tol)
    end
    for i = 1:1000, (λ1, λ2) in ((rand(), rand()), (0., rand()), (rand(), 0.))
      x, y, γ = randn(), randn(), 0.1 + 2 * rand()
      u, v = prox(NormL1Fused(λ1, λ2), (x, y), γ)
      @test optimal(u, v, x, y, λ1, λ2, γ)
      @test !optimal(u + 1e-3, v, x, y, λ1, λ2, γ)     # the check rejects other points
    end
    @test prox(NormL1Fused(0.2, 0.5), (0.3, -1.2)) == prox(NormL1Fused(0.2, 0.5), (0.3, -1.2), 1.)
  end
  @testset "vectors" begin
    f = NormL1Fused(0.3, 0.7)
    x1, x2 = randn(20), randn(20)
    out = (zeros(20), zeros(20))
    @test prox!(out, f, (x1, x2), 0.5) === out
    for i = 1:20
      @test (out[1][i], out[2][i]) == prox(f, (x1[i], x2[i]), 0.5)
    end
    @test prox(f, (x1, x2), 0.5) == out
    @test f((1., -2.)) ≈ 0.3 * 3. + 0.7 * 3.
    @test f((x1, x2)) ≈ sum(f((x1[i], x2[i])) for i in 1:20)
  end
  @testset "Float32" begin
    z = prox(NormL1Fused(0.1f0, 0.5f0), (1f0, 1.1f0))
    @test z isa Tuple{Float32, Float32}
    @test all(z .≈ 0.95f0)
  end
end

@testset "Euclidean penalties preserve extreme directions" begin
  for T in (Float32, Float64)
    large = T == Float32 ? T(1e30) : T(1e200)
    for scale in (large, inv(large))
      x = T[0, scale]
      @test NormL2(one(T))(x) ≈ scale rtol=8eps(T)
      @test prox(NormL2(scale/2), x) ≈ T[0, scale/2] rtol=8eps(T)
      g = GroupNormL2(scale/2, [1:2], T[1])
      @test prox(g, x) ≈ T[0, scale/2] rtol=8eps(T)
      a = AtomIterate(x, g)
      prox_coordinate!(a, g, 1, one(T))
      @test Vector(a) ≈ T[0, scale/2] rtol=8eps(T)
    end
  end
end


@testset "zero norm levels before overflowing reductions" begin
  for T in (Float32, Float64)
    x = T[floatmax(T), -floatmax(T)]
    for f in (NormL2(zero(T)), NormL1(zero(T)), NormL1Fused(zero(T),zero(T)),
              TotalVariation1D(zero(T)), FusedLasso(zero(T),zero(T)),
              SqrNormL2(zero(T)), GroupNormL2(zero(T), [1:2], T[1]))
      @test f(x) == zero(T)
      y = similar(x)
      @test value_and_prox!(y, f, x, one(T)) == zero(T)
      @test y == x
    end
    f = NuclearNorm(zero(T))
    X = reshape(copy(x), 2, 1)
    Y = similar(X)
    @test f(X) == zero(T)
    @test value_and_prox!(Y, f, X, one(T)) == zero(T)
    @test Y == X
  end
end


end
