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
    (IndSphereL2(1.),                    false, true,  false, false),
    (IndSimplex(1.),                     false, true,  false, true),
    (IndHyperplane(ones(3), 1.),         false, true,  false, true),
    (IndHalfspace(ones(3), 1.),          false, true,  false, true),
  ]
  for (f, smooth, proximable, separable, convex) in table
    @testset "$(nameof(typeof(f)))" begin
      check_traits(f; smooth, proximable, separable, convex)
      @test f isa AbstractFunction
      @test iszeropreserving(f) isa Bool
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
  @test ProximalBase.check_parameters(Box2(), x) === nothing
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
  # a coordinate is checked against ncoordinates(x), not length(x) = p²
  @test_throws BoundsError prox_coordinate!(x, f, ncoordinates(x) + 1, γ)
  @test_throws BoundsError prox_coordinate!(x, f, 0, γ)

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

# a user-defined separable function, which is not built in
struct KktCustom{T} <: AbstractFunction
  λ::T
end
ProximalBase.kkt_violation(f::KktCustom, g::AbstractArray, i::Integer) = abs(g[i]) - f.λ

# valid calls with the arguments of the checks below, as `kkt_violation` returned them before the
# checks were added (value and type); the expected values are the scalar formulas
@testset "kkt_violation validates the coordinate and the gradient extent" begin
  # independent formulas
  l1(g, w, i) = abs(g[i]) - w[i]
  grp(g, G, λ) = sqrt(sum(abs2, g[G])) - λ
  big_ids = (0, -1, typemin(Int), typemax(Int), big(10)^30, -big(10)^30, typemax(UInt), UInt(0), typemax(Int128), typemin(Int128))
  for T in (Float64, Float32)
    f = NormL1(T(1))
    g = T[3, -0.5, 7, 1.5]
    gc = copy(g)

    # SparseIterate, vector and matrix: the coordinates are 1:length(x)
    for x in (SparseIterate(T, 4), SparseIterate(T, 2, 2))
      xc = copy(Array(x))
      for k in 1:4
        v = kkt_violation(f, g, x, k)
        @test v isa T
        @test v == l1(g, ones(T, 4), k)
      end
      # the flat gradient and the matrix-shaped gradient of the same length are accepted
      @test [kkt_violation(f, reshape(g, 2, 2), x, k) for k in 1:4] == [l1(g, ones(T, 4), k) for k in 1:4]
      wf = NormL1(T(1), T[1, 2, 3, 4])
      @test [kkt_violation(wf, g, x, k) for k in 1:4] == [abs(g[k]) - k for k in 1:4]
      for k in (5, big_ids..., big(5), UInt(5), Int8(5), Int32(0))
        @test_throws BoundsError kkt_violation(f, g, x, k)
        @test_throws BoundsError kkt_violation(f, vcat(g, T(2)), x, k)   # the id is tested first
      end
      for gbad in (T[], g[1:1], g[1:3], vcat(g, T(2)), zeros(T, 3, 2), zeros(T, 2, 3), zeros(T, 4, 2), zeros(T, 2, 2, 2))
        length(gbad) == 4 && continue
        @test_throws DimensionMismatch kkt_violation(f, gbad, x, 1)
        @test_throws DimensionMismatch kkt_violation(wf, gbad, x, 4)
      end
      @test_throws DimensionMismatch kkt_violation(f, zeros(T, 3), x, 3)
      @test g == gc
      @test Array(x) == xc
      @test nactive(x) == 0
    end
    # a valid id of another integer type, and a view or a transpose as the gradient
    for x in (SparseIterate(T, 4), SparseIterate(T, 2, 2)), k in (Int8(3), UInt(3), UInt8(3), big(3))
      @test kkt_violation(f, g, x, k) == T(6)
    end
    @test kkt_violation(f, view(vcat(T(0), g, T(0)), 2:5), SparseIterate(T, 4), 3) == T(6)
    @test kkt_violation(f, transpose(reshape(g, 2, 2)), SparseIterate(T, 2, 2), 2) == abs(g[3]) - 1
    @test kkt_violation(f, reshape(g, 2, 2)', SparseIterate(T, 4), 2) == abs(g[3]) - 1
    # the issue example: g of length 3 and k = 3 on a SparseIterate(2)
    @test_throws BoundsError kkt_violation(NormL1(T(1)), T[1, 2, 7], SparseIterate(T, 2), 3)
    @test_throws DimensionMismatch kkt_violation(NormL1(T(1)), T[1], SparseIterate(T, 2), 1)
    @test kkt_violation(NormL1(T(1)), T[1, 2], SparseIterate(T, 2), 2) == T(1)
    # a nonzero iterate is checked in the same way
    xn = SparseIterate(T[0, 2, 0, 0]); xn[3] = 1
    @test kkt_violation(f, g, xn, 3) == T(6)
    @test_throws BoundsError kkt_violation(f, g, xn, 5)
    # a function that is not built in takes the same checks
    @test_throws BoundsError kkt_violation(KktCustom(T(1)), g, SparseIterate(T, 4), 5)
    @test_throws DimensionMismatch kkt_violation(KktCustom(T(1)), g[1:3], SparseIterate(T, 4), 1)
    @test kkt_violation(KktCustom(T(1)), g, SparseIterate(T, 4), 2) == T(0.5) - 1

    # an iterate without coordinates has no valid id
    for e in (SparseIterate(T, 0), SparseIterate(T, 0, 3))
      for k in (0, 1, -1, big(10)^30)
        @test_throws BoundsError kkt_violation(f, T[], e, k)
      end
      @test_throws BoundsError kkt_violation(f, T[1], e, 1)    # no valid id, whatever the gradient
    end

    # the owner: coordinates are the p(p+1)/2 entries of the lower triangle
    S = SymmetricSparseIterate(T, 2)
    G = T[1 2; 4 3]                       # not symmetric: entry (2, 1) is the mean of 4 and 2
    sym = (G + G') / 2
    for gg in (G, vec(G))
      @test [kkt_violation(f, gg, S, k) for k in 1:3] == [abs(sym[1, 1]) - 1, abs(sym[2, 1]) - 1, abs(sym[2, 2]) - 1]
      @test all(kkt_violation(f, gg, S, k) isa T for k in 1:3)
    end
    for k in (Int8(2), UInt(2), big(2), UInt8(2))
      @test kkt_violation(f, G, S, k) == abs(sym[2, 1]) - 1
    end
    @test kkt_violation(f, view(vcat(T(0), vec(G), T(0)), 2:5), S, 2) == abs(sym[2, 1]) - 1
    @test kkt_violation(f, transpose(G), S, 2) == abs(sym[2, 1]) - 1
    Gc = copy(G)
    for k in (0, 4, 10^6, big_ids..., big(4), UInt(4), Int8(4))
      @test_throws BoundsError kkt_violation(f, G, S, k)
      @test_throws BoundsError kkt_violation(f, zeros(T, 3), S, k)    # the id is tested first
    end
    for gbad in (T[], zeros(T, 3), zeros(T, 5), zeros(T, 3, 3), zeros(T, 2, 3), zeros(T, 2, 2, 2)[:, :, 1:0])
      @test_throws DimensionMismatch kkt_violation(f, gbad, S, 1)
    end
    @test kkt_violation(f, zeros(T, 1, 4), S, 1) == -1       # a matrix-shaped gradient of length p^2 needs not be square
    Gi = T[1 Inf; -Inf 3]
    err = try kkt_violation(f, Gi, S, 2); nothing catch e; e end
    @test err isa ArgumentError
    @test_throws ArgumentError kkt_violation(f, Gi, S, 2)
    @test_throws BoundsError kkt_violation(f, Gi, S, 4)
    @test G == Gc
    @test nactive(S) == 0
    # p = 1 and p = 0
    @test kkt_violation(f, T[-2;;], SymmetricSparseIterate(T, 1), 1) == T(1)
    @test_throws BoundsError kkt_violation(f, T[-2;;], SymmetricSparseIterate(T, 1), 2)
    @test_throws DimensionMismatch kkt_violation(f, T[-2, 1], SymmetricSparseIterate(T, 1), 1)
    for k in (0, 1, big(10)^30)
      @test_throws BoundsError kkt_violation(f, zeros(T, 0, 0), SymmetricSparseIterate(T, 0), k)
    end
    # a larger owner reads each coordinate at its position
    P = 4
    GP = T[10i + j for i in 1:P, j in 1:P]
    SP = SymmetricSparseIterate(T, P)
    for c in 1:P, r in c:P
      k = P * (c - 1) - div(c * (c - 1), 2) + r
      @test kkt_violation(f, GP, SP, k) == (GP[r, c] + GP[c, r]) / 2 - 1
    end
    @test_throws BoundsError kkt_violation(f, GP, SP, 11)
    @test_throws DimensionMismatch kkt_violation(f, GP[:, 1:3], SP, 1)

    # AtomIterate: unchanged, a group is a coordinate and the layout is checked
    A = AtomIterate(T, 4, [1:2, 3:4])
    gf = GroupNormL2(T(1), [1:2, 3:4])
    @test kkt_violation(gf, g, A, 1) ≈ grp(g, 1:2, sqrt(T(2)))
    @test kkt_violation(gf, g, A, 2) ≈ grp(g, 3:4, sqrt(T(2)))
    @test kkt_violation(gf, g, A, 1) isa T
    @test kkt_violation(f, g, A, 1) == maximum(i -> abs(g[i]) - 1, 1:2)
    @test kkt_violation(f, g, A, 2) == maximum(i -> abs(g[i]) - 1, 3:4)
    for k in (0, 3, -1)
      @test_throws BoundsError kkt_violation(gf, g, A, k)
      @test_throws BoundsError kkt_violation(f, g, A, k)
    end
    @test_throws DimensionMismatch kkt_violation(GroupNormL2(T(1), [1:3, 4:4]), g, A, 1)
    @test_throws DimensionMismatch kkt_violation(GroupNormL2(T(1), [3:4, 1:2]), g, A, 1)
    @test g == gc

    # valid calls as the base returned them (the Float64 values are printed in full)
    sg = SparseGroupNormL2(T(1), T(0.5), [1:2, 3:4])
    sgv = [kkt_violation(sg, g, A, k) for k in 1:2]
    @test sgv isa Vector{T}
    @test sgv ≈ T[1.2928932188134525, 5.3136905082096]
    @test [kkt_violation(gf, g, A, k) for k in 1:2] ≈ T[1.6271677027760145, 5.744696969265082]
    @test [kkt_violation(f, g, A, k) for k in 1:2] == T[2, 6]
    @test [kkt_violation(NormL1(T(1), T[1, 2, 3, 4]), g, A, k) for k in 1:2] == T[2, 4]
    @test [kkt_violation(NormL1(T(1), T[1 2; 2 3]), G, S, k) for k in 1:3] == T[0, 1, 0]
  end

  # the valid path allocates nothing
  kkt_alloc(f, g, x, k) = @allocated kkt_violation(f, g, x, k)
  let g = [3.0, -0.5, 7.0, 1.5], x = SparseIterate(4), xm = SparseIterate(2, 2), S = SymmetricSparseIterate(2),
      G = [1.0 2; 4 3], A = AtomIterate(4, [1:2, 3:4]), f = NormL1(1.0), gf = GroupNormL2(1.0, [1:2, 3:4])
    for (ff, gg, xx, k) in ((f, g, x, 3), (f, g, xm, 3), (f, G, S, 2), (gf, g, A, 1), (f, g, A, 1))
      kkt_violation(ff, gg, xx, k)
      @test kkt_alloc(ff, gg, xx, k) == 0
    end
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

@testset "gradient_mapping" begin
  n, p = 50, 12
  X, y = randn(n, p), randn(n)
  f, g = LeastSquaresLoss(y, X), NormL1(5.)
  L = lipschitz(f)
  β, grad = zeros(p), zeros(p)
  for _ in 1:5000
    gradient!(grad, f, β)
    β .-= grad ./ L
    prox!(β, g, β, 1 / L)
  end
  # zero at the solution, for any step, and not elsewhere
  @test norm(gradient_mapping(f, g, β, 1 / L)) < 1e-10
  @test norm(gradient_mapping(f, g, β, 0.1 / L)) < 1e-10
  @test norm(gradient_mapping(f, g, zeros(p), 1 / L)) > 1
  # the gradient when g = 0, and a proximal gradient step is x - γG
  x = randn(p)
  @test gradient_mapping(f, ZeroFunction(), x, 0.3) ≈ gradient(f, x)
  G = similar(x)
  @test gradient_mapping!(G, f, g, x, 1 / L) === G
  @test x - G / L ≈ prox(g, x - gradient(f, x) / L, 1 / L)
  @test_throws ArgumentError gradient_mapping!(x, f, g, x, 1 / L)
  out = copy(x)
  @test_throws DimensionMismatch gradient_mapping!(out, f, NormL1(1., ones(p - 1)), x, 1 / L)
  @test out == x
end

@testset "proximal operators in place" begin
  # prox!(y, f, y, γ) is prox(f, y, γ): solvers and gradient_mapping! rely on it
  n = 12
  x = randn(n)
  X, y = randn(30, n), randn(30)
  M, P = randn(8, 6), Matrix(Symmetric(randn(8, 8)))
  B = randn(20, 8)
  S = Matrix(Symmetric(B'B / 20))
  vec_fs = (NormL1(0.3), NormL1(0.3, rand(n)), NormL1Nonneg(0.2), ElasticNet(0.3, 0.5), IndBox(-0.5, 0.5),
            IndNonnegative(), IndBallLinf(0.4), NormL0(0.2), MCP(0.4), SCAD(0.4), NormL2(0.5), NormLinf(0.5),
            GroupNormL2(0.4, [1:4, 5:12]), GroupNormLinf(0.4, [1:4, 5:12]), IndBallL0(3), IndBallL1(1.),
            IndBallL2(1.), IndSphereL2(1.), IndSimplex(1.), IndHyperplane(randn(n), 0.5), IndHalfspace(randn(n), -0.5),
            ZeroFunction(), LinearFunction(randn(n)), SqrNormL2(2.), SqrNormL2(2., randn(n)),
            QuadraticFunction(X'X, randn(n)), LeastSquaresLoss(y, X), 2 * NormL1(0.3),
            Tilt(NormL1(0.3), randn(n), 1.), Translate(NormL1(0.3), randn(n)), Conjugate(NormL2(1.)),
            Conjugate(NormL1(1.)))
  mat_fs = ((NuclearNorm(0.5), M), (IndBallRank(2), M), (IndPSD(), P), (IndPSD(rank = 2), P), (TracePSD(0.3), P),
            (LogDetLoss(S), Matrix(1.0I, 8, 8) + 0.1 * P), (LeastSquaresLoss(randn(30, 3), X), randn(n, 3)))
  for (f, z) in (((f, x) for f in vec_fs)..., mat_fs...)
    w = copy(z)
    @test prox!(w, f, w, 0.7) ≈ prox(f, z, 0.7)
  end
end

end
