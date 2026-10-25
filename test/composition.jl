module CompositionTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

@testset "Precompose" begin
  n = 8
  x = randn(n)
  Q = Matrix(qr(randn(n, n)).Q)
  @testset "orthogonal A" begin
    h = Precompose(NormL1(0.4), Q, 1.0)
    @test h(x) ≈ 0.4 * norm(Q * x, 1)
    @test isproximable(h) && isconvex(h) && !isseparable(h)
    p = check_prox(h, x, 0.6)
    @test p ≈ Q' * prox(NormL1(0.4), Q * x, 0.6)
    y = similar(x)
    @test value_and_prox!(y, h, x, 0.6) ≈ h(y)
  end
  @testset "a permutation, a selection and a sum of pairs" begin
    π = randperm(n)
    P = Matrix(1.0I, n, n)[π, :]
    h = Precompose(NormL1(0.5), P, 1.0)
    @test prox(h, x, 0.7) ≈ prox(NormL1(0.5), x, 0.7)     # ℓ1 does not see the order
    S = Matrix(1.0I, n, n)[1:3, :]                       # the first three entries
    hs = Precompose(NormL1(0.5), sparse(S), 1.0)
    @test hs(x) ≈ 0.5 * sum(abs, x[1:3])
    @test prox(hs, x, 0.7) ≈ [prox(NormL1(0.5), x[1:3], 0.7); x[4:end]]
    check_prox(hs, x, 0.7)
    A = [Matrix(1.0I, 4, 4) Matrix(1.0I, 4, 4)]           # AA' = 2I
    ha = Precompose(NormL2(1.0), A, 2.0)
    check_prox(ha, x, 0.3)
    @test A * prox(ha, x, 0.3) ≈ prox(NormL2(1.0), A * x, 0.6)
  end
  @testset "smooth f" begin
    b = randn(n)
    h = Precompose(SqrNormL2(1.0, b), Q, 1.0)
    @test h(x) ≈ norm(Q * x - b)^2 / 2
    check_gradient(h, x)
    @test lipschitz(h) ≈ 1.0
    A = [Matrix(1.0I, 4, 4) Matrix(1.0I, 4, 4)]
    @test lipschitz(Precompose(SqrNormL2(3.0), A, 2.0)) ≈ 6.0
  end
  @testset "an indicator" begin
    h = Precompose(IndBallL2(1.0), Q, 1.0)
    y = similar(x)
    @test value_and_prox!(y, h, 3 .* x, 0.5) == 0
    @test norm(y) ≈ 1
  end
  @testset "a scalar" begin
    for a in (2.0, -0.5)
      h = Precompose(NormL1(0.3), a)
      @test isseparable(h)
      @test h(x) ≈ 0.3 * abs(a) * norm(x, 1)
      @test prox(h, x, 0.7) ≈ shrink.(x, 0.7 * 0.3 * abs(a))
      check_prox(h, x, 0.7)
      @test [prox_entry(h, x[i], i, 0.7) for i in 1:n] ≈ prox(h, x, 0.7)
      g = randn(n)
      @test kkt_violation(h, g, 2) ≈ abs(g[2]) - 0.3 * abs(a)
      M = randn(3, 4)
      @test Precompose(SqrNormL2(1.0), a)(M) ≈ a^2 * sum(abs2, M) / 2
      check_gradient(Precompose(SqrNormL2(1.0), a), M)
      @test restrict(h, [2, 3])(x[2:3]) ≈ h([0; x[2:3]; zeros(n - 3)])
    end
  end
  @test_throws ArgumentError Precompose(NormL1(1.0), randn(3, n), 1.0)
  @test Precompose(NormL1(1.0), randn(3, n), 1.0; check = false) isa Precompose
  @test_throws ArgumentError Precompose(NormL1(1.0), 0.0)
  @test_throws ArgumentError Precompose(NormL1(1.0), Q, -1.0)
end

@testset "Regularize" begin
  n = 8
  x = 2 .* randn(n)
  @testset "the elastic net" begin
    r = Regularize(NormL1(0.3), 0.7)
    en = ElasticNet(0.3, 0.7)
    @test r(x) ≈ en(x)
    @test prox(r, x, 0.4) ≈ prox(en, x, 0.4)
    @test [prox_entry(r, x[i], i, 0.4) for i in 1:n] ≈ prox(en, x, 0.4)
    @test isseparable(r) && isconvex(r) && isproximable(r) && !issmooth(r)
    check_prox(r, x, 0.4)
    g = randn(n)
    @test kkt_violation(r, g, 3) ≈ kkt_violation(en, g, 3)
  end
  @testset "with a center" begin
    c = randn(n)
    r = Regularize(NormL1(0.5), 0.8, c)
    @test r(x) ≈ 0.5 * norm(x, 1) + 0.4 * norm(x - c)^2
    p = check_prox(r, x, 0.6)
    for i in 1:n
      @test p[i] ≈ brute_force_prox(t -> 0.5abs(t) + 0.4 * (t - c[i])^2, x[i], 0.6)[1] atol = 1e-6
    end
    g = randn(n)
    @test kkt_violation(r, g, 4) ≈ abs(g[4] - 0.8 * c[4]) - 0.5
    @test restrict(r, [2, 5])(x[[2, 5]]) ≈ 0.5 * norm(x[[2, 5]], 1) + 0.4 * norm(x[[2, 5]] - c[[2, 5]])^2
    y = similar(x)
    @test value_and_prox!(y, r, x, 0.6) ≈ r(y)
    xs = SparseIterate(sparse(x))
    for k in 1:n
      prox_coordinate!(xs, r, k, 0.6)
    end
    @test Vector(xs) ≈ p
  end
  @testset "the group elastic net" begin
    groups = [1:3, 4:5, 6:8]
    gn = GroupNormL2(0.6, groups)
    r = Regularize(gn, 0.5)
    p = check_prox(r, x, 0.7)
    # on an AtomIterate, group by group
    a = AtomIterate(x, gn)
    for k in 1:3
      prox_coordinate!(a, r, k, 0.7)
    end
    @test Vector(a) ≈ p
    rc = Regularize(gn, 0.5, randn(n))
    pc = check_prox(rc, x, 0.7)
    ac = AtomIterate(x, gn)
    for k in 1:3
      prox_coordinate!(ac, rc, k, 0.7)
    end
    @test Vector(ac) ≈ pc
    grad = randn(n)
    @test kkt_violation(r, grad, AtomIterate(n, gn), 2) ≈ kkt_violation(gn, grad, AtomIterate(n, gn), 2)
  end
  @testset "the matrix elastic net" begin
    M = randn(6, 4)
    r = Regularize(NuclearNorm(0.4), 0.9)
    check_prox(r, M, 0.5)
    @test prox(r, M, 0.5) ≈ prox(NuclearNorm(0.4), M ./ (1 + 0.45), 0.5 / (1 + 0.45))
  end
  @testset "smooth f" begin
    X, yv = randn(30, n), rand(0:1, 30)
    c = randn(n)
    f = Regularize(LogisticLoss(yv, X), 0.3, c)
    @test issmooth(f)
    check_gradient(f, randn(n))
    @test lipschitz(f) ≈ lipschitz(LogisticLoss(yv, X)) + 0.3
    @test check_gradient(Regularize(LeastSquaresLoss(randn(30), X), 2.0), SparseIterate(sprandn(n, 0.5))) isa Vector
  end
  @test_throws ArgumentError Regularize(NormL1(1.0), -1.0)
  @test_throws DimensionMismatch prox(Regularize(NormL1(1.0), 1.0, randn(3)), x, 0.5)
end

@testset "composed second-order oracles" begin
  for T in (Float32, Float64)
    X, y = randn(T, 14, 4), T.(rand(0:1, 14))
    inner = LogisticLoss(y, X)
    A = hcat(Matrix{T}(I, 4, 4), Matrix{T}(I, 4, 4))
    for f in (Regularize(inner, T(0.7), randn(T, 4)),
              Precompose(inner, A, T(2)), Precompose(inner, sparse(A), T(2)),
              Precompose(inner, T(-2)))
      p = f isa Precompose && f.A isa AbstractMatrix ? 8 : 4
      x, v = randn(T, p) ./ T(5), randn(T, p)
      Hv = similar(x)
      @test hashessian(f)
      @test hessian_vec!(Hv, f, x, v) === Hv
      step = cbrt(eps(T))
      ref = (gradient(f, x .+ step .* v) .- gradient(f, x .- step .* v)) ./ (2step)
      tol = T === Float32 ? 3e-3 : 1e-6
      @test Hv ≈ ref rtol = tol
      q = quadratic_model(f, x)
      @test gradient(q, x) ≈ gradient(f, x)
      @test hessian_vec(q, x, v) ≈ Hv
      x2 = x .+ T(0.1) .* v
      @test quadratic_model!(q, f, x2) === q
      @test gradient(q, x2) ≈ gradient(f, x2)
      @test hessian_vec(q, x2, v) ≈ hessian_vec(f, x2, v)
      @test allocs(hessian_vec!, Hv, f, x, v) == 0
      @test allocs(quadratic_model!, q, f, x) == 0
      @inferred hessian_vec!(Hv, f, x, v)
      @inferred quadratic_model!(q, f, x)
      @test_throws DimensionMismatch hessian_vec!(zeros(T, p + 1), f, x, v)
      @test_throws DimensionMismatch hessian_vec!(Hv, f, x, zeros(T, p + 1))
    end
    # Scalar precomposition also acts on matrices and sparse directions.
    x, v = randn(T, 3, 4), randn(T, 3, 4)
    h = Precompose(SqrNormL2(T(0.6)), T(-2))
    @test hessian_vec(h, x, SparseIterate(v)) ≈ T(2.4) .* v
    @test gradient(quadratic_model(h, x), x) ≈ gradient(h, x)
    r = Regularize(SqrNormL2(T(0.6)), T(0.7))
    @test hessian_vec(r, x, v) ≈ T(1.3) .* v
    @test !hashessian(Regularize(NormL1(), T(0.5)))
    @test !hashessian(Precompose(NormL1(), T(2)))
  end
  h = Precompose(SqrNormL2(), -2.)
  @test_throws ArgumentError quadratic_model!(quadratic_model(h, ones(3)), Precompose(h.f, 2.), ones(3))
  r = Regularize(SqrNormL2(), 0.5)
  @test_throws ArgumentError quadratic_model!(quadratic_model(r, ones(3)), Regularize(r.f, 0.7), ones(3))
end

@testset "composed coordinate curvatures" begin
  X, y = randn(16, 5), randn(16)
  f = LeastSquaresLoss(y, X)
  for outer in (Regularize(f, 0.7), Regularize(f, 0.7, ones(5)),
                Precompose(f, -2.), Regularize(Precompose(f, -2.), 0.7))
    ref = [coordinate_lipschitz(outer, k) for k in 1:5]
    @test coordinate_lipschitz!(zeros(5), outer) ≈ ref
    expected = outer isa Precompose ? 4 .* vec(sum(abs2, X; dims = 1)) :
               outer.f isa Precompose ? 4 .* vec(sum(abs2, X; dims = 1)) .+ 0.7 :
               vec(sum(abs2, X; dims = 1)) .+ 0.7
    @test ref ≈ expected
    @test allocs(coordinate_lipschitz!, zeros(5), outer) == 0
    @test allocs(coordinate_lipschitz, outer, 2) == 0
    @test_throws BoundsError coordinate_lipschitz(outer, 6)
    @test_throws DimensionMismatch coordinate_lipschitz!(zeros(4), outer)
  end
end

a_value(f, x) = f(x)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)

@testset "zero allocations" begin
  n = 20
  x = randn(n)
  y, g = similar(x), similar(x)
  Q = Matrix(qr(randn(n, n)).Q)
  for f in (Precompose(NormL1(0.3), Q, 1.0), Precompose(NormL1(0.3), 2.0), Regularize(NormL1(0.3), 0.5),
            Regularize(NormL1(0.3), 0.5, randn(n)), Regularize(GroupNormL2(0.5, [1:10, 11:20]), 0.5))
    @testset "$(nameof(typeof(f)))" begin
      @test allocs(a_value, f, x) == 0
      @test allocs(a_prox!, y, f, x, 0.4) == 0
      @test allocs(a_vprox!, y, f, x, 0.4) == 0
      @test allocs(a_prox!, x, f, x, 0.4) == 0
      @inferred value_and_prox!(y, f, x, 0.4)
    end
  end
  for f in (Precompose(SqrNormL2(1.0, randn(n)), Q, 1.0), Regularize(LeastSquaresLoss(randn(30), randn(30, n)), 0.5))
    @test allocs(a_vgrad!, g, f, x) == 0
  end
  xs = SparseIterate(sprandn(n, 0.3))
  @test allocs(a_cd!, xs, Regularize(NormL1(0.3), 0.5), 3, 0.4) == 0
  gn = GroupNormL2(0.5, [1:10, 11:20])
  @test allocs(a_cd!, AtomIterate(x, gn), Regularize(gn, 0.5), 2, 0.4) == 0
end

end
