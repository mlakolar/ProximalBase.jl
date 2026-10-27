module CalculusTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

@testset "Scaled" begin
  x = randn(8)
  f = NormL1(0.5)
  s = Scaled(f, 3.)
  @test 3 * f isa Scaled && f * 3 isa Scaled
  @test (3 * f)(x) ≈ s(x) ≈ 1.5 * sum(abs, x)
  @test prox(s, x, 0.4) ≈ prox(f, x, 1.2)
  @test 2 * s isa Scaled && (2 * s).α == 6. && (2 * s).f === f      # scaling twice multiplies
  check_prox(s, x, 0.4)
  @test value_and_prox!(similar(x), s, x, 0.4) ≈ s(prox(s, x, 0.4))
  # a scaled loss: the gradient scales
  X, y = randn(20, 8), randn(20)
  ls = (1 / 20) * LeastSquaresLoss(y, X)
  @test ls(x) ≈ norm(y - X * x)^2 / 40
  check_gradient(ls, x)
  @test issmooth(ls) && isproximable(ls)
  @test prox(ls, x, 2.) ≈ prox(LeastSquaresLoss(y, X), x, 0.1)
  # kkt_violation and restrict
  grad = randn(8)
  @test kkt_violation(s, grad, 3) ≈ abs(grad[3]) - 1.5
  @test kkt_violation(0 * f, grad, 3) == abs(grad[3])
  @test restrict(s, [2, 5])(x[[2, 5]]) ≈ s([0, x[2], 0, 0, x[5], 0, 0, 0])
  @test prox_entry(s, 2., 1, 0.4) ≈ shrink(2., 0.6)
  @test_throws ArgumentError Scaled(f, -1.)
  # a scaled group penalty in coordinate descent and active-set code
  gn = GroupNormL2(1.0, [1:2, 3:4])
  sg = 0.5 * gn
  grad = [0.3, -1.2, 0.8, 0.1]
  a0 = AtomIterate(4, gn)
  for k in 1:2
    @test kkt_violation(sg, grad, a0, k) ≈ 0.5 * kkt_violation(gn, 2 .* grad, a0, k)
    @test kkt_violation(sg, grad, a0, k) ≈ norm(grad[gn.groups[k]]) - 0.5 * gn.w[k]
  end
  @test kkt_violation(sg, grad, zeros(4), 1) ≈ norm(grad[1:2]) - 0.5 * sqrt(2)
  a1, a2 = AtomIterate([1., 2., 3., 4.], gn), AtomIterate([1., 2., 3., 4.], gn)
  @test prox_coordinate!(a1, sg, 2, 0.4) ≈ prox_coordinate!(a2, gn, 2, 0.2)
  @test kkt_violation(0 * gn, grad, a0, 1) == maximum(abs, grad[1:2])
end

@testset "Tilt" begin
  x = randn(8)
  c = randn(8)
  for f in (NormL1(0.4), SqrNormL2(1.5), MCP(0.3))
    t = Tilt(f, c, 2.)
    @test t(x) ≈ f(x) + dot(c, x) + 2
    @test prox(t, x, 0.6) ≈ prox(f, x - 0.6c, 0.6)
    check_prox(t, x, 0.6)
    @test [prox_entry(t, x[i], i, 0.6) for i in 1:8] ≈ prox(t, x, 0.6)
  end
  t = Tilt(SqrNormL2(1.5), c)
  check_gradient(t, x)
  @test gradient(t, x) ≈ 1.5x + c
  # at zero, the tilt shifts the gradient
  grad = randn(8)
  @test kkt_violation(Tilt(NormL1(0.4), c), grad, 2) ≈ abs(grad[2] + c[2]) - 0.4
  @test restrict(t, [1, 3])(x[[1, 3]]) ≈ t([x[1], 0, x[3], 0, 0, 0, 0, 0])
end

@testset "Translate" begin
  x = randn(8)
  b = randn(8)
  f = NormL1(0.4)
  t = Translate(f, b)
  @test t(x) ≈ 0.4 * sum(abs, x - b)
  @test prox(t, x, 0.7) ≈ b + prox(f, x - b, 0.7)
  check_prox(t, x, 0.7)
  @test [prox_entry(t, x[i], i, 0.7) for i in 1:8] ≈ prox(t, x, 0.7)
  # the loss ‖x - b‖²/2 of a translated squared norm
  q = Translate(SqrNormL2(), b)
  @test q(x) ≈ norm(x - b)^2 / 2
  check_gradient(q, x)
  check_prox(q, x, 0.7)
  @test value_and_prox!(similar(x), t, x, 0.7) ≈ t(prox(t, x, 0.7))
  @test_throws DimensionMismatch t(randn(3))
  M = randn(3, 4)
  @test Translate(NuclearNorm(1.), M)(M) ≈ 0
  for T in (Float32, Float64)
    s = Translate(NormL1(one(T)), T[1, 2, 3])
    r = restrict(s, [1, 3])
    for z in (T[0, 0], T[2, 4])
      @test s(T[z[1], 0, z[2]]) - r(z) ≈ T(2)
    end
    q = Translate(SqrNormL2(T(2)), T[1, 2, 3])
    rq = restrict(q, [1, 3])
    @test gradient(rq, T[2, 4]) ≈ gradient(q, T[2, 0, 4])[[1, 3]]
    @test q(T[2, 0, 4]) - rq(T[2, 4]) ≈ T(4)
    coupled = Translate(GroupNormL2(one(T), [1:2], T[1]), T[0, 1])
    @test_throws ArgumentError restrict(coupled, [1])
    @test restrict(coupled, [2])(T[2]) ≈ coupled(T[0, 2])
    @test restrict(coupled, [2, 1])(T[2, 1]) ≈ coupled(T[1, 2])
    @test_throws ArgumentError restrict(Translate(IndNonnegative(), T[0, 1]), [1])
    @test_throws ArgumentError restrict(Translate(NegLog(), zeros(T, 2)), [1])
    @test_throws BoundsError restrict(s, [4])
  end
end

@testset "Conjugate wrappers" begin
  for f in (NormL0(), MCP(1.0), GroupSCAD(1.0, [1:2]))
    @test_throws ArgumentError Conjugate(f)
    @test_throws ArgumentError Conjugate{typeof(f)}(f)
  end
  for T in (Float32,Float64)
    y,b = T[.1,-.2,.3], T[1,2,3]
    f = Scaled(Translate(SqrNormL2(T(2)),b),T(.5))
    @test Conjugate(f)(y) ≈ sum(abs2,y)/2 + dot(b,y)
    @test Conjugate(f)(y) isa T
    h = Scaled(Translate(CheckLoss(T(.7)),b),T(.5))
    @test Conjugate(h)(T[.1,-.1,.2]) ≈ dot(b,T[.1,-.1,.2])
    @test Conjugate(h)(T[.5,0,0]) == Inf
    @test_throws DimensionMismatch Conjugate(Translate(NormL1(),b))(zeros(T,2))
  end
  @test Conjugate(Scaled(NormL1(), 0.0))(zeros(3)) == 0
  @test Conjugate(Scaled(NormL1(), 0.0))(ones(3)) == Inf
  # Deep copies own their numerical workspace.
  c = Conjugate(NormL1(0.4))
  expected = prox(c, [2.0, -1.0])
  d = deepcopy(c)
  @test d.scratch !== c.scratch
  @test d.scratch.buf === nothing
  @test prox(d, [2.0, -1.0]) == expected
  @test d.scratch.buf !== c.scratch.buf
end

@testset "zero scales preserve domains" begin
  for T in (Float32, Float64)
    for f in (ZeroFunction(), NormL1(one(T)), GroupNormL2(one(T), [1:2]),
              GroupNormLinf(one(T), [1:2]), SparseGroupNormL2(one(T), one(T), [1:2]),
              NormL2(one(T)), NormLinf(one(T)), NormL0(one(T)),
              MCP(one(T)), SCAD(one(T)), GroupMCP(one(T), [1:2]), GroupSCAD(one(T), [1:2]),
              ElasticNet(one(T), one(T)), SqrNormL2(one(T), T[1, 2]),
              QuadraticFunction(Matrix{T}(I, 2, 2)),
              LeastSquaresLoss(T[1, 2], Matrix{T}(I, 2, 2)), LinearFunction(T[1, 2]))
      z = Scaled(f, zero(T))
      @test z isa ZeroFunction
      @test z(T[2, -1]) == 0
      @test prox(z, T[2, -1]) == T[2, -1]
      @test gradient(z, T[2, -1]) == zeros(T, 2)
      @test prox(z, T[2, -1], T[0.2, 0.8]) == T[2, -1]
      @test Scaled{typeof(f), T}(f, zero(T)) isa ZeroFunction
    end
    positive = Scaled(NormL1(one(T)), one(T))
    @test Scaled{typeof(positive), T}(positive, zero(T)) isa ZeroFunction
    infinite = Scaled(NormL1(one(T)), T(Inf))
    @test_throws ArgumentError Scaled{typeof(infinite), T}(infinite, zero(T))
    @test_throws ArgumentError Scaled(NormL1(one(T), T[1, Inf]), zero(T))
    x, steps = T[2, -1], T[0.2, 0.8]
    out = similar(x)
    @test value_and_prox!(out, ZeroFunction(), x, steps) == zero(T)
    @test out == x
    @test prox!(x, ZeroFunction(), x, steps) === x
    @test x == T[2, -1]
    @test allocs(value_and_prox!, out, ZeroFunction(), x, steps) == 0
    @test_throws DimensionMismatch prox(ZeroFunction(), x, steps[1:1])
    @test_throws DimensionMismatch prox!(ones(T, 3), ZeroFunction(), x, steps)
    C = IndNonnegative()
    @test Scaled(C, zero(T)) === C
    @test Scaled(C, zero(T))(T[-1]) == Inf
    @test Scaled(NormL1Nonneg(), zero(T))(T[-1]) == Inf
    @test_throws ArgumentError Scaled(NegLog(), zero(T))
    @test_throws ArgumentError Scaled{NegLog{Int}, T}(NegLog(), zero(T))
    @test_throws ArgumentError Scaled(NormL1(T(Inf)), zero(T))
    @test Conjugate(SqrNormL2(zero(T)))(T[0]) == 0
    @test Conjugate(SqrNormL2(zero(T)))(T[1]) == Inf
    @test_throws ArgumentError LogDetLoss(zeros(T, 1, 1), zero(T))
    @test_throws ArgumentError LogDetLoss(zeros(T, 1, 1), T(Inf))
  end
end

@testset "Conjugate" begin
  # Moreau: x = prox_{γf}(x) + γ prox_{f*/γ}(x/γ), for vectors and matrices
  vector_functions = [NormL1(0.6), NormL1(0.6, rand(10)), NormL2(0.6), NormLinf(0.6), IndBallL1(1.5),
                      IndBallL2(1.5), IndBox(-1., 2.), IndNonnegative(), SqrNormL2(2.), SqrNormL2(2., randn(10)),
                      ZeroFunction(), LinearFunction(randn(10), 0.4), GroupNormL2(0.6, [1:4, 5:10]),
                      GroupNormLinf(0.6, [1:4, 5:10]), IndSimplex(1.)]
  matrix_functions = [NormL1(0.6), NormL2(0.6), NormLinf(0.6), IndBallL1(1.5), IndBox(-1., 2.),
                      SqrNormL2(2.), NuclearNorm(0.6), IndBallL2(1.5)]
  for (fs, x) in ((vector_functions, randn(10)), (matrix_functions, randn(4, 5))), f in fs, γ in (0.3, 2.)
    c = Conjugate(f)
    @test prox(f, x, γ) + γ * prox(c, x ./ γ, 1 / γ) ≈ x
    @test prox(c, x, γ) ≈ x - γ * prox(f, x ./ γ, 1 / γ)
  end
  # the values: dual balls and support functions
  y = [0.3, -0.5, 0.1]
  @test Conjugate(NormL1(0.6))(y) == 0 && Conjugate(NormL1(0.4))(y) == Inf
  @test Conjugate(NormL2(0.6))(y) == 0 && Conjugate(NormL2(0.5))(y) == Inf
  @test Conjugate(NormLinf(0.9))(y) == 0 && Conjugate(NormLinf(0.8))(y) == Inf
  @test Conjugate(IndBallL1(2.))(y) ≈ 2 * 0.5
  @test Conjugate(IndBallL2(2.))(y) ≈ 2 * norm(y)
  @test Conjugate(IndBox([-1., -1., 0.], [2., 1., 1.]))(y) ≈ 0.3 * 2 + 0.5 * 1 + 0.1 * 1
  @test Conjugate(IndNonnegative())(-abs.(y)) == 0 && Conjugate(IndNonnegative())(y) == Inf
  @test Conjugate(SqrNormL2(2.))(y) ≈ norm(y)^2 / 4
  @test Conjugate(ZeroFunction())(zeros(3)) == 0 && Conjugate(ZeroFunction())(y) == Inf
  c = randn(3)
  @test Conjugate(LinearFunction(c, 1.))(c) == -1 && Conjugate(LinearFunction(c))(y) == Inf
  # the conjugate of the l1 norm is the indicator of the l∞ ball: the projection
  @test prox(Conjugate(NormL1(0.4)), y) ≈ clamp.(y, -0.4, 0.4)
  for f in (NormL1(0.6), NormL2(0.6), NormLinf(0.6), IndBallL1(1.5), IndBox(-1., 2.), SqrNormL2(2.))
    check_prox(Conjugate(f), 2 .* randn(10), 0.8)
  end
  @test_throws ArgumentError Conjugate(GroupNormL2(0.5, [1:2]))(randn(2))
  # separable conjugates have the scalar proximal operators
  cf = Conjugate(NormL1(0.6))
  @test isseparable(cf)
  @test [prox_entry(cf, y[i], i, 0.8) for i in 1:3] ≈ prox(cf, y, 0.8)
end

@testset "Euclidean conjugate values across numerical scales" begin
  for T in (Float32, Float64)
    large = T === Float32 ? T(1e30) : T(1e200)
    for scale in (large, inv(large))
      y = T[scale]
      @test Conjugate(IndBallL2(one(T)))(y) ≈ scale rtol = 8eps(T)
    end
    @test Conjugate(NormL2(large))(T[large / 2]) == zero(T)

    # Compare true inside/outside membership independently at ordinary scale
    # and near floatmax, where adding a relative tolerance used to overflow.
    m = floatmax(T)
    for radius in (one(T), m), fraction in (T(0.25), T(0.8))
      y = fill(fraction * radius, 3)
      exact_norm = sqrt(sum(abs2, BigFloat.(y)))
      expected = exact_norm <= BigFloat(radius) ? zero(T) : T(Inf)
      @test Conjugate(NormL2(radius))(y) == expected
    end
    @test Conjugate(IndBallL2(zero(T)))(fill(T(0.8) * m, 3)) == zero(T)

    # Dividing a squared norm by 2λ must not overflow or underflow its
    # intermediates. The last case has a norm outside T but finite energy.
    cases = ((large, T[large]), (inv(large), T[inv(large)]),
             (m, T[1]), (m, fill(T(0.7) * m, 3)))
    for (λ, y) in cases
      reference = T(sum(abs2, BigFloat.(y)) / (2BigFloat(λ)))
      @test isfinite(reference) && reference > zero(T)
      f = Conjugate(SqrNormL2(λ))
      @test f(y) ≈ reference rtol = 32eps(T) atol = 8nextfloat(zero(T))
      @test f(SparseIterate(y)) ≈ reference rtol = 32eps(T) atol = 8nextfloat(zero(T))
      @test allocs(f, y) == 0
    end
  end

  # Parameter precision may exceed output precision. Keep λ until scaling;
  # converting it to Float32 first would turn these finite energies into 0/Inf.
  for (λ, y) in ((1e100, Float32[1e30]), (1e-100, Float32[1e-40]))
    reference = Float32(sum(abs2, BigFloat.(y)) / (2BigFloat(λ)))
    @test Conjugate(SqrNormL2(λ))(y) ≈ reference rtol = 32eps(Float32) atol = 8nextfloat(0f0)
  end
end

@testset "SeparableSum" begin
  S, L = randn(5, 5), Matrix(Symmetric(randn(5, 5)))
  f = SeparableSum(NormL1(0.3), TracePSD(0.2))
  @test f((S, L)) == NormL1(0.3)(S) + TracePSD(0.2)(L)
  Z = prox(f, (S, L), 0.7)
  @test Z[1] ≈ prox(NormL1(0.3), S, 0.7) && Z[2] ≈ prox(TracePSD(0.2), L, 0.7)
  out = (similar(S), similar(L))
  @test prox!(out, f, (S, L), 0.7) === out
  @test value_and_prox!(out, f, (S, L), 0.7) ≈ f(Z)
  @test isproximable(f) && isconvex(f) && !issmooth(f)
  # smooth blocks
  X, y = randn(10, 4), randn(10)
  g = SeparableSum(SqrNormL2(2.), LeastSquaresLoss(y, X))
  a, b = randn(3), randn(4)
  G = (similar(a), similar(b))
  @test value_and_gradient!(G, g, (a, b)) ≈ g((a, b))
  @test G[1] ≈ 2a && G[2] ≈ gradient(LeastSquaresLoss(y, X), b)
  @test gradient!(G, g, (a, b)) === G
  @test_throws DimensionMismatch f((S,))
end

@testset "Sum" begin
  X, y = randn(20, 6), randn(20)
  f = Sum(LeastSquaresLoss(y, X), SqrNormL2(0.5), LinearFunction(randn(6)))
  x = randn(6)
  @test f(x) ≈ sum(g -> g(x), f.fs)
  check_gradient(f, x)
  @test issmooth(f) && isconvex(f) && !isproximable(f)
  @test Sum(SqrNormL2())(x) ≈ norm(x)^2 / 2
  g = similar(x)
  value_and_gradient!(g, Sum(SqrNormL2()), x)
  @test g ≈ x
end

@testset "wrapper parameter lengths before mutation" begin
  for T in (Float32, Float64)
    x = T[1, 2, 3]
    bad_tilt = Tilt(NormL1(one(T)), T[1])
    bad_translate = Translate(NormL1(one(T)), T[1])
    for f in (bad_tilt, bad_translate), step in (T(0.5), fill(T(0.5), 3))
      y = copy(x)
      @test_throws DimensionMismatch prox!(y, f, x, step)
      @test y == x
      @test_throws DimensionMismatch value_and_prox!(y, f, x, step)
      @test y == x
    end
    for op in (gradient!, value_and_gradient!)
      y = copy(x)
      @test_throws DimensionMismatch op(y, Tilt(SqrNormL2(one(T)), T[1]), x)
      @test y == x
    end
    # Equal entry counts cannot replace Translate's established shape contract.
    shifted = Translate(NormL1(one(T)), reshape(T[1, 2, 3], 1, 3))
    @test_throws DimensionMismatch prox(shifted, x)
    @test_throws DimensionMismatch prox(shifted, x, ones(T, 3))
  end
end

@testset "conjugate values at proximal points without a standalone value" begin
  for T in (Float32, Float64)
    x = T[3, 4]
    f = GroupNormL2(one(T), [1:2], T[1])
    c = Conjugate(f)
    y = similar(x)
    @test value_and_prox!(y, c, x, one(T)) ≈ zero(T) atol = 10eps(T)
    @test y ≈ x ./ 5
    @test_throws ArgumentError c(y) # The standalone value interface is unchanged.
    for step in (T(0.5), T[0.3, 0.8])
      # The double conjugate is f itself, although its standalone value is not implemented.
      original = NormL1(T(0.7))
      twice = Conjugate(Conjugate(original))
      expected = prox(original, x, step)
      for alias in (false, true)
        input = copy(x)
        output = alias ? input : similar(x)
        value = value_and_prox!(output, twice, input, step)
        @test output ≈ expected
        @test value ≈ original(expected)
      end
    end
    # Existing exact indicator values continue to use their closed form.
    @test value_and_prox!(y, Conjugate(NormL1(one(T))), x, one(T)) == zero(T)
    @test allocs(value_and_prox!, y, c, x, one(T)) == 0
  end
end

end
