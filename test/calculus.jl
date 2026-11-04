module CalculusTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

blocks_approx(x::AbstractArray, y::AbstractArray) = isapprox(x, y)
blocks_approx(x::Tuple, y::Tuple) = all(map(blocks_approx, x, y))
fresh(x::AbstractArray) = similar(x)
fresh(x::Tuple) = map(fresh, x)

struct CountedMatrix <: AbstractMatrix{Float64}
  data::Matrix{Float64}
  images::Base.RefValue{Int}
end
Base.size(A::CountedMatrix) = size(A.data)
Base.getindex(A::CountedMatrix, i::Int, j::Int) = A.data[i, j]
function LinearAlgebra.mul!(y::AbstractVector, A::CountedMatrix, x::AbstractVector)
  A.images[] += 1
  mul!(y, A.data, x)
end

@testset "zero preservation" begin
  table = (
    (ZeroFunction(), true), (SqrNormL2(), true), (SqrNormL2(Inf), false),
    (SqrNormL2(1., [1., 2.]), false),
    (Scaled(NormL1(1.), 2.), true), (Scaled(NormL1(1.), Inf), false),
    (Regularize(NormL1(1.), 1.), true), (Regularize(NormL1(1.), Inf), false),
    (Regularize(NormL1(1.), 1., zeros(2)), false),
    (Precompose(NormL1(1.), -2.), true), (Precompose(NormL1(1.), Inf), false),
    (Precompose(ElasticNet(0., 1.), 1e200), false),
    (Precompose(NormL1(1.), Matrix(1.0I, 2, 2), 1.), false),
    (Conjugate(NormL1(1.)), true), (Conjugate(NormL2(1.)), false),
    (MoreauEnvelope(NormL1(1.), 1.), false), (Tilt(NormL1(1.), ones(2)), false),
    (Translate(NormL1(1.), ones(2)), false), (LinearFunction(ones(2)), false),
    (NormL2(), false), (SeparableSum(SqrNormL2(), SqrNormL2()), false),
    (Sum(SqrNormL2(), SqrNormL2()), false),
    (Constrained(NormL1(1.), IndBox(-1., 1.)), true),
    (Constrained(NormL1(1.), IndBox(1., 2.)), false),
    (Constrained(NormL1(1.), IndHyperplane(1., 0.)), false))
  for (f, expected) in table
    @test iszeropreserving(f) === expected
    if expected
      ProximalBase.Testing.test_separable(f, [0., 2.], 0.7)
    end
  end
  @test !isdefined(ProximalBase, :_iszeropreserving)
end

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
  for T in (Float32, Float64)
    f = GroupNormL2(one(T), [1:2, 3:4])
    x, b, steps = ones(T, 4), zeros(T, 4), fill(T(0.5), 2)
    t = Translate(f, b)
    expected = fill(T(0.5), 4)
    @test prox(t, x, steps) ≈ expected
    out = similar(x)
    @test value_and_prox!(out, t, x, steps) ≈ f(expected)
    @test out ≈ expected
    b = T[1, -1, 2, -2]
    t = Translate(f, b)
    expected = b + prox(f, x - b, steps)
    @test prox(t, x, steps) ≈ expected
    @test value_and_prox!(out, t, x, steps) ≈ t(expected)
    @test out ≈ expected
  end
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

@testset "restrict where the dropped terms are not evaluated at zero" begin
  for T in (Float32, Float64)
    box = IndBox(T[1, 0], T[2, 2])
    # f(E_S z) up to the dropped constant
    function agrees(f, r, S, n, zs, c)
      all(zs) do z
        x = zeros(T, n)
        x[S] = z
        isinf(f(x)) ? isinf(r(z)) : isapprox(f(x) - r(z), c; atol = sqrt(eps(T)))
      end
    end
    t = Translate(box, T[-1.5, 0])
    @test agrees(t, restrict(t, [2]), [2], 2, (T[1], T[3], T[-1]), 0)
    @test_throws ArgumentError restrict(Translate(box, zeros(T, 2)), [2])
    t = Translate(IndBox(T[-1, 1], T[1, 2]), T[0, 1.5])
    @test agrees(t, restrict(t, [2]), [2], 2, (T[3], T[0]), 0)
    @test_throws ArgumentError restrict(t, [1])
    t = Translate(NegLog(one(T)), T[-1, -1])
    @test agrees(t, restrict(t, [2]), [2], 2, (T[1], T[-2]), 0)
    e = MoreauEnvelope(box, T(0.5))
    re = restrict(e, [2])
    @test agrees(e, re, [2], 2, (T[1], T[3]), 1)
    @test gradient(re, T[3]) ≈ gradient(e, T[0, 3])[[2]]
    d = SqrDistL2(box, T(2))
    @test agrees(d, restrict(d, [2]), [2], 2, (T[1], T[3]), 1)
    e = MoreauEnvelope(Translate(box, T[-1.5, 0]), T(0.5))
    @test agrees(e, restrict(e, [2]), [2], 2, (T[1], T[3]), 0)
    t = Translate(Scaled(box, T(2)), T[-1.5, 0])
    @test agrees(t, restrict(t, [2]), [2], 2, (T[1], T[3]), 0)
    t = Translate(Constrained(NormL1(one(T)), box), T[-1.5, 0])
    @test agrees(t, restrict(t, [2]), [2], 2, (T[1], T[3]), T(1.5))
    # forwarding wrappers under Translate, whose dropped coordinate is feasible
    for (g, b, c) in ((Tilt(box, T[1, 1]), T[-1.5, 0], T(1.5)), (Regularize(box, one(T)), T[-1.5, 0], T(1.125)),
                      (Precompose(box, T(2)), T[-0.75, 0], 0), (Precompose(box, T[2, 1]), T[-0.75, 0], 0))
      t = Translate(g, b)
      @test agrees(t, restrict(t, [2]), [2], 2, (T[1], T[3]), c)
    end
    @test_throws ArgumentError restrict(Translate(Precompose(box, T(2)), zeros(T, 2)), [2])
    # the zero-translation check of a nonseparable function still applies on
    # the unchecked path; an envelope of it requires a separable function
    coupled = Translate(GroupNormL2(one(T), [1:2], T[1]), T[0, 1])
    @test_throws ArgumentError ProximalBase._restrict_unchecked(coupled, [1])
    @test agrees(coupled, ProximalBase._restrict_unchecked(coupled, [2]), [2], 2, (T[2], T[-1]), 0)
    @test_throws ArgumentError("restrict of MoreauEnvelope requires a separable function") restrict(MoreauEnvelope(coupled, T(0.5)), [2])
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
  c = Conjugate(IndBallL2(0.4))
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
    z = Scaled(NormL1(one(T), T[1, Inf]), zero(T))
    @test z isa NormL1 && iszero(z.λ0)
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
    @test isapprox(prox(c, x, γ), x - γ * prox(f, x ./ γ, 1 / γ); atol = 8 * eps() * max(1, maximum(abs, x)) / min(γ, 1))
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

  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  x = ([1., 2.], [3.])
  expected = map(gradient, s.fs, x)
  @test blocks_approx(gradient(s, x), expected)
  value, g = value_and_gradient(s, x)
  @test value == s(x) == 11.5
  @test blocks_approx(g, expected)
  scaled = Scaled(s, 2)
  g = map(similar, x)
  @test gradient!(g, scaled, x) === g
  @test blocks_approx(g, map(yk -> 2 .* yk, expected))
  @test value_and_gradient!(g, scaled, x) == 23.
  @test blocks_approx(g, map(yk -> 2 .* yk, expected))
  @test blocks_approx(prox(scaled, x, 0.7), map((f, xk) -> prox(2f, xk, 0.7), s.fs, x))

  # Nested tuples and matrix wrappers retain the block structure.
  S = Symmetric([1. 2.; 2. 3.])
  H = Hermitian([1. 2.; 2. 3.])
  nested = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(2)))
  point = (x, (S, H))
  g = gradient(2nested, point)
  @test blocks_approx(g[1], map(yk -> 2 .* yk, expected))
  @test g[2][1] isa Matrix{Float64} && g[2][1] ≈ 2S
  @test g[2][2] isa Matrix{Float64} && g[2][2] ≈ 4H
  value, g = value_and_gradient(nested, point)
  @test value ≈ nested(point)
  @test g[2][1] ≈ S && g[2][2] ≈ 2H
  p = prox(2nested, point, 0.5)
  @test blocks_approx(p[1], prox(2s, x, 0.5))
  @test p[2][1] isa Matrix{Float64} && p[2][1] ≈ S / 2
  @test p[2][2] isa Matrix{Float64} && p[2][2] ≈ H / 3
end

@testset "tuple conjugates" begin
  x = ([1., -2.], [3.])
  known = SeparableSum(NormL1(1.), SqrNormL2())
  fallback = SeparableSum(ElasticNet(0.4, 0.7), SqrNormL2(2.))
  for f in (known, Scaled(known, 2.), fallback, Scaled(fallback, 2.)), step in (0.3, 2.)
    fs = f isa Scaled ? map(fk -> Scaled(fk, f.α), f.f.fs) : f.fs
    cs = map(Conjugate, fs)
    c = Conjugate(f)
    expected = map((ck, xk) -> prox(ck, xk, step), cs, x)
    value = sum(map((ck, xk) -> value_and_prox!(similar(xk), ck, xk, step), cs, x))
    out = map(similar, x)
    @test prox!(out, c, x, step) === out
    @test blocks_approx(out, expected)
    @test blocks_approx(prox(c, x, step), expected)
    @test value_and_prox!(out, c, x, step) ≈ value
    @test blocks_approx(out, expected)
    if f === known || (f isa Scaled && f.f === known)
      before = deepcopy(out)
      @test c(out) ≈ sum(map((ck, yk) -> ck(yk), cs, out))
      @test out == before
    else
      @test_throws ArgumentError c(out)
      # The elastic-net conjugate has this value, independently of Fenchel subtraction.
      α = f isa Scaled ? f.α : 1.
      reference = sum(abs2, max.(abs.(out[1]) .- α * 0.4, 0)) / (2α * 0.7) + sum(abs2, out[2]) / (4α)
      @test value ≈ reference
    end
    for op in (prox!, value_and_prox!)
      alias = map(copy, x)
      result = op(alias, c, alias, step)
      @test op === prox! ? result === alias : isapprox(result, value)
      @test blocks_approx(alias, expected)
    end
    primal = prox(f, x, step)
    dual = prox(c, map(xk -> xk ./ step, x), inv(step))
    @test blocks_approx(map((pk, dk) -> pk .+ step .* dk, primal, dual), x)
    cache = c.scratch.buf
    prox!(out, c, x, step)
    @test c.scratch.buf === cache
    @test ProximalBase.Testing.allocations(prox!, out, c, x, step) == 0
  end

  c = Conjugate(known)
  previous_cache = nothing
  for point in (x, ([1., 2., -3.], [4., 5.]), (Float32[1, -2], Float32[3]),
                (Float32[1, -2], [3.]), (view([1., -2.], :), [3.]))
    out = map(similar, point)
    expected = map((f, xk) -> prox(Conjugate(f), xk, 0.7), known.fs, point)
    @test prox!(out, c, point, 0.7) === out
    @test blocks_approx(out, expected)
    cache = c.scratch.buf
    @test cache !== previous_cache
    @test ProximalBase.Testing.allocations(prox!, out, c, point, 0.7) == 0
    @test c.scratch.buf === cache
    previous_cache = cache
  end

  nested = SeparableSum(known, SeparableSum(SqrNormL2(2.), NormL1(0.5)))
  point = (x, ([2.], [-2., 1.]))
  for f in (nested, Scaled(nested, 2.))
    α = f isa Scaled ? f.α : 1.
    c = Conjugate(f)
    expected = ((clamp.(x[1], -α, α), x[2] ./ (1 + 0.7 / α)),
                (point[2][1] ./ (1 + 0.7 / (2α)), clamp.(point[2][2], -0.5α, 0.5α)))
    out = prox(c, point, 0.7)
    @test blocks_approx(out, expected)
    @test value_and_prox!(out, c, point, 0.7) ≈ c(expected)
    @test blocks_approx(out, expected)
    alias = deepcopy(point)
    @test prox!(alias, c, alias, 0.7) === alias
    @test blocks_approx(alias, expected)
    @test ProximalBase.Testing.allocations(prox!, out, c, point, 0.7) == 0
  end

  f = SeparableSum(fallback, SqrNormL2())
  point = (x, [2., -3.])
  c = Conjugate(f)
  out = prox(c, point, 0.7)
  value = sum(abs2, max.(abs.(out[1][1]) .- 0.4, 0)) / 1.4 +
          sum(abs2, out[1][2]) / 4 + sum(abs2, out[2]) / 2
  @test value_and_prox!(out, c, point, 0.7) ≈ value
  alias = deepcopy(point)
  @test value_and_prox!(alias, c, alias, 0.7) ≈ value
  @test blocks_approx(alias, out)
  @test ProximalBase.Testing.allocations(prox!, out, c, point, 0.7) == 0

  # A mismatch in a later block leaves all earlier blocks untouched.
  bad = SeparableSum(SqrNormL2(), NormL1(1., ones(2)))
  for f in (bad, Scaled(bad, 2.), Conjugate(bad)), op in (prox!, value_and_prox!)
    out = map(copy, x)
    @test_throws DimensionMismatch op(out, f, x, 0.7)
    @test out == x
  end
  for f in (SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))),
            Scaled(SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))), 2)),
      op in (gradient!, value_and_gradient!)
    out = map(copy, x)
    @test_throws DimensionMismatch op(out, f, x)
    @test out == x
  end
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

@testset "Sum on tuples" begin
  # the reproduction of #159
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  f = Sum(s, s)
  x = ([1.0, 2.0], [3.0])
  g = map(similar, x)
  @test value_and_gradient!(g, f, x) ≈ 23.0
  @test g[1] ≈ [2.0, 4.0] && g[2] ≈ [12.0]
  @test gradient!(g, f, x) === g
  @test blocks_approx(gradient(f, x), g)
  value, G = value_and_gradient(f, x)
  @test value ≈ 23.0 && blocks_approx(G, g)
  @test blocks_approx(check_gradient(f, x), g)   # finite differences and all four interfaces

  # independent references: the gradient of Σⱼ Σₖ λⱼₖ‖xₖ - cⱼₖ‖²/2 is Σⱼ λⱼₖ(xₖ - cⱼₖ), and
  # that of ‖Xβ - y‖²/2 is X'(Xβ - y)
  rng = Xoshiro(7)
  X, y = randn(rng, 9, 3), randn(rng, 9)
  c = [0.5, -1.0]
  a = SeparableSum(LeastSquaresLoss(y, X), SqrNormL2(2.))
  b = SeparableSum(SqrNormL2(0.3), SqrNormL2(1.5, c))
  z = (randn(rng, 3), randn(rng, 2))
  expected = (X' * (X * z[1] - y) + 0.3 .* z[1], 2 .* z[2] + 1.5 .* (z[2] - c))
  for fs in ((a, b), (b, a)), summands in (fs, (fs[1],), (fs..., Scaled(a, 0.5)))
    h = Sum(summands...)
    ref = foldl((r, t) -> map(+, r, t), map(u -> u === a ? (X' * (X * z[1] - y), 2 .* z[2]) :
                u === b ? (0.3 .* z[1], 1.5 .* (z[2] - c)) :
                          (0.5 .* (X' * (X * z[1] - y)), 0.5 .* 2 .* z[2]), summands))
    @test blocks_approx(gradient(h, z), ref)
    @test value_and_gradient(h, z)[1] ≈ sum(u -> u(z), summands)
    check_gradient(h, z)
  end
  @test blocks_approx(gradient(Sum(a, b), z), expected)

  # blocks of different sizes, matrices and nested tuples; nested and repeated sums
  s3 = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3))
  x3 = ([1.0, 2.0], [3.0 4.0 5.0; 6.0 7.0 8.0], [9.0])
  @test blocks_approx(gradient(Sum(s3, s3), x3), map((xk, k) -> 2k .* xk, x3, (1, 2, 3)))
  @test blocks_approx(gradient(Sum(s3), x3), map((xk, k) -> k .* xk, x3, (1, 2, 3)))
  @test blocks_approx(gradient(Sum(Sum(s3, s3), s3), x3), map((xk, k) -> 3k .* xk, x3, (1, 2, 3)))
  nested = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(2)))
  xn = (x, ([4.0 5.0], [6.0, 7.0]))
  gn = gradient(Sum(nested, Scaled(nested, 3)), xn)
  @test gn[1][1] ≈ 4 .* xn[1][1] && gn[1][2] ≈ 8 .* xn[1][2]
  @test gn[2][1] ≈ 4 .* xn[2][1] && gn[2][2] ≈ 8 .* xn[2][2]
  @test check_gradient(Sum(nested, nested), xn) isa Tuple
  gs = gradient(Scaled(Sum(s, s), 2.5), x)
  @test blocks_approx(gs, map(xk -> 5 .* xk, ([1.0, 2.0], [6.0])))

  # Float32 variables, also with Float64 model data
  x32 = map(xk -> Float32.(xk), x)
  g32 = gradient(f, x32)
  @test g32 isa Tuple{Vector{Float32}, Vector{Float32}} && g32 == (Float32[2, 4], Float32[12])
  @test value_and_gradient(f, x32)[1] isa Float32
  @test f(x32) isa Float32
  @test check_gradient(f, x32) isa Tuple{Vector{Float32}, Vector{Float32}}
  m = Sum(a, b)
  z32 = map(zk -> Float32.(zk), z)
  gz = gradient(m, z32)
  @test gz isa Tuple{Vector{Float32}, Vector{Float32}}
  @test all(map((u, w) -> isapprox(u, w; rtol = 1e-4), gz, gradient(m, map(zk -> Float64.(zk), z32))))
  @test value_and_gradient(m, z32)[1] isa Float32

  # one scratch tuple, reused across calls and renewed when the shapes or types change
  h = Sum(s, s, Scaled(s, 2))
  out = map(similar, x)
  gradient!(out, h, x); buffer = h.scratch.buf
  gradient!(out, h, x)
  @test h.scratch.buf === buffer
  @test all(b -> all(o -> b !== o, out) && all(o -> b !== o, x), buffer[3])
  y2 = ([1.0, 2.0, 3.0], [4.0])
  out2 = map(similar, y2)
  @test blocks_approx(gradient!(out2, h, y2), (4 .* y2[1], 8 .* y2[2]))
  @test h.scratch.buf !== buffer
  @test blocks_approx(gradient!(out, h, x), (4 .* x[1], 8 .* x[2]))
  out32 = map(similar, x32)
  @test blocks_approx(gradient!(out32, h, x32), (4 .* x32[1], 8 .* x32[2]))
  @test out32 isa Tuple{Vector{Float32}, Vector{Float32}}
  @test blocks_approx(gradient!(out, h, x), (4 .* x[1], 8 .* x[2]))

  # many blocks and warm calls
  many = SeparableSum(ntuple(_ -> SqrNormL2(), 40)...)
  xm = ntuple(i -> fill(Float64(i), 2), 40)
  fm = Sum(many, many)
  gm = map(similar, xm)
  @test value_and_gradient!(gm, fm, xm) ≈ 2 * many(xm)
  @test blocks_approx(gm, map(xk -> 2 .* xk, xm))
  vg!(g, f, x) = value_and_gradient!(g, f, x)
  @test allocs(vg!, gm, fm, xm) == 0
  @test allocs(vg!, out, Sum(s, s, Scaled(s, 2)), x) == 0
  @test allocs(vg!, out, h, x) == 0
  @test allocs(vg!, fresh(xn), Sum(nested, nested), xn) == 0
  @inferred value_and_gradient!(out, h, x)
  @inferred value_and_gradient!(gm, fm, xm)
end

@testset "Sum on tuples checks before writing" begin
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  centered = SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2)))     # the center does not fit the second block
  x = ([1.0, 2.0], [3.0])
  sentinel = ([9.0, 8.0], [7.0])
  for op in (gradient!, value_and_gradient!)
    # an invalid final summand, however many valid ones precede it
    for f in (Sum(s, centered), Sum(s, s, centered), Sum(s, Sum(s, centered)), Sum(s, Scaled(centered, 2)))
      out = map(copy, sentinel)
      @test_throws DimensionMismatch op(out, f, x)
      @test out == sentinel
    end
    # an invalid final block of a long tuple, and a block of a nested tuple
    x3 = ([1.0, 2.0], [3.0], [4.0, 5.0, 6.0])
    s3 = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3))
    bad3 = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(1., ones(2)))
    out = map(x -> fill(-1.0, size(x)), x3)
    @test_throws DimensionMismatch op(out, Sum(s3, s3, bad3), x3)
    @test all(o -> all(==(-1.0), o), out)
    nested = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(2)))
    nested_bad = SeparableSum(s, SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))))
    xn = (x, ([4.0, 5.0], [6.0]))
    out = map(xk -> map(a -> fill(-1.0, size(a)), xk), xn)
    @test_throws DimensionMismatch op(out, Sum(nested, nested, nested_bad), xn)
    @test all(o -> all(a -> all(==(-1.0), a), o), out)
    # a centered SqrNormL2 whose parameter fits a different block
    wrongcenter = SeparableSum(SqrNormL2(1., [1.0, 2.0, 3.0]), SqrNormL2())
    out = map(copy, sentinel)
    @test_throws DimensionMismatch op(out, Sum(s, wrongcenter), x)
    @test out == sentinel
    # wrong arity and block shapes of the output
    for bad in ((similar(x[1]),), (similar(x[1]), similar(x[2]), similar(x[2])), (similar(x[1]), zeros(2)),
                (zeros(3), similar(x[2])))
      @test_throws DimensionMismatch op(bad, Sum(s, s), x)
    end
    out = map(copy, sentinel)
    @test_throws DimensionMismatch op(out, Sum(s, s), (x[1],))
    @test_throws DimensionMismatch op(out, Sum(s, s), (x[1], x[2], x[2]))
    @test out == sentinel
    # an output that refuses off-diagonal writes cannot take a sum, found before the first summand writes
    for wrap in (Symmetric, Hermitian)
      m = [1.0 2.0; 2.0 3.0]
      sm = SeparableSum(SqrNormL2(), SqrNormL2())
      wrongout = wrap(fill(-7.0, 2, 2)); before = copy(wrongout)
      @test_throws ArgumentError op(wrongout, Sum(SqrNormL2(), SqrNormL2(2.)), m)
      @test wrongout == before
      tupleout = (fill(-7.0, 2), wrap(fill(-7.0, 2, 2)))
      @test_throws ArgumentError op(tupleout, Sum(sm, sm), ([1.0, 2.0], m))
      @test all(==(-7.0), tupleout[1]) && tupleout[2] == before
    end
    # the output must not be the point, also one block of it
    @test_throws ArgumentError op(x, Sum(s, s), x)
    out = (similar(x[1]), x[2])
    @test_throws ArgumentError op(out, Sum(s, s), x)
    @test out[2] === x[2] && x[2] == [3.0]
    xa = (x, ([4.0, 5.0], [6.0]))
    out = (map(similar, x), xa[2])
    @test_throws ArgumentError op(out, Sum(nested, nested), xa)
  end
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
    inner = NormL1(one(T), ones(T, 2))
    for f in (Translate(inner, ones(T, 3)), Regularize(inner, one(T)),
              Regularize(inner, one(T), ones(T, 3)), Conjugate(inner),
              Precompose(inner, T(-2)), MoreauEnvelope(inner, one(T))),
        op in (prox!, value_and_prox!), alias in (false, true)
      for step in (T(0.5), fill(T(0.5), 3))
        step isa AbstractArray && f isa Union{Regularize, Precompose, MoreauEnvelope} && continue
        input = copy(x)
        out = alias ? input : copy(x)
        @test_throws DimensionMismatch op(out, f, input, step)
        @test out == x
        @test input == x
      end
    end
    for h in (Precompose(inner, T(-2)), Precompose(inner, Matrix{T}(I, 3, 3), one(T))),
        op in (prox!, value_and_prox!), alias in (false, true)
      input = copy(x)
      out = alias ? input : copy(x)
      @test_throws DimensionMismatch op(out, Scaled(h, T(2)), input, T(0.5))
      @test out == x
      @test input == x
    end
    r = Regularize(SqrNormL2(one(T)), one(T), ones(T, 2))
    for op in (gradient!, value_and_gradient!)
      out = copy(x)
      @test_throws DimensionMismatch op(out, r, x)
      @test out == x
    end
  end
  A = CountedMatrix(Matrix{Float64}(I, 3, 3), Ref(0))
  h = Precompose(SqrNormL2(), A, 1.; check = false)
  x = fill(3., 3)
  out = similar(x)
  for op in (prox!, value_and_prox!)
    A.images[] = 0
    op(out, Scaled(h, 2.), x, 0.5)
    @test A.images[] == 1
    @test out ≈ prox(Scaled(SqrNormL2(), 2.), x, 0.5)
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


# Tilt and Translate on tuples (#159): independent references are the blockwise formulas
# evaluated with the functions of the blocks on arrays, never the tuple methods under test.
tdot(a::AbstractArray, b::AbstractArray) = dot(a, b)
tdot(a::Tuple, b::Tuple) = sum(map(tdot, a, b))
map_blocks(f, a::AbstractArray, b::AbstractArray) = f(a, b)
map_blocks(f, a::Tuple, b::Tuple) = map((u, v) -> map_blocks(f, u, v), a, b)
copy_blocks(a::AbstractArray) = copy(a)
copy_blocks(a::Tuple) = map(copy_blocks, a)
# a separable sum of F() with the structure of x
tuple_sum(x::Tuple, F) = SeparableSum(map(xk -> xk isa Tuple ? tuple_sum(xk, F) : F(), x)...)

@testset "Tilt and Translate on tuples: constructors" begin
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  c = ([1.0, 1.0], [1.0])
  b = ([0.5, -0.5], [1.0])
  t, u = Tilt(s, c, 1.5), Translate(s, b)
  @test t.c === c && u.b === b                       # parameters are held, not copied
  @test Tilt(s, c).d === 0.0 && Tilt(s, c).d isa Float64
  @test Tilt(s, map(a -> Float32.(a), c)).d === 0.0f0     # the default constant follows the coefficients
  @test Tilt(s, (Float32[1], Float64[2])).d === 0.0
  @test Tilt(s, ([1, 2], [3])).d === 0.0
  @test Tilt(s, c, 2).d === 2
  @test Tilt(s, (c[1],)).c == (c[1],)                 # one block
  nested = ((c[1], c[2]), ([1.0 2.0; 3.0 4.0], [1.0]))
  @test Tilt(s, nested).c === nested && Translate(s, nested).b === nested
  @test Translate(s, ([1.0 2.0; 3.0 4.0], [1.0, 2.0])).b[1] isa Matrix{Float64}
  for T in (Tilt, Translate)
    @test_throws ArgumentError T(s, ([1.0 + 1.0im], [1.0]))        # nonreal leaves
    @test_throws ArgumentError T(s, ([1.0], (1.0im,)))
    @test_throws ArgumentError T(s, (1.0, [1.0]))                  # scalar leaves
    @test_throws ArgumentError T(s, ([1.0], (2.0,)))
    @test_throws ArgumentError T(s, ())                            # no blocks
    @test_throws MethodError T(s, (a = [1.0], b = [1.0]))          # named tuples
  end
  # the shapes are known when an oracle is called: the second parameter has the entries of the
  # matrix block but not its shape
  x = ([1.0, 2.0], [3.0 4.0; 5.0 6.0])
  ok, bad = ([0.1, 0.2], [0.3 0.4; 0.5 0.6]), ([0.1, 0.2], [0.3, 0.4, 0.5, 0.6])
  @test Tilt(s, ok)(x) isa Float64 && Translate(s, ok)(x) isa Float64
  @test_throws DimensionMismatch Tilt(s, bad)(x)
  @test_throws DimensionMismatch Translate(s, bad)(x)
  @test_throws DimensionMismatch Tilt(s, ([0.1, 0.2],))(x)         # arity
  @test_throws DimensionMismatch Translate(s, (ok..., [1.0]))(x)
  @test_throws DimensionMismatch Tilt(s, [1.0, 2.0])(x)            # an array for a tuple
  @test_throws DimensionMismatch Translate(s, (ok[1], (ok[2],)))(x)    # nesting
  # arrays are unchanged
  @test Tilt(NormL1(), [1.0, 2.0]).d === 0.0 && Tilt(NormL1(), Float32[1, 2]).d === 0.0f0
  @test_throws ArgumentError Tilt(NormL1(), [1.0im])
  @test_throws ArgumentError Translate(NormL1(), [1.0im])
end

@testset "Tilt on tuples" begin
  s = SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0))
  x = ([2.0, -1.0], [3.0, -2.0])
  c = ([0.25, -0.5], [1.0, 0.0])
  d = 1.25
  t = Tilt(s, c, d)
  @test issmooth(t) && isproximable(t)
  @test t(x) ≈ s(x) + dot(c[1], x[1]) + dot(c[2], x[2]) + d
  refg = (1.5 .* x[1] .+ c[1], 2.0 .* x[2] .+ c[2])
  g = map(similar, x)
  @test gradient!(g, t, x) === g
  @test blocks_approx(g, refg)
  g2 = map(similar, x)
  @test value_and_gradient!(g2, t, x) ≈ t(x)
  @test blocks_approx(g2, refg)
  @test blocks_approx(gradient(t, x), refg)
  v, G = value_and_gradient(t, x)
  @test v ≈ t(x) && blocks_approx(G, refg)
  @test blocks_approx(check_gradient(t, x), refg)    # Testing.test_gradient on a homogeneous tuple

  # proximal operators against the blockwise formula prox_{γf}(x - γc)
  f = SeparableSum(SqrNormL2(1.5), NormL1(0.5))
  γ = 0.4
  ref = (prox(f.fs[1], x[1] .- γ .* c[1], γ), prox(f.fs[2], x[2] .- γ .* c[2], γ))
  tp = Tilt(f, c, d)
  @test isproximable(tp) && !issmooth(tp)
  y = map(similar, x)
  @test prox!(y, tp, x, γ) === y
  @test blocks_approx(y, ref)
  @test blocks_approx(prox(tp, x, γ), ref)
  y = map(similar, x)
  v = value_and_prox!(y, tp, x, γ)
  @test blocks_approx(y, ref)
  @test v ≈ f(ref) + dot(c[1], ref[1]) + dot(c[2], ref[2]) + d    # the value at the stored output
  # in place
  xi = map(copy, x)
  @test prox!(xi, tp, xi, γ) === xi
  @test blocks_approx(xi, ref)
  xi = map(copy, x)
  @test value_and_prox!(xi, tp, xi, γ) ≈ v
  @test blocks_approx(xi, ref)
  @test x == ([2.0, -1.0], [3.0, -2.0])              # the input of the nonaliased calls is untouched
  # the step: zero and infinite steps where the function supports them
  @test blocks_approx(prox(Tilt(SeparableSum(NormL1(1.0), NormL1(1.0)), c, 0.0), x, 0.0), x)

  # matrices, nested tuples of unequal shapes, vectors and one block
  fm = SeparableSum(NormL1(0.3), SqrNormL2(2.0))
  xm = ([1.0, -2.0, 3.0], [1.0 -2.0; 0.5 4.0])
  cm = ([0.1, 0.2, -0.3], [0.5 -0.5; 1.0 0.0])
  refm = (prox(fm.fs[1], xm[1] .- γ .* cm[1], γ), prox(fm.fs[2], xm[2] .- γ .* cm[2], γ))
  @test blocks_approx(prox(Tilt(fm, cm), xm, γ), refm)
  nf = SeparableSum(SeparableSum(SqrNormL2(), NormL1(0.2)), SeparableSum(SqrNormL2(3), NormL1(0.1)))
  xn = (([1.0, 2.0], [3.0]), ([4.0 5.0; 6.0 7.0], [8.0, -9.0]))
  cn = (([0.5, 0.5], [1.0]), ([1.0 0.0; 0.0 1.0], [0.1, -0.1]))
  refn = ((prox(SqrNormL2(), xn[1][1] .- γ .* cn[1][1], γ), prox(NormL1(0.2), xn[1][2] .- γ .* cn[1][2], γ)),
          (prox(SqrNormL2(3), xn[2][1] .- γ .* cn[2][1], γ), prox(NormL1(0.1), xn[2][2] .- γ .* cn[2][2], γ)))
  tn = Tilt(nf, cn, 0.5)
  yn = fresh(xn)
  @test prox!(yn, tn, xn, γ) === yn
  @test blocks_approx(yn, refn)
  vn = value_and_prox!(yn, tn, xn, γ)
  @test vn ≈ nf(refn) + tdot(cn, refn) + 0.5
  @test blocks_approx(prox(Tilt(SeparableSum(SqrNormL2(2.0)), (c[1],)), (x[1],), γ), (prox(SqrNormL2(2.0), x[1] .- γ .* c[1], γ),))
  tg = Tilt(SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2)), SeparableSum(SqrNormL2(3), SqrNormL2(4))), cn, 0.5)
  @test blocks_approx(check_gradient(tg, xn), ((xn[1][1] .+ cn[1][1], 2 .* xn[1][2] .+ cn[1][2]), (3 .* xn[2][1] .+ cn[2][1], 4 .* xn[2][2] .+ cn[2][2])))

  # Float32 variables with Float64 data, and Float64 outputs for Float32 arguments: block by
  # block the same rounding as the array wrappers
  x32 = map(a -> Float32.(a), x)
  f32 = SeparableSum(SqrNormL2(1.5), NormL1(0.5))
  smooth32 = SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0))
  for cdata in (c, map(a -> Float32.(a), c))
    t32 = Tilt(f32, cdata, d)
    @test t32(x32) isa Float32
    g32 = map(similar, x32)
    @test value_and_gradient!(g32, Tilt(smooth32, cdata, d), x32) isa Float32
    @test g32 isa Tuple{Vector{Float32}, Vector{Float32}}
    @test blocks_approx(g32, map((k, xk, ck) -> k .* xk .+ ck, (1.5, 2.0), x32, cdata))
    y32 = map(similar, x32)
    @test prox!(y32, t32, x32, γ) === y32
    @test y32 isa Tuple{Vector{Float32}, Vector{Float32}}
    arrayref = map((fk, ck, xk) -> prox(Tilt(fk, ck, d), xk, γ), f32.fs, cdata, x32)
    @test y32 == arrayref
    y32 = map(similar, x32)
    @test value_and_prox!(y32, t32, x32, γ) isa Float32
    @test y32 == arrayref
    y64 = map(similar, x)
    @test prox!(y64, t32, x32, γ) === y64
    @test y64 isa Tuple{Vector{Float64}, Vector{Float64}}
    @test y64 == map((fk, ck, xk, yk) -> prox!(similar(yk), Tilt(fk, ck, d), xk, γ), f32.fs, cdata, x32, y64)
  end
  @test check_gradient(Tilt(s, map(a -> Float32.(a), c), 1f0), x32) isa Tuple{Vector{Float32}, Vector{Float32}}
end

@testset "Translate on tuples" begin
  s = SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0))
  x = ([2.0, -1.0], [3.0, -2.0])
  b = ([0.3, -0.1], [0.5, 1.5])
  u = Translate(s, b)
  shifted = (x[1] .- b[1], x[2] .- b[2])
  refg = (1.5 .* shifted[1], 2.0 .* shifted[2])
  @test u(x) ≈ s(shifted)
  g = map(similar, x)
  @test gradient!(g, u, x) === g
  @test blocks_approx(g, refg)
  g2 = map(similar, x)
  @test value_and_gradient!(g2, u, x) ≈ u(x)
  @test blocks_approx(g2, refg)
  @test blocks_approx(gradient(u, x), refg)
  @test blocks_approx(check_gradient(u, x), refg)

  f = SeparableSum(SqrNormL2(1.5), NormL1(0.5))
  γ = 0.4
  ref = (b[1] .+ prox(f.fs[1], x[1] .- b[1], γ), b[2] .+ prox(f.fs[2], x[2] .- b[2], γ))
  up = Translate(f, b)
  y = map(similar, x)
  @test prox!(y, up, x, γ) === y
  @test blocks_approx(y, ref)
  @test blocks_approx(prox(up, x, γ), ref)
  y = map(similar, x)
  v = value_and_prox!(y, up, x, γ)
  @test blocks_approx(y, ref)
  @test v ≈ f((y[1] .- b[1], y[2] .- b[2]))
  xi = map(copy, x)
  @test prox!(xi, up, xi, γ) === xi
  @test blocks_approx(xi, ref)
  xi = map(copy, x)
  @test value_and_prox!(xi, up, xi, γ) ≈ v
  @test blocks_approx(xi, ref)
  @test x == ([2.0, -1.0], [3.0, -2.0])

  # matrices and nested tuples of unequal shapes
  nf = SeparableSum(SeparableSum(SqrNormL2(), NormL1(0.2)), SeparableSum(SqrNormL2(3), NormL1(0.1)))
  xn = (([1.0, 2.0], [3.0]), ([4.0 5.0; 6.0 7.0], [8.0, -9.0]))
  bn = (([0.5, 0.5], [1.0]), ([1.0 0.0; 0.0 1.0], [0.1, -0.1]))
  refn = ((bn[1][1] .+ prox(SqrNormL2(), xn[1][1] .- bn[1][1], γ), bn[1][2] .+ prox(NormL1(0.2), xn[1][2] .- bn[1][2], γ)),
          (bn[2][1] .+ prox(SqrNormL2(3), xn[2][1] .- bn[2][1], γ), bn[2][2] .+ prox(NormL1(0.1), xn[2][2] .- bn[2][2], γ)))
  un = Translate(nf, bn)
  yn = fresh(xn)
  @test prox!(yn, un, xn, γ) === yn
  @test blocks_approx(yn, refn)
  @test value_and_prox!(yn, un, xn, γ) ≈ nf(map_blocks((a, b) -> a .- b, yn, bn))
  xi = copy_blocks(xn)
  @test prox!(xi, un, xi, γ) === xi
  @test blocks_approx(xi, refn)
  sm = SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2)), SeparableSum(SqrNormL2(3), SqrNormL2(4)))
  @test check_gradient(Translate(sm, bn), xn) isa Tuple

  # Float32 variables with Float64 data, and Float64 outputs for Float32 arguments
  x32 = map(a -> Float32.(a), x)
  f32 = SeparableSum(SqrNormL2(1.5), NormL1(0.5))
  smooth32 = SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0))
  for bdata in (b, map(a -> Float32.(a), b))
    u32 = Translate(f32, bdata)
    @test u32(x32) isa Float32
    g32 = map(similar, x32)
    @test value_and_gradient!(g32, Translate(smooth32, bdata), x32) isa Float32
    @test g32 isa Tuple{Vector{Float32}, Vector{Float32}}
    @test blocks_approx(g32, map((k, xk, bk) -> k .* (xk .- bk), (1.5, 2.0), x32, bdata))
    y32 = map(similar, x32)
    prox!(y32, u32, x32, γ)
    arrayref = map((fk, bk, xk) -> prox(Translate(fk, bk), xk, γ), f32.fs, bdata, x32)
    @test y32 isa Tuple{Vector{Float32}, Vector{Float32}} && y32 == arrayref
    y32 = map(similar, x32)
    @test value_and_prox!(y32, u32, x32, γ) isa Float32
    @test y32 == arrayref
    y64 = map(similar, x)
    prox!(y64, u32, x32, γ)
    @test y64 isa Tuple{Vector{Float64}, Vector{Float64}}
    @test y64 == map((fk, bk, xk, yk) -> prox!(similar(yk), Translate(fk, bk), xk, γ), f32.fs, bdata, x32, y64)
  end
end

@testset "Tilt and Translate on tuples: many blocks and compositions" begin
  γ = 0.3
  for n in (1, 2, 4, 32, 40)
    f = SeparableSum(ntuple(k -> isodd(k) ? SqrNormL2(k) : NormL1(0.1k), n)...)
    x = ntuple(k -> Float64.(collect(1:(k % 3 + 1))) .- 1.5, n)
    c = ntuple(k -> fill(0.1k, k % 3 + 1), n)
    ref = ntuple(k -> prox(f.fs[k], x[k] .- γ .* c[k], γ), n)
    @test blocks_approx(prox(Tilt(f, c, 1.0), x, γ), ref)
    refu = ntuple(k -> c[k] .+ prox(f.fs[k], x[k] .- c[k], γ), n)
    @test blocks_approx(prox(Translate(f, c), x, γ), refu)
    h = Tilt(SeparableSum(ntuple(k -> SqrNormL2(k), n)...), c, 1.0)
    @test blocks_approx(gradient(h, x), ntuple(k -> k .* x[k] .+ c[k], n))
    @test value_and_gradient(h, x)[1] ≈ h(x)
    hu = Translate(SeparableSum(ntuple(k -> SqrNormL2(k), n)...), c)
    @test blocks_approx(gradient(hu, x), ntuple(k -> k .* (x[k] .- c[k]), n))
  end
  # a deeply nested tuple
  deep(k) = k == 0 ? SqrNormL2(2) : SeparableSum(deep(k - 1), SqrNormL2(k))
  deepx(k) = k == 0 ? [1.0, 2.0] : (deepx(k - 1), fill(Float64(k), 2))
  deepc(k) = k == 0 ? [0.5, -0.5] : (deepc(k - 1), fill(0.1k, 2))
  df, dx, dc = deep(6), deepx(6), deepc(6)
  gt = gradient(Tilt(df, dc, 0.0), dx)
  gu = gradient(Translate(df, dc), dx)
  # the innermost block, then the outer blocks, with weight 2 and then k
  @test gt[1][1][1][1][1][1] ≈ 2 .* [1.0, 2.0] .+ [0.5, -0.5]
  @test gu[1][1][1][1][1][1] ≈ 2 .* ([1.0, 2.0] .- [0.5, -0.5])
  @test gt[2] ≈ 6 .* fill(6.0, 2) .+ fill(0.6, 2) && gu[2] ≈ 6 .* (fill(6.0, 2) .- fill(0.6, 2))
  @test Tilt(df, dc, 0.0)(dx) ≈ df(dx) + tdot(dc, dx)

  # wrappers around the new functions and the new functions around wrappers
  s = SeparableSum(SqrNormL2(1.5), NormL1(0.5))
  x = ([2.0, -1.0], [3.0, -2.0]); c = ([0.25, -0.5], [1.0, 0.0]); b = ([0.3, -0.1], [0.5, 1.5]); d = 0.75; α = 2.5
  tb, ub = Tilt(s, c, d), Translate(s, b)
  ref_t = (prox(s.fs[1], x[1] .- γ .* α .* c[1], α * γ), prox(s.fs[2], x[2] .- γ .* α .* c[2], α * γ))
  # prox of α(f + ⟨c, ·⟩ + d) at x is prox of αf at x - γαc
  @test blocks_approx(prox(Scaled(tb, α), x, γ), ref_t)
  @test blocks_approx(prox(Scaled(ub, α), x, γ), (b[1] .+ prox(s.fs[1], x[1] .- b[1], α * γ), b[2] .+ prox(s.fs[2], x[2] .- b[2], α * γ)))
  @test blocks_approx(prox(Tilt(Scaled(s, α), c, d), x, γ), (prox(s.fs[1], x[1] .- γ .* c[1], α * γ), prox(s.fs[2], x[2] .- γ .* c[2], α * γ)))
  @test blocks_approx(prox(Translate(Scaled(s, α), b), x, γ), (b[1] .+ prox(s.fs[1], x[1] .- b[1], α * γ), b[2] .+ prox(s.fs[2], x[2] .- b[2], α * γ)))
  # Tilt(Translate) and Translate(Tilt): the inner shift happens first in each
  tu = Tilt(Translate(s, b), c, d)
  zz = map((xk, ck) -> xk .- γ .* ck, x, c)
  @test blocks_approx(prox(tu, x, γ), map((zk, bk, fk) -> bk .+ prox(fk, zk .- bk, γ), zz, b, s.fs))
  ut = Translate(Tilt(s, c, d), b)
  zb = map(-, x, b)
  @test blocks_approx(prox(ut, x, γ), map((zk, bk, ck, fk) -> bk .+ prox(fk, zk .- γ .* ck, γ), zb, b, c, s.fs))
  sm = SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0))
  @test value_and_gradient(Scaled(Tilt(sm, c, d), α), x)[1] ≈ α * (sm(x) + tdot(c, x) + d)
  @test blocks_approx(gradient(Scaled(Translate(sm, b), α), x), map((xk, bk, k) -> α * k .* (xk .- bk), x, b, (1.5, 2.0)))
  @test blocks_approx(gradient(Tilt(Translate(sm, b), c, d), x), map((xk, bk, ck, k) -> k .* (xk .- bk) .+ ck, x, b, c, (1.5, 2.0)))
  @test blocks_approx(gradient(Translate(Tilt(sm, c, d), b), x), map((xk, bk, ck, k) -> k .* (xk .- bk) .+ ck, x, b, c, (1.5, 2.0)))
  @test check_gradient(Tilt(Translate(sm, b), c, d), x) isa Tuple
  @test check_gradient(Translate(Scaled(Tilt(sm, c, d), α), b), x) isa Tuple
  # with the tuple Sum of the earlier phase
  m = Sum(sm, Scaled(sm, 2))
  @test blocks_approx(gradient(Tilt(m, c, d), x), map((xk, ck, k) -> 3k .* xk .+ ck, x, c, (1.5, 2.0)))
  @test blocks_approx(gradient(Translate(m, b), x), map((xk, bk, k) -> 3k .* (xk .- bk), x, b, (1.5, 2.0)))
  @test check_gradient(Translate(Tilt(m, c, d), b), x) isa Tuple
end

@testset "Tilt and Translate on tuples check before writing" begin
  x = ([1.0, 2.0], [3.0 4.0; 5.0 6.0])
  sentinel = ([9.0, 8.0], [7.0 7.0; 7.0 7.0])
  ok = ([0.1, 0.2], [0.3 0.4; 0.5 0.6])
  bad = ([0.1, 0.2], [0.3, 0.4, 0.5, 0.6])          # the entries of the matrix block, not its shape
  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  l1 = SeparableSum(NormL1(0.3), NormL1(0.2))
  wrappers = (Tilt => p -> Tilt(s, p, 0.0), Translate => p -> Translate(s, p))
  proxwrappers = (Tilt => p -> Tilt(l1, p, 0.0), Translate => p -> Translate(l1, p))
  fresh_sentinel() = map(copy, sentinel)
  for (T, mk) in wrappers, op in (gradient!, value_and_gradient!)
    for param in (bad, (ok[1],), (ok..., [1.0]), (ok[1], (ok[2],)), ([0.1, 0.2, 0.3], ok[2]), (ok[1], zeros(2, 3)))
      out = fresh_sentinel()
      @test_throws DimensionMismatch op(out, mk(param), x)
      @test out == sentinel
    end
    # the output: arity and block shapes
    for badout in ((sentinel[1],), (sentinel[1], sentinel[2], sentinel[2]), (sentinel[1], zeros(4)), (zeros(3), sentinel[2]))
      out = map(copy, badout)
      @test_throws DimensionMismatch op(out, mk(ok), x)
      @test out == map(copy, badout)
    end
    # a later block of the inner function
    later = T === Tilt ? Tilt(SeparableSum(SqrNormL2(), SqrNormL2(1.0, ones(3))), ok, 0.0) :
                         Translate(SeparableSum(SqrNormL2(), SqrNormL2(1.0, ones(3))), ok)
    out = fresh_sentinel()
    @test_throws DimensionMismatch op(out, later, x)
    @test out == sentinel
    nestedbad = T === Tilt ? Tilt(SeparableSum(SqrNormL2(), Scaled(SeparableSum(SqrNormL2()), 2.0)), ok, 0.0) :
                             Translate(SeparableSum(SqrNormL2(), Scaled(SeparableSum(SqrNormL2()), 2.0)), ok)
    out = fresh_sentinel()
    @test_throws MethodError op(out, nestedbad, x)
    @test out == sentinel
    # the gradient must not overwrite the point or one of its blocks
    xa = map(copy, x)
    @test_throws ArgumentError op(xa, mk(ok), xa)
    @test xa == x
    out = (similar(x[1]), x[2])
    @test_throws ArgumentError op(out, mk(ok), x)
    @test x[2] == [3.0 4.0; 5.0 6.0]
    # outputs that refuse off-diagonal writes
    for wrap in (Symmetric, Hermitian)
      out = (fill(-7.0, 2), wrap(fill(-7.0, 2, 2)))
      before = copy(out[2])
      @test_throws ArgumentError op(out, mk(ok), x)
      @test all(==(-7.0), out[1]) && out[2] == before
    end
  end
  for (T, mk) in proxwrappers, op in (prox!, value_and_prox!), aliased in (false, true)
    for param in (bad, (ok[1],), (ok..., [1.0]), (ok[1], (ok[2],)), (ok[1], zeros(2, 3)))
      out = fresh_sentinel()
      xin = aliased ? out : x
      @test_throws DimensionMismatch op(out, mk(param), xin, 0.5)
      @test out == sentinel
    end
    for badout in ((sentinel[1],), (sentinel[1], sentinel[2], sentinel[2]), (sentinel[1], zeros(4)), (zeros(3), sentinel[2]))
      out = map(copy, badout)
      @test_throws DimensionMismatch op(out, mk(ok), x, 0.5)
      @test out == map(copy, badout)
    end
    # an invalid parameter of the inner function in a later block
    badinner = SeparableSum(NormL1(0.3), NormL1(0.2, [1.0, 2.0, 3.0]))
    wrapped = T === Tilt ? Tilt(badinner, ok, 0.0) : Translate(badinner, ok)
    out = fresh_sentinel()
    xin = aliased ? out : x
    @test_throws DimensionMismatch op(out, wrapped, xin, 0.5)
    @test out == sentinel
    for wrap in (Symmetric, Hermitian)
      out = (fill(-7.0, 2), wrap(fill(-7.0, 2, 2)))
      before = copy(out[2])
      @test_throws ArgumentError op(out, mk(ok), x, 0.5)
      @test all(==(-7.0), out[1]) && out[2] == before
    end
  end
  # steps that a later block rejects, before anything is written
  rej = SeparableSum(NormL1(0.3), IndBallL2(1.0))
  for T in (Tilt, Translate), op in (prox!, value_and_prox!), aliased in (false, true), γ in (-1.0, NaN)
    wrapped = T === Tilt ? Tilt(rej, ok, 0.0) : Translate(rej, ok)
    out = fresh_sentinel()
    xin = aliased ? out : x
    @test_throws ArgumentError op(out, wrapped, xin, γ)
    @test out == sentinel
    # also inside a nested wrapper
    out = fresh_sentinel()
    xin = aliased ? out : x
    @test_throws ArgumentError op(out, Scaled(wrapped, 2.0), xin, γ)
    @test out == sentinel
  end
  # zero and infinite steps where the function takes them
  box = SeparableSum(IndBox(-1.0, 1.0), IndBox(-1.0, 1.0))
  for γ in (0.0, Inf)
    y = fresh(x)
    @test prox!(y, Tilt(box, ok, 0.0), x, γ) === y
    @test blocks_approx(y, map((xk, ck) -> prox(IndBox(-1.0, 1.0), xk .- γ .* ck, γ), x, ok))
    y = fresh(x)
    @test prox!(y, Translate(box, ok), x, γ) === y
    @test blocks_approx(y, map((xk, bk) -> bk .+ prox(IndBox(-1.0, 1.0), xk .- bk, γ), x, ok))
  end
  # a nested function block with a step that its inner block rejects
  nested = SeparableSum(SeparableSum(NormL1(0.3), NormL1(0.3)), rej)
  nx = (x, x)
  nok = (ok, ok)
  nout = (map(copy, sentinel), map(copy, sentinel))
  @test_throws ArgumentError prox!(nout, Translate(nested, nok), nx, -1.0)
  @test nout == (sentinel, sentinel)
  @test_throws ArgumentError prox!(nout, Tilt(nested, nok, 0.0), nx, NaN)
  @test nout == (sentinel, sentinel)
  # a tuple block of a separable sum is checked too
  blk = SeparableSum(Tilt(rej, ok, 0.0), SqrNormL2())
  bout = (map(copy, sentinel), [1.0])
  @test_throws ArgumentError prox!(bout, blk, (x, [1.0]), -1.0)
  @test bout == (sentinel, [1.0])
end

@testset "Tilt and Translate on tuples: buffers and allocations" begin
  s = SeparableSum(SqrNormL2(1.5), NormL1(0.5))
  smooth = SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0))
  x = ([2.0, -1.0], [3.0, -2.0]); c = ([0.25, -0.5], [1.0, 0.0]); γ = 0.4
  t, u = Tilt(s, c, 1.0), Translate(s, c)
  y = fresh(x)
  prox!(y, t, x, γ)
  buffer = t.scratch.buf
  @test buffer !== nothing
  prox!(y, t, x, γ)
  @test t.scratch.buf === buffer                                  # the same shapes and types reuse it
  @test all(bk -> all(a -> bk !== a, y) && all(a -> bk !== a, x) && all(a -> bk !== a, c), buffer[3])
  y2 = ([1.0, 2.0, 3.0], [4.0])
  c2 = ([0.1, 0.2, 0.3], [0.4])
  t2 = Tilt(s, c2, 1.0)
  out2 = fresh(y2)
  prox!(out2, t2, y2, γ)                                          # another tuple with its own wrapper
  @test blocks_approx(out2, map((fk, ck, xk) -> prox(fk, xk .- γ .* ck, γ), s.fs, c2, y2))
  yy = fresh(y2)
  @test_throws DimensionMismatch prox!(yy, t, y2, γ)               # c does not fit y2
  y32 = map(a -> zeros(Float32, size(a)), x)
  prox!(y32, t, x, γ)
  @test t.scratch.buf !== buffer && t.scratch.buf[3] isa Tuple{Vector{Float32}, Vector{Float32}}   # another output type
  yback = fresh(x)
  @test blocks_approx(prox!(yback, t, x, γ), map((fk, ck, xk) -> prox(fk, xk .- γ .* ck, γ), s.fs, c, x))   # back to the original
  # no stale data: the result does not depend on what an earlier call left in the buffer
  @test blocks_approx(prox!(fresh(x), t, map(a -> 2 .* a, x), γ), map((fk, ck, xk) -> prox(fk, 2 .* xk .- γ .* ck, γ), s.fs, c, x))
  @test blocks_approx(prox!(fresh(x), t, x, γ), map((fk, ck, xk) -> prox(fk, xk .- γ .* ck, γ), s.fs, c, x))
  # Translate keeps its two buffers apart: values and gradients in the argument's precision, the
  # proximal operator in the output's
  g = fresh(x)
  ug = Translate(smooth, c)
  @test blocks_approx(gradient!(g, ug, x), map((k, xk, bk) -> k .* (xk .- bk), (1.5, 2.0), x, c))
  vbuf = ug.scratch.buf
  @test vbuf !== nothing && ug.proxscratch.buf === nothing
  prox!(y, u, x, γ)
  pbuf = u.proxscratch.buf
  @test pbuf !== nothing && u.scratch.buf === nothing
  upm = Translate(SeparableSum(SqrNormL2(1.5), SqrNormL2(2.0)), c)
  for _ in 1:2
    @test blocks_approx(gradient!(g, upm, x), map((k, xk, bk) -> k .* (xk .- bk), (1.5, 2.0), x, c))
    @test blocks_approx(prox!(fresh(x), upm, x, γ), map((k, xk, bk) -> bk .+ prox(SqrNormL2(k), xk .- bk, γ), (1.5, 2.0), x, c))
    @test upm.scratch.buf !== nothing && upm.proxscratch.buf !== nothing && upm.scratch.buf !== upm.proxscratch.buf
  end
  sbuf, pbuf2 = upm.scratch.buf, upm.proxscratch.buf
  gradient!(g, upm, x); prox!(y, upm, x, γ); gradient!(g, upm, x)
  @test upm.scratch.buf === sbuf && upm.proxscratch.buf === pbuf2
  x32 = map(a -> Float32.(a), x)
  value_and_gradient!(map(similar, x32), upm, x32)                  # a Float32 argument and a Float64 output
  @test upm.scratch.buf[3] isa Tuple{Vector{Float32}, Vector{Float32}}
  prox!(y, upm, x32, γ)
  @test upm.proxscratch.buf[3] isa Tuple{Vector{Float64}, Vector{Float64}}
  @test all(bk -> all(a -> bk !== a, y) && all(a -> bk !== a, x), upm.scratch.buf[3])

  # warm calls allocate nothing, also for four, nested and forty blocks
  vg!(g, f, x) = value_and_gradient!(g, f, x)
  gr!(g, f, x) = gradient!(g, f, x)
  vp!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  pr!(y, f, x, γ) = prox!(y, f, x, γ)
  cases = Any[]
  push!(cases, (x, c))
  x4 = ntuple(k -> fill(Float64(k), 3), 4); c4 = ntuple(k -> fill(0.1k, 3), 4)
  push!(cases, (x4, c4))
  xn = (([1.0, 2.0], [3.0]), ([4.0 5.0; 6.0 7.0], [8.0, -9.0])); cn = (([0.5, 0.5], [1.0]), ([1.0 0.0; 0.0 1.0], [0.1, -0.1]))
  push!(cases, (xn, cn))
  x40 = ntuple(k -> fill(Float64(k), 2), 40); c40 = ntuple(k -> fill(0.01k, 2), 40)
  push!(cases, (x40, c40))
  for (xx, cc) in cases
    sm = tuple_sum(xx, SqrNormL2)
    pf = tuple_sum(xx, NormL1)
    for W in (Tilt, Translate)
      f = W === Tilt ? Tilt(sm, cc, 1.0) : Translate(sm, cc)
      fp = W === Tilt ? Tilt(pf, cc, 1.0) : Translate(pf, cc)
      gg, yy = fresh(xx), fresh(xx)
      @test allocs(gr!, gg, f, xx) == 0
      @test allocs(vg!, gg, f, xx) == 0
      @test allocs(pr!, yy, fp, xx, 0.4) == 0
      @test allocs(vp!, yy, fp, xx, 0.4) == 0
      @test allocs(pr!, xx, fp, xx, 0.4) == 0
    end
  end
  @inferred value_and_gradient!(fresh(x), Tilt(smooth, c, 1.0), x)
  @inferred value_and_gradient!(fresh(x), Translate(smooth, c), x)
  @inferred value_and_prox!(fresh(x), t, x, 0.4)
  @inferred value_and_prox!(fresh(x), u, x, 0.4)
end

@testset "Conjugate of tuple Tilt and Translate" begin
  l1 = SeparableSum(NormL1(0.3), NormL1(0.2))
  x = ([1.0, 2.0], [3.0 4.0; 5.0 6.0]); c = ([0.1, 0.2], [0.3 0.4; 0.5 0.6]); γ = 0.7
  # Moreau: prox_{γf*}(x) = x - γ prox_{f/γ}(x/γ), with f the tilt of the blocks
  ref = map((xk, ck, fk) -> xk .- γ .* prox(Tilt(fk, ck, 0.0), xk ./ γ, 1 / γ), x, c, l1.fs)
  @test blocks_approx(prox(Conjugate(Tilt(l1, c, 0.0)), x, γ), ref)
  # the conjugate of a translation takes array translations only
  @test_throws ArgumentError prox(Conjugate(Translate(l1, c)), x, γ)
  @test_throws ArgumentError value_and_prox!(map(similar, x), Conjugate(Translate(l1, c)), x, γ)
end

end
