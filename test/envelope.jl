module EnvelopeTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

# min_t fᵢ(t) + (t - v)²/(2γ), by brute force
envelope_brute(fi, v, γ) = brute_force_prox(fi, v, γ)[2]

# a quadratic φ(t) = at²/2, whose proximal operator has the slope 1/(1 + γa):
# the generic slope by differences
struct Quad <: SeparableFunction
  a::Float64
end
PB.value_entry(f::Quad, v, i) = f.a * v^2 / 2
PB.prox_entry(f::Quad, v, i, γ) = v / (1 + γ * f.a)
PB.isconvex(::Type{Quad}) = true

@testset "MoreauEnvelope" begin
  x = 3 .* randn(8)
  γ = 0.8
  @testset "closed forms" begin
    # the envelope of δ|⋅| is the Huber function, scaled by 1/γ
    e = MoreauEnvelope(NormL1(0.5), 1.0)
    @test e(x) ≈ sum(PB.Huber(0.5)(xi, 0.0) for xi in x)
    @test gradient(e, x) ≈ [PB.deriv(PB.Huber(0.5), xi, 0.0) for xi in x]
    # of a box, the squared distance over 2γ; of a quadratic, a shrunk quadratic
    eb = MoreauEnvelope(IndBox(-1.0, 2.0), γ)
    @test eb(x) ≈ sum(abs2, x .- clamp.(x, -1, 2)) / (2γ)
    eq = MoreauEnvelope(SqrNormL2(2.0), γ)
    @test eq(x) ≈ 2 / (2 * (1 + 2γ)) * sum(abs2, x)
    for (f, fi) in ((NormL1(0.7), t -> 0.7abs(t)), (ElasticNet(0.4, 0.6), t -> 0.4abs(t) + 0.3t^2),
                    (IndBox(-1.0, 1.0), t -> abs(t) <= 1 ? 0.0 : Inf))
      e = MoreauEnvelope(f, γ)
      @test e(x) ≈ sum(envelope_brute(fi, xi, γ) for xi in x) rtol = 1e-6
    end
  end

  @testset "$(nameof(typeof(f)))" for (f, z) in ((NormL1(0.7), x), (NormL1(0.3, rand(8)), x), (IndBox(-1.0, 1.0), x),
                                                 (NormL2(1.0), x), (IndBallL1(2.0), x),
                                                 (NuclearNorm(0.5), randn(6, 4)))
    e = MoreauEnvelope(f, γ)
    @test issmooth(e) && isproximable(e) && isconvex(e)
    @test isseparable(e) == isseparable(f)
    @test lipschitz(e) ≈ 1 / γ
    check_gradient(e, z)
    check_prox(e, z, 0.6)
    @test e(z) <= f(z) + 1e-12                       # below f
    # the value at the proximal point comes from that of f
    y = similar(z)
    @test value_and_prox!(y, e, z, 0.6) ≈ e(y)
    @test_throws ArgumentError gradient!(z, e, z)
  end

  @testset "separable" begin
    f = NormL1(0.4, rand(8))
    e = MoreauEnvelope(f, γ)
    p = prox(e, x, 0.6)
    @test [prox_entry(e, x[i], i, 0.6) for i in 1:8] ≈ p
    xs = SparseIterate(sparse(x))
    for k in 1:8
      prox_coordinate!(xs, e, k, 0.6)
    end
    @test Vector(xs) ≈ p
    # the envelope of a norm has a zero derivative at zero
    g = randn(8)
    @test kkt_violation(e, g, 3) ≈ abs(g[3])
    eb = MoreauEnvelope(IndBox(0.5, 2.0), γ)           # derivative -0.5/γ at zero
    @test kkt_violation(eb, g, 3) ≈ abs(g[3] - 0.5 / γ)
    @test restrict(e, [2, 5])(x[[2, 5]]) ≈ e([0, x[2], 0, 0, x[5], 0, 0, 0])
  end

  @test_throws ArgumentError MoreauEnvelope(NormL1(1.0), 0.0)
  for f in (NormL0(), MCP(1.0), GroupSCAD(1.0, [1:2]))
    @test_throws ArgumentError MoreauEnvelope(f, 1.0)
    @test_throws ArgumentError MoreauEnvelope{typeof(f), Float64}(f, 1.0)
  end
  @test_throws ArgumentError MoreauEnvelope(LogisticLoss(rand(0:1, 5), randn(5, 3)), 1.0)
end

@testset "Smoothed" begin
  @test_throws ArgumentError PB.Smoothed(NormL0(), 1.0)
  δ = 0.5
  s, h = PB.Smoothed(NormL1(δ), 1.0), PB.Huber(δ)
  for η in (-2.0, -0.5, -0.2, 0.0, 0.3, 0.5, 1.7), y in (0.0, 0.4)
    @test s(η, y) ≈ h(η, y)
    @test PB.deriv(s, η, y) ≈ PB.deriv(h, η, y)
    @test PB.deriv2(s, η, y) == PB.deriv2(h, η, y)
    @test PB.value_and_deriv(s, η, y)[1] ≈ s(η, y)
  end
  @test PB.curvature(s) == 1.0

  γ = 0.7
  for (φ, fi) in ((NormL1(0.6), t -> 0.6abs(t)), (NormL1Nonneg(0.6), t -> t >= 0 ? 0.6t : Inf),
                  (ElasticNet(0.4, 0.8), t -> 0.4abs(t) + 0.4t^2), (IndBox(-1.0, 2.0), t -> -1 <= t <= 2 ? 0.0 : Inf),
                  (IndNonnegative(), t -> t >= 0 ? 0.0 : Inf), (Quad(1.5), t -> 0.75t^2))
    ℓ = PB.Smoothed(φ, γ)
    @testset "$(nameof(typeof(φ)))" begin
      for r in (-3.1, -0.9, -0.2, 0.15, 0.8, 2.6)
        η, y = r + 0.3, 0.3
        @test ℓ(η, y) ≈ envelope_brute(fi, r, γ) rtol = 1e-6 atol = 1e-9
        hd = 1e-6
        @test PB.deriv(ℓ, η, y) ≈ (ℓ(η + hd, y) - ℓ(η - hd, y)) / (2hd) rtol = 1e-5 atol = 1e-7
        @test PB.deriv2(ℓ, η, y) ≈ (PB.deriv(ℓ, η + hd, y) - PB.deriv(ℓ, η - hd, y)) / (2hd) rtol = 1e-4 atol = 1e-6
        @test 0 <= PB.deriv2(ℓ, η, y) <= PB.curvature(ℓ) + 1e-12
      end
    end
  end
  @test PB.deriv2(PB.Smoothed(Quad(1.5), γ), 0.4, 0.0) ≈ 1.5 / (1 + γ * 1.5) rtol = 1e-6

  # in a GLM loss
  n, p = 40, 5
  X, yv = randn(n, p), randn(n)
  f = GLMLoss(PB.Smoothed(NormL1(0.3), 0.5), yv, X)
  check_gradient(f, randn(p))
  @test lipschitz(f) ≈ opnorm(X)^2 / 0.5 rtol = 1e-6
  @test_throws ArgumentError PB.Smoothed(NormL1(1.0), 0.0)
end

a_value(f, x) = f(x)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
a_grad!(g, f, x) = gradient!(g, f, x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "zero allocations" begin
  x = randn(20)
  for f in (MoreauEnvelope(NormL1(0.3), 0.7), MoreauEnvelope(NormL2(1.0), 0.7))
    y, g = similar(x), similar(x)
    @test allocs(a_value, f, x) == 0
    @test allocs(a_vgrad!, g, f, x) == 0
    @test allocs(a_grad!, g, f, x) == 0
    @test allocs(a_prox!, y, f, x, 0.4) == 0
    @test allocs(a_vprox!, y, f, x, 0.4) == 0
    @test allocs(a_prox!, x, f, x, 0.4) == 0                # in place
    @inferred value_and_prox!(y, f, x, 0.4)
    @inferred value_and_gradient!(g, f, x)
  end
  f32 = MoreauEnvelope(NormL1(0.3f0), 0.7f0)
  x32 = Float32.(x)
  @test allocs(a_vgrad!, similar(x32), f32, x32) == 0
  @test value_and_gradient!(similar(x32), f32, x32) isa Float32
end

@testset "envelope combined step range and stable interpolation" begin
  for T in (Float32, Float64)
    e = MoreauEnvelope(NormL1(one(T)), floatmax(T))
    x, y = T[1], T[7]
    for operation in (prox!, value_and_prox!)
      @test_throws ArgumentError operation(y, e, x, floatmax(T))
      @test y == T[7]
      alias = copy(x)
      @test_throws ArgumentError operation(alias, e, alias, floatmax(T))
      @test alias == x
    end
    @test_throws ArgumentError prox_entry(e, one(T), 1, floatmax(T))
    # The difference between finite opposite endpoints can overflow even when
    # their equal-weight interpolation is exactly zero.
    m = floatmax(T)
    centered = MoreauEnvelope(IndBox(m, m), one(T))
    @test prox(centered, T[-m], one(T)) == T[0]
    @test prox_entry(centered, -m, 1, one(T)) == zero(T)
  end
end

@testset "envelope energy across the squared-distance range" begin
  for T in (Float32, Float64)
    # The exact Huber envelope is γ/2 here. Squaring the distance first
    # overflows at the large scale and underflows at the small scale.
    large = T === Float32 ? T(1e30) : T(1e200)
    small = inv(large)
    for γ in (small, large)
      x = T[γ]
      e = MoreauEnvelope(NormL1(one(T)), γ)
      reference = T(BigFloat(γ) / 2)
      @test e(x) ≈ reference rtol = 8eps(T)
      g = similar(x)
      @test value_and_gradient!(g, e, x) ≈ reference rtol = 8eps(T)
      @test g ≈ T[1] rtol = 8eps(T)
      @test gradient(e, x) ≈ T[1] rtol = 8eps(T)
      ℓ = PB.Smoothed(NormL1(one(T)), γ)
      value, derivative = PB.value_and_deriv(ℓ, γ, zero(T))
      @test value ≈ reference rtol = 8eps(T)
      @test derivative ≈ one(T) rtol = 8eps(T)
    end

    # Even the finite endpoints' difference overflows. Scaling before the
    # subtraction gives a representable energy and derivative; BigFloat is
    # an independent reference for these rounded Float32/Float64 endpoints.
    m = floatmax(T)
    c, γ = T(0.6) * m, m
    x, p = T[-c], T[c]
    e = MoreauEnvelope(IndBox(c, c), γ)
    reference = T((BigFloat(x[1]) - BigFloat(p[1]))^2 / (2BigFloat(γ)))
    derivative = T((BigFloat(x[1]) - BigFloat(p[1])) / BigFloat(γ))
    @test isfinite(reference)
    @test e(x) ≈ reference rtol = 16eps(T)
    g = similar(x)
    @test value_and_gradient!(g, e, x) ≈ reference rtol = 16eps(T)
    @test g[1] ≈ derivative rtol = 8eps(T)
    @test gradient(e, x)[1] ≈ derivative rtol = 8eps(T)

    # Fused prox must use the distance at its returned point. Its former
    # unscaled squared distance was Inf even though the energy is finite.
    γ, step = m / 2, m / 4
    e = MoreauEnvelope(IndBox(c, c), γ)
    for alias in (false, true)
      input = copy(x)
      output = alias ? input : similar(input)
      value = value_and_prox!(output, e, input, step)
      reference = T((BigFloat(output[1]) - BigFloat(c))^2 / (2BigFloat(γ)))
      @test isfinite(reference)
      @test value ≈ reference rtol = 16eps(T)
      @test value ≈ e(output) rtol = 16eps(T)
    end
  end
end

end
