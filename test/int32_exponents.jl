module Int32ExponentsTest

# frexp(::BigFloat) returns an Int32 exponent where a C long has 32 bits (Windows). These tests pass
# such exponents to the helpers that take an exponent, and run the BigFloat paths that produce them.

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase

@testset "_frexp has an Int exponent" begin
  for T in (BigFloat, Float64, Float32), x in (T(3), T(0.1), T(-7.5), zero(T))
    m, e = PB._frexp(x)
    @test e isa Int
    @test (m, e) == frexp(x)
    @test typeof(m) === T
  end
  @test PB._frexp(big(0.75)) == (big(0.75), 0)
end

@testset "helpers accept an Int32 exponent" begin
  for T in (Float32, Float64, BigFloat), k in (-5, 0, 3, 40)
    a, b = PB._power_factors(T, Int32(k))
    c, d = PB._power_factors(T, k)
    @test a == c && b == d && typeof(a) === typeof(c)
  end
  @test PB._comparison_exponent(Int32(3)) === 3
  @test PB._comparison_exponent(Int32(-3)) === 0
  @test PB._comparison_exponent(Int32(3), 5, Int32(-2)) === PB._comparison_exponent(3, 5, -2) === 5
  @test PB._comparison_exponent() === 0
  for x in (big(0.0), big(1.5), big(-2.0)^40, 3f0)
    @test PB._magnitude_exponent(x) isa Int
    @test PB._magnitude_exponent(x) === (iszero(x) ? 0 : Int(exponent(x)))
  end

  for T in (Float64, BigFloat)
    a, b, c = T(1.25), T(-3), T(7)
    for k in (-4, 0, 6)
      @test PB._scaled_product(a, b, Int32(k)) == PB._scaled_product(a, b, k)
      @test PB._scaled_product(a, b, c, Int32(k)) == PB._scaled_product(a, b, c, k)
      @test PB._scaled_product(a, b, Int32(k)) == ldexp(a * b, k)
    end
  end

  # the scaled products of an image
  for T in (Float64, BigFloat), k in (-3, 0, 5)
    s32, s = (T(0.5), Int32(k)), (T(0.5), k)
    @test PB._image_product(T(1), T(2), s32) == PB._image_product(T(1), T(2), s)
    @test PB._image_product(T(3), T(-5), s32) == ldexp(T(-7.5), k)
    a, b = T[1, 2, 3], T[4, -5, 6]
    @test PB._image_scaled_dot(a, b, s32) == PB._image_scaled_dot(a, b, s)
    @test PB._image_scaled_dot(a, b, s32) == ldexp(T(6), k)
  end

  # the squared hinge, at a finite and at an overflowing residual
  for T in (Float64, BigFloat), e in (-3, 0, 4)
    w, r, y = T(1.5), T(0.75), T(-2)
    @test PB._sqrhinge_value(w, r, Int32(e)) == PB._sqrhinge_value(w, r, e)
    @test PB._sqrhinge_slope(y, w, r, Int32(e)) == PB._sqrhinge_slope(y, w, r, e)
  end

  # the margin of the refined window of DistL2
  for T in (Float64, BigFloat)
    x, p = T[3, 4], T[0, 0]
    gm, ge = frexp(T(3))
    lm, le = frexp(T(1.5))
    dm, de = frexp(T(5))
    for k in (0, 2)
      r = PB._distl2_margin(x, p, gm, lm, ge + le + k, de)
      @test PB._distl2_margin(x, p, gm, lm, Int32(ge + le + k), Int32(de)) == r
      @test PB._distl2_margin(x, p, gm, lm, Int32(ge + le + k), de) == r
    end
  end
end

@testset "BigFloat DistL2 prox in the refined window" begin
  setprecision(BigFloat, 256) do
    p = zeros(BigFloat, 2)
    f = DistL2(IndBox(p, p), big(1.0))
    x = big.([3.0, 4.0])
    d = big(5.0)
    # the step γλ is close to the distance: below it, at the 0.75 d end of the window, and above it
    for γ in (big(4.0), prevfloat(big(5.0)), nextfloat(big(5.0)), big(3.8), big(9.9))
      expected = γ >= d ? zeros(BigFloat, 2) :
                 setprecision(() -> x .* (1 - γ / d), BigFloat, 1024)   # the reference, at four times the precision
      for op in (prox!, value_and_prox!)
        y = similar(x)
        op(y, f, copy(x), γ)
        @test all(isfinite, y)
        @test all(abs.(y .- expected) .<= 16 * eps(BigFloat) .* abs.(expected))
      end
      @test prox(f, x, γ) == prox!(similar(x), f, copy(x), γ)
    end
    # x already inside the ball of radius γλ is sent to the projection
    @test prox(f, x, big(5.5)) == p
  end
end

@testset "BigFloat LinearImage with a BigFloat scale" begin
  setprecision(BigFloat, 256) do
    L = big.([1 1; 1 -1])                  # L'L = 2I
    μ = big(2.0)
    λ = big(0.3)
    h = LinearImage(NormL1(λ), L, μ)
    z0 = big.([1.5, -0.25])
    y = L * z0
    @test h(y) == λ * sum(abs, z0)
    for γ in (big(0.1), big(0.9), big(3.0))
      latent = L' * y / μ
      step = γ / μ
      z = sign.(latent) .* max.(abs.(latent) .- step * λ, 0)
      expected = L * z
      for op in (prox!, value_and_prox!)
        out = similar(y)
        op(out, h, copy(y), γ)
        @test all(abs.(out .- expected) .<= 16 * eps(BigFloat) .* max.(abs.(expected), eps(BigFloat)))
      end
    end
  end
  # a Float64 matrix and a BigFloat scale
  h = LinearImage(NormL1(0.3), [1.0 1.0; 1.0 -1.0], big(2.0))
  y = [1.25, 1.75]                         # L'y / μ = (1.5, -0.25)
  @test h(y) ≈ 0.3 * 1.75 rtol = 1e-14
  z = [1.5 - 0.1 / 2 * 0.3, -0.25 + 0.1 / 2 * 0.3]
  @test prox(h, y, 0.1) ≈ [z[1] + z[2], z[1] - z[2]] rtol = 1e-14
end

@testset "BigFloat squared hinge with an overflowing residual" begin
  setprecision(BigFloat, 256) do
    P = 2^62 - 10                            # y v = -2^(P + 20) is beyond the range of a BigFloat
    y, v = ldexp(big(1.0), 20), -ldexp(big(1.0), P)
    @test !isfinite(y * v)
    w = ldexp(big(1.0), -P)
    f = SqrHingeLoss([y], [w])
    # the slope is 2 w y r = 2^(1 + 40 + P - P) = 2^41 with r = 1 + y|v| = 2^(P + 20), and the value overflows
    g = gradient(f, [v])
    expected = -ldexp(big(1.0), 41)
    @test isfinite(g[1])
    @test abs(g[1] - expected) <= 4 * eps(BigFloat) * abs(expected)
    @test f([v]) == Inf
    @test PB.value_entry(f, v, 1) == Inf
    g2 = similar([v])
    @test value_and_gradient!(g2, f, [v]) == Inf
    @test g2 == g
  end
end

end
