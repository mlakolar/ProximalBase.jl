using ProximalBase, Test

@testset "Sorted NormLinf fallback preserves the prefix residual ($T)" for
  (T, k, exponent, split) in ((Float32, 3, 60, 2), (Float64, 5, 120, 3))
  c = ldexp(one(T), exponent)
  a = T(1 / k)
  last = T(1 - (k - 1) * BigFloat(a))
  # The first k values sum to c exactly. A further 256 must survive.
  v = [fill(c * a, k - 1); c * last; T(256); zeros(T, 31 - k)]
  @test sum(Rational{BigInt}, v) == Rational{BigInt}(c) + 256
  score = T(256 / sqrt(BigFloat(k + 1)))
  for x in (zero(v), fill(-one(T), length(v)))
    @test kkt_violation(NormLinf(c), v, x) === score
    @test kkt_violation(Scaled(NormLinf(c / 2), T(2)), v, x) === score
  end

  # Split a small piece of the boundary sum across multiple dyadics.
  spacing = last - prevfloat(last)
  rest = ldexp(spacing, -1 - split)
  head = [fill(a, k - 1); prevfloat(last); spacing / 2; spacing / 2 - rest]
  part = rest * a
  v = c .* [head; fill(part, k - 1); T(BigFloat(rest) - (k - 1) * BigFloat(part))]
  @test sum(Rational{BigInt}, v) == Rational{BigInt}(c)
  append!(v, zeros(T, 32 - length(v)))
  for x in (zero(v), fill(-one(T), length(v)))
    @test kkt_violation(NormLinf(c), v, x) === zero(T)
    @test kkt_violation(Scaled(NormLinf(c / 2), T(2)), v, x) === zero(T)
  end
end

@testset "Sorted NormLinf aggregate rounding ($T)" for T in (Float32, Float64)
  δ = nextfloat(zero(T))
  for n in (3, 4, 5)
    v = fill(δ, n)
    c = (n - 1) * δ
    expected = n == 3 ? δ : zero(T)
    for x in (zero(v), fill(-one(T), n))
      @test kkt_violation(NormLinf(c), v, x) === expected
    end
  end

  # The exact active numerator exceeds floatmax while its norm is finite.
  m = floatmax(T)
  v = T[m, T(0.75) * m, T(0.75) * m]
  expected = T((sum(BigFloat, v) - BigFloat(m)) / sqrt(BigFloat(3)))
  @test isfinite(expected)
  for x in (zero(v), fill(-one(T), length(v)))
    @test kkt_violation(NormLinf(m), v, x) === expected
  end
end

@testset "Long NormLinf residual reductions" begin
  n = 1 << 22
  a = sqrt(0.69f0 * (eps(Float32) / 2))
  v = fill(a, n)
  v[1] = 1
  x = zeros(Float32, n)
  expected = Float32(sqrt(1 + (n - 1) * BigFloat(a)^2))
  @test kkt_violation(NormLinf(0f0), v, x) ≈ expected rtol = 8eps(Float32)
  scale = ldexp(1f0, 100)
  @test kkt_violation(NormLinf(0f0), scale .* v, x) ≈ scale * expected rtol = 8eps(Float32)

  x[end], v[end] = -1f0, 1f0
  # The singleton contributes zero. The long off-face tail remains.
  expected = Float32(sqrt(1 + (n - 2) * BigFloat(a)^2))
  @test kkt_violation(NormLinf(1f0), v, x) ≈ expected rtol = 8eps(Float32)
end
