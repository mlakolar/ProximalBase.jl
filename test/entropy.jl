module EntropyTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random
using LambertW: lambertw

using ..TestUtils

Random.seed!(1)

hobj(pen, z, v, γ) = pen(z) + (z - v)^2 / (2γ)

# the proximal operator is never worse than the brute-force minimizer of the
# scalar problem over the domain [lo, ∞)
function check_scalar_prox(f, pen; vs, γs, lo = -Inf)
  for v in vs, γ in γs
    _, hb = brute_force_prox(pen, v, γ; lo = max(lo, v - 5abs(v) - 5), hi = v + 5abs(v) + 5)
    z = prox_entry(f, v, 1, γ)
    @test hobj(pen, z, v, γ) <= hb + 1e-9 * max(1, abs(hb))
  end
end

@testset "Wright omega near the Taylor-start boundaries" begin
  for T in (Float32, Float64)
    us = vcat(collect(range(T(-2), T(2); length=1001)),
              T[prevfloat(T(-2)), nextfloat(T(-2)), prevfloat(T(2)), nextfloat(T(2))])
    for u in us
      reference = T(lambertw(exp(Float64(u))))
      @test ProximalBase._wright_omega(u) ≈ reference rtol = 8eps(T)
    end
  end
end

@testset "the Wright omega function" begin
  ω = ProximalBase._wright_omega
  # against the Lambert W function where eᵘ is finite
  for u in range(-30, 30; length = 601)
    @test ω(u) ≈ lambertw(exp(u)) rtol = 4eps()
  end
  # ω + log ω = u where eᵘ overflows or underflows
  for u in (50., 1e3, 1e6, 1e12, 1e100, 1e300)
    w = ω(u)
    @test w + log(w) ≈ u rtol = 4eps()
    u >= 1e6 && @test w ≈ u - log(u) + log(u) / u rtol = 1e-10       # the asymptotic series
  end
  for u in (-40., -100., -700., -800.)
    @test ω(u) ≈ exp(u) rtol = 4eps()
  end
  @test ω(-Inf) == 0 && ω(Inf) == Inf && isnan(ω(NaN))
  @test issorted(ω.(range(-50, 50; length = 2001)))
  @test ω(1.0) ≈ 1.0                                    # ω(1) = 1
  @test ω(0) ≈ lambertw(1.0)
  # Float32, to its precision
  for u in Float32[-20, -3, -0.5, 0, 0.7, 4, 90, 1f30]
    @test ω(u) isa Float32
    @test ω(u) ≈ Float32(ω(Float64(u))) rtol = 4eps(Float32)
  end
  @inferred ω(0.3)
  @inferred ω(0.3f0)
end

@testset "NegEntropy" begin
  λ = 0.7
  f = NegEntropy(λ)
  pen(t) = t < 0 ? Inf : (iszero(t) ? 0.0 : λ * t * log(t))
  x = rand(20) .* 3
  @test f(x) ≈ sum(pen, x)
  @test f([0.0, 1.0]) == 0 && f([-1e-3, 1.0]) == Inf
  check_scalar_prox(f, pen; vs = (-5., -1., 0., 0.3, 1., 2.5, 10.), γs = (0.01, 0.3, 1., 5.), lo = 0.)
  # the optimality condition λ(log z + 1) + (z - v)/γ = 0, also far from 1
  for v in (-30., -2., 0., 3., 50.), γ in (1e-6, 1e-2, 1., 100.)
    z = prox_entry(f, v, 1, γ)
    μ = γ * λ
    if v / μ - 1 - log(μ) < log(floatmin(Float64))       # the positive result underflows
      @test z == 0
    else
      @test z > 0
      @test λ * (log(z) + 1) + (z - v) / γ ≈ 0 atol = 1e-8 * max(1, abs(v) / γ)
    end
  end
  # a small step leaves a positive entry nearly unchanged, without overflow
  @test prox_entry(f, 2.0, 1, 1e-9) ≈ 2.0 - 1e-9 * λ * (1 + log(2.0)) rtol = 1e-12
  check_prox(f, x, 0.8)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test !iszeropreserving(f)
  @test restrict(f, [1, 2]) === f
  @test_throws ArgumentError NegEntropy(0.)
end

@testset "KLDivergence" begin
  λ = 0.6
  q = rand(20) .+ 0.2
  f = KLDivergence(q; λ)
  x = rand(20) .* 2
  @test f(x) ≈ λ * sum(@. x * log(x / q) - x + q)
  @test f(q) ≈ 0 atol = 1e-12
  @test f([zeros(19); 1.0]) ≈ λ * (sum(q[1:19]) + log(1 / q[20]) - 1 + q[20])
  for i in (1, 7), v in (-3., 0., 0.5, 2., 9.), γ in (0.05, 1., 4.)
    pen(t) = t < 0 ? Inf : (iszero(t) ? λ * q[i] : λ * (t * log(t / q[i]) - t + q[i]))
    _, hb = brute_force_prox(pen, v, γ; lo = 0., hi = v + 5abs(v) + 5)
    @test hobj(pen, prox_entry(f, v, i, γ), v, γ) <= hb + 1e-9 * max(1, abs(hb))
  end
  # the minimizer q is its own proximal point
  @test prox(f, q, 0.7) ≈ q
  check_prox(f, x, 0.8)
  fs = KLDivergence(2.0)
  @test prox(fs, [2.0, 2.0], 0.3) ≈ [2.0, 2.0]
  @test restrict(f, [2, 5]).q == q[[2, 5]]
  @test_throws DimensionMismatch f(ones(3))
  @test_throws ArgumentError KLDivergence([1.0, 0.0])
  @test_throws ArgumentError KLDivergence(1.0; λ = 0)
end

@testset "NegLog" begin
  λ = 0.9
  f = NegLog(λ)
  pen(t) = t <= 0 ? Inf : -λ * log(t)
  x = rand(20) .+ 0.1
  @test f(x) ≈ -λ * sum(log, x)
  @test f([1.0, 0.0]) == Inf
  check_scalar_prox(f, pen; vs = (-6., -1., 0., 0.4, 2., 30.), γs = (0.02, 0.5, 2.), lo = 1e-12)
  # the positive root of z² - vz - γλ, without cancellation for negative v
  for v in (-1e6, -30., -1., 0., 1., 1e6), γ in (1e-3, 1., 10.)
    μ = γ * λ
    z = prox_entry(f, v, 1, γ)
    @test z > 0
    @test z^2 - v * z - μ ≈ 0 atol = 1e-10 * max(z^2, μ)
  end
  check_prox(f, x, 0.8)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test_throws ArgumentError NegLog(-1.)
  for T in (Float32, Float64)
    tiny = T == Float32 ? T(1e-20) : T(1e-200)
    f = NegLog(one(T))
    for v in T[-1, 0, 1], μ in (tiny, one(T), sqrt(floatmax(T)))
      z = prox_entry(f, v, 1, μ)
      reference = setprecision(BigFloat, 256) do
        b, m = BigFloat(v), BigFloat(μ)
        v < 0 ? 2m / (sqrt(b*b + 4m) - b) : (b + sqrt(b*b + 4m)) / 2
      end
      @test isfinite(z) && z > 0
      @test z ≈ T(reference) rtol=8eps(T)
    end
    @test prox(f, T[1], tiny) ≈ T[1]
  end
end

@testset "Berhu" begin
  λ, M = 0.5, 1.2
  f = Berhu(λ, M)
  pen(t) = abs(t) <= M ? λ * abs(t) : λ * (t^2 + M^2) / (2M)
  x = 3 .* randn(30)
  @test f(x) ≈ sum(pen, x)
  check_scalar_prox(f, pen; vs = (-6., -1.8, -0.7, -0.2, 0., 0.4, 1., 1.5, 1.8, 3., 8.), γs = (0.1, 0.6, 1., 3.))
  # continuous at |v| = γλ and at |v| = M + γλ
  γ = 0.8
  c = γ * λ
  @test prox_entry(f, c, 1, γ) == 0
  @test prox_entry(f, (M + c) * (1 - 1e-12), 1, γ) ≈ prox_entry(f, (M + c) * (1 + 1e-12), 1, γ) rtol = 1e-10
  @test prox_entry(f, 10.0, 1, γ) ≈ 10M / (M + c)
  # weights, the iterates, optimality at zero, restriction
  w = rand(30) .+ 0.5
  fw = Berhu(λ, M, w)
  @test fw(x) ≈ sum(i -> λ * w[i] * (abs(x[i]) <= M ? abs(x[i]) : (x[i]^2 + M^2) / (2M)), 1:30)
  check_prox(f, x, 0.8)
  check_prox(fw, x, 0.8)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test iszeropreserving(f)
  xs = SparseIterate(sprandn(30, 0.3))
  @test fw(xs) ≈ fw(Vector(xs))
  @test Vector(prox(fw, xs, 0.6)) ≈ prox(fw, Vector(xs), 0.6)
  @test kkt_violation(fw, x, x, 3) ≈ abs(x[3]) - λ * w[3]
  @test restrict(fw, [1, 4]).w == w[[1, 4]]
  # with M → ∞ it is the ℓ1 norm, and small M makes it a ridge for most entries
  @test prox(Berhu(λ, 1e12), x, γ) ≈ prox(NormL1(λ), x, γ)
  @test_throws ArgumentError Berhu(1., 0.)
  @test_throws ArgumentError Berhu(-1., 1.)
end

@testset "Float32 and the iterates" begin
  for f in (NegEntropy(0.5), KLDivergence(1.5; λ = 0.5), NegLog(0.5), Berhu(0.5, 1.0))
    x = rand(Float32, 8) .+ 0.1f0
    @test f(x) isa Float32
    @test prox(f, x, 0.3f0) isa Vector{Float32}
    @test prox(f, x, 0.3f0) ≈ Float32.(prox(f, Float64.(x), 0.3)) rtol = 1e-5
    @inferred prox_entry(f, 0.7, 2, 0.6)
    @inferred f(rand(5))
    xs = SparseIterate(sprand(8, 0.5))
    @test Vector(prox(f, xs, 0.4)) ≈ prox(f, Vector(xs), 0.4)
    D = rand(4, 4) .+ 0.1; D = D + D'
    S = SymmetricSparseIterate(D)
    f isa KLDivergence && continue
    @test Matrix(prox(f, S, 0.5)) ≈ prox(f, D, 0.5)
  end
end

@testset "zero allocations" begin
  a_value(f, x) = f(x)
  a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
  a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
  x = rand(20) .+ 0.1
  for f in (NegEntropy(0.5), KLDivergence(rand(20) .+ 0.5; λ = 0.5), NegLog(0.5), Berhu(0.5, 1.), Berhu(0.5, 1., rand(20)))
    for v in (x, Float32.(x))
      y = similar(v)
      @test allocs(a_value, f, v) == 0
      @test allocs(a_prox!, y, f, v, 0.7) == 0
      @test allocs(a_vprox!, y, f, v, 0.7) == 0
    end
    xs = SparseIterate(x)
    @test allocs(a_cd!, xs, f, 3, 0.7) == 0
  end
end

end
