module BernoulliTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

const PT = ProximalBase.Testing

# Bisection on the original stationarity equation, with the original factors.
# Keep both tails: subtracting a tiny BigFloat root from one can also round.
function reference_prox(v, b, γ, w)
  setprecision(BigFloat, 256) do
    x, response, step, weight = BigFloat.((v, b, γ, w))
    μ = step * weight
    iszero(weight) && return (x, 1 - x)
    iszero(response) && x <= μ && return (zero(x), one(x))
    response == 1 && x >= 1 - μ && return (one(x), zero(x))
    reflected = BigFloat(0.5) - x + 2μ * (1 - 2response) < 0
    u, c, d = reflected ? (1 - x, 1 - response, response) : (x, response, 1 - response)
    lo, hi = zero(x), BigFloat(0.5)
    for _ in 1:2400
      t = (lo + hi) / 2
      residual = t - u - (iszero(c) ? zero(t) : μ * c / t) +
        (iszero(d) ? zero(t) : μ * d / (1 - t))
      residual < 0 ? (lo = t) : (hi = t)
      hi - lo <= ldexp(hi, -180) && break
    end
    t = (lo + hi) / 2
    reflected ? (1 - t, t) : (t, 1 - t)
  end
end

function check_reference(v::T, b, γ, w; rtol = T == Float64 ? 1e-12 : 16eps(T)) where {T}
  f = BernoulliLoss(b, w)
  z = prox_entry(f, v, 1, γ)
  p, q = reference_prox(v, b, γ, w)
  rounded = T(p)
  0 < b < 1 && (rounded = clamp(rounded, nextfloat(zero(T)), prevfloat(one(T))))
  accurate = isapprox(z, rounded; rtol, atol = 0) &&
    (q >= eps(T) / 2 ? isapprox(1 - z, 1 - rounded; rtol, atol = 0) : abs(z - rounded) <= eps(rounded))
  accurate || @info "Bernoulli reference mismatch" v b γ w z rounded
  @test z isa T
  @test z ≈ rounded rtol = rtol atol = 0
  if q >= eps(T) / 2
    @test 1 - z ≈ 1 - rounded rtol = rtol atol = 0
  else
    @test abs(z - rounded) <= eps(rounded)
  end
  0 < b < 1 && @test isfinite(value_entry(f, z, 1))
  z
end

@testset "Bernoulli domain, weights and parameters" begin
  f = BernoulliLoss([0., 0.2, 1.], [2., 3., 4.])
  x = [0., 0.3, 1.]
  @test f(x) ≈ -3 * (0.2log(0.3) + 0.8log1p(-0.3))
  @test gradient(f, x) ≈ [2., -3 * 0.2 / 0.3 + 3 * 0.8 / 0.7, -4.]
  @test f([-0.1, 0.3, 1.]) == Inf
  @test f([0., 0., 1.]) == Inf
  @test f([0., 0.3, 1.1]) == Inf
  @test f([0., 1., 1.]) == Inf
  @test all(isnan, gradient(f, [0., 0., 1.]))
  @test BernoulliLoss(0.)([0.]) == 0
  @test BernoulliLoss(1.)([1.]) == 0
  @test BernoulliLoss(0.)([1.]) == Inf
  @test BernoulliLoss(1.)([0.]) == Inf
  @test BernoulliLoss(0.5)([-Inf, 0.3]) == Inf
  @test BernoulliLoss(0.5)([Inf]) == Inf
  @test isnan(BernoulliLoss(0.5)([NaN]))
  @test isnan(prox_entry(BernoulliLoss(0.5), NaN, 1, 1.))
  @test isnan(prox_entry(BernoulliLoss(0.5), 0.3, 1, NaN))
  masked = BernoulliLoss([0., 0.5, 1., 0.2], 0.)
  v = [-Inf, -3., 4., Inf]
  @test masked(v) == 0
  @test gradient(masked, v) == zeros(4)
  @test prox(masked, v, 0.7) == v
  @test isnan(BernoulliLoss(0.5, 0.)([NaN]))
  @test BernoulliLoss([0., 0.5], [0., 2.])([-5., 0.5]) ≈ 2log(2)
  for b in (-0.1, 1.1, Inf, -Inf, NaN, [0., NaN], [0., 2.], [0., 1im])
    @test_throws ArgumentError BernoulliLoss(b)
  end
  for w in (-0.1, Inf, -Inf, NaN, [1., Inf], [0., -1.], [0., 1im])
    @test_throws ArgumentError BernoulliLoss(0.5, w)
  end
  # Parameter arrays use linear indices, irrespective of their shape.
  @test f(reshape(x, 1, 3)) == f(x)
  @test BernoulliLoss(reshape([0., 0.2, 1.], 1, 3), 2.)(x) ≈
    BernoulliLoss([0., 0.2, 1.], 2.)(x)
  @test_throws MethodError restrict(f, [1, 2])
  @test_throws MethodError kkt_violation(f, x, 1)
  @test !iszeropreserving(f)
  @test lipschitz(f) == Inf
  @test lipschitz(masked) == 0
  @test coordinate_lipschitz(BernoulliLoss(0.2, [0., 1.]), 1) == 0
  @test coordinate_lipschitz(BernoulliLoss(0.2, [0., 1.]), 2) == Inf
  @test coordinate_lipschitz!(zeros(3), f) == fill(Inf, 3)
end

@testset "Bernoulli proximal reference grid" begin
  for v in (-1e6, -3., -0.5, 0., 0.3, 0.5, 1., 1.5, 4., 1e6),
      γ in (1e-8, 1e-3, 1., 1e3, 1e8), b in (0., 1e-6, 0.2, 0.5, 0.9, 1 - 1e-6, 1.)
    check_reference(v, b, γ, 1.)
  end
  # Brute force is secondary: moderate inputs, over the full domain [0, 1].
  for v in (-3., 0.3, 1.5), γ in (1e-3, 1., 1e3), b in (0., 0.2, 0.5, 0.9, 1.)
    f = BernoulliLoss(b)
    z, = brute_force_prox(t -> value_entry(f, t, 1), v, γ; lo = 0., hi = 1., n = 2001)
    @test prox_entry(f, v, 1, γ) ≈ z rtol = 0 atol = 1e-7
  end
end

@testset "endpoint branches and extreme factors" begin
  for T in (Float32, Float64)
    for v in T[-1e6, -3, 0, 0.3, 1, 4, 1e6], γ in T[1e-8, 1, 1e8]
      @test prox_entry(BernoulliLoss(one(T), T(2)), v, 1, γ) ==
        min(one(T), prox_entry(NegLog(T(2)), v, 1, γ))
    end
    for μ in T[1e-6, 0.25, 1, 4], center in (μ, 1 - μ),
        v in (prevfloat(center), center, nextfloat(center)), b in (zero(T), one(T))
      check_reference(v, b, μ, one(T))
    end
    large = T == Float32 ? T(1e30) : T(1e200)
    small = inv(large)
    # The cutoff uses the original factors, including their product's error.
    for (γ, w) in ((T(0.1), T(0.3)), (T(1e-4), T(3)), (large, small))
      center = γ * w
      for v in (prevfloat(center), center, nextfloat(center))
        check_reference(v, zero(T), γ, w)
      end
      center = 1 - γ * w
      for v in (prevfloat(center), center, nextfloat(center))
        @test prox_entry(BernoulliLoss(one(T), w), v, 1, γ) ==
          min(one(T), prox_entry(NegLog(w), v, 1, γ))
      end
    end
    for (γ, w) in ((large, large), (small, small), (large, small), (small, large)),
        v in T[-large, -1, 0, 0.3, 1, large], b in T[1e-6, 0.2, 0.9, 1 - 1e-6]
      check_reference(v, b, γ, w)
    end
    # The input precision, rather than a wider parameter type, sets the policy.
    @test prox_entry(BernoulliLoss(0.2, small), -large, 1, small) == nextfloat(zero(T))
    @test prox_entry(BernoulliLoss(0.2, small), large, 1, small) == prevfloat(one(T))
    # At v = μ with a tiny response, retaining μb is essential.
    b = T == Float32 ? T(1e-12) : T(1e-24)
    check_reference(one(T), b, one(T), one(T))
    check_reference(floatmax(T), b, one(T), floatmax(T))
    check_reference(-floatmax(T), T(0.2), floatmax(T), floatmax(T))
    # Reflection must not round 1-v before cancellation with a large step.
    scale = T == Float32 ? T(1e8) : T(1e16)
    for v in (prevfloat(-scale), -scale, nextfloat(-scale))
      check_reference(v, prevfloat(one(T)), one(T), scale)
    end
    for v in T[-large, 0, large], b in (zero(T), one(T))
      check_reference(v, b, large, large)
      check_reference(v, b, small, small)
    end
  end
end

@testset "separable, smooth and proximal oracles" begin
  rng = MersenneTwister(182)
  b = [0., 1e-6, 0.2, 0.5, 0.9, 1 - 1e-6, 1., 0.4]
  w = [0., 0.2, 3., 1., 2., 0.5, 4., 1.]
  x = 0.2 .+ 0.6 .* rand(rng, 8)
  for f in (BernoulliLoss(0.3), BernoulliLoss(b), BernoulliLoss(0.3, w), BernoulliLoss(b, w))
    PT.test_traits(f; smooth = true, proximable = true, separable = true, convex = true)
    PT.test_separable(f, x, 0.7; rng, groups = [1:4, 5:8])
    PT.test_gradient(f, x; rng)
    PT.test_prox(f, x, 0.7; rng)
    g = copy(x)
    @test value_and_gradient!(g, f, g) ≈ f(x)
    @test g ≈ gradient(f, x)
  end
end

@testset "diagonal steps and checks before writes" begin
  x = [-4., -0.2, 0.3, 0.6, 1.2, 4.]
  b, w, γ = [0., 0.2, 0.5, 0.8, 1., 0.3], [1., 2., 0., 0.5, 3., 1.], [0.1, 2., 0.3, 0.7, 4., 1.]
  f = BernoulliLoss(b, w)
  expected = [prox_entry(f, x[i], i, γ[i]) for i in eachindex(x)]
  @test prox(f, x, γ) == expected
  y = similar(x)
  @test value_and_prox!(y, f, x, γ) ≈ f(expected)
  @test y == expected
  alias = copy(x)
  @test prox!(alias, f, alias, γ) == expected
  alias = copy(γ)
  @test prox!(alias, f, x, alias) == expected
  @test prox(f, x, fill(0.7, 6)) == prox(f, x, 0.7)
  @test prox!(y, f, x, reshape(γ, 2, 3)) == expected
  for bad in (BernoulliLoss(b[1:5], w), BernoulliLoss(b, w[1:5]))
    @test_throws DimensionMismatch bad(x)
    for op in (prox!, value_and_prox!, gradient!, value_and_gradient!)
      y = fill(17., 6)
      if op in (prox!, value_and_prox!)
        @test_throws DimensionMismatch op(y, bad, x, 0.7)
      else
        @test_throws DimensionMismatch op(y, bad, x)
      end
      @test y == fill(17., 6)
    end
    xs = SparseIterate(x)
    @test_throws DimensionMismatch prox_coordinate!(xs, bad, 1, 0.7)
    @test Vector(xs) == x
  end
  for op in (prox!, value_and_prox!)
    y = fill(17., 6)
    @test_throws DimensionMismatch op(y, f, x, ones(5))
    @test y == fill(17., 6)
    @test_throws DimensionMismatch op(zeros(5), f, x, 0.7)
  end
  for op in (gradient!, value_and_gradient!)
    y = fill(17., 5)
    @test_throws DimensionMismatch op(y, f, x)
    @test y == fill(17., 5)
  end
end

@testset "precision, gradients and allocations" begin
  rng = MersenneTwister(182)
  for T in (Float32, Float64)
    x = T[0.1, 0.3, 0.6, 0.9]
    for f in (BernoulliLoss(T(0.3), T(2)), BernoulliLoss(T[0, 0.2, 0.9, 1], T[1, 2, 0, 0.5]),
              BernoulliLoss([0., 0.2, 0.9, 1.], [1., 2., 0., 0.5]))
      y, g, step = similar(x), similar(x), T(0.7)
      @test f(x) isa T
      @test value_and_gradient!(g, f, x) isa T
      @test value_and_prox!(y, f, x, step) isa T
      @test prox(f, x, 0.7) isa Vector{T}
      @test gradient(f, x) isa Vector{T}
      @test prox(f, x, step) ≈ T.(prox(f, Float64.(x), Float64(step))) rtol = 16eps(T)
      @inferred prox_entry(f, x[2], 2, step)
      @inferred value_and_gradient!(g, f, x)
      PT.test_gradient(f, x; rng)
      @test allocs(f, x) == 0
      @test allocs(prox!, y, f, x, step) == 0
      @test allocs(value_and_prox!, y, f, x, step) == 0
      @test allocs(gradient!, g, f, x) == 0
      @test allocs(value_and_gradient!, g, f, x) == 0
      @test allocs(prox_entry, f, x[2], 2, step) == 0
      @test allocs(prox!, y, f, x, fill(step, 4)) == 0
      xs = SparseIterate(x)
      @test allocs(prox_coordinate!, xs, f, 2, step) == 0
    end
    small = T == Float32 ? T(1e-30) : T(1e-200)
    f = BernoulliLoss(small, small)
    p = T[small]
    @test gradient(f, p)[1] ≈ zero(T) atol = eps(T) * small
    @test gradient(BernoulliLoss(small, small), T[small / 2])[1] ≈ -small rtol = 16eps(T)
    p = nextfloat(zero(T))
    @test gradient(BernoulliLoss(p, one(T)), T[p])[1] == 0
    @test value_entry(BernoulliLoss(p, T(2)), p, 1) > 0
  end
  f = BernoulliLoss(0.3, 2.)
  x, y = [0.1, 0.3, 0.9], zeros(Float32, 3)
  @test prox!(y, f, x, 0.7) ≈ Float32.(prox(f, x, 0.7))
  @test value_and_prox!(y, f, x, 0.7) isa Float32
  @test value_and_gradient!(y, f, x) isa Float64
  @test y ≈ Float32.(gradient(f, x))
  # A narrower output may round a fractional-response proximal point to 1.
  @test isfinite(value_and_prox!(zeros(1), f, [1e8], 1e-3))
  @test value_and_prox!(zeros(Float32, 1), f, [1e8], 1e-3) == Inf
end

@testset "calculus and a box constraint" begin
  f = BernoulliLoss([0.2, 0.5, 1.], [1., 0., 2.])
  x, γ, α, shift = [-2., 0.3, 1.5], 0.7, 1.3, [0.2, -0.1, 0.4]
  s = Scaled(f, α)
  @test prox(s, x, γ) ≈ prox(f, x, α * γ)
  y = similar(x)
  @test value_and_prox!(y, s, x, γ) ≈ s(y)
  t = Translate(f, shift)
  @test prox(t, x, γ) ≈ prox(f, x - shift, γ) + shift
  @test value_and_prox!(y, t, x, γ) ≈ t(y)
  interior = [0.3, 0.4, 0.8]
  @test gradient(s, interior) ≈ α .* gradient(f, interior)
  @test gradient(t, interior + shift) ≈ gradient(f, interior)
  c = Conjugate(f)
  p = prox(f, x / γ, inv(γ))
  value = value_and_prox!(y, c, x, γ)
  @test y ≈ x - γ * p
  @test value ≈ dot(p, y) - f(p)
  boxed = Constrained(f, IndBox(0.15, 0.85))
  @test prox(boxed, x, γ) ≈ clamp.(prox(f, x, γ), 0.15, 0.85)
  @test value_and_prox!(y, boxed, x, γ) ≈ boxed(y)
end

end
