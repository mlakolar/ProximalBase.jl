module BernoulliTest

using Test
using ProximalBase
using LinearAlgebra
using Random
using SparseArrays
using JLArrays: JLArray
using GPUArraysCore: allowscalar
import Adapt

using ..TestUtils

const PT = ProximalBase.Testing
const PB = ProximalBase

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
    @test_throws ArgumentError value_and_gradient!(g, f, g)
    @test g == x
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

####################################
#  ProximalBase.IdentityBernoulli: the Bernoulli loss with the identity link as a ScalarLoss
####################################

const IB = PB.IdentityBernoulli()
const ib_response_message = "the responses of an IdentityBernoulli loss must be in [0, 1]"

# ℓ, ℓ′ and ℓ″ of one observation in 256-bit arithmetic from the formulas, with a zero
# coefficient omitted before it is divided or multiplied.
function ib_reference(p, y)
  setprecision(BigFloat, 256) do
    p, y = BigFloat(p), BigFloat(y)
    v = (iszero(y) ? zero(p) : -y * log(p)) + (y == 1 ? zero(p) : -(1 - y) * log1p(-p))
    d1 = (y == 1 ? zero(p) : (1 - y) / (1 - p)) - (iszero(y) ? zero(p) : y / p)
    d2 = (y == 1 ? zero(p) : (1 - y) / (1 - p)^2) + (iszero(y) ? zero(p) : y / p^2)
    (v, d1, d2)
  end
end

# the references of a vector of observations, in Float64
function ib_references(η, y)
  r = ib_reference.(η, y)
  (Float64.(getindex.(r, 1)), Float64.(getindex.(r, 2)), Float64.(getindex.(r, 3)))
end

# x is within `ulps` units in the last place of the reference r, and Inf where r leaves the range
function ib_close(x::T, r, ulps = 8) where {T}
  abs(r) > floatmax(T) && return x == sign(r) * T(Inf)
  isfinite(x) && abs(BigFloat(x) - r) <= ulps * max(eps(T) * abs(r), BigFloat(nextfloat(zero(T))))
end

ib_value(f, x) = f(x)
ib_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
ib_grad!(g, f, x) = gradient!(g, f, x)

@testset "IdentityBernoulli: the scalar loss, $T" for T in (Float32, Float64)
  nan, inf = T(NaN), T(Inf)
  # (p, y, value, ℓ′, ℓ″); the endpoint rows are one-sided limits
  table = [(T(0.5), T(0), log(T(2)), T(2), T(4)),
           (T(0.5), T(1), log(T(2)), T(-2), T(4)),
           (T(0.5), T(0.5), log(T(2)), T(0), T(4)),
           (T(0), T(0), T(0), T(1), T(1)),
           (T(1), T(1), T(0), T(-1), T(1)),
           (-T(0), T(0), T(0), T(1), T(1))]
  for (p, y, v, d1, d2) in table
    @test IB(p, y) isa T && PB.deriv(IB, p, y) isa T && PB.deriv2(IB, p, y) isa T
    @test IB(p, y) ≈ v && PB.deriv(IB, p, y) ≈ d1 && PB.deriv2(IB, p, y) ≈ d2
  end
  # outside the domain the value is infinite and the derivatives are NaN
  for (p, y) in ((T(0), T(1)), (T(0), T(0.3)), (T(1), T(0)), (T(1), T(0.3)), (T(-0.1), T(0)),
                 (T(-0.1), T(0.5)), (T(1.1), T(1)), (T(1.1), T(0.5)), (inf, T(0.5)), (-inf, T(1)),
                 (-inf, T(0)), (inf, T(0)), (-T(0), T(1)), (nextfloat(one(T)), T(1)), (-nextfloat(T(0)), T(0)))
    @test IB(p, y) == inf
    @test isnan(PB.deriv(IB, p, y)) && isnan(PB.deriv2(IB, p, y))
  end
  # a NaN predictor gives NaN
  for y in T.((0, 0.2, 1))
    @test isnan(IB(nan, y)) && isnan(PB.deriv(IB, nan, y)) && isnan(PB.deriv2(IB, nan, y))
  end
  # value_and_deriv is the pair of the two functions
  for p in T.((0, 0.2, 0.5, 1, 1.5, -1)), y in T.((0, 0.2, 1))
    @test isequal(PB.value_and_deriv(IB, p, y), (IB(p, y), PB.deriv(IB, p, y)))
  end
  @test isequal(PB.value_and_deriv(IB, nan, T(0.2)), (nan, nan))
  @test PB.curvature(IB) == Inf && isbitstype(typeof(IB)) && IB isa PB.ScalarLoss
  # the derivatives at interior points, by central differences
  h = sqrt(eps(T))
  for p in T.((0.15, 0.3, 0.5, 0.8)), y in T.((0, 1, 0.2, 0.7))
    d1, d2 = PB.deriv(IB, p, y), PB.deriv2(IB, p, y)
    @test abs((IB(p + h, y) - IB(p - h, y)) / 2h - d1) <= 10 * sqrt(eps(T)) * (1 + abs(d1))
    @test abs((PB.deriv(IB, p + h, y) - PB.deriv(IB, p - h, y)) / 2h - d2) <= 10 * sqrt(eps(T)) * (1 + abs(d2))
  end
  # it is the loss of BernoulliLoss at the probability p
  for p in T.((0, 0.2, 0.5, 1)), y in T.((0, 0.2, 1))
    f = BernoulliLoss(y)
    @test isequal(IB(p, y), f([p])) && isequal(PB.deriv(IB, p, y), gradient(f, [p])[1])
  end
end

@testset "IdentityBernoulli: near the boundary against 256-bit references, $T" for T in (Float32, Float64)
  ps = (nextfloat(zero(T)), T(1e-30), eps(T), T(1e-3), T(0.5), 1 - eps(T), prevfloat(one(T)))
  ys = (zero(T), one(T), T(0.2), T(0.7), T(1e-10), 1 - T(1e-6), nextfloat(zero(T)), prevfloat(one(T)))
  for p in ps, y in ys
    v, d1, d2 = ib_reference(p, y)
    # the derivative is a difference of two positive terms: skip a pair where they cancel
    cancels = abs(d1) < 0.01 * ((1 - BigFloat(y)) / (1 - BigFloat(p)) + BigFloat(y) / BigFloat(p))
    @test ib_close(IB(p, y), v)
    cancels || @test ib_close(PB.deriv(IB, p, y), d1)
    @test ib_close(PB.deriv2(IB, p, y), d2)
    cancels || @test ib_close(PB.value_and_deriv(IB, p, y)[2], d1)
  end
  # ℓ″ is a quotient that does not go through p², which underflows below sqrt(floatmin)
  # while the quotient is representable, and is Inf only where it is not
  s = sqrt(floatmin(T))
  extreme = T == Float64 ?
    [(1e-160, 1e-300), (1e-200, 1e-300), (1e-300, 1e-310), (1e-100, 1e-250), (1e-154, 1e-300), (1.5e-154, 1e-300)] :
    [(1f-25, 1f-38), (1f-30, 1f-40), (1f-19, 1f-30), (1f-10, 1f-20)]
  for (p, y) in extreme
    p, y = T(p), T(y)
    for q in (p, s, nextfloat(s), prevfloat(s))
      d2 = ib_reference(q, y)[3]
      @test isfinite(PB.deriv2(IB, q, y)) && ib_close(PB.deriv2(IB, q, y), d2)
      @test ib_close(PB.deriv(IB, q, y), ib_reference(q, y)[2])
    end
  end
  T == Float64 && @test PB.deriv2(IB, 1e-160, 1e-300) ≈ 1e20 rtol = 1e-14
  # the other end of the interval, and a quotient that overflows
  @test PB.deriv2(IB, nextfloat(zero(T)), one(T)) == T(Inf)
  @test ib_close(PB.deriv2(IB, prevfloat(one(T)), zero(T)), ib_reference(prevfloat(one(T)), zero(T))[3])
  @test PB.deriv2(IB, nextfloat(zero(T)), zero(T)) == 1
  @test !isnan(PB.deriv2(IB, nextfloat(zero(T)), T(0.5)))
end

@testset "IdentityBernoulli: GLMLoss construction and responses" begin
  X = [1.0 0.0; 0.0 1.0; 1.0 1.0]
  @test GLMLoss(IB, [0.0, 1.0, 0.5], X) isa GLMLoss{typeof(IB)}
  @test GLMLoss(IB, [0, 1, 1], X) isa GLMLoss                                  # integer responses
  @test GLMLoss(IB, [-0.0, nextfloat(0.0), prevfloat(1.0)], X) isa GLMLoss    # the closed interval
  @test GLMLoss(IB, Float32[0, 1, 0.3], Float32.(X); weights = Float32[1, 0, 2], offset = Float32[0, 0.1, 0]) isa GLMLoss
  # a response outside [0, 1], or not finite, is rejected whatever its weight
  for bad in (NaN, Inf, -Inf, -eps(), nextfloat(1.0), 2.0, -1.0)
    for weights in (nothing, [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], zeros(3))
      @test_throws ArgumentError(ib_response_message) GLMLoss(IB, [0.0, bad, 0.5], X; weights)
    end
    @test_throws ArgumentError(ib_response_message) GLMLoss(IB, [bad, 0.0, 0.5], X; offset = [0.1, 0.2, 0.3])
  end
  @test_throws ArgumentError(ib_response_message) GLMLoss(IB, [0, 2, 1], X)
  for bad in (NaN32, Inf32, nextfloat(1.0f0), -nextfloat(0.0f0))
    @test_throws ArgumentError(ib_response_message) GLMLoss(IB, Float32[0, bad, 1], Float32.(X); weights = Float32[1, 0, 1])
  end
  # every construction path goes through the check: the buffers, restrict and Adapt
  @test_throws ArgumentError(ib_response_message) GLMLoss(IB, [0.0, 2.0, 0.5], X, nothing, nothing, zeros(3), zeros(3))
  @test GLMLoss(IB, [0.0, 1.0, 0.5], X, nothing, nothing, zeros(3), zeros(3)) isa GLMLoss
  f = GLMLoss(IB, [0.0, 1.0, 0.5], X)
  @test restrict(f, [2]) isa GLMLoss{typeof(IB)}
  # the check is an isbits callable, which a device reduction can take, and no cost for other losses
  for T in (Float32, Float64)
    @test isbitstype(typeof(PB._UnitResponse{T}())) && PB._UnitResponse{T}() isa Function
    @test all(PB._UnitResponse{T}(), T[0, 0.3, 1]) && !any(PB._UnitResponse{T}(), T[-1, 2, NaN, Inf])
  end
  @test PB._check_responses(PB.Logistic(), [NaN, 5.0], Float64) === nothing
  @test LogisticLoss([0.0, 2.0, 3.0], X) isa LogisticLoss                      # other losses do not check
  @test GLMLoss(PB.Squared(), [0.0, 7.0, NaN], X) isa GLMLoss
  # the other checks keep their errors
  @test_throws DimensionMismatch GLMLoss(IB, [0.0, 1.0], X)
  @test_throws DimensionMismatch GLMLoss(IB, [0.0, 1.0, 0.5], X; weights = [1.0, 1.0])
  @test_throws DimensionMismatch GLMLoss(IB, [0.0, 1.0, 0.5], X; offset = [0.1])
  @test_throws ArgumentError("the weights must be finite and nonnegative") GLMLoss(IB, [0.0, 1.0, 0.5], X; weights = [1.0, -1.0, 1.0])
  @test_throws ArgumentError("the weights must be finite and nonnegative") GLMLoss(IB, [0.0, 1.0, 0.5], X; weights = [1.0, NaN, 1.0])
  @test_throws ArgumentError("the weights must be finite and nonnegative") GLMLoss(IB, [0.0, 1.0, 0.5], X; weights = [1.0, Inf, 1.0])
  @test_throws ArgumentError("the offset must be finite") GLMLoss(IB, [0.0, 1.0, 0.5], X; offset = [0.0, NaN, 0.0])
  @test_throws ArgumentError("the offset must be finite") GLMLoss(IB, [0.0, 1.0, 0.5], X; offset = [0.0, Inf, 0.0])
  @test_throws ArgumentError("GLMLoss requires real entries") GLMLoss(IB, complex.([0.0, 1.0, 0.5]), X)
  # no observations, and no columns
  f0 = GLMLoss(IB, Float64[], zeros(0, 3))
  @test f0(zeros(3)) == 0 && gradient(f0, zeros(3)) == zeros(3)
  @test hessian_vec(f0, zeros(3), ones(3)) == zeros(3) && PB.lipschitz(f0) == Inf
  y = [0.0, 1.0, 0.5]
  o = [0.2, 0.7, 0.5]
  fp = GLMLoss(IB, y, zeros(3, 0); offset = o)
  @test fp(Float64[]) ≈ sum(first.(ib_reference.(o, y))) && gradient(fp, Float64[]) == Float64[]
  # a public name that is not exported
  @test !Base.isexported(ProximalBase, :IdentityBernoulli)
  VERSION >= v"1.11" && @test Base.ispublic(ProximalBase, :IdentityBernoulli)
end

@testset "IdentityBernoulli: GLMLoss oracles" begin
  Random.seed!(226)
  n, p = 12, 3
  X = 0.1 .+ 0.4 .* rand(n, p)
  β = [0.2, 0.3, 0.1]
  o = 0.05 .* rand(n)
  w = 0.5 .+ rand(n)
  y = [0.0, 1.0, 0.2, 0.7, 0.0, 1.0, 0.5, 0.0, 1.0, 0.35, 0.9, 0.1]
  for (wts, off) in ((nothing, nothing), (w, nothing), (nothing, o), (w, o))
    wv = wts === nothing ? ones(n) : wts
    η = X * β .+ (off === nothing ? zeros(n) : off)
    @test all(0 .< η .< 1)
    v, d1, d2 = ib_references(η, y)
    f = GLMLoss(IB, y, X; weights = wts, offset = off)
    @test f(β) ≈ sum(wv .* v)
    @test f(β) ≈ (wts === nothing ? BernoulliLoss(y) : BernoulliLoss(y, wts))(η)
    @test gradient(f, β) ≈ X' * (wv .* d1)
    g = fill(NaN, p)
    @test gradient!(g, f, β) === g && g ≈ X' * (wv .* d1)
    g = fill(NaN, p)
    @test value_and_gradient!(g, f, β) ≈ sum(wv .* v) && g ≈ X' * (wv .* d1)
    @test value_and_gradient(f, β)[1] ≈ sum(wv .* v) && value_and_gradient(f, β)[2] ≈ g
    check_gradient(f, β)
    # the output must not be the argument
    @test_throws ArgumentError("the gradient must not overwrite x") gradient!(β, f, β)
    @test_throws ArgumentError("the gradient must not overwrite x") value_and_gradient!(β, f, β)
    # a sparse iterate and a restriction
    βs = SparseIterate(sparse([0.2, 0.0, 0.1]))
    @test f(βs) ≈ f(Vector(βs)) && gradient(f, βs) ≈ gradient(f, Vector(βs))
    fr = restrict(f, [1, 3])
    @test fr([β[1], β[3]]) ≈ f([β[1], 0.0, β[3]])
    @test gradient(fr, [β[1], β[3]]) ≈ gradient(f, [β[1], 0.0, β[3]])[[1, 3]]
    # the second-order oracles
    v2 = [0.3, -1.0, 0.5]
    @test hessian_vec(f, β, v2) ≈ X' * (wv .* d2 .* (X * v2))
    q = quadratic_model(f, β)
    @test q.weights ≈ wv .* d2 && all(>(0), q.weights)
    @test q.y ≈ η .- d1 ./ d2
    @test q.X === f.X && q.offset === f.offset
    @test gradient(q, β) ≈ gradient(f, β)
    @test hessian_vec(q, β, v2) ≈ hessian_vec(f, β, v2)
    β2 = [0.1, 0.25, 0.2]
    @test quadratic_model!(q, f, β2) === q
    q2 = quadratic_model(f, β2)
    @test q.weights == q2.weights && q.y == q2.y
    @test gradient(q, β2) ≈ gradient(f, β2)
  end

  # the valid endpoints, against BernoulliLoss: X = I and β = p
  m = 6
  ye = [0.0, 1.0, 0.4, 0.0, 1.0, 0.25]
  pe = [0.0, 1.0, 0.4, 0.0, 1.0, 0.25]
  we = [1.0, 2.0, 0.5, 3.0, 1.0, 0.0]
  Ie = Matrix{Float64}(I, m, m)
  for wts in (nothing, we)
    f = GLMLoss(IB, ye, Ie; weights = wts)
    b = wts === nothing ? BernoulliLoss(ye) : BernoulliLoss(ye, wts)
    wq = wts === nothing ? ones(m) : we
    v, d1, d2 = ib_references(pe, ye)
    @test isfinite(f(pe)) && f(pe) ≈ b(pe) && f(pe) ≈ sum(wq .* v)
    @test gradient(f, pe) ≈ gradient(b, pe) && gradient(f, pe) ≈ wq .* d1
    @test d1[[1, 4]] == [1, 1] && d1[[2, 5]] == [-1, -1] && d2[[1, 2, 4, 5]] == ones(4)   # the one-sided limits
    @test hessian_vec(f, pe, ones(m)) ≈ wq .* d2
    q = quadratic_model(f, pe)
    @test q.weights ≈ wq .* d2 && q.y ≈ ifelse.(iszero.(wq), 0.0, pe .- d1 ./ d2)
  end

  # a zero weight masks an observation outside the domain, a positive weight does not
  I3 = Matrix{Float64}(I, 3, 3)
  y3 = [0.3, 0.6, 0.5]
  r1, r3 = ib_reference(0.2, 0.3), ib_reference(0.5, 0.5)
  for bad in (1.7, -0.4, 1.0 + 1e-12)
    β3 = [0.2, bad, 0.5]
    f0 = GLMLoss(IB, y3, I3; weights = [1.0, 0.0, 1.0])
    @test f0(β3) ≈ Float64(r1[1] + r3[1])
    g = gradient(f0, β3)
    @test all(isfinite, g) && g[2] == 0 && g[1] ≈ Float64(r1[2]) && abs(g[3]) < 1e-12
    @test all(isfinite, hessian_vec(f0, β3, ones(3)))
    q = quadratic_model(f0, β3)
    @test q.weights[2] == 0 && q.y[2] == 0 && all(isfinite, q.y) && all(isfinite, q.weights)
    f1 = GLMLoss(IB, y3, I3)
    @test f1(β3) == Inf
    g = zeros(3)
    @test value_and_gradient!(g, f1, β3) == Inf && isnan(g[2])
    @test isnan(gradient(f1, β3)[2])
    @test isnan(hessian_vec(f1, β3, ones(3))[2])
  end
  @test isnan(GLMLoss(IB, y3, I3)([0.3, NaN, 0.5]))

  # an infeasible predictor with a positive weight: the model has no coefficient there, and
  # q is left as it was (O6); this is not a precision failure
  f1 = GLMLoss(IB, y3, I3; weights = [1.0, 2.0, 1.0])
  q = quadratic_model(f1, [0.3, 0.6, 0.5])
  qy, qw = copy(q.y), copy(q.weights)
  for bad in (1.7, -0.4, 1.0 + 1e-12, 0.0)
    β3 = [0.4, bad, 0.45]
    e = try quadratic_model!(q, f1, β3); nothing catch e e end
    @test e isa ArgumentError && !(e isa PB.PrecisionError)
    @test occursin("observation 2", e.msg) && occursin("unchanged", e.msg)
    @test q.y == qy && q.weights == qw
    @test_throws ArgumentError quadratic_model(f1, β3)
  end
  # the first infeasible observation is the one named, even after a feasible one
  e = try quadratic_model(GLMLoss(IB, y3, I3), [0.3, 0.6, 2.0]); nothing catch e e end
  @test e isa ArgumentError && occursin("observation 3", e.msg)
  # a predictor that is not finite propagates, as for the other losses
  @test all(isnan, quadratic_model(GLMLoss(IB, [0.5], ones(1, 1)), [NaN]).weights)
  @test quadratic_model(GLMLoss(IB, [0.5], ones(1, 1)), [Inf]) isa GLMLoss
  # another response at an endpoint of the response is a domain error at the other endpoint
  @test_throws ArgumentError quadratic_model(GLMLoss(IB, [1.0], ones(1, 1)), [0.0])
  @test_throws ArgumentError quadratic_model(GLMLoss(IB, [0.0], ones(1, 1)), [1.0])
  # the other losses keep their errors: a Poisson overflow is a PrecisionError
  @test PB._irls_domain(PB.Logistic(), 3.0, 1.0) && PB._irls_domain(PB.Poisson(), -3.0, 1.0)
  @test_throws PB.PrecisionError quadratic_model(GLMLoss(PB.Poisson(), Float32[1], ones(Float32, 1, 1)), Float32[89])
end

@testset "IdentityBernoulli: curvature, precisions, allocations and devices" begin
  Random.seed!(227)
  n, p = 12, 3
  X = 0.1 .+ 0.4 .* rand(n, p)
  β = [0.2, 0.3, 0.1]
  o = 0.05 .* rand(n)
  w = 0.5 .+ rand(n)
  y = [0.0, 1.0, 0.2, 0.7, 0.0, 1.0, 0.5, 0.0, 1.0, 0.35, 0.9, 0.1]
  f = GLMLoss(IB, y, X; weights = w, offset = o)
  # no global gradient bound
  @test lipschitz(f) == Inf
  @test lipschitz(GLMLoss(IB, y, sparse(X))) == Inf
  @test all(k -> coordinate_lipschitz(f, k) == Inf, 1:p)
  @test coordinate_lipschitz!(zeros(p), f) == fill(Inf, p)
  @test_throws BoundsError coordinate_lipschitz(f, 0)
  @test_throws BoundsError coordinate_lipschitz(f, p + 1)
  @test lipschitz(GLMLoss(IB, y, X; weights = w)) isa Float64

  # Float32 data, and Float32 responses with Float64 data (the working precision is the
  # promotion, T3)
  η = X * β .+ o
  v, d1, d2 = ib_references(η, y)
  f32 = GLMLoss(IB, Float32.(y), Float32.(X); weights = Float32.(w), offset = Float32.(o))
  β32 = Float32.(β)
  @test f32(β32) isa Float32 && isapprox(f32(β32), sum(w .* v); rtol = 1e-4)
  g32 = similar(β32)
  @test value_and_gradient!(g32, f32, β32) isa Float32 && eltype(g32) == Float32
  @test g32 ≈ X' * (w .* d1) rtol = 1e-4
  @test gradient!(similar(β32), f32, β32) isa Vector{Float32}
  @test eltype(hessian_vec(f32, β32, ones(Float32, p))) == Float32
  @test eltype(quadratic_model(f32, β32).weights) == Float32
  @test_throws ArgumentError quadratic_model(f32, Float32[0.2, 0.3, 4])
  y32 = Float32.(y)
  fm = GLMLoss(IB, y32, X; weights = Float32.(w), offset = o)
  @test fm.η isa Vector{Float64}
  ηm = X * β .+ o
  vm, d1m, _ = ib_references(ηm, Float64.(y32))
  @test fm(β) isa Float64 && isapprox(fm(β), sum(Float64.(Float32.(w)) .* vm); rtol = 1e-12)
  @test gradient(fm, β) ≈ X' * (Float64.(Float32.(w)) .* d1m) rtol = 1e-12

  # the oracles allocate nothing once compiled
  for T in (Float32, Float64)
    fa = GLMLoss(IB, T.(y), T.(X); weights = T.(w), offset = T.(o))
    βa, ga = T.(β), zeros(T, p)
    @test allocs(ib_value, fa, βa) == 0
    @test allocs(ib_vgrad!, ga, fa, βa) == 0
    @test allocs(ib_grad!, ga, fa, βa) == 0
  end

  # a device array: the check of the responses and the value and gradient oracles (the
  # second-order ones loop on the host and are not device oracles of GLMLoss)
  allowscalar(false)
  for T in (Float32, Float64)
    fc = GLMLoss(IB, T.(y), T.(X); weights = T.(w), offset = T.(o))
    fg = Adapt.adapt(JLArray, fc)
    @test fg.X isa JLArray && fg.y isa JLArray && fg.η isa JLArray && fg.d isa JLArray
    βc = T.(β)
    βg = JLArray(βc)
    @test fg(βg) ≈ fc(βc)
    gc, gg = similar(βc), similar(βg)
    @test value_and_gradient!(gg, fg, βg) ≈ value_and_gradient!(gc, fc, βc) && Array(gg) ≈ gc
    fill!(gg, 0)
    @test gradient!(gg, fg, βg) === gg && Array(gg) ≈ gradient(fc, βc)
    # a masked observation outside the domain is skipped on the device too, and a response
    # outside [0, 1] is rejected when the loss is built or rebuilt there
    fz = GLMLoss(IB, T.(y), T.(X); weights = T.([0; ones(n - 1)]), offset = T.([2; o[2:end]]))
    @test isfinite(Adapt.adapt(JLArray, fz)(βg)) && Adapt.adapt(JLArray, fz)(βg) ≈ fz(βc)
    @test_throws ArgumentError(ib_response_message) GLMLoss(IB, JLArray(T[0, 2, 1]), JLArray(ones(T, 3, 1)))
    @test_throws ArgumentError(ib_response_message) GLMLoss(IB, JLArray(T[0, NaN, 1]), JLArray(ones(T, 3, 1));
                                                            weights = JLArray(T[1, 0, 1]))
    @test GLMLoss(IB, JLArray(T[0, 1, 0.5]), JLArray(ones(T, 3, 1))) isa GLMLoss
  end
end

end
