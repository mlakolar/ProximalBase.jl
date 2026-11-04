module HingeStepOracleTest
using Test, ProximalBase, Random
using ..TestUtils

# The step γw|y|² of HingeLoss (issue #519, PR A): an independent high-precision test of the ordinary
# and the exceptional (range-recovering) coefficient paths. The oracle is the exact product in
# BigFloat (512 bits, exact for three 53-bit factors and a square) rounded once to S. Expected
# accuracy: the step is within 2 ulp of S (three roundings in the promotion R of S and the factors,
# then one to S); beyond the range of S it is Inf or zero as the exact value says, except within
# 4 ulp of floatmax, where a 2-ulp error may decide between floatmax and Inf. The decisions of the
# proximal operator are tested where the step equals the oracle's, at the boundary u = 1 - s and
# one ulp on either side of it.

const step = ProximalBase._hinge_step

# number of representable steps between two finite values of one type, for same-sign arguments
ulps(a::T, b::T) where {T<:AbstractFloat} = abs(Int64(reinterpret(Base.uinttype(T), abs(a))) - Int64(reinterpret(Base.uinttype(T), abs(b))))

function exact(γ, w, y)
  setprecision(BigFloat, 512) do
    big(γ) * big(w) * big(y)^2
  end
end

# a draw with the target exponent of the step spread over γ, w and y² in an adversarial way
function draw(rng, ::Type{S}, ::Type{G}, ::Type{W}, lo, hi) where {S,G,W}
  T = lo + (hi - lo) * rand(rng)                 # log2 of the step
  a = (rand(rng) - 0.5) * 2 * (hi - lo) * 0.9    # log2 γ, then w, then y² takes the rest
  b = (rand(rng) - 0.5) * 2 * (hi - lo) * 0.9
  c = T - a - b
  rg(t, ::Type{X}) where {X} = convert(X, ldexp(1 + rand(rng), clamp(floor(Int, t), exponent(floatmin(X)), exponent(floatmax(X)) - 1)))
  (rg(a, G), rg(b, W), rg(c / 2, S))
end

function compare(::Type{S}, γ, w, y) where {S}
  e = exact(γ, w, y)
  o = S(e)
  g = step(S, γ, abs(y), w)
  if isinf(o) || isinf(g)
    isinf(o) && isinf(g) && return true
    # an Inf on one side only: the exact step is within 4 ulp of floatmax
    return abs(e - big(floatmax(S))) <= 4 * big(eps(floatmax(S)))
  end
  if iszero(o)
    return iszero(g) || ulps(g, o) <= 2
  end
  ulps(g, o) <= 2
end

@testset "step γw|y|² against the exact product" begin
  rng = MersenneTwister(519)
  for (S, G, W, lo, hi, n) in ((Float64, Float64, Float64, -1090, 1040, 6000), (Float32, Float32, Float32, -160, 135, 6000),
                               (Float64, Float32, Float32, -1090, 1040, 3000), (Float32, Float64, Float64, -160, 135, 3000),
                               (Float64, Float64, Float32, -1090, 1040, 3000), (Float32, Float32, Float64, -160, 135, 3000),
                               (Float16, Float64, Float64, -28, 20, 2000), (Float16, Float16, Float16, -28, 20, 2000),
                               # far beyond the range: the exact step is Inf (above) or zero (below)
                               (Float64, Float64, Float64, 1000, 3200, 3000), (Float32, Float32, Float32, 120, 420, 3000),
                               (Float64, Float32, Float32, 1000, 3200, 2000), (Float16, Float16, Float16, 14, 70, 2000),
                               (Float64, Float64, Float64, -3300, -1070, 3000), (Float32, Float32, Float32, -450, -150, 3000))
    bad = 0; tested = 0
    for _ in 1:n
      d = draw(rng, S, G, W, lo, hi)
      γ, w, y = d
      tested += 1
      ok = compare(S, γ, w, y)
      ok || (bad += 1; bad <= 3 && @info "step mismatch" S G W γ w y step(S, γ, abs(y), w) S(exact(γ, w, y)))
    end
    @test tested == n
    @test bad == 0
  end
end

@testset "subnormal and adversarial single factors" begin
  # a subnormal weight with |y| < 1: the weight is not rounded before the step is formed
  rng = MersenneTwister(5190)
  bad = 0
  for _ in 1:4000
    w = reinterpret(Float64, rand(rng, UInt64(1):(UInt64(2)^52 - UInt64(1))))
    γ = 2.0^rand(rng, 1:70) * (1 + rand(rng))
    y = 0.05 + 0.95 * rand(rng)
    bad += !compare(Float64, γ, w, y)
  end
  @test bad == 0
  bad = 0
  for _ in 1:4000
    w = reinterpret(Float32, rand(rng, UInt32(1):(UInt32(2)^23 - UInt32(1))))
    γ = 2f0^rand(rng, 1:40) * (1 + rand(rng, Float32))
    y = 0.05f0 + 0.95f0 * rand(rng, Float32)
    bad += !compare(Float32, γ, w, y)
  end
  @test bad == 0
  # exponents whose split would leave the range of S
  @test step(Float64, 2.0^600, 2.0^500, 2.0^600) == Inf
  @test step(Float64, 2.0^1023, 2.0^3, 2.0^1023) == Inf
  @test step(Float64, 2.0^-1000, 2.0^-500, 2.0^-1000) == 0.0
  @test step(Float64, 2.0^-1074, 2.0^500, 2.0^926) == 2.0^852      # γ, |y|, w: no factor is rounded early
end

@testset "special values: weight and step conventions" begin
  for S in (Float16, Float32, Float64)
    inf, big_, tiny = S(Inf), floatmax(S), nextfloat(zero(S))
    for y in (S(0.5), S(3), tiny * 4, big_ / 4)
      @test step(S, S(0), y, inf) == inf            # an infinite weight gives Inf at a zero step
      @test step(S, inf, y, S(0)) == 0              # a zero weight gives 0 at an infinite step
      @test step(S, S(0), y, big_) == 0             # a zero step is zero although w|y| overflows
      @test step(S, inf, y, tiny) == inf            # an infinite step is Inf although w|y| underflows
      @test step(S, S(2), y, inf) == inf
      @test isnan(step(S, S(NaN), y, S(1))) && isnan(step(S, S(1), y, S(NaN)))
    end
  end
end

@testset "decisions of the proximal operator at the boundary u = 1 - s" begin
  rng = MersenneTwister(51900)
  checked = 0
  for S in (Float32, Float64), _ in 1:3000
    γ = ldexp(S(1) + rand(rng, S), rand(rng, -20:20)); w = ldexp(S(1) + rand(rng, S), rand(rng, -20:20))
    y = (rand(rng) < 0.5 ? -1 : 1) * ldexp(S(1) + rand(rng, S), rand(rng, -10:10))
    s = step(S, γ, abs(y), w)
    s == S(exact(γ, w, y)) || continue           # decisions only where the step is the correctly rounded one
    isfinite(s) && 0 < s < 4 || continue
    # the margin u = y v at 1 - s, one ulp below and above: v is chosen so that y v rounds to those
    for u in (prevfloat(1 - s), 1 - s, nextfloat(1 - s))
      v = u / y
      uu = y * v                                  # the margin as the implementation forms it
      want = (uu >= 1 ? uu : (uu < 1 - s ? uu + s : one(S))) / y
      f = HingeLoss([y], [w])
      @test prox_entry(f, v, 1, γ) === want
      checked += 1
    end
  end
  @test checked > 1000
end

end
