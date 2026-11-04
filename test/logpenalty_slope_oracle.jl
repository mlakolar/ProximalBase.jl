module LogPenaltySlopeOracleTest
using Test, ProximalBase, Random
using ..TestUtils

# The slope λw/(ε + |v|) of LogPenalty (issue #519, PR A): lla_weight, lla_weights! and the KKT
# diagnostic against an independent BigFloat reference (512 bits) rounded once to the model
# precision U, for finite nonzero factors. The slope is a quotient and a product in the promotion R
# of U and the factors, then rounded to U: within 2 ulp of U (3 for the two roundings plus the
# conversion where R is wider), and beyond the range of U the reference is Inf or zero. A result
# within 4 ulp of floatmax may be floatmax or Inf. The draws spread the exponents of λ, w, ε and t
# over the whole range, so that ε + t overflows R, or λ/(ε + t) underflows R before w restores it.

const PB = ProximalBase

reference(λ, w, ε, t) = setprecision(() -> big(λ) * big(w) / (big(ε) + big(t)), BigFloat, 512)

# distance of got from the exact r in units of the spacing of U at r; Inf when the range decision is wrong
function errulps(::Type{U}, got, r) where {U}
  if r > big(floatmax(U))
    return (isinf(got) || r < big(floatmax(U)) * (1 + ldexp(big(1.0), -(precision(U) - 4)))) ? 0.0 : Inf
  end
  isinf(got) && return Inf
  sp = r < big(floatmin(U)) ? big(nextfloat(zero(U))) : ldexp(big(1.0), exponent(U(r)) - precision(U) + 1)
  Float64(abs(big(got) - r) / sp)
end

# a positive finite number of type T with the binary exponent x, clamped into the range of T
function pow2(rng, ::Type{T}, x) where {T}
  lo, hi = exponent(nextfloat(zero(T))), exponent(floatmax(T))
  e = clamp(round(Int, x), lo, hi)
  e <= exponent(floatmin(T)) - 1 && return ldexp(T(1 + rand(rng)) / 2, e + 1)
  ldexp(T(1 + rand(rng)), e)
end

function worst(rng, ::Type{U}, ::Type{F}, n, span) where {U, F}
  w = 0.0
  bad = Any[]
  for _ in 1:n
    r = span * (2rand(rng) - 1)
    λ, ε, wt = (pow2(rng, F, span * (2rand(rng) - 1)) for _ in 1:3)
    t = rand(rng) < 0.2 ? zero(U) : pow2(rng, U, r)
    f = LogPenalty(λ, ε, [wt])
    got = PB.lla_weight(f, t, 1)
    ref = reference(λ, wt, ε, t)
    e = errulps(U, got, ref)
    got isa U || push!(bad, (:type, λ, wt, ε, t))
    e <= 3 || push!(bad, (e, λ, wt, ε, t))
    w = max(w, e)
  end
  w, bad
end

@testset "LogPenalty slope: the cases of the review" begin
  a = 2.0^1023
  @test PB.lla_weight(LogPenalty(a, a), a, 1) === 0.5                       # ε + t overflows
  @test PB.lla_weight(LogPenalty(2.0^-100, 2.0^-100, [2.0^200]), 2.0^1023, 1) === 2.0^-923   # λ/(ε+t) underflows, w restores
  @test PB.lla_weight(LogPenalty(a, a, [a]), a, 1) === 2.0^1022
  @test PB.lla_weight(LogPenalty(a, a, [a]), -a, 1) === 2.0^1022            # the sign of v is irrelevant
  p, m = 2.0^800, 2.0^-800
  @test PB.lla_weight(LogPenalty(p, p, [p]), 0.0, 1) === p                  # the earlier cases: λw overflows / underflows,
  @test PB.lla_weight(LogPenalty(m, m, [m]), 0.0, 1) === m                  # the quotient by ε restores it
  # Float32: the same in the single range, and with Float64 factors the slope is formed in Float64
  b = 2f0^127
  @test PB.lla_weight(LogPenalty(b, b), b, 1) === 0.5f0
  @test PB.lla_weight(LogPenalty(2f0^-60, 2f0^-60, Float32[2f0^100]), 2f0^127, 1) === 2f0^-87
  @test PB.lla_weight(LogPenalty(2.0^-100, 2.0^-100, [2.0^200]), 2f0^127, 1) === Float32(2.0^-100 * 2.0^200 / (2.0^-100 + 2.0^127))
  # KKT at t = 0: the denominator is ε alone
  @test PB.kkt_violation(LogPenalty(a, a, [a]), [a], 1) === a - a
  @test PB.kkt_violation(LogPenalty(2.0^-100, 2.0^-1000, [2.0^200]), [2.0^1000], 1) === 2.0^1000 - 2.0^1100
end

@testset "LogPenalty slope: special values" begin
  f(w) = LogPenalty(1.0, 1.0, [w])
  @test PB.lla_weight(f(Inf), 3.0, 1) === Inf
  @test isnan(PB.lla_weight(f(Inf), NaN, 1))
  @test PB.lla_weight(f(0.0), 3.0, 1) === 0.0
  @test PB.lla_weight(LogPenalty(0.0, 1.0, [2.0]), 3.0, 1) === 0.0
  @test isnan(PB.lla_weight(f(2.0), NaN, 1))
  @test PB.lla_weight(f(2.0), Inf, 1) === 0.0                                 # finite λw over an infinite |v|
  @test PB.lla_weight(f(2.0), -Inf, 1) === 0.0
  @test PB.lla_weight(f(2.0), 3.0, 1) === 2.0 / 4.0
  tiny = nextfloat(0.0)
  @test PB.lla_weight(LogPenalty(tiny, tiny, [2.0^1000]), 0.0, 1) === 2.0^1000   # subnormal level and ε
  @test PB.lla_weight(LogPenalty(1.0, tiny), 0.0, 1) === Inf                 # 1/2^-1074 = 2^1074 overflows
  @test PB.lla_weight(LogPenalty(1.0, 1.0, [2.0^-1074]), 1.0, 1) === 0.0     # 2^-1075 ties to even: zero
  @test PB.lla_weight(LogPenalty(1.0, 1.0, [3 * 2.0^-1074]), 1.0, 1) === 2.0^-1073  # 1.5·2^-1074 rounds to 2·2^-1074
  @test PB.lla_weight(LogPenalty(1.0, 1.0), 2, 1) === 1 / 3                  # an integer argument
end

@testset "LogPenalty slope against BigFloat" begin
  rng = MersenneTwister(519)
  for (U, F, span) in ((Float64, Float64, 1100), (Float32, Float32, 150), (Float32, Float64, 400), (Float64, Float32, 400))
    w, bad = worst(rng, U, F, 6000, span)
    @test isempty(bad)
    isempty(bad) || @show U F first(bad, 3)
  end
end

@testset "LogPenalty lla_weights! agrees with the scalar slope" begin
  rng = MersenneTwister(520)
  for (T, X, F) in ((Float32, Float32, Float32), (Float64, Float32, Float32), (Float32, Float64, Float64), (Float64, Float64, Float64), (Float32, Float32, Float64))
    for n in (1, 7, 256), pos in (:first, :mid, :last, :none)
      f = LogPenalty(F(2)^-3, F(2)^-4, F[F(0.5) + rand(rng, F) for _ in 1:n])
      x = X[X(0.1) + 3rand(rng, X) for _ in 1:n]
      if pos != :none
        i = pos === :first ? 1 : pos === :mid ? (n + 1) ÷ 2 : n
        ex = X === Float64 ? 2.0^1023 : 2f0^127
        x[i] = ex
        w = copy(f.w)
        w[i] = F === Float64 ? 2.0^200 : F(2)^100
        # the level and weight are small, ε + |v| is huge: λw/(ε + |v|) restores a range
        f = LogPenalty(F === Float64 ? 2.0^-100 : F(2)^-60, F === Float64 ? 2.0^-100 : F(2)^-60, w)
      end
      W = promote_type(T, X)
      want = T[convert(T, PB.lla_weight(f, convert(W, x[k]), k)) for k in 1:n]
      out = zeros(T, n)
      @test PB.lla_weights!(out, f, x) === out
      @test out == want
      y = copy(x); z = T === X ? y : nothing
      z === nothing || (@test PB.lla_weights!(z, f, y) == want)       # the argument is the destination
      @test PB.lla_weights(f, x) isa Vector
    end
  end
  # nonfinite entries among ordinary ones
  f = LogPenalty(1.0, 1.0, [1.0, Inf, 0.0, 2.0, 1.0])
  x = [1.0, 2.0, 3.0, Inf, NaN]
  out = PB.lla_weights!(zeros(5), f, x)
  @test out[1] === 0.5 && out[2] === Inf && out[3] === 0.0 && out[4] === 0.0 && isnan(out[5])
end

end
