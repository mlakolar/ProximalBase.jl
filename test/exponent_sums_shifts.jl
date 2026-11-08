module ExponentSumsShiftsTest

# MPFR exponents reach ±(2^62 - 1) on 64-bit Unix, so the sum of three or more exponents can leave Int64. The functions
# below form the exponent of a final shift, or a frame that one function builds from exact inputs, and that total must
# reach ldexp without wrapping. The cases are built from exponents relative to the platform's own MPFR range, so the
# same body runs where the range is 2^30 - 1 (Windows). There the sums of these cases stay within Int64: a run there
# exercises none of the wide paths and is evidence of no regression, not of the repair. The expected values come from
# exponent totals formed in Int128, not from a package function.

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase

const EMAX = Int128(Base.MPFR.get_emax())
const EMIN = Int128(Base.MPFR.get_emin())
const TOP = EMAX - 1       # the largest n with 2^n finite
const BOT = EMIN - 1       # the smallest n with 2^n nonzero
const PRECISIONS = (2, 5, 53, 256)

# a signed power of two, exact at the precisions used here
function pw(p::Integer, s::Integer = 1)
  @assert BOT <= p <= TOP
  s * Base.ldexp(one(BigFloat), Clong(p))
end

# the result 2^n with the sign s, by an oracle that does not use the package: class and sign first, then the value
function check_result(r::BigFloat, s::Integer, n::Integer)
  neg = s < 0
  if n >= EMAX
    @test isinf(r)
    @test signbit(r) == neg
  elseif n <= EMIN - 3
    @test iszero(r)
    @test signbit(r) == neg
  elseif n >= EMIN - 1
    @test isfinite(r) && !iszero(r)
    @test signbit(r) == neg
    @test r == pw(n, s)
  end   # n == EMIN - 2 is a tie of MPFR's own rule
end

# the same for Float64, whose range is 2^-1074 to 2^1023
function check_result(r::Float64, s::Integer, n::Integer)
  neg = s < 0
  if n >= 1024
    @test isinf(r)
    @test signbit(r) == neg
  elseif n <= -1076
    @test iszero(r)
    @test signbit(r) == neg
  elseif n >= -1074
    @test isfinite(r) && !iszero(r)
    @test signbit(r) == neg
    @test r == s * Base.ldexp(1.0, Int(n))
  end   # n == -1075 is a tie
end

@testset "the power of a scaled product" begin
  @test PB._product_power(BigFloat, 5) === 5
  @test PB._product_power(BigFloat, Int128(5)) === Int128(5)
  @test PB._product_power(BigFloat, Int128(2)^70) === Int128(2)^70
  for T in (Float64, Float32)
    @test PB._product_power(T, 5) === 5
    @test @inferred(PB._product_power(T, Int32(5))) === 5
    @test PB._product_power(T, Int128(5)) === 5
  end
  @test_throws InexactError PB._product_power(Float64, Int128(2)^70)
end

@testset "scaled product with a power beyond Int64, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    beyond = Int128(typemax(Int64)) + 5
    for s in (1, -1)
      # w v 2^k: the saturated value, not an InexactError
      for k in (beyond, -beyond, Int128(2)^100, -Int128(2)^100)
        for (a, b) in ((0, 0), (TOP, TOP), (BOT, BOT), (TOP, BOT))
          w, v = pw(a), pw(b, s)
          check_result(PB._scaled_product(w, v, k), s, a + b + k)
        end
      end
      # a wide power that brings a product of two huge factors back into the range
      k = -Int128(2)^63 - 10
      check_result(PB._scaled_product(pw(TOP), pw(TOP, s), k), s, 2TOP + k)
      check_result(PB._scaled_product(pw(BOT), pw(BOT, s), -k), s, 2BOT - k)
      # a b c 2^k
      for k in (beyond, -beyond, Int128(2)^100, -Int128(2)^100)
        for (a, b, c) in ((0, 0, 0), (TOP, TOP, TOP), (BOT, BOT, BOT), (TOP, BOT, TOP))
          check_result(PB._scaled_product(pw(a), pw(b, s), pw(c), k), s, a + b + c + k)
        end
      end
      k = -Int128(2)^63 - 10
      check_result(PB._scaled_product(pw(TOP), pw(TOP, s), pw(TOP), k), s, 3TOP + k)
      check_result(PB._scaled_product(pw(BOT), pw(BOT, s), pw(BOT), -k), s, 3BOT - k)
    end
    @test @inferred(PB._scaled_product(pw(1), pw(1), Int128(5))) isa BigFloat
    @test @inferred(PB._scaled_product(pw(1), pw(1), pw(1), Int128(5))) isa BigFloat
  end
end

@testset "scaled product of Float64 keeps its Int power" begin
  @test PB._scaled_product(2.0^1000, 2.0^1000, -1000) == 2.0^1000
  @test PB._scaled_product(2.0^1000, -2.0^1000, Int32(-1000)) == -2.0^1000
  @test PB._scaled_product(2.0^1000, 2.0^1000, 2.0^-1000, Int32(-5)) == 2.0^995
  @test_throws InexactError PB._scaled_product(2.0^1000, 2.0^1000, Int128(2)^70)
  @test_throws InexactError PB._scaled_product(2.0^1000, 2.0^1000, 2.0, Int128(2)^70)
  for T in (Float64, Float32)
    @test @inferred(PB._scaled_product(T(3), T(0.25), 2)) isa T
    @test @inferred(PB._scaled_product(T(3), T(0.25), T(-7), 2)) isa T
  end
end

# ---- _image_product: ea + eb + scale[2] ----

# the product a b 2^s of two powers of two, the scale exponent s last
@testset "image product at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    # the last scale exponent for which 2 (BOT + 1) + s fits Int64, and the first for which it does not
    s_fit = Int128(typemin(Int64)) - 2BOT - 2
    for sg in (1, -1), s in (s_fit, s_fit - 1, s_fit - 1000, Int128(-TOP))
      check_result(PB._image_product(pw(BOT), pw(BOT, sg), (one(BigFloat), Int(s))), sg, 2BOT + s)
    end
    for sg in (1, -1)
      check_result(PB._image_product(pw(TOP), pw(TOP, sg), (one(BigFloat), Int(-2TOP))), sg, 0)
      check_result(PB._image_product(pw(TOP), pw(TOP, sg), (one(BigFloat), Int(-2TOP + 3))), sg, 3)
      check_result(PB._image_product(pw(TOP), pw(BOT, sg), (one(BigFloat), 5)), sg, TOP + BOT + 5)
    end
  end
end

# the orthogonal image of NormL1 under L = [1 0; 0 1; 0 t], t = 2^(BOT + d), at x = [2^TOP, 2^-10, 0]: the
# product t z2 2^-k has the exponent total (BOT + d + 1) - 9 - TOP, which is -2^63 + d - 6 in the default range, and
# the exact value is |z|_1 = 2^TOP + 2^-10
@testset "image value, product total at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    total(d) = BOT + d + 1 - 9 - TOP
    if EMAX == 2^62 - 1
      @test total(5) < typemin(Int64) && total(6) == typemin(Int64)
    end
    for d in (0, 5, 6, 7, 40)
      L = zeros(BigFloat, 3, 2)
      L[1, 1] = 1
      L[2, 2] = 1
      L[3, 2] = pw(BOT + d)
      h = LinearImage(NormL1(), L, 1; check = false)
      r = h(BigFloat[pw(TOP), pw(-10), 0])
      @test isfinite(r)
      @test r == pw(TOP) + pw(-10)
    end
  end
end

# the orthogonal image with L = 2^S [1 0; 0 1; 0 0] + t at (3, 2) and μ = 2^TOP, at x = [2^(S+2), 2^(S+1), 2^-10]: the
# latent point is [4, 2], its prox at the step 1 is [3, 1], and the image of that is [3 2^S, 2^S, t]. The product
# t x3 / μ has the exponent total (BOT + d + 1) - 9 - (TOP + 1), which is -2^63 + d - 7 in the default range
@testset "image prox, product total at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    S = TOP ÷ 2
    mu = pw(TOP)
    total(d) = BOT + d + 1 - 9 - (TOP + 1)
    if EMAX == 2^62 - 1
      @test total(6) < typemin(Int64) && total(7) == typemin(Int64)
    end
    for d in (0, 6, 7, 8, 40)
      t = pw(BOT + d)
      L = zeros(BigFloat, 3, 2)
      L[1, 1] = pw(S)
      L[2, 2] = pw(S)
      L[3, 2] = t
      h = LinearImage(NormL1(), L, mu; check = false)
      x = BigFloat[pw(S + 2), pw(S + 1), pw(-10)]
      y = zeros(BigFloat, 3)
      prox!(y, h, x, mu)
      @test all(isfinite, y)
      @test y == [3 * pw(S), pw(S), t]
      y2 = zeros(BigFloat, 3)
      v = value_and_prox!(y2, h, x, mu)
      @test isfinite(v)
      @test v == 4
      @test y2 == y
    end
  end
end

# the quadratic image, whose right-hand side L'x overflows: Q = I, γ = 1, G = I + L'L = 2^TOP I, so the latent point
# is [2^S, 2^(S-5)] and the image is [2^TOP, 2^(TOP-5), 2^(BOT+S-5)] (t = 2^BOT). The scale of the right-hand side is
# k = S + 6, so the product of t and x3 = 2^(BOT+delta) has the total (BOT+1) + (BOT+delta+1) - (S+6)
# (the factorization of G at 2 or 5 bits fails the pivot test of _image_factor_ok, so the cases run at 53 and 256 bits)
@testset "image quadratic prox, product total at the Int64 boundary, precision $p" for p in (53, 256)
  setprecision(BigFloat, p) do
    S = TOP ÷ 2
    total(delta) = (BOT + 1) + (BOT + delta + 1) - (S + 6)
    delta_fit = Int128(typemin(Int64)) - 2BOT - 2 + S + 6        # the first delta whose total fits Int64
    @test total(delta_fit) == typemin(Int64) && total(delta_fit - 1) < typemin(Int64)
    deltas = filter(d -> 0 <= d <= TOP - BOT, [Int128(0), delta_fit - 1, delta_fit, delta_fit + 100])
    @test 0 in deltas
    for delta in deltas
      L = zeros(BigFloat, 3, 2)
      L[1, 1] = pw(S)
      L[2, 2] = pw(S)
      L[3, 2] = pw(BOT)
      h = LinearImage(QuadraticFunction(Matrix{BigFloat}(I, 2, 2)), L)
      y = zeros(BigFloat, 3)
      prox!(y, h, BigFloat[pw(TOP), pw(TOP - 5), pw(BOT + delta)], 1)
      @test all(isfinite, y)
      @test y[1] ≈ pw(TOP) rtol = 1e-12
      @test y[2] ≈ pw(TOP - 5) rtol = 1e-12
      @test y[3] ≈ pw(BOT + S - 5) rtol = 1e-12
    end
  end
end

@testset "image with Float64 data keeps its sums" begin
  @test @inferred(PB._image_product(2.0^100, 2.0^-3, (1.0, -5))) === 2.0^92
  @test @inferred(PB._image_product(-2.0^100, 2.0^-3, (0.5, 5))) === -2.0^(100 - 3 + 5 - 1)
  h = LinearImage(NormL1(), [1.0 0.0; 0.0 1.0; 0.0 5.0e-324], 1)
  @test h([2.0^1000, 2.0^-10, 0.0]) == 2.0^1000 + 2.0^-10
  S, mu = 500, 2.0^1000
  L = [2.0^S 0.0; 0.0 2.0^S; 0.0 5.0e-324]
  h = LinearImage(NormL1(), L, mu)
  x = [2.0^(S + 2), 2.0^(S + 1), 2.0^-10]
  y = zeros(3)
  prox!(y, h, x, mu)
  @test y == [3 * 2.0^S, 2.0^S, 5.0e-324]
  @test value_and_prox!(zeros(3), h, x, mu) == 4
end

end
