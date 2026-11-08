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

# ---- SqrHingeLoss value and slope: ew + 2 (er + e) and ew + ey + er + e ----

# SqrHingeLoss([2], [2^W]) at x = [-2^TOP]: the margin 1 + 2^(TOP+1) overflows and is kept as r 2^e with e = TOP + 3
# and r = 1/4 + 2^-e. The value w r^2 2^(2e) is 2^(W + 2 EMAX); the slope 2 w |y| r 2^e is 2^(W + TOP + 3).
# In the default range the Int total of the value is W + 1 + 2 (TOP + 2) = 2^63 + W + 1, that of the slope is W + TOP + 5.
sqrhinge_value_n(W) = W + 2EMAX
sqrhinge_slope_n(W) = W + TOP + 3

@testset "SqrHingeLoss value, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    if EMAX == 2^62 - 1
      @test (-2 + 1) + 2 * (TOP + 2) == typemax(Int64)
      @test (-1 + 1) + 2 * (TOP + 2) == Int128(typemax(Int64)) + 1
    end
    for (ysign, W) in ((1, BOT), (1, BOT + 1), (1, -1000), (1, -2), (1, -1), (1, 0), (1, 1), (1, TOP), (-1, -2), (-1, -1), (-1, TOP))
      f = SqrHingeLoss(BigFloat[2ysign], BigFloat[pw(W)])
      check_result(f(BigFloat[pw(TOP, -ysign)]), 1, sqrhinge_value_n(W))
      g = zeros(BigFloat, 1)
      check_result(value_and_gradient!(g, f, BigFloat[pw(TOP, -ysign)]), 1, sqrhinge_value_n(W))
      check_result(g[1], -ysign, sqrhinge_slope_n(W))
    end
  end
end

@testset "SqrHingeLoss gradient, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    if EMAX == 2^62 - 1
      @test (TOP - 2 + TOP + 5) == typemax(Int64)
      @test (TOP - 1 + TOP + 5) == Int128(typemax(Int64)) + 1
    end
    for (ysign, W) in ((1, BOT), (1, -1000), (1, 0), (1, TOP - 3), (1, TOP - 2), (1, TOP - 1), (1, TOP), (-1, TOP - 2), (-1, TOP - 1), (-1, TOP))
      f = SqrHingeLoss(BigFloat[2ysign], BigFloat[pw(W)])
      g = zeros(BigFloat, 1)
      gradient!(g, f, BigFloat[pw(TOP, -ysign)])
      check_result(g[1], -ysign, sqrhinge_slope_n(W))
    end
  end
end

@testset "SqrHingeLoss with Float64 data keeps its sums" begin
  # x = -2^1023: the margin 1 + 2^1024 overflows, the value is 2^(W + 2048), the slope 2^(W + 1026)
  for W in (-1074, -1000, 0, 1023)
    f = SqrHingeLoss([2.0], [Base.ldexp(1.0, W)])
    g = zeros(1)
    check_result(f([-2.0^1023]), 1, W + 2048)
    check_result(value_and_gradient!(g, f, [-2.0^1023]), 1, W + 2048)
    check_result(g[1], -1, W + 1026)
    f = SqrHingeLoss([-2.0], [Base.ldexp(1.0, W)])
    gradient!(g, f, [2.0^1023])
    check_result(g[1], 1, W + 1026)
  end
  for T in (Float64, Float32)
    f = SqrHingeLoss(T[2], T[3])
    @test f(T[-5]) isa T
    @test @inferred(PB._sqrhinge_slope(T(2), T(3), T(11), 0)) isa T
    @test @inferred(PB._sqrhinge_value(T(3), T(0.25), 7)) isa T
  end
end

# ---- SqrHingeLoss prox and conjugate value: frames of one function ----

# a zero result: the sign of a zero that is added to another term (the conjugate value is a sum that starts at +0) or
# computed from a zero significand (the prox at v = 0) is not asserted
function check_value(r, s, n)
  if r isa BigFloat ? n <= EMIN - 3 : n <= -1076
    @test iszero(r)
  else
    check_result(r, s, n)
  end
end

# prox of SqrHingeLoss([sy 2^Y], [2^W]) at v = 0 with the step 2^G is 2 γ w y / (1 + 2 γ w y^2), that is sy 2^N / (1 + 2^D)
# with N = 1 + G + W + Y and D = 1 + G + W + 2Y; for |D| far from zero the result is sy 2^(N - max(D, 0))
function sqrhinge_prox_exponent(G, W, Y)
  N, D = 1 + G + W + Y, 1 + G + W + 2Y
  @assert abs(D) > 400
  N - max(D, 0)
end

# the cases reach _sqrhinge_prox: h = γ w |y| or h |y| leaves the normal range
const PROX_CASES = Tuple{Int128, Int128, Int128}[
  (0, 0, TOP - 7), (0, 0, TOP - 1), (0, 0, TOP), (TOP, TOP, 0), (TOP, TOP, TOP), (BOT, BOT, -TOP), (TOP, 0, TOP ÷ 2),
  (BOT, TOP, TOP)]

@testset "SqrHingeLoss prox, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    if EMAX == 2^62 - 1      # et = (G + 1) + (W + 1) + 2 (Y + 1) of (0, 0, Y)
      @test 2 * (TOP - 1) + 4 == typemax(Int64) - 1
      @test 2 * TOP + 4 == Int128(typemax(Int64)) + 1
    end
    for (G, W, Y) in PROX_CASES, ysign in (1, -1)
      out = zeros(BigFloat, 1)
      prox!(out, SqrHingeLoss(BigFloat[pw(Y, ysign)], BigFloat[pw(W)]), BigFloat[0], pw(G))
      check_value(out[1], ysign, sqrhinge_prox_exponent(G, W, Y))
    end
  end
end

@testset "SqrHingeLoss prox, Float64 argument with BigFloat parameters, precision $p" for p in (5, 53)
  setprecision(BigFloat, p) do
    for (G, W, Y) in PROX_CASES, ysign in (1, -1)
      out = zeros(1)
      prox!(out, SqrHingeLoss(BigFloat[pw(Y, ysign)], BigFloat[pw(W)]), [0.0], pw(G))
      check_value(out[1], ysign, sqrhinge_prox_exponent(G, W, Y))
    end
  end
end

@testset "SqrHingeLoss prox with Float64 data keeps its sums" begin
  for (G, W, Y) in ((0, 0, 1000), (600, 600, 0), (0, 0, 600), (-600, -600, -600), (300, 0, 500)), ysign in (1, -1)
    out = zeros(1)
    prox!(out, SqrHingeLoss([ysign * Base.ldexp(1.0, Y)], [Base.ldexp(1.0, W)]), [0.0], Base.ldexp(1.0, G))
    check_value(out[1], ysign, sqrhinge_prox_exponent(G, W, Y))
  end
end

# Conjugate(SqrHingeLoss([2^Y], [2^W]))([-2^S]) is r + r^2 / (4 w) with r = -2^(S - Y): the terms -2^n1 and 2^n2,
# n1 = S - Y and n2 = 2 n1 - 2 - W. Far apart (gap), the larger one is the value.
function conjugate_exponent(S, Y, W, gap = 400)
  n1 = S - Y
  n2 = 2n1 - 2 - W
  @assert abs(n1 - n2) > gap
  n2 > n1 ? (1, n2) : (-1, n1)
end

# (S, Y, W) of the cases: the total 2 eq - ew - e of the second shift first leaves Int64 at eq = EMAX + 2 for W = BOT (the
# first two cases), the total 2 eq - ew first leaves it below at eq = typemin(Int64) / 2 for W = 0 (the next two)
function conjugate_cases()
  out = Tuple{Int128, Int128, Int128}[]
  a_wrap = Int128(2)^63 + BOT + 1
  for a in (a_wrap - 1, a_wrap, a_wrap + 1000, EMAX + 5)
    abs(a) <= TOP - BOT && push!(out, (TOP, TOP - a, BOT))
  end
  a = Int128(typemin(Int64)) ÷ 2
  for W in (-1, 0, 1, 5)
    abs(a) <= TOP - BOT && push!(out, (BOT, BOT - a, W))
  end
  # a tiny quotient s / y, whose square 2^(2 n1) wraps as well
  push!(out, (BOT + 2, (EMAX + 1) ÷ 2, BOT), (BOT + 2, (EMAX + 1) ÷ 2, TOP), (BOT, TOP, BOT))
  # a huge weight and an underflowing quotient
  push!(out, (BOT + 3, 0, TOP), (TOP, TOP - 2000, BOT + 3))
  out
end

@testset "SqrHingeLoss conjugate value, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    for (S, Y, W) in conjugate_cases()
      sgn, n = conjugate_exponent(S, Y, W)
      f = Conjugate(SqrHingeLoss(BigFloat[pw(Y)], BigFloat[pw(W)]))
      check_value(f(BigFloat[pw(S, -1)]), sgn, n)
    end
  end
end

# the working type is the promoted type of the argument and the parameters, not that of the argument: a Float64 s
# with BigFloat y and w has eq = 1000 - EMIN + ..., and 2 eq leaves Int64
@testset "SqrHingeLoss conjugate value, Float64 argument with BigFloat parameters, precision $p" for p in (5, 53)
  setprecision(BigFloat, p) do
    S = 1000
    for (Y, W) in ((EMIN, BOT), (EMIN, BOT + 3), (EMIN, -5), (EMIN + 1, BOT), (TOP, BOT), (EMIN - 1, BOT))
      sgn, n = conjugate_exponent(S, Y, W)
      f = Conjugate(SqrHingeLoss(BigFloat[pw(Y)], BigFloat[pw(W)]))
      check_value(f([-2.0^S]), sgn, n)
    end
  end
end

@testset "SqrHingeLoss conjugate value with Float64 data keeps its sums" begin
  for (S, Y, W) in ((1000, -500, -3), (10, 0, 100), (-1000, 1000, 10), (1000, 1000, 100), (-500, 500, 1000))
    sgn, n = conjugate_exponent(S, Y, W, 60)
    f = Conjugate(SqrHingeLoss([Base.ldexp(1.0, Y)], [Base.ldexp(1.0, W)]))
    check_value(f([-Base.ldexp(1.0, S)]), sgn, n)
  end
end

# ---- _scaled_square_value: 2k - 1 ----

# SqrNormL2(2^L)([2^q]) is 2^(L + 2q - 1). The residual exponent is k = q, and the Int total 2k - 1 first wraps at
# q = BOT, whose true total is -2^63 - 1 in the default range
@testset "SqrNormL2 value, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    if EMAX == 2^62 - 1
      @test 2BOT - 1 < typemin(Int64)
      @test 2 * (BOT + 1) - 1 >= typemin(Int64)
    end
    for (L, q) in ((0, BOT + 1), (0, BOT), (0, BOT + 2), (0, BOT + 5), (TOP, BOT), (BOT, TOP), (0, TOP), (TOP, TOP), (BOT, BOT))
      f = SqrNormL2(pw(L))
      check_result(f(BigFloat[pw(q)]), 1, L + 2q - 1)
      check_result(f(BigFloat[pw(q), pw(q)]), 1, L + 2q)      # two entries: the sum 2 2^(2q), halved
    end
    check_result(SqrNormL2(1)(BigFloat[pw(BOT)]), 1, 2BOT - 1)
  end
end

@testset "SqrNormL2 value of Float64 data keeps its sums" begin
  check_result(SqrNormL2(1.0)([5.0e-324]), 1, -2149)
  check_result(SqrNormL2(1.0)([2.0^600]), 1, 1199)
  check_result(SqrNormL2(2.0^-1000)([2.0^-600]), 1, -2201)
  check_result(SqrNormL2(2.0^900)([2.0^-600]), 1, -301)       # s = 2^-1200 underflows: the scaled sum is used
  check_result(SqrNormL2(2.0^900)([2.0^-600, 2.0^-600]), 1, -300)
end

# ---- DistL2: k + e of the value and e + j + k of the blend ----

# the box {p} with p = -2^TOP in n coordinates; the projection of x = 2^TOP ones is p, at the distance sqrt(n) 2^(TOP+1)
distl2_box(n) = IndBox([pw(TOP, -1) for _ in 1:n], [pw(TOP, -1) for _ in 1:n])

# DistL2(box, 2^le)(2^TOP ones(n)) is 2^(le + TOP + dexp) with the distance 2^(TOP + dexp) (n = 1: dexp = 1, n = 4: dexp = 2).
# The Int total k + e is (le + 1) + (TOP + dexp + 1). The distance of four coordinates is a chain of hypot, whose rounding
# is not asserted to give exactly 2^(TOP + 2), so the four-coordinate cases are those whose value is infinite.
@testset "DistL2 value, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    total(le, dexp) = le + TOP + dexp + 2
    if EMAX == 2^62 - 1
      @test total(TOP - 1, 2) == typemax(Int64)
      @test total(TOP, 2) == Int128(typemax(Int64)) + 1
      @test total(TOP, 1) == typemax(Int64)
    end
    for (n, dexp, les) in ((1, 1, (BOT, 0, TOP - 3, TOP - 1, TOP)), (4, 2, (0, TOP - 3, TOP - 1, TOP)))
      for le in les
        f = DistL2(distl2_box(n), pw(le))
        check_result(f([pw(TOP) for _ in 1:n]), 1, le + TOP + dexp)
      end
    end
    # value_and_prox! with a step that leaves the point where it is: the value is that of the same distance
    for le in (TOP, TOP - 1)
      x, y = [pw(TOP) for _ in 1:4], [pw(TOP) for _ in 1:4]
      v = value_and_prox!(y, DistL2(distl2_box(4), pw(le)), x, pw(BOT))
      check_result(v, 1, le + TOP + 2)
      @test y == x
    end
  end
end

# The point [A, 2^-q] and the box {[-A, 0]} (A = 2^TOP): the distance is 2^(TOP+1) (frexp exponent de = TOP + 2). With
# γ = 2^(ge-1) and λ = 2^(le-1) the step γλ/dist has the exponent te - de = ge + le - 1 - de = tau. For tau below -2^62
# the step does not move the point, and the blend adds ldexp(m r, e + j + k) to each coordinate, where e = 1 + tau, and
# j = 1 - q, k = 0 for the last coordinate (j = TOP + 1, k = 1 for the first). The Int total of the last coordinate is
# tau + 2 - q, which first wraps at tau - q = -2^63 - 3; te - de itself wraps below tau = -2^63
function distl2_prox_case(tau, q)
  de = TOP + 2
  gesum = tau + de + 1
  ge = fld(gesum, 2)
  le = gesum - ge
  (EMIN <= ge <= EMAX && EMIN <= le <= EMAX) || return false
  point() = [pw(TOP), pw(-q)]
  lower() = [pw(TOP, -1), zero(BigFloat)]
  f = DistL2(IndBox(lower(), lower()), pw(le - 1))
  x, y = point(), point()
  prox!(y, f, x, pw(ge - 1))
  @test all(isfinite, y)
  @test y == x
  true
end

@testset "DistL2 prox, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    tau_min = 2EMIN - 1 - (TOP + 2)        # ge = le = EMIN
    taus = Int128[-Int128(2)^63 + 7, -Int128(2)^63, -Int128(2)^63 - 1, -Int128(2)^63 - 1000, -Int128(2)^62, tau_min, tau_min + 5]
    ran = 0
    for tau in taus, q in (2, 3, 6, 12, 20)
      distl2_prox_case(tau, q) && (ran += 1)
    end
    @test ran >= 10
  end
end

# a Float64 point with BigFloat parameters: de = 1002, so te - de = 2 EMIN - 1 - 1002 leaves Int64 on the 64-bit range
@testset "DistL2 with a Float64 point and BigFloat parameters, precision $p" for p in (5, 53)
  setprecision(BigFloat, p) do
    x = [2.0^1000, 2.0^-3]
    lower() = [-2.0^1000, 0.0]
    f = DistL2(IndBox(lower(), lower()), pw(BOT))
    y = copy(x)
    prox!(y, f, x, pw(BOT))
    @test all(isfinite, y)
    @test y == x
    # the value k + e = (le + 1) + 1002 does not leave Int64; it is rounded to Float64
    for (le, n) in ((TOP, TOP + 1001), (-2000, -999), (BOT, BOT + 1001))
      v = DistL2(IndBox(lower(), lower()), pw(le))([2.0^1000, 0.0])
      check_value(v, 1, n)
    end
  end
end

@testset "DistL2 with Float64 data keeps its sums" begin
  # one coordinate whose difference to the box, 2^1024, overflows Float64
  for (le, n) in ((1023, 2047), (-1000, 24), (-1074, -50))
    check_result(DistL2(IndBox([-2.0^1023], [-2.0^1023]), Base.ldexp(1.0, le))([2.0^1023]), 1, n)
  end
  lower() = [-2.0^1023, 0.0]
  for g in (-1000, -1074)
    x = [2.0^1023, 2.0^-3]
    y = copy(x)
    prox!(y, DistL2(IndBox(lower(), lower()), Base.ldexp(1.0, g)), x, Base.ldexp(1.0, g))
    @test y == x
  end
end

end
