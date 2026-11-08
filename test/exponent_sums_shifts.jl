module ExponentSumsShiftsTest

# MPFR exponents reach ±(2^62 - 1) on 64-bit Unix, so the sum of three or more exponents can leave Int64. The functions
# below form the exponent of a final shift, or a frame that one function builds from exact inputs, and that total must
# reach ldexp without wrapping. The public reproducers are built from exponents relative to the platform's own MPFR
# range, so the same body runs where the range is 2^30 - 1 (Windows). There their exponent totals stay within Int64: those
# cases are evidence of no regression, not of the repair. The cases built from explicit wide integers (the powers of
# _scaled_product beyond Int64, the scale exponents of the _image_product cases taken from typemin(Int64), the exponents
# of the _pow2_sum cases taken from typemax(Int64), the Int128 powers of _power_factors, _comparison_exponent and
# _ExtremeOutput) cross Int64 on that range as well. The expected values come from exponent totals formed in Int128, not
# from a package function.

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
      # a wide power k = -2^63 - 10 with two huge factors: the total is -14 (2TOP + k) or 10 (2BOT - k) on the 64-bit Unix
      # range, a finite value. On the Windows range EMAX = 2^30 - 1 it lies outside the range and the value saturates; a
      # range between the two can give a finite total as well, and check_result takes the class from the range
      k = -Int128(2)^63 - 10
      check_result(PB._scaled_product(pw(TOP), pw(TOP, s), k), s, 2TOP + k)
      check_result(PB._scaled_product(pw(BOT), pw(BOT, s), -k), s, 2BOT - k)
      if EMAX == Int128(2)^62 - 1
        # a finite result m from the powers k = m - 2TOP (-2^63 for m = -4, the last that fits Int64, then -2^63 - 1) and
        # k = m - 2BOT (typemax(Int64) for m = -1, then typemax(Int64) + 1)
        for m in (-4, -5, -6, BOT, TOP)
          check_result(PB._scaled_product(pw(TOP), pw(TOP, s), m - 2TOP), s, m)
        end
        for m in (-1, 0, 1, TOP)
          check_result(PB._scaled_product(pw(BOT), pw(BOT, s), m - 2BOT), s, m)
        end
      end
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

# (S, Y, W) of the cases, with eq = S - Y and ew = W + 1. For W = BOT and eq above ew the frame is e = eq2 = 2 eq - ew,
# so the second shift eq2 - e is 0 and the first shift is eq - e = ew - eq. In the default range the frame eq2 is the last
# that fits Int64 at eq = 2^61 and first leaves it at eq = 2^61 + 1, and the first shift is typemin(Int64) at
# eq = EMAX + 2 (the last that fits) and typemin(Int64) - 1 at eq = EMAX + 3. The positive cases are those eq and
# neighbours of them. The next group has eq = typemin(Int64) / 2, where eq2 = 2 eq - ew fits for ew = 0 (W = -1) and
# is below Int64 for ew >= 1 (W = 0, 1, 5).
function conjugate_cases()
  out = Tuple{Int128, Int128, Int128}[]
  a_shift = BOT + 1 - Int128(typemin(Int64))        # the last eq whose first shift ew - eq fits Int64
  a_frame = fld(Int128(typemax(Int64)) + BOT + 1, 2) + 1    # the first eq whose frame 2 eq - ew exceeds typemax(Int64)
  for a in (a_frame - 1, a_frame, a_shift - 1, a_shift, a_shift + 1, a_shift + 1000, EMAX + 5)
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

# the working type is the promoted type of the argument and the parameters, not that of the argument. With a Float64 s and
# BigFloat y and w, the cases with Y at or next to EMIN have eq = 1000 - Y of about 2^62, so the frame 2 eq - ew leaves
# Int64 in the default range; (TOP, BOT) has eq = 1000 - TOP, whose double fits Int64, and is a control that must give the
# same value. Where the range is 2^30 - 1 the frames of the listed cases fit Int64.
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

# ---- SumLargest: a + b + c of the final shift ----

# SumLargest(k, 2^le)(sg 2^TOP ones(k)) is sg k 2^(le + TOP). The shift a + b + c is exponent(max |x|) = TOP, the frexp
# exponent le + 1 of λ, and the frexp exponent c of the scaled sum k (c = 2, 2, 3, 4 for k = 2, 3, 4, 8): in the
# default range its total first leaves Int64 at k = 4 for le = TOP
@testset "SumLargest value, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    total(le, c) = TOP + (le + 1) + c
    if EMAX == 2^62 - 1
      @test total(TOP, 2) == typemax(Int64)
      @test total(TOP, 3) == Int128(typemax(Int64)) + 1
    end
    for (k, c) in ((2, 2), (4, 3), (8, 4)), le in (BOT, 0, TOP - 3, TOP - 1, TOP), sg in (1, -1)
      r = SumLargest(k, pw(le))([pw(TOP, sg) for _ in 1:k])
      check_result(r, sg, le + TOP + c - 1)
    end
    # k = 3: 3 2^(le + TOP) is not a power of two
    for sg in (1, -1)
      r = SumLargest(3, pw(BOT))([pw(TOP, sg) for _ in 1:3])
      @test isfinite(r) && r == 3 * pw(BOT + TOP, sg)
      for le in (0, TOP - 1, TOP)
        r = SumLargest(3, pw(le))([pw(TOP, sg) for _ in 1:3])
        @test isinf(r) && signbit(r) == (sg < 0)
      end
    end
    # the k largest of a longer array
    for le in (TOP - 1, TOP)
      r = SumLargest(4, pw(le))(vcat([pw(TOP) for _ in 1:4], [pw(0), pw(1)]))
      @test isinf(r) && r > 0
    end
  end
end

@testset "SumLargest with Float64 or mixed data, precision $p" for p in (5, 53)
  setprecision(BigFloat, p) do
    # Float64 data and a BigFloat level: the sum 2^1025 overflows Float64, the working type is BigFloat
    for (le, n) in ((TOP, TOP + 1025), (-2000, -975), (BOT, BOT + 1025))
      check_value(SumLargest(4, pw(le))(fill(2.0^1023, 4)), 1, n)
    end
    # BigFloat data and a Float64 level
    check_result(SumLargest(4, 2.0^-1000)([pw(TOP) for _ in 1:4]), 1, TOP + 2 - 1000)
    check_result(SumLargest(4, 2.0^1000)([pw(TOP, -1) for _ in 1:4]), -1, TOP + 1002)
  end
end

@testset "SumLargest with Float64 data keeps its sums" begin
  for (k, c) in ((2, 2), (4, 3), (8, 4)), (le, sg) in ((1023, 1), (1023, -1), (-1000, 1), (-1000, -1), (-1074, 1))
    check_result(SumLargest(k, Base.ldexp(1.0, le))(fill(sg * 2.0^1023, k)), sg, le + 1023 + c - 1)
  end
end

# ---- _pow2_weighted: the frame of the sum of terms m 2^e ----

# the terms (1/2, e1) and (1/2, e2) with e2 = e1 - 2^63 and e2 - 1, where e1 = EMAX - 7: the difference D - E of the
# frame is -2^63 (the last that fits Int64) and -2^63 - 1 (the first that does not). The sum is 2^(e1 - 1) + 2^(e2 - 1),
# which rounds to 2^(e1 - 1)
@testset "pow2 sum, difference of the frame at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    e1 = Int(EMAX) - 7
    for e2 in (e1 - typemax(Int) - 1, e1 - typemax(Int) - 2, e1 - typemax(Int) - 1000)
      check_result(PB._pow2_sum(BigFloat, (big(0.5), e1), (big(0.5), e2)), 1, e1 - 1)
      check_result(PB._pow2_sum(BigFloat, (big(0.5), e2), (big(0.5), e1)), 1, e1 - 1)
      check_result(PB._pow2_sum(BigFloat, (big(-0.5), e1), (big(0.5), e2)), -1, e1 - 1)
    end
    # two equal terms: the sum 1 2^D has the frame D + 1, which is 2^63 for D = typemax(Int)
    for D in (typemax(Int) - 2, typemax(Int) - 1, typemax(Int))
      check_result(PB._pow2_sum(BigFloat, (big(0.5), D), (big(0.5), D)), 1, Int128(D))
      check_result(PB._pow2_sum(BigFloat, (big(-0.5), D), (big(-0.5), D)), -1, Int128(D))
    end
    for D in (typemin(Int), typemin(Int) + 1)
      check_result(PB._pow2_sum(BigFloat, (big(0.5), D), (big(0.5), D)), 1, Int128(D))
    end
    # the weight is applied through _scaled_product: a wide frame saturates and does not throw
    @test PB._pow2_weighted(BigFloat, big(3), (big(0.5), typemax(Int)), (big(0.5), typemax(Int))) == Inf
    @test PB._pow2_sum(BigFloat, (big(0.5), 3), (big(0.25), 5)) == 12
  end
end

# LinearFunction(c, d)(x) with c = sg 2^a ones(n) and x = 2^b ones(n) overflows: the recovery sums the scaled dot
# p 2^k and d. The frame D = t + 1 for the true exponent t = a + b + log2(n) of the dot first leaves Int64 at
# t = 2^63 - 1 (n = 8 at a = b = TOP) and the sum is +-Inf (the dot is at least 2^(2 TOP), above EMAX).
@testset "LinearFunction value, frame at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    if EMAX == 2^62 - 1
      @test 2TOP + 2 == typemax(Int64) - 1       # t of n = 4
      @test 2TOP + 3 == Int128(typemax(Int64))   # t of n = 8
    end
    for n in (1, 2, 4, 8), (a, b) in ((TOP, TOP), (TOP, TOP - 1), (TOP - 1, TOP - 1)), sg in (1, -1)
      t = a + b + trailing_zeros(n)
      for d in (big(0), big(1), big(-1), pw(TOP, sg), pw(BOT))
        check_result(LinearFunction([pw(a, sg) for _ in 1:n], d)([pw(b) for _ in 1:n]), sg, t)
      end
    end
    fm = floatmax(BigFloat)
    check_result(LinearFunction(fill(fm, 4), big(0))(fill(fm, 4)), 1, 2TOP + 4)
    check_result(LinearFunction(fill(fm, 5), big(1))(fill(fm, 5)), 1, 2TOP + 4)
    check_result(LinearFunction(fill(-fm, 4), big(0))(fill(fm, 4)), -1, 2TOP + 4)
    # products of both signs that overflow and cancel: the value is d
    c = [pw(TOP), pw(TOP), pw(TOP, -1), pw(TOP, -1)]
    @test LinearFunction(c, big(3))([pw(TOP), pw(TOP, -1), pw(TOP), pw(TOP, -1)]) == 3
    @test LinearFunction([pw(TOP), pw(TOP)], big(3))([pw(TOP), pw(TOP, -1)]) == 3
  end
end

@testset "pow2 sum and LinearFunction with Float64 data keep their sums" begin
  @test @inferred(PB._pow2_sum(Float64, (0.5, 3), (0.25, 5))) === 12.0
  @test @inferred(PB._pow2_sum(Float32, (0.5f0, 3), (0.25f0, 5))) === 12.0f0
  @test PB._pow2_sum(Float64, (0.5, 1000), (0.5, 1000)) == 2.0^1000
  @test PB._pow2_sum(Float64, (0.5, 1000), (0.5, -1000)) == 2.0^999
  @test PB._pow2_sum(Float64, (0.5, 1100), (0.5, 1100)) == Inf
  @test PB._pow2_sum(Float64, (-0.5, 1100), (-0.5, 1100)) == -Inf
  fm = floatmax(Float64)
  @test LinearFunction(fill(fm, 4), 0.0)(fill(fm, 4)) == Inf
  @test LinearFunction(fill(-fm, 5), 1.0)(fill(fm, 5)) == -Inf
  @test LinearFunction([2.0^600, 2.0^600], 3.0)([2.0^500, -2.0^500]) == 3.0
  @test LinearFunction([2.0^600, 2.0^600], 3.0)([2.0^500, 2.0^500]) == Inf
end

# ---- the GLM derivative of SquaredHinge keeps its Int arithmetic ----

# _weighted_deriv(SquaredHinge(), w, ηs, k, y) takes the margin exponent of _sqrhinge_margin, an Int that can already
# have wrapped. Its Int total is right modulo 2^64, so a wide sum there would turn a case that is right into a wrong one.
# The table holds the values of the unchanged Int sum, as (class, significand of frexp, exponent), for the margin
# (sign of y, exponent of |y|, sign of ηs, exponent of |ηs|, k) at the weights 2^W, W in GLM_PIN_WEIGHTS. The exponents
# are those of the 64-bit Unix range, so the table is checked only there.
const GLM_PIN_WEIGHTS = (-4611686018427387904, -2, -1, 0, 1, 4611686018427387902)
const GLM_PIN = [
  ((1, 0, -1, 0, 5), [(:fin, -0.515625, -4611686018427387897), (:fin, -0.515625, 5), (:fin, -0.515625, 6), (:fin, -0.515625, 7), (:fin, -0.515625, 8), (:ninf, 1.0, 0)]),
  ((1, 0, -1, 0, 9223372036854775805), [(:fin, -0.5, 4611686018427387903), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:nzero, 0.0, 0), (:nzero, 0.0, 0), (:nzero, 0.0, 0)]),
  ((1, 0, -1, 0, 9223372036854775806), [(:ninf, 1.0, 0), (:ninf, 1.0, 0), (:nzero, 0.0, 0), (:nzero, 0.0, 0), (:nzero, 0.0, 0), (:nzero, 0.0, 0)]),
  ((1, 0, -1, 0, 9223372036854775807), [(:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0)]),
  ((-1, 0, 1, 0, 9223372036854775806), [(:inf, 1.0, 0), (:inf, 1.0, 0), (:zero, 0.0, 0), (:zero, 0.0, 0), (:zero, 0.0, 0), (:zero, 0.0, 0)]),
  ((1, 0, -1, 1048576, 9223372036854775807), [(:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0)]),
  ((1, 0, -1, 4611686018427387902, 9223372036854775807), [(:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0)]),
  ((1, -4611686018427387904, -1, -4611686018427387904, -2), [(:fin, -0.5, 0), (:fin, -0.5, 4611686018427387902), (:fin, -0.5, 4611686018427387903), (:ninf, 1.0, 0), (:ninf, 1.0, 0), (:ninf, 1.0, 0)]),
  ((1, -4611686018427387904, -1, -4611686018427387904, -3), [(:fin, -0.5, -1), (:fin, -0.5, 4611686018427387901), (:fin, -0.5, 4611686018427387902), (:fin, -0.5, 4611686018427387903), (:ninf, 1.0, 0), (:ninf, 1.0, 0)]),
  ((-1, -4611686018427387904, 1, -4611686018427387904, -3), [(:fin, 0.5, -1), (:fin, 0.5, 4611686018427387901), (:fin, 0.5, 4611686018427387902), (:fin, 0.5, 4611686018427387903), (:inf, 1.0, 0), (:inf, 1.0, 0)]),
]

function glm_pin_class(v::BigFloat)
  isnan(v) && return (:nan, 0.0, 0)
  isinf(v) && return (v > 0 ? :inf : :ninf, 1.0, 0)
  iszero(v) && return (signbit(v) ? :nzero : :zero, 0.0, 0)
  m, e = Base.frexp(v)
  (:fin, Float64(m), Int(e))
end

if EMAX == Int128(2)^62 - 1
  @testset "GLM SquaredHinge derivative keeps the Int sum of its margin exponent, precision 53" begin
    setprecision(BigFloat, 53) do
      for ((sy, py, sη, pη, k), expected) in GLM_PIN
        y, ηs = pw(py, sy), pw(pη, sη)
        for (W, want) in zip(GLM_PIN_WEIGHTS, expected)
          v = PB._weighted_deriv(PB.SquaredHinge(), pw(W), ηs, k, y)
          @test glm_pin_class(v) == want
        end
      end
    end
  end
end

# ---- KronQuadratic tier 1: the sum exponent(mn) + k of _scale_exponent! ----

# KronQuadratic(A, B)(Θ) with A = 2^pa [0 1; 1 0], B = 2^pb [0 1; 1 0] and Θ = diag(2^pM, 2^pt) is ½ ⟨Θ, A Θ B⟩ =
# 2^(pa + pb + pM + pt). For pM = TOP and pb = BOT the product A Θ overflows and the value is recomputed in tiers. In these
# calls tier 1 scales Θ by 2^-kd1 with kd1 = (pM + 1) + (pa + 5): the bound exponent of Θ and δ = -(pa + 1 + 1 + 3).
# _scale_exponent! then compares exponent(2^pt) + k, k = -kd1, with exponent(floatmin) = BOT: the entry 2^pt is lost by the
# scaling when pt - kd1 < BOT, and then tier 2 recomputes the value. The Int total pt - kd1 fits Int64 from
# pt_fit = typemin(Int64) + kd1 on.
function kron_values(pa, pM, pt, pb)
  a, b = pw(pa), pw(pb)
  f = KronQuadratic(BigFloat[0 a; a 0], BigFloat[0 b; b 0])
  Θ = BigFloat[pw(pM) 0; 0 pw(pt)]
  f(Θ), value_and_gradient!(zeros(BigFloat, 2, 2), f, Θ)
end

# the exponents pt of the cases that exist in the range of the platform
function kron_exponents(pt_fit)
  filter(t -> BOT <= t <= TOP, Int128[pt_fit - 1000, pt_fit - 1, pt_fit, pt_fit + 1, pt_fit + 100, BOT, BOT + 7, 0, TOP])
end

@testset "KronQuadratic value, scale exponent at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    pa = (TOP + 2) ÷ 2
    pt_fit = Int128(typemin(Int64)) + TOP + pa + 6
    pts = kron_exponents(pt_fit)
    if EMAX == 2^62 - 1
      @test pa == 2^61 && pt_fit == -Int128(2)^61 + 4
      @test (pt_fit - 1) in pts && pt_fit in pts
    end
    for pt in pts
      n = pa + BOT + TOP + pt
      v, w = kron_values(pa, TOP, pt, BOT)
      check_result(v, 1, n)
      check_result(w, 1, n)
    end
  end
end

@testset "scale exponent of BigFloat data at the Int64 boundary, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    pa = (TOP + 2) ÷ 2
    kd1 = TOP + pa + 6
    k = Int(-kd1)
    for pt in kron_exponents(Int128(typemin(Int64)) + kd1)
      X = BigFloat[pw(TOP) 0; 0 pw(pt)]
      lossless = PB._scale_exponent!(X, BigFloat, k)
      survives = pt + Int128(k) >= BOT
      @test lossless == survives
      if lossless && survives
        @test X[1, 1] == pw(TOP + k)
        @test X[2, 2] == pw(pt + k)
      end
    end
  end
end

@testset "KronQuadratic with Float64 data keeps its sums" begin
  f = KronQuadratic([0.0 2.0^500; 2.0^500 0.0], [0.0 2.0^-1000; 2.0^-1000 0.0])
  Θ = [2.0^1000 0.0; 0.0 2.0^-1000]
  check_result(f(Θ), 1, -500)
  check_result(value_and_gradient!(zeros(2, 2), f, Θ), 1, -500)
  for (T, hi, lo, k_fit, k_lost) in ((Float64, 1000, -1000, -10, -30), (Float32, 100, -100, -10, -30))
    X = T[Base.ldexp(one(T), hi) 0; 0 Base.ldexp(one(T), lo)]
    Y = copy(X)
    @test @inferred(PB._scale_exponent!(Y, T, k_fit)) === true       # lo + k_fit >= exponent(floatmin(T))
    @test Y == T[Base.ldexp(one(T), hi + k_fit) 0; 0 Base.ldexp(one(T), lo + k_fit)]
    @test PB._scale_exponent!(copy(X), T, k_lost) === false          # lo + k_lost < exponent(floatmin(T))
  end
end

# ---- _exp_bound: the type constant of the exponent bound in the GLM losses ----

# The bound is (emax + 2 + precision - emin) log 2 for the exponents of floatmax and floatmin. Its integer factor is
# 2^63 + precision for BigFloat on the 64-bit Unix range, a sum that leaves Int64. _logistic_tail and _poisson_exp compare
# the predictor with it. They run in the recovery of a derivative, which starts when the ordinary derivative of a weight
# above one is below the normal range (for BigFloat, zero): Logistic has it at η = 0, y = 1/2 and Poisson at η = 0, y = 1,
# and both at a response that equals the computed σ(η) or exp(η).
@testset "exp bound of BigFloat, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    total = TOP + 2 + p - BOT
    b = PB._exp_bound(BigFloat)
    @test b > 0
    @test b == BigFloat(Float64(total) * 0.6931471805599453)
    @test isapprox(b, big(total) * log(big(2)); rtol = 2.0^(1 - p) + 1e-15)
    if EMAX == 2^62 - 1
      @test total == Int128(2)^63 + p
    end
  end
end

@testset "exp bound of Float64 and Float32" begin
  @test PB._exp_bound(Float64) === 2100 * 0.6931471805599453
  @test PB._exp_bound(Float32) === Float32(279 * 0.6931471805599453)
  @test @inferred(PB._exp_bound(Float64)) isa Float64
end

glm_one(loss, y, x, w) = GLMLoss(loss, BigFloat[y], reshape(BigFloat[x], 1, 1); weights = BigFloat[w])
glm_gradient(f, β) = (g = zeros(BigFloat, 1); gradient!(g, f, BigFloat[β]); g[1])
glm_value_and_gradient(f, β) = (g = zeros(BigFloat, 1); v = value_and_gradient!(g, f, BigFloat[β]); (v, g[1]))

# w ℓ'(η, y) and w ℓ(η, y) for the stored data, from 4 precision + 200 bits, with the scale of the rounding of a
# computation in the precision of the data: 4 2^-p w (e^η + |y| + |y η|) for Poisson and 4 2^-p w (1 + |y| + |y η|) for Logistic
function glm_reference(loss::Symbol, w, η, y, value::Bool)
  p = precision(BigFloat)
  setprecision(BigFloat, 4p + 200) do
    W, H, Y = big(w), big(η), big(y)
    ex = exp(H)
    ref = if loss === :poisson
      value ? W * (ex - Y * H) : W * (ex - Y)
    else
      value ? W * (log1p(exp(-abs(H))) + max(H, 0) - Y * H) : W * (1 / (1 + exp(-H)) - Y)
    end
    scale = loss === :poisson ? ex + abs(Y) + abs(Y * H) : 1 + abs(Y) + abs(Y * H)
    ref, 4 * 2.0^-p * W * scale
  end
end

function check_glm(r::BigFloat, loss, w, η, y, value::Bool)
  ref, tol = glm_reference(loss, w, η, y, value)
  @test isfinite(r)
  isfinite(r) && @test setprecision(() -> abs(r - ref) <= tol, BigFloat, 4 * precision(BigFloat) + 200)
end

@testset "GLM at an exactly zero derivative, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    for w in (2, 3, 1024)
      f = glm_one(PB.Logistic(), 0.5, 1, w)
      @test iszero(glm_gradient(f, 0))
      v, g = glm_value_and_gradient(f, 0)
      @test iszero(g)
      @test isfinite(v) && v > 0
      f = glm_one(PB.Poisson(), 1, 1, w)
      @test iszero(glm_gradient(f, 0))
      v, g = glm_value_and_gradient(f, 0)
      @test iszero(g)
      @test v == w
    end
  end
end

# the response is the prediction as computed in the precision of the data: the exact derivative is a rounding error, and
# the derivative returned is within the scale of that error, not a term of the size w y
@testset "GLM Poisson at a response equal to exp(η), precision $p" for p in (5, 53, 256)
  setprecision(BigFloat, p) do
    ηs = EMAX == 2^62 - 1 ? Any[-1, 1, 20, -20, 2^20, -2^20, 2^40, -2^40, 2^61, -2^61] : Any[-1, 1, 20, -20, 2^20, -2^20]
    for η in ηs, w in (2, 5)
      y = exp(BigFloat(η))
      f = glm_one(PB.Poisson(), y, 1, w)
      check_glm(glm_gradient(f, η), :poisson, w, η, y, false)
      v, g = glm_value_and_gradient(f, η)
      check_glm(g, :poisson, w, η, y, false)
      abs(η) == 1 && check_glm(glm_value_and_gradient(f, η)[1], :poisson, w, η, y, true)
    end
  end
end

@testset "GLM Logistic at a response equal to the sigmoid, precision $p" for p in (5, 53, 256)
  setprecision(BigFloat, p) do
    for η in (-1, -3, -20), w in (2, 5)
      e = exp(BigFloat(η))
      y = e / (1 + e)
      f = glm_one(PB.Logistic(), y, 1, w)
      check_glm(glm_gradient(f, η), :logistic, w, η, y, false)
      check_glm(glm_value_and_gradient(f, η)[2], :logistic, w, η, y, false)
    end
    # η > 0 uses (1 - y) - σ: the response with (1 - y) equal to σ as computed, which exists for some η
    found = 0
    for η in (1, 2, 3)
      e = exp(-BigFloat(η))
      s = e / (1 + e)
      y = nothing
      for cand in (1 - s, nextfloat(1 - s), prevfloat(1 - s))
        ((1 - cand) - s) == 0 && (y = cand; break)
      end
      y === nothing && continue
      found += 1
      f = glm_one(PB.Logistic(), y, 1, 2)
      check_glm(glm_gradient(f, η), :logistic, 2, η, y, false)
    end
    p == 53 && @test found >= 1
  end
end

@testset "GLM with Float64 data keeps its bound" begin
  g = zeros(1)
  gradient!(g, GLMLoss(PB.Logistic(), [0.5], reshape([1.0], 1, 1); weights = [2.0]), [0.0])
  @test g[1] === 0.0
  gradient!(g, GLMLoss(PB.Poisson(), [1.0], reshape([1.0], 1, 1); weights = [2.0]), [0.0])
  @test g[1] === 0.0
  for η in (-1.0, 1.0, 20.0, -20.0)
    y = exp(η)
    gradient!(g, GLMLoss(PB.Poisson(), [y], reshape([1.0], 1, 1); weights = [2.0]), [η])
    @test isfinite(g[1]) && abs(g[1]) <= 8 * 2.0^-52 * 2 * 2 * y
  end
end

# ---- CubeNormL2: the exponent totals of the value, the gradient and the prox ----

# the value c 2^n, with c a significand and the total n formed in Int128: Base.ldexp on the total clamped to a C long
# saturates like the true value
function check_scaled(r::BigFloat, c::BigFloat, n::Integer)
  want = Base.ldexp(c, Clong(clamp(n, typemin(Clong), typemax(Clong))))
  @test isinf(r) == isinf(want)
  @test iszero(r) == iszero(want)
  @test signbit(r) == signbit(want)
  @test r == want
end

# CubeNormL2(2^q)([2^p]) is 2^(q + 3p) / 3. Its norm has the frexp exponent e = p + 1 and λ has k = q + 1, so the Int total
# k + 3e is q + 3p + 4: it is typemax(Int64) at (P1, -3) and the first total beyond Int64 is at (P1, -2); it is typemin(Int64)
# at (-P1, -6) and the first below Int64 is at (-P1, -7), where 3 P1 = 2^63 - 2. A total beyond Int64 means a true value
# above 2^(2^63 - 6) or below 2^(-2^63 + 4): infinite or zero in the range of any MPFR build.
function cube_value_cases()
  P1 = Int128(typemax(Int64) - 1) ÷ 3
  cases = Tuple{Int128, Int128}[(TOP, 0), (TOP, BOT), (BOT, TOP), (BOT, BOT), (TOP ÷ 3, 0), (-(TOP ÷ 3), 0), (100, 100), (0, 0)]
  for q in (-3, -2, -1, BOT, TOP)
    push!(cases, (P1, q))
  end
  for q in (-6, -7, -8, BOT, TOP)
    push!(cases, (-P1, q))
  end
  filter(c -> BOT <= c[1] <= TOP && BOT <= c[2] <= TOP, cases)
end

@testset "CubeNormL2 value, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    cases = cube_value_cases()
    if EMAX == 2^62 - 1
      P1 = Int128(typemax(Int64) - 1) ÷ 3
      @test 3P1 + 4 - 3 == typemax(Int64) && 3P1 + 4 - 2 == Int128(typemax(Int64)) + 1
      @test -3P1 + 4 - 6 == typemin(Int64) && -3P1 + 4 - 7 == Int128(typemin(Int64)) - 1
      @test (P1, -2) in cases && (-P1, -7) in cases
    end
    for (a, q) in cases
      f = CubeNormL2(pw(q))
      x = BigFloat[pw(a)]
      check_scaled(f(x), big(1) / 3, q + 3a)
      v = value_and_gradient!(zeros(BigFloat, 1), f, x)
      check_scaled(v, big(1) / 3, q + 3a)
    end
  end
end

# gradient!(g, CubeNormL2(2^q), [2^a, 2^r]) with 2^r at least 2^400 below 2^a, so that the norm rounds to 2^a at the
# precisions used here: the entries are 2^(q + 2a) and
# 2^(q + a + r). The exponents of the entry i are k_x = a + 1 or r + 1, k = q + 1 of λ and e = a + 1 of the norm, whose
# total k_x + k + e is q + 2a + 3 and q + a + r + 3. The cases are the q with such a total next to typemax(Int64) and to
# typemin(Int64), and q = BOT, 0 and TOP.
function cube_gradient_cases()
  hi, lo = Int128(typemax(Int64)), Int128(typemin(Int64))
  cases = Tuple{Int128, Int128, Int128}[]
  for (a, r) in ((TOP, BOT), (BOT + 400, BOT), (Int128(0), BOT), (TOP, TOP - 400))
    c1, c2 = 2a + 3, a + r + 3
    for q in (hi - c1, hi - c1 + 1, lo - c1, lo - c1 - 1, hi - c2, hi - c2 + 1, lo - c2, lo - c2 - 1, BOT, 0, TOP)
      BOT <= q <= TOP && push!(cases, (a, r, q))
    end
  end
  unique(cases)
end

@testset "CubeNormL2 gradient, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    cases = cube_gradient_cases()
    if EMAX == 2^62 - 1
      @test (TOP, BOT, 0) in cases && (TOP, BOT, 1) in cases
      @test (BOT + 400, BOT, -803) in cases && (BOT + 400, BOT, -804) in cases       # the total of the first entry is typemin(Int64) at q = -803
    end
    for (a, r, q) in cases
      f = CubeNormL2(pw(q))
      x = BigFloat[pw(a), pw(r)]
      g = zeros(BigFloat, 2)
      gradient!(g, f, x)
      check_result(g[1], 1, q + 2a)
      check_result(g[2], 1, q + a + r)
      g2 = zeros(BigFloat, 2)
      v = value_and_gradient!(g2, f, x)
      @test g2 == g
      check_scaled(v, big(1) / 3, q + 3a)
    end
  end
end

# prox!(y, CubeNormL2(2^b), [2^b], 2^b): z = γ λ n = 2^(3b), and y = 2^b 2 / (1 + √(1 + 4z)) is 2^b for b <= -100 and 2^(-b/2)
# for an even b >= 200, at the precisions used here. The Int total of h = k + j + e + i is 3b + 1 (k = j = e = b + 1,
# i = -2): typemax(Int64) at b = P1 and above Int64 from b = P1 + 2 (b is even); on the negative side 3 P1 = 2^63 - 2, so it is
# typemin(Int64) + 3 at b = -P1 (the last even b that fits) and typemin(Int64) - 3 at b = -P1 - 2 (the first that does not).
# On the default 64-bit Unix MPFR range, the mixed-parameter test below includes cases with h = typemin(Int64)
# and h = typemin(Int64) - 1 exactly. On the Windows MPFR range (EMAX = 2^30 - 1), those boundary-hitting candidates
# are filtered out; the two range-relative controls remain.
cube_prox_exponent(b) = b >= 200 ? -(b ÷ 2) : b
cube_prox(γ, λ, x) = (y = similar(x, float(eltype(x))); prox!(y, CubeNormL2(λ), x, γ); y)

function cube_prox_exponents()
  P1 = Int128(typemax(Int64) - 1) ÷ 3
  bs = Int128[TOP, BOT, 200, -100, 2000, -2000, P1, P1 + 2, P1 + 4, P1 - 2, -P1, -P1 - 2, -P1 - 4, -P1 + 2]
  filter(b -> BOT <= b <= TOP, bs)
end

@testset "CubeNormL2 prox, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    bs = cube_prox_exponents()
    if EMAX == 2^62 - 1
      P1 = Int128(typemax(Int64) - 1) ÷ 3
      @test iseven(P1) && 3P1 + 1 == typemax(Int64) && 3 * (P1 + 2) + 1 > typemax(Int64)
      @test -3P1 + 1 > typemin(Int64) && 3 * (-P1 - 2) + 1 < typemin(Int64)
      @test P1 in bs && (P1 + 2) in bs && -P1 in bs && (-P1 - 2) in bs
    end
    for b in bs
      @test iseven(b)
      y = cube_prox(pw(b), pw(b), BigFloat[pw(b)])
      check_result(y[1], 1, cube_prox_exponent(b))
      y = zeros(BigFloat, 1)
      v = value_and_prox!(y, CubeNormL2(pw(b)), BigFloat[pw(b)], pw(b))
      check_result(y[1], 1, cube_prox_exponent(b))
      check_scaled(v, big(1) / 3, b + 3 * cube_prox_exponent(b))
    end
  end
end

# The factor of the prox is 2^-q with q = fld(h, 2). The second coordinate 2^r has the frexp exponent r + 1, and is scaled by
# the total (r + 1) - q, which is below typemin(Int64) for r = BOT from q = BOT + 2 - typemin(Int64) on (2^62 + 2 where the
# range is the 64-bit Unix one, BOT = -2^62; on a smaller range that q is larger); the second coordinate of the
# projection is 2^(r - q), zero in these cases. For the b below the first coordinate is 2^(-b/2) and the second 2^(r - 3b/2).
@testset "CubeNormL2 prox of a tiny entry beside a huge one, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    P1 = Int128(typemax(Int64) - 1) ÷ 3
    cases = filter(c -> BOT <= c[1] <= TOP && BOT <= c[2] <= TOP, Tuple{Int128, Int128}[(TOP, BOT), (TOP, BOT + 1000), (P1 + 2, BOT), (P1, BOT), (200, BOT)])
    @test (TOP, BOT) in cases
    for (b, r) in cases
      y = cube_prox(pw(b), pw(b), BigFloat[pw(b), pw(r)])
      check_result(y[1], 1, -(b ÷ 2))
      check_result(y[2], 1, r - 3 * (b ÷ 2))
    end
  end
end

# a Float64 point x = 2^a with BigFloat parameters γ = 2^g and λ = 2^l: the frexp exponents of γ, λ and of the norm are
# g + 1, l + 1 and a + 1, and that of (1/2)(1/2)(1/2) is -2, so the Int total of h is g + l + a + 1. The prox is x times
# 2 / (1 + √(1 + 4z)) with z = 2^(g + l + a): below the Float64 range for g + l > 4000 at a = 1000, and x for a = -1000 and a
# negligible z. The cases are the l with h = typemax(Int64) (the last that fits) and h = typemax(Int64) + 1, and with
# h = typemin(Int64) and typemin(Int64) - 1.
@testset "CubeNormL2 prox, Float64 point with BigFloat parameters, precision $p" for p in (5, 53)
  setprecision(BigFloat, p) do
    hi, lo = Int128(typemax(Int64)), Int128(typemin(Int64))
    cases = Tuple{Int128, Int128, Int, Float64}[]    # g, l, a, the expected y
    for (g, l, a, want) in ((TOP, TOP, 1000, 0.0), (TOP, hi - TOP - 1002, 1000, 0.0), (TOP, hi - TOP - 1001, 1000, 0.0),
                            (TOP, hi - TOP - 1000, 1000, 0.0),
                            (BOT, BOT, -1000, 2.0^-1000), (BOT, lo - BOT + 999, -1000, 2.0^-1000),
                            (BOT, lo - BOT + 998, -1000, 2.0^-1000), (BOT, lo - BOT + 997, -1000, 2.0^-1000))
      BOT <= g <= TOP && BOT <= l <= TOP && push!(cases, (g, l, a, want))
    end
    @test length(cases) >= 2
    if EMAX == 2^62 - 1
      @test length(cases) == 8
      @test TOP + (hi - TOP - 1001) + 1000 + 1 == hi && BOT + (lo - BOT + 999) - 1000 + 1 == lo
    end
    for (g, l, a, want) in cases
      y = zeros(1)
      prox!(y, CubeNormL2(pw(l)), [Base.ldexp(1.0, a)], pw(g))
      @test y[1] == want
      @test !signbit(y[1])
    end
  end
end

@testset "CubeNormL2 with Float64 data keeps its sums" begin
  @test CubeNormL2(2.0^500)([2.0^500]) === Inf
  @test CubeNormL2(2.0^-500)([2.0^-500]) === 0.0
  @test CubeNormL2(1.0)([2.0^300]) === Base.ldexp(1.0 / 3, 900)
  @test value_and_gradient!(zeros(1), CubeNormL2(1.0), [2.0^300]) === Base.ldexp(1.0 / 3, 900)
  g = zeros(2)
  gradient!(g, CubeNormL2(2.0^-300), [2.0^600, 2.0^-600])        # the norm 2^600 is beyond sqrt(floatmax)
  @test g == [2.0^900, 2.0^-300]
  gradient!(g, CubeNormL2(2.0^500), [2.0^600, 2.0^-600])
  @test g == [Inf, 2.0^500]
  @test cube_prox(2.0^600, 2.0^600, [2.0^600]) == [2.0^-300]
  @test cube_prox(2.0^-600, 2.0^-600, [2.0^-600]) == [2.0^-600]
  @test cube_prox(2.0^300, 2.0^300, [2.0^300]) == [2.0^-150]
  @test cube_prox(2.0^600, 2.0^600, [2.0^600, 2.0^-600]) == [2.0^-300, 0.0]
  for T in (Float64, Float32)
    @test @inferred(PB._cube_value_parts(T(2), T(0.75), 3)) isa T
    @test PB._cube_prox_parts(T(2), T(3), T(0.75), 3)[1] isa T
  end
end

# ---- the exponential cone: the window of _exp_project and the exponential product ----

# Moreau residuals of a candidate projection p of x onto the exponential cone K = cl{(r, s, t): s > 0, s exp(r/s) <= t}, in
# units of the largest entry of x and from 4 precision + 400 bits: the violation of p in K, that of q = x - p in the polar cone
# {(r, s, t): r > 0, t < 0, s <= r (1 + log(-t/r))} (with its closure at r = 0), and |<p, q>| / |x|^2. The entries are first
# multiplied by a power of two, which is exact while the entries stay in range; this keeps the products of the largest entries
# from overflowing, and the entries of the data below are of bounded ratio, so that their products stay in range too.
function exp_cone_residuals(x::Vector{BigFloat}, y::Vector{BigFloat})
  prec = precision(BigFloat)
  setprecision(BigFloat, 4prec + 400) do
    e0 = Int(exponent(maximum(abs, x)))
    X = [Base.ldexp(v, Clong(-e0)) for v in x]
    P = [Base.ldexp(v, Clong(-e0)) for v in y]
    Q = X .- P
    sc = maximum(abs, X)
    function in_cone(v)
      r, s, t = v
      s < 0 && return -s / sc + 1
      iszero(s) && return (max(zero(r), r) + max(zero(t), -t)) / sc
      t <= 0 && return one(r)
      max(zero(r), r - s * log(t / s)) / sc
    end
    function in_polar(v)
      r, s, t = v
      r < 0 && return -r / sc + 1
      iszero(r) && return (max(zero(s), s) + max(zero(t), t)) / sc
      t >= 0 && return one(r)
      max(zero(s), s - r * (1 + log(-t / r))) / sc
    end
    Float64(in_cone(P)), Float64(in_polar(Q)), Float64(abs(sum(P .* Q)) / sc^2)
  end
end

const EXP_CONE_DIRECTIONS = ((1, 1, 1), (1, 2, 3), (1, 0.5, 0.1), (3, 1, 0.5), (1, -1, 2), (2, 3, -1))
# the projection of 2^E (1, 1, 1) is 2^E times that of (1, 1, 1), the values of the Float64 projection
const EXP_CONE_ONES = [0.42630617230379403, 0.7516727774312039, 1.3253666051274098]

# The cases at 53 bits are those with |E| <= 2^20: the accuracy at 53 bits for a larger |E| is not asserted (the logarithm
# of an entry near 2^E is rounded to 53 bits, which is an absolute error of up to about E 2^-53).
@testset "exponential cone projection, Moreau residuals, precision $p" for (p, tol, Es) in (
    (53, 2.0^-40, (0, 10, 1000, 2^20, -2^20, -1000)),
    (256, 2.0^-220, (0, 10, 1000, 2^20, -2^20, 2^40, -2^40, 2^61, -2^61, -1000)))
  setprecision(BigFloat, p) do
    ran = 0
    for E in Es, d in EXP_CONE_DIRECTIONS
      BOT + 10 <= E <= TOP - 10 || continue
      ran += 1
      x = BigFloat[d[1] * pw(E), d[2] * pw(E), d[3] * pw(E)]
      y = zeros(BigFloat, 3)
      prox!(y, IndExpPrimal(), x, 1)
      @test all(isfinite, y)
      vp, vq, orth = exp_cone_residuals(x, y)
      @test vp <= tol
      @test vq <= tol
      @test orth <= tol
    end
    @test ran >= 6 * 6
  end
end

@testset "exponential cone projection of (1, 1, 1) and its dual, precision $p" for (p, Es) in (
    (53, (0, 20, -20)), (256, (0, 2^20, 2^40, 2^61, -2^61)))
  setprecision(BigFloat, p) do
    # the dual projection of -(1, 1, 1), by Moreau: P_K*(-ones) = -P_K°(ones) = -(ones - P_K(ones)) = P_K(ones) - ones
    dual = EXP_CONE_ONES .- 1
    for E in Es
      BOT + 10 <= E <= TOP - 10 || continue
      for (f, want) in ((IndExpPrimal(), EXP_CONE_ONES), (IndExpDual(), dual))
        x = BigFloat[pw(E, f isa IndExpDual ? -1 : 1) for _ in 1:3]
        y = zeros(BigFloat, 3)
        prox!(y, f, x, 1)
        @test all(isfinite, y)
        @test Float64.([Base.ldexp(v, Clong(-E)) for v in y]) ≈ want rtol = 1e-12
      end
    end
  end
end

# x = [r, tiny, 1] with r / tiny = X1 = (TOP - (frexp exponent of tiny) + 2) log 2, which is (2^63 - 1) log 2 on the 64-bit
# Unix range. In _exp_product(tiny, X1) the quotient b / log 2 is 2^63 at 53 bits and about 2^63 - 1 at 256 bits. The
# projection is finite and its first two coordinates are below the square root of the smallest positive number
@testset "exponential cone projection with the quotient at 2^63, precision $p" for p in (53, 256)
  setprecision(BigFloat, p) do
    tiny = nextfloat(zero(BigFloat))
    X1 = (exponent(floatmax(BigFloat)) - Int(Base.frexp(tiny)[2]) + 2) * log(BigFloat(2))
    r = Base.ldexp(X1, Clong(Base.MPFR.get_emin() - 1))              # r / tiny = X1
    x = BigFloat[r, tiny, one(BigFloat)]
    y = zeros(BigFloat, 3)
    prox!(y, IndExpPrimal(), x, 1)
    @test all(isfinite, y)
    @test y[3] == 1
    @test 0 <= y[1] <= pw(BOT ÷ 2)
    @test 0 <= y[2] <= pw(BOT ÷ 2)
    @test PB._exp_product(tiny, X1) isa BigFloat
  end
end

# a exp(b) = m exp(b - j log 2) 2^(k + j) for a = m 2^k and j = round(b / log 2): the value from 4 precision + 400 bits
function exp_product_oracle(a::BigFloat, b::BigFloat)
  prec = precision(BigFloat)
  m, k = Base.frexp(a)
  x, j = setprecision(BigFloat, 4prec + 400) do
    L = log(BigFloat(2))
    j = Int128(round(BigInt, b / L))
    BigFloat(m * exp(b - j * L); precision = prec), j
  end
  Base.ldexp(x, Clong(clamp(Int128(k) + j, typemin(Clong), typemax(Clong))))
end

# a = floatmax(BigFloat) has the frexp exponent k = TOP + 1, so the zero guard is b < g0 log 2 with g0 = BOT - k - 2, which is
# -2^63 - 1 on the 64-bit Unix range. The cases b = (g0 + d) log 2 have the value m exp(r) 2^(BOT + d - 2): about the
# smallest positive number at d = 2 and above it for larger d, while d <= 0 is at the guard or below it, with the value 0.
# The reduction b - j log 2 with j near 2^63 has an absolute error of up to about |j| 2^-p, which is a relative error of
# the value: about 2^-193 at 256 bits and, at 53 bits, a factor of up to about 2^(1.5 2^10) in the significand; so at
# 53 bits the exponent of the value is asserted to within 2^(64 - p).
@testset "exponential product beyond its zero guard at the largest k, precision $p" for p in (53, 256)
  setprecision(BigFloat, p) do
    a = floatmax(BigFloat)
    g0 = BOT - (TOP + 1) - 2
    EMAX == 2^62 - 1 && @test g0 == -Int128(2)^63 - 1
    for d in (p == 256 ? (-1, 0, 2, 3, 10, 1000, 10^6, 2^40, 2^61, 2^62) : (1000, 10^6, 2^40, 2^61, 2^62))
      b = BigFloat(g0 + d) * log(BigFloat(2))
      got = PB._exp_product(a, b)
      want = exp_product_oracle(a, b)
      @test !isnan(got)
      @test isinf(got) == isinf(want)
      @test iszero(got) == iszero(want)
      if isfinite(want) && !iszero(want) && isfinite(got) && !iszero(got)
        if p == 256
          @test setprecision(() -> abs(got - want) <= 2.0^-180 * want, BigFloat, 4p + 400)
        else
          @test abs(log2(got) - log2(want)) <= 2.0^(64 - p)
        end
      end
    end
  end
end

@testset "exponential cone with Float64 data keeps its sums" begin
  y = zeros(3)
  prox!(y, IndExpPrimal(), [1.0, 1.0, 1.0], 1)
  @test y ≈ EXP_CONE_ONES rtol = 1e-13
  for (a, b) in ((2.0^1000, -1000.0), (-2.0^1000, -1000.0), (2.0^-1000, 1000.0), (1.5, -900.0), (1.0, 800.0), (-1.0, 800.0))
    ref = setprecision(() -> Float64(big(a) * exp(big(b))), BigFloat, 400)
    got = @inferred PB._exp_product(a, b)
    @test isinf(got) == isinf(ref)
    @test iszero(got) == iszero(ref)
    @test signbit(got) == signbit(ref)
    isfinite(ref) && !iszero(ref) && @test got ≈ ref rtol = 1e-11
  end
end

# ---- hyperplane, halfspace and hyperslab: the spectrum test and the metric projection ----

# Spectral(IndHyperplane(A, b))(X) with A = 2^TOP and X = F ones(2, 2): the eigenvalues 0 and 2F (about 1.98 floatmax) overflow in the type of
# X, and the test runs on the spectrum scaled by 2^-k with k = exponent(F) (1023 for Float64, 127 for Float32). The normal
# is held as α 2^e with e = -TOP. The test takes j as the larger of 0, the exponent of the scaled spectrum and
# T0 = exponent(b) + e - k, which for b = 2^mb is mb - TOP - k: below typemin(Int64) for mb < typemin(Int64) + TOP + k, where
# the Int sum wraps. The point is infeasible, since ⟨a 1, spectrum⟩ = 2^TOP 2F exceeds |b|: the value is +Inf for the
# hyperplane, the halfspace and the hyperslab [-b, b]. The halfspace with the normal -A and the traceless X = [F F; F -F]
# are feasible, with the value 0. The indicators accept a relative error of sqrt(eps), which rounds to 0.75 at 2 bits
# (the real sqrt(1/2) is 0.707...) and holds the infeasible point within it, so the precisions are 5, 53 and 256.
@testset "Spectral linear sets with BigFloat parameters at the Int64 boundary, precision $p" for p in (5, 53, 256)
  setprecision(BigFloat, p) do
    A = pw(TOP)
    for (F, k) in ((0.99 * floatmax(Float64), 1023), (0.99f0 * floatmax(Float32), 127))
      X, Xtraceless = fill(F, 2, 2), [F F; F -F]
      @test exponent(F) == k
      ran = 0
      for delta in (-1000, -3, -2, -1, 0, 1, 1000)
        mb = Int128(typemin(Int64)) + TOP + k + delta           # T0 = typemin(Int64) + delta
        BOT <= mb <= TOP || continue
        ran += 1
        b = pw(mb)
        for r in (Spectral(IndHyperplane(A, b))(X), Spectral(IndHalfspace(A, b))(X), Spectral(IndHyperslab(A, -b, b))(X))
          @test isinf(r) && r > 0
          @test r isa typeof(F)
        end
        for r in (Spectral(IndHalfspace(-A, b))(X), Spectral(IndHyperplane(A, b))(Xtraceless),
                  Spectral(IndHalfspace(A, b))(Xtraceless), Spectral(IndHyperslab(A, -b, b))(Xtraceless))
          @test iszero(r)
        end
      end
      @test ran >= (EMAX == 2^62 - 1 ? 6 : 0)
      # moderate and extreme exponents of b in the range
      for mb in (BOT, 0, TOP)
        @test isinf(Spectral(IndHyperplane(A, pw(mb)))(X))
        @test isinf(Spectral(IndHalfspace(A, pw(mb)))(X))
      end
    end
  end
end

@testset "Spectral linear sets with Float64 parameters keep their sums" begin
  X = fill(0.99 * floatmax(Float64), 2, 2)
  for b in (2.0^-1000, 2.0^1000)
    for r in (Spectral(IndHyperplane(2.0^1000, b))(X), Spectral(IndHalfspace(2.0^1000, b))(X), Spectral(IndHyperslab(2.0^1000, -b, b))(X))
      @test r === Inf
    end
    @test iszero(Spectral(IndHalfspace(-2.0^1000, b))(X))
  end
  @test @inferred(PB._comparison_exponent(3, -5)) === 3
  @test PB._comparison_exponent(-1, -2) === 0
  @test PB._comparison_exponent() === 0
  @test PB._comparison_exponent(Int32(3), Int32(-5)) === 3
  @test PB._comparison_exponent(Int32(-3), 7, Int8(2)) === 7
  @test @inferred(PB._power_factors(Float64, 5)) === (32.0, 1.0)
  @test PB._power_factors(Float64, Int32(5)) === (32.0, 1.0)
  @test PB._power_factors(Float64, Int128(5)) === (32.0, 1.0)
  @test PB._power_factors(Float64, -2000) === (Base.ldexp(1.0, -1074), Base.ldexp(1.0, -926))
  @test PB._power_factors(Float32, 200) === (Base.ldexp(1.0f0, 127), Base.ldexp(1.0f0, 73))
  @test_throws InexactError PB._power_factors(Float64, Int128(2)^70)
end

@testset "comparison exponent and power factors beyond Int64, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    @test PB._comparison_exponent(Int128(2)^70, 5) === Int128(2)^70
    @test PB._comparison_exponent(3, Int128(2)^70, -7) === Int128(2)^70
    @test PB._comparison_exponent(-Int128(2)^70, 5) === Int128(5)
    @test PB._comparison_exponent(-Int128(2)^70) === Int128(0)
    wide = Int128(typemax(Int64)) + 5
    # 2^k as the product of two factors: for k = TOP + 5 the first is the last finite power and the second is the finite
    # remainder 2^5; for the larger k below, the remainder is itself beyond the range and the second factor is +Inf
    f = PB._power_factors(BigFloat, Int128(TOP) + 5)
    @test f[1] == pw(TOP) && f[2] == 32
    f = PB._power_factors(BigFloat, wide)
    @test f[1] == pw(TOP) && isinf(f[2]) && f[2] > 0
    f = PB._power_factors(BigFloat, -wide)
    @test f[1] == pw(BOT) && iszero(f[2])
    f = PB._power_factors(BigFloat, Int128(2)^70)
    @test f[1] == pw(TOP) && isinf(f[2])
    f = PB._power_factors(BigFloat, -Int128(2)^70)
    @test f[1] == pw(BOT) && iszero(f[2])
    @test PB._power_factors(BigFloat, 7) == (128, 1)
    @test PB._power_factors(BigFloat, Int128(7)) == (128, 1)
  end
end

# prox!(y, IndHyperplane(a, 0), x, γ) with the normal a = [1, 2^n2], x = [2^TOP, 0] and γ = [2^-10, 2^-10] (the metric of a
# diagonal step): the residual is g = 2^TOP and d = Σ γᵢ aᵢ² = 2^-10 (1 + 2^(2 n2)), where for these n2 the term 2^(2 n2) does
# not change d. Then t = g / d = 2^(TOP + 10) overflows and the projection is formed from powers of two
# (_extreme_projection!), with w the largest exponent(γᵢ) + 2 exponent(αᵢ): for the second entry that is -10 + 2 n2, below
# typemin(Int64) for n2 < BOT + 5 where the range is the 64-bit Unix one (2 BOT = typemin(Int64)); in general, when
# -10 + 2 n2 < typemin(Int64). The projection is y = x - t (γ ⊙ a): y₁ = 2^TOP - 2^TOP = 0 and
# y₂ = -2^(TOP + 10 - 10 + n2) = -2^(TOP + n2).
@testset "hyperplane metric prox, tiny normal entry, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    ran = 0
    for n2 in (BOT, BOT + 1, BOT + 3, BOT + 4, BOT + 5, BOT + 6, BOT + 100, Int128(-1000))
      BOT <= n2 <= TOP || continue
      ran += 1
      y = [pw(TOP), zero(BigFloat)]
      prox!(y, IndHyperplane([one(BigFloat), pw(n2)], zero(BigFloat)), [pw(TOP), zero(BigFloat)], [pw(-10), pw(-10)])
      @test iszero(y[1])
      check_result(y[2], -1, TOP + n2)
    end
    @test ran == 8
    if EMAX == 2^62 - 1
      @test -10 + 2 * (BOT + 5) == typemin(Int64) && -10 + 2 * (BOT + 4) == Int128(typemin(Int64)) - 2
    end
  end
end

@testset "hyperplane metric prox with Float64 data keeps its sums" begin
  y = [2.0^1023, 0.0]
  prox!(y, IndHyperplane([1.0, 2.0^-1070], 0.0), [2.0^1023, 0.0], [2.0^-10, 2.0^-10])
  @test y == [0.0, -2.0^-47]
end

# the frame of the denominator d = D 2^w of an array normal: D = Σ γᵢ αᵢ² 2^-w, with w the larger of typemin(Int64) and the
# largest exponent(γᵢ) + 2 exponent(αᵢ). For α = 2^BOT and γ = 2^-10 that exponent is 2 BOT - 10, which is below
# typemin(Int64) on the 64-bit Unix range. For two equal entries d = 2^(g + 2n + 1), and the pair keeps
# exponent(D) + w = g + 2n + 1, as the test asserts
@testset "denominator of an array normal, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    for (n, g) in ((BOT, -10), (BOT + 7, 3), (0, 0), (TOP, BOT), (BOT, TOP))
      a, γ = [pw(n), pw(n)], [pw(g), pw(g)]
      D, w = PB._extreme_denominator(nothing, a, zero(BigFloat), γ, BigFloat)
      @test w isa Int128
      @test isfinite(D) && D > 0
      @test Int128(exponent(D)) + w == 2n + g + 1
    end
  end
  D, w = PB._extreme_denominator(nothing, [1.0, 2.0], 0.0, [0.5, 0.25], Float64)
  @test w === 0 && D === 1.5
end

# the correction of the projection xᵢ - t γᵢ αᵢ 2^J with an Int128 power J beyond Int64: the value is x - 2^n with the exponent
# n = exponent(t) + exponent(γ) + exponent(α) + J of the correction, or its saturation; for t = 2^TOP and γ = α = 2^BOT,
# n = TOP + 2 BOT + J
@testset "extreme output with a power beyond Int64, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    hi, lo = Int128(typemax(Int64)), Int128(typemin(Int64))
    for J in (hi - 1, hi, hi + 1, hi + 2, 3 * Int128(2)^62, lo + 1, lo, lo - 1, -3 * Int128(2)^62, Int128(2)^70, -Int128(2)^70)
      for (t, g, a) in ((pw(TOP), pw(BOT), pw(BOT)), (pw(TOP), pw(TOP), pw(TOP)), (pw(BOT), pw(BOT), pw(BOT)))
        o = PB._ExtremeOutput(t, J)
        @test o isa PB._ExtremeOutput{BigFloat, Int128}
        n = Int128(exponent(t)) + exponent(g) + exponent(a) + J
        want = pw(0) - Base.ldexp(one(BigFloat), Clong(clamp(n, typemin(Clong), typemax(Clong))))
        @test o(pw(0), a, g) == want
      end
    end
    @test PB._ExtremeOutput(1.5, 3) isa PB._ExtremeOutput{Float64, Int}
  end
end

# a = 2^-TOP and b = 2^(TOP - 1) with x = [1, 1] and γ = 2^-(TOP - 8) for both entries: g = 2 2^-TOP - b is -2^(TOP - 1) to the
# precision, d = a² (γ₁ + γ₂) = 2^(9 - 3 TOP), so t = g / d = -2^(4 TOP - 10) and the correction t γᵢ a = -2^(2 TOP - 2) is
# beyond the range (TOP = 2^62 - 2 gives the exponent 2^63 - 6): y = [+Inf, +Inf]. The power J = k - w of the output has
# k = 2 TOP - 1 and w near 9 - TOP, about 1.5 2^63 on the 64-bit Unix range
@testset "hyperplane metric prox with a power beyond Int64, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    @test 2TOP - 2 >= EMAX
    for sx in (1, -1)
      y = [pw(0, sx), pw(0, sx)]
      prox!(y, IndHyperplane(pw(-TOP), pw(TOP - 1)), [pw(0, sx), pw(0, sx)], [pw(-(TOP - 8)), pw(-(TOP - 8))])
      @test all(isinf, y) && all(>(0), y)
    end
  end
end

end
