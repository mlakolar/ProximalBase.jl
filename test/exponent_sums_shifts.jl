module ExponentSumsShiftsTest

# MPFR exponents reach ±(2^62 - 1) on 64-bit Unix, so the sum of three or more exponents can leave Int64. The functions
# below form the exponent of a final shift, or a frame that one function builds from exact inputs, and that total must
# reach ldexp without wrapping. The public reproducers are built from exponents relative to the platform's own MPFR
# range, so the same body runs where the range is 2^30 - 1 (Windows). There their exponent totals stay within Int64: those
# cases are evidence of no regression, not of the repair. The cases built from explicit wide integers (the powers of
# _scaled_product beyond Int64, the scale exponents of the _image_product cases taken from typemin(Int64), the exponents
# of the _pow2_sum cases taken from typemax(Int64)) cross Int64 on that range as well. The expected values come from
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

end
