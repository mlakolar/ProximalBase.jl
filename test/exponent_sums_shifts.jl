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

end
