module ExponentSumsTest

# MPFR exponents reach ±(2^62 - 1) on 64-bit Unix, so the sum of the exponents of three BigFloat factors can leave Int64.
# The kernels that split their factors into significands and exponents must form that sum without wrapping. All cases are
# built from exponents relative to the platform's own range, so the same body runs where the range is smaller.

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase

const EMAX = Int128(Base.MPFR.get_emax())
const EMIN = Int128(Base.MPFR.get_emin())
const PRECISIONS = (2, 5, 53, 256)

# a signed power of two, exact at every precision
function pw(p::Integer, s::Integer = 1)
  @assert EMIN - 1 <= p <= EMAX - 1
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

const A = EMAX - 2
const M = EMIN + 3

# the exponent triples (p1, p2, p3) of the factors 2^p1, 2^p2, 2^p3, with the extra exponent k
function triples()
  out = Tuple{Int128, Int128, Int128, Int}[]
  push!(out, (A, A, A, 0), (M, M, M, 0))
  push!(out, (A, M, A, 0), (A, A, M, 0), (M, A, A, 0), (A, M, M, 0), (M, A, M, 0), (M, M, A, 0))
  push!(out, (A, A, -A, 0), (-A, A, A, 0), (A, -A, A, 0), (M, M, -M, 0), (-A, -A, -A, 0), (A, -A, 0, 0))
  push!(out, (A, 7, -5, 0), (3, -4, 5, 0), (0, 0, 0, 0), (1, 1, 1, 0), (-1, -1, -1, 0))
  # the result 2^n at the boundaries of the range, from factors whose exponents are large
  for n in (EMAX - 1, EMAX, EMAX + 1, EMIN - 1, EMIN - 3, EMIN - 4)
    p1 = n > 0 ? A : -A
    p3 = n - 2p1
    @assert EMIN - 1 <= p3 <= EMAX - 1
    push!(out, (p1, p1, p3, 0), (p1, p3, p1, 0), (p3, p1, p1, 0))
  end
  # a nonzero power k
  push!(out, (A, A, A, Int(-2A)), (A, A, A, 5), (A, A, A, Int(-A)), (M, M, M, Int(-2M)), (M, M, M, Int(-A)))
  push!(out, (A, A, M, Int(-A)), (A, M, M, Int(A)), (A, A, A, Int(-2A) + 1), (A, A, A, Int(-2A) - 1))
  out
end

const SIGNS = Iterators.product((1, -1), (1, -1), (1, -1))

@testset "the exponent sum" begin
  @test PB._exponent_sum(BigFloat, 1, 2, 3) === Int128(6)
  @test PB._exponent_sum(BigFloat, Int32(1), 2, -3) === Int128(0)
  big_e = typemax(Int64)
  @test PB._exponent_sum(BigFloat, big_e, big_e, big_e) == 3 * Int128(big_e)
  @test PB._exponent_sum(BigFloat, typemin(Int64), typemin(Int64), typemax(Int64)) == 2 * Int128(typemin(Int64)) + typemax(Int64)
  for T in (Float64, Float32)
    @test PB._exponent_sum(T, 1, 2, 3) === 6
    @test PB._exponent_sum(T, 1, 2, 3, 4) === 10
    @test @inferred(PB._exponent_sum(T, 1, 2, -3)) === 0
  end
  @test @inferred(PB._exponent_sum(BigFloat, 1, 2, 3)) === Int128(6)
end

@testset "the case leaves Int64 on 64-bit Unix" begin
  if 3 * (EMAX - 1) > typemax(Int64)
    setprecision(BigFloat, 53) do
      a = pw(A)
      e = PB._frexp(a)[2]
      @test e isa Int
      @test 3 * Int128(e) > typemax(Int64)
      @test Int128(e + e + e) != 3 * Int128(e)       # the sum in Int wraps
      @test isinf(PB._product3(a, a, a))
    end
  end
end

@testset "three factors, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    for (p1, p2, p3, k) in triples(), (s1, s2, s3) in SIGNS
      a, b, c = pw(p1, s1), pw(p2, s2), pw(p3, s3)
      n = p1 + p2 + p3
      s = s1 * s2 * s3
      check_result(PB._product3(a, b, c), s, n)
      check_result(PB._scaled_product(a, b, c, k), s, n + k)
      if k == 0
        # a magnitude, the coefficient of a nonnegative term
        aa, bb, cc = abs(a), abs(b), abs(c)
        check_result(PB._exceptional_product(aa, bb, cc, aa * bb * cc), 1, n)
      end
    end
  end
end

@testset "two factors and a power, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    for (p1, p2, _, k) in triples(), s in (1, -1)
      w, v = pw(p1), pw(p2, s)
      k == 0 && (k = 1)
      check_result(PB._scaled_product(w, v, k), s, p1 + p2 + k)
    end
    # a power that cancels the factors, and one that moves a finite product out of range
    for kk in (Int(-2A), Int(-A), Int(A), Int(-A) - 1)
      check_result(PB._scaled_product(pw(A), pw(A, -1), kk), -1, 2A + kk)
    end
  end
end

@testset "special values, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    vals = (big(0.0), big(-0.0), big(NaN), big(Inf), big(-Inf), pw(A), pw(M, -1), pw(1))
    for a in vals, b in vals, c in vals
      if any(iszero, (a, b, c))
        @test iszero(PB._product3(a, b, c))
        @test iszero(PB._exceptional_product(a, b, c, a * b * c))
        @test iszero(PB._scaled_product(a, b, c, 3))
      end
      if !any(iszero, (a, b, c)) && !all(isfinite, (a, b, c))
        @test isequal(PB._product3(a, b, c), a * b * c)
        @test isequal(PB._exceptional_product(a, b, c, a * b * c), a * b * c)
        @test isequal(PB._scaled_product(a, b, c, 3), a * b * c)
      end
    end
    for w in vals, v in vals
      (iszero(w) || iszero(v) || !(isfinite(w) && isfinite(v))) || continue
      @test isequal(PB._scaled_product(w, v, 4), w * v)
    end
  end
end

@testset "public paths, precision $p" for p in PRECISIONS
  setprecision(BigFloat, p) do
    # the threshold of the l1 norm is the product of three scales
    a = pw(EMAX - 2)
    for x in (1, -1)
      y = prox(NormL1(a, [a]), [big(x)], a)
      @test all(iszero, y) && all(isfinite, y)
    end
    b = pw(EMIN + 4)
    for x in (1, -1)
      y = prox(NormL1(b, [b]), [big(x)], b)
      @test y == [big(x)]
    end

    # a scaled symmetric part
    t = pw(EMIN + 5)
    S = PB._SymmetricPart([big(0) t; big(0) big(0)])
    y = zeros(BigFloat, 2)
    mul!(y, S, [big(0), t], pw(EMIN - 1), 0)
    @test all(iszero, y)
    @test !any(isinf, y)

    # the projection onto a sphere, with a margin that spans the range
    f = Constrained(NormL1(pw(EMIN + 2)), IndSphereL2(pw(-20)))
    z = prox(f, [pw(EMAX - 2), pw(EMIN + 5)], 1)
    @test isfinite(z[1]) && isfinite(z[2])
    @test z[1] == pw(-20)
    @test iszero(z[2]) && !signbit(z[2])
  end
end

@testset "boundary pins of the safe sites" begin
  setprecision(BigFloat, 53) do
    @test PB._bernoulli_ratio(pw(EMAX - 1), one(BigFloat), pw(EMIN - 1)) == Inf
    @test PB._bernoulli_curvature(one(BigFloat), pw(EMIN - 1)) == Inf
    if EMAX == 2^62 - 1
      ew = Int128(frexp(pw(EMAX - 1))[2])
      eb = Int128(frexp(one(BigFloat))[2])
      ep = Int128(frexp(pw(EMIN - 1))[2])
      @test ew + eb - ep == typemax(Int64)
    end
  end
end

@testset "Float64 and Float32 keep their Int sums" begin
  @test PB._product3(2.0^1000, 2.0^1000, 2.0^-1000) == 2.0^1000
  @test PB._product3(2.0^1000, -2.0^1000, 2.0^-1000) == -2.0^1000
  @test PB._product3(2.0^1000, 2.0^1000, 2.0^1000) == Inf
  @test PB._product3(2.0^-1000, 2.0^-1000, 2.0^-1000) == 0.0
  @test PB._exceptional_product(2.0^1000, 2.0^1000, 2.0^-1000, 2.0^1000 * 2.0^1000 * 2.0^-1000) == 2.0^1000
  @test PB._exceptional_product(2.0^1000, 2.0^1000, 2.0^1000, Inf) == Inf
  @test PB._scaled_product(2.0^1000, 2.0^1000, 2.0^-1000, 0) == 2.0^1000
  @test PB._scaled_product(2.0^1000, 2.0^1000, 2.0^-1000, -5) == 2.0^995
  @test PB._scaled_product(2.0^1000, 2.0^1000, -1000) == 2.0^1000
  @test PB._scaled_product(2.0^1000, -2.0^1000, -1000) == -2.0^1000

  f1(x) = Base.ldexp(1f0, x)
  @test PB._product3(f1(100), f1(100), f1(-140)) == f1(60)
  @test PB._product3(f1(100), f1(100), f1(100)) == Inf32
  @test PB._exceptional_product(f1(100), f1(100), f1(-140), f1(100) * f1(100) * f1(-140)) == f1(60)
  @test PB._scaled_product(f1(100), f1(100), f1(-140), 3) == f1(63)
  @test PB._scaled_product(f1(100), f1(100), -140) == f1(60)

  for T in (Float64, Float32)
    a, b, c = T(3), T(0.25), T(-7)
    @test @inferred(PB._product3(a, b, c)) isa T
    @test @inferred(PB._exceptional_product(a, b, abs(c), a * b * abs(c))) isa T
    @test @inferred(PB._scaled_product(a, b, c, 2)) isa T
    @test @inferred(PB._scaled_product(a, b, 2)) isa T
    # a function per kernel, so that the measurement does not box the arguments
    alloc_p3(a, b, c) = @allocated PB._product3(a, b, c)
    alloc_ep(a, b, c) = @allocated PB._exceptional_product(a, b, c, a * b * c)
    alloc_sp4(a, b, c, k) = @allocated PB._scaled_product(a, b, c, k)
    alloc_sp3(a, b, k) = @allocated PB._scaled_product(a, b, k)
    alloc_p3(a, b, c); alloc_ep(a, b, abs(c)); alloc_sp4(a, b, c, 2); alloc_sp3(a, b, 2)
    @test alloc_p3(a, b, c) == 0
    @test alloc_ep(a, b, abs(c)) == 0
    @test alloc_sp4(a, b, c, 2) == 0
    @test alloc_sp3(a, b, 2) == 0
  end
end

end
