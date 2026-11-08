module BigFloatIntegerOperandsTest

# Exact integer operands of BigFloat arithmetic (#756). Where a C long is 32 bits (Windows), the
# products BigFloat * Int64 and Int64 * BigFloat and the quotient BigFloat / Int64 convert the integer
# with BigFloat(k) at the working precision before the operation, so that at a precision of two or
# three bits an integer such as 5 or 7 is rounded (5 becomes 4 at two bits) before it is used. On other
# platforms the integer is exact. The integers of NormL1Fused (the divisor n of the mean, the factor n
# of the value, the coefficients 2k - n - 1 of the fusion thresholds and of the value) and of the
# pooling of adjacent violators (the block sizes of the sums compared and of the block means) must be
# exact operands, so that a result does not depend on the platform.
#
# The file states that semantics once, as a model written here and independent of the package: every
# operation between two BigFloat values is the plain operation at the current precision p (correctly
# rounded, the same on every platform), and every operation with an integer operand is formed exactly
# and rounded once to p bits (mul_int, div_int). The package is compared with the model with ==, at
# precisions of two to five bits, where a rounded integer changes the result, and at 256 bits. The
# model follows the structure of the algorithms (the compensated sum and the mean of the route of an
# infinite fusion level, the thresholds and the pooling of the finite levels, the pooling of
# IndIsotonic and SLOPE) so that a result that the model gives is what the algorithm computes, not the
# correctly rounded proximal point.

using Test, ProximalBase

const PB = ProximalBase

####################################
#  the exact integer operand
####################################

# The correctly rounded value, at the current precision, of an exact rational. The quotient is formed
# in 600 bits (the numerator and the denominator are small: the entries have at most 256 bits, so the
# numerator is exact and the quotient is far from a midpoint of the target precision), then rounded
# once to the precision p of the caller.
function rnd(q::Rational{BigInt})
  iszero(numerator(q)) && return BigFloat(0)
  v = setprecision(BigFloat, 600) do
    BigFloat(numerator(q)) / denominator(q)
  end
  BigFloat(v; precision = precision(BigFloat))
end

# the significand of x in [1/2, 1) as an exact rational and the exponent; Base's frexp and the conversion to
# a rational round at the current precision, so they run at the precision of x
function split_exact(x::BigFloat)
  setprecision(BigFloat, precision(x)) do
    fr, ex = frexp(x)
    Rational{BigInt}(fr), Int(ex)
  end
end

# x * k and x / k with the integer k exact, rounded once. The exponent of x is split off (the
# significand lies in [1/2, 1)), so that an entry near floatmax is not expanded.
function mul_int(x::BigFloat, k::Integer)
  isnan(x) && return x
  if iszero(x) || isinf(x)
    isinf(x) && iszero(k) && return BigFloat(NaN)
    return k < 0 ? -x : copy(x)
  end
  iszero(k) && return signbit(x) ? -BigFloat(0) : BigFloat(0)
  fr, ex = split_exact(x)
  Base.ldexp(rnd(fr * big(k)), Clong(ex))
end
function div_int(x::BigFloat, k::Integer)
  iszero(k) && throw(DivideError())
  isnan(x) && return x
  (iszero(x) || isinf(x)) && return k < 0 ? -x : copy(x)
  fr, ex = split_exact(x)
  Base.ldexp(rnd(fr / big(k)), Clong(ex))
end

# the soft threshold, as the package forms it
mshrink(v::T, c::T) where {T<:AbstractFloat} = max(v - c, zero(v)) + min(v + c, zero(v))

# equal values, with the sign of a zero and NaN equal to NaN
same(a::BigFloat, b::BigFloat) = (isnan(a) && isnan(b)) || (a == b && signbit(a) == signbit(b))

# a small deterministic generator of the inputs (the same on every platform)
mutable struct Lcg
  s::UInt64
end
function nextint!(g::Lcg, m::Integer)         # 1:m
  g.s = g.s * 0x5851f42d4c957f2d + 0x14057b7ef767814f
  Int((g.s >> 33) % UInt64(m)) + 1
end
# a value of at most `bits` significant bits (so exact at p ≥ bits bits), times a power of two
function pbit!(g::Lcg, bits::Integer; shifts = (-1, 0, 1), signed::Bool = true)
  m = nextint!(g, 2^bits - 1)
  j = shifts[nextint!(g, length(shifts))]
  v = Base.ldexp(BigFloat(m), Clong(j))
  signed && nextint!(g, 2) == 1 ? -v : v
end

####################################
#  the model of the route of an infinite fusion level
####################################

# the compensated (Neumaier) sum, as the pair (s, c)
function m_neumaier(xs)
  s = c = zero(BigFloat)
  for a in xs
    t = s + a
    c += abs(s) >= abs(a) ? (s - t) + a : (a - t) + s
    s = t
  end
  s, c
end

# the mean of finite entries: the compensated sum over n, and, when the sum is not finite, the sum of the
# entries scaled by the power of two of the largest magnitude, split in significand and exponent
# (the significand over n, restored by an exact shift); clamped into [min(x), max(x)]
function m_mean(x::Vector{BigFloat})
  n = length(x)
  lo, hi = minimum(x), maximum(x)
  s, c = m_neumaier(x)
  t = s + c
  if isfinite(t)
    mean = div_int(t, n)
  else
    e = Int(exponent(maximum(abs, x)))
    s2, c2 = m_neumaier([Base.ldexp(a, Clong(-e)) for a in x])
    fr, ex = frexp(s2 + c2)
    mean = Base.ldexp(div_int(fr, n), Clong(Int(ex) + e))
  end
  clamp(mean, lo, hi)
end

# the common output of the route, for the step γ and the level λ1
m_inf_prox(x::Vector{BigFloat}, λ1::BigFloat, γ) = mshrink(m_mean(x), convert(BigFloat, γ) * λ1)

# λ1 n |z| at a vector of n equal entries z: the product with n exact and the product with λ1, each
# rounded once, on the significand of z, then the exact power of two
function m_equal_value(z::BigFloat, n::Integer, λ1::BigFloat)
  (iszero(λ1) || iszero(z)) && return zero(BigFloat)
  fr, ex = frexp(abs(z))
  Base.ldexp(λ1 * mul_int(fr, n), ex)
end

####################################
#  the model of the pooling and of the finite fusion levels
####################################

# pooling adjacent violators with the sums and the sizes of the blocks, direction σ = ±1; returns the
# block means in the order of the entries and the sizes of the blocks
function m_pool(w::Vector{BigFloat}, σ::Int)
  s = BigFloat[]
  c = Int[]
  for e in w
    push!(s, e)
    push!(c, 1)
    while length(s) > 1
      d = mul_int(s[end - 1], c[end]) - mul_int(s[end], c[end - 1])
      (σ > 0 ? d : -d) > 0 || break
      s[end - 1] += s[end]
      c[end - 1] += c[end]
      pop!(s)
      pop!(c)
    end
  end
  y = BigFloat[]
  for b in eachindex(s), _ in 1:c[b]
    push!(y, div_int(s[b], c[b]))
  end
  y, c
end

# the proximal point of NormL1Fused(λ1, λ2) at x (length > 2), finite levels, a finite step:
# the output in the order of x, the sizes of the pooled blocks, and the output in the order of the sorted
# entries (at a precision of a few bits the pooled means need not be increasing in that order, and the
# value at the output is the one of the sorted output)
function m_finite_prox(x::Vector{BigFloat}, λ1::BigFloat, λ2::BigFloat, γ)
  n = length(x)
  p = sortperm(x)
  a = x[p]
  t = convert(BigFloat, γ) * λ2
  w = BigFloat[a[k] - mul_int(t, 2k - n - 1) for k in 1:n]
  z, sizes = m_pool(w, 1)
  c = convert(BigFloat, γ) * λ1
  y = similar(x)
  ys = BigFloat[mshrink(z[k], c) for k in 1:n]
  for k in 1:n
    y[p[k]] = ys[k]
  end
  y, sizes, ys
end

# the value of NormL1Fused(λ1, λ2) at y (length > 2): λ1 Σ|a_k| + λ2 Σ (2k - n - 1) a_k with the entries
# sorted increasingly, and whether the ordinary expression is the one that the package returns (the value is
# finite and the fusion sum is not negative: the other values are recovered by a scaled sum)
function m_finite_value(y::Vector{BigFloat}, λ1::BigFloat, λ2::BigFloat)
  n = length(y)
  a = sort(y)
  v1 = v2 = zero(BigFloat)
  for k in 1:n
    v1 += abs(a[k])
    v2 += mul_int(a[k], 2k - n - 1)
  end
  v = λ1 * v1 + λ2 * v2
  v, isfinite(v) && v2 >= 0
end

# the SLOPE proximal point: the magnitudes in decreasing order less γλ_k, pooled as nonincreasing,
# clipped at zero, with the signs of x
function m_slope_prox(x::Vector{BigFloat}, λ::Vector{BigFloat}, γ)
  n = length(x)
  p = sortperm(abs.(x); rev = true)
  g = convert(BigFloat, γ)
  w = BigFloat[abs(x[p[k]]) - g * λ[k] for k in 1:n]
  z, sizes = m_pool(w, -1)
  y = similar(x)
  for k in 1:n
    wk = max(z[k], zero(BigFloat))
    y[p[k]] = signbit(x[p[k]]) ? -wk : wk
  end
  y, sizes
end

# every call form of the proximal operator at x against the output y0 (with ==, entry by entry)
function check_prox(f, x, γ, y0)
  y = similar(x)
  @test prox!(y, f, x, γ) === y
  @test y == y0
  x2 = copy(x)
  prox!(x2, f, x2, γ)
  @test x2 == y0
  y3 = similar(x)
  v = value_and_prox!(y3, f, x, γ)
  @test y3 == y0
  y, v
end

####################################
#  the model itself
####################################

@testset "the exact integer operand" begin
  setprecision(BigFloat, 2) do
    # hand-computed values at two bits: the representable numbers near 1 are 1, 1.5, 2, 3, 4, 6, 8
    @test div_int(BigFloat(1.5), 5) == BigFloat(0.25)
    @test mul_int(BigFloat(1.5), 5) == BigFloat(8)
    @test div_int(BigFloat(3), 5) == BigFloat(0.5)
    @test mul_int(BigFloat(3), 5) == BigFloat(16)
    @test mul_int(BigFloat(3), -5) == BigFloat(-16)
    @test div_int(BigFloat(3), -5) == BigFloat(-0.5)
    @test mul_int(BigFloat(3), 0) == 0
    @test div_int(BigFloat(3), 7) == BigFloat(0.375)
    @test mul_int(BigFloat(0.75), 5) == BigFloat(4)
    @test rnd(13 // 5 |> Rational{BigInt}) == BigFloat(3)
    @test rnd(8 // 5 |> Rational{BigInt}) == BigFloat(1.5)
    @test rnd(5 // 2 |> Rational{BigInt}) == BigFloat(2)       # the tie goes to the even significand
    @test rnd(7 // 2 |> Rational{BigInt}) == BigFloat(4)
    @test rnd(-7 // 2 |> Rational{BigInt}) == BigFloat(-4)
    @test iszero(rnd(0 // 1 |> Rational{BigInt}))
    # the operand k is not rounded to the precision before the product (BigFloat(5) is 4 here)
    @test BigFloat(5) == 4
    @test mul_int(BigFloat(3), 5) != BigFloat(3) * BigFloat(5)
    # the conversion of an entry to a rational is exact
    for x in BigFloat[0.5, 1.5, -3, 6, 0.75]
      @test BigFloat(Rational{BigInt}(x)) == x
    end
  end
  setprecision(BigFloat, 256) do
    g = Lcg(0x1234567)
    for _ in 1:20
      x = BigFloat(nextint!(g, 10^6)) / 7
      k = nextint!(g, 1000) - 500
      @test mul_int(x, k) == x * BigFloat(k)         # a 256-bit integer is exact: one rounding
      k == 0 || @test div_int(x, k) == x / BigFloat(k)
    end
  end
  setprecision(BigFloat, 5) do
    # the special values keep their class
    @test isnan(mul_int(BigFloat(NaN), 3)) && isnan(div_int(BigFloat(NaN), 3))
    @test mul_int(BigFloat(Inf), 3) == Inf && mul_int(BigFloat(Inf), -3) == -Inf
    @test isnan(mul_int(BigFloat(Inf), 0))
    @test div_int(BigFloat(-Inf), 3) == -Inf && div_int(BigFloat(Inf), -3) == -Inf
    @test iszero(mul_int(BigFloat(0), 7)) && !signbit(mul_int(BigFloat(0), 7))
    @test signbit(mul_int(-BigFloat(0), 7)) && signbit(div_int(BigFloat(0), -7))
    # an entry at the end of the exponent range is not expanded, and overflows as the product does
    big_ = floatmax(BigFloat)
    @test isinf(mul_int(big_, 3))
    @test div_int(big_, 1) == big_
  end
end

####################################
#  an infinite fusion level
####################################

# The route of NormL1Fused(λ1, Inf): the proximal point is the projection onto the constant vectors
# followed by the l1 shrink. The p-bit algorithm forms the compensated sum of the entries (rounded to p
# bits), divides it by the exact n (rounded once) and shrinks the result, so the common output is what
# the model gives, not the correctly rounded proximal map. For x = [3, 3, 2, 3, 2] at two bits, the real
# mean is 13/5 and the real proximal point at γ = 1 is 8/5, which a two-bit number cannot hold (the
# correctly rounded 8/5 is 1.5); the algorithm rounds the sum 13 to 12, divides by the exact 5 (the
# mean 12/5 = 2.4 is rounded to 2.0) and shrinks by 1, so its mean is 2.0 and its output 1.0.
# A platform that rounds the divisor 5 to 4 at two bits would divide by 4 and obtain 3.0 and 2.0.

function check_inf(x::Vector{BigFloat}, λ1::BigFloat, γ)
  n = length(x)
  f = NormL1Fused(λ1, BigFloat(Inf))
  z = m_inf_prox(x, λ1, γ)
  y0 = fill(z, n)
  y, v = check_prox(f, x, γ, y0)
  @test all(==(y[1]), y)
  # the value at the stored output: λ1 n |z| with the factor n exact
  want = m_equal_value(z, n, λ1)
  @test v == want
  @test f(y) == want
end

@testset "infinite fusion level: the mean and the value" begin
  @testset "the example of the issue" begin
    for p in (2, 3, 5, 256), γ in (1 // 2, 1)
      setprecision(BigFloat, p) do
        check_inf(BigFloat[3, 3, 2, 3, 2], BigFloat(1), γ)
      end
    end
    setprecision(BigFloat, 2) do
      x = BigFloat[3, 3, 2, 3, 2]
      @test m_mean(x) == 2.0
      @test m_inf_prox(x, BigFloat(1), 1) == 1.0
      f = NormL1Fused(BigFloat(1), BigFloat(Inf))
      y = zeros(BigFloat, 5)
      prox!(y, f, x, 1)
      @test all(==(1.0), y)
      # the real mean 13/5 and the real proximal point 8/5 are not two-bit numbers
      @test div_int(BigFloat(8), 5) == 1.5
    end
  end

  # The value at the stored output is λ1 n |z|: the factor n is an integer operand of the product, and a
  # precision of p bits holds n only below 2^p (p = 2: 3, 4, 6, 8). The cases of every n are checked
  # here for the value alone.
  @testset "the factor n of the value" begin
    for p in (2, 3, 5, 256), n in 3:9, λ1 in (BigFloat(1), BigFloat(0.75)), γ in (1 // 2, 1)
      setprecision(BigFloat, p) do
        f = NormL1Fused(λ1, BigFloat(Inf))
        for x in (fill(BigFloat(3), n), BigFloat[i <= n - 2 ? 3 : 2 for i in 1:n])
          z = m_inf_prox(x, λ1, γ)
          want = m_equal_value(z, n, λ1)
          @test value_and_prox!(similar(x), f, x, γ) == want
          @test f(fill(z, n)) == want
        end
      end
    end
  end

  # The value λ1 n |z| of a vector of entries near floatmax, in the scaled form: the product with the exact n
  # is rounded once (on the significand of z), then the product with λ1, then the exact power of two, and
  # the result is Inf when that exceeds floatmax (Base.ldexp with a Clong exponent overflows to Inf, which
  # is what the model does). A count rounded to p bits would give a finite value where the exact one
  # overflows: at two bits n = 5 is 4, and 0.25 · 4 · floatmax is floatmax, while 0.25 · 5 · floatmax is not
  # representable.
  @testset "overflow near floatmax" begin
    for p in (2, 3, 5), n in 3:9, λ1 in (BigFloat(0.25), BigFloat(1)), γ in (1 // 2, 1)
      setprecision(BigFloat, p) do
        fm = floatmax(BigFloat)
        f = NormL1Fused(λ1, BigFloat(Inf))
        x = fill(fm, n)
        @test f(x) == m_equal_value(fm, n, λ1)
        y = similar(x)
        v = value_and_prox!(y, f, x, γ)
        z = m_inf_prox(x, λ1, γ)
        @test y == fill(z, n)
        want = m_equal_value(z, n, λ1)
        @test v == want
        @test f(y) == want
      end
    end
    setprecision(BigFloat, 2) do
      fm = floatmax(BigFloat)
      f = NormL1Fused(BigFloat(0.25), BigFloat(Inf))
      @test f(fill(fm, 5)) == Inf
      @test m_equal_value(fm, 5, BigFloat(0.25)) == Inf
      @test value_and_prox!(similar(fill(fm, 5)), f, fill(fm, 5), 1) == Inf
      v3 = f(fill(fm, 3))
      @test v3 == m_equal_value(fm, 3, BigFloat(0.25))
      @test isfinite(v3)
    end
  end

  @testset "deterministic inputs, n = 3:9" begin
    for p in (2, 3, 5, 256)
      setprecision(BigFloat, p) do
        g = Lcg(UInt64(1000 + p))
        bits = min(p, 5)
        for n in 3:9, λ1 in (BigFloat(1), BigFloat(0.75)), γ in (1 // 2, 1)
          check_inf(BigFloat[pbit!(g, bits) for _ in 1:n], λ1, γ)
          check_inf(BigFloat[pbit!(g, bits; signed = false) for _ in 1:n], λ1, γ)
          check_inf(fill(BigFloat(3), n), λ1, γ)
          check_inf(BigFloat[i <= n - 2 ? 3 : 2 for i in 1:n], λ1, γ)
        end
      end
    end
    setprecision(BigFloat, 2) do          # a longer vector
      g = Lcg(0x16)
      for γ in (1 // 2, 1)
        check_inf(BigFloat[pbit!(g, 2; signed = false) for _ in 1:16], BigFloat(1), γ)
        check_inf(BigFloat[i <= 11 ? 3 : 2 for i in 1:16], BigFloat(1), γ)
      end
    end
  end

  # The sum overflows (the plain mean is not finite): the entries are scaled by the power of two of the
  # largest magnitude, summed, split in significand and exponent, divided by the exact n, restored by
  # an exact shift and clamped into the range of the entries. Entries near floatmax are formed here so
  # that the case is the same relative to the exponent range on every platform.
  @testset "scaled recovery near floatmax" begin
    for p in (2, 3, 5)
      setprecision(BigFloat, p) do
        fm = floatmax(BigFloat)
        h = Base.ldexp(fm, Clong(-1))
        q = Base.ldexp(fm, Clong(-2))
        for n in 3:9, λ1 in (BigFloat(1), BigFloat(0.75)), γ in (1 // 2, 1)
          x = fill(fm, n)
          @test !isfinite(sum(x))
          check_inf(x, λ1, γ)
          mixed = BigFloat[fm, fm, -h, fm, -q, fm, h, -fm, fm][1:n]
          check_inf(mixed, λ1, γ)
          check_inf(vcat(fill(fm, n - 1), -h), λ1, γ)
        end
      end
    end
    setprecision(BigFloat, 2) do
      fm = floatmax(BigFloat)
      x = fill(fm, 5)
      s, c = m_neumaier(x)
      @test !isfinite(s + c)
      @test m_mean(x) <= fm
      check_inf(x, BigFloat(1), 1)
    end
  end
end

####################################
#  finite fusion levels
####################################

# A step of pooled blocks: k entries of value 1 and the rest of value 8
step_input(n::Int, k::Int) = BigFloat[i <= k ? 1 : 8 for i in 1:n]

function check_finite(x::Vector{BigFloat}, λ1::BigFloat, λ2::BigFloat, γ, seen::Set{Int})
  f = NormL1Fused(λ1, λ2)
  y0, sizes, _ = m_finite_prox(x, λ1, λ2, γ)
  union!(seen, sizes)
  y, v = check_prox(f, x, γ, y0)
  # the value at the stored output (the entries of y sorted); a value that the ordinary expression does
  # not give (a fusion sum that rounding made negative) is recovered by a scaled sum and is not modelled
  want, ok = m_finite_value(y0, λ1, λ2)
  ok && @test v == want
  ok && @test f(y) == want
  vx, okx = m_finite_value(x, λ1, λ2)
  okx && @test f(x) == vx
  sizes
end

@testset "finite fusion levels" begin
  levels = ((BigFloat(0.25), BigFloat(0.5)), (BigFloat(0.375), BigFloat(0.75)))
  seen = Dict(p => Set{Int}() for p in (2, 3, 5, 256))
  @testset "deterministic inputs, n = 3:9" begin
    for p in (2, 3, 5, 256)
      setprecision(BigFloat, p) do
        g = Lcg(UInt64(2000 + p))
        bits = min(p, 4)
        for n in 3:9, (λ1, λ2) in levels, γ in (1 // 2, 1)
          s = seen[p]
          check_finite(BigFloat[pbit!(g, bits) for _ in 1:n], λ1, λ2, γ, s)
          check_finite(BigFloat[pbit!(g, bits; signed = false) for _ in 1:n], λ1, λ2, γ, s)
          check_finite(fill(BigFloat(3), n), λ1, λ2, γ, s)
          check_finite(fill(BigFloat(2), n), λ1, λ2, γ, s)
          for k in (3, 5, 7)
            k < n && check_finite(step_input(n, k), λ1, λ2, γ, s)
          end
          # a descending input
          check_finite(sort(BigFloat[pbit!(g, bits) for _ in 1:n]; rev = true), λ1, λ2, γ, s)
          check_finite(reverse(step_input(n, min(5, n - 1))), λ1, λ2, γ, s)
        end
      end
    end
    for p in (2, 3, 5, 256)
      @test 5 in seen[p]
      @test 7 in seen[p]
      @test 9 in seen[p]
    end
  end

  @testset "one longer case, n = 16, p = 2" begin
    setprecision(BigFloat, 2) do
      g = Lcg(0x2016)
      s = Set{Int}()
      for (λ1, λ2) in levels, γ in (1 // 2, 1)
        check_finite(BigFloat[pbit!(g, 2) for _ in 1:16], λ1, λ2, γ, s)
        check_finite(step_input(16, 5), λ1, λ2, γ, s)
        check_finite(step_input(16, 9), λ1, λ2, γ, s)
        check_finite(fill(BigFloat(3), 16), λ1, λ2, γ, s)
      end
      @test 5 in s && 9 in s && 16 in s
    end
  end

  @testset "the nine entries of the issue pool into one block" begin
    for p in (2, 3, 5, 256), γ in (1 // 2, 1)
      setprecision(BigFloat, p) do
        λ1, λ2 = BigFloat(0.25), BigFloat(0.5)
        s = Set{Int}()
        for x in (fill(BigFloat(2), 9), fill(BigFloat(3), 9),
                  BigFloat[2, 2, 2, 2, 2, 2, 2, 2, 3] .* 1)
          check_finite(x, λ1, λ2, γ, s)
        end
        @test 9 in s
        y0, sizes, _ = m_finite_prox(fill(BigFloat(2), 9), λ1, λ2, γ)
        @test sizes == [9]
        @test all(==(y0[1]), y0)
        # a descending input
        d = BigFloat[6, 4, 4, 3, 2, 2, 1, 1, 0.5]
        check_finite(d, λ1, λ2, γ, s)
      end
    end
  end
end

####################################
#  the value at the stored output of the finite fusion levels
####################################

# At a precision of two or three bits the rounded block means of the pooling need not be nondecreasing
# in the order of the sorted entries (a pooled sequence [1, 0.75, ...]). The stored output is that
# sequence soft-thresholded, placed at the entries; f(y) sorts y, which can be another sequence, so a
# value accumulated in the order of the pooled sequence can differ from the value at y. value_and_prox!
# returns the value at the stored output, f(y), and the stored output is the one that prox! stores.

# the number of positions of the sequence at which it decreases
inversions(w) = count(k -> w[k + 1] < w[k], 1:length(w) - 1)

# one input of NormL1Fused: the stored output is the model's proximal point and equals the one of prox,
# the returned value is the value at it, for a separate and for an in-place output
function check_stored_value(x::Vector{BigFloat}, λ1::BigFloat, λ2::BigFloat, γ, counts)
  f = NormL1Fused(λ1, λ2)
  y0, _, ys = m_finite_prox(x, λ1, λ2, γ)
  yp = prox(f, x, γ)
  @test same_vec(yp, y0)
  y = similar(x)
  v = value_and_prox!(y, f, x, γ)
  @test same_vec(y, y0)
  @test same_vec(y, yp)
  @test v == f(y)
  want, ok = m_finite_value(y, λ1, λ2)
  ok && @test v == want
  x2 = copy(x)
  v2 = value_and_prox!(x2, f, x2, γ)
  @test same_vec(x2, y0)
  @test v2 == v
  inversions(ys) > 0 ? (counts[1] += 1) : (counts[2] += 1)
  nothing
end
same_vec(a, b) = length(a) == length(b) && all(same.(a, b))

@testset "NormL1Fused: the value at the stored output" begin
  @testset "the two examples at two bits" begin
    setprecision(BigFloat, 2) do
      f = NormL1Fused(BigFloat(0.25), BigFloat(0.5))
      x = BigFloat[2.0, 0.5, 1.0, 2.0, -1.0, 1.0]
      y = similar(x)
      v = value_and_prox!(y, f, x, 1)
      @test y == BigFloat[0.5, 0.5, 0.5, 0.5, 0.75, 0.5]       # as prox stores it
      @test y == prox(f, x, 1)
      @test f(y) == 2.0
      @test v == f(y)
      x = BigFloat[-1, -1, 1.5, -2, -0.5, 0.5]
      y = similar(x)
      v = value_and_prox!(y, f, x, 1)
      @test y == BigFloat[-0.125, -0.125, -0.25, -0.125, -0.125, -0.125]
      @test y == prox(f, x, 1)
      @test f(y) == 0.375
      @test v == f(y)
    end
  end

  @testset "the same inputs at precision $p" for p in (3, 5, 53)
    setprecision(BigFloat, p) do
      f = NormL1Fused(BigFloat(0.25), BigFloat(0.5))
      counts = [0, 0]
      for x in (BigFloat[2.0, 0.5, 1.0, 2.0, -1.0, 1.0], BigFloat[-1, -1, 1.5, -2, -0.5, 0.5])
        check_stored_value(x, BigFloat(0.25), BigFloat(0.5), 1, counts)
        y = similar(x)
        @test value_and_prox!(y, f, x, 1) == f(y)
        @test all(==(y[1]), y)                                  # one pooled block, no inversion
      end
      @test counts == [0, 2]                                    # no inversion of the stored sequence
      if p == 3
        y = Vector{BigFloat}(undef, 6)
        @test value_and_prox!(y, f, BigFloat[2.0, 0.5, 1.0, 2.0, -1.0, 1.0], 1) == 1.25
        @test value_and_prox!(y, f, BigFloat[-1, -1, 1.5, -2, -0.5, 0.5], 1) == 0.3125
      elseif p == 5
        y = Vector{BigFloat}(undef, 6)
        @test value_and_prox!(y, f, BigFloat[2.0, 0.5, 1.0, 2.0, -1.0, 1.0], 1) == 0.96875
        @test value_and_prox!(y, f, BigFloat[-1, -1, 1.5, -2, -0.5, 0.5], 1) == 0.265625
      end
    end
  end

  # the matrix: signed, tied and descending entries, two pairs of levels, three steps (a zero step among
  # them), a separate and an in-place output; the counts of the inputs whose stored sequence has an
  # inversion and of those whose has none, so that the matrix discriminates
  @testset "the matrix" begin
    levels = ((BigFloat(0.25), BigFloat(0.5)), (BigFloat(0.375), BigFloat(0.75)))
    counts = Dict(p => [0, 0] for p in (2, 3, 4, 5, 53, 256))
    for p in (2, 3, 4, 5, 53, 256)
      setprecision(BigFloat, p) do
        g = Lcg(UInt64(5000 + p))
        bits = min(p, 5)
        c = counts[p]
        for n in 3:10, (λ1, λ2) in levels, γ in (0, 1 // 2, 1)
          for _ in 1:20
            check_stored_value(BigFloat[pbit!(g, bits) for _ in 1:n], λ1, λ2, γ, c)
          end
          check_stored_value(BigFloat[pbit!(g, bits; signed = false) for _ in 1:n], λ1, λ2, γ, c)
          check_stored_value(fill(BigFloat(3), n), λ1, λ2, γ, c)                                 # ties
          check_stored_value(BigFloat[isodd(i) ? 2 : 3 for i in 1:n], λ1, λ2, γ, c)
          check_stored_value(step_input(n, n ÷ 2), λ1, λ2, γ, c)
          check_stored_value(reverse(step_input(n, n ÷ 2)), λ1, λ2, γ, c)
          check_stored_value(sort(BigFloat[pbit!(g, bits) for _ in 1:n]; rev = true), λ1, λ2, γ, c)   # descending
        end
      end
    end
    for p in (2, 3)
      @test counts[p][1] > 0           # inputs with an inversion of the stored sequence
      @test counts[p][2] > 0           # and inputs without
    end
  end

  # the plain types: the value returned is the value at the stored output
  @testset "$T" for T in (Float16, Float32, Float64)
    g = Lcg(UInt64(6000) + UInt64(sizeof(T)))
    n_cases = 0
    for n in 3:12, (l1, l2) in ((0.25, 0.5), (0.1, 0.3), (1.0, 0.5)), γ in (0.5, 1), _ in 1:60
      x = T[T(nextint!(g, 65) - 33) * T(0.37) for _ in 1:n]
      f = NormL1Fused(T(l1), T(l2))
      y = similar(x)
      v = value_and_prox!(y, f, x, γ)
      @test isfinite(v)
      @test v == f(y)
      n_cases += 1
    end
    @test n_cases > 0
  end
end

####################################
#  the pooling shared by IndIsotonic and SLOPE
####################################

@testset "pooling of IndIsotonic" begin
  d = BigFloat[12, 8, 6, 4, 3, 2, 1.5, 1, 0.75]       # decreasing, two significant bits each
  for p in (2, 3)
    setprecision(BigFloat, p) do
      seen = Set{Int}()
      for n in 3:9
        inputs = Vector{BigFloat}[d[1:n]]
        n >= 6 && push!(inputs, vcat(d[1:5], BigFloat[16, 24, 32, 48][1:(n - 5)]))
        n >= 8 && push!(inputs, vcat(d[1:7], BigFloat[24, 32][1:(n - 7)]))
        n >= 4 && push!(inputs, vcat(BigFloat[16, 24, 32, 48, 64, 96][1:(n - 3)], d[(n - 2):n]))
        for x in inputs
          # nondecreasing projection: the model of the pooling alone
          f = IndIsotonic()
          y0, sizes = m_pool(x, 1)
          union!(seen, sizes)
          check_prox(f, x, 1, y0)
          # the nonincreasing projection of the reversed input
          xr = reverse(x)
          g = IndIsotonic(decreasing = true)
          y1, sizes1 = m_pool(xr, -1)
          union!(seen, sizes1)
          check_prox(g, xr, 1, y1)
        end
      end
      @test issubset((5, 7, 9), seen)
    end
  end
end

@testset "pooling of SLOPE" begin
  # |x| sorted in decreasing order, less γλ_k, then pooled as nonincreasing and clipped at zero. The
  # entries are chosen so that the magnitudes are distinct or equal (equal magnitudes end in equal
  # outputs); only the sorting and the signs are added to the pooling, both exact.
  lams = (BigFloat[3, 3, 3, 2, 2, 2, 1, 1, 1], BigFloat[8, 6, 4, 3, 2, 1.5, 1, 0.75, 0.5])
  for p in (2, 3)
    setprecision(BigFloat, p) do
      seen = Set{Int}()
      for n in 3:9, γ in (1 // 2, 1), lam in lams
        λ = lam[1:n]
        f = SLOPE(λ)
        eight = BigFloat[8, -8, 8, -8, 8, -8, 8, -8, 8][1:n]
        sixteen = BigFloat[16, 16, -16, 16, 16, -16, 16, 16, -16][1:n]
        two_levels = BigFloat[i <= 5 ? (isodd(i) ? 8 : -8) : 1 for i in 1:n]
        graded = BigFloat[12, -8, 6, -4, 3, -2, 1.5, -1, 0.75][1:n]
        for x in (eight, sixteen, two_levels, graded, reverse(graded))
          y0, sizes = m_slope_prox(x, λ, γ)
          union!(seen, sizes)
          check_prox(f, x, γ, y0)
        end
      end
      @test issubset((5, 7, 9), seen)
    end
  end
end

####################################
#  special values keep their class
####################################

@testset "special values at two bits, n = 5" begin
  setprecision(BigFloat, 2) do
    f = NormL1Fused(BigFloat(1), BigFloat(Inf))
    inf = BigFloat(Inf)
    nan = BigFloat(NaN)
    for γ in (1 // 2, 1)
      # a NaN entry gives NaN everywhere, and the value NaN
      x = BigFloat[3, 3, nan, 3, 2]
      y = zeros(BigFloat, 5)
      v = value_and_prox!(y, f, x, γ)
      @test all(isnan, y) && isnan(v)
      # an infinite entry gives that infinity, both signs give NaN
      y = zeros(BigFloat, 5)
      prox!(y, f, BigFloat[3, inf, 2, 3, 2], γ)
      @test all(==(inf), y)
      prox!(y, f, BigFloat[3, -inf, 2, 3, 2], γ)
      @test all(==(-inf), y)
      prox!(y, f, BigFloat[3, inf, 2, -inf, 2], γ)
      @test all(isnan, y)
      # zeros of both signs
      z = BigFloat[0, -0.0, 0, -0.0, 0]
      v = value_and_prox!(y, f, z, γ)
      @test all(iszero, y) && iszero(v)
      # a constant vector
      c = fill(BigFloat(3), 5)
      zc = m_inf_prox(c, BigFloat(1), γ)
      y = zeros(BigFloat, 5)
      prox!(y, f, c, γ)
      @test y == fill(zc, 5)
      @test f(y) == m_equal_value(zc, 5, BigFloat(1))
      # finite levels: a NaN entry gives NaN everywhere
      ff = NormL1Fused(BigFloat(0.25), BigFloat(0.5))
      prox!(y, ff, x, γ)
      @test all(isnan, y)
      # zeros and a constant vector
      prox!(y, ff, z, γ)
      @test all(iszero, y)
      y0 = m_finite_prox(c, BigFloat(0.25), BigFloat(0.5), γ)[1]
      prox!(y, ff, c, γ)
      @test y == y0
    end
  end
end

####################################
#  the helpers of the integer operands
####################################

@testset "the integer-operand helpers" begin
  counts = Any[3, 4, 5, 6, 7, 8, 9, -5, 1, 2^40, typemax(Int64), typemin(Int64) + 1,
               Int128(typemax(Int64)) + 1, -Int128(typemax(Int64)) - 2, typemax(UInt64), big(2)^70, -big(2)^70,
               Int32(7), UInt8(5), typemax(Clong), typemin(Clong), Int128(typemax(Clong)) + 1]
  @testset "BigFloat, precision $p" for p in (2, 3, 5, 53, 256)
    setprecision(BigFloat, p) do
      xs = BigFloat[0, -0.0, 1.5, -1.5, 3, -7, 0.75, Inf, -Inf, NaN, floatmax(BigFloat), -floatmax(BigFloat),
                    Base.ldexp(BigFloat(1.5), Clong(10^8)), Base.ldexp(BigFloat(-1.5), Clong(-10^8))]
      for x in xs, k in counts
        @test same(PB._int_mul(x, k), mul_int(x, k))
        @test same(PB._int_mul(k, x), mul_int(x, k))
        @test same(PB._int_div(x, k), div_int(x, k))
      end
      for k in counts
        tiny = floatmin(BigFloat)
        @test same(PB._int_mul(tiny, k), mul_int(tiny, k))
      end
      # an entry with more bits than p, the operand of every count
      g = Lcg(UInt64(77 + p))
      for _ in 1:20
        x = setprecision(BigFloat, 256) do
          BigFloat(nextint!(g, 10^9)) / nextint!(g, 10^4)
        end
        for k in 3:9
          @test same(PB._int_mul(x, k), mul_int(x, k))
          @test same(PB._int_mul(k, x), mul_int(x, k))
          @test same(PB._int_div(x, k), div_int(x, k))
        end
      end
      # the exact products of the sites
      for n in 3:9
        for gap in BigFloat[1.5, -3, 0.75, 7, 0]
          for i in 1:(n - 1)
            @test same(PB._count_pair_mul(BigFloat, i, n - i, gap), mul_int(gap, i * (n - i)))
          end
          @test same(PB._count_mul(BigFloat, n, gap), mul_int(gap, n))
          for k in 1:n
            @test same(PB._int_mul(2k - n - 1, gap), mul_int(gap, 2k - n - 1))
            @test same(PB._int_mul(gap, 2k - n - 1), mul_int(gap, 2k - n - 1))
          end
        end
      end
      # the product i (n - i) of two 64-bit factors does not overflow
      big_gap = BigFloat(3)
      @test same(PB._count_pair_mul(BigFloat, typemax(Int64), typemax(Int64), big_gap),
                 mul_int(big_gap, big(typemax(Int64))^2))
    end
  end

  # The result has the current default precision, whatever the precision of the operand, as x * y
  # does for a BigFloat y: the integer is exact and the product is rounded once to that precision.
  @testset "the precision of the result" begin
    x = setprecision(() -> BigFloat(1) / 3, BigFloat, 256)
    setprecision(BigFloat, 5) do
      for r in (PB._int_mul(x, 7), PB._int_mul(7, x), PB._int_div(x, 7), PB._int_mul(x, big(2)^70))
        @test precision(r) == 5
      end
      @test PB._int_mul(x, 7) == mul_int(x, 7)
      @test PB._int_mul(7, x) == mul_int(x, 7)
      @test PB._int_div(x, 7) == div_int(x, 7)
    end
  end

  @testset "inference" begin
    for K in (Int, Int128, UInt64, Int32, BigInt)
      @test Base.return_types(PB._int_mul, (BigFloat, K)) == [BigFloat]
      @test Base.return_types(PB._int_mul, (K, BigFloat)) == [BigFloat]
      @test Base.return_types(PB._int_div, (BigFloat, K)) == [BigFloat]
    end
    @test Base.return_types(PB._count_mul, (Type{BigFloat}, Int, BigFloat)) == [BigFloat]
    @test Base.return_types(PB._count_pair_mul, (Type{BigFloat}, Int, Int, BigFloat)) == [BigFloat]
  end

  # the other floating types keep the operation as written, bit for bit
  @testset "$T is the plain operation" for T in (Float16, Float32, Float64)
    U = T === Float16 ? UInt16 : T === Float32 ? UInt32 : UInt64
    bits(v) = reinterpret(U, v)
    xs = T[0, -0.0, 1.5, -3.25, Inf, -Inf, NaN, floatmax(T), floatmin(T), nextfloat(T(0)), T(0.1), T(-7)]
    for x in xs, k in (3, 5, 7, 9, -5, 1)
      @test bits(PB._int_mul(x, k)) == bits(x * k)
      @test bits(PB._int_mul(k, x)) == bits(k * x)
      @test bits(PB._int_div(x, k)) == bits(x / k)
    end
    for n in 3:9, i in 1:(n - 1), gap in xs
      @test bits(PB._count_pair_mul(T, i, n - i, gap)) == bits((T(i) * T(n - i)) * gap)
      @test bits(PB._count_mul(T, n, gap)) == bits(T(n) * gap)
    end
  end
end

####################################
#  the pooling in means (entries near the end of the exponent range)
####################################

# The pooling of IndIsotonic in block means, for an entry too large for the sums: the mean of two blocks
# of ca and cb entries is a·wa + b·wb clamped into [min(a, b), max(a, b)], with the weights wa = ca/t and
# wb = cb/t of the exact integer t = ca + cb, each rounded once. The model records every merge.
function m_pool_means(x::Vector{BigFloat}, σ::Int)
  m = BigFloat[]
  c = Int[]
  merges = NamedTuple[]
  for e in x
    push!(m, e)
    push!(c, 1)
    while length(m) > 1 && (σ > 0 ? m[end - 1] > m[end] : m[end - 1] < m[end])
      a, b = m[end - 1], m[end]
      ca, cb = c[end - 1], c[end]
      t = ca + cb
      wa = rnd(Rational{BigInt}(ca, t))
      wb = rnd(Rational{BigInt}(cb, t))
      mean = clamp(a * wa + b * wb, min(a, b), max(a, b))
      push!(merges, (; ca, cb, wa, wb, a, b, mean))
      m[end - 1] = mean
      c[end - 1] += cb
      pop!(m)
      pop!(c)
    end
  end
  y = BigFloat[]
  for b in eachindex(m), _ in 1:c[b]
    push!(y, m[b])
  end
  y, merges
end

@testset "pooling in means at the end of the exponent range" begin
  # unit = 2^(E - 13): the entries are 2-bit multiples of it, at most 384 unit, below floatmax/16
  # (above the bound of the sums, so the pooling is redone in means)
  shapes = (("3+2", [12, 8, 6, 96, -192]),
            ("5+4", [12, 8, 6, 4, 3, 96, 48, 24, -192]),
            ("4+5", [12, 8, 6, 4, 96, 48, 24, 12, -384]))
  wanted = Dict("3+2" => (3, 2), "5+4" => (5, 4), "4+5" => (4, 5))
  for p in (2, 3, 5)
    setprecision(BigFloat, p) do
      E = Int(exponent(floatmax(BigFloat)))
      for (name, cs) in shapes, σ in (1, -1)
        n = length(cs)
        x = BigFloat[Base.ldexp(BigFloat(c), Clong(E - 13)) for c in cs]
        σ < 0 && (x = -x)
        bound = Base.ldexp(floatmax(BigFloat), Clong(-(2 * ceil(Int, log2(n)) + 2)))
        @test any(e -> abs(e) > bound, x)
        @test maximum(abs, x) < floatmax(BigFloat) / 8
        y0, merges = m_pool_means(x, σ)
        @test (wanted[name]) in [(mg.ca, mg.cb) for mg in merges]
        for mg in merges
          # the weights add to one up to their two roundings, and the mean stays between the two means
          @test abs(Rational{BigInt}(mg.wa) + Rational{BigInt}(mg.wb) - 1) <= 1 // big(2)^p
          @test min(mg.a, mg.b) <= mg.mean <= max(mg.a, mg.b)
        end
        f = IndIsotonic(decreasing = σ < 0)
        check_prox(f, x, 1, y0)
      end
    end
  end
end


####################################
#  the segment lengths of the total-variation kernel
####################################

# TotalVariation1D and FusedLasso denoise with Condat's recurrence. The recurrence divides by the
# length k - k0 + 1 of the current segment at three places: the common value of the last segment
# (:last), and the lower and the upper bound of the tube when the running sum leaves it (:lower and
# :upper). The length is an exact integer, so that at two bits 5 is 5 and not 4: each quotient is the
# exact quotient rounded once, as div_int. The model below is the recurrence written here, with the
# division at each site a parameter dv(x, k, site); when a vector rec is given, it records the site and
# the length of each division.
m_div(dv, rec, site::Symbol, x, k::Int) = (rec === nothing || push!(rec, (site, k)); dv(x, k, site))

function m_tv(x::Vector{T}, λ::T, dv, rec = nothing) where {T<:AbstractFloat}
  N = length(x)
  z = similar(x)
  λ > 0 || return copy(x)
  k = k0 = kmin = kplus = 1
  vmin, vmax = x[1] - λ, x[1] + λ
  umin, umax = λ, -λ
  while true
    while k == N
      if umin < 0                                      # a negative jump before the end
        for i in k0:kmin
          z[i] = vmin
        end
        kmin += 1
        k = k0 = kmin
        vmin = x[k]
        umin = λ
        umax = x[k] + λ - vmax
      elseif umax > 0                                  # a positive jump before the end
        for i in k0:kplus
          z[i] = vmax
        end
        kplus += 1
        k = k0 = kplus
        vmax = x[k]
        umax = -λ
        umin = x[k] - λ - vmin
      else                                             # the last segment
        v = vmin + m_div(dv, rec, :last, umin, k - k0 + 1)
        for i in k0:N
          z[i] = v
        end
        return z
      end
      if k == N                                        # the last entry alone
        z[N] = vmin + umin
        return z
      end
    end
    xk = x[k + 1]
    if xk + umin < vmin - λ                            # a negative jump
      for i in k0:kmin
        z[i] = vmin
      end
      kmin += 1
      k = k0 = kplus = kmin
      vmin = x[k]
      vmax = vmin + 2λ
      umin, umax = λ, -λ
    elseif xk + umax > vmax + λ                        # a positive jump
      for i in k0:kplus
        z[i] = vmax
      end
      kplus += 1
      k = k0 = kmin = kplus
      vmax = x[k]
      vmin = vmax - 2λ
      umin, umax = λ, -λ
    else                                               # no jump
      k += 1
      umin += xk - vmin
      umax += xk - vmax
      if umin >= λ
        vmin += m_div(dv, rec, :lower, umin - λ, k - k0 + 1)
        umin = λ
        kmin = k
      end
      if umax <= -λ
        vmax += m_div(dv, rec, :upper, umax + λ, k - k0 + 1)
        umax = -λ
        kplus = k
      end
    end
  end
end

const TV_SITES = (:last, :lower, :upper)

# the divisions of the model: the exact quotient rounded once, the quotient with the length rounded to
# the working precision first (a platform that rounds the integer), and that quotient at one site only
m_exact(x::BigFloat, k::Int, site) = div_int(x, k)
m_rounded(x::BigFloat, k::Int, site) = x / BigFloat(k)
m_rounded_at(s::Symbol) = (x::BigFloat, k::Int, site) -> site === s ? x / BigFloat(k) : div_int(x, k)
m_plain(x::AbstractFloat, k::Int, site) = x / k      # the arithmetic of the type, for Float32 and Float64

# The model on an input; nothing when the recurrence leaves 1:N. That happens at two bits on inputs such
# as x = [1, -3, -1.5, 1, 2, 1, -0.5], λ = 4: the state of the kernel is inconsistent there (a negative
# jump to the entry after the last) for a reason that is not the division, and such an input is never
# given to the package, which reads out of range under @inbounds.
function m_tv_or_nothing(x, λ, dv, rec = nothing)
  try
    m_tv(x, λ, dv, rec)
  catch e
    e isa BoundsError || rethrow()
    nothing
  end
end

# deterministic inputs at the precision p: the entries have at most min(p, 4) significant bits
function tv_inputs(p::Integer)
  g = Lcg(UInt64(3000 + p))
  bits = min(p, 4)
  cases = Tuple{Vector{BigFloat}, BigFloat}[]
  for n in 3:12, λ in BigFloat[0.25, 0.75, 1.5, 4, 32]
    for _ in 1:10
      push!(cases, (BigFloat[pbit!(g, bits) for _ in 1:n], λ))
    end
    push!(cases, (BigFloat[pbit!(g, bits; signed = false) for _ in 1:n], λ))
    push!(cases, (BigFloat[isodd(i) ? 3 : -3 for i in 1:n], λ))                    # alternating
    push!(cases, (BigFloat[isodd(i) ? 4 : -2 for i in 1:n], λ))
    push!(cases, (BigFloat[i % 3 == 0 ? 0.5 : 3 for i in 1:n], λ))
    push!(cases, (BigFloat[i % 4 == 0 ? 6 : 3 for i in 1:n], λ))                   # near the thresholds
    push!(cases, (fill(BigFloat(3), n), λ))                                         # ties
    push!(cases, (BigFloat[i + 1 for i in 1:n], λ))                                 # a ramp
    push!(cases, (BigFloat[i <= n ÷ 2 ? 2 : -2 for i in 1:n], λ))                   # two segments
    for k in (3, 5, 7)
      k < n && push!(cases, (step_input(n, k), λ))
    end
  end
  push!(cases, (BigFloat[4, -2, 0.5, 3, 3], BigFloat(4)))                          # the example
  cases
end

# the package against the model, with ==: prox, prox! into a separate output and in place, and
# value_and_prox! into a separate output, for TotalVariation1D and FusedLasso
function check_tv(x::Vector{BigFloat}, λ::BigFloat, γ, z::Vector{BigFloat})
  x0 = copy(x)
  f = TotalVariation1D(λ)
  @test prox(f, x, γ) == z
  @test x == x0
  y = similar(x)
  @test prox!(y, f, x, γ) === y
  @test y == z
  x2 = copy(x)
  prox!(x2, f, x2, γ)
  @test x2 == z
  y3 = similar(x)
  v = value_and_prox!(y3, f, x, γ)
  @test y3 == z
  @test v == f(y3)
  for λ1 in BigFloat[0.25, 1]
    g = FusedLasso(λ1, λ)
    zf = BigFloat[mshrink(v, γ * λ1) for v in z]
    @test prox(g, x, γ) == zf
    @test x == x0
    y = similar(x)
    @test prox!(y, g, x, γ) === y
    @test y == zf
    x2 = copy(x)
    prox!(x2, g, x2, γ)
    @test x2 == zf
    y3 = similar(x)
    v = value_and_prox!(y3, g, x, γ)
    @test y3 == zf
    @test v == g(y3)
  end
end

@testset "total-variation kernel: the segment length is exact" begin
  @testset "the example at two bits" begin
    setprecision(BigFloat, 2) do
      x = BigFloat[4, -2, 0.5, 3, 3]
      # the one segment has five entries: the divisor is 5 (and not 4, its value rounded to two bits)
      @test prox(TotalVariation1D(BigFloat(4)), x, 1) == fill(BigFloat(1.5), 5)
      @test prox(FusedLasso(BigFloat(0.25), BigFloat(4)), x, 1) == fill(BigFloat(1.0), 5)
      rec = Tuple{Symbol, Int}[]
      @test m_tv(x, BigFloat(4), m_exact, rec) == fill(BigFloat(1.5), 5)
      @test (:last, 5) in rec
      # the division by the length rounded to two bits gives the other value (2.0) that a platform
      # which rounds the integer gives
      @test m_tv(x, BigFloat(4), m_rounded) == fill(BigFloat(2.0), 5)
    end
  end

  @testset "package against the model, precision $p" for p in (2, 3, 4, 5, 53, 256)
    setprecision(BigFloat, p) do
      compared = 0
      for (x, λ) in tv_inputs(p), γ in (1, 2)
        z = m_tv_or_nothing(x, γ * λ, m_exact)
        z === nothing && continue
        compared += 1
        check_tv(x, λ, γ, z)
      end
      @test compared > 0
    end
  end

  @testset "each division is reached and discriminates (precision 2)" begin
    setprecision(BigFloat, 2) do
      lengths = Dict(s => Set{Int}() for s in TV_SITES)
      differs = Dict(s => 0 for s in TV_SITES)
      for (x, λ) in tv_inputs(2), γ in (1, 2)
        rec = Tuple{Symbol, Int}[]
        z = m_tv_or_nothing(x, γ * λ, m_exact, rec)
        z === nothing && continue
        for (s, k) in rec
          push!(lengths[s], k)
        end
        for s in TV_SITES
          zr = m_tv_or_nothing(x, γ * λ, m_rounded_at(s))
          zr === nothing || zr == z || (differs[s] += 1)
        end
      end
      for s in TV_SITES
        @test issubset((5, 7, 9), lengths[s])
        @test differs[s] > 0
      end
    end
  end

  @testset "$T is the plain quotient" for T in (Float32, Float64)
    U = T === Float32 ? UInt32 : UInt64
    bits(v) = reinterpret(U, v)
    g = Lcg(UInt64(4000) + UInt64(sizeof(T)))
    compared = 0
    for n in 3:12, λ in T[0.1, 0.5, 1.25, 3, 10], rep in 1:60
      x = T[T(nextint!(g, 2001) - 1001) / 100 for _ in 1:n]
      z = m_tv_or_nothing(x, λ, m_plain)
      z === nothing && continue
      compared += 1
      y = similar(x)
      prox!(y, TotalVariation1D(λ), x, 1)
      @test bits.(y) == bits.(z)
      for λ1 in T[0.2, 1]
        yf = similar(x)
        prox!(yf, FusedLasso(λ1, λ), x, 1)
        @test bits.(yf) == bits.(mshrink.(z, λ1))
      end
    end
    @test compared > 0
  end
end

end
