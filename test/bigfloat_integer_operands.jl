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

# x * k and x / k with the integer k exact, rounded once. The exponent of x is split off (the
# significand has at most p bits and lies in [1/2, 1)), so that an entry near floatmax is not expanded.
function mul_int(x::BigFloat, k::Integer)
  isnan(x) && return x
  if iszero(x) || isinf(x)
    isinf(x) && iszero(k) && return BigFloat(NaN)
    return k < 0 ? -x : copy(x)
  end
  fr, ex = frexp(x)
  Base.ldexp(rnd(Rational{BigInt}(fr) * big(k)), ex)
end
function div_int(x::BigFloat, k::Integer)
  iszero(k) && throw(DivideError())
  isnan(x) && return x
  (iszero(x) || isinf(x)) && return k < 0 ? -x : copy(x)
  fr, ex = frexp(x)
  Base.ldexp(rnd(Rational{BigInt}(fr) / big(k)), ex)
end

# the soft threshold, as the package forms it
mshrink(v::BigFloat, c::BigFloat) = max(v - c, zero(v)) + min(v + c, zero(v))

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

# the proximal point of NormL1Fused(λ1, λ2) at x (length > 2), finite levels, a positive finite step:
# the output in the order of x, the sizes of the pooled blocks, and the output in the order of the sorted
# entries (at a precision of a few bits the pooled means need not be increasing in that order: the
# value of value_and_prox! is formed in it, and the value of f(y) after sorting y)
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
function m_finite_value(y::Vector{BigFloat}, λ1::BigFloat, λ2::BigFloat; presorted::Bool = false)
  n = length(y)
  a = presorted ? y : sort(y)
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

function check_inf(x::Vector{BigFloat}, λ1::BigFloat, γ; value::Bool = BigFloat(length(x)) == length(x))
  n = length(x)
  f = NormL1Fused(λ1, BigFloat(Inf))
  z = m_inf_prox(x, λ1, γ)
  y0 = fill(z, n)
  y, v = check_prox(f, x, γ, y0)
  @test all(==(y[1]), y)
  # The product with the factor n of the value is tested separately: it differs from the model
  # wherever p bits cannot hold n.
  if value
    want = m_equal_value(z, n, λ1)
    @test v == want
    @test f(y) == want
  end
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
  y0, sizes, ys = m_finite_prox(x, λ1, λ2, γ)
  union!(seen, sizes)
  y, v = check_prox(f, x, γ, y0)
  # the value at the stored output; a value that the ordinary expression does not give (a fusion sum
  # that rounding made negative) is recovered by a scaled sum and is not modelled
  want, ok = m_finite_value(ys, λ1, λ2; presorted = true)
  ok && @test v == want
  want, ok = m_finite_value(y0, λ1, λ2)
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

end
