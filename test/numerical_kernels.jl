module NumericalKernelsTest

# The shared numerical kernels of src/utils.jl (issue #865), tested at the primitive level in
# Float32 and Float64: finite inputs whose intermediate products, differences or squares leave the
# range while the result is representable, underflow restored by a later factor, genuine NaN/Inf
# inputs, exact zeros and signed zeros. BigFloat is the test oracle only; the kernels never use it.

using Test, ProximalBase, Random, SparseArrays
const PB = ProximalBase

pow2(T, k) = ldexp(one(T), k)
emax(T) = exponent(floatmax(T))
emin(T) = exponent(floatmin(T))
# |a - r| within n units in the last place of the rounded reference (subnormal units included)
function within(a::T, r::BigFloat, n) where {T}
  t = T(r)
  isfinite(t) || return a == t
  abs(big(a) - r) <= n * eps(abs(t))
end
relerr(a::BigFloat, r::BigFloat) = iszero(r) ? abs(a) : abs(a - r) / abs(r)
# a random significand in [1/2, 1) with a random sign, times 2^e
signif(rng, T, e) = (rand(rng, Bool) ? -1 : 1) * ldexp(T(0.5) + T(rand(rng)) / 2, e)

setprecision(BigFloat, 1200) do
@testset "numerical kernels $T" for T in (Float64, Float32)
  rng = Xoshiro(865)
  E, e0 = emax(T), emin(T)

  @testset "bounded-product kernel" begin
    # finite factors whose partial products leave the range, with a representable final product
    for _ in 1:2000
      ea, eb = rand(rng, -E:E), rand(rng, -E:E)
      ec = rand(rng, -E:E)
      k = rand(rng, -2E:2E)
      a, b, c = signif(rng, T, ea), signif(rng, T, eb), signif(rng, T, ec)
      r = big(a) * big(b) * big(c) * big(2.0)^k
      (abs(r) > floatmax(T) || abs(r) < nextfloat(zero(T))) && continue
      @test within(PB._scaled_product(a, b, c, k), r, 2)
      w = abs(a)
      @test within(PB._scaled_product(w, b, k), big(w) * big(b) * big(2.0)^k, 2)
      k == 0 && @test within(PB._exceptional_product(a, b, c), r, 2)
    end
    # the ordinary product is kept while ab is normal; outside, a later factor restores the range
    a, b = T(3), T(5) / 7
    @test PB._positive_product(a, b, T(11)) === (a * b) * T(11)
    @test PB._positive_product(pow2(T, -E + 4), pow2(T, -E + 4), pow2(T, E)) === pow2(T, -E + 8)
    @test PB._positive_product(pow2(T, E), pow2(T, E), pow2(T, -E - 3)) === pow2(T, E - 3)
    # a subnormal final product: no later factor, rounded once from the scaled significands
    @test within(PB._positive_product(pow2(T, e0), T(0.75), T(0.5)), big(0.375) * big(2.0)^e0, 1)
    # zero factors: +0 from the three-factor kernel (a zero beats an infinite factor), w v from the weighted one
    @test PB._scaled_product(zero(T), T(-3), T(2), 5) === zero(T)
    @test PB._scaled_product(T(Inf), zero(T), T(2), 5) === zero(T)
    @test PB._exceptional_product(T(Inf), zero(T), T(2)) === zero(T)
    @test PB._scaled_product(T(2), -zero(T), 5) === -zero(T)
    @test PB._scaled_product(zero(T), T(-3), 5) === -zero(T)
    # genuine NaN and Inf factors give the ordinary product
    @test isnan(PB._scaled_product(T(NaN), T(2), T(3), 0))
    @test PB._scaled_product(T(-Inf), T(2), T(3), -10) === T(-Inf)
    @test PB._scaled_product(T(Inf), T(-2), 7) === T(-Inf)
    @test isnan(PB._scaled_product(T(2), T(NaN), 7))
    # the retained-exponent variant agrees with the kernel where the exponent does not wrap
    for _ in 1:200
      w, v, k = abs(signif(rng, T, rand(rng, -E:E))), signif(rng, T, rand(rng, -E:E)), rand(rng, -E:E)
      @test PB._scaled_product_retained(w, v, k) === PB._scaled_product(w, v, k)
    end
  end

  @testset "halved difference" begin
    # a finite difference is exact units (h = 0) and bitwise the ordinary one
    for _ in 1:500
      a, b = signif(rng, T, rand(rng, e0:E)), signif(rng, T, rand(rng, e0:E))
      d, h = PB._halved_difference(a, b, T)
      if isfinite(a - b)
        @test (d, h) === (a - b, 0)
      else
        # an overflowing difference of finite operands: a/2 - b/2, one binade up, finite
        @test h == 1 && isfinite(d)
        @test within(d, (big(a) - big(b)) / 2, 1)
      end
    end
    d, h = PB._halved_difference(floatmax(T), -floatmax(T), T)
    @test (d, h) === (floatmax(T), 1)
    @test PB._residual_exponent(floatmax(T), -floatmax(T), T) == E + 1
    # cancellation keeps its units: the difference is formed before any scaling
    @test PB._halved_difference(nextfloat(T(1)), T(1), T) === (eps(T), 0)
    # zeros, and genuine Inf/NaN operands, which the callers screen before using the result
    @test PB._residual_exponent(T(3), T(3), T) == typemin(Int)
    @test PB._residual_exponent(T(Inf), T(3), T) == typemax(Int)
    @test PB._residual_exponent(T(3), T(NaN), T) == typemax(Int)
    # integer operands take a widened integer difference
    @test PB._halved_difference(typemin(Int), typemax(Int), Float64) === (-18446744073709551615.0, 0)
  end

  @testset "squared-residual kernel" begin
    ref(x, y) = sum(abs2, big.(collect(x)) .- (y === nothing ? 0 : big.(y)); init = big(0))
    parts_value(q) = big(q[1]) * big(2.0)^(2q[2])
    scales = (e0 + 2, e0 ÷ 2 - 3, -20, 0, 20, E ÷ 2 + 3, E - 1, E)
    for _ in 1:400, n in (1, 4, 13)
      x = T[signif(rng, T, rand(rng, scales)) for _ in 1:n]
      y = rand(rng, Bool) ? nothing : T[signif(rng, T, rand(rng, scales)) for _ in 1:n]
      q = PB._residual_squares(x, y, T)
      @test q isa Tuple{T, Int}
      @test relerr(parts_value(q), ref(x, y)) <= (n + 4) * eps(T)
    end
    # every square overflows; every square underflows; the differences overflow
    for (x, y) in ((fill(pow2(T, E - 2), 3), nothing), (fill(pow2(T, e0 + 1), 3), nothing),
                   ([floatmax(T), T(1)], [-floatmax(T), T(1)]),
                   ([T(1), pow2(T, -E)], [nextfloat(T(1)), zero(T)]))   # cancellation beside a tiny residual
      q = PB._residual_squares(x, y, T)
      @test relerr(parts_value(q), ref(x, y)) <= 6eps(T)
    end
    # stored entries only, with the logical multiplicity of a symmetric owner
    xs = PB.SparseIterate(T, 9)
    xs[2], xs[7] = pow2(T, E - 1), T(-3)
    @test relerr(parts_value(PB._residual_squares(xs, nothing, T)), ref(Vector(xs), nothing)) <= 4eps(T)
    xm = PB.SymmetricSparseIterate(T, 3)
    xm[2, 1], xm[3, 3] = pow2(T, e0), T(-0.75) * pow2(T, e0)
    @test relerr(parts_value(PB._residual_squares(xm, nothing, T)), ref(Matrix(xm), nothing)) <= 4eps(T)
    # exact zero residuals; genuine Inf/NaN entries give `nothing` (the ordinary result stands)
    @test PB._residual_squares(zeros(T, 3), nothing, T) === (zero(T), 0)
    @test PB._residual_squares(T[1, -2], T[1, -2], T) === (zero(T), 0)
    @test PB._residual_squares(T[1, Inf], nothing, T) === nothing
    @test PB._residual_squares(T[1, 2], T[NaN, 2], T) === nothing
  end

  @testset "ordinary sum-of-squares trigger" begin
    n = 7
    t = n * (floatmin(T) / eps(T))
    @test PB._squares_normal(t, n, T)
    @test !PB._squares_normal(prevfloat(t), n, T)
    @test PB._squares_normal(T(1), n, T)
  end

  @testset "scaled square value and parts" begin
    # λ/2 Σ (xᵢ - yᵢ)² where the squares overflow or underflow and λ restores the range
    for (λ, x, y) in ((pow2(T, -E + 2), fill(pow2(T, E - 2), 4), nothing),
                      (pow2(T, E - 4), fill(pow2(T, e0 ÷ 2 - 2), 4), nothing),
                      (pow2(T, -E), [floatmax(T), T(2)], [-floatmax(T), T(1)]))
      r = big(λ) * sum(abs2, big.(x) .- (y === nothing ? 0 : big.(y))) / 2
      @test within(PB._square_value(λ, x, y), r, 2)
      m, e = PB._square_parts(λ, x, y)
      @test 0.5 <= abs(m) < 1
      @test relerr(big(m) * big(2.0)^e, r) <= 4eps(T)
    end
    # genuine nonfinite entries keep the ordinary value; a zero or infinite level has no parts
    @test PB._square_value(T(1), T[Inf, 1], nothing) === T(Inf)
    @test isnan(PB._square_value(T(1), T[NaN, 1], nothing))
    @test PB._square_parts(T(1), T[Inf, 1], nothing) === nothing
    @test PB._square_parts(zero(T), T[1, 1], nothing) === nothing
    @test PB._square_parts(T(Inf), T[1, 1], nothing) === nothing
    @test PB._square_parts(T(2), zeros(T, 2), nothing) === (zero(T), 0)
  end
end
end

end
