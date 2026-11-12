module NumericalKernelsTest

# The shared numerical kernels of src/numerical_kernels.jl (issue #865), tested at the primitive level in
# Float32 and Float64: finite inputs whose intermediate products, differences or squares leave the
# range while the result is representable, underflow restored by a later factor, genuine NaN/Inf
# inputs, exact zeros and signed zeros. BigFloat is the test oracle only; the kernels never use it.

using Test, ProximalBase, Random, SparseArrays
using LinearAlgebra: norm
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
      a, b = signif(rng, T, rand(rng, e0:E+1)), signif(rng, T, rand(rng, e0:E+1))
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
    scales = (e0 + 2, e0 ÷ 2 - 3, -20, 0, 20, E ÷ 2 + 3, E - 1, E, E + 1)   # signif(·, E + 1) is the top binade
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
    # stored entries only, with the logical multiplicity of a symmetric iterate
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

  @testset "SqrNormL2 on the shared kernels" begin
    # the gradient λ(x - c) where x - c overflows and λ restores the range; the value and the fused
    # oracle on the same data
    for _ in 1:300
      x, c = abs(signif(rng, T, E + 1)), -abs(signif(rng, T, rand(rng, E:E+1)))   # x - c overflows
      λ = abs(signif(rng, T, rand(rng, -E:-2)))
      r = big(λ) * (big(x) - big(c))
      abs(r) <= floatmax(T) || continue
      f = SqrNormL2(λ, [c])
      g = gradient!(zeros(T, 1), f, [x])
      @test within(g[1], r, 2)
      g2 = zeros(T, 1)
      v = value_and_gradient!(g2, f, [x])
      @test g2 == g && v === f([x])
      @test within(v, big(λ) * (big(x) - big(c))^2 / 2, 3)
    end
    # a genuine infinite level keeps the IEEE gradient λ(x - c), also for a subnormal residual,
    # which halving would have rounded to zero (NaN on master)
    δ = nextfloat(zero(T))
    g = gradient!(zeros(T, 3), SqrNormL2(T(Inf), T[0, 0, 1]), T[-δ, 3δ, 1])
    @test g[1] === T(-Inf) && g[2] === T(Inf) && isnan(g[3])
    # centers are rounded once as the value rounds them, in the value and in the value parts:
    # an entry that leaves T's range on rounding is read unrounded, the others are rounded to T
    W = T === Float32 ? Float64 : BigFloat
    # the second entry rounds to exactly 1 in T: read unrounded, its residual would dominate the value
    c = W[ldexp(W(1), e0 - 4), 1 + ldexp(W(1), -(precision(T) + 20))]
    λ = pow2(T, E)
    x = T[0, 1]
    cr = [big(c[1]), big(T(c[2]))]
    ref = big(λ) * sum(abs2, big.(x) .- cr) / 2
    f = SqrNormL2(λ, c)
    @test within(f(x), ref, 2)
    α = pow2(T, -E ÷ 2)
    @test within(Scaled(f, α)(x), big(α) * ref, 2)
    λs = pow2(T, -E + 2)
    xs = T[0, 0]
    ref_small = big(λs) * sum(abs2, big.(xs) .- cr) / 2
    @test within(Scaled(SqrNormL2(λs, c), pow2(T, E - 1))(xs), big(pow2(T, E - 1)) * ref_small, 2)
    # stored iterates: O(nnz) reductions with the logical multiplicity, on the recovery path
    xsp = SparseIterate(T, 6)
    xsp[2], xsp[5] = pow2(T, E - 1), T(-3)
    xm = SymmetricSparseIterate(T, 3)
    xm[2, 1], xm[3, 3] = pow2(T, E - 1), T(-0.75) * pow2(T, E - 1)
    λt = pow2(T, -E + 1)
    for z in (xsp, xm)
      dense = z isa SparseIterate ? Vector(z) : Matrix(z)
      @test within(SqrNormL2(λt)(z), big(λt) * sum(abs2, big.(dense)) / 2, 2)
      @test SqrNormL2(λt)(z) === SqrNormL2(λt)(dense)
    end
    # prox: the coordinate and array methods, and the in-place alias, agree at exceptional scales
    for (γ, λp) in ((pow2(T, E - 1), pow2(T, E - 1)), (pow2(T, -E), pow2(T, e0)), (T(1), T(3)))
      x = T[pow2(T, E), -pow2(T, e0), T(0.3)]
      cc = T[-pow2(T, E - 1), pow2(T, e0), T(1)]
      for f in (SqrNormL2(λp), SqrNormL2(λp, cc))
        z = prox!(similar(x), f, x, γ)
        @test z == [prox_block(f, x[i], i, γ) for i in eachindex(x)]
        xa = copy(x)
        @test prox!(xa, f, xa, γ) == z
      end
    end
  end

  @testset "recomputed-dot units" begin
    # no partial sum of n products, scaled by 2^-k, can reach 2^(emax - margin)
    for _ in 1:500
      ma, mb = abs(signif(rng, T, rand(rng, e0:E))), abs(signif(rng, T, rand(rng, e0:E)))
      n, margin = rand(rng, 1:1000), rand(rng, 2:3)
      k = PB._product_sum_shift(T, ma, mb, n, margin)
      @test k >= 0
      @test big(n) * big(ma) * big(mb) * big(2.0)^(-k) < big(2.0)^(E + 1 - margin)
      k > 0 && @test big(n) * big(ma) * big(mb) * big(2.0)^(1 - k) >= big(2.0)^(E - 1 - margin - 2)
    end
    @test PB._product_sum_shift(T, T(1), T(1), 1, 2) == 0
    @test PB._product_sum_shift(T, T(Inf), T(1), 10, 2) == 0
  end

  @testset "LeastSquaresLoss and LinearFunction on the shared kernels" begin
    ref(Y, X, β) = sum(abs2, big.(Y) .- big.(X) * big.(β)) / 2
    h = E ÷ 2
    cases = Any[
      # residual squares overflow, value representable only as Inf or not: the reference decides
      (T[pow2(T, h + 2), 1], T[1 0; 0 1], T[0, 0]),
      # residual squares underflow, value restored by summing in power-of-two units
      (T[pow2(T, e0 ÷ 2 - 4), -pow2(T, e0 ÷ 2 - 3)], T[1 0; 0 1], T[0, 0]),
      # Y - Xβ overflows while Xβ is finite (halved residual)
      (T[floatmax(T), 1], T[1 0; 0 1], T[-floatmax(T) / 2, 0]),
      # Xβ overflows in its partial products although Xβ = 0 and the value is representable
      (T[3, -2], T[pow2(T, h + 4) pow2(T, h + 4); 1 1], T[pow2(T, h + 4), -pow2(T, h + 4)]),
      # an exact fit
      (T[pow2(T, E - 3), 1], T[1 0; 0 1], T[pow2(T, E - 3), 1]),
    ]
    for (Y, X, β) in cases
      f = LeastSquaresLoss(Y, X)
      r = ref(Y, X, β)
      v = f(β)
      @test within(v, r, 3)
      g = similar(β)
      @test within(value_and_gradient!(g, f, β), r, 3)
      @test isequal(g, gradient!(similar(β), f, β))   # the gradient itself has no recovery path
      bs = SparseIterate(T, length(β))
      for i in eachindex(β)
        iszero(β[i]) || (bs[i] = β[i])
      end
      @test f(bs) === v
    end
    # genuine NaN/Inf data keep the ordinary result
    f = LeastSquaresLoss(T[NaN, 1], T[1 0; 0 1])
    @test isnan(f(T[1, 1]))
    @test LeastSquaresLoss(T[Inf, 1], T[1 0; 0 1])(T[1, 1]) === T(Inf)
    @test isnan(LeastSquaresLoss(T[1, 1], T[1 0; 0 1])(T[Inf, -Inf]) - T(Inf)) # Inf or NaN, never finite
    # LinearFunction: an overflowing dot at a representable value, with d in its own units
    a = pow2(T, h + 4)
    for (c, x, d) in ((T[a, a, 1], T[a, -a, 1], T(1)), (T[floatmax(T), floatmax(T)], T[1, -1], T(0.25)),
                      (T[a, a], T[a, -a], pow2(T, e0)))
      rr = sum(big.(c) .* big.(x)) + big(d)
      @test within(LinearFunction(c, d)(x), rr, 1)
    end
    @test LinearFunction(T[1, 1], T(0))(T[Inf, 1]) === T(Inf)
    @test isnan(LinearFunction(T[1, 1], T(0))(T[Inf, -Inf]))
  end

  @testset "Euclidean norm recovery" begin
    function owner(vals, p)
      S = SymmetricSparseIterate(T, p)
      k = 0
      for c in 1:p, r in c:p
        k += 1
        k <= length(vals) && (S[r, c] = vals[k])
      end
      S
    end
    frob(S, p) = sqrt(sum(big(S[r, c])^2 for r in 1:p, c in 1:p; init = big(0.0)))
    # residuals whose squares overflow, underflow or are subnormal, and mixed scales; the
    # recomputation (a sum of squares that is not a finite normal number) is within 2 ulps
    es = exponent(nextfloat(zero(T)))
    for (lo, hi) in ((E ÷ 2 + 1, E - 4), (e0 ÷ 2 - 20, e0 ÷ 2 - 1), (es + 3, e0 - 1), (e0 + 2, E - 6))
      for _ in 1:300
        n = rand(rng, 1:30)
        x = T[signif(rng, T, rand(rng, lo:hi)) for _ in 1:n]
        y = T[signif(rng, T, rand(rng, lo:hi)) for _ in 1:n]
        rand(rng) < 0.2 && (y .= nextfloat.(x, rand(rng, -3:3)))    # cancellation
        r = sqrt(sum((big.(x) .- big.(y)) .^ 2))
        s = sum(abs2, x .- y)
        (isfinite(s) && s >= floatmin(T)) || @test within(norm_diff(x, y), r, 2)
        p = rand(rng, 1:7)
        S = owner(x[1:min(n, p * (p + 1) ÷ 2)], p)
        # every scale, O(nnz); the ordinary sum of m squares adds up to (m - 1)/4 ulps to the norm
        @test within(norm(S), frob(S, p), 2 + S.nnz / 4)
      end
    end
    # one residual whose square overflows, a finite norm; an overflowing difference is a norm beyond range
    fm = floatmax(T)
    @test within(norm_diff(T[0.6fm, 0], T[-0.1fm, 0]), big(0.7) * big(fm), 1)
    @test norm_diff(T[0.75fm], T[-0.5fm]) === T(Inf)
    S = owner(T[0, pow2(T, E - 2), 0], 2)                            # an off-diagonal entry counts twice
    @test within(norm(S), sqrt(big(2.0)) * big(pow2(T, E - 2)), 1)
    S = owner(T[pow2(T, es), -pow2(T, es + 1), pow2(T, es)], 2)       # subnormal entries only
    @test within(norm(S), sqrt(big(1.0) + 2 * 4 + 1) * big(pow2(T, es)), 1)
    @test (@allocated norm(S)) == 0
    # genuine NaN/Inf keep the ordinary result; exact and signed zeros give +0
    @test isnan(norm_diff(T[NaN, 1], T[0, 1])) && isnan(norm_diff(T[Inf, NaN], T[0, 0]))
    @test isnan(norm_diff(T[Inf], T[Inf]))                              # Inf - Inf
    @test norm_diff(T[Inf, pow2(T, e0 - 4)], T[0, 0]) === T(Inf)
    @test norm_diff(T[-0.0, 0], T[0, -0.0]) === zero(T)
    @test norm_diff(T[pow2(T, es), 1], T[pow2(T, es), 1]) === zero(T)
    @test isnan(norm(owner(T[pow2(T, E - 1), NaN], 2))) && norm(owner(T[Inf, pow2(T, es)], 2)) === T(Inf)
    @test norm(owner(T[-0.0, -0.0], 2)) === zero(T) && norm(SymmetricSparseIterate(T, 3)) === zero(T)
  end
end
end


# #767: the retained exponent of a _ProjectionScale is an Int that may already have wrapped. The
# retained products sum the exponents in Int, modulo 2^64, so they are right whenever the final
# exponent fits in Int. The ordinary kernel widens BigFloat exponents to Int128 and would read the
# wrapped power at face value. MPFR exponents near -2^62 let two factors carry the wrap back.
setprecision(BigFloat, 256) do
@testset "retained exponent across Int wraparound (#767)" begin
  E = exponent(floatmax(BigFloat))
  a, b = ldexp(big(0.75), -E + 10), ldexp(big(0.625), -E + 10)
  (ma, ea), (mb, eb) = PB._frexp(a), PB._frexp(b)
  # a small third factor gives negative final exponents with the same wrapped power
  for d in (0, 40, 1000), c in (ldexp(big(0.875), 3), ldexp(big(0.875), -3000))
    mc, ec = PB._frexp(c)
    K = Int128(typemax(Int)) + 1 + d          # the true power: not an Int
    kw = K % Int                              # as a _ProjectionScale stores it, wrapped
    final3, final2 = Int128(ea) + eb + ec + K, Int128(ea) + eb + 1 + K   # c = 1 has exponent 1
    # the intended boundary: the power is not an Int, the widened sum of the stored exponents
    # leaves Int (so widening gives the wrong exponent), and the true final exponent fits in Int
    inint(e) = typemin(Int) <= e <= typemax(Int)
    @test !inint(K)
    @test !inint(Int128(ea) + eb + ec + kw) && !inint(Int128(ea) + eb + 1 + kw)
    @test inint(final3) && inint(final2)
    ref3 = (ma * mb * mc) * big(2.0)^Int(final3)
    ref2 = (ma * mb / 2) * big(2.0)^Int(final2)
    @test PB._scaled_product_retained(a, b, c, kw) == ref3
    @test PB._scaled_product_retained(a, b, kw) == ref2
    # the live caller: a scale m 2^e applied to parts whose exponent is added to the retained one
    @test PB._apply_parts(PB._ProjectionScale(a, kw), b, 0) == ldexp(a, -ea) * ldexp(b, -eb) * big(2.0)^Int(Int128(ea) + eb + K)
    # widening the wrapped power, as the ordinary kernel does, loses the value
    @test PB._scaled_product(a, b, c, kw) != ref3
  end
end
end

end
