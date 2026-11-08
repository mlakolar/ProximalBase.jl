module RetainedScaleExponentsTest

# The factors of nested `Scaled` wrappers whose product is not a normal number are retained as a significand and an
# exponent. MPFR exponents reach ±(2^62 - 1) on 64-bit Unix, so a chain of three extreme BigFloat factors has an exponent
# beyond Int64, and the retained exponent must be exact: read on its own (a comparison with one, a shift) and after it
# returns into range through a level, a weight or a step. The oracle is independent of the retained representation: the
# factors are exact powers of two, the true exponent is summed in Int128, and the outcome is classified (finite, ±Inf,
# ±0, sign) with the MPFR bounds before any value is compared. Every case is built from exponents relative to the
# platform's own range, so the same body runs where the range is smaller.

using Test
using ProximalBase

const PB = ProximalBase

const EMAX = Int128(Base.MPFR.get_emax())
const EMIN = Int128(Base.MPFR.get_emin())
const PRECISIONS = (2, 53, 256)

# the power of two 2^n, exact at every precision
function pw(n::Integer)
  @assert EMIN - 1 <= n <= EMAX - 1
  Base.ldexp(one(BigFloat), Clong(n))
end

# the number s 2^N by the rules of MPFR: class and sign first (N == EMIN - 2 is a tie of the rounding and is avoided)
function oracle(N::Integer, s::Integer = 1)
  @assert N != EMIN - 2
  N >= EMAX && return s * big(Inf)
  N <= EMIN - 3 && return s * big(0.0)
  s * pw(N)
end

function agrees(r, N::Integer, s::Integer = 1)
  e = oracle(N, s)
  isinf(e) && return isinf(r) && signbit(r) == signbit(e)
  iszero(e) && return iszero(r) && signbit(r) == signbit(e)
  r == e
end

function expect(r, N::Integer, s::Integer = 1; what = "")
  ok = agrees(r, N, s)
  ok || @info "mismatch" what result = r expected = oracle(N, s) N
  @test ok
end

function with_precisions(f, name; precisions = PRECISIONS)
  @testset "$name, precision $p" for p in precisions
    setprecision(BigFloat, p) do
      f()
    end
  end
end

# the function f under the factors 2^e for e in exps, the first one innermost
chain(f, exps) = foldl((g, e) -> Scaled(g, pw(e)), exps; init = f)
depth(f) = f isa Scaled ? 1 + depth(f.f) : 0
exponent_sum(exps) = sum(Int128, exps)

# the retained product of a chain, whose value is 0.5 2^e
retained(s) = PB._scaled_chain(s)[2]

frac(n) = (EMAX * n) ÷ 1000
const A = EMAX - 2                   # an extreme factor: A + A + A leaves Int64 where the range is the default one
const P = frac(680)                  # three factors 2^P leave Int64 and are cancelled by a level and a weight 2^-Q
const Q = frac(650)
const Y = frac(760)

const BEYOND = EMAX > 2^61           # 3 P > typemax(Int64) only for the default range

# chains of extreme factors, and chains of five to seven factors that have a retained exponent beyond Int64 as well
const POSITIVE = [[A, A, A], [P, P, P], [P, P, P, P], fill(frac(500), 5), fill(frac(450), 6), fill(frac(400), 7)]
const NEGATIVE = [-1 .* F for F in POSITIVE]
const CHAINS = vcat(POSITIVE, NEGATIVE)
# chains whose exponent is just beyond Int64 (at the default range), so that a level and a step, or a level and a weight,
# at the ends of their ranges bring the product back into range
const LONG = [[P, P, P], fill(frac(668), 3), fill(frac(401), 5), fill(frac(334), 6), fill(frac(287), 7)]
const LONG_CHAINS = vcat(LONG, [-1 .* F for F in LONG])

# the splits of an exponent between a level l, a weight w and a step g, each within the range of the exponents
function splits(need)
  out = Tuple{String, Int128, Int128, Int128}[]
  for (name, mask) in (("level", (1, 0, 0)), ("weight", (0, 1, 0)), ("step", (0, 0, 1)), ("level and weight", (1, 1, 0)),
                       ("level and step", (1, 0, 1)), ("weight and step", (0, 1, 1)), ("level, weight and step", (1, 1, 1)))
    c = sum(mask)
    base = fld(need, c)
    extra = need - base * c
    parts = [base + (i <= extra ? 1 : 0) for i in 1:c]
    all(p -> EMIN - 1 <= p <= EMAX - 1, parts) || continue
    vals = Int128[]
    j = 0
    for m in mask
      push!(vals, m == 1 ? parts[j += 1] : Int128(0))
    end
    push!(out, (name, vals...))
  end
  out
end
# the totals T that the splits aim at: a threshold 2^T well inside the range, on the side of the sign of the exponent
targets(S) = S > 0 ? (frac(250), frac(50)) : (-frac(250), -frac(50))

# prox of the chain of NormL1(2^l, [2^w]) with the step 2^g: soft-thresholding at 2^T, T = sum(F) + l + w + g
function check_prox(F, l, w, g, what)
  T = exponent_sum(F) + l + w + g
  s = chain(NormL1(pw(l), [pw(w)]), F)
  γ = pw(g)
  @testset "$what" begin
    if T >= EMAX                         # an infinite threshold
      @test prox(s, [big(1.0)], γ) == [big(0.0)]
    elseif T <= EMIN - 3                 # a zero threshold
      x = [big(1.0)]
      @test prox(s, x, γ) == x
    elseif T >= EMIN - 1 && T + 1 <= EMAX - 1
      @test prox(s, [pw(T + 1)], γ) == [pw(T)]
      @test prox(s, [-pw(T + 1)], γ) == [-pw(T)]
      T - 1 >= EMIN - 1 && @test prox(s, [pw(T - 1)], γ) == [big(0.0)]
    end
  end
end

@testset "setup" begin
  setprecision(BigFloat, 53) do
    for F in POSITIVE
      s = chain(NormL1(big(1.0)), F)
      a = retained(s)
      @test a isa PB._ProjectionScale{BigFloat, Int128}
      @test Int128(a.e) - 1 == exponent_sum(F)
      @test depth(s) >= 3
      BEYOND && @test a.e > typemax(Int64)
    end
    BEYOND && @test 3 * Int128(P) > typemax(Int64)
    for F in NEGATIVE
      a = retained(chain(NormL1(big(1.0)), F))
      @test Int128(a.e) - 1 == exponent_sum(F)
      BEYOND && @test a.e < typemin(Int64)
    end
  end
end

@testset "the cases of the issue" begin
  with_precisions("a chain of three extreme factors") do
    s3 = chain(NormL1(big(1.0)), [A, A, A])
    @test depth(s3) == 3
    BEYOND && @test 3 * Int128(A) > typemax(Int64)
    @test prox(s3, [big(2.0), big(-0.5)], big(1.0)) == [0, 0]
    @test prox(s3, [big(2.0), big(-0.5)], pw(-A)) == [0, 0]
    t3 = chain(NormL1(big(1.0)), [-A, -A, -A])
    @test prox(t3, [big(2.0), big(-0.5)], big(1.0)) == [2.0, -0.5]
  end

  # the retained exponent 3P leaves Int64 and returns into range through the level and the weight 2^-Q: exceeds, then cancels
  with_precisions("exceeds, then cancels") do
    a = pw(P)
    s = Scaled(Scaled(Scaled(CheckLoss(pw(-Q), [pw(-Q)]), a), a), a)
    @test depth(s) == 3
    y = prox(Conjugate(s), [pw(Y)], one(BigFloat))
    @test y == [pw(3 * Int128(P) - 2 * Int128(Q))]
    # a point inside the box is left alone; the lower end of the box, -2^(3P - Q), is beyond the range
    @test prox(Conjugate(s), [pw(3 * Int128(P) - 2 * Int128(Q) - 1)], one(BigFloat)) == [pw(3 * Int128(P) - 2 * Int128(Q) - 1)]
    @test prox(Conjugate(s), [-pw(Y)], one(BigFloat)) == [-pw(Y)]
  end

  # the exponent -3P is below Int64 and does not return into range: exceeds and stays, the box is zero
  with_precisions("exceeds and stays") do
    a = pw(-P)
    s = Scaled(Scaled(Scaled(CheckLoss(big(0.5), [pw(Q)]), a), a), a)
    @test depth(s) == 3
    @test prox(Conjugate(s), [big(1.0)], one(BigFloat)) == [big(0.0)]
    @test prox(Conjugate(s), [-big(1.0)], one(BigFloat)) == [big(0.0)]
  end
end

@testset "a retained exponent read on its own" begin
  with_precisions("value and gradient at a normal argument") do
    for F in CHAINS
      N = exponent_sum(F)
      s = chain(NormL1(big(1.0)), F)
      expect(s([big(1.0)]), N; what = ("NormL1 value", F))
      smooth = chain(SqrNormL2(big(1.0)), F)
      expect(smooth([big(1.0)]), N - 1; what = ("SqrNormL2 value", F))
      g = [big(0.0)]
      v = value_and_gradient!(g, smooth, [big(1.0)])
      expect(v, N - 1; what = ("SqrNormL2 value_and_gradient!", F))
      expect(g[1], N; what = ("SqrNormL2 gradient", F))
    end
  end

  with_precisions("prox with a small step") do
    for F in CHAINS
      S = exponent_sum(F)
      s = chain(NormL1(big(1.0)), F)
      x = [big(2.0), big(-0.5)]
      # the threshold 2^S at the step 1 and at a step that does not bring it into range
      @test prox(s, x, big(1.0)) == (S > 0 ? [0, 0] : x)
      @test prox(s, x, pw(-1)) == (S > 0 ? [0, 0] : x)
      y = zeros(BigFloat, 2)
      prox!(y, s, x, pw(-A))
      @test y == (S - A > 0 ? [0, 0] : x)
    end
  end

  with_precisions("the factors of a chain that cancel") do
    # the constructor merges a factor with the last one when their product is a normal number: the exponent of the
    # merged chain never leaves Int64
    for F in ([A, A, A, -A, -A, -A], [A, A, A, A, -A, -A, -A], [-A, -A, -A, A, A], [P, P, P, -P, -P], [A, -A, A, -A, A])
      s = chain(NormL1(big(1.0)), F)
      @test depth(s) < length(F) || length(F) <= 2
      a = retained(s)
      @test typemin(Int64) <= a.e <= typemax(Int64)
      @test Int128(a.e) - 1 == exponent_sum(F)
      expect(s([big(1.0)]), exponent_sum(F); what = ("cancelling chain", F))
    end
  end

  with_precisions("exactly one and two factors") do
    for F in ([A], [-A], [A, A], [-A, -A], [A, -A], [0], [1, 1], [P, P], [-P, -P])
      s = chain(NormL1(big(1.0)), F)
      S = exponent_sum(F)
      expect(s([big(1.0)]), S; what = ("short chain", F))
      x = [big(2.0), big(-0.5)]
      # the threshold is 2^S: one at S = 0, so that 2 gives 1 and -0.5 gives 0
      expected = S >= 2 ? [big(0.0), big(0.0)] : S <= -300 ? x : [big(1.0), big(0.0)]
      @test prox(s, x, big(1.0)) == expected
    end
  end
end

# The retained exponent leaves Int64 and returns into range through the level and the step (the scale folds into the
# levels, and the step is part of the fold), through the level and the weight in the projections (they form the
# product of the scale, the level and the weight), or through the weights of an L1 norm, which absorb the scale when
# the level cannot (the testset "the weights absorb the exponent that the level cannot take"). The cases below that
# stay beyond the range assert the class of the result only.
@testset "a retained exponent that returns into range" begin
  with_precisions("prox through the level and the step") do
    found = 0
    for F in LONG_CHAINS
      S = exponent_sum(F)
      for t in targets(S), (name, l, w, g) in splits(t - S)
        w == 0 || continue
        found += 1
        check_prox(F, l, w, g, "F = $F, T = $t, split between $name: l = $l, g = $g")
      end
    end
    @test found > 0
  end

  with_precisions("prox with the step alone, for two factors") do
    # two factors leave the range but not Int64; the step at the end of its range brings them back
    check_prox([A, A], 0, 0, EMIN - 1, "two factors 2^A and the step 2^(EMIN - 1)")
    check_prox([-A, -A], 0, 0, EMAX - 1, "two factors 2^-A and the step 2^(EMAX - 1)")
  end

  with_precisions("prox beyond the range, with a level, a weight and a step") do
    for F in CHAINS, (l, w, g) in ((0, 0, 0), (frac(100), 0, 0), (0, frac(-100), 0), (0, 0, frac(100)), (frac(-100), frac(100), frac(-100)))
      S = exponent_sum(F)
      T = S + l + w + g
      (T >= EMAX || T <= EMIN - 3) && check_prox(F, l, w, g, "F = $F, l = $l, w = $w, g = $g")
    end
  end

  with_precisions("values and gradients that stay beyond the range") do
    for F in CHAINS, (l, w) in ((0, 0), (frac(100), 0), (0, frac(-100)), (frac(-100), frac(100)))
      S = exponent_sum(F)
      T = S + l + w
      (T >= EMAX || T <= EMIN - 3) || continue
      expect(chain(NormL1(pw(l), [pw(w)]), F)([big(1.0)]), T; what = ("value", F, l, w))
      w == 0 || continue
      smooth = chain(SqrNormL2(pw(l)), F)
      expect(smooth([big(1.0)]), T - 1; what = ("SqrNormL2 value", F, l))
      v = value_and_gradient!([big(0.0)], smooth, [big(1.0)])
      expect(v, T - 1; what = ("SqrNormL2 value_and_gradient!", F, l))
    end
  end
end

const PW = frac(672)                 # the issue's P and L, as fractions of EMAX: 3 PW leaves Int64, 3 PW - 2 LW is in range
const LW = frac(650)
const XW = frac(737)                 # an entry above the threshold 2^(3 PW - 2 LW), and one below it
const XL = frac(650)

# A weighted L1 norm under a chain whose product with the level is out of range, but whose product with the level and the
# weight is in range: the weights absorb the exponent that the level cannot take.
@testset "the weights absorb the exponent that the level cannot take" begin
  T = 3 * PW - 2 * LW
  @assert XL < T < XW
  BEYOND && @test 3 * Int128(PW) > typemax(Int64)

  with_precisions("the second case of the issue") do
    s = chain(NormL1(pw(-LW), [pw(-LW), pw(-LW)]), [PW, PW, PW])
    @test depth(s) == 3
    thr = pw(T)
    x = [pw(XW), pw(XL)]
    expected = [pw(XW) - thr, big(0.0)]
    @test prox(s, x, big(1.0)) == expected
    @test prox(s, -x, big(1.0)) == -expected
    y = zeros(BigFloat, 2)
    prox!(y, s, x, big(1.0))
    @test y == expected
    # the converse: a threshold above the first entry
    @test prox(s, [pw(T - 1), pw(XL)], big(1.0)) == [0, 0]
    @test prox(s, [pw(T), pw(XL)], big(1.0)) == [0, 0]
    # one entry at a time
    @test prox_entry(s, x[1], 1, big(1.0)) == expected[1]
    @test iszero(prox_entry(s, x[2], 2, big(1.0)))
    # the value of the chain: 2 entries of weight 2^-LW at 1, under the scale 2^(3 PW) and the level 2^-LW
    expect(s([big(1.0), big(1.0)]), T + 1; what = "value")
    # the value at the output: y_1 = x_1 (the threshold is far below it) and the weight is 2^-LW, so the value is
    # 2^(3 PW - 2 LW + XW), which is beyond the range
    y = zeros(BigFloat, 2)
    v = value_and_prox!(y, s, x, big(1.0))
    @test y == expected
    expect(v, T + XW; what = "value_and_prox!")
    # NormL1Nonneg: the same threshold on a positive entry, and everything else at zero
    sn = chain(NormL1Nonneg(pw(-LW), [pw(-LW), pw(-LW)]), [PW, PW, PW])
    @test prox(sn, x, big(1.0)) == expected
    @test prox(sn, -x, big(1.0)) == [0, 0]
    # a negative scale exponent: the chain 2^-PW three times, the level and the weight 2^LW bring it back
    t = chain(NormL1(pw(LW), [pw(LW), pw(LW)]), [-PW, -PW, -PW])
    @test prox(t, [pw(-T + 1), pw(-T - 5)], big(1.0)) == [pw(-T), 0]
  end

  with_precisions("a value that is in range") do
    P3, L3 = frac(680), frac(800)
    T3 = 3 * P3 - 2 * L3
    s = chain(NormL1(pw(-L3), [pw(-L3), pw(-L3)]), [P3, P3, P3])
    x = [pw(T3 + 1), pw(T3 - 4)]
    @test prox(s, x, big(1.0)) == [pw(T3), 0]
    expect(s([big(1.0), big(1.0)]), T3 + 1; what = "value")
    y = zeros(BigFloat, 2)
    v = value_and_prox!(y, s, x, big(1.0))
    @test y == [pw(T3), 0]
    # the weight 2^-L3 and the level 2^-L3 are applied to |y_1| = 2^T3 under the scale 2^(3 P3)
    expect(v, 2 * T3; what = "value_and_prox!")
    @test v == pw(2 * T3)
  end

  with_precisions("weights of different exponents, a zero weight and an infinite weight") do
    d = frac(50)
    s = chain(NormL1(pw(-LW), [pw(-LW), pw(-LW + d)]), [PW, PW, PW])
    # two thresholds, 2^T and 2^(T + d)
    @test prox(s, [pw(T + 1), pw(T + d + 1)], big(1.0)) == [pw(T), pw(T + d)]
    @test prox(s, [pw(T + d), pw(T + d)], big(1.0)) == [pw(T + d) - pw(T), 0]
    @test s([big(1.0), big(1.0)]) == pw(T) + pw(T + d)
    # a zero weight leaves its entry alone, an infinite one fixes it at zero
    z = chain(NormL1(pw(-LW), [pw(-LW), big(0.0), big(Inf)]), [PW, PW, PW])
    @test prox(z, [pw(T + 1), big(3.0), pw(T + 5)], big(1.0)) == [pw(T), 3, 0]
    @test prox_entry(z, big(3.0), 2, big(1.0)) == 3
    @test iszero(prox_entry(z, pw(T + 5), 3, big(1.0)))
  end

  with_precisions("weights spread over most of the range, and a spread that cannot fold") do
    # the scale 2^(-2 EMAX) and the level 2^(0.95 EMAX) have a product outside the range, so the level takes nothing; the
    # weights 2^(0.1 EMAX) and 2^(0.99 EMAX) have thresholds 2^(-0.95 EMAX) and 2^(-0.06 EMAX) that are normal numbers,
    # and the centred shift folds them
    F = [-frac(400), -frac(400), -frac(400), -frac(400), -frac(400)]
    sp = chain(NormL1(pw(frac(950)), [pw(frac(100)), pw(frac(990))]), F)
    N1, N2 = exponent_sum(F) + frac(950) + frac(100), exponent_sum(F) + frac(950) + frac(990)
    @test EMIN + 3 < N1 && N2 < 0     # both thresholds are normal numbers
    @test prox(sp, [pw(N1 + 1), pw(N2 + 1)], big(1.0)) == [pw(N1), pw(N2)]
    @test prox(sp, [pw(N2 + 1), pw(N2 + 1)], big(1.0)) == [pw(N2 + 1) - pw(N1), pw(N2)]
    # The weights 2^(EMAX - 1) and 2^(EMIN - 1) are the largest and the smallest normal weights. Under the scale (its
    # exponent α.e is about 3 PW) and the level 2^-LW their coefficients have the exponents tmax = α.e + 1 - LW + EMAX and
    # tmin = α.e + 1 - LW + EMIN. Here tmax is about 2.37 EMAX, beyond 2 EMAX - 6: no product of a normal level and a normal
    # weight reaches it (for the larger weight the level alone leaves the range, and the scale does not help, the product
    # being beyond the range itself); tmin is about 0.37 EMAX, a normal number, so its condition holds (the product of the
    # smaller weight with the level alone, of exponent EMIN - LW, would underflow; the scale brings it back). The two
    # differ by EMAX - EMIN, more than the room EMAX - EMIN - 6. The fold is refused for the first and the third
    # condition and the weights are left as they are, as without the weight fold; the call must not throw and gives numbers.
    wide = chain(NormL1(pw(-LW), [pw(EMAX - 1), pw(EMIN - 1)]), [PW, PW, PW])
    α = retained(wide)
    tmin, tmax = α.e + 1 - LW + EMIN, α.e + 1 - LW + EMAX
    @test tmax > 2 * EMAX - 6
    @test tmax - tmin > EMAX - EMIN - 6
    @test tmin >= 2 * EMIN + 6
    @test PB._fold_weights(NormL1(pw(-LW), [pw(EMAX - 1), pw(EMIN - 1)]), α) === nothing
    y = prox(wide, [pw(T + 1), pw(T + 1)], big(1.0))
    @test length(y) == 2 && !any(isnan, y)
  end

  # (the L1 projection returns zero at two bits already for ordinary numbers, with or without this change, so this
  # testset runs at the two wider precisions)
  with_precisions("the L1 ball of a conjugate, the radius being the retained product"; precisions = (53, 256)) do
    R = 3 * PW - 2 * LW
    # the radius is 2^R: the part of the first entry above it is cut
    G = chain(GroupNormLinf(pw(-LW), [1:2], [pw(-LW)]), [PW, PW, PW])
    @test prox(Conjugate(G), [pw(R + 1), big(0.0)], big(1.0)) == [pw(R), 0]
    @test prox(Conjugate(G), [pw(R - 3), big(0.0)], big(1.0)) == [pw(R - 3), 0]
    # the radius 2^(3 PW - LW) is beyond the range: the point is inside
    N = chain(NormLinf(pw(-LW)), [PW, PW, PW])
    @test prox(Conjugate(N), [pw(R + 1), big(0.0)], big(1.0)) == [pw(R + 1), 0]
  end
end

# One weight takes all of the exponent that the level cannot take: the coefficient is a normal number, but the weight
# that a fold keeping the significand of the level would form is not. The scale is three factors 2^(±PB) with the level
# 2^(∓QB), and the weight has the exponent that gives the coefficient 2^N. The coefficients at both ends of the range
# are asserted, with the oracle classifying by the exact exponent first.
const PB3 = frac(672)
const QB3 = frac(650)
function one_weight_chain(sign, N, W = x -> [x]; nonneg = false)
  F = sign > 0 ? [PB3, PB3, PB3] : [-PB3, -PB3, -PB3]
  l = -sign * QB3
  w = N - exponent_sum(F) - l
  @assert EMIN - 1 <= w <= EMAX - 1
  leaf = nonneg ? NormL1Nonneg(pw(l), W(pw(w))) : NormL1(pw(l), W(pw(w)))
  chain(leaf, F)
end

@testset "a weight that takes all of the exponent" begin
  cases = vcat([(1, N) for N in (EMAX - 1, EMAX, EMAX + 1, frac(900), frac(400))],
               [(-1, N) for N in (EMIN - 1, EMIN - 2, EMIN - 3, EMIN - 4, -frac(900), -frac(400))])
  for nonneg in (false, true)
    with_precisions("coefficients across the range, $(nonneg ? "NormL1Nonneg" : "NormL1")") do
      for (sign, N) in cases
        s = one_weight_chain(sign, N; nonneg)
        # the point and the threshold: above a threshold in range, and anywhere under an infinite threshold
        x, expected = if N >= EMAX
          floatmax(BigFloat), big(0.0)
        elseif N == EMAX - 1
          floatmax(BigFloat), floatmax(BigFloat) - pw(EMAX - 1)
        elseif N == EMIN - 1
          pw(N + 1), pw(N)
        elseif N <= EMIN - 2
          big(1.0), big(1.0)           # the threshold is zero or too small to move 1
        else
          pw(N + 1), pw(N)
        end
        @testset "sign $sign, coefficient 2^$N" begin
          @test prox(s, [x], big(1.0)) == [expected]
          @test prox_entry(s, x, 1, big(1.0)) == expected
          y = zeros(BigFloat, 1)
          prox!(y, s, [x], big(1.0))
          @test y == [expected]
          nonneg || @test prox(s, [-x], big(1.0)) == [-expected]
          N == EMIN - 2 || expect(s([big(1.0)]), N; what = ("value", sign, N, nonneg))
        end
      end
    end
  end

  # the weight 2^(E - 1 - 3P + Q) under the scale 2^(3P) and the level 2^-Q has the coefficient 2^(E - 1); a fold that
  # kept the level 2^-Q would need the weight 2^(E - 1 + Q), which overflows, so the level takes the exponent
  with_precisions("the review example") do
    if EMAX == 2^62 - 1
      P, Q, E = Int128(3_100_000_000_000_000_000), Int128(3_000_000_000_000_000_000), EMAX
      f = NormL1(pw(-Q), [pw(E - 1 - 3P + Q)])
      s = foldl((g, _) -> Scaled(g, pw(P)), 1:3; init = f)
      x = [floatmax(BigFloat)]
      @test prox(s, x, big(1.0)) == x .- pw(E - 1)
      @test prox_entry(s, x[1], 1, big(1.0)) == x[1] - pw(E - 1)
      @test s([pw(1 - E)]) == 1
    end
  end

  with_precisions("weights that are matrices") do
    # a 1 x 2 weight paired with the entries of a vector by the linear index
    T = 3 * PW - 2 * LW
    for nonneg in (false, true)
      F = [PW, PW, PW]
      W = reshape([pw(-LW), pw(-LW)], 1, 2)
      s = chain(nonneg ? NormL1Nonneg(pw(-LW), W) : NormL1(pw(-LW), W), F)
      x = [pw(T + 1), pw(T - 2)]
      @test prox(s, x, big(1.0)) == [pw(T), 0]
      @test prox_entry(s, x[1], 1, big(1.0)) == pw(T)
      @test iszero(prox_entry(s, x[2], 2, big(1.0)))
      y = zeros(BigFloat, 2)
      v = value_and_prox!(y, s, x, big(1.0))
      @test y == [pw(T), 0]
      expect(v, 2 * T; what = "value_and_prox! of a matrix weight")
      expect(s([pw(-T), big(0.0)]), 0; what = "value of a matrix weight")
    end
    # a 2 x 2 weight whose entries have different exponents, so that the thresholds differ per entry
    d = frac(20)
    W = [pw(-LW) pw(-LW + 2d); pw(-LW + d) pw(-LW + 3d)]
    s = chain(NormL1(pw(-LW), W), [PW, PW, PW])
    X = [pw(T + 1) pw(T + 2d + 1); pw(T + d + 1) pw(T + 3d + 1)]
    @test prox(s, X, big(1.0)) == [pw(T) pw(T + 2d); pw(T + d) pw(T + 3d)]
    @test prox_entry(s, X[2], 2, big(1.0)) == pw(T + d)   # the linear index runs down the columns
    # the largest term 2^(T + 3d); the three others are far below one unit in the last place
    @test s(fill(big(1.0), 2, 2)) == pw(T + 3d)
  end
end

@testset "Conjugate" begin
  with_precisions("projection onto the box of a chain, through the level and the weight") do
    found = 0
    for F in LONG_CHAINS
      S = exponent_sum(F)
      for t in targets(S), (name, l, w, g) in splits(t - S)
        g == 0 || continue
        found += 1
        T = S + l + w
        s = chain(NormL1(pw(l), [pw(w)]), F)
        c = Conjugate(s)
        @testset "F = $F, l = $l, w = $w" begin
          @test prox(c, [pw(T + 1)], big(1.0)) == [pw(T)]
          @test prox(c, [-pw(T + 1)], big(1.0)) == [-pw(T)]
          @test prox(c, [pw(T - 1)], big(1.0)) == [pw(T - 1)]
          z = zeros(BigFloat, 1)
          prox!(z, c, [pw(T + 1)], big(1.0))
          @test z == [pw(T)]
        end
      end
    end
    @test found > 0
  end

  with_precisions("projection and value of the conjugate of a chain whose product stays beyond the range") do
    for F in CHAINS
      S = exponent_sum(F)
      c = Conjugate(chain(NormL1(big(1.0), [big(1.0)]), F))
      # the value of the conjugate is not defined for a product beyond the range of a level that no fold takes: the
      # package says so, and which way the product left the range is read from the exponent alone
      value_error(y) = try c(y); nothing catch e; e end
      if S > 0
        y = [pw(EMAX - 1)]
        @test prox(c, y, big(1.0)) == y
        err = value_error(y)
        @test err isa ArgumentError && occursin("overflows", err.msg)
      else
        @test prox(c, [big(1.0)], big(1.0)) == [big(0.0)]
        err = value_error([big(1.0)])
        @test err isa ArgumentError && occursin("underflows", err.msg)
      end
    end
  end

  with_precisions("the conjugate of CheckLoss under a chain, the box and its sides") do
    a = pw(P)
    s = Scaled(Scaled(Scaled(CheckLoss(pw(-Q), [pw(-Q)]), a), a), a)
    c = Conjugate(s)
    U = 3 * Int128(P) - 2 * Int128(Q)      # the upper end of the box is 2^U; the lower end is -2^(3P - Q) at the weight 2^-Q
    z = zeros(BigFloat, 1)
    prox!(z, c, [pw(U + 3)], big(1.0))
    @test z == [pw(U)]
    prox!(z, c, [pw(U - 3)], big(1.0))
    @test z == [pw(U - 3)]
    # the converse: a box that is below the range
    b = pw(-P)
    t = Scaled(Scaled(Scaled(CheckLoss(big(0.5), [pw(Q)]), b), b), b)
    prox!(z, Conjugate(t), [big(1.0)], big(1.0))
    @test iszero(z[1])
  end
end

@testset "wrappers of the chain" begin
  with_precisions("Translate, Scaled and SeparableSum") do
    s3 = chain(NormL1(big(1.0)), [A, A, A])
    n3 = chain(NormL1(big(1.0)), [-A, -A, -A])
    b = [big(1.0)]
    x = [big(3.0)]
    # a translated chain: the point moves to the translation where the threshold is infinite, and stays where it is zero
    @test prox(Translate(s3, b), x, big(1.0)) == [big(1.0)]
    @test prox(Translate(n3, b), x, big(1.0)) == [big(3.0)]
    # an unnormalized wrapper (a factor of two) and a normalized one (a factor of one) around the translation
    @test prox(Scaled(Translate(s3, b), big(2.0)), x, big(1.0)) == [big(1.0)]
    @test prox(Scaled(Translate(n3, b), big(2.0)), x, big(1.0)) == [big(3.0)]
    @test prox(Scaled(Translate(s3, b), big(1.0)), x, big(1.0)) == [big(1.0)]
    # a separable sum of two scaled blocks
    f = SeparableSum(s3, n3)
    xs = ([big(2.0), big(-0.5)], [big(2.0)])
    ys = map(similar, xs)
    prox!(ys, f, xs, big(1.0))
    @test ys[1] == [0, 0]
    @test ys[2] == [2.0]
    # the value and the proximal point together
    y = zeros(BigFloat, 2)
    v = value_and_prox!(y, s3, [big(2.0), big(-0.5)], big(1.0))
    @test y == [0, 0]
    @test iszero(v)
    y = zeros(BigFloat, 2)
    v = value_and_prox!(y, n3, [big(2.0), big(-0.5)], big(1.0))
    @test y == [2.0, -0.5]
    @test iszero(v)
    # the value of a translated chain at the translation
    @test iszero(Translate(s3, b)(b))
  end

  with_precisions("a chain below an ordinary scale") do
    s3 = chain(NormL1(big(1.0)), [A, A, A])
    h = Scaled(Translate(s3, [big(0.0)]), pw(1))
    @test prox(h, [big(2.0)], big(1.0)) == [0]
  end
end

@testset "special values" begin
  with_precisions("zero, signed zero, NaN and Inf") do
    s3 = chain(NormL1(big(1.0)), [A, A, A])
    n3 = chain(NormL1(big(1.0)), [-A, -A, -A])
    for s in (s3, n3)
      @test iszero(s([big(0.0)]))
      @test iszero(s([-big(0.0)]))
      @test isnan(s([big(NaN)]))
      @test s([big(Inf)]) == Inf
    end
    @test isnan(prox(n3, [big(NaN)], big(1.0))[1])
    @test prox(n3, [big(Inf)], big(1.0)) == [Inf]
    @test iszero(prox(s3, [-big(0.0)], big(1.0))[1])
    # a zero factor at the end of the chain
    z = Scaled(chain(NormL1(big(1.0)), [A, A]), big(0.0))
    @test iszero(z([big(1.0)]))
    @test prox(z, [big(2.0), big(-0.5)], big(1.0)) == [2.0, -0.5]
    z = Scaled(chain(NormL1(big(1.0)), [A, A, A]), big(0.0))
    @test iszero(z([big(1.0)]))
    @test prox(z, [big(2.0), big(-0.5)], big(1.0)) == [2.0, -0.5]
  end
end

@testset "the helpers of the retained scale" begin
  BigScale = PB._ProjectionScale{BigFloat, Int128}
  E1 = Int128(typemax(Int64)) + 1000        # beyond Int64
  E2 = -E1 - 12345
  with_precisions("construction") do
    a = PB._ProjectionScale(big(0.75), E1)
    @test a isa BigScale
    @test a.e === E1
    @test PB._ProjectionScale(big(0.75), 3).e === Int128(3)
    @test a.m == 0.75
    @test !isone(a) && !iszero(a) && isfinite(a)
  end

  with_precisions("_projection_accumulate") do
    # a scale and a real
    a = PB._ProjectionScale(big(0.75), E1)
    for n in (10, -10, 0, Int(frac(300)), -Int(frac(300)))
      r = PB._projection_accumulate(a, pw(n))
      @test r isa BigScale
      @test r.m == 0.75
      @test r.e === E1 + n
      r = PB._projection_accumulate(PB._ProjectionScale(big(0.75), E2), pw(n))
      @test r isa BigScale && r.m == 0.75 && r.e === E2 + n
    end
    # a scale and a real that is not a BigFloat: the exponent type follows the significand
    r = PB._projection_accumulate(a, 8.0)
    @test r isa BigScale && r.e == E1 + 3
    # a scale and a non-finite or a zero factor is that factor
    @test PB._projection_accumulate(a, big(0.0)) == 0
    @test PB._projection_accumulate(a, big(Inf)) == Inf
    # two reals
    r = PB._projection_accumulate(pw(A), pw(A))
    @test r isa BigScale && r.m == 0.5 && r.e === Int128(2A + 1)
    r = PB._projection_accumulate(pw(-A), pw(-A))
    @test r isa BigScale && r.m == 0.5 && r.e === Int128(-2A + 1)
    # the products that are normal numbers stay numbers
    @test PB._projection_accumulate(pw(5), pw(7)) == pw(12)
    # six factors: the exponent of the sum
    r = PB._scale_start(pw(A))
    for _ in 2:6
      r = PB._projection_accumulate(r, pw(A))
    end
    @test r isa BigScale && r.e === Int128(6) * A + 1
  end

  with_precisions("_scale_inverse and _scale_times") do
    a = PB._ProjectionScale(big(0.5), E1)
    r = PB._scale_inverse(a)
    @test r isa BigScale && r.m == 0.5 && r.e === 2 - E1
    r = PB._scale_inverse(PB._ProjectionScale(big(0.5), E2))
    @test r isa BigScale && r.m == 0.5 && r.e === 2 - E2
    r = PB._scale_inverse(PB._scale_inverse(a))
    @test r.m == 0.5 && r.e === E1
    # the product of two scales and of a real and a scale
    b = PB._ProjectionScale(big(0.5), E2 + 5)
    r = PB._scale_times(a, b)
    @test r isa BigScale && r.m == 0.5 && r.e === E1 + (E2 + 5) - 1
    r = PB._scale_times(a, a)
    @test r isa BigScale && r.m == 0.5 && r.e === 2 * E1 - 1
    r = PB._scale_times(pw(10), a)
    @test r isa BigScale && r.m == 0.5 && r.e === E1 + 10
    r = PB._scale_times(big(4.0), PB._ProjectionScale(big(0.5), E2))
    @test r isa BigScale && r.m == 0.5 && r.e === E2 + 2
  end

  with_precisions("_apply_scale, _scale_below_one and _scale_above_one") do
    a, b = PB._ProjectionScale(big(0.5), E1), PB._ProjectionScale(big(0.5), E2)
    @test PB._apply_scale(a, big(1.0)) == Inf
    @test iszero(PB._apply_scale(b, big(1.0)))
    @test PB._apply_scale(a, pw(-5)) == Inf
    @test iszero(PB._apply_scale(b, pw(5)))
    @test isnan(PB._apply_scale(a, big(NaN)))
    @test iszero(PB._apply_scale(a, big(0.0)))
    @test PB._apply_scale(a, -big(1.0)) == -Inf
    @test !PB._scale_below_one(a) && PB._scale_above_one(a)
    @test PB._scale_below_one(b) && !PB._scale_above_one(b)
    # exponents that an Int would read with the wrong sign
    for e in (Int128(2)^63, Int128(2)^63 + 5, Int128(2)^64 + 3, Int128(2)^64 + 2^62)
      u = PB._ProjectionScale(big(0.5), e)
      @test PB._scale_above_one(u) && !PB._scale_below_one(u)
      v = PB._ProjectionScale(big(0.5), -e)
      @test PB._scale_below_one(v) && !PB._scale_above_one(v)
    end
    @test PB._scale_below_one(PB._ProjectionScale(big(0.5), 0))
    @test !PB._scale_above_one(PB._ProjectionScale(big(0.5), 1))
    @test PB._scale_above_one(PB._ProjectionScale(big(0.75), 1))
  end

  with_precisions("_projection_product and _apply_parts") do
    a = PB._ProjectionScale(big(0.5), E1)
    # beyond Int64 and nothing to cancel it
    @test PB._projection_product(a, big(1.0), big(1.0)) == Inf
    @test iszero(PB._projection_product(PB._ProjectionScale(big(0.5), E2), big(1.0), big(1.0)))
    # beyond Int64, cancelled by the two factors: 2^(E + n1 + n2 - 1)
    E = 2 * EMAX + 100
    c = PB._ProjectionScale(big(0.5), E)
    n = EMAX - 10
    @test PB._projection_product(c, pw(-n), pw(-n)) == pw(E - 2n - 1)
    @test PB._projection_product(c, -pw(-n), pw(-n)) == -pw(E - 2n - 1)
    @test PB._projection_product(PB._ProjectionScale(big(0.5), -E), pw(n), pw(n)) == pw(-E + 2n - 1)
    # the parts of a product: α m 2^e
    for (ea, e) in ((E1, 7), (E2, -7), (E, Int(-2n)), (-E, Int(2n)))
      r = PB._apply_parts(PB._ProjectionScale(big(0.5), ea), big(0.5), e)
      expect(r, ea + e - 2; what = ("_apply_parts", ea, e))
    end
    # a real in place of the scale
    @test PB._apply_parts(big(1.0), big(0.5), 3) == 4
    # _scaled_product with a power beyond Int64
    @test PB._scaled_product(pw(5), pw(7), E1) == Inf
    @test iszero(PB._scaled_product(pw(5), pw(7), E2))
    @test PB._scaled_product(pw(5), pw(7), pw(1), E1) == Inf
    @test iszero(PB._scaled_product(pw(5), pw(7), pw(1), E2))
    @test PB._scaled_product(pw(-n), pw(-n), E) == pw(E - 2n)
    @test PB._scaled_product(pw(-n), pw(-n), pw(0), E) == pw(E - 2n)
  end

  with_precisions("_projection_scaled does not loop more than twice the number of factors") do
    count = 0
    # an exponent of six factors of the largest magnitude
    e = Int128(6) * (EMAX + 1)
    for α in (PB._ProjectionScale(big(0.5), e), PB._ProjectionScale(big(0.5), -e))
      g = PB._projection_scaled(NormL1(big(1.0)), α)
      count = depth(g)
      @test count <= 2 * 6 + 1
      @test g isa Scaled
    end
    @test count >= 1
  end

  with_precisions("the folds of a level") do
    a = PB._ProjectionScale(big(0.5), E1)
    @test PB._folded_level(big(1.0), a) === nothing
    b = PB._ProjectionScale(big(0.5), E2)
    @test PB._folded_level(big(1.0), b) === nothing
    c = PB._ProjectionScale(big(0.5), E1)
    @test PB._NormalScale(c).s === c
    r = PB._projection_accumulate(PB._NormalScale(c), pw(10))
    @test r isa PB._NormalScale && r.s.e === E1 + 10
  end
end

@testset "the spectral path" begin
  with_precisions("_spectrum_scaled and the bound of a conjugate ball") do
    BigScale = PB._ProjectionScale{BigFloat, Int128}
    s = [big(1.0), big(1.0)]
    unit = PB._unit_scale(BigFloat)
    @test unit isa BigScale
    # the value 2^S k-fold: b f(2^k s) = 2^(S + k) f(s) = 2^(S + k + 1) for f = NormL1(1)
    big_chain = chain(NormL1(big(1.0)), [A, A, A])
    r = PB._spectrum_scaled(big_chain, s, 3, unit)
    @test r == Inf
    for E in (Int128(typemax(Int64)) + 1000, Int128(2) * EMAX + 100)
      b = PB._ProjectionScale(big(0.5), E)
      @test PB._spectrum_scaled(NormL1(big(1.0)), s, 3, b) == Inf
    end
    # the value is the exact power where the exponent is in range
    b = PB._ProjectionScale(big(0.5), Int128(10))
    expect(PB._spectrum_scaled(NormL1(big(1.0)), s, 3, b), 10 + 3 - 1 + 1; what = "in range")
    # the conjugate of a chain of the norm: the indicator of the ball of radius 2^T
    for (F, expected_inf) in (([A, A, A], false), ([-A, -A, -A], true), ([P, P, P], false), ([-P, -P, -P], true))
      f = chain(NormL2(big(1.0)), F)
      r = PB._spectrum_scaled(Conjugate(f), s, 3, unit)
      @test expected_inf ? isinf(r) : iszero(r)
    end
    # a radius beyond every range is a ball that contains the spectrum
    c = Conjugate(chain(NormL2(big(1.0)), [A, A, A]))
    @test iszero(PB._spectrum_scaled(c, [big(1.0), big(2.0)], 5, unit))
  end
end

@testset "the metric path" begin
  with_precisions("array steps through a chain") do
    F = [P, P, P]
    S = exponent_sum(F)
    l, g = -frac(500), -frac(800)
    T = S + l + g
    s = chain(NormL1(pw(l)), F)
    # an array step: the threshold at the entry i is 2^T
    x = [pw(T + 1), -pw(T + 1)]
    γ = [pw(g), pw(g)]
    @test prox(s, x, γ) == [pw(T), -pw(T)]
    @test prox(s, [pw(T - 1), pw(T - 1)], γ) == [0, 0]
    y = similar(x)
    v = value_and_prox!(y, s, x, γ)
    @test y == [pw(T), -pw(T)]
    expect(v, S + l + T + 1; what = "value at the stored point")
    # the shifted fold: the product of the folded level and the array of the exponents is the threshold
    a = PB._scaled_chain(s)[2]
    @test a isa PB._ProjectionScale{BigFloat, Int128}
    BEYOND && @test a.e > typemax(Int64)
    r = PB._shifted_fold(NormL1(pw(l)), a, γ)
    @test r !== nothing
    folded, scaled = r
    @test folded.λ0 * scaled[1] == pw(T)
    @test folded.λ0 * scaled[2] == pw(T)
    # an exponent of two ranges, which the shift brings back
    r = PB._shifted_fold(NormL1(pw(l)), PB._ProjectionScale(big(0.5), Int128(2) * EMAX + 100), γ)
    @test r !== nothing
    r === nothing || @test r[1].λ0 * r[2][1] == pw(2 * EMAX + 99 + l + g)
  end
end

@testset "Float32 and Float64 keep Int exponents" begin
  @test typeof(PB._ProjectionScale(0.75, 3)) === PB._ProjectionScale{Float64, Int}
  @test typeof(PB._ProjectionScale(0.75f0, 3)) === PB._ProjectionScale{Float32, Int}
  @test typeof(PB._ProjectionScale(big(0.75), 3)) === PB._ProjectionScale{BigFloat, Int128}
  @test PB._scale_exponent_type(Float64) === Int
  @test PB._scale_exponent_type(Float32) === Int
  @test PB._scale_exponent_type(Float16) === Int
  @test PB._scale_exponent_type(BigFloat) === Int128
  for T in (Float64, Float32)
    a = PB._ProjectionScale(T(0.75), 3)
    @test a isa PB._ProjectionScale{T, Int}
    @test all(isconcretetype, fieldtypes(typeof(a)))
    normal = PB._NormalScale(a)
    probe = PB._LevelProbe(PB._ShiftWindow(typemin(Int), typemax(Int)), a, -1000, 1000)
    expscaled = PB._ExpScaledArray(T[1, 2], a)
    for x in (normal, probe, expscaled)
      @test all(isconcretetype, fieldtypes(typeof(x)))
    end
    @test isconcretetype(typeof(normal)) && isconcretetype(typeof(probe)) && isconcretetype(typeof(expscaled))
    @test typeof(normal.s) === PB._ProjectionScale{T, Int}

    # the results that were inferred are inferred, and the accumulation has concrete members
    b = T(2)^(T === Float64 ? 700 : 70)
    members(f, args...) = Base.uniontypes(only(Base.return_types(f, Tuple{map(typeof, args)...})))
    @test all(isconcretetype, members(PB._projection_accumulate, a, b))
    @test all(isconcretetype, members(PB._projection_accumulate, b, b))
    @test PB._projection_accumulate(a, b) isa PB._ProjectionScale{T, Int}
    @test PB._projection_accumulate(b, b) isa PB._ProjectionScale{T, Int}
    @test @inferred(PB._projection_product(a, T(2), T(5))) isa T
    @test @inferred(PB._apply_scale(a, T(3))) isa T
    @test @inferred(PB._scale_inverse(a)) isa PB._ProjectionScale{T, Int}
    @test @inferred(PB._scale_times(a, a)) isa PB._ProjectionScale{T, Int}
    @test @inferred(PB._scale_below_one(a)) isa Bool
    @test PB._projection_product(a, T(2), T(5)) == T(0.75) * 8 * 2 * 5
    @test PB._apply_scale(a, T(3)) == T(0.75) * 8 * 3

    # the weights are not folded into a Float scale: the level fold that fails stays refused
    @test PB._fold_level_step(NormL1(T(1e30), T[1, 2]), PB._ProjectionScale(T(0.5), T === Float64 ? 1000 : 100)) === nothing
    # the weighted chain of nested_scaled.jl ("chain, weights" in "the level and the factors have no product in range"):
    # a level 10^l2 under two factors 10^-fa, the step 10^g2, entries 10^(l2 - 2fa + g2) [3, -3, 0.5]; the reference is
    # the soft threshold at a λ w γ formed in BigFloat, as there
    fa, l2, g2 = T === Float64 ? (200, 80, 80) : (25, 10, 20)
    tens(k) = T(10)^k
    ff = Scaled(Scaled(NormL1(tens(l2), T[1, 2, 4]), tens(-fa)), tens(-fa))
    fx = T[3, -3, 0.5] .* T(10)^(l2 - 2fa + g2)
    soft(v, t) = sign(v) * max(abs(v) - t, 0)
    ref = [soft(big(fx[i]), big(tens(-fa)) * big(tens(-fa)) * big(tens(l2)) * big(T[1, 2, 4][i]) * big(tens(g2))) for i in 1:3]
    near(v, r) = isfinite(v) && (iszero(r) ? iszero(v) : abs(big(v) - r) <= 8 * eps(T) * abs(r))
    fy = zeros(T, 3)
    prox!(fy, ff, fx, tens(g2))
    @test all(i -> near(fy[i], ref[i]), 1:3)
    prox!(fy, ff, fx, fill(tens(g2), 3))
    @test all(i -> near(fy[i], ref[i]), 1:3)

    # no allocation after a warm-up call
    alloc_product(a, b, w) = @allocated PB._projection_product(a, b, w)
    alloc_apply(a, v) = @allocated PB._apply_scale(a, v)
    alloc_accumulate(a, v) = @allocated PB._projection_accumulate(a, v)
    alloc_product(a, T(2), T(5)); alloc_apply(a, T(3)); alloc_accumulate(a, b)
    @test alloc_product(a, T(2), T(5)) == 0
    @test alloc_apply(a, T(3)) == 0
    @test alloc_accumulate(a, b) == 0
  end
end


# The fold of a scale into the weights of an L1 norm, against an oracle written here. The oracle reads every operand as
# an exact dyadic number n 2^z (n a BigInt, z an Int128) from the value as stored, without the helpers of the package and
# without frexp at the precision of the call: a BigFloat by the exact integer and exponent that MPFR stores
# (`Base.decompose`), any other number by its exact rational value. A level or weight that is not a BigFloat is rounded to
# the precision p of the call, as the leaf does when it promotes it; a BigFloat is used as stored. The product of the three
# operands is exact in integers and is rounded to p bits by the routine below, to nearest (ties to even) or in a directed
# way. The positions are Int128 sums of the exponents. The scale, the level and the weights are positive and finite (a
# negative level or weight cannot be constructed); zero and infinite weights are not read.

# the odd integer and the exponent of the positive dyadic number n 2^z
function canon(n::BigInt, z::Integer)
  @assert n > 0
  k = trailing_zeros(n)
  (n >> k, Int128(z) + k)
end
bitlength(n::BigInt) = ndigits(n, base = 2)

# the exact value of a finite positive stored number as canon(n, z)
function exact(x)
  if x isa BigFloat
    n, z, _ = Base.decompose(x)
    canon(BigInt(n), z)
  else
    r = Rational{BigInt}(x)
    d = denominator(r)
    @assert count_ones(d) == 1
    canon(numerator(r), -(bitlength(d) - 1))
  end
end

# the number n 2^z rounded to p bits: to nearest with ties to even (:nearest), or up or down (:up, :down)
function round_dyadic(n::BigInt, z::Integer, p::Integer, mode::Symbol = :nearest)
  L = bitlength(n)
  if L > p
    sh = L - p
    q = n >> sh
    rest = n - (q << sh)
    half = BigInt(1) << (sh - 1)
    up = mode === :up ? rest > 0 : mode === :down ? false : (rest > half || (rest == half && isodd(q)))
    n, z = q + up, z + sh
  end
  canon(n, z)
end

# the position (the exponent e of x in [2^(e-1), 2^e)) of the canonical dyadic number
pos((n, z)) = z + bitlength(n)

# the dyadic number that the leaf uses at the precision p for a level or a weight x
model(x, p) = x isa BigFloat ? exact(x) : round_dyadic(exact(x)..., p)
# a BigFloat that p bits do not hold exactly (equal to or less precise than p, or a value of at most p bits, holds)
loses(x, p) = x isa BigFloat && isfinite(x) && !iszero(x) && bitlength(exact(x)[1]) > p && precision(x) > p
usable(x) = !(iszero(x) || !isfinite(x))

bfq(n, q) = setprecision(BigFloat, q) do
  BigFloat(n) / 7
end
bfh(n, q) = setprecision(BigFloat, q) do
  BigFloat(n) / 14
end
# a value stored at a given precision (0.75 or 3 are held by two bits)
bfw(x, q) = BigFloat(x; precision = q)

const FOLDED = Ref(0)
const REFUSED = Ref(0)
const REFUSED_BY_PRECISION = Ref(0)

# the failures of the fold of the scale mα 2^αe into f, as strings; empty when the fold agrees with the oracle
function fold_failures(f, αm::BigFloat, αe::Integer, mode::Symbol = :nearest)
  λ, w = f.λ0, f.w
  p = precision(BigFloat)
  out = String[]
  fail(msg) = push!(out, "$msg: p = $p, level = $λ, weights = $w, scale = $αm 2^$αe")
  got = PB._fold_weights(f, PB._ProjectionScale(αm, αe))
  got === nothing ? (REFUSED[] += 1) : (FOLDED[] += 1)
  if loses(λ, p) || loses(αm, p) || any(wi -> usable(wi) && loses(wi, p), w)
    REFUSED_BY_PRECISION[] += 1
    got === nothing || fail("a fold of an operand that p bits do not hold")
    return out
  end
  lm = model(λ, p)
  am = exact(αm)
  ts = Int128[Int128(αe) + pos(lm) + pos(model(wi, p)) for wi in w if usable(wi)]
  if isempty(ts)
    got === nothing || fail("a fold without weights to fold")
    return out
  end
  tmin, tmax = extrema(ts)
  empty = !(tmin >= 2 * EMIN + 6 && tmax <= 2 * EMAX - 6 && tmax - tmin <= EMAX - EMIN - 6)
  if empty
    got === nothing || fail("a fold that the interval of shifts refuses")
    return out
  end
  got === nothing && return fail("a refusal where a shift exists")
  typeof(got).name === typeof(f).name || return fail("the type of the result")
  level = got.λ0
  ln, s = exact(level)
  (ln == 1 && precision(level) == p) || fail("the level is not a power of two at the precision")
  (EMIN + 3 <= s <= EMAX - 3) || fail("the level is outside the margin: s = $s")
  size(got.w) == size(w) || return fail("the size of the weights")
  for (i, wi) in enumerate(w)
    nw = got.w[i]
    if !usable(wi)
      isequal(nw, convert(BigFloat, wi)) || fail("a zero or infinite weight changed")
      continue
    end
    wm = model(wi, p)
    t = Int128(αe) + pos(lm) + pos(wm)
    (isfinite(nw) && !iszero(nw) && precision(nw) == p) || fail("a weight is not a normal number of the precision")
    (EMIN + 3 <= t - s <= EMAX - 3) || fail("a weight is outside the margin: t - s = $(t - s)")
    cn, cz = round_dyadic(am[1] * lm[1] * wm[1], am[2] + lm[2] + wm[2] + αe, p, mode)
    on, oz = exact(nw)
    (on == cn && oz + s == cz) || fail("the weight $i is not the coefficient rounded once: got $nw, expected $cn 2^($cz - $s)")
  end
  out
end

const SWEEP_PRECISIONS = vcat(2:32, [34, 37, 40, 44, 48, 53, 56, 64, 72, 80, 90, 100, 113, 120, 128, 150, 160, 192, 200, 224, 240, 255, 256])

# levels, weights and scale significands of the kinds Int, BigInt, Float32, Float64 and BigFloat at a precision below,
# equal to and above p (above: one value that p bits do not hold, one that they hold)
sweep_levels(p) = Any[5, 7, 0.75, 1 / 3, 0.75f0, 1 / 3f0, big(125), bfq(11, max(p - 1, 2)), bfq(11, p), bfq(11, p + 11),
                      bfw(0.75, p + 11), bfq(11, 200)]
function sweep_weights(p)
  lower, upper = max(p - 1, 2), p + 11
  Any[Real[5, 7, 0.75, 1 / 3, 0.75f0, bfq(5, lower), bfq(5, p), bfw(0.75, upper), bfw(3, upper), big(125), 0, big(0.0), Inf,
           big(Inf), bfw(Inf, upper)],
      [5, 7], [0.75, 1 / 3], Float32[0.75, 1 / 3], BigFloat[bfq(5, lower), bfq(5, p), bfw(0.75, upper)], [big(125), big(7)],
      BigFloat[bfq(5, upper)], Real[5, bfq(3, upper), 0.75]]
end
sweep_scales(p) = BigFloat[bfh(11, 2), bfh(11, p), bfh(11, p + 9), bfw(0.75, p + 9), prevfloat(one(BigFloat))]

@testset "weight fold sweep" begin
  failures = String[]
  checked = 0
  for p in SWEEP_PRECISIONS
    setprecision(BigFloat, p) do
      for λ in sweep_levels(p), W in sweep_weights(p), αm in sweep_scales(p)
        lossy = loses(λ, p) || loses(αm, p) || any(wi -> usable(wi) && loses(wi, p), W)
        lm = model(λ, p)
        ews = Int128[pos(model(wi, p)) for wi in W if usable(wi)]
        lo, hi = extrema(ews)
        eλ = pos(lm)
        # the centre, then the positions of the largest and of the smallest coefficient at and one binade outside the margin
        shifts = Int128[-(eλ + (lo + hi) ÷ 2)]
        lossy || append!(shifts, (2 * EMAX - 6 - (eλ + hi) + d for d in (0, 1)))
        lossy || append!(shifts, (2 * EMIN + 6 - (eλ + lo) + d for d in (0, -1)))
        for αe in shifts
          checked += 1
          append!(failures, fold_failures(checked % 2 == 0 ? NormL1(λ, W) : NormL1Nonneg(λ, W), αm, αe))
        end
      end
      # weights spread over the room, at the centre of the range of positions: one binade inside, at, and one outside
      D0 = EMAX - EMIN - 6
      for λ in sweep_levels(p), αm in sweep_scales(p), D in (D0 - 1, D0, D0 + 1)
        a = -(D ÷ 2)
        W = [pw(a), pw(a + D)]
        αe = -(pos(model(λ, p)) + (a + 1)) - D ÷ 2
        checked += 1
        append!(failures, fold_failures(checked % 2 == 0 ? NormL1(λ, W) : NormL1Nonneg(λ, W), αm, αe))
      end
    end
  end
  isempty(failures) || foreach(m -> @info("fold mismatch", m), first(failures, 5))
  @test isempty(failures)
  @test checked > 20_000
  # both outcomes are exercised, near each boundary, and so are the refusals of a wider operand and its converse
  @test FOLDED[] > 5_000 && REFUSED[] > 5_000 && REFUSED_BY_PRECISION[] > 5_000
end

# A chain of two factors whose product with the level of the weighted norm is not in range, at the precision 2 of the
# counterexamples: the weight w = u 2^(2 - E) with u = 6/7 formed at 53 bits is not an operand of two bits.
function counterexample_chain(L, level, w)
  E = EMAX
  e = E ÷ 2
  a = Base.ldexp(one(BigFloat), Clong(e))
  b = Base.ldexp(one(BigFloat), Clong(E + 5 - e))
  Scaled(Scaled(L(level, [w]), a), b)
end

@testset "operands that are wider than the precision" begin
  # the operands of the counterexamples, formed at 53 bits
  u = setprecision(BigFloat, 53) do
    BigFloat(6) / 7
  end
  w53 = setprecision(BigFloat, 53) do
    Base.ldexp(u, Clong(2 - EMAX))
  end
  w34 = setprecision(BigFloat, 53) do
    Base.ldexp(BigFloat(3) / 4, Clong(2 - EMAX))      # 0.75 stored at 53 bits: two bits hold it
  end
  five_ninths = setprecision(BigFloat, 53) do
    BigFloat(5) / 9
  end

  with_precisions("the counterexamples refuse the fold, and the converse folds"; precisions = (2,)) do
    @assert precision(w53) == 53 && precision(w34) == 53
    for L in (NormL1, NormL1Nonneg)
      # a 53-bit weight, with a two-bit level
      s = counterexample_chain(L, big(0.75), w53)
      @test PB._fold_weights(L(big(0.75), [w53]), retained(s)) === nothing
      @test length(prox(s, [big(8192)], one(BigFloat))) == 1
      # a 53-bit level, with a two-bit weight
      wt = Base.ldexp(big(0.75), Clong(2 - EMAX))
      @test PB._fold_weights(L(u, [wt]), retained(counterexample_chain(L, u, wt))) === nothing
      # a 53-bit weight 5/9 and a 53-bit level, at a scale that the fold would otherwise take
      α = PB._ProjectionScale(big(0.75), 5)
      @test PB._fold_weights(L(big(0.75), [five_ninths]), α) === nothing
      @test PB._fold_weights(L(five_ninths, [big(0.75)]), α) === nothing
      # a wide scale significand
      @test PB._fold_weights(L(big(0.75), [big(0.75)]), PB._ProjectionScale(five_ninths, 5)) === nothing
      # one weight that p bits do not hold among others that they do (zero and infinite weights are not read)
      @test PB._fold_weights(L(big(0.75), [big(0.75), five_ninths, big(0.0)]), α) === nothing
      @test PB._fold_weights(L(big(0.75), [big(0.75), bfw(Inf, 53), big(0.0)]), α) isa L
      # the converse: the same values stored at 53 bits are held by two bits, so the fold applies and the result is that of
      # the operands stored at two bits
      s34 = counterexample_chain(L, big(0.75), w34)
      w2 = Base.ldexp(big(0.75), Clong(2 - EMAX))
      f34, f2 = PB._fold_weights(L(big(0.75), [w34]), retained(s34)), PB._fold_weights(L(big(0.75), [w2]), retained(s34))
      @test f34 isa L && f2 isa L
      @test f34.λ0 == f2.λ0 && f34.w == f2.w
      @test prox(s34, [big(8192)], one(BigFloat)) == prox(counterexample_chain(L, big(0.75), w2), [big(8192)], one(BigFloat))
      level34 = bfw(0.75, 53)
      g34, g2 = PB._fold_weights(L(level34, [w2]), retained(s34)), PB._fold_weights(L(big(0.75), [w2]), retained(s34))
      @test g34 isa L && g34.λ0 == g2.λ0 && g34.w == g2.w
      m34 = PB._ProjectionScale(bfw(0.75, 53), 5)
      h34, h2 = PB._fold_weights(L(big(0.75), [big(0.75)]), m34), PB._fold_weights(L(big(0.75), [big(0.75)]), PB._ProjectionScale(big(0.75), 5))
      @test h34 isa L && h34.λ0 == h2.λ0 && h34.w == h2.w
    end
  end

  # the boundary of each kind of operand, as a level, as a weight and as a scale significand, at several precisions: the
  # fold is refused exactly for a BigFloat that is wider than p and holds more than p bits, and nowhere else
  for p in (2, 3, 5, 24, 53, 64)
    setprecision(BigFloat, p) do
      @testset "kinds of operand at precision $p" begin
        base_level, base_weight = big(0.75), big(0.75)
        α0 = PB._ProjectionScale(big(0.75), 5)
        # (operand, refused): Int, BigInt, Float64 and Float32 operands, and BigFloat operands below, at and above p
        # (a BigFloat of q bits whose last bit is set: p bits hold it only for q <= p)
        needs(q) = nextfloat(bfw(1, q))
        kinds = Any[(5, false), ((big(1) << 70) + 1, false), (1 / 3, false), (1 / 3f0, false), (0.75, false), (0.75f0, false),
                    (needs(max(p - 1, 2)), false), (needs(p), false), (needs(p + 1), true), (needs(p + 11), true),
                    (bfw(0.75, p + 1), false), (bfw(0.75, p + 11), false), (bfw(3, p + 11), false)]
        for (x, refused) in kinds
          @testset "$(typeof(x)) $x" begin
            for L in (NormL1, NormL1Nonneg)
              as_level = PB._fold_weights(L(x, [base_weight]), α0)
              as_weight = PB._fold_weights(L(base_level, [x]), α0)
              @test (as_level === nothing) == refused
              @test (as_weight === nothing) == refused
              x isa BigFloat && @test (PB._fold_weights(L(base_level, [base_weight]), PB._ProjectionScale(x, 5)) === nothing) == refused
              refused || @test as_level isa L && as_weight isa L
            end
          end
        end
        # a zero and an infinite weight at any precision are not read
        @test PB._fold_weights(NormL1(base_level, [base_weight, bfw(Inf, p + 11), bfw(0.0, p + 11)]), α0) isa NormL1
      end
    end
  end
end

# The rounding of the helper follows the current mode: three factors 0.75 at two bits, 27/64 = 0.421875, lie between
# 0.375 and 0.5, nearer to 0.375
@testset "the helper honours the rounding mode" begin
  with_precisions("directed rounding"; precisions = (2,)) do
    f = big(0.75)
    n, z = exact(f)
    @test round_dyadic(n * n * n, 3 * z, 2) == exact(big(0.375))
    @test round_dyadic(n * n * n, 3 * z, 2, :up) == exact(big(0.5))
    @test round_dyadic(n * n * n, 3 * z, 2, :down) == exact(big(0.375))
    @test PB._scaled_weight(f, f, f, 0) == 0.375
    setrounding(BigFloat, RoundUp) do
      @test PB._scaled_weight(f, f, f, 0) == 0.5
      @test PB._scaled_weight(f, f, f, 3) == 4
      # the fold forms the same coefficient, and the failures are the oracle's, in the directed mode
      for L in (NormL1, NormL1Nonneg)
        @test isempty(fold_failures(L(f, [f]), f, 5, :up))
        @test PB._fold_weights(L(f, [f]), PB._ProjectionScale(f, 5)) isa L
      end
    end
    setrounding(BigFloat, RoundDown) do
      @test PB._scaled_weight(f, f, f, 0) == 0.375
      @test isempty(fold_failures(NormL1(f, [f]), f, 5, :down))
    end
    @test PB._scaled_weight(f, f, f, 0) == 0.375
    @test rounding(BigFloat) == RoundNearest
  end
end

@testset "_scaled_weight against the exact product" begin
  bad = String[]
  for p in SWEEP_PRECISIONS
    setprecision(BigFloat, p) do
      for pa in (2, p, p + 9), pl in (2, p + 3, 130), pww in (2, p, 140), (na, nl, nw) in ((11, 5, 3), (13, 11, 5))
        ma, ml, w = bfh(na - 2, pa), bfq(nl, pl), bfq(nw, pww)
        for k in (-7, 0, 5)
          (an, az), (ln, lz), (wn, wz) = exact(ma), exact(ml), exact(w)
          cn, cz = round_dyadic(an * ln * wn, az + lz + wz + k, p)
          r = PB._scaled_weight(ma, ml, w, k)
          rn, rz = exact(r)
          (precision(r) == p && rn == cn && rz == cz) || push!(bad, "p = $p, precisions $pa $pl $pww, k = $k: got $r")
        end
      end
    end
  end
  isempty(bad) || foreach(m -> @info("_scaled_weight mismatch", m), first(bad, 5))
  @test isempty(bad)
end

@testset "the earlier counterexamples" begin
  with_precisions("a level that rounds at the precision of the call"; precisions = (2,)) do
    # the level 5 is the model's 4 at two bits, and the same fold results for the level 4
    E = EMAX
    e = E ÷ 2
    a = Base.ldexp(one(BigFloat), Clong(e))
    b = Base.ldexp(big(0.75), Clong(E + 5 - e))
    w = Base.ldexp(big(0.75), Clong(-E + 5))
    for L in (NormL1, NormL1Nonneg)
      s5 = Scaled(Scaled(L(5, [w]), a), b)
      s4 = Scaled(Scaled(L(big(4), [w]), a), b)
      @test prox(s5, [big(8192)], one(BigFloat)) == prox(s4, [big(8192)], one(BigFloat))
      f5, f4 = PB._fold_weights(L(5, [w]), retained(s5)), PB._fold_weights(L(big(4), [w]), retained(s4))
      @test f5 isa L && f4 isa L
      @test f5.λ0 == f4.λ0 && f5.w == f4.w
    end
  end

  with_precisions("a product of three significands is rounded once"; precisions = (2,)) do
    # 0.75 (5/9, as 53 bits) 0.75 = 0.3125000000000000 and a bit: above the midpoint of the two-bit numbers 0.25 and 0.375
    ml = setprecision(BigFloat, 53) do
      BigFloat(5 / 9)
    end
    @test PB._scaled_weight(big(0.75), ml, big(0.75), 0) == 0.375
    @test PB._scaled_weight(big(0.75), ml, big(0.75), 3) == 3
  end

  with_precisions("the refusals"; precisions = (2, 53)) do
    # one weight, no spread: the absolute position of the coefficient is beyond what a level times a weight can express
    @test PB._fold_weights(NormL1(big(1), [big(1)]), PB._ProjectionScale(big(0.5), 3 * (EMAX - 2) + 1)) === nothing
    @test PB._fold_weights(NormL1Nonneg(big(1), [big(1)]), PB._ProjectionScale(big(0.5), 3 * (EMIN + 2) - 1)) === nothing
    # the same weight at an exponent inside the margins folds
    @test PB._fold_weights(NormL1(big(1), [big(1)]), PB._ProjectionScale(big(0.5), 2 * EMAX - 10)) isa NormL1
    # weights spread over the whole range, each position within what a level and a weight can express
    wide = NormL1(big(1), [pw(EMAX - 1), pw(EMIN - 1)])
    @test PB._fold_weights(wide, PB._ProjectionScale(big(0.5), 0)) === nothing
    # no weights, no nonzero finite weight, a zero level, a zero or infinite scale: no fold
    @test PB._fold_weights(NormL1(big(1)), PB._ProjectionScale(big(0.5), 3 * (EMAX - 2))) === nothing
    @test PB._fold_weights(NormL1(big(1), [big(0.0), big(Inf)]), PB._ProjectionScale(big(0.5), 3 * (EMAX - 2))) === nothing
    @test PB._fold_weights(NormL1(big(0), [big(1)]), PB._ProjectionScale(big(0.5), 3 * (EMAX - 2))) === nothing
    @test PB._fold_weights(NormL1(big(1), [big(1)]), PB._ProjectionScale(big(Inf), 0)) === nothing
  end

  @testset "the Float64 and Float32 paths have no weight fold" begin
    folds = methods(PB._fold_weights)
    @test !isempty(folds)
    @test all(m -> m.sig <: Tuple{Any, Union{NormL1, NormL1Nonneg}, PB._ProjectionScale{BigFloat}}, folds)
    for T in (Float32, Float64)
      @test !hasmethod(PB._fold_weights, Tuple{NormL1{T, Vector{T}}, PB._ProjectionScale{T, Int}})
      α = PB._NormalScale(PB._ProjectionScale(T(0.75), T === Float64 ? 5000 : 500))
      @test PB._fold_level(NormL1(T(1), T[1, 2]), α) === nothing
      @test PB._fold_level(NormL1Nonneg(T(1), T[1, 2]), α) === nothing
    end
  end
end

end
