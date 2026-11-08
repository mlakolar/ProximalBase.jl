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
  iszero(e) && return iszero(r)
  r == e
end

function expect(r, N::Integer, s::Integer = 1; what = "")
  ok = agrees(r, N, s)
  ok || @info "mismatch" what result = r expected = oracle(N, s) N
  @test ok
end

function with_precisions(f, name)
  @testset "$name, precision $p" for p in PRECISIONS
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
# levels, and the step is part of the fold), or through the level and the weight (the projections form the product of the
# scale, the level and the weight). The products of a level and a weight that leave the range are not folded by the value
# or by the proximal map of a weighted norm, so those are not asked here.
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

end
