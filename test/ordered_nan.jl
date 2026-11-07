module OrderedNaNTest

# A NaN entry in SLOPE, NormL1Fused (vector form) and IndIsotonic (numerics.md E2, E4, O2): an
# output depends on an entry when, for some finite values of the other entries, changing it
# changes the output. A dependent output is NaN, an independent one is computed as usual, and
# a zero weight, a zero step or a single entry comes first (E4).

using Test
using ProximalBase
using Random
using ..TestUtils

const PB = ProximalBase

nans(y) = all(isnan, y)
softt(v, t) = sign(v) * max(abs(v) - t, zero(v))

# the result of op on x into a fresh output, an alias of x, and an output holding a sentinel
function three_ways(op, f, x, γ...)
  T = float(eltype(x))
  fresh = similar(x, T)
  v = op(fresh, f, x, γ...)
  alias = copy(x)
  op(alias, f, alias, γ...)
  sentinel = fill(T(-7), length(x))
  op(sentinel, f, x, γ...)
  (fresh, alias, sentinel, v)
end

function check_all(f, x, γ, expected; value = nothing)
  for op in (prox!, value_and_prox!)
    fresh, alias, sentinel, v = three_ways(op, f, x, γ)
    @test isequal(fresh, expected) && isequal(alias, expected) && isequal(sentinel, expected)
    value === nothing || op === prox! || @test isequal(v, value)
  end
  @test isequal(prox(f, x, γ), expected)
end

@testset "SLOPE" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    f = SLOPE(T[3, 2, 1])
    # non-constant weights: every output depends on every entry
    check_all(f, T[1, NaN, 0.5], T(1), fill(nan, 3); value = nan)
    check_all(f, T[NaN, 1, 0.5], T(1), fill(nan, 3); value = nan)
    check_all(f, T[1, 0.5, NaN], T(0.3), fill(nan, 3); value = nan)
    check_all(f, T[NaN, NaN, 0.5], T(1), fill(nan, 3); value = nan)
    check_all(SLOPE(T[1, 0, 0]), T[3, NaN, 2], T(1), fill(nan, 3); value = nan)      # the ℓ∞ norm
    check_all(SLOPE(T[2, 2, 0]), T[3, NaN, 2], T(1), fill(nan, 3); value = nan)
    @test isnan(f(T[1, NaN, 0.5]))
    # equal weights: soft thresholding, which a NaN does not reach
    check_all(SLOPE(T[1, 1, 1]), T[3, NaN, 2], T(1), T[2, NaN, 1])
    check_all(SLOPE(T[1, 1, 1]), T[-3, NaN, 0.5], T(2), T[-1, NaN, 0])
    check_all(SLOPE(T[0.5]), T[NaN], T(1), T[NaN])
    # zero weights: the zero function (E4), the identity at every step, and its value 0
    z = SLOPE(zeros(T, 3))
    check_all(z, T[3, NaN, 2], T(1), T[3, NaN, 2]; value = zero(T))
    check_all(z, T[3, NaN, 2], T(Inf), T[3, NaN, 2]; value = zero(T))
    check_all(z, T[3, -1, 2], T(Inf), T[3, -1, 2]; value = zero(T))
    @test z(T[3, NaN, Inf]) === zero(T) && z(T[3, 1, 2]) === zero(T)
    # a zero step: the identity
    check_all(f, T[3, NaN, 2], T(0), T[3, NaN, 2])
    # an infinite step keeps the master computation (LIM-056, #469)
    @test isequal(prox(f, T[3, NaN, 2], T(Inf)), T[0, NaN, 0])
    # an infinite first weight at a positive finite step: the domain is {0}, so the output is zero
    # whatever the entries hold (decision 49)
    @test prox(SLOPE(T[Inf, 1, 0.5]), T[1, 2, 0.5], T(1)) == zeros(T, 3)
    check_all(SLOPE(T[Inf, 1, 0.5]), T[1, NaN, 0.5], T(1), zeros(T, 3); value = zero(T))
    # finite arguments are unchanged
    @test prox(f, T[3, -2, 0.5], T(0.5)) == prox(SLOPE(T[3, 2, 1]), T[3, -2, 0.5], T(0.5))
    @test prox(SLOPE(zeros(T, 3)), T[3, -2, 0.5], T(Inf)) == T[3, -2, 0.5]
  end
  # weights that differ in Float64 but are the same threshold in the call's precision: soft thresholding
  fm = SLOPE([1.0, 1.0 - 2.0^-30, 1.0 - 2.0^-29])
  @test Float32.(fm.λ) == fill(1f0, 3)
  xm = Float32[NaN, 3, 2]
  check_all(fm, xm, 1f0, Float32[NaN, 2, 1])
  check_all(fm, Float32[3, 2, NaN], 1f0, Float32[2, 1, NaN])
  # distinct weights whose products with the step round to one threshold (normal range, no recovery)
  let a = nextfloat(nextfloat(1.5f0)), γ = 1.5f0, b = nextfloat(a)
    λc = Float32[b, a, a]
    @test a != b && all(==(γ * a), γ .* λc)
    c = γ * a
    check_all(SLOPE(λc), Float32[NaN, 4, 3], γ, Float32[NaN, 4 - c, 3 - c])
    check_all(SLOPE(λc), Float32[4, NaN, -3], γ, Float32[4 - c, NaN, -(3 - c)])
    @test isnan(value_and_prox!(zeros(Float32, 3), SLOPE(λc), Float32[NaN, 4, 3], γ))
  end
  # the same weights in Float64 are three different thresholds: all NaN
  check_all(fm, [NaN, 3.0, 2.0], 1.0, fill(NaN, 3))
  # the value and the projection at the stored output
  f = SLOPE([3.0, 2.0, 1.0])
  y = zeros(3)
  @test isnan(value_and_prox!(y, f, [1.0, NaN, 0.5], 1.0)) && nans(y)
  # constant weights, larger vectors, and random NaN patterns: the finite entries are the soft
  # thresholds, with the NaN entries kept
  for n in (5, 12, 60, 5000), seed in 1:20
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    x[randperm(rng, n)[1:rand(rng, 1:max(1, n ÷ 3))]] .= NaN
    λ = 0.2
    y = prox(SLOPE(fill(λ, n)), x, 0.7)
    ref = [isnan(v) ? NaN : softt(v, 0.7 * λ) for v in x]
    @test all(i -> isnan(ref[i]) ? isnan(y[i]) : y[i] == ref[i], 1:n)
    g = SLOPE(collect(range(0.5, 0.1; length = n)))
    @test nans(prox(g, x, 0.7)) && isnan(g(x))
  end
  # buffers stay valid after a NaN call
  g, fresh = SLOPE([3.0, 2.0, 1.0, 0.5]), SLOPE([3.0, 2.0, 1.0, 0.5])
  xf = [1.0, -3.0, 0.5, 2.0]
  for _ in 1:3
    @test nans(prox(g, [1.0, NaN, 0.5, 2.0], 1.0))
    @test prox(g, xf, 0.4) == prox(fresh, xf, 0.4)
    @test g(xf) == fresh(xf)
  end
  # no allocation once warm, NaN or not
  yb = zeros(4)
  for x in ([1.0, NaN, 0.5, 2.0], xf), h in (g, SLOPE(fill(0.3, 4)))
    @test allocs(prox!, yb, h, x, 1.0) == 0
    @test allocs(value_and_prox!, yb, h, x, 1.0) == 0
  end
  # a SparseIterate as the output
  s = SparseIterate(3)
  prox!(s, SLOPE([3.0, 2.0, 1.0]), [1.0, NaN, 0.5], 1.0)
  @test all(i -> isnan(s[i]), 1:3)
end

@testset "NormL1Fused, vector form" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    for (λ1, λ2) in ((1, 1), (0, 1), (0.3, 2))
      f = NormL1Fused(T(λ1), T(λ2))
      check_all(f, T[1, NaN, 0.5], T(1), fill(nan, 3); value = nan)
      check_all(f, T[NaN, 1, 0.5, 2], T(0.4), fill(nan, 4); value = nan)
      check_all(f, T[1, 2, 0.5, 2, 3, NaN, -1], T(0.4), fill(nan, 7); value = nan)
      @test isnan(f(T[1, NaN, 0.5])) && isnan(f(T[1, NaN, 0.5, 3]))
      # a step too small for γλ2 to be a normal number still couples the entries
      check_all(f, T[1, NaN, 0.5, 2], floatmin(T) * T(1e-3), fill(nan, 4); value = nan)
    end
    # no fusion: soft thresholding
    f = NormL1Fused(T(1), T(0))
    check_all(f, T[3, NaN, 2], T(1), T[2, NaN, 1]; value = nan)
    check_all(f, T[-3, NaN, 0.5, 1.5], T(1), T[-2, NaN, 0, 0.5])
    # both levels zero: the identity at every step, and the value 0
    f = NormL1Fused(T(0), T(0))
    check_all(f, T[3, NaN, 2], T(1), T[3, NaN, 2]; value = zero(T))
    check_all(f, T[3, NaN, 2], T(Inf), T[3, NaN, 2]; value = zero(T))
    @test f(T[3, NaN, 2]) === zero(T)
    # a zero step: the identity
    check_all(NormL1Fused(T(1), T(1)), T[3, NaN, 2], T(0), T[3, NaN, 2])
    # an infinite step keeps the master computation (LIM-056)
    @test all(isnan, prox(NormL1Fused(T(1), T(1)), T[3, NaN, 2], T(Inf)))
    # two entries: the closed form of the pair; a pair of the tuple form couples only itself
    f = NormL1Fused(T(1), T(1))
    @test nans(prox(f, T[3, NaN], T(1)))
    u, v = prox(f, (T(NaN), T(1.5)), T(1))
    @test isnan(u) && isnan(v)
    y = prox(f, (T[NaN, 1.5, 1], T[NaN, 1.5, 0]), T(1))
    @test isnan(y[1][1]) && isnan(y[2][1]) && isfinite(y[1][2]) && isfinite(y[1][3]) && isfinite(y[2][3])
    # finite arguments are unchanged: the sorted vector, compared with the pair kernel at K = 2
    @test prox(f, T[3, 1], T(1)) == collect(prox(f, (T(3), T(1)), T(1)))
  end
  # one entry has no pairs: the fusion term is an empty sum (λ1 = 0 gives value 0, λ1 > 0 stays NaN)
  for T in (Float32, Float64)
    for (λ1, λ2) in ((0, 1), (0, 0))
      f = NormL1Fused(T(λ1), T(λ2))
      @test f(T[NaN]) === zero(T)
      y = T[7]
      @test value_and_prox!(y, f, T[NaN], T(1)) === zero(T) && isnan(y[1])
      @test f(T[2]) === zero(T)
    end
    for λ2 in (0, 1)
      f = NormL1Fused(T(0.3), T(λ2))
      @test isnan(f(T[NaN]))
      y = T[7]
      @test isnan(value_and_prox!(y, f, T[NaN], T(1))) && isnan(y[1])
      @test prox(f, T[2], T(1)) == T[2 - 0.3]
    end
  end
  # the identity regimes of the pair route come before the pair kernel: a zero step, and both levels
  # zero at every step (γ = Inf included), keep a NaN in its own entry
  for T in (Float32, Float64)
    nan = T(NaN)
    f, z = NormL1Fused(T(1), T(1)), NormL1Fused(T(0), T(0))
    check_all(f, T[NaN, 3], T(0), T[NaN, 3])
    check_all(f, T[3, NaN], T(0), T[3, NaN])
    check_all(z, T[NaN, 3], T(1), T[NaN, 3]; value = zero(T))
    check_all(z, T[3, 4], T(Inf), T[3, 4]; value = zero(T))
    check_all(z, T[NaN, 3], T(Inf), T[NaN, 3]; value = zero(T))
    u, w = prox(f, (T(NaN), T(3)), T(0))
    @test isnan(u) && w == 3
    u, w = prox(f, (T(3), T(NaN)), T(0))
    @test u == 3 && isnan(w)
    @test prox(z, (T(3), T(4)), T(Inf)) == (T(3), T(4))
    @test prox(z, (T(NaN), T(4)), T(Inf))[2] == 4
    for (g, γ) in ((f, T(0)), (z, T(Inf)), (z, T(1)))
      x = (T[NaN, 3, 1], T[2, 4, NaN])
      y = (zeros(T, 3), zeros(T, 3))
      v = value_and_prox!(y, g, x, γ)
      @test isequal(y[1], x[1]) && isequal(y[2], x[2])
      @test g === z ? v === zero(T) : isnan(v)
      ya = (copy(x[1]), copy(x[2]))
      v = value_and_prox!(ya, g, ya, γ)
      @test isequal(ya[1], x[1]) && isequal(ya[2], x[2])
      @test g === z ? v === zero(T) : isnan(v)
      yb = (copy(x[1]), copy(x[2]))
      prox!(yb, g, yb, γ)
      @test isequal(yb[1], x[1]) && isequal(yb[2], x[2])
      yp = prox(g, x, γ)
      @test isequal(yp[1], x[1]) && isequal(yp[2], x[2])
    end
    # finite pairs are unchanged at a zero step
    @test prox(f, (T(1), T(3)), T(0)) == (T(1), T(3))
    # FiberSum with two slices (the pair kernel), with a diagonal function of its own: the pairs of
    # the identity regimes keep a NaN in its own entry, and the other pairs are coupled
    for (g, γ) in ((f, T(0)), (z, T(Inf)), (z, T(1)))
      for diagonal in (nothing, g)
        fb = FiberSum(g; diagonal)
        X = randn(MersenneTwister(1), T, 3, 3, 2)
        X[1, 2, 1] = NaN
        X[3, 1, 2] = NaN
        Y = similar(X)
        v = value_and_prox!(Y, fb, X, γ)
        @test isequal(Y, X) && isequal(prox(fb, X, γ), X)
        Z = copy(X)
        prox!(Z, fb, Z, γ)
        @test isequal(Z, X)
        W = copy(X)
        @test value_and_prox!(W, fb, W, γ) isa T && isequal(W, X)
        g === z && @test v === zero(T)
        g === f && @test isnan(v)
      end
    end
    # zero levels with a zero step (levels and step already zero): the value is masked, the output the input
    for diagonal in (nothing, z, NormL1Fused(T(0), T(0))), symmetric in (false, true)
      fb = FiberSum(z; diagonal, symmetric)
      X = randn(MersenneTwister(5), T, 4, 4, 2)
      if symmetric
        X .= (X .+ permutedims(X, (2, 1, 3))) ./ 2
        X[3, 1, 1] = NaN
        X[1, 3, 1] = NaN
      else
        X[2, 1, 1] = NaN
        X[1, 4, 2] = NaN
      end
      Y = similar(X)
      @test value_and_prox!(Y, fb, X, T(0)) === zero(T) && isequal(Y, X) && fb(Y) === zero(T)
      W = copy(X)
      @test value_and_prox!(W, fb, W, T(0)) === zero(T) && isequal(W, X)
      @test isequal(prox(fb, X, T(0)), X)
    end
    # a positive step is not an identity regime: the pair that holds the NaN is coupled, the others are not
    fb = FiberSum(f)
    X = randn(MersenneTwister(1), T, 3, 3, 2); X[1, 2, 1] = NaN
    Y = prox(fb, X, T(0.5))
    @test isnan(Y[1, 2, 1]) && isnan(Y[1, 2, 2]) && all(isfinite, Y[2, 3, :])
    # a fiber function that is zero adds no value, and a NaN in its entries is not its value
    fb = FiberSum(z; diagonal = z)
    X = randn(MersenneTwister(2), T, 3, 3, 2); X[2, 2, 1] = NaN
    @test value_and_prox!(similar(X), fb, X, T(0.7)) === zero(T)
  end
  # no fusion, larger vectors, random NaN patterns: the finite entries are the soft thresholds
  for n in (5, 12, 60, 5000), seed in 1:20
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    x[randperm(rng, n)[1:rand(rng, 1:max(1, n ÷ 3))]] .= NaN
    y = prox(NormL1Fused(0.4, 0.0), x, 0.7)
    @test all(i -> isnan(x[i]) ? isnan(y[i]) : y[i] == softt(x[i], 0.7 * 0.4), 1:n)
    @test nans(prox(NormL1Fused(0.4, 0.3), x, 0.7)) && isnan(NormL1Fused(0.4, 0.3)(x))
  end
  # buffers stay valid after a NaN call
  g, fresh = NormL1Fused(0.3, 0.4), NormL1Fused(0.3, 0.4)
  xf = [1.0, -3.0, 0.5, 2.0, 0.1]
  for _ in 1:3
    @test nans(prox(g, [1.0, NaN, 0.5, 2.0, 0.1], 1.0))
    @test prox(g, xf, 0.4) == prox(fresh, xf, 0.4)
    @test g(xf) == fresh(xf)
  end
  yb = zeros(5)
  for x in ([1.0, NaN, 0.5, 2.0, 0.1], xf), h in (g, NormL1Fused(0.3, 0.0))
    @test allocs(prox!, yb, h, x, 1.0) == 0
    @test allocs(value_and_prox!, yb, h, x, 1.0) == 0
  end
  # an infinite ℓ1 level: the domain is {0}, and the output zero (decision 49)
  @test prox(NormL1Fused(Inf, 1.0), [1.0, 2.0, 0.5], 1.0) == zeros(3)
  # a fiber of three or more slices is a vector; the fibers are independent, and K = 2 uses pairs
  for K in (2, 3, 4)
    X = randn(MersenneTwister(K), 3, 3, K)
    Xn = copy(X)
    Xn[1, 2, 1] = NaN
    for diagonal in (nothing, NormL1Fused(0.3, 0.4))
      fb = FiberSum(NormL1Fused(0.3, 0.4); diagonal)
      Y, Yn = prox(fb, X, 0.5), prox(fb, Xn, 0.5)
      for j in 1:3, i in 1:3
        fiber = Yn[i, j, :]
        if (i, j) == (1, 2)
          K == 2 ? (@test isnan(fiber[1]) && isnan(fiber[2])) : @test nans(fiber)
        else
          @test fiber == Y[i, j, :]
        end
      end
      @test isnan(fb(Xn))
      y = similar(Xn)
      @test isnan(value_and_prox!(y, fb, Xn, 0.5))
    end
  end
end

@testset "IndIsotonic" begin
  for T in (Float32, Float64), decreasing in (false, true)
    nan = T(NaN)
    f = IndIsotonic(; decreasing)
    for x in (T[1, NaN, 0.5], T[NaN, 1, 2, 3], T[3, 2, 1, NaN], T[NaN, NaN], T[1, 2, 3, 4, NaN, 5, 6],
              T[Inf, NaN, -Inf], T[NaN, 1])
      n = length(x)
      for γ in (T(1), T(0), T(Inf))
        check_all(f, x, γ, fill(nan, n); value = nan)
      end
      # an array of steps, equal or not
      for γs in (ones(T, n), T.(1:n))
        for op in (prox!, value_and_prox!)
          y = fill(T(-7), n)
          v = op(y, f, x, γs)
          @test nans(y)
          op === value_and_prox! && @test isnan(v)
        end
        z = copy(x)
        prox!(z, f, z, γs)
        @test nans(z)
      end
      @test isnan(f(x)) && isnan(f(reverse(x)))
    end
    # a single entry is feasible, and its own projection
    check_all(f, T[NaN], T(1), T[NaN]; value = zero(T))
    @test f(T[NaN]) === zero(T) && f(T[2]) === zero(T)
    @test isequal(prox(f, T[NaN], T[2]), T[NaN])
    # an infeasible vector before a NaN is still NaN, and without one it is Inf
    @test isnan(f(decreasing ? T[1, 2, 0, NaN] : T[2, 1, 0, NaN]))
    @test f(decreasing ? T[1, 2, 0, 3] : T[2, 1, 0, 3]) === T(Inf)
    @test f(decreasing ? T[Inf, Inf, 1] : T[1, Inf, Inf]) === zero(T)
    # finite arguments: the value at the stored point is 0, and the projection is unchanged
    x = T[3, 1, 2, 5, 4]
    y = zeros(T, 5)
    @test value_and_prox!(y, f, x, T(1)) === zero(T)
    @test y == prox(IndIsotonic(; decreasing), x)
    @test value_and_prox!(y, f, x, T[1, 2, 1, 2, 1]) === zero(T)
  end
  # finite arguments store no NaN: the value is 0 without reading the output, also near the range limit
  for T in (Float32, Float64), xs in (T[3, 1, 2, 5, 4], collect(range(T(-1), T(1); length = 50)),
                                      T[floatmax(T), -floatmax(T), floatmax(T), 0, -floatmax(T)])
    y = zeros(T, length(xs))
    @test value_and_prox!(y, IndIsotonic(), xs, T(1)) === zero(T) && !any(isnan, y)
    @test value_and_prox!(y, IndIsotonic(), xs, T.(1:length(xs))) === zero(T) && !any(isnan, y)
  end
  # an argument entry that is finite but overflows the working type stores a NaN: the value is the one at y
  let f = IndIsotonic(), x = [1e100, -1e100, 1.0], γ = Float32[1, 2, 3]
    y = fill(-7f0, 3)
    v = value_and_prox!(y, f, x, γ)
    @test isnan(v) && isequal(v, f(y)) && isnan(y[1]) && isnan(y[2])
    y64 = fill(-7f0, 3)
    v = value_and_prox!(y64, f, x, 1.0)
    @test isequal(v, f(y64))
    for decreasing in (false, true)
      g = IndIsotonic(; decreasing)
      yy = zeros(Float32, 3)
      @test isequal(value_and_prox!(yy, g, x, γ), g(yy))
      @test isequal(value_and_prox!(yy, g, x, Float32[2, 2, 2]), g(yy))
    end
  end
  # opposite infinities pool to a NaN that is stored (E3 territory); the value is the one at the stored point
  y = zeros(3)
  @test isnan(value_and_prox!(y, IndIsotonic(), [Inf, -Inf, 3.0], 1.0)) && isnan(y[1]) && y[3] == 3
  # buffers stay valid after a NaN call, also with weights
  g, fresh = IndIsotonic(), IndIsotonic()
  xf = [3.0, 1.0, 2.0, 5.0, 4.0]
  γs = [1.0, 2.0, 1.0, 2.0, 1.0]
  for _ in 1:3
    @test nans(prox(g, [3.0, NaN, 2.0, 5.0, 4.0], 1.0))
    @test nans(prox(g, [3.0, NaN, 2.0, 5.0, 4.0], γs))
    @test prox(g, xf, 1.0) == prox(fresh, xf, 1.0)
    @test prox(g, xf, γs) == prox(fresh, xf, γs)
  end
  yb = zeros(5)
  for x in ([3.0, NaN, 2.0, 5.0, 4.0], xf)
    @test allocs(prox!, yb, g, x, 1.0) == 0
    @test allocs(value_and_prox!, yb, g, x, 1.0) == 0
    @test allocs(value_and_prox!, yb, g, x, γs) == 0
    @test allocs(g, x) == 0
  end
  # integer and rational arguments have no NaN
  @test prox(IndIsotonic(), [3, 1, 2]) == [2, 2, 2]
  @test IndIsotonic()([1, 2, 3]) === 0.0
  # a SparseIterate as the output
  s = SparseIterate(4)
  prox!(s, IndIsotonic(), [3.0, NaN, 2.0, 5.0], 1.0)
  @test all(i -> isnan(s[i]), 1:4)
  # the value is validated before anything is written
  @test_throws ArgumentError prox!(zeros(3), IndIsotonic(), [1.0, NaN, 2.0], [1.0, -1.0, 1.0])
  @test_throws DimensionMismatch prox!(zeros(2), IndIsotonic(), [1.0, NaN, 2.0], 1.0)
  w = fill(-7.0, 3)
  @test_throws ArgumentError prox!(w, IndIsotonic(), [1.0, NaN, 2.0], [1.0, 0.0, 1.0])
  @test w == fill(-7.0, 3)
end

end
