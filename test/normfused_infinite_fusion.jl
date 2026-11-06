module NormFusedInfiniteFusionTest

# NormL1Fused with an infinite fusion level and a finite ℓ1 level (#664, decision 49): the domain is the
# vectors of equal entries (and the pairs x1ᵢ = x2ᵢ in the tuple form), and at a positive finite step
# the proximal point is the projection onto them followed by the ℓ1 shrink: every entry of a vector is
# z = shrink(mean(x), γλ1) (a single entry is shrunk by γλ1, no pairs), a pair is shrunk likewise. A NaN
# entry gives NaN in every entry of the group (E2), infinite entries of one sign give that infinity and of
# both signs NaN (the form of the pair). The value is λ1 Σ|v| at a vector of equal finite entries (the
# fusion term is exactly zero: no Inf⋅0), Inf at another finite vector and NaN at a NaN entry, and
# value_and_prox! returns the value at the stored output (O2). λ1 = Inf keeps the hard zero of
# decision 49, and a finite λ2 whose product only overflows is not infinite (R3).
# Rows marked "regression guard" pass at base: the pair, the K = 2 vector, the tuple of arrays and the
# K = 2 FiberSum already give the common value. Outputs are compared with isequal, so NaN equals NaN.
# The mean and the threshold are checked against a BigFloat reference on finite entries.

using Test, ProximalBase, Random

const PB = ProximalBase

####################################
#  the BigFloat reference
####################################

# The proximal point of NormL1Fused(λ1, Inf) at a vector of K ≥ 1 entries and a positive finite step:
# the correctly rounded common value, and the tolerance that the rounding of the mean and of the
# threshold allows. The mean of the entries is formed in 4000 bits, independently of the package.
function refmean(x)
  setprecision(BigFloat, 4000) do
    sum(BigFloat.(x)) / length(x)
  end
end
function refprox(::Type{T}, λ1, γ, x) where {T}
  n = length(x)
  any(isnan, x) && return (fill(T(NaN), n), zero(T))
  if any(isinf, x)
    both = any(==(T(Inf)), x) && any(==(-T(Inf)), x)
    return (fill(both ? T(NaN) : any(==(T(Inf)), x) ? T(Inf) : -T(Inf), n), zero(T))
  end
  setprecision(BigFloat, 4000) do
    m = refmean(x)
    c = BigFloat(γ) * BigFloat(λ1)
    z = sign(m) * max(abs(m) - c, 0)
    (fill(T(z), n), T(4) * eps(T) * T(min(abs(m) + c, BigFloat(floatmax(T)))) + 2 * nextfloat(zero(T)))
  end
end
# the output y of a call against the reference, entry by entry
function closeprox(y, λ1, γ, x)
  T = eltype(y)
  z, tol = refprox(T, λ1, γ, x)
  length(y) == length(z) || return false
  all(i -> isequal(y[i], y[1]), eachindex(y)) || return false
  isempty(y) && return true
  isfinite(z[1]) ? abs(big(y[1]) - big(z[1])) <= big(tol) : isequal(y[1], z[1])
end

# the value λ1 K |c| of a vector of equal entries, correctly rounded
function refvalue(::Type{T}, λ1, n, c) where {T}
  setprecision(BigFloat, 4000) do
    T(BigFloat(λ1) * n * abs(BigFloat(c)))
  end
end
closeval(v, r::T) where {T} = isinf(r) || iszero(r) ? isequal(v, r) : abs(big(v) - big(r)) <= 2 * big(eps(r))

####################################
#  the calls
####################################

# prox, prox! and value_and_prox! into a fresh output, an output holding a sentinel, and the argument
# itself (a copy), against one expected output (isequal); the value returned is that of the stored
# output (O2)
function check(f, x, γ, expected; value = nothing)
  T = float(eltype(x))
  @test isequal(prox(f, x, γ), expected)
  for op in (prox!, value_and_prox!)
    fresh = similar(x, T)
    v1 = op(fresh, f, x, γ)
    sentinel = fill(T(-7), size(x))
    op(sentinel, f, x, γ)
    alias = copy(x)
    v3 = op(alias, f, alias, γ)
    @test isequal(fresh, expected) && isequal(sentinel, expected) && isequal(alias, expected)
    if op === value_and_prox!
      @test isequal(v1, f(fresh)) && isequal(v3, f(alias))
      value === nothing || @test isequal(v1, value) && isequal(v3, value)
    end
  end
end

# The same with the reference: every output is the common value within the tolerance of the reference
function checkref(f, x, γ)
  λ1 = f.λ1
  T = float(eltype(x))
  y = prox(f, x, γ)
  @test eltype(y) === T && closeprox(y, λ1, γ, x)
  fresh = similar(x, T)
  v = value_and_prox!(fresh, f, x, γ)
  alias = copy(x)
  w = value_and_prox!(alias, f, alias, γ)
  s = prox!(fill(T(-7), size(x)), f, x, γ)
  @test isequal(fresh, y) && isequal(alias, y) && isequal(s, y)
  @test isequal(v, f(fresh)) && isequal(w, f(alias))
end

####################################
#  the issue, and the rows that fail without the exceptional route
####################################

@testset "NormL1Fused(λ1, Inf): the vector of K ≠ 2 entries is the shrunk mean" begin
  for T in (Float32, Float64)
    x = T[1, -2, 0.5]
    # the issue: [-Inf, Inf, NaN] at base
    check(NormL1Fused(T(1), T(Inf)), x, T(1), zeros(T, 3); value = zero(T))
    check(NormL1Fused(T(0), T(Inf)), x, T(1), fill(T(-0.5) / 3, 3); value = zero(T))
    check(NormL1Fused(T(1), T(Inf)), x, T(0.1), fill(T(-0.5) / 3 + T(0.1), 3))
    for λ1 in (0, 1e-3, 0.1, 0.3, 1, T === Float32 ? 1e30 : 1e300), γ in (T(1), T(0.1), T(1e-3), 1.0, 2, T(40)), x in (x, T[3, 3, 3], T[1, 2, 3, 4], T[5])
      checkref(NormL1Fused(T(λ1), T(Inf)), x, γ)
    end
    # the threshold is γλ1, not Kγλ1: the mean 2 of [1, 2, 3] is shrunk by 0.5
    check(NormL1Fused(T(0.5), T(Inf)), T[1, 2, 3], T(1), fill(T(1.5), 3))
    check(NormL1Fused(T(0.5), T(Inf)), T[1, 2, 3, 4, 5], T(1), fill(T(2.5), 5))
    check(NormL1Fused(T(0.5), T(Inf)), T[-1, -2, -3], T(2), fill(T(-1), 3))
    # K = 1 has no pairs: the plain ℓ1 shrink; [NaN] at base
    check(NormL1Fused(T(0.5), T(Inf)), T[3], T(1), T[2.5]; value = T(1.25))
    check(NormL1Fused(T(0.5), T(Inf)), T[-3], T(1), T[-2.5])
    check(NormL1Fused(T(0.5), T(Inf)), T[0.25], T(1), T[0]; value = zero(T))
    check(NormL1Fused(T(0), T(Inf)), T[4], T(1), T[4]; value = zero(T))
    # K = 0: unchanged
    @test prox(NormL1Fused(T(0.5), T(Inf)), T[], T(1)) == T[]
    # a large K, past the cutoff of the sort
    rng = MersenneTwister(664)
    for n in (4, 5, 7, 10, 100, 257, 4096, 4097, 10_000)
      xs = T.(randn(rng, n))
      for λ1 in (0.0, 0.01, 0.5), γ in (T(1), T(3))
        checkref(NormL1Fused(T(λ1), T(Inf)), xs, γ)
      end
    end
  end
end

@testset "NormL1Fused(λ1, Inf): repeated values, zeros, mixed signs" begin
  for T in (Float32, Float64)
    for λ1 in (0.0, 0.25, 2.0)
      f = NormL1Fused(T(λ1), T(Inf))
      check(f, zeros(T, 5), T(1), zeros(T, 5); value = zero(T))
      check(f, fill(T(3), 6), T(1), fill(T(max(3 - λ1, 0)), 6))
      check(f, T[2, 2, 2, 1, 1, 1], T(1), fill(T(max(1.5 - λ1, 0)), 6))
      check(f, T[-3, 3, -3, 3], T(1), zeros(T, 4))
      check(f, T[-1, -1, -1, 1], T(1), fill(T(-0.5 + min(λ1, 0.5)), 4))
      checkref(f, T[-0.0, 0.0, -0.0], T(1))
      checkref(f, T[1, -1, 1, -1, 1], T(0.5))
    end
    # levels and step in other precisions
    checkref(NormL1Fused(0.25, Inf), T[1, 2, 3.5], 0.5)
    checkref(NormL1Fused(0.25f0, Inf32), T[1, 2, 3.5], 0.5f0)
    checkref(NormL1Fused(0.25, Inf), T[1, 2, 3.5], 1)
    checkref(NormL1Fused(1//4, Inf), T[1, 2, 3.5], 1)
  end
  # levels stored in one precision, output in the other
  check(NormL1Fused(0.25, Inf), Float32[1, 2, 3.5], 1.0, fill(Float32(2.1666667 - 0.25), 3))
  @test eltype(prox(NormL1Fused(0.25f0, Inf32), [1.0, 2.0, 3.5], 1.0)) === Float64
  # a finite level that does not fit the output stays finite: the product overflows and shrinks to zero
  check(NormL1Fused(1e300, Inf), Float32[1, 2, 3.5], 1.0, zeros(Float32, 3))
  check(NormL1Fused(1e300, Inf), [1.0, 2.0, 3.5], 1.0, zeros(3))
  check(NormL1Fused(1e300, Inf), Float32[1, 2, 3.5], 1e300, zeros(Float32, 3))
end

@testset "NormL1Fused(λ1, Inf): the mean without overflow, cancellation and underflow" begin
  rng = MersenneTwister(20261106)
  for T in (Float32, Float64)
    fm = floatmax(T)
    f0 = NormL1Fused(T(0), T(Inf))
    # same-sign entries near floatmax: the sum overflows, the mean does not
    check(f0, T[fm, fm, fm], T(1), fill(fm, 3))
    check(f0, T[-fm, -fm, -fm, -fm], T(1), fill(-fm, 4))
    checkref(f0, T[fm, fm, fm / 2], T(1))
    # opposite signs: the entries cancel
    check(f0, T[fm, -fm, fm, -fm], T(1), zeros(T, 4))
    check(f0, T[fm, fm, -fm, -fm], T(1), zeros(T, 4))
    checkref(f0, T[fm, fm, -fm], T(1))
    checkref(f0, T[fm, -fm, T(0.5)], T(1))
    checkref(f0, T[fm, fm, -fm, -fm, T(1)], T(1))
    checkref(f0, T[fm, fm, fm, -fm, -fm, 3], T(1))
    # a small entry beside cancelling huge ones is kept
    x = T[fm, -fm, T(1e-30)]
    checkref(f0, x, T(1))
    # subnormal entries
    tiny = nextfloat(zero(T))
    checkref(f0, T[tiny, tiny, tiny], T(1))
    checkref(f0, T[tiny, 2tiny, 5tiny], T(1))
    checkref(f0, T[floatmin(T), tiny, -tiny], T(1))
    checkref(NormL1Fused(T(1e-3), T(Inf)), T[floatmin(T), floatmin(T), floatmin(T)], T(1e-3))
    # random vectors over a wide dynamic range, and near floatmax
    for n in (3, 4, 5, 7, 10, 64, 1000), _ in 1:4
      checkref(f0, T.(randn(rng, n) .* exp.(10 .* randn(rng, n))), T(1))
      checkref(f0, T.(clamp.(randn(rng, n) .* 1e38, -3e38, 3e38)) .* (T === Float64 ? T(1e270) : T(1)), T(1))
      checkref(NormL1Fused(T(0.5), T(Inf)), T.(randn(rng, n)), T(0.75))
    end
    for n in (3, 5, 9)
      x = fm .* T.(rand(rng, n) .* 0.5 .+ 0.5)            # all of the same sign, near floatmax
      checkref(f0, x, T(1))
      checkref(f0, -x, T(1))
      x = fm .* T.(2 .* rand(rng, n) .- 1)
      checkref(f0, x, T(1))
    end
    # the exact mean of an integer sequence
    for n in (3, 5, 9, 101)
      check(f0, T.(1:n), T(1), fill(T((n + 1) / 2), n))
    end
  end
end

@testset "NormL1Fused(λ1, Inf): NaN and infinite entries" begin
  for T in (Float32, Float64)
    nan, inf = T(NaN), T(Inf)
    huge = T === Float32 ? 1e30 : 1e300        # a finite level in T (T(1e300) is Inf in Float32)
    for λ1 in (0.0, 0.5, huge)
      f = NormL1Fused(T(λ1), inf)
      # a NaN entry at every position gives NaN in every entry (E2), whatever else the vector holds
      for n in (1, 3, 4, 5), k in 1:n
        x = T.(1:n)
        x[k] = nan
        # a single entry has no pairs: its value keeps the rule of the standalone call (zero at λ1 = 0)
        check(f, x, T(1), fill(nan, n); value = n == 1 && iszero(λ1) ? nothing : nan)
      end
      check(f, T[nan, nan, nan], T(1), fill(nan, 3))
      check(f, T[nan, inf, 1], T(1), fill(nan, 3))
      check(f, T[-inf, nan, inf], T(1), fill(nan, 3))
      # infinite entries: the form of the pair (the infinity, or NaN for both signs)
      λ1 < huge && check(f, T[1, inf, 2], T(1), fill(inf, 3))
      λ1 < huge && check(f, T[1, -inf, 2], T(1), fill(-inf, 3))
      λ1 < huge && check(f, T[inf, inf, 2], T(1), fill(inf, 3))
      check(f, T[inf, -inf, 2], T(1), fill(nan, 3))
      check(f, T[inf, 1, -inf, 2], T(1), fill(nan, 4))
      λ1 < huge && check(f, T[-inf, -inf, -inf], T(1), fill(-inf, 3))
    end
    # the same entries as a pair, a K = 2 vector and a tuple of arrays: the K = 2 pins
    f = NormL1Fused(T(0.5), inf)
    for (p, e) in ((T[1, nan], (nan, nan)), (T[nan, 1], (nan, nan)), (T[inf, 1], (inf, inf)),
                   (T[inf, inf], (inf, inf)), (T[inf, -inf], (nan, nan)), (T[-inf, -inf], (-inf, -inf)),
                   (T[floatmax(T), floatmax(T)], (floatmax(T), floatmax(T))),
                   (T[floatmax(T), -floatmax(T)], (T(0), T(0))), (T[1, 2], (T(1), T(1))), (T[3, 1], (T(1.5), T(1.5))))
      @test isequal(prox(f, (p[1], p[2]), T(1)), e)
      @test isequal(prox(f, p, T(1)), collect(e))
      @test isequal(prox(f, ([p[1]], [p[2]]), T(1)), ([e[1]], [e[2]]))
    end
  end
end

####################################
#  values
####################################

@testset "NormL1Fused(λ1, Inf): standalone values" begin
  for T in (Float32, Float64)
    inf, nan = T(Inf), T(NaN)
    for λ1 in (0.0, 0.5, 2.0, 1e-3)
      f = NormL1Fused(T(λ1), inf)
      # equal entries: the fusion term is exactly zero (no Inf⋅0), the value is λ1 Σ|v|
      for c in (T(1), T(-2.5), T(0.1), T(1e-5)), n in (1, 2, 3, 4, 7, 100)
        r = refvalue(T, T(λ1), n, c)
        @test closeval(f(fill(c, n)), r)
        n == 2 && @test closeval(f((c, c)), r)
        n == 1 && @test closeval(f([c]), r)
      end
      @test f(zeros(T, 3)) === zero(T) && f(T[-0.0, 0.0, -0.0]) === zero(T) && f((T(0), T(-0.0))) === zero(T)
      @test f(T[]) === zero(T)
    end
    # the overflow of the ℓ1 term and the scaled value: the value of the whole is rounded once
    f = NormL1Fused(T(1e-3), inf)
    @test closeval(f(fill(floatmax(T) / 4, 5)), refvalue(T, T(1e-3), 5, floatmax(T) / 4))
    @test closeval(f(fill(floatmax(T), 3)), refvalue(T, T(1e-3), 3, floatmax(T)))
    @test NormL1Fused(T(2), inf)(fill(floatmax(T), 3)) === inf
    @test closeval(NormL1Fused(T(1e-3), inf)(fill(floatmin(T), 3)), refvalue(T, T(1e-3), 3, floatmin(T)))
  end
  # a level stored in another precision
  @test NormL1Fused(0.5, Inf)(Float32[1, 1, 1]) === 1.5f0
  @test NormL1Fused(0.5f0, Inf32)([1.0, 1.0, 1.0]) === 1.5
  @test NormL1Fused(1e300, Inf)(Float32[1, 1, 1]) === Inf32
  # the pair and the tuple of arrays: independent pairs, each of equal entries
  f = NormL1Fused(0.5, Inf)
  @test f(([1.0, 2.0, -3.0], [1.0, 2.0, -3.0])) === 6.0
  # one pair that differs: Inf, where the constant pairs gave Inf⋅0 = NaN
  @test f(([1.0, 2.0, -3.0], [1.0, 2.0, -3.5])) === Inf
  @test closeval(f(([floatmax()], [floatmax()])), refvalue(Float64, 0.5, 2, floatmax()))
  @test NormL1Fused(0.0, Inf)(([1.0, 2.0], [1.0, 2.0])) === 0.0
  @test NormL1Fused(0.5, Inf)((Float64[], Float64[])) === 0.0
  @test NormL1Fused(0.5, Inf)(([1.0f0, 2.0f0], [1.0f0, 2.0f0])) === 3.0f0
end

@testset "NormL1Fused(λ1, Inf): standalone values at another vector, NaN and infinite entries (regression guards)" begin
  for T in (Float32, Float64)
    inf, nan = T(Inf), T(NaN)
    for λ1 in (0.0, 0.5, 2.0, 1e-3)
      f = NormL1Fused(T(λ1), inf)
      # another finite vector: Inf, however close the entries are
      @test f(T[1, 2, 3]) === inf && f(T[1, 2]) === inf && f((T(1), T(2))) === inf
      @test f(T[1, nextfloat(T(1)), 1]) === inf && f((T(1), nextfloat(T(1)))) === inf
      @test f(T[0, nextfloat(zero(T)), 0]) === inf && f(T[floatmax(T), -floatmax(T)]) === inf
      # NaN: NaN (E2), also with another pair that differs
      @test isnan(f(T[1, nan, 1])) && isnan(f(T[nan, 1])) && isnan(f((T(1), nan))) && isnan(f(T[nan, 2, 3]))
    end
  end
  f = NormL1Fused(0.5, Inf)
  @test isnan(f(([1.0, NaN], [1.0, 2.0]))) && isnan(f(([1.0, 2.0], [1.0, NaN])))
  @test isnan(f(([1.0, 3.0, NaN], [1.0, 4.0, 2.0])))
  # infinite entries keep the ordinary values
  @test NormL1Fused(0.5, Inf)((1.0, Inf)) === Inf && NormL1Fused(0.5, Inf)([Inf, 1.0, 2.0]) === Inf
end

@testset "NormL1Fused(λ1, Inf): value_and_prox! returns the value at the stored output (O2)" begin
  for T in (Float32, Float64)
    for λ1 in (0.0, 0.5, 2.0), x in (T[1, -2, 0.5], T[3, 1, 2, 5], T[3], T[5, 5, 5], T[1e3, 2e3, 3e3, 4e3]), γ in (T(1), T(0.25))
      f = NormL1Fused(T(λ1), T(Inf))
      y = similar(x)
      v = value_and_prox!(y, f, x, γ)
      z = y[1]
      @test all(==(z), y) && v === f(y) && closeval(v, refvalue(T, T(λ1), length(x), z))
    end
    # at the zero output the value is zero, not NaN
    y = zeros(T, 3)
    @test value_and_prox!(y, NormL1Fused(T(5), T(Inf)), T[1, -2, 0.5], T(1)) === zero(T) && y == zeros(T, 3)
    # the pair and a K = 2 vector
    for f in (NormL1Fused(T(0.5), T(Inf)),)
      y = zeros(T, 2)
      @test value_and_prox!(y, f, T[3, 1], T(1)) === T(1.5) && y == T[1.5, 1.5]
      @test value_and_prox!(y, f, T[0.25, 0.1], T(1)) === zero(T) && y == zeros(T, 2)
    end
  end
end

####################################
#  the pair, the tuple of arrays and the K = 2 vector (regression guards for the outputs)
####################################

@testset "NormL1Fused(λ1, Inf): pair, tuple of arrays and K = 2 vector, outputs (regression guards)" begin
  for T in (Float32, Float64)
    f = NormL1Fused(T(0.25), T(Inf))
    @test isequal(prox(f, (T(1), T(-2)), T(1)), (T(-0.25), T(-0.25)))
    @test isequal(prox(NormL1Fused(T(0), T(Inf)), (T(1), T(-2)), T(1)), (T(-0.5), T(-0.5)))
    @test isequal(prox(f, (T(3), T(1)), T(1)), (T(1.75), T(1.75)))
    @test isequal(prox(f, T[3, 1], T(1)), T[1.75, 1.75])
    x = ([T(1), T(3), T(-5), T(NaN)], [T(-2), T(3), T(-5), T(1)])
    expected = ([T(-0.25), T(2.75), T(-4.75), T(NaN)], [T(-0.25), T(2.75), T(-4.75), T(NaN)])
    for op in (prox!, value_and_prox!)
      y = (zeros(T, 4), zeros(T, 4))
      op(y, f, x, T(1))
      @test isequal(y, expected)
      a = (copy(x[1]), copy(x[2]))
      op(a, f, a, T(1))
      @test isequal(a, expected)
    end
  end
end

@testset "NormL1Fused(λ1, Inf): pairs, values at the stored output and a pair near floatmax" begin
  for T in (Float32, Float64)
    f = NormL1Fused(T(0.25), T(Inf))
    # constant pairs have a finite value, a pair with NaN is NaN
    y = (zeros(T, 3), zeros(T, 3))
    v = value_and_prox!(y, f, (T[1, 3, -5], T[-2, 3, -5]), T(1))
    @test v === f(y) && v === T(3.875)
    y = (zeros(T, 2), zeros(T, 2))
    @test isnan(value_and_prox!(y, f, (T[1, NaN], T[3, 1]), T(1)))
  end
  # a pair near floatmax: the separate and the aliased output are the same (O7)
  for T in (Float32, Float64)
    fm = floatmax(T)
    f = NormL1Fused(T(0.5), T(Inf))
    x = ([fm, T(1)], [fm, T(3)])
    y = (zeros(T, 2), zeros(T, 2))
    v = value_and_prox!(y, f, x, T(1))
    a = (copy(x[1]), copy(x[2]))
    w = value_and_prox!(a, f, a, T(1))
    p = prox!((zeros(T, 2), zeros(T, 2)), f, x, T(1))
    @test isequal(y, p) && isequal(a, p) && isequal(p[1][1], fm)
    @test isequal(v, f(y)) && isequal(w, f(a)) && isfinite(v) && v == w
  end
end

####################################
#  FiberSum
####################################

# the fibers of the array against the per-fiber reference: the function of the fiber on every
# off-diagonal fiber and the diagonal function on the diagonal fibers (a free diagonal keeps them)
function refprox_fibers(fb, A, γ)
  T = float(eltype(A))
  Y = zeros(T, size(A))
  own = fb.diagonal !== nothing
  for i in axes(A, 1), j in axes(A, 2)
    g = i == j ? fb.diagonal : fb.f
    if g === nothing
      Y[i, j, :] .= A[i, j, :]
    else
      z, _ = refprox(T, g.λ1, γ, A[i, j, :])
      Y[i, j, :] .= z
    end
  end
  Y
end
# the fibers agree with the reference where both are finite within the tolerance, else exactly
function samefibers(Y, R, A, fb, γ)
  T = eltype(Y)
  all(CartesianIndices(Y)) do I
    isequal(Y[I], R[I]) && return true
    i, j, k = Tuple(I)
    g = i == j ? fb.diagonal : fb.f
    g === nothing && return false
    _, tol = refprox(T, g.λ1, γ, A[i, j, :])
    isfinite(Y[I]) && isfinite(R[I]) && abs(big(Y[I]) - big(R[I])) <= big(tol)
  end
end

function symmetrized(rng, T, n, K; special = false)
  A = zeros(T, n, n, K)
  for k in 1:K
    B = T.(randn(rng, n, n))
    A[:, :, k] .= (B + B') / 2
  end
  A
end

# the fibers of random arrays for the functions with and without a diagonal, symmetric or not
function fibercheck(Ks; outputs, values)
  rng = MersenneTwister(664)
  for T in (Float32, Float64), K in Ks, sym in (false, true), (λ1, λ1d) in ((0.1, 0.05), (0.0, 0.3), (0.5, 0.0))
    A = sym ? symmetrized(rng, T, 4, K) : T.(randn(rng, 4, 4, K))
    for diagonal in (nothing, :same, NormL1Fused(T(λ1d), T(Inf)), NormL1Fused(T(λ1d), T(0.5)))
      f = NormL1Fused(T(λ1), T(Inf))
      fb = FiberSum(f; diagonal = diagonal === :same ? f : diagonal, symmetric = sym)
      for γ in (T(1), T(0.2))
        diagonal isa NormL1Fused && isfinite(diagonal.λ2) && continue  # a finite λ2 on the diagonal: its own tests (fibers)
        Y = prox(fb, copy(A), γ)
        S = fill(T(-7), size(A))
        v = value_and_prox!(S, fb, A, γ)
        B = copy(A)
        w = value_and_prox!(B, fb, B, γ)
        P = prox!(fill(T(-7), size(A)), fb, A, γ)
        if outputs
          @test samefibers(Y, refprox_fibers(fb, A, γ), A, fb, γ)
          @test isequal(S, Y) && isequal(B, Y) && isequal(P, Y)
        end
        values && @test isequal(v, fb(S)) && isequal(w, fb(B)) && isfinite(v)
      end
    end
  end
end

@testset "FiberSum of NormL1Fused(λ1, Inf): K = 1, 3, 4 slices (the view route), outputs and values" begin
  fibercheck((1, 3, 4); outputs = true, values = true)
  # the value of the fibers against the sum of the ℓ1 terms of the stored output
  T = Float64
  A = T[1 2; 3 4;;; -1 0; 5 6;;; 0.5 1; 1.5 2]
  fb = FiberSum(NormL1Fused(0.25, Inf); diagonal = nothing)
  S = zeros(T, 2, 2, 3)
  v = value_and_prox!(S, fb, A, 1.0)
  @test v === fb(S) && isapprox(v, 0.25 * 3 * (abs(S[1, 2, 1]) + abs(S[2, 1, 1])); rtol = 1e-14)
end

@testset "FiberSum of NormL1Fused(λ1, Inf): K = 2 slices, outputs (regression guards)" begin
  fibercheck((2,); outputs = true, values = false)
end

@testset "FiberSum of NormL1Fused(λ1, Inf): K = 2 slices, values at the stored output" begin
  fibercheck((2,); outputs = false, values = true)
end

@testset "FiberSum of NormL1Fused(λ1, Inf): K = 2 pins (regression guards)" begin
  nan = NaN
  for T in (Float32, Float64)
    fm = floatmax(T)
    # entries: NaN in a coupled pair, ±Inf, floatmax of both signs
    A = zeros(T, 3, 3, 2)
    A[:, :, 1] .= T[1 fm nan; Inf -fm 2; 3 4 -Inf]
    A[:, :, 2] .= T[-2 fm 1; Inf fm 2.5; Inf 6 1]
    f = NormL1Fused(T(0.5), T(Inf))
    for diag in (nothing, f, NormL1Fused(T(0.1), T(Inf)))
      fb = FiberSum(f; diagonal = diag)
      Y = prox(fb, A, T(1))
      # off-diagonal fibers: the pair rule
      @test isequal(Y[1, 2, :], T[fm, fm])
      @test isequal(Y[1, 3, :], T[nan, nan])
      @test isequal(Y[2, 1, :], T[Inf, Inf])
      @test isequal(Y[2, 3, :], T[1.75, 1.75])
      @test isequal(Y[3, 1, :], T[Inf, Inf])
      @test isequal(Y[3, 2, :], T[4.5, 4.5])
      # the diagonal fibers: the diagonal function, or free
      if diag === nothing
        @test isequal(Y[1, 1, :], T[1, -2]) && isequal(Y[2, 2, :], T[-fm, fm]) && isequal(Y[3, 3, :], T[-Inf, 1])
      else
        t = T(diag.λ1)
        @test isequal(Y[1, 1, :], fill(T(-0.5) + t, 2)) && isequal(Y[2, 2, :], T[0, 0]) && isequal(Y[3, 3, :], T[-Inf, -Inf])
      end
      @test isequal(prox!(fill(T(-7), size(A)), fb, A, T(1)), Y)
      B = copy(A)
      prox!(B, fb, B, T(1))
      @test isequal(B, Y)
    end
  end
end

@testset "FiberSum of NormL1Fused(λ1, Inf): separate, aliased outputs and values with NaN, infinite and floatmax fibers" begin
  nan = NaN
  for T in (Float32, Float64)
    fm = floatmax(T)
    A = zeros(T, 3, 3, 2)
    A[:, :, 1] .= T[1 fm nan; Inf -fm 2; 3 4 -Inf]
    A[:, :, 2] .= T[-2 fm 1; Inf fm 2.5; Inf 6 1]
    f = NormL1Fused(T(0.5), T(Inf))
    for diag in (nothing, f, NormL1Fused(T(0.1), T(Inf)))
      fb = FiberSum(f; diagonal = diag)
      Y = prox(fb, A, T(1))
      # one output through the separate and the aliased calls (O7), the value at it (O2): NaN fibers
      S = fill(T(-7), size(A))
      v = value_and_prox!(S, fb, A, T(1))
      B = copy(A)
      w = value_and_prox!(B, fb, B, T(1))
      @test isequal(S, Y) && isequal(B, Y)
      @test isnan(v) && isnan(w)
    end
    # without the NaN and infinite fibers: the pairs near floatmax, the separate output is that of prox!
    C = zeros(T, 2, 2, 2)
    C[:, :, 1] .= T[1 fm; fm -fm]
    C[:, :, 2] .= T[-2 fm; fm fm]
    for diag in (nothing, f)
      fb = FiberSum(f; diagonal = diag)
      Y = prox(fb, C, T(1))
      S = fill(T(-7), size(C))
      v = value_and_prox!(S, fb, C, T(1))
      B = copy(C)
      w = value_and_prox!(B, fb, B, T(1))
      @test isequal(S, Y) && isequal(B, Y) && isequal(Y[1, 2, :], T[fm, fm])
      @test isequal(v, fb(S)) && isequal(w, fb(B)) && v == w
    end
  end
end

####################################
#  λ1 = Inf, finite λ2 and the steps that keep their results
####################################

@testset "NormL1Fused: λ1 = Inf keeps the hard zero, and a finite λ2 is not infinite" begin
  for T in (Float32, Float64)
    for f in (NormL1Fused(T(Inf), T(Inf)), NormL1Fused(T(Inf), T(1)), NormL1Fused(T(Inf), T(0)))
      for x in (T[1, -2, 0.5], T[1, NaN, 2], T[Inf, 1], T[1], T[Inf, 1, -Inf])
        check(f, x, T(1), zeros(T, length(x)); value = zero(T))
      end
      @test f(zeros(T, 3)) === zero(T) && f(T[1, 2, 3]) === T(Inf) && isnan(f(T[1, NaN, 3]))
    end
  end
  # a huge finite λ2 is not infinite (R3): the ordinary route and the finite value, not the equal-entries
  # route (Inf at a nonconstant vector). The output of the finite-level arithmetic is the earlier one:
  # the entries are lost to the products γλ2 (2k - n - 1), which is not a supported value
  x = [1.0, -2.0, 0.5]
  @test isequal(prox(NormL1Fused(0.0, 1e300), x, 1.0), zeros(3))
  @test NormL1Fused(0.5, 1e300)(fill(1.0, 3)) === 1.5 && NormL1Fused(0.5, floatmax())(fill(1.0, 3)) === 1.5
  @test isfinite(NormL1Fused(0.5, 1e300)([1.0, 2.0, 3.0])) && NormL1Fused(0.5, 1e300)([1.0, 2.0, 3.0]) > 1e300
  @test NormL1Fused(0.5, Inf)([1.0, 2.0, 3.0]) === Inf
end

@testset "NormL1Fused(λ1, Inf): the steps and the lengths that keep their results" begin
  nan = NaN
  for T in (Float32, Float64)
    f = NormL1Fused(T(0.5), T(Inf))
    x = T[1, 2, 3]
    # not covered by the rule, no endpoint is documented: the results are the earlier ones
    @test isequal(prox(f, x, T(0)), fill(T(nan), 3))
    @test isequal(prox(f, x, T(Inf)), fill(T(nan), 3))
    @test isequal(prox(f, x, T(nan)), fill(T(nan), 3))
    y = prox(f, x, T(-1))
    @test !all(isfinite, y) && all(isnan(y[i]) || isinf(y[i]) for i in 1:3)
    for γ in (T(0), T(Inf), T(-1), T(nan))
      @test isequal(prox!(similar(x), f, x, γ), prox(f, x, γ))
    end
    # a K = 2 vector and the pair at the endpoints
    @test isequal(prox(f, T[1, 2], T(0)), T[1.5, 1.5]) && isequal(prox(f, T[1, 2], T(Inf)), T[0, 0])
    @test isequal(prox(f, (T(1), T(2)), T(0)), (T(1.5), T(1.5))) && isequal(prox(f, (T(1), T(2)), T(Inf)), (T(0), T(0)))
  end
end

@testset "NormL1Fused(λ1, Inf): validation before writes (O6), views and aliasing (O7)" begin
  f = NormL1Fused(0.5, Inf)
  y = fill(-7.0, 2)
  @test_throws DimensionMismatch prox!(y, f, [1.0, 2.0, 3.0], 1.0)
  @test y == fill(-7.0, 2)
  @test_throws DimensionMismatch value_and_prox!(y, f, [1.0, 2.0, 3.0], 1.0)
  @test y == fill(-7.0, 2)
  y = fill(-7.0, 3)
  @test_throws DimensionMismatch prox!(y, f, [1.0, 2.0], 1.0)
  @test y == fill(-7.0, 3)
  # views of larger arrays, and an output that is a view of the argument's storage
  base = [9.0, 1.0, -2.0, 8.0, 0.5, 7.0]
  x = view(base, 2:5)
  out = fill(-7.0, 6)
  prox!(view(out, 2:5), f, x, 1.0)
  @test out[1] == -7.0 && out[6] == -7.0 && closeprox(out[2:5], 0.5, 1.0, [1.0, -2.0, 8.0, 0.5])
  y = copy(base)
  prox!(view(y, 2:5), f, view(y, 2:5), 1.0)
  @test y[1] == 9.0 && y[6] == 7.0 && isequal(y[2:5], out[2:5])
  z = copy(base)
  value_and_prox!(view(z, 2:5), f, view(z, 2:5), 1.0)
  @test isequal(z, y)
  # a strided view, a matrix column
  M = reshape(collect(1.0:12.0), 3, 4)
  col = view(M, :, 2)
  @test closeprox(prox(f, col, 1.0), 0.5, 1.0, [4.0, 5.0, 6.0])
  @test closeprox(prox(f, view(M, 1, :), 1.0), 0.5, 1.0, [1.0, 4.0, 7.0, 10.0])
end

####################################
#  round 2: the working type, the mean at cancellation, the endpoint steps
####################################

@testset "NormL1Fused(λ1, Inf): the mean and the threshold keep the precision of the argument (T3, R3)" begin
  f0 = NormL1Fused(0.0, Inf)
  # finite entries of a wider argument are not narrowed to infinities or rounded before the mean
  for op in (prox!, value_and_prox!)
    y = zeros(Float32, 3)
    op(y, f0, [1e100, -1e100, 3.0], 1.0)
    @test y == ones(Float32, 3)
    y = zeros(Float32, 3)
    op(y, f0, [16777217.0, -16777216.0, 2.0], 1.0)
    @test y == ones(Float32, 3)
    y = zeros(Float16, 4)
    op(y, f0, [1e10, -1e10, 3.0, 1.0], 1.0)
    @test y == fill(Float16(1), 4)
    y = zeros(Float32, 3)
    op(y, f0, Float64[1e300, -1e300, 6.0], 1.0)
    @test y == fill(Float32(2), 3)
  end
  @test prox(f0, [1e100, -1e100, 3.0], 1.0) == ones(3)
  # the threshold is formed in the working type too, and z is rounded once, on store
  f = NormL1Fused(0.5, Inf)
  y = zeros(Float32, 3)
  v = value_and_prox!(y, f, [1e100, -1e100, 3.0], 1.0)
  @test y == fill(Float32(0.5), 3) && v === f(y) && v === Float32(0.75)
  y = zeros(Float32, 3)
  value_and_prox!(y, NormL1Fused(1e-9, Inf), [1e100, -1e100, 3.0], 0.5)
  @test y == ones(Float32, 3)       # 1 - 5e-10 rounds to 1 in Float32, after the mean in Float64
  y = zeros(Float32, 3)
  prox!(y, f0, fill(1e300, 3), 1.0)  # a result that overflows the destination is Inf, not an error
  @test y == fill(Inf32, 3)
  # integer, Float32 and BigFloat arguments
  @test prox(f0, [1, -2, 7, 4], 1.0) == fill(2.5, 4)
  @test prox(f0, Float32[1, -2, 7, 4], 1.0) == fill(2.5f0, 4) && eltype(prox(f0, Float32[1, -2, 7, 4], 1.0)) === Float32
  y = zeros(3)
  prox!(y, f0, Float32[1e30, -1e30, 3], 1.0)
  @test y == ones(3)
  @test prox(f0, BigFloat[1, 2, 6], BigFloat(1)) == fill(BigFloat(3), 3)
  # FiberSum, the view route of K = 3 slices: the fibers are narrowed to the output only on store
  for sym in (false, true), diagonal in (nothing, :same)
    f = NormL1Fused(0.0, Inf)
    fb = FiberSum(f; diagonal = diagonal === :same ? f : nothing, symmetric = sym)
    A = zeros(3, 3, 3)
    for i in 1:3, j in 1:3
      A[i, j, :] = [1e100, -1e100, 3.0]
    end
    Y = zeros(Float32, 3, 3, 3)
    v = value_and_prox!(Y, fb, A, 1.0)
    # the fibers of the off-diagonal pairs; the free diagonal fibers keep their entries, narrowed
    off = [I for I in CartesianIndices(Y) if I[1] != I[2] || diagonal === :same]
    @test all(==(1f0), Y[off]) && isequal(v, fb(Y))
    Y = zeros(Float32, 3, 3, 3)
    prox!(Y, fb, A, 1.0)
    @test all(==(1f0), Y[off])
  end
end

# the mean of a vector against the BigFloat sum, correct to the rounding of the result and, in the
# subnormal range, to two minimum subnormals
function meancheck(T, x)
  y = prox(NormL1Fused(zero(T), T(Inf)), x, one(T))
  length(y) == length(x) && closeprox(y, zero(T), one(T), x)
end

@testset "NormL1Fused(λ1, Inf): the mean after an overflow keeps a small residual (BigFloat reference)" begin
  for T in (Float32, Float64)
    d, M = nextfloat(zero(T)), floatmax(T)
    # the reviewer's cases: the exact means are 32 d and 8 d / 5 (a minimum subnormal)
    x = zeros(T, 65)
    x[1:4] = T[M, M, -M, -M]
    x[end] = 2080d
    @test prox(NormL1Fused(zero(T), T(Inf)), x, one(T)) == fill(32d, 65)
    @test meancheck(T, x)
    x = T[M, M, -M, -M, 8d]
    @test prox(NormL1Fused(zero(T), T(Inf)), x, one(T)) == fill(2d, 5)     # 1.6 d rounds to 2 d
    @test meancheck(T, x)
    for perm in ([1, 2, 3, 4, 5], [5, 1, 3, 2, 4], [3, 5, 1, 4, 2])
      @test meancheck(T, x[perm])
    end
    # the same, aliased and by value_and_prox!
    y = copy(x)
    value_and_prox!(y, NormL1Fused(zero(T), T(Inf)), y, one(T))
    @test y == fill(2d, 5)
    # a grid: pairs of large entries that cancel exactly, a tail of multiples of d, a subnormal final mean
    rng = MersenneTwister(20261107)
    bigs = T[M, M / 2, M / 3, M * T(0.75), nextfloat(M / 2), prevfloat(M), M / 7]
    for n in (5, 6, 9, 17, 33, 65, 100, 257, 1000), pairs in (1, 2, 3, 4), tail in (1, 8, 2080), _ in 1:3
      2pairs < n || continue
      x = zeros(T, n)
      for j in 1:pairs
        v = rand(rng, bigs)
        x[2j-1] = v
        x[2j] = -v
      end
      x[2pairs+1:end] .= d .* T.(rand(rng, 0:tail, n - 2pairs))
      shuffle!(rng, x)
      @test meancheck(T, x)
    end
    # the large entries cancel exactly in pairs of unequal magnitude, a small tail of other sizes
    for n in (7, 12, 40), _ in 1:6
      big = [rand(rng, bigs) * T(1 - 0.01 * rand(rng)) for _ in 1:3]
      x = T[big; -big; T(1e-30) * T.(randn(rng, n - 6))]
      shuffle!(rng, x)
      @test meancheck(T, x)
    end
    # same-sign large entries, and large entries that cancel to a few quanta, plus a small tail
    for n in (3, 4, 5, 8, 31, 64, 200), _ in 1:6
      @test meancheck(T, M .* T.(rand(rng, n) .* 0.5 .+ 0.5))
      x = M .* T.(2 .* rand(rng, n) .- 1)
      @test meancheck(T, x)
      x = [M * T(1 - 2.0^-20 * rand(rng)) for _ in 1:n]
      x[1:2:end] .*= -1
      @test meancheck(T, x)
      x = T[M, M, M * T(0.5), -M, -M, -M * T(0.5), 1, 2, 3, 4]
      @test meancheck(T, shuffle(rng, x))
    end
  end
end

@testset "NormL1Fused(λ1, Inf): endpoint steps keep their outputs, the repair of a separate output included" begin
  # the earlier results at a zero, infinite, negative and NaN step (#311, #665): not supported values.
  # The repair of a separate output at a pair near floatmax is taken at a finite positive step only.
  M = floatmax()
  f = NormL1Fused(0.5, Inf)
  fb = FiberSum(f; diagonal = nothing)
  A = zeros(2, 2, 2)
  A[:, :, 1] .= M
  A[:, :, 2] .= M
  for (γ, sep, alias, value) in ((Inf, NaN, 0.0, NaN), (0.0, Inf, M, NaN), (-1.0, Inf, Inf, Inf), (NaN, NaN, NaN, NaN))
    y = ([0.0], [0.0])
    v = value_and_prox!(y, f, ([M], [M]), γ)
    @test isequal(y[1][1], sep) && isequal(v, value) && (γ == -1 ? isequal(y[2][1], -Inf) : isequal(y[2][1], sep))
    a = ([M], [M])
    value_and_prox!(a, f, a, γ)
    @test isequal(a[1][1], alias) && isequal(prox!(([0.0], [0.0]), f, ([M], [M]), γ)[1][1], alias)
    Y = zeros(2, 2, 2)
    u = value_and_prox!(Y, fb, A, γ)
    @test isequal(Y[1, 2, 1], sep) && isequal(u, value) && (γ == -1 ? isequal(Y[1, 2, 2], -Inf) : isequal(Y[1, 2, 2], sep))
    B = copy(A)
    value_and_prox!(B, fb, B, γ)
    @test isequal(B[1, 2, 1], alias)
  end
  # at a finite positive step the repair is taken: the separate and the aliased outputs agree
  y = ([0.0], [0.0])
  v = value_and_prox!(y, f, ([M], [M]), 1.0)
  @test y == ([M], [M]) && v === M
end

end
