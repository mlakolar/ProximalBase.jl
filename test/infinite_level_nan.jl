module InfiniteLevelNaNTest

# An admitted infinite level that makes a proximal map, or an entry of it, constant on finite
# arguments (decision 49, E2 after E4): a NaN or infinite entry of the argument does not reach the
# constant output. NormL1 with an infinite level at a positive step (γ = Inf included, since the
# domain of a fixed entry is {0} whatever a wrapper adds), SLOPE with an infinite first weight and
# NormL1Fused with an infinite ℓ1 level at a positive finite step. A standalone value stays NaN at
# a NaN entry, and value_and_prox! returns the value at the stored output (O2).
# Expected outputs are constants, identities and exact soft thresholds written here, or the results
# of the unchanged cases (zero, infinite and rejected steps, finite parameters that overflow,
# configurations that are not constant maps). NaN is compared with isequal.

using Test, ProximalBase, SparseArrays
using JLArrays: JLArray

const PB = ProximalBase

# prox, then prox! and value_and_prox! into a fresh output, an output holding a sentinel, and the
# argument itself; every entry with isequal, and the value against the standalone value at the output
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
      value === nothing || @test(isequal(v1, value) && isequal(v3, value))
    end
  end
end

# the zero vector with the signs that SLOPE gives a finite entry (a NaN entry gives 0.0)
signed_zeros(x) = [signbit(v) & !isnan(v) ? -zero(float(eltype(x))) : zero(float(eltype(x))) for v in x]

a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
allocs_vprox(y, f, x, γ) = (a_vprox!(y, f, x, γ); @allocated a_vprox!(y, f, x, γ))

####################################
#  NormL1
####################################

@testset "NormL1: an infinite level fixes the penalized entries at a positive step" begin
  for T in (Float32, Float64)
    f = NormL1(T(Inf))
    xs = (T[1, NaN], T[NaN, 1], T[-3, NaN, 2], T[NaN, NaN, NaN], T[Inf, -Inf, 1], T[-Inf, NaN, 0],
          T[floatmax(T), -floatmax(T)], T[-0.0, 0.0, -1], T[1, -2])
    for γ in (T(1), T(0.5), floatmin(T), nextfloat(zero(T)), T(Inf), 1e-300, 1e300), x in xs
      check(f, x, γ, zeros(T, length(x)); value = zero(T))
    end
    for v in T[NaN, Inf, -Inf, 1, -0.0], γ in (T(1), floatmin(T), T(Inf))
      @test PB.prox_entry(f, v, 1, γ) === zero(T)
    end
    # array steps, entry by entry; a zero step entry is the identity, NaN included
    check(f, T[NaN, 1, Inf], T[1, 2, 0.5], zeros(T, 3); value = zero(T))
    check(f, T[NaN, NaN], T[Inf, 1], zeros(T, 2); value = zero(T))
    check(f, T[NaN, 1], T[0, 1], T[NaN, 0])
    # weights: an infinite weight is fixed and a zero weight leaves its entry free (E4)
    g = NormL1(T(Inf), T[Inf, 0, 1])
    for γ in (T(1), T(Inf))
      check(g, T[NaN, NaN, NaN], γ, T[0, NaN, 0])
      check(g, T[NaN, 5, NaN], γ, T[0, 5, 0]; value = zero(T))
      check(g, T[Inf, -Inf, -Inf], γ, T[0, -Inf, 0])
    end
    check(NormL1(T(Inf), [1, 0]), T[NaN, NaN], T(1), T[0, NaN])
    # standalone values: 0 at the zero vector, Inf at a nonzero entry, NaN at a NaN entry
    @test f(zeros(T, 2)) === zero(T) && f(T[]) === zero(T) && f(T[-0.0, 0.0]) === zero(T)
    @test f(T[0, 1]) === T(Inf) && f(T[0, floatmin(T) / 4]) === T(Inf)
    @test isnan(f(T[0, NaN])) && isnan(f(T[NaN, Inf]))
    @test f(view(zeros(T, 4), 1:2:4)) === zero(T)
    @test NormL1(T(Inf), T[1, 1])(zeros(T, 2)) === zero(T)
  end
  @test NormL1(Inf)(zeros(Float32, 2)) === 0f0 && NormL1(Inf32)(zeros(2)) === 0.0
end

@testset "NormL1: a finite level at the γ = Inf endpoint keeps its results" begin
  # the endpoint map is zero at every finite argument, but a NaN entry is not decided at γ = Inf
  # unless the entry is fixed by its own domain: a tilt forms NaN from x - γc by itself there
  for T in (Float32, Float64)
    nan = T(NaN)
    for λ in (T(1), floatmin(T), nextfloat(0.0), 1e300)
      f = NormL1(λ)
      check(f, T[NaN, 1, -3], T(Inf), T[nan, 0, 0]; value = nan)
      @test isequal(prox(f, T[Inf, NaN, -Inf], T(Inf)), fill(nan, 3))
    end
    check(NormL1(T(1), T[0, 1]), T[NaN, NaN], T(Inf), T[nan, nan])
    check(NormL1(T(1), T[0, 1]), T[-Inf, NaN], T(Inf), T[-Inf, nan])
    check(NormL1(T(0)), T[NaN, 1], T(Inf), T[nan, 1]; value = zero(T))
    check(NormL1(T(1)), T[NaN, 1], T[Inf, 1], T[nan, 0])
    check(Tilt(NormL1(T(1)), T[1, 1]), T[1, NaN], T(Inf), T[nan, nan])
  end
end

@testset "NormL1: unchanged cases" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    f = NormL1(T(Inf))
    # a zero step: the identity (the γ = 0 limit with an infinite level is a separate question)
    check(f, T[1, NaN], T(0), T[1, NaN])
    check(f, T[1, 2], T(0), T[1, 2])
    # steps that are not validated yet (#311) keep their results
    for γ in (T(-1), nan)
      @test isequal(prox(f, T[1, NaN], γ), T[nan, nan])
      @test isequal(prox(f, T[1, 2], γ), T[nan, nan])
    end
    @test isequal(prox(NormL1(T(Inf), T[Inf, 1]), T[1, 1], T(-1)), T[0, nan])
    @test isequal(prox(f, T[NaN, 1], T[0, -1]), T[nan, nan])
    # an infinite weight under a finite level at a finite step keeps NaN (a separate rule)
    for γ in (T(0), T(1))
      @test isequal(prox(NormL1(T(1), T[Inf, 1]), T[NaN, NaN], γ), T[nan, nan])
    end
    # finite parameters that overflow the working precision are not infinite (R3)
    @test isequal(prox(NormL1(1e300), Float32[1, NaN], 1f0), Float32[0, NaN])
    @test isequal(prox(NormL1(1e300), Float32[Inf], 1f0), Float32[NaN])
    @test isequal(prox(NormL1(1.0), Float32[NaN], 1e300), Float32[NaN])
    @test isequal(prox(NormL1(1e308), [NaN, 1.0], 10.0), [NaN, 0.0])
    # finite levels: soft thresholding, as before
    check(NormL1(T(1)), T[3, -3, 0.5, -0.0, NaN], T(1), T[2, -2, 0, 0, NaN])
    check(NormL1(T(0.5), T[2, 4, 0]), T[3, -3, 5], T(1), T[2, -1, 5]; value = T(4))
    check(NormL1(T(1)), T[floatmax(T), -floatmax(T)], T(1), T[floatmax(T), -floatmax(T)])
    check(NormL1(T(1)), T[NaN, 4], T[1, 2], T[NaN, 2])
  end
end

@testset "NormL1: iterates and wrappers" begin
  for T in (Float32, Float64)
    f = NormL1(T(Inf))
    x = SparseIterate(sparsevec(T[1, NaN, 0]))
    @test isequal(Vector(prox(f, x, T(1))), zeros(T, 3))
    @test isequal(value_and_prox!(x, f, x, T(1)), zero(T)) && isequal(Vector(x), zeros(T, 3))
    @test isequal(Matrix(prox(f, SymmetricSparseIterate(T[1 NaN; NaN 2]), T(1))), zeros(T, 2, 2))
    x = SparseIterate(sparsevec(T[1, NaN]))
    v = PB.prox_coordinate!(x, f, 2, T(1))
    @test isequal(v, zero(T)) && isequal(Vector(x), T[1, 0])
    check(Constrained(f, IndBox(T(-1), T(1))), T[1, NaN], T(1), T[0, 0])
    check(Constrained(f, IndBox(T(-1), T(1))), T[1, NaN], T[1, 2], T[0, 0])
    check(Constrained(f, IndBox(T(-1), T(1))), T[Inf, NaN], T(1), T[0, 0])
    check(Scaled(f, T(2)), T[1, NaN], T(1), T[0, 0])
    # a finite level scaled by Inf is not an infinite level here: the step γ⋅Inf keeps its result
    @test isequal(prox(Scaled(NormL1(T(1)), T(Inf)), T[1, NaN], T(1)), T[0, NaN])
    check(Translate(f, T[2, 3]), T[1, NaN], T(1), T[2, 3])
    check(Tilt(f, T[1, 1]), T[1, NaN], T(1), T[0, 0])
    # at γ = Inf the shifted argument x - γc is not finite, and the fixed entries are still zero
    check(Tilt(f, T[1, -1]), T[1, NaN], T(Inf), T[0, 0])
    check(Tilt(f, T[0, 1]), T[1, 2], T(Inf), T[0, 0])
    check(Precompose(f, T(2)), T[1, NaN], T(1), T[0, 0])
    check(Precompose(f, T[2, 3]), T[1, NaN], T(1), T[0, 0])
    s = SeparableSum(f, NormL1(T(1)))
    y = (zeros(T, 2), zeros(T, 2))
    @test isequal(value_and_prox!(y, s, (T[1, NaN], T[NaN, 3]), T(1)), T(NaN))
    @test isequal(y, (T[0, 0], T[NaN, 2]))
    # the affine constraint: the entry with an infinite level is fixed, the free one meets the constraint
    check(Constrained(NormL1(T(Inf), T[1, 0]), IndHyperplane(T[1, 1], T(1))), T[NaN, 0.5], T(1), T[0, 1])
  end
  # the ball and the sphere classify the entry with an infinite level as fixed, so a separate output
  # and the argument itself give the same result (O7); an infinite entry there gives zero, not ±r
  for (c, x, expected) in ((Constrained(NormL1(Inf), IndBallL2(1.0)), [1.0, 2.0, NaN], [0.0, 0.0, 0.0]),
                           (Constrained(NormL1(Inf), IndBallL2(1.0)), [Inf, 2.0, 0.5], [0.0, 0.0, 0.0]),
                           (Constrained(NormL1(Inf, [1.0, 0, 1]), IndBallL2(1.0)), [NaN, 2.0, 0.5], [0.0, 1.0, 0.0]),
                           (Constrained(NormL1(Inf, [1.0, 0, 1]), IndBallL2(1.0)), [0.3, 2.0, -Inf], [0.0, 1.0, 0.0]),
                           (Constrained(NormL1(Inf, [1.0, 0, 1]), IndSphereL2(1.0)), [1.0, 2.0, NaN], [0.0, 1.0, 0.0]),
                           (Constrained(NormL1(Inf, [1.0, 0, 1]), IndSphereL2(1.0)), [Inf, 2.0, 0.5], [0.0, 1.0, 0.0]))
    check(c, x, 1.0, expected; value = 0.0)
  end
  # a finite level that overflows to an infinite threshold stays a penalty: the limit ±r e_k
  y = zeros(2)
  @test value_and_prox!(y, Constrained(NormL1(1e308), IndBallL2(1.0)), [Inf, 2.0], 10.0) == 1e308 && y == [1.0, 0.0]
end

@testset "NormL1: device arrays agree with the CPU" begin
  for T in (Float32, Float64)
    for (fd, fc) in ((NormL1(T(Inf)), NormL1(T(Inf))),
                     (NormL1(T(Inf), JLArray(T[1, 0, 1])), NormL1(T(Inf), T[1, 0, 1])),
                     (NormL1(T(1), JLArray(T[1, 0, 1])), NormL1(T(1), T[1, 0, 1])),
                     (NormL1(T(1)), NormL1(T(1))), (NormL1(1e300), NormL1(1e300)))
      for x in (T[NaN, -Inf, 2], T[NaN, 1, -0.0]), γ in (T(1), T(Inf), floatmin(T), :array)
        γd, γc = γ === :array ? (JLArray(T[1, 2, 0.5]), T[1, 2, 0.5]) : (γ, γ)
        expected = prox(fc, x, γc)
        @test isequal(Array(prox(fd, JLArray(x), γd)), expected)
        if γ !== :array
          y = JLArray(zeros(T, 3))
          v = value_and_prox!(y, fd, JLArray(x), γd)
          @test isequal(Array(y), expected) && isequal(v, fc(expected))
        end
      end
    end
    @test isequal(Array(prox(NormL1(T(Inf)), JLArray(T[1, NaN]), T(1))), zeros(T, 2))
    @test NormL1(T(Inf))(JLArray(zeros(T, 2))) === zero(T)
    @test NormL1(Inf)(JLArray(zeros(T, 2))) === zero(T)
    @test isnan(NormL1(T(Inf))(JLArray(T[0, NaN])))
  end
  @test isequal(Array(prox(NormL1(1e300), JLArray(Float32[1, NaN]), 1f0)), Float32[0, NaN])
end

####################################
#  SLOPE
####################################

@testset "SLOPE: an infinite first weight at a positive finite step" begin
  for T in (Float32, Float64)
    f = SLOPE(T[Inf, 1, 0.5])
    xs = (T[1, NaN, 0.5], T[NaN, 1, 0.5], T[1, 0.5, NaN], T[NaN, NaN, 0.5], T[NaN, NaN, NaN],
          T[1, 1, NaN], T[-2, 2, NaN], T[NaN, -0.0, 0.0], T[-NaN, -1, 1], T[Inf, NaN, -Inf],
          T[floatmax(T), NaN, -floatmax(T)], T[1, -2, 0.5], T[-0.0, 0.0, -1], T[Inf, -1, 0.5])
    for x in xs, γ in (T(1), floatmin(T), T(1e30), 1e300, 1e-300)
      check(f, x, γ, signed_zeros(x); value = zero(T))
    end
    # every order of the entries, and other admitted weights with an infinite first one
    for x in (T[1, NaN, -0.5], T[1, -0.5, NaN], T[NaN, 1, -0.5], T[NaN, -0.5, 1], T[-0.5, 1, NaN], T[-0.5, NaN, 1])
      check(f, x, T(1), signed_zeros(x); value = zero(T))
    end
    for λ in (T[Inf, Inf, 1], T[Inf, 0.5, 0.5], T[Inf, 0, 0], T[Inf, Inf, Inf])
      check(SLOPE(λ), T[NaN, 1, -2], T(1), T[0, 0, -0.0]; value = zero(T))
    end
    check(SLOPE(T[Inf]), T[NaN], T(1), T[0]; value = zero(T))
    check(SLOPE(T[Inf]), T[-3], T(1), T[-0.0]; value = zero(T))
    check(SLOPE(T[Inf, Inf]), T[Inf, -Inf], T(1), T[0, -0.0]; value = zero(T))
    # long vectors, through the other sort and value branches
    for n in (2000, 5000)
      x = T.(sin.(1:n))
      x[n ÷ 2] = NaN
      check(SLOPE([T(Inf); ones(T, n - 1)]), x, T(1), signed_zeros(x); value = zero(T))
    end
    # standalone values
    @test f(zeros(T, 3)) === zero(T) && f(T[0, 1, 0]) === T(Inf) && isnan(f(T[0, NaN, 0]))
    # a narrower output
    y = zeros(Float32, 3)
    @test value_and_prox!(y, SLOPE([Inf, 1, 0.5]), [1.0, NaN, -3], 1.0) === 0f0 && isequal(y, Float32[0, 0, -0.0])
    @test allocs_vprox(zeros(T, 3), f, T[1, NaN, -1], T(1)) == 0
  end
end

@testset "SLOPE: unchanged cases" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    f = SLOPE(T[Inf, 1, 0.5])
    # a zero, an infinite and a rejected step keep their results (no named endpoint, #311)
    @test isequal(prox(f, T[1, NaN, 0.5], T(0)), T[nan, nan, 0.5])
    @test isequal(prox(f, T[1, 2, 0.5], T(0)), T[1, nan, 0.5])
    @test isequal(prox(f, T[1, NaN, 0.5], T(Inf)), T[0, nan, 0])
    @test isequal(prox(f, T[1, -2, 0.5], T(Inf)), T[0, -0.0, 0])
    @test isequal(prox(f, T[1, 2, 0.5], T(-1)), T[2, Inf, 1])
    @test isequal(prox(f, T[1, NaN, 0.5], T(-1)), T[Inf, nan, 1])
    @test isequal(prox(f, T[1, 2, 0.5], nan), fill(nan, 3))
    # finite weights: every output depends on every entry
    check(SLOPE(T[3, 2, 1]), T[3, NaN, 2], T(1), fill(nan, 3); value = nan)
    @test_throws DimensionMismatch prox(SLOPE(T[Inf, 1]), T[NaN, 1, 2], T(1))
  end
  # a finite first weight that overflows Float32 is not infinite (R3)
  @test isequal(prox(SLOPE([1e300, 1, 0.5]), Float32[1, NaN, 0.5], 1f0), fill(NaN32, 3))
end

####################################
#  NormL1Fused
####################################

@testset "NormL1Fused: an infinite ℓ1 level at a positive finite step" begin
  for T in (Float32, Float64), λ2 in (T(0), T(1), T(Inf))
    f = NormL1Fused(T(Inf), λ2)
    xs = (T[NaN], T[2], T[NaN, 1], T[3, 1], T[1, NaN, 0.5], T[NaN, 1, 2, 3], T[1, 2, 3, NaN],
          T[NaN, NaN, 2, 3], T[NaN, NaN, NaN], T[Inf, NaN, -Inf, 1], T[-1, -0.0, 0.0, 2], T[1, -2, 0.5])
    for x in xs, γ in (T(1), floatmin(T), 1e-300, 1e300)
      check(f, x, γ, zeros(T, length(x)); value = zero(T))
    end
    x = T.(cos.(1:5000))
    x[17] = NaN
    check(f, x, T(1), zeros(T, 5000); value = zero(T))
    # the pair of scalars and the tuple form
    @test prox(f, (T(NaN), T(1)), T(1)) === (zero(T), zero(T))
    @test prox(f, (T(-1), T(-0.0)), T(1)) === (zero(T), zero(T))
    @test prox(f, (T(Inf), T(-Inf)), T(1)) === (zero(T), zero(T))
    xt = (T[NaN, 1, 3], T[2, NaN, -Inf])
    @test isequal(prox(f, xt, T(1)), (zeros(T, 3), zeros(T, 3)))
    y = (fill(T(-7), 3), fill(T(-7), 3))
    @test value_and_prox!(y, f, xt, T(1)) === zero(T) && isequal(y, (zeros(T, 3), zeros(T, 3)))
    a = (copy(xt[1]), copy(xt[2]))
    @test value_and_prox!(a, f, a, T(1)) === zero(T) && isequal(a, (zeros(T, 3), zeros(T, 3)))
    # standalone values: 0 at zero, Inf at a nonzero argument, NaN at a NaN entry
    for n in 0:3
      @test f(zeros(T, n)) === zero(T)
    end
    @test f((zero(T), zero(T))) === zero(T) && f((zeros(T, 2), zeros(T, 2))) === zero(T)
    @test f(T[0, 1, 0]) === T(Inf) && f(T[1, 1]) === T(Inf) && f((zeros(T, 2), T[0, 1])) === T(Inf)
    @test isnan(f(T[0, NaN, 0])) && isnan(f((zero(T), T(NaN)))) && isnan(f((zeros(T, 2), T[0, NaN])))
    @test allocs_vprox(zeros(T, 3), f, T[1, NaN, 2], T(1)) == 0
  end
  # a step that is positive as given, also where it underflows the argument's precision
  check(NormL1Fused(Inf, 1.0), Float32[1, 2, 0.5], 1e-300, zeros(Float32, 3); value = 0f0)
end

@testset "NormL1Fused: unchanged cases" begin
  for T in (Float32, Float64)
    nan = T(NaN)
    f = NormL1Fused(T(Inf), T(1))
    # a zero, an infinite and a rejected step keep their results (no named endpoint, #311)
    for γ in (T(0), T(Inf), T(-1), nan)
      @test isequal(prox(f, T[1, NaN, 0.5], γ), fill(nan, 3))
      @test isequal(prox(f, T[1, -2, 0.5], γ), fill(nan, 3))
      @test isequal(prox(f, T[NaN, 1], γ), fill(nan, 2))
      @test isequal(prox(f, (nan, T(1)), γ), (nan, nan))
      @test isequal(prox(f, (T[NaN, 1], T[2, NaN]), γ), (fill(nan, 2), fill(nan, 2)))
    end
    # an infinite fusion level alone restricts the domain to equal entries: not a constant map
    g = NormL1Fused(T(1), T(Inf))
    @test isequal(prox(g, T[1, -2, 0.5], T(1)), T[-Inf, Inf, nan])
    @test isequal(prox(g, T[3, 1], T(1)), T[1, 1])
    @test isequal(prox(g, T[1, NaN, 0.5], T(1)), T[Inf, nan, -Inf])
  end
  # a finite ℓ1 level that overflows Float32 is not infinite (R3)
  f = NormL1Fused(1e300, 1.0)
  @test isequal(prox(f, Float32[1, NaN, 0.5], 1f0), fill(NaN32, 3))
  @test isequal(prox(f, Float32[NaN, 1], 1f0), fill(NaN32, 2))
  @test isequal(prox(f, (NaN32, 1f0), 1f0), (NaN32, NaN32))
end

@testset "FiberSum of NormL1Fused" begin
  for T in (Float32, Float64), K in (2, 3)
    fo = NormL1Fused(T(Inf), T(1))
    fin = NormL1Fused(T(0.3), T(0.4))
    X = reshape(T.(1:4K) ./ 3, 2, 2, K)
    X[1, 2, 1] = NaN
    X[2, 2, 1] = NaN
    for (diagonal, dexp) in ((fo, :zero), (nothing, :identity), (NormL1Fused(T(Inf), T(0.5)), :zero), (fin, :fin))
      fb = FiberSum(fo; diagonal)
      Y = prox(fb, X, T(1))
      Z = similar(X)
      v = value_and_prox!(Z, fb, X, T(1))
      @test isequal(Z, Y) && isequal(v, fb(Y))
      W = copy(X)
      @test isequal(value_and_prox!(W, fb, W, T(1)), v) && isequal(W, Y)
      for j in 1:2, i in 1:2
        fiber = Y[i, j, :]
        if i != j || dexp === :zero
          @test isequal(fiber, zeros(T, K))
        elseif dexp === :identity
          @test isequal(fiber, X[i, i, :])
        else
          @test isequal(fiber, prox(fin, X[i, i, :], T(1)))
        end
      end
      dexp === :fin ? @test(isnan(v)) : @test(v === zero(T))
    end
    @test FiberSum(fo)(zeros(T, 2, 2, K)) === zero(T)
    # a finite off-diagonal function with an infinite diagonal level: only the diagonal is fixed
    fb = FiberSum(fin; diagonal = fo)
    X = ones(T, 2, 2, K)
    X[1, 1, 1] = NaN
    Y = prox(fb, X, T(1))
    @test isequal(Y[1, 1, :], zeros(T, K)) && isequal(Y[2, 2, :], zeros(T, K))
    @test isequal(Y[1, 2, :], prox(fin, X[1, 2, :], T(1))) && isequal(Y[2, 1, :], prox(fin, X[2, 1, :], T(1)))
    Z = similar(X)
    @test value_and_prox!(Z, fb, X, T(1)) ≈ fb(Y) && isequal(Z, Y)
    # symmetric fibers
    fb = FiberSum(fo; symmetric = true)
    X = ones(T, 2, 2, K)
    X[2, 1, 1] = X[1, 2, 1] = NaN
    Z = similar(X)
    @test value_and_prox!(Z, fb, X, T(1)) === zero(T) && isequal(Z, zeros(T, 2, 2, K))
  end
end

end
