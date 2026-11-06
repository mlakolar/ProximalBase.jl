module ScaledInfiniteLevelUnderflowTest

# A wrapper step that is not a normal number keeps the hard domain of an infinite level (decision 49,
# R3). `Scaled(f, α)` at a positive finite step γ whose product αγ underflows to zero, is subnormal or
# overflows reads f at the product αγ. An infinite level (a stored Inf, not a finite level that
# overflows the working precision) stays infinite under a positive finite scale, so the proximal map
# is the constant map of the domain {0}, and not the identity (γ = 0) or the NaN of Inf * 0 that a
# product rounded to zero gave. Scalar steps, array steps, chains of Scaled, prox_entry and
# prox_coordinate! take the same route; zero, infinite, negative and NaN steps, finite levels and the
# values of scaled functions keep their results.
# Outputs are compared with isequal, so a stored zero is +0.0 and NaN equals NaN. Expected outputs are
# the map of the unscaled function at step 1, which is the same constant map at every positive step
# (decision 49), literal zeros and identities written here, or BigFloat references.

using Test, ProximalBase, Random

const PB = ProximalBase

# prox, then prox! and value_and_prox! into a fresh output, an output holding a sentinel, and the
# argument itself (a copy); every entry with isequal, and the value against the value at the stored
# output (O2). A value that is `nothing` is not pinned.
function check(f, x, γ, expected; value = zero(float(eltype(x))))
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
      # a value that is pinned is the value at the stored output
      value === nothing || @test isequal(v1, f(fresh)) && isequal(v3, f(alias))
      value === nothing || @test v1 === value && v3 === value
    end
  end
end

# the same result, or the same error, of two calls
function same_outcome(a, b)
  ra = try a() catch e; e end
  rb = try b() catch e; e end
  ra isa Exception || rb isa Exception ? (ra isa Exception && rb isa Exception && typeof(ra) === typeof(rb)) : isequal(ra, rb)
end

# the leaves with an infinite level, the argument [1, 2, 3], and whether the value is pinned at 0
# (NormL2(Inf) has the value NaN at the zero vector, which is not part of this rule)
function leaves(::Type{T}) where {T}
  inf = T(Inf)
  groups = [1:2, 3:3]
  [("NormL1", NormL1(inf), true),
   ("NormL1 weighted", NormL1(inf, T[1, 1, 0]), true),
   ("NormL1Nonneg", NormL1Nonneg(inf), true),
   ("NormLinf", NormLinf(inf), true),
   ("NormL2", NormL2(inf), false),
   ("SLOPE", SLOPE(T[inf, 1, 1]), true),
   ("NormL1Fused", NormL1Fused(inf, T(1)), true),
   ("NormL1Fused, zero fusion", NormL1Fused(inf, T(0)), true),
   ("ElasticNet, Inf l1", ElasticNet(inf, T(1)), true),
   ("ElasticNet, Inf l2", ElasticNet(T(1), inf), true),
   ("GroupNormL2", GroupNormL2(inf, groups), true),
   ("GroupNormLinf", GroupNormLinf(inf, groups), true),
   ("SparseGroupNormL2", SparseGroupNormL2(inf, T(1), groups), true),
   ("SparseGroupNormL2, Inf group level", SparseGroupNormL2(T(1), inf, groups), true)]
end

# products of a scale and a step that are not normal numbers (the first four) and that overflow
products(::Type{Float64}) = [(1e-300, 1e-300), (1e-200, 1e-200), (1e-160, 1e-150), (1e-320, 1.0), (1e-300, 1e-5)]
products(::Type{Float32}) = [(1f-30, 1f-30), (1f-20, 1f-20), (1f-30, 1f-10), (1f-40, 1f0), (1f-30, 1f-5)]
overflows(::Type{Float64}) = [(1e300, 1e300), (floatmax(Float64), 2.0), (2.0, floatmax(Float64)), (1e200, 1e200)]
overflows(::Type{Float32}) = [(1f30, 1f30), (floatmax(Float32), 2f0), (2f0, floatmax(Float32)), (1f20, 1f20)]

####################################
#  the reproduction table: these rows fail without the fix
####################################

@testset "the rows of the issue" begin
  x = [1.0, 2.0, 3.0]
  z = zeros(3)
  # a scale and a step that underflow together, and a chain of scales
  check(Scaled(NormL1(Inf), 1e-300), x, 1e-300, z)
  check(Scaled(NormL1(Inf), 1e-200), x, 1e-200, z)
  check(Scaled(Scaled(NormL1(Inf), 1e-200), 1e-200), x, 1.0, z)
  check(Scaled(Scaled(NormL1(Inf), 1e-200), 1e-200), x, 1e-200, z)
  check(Scaled(Scaled(Scaled(NormL1(Inf), 1e-100), 1e-100), 1e-100), x, 1e-100, z)
  check(Scaled(Scaled(NormL1(Inf), 1e200), 1e-300), x, 1e-300, z)
  check(Scaled(NormL1(Inf), 1e-300), x, [1e-300, 1.0, 1e-300], z)
  check(Scaled(NormL1(Inf), 1e-300), Float32[1, 2, 3], 1f-30, zeros(Float32, 3))
  check(Scaled(NormL1(Inf32), 1f-30), Float32[1, 2, 3], 1f-30, zeros(Float32, 3))
  check(Scaled(NormL1(Inf), 1e-300), Float32[1, 2, 3], 1f-30, zeros(Float32, 3))
  check(Scaled(NormL1(Inf), 1f-30), [1.0, 2.0, 3.0], 1e-300, z)
  check(Scaled(SLOPE([Inf, 1.0, 1.0]), 1e-300), x, 1e-300, z)
  check(Scaled(NormL1Fused(Inf, 1.0), 1e-300), x, 1e-300, z)
  check(Scaled(NormLinf(Inf), 1e-300), x, 1e-300, z)
  # an overflowing product takes the same route
  check(Scaled(NormL1(Inf), floatmax(Float64)), x, 2.0, z)
  check(Scaled(NormL1(Inf), 1e300), x, 1e300, z)
  # the value of value_and_prox! is the value at the output: zero, not the Inf of the identity output
  y = similar(x)
  @test value_and_prox!(y, Scaled(NormL1(Inf), 1e-300), x, 1e-300) === 0.0 && y == z
  # regression guards: these rows gave zeros before
  check(Scaled(NormL1(Inf), 1e-300), x, 1.0, z)
  check(Scaled(NormL1(Inf), 1.0), x, 1e-300, z)
  check(Scaled(NormL1(Inf), 1e-300), x, 1e-5, z)
  check(NormL1(Inf), x, 1e-300, z)
  check(Scaled(NormL2(Inf), 1e-300), x, 1e-300, z; value = nothing)
  # zero scaled to the exact zero step keeps the identity (γ = 0)
  @test prox(Scaled(NormL1(Inf), 1e-300), x, 0.0) == x
  @test prox(Scaled(Scaled(NormL1(Inf), 1e-200), 1e-200), x, 0.0) == x
  @test prox(Scaled(NormL1(Inf), 1e-300), x, [0.0, 0.0, 0.0]) == x
  @test prox(Scaled(NormL1(Inf), 1e-300), x, [0.0, 1e-300, 1.0]) == [1.0, 0.0, 0.0]
end

####################################
#  every family with an infinite level, over the classes of products
####################################

@testset "$name, $T: products that are not normal numbers" for T in (Float32, Float64), (name, leaf, pinned) in leaves(T)
  x = T[1, 2, 3]
  ref = prox(leaf, x, one(T))
  value = pinned ? zero(T) : nothing
  for (α, γ) in vcat(products(T), overflows(T))
    # Fused with a finite fusion level at an overflowing product is not a constant of this rule
    # (the finite level overflows the leaf); the one with a zero fusion level is
    name == "NormL1Fused" && α * γ > 1 && continue
    s = Scaled(leaf, α)
    check(s, x, γ, ref; value)
  end
  # a Float64 scale on a Float32 argument, and the product in the step type
  if T === Float32
    check(Scaled(leaf, 1e-300), x, 1f-30, ref; value)
    check(Scaled(leaf, 1f-30), x, 1e-300, ref; value)
  end
  # chains of scales, with the step on either side
  check(Scaled(Scaled(leaf, T(1e-20)), T(1e-20)), x, T(1e-20), ref; value)
  check(Scaled(Scaled(leaf, T(1e-20)), T(1e-20)), x, one(T), ref; value)
end

@testset "NaN and infinite entries of the argument at an underflowing step" for T in (Float32, Float64)
  for leaf in (NormL1(T(Inf)), NormLinf(T(Inf)), SLOPE(T[Inf, 1, 1, 1]), NormL1Fused(T(Inf), T(1)))
    for x in (T[NaN, 1, 2, 3], T[1, NaN, -2, 3], T[Inf, 1, 2, 3], T[-Inf, 2, NaN, 3], T[NaN, NaN, NaN, NaN])
      leaf isa NormL1Fused && any(isinf, x) && continue
      ref = prox(leaf, x, one(T))
      @test all(iszero, ref)
      for (α, γ) in products(T)
        s = Scaled(leaf, α)
        @test isequal(prox(s, x, γ), ref)
        y = similar(x)
        @test value_and_prox!(y, s, x, γ) === zero(T)
        @test isequal(y, ref)
      end
    end
  end
end

####################################
#  array steps, prox_entry and the coordinate updates
####################################

@testset "array steps, $T" for T in (Float32, Float64)
  x = T[1, 2, 3]
  inf = T(Inf)
  tiny = T === Float64 ? 1e-300 : 1f-30
  for leaf in (NormL1(inf), NormL1Nonneg(inf), ElasticNet(inf, T(1)), ElasticNet(T(1), inf), NormL1(inf, T[1, 0, 1]))
    ref = prox(leaf, x, ones(T, 3))
    for γs in (fill(T(tiny), 3), T[tiny, 1, tiny], T[tiny, tiny / 2, 1], fill(T(1), 3), fill(T(1e-5), 3), T[tiny, T(1e5), tiny])
      check(Scaled(leaf, tiny), x, γs, ref)
      check(Scaled(Scaled(leaf, tiny), T(tiny)), x, γs, ref)
    end
    # an exact zero entry of the step is the result of the function at that step
    check(Scaled(leaf, tiny), x, T[0, tiny, tiny], prox(leaf, x, T[0, 1, 1]); value = nothing)
  end
  # the weights decide which entries are fixed
  leaf = NormL1(inf, T[1, 0, 1])
  @test isequal(prox(Scaled(leaf, tiny), x, fill(T(tiny), 3)), T[0, 2, 0])
  # entry by entry
  s = Scaled(NormL1(inf), tiny)
  for v in T[2, -3, 0, NaN, Inf, -Inf], i in 1:3
    @test isequal(prox_entry(s, v, i, T(tiny)), zero(T))
  end
  sw = Scaled(NormL1(inf, T[1, 0, 1]), tiny)
  @test isequal(prox_entry(sw, T(2), 1, T(tiny)), zero(T))
  @test isequal(prox_entry(sw, T(2), 2, T(tiny)), T(2))
  @test isequal(prox_entry(Scaled(Scaled(NormL1(inf), tiny), tiny), T(5), 1, T(1)), zero(T))
  @test isequal(prox_entry(s, T(5), 1, T(0)), T(5))
end

@testset "coordinate updates of an AtomIterate, $T" for T in (Float32, Float64)
  tiny = T === Float64 ? 1e-300 : 1f-30
  for leaf in (GroupNormL2(T(Inf), [1:2, 3:3]), GroupNormLinf(T(Inf), [1:2, 3:3]), SparseGroupNormL2(T(Inf), T(1), [1:2, 3:3]))
    for (k, expected) in ((1, T[0, 0, 3]), (2, T[1, 2, 0]))
      for s in (Scaled(leaf, tiny), Scaled(Scaled(leaf, tiny), tiny))
        a = AtomIterate(T[1, 2, 3], leaf)
        prox_coordinate!(a, s, k, T(tiny))
        @test isequal(collect(a), expected)
        # the exact zero step is that of the function itself
        b = AtomIterate(T[1, 2, 3], leaf)
        prox_coordinate!(b, s, k, zero(T))
        c = AtomIterate(T[1, 2, 3], leaf)
        prox_coordinate!(c, leaf, k, zero(T))
        @test isequal(collect(b), collect(c))
      end
    end
  end
end

####################################
#  wrappers around the function
####################################

@testset "Translate, SeparableSum and nested wrappers" begin
  x = [1.0, 2.0, 3.0]
  b = [0.5, -1.0, 2.0]
  # the translate of a constant map is the shift
  t = Translate(NormL1(Inf), b)
  check(Scaled(t, 1e-300), x, 1e-300, b)
  check(Scaled(Scaled(t, 1e-200), 1e-200), x, 1.0, b)
  check(Translate(Scaled(NormL1(Inf), 1e-300), b), x, 1e-300, b)
  check(Scaled(Translate(SLOPE([Inf, 1.0, 1.0]), b), 1e-300), x, 1e-300, b)
  check(Scaled(Translate(NormL1Fused(Inf, 1.0), b), 1e-300), x, 1e-300, b)
  # blocks of different kinds: an infinite level is fixed at 0, a finite level is the identity at a step
  # whose product with the level falls below the range, and an indicator keeps its own step
  arg = ([1.0, 2.0], [3.0, 4.0])
  for (blocks, expected) in (((NormL1(Inf), NormL1(Inf)), ([0.0, 0.0], [0.0, 0.0])),
                             ((NormL1(Inf), NormL2(Inf)), ([0.0, 0.0], [0.0, 0.0])),
                             ((NormL1(Inf), NormL1(1.0)), ([0.0, 0.0], [3.0, 4.0])),
                             ((NormL1(1.0), NormL1(Inf)), ([1.0, 2.0], [0.0, 0.0])),
                             ((NormL1(Inf), IndNonnegative()), ([0.0, 0.0], [3.0, 4.0])),
                             ((SLOPE([Inf, 1.0]), NormL1(2.0)), ([0.0, 0.0], [3.0, 4.0])))
    sep = SeparableSum(blocks...)
    for (α, γ) in ((1e-300, 1e-300), (1e-200, 1e-200), (1e-160, 1e-150))
      for s in (Scaled(sep, α), Scaled(Scaled(sep, α), 1.0))
        @test isequal(prox(s, arg, γ), expected)
      end
    end
  end
  # an indicator keeps its own step
  @test isequal(prox(Scaled(SeparableSum(NormL1(Inf), IndNonnegative()), 1e-300), ([1.0, 2.0], [-3.0, 4.0]), 1e-300),
                ([0.0, 0.0], [0.0, 4.0]))
end

####################################
#  steps and levels that keep their results
####################################

@testset "excluded steps keep their results" begin
  x = [1.0, 2.0, 3.0]
  for leaf in (NormL1(Inf), NormLinf(Inf), SLOPE([Inf, 1.0, 1.0]), NormL1Fused(Inf, 1.0), NormL2(Inf),
               ElasticNet(Inf, 1.0), GroupNormL2(Inf, [1:2, 3:3]))
    for (α, γ) in ((1e-300, Inf), (1e-300, NaN), (1e-300, -1e-300), (1e-300, -1.0), (1e300, -1e300), (1e-300, 0.0),
                   (Inf, 1e-300), (Inf, 1e300), (Inf, Inf))
      @test same_outcome(() -> prox(Scaled(leaf, α), x, γ), () -> prox(leaf, x, α * γ))
      @test same_outcome(() -> (y = similar(x); value_and_prox!(y, Scaled(leaf, α), x, γ); y),
                         () -> (y = similar(x); value_and_prox!(y, leaf, x, α * γ); y))
    end
  end
  # an exact zero step is the identity of the infinite level (#665 owns the choice)
  @test prox(Scaled(NormL1(Inf), 1e-300), x, 0.0) == x
  # an infinite factor of an indicator or of a penalty is not a scale to fold
  @test same_outcome(() -> prox(Scaled(NormL1(Inf), Inf), x, 1e-300), () -> prox(NormL1(Inf), x, Inf * 1e-300))
  @test same_outcome(() -> prox(Scaled(NormL1(Inf), Inf), x, 0.0), () -> prox(NormL1(Inf), x, Inf * 0.0))
end

@testset "values of scaled functions do not change" begin
  x = [1.0, -2.0, 3.0]
  z = zeros(3)
  for (leaf, f0) in ((NormL1(Inf), 0.0), (NormLinf(Inf), 0.0), (SLOPE([Inf, 1.0, 1.0]), 0.0), (NormL1Fused(Inf, 1.0), 0.0))
    for s in (Scaled(leaf, 1e-300), Scaled(Scaled(leaf, 1e-200), 1e-200), Scaled(leaf, 1e300), Scaled(Scaled(leaf, 1e200), 1e200))
      @test s(x) === Inf
      @test s(z) === f0
    end
  end
  # finite levels: the value is the scaled value, in range or not, before and after the step route
  for λ in (0.5, 4.0, 1e300)
    f = NormL1(λ)
    for α in (1e-300, 1e-200, 1e-160, 1e300)
      @test isequal(Scaled(f, α)(x), α * f(x)) || isapprox(Scaled(f, α)(x), α * f(x); rtol = 4eps())
      s = Scaled(f, α)
      y = similar(x)
      v = value_and_prox!(y, s, x, 1e-300)
      @test isequal(v, s(y))
    end
  end
end

####################################
#  finite levels keep their results, and a finite level is never read as infinite (R3)
####################################

# the proximal map of the scaled function in BigFloat, where αγ does not leave the range
function bigref(leaf, x, α, γ)
  step = big(α) * big(γ)
  bx = big.(x)
  if leaf isa NormL1
    λ = big(leaf.λ0)
    return sign.(bx) .* max.(abs.(bx) .- λ * step, 0)
  elseif leaf isa SLOPE
    return prox(SLOPE(big.(leaf.λ)), bx, step)
  else
    return prox(NormL1Fused(big(leaf.λ1), big(leaf.λ2)), bx, step)
  end
end
# the entries agree to a few units in the last place of the argument (a threshold is subtracted from it)
near(y, r, x, ::Type{T}) where {T} = all(abs.(y .- T.(r)) .<= 4eps(T) .* max.(abs.(x), abs.(T.(r))))

@testset "finite levels, $T" for T in (Float32, Float64)
  rng = MersenneTwister(666)
  scales = T === Float64 ? [1e-310, 1e-300, 1e-200, 1e-160, 1e-100, 1e-5, 1.0, 3.0, 1e100, 1e200, 1e300] :
                           T[1f-44, 1f-38, 1f-30, 1f-20, 1f-10, 1f-5, 1f0, 3f0, 1f10, 1f20, 1f30]
  big_level = T === Float64 ? T(1e300) : T(1e30)      # finite, and its product with a tiny step is not
  levels = T[0, 0.3, 4, big_level]
  checked = 0
  for λ in levels, α in scales, γ in scales
    p = big(α) * big(γ)
    n = 6
    x = T.(3 .* randn(rng, n))
    # the product of the level and the step must be representable for a reference to compare with
    big(λ) * p < big(floatmax(T)) || continue
    for leaf in (NormL1(λ), SLOPE(sort(T.(abs.(randn(rng, n))) .* λ, rev = true)), NormL1Fused(λ, T(0.3)))
      # SLOPE and NormL1Fused are compared where the threshold is moderate; their leaves are not
      # accurate at thresholds that dwarf the argument, with or without a scale
      leaf isa NormL1 || big(max(λ, 1)) * p < 100 || continue
      r = bigref(leaf, x, α, γ)
      @test near(prox(Scaled(leaf, α), x, γ), r, x, T)
      @test near(prox(Scaled(Scaled(leaf, α), one(T)), x, γ), r, x, T)
      checked += 1
    end
  end
  @test checked > 100
  # R3: a finite level that is huge, under a tiny step, is the finite level: soft thresholding by a
  # threshold that is far below the entries leaves them as they are, and is not the zero map
  tiny = T === Float64 ? 1e-300 : T(1f-30)
  x = T[1, 2, 3]
  for leaf in (NormL1(big_level), SLOPE(T[big_level, 1, 1]), NormL1Fused(big_level, T(1)), NormLinf(big_level))
    @test prox(Scaled(leaf, tiny), x, tiny) == x
    @test prox(Scaled(Scaled(leaf, tiny), one(T)), x, tiny) == x
  end
  @test prox(Scaled(NormL1(big_level), tiny), x, T[tiny, tiny / 2, tiny]) == x
  @test prox(Scaled(NormL1(big_level), tiny), x, fill(tiny, 3)) == x
  # the finite level that overflows only in the product is a finite threshold: a huge one, not a hard domain
  @test prox(Scaled(NormL1(T(1)), floatmax(T)), T[1, 2, 3], T(2)) == zeros(T, 3)
  @test isequal(prox(Scaled(NormL1(T(1), T[0, 1, 1]), floatmax(T)), T[1, 2, 3], T(2)), T[1, 0, 0])
end

####################################
#  a zero weight leaves entries free: the finite level still acts on them
####################################

# With a zero weight an entry is free, and the finite level (a ridge, or the ℓ1 level of a group that has
# weight zero) acts on it. On an array step it is kept when the scale folds into it and the fold is
# otherwise declined unless the infinite level fixes every entry; on a scalar step it is folded, or
# discarded when its whole coefficient is below floatmin. The value is the value at the stored output (O2).
@testset "zero weights, $T" for T in (Float32, Float64)
  inf = T(Inf)
  α = T === Float64 ? T(1e-310) : T(1f-40)           # subnormal in its type
  hi = T === Float64 ? T(1e300) : T(1f30)            # αγ = 1e-10 for γ = hi
  x = T[1, 2]
  # the reference: entry 1 is free, entry 2 is fixed at 0
  ridge(γ1) = isinf(γ1) ? zero(T) : T(big(x[1]) / (1 + big(γ1) * big(α) * 1))
  soft(γ1) = isinf(γ1) ? zero(T) : T(max(big(x[1]) - big(γ1) * big(α), 0))
  for (name, f, ref) in (("ElasticNet", ElasticNet(inf, T(1), T[0, 1]), ridge),
                         ("SparseGroupNormL2", SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[0, 1]), soft))
    s = Scaled(f, α)
    for γ in (T[hi, hi], T[inf, hi], T[hi, inf], T[hi / 2, hi / 2], T[hi, T(1)])
      expected = T[ref(γ[1]), 0]
      y = zeros(T, 2)
      v = value_and_prox!(y, s, x, γ)
      @test all(abs.(y .- expected) .<= 2eps(T) .* abs.(expected))
      @test isequal(v, s(y))
      isinf(γ[1]) || @test v > 0          # the finite penalty of the free entry is not removed
      a = copy(x)
      va = value_and_prox!(a, s, a, γ)
      @test isequal(a, y) && isequal(va, v)
      sentinel = fill(T(-7), 2)
      prox!(sentinel, s, x, γ)
      @test isequal(sentinel, y) && isequal(prox(s, x, γ), y)
    end
    # A scalar step folds the whole exponent of the product into the levels, and the fixed entry is
    # zero whatever the weights of the other entries are; the free entry keeps its finite penalty
    # (its threshold is below floatmin and the value at the output is the scaled value of the
    # function, a little above zero)
    for (αs, γs) in ((T === Float64 ? (1e-160, 1e-160) : (1f-20, 1f-20)), (T === Float64 ? (1e-300, 1e-300) : (1f-30, 1f-30)))
      s2 = Scaled(f, αs)
      y = zeros(T, 2)
      v = value_and_prox!(y, s2, x, γs)
      @test isequal(y, T[1, 0])
      @test isequal(v, s2(y))
      name == "SparseGroupNormL2" && @test v ≈ αs rtol = 8eps(T)      # λ1 |y1| = 1 at the scale αs
      a = copy(x)
      va = value_and_prox!(a, s2, a, γs)
      @test isequal(a, y) && isequal(va, v)
      sentinel = fill(T(-7), 2)
      prox!(sentinel, s2, x, γs)
      @test isequal(sentinel, y) && isequal(prox(s2, x, γs), y)
      # a chain of factors takes the route of the scalar step
      @test isequal(prox(Scaled(Scaled(f, αs), one(T)), x, γs), y)
    end
  end
  # the same call for the group function on an AtomIterate, a coordinate at a time
  let f = SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[0, 1]), tiny = T === Float64 ? T(1e-300) : T(1f-30)
    s = Scaled(f, tiny)
    for (k, expected) in ((1, T[1, 2]), (2, T[1, 0]))
      a = AtomIterate(T[1, 2], f)
      prox_coordinate!(a, s, k, tiny)
      @test isequal(collect(a), expected)
    end
  end
  # an entry of the free group of subnormal magnitude: the finite level below the range that is dropped
  # (a threshold of 1e-400, or 1e-60 in Float32) is far below half the spacing of the subnormal numbers,
  # so the entry equals the exact result, the BigFloat one rounded to the type, and differs from the
  # unrounded one by less than floatmin
  let tiny = T === Float64 ? T(1e-200) : T(1f-30), x1 = T === Float64 ? T(1e-310) : T(1f-40)
    for f in (SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[0, 1]), ElasticNet(inf, T(1), T[0, 1]))
      y = zeros(T, 2)
      value_and_prox!(y, Scaled(f, tiny), T[x1, 2], tiny)
      exact = f isa ElasticNet ? big(x1) / (1 + big(tiny) * big(tiny)) : big(x1) - big(tiny) * big(tiny)
      @test y[1] == T(exact) && y[2] == 0
      @test abs(big(y[1]) - exact) < big(floatmin(T))
    end
  end
  # an infinite level with positive weights still drops it: the same functions without the zero weight
  check(Scaled(ElasticNet(inf, T(1), T[1, 1]), α), x, T[hi, hi], zeros(T, 2); value = nothing)
  check(Scaled(SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[1, 1]), α), x, T[hi, hi], zeros(T, 2); value = nothing)
end

####################################
#  a block that is itself scaled
####################################

@testset "Scaled blocks of a sum" begin
  arg = ([1.0, 2.0], [3.0, 4.0])
  z = ([0.0, 0.0], [0.0, 0.0])
  # both blocks have the domain {0}; the first block is a scaled SLOPE that the step fold reaches through its factor
  for f in (Scaled(SeparableSum(Scaled(SLOPE([Inf, 1.0]), 2.0), NormL1(Inf)), 1e-300),
            Scaled(SeparableSum(Scaled(Scaled(SLOPE([Inf, 1.0]), 2.0), 3.0), NormL1(Inf)), 1e-300),
            Scaled(Scaled(SeparableSum(Scaled(Scaled(SLOPE([Inf, 1.0]), 1e-100), 1e100), Scaled(NormL1(Inf), 5.0)), 1e-200), 1e-100),
            Scaled(SeparableSum(Scaled(NormL1Fused(Inf, 1.0), 2.0), NormL1(Inf)), 1e-300),
            Scaled(SeparableSum(Scaled(Scaled(NormL1Fused(Inf, 1.0), 2.0), 0.5), Scaled(SLOPE([Inf, 1.0]), 3.0)), 1e-300),
            Scaled(SeparableSum(Translate(Scaled(SLOPE([Inf, 1.0]), 2.0), [0.0, 0.0]), NormL1(Inf)), 1e-300))
    @test isequal(prox(f, arg, 1e-300), z)
    @test isequal(prox(f, arg, 1e-160), z)
  end
  # a finite block inside a scaled block keeps its meaning: the identity at a step this small
  f = Scaled(SeparableSum(Scaled(NormL1(1.0), 2.0), Scaled(NormL1(Inf), 3.0)), 1e-300)
  @test isequal(prox(f, arg, 1e-300), ([1.0, 2.0], [0.0, 0.0]))
end

####################################
#  a finite block next to an infinite one
####################################

# The block with the infinite level and the finite block of a sum share the exponent of the chain: the
# product of the scale and the step, 1e-360, is below the range, and the finite block is folded as every
# function in a chain is. The levels of the SLOPE become 1e-60 and 5e-61, the thresholds 1e-260 and
# 5e-261, which are above the subnormal entries 1e-310, so their proximal point is zero. Without the fold
# the product rounds to the zero step and the entries come back unchanged.
@testset "a finite sibling of an infinite block is folded with it" begin
  f = Scaled(SeparableSum(NormL1(Inf), SLOPE([1e100, 5e99])), 1e-160)
  @test isequal(prox(f, ([1.0, 2.0], [1e-310, -1e-310]), 1e-200), ([0.0, 0.0], [0.0, -0.0]))
end

####################################
#  levels and weights: references in BigFloat, formed here, not by the package
####################################

# What a finite level does once an infinite level is present (see `_HardScale` in src/calculus.jl):
# irrelevant (the infinite level fixes every coordinate it touches): removed; relevant on the free
# coordinates (a zero weight): kept, or discarded only when its whole coefficient is below floatmin, which
# changes an entry by less than floatmin (soft threshold) or by a relative amount below floatmin (ridge).
# The references below are the exact proximal maps in BigFloat; entries are compared to the rounded
# reference with 4 units in the last place of the larger of the argument and the result, plus floatmin.
bsoft(v, t) = iszero(t) ? v : (isinf(t) ? zero(v) : sign(v) * max(abs(v) - t, 0))
bcoef(λ, w) = iszero(w) ? big(0) : (isinf(λ) ? big(Inf) : big(λ) * big(w))     # λ w with 0 Inf = 0 (a free entry)
# the elastic net with the steps t (the product of the scale and the step, per entry)
function ref_elastic(x, t, λ1, λ2, w)
  map(eachindex(x)) do i
    u = bsoft(big(x[i]), t[i] * bcoef(λ1, w[i]))
    isinf(λ2) ? zero(u) : u / (1 + t[i] * big(λ2))
  end
end
# the sparse group lasso with the steps t (one per entry, constant on a group)
function ref_sparsegroup(x, t, λ1, λ0, w, groups)
  y = zeros(BigFloat, length(x))
  for (k, g) in enumerate(groups)
    u = [bsoft(big(x[i]), t[i] * big(λ1)) for i in g]
    nu = sqrt(sum(abs2, u))
    thr = t[g[1]] * bcoef(λ0, w[k])
    y[g] .= (iszero(nu) ? big(1) : (isinf(thr) ? big(0) : max(1 - thr / nu, 0))) .* u
  end
  y
end
# the value of the scale times the function at y
function val_elastic(α, y, λ1, λ2, w)
  tot = big(0)
  for i in eachindex(y)
    yi = big(y[i])
    a = iszero(yi) || iszero(w[i]) ? big(0) : (isinf(λ1) ? big(Inf) : big(λ1) * big(w[i]) * abs(yi))
    tot += a + (isinf(λ2) ? (iszero(yi) ? big(0) : big(Inf)) : big(λ2) / 2 * yi^2)
  end
  big(α) * tot
end
function val_sparsegroup(α, y, λ1, λ0, w, groups)
  tot = big(0)
  for (k, g) in enumerate(groups)
    yg = big.(y[g])
    nrm = sqrt(sum(abs2, yg))
    tot += isinf(λ1) ? (all(iszero, yg) ? big(0) : big(Inf)) : big(λ1) * sum(abs, yg)
    tot += iszero(nrm) || iszero(w[k]) ? big(0) : (isinf(λ0) ? big(Inf) : big(λ0) * big(w[k]) * nrm)
  end
  big(α) * tot
end
nearref(y, r, x, ::Type{T}) where {T} = all(abs.(y .- T.(r)) .<= 4eps(T) .* max.(abs.(x), abs.(T.(r))) .+ floatmin(T))
# the value of value_and_prox! at the stored output, against the BigFloat value there (the value in the
# type of the output can itself be inexact, as a square below floatmin)
valnear(v, r, ::Type{T}) where {T} = (isinf(r) && isinf(v)) || abs(big(v) - r) <= 8eps(T) * abs(r) + floatmin(T)

# check a scaled function at the step γ against the reference r(y), and the value against val(y): separate,
# sentinel and aliased outputs
function check_ref(s, x, γ, ref, val; T = float(eltype(x)))
  y = zeros(T, length(x))
  v = value_and_prox!(y, s, x, γ)
  @test nearref(y, ref, x, T)
  @test valnear(v, val(y), T)
  @test isequal(prox(s, x, γ), y)
  a = copy(x)
  va = value_and_prox!(a, s, a, γ)
  @test isequal(a, y) && isequal(va, v)
  z = fill(T(-7), length(x))
  prox!(z, s, x, γ)
  @test isequal(z, y)
end

@testset "the reproducers of the reviews" begin
  # an array step on a function with a zero weight: the ridge acts on the free entry, which is not dropped
  f = Scaled(ElasticNet(Inf, 1.0, [0.0, 1.0]), 1e-310)
  x = [1.0, 2.0]
  y = zeros(2)
  v = value_and_prox!(y, f, x, [1e300, 1e300])
  @test nearref(y, ref_elastic(x, big(1e-310) .* [big(1e300), big(1e300)], Inf, 1.0, [0.0, 1.0]), x, Float64)
  @test y[2] == 0 && y[1] < 1 && v > 0 && isequal(v, f(y))
  # a scalar step on a group function with a zero weight: the group with a positive weight stays fixed
  f = Scaled(SparseGroupNormL2(1.0, Inf, [1:1, 2:2], [0.0, 1.0]), 1e-300)
  y = zeros(2)
  v = value_and_prox!(y, f, x, 1e-300)
  @test y == [1.0, 0.0] && v ≈ 1e-300 && isequal(v, f(y))
  # a finite block with a large weight next to an infinite block: its coefficient 1e-160 1e-150 1e-100 1e300
  # = 1e-110 is far above floatmin, and the entry 2e-110 loses half its size in the exact map
  g = ElasticNet(1e-100, 0.0, [0.0, 1e300])
  xs = ([1.0, 2.0], [1.0, 2e-110])
  ref = ref_elastic(xs[2], fill(big(1e-160) * big(1e-150), 2), 1e-100, 0.0, [0.0, 1e300])
  for f in (Scaled(SeparableSum(NormL1(Inf), g), 1e-160), Scaled(g, 1e-160))
    res = prox(f, f.f isa SeparableSum ? xs : xs[2], 1e-150)
    r = f.f isa SeparableSum ? res[2] : res
    # the product 1e-310 is a subnormal step, which carries about 14 digits, as before the change
    @test isapprox(r, Float64.(ref); rtol = 1e-13)
    @test abs(r[2] - 1e-110) < 1e-120          # about 1e-110, not 2e-110
    f.f isa SeparableSum && @test res[1] == [0.0, 0.0]
  end
end

# the exponents: products of the scale and the step from just above floatmin to far below the smallest
# subnormal, split between the scale and the step in two ways
function exponent_grid(::Type{T}) where {T}
  ks = T === Float64 ? (290, 305, 308, 310, 320, 323, 330, 400, 500) : (30, 35, 38, 40, 44, 45, 50, 60, 80)
  lim = T === Float64 ? 323 : 44
  [(T(10.0^(-k * f)), T(10.0^(-k * (1 - f)))) for k in ks for f in (0.5, 0.3) if k * f <= lim && k * (1 - f) <= lim]
end

@testset "levels and weights at a scalar step, $T" for T in (Float32, Float64)
  inf = T(Inf)
  wbig = T === Float64 ? T(1e300) : T(1f30)
  wsmall = T(1e-3)
  levels = (T(1), T === Float64 ? T(1e-100) : T(1f-10))
  x = T[1, 2]
  groups = [1:1, 2:2]
  for (α, γ) in exponent_grid(T), w1 in (T(0), wsmall, T(1), wbig), w2 in (T(0), wsmall, T(1), wbig), λ in levels
    t = fill(big(α) * big(γ), 2)
    w = T[w1, w2]
    for (f, ref, val) in (
        (ElasticNet(inf, λ, w), ref_elastic(x, t, inf, λ, w), y -> val_elastic(α, y, inf, λ, w)),
        (ElasticNet(λ, inf, w), ref_elastic(x, t, λ, inf, w), y -> val_elastic(α, y, λ, inf, w)),
        (SparseGroupNormL2(λ, inf, groups, w), ref_sparsegroup(x, t, λ, inf, w, groups), y -> val_sparsegroup(α, y, λ, inf, w, groups)),
        (SparseGroupNormL2(inf, λ, groups, w), ref_sparsegroup(x, t, inf, λ, w, groups), y -> val_sparsegroup(α, y, inf, λ, w, groups)))
      check_ref(Scaled(f, α), x, γ, ref, val)
    end
  end
end

@testset "mixed hard and free groups and entries, $T" for T in (Float32, Float64)
  inf = T(Inf)
  α, γ = T === Float64 ? (T(1e-200), T(1e-200)) : (T(1f-25), T(1f-25))
  x = T[1, 2, 3, 4, 5]
  t = fill(big(α) * big(γ), 5)
  groups = [1:2, 3:3, 4:5]
  for w in (T[0, 1, 0], T[1, 0, 1], T[0, 0, 1], T[1, 1, 1], T[0, 0, 0])
    for (f, ref, val) in (
        (SparseGroupNormL2(T(1), inf, groups, w), ref_sparsegroup(x, t, T(1), inf, w, groups), y -> val_sparsegroup(α, y, T(1), inf, w, groups)),
        (SparseGroupNormL2(inf, T(1), groups, w), ref_sparsegroup(x, t, inf, T(1), w, groups), y -> val_sparsegroup(α, y, inf, T(1), w, groups)),
        (ElasticNet(inf, T(1), T[0, 1, 0, 1, 1] .* w[[1, 2, 3, 3, 3]]), ref_elastic(x, t, inf, T(1), T[0, 1, 0, 1, 1] .* w[[1, 2, 3, 3, 3]]),
         y -> val_elastic(α, y, inf, T(1), T[0, 1, 0, 1, 1] .* w[[1, 2, 3, 3, 3]])))
      check_ref(Scaled(f, α), x, γ, ref, val)
    end
  end
  # a coordinate at a time on an AtomIterate: the hard groups are zero, the free groups are not
  f = SparseGroupNormL2(T(1), inf, groups, T[0, 1, 0])
  ref = ref_sparsegroup(x, t, T(1), inf, T[0, 1, 0], groups)
  for k in 1:3
    a = AtomIterate(copy(x), f)
    prox_coordinate!(a, Scaled(f, α), k, γ)
    @test nearref(collect(a)[groups[k]], ref[groups[k]], x[groups[k]], T)
    @test all(iszero, collect(a)[groups[k]]) == (k == 2)
  end
end

@testset "array steps and weights, $T" for T in (Float32, Float64)
  inf = T(Inf)
  wbig = T === Float64 ? T(1e300) : T(1f30)
  αs = T === Float64 ? T[1e-200, 1e-100, 1e100, 1e300] : T[1f-25, 1f-20, 1f20, 1f30]
  γs = T === Float64 ? T[1e300, 1e-5, 1, 1e-300] : T[1f30, 1f-5, 1, 1f-30]
  x = T[1, 2]
  for α in αs, g1 in γs, g2 in γs, w1 in (T(0), T(1e-3), T(1), wbig), w2 in (T(0), T(1e-3), T(1), wbig), λ in (T(1), T(1e-10))
    # with a zero weight the finite level acts on a free entry: the cases of a finite level that is a normal
    # number once the scale is folded in (no fixed entry is then affected); with positive weights the
    # infinite level fixes every entry, and the finite level is irrelevant
    (iszero(w1) || iszero(w2)) && !(big(α) * big(λ) >= big(floatmin(T))) && continue
    t = [big(α) * big(g1), big(α) * big(g2)]
    w = T[w1, w2]
    γ = T[g1, g2]
    for (f, ref, val) in (
        (ElasticNet(inf, λ, w), ref_elastic(x, t, inf, λ, w), y -> val_elastic(α, y, inf, λ, w)),
        (ElasticNet(λ, inf, w), ref_elastic(x, t, λ, inf, w), y -> val_elastic(α, y, λ, inf, w)),
        (SparseGroupNormL2(λ, inf, [1:1, 2:2], w), ref_sparsegroup(x, t, λ, inf, w, [1:1, 2:2]), y -> val_sparsegroup(α, y, λ, inf, w, [1:1, 2:2])),
        (SparseGroupNormL2(inf, λ, [1:1, 2:2], w), ref_sparsegroup(x, t, inf, λ, w, [1:1, 2:2]), y -> val_sparsegroup(α, y, inf, λ, w, [1:1, 2:2])))
      check_ref(Scaled(f, α), x, γ, ref, val)
    end
  end
end

@testset "the entry around the threshold, $T" for T in (Float32, Float64)
  # entries from the smallest subnormal to 1 with a threshold t = αγ near floatmin and the smallest
  # subnormal: the result is the exact one rounded, up to floatmin where a level below the range is dropped
  inf = T(Inf)
  smallest = nextfloat(zero(T))
  xs = T[smallest, 4smallest, floatmin(T) / 2, floatmin(T), 3floatmin(T), T(1e-3), T(1)]
  for (α, γ) in exponent_grid(T), x1 in xs
    x = T[x1, 2]
    t = fill(big(α) * big(γ), 2)
    for (f, ref, val) in (
        (SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[0, 1]), ref_sparsegroup(x, t, T(1), inf, T[0, 1], [1:1, 2:2]), y -> val_sparsegroup(α, y, T(1), inf, T[0, 1], [1:1, 2:2])),
        (ElasticNet(inf, T(1), T[0, 1]), ref_elastic(x, t, inf, T(1), T[0, 1]), y -> val_elastic(α, y, inf, T(1), T[0, 1])))
      s = Scaled(f, α)
      y = zeros(T, 2)
      v = value_and_prox!(y, s, x, γ)
      @test y[2] == 0
      # the change of the free entry against the exact result is below floatmin, and a ridge below its relative bound
      @test abs(big(y[1]) - ref[1]) <= 4eps(T) * max(abs(big(x1)), abs(ref[1])) + big(floatmin(T))
      @test valnear(v, val(y), T)
    end
  end
end

@testset "finite siblings of an infinite block, $T" for T in (Float32, Float64)
  # The finite block is folded with the chain, discarded only when its whole coefficient (the product,
  # the level and the largest weight) is below floatmin, and otherwise the fold is declined. A product
  # that is a representable number keeps the step as it is, so the hard block stays fixed. The
  # configurations with a product that underflows to zero while a large weight brings the coefficient
  # above floatmin are not covered: there the fold is declined and the infinite block is not fixed.
  inf = T(Inf)
  wbig = T === Float64 ? T(1e300) : T(1f30)
  for (α, γ) in exponent_grid(T), w1 in (T(0), T(1e-3), T(1)), w2 in (T(0), T(1e-3), T(1), wbig), λ1 in (T(1), T === Float64 ? T(1e-100) : T(1f-10)), λ2 in (T(0), T(1))
    tt = big(α) * big(γ)
    # a product within a factor 2 of half the smallest subnormal number can round either way (the rounding
    # of ldexp differs between Julia versions), so only products from twice the smallest subnormal count
    supported = α * γ >= 2 * nextfloat(zero(T)) || tt * big(λ1) * big(max(w1, w2)) < big(floatmin(T))
    supported || continue
    f = Scaled(SeparableSum(NormL1(inf), ElasticNet(λ1, λ2, T[w1, w2])), α)
    # the second entry is twice its own threshold where that is a normal number, so that the threshold matters
    thr = big(λ1) * big(w2) * tt
    x2 = T[1, thr > big(floatmin(T)) ? T(2thr) : T(2)]
    r = ref_elastic(x2, [tt, tt], λ1, λ2, T[w1, w2])
    res = prox(f, (T[1, 2], x2), γ)
    @test all(iszero, res[1])
    # a product that is a subnormal number is a step with few digits: its relative error, at most the
    # smallest subnormal over the product, multiplies the change that the threshold makes
    δ = α * γ > 0 && α * γ < floatmin(T) ? 2 * big(nextfloat(zero(T))) / big(α * γ) : big(0)
    @test all(abs.(res[2] .- T.(r)) .<= 4eps(T) .* max.(abs.(x2), abs.(T.(r))) .+ floatmin(T) .+ T.(δ .* abs.(big.(x2) .- r)))
  end
end

@testset "nested Scaled, SeparableSum and Translate around the group functions, $T" for T in (Float32, Float64)
  inf = T(Inf)
  α1, α2 = T === Float64 ? (T(1e-100), T(1e-100)) : (T(1f-20), T(1f-20))
  γ = α1
  x = T[1, 2]
  b = T[0.5, -1]
  g = SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[0, 1])
  e = ElasticNet(inf, T(1), T[0, 1])
  t = fill(big(α1) * big(α2) * big(γ) * 2, 2)       # the inner factor 2 of the scaled block
  rg = ref_sparsegroup(x, t, T(1), inf, T[0, 1], [1:1, 2:2])
  re = ref_elastic(x, t, inf, T(1), T[0, 1])
  for (f, ref) in ((Scaled(Scaled(Scaled(g, T(2)), α1), α2), rg),
                   (Scaled(Scaled(Scaled(e, T(2)), α1), α2), re))
    y = zeros(T, 2)
    v = value_and_prox!(y, f, x, γ)
    @test nearref(y, ref, x, T) && y[2] == 0
    @test isequal(prox(f, x, γ), y)
  end
  # a sum of blocks with a scaled block and a translate: each block as alone
  tr = T[3, 4]
  f = Scaled(Scaled(SeparableSum(Scaled(g, T(2)), Translate(e, tr)), α1), α2)
  res = prox(f, (x, x), γ)
  @test nearref(res[1], rg, x, T) && res[1][2] == 0
  re2 = ref_elastic(x .- tr, fill(big(α1) * big(α2) * big(γ), 2), inf, T(1), T[0, 1])
  @test nearref(res[2], re2 .+ tr, x, T) && res[2][2] == tr[2]
end


end
