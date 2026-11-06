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

end
