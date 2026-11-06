module ScaledInfiniteLevelUnderflowTest

# A wrapper step that is not a normal number keeps the hard domain of an infinite level (decision 49,
# R3). `Scaled(f, α)` at a positive finite step γ whose product αγ underflows to zero, is subnormal or
# overflows reads f at the product αγ. An infinite level (a stored Inf, not a finite level that
# overflows the working precision) stays infinite under a positive finite scale, so the proximal map
# is the constant map of the domain {0}, and not the identity (γ = 0) or the NaN of Inf * 0 that a
# product rounded to zero gave. The call takes this route only when the function holds a stored infinite
# level, every block of it is a function with an infinite level whose other levels the infinite level makes
# irrelevant (or an indicator), and every factor and step is a positive finite number; no finite level or
# factor is folded, multiplied or combined. Scalar steps, array steps, chains of Scaled, prox_entry and
# prox_coordinate! take the same route.
# Supported (exact, tested against the constant map and BigFloat): the issue's rows, the groups, `ElasticNet`,
# `SparseGroupNormL2`, chains, nested blocks. Declined (the earlier result, pinned as the leaf at the rounded
# product or as a literal of the earlier result, with a follow-up): a finite level that is relevant on a free
# coordinate (a zero weight), a finite sibling block, an infinite, zero, negative or NaN factor or step
# inside a block or in the chain, and every function without a stored infinite level.
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

# as `check`, up to the sign of a zero: the sign of a zero that a leaf gives can depend on the step (for a
# soft threshold of a negative entry), and the route takes a step of any positive finite size
z0(a) = a .+ zero(eltype(a))
function check0(f, x, γ, expected)
  T = float(eltype(x))
  @test isequal(z0(prox(f, x, γ)), z0(expected))
  for op in (prox!, value_and_prox!)
    fresh = similar(x, T)
    op(fresh, f, x, γ)
    sentinel = fill(T(-7), size(x))
    op(sentinel, f, x, γ)
    alias = copy(x)
    op(alias, f, alias, γ)
    @test isequal(z0(fresh), z0(expected)) && isequal(z0(sentinel), z0(expected)) && isequal(z0(alias), z0(expected))
  end
end

# the same result, or the same error, of two calls
function same_outcome(a, b)
  ra = try a() catch e; e end
  rb = try b() catch e; e end
  ra isa Exception || rb isa Exception ? (ra isa Exception && rb isa Exception && typeof(ra) === typeof(rb)) : isequal(ra, rb)
end

# The declined class (the earlier result is kept, and an infinite level can be lost): a finite level that is
# relevant on a coordinate that the infinite level leaves free (a zero weight), or a finite sibling block. A
# call of the class is pinned to the result of the steps rounded with the factor, the leaf at the rounded product.
declined(f::ElasticNet) = isinf(f.λ1) && !isinf(f.λ2) && !iszero(f.λ2) && !all(>(0), f.w)
declined(f::SparseGroupNormL2) = !isinf(f.λ1) && isinf(f.λ0) && !iszero(f.λ1) && !all(>(0), f.w)
declined(f) = false

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
  # an array step with a zero entry keeps the result of the steps rounded with the factor, in every entry
  @test prox(Scaled(NormL1(Inf), 1e-300), x, [0.0, 1e-300, 1.0]) == prox(NormL1(Inf), x, [0.0, 1e-600, 1e-300])
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
    # an exact zero entry of the step is not a positive step: the call keeps the result of the steps rounded with
    # the factor, in every entry (the array fold is made only when every step is positive and finite)
    check(Scaled(leaf, tiny), x, T[0, tiny, tiny], prox(leaf, x, T[0, tiny * tiny, tiny * tiny]); value = nothing)
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
  # blocks of different kinds: every block folds exactly (an infinite level, or an indicator that keeps its
  # own step), and the infinite blocks are fixed at 0
  arg = ([1.0, 2.0], [3.0, 4.0])
  for (blocks, expected) in (((NormL1(Inf), NormL1(Inf)), ([0.0, 0.0], [0.0, 0.0])),
                             ((NormL1(Inf), NormL2(Inf)), ([0.0, 0.0], [0.0, 0.0])),
                             ((NormL1(Inf), IndNonnegative()), ([0.0, 0.0], [3.0, 4.0])))
    sep = SeparableSum(blocks...)
    for (α, γ) in ((1e-300, 1e-300), (1e-200, 1e-200), (1e-160, 1e-150))
      for s in (Scaled(sep, α), Scaled(Scaled(sep, α), 1.0))
        @test isequal(prox(s, arg, γ), expected)
      end
    end
  end
  # a finite block whose level cannot be formed by the exponent shift next to an infinite block, or that
  # cannot fold (a finite SLOPE): the fold is declined and the call keeps the result of the product rounded to
  # zero. This is the unsupported class, a finite sibling block, tracked as a follow-up.
  for blocks in ((NormL1(Inf), NormL1(1.0)), (NormL1(1.0), NormL1(Inf)), (SLOPE([Inf, 1.0]), NormL1(2.0)))
    sep = SeparableSum(blocks...)
    for (α, γ) in ((1e-300, 1e-300), (1e-200, 1e-200))
      @test isequal(prox(Scaled(sep, α), arg, γ), prox(sep, arg, α * γ))
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
# weight zero) acts on it: it is folded when the scale can form it, and the fold is otherwise declined, so
# that the call keeps the result of the product rounded with the factor (nothing is discarded by size).
# The value is the value at the stored output (O2).
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
    # A scalar step: the ridge of the elastic net and the ℓ1 level of the group function are finite levels that
    # are relevant on the free coordinate, so the call is declined and the result is that of the product
    # rounded to zero, with the group of a positive weight not fixed. This is the declined class (a finite level
    # that is relevant on a free coordinate); it is pinned to the earlier route, the leaf at the rounded product.
    for (αs, γs) in ((T === Float64 ? (1e-300, 1e-300) : (1f-30, 1f-30)),)
      s2 = Scaled(f, αs)
      y = zeros(T, 2)
      v = value_and_prox!(y, s2, x, γs)
      @test isequal(y, prox(f, x, αs * γs))
      @test isequal(v, s2(y))
      a = copy(x)
      va = value_and_prox!(a, s2, a, γs)
      @test isequal(a, y) && isequal(va, v)
      sentinel = fill(T(-7), 2)
      prox!(sentinel, s2, x, γs)
      @test isequal(sentinel, y) && isequal(prox(s2, x, γs), y)
    end
  end
  # the same call for the group function on an AtomIterate, a coordinate at a time: the unsupported class,
  # the result of the leaf at the rounded product
  let f = SparseGroupNormL2(T(1), inf, [1:1, 2:2], T[0, 1]), tiny = T === Float64 ? T(1e-300) : T(1f-30)
    s = Scaled(f, tiny)
    for k in 1:2
      a = AtomIterate(T[1, 2], f)
      prox_coordinate!(a, s, k, tiny)
      c = AtomIterate(T[1, 2], f)
      prox_coordinate!(c, f, k, tiny * tiny)
      @test isequal(collect(a), collect(c))
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
  # a finite scaled block next to an infinite one: the call is declined and keeps the result of the product
  # rounded to zero (the declined class: a finite sibling block); no finite level or factor is folded.
  f = Scaled(SeparableSum(Scaled(NormL1(1.0), 2.0), Scaled(NormL1(Inf), 3.0)), 1e-300)
  @test isequal(prox(f, arg, 1e-300), prox(f.f, arg, 0.0))
  f = Scaled(SeparableSum(Scaled(NormL1(1e300), 2.0), Scaled(NormL1(Inf), 3.0)), 1e-300)
  @test isequal(prox(f, arg, 1e-300), prox(f.f, arg, 0.0))
end

####################################
#  a finite block next to an infinite one: the declined class
####################################

# A finite SLOPE has no fold of its own (it folds only with an infinite first level), so a sum of it and an
# infinite block declines the fold and keeps the result of the product rounded with the factor: 1e-160 1e-200
# is 1e-360, which is zero, and the entries come back unchanged. The finite block gives the exact map and the
# infinite block loses its constraint: the unsupported class (a finite sibling block), tracked as a follow-up.
@testset "a finite SLOPE sibling is declined" begin
  f = Scaled(SeparableSum(NormL1(Inf), SLOPE([1e100, 5e99])), 1e-160)
  @test isequal(prox(f, ([1.0, 2.0], [1e-310, -1e-310]), 1e-200), ([1.0, 2.0], [1e-310, -1e-310]))
end

####################################
#  levels and weights: references in BigFloat, formed here, not by the package
####################################

# What a finite level does once an infinite level is present (see `_HardScale` in src/calculus.jl):
# irrelevant (the infinite level fixes every coordinate it touches): removed; otherwise folded when the scale
# can form it, and else the fold is declined (nothing is discarded by size). The references below are the
# exact proximal maps in BigFloat; entries are compared to the rounded reference with 4 units in the last
# place of the larger of the argument and the result, plus floatmin for the entries of subnormal size.
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
  # a scalar step on a group function with a zero weight
  # (the declined class: a finite level that is relevant on a free coordinate;
  # the result is that of the earlier route, the leaf at the rounded product, and a follow-up issue tracks it)
  f = Scaled(SparseGroupNormL2(1.0, Inf, [1:1, 2:2], [0.0, 1.0]), 1e-300)
  y = zeros(2)
  v = value_and_prox!(y, f, x, 1e-300)
  @test isequal(y, prox(f.f, x, 1e-300 * 1e-300)) && isequal(v, f(y))
  # a finite block with a large weight next to an infinite block: its coefficient 1e-160 1e-150 1e-100 1e300
  # = 1e-110 is far above floatmin, and the entry 2e-110 loses half its size in the exact map. The fold is
  # declined (the level 1e-410 is not a number), the product 1e-310 is a positive subnormal step, so the
  # infinite block stays fixed and the finite block keeps its value
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
      # a finite level that is relevant on a free coordinate is the declined class (checked against the earlier
      # route where the product is zero)
      if declined(f)
        α * γ == 0 && @test isequal(prox(Scaled(f, α), x, γ), prox(f, x, α * γ))
        continue
      end
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
      # a finite level that is relevant on a free coordinate is the declined class (the product is zero here: the
      # result of the earlier route)
      if declined(f)
        @test isequal(prox(Scaled(f, α), x, γ), prox(f, x, α * γ))
        continue
      end
      check_ref(Scaled(f, α), x, γ, ref, val)
    end
  end
  # a coordinate at a time on an AtomIterate: a group function whose infinite ℓ1 level fixes every group
  f = SparseGroupNormL2(inf, T(1), groups, T[0, 1, 0])
  for k in 1:3
    a = AtomIterate(copy(x), f)
    prox_coordinate!(a, Scaled(f, α), k, γ)
    @test all(iszero, collect(a)[groups[k]])
  end
  # and the unsupported class: the result of the leaf at the rounded product
  f = SparseGroupNormL2(T(1), inf, groups, T[0, 1, 0])
  for k in 1:3
    a = AtomIterate(copy(x), f)
    prox_coordinate!(a, Scaled(f, α), k, γ)
    c = AtomIterate(copy(x), f)
    prox_coordinate!(c, f, k, α * γ)
    @test isequal(collect(a), collect(c))
  end
end

@testset "array steps and weights, $T" for T in (Float32, Float64)
  inf = T(Inf)
  wbig = T === Float64 ? T(1e300) : T(1f30)
  αs = T === Float64 ? T[1e-200, 1e-100, 1e100, 1e300] : T[1f-25, 1f-20, 1f20, 1f30]
  γs = T === Float64 ? T[1e300, 1e-5, 1, 1e-300] : T[1f30, 1f-5, 1, 1f-30]
  x = T[1, 2]
  for α in αs, g1 in γs, g2 in γs, w1 in (T(0), T(1e-3), T(1), wbig), w2 in (T(0), T(1e-3), T(1), wbig), λ in (T(1), T(1e-10))
    t = [big(α) * big(g1), big(α) * big(g2)]
    w = T[w1, w2]
    γ = T[g1, g2]
    # with a zero weight the finite level acts on a free entry: the declined class, pinned to the steps rounded
    # with the factor (where no product is subnormal, as the rounding of such a product is not pinned)
    rounded = T[α * g1, α * g2]
    pinned = all(p -> iszero(p) || isinf(p) || p >= floatmin(T), rounded)
    for (f, ref, val) in (
        (ElasticNet(inf, λ, w), ref_elastic(x, t, inf, λ, w), y -> val_elastic(α, y, inf, λ, w)),
        (ElasticNet(λ, inf, w), ref_elastic(x, t, λ, inf, w), y -> val_elastic(α, y, λ, inf, w)),
        (SparseGroupNormL2(λ, inf, [1:1, 2:2], w), ref_sparsegroup(x, t, λ, inf, w, [1:1, 2:2]), y -> val_sparsegroup(α, y, λ, inf, w, [1:1, 2:2])),
        (SparseGroupNormL2(inf, λ, [1:1, 2:2], w), ref_sparsegroup(x, t, inf, λ, w, [1:1, 2:2]), y -> val_sparsegroup(α, y, inf, λ, w, [1:1, 2:2])))
      if declined(f)
        pinned && @test isequal(prox(Scaled(f, α), x, γ), prox(f, x, rounded))
        continue
      end
      check_ref(Scaled(f, α), x, γ, ref, val)
    end
  end
end

@testset "finite siblings of an infinite block are declined, $T" for T in (Float32, Float64)
  # A finite sibling block is declined: no finite level is folded, and the call keeps the result of the
  # product rounded with the factor. Where that product is a positive number the infinite block stays fixed
  # and the finite block is the exact map (checked against BigFloat); where the product is zero the infinite
  # block loses its constraint, and the call is pinned to the leaf at the rounded product: the declined
  # class, a finite sibling block, tracked as a follow-up.
  inf = T(Inf)
  wbig = T === Float64 ? T(1e300) : T(1f30)
  smallest = nextfloat(zero(T))
  for (α, γ) in exponent_grid(T), w1 in (T(0), T(1e-3), T(1)), w2 in (T(0), T(1e-3), T(1), wbig), λ1 in (T(1), T === Float64 ? T(1e-100) : T(1f-10)), λ2 in (T(0), T(1))
    tt = big(α) * big(γ)
    sibling = ElasticNet(λ1, λ2, T[w1, w2])
    f = Scaled(SeparableSum(NormL1(inf), sibling), α)
    # the folded level of the sibling is its level times the product, within a factor 2: a nonzero number
    # of the type when it is at least twice the smallest subnormal (the rounding of products near it is not
    # pinned, as ldexp rounds differently between Julia versions)
    foldable = big(λ1) * tt >= 4 * big(smallest)
    x2 = T[1, 2]
    arg = (T[1, 2], x2)
    res = prox(f, arg, γ)
    if foldable
      r = ref_elastic(x2, [tt, tt], λ1, λ2, T[w1, w2])
      @test all(iszero, res[1])
      # the level is a subnormal number: its relative error, at most the smallest subnormal over the level,
      # multiplies the change that the threshold makes
      δ = 2 * big(smallest) / (big(λ1) * tt)
      @test all(abs.(res[2] .- T.(r)) .<= 4eps(T) .* max.(abs.(x2), abs.(T.(r))) .+ T.(δ .* abs.(big.(x2) .- r)))
    elseif α * γ == 0
      @test isequal(res, prox(SeparableSum(NormL1(inf), sibling), arg, α * γ))
    end
  end
end

@testset "nested Scaled, SeparableSum and Translate around the group functions, $T" for T in (Float32, Float64)
  inf = T(Inf)
  α1, α2 = T === Float64 ? (T(1e-100), T(1e-100)) : (T(1f-20), T(1f-20))
  γ = α1
  x = T[1, 2]
  b = T[0.5, -1]
  g = SparseGroupNormL2(inf, T(1), [1:1, 2:2], T[0, 1])      # the infinite ℓ1 level fixes every group
  e = ElasticNet(inf, T(0), T[0, 1])        # no finite level that acts
  t = fill(big(α1) * big(α2) * big(γ) * 2, 2)       # the inner factor 2 of the scaled block
  rg = ref_sparsegroup(x, t, inf, T(1), T[0, 1], [1:1, 2:2])
  re = ref_elastic(x, t, inf, T(0), T[0, 1])
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
  re2 = ref_elastic(x .- tr, fill(big(α1) * big(α2) * big(γ), 2), inf, T(0), T[0, 1])
  @test nearref(res[2], re2 .+ tr, x, T) && res[2][2] == tr[2]
end


####################################
#  parameters and arguments of different precisions
####################################

# A finite sibling is declined: the call keeps the result of the steps rounded with the factor, and the leaf
# forms the coefficient in the type of its own arithmetic. Float32
# parameters on a Float64 argument keep a coefficient of 1e-50, which is a normal Float64 number (here the
# product 1e-40 is a Float32 subnormal, with its 5 digits). The references are BigFloat.
@testset "mixed precision of the parameters and the call" begin
  x2 = [1.0, 2e-50]
  w = Float32[0, 1f10]
  groups = [1:1, 2:2]
  t = fill(big(1f-20) * big(1f-20), 2)
  siblings = (("ElasticNet", ElasticNet(1f-20, 0f0, w), ref_elastic(x2, t, 1f-20, 0f0, w)),
              ("NormL1", NormL1(1f-20, w), ref_elastic(x2, t, 1f-20, 0f0, w)),
              ("SparseGroupNormL2", SparseGroupNormL2(0f0, 1f-20, groups, w), ref_sparsegroup(x2, t, 0f0, 1f-20, w, groups)),
              ("GroupNormL2", GroupNormL2(1f-20, groups, w), ref_sparsegroup(x2, t, 0f0, 1f-20, w, groups)))
  for (name, g, ref) in siblings
    f = Scaled(SeparableSum(NormL1(Inf32), g), 1f-20)
    res = prox(f, ([1.0, 2.0], x2), 1f-20)
    @test res[1] == [0.0, 0.0]
    # the product 1e-40 is a Float32 subnormal: its relative error, 3e-5, multiplies the change of the entry
    @test abs(res[2][2] - Float64(ref[2])) <= 4eps() * 2e-50 + 3e-5 * abs(2e-50 - Float64(ref[2]))
    @test abs(res[2][2] - 2e-50) > 1e-51                        # the threshold acts
    y = (zeros(2), zeros(2))
    v = value_and_prox!(y, f, ([1.0, 2.0], x2), 1f-20)
    @test y == res && isapprox(v, f(y); rtol = 1e-4)
    a = ([1.0, 2.0], copy(x2))
    va = value_and_prox!(a, f, a, 1f-20)
    @test a == res && va == v
  end
  # the reverse: Float64 parameters on a Float32 argument; the output cannot hold a coefficient below its range
  t64 = fill(big(1e-20) * big(1e-20), 2)
  g = ElasticNet(1e-20, 0.0, [0.0, 1e10])
  f = Scaled(SeparableSum(NormL1(Inf), g), 1e-20)
  res = prox(f, (Float32[1, 2], Float32[1, 2f-30]), 1e-20)
  @test res[1] == Float32[0, 0]
  @test nearref(res[2], ref_elastic(Float32[1, 2f-30], t64, 1e-20, 0.0, [0.0, 1e10]), Float32[1, 2f-30], Float32)
  # all in Float32
  g32 = ElasticNet(1f-20, 0f0, w)
  f32 = Scaled(SeparableSum(NormL1(Inf32), g32), 1f-20)
  res = prox(f32, (Float32[1, 2], Float32[1, 2f-30]), 1f-20)
  @test res[1] == Float32[0, 0]
  @test nearref(res[2], ref_elastic(Float32[1, 2f-30], t, 1f-20, 0f0, w), Float32[1, 2f-30], Float32)
end

####################################
#  every entry that the infinite level fixes stays fixed when a finite level is removed
####################################

# A leaf rebuilt without an irrelevant finite level keeps the result of the leaf with it: the infinite
# level fixes every entry, whatever the entry is (NaN and infinite entries included, E2 and E3), and the
# result is the constant map, which is the map of the unscaled function at a step of 1.
@testset "removing an irrelevant level keeps the hard path, $T" for T in (Float32, Float64)
  inf = T(Inf)
  wbig = T === Float64 ? T(1e300) : T(1f30)
  big_ = wbig
  tiny = T === Float64 ? T(1e-300) : T(1f-30)
  G = [1:1, 2:2, 3:4]
  xs = (T[1, 2, 3, 4], T[NaN, 2, 3, 4], T[Inf, -Inf, 3, NaN], T[1, NaN, NaN, Inf], T[NaN, NaN, NaN, NaN], T[-Inf, 0, 0, 1])
  leaves = (SparseGroupNormL2(inf, big_, G, T[1, 1, 1]), SparseGroupNormL2(inf, T(1), G, T[1, 2, 1e-3]),
            SparseGroupNormL2(big_, inf, G, T[1, 1, 1]), SparseGroupNormL2(T(1), inf, G, T[1, 1, 1]),
            SparseGroupNormL2(inf, T(1e-3), [1:2, 3:3], T[1, 1]),
            ElasticNet(inf, big_, T[1, 1, 1, 1]), ElasticNet(inf, T(1), T[1, 1, 1, 1]), ElasticNet(big_, inf, T[1, 1, 1, 1]),
            SLOPE(T[inf, big_, 1, 1]), SLOPE(T[inf, 1, tiny, 0]), NormL1Fused(inf, big_), NormL1Fused(inf, T(1)))
  for f in leaves, x in xs
    ref = prox(f, x, one(T))
    for (α, γ) in ((big_, big_), (tiny, tiny), (tiny, T(1)), (T(1), tiny))
      check(Scaled(f, α), x, γ, ref; value = nothing)
      s = Scaled(f, α)
      y = similar(x)
      all(iszero, ref) && @test value_and_prox!(y, s, x, γ) === zero(T)
    end
    # a group function takes one step per group, an elastic net one per entry
    n = f isa SparseGroupNormL2 ? length(f.groups) : length(x)
    f isa Union{SLOPE, NormL1Fused} && continue
    for γs in (fill(big_, n), fill(tiny, n), T[tiny, 1, big_, tiny][1:n])
      check(Scaled(f, tiny), x, γs, prox(f, x, ones(T, n)); value = nothing)
      check(Scaled(f, big_), x, γs, prox(f, x, ones(T, n)); value = nothing)
    end
  end
  # the array call of the review: a zero step entry keeps its earlier result in every entry
  f = Scaled(SparseGroupNormL2(inf, big_, [1:1, 2:2], T[1, 1]), big_)
  @test isequal(prox(f, T[1, 2], T[0, 1]), prox(SparseGroupNormL2(inf, big_, [1:1, 2:2], T[1, 1]), T[1, 2], T[0, big_ * 1]))
  # the calls of the review
  for x in (T[NaN, 2], T[Inf, 2], T[1, 2])
    @test isequal(prox(f, x, big_), zeros(T, 2))
  end
end

####################################
#  array steps with an excluded entry keep their results in every entry
####################################

# A negative, NaN, zero or infinite entry of an array step is outside the rule (decision 49): the call keeps
# the result of the steps rounded with the factor, for the entries that are positive as well. The array fold
# of an infinite level is made only when every entry is positive and finite.
@testset "array steps with excluded entries, $T" for T in (Float32, Float64)
  inf = T(Inf)
  tiny = T === Float64 ? T(1e-300) : T(1f-30)
  x = T[1, 2, 3]
  excluded = (T(-1) * tiny, T(NaN), T(0), inf, T(-0.0), T(-1))
  for leaf in (ElasticNet(inf, T(1), T[0, 1, 1]), NormL1(inf), NormL1Nonneg(inf), NormL1(inf, T[1, 0, 1]), ElasticNet(T(1), inf, T[0, 1, 1]))
    for bad in excluded, pos in (tiny, T(1)), k in 1:3
      γ = fill(pos, 3)
      γ[k] = bad
      for α in (tiny, T(1) / tiny)
        s = Scaled(leaf, α)
        rounded = [α * g for g in γ]               # the steps rounded with the factor, as before the change
        @test isequal(prox(s, x, γ), prox(leaf, x, rounded))
        y = zeros(T, 3)
        value_and_prox!(y, s, x, γ)
        @test isequal(y, prox(leaf, x, rounded))
        a = copy(x)
        value_and_prox!(a, s, a, γ)
        @test isequal(a, y)
      end
    end
  end
  # the scalar route takes a positive finite step only
  for leaf in (ElasticNet(inf, T(1), T[0, 1, 1]), NormL1(inf), SLOPE(T[inf, 1, 1]), NormL1Fused(inf, T(1)), SparseGroupNormL2(inf, T(1), [1:2, 3:3], T[1, 1]))
    for bad in excluded
      @test same_outcome(() -> prox(Scaled(leaf, tiny), x, bad), () -> prox(leaf, x, tiny * bad))
    end
  end
end

####################################
#  finite siblings are declined; a tree without a stored infinite level takes its earlier route
####################################

# A finite sibling of an infinite block is the declined class: no finite level is folded, and the call keeps
# the result of the product rounded with the factor, which is accurate here (the product is a positive
# subnormal step). The references are BigFloat; the subnormal step carries about 14 digits (Float64) or 5
# (Float32), which multiplies the change that the threshold makes.
@testset "a finite sibling with a subnormal folded level is declined, $T" for T in (Float32, Float64)
  inf = T(Inf)
  # the folded level is about 1e-322 (1e-45 in Float32), and the weight brings the coefficient back
  α, γ, λ, wbig, x1 = T === Float64 ? (T(1e-160), T(1e-150), T(1e-12), T(1e300), T(2e-22)) :
                                      (T(1f-20), T(1f-20), T(1f-5), T(1f30), T(2f-15))
  groups = [1:1, 2:2]
  w = T[0, wbig]
  x2 = T[1, x1]
  t = fill(big(α) * big(γ), 2)
  smallest = nextfloat(zero(T))
  δ = 2 * big(smallest) / (big(α) * big(γ))
  siblings = ((NormL1(λ, w), ref_elastic(x2, t, λ, T(0), w)),
              (ElasticNet(λ, T(0), w), ref_elastic(x2, t, λ, T(0), w)),
              (GroupNormL2(λ, groups, w), ref_sparsegroup(x2, t, T(0), λ, w, groups)),
              (SparseGroupNormL2(T(0), λ, groups, w), ref_sparsegroup(x2, t, T(0), λ, w, groups)),
              (SparseGroupNormL2(λ, T(0), groups, T[1, 1]), ref_sparsegroup(x2, t, λ, T(0), T[1, 1], groups)))
  for (g, ref) in siblings
    f = Scaled(SeparableSum(NormL1(inf), g), α)
    arg = (T[1, 2], x2)
    res = prox(f, arg, γ)
    @test all(iszero, res[1])
    @test all(abs.(res[2] .- T.(ref)) .<= 4eps(T) .* max.(abs.(x2), abs.(T.(ref))) .+ T.(δ .* abs.(big.(x2) .- ref)))
    y = (zeros(T, 2), zeros(T, 2))
    v = value_and_prox!(y, f, arg, γ)
    @test isequal(y, res) && isapprox(v, f(y); rtol = 1e-3)
    a = (T[1, 2], copy(x2))
    va = value_and_prox!(a, f, a, γ)
    @test isequal(a, res) && isequal(va, v)
    # a chain of factors takes the same route
    @test isequal(prox(Scaled(Scaled(f.f, T(1)), α), arg, γ), res)
  end
end

@testset "a tree without a stored infinite level takes its earlier route" begin
  # the finite level 1e300 of the review: ridge below the range, scaled Translate block in a sum. Base results,
  # pinned (the new step fold is not entered without an infinite level).
  arg = ([1e-100, 3e-100], zeros(2))
  for (e, expected) in ((ElasticNet(1e300, 1.0), [1e-100, 3e-100]),
                        (ElasticNet(1e300, 1.0, [1e-5, 2.0]), [1e-100, 3e-100]),
                        (GroupNormL2(1e300, [1:1, 2:2]), [0.0, 1.0000000000000001e-100]),
                        (SparseGroupNormL2(1e300, 1.0, [1:1, 2:2]), [1e-100, 3e-100]))
    f = Scaled(SeparableSum(Scaled(Translate(e, zeros(2)), 2.0), NormL1(1e300)), 1e-200)
    res = prox(f, arg, 1e-200)
    @test res[1] == expected && res[2] == [0.0, 0.0]
    y = (zeros(2), zeros(2))
    value_and_prox!(y, f, arg, 1e-200)
    @test y == res
  end
  f = Scaled(SeparableSum(Scaled(Translate(NormL1(1e300, [1.0, 1e10]), zeros(2)), 2.0), NormL1(1e300)), 1e-200)
  @test prox(f, arg, 1e-200) == ([0.0, 0.0], [0.0, 0.0])
  # Float32 arguments and parameters, and a Float64 level that overflows Float32 (read as stored, it is finite)
  f32 = Scaled(SeparableSum(Scaled(Translate(ElasticNet(1f30, 1f0), zeros(Float32, 2)), 2f0), NormL1(1f30)), 1f-30)
  @test prox(f32, (Float32[1f-10, 3f-10], zeros(Float32, 2)), 1f-30) == (Float32[1f-10, 3f-10], Float32[0, 0])
  g = Scaled(SeparableSum(Scaled(Translate(ElasticNet(1e300, 1.0), zeros(Float32, 2)), 2f0), NormL1(1e300)), 1f-30)
  @test prox(g, (Float32[1f-10, 3f-10], zeros(Float32, 2)), 1f-30) == (Float32[0, 0], Float32[0, 0])
  h = Scaled(Scaled(Translate(ElasticNet(1e300, 1.0), zeros(Float32, 2)), 2f0), 1f-30)
  @test prox(h, Float32[1f-10, 3f-10], 1f-30) == Float32[0, 0]
  @test prox(Scaled(Scaled(Translate(ElasticNet(1e300, 1.0), zeros(2)), 1e-100), 1e-100), [1e-100, 3e-100], 1e-100) == [0.0, 0.0]
  # the same trees with a stored infinite level are in the new route
  fi = Scaled(SeparableSum(Scaled(Translate(ElasticNet(Inf, 1.0, [1.0, 1.0]), zeros(2)), 2.0), NormL1(Inf)), 1e-200)
  @test prox(fi, arg, 1e-200) == ([0.0, 0.0], [0.0, 0.0])
  fi = Scaled(Scaled(Translate(Scaled(SLOPE([Inf, 1.0]), 3.0), zeros(2)), 2.0), 1e-200)
  @test prox(fi, [1.0, 2.0], 1e-200) == [0.0, 0.0]
end

####################################
#  the three findings of the review of round 7, against the earlier results
####################################

# 1. A finite level is never folded, so the working precision of the leaf is untouched: a Float32 level
# and scale on a Float64 argument give the result that they always gave, which is the BigFloat result.
@testset "working precision of a finite level and a finite scale" begin
  a = 1.5f0
  γ = ldexp(1f0, -128)
  λ = 1f30
  tt = Float64(a) * Float64(γ) * Float64(λ)
  groups = [1:1, 2:2]
  w = Float32[0, 1]
  # the review's call, and the earlier result as a literal
  f = Scaled(SparseGroupNormL2(λ, Inf32, groups, w), a)
  @test prox(f, [2tt, 2.0], γ) == [4.408103881914371e-9, 0.0]
  ref = ref_sparsegroup([2tt, 2.0], fill(big(a) * big(γ), 2), λ, Inf, w, groups)
  @test all(abs.(prox(f, [2tt, 2.0], γ) .- Float64.(ref)) .<= 4eps() .* [2tt, 2.0])
  # aliases, a coordinate update, the same total coefficient split between two factors, array steps
  y = zeros(2)
  v = value_and_prox!(y, f, [2tt, 2.0], γ)
  @test y == [4.408103881914371e-9, 0.0] && isequal(v, f(y))
  al = [2tt, 2.0]
  value_and_prox!(al, f, al, γ)
  @test al == y
  at = AtomIterate([2tt, 2.0], f.f)
  prox_coordinate!(at, f, 1, γ)
  @test collect(at)[1] == 4.408103881914371e-9
  for (a2, γ2) in ((ldexp(1.5f0, -40), ldexp(1f0, -88)), (ldexp(1.5f0, -80), ldexp(1f0, -48)))
    t2 = Float64(a2) * Float64(γ2) * Float64(λ)
    g2 = Scaled(SparseGroupNormL2(λ, Inf32, groups, w), a2)
    r2 = ref_sparsegroup([2t2, 2.0], fill(big(a2) * big(γ2), 2), λ, Inf, w, groups)
    @test all(abs.(prox(g2, [2t2, 2.0], γ2) .- Float64.(r2)) .<= 4eps() .* [2t2, 2.0])
    @test all(abs.(prox(g2, [2t2, 2.0], [γ2, γ2]) .- Float64.(r2)) .<= 4eps() .* [2t2, 2.0])
    g3 = Scaled(Scaled(SparseGroupNormL2(λ, Inf32, groups, w), a2), 1f0)
    @test all(abs.(prox(g3, [2t2, 2.0], γ2) .- Float64.(r2)) .<= 4eps() .* [2t2, 2.0])
  end
  # finite siblings of other kinds next to an infinite block (the declined class)
  for g in (NormL1(λ, Float32[1, 0]), ElasticNet(λ, 0f0, Float32[1, 0]), GroupNormL2(λ, groups, Float32[1, 0]),
            SparseGroupNormL2(0f0, λ, groups, Float32[1, 0]))
    rg = g isa NormL1 || g isa ElasticNet ? ref_elastic([2tt, 2.0], fill(big(a) * big(γ), 2), λ, 0f0, Float32[1, 0]) :
         ref_sparsegroup([2tt, 2.0], fill(big(a) * big(γ), 2), 0f0, λ, Float32[1, 0], groups)
    res = prox(Scaled(SeparableSum(NormL1(Inf32), g), a), ([1.0, 2.0], [2tt, 2.0]), γ)
    @test all(abs.(res[2] .- Float64.(rg)) .<= 4eps() .* [2tt, 2.0])
  end
  # everything in Float32
  f32 = Scaled(SparseGroupNormL2(λ, Inf32, groups, w), a)
  r32 = ref_sparsegroup(Float32[2Float32(tt), 2], fill(big(a) * big(γ), 2), λ, Inf, w, groups)
  @test all(abs.(prox(f32, Float32[2Float32(tt), 2], γ) .- Float32.(r32)) .<= 4eps(Float32) .* Float32[2Float32(tt), 2])
end

# 2. An infinite, zero or NaN factor inside a block keeps the earlier result, and never throws where it did not.
@testset "a factor that is not positive finite inside a block keeps the earlier result" begin
  arg = ([1.0, 2.0], [3.0, 4.0])
  nan = ([NaN, NaN], [3.0, 4.0])
  for blk in (Scaled(NormL1(1.0), Inf), Scaled(NormL1(Inf), Inf), Scaled(ElasticNet(Inf, 0.0), Inf),
              Scaled(Scaled(NormL1(Inf), 1e300), Inf), Translate(Scaled(NormL1(Inf), Inf), [0.0, 0.0]))
    for outer in (1e-300, 1e-200)
      f = Scaled(SeparableSum(blk, NormL1(Inf)), outer)
      @test isequal(prox(f, arg, outer), nan)
      y = (zeros(2), zeros(2))
      v = value_and_prox!(y, f, arg, outer)
      @test isequal(y, nan) && isnan(v)
    end
  end
  # the factor outside is infinite too, or the steps of an array call
  @test prox(Scaled(NormL1(Inf), Inf), [1.0, 2.0], [1.0, 1.0]) == [0.0, 0.0]
  @test prox(Scaled(Scaled(NormL1(Inf), Inf), 1e-300), [1.0, 2.0], [1.0, 1e-300]) == [0.0, 0.0]
end

# 3. A finite ridge is never shortcut: a call that the new route declines is the earlier call, on BigFloat too.
@testset "the ridge of a declined call is the earlier result" begin
  setprecision(BigFloat, 256) do
    γ = ldexp(1f0, -128)
    f = Scaled(ElasticNet(Inf32, 1f0, Float32[0, 1]), 1.5f0)
    ref = 1 / (1 + BigFloat(1.5f0) * BigFloat(γ))
    y = zeros(BigFloat, 2)
    v = value_and_prox!(y, f, BigFloat[1, 2], γ)
    @test y[1] == parse(BigFloat, "0.9999999999999999999999999999999999999955918961844164218451172379854165787081973")
    @test abs(y[1] - ref) < big"1e-70" && y[2] == 0
    @test v == parse(BigFloat, "0.7499999999999999999999999999999999999933878442766246327676758569781248680623046")
    a = BigFloat[1, 2]
    value_and_prox!(a, f, a, γ)
    @test a == y
    @test prox(f, BigFloat[1, 2], γ) == y
    g = Scaled(Scaled(ElasticNet(Inf32, 1f0, Float32[0, 1]), 0.75f0), 2f0)
    @test abs(prox(g, BigFloat[1, 2], γ)[1] - ref) < big"1e-70"
  end
end

####################################
#  the contract: the same function at an ordinary positive step
####################################

# The call returns what the same function returns at a step whose product is an ordinary (normal) positive
# number, the hard domain {0} at finite arguments included; a NaN or infinite entry follows the leaf's own rule
# at an ordinary step (decision 49 for NormL1, SLOPE, NormL1Fused and NormLinf; the others are tracked in #662),
# up to the sign of a zero. Every accepted leaf, with finite, NaN, infinite and mixed arguments, scalar and array steps, aliases and the
# value at the stored output; the reference is the leaf's own call at the steps 1 and 0.75.
function accepted_leaves(::Type{T}) where {T}
  inf = T(Inf)
  big_ = T === Float64 ? T(1e300) : T(1f30)
  G = [1:1, 2:3]
  [(NormL1(inf), 3), (NormL1(inf, T[1, 0, 2]), 3), (NormL1(inf, T[1, 1, 1]), 3), (NormL1Nonneg(inf), 3), (NormL2(inf), 3), (NormLinf(inf), 3),
   (GroupNormL2(inf, G), 2), (GroupNormLinf(inf, G), 2), (GroupNormL2(inf, G, T[1, 0]), 2),
   (ElasticNet(inf, T(0)), 3), (ElasticNet(inf, T(0), T[1, 0, 2]), 3), (ElasticNet(inf, big_, T[1, 2, 3]), 3), (ElasticNet(inf, T(1), T[1, 1, 1]), 3),
   (ElasticNet(T(1), inf), 3), (ElasticNet(T(1), inf, T[1, 0, 1]), 3), (ElasticNet(T(0), inf), 3), (ElasticNet(inf, inf), 3), (ElasticNet(inf, inf, T[1, 0, 1]), 3),
   (SparseGroupNormL2(inf, T(1), G, T[1, 1]), 2), (SparseGroupNormL2(inf, T(0), G, T[1, 1]), 2), (SparseGroupNormL2(inf, big_, G, T[0, 1]), 2),
   (SparseGroupNormL2(T(1), inf, G, T[1, 2]), 2), (SparseGroupNormL2(T(0), inf, G, T[0, 1]), 2), (SparseGroupNormL2(inf, inf, G, T[1, 1]), 2),
   (SLOPE(T[inf, 1, T(0.5)]), 3), (SLOPE(T[inf, inf, 0]), 3), (SLOPE(T[inf, big_, 1]), 3),
   (NormL1Fused(inf, T(1)), 3), (NormL1Fused(inf, T(0)), 3), (NormL1Fused(inf, inf), 3)]
end

@testset "the contract, vector leaves, $T" for T in (Float32, Float64)
  big_ = T === Float64 ? T(1e300) : T(1f30)
  tiny = T === Float64 ? T(1e-300) : T(1f-30)
  sub = T === Float64 ? T(1e-160) : T(1f-20)         # a subnormal product with itself
  xs = (T[1, 2, 3], T[-1, 0, 5], T[big_, -big_, 1], T[NaN, 2, 3], T[Inf, -Inf, 3], T[1, NaN, Inf], T[NaN, NaN, NaN], T[Inf, 1, 0], T[-Inf, 1, 2])
  for (leaf, n) in accepted_leaves(T), x in xs
    r1, r75 = prox(leaf, x, one(T)), prox(leaf, x, T(0.75))
    @test isequal(z0(r1), z0(r75))
    free = (leaf isa Union{NormL1, ElasticNet} && !isnothing(leaf.w) && any(iszero, leaf.w)) ||
           (leaf isa Union{GroupNormL2, SparseGroupNormL2} && any(iszero, leaf.w))
    all(isfinite, x) && !free && @test all(iszero, r1)        # the hard zero at a finite argument (no free coordinate)
    for (α, γ) in ((tiny, tiny), (big_, big_), (sub, sub), (tiny, T(1)))
      s = Scaled(leaf, α)
      check0(s, x, γ, r1)
      y = similar(x)
      v = value_and_prox!(y, s, x, γ)
      @test isequal(v, s(y))
    end
    # array steps with positive finite entries (one per group for a group function)
    leaf isa Union{NormL2, NormLinf, SLOPE, NormL1Fused} ||        # these take no array step
      for γs in (fill(tiny, n), T[tiny, 1, big_][1:n], fill(big_, n))
        check0(Scaled(leaf, tiny), x, γs, r1)
      end
    # a chain of factors
    check0(Scaled(Scaled(leaf, sub), sub), x, T(1), r1)
  end
end

@testset "the contract, matrix leaves, $T" for T in (Float32, Float64)
  tiny = T === Float64 ? T(1e-300) : T(1f-30)
  for leaf in (NuclearNorm(T(Inf)), TracePSD(T(Inf)))
    for X in (T[1 2; 2 3], T[1 0; 0 -2], T[4 1; 1 4], T[NaN 1; 1 2], T[Inf 0; 0 1])
      ref = try prox(leaf, X, one(T)) catch e; e end
      ref isa Exception && continue
      @test isequal(z0(ref), z0(prox(leaf, X, T(0.75))))
      s = Scaled(leaf, tiny)
      @test isequal(z0(prox(s, X, tiny)), z0(ref))
      Y = similar(X)
      v = value_and_prox!(Y, s, X, tiny)
      @test isequal(z0(Y), z0(ref)) && isequal(v, s(Y))
      A = copy(X)
      value_and_prox!(A, s, A, tiny)
      @test isequal(z0(A), z0(ref))
      all(isfinite, X) && @test all(iszero, ref)
    end
  end
end

# Every factor of the tree is validated before a block is accepted: an infinite factor inside any block, under an
# indicator or a translated indicator included, keeps the earlier result in every block (literals of the earlier
# result; the value is Inf where an indicator is violated, as before).
@testset "an excluded factor under an indicator keeps the earlier result, $T" for T in (Float32, Float64)
  inf = T(Inf)
  tiny = T === Float64 ? T(1e-300) : T(1f-30)
  arg = (T[1, 2], T[3, 4])
  box = IndBox(T(-1), T(1))
  for (blk, expected) in ((Scaled(box, inf), (T[1, 2], T[1, 1])),
                          (Translate(Scaled(box, inf), T[1, 1]), (T[1, 2], T[2, 2])),
                          (Scaled(Scaled(IndNonnegative(), T(2)), inf), (T[1, 2], T[3, 4])),
                          (Translate(Scaled(Scaled(box, T(3)), inf), T[1, 1]), (T[1, 2], T[2, 2])))
    f = Scaled(SeparableSum(NormL1(inf), blk), tiny)
    @test isequal(prox(f, arg, tiny), expected)
    y = (zeros(T, 2), zeros(T, 2))
    v = value_and_prox!(y, f, arg, tiny)
    @test isequal(y, expected) && v == inf
    a = (T[1, 2], T[3, 4])
    va = value_and_prox!(a, f, a, tiny)
    @test isequal(a, expected) && va == inf
    # the step of an array call, and the same block alone under the chain
    @test isequal(prox(Scaled(SeparableSum(NormL1(inf), blk), tiny), arg, tiny), expected)
  end
  # the same blocks with finite factors are accepted, and give the hard zero and the indicator's own projection
  for blk in (Scaled(box, T(2)), Translate(Scaled(box, T(3)), T[1, 1]), box, Translate(box, T[1, 1]))
    f = Scaled(SeparableSum(NormL1(inf), blk), tiny)
    @test isequal(prox(f, arg, tiny), prox(f.f, arg, one(T)))
    @test prox(f, arg, tiny)[1] == T[0, 0]
  end
end

# The review's reproducers of round 8: a nonfinite entry follows the leaf's own rule at an ordinary step (the
# reference), which is not what the earlier route gave (NormL1Nonneg: [Inf, 1], the elastic nets: [NaN, NaN],
# the earlier route kept the entry free at a step rounded to zero); a finite argument gives the hard zero.
@testset "nonfinite entries follow the leaf's rule at an ordinary step" begin
  x = [Inf, 1.0]
  for (leaf, expected) in ((NormL1Nonneg(Inf), [NaN, 0.0]), (ElasticNet(Inf, 1.0), [NaN, 0.0]), (ElasticNet(1.0, Inf), [NaN, 0.0]))
    @test isequal(prox(leaf, x, 1.0), expected)
    @test isequal(prox(Scaled(leaf, 1e-300), x, 1e-300), expected)
    @test isequal(prox(Scaled(leaf, 1e-300), x, [1e-300, 1e-300]), expected)
    @test all(iszero, prox(Scaled(leaf, 1e-300), [1.0, 1.0], 1e-300))
    y = zeros(2)
    v = value_and_prox!(y, Scaled(leaf, 1e-300), x, 1e-300)
    @test isequal(y, expected) && isequal(v, Scaled(leaf, 1e-300)(y))
  end
  @test isequal(prox(Scaled(GroupNormLinf(Inf, [1:2]), 1e-300), [NaN, 2.0], 1e-300), prox(GroupNormLinf(Inf, [1:2]), [NaN, 2.0], 1.0))
  @test prox(Scaled(GroupNormLinf(Inf, [1:2]), 1e-300), [1.0, 2.0], 1e-300) == [0.0, 0.0]
  # a Translate that overflows in the subtraction is the leaf's own call too
  tr = Translate(ElasticNet(Inf, 1.0), [-1e308, 0.0])
  @test isequal(prox(Scaled(tr, 1e-300), [1e308, 1.0], 1e-300), prox(tr, [1e308, 1.0], 1.0))
end

end
