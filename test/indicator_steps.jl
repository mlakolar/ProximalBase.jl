module IndicatorStepsTest

# #311, the part that the direct ball, sphere and simplex projections and the constrained ball and
# sphere need (decision 48, P1/E1/O6): a negative or NaN step is an ArgumentError before any write,
# and every step in [0, Inf] that the method supports keeps its result. The sphere rejects γ = Inf
# only when every free entry (group) is penalized.

using Test
using ProximalBase
using LinearAlgebra

const proximal = (prox!, value_and_prox!)
const SENTINEL = 7.0
# equal as numbers: the sign of a zero entry is not part of the result
same(a, b) = size(a) == size(b) && all(i -> a[i] == b[i], eachindex(a))

# the exception that op throws into a sentinel output y, or nothing
function thrown(op, f, y, x, γ)
  @nospecialize
  try
    op(y, f, x, γ)
  catch e
    return e
  end
  nothing
end

# op rejects γ with an ArgumentError, with a separate output and with the output aliasing the
# input, and leaves the output and the input as they were
function rejected(op, f, x, γ; message = "")
  @nospecialize
  y = fill(SENTINEL, size(x))
  x0 = copy(x)
  e = thrown(op, f, y, x, γ)
  ok = e isa ArgumentError && occursin(message, e.msg)
  ok || @show op f γ e
  @test ok
  @test isequal(y, fill(SENTINEL, size(x)))
  @test isequal(x, x0)
  z = copy(x)
  e = thrown(op, f, z, z, γ)
  @test e isa ArgumentError && occursin(message, e.msg)
  @test isequal(z, x0)
end

# the result at a supported step, into a sentinel output
function result(op, f, x, γ)
  @nospecialize
  y = fill(SENTINEL, size(x))
  op(y, f, copy(x), γ)
  y
end

const direct = (IndBallL0(1), IndBallL1(1.0), IndBallL2(1.0), IndSphereL2(1.0),
                IndSimplex(1.0), IndSimplex(1.0; equality = false), IndSimplex(0.0))
const groups = [[1, 2], [3]]
const constrained = (
  Constrained(NormL1(0.5), IndBallL2(1.0)),
  Constrained(NormL1(0.5, [1.0, 2.0, 0.0]), IndBallL2(1.0)),
  Constrained(GroupNormL2(0.5, groups), IndBallL2(1.0)),
  Constrained(IndBallL0(1), IndBallL2(1.0)),
  Constrained(ZeroFunction(), IndBallL2(1.0)),
  Constrained(NormL1(0.5), IndSphereL2(1.0)),
  Constrained(GroupNormL2(0.5, groups), IndSphereL2(1.0)),
  Constrained(IndBallL0(1), IndSphereL2(1.0)),
)
const x3 = [3.0, -2.0, 0.5]

# steps that no projection accepts, of every real type
const invalid = (-1, -1.0, -1.0f0, -1 // 2, -1e-300, -floatmin(Float64), -Inf, -Inf32,
                 NaN, NaN32, big"-1.5", BigFloat(NaN), BigInt(-3))

@testset "negative and NaN steps are rejected before any write" begin
  @testset "$(nameof(typeof(f)))" for f in (direct..., constrained...)
    for op in proximal, γ in invalid
      rejected(op, f, x3, γ; message = "nonnegative")
    end
    # a NaN or infinite entry does not change which error comes first
    for op in proximal, γ in (-1.0, NaN)
      rejected(op, f, [NaN, Inf, 1.0], γ; message = "nonnegative")
      rejected(op, f, [-Inf, 0.0, NaN], γ; message = "nonnegative")
    end
    @test_throws ArgumentError prox(f, x3, -1.0)
    @test_throws ArgumentError prox(f, x3, NaN)
  end
  # the form of the predicate: NaN is rejected, not passed
  @test_throws ArgumentError prox!(zeros(2), IndBallL2(1.0), ones(2), NaN)
  # Float32 output and a Float64 input
  for f in (IndBallL2(1.0), IndSimplex(1.0), Constrained(NormL1(0.5), IndSphereL2(1.0)))
    y = fill(7.0f0, 3)
    @test_throws ArgumentError prox!(y, f, x3, NaN)
    @test_throws ArgumentError prox!(y, f, x3, -1.0f0)
    @test all(==(7.0f0), y)
  end
  # matrices, a symmetric input, and the sparse iterates
  M = [2.0 -1.0 0.5; -1.0 3.0 0.0; 0.5 0.0 -4.0]
  for f in (IndBallL0(2), IndBallL0(2; diagonal = false)), γ in (-1.0, NaN)
    Y = fill(SENTINEL, 3, 3)
    @test_throws ArgumentError prox!(Y, f, Symmetric(M), γ)
    @test all(==(SENTINEL), Y)
    S = SymmetricSparseIterate(3)
    @test_throws ArgumentError prox!(S, f, Symmetric(M, :L), γ)
    @test nactive(S) == 0
    @test_throws ArgumentError prox!(SparseIterate(3), f, SparseIterate([1.0, 0.0, 2.0]), γ)
  end
  for f in (IndBallL2(1.0), Constrained(NormL1(0.5), IndBallL2(1.0)), Constrained(NormL1(0.5), IndSphereL2(1.0)))
    Y = fill(SENTINEL, 3, 3)
    @test_throws ArgumentError prox!(Y, f, M, -1.0)
    @test all(==(SENTINEL), Y)
  end
end

@testset "supported steps keep their results" begin
  steps = (0, 0.0, 0.0f0, 1e-300, floatmin(Float64), 1, 1 // 2, 1e300, floatmax(Float64), Inf, Inf32)
  # an indicator projection does not depend on the step
  @testset "$(nameof(typeof(f)))" for f in direct
    for op in proximal
      ref = result(op, f, x3, 1.0)
      @test all(!=(SENTINEL), ref)
      for γ in steps
        @test isequal(result(op, f, x3, γ), ref)
      end
    end
  end
  # the values
  for (f, x, expected) in (
        (IndBallL2(1.0), [3.0, 4.0], [0.6, 0.8]),
        (IndSphereL2(1.0), [3.0, 4.0], [0.6, 0.8]),
        (IndSphereL2(1.0), [0.0, 0.0], [1.0, 0.0]),            # the zero point goes to the first entry
        (IndBallL1(1.0), [3.0, 0.0], [1.0, 0.0]),
        (IndSimplex(1.0), [3.0, 0.0], [1.0, 0.0]),
        (IndSimplex(1.0; equality = false), [-3.0, 0.5], [0.0, 0.5]),
        (IndSimplex(0.0), [3.0, 2.0], [0.0, 0.0]),
        (IndBallL0(1), [3.0, -4.0], [0.0, -4.0]),
      ), γ in (0.0, 1.0, Inf), op in proximal
    @test isapprox(result(op, f, x, γ), expected; rtol = 1e-14, atol = 1e-15)
  end
  # the zero-radius ball projects to zero at every step
  for γ in (0.0, 1.0, Inf)
    @test isequal(prox(IndBallL2(0.0), [3.0, 1.0], γ), [0.0, 0.0])
  end

  @testset "constrained ball: $(nameof(typeof(c.f)))" for c in constrained[1:5]
    for op in proximal
      # γ = 0 leaves the point of f at x, scaled onto the ball; an indicator projects first
      @test isequal(result(op, c, x3, 0), c.f isa IndBallL0 ? [1.0, 0.0, 0.0] : result(op, IndBallL2(1.0), x3, 1))
      # a finite step gives the same result at a Float32 or an Int step type
      @test isequal(result(op, c, x3, 2), result(op, c, x3, 2.0))
    end
  end
  # γ = Inf projects onto the minimizers of f: zero for a norm, the zero-weight entries of a weighted
  # norm, the whole ball for the zero function
  for op in proximal, c in (constrained[1], constrained[3])
    @test isequal(result(op, c, x3, Inf), zeros(3))
  end
  for op in proximal
    @test isequal(result(op, constrained[2], x3, Inf), [0.0, 0.0, 0.5])
  end
  for op in proximal
    @test isequal(result(op, constrained[5], x3, Inf), result(op, IndBallL2(1.0), x3, 1))
    @test isequal(result(op, constrained[4], x3, Inf), result(op, constrained[4], x3, 1e300))
  end

  @testset "constrained sphere at γ = 0" begin
    for op in proximal, c in constrained[6:7]
      @test isequal(result(op, c, x3, 0), result(op, IndSphereL2(1.0), x3, 1))
    end
    for op in proximal
      @test isequal(result(op, constrained[8], x3, 0), [1.0, 0.0, 0.0])
      @test isequal(result(op, constrained[8], x3, Inf), [1.0, 0.0, 0.0])
    end
  end
end

@testset "penalized sphere at γ = Inf" begin
  sphere = IndSphereL2(1.0)
  # zero level: the projection onto the sphere
  for op in proximal
    @test same(result(op, Constrained(NormL1(0.0), sphere), x3, Inf), result(op, sphere, x3, 1))
    @test same(result(op, Constrained(NormL1(0, [1.0, 2.0, 3.0]), sphere), x3, Inf), result(op, sphere, x3, 1))
    @test same(result(op, Constrained(GroupNormL2(0.0, groups), sphere), x3, Inf), result(op, sphere, x3, 1))
    # a Float32 level and output: the same as the projection onto the sphere at the same types
    y32, z32 = fill(7.0f0, 3), fill(7.0f0, 3)
    op(y32, Constrained(NormL1(0.0f0), sphere), Float32.(x3), Inf)
    op(z32, sphere, Float32.(x3), 1)
    @test isapprox(y32, z32; rtol = 4eps(Float32))
  end
  # a zero-weight free entry: the sphere within the zero-penalty entries, along the sign of x
  for op in proximal
    @test same(result(op, Constrained(NormL1(1.0, [1.0, 0.0, 1.0]), sphere), x3, Inf), [0.0, -1.0, 0.0])
    @test same(result(op, Constrained(NormL1(1.0, [1.0, 1.0, 0.0]), sphere), x3, Inf), [0.0, 0.0, 1.0])
    # two zero-weight entries: the projection onto the sphere of their subspace
    @test isapprox(result(op, Constrained(NormL1(1.0, [0.0, 1.0, 0.0]), sphere), x3, Inf), [3.0, 0.0, 0.5] / norm([3.0, 0.5]); rtol = 1e-14, atol = 1e-15)
    # a group with zero weight, and an entry in no group
    @test same(result(op, Constrained(GroupNormL2(1.0, groups, [1.0, 0.0]), sphere), x3, Inf), [0.0, 0.0, 1.0])
    @test same(result(op, Constrained(GroupNormL2(1.0, [[1, 2]], [1.0]), sphere), x3, Inf), [0.0, 0.0, 1.0])
  end
  # every free entry (group) penalized: no named endpoint, an error before any write
  for f in (NormL1(1.0), NormL1(1.0, [1.0, 2.0, 3.0]), GroupNormL2(1.0, groups), GroupNormL2(1.0, groups, [1.0, 1.0]),
            GroupNormL2(0.5, [[1], [2], [3]]))
    for op in proximal
      rejected(op, Constrained(f, sphere), x3, Inf; message = "sphere thresholds must be finite")
    end
  end
  # a finite step with an overflowing threshold is the same error as before
  for op in proximal
    rejected(op, Constrained(NormL1(floatmax(Float64)), sphere), x3, floatmax(Float64); message = "sphere thresholds must be finite")
  end
  # and γ = Inf stays a step of the projection onto the penalized ball
  for op in proximal
    @test same(result(op, Constrained(NormL1(1.0), IndBallL2(1.0)), x3, Inf), zeros(3))
    @test same(result(op, Constrained(NormL1(0.0), IndBallL2(1.0)), x3, Inf), result(op, IndBallL2(1.0), x3, 1))
  end
end

@testset "a later block's step is checked before an earlier block is written" begin
  first_block = IndNonnegative()
  sphere = IndSphereL2(1.0)
  later = (
    (IndBallL2(1.0), -1.0), (IndSimplex(1.0), NaN), (IndBallL1(1.0), -Inf), (IndSphereL2(1.0), -1.0),
    (IndBallL0(1), NaN), (Constrained(NormL1(0.5), IndBallL2(1.0)), -1.0),
    (Constrained(GroupNormL2(0.5, groups), sphere), NaN), (Constrained(IndBallL0(1), sphere), -1.0),
    (Constrained(NormL1(1.0), sphere), Inf), (Constrained(GroupNormL2(1.0, groups), sphere), Inf),
  )
  @testset "$(nameof(typeof(g)))" for (g, γ) in later
    s = SeparableSum(first_block, g)
    for op in proximal
      x = (copy(x3), copy(x3))
      y = (fill(SENTINEL, 3), fill(SENTINEL, 3))
      @test thrown(op, s, y, x, γ) isa ArgumentError
      @test all(==(SENTINEL), y[1]) && all(==(SENTINEL), y[2])
      @test x[1] == x3 && x[2] == x3
      # the output aliasing the input
      z = (copy(x3), copy(x3))
      @test thrown(op, s, z, z, γ) isa ArgumentError
      @test z[1] == x3 && z[2] == x3
    end
  end
  # the supported endpoints pass through the sum
  s = SeparableSum(first_block, Constrained(NormL1(0.0), sphere))
  y = (zeros(3), zeros(3))
  prox!(y, s, (copy(x3), copy(x3)), Inf)
  @test y[1] == max.(x3, 0) && isequal(y[2], result(prox!, sphere, x3, 1))
  s = SeparableSum(first_block, Constrained(NormL1(1.0, [1.0, 0.0, 1.0]), sphere))
  prox!(y, s, (copy(x3), copy(x3)), Inf)
  @test y[2] == [0.0, -1.0, 0.0]
  s = SeparableSum(IndBallL2(1.0), IndSimplex(1.0))
  for γ in (0.0, Inf)
    prox!(y, s, (copy(x3), copy(x3)), γ)
    @test isequal(y[1], result(prox!, IndBallL2(1.0), x3, 1)) && isequal(y[2], result(prox!, IndSimplex(1.0), x3, 1))
  end
end

@testset "wrappers pass their step on" begin
  for f in (IndBallL2(1.0), IndSimplex(1.0), Constrained(NormL1(0.5), IndSphereL2(1.0)))
    for op in proximal
      rejected(op, Scaled(f, 2.0), x3, -1.0; message = "nonnegative")
      rejected(op, Scaled(f, 2.0), x3, NaN; message = "nonnegative")
      rejected(op, Translate(f, [0.1, 0.2, 0.3]), x3, -1.0; message = "nonnegative")
      rejected(op, Regularize(f, 0.5), x3, NaN; message = "nonnegative")
    end
  end
  # a supported scalar endpoint through a wrapper
  @test isequal(prox(Scaled(IndBallL2(1.0), 2.0), x3, 0.0), prox(IndBallL2(1.0), x3))
  @test isequal(prox(Scaled(IndBallL2(1.0), 2.0), x3, Inf), prox(IndBallL2(1.0), x3))
end


# the stored point and the returned value of value_and_prox!
function stored(f, x, γ, ::Type{T} = Float64) where {T}
  @nospecialize
  y = fill(T(SENTINEL), size(x))
  v = value_and_prox!(y, f, convert(Vector{T}, x), γ)
  y, v
end

@testset "value_and_prox! returns the value at the stored point (O2)" begin
  sphere, ball = IndSphereL2(1.0), IndBallL2(1.0)
  # an indicator returns zero in the output's precision, at every supported step
  for f in direct, γ in (0, 0.0, 1.0, 1e300, Inf), T in (Float32, Float64)
    y, v = stored(f, x3, γ, T)
    @test v === zero(T)
    @test all(!=(T(SENTINEL)), y)
  end
  # a penalized function: the value of f at the stored point y, as the constrained function evaluates it
  cases = (
    (Constrained(NormL1(0.5), ball), (0, 1.0, 1e-3, Inf)),
    (Constrained(NormL1(0.5, [1.0, 2.0, 0.0]), ball), (0, 1.0, Inf)),
    (Constrained(GroupNormL2(0.5, groups), ball), (0, 1.0, Inf)),
    (Constrained(IndBallL0(1), ball), (0, 1.0, Inf)),
    (Constrained(ZeroFunction(), ball), (0, 1.0, Inf)),
    (Constrained(NormL1(0.5), sphere), (0, 1.0, 1e-3)),
    (Constrained(GroupNormL2(0.5, groups), sphere), (0, 1.0, 1e-3)),
    (Constrained(IndBallL0(1), sphere), (0, 1.0, Inf)),
    # γ = Inf on the sphere: a zero level, a zero-weight entry, a zero-weight group, an ungrouped entry
    (Constrained(NormL1(0.0), sphere), (0, 1.0, Inf)),
    (Constrained(NormL1(1.0, [1.0, 0.0, 1.0]), sphere), (Inf,)),
    (Constrained(GroupNormL2(0.0, groups), sphere), (Inf,)),
    (Constrained(GroupNormL2(1.0, groups, [1.0, 0.0]), sphere), (Inf,)),
    (Constrained(GroupNormL2(1.0, [[1, 2]], [1.0]), sphere), (Inf,)),
  )
  @testset "$(nameof(typeof(c.f)))" for (c, steps) in cases, γ in steps, T in (Float64, Float32)
    y, v = stored(c, x3, γ, T)
    @test v isa T
    @test isapprox(v, T(c(y)); rtol = 8eps(T), atol = 8eps(T))
    c.f isa Union{IndBallL0, ZeroFunction} || @test isapprox(v, T(c.f(y)); rtol = 8eps(T), atol = 8eps(T))
  end
  # values that follow from the penalty alone
  for (c, expected) in (
        (Constrained(NormL1(0.5), ball), 0.0),                                     # γ = Inf: zero
        (Constrained(NormL1(0.5, [1.0, 2.0, 0.0]), ball), 0.0),                    # the zero weight is on the free entry
        (Constrained(NormL1(0.0), sphere), 0.0),
        (Constrained(NormL1(1.0, [1.0, 0.0, 1.0]), sphere), 0.0),
        (Constrained(GroupNormL2(1.0, groups, [1.0, 0.0]), sphere), 0.0),
      )
    @test stored(c, x3, Inf)[2] == expected
  end
end

@testset "Scaled indicators at a zero step" begin
  # the scale of an indicator does not matter, so a zero step is a step for any scale, also α = Inf
  for f in (IndBallL2(1.0), IndSimplex(1.0), IndSphereL2(1.0), IndBallL1(1.0), IndBallL0(1), Constrained(IndBallL0(1), IndBallL2(1.0))),
      α in (0.5, 2.0, 1e300, Inf)
    s = Scaled(f, α)
    for op in proximal
      ref = result(op, f, x3, 1)
      @test same(result(op, s, x3, 0), ref)
      @test same(result(op, s, x3, 1), ref)
      @test same(result(op, s, x3, Inf), ref)
      rejected(op, s, x3, -1; message = "nonnegative")
      rejected(op, s, x3, NaN; message = "nonnegative")
    end
    # through a sum, where the step is checked first
    sum2 = SeparableSum(IndNonnegative(), s)
    y = (zeros(3), zeros(3))
    prox!(y, sum2, (copy(x3), copy(x3)), 0)
    @test same(y[2], result(prox!, f, x3, 1))
    y = (fill(SENTINEL, 3), fill(SENTINEL, 3))
    @test thrown(prox!, sum2, y, (copy(x3), copy(x3)), -1) isa ArgumentError
    @test all(==(SENTINEL), y[1]) && all(==(SENTINEL), y[2])
  end
  # a function that is not an indicator keeps the indeterminate Inf * 0 as an error, before any write
  c = Constrained(NormL1(0.5), IndBallL2(1.0))
  for op in proximal
    rejected(op, Scaled(c, Inf), x3, 0; message = "nonnegative")
  end
  # the product is unchanged for a finite scale
  @test same(result(prox!, Scaled(c, 2.0), x3, 0.25), result(prox!, c, x3, 0.5))
end

@testset "a bad shape together with a bad step writes nothing" begin
  # which error comes first is not part of the contract; neither leaves a write
  for f in (direct..., constrained...), γ in (-1.0, NaN)
    y = fill(SENTINEL, 2)
    e = thrown(prox!, f, y, ones(3), γ)
    @test e isa Union{ArgumentError, DimensionMismatch}
    @test all(==(SENTINEL), y)
    e = thrown(value_and_prox!, f, y, ones(3), γ)
    @test e isa Union{ArgumentError, DimensionMismatch}
    @test all(==(SENTINEL), y)
  end
  s = SeparableSum(IndNonnegative(), IndBallL2(1.0))
  y = (fill(SENTINEL, 3), fill(SENTINEL, 2))
  @test thrown(prox!, s, y, (copy(x3), copy(x3)), -1.0) isa Union{ArgumentError, DimensionMismatch}
  @test all(==(SENTINEL), y[1]) && all(==(SENTINEL), y[2])
end


# a throwing call leaves every block as it was, with separate outputs and, where the output has the
# type of the input, with outputs aliasing the inputs (an aliased output takes the input's type, so
# for another destination type the aliased call is a different computation)
function blocks_unchanged(op, s, x, γ, ::Type{T}) where {T}
  @nospecialize
  y = (fill(T(SENTINEL), length(x[1])), fill(T(SENTINEL), length(x[2])))
  x0 = map(copy, x)
  thrown(op, s, y, map(copy, x), γ) isa ArgumentError || return false
  all(==(T(SENTINEL)), y[1]) && all(==(T(SENTINEL)), y[2]) || return false
  eltype(x[1]) === T || return true
  z = map(copy, x)
  thrown(op, s, z, z, γ) isa ArgumentError && z[1] == x0[1] && z[2] == x0[2]
end

@testset "group sphere: the preflight takes the kernel's threshold, at mixed precisions" begin
  # Float32 level, weight and step with a Float64 destination: the product γλ0w is formed in Float64,
  # so 1e20 ⋅ 1e20 ⋅ 1e20 is finite for the kernel and for the preflight alike, and nothing throws.
  f32 = GroupNormL2(1.0f20, [[1, 2]], Float32[1.0f20])
  c = Constrained(f32, IndSphereL2(1.0))
  s = SeparableSum(IndNonnegative(), c)
  for op in proximal
    y = (fill(SENTINEL, 2), fill(SENTINEL, 2))
    op(y, s, ([-1.0, 2.0], [3.0, 4.0]), 1.0f20)
    @test y[1] == [0.0, 2.0] && isapprox(y[2], [0.6, 0.8]; rtol = 1e-14)
    z = fill(SENTINEL, 2)
    op(z, c, [3.0, 4.0], 1.0f20)
    @test z == y[2]
  end
  # the same Float32 factors with a Float32 destination overflow in Float32 and throw, before any write
  for op in proximal
    @test blocks_unchanged(op, s, (Float32[-1, 2], Float32[3, 4]), 1.0f20, Float32)
    # and an infinite Float32 step, with a Float64 destination and Float64 inputs
    @test blocks_unchanged(op, SeparableSum(IndNonnegative(), Constrained(GroupNormL2(1.0f0, [[1, 2]], Float32[1]), IndSphereL2(1.0))),
                           ([-1.0, 2.0], [3.0, 4.0]), Inf32, Float64)
    @test blocks_unchanged(op, SeparableSum(IndNonnegative(), Constrained(NormL1(1.0f0, Float32[1, 2]), IndSphereL2(1.0))),
                           ([-1.0, 2.0], [3.0, 4.0]), Inf32, Float64)
  end
  # every precision of destination, input, level and step, from ordinary to overflowing magnitudes:
  # the preflight throws exactly when the kernel does, and a sum then writes no block
  for Ty in (Float32, Float64), Tx in (Float32, Float64), Tc in (Float32, Float64), Tγ in (Float32, Float64),
      mag in (1.0, 1e20, 1e35, 1e39, 1e60, 1e300)
    big = min(mag, floatmax(Tc))
    for (g, n) in ((Constrained(GroupNormL2(Tc(big), [[1, 2]], Tc[big]), IndSphereL2(1.0)), 2),
                   (Constrained(GroupNormL2(Tc(big), [[1, 2], [3]], Tc[big, 1]), IndSphereL2(1.0)), 3),
                   (Constrained(NormL1(Tc(big), Tc[big, 1, 1]), IndSphereL2(1.0)), 3))
      γ = Tγ(min(mag, floatmax(Tγ)))
      x = Tx[3, -4, 1][1:n]
      for op in proximal
        direct = thrown(op, g, fill(Ty(SENTINEL), n), copy(x), γ)
        @test direct === nothing || direct isa ArgumentError
        pre = try; ProximalBase._checkstep(op, g, fill(Ty(SENTINEL), n), x, γ); nothing; catch e; e; end
        @test (pre === nothing) == (direct === nothing)
        sum2 = SeparableSum(IndNonnegative(), g)
        if direct === nothing
          y = (fill(Ty(SENTINEL), n), fill(Ty(SENTINEL), n))
          @test thrown(op, sum2, y, (copy(x), copy(x)), γ) === nothing
        else
          @test blocks_unchanged(op, sum2, (copy(x), copy(x)), γ, Ty)
        end
      end
    end
  end
end

end
