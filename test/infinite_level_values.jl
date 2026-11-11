module InfiniteLevelValuesTest

# E4 and O2 for an infinite level (#621). With λ = Inf the domain of NormL2, NormLinf and NuclearNorm
# is {0}: the value is 0 at the zero vector or matrix, Inf at any other argument without a NaN entry,
# and NaN at a NaN entry (E2). Scaled(f, Inf) has the value 0 where f is certainly 0 (an indicator inside
# its set, the level families at an argument that is zero where they penalize), and NaN at any other
# computed zero, which may be a positive value that underflowed. The value returned by
# value_and_prox! is the value at the stored output (O2). The infinite level is decided by the stored
# level, never by a converted level or by a product: a finite level such as 2^128 on Float32 data
# keeps its finite value (R3). The controls at the end pass before and after the change.

using Test, ProximalBase, LinearAlgebra, SparseArrays
using JLArrays: JLArray
using GPUArraysCore: allowscalar

const PB = ProximalBase

# the same value and type; NaN equals NaN, and 0.0 differs from -0.0
same(a, b) = isequal(a, b) && typeof(a) == typeof(b)

# value_and_prox! into an output filled with a sentinel, so that every entry is checked
function vp(f, x, γ)
  y = fill!(similar(x), -7)
  v = value_and_prox!(y, f, x, γ)
  v, y
end

# the value and the output are (0, zeros) through every route of the proximal operator
function check_zero_prox(f, x, γ)
  T = float(eltype(x))
  v, y = vp(f, x, γ)
  @test same(v, zero(T))
  @test isequal(Array(y), zeros(T, size(x)))
  @test same(f(y), zero(T))                                   # O2
  y = fill!(similar(x), -7)
  @test prox!(y, f, x, γ) === y
  @test isequal(Array(y), zeros(T, size(x)))
  z = copy(x)
  @test same(value_and_prox!(z, f, z, γ), zero(T))            # in place
  @test isequal(Array(z), zeros(T, size(x)))
  @test isequal(Array(prox(f, x, γ)), zeros(T, size(x)))
end

zero_arguments(T) = (zeros(T, 1), zeros(T, 2), zeros(T, 0), zeros(T, 2, 2), spzeros(T, 3),
                     SymmetricSparseIterate(T, 3), BlockIterate(zeros(T, 4), GroupNormL2(one(T), [1:2, 3:4])))
nonzero_arguments(T) = (T[1, 0], T[floatmin(T) / 4, 0], T[0, Inf], T[0, -Inf], T[floatmax(T), floatmax(T)], T[0 0; 0 -2])
nan_arguments(T) = (T[0, NaN], T[1, NaN], T[NaN 0; 0 0])

@testset "NormL2 and NormLinf with an infinite level: values" begin
  allowscalar(false)
  for T in (Float32, Float64), F in (NormL2, NormLinf), λ in (T(Inf), Inf)
    f = F(λ)
    for x in zero_arguments(T)
      x isa SymmetricSparseIterate && F === NormLinf && continue      # the value of NormLinf at an owner is not defined
      @test same(f(x), zero(T))
    end
    for x in nonzero_arguments(T)
      @test same(f(x), T(Inf))
    end
    for x in nan_arguments(T)
      @test same(f(x), T(NaN))
    end
    # device arrays (the device ℓ∞ value has its own method)
    @test same(f(JLArray(zeros(T, 2))), zero(T))
    @test same(f(JLArray(zeros(T, 0))), zero(T))
    @test same(f(JLArray(T[1, 0])), T(Inf))
    @test same(f(JLArray(T[0, NaN])), T(NaN))
  end
end

@testset "NormL2 with an infinite level: the value at the stored output" begin
  # the output is zero at every step here, also at γ = 0 and NaN (O2 only for those)
  for T in (Float32, Float64), x in (T[3, 4], T[1, NaN], zeros(T, 2), T[Inf, 0]),
      γ in (one(T), T(Inf), floatmin(T), zero(T), T(NaN))
    f = NormL2(T(Inf))
    v, y = vp(f, x, γ)
    @test y == zeros(T, 2)
    @test same(v, f(y))                                       # O2: the value at the zero output is 0
  end
end

# NormLinf(Inf) follows its own rule (the proximal map is zero at a finite positive step, decision of
# #663); these rows check that rule here, beside the values of this file, and pass with it.
@testset "controls: NormLinf with an infinite level at a finite positive step" begin
  allowscalar(false)
  for T in (Float32, Float64)
    for x in (T[1, NaN], T[NaN, NaN], T[NaN, 2, 3], T[Inf, NaN], T[NaN 1; 1 0]),
        γ in (one(T), floatmin(T), nextfloat(zero(T)), 1e-300, 2.0^200)
      check_zero_prox(NormLinf(T(Inf)), x, γ)
      check_zero_prox(NormLinf(Inf), x, γ)                    # a Float64 level on Float32 data
    end
    # a symmetric output
    A = T[1 2 0; 2 NaN 0; 0 0 3]
    y = SymmetricSparseIterate(T, 3)
    fill!(coordinates(y), -7)
    @test_throws MethodError value_and_prox!(y, NormLinf(T(Inf)), A, one(T))     # NormLinf has no owner method
    @test same(value_and_prox!(y, NormL2(T(Inf)), A, one(T)), zero(T))
    @test Matrix(y) == zeros(T, 3, 3)
    # device arrays
    for x in (T[1, NaN], T[NaN, NaN])
      v, y = vp(NormLinf(T(Inf)), JLArray(x), one(T))
      @test same(v, zero(T))
      @test isequal(Array(y), zeros(T, 2))
    end
    # an infinite step is not covered by that rule and keeps its earlier result
    for d in (identity, JLArray)
      v, y = vp(NormLinf(T(Inf)), d(T[1, NaN]), T(Inf))
      @test same(v, T(-Inf))
      @test isequal(Array(y), T[-Inf, NaN])
    end
  end
end

@testset "NuclearNorm with an infinite level: values" begin
  allowscalar(false)
  for T in (Float32, Float64), λ in (T(Inf), Inf), partial in (false, true)
    f = NuclearNorm(λ; partial)
    for X in (zeros(T, 2, 2), zeros(T, 0, 0), zeros(T, 3, 2), zeros(T, 0, 3), SymmetricSparseIterate(T, 3))
      @test same(f(X), zero(T))
    end
    M = floatmax(T) * T(0.75)
    for X in (T[1 0; 0 0], T[floatmin(T) / 4 0; 0 0], T[M M; M M])   # the last one's spectrum overflows
      @test same(f(X), T(Inf))
    end
    @test same(f(JLArray(zeros(T, 2, 2))), zero(T))
  end
end

@testset "NuclearNorm with an infinite level: the value at the stored zero output" begin
  allowscalar(false)
  for T in (Float32, Float64)
    M = floatmax(T) * T(0.75)
    Xo = T[M M; M M]
    # the overflowing route is taken: the input is finite and its spectrum is not
    ws = PB._svd_workspace(nothing, T, 2, 2)
    copyto!(ws.A, Xo)
    s, = PB._svd!(ws)
    @test all(isfinite, Xo) && PB._spectral_overflow(s, Xo, ws)
    for partial in (false, true), λ in (T(Inf), Inf),
        X in (T[3 0; 0 4], zeros(T, 2, 2), T.(reshape(1:36, 6, 6)), Xo), γ in (one(T), T(Inf))
      f = NuclearNorm(λ; partial)
      v, Y = vp(f, X, γ)
      @test same(v, zero(T))
      @test Y == zeros(T, size(X))
      @test same(f(Y), v)                                     # O2
    end
    v, Y = vp(NuclearNorm(T(Inf)), JLArray(T[3 0; 0 4]), one(T))
    @test same(v, zero(T))
    @test Array(Y) == zeros(T, 2, 2)
  end
end

# the families of the reproduction: the function, a point where it is 0, a point where it is positive
scaled_cases(T) = (
  (SparseGroupNormL2(one(T), one(T), [1:2], [one(T)]), T[0, 0, 9], T[1, 0, 9]),
  (GroupNormL2(one(T), [1:2], [one(T)]), T[0, 0, 9], T[1, 0, 9]),
  (NormL2(one(T)), zeros(T, 2), T[1, 0]),
  (NormL1(one(T)), zeros(T, 2), T[floatmin(T) / 4, 0]),
  (NormLinf(one(T)), zeros(T, 2), T[1, 0]),
  (NuclearNorm(one(T)), zeros(T, 2, 2), T[1 0; 0 0]),
  (ElasticNet(one(T), one(T)), zeros(T, 2), T[1, 0]),
  (IndNonnegative(), T[1, 2], T[-1, 2]),
  (IndBallL2(one(T)), zeros(T, 2), T[2, 0]),
)

@testset "Scaled with an infinite factor: values" begin
  allowscalar(false)
  for T in (Float32, Float64)
    for (f, x0, x1) in scaled_cases(T)
      s = Scaled(f, T(Inf))
      @test same(s(x0), zero(T))
      @test same(s(x1), T(Inf))
      @test same(Scaled(f, Inf)(x0), zero(T))                # a Float64 factor
      xn = copy(x1)
      xn[1] = T(NaN)
      fn = f(xn)                                              # NaN, or Inf off a domain
      @test same(s(xn), iszero(fn) ? zero(T) : T(Inf) * fn)
    end
    @test same(Scaled(IndBallL2(one(T)), T(Inf))(T[0.5, 0]), zero(T))   # inside the ball
    # factors that merge at construction
    for s in (Scaled(Scaled(NormL1(one(T)), T(Inf)), T(2)), Scaled(Scaled(NormL1(one(T)), T(2)), T(Inf)))
      @test s isa Scaled{<:NormL1}
      @test same(s(zeros(T, 2)), zero(T))
    end
    # nestings that do not merge: the infinite factor inside the translation decides on its own
    # function; an infinite factor over a wrapper is not certain of a zero, which stays NaN
    o = zeros(T, 2)
    s = Scaled(Translate(Scaled(NormL1(one(T)), T(Inf)), o), T(2))
    @test same(s(zeros(T, 2)), zero(T))
    @test same(s(T[1, 0]), T(Inf))
    @test same(Translate(Scaled(NormL1(one(T)), T(Inf)), T[1, 1])(T[1, 1]), zero(T))
    s = Scaled(Translate(Scaled(NormL1(one(T)), T(2)), o), T(Inf))
    @test same(s(zeros(T, 2)), T(NaN))
    @test same(s(T[1, 0]), T(Inf))
    # tuples (a wrapper: NaN at zero), iterates and device arrays
    sep = Scaled(SeparableSum(NormL2(one(T)), NormL1(one(T))), T(Inf))
    @test same(sep((zeros(T, 2), zeros(T, 2))), T(NaN))
    @test same(sep((T[1, 0], zeros(T, 2))), T(Inf))
    @test same(Scaled(NormL1(one(T)), T(Inf))(SymmetricSparseIterate(T, 3)), zero(T))
    g = GroupNormL2(one(T), [1:2, 3:4])
    @test same(Scaled(g, T(Inf))(BlockIterate(zeros(T, 4), g)), zero(T))
    @test same(Scaled(NormL2(one(T)), T(Inf))(JLArray(zeros(T, 2))), zero(T))
    @test same(Scaled(NormL1(one(T)), T(Inf))(JLArray(zeros(T, 2))), zero(T))
  end
end

@testset "Scaled with an infinite factor: value_and_prox! and value_and_gradient!" begin
  for T in (Float32, Float64)
    x = T[3, 4]
    for f in (NormL2(one(T)), NormL1(one(T)), NormLinf(one(T)), GroupNormL2(one(T), [1:2], [one(T)]), ElasticNet(one(T), one(T)))
      check_zero_prox(Scaled(f, T(Inf)), x, one(T))
    end
    check_zero_prox(Scaled(NuclearNorm(one(T)), T(Inf)), T[3 0; 0 4], one(T))
    # an array step (its own method) and nestings through Translate, with scalar and array steps
    o = zeros(T, 2)
    for (s, want) in ((Scaled(NormL1(one(T)), T(Inf)), zero(T)), (Scaled(Translate(Scaled(NormL1(one(T)), T(Inf)), o), T(2)), zero(T)),
                      (Scaled(Translate(Scaled(NormL1(one(T)), T(2)), o), T(Inf)), T(NaN)))
      for γ in (one(T), T[1, 1])
        v, y = vp(s, x, γ)
        @test same(v, want)
        @test y == zeros(T, 2)
        @test same(s(y), v)
      end
    end
    # an indicator: the projection, at which the value is 0
    s = Scaled(IndBallL2(one(T)), T(Inf))
    v, y = vp(s, x, one(T))
    @test y ≈ T[0.6, 0.8]
    @test same(v, zero(T))
    # tuples and iterates
    sep = Scaled(SeparableSum(NormL2(one(T)), NormL1(one(T))), T(Inf))
    y = (zeros(T, 2), zeros(T, 2))
    @test same(value_and_prox!(y, sep, (T[3, 4], T[1, 2]), one(T)), T(NaN))   # a wrapper
    @test y == (zeros(T, 2), zeros(T, 2))
    g = GroupNormL2(one(T), [1:2, 3:4])
    a = BlockIterate(T[1, 2, 3, 4], g)
    y = copy(a)
    @test same(value_and_prox!(y, Scaled(g, T(Inf)), a, one(T)), zero(T))
    @test y.storage == zeros(T, 4)
    # value_and_gradient! uses the same rule: SqrNormL2 is not certain of its zero (NaN, as f(x)),
    # and the gradient is as before (Inf⋅0 at a zero entry)
    s = Scaled(SqrNormL2(one(T)), T(Inf))
    g = fill(T(-7), 2)
    v = value_and_gradient!(g, s, zeros(T, 2))
    @test same(v, T(NaN))
    @test same(v, s(zeros(T, 2)))
    @test isequal(g, gradient(s, zeros(T, 2)))
    @test isequal(g, T[NaN, NaN])
  end
end

# A computed zero is not an exact zero: a positive value that underflowed has the exact scaled value
# Inf. The value is 0 only where f is certainly 0 at the argument; elsewhere a computed zero keeps the
# value it has without this rule: NaN, or Inf where the value of f is recovered from its significand
# and exponent (the scaled-value mechanism of functions without a level). These pass before and after.
nonzero_tiny(T) = (T[1, 2], T[1.0f-23, 0], T[floatmin(T) / 4, 0])
tiny_families(T, λ) = (NormL1(λ), NormL1(λ, ones(T, 2)), NormL1Nonneg(λ), NormL2(λ), NormLinf(λ),
                       ElasticNet(λ, λ), GroupNormL2(λ, [1:2], [1.0]), GroupNormLinf(λ, [1:2], [1.0]),
                       SparseGroupNormL2(λ, λ, [1:2], [1.0]))
never_false_zero(v, ::Type{T}) where {T} = (isnan(v) || v === T(Inf)) && v isa T

@testset "controls: no zero where the value underflowed under an infinite factor" begin
  allowscalar(false)
  T = Float32
  for λ in (1e-300, 2.0^-130), f in tiny_families(T, λ), x in nonzero_tiny(T)
    s = Scaled(f, Inf)
    @test never_false_zero(s(x), T)
    @test never_false_zero(s(JLArray(x)), T)
    v, y = vp(s, x, T(1))
    @test same(v, s(y))                                       # O2 at the stored output
  end
  for λ in (1e-300, 2.0^-130)
    s = Scaled(NuclearNorm(λ), Inf)
    for X in (T[1 0; 0 2], T[1.0f-23 0; 0 0], T[floatmin(T) / 4 0; 0 0])
      @test never_false_zero(s(X), T)
    end
  end
  # the reported cases
  @test same(Scaled(NormL1(1e-300), Inf)(Float32[1, 2]), NaN32)
  @test same(Scaled(NormL2(1e-300), Inf)(Float32[1, 2]), NaN32)
  @test same(Scaled(NormL1(1e-300), Inf)([1e-30, 0.0]), NaN)
  @test same(Scaled(NormL1(1e-300), 1e300)(Float32[1, 2]), 3.0f0)          # a finite factor recovers it
  # a function that is not certain of its zero: its value recovered from the significand and the
  # exponent where it underflowed (Inf), NaN at the zero vector and on a device
  for (s, x, want) in ((Scaled(SqrNormL2(1.0f0), Inf), Float32[1.0f-23, 0], Inf32), (Scaled(SqrNormL2(1.0), Inf), [1e-170, 0.0], Inf),
                       (Scaled(SqrNormL2(1.0f0), Inf), zeros(Float32, 2), NaN32))
    @test same(s(x), want)
    g = fill(eltype(x)(-7), 2)
    @test same(value_and_gradient!(g, s, x), want)
    @test isequal(g, gradient(s, x))
  end
  @test same(Scaled(SqrNormL2(1.0f0), Inf)(JLArray(Float32[1.0f-23, 0])), NaN32)
  @test same(Scaled(Scaled(Scaled(SqrNormL2(1.0), 2.0^600), 2.0^600), Inf)([1e-300, 0.0]), Inf)
  # a zero level folded with the factor keeps its own value (TracePSD has no zero certificate)
  @test same(Scaled(TracePSD(0.0), Inf)([1.0 0; 0 1]), 0.0)
end

@testset "Scaled with an infinite factor: the certain zeros of each family" begin
  allowscalar(false)
  for T in (Float32, Float64)
    for f in tiny_families(T, one(T))
      s = Scaled(f, T(Inf))
      @test same(s(zeros(T, 2)), zero(T))
      @test same(s(JLArray(zeros(T, 2))), zero(T))
      @test never_false_zero(s(T[0, NaN]), T)
      check_zero_prox(s, T[3, 4], one(T))
      if isseparable(f)
        v, y = vp(s, T[3, 4], T[1, 1])                        # the array step
        @test same(v, zero(T))
        @test y == zeros(T, 2)
      end
    end
    s = Scaled(NuclearNorm(one(T)), T(Inf))
    @test same(s(zeros(T, 2, 2)), zero(T))
    @test same(s(JLArray(zeros(T, 2, 2))), zero(T))
    # entries left free: a zero weight, a zero level with a finite weight, an index in no group
    for (f, x, want) in ((NormL1(one(T), T[1, 0]), T[0, 5], zero(T)),
                         (NormL1(one(T), T[1, 0]), T[0, NaN], zero(T)),  # a free entry may hold anything
                         (NormL1(one(T), T[1, 0]), T[0, Inf], zero(T)),
                         (GroupNormL2(one(T), [1:2]), T[0, 0, NaN], zero(T)),
                         (NormL1(zero(T), T[0, Inf]), T[5, 0], zero(T)),
                         (NormL1(zero(T), T[0, Inf]), T[0, 1], T(Inf)),
                         (NormL1Nonneg(one(T), T[1, 0]), T[0, 5], zero(T)),
                         (NormL1Nonneg(one(T), T[1, 0]), T[0, -5], T(Inf)),
                         (ElasticNet(one(T), zero(T), T[1, 0]), T[0, 5], zero(T)),
                         (ElasticNet(one(T), one(T), T[1, 0]), T[0, 5], T(Inf)),
                         (SparseGroupNormL2(one(T), one(T), [1:2], [one(T)]), T[0, 0, 9], zero(T)),
                         (GroupNormL2(one(T), [1:2, 3:4], T[1, 0]), T[0, 0, 5, 6], zero(T)),
                         (GroupNormLinf(one(T), [1:2, 3:4], T[1, 0]), T[0, 0, 5, 6], zero(T)),
                         (GroupNormL2(one(T), [1:2, 3:4], T[1, 0]), T[0, 1, 5, 6], T(Inf)))
      @test same(Scaled(f, T(Inf))(x), want)
    end
  end
end

@testset "Spectral and SingularValue of an indicator under an infinite factor agree with value_and_prox!" begin
  for T in (Float32, Float64), W in (Spectral, SingularValue),
      f in (IndNonnegative(), IndBallL2(T(5)), IndBox(T(-5), T(5))),
      X in (zeros(T, 2, 2), T[1 0; 0 1], T[1 0; 0 0], T[2 0; 0 -1])
    s = W(Scaled(f, T(Inf)))
    v, Y = vp(s, X, one(T))
    @test !isnan(v)
    @test same(s(Y), v)                                       # O2
  end
end

@testset "Scaled with an infinite factor over an indicator: scalar and array steps agree" begin
  allowscalar(false)
  for T in (Float32, Float64)
    x = T[3, 0.5, -7]
    # the indicators that take array steps (a coupling set such as IndBallL2 has no array-step method)
    for f in (IndNonnegative(), IndBox(T(-1), T(1)), Translate(IndBox(T(-1), T(1)), T[0.5, 0, 0]))
      s = Scaled(f, T(Inf))
      ys = fill(T(-7), 3)
      vs = value_and_prox!(ys, s, x, one(T))
      for γ in (T[1, 1, 1], T[2, 0.5, 1], fill(T(Inf), 3))
        y = fill(T(-7), 3)
        v = value_and_prox!(y, s, x, γ)
        @test y == ys                                         # the projection, whatever the step
        @test same(v, zero(T))
        @test same(v, vs)
        @test same(s(y), v)                                   # O2
        z = fill(T(-7), 3)
        @test prox!(z, s, x, γ) === z
        @test z == ys
      end
    end
  end
end

@testset "conjugates of balls and spectral functions with an infinite level" begin
  allowscalar(false)
  for T in (Float32, Float64)
    for f in (Conjugate(IndBallL2(T(Inf))), Conjugate(IndBallL1(T(Inf))))
      @test same(f(zeros(T, 2)), zero(T))
      @test same(f(T[1, 0]), T(Inf))
    end
    @test same(Conjugate(IndBallL1(T(Inf)))(JLArray(zeros(T, 2))), zero(T))
    for F in (NormL2, NormLinf)
      @test same(SingularValue(F(T(Inf)))(zeros(T, 2, 2)), zero(T))
      @test same(SingularValue(F(T(Inf)))(T[1 0; 0 1]), T(Inf))
      @test same(Spectral(F(T(Inf)))(zeros(T, 2, 2)), zero(T))
    end
    @test same(SeparableSum(NormL2(T(Inf)), NormLinf(T(Inf)))((zeros(T, 2), zeros(T, 2))), zero(T))
  end
end

####################################
#  Controls: unchanged behaviour
####################################

@testset "controls: a finite level never takes an infinite-level branch (R3)" begin
  allowscalar(false)
  λ = 2.0^128                                     # finite in Float64, not in Float32
  x = Float32[2.0f0^-100, 0]
  for f in (NormL2(λ), NormLinf(λ), Scaled(NormL1(1.0f0), λ))
    @test same(f(x), 2.0f0^28)
    @test same(f(JLArray(x)), 2.0f0^28)
  end
  @test same(NuclearNorm(λ)(Float32[2.0f0^-100 0; 0 0]), 2.0f0^28)
  # the new branch of the ℓ∞ proximal operator: the finite level keeps its own path
  for d in (identity, JLArray)
    v, y = vp(NormLinf(λ), d(Float32[1, NaN]), 1.0f0)
    @test same(v, NaN32)
    @test isequal(Array(y), Float32[-Inf, NaN])
    v, y = vp(NormLinf(λ), d(Float32[1, 2]), 1.0f0)
    @test same(v, 0.0f0)
    @test Array(y) == zeros(Float32, 2)
  end
  # a chain whose product overflows is not an infinite factor
  c = Scaled(Scaled(NormL2(1.0), 2.0^600), 2.0^600)
  @test same(c(zeros(2)), 0.0)
  @test same(c([1.0, 0]), Inf)
  v, y = vp(c, [3.0, 4.0], 2.0^-1300)
  @test same(v, Inf)
  @test y == [3.0, 4.0]
  # a zero level keeps 0⋅Inf at an infinite entry
  @test same(NormLinf(0.0)([Inf, 1.0]), NaN)
  @test same(NormLinf(0.0)(JLArray([Inf, 1.0])), NaN)
end

@testset "controls: finite levels, exact values" begin
  for T in (Float32, Float64)
    x = T[2, -2, 0, 2, 2]                                       # a norm that every algorithm forms exactly
    @test same(NormL2(T(0.5))(x), T(2))
    v, y = vp(NormL2(T(0.5)), x, T(4))                        # γλ = 2 halves x
    @test same(v, one(T))
    @test y == x ./ 2
    x = T[3, -1, 0.5, 2]
    @test same(NormLinf(T(0.5))(x), T(1.5))
    v, y = vp(NormLinf(T(0.5)), x, T(2))                      # clipped at 2
    @test same(v, one(T))
    @test y == T[2, -1, 0.5, 2]
    s = Scaled(NormL1(T(0.5)), T(0.5))
    @test same(s(x), T(1.625))
    v, y = vp(s, x, T(4))
    @test same(v, T(0.75))
    @test y == T[2, 0, 0, 1]
    v, y = vp(s, x, T[4, 4, 2, 8])
    @test same(v, T(0.5))
    @test y == T[2, 0, 0, 0]
  end
end

@testset "controls: unchanged results with an infinite level" begin
  allowscalar(false)
  for T in (Float32, Float64)
    # NormL2 and NormLinf at finite x and positive steps
    for F in (NormL2, NormLinf), x in (T[3, 4], zeros(T, 2), T[1, Inf]), γ in (one(T), T(Inf))
      v, y = vp(F(T(Inf)), x, γ)
      @test same(v, zero(T))
      @test y == zeros(T, 2)
    end
    for γ in (one(T), T(Inf))
      v, y = vp(NormL2(T(Inf)), T[1, NaN], γ)
      @test same(v, zero(T))
      @test y == zeros(T, 2)
    end
    # NormLinf at steps that are not positive: as before (step validation is not part of this)
    for d in (identity, JLArray)
      for (x, γ, want, wanty) in ((T[3, 4], -one(T), T(Inf), T[3, 4]),
                                  (T[3, 4], T(NaN), T(NaN), T[NaN, NaN]),
                                  (T[3, 4], zero(T), T(NaN), T[NaN, NaN]),
                                  (T[1, NaN], zero(T), T(NaN), T[NaN, NaN]))
        v, y = vp(NormLinf(T(Inf)), d(x), γ)
        @test same(v, want)
        @test isequal(Array(y), wanty)
      end
    end
    for (x, γ, want, wanty) in ((T[1, NaN], -one(T), T(Inf), T[1, NaN]), (T[1, NaN], T(NaN), T(NaN), T[NaN, NaN]))
      v, y = vp(NormLinf(T(Inf)), x, γ)
      @test same(v, want)
      @test isequal(y, wanty)
    end
    # NuclearNorm at a non-finite input keeps its documented rule: NaN at a NaN entry (E2, PR #384),
    # Inf at an infinite one; an infinite level is never constant
    for X in (T[1 NaN; 0 0], T[0 Inf; 0 0])
      want = any(isnan, X) ? T(NaN) : T(Inf)
      @test same(NuclearNorm(T(Inf))(X), want)
      v, Y = vp(NuclearNorm(T(Inf)), X, one(T))
      @test same(v, want)
      @test all(isnan, Y)
    end
    # the rank indicator shares the overflowing route of the nuclear norm
    M = floatmax(T) * T(0.75)
    v, Y = vp(IndBallRank(1), T[M 0 0; 0 M 0; 0 0 M / 2], one(T))
    @test same(v, zero(T))
    @test count(!iszero, Y) == 1
    # a nonzero value under an infinite factor, and the gradient there
    s = Scaled(SqrNormL2(one(T)), T(Inf))
    g = fill(T(-7), 2)
    @test same(value_and_gradient!(g, s, T[1, 0]), T(Inf))
    @test isequal(g, T[Inf, NaN])
    @test same(Scaled(NormL2(one(T)), T(Inf))(T[NaN, 0]), T(NaN))
  end
end

@testset "controls: a zero level or weight frees an entry, as before" begin
  # an entry the function does not read (a zero level, or a zero weight under a zero level) may hold
  # anything; these values are 0 before and after the infinite-level change
  for T in (Float32, Float64), v in (T(NaN), T(Inf), T(-Inf))
    for f in (NormL1(zero(T)), NormL1(zero(T), T[1, Inf]), NormL1(zero(T), T[0, Inf]), NormL2(zero(T)),
              GroupNormLinf(zero(T), [1:2], T[1]))
      @test same(Scaled(f, T(Inf))(T[v, 0]), zero(T))
    end
    for f in (GroupNormL2(zero(T), [1:2], T[1]), GroupNormL2(zero(T), [1:2, 3:3], T[1, Inf]),
              SparseGroupNormL2(zero(T), zero(T), [1:2], T[1]))
      @test same(Scaled(f, T(Inf))(T[v, 0, 0]), zero(T))
    end
    isnan(v) && continue
    for f in (ElasticNet(zero(T), zero(T)), ElasticNet(zero(T), zero(T), T[1, Inf]))
      @test same(Scaled(f, T(Inf))(T[v, 0]), zero(T))
    end
  end
  for T in (Float32, Float64), f in (NormL1Nonneg(zero(T)), NormL1Nonneg(zero(T), T[1, Inf]))
    @test same(Scaled(f, T(Inf))(T[Inf, 0]), zero(T))
  end
end

@testset "controls: γ = 0 and NaN steps of an infinite level are O2 only" begin
  for T in (Float32, Float64)
    for (f, x) in ((NormLinf(T(Inf)), T[3, 4]), (NuclearNorm(T(Inf)), T[3 0; 0 4]),
                   (NuclearNorm(T(Inf); partial = true), T[3 0; 0 4]),
                   (Scaled(NormL2(one(T)), T(Inf)), T[3, 4]), (Scaled(NormL1(one(T)), T(Inf)), T[3, 4]))
      for γ in (zero(T), T(NaN))
        f isa NuclearNorm && isnan(γ) && continue
        v, y = vp(f, x, γ)
        @test same(v, f(y))
      end
    end
    v, y = vp(Scaled(NormL1(one(T)), T(Inf)), T[3, 4], T[0, 0])
    @test same(v, Scaled(NormL1(one(T)), T(Inf))(y))
  end
end

end
