module BlockChecksStepsTest

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase
const proximal = (prox!, value_and_prox!)
const I2 = [1.0 0; 0 1.0]

function thrown(op, y, f, x, γ)
  @nospecialize
  try
    op(y, f, x, γ)
  catch e
    return e
  end
  nothing
end
# exact = false compares only the exception types: a preflight that mirrors a kernel's
# rejection may word it differently from the kernel
same_exception(e, ref; exact = true) =
  typeof(e) === typeof(ref) && (!exact || sprint(showerror, e) == sprint(showerror, ref))

function unchanged(op, h, x, γ, ref; output = x, exact = true)
  @nospecialize
  for alias in (false, true)
    alias && typeof(output) !== typeof(x) && continue
    z = deepcopy(x)
    y = alias ? z : deepcopy(output)
    before, input = deepcopy(y), deepcopy(z)
    e = thrown(op, y, deepcopy(h), z, γ)
    @test same_exception(e, ref; exact)
    @test isequal(y, before)
    @test isequal(z, input)
  end
end

function block_failure(f, x, γ; output = x, exact = true)
  @nospecialize
  for op in proximal
    ref = thrown(op, deepcopy(output), deepcopy(f), deepcopy(x), γ)
    @test ref !== nothing
    ref === nothing && continue
    point, dest = ([3.0], x), ([7.0], output)
    s = SeparableSum(SqrNormL2(2.0), f)
    unchanged(op, s, point, γ, ref; output = dest, exact)
    unchanged(op, Scaled(s, 2.0), point, γ / 2, ref; output = dest, exact)
    unchanged(op, SeparableSum(SqrNormL2(2.0), SeparableSum(f)),
              ([3.0], (x,)), γ, ref; output = ([7.0], (output,)), exact)
    unchanged(op, SeparableSum(SqrNormL2(2.0), Scaled(SeparableSum(f), 2.0)),
              ([3.0], (x,)), γ / 2, ref; output = ([7.0], (output,)), exact)
    x isa Vector || continue
    refslice = thrown(value_and_prox!, deepcopy(output), deepcopy(f), deepcopy(x), γ)
    refslice === nothing || unchanged(op, SliceSum(SqrNormL2(2.0), f), hcat(x, x), γ,
                                     refslice; output = hcat(output, output), exact)
    fibers = reshape(repeat(x, 4), 2, 2, length(x))
    out = reshape(repeat(output, 4), 2, 2, length(output))
    unchanged(op, FiberSum(f; diagonal = SqrNormL2(2.0)), fibers, γ, ref; output = out, exact)
    unchanged(op, FiberSum(SqrNormL2(2.0); diagonal = f, symmetric = true), fibers, γ, ref; output = out, exact)
  end
end

# Compare successful calls and failures to the standalone oracle, including its writes.
function same_outcome(f, x, γ; output = x, exact = true)
  @nospecialize
  for op in proximal, alias in (false, true)
    alias && typeof(output) !== typeof(x) && continue
    input = deepcopy(x)
    expected = alias ? input : deepcopy(output)
    ref = thrown(op, expected, deepcopy(f), input, γ)
    point = ([3.0], deepcopy(x))
    dest = alias ? point : ([7.0], deepcopy(output))
    e = thrown(op, dest, SeparableSum(SqrNormL2(2.0), deepcopy(f)), point, γ)
    if ref === nothing
      @test e === nothing
      @test isequal(dest[2], expected)
    else
      @test same_exception(e, ref; exact)
    end
  end
end

@testset "Step failures before any block is written" begin
  envelope = MoreauEnvelope(NormL1(1.0), 1.0)
  image = LinearImage(NormL1(1.0), I2, 1.0)
  for (f, γ) in ((envelope, -2.0), (SumLargest(1), -1.0), (image, -1.0))
    block_failure(f, [2.0, 3.0], γ)
    for h in (Scaled(f, 2.0), Translate(f, [0.5, 0.25]), Tilt(f, [0.5, 0.25]),
              Regularize(f, 0.25), Regularize(f, 0.25, [0.5, 0.25]),
              Precompose(f, 2.0), Precompose(f, I2, 1.0),
              PointwiseMinimum(SqrNormL2(), f), Conjugate(Conjugate(f)))
      block_failure(h, [2.0, 3.0], γ)
    end
  end
  # The outer envelope forwards its combined step, which is negative here.
  block_failure(MoreauEnvelope(SumLargest(1), 1.0), [2.0, 3.0], -0.5 - 1.0)
  block_failure(MoreauEnvelope(envelope, 1.0), [2.0, 3.0], Inf)
  for T in (Float32, Float64)
    block_failure(envelope, T[2, 3], -2.0; output = zeros(T === Float32 ? Float64 : Float32, 2))
  end
end

@testset "Leaves and wrappers that had no step check" begin
  v = [0.3, -0.2, 0.5]
  S = [2.0 0.5 0.0; 0.5 1.5 0.2; 0.0 0.2 1.0]
  wrappers(f) = (Scaled(f, 2.0), Translate(f, [0.1, 0.2, 0.3]), Tilt(f, [0.1, 0.2, 0.3]),
                 Regularize(f, 0.25), Precompose(f, 2.0), PointwiseMinimum(SqrNormL2(), f),
                 Conjugate(Conjugate(f)))
  # A negative step is a DomainError at the first entry of these proximal points.
  for f in (NegLog(1.0), NegEntropy(2.0), KLDivergence([0.2, 0.3, 0.5]))
    for T in (Float32, Float64), γ in (-1.0, -1e-3, -Inf)
      block_failure(f, T.(v), γ; exact = false)
      block_failure(f, T.(v), γ; exact = false, output = zeros(T === Float32 ? Float64 : Float32, 3))
    end
    for h in wrappers(f)
      h isa Translate && f isa KLDivergence && continue
      block_failure(h, v, -1.0; exact = false)
    end
    # The other steps are the leaf's: the block agrees with the standalone oracle.
    for γ in (NaN, 0.0, 0.5, Inf)
      same_outcome(f, v, γ)
    end
    # An empty input has no entry to fail at.
    f isa KLDivergence || same_outcome(f, Float64[], -1.0)
    # The kernel converts the step to the type of the input: -1e-50 is -0.0 in Float32.
    for T in (Float32, Float64), γ in (-1e-50, -1e-320)
      same_outcome(f, T.(v), γ; exact = false)
    end
  end
  # LogDetLoss: the step precedes the input, so that an invalid step is not hidden by a
  # nonfinite entry, and a valid step with a nonfinite entry still gives NaN.
  for T in (Float32, Float64)
    f = LogDetLoss(Matrix{T}(I, 3, 3))
    for γ in (-1.0, -1e-3, -Inf, NaN)
      block_failure(f, T.(S), γ)
      block_failure(f, T.(S), γ; output = zeros(T === Float32 ? Float64 : Float32, 3, 3))
      @test_throws DomainError prox!(zeros(T, 3, 3), f, fill(T(NaN), 3, 3), γ)
    end
    for h in wrappers(f)
      h isa Union{Translate, Tilt, Regularize} && continue
      block_failure(h, T.(S), -1.0; exact = false)
    end
    for γ in (0.0, 0.5, Inf), x in (T.(S), fill(T(NaN), 3, 3), fill(T(Inf), 3, 3))
      same_outcome(f, x, γ)
    end
  end
  # An empty input has no eigenvalue to fail at, so a NaN step is accepted there.
  for T in (Float32, Float64)
    same_outcome(LogDetLoss(zeros(T, 0, 0)), zeros(T, 0, 0), NaN)
    same_outcome(SingularValue(NegLog(1.0)), T[2 0.5; 0.5 1], -1e-50)
  end
  # The spectral wrappers forward the step to the function of the spectrum.
  for T in (Float32, Float64), (f, γs) in ((Spectral(IndBallL1(1.0)), (-1.0, NaN, -Inf)),
                                           (SingularValue(IndBallL1(1.0)), (-1.0, NaN, -Inf)),
                                           (SingularValue(NegLog(1.0)), (-1.0, -Inf)),
                                           (Spectral(SumLargest(1)), (-1.0, NaN, 0.0, Inf)))
    for γ in γs
      block_failure(f, T.(S), γ; exact = !(f isa SingularValue{<:NegLog}))
      @test_throws Union{ArgumentError, DomainError} prox!(zeros(T, 3, 3), f, fill(T(NaN), 3, 3), γ)
    end
    f isa Spectral{<:IndBallL1} || continue
    block_failure(SingularValue(IndBallL1(1.0)), T.(S[:, 1:2]), -1.0; output = zeros(T, 3, 2))
    block_failure(Translate(f, T.(S)), T.(S), -1.0)
    block_failure(Conjugate(f), T.(S), -1.0)
    for γ in (0.0, 0.5, Inf), x in (T.(S), fill(T(NaN), 3, 3))
      same_outcome(f, x, γ)
      same_outcome(SingularValue(IndBallL1(1.0)), x, γ)
    end
    same_outcome(Spectral(IndBox(0.0, 1.0)), T.(S), NaN)
    # inputs near the overflow threshold take the rescaled path and the same step check
    same_outcome(f, T(floatmax(T) / 4) .* T.(S), 0.5)
    block_failure(f, T(floatmax(T) / 4) .* T.(S), -1.0)
  end
  # The set of the Moreau point of a conjugate ball has radius γ r.
  for T in (Float32, Float64), γ in (-1.0, -1e-3, -Inf, NaN)
    f = Conjugate(IndBallL2(1.0))
    block_failure(f, T.(v), γ)
    block_failure(f, T.(v), γ; output = zeros(T === Float32 ? Float64 : Float32, 3))
    for h in (Conjugate(Translate(IndBallL2(1.0), T.(v))), Conjugate(Scaled(IndBallL2(1.0), 2.0)))
      block_failure(h, T.(v), γ; exact = false)
    end
    for h in wrappers(f)
      # the regularized step γ / (1 + γρ) of γ = -Inf is NaN, which a ball radius rejects
      # only through the conjugate, not through this wrapper's own step
      h isa Regularize && γ == -Inf && continue
      block_failure(h, T.(v), γ; exact = false)
    end
    # a conjugate of a block sum accepts the step, as its standalone oracle does
    same_outcome(Conjugate(SeparableSum(IndBallL2(1.0))), (T.(v),), γ)
  end
  for γ in (0.0, 0.5, Inf)
    same_outcome(Conjugate(IndBallL2(1.0)), v, γ)
  end
  # a radius of zero stays zero at every step, so the check must not reject any of them
  for γ in (-1.0, -Inf, NaN)
    same_outcome(Conjugate(IndBallL2(0.0)), v, γ)
  end
  # PointwiseMinimum: sqrt(-Inf) in the scores of non-indicator candidates.
  for T in (Float32, Float64), g in (PointwiseMinimum(SqrNormL2(), NormL1(1.0)),
                                     PointwiseMinimum(NormL1(1.0), SumLargest(1)),
                                     PointwiseMinimum(SqrNormL2(), IndBox(-1.0, 1.0)))
    block_failure(g, T.(v), -Inf; exact = !(g.fs[2] isa SumLargest))
    # a candidate whose proximal point at -Inf is NaN used to return NaN
    block_failure(PointwiseMinimum(NormL1(1.0)), T.(v), -Inf)
    block_failure(g, T.(v), -Inf; exact = !(g.fs[2] isa SumLargest), output = zeros(T === Float32 ? Float64 : Float32, 3))
    # Regularize turns the step -Inf into NaN, which no candidate here rejects
    for h in (Scaled(g, 2.0), Translate(g, T.(v)), Tilt(g, T.(v)), Precompose(g, 2.0))
      block_failure(h, T.(v), -Inf; exact = false)
    end
    @test_throws DomainError prox!(zeros(T, 3), g, fill(T(NaN), 3), -Inf)
    # finite negative steps, NaN and +Inf are the candidates' (here: NormL1 and SqrNormL2)
    g.fs[2] isa SumLargest || for γ in (-1.0, NaN, 0.0, Inf)
      same_outcome(g, T.(v), γ)
    end
  end
  # indicators have no energy, and an infinite step is the projection
  same_outcome(PointwiseMinimum(IndBox(-1.0, 1.0), IndBallL2(1.0)), v, Inf)
  same_outcome(PointwiseMinimum(IndBox(-1.0, 1.0), IndNonnegative()), v, -Inf)
end

# A real spectrum is a plain vector, which a conjugate of a norm projects directly. Before the
# check of a spectral wrapper saw arrays of the kernel's types, it saw a size-only placeholder,
# which sent the conjugate of NormLinf to the Moreau branch and allocated; the check now forwards
# the storage of the spectrum (`_projection_host`) and takes the kernel's branch. The standalone
# oracle and a later block must agree: every call the standalone oracle accepts is accepted in a
# block with the same result, and every rejection is the same exception.
@testset "Spectral wrappers of conjugates agree with their blocks" begin
  S = [2.0 0.5 0.0; 0.5 1.5 0.2; 0.0 0.2 1.0]
  W = [2.0 0.5 0.1; 0.3 1.5 0.2]
  nonfinite = copy(S); nonfinite[1, 2] = NaN
  groups = [1:2, 3:3]
  inner = (NormLinf(1.0), NormLinf(0.0), GroupNormLinf(1.0, groups), Scaled(NormLinf(1.0), 2.0),
           NormL1(1.0), NormL2(1.0), GroupNormL2(1.0, groups), IndNonnegative(),
           IndBallL2(1.0), IndBallL1(1.0), IndSimplex(1.0), SumLargest(2), SqrNormL2(1.0))
  steps = (-Inf, -1.0, -1e-3, -0.0, 0.0, 0.5, 1.0, 1e300, Inf, NaN)
  # Spectral rejects functions that depend on the order of the entries; SingularValue accepts them
  wrap(W, g) = try W(Conjugate(deepcopy(g))) catch e; e isa ArgumentError ? nothing : rethrow() end
  wrapped = [(g, wrap(Spectral, g), wrap(SingularValue, g)) for g in inner]
  # every wrapper is constructed, but the two that Spectral refuses
  for (g, s, v) in wrapped
    @test v isa SingularValue
    @test (s === nothing) == (g isa Union{GroupNormLinf, GroupNormL2})
  end
  @test count(!isnothing, getindex.(wrapped, 2)) == length(inner) - 2
  @test wrap(Spectral, NormLinf(1.0)) isa Spectral{<:Conjugate{<:NormLinf}}
  @test wrap(SingularValue, NormLinf(1.0)) isa SingularValue{<:Conjugate{<:NormLinf}}
  @test wrap(SingularValue, GroupNormLinf(1.0, groups)) isa SingularValue{<:Conjugate{<:GroupNormLinf}}
  for T in (Float32, Float64), (g, s, v) in wrapped, γ in steps
    for f in (s, v)
      f === nothing && continue
      same_outcome(f, T.(S), γ)
      same_outcome(f, T.(nonfinite), γ)
      f isa SingularValue && same_outcome(f, T.(W), γ)
    end
  end
  # a rejection that the placeholder can foresee is found before the first block is written
  for T in (Float32, Float64), γ in (-1.0, NaN)
    block_failure(Spectral(Conjugate(IndBallL2(1.0))), T.(S), γ)
    block_failure(SingularValue(Conjugate(IndBallL2(1.0))), T.(W), γ; output = zeros(T, 2, 3))
  end
end

@testset "Edge cases of the checks added for leaves that had none" begin
  v = [0.3, -0.2, 0.5]
  # The shape is checked before the precision: a nonsquare BigFloat matrix is a shape error, a
  # square one a precision error, whichever the oracle
  for op in proximal, f in (Spectral(NormL1(1.0)), Spectral(Conjugate(NormLinf(1.0))))
    @test_throws DimensionMismatch op(zeros(BigFloat, 2, 3), f, zeros(BigFloat, 2, 3), 1.0)
    @test_throws DimensionMismatch op(zeros(BigFloat, 2, 3), f, zeros(BigFloat, 2, 3), -1.0)
    @test_throws ArgumentError op(zeros(BigFloat, 3, 3), f, zeros(BigFloat, 3, 3), 1.0)
  end
  S = [2.0 0.5 0.0; 0.5 1.5 0.2; 0.0 0.2 1.0]
  # LogDetLoss: the shape, then the step, then the input
  f = LogDetLoss(Matrix(1.0I, 3, 3))
  for op in proximal
    @test_throws DimensionMismatch op(zeros(2, 2), f, zeros(2, 2), -1.0)
    @test_throws DomainError op(zeros(3, 3), f, fill(NaN, 3, 3), -1.0)
    @test_throws DomainError op(zeros(3, 3), f, fill(Inf, 3, 3), NaN)
    @test_throws DomainError op(zeros(3, 3), f, S, NaN)
    @test_throws DomainError op(zeros(3, 3), f, S, -Inf)
    @test_throws DimensionMismatch op(zeros(3, 3), f, zeros(3, 2), NaN)
    # a valid step with a nonfinite input still gives NaN, as before
    y = zeros(3, 3)
    op(y, f, fill(NaN, 3, 3), 1.0)
    @test all(isnan, y)
    # an empty input accepts a NaN step and rejects a negative one, as before
    @test op(zeros(0, 0), LogDetLoss(zeros(0, 0)), zeros(0, 0), NaN) isa Union{Real, AbstractMatrix}
    @test_throws DomainError op(zeros(0, 0), LogDetLoss(zeros(0, 0)), zeros(0, 0), -1.0)
  end
  # a negative step that underflows to -0.0 in Float32 is accepted, as by the kernel
  same_outcome(LogDetLoss(Matrix{Float32}(I, 3, 3)), Float32.(S), -1e-50)
  # Conjugate(IndBallL2): radius zero, and the effective step through wrappers
  for h in (Conjugate(IndBallL2(0.0)), Scaled(Conjugate(IndBallL2(0.0)), 2.0),
            Precompose(Conjugate(IndBallL2(0.0)), 2.0), Tilt(Conjugate(IndBallL2(0.0)), v),
            Translate(Conjugate(IndBallL2(0.0)), v)), γ in (-1.0, -Inf, NaN, 0.0, Inf)
    same_outcome(h, v, γ)
  end
  for γ in (-1.0, NaN)
    block_failure(Scaled(Conjugate(IndBallL2(1.0)), 2.0), v, γ)
    block_failure(Precompose(Conjugate(IndBallL2(1.0)), 0.5), v, γ)
    block_failure(Conjugate(Scaled(IndBallL2(1.0), 3.0)), v, γ; exact = false)
  end
  # PointwiseMinimum at -Inf: an indicator is the projection, a candidate rejects its own steps,
  # and any non-indicator is rejected before a candidate runs
  same_outcome(PointwiseMinimum(IndBox(-1.0, 1.0), IndNonnegative()), v, -Inf)
  block_failure(PointwiseMinimum(IndBallL2(1.0), IndNonnegative()), v, -Inf)
  block_failure(PointwiseMinimum(IndBox(-1.0, 1.0), NormL1(1.0)), v, -Inf)
  block_failure(PointwiseMinimum(NormL1(1.0), IndBox(-1.0, 1.0)), v, -Inf)
  for γ in (0.0, 1.0, Inf, NaN)
    same_outcome(PointwiseMinimum(IndBox(-1.0, 1.0), NormL1(1.0)), v, γ)
  end
end

@testset "Conjugate branches" begin
  # A conjugate indicator projects at step one, including the scale Inf.
  block_failure(Conjugate(Scaled(SumLargest(1), Inf)), [2.0, 3.0], 1.0)
  # MoreauEnvelope has no closed-form conjugate value: the value oracle uses
  # the inverse step, while prox! follows the Moreau path.
  block_failure(Conjugate(MoreauEnvelope(NormL1(), 1.0)), [2.0, 3.0], -0.5)
  block_failure(Conjugate(LinearImage(NormL1(), I2, 1.0)), [2.0, 3.0], -1.0)
  block_failure(Conjugate(SeparableSum(MoreauEnvelope(NormL1(), 1.0))), ([2.0, 3.0],), -0.5)
  for f in (Conjugate(Translate(Scaled(SumLargest(1), Inf), [0.0, 0.0])),
            Conjugate(Scaled(Translate(SumLargest(1), [0.0, 0.0]), Inf)))
    block_failure(f, [2.0, 3.0], 1.0)
  end
  for f in (Conjugate(IndBox(-1.0, 1.0)), Conjugate(IndBallL2()),
            Conjugate(Translate(IndBox(-1.0, 1.0), [0.5, 0.25])),
            Conjugate(Scaled(IndBox(-1.0, 1.0), 2.0)), Conjugate(SumLargest(1)))
    same_outcome(f, [2.0, 3.0], 1.0)
  end
end

# A conjugate quadratic at a step where x/γ overflows but the point is representable
@testset "Conjugate(SqrNormL2) in a block at an extreme step" begin
  setprecision(BigFloat, 400) do
    ref(λ, x, γ) = big(λ) * big(x) / (big(λ) + big(γ))
    near(::Type{T}, got, r) where {T} = abs(big(got) - r) <= 2 * (abs(T(r)) < floatmin(T) ? big(nextfloat(zero(T))) : big(eps(T(r))))
    for (T, x2, γ, λ1) in ((Float64, 1e300, 1e-10, 2.0), (Float32, 3f38, 0.1f0, 2f0))
      c = Conjugate(SqrNormL2(T(1)))
      s = SeparableSum(SqrNormL2(λ1), c, NormL1(T(0.25)))
      x = (T[3], T[x2], T[2, -0.1])
      r1 = (big(T(3))) / (1 + big(γ) * big(λ1))
      r2 = ref(T(1), T(x2), γ)
      r3 = (big(T(2)) - big(γ) / 4, -(big(T(0.1)) - big(γ) / 4))
      for op in proximal, alias in (false, true)
        input = deepcopy(x)
        y = alias ? input : (T[7], T[7], T[7, 7])
        v = op(y, s, input, γ)
        @test near(T, y[1][1], r1) && near(T, y[2][1], r2) && near(T, y[3][1], r3[1]) && near(T, y[3][2], r3[2])
        alias || @test isequal(input, x)
        op === value_and_prox! && @test v isa T
      end
      # the standalone conjugate agrees
      @test near(T, prox(c, T[x2], γ)[1], r2)
      @test prox_entry(c, T(x2), 1, γ) == prox(c, T[x2], γ)[1]
      # the value of the point is not representable: Inf, as for the standalone call
      if T === Float64
        y = (T[7], T[7], T[7, 7])
        @test value_and_prox!(y, s, x, γ) == Inf
        @test value_and_prox!(T[7], c, T[x2], γ) == Inf
      end
    end
  end
end

@testset "Conjugate(SqrNormL2) in a block at a step that is not finite and positive" begin
  # The conjugate raises its error before it writes: the standalone call leaves its output as it was and
  # the block call raises the same error, with the block of the conjugate and the input of that block
  # unchanged. A block before it is written, as it was before the closed form.
  s = SeparableSum(SqrNormL2(2.0), Conjugate(SqrNormL2()))
  for γ in (0.0, NaN, Inf, -1.0), op in proximal, alias in (false, true)
    alone = [7.0]
    ref = thrown(op, alone, Conjugate(SqrNormL2()), [1e300], γ)
    @test ref isa ArgumentError
    @test alone == [7.0]
    x = ([3.0], [1e300])
    input = deepcopy(x)
    y = alias ? input : ([7.0], [7.0])
    e = thrown(op, y, s, input, γ)
    @test same_exception(e, ref)
    @test isequal(y[2], alias ? x[2] : [7.0])
    @test isequal(input[2], x[2])
  end
end

@testset "Exact empty and bounded box-linear failures" begin
  for set in (IndHyperplane(1.0, 1.0), IndHalfspace(1.0, -1.0))
    block_failure(Constrained(NormL1(), set), Float64[], 1.0)
  end
  for set in (IndHyperplane(1.0, 5.0), IndHyperplane(-1.0, -5.0),
              IndHalfspace(1.0, -1.0), IndHyperslab(1.0, 5.0, 6.0),
              IndHyperslab(1.0, -3.0, -1.0))
    for T in (Float32, Float64)
      f = Constrained(IndBox(0.0, 1.0), set)
      block_failure(f, T[0.25, 0.5], 1.0)
      same_outcome(f, T[0.25, 0.5], 1.0; output = zeros(T === Float32 ? Float64 : Float32, 2))
    end
  end
  for warm in (false, true)
    for (box, set, good, bad) in (
        (IndBox(0.0, 1.0), IndHyperplane(1.0, 2.0), [2.0, 2.0, 2.0], [0.25]),
        (IndBox(1.0, 2.0), IndHalfspace(1.0, 3.0), [3.0, 3.0], ones(4)),
        (IndBox(1.0, 2.0), IndHyperslab(1.0, 1.0, 1.5), [3.0], ones(3)))
      f = Constrained(box, set)
      warm && prox!(similar(good), f, good, 1.0)
      block_failure(f, bad, 1.0)
    end
  end
end

@testset "Checks are never stricter than their oracles" begin
  empty = Constrained(IndBox(-1e100, 1e100), IndHyperplane(1.0, 1.0))
  for op in proximal
    ref = thrown(op, Float64[], empty, Float64[], [1.0])
    @test ref isa DimensionMismatch
    @test PB._checkstep(op, empty, Float64[], Float64[], [1.0]) === true
    @test same_exception(ref, DimensionMismatch("a parameter array has length 1 and the argument has length 0"))
    scalar_ref = thrown(op, Float64[], empty, Float64[], 1.0)
    checked = try
      PB._checkstep(op, empty, Float64[], PB._StepInput(Float64[]), 1.0)
    catch e
      e
    end
    @test same_exception(checked, scalar_ref)
  end
  for (Tx, Ty) in ((ComplexF64, ComplexF64), (Float64, ComplexF64), (ComplexF64, Float64))
    f = SumLargest(1)
    x, y = Tx[2, 3], zeros(Ty, 2)
    for op in (prox!, value_and_prox!)
      @test_throws ArgumentError("SumLargest requires real entries") op(y, f, x, -1.0)
      @test iszero(y)
    end
  end
  for (Tx, Ty) in ((Float32, Float64), (Float64, Float32)), alias in (false, true)
    e = MoreauEnvelope(NormL1(), 1.0)
    x, y = Tx[2, 3], zeros(Ty, 2)
    same_outcome(e, x, -1.0 + eps(Ty); output = alias ? x : y)
    same_outcome(e, x, -1.0 + eps(Float64); output = alias ? x : y)
    same_outcome(Precompose(e, 2.0), x, (-1.0 + eps(Tx)) / 4; output = y)
    same_outcome(SumLargest(1), x, 1e-100; output = y)
  end
  for set in (IndHyperplane(1.0, 1.0), IndHalfspace(1.0, 1.0), IndHyperslab(1.0, 0.5, 1.5))
    for warm in (false, true)
      f = Constrained(IndBox(0.0, 1.0), set)
      warm && prox!(zeros(2), f, [2.0, -1.0], 1.0)
      same_outcome(f, [0.25, 0.5], 1.0)
      # Same search history, but a dimension whose attainable range misses the set.
      same_outcome(f, [0.25], 1.0)
    end
  end
  for T in (Float32, Float64)
    for b in (prevfloat(T(2)), T(2), nextfloat(T(2)))
      same_outcome(Constrained(IndBox(T(0), T(1)), IndHyperplane(T(1), b)), T[0, 0], T(1))
    end
  end
  same_outcome(Constrained(NormL1(), IndHyperplane(1.0, 1e100)), Float64[], 1.0; output = Float32[])
  overflow = Constrained(IndBox(-1e308, 1.0), IndHyperplane(1.0, 5.0))
  same_outcome(overflow, [-1e308, -1e308], 1.0)
  for v in (NaN, Inf, -Inf), set in (IndHyperplane(1.0, 5.0), IndHalfspace(1.0, -1.0))
    same_outcome(Constrained(IndBox(0.0, 1.0), set), [v, 0.0], 1.0)
  end
  for set in (IndHyperplane(1.0, 0.0), IndHalfspace(1.0, 1.0))
    same_outcome(Constrained(NormL1(), set), Float64[], 1.0)
  end
  # A transformed input must never be inferred from a stale output buffer.
  for wrap in (f -> Translate(f, [Inf, 0.0]), f -> Tilt(f, [Inf, 0.0]),
               f -> Regularize(f, 1.0, [Inf, 0.0]), f -> Precompose(f, 2.0))
    same_outcome(wrap(Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 5.0))), [0.0, 0.0], 1.0)
  end
end

_allocation_check(op::O, y, f, x, γ) where {O} = PB._checkstep(op, f, y, x, γ)
_allocation_prox!(y, f, x, γ) = prox!(y, f, x, γ)

@testset "Warmed checks allocate nothing" begin
  nested = SeparableSum(SqrNormL2(2.0), Conjugate(SeparableSum(Conjugate(SeparableSum(SqrNormL2(1.0))))))
  point = ([3.0], (([2.0, 3.0],),))
  dest = deepcopy(point)
  for op in proximal
    op(dest, nested, point, 1.0)
    @test PB.Testing.allocations(_allocation_check, op, dest, nested, point, 1.0) == 0
  end
  inner = SeparableSum(Precompose(MoreauEnvelope(NormL1(), 0.5), 2.0), SumLargest(1))
  f = SeparableSum(SqrNormL2(2.0), inner)
  x = ([3.0], ([2.0, 3.0], [4.0, 5.0]))
  y = deepcopy(x)
  for op in proximal
    op(y, f, x, 1.0)
    @test PB.Testing.allocations(_allocation_check, op, y, f, x, 1.0) == 0
  end
  _allocation_prox!(y, f, x, 1.0)
  @test (@allocated _allocation_prox!(y, f, x, 1.0)) == 0
  # the checks added for leaves that had none
  S = [2.0 0.5 0.0; 0.5 1.5 0.2; 0.0 0.2 1.0]
  for (h, z) in ((NegLog(1.0), [0.3, -0.2, 0.5]), (NegEntropy(1.0), [0.3, -0.2, 0.5]),
                 (KLDivergence([0.2, 0.3, 0.5]), [0.3, -0.2, 0.5]),
                 (Conjugate(IndBallL2(1.0)), [0.3, -0.2, 0.5]),
                 (PointwiseMinimum(SqrNormL2(), NormL1(1.0)), [0.3, -0.2, 0.5]),
                 (LogDetLoss(Matrix(1.0I, 3, 3)), S), (Spectral(IndBallL1(1.0)), S),
                 (SingularValue(IndBallL1(1.0)), S),
                 (Spectral(Conjugate(NormLinf(1.0))), S), (SingularValue(Conjugate(NormLinf(1.0))), S))
    sum = SeparableSum(SqrNormL2(2.0), h)
    point, dest = ([3.0], z), ([7.0], similar(z))
    for op in proximal
      op(dest, sum, point, 1.0)
      @test PB.Testing.allocations(_allocation_check, op, similar(z), h, z, 1.0) == 0
      @test PB.Testing.allocations(_allocation_check, op, dest, sum, point, 1.0) == 0
    end
  end
  for h in (SliceSum(Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), 2.0)),
            SliceSum([Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), 2.0)]),
            FiberSum(NormL1(); diagonal = SqrNormL2()))
    z = h isa FiberSum ? ones(2, 2, 2) : ones(2, 2)
    dest = similar(z)
    for op in proximal
      @test PB.Testing.allocations(op, dest, h, z, 1.0) == 0
      @test PB.Testing.allocations(_allocation_check, op, dest, h, z, 1.0) == 0
    end
  end
end

####################################
#  A block without a proximal operator fails before any block is written
####################################

placeholder_free(a) = !(a isa Union{PB._StepInput, PB._StepBuffer}) && (!(a isa Tuple) || all(placeholder_free, a))

# The call raises the MethodError of the proximal operator of the leaf, and writes nothing.
function nonproximable_unchanged(op, h, point, γ, Leaf; output = point, ref = nothing, exact = false)
  @nospecialize
  for alias in (false, true)
    alias && typeof(output) !== typeof(point) && continue
    z = deepcopy(point)
    y = alias ? z : deepcopy(output)
    before, input = deepcopy(y), deepcopy(z)
    e = thrown(op, y, deepcopy(h), z, γ)
    @test e isa MethodError
    if e isa MethodError
      @test e.f === prox!
      @test any(a -> a isa Leaf, e.args)
      @test all(placeholder_free, e.args)
      ref === nothing || @test same_exception(e, ref; exact)
    end
    @test isequal(y, before)
    @test isequal(z, input)
  end
end

# The routes to a leaf without a proximal operator: separable sums (nested and scaled), slice sums over a
# tuple and over an abstract-element vector, and fiber sums.
function nonproximable_routes(f, x, γ; output = x)
  @nospecialize
  T = real(eltype(x))
  Leaf = typeof(f)
  two = SqrNormL2(T(2))
  for op in proximal
    ref = thrown(op, deepcopy(output), deepcopy(f), deepcopy(x), γ)
    @test ref isa MethodError
    ref isa MethodError || continue
    point, dest = (T[3], x), (T[7], output)
    nonproximable_unchanged(op, SeparableSum(two, f), point, γ, Leaf; output = dest, ref, exact = true)
    nonproximable_unchanged(op, Scaled(SeparableSum(two, f), T(2)), point, γ / 2, Leaf; output = dest, ref, exact = true)
    nonproximable_unchanged(op, SeparableSum(two, SeparableSum(f)), (T[3], (x,)), γ, Leaf;
                            output = (T[7], (output,)), ref, exact = true)
    nonproximable_unchanged(op, SeparableSum(two, Scaled(SeparableSum(f), T(2))), (T[3], (x,)), γ / 2, Leaf;
                            output = (T[7], (output,)), ref, exact = true)
    nonproximable_unchanged(op, SeparableSum(SeparableSum(two, SeparableSum(two, f))),
                            ((T[3], (T[3], x)),), γ, Leaf; output = ((T[7], (T[7], output)),), ref)
    slices = cat(x, x; dims = ndims(x) + 1)
    sliced = cat(output, output; dims = ndims(output) + 1)
    nonproximable_unchanged(op, SliceSum(two, f), slices, γ, Leaf; output = sliced)
    nonproximable_unchanged(op, SliceSum(AbstractFunction[two, f]), slices, γ, Leaf; output = sliced)
    nonproximable_unchanged(op, SliceSum(AbstractFunction[two, two, f]), cat(slices, x; dims = ndims(x) + 1), γ, Leaf;
                            output = cat(sliced, output; dims = ndims(x) + 1))
    x isa Vector || continue
    fibers = reshape(repeat(x, 4), 2, 2, length(x))
    out = reshape(repeat(output, 4), 2, 2, length(output))
    nonproximable_unchanged(op, FiberSum(f; diagonal = two), fibers, γ, Leaf; output = out)
    nonproximable_unchanged(op, FiberSum(two; diagonal = f), fibers, γ, Leaf; output = out)
    nonproximable_unchanged(op, FiberSum(two; diagonal = f, symmetric = true), fibers, γ, Leaf; output = out)
  end
end

@testset "A block without a proximal operator fails before any block is written" begin
  for T in (Float32, Float64)
    L = LogisticLoss(ones(T, 2), ones(T, 2, 4))
    nonproximable_routes(L, ones(T, 4), T(0.5))
    nonproximable_routes(Sum(L, NormL1(T(1))), ones(T, 4), T(0.5))
    nonproximable_routes(Sum(NormL1(T(1)), L), ones(T, 4), T(0.5))
    nonproximable_routes(KronQuadratic(ones(T, 2, 2), ones(T, 2, 2)), ones(T, 2, 2), T(0.5))
  end
end

####################################
#  Calls that succeed keep their results
####################################

# A user function with a proximal operator typed on Vector, and one that defines prox_entry and no prox!.
struct UserShrink <: AbstractFunction
  c::Float64
end
(f::UserShrink)(x) = f.c * sum(abs2, x) / 2
ProximalBase.prox!(y::Vector{Float64}, f::UserShrink, x::Vector{Float64}, γ::Real) = (y .= x ./ (1 + γ * f.c); y)
struct EntryShrink <: SeparableFunction
  c::Float64
end
ProximalBase.value_entry(f::EntryShrink, v, i) = f.c * v^2 / 2
ProximalBase.prox_entry(f::EntryShrink, v, i, γ) = v / (1 + γ * f.c)
ProximalBase.check_parameters(f::EntryShrink, x) = nothing

element_type(a::Tuple) = element_type(first(a))
element_type(a::AbstractArray) = real(eltype(a))
fresh(a::Tuple) = map(fresh, a)
fresh(a::AbstractArray) = similar(a)
# The block of f in a sum against f alone, in a separate and in an aliased output.
function same_values(f, x, γ)
  @nospecialize
  T = element_type(x)
  for op in proximal
    expected = fresh(x)
    ref = op(expected, deepcopy(f), deepcopy(x), γ)
    first_block = T[3] ./ (1 + 2γ)
    s = SeparableSum(SqrNormL2(T(2)), f)
    dest = (T[7], fresh(x))
    out = op(dest, s, (T[3], deepcopy(x)), γ)
    @test isequal(dest[2], expected)
    @test isequal(dest[1], first_block)
    op === value_and_prox! && @test out ≈ ref + SqrNormL2(T(2))(dest[1]) rtol = 100eps(T)
    alias = (T[3], deepcopy(x))
    op(alias, s, alias, γ)
    @test isequal(alias[2], expected)
    @test isequal(alias[1], first_block)
  end
end

@testset "Blocks with a proximal operator are unchanged" begin
  for T in (Float32, Float64)
    x = T[0.3, -1.5, 2.0, 0.1]
    γ = T(0.5)
    for f in (NormL1(T(1)), SqrNormL2(T(2)), SumLargest(2), Conjugate(SumLargest(2)),
              Scaled(NormL1(T(1)), T(2)), Translate(SqrNormL2(T(1)), ones(T, 4)), Conjugate(SqrNormL2(T(2))),
              Tilt(SqrNormL2(T(1)), ones(T, 4)), Regularize(NormL1(T(1)), T(0.5), zeros(T, 4)),
              Precompose(NormL1(T(1)), T(2)), EntryShrink(1.0))
      same_values(f, x, γ)
    end
    u, w = reverse(x), 2 .* x
    same_values(SeparableSum(NormL1(T(1)), SqrNormL2(T(2))), (x, u), γ)
    same_values(SeparableSum(NormL1(T(1)), SeparableSum(SqrNormL2(T(2)), SumLargest(2))), (x, (u, w)), γ)
    same_values(Scaled(SeparableSum(NormL1(T(1)), SqrNormL2(T(2))), T(2)), (x, u), γ)
  end
  x, γ = [0.3, -1.5, 2.0, 0.1], 0.5
  same_values(UserShrink(2.0), x, γ)
  same_values(SeparableSum(UserShrink(2.0), NormL1(1.0)), (x, reverse(x)), γ)
  # slice and fiber sums of functions that have a proximal operator
  for T in (Float32, Float64), op in proximal
    X = T[0.3 -1.5; 2.0 0.1; 1.0 -0.4]
    for fs in ((NormL1(T(1)), SqrNormL2(T(2))), AbstractFunction[NormL1(T(1)), SqrNormL2(T(2))])
      Y = similar(X)
      op(Y, SliceSum(fs), X, T(0.5))
      @test isequal(Y[:, 1], prox!(similar(X[:, 1]), NormL1(T(1)), X[:, 1], T(0.5)))
      @test isequal(Y[:, 2], prox!(similar(X[:, 2]), SqrNormL2(T(2)), X[:, 2], T(0.5)))
      Z = copy(X)
      op(Z, SliceSum(fs), Z, T(0.5))
      @test isequal(Z, Y)
    end
    F = reshape(T.(1:12) ./ 5, 2, 2, 3)
    G = similar(F)
    op(G, FiberSum(NormL1(T(1)); diagonal = SqrNormL2(T(2))), F, T(0.5))
    @test isequal(G[1, 2, :], prox!(similar(F[1, 2, :]), NormL1(T(1)), F[1, 2, :], T(0.5)))
    @test isequal(G[2, 2, :], prox!(similar(F[2, 2, :]), SqrNormL2(T(2)), F[2, 2, :], T(0.5)))
    # the paired branch of a fused fiber sum has no oracle for its fibers
    F2 = F[:, :, 1:2]
    P = similar(F2)
    op(P, FiberSum(NormL1Fused(T(1), T(1))), F2, T(0.5))
    @test all(isfinite, P)
    @test P != F2
  end
end

# a function defined after a block call has run is judged by its own type
let y = ([0.0], [0.0, 0.0])
  prox!(y, SeparableSum(SqrNormL2(2.0), NormL1(1.0)), ([3.0], [1.0, 2.0]), 0.5)
end
struct LateShrink <: AbstractFunction
  c::Float64
end
(f::LateShrink)(x) = f.c * sum(abs2, x) / 2
ProximalBase.prox!(y::Vector{Float64}, f::LateShrink, x::Vector{Float64}, γ::Real) = (y .= x ./ (1 + γ * f.c); y)

@testset "A function defined after a first block call" begin
  y = ([7.0], [7.0, 7.0])
  prox!(y, SeparableSum(SqrNormL2(2.0), LateShrink(2.0)), ([3.0], [1.0, 2.0]), 0.5)
  @test y[2] == [1.0, 2.0] ./ 2
  same_values(LateShrink(2.0), [1.0, 2.0], 0.5)
end

####################################
#  The functions without a proximal operator are the three that the check lists
####################################

# the concrete function types that the package defines, found through the names of its module
function concrete_functions()
  types = Any[getfield(PB, n) for n in names(PB; all = true) if isdefined(PB, n)]
  types = filter(T -> T isa Type && Base.unwrap_unionall(T) isa DataType && !isabstracttype(T) &&
                      T <: AbstractFunction && Base.unwrap_unionall(T).name.module === PB, types)
  # an alias such as LogisticLoss names the type of GLMLoss that it stands for
  unique!(map(T -> Base.unwrap_unionall(T).name.wrapper, types))
end
# a built-in function type has a scalar-step proximal operator when a method of prox! with four arguments
# takes it as the function and is not the owner fallback on AbstractFunction
function has_scalar_prox(T)
  any(methods(prox!)) do m
    sig = Base.unwrap_unionall(m.sig)
    length(sig.parameters) == 5 || return false
    ft = sig.parameters[3]
    Base.unwrap_unionall(ft) === AbstractFunction && return false
    typeintersect(T, ft) !== Union{} && Base.unwrap_unionall(sig.parameters[2]) !== SymmetricSparseIterate
  end
end

@testset "The functions that the check rejects are the built-in functions without a scalar proximal operator" begin
  builtin = concrete_functions()
  @test length(builtin) > 50
  without = filter(!has_scalar_prox, builtin)
  @test Set(without) == Set([PB.GLMLoss, PB.KronQuadratic, PB.Sum])
  @test Set(filter(T -> !PB._prox_supported(T), builtin)) == Set(without)
  L = LogisticLoss(ones(2), ones(2, 4))
  for f in (L, KronQuadratic(ones(2, 2), ones(2, 2)), Sum(L, NormL1(1.0)), LogisticLoss(ones(2), ones(2, 4)))
    @test !applicable(prox!, [1.0], f, [1.0], 1.0)
    @test !PB._prox_supported(f)
  end
  @test PB._prox_supported(UserShrink(1.0))
  @test PB._prox_supported(EntryShrink(1.0))
  @test PB._prox_supported(NormL1Fused(1.0, 1.0))
end

####################################
#  The check allocates what it did
####################################

_scalar_prox!(y, f, x, γ) = prox!(y, f, x, γ)
_scalar_value_and_prox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
_scalar_check(op::O, y, f, x, γ) where {O} = PB._checkstep(op, f, y, x, γ)

@testset "Scalar-step block calls allocate what they did without the check" begin
  allocations = Dict{Any, Int}()
  for T in (Float32, Float64)
    s = SeparableSum(NormL1(T(1)), NormL1(T(1)))
    x = (T[1, -2], T[3, -4])
    y = deepcopy(x)
    X = T[1 -2 3; 4 5 6]
    slices = SliceSum(AbstractFunction[NormL1(T(1)), NormL1(T(2)), NormL1(T(3))])
    Y = similar(X)
    for (op, name) in ((_scalar_prox!, "prox!"), (_scalar_value_and_prox!, "value_and_prox!"), (_scalar_check, "check"))
      args = op === _scalar_check ? (value_and_prox!,) : ()
      op(args..., y, s, x, T(0.5))
      allocations[(T, name, "SeparableSum")] = PB.Testing.allocations(op, args..., y, s, x, T(0.5))
      op(args..., Y, slices, X, T(0.5))
      allocations[(T, name, "SliceSum")] = PB.Testing.allocations(op, args..., Y, slices, X, T(0.5))
    end
  end
  # the bytes of the call with the abstract-element vector, before the check had a trait
  base = Dict("prox!" => 1040, "value_and_prox!" => 608, "check" => 336)
  for ((T, name, kind), bytes) in allocations
    @test bytes <= (kind == "SeparableSum" ? 0 : base[name])
  end
end

end
