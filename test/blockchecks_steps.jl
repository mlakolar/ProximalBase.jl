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
                 (SingularValue(IndBallL1(1.0)), S))
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

end
