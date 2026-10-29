module BlockChecksStepsTest

using Test
using ProximalBase

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
same_exception(e, ref) = typeof(e) === typeof(ref) && sprint(showerror, e) == sprint(showerror, ref)

function unchanged(op, h, x, γ, ref; output = x)
  @nospecialize
  for alias in (false, true)
    alias && typeof(output) !== typeof(x) && continue
    z = deepcopy(x)
    y = alias ? z : deepcopy(output)
    before, input = deepcopy(y), deepcopy(z)
    e = thrown(op, y, deepcopy(h), z, γ)
    @test same_exception(e, ref)
    @test isequal(y, before)
    @test isequal(z, input)
  end
end

function block_failure(f, x, γ; output = x)
  @nospecialize
  for op in proximal
    ref = thrown(op, deepcopy(output), deepcopy(f), deepcopy(x), γ)
    @test ref !== nothing
    ref === nothing && continue
    point, dest = ([3.0], x), ([7.0], output)
    s = SeparableSum(SqrNormL2(2.0), f)
    unchanged(op, s, point, γ, ref; output = dest)
    unchanged(op, Scaled(s, 2.0), point, γ / 2, ref; output = dest)
    unchanged(op, SeparableSum(SqrNormL2(2.0), SeparableSum(f)),
              ([3.0], (x,)), γ, ref; output = ([7.0], (output,)))
    unchanged(op, SeparableSum(SqrNormL2(2.0), Scaled(SeparableSum(f), 2.0)),
              ([3.0], (x,)), γ / 2, ref; output = ([7.0], (output,)))
    x isa Vector || continue
    refslice = thrown(value_and_prox!, deepcopy(output), deepcopy(f), deepcopy(x), γ)
    refslice === nothing || unchanged(op, SliceSum(SqrNormL2(2.0), f), hcat(x, x), γ,
                                     refslice; output = hcat(output, output))
    fibers = reshape(repeat(x, 4), 2, 2, length(x))
    out = reshape(repeat(output, 4), 2, 2, length(output))
    unchanged(op, FiberSum(f; diagonal = SqrNormL2(2.0)), fibers, γ, ref; output = out)
    unchanged(op, FiberSum(SqrNormL2(2.0); diagonal = f, symmetric = true), fibers, γ, ref; output = out)
  end
end

# Compare successful calls and failures to the standalone oracle, including its writes.
function same_outcome(f, x, γ; output = x)
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
      @test same_exception(e, ref)
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
  inner = SeparableSum(Precompose(MoreauEnvelope(NormL1(), 0.5), 2.0), SumLargest(1))
  f = SeparableSum(SqrNormL2(2.0), inner)
  x = ([3.0], ([2.0, 3.0], [4.0, 5.0]))
  y = deepcopy(x)
  for op in proximal
    op(y, f, x, 1.0)
    @test PB.Testing.allocations(_allocation_check, op, y, f, x, 1.0) == 0
  end
  if VERSION >= v"1.11"
    _allocation_prox!(y, f, x, 1.0)
    @test (@allocated _allocation_prox!(y, f, x, 1.0)) == 0
  end
  for h in (SliceSum(Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), 2.0)),
            SliceSum([Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), 2.0)]),
            FiberSum(NormL1(); diagonal = SqrNormL2()))
    z = h isa FiberSum ? ones(2, 2, 2) : ones(2, 2)
    dest = similar(z)
    for op in proximal
      @test PB.Testing.allocations(op, dest, h, z, 1.0) == 0
    end
  end
end

end
