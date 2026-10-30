module NestedBoxValuesTest

using Test, ProximalBase
using ..TestUtils

const PB = ProximalBase

nested(f, box) = Constrained(Constrained(f, IndBox(0, 1)), box)

function check_pair(inside, outside, x, E, expected)
  for alias in (false, true)
    point = ([3.0], copy(x))
    dest = alias ? point : ([7.0], fill(9.0, length(x)))
    input, before = deepcopy(point), deepcopy(dest)
    f = SeparableSum(SqrNormL2(2.0), inside)
    @test_throws E value_and_prox!(dest, f, point, 1.0)
    @test isequal(point, input)
    @test isequal(dest, before)
    @test point[1] == [3.0]

    ref = similar(x)
    value = value_and_prox!(ref, outside, x, 1.0)
    @test value === Inf
    @test ref == expected
    point = ([3.0], copy(x))
    dest = alias ? point : ([7.0], fill(9.0, length(x)))
    input = deepcopy(point)
    f = SeparableSum(SqrNormL2(2.0), outside)
    @test value_and_prox!(dest, f, point, 1.0) === 1.0 + value
    @test dest[1] == [1.0]
    @test isequal(dest[2], ref)
    alias || @test isequal(point, input)
  end
end

@testset "Unavailable values through outer wrappers" begin
  x = [0.5, 0.5]
  c, b = [0.125, -0.25], [0.25, -0.25]
  for leaf in (Conjugate(ElasticNet(1.0, 1.0)), SqrNormL2(1.0, ones(1, 2)))
    E = leaf isa Conjugate ? ArgumentError : DimensionMismatch
    inside = nested(leaf, IndBox(-1, 2))
    outside = nested(leaf, IndBox(2, 3))
    check_pair(Scaled(inside, 2.0), Scaled(outside, 2.0), x, E, [2.0, 2.0])
    check_pair(Tilt(inside, c), Tilt(outside, c), x, E, [2.0, 2.0])
    check_pair(Translate(inside, b), Translate(outside, b), x, E, [2.25, 1.75])
  end
end

@testset "Availability forwarding inside nested boxes" begin
  x = [0.5, 0.5]
  for leaf in (Conjugate(ElasticNet(1.0, 1.0)), SqrNormL2(1.0, ones(1, 2))),
      wrap in (f -> Scaled(f, 2.0), f -> Tilt(f, [0.125, -0.25]),
               f -> Translate(f, [0.25, -0.25]))
    E = leaf isa Conjugate ? ArgumentError : DimensionMismatch
    inner = Constrained(wrap(Constrained(leaf, IndBox(0, 1))), IndBox(-1, 2))
    inside = Constrained(inner, IndBox(-1, 2))
    outside = Constrained(inner, IndBox(2, 3))
    check_pair(inside, outside, x, E, [2.0, 2.0])
  end
  # An available inner value takes Tilt's generic check as well.
  available = Tilt(SqrNormL2(), [0.125, -0.25])
  @test PB._box_value_available(available, x) === true
  f = nested(available, IndBox(-1, 2))
  point = ([3.0], copy(x))
  value = value_and_prox!(point, SeparableSum(SqrNormL2(2.0), f), point, 1.0)
  @test point[1] == [1.0]
  @test point[2] == [0.1875, 0.375]
  @test value === 1.017578125
  # Precompose uses the generic availability check, which catches this mismatch.
  leaf = Precompose(SqrNormL2(1.0, ones(1, 2)), 2.0)
  @test_throws DimensionMismatch PB._checkvalue(leaf, x)
  @test PB._box_value_available(leaf, x) === false
  check_pair(nested(leaf, IndBox(-1, 2)), nested(leaf, IndBox(2, 3)),
    x, DimensionMismatch, [2.0, 2.0])
end

@testset "Array steps through the scaled block preflight" begin
  x, γ = [0.5, 0.5], [0.5, 1.5]
  for leaf in (Conjugate(ElasticNet(1.0, 1.0)), SqrNormL2(1.0, ones(1, 2)))
    E = leaf isa Conjugate ? ArgumentError : DimensionMismatch
    inside = Scaled(nested(leaf, IndBox(-1, 2)), 2.0)
    outside = Scaled(nested(leaf, IndBox(2, 3)), 2.0)
    for alias in (false, true)
      point = ([3.0], copy(x))
      dest = alias ? point : ([7.0], [9.0, 9.0])
      input, before = deepcopy(point), deepcopy(dest)
      sumf = SeparableSum(SqrNormL2(2.0), inside)
      # Block oracles take scalar steps; check array forwarding before any write.
      @test_throws E PB._checkwrite(value_and_prox!, sumf, dest, point, γ)
      @test isequal(point, input)
      @test isequal(dest, before)
      y = alias ? point[2] : dest[2]
      @test_throws E value_and_prox!(y, inside, point[2], γ)
      @test isequal(point, input)
      @test isequal(dest, before)

      sumf = SeparableSum(SqrNormL2(2.0), outside)
      PB._checkwrite(value_and_prox!, sumf, dest, point, γ)
      @test isequal(point, input)
      @test isequal(dest, before)
      ref = zeros(2)
      value = value_and_prox!(ref, outside, x, γ)
      @test value === Inf
      @test ref == [2.0, 2.0]
      @test value_and_prox!(y, outside, point[2], γ) === value
      @test isequal(y, ref)
      @test point[1] == [3.0]
    end
  end
end

@testset "Empty step-aware block recursion" begin
  f = SeparableSum()
  @test PB._checkwrite(value_and_prox!, f, (), (), 1.0) === true
  @test Base.invokelatest(PB._checkwrite_blocks, value_and_prox!, (), (), (), 1.0) === true
  @test prox!((), f, (), 1.0) === ()
end

end
