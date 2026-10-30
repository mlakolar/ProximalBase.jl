module MetricStepsTest

using Test
using ProximalBase
using ..TestUtils

function check_rejection(f, x, γ, name = nameof(typeof(f)))
  expected = ArgumentError("the steps of $name must be positive and finite")
  original = copy(x)
  for op in (prox!, value_and_prox!)
    y = fill(NaN, size(x))
    @test_throws expected op(y, f, x, γ)
    @test all(isnan, y)
    @test isequal(x, original)
    alias = copy(x)
    @test_throws expected op(alias, f, alias, γ)
    @test isequal(alias, original)
  end
end

@testset "coupling sets reject invalid metric steps before writing" begin
  sets = (IndHyperplane(ones(4), 1), IndHalfspace(ones(4), 1),
          IndHyperslab(ones(4), 0, 1), IndSimplex(1), IndSimplex(1; equality = false),
          IndSimplex(0), IndSimplex(0; equality = false), IndBallL1(1), IndBallL1(0),
          IndIsotonic(), IndIsotonic(decreasing = true))
  inputs = ([0.1, 0.2, 0.3, 0.4], [2., -1., 3., -2.], zeros(4), fill(-1., 4))
  for f in sets, x in inputs, i in (1, 4), bad in (0., -1., Inf, -Inf, NaN)
    γ = ones(4)
    γ[i] = bad
    check_rejection(f, x, γ)
  end
end

@testset "warm multiplier searches reject invalid metric steps" begin
  for set in (IndHyperplane(ones(4), 1), IndHalfspace(ones(4), 1),
              IndHyperslab(ones(4), 0, 1), IndSimplex(1), IndSimplex(1; equality = false),
              IndSimplex(0), IndSimplex(0; equality = false)),
      inner in (NormL1(0.2), Constrained(NormL1(0.2), IndBox(-2, 2)))
    f = Constrained(inner, set)
    for x in (fill(0.1, 4), [2., -1., 3., -2.], zeros(4))
      prox(f, x, ones(4))
      for i in (1, 4), bad in (0., -1., Inf, -Inf, NaN)
        γ = ones(4)
        γ[i] = bad
        check_rejection(f, x, γ)
      end
    end
  end
end

@testset "wrappers validate effective metric steps" begin
  x = [2., -1., 3., -2.]
  for set in (IndHyperplane(ones(4), 1), IndHalfspace(ones(4), 1),
              IndHyperslab(ones(4), 0, 1), IndSimplex(1), IndBallL1(1), IndIsotonic())
    for f in (Scaled(set, 2.), Tilt(set, ones(4)), Translate(set, ones(4)), Conjugate(set))
      prox(f, x, ones(4))
      for i in (1, 4), bad in (0., -1., Inf, -Inf, NaN)
        γ = ones(4)
        γ[i] = bad
        check_rejection(f, x, γ, nameof(typeof(set)))
      end
    end
    # Positive caller steps may overflow or underflow after scaling.
    for (α, step) in ((floatmax(Float64), 2.), (floatmin(Float64), nextfloat(0.)))
      check_rejection(Scaled(set, α), x, fill(step, 4), nameof(typeof(set)))
    end
  end
end

@testset "scalar and separable steps retain their behavior" begin
  x = [2., -1., 3., -2.]
  for f in (IndHyperplane(ones(4), 1), IndHalfspace(ones(4), 1),
            IndHyperslab(ones(4), 0, 1), IndSimplex(1), IndBallL1(1), IndIsotonic())
    expected = prox(f, x, 1.)
    for γ in (0., Inf, -1., NaN)
      @test isequal(prox(f, x, γ), expected)
    end
  end
  γ = [0., Inf, 0., Inf]
  for f in (NormL1(1), Constrained(NormL1(1), IndBox(-2, 2)))
    @test prox(f, x, γ) == [prox_entry(f, x[i], i, γ[i]) for i in eachindex(x)]
  end
  f = GroupNormL2(1, [1:2, 3:4])
  @test prox(f, x, [0., Inf]) == [2., -1., 0., 0.]
end

end
