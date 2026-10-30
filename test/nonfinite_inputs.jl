module NonfiniteInputsTest

using Test
using ProximalBase
using LinearAlgebra
using ..TestUtils

const PB = ProximalBase

@testset "Equivalent linear sets on nonfinite inputs" begin
  for a in (1.0, 1e200, [1.0, -2.0], [0.0, 1.0], [1e200, -2e200]), γ in (0.7, [0.5, 1.0]), v in (NaN, Inf, -Inf)
    x = [v, 0.0]
    for (slab, set) in ((IndHyperslab(a, 1, 1), IndHyperplane(a, 1)),
                        (IndHyperslab(a, -Inf, 1), IndHalfspace(a, 1)),
                        (IndHyperslab(a, -1, Inf), IndHalfspace(-a, 1)))
      for op in (prox!, value_and_prox!), alias in (false, true)
        input = copy(x)
        y = alias ? input : zeros(2)
        ref = prox(set, x, γ)
        result = op(y, slab, input, γ)
        @test isequal(y, ref)
        op === value_and_prox! && @test result === 0.0
        alias || @test isequal(input, x)
      end
    end
    @test isequal(prox(IndHyperslab(a, -Inf, Inf), x, γ), x)
    @test all(isnan, prox(IndHyperslab(a, -1, 1), [NaN, 0.0], γ))
  end
end

@testset "NaN multiplier and preserved history" begin
  for set in (IndHyperplane(1, 1), IndHalfspace(1, 1), IndSimplex(1), IndHyperslab(1, 0, 1)), γ in (1.0, [0.5, 1.0])
    c = Constrained(SqrNormL2(), set)
    x = [3.0, 2.0]
    prox(c, x, γ)
    last, slope = c.last[], c.slope[]
    for op in (prox!, value_and_prox!)
      y = zeros(2)
      result = op(y, c, [NaN, 0.0], γ)
      @test all(isnan, y)
      @test isequal(c.last[], last)
      @test isequal(c.slope[], slope)
      op === value_and_prox! && @test result === (set isa IndSimplex ? Inf : NaN)
    end
    @test isequal(prox(c, x, γ), prox(Constrained(SqrNormL2(), set), x, γ))
  end
end

@testset "NaN termination inside bracketing and refinement" begin
  c = Constrained(SqrNormL2(), IndHyperplane(1, 1))
  x = [NaN, 0.0]
  @test isnan(PB._bracket_and_refine(c, x, 1.0, 1.0, 0.0, 1.0, 1.0, nothing))
  @test isnan(PB._refine(c, x, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0, nothing))
  c = Constrained(IndBox(0, 1), IndHyperplane(1, 1))
  finite = collect(range(-1.0, 2.0; length = 5000))
  expected = prox(deepcopy(c), finite)
  prox(c, finite)
  last, slope = c.last[], c.slope[]
  bad = copy(finite)
  bad[end] = NaN
  @test all(isnan, prox(c, bad))
  @test isequal(c.last[], last)
  @test isequal(c.slope[], slope)
  @test isequal(prox(c, finite), expected)
end

@testset "Coupled algorithms return NaN" begin
  for v in (NaN, Inf, -Inf), op in (prox!, value_and_prox!), alias in (false, true)
    for (f, input) in ((IndPolyhedron([1.0 1.0], -Inf, 1), [v, 0.0]),
                       (IndStiefel(), reshape([v, 0.0], 2, 1)),
                       (SumLargest(1), [v, 0.0]), (Conjugate(SumLargest(1)), [v, 0.0]))
      x = copy(input)
      y = alias ? x : zero.(x)
      result = op(y, f, x, 1.0)
      @test all(isnan, y)
      alias || @test isequal(x, input)
      op === value_and_prox! && @test result === (f isa SumLargest ? NaN : f isa IndStiefel ? Inf : 0.0)
    end
  end
  f = IndPolyhedron(ones(Float32, 1, 2), -Inf, 1)
  @test all(isnan, prox(f, [1e100, 0.0]))
  @test !f.valid
  @test prox(f, [0.0, 0.0]) == zeros(2)
end

@testset "Nonfinite block reproductions" begin
  slab = IndHyperslab([1.0, 1.0], 1, 1)
  for v in (NaN, Inf, -Inf), f in (slab, Constrained(SqrNormL2(), slab),
      MoreauEnvelope(slab, 0.5), DistL2(slab), SqrDistL2(slab), Scaled(slab, 2.0),
      PointwiseMinimum(slab, IndHyperslab([1.0, 1.0], 2, 2))),
      op in (prox!, value_and_prox!), alias in (false, true)
    input = ([3.0], [v, 0.0])
    x = deepcopy(input)
    y = alias ? x : ([7.0], [9.0, 9.0])
    result = op(y, SeparableSum(SqrNormL2(2.0), f), x, 1.0)
    @test y[1] == [1.0]
    # The equality projection subtracts v/2 from both entries.
    expected = [v - v / 2, -v / 2]
    if f isa PB.LinearConstrained
      expected = isnan(v) ? [NaN, NaN] : [v, 0.0]
    elseif f isa DistL2 || f isa MoreauEnvelope && v == -Inf
      expected = [NaN, NaN]
    end
    @test isequal(y[2], expected)
    alias || @test isequal(x, input)
    if op === value_and_prox!
      value = f isa Union{IndHyperslab, Scaled} ? 1.0 : f isa PB.LinearConstrained && !isnan(v) ? Inf : NaN
      @test result === value
    end
  end
end

@testset "Zero-step distance shortcuts" begin
  slab = IndHyperslab(1, 0, 1)
  for F in (DistL2, SqrDistL2), v in (NaN, Inf, -Inf)
    x = [v, 0.0]
    @test isequal(prox(F(slab), x, 0.0), x)
  end
end

@testset "Nonfinite decompositions and Moreau checks" begin
  for f in (NuclearNorm(), IndBallRank(1), IndPSD(), IndPSD(rank = 1), TracePSD()), v in (NaN, Inf, -Inf)
    x = [v 0.0; 0.0 1.0]
    y = zeros(2, 2)
    fresh = deepcopy(f)
    value = value_and_prox!(y, f, x, 1.0)
    @test all(isnan, y)
    @test value === Inf
    finite = [2.0 0.0; 0.0 1.0]
    @test isequal(prox(f, finite), prox(fresh, finite))
  end
  for f in (Conjugate(SumLargest(1)), Precompose(SumLargest(1), 2.0))
    @test all(isnan, prox(f, [NaN, 0.0]))
  end
  # Moreau subtraction must still be finite at each finite input entry.
  f = Conjugate(IndPolyhedron([1.0 1.0], -Inf, 1))
  y = [7.0, 9.0]
  @test_throws ArgumentError prox!(y, f, [NaN, 0.0], 1.0)
  @test y == [7.0, 9.0]
  # This remains a documented failure during computation in a block sum.
  point = ([3.0], [NaN, 0.0])
  @test_throws ArgumentError prox!(point, SeparableSum(SqrNormL2(2.0), f), point, 1.0)
  @test point[1] == [1.0]
  @test isequal(point[2], [NaN, 0.0])
  c = Constrained(IndBox(0, 1), IndHyperplane(1, 1))
  @test_throws ArgumentError prox(c, [1e300, 1e300])
end

@testset "Excluded empty arrays fail before any block writes" begin
  for f in (IndHyperslab(1, 1, 2), Constrained(SqrNormL2(), IndHyperslab(1, 1, 2))),
      op in (prox!, value_and_prox!), alias in (false, true)
    x = ([3.0], Float64[])
    y = alias ? x : ([7.0], Float64[])
    before, input = deepcopy(y), deepcopy(x)
    @test_throws ArgumentError op(y, SeparableSum(SqrNormL2(2.0), f), x, 1.0)
    @test isequal(y, before)
    @test isequal(x, input)
  end
end

@testset "Equivalent spectral proximal oracles on nonfinite inputs" begin
  for T in (Float32, Float64), v in (T(NaN), T(Inf), -T(Inf)),
      I in (CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(2, 1))
    input = T[2 0; 0 1]
    input[I] = v
    pairs = ((SingularValue(NormL1(one(T))), NuclearNorm(one(T))),
             (SingularValue(NormL1(zero(T))), NuclearNorm(zero(T))),
             (Spectral(NormL1Nonneg(one(T))), TracePSD(one(T))),
             (Spectral(NormL1Nonneg(zero(T))), TracePSD(zero(T))),
             (Spectral(IndNonnegative()), IndPSD()),
             (SingularValue(IndBallL0(1)), IndBallRank(1)),
             (SingularValue(IndBallL0(2)), IndBallRank(2)))
    for (f, g) in pairs, op in (prox!, value_and_prox!), alias in (false, true)
      x, z = copy(input), copy(input)
      y, w = alias ? (x, z) : (zeros(T, 2, 2), zeros(T, 2, 2))
      vf = op(y, f, x, one(T))
      vg = op(w, g, z, one(T))
      identity = g isa NuclearNorm && iszero(g.λ) || g isa IndBallRank && g.r >= minimum(size(input))
      @test isequal(y, fill(T(NaN), 2, 2))
      # The native identity shortcut needs no decomposition; the wrapper does.
      @test isequal(w, identity ? input : y)
      if op === value_and_prox!
        @test vg === vf
        @test vf === T(Inf)
      else
        @test vf === y
        @test vg === w
      end
      if !alias
        @test isequal(x, input)
        @test isequal(z, input)
      end
      # No decomposition workspace is created for a nonfinite argument.
      f isa Spectral && @test f.eig === nothing
      f isa SingularValue && @test f.ws === nothing
    end
    for f in (Spectral(NormL1(one(T))), LogDetLoss(Matrix{T}(LinearAlgebra.I, 2, 2), one(T))),
        op in (prox!, value_and_prox!), alias in (false, true)
      x = copy(input)
      y = alias ? x : zeros(T, 2, 2)
      result = op(y, f, x, one(T))
      @test isequal(y, fill(T(NaN), 2, 2))
      op === value_and_prox! && @test result === T(Inf)
      alias || @test isequal(x, input)
      f isa Spectral && @test f.eig === nothing
      f isa LogDetLoss && @test iszero(f.eig.A)
    end
  end
end

@testset "Spectral proximal history and warmed allocations" begin
  for T in (Float32, Float64), f in (Spectral(NormL1(T(0.3))),
      Spectral(IndNonnegative()), SingularValue(NormL1(T(0.3))),
      SingularValue(IndBallL0(1)), LogDetLoss(Matrix{T}(I, 2, 2), one(T)))
    x = T[2 0.25; 0.25 1]
    y = similar(x)
    expected = prox(deepcopy(f), x, one(T))
    for op in (prox!, value_and_prox!)
      op(y, f, x, one(T))
      @test allocs(op, y, f, x, one(T)) == 0
      bad = copy(x)
      bad[2, 1] = T(NaN)
      op(y, f, bad, one(T))
      @test allocs(op, y, f, bad, one(T)) == 0
      op(y, f, x, one(T))
      @test isequal(y, expected)
    end
  end
end

end
