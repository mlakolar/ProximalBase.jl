module GradientAliasTest

using Test, ProximalBase, LinearAlgebra, SparseArrays
using JLArrays: JLArray
import Adapt
using ..TestUtils: allocs

const PB = ProximalBase
const alias_error = ArgumentError("the gradient must not overwrite x")

# Only the mutating value/gradient oracle is needed for the gradient fallback.
struct CustomSquare <: AbstractFunction end
PB.issmooth(::Type{CustomSquare}) = true
(::CustomSquare)(x) = sum(abs2, x) / 2
function PB.value_and_gradient!(g, f::CustomSquare, x)
  PB._checkwrite(value_and_gradient!, f, g, x)
  copyto!(g, x)
  f(x)
end

function check_alias(f, x)
  before = copy(x)
  reference = gradient(f, x)
  value = f(x)
  for op in (gradient!, value_and_gradient!)
    @test_throws alias_error op(x, f, x)
    @test x == before
    g = similar(reference)
    result = op(g, f, x)
    @test g ≈ reference
    @test op === gradient! ? result === g : result ≈ value
    @test x == before
  end
end

@testset "gradient output must be separate" begin
  x = [0.2, 0.4, 0.6, 0.7]
  c = [1.0, 0.0, 0.5, -1.0]
  Q = Matrix{Float64}(I, 4, 4) + ones(4, 4)
  A = Matrix{Float64}(I, 4, 4)
  for f in (ZeroFunction(), LinearFunction(ones(4)), SqrNormL2(1.5),
            SqrNormL2(1.0, c), QuadraticFunction(Q, c),
            SqrHingeLoss(), BernoulliLoss(ones(4)),
            LogisticLoss([0., 1., 0., 1.], A), PoissonLoss(ones(4), A),
            HuberLoss(ones(4), A, 0.5), LeastSquaresLoss(c, A),
            Scaled(SqrNormL2(1.0, c), 2.0), 2.0 * SqrNormL2(1.0, c),
            Scaled(CubeNormL2(), 2.0),
            Tilt(SqrNormL2(), ones(4)), Translate(SqrNormL2(), c),
            Regularize(LinearFunction(ones(4)), 1.0),
            Regularize(LinearFunction(ones(4)), 1.0, c),
            Sum(LinearFunction(ones(4)), SqrNormL2()),
            Sum(LinearFunction(ones(4)), SqrNormL2(), CubeNormL2()),
            Precompose(SqrNormL2(), A, 1.0), Precompose(SqrNormL2(), -2.0),
            Precompose(SqrNormL2(), [1., 2., 3., 4.]),
            MoreauEnvelope(NormL1(), 0.5), SqrDistL2(IndBallL2()), CustomSquare())
    @testset "$(typeof(f))" begin
      check_alias(f, copy(x))
      check_alias(f, SparseIterate(x))
    end
  end

  X = [0.2 0.4; 0.4 0.7]
  C = ones(2, 2)
  P = Matrix{Float64}(I, 2, 2)
  for f in (ZeroFunction(), LinearFunction(C), SqrNormL2(1.5), SqrNormL2(1., C),
            SqrHingeLoss(), BernoulliLoss(C),
            KronQuadratic(P, P, C), LogDetLoss(P), LeastSquaresLoss(C, P),
            Scaled(SqrNormL2(), 2.), Tilt(SqrNormL2(), C), Translate(SqrNormL2(), C),
            Regularize(ZeroFunction(), 1., C), Sum(SqrNormL2(), CubeNormL2()),
            Precompose(SqrNormL2(), -2.), Precompose(SqrNormL2(), C),
            MoreauEnvelope(NormL1(), 0.5), SqrDistL2(IndBallL2()))
    @testset "matrix $(typeof(f))" begin
      input = f isa LogDetLoss ? X + P : X
      check_alias(f, copy(input))
      check_alias(f, SymmetricSparseIterate(input))
    end
  end

  check_alias(SliceSum(QuadraticFunction(Q, c), SqrNormL2()), hcat(x, x))
  check_alias(SliceSum([SqrNormL2(), SqrNormL2()]), hcat(x, x))
  check_alias(SliceSum(KronQuadratic(P, P), LogDetLoss(P)), cat(X, X + P; dims=3))
  @test !issmooth(FiberSum(SqrNormL2()))
end

@testset "CubeNormL2 documented aliasing" begin
  f = CubeNormL2()
  X = [0.2 0.4; 0.4 0.7]
  for x in ([0.2, 0.4, 0.6, 0.7], X, SymmetricSparseIterate(X))
    reference = gradient(f, x)
    value = f(x)
    for op in (gradient!, value_and_gradient!)
      alias = copy(x)
      result = op(alias, f, alias)
      @test alias ≈ reference
      @test op === gradient! ? result === alias : result ≈ value
    end
  end
end

@testset "tuple blocks checked before any writes" begin
  f = SeparableSum(SqrNormL2(1.5), Translate(SqrNormL2(), ones(2)))
  x = ([0.2, 0.4], [0.6, 0.7])
  reference = gradient(f, x)
  for op in (gradient!, value_and_gradient!), block in (1, 2)
    g = block == 1 ? (x[1], fill(-7., 2)) : (fill(-7., 2), x[2])
    before_x = map(copy, x)
    before_g = map(copy, g)
    @test_throws alias_error op(g, f, x)
    @test x == before_x
    @test g == before_g
    g = map(similar, x)
    result = op(g, f, x)
    @test all(map(isapprox, g, reference))
    @test op === gradient! ? result === g : result ≈ f(x)
  end
  @test_throws alias_error gradient!(x, f, x)
  @test_throws alias_error value_and_gradient!(x, f, x)
  g = map(similar, x)
  @test allocs(PB._checkgradient_output, g, x) == 0
  @test allocs(gradient!, g, f, x) == 0
  @test allocs(value_and_gradient!, g, f, x) == 0
  @test allocs(gradient!, zeros(2), SqrNormL2(), x[1]) == 0
  @test allocs(value_and_gradient!, zeros(2), SqrNormL2(), x[1]) == 0
end

@testset "device GLM gradient output" begin
  A = Matrix{Float64}(I, 2, 2)
  f = Adapt.adapt(JLArray, LogisticLoss([0., 1.], A))
  x = JLArray([0.2, 0.4])
  before = Array(x)
  reference = gradient(f, x)
  for op in (gradient!, value_and_gradient!)
    @test_throws alias_error op(x, f, x)
    @test Array(x) == before
    g = similar(x)
    result = op(g, f, x)
    @test Array(g) ≈ Array(reference)
    @test op === gradient! ? result === g : result ≈ f(x)
  end
end

end
