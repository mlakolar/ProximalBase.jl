module BlockChecksGridsTest

# The Scaled and nested SeparableSum variants, and the FiberSum variants with the function as the diagonal or with no
# diagonal, of the cases of "Block checks before writes" that are not in WRAPPER_VARIANT_CASES of blockchecks.jl.
# For the cases in that list blockchecks.jl runs these variants, and for all cases it runs the `@test E <: ...` check,
# the flat SeparableSum and the SliceSum, the FiberSum variant with the function as the off-diagonal part when the
# point is a vector and the function is proximable (for the proximal operations only), and the other testsets.
# `oracle!`, `failure_type`, `unchanged` and the cases of `bad_cases` are copied from it, except the
# "check_parameters" case (and its type), which is in that list. The fixtures are the same as in blockchecks.jl. No
# case here draws random numbers.

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase
const proximal = (prox!, value_and_prox!)
const smooth = (gradient!, value_and_gradient!)
const oracles = (prox!, value_and_prox!, gradient!, value_and_gradient!)

function oracle!(op, y, f, x)
  @nospecialize op y f x
  op === prox! || op === value_and_prox! ? op(y, f, x, 1.0) : op(y, f, x)
end

function failure_type(op, f, x)
  @nospecialize op f x
  try
    oracle!(op, copy(x), f, copy(x))
  catch e
    return typeof(e)
  end
  error("the standalone oracle must throw")
end

function unchanged(op, f, point, E)
  @nospecialize op f point E
  for alias in (false, true)
    x = deepcopy(point)
    y = alias ? x : deepcopy(point)
    before, input = deepcopy(y), deepcopy(x)
    expected_error = alias && op in smooth && applicable(op, y, f, x) ? ArgumentError : E
    @test_throws expected_error oracle!(op, y, f, x)
    @test isequal(y, before)
    @test isequal(x, input)
  end
end

function bad_cases()
  v, M = [3.0, 4.0], [2.0 1.0; 1.0 3.0]
  A = [1.0 0 0; 0 1 0]
  rectangular = ones(2, 3)
  cases = Any[
    ("SqrNormL2 center", SqrNormL2(1.0, ones(1, 2)), v, oracles),
    ("SqrNormL2 matrix center", SqrNormL2(1.0, ones(2, 1)), ones(1, 2), oracles),
    ("Translate translation", Translate(SqrNormL2(), ones(1, 2)), v, oracles),
    ("Translate inner", Translate(SqrNormL2(1.0, ones(1, 2)), ones(2)), v, oracles),
    ("Tilt coefficient", Tilt(SqrNormL2(), ones(1, 2)), v, oracles),
    ("Tilt inner", Tilt(SqrNormL2(1.0, ones(1, 2)), ones(2)), v, oracles),
    ("Regularize broadcast", Regularize(SqrNormL2(), 1.0, ones(1, 2)), v, oracles),
    ("Regularize value center", Regularize(SqrNormL2(), 1.0, ones(2, 1)), v, (value_and_prox!, value_and_gradient!)),
    ("Regularize inner", Regularize(SqrNormL2(1.0, ones(1, 2)), 1.0, ones(2)), v, oracles),
    ("GroupNormL2", GroupNormL2(1.0, [1:3]), v, proximal),
    ("GroupNormLinf", GroupNormLinf(1.0, [1:3]), v, proximal),
    ("SparseGroupNormL2", SparseGroupNormL2(1.0, 1.0, [1:3]), v, proximal),
    ("GroupMCP", GroupMCP(1.0, [1:3]), v, proximal),
    ("GroupSCAD", GroupSCAD(1.0, [1:3]), v, proximal),
    ("IndBallGroupL0", IndBallGroupL0(1, [1:3]), v, proximal),
    ("IndHyperplane shape", IndHyperplane(ones(1, 2), 1.0), v, proximal),
    ("IndHalfspace shape", IndHalfspace(ones(1, 2), 1.0), v, proximal),
    ("IndHyperplane length", IndHyperplane(ones(3), 1.0), v, proximal),
    ("IndHalfspace length", IndHalfspace(ones(3), 1.0), v, proximal),
    ("IndAffine Cholesky", IndAffine(ones(1, 3), ones(1)), v, proximal),
    ("IndAffine tight frame", IndAffine(ones(1, 3), ones(1); μ = 3.0), v, proximal),
    ("IndAffine SVD", IndAffine(ones(2, 3), ones(2)), v, proximal),
    ("LeastSquaresLoss columns", LeastSquaresLoss(ones(2), A), v, oracles),
    ("LeastSquaresLoss responses", LeastSquaresLoss(ones(2, 2), ones(2, 2)), ones(2, 3), oracles),
    ("LogisticLoss", LogisticLoss(ones(2), A), v, smooth),
    ("PoissonLoss", PoissonLoss(ones(2), A), v, smooth),
    ("HuberLoss", HuberLoss(ones(2), A, 1.0), v, smooth),
    ("QuadraticFunction", QuadraticFunction(Matrix(1.0I, 3, 3)), v, oracles),
    ("KronQuadratic vector", KronQuadratic(Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2)), v, smooth),
    ("KronQuadratic shape", KronQuadratic(Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2)), ones(1, 4), smooth),
    ("LogDetLoss data", LogDetLoss(Matrix(1.0I, 3, 3)), M, oracles),
    ("LogDetLoss square", LogDetLoss(Matrix(1.0I, 2, 2)), rectangular, oracles),
    ("Spectral", Spectral(NormL1(1.0)), rectangular, proximal),
  ]
  for partial in (false, true)
    push!(cases, ("IndPSD partial=$partial", IndPSD(; partial), rectangular, proximal))
    push!(cases, ("IndPSD rank partial=$partial", IndPSD(; rank = 1, partial), rectangular, proximal))
    push!(cases, ("TracePSD partial=$partial", TracePSD(1.0; partial), rectangular, proximal))
  end
  cases
end

# the cases for which blockchecks.jl runs the variants
const WRAPPER_VARIANT_CASES = ("SqrNormL2 center", "Tilt coefficient", "GroupNormL2", "IndHyperplane shape",
                               "LeastSquaresLoss columns", "LogisticLoss", "LogDetLoss data", "check_parameters",
                               "IndPSD partial=false")

@testset "Block checks before writes" begin
  for (name, f, x, ops) in filter(c -> !(c[1] in WRAPPER_VARIANT_CASES), bad_cases())
    @testset "$name" begin
      for op in ops
        E = failure_type(op, f, x)
        fs = SeparableSum(SqrNormL2(2.0), SqrNormL2(3.0), f)
        point = ([3.0], [4.0], x)
        unchanged(op, Scaled(fs, 2.0), point, E)
        nested = SeparableSum(SqrNormL2(2.0), SeparableSum(SqrNormL2(3.0), f))
        unchanged(op, nested, ([3.0], ([4.0], x)), E)
      end

      if x isa AbstractVector && isproximable(f)
        fibers = reshape(repeat(x, 4), 2, 2, length(x))
        for op in ops
          op in proximal || continue
          xv = PB._fiber(fibers, CartesianIndex(2, 1))
          E = failure_type(op, f, xv)
          for symmetric in (false, true)
            unchanged(op, FiberSum(SqrNormL2(2.0); diagonal = f, symmetric), fibers, E)
            unchanged(op, FiberSum(f; diagonal = nothing, symmetric), fibers, E)
          end
        end
      end
    end
  end
end

end
