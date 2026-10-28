module BlockChecksTest

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
    @test_throws E oracle!(op, y, f, x)
    @test isequal(y, before)
    @test isequal(x, input)
  end
end

struct CheckedSquare <: SeparableFunction
  n::Int
end
PB.check_parameters(f::CheckedSquare, x) = length(x) == f.n ? nothing : throw(DimensionMismatch("wrong number of entries"))
PB.value_entry(::CheckedSquare, v, i) = abs2(v) / 2
PB.prox_entry(::CheckedSquare, v, i, γ) = v / (1 + γ)

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
    ("check_parameters", CheckedSquare(3), v, proximal),
  ]
  for partial in (false, true)
    push!(cases, ("IndPSD partial=$partial", IndPSD(; partial), rectangular, proximal))
    push!(cases, ("IndPSD rank partial=$partial", IndPSD(; rank = 1, partial), rectangular, proximal))
    push!(cases, ("TracePSD partial=$partial", TracePSD(1.0; partial), rectangular, proximal))
  end
  cases
end

@testset "Block checks before writes" begin
  for (name, f, x, ops) in bad_cases()
    @testset "$name" begin
      for op in ops
        E = failure_type(op, f, x)
        @test E <: Union{DimensionMismatch, BoundsError}
        fs = SeparableSum(SqrNormL2(2.0), SqrNormL2(3.0), f)
        point = ([3.0], [4.0], x)
        unchanged(op, fs, point, E)
        unchanged(op, Scaled(fs, 2.0), point, E)
        nested = SeparableSum(SqrNormL2(2.0), SeparableSum(SqrNormL2(3.0), f))
        unchanged(op, nested, ([3.0], ([4.0], x)), E)
      end

      slices = cat(x, x, x; dims = ndims(x) + 1)
      s = SliceSum(SqrNormL2(2.0), SqrNormL2(3.0), f)
      # SliceSum's proximal and gradient paths call the value oracles.
      for op in ops
        inner = op === prox! ? value_and_prox! : op === gradient! ? value_and_gradient! : op
        xv = PB._slice(slices, 3)
        E = failure_type(inner, f, xv)
        unchanged(op, s, slices, E)
      end

      if x isa AbstractVector && isproximable(f)
        fibers = reshape(repeat(x, 4), 2, 2, length(x))
        for op in ops
          op in proximal || continue
          xv = PB._fiber(fibers, CartesianIndex(2, 1))
          E = failure_type(op, f, xv)
          for symmetric in (false, true)
            unchanged(op, FiberSum(f; diagonal = SqrNormL2(2.0), symmetric), fibers, E)
            unchanged(op, FiberSum(SqrNormL2(2.0); diagonal = f, symmetric), fibers, E)
            unchanged(op, FiberSum(f; diagonal = nothing, symmetric), fibers, E)
          end
        end
      end
    end
  end

  @testset "Regularize value paths" begin
    for c in (ones(1, 2), ones(2, 1))
      r = Regularize(SqrNormL2(), 1.0, c)
      for op in (value_and_prox!, value_and_gradient!)
        unchanged(op, r, [3.0, 4.0], failure_type(op, r, [3.0, 4.0]))
      end
    end
    r = Regularize(SqrNormL2(), 1.0, ones(2, 1))
    for op in (prox!, gradient!)
      inner = op === prox! ? value_and_prox! : value_and_gradient!
      unchanged(op, SliceSum(SqrNormL2(2.0), r), ones(2, 2), failure_type(inner, r, ones(2)))
    end
  end

  @testset "Output sizes" begin
    f = SeparableSum(SqrNormL2(2.0), SqrNormL2())
    x = ([3.0], [4.0, 5.0])
    for op in oracles
      y = ([7.0], [8.0])
      before = deepcopy(y)
      @test_throws DimensionMismatch oracle!(op, y, f, x)
      @test y == before
    end
  end
end

function same_as_blocks(op, f, point)
  @nospecialize op f point
  f1 = SqrNormL2(2.0)
  for alias in (false, true)
    for nested in (false, true), scaled in (false, true)
      x = ([3.0], copy(point))
      expected_x = deepcopy(x)
      expected = alias ? expected_x : deepcopy(x)
      a = oracle!(op, expected[1], scaled ? Scaled(f1, 2.0) : f1, expected_x[1])
      b = oracle!(op, expected[2], scaled ? Scaled(f, 2.0) : f, expected_x[2])
      s = nested ? SeparableSum(f1, SeparableSum(f)) : SeparableSum(f1, f)
      scaled && (s = Scaled(s, 2.0))
      nested && (x = (x[1], (x[2],)))
      y = alias ? x : deepcopy(x)
      result = oracle!(op, y, s, x)
      @test y[1] ≈ expected[1]
      @test (nested ? y[2][1] : y[2]) ≈ expected[2]
      if op === prox! || op === gradient!
        @test result === y
      else
        @test result ≈ a + b
      end
    end
  end
end

function same_as_views(op, f, point, slices)
  @nospecialize op f point
  f1 = SqrNormL2(2.0)
  for alias in (false, true)
    x = slices ? cat(point, point; dims = ndims(point) + 1) : reshape(repeat(point, 4), 2, 2, length(point))
    expected_x = copy(x)
    expected = alias ? expected_x : copy(x)
    v = 0.0
    if slices
      inner = op === prox! ? value_and_prox! : op === gradient! ? value_and_gradient! : op
      a = oracle!(inner, PB._slice(expected, 1), f1, PB._slice(expected_x, 1))
      b = oracle!(inner, PB._slice(expected, 2), f, PB._slice(expected_x, 2))
      v = a + b
    else
      for I in CartesianIndices((2, 2))
        a = oracle!(op, PB._fiber(expected, I), f, PB._fiber(expected_x, I))
        op === value_and_prox! && (v += a)
      end
    end
    y = alias ? x : copy(x)
    result = oracle!(op, y, slices ? SliceSum(f1, f) : FiberSum(f), x)
    @test y ≈ expected
    @test op === prox! || op === gradient! ? result === y : result ≈ v
  end
end

@testset "Accepted block arguments" begin
  v = [3.0, 4.0]
  cases = Any[
    (NormL1(1.0, ones(1, 2)), v, proximal),
    (IndBox(-ones(1, 2), ones(1, 2)), v, proximal),
    (LinearFunction(ones(1, 2)), v, oracles),
    (HingeLoss(ones(1, 2)), v, proximal),
    (KLDivergence(ones(1, 2)), v, proximal),
    (IndPattern(trues(4)), [2.0 1.0; 1.0 3.0], proximal),
    (Regularize(SqrNormL2(), 1.0, ones(2, 1)), v, (prox!, gradient!)),
    (Tilt(SqrNormL2(), ones(2, 1)), v, oracles),
    (Constrained(SqrNormL2(1.0, ones(1, 2)), IndBox(-10.0, 10.0)), v, (prox!,)),
    (LinearFunction(ones(1, 4)), [2.0 1.0; 1.0 3.0], oracles),
    (LinearFunction(ones(1, 4)), SparseIterate([2.0 1.0; 1.0 3.0]), oracles),
    (Translate(LinearFunction(ones(1, 4)), ones(2, 2)), SparseIterate([2.0 1.0; 1.0 3.0]), oracles),
    (Precompose(SqrNormL2(1.0, ones(1)), ones(1, 2), 2.0), v, oracles),
    (IndHyperplane(ones(2, 1), 1.0), v, proximal),
    (IndHalfspace(ones(2, 1), 1.0), v, proximal),
  ]
  for (f, x, ops) in cases, op in ops
    same_as_blocks(op, f, x)
    inner = op === prox! ? value_and_prox! : op === gradient! ? value_and_gradient! : op
    x isa Array && inner in ops && same_as_views(op, f, x, true)
    x isa Vector && op in proximal && same_as_views(op, f, x, false)
  end

  # An unused kind of fiber imposes no checks on its function.
  f = FiberSum(SqrNormL2(1.0, ones(1, 2)); diagonal = SqrNormL2(2.0))
  x = reshape([3.0, 4.0], 1, 1, 2)
  @test vec(prox(f, x)) ≈ prox(f.diagonal, vec(x))
  @test isempty(prox(f, zeros(0, 0, 2)))
end

@testset "Block checks allocate nothing" begin
  flat = SeparableSum(SqrNormL2(2.0, ones(2)), SqrNormL2(3.0))
  x = ([3.0, 4.0], [5.0])
  nested = SeparableSum(flat, SqrNormL2())
  wrapped = SeparableSum(Translate(SqrNormL2(), ones(2)), Tilt(SqrNormL2(), ones(2, 1)),
                         Regularize(SqrNormL2(), 1.0, ones(2)))
  for (f, point, isnested) in ((flat, x, false), (Scaled(flat, 2.0), x, false),
                              (nested, (x, [6.0]), true), (wrapped, (x[1], x[1], x[1]), false))
    # Nested SeparableSum already allocates on Julia 1.10.
    isnested && VERSION < v"1.11" && continue
    y = deepcopy(point)
    for op in oracles
      if op in proximal
        @test PB.Testing.allocations(op, y, f, point, 1.0) == 0
      else
        @test PB.Testing.allocations(op, y, f, point) == 0
      end
    end
  end
  x = [3.0 4.0; 5.0 6.0]
  for s in (SliceSum(SqrNormL2(2.0), SqrNormL2(3.0)), SliceSum([SqrNormL2(2.0), SqrNormL2(3.0)]))
    @test PB.Testing.allocations(prox!, similar(x), s, x, 1.0) == 0
  end
end

end
