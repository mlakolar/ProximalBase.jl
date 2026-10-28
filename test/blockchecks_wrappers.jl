module BlockChecksWrappersTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays

const PB = ProximalBase
const proximal = (prox!, value_and_prox!)
const smooth = (gradient!, value_and_gradient!)
const oracles = (prox!, value_and_prox!, gradient!, value_and_gradient!)

function oracle!(op, y, f, x)
  @nospecialize op y f x
  op in proximal ? op(y, f, x, 1.0) : op(y, f, x)
end

function unchanged(op, f, point, E; output = point, modes = (false, true))
  @nospecialize op f point E output
  for alias in modes
    x = deepcopy(point)
    y = alias ? x : deepcopy(output)
    input, before = deepcopy(x), deepcopy(y)
    @test_throws E oracle!(op, y, f, x)
    @test isequal(x, input)
    @test isequal(y, before)
  end
end

function block_failure(f, x, errors; output = x, modes = (false, true))
  @nospecialize f x errors output
  for (op, E) in errors
    unchanged(op, f, x, E; output, modes)
    s = SeparableSum(SqrNormL2(2.0), f)
    point, dest = ([3.0], x), ([7.0], output)
    unchanged(op, s, point, E; output = dest, modes)
    unchanged(op, Scaled(s, 2.0), point, E; output = dest, modes)
    nested = SeparableSum(SqrNormL2(2.0), SeparableSum(f))
    unchanged(op, nested, ([3.0], (x,)), E; output = ([7.0], (output,)), modes)
    if x isa Array && output isa Array
      # SliceSum calls the value oracle, including from prox! and gradient!.
      inner = op === prox! ? value_and_prox! : op === gradient! ? value_and_gradient! : op
      k = findfirst(p -> first(p) === inner, errors)
      k === nothing && continue
      Eslice = last(errors[k])
      slices = cat(x, x; dims = ndims(x) + 1)
      unchanged(op, SliceSum(SqrNormL2(2.0), f), slices, Eslice; modes)
      if x isa Vector && op in proximal
        fibers = reshape(repeat(x, 4), 2, 2, length(x))
        unchanged(op, FiberSum(f; diagonal = SqrNormL2(2.0)), fibers, E; modes)
      end
    end
  end
end

errors(ops, E) = map(op -> op => E, ops)

@testset "Wrapper checks before writes" begin
  v = [3.0, 4.0]
  M = [2.0 1.0; 1.0 3.0]
  I2 = Matrix(1.0I, 2, 2)
  matrix_only = (NuclearNorm(1.0), TracePSD(1.0), IndPSD(), IndPSD(rank = 1),
                 IndBallRank(1), SingularValue(NormL1(1.0)), Spectral(NormL1(1.0)),
                 LogDetLoss(I2), IndToeplitz())
  vector_only = (TotalVariation1D(), NormL1Fused(1.0, 1.0), FusedLasso(1.0, 1.0),
                 IndIsotonic(), SLOPE([4.0, 3.0, 2.0, 1.0]), IndAffine(ones(1, 4), ones(1)),
                 QuadraticFunction(Matrix(1.0I, 4, 4)), LogisticLoss(ones(2), ones(2, 4)),
                 Constrained(QuadraticFunction(I2), IndAffine(ones(1, 2), ones(1))))
  for (fs, x) in ((matrix_only, v), (vector_only, M)), f in fs
    @testset "$(nameof(typeof(f))) argument kind" begin
      ops = (isproximable(f) ? proximal : (), issmooth(f) ? smooth : ())
      block_failure(f, x, errors((ops[1]..., ops[2]...), MethodError))
    end
  end
  block_failure(LeastSquaresLoss(ones(2), I2), ones(2, 1, 1), errors(oracles, MethodError))
  block_failure(SeparableSum(SqrNormL2()), v, errors(oracles, MethodError))
  block_failure(SLOPE([2.0, 1.0]), M, (prox! => MethodError, value_and_prox! => DimensionMismatch))
  block_failure(NormL1Fused(1.0, 1.0), (1.0, 2.0), errors(proximal, MethodError))
  block_failure(NormL1Fused(1.0, 1.0), (ones(2), ones(3)), errors(proximal, DimensionMismatch))
  block_failure(IndToeplitz(symmetric = true), ones(2, 3), errors(proximal, DimensionMismatch))
  block_failure(IndToeplitz(), M, errors(proximal, ArgumentError);
                output = SymmetricSparseIterate(M), modes = (false,))

  cases = Any[
    (Tilt(NuclearNorm(1.0), ones(2)), v, proximal, MethodError),
    (Translate(NuclearNorm(1.0), ones(2)), v, proximal, MethodError),
    (Regularize(NuclearNorm(1.0), 1.0), v, proximal, MethodError),
    (Regularize(LogDetLoss(I2), 1.0), v, oracles, MethodError),
    (Precompose(NuclearNorm(1.0), 2.0), v, proximal, MethodError),
    (Precompose(TotalVariation1D(), I2, 1.0), M, proximal, MethodError),
    (Precompose(SqrNormL2(), ones(1, 3), 3.0), M, oracles, MethodError),
    (Precompose(SqrNormL2(1.0, ones(1, 2)), 2.0), v, oracles, DimensionMismatch),
    (Precompose(SqrNormL2(1.0, ones(3)), I2, 1.0), v, oracles, DimensionMismatch),
    (Sum(SqrNormL2(), QuadraticFunction(I2)), M, smooth, MethodError),
    (Sum(SqrNormL2(), SqrNormL2(1.0, ones(1, 2))), v, smooth, DimensionMismatch),
    (Constrained(SqrNormL2(1.0, ones(1, 2)), IndBox(-10.0, 10.0)), v, (value_and_prox!,), DimensionMismatch),
    (Constrained(SqrNormL2(1.0, ones(1, 2)), IndHyperplane(1.0, 1.0)), v, (value_and_prox!,), DimensionMismatch),
    (Constrained(GroupNormL2(1.0, [1:3]), IndBallL2(1.0)), v, proximal, BoundsError),
    (Constrained(IndBallL0(0), IndSphereL2(1.0)), v, proximal, ArgumentError),
    (Constrained(NormL1(1.0), IndSphereL2(1.0)), Float64[], proximal, ArgumentError),
    (Constrained(LeastSquaresLoss(ones(2), ones(2, 3)), IndAffine(ones(1, 3), ones(1))), v, proximal, DimensionMismatch),
    (IndIntersection(IndPSD(), IndBox(-1.0, 1.0)), v, proximal, MethodError),
    (IndIntersection(IndBallL2(1.0), IndHyperplane(ones(1, 4), 1.0)), M, proximal, DimensionMismatch),
    (SingularValue(NormL1(1.0, ones(3))), M, proximal, DimensionMismatch),
    (SingularValue(SqrNormL2(1.0, ones(1, 2))), M, proximal, DimensionMismatch),
    (Spectral(Conjugate(Constrained(IndBox(-1e100, 1e100), IndHalfspace(1.0, 1.0)))), Float32.(M), proximal, ArgumentError),
    (MoreauEnvelope(NuclearNorm(1.0), 0.5), v, proximal, MethodError),
    (MoreauEnvelope(SqrNormL2(1.0, ones(1, 2)), 0.5), v, proximal, DimensionMismatch),
    (Conjugate(NuclearNorm(1.0)), v, proximal, MethodError),
    (Conjugate(TotalVariation1D()), M, proximal, MethodError),
    (Conjugate(LinearFunction(ones(1, 2))), v, (value_and_prox!,), DimensionMismatch),
    (Conjugate(Translate(NormL1(1.0), ones(1, 2))), v, proximal, DimensionMismatch),
    (Conjugate(Scaled(LinearFunction(ones(1, 2)), 2.0)), v, (value_and_prox!,), DimensionMismatch),
    (Conjugate(NormL1(1.0, ones(3))), v, proximal, DimensionMismatch),
    (Conjugate(IndBox(-ones(3), ones(3))), v, proximal, DimensionMismatch),
    (Conjugate(Constrained(IndBox(-1e100, 1e100), IndHyperplane(1.0, 1.0))), Float32[1, 2], (value_and_prox!,), ArgumentError),
    (Conjugate(SeparableSum(NormL1(1.0), LinearFunction(ones(1, 2)))), (v, v), (value_and_prox!,), DimensionMismatch),
    (Conjugate(SeparableSum(SqrNormL2(), NuclearNorm(1.0))), (v, v), proximal, MethodError),
  ]
  for (f, x, ops, E) in cases
    @testset "$(typeof(f))" begin
      block_failure(f, x, errors(ops, E))
    end
  end

  @testset "Gradient branches and aliasing" begin
    block_failure(MoreauEnvelope(NormL1(1.0), 0.5), v, errors(smooth, ArgumentError); modes = (true,))
    for f in (MoreauEnvelope(NuclearNorm(1.0), 0.5),
              MoreauEnvelope(SqrNormL2(1.0, ones(1, 2)), 0.5))
      E = f.f isa NuclearNorm ? MethodError : DimensionMismatch
      block_failure(f, v, errors(smooth, E); modes = (false,))
      block_failure(f, v, errors(smooth, ArgumentError); modes = (true,))
    end
  end

  @testset "Conjugate with entrywise steps" begin
    for f in (LinearFunction(ones(1, 2)), Constrained(IndBox(-1e100, 1e100), IndHyperplane(1.0, 1.0)))
      for alias in (false, true)
        x = f isa LinearFunction ? copy(v) : Float32[1, 2]
        y = alias ? x : copy(x)
        x0, y0 = copy(x), copy(y)
        E = f isa LinearFunction ? DimensionMismatch : ArgumentError
        @test_throws E value_and_prox!(y, Conjugate(f), x, ones(eltype(x), 2))
        @test isequal(x, x0)
        @test isequal(y, y0)
      end
    end
  end
end

function representation_cases()
  M = [2.0 1.0; 1.0 3.0]
  (M, SparseIterate(M), SymmetricSparseIterate(M), sparse(M), view(hcat(M, M), :, 2:3))
end

@testset "LinearFunction representation checks" begin
  reps = representation_cases()
  for c in (ones(1, 4), sparse(ones(1, 4))), y in reps, x in reps
    f = LinearFunction(c)
    # These domains select the shape-checking dot or copyto! methods.
    dot_y = c isa Matrix ? y isa Union{SymmetricSparseIterate, SparseMatrixCSC} : y isa Union{Matrix, SparseMatrixCSC, SubArray}
    dot_x = c isa Matrix ? x isa Union{SymmetricSparseIterate, SparseMatrixCSC} : x isa Union{Matrix, SparseMatrixCSC, SubArray}
    for op in oracles
      bad = op === value_and_prox! ? dot_y : op === value_and_gradient! ? dot_x || y isa SymmetricSparseIterate :
            op === gradient! && y isa SymmetricSparseIterate
      bad || continue
      # Cross distinct representations, and alias when their representations agree.
      block_failure(f, x, (op => DimensionMismatch,); output = y,
                    modes = typeof(x) === typeof(y) ? (false, true) : (false,))
    end
  end
end

function same_as_blocks(op, f, point; output = point, alias = false)
  @nospecialize op f point output
  x = ([3.0], deepcopy(point))
  y = alias ? x : ([7.0], deepcopy(output))
  input = deepcopy(x)
  expected = alias ? input : deepcopy(y)
  a = oracle!(op, expected[1], SqrNormL2(2.0), input[1])
  b = oracle!(op, expected[2], deepcopy(f), input[2])
  result = oracle!(op, y, SeparableSum(SqrNormL2(2.0), deepcopy(f)), x)
  @test isequal(y, expected)
  @test op in (prox!, gradient!) ? result === y : isequal(result, a + b)
end

@testset "Accepted wrapper arguments" begin
  v = [3.0, 4.0]
  M = [2.0 1.0; 1.0 3.0]
  I2 = Matrix(1.0I, 2, 2)
  cases = Any[
    (Constrained(Tilt(SqrNormL2(), ones(1, 2)), IndBox(-10.0, 10.0)), v, proximal),
    (Constrained(Translate(IndNonnegative(), ones(1, 2)), IndBox(-10.0, 10.0)), v, proximal),
    (Constrained(Constrained(SqrNormL2(1.0, ones(1, 2)), IndBox(0.0, 1.0)), IndBox(2.0, 3.0)), v, proximal),
    (Precompose(SqrNormL2(), 2.0), v, oracles),
    (Precompose(SqrNormL2(), I2, 1.0), v, oracles),
    (Sum(SqrNormL2(), SqrNormL2(2.0)), v, smooth),
    (Sum(SqrNormL2(), MoreauEnvelope(NormL1(1.0), 0.5)), v, smooth),
    (IndIntersection(IndNonnegative(), IndHyperplane(1.0, 1.0)), v, proximal),
    (SingularValue(NormL1(1.0, ones(2))), M, proximal),
    (Spectral(NormL1(1.0)), M, proximal),
    (MoreauEnvelope(NormL1(1.0), 0.5), v, oracles),
    (Precompose(MoreauEnvelope(NormL1(1.0), 0.5), 2.0), v, oracles),
    (Sum(MoreauEnvelope(NormL1(1.0), 0.5), MoreauEnvelope(NormL1(1.0), 0.5)), v, smooth),
    (NormL1Fused(1.0, 1.0), (v, reverse(v)), proximal),
  ]
  known = (NormL1(1.0, ones(2)), NormL2(), NormLinf(), IndBallL1(), IndBallL2(), ZeroFunction(),
           IndNonnegative(), IndBox(-ones(2), ones(2)), SqrNormL2(1.0, ones(2)),
           LinearFunction(ones(2)), CheckLoss(0.5, ones(2)), HingeLoss(ones(2)),
           EpsInsensitiveLoss(0.5, ones(2)), Translate(NormL1(1.0), ones(2)),
           Scaled(NormL1(1.0), 2.0), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 1.0)))
  for f in known
    push!(cases, (Conjugate(f), v, proximal))
  end
  push!(cases, (Conjugate(SingularValue(NormL1(1.0))), M, proximal))
  push!(cases, (Conjugate(SeparableSum(NormL1(1.0), LinearFunction(ones(2)))), (v, v), proximal))
  for (f, x, ops) in cases, op in ops, alias in (false, true)
    alias && op in smooth && (f isa MoreauEnvelope || f isa Sum && first(f.fs) isa MoreauEnvelope) && continue
    same_as_blocks(op, f, x; alias)
  end
  for c in (ones(4), view(ones(2, 4), [1], :)), x in representation_cases(), op in oracles
    # A matrix copied into a symmetric iterate still needs matching shapes.
    c isa SubArray && x isa SymmetricSparseIterate && op in smooth && continue
    c isa SubArray && x isa SparseMatrixCSC && op in (value_and_prox!, value_and_gradient!) && continue
    same_as_blocks(op, LinearFunction(c), x; alias = true)
  end
end

@testset "Shared cache history" begin
  D = IndIntersection(IndNonnegative(), IndHyperplane(1.0, 1.0); maxiter = 1)
  prox!(zeros(2), D, [2.0, -1.0], 1.0)
  saved = D.buffers
  PB._checkwrite(prox!, SeparableSum(D, D), (zeros(2), zeros(3)), (ones(2), ones(3)))
  @test D.buffers === saved
  out = (zeros(2), zeros(3))
  prox!(out, SeparableSum(D, D), ([2.0, -1.0], ones(3)), 1.0)
  @test isequal(out[1], [1.25, -0.25])
  s = Spectral(NormL1(1.0))
  prox(s, Matrix(2.0I, 3, 3))
  block_failure(s, Matrix(2.0I, 2, 2), errors(proximal, DimensionMismatch))
end

@testset "Wrapper checks allocate nothing" begin
  for (f, point, ops) in (
      (Precompose(SqrNormL2(), 2.0), [3.0, 4.0], oracles),
      (Sum(SqrNormL2(), SqrNormL2(2.0)), [3.0, 4.0], smooth),
      (SingularValue(NormL1(1.0)), [2.0 1.0; 1.0 3.0], proximal),
      (MoreauEnvelope(NormL1(1.0), 0.5), [3.0, 4.0], oracles),
      (Conjugate(NormL1(1.0)), [3.0, 4.0], proximal))
    s = SeparableSum(SqrNormL2(2.0), f)
    x = ([3.0], point)
    y = deepcopy(x)
    for op in ops
      args = op in proximal ? (1.0,) : ()
      @test PB.Testing.allocations(op, y, s, x, args...) == 0
    end
  end
  f = SeparableSum(SqrNormL2(2.0), SeparableSum(Precompose(SqrNormL2(), 2.0), SqrNormL2()))
  x = ([3.0], ([3.0, 4.0], [5.0]))
  y = deepcopy(x)
  # These are the nested tuple's measured counts before the wrapper checks.
  for (op, base) in zip(oracles, (704, 576, 192, 192))
    args = op in proximal ? (1.0,) : ()
    @test PB.Testing.allocations(op, y, f, x, args...) <= (VERSION < v"1.11" ? base : 0)
  end
  x = [3.0 4.0; 5.0 6.0]
  for fs in ((Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), 2.0)),
             [Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), 2.0)])
    @test PB.Testing.allocations(prox!, similar(x), SliceSum(fs), x, 1.0) == 0
  end
end

end
