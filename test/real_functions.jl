using Test, ProximalBase, LinearAlgebra

const REAL_FUNCTION_TYPES = [
  ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction, KronQuadratic,
  LeastSquaresLoss, GLMLoss, LogisticLoss, PoissonLoss, HuberLoss, LogDetLoss, BernoulliLoss,
  NormL1, NormL1Nonneg, ElasticNet, NormL2, CubeNormL2, NormLinf,
  GroupNormL2, GroupNormLinf, SparseGroupNormL2, NormL1Fused, NuclearNorm,
  TracePSD, NormL0, MCP, SCAD, TotalVariation1D, FusedLasso, SLOPE, SumLargest,
  NegEntropy, KLDivergence, NegLog, Berhu, CheckLoss, HingeLoss, SqrHingeLoss,
  EpsInsensitiveLoss, CappedL1, LogPenalty, NormLq, GroupMCP, GroupSCAD,
  SingularValue, IndBox, IndNonnegative, IndBinary, IndBallL0, IndBallL1,
  IndBallL2, IndSimplex, IndIntersection, IndHyperplane, IndHalfspace,
  IndHyperslab, IndPSD, IndBallRank, IndStiefel, IndIsotonic, IndSOC,
  IndRotatedSOC, IndExpPrimal, IndExpDual, IndSphereL2, IndBallGroupL0,
  IndPattern, IndBanded, IndToeplitz, IndAffine, IndPolyhedron, IndGraph,
  SliceSum, FiberSum, Spectral, SqrDistL2, DistL2,
]

struct RealCatalogSeparable <: SeparableFunction end
ProximalBase.value_entry(::RealCatalogSeparable, v::Real, ::Integer) = abs2(v)
ProximalBase.prox_entry(::RealCatalogSeparable, v::Real, ::Integer, γ::Real) = v / (1 + 2γ)

_real_complex(x::AbstractArray, imag = 1) = complex.(x, imag)
_real_complex(x::Tuple, imag = 1) = map(xk -> _real_complex(xk, imag), x)
_real_fill(x::AbstractArray, T) = fill(T(17), size(x))
_real_fill(x::Tuple, T) = map(xk -> _real_fill(xk, T), x)
_real_copy(x::AbstractArray) = copy(x)
_real_copy(x::Tuple) = map(_real_copy, x)

function _real_reject(call, name)
  Base.@nospecialize call
  err = try
    call()
    nothing
  catch e
    e
  end
  @test err isa ArgumentError
  @test err isa ArgumentError && err.msg == "$name requires real entries"
end

function _real_write_rejections(op, f, x, args...)
  Base.@nospecialize op f x args
  name = nameof(typeof(f))
  for imaginary in (0, 1), (complex_input, complex_output) in ((true, false), (false, true), (true, true))
    input = complex_input ? _real_complex(x, imaginary) : _real_copy(x)
    output = _real_fill(x, complex_output ? ComplexF64 : Float64)
    before, input_before = _real_copy(output), _real_copy(input)
    _real_reject(() -> op(output, f, input, args...), name)
    @test isequal(output, before)
    @test isequal(input, input_before)
  end
  aliased = _real_complex(x)
  before = _real_copy(aliased)
  _real_reject(() -> op(aliased, f, aliased, args...), name)
  @test isequal(aliased, before)
end

function _real_function_cases()
  x = [1.0, 2.0, 3.0]
  M = [2.0 0.1; 0.1 1.0]
  groups = [1:2, 3:3]
  fs = [ZeroFunction(), LinearFunction(ones(3)), SqrNormL2(),
        QuadraticFunction(Matrix{Float64}(I, 3, 3)),
        LeastSquaresLoss(zeros(3), Matrix{Float64}(I, 3, 3)),
        LogisticLoss(zeros(3), Matrix{Float64}(I, 3, 3)),
        PoissonLoss(ones(3), Matrix{Float64}(I, 3, 3)),
        HuberLoss(ones(3), Matrix{Float64}(I, 3, 3), 1.0), BernoulliLoss(0.5), NormL1(), NormL1Nonneg(), ElasticNet(1, 1),
        NormL2(), CubeNormL2(), NormLinf(), GroupNormL2(1, groups),
        GroupNormLinf(1, groups), SparseGroupNormL2(1, 1, groups),
        NormL1Fused(1, 1), NormL0(), MCP(1, 3), SCAD(1, 3.7),
        TotalVariation1D(), FusedLasso(1, 1), SLOPE([3.0, 2, 1]), SumLargest(1),
        NegEntropy(), KLDivergence(1), NegLog(), Berhu(1, 1), CheckLoss(0.5),
        HingeLoss(), SqrHingeLoss(), EpsInsensitiveLoss(0.1), CappedL1(1, 1),
        LogPenalty(1, 1), NormLq(1, 0.5), GroupMCP(1, groups), GroupSCAD(1, groups),
        IndBox(-1, 1), IndNonnegative(), IndBinary(0, 1), IndBallL0(1), IndBallL1(),
        IndBallL2(), IndSimplex(), IndIntersection(IndBox(-1, 1), IndBallL2()),
        IndHyperplane(ones(3), 1), IndHalfspace(ones(3), 1),
        IndHyperslab(ones(3), 0, 1), IndIsotonic(), IndSOC(), IndRotatedSOC(),
        IndExpPrimal(), IndExpDual(), IndSphereL2(), IndBallGroupL0(1, groups),
        IndPattern(trues(3)), IndAffine([1.0 0 0], [0.0]),
        IndPolyhedron(Matrix{Float64}(I, 3, 3), -1, 1), RealCatalogSeparable(),
        SqrDistL2(IndBallL2()), DistL2(IndBallL2())]
  cases = Tuple{AbstractFunction, Any}[(f, copy(x)) for f in fs]
  append!(cases, [(f, copy(M)) for f in
    (KronQuadratic(M, M), LogDetLoss(M), NuclearNorm(), TracePSD(),
     SingularValue(NormL1()), Spectral(NormL1()), IndPSD(), IndBallRank(1),
     IndStiefel(), IndBanded(1, 1, 2, 2), IndToeplitz(),
     SliceSum(NormL1(), NormL2()), SliceSum(SqrNormL2(), SqrNormL2()))])
  push!(cases, (FiberSum(NormL2()), ones(2, 2, 2)))
  push!(cases, (IndGraph(Matrix{Float64}(I, 2, 2)), (ones(2), ones(2))))
  push!(cases, (NormL1Fused(1, 1), (ones(2), ones(2))))
  # Zero levels and empty domains still check storage.
  append!(cases, [(NormL2(0), Float64[]), (ZeroFunction(), Float64[]),
                  (NormL1(0), Float64[]), (GroupNormL2(0, Vector{Int}[]), Float64[]),
                  (FiberSum(NormL2(0)), zeros(0, 0, 0))])
  cases
end

@testset "real function oracles" begin
  cases = _real_function_cases()
  @test all(T -> any(c -> c[1] isa T, cases), REAL_FUNCTION_TYPES)
  for (f, x) in cases
    @testset "$(nameof(typeof(f))) $(size(x isa Tuple ? first(x) : x))" begin
      for imaginary in (0, 1)
        z = _real_complex(x, imaginary)
        _real_reject(() -> f(z), nameof(typeof(f)))
        isproximable(f) && _real_reject(() -> prox(f, z), nameof(typeof(f)))
        if issmooth(f)
          _real_reject(() -> gradient(f, z), nameof(typeof(f)))
          _real_reject(() -> value_and_gradient(f, z), nameof(typeof(f)))
        end
      end
      if isproximable(f)
        for op in (prox!, value_and_prox!)
          _real_write_rejections(op, f, x, 0.5)
          if x isa AbstractArray && (f isa SeparableFunction || f isa ZeroFunction ||
              f isa Union{IndBallL1, IndSimplex, IndHyperplane, IndHalfspace, IndHyperslab, IndIsotonic})
            _real_write_rejections(op, f, x, fill(0.5, size(x)))
            y = _real_fill(x, Float64)
            before = copy(y)
            _real_reject(() -> op(y, f, x, _real_complex(fill(0.5, size(x)), 0)), nameof(typeof(f)))
            @test y == before
          elseif f isa ProximalBase.AbstractGroupPenalty
            steps = fill(0.5, length(f.groups))
            _real_write_rejections(op, f, x, steps)
            y = _real_fill(x, Float64)
            _real_reject(() -> op(y, f, x, complex.(steps)), nameof(typeof(f)))
            @test y == _real_fill(x, Float64)
          end
        end
      end
      if issmooth(f)
        _real_write_rejections(gradient!, f, x)
        _real_write_rejections(value_and_gradient!, f, x)
      end
      if hashessian(f)
        for (point, direction, T) in ((_real_complex(x), x, Float64), (x, _real_complex(x), Float64), (x, x, ComplexF64))
          out = _real_fill(x, T)
          before = _real_copy(out)
          _real_reject(() -> hessian_vec!(out, f, point, direction), nameof(typeof(f)))
          @test isequal(out, before)
        end
        _real_reject(() -> quadratic_model(f, _real_complex(x)), nameof(typeof(f)))
        q = quadratic_model(f, x)
        _real_reject(() -> quadratic_model!(q, f, _real_complex(x)), nameof(typeof(f)))
      end
    end
  end
end

@testset "real parameter arrays and utilities" begin
  z = complex.(ones(3))
  Z = complex.(Matrix{Float64}(I, 3, 3))
  constructors = [
    (:LinearFunction, () -> LinearFunction(z)),
    (:SqrNormL2, () -> SqrNormL2(1, z)),
    (:QuadraticFunction, () -> QuadraticFunction(Z)),
    (:QuadraticFunction, () -> QuadraticFunction(real.(Z), z)),
    (:KronQuadratic, () -> KronQuadratic(Z, real.(Z))),
    (:KronQuadratic, () -> KronQuadratic(real.(Z), Z)),
    (:KronQuadratic, () -> KronQuadratic(real.(Z), real.(Z), Z)),
    (:LeastSquaresLoss, () -> LeastSquaresLoss(z, real.(Z))),
    (:LeastSquaresLoss, () -> LeastSquaresLoss(ones(3), Z)),
    (:GLMLoss, () -> LogisticLoss(z, real.(Z))),
    (:GLMLoss, () -> LogisticLoss(ones(3), Z)),
    (:GLMLoss, () -> LogisticLoss(ones(3), real.(Z); weights = z)),
    (:GLMLoss, () -> LogisticLoss(ones(3), real.(Z); offset = z)),
    (:LogDetLoss, () -> LogDetLoss(Z)),
    (:IndBox, () -> IndBox(z, ones(3))), (:IndBinary, () -> IndBinary(z, ones(3))),
    (:IndBox, () -> IndBox(ones(3), z)), (:IndBinary, () -> IndBinary(zeros(3), z)),
    (:IndHyperplane, () -> IndHyperplane(z, 1)),
    (:IndHalfspace, () -> IndHalfspace(z, 1)), (:IndHyperslab, () -> IndHyperslab(z, 0, 1)),
    (:IndAffine, () -> IndAffine(Z, ones(3))),
    (:IndAffine, () -> IndAffine(real.(Z), z)),
    (:IndPolyhedron, () -> IndPolyhedron(Z, -1, 1)),
    (:IndPolyhedron, () -> IndPolyhedron(real.(Z), z, 1)),
    (:IndPolyhedron, () -> IndPolyhedron(real.(Z), -1, z)),
    (:IndPolyhedron, () -> IndPolyhedron(real.(Z), -1, 1, z)),
    (:IndPolyhedron, () -> IndPolyhedron(real.(Z), -1, 1, -1, z)),
    (:IndGraph, () -> IndGraph(Z)), (:IndPattern, () -> IndPattern(Z)),
    (:SLOPE, () -> SLOPE(z)), (:NormL1, () -> NormL1(1, z)),
    (:BernoulliLoss, () -> BernoulliLoss(z)),
    (:BernoulliLoss, () -> BernoulliLoss(0.5, z)), (:KLDivergence, () -> KLDivergence(z))]
  for (name, build) in ((:NormL1Nonneg, w -> NormL1Nonneg(1, w)),
      (:ElasticNet, w -> ElasticNet(1, 1, w)), (:CheckLoss, w -> CheckLoss(0.5, w)),
      (:HingeLoss, w -> HingeLoss(w)), (:SqrHingeLoss, w -> SqrHingeLoss(w)),
      (:EpsInsensitiveLoss, w -> EpsInsensitiveLoss(1, w)),
      (:CappedL1, w -> CappedL1(1, 1, w)), (:LogPenalty, w -> LogPenalty(1, 1, w)),
      (:NormLq, w -> NormLq(1, 0.5, w)), (:Berhu, w -> Berhu(1, 1, w)),
      (:GroupNormL2, w -> GroupNormL2(1, [1:1, 2:2, 3:3], w)),
      (:GroupNormLinf, w -> GroupNormLinf(1, [1:1, 2:2, 3:3], w)),
      (:SparseGroupNormL2, w -> SparseGroupNormL2(1, 1, [1:1, 2:2, 3:3], w)),
      (:GroupMCP, w -> GroupMCP(1, [1:1, 2:2, 3:3], w)),
      (:GroupSCAD, w -> GroupSCAD(1, [1:1, 2:2, 3:3], w)))
    _real_reject(() -> build(z), name)
  end
  for (name, call) in constructors
    _real_reject(call, name)
  end
  for (a, b) in ((z, ones(3)), (ones(3), z), (z, z))
    _real_reject(() -> norm_diff(a, b), :norm_diff)
  end
  for (w, x) in ((z, ones(3)), (ones(3), z), (z, z))
    before = copy(w)
    _real_reject(() -> lla_weights!(w, MCP(1, 3), x), :lla_weights!)
    @test w == before
  end
  _real_reject(() -> coordinate_lipschitz!(z, SqrNormL2()), :coordinate_lipschitz!)
  _real_reject(() -> kkt_violation(NormL1(), z, 1), :kkt_violation)
  _real_reject(() -> kkt_violation(NormL1(), ones(3), z, 1), :kkt_violation)
end

@testset "second order late blocks" begin
  f = SeparableSum(LogisticLoss([0.0], ones(1, 1)), SqrNormL2())
  x = ([1.0], [2.0])
  q = quadratic_model(f, x)
  model_before = (copy(q.fs[1].y), copy(q.fs[1].weights))
  bad = (x[1], complex.(x[2]))
  for (point, direction, output) in ((bad, x, _real_fill(x, Float64)),
      (x, bad, _real_fill(x, Float64)), (x, x, (fill(17.0, 1), fill(17.0 + 0im, 1))))
    before = _real_copy(output)
    _real_reject(() -> hessian_vec!(output, f, point, direction), :SeparableSum)
    @test isequal(output, before)
  end
  _real_reject(() -> quadratic_model(f, bad), :SeparableSum)
  _real_reject(() -> quadratic_model!(q, f, bad), :SeparableSum)
  @test (q.fs[1].y, q.fs[1].weights) == model_before
end

@testset "second order composite entries" begin
  x = ones(2)
  for f in (Sum(SqrNormL2(), SqrNormL2()), Tilt(SqrNormL2(), ones(2)),
            Scaled(SqrNormL2(), 2), Translate(SqrNormL2(), zeros(2)))
    for imaginary in (0, 1), (point, direction, T) in
        ((_real_complex(x, imaginary), x, Float64), (x, _real_complex(x, imaginary), Float64), (x, x, ComplexF64))
      out = fill(T(17), 2)
      before = copy(out)
      _real_reject(() -> hessian_vec!(out, f, point, direction), nameof(typeof(f)))
      @test out == before
    end
    q = quadratic_model(f, x)
    _real_reject(() -> quadratic_model(f, complex.(x)), nameof(typeof(f)))
    _real_reject(() -> quadratic_model!(q, f, complex.(x)), nameof(typeof(f)))
  end
end
