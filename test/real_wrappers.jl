using Test, ProximalBase, LinearAlgebra

const REAL_WRAPPER_TYPES = [Scaled, Translate, Tilt, Conjugate, Sum, SeparableSum,
  Precompose, ProximalBase._ArrayPrecompose, Regularize, LinearImage, MoreauEnvelope,
  PointwiseMinimum, ProximalBase.BoxConstrained, ProximalBase.LinearConstrained,
  ProximalBase.BallConstrained, ProximalBase.SphereConstrained, ProximalBase.AffineConstrained]

_rwcomplex(x::AbstractArray) = complex.(x)
_rwcomplex(x::Tuple) = map(_rwcomplex, x)
_rwimag(x::AbstractArray) = x .+ im
_rwimag(x::Tuple) = map(_rwimag, x)
_rwcopy(x::AbstractArray) = copy(x)
_rwcopy(x::Tuple) = map(_rwcopy, x)
_rwfill(x::AbstractArray) = fill!(similar(x), 73)
_rwfill(x::Tuple) = map(_rwfill, x)

function _rwreject(call, f, arrays...)
  before = map(_rwcopy, arrays)
  @test_throws ArgumentError("$(nameof(typeof(f))) requires real entries") call()
  @test all(map(isequal, arrays, before))
end

function _rwfirstorder(f, x; smooth = issmooth(f), proximal = isproximable(f), steps = (0.5,))
  for z in (_rwcomplex(x), _rwimag(x))
    _rwreject(() -> f(z), f, z)
    if proximal
      for step in steps
        _rwreject(() -> prox(f, z, step), f, z)
        for oracle in (prox!, value_and_prox!)
          for (point, output) in ((z, _rwfill(x)), (x, _rwfill(z)), (z, _rwfill(z)), (z, z))
            _rwreject(() -> oracle(output, f, point, step), f, output, point)
          end
        end
      end
    end
    if smooth
      _rwreject(() -> gradient(f, z), f, z)
      _rwreject(() -> value_and_gradient(f, z), f, z)
      for oracle in (gradient!, value_and_gradient!)
        for (point, output) in ((z, _rwfill(x)), (x, _rwfill(z)), (z, _rwfill(z)), (z, z))
          _rwreject(() -> oracle(output, f, point), f, output, point)
        end
      end
    end
  end
end

@testset "real wrapper boundaries" begin
  x = [1.0, -2.0]
  I2 = Matrix{Float64}(I, 2, 2)
  cases = (Scaled(SqrNormL2(), 2), Translate(SqrNormL2(), zeros(2)),
    Tilt(SqrNormL2(), ones(2)), Conjugate(NormL1()),
    Conjugate(Conjugate(NormL1())), Conjugate(SumLargest(1)),
    Sum(SqrNormL2(), LinearFunction(ones(2))),
    Precompose(SqrNormL2(), 2.0), Precompose(SqrNormL2(), I2, 1.0),
    Precompose(SqrNormL2(), [1.0, 2.0]), Regularize(SqrNormL2(), 0.5, zeros(2)),
    LinearImage(NormL1(), I2, 1.0), LinearImage(QuadraticFunction(I2), I2),
    MoreauEnvelope(NormL1(), 0.5), PointwiseMinimum(NormL1(), NormL2()),
    Constrained(NormL1(), IndBox(-1.0, 1.0)),
    Constrained(NormL1(), IndHyperplane(ones(2), 0.0)),
    Constrained(MCP(1), IndBox(-1, 1)),
    Constrained(NormL1(), IndBallL2()), Constrained(NormL1(), IndSphereL2()),
    Constrained(GroupNormL2(1, [[1], [2]]), IndSphereL2()),
    Constrained(IndBallL0(1), IndSphereL2()),
    Constrained(QuadraticFunction(I2), IndAffine(reshape(ones(2), 1, 2), [0.0])))
  for f in cases
    @testset "$(nameof(typeof(f)))" begin
      _rwfirstorder(f, x)
    end
  end
  # Every block is checked before any earlier block is written.
  for f in (SeparableSum(SqrNormL2(), SqrNormL2()),
            Scaled(SeparableSum(SqrNormL2(), SqrNormL2()), 2),
            Conjugate(SeparableSum(NormL1(), NormL1())))
    point = (copy(x), complex.(x))
    output = (fill(73.0, 2), fill(73.0, 2))
    _rwreject(() -> f(point), f, point)
    _rwreject(() -> prox(f, point, 0.5), f, point)
    for oracle in (prox!, value_and_prox!)
      _rwreject(() -> oracle(output, f, point, 0.5), f, output, point)
      _rwreject(() -> oracle(point, f, point, 0.5), f, point)
      badout = (fill(73.0, 2), fill(73.0 + 0im, 2))
      _rwreject(() -> oracle(badout, f, (x, x), 0.5), f, badout)
    end
    if issmooth(f)
      _rwreject(() -> gradient(f, point), f, point)
      _rwreject(() -> value_and_gradient(f, point), f, point)
      for oracle in (gradient!, value_and_gradient!)
        _rwreject(() -> oracle(output, f, point), f, output, point)
        badout = (fill(73.0, 2), fill(73.0 + 0im, 2))
        _rwreject(() -> oracle(badout, f, (x, x)), f, badout)
      end
    end
    _rwfirstorder(f, (x, x))
  end
  # Zero levels and empty inputs cannot bypass the boundary.
  for f in (Scaled(NormL1(), 0), Regularize(SqrNormL2(), 0), Sum(), SeparableSum())
    point = f isa SeparableSum ? (ComplexF64[],) : ComplexF64[]
    _rwreject(() -> f(point), f, point)
  end
end

@testset "real wrapper array steps" begin
  x = [1.0, -2.0]
  for f in (Scaled(NormL1(), 2), Translate(NormL1(), zeros(2)), Tilt(NormL1(), ones(2)),
            Conjugate(NormL1()), Precompose(NormL1(), 2.0),
            Precompose(NormL1(), [1.0, 2.0]), Regularize(NormL1(), 0.5),
            Constrained(NormL1(), IndBox(-1.0, 1.0)),
            Constrained(NormL1(), IndHyperplane(ones(2), 0.0)))
    _rwfirstorder(f, x; steps = ([0.5, 0.5],))
    for oracle in (prox!, value_and_prox!)
      output = fill(73.0, 2)
      step = fill(0.5 + 0im, 2)
      _rwreject(() -> oracle(output, f, x, step), f, output, x, step)
    end
  end
end

@testset "real wrapper second order" begin
  x = [1.0, -2.0]
  for f in (Scaled(SqrNormL2(), 2), Tilt(SqrNormL2(), zeros(2)),
            Sum(SqrNormL2(), SqrNormL2()), Precompose(SqrNormL2(), 2.0),
            Precompose(SqrNormL2(), Matrix{Float64}(I, 2, 2), 1.0),
            Regularize(SqrNormL2(), 0.5))
    z = complex.(x)
    for (point, direction, output) in ((z, x, _rwfill(x)), (x, z, _rwfill(x)),
                                      (x, x, _rwfill(z)), (z, z, z))
      _rwreject(() -> hessian_vec!(output, f, point, direction), f, output, point, direction)
    end
    _rwreject(() -> quadratic_model(f, z), f, z)
    q = quadratic_model(f, x)
    before = q(x)
    _rwreject(() -> quadratic_model!(q, f, z), f, z)
    @test q(x) == before
  end
end

@testset "real wrapper tuple second order" begin
  x = ([1.0], [2.0])
  f = SeparableSum(SqrNormL2(), SqrNormL2())
  z = (x[1], complex.(x[2]))
  for (point, direction, output) in ((z, x, _rwfill(x)), (x, z, _rwfill(x)),
                                    (x, x, (fill(73.0, 1), fill(73.0 + 0im, 1))))
    _rwreject(() -> hessian_vec!(output, f, point, direction), f, output, point, direction)
  end
  _rwreject(() -> quadratic_model(f, z), f, z)
  q = quadratic_model(f, x)
  before = q(x)
  _rwreject(() -> quadratic_model!(q, f, z), f, z)
  @test q(x) == before
end

@testset "real wrapper group steps" begin
  x = [1.0, -2.0]
  groups = [[1], [2]]
  for f in (Scaled(GroupNormL2(1, groups), 2),
            Translate(GroupNormL2(1, groups), zeros(2)),
            Regularize(GroupNormL2(1, groups), 0.5),
            Precompose(GroupNormL2(1, groups), [1.0, 2.0]))
    _rwfirstorder(f, x; steps = ([0.5, 0.5],))
    for oracle in (prox!, value_and_prox!)
      output = fill(73.0, 2)
      step = fill(0.5 + 0im, 2)
      _rwreject(() -> oracle(output, f, x, step), f, output, x, step)
    end
  end
end

@testset "real wrapper parameters" begin
  z = ComplexF64[0, 0]
  for (name, call) in ((:Translate, () -> Translate(NormL1(), z)),
                       (:Tilt, () -> Tilt(NormL1(), z)),
                       (:Regularize, () -> Regularize(NormL1(), 0.0, z)),
                       (:Precompose, () -> Precompose(NormL1(), z)),
                       (:Precompose, () -> Precompose(NormL1(), reshape(z, 1, 2), 1.0)),
                       (:LinearImage, () -> LinearImage(NormL1(), reshape(z, 2, 1), 1.0)),
                       (:LinearImage, () -> LinearImage(NormL1(), reshape(z, 2, 1))),
                       (:LinearImage, () -> LinearImage(QuadraticFunction(ones(1, 1)), reshape(z, 2, 1))))
    @test_throws ArgumentError("$name requires real entries") call()
  end
end

@testset "real wrapper optimality utilities" begin
  x = [1.0, -2.0]
  for f in (Scaled(NormL1(), 2), Tilt(NormL1(), zeros(2)), Regularize(NormL1(), 1),
            Precompose(NormL1(), 2.0), Precompose(NormL1(), ones(2)),
            MoreauEnvelope(NormL1(), 0.5), Constrained(NormL1(), IndBox(-1, 1)),
            Constrained(MCP(1), IndBox(-1, 1)))
    @test_throws ArgumentError("kkt_violation requires real entries") kkt_violation(f, complex.(x), 1)
  end
end

@testset "real wrapper unsupported step boundaries" begin
  f = PointwiseMinimum(NormL1(), NormL2())
  x = [1.0, -2.0]
  z = complex.(x)
  for (point, step) in ((z, ones(2)), (x, complex.(ones(2))))
    _rwreject(() -> prox(f, point, step), f, point, step)
    for oracle in (prox!, value_and_prox!)
      output = fill(73.0, 2)
      _rwreject(() -> oracle(output, f, point, step), f, output, point, step)
    end
  end
end
