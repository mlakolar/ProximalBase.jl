module HessianAliasTest

using Test, ProximalBase, LinearAlgebra, SparseArrays
using JLArrays: JLArray
import Adapt
using ..TestUtils: allocs

const PB = ProximalBase
const x_alias_error = ArgumentError("the Hessian product must not overwrite x")
const v_alias_error = ArgumentError("the Hessian product must not overwrite v")

snapshot(x) = copy(x)
snapshot(x::Tuple) = map(snapshot, x)
buffer(x) = similar(x)
buffer(x::Tuple) = map(buffer, x)
product_isapprox(x, y) = isapprox(x, y)
product_isapprox(x::Tuple, y::Tuple) = all(map(product_isapprox, x, y))

function check_alias(f, x, v)
  before_x, before_v = snapshot(x), snapshot(v)
  for (Hv, error) in ((x, x_alias_error), (v, v_alias_error))
    @test_throws error hessian_vec!(Hv, f, x, v)
    @test x == before_x
    @test v == before_v
  end
  reference = hessian_vec(f, x, v)
  Hv = buffer(reference)
  @test hessian_vec!(Hv, f, x, v) === Hv
  @test product_isapprox(Hv, reference)
  reference = hessian_vec(f, x, snapshot(x))
  @test hessian_vec!(Hv, f, x, x) === Hv
  @test product_isapprox(Hv, reference)
  @test x == before_x
  @test v == before_v
  # both identities hold; which one is reported is not part of the contract
  @test_throws ArgumentError hessian_vec!(x, f, x, x)
  @test x == before_x
end

@testset "Hessian output must be separate" begin
  x = [0.2, 0.4, 0.6, 0.7]
  v = [0.3, -0.2, 0.5, 0.1]
  c = [1.0, 0.0, 0.5, -1.0]
  Q = Matrix{Float64}(I, 4, 4) + ones(4, 4)
  A = [1.0 0.2 0.0 0.1; 0.0 1.0 0.3 0.0; 0.2 0.0 1.0 0.4; 0.1 0.0 0.0 1.0]
  y = [0.0, 1.0, 0.0, 1.0]
  P = Matrix{Float64}(I, 2, 2)
  for f in (ZeroFunction(), LinearFunction(c), SqrNormL2(1.5), SqrNormL2(1.0, c),
            QuadraticFunction(Q), LeastSquaresLoss(y, A), LogisticLoss(y, A),
            PoissonLoss(ones(4), A), HuberLoss(y, A, 0.5),
            KronQuadratic(P, P, zeros(2, 2)), Scaled(SqrNormL2(), 2.0),
            Tilt(SqrNormL2(), c), Translate(SqrNormL2(), c),
            Sum(SqrNormL2(), LogisticLoss(y, A)),
            Precompose(SqrNormL2(), A, 1.0; check = false), Precompose(SqrNormL2(), -2.0),
            Regularize(LogisticLoss(y, A), 1.0), Regularize(SqrNormL2(), 1.0, c))
    @testset "vector $(typeof(f))" begin
      check_alias(f, copy(x), copy(v))
      check_alias(f, SparseIterate(x), SparseIterate(v))
    end
  end

  X = [1.2 0.2; 0.2 1.7]
  V = [0.3 -0.2; -0.2 0.5]
  C = ones(2, 2)
  for f in (ZeroFunction(), LinearFunction(C), SqrNormL2(1.5), SqrNormL2(1.0, C),
            LeastSquaresLoss(C, P), KronQuadratic(P, P, C), LogDetLoss(P),
            Scaled(SqrNormL2(), 2.0), Tilt(SqrNormL2(), C), Translate(SqrNormL2(), C),
            Sum(SqrNormL2(), LogDetLoss(P)),
            Precompose(SqrNormL2(), -2.0),
            Regularize(SqrNormL2(), 1.0), Regularize(SqrNormL2(), 1.0, C))
    @testset "matrix $(typeof(f))" begin
      check_alias(f, copy(X), copy(V))
      # ZeroFunction and LeastSquaresLoss have no products at an owner
      f isa Union{ZeroFunction, LeastSquaresLoss} ||
        check_alias(f, SymmetricSparseIterate(X), SymmetricSparseIterate(V))
    end
  end

  x32, v32 = Float32.(x), Float32.(v)
  A32, y32 = Float32.(A), Float32.(y)
  for f in (QuadraticFunction(Float32.(Q)), LogisticLoss(y32, A32),
            Regularize(LogisticLoss(y32, A32), 1.0f0),
            Sum(SqrNormL2(1.0f0), LogisticLoss(y32, A32)))
    check_alias(f, copy(x32), copy(v32))
  end
end

@testset "named regressions with closed-form products" begin
  x = [0.2, 0.4, 0.6, 0.7]
  v = [0.3, -0.2, 0.5, 0.1]
  Q = Matrix{Float64}(I, 4, 4) + ones(4, 4)
  A = [1.0 0.2 0.0 0.1; 0.0 1.0 0.3 0.0; 0.2 0.0 1.0 0.4; 0.1 0.0 0.0 1.0]
  y = [0.0, 1.0, 0.0, 1.0]
  # LogisticLoss defaults to unit weights and zero offset, without averaging.
  s = 1 ./ (1 .+ exp.(-(A * x)))
  w = s .* (1 .- s)
  reference = v + A' * (w .* (A * v))
  for (f, Hv, error, expected) in (
      (QuadraticFunction(Q), v, v_alias_error, Q * v),
      (Regularize(LogisticLoss(y, A), 1.0), v, v_alias_error, reference),
      (Sum(SqrNormL2(), LogisticLoss(y, A)), x, x_alias_error, reference),
      (Sum(LogisticLoss(y, A), SqrNormL2()), v, v_alias_error, reference))
    before_x, before_v = copy(x), copy(v)
    @test_throws error hessian_vec!(Hv, f, x, v)
    @test x == before_x
    @test v == before_v
    out = similar(x)
    @test hessian_vec!(out, f, x, v) ≈ expected
  end
end

@testset "wrappers around specialized Hessians" begin
  x = [0.2, 0.4, 0.6, 0.7]
  v = [0.3, -0.2, 0.5, 0.1]
  c = ones(4)
  Q = Matrix{Float64}(I, 4, 4) + ones(4, 4)
  Q3 = Matrix{Float64}(I, 3, 3) + ones(3, 3)
  A = [1.0 0.2 0.0 0.1; 0.0 1.0 0.3 0.0; 0.2 0.0 1.0 0.4]
  y = [0.0, 1.0, 0.0]
  s = 1 ./ (1 .+ exp.(-(A * x)))
  w = s .* (1 .- s)
  for (f, H) in ((Regularize(QuadraticFunction(Q), 1.0), Q + I),
                 (Precompose(QuadraticFunction(Q3), A, 1.0; check = false), A' * Q3 * A),
                 (Translate(QuadraticFunction(Q), c), Q),
                 (Scaled(LeastSquaresLoss(y, A), 2.0), 2 .* (A' * A)),
                 (Sum(LogisticLoss(y, A), LeastSquaresLoss(y, A)), A' * Diagonal(w .+ 1) * A))
    check_alias(f, copy(x), copy(v))
    @test hessian_vec!(similar(x), f, x, v) ≈ H * v
    @test hessian_vec!(similar(x), f, x, x) ≈ H * x
  end
  S = [1.0 0.2 0.0; 0.2 1.5 0.1; 0.0 0.1 1.2]
  X = [1.5 0.1 0.0; 0.1 1.3 0.2; 0.0 0.2 1.7]
  V = [0.2 -0.1 0.0; -0.1 0.4 0.3; 0.0 0.3 0.5]
  f = Sum(LogDetLoss(S), KronQuadratic(S, X, zeros(3, 3)))
  check_alias(f, copy(X), copy(V))
  W = inv(X)
  @test hessian_vec!(similar(X), f, X, V) ≈ W * V * W + S * V * X
  @test hessian_vec!(similar(X), f, X, X) ≈ W * X * W + S * X * X
end

@testset "tuple blocks checked before any writes" begin
  n = 4
  Q = Matrix{Float64}(I, n, n) + ones(n, n)
  A = Matrix{Float64}(I, n, n)
  y = [0.0, 1.0, 0.0, 1.0]
  f = SeparableSum(QuadraticFunction(Q), Regularize(LogisticLoss(y, A), 1.0))
  x = ([0.2, 0.4, 0.6, 0.7], [0.1, 0.3, 0.5, 0.8])
  v = ([0.3, -0.2, 0.5, 0.1], [0.2, 0.4, -0.1, 0.3])
  nested = SeparableSum(SqrNormL2(), f)
  for (g, point, direction) in ((f, x, v), (Scaled(f, 2.0), x, v),
                                (nested, (ones(n), x), (fill(0.5, n), v)))
    for (input, error) in ((point, x_alias_error), (direction, v_alias_error))
      Hv = g === nested ? (fill(-7.0, n), (fill(-7.0, n), input[2][2])) :
                          (fill(-7.0, n), input[2])
      before_Hv, before_x, before_v = snapshot(Hv), snapshot(point), snapshot(direction)
      @test_throws error hessian_vec!(Hv, g, point, direction)
      @test Hv == before_Hv
      @test point == before_x
      @test direction == before_v
    end
    check_alias(g, point, direction)
  end
end

# Deliberately unchecked: wrappers must reject aliases before calling the child.
struct CountingHessian <: AbstractFunction
  calls::Base.RefValue{Int}
end
PB.issmooth(::Type{CountingHessian}) = true
PB.hashessian(::Type{CountingHessian}) = true
(::CountingHessian)(x) = sum(abs2, x)
function PB.hessian_vec!(Hv, f::CountingHessian, x, v)
  f.calls[] += 1
  Hv .= 2 .* v
end

@testset "wrappers check at their own entry" begin
  x = [0.2, 0.4, 0.6, 0.7]
  v = [0.3, -0.2, 0.5, 0.1]
  A = [1.0 0.2 0.0 0.1; 0.0 1.0 0.3 0.0; 0.2 0.0 1.0 0.4]
  calls = Ref(0)
  child = CountingHessian(calls)
  for (f, expected) in ((Scaled(child, 2.0), 4 .* v), (Tilt(child, ones(4)), 2 .* v),
                         (Translate(child, ones(4)), 2 .* v),
                         (Sum(SqrNormL2(), child), 3 .* v),
                         (Precompose(child, A, 1.0; check = false), 2 .* (A' * (A * v))),
                         (Regularize(child, 1.0), 3 .* v))
    calls[] = 0
    for (Hv, error) in ((x, x_alias_error), (v, v_alias_error))
      before_Hv, before_x, before_v = copy(Hv), copy(x), copy(v)
      @test_throws error hessian_vec!(Hv, f, x, v)
      @test calls[] == 0
      @test Hv == before_Hv
      @test x == before_x
      @test v == before_v
    end
    @test hessian_vec!(similar(x), f, x, v) ≈ expected
    @test calls[] > 0
  end
  f = SeparableSum(SqrNormL2(), child)
  point, direction = (copy(x), copy(x)), (copy(v), copy(v))
  calls[] = 0
  for (input, error) in ((point, x_alias_error), (direction, v_alias_error))
    Hv = (fill(-7.0, 4), input[2])
    before_Hv, before_x, before_v = snapshot(Hv), snapshot(point), snapshot(direction)
    @test_throws error hessian_vec!(Hv, f, point, direction)
    @test calls[] == 0
    @test Hv == before_Hv
    @test point == before_x
    @test direction == before_v
  end
  Hv = (similar(x), similar(x))
  @test hessian_vec!(Hv, f, point, direction) === Hv
  @test product_isapprox(Hv, (v, 2 .* v))
  @test calls[] > 0
end

@testset "empty blocks stay valid" begin
  H1 = fill(-7.0, 1)
  @test hessian_vec!((H1, ()), SeparableSum(SqrNormL2(), SeparableSum(())),
                     ([1.0], ()), ([2.0], ())) == ([2.0], ())
end

@testset "device least-squares Hessian output" begin
  A = [1.0 0.2; 0.3 1.0]
  y = [0.0, 1.0]
  host_x, host_v = [0.2, 0.4], [0.3, -0.1]
  host_f = LeastSquaresLoss(y, A)
  reference = hessian_vec(host_f, host_x, host_v)
  f = Adapt.adapt(JLArray, host_f)
  x, v = JLArray(host_x), JLArray(host_v)
  for (Hv, error) in ((v, v_alias_error), (x, x_alias_error))
    @test_throws error hessian_vec!(Hv, f, x, v)
    @test Array(x) == host_x
    @test Array(v) == host_v
  end
  Hv = similar(x)
  @test hessian_vec!(Hv, f, x, v) === Hv
  @test Array(Hv) ≈ reference
  @test Array(x) == host_x
  @test Array(v) == host_v
end

@testset "functions without a Hessian" begin
  x, v = ones(4), fill(0.5, 4)
  @test_throws ArgumentError hessian_vec!(x, NormL1(), x, v)
  @test_throws ArgumentError hessian_vec!(x, Precompose(SqrNormL2(), [1.0, 2.0, 3.0, 4.0]), x, v)
end

a_hv!(Hv, f, x, v) = hessian_vec!(Hv, f, x, v)

@testset "zero allocations and inference" begin
  x, v, Hv = ones(4), fill(0.5, 4), zeros(4)
  @test allocs(PB._checkhessian_output, Hv, x, v) == 0
  point, direction, out = (copy(x), (copy(x),)), (copy(v), (copy(v),)), (copy(Hv), (copy(Hv),))
  @test allocs(PB._checkhessian_output, (Hv,), (x,), (v,)) == 0
  @test allocs(PB._checkhessian_output, out, point, direction) == 0
  Q = Matrix{Float64}(I, 4, 4) + ones(4, 4)
  f = QuadraticFunction(Q)
  hessian_vec!(Hv, f, x, v)
  @test allocs(a_hv!, Hv, f, x, v) == 0
  @test (@inferred hessian_vec!(Hv, f, x, v)) === Hv
  ftuple = SeparableSum(SqrNormL2(), SeparableSum(QuadraticFunction(Q)))
  hessian_vec!(out, ftuple, point, direction)
  @test allocs(a_hv!, out, ftuple, point, direction) == 0
  @test (@inferred hessian_vec!(out, ftuple, point, direction)) === out
  S = Matrix{Float64}(I, 3, 3)
  X, V, HV = 2 .* S, copy(S), zeros(3, 3)
  f = LogDetLoss(S)
  hessian_vec!(HV, f, X, V)
  @test allocs(a_hv!, HV, f, X, V) == 0
  @test (@inferred hessian_vec!(HV, f, X, V)) === HV
end


# the matrix-shaped sources of the storage of an owner (the aliases of test/symmetric_iterate_copies.jl): its matrix
# view, and views, transposes and Symmetric and Hermitian wrappers of it, in both triangles
function view_aliases(A)
  out = Any["plain" => A, "view" => view(A, :, :), "nested view" => view(view(A, :, :), :, :), "transpose" => transpose(A),
            "adjoint" => A', "view of 1:3" => view(A, 1:3, 1:3)]
  for tri in (:L, :U), F in (Symmetric, Hermitian)
    C = F(A, tri)
    push!(out, "$(nameof(F)) $tri" => C, "view of $(nameof(F)) $tri" => view(C, :, :))
  end
  out
end
state(y) = (Matrix(y), nactive(y), copy(activecoordinates(y)), copy(y.nzval), copy(y.nzval2ind), copy(y.ind2nzval))

@testset "an owner output aliased through its matrix view is refused" begin
  C = [1.0 0.5 -1.0; 0.5 2.0 0.25; -1.0 0.25 1.5]
  X = [0.5 0.0 0.3; 0.0 0.6 -1.1; 0.3 -1.1 0.0]
  V = [0.3 -0.2 0.1; -0.2 0.5 0.4; 0.1 0.4 -0.6]
  fs = ("SqrNormL2" => SqrNormL2(1.5), "Sum" => Sum(SqrNormL2(1.5), LinearFunction(C)), "Scaled" => Scaled(SqrNormL2(1.5), 2.0),
        "Regularize" => Regularize(SqrNormL2(1.5), 1.0), "Tilt" => Tilt(SqrNormL2(1.5), C, 0.2),
        "KronQuadratic" => KronQuadratic(Matrix(1.0I, 3, 3), Matrix(1.0I, 3, 3), C))
  for (fname, f) in fs, (name, _) in view_aliases(matrixview(SymmetricSparseIterate(X)))
    # the point, or the direction, is a view of the output
    y = SymmetricSparseIterate(X)
    a = Dict(view_aliases(matrixview(y)))[name]
    before = state(y)
    @test_throws x_alias_error hessian_vec!(y, f, a, V)
    @test isequal(state(y), before)
    @test_throws v_alias_error hessian_vec!(y, f, V, a)
    @test isequal(state(y), before)
    # an owner as the point or the direction, which is the output itself
    @test_throws ArgumentError hessian_vec!(y, f, y, V)
    @test_throws ArgumentError hessian_vec!(y, f, V, y)
    @test isequal(state(y), before)
    # a view of another owner with the same values is not an alias
    z = SymmetricSparseIterate(X)
    other = Dict(view_aliases(matrixview(z)))[name]
    @test hessian_vec!(y, f, other, V) === y
  end
end

end
