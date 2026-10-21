module  DifferentiableFunctionsTests

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)

@testset "L2Loss" begin

  x = randn(5)
  y = randn(5)
  f1 = L2Loss()
  f2 = L2Loss(y)

  @test value(f1, x) ≈ sum(abs2,x)/2.
  @test value(f2, x) ≈ sum(abs2,x-y)/2.

  hat_x = similar(x)
  @test value_and_gradient!(f1, hat_x, x) ≈ sum(abs2,x)/2.
  @test hat_x ≈ x
  hat_x = similar(x)
  @test value_and_gradient!(f2, hat_x, x) ≈ sum(abs2,x-y)/2.
  @test hat_x ≈ x-y

  # gradient written over its own input
  z = copy(x)
  @test value_and_gradient!(f2, z, z) ≈ sum(abs2,x-y)/2.
  @test z ≈ x-y

  @test_throws DimensionMismatch value_and_gradient!(f2, zeros(4), x)

  # Float32 in, Float32 out
  x32, y32 = randn(Float32, 5), randn(Float32, 5)
  @test value(L2Loss(y32), x32) isa Float32
  @test value_and_gradient!(L2Loss(y32), similar(x32), x32) isa Float32
  @test value(L2Loss(y32), x32) ≈ sum(abs2, x32 - y32) / 2

end

@testset "Quadratic Function" begin

  A = randn(10, 10)
  A = A + A'
  b = randn(10)
  c = 1.

  x = randn(10)

  q1 = QuadraticFunction(A)
  q2 = QuadraticFunction(A, b)
  q3 = QuadraticFunction(A, b, c)

  @test value(q1, x) ≈ dot(x, A*x)/2.
  @test value(q2, x) ≈ dot(x, A*x)/2. + dot(x, b)
  @test value(q3, x) ≈ dot(x, A*x)/2. + dot(x, b) + c

  hat_x = similar(x)
  @test value_and_gradient!(q1, hat_x, x) ≈ dot(x, A*x)/2.
  @test hat_x ≈ A*x

  @test value_and_gradient!(q2, hat_x, x) ≈ dot(x, A*x)/2. + dot(x, b)
  @test hat_x ≈ A*x + b

  @test value_and_gradient!(q3, hat_x, x) ≈ dot(x, A*x)/2. + dot(x, b) + c
  @test hat_x ≈ A*x + b

  @test gradient!(q1, hat_x, x) ≈ A*x
  @test gradient!(q2, hat_x, x) ≈ A*x + b
  @test gradient!(q3, hat_x, x) ≈ A*x + b
end


@testset "Least Squares Loss" begin

  y = randn(10)
  X = randn(10, 5)
  b = rand(5)
  f = LeastSquaresLoss(y, X)

  @test value(f, b) ≈ norm(y-X*b)^2 / 2.
  grad_out = zeros(5)
  @test value_and_gradient!(f, grad_out, b) ≈ norm(y-X*b)^2 / 2.
  @test grad_out ≈ -X'*(y - X*b)

end

p = 30
A = randn(p, p)
A = A + A'
b, c = randn(p), 1.5
y = randn(p)
X, Y = randn(20, p), randn(20)
smooth_functions = (QuadraticFunction(A), QuadraticFunction(A, b), QuadraticFunction(A, b, c),
                    L2Loss(), L2Loss(y), LeastSquaresLoss(Y, X))

@testset "restrict" begin
  @testset "$(nameof(typeof(f)))" for f in smooth_functions
    S = [8, 3, 25, 1]
    z = zeros(p)
    z[S] = randn(length(S))     # supported on S
    zS = z[S]

    fS = restrict(f, S)
    @test nameof(typeof(fS)) == nameof(typeof(f))

    grad, gradS = zeros(p), zeros(length(S))
    v = value_and_gradient!(f, grad, z)
    vS = value_and_gradient!(fS, gradS, zS)
    @test gradS ≈ grad[S]
    @test value(fS, zS) ≈ vS

    # the restriction of L2Loss(y) drops the constant Σⱼ∉S yⱼ² / 2
    offset = f isa L2Loss{Float64, 1} ? sum(abs2, y[setdiff(1:p, S)]) / 2 : 0.
    @test vS + offset ≈ v
  end
end

@testset "gradient at a SparseIterate" begin
  @testset "$(nameof(typeof(f)))" for f in smooth_functions
    xs = SparseIterate(sprandn(p, 0.2))
    activate!(xs, findfirst(iszero, Vector(xs)))    # a stored zero
    for x in (xs, SparseIterate(p))
      grad, grad_sparse = zeros(p), fill(NaN, p)    # every entry must be written
      @test value_and_gradient!(f, grad_sparse, x) ≈ value_and_gradient!(f, grad, Vector(x))
      @test grad_sparse ≈ grad
    end
  end
end


end # module
