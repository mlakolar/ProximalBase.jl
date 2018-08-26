module  DifferentiableFunctionsTests

using Test
using ProximalBase
using Random
using LinearAlgebra

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


end # module
