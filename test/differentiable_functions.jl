facts("L2Loss") do

  x = randn(5)
  y = randn(5)
  f1 = L2Loss()
  f2 = L2Loss(y)

  @fact value(f1, x) --> roughly(sum(abs2,x)/2.)
  @fact value(f2, x) --> roughly(sum(abs2,x-y)/2.)

  hat_x = similar(x)
  @fact value_and_gradient!(f1, hat_x, x) --> roughly(sum(abs2,x)/2.)
  @fact hat_x --> roughly(x)
  hat_x = similar(x)
  @fact value_and_gradient!(f2, hat_x, x) --> roughly(sum(abs2,x-y)/2.)
  @fact hat_x --> roughly(x-y)

end

facts("Quadratic Function") do

  A = randn(10, 10)
  A = A + A'
  b = randn(10)
  c = 1.

  x = randn(10)

  q1 = QuadraticFunction(A)
  q2 = QuadraticFunction(A, b)
  q3 = QuadraticFunction(A, b, c)

  @fact value(q1, x) --> roughly(dot(x, A*x)/2.)
  @fact value(q2, x) --> roughly(dot(x, A*x)/2. + dot(x, b))
  @fact value(q3, x) --> roughly(dot(x, A*x)/2. + dot(x, b) + c)

  hat_x = similar(x)
  @fact value_and_gradient!(q1, hat_x, x) --> roughly(dot(x, A*x)/2.)
  @fact hat_x --> roughly(A*x)

  @fact value_and_gradient!(q2, hat_x, x) --> roughly(dot(x, A*x)/2. + dot(x, b))
  @fact hat_x --> roughly(A*x + b)

  @fact value_and_gradient!(q3, hat_x, x) --> roughly(dot(x, A*x)/2. + dot(x, b) + c)
  @fact hat_x --> roughly(A*x + b)
end


facts("Least Squares Loss") do

  y = randn(10)
  X = randn(10, 5)
  b = rand(5)
  f = LeastSquaresLoss(y, X)

  @fact value(f, b) --> roughly( vecnorm(y-X*b)^2 / 2. )
  grad_out = zeros(5)
  @fact value_and_gradient!(f, grad_out, b) --> roughly( vecnorm(y-X*b)^2 / 2. )
  @fact grad_out --> roughly( -X'*(y - X*b) )

end
