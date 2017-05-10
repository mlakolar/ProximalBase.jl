facts("prox zero") do
  f = ProxZero()

  p = 10
  x = randn(p)
  z = zeros(p)

  @fact prox!(f, z, x) --> x
  @fact prox(f,x) --> z
end

facts("proximal_l1") do
  context("shrink to zero") do
    x = randn(10)
    lambda = maximum(abs, x) + 0.1
    g = ProxL1(lambda)

    hat_x = randn(10)
    prox!(g, hat_x, x)
    @fact maximum(abs, hat_x) --> 0.

    @fact value(g, x) --> roughly(lambda * sum(abs, x))
  end

  context("shrink towards zero") do
    x = [1.0, 1.4, -3.2]
    lambda = 0.1
    g = ProxL1(lambda)

    hat_x = randn(3)
    prox!(g, hat_x, x)
    @fact  hat_x --> roughly([0.9, 1.3, -3.1])

    @fact value(g, x) --> roughly(lambda * sum(abs, x))
    @fact value(g, hat_x) --> roughly(lambda * sum(abs, hat_x))
  end

  context("shrink with vector") do
    x = [1.0, 1.4, -3.2]
    lambda = [0.1, 1.5, 1]
    g = AProxL1(lambda)

    hat_x = randn(3)
    prox!(g, hat_x, x)
    @fact hat_x --> roughly([0.9, 0., -2.2])
    @fact prox(g, x) --> roughly([0.9, 0., -2.2])

    @fact value(g, x) --> roughly( dot(lambda, abs.(x)) )
    @fact value(g, hat_x) --> roughly( dot(lambda, abs.(hat_x)) )
  end

end

facts("proximal_l2") do

  context("shrink to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)

    lambda = norm(x) + 0.1
    g = ProxL2(lambda)
    # check the norm
    @fact value(g, x) --> roughly(lambda * norm(x))
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @fact value(g, x) --> roughly(lambda * sqrt(ln))

    # check shrinkage
    prox!(g, hat_x, x, 1.)
    @fact hat_x --> roughly(zeros(Float64, p))
    @fact norm(hat_x) --> roughly(0.)
  end

  context("shrink not to zero") do
    p = 2
    x = [1., 2.]
    hat_x = randn(p)

    lambda = 1.
    # check the norm
    g = ProxL2(lambda)

    @fact value(g, x) --> roughly(lambda * norm(x))
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @fact value(g, x) --> roughly(lambda * sqrt(ln))

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @fact hat_x --> roughly((1.-1./sqrt(5.))*x)
    @fact norm(hat_x) --> roughly(sqrt(5.)-1.)
  end

end

facts("proximal_l2sq") do

  context("prox operator") do
    x = [1., 2.]
    lambda = 1.

    # norm value
    g = ProxL2Sq(0.)
    @fact value(g, x) --> 0.

    g = ProxL2Sq(1.)
    @fact value(g, x) --> sum(abs2, x)

    g = ProxL2Sq(2.)
    @fact value(g, x) --> roughly(2. * norm(x)^2)

    # prox
    hat_x = similar(x)

    g = ProxL2Sq(0.)
    @fact prox!(g, hat_x, x) --> roughly(x)

    g = ProxL2Sq(1.)
    @fact prox!(g, hat_x, x) --> roughly(x / 3.)

    g = ProxL2Sq(2.)
    @fact prox!(g, hat_x, x) --> roughly(x / 5.)


  end

end


facts("proximal_nuclear") do

  y = randn(30, 10)
  y = y' * y

  U, S, V = svd(y)
  for k=1:10
    S[k] = max(S[k] - 0.1, 0.)
  end
  tx = U * (Diagonal(S)*V')

  x = randn(10, 10)
  g = ProxNuclear(0.1)
  prox!(g, x, y)

  @fact x --> roughly(tx)
end



#
# facts("proximal_l1l2") do
#
#   context("shrink to zero") do
#     p = 10
#     x = randn(p)
#     hat_x = randn(p)
#
#     groups = Array(UnitRange{Int64}, 2)
#     groups[1] = 1:5
#     groups[2] = 6:10
#
#     lambda = norm(x) + 0.1
#
#     # check the norm
#     g = ProxL1L2(lambda, groups)
#     @fact value(g, x) --> roughly(lambda * norm(x[1:5]) + lambda * norm(x[6:10]))
#
#     # check shrinkage
#     prox!(g, hat_x, x, 1.)
#
#     @fact hat_x --> roughly(zeros(Float64, p))
#     @fact norm(hat_x) --> roughly(0.)
#   end
#
#   context("shrink not to zero") do
#     p = 10
#     x = randn(p)
#     hat_x = randn(p)
#
#     groups = Array(UnitRange{Int64}, 2)
#     groups[1] = 1:5
#     groups[2] = 6:10
#
#     lambda = min(norm(x[1:5]), norm(x[6:10])) - 0.2
#
#     # check the norm
#     g = ProxL1L2(lambda, groups)
#     @fact value(g, x) --> roughly(lambda * norm(x[1:5]) + lambda * norm(x[6:10]))
#
#     # check shrinkage
#     prox!(g, hat_x, x, 1.)
#
#     @fact hat_x[1:5] --> roughly( (1.-lambda/norm(x[1:5])) * x[1:5] )
#     @fact hat_x[6:10] --> roughly( (1.-lambda/norm(x[6:10])) * x[6:10] )
#   end
#
# end
#
#
#
