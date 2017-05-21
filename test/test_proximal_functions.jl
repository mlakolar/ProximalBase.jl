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


facts("proximal_l1_fused") do

context("random") do
  if grb
    solver = Gurobi.GurobiSolver(OutputFlag=0)
  else
    solver = SCS.SCSSolver(eps=1e-6, verbose=0)
  end

  if jmp
    srand(123)
    m = JuMP.Model(solver=solver)
    JuMP.@variable(m, z1)
    JuMP.@variable(m, z2)
    JuMP.@variable(m, t[1:3] >= 0)
    JuMP.@constraint(m, z1 <= t[1] )
    JuMP.@constraint(m, -t[1] <= z1 )
    JuMP.@constraint(m, z2 <= t[2] )
    JuMP.@constraint(m, -t[2] <= z2 )
    JuMP.@constraint(m, z1 - z2 <= t[3])
    JuMP.@constraint(m, -t[3] <= z1 - z2 )

    for i=1:1000
      x1 = randn()
      x2 = randn()
      λ1 = rand(Uniform(0.,1.))
      λ2 = rand(Uniform(0.,1.))

      JuMP.@objective(m, Min, ((x1-z1)^2+(x2-z2)^2)/2. + λ1 * (t[1]+t[2]) + λ2 * t[3])
      JuMP.solve(m)

      zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, λ1, λ2)

      @fact abs(JuMP.getvalue(z1) - zp1) + abs(JuMP.getvalue(z2) - zp2)  --> roughly(0.; atol=1e-4)
    end

    m = JuMP.Model(solver=solver)
    JuMP.@variable(m, z1)
    JuMP.@variable(m, z2)
    JuMP.@variable(m, t >= 0)
    JuMP.@constraint(m, z1 - z2 <= t)
    JuMP.@constraint(m, -t <= z1 - z2 )

    for i=1:1000
      x1 = randn()
      x2 = randn()
      λ = rand(Uniform(0.,1.))

      JuMP.@objective(m, Min, ((x1-z1)^2+(x2-z2)^2)/2. + λ * t)
      JuMP.solve(m)

      zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, 0., λ)

      @fact abs(JuMP.getvalue(z1) - zp1) + abs(JuMP.getvalue(z2) - zp2)  --> roughly(0.; atol=1e-4)
    end
  end
end

context("nonrandom") do
  srand(123)

  x1 = randn()
  x2 = randn()
  λ1 = rand(Uniform(0.,1.))

  zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, λ1, 0.)
  @fact ProximalBase.shrink(x1, λ1) --> zp1
  @fact ProximalBase.shrink(x2, λ1) --> zp2

  λ1 = 1000.
  zp1, zp2 = ProximalBase.proxL1Fused(x1, x2, λ1, 0.)
  @fact ProximalBase.shrink(x1, λ1) --> 0.
  @fact ProximalBase.shrink(x2, λ1) --> 0.

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



facts("proximal_logdet") do
  p = 10

  out = zeros(p,p)

  # identity
  Σ = eye(p)
  V = eye(p)
  g = ProxGaussLikelihood(Σ)
  @fact prox(g, V) --> roughly(eye(p))
  @fact prox!(g, out, V) --> roughly(eye(p))

  γ = 0.5
  ρ = 2.
  sol = eye(p) * (-(1.-ρ)+sqrt((1.-ρ)^2+4.*ρ)) / (2.*ρ)
  @fact prox!(g, out, V, γ) --> roughly(sol)

end
