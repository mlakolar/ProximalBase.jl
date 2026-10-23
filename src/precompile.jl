####################################
#
#  Precompilation
#
####################################

# The calls that solvers make most, compiled into the package image so that their
# first use does not pay for compilation. PrecompileTools runs this workload when
# the package is precompiled; the preference `precompile_workload = false` skips it.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  n, p = 30, 10
  x = [sin(1.0 * i) for i in 1:p]
  X = [cos(1.0 * (i + 3j)) for i in 1:n, j in 1:p]
  b = [sin(0.5 * i) for i in 1:n]
  P = X'X / n + I                       # symmetric positive definite
  groups = [1:5, 6:10]
  @compile_workload begin
    # proximal operators and values on a vector
    y = similar(x)
    for f in (NormL1(0.1), NormL1(0.1, fill(1.0, p)), ElasticNet(0.1, 0.2), IndBox(-1.0, 1.0), MCP(0.1), SCAD(0.1),
              NormL2(0.1), GroupNormL2(0.1, groups), IndSimplex(1.0), IndBallL1(1.0), IndBallL2(1.0), IndBallL0(3))
      prox!(y, f, x, 1.0)
      value_and_prox!(y, f, x, 1.0)
      f(x)
    end
    # values and gradients of the losses
    g = similar(x)
    for f in (LeastSquaresLoss(b, X), LogisticLoss(Float64.(b .> 0), X), QuadraticFunction(P))
      value_and_gradient!(g, f, x)
      gradient!(g, f, x)
      f(x)
    end
    # the functions on matrices that call LAPACK
    Y = similar(P)
    for f in (LogDetLoss(P), NuclearNorm(0.1), IndPSD())
      value_and_prox!(Y, f, P, 0.5)
    end
    # coordinate steps on the three iterates
    prox_coordinate!(SparseIterate(x), NormL1(0.1), 2, 1.0)
    S = SymmetricSparseIterate(P)
    prox_coordinate!(S, NormL1(0.1), 2, 1.0)
    prox_coordinate!(S, NormL1(0.1, fill(1.0, p, p)), 3, 1.0)
    fg = GroupNormL2(0.1, groups)
    prox_coordinate!(AtomIterate(x, fg), fg, 1, 1.0)
    # the Kronecker quadratic and the hard threshold of a symmetric matrix
    value_and_gradient!(similar(P), KronQuadratic(P, P, P), S)
    prox!(similar(S), IndBallL0(5; diagonal = false), Symmetric(P, :L), 1.0)
  end
end
