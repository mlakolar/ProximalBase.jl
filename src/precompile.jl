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
  x32 = Float32.(x)
  X = [cos(1.0 * (i + 3j)) for i in 1:n, j in 1:p]
  b = [sin(0.5 * i) for i in 1:n]
  P = X'X / n + I                       # symmetric positive definite
  groups = [1:5, 6:10]
  slope_weights = collect(range(0.2, 0.1; length = p))
  @compile_workload begin
    # proximal operators and values on a vector
    y = similar(x)
    for f in (NormL1(0.1), NormL1(0.1, fill(1.0, p)), ElasticNet(0.1, 0.2), IndBox(-1.0, 1.0), MCP(0.1), SCAD(0.1),
              NormL2(0.1), GroupNormL2(0.1, groups), IndSimplex(1.0), IndBallL1(1.0), IndBallL2(1.0), IndBallL0(3),
              SLOPE(slope_weights), TotalVariation1D(0.1), IndIsotonic())
      prox!(y, f, x, 1.0)
      value_and_prox!(y, f, x, 1.0)
      f(x)
    end
    prox!(similar(x32), NormL1(0.1f0), x32, 1.0f0)
    prox(NormL1(0.1), x, 1.0)
    # values and gradients of the losses
    g = similar(x)
    for f in (LeastSquaresLoss(b, X), LogisticLoss(Float64.(b .> 0), X), QuadraticFunction(P))
      value_and_gradient!(g, f, x)
      gradient!(g, f, x)
      f(x)
    end
    gradient(LeastSquaresLoss(b, X), x)
    # block sums, their conjugate and gradients
    blocks = (copy(x), copy(x))
    out = (similar(x), similar(x))
    s = SeparableSum(NormL1(0.1), SqrNormL2())
    prox!(out, s, blocks, 1.0)
    value_and_prox!(out, s, blocks, 1.0)
    s(blocks)
    c = Conjugate(s)
    prox!(out, c, blocks, 1.0)
    value_and_prox!(out, c, blocks, 1.0)
    smooth = SeparableSum(SqrNormL2(), SqrNormL2(2.0))
    gradient!(out, smooth, blocks)
    value_and_gradient!(out, smooth, blocks)
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
    xa = AtomIterate(x, fg)
    prox_coordinate!(xa, fg, 1, 1.0)
    kkt_violation(fg, g, xa, 1)
    # the Kronecker quadratic and the hard threshold of a symmetric matrix
    value_and_gradient!(similar(P), KronQuadratic(P, P, P), S)
    prox!(similar(S), IndBallL0(5; diagonal = false), Symmetric(P, :L), 1.0)
  end
end
