"""
    ProximalBase

Building blocks for proximal, coordinate-descent and active-set methods:
functions with gradients and proximal operators, and sparse iterates.

A function `f` is callable, `f(x)` being its value, and its traits say what it
provides: [`issmooth`](@ref) functions have [`gradient!`](@ref) and
[`value_and_gradient!`](@ref), [`isproximable`](@ref) ones [`prox!`](@ref) and
[`value_and_prox!`](@ref), and [`isseparable`](@ref) ones the scalar proximal
operators [`prox_entry`](@ref), with which [`prox_coordinate!`](@ref) updates
one coordinate of an iterate. A function may be both smooth and proximable.

Proximal operators follow the convention

    prox_{γf}(x) = argmin_z  f(z) + ‖z - x‖² / (2γ),

and mutating functions take their output first, as `mul!` does:
`prox!(y, f, x, γ)`, `gradient!(g, f, x)`, `prox_coordinate!(x, f, k, γ)`.
"""
module ProximalBase

using LinearAlgebra
using SparseArrays
using LinearAlgebra: BlasInt, LAPACKException, checksquare, libblastrampoline
using LinearAlgebra.BLAS: @blasfunc

import Base: *

export
  # the interface
  AbstractFunction,
  issmooth, isproximable, isseparable, isconvex,
  prox!, prox, value_and_prox!,
  gradient!, gradient, value_and_gradient!, value_and_gradient,
  prox_coordinate!, prox_entry, kkt_violation, restrict,
  SeparableFunction, value_entry, iszeropreserving,
  gradient_mapping, gradient_mapping!, lipschitz,
  hashessian, hessian_vec!, hessian_vec, quadratic_model, quadratic_model!,
  coordinate_lipschitz, coordinate_lipschitz!,

  # smooth functions
  ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction, KronQuadratic,
  LeastSquaresLoss, GLMLoss, LogisticLoss, PoissonLoss, HuberLoss, LogDetLoss,

  # penalties
  NormL1, NormL1Nonneg, ElasticNet, NormL2, NormLinf, GroupNormL2, GroupNormLinf,
  SparseGroupNormL2,
  NormL1Fused, NuclearNorm, TracePSD, NormL0, MCP, SCAD, TotalVariation1D, FusedLasso, SLOPE,
  NegEntropy, KLDivergence, NegLog, Berhu,

  # nonsmooth losses
  CheckLoss, HingeLoss, EpsInsensitiveLoss,
  CappedL1, LogPenalty, NormLq, GroupMCP, GroupSCAD,

  # functions of the singular values
  SingularValue,

  # indicator functions
  IndBox, IndNonnegative, IndBallLinf, IndBallL0, IndBallL1, IndBallL2, IndSimplex,
  IndIntersection,
  IndHyperplane, IndHalfspace, IndPSD, IndBallRank, IndIsotonic,
  IndSphereL2,
  IndBallGroupL0,
  IndPattern, IndBanded, IndToeplitz,
  IndAffine,

  # functions built from functions
  Scaled, Tilt, Translate, Conjugate, SeparableSum, Sum, Constrained, Spectral,
  MoreauEnvelope, Precompose, Regularize,

  # functions of stacked arrays
  SliceSum, FiberSum,

  # iterates
  SparseIterate, SymmetricSparseIterate, AtomIterate,
  ncoordinates, coordinates, nactive, activecoordinates, isactive, activate!, atom,
  nnz, dropzeros!,

  # utilities
  shrink, norm_diff, tril_index, tril_cartesian

include("utils.jl")
include("lapack.jl")

include("sparseIterate.jl")
include("atomIterate.jl")

include("interface.jl")

include("functions/smooth.jl")
include("functions/leastsquares.jl")
include("functions/glm.jl")
include("functions/logdet.jl")
include("functions/separable.jl")
include("functions/losses.jl")
include("functions/entropy.jl")
include("functions/norms.jl")
include("functions/sparsegroup.jl")
include("functions/fibers.jl")
include("functions/spectral.jl")
include("functions/singular.jl")
include("functions/intersection.jl")
include("functions/sets.jl")
include("functions/ordered.jl")
include("functions/nonconvex.jl")
include("functions/matrixsets.jl")
include("functions/affine.jl")

include("calculus.jl")
include("constrained.jl")
include("spectralfunction.jl")
include("ballconstrained.jl")
include("constrained_nonconvex.jl")
include("second_order.jl")
include("envelope.jl")
include("composition.jl")
include("metric.jl")
include("deprecated.jl")
include("testing.jl")
include("precompile.jl")

end
