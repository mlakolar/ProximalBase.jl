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
using LinearAlgebra: LAPACKException, checksquare

import Base: *

export
  # the interface
  AbstractFunction,
  issmooth, isproximable, isseparable, isconvex,
  prox!, prox, value_and_prox!,
  gradient!, gradient, value_and_gradient!, value_and_gradient,
  prox_coordinate!, prox_entry, kkt_violation, restrict,
  SeparableFunction, value_entry, iszeropreserving,
  gradient_mapping, gradient_mapping!, lipschitz, opnorm2,
  reset_cache!,
  hashessian, hessian_vec!, hessian_vec, quadratic_model, quadratic_model!,
  coordinate_lipschitz, coordinate_lipschitz!,
  lla_weight, lla_weights!, lla_weights,

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
  MoreauEnvelope, Precompose, Regularize, PointwiseMinimum,

  # functions of stacked arrays
  SliceSum, FiberSum,

  # iterates
  SparseIterate, SymmetricSparseIterate, AtomIterate,
  ncoordinates, coordinates, nactive, activecoordinates, isactive, activate!, atom,
  nnz, dropzeros!,

  # utilities
  shrink, norm_diff, tril_index, tril_cartesian

# A public statement does not parse on Julia 1.10.
VERSION >= v"1.11" && Core.eval(@__MODULE__, Expr(:public,
  :ScalarLoss, :Squared, :Logistic, :Poisson, :Huber, :SquaredHinge,
  :SmoothCheck, :ConvolutionCheck, :deriv, :deriv2, :curvature,
  :value_and_deriv, :Smoothed, :AbstractGroupPenalty, :Testing, :check_parameters))

include("compat.jl")
include("utils.jl")
include("lapack.jl")
include("partial.jl")

include("sparse_iterate.jl")
include("atom_iterate.jl")

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
include("lla.jl")

include("calculus.jl")
include("constrained.jl")
include("spectralfunction.jl")
include("ballconstrained.jl")
include("affineconstrained.jl")
include("constrained_nonconvex.jl")
include("second_order.jl")
include("envelope.jl")
include("composition.jl")
include("minimum.jl")
include("metric.jl")
include("caches.jl")
include("deprecated.jl")
include("testing.jl")
include("precompile.jl")

end
