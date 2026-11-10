"""
    ProximalBase

Building blocks for proximal, coordinate-descent and active-set methods:
functions with gradients and proximal operators, and sparse iterates.

A function `f` is callable, `f(x)` being its value, and its traits say what it
provides: [`issmooth`](@ref) functions have [`gradient!`](@ref) and
[`value_and_gradient!`](@ref), [`isproximable`](@ref) ones [`prox!`](@ref) and
[`value_and_prox!`](@ref), and [`isseparable`](@ref) ones the scalar proximal
operators [`prox_block`](@ref). Solvers write the result through the iterate
storage API. A function may be both smooth and proximable.

Proximal operators follow the convention

    prox_{γf}(x) = argmin_z  f(z) + ‖z - x‖² / (2γ),

and mutating functions take their output first, as `mul!` does:
`prox!(y, f, x, γ)`, `gradient!(g, f, x)`, `prox_block!(out, f, u, k, γ)`.
"""
module ProximalBase

# ProximalBase defines its own ldexp. Base's ldexp(::BigFloat, n) is exact (mpfr_mul_2si, mpfr_mul_2ui) for a Clong or a
# Culong n, and forwards narrower integers to them; any other integer type reaches the fallback x * exp2(BigFloat(n)),
# which overflows when 2^n alone leaves the range although x·2^n is representable, and rounds n at a low precision. A C
# long is 32 bits on Windows, where an Int64 shift takes that fallback.
# This function shadows Base.ldexp for every unqualified call in the module (it does not extend Base's), so it must stay
# above every use of the name; write Base.ldexp where a query of Base's method table is meant. MPFR's exponent range
# lies within a Clong, so a shift saturated at the Clong limits still overflows or underflows exactly as it should.
@inline ldexp(x, n) = Base.ldexp(x, n)
@inline ldexp(x::BigFloat, n::Integer) = Base.ldexp(x, Clong(clamp(n, typemin(Clong), typemax(Clong))))

using LinearAlgebra
using SparseArrays
using LinearAlgebra: LAPACKException, checksquare

import Base: *

export
  # the interface
  PrecisionError,
  AbstractFunction,
  issmooth, isproximable, isseparable, isconvex,
  prox!, prox, value_and_prox!,
  gradient!, gradient, value_and_gradient!, value_and_gradient,
  prox_block, prox_block!, kkt_violation, restrict,
  SeparableFunction, value_entry, iszeropreserving,
  gradient_mapping, gradient_mapping!, lipschitz, opnorm2,
  reset_cache!,
  haslmo, lmo!, lmo, lmo_atom, VertexAtom, LowRankAtom,
  hashessian, hessian_vec!, hessian_vec, quadratic_model, quadratic_model!,
  coordinate_lipschitz, coordinate_lipschitz!,
  lla_weight, lla_weights!, lla_weights,

  # smooth functions
  ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction, KronQuadratic,
  LeastSquaresLoss, GLMLoss, LogisticLoss, PoissonLoss, HuberLoss, LogDetLoss, BernoulliLoss,

  # penalties
  NormL1, NormL1Nonneg, ElasticNet, NormL2, CubeNormL2, NormLinf, GroupNormL2, GroupNormLinf,
  SparseGroupNormL2,
  NormL1Fused, NuclearNorm, TracePSD, NormL0, MCP, SCAD, TotalVariation1D, FusedLasso, SLOPE, SumLargest,
  NegEntropy, KLDivergence, NegLog, Berhu,

  # separable losses
  CheckLoss, HingeLoss, SqrHingeLoss, EpsInsensitiveLoss,
  CappedL1, LogPenalty, NormLq, GroupMCP, GroupSCAD,

  # functions of the singular values
  SingularValue,

  # indicator functions
  IndBox, IndNonnegative, IndBinary, IndBallLinf, IndBallL0, IndBallL1, IndBallL2, IndSimplex,
  IndIntersection,
  IndHyperplane, IndHalfspace, IndHyperslab, IndPSD, IndBallRank, IndStiefel, IndIsotonic, IndSOC,
  IndRotatedSOC, IndExpPrimal, IndExpDual, IndSphereL2,
  IndBallGroupL0,
  IndPattern, IndBanded, IndToeplitz,
  IndAffine, IndPolyhedron, IndGraph,

  # functions built from functions
  Scaled, Tilt, Translate, Conjugate, SeparableSum, Sum, Constrained, Spectral,
  MoreauEnvelope, SqrDistL2, DistL2, Precompose, LinearImage, Regularize, PointwiseMinimum,

  # functions of stacked arrays
  SliceSum, FiberSum,

  # iterates
  SparseIterate, SymmetricSparseIterate, AtomIterate,
  ncoordinates, coordinates, nactive, activecoordinates, activevalues, matrixview, isactive,
  activate!, atom, atomindices,
  nnz, dropzeros!,

  # utilities
  shrink, norm_diff, tril_index, tril_cartesian

# A public statement does not parse on Julia 1.10.
VERSION >= v"1.11" && Core.eval(@__MODULE__, Expr(:public,
  :ScalarLoss, :Squared, :Logistic, :Poisson, :IdentityBernoulli, :Huber, :SquaredHinge,
  :SmoothCheck, :ConvolutionCheck, :deriv, :deriv2, :curvature,
  :value_and_deriv, :Smoothed, :AbstractGroupPenalty, :Testing, :check_parameters,
  :entry_parameters, :check_symmetric_parameters, :samegroups))

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
include("functions/cones.jl")
include("functions/ordered.jl")
include("functions/nonconvex.jl")
include("functions/matrixsets.jl")
include("lmo.jl")
include("functions/affine.jl")
include("functions/polyhedron.jl")
include("functions/graph.jl")
include("lla.jl")

include("calculus.jl")
include("constrained.jl")
include("spectralfunction.jl")
include("lmo_atoms.jl")
include("ballconstrained.jl")
include("affineconstrained.jl")
include("constrained_nonconvex.jl")
include("second_order.jl")
include("envelope.jl")
include("distance.jl")
include("composition.jl")
include("image.jl")
include("minimum.jl")
include("model_data.jl")
include("metric.jl")
include("symmetric_parameters.jl")
include("caches.jl")
include("deprecated.jl")
include("testing.jl")
include("block_prox.jl")
include("precompile.jl")

end
