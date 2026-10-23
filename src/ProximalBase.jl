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

  # smooth functions
  ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction, KronQuadratic,
  LeastSquaresLoss, GLMLoss, LogisticLoss, PoissonLoss, HuberLoss, LogDetLoss,

  # penalties
  NormL1, NormL1Nonneg, ElasticNet, NormL2, NormLinf, GroupNormL2, GroupNormLinf,
  NormL1Fused, NuclearNorm, TracePSD, NormL0, MCP, SCAD,

  # indicator functions
  IndBox, IndNonnegative, IndBallLinf, IndBallL0, IndBallL1, IndBallL2, IndSimplex,
  IndHyperplane, IndHalfspace, IndPSD, IndBallRank,

  # functions built from functions
  Scaled, Tilt, Translate, Conjugate, SeparableSum, Sum,

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
include("functions/norms.jl")
include("functions/spectral.jl")
include("functions/sets.jl")

include("calculus.jl")
include("deprecated.jl")

end
