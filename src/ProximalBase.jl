"""
    ProximalBase

Sparse iterate storage and mathematical functions with values, gradients and
proximal operators. Mutating oracles take their output first. The convention is
`prox_{γf}(x) = argmin_z f(z) + ‖z-x‖²/(2γ)`.
"""
module ProximalBase

@inline ldexp(x, n) = Base.ldexp(x, n)
@inline ldexp(x::BigFloat, n::Integer) = Base.ldexp(x, Clong(clamp(n, typemin(Clong), typemax(Clong))))

using LinearAlgebra
using LinearAlgebra: checksquare
using SparseArrays
import Base: *

export AbstractFunction, SeparableFunction,
  issmooth, isproximable, isseparable, isconvex, iszeropreserving,
  prox!, prox, value_and_prox!, gradient!, gradient, value_and_gradient!, value_and_gradient,
  prox_block, prox_block!, value_entry, kkt_violation, restrict, lipschitz, opnorm2, reset_cache!,
  ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction, KronQuadratic, LeastSquaresLoss, SqrtLeastSquaresLoss, LogDetLoss, LogisticLoss,
  NormL0, NormL1, NormL1Nonneg, ElasticNet, SCAD, IndNonnegative, IndBox, IndBallL0, IndBallL1, IndBallL2, IndBallLinf, NormL2, NormLinf, GroupNormL2, GroupNormLinf, SparseGroupNormL2, Scaled,
  SparseIterate, SymmetricSparseIterate, BlockIterate,
  ncoordinates, coordinates, nactive, activecoordinates, activevalues, matrixview, isactive,
  activate!, blockview, blockindices, nnz, dropzeros!, shrink, norm_diff, tril_index, tril_cartesian

export MCP

VERSION >= v"1.11" && Core.eval(@__MODULE__, Expr(:public,
  :check_parameters, :entry_parameters, :check_symmetric_parameters, :samegroups,
  :value_from_predictor, :value_and_gradient_from_predictor!, :value_from_residual, :leastsquares_data,
  :kronquadratic_data))

include("compat.jl")
include("utils.jl")
include("numerical_kernels.jl")
include("iterate_storage.jl")
include("sparse_iterate.jl")
include("symmetric_sparse_iterate.jl")
include("block_iterate.jl")
include("interface.jl")
include("functions/zero.jl")
include("functions/linear.jl")
include("functions/sqrnorml2.jl")
include("functions/quadratic.jl")
include("functions/leastsquares.jl")
include("functions/sqrt_leastsquares.jl")
include("functions/logdet.jl")
include("functions/separable.jl")
include("functions/norml1.jl")
include("functions/elasticnet.jl")
include("functions/norml0.jl")
include("functions/norml0offdiag.jl")
include("functions/mcp.jl")
include("functions/scad.jl")
include("functions/norml1nonneg.jl")
include("functions/indnonnegative.jl")
include("functions/indbox.jl")
include("functions/norml2.jl")
include("functions/normlinf.jl")
include("functions/groupnormlinf.jl")   # before groupnorml2.jl: its shared methods dispatch on both group types
include("functions/groupnorml2.jl")
include("functions/sparsegroupnorml2.jl")
include("functions/scaled.jl")
include("functions/kronquadratic.jl")   # after scaled.jl: its allocating oracles have Scaled methods
include("functions/indballl1.jl")
include("functions/indballl2.jl")
include("functions/indballl0.jl")
include("functions/leastsquares_predictor.jl")
include("functions/logisticloss.jl")
include("array_steps.jl")
include("functions/scaled_array.jl")
include("symmetric_parameters.jl")
include("block_prox.jl")
include("caches.jl")
include("precompile.jl")

end
