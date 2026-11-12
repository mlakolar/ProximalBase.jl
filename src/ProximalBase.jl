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
  ZeroFunction, LinearFunction, SqrNormL2, QuadraticFunction, LeastSquaresLoss, NormL1, GroupNormL2, Scaled,
  SparseIterate, SymmetricSparseIterate, BlockIterate,
  ncoordinates, coordinates, nactive, activecoordinates, activevalues, matrixview, isactive,
  activate!, blockview, blockindices, nnz, dropzeros!, shrink, norm_diff, tril_index, tril_cartesian

VERSION >= v"1.11" && Core.eval(@__MODULE__, Expr(:public,
  :ScalarLoss, :Squared, :deriv, :deriv2, :curvature, :value_and_deriv,
  :check_parameters, :entry_parameters, :check_symmetric_parameters, :samegroups))

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
include("functions/scalar_loss.jl")
include("functions/separable.jl")
include("functions/norml1.jl")
include("functions/groupnorml2.jl")
include("functions/scaled.jl")
include("array_steps.jl")
include("functions/scaled_array.jl")
include("symmetric_parameters.jl")
include("block_prox.jl")
include("caches.jl")
include("precompile.jl")

end
