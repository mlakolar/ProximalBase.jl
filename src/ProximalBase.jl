module ProximalBase

export
  # types
  DifferentiableFunction,
  ProximableFunction,
  SparseIterate, SparseMatrixIterate,

  # smooth functions
  QuadraticFunction, L2Loss, LeastSquaresLoss,
  value, value_and_gradient!,

  # proximal functions
  ProxZero,
  ProxL1, AProxL1,
  ProxL2, ProxL2Sq, ProxNuclear,
  # ProxSumProx, ProxL1L2, ProxL1Nuclear,
  # AProxSumProx,
  ProxGaussLikelihood,
  prox!, prox,

  # utils
  shrink,
  proxL1Fused

include("utils.jl")
include("sparseIterate.jl")

# DifferentiableFunctions
include("differentiable_functions.jl")

# ProximableFunctions
include("proximal_functions.jl")

end
