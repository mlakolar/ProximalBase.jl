module DocstringsTest

using Test
using ProximalBase

const PB = ProximalBase

# Compatibility exports in src/deprecated.jl are removed in 0.6.
const COMPATIBILITY_EXPORTS = [
  :value, :cdprox!,
  :DifferentiableFunction, :ProximableFunction,
  :ProxZero, :ProxL1, :ProxL2, :ProxGroupL2, :ProxNuclear, :ProxL1Fused,
  :ProxGaussLikelihood, :ProxL2Sq, :L2Loss,
  :numCoordinates, :numActive, :activeCoordinate, :isActive,
  :ind2subLowerTriangular, :sub2indLowerTriangular, :proxL1Fused, :shrinkL2!,
  :A_mul_B_row, :At_mul_B_row, :At_mul_B_group!, :A_mul_B_group!,
  :A_mul_X_mul_B, :A_mul_X_mul_B_rc, :A_mul_UUt_mul_B, :A_mul_UUt_mul_B_rc,
]

const QUALIFIED_NAMES = [
  :ScalarLoss, :Squared, :Logistic, :Poisson, :Huber, :SquaredHinge,
  :SmoothCheck, :ConvolutionCheck, :deriv, :deriv2, :curvature,
  :value_and_deriv, :Smoothed, :AbstractGroupPenalty,
]

const TESTING_NAMES = [
  :Testing, :test_prox, :test_gradient, :test_separable, :test_conjugate,
  :test_kkt, :test_traits, :allocations, :brute_force_prox,
]

@testset "package docstrings" begin
  exports = filter(name -> Base.isexported(PB, name), names(PB; all = true))
  inventory = union(setdiff(exports, COMPATIBILITY_EXPORTS), QUALIFIED_NAMES)
  for (M, entries) in ((PB, inventory), (PB.Testing, TESTING_NAMES))
    @testset "$M.$name" for name in entries
      @test haskey(Docs.meta(M), Docs.Binding(M, name))
    end
  end
end

end
