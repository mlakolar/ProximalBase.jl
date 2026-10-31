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
  :value_and_deriv, :Smoothed, :AbstractGroupPenalty, :check_parameters,
]

const TESTING_NAMES = [
  :Testing, :test_prox, :test_gradient, :test_separable, :test_conjugate,
  :test_kkt, :test_iterate, :test_traits, :allocations, :brute_force_prox,
]

@testset "public qualified names" begin
  if VERSION >= v"1.11"
    for (M, entries) in ((PB, union(QUALIFIED_NAMES, [:Testing])),
                         (PB.Testing, setdiff(TESTING_NAMES, [:Testing])))
      @testset "$M.$name" for name in entries
        @test Base.ispublic(M, name)
        @test !Base.isexported(M, name)
      end
      actual = Set(name for name in names(M; all = true)
        if Base.ispublic(M, name) && !Base.isexported(M, name))
      @test actual == Set(entries)
    end
  end
end

@testset "package docstrings" begin
  exports = filter(name -> Base.isexported(PB, name), names(PB; all = true))
  inventory = union(setdiff(exports, COMPATIBILITY_EXPORTS), QUALIFIED_NAMES)
  for (M, entries) in ((PB, inventory), (PB.Testing, TESTING_NAMES))
    @testset "$M.$name" for name in entries
      documented = haskey(Docs.meta(M), Docs.Binding(M, name))
      @test documented
    end
  end
end

@testset "manual docstring entries" begin
  entries = Set{String}()
  source = joinpath(dirname(@__DIR__), "docs", "src")
  for (root, _, files) in walkdir(source), file in files
    endswith(file, ".md") || continue
    page = read(joinpath(root, file), String)
    for block in eachmatch(r"(?m)^```@docs\s*\r?\n(.*?)^```\s*$"s, page)
      for line in split(block.captures[1], '\n')
        entry = match(r"^(ProximalBase\.(?:Testing\.)?\w+)(?:\s*\(.*\))?$", strip(line))
        entry === nothing || push!(entries, entry.captures[1])
      end
    end
  end
  required = ["ProximalBase.$name" for name in QUALIFIED_NAMES]
  append!(required, [name == :Testing ? "ProximalBase.Testing" :
    "ProximalBase.Testing.$name" for name in TESTING_NAMES])
  @testset "$name" for name in required
    @test name in entries
  end
end

end
