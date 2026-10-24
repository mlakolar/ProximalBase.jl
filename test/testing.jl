module TestingTest

# The public checks of ProximalBase.Testing pass on correct functions and fail
# on broken ones, so that none passes vacuously.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

const PB = ProximalBase
const PT = ProximalBase.Testing

Random.seed!(1)

# a test set that records its results without printing them
struct Collect <: Test.AbstractTestSet
  description::String
  results::Vector{Any}
end
Collect(description; kwargs...) = Collect(description, Any[])
Test.record(ts::Collect, r) = (push!(ts.results, r); r)
Test.finish(ts::Collect) = ts

# the number of failed or erroring tests of a check
function failures(check)
  ts = @testset Collect "probe" begin
    check()
  end
  count(r -> r isa Test.Fail || r isa Test.Error, ts.results)
end
passes(check) = failures(check) == 0

# broken functions, each wrong in one way
struct IdentityProx <: SeparableFunction               # λ|x| with the identity as its prox
  λ::Float64
end
PB.value_entry(f::IdentityProx, v, i) = f.λ * abs(v)
PB.prox_entry(::IdentityProx, v, i, γ) = float(v)
PB.isconvex(::Type{IdentityProx}) = true

struct WrongValue <: AbstractFunction                  # value_and_prox! reports twice the value
  λ::Float64
end
PB.isproximable(::Type{WrongValue}) = true
(f::WrongValue)(x) = f.λ * norm(x, 1)
PB.prox!(y::AbstractArray, f::WrongValue, x::AbstractArray, γ::Real) = (y .= shrink.(x, γ * f.λ))
PB.value_and_prox!(y::AbstractArray, f::WrongValue, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); 2 * f(y))

struct WrongGradient <: AbstractFunction               # ‖x‖²/2 with the gradient 2x
end
PB.issmooth(::Type{WrongGradient}) = true
(::WrongGradient)(x) = sum(abs2, x) / 2
PB.value_and_gradient!(g::AbstractArray, f::WrongGradient, x::AbstractArray) = (g .= 2 .* x; f(x))

struct FalseZero <: SeparableFunction                  # declares that it preserves zero, but shifts it
end
PB.value_entry(::FalseZero, v, i) = abs2(v - 0.1) / 2
PB.prox_entry(::FalseZero, v, i, γ) = (v + 0.1γ) / (1 + γ)
PB.iszeropreserving(::FalseZero) = true

struct WrongEntries <: SeparableFunction               # prox! overridden, disagreeing with prox_entry
  λ::Float64
end
PB.value_entry(f::WrongEntries, v, i) = f.λ * abs(v)
PB.prox_entry(f::WrongEntries, v, i, γ) = shrink(v, γ * f.λ)
PB.prox!(y::AbstractArray, ::WrongEntries, x::AbstractArray, γ::Real) = (y .= x)

struct WrongKKT <: SeparableFunction                   # NormL1 with twice the threshold in kkt_violation
  λ::Float64
end
PB.value_entry(f::WrongKKT, v, i) = f.λ * abs(v)
PB.prox_entry(f::WrongKKT, v, i, γ) = shrink(v, γ * f.λ)
PB.kkt_violation(f::WrongKKT, g::AbstractArray, i::Integer) = abs(g[i]) - 2f.λ

@testset "ProximalBase.Testing" begin
  x = 2 .* randn(12)
  w = rand(12) .+ 0.5
  γ = 0.7

  @testset "correct functions pass" begin
    @test passes(() -> PT.test_traits(NormL1(0.5); proximable = true, separable = true, convex = true))
    @test passes(() -> PT.test_prox(NormL1(0.5, w), x, γ))
    @test passes(() -> PT.test_prox(IndPSD(), Matrix(Symmetric(randn(5, 5))), γ; symmetric = true))
    X, y = randn(30, 12), randn(30)
    @test passes(() -> PT.test_gradient(LeastSquaresLoss(y, X), x))
    @test passes(() -> PT.test_gradient(LogisticLoss(rand(0:1, 30), X), x ./ 5; rng = Xoshiro(3)))
    groups = [1:3, 4:8, 9:12]
    for f in (NormL1(0.5, w), ElasticNet(0.3, 0.6), MCP(0.4), IndBox(-1.0, 2.0), NormL0(0.2))
      @test passes(() -> PT.test_separable(f, x, γ; groups))
    end
    Ws = Matrix(Symmetric(rand(5, 5)))
    @test passes(() -> PT.test_separable(NormL1(0.5, Ws), randn(5, 5), γ; symmetric = true))
    @test passes(() -> PT.test_conjugate(NormL1(0.5), x, γ; conjugate = IndBallLinf(0.5)))
    @test passes(() -> PT.test_conjugate(NormL1(0.5), x, γ))
    @test passes(() -> PT.test_conjugate(SqrNormL2(2.0), x, γ))
    @test passes(() -> PT.test_conjugate(NormL2(1.0), x, γ))
    # the value of its conjugate is not known: Moreau's decomposition alone
    @test passes(() -> PT.test_conjugate(ElasticNet(0.3, 0.6), x, γ))
    for f in (NormL1(0.5, w), ElasticNet(0.3, 0.6), MCP(0.4), SCAD(0.4), NormL1Nonneg(0.3), IndBox(-1.0, 2.0))
      @test passes(() -> PT.test_kkt(f, x, 0.5))
    end
    gn = GroupNormL2(0.5, groups)
    @test passes(() -> PT.test_kkt(gn, AtomIterate(12, gn), γ))
    @test passes(() -> PT.test_kkt(NormL1(0.5), AtomIterate(12, groups), γ))
  end

  @testset "broken functions fail" begin
    @test failures(() -> PT.test_traits(NormL1(0.5); smooth = true, proximable = true, separable = true,
                                        convex = true)) > 0
    @test failures(() -> PT.test_prox(IdentityProx(0.5), x, γ)) > 0
    @test failures(() -> PT.test_prox(WrongValue(0.5), x, γ)) > 0
    @test failures(() -> PT.test_gradient(WrongGradient(), x)) > 0
    @test failures(() -> PT.test_separable(FalseZero(), x, γ)) > 0
    @test failures(() -> PT.test_separable(WrongEntries(0.5), x, γ)) > 0
    @test failures(() -> PT.test_conjugate(NormL1(0.5), x, γ; conjugate = IndBallLinf(1.0))) > 0
    @test failures(() -> PT.test_kkt(WrongKKT(0.5), x, γ)) > 0
  end

  @testset "allocations and the scalar reference" begin
    a_sum(x) = sum(copy(x))
    @test PT.allocations(a_sum, x) > 0
    @test PT.allocations(prox!, similar(x), NormL1(0.5), x, γ) == 0
    for v in (-2.0, -0.2, 0.1, 1.5)
      t, h = PT.brute_force_prox(u -> 0.5 * abs(u), v, γ)
      @test t ≈ shrink(v, 0.5γ) atol = 1e-6
      @test h ≈ 0.5 * abs(t) + (t - v)^2 / (2γ)
    end
  end
end

end
