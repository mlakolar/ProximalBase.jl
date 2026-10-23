module SingularTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# the reference: the proximal operator of f on the singular values of a dense SVD
function sv_prox(f, X, γ)
  F = svd(X)
  F.U * Diagonal(prox(f, F.S, γ)) * F.Vt
end

const CATALOG = (NormL1(0.5), NormLinf(0.8), IndBallL1(2.), IndBallLinf(1.), IndBallL0(2), ElasticNet(0.3, 0.4),
                 MCP(0.6, 2.5), SCAD(0.6), NormL0(0.3))

@testset "against the dense reference" begin
  for (m, n) in ((9, 6), (6, 9), (7, 7)), f in CATALOG
    X = 2 .* randn(m, n)
    F = SingularValue(f)
    @testset "$(nameof(typeof(f))), $m × $n" begin
      for γ in (0.3, 1., 3.)
        P = prox(F, X, γ)
        @test P ≈ sv_prox(f, X, γ) atol = 1e-10
        # the value from the proximal operator is f at the new singular values
        @test value_and_prox!(similar(X), F, X, γ) ≈ f(prox(f, svdvals(X), γ)) atol = 1e-10
      end
      check_prox(F, X, 0.7)
      @test F(X) ≈ f(svdvals(X)) atol = 1e-12
    end
  end
end

@testset "the nuclear norm and the rank" begin
  X = randn(12, 8)
  for γ in (0.1, 1., 4.)
    @test prox(SingularValue(NormL1(0.5)), X, γ) ≈ prox(NuclearNorm(0.5), X, γ)
    @test value_and_prox!(similar(X), SingularValue(NormL1(0.5)), X, γ) ≈ value_and_prox!(similar(X), NuclearNorm(0.5), X, γ)
    @test prox(SingularValue(IndBallL0(3)), X, γ) ≈ prox(IndBallRank(3), X, γ)
  end
  @test SingularValue(NormL1(0.5))(X) ≈ NuclearNorm(0.5)(X)
  # the rank indicator counts singular values within rounding of zero as zero
  P = prox(SingularValue(IndBallL0(3)), X, 1.)
  @test SingularValue(IndBallL0(3))(P) == 0 == IndBallRank(3)(P)
  @test SingularValue(IndBallL0(3))(X) == Inf == IndBallRank(3)(X)
  # and so does the value of a penalty, which then agrees with value_and_prox!
  f = SingularValue(NormL0(0.4))
  v = value_and_prox!(similar(X), f, X, 3.)
  @test f(prox(f, X, 3.)) == v
end

@testset "norms and balls" begin
  X = 2 .* randn(10, 7)
  σ = svdvals(X)
  @test SingularValue(NormLinf(0.8))(X) ≈ 0.8 * opnorm(X)
  # the projection onto the nuclear-norm ball reaches its boundary, and keeps a point inside
  P = prox(SingularValue(IndBallL1(2.)), X)
  @test sum(svdvals(P)) ≈ 2
  @test SingularValue(IndBallL1(2.))(P) == 0
  Xs = X ./ (2 * sum(σ))
  @test prox(SingularValue(IndBallL1(2.)), Xs) ≈ Xs
  # the projection onto the spectral-norm ball clips the singular values
  P = prox(SingularValue(IndBallLinf(1.5)), X)
  @test svdvals(P) ≈ min.(σ, 1.5)
  # the matrix elastic net shrinks and scales the singular values
  @test svdvals(prox(SingularValue(ElasticNet(0.3, 0.4)), X, 0.5)) ≈ max.(σ .- 0.15, 0) ./ 1.2
  # nonconvex thresholding keeps large singular values unchanged
  P = prox(SingularValue(MCP(0.5, 2.)), X, 1.)
  s = svdvals(P)
  @test s[σ .> 1.] ≈ σ[σ .> 1.]
end

@testset "conjugates" begin
  X = 2 .* randn(8, 5)
  λ, γ = 0.8, 0.6
  # the spectral norm and the indicator of the nuclear-norm ball are conjugate
  for F in (SingularValue(NormLinf(λ)), SingularValue(IndBallL1(λ)))
    C = Conjugate(F)
    P = prox(F, X, γ)
    S = (X .- P) ./ γ
    @test F(P) + C(S) ≈ dot(P, S) rtol = 1e-10 atol = 1e-10       # Fenchel–Young, with equality
    @test prox(F, X, γ) .+ γ .* prox(C, X ./ γ, inv(γ)) ≈ X          # Moreau's decomposition
    check_prox(C, X, γ)
  end
  Y = randn(8, 5)
  @test Conjugate(SingularValue(IndBallL1(λ)))(Y) ≈ λ * opnorm(Y)
  @test Conjugate(SingularValue(NormLinf(λ)))(Y ./ (2 * sum(svdvals(Y)) / λ)) == 0
  @test Conjugate(SingularValue(NormLinf(λ)))(Y) == Inf
end

@testset "traits, inputs and outputs" begin
  check_traits(SingularValue(NormL1(0.5)); proximable = true, convex = true)
  check_traits(SingularValue(MCP(0.5)); proximable = true)
  # an indicator of singular values is an indicator, which other functions may ask
  @test ProximalBase._isindicator(SingularValue(IndBallL1(1.))) && !ProximalBase._isindicator(SingularValue(NormL1(1.)))
  @test value_and_prox!(zeros(6, 4), SingularValue(IndBallLinf(1.)), randn(6, 4), 1.) == 0
  # a symmetric matrix, and a SymmetricSparseIterate as the output
  A = randn(6, 6); A = A + A'
  S = SymmetricSparseIterate(6)
  prox!(S, SingularValue(NormL1(0.4)), A, 1.)
  @test Matrix(S) ≈ prox(NuclearNorm(0.4), A, 1.)
  @test prox(SingularValue(NormL1(0.4)), Symmetric(A), 1.) ≈ prox(NuclearNorm(0.4), A, 1.)
  # in place
  X = randn(6, 4)
  Y = copy(X)
  prox!(Y, SingularValue(NormLinf(0.5)), Y, 1.)
  @test Y ≈ sv_prox(NormLinf(0.5), X, 1.)
  # a new size or element type gets a new workspace
  F = SingularValue(NormL1(0.5))
  @test prox(F, X, 1.) ≈ prox(NuclearNorm(0.5), X, 1.)
  X2 = randn(3, 7)
  @test prox(F, X2, 1.) ≈ prox(NuclearNorm(0.5), X2, 1.)
  @test prox(F, Float32.(X2), 1.) isa Matrix{Float32}
  @test prox(F, Float32.(X2), 1.) ≈ prox(NuclearNorm(0.5), X2, 1.) rtol = 1e-5
  @test_throws ArgumentError SingularValue(LogisticLoss(rand(0:1, 5), randn(5, 2)))
  @test_throws DimensionMismatch prox(SingularValue(NormL1(1., rand(3))), randn(6, 4), 1.)
end

@testset "Float32" begin
  X = randn(Float32, 7, 5)
  for f in (NormL1(0.5), NormLinf(0.5), IndBallL1(1.), MCP(0.4))
    F = SingularValue(f)
    @test F(X) isa Float32
    @test @inferred(value_and_prox!(similar(X), F, X, 0.5f0)) isa Float32
    @test prox(F, X, 0.5) ≈ sv_prox(f, Float64.(X), 0.5) rtol = 1e-4 atol = 1e-5
  end
end

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "zero allocations" begin
  M = randn(12, 8)
  for f in CATALOG, Z in (M, Float32.(M))
    F = SingularValue(f)
    Y = similar(Z)
    @test allocs(a_value, F, Z) == 0
    @test allocs(a_prox!, Y, F, Z, 0.7) == 0
    @test allocs(a_vprox!, Y, F, Z, 0.7) == 0
  end
  C = Conjugate(SingularValue(NormLinf(0.5)))
  @test allocs(a_value, C, M) == 0
  @test allocs(a_prox!, similar(M), C, M, 0.7) == 0
end

end
