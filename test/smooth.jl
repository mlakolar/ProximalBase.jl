module SmoothTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

@testset "ZeroFunction and LinearFunction" begin
  x = randn(5)
  f = ZeroFunction()
  @test f(x) === 0.0
  @test prox(f, x, 3.) == x
  @test gradient(f, x) == zeros(5)
  check_prox(f, x, 0.5)
  @test restrict(f, [1, 2]) === f

  c = randn(5)
  f = LinearFunction(c, 2.)
  @test f(x) ≈ dot(c, x) + 2
  @test prox(f, x, 0.5) ≈ x - 0.5c
  @test gradient(f, x) == c
  check_gradient(f, x)
  check_prox(f, x, 0.5)
  @test LinearFunction(c)(x) ≈ dot(c, x)
  @test f(SparseIterate(x)) ≈ f(x)
end

@testset "SqrNormL2" begin
  x = [1., 2.]
  @test SqrNormL2(0.)(x) == 0.
  @test SqrNormL2()(x) == 2.5
  @test SqrNormL2(4.)(x) ≈ 2 * norm(x)^2
  @test prox(SqrNormL2(0.), x) == x
  @test prox(SqrNormL2(1.), x) ≈ x / 2
  @test prox(SqrNormL2(4.), x, 0.5) ≈ x / 3
  y = [3., -1.]
  f = SqrNormL2(2., y)
  @test f(x) ≈ norm(x - y)^2
  @test prox(f, x, 0.5) ≈ (x + y) / 2
  @test gradient(f, x) ≈ 2 * (x - y)
  for f in (SqrNormL2(1.5), SqrNormL2(0.5, randn(6)))
    x = randn(6)
    check_gradient(f, x)
    check_prox(f, x, 0.8)
  end
  # sparse iterates take O(nnz) sums
  xs = SparseIterate(sprandn(50, 0.1))
  @test SqrNormL2(3.)(xs) ≈ SqrNormL2(3.)(Vector(xs))
  D = randn(4, 4); D = D + D'
  @test SqrNormL2(3.)(SymmetricSparseIterate(D)) ≈ SqrNormL2(3.)(D)
  @test_throws ArgumentError SqrNormL2(-1.)
  @test SqrNormL2(1f0)(Float32[1, 2]) isa Float32
end

@testset "QuadraticFunction" begin
  p = 8
  B = randn(2p, p)
  A = B'B / 2p
  b, c = randn(p), 1.5
  x = randn(p)
  f = QuadraticFunction(A, b, c)
  @test f(x) ≈ dot(x, A * x) / 2 + dot(b, x) + c
  @test QuadraticFunction(A)(x) ≈ dot(x, A * x) / 2
  @test QuadraticFunction(A, b)(x) ≈ dot(x, A * x) / 2 + dot(b, x)
  check_gradient(f, x)
  # the proximal operator solves (I + γA) z = x - γb
  for γ in (0.1, 1., 10.)
    z = check_prox(f, x, γ)
    @test (I + γ * A) * z ≈ x - γ * b
  end
  # sparse iterates: the product with A is O(p ⋅ nnz)
  xs = SparseIterate(sprandn(p, 0.4))
  @test f(xs) ≈ f(Vector(xs))
  @test gradient(f, xs) ≈ gradient(f, Vector(xs))
  @test_throws DimensionMismatch QuadraticFunction(A, randn(p + 1))
  @test_throws DimensionMismatch QuadraticFunction(randn(3, 4))
  f32 = QuadraticFunction(Float32.(A), Float32.(b))
  @test f32(Float32.(x)) isa Float32
end

@testset "LeastSquaresLoss" begin
  for (n, p) in ((30, 10), (10, 30))         # p > n: the proximal operator still works
    X, y = randn(n, p), randn(n)
    f = LeastSquaresLoss(y, X)
    β = randn(p)
    @test f(β) ≈ norm(y - X * β)^2 / 2
    check_gradient(f, β)
    @test gradient(f, β) ≈ X' * (X * β - y)
    for γ in (0.01, 1., 100.)
      z = check_prox(f, β, γ)
      @test (X'X + I / γ) * z ≈ X'y + β / γ
    end
    βs = SparseIterate(sprandn(p, 0.3))
    @test f(βs) ≈ f(Vector(βs))
    @test gradient(f, βs) ≈ gradient(f, Vector(βs))
    βa = AtomIterate(randn(p), p ÷ 2)
    @test f(βa) ≈ f(Vector(βa))
  end
  # a matrix response: multi-task regression
  X, Y = randn(20, 6), randn(20, 3)
  f = LeastSquaresLoss(Y, X)
  B = randn(6, 3)
  @test f(B) ≈ norm(Y - X * B)^2 / 2
  check_gradient(f, B)
  z = check_prox(f, B, 0.5)
  @test (X'X + 2I) * z ≈ X'Y + 2B
  Bs = SparseIterate(sprandn(6, 3, 0.5))
  @test f(Bs) ≈ f(Matrix(Bs))
  @test_throws DimensionMismatch LeastSquaresLoss(randn(5), randn(4, 2))
  # a prox into an output that is not strided
  zs = SparseIterate(6)
  f = LeastSquaresLoss(randn(20), X)
  β = randn(6)
  @test Vector(prox!(zs, f, β, 0.3)) ≈ prox(f, β, 0.3)
end

@testset "scalar losses" begin
  for ℓ in (PB.Squared(), PB.Logistic(), PB.Poisson(), PB.Huber(0.7)), η in (-3., -0.5, 0.2, 2.5), y in (0., 1., 2.)
    h = 1e-6
    @test PB.deriv(ℓ, η, y) ≈ (ℓ(η + h, y) - ℓ(η - h, y)) / 2h rtol = 1e-6 atol = 1e-8
    if !(ℓ isa PB.Huber && abs(abs(η - y) - 0.7) < 1e-3)
      @test PB.deriv2(ℓ, η, y) ≈ (PB.deriv(ℓ, η + h, y) - PB.deriv(ℓ, η - h, y)) / 2h rtol = 1e-5 atol = 1e-7
    end
    @test PB.deriv2(ℓ, η, y) <= PB.curvature(ℓ)
    v, d = PB.value_and_deriv(ℓ, η, y)
    @test v ≈ ℓ(η, y) && d ≈ PB.deriv(ℓ, η, y)
  end
  @test PB.curvature(PB.Logistic()) == 0.25 && PB.curvature(PB.Poisson()) == Inf
  # the logistic loss does not overflow
  @test PB.Logistic()(1000., 1.) ≈ 0. atol = 1e-12
  @test PB.Logistic()(-1000., 0.) ≈ 0. atol = 1e-12
  @test PB.Logistic()(1000., 0.) ≈ 1000.
  @test PB.deriv(PB.Logistic(), 1000., 0.) ≈ 1.
  @test PB.Logistic()(0.3f0, 1f0) isa Float32
  @test_throws ArgumentError PB.Huber(0.)
end

@testset "GLMLoss" begin
  n, p = 40, 6
  X = randn(n, p)
  β = 0.3 .* randn(p)
  η = X * β
  yb = Float64.(rand(n) .< 1 ./ (1 .+ exp.(-η)))
  yc = Float64.(rand(0:4, n))
  yr = η .+ randn(n)
  w = rand(n)
  off = 0.1 .* randn(n)
  # the losses are what they say
  @test LogisticLoss(yb, X)(β) ≈ sum(log1p.(exp.(η)) .- yb .* η)
  @test PoissonLoss(yc, X)(β) ≈ sum(exp.(η) .- yc .* η)
  r = η .- yr
  @test HuberLoss(yr, X, 0.5)(β) ≈ sum(ifelse.(abs.(r) .<= 0.5, r .^ 2 ./ 2, 0.5 .* (abs.(r) .- 0.25)))
  @test GLMLoss(PB.Squared(), yr, X)(β) ≈ norm(η - yr)^2 / 2
  @test LogisticLoss(yb, X; weights = w)(β) ≈ sum(w .* (log1p.(exp.(η)) .- yb .* η))
  @test LogisticLoss(yb, X; offset = off)(β) ≈ sum(log1p.(exp.(η .+ off)) .- yb .* (η .+ off))
  for f in (LogisticLoss(yb, X), PoissonLoss(yc, X), HuberLoss(yr, X, 0.5), GLMLoss(PB.Squared(), yr, X),
            LogisticLoss(yb, X; weights = w, offset = off))
    @testset "$(typeof(f.loss))" begin
      check_gradient(f, β)
      βs = SparseIterate(sprandn(p, 0.5))
      @test f(βs) ≈ f(Vector(βs))
      @test gradient(f, βs) ≈ gradient(f, Vector(βs))
      @test f isa GLMLoss
    end
  end
  @test LogisticLoss(yb, X) isa LogisticLoss
  @test HuberLoss(yr, X, 1.) isa HuberLoss
  @test !(PoissonLoss(yc, X) isa LogisticLoss)
  # squared loss: the least-squares loss
  @test gradient(GLMLoss(PB.Squared(), yr, X), β) ≈ gradient(LeastSquaresLoss(yr, X), β)
  @test_throws DimensionMismatch LogisticLoss(yb[1:5], X)
  @test_throws DimensionMismatch LogisticLoss(yb, X; weights = w[1:5])
  @test_throws ArgumentError LogisticLoss(yb, X; weights = -w)
  @test_throws DimensionMismatch LogisticLoss(yb, X)(randn(p + 1))
  f32 = LogisticLoss(Float32.(yb), Float32.(X))
  @test f32(Float32.(β)) isa Float32
end

@testset "LogDetLoss" begin
  @testset "identity" begin
    p = 10
    f = LogDetLoss(Matrix(1.0I, p, p))
    V = Matrix(1.0I, p, p)
    @test prox(f, V) ≈ V
    @test f(V) ≈ p
    γ, ρ = 0.5, 2.
    sol = Matrix(1.0I, p, p) * (-(1 - ρ) + sqrt((1 - ρ)^2 + 4ρ)) / (2ρ)
    @test prox(f, V, γ) ≈ sol
  end

  p = 15
  A = randn(2p, p)
  S = Symmetric(A'A / 2p)
  B = randn(2p, p)
  X = B'B / 2p + I
  for w in (1., 0.3)
    f = LogDetLoss(S, w)
    @test f(X) ≈ w * (tr(S * X) - logdet(X))
    # the proximal operator satisfies w(S - Z⁻¹) + (Z - X)/γ = 0
    for γ in (0.1, 0.7, 5.)
      Z = prox(f, X, γ)
      @test w * (S - inv(Z)) + (Z - X) / γ ≈ zeros(p, p) atol = 1e-8
      Y = similar(Z)
      @test value_and_prox!(Y, f, X, γ) ≈ f(Z)
      @test Y ≈ Z
    end
    check_prox(f, X, 0.7; symmetric = true)
    # smooth on the positive definite matrices
    G = gradient(f, X)
    @test G ≈ w * (S - inv(X))
    @test G == G'
    h = 1e-6
    D = randn(p, p); D = D + D'
    @test (f(X + h * D) - f(X - h * D)) / 2h ≈ dot(G, D) rtol = 1e-5
  end
  # a very negative eigenvalue of ρX - S: no cancellation, a positive definite result
  L = LogDetLoss(1e9 * Matrix(1.0I, 2, 2))
  Y = similar(zeros(2, 2))
  v = value_and_prox!(Y, L, zeros(2, 2), 1.0)
  @test isposdef(Y)
  @test Y ≈ 1e-9 * I rtol = 1e-6
  @test isfinite(v) && v ≈ L(Y)
  L2 = LogDetLoss(S)
  Z = prox(L2, X, 1e6)                                  # a large step
  @test isposdef(Z)
  @test S - inv(Z) + (Z - X) / 1e6 ≈ zeros(p, p) atol = 1e-6
  f = LogDetLoss(S)
  # outside the cone the value is Inf, without an error, even when the determinant is positive
  @test f(-Matrix(1.0I, p, p)) == Inf
  @test f(Diagonal([-1., -2., ones(p - 2)...]) |> Matrix) == Inf
  G = zeros(p, p)
  @test value_and_gradient!(G, f, -Matrix(1.0I, p, p)) == Inf
  @test all(isnan, G)
  # S may be given as a Symmetric or a full matrix; a SymmetricSparseIterate argument works
  @test LogDetLoss(Matrix(S))(X) ≈ f(X)
  Xs = SymmetricSparseIterate(sparse(Matrix(I + 0.1 * Symmetric(sprandn(p, p, 0.2)))))
  @test f(Xs) ≈ f(Matrix(Xs))
  @test prox(f, Matrix(Xs), 0.5) ≈ prox!(zeros(p, p), f, Xs, 0.5)
  @test_throws DimensionMismatch f(randn(3, 3))
  @test_throws ArgumentError LogDetLoss(big.([2. 1.; 1. 2.]))
  f32 = LogDetLoss(Float32.(S))
  @test f32(Float32.(X)) isa Float32
  @test prox(f32, Float32.(X), 0.5f0) ≈ prox(f, X, 0.5) rtol = 1e-4
end

end
