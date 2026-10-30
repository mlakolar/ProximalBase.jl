module SmoothTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

# Count objective evaluations independently of derivative evaluations.
struct CountedSquared <: PB.ScalarLoss
  calls::Base.RefValue{Int}
end
(ℓ::CountedSquared)(η, y) = (ℓ.calls[] += 1; abs2(η - y) / 2)
PB.deriv(::CountedSquared, η, y) = η - y

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

@testset "KronQuadratic" begin
  p, q = 6, 4
  X, Y = randn(30, p), randn(30, q)
  A, B = Matrix(Symmetric(X'X / 30)), Matrix(Symmetric(Y'Y / 30))
  C = randn(p, q)
  f = KronQuadratic(A, B, C)
  check_traits(f; smooth = true, convex = true)
  @test size(f) == (p, q)
  # vec(Θ)'(B ⊗ A)vec(Θ)/2 + ⟨C, Θ⟩, with gradient AΘB + C
  K = kron(B, A)
  Θ = randn(p, q)
  @test f(Θ) ≈ dot(vec(Θ), K * vec(Θ)) / 2 + dot(C, Θ)
  @test KronQuadratic(A, B)(Θ) ≈ dot(vec(Θ), K * vec(Θ)) / 2
  @test gradient(f, Θ) ≈ A * Θ * B + C
  check_gradient(f, Θ)
  # vec(Θ) and sparse iterates of either shape
  @test f(vec(Θ)) ≈ f(Θ) && gradient(f, vec(Θ)) ≈ vec(A * Θ * B + C)
  Θs = SparseIterate(sprandn(p, q, 0.4))
  @test f(Θs) ≈ f(Matrix(Θs)) && gradient(f, Θs) ≈ gradient(f, Matrix(Θs))
  θs = SparseIterate(sprandn(p * q, 0.4))
  @test f(θs) ≈ f(Vector(θs)) && gradient(f, θs) ≈ gradient(f, Vector(θs))
  # on a SymmetricSparseIterate, the gradient over symmetric matrices
  Z = randn(30, p)
  Σy = Matrix(Symmetric(Z'Z / 30))
  Cs = randn(p, p)
  g = KronQuadratic(A, Σy, Cs)
  S = SymmetricSparseIterate(Matrix(Symmetric(randn(p, p))))
  D = Matrix(S)
  @test g(S) ≈ g(D)
  @test gradient(g, S) ≈ (A * D * Σy + Σy * D * A) / 2 + (Cs + Cs') / 2
  @test value_and_gradient(g, S)[1] ≈ g(D)
  @test gradient(g, D) ≈ A * D * Σy + Cs          # a dense matrix: over all matrices
  @test Matrix(gradient!(similar(S), g, S)) ≈ gradient(g, S)
  # CovSel's D-trace loss and DiffPrecTest's loss
  @test KronQuadratic(A, Σy, A - Σy)(D) ≈ tr(A * D * Σy * D) / 2 - tr(D * (Σy - A))
  E = zeros(p, q)
  E[2, 3] = 1
  fE = KronQuadratic(A, B, -E)
  @test fE(Θ) ≈ tr(A * Θ * B * Θ') / 2 - Θ[2, 3]
  # whose minimizer is the column (b - 1)p + a of (B ⊗ A)⁻¹
  @test isapprox(gradient(fE, reshape(inv(K)[:, (3 - 1) * p + 2], p, q)), zeros(p, q); atol = 1e-10)
  # A and B are stored as dense matrices, whatever their type
  h = KronQuadratic(Symmetric(A), sparse(B))
  @test h.A isa Matrix{Float64} && h.B isa Matrix{Float64} && h.C === nothing
  @test h(Θ) ≈ KronQuadratic(A, B)(Θ)
  @test KronQuadratic(Float32.(A), Float32.(B))(Float32.(Θ)) isa Float32
  # shapes and arguments
  @test_throws DimensionMismatch f(randn(q, p))             # as many entries, the wrong shape
  @test_throws DimensionMismatch f(randn(p * q + 1))
  @test_throws DimensionMismatch gradient!(zeros(q, p), f, Θ)
  @test_throws DimensionMismatch g(SymmetricSparseIterate(p + 1))
  @test_throws DimensionMismatch KronQuadratic(A, B, randn(q, p))
  @test_throws DimensionMismatch KronQuadratic(randn(3, 4), B)
  @test_throws ArgumentError KronQuadratic(randn(p, p), B)   # not symmetric
  @test_throws ArgumentError KronQuadratic(A, B, fill(NaN, p, q))
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

  @testset "large steps and reusable output buffers" begin
    for T in (Float32, Float64), (n, p) in ((30, 10), (10, 30)), m in (1, 3)
      X = randn(T, n, p)
      Y = m == 1 ? randn(T, n) : randn(T, n, m)
      v = m == 1 ? randn(T, p) : randn(T, p, m)
      f = LeastSquaresLoss(Y, X)
      X64, Y64, v64 = Float64.(X), Float64.(Y), Float64.(v)
      for γ in T.((0.01, 0.7, T === Float32 ? 1e6 : 1e12))
        ρ = inv(Float64(γ))
        ref = n >= p ? (X64'X64 + ρ * I) \ (X64'Y64 + ρ * v64) :
                       v64 + X64' * ((X64 * X64' + ρ * I) \ (Y64 - X64 * v64))
        tol = T === Float32 ? 2e-5 : 2e-12
        @test prox(f, v, γ) ≈ ref rtol = tol
        alias = copy(v)
        @test prox!(alias, f, alias, γ) ≈ ref rtol = tol
        sparseout = SparseIterate(zeros(T, size(v)))
        @test Array(prox!(sparseout, f, v, γ)) ≈ ref rtol = tol
        @test allocs(prox!, sparseout, f, v, γ) == 0
        if m == 1
          atomout = AtomIterate(zeros(T, p), [1:p])
          @test Vector(prox!(atomout, f, v, γ)) ≈ ref rtol = tol
          @test allocs(prox!, atomout, f, v, γ) == 0
        end
      end
      @test_throws DimensionMismatch prox!(zeros(T, p, m + 1), f, zeros(T, p, m + 1), 0.7)
      @test_throws DimensionMismatch prox!(zeros(T, p, 1), LeastSquaresLoss(randn(T, n), X), zeros(T, p, 1), 0.7)
    end
    # Tiny steps can overflow the reciprocal or its product with V'v.
    for T in (Float32, Float64), (n, p) in ((10, 10), (30, 10), (10, 30)), m in (1, 3)
      X = randn(T, n, p)
      Y = m == 1 ? randn(T, n) : randn(T, n, m)
      v = T(10) .* (m == 1 ? randn(T, p) : randn(T, p, m))
      f = LeastSquaresLoss(Y, X)
      for γ in (floatmin(T), floatmin(T) / T(8), nextfloat(zero(T)))
        @test prox(f, v, γ) ≈ v rtol = 20eps(T)
        alias = copy(v)
        @test prox!(alias, f, alias, γ) ≈ v rtol = 20eps(T)
        @test allocs(prox!, alias, f, v, γ) == 0
        sparseout = SparseIterate(zeros(T, size(v)))
        @test Array(prox!(sparseout, f, v, γ)) ≈ v rtol = 20eps(T)
        @test allocs(prox!, sparseout, f, v, γ) == 0
      end
    end
    # An empty design has a constant loss and the identity proximal operator.
    for (n, p) in ((0, 4), (4, 0), (0, 0)), m in (1, 3)
      X = zeros(n, p)
      Y = m == 1 ? zeros(n) : zeros(n, m)
      v = m == 1 ? randn(p) : randn(p, m)
      f = LeastSquaresLoss(Y, X)
      out = similar(v)
      @test prox!(out, f, v, 0.7) == v
      @test allocs(prox!, out, f, v, 0.7) == 0
    end
  end
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
  # Correctly classified observations still have a positive loss and curvature
  # after sigmoid(η) has rounded to 1; compare to independent high-precision formulas.
  setprecision(BigFloat, 512) do
    for T in (Float32, Float64), η in T.((-80, -40, -20, 0, 20, 40, 80)), y in T.((0, 0.3, 1))
      bη, by = BigFloat(η), BigFloat(y)
      e = exp(bη)
      refv = T(log1p(e) - by * bη)
      refd = T(e / (1 + e) - by)
      refh = T(e / (1 + e)^2)
      v, d = PB.value_and_deriv(PB.Logistic(), η, y)
      @test v ≈ refv rtol = 8eps(T)
      @test d ≈ refd rtol = 8eps(T)
      @test PB.deriv(PB.Logistic(), η, y) ≈ refd rtol = 8eps(T)
      @test PB.deriv2(PB.Logistic(), η, y) ≈ refh rtol = 8eps(T)
    end
    for (η, y) in ((40., 0.3f0), (40f0, 0.3), (40f0, 1), (40, 1f0))
      bη, by = BigFloat(η), BigFloat(y)
      T = typeof(first(promote(float(η), y)))
      refv = T(log1p(exp(bη)) - by * bη)
      refd = T(inv(1 + exp(-bη)) - by)
      v, d = PB.value_and_deriv(PB.Logistic(), η, y)
      @test v ≈ refv rtol = 8eps(T)
      @test d ≈ refd rtol = 8eps(T)
      @test PB.deriv(PB.Logistic(), η, y) ≈ refd rtol = 8eps(T)
    end
  end
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
  # observation weights and offsets must be finite in the working precision (#251)
  wmsg, omsg = "the weights must be finite and nonnegative", "the offset must be finite"
  for bad in ([Inf, 1.0], [1.0, NaN], [1.0, -Inf]), loss in (PB.Logistic(), PB.Squared(), PB.Poisson(), PB.Huber(1.0))
    @test_throws ArgumentError(wmsg) GLMLoss(loss, Float64[0, 1], [1.0 0.0; 0.0 1.0]; weights = bad)
  end
  @test_throws ArgumentError(wmsg) LogisticLoss(yb, X; weights = [w[1:end-1]; Inf])
  @test_throws ArgumentError(wmsg) HuberLoss(yr, X, 1.0; weights = [w[1:end-1]; Inf])
  @test_throws ArgumentError(wmsg) LogisticLoss(yb, X; weights = complex.(w))
  for bad in ([Inf, 0.0], [0.0, NaN], [-Inf, 0.0])
    @test_throws ArgumentError(omsg) PoissonLoss(Float64[0, 1], [1.0 0.0; 0.0 1.0]; offset = bad)
  end
  # a weight or offset finite in Float64 overflows Float32 data
  y32, X32 = Float32[1, 0], Float32[1 0; 0 1]
  @test_throws ArgumentError(wmsg) LogisticLoss(y32, X32; weights = [1.0, 1e39])
  @test_throws ArgumentError(omsg) LogisticLoss(y32, X32; offset = [0.0, 1e39])
  @test LogisticLoss(y32, X32; weights = [1.0, 1e38], offset = [0.0, -1e38]) isa GLMLoss
  # zero weights, -0.0 included, are allowed, and restrict rebuilds through the checks
  fz = LogisticLoss(yb, X; weights = [zeros(n - 1); -0.0])
  @test fz(β) == 0 && iszero(gradient(fz, β))
  @test restrict(LogisticLoss(yb, X; weights = w, offset = w), 1:2)(β[1:2]) ≈
    LogisticLoss(yb, X[:, 1:2]; weights = w, offset = w)(β[1:2])
  @test_throws DimensionMismatch LogisticLoss(yb, X)(randn(p + 1))
  f32 = LogisticLoss(Float32.(yb), Float32.(X))
  @test f32(Float32.(β)) isa Float32

  calls = Ref(0)
  f = GLMLoss(CountedSquared(calls), yr, X; weights = w, offset = off)
  g = similar(β)
  @test gradient!(g, f, β) === g
  @test g ≈ X' * (w .* (X * β + off - yr))
  @test calls[] == 0
  @test value_and_gradient!(g, f, β) ≈ sum(w .* abs2.(X * β + off - yr)) / 2
  @test calls[] == n

  # The loss, gradient and Hessian-vector product retain the logistic tail too.
  for T in (Float32, Float64)
    f = LogisticLoss(T[1], ones(T, 1, 1))
    β = T[40]
    e = exp(T(-40))
    @test f(β) ≈ log1p(e)
    @test gradient(f, β) ≈ T[-e / (1 + e)]
    @test hessian_vec(f, β, T[1]) ≈ T[e / (1 + e)^2]
  end
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
  for T in (Float32, Float64)
    tiny = T == Float32 ? T(1e-20) : T(1e-200)
    fsmall = LogDetLoss(zeros(T, 2, 2), one(T))
    Xsmall = Matrix{T}(I, 2, 2)
    Ysmall = similar(Xsmall)
    value = value_and_prox!(Ysmall, fsmall, Xsmall, tiny)
    @test all(isfinite, Ysmall) && isfinite(value)
    @test Ysmall ≈ Xsmall
    @test value ≈ fsmall(Ysmall)
    alias = copy(Xsmall)
    @test prox!(alias, fsmall, alias, tiny) ≈ Xsmall
    extreme = T == Float32 ? T(1e30) : T(1e200)
    for scale in (extreme, inv(extreme))
      @test prox(LogDetLoss(zeros(T, 1, 1), scale), zeros(T, 1, 1), scale) ≈ fill(scale, 1, 1) rtol=16eps(T)
    end
    @test prox(LogDetLoss(fill(extreme, 1, 1)), ones(T, 1, 1), extreme) ≈ fill(inv(extreme), 1, 1) rtol=16eps(T)
    # The subtraction X - γS can overflow even with a unit step.
    z = only(prox(LogDetLoss(fill(floatmax(T), 1, 1)), fill(-floatmax(T), 1, 1)))
    @test isfinite(z) && z > 0
    @test z ≈ T(inv(2BigFloat(floatmax(T)))) rtol=32eps(T)
  end
  @test f32(Float32.(X)) isa Float32
  @test prox(f32, Float32.(X), 0.5f0) ≈ prox(f, X, 0.5) rtol = 1e-4

  @testset "precision" begin
    for Td in (Float32, Float64), Tx in (Float32, Float64)
      f = LogDetLoss(Td[2 0.5 0.1; 0.5 3 0; 0.1 0 1], one(Td))
      X = Tx[1.1 0.2 0; 0.2 2.1 0.3; 0 0.3 1.5]
      original = copy(X)
      finite = @inferred f(X)
      @test isfinite(finite)
      for bad in (-X, zeros(Tx, 3, 3), Tx[1 1 0; 0 1 0; 0 0 1], fill(Tx(Inf), 3, 3), fill(Tx(NaN), 3, 3))
        @test (@inferred f(bad)) === oftype(finite, Inf)
      end
      for γ in (Tx(0.25), one(Tx), Tx(4))
        Y_data = zeros(Td, 3, 3)
        v_data = value_and_prox!(Y_data, f, X, γ)
        for Ty in (Float32, Float64)
          expected = Ty.(Y_data)
          tol = 32max(eps(Td), eps(Ty))
          Y = zeros(Ty, 3, 3)
          @test prox!(Y, f, X, γ) === Y
          @test Y == expected
          @test value_and_prox!(Y, f, X, γ) ≈ v_data rtol = tol
          @test Y == expected
          @test allocs(prox!, Y, f, X, γ) == 0
          @test allocs(value_and_prox!, Y, f, X, γ) == 0

          storage = fill(Ty(-7), 3, 5)
          output = view(storage, :, 2:4)
          @test prox!(output, f, X, γ) === output
          @test output == expected
          @test value_and_prox!(output, f, X, γ) ≈ v_data rtol = tol
          @test output == expected
          @test all(==(-7), storage[:, [1, 5]])

          blocks = SeparableSum(f, ZeroFunction())
          input = (X, Tx[1, -2])
          output = (Y, zeros(Ty, 2))
          @test prox!(output, blocks, input, γ) === output
          @test Y == expected && output[2] == input[2]
          @test value_and_prox!(output, blocks, input, γ) ≈ v_data rtol = tol
          @test Y == expected && output[2] == input[2]

          # Moreau's identity uses the conjugate output's precision for its buffer.
          c = Conjugate(f)
          step = Ty(γ)
          P = Ty.(X ./ step)
          fp = value_and_prox!(P, f, P, inv(step))
          conjugate_expected = Ty.(X .- step .* P)
          conjugate_value = dot(P, conjugate_expected) - fp
          for output in (Y, SymmetricSparseIterate(zeros(Ty, 3, 3)))
            @test prox!(output, c, X, γ) === output
            @test Matrix(output) == conjugate_expected
            @test value_and_prox!(output, c, X, γ) ≈ conjugate_value rtol = tol
            @test Matrix(output) == conjugate_expected
          end
          if Ty == Tx
            alias = copy(X)
            @test prox!(alias, f, alias, γ) === alias
            @test alias == expected
            copyto!(alias, X)
            @test value_and_prox!(alias, f, alias, γ) ≈ v_data rtol = tol
            @test alias == expected
          end
          @test X == original
        end
        @test prox(f, X, γ) == Tx.(Y_data)
        @test X == original
      end
    end
  end
end

@testset "zero-weight observations are ignored" begin
  for T in (Float32, Float64)
    f = PoissonLoss(zeros(T, 2), ones(T, 2, 1); weights=T[0, 1], offset=T[1000, 0])
    for β in (T[0], T[-2], T[2])
      expected = exp(β[1])
      @test f(β) ≈ expected
      @test gradient(f, β) ≈ T[expected]
      g = similar(β)
      @test value_and_gradient!(g, f, β) ≈ expected
      @test g ≈ T[expected]
      @test hessian_vec(f, β, T[2]) ≈ T[2expected]
      q = quadratic_model(f, β)
      @test all(isfinite, q.y) && all(isfinite, q.weights)
      @test gradient(q, β) ≈ T[expected]
      @test hessian_vec(q, β, T[2]) ≈ T[2expected]
    end
    fzero = PoissonLoss(zeros(T, 2), ones(T, 2, 1); weights=zeros(T, 2), offset=fill(T(1000), 2))
    @test fzero(T[0]) == 0
    @test gradient(fzero, T[0]) == T[0]
    @test hessian_vec(fzero, T[0], T[2]) == T[0]
  end
end

@testset "lipschitz" begin
  n, p = 40, 15
  X, y = randn(n, p), randn(n)
  L = opnorm(X)^2
  # closed forms, and the largest eigenvalues the losses involve
  @test lipschitz(ZeroFunction()) == 0 && lipschitz(LinearFunction(randn(p))) == 0
  @test lipschitz(SqrNormL2(3.)) == 3 && lipschitz(SqrNormL2(3., randn(p))) == 3
  @test lipschitz(LeastSquaresLoss(y, X)) ≈ L rtol = 1e-8
  @test lipschitz(LeastSquaresLoss(randn(n, 3), X)) ≈ L rtol = 1e-8
  Xw = randn(10, 60)                                  # more columns than rows
  @test lipschitz(LeastSquaresLoss(randn(10), Xw)) ≈ opnorm(Xw)^2 rtol = 1e-8
  wt = rand(n)
  @test lipschitz(LogisticLoss(rand(0:1, n), X)) ≈ L / 4 rtol = 1e-8
  @test lipschitz(LogisticLoss(rand(0:1, n), X; weights = wt)) ≈ opnorm(Diagonal(sqrt.(wt)) * X)^2 / 4 rtol = 1e-8
  @test lipschitz(HuberLoss(y, X, 0.5)) ≈ L rtol = 1e-8
  @test lipschitz(GLMLoss(ProximalBase.Squared(), y, X)) ≈ L rtol = 1e-8
  @test lipschitz(PoissonLoss(rand(0:3, n), X)) == Inf
  A = X'X / n
  @test lipschitz(QuadraticFunction(A, randn(p))) ≈ opnorm(A) rtol = 1e-8
  B = A[1:5, 1:5]
  @test lipschitz(KronQuadratic(A, B, randn(p, 5))) ≈ opnorm(A) * opnorm(B) rtol = 1e-8
  @test lipschitz(LogDetLoss(Matrix(1.0I, 3, 3))) == Inf
  # functions built from functions
  f = LeastSquaresLoss(y, X)
  @test lipschitz(2 * f) ≈ 2L rtol = 1e-8
  @test lipschitz(Tilt(f, randn(p), 1.)) ≈ L rtol = 1e-8
  @test lipschitz(Translate(f, randn(p))) ≈ L rtol = 1e-8
  @test lipschitz(Sum(f, SqrNormL2(3.))) ≈ L + 3 rtol = 1e-8
  @test lipschitz(SeparableSum(f, SqrNormL2(3.))) ≈ max(L, 3) rtol = 1e-8
  # the step 1/L decreases the objective along the gradient
  β = randn(p)
  @test f(β - gradient(f, β) / lipschitz(f)) < f(β)
  # edge cases and errors
  @test lipschitz(LeastSquaresLoss(zeros(n), zeros(n, p))) == 0
  @test lipschitz(LeastSquaresLoss(Float32.(y), Float32.(X))) isa Float32
  @test_throws ArgumentError lipschitz(NormL1(1.))
  @test_throws ArgumentError lipschitz(Conjugate(NormL2(1.)))
  # A fixed Lanczos start can be orthogonal to the leading eigenspace. Bounds
  # used for a fixed gradient step must still account for that direction.
  for T in (Float32, Float64)
    start = T[sin(i) + cos(sqrt(2) * i) for i in 1:3]
    u = normalize(T[start[2], -start[1], 0])
    A = Matrix{T}(I, 3, 3) + T(99) * (u * u')
    for matrix in (A, sparse(A))
      q = QuadraticFunction(matrix)
      Lq = lipschitz(q)
      @test Lq >= T(100) * (1 - 16eps(T))
      @test q(u - gradient(q, u) / Lq) < q(u)
      ls = LeastSquaresLoss(zeros(T, 3), matrix)
      Ll = lipschitz(ls)
      @test Ll >= T(10000) * (1 - 32eps(T))
      @test ls(u - gradient(ls, u) / Ll) < ls(u)
      logistic = LogisticLoss(zeros(T, 3), matrix; weights=T[0.5, 1, 2])
      reference = opnorm(Diagonal(sqrt.(T[0.5, 1, 2])) * A)^2 / 4
      @test lipschitz(logistic) >= reference * (1 - 32eps(T))
    end
    Xs = sprandn(T, 30, 15, 0.1)
    @test lipschitz(LeastSquaresLoss(zeros(T, 30), Xs)) >= opnorm(Matrix(Xs))^2 * (1 - 32eps(T))
    # Transposed inputs retain the full dense calculation; empty losses have
    # zero curvature on both the dense and sparse paths.
    rectangular = T[1 2 3; 4 5 6]
    for matrix in (transpose(rectangular), adjoint(rectangular))
      @test lipschitz(LeastSquaresLoss(zeros(T, 3), matrix)) ≈ opnorm(rectangular)^2
    end
    for matrix in (zeros(T, 0, 3), spzeros(T, 0, 3))
      @test lipschitz(LeastSquaresLoss(T[], matrix)) == zero(T)
    end
  end
end

@testset "KronQuadratic from matrices symmetric up to rounding" begin
  p = 6
  Ω = Matrix(SymTridiagonal(fill(1.0, p), fill(0.3, p - 1)))
  @test !issymmetric(inv(Ω))                          # an inverse computed by inv
  f = KronQuadratic(inv(Ω), inv(Ω))
  @test issymmetric(f.A) && f.A ≈ inv(Ω)
  @test_throws ArgumentError KronQuadratic(Ω + [i > j ? 1e-3 : 0. for i in 1:p, j in 1:p], Ω)
end

@testset "centered ridge at large steps" begin
  for T in (Float32, Float64)
    large = T == Float32 ? T(1e30) : T(1e200)
    center = T[2, -3]
    f = SqrNormL2(large, center)
    x = T[-1, 4]
    y = similar(x)
    @test prox!(y, f, x, large) === y
    @test y ≈ center rtol=8eps(T)
    @test prox_entry(f, x[1], 1, large) ≈ center[1]
    @test prox_entry(f, x[1], 1, large) isa T
    @test allocs(prox!, y, f, x, large) == 0
    # A convex combination of finite same-sign endpoints stays finite.
    nearmax = floatmax(T) / T(2)
    @test prox(SqrNormL2(one(T), T[nearmax]), T[nearmax], T(2)) ≈ T[nearmax]
  end
end


@testset "LogDetLoss symmetric domain and general proximal input" begin
  for T in (Float32, Float64)
    f = LogDetLoss(Matrix{T}(I, 2, 2))
    X = T[2 1; -1 3]
    G = similar(X)
    @test f(X) == T(Inf)
    @test value_and_gradient!(G, f, X) == T(Inf)
    @test all(isnan, G)
    for bad in (T(NaN), T(Inf))
      Z = Matrix{T}(I, 2, 2)
      Z[1,1] = bad
      @test f(Z) == T(Inf)
    end
    symmetric_part = (X + transpose(X)) / 2
    for γ in (T(0.3), T(3))
      expected = prox(f, symmetric_part, γ)
      Y = similar(X)
      @test value_and_prox!(Y, f, X, γ) ≈ f(expected) rtol=32eps(T)
      @test Y ≈ expected rtol=32eps(T)
      alias = copy(X)
      @test prox!(alias, f, alias, γ) ≈ expected rtol=32eps(T)
      @test allocs(prox!, Y, f, X, γ) == 0
    end
  end
end

@testset "ridge interpolation near floatmax" begin
  for T in (Float32, Float64)
    endpoint = floatmax(T)
    γ = T(2.9121568528227888e-6)
    f = SqrNormL2(one(T), T[endpoint])
    @test only(prox(f, T[endpoint], γ)) == endpoint
    @test prox_entry(f, endpoint, 1, γ) == endpoint
    @test prox(SqrNormL2(1.0, [2.0]), Float32[3], 0.2) ≈ Float32[17/6]
  end
end


@testset "least-squares rank-zero caches" begin
  for T in (Float32, Float64), (n,p) in ((0,4),(4,0),(0,0)), responses in (nothing,0,3)
    X = zeros(T,n,p)
    Y = responses === nothing ? ones(T,n) : ones(T,n,responses)
    x = responses === nothing ? ones(T,p) : ones(T,p,responses)
    f = LeastSquaresLoss(Y,X)
    @test ProximalBase._svd_cache!(f) === f
    @test size(f.V) == (p,0)
    @test isempty(f.s2)
    dims = responses === nothing ? (0,) : (0,responses)
    @test size(f.sUtY) == dims
    @test size(f.w) == dims
    @test f.svd_ready
    @test ProximalBase._xty_cache!(f) == transpose(X)*Y
    @test allocs(ProximalBase._svd_cache!, f) == 0
    @test allocs(ProximalBase._xty_cache!, f) == 0
    y,g = similar(x),similar(x)
    @test value_and_prox!(y,f,x,one(T)) ≈ sum(abs2,Y)/2
    @test y == x
    @test value_and_gradient!(g,f,x) ≈ sum(abs2,Y)/2
    @test all(iszero,g)
    @test allocs(prox!,y,f,x,one(T)) == 0
    reset_cache!(f)
    @test ProximalBase._svd_cache!(f) === f
    @test size(f.V) == (p,0)
  end
end

@testset "direct smooth parameter checks precede mutation" begin
  for T in (Float32, Float64), parameter in (T[1], T[1,2])
    for f in (LinearFunction(parameter), SqrNormL2(one(T),parameter), SqrNormL2(zero(T),parameter))
      x = T[3,4,5]
      out = fill(T(-7),3)
      @test_throws DimensionMismatch prox!(out,f,x,one(T))
      @test out == fill(T(-7),3)
      @test_throws DimensionMismatch gradient!(out,f,x)
      @test out == fill(T(-7),3)
      @test_throws DimensionMismatch value_and_gradient!(out,f,x)
      @test out == fill(T(-7),3)
      @test_throws DimensionMismatch f(x)
      alias = copy(x)
      @test_throws DimensionMismatch prox!(alias,f,alias,one(T))
      @test alias == x
    end
    # LinearFunction follows dot's linear-index convention for same-length arrays.
    c = reshape(T[1,2,3],1,3)
    f = LinearFunction(c)
    x = T[4,5,6]
    @test prox(f,x,T(0.5)) == x-T(0.5)*vec(c)
    @test gradient(f,x) == vec(c)
  end
end

@testset "centered ridge shape checks precede writes" begin
  for T in (Float32,Float64), λ in (zero(T),one(T))
    f = SqrNormL2(λ,reshape(T[1,2,3],1,3))
    x = T[4,5,6]
    out = fill(T(-7),3)
    @test_throws DimensionMismatch f(x)
    @test_throws DimensionMismatch prox!(out,f,x,one(T))
    @test out == fill(T(-7),3)
    @test_throws DimensionMismatch gradient!(out,f,x)
    @test out == fill(T(-7),3)
    @test_throws DimensionMismatch value_and_gradient!(out,f,x)
    @test out == fill(T(-7),3)
    # Coordinate kernels retain their established linear-entry convention.
    a = SparseIterate(x)
    @test prox_coordinate!(a,f,1,one(T)) == prox_entry(f,x[1],1,one(T))
  end
end

end
