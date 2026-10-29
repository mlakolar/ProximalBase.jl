module SecondOrderTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

# central differences of the gradient along v
fd_hessian_vec(f, x, v; h = 1e-6) = (gradient(f, x .+ h .* v) .- gradient(f, x .- h .* v)) ./ (2h)

# hessian_vec! against central differences of the gradient, its allocating
# version, and the symmetry of the Hessian; with symmetric = true the
# directions are symmetric matrices
function check_hessian(f, x; rtol = 1e-5, symmetric = false)
  sym(w) = symmetric ? (w .+ w') ./ 2 : w
  @test hashessian(f)
  @test hashessian(typeof(f))
  xd = Array(x)
  v, u = sym(randn(size(xd)...)), sym(randn(size(xd)...))
  Hv = hessian_vec(f, x, v)
  @test size(Hv) == size(x)
  Hv2 = similar(Hv)
  @test hessian_vec!(Hv2, f, x, v) === Hv2
  @test Hv2 ≈ Hv
  @test Hv ≈ fd_hessian_vec(f, xd, v) rtol = rtol atol = 1e-6 * max(1, norm(Hv))
  @test dot(u, Hv) ≈ dot(v, hessian_vec(f, x, u)) rtol = 1e-8
  Hv
end

# the model q of f at x has the gradient and the Hessian of f at x, and its
# change from x is the second-order expansion of f
function check_model(f, x; rtol = 1e-8, symmetric = false)
  sym(w) = symmetric ? (w .+ w') ./ 2 : w
  q = quadratic_model(f, x)
  @test issmooth(q) && hashessian(q)
  xd = Array(x)
  g = gradient(f, x)
  @test gradient(q, x) ≈ g rtol = rtol
  v = sym(randn(size(xd)...))
  @test hessian_vec(q, x, v) ≈ hessian_vec(f, x, v) rtol = rtol
  Δ = sym(randn(size(xd)...)) ./ 10
  @test q(xd .+ Δ) - q(xd) ≈ dot(g, Δ) + dot(Δ, hessian_vec(f, x, Δ)) / 2 rtol = sqrt(rtol)
  q
end

@testset "quadratics are their own models" begin
  n, p = 30, 8
  X, y, x = randn(n, p), randn(n), randn(p)
  A = Matrix(Symmetric(X'X / n))
  for f in (ZeroFunction(), LinearFunction(randn(p), 1.), SqrNormL2(2.), SqrNormL2(0.5, randn(p)),
            QuadraticFunction(A, randn(p), 3.), LeastSquaresLoss(y, X))
    @testset "$(nameof(typeof(f)))" begin
      Hv = check_hessian(f, x)
      @test quadratic_model(f, x) === f
      @test quadratic_model!(f, f, randn(p)) === f
      check_model(f, x)
    end
  end
  v = randn(p)
  @test hessian_vec(ZeroFunction(), x, v) == zeros(p)
  @test hessian_vec(SqrNormL2(2.), x, v) ≈ 2v
  @test hessian_vec(QuadraticFunction(A), x, v) ≈ A * v
  @test hessian_vec(LeastSquaresLoss(y, X), x, v) ≈ X' * (X * v)

  # a matrix response, and sparse points and directions
  Y = randn(n, 3)
  fm = LeastSquaresLoss(Y, X)
  check_hessian(fm, randn(p, 3))
  @test hessian_vec(fm, randn(p, 3), ones(p, 3)) ≈ X' * (X * ones(p, 3))
  xs, vs = SparseIterate(sprandn(p, 0.4)), SparseIterate(sprandn(p, 0.4))
  @test hessian_vec(LeastSquaresLoss(y, X), xs, vs) ≈ X' * (X * Vector(vs))
  @test hessian_vec(LeastSquaresLoss(y, X), x, AtomIterate(Vector(vs), [1:4, 5:8])) ≈ X' * (X * Vector(vs))

  @testset "KronQuadratic" begin
    B = randn(5, 5); Bq = Matrix(Symmetric(B'B / 5))
    C = randn(p, 5)
    f = KronQuadratic(A, Bq, C)
    Θ = randn(p, 5)
    Hv = check_hessian(f, Θ)
    V = randn(p, 5)
    @test hessian_vec(f, Θ, V) ≈ A * V * Bq
    @test hessian_vec(f, vec(Θ), vec(V)) ≈ vec(A * V * Bq)
    @test hessian_vec(f, SparseIterate(sparse(Θ)), V) ≈ A * V * Bq
    check_model(f, Θ)
    # over symmetric matrices the product is symmetrized, as the gradient is
    Bp = randn(p, p)
    fs = KronQuadratic(A, Bp'Bp / p, randn(p, p))
    Θs = SymmetricSparseIterate(Matrix(Symmetric(randn(p, p))))
    Vs = Matrix(Symmetric(randn(p, p)))
    Hs = hessian_vec(fs, Θs, Vs)
    @test Hs ≈ (fs.A * Vs * fs.B + fs.B * Vs * fs.A) / 2
    @test issymmetric(Hs)
  end
end

@testset "generalized linear models" begin
  n, p = 60, 6
  X = randn(n, p)
  x = randn(p) ./ 3
  w, o = rand(n) .+ 0.5, randn(n) ./ 4
  fs = [LogisticLoss(rand(0:1, n), X), LogisticLoss(rand(0:1, n), X; weights = w, offset = o),
        PoissonLoss(rand(0:4, n), X), PoissonLoss(rand(0:4, n), X; weights = w, offset = o),
        HuberLoss(randn(n), X, 0.5), GLMLoss(PB.Squared(), randn(n), X; weights = w)]
  for f in fs
    @testset "$(nameof(typeof(f.loss)))" begin
      check_hessian(f, x)
      # the Huber model has the curvature √eps where ℓ'' vanishes
      check_model(f, x; rtol = f.loss isa PB.Huber ? 1e-6 : 1e-8)
      check_hessian(f, SparseIterate(sparse(x)))
    end
  end

  @testset "the IRLS model" begin
    f = LogisticLoss(rand(0:1, n), X; weights = w, offset = o)
    q = quadratic_model(f, x)
    @test q isa GLMLoss{PB.Squared}
    η = X * x .+ o
    μ = 1 ./ (1 .+ exp.(-η))
    @test q.weights ≈ w .* μ .* (1 .- μ)
    @test q.y ≈ η .- (μ .- f.y) ./ (μ .* (1 .- μ))
    @test q.X === f.X && q.offset === f.offset
    # in place, at another point, the model at that point
    x2 = randn(p) ./ 3
    @test quadratic_model!(q, f, x2) === q
    q2 = quadratic_model(f, x2)
    @test q.y ≈ q2.y && q.weights ≈ q2.weights
    @test_throws ArgumentError quadratic_model!(quadratic_model(LogisticLoss(rand(0:1, n), randn(n, p)), x), f, x)
    # the gradient of the model is exact where the curvature vanishes
    fh = HuberLoss(10 .* randn(n), X, 0.1)
    @test gradient(quadratic_model(fh, x), x) ≈ gradient(fh, x)
  end

  @testset "Float32" begin
    f = LogisticLoss(Float32.(rand(0:1, n)), Float32.(X))
    x32 = Float32.(x)
    @test hessian_vec(f, x32, ones(Float32, p)) isa Vector{Float32}
    q = quadratic_model(f, x32)
    @test eltype(q.y) == Float32
    @test gradient(q, x32) ≈ gradient(f, x32) rtol = 1f-4
  end
end

@testset "LogDetLoss" begin
  p = 6
  B = randn(3p, p)
  S = Matrix(Symmetric(B'B / 3p))
  X = Matrix(Symmetric(inv(S) + 0.3I))
  for w in (1.0, 0.4)
    f = LogDetLoss(S, w)
    @testset "w = $w" begin
      check_hessian(f, X; symmetric = true)
      V = Matrix(Symmetric(randn(p, p)))
      W = inv(X)
      @test hessian_vec(f, X, V) ≈ w * W * V * W
      @test hessian_vec(f, SymmetricSparseIterate(X), SymmetricSparseIterate(V)) ≈ w * W * V * W

      q = check_model(f, X; symmetric = true)
      @test q isa KronQuadratic
      @test q.A ≈ w * W && q.B ≈ W && q.C ≈ w * (S - 2W)
      @test issymmetric(q.A) && issymmetric(q.B) && issymmetric(q.C)
      # the expansion in the step Δ = D - X, w(tr((S - W)Δ) + tr(WΔWΔ)/2), exactly
      for _ in 1:3
        Δ = Matrix(Symmetric(randn(p, p)))
        @test q(X + Δ) - q(X) ≈ w * (tr((S - W) * Δ) + tr(W * Δ * W * Δ) / 2) rtol = 1e-10
      end
      # on a symmetric iterate the model has the gradient of f
      @test gradient(q, SymmetricSparseIterate(X)) ≈ gradient(f, X)
      # without a penalty, the minimizer of the model is Newton's step 2X - XSX
      @test norm(gradient(q, 2X - X * S * X)) <= 1e-10 * norm(S)
      # in place, at another point
      X2 = Matrix(Symmetric(X + 0.1 * Matrix(Symmetric(randn(p, p))) / sqrt(p)))
      @test isposdef(X2)
      @test quadratic_model!(q, f, X2) === q
      q2 = quadratic_model(f, X2)
      @test q.A ≈ q2.A && q.B ≈ q2.B && q.C ≈ q2.C
    end
  end

  f = LogDetLoss(S)
  # off the cone: no Hessian, no model
  N = Matrix(Diagonal([-1.0; ones(p - 1)]))
  @test all(isnan, hessian_vec(f, N, Matrix(1.0I, p, p)))
  @test_throws ArgumentError quadratic_model(f, N)
  @test_throws ArgumentError hessian_vec!(zeros(Float32, p, p), f, X, X)
  # an output whose columns are not contiguous (issue #214)
  V = Matrix(Symmetric(randn(p, p)))
  Hv = view(zeros(2p, p), 1:2:2p, :)
  @test hessian_vec!(Hv, f, X, V) === Hv
  @test Hv ≈ hessian_vec(f, X, V)
  @test all(iszero, parent(Hv)[2:2:end, :])

  @testset "Float32" begin
    f32 = LogDetLoss(Float32.(S))
    X32 = Float32.(X)
    V32 = Float32.(Matrix(Symmetric(randn(p, p))))
    H32 = hessian_vec(f32, X32, V32)
    @test H32 isa Matrix{Float32}
    @test H32 ≈ Float32.(hessian_vec(f, X, Float64.(V32))) rtol = 1f-3
    q32 = quadratic_model(f32, X32)
    @test eltype(q32.A) == Float32
  end
end

@testset "functions built from functions" begin
  n, p = 40, 5
  X, y = randn(n, p), randn(n)
  x, c = randn(p) ./ 3, randn(p)
  fl = LogisticLoss(rand(0:1, n), X)
  for f in (Scaled(fl, 0.5), Tilt(fl, c, 2.), Translate(fl, c), Sum(fl, SqrNormL2(0.5)),
            Sum(LeastSquaresLoss(y, X), fl, LinearFunction(c)), (1 / n) * LeastSquaresLoss(y, X))
    @testset "$(nameof(typeof(f)))" begin
      check_hessian(f, x)
      check_model(f, x)
      # in place at another point
      q = quadratic_model(f, x)
      x2 = randn(p) ./ 3
      @test quadratic_model!(q, f, x2) === q
      @test gradient(q, x2) ≈ gradient(f, x2)
      @test hessian_vec(q, x2, c) ≈ hessian_vec(f, x2, c)
    end
  end
  @test hessian_vec(Scaled(fl, 0.5), x, c) ≈ 0.5 * hessian_vec(fl, x, c)
  @test hessian_vec(Translate(fl, c), x, c) ≈ hessian_vec(fl, x - c, c)
  @test quadratic_model(Scaled(LeastSquaresLoss(y, X), 2.), x) isa Scaled{<:LeastSquaresLoss}

  # blocks
  s = SeparableSum(LeastSquaresLoss(y, X), SqrNormL2(2.))
  @test hashessian(s)
  xb, vb = (randn(p), randn(3)), (randn(p), randn(3))
  Hb = hessian_vec(s, xb, vb)
  @test Hb[1] ≈ X' * (X * vb[1]) && Hb[2] ≈ 2vb[2]
  qs = quadratic_model(s, xb)
  @test qs isa SeparableSum
  @test quadratic_model!(qs, s, xb) === qs

  s = SeparableSum(SqrNormL2(), SqrNormL2(2))
  point, direction = ([1., 2.], [3.]), ([2., -1.], [4.])
  out = map(similar, point)
  expected = map((f, xk, vk) -> hessian_vec(2f, xk, vk), s.fs, point, direction)
  @test hessian_vec!(out, Scaled(s, 2), point, direction) === out
  @test all(map(isapprox, out, expected))
  @test all(map(isapprox, hessian_vec(Scaled(s, 2), point, direction), expected))
  @test allocs(hessian_vec!, out, Scaled(s, 2), point, direction) == 0

  nested = Scaled(SeparableSum(s, SqrNormL2(3)), 2)
  S = Symmetric([1. 2.; 2. 3.])
  H = Hermitian(ComplexF64[1 2+im; 2-im 3])
  for matrix in (S, H)
    product = hessian_vec(nested, (point, matrix), (direction, matrix))
    @test all(map(isapprox, product[1], expected))
    @test product[2] isa Matrix
    @test product[2] ≈ 6matrix
  end
  # Later invalid blocks and directions are checked before the first product.
  for f in (SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))),
            Scaled(SeparableSum(SqrNormL2(), SqrNormL2(1., ones(2))), 2))
    out = map(copy, point)
    @test_throws DimensionMismatch hessian_vec!(out, f, point, direction)
    @test out == point
  end
  out = map(copy, point)
  @test_throws DimensionMismatch hessian_vec!(out, s, point, (direction[1], ones(2)))
  @test out == point

  # no Hessian
  @test !hashessian(NormL1(1.))
  @test !hashessian(Sum(fl, NormL1(1.)))
  @test hashessian(LogisticLoss)
  @test_throws ArgumentError hessian_vec(NormL1(1.), x, x)
  @test_throws ArgumentError quadratic_model(NormL1(1.), x)
  @test_throws DimensionMismatch hessian_vec(fl, x, randn(p + 1))
end

a_hv!(Hv, f, x, v) = hessian_vec!(Hv, f, x, v)
a_qm!(q, f, x) = quadratic_model!(q, f, x)

@testset "zero allocations and inference" begin
  n, p = 50, 10
  X, y = randn(n, p), randn(n)
  x, v = randn(p) ./ 3, randn(p)
  B = randn(2n, p)
  S = Matrix(Symmetric(B'B / 2n))
  P = Matrix(Symmetric(inv(S) + 0.3I))
  V = Matrix(Symmetric(randn(p, p)))
  Hv, HV = similar(x), similar(P)
  fl = LogisticLoss(rand(0:1, n), X; weights = rand(n), offset = randn(n))
  for f in (SqrNormL2(2.), QuadraticFunction(X'X, randn(p)), LeastSquaresLoss(y, X), fl,
            PoissonLoss(rand(0:3, n), X), HuberLoss(y, X, 0.5), Scaled(fl, 0.5), Tilt(fl, randn(p)),
            Translate(fl, randn(p)), Sum(LeastSquaresLoss(y, X), SqrNormL2(0.5)))
    @testset "$(nameof(typeof(f)))" begin
      @test allocs(a_hv!, Hv, f, x, v) == 0
      @inferred hessian_vec!(Hv, f, x, v)
      q = quadratic_model(f, x)
      @test allocs(a_qm!, q, f, x) == 0
      @inferred quadratic_model!(q, f, x)
    end
  end
  xs = SparseIterate(sprandn(p, 0.3))
  @test allocs(a_hv!, Hv, LeastSquaresLoss(y, X), xs, xs) == 0
  @test allocs(a_hv!, Hv, fl, xs, xs) == 0

  f = LogDetLoss(S)
  @test allocs(a_hv!, HV, f, P, V) == 0
  @inferred hessian_vec!(HV, f, P, V)
  q = quadratic_model(f, P)
  @test allocs(a_qm!, q, f, P) == 0
  @inferred quadratic_model!(q, f, P)
  f32 = LogDetLoss(Float32.(S))
  @test allocs(a_hv!, Float32.(HV), f32, Float32.(P), Float32.(V)) == 0

  fk = KronQuadratic(S, P, randn(p, p))
  @test allocs(a_hv!, HV, fk, P, V) == 0
  @test allocs(a_hv!, HV, fk, SymmetricSparseIterate(P), V) == 0
end

end
