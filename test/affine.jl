module AffineTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# the projection by a dense least-norm solve
reference(A, b, x) = x - pinv(Matrix(A)) * (Matrix(A) * x - b)

@testset "IndAffine" begin
  m, n = 4, 10
  A = randn(m, n)
  b = randn(m)
  f = IndAffine(A, b)
  @test f.fact isa Cholesky
  check_traits(f; proximable = true, convex = true)
  x = randn(n)
  y = prox(f, x)
  @test y ≈ reference(A, b, x)
  @test A * y ≈ b
  @test prox(f, y) ≈ y                          # idempotent
  @test f(y) == 0 && f(x) == Inf
  @test value_and_prox!(similar(x), f, x, 1.) == 0
  # x - y is orthogonal to the null space of A, so no point of the set is closer
  N = nullspace(A)
  @test norm(N' * (x - y)) <= 1e-10 * norm(x)
  z = copy(x)
  prox!(z, f, z, 1.)
  @test z ≈ y                                  # in place
  # rank deficient but consistent: a repeated row takes the SVD path
  A2 = [A; A[2:2, :]]
  b2 = A2 * randn(n)
  f2 = IndAffine(A2, b2)
  @test f2.fact isa Tuple
  y2 = prox(f2, x)
  @test y2 ≈ reference(A2, b2, x)
  @test A2 * y2 ≈ b2
  @test f2(y2) == 0
  @test_throws ArgumentError IndAffine(A2, [b; 0.0] .+ [zeros(m); 1.0])   # inconsistent: the set is empty
  # a sparse A gives the same projection
  As = sprandn(m, n, 0.5) + sparse(1:m, 1:m, ones(m), m, n)
  fs = IndAffine(As, b)
  @test prox(fs, x) ≈ reference(As, b, x)
  # a tight frame, AAᵀ = μI, needs no factorization
  Q = Matrix(qr(randn(n, m)).Q)[:, 1:m]
  At = 3 * Q'
  ft = IndAffine(At, b; μ = 9)
  @test prox(ft, x) ≈ prox(IndAffine(At, b), x)
  @test_throws ArgumentError IndAffine(At, b; μ = 0)
  # one row is the hyperplane
  a = randn(n)
  @test prox(IndAffine(reshape(a, 1, n), [0.7]), x) ≈ prox(IndHyperplane(a, 0.7), x)
  # Float32
  f32 = IndAffine(Float32.(A), Float32.(b))
  @test prox(f32, Float32.(x)) isa Vector{Float32}
  @test prox(f32, Float32.(x)) ≈ y rtol = 1e-4
  @test_throws DimensionMismatch IndAffine(A, randn(m + 1))
  @test_throws DimensionMismatch prox(f, randn(n + 1))
  # its conjugate, by Moreau's identity: x = prox(f, x) + prox(f*, x) with γ = 1
  @test prox(f, x) + prox(Conjugate(f), x) ≈ x
end

@testset "zero allocations" begin
  a_value(f, x) = f(x)
  a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
  a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  m, n = 5, 30
  A = randn(m, n)
  x = randn(n)
  y = similar(x)
  for f in (IndAffine(A, randn(m)), IndAffine([A; A[1:1, :]], [A; A[1:1, :]] * randn(n)),
            IndAffine(sprandn(m, n, 0.4) + sparse(1:m, 1:m, ones(m), m, n), randn(m)),
            IndAffine(Matrix(qr(randn(n, m)).Q)[:, 1:m]', randn(m); μ = 1))
    @test allocs(a_value, f, x) == 0
    @test allocs(a_prox!, y, f, x, 1.) == 0
    @test allocs(a_vprox!, y, f, x, 1.) == 0
  end
end

# the restricted step by a dense KKT solve: (H + I/γ)z + Aᵀλ = c, Az = b
function kkt_reference(H, c, A, b, γ)
  n, m = size(H, 1), size(A, 1)
  K = [H + I / γ A'; A zeros(m, m)]
  (K \ [c; b])[1:n]
end
kkt_reference(f::LeastSquaresLoss, A, b, v, γ) = kkt_reference(f.X' * f.X, f.X' * f.Y + v / γ, Matrix(A), b, γ)
kkt_reference(f::QuadraticFunction, A, b, v, γ) =
  kkt_reference(Matrix(f.A), v / γ - (f.b === nothing ? zero(v) : f.b), Matrix(A), b, γ)

@testset "Constrained(f, IndAffine): the restricted least-squares step" begin
  rng = Xoshiro(5)
  m, p = 3, 12
  A = randn(rng, m, p)
  b = randn(rng, m)
  P = (B = randn(rng, p, p); B * B' / p)
  for (n, f) in ((40, LeastSquaresLoss(randn(rng, 40), randn(rng, 40, p))),     # V square
                 (7, LeastSquaresLoss(randn(rng, 7), randn(rng, 7, p))),        # p > n: V has 7 columns
                 (0, QuadraticFunction(P, randn(rng, p), 0.3)),
                 (0, QuadraticFunction(P)))
    c = Constrained(f, IndAffine(A, b))
    check_traits(c; proximable = true, convex = true)
    v = randn(rng, p)
    for γ in (1.0, 0.1, 7.5, 1.0)            # a change of γ between calls refactors
      z = prox(c, v, γ)
      @test z ≈ kkt_reference(f, A, b, v, γ) rtol = 1e-10
      @test A * z ≈ b
      @test c(z) ≈ f(z) && c(v) == Inf
      @test value_and_prox!(similar(v), c, v, γ) ≈ f(z)
      # the residual of the proximal problem is normal to the set: in the range of Aᵀ
      g = similar(z)
      gradient!(g, f, z)
      @test norm(nullspace(A)' * ((v - z) / γ - g)) <= 1e-9 * max(1, norm(g))
    end
    check_prox(c, v, 0.7; local_points = false)
    w = copy(v)
    prox!(w, c, w, 2.0)
    @test w ≈ prox(c, v, 2.0)                 # in place
    @test prox!(SparseIterate(p), c, v, 2.0) ≈ prox(c, v, 2.0)
    @test (@inferred prox(c, v, 0.5)) isa Vector{Float64}
  end
  # the other paths of IndAffine: rank deficient, sparse, a tight frame
  f = LeastSquaresLoss(randn(rng, 7), randn(rng, 7, p))
  v = randn(rng, p)
  A2 = [A; A[2:2, :] + A[3:3, :]]
  b2 = A2 * randn(rng, p)
  @test IndAffine(A2, b2).fact isa Tuple
  z = prox(Constrained(f, IndAffine(A2, b2)), v, 0.4)
  @test z ≈ kkt_reference(f, A, b2[1:m], v, 0.4)
  @test A2 * z ≈ b2
  As = sprandn(rng, m, p, 0.4) + sparse(1:m, 1:m, ones(m), m, p)
  @test prox(Constrained(f, IndAffine(As, b)), v, 0.4) ≈ kkt_reference(f, As, b, v, 0.4)
  Q = Matrix(qr(randn(rng, p, m)).Q)[:, 1:m]
  @test prox(Constrained(f, IndAffine(2 * Q', b; μ = 4)), v, 0.4) ≈ kkt_reference(f, 2 * Q', b, v, 0.4)
  # Float32
  f32 = LeastSquaresLoss(Float32.(f.Y), Float32.(f.X))
  z32 = prox(Constrained(f32, IndAffine(Float32.(A), Float32.(b))), Float32.(v), 0.4f0)
  @test z32 isa Vector{Float32}
  @test z32 ≈ kkt_reference(f, A, b, v, 0.4) rtol = 1e-3
  # errors
  @test_throws ArgumentError Constrained(LeastSquaresLoss(randn(7, 2), f.X), IndAffine(A, b))
  @test_throws DimensionMismatch Constrained(f, IndAffine(randn(m, p + 1), b))
  @test_throws ArgumentError Constrained(NormL1(1.0), IndAffine(A, b))
  @test_throws DimensionMismatch prox(Constrained(f, IndAffine(A, b)), randn(p + 1))
end

# ADMM on x = z, with the restricted least-squares step as f
function admm!(x, z, u, f, g, γ; iters = 20_000, tol = 1e-12)
  w = similar(x)
  for _ in 1:iters
    w .= z .- u
    prox!(x, f, w, γ)
    w .= x .+ u
    zold = copy(z)
    prox!(z, g, w, γ)
    u .+= x .- z
    max(norm(x - z), norm(z - zold)) <= tol * max(1, norm(z)) && break
  end
  z
end

@testset "ADMM: the lasso under equality constraints" begin
  rng = Xoshiro(8)
  n, p, m = 50, 15, 3
  X = randn(rng, n, p)
  β = [randn(rng, 5); zeros(p - 5)]
  Y = X * β + 0.1 * randn(rng, n)
  A = randn(rng, m, p)
  b = A * β
  λ, γ = 4.0, 0.05
  f = Constrained(LeastSquaresLoss(Y, X), IndAffine(A, b))
  x, z, u = zeros(p), zeros(p), zeros(p)
  admm!(x, z, u, f, NormL1(λ), γ)
  @test A * z ≈ b rtol = 1e-8
  # u/γ is a subgradient of λ‖⋅‖₁ at z, and -u/γ - ∇(‖Y - Xz‖²/2) is in the range of Aᵀ
  s = u / (γ * λ)
  @test all(abs.(s) .<= 1 + 1e-8)
  @test all(isapprox(s[i], sign(z[i]); atol = 1e-8) for i in findall(!iszero, z))
  @test count(!iszero, z) < p
  r = -u / γ - X' * (X * z - Y)
  @test norm(nullspace(A)' * r) <= 1e-7 * norm(r)
  # one constraint, coefficients that sum to zero: the same minimizer as proximal
  # gradient with the hyperplane-constrained penalty
  f1 = Constrained(LeastSquaresLoss(Y, X), IndAffine(ones(1, p), [0.0]))
  x, z, u = zeros(p), zeros(p), zeros(p)
  admm!(x, z, u, f1, NormL1(λ), γ)
  g1 = Constrained(NormL1(λ), IndHyperplane(1, 0))
  ls = LeastSquaresLoss(Y, X)
  t = 1 / opnorm(X)^2
  y, grad = zeros(p), zeros(p)
  for _ in 1:20_000
    gradient!(grad, ls, y)
    prox!(y, g1, y - t * grad, t)
  end
  @test z ≈ y rtol = 1e-7
end

@testset "Constrained(f, IndAffine): zero allocations" begin
  a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
  a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  a_two!(y, f, x) = (prox!(y, f, x, 0.5); prox!(y, f, x, 2.0))   # each call refactors
  m, p = 4, 20
  A = randn(m, p)
  x = randn(p)
  y = similar(x)
  for f in (LeastSquaresLoss(randn(40), randn(40, p)), LeastSquaresLoss(randn(9), randn(9, p)),
            QuadraticFunction(Symmetric(randn(p, p)) + 2p * I, randn(p)))
    for C in (IndAffine(A, randn(m)), IndAffine([A; A[1:1, :]], [A; A[1:1, :]] * randn(p)))
      c = Constrained(f, C)
      @test allocs(a_prox!, y, c, x, 1.) == 0
      @test allocs(a_vprox!, y, c, x, 1.) == 0
      @test allocs(a_two!, y, c, x) == 0
      for out in (SparseIterate(p), AtomIterate(p, [1:p]))
        @test prox!(out, c, x, 1.) ≈ prox(c, x, 1.)
        @test value_and_prox!(out, c, x, 1.) ≈ f(out)
        @test allocs(a_prox!, out, c, x, 1.) == 0
        @test allocs(a_vprox!, out, c, x, 1.) == 0
      end
      # The value is at the rounded destination, including when its type differs
      # from that of the loss and its dense workspace.
      out32 = SparseIterate(zeros(Float32, p))
      v32 = value_and_prox!(out32, c, x, 1.)
      @test v32 isa Float32
      @test v32 ≈ Float32(f(out32))
      @test allocs(a_vprox!, out32, c, x, 1.) == 0
    end
  end
end

end
