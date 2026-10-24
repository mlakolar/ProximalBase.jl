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

end
