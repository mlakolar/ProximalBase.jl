module SpectralFunctionTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

randsym(n) = (B = randn(n, n); (B + B') / 2)
fantope(k) = Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))

@testset "functions of the eigenvalues" begin
  n = 7
  A = randsym(n)
  λ, V = eigen(Symmetric(A))
  # the functions of the eigenvalues that ProximalBase has
  @test prox(Spectral(IndNonnegative()), A) ≈ prox(IndPSD(), A)
  @test prox(Spectral(NormL1Nonneg(0.3)), A, 0.7) ≈ prox(TracePSD(0.3), A, 0.7)
  @test prox(Spectral(NormL1(0.3)), A, 0.7) ≈ prox(NuclearNorm(0.3), A, 0.7)
  @test Spectral(NormL1(0.3))(A) ≈ NuclearNorm(0.3)(A)
  # the Frobenius norm and its ball, norms of the eigenvalues
  @test prox(Spectral(NormL2(0.5)), A, 0.7) ≈ prox(NormL2(0.5), A, 0.7)
  @test Spectral(NormL2(0.5))(A) ≈ NormL2(0.5)(A)
  @test prox(Spectral(IndBallL2(1.)), A) ≈ prox(IndBallL2(1.), A)
  # the spectral norm and its ball
  @test Spectral(NormLinf(0.5))(A) ≈ 0.5 * opnorm(A)
  P = prox(Spectral(IndBox(-0.5, 0.5)), A)
  @test P ≈ V * Diagonal(clamp.(λ, -0.5, 0.5)) * V'
  @test prox(Spectral(Conjugate(NormL1(0.5))), A) ≈ P
  @test prox(Spectral(NormL1(100.)), A) == zeros(n, n)               # every eigenvalue to zero
  @test prox(Spectral(Constrained(NormL1(0.1), IndBox(-0.5, 0.5))), A) ≈ V * Diagonal(clamp.(shrink.(λ, 0.1), -0.5, 0.5)) * V'
  for f in (Spectral(NormL1(0.3)), Spectral(IndBox(-0.5, 0.5)), Spectral(ElasticNet(0.2, 0.3)), Spectral(NormLinf(0.4)))
    check_prox(f, A, 0.8; symmetric = true)
  end
  @test Spectral(NormL1(0.3))(A + triu(ones(n, n), 1)) == Inf       # not symmetric
  check_traits(Spectral(NormL1(0.3)); proximable = true, convex = true)
  check_traits(Spectral(NormL0(0.3)); proximable = true)
  @test ProximalBase._isindicator(Spectral(IndBox(-1, 1))) && !ProximalBase._isindicator(Spectral(NormL1(0.3)))
end

@testset "the Fantope" begin
  n = 8
  A = randsym(n)
  for k in (1, 2, 3.5)
    F = fantope(k)
    P = check_prox(F, A, 1.; symmetric = true)
    μ = eigvals(Symmetric(P))
    @test all(-1e-12 .<= μ .<= 1 + 1e-12) && tr(P) ≈ k
    @test F(P) == 0 && F(A) == Inf
    # nearer to A than any projection of rank k
    if isinteger(k)
      for _ in 1:20
        Q = Matrix(qr(randn(n, Int(k))).Q)[:, 1:Int(k)]
        @test norm(P - A) <= norm(Q * Q' - A) + 1e-12
      end
    end
  end
  Q = Matrix(qr(randn(n, n)).Q)
  P0 = Q * Diagonal([0.2, 0.9, 0.4, 0.5, 0., 0., 0., 0.]) * Q'
  @test prox(fantope(2), P0) ≈ P0                                    # a point of the Fantope
  D = prox(Spectral(IndSimplex(1.)), A)                              # a density matrix
  @test tr(D) ≈ 1 && minimum(eigvals(Symmetric(D))) >= -1e-12
end

# the projection onto the capped simplex {0 ≤ x ≤ 1, Σᵢ xᵢ = k}, by bisection on
# the level τ of clamp.(x .- τ, 0, 1)
function capped_simplex_proj(x, k)
  lo, hi = minimum(x) - 1, maximum(x)
  for _ in 1:200
    τ = (lo + hi) / 2
    sum(clamp.(x .- τ, 0, 1)) > k ? (lo = τ) : (hi = τ)
  end
  clamp.(x .- (lo + hi) / 2, 0, 1)
end

@testset "the sum of the k largest eigenvalues" begin
  n = 7
  A = randsym(n)
  λ, V = eigen(Symmetric(A))
  top = sort(λ; rev = true)
  for k in (1, 3, 2.5)
    F = Spectral(Conjugate(Constrained(IndBox(0, 1), IndHyperplane(1, k))))
    j = floor(Int, k)
    @test F(A) ≈ sum(top[1:j]) + (k - j) * top[j + 1]
    for γ in (0.3, 1., 3.)
      # Moreau's identity on the eigenvalues, with a projection by bisection,
      # and on the matrices, with the Fantope
      z = λ .- γ .* capped_simplex_proj(λ ./ γ, k)
      @test prox(F, A, γ) ≈ V * Diagonal(z) * V' atol = 1e-9
      @test prox(F, A, γ) .+ γ .* prox(fantope(k), A ./ γ) ≈ A
    end
    check_prox(F, A, 0.7; symmetric = true)
  end
end

@testset "outputs, element types and errors" begin
  n = 6
  A = randsym(n)
  F = fantope(2)
  P = prox(F, A)
  S = SymmetricSparseIterate(n)
  @test Array(prox!(S, F, A, 1.)) ≈ P
  B = copy(A)
  @test prox!(B, F, B, 1.) ≈ P
  P32 = prox(F, Float32.(A))
  @test eltype(P32) == Float32
  @test P32 ≈ P atol = 1e-5
  g = Spectral(NormL1(0.3))
  @test value_and_prox!(similar(A), g, A, 0.5) ≈ g(prox(g, A, 0.5))
  @test_throws ArgumentError Spectral(NormL1(1., rand(n)))
  @test_throws ArgumentError Spectral(IndBox(zeros(n), ones(n)))
  @test_throws ArgumentError Spectral(Constrained(NormL1(1.), IndHyperplane(rand(n), 1.)))
  @test_throws ArgumentError Spectral(GroupNormL2(1., [1:3, 4:6]))
  @test_throws ArgumentError Spectral(LeastSquaresLoss(randn(3), randn(3, 3)))
  # eigenvalues of the result out of order take rank-one updates
  V = Matrix(qr(randn(n, n)).Q)
  z = [0.5, -1., 2., 0., -0.3, 1.]
  Y = zeros(n, n)
  ProximalBase._rebuild_sym_signed!(Y, copy(V), z, Y)
  @test Y ≈ V * Diagonal(z) * V'
  # an output whose columns are not contiguous goes through the buffer: syr
  # would write past it (issue #214)
  for h in (F, g)
    Yv = view(zeros(2n, n), 1:2:2n, :)
    @test prox!(Yv, h, A, 0.5) ≈ prox(h, A, 0.5)
    @test all(iszero, parent(Yv)[2:2:end, :])
  end
end

end
