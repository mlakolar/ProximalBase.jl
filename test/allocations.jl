module AllocationsTest

# The hot calls allocate nothing once a function object exists and has seen its
# input once (to size its workspaces): f(x), prox!, value_and_prox!,
# value_and_gradient!, gradient! and prox_coordinate!.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
a_grad!(g, f, x) = gradient!(g, f, x)
a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
a_kkt(f, g, x, k) = kkt_violation(f, g, x, k)
a_gmap!(G, f, g, x, γ) = gradient_mapping!(G, f, g, x, γ)

function zero_alloc_prox(f, x; γ = 0.7, value = true)
  y = similar(x)
  value && @test allocs(a_value, f, x) == 0
  @test allocs(a_prox!, y, f, x, γ) == 0
  @test allocs(a_vprox!, y, f, x, γ) == 0
  z = copy(x)
  @test allocs(a_prox!, z, f, z, γ) == 0                  # in place
end

function zero_alloc_grad(f, x)
  g = similar(Array(x))
  @test allocs(a_value, f, x) == 0
  @test allocs(a_vgrad!, g, f, x) == 0
  @test allocs(a_grad!, g, f, x) == 0
end

@testset "zero allocations" begin
  n, p = 50, 20
  X = randn(n, p)
  y = randn(n)
  x = randn(p)
  xs = SparseIterate(sprandn(p, 0.3))
  w = rand(p)
  groups = [1:5, 6:10, 11:15, 16:20]
  xa = AtomIterate(randn(p), groups)

  @testset "separable penalties" begin
    for T in (Float32, Float64), F in (NormL1, NormL1Nonneg)
      z = rand(T, 513)
      zero_alloc_prox(F(T(0.3), rand(T, length(z))), z; γ=T(0.7))
      # The exceptional fallback also remains allocation free.
      zero_alloc_prox(F(floatmin(T), fill(floatmin(T), length(z))), z; γ=T(0.7))
    end
    for f in (ZeroFunction(), LinearFunction(randn(p)), SqrNormL2(2.), SqrNormL2(1., randn(p)),
              NormL1(0.3), NormL1(0.3, w), NormL1Nonneg(0.3, w), ElasticNet(0.3, 0.4, w), IndBox(-0.5, 0.5),
              IndBox(-w, w), IndNonnegative(), NormL0(0.1), MCP(0.3), SCAD(0.3))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
        zero_alloc_prox(f, xs)
        xc = copy(xs)
        @test allocs(a_cd!, xc, f, 3, 0.7) == 0
        xc = copy(xa)
        @test allocs(a_cd!, xc, f, 2, 0.7) == 0
        f isa NormL0 || @test allocs(a_kkt, f, x, x, 4) == 0       # no optimality test at a jump
      end
    end
    # a SymmetricSparseIterate
    D = randn(8, 8); D = D + D'
    s = SymmetricSparseIterate(D)
    for f in (NormL1(0.2), MCP(0.2), IndBox(-0.3, 0.3))
      zero_alloc_prox(f, s)
      @test allocs(a_cd!, s, f, 5, 0.7) == 0
    end
  end

  @testset "norms" begin
    for f in (NormL2(0.4), NormLinf(0.4), GroupNormL2(0.4, groups), GroupNormLinf(0.4, groups))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
      end
    end
    for f in (GroupNormL2(0.4, groups), GroupNormLinf(0.4, groups))
      xg = AtomIterate(randn(p), f)
      zero_alloc_prox(f, xg)
      @test allocs(a_cd!, copy(xg), f, 2, 0.7) == 0
      @test allocs(a_kkt, f, x, xg, 3) == 0
    end
    fl = NormL1Fused(0.3, 0.4)
    pair, out = (randn(p), randn(p)), (zeros(p), zeros(p))
    @test allocs(a_prox!, out, fl, pair, 0.7) == 0
    @test allocs(a_value, fl, pair) == 0
  end

  @testset "sets" begin
    for f in (IndBallL0(5), IndBallL1(1.), IndBallL2(1.), IndSimplex(1.), IndSimplex(1.; equality = false),
              IndHyperplane(randn(p), 0.3), IndHalfspace(randn(p), 0.3))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
      end
    end
    zero_alloc_prox(IndBallL0(3), xs)
    # long enough for the simplex threshold to hand over to Michelot's algorithm
    xl = randn(20_000)
    for f in (IndSimplex(10_000.), IndBallL1(5_000.), NormLinf(0.01))
      zero_alloc_prox(f, xl)
    end
  end

  @testset "functions of the order of the entries" begin
    for f in (TotalVariation1D(0.4), FusedLasso(0.2, 0.4), IndIsotonic(), IndIsotonic(decreasing = true),
              SLOPE(sort(rand(p); rev = true)))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
      end
    end
  end

  @testset "spectral functions" begin
    M = randn(12, 8)
    A = randn(10, 10); A = A + A'
    for (f, Z) in ((NuclearNorm(0.5), M), (IndBallRank(3), M), (IndPSD(), A), (IndPSD(rank = 3), A),
                   (IndPSD(rank = 10), A), (TracePSD(0.2), A))
      @testset "$(typeof(f))" begin
        zero_alloc_prox(f, Z)
        zero_alloc_prox(f, Float32.(Z))
      end
    end
    for f in (Spectral(NormL1(0.3)), Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, 2))), Spectral(IndBox(-0.5, 0.5)))
      @testset "Spectral($(nameof(typeof(f.f))))" begin
        zero_alloc_prox(f, A)
        zero_alloc_prox(f, Float32.(A))
      end
    end
  end

  @testset "partial decompositions" begin
    # low-rank inputs, where the partial path accepts its result; two warm-up
    # calls, since the first one also learns the rank and sizes the block
    M = Matrix(qr(randn(120, 4)).Q) * Diagonal([20.0, 16.0, 13.0, 10.0]) * Matrix(qr(randn(80, 4)).Q)' +
        0.002 * randn(120, 80)
    Q = Matrix(qr(randn(100, 4)).Q)
    A = Q * Diagonal([20.0, 16.0, 13.0, 10.0]) * Q' + 0.001 * Matrix(Symmetric(randn(100, 100)))
    for (f, Z) in ((NuclearNorm(0.5; partial = true), M), (IndBallRank(4; partial = true), M),
                   (IndPSD(rank = 4, partial = true), A), (IndPSD(partial = true), A - 0.5I),
                   (TracePSD(0.2; partial = true), A))
      @testset "$(typeof(f))" begin
        for z in (Z, Float32.(Z))
          prox(f, z, 0.7f0)
          prox(f, z, 0.7f0)
          zero_alloc_prox(f, z)
        end
        @test f.pws isa Union{ProximalBase.PartialSVDWorkspace, ProximalBase.PartialEigenWorkspace}
      end
    end
    # a rank too high for the partial path: every call takes the full decomposition
    R = randn(120, 80)
    D = randn(100, 100); D = D + D'
    for (f, Z) in ((NuclearNorm(0.01; partial = true), R), (IndPSD(partial = true), D))
      prox(f, Z, 0.7)
      prox(f, Z, 0.7)
      zero_alloc_prox(f, Z; value = false)
    end
  end

  @testset "symmetric outputs that are not strided" begin
    A = randn(10, 10); A = A + A'
    S = SymmetricSparseIterate(10)
    for f in (IndPSD(), IndPSD(rank = 3), TracePSD(0.2), Spectral(NormL1(0.3)))
      @test allocs(a_prox!, S, f, A, 0.7) == 0
    end
    B = randn(40, 10)
    L = LogDetLoss(Matrix(Symmetric(B'B / 40)))
    @test allocs(a_vprox!, S, L, Matrix(1.0I, 10, 10), 0.7) == 0
  end

  @testset "smooth functions" begin
    B = randn(2n, p)
    S = Matrix(Symmetric(B'B / 2n))
    P = Matrix(I + 0.1 * Symmetric(randn(p, p)))
    for (f, z) in ((QuadraticFunction(X'X, randn(p), 1.), x), (LeastSquaresLoss(y, X), x),
                   (LeastSquaresLoss(randn(n, 3), X), randn(p, 3)), (LogisticLoss(rand(0:1, n), X), x),
                   (PoissonLoss(rand(0:3, n), X), 0.1 .* x), (HuberLoss(y, X, 0.5), x),
                   (LogisticLoss(rand(0:1, n), X; weights = rand(n), offset = randn(n)), x),
                   (LogDetLoss(S), P), (LogDetLoss(Float32.(S)), Float32.(P)))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_grad(f, z)
        isproximable(f) && zero_alloc_prox(f, z)
      end
    end
    # KronQuadratic on a matrix, vec Θ, sparse iterates of either shape and a
    # symmetric iterate
    Bq = Matrix(Symmetric(randn(4, 4)))
    fk = KronQuadratic(S, Bq, randn(p, 4))
    for z in (randn(p, 4), randn(4p), SparseIterate(sprandn(p, 4, 0.3)), SparseIterate(sprandn(4p, 0.3)))
      zero_alloc_grad(fk, z)
    end
    zero_alloc_grad(KronQuadratic(S, P, randn(p, p)), SymmetricSparseIterate(P))
    # products with sparse iterates
    zero_alloc_grad(LeastSquaresLoss(y, X), xs)
    zero_alloc_grad(LogisticLoss(rand(0:1, n), X), xs)
    zero_alloc_grad(LeastSquaresLoss(y, X), xa)
    # the gradient mapping
    @test allocs(a_gmap!, similar(x), LeastSquaresLoss(y, X), NormL1(0.3), x, 0.01) == 0
  end

  @testset "calculus" begin
    c = randn(p)
    for f in (3 * NormL1(0.2), Tilt(NormL1(0.2), c), Translate(NormL1(0.2), c), Conjugate(NormL2(0.5)),
              Conjugate(NormL1(0.5)))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
      end
    end
    zero_alloc_grad(Sum(LeastSquaresLoss(y, X), SqrNormL2(0.5)), x)
    zero_alloc_grad(Translate(SqrNormL2(), c), x)
    sep = SeparableSum(NormL1(0.2), SqrNormL2(1.))
    xy, out = (randn(p), randn(p)), (zeros(p), zeros(p))
    @test allocs(a_prox!, out, sep, xy, 0.7) == 0
    @test allocs(a_vprox!, out, sep, xy, 0.7) == 0
  end

  @testset "constrained" begin
    for f in (Constrained(NormL1(0.3, w), IndBox(-w, w)), Constrained(ElasticNet(0.3, 0.4), IndNonnegative()),
              Constrained(NormL1(0.3), IndHyperplane(1, 0)), Constrained(IndBox(0, 1), IndHyperplane(randn(p), 1.)),
              Constrained(SqrNormL2(1.), IndSimplex(2.)), Constrained(NormL1(0.3), IndHalfspace(rand(p), -1.)))
      @testset "$(nameof(typeof(f)))" begin
        zero_alloc_prox(f, x)
        zero_alloc_prox(f, Float32.(x))
      end
    end
    f = Constrained(NormL1(0.3), IndBox(-1, 1))
    zero_alloc_prox(f, xs)
    @test allocs(a_cd!, copy(xs), f, 3, 0.7) == 0
    @test allocs(a_cd!, copy(xa), f, 2, 0.7) == 0
    @test allocs(a_kkt, f, x, x, 4) == 0
  end
end

end
