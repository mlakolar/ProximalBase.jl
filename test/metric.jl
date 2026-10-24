module MetricTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# a smooth function that does not know its per-coordinate constants
struct SmoothWithoutConstants <: AbstractFunction end
ProximalBase.issmooth(::Type{SmoothWithoutConstants}) = true

# the subgradient inequality in the metric Diagonal(1 ./ γ): p = prox_{γf}(x) when
# f(z) ≥ f(p) + ⟨(x - p) ./ γ, z - p⟩ for every z
function check_metric_optimality(f, x, γ, p; points = 5)
  fp = f(p)
  for _ in 1:points, δ in (1e-3, 1e-1, 1.)
    z = p .+ δ .* randn(size(p)...)
    fz = f(z)
    isfinite(fz) && @test fz >= fp + dot((x .- p) ./ γ, z .- p) - 1e-9 * max(1, abs(fp))
  end
end

@testset "a step per entry for separable functions" begin
  n = 12
  w = rand(n) .+ 0.2
  x = 2 .* randn(n)
  γ = rand(n) .+ 0.1
  for f in (NormL1(0.3), NormL1(0.3, w), NormL1Nonneg(0.2), ElasticNet(0.3, 0.4, w), IndBox(-0.5, 0.5),
            NormL0(0.05), MCP(0.3), SCAD(0.3))
    p = prox(f, x, γ)
    @test p == [prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
    @test prox(f, x, fill(0.7, n)) == prox(f, x, 0.7)        # equal steps give the scalar step
    z = copy(x)
    prox!(z, f, z, γ)
    @test z == p                                               # in place
    isconvex(f) && check_metric_optimality(f, x, γ, p)
    # the iterates: a SparseIterate, and a SymmetricSparseIterate with symmetric steps
    xs = SparseIterate(sprandn(n, 0.4))
    @test Vector(prox(f, xs, γ)) ≈ prox(f, Vector(xs), γ)
    if !(hasproperty(f, :w) && f.w isa AbstractArray)        # weights for a vector of length n
      D = randn(4, 4); D = D + D'
      G = rand(4, 4) .+ 0.1; G = G + G'
      S = SymmetricSparseIterate(D)
      @test Matrix(prox(f, S, G)) ≈ prox(f, D, G)
    end
    @test_throws DimensionMismatch prox(f, x, rand(n - 1))
  end
end

@testset "a step per group for group norms" begin
  groups = [1:3, 4:5, [6, 8], 9:12]               # x[7] is in no group
  x = 2 .* randn(12)
  γ = [0.2, 0.9, 0.5, 1.4]
  for F in (GroupNormL2, GroupNormLinf)
    f = F(0.5, groups)
    p = prox(f, x, γ)
    @test p[7] == x[7]
    for k in eachindex(groups)
      G = groups[k]
      @test p[G] ≈ prox(F(0.5, [1:length(G)], [f.w[k]]), x[G], γ[k])
    end
    @test prox(f, x, fill(0.6, 4)) ≈ prox(f, x, 0.6)
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
    # the metric is constant on each group
    γe = zeros(12); γe[7] = 1.0
    for k in eachindex(groups)
      γe[groups[k]] .= γ[k]
    end
    check_metric_optimality(f, x, γe, p)
    # an AtomIterate with the groups of f
    fa = F(0.5, [groups; [[7]]], [f.w; 0.])
    xa = AtomIterate(x, fa)
    ya = similar(xa)
    γa = [γ; 1.0]
    @test value_and_prox!(ya, fa, xa, γa) ≈ fa(prox(fa, x, γa))
    @test Vector(ya) ≈ prox(fa, x, γa)
    @test_throws DimensionMismatch prox(f, x, rand(3))
  end
end

@testset "functions built from functions pass the steps on" begin
  n = 10
  x = 2 .* randn(n)
  γ = rand(n) .+ 0.1
  c, b = randn(n), randn(n)
  for f in (Scaled(NormL1(0.3, rand(n)), 2.), Tilt(NormL1(0.3), c), Translate(NormL1(0.3), b),
            Conjugate(NormL1(0.3)), Conjugate(IndBox(-1., 1.)), 0.5 * Translate(ElasticNet(0.2, 0.3), b))
    p = prox(f, x, γ)
    @test p ≈ [prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    @test prox(f, x, fill(0.4, n)) ≈ prox(f, x, 0.4)
    f isa Conjugate || @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
  end
  # Moreau's decomposition in each entry: x = prox_{γf}(x) + γ ⊙ prox_{f*/γ}(x ./ γ)
  f = NormL1(0.3)
  @test prox(f, x, γ) .+ γ .* prox(Conjugate(f), x ./ γ, 1 ./ γ) ≈ x
  # a scaled group norm with a step per group
  groups = [1:4, 5:10]
  g = GroupNormL2(0.4, groups)
  @test prox(2.0 * g, x, [0.3, 0.8]) ≈ prox(g, x, [0.6, 1.6])
end

@testset "per-coordinate Lipschitz constants" begin
  n, p = 30, 8
  X = randn(n, p)
  y = randn(n)
  H = X' * X
  # least squares, with a vector and with a matrix response
  f = LeastSquaresLoss(y, X)
  @test coordinate_lipschitz!(zeros(p), f) ≈ diag(H)
  @test [coordinate_lipschitz(f, k) for k in 1:p] ≈ diag(H)
  fm = LeastSquaresLoss(randn(n, 3), X)
  @test coordinate_lipschitz!(zeros(p, 3), fm) ≈ repeat(diag(H), 1, 3)
  @test coordinate_lipschitz(fm, p + 2) ≈ H[2, 2]
  @test_throws BoundsError coordinate_lipschitz(f, p + 1)
  @test_throws DimensionMismatch coordinate_lipschitz!(zeros(p + 1), f)
  # a sparse design reads its stored entries
  Xs = sprandn(n, p, 0.3)
  @test coordinate_lipschitz!(zeros(p), LeastSquaresLoss(y, Xs)) ≈ vec(sum(abs2, Matrix(Xs); dims = 1))
  # the diagonal of the Hessian, against finite differences of the gradient
  β = randn(p)
  for g in (f, QuadraticFunction(Symmetric(H) + I, randn(p)), (1 / n) * f, Sum(f, SqrNormL2(2.)),
            Tilt(f, randn(p)), Translate(f, randn(p)))
    for k in (1, 4, p)
      e = zeros(p); e[k] = 1e-4
      fd = (gradient(g, β .+ e)[k] - gradient(g, β .- e)[k]) / 2e-4
      @test coordinate_lipschitz(g, k) ≈ fd rtol = 1e-6
    end
  end
  # GLMs: the curvature bound times the weighted squared column norms, which bounds
  # the diagonal of the Hessian everywhere
  wobs = rand(n) .+ 0.5
  fl = LogisticLoss(Float64.(rand(n) .< 0.5), X; weights = wobs)
  L = coordinate_lipschitz!(zeros(p), fl)
  @test L ≈ 0.25 .* vec(sum(wobs .* X .^ 2; dims = 1))
  η = X * β
  hdiag = vec(sum(wobs .* (exp.(η) ./ (1 .+ exp.(η)) .^ 2) .* X .^ 2; dims = 1))
  @test all(hdiag .<= L)
  @test coordinate_lipschitz(HuberLoss(y, X, 1.0), 3) ≈ H[3, 3]
  Xsp = sprandn(n, p, 0.4)
  @test coordinate_lipschitz!(zeros(p), LogisticLoss(Float64.(rand(n) .< 0.5), Xsp; weights = wobs)) ≈
        0.25 .* vec(sum(wobs .* Matrix(Xsp) .^ 2; dims = 1))       # sparse, weighted
  @test coordinate_lipschitz(PoissonLoss(rand(0:3, n), X), 2) == Inf
  # Kronecker quadratics: A[r, r] B[c, c] for entry (r, c) of Θ
  A = randn(5, 5); A = A * A'
  B = randn(4, 4); B = B * B'
  fk = KronQuadratic(A, B)
  @test vec(coordinate_lipschitz!(zeros(5, 4), fk)) ≈ kron(diag(B), diag(A))
  @test coordinate_lipschitz(SqrNormL2(2.5), 3) == 2.5
  @test coordinate_lipschitz(ZeroFunction(), 1) == 0
  @test coordinate_lipschitz(LinearFunction(randn(p)), 2) == 0
  # the wrappers keep the constants of the loss, also all at once
  @test coordinate_lipschitz!(zeros(p), Tilt(f, randn(p))) ≈ diag(H)
  @test coordinate_lipschitz!(zeros(p), Translate(f, randn(p))) ≈ diag(H)
  @test coordinate_lipschitz!(zeros(p), Sum(f, SqrNormL2(2.))) ≈ diag(H) .+ 2
  @test coordinate_lipschitz!(zeros(p), LeastSquaresLoss(y, sprandn(n, p, 0.3))) isa Vector{Float64}
  @test coordinate_lipschitz(LogDetLoss(Matrix(1.0I, 3, 3)), 2) == Inf
  @test_throws ArgumentError coordinate_lipschitz(NormL1(1.), 1)
  @test_throws ArgumentError coordinate_lipschitz(SmoothWithoutConstants(), 1)
end

@testset "zero allocations" begin
  a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
  a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  a_cl!(L, f) = coordinate_lipschitz!(L, f)
  a_cl(f, k) = coordinate_lipschitz(f, k)
  n = 20
  x = randn(n)
  γ = rand(n) .+ 0.1
  xs = SparseIterate(sprandn(n, 0.3))
  for f in (NormL1(0.3), NormL1(0.3, rand(n)), MCP(0.3), IndBox(-0.5, 0.5), Scaled(NormL1(0.3), 2.),
            Tilt(NormL1(0.3), randn(n)), Translate(NormL1(0.3), randn(n)), Conjugate(NormL1(0.3)))
    y = similar(x)
    @test allocs(a_prox!, y, f, x, γ) == 0
    f isa Conjugate || @test allocs(a_vprox!, y, f, x, γ) == 0
  end
  ys = similar(xs)
  @test allocs(a_prox!, ys, NormL1(0.3), xs, γ) == 0
  groups = [1:5, 6:10, 11:15, 16:20]
  g = GroupNormL2(0.4, groups)
  γg = rand(4) .+ 0.1
  @test allocs(a_prox!, similar(x), g, x, γg) == 0
  xa = AtomIterate(randn(n), g)
  @test allocs(a_vprox!, similar(xa), g, xa, γg) == 0
  X = randn(30, n)
  for f in (LeastSquaresLoss(randn(30), X), LogisticLoss(Float64.(rand(30) .< 0.5), X), QuadraticFunction(X' * X),
            (1 / 30) * LeastSquaresLoss(randn(30), X))
    L = zeros(n)
    @test allocs(a_cl!, L, f) == 0
    @test allocs(a_cl, f, 3) == 0
  end
end

end
