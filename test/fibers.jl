module FibersTest

# SliceSum and FiberSum: functions of stacked arrays, slice by slice and fiber by
# fiber, against the functions they apply

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

# K symmetric p × p slices
function symslices(p, K)
  X = randn(p, p, K)
  for k in 1:K
    X[:, :, k] .= (X[:, :, k] .+ X[:, :, k]') ./ 2
  end
  X
end

a_value(f, x) = f(x)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)

@testset "SliceSum" begin
  p, K = 6, 3
  Ss = [(A = randn(20, p); A' * A / 20) for _ in 1:K]
  ws = [0.2, 0.3, 0.5]
  fs = [LogDetLoss(Ss[k], ws[k]) for k in 1:K]
  Θ = zeros(p, p, K)
  for k in 1:K
    Θ[:, :, k] = I + 0.05 * Symmetric(randn(p, p))
  end
  # the K log-det terms of the joint graphical lasso, as a vector and as a tuple
  for s in (SliceSum(fs), SliceSum(fs...))
    @test issmooth(s) && isproximable(s) && isconvex(s)
    @test s(Θ) ≈ sum(fs[k](Θ[:, :, k]) for k in 1:K)
    Y = similar(Θ)
    v = value_and_prox!(Y, s, Θ, 0.7)
    @test all(Y[:, :, k] == prox(fs[k], Θ[:, :, k], 0.7) for k in 1:K)
    @test v ≈ s(Y)
    @test prox!(similar(Θ), s, Θ, 0.7) == Y
    G = similar(Θ)
    @test value_and_gradient!(G, s, Θ) ≈ s(Θ)
    @test all(G[:, :, k] ≈ gradient(fs[k], Θ[:, :, k]) for k in 1:K)
    @test lipschitz(s) == Inf
    @test allocs(a_value, s, Θ) == 0
    @test allocs(a_vprox!, Y, s, Θ, 0.7) == 0
    @test allocs(a_vgrad!, G, s, Θ) == 0
  end
  # the slices of a matrix are its columns, and the functions may differ
  s = SliceSum(NormL1(0.3), NormL2(0.5), ZeroFunction())
  @test !issmooth(s) && isproximable(s) && isconvex(s)
  x = randn(4, 3)
  y = prox(s, x, 0.7)
  @test y[:, 1] == prox(NormL1(0.3), x[:, 1], 0.7)
  @test y[:, 2] == prox(NormL2(0.5), x[:, 2], 0.7)
  @test y[:, 3] == x[:, 3]
  check_prox(s, x, 0.7)
  @test lipschitz(SliceSum(SqrNormL2(2.0), SqrNormL2(3.0))) == 3.0
  @test_throws DimensionMismatch prox(s, randn(4, 2), 0.7)
  @test_throws DimensionMismatch prox!(randn(4, 2), s, x, 0.7)
  @test_throws ArgumentError SliceSum(AbstractFunction[])
  # Float32 stays Float32
  s32 = SliceSum([LogDetLoss(Float32.(S)) for S in Ss])
  Θ32 = Float32.(Θ)
  @test prox(s32, Θ32, 0.7f0) isa Array{Float32, 3}
  @test s32(Θ32) isa Float32
end

@testset "FiberSum" begin
  p, γ, λ1, λ2 = 6, 0.3, 0.4, 0.2

  # two classes: the fused graphical lasso, fiber by fiber as the Z-update of
  # its ADMM, with the diagonal penalized, only fused, or free
  X = symslices(p, 2)
  fused = NormL1Fused(λ1, λ2)
  for diagonal in (fused, NormL1Fused(0.0, λ2), nothing)
    fb = FiberSum(fused; diagonal)
    @test isproximable(fb) && isconvex(fb) && !issmooth(fb)
    Y = similar(X)
    v = value_and_prox!(Y, fb, X, γ)
    for j in 1:p, i in 1:p
      d = i == j ? diagonal : fused
      @test (Y[i, j, 1], Y[i, j, 2]) == (d === nothing ? (X[i, j, 1], X[i, j, 2]) : prox(d, (X[i, j, 1], X[i, j, 2]), γ))
    end
    @test Y[:, :, 1] == Y[:, :, 1]' && Y[:, :, 2] == Y[:, :, 2]'    # symmetric slices stay symmetric
    value = sum(fused((Y[i, j, 1], Y[i, j, 2])) for j in 1:p, i in 1:p if i != j) +
            (diagonal === nothing ? 0.0 : sum(diagonal((Y[i, i, 1], Y[i, i, 2])) for i in 1:p))
    @test v ≈ value
    @test fb(Y) ≈ value
    check_prox(fb, X, γ)
    @test allocs(a_value, fb, X) == 0
    @test allocs(a_vprox!, Y, fb, X, γ) == 0
    Z = copy(X)
    @test prox!(Z, fb, Z, γ) == Y                                    # in place
    # symmetric slices: the fibers with i ≥ j, mirrored, give the same result
    fs = FiberSum(fused; diagonal, symmetric = true)
    Ys = similar(X)
    @test value_and_prox!(Ys, fs, X, γ) ≈ v
    @test Ys == Y
    @test fs(X) ≈ fb(X)
    @test prox!(similar(X), fs, X, γ) == Y
    Z = copy(X)
    @test prox!(Z, fs, Z, γ) == Y                                    # in place
    @test allocs(a_vprox!, Ys, fs, X, γ) == 0
  end
  @test_throws DimensionMismatch prox(FiberSum(fused; symmetric = true), randn(3, 4, 2), γ)
  @test_throws DimensionMismatch prox(FiberSum(NormL2(0.3); symmetric = true), randn(3, 4), γ)

  # K classes that share a sparsity pattern: GroupNormL2 with strided groups
  K = 4
  X = randn(p, p, K)
  gl = GroupNormL2(0.3, [i:p^2:p^2*K for i in 1:p^2], ones(p^2))
  fb = FiberSum(NormL2(0.3))
  @test prox(fb, X, γ) ≈ reshape(prox(gl, vec(X), γ), p, p, K)
  @test fb(X) ≈ gl(vec(X))
  check_prox(fb, X, γ)
  @test allocs(a_vprox!, similar(X), fb, X, γ) == 0
  # simultaneous selection, with a free diagonal
  check_prox(FiberSum(NormLinf(0.3); diagonal = nothing), X, γ)

  # the fibers of a matrix are its rows: the row-sparse penalty of multitask regression
  B = randn(5, 3)
  @test prox(FiberSum(NormL2(0.3)), B, γ) ≈ reshape(prox(GroupNormL2(0.3, [i:5:15 for i in 1:5], ones(5)), vec(B), γ), 5, 3)

  # a rule for the diagonal needs square slices
  @test_throws DimensionMismatch prox(FiberSum(NormL2(0.3); diagonal = nothing), randn(3, 4, 2), γ)
  @test_throws DimensionMismatch prox(FiberSum(NormL2(0.3); diagonal = nothing), B, γ)
  @test_throws DimensionMismatch prox!(randn(p, p, 3), fb, X, γ)

  # Float32 stays Float32
  X32 = Float32.(X)
  @test prox(FiberSum(NormL2(0.3f0)), X32, 0.3f0) isa Array{Float32, 3}
  @test FiberSum(NormL2(0.3f0))(X32) isa Float32
end

@testset "NormL1Fused on pairs of arrays and on one pair" begin
  f, γ = NormL1Fused(0.3, 0.5), 0.7
  # a pair of matrices, entry by entry
  A, B = randn(4, 5), randn(4, 5)
  P, Q = prox(f, (A, B), γ)
  @test all((P[i], Q[i]) == prox(f, (A[i], B[i]), γ) for i in eachindex(A))
  @test f((A, B)) ≈ sum(f((A[i], B[i])) for i in eachindex(A))
  @test_throws DimensionMismatch f((A, randn(5, 4)))
  # one pair, as a vector of length 2
  v = [0.9, -0.4]
  y = similar(v)
  @test value_and_prox!(y, f, v, γ) ≈ f(y)
  @test Tuple(y) == prox(f, (v[1], v[2]), γ)
  @test f(v) == f((v[1], v[2]))
  check_prox(f, v, γ)
  @test_throws DimensionMismatch f(randn(3))
  @test_throws DimensionMismatch prox(f, randn(3), γ)
end

end
