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

  # symmetric slices with a rule for the diagonal, through views
  Xs = symslices(p, K)
  for diagonal in (NormL2(0.1), nothing)
    fb = FiberSum(NormL2(0.3); diagonal)
    fs = FiberSum(NormL2(0.3); diagonal, symmetric = true)
    Y, Ys = similar(Xs), similar(Xs)
    v = value_and_prox!(Y, fb, Xs, γ)
    @test value_and_prox!(Ys, fs, Xs, γ) ≈ v
    @test Ys == Y
    @test prox!(similar(Xs), fs, Xs, γ) == Y
    @test fs(Xs) ≈ fb(Xs)
    @test all(Y[i, i, :] == (diagonal === nothing ? Xs[i, i, :] : prox(diagonal, Xs[i, i, :], γ)) for i in 1:p)
    check_prox(fb, Xs, γ)
    @test allocs(a_vprox!, Ys, fs, Xs, γ) == 0
  end

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

# the proximal operator of λ1 ‖z‖₁ + λ2 Σₖ<ₗ |zₖ - zₗ|, a reference: z = v - A'u
# for the u that minimizes ‖v - A'u‖²/2 over |uᵢ| ≤ γ, with A = [λ1 I; λ2 D] and D
# the differences of all pairs, by projected gradient on this dual
function fused_reference(v, λ1, λ2, γ; iters = 200_000)
  K = length(v)
  pairs = [(k, l) for k in 1:K for l in k+1:K]
  D = zeros(length(pairs), K)
  for (r, (k, l)) in enumerate(pairs)
    D[r, k], D[r, l] = 1, -1
  end
  A = vcat(λ1 * Matrix(1.0I, K, K), λ2 * D)
  L = opnorm(A)^2
  u = zeros(size(A, 1))
  for _ in 1:iters
    u .= clamp.(u .+ (A * (v .- A' * u)) ./ L, -γ, γ)
  end
  v .- A' * u
end
fused_direct(v, λ1, λ2) = λ1 * sum(abs, v) + λ2 * sum((abs(v[k] - v[l]) for k in eachindex(v) for l in k+1:length(v)); init = 0.0)

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
end

@testset "NormL1Fused on K entries, all pairs fused" begin
  rng = Xoshiro(5)
  for K in (1, 3, 4, 6), _ in 1:2
    v = 2 .* randn(rng, K)
    λ1, λ2, γ = rand(rng), rand(rng), 0.5 + rand(rng)
    f = NormL1Fused(λ1, λ2)
    y = similar(v)
    @test value_and_prox!(y, f, v, γ) ≈ fused_direct(y, λ1, λ2)
    @test y ≈ fused_reference(v, λ1, λ2, γ) atol = 1e-10
    @test f(v) ≈ fused_direct(v, λ1, λ2)
    @test sortperm(y) == sortperm(v) || issorted(y[sortperm(v)])   # the order of v is kept
    check_prox(f, v, γ)
  end
  # a large fusion penalty fuses all the entries at their mean, then shrinks it
  v = [0.5, 2.0, -1.0, 1.5]
  @test prox(NormL1Fused(0.1, 10.0), v, 1.0) ≈ fill(shrink(sum(v) / 4, 0.1), 4)
  # the closed form for two entries, bit for bit
  f = NormL1Fused(0.3, 0.4)
  @test prox(f, [0.9, -0.2], 0.7) == collect(prox(f, (0.9, -0.2), 0.7))
  # in place, and after a longer vector has grown the buffers
  f = NormL1Fused(0.1, 0.2)
  prox(f, randn(rng, 10), 1.0)
  v = randn(rng, 4)
  z = copy(v)
  @test prox!(z, f, z, 1.0) ≈ fused_reference(v, 0.1, 0.2, 1.0) atol = 1e-10
  @test allocs(a_vprox!, similar(v), f, v, 1.0) == 0
  @test allocs(a_value, f, v) == 0
  @test_throws DimensionMismatch prox!(zeros(3), f, v, 1.0)
  # a long vector takes Base's sort: the steps by hand, with IndIsotonic
  K, λ1, λ2, γ = 5000, 0.01, 0.001, 0.8
  v = randn(rng, K)
  p = sortperm(v)
  w = prox(IndIsotonic(), v[p] .- γ * λ2 .* (2 .* (1:K) .- K .- 1), 1.0)
  z = similar(v)
  z[p] = shrink.(w, γ * λ1)
  sorted_value(u) = λ1 * sum(abs, u) + λ2 * sum((2k - K - 1) * s for (k, s) in enumerate(sort(u)))
  f, y = NormL1Fused(λ1, λ2), similar(v)
  @test value_and_prox!(y, f, v, γ) ≈ sorted_value(y)
  @test y ≈ z
  @test f(v) ≈ sorted_value(v)
  @test allocs(a_vprox!, y, f, v, γ) == 0
end

@testset "FiberSum: K classes all fused, and fusion over time" begin
  p, K, γ = 5, 4, 0.5
  X = symslices(p, K)
  # the fused graphical lasso of K classes, with the diagonal penalized, only
  # fused, or free, through views
  fused = NormL1Fused(0.2, 0.3)
  for diagonal in (fused, NormL1Fused(0.0, 0.3), nothing), symmetric in (false, true)
    fb = FiberSum(fused; diagonal, symmetric)
    Y = similar(X)
    v = value_and_prox!(Y, fb, X, γ)
    @test all(Y[i, j, :] ≈ (i == j && diagonal === nothing ? X[i, i, :] : prox(i == j ? diagonal : fused, X[i, j, :], γ))
              for i in 1:p, j in 1:p)
    @test all(Y[:, :, k] == Y[:, :, k]' for k in 1:K)
    @test v ≈ fb(Y)
    @test allocs(a_vprox!, Y, fb, X, γ) == 0
  end
  check_prox(FiberSum(fused), randn(p, p, K), γ)
  # graphs that change over time with jumps: neighboring slices fused
  for f in (FusedLasso(0.1, 0.4), TotalVariation1D(0.4))
    fb = FiberSum(f; diagonal = nothing, symmetric = true)
    Y = prox(fb, X, γ)
    @test all(Y[i, j, :] ≈ prox(f, X[i, j, :], γ) for i in 1:p, j in 1:p if i != j)
    @test all(Y[i, i, :] == X[i, i, :] for i in 1:p)
    @test allocs(a_vprox!, Y, fb, X, γ) == 0
    check_prox(FiberSum(f), randn(p, p, K), γ)
  end
end

end
