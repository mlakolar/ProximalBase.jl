module PartialTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

orth(m, k) = Matrix(qr(randn(m, k)).Q)
# a rank-k signal with singular values in [10, 20], plus noise of spectral norm about 0.2
lowrank(m, n, k) = orth(m, k) * Diagonal(range(20, 10; length = k)) * orth(n, k)' + 0.1 * randn(m, n) / sqrt(max(m, n))
function symlowrank(n, k)
  G = randn(n, n) / sqrt(2n)
  V = orth(n, k)
  V * Diagonal(range(20, 10; length = k)) * V' + 0.1 * (G + G')
end

# references through LinearAlgebra
nuclear_prox(X, c) = (F = svd(X); F.U * Diagonal(max.(F.S .- c, 0)) * F.Vt)
rank_proj(X, r) = (F = svd(X); F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :])
psd_proj(X, f = t -> max(t, 0)) = (E = eigen(Symmetric(X)); E.vectors * Diagonal(f.(E.values)) * E.vectors')
function psd_rank_proj(X, r)
  E = eigen(Symmetric(X))
  n = size(X, 1)
  V = E.vectors[:, n-r+1:n]
  V * Diagonal(max.(E.values[n-r+1:n], 0)) * V'
end

# whether f ran a full decomposition since forget!(f): the full workspaces are
# created only then
const FULL = (:ws, :eig, :top)
forget!(f) = (foreach(s -> hasproperty(f, s) && setproperty!(f, s, nothing), FULL); f)
usedfull(f) = any(s -> hasproperty(f, s) && getproperty(f, s) !== nothing, FULL)

@testset "thin QR decomposition" begin
  for T in (Float64, Float32)
    tol = T == Float64 ? 1e-12 : 1e-5
    A = randn(T, 30, 7)
    ws = PB.QRWorkspace(T, 30, 7)
    Q, R = copy(A), zeros(T, 7, 7)
    PB._thin_qr!(Q, ws, R)
    @test Q' * Q ≈ I atol = tol
    @test Q * R ≈ A rtol = tol
    @test istriu(R)
    B = randn(T, 30, 2) * randn(T, 2, 7)                  # rank deficient: Q is still orthonormal
    PB._thin_qr!(B, ws)
    @test B' * B ≈ I atol = tol
    f(Q, ws, R) = PB._thin_qr!(Q, ws, R)
    @test allocs(f, Q, ws, R) == 0
  end
  @test_throws ArgumentError PB.QRWorkspace(Float64, 3, 4)
end

@testset "the random block" begin
  ws1, ws2 = PB.PartialSVDWorkspace(Float64, 40, 30, 12), PB.PartialSVDWorkspace(Float64, 40, 30, 12)
  A, B = zeros(40, 12), zeros(40, 12)
  PB._randomize!(A, 1:12, ws1)
  PB._randomize!(B, 1:12, ws2)
  @test A == B                                             # the same numbers in every session
  @test all(x -> -1 <= x < 1, A)
  @test abs(sum(A) / length(A)) < 0.1
  PB._randomize!(B, 1:12, ws2)
  @test A != B                                             # the state advances
end

@testset "NuclearNorm, partial" begin
  for (m, n, k) in ((200, 120, 3), (120, 200, 3), (150, 150, 8))
    X = lowrank(m, n, k)
    f = NuclearNorm(0.5; partial = true)
    Y = similar(X)
    for γ in (2.0, 1.0, 4.0)
      forget!(f)
      v = value_and_prox!(Y, f, X, γ)
      @test Y ≈ nuclear_prox(X, 0.5γ) rtol = 1e-10
      @test v ≈ 0.5 * sum(max.(svdvals(X) .- 0.5γ, 0)) rtol = 1e-10
      @test !usedfull(f)                                   # the partial SVD was enough
    end
    @test f.pws.rank == k
    # a sequence of nearby matrices, as a solver calls it
    E = randn(m, n) / sqrt(n)
    for t in 1:5
      Xt = X + 1e-3 * t * E
      @test prox(f, Xt, 2.0) ≈ nuclear_prox(Xt, 1.0) rtol = 1e-10
    end
    @test !usedfull(f)
    check_prox(f, X, 2.0)
  end
  X = lowrank(160, 120, 3)
  # in place, and inputs and outputs that are not strided
  f = NuclearNorm(1.0; partial = true)
  Z = copy(X)
  @test prox!(Z, f, Z, 1.0) ≈ nuclear_prox(X, 1.0) rtol = 1e-10
  @test Matrix(prox!(SparseIterate(160, 120), f, X, 1.0)) ≈ nuclear_prox(X, 1.0) rtol = 1e-10
  @test prox(f, SparseIterate(X), 1.0) ≈ nuclear_prox(X, 1.0) rtol = 1e-10
  @test f(X) ≈ sum(svdvals(X))                             # the value alone takes the full SVD
  # an exactly low-rank matrix, and zero
  X0 = orth(160, 3) * Diagonal([5.0, 4.0, 3.0]) * orth(120, 3)'
  @test prox(forget!(f), X0, 1.0) ≈ nuclear_prox(X0, 1.0) rtol = 1e-10
  @test prox(f, zeros(160, 120), 1.0) == zeros(160, 120)
  @test !usedfull(f)
  # a rank above the first block: the block grows
  Xg = lowrank(200, 150, 20)
  fg = NuclearNorm(1.0; partial = true)
  @test prox(fg, Xg, 1.0) ≈ nuclear_prox(Xg, 1.0) rtol = 1e-10
  @test !usedfull(fg) && fg.pws.rank == 20
  # too many singular values above the threshold: the full SVD, then not even an attempt
  R = randn(80, 60)
  fr = NuclearNorm(0.01; partial = true)
  @test prox(fr, R, 1.0) ≈ nuclear_prox(R, 0.01) rtol = 1e-10
  @test usedfull(fr)
  R2 = R + 0.01 * randn(80, 60)
  forget!(fr)
  @test prox(fr, R2, 1.0) ≈ nuclear_prox(R2, 0.01) rtol = 1e-10
  @test usedfull(fr)
  # small matrices take the full SVD
  fs = NuclearNorm(1.0; partial = true)
  Xs = randn(12, 8)
  @test prox(fs, Xs, 1.0) ≈ nuclear_prox(Xs, 1.0)
  @test fs.pws === nothing
  # Float32
  X32 = Float32.(X)
  f32 = NuclearNorm(1.0f0; partial = true)
  @test prox(f32, X32, 1.0f0) ≈ nuclear_prox(X32, 1.0f0) rtol = 1e-4
  @test f32.pws isa PB.PartialSVDWorkspace{Float32} && !usedfull(f32)
end

# A first call starts from a random block, whose Ritz values are small. With p
# large and signals just above the threshold, such a block passes the test of
# the pairs below the threshold, and power steps must come first.
@testset "a first call from a random block" begin
  p = 1500
  V = orth(p, 2)
  G = randn(p, p) / sqrt(2p)
  S = V * Diagonal([3.0, 2.0]) * V' + 0.1 * (G + G')
  for f in (TracePSD(1.0; partial = true), IndPSD(rank = 2, partial = true))
    Y = prox(f, S, 1.0)
    @test Y ≈ (f isa TracePSD ? psd_proj(S, t -> max(t - 1, 0)) : psd_rank_proj(S, 2)) rtol = 1e-10
    @test !usedfull(f)
  end
  X = orth(p, 2) * Diagonal([3.0, 2.0]) * orth(1000, 2)' + 0.1 * randn(p, 1000) / sqrt(p)
  for f in (NuclearNorm(1.0; partial = true), IndBallRank(2; partial = true))
    Y = prox(f, X, 1.0)
    @test Y ≈ (f isa NuclearNorm ? nuclear_prox(X, 1.0) : rank_proj(X, 2)) rtol = 1e-10
    @test !usedfull(f)
  end
end

@testset "IndBallRank, partial" begin
  X = lowrank(100, 70, 4)
  f = IndBallRank(4; partial = true)
  @test prox(f, X) ≈ rank_proj(X, 4) rtol = 1e-10
  @test !usedfull(f)
  E = randn(100, 70) / sqrt(70)
  for t in 1:4
    Xt = X + 1e-3 * t * E
    @test prox(f, Xt) ≈ rank_proj(Xt, 4) rtol = 1e-10
  end
  @test !usedfull(f)
  check_prox(f, X, 1.0)
  @test value_and_prox!(similar(X), f, X, 1.0) == 0
  # a rank below that of the signal, and no gap at all: correct either way
  @test prox(IndBallRank(2; partial = true), X) ≈ rank_proj(X, 2) rtol = 1e-10
  R = randn(100, 70)
  @test prox(IndBallRank(3; partial = true), R) ≈ rank_proj(R, 3) rtol = 1e-8
  @test prox(IndBallRank(4; partial = true), Float32.(X)) ≈ rank_proj(Float32.(X), 4) rtol = 1e-4
end

@testset "IndPSD, partial" begin
  n = 120
  S = symlowrank(n, 4)
  # with a rank
  f = IndPSD(rank = 4, partial = true)
  @test prox(f, S) ≈ psd_rank_proj(S, 4) rtol = 1e-10
  @test !usedfull(f)
  G = randn(n, n) / sqrt(2n)
  for t in 1:4
    St = S + 1e-3 * t * (G + G')
    @test prox(f, St) ≈ psd_rank_proj(St, 4) rtol = 1e-10
  end
  @test !usedfull(f)
  check_prox(f, S, 1.0; symmetric = true)
  @test prox(IndPSD(rank = 2, partial = true), S) ≈ psd_rank_proj(S, 2) rtol = 1e-10
  # without a rank, when few eigenvalues are positive
  Sm = S - 0.5I
  g = IndPSD(partial = true)
  @test prox(g, Sm) ≈ psd_proj(Sm) rtol = 1e-10
  @test !usedfull(g)
  check_prox(g, Sm, 1.0; symmetric = true)
  @test g(prox(g, Sm)) == 0
  # eigenvalues that are negative and large: the shifted iteration still finds the positive ones
  Q = orth(n, 20)
  C = S - 30 * Q * Q'
  for h in (IndPSD(partial = true), IndPSD(rank = 4, partial = true))
    @test prox(h, C) ≈ (h.rank === nothing ? psd_proj(C) : psd_rank_proj(C, 4)) rtol = 1e-9
  end
  # half the eigenvalues positive: the full eigendecomposition, correctly
  D = randn(n, n); D = (D + D') / 2
  h = IndPSD(partial = true)
  @test prox(h, D) ≈ psd_proj(D) rtol = 1e-10
  @test usedfull(h)
  # symmetric sparse iterates in and out, in place, and Float32
  @test prox(g, SymmetricSparseIterate(Sm)) ≈ psd_proj(Sm) rtol = 1e-10
  @test Matrix(prox!(SymmetricSparseIterate(n), g, Sm, 1.0)) ≈ psd_proj(Sm) rtol = 1e-10
  Z = copy(Sm)
  @test prox!(Z, g, Z, 1.0) ≈ psd_proj(Sm) rtol = 1e-10
  g32 = IndPSD(partial = true)
  @test prox(g32, Float32.(Sm)) ≈ psd_proj(Float32.(Sm)) rtol = 1e-4
  @test g32.pws isa PB.PartialEigenWorkspace{Float32} && !usedfull(g32)
end

@testset "TracePSD, partial" begin
  n = 120
  S = symlowrank(n, 5)
  f = TracePSD(0.3; partial = true)
  for γ in (1.0, 3.0)
    forget!(f)
    Y = similar(S)
    v = value_and_prox!(Y, f, S, γ)
    @test Y ≈ psd_proj(S, t -> max(t - 0.3γ, 0)) rtol = 1e-10
    @test v ≈ 0.3 * tr(Y) rtol = 1e-10
    @test !usedfull(f)
  end
  check_prox(f, S, 1.0; symmetric = true)
  # all eigenvalues below the threshold
  @test prox(f, S, 1000.0) ≈ zeros(n, n) atol = 1e-12
  # a rank above the first block: the block grows
  Sg = symlowrank(200, 20)
  fg = TracePSD(1.0; partial = true)
  @test prox(fg, Sg, 1.0) ≈ psd_proj(Sg, t -> max(t - 1.0, 0)) rtol = 1e-10
  @test !usedfull(fg) && fg.pws.rank == 20
end

end
