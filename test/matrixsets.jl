module MatrixSetsTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(3)

# references from the definitions
band_ref(X, kl, ku) = [-kl <= c - r <= ku ? X[r, c] : zero(X[r, c]) for r in axes(X, 1), c in axes(X, 2)]
function toeplitz_ref(X; symmetric = false)
  m, n = size(X)
  Z = zeros(float(eltype(X)), m, n)
  for d in -(m - 1):(n - 1)
    μ = symmetric ? (sum(diag(X, d)) + sum(diag(X, -d))) / (2 * length(diag(X, d))) : sum(diag(X, d)) / length(diag(X, d))
    for r in max(1, 1 - d):min(m, n - d)
      Z[r, r + d] = μ
    end
  end
  Z
end

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
a_kkt(f, g, x, k) = kkt_violation(f, g, x, k)

function zero_alloc(f, x, y = similar(x); γ = 0.7)
  @test allocs(a_value, f, x) == 0
  @test allocs(a_prox!, y, f, x, γ) == 0
  @test allocs(a_vprox!, y, f, x, γ) == 0
  x isa Union{Symmetric, Hermitian} && return
  z = copy(x)
  @test allocs(a_prox!, z, f, z, γ) == 0                  # in place
end

@testset "IndPattern" begin
  p = 7
  P = rand(p, p) .< 0.4
  X = randn(p, p)
  f = IndPattern(P)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test prox(f, X) == X .* P
  @test f(X .* P) == 0 && f(X) == Inf
  @test f(X .* P .+ 1e-12) == 0                             # the tolerance of rounding
  @test value_and_prox!(similar(X), f, X, 1.) == 0
  check_prox(f, X, 1.)
  # the same set as a box with array bounds
  box = IndBox(ifelse.(P, -Inf, 0.), ifelse.(P, Inf, 0.))
  @test prox(f, X) == prox(box, X)
  @test f(X) == box(X) && f(prox(f, X)) == box(prox(f, X))
  # the pattern from a BitArray (shared), a sparse matrix (its nonzeros) or an estimate
  @test IndPattern(BitArray(P)).P == P
  @test IndPattern(sparse(P)).P == P
  @test IndPattern(X .* P).P == (X .* P .!= 0)
  Xs = sparse(X .* P)
  Xs.nzval[1] = 0                                           # a stored zero is not in the pattern
  @test IndPattern(Xs).P == (Matrix(Xs) .!= 0)
  # Float32
  @test prox(f, Float32.(X)) isa Matrix{Float32}
  @test prox(f, Float32.(X)) == Float32.(X .* P)

  # vectors and SparseIterates: the O(nnz) paths
  n = 40
  q = rand(n) .< 0.5
  fv = IndPattern(q)
  x = randn(n)
  xs = SparseIterate(sprandn(n, 0.3))
  @test Vector(prox(fv, xs)) == prox(fv, Vector(xs))
  @test fv(xs) == fv(Vector(xs))
  ws = copy(xs)
  prox!(ws, fv, ws, 1.)
  @test Vector(ws) == Vector(xs) .* q
  @test IndPattern(xs).P == (Vector(xs) .!= 0)
  # coordinate steps
  k = findfirst(!, q)
  xc = SparseIterate(copy(x))
  @test prox_coordinate!(xc, fv, k, 0.5) == 0 && xc[k] == 0
  j = findfirst(q)
  @test prox_coordinate!(xc, fv, j, 0.5) == x[j]
  xa = AtomIterate(copy(x), [1:10, 11:20, 21:30, 31:40])
  prox_coordinate!(xa, fv, 2, 0.5)
  @test Vector(xa)[11:20] == x[11:20] .* q[11:20]
  # optimality at zero and restriction
  g = randn(n)
  @test kkt_violation(fv, g, j) == abs(g[j]) && kkt_violation(fv, g, k) == -Inf
  @test kkt_violation(fv, g, x, j) == abs(g[j])
  S = [3, 8, 15, 22, 40]
  @test prox(restrict(fv, S), x[S]) == prox(fv, x)[S]

  # symmetric matrices: a symmetric pattern on a SymmetricSparseIterate
  Ps = P .| P'
  fs = IndPattern(Ps)
  D = randn(p, p); D = D + D'
  Sd = SymmetricSparseIterate(D)
  @test Matrix(prox(fs, Sd)) == D .* Ps
  @test fs(Sd) == Inf && fs(prox(fs, Sd)) == 0
  @test prox(fs, Symmetric(D)) == D .* Ps
  Sc = copy(Sd)
  for kc in 1:ncoordinates(Sc)
    r, c = tril_cartesian(p, kc)
    @test prox_coordinate!(Sc, fs, kc, 1.) == (Ps[r, c] ? D[r, c] : 0)
  end
  @test Matrix(Sc) == D .* Ps
  @test IndPattern(SymmetricSparseIterate(D .* Ps)).P == (D .* Ps .!= 0)

  @testset "allocations" begin
    zero_alloc(f, X)
    zero_alloc(f, Float32.(X))
    zero_alloc(fv, xs)
    zero_alloc(fs, Sd)
    @test allocs(a_cd!, copy(xs), fv, 3, 0.7) == 0
    @test allocs(a_cd!, copy(Sd), fs, 5, 0.7) == 0
    @test allocs(a_kkt, fv, g, x, 4) == 0
  end
end

@testset "IndBanded" begin
  for (m, n) in ((6, 6), (5, 8), (8, 5)), kl in (0, 1, 3, 9), ku in (0, 2, 9)
    X = randn(m, n)
    f = IndBanded((kl, ku), (m, n))
    ref = band_ref(X, kl, ku)
    @test prox(f, X) == ref
    @test f(ref) == 0
    @test f(X) == (ref == X ? 0 : Inf)
    # the same set as the pattern of the band
    band = band_ref(trues(m, n), kl, ku)
    @test prox(IndPattern(band), X) == ref
    # a SparseIterate, by its stored entries
    Xs = SparseIterate(sparse(X .* (rand(m, n) .< 0.4)))
    @test Matrix(prox(f, Xs)) == band_ref(Matrix(Xs), kl, ku)
    @test f(Xs) == f(Matrix(Xs))
  end
  p = 7
  f = IndBanded(2, p)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test IndBanded(2, (p, p)) == f
  X = randn(p, p)
  check_prox(f, X, 1.)
  z = copy(X)
  prox!(z, f, z, 1.)                                        # in place
  @test z == band_ref(X, 2, 2)
  @test value_and_prox!(similar(X), f, X, 1.) == 0
  @test prox(f, Float32.(X)) == Float32.(band_ref(X, 2, 2))
  # symmetric inputs
  D = randn(p, p); D = D + D'
  @test prox(f, Symmetric(D)) == band_ref(D, 2, 2)
  Sd = SymmetricSparseIterate(D)
  @test Matrix(prox(f, Sd)) == band_ref(D, 2, 2)
  @test f(Sd) == Inf && f(prox(f, Sd)) == 0
  Sc = copy(Sd)
  for kc in 1:ncoordinates(Sc)
    r, c = tril_cartesian(p, kc)
    @test prox_coordinate!(Sc, f, kc, 1.) == (r - c <= 2 ? D[r, c] : 0)
  end
  @test Matrix(Sc) == band_ref(D, 2, 2)
  @test_throws ArgumentError prox(IndBanded((1, 2), (p, p)), Sd)
  # the entry with linear index i of a 5 × 8 matrix
  g = randn(5, 8)
  fb = IndBanded((1, 2), (5, 8))
  for i in eachindex(g)
    r, c = Tuple(CartesianIndices(g)[i])
    inband = -1 <= c - r <= 2
    @test kkt_violation(fb, g, i) == (inband ? abs(g[i]) : -Inf)
    @test prox_entry(fb, 1.5, i, 1.) == (inband ? 1.5 : 0)
  end
  S = [1, 7, 12, 30, 40]
  @test restrict(fb, S) isa IndPattern
  @test prox(restrict(fb, S), g[S]) == prox(fb, g)[S]
  # errors
  @test_throws DimensionMismatch prox(f, randn(p, p + 1))
  @test_throws DimensionMismatch f(randn(p))
  @test_throws ArgumentError IndBanded(-1, p)
  @test_throws ArgumentError IndBanded((1, -1), (p, p))

  @testset "allocations" begin
    zero_alloc(f, X)
    zero_alloc(f, Float32.(X))
    zero_alloc(f, Sd)
    zero_alloc(f, Symmetric(D), zeros(p, p))
    Xs = SparseIterate(sparse(X .* (rand(p, p) .< 0.4)))
    zero_alloc(f, Xs)
    @test allocs(a_cd!, copy(Sd), f, 5, 0.7) == 0
    @test allocs(a_kkt, f, X, X, 4) == 0
  end
end

@testset "IndToeplitz" begin
  for (m, n) in ((5, 5), (4, 7), (7, 4), (1, 3))
    X = randn(m, n)
    f = IndToeplitz()
    P = prox(f, X)
    @test P ≈ toeplitz_ref(X)
    @test f(P) == 0 && f(X) == (min(m, n) == 1 ? 0 : Inf)       # a row or a column is Toeplitz
    @test prox(f, P) ≈ P                                    # a projection
    # X - P is orthogonal to the Toeplitz matrices, a subspace
    Z = toeplitz_ref(randn(m, n))
    @test abs(dot(X - P, Z)) < 1e-10 * norm(X) * norm(Z)
    check_prox(f, X, 1.)
  end
  f, fs = IndToeplitz(), IndToeplitz(symmetric = true)
  check_traits(f; proximable = true, convex = true)
  p = 6
  X = randn(p, p)
  Ps = prox(fs, X)
  @test Ps ≈ toeplitz_ref(X; symmetric = true)
  @test issymmetric(Ps)
  @test fs(Ps) == 0 && fs(prox(f, X)) == Inf                # a Toeplitz matrix that is not symmetric
  check_prox(fs, X, 1.; symmetric = true)
  @test fs(randn(p, p + 1)) == Inf
  @test_throws DimensionMismatch prox(fs, randn(p, p + 1))
  # a symmetric input: both projections are symmetric and agree
  D = randn(p, p); D = D + D'
  @test prox(f, Symmetric(D)) ≈ toeplitz_ref(D)
  @test prox(f, Symmetric(D)) ≈ prox(fs, D)
  # outputs: in place, a SymmetricSparseIterate, Float32
  z = copy(X)
  prox!(z, f, z, 1.)
  @test z ≈ toeplitz_ref(X)
  Sout = SymmetricSparseIterate(p)
  @test Matrix(prox!(Sout, fs, X, 1.)) ≈ Ps
  @test Matrix(prox!(SymmetricSparseIterate(p), f, SymmetricSparseIterate(D), 1.)) ≈ toeplitz_ref(D)
  @test_throws ArgumentError prox!(SymmetricSparseIterate(p), f, X, 1.)
  @test prox(f, Float32.(X)) isa Matrix{Float32}
  @test prox(f, Float32.(X)) ≈ Float32.(toeplitz_ref(X))
  @test value_and_prox!(similar(X), f, X, 1.) == 0

  @testset "allocations" begin
    zero_alloc(f, X)
    zero_alloc(fs, X)
    zero_alloc(f, Float32.(X))
    zero_alloc(f, Symmetric(D), zeros(p, p))
    @test allocs(a_prox!, Sout, fs, X, 0.7) == 0
  end
end

end
