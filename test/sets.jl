module SetsTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# projections by sorting
function simplex_proj(y, a)
  u = sort(y; rev = true)
  c, τ = 0.0, 0.0
  for (j, uj) in enumerate(u)
    c += uj
    t = (c - a) / j
    uj > t && (τ = t)
  end
  max.(y .- τ, 0)
end
l1ball_proj(y, r) = norm(y, 1) <= r ? y : sign.(y) .* simplex_proj(abs.(y), r)

@testset "IndBallL0" begin
  x = [0.5, -2., 0.1, 1.2, -0.3]
  f = IndBallL0(2)
  @test prox(f, x) == [0., -2., 0., 1.2, 0.]
  @test f(x) == Inf && f(prox(f, x)) == 0
  @test prox(IndBallL0(5), x) == x
  @test prox(IndBallL0(0), x) == zeros(5)
  # ties: exactly s entries are kept
  y = [1., -1., 1., 0.5, 1.]
  z = prox(IndBallL0(2), y)
  @test count(!iszero, z) == 2 && all(abs.(z[z .!= 0]) .== 1)
  for _ in 1:20
    v = rand(-3:3, 30) .+ 0.0
    @test count(!iszero, prox(IndBallL0(7), v)) == min(7, count(!iszero, v))
  end
  x = randn(40)
  z = check_prox(IndBallL0(8), x, 1.)
  @test sort(abs.(z[z .!= 0])) == sort(abs.(x); rev = true)[1:8] |> sort
  @test value_and_prox!(similar(x), IndBallL0(8), x, 1.) == 0
  # matrices: the s largest entries
  M = randn(6, 5)
  Z = prox(IndBallL0(4), M)
  @test count(!iszero, Z) == 4 && minimum(abs, Z[Z .!= 0]) >= sort(abs.(vec(M)); rev = true)[4]
  # a SparseIterate: the stored entries hold every nonzero
  xs = SparseIterate(sprandn(100, 0.2))
  for s in (0, 3, 10, 50)
    zs = prox(IndBallL0(s), xs)
    @test Vector(zs) == prox(IndBallL0(s), Vector(xs))
    ws = copy(xs)
    prox!(ws, IndBallL0(s), ws, 1.)                       # in place
    @test Vector(ws) == Vector(zs)
  end
  @test IndBallL0(3)(xs) == (count(!iszero, Vector(xs)) <= 3 ? 0 : Inf)
  # a SymmetricSparseIterate: s counts the coordinates, the entries of the lower triangle
  p = 6
  D = randn(p, p); D = D + D'
  S = SymmetricSparseIterate(D)
  Z = prox(IndBallL0(5), S)
  @test nnz(dropzeros!(copy(Z))) == 5
  @test Matrix(Z) == Matrix(Z)'                          # an entry and its transpose together
  kept = sort([abs(Z[r, c]) for c in 1:p for r in c:p if Z[r, c] != 0])
  @test kept == sort([abs(D[r, c]) for c in 1:p for r in c:p]; rev = true)[1:5] |> sort
  @test IndBallL0(5)(Z) == 0 && IndBallL0(4)(Z) == Inf
  @test_throws ArgumentError IndBallL0(-1)
end

# the s largest entries of the lower triangle, kept with their transposes; with
# diagonal = false the diagonal is kept and does not count
function sym_hard_threshold(X, s, diagonal)
  p = size(X, 1)
  cand = [(abs(X[r, c]), r, c) for c in 1:p for r in (diagonal ? c : c + 1):p]
  sort!(cand; rev = true, by = first)
  Z = zeros(p, p)
  for (_, r, c) in cand[1:min(s, length(cand))]
    Z[r, c] = Z[c, r] = X[r, c]
  end
  diagonal || for i in 1:p
    Z[i, i] = X[i, i]
  end
  Z
end

@testset "IndBallL0 on symmetric matrices" begin
  p = 7
  D = randn(p, p); D = (D + D') / 2
  for diagonal in (true, false), s in (0, 1, 4, 10, 40)
    f = IndBallL0(s; diagonal)
    ref = sym_hard_threshold(D, s, diagonal)
    # a dense symmetric input, a dense or a sparse symmetric output
    @test prox!(zeros(p, p), f, Symmetric(D), 1.) == ref
    @test prox(f, Symmetric(D)) == ref
    @test Matrix(prox!(SymmetricSparseIterate(p), f, Symmetric(D, :L), 1.)) == ref
    # a SymmetricSparseIterate, also in place
    S = SymmetricSparseIterate(D)
    @test Matrix(prox(f, S)) == ref
    prox!(S, f, S, 1.)
    @test Matrix(S) == ref
    @test f(S) == 0 && f(Symmetric(ref)) == 0
  end
  @test IndBallL0(3; diagonal = false)(Symmetric(D)) == Inf
  # the diagonal does not count for a dense matrix either, whose entries count one by one
  M = randn(4, 4)
  Z = prox(IndBallL0(2; diagonal = false), M)
  @test diag(Z) == diag(M)
  @test count(!iszero, Z - Diagonal(diag(Z))) == 2
  @test IndBallL0(2; diagonal = false)(Z) == 0 && IndBallL0(2)(Z) == Inf
  # the use in iterative hard thresholding: no allocation
  a(S, f, X) = @allocated prox!(S, f, X, 1.)
  S, f, X = SymmetricSparseIterate(p), IndBallL0(5; diagonal = false), Symmetric(D, :L)
  a(S, f, X)
  @test a(S, f, X) == 0
end

@testset "IndBallL1" begin
  for n in (1, 5, 50), r in (0.5, 2.)
    x = 3 .* randn(n)
    f = IndBallL1(r)
    z = prox(f, x)
    @test z ≈ l1ball_proj(x, r)
    @test norm(z, 1) <= r * (1 + 1e-12) || norm(x, 1) <= r
    @test f(z) == 0
    check_prox(f, x, 1.)
  end
  @test prox(IndBallL1(10.), [1., -2.]) == [1., -2.]       # inside
  @test prox(IndBallL1(0.), [1., -2.]) == [0., 0.]
  @test IndBallL1(1.)([0.5, 0.5 + 1e-12]) == 0             # the tolerance of rounding
  @test IndBallL1(1.)([0.5, 0.6]) == Inf
  M = randn(4, 6)
  @test prox(IndBallL1(1.), M) ≈ reshape(l1ball_proj(vec(M), 1.), 4, 6)
  @test prox(IndBallL1(1.), Float32.(M)) isa Matrix{Float32}
  @test_throws ArgumentError IndBallL1(-1.)
end

@testset "IndBallL2" begin
  x = [3., 4.]
  @test prox(IndBallL2(1.), x) ≈ [0.6, 0.8]
  @test prox(IndBallL2(10.), x) == x
  @test IndBallL2(5.)(x) == 0 && IndBallL2(4.9)(x) == Inf
  for x in (randn(10), randn(3, 4))
    check_prox(IndBallL2(0.7), x, 1.)
  end
end

@testset "IndSimplex" begin
  for n in (1, 4, 50), r in (1., 3.)
    x = randn(n)
    z = prox(IndSimplex(r), x)
    @test z ≈ simplex_proj(x, r)
    @test all(z .>= 0) && sum(z) ≈ r
    check_prox(IndSimplex(r), x, 1.)
    # with an inequality, points already inside after clipping stay
    zi = prox(IndSimplex(r; equality = false), x)
    @test zi ≈ (sum(max.(x, 0)) <= r ? max.(x, 0) : simplex_proj(x, r))
    check_prox(IndSimplex(r; equality = false), x, 1.)
  end
  @test IndSimplex(1.)([0.5, 0.5]) == 0 && IndSimplex(1.)([0.5, 0.6]) == Inf
  @test IndSimplex(1.; equality = false)([0.2, 0.3]) == 0
  @test IndSimplex(1.)([1.5, -0.5]) == Inf
  @test prox(IndSimplex(0.), [1., 2.]) == [0., 0.]
end

@testset "hyperplanes and halfspaces" begin
  a, b = randn(6), 0.7
  x = randn(6)
  z = prox(IndHyperplane(a, b), x)
  @test dot(a, z) ≈ b
  @test z ≈ x - (dot(a, x) - b) / dot(a, a) * a
  check_prox(IndHyperplane(a, b), x, 1.)
  h = IndHalfspace(a, b)
  inside = x .- (dot(a, x) - b + 1) / dot(a, a) .* a      # ⟨a, x⟩ = b - 1
  @test prox(h, inside) ≈ inside
  outside = x .- (dot(a, x) - b - 1) / dot(a, a) .* a
  @test dot(a, prox(h, outside)) ≈ b
  check_prox(h, outside, 1.)
  @test h(inside) == 0 && h(outside) == Inf
  @test_throws ArgumentError IndHyperplane(zeros(3), 1.)
  # matrices, with the trace inner product
  A = randn(3, 3)
  Z = prox(IndHyperplane(A, 2.), randn(3, 3))
  @test dot(A, Z) ≈ 2.
end

end
