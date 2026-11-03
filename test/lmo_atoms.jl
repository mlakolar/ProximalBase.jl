module LMOAtomsTest

using Test
using LinearAlgebra
using SparseArrays
using Random
using ProximalBase

using ..TestUtils

Random.seed!(518)

# the box of matrices ‖s‖_max ≤ c, defined by the user with an oracle only
struct MatrixCube <: ProximalBase.AbstractFunction
  c::Float64
end
ProximalBase.isconvex(::Type{MatrixCube}) = true
ProximalBase._isindicator(::Type{MatrixCube}) = true
ProximalBase.haslmo(::MatrixCube) = true
function ProximalBase.lmo!(s::AbstractArray, C::MatrixCube, g::AbstractArray)
  for i in eachindex(g)
    s[i] = g[i] > 0 ? -C.c : C.c
  end
  s
end

const Fantope = (k) -> Spectral(Constrained(IndBox(0, 1), IndHyperplane(1, k)))

# a random matrix with the given singular values or eigenvalues, from the same generator for every test
function with_singular_values(::Type{T}, m, n, σ) where {T}
  U, V = Matrix(qr(randn(T, m, length(σ))).Q), Matrix(qr(randn(T, n, length(σ))).Q)
  U * Diagonal(T.(σ)) * V'
end
function with_eigenvalues(::Type{T}, λ) where {T}
  V = Matrix(qr(randn(T, length(λ), length(λ))).Q)
  A = V * Diagonal(T.(λ)) * V'
  (A + A') / 2
end

# the nuclear norm of a rank-one matrix a b' is ‖a‖‖b‖
nuclear_norm(a::LowRankAtom) = sum(svdvals(Array(a)))
opnorm_ref(G) = svdvals(Matrix(G))[1]

@testset "vertex atoms agree with lmo! (simplex, ℓ1 ball)" begin
  for T in (Float32, Float64), dims in ((7,), (3, 4), (2, 3, 2)), r in (0.0, 0.7, 3.0)
    for C in (IndSimplex(r), IndSimplex(r; equality = false), IndBallL1(r))
      for trial in 1:6
        g = randn(T, dims)
        trial == 1 && (g[1] = g[end])         # a tie
        trial == 2 && (g .= abs.(g))          # a positive minimum: the origin of the inequality simplex
        trial == 3 && fill!(g, zero(T))
        a = lmo_atom(C, g)
        @test a isa VertexAtom{T, length(dims)}
        @test size(a) == dims
        @test Array(a) == lmo(C, g)
        @test dot(g, a) == dot(g, lmo(C, g))
        @test eltype(a) == T
      end
    end
  end
  # ties go to the first index, and the atom is O(1)
  a = lmo_atom(IndSimplex(2.0), [3.0, 1.0, 1.0, 5.0])
  @test (a.index, a.value) == (2, 2.0)
  @test sizeof(a) <= 6 * sizeof(Int)
  @test lmo_atom(IndBallL1(2.0), [1.0, -3.0, 3.0]).value == 2.0     # -sign(g_i) r, i = 2
  @test lmo_atom(IndSimplex(1.0; equality = false), [0.0, 1.0]).index == 0
  @test lmo_atom(IndSimplex(1.0), zeros(3)) == VertexAtom{Float64, 1}((3,), 1, 1.0)
  @test lmo_atom(IndBallL1(1.0), zeros(3)).index == 0
  @test lmo_atom(IndBallL1(0.0), [1.0, 2.0]).index == 0
  @test lmo_atom(IndSimplex(1.0), [1, 2, 0]).value === 1.0           # integers promote to floats
  @test lmo_atom(IndSimplex(1.0), [-Inf, 1.0]).index == 1            # the limit is determined
  @test_throws ArgumentError lmo_atom(IndSimplex(1.0), [1.0, NaN])
  @test_throws ArgumentError lmo_atom(IndSimplex(Inf), [1.0, 2.0])
  @test_throws ArgumentError lmo_atom(IndBox(0, 1), [1.0, 2.0])
  @test_throws ArgumentError lmo_atom(IndSimplex(1.0), [1.0]; method = :krylov)
  @test_throws ArgumentError lmo_atom(IndSimplex(1.0), [1.0]; rtol = 1e-3)
  @test_throws ArgumentError lmo_atom(IndSimplex(1.0), [1.0]; method = :other)
  @test_throws ArgumentError lmo_atom(IndSimplex(1.0), Float64[])       # no simplex of size 0
  @test lmo_atom(IndSimplex(0.0), Float64[]).index == 0
  # a weight vector that does not fit the element type, as in lmo!
  @test_throws ArgumentError lmo_atom(IndSimplex(1e300), Float32[1, 2])
  # axpy! and dot
  x = zeros(3, 4)
  a = lmo_atom(IndBallL1(2.0), reshape(collect(1.0:12.0), 3, 4))
  axpy!(0.5, a, x)
  @test x == 0.5 * Array(a)
  @test_throws DimensionMismatch dot(zeros(3, 3), a)
  @test_throws DimensionMismatch axpy!(1.0, a, zeros(4, 3))
end

@testset "nuclear-norm ball: atom, dense oracle, support value" begin
  for T in (Float32, Float64), (m, n) in ((12, 12), (30, 9), (9, 30)), r in (0.5, 4.0)
    C = SingularValue(IndBallL1(r))
    @test haslmo(C)
    G = randn(T, m, n)
    σ1 = opnorm_ref(G)
    tol = 100 * eps(T) * r * σ1
    a = lmo_atom(C, G)
    @test a isa LowRankAtom{T}
    @test size(a) == (m, n) && size(a.U) == (m, 1) && size(a.V) == (n, 1)
    S = Array(a)
    @test sum(svdvals(S)) <= r * (1 + 20 * eps(T))                    # feasible (rank one: ‖S‖_* = r)
    @test abs(dot(G, a) + r * σ1) <= tol                              # objective -r σ₁
    @test abs(dot(G, S) + r * σ1) <= tol                              # the same through the dense matrix
    @test lmo(C, G) == S                                              # lmo! materializes the atom
    @test abs(Conjugate(C)(G) - r * σ1) <= tol                        # the support value, r σ₁
    @test abs(-dot(G, a) - Conjugate(C)(G)) <= tol
    # lmo! into a view, and with the output the gradient
    B = zeros(T, m + 2, n)
    lmo!(view(B, 2:m + 1, :), C, G)
    @test B[2:m + 1, :] == S && all(iszero, B[1, :]) && all(iszero, B[end, :])
    H = copy(G)
    @test lmo!(H, C, H) === H
    @test H == S
  end
  C = SingularValue(IndBallL1(2.0))
  # zero gradient and zero radius: the zero atom
  @test size(lmo_atom(C, zeros(4, 3)).U) == (4, 0)
  @test Array(lmo_atom(C, zeros(4, 3))) == zeros(4, 3)
  @test Array(lmo_atom(SingularValue(IndBallL1(0.0)), randn(4, 3))) == zeros(4, 3)
  @test Array(lmo_atom(SingularValue(IndBallL1(0.0)), fill(NaN, 2, 2))) == zeros(2, 2)    # as lmo!: radius 0, whatever g holds
  @test dot(randn(4, 3), lmo_atom(C, zeros(4, 3))) == 0
  # an empty matrix
  @test size(lmo_atom(C, zeros(0, 3))) == (0, 3)
  # the sign convention and determinism
  G = randn(8, 5)
  @test lmo_atom(C, G).U == lmo_atom(C, G).U && lmo_atom(C, copy(G)).V == lmo_atom(C, G).V
  v = lmo_atom(C, G).V[:, 1]
  @test v[argmax(abs.(v))] > 0
end

@testset "nuclear-norm ball: degenerate spectra" begin
  r = 1.5
  C = SingularValue(IndBallL1(r))
  for (name, σ) in (("rank one", [3.0]), ("rank deficient", [3.0, 2.0, 0.5]), ("repeated", [2.0, 2.0, 2.0, 0.1]),
                    ("clustered", [1.0, 1 - 1e-9, 1 - 2e-9, 0.2]), ("ill conditioned", [1.0, 1e-14]))
    for (m, n) in ((12, 9), (9, 12), (10, 10))
      G = with_singular_values(Float64, m, n, σ)
      a = lmo_atom(C, G)
      @test abs(dot(G, a) + r * σ[1]) <= 100 * eps() * r * σ[1]
      @test sum(svdvals(Array(a))) <= r * (1 + 20 * eps())
    end
  end
  # a transposed and a column-sliced gradient are ordinary matrices
  G = randn(7, 11)
  @test dot(G', lmo_atom(C, G')) ≈ dot(G, lmo_atom(C, G))
  @test dot(view(G, :, 2:9), lmo_atom(C, view(G, :, 2:9))) ≈ -r * opnorm_ref(G[:, 2:9])
  # ill-scaled inputs: the scale is removed before the decomposition
  G = randn(9, 6)
  for (T, scales) in ((Float64, (1e200, 1e-200, 1e300, 5e-320)), (Float32, (1f30, 1f-30, 1f35)))
    for c in scales
      H = T.(G) .* T(c)
      a = lmo_atom(C, H)
      @test all(isfinite, a.U) && all(isfinite, a.V)
      # the reference is the rounded matrix itself, brought back to unit scale by an exact power of two
      Hs = ldexp.(H, -exponent(maximum(abs, H)))
      @test abs(dot(Hs, a) + r * opnorm_ref(Hs)) <= 100 * eps(T) * r * opnorm_ref(Hs)
    end
  end
  # one huge entry among tiny ones, in single precision
  H = fill(1f-3, 5, 5)
  H[2, 3] = 1f30
  a = lmo_atom(C, H)
  @test abs(a.U[2, 1]) ≈ r && abs(a.V[3, 1]) ≈ 1
end

@testset "spectral sets: density matrices" begin
  for T in (Float32, Float64), n in (1, 2, 11), r in (0.5, 3.0)
    D = Spectral(IndSimplex(r))
    @test haslmo(D)
    A = with_eigenvalues(T, n == 1 ? [0.3] : randn(n))
    λ = eigvals(Symmetric(A))
    tol = 100 * eps(T) * r * max(abs(λ[1]), abs(λ[end]), 1)
    a = lmo_atom(D, A)
    @test a isa LowRankAtom{T} && a.U === a.V && size(a.U) == (n, 1)
    S = Array(a)
    @test issymmetric(S)
    @test abs(tr(S) - r) <= 10 * eps(T) * r                                 # trace r
    @test minimum(eigvals(Symmetric(S))) >= -10 * eps(T) * r                  # PSD
    @test abs(dot(A, a) - r * λ[1]) <= tol                                   # r λ_min
    @test lmo(D, A) == S
    @test abs(Conjugate(D)(A) - r * λ[end]) <= tol                            # r λ_max
  end
  # the inequality form: the origin when λ_min ≥ 0, an atom otherwise
  D = Spectral(IndSimplex(2.0; equality = false))
  P = with_eigenvalues(Float64, [0.0, 0.5, 2.0])
  @test size(lmo_atom(D, P + 1e-3 * I).U, 2) == 0
  @test size(lmo_atom(D, Diagonal([0.0, 1.0, 2.0]) |> Matrix).U, 2) == 0   # a tie at 0 goes to the origin
  A = with_eigenvalues(Float64, [-1.0, 0.5, 2.0])
  a = lmo_atom(D, A)
  @test size(a.U, 2) == 1 && dot(A, a) ≈ -2.0
  @test Conjugate(D)(A) ≈ 2 * 2.0 && Conjugate(D)(-A) ≈ 2 * 1.0
  @test Conjugate(D)(-P - 1e-3I) == 0                                         # all eigenvalues negative: the origin
  # a dominant eigenvalue that is negative, diagonal and rank-one matrices, repeated and clustered spectra
  D = Spectral(IndSimplex(1.0))
  for λ in ([-10.0, 0.1, 0.2, 0.3], [-1.0, -1.0, 2.0], [1.0, 1 + 1e-10, 1 + 2e-10, 5.0], [0.0, 0.0, 7.0], [4.0, 4.0, 4.0])
    A = with_eigenvalues(Float64, λ)
    @test abs(dot(A, lmo_atom(D, A)) - minimum(λ)) <= 100 * eps() * maximum(abs, λ)
  end
  A = Diagonal([3.0, -2.0, 1.0]) |> Matrix
  @test Array(lmo_atom(D, A)) ≈ Diagonal([0.0, 1.0, 0.0])
  A = 3.0 * (v = normalize(randn(5)); v * v')
  @test dot(A, lmo_atom(D, A)) ≈ 0 atol = 1e-14                               # λ_min = 0 with multiplicity 4
  # zero gradient: a feasible atom for a fixed trace, and the origin otherwise
  @test Array(lmo_atom(Spectral(IndSimplex(2.0)), zeros(3, 3))) ≈ Diagonal([2.0, 0, 0]) rtol = 4eps()   # the factor is √r
  @test Array(lmo_atom(Spectral(IndSimplex(2.0; equality = false)), zeros(3, 3))) == zeros(3, 3)
  @test Array(lmo_atom(Spectral(IndSimplex(0.0)), randn(3, 3))) == zeros(3, 3)
  # ill-scaled
  A = with_eigenvalues(Float64, [-1.0, 0.5, 2.0])
  for c in (1e200, 1e-200, 1e300)
    a = lmo_atom(Spectral(IndSimplex(1.0)), A .* c)
    @test dot(A, a) ≈ -1.0
  end
end

@testset "spectral sets: the pairing uses the symmetric part of the gradient" begin
  D = Spectral(IndSimplex(2.0))
  A = with_eigenvalues(Float64, [-2.0, 1.0, 3.0, 4.0])
  K = randn(4, 4)
  K = K - K'                                   # skew
  G = A + K
  a = lmo_atom(D, G)
  S = Array(a)
  @test dot(G, S) ≈ dot(A, S)                  # the skew part does not pair with a symmetric matrix
  @test dot(A, S) ≈ 2 * -2.0
  @test dot(Symmetric(G), lmo_atom(D, Symmetric(G))) ≈ 2 * eigmin(Symmetric(G))          # a Symmetric wrapper reads its own triangle
  @test Array(lmo_atom(D, Symmetric(A))) == Array(lmo_atom(D, A))
  # upper triangle only: this differs from the symmetric part, and is not what the oracle uses
  U = triu(G)
  @test !(dot(G, Array(lmo_atom(D, U))) ≈ dot(G, S))
  F = Fantope(2)
  Q = lmo_atom(F, G)
  @test dot(G, Q) ≈ sum(eigvals(Symmetric(A))[1:2])
  @test_throws DimensionMismatch lmo_atom(D, randn(3, 4))
end

@testset "Fantope" begin
  for T in (Float32, Float64), n in (8, 13), k in (1, 3, n - 2)
    F = Fantope(k)
    @test haslmo(F)
    A = with_eigenvalues(T, randn(n))
    λ = eigvals(Symmetric(A))
    tol = 100 * eps(T) * k * max(abs(λ[1]), abs(λ[end]), 1)
    a = lmo_atom(F, A)
    @test a isa LowRankAtom{T} && a.U === a.V && size(a.U) == (n, k)
    S = Array(a)
    @test issymmetric(S)
    @test norm(a.U' * a.U - I) <= 50 * eps(T) * n                              # orthonormal columns: a projector
    μ = eigvals(Symmetric(S))
    @test all(m -> abs(m) <= 50 * eps(T) || abs(m - 1) <= 50 * eps(T), μ)       # eigenvalues in {0, 1}, within [0, 1]
    @test abs(tr(S) - k) <= 50 * eps(T) * n
    @test abs(dot(A, a) - sum(λ[1:k])) <= tol                                   # the k smallest eigenvalues
    @test lmo(F, A) == S
    @test abs(Conjugate(F)(A) - sum(λ[n - k + 1:n])) <= tol                     # the k largest
  end
  # the k-th eigenvalue is tied: any minimizer, judged by feasibility and objective
  A = with_eigenvalues(Float64, [1.0, 2.0, 2.0, 2.0, 5.0, 6.0])
  for k in 2:3
    a = lmo_atom(Fantope(k), A)
    @test dot(A, a) ≈ 1.0 + 2.0 * (k - 1)
    @test norm(a.U' * a.U - I) < 1e-13
  end
  # k = 0 and k = n do not decompose, and do not read the gradient
  @test size(lmo_atom(Fantope(0), randn(4, 4)).U) == (4, 0)
  @test Array(lmo_atom(Fantope(4), fill(NaN, 4, 4))) == Matrix(1.0I, 4, 4)
  @test Array(lmo_atom(Fantope(0), fill(NaN, 3, 3))) == zeros(3, 3)
  # zero gradient: the first k columns of the identity
  @test Array(lmo_atom(Fantope(2), zeros(4, 4))) == Diagonal([1.0, 1, 0, 0])
  # invalid k and unsupported variants
  @test_throws ArgumentError lmo_atom(Fantope(5), randn(4, 4))
  @test_throws ArgumentError lmo!(zeros(4, 4), Fantope(5), randn(4, 4))
  @test !haslmo(Fantope(-1)) && !haslmo(Fantope(1.5))
  @test_throws ArgumentError lmo_atom(Fantope(-1), randn(3, 3))
  @test_throws ArgumentError lmo_atom(Fantope(1.5), randn(3, 3))
  @test_throws ArgumentError lmo_atom(Fantope(1), randn(4, 4); method = :krylov)
  @test !haslmo(Spectral(IndBox(0, 1))) && !haslmo(Spectral(IndNonnegative())) && !haslmo(SingularValue(NormL1(1.0)))
  @test !haslmo(SingularValue(IndBallL1(Inf))) && !haslmo(Spectral(IndSimplex(Inf)))
  @test_throws ArgumentError lmo_atom(Spectral(IndBox(0, 1)), randn(3, 3))
  # the identity for k = n and the zero atom are the same objects the dense oracle writes
  @test lmo(Fantope(4), randn(4, 4)) == Matrix(1.0I, 4, 4)
end

@testset "dense lmo!: validation before writes, and the output unchanged on error" begin
  C = SingularValue(IndBallL1(1.0))
  D = Spectral(IndSimplex(1.0))
  F = Fantope(2)
  g = randn(4, 4)
  for (set, bad) in ((C, randn(4, 5)), (D, randn(4, 5)), (F, randn(4, 5)))
    s = fill(7.0, size(bad))
    @test_throws DimensionMismatch lmo!(s, set, g)
    @test s == fill(7.0, size(bad))
  end
  for set in (C, D, F)
    s = fill(7.0, 4, 4)
    @test_throws ArgumentError lmo!(zeros(Int, 4, 4), set, g)                    # not floating point
    @test_throws ArgumentError lmo!(zeros(ComplexF64, 4, 4), set, g)
    @test_throws ArgumentError lmo!(Symmetric(zeros(4, 4)), set, g)
    @test_throws MethodError lmo!(s, set, 1.0)
    @test s == fill(7.0, 4, 4)
  end
  # a radius that does not fit the element type of the output
  s = fill(7.0f0, 4, 4)
  @test_throws ArgumentError lmo!(s, SingularValue(IndBallL1(1e300)), g)
  @test_throws ArgumentError lmo!(s, Spectral(IndSimplex(1e300)), g)
  @test s == fill(7.0f0, 4, 4)
  # an empty simplex of fixed trace
  @test_throws ArgumentError lmo_atom(Spectral(IndSimplex(1.0)), zeros(0, 0))
  @test size(lmo_atom(Spectral(IndSimplex(0.0)), zeros(0, 0)).U) == (0, 0)
  # NaN and infinity: all-NaN output from lmo!, an error from lmo_atom (no compact NaN atom)
  for set in (C, D, F), bad in (NaN, Inf, -Inf)
    h = copy(g)
    h[2, 3] = h[3, 2] = bad
    s = fill(7.0, 4, 4)
    @test lmo!(s, set, h) === s
    @test all(isnan, s)
    @test_throws ArgumentError lmo_atom(set, h)
  end
  # the output is exactly symmetric for the spectral sets, and a mixed precision is converted
  s = zeros(Float64, 4, 4)
  lmo!(s, D, Float32.(g + g'))
  @test issymmetric(s)
  s32 = zeros(Float32, 4, 4)
  lmo!(s32, F, g + g')
  @test issymmetric(s32) && abs(tr(s32) - 2) < 1e-5
  # non-BLAS element types and device-like inputs are rejected before anything is written
  @test_throws ArgumentError lmo_atom(C, rand(Float16, 3, 3))
  @test_throws ArgumentError lmo_atom(D, rand(BigFloat, 3, 3))
  @test_throws ArgumentError lmo_atom(C, rand(ComplexF64, 3, 3))
end

@testset "atoms stay valid after later calls, nested calls, other sizes and precisions" begin
  C = SingularValue(IndBallL1(2.0))
  D = Spectral(IndSimplex(1.0))
  F = Fantope(2)
  G1, G2, G3 = randn(9, 6), randn(5, 7), randn(Float32, 9, 6)
  a1 = lmo_atom(C, G1)
  U1, V1 = copy(a1.U), copy(a1.V)
  lmo_atom(C, G2); lmo_atom(C, G3); lmo!(zeros(9, 6), C, randn(9, 6)); lmo_atom(C, randn(9, 6); method = :krylov)
  @test a1.U == U1 && a1.V == V1
  a2 = lmo_atom(C, G2)
  @test lmo_atom(C, G1).U == U1 && a2.U !== a1.U
  A1, A2 = with_eigenvalues(Float64, randn(6)), with_eigenvalues(Float32, randn(4))
  d1 = lmo_atom(D, A1)
  f1 = lmo_atom(F, A1)
  d1c, f1c = copy(d1.U), copy(f1.U)
  lmo_atom(D, A2); lmo_atom(F, A2); lmo_atom(D, A1; method = :krylov); lmo_atom(F, A1)
  @test d1.U == d1c && f1.U == f1c
  # the output of one oracle call is the input of the next
  X = randn(6, 6)
  for _ in 1:3
    a = lmo_atom(C, X)
    X = Array(a)
  end
  @test size(X) == (6, 6)
  # one set object per task: separate sets in parallel tasks agree with the sequential results
  sets = [SingularValue(IndBallL1(2.0)) for _ in 1:4]
  Gs = [randn(20, 15) for _ in 1:4]
  seq = [lmo_atom(SingularValue(IndBallL1(2.0)), G).U for G in Gs]
  par = fetch.([Threads.@spawn lmo_atom(sets[i], Gs[i]).U for i in 1:4])
  @test par == seq
  sets = [Spectral(IndSimplex(1.0)) for _ in 1:4]
  Hs = [with_eigenvalues(Float64, randn(10)) for _ in 1:4]
  @test fetch.([Threads.@spawn lmo_atom(sets[i], Hs[i]; method = :krylov).U for i in 1:4]) ==
        [lmo_atom(Spectral(IndSimplex(1.0)), H; method = :krylov).U for H in Hs]
end

@testset "dot and axpy! of low-rank atoms; dense, sparse" begin
  C = SingularValue(IndBallL1(2.0))
  G = randn(9, 6)
  a = lmo_atom(C, G)
  Gs = sprandn(9, 6, 0.4)
  @test dot(Gs, a) ≈ dot(Matrix(Gs), Array(a))
  @test dot(a, Gs) == dot(Gs, a)
  X = randn(9, 6)
  Y = copy(X)
  axpy!(0.3, a, Y)
  @test Y ≈ X + 0.3 * Array(a)
  @test_throws DimensionMismatch dot(randn(5, 6), a)
  # a symmetric atom stays exactly symmetric when added to a symmetric matrix
  A = with_eigenvalues(Float64, randn(7))
  d = lmo_atom(Fantope(3), A)
  Z = zeros(7, 7)
  axpy!(0.4, d, Z); axpy!(0.2, d, Z)
  @test issymmetric(Z)
  @test Array{Float32}(d) isa Matrix{Float32}
  @test occursin("symmetric", sprint(show, d)) && occursin("VertexAtom", sprint(show, lmo_atom(IndSimplex(1.0), [1.0, 2.0])))
  # the zero atom
  z = lmo_atom(C, zeros(3, 3))
  @test dot(randn(3, 3), z) == 0
  W = randn(3, 3)
  W0 = copy(W)
  axpy!(1.0, z, W)
  @test W == W0
end

@testset "the Krylov method: residual-checked, with a full fallback" begin
  r = 2.0
  C = SingularValue(IndBallL1(r))
  for T in (Float32, Float64), (m, n) in ((60, 40), (40, 60))
    G = with_singular_values(T, m, n, [3.0; 2.0; 1.0 ./ (1:25)])
    full, kry = lmo_atom(C, G), lmo_atom(C, G; method = :krylov)
    @test abs(dot(G, kry) + r * 3) <= 20 * sqrt(eps(T)) * r * 3
    @test dot(G, kry) >= dot(G, full) - 100 * eps(T) * r * 3                     # never better than the exact oracle
    @test sum(svdvals(Array(kry))) <= r * (1 + 20 * eps(T))
  end
  # sparse input: no dense copy, and the same optimum
  Gs = sprandn(500, 400, 0.02)
  kry = lmo_atom(C, Gs; method = :krylov, rtol = 1e-10)
  @test abs(dot(Gs, kry) + r * opnorm_ref(Gs)) <= 1e-8 * r * opnorm_ref(Gs)
  # the fallback: one step cannot converge, so the full decomposition answers
  G = randn(30, 20)
  @test Array(lmo_atom(C, G; method = :krylov, maxiter = 1)) == Array(lmo_atom(C, G))
  @test_throws ErrorException lmo_atom(C, G; method = :krylov, maxiter = 1, fallback = false)
  @test_throws ErrorException lmo_atom(Spectral(IndSimplex(1.0)), G * G'; method = :krylov, maxiter = 1, fallback = false)
  # a gradient of rank one converges in two steps, an invariant subspace; a zero gradient does not iterate
  R = randn(25, 1) * randn(1, 18)
  @test abs(dot(R, lmo_atom(C, R; method = :krylov)) + r * opnorm_ref(R)) <= 1e-12 * r * opnorm_ref(R)
  @test size(lmo_atom(C, zeros(5, 4); method = :krylov).U, 2) == 0
  # clustered and repeated top values: the atom is feasible and never beats the optimum
  for σ in ([1.0, 1 - 1e-9, 0.5, 0.2], [2.0, 2.0, 2.0, 0.1], [1.0, 1 - 1e-6, 1 - 2e-6, 0.3])
    G = with_singular_values(Float64, 40, 30, σ)
    a = lmo_atom(C, G; method = :krylov)
    @test dot(G, a) >= -r * σ[1] - 1e-13
    @test dot(G, a) <= -r * σ[1] * (1 - 1e-6)
    @test sum(svdvals(Array(a))) <= r * (1 + 1e-12)
  end
  # the density matrices: smallest eigenvalue, with a negative dominant eigenvalue
  D = Spectral(IndSimplex(1.0))
  A = with_eigenvalues(Float64, [-10.0; 0.0; rand(30)])
  @test abs(dot(A, lmo_atom(D, A; method = :krylov)) + 10) <= 1e-8
  A = with_eigenvalues(Float64, [10.0; 1.0; 2.0; rand(40) .+ 3])
  @test abs(dot(A, lmo_atom(D, A; method = :krylov)) - 1.0) <= 1e-7
  As = sprandn(300, 300, 0.03); As = As + As'
  kry = lmo_atom(D, As; method = :krylov)
  @test abs(dot(As, kry) - eigmin(Symmetric(Matrix(As)))) <= 1e-6
  # a nonsymmetric sparse gradient: the symmetric part
  Gn = sprandn(200, 200, 0.05)
  @test abs(dot(Gn, lmo_atom(D, Gn; method = :krylov)) - eigmin(Symmetric((Matrix(Gn) + Matrix(Gn)') / 2))) <= 1e-6
  # the options are validated
  @test_throws ArgumentError lmo_atom(C, G; method = :krylov, rtol = -1.0)
  @test_throws ArgumentError lmo_atom(C, G; method = :krylov, maxiter = 0)
  @test_throws ArgumentError lmo_atom(C, G; rtol = 1e-3)                        # rtol belongs to :krylov
  # a scale outside the range of the iteration goes to the full decomposition
  Gh = G .* 1e-310
  @test abs(dot(G, lmo_atom(C, Gh; method = :krylov)) + r * opnorm_ref(G)) <= 100 * eps() * r * opnorm_ref(G)
  @test occursin("not certified", string(@doc lmo_atom))
end

@testset "conjugate values: the sets with an oracle keep their formulas, user sets the fallback" begin
  # SingularValue keeps its own conjugate through the spectrum
  Y = randn(7, 5)
  @test Conjugate(SingularValue(IndBallL1(3.0)))(Y) ≈ 3 * opnorm_ref(Y)
  @test Conjugate(SingularValue(IndBallL1(3.0)))(fill(NaN, 2, 2)) == Inf
  # the spectral sets: finite domain, symmetric part
  S = Spectral(IndSimplex(2.0))
  A = with_eigenvalues(Float64, [-1.0, 0.0, 3.0])
  @test Conjugate(S)(A) ≈ 6.0
  @test Conjugate(S)(A + (K = randn(3, 3); K - K')) ≈ 6.0
  @test Conjugate(S)(fill(Inf, 2, 2)) == Inf
  @test Conjugate(S)(zeros(3, 3)) == 0
  @test Conjugate(Fantope(2))(A) ≈ 3.0 + 0.0
  @test Conjugate(Fantope(0))(A) == 0 && Conjugate(Fantope(3))(A) ≈ tr(A)
  @test_throws ArgumentError Conjugate(Fantope(4))(A)
  @test_throws DimensionMismatch Conjugate(S)(randn(3, 4))
  @test Conjugate(S)(A .* 1e300) ≈ 6e300 rtol = 1e-14
  @test Conjugate(S)(Float32.(A) .* 1f37) ≈ 6f37 rtol = 1f-5
  # the proximal operator of the conjugate (Moreau) uses these values
  P = zeros(3, 3)
  v = value_and_prox!(P, Conjugate(S), A, 0.7)
  @test isfinite(v)
  # a user-defined matrix set keeps the generic support-function fallback
  C = MatrixCube(1.0)
  @test Conjugate(C)(A) ≈ sum(abs, A)
end

end
