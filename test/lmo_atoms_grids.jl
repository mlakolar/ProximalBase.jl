module LMOAtomsGridsTest

# The Float32 passes of lmo_atoms.jl that repeat a Float64 pass which stays there: the density matrices,
# the Krylov method, the Symmetric and Hermitian wrappers and the NaN and Inf support values. Each
# testset has the name of its twin in lmo_atoms.jl, which holds the rest. The fixtures and helpers
# are copied from that file.

using Test
using LinearAlgebra
using SparseArrays
using Random
using ProximalBase

using ..TestUtils

Random.seed!(518)

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

@testset "spectral sets: density matrices" begin
  for T in (Float32,), n in (1, 2, 11), r in (0.5, 3.0)
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
end

@testset "the Krylov method: residual-checked, with a full fallback" begin
  r = 2.0
  C = SingularValue(IndBallL1(r))
  for T in (Float32,), (m, n) in ((60, 40), (40, 60))
    G = with_singular_values(T, m, n, [3.0; 2.0; 1.0 ./ (1:25)])
    full, kry = lmo_atom(C, G), lmo_atom(C, G; method = :krylov)
    @test abs(dot(G, kry) + r * 3) <= 20 * sqrt(eps(T)) * r * 3
    @test dot(G, kry) >= dot(G, full) - 100 * eps(T) * r * 3                     # never better than the exact oracle
    @test sum(svdvals(Array(kry))) <= r * (1 + 20 * eps(T))
  end
end

@testset "review P1: Symmetric and Hermitian read only their logical entries" begin
  D = Spectral(IndSimplex(1.0))

  n = 7
  for T in (Float32,), wrapper in (Symmetric, Hermitian), uplo in (:U, :L)
    tol = T == Float64 ? 1e-9 : 2e-3
    S = let B = randn(T, n, n); (B + B') / 2 end              # the logical matrix
    Sref = Matrix(wrapper(S, uplo))
    λ, V = eigen(Symmetric(Sref))                              # independent references
    clean = wrapper(copy(S), uplo)
    a0 = lmo_atom(D, clean)
    for hidden in (T(NaN), T(Inf), -T(Inf), floatmax(T), -floatmax(T), floatmax(T) / 4)
      P = copy(S)
      for j in 1:n, i in 1:n
        (uplo == :U ? i > j : i < j) && (P[i, j] = hidden)       # the unused triangle
      end
      Gh = wrapper(P, uplo)
      @test Matrix(Gh) == Sref                                   # the logical matrix is unchanged
      @test ProximalBase._max_abs(Gh, T) == ProximalBase._max_abs(clean, T)
      for method in (:full, :krylov)
        a = lmo_atom(D, Gh; method)
        @test abs(dot(Sref, a) - λ[1]) <= tol * max(1, abs(λ[1]))              # λ_min, the support value of −g
        @test norm(Array(a) - V[:, 1] * V[:, 1]') <= 10 * tol                      # the eigenvector
        method == :full && @test Array(a) == Array(a0)                          # the hidden entries are not read
      end
      for k in (2, n - 2)
        F = Fantope(k)
        a = lmo_atom(F, Gh)
        @test abs(dot(Sref, a) - sum(λ[1:k])) <= tol * n
        Q = V[:, 1:k]
        @test norm(Array(a) - Q * Q') <= 10 * tol
      end
      Out = zeros(T, n, n)
      lmo!(Out, D, Gh)
      @test all(isfinite, Out) && Out == Array(a0)
      @test abs(Conjugate(D)(Gh) - λ[end]) <= tol * max(1, abs(λ[end]))
      @test isequal(parent(Gh)[uplo == :U ? 2 : 1, uplo == :U ? 1 : 2], hidden)       # the input is not modified
    end
    # a nonfinite entry of the logical triangle is still rejected
    P = copy(S); P[uplo == :U ? 1 : 2, uplo == :U ? 2 : 1] = T(NaN)
    @test_throws ArgumentError lmo_atom(D, wrapper(P, uplo))
    P = copy(S); P[2, 2] = T(Inf)
    @test_throws ArgumentError lmo_atom(D, wrapper(P, uplo))
    @test Conjugate(D)(wrapper(P, uplo)) == T(Inf)
  end
end

@testset "review round 3: NaN and Inf in the support values of the spectral sets (E2)" begin
  # a NaN entry of the logical matrix gives NaN, which takes precedence over an infinite entry elsewhere; an infinite
  # entry alone gives Inf; an entry of the unused triangle of a Symmetric or Hermitian wrapper has no influence
  n = 4
  for T in (Float32,)
    tol = T == Float32 ? 1e-4 : 1e-10
    A = randn(Xoshiro(11), T, n, n); S = (A + A') / 2
    λ = eigvals(Symmetric(S))
    sets = ((Spectral(IndSimplex(T(2))), r -> 2 * λ[end]),
            (Spectral(IndSimplex(T(2); equality = false)), r -> 2 * max(λ[end], 0)),
            (Fantope(1), r -> λ[end]), (Fantope(2), r -> λ[end] + λ[end - 1]))
    for (set, ref) in sets
      @test abs(Conjugate(set)(S) - ref(nothing)) <= tol * max(1, abs(ref(nothing)))
      for (i, j) in ((2, 1), (1, 2), (3, 3))
        Y = copy(S); Y[i, j] = T(NaN)
        @test isnan(Conjugate(set)(Y))
        Y[4, 4] = T(Inf)                                              # NaN beats Inf
        @test isnan(Conjugate(set)(Y))
        Y = copy(S); Y[i, j] = T(-Inf)
        @test Conjugate(set)(Y) == T(Inf)                              # an infinite entry alone: Inf
      end
      @test Conjugate(set)(fill(T(Inf), n, n)) == T(Inf)
      @test isnan(Conjugate(set)(fill(T(NaN), n, n)))
      for wrapper in (Symmetric, Hermitian), uplo in (:U, :L)
        logical = uplo == :U ? (1, 2) : (2, 1)
        hidden = uplo == :U ? (2, 1) : (1, 2)
        for parent_of in (identity, sparse)
          P = copy(S); P[logical...] = T(NaN)
          @test isnan(Conjugate(set)(wrapper(parent_of(P), uplo)))
          P[3, 3] = T(Inf)
          @test isnan(Conjugate(set)(wrapper(parent_of(P), uplo)))     # logical NaN and logical Inf
          P = copy(S); P[3, 3] = T(Inf)
          @test Conjugate(set)(wrapper(parent_of(P), uplo)) == T(Inf)
          for h in (T(NaN), T(Inf), -T(Inf))                            # hidden entries are not read
            P = copy(S); P[hidden...] = h
            Ref = Matrix(wrapper(S, uplo))
            v = Conjugate(set)(wrapper(parent_of(P), uplo))
            λw = eigvals(Symmetric(Ref))
            expected = set === sets[1][1] ? 2 * λw[end] : set === sets[2][1] ? 2 * max(λw[end], 0) :
                       set === sets[3][1] ? λw[end] : λw[end] + λw[end - 1]
            @test isfinite(v) && abs(v - expected) <= tol * max(1, abs(expected))
          end
        end
      end
    end
    # the input-independent cases keep their short-circuits (nothing is read)
    @test Conjugate(Spectral(IndSimplex(zero(T))))(fill(T(NaN), n, n)) == 0
    @test Conjugate(Fantope(0))(fill(T(NaN), n, n)) == 0
  end
end

end
