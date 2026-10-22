module LapackTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

@testset "LAPACK workspaces" begin
  for T in (Float64, Float32)
    tol = T == Float64 ? 1e-10 : 1e-4
    @testset "symmetric eigendecomposition, $T" begin
      n = 30
      A = Matrix{T}(Symmetric(randn(T, n, n)))
      ws = PB.SymEigenWorkspace(T, n)
      copyto!(ws.A, A)
      λ, V = PB._eigen!(ws)
      @test λ ≈ eigvals(Symmetric(A)) rtol = tol
      @test V * Diagonal(λ) * V' ≈ A rtol = tol
      @test V' * V ≈ I rtol = tol atol = tol
      copyto!(ws.A, A)
      @test PB._eigvals!(ws) ≈ eigvals(Symmetric(A)) rtol = tol
      @test PB._eig_workspace(ws, T, n) === ws                   # reused when it fits
      @test PB._eig_workspace(ws, T, n + 1) !== ws
      @test PB._eig_workspace(nothing, T, n) isa PB.SymEigenWorkspace{T}
      f(ws, A) = (copyto!(ws.A, A); PB._eigen!(ws))
      @test allocs(f, ws, A) == 0
    end

    @testset "largest eigenpairs, $T" begin
      n, r = 25, 4
      A = Matrix{T}(Symmetric(randn(T, n, n)))
      ws = PB.SymEigenTopWorkspace(T, n, r)
      copyto!(ws.A, A)
      λ, Z = PB._eigen_top!(ws)
      E = eigen(Symmetric(A))
      @test λ ≈ E.values[end-r+1:end] rtol = tol
      @test abs.(Z' * E.vectors[:, end-r+1:end]) ≈ I rtol = sqrt(tol) atol = sqrt(tol)
      @test_throws ArgumentError PB.SymEigenTopWorkspace(T, n, n + 1)
      f(ws, A) = (copyto!(ws.A, A); PB._eigen_top!(ws))
      @test allocs(f, ws, A) == 0
    end

    @testset "singular value decomposition, $T" begin
      for (m, n) in ((20, 12), (12, 20))
        A = randn(T, m, n)
        ws = PB.SVDWorkspace(T, m, n)
        copyto!(ws.A, A)
        s, U, Vt = PB._svd!(ws)
        @test s ≈ svdvals(A) rtol = tol
        @test U * Diagonal(s) * Vt ≈ A rtol = tol
        copyto!(ws.A, A)
        s, = PB._svd!(ws; vectors = false)
        @test s ≈ svdvals(A) rtol = tol
        @test PB._svd_workspace(ws, T, m, n) === ws
        f(ws, A) = (copyto!(ws.A, A); PB._svd!(ws))
        @test allocs(f, ws, A) == 0
      end
    end

    @testset "rebuilding from eigenvectors, $T" begin
      n = 15
      A = Matrix{T}(Symmetric(randn(T, n, n)))
      E = eigen(Symmetric(A))
      Y = zeros(T, n, n)
      V = copy(E.vectors)
      s = PB._rebuild_sym!(Y, V, E.values, t -> max(t, zero(t)); logs = true)
      @test Y ≈ E.vectors * Diagonal(max.(E.values, 0)) * E.vectors' rtol = tol
      @test Y == Y'
      @test s ≈ sum(log, filter(>(0), E.values)) rtol = tol
      # no positive weight gives zero
      @test PB._rebuild_sym!(Y, copy(E.vectors), E.values, t -> zero(t)) == 0 && Y == zeros(T, n, n)
    end
  end
end

end
