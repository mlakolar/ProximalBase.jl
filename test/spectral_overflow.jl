module SpectralOverflowTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

const PB = ProximalBase

threshold_matrix(c::T, d::T) where {T} = T[c + d c - d; c - d c + d]
samebits(a::T, b::T) where {T} =
  reinterpret(T === Float32 ? UInt32 : UInt64, a) == reinterpret(T === Float32 ? UInt32 : UInt64, b)

function value_functions(::Type{T}) where {T}
  (IndBallRank(1), IndBallRank(1; partial = true), IndBallRank(2),
   IndPSD(), IndPSD(rank = 1), IndPSD(partial = true), IndPSD(rank = 1, partial = true),
   TracePSD(one(T)), TracePSD(one(T); partial = true),
   SingularValue(IndBallL0(1)), SingularValue(IndBallLinf(one(T))),
   SingularValue(NormL1(one(T))), SingularValue(NormL0(one(T))),
   Spectral(IndNonnegative()), Spectral(IndBox(-one(T), one(T))))
end

function logical_inputs(X::Matrix{T}) where {T}
  padded = fill(T(NaN), size(X, 1) + 2, size(X, 2) + 2)
  V = view(padded, 2:size(X, 1)+1, 2:size(X, 2)+1)
  copyto!(V, X)
  (X, sparse(X), SparseIterate(X), SymmetricSparseIterate(X), V)
end

# The value formulas at the base, through the same LAPACK workspaces.
function base_value(f::IndBallRank, X)
  T = float(eltype(X))
  minimum(size(X)) <= f.r && return zero(T)
  ws = PB._svd_workspace(f.ws, T, size(X)...)
  copyto!(ws.A, X)
  s, = PB._svd!(ws; vectors = false)
  s[f.r + 1] <= sqrt(eps(T)) * max(one(T), s[1]) ? zero(T) : T(Inf)
end

function base_ispsd(f, X, r)
  T = float(eltype(X))
  n = LinearAlgebra.checksquare(X)
  PB._issymmetric_tol(X) || return false
  ws = PB._eig_workspace(f.eig, T, n)
  copyto!(ws.A, X)
  λ = PB._eigvals!(ws)
  n == 0 && return true
  tol = sqrt(eps(T)) * max(one(T), abs(λ[1]), abs(λ[n]))
  λ[1] >= -tol || return false
  r === nothing || count(>(tol), λ) <= r
end

base_value(f::IndPSD, X) = base_ispsd(f, X, f.rank) ? zero(eltype(X)) : eltype(X)(Inf)
base_value(f::TracePSD, X) =
  base_ispsd(f, X, nothing) ? convert(eltype(X), f.λ) * tr(X) : eltype(X)(Inf)

function base_value(F::SingularValue, X)
  T = float(eltype(X))
  ws = PB._svd_workspace(F.ws, T, size(X)...)
  copyto!(ws.A, X)
  s, = PB._svd!(ws; vectors = false)
  c = isempty(s) ? zero(T) : sqrt(eps(T)) * s[1]
  convert(T, F.f([v <= c ? zero(T) : v for v in s]))
end

function base_value(f::Spectral, X)
  T = float(eltype(X))
  PB._issymmetric_tol(X) || return T(Inf)
  ws = PB._eig_workspace(f.eig, T, LinearAlgebra.checksquare(X))
  copyto!(ws.A, X)
  convert(T, f.f(PB._eigvals!(ws)))
end

@testset "spectral overflow" begin
  for T in (Float32, Float64)
    @testset "$T" begin
      M = floatmax(T)
      τ = sqrt(eps(T))
      c = (T(3) / T(4)) * M
      indefinite = T[M M; M -M]
      psd = threshold_matrix(c, M / 8)

      @testset "audit" begin
        for partial in (false, true)
          for X in (indefinite, T[M M M; M -M M])
            @test IndBallRank(1; partial)(X) === T(Inf)
          end
          for X in (indefinite, T[M M; M M / 2])
            @test IndPSD(; partial)(X) === T(Inf)
            @test IndPSD(; rank = 1, partial)(X) === T(Inf)
            @test TracePSD(one(T); partial)(X) === T(Inf)
          end
          for X in (fill(M, 2, 2), fill(M, 2, 3))
            @test IndBallRank(1; partial)(X) === zero(T)
          end
          @test IndPSD(; partial)(fill(M, 2, 2)) === zero(T)
          @test IndPSD(; rank = 1, partial)(fill(M, 2, 2)) === zero(T)
          @test IndPSD(; partial)(psd) === zero(T)
          @test IndPSD(; rank = 1, partial)(psd) === T(Inf)
          @test IndPSD(; rank = 2, partial)(psd) === zero(T)
        end
        @test SingularValue(IndBallL0(1))(indefinite) === T(Inf)
        @test SingularValue(NormL1(one(T)))(indefinite) === T(Inf)
        @test SingularValue(NormL0(one(T)))(indefinite) === T(2)
        for X in (fill(M, 2, 2), fill(M, 2, 3))
          @test SingularValue(IndBallL0(1))(X) === zero(T)
          @test SingularValue(NormL0(one(T)))(X) === one(T)
        end
        @test NuclearNorm(one(T))(indefinite) === T(Inf)
        @test Spectral(IndNonnegative())(indefinite) === T(Inf)
        @test TracePSD(one(T))(fill(M, 2, 2)) === T(Inf)
      end

      @testset "thresholds" begin
        @test isinf(2c)
        for factor in (T(7) / T(8), T(9) / T(8)), sign in (-one(T), one(T))
          d = sign * factor * τ * c
          X = threshold_matrix(c, d)
          @test all(isfinite, X)
          inside = factor < one(T)
          expected_rank = inside ? zero(T) : T(Inf)
          expected_psd = sign < 0 && !inside ? T(Inf) : zero(T)
          for Z in logical_inputs(X)
            before = Matrix(Z)
            for partial in (false, true)
              @test IndBallRank(1; partial)(Z) === expected_rank
              @test IndPSD(; partial)(Z) === expected_psd
              @test IndPSD(; rank = 1, partial)(Z) === expected_rank
            end
            @test SingularValue(IndBallL0(1))(Z) === expected_rank
            @test SingularValue(NormL0(one(T)))(Z) === T(inside ? 1 : 2)
            @test isequal(Matrix(Z), before)
          end
        end
        for X in (psd, indefinite)
          X3 = zeros(T, 3, 3)
          X3[1:2, 1:2] .= X
          Z = SymmetricSparseIterate(X3)
          @test IndBallRank(1)(Z) === T(Inf)
          @test IndPSD()(Z) === (X === psd ? zero(T) : T(Inf))
          @test IndPSD(rank = 1)(Z) === T(Inf)
          @test IndPSD(rank = 2)(Z) === (X === psd ? zero(T) : T(Inf))
          @test TracePSD(one(T))(Z) === T(Inf)
          @test SingularValue(IndBallL0(1))(Z) === T(Inf)
          @test SingularValue(NormL0(one(T)))(Z) === T(2)
          @test Matrix(Z) == X3
        end
      end

      @testset "nonfinite entries" begin
        functions = filter(f -> PB._isindicator(f) || f isa TracePSD, value_functions(T))
        for v in (T(NaN), T(Inf), T(-Inf))
          for X in (T[v 0; 0 1], T[1 v; v 1]), Z in logical_inputs(X)
            before = Matrix(Z)
            for f in functions
              @test f(Z) === T(Inf)
            end
            @test isequal(Matrix(Z), before)
          end
          for Z in logical_inputs(fill(v, 1, 1))
            @test IndBallRank(1)(Z) === T(Inf)
          end
        end
      end

      @testset "logical storage" begin
        X = Matrix{T}(I, 2, 2)
        padded = fill(T(NaN), 4, 4)
        padded[2:3, 2:3] .= X
        V = view(padded, 2:3, 2:3)
        S = view(sparse(padded), 2:3, 2:3)
        U = Symmetric(T[1 0; NaN 1], :U)
        for Z in (V, S, U), f in value_functions(T)
          @test samebits(f(Z), f(X))
        end
        for Z in (SparseIterate(X), SymmetricSparseIterate(X))
          @test nnz(Z) < length(Z.nzval)
          Z.nzval[nnz(Z)+1:end] .= T(NaN)
          before = copy(Z.nzval)
          for f in value_functions(T)
            @test samebits(f(Z), f(X))
          end
          @test isequal(Z.nzval, before)
        end
      end

      @testset "empty inputs" begin
        for X in (zeros(T, 0, 0), zeros(T, 0, 2), zeros(T, 2, 0))
          @test IndBallRank(1)(X) === zero(T)
          @test SingularValue(IndBallL0(1))(X) === zero(T)
          @test SingularValue(NormL1(one(T)))(X) === zero(T)
          @test SingularValue(NormL0(one(T)))(X) === zero(T)
          for f in (IndPSD(), IndPSD(rank = 1), TracePSD(one(T)), Spectral(IndNonnegative()))
            if size(X, 1) == size(X, 2)
              @test f(X) === zero(T)
            else
              @test_throws DimensionMismatch f(X)
            end
          end
        end
      end

      @testset "ordinary values match the base bitwise" begin
        rng = Xoshiro(165)
        for (m, n) in ((3, 3), (4, 2), (2, 5))
          X = randn(rng, T, m, n)
          L = randn(rng, T, m, 1) * randn(rng, T, 1, n)
          for f in (IndBallRank(1), IndBallRank(2), SingularValue(IndBallL0(1)),
                    SingularValue(NormL1(one(T))), SingularValue(NormL0(one(T)))), Z in (X, L)
            before = copy(Z)
            @test samebits(f(Z), base_value(f, Z))
            @test Z == before
          end
        end
        for n in (2, 4)
          B = randn(rng, T, n, n)
          for X in (B + B', B * B', B[:, 1:1] * B[:, 1:1]', B)
            for Z in logical_inputs(X), f in value_functions(T)
              before = Matrix(Z)
              @test samebits(f(Z), base_value(f, Z))
              @test Matrix(Z) == before
            end
          end
        end
      end

      @testset "workspace reuse and zero allocations" begin
        ordinary = T[2 1; 1 2]
        for wrap in (identity, SymmetricSparseIterate), f in value_functions(T)
          A, B, C = wrap(ordinary), wrap(psd), wrap(indefinite)
          @testset "$(typeof(f)), $(typeof(A))" begin
            before = f(A)
            f(B)
            @test samebits(f(A), before)
            f(C)
            @test samebits(f(A), before)
            for (label, X) in zip(("ordinary", "overflow PSD", "overflow indefinite"), (A, B, C))
              @testset "$label" begin
                input = Matrix(X)
                # Generic tr allocates a diagonal vector at the base; non-members skip the trace.
                expected = f isa TracePSD && X isa SymmetricSparseIterate && X !== C ? allocs(tr, X) : 0
                @test allocs(f, X) == expected
                @test Matrix(X) == input
              end
            end
          end
        end
      end
    end
  end
end

end
