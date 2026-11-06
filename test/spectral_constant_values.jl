module SpectralConstantValuesTest

# Functions of a spectrum whose value does not depend on the entries, at non-finite entries
# (E2, E3, E4; PR #384, ruling 2b, and the maintainer's decision, option A: a zero-valued function
# is 0 also at an infinite entry, the indicator of a set of real matrices Inf there). The
# references are exact: 0, Inf or NaN. The fused value of a counting function (#614) is its value
# at the stored output (O2): a copy for IndBallRank(r ≥ p), otherwise all NaN, where a constant
# counting function is 0 and another NaN.

using Test, ProximalBase, LinearAlgebra, SparseArrays
using JLArrays: JLArray
using ..TestUtils

const PB = ProximalBase

# the zero function under E4: a zero level and no domain constraint
zero_inner(T) = (NormL1(zero(T)), NormL2(zero(T)), NormLinf(zero(T)), SumLargest(1, zero(T)),
                 SumLargest(2, zero(T)), CubeNormL2(zero(T)), SqrNormL2(zero(T)),
                 ElasticNet(zero(T), zero(T)), ZeroFunction(), NormL0(zero(T)))
# general rule: weights, centers, infinite levels, domain constraints, other zero-level functions
general_inner(T) = (NormL1(zero(T), T[1, Inf]), NormL1(zero(T), T[0, 1]), NormL1(T(Inf)), NormL2(T(Inf)),
                    SqrNormL2(zero(T), T[1, 2]), NormL0(one(T)), NormL1(one(T)), IndBallL0(1),
                    NormL1Nonneg(zero(T)), IndNonnegative(), Conjugate(NormL1(zero(T))),
                    Constrained(NormL1(zero(T)), IndBox(0, Inf)))

general_value(X, ::Type{T}) where {T} = any(isnan, X) ? T(NaN) : T(Inf)
same(a, b) = typeof(a) === typeof(b) && isequal(a, b)
# the symmetric part that value_and_prox! of Spectral reads
_sympart(X) = [(X[i, j] + X[j, i]) / 2 for i in axes(X, 1), j in axes(X, 2)]

# patterns of a 2 × 2 matrix, and the rectangular ones
function patterns(T)
  n, i = T(NaN), T(Inf)
  (T[n 0; 0 1], T[1 n; 0 1], T[1 0; n 1], T[1 n; n 1], fill(n, 2, 2),
   T[i 0; 0 1], T[1 i; i 1], T[-i 0; 0 1], T[1 -i; 0 1],
   T[n i; 0 1], T[i 0; 0 n], T[1 n; -i 1], T[n 0; 0 i],
   T[n 0 1; 0 1 0], T[1 0 0; 0 i 0], Matrix(T[n i 0; 0 1 0]'))
end

@testset "Constant functions of a spectrum at non-finite entries" begin
  for T in (Float32, Float64), X in patterns(T)
    p = minimum(size(X))
    hasinf = any(isinf, X)
    whole = hasinf ? T(Inf) : zero(T)
    # the zero function: 0 at every input (E4; E3, its formula)
    for f in (NuclearNorm(zero(T)), NuclearNorm(zero(T); partial = true),
              (SingularValue(g) for g in zero_inner(T))...)
      @test same(f(X), zero(T))
      for alias in (false, true)
        x = copy(X)
        y = alias ? x : fill(T(17), size(X))
        @test same(value_and_prox!(y, f, x, one(T)), zero(T))
        # the outputs are unchanged: NuclearNorm(0) copies, SingularValue gives NaN
        @test isequal(y, f isa NuclearNorm ? X : fill(T(NaN), size(X)))
        alias || @test isequal(x, X)
      end
      f isa SingularValue && @test f.ws === nothing
    end
    # the indicator of all matrices: independent of a NaN entry (E2), Inf at an infinite one (E3)
    for f in (IndBallRank(p), IndBallRank(p + 1; partial = true), SingularValue(IndBallL0(p)),
              SingularValue(IndBallL0(p + 3)))
      @test same(f(X), whole)
      for alias in (false, true)
        x = copy(X)
        y = alias ? x : fill(T(17), size(X))
        # IndBallRank copies X; SingularValue(IndBallL0) is 0 at its all-NaN output (O2, #614)
        @test same(value_and_prox!(y, f, x, one(T)), f isa IndBallRank ? whole : zero(T))
        @test isequal(y, f isa IndBallRank ? X : fill(T(NaN), size(X)))
      end
    end
    # the general rule
    for f in (NuclearNorm(one(T)), IndBallRank(p - 1), SingularValue(IndBallL0(p - 1)),
              TracePSD(zero(T)), IndPSD(), (SingularValue(g) for g in general_inner(T))...)
      size(X, 1) == size(X, 2) || !(f isa Union{TracePSD, IndPSD}) || continue
      # weights and centers have the length of the spectrum, 2
      f isa SingularValue && f.f isa Union{NormL1{<:Real, <:AbstractArray}, SqrNormL2{<:Real, <:AbstractArray}} &&
        p != 2 && continue
      @test same(f(X), general_value(X, T))
      y = fill(T(17), size(X))
      v = try
        value_and_prox!(y, f, copy(X), one(T))
      catch e
        e
      end
      # a counting function at its all-NaN output (O2, #614); the others E2
      counting = f isa IndBallRank || f isa SingularValue && PB._counts_nonzeros(f.f)
      v isa Exception || @test same(v, counting ? T(NaN) : general_value(X, T))
    end
  end
  # the same functions at a finite input
  for T in (Float32, Float64)
    X = T[3 1; 1 2]
    for f in (NuclearNorm(zero(T)), (SingularValue(g) for g in zero_inner(T))..., IndBallRank(2),
              SingularValue(IndBallL0(2)), (Spectral(g) for g in zero_inner(T))...)
      @test same(f(X), zero(T))
      @test same(value_and_prox!(similar(X), f, X, one(T)), zero(T))
    end
  end

  @testset "Spectral of a constant function" begin
    for T in (Float32, Float64)
      n, i = T(NaN), T(Inf)
      fs = (map(Spectral, zero_inner(T))..., Spectral(IndBallL0(2)), Spectral(IndBallL0(5)))
      cases = (
        (T[n 0; 0 1], zero(T)),         # a diagonal NaN does not decide symmetry (E2)
        (T[n 2; 2 n], zero(T)),
        (T[n 2; 0 1], T(Inf)),          # the finite remainder is not symmetric (E6)
        (T[1 n; 0 1], T(NaN)),          # an off-diagonal NaN decides symmetry (E2)
        (T[1 0; n 1], T(NaN)),
        (T[1 n; n 1], T(NaN)),          # mirrored NaN: its two values may differ
        (T[i 0; 0 1], T(Inf)),          # an indicator at an infinite entry (E3)
        (T[1 i; i 1], T(Inf)),
        (T[n 0; 0 i], T(Inf)),          # diagonal NaN, infinite entry
        (T[n i; i 1], T(Inf)),
        (T[1 n; i 1], T(NaN)),          # off-diagonal NaN with an infinite entry: E2 determines NaN
                                        # (dependence over finite values of the other entries)
        (fill(n, 2, 2), T(NaN)))
      for f in fs, (X, expected) in cases
        @test same(f(X), expected)
        @test same(f(sparse(X)), expected)
        @test same(f(JLArray(X)), expected)
        # value_and_prox! reads the symmetric part and keeps the general rule
        for alias in (false, true)
          x = copy(X)
          y = alias ? x : fill(T(17), 2, 2)
          @test same(value_and_prox!(y, f, x, one(T)), general_value(_sympart(X), T))
          @test all(isnan, y) || all(isfinite, X)
        end
      end
      # symmetric by type: the NaN entries do not decide membership
      for f in fs
        for S in (Symmetric(T[1 n; 0 1]), Symmetric(T[1 0; n 1], :L), Hermitian(T[n n; 0 1]),
                  Diagonal(T[n, 1]), SymTridiagonal(T[1, n], T[n]),
                  SymmetricSparseIterate(Symmetric(T[1 0; n 1], :L)), transpose(Symmetric(T[1 n; 0 1])))
          @test same(f(S), zero(T))
        end
        for S in (Symmetric(T[1 i; 0 n]), Diagonal(T[n, i]), SymmetricSparseIterate(Symmetric(T[n 0; i 1], :L)))
          @test same(f(S), T(Inf))
        end
        # an unread NaN does not enter
        @test same(f(Symmetric(T[1 0; n 1], :U)), zero(T))
      end
      # the general rule for other inner functions
      for g in general_inner(T)
        PB._ispermutationinvariant(g) || continue
        f = Spectral(g)
        for (X, _) in cases
          @test same(f(X), general_value(X, T))
        end
        @test same(f(Symmetric(T[1 n; 0 1])), T(NaN))
      end
    end
  end

  @testset "SumLargest(k, 0) needs k entries (E1)" begin
    for T in (Float32, Float64)
      f = SingularValue(SumLargest(5, zero(T)))
      @test_throws DimensionMismatch f(T[1 0; 0 2])
      @test_throws DimensionMismatch f(T[NaN 0; 0 2])
      @test_throws DimensionMismatch f(T[Inf 0; 0 2])
      @test_throws DimensionMismatch Spectral(SumLargest(5, zero(T)))(T[1 0; 0 2])
      @test_throws DimensionMismatch Spectral(SumLargest(5, zero(T)))(T[NaN 0; 0 2])
      @test_throws DimensionMismatch Spectral(SumLargest(5, one(T)))(T[NaN 0; 0 2])
      @test_throws DimensionMismatch value_and_prox!(zeros(T, 2, 2), f, T[NaN 0; 0 2], one(T))
      @test same(SingularValue(SumLargest(2, zero(T)))(T[NaN 0; 0 2]), zero(T))
      @test same(SingularValue(SumLargest(3, zero(T)))(T[NaN 0 0; 0 2 0; 0 0 1]), zero(T))
    end
  end

  @testset "device inputs" begin
    for T in (Float32, Float64), X in patterns(T)
      for f in (NuclearNorm(zero(T)), IndBallRank(minimum(size(X))), SingularValue(NormL1(zero(T))),
                SingularValue(IndBallL0(minimum(size(X)))), SingularValue(NormL1(one(T))))
        @test same(f(JLArray(X)), f(X))
      end
    end
  end
end

end
