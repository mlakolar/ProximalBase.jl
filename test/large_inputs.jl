module LargeInputsTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays

function linear_value(a, b, x; half = false)
  T = float(eltype(x))
  A = a isa Real ? fill(BigFloat(a), length(x)) : vec(BigFloat.(a))
  X = vec(BigFloat.(x))
  gap = dot(A, X) - BigFloat(b)
  slack = BigFloat(sqrt(eps(T))) * max(1, abs(BigFloat(b)), norm(A) * norm(X))
  (half ? gap : abs(gap)) <= slack ? zero(T) : T(Inf)
end

# Adjacent floating-point inputs straddling an exact arithmetic boundary.
function neighbours(::Type{T}, boundary::BigFloat) where {T}
  lo = T(boundary)
  BigFloat(lo) >= boundary && (lo = prevfloat(lo))
  lo, nextfloat(lo)
end

function check_linear(a, b, x)
  for sign in (-1, 1), F in (IndHyperplane, IndHalfspace)
    normal, offset = sign .* a, sign * b
    @test F(normal, offset)(x) == linear_value(normal, offset, x; half = F === IndHalfspace)
  end
end

function check_shapes(a::Vector{T}, b::T, x::Vector{T}) where {T}
  check_linear(a, b, x)
  check_linear(reshape(a, 1, :), b, reshape(x, 1, :))
  all(==(first(a)), a) && check_linear(first(a), b, x)
end

@noinline function membership_batch(f, x, n)
  count = 0
  for _ in 1:n
    count += isfinite(f(x))
  end
  count
end

function minimum_value_time(f, x)
  membership_batch(f, x, 1)
  single = @elapsed membership_batch(f, x, 1)
  n = single < 1e-5 ? 64 : 1
  minimum(@elapsed(membership_batch(f, x, n)) for _ in 1:5) / n
end

@testset "large inputs" begin
  setprecision(BigFloat, 256) do
    for T in (Float32, Float64)
      @testset "$T" begin
        M = floatmax(T)
        overflow = BigFloat(M) + BigFloat(eps(M)) / 2

        @testset "individual square" begin
          lo, hi = neighbours(T, sqrt(overflow))
          @test nextfloat(lo) == hi
          @test isfinite(lo * lo)
          @test isinf(hi * hi)
          for z in (lo, hi)
            check_shapes(T[1], zero(T), T[z])
          end
        end

        @testset "sum of squares" begin
          lo, hi = neighbours(T, sqrt(overflow / 2))
          @test nextfloat(lo) == hi
          @test isfinite(hi * hi)
          @test isfinite(lo * lo + lo * lo)
          @test isinf(hi * hi + hi * hi)
          for z in (lo, hi)
            check_shapes(T[1, 0], zero(T), T[z, z])
          end
        end

        @testset "normal times squared norm" begin
          lo, hi = neighbours(T, sqrt(overflow / 4))
          @test nextfloat(lo) == hi
          @test isfinite(hi * hi)
          @test isfinite(T(4) * (lo * lo))
          @test isinf(T(4) * (hi * hi))
          for z in (lo, hi)
            check_shapes(T[2], zero(T), T[z])
          end
        end

        @testset "scalar sum" begin
          lo, hi = neighbours(T, overflow / 2)
          @test nextfloat(lo) == hi
          @test isfinite(lo + lo)
          @test isinf(hi + hi)
          for z in (lo, hi)
            check_linear(one(T), zero(T), T[z, z])
            check_linear(one(T), zero(T), fill(z, 1, 2))
          end
        end

        @testset "dot product" begin
          lo, hi = neighbours(T, overflow / 4)
          @test nextfloat(lo) == hi
          @test isfinite(T(2) * hi)
          @test isfinite(T(2) * lo + T(2) * lo)
          @test isinf(T(2) * hi + T(2) * hi)
          for z in (lo, hi)
            check_shapes(T[2, 2], zero(T), T[z, z])
          end
        end

        @testset "residual subtraction" begin
          b = -M / 2
          lo, hi = neighbours(T, overflow + BigFloat(b))
          @test nextfloat(lo) == hi
          @test isfinite(lo - b)
          @test isinf(hi - b)
          for z in (lo, hi)
            check_shapes(T[1], b, T[z])
          end
        end

        @testset "scaled normal offset" begin
          tiny = nextfloat(zero(T))
          lo, hi = neighbours(T, overflow * BigFloat(tiny))
          @test nextfloat(lo) == hi
          @test BigFloat(lo) / BigFloat(tiny) < overflow
          @test BigFloat(hi) / BigFloat(tiny) >= overflow
          for b in (lo, hi, one(T)), sign in (-1, 1)
            check_shapes(T[tiny], sign * b, T[0])
          end
        end

        @testset "norm above floatmax" begin
          lo, hi = neighbours(T, BigFloat(M) / sqrt(BigFloat(2)))
          @test nextfloat(lo) == hi
          @test norm(BigFloat[lo, lo]) < BigFloat(M)
          @test norm(BigFloat[hi, hi]) > BigFloat(M)
          c = ldexp(one(T), round(Int, 0.6 * exponent(M)))
          for z in (lo, hi), a in (one(T), c)
            check_shapes(fill(a, 2), zero(T), T[z, z])
          end
        end

        @testset "points outside sets" begin
          large = ldexp(one(T), round(Int, 0.7 * exponent(M)))
          @test IndHyperplane(T[1], zero(T))(T[large]) == Inf
          @test IndHalfspace(T[1], zero(T))(T[large]) == Inf
          @test IndHyperplane(one(T), zero(T))(T[large, large]) == Inf
          for equality in (false, true)
            @test IndSimplex(one(T); equality)(T[M, M]) == Inf
            @test IndSimplex(one(T); equality)(T[large, 1]) == Inf
          end
          @test IndAffine(Matrix{T}(I, 2, 2), fill(M * T(0.75), 2))(ones(T, 2)) == Inf
          A = T[1e10 0]
          for matrix in (A, sparse(A))
            @test IndAffine(matrix, T[0])(T[M / 16, 0]) == Inf
          end
          for symmetric in (false, true)
            @test IndToeplitz(; symmetric)(T[M * T(0.6) 0; 0 M * T(0.8)]) == Inf
          end
          for F in (IndBallL1, IndBallL2)
            @test F(M)(T[M, M]) == Inf
            @test F(one(T))(T[large, large]) == Inf
          end
        end

        @testset "points on sets" begin
          @test IndHyperplane(T[1, -1], zero(T))(T[M, M]) == 0
          @test IndHyperplane(one(T), zero(T))(T[M, -M]) == 0
          @test IndHalfspace(T[1], zero(T))(T[-M]) == 0
          @test IndHalfspace(T[-1], zero(T))(T[M]) == 0
          x = T[M / 2, M / 2, sqrt(eps(T)) * M / 4]
          @test abs(sum(BigFloat.(x)) - BigFloat(M)) < BigFloat(sqrt(eps(T))) * BigFloat(M)
          for equality in (false, true)
            @test IndSimplex(M; equality)(x) == 0
          end
          for A in (T[1 -1], sparse(T[1 -1]))
            @test IndAffine(A, T[0])(T[M, M]) == 0
          end
          @test IndToeplitz()(T[M M / 2; -M / 4 M]) == 0
          @test IndToeplitz(; symmetric = true)(T[M M / 2; M / 2 M]) == 0
          for F in (IndBallL1, IndBallL2)
            @test F(M)(T[prevfloat(M), 0]) == 0
          end
        end

        @testset "iterate values" begin
          large = ldexp(one(T), exponent(M) - 3)
          extreme = ldexp(one(T), round(Int, 0.6 * exponent(M)))
          for x in (T[large, 0, large, 0], T[large, 0, -large, 0])
            xs = SparseIterate(x)
            xs[2] = one(T)
            xs[2] = zero(T)
            @test nnz(xs) == 3
            xa = AtomIterate(x, [1:2, 3:4])
            X = reshape(x, 2, 2)
            Xm = SparseIterate(Matrix(X))
            Xm[2] = one(T)
            Xm[2] = zero(T)
            for c in (one(T), extreme), sign in (-1, 1), F in (IndHyperplane, IndHalfspace)
              for a in (sign * c, fill(sign * c, 4))
                f = F(a, zero(T))
                expected = linear_value(a, zero(T), x; half = F === IndHalfspace)
                @test f(x) == expected
                @test f(xs) == f(x)
                @test f(xa) == f(x)
              end
              for a in (sign * c, fill(sign * c, 2, 2))
                f = F(a, zero(T))
                @test f(X) == linear_value(a, zero(T), X; half = F === IndHalfspace)
                @test f(Xm) == f(X)
              end
            end
          end
          for c in (one(T), extreme), sign in (-1, 1), F in (IndHyperplane, IndHalfspace)
            for a in (T[1 0.5; 0.5 1.5], T[1 2; -1 1.5]), off in (false, true)
              X = T[large large / 2; large / 2 (off ? large : -large)]
              S = SymmetricSparseIterate(X)
              normal = (sign * c) .* a
              f = F(normal, zero(T))
              @test f(X) == linear_value(normal, zero(T), X; half = F === IndHalfspace)
              @test f(S) == f(X)
            end
            for off in (false, true)
              X = T[large large / 2; large / 2 (off ? large : -2large)]
              f = F(sign * c, zero(T))
              @test f(X) == linear_value(sign * c, zero(T), X; half = F === IndHalfspace)
              @test f(SymmetricSparseIterate(X)) == f(X)
            end
          end
        end

        @testset "other iterate values" begin
          for (x, expected) in ((T[M / 2, M / 2], zero(T)), (T[M, M], T(Inf)))
            for f in (IndSimplex(M), IndSimplex(M; equality = false), IndBallL1(M), IndBallL2(M))
              @test f(x) == expected
              @test f(SparseIterate(x)) == f(x)
              @test f(AtomIterate(x, [1:1, 2:2])) == f(x)
              X = T[x[1] 0; 0 x[2]]
              @test f(SparseIterate(X)) == f(X)
              @test f(SymmetricSparseIterate(X)) == f(X)
            end
          end
          for A in (T[1 -1], sparse(T[1 -1])), off in (false, true)
            f = IndAffine(A, T[0])
            x = T[M, off ? -M : M]
            @test f(x) == (off ? T(Inf) : zero(T))
            @test f(SparseIterate(x)) == f(x)
            @test f(AtomIterate(x, [1:1, 2:2])) == f(x)
          end
          for symmetric in (false, true), off in (false, true)
            f = IndToeplitz(; symmetric)
            X = T[M 0; 0 (off ? M / 2 : M)]
            @test f(X) == (off ? T(Inf) : zero(T))
            @test f(SparseIterate(X)) == f(X)
            @test f(SymmetricSparseIterate(X)) == f(X)
          end
        end

        @testset "empty and zero inputs" begin
          for F in (IndHyperplane, IndHalfspace), x in (T[], zeros(T, 0, 0), zeros(T, 2))
            @test F(one(T), zero(T))(x) == 0
          end
          for F in (IndBallL1, IndBallL2), x in (T[], zeros(T, 2))
            @test F(zero(T))(x) == 0
          end
          @test IndSimplex(one(T))(T[]) == Inf
          @test IndSimplex(one(T); equality = false)(T[]) == 0
          for equality in (false, true), x in (T[], zeros(T, 2))
            @test IndSimplex(zero(T); equality)(x) == 0
          end
          @test IndAffine(zeros(T, 0, 0), T[])(T[]) == 0
          @test IndAffine(T[1 -1], T[0])(zeros(T, 2)) == 0
          for symmetric in (false, true), X in (zeros(T, 0, 0), zeros(T, 2, 2))
            @test IndToeplitz(; symmetric)(X) == 0
          end
        end

        @testset "nonfinite entries" begin
          extreme = ldexp(one(T), round(Int, 0.6 * exponent(M)))
          for v in (T(NaN), T(Inf), T(-Inf))
            x = T[v, 0]
            for F in (IndHyperplane, IndHalfspace), c in (one(T), extreme), sign in (-1, 1)
              for a in (sign * c, T[sign * c, sign * c])
                f = F(a, zero(T))
                @test f(x) == Inf
                @test f(SparseIterate(x)) == Inf
                @test f(AtomIterate(x, [1:1, 2:2])) == Inf
              end
            end
            for F in (IndBallL1, IndBallL2)
              @test F(M)(x) == Inf
            end
            for equality in (false, true)
              @test IndSimplex(M; equality)(x) == Inf
            end
            @test IndAffine(T[1 0], T[0])(x) == Inf
            for symmetric in (false, true)
              @test IndToeplitz(; symmetric)(fill(v, 2, 2)) == Inf
            end
          end
        end
      end
    end

    @testset "floating point examples" begin
      @test IndHyperplane([1.0], 0.0)([1e200]) == Inf
      @test IndHalfspace([1.0], 0.0)([1e200]) == Inf
      @test IndHyperplane(1.0, 0.0)([1e200, 1e200]) == Inf
      @test IndHyperplane([2.0], 0.0)([1e154]) == Inf
      @test IndHyperplane([1.0], 0.0)([1e154]) == Inf
      @test IndHyperplane([1.0], 0.0)([1.5e154]) == Inf
      @test IndHyperplane([1f0], 0f0)([1f20]) == Inf
      @test IndHalfspace(1f0, 0f0)(Float32[1f20, 1f20]) == Inf
      xs = SparseIterate(4)
      xs[2] = 1e200
      @test IndHyperplane(ones(4), 0.0)(xs) == Inf
      @test IndHyperplane(ones(2, 2), 0.0)(SymmetricSparseIterate([1e200 0.0; 0.0 0.0])) == Inf
      @test IndSimplex(1e40; equality = false)(fill(1f38, 4)) == 0
      for equality in (false, true)
        @test IndSimplex(1.0; equality)([1e308, 1e308]) == Inf
      end
      @test IndAffine(Matrix(1.0I, 2, 2), [1.3e308, 1.3e308])([1.0, 1.0]) == Inf
      for A in ([1e10 0.0], sparse([1e10 0.0]))
        @test IndAffine(A, [0.0])([1e300, 0.0]) == Inf
      end
      fd = IndAffine(Float32[1 -1], Float32[0])
      fs = IndAffine(sparse(Float32[1 -1]), Float32[0])
      M = floatmax(Float64)
      for (x, expected) in (([M, M], 0.0), ([M, -M], Inf))
        @test fd(x) == expected
        @test fs(x) == expected
        @test fs(x) == fd(x)
      end
      for symmetric in (false, true)
        @test IndToeplitz(; symmetric)([1e308 0.0; 0.0 1.2e308]) == Inf
      end
      @test Constrained(NormL1(0.1), IndHyperplane(1.0, 0))([1e200, 1e200]) == Inf
      @test IndIntersection(IndBallL2(floatmax(Float64)), IndHyperplane(1.0, 0))([1e200, 1e200]) == Inf
    end

    @testset "integer inputs" begin
      for F in (IndHyperplane, IndHalfspace)
        @test F(Int64[4], 0)([Int64(1) << 62]) == Inf
        @test F(4, 0)([Int64(1) << 62]) == Inf
        @test F(Int[1, 1], 0)(Int[2, -2]) == 0
        @test F(1, 0)(Int[2, -2]) == 0
      end
      for equality in (false, true)
        @test IndSimplex(1; equality)(Int[1, 0]) == 0
        @test IndSimplex(1; equality)(fill(typemax(Int), 2)) == Inf
      end
      @test IndAffine([1 -1], [0])(Int[1, 1]) == 0
      @test IndToeplitz()(Int[1 2; 3 1]) == 0
      @test IndToeplitz(; symmetric = true)(Int[1 2; 2 1]) == 0
      for F in (IndBallL1, IndBallL2)
        @test F(2)(Int[1, 0]) == 0
        @test F(1)(Int[2, 2]) == Inf
      end
    end
  end

  @testset "sparse symmetric value cost" begin
    for c in (1.0, 1e200)
      @testset "normal $c" begin
        times = map((500, 4000)) do p
          x = SymmetricSparseIterate(p)
          x[1, 1] = 1.0
          x[2, 1] = 0.5
          @test nnz(x) == 2
          f = IndHyperplane(fill(c, p, p), 0.0)
          @test f(x) == Inf
          minimum_value_time(f, x)
        end
        @test times[2] <= 2times[1]
      end
    end
  end
end

end
