module IterateQueriesTest

using Test
using ProximalBase
using SparseArrays

function check_query_copies(x)
  result = findnz(x)
  saved = deepcopy(result)
  x[first(activecoordinates(x))] = 9
  @test result == saved
  entries = Array(x)
  coordinates = copy(activecoordinates(x))
  values = copy(nonzeros(x))
  for a in result
    fill!(a, zero(eltype(a)))
  end
  @test Array(x) == entries
  @test activecoordinates(x) == coordinates
  @test nonzeros(x) == values
end

@testset "SparseIterate queries" begin
  @testset "$T" for T in (Float64, Float32, Int, ComplexF32)
    @testset "vector" begin
      x = SparseIterate(T, 7)
      x[6] = 4
      x[2] = -3
      activate!(x, 4)
      reference = sparsevec([6, 2, 4], T[4, -3, 0], 7)
      @test findnz(x) == findnz(reference)
      @test length(last(findnz(x))) == nnz(x) == 3
      @test eltype(last(findnz(x))) === T
      @test collect(zip(activecoordinates(x), nonzeros(x))) == [(6, T(4)), (2, T(-3)), (4, T(0))]
      values = nonzeros(x)
      values[2] = 5
      @test x[2] == T(5)
      x[6] = 8
      @test values[1] == T(8)
      check_query_copies(x)
      @test Array(SparseIterate(reference)) == Array(reference)
      @test eltype(SparseIterate(reference)) === T
    end

    @testset "matrix" begin
      x = SparseIterate(T, 3, 4)
      x[3, 4] = 4
      x[2, 1] = -3
      activate!(x, 7)
      reference = sparse([3, 2, 1], [4, 1, 3], T[4, -3, 0], 3, 4)
      @test findnz(x) == findnz(reference)
      @test length(last(findnz(x))) == nnz(x) == 3
      @test eltype(last(findnz(x))) === T
      @test collect(zip(activecoordinates(x), nonzeros(x))) == [(12, T(4)), (2, T(-3)), (7, T(0))]
      values = nonzeros(x)
      values[2] = 5
      @test x[2, 1] == T(5)
      x[3, 4] = 8
      @test values[1] == T(8)
      check_query_copies(x)
    end

    @testset "empty, $dims" for dims in ((0,), (5,), (0, 3), (3, 0), (3, 4))
      x = SparseIterate(T, dims...)
      result = findnz(x)
      @test length(result) == length(dims) + 1
      @test all(isempty, result)
      @test all(a -> eltype(a) === Int, result[1:end-1])
      @test eltype(last(result)) === T
      @test isempty(nonzeros(x))
      @test eltype(nonzeros(x)) === T
    end

    @testset "after dropzeros!" begin
      x = SparseIterate(T, 7)
      x[6] = 4
      activate!(x, 2)
      x[4] = -3
      dropzeros!(x)
      @test collect(zip(activecoordinates(x), nonzeros(x))) == [(6, T(4)), (4, T(-3))]
      @test findnz(x) == findnz(sparsevec([6, 4], T[4, -3], 7))
    end
  end
end

end
