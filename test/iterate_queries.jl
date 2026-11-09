module IterateQueriesTest

using Test
using ProximalBase
using SparseArrays
using LinearAlgebra

const PT = ProximalBase.Testing

# Explicit sorted export used only by these conformance tests. This is deliberately
# not a public SparseIterate method: solver sweeps use the unsorted active handles.
function _sorted_stored(x::SparseIterate{T,1}) where T
    p = sortperm(activecoordinates(x))
    (collect(activecoordinates(x))[p], collect(activevalues(x))[p])
end
function _sorted_stored(x::SparseIterate{T,2}) where T
    p = sortperm(activecoordinates(x))
    ids = collect(activecoordinates(x))[p]
    m = size(x, 1)
    ([(k - 1) % m + 1 for k in ids], [(k - 1) ÷ m + 1 for k in ids], collect(activevalues(x))[p])
end


function check_query_copies(x)
  result = _sorted_stored(x)
  saved = deepcopy(result)
  x[first(activecoordinates(x))] = 9
  @test result == saved
  entries = Array(x)
  coordinates = copy(activecoordinates(x))
  values = copy(activevalues(x))
  for a in result
    fill!(a, zero(eltype(a)))
  end
  @test Array(x) == entries
  @test activecoordinates(x) == coordinates
  @test activevalues(x) == values
end

@testset "SparseIterate queries" begin
  @testset "$T" for T in (Float64, Float32, Int)
    @testset "vector" begin
      x = SparseIterate(T, 7)
      x[6] = 4
      x[2] = -3
      activate!(x, 4)
      reference = sparsevec([6, 2, 4], T[4, -3, 0], 7)
      @test !issparse(x)
      @test all(f -> !applicable(f, x), (nnz, nonzeros, findnz))
      @test _sorted_stored(x) == findnz(reference)
      @test length(last(_sorted_stored(x))) == nactive(x) == 3
      @test eltype(last(_sorted_stored(x))) === T
      @test collect(zip(activecoordinates(x), activevalues(x))) == [(6, T(4)), (2, T(-3)), (4, T(0))]
      values = activevalues(x)
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
      @test !issparse(x)
      @test all(f -> !applicable(f, x), (nnz, nonzeros, findnz))
      @test _sorted_stored(x) == findnz(reference)
      @test length(last(_sorted_stored(x))) == nactive(x) == 3
      @test eltype(last(_sorted_stored(x))) === T
      @test collect(zip(activecoordinates(x), activevalues(x))) == [(12, T(4)), (2, T(-3)), (7, T(0))]
      values = activevalues(x)
      values[2] = 5
      @test x[2, 1] == T(5)
      x[3, 4] = 8
      @test values[1] == T(8)
      check_query_copies(x)
    end

    @testset "empty, $dims" for dims in ((0,), (5,), (0, 3), (3, 0), (3, 4))
      x = SparseIterate(T, dims...)
      result = _sorted_stored(x)
      @test length(result) == length(dims) + 1
      @test all(isempty, result)
      @test all(a -> eltype(a) === Int, result[1:end-1])
      @test eltype(last(result)) === T
      @test isempty(activevalues(x))
      @test eltype(activevalues(x)) === T
    end

    @testset "after dropzeros!" begin
      x = SparseIterate(T, 7)
      x[6] = 4
      activate!(x, 2)
      x[4] = -3
      dropzeros!(x)
      @test collect(zip(activecoordinates(x), activevalues(x))) == [(6, T(4)), (4, T(-3))]
      @test _sorted_stored(x) == findnz(sparsevec([6, 4], T[4, -3], 7))
    end
  end
end


# FT01: coordinates(x) of a SparseIterate is a coordinate-vector view, for both shapes
@testset "coordinates of a SparseIterate" begin
  ws(x) = collect(activecoordinates(x))
  x = SparseIterate(Float64, 2, 3)
  y = SparseIterate(Float64, 2, 3); y[1, 1] = 3
  v = SparseIterate(4); v[1] = 1
  w = SparseIterate([1.0, 2.0, 3.0, 4.0])

  c = coordinates(x)
  @test c isa AbstractVector{Float64}
  @test length(c) == 6 == ncoordinates(x)
  @test axes(c, 1) == 1:6
  @test c !== x
  c[4] = 2.0                        # 4 = (2 - 1) * 2 + 2, by hand
  @test x[2, 2] == 2.0
  @test ws(x) == [4]
  activate!(x, 6)
  @test x[2, 3] == 0.0 && ws(x) == [4, 6]
  @test_throws BoundsError c[7]
  @test_throws BoundsError c[0] = 1.0
  err = try c[7]; catch e; e; end
  @test err isa BoundsError && err.a isa AbstractVector && length(err.a) == 6 && err.i == 7

  cv = coordinates(v)
  @test cv isa AbstractVector{Float64} && cv !== v
  @test length(cv) == 4 && cv[1] == 1
  cv[3] = 5
  @test v[3] == 5
  @test Base.dataids(c) == Base.dataids(x)
  @test Base.mightalias(c, x) && !Base.mightalias(c, y) && !Base.mightalias(c, coordinates(y))

  # products through the view are ordinary-vector products (IEEE), unlike those of x
  @test isnan(dot([1, NaN, 0, 0], cv))
  @test isnan(dot([1, NaN, 0, 0, 0, 0], c))
  @test dot([1, NaN, 0, 0], v) == 1.0

  # a copy between two views is a value write, not a state copy
  copyto!(coordinates(x), coordinates(y))
  @test ws(x) == [4, 6, 1]
  @test Array(x) == [3.0 0 0; 0 0 0]
  @test ws(y) == [1]

  # an aliased import copies the source first
  copyto!(w, coordinates(w))
  @test Vector(w) == [1, 2, 3, 4]
  w = SparseIterate([1.0, 2.0, 3.0, 4.0])
  copyto!(w, 2, coordinates(w), 1, 3)
  @test Vector(w) == [1, 1, 2, 3]

  # the view is live and independent of the state copies
  z = SparseIterate(Float64, 2, 3); cz = coordinates(z)
  z[1, 2] = 7
  @test cz[3] == 7
  copyto!(z, y)
  @test cz[1] == 3 && cz[3] == 0

  # building the view allocates nothing
  @testset "$T" for T in (Float32, Float64)
    for shape in ((5,), (2, 3))
      a = SparseIterate(T, shape...); a[2] = 1
      @test PT.allocations(coordinates, a) == 0
    end
  end
end

end
