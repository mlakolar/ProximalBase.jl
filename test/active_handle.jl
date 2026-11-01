module ActiveHandleTest

# activecoordinates returns a read-only handle: a write throws and leaves the iterate
# unchanged (issue #405).

using Test
using ProximalBase
using ..TestUtils

function make_iterates(T)
  x = SparseIterate(T, 6); x[2] = 1; x[5] = 3
  m = SparseIterate(T, 4, 4); m[2, 1] = 1; m[3, 3] = 2
  s = SymmetricSparseIterate(T, 4); s[2, 1] = 1; s[3, 2] = 2
  a = AtomIterate(T, 6, [1:2, 3:4, 5:6]); a[1] = 1; a[5] = 2
  (vector = x, matrix = m, symmetric = s, atom = a)
end

snapshot(x) = (copy(activecoordinates(x)), Array(x), nactive(x))

sum_ids(x) = (acc = 0; for k in activecoordinates(x); acc += k; end; acc)

@testset "activecoordinates is read-only" begin
  @testset "$T $name" for T in (Float64, Float32), (name, x) in pairs(make_iterates(T))
    h = activecoordinates(x)
    @test h isa AbstractVector{Int}
    @test length(h) == nactive(x)
    @test h == copy(h) && copy(h) isa Vector{Int}
    @test sortperm(h) == sortperm(collect(h))
    @test Base.dataids(h) == Base.dataids(activecoordinates(x)) && Base.mightalias(h, activecoordinates(x))
    @test !Base.mightalias(h, copy(h))
    @test_throws BoundsError h[nactive(x) + 1]
    before = snapshot(x)
    @test_throws ArgumentError h[1] = 3
    @test_throws ArgumentError fill!(h, 1)
    @test_throws ArgumentError sort!(h; rev=true)   # the fixtures are ascending: force a write
    @test_throws ArgumentError reverse!(h)
    @test_throws ArgumentError (h .= 1)
    @test_throws ArgumentError copyto!(h, fill(1, length(h)))
    after = snapshot(x)
    @test after == before
    @test activecoordinates(x) == before[1]
    # the state still works: dropzeros! returns, activation is seen by a new handle
    dropzeros!(x)
    @test nactive(x) == before[3]
    @test sum_ids(x) == sum(before[1])
  end

  @testset "the handle is a valid index into the iterate it belongs to" begin
    s = SymmetricSparseIterate(3); s[2, 1] = 7
    @test collect(view(coordinates(s), activecoordinates(s))) == [7.0]
    x = SparseIterate(4); x[2] = 1; x[4] = 3
    @test collect(view(x, activecoordinates(x))) == [1.0, 3.0]
    a = AtomIterate(4, [1:2, 3:4]); a[1] = 1
    @test collect(view(a, activecoordinates(a))) == [1.0]
  end

  @testset "a new handle sees an activation" begin
    x = SparseIterate(5); x[1] = 1
    h = activecoordinates(x)
    x[4] = 2
    @test length(h) == 1
    @test activecoordinates(x) == [1, 4]
    a = AtomIterate(4, [1:2, 3:4]); a[1] = 1
    h = activecoordinates(a); a[3] = 1
    @test length(h) == 1 && activecoordinates(a) == [1, 2]
  end

  @testset "a handle taken before copyto! is stale" begin
    x = SparseIterate(5); x[2] = 1; x[4] = 2
    y = SparseIterate(5); y[4] = 1; y[2] = 2
    h = activecoordinates(x)
    copyto!(x, y)
    @test collect(h) == [4, 2] && activecoordinates(x) == [4, 2]   # same set, new order
  end

  @testset "values stay writable" begin
    x = SparseIterate(5); x[1] = 1; x[4] = 2
    activevalues(x)[2] = 7
    @test x[4] == 7
    a = AtomIterate(4, [1:2, 3:4]); a[1] = 1
    atom(a, 1)[2] = 5
    @test a[2] == 5
  end

  @testset "$T a sweep through the handle does not allocate" for T in (Float64, Float32), (name, x) in pairs(make_iterates(T))
    sum_ids(x)
    @test allocs(sum_ids, x) == 0
  end
end

end
