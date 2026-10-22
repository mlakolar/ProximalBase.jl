module AtomIterateTest

using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)

# the active groups of x, as a set
active_set(x::AtomIterate) = Set(x.active[1:x.nactive])

# the bookkeeping of the active groups is consistent, and inactive groups are zero
function consistent(x::AtomIterate)
  K = length(x.groups)
  0 <= x.nactive <= K || return false
  all(a -> x.position[x.active[a]] == a, 1:x.nactive) || return false
  count(!iszero, x.position) == x.nactive || return false
  all(k -> x.position[k] != 0 || all(iszero, x.storage[x.groups[k]]), 1:K)
end

# groups of unequal sizes, two not contiguous
groups = [1:3, [4, 7], 5:5, [8, 6], [9]]

@testset "AtomIterate" begin

  @testset "constructors" begin
    x = AtomIterate(9, groups)
    @test x isa AtomIterate{Float64, Vector{Vector{Int}}}     # mixed groups become vectors
    @test length(x) == 9 && size(x) == (9,)
    @test IndexStyle(typeof(x)) == IndexLinear()
    @test x == zeros(9)
    @test x.nactive == 0
    @test numCoordinates(x) == 5
    @test x.group_of == [1, 1, 1, 2, 3, 4, 2, 4, 5]
    @test eltype(AtomIterate(Float32, 9, groups)) == Float32

    # ranges stay ranges; the groups are copied
    rs = [1:4, 5:10]
    y = AtomIterate(10, rs)
    @test y.groups isa Vector{UnitRange{Int}}
    rs[1] = 1:2
    @test y.groups == [1:4, 5:10]

    # from a vector: the groups with a nonzero entry are active
    v = [0., 0., 0., 0., 0., 2., 0., -1., 0.]
    x = AtomIterate(v, groups)
    @test x == v
    @test active_set(x) == Set([4])
    v[1] = 5.
    @test x[1] == 0.                       # not aliased
    @test eltype(AtomIterate(Float32[1, 0, 0, 0, 0, 0, 0, 0, 0], groups)) == Float32

    # equal contiguous groups
    @test AtomIterate(12, 3).groups == [1:4, 5:8, 9:12]
    @test eltype(AtomIterate(Float32, 12, 3)) == Float32
    @test_throws ArgumentError AtomIterate(10, 3)

    # the groups must partition 1:p
    @test_throws ArgumentError AtomIterate(10, groups)         # misses index 10
    @test_throws ArgumentError AtomIterate(8, groups)          # index 9 is out of range
    @test_throws ArgumentError AtomIterate(4, [1:3, 3:4])      # overlapping
    @test_throws ArgumentError AtomIterate(3, [1:3, Int[]])    # empty
    @test_throws ArgumentError AtomIterate(3, [0:2])
  end

  @testset "entries and active groups" begin
    x = AtomIterate(9, groups)
    x[7] = 2.
    @test x[7] == 2. && x[4] == 0.
    @test active_set(x) == Set([2])
    x[5] = 0.                              # a zero does not activate its group
    @test active_set(x) == Set([2])
    @test activate!(x, 3) === x
    activate!(x, 3)
    @test x.nactive == 2
    @test dropzeros!(x) === x
    @test active_set(x) == Set([2])        # group 3 is zero
    x[7] = 0.
    @test active_set(x) == Set([2])        # zeros stay active until dropzeros!
    dropzeros!(x)
    @test x.nactive == 0
    @test_throws BoundsError activate!(x, 6)

    x[9], x[2] = 1., 2.
    @test numActive(x) == 2
    @test [activeCoordinate(x, i) for i in 1:2] == [5, 1]
    @test isActive(x, 5) && !isActive(x, 2)
    @test_throws BoundsError activeCoordinate(x, 3)
    fill!(x, 0.)
    @test_throws BoundsError x[10]
    @test_throws BoundsError (x[0] = 1.)

    @test atom(x, 2) isa SubArray
    x[4], x[7] = 1., 3.
    @test atom(x, 2) == [1., 3.]
  end

  @testset "random operations" begin
    # every step agrees with a plain vector and a set of active groups
    x, v, act = AtomIterate(9, groups), zeros(9), Set{Int}()
    ok = true
    for step in 1:2000
      op = rand()
      if op < 0.5
        i = rand(1:9)
        val = rand() < 0.3 ? 0. : randn()
        x[i] = val
        v[i] = val
        iszero(val) || push!(act, x.group_of[i])
      elseif op < 0.7
        k = rand(1:5)
        activate!(x, k)
        push!(act, k)
      elseif op < 0.95
        dropzeros!(x)
        filter!(k -> any(!iszero, v[groups[k]]), act)
      else
        fill!(x, 0.)
        fill!(v, 0.)
        empty!(act)
      end
      ok &= x == v && active_set(x) == act && consistent(x)
      ok &= Set(activeCoordinate(x, i) for i in 1:numActive(x)) == act
      ok &= all(k -> isActive(x, k) == (k in act), 1:5)
    end
    @test ok
  end

  @testset "copies" begin
    x = AtomIterate([1., 0., 0., 0., 2., 0., 0., 0., 0.], groups)
    activate!(x, 5)                        # an active group of zeros

    y = copy(x)
    @test y == x && y.active[1:y.nactive] == x.active[1:x.nactive]
    @test y.groups === x.groups
    y[1] = 7.
    @test x[1] == 1.

    z = similar(x)
    @test z == zeros(9) && z.nactive == 0 && z.groups === x.groups
    @test eltype(similar(x, Float32)) == Float32

    @test copyto!(z, x) === z
    @test z == x && z.active[1:z.nactive] == x.active[1:x.nactive] && consistent(z)
    z[9] = 4.
    copyto!(z, x)                          # the entries of z's other groups are cleared
    @test z == x && consistent(z)
    @test copyto!(z, z) == x               # copying onto itself changes nothing

    # equal groups from another construction take the same path; other groups copy the entries
    w = AtomIterate(9, groups)
    copyto!(w, x)
    @test w == x && active_set(w) == active_set(x)
    u = AtomIterate(9, 3)
    copyto!(u, x)
    @test u == x && active_set(u) == Set([1, 2]) && consistent(u)

    @test fill!(y, 0.) == zeros(9) && y.nactive == 0
    fill!(y, 1.5)
    @test y == fill(1.5, 9) && y.nactive == 5

    @test Vector(x) == x.storage && !(Vector(x) === x.storage)
  end

  @testset "show" begin
    x = AtomIterate(9, groups)
    @test sprint(show, MIME"text/plain"(), x) == "9-element $(typeof(x)) with 0 of 5 groups active"
    x[7], x[1] = 2., 1.
    str = sprint(show, MIME"text/plain"(), x)
    @test startswith(str, "9-element $(typeof(x)) with 2 of 5 groups active:")
    @test occursin("1: [1, 2, 3] => [1.0, 0.0, 0.0]", str)
    @test occursin("2: [4, 7] => [0.0, 2.0]", str)
  end

  @testset "allocations" begin
    x = AtomIterate(9, groups)
    v = randn(9)
    function write_all!(x, v)
      @inbounds for i in eachindex(v)
        x[i] = v[i]
      end
      x
    end
    read_all(x) = (s = 0.; @inbounds for i in eachindex(x); s += x[i]; end; s)
    write_all!(x, v); read_all(x); activate!(x, 2); dropzeros!(x)
    @test (@allocated write_all!(x, v)) == 0
    @test (@allocated read_all(x)) == 0
    @test (@allocated activate!(x, 2)) == 0
    @test (@allocated dropzeros!(x)) == 0
  end

end

end
