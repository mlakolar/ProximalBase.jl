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

function check_scaled_product(out, A, x, α, β)
  @nospecialize out A x α β
  expected = Array(out)
  mul!(expected, A, Array(x), α, β)
  @test parentmodule(which(mul!, typeof.((out, A, x, α, β)))) == ProximalBase
  @test mul!(out, A, x, α, β) === out
  @test out ≈ expected
end

function check_product_state(x, before)
  @nospecialize x before
  @test x.storage == before.storage
  @test activecoordinates(x) == activecoordinates(before)
  @test x.position == before.position
end

@testset "AtomIterate" begin

  @testset "constructors" begin
    x = AtomIterate(9, groups)
    @test x isa AtomIterate{Float64, Vector{Vector{Int}}}     # mixed groups become vectors
    @test length(x) == 9 && size(x) == (9,)
    @test IndexStyle(typeof(x)) == IndexLinear()
    @test x == zeros(9)
    @test x.nactive == 0
    @test ncoordinates(x) == 5
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
    @test nactive(x) == 2
    @test activecoordinates(x) == [5, 1]
    @test isactive(x, 5) && !isactive(x, 2)
    @test_throws BoundsError activecoordinates(x)[3]
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
      ok &= Set(activecoordinates(x)) == act
      ok &= all(k -> isactive(x, k) == (k in act), 1:5)
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

  @testset "products" begin
    n, p = 7, 9
    X = randn(n, p)
    x = AtomIterate(9, groups)
    x[7], x[1], x[3] = 2., -1., 0.5
    activate!(x, 3)                        # an active group of zeros
    xd = Vector(x)

    out = fill(NaN, n)
    @test mul!(out, X, x) === out
    @test out ≈ X * xd
    @test X * x ≈ X * xd
    @test mul!(fill(NaN, n), X, AtomIterate(9, groups)) == zeros(n)
    @test_throws DimensionMismatch mul!(zeros(n), X[:, 1:8], x)
    @test_throws DimensionMismatch mul!(zeros(n + 1), X, x)

    v = randn(p)
    @test dot(v, x) ≈ dot(v, xd)
    @test dot(x, v) ≈ dot(xd, v)
    y = AtomIterate(randn(p), [1:4, 5:9])
    @test dot(x, y) ≈ dot(xd, Vector(y))
    @test dot(SparseIterate(v), x) ≈ dot(v, xd)
    @test all(row -> dot(view(X, row, :), x) ≈ (X * xd)[row], 1:n)

    x32 = AtomIterate(Float32.(xd), groups)
    @test mul!(zeros(Float32, n), Float32.(X), x32) ≈ Float32.(X * xd)

    # groups of one coordinate give the products of a SparseIterate
    xs = SparseIterate(xd)
    xa = AtomIterate(xd, [[i] for i in 1:p])
    @test X * xa ≈ X * xs
    @test dot(v, xa) ≈ dot(v, xs)
    @test activecoordinates(xa) == activecoordinates(xs)

    mul_allocs(out, X, x) = @allocated mul!(out, X, x)
    dot_allocs(v, x) = @allocated dot(v, x)
    mul_allocs(out, X, x); dot_allocs(v, x)
    @test mul_allocs(out, X, x) == 0
    @test dot_allocs(v, x) == 0
  end

  @testset "five-argument products" begin
    n, p = 7, 9
    types = ((Float32, Float32, Float32), (Float64, Float64, Float64),
             (Float64, Float32, Float64), (Float32, Float64, Float64),
             (ComplexF64, Float64, ComplexF64), (ComplexF64, ComplexF64, ComplexF64))
    for (Ta, Tx, To) in types, support in (:empty, :partial, :full), strided in (false, true)
      strided && !(Ta == Tx == To && Ta <: Real) && continue
      @testset "$Ta × $Tx → $To, $support, views=$strided" begin
        A = strided ? view(randn(Ta, 2n, 2p), 1:2:2n, 1:2:2p) : randn(Ta, n, p)
        dense = randn(Tx, p)
        support == :empty && fill!(dense, 0)
        support == :partial && (dense[1:2:end] .= 0)
        for G in (groups,)
          x = AtomIterate(dense, G)
          support == :partial && activate!(x, length(G))
          before = copy(x)
          out = strided ? view(randn(To, 2n), 1:2:2n) : randn(To, n)
          initial = copy(out)
          S = real(Ta)
          for α in S.((0, 1, 2.5)), β in S.((0, 1, -0.5))
            copyto!(out, initial)
            iszero(β) && fill!(out, NaN)
            check_scaled_product(out, A, x, α, β)
          end
          if support == :partial && !strided
            for (α, β) in ((true, false), (0, 0))
              fill!(out, NaN)
              check_scaled_product(out, A, x, α, β)
            end
          end
          if To <: Complex
            copyto!(out, initial)
            check_scaled_product(out, A, x, 1 + 0.5im, -0.5 + 0.25im)
          end
          check_product_state(x, before)
        end
      end
    end
    A = randn(n, p)
    x = AtomIterate(randn(p), groups)
    before = copy(x)
    out = view(zeros(2n), collect(1:2:2n))
    @test !(out isa StridedVector)
    for α in (0.0, 1.0, 2.5), β in (0.0, 1.0, -0.5)
      fill!(out, iszero(β) ? NaN : 3.)
      check_scaled_product(out, A, x, α, β)
    end
    check_product_state(x, before)
    x = AtomIterate(randn(p), [1:3, 4:6, 7:9])
    before = copy(x)
    for α in (0.0, 1.0, 2.5), β in (0.0, 1.0, -0.5)
      out = fill(iszero(β) ? NaN : 3., n)
      check_scaled_product(out, A, x, α, β)
    end
    check_product_state(x, before)
  end

  @testset "five-argument zero coefficients" begin
    A = [Inf 1. 3. 4.; 2. 5. Inf 7.]
    for G in ([1:2, 3:4], [[2, 1], [4, 3]])
      x = AtomIterate([0., 2., 0., 0.], G)
      activate!(x, 2)
      before = copy(x)
      reference = mul!(fill(NaN, 2), A, x)
      @test reference == [2., 10.]
      @test any(isnan, A * Array(x))
      for α in (1.0, 2.5), β in (0.0, 1.0, -0.5)
        out = fill(iszero(β) ? NaN : 3., 2)
        expected = iszero(β) ? α .* reference : α .* reference .+ β * 3.
        @test mul!(out, A, x, α, β) == expected
      end
      x[2] = NaN
      @test mul!(fill(NaN, 2), A, x, 0.0, 0.0) == zeros(2)
      @test mul!(fill(3., 2), A, x, 0.0, -0.5) == fill(-1.5, 2)
      x[2] = before[2]
      @test all(isnan, mul!(fill(NaN, 2), A, x, 1.0, 1.0))
      check_product_state(x, before)
    end
  end

  @testset "five-argument product dimensions" begin
    A = zeros(3, 9)
    x = AtomIterate(9, groups)
    for α in (0.0, 2.5), β in (0.0, -0.5)
      out = fill(3., 3)
      @test_throws DimensionMismatch("A has second dimension 8, length of x is 9") mul!(out, A[:, 1:8], x, α, β)
      @test out == fill(3., 3)
      @test_throws DimensionMismatch("A has first dimension 3, length of out is 2") mul!(zeros(2), A, x, α, β)
    end
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
    # measured inside functions: called from the test scope, Julia 1.10 boxes a
    # returned Float64
    write_allocs(x, v) = @allocated write_all!(x, v)
    read_allocs(x) = @allocated read_all(x)
    activate_allocs(x, k) = @allocated activate!(x, k)
    drop_allocs(x) = @allocated dropzeros!(x)
    write_allocs(x, v); read_allocs(x); activate_allocs(x, 2); drop_allocs(x)
    @test write_allocs(x, v) == 0
    @test read_allocs(x) == 0
    @test activate_allocs(x, 2) == 0
    @test drop_allocs(x) == 0
  end

end

end
