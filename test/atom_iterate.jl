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

# a real number whose conversion to Int succeeds once and then throws, to catch a second conversion
struct OnceReal <: Real
  count::Base.RefValue{Int}
end
Base.iszero(::OnceReal) = false
Base.convert(::Type{Int}, x::OnceReal) =
  (x.count[] += 1) == 1 ? 1 : throw(InexactError(:Int, Int, x))

# groups of unequal sizes, two not contiguous
groups = [1:3, [4, 7], 5:5, [8, 6], [9]]

mutable struct CountedIndices <: AbstractVector{Int}
  indices::Vector{Int}
  reads::Int
end
Base.size(G::CountedIndices) = size(G.indices)
Base.getindex(G::CountedIndices, i::Int) = (G.reads += 1; G.indices[i])

mutable struct CountedGroups <: AbstractVector{UnitRange{Int}}
  groups::Vector{UnitRange{Int}}
  comparisons::Int
end
Base.size(groups::CountedGroups) = size(groups.groups)
Base.getindex(groups::CountedGroups, k::Int) = groups.groups[k]
function Base.:(==)(a::CountedGroups, b::CountedGroups)
  a.comparisons += 1
  a.groups == b.groups
end

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

  @testset "scaling by Inf writes inactive groups (FN-9)" begin
    b = AtomIterate(4, [1:2, 3:4]); b[1] = 1
    @test rmul!(b, Inf) === b
    @test isequal(Vector(b), [Inf, NaN, NaN, NaN])
    @test collect(activecoordinates(b)) == [1, 2]
  end

  @testset "constructors" begin
    x = AtomIterate(9, groups)
    @test x isa AtomIterate{Float64, Vector{Vector{Int}}}     # mixed groups become vectors
    @test length(x) == 9 && size(x) == (9,)
    @test IndexStyle(typeof(x)) == IndexLinear()
    @test x == zeros(9)
    @test x.nactive == 0
    @test x.layout_cache === nothing
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

  @testset "layout cache" begin
    gs = CountedGroups([1:2, 3:4, 5:6], 0)
    fgs = CountedGroups(copy(gs.groups), 0)
    ygs = CountedGroups(copy(gs.groups), 0)
    f = GroupNormL2{Float64, CountedGroups}(0.4, fgs, ones(3))
    x = ProximalBase._AtomIterate([3., -4., 0., 0., 0., 0.], gs, [1, 1, 2, 2, 3, 3])
    y = ProximalBase._AtomIterate(ones(6), ygs, copy(x.group_of))
    activate!(x, 3)
    @test x.layout_cache === y.layout_cache === nothing
    @test f(x) ≈ f(Vector(x))
    @test f(y) ≈ f(Vector(y))
    @test x.layout_cache === y.layout_cache === f.groups
    @test gs.comparisons == ygs.comparisons == 1

    for _ in 1:3
      @test ProximalBase._samelayout(x, y)
      @test ProximalBase._samelayout(y, x)
      copyto!(y, x)
      @test active_set(y) == active_set(x)
      @test f(x) ≈ f(Vector(x))
      @test prox!(y, f, x, 0.7) === y
      @test Vector(y) ≈ prox(f, Vector(x), 0.7)
      @test value_and_prox!(y, f, x, 0.7) ≈ f(Vector(y))
      @test kkt_violation(f, ones(6), x, 2) ≈ sqrt(2) - 0.4
      prox_coordinate!(x, f, 1, 0.1)
      @test x.layout_cache === y.layout_cache === f.groups
    end
    @test gs.comparisons == ygs.comparisons == 1

    shared = AtomIterate(6, f)
    @test ProximalBase._samelayout(shared, x)
    @test ProximalBase._samelayout(x, shared)
    @test shared.layout_cache === nothing
    @test gs.comparisons == 1 && fgs.comparisons == 0

    for z in (copy(x), similar(x), similar(x, Float32))
      @test z.groups === x.groups && z.group_of === x.group_of
      @test z.layout_cache === f.groups
      @test ProximalBase._samelayout(f, z)
      @test f(z) ≈ f(Vector(z))
      reference = prox(f, Vector(z), 0.7)
      @test value_and_prox!(z, f, z, 0.7) ≈ f(reference)
      @test Vector(z) ≈ reference && z.layout_cache === f.groups
    end
    @test eltype(similar(x, Float32)) == Float32

    g = GroupNormL2(0.2, copy(gs.groups))
    @test ProximalBase._samelayout(g, y)
    copyto!(y, x)
    @test y == Vector(x) && consistent(y)
    @test y.layout_cache === g.groups && x.layout_cache === f.groups
    for h in (g, f, g, f)
      @test h(x) ≈ h(Vector(x))
      @test x.layout_cache === h.groups
    end

    bad = GroupNormL2(0.4, [1:3, 4:5, 6:6])
    before = copy(x)
    @test !ProximalBase._samelayout(bad, x)
    @test bad(x) ≈ bad(Vector(x))
    @test_throws DimensionMismatch prox_coordinate!(x, bad, 1, 0.7)
    @test_throws DimensionMismatch kkt_violation(bad, ones(6), x, 1)
    @test x == Vector(before) && activecoordinates(x) == activecoordinates(before)
    @test x.layout_cache === f.groups
    @test f(x) ≈ f(Vector(x))

    other = AtomIterate(ones(6), bad)
    @test other.layout_cache === nothing
    @test !ProximalBase._samelayout(shared, other)
    @test shared.layout_cache === other.layout_cache === nothing
    @test ProximalBase._samelayout(g, other) == false
    copyto!(other, x)
    @test other == Vector(x) && consistent(other)
    @test other.layout_cache === nothing && x.layout_cache === f.groups

    S = [1, 2, 5, 6]
    r = restrict(f, S)
    restricted = AtomIterate(Vector(x)[S], r)
    @test restricted.layout_cache === nothing
    @test restricted.groups === r.groups && restricted.groups !== x.groups
    @test r(restricted) ≈ r(Vector(restricted))
  end

  @testset "group wrappers preserve a failed coordinate step" begin
    gs = [1:2, 3:4, 5:6]
    other = [[1, 3], [2, 5], [4, 6]]
    v = [3., -4., 0., 0., 0., 0.]
    for F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2, GroupMCP, GroupSCAD)
      f = F === SparseGroupNormL2 ? F(0.2, 0.4, gs) : F(0.4, gs)
      bad = F === SparseGroupNormL2 ? F(0.2, 0.4, other) : F(0.4, other)
      for wrap in (h -> 2.0 * h, h -> Regularize(h, 0.3, ones(6)))
        x = AtomIterate(v, gs)
        activate!(x, 3)
        prox_coordinate!(x, wrap(f), 1, 0.7)
        reference = prox(wrap(f), v, 0.7)
        @test Vector(x)[gs[1]] ≈ reference[gs[1]]
        @test Vector(x)[3:6] == v[3:6]
        before = copy(x)
        @test_throws DimensionMismatch prox_coordinate!(x, wrap(bad), 1, 0.7)
        @test_throws DimensionMismatch kkt_violation(wrap(bad), ones(6), x, 1)
        @test x.storage == before.storage
        @test activecoordinates(x) == activecoordinates(before)
        @test x.position == before.position && x.layout_cache === f.groups
        prox_coordinate!(x, wrap(f), 1, 0.7)
        @test Vector(x)[gs[1]] ≈ prox(wrap(f), Vector(before), 0.7)[gs[1]]
      end
    end
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

  @testset "fill! decides activation after conversion (#421)" begin
    c = 1e-100                                   # nonzero as a Float64, zero as a Float32
    @test !iszero(c) && iszero(Float32(c))
    fresh() = AtomIterate(Float32, 4, 2)
    function check_none_active(x)
      @test Vector(x) == zeros(Float32, 4)
      @test nactive(x) == 0 && isempty(activecoordinates(x))
      @test !isactive(x, 1) && !isactive(x, 2)
      @test consistent(x)
    end

    @testset "underflow activates nothing" begin
      x = fresh()
      @test fill!(x, c) === x
      check_none_active(x)
      x = fresh()
      @test (x .= c) === x
      check_none_active(x)
      x = fresh()
      vec(x) .= c                                # vec of a vector is the vector itself
      check_none_active(x)
      x = fresh()
      x[1] = c                                   # control: setindex! converts first
      check_none_active(x)
    end

    @testset "other write paths of the same value" begin
      u = fill(c, 4)
      for write! in (x -> (x .= u), x -> (x .= 1 .* u), x -> copyto!(x, u), x -> (x[:] = u),
                     x -> copyto!(x, 2, u, 1, 2), x -> (view(x, 1:4) .= c),
                     x -> (atom(x, 1) .= c), x -> fill!(view(x, 1:2), c), x -> fill!(atom(x, 1), c))
        x = fresh()
        write!(x)
        check_none_active(x)
      end
    end

    @testset "a representable nonzero value activates every group" begin
      x = fresh()
      @test fill!(x, 0.1) === x
      @test Vector(x) == fill(0.1f0, 4)           # the converted value is stored
      @test active_set(x) == Set([1, 2]) && nactive(x) == 2
      @test consistent(x)
      x = fresh()
      x .= 2
      @test Vector(x) == fill(2f0, 4) && active_set(x) == Set([1, 2]) && consistent(x)
      x = fresh()                                # the smallest Float32 subnormal is nonzero
      fill!(x, 1e-45)
      @test all(==(nextfloat(0f0)), Vector(x)) && nactive(x) == 2
    end

    @testset "NaN and infinity" begin
      for v in (NaN, Inf, -Inf, NaN32, Inf32, 1e300)   # 1e300 converts to Float32 Inf
        x = fresh()
        fill!(x, v)
        @test all(isequal(Float32(v)), Vector(x))
        @test active_set(x) == Set([1, 2]) && consistent(x)
      end
    end

    @testset "signed zero" begin
      for v in (0.0, -0.0, 0, 0f0, -0f0)
        x = fresh()
        fill!(x, v)
        check_none_active(x)
      end
      x = fresh(); x .= -0.0
      check_none_active(x)
    end

    @testset "active groups are cleared by a value that converts to zero" begin
      # the 0.5.x rule: a zero fill! empties the working set (#424 changes it for 0.6)
      for clear! in (x -> fill!(x, c), x -> (x .= c))
        x = fresh()
        x[1], x[4] = 3, 5
        @test active_set(x) == Set([1, 2])
        @test clear!(x) === x
        check_none_active(x)
      end
      x = fresh()
      x[3] = 1
      fill!(x, 0.0)                              # same result as the unconverted zero
      check_none_active(x)
    end

    @testset "a failed conversion leaves the iterate unchanged" begin
      x = AtomIterate(Int, 6, [1:2, 3:4, 5:6])
      x[5], x[1] = 7, 4
      before = (copy(x.storage), copy(activecoordinates(x)), copy(x.position), x.nactive)
      for v in (1.5, 1e-100, NaN, Inf)
        @test_throws InexactError fill!(x, v)
        @test_throws InexactError (x .= v)
        @test (x.storage, collect(activecoordinates(x)), x.position, x.nactive) == before
        @test activecoordinates(x) == [3, 1]     # the order of the working set is kept
        @test consistent(x)
      end
    end

    @testset "input is converted exactly once" begin
      count = Ref(0)
      x = AtomIterate(Int, 4, 2)

      @test fill!(x, OnceReal(count)) === x
      @test count[] == 1
      @test Vector(x) == ones(Int, 4)
      @test collect(activecoordinates(x)) == [1, 2]
      @test consistent(x)
    end
  end

  @testset "atom views activate writes" begin
    gs = [1:2, 3:4, 5:6]
    a = AtomIterate(6, gs)
    atom(a, 2) .= 1.0
    @test nactive(a) == 1
    @test GroupNormL2(1.0, gs)(a) ≈ 2
    b = similar(a)
    copyto!(b, a)
    @test b[3:4] == [1, 1] && a == b
    @test parent(atom(a, 2)) === a
    @test Base.mightalias(atom(a, 1), a.storage)
    @test !Base.mightalias(atom(a, 1), b.storage)
    atom(a, 2)[1] = 0
    fill!(atom(a, 2), 0)
    @test isactive(a, 2)
    atom(a, 1)[1] = 0
    @test !isactive(a, 1)
    @test consistent(a)
    a = AtomIterate([1., 2., 3., 4., 5., 6.], gs)
    atom(a, 1) .= atom(a, 2)
    @test Vector(a) == [3, 4, 3, 4, 5, 6]
    expected = copy(a.storage)
    expected[3:4] .= copy(view(a, 2:3))
    atom(a, 2) .= view(a, 2:3)
    @test Vector(a) == expected && consistent(a)
  end

  @testset "group views respect bounds-check mode" begin
    G = CountedIndices([1, 3, 2, 4], 0)
    x = ProximalBase._AtomIterate(zeros(4), [G], ones(Int, 4))
    # forced bounds checking overrides @inbounds
    scans = Base.JLOptions().check_bounds == 1 ? length(G) : 0
    G.reads = 0
    v = atom(x, 1)
    @test G.reads == scans
    @test parent(v) === x
    raw = ProximalBase._atom_storage(x, 1)
    @test G.reads == 2scans
    @test parent(raw) === x.storage
    @test_throws BoundsError atom(x, 0)
    @test_throws BoundsError ProximalBase._atom_storage(x, 2)
  end

  @testset "group view norms and dots read storage" begin
    for T in (Float32, Float64), m in (4, 64), gs in ([1:m, (m+1):2m], [collect(1:2:2m), collect(2:2:2m)])
      x = AtomIterate(randn(T, 2m), gs)
      v, w = atom(x, 1), atom(x, 2)
      rawv, raww = view(x.storage, gs[1]), view(x.storage, gs[2])
      z = randn(T, m)
      for p in (1, 2, Inf)
        @test norm(v, p) === norm(rawv, p)
      end
      @test norm(v) === norm(rawv)
      @test dot(v, z) === dot(rawv, z)
      @test dot(z, v) === dot(z, rawv)
      @test dot(v, w) === dot(rawv, raww)
    end
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
    @test w.layout_cache === x.layout_cache === nothing
    u = AtomIterate(9, 3)
    copyto!(u, x)
    @test u == x && active_set(u) == Set([1, 2]) && consistent(u)

    # a conversion that throws leaves the destination unchanged
    xi = AtomIterate(9, groups); xi[1] = 1.0; xi[7] = 1.5     # groups 1 and 2 active, 1.5 does not convert
    di = AtomIterate(Int, 9, groups); di[5] = 7; di[9] = 3
    before = (Vector(di), copy(di.active[1:di.nactive]), copy(di.position), di.nactive)
    @test_throws InexactError copyto!(di, xi)
    @test (Vector(di), di.active[1:di.nactive], di.position, di.nactive) == before && consistent(di)
    de = AtomIterate(Int, 9, groups)                           # an empty destination gains no active group
    @test_throws InexactError copyto!(de, xi)
    @test de.nactive == 0 && Vector(de) == zeros(Int, 9) && consistent(de)
    # a pair with no conversion, built-in types only
    du = AtomIterate(Union{Float32, Float64}[0f0, 0.0], 2)
    @test_throws MethodError copyto!(du, AtomIterate([1, 0], 2))
    @test du.nactive == 0 && consistent(du)
    c = Ref(0)                                                  # one conversion per value
    ro = AtomIterate(Real, 9, groups); ro[1] = OnceReal(c)
    @test copyto!(AtomIterate(Int, 9, groups), ro)[1] == 1 && c[] == 1
    xi[7] = 2.0
    @test copyto!(di, xi) === di && Vector(di) == Vector(xi) && consistent(di)

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
             (Float64, Float32, Float64), (Float32, Float64, Float64))
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
          S = Ta
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

  @testset "atomindices (FT02)" begin
    B = [1 2; 3 4; 5 6]
    x = AtomIterate(vec(B), [collect(r:3:6) for r in 1:3])
    expected = ([1, 4], [2, 5], [3, 6])        # by hand
    for r in 1:3
      @test atomindices(x, r) == expected[r]
      @test atomindices(x, r) isa AbstractVector{<:Integer}
      @test eltype(atomindices(x, r)) == Int
      @test collect(atomindices(x, r)) == expected[r]
      @test length(atomindices(x, r)) == 2 && axes(atomindices(x, r)) == (Base.OneTo(2),)
    end
    y = AtomIterate(6, 3)
    @test atomindices(y, 2) == 3:4
    @test atomindices(y, 2) isa UnitRange{Int}
    @test atomindices(y, 2) === y.groups[2]          # the stored range itself

    @testset "invalid k" begin
      for z in (x, y), k in (0, 4, -1, typemin(Int), typemax(Int), typemax(UInt), false, true)
        @test_throws BoundsError atomindices(z, k)
      end
      @test atomindices(x, Int8(2)) == [2, 5] && atomindices(y, UInt16(3)) == 5:6
    end

    @testset "read-only" begin
      layout = deepcopy(x.groups)
      group_of = copy(x.group_of)
      cache = x.layout_cache
      h = atomindices(x, 2)
      function attempts(h)
        n = length(h)
        src = fill(9, n)
        (() -> (h[1] = 9), () -> (h[end] = 9), () -> fill!(h, 9), () -> copyto!(h, src),
         () -> (h .= 9), () -> (h .= src), () -> map!(identity, h, src),
         () -> sort!(h; rev = true), () -> reverse!(h), () -> (h[1:1] = [9]),
         () -> (view(h, 1:1)[1] = 9), () -> (h[:] = src))
      end
      for attempt in attempts(h)
        @test_throws Exception attempt()
        @test x.groups == layout && x.group_of == group_of && x.layout_cache === cache
        @test atomindices(x, 2) == [2, 5]
      end
      @test_throws ArgumentError h[1] = 9
      @test_throws MethodError push!(h, 9)
      @test_throws MethodError resize!(h, 5)
      # a group of length 1 has nothing to write: the layout is unchanged
      z = AtomIterate(9, groups)
      single = atomindices(z, 5)
      @test single == [9]
      for op in (sort!, reverse!)
        try op(single) catch end
      end
      @test z.groups == groups && single == [9]

      # copies of the handle are ordinary arrays and leave the layout alone
      c = copy(h)
      @test c isa Vector{Int} && c == [2, 5]
      c[1] = 7
      @test sortperm(h) == [1, 2] && sortperm(atomindices(x, 3)) == [1, 2]
      @test collect(h) == [2, 5] && Vector(h) == [2, 5] && h == [2, 5] && [2, 5] == h
      @test h[2:2] == [5] && h[:] == [2, 5]
      @test_throws BoundsError h[0]
      @test_throws BoundsError h[3]
      # the exceptions of atomindices and of its wrapper name the valid range or the wrapper, not a stored
      # group (field access, `h.ids`, is internal and unsupported; atom(x, k) is unchanged)
      for i in (0, 3, 5, Int32(5), typemax(Int))
        err = try h[i]; nothing catch e e end
        @test err isa BoundsError && err.a === h && !(err.a isa Array)
      end
      for k in (0, 4, false, true, -1, typemax(UInt))
        err = try atomindices(x, k); nothing catch e e end
        @test err isa BoundsError && !(err.a isa Array) && err.a !== x.groups
      end
      @test x.groups == layout
      @test x.groups == layout && atomindices(x, 2) == [2, 5]
      @test Base.IndexStyle(typeof(h)) == IndexLinear()

      # aliasing: the handle reports the group's memory, a copy does not
      @test Base.mightalias(h, h)
      @test Base.mightalias(h, atomindices(x, 2))
      @test !Base.mightalias(h, atomindices(x, 1))
      @test !Base.mightalias(h, copy(h))
      @test !Base.mightalias(h, x)
      @test Base.dataids(h) == Base.dataids(x.groups[2])
      u = Base.unaliascopy(h)
      @test u == h && !Base.mightalias(u, h) && x.groups == layout
    end

    @testset "views indexed by atomindices" begin
      z = AtomIterate(9, groups)
      dense = collect(10.0:18.0)
      for k in 1:length(groups)
        h = atomindices(z, k)
        @test dense[h] == dense[groups[k]]
        w = view(dense, h)
        @test w == dense[groups[k]] && sum(w) == sum(dense[groups[k]])
        @test parentindices(w)[1] === h
        @test view(z, h) == z[groups[k]]
      end
      layout = deepcopy(z.groups)
      group_of = copy(z.group_of)
      @test dense[atomindices(z, 2)] == [13.0, 16.0]
      z[4] = 3.0
      @test view(z, atomindices(z, 2)) == [3.0, 0.0]
      z .= 2
      fill!(z, 0)
      z[atomindices(z, 2)] .= 5.0            # a write to the entries, not to the indices
      @test z[[4, 7]] == [5.0, 5.0] && isactive(z, 2)
      for i in (1, 4)
        z[i] = 1.0
      end
      @test z.groups == layout && z.group_of == group_of
      @test all(k -> atomindices(z, k) == groups[k], 1:length(groups))
      # the vector that built the iterate is not the layout
      vals = collect(1.0:9.0)
      zz = AtomIterate(vals, groups)
      vals .= 0
      @test zz[3] == 3.0 && atomindices(zz, 1) == 1:3
    end

    @testset "retained handles stay valid" begin
      z = AtomIterate(9, groups)
      snapshot = deepcopy(z.groups)
      handles = [atomindices(z, k) for k in 1:length(groups)]
      activate!(z, 2)
      z[1] = 4.0
      z[8] = 1.0
      @test nactive(z) == 3
      z[1] = 0.0
      dropzeros!(z)
      other = AtomIterate(9, groups)         # an equal, separately built layout
      other[5] = 2.0
      copyto!(z, other)                      # state replacement
      c, s = copy(z), similar(z)
      fill!(z, 0.0)
      dropzeros!(z)
      @test nactive(z) == 0
      for k in 1:length(groups)
        fresh = atomindices(z, k)
        @test handles[k] == snapshot[k] && fresh == snapshot[k]
        @test length(handles[k]) == length(groups[k])
        @test typeof(handles[k]) == typeof(fresh)
        @test atomindices(c, k) == snapshot[k] && atomindices(s, k) == snapshot[k]
      end
    end

    @testset "index types (FC-6)" begin
      ai = AtomIterate(4, [Int32(1):Int32(2), Int32(3):Int32(4)])
      @test atomindices(ai, 1) == 1:2
      @test atomindices(ai, 1) isa UnitRange{Int32}
      @test atomindices(ai, 1) === ai.groups[1]
      f32 = GroupNormL2(1.0, [Int32(1):Int32(2), Int32(3):Int32(4)])
      @test atomindices(AtomIterate(4, f32), 2) isa UnitRange{Int32}
      @test atomindices(AtomIterate(4, f32), 2) == 3:4
      av = AtomIterate(4, [Int32[1, 3], Int32[2, 4]])
      @test av.groups isa Vector{Vector{Int}}
      @test eltype(atomindices(av, 1)) == Int
      @test atomindices(av, 1) == [1, 3] && atomindices(av, 2) == [2, 4]
      @test eltype(atomindices(AtomIterate(4, [Int8[1, 2], Int8[3, 4]]), 1)) == Int
      mixed = AtomIterate(4, [1:2, [3, 4]])      # mixed groups are stored as vectors
      @test eltype(atomindices(mixed, 1)) == Int && atomindices(mixed, 1) == [1, 2]
      @test atomindices(AtomIterate(4, [1:2:3, 2:2:4]), 1) isa StepRange{Int, Int}
    end

    @testset "ownership of the caller's arrays" begin
      gv = [[1, 2], [3, 4]]
      a = AtomIterate(4, gv)
      gv[1][1] = 3
      @test atomindices(a, 1) == [1, 2]
    end

    @testset "atom views" begin
      x = AtomIterate(vec(B), [collect(r:3:6) for r in 1:3])
      for r in 1:3, c in 1:2
        @test x[(c - 1) * 3 + r] == B[r, c]
      end
      @test all(r -> atom(x, r) == B[r, :], 1:3)
      @test_throws MethodError coordinates(x)
      w = atom(x, 2)
      activate!(x, 1)
      dropzeros!(x)
      @test w == B[2, :]
      w[1] = 0
      @test x[2] == 0
    end

    @testset "allocations (FT23)" begin
      # measured inside functions: at global scope the result is boxed
      index_allocs(x, k) = @allocated atomindices(x, k)
      function index_sum(x)
        s = 0
        for k in 1:ncoordinates(x), j in atomindices(x, k)
          s += j
        end
        s
      end
      loop_allocs(x) = @allocated index_sum(x)
      for T in (Float32, Float64)
        xr = AtomIterate(T, 6, 3)                           # ranges
        xv = AtomIterate(T, 6, [[1, 4], [2, 5], [3, 6]])    # vectors
        x32 = AtomIterate(T, 4, [Int32(1):Int32(2), Int32(3):Int32(4)])
        for z in (xr, xv, x32)
          index_allocs(z, 2); loop_allocs(z)
          @test index_allocs(z, 2) == 0
          @test index_allocs(z, 1) == 0
          @test loop_allocs(z) == 0
        end
        @test index_sum(xr) == 21 && index_sum(xv) == 21 && index_sum(x32) == 10
        activate!(xv, 2)
        dropzeros!(xv)
        @test index_allocs(xv, 3) == 0
      end
    end
  end

  @testset "samegroups (FT17)" begin
    samegroups = ProximalBase.samegroups
    @test !Base.isexported(ProximalBase, :samegroups)
    @test Base.isexported(ProximalBase, :atomindices)
    gv = [[1, 2], [3, 4]]
    a = AtomIterate(4, gv)

    @testset "penalties" begin
      same = GroupNormL2(1, [[1, 2], [3, 4]])
      for (u, v) in ((a, same), (same, a))
        @test samegroups(u, v) === true
      end
      for bad in (GroupNormL2(1, [[2, 1], [3, 4]]),        # ordered lists
                  GroupNormL2(1, [[3, 4], [1, 2]]),        # group order
                  GroupNormL2(1, [[1, 3], [2, 4]]),        # grouped differently
                  GroupNormL2(1, [[1, 2, 3, 4]]),          # fewer groups
                  GroupNormL2(1, [[1], [2], [3, 4]]),      # more groups
                  GroupNormL2(1, [[1, 2], [3]]))
        @test samegroups(a, bad) === false
        @test samegroups(bad, a) === false
      end
      # every group-bearing function
      for f in (GroupNormL2(1, [1:2, 3:4]), GroupNormLinf(1, [1:2, 3:4]), GroupMCP(1.0, [1:2, 3:4]),
                GroupSCAD(1.0, [1:2, 3:4]), SparseGroupNormL2(0.5, 1.0, [1:2, 3:4]),
                IndBallGroupL0(1, [1:2, 3:4]), IndBallGroupL0(1, [[1, 2], [3, 4]]; k = 1))
        @test samegroups(a, f) === true && samegroups(f, a) === true
      end
      for f in (GroupNormLinf(1, [[2, 1], [3, 4]]), GroupMCP(1.0, [1:3, 4:4]),
                GroupSCAD(1.0, [[1, 3], [2, 4]]), SparseGroupNormL2(0.5, 1.0, [[1, 2, 3], [4]]),
                IndBallGroupL0(1, [[1, 3], [2, 4]]))
        @test samegroups(a, f) === false && samegroups(f, a) === false
      end
      # no method for anything else
      @test_throws MethodError samegroups(a, 1)
      @test_throws MethodError samegroups(1, a)
      @test_throws MethodError samegroups(a, IndBallL1(1.0))
      @test_throws MethodError samegroups(same, same)
      @test_throws MethodError samegroups(a, [[1, 2], [3, 4]])
    end

    @testset "iterates" begin
      @test samegroups(a, copy(a)) === true
      @test samegroups(a, similar(a)) === true
      @test samegroups(a, a) === true
      @test samegroups(a, AtomIterate(4, [[1, 2], [3, 4]])) === true
      @test samegroups(a, AtomIterate(4, [[2, 1], [3, 4]])) === false
      @test samegroups(a, AtomIterate(4, [[1, 3], [2, 4]])) === false
      @test samegroups(a, AtomIterate(4, [1:4])) === false
      @test samegroups(a, AtomIterate(4, 4)) === false
      @test samegroups(a, AtomIterate(8, [1:2, 3:4, 5:6, 7:8])) === false
      @test samegroups(AtomIterate(4, [[3, 4], [1, 2]]), a) === false
      # the same ordered lists in other representations
      r = AtomIterate(4, [1:2, 3:4])
      r32 = AtomIterate(4, [Int32(1):Int32(2), Int32(3):Int32(4)])
      v32 = AtomIterate(4, [Int32[1, 2], Int32[3, 4]])
      for (u, v) in ((a, r), (r, a), (a, r32), (r32, a), (r, r32), (r32, r), (v32, r32), (r32, v32), (v32, a))
        @test samegroups(u, v) === true
      end
      @test samegroups(AtomIterate(4, [1:2, [3, 4]]), a) === true
      # equality is of ordered lists, not of sets
      @test samegroups(AtomIterate(4, [[2, 1], [3, 4]]), AtomIterate(4, [[1, 2], [3, 4]])) === false
      @test samegroups(AtomIterate(4, [[1, 2], [4, 3]]), a) === false
      # an unequal comparison writes nothing
      b = AtomIterate(4, [[1, 2], [3, 4]])
      @test samegroups(a, b) && a.layout_cache === b.groups
      bad = AtomIterate(4, [[1, 3], [2, 4]])
      before = a.layout_cache
      @test !samegroups(a, bad) && a.layout_cache === before && bad.layout_cache === nothing
      @test !samegroups(bad, a) && a.layout_cache === before && bad.layout_cache === nothing
    end

    counted() = CountedGroups([1:2, 3:4, 5:6], 0)
    mkiterate() = ProximalBase._AtomIterate(zeros(6), counted(), [1, 1, 2, 2, 3, 3])
    mkpenalty() = GroupNormL2{Float64, CountedGroups}(0.4, counted(), ones(3))

    @testset "counted comparisons, iterate and penalty" begin
      x = mkiterate()
      f1, f2 = mkpenalty(), mkpenalty()
      shared = ProximalBase._AtomIterate(zeros(6), f1.groups, [1, 1, 2, 2, 3, 3])
      # a shared layout object: identity, no comparison, nothing remembered
      @test samegroups(shared, f1) && samegroups(f1, shared)
      @test f1.groups.comparisons == 0 && shared.layout_cache === nothing
      # a distinct equal layout: one comparison on first sight, then none
      @test x.layout_cache === nothing
      @test samegroups(x, f1) === true
      @test x.groups.comparisons == 1 && x.layout_cache === f1.groups
      for _ in 1:4
        @test samegroups(x, f1) && samegroups(f1, x)
      end
      @test x.groups.comparisons == 1 && f1.groups.comparisons == 0
      # copies and similar iterates share what is remembered
      for z in (copy(x), similar(x), similar(x, Float32))
        @test samegroups(z, f1) && samegroups(f1, z) && z.layout_cache === f1.groups
      end
      @test x.groups.comparisons == 1
      # two distinct equal layouts alternating: one comparison per call
      for i in 1:6
        @test samegroups(x, isodd(i) ? f2 : f1)
        @test x.groups.comparisons == 1 + i
        @test x.layout_cache === (isodd(i) ? f2 : f1).groups
      end
      # the penalty first is the same question: a hit on f1, and one comparison on f2
      @test samegroups(f1, x) && x.groups.comparisons == 7 && x.layout_cache === f1.groups
      @test samegroups(f2, x) && x.groups.comparisons == 8 && x.layout_cache === f2.groups
      # an unequal layout: false, counted, and the remembered layout stays
      bad = GroupNormL2{Float64, CountedGroups}(0.4, CountedGroups([1:3, 4:5, 6:6], 0), ones(3))
      @test samegroups(x, bad) === false && samegroups(bad, x) === false
      @test x.layout_cache === f2.groups
      @test x.groups.comparisons == 10
      @test samegroups(x, f2) && x.groups.comparisons == 10
      # the coordinate oracles use the same cache
      @test kkt_violation(f2, ones(6), x, 2) ≈ sqrt(2) - 0.4
      @test x.groups.comparisons == 10
      @test_throws DimensionMismatch kkt_violation(bad, ones(6), x, 2)
    end

    @testset "counted comparisons, two iterates" begin
      a, b, c = mkiterate(), mkiterate(), mkiterate()
      @test samegroups(a, b) === true
      @test a.groups.comparisons == 1 && b.groups.comparisons == 0
      @test a.layout_cache === b.groups && b.layout_cache === nothing
      for _ in 1:3
        @test samegroups(a, b)
        @test samegroups(b, a)                      # a remembers b's layout: a hit from either side
      end
      @test a.groups.comparisons == 1 && b.groups.comparisons == 0
      @test a.layout_cache === b.groups && b.layout_cache === nothing
      # two distinct equal layouts alternating on the remembering iterate: a comparison per call
      for i in 1:6
        @test samegroups(a, isodd(i) ? c : b)
        @test a.groups.comparisons == 1 + i
        @test a.layout_cache === (isodd(i) ? c : b).groups
      end
      # a failed comparison keeps the remembered layout
      bad = ProximalBase._AtomIterate(zeros(5), [1:2, 3:4, 5:5], [1, 1, 2, 2, 3])
      remembered = a.layout_cache
      @test samegroups(a, bad) === false && samegroups(bad, a) === false
      @test a.layout_cache === remembered && bad.layout_cache === nothing
      # iterates that remember the same validated layout agree without a comparison and without a write
      f = mkpenalty()
      p, q = mkiterate(), mkiterate()
      @test samegroups(p, f) && samegroups(q, f)
      @test p.groups.comparisons == 1 && q.groups.comparisons == 1
      @test samegroups(p, q) && samegroups(q, p)
      @test p.groups.comparisons == 1 && q.groups.comparisons == 1
      @test p.layout_cache === f.groups && q.layout_cache === f.groups
      # the same layout object: no comparison, nothing written
      r = mkiterate()
      s = copy(r)
      @test samegroups(s, r) && samegroups(r, s)
      @test r.groups.comparisons == 0 && r.layout_cache === nothing && s.layout_cache === nothing
    end

    @testset "allocations of samegroups" begin
      # warmed, on ordinary layout arrays (ranges, vectors, Int32 ranges, penalties), inside functions
      hit_allocs(a, b) = @allocated samegroups(a, b)
      fresh_allocs(a, b) = (a.layout_cache = nothing; @allocated samegroups(a, b))
      function alternate_allocs(a, f1, f2)
        samegroups(a, f1)
        @allocated (samegroups(a, f2); samegroups(a, f1))
      end
      for T in (Float32, Float64)
        ranges() = [1:4, 5:8, 9:12]
        vectors() = [collect(1:4), collect(5:8), collect(9:12)]
        strided() = [collect(r:3:12) for r in 1:3]
        r32() = [Int32(1):Int32(4), Int32(5):Int32(8), Int32(9):Int32(12)]
        for (name, mk) in (("ranges", ranges), ("vectors", vectors), ("strided", strided), ("Int32 ranges", r32))
          a, b, c = AtomIterate(T, 12, mk()), AtomIterate(T, 12, mk()), AtomIterate(T, 12, mk())
          f1, f2 = GroupNormL2(1.0, mk()), GroupNormL2(1.0, mk())
          bad = AtomIterate(T, 12, [[1, 2, 3, 5], [4, 6, 7, 8], [9, 10, 11, 12]])
          # the groups of ranges() in another representation: equal for ranges, vectors and Int32 ranges
          d = AtomIterate(T, 12, name == "ranges" ? vectors() : ranges())
          for (u, v) in ((a, b), (b, a), (a, f1), (f1, a), (a, a), (a, copy(a)))
            hit_allocs(u, v); fresh_allocs(u isa AtomIterate ? u : v, u isa AtomIterate ? v : u)
          end
          hit_allocs(a, bad); hit_allocs(a, d); alternate_allocs(a, f1, f2)
          @test hit_allocs(a, a) == 0
          @test hit_allocs(a, copy(a)) == 0
          @test fresh_allocs(a, b) == 0 && hit_allocs(a, b) == 0 && hit_allocs(b, a) == 0
          @test fresh_allocs(a, f1) == 0 && hit_allocs(a, f1) == 0 && hit_allocs(f1, a) == 0
          @test fresh_allocs(c, f2) == 0 && hit_allocs(f2, c) == 0
          @test hit_allocs(a, bad) == 0 && hit_allocs(bad, a) == 0
          @test alternate_allocs(a, f1, f2) == 0           # a recompare at every call
          @test hit_allocs(a, d) == 0 && hit_allocs(d, a) == 0
        end
      end
    end

    @testset "copyto! does not remember" begin
      u = AtomIterate(4, [[1, 2], [3, 4]])
      v = AtomIterate(4, [[1, 2], [3, 4]])
      v[1] = 1.0
      f = GroupNormL2(1.0, [[1, 2], [3, 4]])
      @test samegroups(u, f) && u.layout_cache === f.groups
      copyto!(u, v)
      @test u == v && u.layout_cache === f.groups && v.layout_cache === nothing
      w = AtomIterate(4, [[1, 2], [3, 4]])
      copyto!(w, v)
      @test w == v && w.layout_cache === nothing && v.layout_cache === nothing
    end

    @testset "whole-array oracles accept any layout (FL-5)" begin
      fm = GroupNormL2(1, [[1, 3], [2, 4]])
      am = AtomIterate([1, 2, 3, 4], [1:2, 3:4])
      @test !samegroups(am, fm)
      @test fm(am) == fm(Vector(am))
      ym = AtomIterate(4, [1:2, 3:4])
      @test prox!(ym, fm, am, 1) === ym
      @test Vector(ym) == prox(fm, Vector(am), 1)
      @test_throws DimensionMismatch prox_coordinate!(am, fm, 1, 0.5)
    end
  end

end

@testset "aliasing of the views of an AtomIterate" begin
  function alloc_unaliased(y, x)
    y .= x .+ 1
    copyto!(y, x)
    return nothing
  end
  function alloc_aliased(x)
    copyto!(x, x.storage)
    return nothing
  end
  a = AtomIterate(6, 3)
  for k in (1, 3)
    activate!(a, k)
    a[a.groups[k]] .= k
  end
  @test Base.mightalias(a, activecoordinates(a))
  @test Base.mightalias(activecoordinates(a), a)
  @test Base.mightalias(a, view(a, 1:2))
  @test Base.mightalias(a, reshape(a, :))
  @test all(k -> Base.mightalias(a, atom(a, k)), 1:3)
  @test all(id -> id in Base.dataids(a), Base.dataids(activecoordinates(a)))

  b = copy(a)    # shares the groups
  @test !Base.mightalias(a, b)
  @test !Base.mightalias(a, AtomIterate(6, 3))
  @test !Base.mightalias(a, zeros(6))
  @test !Base.mightalias(activecoordinates(a), b)
  @test !Base.mightalias(atom(a, 1), b)

  source = copy(a)
  copyto!(a, a)
  @test a == source
  a .= a .+ 0
  @test a == source

  alloc_unaliased(b, a)
  @test (@allocated alloc_unaliased(b, a)) == 0
  v = view(a, 1:6)
  alloc_unaliased(v, a)
  @test (@allocated alloc_unaliased(v, a)) > 0
end

end
