module BlockReplacementTest
using Test, ProximalBase, SparseArrays, Random

function check(x, values, ids)
    @test isequal(Vector(x), values)
    @test collect(activecoordinates(x)) == ids
    @test nactive(x) == length(ids)
    for k in 1:ncoordinates(x)
        p = findfirst(==(k), ids)
        @test x.position[k] == (p === nothing ? 0 : p)
        @test isactive(x, k) == (p !== nothing)
        p === nothing && @test all(iszero, values[collect(blockindices(x, k))])
    end
end

@testset "whole-iterate replacement and independent state references" begin
    layouts = ([1:3,4:6,7:9,10:12], [1:4:12,2:4:12,3:4:12,4:4:12],
               [[9,1,5],[10,2,6],[11,3,7],[12,4,8]], [k:12:12 for k in 1:12],
               [Int32(k):Int32(k+2) for k in 1:3:12],
               [Int32(k):Int32(4):Int32(12) for k in 1:4])
    for gs in layouts, T in (Float32,Float64), S in (Float32,Float64),
        ownership in (:shared,:independent,:cached), populated in (false,true), op in (copyto!,copy!)
        src = BlockIterate(S,12,gs)
        expected = zeros(T,12)
        ids = [length(gs),2,1]
        for (a,k) in enumerate(ids)
            activate!(src,k)
            for (j,i) in enumerate(gs[k])
                value = a == 2 ? -zero(S) : S((a+j)/8)
                src[i] = value
                expected[i] = T(value)
            end
        end
        dest = ownership == :shared ? similar(src,T) : BlockIterate(T,12,gs)
        # Distinct penalty cache objects must survive either path.
        src.layout_cache = deepcopy(gs)
        dest.layout_cache = deepcopy(gs)
        ownership == :cached && @test ProximalBase.samegroups(dest,src)
        scache,dcache = src.layout_cache,dest.layout_cache
        populated && foreach(k -> (dest[first(gs[k])] = T(7)), reverse(eachindex(gs)))
        h = blockindices(dest,1)
        @test op(dest,src) === dest
        check(dest,expected,ids)
        @test src.layout_cache === scache
        @test dest.layout_cache === dcache
        @test collect(h) == collect(gs[1])
        @test dest.groups === (ownership == :shared ? src.groups : dest.groups)
        ownership != :shared && @test dest.groups !== src.groups
        @test !Base.mightalias(dest,src)
        @test dest.storage !== src.storage
        @test dest.active !== src.active && dest.position !== src.position
        before = copy(expected)
        @test op(dest,dest) === dest
        check(dest,before,ids)
        copied = copy(src)
        check(copied,S.(expected),ids)
        @test copied.groups === src.groups && copied.group_of === src.group_of
        @test copied.layout_cache === scache
        @test !Base.mightalias(copied,src)
        # A subsequent write activates an absent group; pruning removes active
        # zeros while maintaining the reverse map and leaving all values intact.
        absent = 3
        dest[first(gs[absent])] = T(9)
        expected[first(gs[absent])] = T(9)
        check(dest,expected,vcat(ids,absent))
        dropzeros!(dest)
        check(dest,expected,[ids[1],absent,ids[3]])
        @test isequal(Vector(src),S.(before))
    end
end

@testset "empty source, layout cache hits, and equal descriptor representations" begin
    for T in (Float32,Float64), op in (copyto!,copy!),
        gs in ([1:3,4:6,7:9], [1:3:9,2:3:9,3:3:9], [[7,1,4],[8,2,5],[9,3,6]])
        src=BlockIterate(T,9,gs)
        # A different descriptor type can still express exactly the same layout.
        dest=BlockIterate(T.(1:9),[collect(g) for g in gs])
        @test ProximalBase.samegroups(dest,src)
        @test op(dest,src) === dest
        check(dest,zeros(T,9),Int[])
        dest[first(gs[2])]=T(4)
        @test op(dest,src) === dest
        check(dest,zeros(T,9),Int[])
        # Both caches can remember a third validated layout (a penalty).
        shared=deepcopy(src.groups)
        src.layout_cache=shared;dest.layout_cache=shared
        src[first(gs[3])]=T(2)
        op(dest,src)
        check(dest,Vector(src),[3])
        @test src.layout_cache === dest.layout_cache === shared
    end
end

@testset "different compatible layouts retain ordinary value-write semantics" begin
    for T in (Float32,Float64), S in (Float32,Float64), op in (copy!,copyto!),
        dg in ([[1,5,9],[2,6,10],[3,7,11],[4,8,12]], [1:6,7:12])
        src = BlockIterate(S,12,[1:3,4:6,7:9,10:12])
        src[8] = S(3); activate!(src,1)
        dest = BlockIterate(T,12,dg)
        dest[3] = T(7) # retained zero activity after generic fallback
        ref = copy(dest)
        for i in 1:12; ref[i] = src[i]; end
        @test op(dest,src) === dest
        check(dest,Vector(ref),collect(activecoordinates(ref)))
        dropzeros!(dest)
        check(dest,T.(Vector(src)),[findfirst(g -> 8 in g,dg)])
    end
end

@testset "zero-length replacement and empty conversion staging" begin
    for T in (Float32,Float64,Int), S in (Float32,Float64), op in (copyto!,copy!)
        src=BlockIterate(S,0,UnitRange{Int}[])
        for dest in (similar(src,T),BlockIterate(T,0,Vector{Int}[]))
            @test op(dest,src) === dest
            check(dest,T[],Int[])
        end
    end
end

@testset "replacement preserves nonfinite values and signed active zeros" begin
    for T in (Float32,Float64), S in (Float32,Float64), op in (copyto!,copy!)
        gs=[[6,1],[5,2],[4,3]]
        src=BlockIterate(S,6,gs)
        src[6]=S(NaN);src[1]=S(Inf);src[5]=S(-Inf)
        activate!(src,3);src[4]=-zero(S)
        dest=similar(src,T)
        @test op(dest,src) === dest
        check(dest,T.(Vector(src)),[1,2,3])
        dropzeros!(dest)
        check(dest,T.(Vector(src)),[1,2])
    end
end

@testset "mixed float conversion retains groups that underflow to zero" begin
    src=BlockIterate(Float64,4,[1:2,3:4])
    src[3]=1e-100;src[4]=-1e-100;src[1]=1e100
    dest=similar(src,Float32)
    @test copyto!(dest,src) === dest
    check(dest,Float32[Inf,0,0,-0.0],[2,1])
    dropzeros!(dest)
    check(dest,Float32[Inf,0,0,-0.0],[1])
end

@testset "destination-only groups reuse one constructed zero per group" begin
    gs=[1:3,4:6,7:9]
    src=BlockIterate(BigFloat,9,gs);src[8]=BigFloat(2.5)
    dest=BlockIterate(BigFloat.(1:9),gs)
    @test copyto!(dest,src) === dest
    check(dest,Vector(src),[3])
    @test dest.storage[1] === dest.storage[2] === dest.storage[3]
    @test dest.storage[4] === dest.storage[5] === dest.storage[6]
    @test dest.storage[1] !== dest.storage[4]
    @test dest.storage[8] === src.storage[8]
end

@testset "preflight errors preserve complete destination and layout caches" begin
    for gs in ([1:3,4:6,7:9], [1:3:9,2:3:9,3:3:9], [[7,1,4],[8,2,5],[9,3,6]]),
        op in (copyto!,copy!)
        src = BlockIterate(Float64,9,gs)
        src[first(gs[3])] = 2.0
        src[last(gs[2])] = 0.5 # fail after a successfully converted group
        dest = BlockIterate(Int,9,gs)
        dest[first(gs[1])] = 4; activate!(dest,3)
        dest.layout_cache = deepcopy(gs)
        state = (copy(dest.storage),copy(dest.active),copy(dest.position),dest.nactive,dest.layout_cache)
        @test_throws InexactError op(dest,src)
        @test (dest.storage,dest.active,dest.position,dest.nactive) == state[1:4]
        @test dest.layout_cache === state[5]
        src[last(gs[2])] = 5.0
        @test op(dest,src) === dest
        check(dest,Int.(Vector(src)),[3,2])
        before = copy(dest)
        @test_throws DimensionMismatch op(dest,BlockIterate(8,2))
        check(dest,Vector(before),collect(activecoordinates(before)))
    end
end

# An independent model of replacement between equal layouts with a conversion that cannot
# throw: groups active in src take src's converted entries and src's activation order,
# groups active only in dest become +0, and groups inactive in both keep dest's entries.
function model_replacement(dvals, dids, svals, sids, gs, ::Type{T}) where {T}
    vals = copy(dvals)
    for k in dids
        k in sids && continue
        for i in gs[k]; vals[i] = zero(T); end
    end
    for k in sids, i in gs[k]
        vals[i] = convert(T, svals[i])
    end
    vals, copy(sids)
end

const REPLACEMENT_LAYOUTS = ([1:3,4:6,7:9,10:12], [1:4:12,2:4:12,3:4:12,4:4:12],
    [[9,1,5],[10,2,6],[11,3,7],[12,4,8]], [k:k for k in 1:12], [k:12:12 for k in 1:12],
    [[i] for i in (5,3,12,1,7,9,2,11,4,10,6,8)], [Int32(k):Int32(k+2) for k in 1:3:12],
    [[1,7],[2,3,4,5,6],[8],[9,10,11,12]])

@testset "fully active sources match the model: values, order, maps, ownership" begin
    for gs in REPLACEMENT_LAYOUTS,
        (T,S) in ((Float64,Float64),(Float32,Float32),(Float32,Float64),(Float64,Float32),(Float64,Int)),
        ownership in (:shared,:independent,:cached), populated in (:empty,:partial,:full), op in (copyto!,copy!)
        K = length(gs)
        src = BlockIterate(S,12,gs)
        order = circshift(collect(K:-1:1), 1)      # activation order differs from group order
        for (a,k) in enumerate(order)
            activate!(src,k)                        # group order[1] stays an active zero group
            for (j,i) in enumerate(gs[k])
                src[i] = a == 1 ? -zero(S) : S <: AbstractFloat && a == 2 && j == 1 ? S(NaN) :
                         S <: AbstractFloat && a == 3 && j == 1 ? S(-Inf) : S(a % 2 == 0 ? -(a+j) : a+2j)
            end
        end
        @test nactive(src) == K
        dest = ownership == :shared ? similar(src,T) : BlockIterate(T,12,gs)
        src.layout_cache = deepcopy(gs); dest.layout_cache = deepcopy(gs)
        ownership == :cached && @test ProximalBase.samegroups(dest,src)
        if populated != :empty
            for k in (populated == :full ? (1:K) : (1:2:K)); dest[first(gs[k])] = T(7); end
            # an inactive destination group that holds -0.0
            populated == :partial && K > 1 && (dest[first(gs[2])] = -zero(T))
        end
        dvals, dids = copy(dest.storage), collect(activecoordinates(dest))
        owned = (dest.storage, dest.active, dest.position)
        caches = (src.layout_cache, dest.layout_cache)
        groups0 = dest.groups
        svals, sids = copy(src.storage), collect(activecoordinates(src))
        @test op(dest,src) === dest
        vals, ids = model_replacement(dvals, dids, svals, sids, gs, T)
        check(dest, vals, ids)
        check(src, svals, sids)
        @test (dest.storage, dest.active, dest.position) === owned
        @test (src.layout_cache, dest.layout_cache) === caches
        @test dest.groups === groups0
        @test (dest.groups === src.groups) == (ownership == :shared)
        @test !Base.mightalias(dest,src)
        # later writes and pruning follow the usual rules
        dest[first(gs[order[1]])] = T(5)
        vals[first(gs[order[1]])] = T(5)
        check(dest, vals, ids)
        dropzeros!(dest)
        @test nactive(dest) == count(k -> any(!iszero, vals[collect(gs[k])]), 1:K)
    end
end

@testset "fully active BigFloat sources share their entry objects" begin
    gs = [1:2,3:3,4:6]
    src = BlockIterate(BigFloat,6,gs)
    for k in (3,1,2), i in gs[k]; src[i] = BigFloat(i)/3; end
    src[3] = -zero(BigFloat)                        # group 2 stays active with -0.0
    for dest in (BlockIterate(BigFloat.(10:15),gs), BlockIterate(BigFloat,6,gs), similar(src))
        @test copyto!(dest,src) === dest
        check(dest, Vector(src), [3,1,2])
        @test all(i -> dest.storage[i] === src.storage[i], 1:6)
        @test signbit(dest[3])
    end
end

# The seven-argument constructor accepts metadata vectors longer than the number of
# groups and arrays that share memory between iterates. Expected states follow the
# group-by-group replacement: dest's ids, positions and entries are written in src's
# activation order, and metadata entries past the number of groups are left alone.
@testset "fully active sources with spare or shared metadata arrays" begin
    gs, go = [1:1,2:2], [1,2]
    # spare capacity in the active list, the positions, or both, of src and of dest
    for (sa, sp) in (([2,1,0],[2,1]), ([2,1],[2,1,9]), ([2,1,0],[2,1,9])),
        (da, dp) in (([1,0],[1,0]), ([1,0,7],[1,0,8]))
        src = BlockIterate([1.0,-0.0], gs, go, nothing, copy(sa), copy(sp), 2)
        dest = BlockIterate([5.0,0.0], gs, go, nothing, copy(da), copy(dp), 1)
        @test copyto!(dest,src) === dest
        check(dest, [1.0,-0.0], [2,1])
        @test dest.active[3:end] == da[3:end] && dest.position[3:end] == dp[3:end]
        @test src.active == sa && src.position == sp
    end
    # the active list of dest is the memory of src's positions: the same array, or
    # another array header over it
    gs3, go3 = [1:1,2:2,3:3], [1,2,3]
    for alias in (identity, v -> vec(reshape(v, length(v), 1)))
        src = BlockIterate([1.0,2.0,3.0], gs3, go3, nothing, [3,1,2], [2,3,1], 3)
        da = alias(src.position)
        @test Base.mightalias(da, src.position)
        dest = BlockIterate([4.0,5.0,6.0], gs3, go3, nothing, da, [3,1,2], 3)
        @test copyto!(dest,src) === dest
        @test dest.storage == [1.0,2.0,3.0] && dest.nactive == 3
        @test dest.active == [3,1,2] && dest.position == [2,3,1]
    end
    # value memory shared with src, through the same array or another header, and
    # independent metadata: every entry is src's
    for alias in (identity, v -> vec(reshape(v, length(v), 1)))
        src = BlockIterate([1.0,-0.0,3.0], gs3, go3, nothing, [2,3,1], [3,1,2], 3)
        dest = BlockIterate(alias(src.storage), gs3, go3, nothing, [1,2,3], [1,2,3], 3)
        @test copyto!(dest,src) === dest
        check(dest, [1.0,-0.0,3.0], [2,3,1])
        check(src, [1.0,-0.0,3.0], [2,3,1])
    end
end

@testset "one inactive source group keeps active-only replacement" begin
    for gs in ([1:1,2:2,3:3], [[3],[1],[2]], [1:2,3:4,5:6], [1:3:6,2:3:6,3:3:6])
        n = sum(length, gs)
        src = BlockIterate(Float64,n,gs)
        for k in (3,1), i in gs[k]; src[i] = i + 0.5; end
        for i in gs[2]; src[i] = -0.0; end          # zero writes: group 2 stays inactive
        @test collect(activecoordinates(src)) == [3,1]
        # group 2 active only in dest: cleared to +0.0, not given src's -0.0
        dest = BlockIterate(Float64,n,gs)
        for i in gs[2]; dest[i] = 2.0; end
        @test copyto!(dest,src) === dest
        expected = copy(src.storage); expected[collect(gs[2])] .= 0.0
        check(dest, expected, [3,1])
        # group 2 inactive in both: dest keeps its own -0.0 rather than src's +0.0
        for i in gs[2]; src[i] = 0.0; end
        dest = BlockIterate(Float64,n,gs)
        for i in gs[2]; dest[i] = -0.0; end
        @test copyto!(dest,src) === dest
        expected = copy(src.storage); expected[collect(gs[2])] .= -0.0
        check(dest, expected, [3,1])
        # with every source group active, every entry comes from src
        activate!(src,2)
        @test copyto!(dest,src) === dest
        check(dest, copy(src.storage), [3,1,2])
    end
end

@testset "randomized replacement against the model" begin
    rng = Xoshiro(872)
    randval(::Type{T}) where {T} = rand(rng) < 0.15 ? rand(rng, T[-0.0, 0.0, Inf, NaN]) : T(rand(rng, -8:8) / 4)
    function populate!(x, gs, frac)
        T = eltype(x)
        for k in shuffle(rng, 1:length(gs))[1:round(Int, frac * length(gs))]
            if rand(rng) < 0.2
                activate!(x, k); for i in gs[k]; x[i] = -zero(T); end
            else
                for i in gs[k]; x[i] = randval(T); end
                all(i -> iszero(x[i]), gs[k]) && activate!(x, k)
            end
        end
        for k in eachindex(gs)
            isactive(x, k) || rand(rng) < 0.5 || (x[first(gs[k])] = -zero(T))
        end
        x
    end
    for trial in 1:600
        gs = rand(rng, REPLACEMENT_LAYOUTS)
        T, S = rand(rng, ((Float64,Float64),(Float32,Float32),(Float32,Float64),(Float64,Float32)))
        src = populate!(BlockIterate(S,12,gs), gs, rand(rng, (0.0, 0.5, 1.0, 1.0, rand(rng))))
        dest = rand(rng, Bool) ? similar(src,T) : BlockIterate(T,12,gs)
        populate!(dest, gs, rand(rng, (0.0, 0.5, 1.0, rand(rng))))
        dvals, dids = copy(dest.storage), collect(activecoordinates(dest))
        svals, sids = copy(src.storage), collect(activecoordinates(src))
        copyto!(dest,src)
        vals, ids = model_replacement(dvals, dids, svals, sids, gs, T)
        @test isequal(dest.storage, vals)
        @test collect(activecoordinates(dest)) == ids
        @test all(k -> dest.position[k] == something(findfirst(==(k), ids), 0), eachindex(gs))
    end
end
end
