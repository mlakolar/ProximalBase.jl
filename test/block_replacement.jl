module BlockReplacementTest
using Test, ProximalBase, SparseArrays

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

@testset "whole-owner replacement and independent state references" begin
    layouts = ([1:3,4:6,7:9,10:12], [1:4:12,2:4:12,3:4:12,4:4:12],
               [[9,1,5],[10,2,6],[11,3,7],[12,4,8]], [k:12:12 for k in 1:12])
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
    for T in (Float32,Float64), S in (Float32,Float64), op in (copy!,copyto!)
        src = BlockIterate(S,12,[1:3,4:6,7:9,10:12])
        src[8] = S(3); activate!(src,1)
        dest = BlockIterate(T,12,[[1,5,9],[2,6,10],[3,7,11],[4,8,12]])
        dest[3] = T(7) # retained zero activity after generic fallback
        ref = copy(dest)
        for i in 1:12; ref[i] = src[i]; end
        @test op(dest,src) === dest
        check(dest,Vector(ref),collect(activecoordinates(ref)))
        dropzeros!(dest)
        check(dest,T.(Vector(src)),[4])
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
end
