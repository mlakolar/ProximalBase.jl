module BlockValueUpdatesTest
using Test, ProximalBase, Random

function check_state(x, dense, active, groups)
    @test isequal(Vector(x),dense)
    @test collect(activecoordinates(x)) == active
    @test nactive(x) == length(active)
    for k in eachindex(groups)
        slot=findfirst(==(k),active)
        @test x.position[k] == (slot === nothing ? 0 : slot)
        @test isactive(x,k) == (slot !== nothing)
        @test slot !== nothing || all(iszero,dense[groups[k]])
    end
end

# A plain Vector supplies Base's conversion, prefix-copy and aliasing behavior.
# Expected activity is derived from the independent partition and dense values.
function checked_copy!(x,dense,active,groups,k,source)
    v=blockview(x,k)
    expected_error=try
        copyto!(view(dense,groups[k]),source)
        nothing
    catch e
        typeof(e)
    end
    actual_error=try
        result=copyto!(v,source)
        @test result === v
        nothing
    catch e
        e isa BoundsError && (@test e.a === v)
        typeof(e)
    end
    @test actual_error == expected_error
    if !(k in active) && any(!iszero,dense[groups[k]])
        push!(active,k)
    end
    check_state(x,dense,active,groups)
end

@testset "activation follows the converted value" begin
    x=BlockIterate(Float32,4,2)
    copyto!(blockview(x,1),[1e-100,0.0])
    @test Vector(x)==zeros(Float32,4)
    @test nactive(x)==0
    @test x.position==[0,0]
end

struct OffsetIndices <: AbstractVector{Int}
    ids::Vector{Int}
end
Base.size(h::OffsetIndices)=size(h.ids)
Base.axes(h::OffsetIndices)=(0:length(h.ids)-1,)
Base.IndexStyle(::Type{OffsetIndices})=IndexLinear()
Base.getindex(h::OffsetIndices,i::Int)=h.ids[i+1]

@testset "scalar buffer starts at the view's first index" begin
    for T in (Float32,Float64)
        x=BlockIterate(T,6,2);dense=zeros(T,6)
        inds=OffsetIndices([1,4]);vx=view(x,inds);vd=view(dense,inds)
        @test axes(vx)==(0:1,)
        copyto!(vd,T[7])
        @test copyto!(vx,T[7]) === vx
        check_state(x,dense,[1],[1:3,4:6])
    end
end

struct AliasCheckedIndices <: AbstractVector{Int}
    ids::Vector{Int}
    armed::Base.RefValue{Bool}
end
Base.size(h::AliasCheckedIndices)=size(h.ids)
Base.IndexStyle(::Type{AliasCheckedIndices})=IndexLinear()
Base.getindex(h::AliasCheckedIndices,i::Int)=h.ids[i]
function Base.dataids(h::AliasCheckedIndices)
    h.armed[] && error("alias metadata failure")
    Base.dataids(h.ids)
end

@testset "singleton copies preserve custom alias-metadata errors" begin
    for T in (Float32,Float64)
        x=BlockIterate(T,6,2);dense=zeros(T,6)
        ix=AliasCheckedIndices([1,4],Ref(false))
        idense=AliasCheckedIndices([1,4],Ref(false))
        vx=view(x,ix);vd=view(dense,idense)
        ix.armed[]=idense.armed[]=true
        @test_throws ErrorException copyto!(vd,T[2])
        @test_throws ErrorException copyto!(vx,T[2])
        check_state(x,dense,Int[],[1:3,4:6])
    end
end

struct CountedIndices <: AbstractVector{Int}
    ids::Vector{Int}
    reads::Base.RefValue{Int}
    throw_at::Base.RefValue{Int}
end
Base.size(h::CountedIndices)=size(h.ids)
Base.IndexStyle(::Type{CountedIndices})=IndexLinear()
function Base.getindex(h::CountedIndices,i::Int)
    h.reads[]+=1
    h.reads[]==h.throw_at[] && error("index read failure")
    h.ids[i]
end

@testset "custom view indices keep Base's read and partial-write behavior" begin
    for T in (Float32,Float64),limit in (2,3)
        x=BlockIterate(T,6,2);dense=zeros(T,6)
        ix=CountedIndices([1,4],Ref(0),Ref(0))
        idense=CountedIndices([1,4],Ref(0),Ref(0))
        vx=view(x,ix);vd=view(dense,idense)
        ix.reads[]=idense.reads[]=0
        ix.throw_at[]=idense.throw_at[]=limit
        actual=try copyto!(vx,T[2,3]);nothing catch e;sprint(showerror,e) end
        expected=try copyto!(vd,T[2,3]);nothing catch e;sprint(showerror,e) end
        @test actual==expected
        @test ix.reads[]==idense.reads[]
        check_state(x,dense,limit==2 ? [1] : [1,2],[1:3,4:6])
    end
end

for T in (Float32,Float64), b in (1,8,64), layout in (:contiguous,:strided,:arbitrary,:reversed)
    rng=Xoshiro(860);n=4b
    groups=layout==:contiguous ? [(k-1)*b+1:k*b for k in 1:4] :
           layout==:strided ? [k:4:n for k in 1:4] :
           layout==:reversed ? [k*b:-1:(k-1)*b+1 for k in 1:4] :
           [p[(k-1)*b+1:k*b] for p in (randperm(rng,n),) for k in 1:4]
    @testset "$T b=$b $layout block value copies" begin
        x=BlockIterate(T,n,groups);dense=zeros(T,n);active=Int[]
        for k in (3,1);activate!(x,k);push!(active,k);end
        for (k,src) in ((2,zeros(T,b)),(2,fill(T(2),b)),(2,zeros(T,b)),
                        (4,T[]),(1,T[-0.0]),(4,randn(rng,T,b)),
                        (1,fill(T(NaN),b)),(3,fill(T(Inf),b)),
                        (2,zeros(T,b+1)),(4,Any[1,2im]),
                        (1,ComplexF64[2+0im,3im]),(3,Float64[1e-100]))
            checked_copy!(x,dense,active,groups,k,src)
        end
        v=blockview(x,4)
        checked_copy!(x,dense,active,groups,4,zeros(T,b))
        dropzeros!(x)
        # Independently reproduce documented swap-removal order.
        i=1
        while i<=length(active)
            if all(iszero,dense[groups[active[i]]])
                active[i]=active[end];pop!(active)
            else
                i+=1
            end
        end
        check_state(x,dense,active,groups)
        copyto!(v,ones(T,b));dense[groups[4]].=one(T)
        4 in active || push!(active,4)
        check_state(x,dense,active,groups)
        # General views spanning groups must continue to activate each group.
        inds=[groups[2][1],groups[4][1]]
        copyto!(view(x,inds),T[7,8]);dense[inds]=T[7,8]
        2 in active || push!(active,2)
        check_state(x,dense,active,groups)
        @test_throws BoundsError blockview(x,0)
        @test_throws BoundsError activate!(x,5)
    end
end

@testset "live block view after whole-iterate state replacement" begin
    layouts=([1:2,3:4,5:6,7:8], [2:-1:1,4:-1:3,6:-1:5,8:-1:7],
             [k:4:8 for k in 1:4], [[3,1],[7,2],[8,6],[4,5]])
    for T in (Float32,Float64),groups in layouts
        x=BlockIterate(T,8,groups);v=blockview(x,1)
        x[groups[1][1]]=T(9);x[groups[4][1]]=T(7)
        y=BlockIterate(T,8,groups)
        y[groups[3][1]]=T(2);activate!(y,2)
        copyto!(x,y)
        dense=zeros(T,8);dense[groups[3][1]]=T(2)
        copyto!(v,T[4,0]);dense[groups[1]]=T[4,0]
        check_state(x,dense,[3,2,1],groups)
        w=BlockIterate(T,8,groups)
        activate!(w,1);w[groups[4][1]]=T(5)
        copyto!(x,w)
        dense=zeros(T,8);dense[groups[4][1]]=T(5)
        copyto!(v,zeros(T,2))
        check_state(x,dense,[1,4],groups)
        copyto!(v,ones(T,2));dense[groups[1]].=one(T)
        check_state(x,dense,[1,4],groups)
    end
end

@testset "aliased Vector source and ordered block layouts" begin
    for T in (Float32,Float64),G in (1:8,8:-1:1,[3,7,1,8,4,6,2,5])
        dense=T.(1:8);x=BlockIterate(dense,[G]);v=blockview(x,1)
        copyto!(view(dense,G),dense)
        @test copyto!(v,x.storage) === v
        check_state(x,dense,[1],[G])
        # An overlapping SubArray source takes Base's ordinary aliasing path.
        source=view(x,reverse(collect(G)))
        expected=copy(Vector(source))
        copyto!(v,source)
        dense[G]=expected
        check_state(x,dense,[1],[G])
    end
end
end
