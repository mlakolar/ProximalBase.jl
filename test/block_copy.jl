module BlockCopyTest
using Test, ProximalBase, LinearAlgebra

struct FallibleValues{T} <: AbstractVector{T}
  values::Vector{T}
  fail::Int
end
Base.size(v::FallibleValues) = size(v.values)
Base.IndexStyle(::Type{<:FallibleValues}) = IndexLinear()
Base.getindex(v::FallibleValues, i::Int) = i == v.fail ? error("source read") : v.values[i]

# DenseArray alone is not sufficient evidence that a user-defined reader cannot fail.
struct FallibleDense{T} <: DenseArray{T, 1}
  values::Vector{T}
  fail::Int
end
Base.size(v::FallibleDense) = size(v.values)
Base.IndexStyle(::Type{<:FallibleDense}) = IndexLinear()
Base.getindex(v::FallibleDense, i::Int) = i == v.fail ? error("dense source read") : v.values[i]
const dense_copies = Ref(0)
function Base.copyto!(dest::AbstractArray, src::FallibleDense)
  dense_copies[] += 1
  invoke(copyto!, Tuple{AbstractArray, AbstractArray}, dest, src)
end

function invariants(x)
  ids = collect(activecoordinates(x))
  @test length(ids) == nactive(x) == length(unique(ids))
  for k in 1:ncoordinates(x)
    @test isactive(x,k) == (k in ids)
    @test isactive(x,k) || all(iszero,blockview(x,k))
  end
end
function same_state(x,y)
  @test isequal(Vector(x),Vector(y))
  @test collect(activecoordinates(x)) == collect(activecoordinates(y))
  invariants(x)
end
function scalar_copy!(x,k,values)
  for (i,v) in zip(blockindices(x,k),values)
    x[i] = v
  end
  x
end
function initial(::Type{T},groups,k,active) where T
  x = BlockIterate(T,maximum(maximum,groups),groups)
  for other in reverse(1:ncoordinates(x))
    other == k && continue
    x[first(blockindices(x,other))] = T(7)
  end
  active && activate!(x,k)
  x
end
caught(op) = try op(); nothing catch err; typeof(err) end

@testset "whole-block copies match scalar value writes" begin
  layouts = ([1:3,4:6,7:9], [[1,4,7],[2,5,8],[3,6,9]],
    [[7,1,4],[8,2,5],[9,3,6]], [1:1,2:4,5:9], [1:8,9:9],
    [Int32(1):Int32(3),Int32(4):Int32(6),Int32(7):Int32(9)],
    [1:16,17:32,33:48], [1:32,33:64,65:96],
    [collect(1:3:96),collect(2:3:96),collect(3:3:96)],
    [collect(32:-1:1),collect(64:-1:33),collect(96:-1:65)])
  for T in (Float32,Float64), gs in layouts, k in eachindex(gs), active in (false,true)
    n = length(gs[k])
    for data in (zeros(T,n),fill(-zero(T),n),ones(T,n),fill(T(NaN),n),fill(T(Inf),n),
                 T[isodd(i) ? -Inf : 0 for i in 1:n], T[isodd(i) ? 0 : 2 for i in 1:n])
      for source in (data,view(vcat(data,data),1:n),view(repeat(data;inner=2),1:2:2n))
        x = initial(T,gs,k,active); ref = copy(x)
        h = blockindices(x,k); before_indices = collect(h)
        scalar_copy!(ref,k,source)
        dest = blockview(x,k)
        @test copyto!(dest,source) === dest
        same_state(x,ref)
        @test collect(h) == before_indices
        @test parent(dest) === x
        @test Base.mightalias(dest,x)
        @test !Base.mightalias(dest,source)
      end
    end
    # Checked conversions keep the same converted-value activation rule.
    for source in (fill(BigFloat("1e-1000"),n), Any[fill(-zero(T),n)...],
                   fill(T===Float32 ? 1e-100 : 0f0,n))
      x = initial(T,gs,k,active); ref = copy(x)
      scalar_copy!(ref,k,source)
      copyto!(blockview(x,k),source)
      same_state(x,ref)
    end
  end
end

@testset "aliasing, prefixes and arbitrary views" begin
  for T in (Float32,Float64), gs in ([1:3,4:6,7:9],[[3,1,2],[6,4,5],[9,7,8]])
    for source in (x->blockview(x,1), x->view(x,2:4), x->view(x,[3,2,1]),
                   x->blockview(x,2), x->view(x.storage,2:4))
      x = BlockIterate(T.(1:9),gs); ref = copy(x)
      values = source(x)
      scalar_copy!(ref,1,collect(values))   # Base copy semantics snapshot overlap
      dest = blockview(x,1)
      @test copyto!(dest,values) === dest
      same_state(x,ref)
    end
    for ids in (Int[],[1,3],[1,4,8],[1,4,1,8],collect(gs[1]),1:2:9),
        zero_prefix in (false,true)
      x = initial(T,gs,1,false);ref=copy(x)
      values = T[zero_prefix && i <= 2 ? -zero(T) : i for i in 1:length(ids)]
      for (i,v) in zip(ids,values);ref[i]=v;end
      dest = view(x,ids)
      @test copyto!(dest,values) === dest
      same_state(x,ref)
    end
    # The zero prefix overwrites a location that a later source read needs.
    # Base snapshots the source before any scalar destination writes.
    for ids in ([3,4,1,8],[3,4,1,1])
      x=BlockIterate(T[0,-0.0,0,7,0,0,0,0,9],gs);ref=copy(x)
      source=view(x.storage,2:5);values=collect(source)
      for (i,v) in zip(ids,values);ref[i]=v;end
      dest=view(x,ids)
      @test copyto!(dest,source) === dest
      same_state(x,ref)
    end
    x=initial(T,gs,1,false);ref=copy(x)
    copyto!(blockview(x,1),T[2,3])
    scalar_copy!(ref,1,T[2,3])
    same_state(x,ref)
    before=copy(x)
    @test_throws BoundsError copyto!(blockview(x,1),ones(T,4))
    same_state(x,before)
    @test copyto!(blockview(x,1),T[]) === blockview(x,1)
    same_state(x,before)
    for k in (0,4)
      @test_throws BoundsError blockview(x,k)
      same_state(x,before)
    end
    h=blockindices(x,1)
    if !(h isa AbstractRange)
      @test_throws ArgumentError copyto!(h,[1,2,3])
      same_state(x,before)
    end
  end
end

@testset "fallible reads and conversions retain scalar failure state" begin
  for T in (Float32,Float64), gs in ([1:3,4:6,7:9],[[3,1,2],[6,4,5],[9,7,8]]),
      active in (false,true), firstvalue in (zero(T),T(2))
    for source in (FallibleValues(T[firstvalue,3,4],2),FallibleDense(T[firstvalue,3,4],2),
                   Any[firstvalue,1+im,T(4)])
      x=initial(T,gs,1,active);ref=copy(x)
      expected=caught(()->scalar_copy!(ref,1,source))
      @test expected !== nothing
      previous_copies=dense_copies[]
      @test caught(()->copyto!(blockview(x,1),source)) == expected
      source isa FallibleDense && @test dense_copies[] == previous_copies+1
      same_state(x,ref)
    end
  end
end

@testset "large active copy snapshots an aliased ordinary vector" begin
  for T in (Float32,Float64), ids in (collect(1:32),collect(32:-1:1))
    x=BlockIterate(T.(1:32),[ids]);ref=copy(x)
    scalar_copy!(ref,1,copy(x.storage))
    dest=blockview(x,1)
    @test copyto!(dest,x.storage) === dest
    same_state(x,ref)
  end
end

@testset "view math and the same-type active writer allocate no storage" begin
  for T in (Float32,Float64), n in (4,32,256), gs in ([1:n,n+1:2n],[collect(1:2:2n),collect(2:2:2n)])
    x=BlockIterate(ones(T,2n),gs);v=blockview(x,1);u=ones(T,n)
    copyto!(v,u)
    write!(v,u) = (copyto!(v,u);nothing)
    write!(v,u)
    @test @allocated(write!(v,u)) == 0
    @test norm(v) === norm(u)
    @test norm(v,1) === norm(u,1)
    @test dot(v,u) === dot(u,u)
    @test dot(u,v) === dot(u,u)
    @test copy(v) == u
  end
end
end
