# Additional state-transition checks; run separately from timing.
using LinearAlgebra, SparseArrays, Random
include(joinpath(ARGS[1],"src","ProximalBase.jl"));const PB=Main.ProximalBase
include("adapters.jl");include("cases.jl")
for T in (Float32,Float64),fam in ("sparse","symmetric")
    n=fam=="sparse" ? 10000 : 144
    rows,cols=fam=="sparse" ? (Int[],Int[]) : triangle(n)
    q=fam=="sparse" ? n : length(rows)
    x=newscalar(fam,T,n);idx=[1,23,75,901];vals=T[1,2,3,4]
    ref=zeros(T,q);ref[idx]=vals;active=Set(idx)
    populate_scalar!(x,idx,vals,rows,cols)
    copyto!(x,x);verify_scalar(x,ref,active,rows,cols)
    coordset!(x,zero(T),idx[1],rows,cols);zero_ref=copy(ref);zero_ref[idx[1]]=zero(T)
    z=copy(x);verify_scalar(z,zero_ref,active,rows,cols)
    y=newscalar(fam,T,n);coordset!(y,T(9),99,rows,cols);copyto!(y,x)
    verify_scalar(y,zero_ref,active,rows,cols)
    coordset!(z,T(8),idx[1],rows,cols);@assert coord(x,idx[1],rows,cols)==zero(T)
    populate_scalar!(x,idx,vals,rows,cols)
    for cycle in 1:4
        populate_scalar!(x,idx,zeros(T,length(idx)),rows,cols)
        verify_scalar(x,zeros(T,q),active,rows,cols)
        compact!(x);verify_scalar(x,zeros(T,q),Set{Int}(),rows,cols)
        populate_scalar!(x,idx,vals,rows,cols);verify_scalar(x,ref,active,rows,cols)
    end
    println(T,'/',fam," self-copy and four zero/compact/reconstruct transitions pass")
end
for T in (Float32,Float64),layout in ("contiguous","strided","arbitrary")
    ORIGINAL && (T==Float32 || layout!="contiguous") && continue
    n=8192;b=8;groups=group_layout(n,b,layout,Xoshiro(852))
    x=newblock(T,n,b,groups);y=newblock(T,n,b,groups)
    ks=[1,3,17];ref=zeros(T,n)
    for k in ks;fill!(block(x,k),T(1/2));ref[groups[k]].=T(1/2);end
    copyto!(x,x);verify_block(x,ref,Set(ks),groups)
    fill!(block(x,1),zero(T));zero_ref=copy(ref);zero_ref[groups[1]].=zero(T)
    z=copy(x);verify_block(z,zero_ref,Set(ks),groups)
    copyto!(y,x);verify_block(y,zero_ref,Set(ks),groups)
    fill!(block(z,1),T(8));@assert x[groups[1][1]]==zero(T)
    fill!(block(x,1),T(1/2))
    # Destination has a disjoint nonzero group and an explicit active zero.
    fill!(block(y,29),T(3));fill!(block(y,31),T(4));fill!(block(y,31),zero(T))
    copyto!(y,x);verify_block(y,ref,Set(ks),groups)
    fill!(block(y,1),T(9));@assert x[groups[1][1]]==T(1/2)
    println(T,'/',layout," block self-copy, disjoint destination replacement and non-aliasing pass")
end
