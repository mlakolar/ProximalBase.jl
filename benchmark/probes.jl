using LinearAlgebra, Random, SparseArrays
struct Probe{F,S,C}
    id::String
    op::F
    setup::S
    check::C
end
noop() = nothing
function reads(c)
    s = zero(eltype(c))
    @inbounds for rep in 1:8, i in eachindex(c)
        s += c[i]
    end
    s
end
function writes(c, ids, repetitions=8)
    @inbounds for rep in 1:repetitions, i in ids
        c[i] = eltype(c)(1 + (i % 7)/10)
    end
    nothing
end
function sweep(c)
    @inbounds for i in eachindex(c)
        c[i] = c[i] * eltype(c)(.9) + eltype(c)(.01)
    end
    nothing
end
function pairreads(x,p)
    s=zero(eltype(x))
    @inbounds for rep in 1:4,j in 1:p,i in 1:p
        s+=x[i,j]
    end
    s
end
function pairwrites(x,p,repetitions)
    @inbounds for rep in 1:repetitions,j in 1:p
        x[j,j]=eltype(x)(1+j/100)
    end
    nothing
end
function makeprobes(::Type{T}) where T
    ps = Any[]
    add(id,op,setup,check) = push!(ps, Probe("$T/$id",op,setup,check))
    rng = Xoshiro(849)
    p=1024
    groups = [collect(k:32:p) for k in 1:32]
    for family in (:sparse,:symmetric,:block)
        x = family === :sparse ? PB.SparseIterate(T,p) : family === :symmetric ? PB.SymmetricSparseIterate(T,45) : PB.BlockIterate(T,p,groups)
        c = family === :block ? x : PB.coordinates(x)
        q=length(c)
        ids=collect(1:17:q)
        @inbounds for i in ids
            c[i]=T(sin(i)+1)
        end
        src=copy(x)
        dense = family === :symmetric ? Matrix(x) : Vector(x)
        cdense = collect(c)
        reset=()->(copyto!(x,src);nothing)
        check=()->begin
            actual=family === :symmetric ? Matrix(x) : Vector(x)
            all(isfinite,actual) || error("nonfinite fixture")
            (actual,collect(PB.activecoordinates(x)))
        end
        add("$family/read",()->reads(c),reset,check)
        add("$family/active_write",()->writes(c,ids),reset,check)
        clear=()->begin
            fill!(c,zero(T));dropzeros!(x);nothing
        end
        add("$family/first_activation",()->writes(c,ids,1),clear,check)
        add("$family/full_sweep",()->sweep(c),reset,check)
        dest=similar(x)
        add("$family/copyto_iterate",()->(copyto!(dest,x);nothing),reset,()->family===:symmetric ? Matrix(dest) : Vector(dest))
        add("$family/copyto_dense",()->(copyto!(dest,dense);nothing),reset,()->family===:symmetric ? Matrix(dest) : Vector(dest))
        add("$family/conversion",()->family===:symmetric ? Matrix(x) : Vector(x),reset,check)
        if family===:symmetric
            b=randn(rng,T,45);symout=zeros(T,45)
            add("$family/product",()->(mul!(symout,x,b);nothing),reset,()->copy(symout))
            add("$family/product_allocating",()->x*b,reset,check)
        else
            A=randn(rng,T,128,p);prodout=zeros(T,128)
            add("$family/product",()->(mul!(prodout,A,x);nothing),reset,()->copy(prodout))
            add("$family/product_allocating",()->A*x,reset,check)
            y=randn(rng,T,128); lsf=PB.LeastSquaresLoss(y,A);g=zeros(T,p)
            add("$family/leastsquares_value",()->lsf(x),reset,()->lsf(x))
            add("$family/leastsquares_gradient",()->(PB.gradient!(g,lsf,x);nothing),reset,()->copy(g))
        end
        penalty=PB.NormL1(T(.3));l1out=similar(x)
        add("$family/l1_value",()->penalty(x),reset,()->penalty(x))
        add("$family/l1_prox",()->(PB.prox!(l1out,penalty,x,T(.7));nothing),reset,()->family===:symmetric ? Matrix(l1out) : Vector(l1out))
    end
    pairx=PB.SymmetricSparseIterate(T,45)
    pairwrites(pairx,45,1);pairsource=copy(pairx)
    pairreset=()->(copyto!(pairx,pairsource);nothing)
    pairclear=()->(fill!(PB.coordinates(pairx),zero(T));dropzeros!(pairx);nothing)
    paircheck=()->(Matrix(pairx),collect(PB.activecoordinates(pairx)))
    add("symmetric/pair_read",()->pairreads(pairx,45),pairreset,paircheck)
    add("symmetric/pair_active_write",()->pairwrites(pairx,45,8),pairreset,paircheck)
    add("symmetric/pair_first_activation",()->pairwrites(pairx,45,1),pairclear,paircheck)
    for layout in (:range,:arbitrary,:reversed)
        gs=layout===:range ? [1:32,33:64] : layout===:arbitrary ? [collect(1:2:64),collect(2:2:64)] : [collect(32:-1:1),collect(64:-1:33)]
        x=PB.BlockIterate(T,64,gs);u=T[sin(i)+1 for i in 1:32];v=PB.blockview(x,1)
        clear=()->(fill!(x,zero(T));dropzeros!(x);nothing)
        check=()->(Vector(x),collect(PB.activecoordinates(x)))
        add("blockview/$layout/read",()->reads(v),()->copyto!(v,u),check)
        add("blockview/$layout/write_active",()->(copyto!(PB.blockview(x,1),u);nothing),()->copyto!(v,u),check)
        add("blockview/$layout/write_inactive",()->(copyto!(PB.blockview(x,1),u);nothing),clear,check)
    end
    f=PB.NormL1(T(.3));scalar_u=randn(rng,T,1024);out=similar(scalar_u)
    add("l1/scalar_prox",()->begin
        s=zero(T)
        @inbounds for i in eachindex(scalar_u)
            s+=PB.prox_block(f,scalar_u[i],i,T(.7))
        end
        s
    end,noop,()->PB.prox_block(f,T(2),1,T(.7)))
    add("l1/vector_prox",()->(PB.prox!(out,f,scalar_u,T(.7));nothing),noop,()->copy(out))
    # ElasticNet on ordinary inputs with finite levels: the stored entries of the three iterates, and dense
    # weighted prox_block, prox! and value (the weights do not draw from rng, so the data above is unchanged)
    for family in (:sparse,:symmetric,:block)
        x = family === :sparse ? PB.SparseIterate(T,p) : family === :symmetric ? PB.SymmetricSparseIterate(T,45) : PB.BlockIterate(T,p,groups)
        c = family === :block ? x : PB.coordinates(x)
        @inbounds for i in 1:13:length(c)
            c[i]=T(sin(i)*2)
        end
        src=copy(x);reset=()->(copyto!(x,src);nothing)
        en=PB.ElasticNet(T(.3),T(.5));enout=similar(x)
        add("$family/elasticnet_value",()->en(x),reset,()->en(x))
        add("$family/elasticnet_prox",()->(PB.prox!(enout,en,x,T(.7));nothing),reset,()->family===:symmetric ? Matrix(enout) : Vector(enout))
    end
    en=PB.ElasticNet(T(.3),T(.5),T(.5).+T.(mod.(1:1024,7))./4)
    add("elasticnet/scalar_prox",()->begin
        s=zero(T)
        @inbounds for i in eachindex(scalar_u)
            s+=PB.prox_block(en,scalar_u[i],i,T(.7))
        end
        s
    end,noop,()->[PB.prox_block(en,scalar_u[i],i,T(.7)) for i in eachindex(scalar_u)])
    add("elasticnet/vector_prox",()->(PB.prox!(out,en,scalar_u,T(.7));nothing),noop,()->copy(out))
    add("elasticnet/vector_value",()->en(scalar_u),noop,()->en(scalar_u))
    ps
end
