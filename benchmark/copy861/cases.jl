# Reuse the #852 descriptors, independent-reference helpers and case container.
include(joinpath(@__DIR__, "..", "audit852", "cases.jl"))

function owner(::Type{T}, n, groups) where T
    @static if ORIGINAL
        storage=zeros(T,n)
        PB.AtomIterate(storage,[view(storage,G) for G in groups])
    else
        PB.BlockIterate(T,n,groups)
    end
end
function sharedowner(src, ::Type{T}, n, groups) where T
    @static if ORIGINAL
        owner(T,n,groups) # original has prebuilt views, no owner layout cache
    else
        similar(src,T)
    end
end
function restore!(dest, template)
    copyto!(dest.storage,template.storage)
    @static if !ORIGINAL
        copyto!(dest.active,template.active)
        copyto!(dest.position,template.position)
        dest.nactive=template.nactive
    end
    nothing
end
function validate_state(x,ref,ordered)
    @assert isequal(collect(x.storage),ref)
    @static if !ORIGINAL
        @assert collect(PB.activecoordinates(x))==ordered
        for k in eachindex(x.position)
            a=findfirst(==(k),ordered)
            @assert x.position[k]==(a===nothing ? 0 : a)
        end
    end
end
function restart!(x,src,checkpoint,groups,ks)
    for _ in 1:4
        copyto!(x,src)
        for k in ks
            i=first(groups[k]); x[i]+=eltype(x)(1/8)
        end
        copyto!(checkpoint,x)
        copyto!(x,src)
        copyto!(x,checkpoint)
    end
    x
end
@static if ORIGINAL
    # The same raw native-vector constructor used by #852's raw-block control.
    newblock(::Type{T},n,b,groups) where T=owner(T,n,groups)
end
function replacement_cases(T,n,s,b,layout,ops,seed)
    get(ENV,"PB_EXACT852","0")=="1" && return block_cases(T,n,s,b,layout,ops,seed)
    rng=Xoshiro(seed);groups=group_layout(n,b,layout,rng);K=length(groups)
    ks=randperm(rng,K)[1:s]
    src=owner(T,n,groups)
    base=zeros(T,n)
    for (a,k) in enumerate(ks), (j,i) in enumerate(groups[k])
        value=T((mod(a+j,15)+1)/16)
        src[i]=value; base[i]=value
    end
    cases=AuditCase[]
    for name in ops
        mixed=startswith(name,"mixed")
        D=mixed ? (T==Float32 ? Float64 : Float32) : T
        shared=occursin("shared",name)
        dest=shared ? sharedowner(src,D,n,groups) : owner(D,n,groups)
        @static if !ORIGINAL
            occursin("cached",name) && @assert PB.samegroups(dest,src)
        end
        different=name=="different"
        if different
            dg=reverse(groups)
            dest=owner(D,n,dg)
        end
        active=!occursin("empty",name)
        if active
            for k in reverse(ks), i in groups[k];dest[i]=D(3/4);end
            # Include groups that must be cleared, distinct from source activity.
            for k in setdiff(1:K,ks)[1:min(s,K-s)]
                dest[first(groups[k])]=D(1/4)
            end
        end
        source=src;ref=D.(base);expected=copy(ks)
        if occursin("zeros",name)
            source=copy(src)
            for k in ks[1:2:end],i in groups[k];source[i]=zero(T);ref[i]=zero(D);end
        end
        if different
            # Independent simulation of scalar value-write activation order.
            destids=Int[]
            @static if !ORIGINAL
                append!(destids,PB.activecoordinates(dest))
            end
            for i in 1:n
                k=K+1-findfirst(g->i in g,groups)
                !iszero(base[i]) && !(k in destids) && push!(destids,k)
            end
            expected=destids
        end
        template=copy(dest)
        prep=let dest=dest, template=template; ()->restore!(dest,template); end
        f=let dest=dest, source=source; ()->copyto!(dest,source); end
        valid=let dest=dest, ref=ref, expected=expected; r->validate_state(dest,ref,expected); end
        if name=="copy"
            f=()->copy(src);prep=noop
            valid=r->begin
                validate_state(r,base,ks)
                r[first(groups[ks[1]])]=T(9)
                @assert isequal(src.storage,base)
            end
        elseif name=="self"
            f=()->copyto!(src,src);prep=noop
            valid=r->validate_state(src,base,ks)
        elseif name=="copybang_shared"
            f=let dest=dest, source=source; ()->copy!(dest,source); end
        elseif startswith(name,"restart")
            checkpoint=shared ? sharedowner(src,D,n,groups) : owner(D,n,groups)
            f=let dest=dest, checkpoint=checkpoint; ()->restart!(dest,src,checkpoint,groups,ks); end
            result=copy(ref)
            for k in ks;result[first(groups[k])]+=D(1/8);end
            valid=let dest=dest, result=result; r->validate_state(dest,result,ks); end
        end
        push!(cases,AuditCase(name,f,prep,valid,s*b))
    end
    cases,src
end
