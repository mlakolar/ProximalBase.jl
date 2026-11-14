# #872 whole-iterate replacement fixtures. Reuses #861's owner/restore/validate/restart
# helpers (which reuse #852's descriptors and case container) and adds activity
# fractions, destination-only activity and mixed-precision replacement.
include(joinpath(@__DIR__, "..", "copy861", "cases.jl"))

function cases872(T,n,s,b,layout,ops,seed)
    rng=Xoshiro(seed);groups=group_layout(n,b,layout,rng);K=length(groups)
    ks=randperm(rng,K)[1:s]
    src=owner(T,n,groups)
    base=zeros(T,n)
    for (a,k) in enumerate(ks), (j,i) in enumerate(groups[k])
        value=T((mod(a+j,15)+1)/16)
        src[i]=value; base[i]=value
    end
    others=setdiff(1:K,ks)
    destonly=others[1:min(s,K-s)]
    cases=AuditCase[]
    for name in ops
        mixed=startswith(name,"mixed")
        D=mixed ? (T==Float32 ? Float64 : Float32) : T
        shared=occursin("shared",name) || mixed
        dest=shared ? sharedowner(src,D,n,groups) : owner(D,n,groups)
        @static if !ORIGINAL
            startswith(name,"cached") && @assert PB.samegroups(dest,src)
        end
        if occursin("active",name) || startswith(name,"restart") || mixed
            for k in reverse(ks), i in groups[k];dest[i]=D(3/4);end
            for k in destonly;dest[first(groups[k])]=D(1/4);end
        elseif occursin("destonly",name)
            for k in destonly;dest[first(groups[k])]=D(1/4);end
        end
        ref=D.(base);expected=copy(ks)
        template=copy(dest)
        prep=let dest=dest, template=template; ()->restore!(dest,template); end
        f=let dest=dest, src=src; ()->copyto!(dest,src); end
        valid=let dest=dest, ref=ref, expected=expected; r->validate_state(dest,ref,expected); end
        if startswith(name,"restart")
            checkpoint=shared ? sharedowner(src,D,n,groups) : owner(D,n,groups)
            f=let dest=dest, checkpoint=checkpoint, src=src, groups=groups, ks=ks
                ()->restart!(dest,src,checkpoint,groups,ks)
            end
            result=copy(ref)
            for k in ks;result[first(groups[k])]+=D(1/8);end
            valid=let dest=dest, result=result, ks=ks; r->validate_state(dest,result,ks); end
        end
        push!(cases,AuditCase(name,f,prep,valid,s*b))
    end
    cases,src
end
