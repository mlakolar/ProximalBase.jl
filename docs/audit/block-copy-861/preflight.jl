# Single-process component diagnostic; run separately for master/final sources.
using BenchmarkTools, Statistics
include(joinpath(ARGS[1],"src","ProximalBase.jl"))
const PB=ProximalBase
function buffers(::Type{T},src) where T
    [T[convert(T,v) for v in PB._block_storage(src,src.active[a])] for a in 1:src.nactive]
end
function checked_failure!(dest,src)
    try
        copyto!(dest,src)
        error("expected conversion failure")
    catch err
        err isa InexactError || rethrow()
    end
    nothing
end
function main()
    println("block_size,layout,operation,median_ns,bytes,allocs")
    for b in (1,8,64), layout in ("contiguous","strided","arbitrary")
        n=8192;K=n÷b;s=K÷10
        groups=layout=="contiguous" ? [(k-1)*b+1:k*b for k in 1:K] :
               layout=="strided" ? [k:K:n for k in 1:K] : [collect(k:K:n) for k in 1:K]
        src=PB.BlockIterate(Float64,n,groups)
        for k in 1:s,i in groups[k];src[i]=2.0;end
        dest=similar(src,Int)
        failing=copy(src);failing[last(groups[s])]=0.5
        prep=()->copyto!(dest,src)
        for (name,f) in (("buffers",()->buffers(Int,src)),
                         ("copyto_int",()->copyto!(dest,src)),
                         ("copyto_error",()->checked_failure!(dest,failing)))
            prep();f()
            t=@benchmark $f() setup=($prep()) evals=1 samples=200 seconds=.04
            e=median(t)
            println(join((b,layout,name,e.time,e.memory,e.allocs),','))
        end
        # The failing source must leave complete destination state unchanged.
        prep();before=(copy(dest.storage),copy(dest.active),copy(dest.position),dest.nactive,dest.layout_cache)
        checked_failure!(dest,failing)
        @assert (dest.storage,dest.active,dest.position,dest.nactive)==before[1:4]
        @assert dest.layout_cache===before[5]
    end
end
main()
