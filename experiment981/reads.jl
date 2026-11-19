include("timing.jl")

struct ReadCount{T} <: AbstractVector{T}
    data::Vector{T}
    reads::Base.RefValue{Int}
end
ReadCount(data::Vector) = ReadCount(data,Ref(0))
Base.size(a::ReadCount) = size(a.data)
Base.IndexStyle(::Type{<:ReadCount}) = IndexLinear()
@inline function Base.getindex(a::ReadCount, i::Int)
    a.reads[] += 1
    return a.data[i]
end

function input_reads(label,pb,out,sizes)
    open(out,"w") do io
        tsv(io,"label","julia","type","case","n","v_reads","x_reads","v_reads_per_n","x_reads_per_n","instrumented","ordinary","same_value")
        timing_cases(sizes) do id,T,n,v,x,level,factors
            counted_v, counted_x = ReadCount(v), ReadCount(x)
            f = penalty(pb,level,factors)
            ordinary = pb.kkt_violation(f,v,x)
            r = pb.kkt_violation(f,counted_v,counted_x)
            nv, nx = counted_v.reads[], counted_x.reads[]
            tsv(io,label,VERSION,T,id,n,nv,nx,nv/n,nx/n,r,ordinary,isequal(r,ordinary))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: reads.jl LABEL CHECKOUT OUT.tsv")
    sizes = parse.(Int,split(get(ENV,"N981_SIZES","2,8,32,256,4096,100000,1000000"),','))
    pb = load_checkout(ARGS[2],:CountCheckout)
    Base.invokelatest(input_reads, ARGS[1],pb,ARGS[3],sizes)
end
