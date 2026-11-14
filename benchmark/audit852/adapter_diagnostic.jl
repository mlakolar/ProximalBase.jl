# Test inference effects of API-selection adapters, outside headline timings.
using LinearAlgebra,SparseArrays,Random,TOML,InteractiveUtils
include(joinpath(ARGS[1],"src","ProximalBase.jl"));const PB=Main.ProximalBase
include("adapters.jl");include("cases.jl")
function inspect_case(p)
    for _ in 1:4;p.setup();p.op();end
    p.setup();result=p.op();p.validate(result)
    p.setup();@allocated p.op()
end
cases,src=block_cases(Float64,8192,12,64,"contiguous",["block_cd"],852)
p=only(cases);before=inspect_case(p)
println("runtime-selection block adapter:");code_warntype(stdout,block,Tuple{typeof(src),Int};debuginfo=:none)
println("runtime-selection block update:");code_warntype(stdout,block_update!,Tuple{typeof(src),Vector{Int},Int,Bool};debuginfo=:none)
@static if !ORIGINAL && isdefined(PB,:blockview)
    println("direct native blockview:");code_warntype(stdout,PB.blockview,Tuple{typeof(src),Int};debuginfo=:none)
elseif !ORIGINAL && isdefined(PB,:atom)
    println("direct native atom:");code_warntype(stdout,PB.atom,Tuple{typeof(src),Int};debuginfo=:none)
end
# Define only a Main harness adapter; never patch a package method.
@static if ORIGINAL
    block(x,k)=x.atoms[k]
elseif isdefined(PB,:blockview)
    block(x,k)=PB.blockview(x,k)
else
    block(x,k)=PB.atom(x,k)
end
after=inspect_case(p)
println("static-selection block adapter:");code_warntype(stdout,block,Tuple{typeof(src),Int};debuginfo=:none)
println("static-selection block update:");code_warntype(stdout,block_update!,Tuple{typeof(src),Vector{Int},Int,Bool};debuginfo=:none)
println("ADAPTER_ALLOCATION_BYTES runtime=",before," static=",after)
