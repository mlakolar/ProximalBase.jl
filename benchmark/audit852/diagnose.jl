# Dispatch, specialization and generated-code evidence, outside measurement runs.
using BenchmarkTools,LinearAlgebra,Statistics,SHA,TOML,Dates,InteractiveUtils
BLAS.set_num_threads(1)
include(joinpath(ARGS[1],"src","ProximalBase.jl"));const PB=Main.ProximalBase
harness=length(ARGS)>=3 ? ARGS[3] : @__DIR__
include(joinpath(harness,"adapters.jl"));include(joinpath(harness,"cases.jl"))
out=ARGS[2];mkpath(out)
function dumpcode(name,f,types)
    open(joinpath(out,name*".warntype.txt"),"w") do io;code_warntype(io,f,types;debuginfo=:none);end
    open(joinpath(out,name*".llvm.txt"),"w") do io;code_llvm(io,f,types;debuginfo=:none);end
    open(joinpath(out,name*".native.txt"),"w") do io;code_native(io,f,types;debuginfo=:none,syntax=:intel);end
end
for (b,s) in ((1,819),(8,102),(64,12))
    cases,src=block_cases(Float64,8192,s,b,"contiguous",["block_cd","copyto"],852)
    for p in cases
        for _ in 1:3;p.setup();p.op();end
        dumpcode("block_b$(b)_$(p.id)",p.op,Tuple{})
    end
    b==64 && dumpcode("block_update_kernel",block_update!,Tuple{typeof(src),Vector{Int},Int,Bool})
end
if isdefined(Main,:residual_cd!)
    cases,src=scalar_cases("sparse",Float64,1000000,1000,["residual_cd_fixed"],852)
    p=only(cases);for _ in 1:3;p.setup();p.op();end
    dumpcode("sparse_residual_cd_fixed",p.op,Tuple{})
    for layout in ("contiguous","strided")
        cases,src=block_cases(Float64,8192,102,8,layout,["residual_bcd"],852)
        p=only(cases);for _ in 1:3;p.setup();p.op();end
        dumpcode("block_residual_bcd_"*layout,p.op,Tuple{})
    end
    cases,src=scalar_cases("symmetric",Float64,144,10000,["bilinear","bilinear_native","matmat"],852)
    for p in cases
        for _ in 1:3;p.setup();p.op();end
        dumpcode("symmetric_"*p.id,p.op,Tuple{})
    end
end
cases,src=scalar_cases("sparse",Float64,1000000,1000,["dropzeros","prune_rebuild"],852)
for p in cases
    for _ in 1:3;p.setup();p.op();end
    dumpcode("sparse_"*p.id,p.op,Tuple{})
end
dumpcode("sparse_dropzeros_kernel",SparseArrays.dropzeros!,Tuple{typeof(src)})
dumpcode("sparse_constructor_f64",PB.SparseIterate,Tuple{Type{Float64},Int})
dumpcode("sparse_constructor_f32",PB.SparseIterate,Tuple{Type{Float32},Int})
open(joinpath(out,"dispatch.txt"),"w") do io
    x=newblock(Float64,8192,8,group_layout(8192,8,"contiguous",Random.Xoshiro(852)))
    v=block(x,1)
    println(io,"owner type: ",typeof(x),"\nblock view type: ",typeof(v))
    for (f,ts) in ((copyto!,Tuple{typeof(x),typeof(x)}),(getindex,Tuple{typeof(v),Int}),
                   (setindex!,Tuple{typeof(v),Float64,Int}))
        println(io,f," => ",which(f,ts))
    end
    println(io,"bounds=",Base.JLOptions().check_bounds," opt=",Base.JLOptions().opt_level," threads=",Threads.nthreads())
end
