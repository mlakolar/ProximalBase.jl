# Dispatch, specialization and generated-code evidence, outside measurement runs.
using BenchmarkTools,LinearAlgebra,Statistics,SHA,TOML,Dates,InteractiveUtils
BLAS.set_num_threads(1)
include(joinpath(ARGS[1],"src","ProximalBase.jl"));const PB=Main.ProximalBase
include("adapters.jl");include("cases.jl")
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
end
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
