# Source-loaded first-use diagnostics, separate from warmed timing claims.
using LinearAlgebra, SparseArrays, Random, TOML
BLAS.set_num_threads(1)
load_stats=@timed include(joinpath(ARGS[1],"src","ProximalBase.jl"))
const PB=Main.ProximalBase
harness=length(ARGS)>=3 ? ARGS[3] : @__DIR__
include(joinpath(harness,"adapters.jl"));include(joinpath(harness,"cases.jl"))
function stats(t)
    Dict("wall_seconds"=>t.time,"bytes"=>t.bytes,"gc_seconds"=>t.gctime,
         "compile_seconds"=>t.compile_time,"recompile_seconds"=>t.recompile_time)
end
result=Dict{String,Any}("julia"=>string(VERSION),"source"=>ARGS[1],"source_load"=>stats(load_stats))
for fam in ("sparse","symmetric","block")
    n=fam=="sparse" ? 1000000 : fam=="symmetric" ? 1414 : 8192
    op=fam=="sparse" ? "cd_fixed" : fam=="symmetric" ? "symmetric_updates" : "block_cd"
    build_stats=@timed (fam=="block" ? block_cases(Float64,n,100,8,"contiguous",[op],852) : scalar_cases(fam,Float64,n,1000,[op],852))
    cases,src=build_stats.value;p=only(cases);p.setup()
    first_stats=@timed p.op();p.validate(first_stats.value)
    for _ in 1:3;p.setup();p.op();end
    p.setup();warm_stats=@timed p.op()
    result[fam]=Dict("build_and_reference_fixture"=>stats(build_stats),"first_operation"=>stats(first_stats),"warm_single_call"=>stats(warm_stats))
end
open(ARGS[2],"w") do io;TOML.print(io,result);end
