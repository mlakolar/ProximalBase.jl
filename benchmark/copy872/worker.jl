# Reuse #852's unchanged measurement, sample export, warmup, allocation and
# provenance implementation, as #861's worker does. Two adaptations: the fixtures are
# #872's, and #852's guard that skips the original implementation for Float32 and
# non-contiguous block layouts is off, since #861's raw constructor supports them.
path=joinpath(@__DIR__,"..","audit852","worker.jl")
code=read(path,String)
for (from,to) in (
    "include(\"adapters.jl\")" => "include($(repr(joinpath(@__DIR__,"..","audit852","adapters.jl"))))",
    "include(\"cases.jl\")" => "include($(repr(joinpath(@__DIR__,"cases.jl"))))",
    "incompatible=ORIGINAL && fam==\"block\" && (T==Float32 || layout!=\"contiguous\")" => "incompatible=false",
    "block_cases(T,n,s,b,layout,ops,seed)" => "cases872(T,n,s,b,layout,ops,seed)")
    occursin(from,code) || error("#852 worker changed: $from")
    global code=replace(code,from=>to)
end
include_string(Main,code,joinpath(@__DIR__,"worker-generated.jl"))
