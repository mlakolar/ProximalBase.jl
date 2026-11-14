# Reuse #852's unchanged measurement, sample export, warmup, allocation and
# provenance implementation. Only choose this issue's fixtures and allow the
# original raw AtomIterate constructor (as in #852's raw-block control).
path=joinpath(@__DIR__,"..","audit852","worker.jl")
code=read(path,String)
for (from,to) in (
    "include(\"adapters.jl\")" => "include($(repr(joinpath(@__DIR__,"..","audit852","adapters.jl"))))",
    "include(\"cases.jl\")" => "include($(repr(joinpath(@__DIR__,"cases.jl"))))",
    "incompatible=ORIGINAL && fam==\"block\" && (T==Float32 || layout!=\"contiguous\")" => "incompatible=false",
    "block_cases(T,n,s,b,layout,ops,seed)" => "replacement_cases(T,n,s,b,layout,ops,seed)")
    occursin(from,code) || error("#852 worker changed: $from")
    global code=replace(code,from=>to)
end
include_string(Main,code,joinpath(@__DIR__,"worker-generated.jl"))
