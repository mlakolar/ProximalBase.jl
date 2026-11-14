using LinearAlgebra,Test
include(joinpath(get(ENV,"PB_BASELINE","/tmp/pb849-baseline"),"src","ProximalBase.jl"))
repo=normpath(joinpath(@__DIR__,"..","..","..",".."))
s=read(`git -C $repo show eff21b6d7d33c622313122f46c6ee71693c101ee:test/symmetric_iterate_interface.jl`,String)
s=replace(s,"using Test, ProximalBase,"=>"using Test, Main.ProximalBase,")
Base.include_string(Main,s,"frozen-symmetric-iterate-interface.jl")
