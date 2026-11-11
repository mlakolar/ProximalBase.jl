# Print the actual supported test inventory used by the single-platform CI.
include(joinpath(@__DIR__,"..","test","tiers.jl"))
manifest=TestTiers.inventory(joinpath(@__DIR__,"..","test"))
println("platforms=linux; julia=1; tier=full")
println(join(TestTiers.select(manifest,"full")," "))
