# Print the actual supported test inventory used by the single-platform CI.
# Documentation-only changes skip the test inventory (see scripts/ci_changes.py).
include(joinpath(@__DIR__,"..","test","tiers.jl"))
manifest=TestTiers.inventory(joinpath(@__DIR__,"..","test"))
println("platforms=linux; julia=1.13; tier=full")
println(join(TestTiers.select(manifest,"full")," "))
# CI probe comment, never merged.
