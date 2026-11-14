using Test, ProximalBase, LinearAlgebra
BLAS.set_num_threads(1)
include("tiers.jl")
include("testutils.jl")
manifest = TestTiers.inventory()
tier = "full"
selectors = String[]
list_only = false
shard = nothing
for arg in ARGS
    if startswith(arg,"--tier=")
        global tier = split(arg,'=';limit=2)[2]
    elseif arg == "--list"
        global list_only = true
    elseif startswith(arg,"--shard=")
        shard === nothing || error("shard specified twice")
        parts = split(split(arg,'=';limit=2)[2],'/')
        length(parts) == 2 || error("shard must be in the form index/count")
        global shard = (parse(Int,parts[1]),parse(Int,parts[2]))
    elseif startswith(arg,"--")
        error("unknown test option: $arg")
    else
        push!(selectors,arg)
    end
end
selected = TestTiers.select(manifest,tier)
for selector in selectors
    any(n -> startswith(n,selector),selected) || error("no test file matches $selector in tier $tier")
end
isempty(selectors) || filter!(n -> any(s -> startswith(n,s),selectors),selected)
if shard !== nothing
    selected = TestTiers.shard_files(selected,shard...)
end
isempty(selected) && error("no tests selected")
shard_label = shard === nothing ? "all" : "$(shard[1])/$(shard[2])"
println("ProximalBase minimal tests: tier $tier; shard $shard_label; $(length(selected)) files; Julia $VERSION on $(Sys.MACHINE)")
println("  ",join(selected," "))
flush(stdout)
if !list_only
    @testset "ProximalBase minimal core" begin
        for name in selected
            @testset "$name" begin
                # Each file gets a fresh process, preserving isolation from fixture methods.
                runner = "include(ARGS[1]); include(ARGS[2]); include(ARGS[3])"
                cmd = `$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(dirname(Base.active_project())) -e $runner $(joinpath(@__DIR__,"tiers.jl")) $(joinpath(@__DIR__,"testutils.jl")) $(joinpath(@__DIR__,name*".jl"))`
                @test success(run(ignorestatus(cmd)))
            end
        end
    end
end
