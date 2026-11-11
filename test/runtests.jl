using Test, ProximalBase, LinearAlgebra
BLAS.set_num_threads(1)
include("tiers.jl")
include("testutils.jl")
manifest = TestTiers.inventory()
tier = "full"
selectors = String[]
list_only = false
for arg in ARGS
    if startswith(arg,"--tier=")
        global tier = split(arg,'=';limit=2)[2]
    elseif arg == "--list"
        global list_only = true
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
isempty(selected) && error("no tests selected")
println("ProximalBase minimal tests: tier $tier; $(length(selected)) files; Julia $VERSION on $(Sys.MACHINE)")
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
