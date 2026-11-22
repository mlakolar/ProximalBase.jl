using Test, ProximalBase, LinearAlgebra
BLAS.set_num_threads(1)
include("tiers.jl")
include("testutils.jl")
manifest = TestTiers.inventory()
tier = "full"
selectors = String[]
list_only = false
exact = false
shard = nothing
for arg in ARGS
    if startswith(arg,"--tier=")
        global tier = split(arg,'=';limit=2)[2]
    elseif arg == "--list"
        global list_only = true
    elseif arg == "--exact"
        global exact = true
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
# With --exact every positional selector is a manifest name of the tier; otherwise it is a name prefix.
exact && isempty(selectors) && error("--exact needs at least one test name")
for selector in selectors
    if exact
        selector in selected || error("no test file named $selector in tier $tier")
    else
        any(n -> startswith(n,selector),selected) || error("no test file matches $selector in tier $tier")
    end
end
if !isempty(selectors)
    exact ? filter!(in(selectors),selected) : filter!(n -> any(s -> startswith(n,s),selectors),selected)
end
if shard !== nothing
    selected = TestTiers.shard_files(selected,shard...; weights=manifest.weights)
end
isempty(selected) && error("no tests selected")
shard_label = shard === nothing ? "all" : "$(shard[1])/$(shard[2])"
println("ProximalBase minimal tests: tier $tier; shard $shard_label; $(length(selected)) files; Julia $VERSION on $(Sys.MACHINE)")
println("  ",join(selected," "))
flush(stdout)

function timed_test_file(cmd, name)
    status = "error"
    start_ns = time_ns()
    try
        passed = success(run(ignorestatus(cmd)))
        status = passed ? "pass" : "fail"
        return passed
    finally
        elapsed_s = round((time_ns() - start_ns) / 1e9; digits=3)
        println("PB_TEST_FILE_TIME file=$name elapsed_s=$elapsed_s status=$status")
        flush(stdout)
    end
end

if !list_only
    shard_start_ns = time_ns()
    try
        @testset "ProximalBase minimal core" begin
            for name in selected
                @testset "$name" begin
                    # Each file gets a fresh process, preserving isolation from fixture methods.
                    runner = "include(ARGS[1]); include(ARGS[2]); include(ARGS[3])"
                    cmd = `$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(dirname(Base.active_project())) -e $runner $(joinpath(@__DIR__,"tiers.jl")) $(joinpath(@__DIR__,"testutils.jl")) $(joinpath(@__DIR__,name*".jl"))`
                    @test timed_test_file(cmd, name)
                end
            end
        end
    finally
        elapsed_s = round((time_ns() - shard_start_ns) / 1e9; digits=3)
        println("PB_TEST_SHARD_TIME shard=$shard_label files=$(length(selected)) elapsed_s=$elapsed_s")
        flush(stdout)
    end
end
