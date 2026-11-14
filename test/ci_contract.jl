module CIContractTest
using Test, TOML
const root=dirname(@__DIR__)
workflow=read(joinpath(root,".github","workflows","CI.yml"),String)
@testset "minimal required CI contract" begin
    @test occursin("name: PR required jobs",workflow)
    @test occursin("name: PR merge gate",workflow)
    @test occursin("version: '1'",workflow)
    @test occursin("--check-bounds=yes",workflow)
    @test occursin("shard: [1, 2, 3]",workflow)
    @test occursin("needs: [test, benchmark, docs]",workflow)
    @test occursin("BENCHMARK_RESULT:",workflow)
    @test occursin("DOCS_RESULT:",workflow)
    @test occursin("PB_CHECK_ONLY: '1'",workflow)
    @test occursin("docs/make.jl",workflow)
    @test !occursin("continue-on-error",workflow)
    rules=TOML.parsefile(joinpath(root,".github","tier-rules.toml"))
    @test Set(keys(rules["platforms"]))==Set(["linux"])
end
@testset "shards cover the complete supported inventory exactly once" begin
    selected=Main.TestTiers.select(Main.TestTiers.inventory(),"full")
    parts=[Main.TestTiers.shard_files(selected,i,3) for i in 1:3]
    @test all(x -> !isempty(x),parts)
    combined=reduce(vcat,parts)
    @test sort(combined)==sort(selected)
    @test length(unique(combined))==length(selected)
    @test_throws ErrorException Main.TestTiers.shard_files(selected,0,3)
    @test_throws ErrorException Main.TestTiers.shard_files(selected,4,3)
    @test_throws ErrorException Main.TestTiers.shard_files(selected,1,0)
    @test_throws ErrorException Main.TestTiers.shard_files(selected,1,length(selected)+1)
end
@testset "manifest rejects unclassified tests" begin
    mktempdir() do dir
        write(joinpath(dir,"tiers.toml"),"[tiers]\nquick=[]\nstandard=[]\nthorough=[]\nfull=[]\n[not_tests]\nfiles=[]\n")
        @test isempty(Main.TestTiers.select(Main.TestTiers.inventory(dir),"full"))
        write(joinpath(dir,"unexpected.jl"),"nothing\n")
        @test_throws ErrorException Main.TestTiers.inventory(dir)
    end
end
end
