module CIContractTest
using Test, TOML
const root=dirname(@__DIR__)
workflow=read(joinpath(root,".github","workflows","CI.yml"),String)
@testset "minimal required CI contract" begin
    @test occursin("name: PR required jobs",workflow)
    @test occursin("name: PR merge gate",workflow)
    @test occursin("version: '1'",workflow)
    @test occursin("--check-bounds=yes",workflow)
    @test occursin("docs/make.jl",workflow)
    @test !occursin("continue-on-error",workflow)
    @test !occursin("matrix:",workflow)
    rules=TOML.parsefile(joinpath(root,".github","tier-rules.toml"))
    @test Set(keys(rules["platforms"]))==Set(["linux"])
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
