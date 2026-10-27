module ReadmeTest

using Test

const README = joinpath(dirname(@__DIR__), "README.md")

@testset "README" begin
  blocks = [m.captures[1] for m in eachmatch(r"(?m)^```julia\r?\n(.*?)^```\s*$"s, read(README, String))]
  @test length(blocks) == 5
  installation = replace(first(blocks), "\r\n" => "\n")
  @test strip(installation) == "using Pkg\nPkg.add(\"ProximalBase\")"
  examples = blocks[2:end]
  @test length(examples) == 4
  m = Module(:ReadmeExample)
  for (i, code) in enumerate(examples)
    @testset "block $i" begin
      Base.include_string(m, code, README)
      @test true
    end
  end
end

end
