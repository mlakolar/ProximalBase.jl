module TierSelectionTest

# The tier manifest, test selection and the CI risk policy (test/tiers.jl, test/tiers.toml).

using Test
import ParallelTestRunner

include(joinpath(@__DIR__, "tiers.jl"))
using .TestTiers: TestTiers, TIERS, ManifestError

const DIR = @__DIR__
const MANIFEST = TestTiers.load(joinpath(DIR, "tiers.toml"))

@testset "the manifest covers every test file" begin
  found = TestTiers.discover(DIR)
  @test Set(found) == Set(keys(ParallelTestRunner.find_tests(DIR)))
  @test TestTiers.check(MANIFEST, found) === nothing
  # full runs every file except the helpers and the separate environments
  tests = filter(n -> !(n in MANIFEST.helpers || TestTiers.isenvironment(MANIFEST, n)), found)
  @test TestTiers.select(MANIFEST, "full") == sort(tests)
  @test any(startswith("cuda/"), found)  # cuda/instantiate.jl; runtests.jl files are never found
  @test !any(n -> startswith(n, "cuda/") || startswith(n, "oracles/"), TestTiers.select(MANIFEST, "full"))
end

@testset "tiers are cumulative and each adds files" begin
  selections = [TestTiers.select(MANIFEST, t) for t in TIERS]
  for (smaller, larger) in zip(selections, selections[2:end])
    @test issubset(smaller, larger)
    @test length(smaller) < length(larger)
  end
  @test !isempty(first(selections))
  @test_throws ArgumentError TestTiers.select(MANIFEST, "fast")
  @test_throws ArgumentError TestTiers.select(MANIFEST, "")
end

const SMALL = """
  [tiers]
  quick = ["a"]
  standard = ["b"]
  thorough = ["c"]
  full = ["d/e"]
  [not_tests]
  helpers = ["h"]
  environments = ["env"]
  """

problems(f) = try
  f()
  String[]
catch err
  err isa ManifestError || rethrow()
  err.problems
end

@testset "manifest problems are reported" begin
  m = TestTiers.parse_manifest(SMALL)
  @test TestTiers.select(m, "quick") == ["a"]
  @test TestTiers.select(m, "thorough") == ["a", "b", "c"]
  @test TestTiers.select(m, "full") == ["a", "b", "c", "d/e"]
  @test TestTiers.check(m, ["a", "b", "c", "d/e", "h", "env/runtests2", "env/x/y"]) === nothing
  # a new file without a tier
  p = problems(() -> TestTiers.check(m, ["a", "b", "c", "d/e", "h", "new"]))
  @test length(p) == 1 && occursin("`test/new.jl` has no tier", p[1])
  # a listed file that does not exist, and a missing helper
  p = problems(() -> TestTiers.check(m, ["a", "b", "c"]))
  @test any(occursin("`d/e` is in tier `full`", s) for s in p)
  @test any(occursin("helper `h` does not exist", s) for s in p)
  # a file in two tiers, an empty tier, an unknown tier
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "standard = [\"b\"]" => "standard = [\"a\"]")))
  @test any(occursin("`a` is listed in both `quick` and `standard`", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "thorough = [\"c\"]" => "thorough = []")))
  @test any(occursin("tier `thorough` lists no test files", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "fast = [\"z\"]\n"))
  @test isempty(p)  # a key after [not_tests] belongs to that table
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "[not_tests]" => "fast = [\"z\"]\n[not_tests]")))
  @test any(occursin("unknown tier `fast`", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[[risk]]\ntier = \"fast\"\npaths = [\"x\"]\n"))
  @test any(occursin("risk rule 1 has tier `fast`", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(replace(SMALL, "helpers = [\"h\"]" => "helpers = [\"a\"]")))
  @test any(occursin("`a` is both a helper and in tier `quick`", s) for s in p)
end

@testset "path patterns" begin
  matches(pattern, path) = occursin(TestTiers.glob_regex(pattern), path)
  @test matches("src/**", "src/functions/norms.jl")
  @test matches("src/*.jl", "src/utils.jl")
  @test !matches("src/*.jl", "src/functions/norms.jl")
  @test matches("**/*.md", "README.md") && matches("**/*.md", "test/oracles/README.md")
  @test !matches("**/*.md", "README.mdx")
  @test matches("docs/**", "docs/src/index.md") && !matches("docs/**", "docsx/a")
  @test matches("test/gpu_suite.jl", "test/gpu_suite.jl") && !matches("test/gpu_suite.jl", "test/gpu_suiteXjl")
end

classify(paths...) = TestTiers.classify(MANIFEST, collect(paths))[1:2]

@testset "risk policy" begin
  # documentation needs no package tests
  @test classify("docs/src/index.md", "CONTRIBUTING.md") == ("none", String[])
  @test classify("benchmark/run.jl", "test/oracles/runtests.jl") == ("none", String[])
  # README examples are tests
  @test classify("README.md") == ("quick", ["readme"])
  # a changed test file runs beside the tier
  @test classify("test/norms.jl") == ("quick", ["norms"])
  @test classify("test/gpu_suite.jl") == ("quick", ["gpu", "gpu_smoke"])
  # a self-contained family with a test change: quick and its own tests
  @test classify("src/functions/entropy.jl", "test/entropy.jl") == ("quick", ["entropy"])
  @test classify("src/functions/cones.jl", "test/soc.jl") == ("quick", ["expcone", "rotatedsoc", "soc"])
  # package code without any test change: at least standard
  @test classify("src/functions/entropy.jl") == ("standard", ["entropy"])
  @test classify("src/functions/norms.jl", "test/norms.jl") == ("standard", ["norms"])
  # a family whose own tests are in the full tier runs them beside standard
  @test classify("src/functions/sets.jl", "test/sets.jl") == ("standard", ["hyperslab", "sets"])
  @test classify("src/composition.jl") == ("standard", ["composition", "precompose_array"])
  # shared code
  @test classify("src/sparse_iterate.jl", "test/sparse_iterate.jl")[1] == "thorough"
  @test classify("src/utils.jl")[1] == "thorough"
  @test classify("src/interface.jl", "docs/src/interface.md")[1] == "thorough"
  # the conformance suite and its coverage map
  @test classify("test/conformance/kernels.jl") == ("thorough", ["conformance/kernels"])
  @test classify("docs/design/numerical-contract/coverage.md") == ("thorough", String[])
  # infrastructure, platform code and paths no rule knows
  @test classify("src/lapack.jl")[1] == "full"
  @test classify("ext/ProximalBaseGPUArraysCoreExt.jl", "test/gpu.jl")[1] == "full"
  @test classify("test/runtests.jl")[1] == "full"
  @test classify("test/tiers.toml")[1] == "full"
  @test classify(".github/workflows/CI.yml")[1] == "full"
  @test classify("Project.toml")[1] == "full"
  @test classify("a/new/file.txt")[1] == "full"
  @test classify(".github/dependabot.yml")[1] == "full"
  # a deleted or unlisted test file is no focus file; the runner rejects an unlisted one
  @test classify("test/removed_file.jl") == ("quick", String[])
  # every rule decides the path it is written for
  for rule in MANIFEST.rules, pattern in rule.patterns
    occursin('*', pattern) && continue
    t, = TestTiers.classify(MANIFEST, [pattern])
    i = findfirst(r -> any(re -> occursin(re, pattern), r.regexes), MANIFEST.rules)
    @test MANIFEST.rules[i] === rule
  end
end

end
