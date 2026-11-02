module TierSelectionTest

# The tier manifest, test selection and the CI risk policy (test/tiers.jl, test/tiers.toml).

using Test
import ParallelTestRunner

include(joinpath(@__DIR__, "tiers.jl"))
using .TestTiers: TestTiers, TIERS, ManifestError

const DIR = @__DIR__
const MANIFEST = TestTiers.load(joinpath(DIR, "tiers.toml"), joinpath(dirname(DIR), ".github", "tier-rules.toml"))

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
  # a focus file that no tier lists, a bad timeout
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[[risk]]\ntier = \"quick\"\npaths = [\"x\"]\nfocus = [\"zz\"]\n"))
  @test any(occursin("focus `zz`, which is in no tier", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[[risk]]\ntier = \"standard\"\nnarrow = \"standard\"\npaths = [\"x\"]\n"))
  @test any(occursin("narrow tier `standard`; use a tier below", s) for s in p)
  p = problems(() -> TestTiers.parse_manifest(SMALL * "[timeouts]\nquick = 0\n"))
  @test any(occursin("timeouts.quick", s) for s in p)
  @test MANIFEST.timeouts == Dict("quick" => 20, "standard" => 30, "thorough" => 45, "full" => 90)
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
  @test classify("src/utils.jl")[1] == "thorough"
  @test classify("src/calculus.jl", "test/calculus.jl") == ("thorough", ["calculus", "stepchecks"])
  @test classify("src/interface.jl", "docs/src/interface.md")[1] == "thorough"
  # the conformance suite and its coverage map
  @test classify("test/conformance/kernels.jl") == ("thorough", ["conformance/kernels"])
  @test classify("docs/design/numerical-contract/coverage.md") == ("thorough", String[])
  # infrastructure, platform code and paths no rule knows
  @test classify("src/lapack.jl")[1] == "full"
  @test classify("ext/ProximalBaseGPUArraysCoreExt.jl", "test/gpu.jl")[1] == "full"
  @test classify("test/runtests.jl")[1] == "full"
  @test classify(".github/tier-rules.toml")[1] == "full"
  @test classify("test/tiers.jl")[1] == "full"
  @test classify("scripts/ci_gate.jl")[1] == "full"
  # the tier lists decide what every tier runs, so any change to them gets full, also the line
  # of a new test file
  @test classify("test/tiers.toml")[1] == "full"
  @test classify("test/tiers.toml", "test/norms.jl") == ("full", ["norms"])
  # the API pages have @docs entries that test/docstrings.jl checks; prose needs no tests
  @test classify("docs/src/api/functions.md")[1] == "quick"
  @test classify("docs/src/numerics.md")[1] == "none"
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


# The tier and focus files a PR gets, as the plan job reports them.
pr(paths...; labels = String[]) = TestTiers.pr_tier(MANIFEST, collect(paths), labels)[1:2]
const NARROW = [TestTiers.NARROW_LABEL]
const ATOM = ["block_step", "infinite_group_weights", "norms", "real_iterates", "sparsegroup"]
const SPARSE = ["block_step", "conformance/failures_storage", "real_iterates", "sparse_iterate", "sparse_value_cost",
                "symmetric_argument", "symmetric_mul", "symmetric_mutator_errors", "symmetric_mutators", "symmetric_writes"]
const SYMPARAMS = ["symmetric_parameters_extensions", "symmetric_parameters_merge", "symmetric_parameters_paths",
                   "symmetric_parameters_trait", "symmetric_parameters_writes"]
instandard(names) = filter(n -> !(n in TestTiers.select(MANIFEST, "standard")), names)

@testset "iterate policy" begin
  # a narrow change to one implementation, declared by the label: quick and its focus files,
  # also without a test change
  @test pr("src/atom_iterate.jl"; labels = NARROW) == ("quick", ATOM)
  @test pr("src/atom_iterate.jl", "test/atom_iterate.jl"; labels = NARROW) == ("quick", ATOM)
  @test pr("src/sparse_iterate.jl"; labels = NARROW) == ("quick", SPARSE)
  @test pr("src/symmetric_parameters.jl"; labels = NARROW) == ("quick", SYMPARAMS)
  # without the label: standard and the same focus files (those standard lacks)
  @test pr("src/atom_iterate.jl") == ("standard", instandard(ATOM))
  @test pr("src/sparse_iterate.jl", "test/sparse_iterate.jl") == ("standard", instandard(SPARSE))
  @test pr("src/symmetric_parameters.jl") == ("standard", SYMPARAMS)
  # two implementations: the label does not apply; standard and both groups
  @test pr("src/sparse_iterate.jl", "src/atom_iterate.jl"; labels = NARROW) ==
        ("standard", instandard(sort!(union(SPARSE, ATOM))))
  @test pr("src/sparse_iterate.jl", "src/symmetric_parameters.jl"; labels = NARROW)[1] == "standard"
  # a shared or architectural change gets test:thorough by hand; test:full always wins
  @test pr("src/sparse_iterate.jl"; labels = ["test:thorough"])[1] == "thorough"
  @test pr("src/sparse_iterate.jl"; labels = [NARROW; "test:thorough"])[1] == "thorough"
  @test pr("src/atom_iterate.jl"; labels = [NARROW; "test:full"]) == ("full", String[])
  @test pr("src/atom_iterate.jl"; labels = [NARROW; "test:standard"]) == ("standard", instandard(ATOM))
  # the label never lowers another path: shared code, infrastructure, unknown paths
  @test pr("src/sparse_iterate.jl", "src/utils.jl"; labels = NARROW)[1] == "thorough"
  @test pr("src/atom_iterate.jl", "src/interface.jl"; labels = NARROW)[1] == "thorough"
  @test pr("src/atom_iterate.jl", "Project.toml"; labels = NARROW)[1] == "full"
  @test pr("src/atom_iterate.jl", ".github/workflows/CI.yml"; labels = NARROW)[1] == "full"
  @test pr("src/atom_iterate.jl", "a/new/file.txt"; labels = NARROW)[1] == "full"
  # nor package code without tests: other source still needs standard, with or without a test
  # change elsewhere
  @test pr("src/atom_iterate.jl", "src/functions/entropy.jl"; labels = NARROW)[1] == "standard"
  @test pr("src/atom_iterate.jl", "src/functions/norms.jl", "test/norms.jl"; labels = NARROW)[1] == "standard"
  @test pr("src/functions/entropy.jl", "test/entropy.jl"; labels = NARROW) == ("quick", String[])
  @test pr("src/functions/norms.jl"; labels = NARROW) == pr("src/functions/norms.jl") == ("standard", String[])
  # the label only adds information: the same paths without it never get a lower tier, and the
  # focus files are the same
  for paths in (["src/atom_iterate.jl"], ["src/sparse_iterate.jl", "src/utils.jl"], ["src/functions/sets.jl"],
                ["docs/src/index.md"], ["src/atom_iterate.jl", "test/norms.jl"])
    t1, f1, _ = TestTiers.classify(MANIFEST, paths; narrow = true)
    t2, f2, _ = TestTiers.classify(MANIFEST, paths)
    @test TestTiers.rank(t1) <= TestTiers.rank(t2) && f1 == f2
  end
  # documentation keeps no tests with the label; an unknown test label is an error
  @test pr("docs/src/index.md"; labels = NARROW) == ("none", String[])
  @test_throws ArgumentError pr("src/atom_iterate.jl"; labels = ["test:iterate"])
  # the decision table says when and why the label applies
  _, _, d = TestTiers.classify(MANIFEST, ["src/atom_iterate.jl"]; narrow = true)
  @test occursin("narrow change", d[1][3])
  _, _, d = TestTiers.classify(MANIFEST, ["src/atom_iterate.jl", "src/sparse_iterate.jl"]; narrow = true)
  @test d[1][1] == "(label test:iterate-narrow)" && occursin("more than one iterate implementation", d[1][3])
  # the result does not depend on the order of the paths
  paths = ["src/sparse_iterate.jl", "test/norms.jl", "src/atom_iterate.jl", "README.md"]
  @test TestTiers.classify(MANIFEST, paths)[1:2] == TestTiers.classify(MANIFEST, reverse(paths))[1:2]
  # every focus file runs in full, and the iterate rules are the only ones the label can lower
  full = TestTiers.select(MANIFEST, "full")
  for rule in MANIFEST.rules
    @test issubset(rule.focus, full)
    isempty(rule.narrow) || @test all(p -> startswith(p, "src/") && occursin("iterate", p) || p == "src/symmetric_parameters.jl", rule.patterns)
  end
end

@testset "additive changes to the tier lists" begin
  # the edits below work on LF text; a Windows checkout has CRLF line endings
  old = replace(read(joinpath(DIR, "tiers.toml"), String), "\r\n" => "\n")
  add(text, tier, name) = replace(text, "\n$tier = [\n" => "\n$tier = [\n  \"$name\",\n"; count = 1)
  # new files only: their names, which then run as changed test files
  @test TestTiers.manifest_additions(old, add(old, "quick", "iterate_new"), ["iterate_new"]) == (["iterate_new"], String[])
  # line endings do not matter
  @test TestTiers.manifest_additions(old, replace(add(old, "quick", "iterate_new"), "\n" => "\r\n"), ["iterate_new"]) ==
        (["iterate_new"], String[])
  two = add(add(old, "full", "b_new"), "standard", "a_new")
  @test TestTiers.manifest_additions(old, two, ["a_new", "b_new"]) == (["a_new", "b_new"], String[])
  # comments and order alone change nothing
  @test TestTiers.manifest_additions(old, replace(old, "# iterates:" => "# the iterates:"), String[]) == (String[], String[])
  swapped = replace(old, "  \"aqua\",\n  \"ci_gate\",\n" => "  \"ci_gate\",\n  \"aqua\",\n")
  @test swapped != old && TestTiers.manifest_additions(old, swapped, String[]) == (String[], String[])
  # anything else needs full
  a, p = TestTiers.manifest_additions(old, add(old, "quick", "iterate_new"), String[])
  @test a === nothing && occursin("not a new file of this change", only(p))
  moved = add(replace(old, "  \"utils\",\n" => ""; count = 1), "standard", "utils")
  a, p = TestTiers.manifest_additions(old, moved, String[])
  @test a === nothing && only(p) == "`utils` moves from `quick` to `standard`"
  a, p = TestTiers.manifest_additions(old, replace(old, "  \"utils\",\n" => ""; count = 1), String[])
  @test a === nothing && only(p) == "`utils` is removed from tier `quick`"
  a, p = TestTiers.manifest_additions(old, add(replace(old, "  \"utils\",\n" => ""; count = 1), "quick", "utils2"), ["utils2"])
  @test a === nothing && only(p) == "`utils` is removed from tier `quick`" # a rename
  a, p = TestTiers.manifest_additions(old, replace(old, "\"testutils\", \"tiers\"]" => "\"testutils\", \"tiers\", \"new_helper\"]"), String[])
  @test a === nothing && occursin("outside [tiers]", only(p))
  a, p = TestTiers.manifest_additions(old, replace(old, "environments = [\"cuda\", \"oracles\"]" => "environments = [\"cuda\"]"), String[])
  @test a === nothing && occursin("outside [tiers]", only(p))
  a, p = TestTiers.manifest_additions(old, old * "\n[tiers.extra]\n", String[])
  @test a === nothing
  a, p = TestTiers.manifest_additions(old, "not toml [", String[])
  @test a === nothing && occursin("could not be compared", only(p))
  # the plan: additions give quick and run the new file; otherwise the tier lists need full
  mktempdir() do dir
    path = joinpath(dir, "tiers.toml")
    write(path, add(old, "full", "iterate_new"))
    m = TestTiers.load(path, joinpath(dirname(DIR), ".github", "tier-rules.toml"))
    paths = ["test/iterate_new.jl", "test/tiers.toml"]
    @test TestTiers.pr_tier(m, paths, String[]; manifest_additions = ["iterate_new"])[1:2] == ("quick", ["iterate_new"])
    @test TestTiers.pr_tier(m, paths, String[])[1:2] == ("full", String[])
    @test TestTiers.pr_tier(m, [paths; "src/atom_iterate.jl"], NARROW; manifest_additions = ["iterate_new"])[1:2] ==
          ("quick", sort!([ATOM; "iterate_new"]))
    @test TestTiers.pr_tier(m, [paths; "src/utils.jl"], String[]; manifest_additions = ["iterate_new"])[1] == "thorough"
    # an unlisted test file still stops the plan
    @test_throws ManifestError TestTiers.check(m, [TestTiers.discover(DIR); "unlisted_new"])
  end
end

end
