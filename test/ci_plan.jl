module CIPlanTest

# The planner of the CI workflow (scripts/ci_plan.jl): which tier, platforms, coverage and
# benchmark reference checks each kind of run gets, from the policy in .github/tier-rules.toml.
# `decide` takes the changed files, labels and inputs as data, so no test needs git history.

using Test

module CIPlan
include(joinpath(dirname(@__DIR__), "scripts", "ci_plan.jl"))
end

const ROOT = dirname(@__DIR__)
const RULES = joinpath(ROOT, ".github", "tier-rules.toml")
const MANIFEST = CIPlan.default_manifest()
const POLICY = CIPlan.load_policy()
const TT = CIPlan.TestTiers

decide(event; kw...) = CIPlan.decide(event; manifest = MANIFEST, policy = POLICY, kw...)
planned(event; kw...) = decide(event; kw...)[1]
pr(paths...; labels = String[]) = planned("pull_request"; paths = collect(paths), labels)
dispatch(; kw...) = planned("workflow_dispatch"; inputs = Dict{String, String}(String(k) => string(v) for (k, v) in kw))
jobnames(s) = isempty(s) ? String[] : String.(split(s, ';'))

const L1 = "Julia 1 - ubuntu-latest - x64"
const LTS = "Julia lts - ubuntu-latest - x64"
const MAC = "Julia 1 - macos-latest - aarch64"
const WIN = "Julia 1 - windows-latest - x64"
const PRE = "Julia pre - ubuntu-latest - x64"

# The same text, with LF endings (a Windows checkout may have CRLF).
lf(path) = replace(read(path, String), "\r\n" => "\n")

@testset "pull request: the tier decides, the platforms follow it" begin
  # quick, standard: Julia 1 on Linux only, no coverage, no benchmark jobs, no conformance reports
  for (paths, tier, minutes) in ((["test/norms.jl"], "quick", 20),
                                 (["src/functions/norms.jl", "test/norms.jl"], "standard", 30),
                                 (["src/functions/entropy.jl"], "standard", 30))  # source without a test change
    r = pr(paths...)
    @test r.tier == tier && r.run_tests && r.timeout == minutes
    @test r.platforms == "stable" && jobnames(r.required) == ["$L1 - $tier"] && isempty(r.advisory)
    @test !r.coverage && !r.benchmark && !r.conformance
  end
  # thorough, full: Julia 1 and lts on Linux, and the conformance reports
  for (paths, tier) in ((["src/utils.jl"], "thorough"), (["test/conformance/kernels.jl"], "thorough"),
                        (["Project.toml"], "full"), (["scripts/ci_plan.jl"], "full"), (["a/new/file.txt"], "full"))
    r = pr(paths...)
    @test r.tier == tier && r.platforms == "linux" && r.conformance
    @test sort(jobnames(r.required)) == sort(["$LTS - $tier", "$L1 - $tier"]) && isempty(r.advisory)
    @test !r.coverage
  end
  # documentation only: no tests, but a valid matrix and no job names
  r = pr("docs/src/index.md", "CONTRIBUTING.md")
  @test r.tier == "none" && !r.run_tests && r.timeout == 0 && r.focus == ""
  @test r.platforms == "stable" && isempty(r.required) && isempty(r.advisory) && !r.benchmark && !r.conformance
  @test startswith(r.matrix, "{\"include\":[{") && !r.coverage
  # a changed test file runs beside the tier; the focus files of a rule too
  @test pr("test/norms.jl").focus == "norms"
  @test pr("test/aqua.jl").focus == "" # quick runs it anyway
  # an added line of a new test file in test/tiers.toml keeps quick; any other change is full
  r = planned("pull_request"; paths = ["test/tiers.toml", "test/ci_plan.jl"], additions = ["ci_plan"])
  @test r.tier == "quick" && r.platforms == "stable"
  r, lines = decide("pull_request"; paths = ["test/tiers.toml", "test/ci_plan.jl"], additions = nothing,
                    problems = ["`utils` is removed from tier `quick`"])
  @test r.tier == "full" && r.platforms == "linux" && any(occursin("does more than add", l) for l in lines)
  # no changed files: quick
  @test pr().tier == "quick"
  # the plan of a pull request needs its files
  @test_throws ErrorException decide("pull_request")
  # the decision lines name the platforms and the rule of each path
  _, lines = decide("pull_request"; paths = ["src/utils.jl"])
  @test any(occursin("Platforms: **linux**", l) for l in lines) && any(occursin("`src/utils.jl` | thorough", l) for l in lines)
end

@testset "pull request: benchmark reference checks" begin
  # the harness paths, with or without other files, and the workflow file that defines the jobs
  r = pr("benchmark/cases.jl")
  @test r.benchmark && r.tier == "none" && !r.run_tests
  @test pr("benchmark/Project.toml").benchmark
  @test pr("docs/src/index.md", "benchmark/check_only.jl").benchmark
  r = pr(".github/workflows/CI.yml")
  @test r.benchmark && r.tier == "full" && r.platforms == "linux"
  # other changes, package tests included, start none
  for path in ("src/sparse_iterate.jl", "src/utils.jl", "benchmark/README.md", "benchmark/compare.py", "docs/src/index.md")
    @test !pr(path).benchmark
  end
  # the label test:benchmark adds the checks and nothing else
  r = pr("test/norms.jl"; labels = ["test:benchmark"])
  @test r.benchmark && r.tier == "quick" && r.platforms == "stable" && jobnames(r.required) == ["$L1 - quick"]
  r = pr("docs/src/index.md"; labels = ["test:benchmark"])
  @test r.benchmark && r.tier == "none" && !r.run_tests
end

@testset "pull request: labels" begin
  # a label raises the tier and never lowers it
  @test pr("test/norms.jl"; labels = ["test:standard"]).tier == "standard"
  r = pr("test/norms.jl"; labels = ["test:thorough"])
  @test r.tier == "thorough" && r.platforms == "linux" && !r.benchmark
  @test pr("src/utils.jl"; labels = ["test:standard"]).tier == "thorough"
  @test pr("Project.toml"; labels = ["test:quick", "test:thorough"]).tier == "full"
  @test pr("docs/src/index.md"; labels = ["test:standard"]).tier == "standard"
  # test:full opts into every platform and the benchmark checks; the path rule that needs full alone does not
  r = pr("test/norms.jl"; labels = ["test:full"])
  @test r.tier == "full" && r.platforms == "cross" && r.benchmark && !r.coverage && r.conformance
  @test sort(jobnames(r.required)) == sort(["$LTS - full", "$L1 - full", "$MAC - full", "$WIN - full"])
  @test jobnames(r.advisory) == ["$PRE - full"]
  @test pr("Project.toml").platforms == "linux" && !pr("Project.toml").benchmark
  @test pr("docs/src/index.md"; labels = ["test:full"]).platforms == "cross"
  # labels that are not tiers do not throw and change nothing
  base = pr("src/utils.jl")
  @test pr("src/utils.jl"; labels = ["bug", "test:iterate-narrow"]) == base
  @test pr("src/utils.jl"; labels = ["test:benchmark"]).tier == base.tier
  @test_throws ArgumentError pr("src/utils.jl"; labels = ["test:fast"])
  # the narrow label still lowers an iterate rule: quick on Julia 1
  r = pr("src/atom_iterate.jl"; labels = ["test:iterate-narrow"])
  @test r.tier == "quick" && r.platforms == "stable"
end

@testset "push to master" begin
  push(; kw...) = planned("push"; ref_type = "branch", kw...)
  r = push(paths = ["src/utils.jl"])
  @test r.tier == "thorough" && r.run_tests && r.timeout == 45 && r.platforms == "linux"
  @test !r.coverage && !r.benchmark && r.conformance
  @test sort(jobnames(r.required)) == sort(["$LTS - thorough", "$L1 - thorough"]) && isempty(r.advisory)
  # whatever the paths need, a push runs thorough (labels and inputs mean nothing)
  @test push(paths = ["test/norms.jl"]).tier == "thorough"
  @test push(paths = ["Project.toml"]).tier == "thorough"
  @test push(paths = ["src/utils.jl"], labels = ["test:full"]).platforms == "linux"
  # nothing that needs package tests: none, and the matrix stays valid
  r = push(paths = ["docs/src/index.md", "README_extra.md"])
  @test r.tier == "none" && !r.run_tests && isempty(r.required) && startswith(r.matrix, "{\"include\"")
  # the benchmark harness alone: no tests, the reference checks run
  r = push(paths = ["benchmark/run.jl"])
  @test r.tier == "none" && r.benchmark
  @test push(paths = [".github/workflows/CI.yml"]).benchmark
  @test !push(paths = ["src/utils.jl"]).benchmark
  # the changed files unknown (no previous commit, a zero or an unfetchable one), or an empty list: thorough
  @test push().tier == "thorough" && push(paths = nothing).tier == "thorough"
  @test push(paths = String[]).tier == "thorough"
  # only a branch or a tag
  for ref in ("", "branch ", "tags", "commit")
    @test_throws ErrorException planned("push"; ref_type = ref, paths = ["src/utils.jl"])
  end
  @test_throws ErrorException planned("push")
end

@testset "tag push" begin
  for paths in (nothing, ["docs/src/index.md"], ["src/utils.jl"])
    r = planned("push"; ref_type = "tag", paths)
    @test r.tier == "full" && r.run_tests && r.timeout == 90 && r.platforms == "cross"
    @test r.benchmark && !r.coverage && r.conformance
    @test sort(jobnames(r.required)) == sort(["$LTS - full", "$L1 - full", "$MAC - full", "$WIN - full"])
    @test jobnames(r.advisory) == ["$PRE - full"]
  end
end

@testset "nightly" begin
  r = planned("schedule")
  @test r.tier == "full" && r.run_tests && r.platforms == "cross" && r.coverage && r.benchmark && r.conformance
  @test sort(jobnames(r.required)) == sort(["$LTS - full", "$L1 - full", "$MAC - full", "$WIN - full"])
  @test jobnames(r.advisory) == ["$PRE - full"]
  # always: not skipped for a documentation-only commit, and nothing earlier can skip it
  @test planned("schedule"; paths = ["README.md"]).tier == "full"
  @test planned("schedule"; paths = ["docs/src/index.md"]).run_tests
  @test !isdefined(CIPlan, :recent_full_run)
  @test_throws MethodError CIPlan.decide("schedule"; gh = identity, manifest = MANIFEST, policy = POLICY)
  # the options of the planner no longer include the commit and the repository
  @test_throws ErrorException CIPlan.options(["--event=schedule", "--sha=abc"])
  @test_throws ErrorException CIPlan.options(["--event=schedule", "--repo=o/n"])
  # every entry of the tier lists is selected by the nightly tier
  @test Set(TT.select(MANIFEST, r.tier)) == Set(keys(MANIFEST.tier))
  @test r.timeout == MANIFEST.timeouts["full"]
end

@testset "manual run" begin
  # the defaults: the full tier on every platform, no coverage, the benchmark checks (tier full)
  r = dispatch()
  @test r.tier == "full" && r.platforms == "cross" && !r.coverage && r.benchmark && r.run_tests
  @test r == dispatch(tier = "", platforms = "", benchmark = "", coverage = "")
  # each tier; auto means every platform for full and Julia 1 and lts on Linux otherwise
  for tier in ("quick", "standard", "thorough", "full")
    r = dispatch(tier = tier)
    @test r.tier == tier && r.platforms == (tier == "full" ? "cross" : "linux")
    @test r.timeout == MANIFEST.timeouts[tier] && r.conformance == (tier in ("thorough", "full"))
    @test r.benchmark == (tier == "full")
    @test dispatch(tier = tier, platforms = "auto") == r
  end
  # each platforms value, whatever the tier
  for tier in ("quick", "full"), (platforms, jobs) in (("stable", 1), ("linux", 2), ("cross", 5))
    r = dispatch(; tier, platforms)
    @test r.platforms == platforms && length(jobnames(r.required)) + length(jobnames(r.advisory)) == jobs
  end
  @test jobnames(dispatch(platforms = "stable").required) == ["$L1 - full"]
  # coverage and benchmark inputs
  @test dispatch(coverage = "true").coverage && !dispatch(coverage = "false").coverage
  @test dispatch(tier = "quick", benchmark = "true").benchmark && !dispatch(tier = "quick", benchmark = "false").benchmark
  @test dispatch(tier = "quick", platforms = "stable", coverage = "true", benchmark = "true").platforms == "stable"
  # invalid values are errors
  for (key, value) in ((:tier, "none"), (:tier, "fast"), (:platforms, "all"), (:platforms, "Linux"),
                       (:benchmark, "yes"), (:benchmark, "1"), (:coverage, "True"), (:coverage, "1"))
    @test_throws ErrorException dispatch(; (key => value,)...)
  end
  # a manual run ignores paths and labels
  @test planned("workflow_dispatch"; paths = ["docs/src/index.md"], labels = ["test:standard"]).tier == "full"
end

@testset "unknown events" begin
  @test_throws ErrorException decide("release")
  @test_throws ErrorException decide("")
end

@testset "matrix" begin
  mode(name) = POLICY.platforms[name]
  entries(name) = CIPlan.matrix_entries(mode(name))
  get_(entry, key) = (i = findfirst(p -> p.first == key, entry); i === nothing ? nothing : entry[i].second)
  jsonof(name) = CIPlan.matrix_json(entries(name))
  # exact text per mode: what the workflow reads
  @test jsonof("stable") == """{"include":[{"version":"1","os":"ubuntu-latest","arch":"x64","advisory":false,"map":true,"coverage":true}]}"""
  @test jsonof("linux") ==
        """{"include":[{"version":"lts","os":"ubuntu-latest","arch":"x64","advisory":false,"map":true,"coverage":false},""" *
        """{"version":"1","os":"ubuntu-latest","arch":"x64","advisory":false,"map":false,"coverage":true}]}"""
  @test jsonof("cross") ==
        """{"include":[{"version":"lts","os":"ubuntu-latest","arch":"x64","advisory":false,"map":true,"coverage":false},""" *
        """{"version":"1","os":"ubuntu-latest","arch":"x64","advisory":false,"map":false,"coverage":true},""" *
        """{"version":"1","os":"macos-latest","arch":"aarch64","ptr_jobs":"2","timeout":120,"advisory":false,"map":false,"coverage":false},""" *
        """{"version":"1","os":"windows-latest","arch":"x64","advisory":false,"map":false,"coverage":false},""" *
        """{"version":"pre","os":"ubuntu-latest","arch":"x64","advisory":true,"map":false,"coverage":false}]}"""
  # one map entry and one coverage entry in every mode; the map is Julia lts on Linux when the mode has it
  for name in ("stable", "linux", "cross")
    es = entries(name)
    @test count(e -> get_(e, "map"), es) == 1 && count(e -> get_(e, "coverage"), es) == 1
    m = only(filter(e -> get_(e, "map"), es))
    @test get_(m, "os") == "ubuntu-latest" && get_(m, "version") == (name == "stable" ? "1" : "lts")
    c = only(filter(e -> get_(e, "coverage"), es))
    @test get_(c, "os") == "ubuntu-latest" && get_(c, "version") == "1" && !get_(c, "advisory")
  end
  # Julia pre is the only advisory job; the macOS entry sets the workers and its own time limit
  advisory = [(get_(e, "version"), get_(e, "os")) for e in entries("cross") if get_(e, "advisory")]
  @test advisory == [("pre", "ubuntu-latest")]
  mac = only(filter(e -> get_(e, "os") == "macos-latest", entries("cross")))
  @test get_(mac, "ptr_jobs") == "2" && get_(mac, "timeout") == 120 && get_(mac, "arch") == "aarch64"
  @test all(e -> get_(e, "ptr_jobs") === nothing && get_(e, "timeout") === nothing,
            filter(e -> get_(e, "os") != "macos-latest", entries("cross")))
  # the linux mode is the stable one plus Julia lts; the cross mode adds to linux
  @test Set((j.version, j.os) for j in mode("stable")) < Set((j.version, j.os) for j in mode("linux")) <
        Set((j.version, j.os) for j in mode("cross"))
  # without Julia lts the map entry is Julia 1; without any Linux job there is none
  jobs = [CIPlan.PlatformJob("1", "ubuntu-latest", "x64", nothing, nothing, false)]
  @test get_(only(CIPlan.matrix_entries(jobs)), "map")
  jobs = [CIPlan.PlatformJob("1", "windows-latest", "x64", nothing, nothing, false)]
  @test !get_(only(CIPlan.matrix_entries(jobs)), "map") && !get_(only(CIPlan.matrix_entries(jobs)), "coverage")
  # a plan's output is the same text, for every event
  @test planned("schedule").matrix == jsonof("cross")
  @test planned("push"; ref_type = "branch").matrix == jsonof("linux")
  @test pr("test/norms.jl").matrix == jsonof("stable")
  # the JSON writer
  @test CIPlan.json_string("a\"b\\c\nd\te\x01") == "\"a\\\"b\\\\c\\nd\\te\\u0001\""
  @test CIPlan.json([1, true, "x", [false]]) == "[1,true,\"x\",[false]]"
  @test CIPlan.json(Pair{String, Any}["k" => 2, "l" => Pair{String, Any}[]]) == "{\"k\":2,\"l\":{}}"
end

@testset "the plan lists the jobs the workflow names" begin
  workflow = lf(joinpath(ROOT, ".github", "workflows", "CI.yml"))
  m = match(r"\n  test:\n    name: (.+)\n", workflow)
  @test m !== nothing
  template = m.captures[1]
  @test template == "Julia \${{ matrix.version }} - \${{ matrix.os }} - \${{ matrix.arch }} - \${{ needs.plan.outputs.tier }}"
  # the names of a plan are that template filled with the entries of its matrix
  function filled(r)
    out = String[]
    for mode in (r.platforms,), j in POLICY.platforms[mode]
      push!(out, replace(template, "\${{ matrix.version }}" => j.version, "\${{ matrix.os }}" => j.os,
                         "\${{ matrix.arch }}" => j.arch, "\${{ needs.plan.outputs.tier }}" => r.tier))
    end
    return sort!(out)
  end
  runs = [pr("test/norms.jl"), pr("src/utils.jl"), pr("test/norms.jl"; labels = ["test:full"]),
          planned("push"; ref_type = "branch", paths = ["src/utils.jl"]), planned("push"; ref_type = "tag"),
          planned("schedule"), dispatch(), dispatch(tier = "quick", platforms = "stable")]
  for r in runs
    @test sort([jobnames(r.required); jobnames(r.advisory)]) == filled(r)
  end
  # the workflow reads the plan's outputs under the names the planner writes
  for key in ("tier", "focus", "run_tests", "timeout", "benchmark", "coverage", "conformance", "platforms",
              "matrix", "required", "advisory")
    @test occursin("steps.plan.outputs.$key", workflow)
    @test haskey(planned("schedule"), Symbol(key))
  end
  @test occursin("fromJSON(needs.plan.outputs.matrix)", workflow) && !occursin("FULL_TESTS", workflow)
  @test occursin("--ref-type=", workflow) && !occursin("--sha=\$GITHUB_SHA", workflow)
  @test !occursin("actions: read # nightly", workflow)
  # the matrix flags are the ones the workflow reads
  for flag in ("matrix.advisory", "matrix.map", "matrix.coverage")
    @test occursin(flag, workflow)
  end
end

@testset "the policy file is validated" begin
  text = lf(RULES)
  function loads(edit)
    mktempdir() do dir
      path = joinpath(dir, "rules.toml")
      write(path, edit(text))
      try
        CIPlan.load_policy(path)
        return String[]
      catch err
        err isa TT.ManifestError || rethrow()
        return err.problems
      end
    end
  end
  swap(a, b) = t -> (@assert occursin(a, t); replace(t, a => b; count = 1))
  @test loads(identity) == String[]
  # unknown keys, modes and tiers
  p = loads(swap("[events.tag]\ntier = \"full\"", "[events.tag]\nrank = 3\ntier = \"full\""))
  @test any(occursin("events.tag has an unknown key `rank`", s) for s in p)
  p = loads(swap("[events.tag]\ntier = \"full\"", "[events.tag]\ntier = \"fast\""))
  @test any(occursin("events.tag: unknown tier `fast`", s) for s in p)
  p = loads(swap("[events.tag]\ntier = \"full\"\nplatforms = \"cross\"", "[events.tag]\ntier = \"full\"\nplatforms = \"everything\""))
  @test any(occursin("events.tag: unknown platforms `everything`", s) for s in p)
  p = loads(swap("[platforms.cross]", "[platforms.mac]\njobs = [{ version = \"1\", os = \"macos-latest\", arch = \"aarch64\" }]\n[platforms.cross]"))
  @test any(occursin("unknown platform mode `mac`", s) for s in p)
  p = loads(swap("{ version = \"1\", os = \"windows-latest\", arch = \"x64\" }", "{ version = \"1\", os = \"windows-latest\", arch = \"x64\", cores = 4 }"))
  @test any(occursin("unknown key `cores`", s) for s in p)
  p = loads(swap("timeout = 120", "timeout = 0"))
  @test any(occursin("timeout must be a positive", s) for s in p)
  p = loads(swap("benchmark = \"always\"\n\n[events.schedule]", "benchmark = \"sometimes\"\n\n[events.schedule]"))
  @test any(occursin("unknown benchmark `sometimes`", s) for s in p)
  p = loads(swap("platforms_by_tier = { none = \"stable\", quick = \"stable\"", "platforms_by_tier = { none = \"stable\", quick = \"remote\""))
  @test any(occursin("needs a mode for tier `quick`", s) for s in p)
  p = loads(swap("label_platforms = { \"test:full\" = \"cross\" }", "label_platforms = { \"full\" = \"cross\" }"))
  @test any(occursin("`full` is not a test: label", s) for s in p)
  # a kind of run that is missing, or that takes another's setting
  p = loads(t -> replace(t, "[events.schedule]" => "[events.nightly]"))
  @test any(occursin("unknown event `nightly`", s) for s in p) && any(occursin("[events.schedule] is missing", s) for s in p)
  p = loads(t -> replace(t, "tier = \"thorough\"\nskip_without_tests" => "tier = \"risk\"\nskip_without_tests"))
  @test any(occursin("tier \"risk\" is for pull requests only", s) for s in p)
  p = loads(swap("coverage = true", "coverage = \"input\""))
  @test any(occursin("coverage \"input\" is for manual runs only", s) for s in p)
  # a platform mode without jobs
  p = loads(t -> replace(t, r"\[platforms\.stable\]\njobs = \[\n.*\n\]" => "[platforms.stable]\njobs = []"))
  @test any(occursin("platforms.stable.jobs must be a non-empty list", s) for s in p)
  # the planner reports the problems when it loads the policy
  mktempdir() do dir
    path = joinpath(dir, "rules.toml")
    write(path, replace(text, "[events.tag]\ntier = \"full\"" => "[events.tag]\ntier = \"fast\""))
    @test_throws TT.ManifestError CIPlan.load_policy(path)
  end
end

@testset "options" begin
  o = CIPlan.options(["--event=push", "--ref-type=tag", "--base=abc", "--head=HEAD", "--labels=a,b", "--tier=full",
                      "--platforms=cross", "--benchmark=true", "--coverage=false"])
  @test o["ref-type"] == "tag" && o["platforms"] == "cross" && o["coverage"] == "false"
  @test_throws ErrorException CIPlan.options(["--event"])
  @test_throws ErrorException CIPlan.options(["--bogus=1"])
  # the wrapper without git: events that need no changed files, and a push without a previous commit
  r, _ = CIPlan.plan(Dict("event" => "schedule"))
  @test r.tier == "full" && r.coverage
  r, _ = CIPlan.plan(Dict("event" => "push", "ref-type" => "tag"))
  @test r.tier == "full" && r.platforms == "cross"
  for base in ("", "0" ^ 40)
    r, _ = CIPlan.plan(Dict("event" => "push", "ref-type" => "branch", "base" => base, "head" => "HEAD"))
    @test r.tier == "thorough" && r.platforms == "linux"
  end
  r, _ = CIPlan.plan(Dict("event" => "workflow_dispatch", "tier" => "quick", "platforms" => "stable", "benchmark" => "true"))
  @test r.tier == "quick" && r.platforms == "stable" && r.benchmark
  @test_throws ErrorException CIPlan.plan(Dict("event" => "push", "ref-type" => "branch2"))
  # a previous commit that cannot be fetched: still thorough
  if Sys.which("git") !== nothing
    r, _ = CIPlan.plan(Dict("event" => "push", "ref-type" => "branch", "base" => "f" ^ 40, "head" => "HEAD"))
    @test r.tier == "thorough"
  end
  # the output is written under the names the workflow reads
  mktemp() do path, io
    close(io)
    withenv("GITHUB_OUTPUT" => path, "GITHUB_STEP_SUMMARY" => nothing) do
      redirect_stdout(devnull) do
        CIPlan.main(["--event=schedule"])
      end
    end
    written = Dict{String, String}()
    for l in eachline(path)
      k, v = split(l, '='; limit = 2)
      written[k] = v
    end
    @test sort(collect(keys(written))) == sort(["tier", "focus", "run_tests", "timeout", "benchmark", "coverage",
                                                "conformance", "platforms", "matrix", "required", "advisory"])
    @test written["tier"] == "full" && written["run_tests"] == "true" && written["coverage"] == "true"
    @test written["matrix"] == CIPlan.matrix_json(CIPlan.matrix_entries(POLICY.platforms["cross"]))
  end
end

end
