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
const SHA40 = "0123456789abcdef0123456789abcdef01234567"

# The message of the error a call raises, "" when it raises none.
function message(f)
  try
    f()
    return ""
  catch err
    err isa ErrorException || rethrow()
    return err.msg
  end
end

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
    @test !r.inventory && !r.release && r.sha == ""
  end
  # thorough, full: Julia 1 on Linux, and the conformance reports
  for (paths, tier) in ((["src/utils.jl"], "thorough"), (["test/conformance/kernels.jl"], "thorough"),
                        (["Project.toml"], "full"), (["scripts/ci_plan.jl"], "full"), (["a/new/file.txt"], "full"))
    r = pr(paths...)
    @test r.tier == tier && r.platforms == "linux" && r.conformance
    @test sort(jobnames(r.required)) == sort(["$L1 - $tier"]) && isempty(r.advisory)
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
  # a tier label raises the tier or leaves it alone
  @test pr("test/norms.jl"; labels = ["test:standard"]).tier == "standard"
  r = pr("test/norms.jl"; labels = ["test:thorough"])
  @test r.tier == "thorough" && r.platforms == "linux" && !r.benchmark
  @test pr("src/utils.jl"; labels = ["test:standard"]).tier == "thorough"
  @test pr("Project.toml"; labels = ["test:quick", "test:thorough"]).tier == "full"
  @test pr("docs/src/index.md"; labels = ["test:standard"]).tier == "standard"
  # test:full opts into every platform and the benchmark checks; the path rule that needs full alone does not
  r = pr("test/norms.jl"; labels = ["test:full"])
  @test r.tier == "full" && r.platforms == "cross" && r.benchmark && !r.coverage && r.conformance
  @test sort(jobnames(r.required)) == sort(["$L1 - full", "$MAC - full", "$WIN - full"])
  @test jobnames(r.advisory) == ["$PRE - full"]
  @test pr("Project.toml").platforms == "linux" && !pr("Project.toml").benchmark
  @test pr("docs/src/index.md"; labels = ["test:full"]).platforms == "cross"
  # labels that are not tiers do not throw and change nothing
  base = pr("src/utils.jl")
  @test pr("src/utils.jl"; labels = ["bug", "test:iterate-narrow"]) == base
  @test pr("src/utils.jl"; labels = ["test:benchmark"]).tier == base.tier
  @test_throws ArgumentError pr("src/utils.jl"; labels = ["test:fast"])
  # the narrow label still lowers an iterate rule: quick on Julia 1
  r = pr("src/block_iterate.jl"; labels = ["test:iterate-narrow"])
  @test r.tier == "quick" && r.platforms == "stable"
end

@testset "push to master" begin
  push(; kw...) = planned("push"; ref_type = "branch", kw...)
  r = push(paths = ["src/utils.jl"])
  @test r.tier == "thorough" && r.run_tests && r.timeout == 45 && r.platforms == "linux"
  @test !r.coverage && !r.benchmark && r.conformance
  @test !r.inventory && !r.release && r.sha == ""
  @test sort(jobnames(r.required)) == sort(["$L1 - thorough"]) && isempty(r.advisory)
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
    # every job of a tag run must have run the whole catalog
    @test r.inventory && !r.release && r.sha == ""
    @test sort(jobnames(r.required)) == sort(["$L1 - full", "$MAC - full", "$WIN - full"])
    @test jobnames(r.advisory) == ["$PRE - full"]
  end
end

@testset "nightly" begin
  r = planned("schedule")
  @test r.tier == "full" && r.run_tests && r.platforms == "cross" && r.coverage && r.benchmark && r.conformance
  @test r.inventory && !r.release && r.sha == ""
  @test sort(jobnames(r.required)) == sort(["$L1 - full", "$MAC - full", "$WIN - full"])
  @test jobnames(r.advisory) == ["$PRE - full"]
  # always: not skipped for a documentation-only commit, and nothing earlier can skip it
  @test planned("schedule"; paths = ["README.md"]).tier == "full"
  @test planned("schedule"; paths = ["docs/src/index.md"]).run_tests
  @test !isdefined(CIPlan, :recent_full_run)
  @test_throws MethodError CIPlan.decide("schedule"; gh = identity, manifest = MANIFEST, policy = POLICY)
  # the planner takes no repository, and a nightly takes no commit: only a manual run is planned with its SHA
  @test_throws ErrorException CIPlan.options(["--event=schedule", "--repo=o/n"])
  @test occursin("only a manual run takes one", message(() -> decide("schedule"; sha = SHA40)))
  # every entry of the tier lists is selected by the nightly tier
  @test Set(TT.select(MANIFEST, r.tier)) == Set(keys(MANIFEST.tier))
  @test r.timeout == MANIFEST.timeouts["full"]
end

@testset "manual run" begin
  # the defaults: the full tier on every platform, no coverage, the benchmark checks (tier full)
  r = dispatch()
  @test r.tier == "full" && r.platforms == "cross" && !r.coverage && r.benchmark && r.run_tests
  @test r == dispatch(tier = "", platforms = "", benchmark = "", coverage = "")
  # each tier; auto means every platform for full and Julia 1 on Linux otherwise
  for tier in ("quick", "standard", "thorough", "full")
    r = dispatch(tier = tier)
    @test r.tier == tier && r.platforms == (tier == "full" ? "cross" : "linux")
    @test r.timeout == MANIFEST.timeouts[tier] && r.conformance == (tier in ("thorough", "full"))
    @test r.benchmark == (tier == "full")
    @test dispatch(tier = tier, platforms = "auto") == r
  end
  # each platforms value, whatever the tier
  for tier in ("quick", "full"), (platforms, jobs) in (("stable", 1), ("linux", 1), ("cross", 4))
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
  # a plain manual run is no release run and checks no inventory
  r = dispatch()
  @test !r.release && !r.inventory && r.sha == ""
  # release=false or empty changes nothing; the commit is only passed on
  @test dispatch(release = "false") == r && dispatch(release = "") == r
  @test planned("workflow_dispatch"; sha = SHA40, inputs = Dict("release" => "false")) == merge(r, (; sha = SHA40))
end

@testset "release run" begin
  inputs(; kw...) = Dict{String, String}("release" => "true", "expected_sha" => SHA40,
                                         (String(k) => string(v) for (k, v) in kw)...)
  release(; sha = SHA40, kw...) = decide("workflow_dispatch"; inputs = inputs(; kw...), sha)
  r, lines = release()
  # the full tier on every platform, with coverage and the benchmark checks, and every job's inventory
  @test r.tier == "full" && r.run_tests && r.platforms == "cross" && r.coverage && r.benchmark && r.conformance
  @test r.release && r.inventory && r.sha == SHA40
  @test sort(jobnames(r.required)) == sort(["$L1 - full", "$MAC - full", "$WIN - full"])
  @test jobnames(r.advisory) == ["$PRE - full"] && r.timeout == MANIFEST.timeouts["full"]
  @test r.matrix == CIPlan.matrix_json(CIPlan.matrix_entries(POLICY.platforms["cross"]))
  # the table names the commit and says that the coverage and benchmark inputs mean nothing
  @test any(occursin(SHA40, l) for l in lines) && any(occursin("inputs are ignored", l) for l in lines)
  # the plan is the policy's [events.release]
  ev = POLICY.events["release"]
  @test (ev["tier"], ev["platforms"], ev["coverage"], ev["benchmark"], ev["inventory"]) == ("full", "cross", true, "always", true)
  # the coverage and benchmark inputs are ignored, the tier and platforms inputs may be spelled out
  @test release(coverage = "false", benchmark = "false")[1] == r
  @test release(tier = "full", platforms = "cross")[1] == r && release(tier = "", platforms = "auto")[1] == r
  # paths and labels mean nothing, as for any manual run
  @test decide("workflow_dispatch"; inputs = inputs(), sha = SHA40, paths = ["docs/src/index.md"], labels = ["test:standard"])[1] == r
  # a release run is the full tier on every platform: anything else is an error, not an override
  for tier in ("quick", "standard", "thorough")
    @test occursin("full tier", message(() -> release(tier = tier)))
  end
  for platforms in ("stable", "linux")
    @test occursin("every platform", message(() -> release(platforms = platforms)))
  end
  @test occursin("tier input", message(() -> release(tier = "none"))) || occursin("unknown tier", message(() -> release(tier = "none")))
  @test !isempty(message(() -> release(platforms = "all")))
  # the expected SHA: the full lowercase 40 digits of the commit that is planned
  for bad in ("", SHA40[1:7], SHA40[1:39], SHA40 * "0", uppercase(SHA40), SHA40 * "\n", "g" * SHA40[2:end], " " * SHA40)
    @test occursin("expected_sha", message(() -> release(expected_sha = bad)))
  end
  other = "f" ^ 40
  @test occursin("not the expected commit $SHA40", message(() -> release(sha = other)))
  @test occursin("not known", message(() -> release(sha = "")))
  @test occursin("full lowercase SHA", message(() -> release(sha = uppercase(SHA40))))
  @test !isempty(message(() -> release(release = "yes")))
  # release and expected_sha belong to manual runs
  @test occursin("only for a release run", message(() -> decide("workflow_dispatch"; inputs = Dict("expected_sha" => SHA40), sha = SHA40)))
  @test !isempty(message(() -> decide("workflow_dispatch"; inputs = Dict("expected_sha" => SHA40, "release" => "false"), sha = SHA40)))
  for (event, kw) in (("schedule", (;)), ("push", (; ref_type = "branch", paths = ["src/utils.jl"])), ("push", (; ref_type = "tag")),
                      ("pull_request", (; paths = ["src/utils.jl"])))
    @test occursin("manual runs", message(() -> decide(event; inputs = Dict("release" => "true"), kw...)))
    @test occursin("manual runs", message(() -> decide(event; inputs = Dict("expected_sha" => SHA40), kw...)))
    @test occursin("manual runs", message(() -> decide(event; inputs = inputs(), kw...)))
    # release=false is nothing
    @test decide(event; inputs = Dict("release" => "false"), kw...)[1] == decide(event; kw...)[1]
    # a pull request or push is never planned with a runner-chosen commit
    @test occursin("only a manual run takes one", message(() -> decide(event; sha = SHA40, kw...)))
  end
  # a plan of the pseudo-event is an error: it is a manual run with an input
  @test occursin("not an event", message(() -> decide("release")))
  @test !isempty(message(() -> decide("release"; inputs = inputs(), sha = SHA40)))
  # the commit given to a plan must itself be a full SHA (it is written to the step output)
  for bad in (SHA40[1:7], SHA40 * "\nevil=1", "abc")
    @test occursin("full lowercase SHA", message(() -> decide("workflow_dispatch"; sha = bad)))
  end
end

@testset "unknown events" begin
  @test_throws ErrorException decide("release")
  @test_throws ErrorException decide("")
  @test_throws ErrorException decide("issues")
end

@testset "matrix" begin
  mode(name) = POLICY.platforms[name]
  entries(name) = CIPlan.matrix_entries(mode(name))
  get_(entry, key) = (i = findfirst(p -> p.first == key, entry); i === nothing ? nothing : entry[i].second)
  jsonof(name) = CIPlan.matrix_json(entries(name))
  # exact text per mode: what the workflow reads
  @test jsonof("stable") == """{"include":[{"version":"1","os":"ubuntu-latest","arch":"x64","advisory":false,"map":true,"coverage":true}]}"""
  @test jsonof("linux") ==
        """{"include":[{"version":"1","os":"ubuntu-latest","arch":"x64","advisory":false,"map":true,"coverage":true}]}"""
  @test jsonof("cross") ==
        """{"include":[{"version":"1","os":"ubuntu-latest","arch":"x64","advisory":false,"map":true,"coverage":true},""" *
        """{"version":"1","os":"macos-latest","arch":"aarch64","ptr_jobs":"2","timeout":120,"advisory":false,"map":false,"coverage":false},""" *
        """{"version":"1","os":"windows-latest","arch":"x64","advisory":false,"map":false,"coverage":false},""" *
        """{"version":"pre","os":"ubuntu-latest","arch":"x64","advisory":true,"map":false,"coverage":false}]}"""
  # one map entry and one coverage entry in every mode, both on current Julia 1 Linux
  for name in ("stable", "linux", "cross")
    es = entries(name)
    @test count(e -> get_(e, "map"), es) == 1 && count(e -> get_(e, "coverage"), es) == 1
    m = only(filter(e -> get_(e, "map"), es))
    @test get_(m, "os") == "ubuntu-latest" && get_(m, "version") == "1"
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
  # Stable and linux both test current stable Linux; cross adds macOS, Windows and Julia pre.
  @test Set((j.version, j.os) for j in mode("stable")) == Set((j.version, j.os) for j in mode("linux"))
  @test Set((j.version, j.os) for j in mode("linux")) < Set((j.version, j.os) for j in mode("cross"))
  # Julia 1 on Linux provides the map baseline; without any Linux job there is none
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
              "matrix", "required", "advisory", "release", "inventory", "sha")
    @test occursin("steps.plan.outputs.$key", workflow)
    @test haskey(planned("schedule"), Symbol(key))
  end
  @test occursin("fromJSON(needs.plan.outputs.matrix)", workflow) && !occursin("FULL_TESTS", workflow)
  # the planner is given no runner-chosen commit on a pull request: the SHA goes to it for a manual
  # run only, as the value of an environment variable, and nothing from the event is in the script
  @test occursin("--ref-type=", workflow) && !occursin("--sha=\$GITHUB_SHA", workflow)
  step = let from = findfirst("      - name: Plan the run\n", workflow), to = findfirst("\n  test:\n", workflow)
    workflow[first(from):first(to)]
  end
  @test count("--sha=", step) == 1 && occursin("          SHA: \${{ github.sha }}\n", step)
  branch = match(r"\n +workflow_dispatch\)\n(.*?);;"s, step)
  @test branch !== nothing && occursin("\"--sha=\$SHA\"", branch.captures[1])
  @test occursin("\"--release=\${DISPATCH_RELEASE:-false}\"", branch.captures[1]) && occursin("\"--expected-sha=\$DISPATCH_EXPECTED_SHA\"", branch.captures[1])
  @test occursin("          DISPATCH_RELEASE: \${{ inputs.release }}\n", step) && occursin("          DISPATCH_EXPECTED_SHA: \${{ inputs.expected_sha }}\n", step)
  @test !occursin("\${{", step[last(findfirst("        run: |\n", step)):end])
  # the manual inputs, and a concurrency group that keeps a release run apart from a manual full run
  @test occursin(r"\n      release:\n        description: '[^\n]*'\n        type: boolean\n        default: false\n", workflow)
  @test occursin(r"\n      expected_sha:\n        description: '[^\n]*'\n        type: string\n        default: ''\n", workflow)
  group = match(r"\n  group: (.*)\n", workflow).captures[1]
  @test occursin("inputs.release", group) && occursin("inputs.expected_sha", group)
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
  p = loads(swap("benchmark = \"always\"\ninventory = true\n\n[events.schedule]", "benchmark = \"sometimes\"\ninventory = true\n\n[events.schedule]"))
  @test any(occursin("unknown benchmark `sometimes`", s) for s in p)
  # inventory is a boolean, and the release table must exist
  p = loads(swap("[events.tag]\ntier = \"full\"\nplatforms = \"cross\"\ncoverage = false\nbenchmark = \"always\"\ninventory = true",
                 "[events.tag]\ntier = \"full\"\nplatforms = \"cross\"\ncoverage = false\nbenchmark = \"always\"\ninventory = \"yes\""))
  @test any(occursin("events.tag: inventory must be true or false", s) for s in p)
  p = loads(swap("[events.release]", "[events.releases]"))
  @test any(occursin("[events.release] is missing", s) for s in p) && any(occursin("unknown event `releases`", s) for s in p)
  inrelease(a, b) = t -> (@assert occursin("[events.release]", t); replace(t, Regex("(\\[events\\.release\\]\\n(?:#[^\\n]*\\n)*[^\\[]*?)" * a) => SubstitutionString("\\1" * b)))
  p = loads(inrelease("tier = \"full\"", "tier = \"input\""))
  @test any(occursin("events.release: tier \"input\" is for manual runs only", s) for s in p)
  p = loads(inrelease("coverage = true", "coverage = \"input\""))
  @test any(occursin("events.release: coverage \"input\" is for manual runs only", s) for s in p)
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
                      "--platforms=cross", "--benchmark=true", "--coverage=false", "--release=true",
                      "--expected-sha=$SHA40", "--sha=$SHA40"])
  @test o["ref-type"] == "tag" && o["platforms"] == "cross" && o["coverage"] == "false"
  @test o["release"] == "true" && o["expected-sha"] == SHA40 && o["sha"] == SHA40
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
  # the options of a release run reach the plan
  r, _ = CIPlan.plan(Dict("event" => "workflow_dispatch", "tier" => "full", "platforms" => "auto", "release" => "true",
                          "expected-sha" => SHA40, "sha" => SHA40))
  @test r.release && r.inventory && r.sha == SHA40 && r.platforms == "cross"
  @test_throws ErrorException CIPlan.plan(Dict("event" => "workflow_dispatch", "release" => "true", "expected-sha" => SHA40,
                                               "sha" => "e" ^ 40))
  @test_throws ErrorException CIPlan.plan(Dict("event" => "schedule", "release" => "true", "expected-sha" => SHA40, "sha" => SHA40))
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
                                                "conformance", "platforms", "matrix", "required", "advisory",
                                                "release", "inventory", "sha"])
    @test written["tier"] == "full" && written["run_tests"] == "true" && written["coverage"] == "true"
    @test written["release"] == "false" && written["inventory"] == "true" && written["sha"] == ""
    @test written["matrix"] == CIPlan.matrix_json(CIPlan.matrix_entries(POLICY.platforms["cross"]))
  end
  # a release run writes release=true, inventory=true and its commit
  mktemp() do path, io
    close(io)
    withenv("GITHUB_OUTPUT" => path, "GITHUB_STEP_SUMMARY" => nothing) do
      redirect_stdout(devnull) do
        CIPlan.main(["--event=workflow_dispatch", "--tier=full", "--platforms=auto", "--benchmark=false", "--coverage=false",
                     "--release=true", "--expected-sha=$SHA40", "--sha=$SHA40"])
      end
    end
    written = Dict(split(l, '='; limit = 2)[1] => split(l, '='; limit = 2)[2] for l in eachline(path))
    @test written["release"] == "true" && written["inventory"] == "true" && written["sha"] == SHA40
    @test written["coverage"] == "true" && written["benchmark"] == "true" && written["platforms"] == "cross"
  end
end

const TESTING_CONSUMERS = ["allocations_basics", "ballconstrained", "binary", "composition", "distance",
                           "iterate_writes", "losses", "nonconvex", "norms", "separable", "smooth",
                           "soc", "sqrhinge", "sumlargest"]

@testset "Testing extension plan and consumer inventory" begin
  ext = "ext/ProximalBaseTestExt.jl"
  selected(r) = sort!(union(TT.select(MANIFEST, r.tier), filter(!isempty, split(r.focus, ','))))
  standard = TT.select(MANIFEST, "standard")
  # Keep the documented consumer mapping and its selection guard in agreement.
  section = match(r"(?s)## Testing extension selection\n(.*?)(?:\n## |\z)", lf(joinpath(ROOT, "test", "TESTING.md"))).captures[1]
  rows = filter(line -> startswith(line, "| `"), split(section, '\n'))
  documented = sort!(unique([m.captures[1] for row in rows for m in eachmatch(r"`([^`]+)`", split(row, '|')[4])]))
  @test documented == TESTING_CONSUMERS
  for paths in ([ext], [ext, "test/testing.jl"], [ext, "src/functions/norms.jl", "test/norms.jl"],
                [ext, "test/precompose_array.jl"])
    r = pr(paths...)
    expected = ["testing"]
    "src/functions/norms.jl" in paths && push!(expected, "block_iterate_grids")
    "test/precompose_array.jl" in paths && push!(expected, "precompose_array")
    @test r.tier == "standard" && r.focus == join(sort(expected), ",")
    @test selected(r) == sort!(union(standard, expected))
    @test r.platforms == "stable" && jobnames(r.required) == ["$L1 - standard"]
    @test isempty(r.advisory) && !r.benchmark && !r.conformance && !r.coverage && !r.inventory
    @test issubset(TESTING_CONSUMERS, selected(r))
  end
  # Complete changed-path set of the SparseIterate migration (#835).
  paths835 = [
    "benchmark/cases.jl",
    "benchmark/cases_test.jl",
    "docs/audit/sparse-iterate-06-method-inventory.md",
    "docs/design/iterate-family-0.6-acceptance.md",
    "docs/design/iterate-family-0.6.md",
    "docs/make.jl",
    "docs/src/api/iterates.md",
    "docs/src/developer/index.md",
    "docs/src/extending.md",
    "docs/src/index.md",
    "docs/src/iterates.md",
    "docs/src/migration.md",
    "docs/src/sparse_iterate.md",
    "docs/src/symmetric_sparse_iterate.md",
    "examples/02_lasso_path_coordinate_descent.jl",
    "examples/15_calculus_and_custom_functions.jl",
    "ext/ProximalBaseTestExt.jl",
    "src/deprecated.jl",
    "src/functions/smooth.jl",
    "src/image.jl",
    "src/sparse_iterate.jl",
    "test/allocations.jl",
    "test/binary.jl",
    "test/constrained.jl",
    "test/constrained_basics.jl",
    "test/cubenorm.jl",
    "test/image.jl",
    "test/interface.jl",
    "test/iterate_queries.jl",
    "test/large_inputs.jl",
    "test/losses.jl",
    "test/precompose_array.jl",
    "test/real_iterates.jl",
    "test/separable.jl",
    "test/sparse_iterate.jl",
    "test/sparse_iterate_basics.jl",
    "test/sparse_value_cost.jl",
    "test/spectral_overflow.jl",
    "test/sqrhinge.jl",
    "test/three_factor_product.jl",
  ]
  r = pr(paths835...)
  # If tier placement or family-focus rules change this case, update its expected selection.
  expected = ["allocations", "block_step", "block_step_grids", "conformance/failures_storage",
              "constrained", "image", "large_inputs", "precompose_array", "real_iterates",
              "sparse_iterate", "sparse_iterate_basics_grids", "spectral_overflow", "symmetric_argument",
              "symmetric_mul", "testing", "three_factor_product"]
  @test r.tier == "standard"
  @test selected(r) == sort!(union(standard, expected))
  @test split(r.focus, ',') == sort(setdiff(expected, standard))
  @test r.platforms == "stable" && jobnames(r.required) == ["$L1 - standard"]
  @test isempty(r.advisory) && r.benchmark && !r.conformance && !r.coverage && !r.inventory
  for paths in ([ext, "src/testing.jl"], [ext, "src/utils.jl", "test/utils.jl"])
    r = pr(paths...)
    @test r.tier == "thorough" && r.platforms == "linux" && r.conformance
    @test sort(jobnames(r.required)) == sort(["$L1 - thorough"])
    @test "testing" in selected(r) && !("testing" in split(r.focus, ','))
  end
  for label in ("test:thorough", "test:full")
    r = pr(ext; labels = [label])
    tier = label[6:end]
    @test r.tier == tier && isempty(r.focus) && selected(r) == TT.select(MANIFEST, tier)
    @test r.platforms == (tier == "full" ? "cross" : "linux")
    @test sort(jobnames(r.required)) == sort(tier == "full" ?
      ["$L1 - full", "$MAC - full", "$WIN - full"] : ["$L1 - thorough"])
    @test jobnames(r.advisory) == (tier == "full" ? ["$PRE - full"] : String[])
    @test r.benchmark == (tier == "full") && r.conformance
  end
  for path in ("ext/OtherExt.jl", "ext/ProximalBaseGPUArraysCoreExt.jl", "unknown/path", ".github/tier-rules.toml")
    r = pr(ext, path)
    @test r.tier == "full" && selected(r) == TT.select(MANIFEST, "full")
    @test r.platforms == "linux" && sort(jobnames(r.required)) == sort(["$L1 - full"])
    @test isempty(r.advisory) && r.conformance && !r.benchmark
  end
end

end
