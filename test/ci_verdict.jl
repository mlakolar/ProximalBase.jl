module CIVerdictTest

# The decisions of the CI verdict job (scripts/ci_verdict.jl): it must not pass when a job the
# plan asked for is missing, failed, was cancelled or skipped, when another platform ran, when the
# plan is incomplete, or when the jobs of the run cannot be listed.

using Test

include(joinpath(dirname(@__DIR__), "scripts", "ci_verdict.jl"))
using .CIVerdict: verdict, minimum_jobs, floor_jobs, mode_jobs, list_jobs, parse_jobs

module CIPlan
include(joinpath(dirname(@__DIR__), "scripts", "ci_plan.jl"))
end

const ROOT = dirname(@__DIR__)
const TEMPLATE = "Julia \${{ matrix.version }} - \${{ matrix.os }} - \${{ matrix.arch }} - \${{ needs.plan.outputs.tier }}"
const L1 = "Julia 1 - ubuntu-latest - x64"
const LTS = "Julia lts - ubuntu-latest - x64"
const MAC = "Julia 1 - macos-latest - aarch64"
const WIN = "Julia 1 - windows-latest - x64"
const PRE = "Julia pre - ubuntu-latest - x64"

cross(tier) = ["$LTS - $tier", "$L1 - $tier", "$MAC - $tier", "$WIN - $tier"]

# The platform mode the planner gives a run of this kind.
default_platforms(event, ref_type, tier) =
  event == "pull_request" ? (tier in ("thorough", "full") ? "linux" : "stable") :
  event == "workflow_dispatch" ? "stable" : event == "push" && ref_type == "branch" ? "linux" : "cross"

# The decision for a run whose plan lists `required` and `advisory` jobs. The jobs of the run
# default to every listed job with the conclusion success; `conclusions` replaces some, `drop`
# removes some, `extra` adds jobs.
function go(; event = "pull_request", ref_type = "", tier = "quick", required = nothing, advisory = String[],
             conclusions = Dict{String, String}(), drop = String[], extra = Pair{String, String}[],
             plan = "success", tests = tier == "none" ? "skipped" : "success", analysis = "success",
             benchmark = "skipped", run_benchmark = "false", run_tests = tier == "none" ? "false" : "true",
             jobs = :default, req_text = nothing, adv_text = nothing, platforms = nothing)
  required = required === nothing ? (tier == "none" ? String[] : ["$L1 - $tier"]) : required
  listed = jobs === :default ?
           Tuple{String, String}[(n, get(conclusions, n, "success")) for n in [required; advisory] if !(n in drop)] : jobs
  jobs === :default && append!(listed, [(first(p), last(p)) for p in extra])
  return verdict(; event, ref_type, plan, tests, analysis, benchmark, run_benchmark, run_tests, tier,
                 platforms = platforms === nothing ? default_platforms(event, ref_type, tier) : platforms,
                 required = req_text === nothing ? join(required, ";") : req_text,
                 advisory = adv_text === nothing ? join(advisory, ";") : adv_text, jobs = listed)
end
passes(; kw...) = isempty(go(; kw...).errors)

@testset "a run that passes" begin
  @test passes()
  @test passes(tier = "standard")
  @test passes(tier = "thorough", required = ["$LTS - thorough", "$L1 - thorough"])
  @test passes(event = "push", ref_type = "branch", tier = "thorough", required = ["$LTS - thorough", "$L1 - thorough"])
  @test passes(event = "push", ref_type = "tag", tier = "full", required = cross("full"), advisory = ["$PRE - full"],
               benchmark = "success", run_benchmark = "true")
  @test passes(event = "schedule", tier = "full", required = cross("full"), advisory = ["$PRE - full"])
  @test passes(event = "workflow_dispatch", tier = "quick")
  # pull requests may run more than the minimum
  @test passes(tier = "full", required = cross("full"), advisory = ["$PRE - full"])
  r = go()
  @test isempty(r.errors) && isempty(r.failed) && !r.linux_failed
end

@testset "the plan and the verdict agree on every kind of run" begin
  manifest = CIPlan.default_manifest()
  policy = CIPlan.load_policy()
  plans = [("pull_request", "", CIPlan.decide("pull_request"; paths = ["test/norms.jl"], manifest, policy)[1]),
           ("pull_request", "", CIPlan.decide("pull_request"; paths = ["src/utils.jl"], manifest, policy)[1]),
           ("pull_request", "", CIPlan.decide("pull_request"; paths = ["test/norms.jl"], labels = ["test:full"], manifest, policy)[1]),
           ("push", "branch", CIPlan.decide("push"; ref_type = "branch", paths = ["src/utils.jl"], manifest, policy)[1]),
           ("push", "tag", CIPlan.decide("push"; ref_type = "tag", manifest, policy)[1]),
           ("schedule", "", CIPlan.decide("schedule"; manifest, policy)[1]),
           ("workflow_dispatch", "", CIPlan.decide("workflow_dispatch"; manifest, policy)[1]),
           ("workflow_dispatch", "", CIPlan.decide("workflow_dispatch"; inputs = Dict("tier" => "quick", "platforms" => "stable"), manifest, policy)[1])]
  for (event, ref_type, r) in plans
    required = String.(split(r.required, ';'))
    advisory = isempty(r.advisory) ? String[] : String.(split(r.advisory, ';'))
    jobs = [(n, "success") for n in [required; advisory]]
    out = verdict(; event, ref_type, plan = "success", tests = "success", analysis = "success",
                  benchmark = r.benchmark ? "success" : "skipped", run_benchmark = string(r.benchmark),
                  run_tests = string(r.run_tests), tier = r.tier, platforms = r.platforms, required = r.required, advisory = r.advisory, jobs)
    @test isempty(out.errors)
    # and the minimum of the event is inside what the plan lists
    @test issubset(minimum_jobs(event, ref_type, r.tier, r.platforms), required)
  end
  # documentation only: no jobs but the skipped template
  r = CIPlan.decide("pull_request"; paths = ["docs/src/index.md"], manifest, policy)[1]
  out = verdict(; event = "pull_request", plan = "success", tests = "skipped", analysis = "success", benchmark = "skipped",
                run_benchmark = string(r.benchmark), run_tests = string(r.run_tests), tier = r.tier, platforms = r.platforms, required = r.required, advisory = r.advisory,
                jobs = [(TEMPLATE, "skipped")])
  @test isempty(out.errors)
end

@testset "a required job that is missing or did not succeed" begin
  req = ["$LTS - thorough", "$L1 - thorough"]
  for conclusion in ("failure", "cancelled", "skipped", "timed_out", "neutral", "action_required", "stale", "")
    r = go(tier = "thorough", required = req, conclusions = Dict(req[1] => conclusion))
    @test !isempty(r.errors) && r.failed == [req[1]] && r.linux_failed
    @test !passes(tier = "thorough", required = req, conclusions = Dict(req[2] => conclusion))
  end
  r = go(tier = "thorough", required = req, drop = [req[2]])
  @test r.failed == [req[2]] && any(occursin("missing", e) for e in r.errors)
  # the listing of another tier's name does not stand in for it
  @test !passes(tier = "thorough", required = req, drop = [req[2]], extra = ["$L1 - quick" => "success"])
  # a name listed twice (two jobs of one name) is not one job that succeeded
  @test !passes(tier = "thorough", required = req, extra = [req[1] => "success"])
  # the test jobs that were cancelled or skipped as a whole
  for t in ("cancelled", "skipped")
    @test !passes(tests = t)
  end
end

@testset "another platform than the plan named" begin
  @test !passes(extra = ["Julia 1 - macos-latest - aarch64 - quick" => "success"])
  @test !passes(extra = ["Julia lts - ubuntu-latest - x64 - quick" => "failure"])
  @test !passes(extra = [TEMPLATE => "skipped"]) # a template is listed only when no tests run
  # the plan names Linux Julia 1 only, but a job on Windows ran in its place
  @test !passes(required = ["$WIN - quick"])
  # a job that is not a test job does not matter
  @test passes(extra = ["Benchmark analysis controls" => "success", "PR required jobs" => ""])
end

@testset "advisory jobs are reported and do not change the result" begin
  adv = ["$PRE - full"]
  for conclusion in ("failure", "cancelled", "skipped", "timed_out")
    r = go(event = "schedule", tier = "full", required = cross("full"), advisory = adv, conclusions = Dict(adv[1] => conclusion))
    @test isempty(r.errors) && isempty(r.failed)
    @test any(occursin("Advisory: $(adv[1]): $conclusion", n) for n in r.notes)
  end
  r = go(event = "schedule", tier = "full", required = cross("full"), advisory = adv, drop = adv)
  @test isempty(r.errors) && any(occursin("missing", n) for n in r.notes)
  # a name in both lists fails
  r = go(tier = "thorough", required = ["$LTS - thorough", "$L1 - thorough"], advisory = ["$LTS - thorough"])
  @test any(occursin("both required and advisory", e) for e in r.errors)
  # an advisory job is not a platform the plan forgot: it is listed in the plan
  @test passes(tier = "quick", advisory = ["$PRE - quick"], conclusions = Dict("$PRE - quick" => "failure"))
end

@testset "tier none: every Julia job must have been skipped" begin
  none(js...) = go(tier = "none", jobs = [("Test plan", "success"), js...])
  @test isempty(none((TEMPLATE, "skipped")).errors)
  @test isempty(go(tier = "none", jobs = Tuple{String, String}[]).errors)
  @test isempty(go(tier = "none", jobs = [("PR merge gate", ""), ("Benchmark analysis controls", "success")]).errors)
  for c in ("success", "failure", "cancelled", "")
    r = none((TEMPLATE, c))
    @test !isempty(r.errors) && r.failed == [TEMPLATE]
  end
  @test !isempty(none(("$L1 - quick", "success")).errors)
  @test isempty(none(("$L1 - quick", "skipped")).errors)
  # the plan says none but also asks for tests, or lists jobs, or the test jobs ran
  @test !passes(tier = "none", run_tests = "true")
  @test !passes(tier = "none", tests = "success")
  @test !passes(tier = "none", required = ["$L1 - quick"], jobs = Tuple{String, String}[])
  # the jobs cannot be listed: none is not confirmed
  @test !isempty(verdict(; event = "pull_request", plan = "success", tests = "skipped", analysis = "success",
                         benchmark = "skipped", run_benchmark = "false", run_tests = "false", tier = "none",
                         platforms = "stable", required = "", advisory = "", jobs = nothing).errors)
end

@testset "an empty, truncated or inconsistent plan" begin
  # an empty or unknown tier
  for tier in ("", "bogus", "Quick", "all")
    @test !passes(tier = tier, run_tests = "true", req_text = "$L1 - $tier", jobs = [("$L1 - $tier", "success")])
  end
  # tests run, but the plan lists no required job (or only separators)
  for text in ("", ";", " ; ")
    @test !passes(req_text = text)
  end
  # run_tests missing or false while the tier asks for tests
  for v in ("", "false", "maybe")
    @test !passes(run_tests = v)
  end
  # the benchmark flag missing
  @test !passes(run_benchmark = "")
  @test !passes(run_benchmark = "yes")
  # the plan lists a job whose name has another tier
  @test !passes(tier = "thorough", required = ["$LTS - quick", "$L1 - quick"])
  # a plan that failed: nothing else is believed, and outputs are empty
  r = verdict(; event = "pull_request", plan = "failure", tests = "skipped", analysis = "success", benchmark = "skipped",
              run_benchmark = "", run_tests = "", tier = "", platforms = "", required = "", advisory = "", jobs = nothing)
  @test !isempty(r.errors) && "Test plan" in r.failed && r.linux_failed
  @test !passes(plan = "cancelled") && !passes(plan = "skipped")
end

@testset "the minimum of each kind of run, whatever the plan lists" begin
  # a planner that drops a platform from the matrix and the list together
  @test !passes(tier = "thorough", event = "push", ref_type = "branch", required = ["$L1 - thorough"])
  @test !passes(tier = "thorough", event = "push", ref_type = "branch", required = ["$LTS - thorough"])
  @test !passes(tier = "full", event = "push", ref_type = "tag", required = ["$LTS - full", "$L1 - full"])
  @test !passes(tier = "full", event = "schedule", required = ["$LTS - full", "$L1 - full", "$MAC - full"])
  @test !passes(tier = "full", event = "schedule", required = ["$L1 - full", "$MAC - full", "$WIN - full"])
  @test !passes(tier = "quick", required = ["$LTS - quick"])
  @test !passes(tier = "quick", event = "workflow_dispatch", required = ["$LTS - quick"])
  # macOS or Windows as advisory does not meet the minimum of the nightly run
  @test !passes(tier = "full", event = "schedule", required = ["$LTS - full", "$L1 - full"], advisory = ["$MAC - full", "$WIN - full"])
  # a push needs a ref type; so does nothing else
  @test !passes(event = "push")
  @test !passes(event = "push", ref_type = "commit")
  @test !passes(event = "release")
  @test floor_jobs("pull_request", "", "quick") == ["$L1 - quick"]
  @test floor_jobs("pull_request", "", "standard") == ["$L1 - standard"]
  @test sort(floor_jobs("pull_request", "", "thorough")) == sort(["$LTS - thorough", "$L1 - thorough"])
  @test sort(floor_jobs("pull_request", "", "full")) == sort(["$LTS - full", "$L1 - full"])
  @test floor_jobs("workflow_dispatch", "", "full") == ["$L1 - full"]
  @test sort(floor_jobs("push", "branch", "thorough")) == sort(["$LTS - thorough", "$L1 - thorough"])
  @test sort(floor_jobs("push", "tag", "full")) == sort(cross("full"))
  @test sort(floor_jobs("schedule", "", "full")) == sort(cross("full"))
  @test_throws ErrorException floor_jobs("push", "", "full")
  @test_throws ErrorException floor_jobs("issues", "", "full")
  # the minimum adds the jobs of the platform mode to the floor
  @test minimum_jobs("pull_request", "", "quick", "stable") == ["$L1 - quick"]
  @test sort(minimum_jobs("pull_request", "", "quick", "linux")) == sort(["$LTS - quick", "$L1 - quick"])
  @test sort(minimum_jobs("pull_request", "", "full", "cross")) == sort(cross("full"))
  @test sort(minimum_jobs("workflow_dispatch", "", "thorough", "cross")) == sort(cross("thorough"))
  @test sort(minimum_jobs("push", "branch", "thorough", "stable")) == sort(["$LTS - thorough", "$L1 - thorough"])
  @test_throws ErrorException minimum_jobs("pull_request", "", "quick", "")
  @test_throws ErrorException minimum_jobs("pull_request", "", "quick", "all")
end

@testset "the minimum does not depend on the plan's own lists" begin
  thorough = ["$LTS - thorough", "$L1 - thorough"]
  for tier in ("thorough", "full")
    both = ["$LTS - $tier", "$L1 - $tier"]
    @test passes(tier = tier, required = both, platforms = "linux")
    # lts moved into the advisory list: the list is wrong and the minimum is missing
    r = go(tier = tier, required = ["$L1 - $tier"], advisory = ["$LTS - $tier"])
    @test any(occursin("only `$PRE - $tier` may be", e) for e in r.errors) && any(occursin("do not include `$LTS - $tier`", e) for e in r.errors)
    # lts dropped from the required list, and from the matrix and the list together
    @test !passes(tier = tier, required = ["$L1 - $tier"], platforms = "linux")
    @test !passes(tier = tier, required = ["$L1 - $tier"], platforms = "stable")
    @test !passes(tier = tier, required = ["$L1 - $tier"], drop = ["$LTS - $tier"])
  end
  # a stricter mode than the floor: test:full asks for the cross jobs, and they must stay
  for dropped in (["$MAC - full"], ["$WIN - full"], ["$MAC - full", "$WIN - full"])
    kept = setdiff(cross("full"), dropped)
    @test !passes(tier = "full", required = kept, advisory = ["$PRE - full"], platforms = "cross")
    @test !passes(tier = "full", required = kept, advisory = [["$PRE - full"]; dropped], platforms = "cross")
    @test passes(tier = "full", required = kept, advisory = ["$PRE - full"], platforms = "linux")
  end
  @test passes(tier = "full", required = cross("full"), advisory = ["$PRE - full"], platforms = "cross")
  @test !passes(tier = "thorough", required = thorough, platforms = "cross")
  @test !passes(tier = "quick", required = ["$L1 - quick"], platforms = "linux")
  # the advisory list may hold Julia pre on Ubuntu at the plan's tier, and nothing else
  @test passes(advisory = ["$PRE - quick"])
  for name in ("$LTS - quick", "$L1 - quick", "$MAC - quick", "$WIN - quick", "$PRE - full", "Julia nightly - ubuntu-latest - x64 - quick",
               "Julia pre - macos-latest - aarch64 - quick", "something else")
    r = go(advisory = [name])
    @test any(occursin("is listed as advisory", e) for e in r.errors)
  end
  @test !passes(advisory = ["$PRE - quick", "$MAC - quick"])
  # the platforms are an output of the plan that must be one of the three modes
  for value in ("", "Linux", "all", "stable;linux", "auto")
    r = go(platforms = value)
    @test any(occursin("unknown platforms", e) for e in r.errors)
  end
  # tier none needs no platforms, and no jobs
  @test passes(tier = "none", platforms = "")
  @test !passes(tier = "none", advisory = ["$PRE - none"], jobs = Tuple{String, String}[])
  # tier none is planned for a pull request and a push to master only
  @test passes(tier = "none")
  @test passes(tier = "none", event = "push", ref_type = "branch")
  for (event, ref_type) in (("schedule", ""), ("push", "tag"), ("workflow_dispatch", ""))
    r = go(tier = "none", event = event, ref_type = ref_type, platforms = "linux")
    @test any(occursin("Tier none is planned for a pull request or a push to master only", e) for e in r.errors)
  end
  # every kind of run and mode with the jobs the plan lists passes
  for (event, ref_type, tier, platforms, required, advisory) in (
      ("pull_request", "", "quick", "stable", ["$L1 - quick"], String[]),
      ("pull_request", "", "standard", "stable", ["$L1 - standard"], String[]),
      ("pull_request", "", "thorough", "linux", thorough, String[]),
      ("pull_request", "", "full", "linux", ["$LTS - full", "$L1 - full"], String[]),
      ("pull_request", "", "full", "cross", cross("full"), ["$PRE - full"]),
      ("push", "branch", "thorough", "linux", thorough, String[]),
      ("push", "tag", "full", "cross", cross("full"), ["$PRE - full"]),
      ("schedule", "", "full", "cross", cross("full"), ["$PRE - full"]),
      ("workflow_dispatch", "", "quick", "stable", ["$L1 - quick"], String[]),
      ("workflow_dispatch", "", "thorough", "linux", thorough, String[]),
      ("workflow_dispatch", "", "full", "cross", cross("full"), ["$PRE - full"]),
      ("workflow_dispatch", "", "full", "stable", ["$L1 - full"], String[]))
    @test passes(; event, ref_type, tier, platforms, required, advisory)
  end
end

@testset "the verdict's job tables are the planner's" begin
  policy = CIPlan.load_policy()
  @test sort(collect(keys(CIVerdict.MODE_JOBS))) == sort(collect(CIPlan.MODES))
  for mode in CIPlan.MODES, tier in ("quick", "standard", "thorough", "full")
    jobs = policy.platforms[mode]
    named(js) = sort!([CIPlan.jobname(j, tier) for j in js])
    @test sort(mode_jobs(mode, tier)) == named(filter(j -> !j.advisory, jobs))
    advisory = named(filter(j -> j.advisory, jobs))
    @test advisory == (mode == "cross" ? ["$PRE - $tier"] : String[])
  end
  # the only advisory job of the planner is the verdict's one
  advisory = [j for mode in CIPlan.MODES for j in policy.platforms[mode] if j.advisory]
  @test all(j -> (j.version, j.os, j.arch) == CIVerdict.ADVISORY_JOB, advisory)
  # the planner's required list holds the verdict's minimum for each kind of run
  manifest = CIPlan.default_manifest()
  for (event, ref_type, kw) in (("pull_request", "", (; paths = ["test/norms.jl"])), ("pull_request", "", (; paths = ["src/utils.jl"])),
                                ("pull_request", "", (; paths = ["Project.toml"])),
                                ("pull_request", "", (; paths = ["test/norms.jl"], labels = ["test:full"])),
                                ("push", "branch", (; paths = ["src/utils.jl"])), ("push", "tag", (;)), ("schedule", "", (;)),
                                ("workflow_dispatch", "", (;)),
                                ("workflow_dispatch", "", (; inputs = Dict("tier" => "thorough", "platforms" => "stable"))))
    r = CIPlan.decide(event; ref_type, manifest, policy, kw...)[1]
    required = String.(split(r.required, ';'))
    @test issubset(minimum_jobs(event, ref_type, r.tier, r.platforms), required)
  end
end

@testset "the benchmark reference checks and the controls" begin
  @test passes(benchmark = "success", run_benchmark = "true")
  for r in ("failure", "cancelled", "skipped", "")
    @test !passes(benchmark = r, run_benchmark = "true")
  end
  @test passes(benchmark = "skipped", run_benchmark = "false")
  for r in ("success", "failure", "cancelled")
    @test !passes(benchmark = r, run_benchmark = "false")
  end
  # also when no tests run
  @test passes(tier = "none", benchmark = "success", run_benchmark = "true")
  @test !passes(tier = "none", benchmark = "skipped", run_benchmark = "true")
  # the controls always run
  for r in ("failure", "cancelled", "skipped", "")
    @test !passes(analysis = r)
    @test !passes(tier = "none", analysis = r)
  end
end

@testset "what failed, for the report" begin
  req = cross("full")
  nightly(; kw...) = go(; event = "schedule", tier = "full", required = req, advisory = ["$PRE - full"], kw...)
  # only jobs on other systems: no Linux job failed
  r = nightly(conclusions = Dict("$WIN - full" => "failure", "$MAC - full" => "cancelled"))
  @test r.failed == ["$MAC - full", "$WIN - full"] || r.failed == ["$WIN - full", "$MAC - full"]
  @test !r.linux_failed && length(r.errors) == 2
  # a Linux job, the plan, the controls or a missing minimum: unknown or Linux
  @test nightly(conclusions = Dict("$L1 - full" => "failure")).linux_failed
  @test nightly(conclusions = Dict("$WIN - full" => "failure", "$LTS - full" => "failure")).linux_failed
  @test nightly(analysis = "failure").linux_failed
  @test nightly(analysis = "failure").failed == ["Benchmark analysis controls"]
  @test nightly(tests = "cancelled").linux_failed
  @test nightly(drop = ["$MAC - full"]).failed == ["$MAC - full"] && !nightly(drop = ["$MAC - full"]).linux_failed
  @test nightly(benchmark = "failure", run_benchmark = "true").failed == ["Benchmark reference checks"]
  # no failure: nothing to report
  r = nightly()
  @test isempty(r.failed) && !r.linux_failed
  # the advisory job is not among the failed jobs
  @test isempty(nightly(conclusions = Dict("$PRE - full" => "failure")).failed)
end

@testset "listing the jobs of the run" begin
  @test parse_jobs("Test plan\tsuccess\nJulia 1 - ubuntu-latest - x64 - quick\t\n\n") ==
        [("Test plan", "success"), ("Julia 1 - ubuntu-latest - x64 - quick", "")]
  @test isempty(parse_jobs(""))
  @test_throws ErrorException parse_jobs("Test plan\n")
  @test_throws ErrorException parse_jobs("a\tb\tc\n")
  # failures are retried, and do not become an empty list
  slept = Int[]
  quiet(args...) = nothing
  calls = Ref(0)
  flaky() = (calls[] += 1; calls[] < 3 ? error("HTTP 502") : [("Test plan", "success")])
  @test list_jobs(flaky; sleep = s -> push!(slept, s), log = quiet) == [("Test plan", "success")]
  @test calls[] == 3 && slept == [15, 15]
  calls[] = -100
  empty!(slept)
  @test list_jobs(flaky; sleep = s -> push!(slept, s), log = quiet) === nothing
  @test length(slept) == 3
  @test list_jobs(() -> error("down"); attempts = 2, sleep = _ -> nothing, log = quiet) === nothing
end

@testset "the step: environment, exit status and outputs" begin
  env(; kw...) = Dict{String, String}("EVENT" => "pull_request", "REF_TYPE" => "", "PLAN" => "success", "TESTS" => "success",
                                      "ANALYSIS" => "success", "BENCHMARK" => "skipped", "RUN_BENCHMARK" => "false",
                                      "RUN_TESTS" => "true", "TIER" => "quick", "FOCUS" => "", "PLATFORMS" => "stable", "REQUIRED" => "$L1 - quick",
                                      "ADVISORY" => "", "COMMIT" => "abc", (string(k) => v for (k, v) in kw)...)
  good() = [("Test plan", "success"), ("$L1 - quick", "success"), ("PR merge gate", "")]
  mktemp() do path, io
    close(io)
    run_main(e, fetch) = CIVerdict.main(merge(e, Dict("GITHUB_OUTPUT" => path)); fetch, sleep = _ -> nothing, log = (a...) -> nothing)
    @test run_main(env(), good) == 0
    @test run_main(env(), () -> [("$L1 - quick", "failure")]) == 1
    @test run_main(env(TIER = "none", RUN_TESTS = "false", TESTS = "skipped", REQUIRED = ""), () -> [(TEMPLATE, "skipped")]) == 0
    # the API fails: not a pass, even for a plan that needs no jobs to be present
    @test run_main(env(), () -> error("HTTP 500")) == 1
    @test run_main(env(TIER = "none", RUN_TESTS = "false", TESTS = "skipped", REQUIRED = ""), () -> error("HTTP 500")) == 1
    # a plan that failed needs no listing
    @test run_main(env(PLAN = "failure", TIER = "", RUN_TESTS = "", REQUIRED = ""), () -> error("not called")) == 1
    # a plan without a tier needs none either
    @test run_main(env(TIER = "", RUN_TESTS = "", REQUIRED = ""), () -> error("not called")) == 1
    # an output the step cannot decide on fails
    @test run_main(env(EVENT = "release"), good) == 1
    lines = readlines(path)
    @test lines == ["failed=", "linux_failed=false", "failed=$L1 - quick", "linux_failed=true",
                    "failed=", "linux_failed=false", "failed=", "linux_failed=true", "failed=", "linux_failed=true",
                    "failed=Test plan", "linux_failed=true", "failed=", "linux_failed=true", "failed=", "linux_failed=true"]
  end
  # the nightly run: only Windows failed
  mktemp() do path, io
    close(io)
    e = env(EVENT = "schedule", TIER = "full", PLATFORMS = "cross", REQUIRED = join(cross("full"), ";"), ADVISORY = "$PRE - full")
    jobs() = [(n, n == "$WIN - full" ? "failure" : "success") for n in [cross("full"); "$PRE - full"]]
    @test CIVerdict.main(merge(e, Dict("GITHUB_OUTPUT" => path)); fetch = jobs, sleep = _ -> nothing, log = (a...) -> nothing) == 1
    @test readlines(path) == ["failed=$WIN - full", "linux_failed=false"]
  end
end

@testset "the workflow gives the step what it reads" begin
  workflow = replace(read(joinpath(ROOT, ".github", "workflows", "CI.yml"), String), "\r\n" => "\n")
  start = findfirst("\n  verdict:\n", workflow)
  stop = findfirst("\n  gate:\n", workflow)
  @test start !== nothing && stop !== nothing && start[1] < stop[1]
  segment = workflow[first(start):first(stop)]
  for key in ("EVENT", "REF_TYPE", "PLAN", "TESTS", "ANALYSIS", "BENCHMARK", "RUN_BENCHMARK", "RUN_TESTS", "TIER", "FOCUS",
              "PLATFORMS", "REQUIRED", "ADVISORY", "GH_TOKEN")
    @test occursin("          $key: \${{", segment)
  end
  for key in ("failed", "linux_failed")
    @test occursin("      $key: \${{ steps.verdict.outputs.$key }}", segment)
  end
  @test occursin("scripts/ci_verdict.jl", segment) && occursin("actions: read", segment) && occursin("timeout 120", segment)
  # the gate and the reporter read the outputs and the labels
  @test occursin("needs.verdict.outputs.failed", workflow) && occursin("Nightly CI failing", workflow)
  @test occursin("pull-requests: read", workflow)
end

end
