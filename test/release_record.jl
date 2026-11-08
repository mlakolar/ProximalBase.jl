module ReleaseRecordTest

# The release record (scripts/release_record.jl): it says PASSED only for a release run of the exact
# commit whose plan, tests, verdict and gate succeeded, whatever else is listed, and a listing it
# cannot read never gives a PASSED. The record is a pure function of its data; `main` is tested
# with a fake `gh` (functions that return the jobs and artifacts, or fail) and an environment
# dictionary. Nothing here uses the network or the process environment.

using Test

include(joinpath(dirname(@__DIR__), "scripts", "release_record.jl"))
using .ReleaseRecord: build_record, main
const RR = ReleaseRecord
const V = RR.CIVerdict

const SHA = "0123456789abcdef0123456789abcdef01234567"
const OTHER = "fedcba9876543210fedcba9876543210fedcba98"

required() = V.mode_jobs("cross", "full")
const PRE = V.jobname(V.ADVISORY_JOB..., "full")

# The jobs of a release run, each with the attempt 1; `conclusions` replaces some, `drop` removes some.
function jobs_of(; conclusions = Dict{String, String}(), drop = String[], extra = Tuple{String, String, String}[])
  names = ["Test plan"; required(); PRE; "Benchmark reference checks - Julia lts"; "Benchmark analysis controls";
           "CI required jobs"; "CI result"]
  listed = [(n, get(conclusions, n, "success"), "1") for n in names if !(n in drop)]
  return [listed; extra]
end

artifacts_of() = ["conformance-reports-lts-ubuntu-latest-x64-1", "conformance-reports-1-macos-latest-aarch64-1"]

record(; kw...) = build_record(; plan = "success", tests = "success", verdict = "success", gate = "success",
                               expected_sha = SHA, sha = SHA, event = "workflow_dispatch", ref = "refs/heads/release-1",
                               run_url = "https://github.com/owner/repo/actions/runs/42", attempt = "1", workflow_sha = SHA,
                               workflow_differs = "false", version = "0.5.4", jobs = jobs_of(), artifacts = artifacts_of(), kw...)

heading(r) = first(split(r.text, '\n'))

@testset "PASSED needs all four results and the exact commit" begin
  r = record()
  @test r.passed && isempty(r.reasons)
  @test heading(r) == "# Release validation: PASSED"
  @test occursin("`$SHA`", r.text) && occursin("| Run | https://github.com/owner/repo/actions/runs/42 (attempt 1) |", r.text)
  @test occursin("`refs/heads/release-1`", r.text) && occursin("`0.5.4`", r.text)
  @test occursin("| Workflow file commit | `$SHA` |", r.text)
  for key in (:plan, :tests, :verdict, :gate), result in ("failure", "cancelled", "skipped", "bogus")
    r = record(; (key => result,)...)
    @test !r.passed && heading(r) == "# Release validation: FAILED" && !isempty(r.reasons)
    @test !occursin("PASSED", r.text)
  end
  # all four failing at once name all four
  r = record(plan = "failure", tests = "failure", verdict = "failure", gate = "failure")
  @test length(r.reasons) == 4
end

@testset "the commit must be the expected one, in full" begin
  r = record(expected_sha = OTHER)
  @test !r.passed && any(occursin("not the expected commit", x) for x in r.reasons)
  @test heading(r) == "# Release validation: FAILED"
  for bad in (SHA[1:39], SHA[1:7], uppercase(SHA), "", SHA * "0", SHA * "\n", "  $SHA", "x" ^ 40, "abc; rm -rf /")
    r = record(expected_sha = bad)
    @test !r.passed && any(occursin("expected_sha", x) for x in r.reasons)
    @test heading(r) == "# Release validation: FAILED"
  end
  # the commit of the run must itself be a full SHA
  @test !record(sha = "abc", expected_sha = "abc").passed
  @test !record(sha = uppercase(SHA), expected_sha = uppercase(SHA)).passed
  # only a manual run can be a release run
  for event in ("schedule", "push", "pull_request", "")
    r = record(; event)
    @test !r.passed && any(occursin("not a manual release run", x) for x in r.reasons)
  end
end

@testset "the jobs and the artifacts must be listed" begin
  r = record(jobs = nothing)
  @test !r.passed && heading(r) == "# Release validation: FAILED"
  @test occursin("The jobs of the run could not be listed.", r.text)
  r = record(artifacts = nothing)
  @test !r.passed && occursin("The artifacts of the run could not be listed.", r.text)
  @test heading(r) == "# Release validation: FAILED"
  r = record(jobs = nothing, artifacts = nothing)
  @test !r.passed && length(r.reasons) == 2
  # an empty list of artifacts is readable: the record says the run has none
  r = record(artifacts = String[])
  @test r.passed && occursin("The run has no artifacts.", r.text)
end

@testset "required and advisory jobs" begin
  r = record()
  for name in required()
    @test occursin("| $name | required | success | 1 |", r.text)
  end
  @test occursin("| $PRE | advisory | success | 1 |", r.text)
  @test occursin("| Test plan |  | success | 1 |", r.text)
  # Julia pre failing, cancelled, skipped or missing is listed and does not change the heading
  for c in ("failure", "cancelled", "skipped")
    r = record(jobs = jobs_of(conclusions = Dict(PRE => c)))
    @test r.passed && heading(r) == "# Release validation: PASSED"
    @test occursin("| $PRE | advisory | $c | 1 |", r.text)
  end
  @test record(jobs = jobs_of(drop = [PRE])).passed
  # a required job that failed, is missing, is listed twice or has not finished fails the record,
  # even when the four results say success
  for name in required()
    for c in ("failure", "cancelled", "skipped", "")
      r = record(jobs = jobs_of(conclusions = Dict(name => c)))
      @test !r.passed && any(startswith(x, "$name:") for x in r.reasons)
    end
    r = record(jobs = jobs_of(drop = [name]))
    @test !r.passed && any(occursin("$name: missing", x) for x in r.reasons)
    r = record(jobs = jobs_of(extra = [(name, "success", "1")]))
    @test !r.passed && any(occursin("listed 2 times", x) for x in r.reasons)
  end
  # this job is in progress while it writes the record
  r = record(jobs = jobs_of(extra = [("Release record", "", "1")]))
  @test r.passed && occursin("| Release record |  | in progress (this job) | 1 |", r.text)
  r = record(jobs = jobs_of(extra = [("Other job", "", "2")]))
  @test occursin("| Other job |  | not finished | 2 |", r.text)
end

@testset "whether the workflow differs from the default branch" begin
  scope = "The CI workflow, its rules, the check scripts and the test selection files"
  @test occursin("$scope are the same as on the default branch", record(workflow_differs = "false").text)
  @test occursin("The CI workflow, its rules, the check scripts or the test selection files differ from the default branch", record(workflow_differs = "true").text)
  @test occursin("Whether $(lowercasefirst(scope)) differ from the default branch is unknown", record(workflow_differs = "unknown").text)
  @test occursin("is unknown", record(workflow_differs = "unknown").text)
  @test occursin("is unknown", record(workflow_differs = "").text)
  @test occursin("is unknown", record(workflow_differs = "maybe").text)
  # a workflow that differs is stated and does not change the heading
  @test record(workflow_differs = "true").passed
  # the same text for the result of the three cases
  @test count("default branch", record(workflow_differs = "true").text) == 1
end

@testset "the record ends with the maintainer's section" begin
  for r in (record(), record(gate = "failure"))
    lines = split(r.text, '\n')
    @test lines[end] == "" && lines[end-1] == ""
    @test lines[end-2] == "## Maintainer decisions"
    @test RR.EXCEPTIONS_LINE == "Known exceptions are recorded by the maintainer below; the workflow has none."
    @test lines[end-4] == RR.EXCEPTIONS_LINE
    @test count("Maintainer decisions", r.text) == 1
  end
  @test occursin("## Results", record().text) && occursin("- Result gate: success", record().text)
  @test occursin("- `conformance-reports-1-macos-latest-aarch64-1`", record().text)
end

@testset "text from an input is reduced to plain characters" begin
  r = record(expected_sha = "@team ![x](http://e.com) `|`\n# H", ref = "refs/heads/a`b|c\n@d", jobs = jobs_of(extra = [("A | B\n@x", "success", "1")]),
             artifacts = ["x`y|z"], workflow_sha = "`\n@z", version = "1`.0")
  @test !r.passed
  @test !occursin("@", r.text) && !occursin("![", r.text) && !occursin("\n# H", r.text)
  @test !occursin("a`b", r.text) && !occursin("x`y", r.text)
  # one backtick pair per code span: no span can be closed by an input
  for line in split(r.text, '\n')
    startswith(line, "| Ref ") && @test count('`', line) == 2
  end
  # a result that is not one of GitHub's is shown as unknown
  @test occursin("- Test plan: unknown", record(plan = "[x](http://e.com)").text)
end

@testset "main" begin
  env(; kw...) = merge(Dict("GITHUB_SHA" => SHA, "GITHUB_REPOSITORY" => "owner/repo", "GITHUB_RUN_ID" => "42", "GITHUB_RUN_ATTEMPT" => "1",
                            "GITHUB_EVENT_NAME" => "workflow_dispatch", "GITHUB_REF" => "refs/heads/release-1",
                            "PLAN" => "success", "TESTS" => "success", "VERDICT" => "success", "GATE" => "success",
                            "EXPECTED_SHA" => SHA, "WORKFLOW_DIFFERS" => "false", "WORKFLOW_SHA" => SHA),
                       Dict{String, String}(String(k) => v for (k, v) in kw))
  quiet = (a...) -> nothing
  project = joinpath(dirname(@__DIR__), "Project.toml")
  version = RR.project_version(project)
  @test version != "unknown"
  mktempdir() do dir
    out = joinpath(dir, "sub", "release-record.md")
    summary = joinpath(dir, "summary.md")
    run_main(e; jobs = () -> jobs_of(), artifacts = () -> artifacts_of(), path = out) =
      main([path], e; fetch_jobs = jobs, fetch_artifacts = artifacts, project, sleep = _ -> nothing, log = quiet, err = devnull)
    # a passed run: the record is written, appended to the summary, exit 0
    @test run_main(merge(env(), Dict("GITHUB_STEP_SUMMARY" => summary))) == 0
    text = read(out, String)
    @test startswith(text, "# Release validation: PASSED\n") && occursin("`$version`", text)
    @test occursin("https://github.com/owner/repo/actions/runs/42", text)
    @test read(summary, String) == text
    # a second record appends to the summary and replaces the file
    @test run_main(merge(env(GATE = "failure"), Dict("GITHUB_STEP_SUMMARY" => summary))) == 0
    @test startswith(read(out, String), "# Release validation: FAILED\n")
    @test startswith(read(summary, String), text) && count("# Release validation:", read(summary, String)) == 2
    # a FAILED record is written like a PASSED one, with exit 0, for each failing result, a wrong or malformed expected SHA and an unreadable listing
    # (an invalid validation input is not an invalid operational variable, which is status 2 below)
    # a non-empty but unrecognized result or GITHUB_SHA is a FAILED record (exit 0); an empty one is status 2 (below)
    for kw in (Dict("PLAN" => "bogus"), Dict("GATE" => "Success"), Dict("GITHUB_SHA" => "bad", "EXPECTED_SHA" => "bad"), Dict("GITHUB_SHA" => "bad"))
      rm(out; force = true)
      @test run_main(merge(env(), kw)) == 0
      @test startswith(read(out, String), "# Release validation: FAILED\n")
    end
    @test occursin("- Test plan: unknown", (run_main(merge(env(), Dict("PLAN" => "bogus"))); read(out, String)))
    for key in ("PLAN", "GITHUB_SHA")
      rm(out; force = true)
      @test run_main(merge(env(), Dict(key => ""))) == 2 && !isfile(out)
      @test run_main(merge(env(), Dict(key => "  "))) == 2 && !isfile(out)
    end
    for kw in (Dict("PLAN" => "failure"), Dict("TESTS" => "cancelled"), Dict("VERDICT" => "failure"), Dict("GATE" => "skipped"),
               Dict("EXPECTED_SHA" => OTHER), Dict("EXPECTED_SHA" => ""), Dict("EXPECTED_SHA" => SHA[1:7]), Dict("GITHUB_EVENT_NAME" => "schedule"))
      rm(out; force = true)
      @test run_main(merge(env(), kw)) == 0
      @test startswith(read(out, String), "# Release validation: FAILED\n")
    end
    rm(out; force = true)
    @test run_main(env(); jobs = () -> error("HTTP 500")) == 0
    @test startswith(read(out, String), "# Release validation: FAILED\n") && occursin("could not be listed", read(out, String))
    @test run_main(env(); artifacts = () -> error("HTTP 500")) == 0
    @test startswith(read(out, String), "# Release validation: FAILED\n") && occursin("artifacts of the run could not be listed", read(out, String))
    # the listings are retried before the record gives up
    calls = Ref(0)
    flaky() = (calls[] += 1; calls[] < 3 ? error("HTTP 502") : jobs_of())
    @test run_main(env(); jobs = flaky) == 0 && calls[] == 3
    @test startswith(read(out, String), "# Release validation: PASSED\n")
    # an unset or unreadable WORKFLOW_DIFFERS is recorded as unknown
    @test run_main(Dict(k => v for (k, v) in env() if k != "WORKFLOW_DIFFERS")) == 0
    @test occursin("is unknown", read(out, String))
    # the record cannot be written: the output path is a directory, the summary is in a missing directory
    @test run_main(env(); path = dir) == 1
    @test run_main(merge(env(), Dict("GITHUB_STEP_SUMMARY" => joinpath(dir, "missing", "s.md")))) == 1
    # a missing or malformed variable, or a wrong command line: status 2, and nothing is listed
    listed = Ref(0)
    count_jobs() = (listed[] += 1; jobs_of())
    for key in ("GITHUB_SHA", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "PLAN", "TESTS", "VERDICT", "GATE")
      e = Dict(k => v for (k, v) in env() if k != key)
      rm(out; force = true)
      @test run_main(e; jobs = count_jobs) == 2 && !isfile(out)
    end
    @test run_main(env(GITHUB_REPOSITORY = "owner repo"); jobs = count_jobs) == 2
    @test run_main(env(GITHUB_RUN_ID = "4x"); jobs = count_jobs) == 2
    @test run_main(env(GITHUB_RUN_ATTEMPT = "x"); jobs = count_jobs) == 2
    @test run_main(env(GITHUB_SERVER_URL = "javascript:alert(1)"); jobs = count_jobs) == 2
    @test listed[] == 0
    @test main(String[], env(); err = devnull) == 2
    @test main(["a", "b"], env(); err = devnull) == 2
    @test main(["--out"], env(); err = devnull) == 2
  end
  # the version of an unreadable Project.toml is unknown, and the record still stands
  @test RR.project_version(joinpath(tempdir(), "no-such-dir", "Project.toml")) == "unknown"
end

end
