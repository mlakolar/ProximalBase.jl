using Test

# The decisions of the CI merge gate (scripts/ci_gate.jl) on job states as the GitHub API
# reports them: the gate must never pass on a stale, missing, failed, cancelled or skipped
# verdict, or on API errors. At the end, the conformance report upload result of the test job
# (scripts/ci_upload_result.jl).
include(joinpath(dirname(@__DIR__), "scripts", "ci_gate.jl"))
include(joinpath(dirname(@__DIR__), "scripts", "ci_upload_result.jl"))
using .CIGate: Job, decide, watch, settled, parse_jobs, identity_problems
using .CIUploadResult: upload_result

const V = "PR required jobs"
const G = "PR merge gate"

job(name, status = "completed", conclusion = status == "completed" ? "success" : ""; attempt = 1) =
  Job(name, status, conclusion, attempt)
gate(; attempt = 1) = job(G, "in_progress"; attempt)
plan(args...; kw...) = job("Test plan", args...; kw...)
analysis(args...; kw...) = job("Benchmark analysis controls", args...; kw...)
lts(args...; kw...) = job("Julia lts - ubuntu-latest - x64 - quick", args...; kw...)
jl1(args...; kw...) = job("Julia 1 - ubuntu-latest - x64 - quick", args...; kw...)
verdict(args...; kw...) = job(V, args...; kw...)

d(jobs; attempt = 1, elapsed = 600, settled_for = 0) =
  decide(jobs; attempt, verdict = V, gate = G, elapsed, settled_for)
action(jobs; kw...) = d(jobs; kw...)[1]

@testset "a run: pending until the verdict passes" begin
  # only the gate has appeared, then the plan runs, then the tests
  @test action([gate()]; elapsed = 5) == :wait
  @test action([gate(), plan("in_progress"), analysis("queued")]) == :wait
  @test action([gate(), plan(), analysis(), lts("in_progress"), jl1("queued")]) == :wait
  # a failed test job alone does not decide: the verdict reads every job
  @test action([gate(), plan(), analysis(), lts("completed", "failure"), jl1("in_progress")]) == :wait
  @test action([gate(), plan(), analysis(), lts(), jl1(), verdict("queued")]) == :wait
  @test action([gate(), plan(), analysis(), lts(), jl1(), verdict("in_progress")]) == :wait
  @test d([gate(), plan(), analysis(), lts(), jl1(), verdict()]) == (:pass, "`PR required jobs` passed (attempt 1)")
end

@testset "a verdict that is not success fails the gate" begin
  for c in ("failure", "cancelled", "skipped", "timed_out", "action_required", "neutral", "stale", "")
    @test action([gate(), plan(), analysis(), lts(), jl1(), verdict("completed", c)]) == :fail
  end
  # documentation only: no test jobs, the verdict decides
  @test action([gate(), plan(), analysis(), verdict()]) == :pass
  @test action([gate(), plan("completed", "failure"), analysis(), verdict("completed", "failure")]) == :fail
  # the gate's own job never counts as the verdict
  @test action([job(G, "completed", "success"), plan(), analysis()]; settled_for = 1000) == :fail
end

@testset "a missing verdict fails after the grace period" begin
  jobs = [gate(), plan(), analysis(), lts(), jl1()]
  @test settled(jobs; verdict = V, gate = G)
  @test !settled([gate(), plan(), analysis(), lts(), jl1("in_progress")]; verdict = V, gate = G)
  @test !settled([gate()]; verdict = V, gate = G)
  @test !settled([jobs; verdict("queued")]; verdict = V, gate = G)
  @test action(jobs; settled_for = 30) == :wait
  @test action(jobs; settled_for = 90) == :fail
  # the plan failed and the test jobs were never created
  @test action([gate(), plan("completed", "failure"), analysis()]; settled_for = 120) == :fail
  # two verdicts cannot be told apart
  @test action([gate(), verdict(), verdict()]) == :fail
  # the verdict of another event's run has another name and never counts
  @test action([jobs; job("CI required jobs")]; settled_for = 90) == :fail
end

@testset "re-runs: only the gate's own attempt or a verdict nothing re-runs" begin
  # "Re-run failed jobs" after a failed Julia 1 job: the new test job and gate are attempt 2,
  # the passing lts job stays at attempt 1, and the failed verdict waits for the test job.
  stale = [plan(), analysis(), lts(), verdict("completed", "failure")]
  @test action([stale; jl1("queued"; attempt = 2); gate(attempt = 2)]; attempt = 2, elapsed = 5) == :wait
  @test action([stale; jl1("queued"; attempt = 2); gate(attempt = 2)]; attempt = 2, elapsed = 3000) == :wait
  @test action([plan(), analysis(), lts(), jl1(attempt = 2), verdict("queued"; attempt = 2), gate(attempt = 2)];
               attempt = 2) == :wait
  @test action([plan(), analysis(), lts(), jl1(attempt = 2), verdict(attempt = 2), gate(attempt = 2)];
               attempt = 2) == :pass
  @test action([plan(), analysis(), lts(), jl1("completed", "failure"; attempt = 2),
                verdict("completed", "failure"; attempt = 2), gate(attempt = 2)]; attempt = 2) == :fail
  # "Re-run all jobs" after a passing run: the old success must not pass the new attempt
  rerun_all = [plan("in_progress"; attempt = 2), analysis("queued"; attempt = 2), lts(), jl1(),
               verdict(), gate(attempt = 2)]
  @test action(rerun_all; attempt = 2, elapsed = 5) == :wait
  @test action(rerun_all; attempt = 2, elapsed = 3000) == :wait
  # only the gate re-runs (it had failed on its own): the earlier verdict decides after the grace
  only_gate = [plan(), analysis(), lts(), jl1(), verdict(), gate(attempt = 2)]
  @test action(only_gate; attempt = 2, elapsed = 10) == :wait
  @test action(only_gate; attempt = 2, elapsed = 90) == :pass
  failed = [plan(), analysis(), lts(), jl1("completed", "failure"), verdict("completed", "failure"), gate(attempt = 2)]
  @test action(failed; attempt = 2, elapsed = 90) == :fail
  # as GitHub lists them (PR #445): a new attempt shows the jobs it keeps under its own number
  # and leaves out the jobs it re-runs until they are created
  @test action([plan(attempt = 2), analysis(attempt = 2), lts(attempt = 2), jl1("in_progress"; attempt = 2),
                gate(attempt = 2)]; attempt = 2, elapsed = 5) == :wait
  @test action([plan(attempt = 2), analysis(attempt = 2), lts(attempt = 2), jl1(attempt = 2),
                gate(attempt = 2)]; attempt = 2, settled_for = 30) == :wait
  @test action([plan(attempt = 3), analysis(attempt = 3), lts(attempt = 3), jl1(attempt = 3),
                verdict(attempt = 3), gate(attempt = 3)]; attempt = 3, elapsed = 2) == :pass
  @test action([plan("in_progress"; attempt = 4), analysis(attempt = 4), gate(attempt = 4)];
               attempt = 4, elapsed = 5) == :wait
  # a verdict newer than the gate's attempt
  @test action([gate(), verdict(attempt = 3)]; attempt = 2) == :fail
end

@testset "the run must be this commit's run" begin
  @test isempty(identity_problems("abc", "pull_request", "abc", "pull_request"))
  @test length(identity_problems("abc", "push", "abd", "pull_request")) == 2
  args = ["--repo=o/n", "--run-id=1", "--attempt=1", "--sha=abc", "--event=pull_request", "--gate=$G", "--verdict=$V"]
  @test CIGate.options(args)["verdict"] == V
  @test_throws ErrorException CIGate.options(args[1:end-1]) # the verdict's name is required
end

@testset "job list from the API" begin
  text = "Test plan\tcompleted\tsuccess\t1\nJulia 1 - ubuntu-latest - x64 - quick\tin_progress\t\t2\n\n"
  @test parse_jobs(text) == [Job("Test plan", "completed", "success", 1),
                             Job("Julia 1 - ubuntu-latest - x64 - quick", "in_progress", "", 2)]
  @test isempty(parse_jobs(""))
  @test_throws ErrorException parse_jobs("Test plan\tcompleted\n")
end

# A fake clock: sleep advances it, and fetch returns the states in turn (an exception
# stands for an API error).
function simulate(states; attempt = 1, deadline = 10_000, api_budget = 900)
  clock = Ref(0.0)
  i = Ref(0)
  fetch() = (i[] = min(i[] + 1, length(states)); s = states[i[]]; s isa Exception ? throw(s) : s)
  result = watch(fetch; attempt, verdict = V, gate = G, deadline, api_budget,
                 now = () -> clock[], sleep = s -> (clock[] += s), log = _ -> nothing)
  return result, clock[]
end

@testset "polling" begin
  running = [gate(), plan(), analysis(), lts("in_progress"), jl1("in_progress")]
  passed = [gate(), plan(), analysis(), lts(), jl1(), verdict()]
  (r, m), t = simulate([running, running, passed])
  @test r == :pass
  # API errors in between are retried and never pass on their own
  (r, m), t = simulate([ErrorException("HTTP 502"), running, ErrorException("timeout"), passed])
  @test r == :pass
  (r, m), t = simulate([running, ErrorException("HTTP 502")]; api_budget = 300)
  @test r == :fail && occursin("API failed", m) && t >= 300
  # a failed verdict fails at once
  (r, m), t = simulate([running, [gate(), plan(), analysis(), lts(), jl1("completed", "failure"),
                                  verdict("completed", "failure")]])
  @test r == :fail && occursin("failure", m)
  # a run that never decides fails at the deadline
  (r, m), t = simulate([running]; deadline = 3600)
  @test r == :fail && occursin("no decision", m) && 3600 <= t < 3700
  # every job finished but no verdict appears: fails after the grace period, not at the deadline
  (r, m), t = simulate([[gate(), plan(), analysis(), lts(), jl1()]])
  @test r == :fail && occursin("no job `PR required jobs`", m) && t < 200
  # the fast interval keeps the gate within seconds of the verdict
  states = [fill(running, 30); [[gate(), plan(), analysis(), lts(), jl1(), verdict("queued")]]; [passed]]
  (r, m), t = simulate(states)
  @test r == :pass
end

@testset "conformance report upload result" begin
  r(; required = false, tests = "success", first = "success", first_id = "", retry = "", retry_id = "") =
    upload_result(; required, tests, first, first_id, retry, retry_id, job = "Julia lts, ubuntu-latest")
  # required (Julia lts, Linux): only a step that succeeded and returned an id counts
  @test r(required = true, first_id = "11")[1] == :ok
  @test r(required = true, first = "failure", retry = "success", retry_id = "12")[1] == :notice
  @test r(required = true, first = "failure", retry = "failure")[1] == :error
  @test r(required = true, first = "success", retry = "success")[1] == :error # no id from either
  @test r(required = true, first = "success", retry = "failure")[1] == :error
  @test r(required = true, first = "failure", first_id = "11", retry = "failure", retry_id = "12")[1] == :error
  @test r(required = true, first = "failure", first_id = "11", retry = "success", retry_id = "12")[1] == :notice
  # another job: a warning, the job keeps the result of its tests
  level, line = r(first = "failure", retry = "failure")
  @test level == :warning && occursin("The tests passed", line)
  # the test step failed: no claim that the tests passed, and the failure is named
  level, line = r(tests = "failure", first = "failure", retry = "failure")
  @test level == :warning && !occursin("passed", line) && occursin("test step failed (failure)", line)
  level, line = r(tests = "failure", first_id = "13")
  @test level == :ok && occursin("Diagnostic", line) && occursin("test step failed", line)
  level, line = r(required = true, tests = "failure", first = "failure", retry = "failure")
  @test level == :error && !occursin("passed", line) && occursin("test step failed", line)
  # the step: exit status and the job summary, written also when the step fails
  mktemp() do path, io
    close(io)
    env(; kw...) = Dict("REQUIRED" => "true", "TESTS" => "success", "FIRST" => "success", "FIRST_ID" => "",
                        "RETRY" => "", "RETRY_ID" => "", "JOB" => "Julia lts, ubuntu-latest",
                        "GITHUB_STEP_SUMMARY" => path, (string(k) => v for (k, v) in kw)...)
    quietly(f) = redirect_stdout(f, devnull)
    @test quietly(() -> CIUploadResult.main(env(FIRST_ID = "21"))) == 0
    @test quietly(() -> CIUploadResult.main(env(FIRST = "failure", RETRY = "success", RETRY_ID = "22"))) == 0
    @test quietly(() -> CIUploadResult.main(env(FIRST = "failure", RETRY = "failure"))) == 1
    @test quietly(() -> CIUploadResult.main(env(RETRY = "success"))) == 1 # success, but no id
    @test quietly(() -> CIUploadResult.main(env(REQUIRED = "false", FIRST = "failure", RETRY = "failure"))) == 0
    @test quietly(() -> CIUploadResult.main(env(REQUIRED = "false", TESTS = "failure", FIRST = "failure", RETRY = "failure"))) == 0
    summary = read(path, String)
    @test count("**Conformance reports**", summary) == 6
    @test occursin("artifact 21", summary) && occursin("artifact 22", summary)
    @test count("This artifact is required, so the job fails", summary) == 2
    @test_throws ErrorException quietly(() -> CIUploadResult.main(env(REQUIRED = "")))
  end
end

# Scenario F (PR #430): fails only in attempt 1 of the Julia 1 job.
@testset "scenario: attempt 1 of Julia 1 fails" begin
  @test !(VERSION >= v"1.11" && get(ENV, "GITHUB_RUN_ATTEMPT", "") == "1")
end
