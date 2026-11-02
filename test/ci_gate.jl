using Test

# The decisions of the CI merge gate (scripts/ci_gate.jl) on job states as the GitHub API
# reports them: the gate must never pass on a stale, missing, failed, cancelled or skipped
# verdict, or on API errors.
include(joinpath(dirname(@__DIR__), "scripts", "ci_gate.jl"))
using .CIGate: Job, decide, watch, settled, parse_jobs, identity_problems

const V = "Required jobs"
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
  @test d([gate(), plan(), analysis(), lts(), jl1(), verdict()]) == (:pass, "`Required jobs` passed (attempt 1)")
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
  # a verdict newer than the gate's attempt
  @test action([gate(), verdict(attempt = 3)]; attempt = 2) == :fail
end

@testset "the run must be this commit's run" begin
  @test isempty(identity_problems("abc", "pull_request", "abc", "pull_request"))
  @test length(identity_problems("abc", "push", "abd", "pull_request")) == 2
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
  @test r == :fail && occursin("no job `Required jobs`", m) && t < 200
  # the fast interval keeps the gate within seconds of the verdict
  states = [fill(running, 30); [[gate(), plan(), analysis(), lts(), jl1(), verdict("queued")]]; [passed]]
  (r, m), t = simulate(states)
  @test r == :pass
end
