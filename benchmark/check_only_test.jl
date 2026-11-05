# Fast tests of the check-only driver on synthetic cases, followed by cases_test.jl: the sparse
# operands of the `products` cases and their reference checks for repetitions 1-10.
#
#   julia --project=<benchmark environment> benchmark/check_only_test.jl
#
# CI runs it in the benchmark reference-check job before the quick suite.
include("check_only.jl")
using Test

mutable struct Calls
    prepare::Int
    op::Int
    validate::Int
end

# A synthetic case whose callbacks count their calls; `fail_validate_at = k` throws on the
# k-th call of validate.
function counting_case(id; evals, fail_validate_at = 0)
    n = Calls(0, 0, 0)
    validate() = (n.validate += 1; n.validate == fail_validate_at && error("validate failed at call $(n.validate)"); nothing)
    meta = (; id)
    PerfCase(meta, () -> (n.op += 1; nothing), () -> (n.prepare += 1; nothing), validate, evals, false), n
end

# What `check!` runs for one case (see `measure!` in run.jl): validate, three warm-up rounds and
# twelve allocation probes (prepare and op each), then, for an automatic evaluation count,
# CHECK_SAMPLES samples of prepare and CHECK_EVALS ops and a second validate.
const WARM = 3 + 12

@testset "check!" begin
    @testset "a failing reference check throws before the operation runs" begin
        c, n = counting_case("synthetic/validate_throws"; evals = nothing, fail_validate_at = 1)
        @test_throws ErrorException("validate failed at call 1") check!(c)
        @test (n.validate, n.prepare, n.op) == (1, 0, 0)
    end
    @testset "automatic evaluation count: repeated evaluations, then validate again" begin
        c, n = counting_case("synthetic/auto"; evals = nothing)
        check!(c)
        @test n.validate == 2
        @test n.prepare == WARM + CHECK_SAMPLES
        @test n.op == WARM + CHECK_SAMPLES * CHECK_EVALS
        # State corrupted by the repeated evaluations is caught by the second validate.
        c, n = counting_case("synthetic/auto_post"; evals = nothing, fail_validate_at = 2)
        @test_throws ErrorException("validate failed at call 2") check!(c)
        @test n.op == WARM + CHECK_SAMPLES * CHECK_EVALS
    end
    @testset "fixed evaluation count: validated once, as measure! does" begin
        c, n = counting_case("synthetic/fixed"; evals = 1, fail_validate_at = 2)
        check!(c)          # a second validate would throw; a fixed-evals case has none
        @test (n.validate, n.prepare, n.op) == (1, WARM, WARM)
    end
end

@testset "check_cases!" begin
    good, ngood = counting_case("synthetic/good"; evals = nothing)
    bad, _ = counting_case("synthetic/bad"; evals = 1, fail_validate_at = 1)
    late, nlate = counting_case("synthetic/late"; evals = nothing, fail_validate_at = 2)
    io = IOBuffer()
    failed = redirect_stderr(devnull) do
        check_cases!(Any[bad, good, late], io)
    end
    @test failed == ["synthetic/bad", "synthetic/late"]
    @test ngood.validate == 2 && nlate.validate == 2     # the cases after a failure still run
    rows = split.(split(chomp(String(take!(io))), '\n'), '\t')
    @test rows[1] == ["id", "exception", "stack"]
    @test [r[1] for r in rows[2:end]] == failed
    @test all(length(r) == 3 for r in rows)
    @test occursin("validate failed at call 1", rows[2][2]) && occursin("validate failed at call 2", rows[3][2])
    @test all(occursin("check!", r[3]) for r in rows[2:end])   # the full stack trace, not a summary
    # Streaming: each row is flushed when its case fails, before the next case runs.
    path = tempname()
    seen = Ref("")
    bad, _ = counting_case("synthetic/bad"; evals = 1, fail_validate_at = 1)
    spy = PerfCase((; id = "synthetic/spy"), () -> (seen[] = read(path, String); nothing), noop, noop, 1, false)
    redirect_stderr(devnull) do
        open(io -> check_cases!(Any[bad, spy], io), path, "w")
    end
    @test occursin("synthetic/bad", seen[])
end

include("cases_test.jl")
