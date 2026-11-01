module ConformanceHarnessTest

using Test
using ..Conformance

module ReportScript
include(joinpath(@__DIR__, "..", "..", "scripts", "conformance_map.jl"))
end

runmain(args) = redirect_stdio(() -> ReportScript.main(args); stdout = devnull, stderr = devnull)

mutable struct Capture <: Test.AbstractTestSet
  description::String
  results::Vector{Any}
end
Capture(description; kwargs...) = Capture(description, Any[])
Test.record(ts::Capture, result) = push!(ts.results, result)
function Test.finish(ts::Capture)
  parent = Test.get_testset()
  parent isa Capture && Test.record(parent, ts)
  ts
end
function capture(f)
  verdict = Ref{Any}()
  ts = @testset Capture "capture" begin
    verdict[] = f()
  end
  verdict[], ts
end
function results(ts)
  reduce(vcat, [r isa Test.AbstractTestSet ? results(r) : [r] for r in ts.results]; init = Any[])
end

# Every call still goes through the public macro, including its caller scope.
function emit(id; expected_fn = () -> 1.0, observed_fn = () -> 1.0,
              compare_fn = same, broken_spec = nothing, case_id = "V1", family_id = :harness)
  @row begin
    id = id; case = case_id; family = family_id; oracle = :value
    types = "Float64"; backend = :cpu; rules = "T3"
    problem = "identity after conversion"
    expected = expected_fn()
    observed = observed_fn()
    compare = compare_fn
    broken = broken_spec
  end
end

withenv("CONFORMANCE_REPORT" => nothing) do
  @testset "Harness classifications" begin
    baseline = (group = :G1, baseline = Returned(v -> isequal(v, 2.0)))
    for (id, kwargs, want, kind) in (
      ("pass", (;), :pass, Test.Pass),
      ("broken", (; observed_fn = () -> 2.0, broken_spec = baseline), :broken, Test.Broken),
      ("promote", (; broken_spec = baseline), :unexpected_pass, Test.Fail),
      ("unrelated", (; observed_fn = () -> throw(MethodError(identity, ())), broken_spec = baseline), :error, Test.Fail),
      ("wrong", (; observed_fn = () -> 3.0), :error, Test.Fail),
      ("wrong-baseline", (; observed_fn = () -> 3.0, broken_spec = baseline), :error, Test.Fail),
      ("throws", (; expected_fn = () -> Throws(:ArgumentError, r"bad"), observed_fn = () -> throw(ArgumentError("bad"))), :pass, Test.Pass),
      ("absent-type", (; expected_fn = () -> Throws(:NonexistentConformanceException), observed_fn = () -> throw(ArgumentError("bad"))), :error, Test.Fail),
      ("compare", (; compare_fn = (a, b) -> error("comparison"), broken_spec = baseline), :error, Test.Fail),
      ("reference", (; expected_fn = () -> error("reference"), broken_spec = baseline), :error, Test.Fail),
      ("expected-throw-baseline", (; expected_fn = () -> Throws(:PrecisionError), observed_fn = () -> 2.0, broken_spec = baseline), :broken, Test.Broken),
      ("threw-baseline", (; observed_fn = () -> throw(ArgumentError("old")), broken_spec = (group = :G12, baseline = Threw(:ArgumentError, "old"))), :broken, Test.Broken),
      ("value-is-exception", (; expected_fn = () -> Throws(:ArgumentError), observed_fn = () -> ArgumentError("returned")), :error, Test.Fail),
      ("compare-nonbool", (; compare_fn = (a, b) -> 1), :error, Test.Fail),
      ("baseline-error", (; observed_fn = () -> 2.0, broken_spec = (group = :G1, baseline = Returned(v -> error("baseline")))), :error, Test.Fail),
    )
      verdict, ts = capture(() -> emit(id; kwargs...))
      @test verdict == want
      @test length(results(ts)) == 1
      @test only(results(ts)) isa kind
      if want == :unexpected_pass
        @test occursin("promote the row", only(ts.results).description)
      end
    end
    @test_throws r"duplicate conformance row id" emit("pass")
    @test_throws r"G1 to G12" emit("bad-group"; broken_spec = (group = :G13, baseline = Returned(isinf)))
    @test_throws r"G1 to G12" emit("bad-group-zero"; broken_spec = (group = :G0, baseline = Threw(:ErrorException)))
  end

  @testset "Reference helpers" begin
    @test pow2(Float32, -80) === ldexp(1f0, -80)
    @test bigref(() -> BigFloat(1) / 3) === Float64(1 // 3)
    @test bigref(() -> BigFloat(1) / 3, Float32) === Float32(1 // 3)
    @test bigref(() -> (BigFloat(0), BigFloat(Inf)); T = Float32) === (0f0, Inf32)
    @test bigref(() -> BigFloat[1, 2]; T = Float32) == Float32[1, 2]
    @test bigref(() -> BigFloat(NaN)) |> isnan
    @test_throws r"did not stabilize" bigref(() -> iseven(trailing_zeros(precision(BigFloat))) ? BigFloat(1) : BigFloat(2); bits = 256)
    @test combine(Float64, 1f0 + pow2(Float32, -23), 1f0 - pow2(Float32, -23)) === 1.0 - pow2(Float64, -46)
    @test combine(Float32, pow2(Float64, 160), pow2(Float64, -160)) === 1f0
    @test same([NaN, -0.0], copy([NaN, -0.0]))
    @test same([1.0, 2.0], view([1.0, 2.0], :))
    @test same((Float32[1], NaN), (Float32[1], NaN))
    @test !same(Float32[1], [1.0])
    @test !same((Float32[1],), ([1.0],))
    @test !same(zeros(2), zeros(1, 2))
    @test same(NaN, NaN)
    @test !same(0.0, -0.0)
    @test !same([0.0], [-0.0])
    @test within_ulps(nextfloat(1f0), 1f0, 1)
    @test !within_ulps(nextfloat(1f0, 2), 1f0, 1)
    @test within_ulps(nextfloat(0.0), nextfloat(nextfloat(0.0)), 1)
    @test !within_ulps(0.0, nextfloat(0.0), 1)
    # A narrow result must not pass a wide reference by being near it in its own ulps.
    ref64 = 2.0^-23 + 2.0^-72
    @test within_ulps(ref64, ref64, 0)
    @test !within_ulps(Float32(2.0^-23), ref64, 2)
    @test !within_ulps(Inf32, Inf, 0)
    @test within_ulps(Inf, Inf, 0)
    @test !within_ulps(Inf, -Inf, 0)
    @test within_ulps(NaN, NaN, 0)
    @test !within_ulps(0.0, -0.0, 1)
    @test within_bound(1.25, 1.0, 0.25)
    @test !within_bound(1.25, 1.0, 0.24)
    @test !within_bound(0.0, nextfloat(0.0), 1.0)
    @test within_bound(Inf, Inf, 0.0)
    @test !within_bound(Inf, -Inf, Inf)
    @test within_bound(NaN, NaN, 0.0)
    @test_throws ArgumentError within_ulps(1.0, 1.0, -1)
    @test_throws ArgumentError within_bound(1.0, 1.0, -1)
  end

  @testset "Reports and coverage script" begin
    mktempdir() do dir
      withenv("CONFORMANCE_REPORT" => dir, "CONFORMANCE_PKG_SHA" => "fixture-sha") do
        emit("fixture-pass"; observed_fn = () -> -0.0, expected_fn = () -> -0.0)
        emit("fixture-broken"; observed_fn = () -> NaN,
             broken_spec = (group = :G2, baseline = Returned(isnan)))
        @notcovered :harness :prox "Float64" :cpu "R1: unsupported transformation\nwith \"quotes\", \\ and |"
        run = only(ReportScript.run_directories(dir))
        path = joinpath(run, "harness.jsonl")
        lines = readlines(path)
        @test length(lines) == 4
        h = ReportScript.parse_json_line(lines[1])
        @test h["pkgsha"] == "fixture-sha"
        @test h["bounds"] == string(Base.JLOptions().check_bounds)
        @test h["julia"] == string(VERSION)
        @test !isempty(h["platform"]) && !isempty(h["harness"])
        @test ReportScript.parse_json_line(lines[2])["observed"] == "-0.0"
        @test ReportScript.parse_json_line(lines[3])["observed"] == "NaN"
        @test ReportScript.parse_json_line(lines[4])["status"] == "notcovered"
        @test occursin("\"quotes\", \\ and |", ReportScript.parse_json_line(lines[4])["problem"])
        md, complete = ReportScript.coverage(dir; case_ids = ["V1"],
          required_ids = ["fixture-pass", "fixture-broken"], expected_cells = [])
        @test complete
        @test occursin("| G2 | 1 |", md)
        @test occursin("not covered", md)
        @test occursin("fixture-sha", md)
        @test ReportScript.coverage(run; case_ids = ["V1"],
          required_ids = ["fixture-pass", "fixture-broken"], expected_cells = []) == (md, true)
        default_md, complete = ReportScript.coverage(dir)
        @test !complete
        @test occursin("V5.value", default_md)
        @test occursin("missing", default_md)
        out = joinpath(dir, "coverage.md")
        @test runmain([dir, "--out", out]) == 1 # incomplete by design
        @test read(out, String) == default_md
        @test runmain([dir, "--out", out, "--check"]) == 1
        write(out, "stale")
        @test runmain([dir, "--out", out, "--check"]) == 1
        @test read(out, String) == "stale"
        diff = IOBuffer()
        ReportScript.print_diff(diff, out, default_md)
        @test occursin("-stale\n+# Numerical contract coverage", String(take!(diff)))
        @test runmain(["--help"]) == 0
        @test runmain([dir, "--bogus"]) == 1
        @test_throws ErrorException ReportScript.parse_json_line("{\"a\":true}")
        @test_throws ErrorException ReportScript.parse_json_line("{\"a\":\"x\",\"a\":\"y\"}")
        @test_throws ErrorException ReportScript.parse_json_line("{} trailing")
        @test ReportScript.parse_json_line(Conformance.json_object((; text = "\0\t\r\n雪")))["text"] == "\0\t\r\n雪"
        # A new file starts by overwriting its old report, never by appending.
        delete!(Conformance._reports, (abspath(@__FILE__), abspath(run)))
        emit("fixture-restart")
        @test length(readlines(path)) == 2
        # Per-run provenance conflicts are rejected, not pooled.
        other = joinpath(run, "other.jsonl")
        for key in ("pkgsha", "bounds", "julia", "platform", "harness")
          changed = merge(h, Dict(key => "different"))
          write(other, Conformance.json_object(changed) * "\n")
          @test_throws ErrorException ReportScript.read_run(run)
        end
        rm(other)
        bad = ReportScript.parse_json_line(lines[3])
        bad["group"] = ""
        write(path, lines[1] * "\n" * Conformance.json_object(bad) * "\n")
        @test_throws r"broken row needs group" ReportScript.read_run(run)
      end
    end
    # A complete throwaway run checks CLI success as well as the incomplete
    # fixture above. These identity rows are fixtures, not package evidence.
    mktempdir() do dir
      withenv("CONFORMANCE_REPORT" => dir, "CONFORMANCE_PKG_SHA" => "complete-fixture") do
        families = getproperty.(ReportScript.EXPECTED_CELLS, :family)
        for (i, id) in enumerate(ReportScript.REQUIRED_ROW_IDS)
          emit(id; case_id = first(split(id, '.')),
               family_id = Symbol(families[mod1(i, length(families))]))
        end
        out = joinpath(dir, "coverage.md")
        @test runmain([dir, "--out", out]) == 0
        @test runmain([dir, "--out", out, "--check"]) == 0
        @test !occursin("| missing |", read(out, String))
        # Changing the package SHA in the same process creates a new run
        # with its own header instead of pooling configurations.
        withenv("CONFORMANCE_PKG_SHA" => "second-fixture") do
          emit("second-run")
        end
        runs = ReportScript.run_directories(dir)
        @test length(runs) == 2
        @test all(run -> !isempty(first(ReportScript.read_run(run))["pkgsha"]), runs)
        md, complete = ReportScript.coverage(dir)
        @test !complete
        @test occursin("second-fixture", md) && occursin("complete-fixture", md)
      end
    end
  end
end

end
