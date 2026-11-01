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
      ("volatile-pass", (; broken_spec = (group = :G4, baseline = Returned(v -> isequal(v, 2.0)), volatile = true)), :pass, Test.Pass),
      ("volatile-broken", (; observed_fn = () -> 2.0, broken_spec = (group = :G4, baseline = Returned(v -> isequal(v, 2.0)), volatile = true)), :broken, Test.Broken),
      ("volatile-wrong", (; observed_fn = () -> 3.0, broken_spec = (group = :G4, baseline = Returned(v -> isequal(v, 2.0)), volatile = true)), :error, Test.Fail),
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
    # near: the same type and class, last bits tolerated, nothing else.
    @test near(0.4f0)(nextfloat(0.4f0, 3))
    @test !near(0.4f0, 2)(nextfloat(0.4f0, 3))
    @test !near(0.4f0)(0.4)
    @test near(Float32[1, 2])(Float32[nextfloat(1f0), 2])
    @test !near(Float32[1, 2])(Float32[1])
    @test !near(Float32[1, 2])(Float64[1, 2])
    @test !near(1f0)(NaN32)
    @test within_ulps(Inf, Inf, 0)
    @test !within_ulps(Inf, -Inf, 0)
    @test within_ulps(NaN, NaN, 0)
    @test !within_ulps(0.0, -0.0, 1)
    @test within_bound(1.25, 1.0, 0.25)
    @test !within_bound(1.25, 1.0, 0.24)
    @test !within_bound(0.0, nextfloat(0.0), 1.0)
    # typed_bound: the delivery type is part of the promise; within_bound is untyped.
    @test !typed_bound(Float64(1), 1f0, 0.0)
    @test within_bound(Float64(1), 1f0, 0.0)
    @test typed_bound(1f0, 1f0, 0.0)
    @test typed_bound(1.25f0, 1f0, 0.25)
    @test !typed_bound(1.25f0, 1f0, 0.24)
    @test !typed_bound(1f0, Float64(1), 0.0)
    @test !typed_bound(Inf, Inf32, 0.0)
    @test typed_bound(Inf32, Inf32, 0.0)
    @test typed_bound(NaN32, NaN32, 0.0)
    @test !typed_bound(BigFloat(1), 1.0, 0.0)
    @test typed_bound(BigFloat(1), BigFloat(1), 0.0)
    @test_throws ArgumentError typed_bound(1f0, 1f0, -1)
    # once: one invocation, replayed value or exception.
    calls = Ref(0)
    replay = once(() -> (calls[] += 1; calls[]))
    @test replay() == 1 && replay() == 1 && calls[] == 1
    failing = once(() -> throw(ArgumentError("once")))
    @test_throws ArgumentError failing()
    @test_throws ArgumentError failing()
    @test within_bound(Inf, Inf, 0.0)
    @test !within_bound(Inf, -Inf, Inf)
    @test within_bound(NaN, NaN, 0.0)
    @test_throws ArgumentError within_ulps(1.0, 1.0, -1)
    @test_throws ArgumentError within_bound(1.0, 1.0, -1)
  end

  @testset "rank_indicator_ref and the L5.value rule" begin
    h = nextfloat(0f0)
    rankone = fill(h, 2, 2)
    ranktwo = h .* Float32[2 1; 1 1]
    @test rank_indicator_ref(zeros(Float32, 2, 2)) === 0f0
    @test rank_indicator_ref(rankone) === 0f0
    @test rank_indicator_ref(ranktwo) === Inf32
    @test rank_indicator_ref(Float32[1 2; 2 4]) === 0f0
    @test rank_indicator_ref(Float32[1 0; 0 1]) === Inf32
    @test rank_indicator_ref(Float32[1 0; 0 1f-3]) === Inf32
    @test rank_indicator_ref(Float32[1 0; 0 1f-5]) === 0f0
    @test_throws ArgumentError rank_indicator_ref(zeros(Float32, 3, 3))
    # The rows of spectral.jl: the scalar is compared with the reference of the
    # matrix the same call stored; the baseline is a zero scalar at rank two.
    compare = (o, _) -> same(o.scalar, o.reference)
    baseline = (group = :G6, baseline = Returned(o -> same(o.scalar, 0f0) && same(o.reference, Inf32)))
    fused(scalar, Y) = () -> (scalar = scalar, reference = rank_indicator_ref(Y))
    for (id, scalar, Y, want) in (
      ("l5-zero-rank-one", 0f0, rankone, :unexpected_pass),
      ("l5-inf-rank-one", Inf32, rankone, :error),
      ("l5-inf-rank-two", Inf32, ranktwo, :unexpected_pass),
      ("l5-zero-rank-two", 0f0, ranktwo, :broken),
    )
      verdict, _ = capture(() -> emit(id; expected_fn = () -> "scalar === reference",
        observed_fn = fused(scalar, Y), compare_fn = compare, broken_spec = baseline))
      @test verdict == want
    end
    # Without the baseline the two agreeing outputs are plain passes.
    for (id, scalar, Y) in (("l5-plain-rank-one", 0f0, rankone), ("l5-plain-rank-two", Inf32, ranktwo))
      verdict, _ = capture(() -> emit(id; expected_fn = () -> "scalar === reference",
        observed_fn = fused(scalar, Y), compare_fn = compare))
      @test verdict == :pass
    end
    @test compare(fused(0f0, rankone)(), nothing)
    @test !compare(fused(Inf32, rankone)(), nothing)
    @test compare(fused(Inf32, ranktwo)(), nothing)
    @test !compare(fused(0f0, ranktwo)(), nothing)
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
          required_ids = ["fixture-pass", "fixture-broken"], expected_cells = [], expected_files = [])
        @test complete
        @test occursin("| G2 | 1 |", md)
        @test occursin("not covered", md)
        @test occursin("fixture-sha", md)
        @test ReportScript.coverage(run; case_ids = ["V1"],
          required_ids = ["fixture-pass", "fixture-broken"], expected_cells = [], expected_files = []) == (md, true)
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
        # Case-less rows, as in linear_sets_2: no row id or cell depends on them.
        emit("fixture-split-1"; case_id = "", family_id = :linear_sets)
        emit("fixture-split-2"; case_id = "", family_id = :linear_sets)
        # The fixture emits into harness.jsonl; assemble the expected report files
        # by copying its header and rows, the case-less rows into linear_sets_2.
        run = only(ReportScript.run_directories(dir))
        lines = readlines(joinpath(run, "harness.jsonl"))
        header, rows = lines[1], lines[2:end]
        split_rows = filter(l -> occursin("fixture-split-", l), rows)
        other_rows = filter(l -> !(l in split_rows), rows)
        others = filter(!=("linear_sets_2"), ReportScript.EXPECTED_FILES)
        @test length(others) + 1 == length(ReportScript.EXPECTED_FILES)
        for (k, name) in enumerate(others)
          write(joinpath(run, name * ".jsonl"), join([header; other_rows[k:length(others):end]], "\n") * "\n")
        end
        write(joinpath(run, "linear_sets_2.jsonl"), join([header; split_rows], "\n") * "\n")
        rm(joinpath(run, "harness.jsonl"))
        @test sort(first.(splitext.(readdir(run)))) == sort(ReportScript.EXPECTED_FILES)
        out = joinpath(dir, "coverage.md")
        @test runmain([dir, "--out", out]) == 0
        @test runmain([dir, "--out", out, "--check"]) == 0
        @test !occursin("| missing |", read(out, String))
        @test occursin("(missing report file): none.", read(out, String))
        # Removing one split file from the complete fresh report must fail the
        # generation, though no row id, case or cell is missing.
        split_path = joinpath(run, "linear_sets_2.jsonl")
        saved = read(split_path, String)
        rm(split_path)
        out_missing = joinpath(dir, "coverage-missing.md")
        @test runmain([dir, "--out", out_missing]) == 1
        text = read(out_missing, String)
        @test occursin("(missing report file): linear_sets_2.", text)
        @test occursin("Required row ids without a row (missing): none.", text)
        @test occursin("Case ids without a row: none.", text)
        @test !occursin("| missing |", text)
        @test runmain([dir, "--out", out_missing, "--check"]) == 1
        write(split_path, saved)
        @test runmain([dir, "--out", out, "--check"]) == 0
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
