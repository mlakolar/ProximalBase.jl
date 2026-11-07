# Failure-path smoke test of the two benchmark scripts (PR #538, review round 3).
#
#   julia --project=benchmark benchmark/csv_failure_smoke.jl
#
# It runs `lmo_atoms.jl` (suite `smoke`) and `frank_wolfe_prototype.jl` on tiny problems, once normally and once with
# the test hooks that force every Krylov call not to converge (`LMO_BENCH_KRYLOV_MAXITER`, `FW_BENCH_KRYLOV_MAXITER`)
# and with a budget too small for any target, then parses the saved files and checks that every row has the schema's
# number of fields, that nonconvergence and censoring are recorded as rows, and that the rows completed before a
# failure are still in the file. It measures nothing.
using Test

const DIR = @__DIR__
const JULIA = Base.julia_cmd()

function run_script(script, args; env = Pair{String, String}[])
  cmd = `$JULIA --startup-file=no --project=$DIR $(joinpath(DIR, script)) $args`
  for (k, v) in env
    cmd = addenv(cmd, k => v)
  end
  run(pipeline(cmd; stdout = devnull))
end

function read_csv(path)
  @test isfile(path)
  @test !isfile(path * ".tmp")
  lines = readlines(path)
  header = split(lines[1], ",")
  rows = [split(l, ",") for l in lines[2:end]]
  @test all(r -> length(r) == length(header), rows)           # the schema, for every row
  @test !isempty(rows)
  header, [Dict(zip(header, r)) for r in rows]
end

mktempdir() do tmp
  @testset "lmo_atoms.jl, normal run" begin
    out = joinpath(tmp, "lmo_ok.csv")
    run_script("lmo_atoms.jl", ["smoke-ok", out, "smoke"])
    header, rows = read_csv(out)
    @test length(header) == 19
    @test !any(r -> r["converged"] == "false", rows)
    @test count(r -> r["family"] == "vertex", rows) == 8
    @test any(r -> r["family"] == "storage" && r["converged"] == "true", rows)
    @test any(r -> r["family"] == "density" && r["method"] == "krylov" && r["fallback"] == "disabled", rows)
  end
  @testset "lmo_atoms.jl, every Krylov call fails to converge" begin
    out = joinpath(tmp, "lmo_fail.csv")
    run_script("lmo_atoms.jl", ["smoke-fail", out, "smoke"]; env = ["LMO_BENCH_KRYLOV_MAXITER" => "1"])
    header, rows = read_csv(out)
    @test length(header) == 19
    failed = filter(r -> r["converged"] == "false", rows)
    for family in ("nuclear-dense", "nuclear-sparse", "density", "storage")
      @test any(r -> r["family"] == family, failed)             # recorded, not silently densified
    end
    @test all(r -> r["median_ms"] == "" && r["fallback"] == "disabled", failed)
    @test count(r -> r["family"] == "vertex", rows) == 8        # rows from before the first failure survive
    @test any(r -> r["family"] == "nuclear-dense" && r["method"] == "full", rows)   # and the full rows around the failures
    @test any(r -> r["family"] == "fantope", rows)              # the run went on to the end
    @test any(r -> r["family"] == "density" && r["method"] == "full", rows)
  end
  common = ["60", "60", "3", "0.3", "1", "145"]
  @testset "frank_wolfe_prototype.jl, normal run" begin
    out = joinpath(tmp, "fw_ok.csv")
    run_script("frank_wolfe_prototype.jl", [out, common..., "60"])
    header, rows = read_csv(out)
    @test length(header) == 17
    @test any(r -> r["method"] == "reference" && startswith(r["note"], "accepted"), rows)
    @test any(r -> r["method"] == "fw-krylov" && r["time_s"] != "NaN", rows)
  end
  @testset "frank_wolfe_prototype.jl, the Krylov oracle fails" begin
    out = joinpath(tmp, "fw_fail.csv")
    run_script("frank_wolfe_prototype.jl", [out, common..., "60"]; env = ["FW_BENCH_KRYLOV_MAXITER" => "1"])
    header, rows = read_csv(out)
    @test length(header) == 17
    krylov = filter(r -> r["method"] == "fw-krylov", rows)
    @test !isempty(krylov) && all(r -> r["time_s"] == "NaN" && occursin("did not converge", r["note"]), krylov)
    @test any(r -> r["method"] == "fista-nuclear", rows)         # the comparator rows of the same repetition are kept
    @test any(r -> r["method"] == "reference", rows)
  end
  @testset "frank_wolfe_prototype.jl, censored targets" begin
    out = joinpath(tmp, "fw_censored.csv")
    run_script("frank_wolfe_prototype.jl", [out, common..., "0.0001"])
    header, rows = read_csv(out)
    @test length(header) == 17
    @test any(r -> occursin("censored", r["note"]), rows)
    @test any(r -> r["method"] == "fista-nuclear", rows)
  end
end
