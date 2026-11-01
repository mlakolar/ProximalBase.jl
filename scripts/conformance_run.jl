# Runs the conformance suite once into an empty report directory, gives every report
# file the same run id, and builds the coverage map from those reports.
module MapScript
include(joinpath(@__DIR__, "conformance_map.jl"))
end

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const USAGE = "Usage: julia scripts/conformance_run.jl <report-dir> [--out FILE] [--check] [--jobs N]"

# UTC time as yyyymmddTHHMMSS, from Base only (the test environment has no Dates);
# days to civil date by the standard proleptic Gregorian algorithm.
function utc_stamp(t = time())
  secs = floor(Int, t)
  days, rest = fldmod(secs, 86400)
  z = days + 719468
  era = fld(z, 146097)
  doe = z - era * 146097
  yoe = div(doe - div(doe, 1460) + div(doe, 36524) - div(doe, 146096), 365)
  doy = doe - (365 * yoe + div(yoe, 4) - div(yoe, 100))
  mp = div(5 * doy + 2, 153)
  day = doy - div(153 * mp + 2, 5) + 1
  month = mp < 10 ? mp + 3 : mp - 9
  year = yoe + era * 400 + (month <= 2)
  string(lpad(year, 4, '0'), lpad(month, 2, '0'), lpad(day, 2, '0'), 'T',
    lpad(div(rest, 3600), 2, '0'), lpad(div(rest, 60) % 60, 2, '0'), lpad(rest % 60, 2, '0'))
end
new_run_id() = utc_stamp() * "-" * string(rand(UInt32); base = 16, pad = 8)

# One report file per test file under test/conformance/ other than common and harness.
function conformance_stems()
  files = readdir(joinpath(REPO_ROOT, "test", "conformance"))
  sort!([first(splitext(f)) for f in files if endswith(f, ".jl") && !(f in ("common.jl", "harness.jl"))])
end

function check_stems(stems)
  expected = sort(collect(String, MapScript.EXPECTED_FILES))
  stems == expected && return
  error("test files and EXPECTED_FILES differ; test files without an expected report: ",
    join(setdiff(stems, expected), ", "), "; expected reports without a test file: ",
    join(setdiff(expected, stems), ", "))
end

# One test command for all stems, from the repository root, inheriting stdout and stderr.
function default_test_command(stems, env; jobs = nothing)
  args = ["conformance/" * s for s in stems]
  jobs === nothing || push!(args, "--jobs=$jobs")
  cmd = Cmd(`$(Base.julia_cmd()) --startup-file=no scripts/test.jl $args`; dir = REPO_ROOT)
  run(ignorestatus(addenv(cmd, env...))).exitcode == 0
end

# `test_command(stems, env)` returns true/false or an exit code; tests stub it.
# By default it runs the real suite.
function main(args = ARGS; test_command = nothing)
  if "--help" in args
    println(USAGE)
    return 0
  end
  try
    isempty(args) && error(USAGE)
    dir = first(args)
    out = joinpath(dir, "coverage.md")
    check = false
    jobs = nothing
    i = 2
    while i <= length(args)
      if args[i] == "--out" && i < length(args)
        i += 1
        out = args[i]
      elseif args[i] == "--jobs" && i < length(args)
        i += 1
        jobs = tryparse(Int, args[i])
        jobs !== nothing && jobs >= 1 || error("--jobs needs a positive integer: $(args[i])")
      elseif args[i] == "--check"
        check = true
      else
        error("unknown or incomplete option: $(args[i])\n$USAGE")
      end
      i += 1
    end
    (isdir(dir) && !isempty(readdir(dir))) && error("report directory is not empty: $dir")
    stems = conformance_stems()
    check_stems(stems)
    mkpath(dir)
    id = new_run_id()
    env = ["CONFORMANCE_REPORT" => abspath(dir), "CONFORMANCE_RUN_ID" => id]
    command = test_command === nothing ? (s, e) -> default_test_command(s, e; jobs) : test_command
    result = command(stems, env)
    tests_ok = result isa Bool ? result : result == 0
    tests_ok || println(stderr, "conformance tests failed; building the map anyway")
    map_args = [dir, "--out", out, "--run-id", id]
    check && push!(map_args, "--check")
    map_ok = MapScript.main(map_args) == 0
    tests_ok && map_ok ? 0 : 1
  catch e
    println(stderr, sprint(showerror, e))
    1
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  exit(main())
end
