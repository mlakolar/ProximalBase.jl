# The verdict job of the CI workflow ("PR required jobs" on pull requests, "CI required jobs"
# otherwise): it runs after the plan, the test jobs, the benchmark reference checks and the
# benchmark controls, even when one of them failed, and passes only if everything the plan asked
# for passed. It fails closed:
#
# - the plan succeeded and gave a valid tier; the benchmark controls succeeded; the benchmark
#   reference checks succeeded when the plan asked for them and were skipped when it did not;
# - when tests run (a tier other than none), `required` (the ';'-separated names of the test
#   jobs the plan lists) is not empty, holds the minimum of the event ("Julia 1" on Ubuntu for a
#   pull request or a manual run, Julia 1 and lts on Ubuntu for a push to master, the four cross
#   jobs for a tag and the nightly run), shares no name with `advisory`, and every required job
#   of this run is listed with the conclusion success (missing, failed, cancelled, skipped and
#   timed out fail); any other "Julia ..." job fails (a platform the plan did not name);
# - with tier none every "Julia ..." job must be skipped (GitHub lists a skipped matrix job
#   under its unexpanded name template);
# - advisory jobs (Julia pre) are printed and never counted;
# - the job list comes from `gh api`, retried; without it there is no pass.
#
# The step reads the environment (see `main`) and writes `failed` (the names of the failed
# jobs, ';'-separated) and `linux_failed` (whether the failure is anything but a failed
# non-Linux test job) to $GITHUB_OUTPUT for the report job. test/ci_verdict.jl tests the
# decisions. No packages are needed.

module CIVerdict

const TIERS = ("quick", "standard", "thorough", "full")

# One tab-separated line per job: its name and its conclusion (empty until it has finished).
const JOBS_JQ = """.jobs[] | [.name, (.conclusion // "")] | @tsv"""

jobname(version, os, arch, tier) = "Julia $version - $os - $arch - $tier"

"""
    minimum_jobs(event, ref_type, tier) -> Vector{String}

The test jobs an event must pass, whatever the plan lists: Julia 1 on Ubuntu for a pull request
or a manual run, Julia 1 and lts on Ubuntu for a push to master, Julia 1 and lts on Ubuntu,
Julia 1 on macOS and Julia 1 on Windows for a tag push and the nightly run. Throws for another
event or ref type.
"""
function minimum_jobs(event, ref_type, tier)
  linux1 = jobname("1", "ubuntu-latest", "x64", tier)
  lts = jobname("lts", "ubuntu-latest", "x64", tier)
  cross = [lts, linux1, jobname("1", "macos-latest", "aarch64", tier), jobname("1", "windows-latest", "x64", tier)]
  event in ("pull_request", "workflow_dispatch") && return [linux1]
  event == "schedule" && return cross
  if event == "push"
    ref_type == "branch" && return [lts, linux1]
    ref_type == "tag" && return cross
    error("a push of ref type `$ref_type` has no minimum of jobs")
  end
  error("unknown event `$event`")
end

splitnames(text) = String[n for n in split(text, ';') if !isempty(strip(n))]

isjulia(name) = startswith(name, "Julia ")

"""
    verdict(; event, ref_type, plan, tests, analysis, benchmark, run_benchmark, run_tests, tier,
            required, advisory, jobs) -> (; errors, notes, failed, linux_failed)

The decision. `plan`, `tests`, `analysis` and `benchmark` are the results of the jobs
(`success`, `failure`, `cancelled`, `skipped`), `run_benchmark`, `run_tests`, `tier`, `required`
and `advisory` the plan's outputs (strings, possibly empty), and `jobs` this run's test jobs as
`(name, conclusion)` pairs, or `nothing` when they could not be listed. The run passes exactly
when `errors` is empty. `notes` are lines to print, `failed` the names of the failed jobs and
`linux_failed` whether a failure is anything but a failed test job that is not on Linux.
"""
function verdict(; event, ref_type = "", plan, tests, analysis, benchmark, run_benchmark, run_tests, tier,
                 required, advisory, jobs)
  errors = String[]
  notes = String[]
  failed = String[]
  other = false # a failure that is not a failed non-Linux test job
  function problem(message; job = nothing, linux = true)
    push!(errors, message)
    job === nothing || push!(failed, job)
    other |= linux
  end
  plan == "success" || problem("The test plan did not succeed ($plan)."; job = "Test plan")
  analysis == "success" || problem("Benchmark analysis controls: $analysis."; job = "Benchmark analysis controls")
  # An intended skip passes; a planned check must succeed on both Julia versions.
  if run_benchmark == "true"
    benchmark == "success" || problem("Benchmark reference checks: $benchmark."; job = "Benchmark reference checks")
  elseif run_benchmark == "false"
    benchmark == "skipped" ||
      problem("Benchmark reference checks not planned, but $benchmark."; job = "Benchmark reference checks")
  else
    problem("The plan did not say whether the benchmark reference checks run ('$run_benchmark').")
  end
  if plan == "success"
    if !(tier == "none" || tier in TIERS)
      problem("The plan gave no valid tier ('$tier').")
    elseif tier == "none"
      run_tests == "false" || problem("Tier none, but run_tests is '$run_tests'.")
      tests == "skipped" || problem("Tier none, but the test jobs are $tests.")
      isempty(splitnames(required)) || problem("Tier none, but the plan lists required jobs.")
      if jobs === nothing
        problem("The jobs of this run could not be listed, so tier none cannot be confirmed.")
      else
        for (name, conclusion) in jobs
          isjulia(name) && conclusion != "skipped" &&
            problem("Tier none, but the job `$name` is $(isempty(conclusion) ? "not finished" : conclusion)."; job = name)
        end
      end
    else
      run_tests == "true" || problem("Tier $tier without run_tests ('$run_tests').")
      req = splitnames(required)
      adv = splitnames(advisory)
      isempty(req) && problem("The plan lists no required test jobs for tier $tier.")
      for name in unique(req)
        count(==(name), req) > 1 && problem("The plan lists the required job `$name` more than once.")
      end
      for name in intersect(req, adv)
        problem("`$name` is both required and advisory.")
      end
      minimum = try
        minimum_jobs(event, ref_type, tier)
      catch err
        problem(first(sprint(showerror, err), 300))
        String[]
      end
      for name in minimum
        name in req || problem("The required jobs do not include `$name`, which every $event run needs.")
      end
      tests in ("cancelled", "skipped") && problem("The test jobs are $tests.")
      if jobs === nothing
        problem("The jobs of this run could not be listed.")
      else
        conclusions(name) = [c for (n, c) in jobs if n == name]
        for name in unique(req)
          cs = conclusions(name)
          if isempty(cs)
            problem("$name: missing."; job = name, linux = occursin("ubuntu", name))
          elseif length(cs) > 1
            problem("$name: listed $(length(cs)) times."; job = name, linux = occursin("ubuntu", name))
          elseif only(cs) != "success"
            problem("$name: $(isempty(only(cs)) ? "not finished" : only(cs))."; job = name, linux = occursin("ubuntu", name))
          end
        end
        known = Set([req; adv])
        for (name, _) in jobs
          isjulia(name) && !(name in known) &&
            problem("The job `$name` is not one the plan asked for.")
        end
        for name in adv
          cs = conclusions(name)
          push!(notes, "Advisory: $name: $(isempty(cs) ? "missing" : join(map(c -> isempty(c) ? "not finished" : c, cs), ", "))")
        end
      end
    end
  end
  unique!(failed)
  return (; errors, notes, failed, linux_failed = !isempty(errors) && (other || any(occursin("ubuntu"), failed)))
end

# `gh api` with a time limit: a hung request counts as an API error.
function gh(args::Vector{String}; timeout = 60)
  out, err = IOBuffer(), IOBuffer()
  p = run(pipeline(`gh api $args`; stdout = out, stderr = err); wait = false)
  if timedwait(() -> process_exited(p), timeout) === :timed_out
    kill(p)
    error("gh api timed out after $timeout s")
  end
  success(p) || error("gh api failed: ", strip(String(take!(err))))
  return String(take!(out))
end

function parse_jobs(text::AbstractString)
  jobs = Tuple{String, String}[]
  for line in split(text, '\n')
    isempty(strip(line)) && continue
    f = split(line, '\t')
    length(f) == 2 || error("unexpected job line: $(repr(line))")
    push!(jobs, (String(f[1]), String(f[2])))
  end
  return jobs
end

# The latest attempt of each job, so a re-run of failed jobs counts.
function github_jobs(repo, run_id)
  text = gh(["repos/$repo/actions/runs/$run_id/jobs?filter=latest&per_page=100", "--paginate", "--jq", JOBS_JQ])
  return parse_jobs(text)
end

"""
    list_jobs(fetch; attempts = 4, wait = 15, sleep = Base.sleep, log = println) -> jobs or nothing

Call `fetch()` until it returns the job list; `nothing` after `attempts` failures, which the
verdict treats as a failure and never as a pass.
"""
function list_jobs(fetch; attempts = 4, wait = 15, sleep = Base.sleep, log = println)
  for attempt in 1:attempts
    try
      return fetch()
    catch err
      err isa InterruptException && rethrow()
      log("::warning::Could not list the jobs (attempt $attempt): ", first(sprint(showerror, err), 300))
      attempt < attempts && sleep(wait)
    end
  end
  return nothing
end

"""
    main(env = ENV; fetch, sleep, log) -> exit status

Reads PLAN, TESTS, ANALYSIS, BENCHMARK (the results of the jobs), RUN_BENCHMARK, RUN_TESTS, TIER,
FOCUS, REQUIRED and ADVISORY (the plan's outputs), EVENT, REF_TYPE, COMMIT, GITHUB_REPOSITORY,
GITHUB_RUN_ID and GH_TOKEN; prints the decision; appends `failed` and `linux_failed` to
the file named by GITHUB_OUTPUT. Returns 0 when the run passes, 1 when it does not.
"""
function main(env::AbstractDict = ENV;
              fetch = () -> github_jobs(env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"]),
              sleep = Base.sleep, log = println)
  get_(k) = String(get(env, k, ""))
  tier = get_("TIER")
  focus = get_("FOCUS")
  log("Commit $(get_("COMMIT")); event: $(get_("EVENT"))$(isempty(get_("REF_TYPE")) ? "" : " ($(get_("REF_TYPE")))"); ",
      "plan: $(get_("PLAN")); tier: $(isempty(tier) ? "unknown" : tier)$(isempty(focus) ? "" : " with $focus"); ",
      "tests: $(get_("TESTS")); benchmark reference checks: $(get_("BENCHMARK")) ",
      "(planned: $(isempty(get_("RUN_BENCHMARK")) ? "unknown" : get_("RUN_BENCHMARK"))); ",
      "benchmark controls: $(get_("ANALYSIS"))")
  # The job list matters only when the plan decided something to check against it.
  jobs = get_("PLAN") == "success" && (tier == "none" || tier in TIERS) ? list_jobs(fetch; sleep, log) : nothing
  result = try
    verdict(; event = get_("EVENT"), ref_type = get_("REF_TYPE"), plan = get_("PLAN"), tests = get_("TESTS"),
            analysis = get_("ANALYSIS"), benchmark = get_("BENCHMARK"), run_benchmark = get_("RUN_BENCHMARK"),
            run_tests = get_("RUN_TESTS"), tier, required = get_("REQUIRED"), advisory = get_("ADVISORY"), jobs)
  catch err
    err isa InterruptException && rethrow()
    (; errors = ["The verdict could not be decided: " * first(sprint(showerror, err), 300)], notes = String[],
       failed = String[], linux_failed = true)
  end
  foreach(n -> log(n), result.notes)
  foreach(e -> log("::error::", e), result.errors)
  isempty(result.errors) && log("All required jobs passed.")
  path = get_("GITHUB_OUTPUT")
  isempty(path) || open(path, "a") do io
    println(io, "failed=", join(result.failed, ";"))
    println(io, "linux_failed=", result.linux_failed)
  end
  return isempty(result.errors) ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
  # 0: passed, 1: failed; anything else (an exception, a timeout) is for the workflow to retry.
  exit(try
    CIVerdict.main()
  catch err
    err isa InterruptException && rethrow()
    println(stderr, "::error::scripts/ci_verdict.jl: ", sprint(showerror, err))
    2
  end)
end
