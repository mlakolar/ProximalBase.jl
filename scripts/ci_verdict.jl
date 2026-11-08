# The verdict job of the CI workflow ("PR required jobs" on pull requests, "CI required jobs"
# otherwise): it runs after the plan, the test jobs, the benchmark reference checks and the
# benchmark controls, even when one of them failed, and passes only if the checks below hold.
# The plan outputs it validates are the tier, run_tests, the benchmark flag, `required`,
# `advisory`, and the platforms mode when tests run; it does not validate the others (focus, matrix,
# coverage, conformance, timeout):
#
# - the plan succeeded and gave a tier (tier none only for a pull request or a push to master),
#   and run_tests agrees with it; the benchmark controls succeeded; the benchmark reference
#   checks succeeded when the plan asked for them and were skipped when it did not;
# - when tests run (a tier other than none), a recognized platforms mode (stable, linux or
#   cross) is required, and `required` (the ';'-separated names of the test jobs the plan
#   lists) is not empty and contains a minimum that the verdict writes out itself: the floor of
#   the event (Julia 1 on Ubuntu for a pull request, with lts from the thorough tier, and for a
#   manual run; Julia 1 and lts on Ubuntu for a push to master; the four cross jobs for a tag,
#   the nightly run and a release run) together with the jobs of the plan's platforms mode.
#   `advisory` holds
#   at most Julia pre on Ubuntu, and shares no name with `required`. Each required job of this
#   run is listed with the conclusion success (missing, failed, cancelled, skipped and timed out
#   fail), and a "Julia ..." job that is neither required nor advisory fails;
# - a release run (a manual run with RELEASE true) must be of the full tier, and its EXPECTED_SHA
#   must be the full lowercase SHA of the commit that ran (COMMIT);
# - the nightly run, a tag push and a release run also need the conformance reports of each
#   required test job that succeeded: the artifact `conformance-reports-<version>-<os>-<arch>-<attempt>`
#   of the attempt that job ran in. An artifact of an earlier attempt, one named `...-partial-...`
#   or `diagnostic-...`, and a list of artifacts that could not be read do not count. Julia pre is
#   not checked; pull requests, pushes to master and plain manual runs never run this check;
# - with tier none each listed "Julia ..." job must be skipped (GitHub lists a skipped matrix job
#   under its unexpanded name template), and the plan lists no test jobs;
# - advisory jobs (Julia pre) are printed and do not change the result;
# - the job list comes from `gh api`, retried; without it the run does not pass.
#
# The minimum does not validate the planner's choice of mode or tier: a plan that lists the
# minimum and more passes, and so does a plan that lists a smaller mode than the event's labels
# asked for. Omitting a job of the minimum from `required` fails. The minimum is written out
# here, not read from the planner; test/ci_verdict.jl compares its table with the planner's to
# detect drift. The step reads the environment (see `main`) and writes `failed` (the names of
# the failed jobs, ';'-separated) and `linux_failed` (whether the failure is anything but a
# failed non-Linux test job) to $GITHUB_OUTPUT for the report job. test/ci_verdict.jl tests the
# decisions. No packages are needed.
#
# The environment: PLAN, TESTS, ANALYSIS, BENCHMARK, RUN_BENCHMARK, RUN_TESTS, TIER, FOCUS,
# PLATFORMS, REQUIRED, ADVISORY, RELEASE, EXPECTED_SHA, EVENT, REF_TYPE, COMMIT,
# GITHUB_REPOSITORY, GITHUB_RUN_ID and GH_TOKEN (see `main`).

module CIVerdict

const TIERS = ("quick", "standard", "thorough", "full")

# One tab-separated line per job: its name, its conclusion (empty until it has finished) and the
# attempt of the run it belongs to.
const JOBS_JQ = """.jobs[] | [.name, (.conclusion // ""), (.run_attempt | tostring)] | @tsv"""

# The names of the artifacts of a run that have not expired, one per line.
const ARTIFACTS_JQ = ".artifacts[] | select(.expired | not) | .name"

# The full lowercase SHA of a commit; `\z`, not `$`, which also matches before a final newline.
const COMMIT_PATTERN = r"^[0-9a-f]{40}\z"

jobname(version, os, arch, tier) = "Julia $version - $os - $arch - $tier"

# The jobs (version, os, arch) of each platform mode, apart from the advisory Julia pre job. They
# are the verdict's own copy of [platforms.*] in .github/tier-rules.toml.
const MODE_JOBS = Dict(
  "stable" => [("1", "ubuntu-latest", "x64")],
  "linux" => [("lts", "ubuntu-latest", "x64"), ("1", "ubuntu-latest", "x64")],
  "cross" => [("lts", "ubuntu-latest", "x64"), ("1", "ubuntu-latest", "x64"),
              ("1", "macos-latest", "aarch64"), ("1", "windows-latest", "x64")],
)
const ADVISORY_JOB = ("pre", "ubuntu-latest", "x64")

"""
    mode_jobs(platforms, tier) -> Vector{String}

The names of the required test jobs of a platform mode (`stable`, `linux` or `cross`) at `tier`.
Throws for another mode.
"""
function mode_jobs(platforms, tier)
  haskey(MODE_JOBS, platforms) || error("unknown platforms `$platforms`; the modes are stable, linux and cross")
  return [jobname(v, os, arch, tier) for (v, os, arch) in MODE_JOBS[platforms]]
end

"""
    floor_jobs(event, ref_type, tier; release = false) -> Vector{String}

The test jobs an event needs, independent of the plan's platform mode: Julia 1 on Ubuntu for a pull
request (and Julia lts on Ubuntu from the thorough tier) and for a manual run, Julia 1 and lts
on Ubuntu for a push to master, and those with Julia 1 on macOS and on Windows for a tag push, the
nightly run and a release run (a manual run with `release`, which must be of the full tier).
Throws for another event or ref type, and for a release that is not a manual run of the full tier.
"""
function floor_jobs(event, ref_type, tier; release = false)
  if release
    event == "workflow_dispatch" && tier == "full" && return mode_jobs("cross", tier)
    error("a release run is a manual run of the full tier, not a $event run of tier $tier")
  end
  linux1 = jobname("1", "ubuntu-latest", "x64", tier)
  lts = jobname("lts", "ubuntu-latest", "x64", tier)
  cross = mode_jobs("cross", tier)
  event == "pull_request" && return tier in ("thorough", "full") ? [lts, linux1] : [linux1]
  event == "workflow_dispatch" && return [linux1]
  event == "schedule" && return cross
  if event == "push"
    ref_type == "branch" && return [lts, linux1]
    ref_type == "tag" && return cross
    error("a push of ref type `$ref_type` has no minimum of jobs")
  end
  error("unknown event `$event`")
end

"""
    minimum_jobs(event, ref_type, tier, platforms; release = false) -> Vector{String}

The test jobs a run must list as required: the floor of the event ([`floor_jobs`](@ref)) together
with the jobs of the plan's platform mode ([`mode_jobs`](@ref)).
"""
minimum_jobs(event, ref_type, tier, platforms; release = false) =
  unique([floor_jobs(event, ref_type, tier; release); mode_jobs(platforms, tier)])

# The runs whose test jobs must each have uploaded their conformance reports: the nightly run, a
# tag push and a release run.
needs_reports(event, ref_type, release) = release || event == "schedule" || (event == "push" && ref_type == "tag")

"""
    report_artifact(job, attempt) -> String

The name of the artifact with the conformance reports of the test job named `job` (`Julia
<version> - <os> - <arch> - <tier>`) that ran in `attempt` of the run, as the workflow's upload
step names it. Throws for a name that is not a test job's.
"""
function report_artifact(job, attempt)
  parts = split(job, " - ")
  length(parts) == 4 && startswith(parts[1], "Julia ") || error("`$job` is not the name of a test job")
  version, os, arch = replace(parts[1], "Julia " => ""; count = 1), parts[2], parts[3]
  return "conformance-reports-$version-$os-$arch-$attempt"
end

# The artifact names that stand in the place of `name` without meeting the requirement: the reports
# of a run below the full tier, and those of a job whose tests did not succeed.
function lookalikes(name)
  rest = chopprefix(name, "conformance-reports-")
  return ["conformance-reports-partial-" * rest, "diagnostic-conformance-reports-" * rest,
          "diagnostic-conformance-reports-partial-" * rest]
end

# Tier none (no package tests) is planned for a pull request and for a push to master only.
none_allowed(event, ref_type) = event == "pull_request" || (event == "push" && ref_type == "branch")

splitnames(text) = String[n for n in split(text, ';') if !isempty(strip(n))]

isjulia(name) = startswith(name, "Julia ")

"""
    verdict(; event, ref_type, plan, tests, analysis, benchmark, run_benchmark, run_tests, tier,
            platforms, required, advisory, jobs, release = false, expected_sha = "", commit = "",
            artifacts = nothing) -> (; errors, notes, failed, linux_failed)

The decision. `plan`, `tests`, `analysis` and `benchmark` are the results of the jobs
(`success`, `failure`, `cancelled`, `skipped`), `run_benchmark`, `run_tests`, `tier`, `platforms`,
`required` and `advisory` the plan's outputs (strings, possibly empty), and `jobs` this run's test jobs as
`(name, conclusion)` pairs, or `nothing` when they could not be listed. A job may carry a third
element, the attempt of the run it ran in, which the check of the conformance reports needs.
`release` says that the run is a release validation: it needs the floor of the cross jobs and an
`expected_sha` that is the full lowercase SHA `commit`. `artifacts` are the names of this run's
artifacts, or `nothing` when they could not be listed; the nightly run, a tag push and a release
run need each succeeded required test job's reports among them (see [`report_artifact`](@ref)), the
other runs ignore them. The run passes exactly when `errors` is empty. `notes` are lines to print,
`failed` the names of the failed jobs and `linux_failed` whether a failure is anything but a failed
test job that is not on Linux.
"""
function verdict(; event, ref_type = "", plan, tests, analysis, benchmark, run_benchmark, run_tests, tier,
                 platforms, required, advisory, jobs, release = false, expected_sha = "", commit = "",
                 artifacts = nothing)
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
  if release
    if match(COMMIT_PATTERN, expected_sha) === nothing
      problem("A release run needs expected_sha, the full 40-character lowercase SHA of the commit; got '$(first(expected_sha, 60))'.")
    elseif expected_sha != commit
      problem("The release run is of commit '$(first(commit, 60))', not the expected commit $expected_sha.")
    end
  end
  if plan == "success"
    if !(tier == "none" || tier in TIERS)
      problem("The plan gave no valid tier ('$tier').")
    elseif tier == "none"
      none_allowed(event, ref_type) ||
        problem("Tier none is planned for a pull request or a push to master only, not for a $event run.")
      release && problem("A release run is a manual run of the full tier, not of tier none.")
      run_tests == "false" || problem("Tier none, but run_tests is '$run_tests'.")
      tests == "skipped" || problem("Tier none, but the test jobs are $tests.")
      isempty(splitnames(required)) && isempty(splitnames(advisory)) || problem("Tier none, but the plan lists test jobs.")
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
      allowed = jobname(ADVISORY_JOB..., tier)
      for name in unique(adv)
        name == allowed || problem("`$name` is listed as advisory, but only `$allowed` may be.")
      end
      floor = try
        floor_jobs(event, ref_type, tier; release)
      catch err
        problem(first(sprint(showerror, err), 300))
        String[]
      end
      mode = try
        mode_jobs(platforms, tier)
      catch err
        problem(first(sprint(showerror, err), 300))
        String[]
      end
      for name in unique([floor; mode])
        name in req || problem("The required jobs do not include `$name`, which a $event run on the platforms `$platforms` needs.")
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
        if needs_reports(event, ref_type, release)
          attempts(name) = [length(j) > 2 ? j[3] : "" for j in jobs if j[1] == name]
          for name in unique(req)
            conclusions(name) == ["success"] || continue # a job without a verdict of its own is reported above
            linux = occursin("ubuntu", name)
            attempt = only(attempts(name))
            if isempty(attempt)
              problem("$name: the attempt it ran in is not known, so its conformance reports cannot be found."; job = name, linux)
              continue
            end
            wanted = try
              report_artifact(name, attempt)
            catch err
              problem(first(sprint(showerror, err), 300); job = name, linux)
              continue
            end
            if artifacts === nothing
              problem("$name: the artifacts of this run could not be listed, so `$wanted` is not confirmed."; job = name, linux)
            elseif !(wanted in artifacts)
              others = [a for a in lookalikes(wanted) if a in artifacts]
              problem("$name: the conformance reports `$wanted` were not uploaded" *
                      (isempty(others) ? "." : " (the run has $(join(map(a -> "`$a`", others), ", ")) instead)."); job = name, linux)
            end
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

# `(name, conclusion, attempt)` for each line of the job list.
function parse_jobs(text::AbstractString)
  jobs = Tuple{String, String, String}[]
  for line in split(text, '\n')
    isempty(strip(line)) && continue
    f = split(line, '\t')
    length(f) == 3 || error("unexpected job line: $(repr(line))")
    push!(jobs, (String(f[1]), String(f[2]), String(f[3])))
  end
  return jobs
end

parse_artifacts(text::AbstractString) = String[String(strip(l)) for l in split(text, '\n') if !isempty(strip(l))]

# The names of all artifacts of the run, across the pages of the listing.
function github_artifacts(repo, run_id)
  text = gh(["repos/$repo/actions/runs/$run_id/artifacts?per_page=100", "--paginate", "--jq", ARTIFACTS_JQ])
  return parse_artifacts(text)
end

# The latest attempt of each job, so a re-run of failed jobs counts.
function github_jobs(repo, run_id)
  text = gh(["repos/$repo/actions/runs/$run_id/jobs?filter=latest&per_page=100", "--paginate", "--jq", JOBS_JQ])
  return parse_jobs(text)
end

# Call `fetch()` until it returns; `nothing` after `attempts` failures.
function retried(fetch, what; attempts = 4, wait = 15, sleep = Base.sleep, log = println)
  for attempt in 1:attempts
    try
      return fetch()
    catch err
      err isa InterruptException && rethrow()
      log("::warning::Could not list the $what (attempt $attempt): ", first(sprint(showerror, err), 300))
      attempt < attempts && sleep(wait)
    end
  end
  return nothing
end

"""
    list_jobs(fetch; attempts = 4, wait = 15, sleep = Base.sleep, log = println) -> jobs or nothing

Call `fetch()` until it returns the job list; `nothing` after `attempts` failures, which the
verdict treats as a failure, not as a pass.
"""
list_jobs(fetch; kw...) = retried(fetch, "jobs"; kw...)

"""
    list_artifacts(fetch; attempts = 4, wait = 15, sleep = Base.sleep, log = println) -> names or nothing

The same for the names of the run's artifacts.
"""
list_artifacts(fetch; kw...) = retried(fetch, "artifacts"; kw...)

"""
    main(env = ENV; fetch, sleep, log) -> exit status

Reads PLAN, TESTS, ANALYSIS, BENCHMARK (the results of the jobs), RUN_BENCHMARK, RUN_TESTS, TIER,
FOCUS, PLATFORMS, REQUIRED and ADVISORY (the plan's outputs), RELEASE (the plan's `release` output:
`true`, `false` or empty), EXPECTED_SHA (the release run's input), EVENT, REF_TYPE, COMMIT,
GITHUB_REPOSITORY, GITHUB_RUN_ID and GH_TOKEN; prints the decision; appends `failed` and
`linux_failed` to the file named by GITHUB_OUTPUT. Returns 0 when the run passes, 1 when it does
not. `fetch` and `fetch_artifacts` list the run's jobs and artifacts.
"""
function main(env::AbstractDict = ENV;
              fetch = () -> github_jobs(env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"]),
              fetch_artifacts = () -> github_artifacts(env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"]),
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
  decided = get_("PLAN") == "success" && (tier == "none" || tier in TIERS)
  jobs = decided ? list_jobs(fetch; sleep, log) : nothing
  result = try
    release = get_("RELEASE")
    release in ("", "true", "false") || error("RELEASE must be true, false or empty, not '$(first(release, 60))'")
    release = release == "true"
    reports = decided && tier != "none" && needs_reports(get_("EVENT"), get_("REF_TYPE"), release)
    artifacts = reports ? list_artifacts(fetch_artifacts; sleep, log) : nothing
    verdict(; event = get_("EVENT"), ref_type = get_("REF_TYPE"), plan = get_("PLAN"), tests = get_("TESTS"),
            analysis = get_("ANALYSIS"), benchmark = get_("BENCHMARK"), run_benchmark = get_("RUN_BENCHMARK"),
            run_tests = get_("RUN_TESTS"), tier, platforms = get_("PLATFORMS"), required = get_("REQUIRED"), advisory = get_("ADVISORY"), jobs,
            release, expected_sha = get_("EXPECTED_SHA"), commit = get_("COMMIT"), artifacts)
  catch err
    err isa InterruptException && rethrow()
    (; errors = ["The verdict could not be decided: " * first(sprint(showerror, err), 300)], notes = String[],
       failed = String[], linux_failed = true)
  end
  foreach(n -> log(n), result.notes)
  foreach(e -> log("::error::", e), result.errors)
  isempty(result.errors) && log("No problem found.")
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
