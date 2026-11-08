# The release record of a release validation run (the `release-record` job of the CI workflow): a
# Markdown text with the verdict of the run, the exact commit, the jobs and artifacts of the run
# and a section for the decisions of a maintainer.
#
#   julia scripts/release_record.jl OUTPUT.md
#
# The record says `Release validation: PASSED` only when
#
# - the jobs `plan`, `test`, `verdict` and `gate` of the run all succeeded (PLAN, TESTS, VERDICT
#   and GATE hold their results);
# - the run is a manual run (GITHUB_EVENT_NAME is workflow_dispatch) of the commit it was asked to
#   validate: EXPECTED_SHA is the full lowercase 40-character SHA and equals GITHUB_SHA;
# - the jobs of the run could be listed, the four required test jobs of the cross platforms (the
#   verdict's own list, `CIVerdict.mode_jobs`) are each listed once with the conclusion success,
#   and the artifacts of the run could be listed.
#
# Anything else gives `Release validation: FAILED` and the reasons, never a PASSED: a listing that
# cannot be read gives a record that says so. Julia pre is advisory: the record lists its
# conclusion and a failure of it does not change the heading. The record also holds the ref, the
# run URL and attempt, the commit of the workflow file and whether the CI workflow, its rules, the
# check scripts and the test selection files differ from the default branch (WORKFLOW_DIFFERS:
# true, false or unknown; the files are listed in the step that sets it), the version in
# Project.toml of the checkout, a table of the latest attempt of every job, the names of the run's
# artifacts, and, last, the line that known exceptions are decided by a maintainer, followed by the
# section "Maintainer decisions", which is empty.
#
# `build_record` is pure (data in, text out). `main` reads the environment, lists the jobs and the
# artifacts of the run through `gh api` (the functions of scripts/ci_verdict.jl, retried), writes the
# record to OUTPUT and appends it to the file named by GITHUB_STEP_SUMMARY when that is set. Exit
# status: 0 when the record was written (a FAILED record is written like a PASSED one: the
# conclusion of the run is the verdict's, and the record is its evidence), 1 when it could not be
# written or appended, 2 for a usage error or an operational variable that fails the checks before
# anything is written: GITHUB_SHA, PLAN, TESTS, VERDICT and GATE must be non-empty (their format is
# not checked here), GITHUB_REPOSITORY must be owner/name, GITHUB_RUN_ID and GITHUB_RUN_ATTEMPT
# (empty means 1) numbers, and GITHUB_SERVER_URL (empty means https://github.com) an http(s) URL.
# Everything else gives a record: an EXPECTED_SHA that is not a full lowercase SHA, a GITHUB_SHA
# that is not one, and a result that is not success, failure, cancelled or skipped (shown as
# unknown) give a FAILED record and exit status 0. The job only
# runs when the plan's `release` output is true, so a dispatch that the planner rejected has no
# release-record job and no record.
#
# The environment: GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT,
# GITHUB_EVENT_NAME, GITHUB_REF, GITHUB_SERVER_URL, GITHUB_STEP_SUMMARY, GH_TOKEN, PLAN, TESTS,
# VERDICT, GATE, EXPECTED_SHA, WORKFLOW_DIFFERS and WORKFLOW_SHA. Text that comes from an input of
# the run is reduced to a safe set of characters before it is written (see `clean`). No packages
# are needed.

module ReleaseRecord

using TOML

include(joinpath(@__DIR__, "ci_verdict.jl"))

const ROOT = dirname(@__DIR__)
const USAGE = "usage: julia scripts/release_record.jl OUTPUT.md"

const HEADING = "Release validation"
const EXCEPTIONS_LINE = "Known exceptions are recorded by the maintainer below; the workflow has none."
const DECISIONS_HEADING = "Maintainer decisions"

# The name of this job, which is still running while it lists the jobs.
const RECORD_JOB = "Release record"

# The results GitHub gives a job in `needs.<job>.result`.
const RESULTS = ("success", "failure", "cancelled", "skipped")

"""
`s` reduced to letters, digits and the marks `._:/ ,()+-` (every other character becomes `?`), at
most `n` characters. This prevents markup, mention (`@`) and reference (`#`) injection and breaking
out of a code span or a table cell; a plain URL is kept, and GitHub may still link it.
"""
clean(s; n = 80) = first(replace(String(s), r"[^0-9A-Za-z._:/ ,()+-]" => "?"), n)

"The result of a job as the record shows it: one of GitHub's results, or `unknown`."
result_text(s) = String(s) in RESULTS ? String(s) : "unknown"

"The names of the required test jobs: the cross platforms at the full tier."
required_names() = CIVerdict.mode_jobs("cross", "full")

"The name of the advisory test job (Julia pre) at the full tier."
advisory_name() = CIVerdict.jobname(CIVerdict.ADVISORY_JOB..., "full")

function differs_text(value)
  value == "false" && return "The CI workflow, its rules, the check scripts and the test selection files are the same as on the default branch."
  value == "true" && return "The CI workflow, its rules, the check scripts or the test selection files differ from the default branch: this run used the files of its own ref."
  return "Whether the CI workflow, its rules, the check scripts and the test selection files differ from the default branch is unknown (the comparison did not run)."
end

# A table cell: no pipes, no line breaks.
cell(s) = replace(clean(s; n = 120), "|" => "/")

function job_rows(jobs)
  required, advisory = required_names(), advisory_name()
  rows = String[]
  for (name, conclusion, attempt) in jobs
    role = name in required ? "required" : name == advisory ? "advisory" : ""
    shown = isempty(conclusion) ? (name == RECORD_JOB ? "in progress (this job)" : "not finished") : conclusion
    push!(rows, "| $(cell(name)) | $role | $(cell(shown)) | $(cell(attempt)) |")
  end
  return rows
end

"""
    build_record(; plan, tests, verdict, gate, expected_sha, sha, event, ref, run_url, attempt,
                 workflow_sha, workflow_differs, version, jobs, artifacts) -> (; passed, reasons, text)

The release record as Markdown. `plan`, `tests`, `verdict` and `gate` are the results of the jobs,
`jobs` the latest attempt of each job of the run as `(name, conclusion, attempt)` and `artifacts` the
names of its artifacts, each `nothing` when it could not be listed. `passed` is true exactly when
`reasons` is empty; see the top of the file for the conditions.
"""
function build_record(; plan, tests, verdict, gate, expected_sha, sha, event = "workflow_dispatch", ref = "", run_url = "",
                      attempt = "", workflow_sha = "", workflow_differs = "unknown", version = "unknown", jobs, artifacts)
  reasons = String[]
  for (what, result) in (("The test plan", plan), ("The test jobs", tests), ("The required jobs check", verdict), ("The result gate", gate))
    result == "success" || push!(reasons, "$what: $(result_text(result)).")
  end
  event == "workflow_dispatch" || push!(reasons, "The run is a `$(clean(event))` run, not a manual release run.")
  if match(CIVerdict.COMMIT_PATTERN, String(sha)) === nothing
    push!(reasons, "The commit of the run is not a full 40-character lowercase SHA.")
  elseif match(CIVerdict.COMMIT_PATTERN, String(expected_sha)) === nothing
    push!(reasons, "expected_sha is not the full 40-character lowercase SHA of a commit (got `$(clean(expected_sha))`).")
  elseif expected_sha != sha
    push!(reasons, "The run is of commit `$sha`, not the expected commit `$expected_sha`.")
  end
  if jobs === nothing
    push!(reasons, "The jobs of the run could not be listed.")
  else
    for name in required_names()
      cs = [c for (n, c, _) in jobs if n == name]
      if isempty(cs)
        push!(reasons, "$name: missing.")
      elseif length(cs) > 1
        push!(reasons, "$name: listed $(length(cs)) times.")
      elseif only(cs) != "success"
        push!(reasons, "$name: $(isempty(only(cs)) ? "not finished" : only(cs)).")
      end
    end
  end
  artifacts === nothing && push!(reasons, "The artifacts of the run could not be listed.")
  passed = isempty(reasons)

  out = String[]
  push!(out, "# $HEADING: $(passed ? "PASSED" : "FAILED")", "")
  if !passed
    push!(out, "Reasons:", "")
    append!(out, ["- $r" for r in reasons])
    push!(out, "")
  end
  push!(out, "| Item | Value |", "|---|---|")
  push!(out, "| Commit | `$(clean(sha; n = 64))` |")
  push!(out, "| Expected commit | `$(clean(expected_sha; n = 64))` |")
  push!(out, "| Ref | `$(clean(ref))` |")
  push!(out, "| Run | $(clean(run_url; n = 200)) (attempt $(clean(attempt))) |")
  push!(out, "| Workflow file commit | `$(clean(workflow_sha; n = 64))` |")
  push!(out, "| Package version (Project.toml of the checkout) | `$(clean(version))` |")
  push!(out, "")
  push!(out, differs_text(workflow_differs), "")
  push!(out, "## Results", "")
  for (what, result) in (("Test plan", plan), ("Test jobs", tests), ("Required jobs check", verdict), ("Result gate", gate))
    push!(out, "- $what: $(result_text(result))")
  end
  push!(out, "", "## Jobs", "")
  push!(out, "Latest attempt of each job of the run. Julia pre is advisory: it is reported and not counted.", "")
  if jobs === nothing
    push!(out, "The jobs of the run could not be listed.")
  else
    push!(out, "| Job | Role | Conclusion | Attempt |", "|---|---|---|---|")
    append!(out, job_rows(jobs))
  end
  push!(out, "", "## Artifacts", "")
  if artifacts === nothing
    push!(out, "The artifacts of the run could not be listed.")
  elseif isempty(artifacts)
    push!(out, "The run has no artifacts.")
  else
    append!(out, ["- `$(clean(a; n = 120))`" for a in sort(artifacts)])
  end
  push!(out, "", EXCEPTIONS_LINE, "", "## $DECISIONS_HEADING", "")
  return (; passed, reasons, text = join(out, "\n") * "\n")
end

"The version of the package at `path` (a Project.toml), or `unknown`."
function project_version(path)
  try
    return String(get(TOML.parsefile(path), "version", "unknown"))
  catch err
    err isa InterruptException && rethrow()
    return "unknown"
  end
end

"""
    main(args, env; fetch_jobs, fetch_artifacts, project, sleep, log, err) -> exit status

Writes the record to the file named by `args[1]` and appends it to GITHUB_STEP_SUMMARY when that
is set. 0 when it was written, whatever it says (an invalid `EXPECTED_SHA`, a non-empty but
malformed `GITHUB_SHA` and an unrecognized result such as `PLAN=bogus` give a FAILED record);
1 when it could not be written; 2 for a usage error, a missing or empty `GITHUB_SHA`, `PLAN`,
`TESTS`, `VERDICT` or `GATE`, or a malformed `GITHUB_REPOSITORY`, `GITHUB_RUN_ID`,
`GITHUB_RUN_ATTEMPT` or `GITHUB_SERVER_URL`.
"""
function main(args = ARGS, env = ENV;
              fetch_jobs = () -> CIVerdict.github_jobs(env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"]),
              fetch_artifacts = () -> CIVerdict.github_artifacts(env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"]),
              project = joinpath(ROOT, "Project.toml"), sleep = Base.sleep, log = println, err::IO = stderr)
  if length(args) != 1 || startswith(args[1], "--")
    println(err, USAGE)
    return 2
  end
  get_(k) = String(strip(get(env, k, "")))
  problems = String[]
  repo, run_id = get_("GITHUB_REPOSITORY"), get_("GITHUB_RUN_ID")
  occursin(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo) || push!(problems, "GITHUB_REPOSITORY is not set to owner/name")
  occursin(r"^\d+$", run_id) || push!(problems, "GITHUB_RUN_ID is not set to a number")
  isempty(get_("GITHUB_SHA")) && push!(problems, "GITHUB_SHA is not set")
  for key in ("PLAN", "TESTS", "VERDICT", "GATE")
    isempty(get_(key)) && push!(problems, "$key is not set (the result of the job)")
  end
  attempt = isempty(get_("GITHUB_RUN_ATTEMPT")) ? "1" : get_("GITHUB_RUN_ATTEMPT")
  occursin(r"^\d+$", attempt) || push!(problems, "GITHUB_RUN_ATTEMPT is not a number")
  server = rstrip(isempty(get_("GITHUB_SERVER_URL")) ? "https://github.com" : get_("GITHUB_SERVER_URL"), '/')
  occursin(r"^https?://[A-Za-z0-9.:-]+$", server) || push!(problems, "GITHUB_SERVER_URL is not a URL")
  if !isempty(problems)
    println(err, "scripts/release_record.jl: ", join(problems, "; "), "\n", USAGE)
    return 2
  end
  jobs = CIVerdict.list_jobs(fetch_jobs; sleep, log)
  artifacts = CIVerdict.list_artifacts(fetch_artifacts; sleep, log)
  record = build_record(; plan = get_("PLAN"), tests = get_("TESTS"), verdict = get_("VERDICT"), gate = get_("GATE"),
                        expected_sha = get_("EXPECTED_SHA"), sha = get_("GITHUB_SHA"), event = get_("GITHUB_EVENT_NAME"),
                        ref = get_("GITHUB_REF"), run_url = "$server/$repo/actions/runs/$run_id", attempt,
                        workflow_sha = get_("WORKFLOW_SHA"), workflow_differs = get_("WORKFLOW_DIFFERS"),
                        version = project_version(project), jobs, artifacts)
  try
    mkpath(dirname(abspath(args[1])))
    write(args[1], record.text)
    summary = get_("GITHUB_STEP_SUMMARY")
    isempty(summary) || open(io -> write(io, record.text), summary, "a")
  catch e
    e isa InterruptException && rethrow()
    log("::error::The release record could not be written: ", first(sprint(showerror, e), 300))
    return 1
  end
  log("Release validation: ", record.passed ? "PASSED" : "FAILED", " (record written to ", args[1], ")")
  foreach(r -> log("::notice::", r), record.reasons)
  return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
  exit(ReleaseRecord.main())
end
