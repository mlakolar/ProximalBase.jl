# Turn the result of a CI run into GitHub issue actions.
#
#   julia scripts/ci_triage.jl <alert|triage|stale> [--dry-run] [--rehearsal]
#
# alert   a failed push to master: the issue "CI failing on master" (one open issue; a comment on
#         a repeat), with a line per failed test file and test set when the job's log is readable.
# triage  every nightly run, green or red: one issue per system and test file (`<os>/<file>`, or
#         `<os>/job:<alias>` for a job that failed without a test failure), labelled with a
#         severity (core: a Linux job on Julia lts or 1 and a file that is not on the deferred
#         list; deferred: everything else) and a kind (unclassified, or infrastructure when the
#         job failed outside the test step, was cancelled, has no log or is missing; a maintainer
#         sets bug, flaky or portability). A repeat is a comment, once per run; a closed issue
#         that recurs is reopened (a comment only when it was closed as not planned); an open
#         issue is closed with a comment once every Julia version recorded for it has run the
#         file and passed it.
# stale   no completed scheduled run (the nightly) created in the last `stale_hours`: the issue
#         "Nightly CI is stale". Manual, release and tag runs do not count.
#
# The script reads everything first (the run's jobs, the logs of the jobs that did not succeed, the
# labelled issues and their comments), builds the list of actions from that data alone (the
# `plan_*` functions are pure), and only then writes through `apply!`. An API error while reading
# stops the run with exit status 1 and writes nothing, and a resolution is never reported on data
# that could not be read; the one exception is `alert`, where a failed read of the jobs or of a
# log only drops the lines per failed file and the alert is still sent (its reads of the issues
# and of their comments, like those of every other mode, are mandatory). A write that fails is
# reported and the rest still run, and the exit status is then 1. A run can be repeated (a rerun
# of failed jobs, or this script again after a failed write) because the plan is made again from
# the current state of GitHub: a comment is written once per run (the hidden marker `run:`,
# `resolved:` or `passed:` with the run id in the issue text), while a reopen, a close and a
# severity label are planned whenever the issue still needs them, whether or not their comment is
# already there. Exit status 2 is a usage, environment or policy error.
#
# --dry-run prints the actions and writes nothing. --rehearsal uses the label `rehearsal_label`
# instead of `label`, prefixes every title with "[rehearsal] " and matches only issues that carry
# the rehearsal label and not `label` and whose title has the prefix (a normal run matches only
# issues with `label` and not `rehearsal_label`), so it can never touch a real triage issue; an
# issue with both labels is skipped in both modes, with a warning.
#
# The policy (which files are deferred, the labels and their colors, the titles, `stale_hours`) is
# the [triage] table of .github/tier-rules.toml, validated here. The environment holds the only
# input from the workflow: GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SHA,
# GITHUB_SERVER_URL, GITHUB_EVENT_NAME, REQUIRED and ADVISORY (the plan's job lists), FAILED and
# LINUX_FAILED (the verdict's outputs) and GH_TOKEN; nothing is read from the event payload.
# Everything on GitHub goes through `gh` with an argument vector (the seam `gh`, replaced by a
# fake in test/ci_triage.jl); text for an issue is passed in a file. No packages are needed.

module CITriage

using TOML

include(joinpath(@__DIR__, "ci_result.jl"))
include(joinpath(@__DIR__, "ci_verdict.jl"))
using .CIResult: parse_log, ParsedLog, error_line, scrub

const ROOT = dirname(@__DIR__)
const RULES = joinpath(ROOT, ".github", "tier-rules.toml")

# The step of a test job that runs the test files (a failure in another step is not a test failure).
const TEST_STEP = "Run package tests"

const UNCLASSIFIED = "triage:unclassified"
const INFRASTRUCTURE = "triage:infrastructure"
const FLAKY = "triage:flaky"

const MODES = ("alert", "triage", "stale")
# The event each mode runs for; anything else is refused.
const EVENTS = Dict("alert" => "push", "triage" => "schedule", "stale" => "push")
const USAGE = "usage: julia scripts/ci_triage.jl <alert|triage|stale> [--dry-run] [--rehearsal]"

const ZWSP = "\u200b"
const REHEARSAL_PREFIX = "[rehearsal] "
const MAX_TITLE = 200
const MAX_BODY = 60000
const MAX_FIELD = 300      # characters of a name or signature in an issue text
const MAX_EXCERPT = 3000   # characters of an excerpt
const MAX_RECORDS = 12     # failures described in one issue text
const CUT_NOTE = "\n\n_The text was cut here to fit the size limit of an issue._"

const CORE_TEXT = "A core CPU correctness failure on Linux: fix promptly; if a recent merge caused it, consider reverting it."
const DEFERRED_TEXT = "Normally deferred: triage by impact (see CONTRIBUTING.md); this does not block pull requests or master."

# ---------------------------------------------------------------------------------------------
# Policy: the [triage] table

const POLICY_KEYS = ("deferred", "stale_hours", "title_alert", "title_stale", "label", "rehearsal_label",
                     "severity_labels", "kind_labels", "label_style")

"A group of test files that are normally deferred: `why` and the globs over test names."
struct Deferred
  why::String
  files::Vector{String}
  regexes::Vector{Regex}
end

"The validated [triage] table. `styles` maps a label to its color and description."
struct Policy
  deferred::Vector{Deferred}
  stale_hours::Int
  title_alert::String
  title_stale::String
  label::String
  rehearsal_label::String
  severity_labels::Dict{String, String}
  kind_labels::Vector{String}
  styles::Dict{String, Tuple{String, String}}
end

struct PolicyError <: Exception
  problems::Vector{String}
end

function Base.showerror(io::IO, e::PolicyError)
  print(io, "the [triage] table of .github/tier-rules.toml has problems:")
  foreach(p -> print(io, "\n  - ", p), e.problems)
end

function text_key(table, key, problems)
  v = get(table, key, nothing)
  v === nothing && return "" # reported as missing
  v isa AbstractString && !isempty(strip(v)) && !occursin(r"[\r\n]", v) && return String(v)
  push!(problems, "triage.$key must be a non-empty string of one line")
  return ""
end

"""
    parse_policy(data) -> Policy

Validate the parsed rules file (`data["triage"]`). Every problem is listed at once in a
[`PolicyError`](@ref): an unknown key, a missing key, a value of the wrong type, a label without a
color and description. Throws for a table that is missing.
"""
function parse_policy(data::AbstractDict)
  table = get(data, "triage", nothing)
  table isa AbstractDict || throw(PolicyError(["[triage] is missing"]))
  problems = String[]
  for k in sort!(collect(keys(table)))
    k in POLICY_KEYS || push!(problems, "triage has an unknown key `$k`; the keys are $(join(POLICY_KEYS, ", "))")
  end
  for k in POLICY_KEYS
    haskey(table, k) || push!(problems, "triage.$k is missing")
  end

  deferred = Deferred[]
  list = get(table, "deferred", nothing)
  if list !== nothing
    if !(list isa AbstractVector)
      push!(problems, "triage.deferred must be a list of tables ([[triage.deferred]])")
    else
      for (i, entry) in enumerate(list)
        at = "triage.deferred[$i]"
        if !(entry isa AbstractDict)
          push!(problems, "$at must be a table")
          continue
        end
        for k in sort!(collect(keys(entry)))
          k in ("why", "files") || push!(problems, "$at has an unknown key `$k`; the keys are why, files")
        end
        why = get(entry, "why", nothing)
        why isa AbstractString && !isempty(strip(why)) || push!(problems, "$at needs a non-empty string `why`")
        files = get(entry, "files", nothing)
        if !(files isa AbstractVector) || isempty(files) ||
           !all(f -> f isa AbstractString && !isempty(strip(f)), files)
          push!(problems, "$at needs `files`, a non-empty list of test-name globs")
          continue
        end
        why isa AbstractString || continue
        names = String.(files)
        push!(deferred, Deferred(String(why), names, map(CIResult.TestTiers.glob_regex, names)))
      end
    end
  end

  hours = get(table, "stale_hours", nothing)
  stale_hours = 0
  if hours !== nothing
    hours isa Integer && !(hours isa Bool) && hours > 0 ? (stale_hours = Int(hours)) :
      push!(problems, "triage.stale_hours must be a positive number of hours")
  end
  title_alert = text_key(table, "title_alert", problems)
  title_stale = text_key(table, "title_stale", problems)
  label = text_key(table, "label", problems)
  rehearsal_label = text_key(table, "rehearsal_label", problems)
  isempty(title_alert) || title_alert != title_stale || push!(problems, "triage.title_alert and triage.title_stale must differ")
  isempty(label) || label != rehearsal_label || push!(problems, "triage.label and triage.rehearsal_label must differ")

  severity = Dict{String, String}()
  sv = get(table, "severity_labels", nothing)
  if sv !== nothing
    if !(sv isa AbstractDict)
      push!(problems, "triage.severity_labels must be a table with the keys core and deferred")
    else
      for k in sort!(collect(keys(sv)))
        k in ("core", "deferred") || push!(problems, "triage.severity_labels has an unknown key `$k`; the keys are core, deferred")
      end
      for k in ("core", "deferred")
        v = get(sv, k, nothing)
        v isa AbstractString && !isempty(strip(v)) ? (severity[k] = String(v)) :
          push!(problems, "triage.severity_labels.$k must be a non-empty string")
      end
    end
  end

  kinds = String[]
  kl = get(table, "kind_labels", nothing)
  if kl !== nothing
    if !(kl isa AbstractVector) || !all(k -> k isa AbstractString && !isempty(strip(k)), kl)
      push!(problems, "triage.kind_labels must be a list of label names")
    else
      kinds = String.(kl)
      length(unique(kinds)) == length(kinds) || push!(problems, "triage.kind_labels lists a label twice")
      for k in (UNCLASSIFIED, INFRASTRUCTURE, FLAKY)
        k in kinds || push!(problems, "triage.kind_labels must include `$k`, which the reporter uses")
      end
    end
  end

  styles = Dict{String, Tuple{String, String}}()
  ls = get(table, "label_style", nothing)
  if ls !== nothing
    if !(ls isa AbstractDict)
      push!(problems, "triage.label_style must be a table of label = { color, description }")
    else
      for name in sort!(collect(keys(ls)))
        entry = ls[name]
        at = "triage.label_style.$name"
        if !(entry isa AbstractDict)
          push!(problems, "$at must be a table with `color` and `description`")
          continue
        end
        for k in sort!(collect(keys(entry)))
          k in ("color", "description") || push!(problems, "$at has an unknown key `$k`; the keys are color, description")
        end
        color, description = get(entry, "color", nothing), get(entry, "description", nothing)
        color isa AbstractString && occursin(r"^[0-9a-fA-F]{6}$", color) ||
          push!(problems, "$at needs `color`, six hexadecimal digits without `#`")
        description isa AbstractString && 0 < length(description) <= 100 ||
          push!(problems, "$at needs `description`, a string of 1 to 100 characters")
        color isa AbstractString && description isa AbstractString && (styles[name] = (String(color), String(description)))
      end
      used = [label; rehearsal_label; collect(values(severity)); kinds]
      for name in unique(filter(!isempty, used))
        haskey(ls, name) || push!(problems, "triage.label_style has no entry for the label `$name`")
      end
      for name in sort!(collect(keys(ls)))
        name in used || push!(problems, "triage.label_style has the entry `$name`, which no key of triage uses as a label")
      end
    end
  end

  isempty(problems) || throw(PolicyError(problems))
  return Policy(deferred, stale_hours, title_alert, title_stale, label, rehearsal_label, severity, kinds, styles)
end

"Read and validate the [triage] table of the rules file."
load_policy(rules = RULES) = parse_policy(TOML.parsefile(rules))

"The first deferred group whose globs match the test file `file`, or `nothing`."
function deferred_group(policy::Policy, file::AbstractString)
  for d in policy.deferred
    any(r -> occursin(r, file), d.regexes) && return d
  end
  return nothing
end

is_deferred(policy::Policy, file::AbstractString) = deferred_group(policy, file) !== nothing

# ---------------------------------------------------------------------------------------------
# Names, severity and keys

# `Julia <alias> - <runner> - <arch> - <tier>`; the alias is lts, 1 or pre, never a concrete version.
function parse_jobname(name::AbstractString)
  parts = split(name, " - ")
  length(parts) == 4 && startswith(parts[1], "Julia ") || return nothing
  return (alias = String(chopprefix(parts[1], "Julia ")), runner = String(parts[2]), arch = String(parts[3]), tier = String(parts[4]))
end

# A name or file in a key or marker: no space, no `>` and nothing else that could end a comment.
safe_part(s::AbstractString) = String(replace(s, r"[^A-Za-z0-9_.+-]" => "_"))
safe_path(s::AbstractString) = String(replace(s, r"[^A-Za-z0-9_./+-]" => "_"))

function os_key(runner::AbstractString)
  r = lowercase(runner)
  startswith(r, "ubuntu") && return "linux"
  startswith(r, "macos") && return "macos"
  startswith(r, "windows") && return "windows"
  return safe_part(r)
end

const OS_NAMES = Dict("linux" => "Linux", "macos" => "macOS", "windows" => "Windows")
os_display(os) = get(OS_NAMES, os, os)

# (os, alias) of an expected job; a name that is not a test job's is its own alias on an unknown system.
function job_identity(name::AbstractString)
  p = parse_jobname(name)
  p === nothing && return ("unknown", safe_part(name))
  return (os_key(p.runner), safe_part(p.alias))
end

"""
    severity(policy, os, alias, file) -> :core or :deferred

Core only for a Linux job on Julia lts or 1 (not pre) and a test file that is not deferred. A record
with no file (an infrastructure failure of the job) takes the severity of its platform.
"""
function severity(policy::Policy, os, alias, file)
  os == "linux" || return :deferred
  file === nothing && return alias in ("lts", "1") ? :core : :deferred
  return alias != "pre" && !is_deferred(policy, file) ? :core : :deferred
end

# CRC-32C (Castagnoli), table driven; the CRC32c standard library is not among the test dependencies.
const CRC_TABLE = ntuple(256) do i
  c = UInt32(i - 1)
  for _ in 1:8
    c = (c & 0x1) != 0 ? (c >> 1) ⊻ 0x82f63b78 : c >> 1
  end
  c
end

function crc32c(data::AbstractVector{UInt8})
  crc = ~UInt32(0)
  for b in data
    crc = CRC_TABLE[Int((crc ⊻ b) & 0xff) + 1] ⊻ (crc >> 8)
  end
  return ~crc
end

"The hash of a signature in the markers `<!-- sig:<hash> -->`: its CRC-32C in hexadecimal."
sig_hash(signature::AbstractString) = string(crc32c(codeunits(signature)); base = 16)

# ---------------------------------------------------------------------------------------------
# Text that goes into an issue

"`@name` and `#123` would notify or link: a zero-width space after the `@` and the `#` breaks that; `<!--` would be read as a marker."
neutralize(s::AbstractString) = String(replace(s, "<!--" => "<!$ZWSP--", r"@(?=[A-Za-z0-9_])" => "@$ZWSP", r"#(?=[0-9])" => "#$ZWSP"))

oneline(s::AbstractString) = String(strip(replace(s, r"[\x00-\x1f\x7f\s]+" => " ")))

function shorten(s::AbstractString, limit::Int)
  length(s) <= limit && return String(s)
  return String(first(s, max(limit - 1, 0))) * "…"
end

longest_run(s) = maximum((length(m.match) for m in eachmatch(r"`+", s)); init = 0)

"`s` as inline code on one line, in backticks longer than any inside it."
function inline(s::AbstractString; limit = MAX_FIELD)
  t = neutralize(shorten(oneline(s), limit))
  isempty(t) && return "(none)"
  pad = startswith(t, "`") || endswith(t, "`") ? " " : ""
  ticks = "`"^(longest_run(t) + 1)
  return string(ticks, pad, t, pad, ticks)
end

"`text` in a fence longer than any run of backticks inside it (at least three), cut at `limit` characters."
function fenced(text::AbstractString; limit = MAX_EXCERPT)
  t = neutralize(rstrip(String(text)))
  if length(t) > limit
    t = first(t, limit) * "\n... (excerpt cut)"
  end
  fence = "`"^max(3, longest_run(t) + 1)
  return string(fence, "\n", t, "\n", fence)
end

"A title: one line of at most $MAX_TITLE characters, prefixed in a rehearsal."
function make_title(text::AbstractString; rehearsal = false)
  t = neutralize(oneline((rehearsal ? REHEARSAL_PREFIX : "") * text))
  return shorten(t, MAX_TITLE)
end

"`text` cut under `limit` characters, with a last line that says so."
function cap(text::AbstractString; limit = MAX_BODY)
  length(text) < limit && return String(text)
  return String(first(text, limit - length(CUT_NOTE) - 1)) * CUT_NOTE
end

marker(name, value) = "<!-- $name:$value -->"
marker_regex(name) = Regex("<!-- $name:([A-Za-z0-9._/-]+) -->")
markers_in(texts, name) = Set{String}(String(m.captures[1]) for t in texts for m in eachmatch(marker_regex(name), t))
has_marker(texts, name, value) = value in markers_in(texts, name)
const KEY_MARKER = r"<!-- nightly-triage key=(\S+) -->"

# ---------------------------------------------------------------------------------------------
# Data

struct Step
  number::Int
  name::String
  conclusion::String
end

"One job of the run: its id, name, conclusion (empty until it finished), run attempt, page and steps."
struct Job
  id::String
  name::String
  conclusion::String
  attempt::String
  url::String
  steps::Vector{Step}
end

"One failure of one job, ready for an issue. `kind` is the label kind (`:unclassified` or `:infrastructure`)."
struct Record
  key::String
  os::String
  alias::String
  file::Union{Nothing, String}
  testset::String
  kind::Symbol
  what::String
  signature::String
  excerpt::String
  detail::String
  severity::Symbol
  job::String
  job_url::String
  log_url::String
end

"What a job says about the test files: `:success`, `:log` (a recognized log of a failed job) or `:unknown`."
struct Facts
  state::Symbol
  log::Union{Nothing, ParsedLog}
end

struct Observation
  records::Vector{Record}
  facts::Dict{Tuple{String, String}, Facts}
end

struct Issue
  number::Int
  state::String
  reason::String
  title::String
  body::String
  labels::Vector{String}
end

"The identity of the run being reported."
struct Run
  repo::String
  id::String
  attempt::String
  sha::String
  server::String
end

run_url(r::Run) = "$(r.server)/$(r.repo)/actions/runs/$(r.id)"

"""
    Action

One write: `kind` is `:create` (an issue), `:comment`, `:reopen`, `:close` (as completed) or `:label`
(add `labels`). `number` is the issue, `nothing` for a new one; `key` says what the action is
about (for the report), `title` and `body` are the text.
"""
struct Action
  kind::Symbol
  number::Union{Nothing, Int}
  key::String
  title::String
  body::String
  labels::Vector{String}
end
Action(kind, number, key; title = "", body = "", labels = String[]) = Action(kind, number, key, title, body, labels)

# ---------------------------------------------------------------------------------------------
# Observing a run: records and facts from jobs and logs

first_step(job::Job, conclusions) = (i = findfirst(s -> s.conclusion in conclusions, job.steps); i === nothing ? nothing : job.steps[i])

function make_record(policy, name, job; file = nothing, testset = "", kind = :unclassified, what, signature, excerpt = "",
                     detail = "", step = nothing)
  os, alias = job_identity(name)
  key = file === nothing ? "$os/job:$alias" : "$os/$(safe_path(file))"
  url = job === nothing ? "" : job.url
  log = isempty(url) ? "" : (step === nothing ? url : "$url#step:$(step.number):1")
  return Record(key, os, alias, file, testset, kind, what, isempty(signature) ? what : signature, excerpt, detail,
                severity(policy, os, alias, file), name, url, log)
end

failure_what(kind) = kind == :test ? "failed test" : kind == :exception ? "error during a test" : "worker crash"

# One record per failing file in the log of a job whose test step failed.
function file_records(policy, name, job, parsed::ParsedLog, step)
  records = Record[]
  seen = Set{String}()
  for f in parsed.failures
    push!(seen, f.file)
    push!(records, make_record(policy, name, job; file = f.file, testset = f.testset, what = failure_what(f.kind),
                               signature = f.signature, excerpt = f.excerpt, step))
  end
  for file in sort!(collect(keys(parsed.status)))
    st = parsed.status[file]
    st in (:failed, :crashed, :unfinished) && !(file in seen) || continue
    what = st == :failed ? "failed without a readable failure block" :
           st == :crashed ? "worker crash" : "started and never finished"
    push!(records, make_record(policy, name, job; file, what, signature = what, step))
  end
  return records
end

function job_what(job, test_failed)
  job === nothing && return "job missing"
  c = job.conclusion
  c in ("cancelled", "timed_out") && return "job cancelled or timed out"
  c == "skipped" && return "job skipped"
  test_failed && return "test step failed, nothing readable in the log"
  c == "failure" && return "job failed outside the test step"
  return "job ended `$(isempty(c) ? "not finished" : c)`"
end

# The record of a job as a whole: `job` is `nothing` for a job that is not in the run (or is listed twice).
function job_record(policy, name, job, parsed, test_failed; listed = 0)
  if job === nothing
    detail = listed == 0 ? "The expected job is missing from the run." : "The expected job is listed $listed times in the run."
    return make_record(policy, name, nothing; kind = :infrastructure, what = "job missing", signature = "job missing", detail)
  end
  step = first_step(job, ("failure", "cancelled", "timed_out"))
  what = job_what(job, test_failed)
  concl = isempty(job.conclusion) ? "not finished" : job.conclusion
  detail = "The job ended with `$concl`" * (step === nothing ? "" : " at the step `$(step.name)`") * "."
  parsed === nothing && (detail *= " Its log could not be read.")
  if parsed !== nothing && parsed.recognized
    unfinished = sort!([f for (f, s) in parsed.status if s == :unfinished])
    isempty(unfinished) || (detail *= " Test files started and not finished when the log ends: $(join(first(unfinished, 12), ", ")).")
  end
  # nothing parseable after a failed test step is a test failure of unknown cause; every other
  # job-level failure (a missing log, a cancelled job, another step) is the infrastructure's
  kind = job.conclusion == "failure" && test_failed && parsed !== nothing ? :unclassified : :infrastructure
  signature = "job $(isempty(job.conclusion) ? "not finished" : job.conclusion)" * (step === nothing ? "" : " at step $(step.name)")
  return make_record(policy, name, job; kind, what, signature, detail, step)
end

"""
    observe(policy, jobs, expected, logs) -> Observation

The failure records of the expected jobs and what each job says about the files. `logs` maps a job id
to its text, or to `nothing` for a log that could not be found; a job whose id is not in it has no
readable log. A job that succeeded ran the whole catalog (the inventory step fails it otherwise).
"""
function observe(policy::Policy, jobs::Vector{Job}, expected::Vector{String}, logs)
  records = Record[]
  facts = Dict{Tuple{String, String}, Facts}()
  for name in expected
    id = job_identity(name)
    matches = [j for j in jobs if j.name == name]
    if length(matches) != 1
      facts[id] = Facts(:unknown, nothing)
      push!(records, job_record(policy, name, nothing, nothing, false; listed = length(matches)))
      continue
    end
    job = only(matches)
    if job.conclusion == "success"
      facts[id] = Facts(:success, nothing)
      continue
    end
    text = get(logs, job.id, nothing)
    parsed = text === nothing ? nothing : parse_log(text)
    readable = parsed !== nothing && parsed.recognized
    ti = findfirst(s -> s.name == TEST_STEP, job.steps)
    step = ti === nothing ? nothing : job.steps[ti]
    test_failed = step !== nothing && step.conclusion == "failure"
    facts[id] = job.conclusion == "failure" && readable ? Facts(:log, parsed) : Facts(:unknown, nothing)
    found = job.conclusion == "failure" && test_failed && readable ? file_records(policy, name, job, parsed, step) : Record[]
    if isempty(found)
      push!(records, job_record(policy, name, job, parsed, test_failed))
    else
      append!(records, found)
    end
  end
  return Observation(records, facts)
end

# ---------------------------------------------------------------------------------------------
# Issue texts

function describe(r::Record)
  io = IOBuffer()
  println(io, "**Julia ", r.alias, " on ", os_display(r.os), "**")
  r.file === nothing || println(io, "- Test file: ", inline(r.file))
  isempty(r.testset) || println(io, "- Test set: ", inline(r.testset))
  println(io, "- Kind of failure: ", r.what)
  println(io, "- Signature: ", inline(r.signature))
  isempty(r.detail) || println(io, "- Detail: ", neutralize(r.detail))
  println(io, "- Job: ", inline(r.job), isempty(r.job_url) ? "" : ": $(r.job_url)")
  isempty(r.log_url) || println(io, "- Log: ", r.log_url)
  isempty(r.excerpt) || println(io, "\n", fenced(r.excerpt))
  return String(take!(io))
end

function describe_all(records)
  shown = first(records, MAX_RECORDS)
  text = join(describe.(shown), "\n")
  length(records) > MAX_RECORDS && (text *= "\n... and $(length(records) - MAX_RECORDS) more failures, not listed here; see the run.\n")
  return text
end

function markers_for(run::Run, records)
  lines = [marker("run", run.id)]
  append!(lines, [marker("sig", h) for h in unique(sig_hash(r.signature) for r in records)])
  append!(lines, [marker("alias", a) for a in unique(r.alias for r in records)])
  return join(lines, "\n")
end

function issue_title(records)
  r = first(records)
  return r.file === nothing ? "Nightly job failure on $(os_display(r.os)) (Julia $(r.alias))" :
                              "Nightly failure on $(os_display(r.os)): $(r.file)"
end

function severity_text(policy, records, sev)
  sev == :core && return CORE_TEXT
  files = unique([r.file for r in records if r.file !== nothing])
  groups = unique([d.why for d in (deferred_group(policy, f) for f in files) if d !== nothing])
  extra = isempty(groups) ? "" : " This test file is on the deferred list: $(join(groups, "; "))."
  return DEFERRED_TEXT * extra
end

function create_body(policy, run::Run, key, records, sev)
  r = first(records)
  systems = join(unique(["$(os_display(x.os)), Julia $(x.alias)" for x in records]), "; ")
  io = IOBuffer()
  println(io, "<!-- nightly-triage key=", key, " -->")
  println(io, markers_for(run, records), "\n")
  println(io, "A nightly run failed on ", os_display(r.os), r.file === nothing ? " (the job as a whole)." : " in the test file $(inline(r.file)).", "\n")
  println(io, "- Run: ", run_url(run), " (attempt ", run.attempt, ")")
  println(io, "- Commit: ", run.sha)
  println(io, "- System: ", systems, "\n")
  println(io, describe_all(records))
  println(io, severity_text(policy, records, sev))
  return cap(String(take!(io)))
end

function comment_body(run::Run, records; intro, changed = false, escalated = false)
  io = IOBuffer()
  println(io, markers_for(run, records), "\n")
  println(io, intro, " Run: ", run_url(run), " (attempt ", run.attempt, "), commit ", run.sha, ".")
  changed && println(io, "\nSignature changed: the failure of tonight is not one recorded for this issue.")
  escalated && println(io, "\nSeverity raised to core: the failure now also occurs on Linux with Julia lts or 1.")
  println(io, "\n", describe_all(records))
  return cap(String(take!(io)))
end

# ---------------------------------------------------------------------------------------------
# Planning (pure)

issue_key(i::Issue) = (m = match(KEY_MARKER, i.body); m === nothing ? nothing : String(m.captures[1]))

texts(i::Issue, comments) = [i.body; get(comments, i.number, String[])]

active_label(policy, rehearsal) = rehearsal ? policy.rehearsal_label : policy.label

# The issue a failure of tonight belongs to: the open one (the lowest number), else the latest closed.
function pick(issues)
  sorted = sort(issues; by = i -> i.number)
  i = findfirst(x -> x.state == "open", sorted)
  return i !== nothing ? sorted[i] : isempty(sorted) ? nothing : last(sorted)
end

"""
    in_scope(policy, issue, rehearsal) -> Bool

Does the run of this mode own `issue`? A normal run owns the issues with the label and without the
rehearsal label; a rehearsal owns those with the rehearsal label, without the label and with a
title that starts with "[rehearsal] ". An issue with both labels belongs to neither.
"""
function in_scope(policy::Policy, i::Issue, rehearsal)
  normal, rehearsed = policy.label in i.labels, policy.rehearsal_label in i.labels
  normal && rehearsed && return false
  return rehearsal ? rehearsed && startswith(i.title, REHEARSAL_PREFIX) : normal
end

"The numbers of the issues that carry both labels, which no mode touches."
conflicting(policy::Policy, issues) = sort!([i.number for i in issues if policy.label in i.labels && policy.rehearsal_label in i.labels])

function key_index(policy, issues, rehearsal)
  index = Dict{String, Vector{Issue}}()
  for i in issues
    in_scope(policy, i, rehearsal) || continue
    k = issue_key(i)
    k === nothing && continue
    push!(get!(index, k, Issue[]), i)
  end
  return index
end

"""
    issues_to_read(policy, issues, failing; rehearsal = false) -> Vector{Issue}

The issues whose comments the plan needs: every open issue of the mode ([`in_scope`](@ref)) with a
key (resolution needs its recorded Julia versions) and a closed one whose key fails tonight
(`failing`, a set of keys).
"""
function issues_to_read(policy::Policy, issues, failing; rehearsal = false)
  return [i for i in issues if in_scope(policy, i, rehearsal) && (k = issue_key(i)) !== nothing && (i.state == "open" || k in failing)]
end

function group_records(records)
  d = Dict{String, Vector{Record}}()
  for r in records
    push!(get!(d, r.key, Record[]), r)
  end
  return d
end

# Did the job of (os, alias) run `file` and pass it (file `nothing`: did the job succeed)?
function passed(obs::Observation, os, alias, file)
  f = get(obs.facts, (os, alias), nothing)
  f === nothing && return false
  f.state == :success && return true
  (file === nothing || f.state != :log || f.log === nothing) && return false
  return file in f.log.files && get(f.log.status, file, :none) == :passed
end

"""
    plan_triage(policy, run, obs, issues, comments; rehearsal = false) -> Vector{Action}

The actions of a nightly run. `obs` is the [`observe`](@ref) result, `issues` the issues with the
triage label (state, text and labels), `comments` maps an issue number to the texts of its
comments. Only the issues of the mode are considered ([`in_scope`](@ref)), so a rehearsal and a
real run cannot see each other's issues.

The plan reconciles the state on every call: a comment is planned unless its run marker is
already in the issue (`run:`, `resolved:` or `passed:` with the run id), while a reopen, a close and
a severity label are planned whenever the issue still needs them. Planning again after a partial
failure therefore completes the work without repeating a comment.
"""
function plan_triage(policy::Policy, run::Run, obs::Observation, issues, comments; rehearsal = false)
  label = active_label(policy, rehearsal)
  index = key_index(policy, issues, rehearsal)
  failing = group_records(obs.records)
  actions = Action[]
  for key in sort!(collect(keys(failing)))
    records = failing[key]
    sev = any(r -> r.severity == :core, records) ? :core : :deferred
    sev_label = policy.severity_labels[String(sev)]
    kind_label = all(r -> r.kind == :infrastructure, records) ? INFRASTRUCTURE : UNCLASSIFIED
    issue = pick(get(index, key, Issue[]))
    title = make_title(issue_title(records); rehearsal)
    if issue === nothing
      push!(actions, Action(:create, nothing, key; title, body = create_body(policy, run, key, records, sev),
                            labels = [label, sev_label, kind_label]))
      continue
    end
    tx = texts(issue, comments)
    written = has_marker(tx, "run", run.id) # an earlier attempt of this run commented already
    known = markers_in(tx, "sig")
    changed = !isempty(known) && any(h -> !(h in known), (sig_hash(r.signature) for r in records))
    if issue.state == "closed" && issue.reason == "not_planned"
      written || push!(actions, Action(:comment, issue.number, key; title = issue.title,
                                       body = comment_body(run, records; intro = "The failure occurred again, but this issue was closed as not planned, so it stays closed.", changed)))
      continue
    end
    escalate = sev == :core && !(sev_label in issue.labels)
    if issue.state == "closed"
      push!(actions, Action(:reopen, issue.number, key; title = issue.title))
      intro = "The failure recurred after this issue was closed."
    else
      intro = "The failure occurred again."
    end
    written || push!(actions, Action(:comment, issue.number, key; title = issue.title, body = comment_body(run, records; intro, changed, escalated = escalate)))
    escalate && push!(actions, Action(:label, issue.number, key; title = issue.title, labels = [sev_label]))
  end
  for key in sort!(collect(keys(index)))
    haskey(failing, key) && continue
    os, rest = let p = split(key, '/'; limit = 2)
      length(p) == 2 ? (String(p[1]), String(p[2])) : (nothing, nothing)
    end
    os === nothing && continue
    file = startswith(rest, "job:") ? nothing : rest
    for issue in sort(index[key]; by = i -> i.number)
      issue.state == "open" || continue
      tx = texts(issue, comments)
      aliases = markers_in(tx, "alias")
      isempty(aliases) && continue
      all(a -> passed(obs, os, a, file), aliases) || continue
      what = file === nothing ? "the job" : inline(file)
      list = join(sort!(collect(aliases)), ", ")
      if FLAKY in issue.labels
        has_marker(tx, "passed", run.id) && continue
        push!(actions, Action(:comment, issue.number, key; title = issue.title,
                              body = "$(marker("passed", run.id))\n\nPassed tonight: $what passed with Julia $list. This issue is labelled flaky, so it stays open. Run: $(run_url(run)), commit $(run.sha)."))
      else
        has_marker(tx, "resolved", run.id) ||
          push!(actions, Action(:comment, issue.number, key; title = issue.title,
                                body = "$(marker("resolved", run.id))\n\nResolved at $(run.sha), run $(run_url(run)): $what passed with Julia $list, every version recorded for this issue."))
        push!(actions, Action(:close, issue.number, key; title = issue.title))
      end
    end
  end
  return actions
end

"The failed test files and test sets of each failed job whose log is readable: job name => [(file, test set)]."
function alert_details(failed::Vector{String}, jobs::Vector{Job}, logs)
  details = Dict{String, Vector{Tuple{String, String}}}()
  for name in failed
    for job in jobs
      job.name == name && job.conclusion != "success" || continue
      text = get(logs, job.id, nothing)
      text === nothing && continue
      parsed = parse_log(text)
      parsed.recognized || continue
      lines = unique([(f.file, f.testset) for f in parsed.failures])
      isempty(lines) || (details[name] = lines)
    end
  end
  return details
end

function alert_body(run::Run, event, failed, details, linux_failed)
  io = IOBuffer()
  println(io, marker("run", run.id), "\n")
  println(io, "The ", neutralize(event), " run ", run_url(run), " failed at ", run.sha, ".")
  if isempty(failed)
    println(io, "\nThe failed jobs are not known; see the run.")
  else
    println(io, "\nFailed jobs:")
    for name in failed
      println(io, "- ", neutralize(oneline(name)))
      for (file, testset) in first(get(details, name, Tuple{String, String}[]), MAX_RECORDS)
        println(io, "  - ", inline(file), isempty(testset) ? "" : ", test set " * inline(testset))
      end
    end
  end
  linux_failed == "true" && println(io, "\nA Linux job, or a check that is not a test job, failed.")
  linux_failed == "false" && println(io, "\nOnly jobs on other systems failed (no Linux job did).")
  println(io, "\nFix or revert promptly; see CONTRIBUTING.md.")
  return cap(String(take!(io)))
end

# The open issue with exactly this title (the lowest number).
function find_open(issues, title)
  hits = sort([i for i in issues if i.state == "open" && i.title == title]; by = i -> i.number)
  return isempty(hits) ? nothing : first(hits)
end

"""
    plan_alert(policy, run, event, failed, linux_failed, details, issues, comments; rehearsal = false)

The prompt alert of a failed push to master: the issue "CI failing on master" is created, or
commented on when it is open, once per run. `failed` are the names of the failed jobs (the
verdict's output), `details` the [`alert_details`](@ref), `issues` the open issues.
"""
function plan_alert(policy::Policy, run::Run, event, failed, linux_failed, details, issues, comments; rehearsal = false)
  title = make_title(policy.title_alert; rehearsal)
  body = alert_body(run, event, failed, details, linux_failed)
  issue = find_open(issues, title)
  issue === nothing && return [Action(:create, nothing, "alert"; title, body)]
  has_marker(texts(issue, comments), "run", run.id) && return Action[]
  return [Action(:comment, issue.number, "alert"; title, body)]
end

"""
    epoch(y, m, d, H = 0, M = 0, S = 0) -> seconds since 1970-01-01 00:00:00 UTC

The Gregorian calendar by days-from-civil (no standard library is needed for the few times here).
"""
function epoch(y, m, d, H = 0, M = 0, S = 0)
  y -= m <= 2
  era = fld(y, 400)
  yoe = y - era * 400
  doy = fld(153 * (m + (m > 2 ? -3 : 9)) + 2, 5) + d - 1
  doe = yoe * 365 + fld(yoe, 4) - fld(yoe, 100) + doy
  return (era * 146097 + doe - 719468) * 86400 + H * 3600 + M * 60 + S
end

# "2026-11-07 16:12" of a time in seconds since the epoch (UTC).
function format_time(t)
  days, rest = fldmod(t, 86400)
  z = days + 719468
  era = fld(z, 146097)
  doe = z - era * 146097
  yoe = fld(doe - fld(doe, 1460) + fld(doe, 36524) - fld(doe, 146096), 365)
  doy = doe - (365 * yoe + fld(yoe, 4) - fld(yoe, 100))
  mp = fld(5 * doy + 2, 153)
  d = doy - fld(153 * mp + 2, 5) + 1
  m = mp + (mp < 10 ? 3 : -9)
  y = yoe + era * 400 + (m <= 2)
  return string(lpad(y, 4, '0'), "-", lpad(m, 2, '0'), "-", lpad(d, 2, '0'), " ", lpad(fld(rest, 3600), 2, '0'), ":", lpad(fld(rest % 3600, 60), 2, '0'))
end
format_date(t) = format_time(t)[1:10]

function parse_time(s::AbstractString)
  m = match(r"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.\d+)?Z$", s)
  m === nothing && error("unexpected time (UTC, ending in Z, is expected): $(repr(s))")
  return epoch(parse.(Int, m.captures)...)
end

"A completed scheduled run: when it was created (seconds since the epoch, UTC) and how it ended."
struct Scheduled
  created::Int
  status::String
  conclusion::String
end

"""
    plan_stale(policy, run, runs, issues, comments, now; rehearsal = false)

No completed scheduled run (the nightly) created within `stale_hours` of `now` opens the issue
"Nightly CI is stale" or comments on the open one (once per UTC date); a recent one closes an open
stale issue, with a comment written once per date. Manual, release and tag runs are not among
`runs`. `issues` are the open issues.
"""
function plan_stale(policy::Policy, run::Run, runs, issues, comments, now::Integer; rehearsal = false)
  title = make_title(policy.title_stale; rehearsal)
  issue = find_open(issues, title)
  date = format_date(now)
  recent = [r for r in runs if r.status == "completed" && now - r.created <= 3600 * policy.stale_hours]
  if isempty(recent)
    newest = isempty(runs) ? "There is no completed scheduled run." :
             "The newest completed scheduled run was created at $(format_time(maximum(r.created for r in runs))) UTC."
    body = "$(marker("stale", date))\n\nNo completed scheduled run was created within the last $(policy.stale_hours) hours (checked on $date, by the run $(run_url(run)) at commit $(run.sha)). $newest\n\nThe scheduled run (the nightly) is the recurring full-platform run; manual, release and tag runs do not count here. Check the schedule of the workflow and whether scheduled runs are disabled."
    issue === nothing && return [Action(:create, nothing, "stale"; title, body)]
    has_marker(texts(issue, comments), "stale", date) && return Action[]
    return [Action(:comment, issue.number, "stale"; title, body)]
  end
  issue === nothing && return Action[]
  newest = maximum(r.created for r in recent)
  actions = Action[]
  has_marker(texts(issue, comments), "recovered", date) ||
    push!(actions, Action(:comment, issue.number, "stale"; title,
                          body = "$(marker("recovered", date))\n\nA completed scheduled run was created again within the last $(policy.stale_hours) hours (at $(format_time(newest)) UTC), so this issue is closed. Seen by the run $(run_url(run)) at commit $(run.sha)."))
  push!(actions, Action(:close, issue.number, "stale"; title))
  return actions
end

# ---------------------------------------------------------------------------------------------
# The seam: `gh`

"A failed `gh` call: the message and the HTTP status when `gh` reported one."
struct GhError <: Exception
  message::String
  status::Union{Nothing, Int}
end
Base.showerror(io::IO, e::GhError) = print(io, e.message)

"""
    gh(args; input = nothing) -> String

Run `gh args...` with a time limit (a hung request is an error) and return its standard output;
throws a [`GhError`](@ref) when it fails. `input` is text for its standard input. The tests replace
this function by one that answers from memory.
"""
function gh(args::Vector{String}; input = nothing)
  out, err = IOBuffer(), IOBuffer()
  p = run(pipeline(`gh $args`; stdin = input === nothing ? devnull : IOBuffer(input), stdout = out, stderr = err); wait = false)
  if timedwait(() -> process_exited(p), 120) === :timed_out
    kill(p)
    throw(GhError("gh $(first(args)) timed out after 120 s", nothing))
  end
  if !success(p)
    stderr_text = strip(String(take!(err)))
    m = match(r"HTTP (\d{3})", stderr_text)
    message = first(stderr_text * " " * strip(String(take!(out))), 300)
    throw(GhError("gh $(first(args)) failed: $message", m === nothing ? nothing : parse(Int, m.captures[1])))
  end
  return String(take!(out))
end

# One tab-separated line per record; text that can hold tabs and line breaks is base64.
const JOBS_JQ = """.jobs[] | [(.id | tostring), .name, (.conclusion // ""), (.run_attempt | tostring), .html_url, ([(.steps // [])[] | [(.number | tostring), (.conclusion // ""), .name] | @tsv] | join("\\n") | @base64)] | @tsv"""
const ISSUES_JQ = """.[] | select(.pull_request == null) | [(.number | tostring), .state, (.state_reason // ""), (.title | @base64), ((.body // "") | @base64), ([.labels[].name] | join("\\n") | @base64)] | @tsv"""
const COMMENTS_JQ = """.[] | (.body // "") | @base64"""
const RUNS_JQ = """.workflow_runs[] | [.created_at, .status, (.conclusion // "")] | @tsv"""

# jq's @tsv escapes a tab, a line feed, a carriage return and a backslash.
unescape_tsv(s::AbstractString) = String(replace(s, r"\\([tnr\\])" => m -> m[2] == 't' ? "\t" : m[2] == 'n' ? "\n" : m[2] == 'r' ? "\r" : "\\"))

const B64 = Dict(c => UInt32(i - 1) for (i, c) in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"))

# The text of a base64 string (the Base64 standard library is not among the test dependencies);
# bytes that are not valid UTF-8 become U+FFFD.
function decode64(s::AbstractString)
  text = String(strip(s))
  data = rstrip(text, '=')
  pad = length(text) - length(data)
  # padding is at most two `=` at the end, and then completes the last group of four
  ('=' in data || pad > 2 || (pad > 0 && (length(data) + pad) % 4 != 0)) &&
    throw(ArgumentError("not base64: misplaced padding in $(repr(first(s, 20)))"))
  digits = UInt32[]
  for c in data
    haskey(B64, c) || throw(ArgumentError("not base64: $(repr(first(s, 20)))"))
    push!(digits, B64[c])
  end
  bytes = UInt8[]
  for i in 1:4:length(digits)
    chunk = digits[i:min(i + 3, end)]
    n = length(chunk)
    n == 1 && throw(ArgumentError("not base64: a dangling digit"))
    acc = UInt32(0)
    for k in 1:4
      acc = (acc << 6) | (k <= n ? chunk[k] : UInt32(0))
    end
    push!(bytes, UInt8((acc >> 16) & 0xff))
    n >= 3 && push!(bytes, UInt8((acc >> 8) & 0xff))
    n == 4 && push!(bytes, UInt8(acc & 0xff))
  end
  return scrub(String(bytes))
end

lines_of(text) = [l for l in split(text, '\n') if !isempty(strip(l))]

function fields(line, n)
  f = split(chomp(line), '\t'; keepempty = true)
  length(f) == n || error("unexpected line with $(length(f)) fields instead of $n: $(repr(first(line, 80)))")
  return f
end

function parse_jobs(text::AbstractString)
  jobs = Job[]
  for line in lines_of(text)
    id, name, conclusion, attempt, url, steps = fields(line, 6)
    list = Step[]
    for l in lines_of(decode64(steps))
      n, c, sname = fields(l, 3)
      push!(list, Step(something(tryparse(Int, n), 0), unescape_tsv(sname), String(c)))
    end
    push!(jobs, Job(String(id), unescape_tsv(name), String(conclusion), String(attempt), String(url), list))
  end
  return jobs
end

function parse_issues(text::AbstractString)
  issues = Issue[]
  for line in lines_of(text)
    number, state, reason, title, body, labels = fields(line, 6)
    push!(issues, Issue(parse(Int, number), String(state), String(reason), decode64(title), decode64(body),
                        String[l for l in split(decode64(labels), '\n') if !isempty(l)]))
  end
  return issues
end

parse_comments(text::AbstractString) = String[decode64(l) for l in lines_of(text)]

function parse_runs(text::AbstractString)
  runs = Scheduled[]
  for line in lines_of(text)
    created, status, conclusion = fields(line, 3)
    push!(runs, Scheduled(parse_time(created), String(status), String(conclusion)))
  end
  return runs
end

read_jobs(gh, run::Run) =
  parse_jobs(gh(["api", "repos/$(run.repo)/actions/runs/$(run.id)/jobs?filter=latest&per_page=100", "--paginate", "--jq", JOBS_JQ]))

"The log of a job: its text, or `nothing` when GitHub has none (not found or expired); any other error throws."
function read_log(gh, run::Run, id)
  try
    return gh(["api", "repos/$(run.repo)/actions/jobs/$id/logs"])
  catch err
    err isa GhError && err.status in (404, 410) && return nothing
    rethrow()
  end
end

"The issues of the repository with `state` (`all` or `open`) and, when given, the label, every page of them; pull requests are left out."
function read_issues(gh, repo; label = nothing, state = "all")
  args = ["api", "-X", "GET", "repos/$repo/issues", "-f", "state=$state", "-f", "per_page=100"]
  label === nothing || append!(args, ["-f", "labels=$label"])
  append!(args, ["--paginate", "--jq", ISSUES_JQ])
  return parse_issues(gh(args))
end

read_comments(gh, run::Run, number) =
  parse_comments(gh(["api", "repos/$(run.repo)/issues/$number/comments?per_page=100", "--paginate", "--jq", COMMENTS_JQ]))

function read_scheduled_runs(gh, run::Run)
  workflow = strip(gh(["api", "repos/$(run.repo)/actions/runs/$(run.id)", "--jq", ".workflow_id"]))
  occursin(r"^\d+$", workflow) || error("the run has no workflow id")
  return parse_runs(gh(["api", "repos/$(run.repo)/actions/workflows/$workflow/runs?event=schedule&status=completed&per_page=100", "--jq", RUNS_JQ]))
end

# ---------------------------------------------------------------------------------------------
# Writing

"The names of the repository's labels, every page of them."
read_labels(gh, repo) = String[strip(l) for l in lines_of(gh(["api", "repos/$repo/labels?per_page=100", "--paginate", "--jq", ".[].name"]))]

function create_label(gh, repo, policy::Policy, name)
  color, description = get(policy.styles, name, ("ededed", ""))
  gh(["api", "-X", "POST", "repos/$repo/labels", "-f", "name=$name", "-f", "color=$color", "-f", "description=$description", "--silent"])
end

# The labels the actions add, in order of first use.
labels_needed(actions) = unique(reduce(vcat, [a.labels for a in actions if a.kind in (:create, :label)]; init = String[]))

function bodyfile(dir, i, text)
  path = joinpath(dir, "text-$i.md")
  write(path, text)
  return path
end

function write_action(gh, repo, dir, i, a::Action)
  n = a.number
  if a.kind == :create
    args = ["api", "-X", "POST", "repos/$repo/issues", "-f", "title=$(a.title)", "-F", "body=@$(bodyfile(dir, i, a.body))"]
    foreach(l -> append!(args, ["-f", "labels[]=$l"]), a.labels)
    return strip(gh([args; ["--jq", ".html_url"]]))
  elseif a.kind == :comment
    return strip(gh(["api", "-X", "POST", "repos/$repo/issues/$n/comments", "-F", "body=@$(bodyfile(dir, i, a.body))", "--jq", ".html_url"]))
  elseif a.kind == :reopen
    return gh(["api", "-X", "PATCH", "repos/$repo/issues/$n", "-f", "state=open", "--silent"])
  elseif a.kind == :close
    return gh(["api", "-X", "PATCH", "repos/$repo/issues/$n", "-f", "state=closed", "-f", "state_reason=completed", "--silent"])
  elseif a.kind == :label
    args = ["api", "-X", "POST", "repos/$repo/issues/$n/labels"]
    foreach(l -> append!(args, ["-f", "labels[]=$l"]), a.labels)
    return gh([args; ["--silent"]])
  end
  error("unknown action kind $(a.kind)")
end

function action_line(a::Action)
  target = a.number === nothing ? "new issue" : "issue #$(a.number)"
  return "$(a.kind) $target [$(a.key)] $(a.title)"
end

"""
    apply!(actions, gh; repo, policy, labels = String[], log = println) -> number of failed writes

Execute the actions in order. `labels` are the names of the repository's labels, read before any
write; the labels the actions add that are not among them are created first, once each. `gh` does
not say why a creation was refused (a 422 carries no reason on its standard error), so after a
failed creation the labels are read again: the label being there now means that another run made
it, and anything else is reported. A write that fails is reported with `::error::` and the rest
still run. Running the plan again after a failure completes it: the plan is made from the current
state, so a comment (once per run, by its marker) is not repeated, while a reopen, a close and a
label that did not happen are planned again.
"""
function apply!(actions::Vector{Action}, gh; repo, policy::Policy, labels = String[], log = println)
  failures = 0
  isempty(actions) && return failures
  dir = mktempdir()
  try
    for name in labels_needed(actions)
      name in labels && continue
      try
        create_label(gh, repo, policy, name)
      catch err
        err isa InterruptException && rethrow()
        there = try
          name in read_labels(gh, repo)
        catch
          false
        end
        there && continue
        failures += 1
        log(error_line("could not create the label `$name`: " * first(sprint(showerror, err), 300)))
      end
    end
    for (i, a) in enumerate(actions)
      try
        result = write_action(gh, repo, dir, i, a)
        log("done: ", action_line(a), isempty(result) ? "" : " ($result)")
      catch err
        err isa InterruptException && rethrow()
        failures += 1
        log(error_line("could not write [$(action_line(a))]: " * first(sprint(showerror, err), 300)))
      end
    end
  finally
    rm(dir; recursive = true, force = true)
  end
  return failures
end

function print_actions(io::IO, actions)
  println(io, isempty(actions) ? "No action." : "$(length(actions)) planned actions:")
  for a in actions
    println(io, "- ", action_line(a))
    isempty(a.labels) || println(io, "    labels: ", join(a.labels, ", "))
    isempty(a.body) || println(io, "    ", shorten(oneline(a.body), 160))
  end
end

# ---------------------------------------------------------------------------------------------
# The command line

"The run of the environment; throws an `ArgumentError` listing what is wrong."
function parse_run(env)
  get_(k) = String(strip(get(env, k, "")))
  problems = String[]
  repo = get_("GITHUB_REPOSITORY")
  occursin(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo) || push!(problems, "GITHUB_REPOSITORY is not set to owner/name")
  id = get_("GITHUB_RUN_ID")
  occursin(r"^\d+$", id) || push!(problems, "GITHUB_RUN_ID is not set to a number")
  attempt = isempty(get_("GITHUB_RUN_ATTEMPT")) ? "1" : get_("GITHUB_RUN_ATTEMPT")
  occursin(r"^\d+$", attempt) || push!(problems, "GITHUB_RUN_ATTEMPT is not a number")
  sha = get_("GITHUB_SHA")
  occursin(r"^[0-9a-f]{7,64}$", sha) || push!(problems, "GITHUB_SHA is not set to a commit SHA")
  server = rstrip(isempty(get_("GITHUB_SERVER_URL")) ? "https://github.com" : get_("GITHUB_SERVER_URL"), '/')
  occursin(r"^https?://[^\s/]+$", server) || push!(problems, "GITHUB_SERVER_URL is not a URL")
  isempty(get_("GITHUB_EVENT_NAME")) && push!(problems, "GITHUB_EVENT_NAME is not set")
  isempty(problems) || throw(ArgumentError(join(problems, "; ")))
  return Run(repo, id, attempt, sha, server)
end

"The test jobs the run was meant to have: the plan's lists, or the written-out nightly list when the plan failed."
function expected_jobs(env)
  required = CIVerdict.splitnames(String(get(env, "REQUIRED", "")))
  advisory = CIVerdict.splitnames(String(get(env, "ADVISORY", "")))
  if isempty(required) && isempty(advisory)
    return [CIVerdict.mode_jobs("cross", "full"); CIVerdict.jobname(CIVerdict.ADVISORY_JOB..., "full")]
  end
  return unique(String.(strip.([required; advisory])))
end

function plan_for(mode, gh, policy, run, env, now, rehearsal, io)
  if mode == "triage"
    jobs = read_jobs(gh, run)
    expected = expected_jobs(env)
    logs = Dict{String, Union{Nothing, String}}()
    for name in expected, job in jobs
      job.name == name && job.conclusion != "success" && (logs[job.id] = read_log(gh, run, job.id))
    end
    obs = observe(policy, jobs, expected, logs)
    issues = read_issues(gh, run.repo; label = active_label(policy, rehearsal))
    for n in conflicting(policy, issues)
      println(io, "::warning::Issue #$n has both the label `$(policy.label)` and the label `$(policy.rehearsal_label)`; it is skipped.")
    end
    failing = Set(r.key for r in obs.records)
    comments = Dict{Int, Vector{String}}()
    for i in issues_to_read(policy, issues, failing; rehearsal)
      comments[i.number] = read_comments(gh, run, i.number)
    end
    return plan_triage(policy, run, obs, issues, comments; rehearsal)
  elseif mode == "alert"
    failed = CIVerdict.splitnames(String(get(env, "FAILED", "")))
    # the lines per file are an addition: without them the alert is still sent
    details = try
      jobs = read_jobs(gh, run)
      logs = Dict{String, Union{Nothing, String}}()
      for name in failed, job in jobs
        job.name == name && job.conclusion != "success" && startswith(name, "Julia ") && (logs[job.id] = read_log(gh, run, job.id))
      end
      alert_details(failed, jobs, logs)
    catch err
      err isa InterruptException && rethrow()
      println(io, "::warning::The failed files are not listed: ", first(sprint(showerror, err), 300))
      Dict{String, Vector{Tuple{String, String}}}()
    end
    issues = read_issues(gh, run.repo; state = "open")
    title = make_title(policy.title_alert; rehearsal)
    existing = find_open(issues, title)
    comments = existing === nothing ? Dict{Int, Vector{String}}() : Dict(existing.number => read_comments(gh, run, existing.number))
    return plan_alert(policy, run, String(get(env, "GITHUB_EVENT_NAME", "")), failed, String(get(env, "LINUX_FAILED", "")),
                      details, issues, comments; rehearsal)
  else
    runs = read_scheduled_runs(gh, run)
    issues = read_issues(gh, run.repo; state = "open")
    title = make_title(policy.title_stale; rehearsal)
    existing = find_open(issues, title)
    comments = existing === nothing ? Dict{Int, Vector{String}}() : Dict(existing.number => read_comments(gh, run, existing.number))
    return plan_stale(policy, run, runs, issues, comments, now; rehearsal)
  end
end

"""
    main(args, env; gh, now, rules, io, err) -> exit status

0 when the actions were written (or printed with `--dry-run`); 1 when a write failed, or when
something mandatory could not be read, in which case nothing is written (the jobs, their logs, the
issues, their comments, the scheduled runs and the labels; for `alert` a failed read of the jobs
or of a log only drops the lines per failed file, and the alert is still sent); 2 for a usage,
environment or policy error, or an event that the mode does not run for.
"""
function main(args = ARGS, env = ENV; gh = CITriage.gh, now = () -> floor(Int, time()), rules = RULES,
              io::IO = stdout, err::IO = stderr)
  flags = [a for a in args if startswith(a, "--")]
  modes = [a for a in args if !startswith(a, "--")]
  if length(modes) != 1 || !(modes[1] in MODES) || !all(f -> f in ("--dry-run", "--rehearsal"), flags)
    println(err, USAGE)
    return 2
  end
  mode = modes[1]
  dry_run, rehearsal = "--dry-run" in flags, "--rehearsal" in flags
  run = try
    parse_run(env)
  catch e
    e isa ArgumentError || rethrow()
    println(err, "scripts/ci_triage.jl: ", e.msg, "\n", USAGE)
    return 2
  end
  event = String(strip(get(env, "GITHUB_EVENT_NAME", "")))
  if event != EVENTS[mode]
    println(err, "scripts/ci_triage.jl: `$mode` runs for the event `$(EVENTS[mode])`, not `$event`")
    return 2
  end
  policy = try
    load_policy(rules)
  catch e
    e isa PolicyError || e isa SystemError || e isa TOML.ParserError || rethrow()
    println(io, error_line(sprint(showerror, e)))
    return 2
  end
  actions = try
    plan_for(mode, gh, policy, run, env, now(), rehearsal, io)
  catch e
    e isa InterruptException && rethrow()
    println(io, error_line("scripts/ci_triage.jl: could not read the run, the issues or their logs, so nothing was changed: " *
                           first(sprint(showerror, e), 300)))
    return 1
  end
  if dry_run
    println(io, "Dry run: nothing is written.")
    print_actions(io, actions)
    return 0
  end
  print_actions(io, actions)
  # the labels are read before the first write; a label that is missing is created by `apply!`
  labels = String[]
  if !isempty(labels_needed(actions))
    try
      labels = read_labels(gh, run.repo)
    catch e
      e isa InterruptException && rethrow()
      println(io, error_line("scripts/ci_triage.jl: could not read the labels, so nothing was changed: " * first(sprint(showerror, e), 300)))
      return 1
    end
  end
  failures = apply!(actions, gh; repo = run.repo, policy, labels, log = (xs...) -> println(io, xs...))
  failures == 0 || println(io, error_line("$failures of the writes failed"))
  return failures == 0 ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
  exit(CITriage.main())
end
