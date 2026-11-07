# Plan a CI run from the event, the changed files and the PR labels: the test tier, the
# platforms (the job matrix), coverage, the benchmark reference checks and the conformance
# reports are decided independently.
#
#   julia scripts/ci_plan.jl [--base=REF]     # your working tree against REF (origin/master)
#   julia scripts/ci_plan.jl --event=pull_request --base=A --head=B --labels=test:full,bug
#   julia scripts/ci_plan.jl --labels=test:iterate-narrow   # your working tree, with a label
#   julia scripts/ci_plan.jl --event=workflow_dispatch --tier=standard --platforms=linux
#   julia scripts/ci_plan.jl --event=push --ref-type=branch --base=BEFORE --head=HEAD
#   julia scripts/ci_plan.jl --event=push --ref-type=tag
#   julia scripts/ci_plan.jl --event=schedule
#
# The policy of each event and the platform modes are in .github/tier-rules.toml, the tier
# lists in test/tiers.toml; test/tiers.jl applies the risk rules. Writes `tier`, `focus`,
# `run_tests`, `timeout`, `benchmark` (whether the benchmark reference checks run), `coverage`,
# `conformance`, `platforms` (the mode), `matrix` (the JSON of the test job matrix), `required`
# and `advisory` (the ';'-separated names of the test jobs that must pass and of those that are
# only reported) to $GITHUB_OUTPUT and a decision table to $GITHUB_STEP_SUMMARY when those are
# set, and prints both. `decide` makes the decision from data alone; `plan` adds the git
# calls. No packages are needed.

include(joinpath(dirname(@__DIR__), "test", "tiers.jl"))
using .TestTiers
using TOML

const ROOT = dirname(@__DIR__)
const RULES = joinpath(ROOT, ".github", "tier-rules.toml")

# The kinds of run the rules file describes, and the platform modes it defines.
const EVENTS = ("pull_request", "push", "tag", "schedule", "workflow_dispatch")
const MODES = ("stable", "linux", "cross")
const EVENT_KEYS = ("tier", "skip_without_tests", "platforms", "platforms_by_tier", "label_platforms",
                    "auto_platforms", "coverage", "benchmark", "benchmark_labels", "benchmark_tiers")
const JOB_KEYS = ("version", "os", "arch", "ptr_jobs", "timeout", "advisory")

"One job of a platform mode: a Julia version on a system."
struct PlatformJob
  version::String
  os::String
  arch::String
  ptr_jobs::Union{Nothing, String}
  timeout::Union{Nothing, Int}
  advisory::Bool
end

"The validated `[events.*]` and `[platforms.*]` tables of the rules file."
struct Policy
  events::Dict{String, Dict{String, Any}}
  platforms::Dict{String, Vector{PlatformJob}}
end

isstr(x) = x isa AbstractString
istier(x) = isstr(x) && TestTiers.rank(x) > 0

# Read and validate the policy tables of the rules file; every problem is listed at once.
function load_policy(rules = RULES)
  data = TOML.parsefile(rules)
  problems = String[]
  events = Dict{String, Dict{String, Any}}()
  platforms = Dict{String, Vector{PlatformJob}}()
  alltiers = [TestTiers.TIERS..., "none"]
  ismode(x) = isstr(x) && x in MODES

  tables = get(data, "platforms", nothing)
  if !(tables isa AbstractDict)
    push!(problems, "[platforms.*] is missing")
    tables = Dict{String, Any}()
  end
  for mode in sort!(collect(keys(tables)))
    mode in MODES || push!(problems, "unknown platform mode `$mode`; the modes are $(join(MODES, ", "))")
  end
  for mode in MODES
    table = get(tables, mode, nothing)
    if !(table isa AbstractDict)
      push!(problems, "[platforms.$mode] is missing")
      continue
    end
    for k in keys(table)
      k == "jobs" || push!(problems, "platforms.$mode has an unknown key `$k`; the key is jobs")
    end
    jobs = PlatformJob[]
    list = get(table, "jobs", nothing)
    if !(list isa AbstractVector) || isempty(list)
      push!(problems, "platforms.$mode.jobs must be a non-empty list of jobs")
      list = Any[]
    end
    for (i, entry) in enumerate(list)
      where = "platforms.$mode.jobs[$i]"
      if !(entry isa AbstractDict)
        push!(problems, "$where must be a table")
        continue
      end
      for k in sort!(collect(keys(entry)))
        k in JOB_KEYS || push!(problems, "$where has an unknown key `$k`; the keys are $(join(JOB_KEYS, ", "))")
      end
      for k in ("version", "os", "arch")
        isstr(get(entry, k, nothing)) && !isempty(entry[k]) || push!(problems, "$where needs a string `$k`")
      end
      ptr = get(entry, "ptr_jobs", nothing)
      ptr === nothing || isstr(ptr) || push!(problems, "$where: ptr_jobs must be a string")
      timeout = get(entry, "timeout", nothing)
      timeout === nothing || (timeout isa Integer && timeout > 0) ||
        push!(problems, "$where: timeout must be a positive number of minutes")
      advisory = get(entry, "advisory", false)
      advisory isa Bool || push!(problems, "$where: advisory must be true or false")
      all(k -> isstr(get(entry, k, nothing)), ("version", "os", "arch")) || continue
      job = PlatformJob(entry["version"], entry["os"], entry["arch"],
                        ptr isa AbstractString ? String(ptr) : nothing,
                        timeout isa Integer && timeout > 0 ? Int(timeout) : nothing, advisory === true)
      any(j -> (j.version, j.os, j.arch) == (job.version, job.os, job.arch), jobs) &&
        push!(problems, "$where repeats Julia $(job.version) on $(job.os) ($(job.arch))")
      push!(jobs, job)
    end
    platforms[mode] = jobs
  end

  tables = get(data, "events", nothing)
  if !(tables isa AbstractDict)
    push!(problems, "[events.*] is missing")
    tables = Dict{String, Any}()
  end
  for name in sort!(collect(keys(tables)))
    name in EVENTS || push!(problems, "unknown event `$name`; the events are $(join(EVENTS, ", "))")
  end
  for name in EVENTS
    ev = get(tables, name, nothing)
    if !(ev isa AbstractDict)
      push!(problems, "[events.$name] is missing")
      continue
    end
    where = "events.$name"
    for k in sort!(collect(keys(ev)))
      k in EVENT_KEYS || push!(problems, "$where has an unknown key `$k`; the keys are $(join(EVENT_KEYS, ", "))")
    end
    for k in ("tier", "platforms", "coverage", "benchmark")
      haskey(ev, k) || push!(problems, "$where needs `$k`")
    end
    tier = get(ev, "tier", "")
    if tier == "risk"
      name == "pull_request" || push!(problems, "$where: tier \"risk\" is for pull requests only")
    elseif tier == "input"
      name == "workflow_dispatch" || push!(problems, "$where: tier \"input\" is for manual runs only")
    elseif haskey(ev, "tier") && !istier(tier)
      push!(problems, "$where: unknown tier `$tier`; use risk, input or one of $(join(TestTiers.TIERS, ", "))")
    end
    skip = get(ev, "skip_without_tests", false)
    skip isa Bool || push!(problems, "$where: skip_without_tests must be true or false")
    spec = get(ev, "platforms", "")
    if spec == "by_tier"
      table = get(ev, "platforms_by_tier", nothing)
      if !(table isa AbstractDict)
        push!(problems, "$where: platforms \"by_tier\" needs the table platforms_by_tier")
      else
        for t in alltiers
          ismode(get(table, t, nothing)) || push!(problems, "$where.platforms_by_tier needs a mode for tier `$t`")
        end
        for t in sort!(collect(keys(table)))
          t in alltiers || push!(problems, "$where.platforms_by_tier has an unknown tier `$t`")
        end
      end
    elseif spec == "input"
      name == "workflow_dispatch" || push!(problems, "$where: platforms \"input\" is for manual runs only")
      table = get(ev, "auto_platforms", nothing)
      if !(table isa AbstractDict)
        push!(problems, "$where: platforms \"input\" needs the table auto_platforms")
      else
        for t in TestTiers.TIERS
          ismode(get(table, t, nothing)) || push!(problems, "$where.auto_platforms needs a mode for tier `$t`")
        end
        for t in sort!(collect(keys(table)))
          t in TestTiers.TIERS || push!(problems, "$where.auto_platforms has an unknown tier `$t`")
        end
      end
    elseif haskey(ev, "platforms") && !ismode(spec)
      push!(problems, "$where: unknown platforms `$spec`; use by_tier, input or one of $(join(MODES, ", "))")
    end
    spec == "by_tier" || haskey(ev, "platforms_by_tier") &&
      push!(problems, "$where: platforms_by_tier needs platforms = \"by_tier\"")
    spec == "input" || haskey(ev, "auto_platforms") &&
      push!(problems, "$where: auto_platforms needs platforms = \"input\"")
    if haskey(ev, "label_platforms")
      table = ev["label_platforms"]
      if !(table isa AbstractDict)
        push!(problems, "$where.label_platforms must be a table of label = mode")
      else
        for (label, mode) in table
          startswith(label, "test:") || push!(problems, "$where.label_platforms: `$label` is not a test: label")
          ismode(mode) || push!(problems, "$where.label_platforms: unknown mode `$mode` for `$label`")
        end
      end
    end
    cov = get(ev, "coverage", false)
    cov isa Bool || cov == "input" || push!(problems, "$where: coverage must be true, false or \"input\"")
    cov == "input" && name != "workflow_dispatch" && push!(problems, "$where: coverage \"input\" is for manual runs only")
    bench = get(ev, "benchmark", "")
    if haskey(ev, "benchmark") && !(bench in ("paths", "always", "never", "input"))
      push!(problems, "$where: unknown benchmark `$bench`; use paths, always, never or input")
    end
    bench == "input" && name != "workflow_dispatch" && push!(problems, "$where: benchmark \"input\" is for manual runs only")
    for (key, only) in (("benchmark_labels", "paths"), ("benchmark_tiers", "input"))
      haskey(ev, key) || continue
      bench == only || push!(problems, "$where: $key needs benchmark = \"$only\"")
      list = ev[key]
      if !(list isa AbstractVector) || !all(isstr, list)
        push!(problems, "$where.$key must be a list of strings")
      elseif key == "benchmark_tiers"
        for t in list
          istier(t) || push!(problems, "$where.benchmark_tiers: unknown tier `$t`")
        end
      else
        for label in list
          startswith(label, "test:") || push!(problems, "$where.benchmark_labels: `$label` is not a test: label")
        end
      end
    end
    events[name] = Dict{String, Any}(ev)
  end
  isempty(problems) || throw(TestTiers.ManifestError(problems))
  return Policy(events, platforms)
end

# The names of the paths that start the benchmark reference checks.
benchmark_paths(rules = RULES) = String[TOML.parsefile(rules)["benchmark_check"]["paths"]...]

# Booleans of the manual inputs as GitHub passes them; an empty value is the default.
function boolean(inputs, name, default)
  v = get(inputs, name, "")
  isempty(v) && return default
  v in ("true", "false") || error("input $name must be true or false, not `$v`")
  return v == "true"
end
input(inputs, name, default) = (v = get(inputs, name, ""); isempty(v) ? default : String(v))

"""
    benchmark_check(event, paths, labels, inputs; tier = nothing, policy = load_policy(), patterns = benchmark_paths()) -> Bool

Whether the benchmark reference checks (benchmark/check_only.jl) run, as `[events.<event>]
benchmark` of the rules file says: `paths` matches a pattern of `[benchmark_check] paths` or a
label is in `benchmark_labels`; always; never; or, for a manual run, the input `benchmark` or a
tier of `benchmark_tiers`. `event` is the name of the table (`push` for master, `tag` for a tag
push). `paths === nothing` (no list of changed files) decides on the labels alone.
Documentation-only changes match nothing.
"""
function benchmark_check(event, paths, labels, inputs; tier = nothing, policy = load_policy(), patterns = benchmark_paths())
  ev = policy.events[event]
  mode = ev["benchmark"]
  mode == "always" && return true
  mode == "never" && return false
  if mode == "input"
    return boolean(inputs, "benchmark", false) || tier in get(ev, "benchmark_tiers", String[])
  end
  any(in(labels), get(ev, "benchmark_labels", String[])) && return true
  paths === nothing && return false
  regexes = map(TestTiers.glob_regex, patterns)
  return any(p -> any(r -> occursin(r, p), regexes), paths)
end

# The mode of platforms for a tier.
function platform_mode(ev, tier, labels, inputs)
  spec = ev["platforms"]
  if spec == "by_tier"
    mode = ev["platforms_by_tier"][tier]
    for (label, m) in get(ev, "label_platforms", Dict{String, Any}())
      label in labels && (mode = m)
    end
    return mode
  elseif spec == "input"
    p = input(inputs, "platforms", "auto")
    p == "auto" && return ev["auto_platforms"][tier]
    p in MODES || error("unknown platforms `$p`; use auto or one of $(join(MODES, ", "))")
    return p
  end
  return spec
end

linux(job) = startswith(job.os, "ubuntu")
jobname(job, tier) = "Julia $(job.version) - $(job.os) - $(job.arch) - $tier"

"""
    matrix_entries(jobs) -> Vector{Vector{Pair{String, Any}}}

The entries of the test job matrix for a platform mode, with the keys the workflow reads:
`version`, `os`, `arch`, `ptr_jobs` and `timeout` when set, and the booleans `advisory`, `map`
and `coverage`. `map` marks the one job that checks the conformance coverage map and whose report
upload is required: Julia lts on Linux when the mode has it, otherwise Julia 1 on Linux.
`coverage` marks Julia 1 on Linux. Neither flag is set when the mode has no such job.
"""
function matrix_entries(jobs)
  lts = findfirst(j -> linux(j) && j.version == "lts" && !j.advisory, jobs)
  jl1 = findfirst(j -> linux(j) && j.version == "1" && !j.advisory, jobs)
  mapped = lts !== nothing ? lts : jl1
  entries = Vector{Pair{String, Any}}[]
  for (i, j) in enumerate(jobs)
    entry = Pair{String, Any}["version" => j.version, "os" => j.os, "arch" => j.arch]
    j.ptr_jobs === nothing || push!(entry, "ptr_jobs" => j.ptr_jobs)
    j.timeout === nothing || push!(entry, "timeout" => j.timeout)
    push!(entry, "advisory" => j.advisory, "map" => i == mapped, "coverage" => i == jl1)
    push!(entries, entry)
  end
  return entries
end

# A small JSON writer for the matrix: strings, numbers, booleans, arrays, and objects given as
# vectors of pairs. Nothing else is needed, and no package is available here.
function json_string(s::AbstractString)
  out = IOBuffer()
  print(out, '"')
  for c in s
    if c == '"'
      print(out, "\\\"")
    elseif c == '\\'
      print(out, "\\\\")
    elseif c == '\n'
      print(out, "\\n")
    elseif c == '\r'
      print(out, "\\r")
    elseif c == '\t'
      print(out, "\\t")
    elseif c < ' '
      print(out, "\\u", lpad(string(Int(c); base = 16), 4, '0'))
    else
      print(out, c)
    end
  end
  print(out, '"')
  return String(take!(out))
end
json(x::AbstractString) = json_string(x)
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::AbstractVector{<:Pair}) = "{" * join((json_string(k) * ":" * json(v) for (k, v) in x), ",") * "}"
json(x::AbstractVector) = "[" * join(map(json, x), ",") * "]"

matrix_json(entries) = json(Pair{String, Any}["include" => entries])

function default_manifest()
  manifest = TestTiers.load(joinpath(ROOT, "test", "tiers.toml"), RULES)
  # The runner checks this too, but here an unclassified test file fails within seconds.
  TestTiers.check(manifest, TestTiers.discover(joinpath(ROOT, "test")))
  return manifest
end

"""
    decide(event; paths = nothing, labels = String[], inputs = Dict(), ref_type = "",
           additions = nothing, problems = String[], manifest, policy) -> (result, lines)

The plan of a run, from data alone (no git, no network). `event` is the GitHub event name; a
`push` needs `ref_type` `branch` (master) or `tag`. `paths` are the changed files (`nothing`
when unknown: required for a pull request; a push with unknown paths still runs its tier),
`labels` the PR's labels, `inputs` the manual run's inputs (`tier`, `platforms`, `benchmark`,
`coverage`, as strings; empty means the default). `additions` and `problems` are the result of
comparing `test/tiers.toml` with the base (see [`tiers_additions`](@ref)). `result` is a
NamedTuple of the outputs listed at the top of the file and `lines` the decision table.
"""
function decide(event::AbstractString; paths = nothing, labels = String[], inputs = Dict{String, String}(),
                ref_type = "", additions = nothing, problems = String[],
                manifest = default_manifest(), policy = load_policy())
  labels = String[labels...]
  key = event
  if event == "push"
    ref_type in ("branch", "tag") ||
      error("a push of ref type `$ref_type` is not planned; use --ref-type=branch or --ref-type=tag")
    key = ref_type == "tag" ? "tag" : "push"
  elseif !(event in EVENTS)
    error("unknown event `$event`")
  end
  ev = policy.events[key]
  lines = String[]
  focus = String[]
  spec = ev["tier"]
  if spec == "risk"
    paths === nothing && error("the plan of a pull request needs its changed files")
    tier, focus, decisions, notes = TestTiers.pr_tier(manifest, paths, labels; manifest_additions = additions)
    push!(lines, "Changed files ($(length(paths))), the rule that decides each, and its tier:", "",
          "| Path | Tier | Why |", "|---|---|---|")
    for (path, t, why) in decisions
      push!(lines, "| `$path` | $t | $why |")
    end
    isempty(problems) || push!(lines, "", "`test/tiers.toml` does more than add new test files, so it needs full: " *
                                       join(problems, "; ") * ".")
    isempty(notes) || push!(lines, "", notes...)
  elseif spec == "input"
    tier = input(inputs, "tier", "full")
    TestTiers.rank(tier) > 0 || error("unknown tier `$tier`; use one of $(join(TestTiers.TIERS, ", "))")
    push!(lines, "Manual run of tier $tier.")
  else
    tier = spec
    what = key == "push" ? "A push to master" : key == "tag" ? "A tag push" : "A scheduled run"
    if get(ev, "skip_without_tests", false) && paths !== nothing && !isempty(paths) &&
       first(TestTiers.classify(manifest, paths)) == "none"
      tier = "none"
      push!(lines, "Only files that need no package tests changed ($(length(paths))).")
    else
      push!(lines, "$what runs the $tier tier.")
    end
  end
  # Changed files the tier already runs need no separate mention.
  if tier != "none"
    intier = Set(TestTiers.select(manifest, tier))
    filter!(name -> !(name in intier), focus)
  else
    empty!(focus)
  end
  run_tests = tier != "none"
  timeout = run_tests ? manifest.timeouts[tier] : 0
  benchmark = benchmark_check(key, paths, labels, inputs; tier, policy)
  coverage = run_tests && (ev["coverage"] === true || (ev["coverage"] == "input" && boolean(inputs, "coverage", false)))
  conformance = run_tests && tier in ("thorough", "full")
  mode = platform_mode(ev, tier, labels, inputs)
  jobs = policy.platforms[mode]
  entries = matrix_entries(jobs)
  required = run_tests ? [jobname(j, tier) for j in jobs if !j.advisory] : String[]
  advisory = run_tests ? [jobname(j, tier) for j in jobs if j.advisory] : String[]
  summary = run_tests ? "Tests: tier **$tier**" * (isempty(focus) ? "" : " and the focus files $(join(focus, ", "))") :
            "Tests: **none needed**"
  pushfirst!(lines, summary,
             "Platforms: **$mode**" * (run_tests ? " ($(length(required)) required jobs" *
                                       (isempty(advisory) ? "" : ", $(length(advisory)) advisory") * ")" : ""),
             "Coverage: **$(coverage ? "yes" : "no")**; conformance reports: **$(conformance ? "yes" : "no")**",
             "Benchmark reference checks: **$(benchmark ? "yes" : "no")**", "")
  result = (; tier, focus = join(focus, ","), run_tests, timeout, benchmark, coverage, conformance,
            platforms = mode, matrix = matrix_json(entries), required = join(required, ";"),
            advisory = join(advisory, ";"))
  return result, lines
end

function options(args)
  opts = Dict{String, String}()
  for arg in args
    m = match(r"^--([a-z-]+)=(.*)$", arg)
    m === nothing && error("unknown argument `$arg`; expected --name=value")
    opts[m.captures[1]] = m.captures[2]
  end
  known = ("event", "base", "head", "labels", "tier", "platforms", "benchmark", "coverage", "ref-type")
  for k in keys(opts)
    k in known || error("unknown option --$k; the options are $(join(known, ", "))")
  end
  return opts
end

# The commit a change is compared with: `base` itself, or without `head` the merge base of
# `base` and HEAD.
base_commit(base, head) = head === nothing ? readchomp(Cmd(`git merge-base $base HEAD`; dir = ROOT)) : base

# Without `head`, the working tree (and untracked files) against the merge base with `base`,
# as a PR would show it: a local check before a push.
function changed_files(base, head)
  # --no-renames lists both paths of a rename, so a removed path still counts.
  revs = head === nothing ? (base_commit(base, head),) : (base, head)
  out = read(Cmd(`git diff --name-only --no-renames $revs`; dir = ROOT), String)
  head === nothing && (out *= "\n" * read(Cmd(`git ls-files --others --exclude-standard`; dir = ROOT), String))
  return sort!(unique(filter(!isempty, split(out, '\n'))))
end

# Whether a change to test/tiers.toml only adds test files that the change also adds: compares
# the parsed tier lists at `base` with those at `head` (or in the working tree without `head`),
# see TestTiers.manifest_additions.
function tiers_additions(base, head, paths)
  exists(rev, path) = success(Cmd(`git cat-file -e $rev:$path`; dir = ROOT))
  athead(path) = head === nothing ? isfile(joinpath(ROOT, path)) : exists(head, path)
  exists(base, "test/tiers.toml") || return nothing, ["it does not exist at the base $base"]
  new_files = String[]
  for path in paths
    m = match(r"^test/(.+)\.jl$", path)
    m !== nothing && !exists(base, path) && athead(path) && push!(new_files, m.captures[1])
  end
  old = read(Cmd(`git show $base:test/tiers.toml`; dir = ROOT), String)
  new = head === nothing ? read(joinpath(ROOT, "test", "tiers.toml"), String) :
        read(Cmd(`git show $head:test/tiers.toml`; dir = ROOT), String)
  return TestTiers.manifest_additions(old, new, new_files)
end

# `decide` with the changed files read from git: those of a pull request, and those of a push to
# master when its previous commit is known (a missing, zero or unfetchable one leaves them unknown).
function plan(opts)
  event = get(opts, "event", "pull_request")
  ref_type = get(opts, "ref-type", "")
  labels = filter(!isempty, split(get(opts, "labels", ""), ','))
  inputs = Dict{String, String}(k => get(opts, k, "") for k in ("tier", "platforms", "benchmark", "coverage"))
  manifest = default_manifest()
  paths = nothing
  additions, problems = nothing, String[]
  if event == "pull_request"
    base, head = get(opts, "base", "origin/master"), get(opts, "head", nothing)
    paths = changed_files(base, head)
    "test/tiers.toml" in paths && ((additions, problems) = tiers_additions(base_commit(base, head), head, paths))
  elseif event == "push" && ref_type == "branch"
    base, head = get(opts, "base", ""), get(opts, "head", "HEAD")
    if !isempty(base) && !all(==('0'), base) && success(Cmd(`git cat-file -e $(base * "^{commit}")`; dir = ROOT))
      paths = try
        changed_files(base, head)
      catch err
        err isa InterruptException && rethrow()
        nothing
      end
    end
  end
  return decide(event; paths, labels, inputs, ref_type, additions, problems, manifest)
end

function main(args)
  result, lines = plan(options(args))
  text = join(lines, '\n')
  println(text)
  println()
  for (k, v) in pairs(result)
    println("$k=$v")
  end
  if haskey(ENV, "GITHUB_OUTPUT")
    open(ENV["GITHUB_OUTPUT"], "a") do io
      for (k, v) in pairs(result)
        println(io, "$k=$v")
      end
    end
  end
  if haskey(ENV, "GITHUB_STEP_SUMMARY")
    open(io -> println(io, "## Test plan\n\n", text), ENV["GITHUB_STEP_SUMMARY"], "a")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(ARGS)
end
