# Plan and check a focused test run of one test file on a native system: the helper of
# .github/workflows/platform-focused.yml.
#
#   julia scripts/platform_focus.jl plan                      # the inputs are FOCUS_* variables
#   julia scripts/platform_focus.jl check-log LOG TEST_FILE   # what the runner said it ran
#
# `plan` reads FOCUS_OS, FOCUS_TEST_FILE, FOCUS_JULIA_VERSION and FOCUS_EXPECTED_SHA (an empty
# FOCUS_OS means both, an empty FOCUS_JULIA_VERSION means 1) and GITHUB_SHA, validates all of them
# and reports every problem at once as `::error::` lines (exit 1). Without problems it writes
# `matrix` (the JSON job list), `test_file`, `safe_name` (the name without `/`, for artifact names),
# `selector` (the arguments of scripts/test.jl, separated by spaces), `julia_version` and `sha` to
# $GITHUB_OUTPUT and a table to $GITHUB_STEP_SUMMARY when those are set.
#
# The selector selects exactly one file. test/runtests.jl keeps every test whose name starts with
# a given name and drops every test whose name starts with an excluded name (`!name`), so the
# selector is the name followed by an exclusion for each other test that starts with it.
# `check-log` reads the line that the runner prints before any worker starts (`ProximalBase tests:
# ...: N files; Julia ...`, then the sorted names) and fails unless it names exactly the test file.
# `plan` and the functions it calls work from data alone. No packages are needed.

include(joinpath(dirname(@__DIR__), "test", "tiers.jl"))
using .TestTiers
using TOML

const ROOT = dirname(@__DIR__)

# `\z`, not `$`: a `$` also matches before a final newline.
const NAME_PATTERN = r"^[A-Za-z0-9_]+(/[A-Za-z0-9_]+)*\z"
const MAX_NAME = 100
const OS_MODES = ("windows", "macos", "linux", "both", "all")
const RUNNERS = Dict("windows" => (os = "windows-latest", arch = "x64", label = "windows"),
                     "macos" => (os = "macos-latest", arch = "aarch64", label = "macos"),
                     "linux" => (os = "ubuntu-latest", arch = "x64", label = "linux"))
const MODE_LABELS = Dict("windows" => ["windows"], "macos" => ["macos"], "linux" => ["linux"],
                         "both" => ["windows", "macos"], "all" => ["windows", "macos", "linux"])
const JULIA_VERSION_PATTERN = r"^1\.(0|[1-9][0-9]{0,2})(?:\.(0|[1-9][0-9]{0,3}))?\z"
const SHA_PREFIX_PATTERN = r"^[0-9a-fA-F]{7,40}\z"
const COMMIT_PATTERN = r"^([0-9a-f]{40}|[0-9a-f]{64})\z"
const OUTPUT_KEYS = (:matrix, :test_file, :safe_name, :selector, :julia_version, :sha)

# A rejected input, quoted and escaped, so that it shows what was given and stays on one line.
shown(x) = repr(String(x))

# A workflow command ends at a line break and `%` starts an escape, so a message that holds input
# is percent-encoded.
escape_command(text) = replace(String(text), "%" => "%25", "\r" => "%0D", "\n" => "%0A")
error_line(message) = "::error::" * escape_command(message)

# The first few names of a list, for a message.
cut(names, n = 5) = join(first(names, n), ", ") * (length(names) > n ? ", ..." : "")

"""
    selector(names, name) -> Vector{String}

The arguments of `scripts/test.jl` that select the test `name` and no other test of `names`: the
name, then `!other` for every other name that starts with it. An exclusion never removes `name`
itself, because the excluded name is longer than `name`.
"""
selector(names, name) = [String(name); "!" .* sort!(String[n for n in names if n != name && startswith(n, name)])]

"""
    julia_bound(root = ROOT) -> VersionNumber

The lowest Julia version that `Project.toml` allows (its `[compat]` entry for `julia`).
"""
function julia_bound(root = ROOT)
  compat = get(get(TOML.parsefile(joinpath(root, "Project.toml")), "compat", Dict{String, Any}()), "julia", nothing)
  compat isa AbstractString || error("Project.toml has no [compat] entry for julia")
  bounds = VersionNumber[]
  for part in split(compat, ',')
    m = match(r"^\s*[\^~=]?\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*\z", part)
    m === nothing && error("cannot read `$part` in the julia entry of Project.toml")
    number(x) = x === nothing ? 0 : parse(Int, x)
    push!(bounds, VersionNumber(number(m[1]), number(m[2]), number(m[3])))
  end
  return minimum(bounds)
end

function check_os(os)
  os in OS_MODES && return String[]
  return ["`os` must be one of $(join(OS_MODES, ", ")); got $(shown(os))"]
end

function check_julia_version(version, bound)
  version in ("1", "lts", "pre") && return String[]
  m = match(JULIA_VERSION_PATTERN, version)
  m === nothing &&
    return ["`julia_version` must be 1, lts, pre, 1.N or 1.N.P (the syntax of setup-julia); got $(shown(version))"]
  v = VersionNumber(1, parse(Int, m[1]), m[2] === nothing ? 0 : parse(Int, m[2]))
  v < bound && return ["`julia_version` $(shown(version)) is below the Julia version that Project.toml allows ($bound)"]
  return String[]
end

# The test name: its form first, then what it is in the catalog of test/tiers.toml.
function check_name(name, manifest, root)
  isempty(name) && return ["`test_file` is empty; give a test name as listed in test/tiers.toml, for example `three_factor_product`"]
  ncodeunits(name) > MAX_NAME &&
    return ["`test_file` is $(ncodeunits(name)) bytes long; a test name has at most $MAX_NAME"]
  if match(NAME_PATTERN, name) === nothing
    option = startswith(name, '-') || startswith(name, '!') ? "; an option or an exclusion is not a test name" : ""
    return ["`test_file` $(shown(name)) is not a test name: it holds only letters, digits and `_`, with `/` between " *
            "the parts of a path, as in test/tiers.toml" * option]
  end
  names = sort!(collect(keys(manifest.tier)))
  if haskey(manifest.tier, name)
    isfile(joinpath(root, "test", name * ".jl")) && return String[]
    return ["`test_file` $(shown(name)) is listed in test/tiers.toml, but test/$name.jl does not exist"]
  end
  if name in manifest.helpers
    through = sort(get(manifest.run_through, name, String[]))
    return ["`test_file` $(shown(name)) is a helper that test files include, not a test" *
            (isempty(through) ? "" : "; it runs through $(join(through, ", "))")]
  end
  (name in manifest.environments || TestTiers.isenvironment(manifest, name)) &&
    return ["`test_file` $(shown(name)) is in a directory with an environment of its own that is run by hand, not a test"]
  inside = filter(n -> startswith(n, name * "/"), names)
  isempty(inside) ||
    return ["`test_file` $(shown(name)) is a directory of $(length(inside)) tests, not one test; name one of them: $(cut(inside))"]
  longer = filter(n -> startswith(n, name), names)
  isempty(longer) ||
    return ["`test_file` $(shown(name)) is only the start of $(length(longer)) test names ($(cut(longer))); " *
            "name one complete test"]
  return ["`test_file` $(shown(name)) is not a test in test/tiers.toml; `julia scripts/test.jl --list` shows the names"]
end

function check_sha(expected, github_sha, head)
  problems = String[]
  if match(COMMIT_PATTERN, github_sha) === nothing
    push!(problems, "GITHUB_SHA is not a commit hash: $(shown(github_sha))")
  elseif head === nothing
    push!(problems, "the checked-out commit could not be read")
  elseif lowercase(head) != github_sha
    push!(problems, "the checked-out commit $(shown(head)) is not the dispatched commit $github_sha")
  end
  isempty(expected) && return problems
  if match(SHA_PREFIX_PATTERN, expected) === nothing
    push!(problems, "`expected_sha` must be 7 to 40 hexadecimal digits; got $(shown(expected))")
  elseif !startswith(github_sha, lowercase(expected))
    push!(problems, "`expected_sha` $(shown(expected)) is not the start of the dispatched commit $github_sha")
  end
  return problems
end

function matrix_json(os)
  entries = ["{\"os\":\"$(r.os)\",\"arch\":\"$(r.arch)\",\"label\":\"$(r.label)\"}" for r in (RUNNERS[l] for l in MODE_LABELS[os])]
  return "{\"include\":[" * join(entries, ",") * "]}"
end

"""
    plan(inputs; root = ROOT, manifest = nothing, github_sha = "", head = github_sha)

Validate the dispatch `inputs` (a dictionary with `os`, `test_file`, `julia_version` and
`expected_sha`) against the catalog `manifest` (default: test/tiers.toml under `root`), the commit
`github_sha` that was dispatched and the checked-out commit `head`. Returns the named tuple
`(result, problems, lines)`: `result` holds the outputs of the plan (`OUTPUT_KEYS`) and is `nothing`
when there are `problems`; `lines` is the Markdown of the job summary.
"""
function plan(inputs; root = ROOT, manifest = nothing, github_sha = "", head = github_sha)
  value(key, default = "") = (v = String(get(inputs, key, "")); isempty(v) ? default : v)
  os, name, expected = value("os", "both"), String(get(inputs, "test_file", "")), String(get(inputs, "expected_sha", ""))
  version = value("julia_version", "1")
  github_sha = lowercase(String(github_sha))
  problems = String[]
  append!(problems, check_os(os))
  try
    append!(problems, check_julia_version(version, julia_bound(root)))
  catch err
    err isa InterruptException && rethrow()
    push!(problems, "the Julia version could not be checked: " * first(split(sprint(showerror, err), '\n')))
  end
  names = String[]
  try
    manifest === nothing && (manifest = TestTiers.load(joinpath(root, "test", "tiers.toml")))
    append!(problems, check_name(name, manifest, root))
    names = collect(keys(manifest.tier))
  catch err
    err isa InterruptException && rethrow()
    push!(problems, "the test catalog could not be read: " * first(split(sprint(showerror, err), '\n')))
  end
  append!(problems, check_sha(expected, github_sha, head === nothing ? nothing : String(head)))
  isempty(problems) || return (; result = nothing, problems, lines = problem_lines(problems))
  result = (; matrix = matrix_json(os), test_file = name, safe_name = replace(name, "/" => "-"),
            selector = join(selector(names, name), " "), julia_version = version, sha = github_sha)
  systems = join(["$(RUNNERS[l].label) ($(RUNNERS[l].os), $(RUNNERS[l].arch))" for l in MODE_LABELS[os]], ", ")
  lines = ["## Focused test run", "", "| | |", "|---|---|",
           "| Test file | `$name` |", "| Selector | `$(result.selector)` |", "| Systems | $systems |",
           "| Julia | `$version` |", "| Commit | `$github_sha` |"]
  return (; result, problems, lines)
end

# The refused inputs as a summary: one line per problem, in a code block (each message is one line).
problem_lines(problems) = ["## Focused test run: inputs refused", "", "No test job was started.", "", "```", problems..., "```"]

# The environment variables of the workflow step, and the commit that is checked out.
function git_head()
  try
    return readchomp(Cmd(`git rev-parse HEAD`; dir = ROOT))
  catch err
    err isa InterruptException && rethrow()
    return nothing
  end
end

function plan_from_env(env = ENV; head = git_head(), kw...)
  inputs = Dict{String, String}(k => get(env, "FOCUS_" * uppercase(k), "") for k in ("os", "test_file", "julia_version", "expected_sha"))
  return plan(inputs; github_sha = get(env, "GITHUB_SHA", ""), head, kw...)
end

"""
    emit_plan(planned; io = stdout, output = nothing, summary = nothing) -> Int

Report a plan: the `::error::` lines or the outputs on `io`, the outputs appended to the file
`output` (only when there are no problems) and the summary to the file `summary`. Returns the exit
status, 0 or 1.
"""
function emit_plan(planned; io::IO = stdout, output = nothing, summary = nothing)
  (; result, problems, lines) = planned
  if summary !== nothing
    open(f -> println(f, join(lines, '\n')), summary, "a")
  end
  if result === nothing
    foreach(p -> println(io, error_line(p)), problems)
    return 1
  end
  println(io, join(lines, '\n'), "\n")
  for (k, v) in pairs(result)
    println(io, "$k=$v")
  end
  if output !== nothing
    open(output, "a") do f
      for (k, v) in pairs(result)
        println(f, "$k=$v")
      end
    end
  end
  return 0
end

const HEADER = r"^ProximalBase tests: .*: (\d+) files; Julia"

"""
    check_log(text, name) -> Vector{String}

The problems with the log of `scripts/test.jl`: it must hold the header line that test/runtests.jl
prints before any worker starts, saying that it selected one file, and the line after it must list
exactly `name`. A trailing carriage return and colors are ignored.
"""
function check_log(text::AbstractString, name::AbstractString)
  lines = [replace(rstrip(l, '\r'), r"\e\[[0-9;]*[A-Za-z]" => "") for l in split(text, '\n')]
  at = findall(l -> occursin(HEADER, l), lines)
  isempty(at) && return ["the test log has no line `ProximalBase tests: ...: N files; Julia ...`, so the selection is not known"]
  length(at) > 1 && return ["the test log has $(length(at)) lines `ProximalBase tests: ...: N files; Julia ...`; expected one"]
  count = something(tryparse(Int, match(HEADER, lines[only(at)])[1]), -1)
  following = only(at) < length(lines) ? lines[only(at) + 1] : ""
  startswith(following, "  ") ||
    return ["the line after the runner's header does not list the selected files: $(shown(following))"]
  names = String.(split(following))
  count == 1 && names == [name] && return String[]
  return ["the runner selected $count files ($(shown(join(names, " ")))), not only $(shown(name))"]
end

# The bytes of a log as ASCII text: only the ASCII header matters, and a byte that is not valid
# UTF-8 must not make the pattern match fail.
ascii_text(bytes) = String(UInt8[b < 0x80 ? b : UInt8('?') for b in bytes])

function main(args)
  command = isempty(args) ? "" : args[1]
  summary = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
  if command == "plan" && length(args) == 1
    code = emit_plan(plan_from_env(); output = get(ENV, "GITHUB_OUTPUT", nothing), summary)
    code == 0 || exit(code)
  elseif command == "check-log" && length(args) == 3
    log, name = args[2], args[3]
    problems = check_log(ascii_text(read(log)), name)
    if isempty(problems)
      message = "The runner selected one file, $name."
      println(message)
      summary === nothing || open(io -> println(io, "\n", message), summary, "a")
    else
      foreach(p -> println(error_line(p)), problems)
      summary === nothing || open(io -> println(io, "\n", join(problems, "\n")), summary, "a")
      exit(1)
    end
  else
    println(stderr, "usage: julia scripts/platform_focus.jl plan | check-log LOG TEST_FILE")
    exit(2)
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(ARGS)
end
