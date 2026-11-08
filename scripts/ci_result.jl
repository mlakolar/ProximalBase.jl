# Read the log of a test job (the output of scripts/test.jl, a ParallelTestRunner run): which test
# files the runner selected, what became of each, and every failure with its file, test set and a
# signature that stays the same from night to night.
#
#   julia scripts/ci_result.jl check-inventory LOG --tier=full [--extra=a,b]
#
# `check-inventory` fails (prints one `::error::` line per problem, exit status 1) unless the
# files the runner announced in its header are exactly the files of the tier plus the extra files
# (`--extra` of scripts/test.jl, which CI passes only for a focus) and the count in the header
# agrees; a log without the header is a problem. Exit status 2 is a usage error.
#
# `parse_log` takes the text of a log in any of the shapes CI gives it: the raw output, colored
# output, the lines of the Actions API (a timestamp before each line) and `gh run view --log` (job
# and step names, a tab and a timestamp before each line). It never throws on text it does not
# understand: such a log has `recognized` false and no selected files, and failures are still
# found without the header. The functions are pure (text in, data out); the module can be
# included by other scripts and by tests. No packages are needed.

module CIResult

include(joinpath(dirname(@__DIR__), "test", "tiers.jl"))
using .TestTiers

const ROOT = dirname(@__DIR__)

# The line that test/runtests.jl prints before any worker starts, then the sorted names, indented.
const HEADER = r"^ProximalBase tests: (.*): (\d+) files; Julia"

const ANSI = r"\e\[[0-9;]*[A-Za-z]"
# `gh run view --log` puts "<job>\t<step>\t" before the timestamp; the API puts only the timestamp.
const GH_PREFIX = r"^[^\t]*\t[^\t]*\t(?=\d{4}-\d\d-\d\dT)"
const API_TIME = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z ?"

const ROW = r"^(\S+)\s+\((\d+)\)\s+│(.*)$"
const TEST_FAILED = r"^\s*(Test Failed|Error During Test) at (.+):(\d+)\s*$"
const TESTSET = r"^\s*Error in testset (.*):\s*$"
const BLOCK_START = r"^\s*(?:Test Failed at |Error During Test at |Error in testset |ERROR: |Test Summary:)"

const MAX_EXCERPT_LINES = 12
const MAX_EXCERPT_CHARS = 1000

"One failure of a test file: `kind` is `:test` (a failed `@test`), `:exception` or `:crash`."
struct Failure
  file::String
  testset::String
  kind::Symbol
  signature::String
  excerpt::String
end

"""
    ParsedLog

What `parse_log` found. `header` says whether the runner's header line was there (`headers`: how
many times), `label` and `count` are what it says of the selection (`tier full`, 183) and `files`
the names it lists after it. `count` is `nothing` without a header and when the header's number is
too large for an `Int`; `count_text` is the number as printed (empty without a header). `status` maps each test file that has a row in the runner's table to
`:passed`, `:failed`, `:crashed` or `:unfinished` (started, with no later row: a log cut by a time
limit). `recognized` is false when the log has neither header nor table rows.
"""
struct ParsedLog
  recognized::Bool
  header::Bool
  headers::Int
  label::String
  count::Union{Nothing, Int}
  count_text::String
  files::Vector{String}
  status::Dict{String, Symbol}
  failures::Vector{Failure}
end

# A String with bytes that are not valid UTF-8 would make the patterns throw.
scrub(s::AbstractString) = isvalid(s) ? String(s) : String(map(c -> isvalid(c) ? c : '\ufffd', collect(s)))

"""
    clean_line(line) -> String

The text of one log line: carriage returns, a byte order mark, the `gh` job and step columns, the
timestamp and the colors removed. The indentation is kept.
"""
function clean_line(line::AbstractString)
  s = replace(line, '\r' => "")
  s = chopprefix(s, "\ufeff")
  s = replace(s, GH_PREFIX => ""; count = 1)
  s = replace(s, API_TIME => ""; count = 1)
  return replace(s, ANSI => "")
end

"""
    test_file(path) -> String

The name of the test file of a path in a failure message: what follows the last `test/` directory,
without `.jl` (`D:\\a\\ProximalBase.jl\\ProximalBase.jl\\test\\conformance\\x.jl` is `conformance/x`),
else the name of the file.
"""
function test_file(path::AbstractString)
  p = replace(path, '\\' => '/')
  m = match(r"^(?:.*/)?test/(.+)\.jl$", p)
  m !== nothing && return String(m.captures[1])
  return String(replace(last(split(p, '/')), r"\.jl$" => ""))
end

"""
    normalize_signature(text) -> String

`text` as one comparable line: absolute paths without whitespace that have at least two
components (`/a/b`), Windows drive paths without whitespace (`C:\\a`) and hexadecimal addresses
removed, each run of digits replaced by `#`, whitespace collapsed. Other path forms stay in the
signature: a path with a space keeps the part after the space, and a single-component path such as
`/artifact` is kept. Failures that differ only in numbers (sizes, counts, times) share a signature.
"""
function normalize_signature(text::AbstractString)
  s = replace(text, r"(?<!\w)[A-Za-z]:[\\/]\S*" => "")
  s = replace(s, r"(?<![\w)\]])(?:/[^\s/:]+){2,}/?(?::\d+)*" => "")
  s = replace(s, r"@?0x[0-9a-fA-F]+" => "")
  s = replace(s, r"\d+" => "#")
  return String(strip(replace(s, r"\s+" => " ")))
end

# The first lines of a block, at most MAX_EXCERPT_LINES lines and MAX_EXCERPT_CHARS characters.
function excerpt(lines)
  text = join(rstrip.(first(lines, MAX_EXCERPT_LINES)), '\n')
  return String(first(text, MAX_EXCERPT_CHARS))
end

# The lines of a failure block: the line of `Test Failed at` or `Error During Test at` and those
# after it up to the next blank line or the start of another block.
function block(lines, i)
  stop = i
  while stop < length(lines) && !isempty(strip(lines[stop + 1])) && !occursin(BLOCK_START, lines[stop + 1])
    stop += 1
  end
  return lines[i:stop]
end

# The signature of a failure block: the expression of a failed test, the first line of an exception.
function block_signature(kind, body)
  if kind == :test
    for l in body[2:end]
      m = match(r"^\s*Expression:\s*(.*)$", l)
      m === nothing || return normalize_signature(m.captures[1])
    end
  else
    for l in body[2:end]
      t = strip(l)
      (isempty(t) || t == "Got exception outside of a @test" || t == "Test threw exception" || startswith(t, "Expression:")) && continue
      return normalize_signature(t)
    end
  end
  # nothing recognizable: the first line below the header
  rest = [strip(l) for l in body[2:end] if !isempty(strip(l))]
  return isempty(rest) ? "" : normalize_signature(first(rest))
end

"""
    parse_log(text) -> ParsedLog

Parse the log of a test job; `text` is a string or the bytes of a file. See [`ParsedLog`](@ref) and
[`Failure`](@ref).
"""
parse_log(bytes::AbstractVector{UInt8}) = parse_log(String(copy(bytes)))
function parse_log(text::AbstractString)
  lines = [clean_line(l) for l in split(scrub(text), '\n')]
  headers, label, count, count_text, files = 0, "", nothing, "", String[]
  status = Dict{String, Symbol}()
  failures = Failure[]
  testset = ""
  for (i, line) in enumerate(lines)
    m = match(HEADER, line)
    if m !== nothing
      headers += 1
      if headers == 1
        label, count_text = String(m.captures[1]), String(m.captures[2])
        count = tryparse(Int, count_text)
        following = i < length(lines) ? lines[i + 1] : ""
        startswith(following, "  ") && (files = String.(split(following)))
      end
      continue
    end
    m = match(ROW, line)
    if m !== nothing
      name, cells = String(m.captures[1]), String(strip(m.captures[3]))
      last_cell = strip(last(split(cells, '│')))
      if occursin("crashed at", cells)
        status[name] = :crashed
        push!(failures, Failure(name, "", :crash, normalize_signature(last_cell), excerpt([strip(line)])))
      elseif occursin("failed at", cells)
        status[name] = :failed
      elseif occursin("started at", cells)
        status[name] = :unfinished
      elseif occursin(r"^[\d.]+\s*│", cells)
        status[name] = :passed
      end
      continue
    end
    m = match(TESTSET, line)
    if m !== nothing
      testset = String(m.captures[1])
      continue
    end
    m = match(TEST_FAILED, line)
    if m !== nothing
      kind = m.captures[1] == "Test Failed" ? :test : :exception
      body = block(lines, i)
      push!(failures, Failure(test_file(m.captures[2]), testset, kind, block_signature(kind, body), excerpt(body)))
    end
  end
  return ParsedLog(headers > 0 || !isempty(status), headers > 0, headers, label, count, count_text, files, status, failures)
end

# The catalog of test/tiers.toml.
default_manifest() = TestTiers.load(joinpath(ROOT, "test", "tiers.toml"))

cut(names, n = 8) = join(first(names, n), ", ") * (length(names) > n ? ", ... ($(length(names)) in all)" : "")

"""
    check_inventory(names, tier; extra = String[], count = nothing, manifest = default_manifest()) -> Vector{String}

The problems with the list of test files that a runner announced: it must be the files of `tier`
(`TestTiers.select`) together with the `extra` files, each once, and `count`, the number in the
runner's header, when given, must be the number of names. An empty result means that the
announced selection matches the catalog of the tier; it does not show that the files ran to
completion (the runner may stop after the header), which the job's own result must say. Throws
`ArgumentError` for an unknown tier.
"""
function check_inventory(names, tier::AbstractString; extra = String[], count = nothing, manifest = default_manifest())
  names = String.(names)
  expected = sort!(unique([TestTiers.select(manifest, tier); String.(extra)]))
  problems = String[]
  count === nothing || count == length(names) ||
    push!(problems, "the runner's header says $count files, but it lists $(length(names)) names")
  repeated = sort!(unique([n for n in names if count_of(names, n) > 1]))
  isempty(repeated) || push!(problems, "the runner lists $(length(repeated)) names more than once: $(cut(repeated))")
  absent = setdiff(expected, names)
  isempty(absent) ||
    push!(problems, "$(length(absent)) test files of tier $tier did not run: $(cut(absent))")
  surplus = sort!(setdiff(unique(names), expected))
  isempty(surplus) ||
    push!(problems, "$(length(surplus)) test files ran that tier $tier does not select: $(cut(surplus))")
  return problems
end
count_of(names, n) = count(==(n), names)

"""
    check_inventory(log::ParsedLog, tier; extra = String[], manifest = default_manifest()) -> Vector{String}

The same for a parsed log, which must also have exactly one runner header; the count of the header
and the line of names are checked against each other, and a count that is not a representable
number is a problem. As above, this checks the announced selection only.
"""
function check_inventory(log::ParsedLog, tier::AbstractString; extra = String[], manifest = default_manifest())
  log.header ||
    return ["the test log has no line `ProximalBase tests: ...: N files; Julia ...`, so the files that ran are not known"]
  log.headers > 1 &&
    return ["the test log has $(log.headers) lines `ProximalBase tests: ...: N files; Julia ...`; expected one"]
  problems = check_inventory(log.files, tier; extra, count = log.count, manifest)
  log.count === nothing &&
    pushfirst!(problems, "the runner's header gives a file count that is not a usable number: $(first(log.count_text, 40))")
  return problems
end

# A workflow command ends at a line break and `%` starts an escape, so a message is percent-encoded.
error_line(message) = "::error::" * replace(String(message), "%" => "%25", "\r" => "%0D", "\n" => "%0A")

const USAGE = "usage: julia scripts/ci_result.jl check-inventory LOG --tier=TIER [--extra=a,b]"

"""
    main(args; io = stdout) -> exit status

The command line (see the top of the file): 0 when the log is fine, 1 with a problem, 2 for a
usage error.
"""
function main(args; io::IO = stdout)
  if isempty(args) || args[1] != "check-inventory"
    println(stderr, USAGE)
    return 2
  end
  positional = String[]
  options = Dict{String, String}()
  for arg in args[2:end]
    m = match(r"^--(tier|extra)=(.*)$", arg)
    m === nothing ? push!(positional, arg) : (options[m.captures[1]] = m.captures[2])
  end
  if length(positional) != 1 || !haskey(options, "tier") || any(startswith("--"), positional)
    println(stderr, USAGE)
    return 2
  end
  extra = String[strip(e) for e in split(get(options, "extra", ""), ',') if !isempty(strip(e))]
  problems = try
    check_inventory(parse_log(read(positional[1])), options["tier"]; extra)
  catch err
    err isa InterruptException && rethrow()
    err isa SystemError || err isa ArgumentError || rethrow()
    println(stderr, "scripts/ci_result.jl: ", first(split(sprint(showerror, err), '\n')))
    return 2
  end
  foreach(p -> println(io, error_line(p)), problems)
  isempty(problems) && println(io, "The runner announced exactly the test files of tier $(options["tier"])" *
                                   (isempty(extra) ? "." : " and $(join(extra, ", "))."))
  return isempty(problems) ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
  exit(CIResult.main(ARGS))
end
