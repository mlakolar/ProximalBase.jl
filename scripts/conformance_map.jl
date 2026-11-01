# Required coverage data tables: the row ids of each acceptance case (the parts of
# a case are separate rows, so a passing part cannot hide a missing one).
const CASE_IDS = vcat(["V$i" for i in 1:13], ["S$i" for i in 1:10],
  ["L$i" for i in 1:5], ["P2", "P2m", "P3", "Q1", "W1", "F16a", "F16b", "F16c"])
const SPLIT_ROWS = Dict(
  "F16a" => ["F16a", "F16a.unchanged"],
  "F16b" => ["F16b", "F16b.unchanged"],
  "F16c" => ["F16c.level", "F16c.step", "F16c.weights", "F16c.array-step"],
  "L1" => ["L1.value", "L1.gradient"],
  "L2" => ["L2.value", "L2.prox", "L2.prox.standalone"],
  "L3" => ["L3.prox"],
  "L4" => ["L4.gram"],
  "L5" => ["L5.value", "L5.residual", "L5.residual.standalone"],
  "P2m" => ["P2m.exception", "P2m.unchanged"],
  "P3" => ["P3.weight", "P3.response"],
  "S1" => ["S1.value", "S1.prox", "S1.joint-value"],
  "S2" => ["S2.value", "S2.gradient", "S2.prox"],
  "S3" => ["S3.NormL1", "S3.NormL2", "S3.GroupNormL2"],
  "S4" => ["S4.value", "S4.prox", "S4.joint-value"],
  "S5" => ["S5.boundary", "S5.outside", "S5.nan"],
  "S6" => ["S6.value", "S6.prox"],
  "S7" => ["S7.exception", "S7.unchanged"],
  "S8" => ["S8.separate", "S8.aliased"],
  "S9" => ["S9.box.prox", "S9.box.value", "S9.ball.prox", "S9.ball.value"],
  "S10" => ["S10.negative.exception", "S10.negative.unchanged", "S10.nan.exception", "S10.nan.unchanged"],
  "V5" => ["V5.value", "V5.gradient"],
  "V7" => ["V7.value", "V7.gradient"],
  "V9" => ["V9.value", "V9.prox", "V9.prox.standalone"],
  "V11" => ["V11.value", "V11.gradient"],
)
const REQUIRED_ROW_IDS = reduce(vcat, [get(SPLIT_ROWS, id, [id]) for id in CASE_IDS])
# Expected family cells: * is any executed oracle/type/backend cell in that
# family. Explicit cells can be added here as their family interfaces freeze.
const EXPECTED_CELLS = [
  (family = family, oracle = "*", types = "*", backend = "*") for family in (
    "kernels", "special_values", "steps", "data_functions", "linear_sets",
    "spectral", "wrappers", "second_order", "failures_storage",
    "element_types", "tolerances", "gpu_arrays")
]

# Expected report files: one per test file under test/conformance/ other than
# common and harness, split files included. A file whose rows are all case-less
# (linear_sets_2) can be dropped without a missing row id or cell, so a missing
# file is reported by name.
const EXPECTED_FILES = [
  "data_functions", "data_functions_2", "element_types", "failures_storage",
  "gpu_arrays", "kernels", "linear_sets", "linear_sets_2", "second_order",
  "special_values", "spectral", "steps", "tolerances", "wrappers",
]

# The report format is a flat JSON object of strings. Parse without eval or a
# dependency, and reject malformed/truncated input rather than losing rows.
function parse_json_line(line)
  chars = collect(line)
  pos = Ref(1)
  skipws() = begin
    while pos[] <= length(chars) && isspace(chars[pos[]])
      pos[] += 1
    end
  end
  function take(c)
    skipws()
    pos[] <= length(chars) && chars[pos[]] == c || error("expected JSON '$c'")
    pos[] += 1
  end
  function quoted()
    take('"')
    io = IOBuffer()
    while pos[] <= length(chars)
      c = chars[pos[]]
      pos[] += 1
      c == '"' && return String(take!(io))
      if c == '\\'
        pos[] <= length(chars) || error("incomplete JSON escape")
        c = chars[pos[]]
        pos[] += 1
        escapes = Dict('"' => '"', '\\' => '\\', '/' => '/', 'n' => '\n',
                       'r' => '\r', 't' => '\t', 'b' => '\b', 'f' => '\f')
        if c == 'u'
          pos[] + 3 <= length(chars) || error("incomplete unicode escape")
          digits = String(chars[pos[]:pos[]+3])
          print(io, Char(parse(UInt32, digits; base = 16)))
          pos[] += 4
        else
          haskey(escapes, c) || error("invalid JSON escape")
          print(io, escapes[c])
        end
      else
        Int(c) >= 0x20 || error("unescaped JSON control character")
        print(io, c)
      end
    end
    error("unterminated JSON string")
  end
  take('{')
  fields = Dict{String, String}()
  skipws()
  if pos[] <= length(chars) && chars[pos[]] != '}'
    while true
      key = quoted()
      haskey(fields, key) && error("duplicate JSON key: $key")
      take(':')
      fields[key] = quoted()
      skipws()
      pos[] <= length(chars) || error("incomplete JSON object")
      chars[pos[]] == '}' && break
      take(',')
    end
  end
  take('}')
  skipws()
  pos[] > length(chars) || error("trailing JSON content")
  fields
end

function run_directories(root)
  isdir(root) || error("report directory does not exist: $root")
  files = readdir(root; join = true)
  any(p -> endswith(p, ".jsonl"), files) && return [root]
  sort!(filter(p -> isdir(p) && any(endswith(".jsonl"), readdir(p)), files))
end

function read_run(dir; expected_files = EXPECTED_FILES, run_id = nothing)
  headers, rows = Dict{String, String}[], Dict{String, String}[]
  paths = sort!(filter(endswith(".jsonl"), readdir(dir; join = true)))
  for path in paths
    # An empty expected list (fixtures that name no files) accepts any file.
    !isempty(expected_files) && !(first(splitext(basename(path))) in expected_files) &&
      error("unexpected report file: $path")
    lines = readlines(path)
    isempty(lines) && error("empty report: $path")
    header = parse_json_line(first(lines))
    get(header, "kind", "") == "header" || error("missing header: $path")
    all(k -> haskey(header, k), ("pkgsha", "bounds", "julia", "platform", "harness", "run")) || error("incomplete header: $path")
    run_id isa String && header["run"] != run_id &&
      error("report run id $(header["run"]) does not match expected $run_id: $path")
    push!(headers, header)
    ids = Set{String}()
    for line in lines[2:end]
      row = parse_json_line(line)
      all(k -> haskey(row, k), ("id", "case", "family", "oracle", "types", "backend",
        "rules", "problem", "group", "status", "expected", "observed")) || error("incomplete row: $path")
      status = row["status"]
      status in ("pass", "broken", "notcovered") || error("invalid row status: $status")
      if status == "broken"
        row["group"] in ["G$i" for i in 1:12] || error("broken row needs group G1 to G12")
      end
      if status == "notcovered"
        isempty(strip(row["problem"])) && error("notcovered row needs a contract reason")
      else
        isempty(row["id"]) && error("executed row needs an id")
        row["id"] in ids && error("duplicate row id in $path")
        push!(ids, row["id"])
      end
      push!(rows, row)
    end
  end
  isempty(headers) && error("no reports in $dir")
  if length(unique(h["run"] for h in headers)) > 1
    error("reports come from different runs in $dir: ",
      join(("$(basename(path))=$(h["run"])" for (path, h) in zip(paths, headers)), ", "))
  end
  for key in ("pkgsha", "bounds", "julia", "platform", "harness")
    length(unique(h[key] for h in headers)) == 1 || error("reports disagree on $key in $dir")
  end
  first(headers), rows
end

_md(s) = replace(string(s), "|" => "\\|", "\n" => "<br>", "\r" => "", "`" => "\\`")
_list(xs) = isempty(xs) ? "none" : join(sort!(unique(collect(xs))), ", ")

function print_diff(io, out, markdown)
  old = isfile(out) ? split(read(out, String), '\n') : String[]
  new = split(markdown, '\n')
  println(io, "--- ", out, "\n+++ regenerated coverage")
  for i in 1:max(length(old), length(new))
    before = i <= length(old) ? old[i] : nothing
    after = i <= length(new) ? new[i] : nothing
    before == after && continue
    println(io, "@@ line ", i, " @@")
    before === nothing || println(io, "-", before)
    after === nothing || println(io, "+", after)
  end
end

function coverage(root; case_ids = CASE_IDS, required_ids = REQUIRED_ROW_IDS,
                  expected_cells = EXPECTED_CELLS, expected_files = EXPECTED_FILES, run_id = nothing)
  dirs = run_directories(root)
  isempty(dirs) && error("no report files in $root")
  io = IOBuffer()
  println(io, "# Numerical contract coverage\n")
  println(io, "Generated by `scripts/conformance_map.jl` from the reports of the conformance suite ",
    "(`test/conformance/`), produced by `scripts/conformance_run.jl`, which starts from an empty ",
    "report directory and gives every file the same run id. ",
    "One row per family, oracle, element types and backend. *Problem after conversion* is the ",
    "problem the call solves once its parameters are converted (rule T3); *Promised result* names ",
    "the contract rules; *Status* is `passes`, `broken (Gk)` (the package does not meet the promise ",
    "today; Gk is its change group in issue #366), or `not covered` (an explicit record with the ",
    "contract's reason). A cell that mixes outcomes lists each. Counts of broken rows are executed ",
    "rows, not source lines.\n")
  complete = true
  for dir in dirs
    h, rows = read_run(dir; expected_files, run_id)
    println(io, "## Run: ", _md(basename(rstrip(dir, '/'))), "\n")
    println(io, "Baseline SHA (package under test): `", h["pkgsha"], "`  ")
    println(io, "Run id: `", h["run"], "`  ")
    println(io, "Harness revision (common.jl blob): `", h["harness"], "`  ")
    println(io, "Julia: ", h["julia"], "; platform: ", h["platform"], "; bounds: ", h["bounds"], "\n")
    println(io, "| Family | Oracle | Types | Backend | Problem after conversion | Promised result (rules) | Cases | Status |")
    println(io, "|:---|:---|:---|:---|:---|:---|:---|:---|")
    cells = Dict{NTuple{4, String}, Vector{Dict{String, String}}}()
    for row in rows
      key = (row["family"], row["oracle"], row["types"], row["backend"])
      push!(get!(() -> Dict{String, String}[], cells, key), row)
    end
    for key in sort!(collect(keys(cells)))
      cell = cells[key]
      statuses = String[]
      any(r -> r["status"] == "pass", cell) && push!(statuses, "passes")
      groups = [r["group"] for r in cell if r["status"] == "broken"]
      !isempty(groups) && push!(statuses, "broken (" * _list(groups) * ")")
      any(r -> r["status"] == "notcovered", cell) && push!(statuses, "not covered")
      fields = (key..., _list(r["problem"] for r in cell), _list(r["rules"] for r in cell),
                _list(r["case"] for r in cell if !isempty(r["case"])), join(statuses, "; "))
      println(io, "| ", join(_md.(fields), " | "), " |")
    end
    for cell in expected_cells
      values = (cell.family, cell.oracle, cell.types, cell.backend)
      if !any(key -> all(v == "*" || v == k for (v, k) in zip(values, key)), keys(cells))
        println(io, "| ", join(_md.(values), " | "), " | | | | missing |")
        complete = false
      end
    end
    println(io, "\n| Group | Executed broken rows |\n|:---|---:|")
    for group in ["G$i" for i in 1:12]
      count = sum(r["status"] == "broken" && r["group"] == group for r in rows; init = 0)
      println(io, "| ", group, " | ", count, " |")
    end
    executed = filter(r -> r["status"] != "notcovered", rows)
    missing_cases = setdiff(case_ids, [r["case"] for r in executed])
    missing_rows = setdiff(required_ids, [r["id"] for r in executed])
    missing_files = setdiff(expected_files, first.(splitext.(basename.(filter(endswith(".jsonl"), readdir(dir))))))
    println(io, "\nExpected report files without a file (missing report file): ", _list(missing_files), ".")
    println(io, "\nCase ids without a row: ", _list(missing_cases), ".")
    println(io, "\nRequired row ids without a row (missing): ", _list(missing_rows), ".\n")
    complete &= isempty(missing_cases) && isempty(missing_rows) && isempty(missing_files)
  end
  String(take!(io)), complete
end

function main(args = ARGS)
  usage = "Usage: julia scripts/conformance_map.jl <report-dir> [--out docs/design/numerical-contract/coverage.md] [--check] [--run-id ID]"
  if "--help" in args
    println(usage)
    return 0
  end
  try
    isempty(args) && error(usage)
    root = first(args)
    out = joinpath(@__DIR__, "..", "docs", "design", "numerical-contract", "coverage.md")
    check = false
    run_id = nothing
    i = 2
    while i <= length(args)
      if args[i] == "--out" && i < length(args)
        i += 1
        out = args[i]
      elseif args[i] == "--run-id" && i < length(args)
        i += 1
        run_id = args[i]
      elseif args[i] == "--check"
        check = true
      else
        error("unknown or incomplete option: $(args[i])\n$usage")
      end
      i += 1
    end
    markdown, complete = coverage(root; run_id)
    if check
      if !isfile(out) || read(out, String) != markdown
        println(stderr, "coverage differs from $out; regenerate without --check")
        print_diff(stderr, out, markdown)
        return 1
      end
    else
      mkpath(dirname(abspath(out)))
      write(out, markdown)
    end
    complete || println(stderr, "conformance coverage is missing required rows, cells or report files")
    complete ? 0 : 1
  catch e
    println(stderr, sprint(showerror, e))
    1
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  exit(main())
end
