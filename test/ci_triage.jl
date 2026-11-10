module CITriageTest

# The nightly triage reporter (scripts/ci_triage.jl): what it plans from the jobs, logs and issues
# of a run (pure functions), what it writes through `gh`, and the [triage] table it reads. The
# GitHub of these tests is in memory: a `Hub` answers the `gh api` calls from issues, jobs, logs and
# scheduled runs it holds, writes to them, and records every argument vector. The logs are cut
# from the real failed log of a run whose macOS and Windows jobs failed test/three_factor_product.jl
# (the shape of the Actions API: a timestamp before each line, colors, carriage returns on Windows).
# Nothing here uses the network, the environment or a shared file.

using Test
using TOML

include(joinpath(dirname(@__DIR__), "scripts", "ci_triage.jl"))
using .CITriage
const T = CITriage

const ROOT = dirname(@__DIR__)
const RULES = joinpath(ROOT, ".github", "tier-rules.toml")
const POLICY = T.load_policy(RULES)

const REPO = "owner/repo"
const SERVER = "https://github.com"
const SHA = "45e3b75334d62a3e07d741a3bc028f07ecd5f4ac"
const NOW = T.epoch(2026, 11, 8, 3, 20)

const LINUX_LTS = "Julia lts - ubuntu-latest - x64 - full"
const LINUX_1 = "Julia 1 - ubuntu-latest - x64 - full"
const MAC = "Julia 1 - macos-latest - aarch64 - full"
const WIN = "Julia 1 - windows-latest - x64 - full"
const LINUX_PRE = "Julia pre - ubuntu-latest - x64 - full"
const NAMES = [LINUX_LTS, LINUX_1, MAC, WIN, LINUX_PRE]

const ZW = "\u200b"
const TESTSET = "allocation does not grow with the number of stored coefficients"
const FILES = ["calculus", "conjugate_projections", "gpu", "precision_mixed", "three_factor_product"]

# ---------------------------------------------------------------------------------------------
# Logs

stamp(lines; cr = false) = join(["2026-11-07T16:14:40.3688732Z " * l * (cr ? "\r" : "") for l in lines], "\n") * "\n"

started_row(f, w) = "$(rpad(f, 37))($w) │\e[2m         started at 2026-11-07T16:14:49.349\e[22m"
passed_row(f, w) = "$(rpad(f, 37))($w) │   400.10 │     0.55 │   95.00 │  10.00 │  2.5 │   20000.00 │  900.00 │"
failed_row(f, w) = "\e[31m$(rpad(f, 37))($w) │    34.91 │     5.86 │    failed at 2026-11-07T16:15:26.922\e[39m"
crashed_row(f, w) = "\e[31m$(rpad(f, 37))($w) │    34.91 │     5.86 │    crashed at 2026-11-07T16:15:26.922\e[39m"

function failblock(; testset = TESTSET, path, line, expr, evaluated, kind = "Test Failed")
  return ["", "Error in testset $testset:",
          "\e[91m\e[1m$kind\e[22m\e[39m at \e[39m\e[1m$path:$line\e[22m",
          "  Expression: $expr", "   Evaluated: $evaluated"]
end

const WIN_PATH = "D:\\a\\ProximalBase.jl\\ProximalBase.jl\\test\\three_factor_product.jl"
const MAC_PATH = "/Users/runner/work/ProximalBase.jl/ProximalBase.jl/test/three_factor_product.jl"
const LIN_PATH = "/home/runner/work/ProximalBase.jl/ProximalBase.jl/test"

# The text of a test job's log: the runner's header, a row for each started and finished file, the
# failure blocks and the end of the run.
function testlog(; passed = String[], failed = String[], crashed = String[], unfinished = String[], blocks = Vector{String}[],
                 files = FILES, system = "x86_64-linux-gnu", cr = false, header = true)
  lines = String[]
  if header
    push!(lines, "ProximalBase tests: tier full: $(length(files)) files; Julia 1.13.3 on $system (4 threads), commit $SHA")
    push!(lines, "  " * join(sort(files), " "))
  end
  push!(lines, "Test                             (Worker) │ time (s) │ time (s) │   (%)   │ GC (s) │ GC % │ Alloc (MB) │ RSS (MB) │")
  for f in unfinished
    push!(lines, started_row(f, 1))
  end
  for f in passed
    push!(lines, started_row(f, 2), passed_row(f, 2))
  end
  for f in failed
    push!(lines, started_row(f, 1), failed_row(f, 1))
  end
  for f in crashed
    push!(lines, started_row(f, 3), crashed_row(f, 3))
  end
  isempty(blocks) || push!(lines, "Test Summary:   | Pass  Fail  Total", "    FAILURE")
  foreach(b -> append!(lines, b), blocks)
  isempty(blocks) || push!(lines, "\e[91m\e[1mERROR: \e[22m\e[39mLoadError: \e[91mTest run finished with errors\e[39m")
  return stamp(lines; cr)
end

const WIN_LOG = testlog(; failed = ["three_factor_product"], passed = ["calculus", "gpu"], system = "x86_64-w64-mingw32", cr = true,
                        blocks = [failblock(; path = WIN_PATH, line = 972, expr = "a1 == a2 == a0", evaluated = "4279 == 4263 == 4263")])
const WIN_SIG = "a# == a# == a#"
const WIN_LOG_CHANGED = testlog(; failed = ["three_factor_product"], passed = ["calculus", "gpu"], system = "x86_64-w64-mingw32", cr = true,
                                blocks = [failblock(; path = WIN_PATH, line = 972, expr = "length(xs) == 3", evaluated = "4 == 3")])
const WIN_LOG_OTHER = testlog(; failed = ["conjugate_projections"], passed = ["three_factor_product", "gpu"], system = "x86_64-w64-mingw32", cr = true,
                              blocks = [failblock(; testset = "projection", path = "D:\\a\\x\\test\\conjugate_projections.jl", line = 40,
                                                  expr = "iszero(r)", evaluated = "false")])
const MAC_LOG = testlog(; failed = ["three_factor_product"], passed = ["calculus", "gpu"], system = "arm64-apple-darwin27.0.0",
                        blocks = [failblock(; path = MAC_PATH, line = 973, expr = "a1 <= sizeof(T) * (p * p + 3p) + 1024", evaluated = "9184 <= 8944")])
const CALC_BLOCK = failblock(; testset = "gradient", path = "$LIN_PATH/calculus.jl", line = 55, expr = "norm(g - ∇f(x)) <= 1.0e-8",
                             evaluated = "0.5 <= 1.0e-8")
const CALC_SIG = "norm(g - ∇f(x)) <= #.#e-#"
const LIN1_LOG = testlog(; failed = ["calculus"], passed = ["gpu", "three_factor_product"], blocks = [CALC_BLOCK])
const LTS_LOG = testlog(; failed = ["gpu"], passed = ["calculus", "three_factor_product"],
                        blocks = [failblock(; testset = "jlarray", path = "$LIN_PATH/gpu.jl", line = 9, expr = "x == y", evaluated = "1 == 2")])
const PASS_LOG = testlog(; passed = FILES)

# ---------------------------------------------------------------------------------------------
# Jobs

const STEPS = ["Set up job", "Run actions/checkout@v7", "Run julia-actions/setup-julia@v3", "Run julia-actions/cache@v3", T.TEST_STEP,
               "Timing summary", "Run codecov/codecov-action@v7", "Complete job"]

# The conclusions of the steps of a job that ended with `conclusion` at the step `at`.
function default_steps(conclusion; at = T.TEST_STEP)
  conclusion == "success" && return [(n, "success") for n in STEPS]
  conclusion == "skipped" && return [(n, "skipped") for n in STEPS]
  i = findfirst(==(at), STEPS)
  return [(n, k < i ? "success" : k == i ? conclusion : conclusion == "failure" ? "success" : "skipped") for (k, n) in enumerate(STEPS)]
end

spec(name, conclusion = "success"; log = nothing, at = T.TEST_STEP) = (; name, conclusion, log, steps = default_steps(conclusion; at))
failed_spec(name, log) = spec(name, "failure"; log)

job_url(run_id, id) = "$SERVER/$REPO/actions/runs/$run_id/job/$id"

function jobs_of(run_id, specs)
  jobs = T.Job[]
  logs = Dict{String, Union{Nothing, String}}()
  for (i, s) in enumerate(specs)
    id = "$run_id$i"
    push!(jobs, T.Job(id, s.name, s.conclusion, "1", job_url(run_id, id), [T.Step(k, n, c) for (k, (n, c)) in enumerate(s.steps)]))
    s.conclusion == "success" || (logs[id] = s.log)
  end
  return jobs, logs
end

const RUN = T.Run(REPO, "1001", "1", SHA, SERVER)
run_of(id; attempt = "1", sha = SHA) = T.Run(REPO, string(id), attempt, sha, SERVER)

const NIGHT1 = [failed_spec(LINUX_LTS, LTS_LOG), failed_spec(LINUX_1, LIN1_LOG), failed_spec(MAC, MAC_LOG), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
green_night() = [spec(n) for n in NAMES]

function observation(specs, run = RUN; expected = NAMES)
  jobs, logs = jobs_of(run.id, specs)
  return T.observe(POLICY, jobs, expected, logs)
end

no_comments() = Dict{Int, Vector{String}}()

# The data structs of the script have no `==`: compare them field by field.
function eqv(a, b)
  typeof(a) == typeof(b) || return false
  a isa AbstractVector && return length(a) == length(b) && all(eqv(x, y) for (x, y) in zip(a, b))
  a isa Union{AbstractString, Number, Symbol, Nothing} && return a == b
  return all(eqv(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
end

function plan(specs; run = RUN, issues = T.Issue[], comments = no_comments(), rehearsal = false, expected = NAMES)
  return T.plan_triage(POLICY, run, observation(specs, run; expected), issues, comments; rehearsal)
end

issue_from(a::T.Action, number; state = "open", reason = "", labels = copy(a.labels)) = T.Issue(number, state, reason, a.title, a.body, labels)

# The issues the actions create, numbered from 1 in the order of the actions' keys.
issues_of(actions; start = 1) = [issue_from(a, start + i - 1) for (i, a) in enumerate(filter(a -> a.kind == :create, actions))]
by_key(actions) = Dict(a.key => a for a in actions if a.kind == :create)
kinds(actions) = [(a.kind, a.number, a.key) for a in actions]
labels_of(a) = sort(a.labels)

# Comments of the actions as the texts of the issues' comments, by issue number.
function commented(actions, comments = no_comments())
  out = Dict{Int, Vector{String}}(k => copy(v) for (k, v) in comments)
  for a in actions
    a.kind == :comment && push!(get!(out, a.number, String[]), a.body)
  end
  return out
end

# ---------------------------------------------------------------------------------------------
# An in-memory GitHub behind `gh`

const JobRow = Tuple{String, String, String, String, String, Vector{Tuple{Int, String, String}}}

mutable struct Hub
  issues::Vector{Dict{String, Any}}
  jobs::Vector{JobRow}
  logs::Dict{String, Any}
  runs::Vector{Tuple{String, String, String}}
  labels::Vector{String}
  calls::Vector{Vector{String}}
  failing::Function
  workflow::String
  events::Vector{Tuple{Symbol, Int}} # the writes that took effect, in order
end
Hub() = Hub(Dict{String, Any}[], JobRow[], Dict{String, Any}(), Tuple{String, String, String}[], String[], Vector{String}[], args -> nothing, "777",
            Tuple{Symbol, Int}[])

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

# Base64 of bytes (the encoder of the fake; the script has its own decoder).
function base64encode(bytes::Vector{UInt8})
  out = IOBuffer()
  for i in 1:3:length(bytes)
    chunk = bytes[i:min(i + 2, end)]
    n = length(chunk)
    acc = UInt32(0)
    for k in 1:3
      acc = (acc << 8) | (k <= n ? UInt32(chunk[k]) : UInt32(0))
    end
    for k in 0:3
      write(out, k <= n ? ALPHABET[Int((acc >> (18 - 6k)) & 0x3f) + 1] : '=')
    end
  end
  return String(take!(out))
end
base64encode(s::AbstractString) = base64encode(Vector{UInt8}(s))

tsv(s) = replace(s, "\\" => "\\\\", "\t" => "\\t", "\n" => "\\n", "\r" => "\\r")

function load!(h::Hub, run_id, specs)
  empty!(h.jobs)
  empty!(h.logs)
  for (i, s) in enumerate(specs)
    id = "$run_id$i"
    push!(h.jobs, (id, s.name, s.conclusion, "1", job_url(run_id, id), [(k, n, c) for (k, (n, c)) in enumerate(s.steps)]))
    s.log === nothing || (h.logs[id] = s.log)
  end
  return h
end

function (h::Hub)(args::Vector{String}; input = nothing)
  push!(h.calls, copy(args))
  failure = h.failing(args)
  failure === nothing || throw(failure)
  args[1] == "api" || error("not an api call: $args")
  method, endpoint = "GET", ""
  fields = Dict{String, Vector{String}}()
  i = 2
  while i <= length(args)
    a = args[i]
    if a == "-X"
      method = args[i + 1]
      i += 2
    elseif a in ("-f", "-F")
      k, v = split(args[i + 1], '='; limit = 2)
      a == "-F" && startswith(v, "@") && (v = read(v[2:end], String))
      push!(get!(fields, String(k), String[]), String(v))
      i += 2
    elseif a == "--jq"
      i += 2
    elseif startswith(a, "repos/")
      endpoint = a
      i += 1
    else
      i += 1
    end
  end
  m = match(r"^repos/[^/]+/[^/]+/([^?]*)(?:\?.*)?$", endpoint)
  m === nothing && error("unexpected endpoint $endpoint")
  path = String(m.captures[1])
  field(k) = haskey(fields, k) ? only(fields[k]) : nothing
  if method == "GET" && path == "issues"
    state, label = field("state"), field("labels")
    rows = [x for x in h.issues if (state == "all" || x["state"] == state) && (label === nothing || label in x["labels"])]
    return join([join([string(x["number"]), x["state"], x["reason"], base64encode(x["title"]), base64encode(x["body"]),
                       base64encode(join(x["labels"], "\n"))], "\t") for x in rows], "\n")
  elseif method == "GET" && (m2 = match(r"^issues/(\d+)/comments$", path)) !== nothing
    x = only(x for x in h.issues if x["number"] == parse(Int, m2.captures[1]))
    return join(base64encode.(x["comments"]), "\n")
  elseif method == "GET" && occursin(r"^actions/runs/\d+/jobs$", path)
    return join([join([id, tsv(name), c, att, url, base64encode(join(["$n\t$sc\t$(tsv(sn))" for (n, sn, sc) in steps], "\n"))], "\t")
                 for (id, name, c, att, url, steps) in h.jobs], "\n")
  elseif method == "GET" && (m2 = match(r"^actions/jobs/(\d+)/logs$", path)) !== nothing
    log = get(h.logs, String(m2.captures[1]), nothing)
    log === nothing && throw(T.GhError("gh api failed: gh: Not Found (HTTP 404)", 404))
    log isa Exception && throw(log)
    return log
  elseif method == "GET" && occursin(r"^actions/runs/\d+$", path)
    return h.workflow * "\n"
  elseif method == "GET" && occursin(r"^actions/workflows/\d+/runs$", path)
    return join([join(r, "\t") for r in h.runs], "\n")
  elseif method == "GET" && path == "labels"
    return join(h.labels, "\n")
  elseif method == "POST" && path == "labels"
    name = field("name")
    # real gh: nothing on the standard output, and only this line on the standard error
    name in h.labels && throw(T.GhError("gh api failed: gh: Validation Failed (HTTP 422)", 422))
    push!(h.labels, name)
    return ""
  elseif method == "POST" && path == "issues"
    n = length(h.issues) + 1
    push!(h.issues, Dict{String, Any}("number" => n, "state" => "open", "reason" => "", "title" => field("title"), "body" => field("body"),
                                      "labels" => copy(get(fields, "labels[]", String[])), "comments" => String[]))
    push!(h.events, (:create, n))
    return "$SERVER/$REPO/issues/$n\n"
  elseif method == "POST" && (m2 = match(r"^issues/(\d+)/comments$", path)) !== nothing
    x = only(x for x in h.issues if x["number"] == parse(Int, m2.captures[1]))
    push!(x["comments"], field("body"))
    push!(h.events, (:comment, x["number"]))
    return "$SERVER/$REPO/issues/$(x["number"])#comment\n"
  elseif method == "POST" && (m2 = match(r"^issues/(\d+)/labels$", path)) !== nothing
    x = only(x for x in h.issues if x["number"] == parse(Int, m2.captures[1]))
    append!(x["labels"], fields["labels[]"])
    push!(h.events, (:label, x["number"]))
    return ""
  elseif method == "PATCH" && (m2 = match(r"^issues/(\d+)$", path)) !== nothing
    x = only(x for x in h.issues if x["number"] == parse(Int, m2.captures[1]))
    x["state"] = field("state")
    x["reason"] = something(field("state_reason"), x["state"] == "open" ? "reopened" : "")
    push!(h.events, (x["state"] == "open" ? :reopen : :close, x["number"]))
    return ""
  end
  error("the hub does not know $method $endpoint")
end

writes(h) = [c for c in h.calls if "-X" in c && c[findfirst(==("-X"), c) + 1] in ("POST", "PATCH")]
issue(h, n) = only(x for x in h.issues if x["number"] == n)
titled(h, title) = only(x for x in h.issues if x["title"] == title)

failing_on(sub; status = 503) = args -> any(a -> occursin(sub, a), args) ? T.GhError("gh api failed: gh: Service Unavailable (HTTP $status)", status) : nothing

function env(; run_id = "1001", attempt = "1", sha = SHA, event = "schedule", failed = "", linux_failed = "", required = join(NAMES[1:4], ";"),
             advisory = LINUX_PRE)
  return Dict("GITHUB_REPOSITORY" => REPO, "GITHUB_RUN_ID" => string(run_id), "GITHUB_RUN_ATTEMPT" => attempt, "GITHUB_SHA" => sha,
              "GITHUB_SERVER_URL" => SERVER, "GITHUB_EVENT_NAME" => event, "REQUIRED" => required, "ADVISORY" => advisory,
              "FAILED" => failed, "LINUX_FAILED" => linux_failed)
end

function go(h, mode; flags = String[], now = NOW, kw...)
  io, errio = IOBuffer(), IOBuffer()
  code = T.main([mode; flags], env(; kw...); gh = h, now = () -> now, rules = RULES, io, err = errio)
  return (; code, out = String(take!(io)), err = String(take!(errio)))
end

# One nightly run of the hub: the jobs of `specs`, then the reporter.
function nightly!(h, run_id, specs; kw...)
  load!(h, run_id, specs)
  return go(h, "triage"; run_id, event = "schedule", kw...)
end

# ---------------------------------------------------------------------------------------------

@testset "ci_triage" begin

  @testset "names, severity and keys" begin
    @test T.parse_jobname(WIN) == (alias = "1", runner = "windows-latest", arch = "x64", tier = "full")
    @test T.parse_jobname("Test plan") === nothing
    @test [T.os_key(r) for r in ("ubuntu-latest", "macos-latest", "windows-latest")] == ["linux", "macos", "windows"]
    @test T.os_display("macos") == "macOS" && T.os_display("windows") == "Windows" && T.os_display("linux") == "Linux"
    sev(os, alias, file) = T.severity(POLICY, os, alias, file)
    @test sev("linux", "1", "calculus") == :core
    @test sev("linux", "lts", "calculus") == :core
    @test sev("linux", "pre", "calculus") == :deferred
    @test sev("windows", "1", "calculus") == :deferred
    @test sev("macos", "1", "calculus") == :deferred
    @test sev("linux", "1", "gpu") == :deferred
    @test sev("linux", "1", "precision_mixed") == :deferred
    @test sev("linux", "1", nothing) == :core && sev("linux", "lts", nothing) == :core
    @test sev("linux", "pre", nothing) == :deferred && sev("windows", "1", nothing) == :deferred
    @test T.crc32c(codeunits("123456789")) == 0xe3069283 && T.crc32c(UInt8[]) == 0x00000000
    @test T.sig_hash("123456789") == "e3069283" && T.sig_hash("a# == a# == a#") != T.sig_hash("a# == a#")
    expected = T.expected_jobs(env())
    @test expected == NAMES
    # A failed plan uses the current stable-only cross matrix (not the
    # historical LTS-inclusive fixture used to exercise old triage reports).
    current = [LINUX_1, MAC, WIN, LINUX_PRE]
    @test T.expected_jobs(env(required = join(current[1:3], ";"), advisory = LINUX_PRE)) == current
    @test sort(T.expected_jobs(env(required = "", advisory = ""))) == sort(current)
    @test !(LINUX_LTS in T.expected_jobs(env(required = "", advisory = "")))
  end

  @testset "times and base64" begin
    @test T.epoch(1970, 1, 1) == 0 && T.epoch(2026, 11, 8, 3, 20) == 1794108000
    @test T.epoch(2000, 3, 1) == 951868800 && T.epoch(2024, 2, 29, 23, 59, 59) == 1709251199
    @test T.format_time(1794108000) == "2026-11-08 03:20" && T.format_date(1709251199) == "2024-02-29"
    @test T.format_time(0) == "1970-01-01 00:00" && T.format_time(951868800) == "2000-03-01 00:00"
    @test T.parse_time("2026-11-08T03:20:00Z") == 1794108000 && T.parse_time("2024-02-29T23:59:59.123Z") == 1709251199
    @test_throws ErrorException T.parse_time("yesterday")
    for bad in ("2026-11-08T03:20:00", "2026-11-08T03:20:00+01:00", "2026-11-08T03:20:00-05:00", "2026-11-08T03:20:00Zjunk", "2026-11-08 03:20:00Z")
      @test_throws ErrorException T.parse_time(bad)
    end
    for (plain, coded) in (("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="),
                           ("foobar", "Zm9vYmFy"))
      @test base64encode(plain) == coded
      @test T.decode64(coded) == plain
    end
    @test T.decode64("Zm9v\n") == "foo" && T.decode64("Zm9vYg") == "foob"
    @test_throws ArgumentError T.decode64("Zm9v!")
    @test_throws ArgumentError T.decode64("Z")
    for bad in ("Z=g=", "====", "Zg=", "Zg===", "Zm9v=", "=Zg=", "Z=g")
      @test_throws ArgumentError T.decode64(bad)
    end
    @test T.decode64("Zg==") == "f" && T.decode64("Zm8=") == "fo"
    @test T.decode64(base64encode("é ✓ 日本")) == "é ✓ 日本"
  end

  @testset "case 1: a new nightly failure on Windows, and the same night's Linux failures" begin
    actions = plan(NIGHT1)
    @test kinds(actions) == [(:create, nothing, "linux/calculus"), (:create, nothing, "linux/gpu"),
                             (:create, nothing, "macos/three_factor_product"), (:create, nothing, "windows/three_factor_product")]
    made = by_key(actions)
    # a core CPU correctness failure on Linux; a deferred file on Linux; macOS and Windows are deferred
    @test labels_of(made["linux/calculus"]) == ["nightly-triage", "triage:core", "triage:unclassified"]
    @test labels_of(made["linux/gpu"]) == ["nightly-triage", "triage:deferred", "triage:unclassified"]
    @test labels_of(made["windows/three_factor_product"]) == ["nightly-triage", "triage:deferred", "triage:unclassified"]
    @test labels_of(made["macos/three_factor_product"]) == ["nightly-triage", "triage:deferred", "triage:unclassified"]
    @test length(unique(a.title for a in actions)) == 4
    w = made["windows/three_factor_product"]
    @test w.title == "Nightly failure on Windows: three_factor_product"
    @test startswith(w.body, "<!-- nightly-triage key=windows/three_factor_product -->\n")
    for part in ("$SERVER/$REPO/actions/runs/1001", SHA, "Windows", "Julia 1", "`three_factor_product`", "`$TESTSET`",
                 "`$WIN_SIG`", "failed test", job_url(1001, "10014"), job_url(1001, "10014") * "#step:5:1", "Expression: a1 == a2 == a0",
                 "Evaluated: 4279 == 4263 == 4263", T.DEFERRED_TEXT, T.marker("run", "1001"), T.marker("alias", "1"),
                 T.marker("sig", T.sig_hash(WIN_SIG)))
      @test occursin(part, w.body)
    end
    @test !occursin("revert", w.body) && !occursin(T.CORE_TEXT, w.body)
    # the excerpt is the failure block, fenced
    @test occursin("```\nTest Failed at $WIN_PATH:972\n  Expression: a1 == a2 == a0\n   Evaluated: 4279 == 4263 == 4263\n```", w.body)
    c = made["linux/calculus"]
    @test occursin(T.CORE_TEXT, c.body) && occursin("`gradient`", c.body) && occursin("`$CALC_SIG`", c.body)
    @test occursin("deferred list", made["linux/gpu"].body)
    # a night with nothing failing, and a green night with no issue, plan nothing
    @test isempty(plan(green_night()))
  end

  @testset "case 2: a repeat is one comment, once per run; a changed signature says so" begin
    first_night = plan(NIGHT1)
    issues = issues_of(first_night)
    @test [i.number for i in issues] == [1, 2, 3, 4]
    second = plan(NIGHT1; run = run_of(1002), issues)
    @test kinds(second) == [(:comment, 1, "linux/calculus"), (:comment, 2, "linux/gpu"), (:comment, 3, "macos/three_factor_product"),
                            (:comment, 4, "windows/three_factor_product")]
    @test all(a -> occursin(T.marker("run", "1002"), a.body) && occursin("The failure occurred again.", a.body), second)
    @test !any(a -> occursin("Signature changed", a.body), second)
    @test occursin("$SERVER/$REPO/actions/runs/1002", second[4].body)
    # the same run again (a rerun of failed jobs): the markers of the comments are there
    again = plan(NIGHT1; run = run_of(1002, attempt = "2"), issues, comments = commented(second))
    @test isempty(again)
    # a rerun of the first run: the body has its marker
    @test isempty(plan(NIGHT1; run = RUN, issues))
    # a changed signature in the Windows file
    changed = [failed_spec(WIN, WIN_LOG_CHANGED)]
    night3 = plan([NIGHT1[1], NIGHT1[2], NIGHT1[3], changed[1], NIGHT1[5]]; run = run_of(1003), issues, comments = commented(second))
    @test kinds(night3) == [(:comment, 1, "linux/calculus"), (:comment, 2, "linux/gpu"), (:comment, 3, "macos/three_factor_product"),
                            (:comment, 4, "windows/three_factor_product")]
    @test occursin("Signature changed", night3[4].body) && occursin("`length(xs) == #`", night3[4].body)
    @test !any(a -> occursin("Signature changed", a.body), night3[1:3])
    # a changed signature that was recorded in an earlier comment is not new
    night4 = plan([NIGHT1[1], NIGHT1[2], NIGHT1[3], changed[1], NIGHT1[5]]; run = run_of(1004), issues, comments = commented(night3))
    @test !occursin("Signature changed", night4[4].body)
  end

  @testset "case 3: recovery resolves, unless the data is incomplete" begin
    issues = issues_of(plan(NIGHT1)) # 1 linux/calculus, 2 linux/gpu, 3 macos/..., 4 windows/...
    pairs(actions) = [(a.kind, a.number) for a in actions]
    green = plan(green_night(); run = run_of(1002), issues)
    @test pairs(green) == [(:comment, 1), (:close, 1), (:comment, 2), (:close, 2), (:comment, 3), (:close, 3), (:comment, 4), (:close, 4)]
    @test occursin("Resolved at $SHA, run $SERVER/$REPO/actions/runs/1002", green[1].body)
    @test occursin("Resolved at $SHA, run $SERVER/$REPO/actions/runs/1002", green[7].body)
    win_issue, mac_issue = [issues[4]], [issues[3]]
    swap(name, new) = [s.name == name ? new : s for s in green_night()]
    windows_alone(specs) = plan(specs; run = run_of(1002), issues = win_issue)
    job1 = [(:create, nothing, "windows/job:1")]
    # an unreadable log of the failed Windows job: nothing is resolved, and the job is reported
    r = windows_alone(swap(WIN, failed_spec(WIN, nothing)))
    @test kinds(r) == job1
    @test labels_of(r[1]) == ["nightly-triage", "triage:deferred", "triage:infrastructure"]
    # a log that no one can read
    @test kinds(windows_alone(swap(WIN, failed_spec(WIN, "no table here")))) == job1
    # a missing job
    @test kinds(windows_alone([s for s in green_night() if s.name != WIN])) == job1
    # a cancelled job, whose log even lists the file as passed
    @test kinds(windows_alone(swap(WIN, spec(WIN, "cancelled"; log = testlog(; passed = FILES))))) == job1
    # a skipped job
    @test kinds(windows_alone(swap(WIN, spec(WIN, "skipped")))) == job1
    # only another OS passed: the macOS issue is not resolved by Windows and Linux passing
    r = plan(swap(MAC, spec(MAC, "cancelled")); run = run_of(1002), issues = mac_issue)
    @test kinds(r) == [(:create, nothing, "macos/job:1")]
    # a failed job whose log lists the file as failed is a repeat, not a resolution
    @test kinds(windows_alone(swap(WIN, failed_spec(WIN, WIN_LOG)))) == [(:comment, 4, "windows/three_factor_product")]
    # a failed job whose log lists the file as passed resolves it; the file that failed is a new issue
    r = windows_alone(swap(WIN, failed_spec(WIN, WIN_LOG_OTHER)))
    @test kinds(r) == [(:create, nothing, "windows/conjugate_projections"), (:comment, 4, "windows/three_factor_product"),
                       (:close, 4, "windows/three_factor_product")]
    # an issue recorded for two Julia versions needs both to pass
    both = plan([failed_spec(LINUX_LTS, LIN1_LOG), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    calc = issues_of(both)
    @test length(calc) == 1 && occursin(T.marker("alias", "lts"), calc[1].body) && occursin(T.marker("alias", "1"), calc[1].body)
    r = plan([failed_spec(LINUX_LTS, nothing), spec(LINUX_1), spec(MAC), spec(WIN), spec(LINUX_PRE)]; run = run_of(1002), issues = calc)
    @test kinds(r) == [(:create, nothing, "linux/job:lts")]
    @test pairs(plan(green_night(); run = run_of(1002), issues = calc)) == [(:comment, 1), (:close, 1)]
    # a flaky issue is only commented, once per run
    flaky = [T.Issue(i.number, i.state, i.reason, i.title, i.body, [i.labels; "triage:flaky"]) for i in win_issue]
    r = plan(green_night(); run = run_of(1002), issues = flaky)
    @test kinds(r) == [(:comment, 4, "windows/three_factor_product")]
    @test occursin("Passed tonight", r[1].body) && occursin(T.marker("passed", "1002"), r[1].body)
    @test isempty(plan(green_night(); run = run_of(1002), issues = flaky, comments = commented(r)))
    # an issue with no recorded Julia version resolves nothing
    bare = [T.Issue(7, "open", "", "t", "<!-- nightly-triage key=windows/three_factor_product -->\n", ["nightly-triage"])]
    @test isempty(plan(green_night(); run = run_of(1002), issues = bare))
    # an issue that fails again tonight is commented, not resolved; the others are
    r = plan(swap(WIN, failed_spec(WIN, WIN_LOG)); run = run_of(1002), issues)
    @test pairs(r) == [(:comment, 4), (:comment, 1), (:close, 1), (:comment, 2), (:close, 2), (:comment, 3), (:close, 3)]
  end

  @testset "case 4: closed issues" begin
    made = by_key(plan(NIGHT1))
    win = made["windows/three_factor_product"]
    mac = made["macos/three_factor_product"]
    night = [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
    # closed as completed: reopened with a comment about the recurrence
    r = plan(night; run = run_of(1005), issues = [issue_from(win, 4; state = "closed", reason = "completed")])
    @test [(a.kind, a.number) for a in r] == [(:comment, 4), (:reopen, 4)]
    @test occursin("recurred", r[1].body) && occursin(T.marker("run", "1005"), r[1].body)
    # closed as not planned: a comment, the issue stays closed
    r = plan(night; run = run_of(1005), issues = [issue_from(win, 4; state = "closed", reason = "not_planned")])
    @test [(a.kind, a.number) for a in r] == [(:comment, 4)]
    @test occursin("not planned", r[1].body)
    # a closed issue of another key is untouched; the closed issue of the same key is the one reopened
    r = plan(night; run = run_of(1005), issues = [issue_from(mac, 3; state = "closed", reason = "completed")])
    @test kinds(r) == [(:create, nothing, "windows/three_factor_product")]
    # the open issue is preferred to a closed one of the same key
    r = plan(night; run = run_of(1005), issues = [issue_from(win, 4; state = "closed", reason = "completed"), issue_from(win, 9)])
    @test [(a.kind, a.number) for a in r] == [(:comment, 9)]
    # a closed issue is not resolved or commented when its key passes
    @test isempty(plan(green_night(); run = run_of(1005), issues = [issue_from(win, 4; state = "closed", reason = "completed")]))
    # a Linux failure of an issue created for Julia pre raises its severity
    pre = plan([spec(LINUX_LTS), spec(LINUX_1), spec(MAC), spec(WIN), failed_spec(LINUX_PRE, LIN1_LOG)])
    @test labels_of(pre[1]) == ["nightly-triage", "triage:deferred", "triage:unclassified"]
    r = plan([spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)]; run = run_of(1006), issues = issues_of(pre))
    @test [(a.kind, a.number) for a in r] == [(:comment, 1), (:label, 1)]
    @test r[2].labels == ["triage:core"] && occursin("Severity raised to core", r[1].body)
  end

  @testset "case 5: jobs that failed without a test failure" begin
    # a missing expected job
    r = plan([failed_spec(LINUX_LTS, LTS_LOG), spec(LINUX_1), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)])
    jobless = by_key(r)["macos/job:1"]
    @test labels_of(jobless) == ["nightly-triage", "triage:deferred", "triage:infrastructure"]
    @test jobless.title == "Nightly job failure on macOS (Julia 1)" && occursin("missing from the run", jobless.body)
    @test occursin(T.marker("sig", T.sig_hash("job missing")), jobless.body)
    # a job that timed out: the log ends in the table, the unfinished file is named
    cut = testlog(; passed = ["gpu"], unfinished = ["calculus"])
    r = plan([spec(LINUX_LTS), spec(LINUX_1, "cancelled"; log = cut), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test kinds(r) == [(:create, nothing, "linux/job:1")]
    @test labels_of(r[1]) == ["nightly-triage", "triage:core", "triage:infrastructure"]
    @test occursin("not finished when the log ends: calculus", r[1].body) && occursin("cancelled", r[1].body)
    @test occursin("job cancelled or timed out", r[1].body) && occursin("at the step `$(T.TEST_STEP)`", r[1].body)
    # a failure in the Codecov step: the tests passed, so this is the infrastructure's
    r = plan([spec(LINUX_LTS), spec(LINUX_1, "failure"; log = PASS_LOG, at = "Run codecov/codecov-action@v7"), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test kinds(r) == [(:create, nothing, "linux/job:1")]
    @test labels_of(r[1]) == ["nightly-triage", "triage:core", "triage:infrastructure"]
    @test occursin("job failed outside the test step", r[1].body) && occursin("Run codecov/codecov-action@$(ZW)v7", r[1].body)
    # the test step failed and the log has nothing to read: unclassified; no log at all: infrastructure
    r = plan([spec(LINUX_LTS), failed_spec(LINUX_1, "Segmentation fault"), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test labels_of(r[1]) == ["nightly-triage", "triage:core", "triage:unclassified"]
    @test occursin("nothing readable", r[1].body)
    r = plan([spec(LINUX_LTS), failed_spec(LINUX_1, nothing), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test labels_of(r[1]) == ["nightly-triage", "triage:core", "triage:infrastructure"]
    @test occursin("could not be read", r[1].body)
    # a worker crash and a file that failed without a failure block
    r = plan([spec(LINUX_LTS), failed_spec(LINUX_1, testlog(; crashed = ["calculus"], failed = ["gpu"], passed = ["conjugate_projections"],
                                                          blocks = [["", "ERROR: LoadError: boom"]])), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test [(a.kind, a.key) for a in r] == [(:create, "linux/calculus"), (:create, "linux/gpu")]
    @test occursin("worker crash", r[1].body) && occursin("failed without a readable failure block", r[2].body)
    # a skipped job counts as not run
    r = plan([spec(LINUX_LTS), spec(LINUX_1), spec(MAC, "skipped"), spec(WIN), spec(LINUX_PRE)])
    @test kinds(r) == [(:create, nothing, "macos/job:1")] && occursin("job skipped", r[1].body)
    # the advisory job on Julia pre fails: deferred, never core
    r = plan([spec(LINUX_LTS), spec(LINUX_1), spec(MAC), spec(WIN), failed_spec(LINUX_PRE, LIN1_LOG)])
    @test labels_of(r[1]) == ["nightly-triage", "triage:deferred", "triage:unclassified"]
  end

  @testset "case 6: outages and failed writes" begin
    h = Hub()
    @test nightly!(h, 1001, NIGHT1).code == 0
    before = deepcopy(h.issues)
    for sub in ("/actions/runs/1002/jobs", "/logs", "state=all", "/comments")
      load!(h, 1002, NIGHT1)
      empty!(h.calls)
      h.failing = failing_on(sub)
      r = go(h, "triage"; run_id = 1002)
      @test r.code == 1
      @test occursin("nothing was changed", r.out) && occursin("503", r.out)
      @test isempty(writes(h))
      @test h.issues == before
    end
    # a log that GitHub does not have is data: the job is reported
    h = Hub()
    h.failing = failing_on("/logs"; status = 404)
    r = nightly!(h, 1001, [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)])
    @test r.code == 0 && [x["title"] for x in h.issues] == ["Nightly job failure on Windows (Julia 1)"]
    @test "triage:infrastructure" in h.issues[1]["labels"]
    # a failing write is reported and the others still run
    h = Hub()
    h.failing = args -> any(a -> occursin("title=Nightly failure on Linux: calculus", a), args) ? T.GhError("gh api failed: HTTP 500", 500) : nothing
    r = nightly!(h, 1001, NIGHT1)
    @test r.code == 1
    @test occursin("::error::could not write", r.out) && occursin("calculus", r.out)
    @test sort([x["title"] for x in h.issues]) ==
          ["Nightly failure on Linux: gpu", "Nightly failure on Windows: three_factor_product", "Nightly failure on macOS: three_factor_product"]
    # a label that cannot be created is reported, the issues are still written
    h = Hub()
    h.failing = args -> any(a -> a == "name=triage:core", args) ? T.GhError("gh api failed: HTTP 500", 500) : nothing
    r = nightly!(h, 1001, NIGHT1)
    @test r.code == 1 && occursin("could not create the label `triage:core`", r.out) && length(h.issues) == 4
    # an unreadable list of labels stops the run before the first write
    h = Hub()
    h.failing = failing_on("repos/$REPO/labels?per_page=100")
    r = nightly!(h, 1001, NIGHT1)
    @test r.code == 1 && occursin("could not read the labels, so nothing was changed", r.out) && isempty(writes(h)) && isempty(h.issues)
    # unreadable data never resolves an issue
    h = Hub()
    nightly!(h, 1001, NIGHT1)
    h.failing = failing_on("/comments")
    r = nightly!(h, 1002, green_night())
    @test r.code == 1 && all(x -> x["state"] == "open", h.issues)
  end

  @testset "case 7: the alert of a failed push to master" begin
    h = Hub()
    load!(h, 2001, [spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    r = go(h, "alert"; event = "push", run_id = 2001, failed = LINUX_1, linux_failed = "true")
    @test r.code == 0
    @test length(h.issues) == 1
    x = h.issues[1]
    @test x["title"] == "CI failing on master" && isempty(x["labels"])
    @test x["body"] == """
      <!-- run:2001 -->

      The push run $SERVER/$REPO/actions/runs/2001 failed at $SHA.

      Failed jobs:
      - $LINUX_1
        - `calculus`, test set `gradient`

      A Linux job, or a check that is not a test job, failed.

      Fix or revert promptly; see CONTRIBUTING.md.
      """
    # a second failed run comments on the open issue; the same run does not
    load!(h, 2002, [spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test go(h, "alert"; event = "push", run_id = 2002, failed = LINUX_1, linux_failed = "true").code == 0
    @test length(h.issues) == 1 && length(x["comments"]) == 1 && occursin("<!-- run:2002 -->", x["comments"][1])
    @test go(h, "alert"; event = "push", run_id = 2002, attempt = "2", failed = LINUX_1, linux_failed = "true").code == 0
    @test length(x["comments"]) == 1
    # a closed issue is not reused
    x["state"] = "closed"
    @test go(h, "alert"; event = "push", run_id = 2003, failed = LINUX_1, linux_failed = "true").code == 0
    @test length(h.issues) == 2 && h.issues[2]["title"] == "CI failing on master"
    # a failure that is not a test job: no lines, the other sentence
    h = Hub()
    load!(h, 2004, [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)])
    @test go(h, "alert"; event = "push", run_id = 2004, failed = "$WIN;Test plan", linux_failed = "false").code == 0
    body = h.issues[1]["body"]
    @test occursin("- $WIN\n  - `three_factor_product`, test set `$TESTSET`\n- Test plan\n", body)
    @test occursin("Only jobs on other systems failed (no Linux job did).", body) && occursin("Fix or revert promptly; see CONTRIBUTING.md.", body)
    # no failed job is known
    h = Hub()
    @test go(h, "alert"; event = "push", run_id = 2005, failed = "", linux_failed = "").code == 0
    @test occursin("The failed jobs are not known; see the run.", h.issues[1]["body"])
    # an unreadable log or job list does not stop the alert
    h = Hub()
    load!(h, 2006, [spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    h.failing = failing_on("/jobs?")
    r = go(h, "alert"; event = "push", run_id = 2006, failed = LINUX_1, linux_failed = "true")
    @test r.code == 0 && occursin("::warning::", r.out) && length(h.issues) == 1 && !occursin("calculus", h.issues[1]["body"])
    # the issues cannot be listed: nothing is written
    h = Hub()
    h.failing = failing_on("state=open")
    r = go(h, "alert"; event = "push", run_id = 2007, failed = LINUX_1, linux_failed = "true")
    @test r.code == 1 && isempty(writes(h))
    # other events produce no action
    for event in ("pull_request", "workflow_dispatch", "schedule", "release")
      h = Hub()
      r = go(h, "alert"; event, run_id = 2008, failed = LINUX_1, linux_failed = "true")
      @test r.code == 2 && isempty(h.calls) && occursin("not `$event`", r.err)
    end
    for (mode, event) in (("triage", "push"), ("triage", "pull_request"), ("stale", "schedule"), ("stale", "workflow_dispatch"))
      h = Hub()
      r = go(h, mode; event)
      @test r.code == 2 && isempty(h.calls)
    end
  end

  @testset "case 8: a stale nightly" begin
    h = Hub()
    h.runs = [("2026-11-05T03:00:00Z", "completed", "success"), ("2026-11-04T03:00:00Z", "completed", "failure")]
    r = go(h, "stale"; event = "push", run_id = 3001)
    @test r.code == 0 && length(h.issues) == 1
    x = h.issues[1]
    @test x["title"] == "Nightly CI is stale" && isempty(x["labels"])
    @test occursin("<!-- stale:2026-11-08 -->", x["body"]) && occursin("No completed scheduled run was created within the last 48 hours", x["body"]) &&
          occursin("The newest completed scheduled run was created at 2026-11-05 03:00 UTC", x["body"])
    @test occursin("recurring full-platform run", x["body"]) && !occursin("only one", x["body"]) && !occursin("No nightly run completed", x["body"])
    # a second push the same day: no second comment
    @test go(h, "stale"; event = "push", run_id = 3002).code == 0
    @test isempty(x["comments"]) && length(h.issues) == 1
    # the next day: one comment
    @test go(h, "stale"; event = "push", run_id = 3003, now = NOW + 86400).code == 0
    @test length(x["comments"]) == 1 && occursin("<!-- stale:2026-11-09 -->", x["comments"][1])
    @test go(h, "stale"; event = "push", run_id = 3004, now = NOW + 86400).code == 0
    @test length(x["comments"]) == 1
    # a fresh nightly closes it
    pushfirst!(h.runs, ("2026-11-08T02:00:00Z", "completed", "failure"))
    @test go(h, "stale"; event = "push", run_id = 3005).code == 0
    @test x["state"] == "closed" && length(x["comments"]) == 2 && occursin("closed", x["comments"][2])
    @test go(h, "stale"; event = "push", run_id = 3006).code == 0
    @test length(x["comments"]) == 2
    # no scheduled run at all, and the edge of the limit
    h = Hub()
    @test go(h, "stale"; event = "push", run_id = 3007).code == 0
    @test occursin("There is no completed scheduled run.", h.issues[1]["body"])
    h = Hub()
    h.runs = [("2026-11-06T03:20:00Z", "completed", "success")]
    @test go(h, "stale"; event = "push", run_id = 3008).code == 0 && isempty(h.issues)
    h.runs = [("2026-11-06T03:19:59Z", "completed", "success")]
    @test go(h, "stale"; event = "push", run_id = 3008).code == 0 && length(h.issues) == 1
    # an API error: exit 1 and nothing written
    for sub in ("/workflows/", "state=open", "/actions/runs/3009")
      h = Hub()
      h.failing = failing_on(sub)
      r = go(h, "stale"; event = "push", run_id = 3009)
      @test r.code == 1 && isempty(writes(h)) && occursin("nothing was changed", r.out)
    end
    # the issue of a rehearsal is not the stale issue
    h = Hub()
    r = go(h, "stale"; event = "push", run_id = 3010, flags = ["--rehearsal"])
    @test r.code == 0 && h.issues[1]["title"] == "[rehearsal] Nightly CI is stale"
    @test go(h, "stale"; event = "push", run_id = 3011).code == 0 && length(h.issues) == 2
  end

  @testset "a story of several nights" begin
    h = Hub()
    r = nightly!(h, 1001, NIGHT1)
    @test r.code == 0 && length(h.issues) == 4
    win = titled(h, "Nightly failure on Windows: three_factor_product")
    @test sort(win["labels"]) == ["nightly-triage", "triage:deferred", "triage:unclassified"] && win["state"] == "open"
    @test isempty(win["comments"])
    # the second night, the same failures
    @test nightly!(h, 1002, NIGHT1).code == 0
    @test length(h.issues) == 4 && all(x -> length(x["comments"]) == 1, h.issues)
    # a rerun of failed jobs of the second night
    @test nightly!(h, 1002, NIGHT1; attempt = "2").code == 0
    @test all(x -> length(x["comments"]) == 1, h.issues)
    # the third night is green: everything is resolved
    @test nightly!(h, 1003, green_night()).code == 0
    @test all(x -> x["state"] == "closed" && x["reason"] == "completed" && length(x["comments"]) == 2, h.issues)
    @test occursin("Resolved at $SHA, run $SERVER/$REPO/actions/runs/1003", win["comments"][2])
    # the fourth night: Windows fails again
    @test nightly!(h, 1004, [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]).code == 0
    @test win["state"] == "open" && length(win["comments"]) == 3 && occursin("recurred", win["comments"][3])
    @test count(x -> x["state"] == "open", h.issues) == 1
    # a maintainer closes it as not planned: the next failure only comments
    win["state"], win["reason"] = "closed", "not_planned"
    @test nightly!(h, 1005, [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]).code == 0
    @test win["state"] == "closed" && win["reason"] == "not_planned" && length(win["comments"]) == 4
    @test length(h.issues) == 4
  end

  @testset "case 9: dry run, rehearsal, labels" begin
    # --dry-run reads, prints the actions and writes nothing
    h = Hub()
    load!(h, 1001, NIGHT1)
    r = go(h, "triage"; flags = ["--dry-run"])
    @test r.code == 0 && isempty(writes(h)) && isempty(h.issues)
    @test occursin("Dry run: nothing is written.", r.out) && occursin("4 planned actions", r.out)
    @test occursin("create new issue [windows/three_factor_product] Nightly failure on Windows: three_factor_product", r.out)
    @test occursin("labels: nightly-triage, triage:deferred, triage:unclassified", r.out)
    # labels are created once each, before the first issue is written
    h = Hub()
    @test nightly!(h, 1001, NIGHT1).code == 0
    label_calls = [i for (i, c) in enumerate(h.calls) if "repos/$REPO/labels" in c]
    issue_calls = [i for (i, c) in enumerate(h.calls) if "repos/$REPO/issues" in c && "-X" in c && c[findfirst(==("-X"), c) + 1] == "POST"]
    @test length(label_calls) == 4 && length(issue_calls) == 4 && maximum(label_calls) < minimum(issue_calls)
    @test sort([only(c[i + 1] for i in eachindex(c) if c[i] == "-f" && startswith(c[i + 1], "name=")) for c in h.calls if "repos/$REPO/labels" in c]) ==
          ["name=nightly-triage", "name=triage:core", "name=triage:deferred", "name=triage:unclassified"]
    @test sort(h.labels) == ["nightly-triage", "triage:core", "triage:deferred", "triage:unclassified"]
    color = only(c for c in h.calls if "name=triage:core" in c)
    @test "color=b60205" in color && "description=Core CPU correctness failure: fix promptly" in color
    # a label that exists is fine, and a night with only comments creates none
    n_labels = length(h.labels)
    empty!(h.calls)
    @test nightly!(h, 1002, NIGHT1).code == 0
    @test isempty([c for c in h.calls if "repos/$REPO/labels" in c]) && length(h.labels) == n_labels
    label_posts(h) = [i for (i, c) in enumerate(h.calls) if "repos/$REPO/labels" in c]
    first_issue_write(h) = findfirst(c -> "repos/$REPO/issues" in c && "POST" in c, h.calls)
    label_reads(h) = count(c -> "repos/$REPO/labels?per_page=100" in c, h.calls)
    # all the labels exist: they are read (paginated) and none is created
    all_labels = ["nightly-triage", "triage:core", "triage:deferred", "triage:unclassified"]
    h2 = Hub()
    h2.labels = copy(all_labels)
    @test nightly!(h2, 1001, NIGHT1).code == 0 && length(h2.issues) == 4
    @test isempty(label_posts(h2)) && label_reads(h2) == 1 && any(c -> "--paginate" in c && "repos/$REPO/labels?per_page=100" in c, h2.calls)
    # some are missing: only these are created, once each, before the first issue is written
    h2 = Hub()
    h2.labels = ["nightly-triage", "triage:core"]
    @test nightly!(h2, 1001, NIGHT1).code == 0 && length(h2.issues) == 4
    @test length(label_posts(h2)) == 2 && maximum(label_posts(h2)) < first_issue_write(h2)
    @test sort([c[findfirst(==("-f"), c) + 1] for c in h2.calls if "repos/$REPO/labels" in c && "POST" in c]) == ["name=triage:deferred", "name=triage:unclassified"]
    @test sort(h2.labels) == all_labels && label_reads(h2) == 1
    # a creation that fails because another run made the label meanwhile: the new list shows it
    h2 = Hub()
    h2.failing = args -> if "name=triage:core" in args
      push!(h2.labels, "triage:core")
      T.GhError("gh api failed: gh: Validation Failed (HTTP 422)", 422)
    end
    r = nightly!(h2, 1001, NIGHT1)
    @test r.code == 0 && length(h2.issues) == 4 && !occursin("::error::", r.out)
    @test label_reads(h2) == 2 && maximum(label_posts(h2)) < first_issue_write(h2)
    # a creation that fails and leaves the label missing is reported; the issues are still written
    h2 = Hub()
    h2.failing = args -> "name=triage:core" in args ? T.GhError("gh api failed: gh: Validation Failed (HTTP 422)", 422) : nothing
    r = nightly!(h2, 1001, NIGHT1)
    @test r.code == 1 && occursin("::error::could not create the label `triage:core`", r.out) && length(h2.issues) == 4
    @test label_reads(h2) == 2 && !("triage:core" in h2.labels)
    # a second failure to read the list of labels is not taken for success
    h2 = Hub()
    h2.failing = args -> if "name=triage:core" in args
      T.GhError("gh api failed: HTTP 500", 500)
    elseif "repos/$REPO/labels?per_page=100" in args && label_reads(h2) >= 2
      T.GhError("gh api failed: HTTP 503", 503)
    end
    r = nightly!(h2, 1001, NIGHT1)
    @test r.code == 1 && occursin("could not create the label `triage:core`", r.out) && length(h2.issues) == 4
    # --rehearsal: its own label and titles, and it never sees a real issue
    h = Hub()
    @test nightly!(h, 1001, NIGHT1).code == 0
    real = deepcopy(h.issues)
    load!(h, 1002, NIGHT1)
    r = go(h, "triage"; run_id = 1002, flags = ["--rehearsal"])
    @test r.code == 0 && length(h.issues) == 8
    @test h.issues[1:4] == real
    rehearsal = h.issues[5:8]
    @test all(x -> startswith(x["title"], "[rehearsal] ") && "nightly-triage-rehearsal" in x["labels"] && !("nightly-triage" in x["labels"]), rehearsal)
    @test any(c -> "labels=nightly-triage-rehearsal" in c, h.calls)
    # and the other way: a real run never sees a rehearsal issue
    h = Hub()
    load!(h, 1001, NIGHT1)
    @test go(h, "triage"; flags = ["--rehearsal"]).code == 0
    rehearsal = deepcopy(h.issues)
    @test nightly!(h, 1002, NIGHT1).code == 0
    @test length(h.issues) == 8 && h.issues[1:4] == rehearsal && all(x -> isempty(x["comments"]), h.issues[1:4])
    @test all(x -> !startswith(x["title"], "[rehearsal]") && "nightly-triage" in x["labels"], h.issues[5:8])
    # the same in the pure planning, with both kinds of issue in the list
    made = plan(NIGHT1)
    mixed = [T.Issue(i, "open", "", "[rehearsal] " * a.title, a.body, replace(a.labels, "nightly-triage" => "nightly-triage-rehearsal"))
             for (i, a) in enumerate(made)]
    @test kinds(plan(NIGHT1; run = run_of(1002), issues = mixed)) == kinds(made)
    @test all(a -> a.kind == :create, plan(NIGHT1; run = run_of(1002), issues = mixed))
    @test all(a -> a.kind == :comment, plan(NIGHT1; run = run_of(1002), issues = mixed, rehearsal = true))
    @test isempty(plan(green_night(); run = run_of(1002), issues = mixed))
    @test !isempty(plan(green_night(); run = run_of(1002), issues = mixed, rehearsal = true))
    @test all(a -> startswith(a.title, "[rehearsal] "), plan(NIGHT1; rehearsal = true))
    # the rehearsal of the alert has a title of its own
    h = Hub()
    load!(h, 2001, NIGHT1)
    @test go(h, "alert"; event = "push", run_id = 2001, failed = LINUX_1, flags = ["--rehearsal"]).code == 0
    @test h.issues[1]["title"] == "[rehearsal] CI failing on master"
    h.issues[1]["title"] = "CI failing on master"
    @test go(h, "alert"; event = "push", run_id = 2002, failed = LINUX_1, flags = ["--rehearsal"]).code == 0
    @test length(h.issues) == 2 && isempty(h.issues[1]["comments"])
  end

  @testset "case 10: text safety" begin
    @test T.neutralize("ping @user and @org/team, mail a@b, issue #123 and #x") ==
          "ping @$(ZW)user and @$(ZW)org/team, mail a@$(ZW)b, issue #$(ZW)123 and #x"
    @test T.neutralize("<!-- run:9 -->") == "<!$(ZW)-- run:9 -->"
    @test T.inline("a `b` ``c``") == "``` a `b` ``c`` ```"
    @test T.inline("x | y") == "`x | y`" && T.inline("") == "(none)" && T.inline("line\nbreak") == "`line break`"
    @test T.inline("`x`") == "`` `x` ``"
    fence = T.fenced("a\n`````\nb")
    @test startswith(fence, "``````\n") && endswith(fence, "\n``````")
    @test startswith(T.fenced("plain"), "```\nplain\n```")
    title = T.make_title("a\nb @x #1\r\n" * "z"^300)
    @test !occursin(r"[\r\n]", title) && length(title) <= 200 && occursin("@$(ZW)x", title)
    @test T.make_title("x"; rehearsal = true) == "[rehearsal] x"
    capped = T.cap("x"^100000)
    @test length(capped) < 60000 && endswith(capped, "to fit the size limit of an issue._") && T.cap("short") == "short"
    # a failure with @, #, backticks, pipes and a marker in its names and excerpt
    nasty = "set `a` | ``b`` @user #123 <!-- run:9999 -->"
    log = testlog(; failed = ["calculus"], blocks = [failblock(; testset = nasty, path = "$LIN_PATH/calculus.jl", line = 3,
                                                               expr = "x == \"`````\" # @user #123 <!-- run:9999 -->", evaluated = "false")])
    a = only(plan([spec(LINUX_LTS), failed_spec(LINUX_1, log), spec(MAC), spec(WIN), spec(LINUX_PRE)]))
    @test !occursin(r"@(?=user)", a.body) && !occursin(r"#(?=123)", a.body) && occursin("@$(ZW)user", a.body) && occursin("#$(ZW)123", a.body)
    @test !T.has_marker([a.body], "run", "9999") && T.has_marker([a.body], "run", "1001")
    fences = [l for l in split(a.body, '\n') if occursin(r"^`{3,}$", l)]
    @test length(fences) == 2 && length(fences[1]) > 5 && fences[1] == fences[2]
    inline_line = only(l for l in split(a.body, '\n') if startswith(l, "- Test set: "))
    @test occursin(r"^- Test set: ```[^`]", inline_line) && endswith(inline_line, "```") && occursin("| ``b`` @$(ZW)user", inline_line)
    # a very long excerpt and test set name
    long = testlog(; failed = ["calculus"], blocks = [failblock(; testset = "t"^1000, path = "$LIN_PATH/calculus.jl", line = 3,
                                                               expr = "x == " * "ab"^50000, evaluated = "false")])
    a = only(plan([spec(LINUX_LTS), failed_spec(LINUX_1, long), spec(MAC), spec(WIN), spec(LINUX_PRE)]))
    @test length(a.body) < 60000 && length(a.body) > 500 && !occursin("t"^400, a.body)
    # a record with an excerpt of 100000 characters
    rec = T.Record("linux/calculus", "linux", "1", "calculus", "ts", :unclassified, "failed test", "sig", "line\n" * "e"^100000, "", :core,
                   LINUX_1, "https://x/job/1", "https://x/job/1#step:5:1")
    body = T.create_body(POLICY, RUN, "linux/calculus", [rec], :core)
    @test length(body) < 60000 && occursin("(excerpt cut)", body) && count("```", body) == 2
    many = [rec for _ in 1:50]
    body = T.create_body(POLICY, RUN, "linux/calculus", many, :core)
    @test length(body) < 60000 && occursin("and 38 more failures", body)
    @test length(T.cap("y"^70000; limit = 60000)) < 60000
    # a file name that is odd in a key, a title and an argument
    odd = "we ird\$(id) \"q\" ->"
    odd_log = testlog(; blocks = [failblock(; path = "$LIN_PATH/$odd.jl", line = 3, expr = "x", evaluated = "y")])
    a = only(plan([spec(LINUX_LTS), failed_spec(LINUX_1, odd_log), spec(MAC), spec(WIN), spec(LINUX_PRE)]))
    @test a.key == "linux/we_ird__id___q__-_" && occursin("key=linux/we_ird__id___q__-_ -->", a.body)
    @test a.title == "Nightly failure on Linux: we ird\$(id) \"q\" ->"
  end

  @testset "case 11: the adapter against a fake gh" begin
    h = Hub()
    weird = "we ird\$(id) \"q\" ->"
    log = testlog(; blocks = [failblock(; path = "$LIN_PATH/$weird.jl", line = 3, expr = "x", evaluated = "y")])
    load!(h, 1001, [spec(LINUX_LTS), failed_spec(LINUX_1, log), spec(MAC), spec(WIN), spec(LINUX_PRE)])
    @test go(h, "triage").code == 0
    for c in h.calls
      @test c[1] == "api" && all(x -> x isa String, c)
    end
    create = only(c for c in h.calls if "repos/$REPO/issues" in c && "-X" in c && "POST" in c)
    @test "title=Nightly failure on Linux: $weird" in create
    @test count(x -> occursin("we ird", x), create) == 1
    body_arg = only(x for x in create if startswith(x, "body=@"))
    @test !isfile(body_arg[7:end]) # the temporary file is gone
    stored = only(h.issues)["body"]
    @test occursin("key=linux/we_ird", stored) && !any(x -> occursin("key=linux", x), create)
    @test all(x -> !occursin(r"[\r\n]", x), create) && ("labels[]=triage:core" in create) && ("labels[]=nightly-triage" in create)
    @test create[1:4] == ["api", "-X", "POST", "repos/$REPO/issues"]
    # every list is paginated, 100 at a time
    list_jobs = only(c for c in h.calls if any(x -> occursin("/jobs?", x), c))
    @test "--paginate" in list_jobs && any(x -> occursin("per_page=100", x), list_jobs) && any(x -> occursin("filter=latest", x), list_jobs)
    list_issues = only(c for c in h.calls if "repos/$REPO/issues" in c && "GET" in c)
    @test "--paginate" in list_issues && "per_page=100" in list_issues && "state=all" in list_issues && "labels=nightly-triage" in list_issues
    @test list_issues[2:3] == ["-X", "GET"]
    log_call = only(c for c in h.calls if any(x -> endswith(x, "/logs"), c))
    @test log_call == ["api", "repos/$REPO/actions/jobs/10012/logs"]
    # comments: one call per issue it may touch, paginated
    load!(h, 1002, [spec(n) for n in NAMES])
    empty!(h.calls)
    @test go(h, "triage"; run_id = 1002).code == 0
    comment_calls = [c for c in h.calls if any(x -> occursin("/comments", x), c) && !("POST" in c)]
    @test length(comment_calls) == 1 && "--paginate" in comment_calls[1] && any(x -> occursin("per_page=100", x), comment_calls[1])
    @test length(writes(h)) == 2 # the resolving comment and the closing
    closing = only(c for c in writes(h) if "state=closed" in c)
    @test "state_reason=completed" in closing && "-f" in closing
    # the body of a comment goes through a file as well
    post = only(c for c in writes(h) if any(x -> endswith(x, "/comments"), c))
    @test any(x -> startswith(x, "body=@"), post) && !any(x -> occursin("Resolved", x), post)
    @test occursin("Resolved at", only(h.issues)["comments"][1])

    # parsing of the canned output of the jq programs
    odd_body = "tab\there\nline two\r\nnon-ASCII: é ✓ 日本\n"
    issues_text = join([join(["12", "closed", "not_planned", base64encode("T\tA"), base64encode(odd_body), base64encode("nightly-triage\ntriage:flaky")], "\t"),
                        join(["13", "open", "", base64encode("title"), "", ""], "\t")], "\n") * "\n"
    parsed = T.parse_issues(issues_text)
    @test length(parsed) == 2
    @test eqv(parsed[1], T.Issue(12, "closed", "not_planned", "T\tA", odd_body, ["nightly-triage", "triage:flaky"]))
    @test eqv(parsed[2], T.Issue(13, "open", "", "title", "", String[]))
    @test T.parse_issues("") == T.Issue[]
    @test T.parse_comments(join(base64encode.([odd_body, "second"]), "\n") * "\n") == [odd_body, "second"]
    steps = base64encode("1\tsuccess\tSet up job\n2\tfailure\tRun package tests\n3\t\tPost\\ttab")
    jobs = T.parse_jobs("42\tJulia 1 - windows-latest - x64 - full\tfailure\t2\thttps://x/job/42\t$steps\n43\ta\\tb\t\t1\thttps://x/job/43\t\n")
    @test eqv(jobs[1], T.Job("42", WIN, "failure", "2", "https://x/job/42", [T.Step(1, "Set up job", "success"), T.Step(2, T.TEST_STEP, "failure"), T.Step(3, "Post\ttab", "")]))
    @test jobs[2].name == "a\tb" && jobs[2].conclusion == "" && isempty(jobs[2].steps)
    @test eqv(T.parse_runs("2026-11-05T03:00:00Z\tcompleted\tsuccess\n"), [T.Scheduled(T.epoch(2026, 11, 5, 3), "completed", "success")])
    @test T.unescape_tsv("a\\tb\\nc\\\\d\\re") == "a\tb\nc\\d\re"
    @test_throws ErrorException T.parse_jobs("only\tthree\tfields\n")
    # an invalid UTF-8 byte in a body is replaced, not thrown at
    @test T.decode64(base64encode(UInt8[0x61, 0xff, 0x62])) == "a\ufffdb"
    # a log: found, not found, gone, or an error that is not data
    hub = Hub()
    hub.logs["5"] = "text"
    hub.logs["6"] = T.GhError("gone", 410)
    hub.logs["7"] = T.GhError("boom", 500)
    run = run_of(1)
    @test T.read_log(hub, run, "5") == "text" && T.read_log(hub, run, "4") === nothing && T.read_log(hub, run, "6") === nothing
    @test_throws T.GhError T.read_log(hub, run, "7")
    # apply!: the failure count, and the order of the writes
    hub = Hub()
    acts = [T.Action(:create, nothing, "k"; title = "A", body = "b", labels = ["nightly-triage"]), T.Action(:comment, 1, "k"; body = "c"),
            T.Action(:reopen, 1, "k"), T.Action(:close, 1, "k"), T.Action(:label, 1, "k"; labels = ["triage:core"])]
    lines = String[]
    @test T.apply!(acts, hub; repo = REPO, policy = POLICY, log = (xs...) -> push!(lines, string(xs...))) == 0
    @test hub.issues[1]["state"] == "closed" && hub.issues[1]["comments"] == ["c"] && hub.issues[1]["labels"] == ["nightly-triage", "triage:core"]
    @test length(lines) == 5 && startswith(lines[1], "done: create new issue [k] A")
    @test T.apply!(T.Action[], hub; repo = REPO, policy = POLICY) == 0
    @test_throws ErrorException T.write_action(hub, REPO, mktempdir(), 1, T.Action(:nonsense, 1, "k"))
  end

  @testset "retries: the state is reconciled, a comment is written once" begin
    pairs(actions) = [(a.kind, a.number) for a in actions]
    win_fails = [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
    # in the planning: an earlier attempt of this run wrote its comment, the rest of the work is still planned
    made = by_key(plan(NIGHT1))
    closed = issue_from(made["windows/three_factor_product"], 4; state = "closed", reason = "completed")
    first_try = plan(win_fails; run = run_of(1005), issues = [closed])
    @test pairs(first_try) == [(:comment, 4), (:reopen, 4)]
    @test pairs(plan(win_fails; run = run_of(1005, attempt = "2"), issues = [closed], comments = commented(first_try))) == [(:reopen, 4)]
    reopened = issue_from(made["windows/three_factor_product"], 4)
    @test isempty(plan(win_fails; run = run_of(1005, attempt = "2"), issues = [reopened], comments = commented(first_try)))
    # closed as not planned: the comment is not repeated, and nothing else is planned
    not_planned = issue_from(made["windows/three_factor_product"], 4; state = "closed", reason = "not_planned")
    np = plan(win_fails; run = run_of(1005), issues = [not_planned])
    @test pairs(np) == [(:comment, 4)] && isempty(plan(win_fails; run = run_of(1005), issues = [not_planned], comments = commented(np)))
    # the severity label is planned whenever it is missing, with or without its comment
    pre = plan([spec(LINUX_LTS), spec(LINUX_1), spec(MAC), spec(WIN), failed_spec(LINUX_PRE, LIN1_LOG)])
    deferred_issue = issues_of(pre)
    core_night = [spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)]
    esc = plan(core_night; run = run_of(1006), issues = deferred_issue)
    @test pairs(esc) == [(:comment, 1), (:label, 1)]
    @test pairs(plan(core_night; run = run_of(1006, attempt = "2"), issues = deferred_issue, comments = commented(esc))) == [(:label, 1)]
    done = [T.Issue(1, "open", "", deferred_issue[1].title, deferred_issue[1].body, [deferred_issue[1].labels; "triage:core"])]
    @test isempty(plan(core_night; run = run_of(1006, attempt = "2"), issues = done, comments = commented(esc)))
    # a resolution: the close is planned while the issue is open, the comment once
    open_issues = issues_of(plan(NIGHT1))
    resolving = plan(green_night(); run = run_of(1007), issues = open_issues)
    @test pairs(resolving) == [(:comment, 1), (:close, 1), (:comment, 2), (:close, 2), (:comment, 3), (:close, 3), (:comment, 4), (:close, 4)]
    again = plan(green_night(); run = run_of(1007, attempt = "2"), issues = open_issues, comments = commented(resolving))
    @test pairs(again) == [(:close, 1), (:close, 2), (:close, 3), (:close, 4)]
    # the flaky note has a marker of its own, and a failed attempt's comment does not hide a later resolution
    failed_attempt = plan(NIGHT1; run = run_of(1008), issues = open_issues)
    @test pairs(plan(green_night(); run = run_of(1008, attempt = "2"), issues = open_issues, comments = commented(failed_attempt))) ==
          pairs(resolving)
  end

  # Every subset of failing writes of the first attempt of a run, then the same run again, and the
  # next night. Each scenario is a state of the hub (`setup`) and a run that changes it (`exercise`).
  @testset "fault injection: every subset of failing writes, then a replan" begin
    all_labels = ["nightly-triage", "triage:core", "triage:deferred", "triage:unclassified", "triage:infrastructure"]
    win_fails = [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
    both_fail = [spec(LINUX_LTS), spec(LINUX_1), failed_spec(MAC, MAC_LOG), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
    mac_only = [spec(LINUX_LTS), spec(LINUX_1), failed_spec(MAC, MAC_LOG), spec(WIN), spec(LINUX_PRE)]
    pre_fails = [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), spec(WIN), failed_spec(LINUX_PRE, LIN1_LOG)]
    core_fails = [spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), spec(WIN), spec(LINUX_PRE)]
    mixed = [spec(LINUX_LTS), failed_spec(LINUX_1, LIN1_LOG), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
    nightly(specs) = (h, id, attempt, shift) -> nightly!(h, id, specs; attempt)
    stale(h, id, attempt, shift) = begin
      h.runs = [("2026-11-08T02:00:00Z", "completed", "success")]
      go(h, "stale"; event = "push", run_id = id, attempt, now = NOW + shift)
    end

    function fresh(setup)
      h = Hub()
      h.labels = copy(all_labels)
      setup(h)
      empty!(h.calls)
      empty!(h.events)
      return h
    end

    # which write a call is: "comment:7", "reopen:7", "close:7", "label:7" or "create:<title>"
    function write_id(args)
      i = findfirst(==("-X"), args)
      (i === nothing || !(args[i + 1] in ("POST", "PATCH"))) && return nothing
      endpoint = only(a for a in args if startswith(a, "repos/"))
      n = (m = match(r"/issues/(\d+)", endpoint)) === nothing ? 0 : parse(Int, m.captures[1])
      endswith(endpoint, "/comments") && return "comment:$n"
      occursin(r"/issues/\d+/labels$", endpoint) && return "label:$n"
      endswith(endpoint, "/labels") && return "newlabel"
      args[i + 1] == "PATCH" && return "state=open" in args ? "reopen:$n" : "close:$n"
      return "create:" * only(a[7:end] for a in args if startswith(a, "title="))
    end
    faults(bad, fired) = args -> begin
      id = write_id(args)
      if id !== nothing && id in bad
        fired[] += 1
        T.GhError("gh api failed: HTTP 500", 500)
      end
    end
    no_faults(args) = nothing

    attempt_less(c) = replace(c, r"\(attempt \d+\)" => "(attempt #)")
    snapshot(h) = sort([(x["title"], x["state"], sort(x["labels"]), sort(attempt_less.(x["comments"]))) for x in h.issues]; by = first)
    states(h) = sort([(x["title"], x["state"], sort(x["labels"])) for x in h.issues]; by = first)
    # how often each comment marker of a run occurs in an issue (a comment is written once per run id)
    function marker_counts(h)
      out = Dict{Tuple{String, String, String}, Int}()
      for x in h.issues, t in [x["body"]; x["comments"]], m in eachmatch(r"<!-- (run|resolved|passed|recovered|stale):(\S+) -->", t)
        key = (x["title"], String(m.captures[1]), String(m.captures[2]))
        out[key] = get(out, key, 0) + 1
      end
      return out
    end
    # a state change is never in effect without the comment that goes with it (when the plan has one)
    function comment_first(events, ids)
      for (p, (kind, n)) in enumerate(events)
        kind in (:reopen, :close, :label) && "comment:$n" in ids || continue
        (:comment, n) in events[1:p - 1] || return false
      end
      return true
    end
    effect(e) = (e[1], e[1] == :create ? 0 : e[2])

    function explore(name, setup, exercise; writes_expected)
      ref = fresh(setup)
      @test exercise(ref, 5001, "1", 0).code == 0
      ids = String[write_id(c) for c in writes(ref)]
      @test length(ids) == writes_expected && !("newlabel" in ids)
      ref_snapshot, ref_states = snapshot(ref), states(ref)
      for mask in 0:(2^length(ids) - 1)
        bad = Set(ids[i] for i in eachindex(ids) if (mask >> (i - 1)) & 1 == 1)
        # the first attempt: what happens is what the dependency rule says, and nothing else
        h = fresh(setup)
        fired = Ref(0)
        h.failing = faults(bad, fired)
        first_try = exercise(h, 5001, "1", 0)
        blocked, expected, fired_expected, skipped = Set{Int}(), Tuple{Symbol, Int}[], 0, 0
        for id in ids
          kind, rest = split(id, ':'; limit = 2)
          n = kind == "create" ? 0 : parse(Int, rest)
          if kind in ("reopen", "close", "label") && n in blocked
            skipped += 1
          elseif id in bad
            fired_expected += 1
            kind == "comment" && push!(blocked, n)
          else
            push!(expected, (Symbol(kind), n))
          end
        end
        @test (name, mask, effect.(h.events)) == (name, mask, expected)
        @test (name, mask, fired[], first_try.code, count("::error::could not write", first_try.out), count("skipped:", first_try.out)) ==
              (name, mask, fired_expected, fired_expected > 0 ? 1 : 0, fired_expected, skipped)
        @test (name, mask, comment_first(h.events, ids)) == (name, mask, true)
        # the same run again, with no failure: the end state is the failure-free one, each comment once
        h.failing = no_faults
        @test (name, mask, exercise(h, 5001, "2", 0).code) == (name, mask, 0)
        @test (name, mask, snapshot(h)) == (name, mask, ref_snapshot)
        @test (name, mask, comment_first(h.events, ids), all(==(1), values(marker_counts(h)))) == (name, mask, true, true)
        empty!(h.calls)
        @test exercise(h, 5001, "3", 0).code == 0 && isempty(writes(h)) # nothing is left to do
        # the next night instead: a new run id repairs the state, and every change has its own comment
        next = fresh(setup)
        next.failing = faults(bad, Ref(0))
        exercise(next, 5001, "1", 0)
        next.failing = no_faults
        @test (name, mask, exercise(next, 5002, "1", 86400).code) == (name, mask, 0)
        @test (name, mask, states(next)) == (name, mask, ref_states)
        @test (name, mask, comment_first(next.events, ids), all(==(1), values(marker_counts(next)))) == (name, mask, true, true)
        for x in next.issues
          text = join(x["comments"], "\n")
          x["state"] == "closed" && x["reason"] == "completed" &&
            @test (name, mask, occursin(r"<!-- (resolved|recovered):", text)) == (name, mask, true)
          x["reason"] == "reopened" && @test (name, mask, occursin("recurred", text)) == (name, mask, true)
        end
      end
    end

    explore("recurrence", h -> (nightly!(h, 1001, win_fails); nightly!(h, 1002, green_night())), nightly(win_fails); writes_expected = 2)
    explore("resolution of two issues", h -> nightly!(h, 1001, both_fail), nightly(green_night()); writes_expected = 4)
    explore("severity escalation", h -> nightly!(h, 1001, pre_fails), nightly(core_fails); writes_expected = 2)
    explore("flaky note", h -> (nightly!(h, 1001, win_fails); push!(h.issues[1]["labels"], "triage:flaky")), nightly(green_night()); writes_expected = 1)
    explore("closed as not planned", h -> (nightly!(h, 1001, win_fails); h.issues[1]["state"] = "closed"; h.issues[1]["reason"] = "not_planned"),
            nightly(win_fails); writes_expected = 1)
    explore("new issues", h -> nothing, nightly(both_fail); writes_expected = 2)
    explore("recurrence, resolution and a new issue in one plan",
            h -> (nightly!(h, 1001, both_fail); nightly!(h, 1002, mac_only)), nightly(mixed); writes_expected = 5)
    explore("stale nightly recovered", h -> begin
              h.runs = [("2026-11-04T03:00:00Z", "completed", "success")]
              go(h, "stale"; event = "push", run_id = 3001)
            end, stale; writes_expected = 2)

    # an issue whose comment fails does not hold up the others
    h = fresh(h -> (nightly!(h, 1001, both_fail); nightly!(h, 1002, mac_only)))
    win = titled(h, "Nightly failure on Windows: three_factor_product")
    mac = titled(h, "Nightly failure on macOS: three_factor_product")
    h.failing = args -> any(a -> endswith(a, "/issues/$(win["number"])/comments"), args) ? T.GhError("gh api failed: HTTP 500", 500) : nothing
    r = nightly!(h, 5001, mixed)
    @test r.code == 1 && win["state"] == "closed" && length(win["comments"]) == 1
    @test mac["state"] == "closed" && length(mac["comments"]) == 2
    @test any(x -> x["title"] == "Nightly failure on Linux: calculus", h.issues)
    @test occursin("skipped: reopen issue #$(win["number"])", r.out) && count("::error::could not write", r.out) == 1
  end

  @testset "rehearsal and normal runs own disjoint issues" begin
    for (labels, title, normal, rehearsal) in (
        (["nightly-triage"], "t", true, false),
        (["nightly-triage"], "[rehearsal] t", true, false),
        (["nightly-triage-rehearsal"], "[rehearsal] t", false, true),
        (["nightly-triage-rehearsal"], "t", false, false),
        (["nightly-triage", "nightly-triage-rehearsal"], "[rehearsal] t", false, false),
        (["nightly-triage", "nightly-triage-rehearsal"], "t", false, false),
        (String[], "[rehearsal] t", false, false))
      i = T.Issue(1, "open", "", title, "", labels)
      @test T.in_scope(POLICY, i, false) == normal
      @test T.in_scope(POLICY, i, true) == rehearsal
    end
    @test T.conflicting(POLICY, [T.Issue(9, "open", "", "a", "", ["nightly-triage", "nightly-triage-rehearsal"]),
                                 T.Issue(3, "closed", "", "a", "", ["nightly-triage-rehearsal", "nightly-triage"]),
                                 T.Issue(5, "open", "", "a", "", ["nightly-triage"])]) == [3, 9]
    win_fails = [spec(LINUX_LTS), spec(LINUX_1), spec(MAC), failed_spec(WIN, WIN_LOG), spec(LINUX_PRE)]
    made = by_key(plan(win_fails))["windows/three_factor_product"]
    # an issue with both labels is untouched in both modes: no comment when it fails, no close when it passes
    both = [T.Issue(1, "open", "", "[rehearsal] " * made.title, made.body, ["nightly-triage", "nightly-triage-rehearsal"])]
    for rehearsal in (false, true)
      @test kinds(plan(win_fails; run = run_of(1002), issues = both, rehearsal)) == [(:create, nothing, "windows/three_factor_product")]
      @test isempty(plan(green_night(); run = run_of(1002), issues = both, rehearsal))
      @test isempty(T.issues_to_read(POLICY, both, Set(["windows/three_factor_product"]); rehearsal))
    end
    # a rehearsal issue whose title lacks the prefix is untouched in a rehearsal
    bare = [T.Issue(1, "open", "", made.title, made.body, ["nightly-triage-rehearsal"])]
    @test kinds(plan(win_fails; run = run_of(1002), issues = bare, rehearsal = true)) == [(:create, nothing, "windows/three_factor_product")]
    @test isempty(plan(green_night(); run = run_of(1002), issues = bare, rehearsal = true))
    @test isempty(T.issues_to_read(POLICY, bare, Set(["windows/three_factor_product"]); rehearsal = true))
    # through the hub: untouched, and one warning per such issue
    for flags in (String[], ["--rehearsal"])
      h = Hub()
      push!(h.issues, Dict{String, Any}("number" => 1, "state" => "open", "reason" => "", "title" => "[rehearsal] " * made.title,
                                        "body" => made.body, "labels" => ["nightly-triage", "nightly-triage-rehearsal"], "comments" => String[]))
      load!(h, 1002, green_night())
      r = go(h, "triage"; run_id = 1002, flags)
      @test r.code == 0 && h.issues[1]["state"] == "open" && isempty(h.issues[1]["comments"]) && isempty(writes(h))
      @test count("::warning::", r.out) == 1 && occursin("::warning::Issue #1 has both the label `nightly-triage` and the label `nightly-triage-rehearsal`", r.out)
      load!(h, 1003, win_fails)
      r = go(h, "triage"; run_id = 1003, flags)
      @test r.code == 0 && length(h.issues) == 2 && isempty(h.issues[1]["comments"]) && h.issues[1]["state"] == "open"
      @test count("::warning::", r.out) == 1
    end
  end

  @testset "the command line" begin
    h = Hub()
    for args in (String[], ["triage", "alert"], ["bogus"], ["triage", "--dry"], ["--dry-run"])
      io, errio = IOBuffer(), IOBuffer()
      @test T.main(args, env(); gh = h, io, err = errio) == 2
      @test startswith(String(take!(errio)), "usage: julia scripts/ci_triage.jl")
    end
    io, errio = IOBuffer(), IOBuffer()
    @test T.main(["triage", "--dry-run"], Dict{String, String}(); gh = h, io, err = errio) == 2
    message = String(take!(errio))
    @test occursin("GITHUB_REPOSITORY", message) && occursin("usage:", message) && !occursin("Stacktrace", message) && !occursin("ERROR:", message)
    @test isempty(h.calls)
    bad = env(); bad["GITHUB_SHA"] = "not a sha"; bad["GITHUB_RUN_ID"] = "x"
    @test T.main(["triage"], bad; gh = h, io, err = errio) == 2
    @test occursin("GITHUB_SHA", String(take!(errio)))
    # a broken policy file is an error of the configuration
    path = joinpath(mktempdir(), "rules.toml")
    write(path, "[triage]\nstale_hours = \"x\"\n")
    io, errio = IOBuffer(), IOBuffer()
    @test T.main(["triage"], env(); gh = h, rules = path, io, err = errio) == 2
    @test occursin("triage.stale_hours must be a positive number of hours", String(take!(io)))
    # the usual flags are accepted in either order
    load!(h, 1001, green_night())
    @test T.main(["--rehearsal", "triage", "--dry-run"], env(); gh = h, now = () -> NOW, io = IOBuffer(), err = errio) == 0
  end

  @testset "case 12: the [triage] table" begin
    data = TOML.parsefile(RULES)
    p = T.load_policy(RULES)
    @test p.stale_hours == 48 && p.title_alert == "CI failing on master" && p.title_stale == "Nightly CI is stale"
    @test p.label == "nightly-triage" && p.rehearsal_label == "nightly-triage-rehearsal"
    @test p.severity_labels == Dict("core" => "triage:core", "deferred" => "triage:deferred")
    @test p.kind_labels == ["triage:unclassified", "triage:bug", "triage:flaky", "triage:portability", "triage:infrastructure"]
    @test all(l -> haskey(p.styles, l), [p.label, p.rehearsal_label, values(p.severity_labels)..., p.kind_labels...])
    @test length(p.deferred) >= 2
    # every deferred glob matches a test of test/tiers.toml, and a name nobody listed is core
    manifest = T.CIResult.default_manifest()
    names = sort!(collect(keys(manifest.tier)))
    for d in p.deferred, (glob, re) in zip(d.files, d.regexes)
      @test any(n -> occursin(re, n), names)
      @test glob in names # exact names, not broader globs
    end
    for name in names
      listed = any(d -> any(g -> g == name, d.files), p.deferred)
      @test T.is_deferred(p, name) == listed
      @test T.severity(p, "linux", "1", name) == (listed ? :deferred : :core)
    end
    @test T.severity(p, "linux", "1", "a_test_nobody_listed") == :core
    @test T.is_deferred(p, "gpu") && T.is_deferred(p, "conformance/gpu_arrays") && !T.is_deferred(p, "calculus")
    @test T.deferred_group(p, "extreme_normals").why == "extreme precision and exponent ranges"
    @test T.deferred_group(p, "extreme_normals_grids").why == "extreme precision and exponent ranges"
    # globs follow test/tiers.jl
    glob = T.parse_policy(let d = deepcopy(data); d["triage"]["deferred"][1]["files"] = ["gpu*", "conformance/*"]; d end)
    @test T.is_deferred(glob, "gpu_smoke") && T.is_deferred(glob, "conformance/kernels") && !T.is_deferred(glob, "conformance/a/b")

    function bad(mutate)
      d = deepcopy(data)
      mutate(d["triage"])
      try
        T.parse_policy(d)
        return nothing
      catch e
        e isa T.PolicyError || rethrow()
        return e.problems
      end
    end
    has(problems, text) = problems !== nothing && any(p -> occursin(text, p), problems)
    @test bad(t -> nothing) === nothing
    @test has(bad(t -> t["bogus"] = 1), "unknown key `bogus`")
    @test has(bad(t -> delete!(t, "label")), "triage.label is missing")
    @test has(bad(t -> t["stale_hours"] = "48"), "triage.stale_hours must be a positive number of hours")
    @test has(bad(t -> t["stale_hours"] = 0), "triage.stale_hours must be a positive number of hours")
    @test has(bad(t -> t["stale_hours"] = true), "triage.stale_hours must be a positive number of hours")
    @test has(bad(t -> t["title_alert"] = 5), "triage.title_alert must be a non-empty string")
    @test has(bad(t -> t["title_alert"] = "two\nlines"), "triage.title_alert must be a non-empty string")
    @test has(bad(t -> t["title_stale"] = t["title_alert"]), "triage.title_alert and triage.title_stale must differ")
    @test has(bad(t -> t["rehearsal_label"] = t["label"]), "triage.label and triage.rehearsal_label must differ")
    @test has(bad(t -> t["severity_labels"]["core"] = 5), "triage.severity_labels.core must be a non-empty string")
    @test has(bad(t -> t["severity_labels"]["urgent"] = "x"), "triage.severity_labels has an unknown key `urgent`")
    @test has(bad(t -> delete!(t["severity_labels"], "deferred")), "triage.severity_labels.deferred must be a non-empty string")
    @test has(bad(t -> t["severity_labels"] = ["triage:core"]), "triage.severity_labels must be a table")
    @test has(bad(t -> t["kind_labels"] = ["triage:bug"]), "triage.kind_labels must include `triage:unclassified`")
    @test has(bad(t -> t["kind_labels"] = ["triage:bug"]), "triage.kind_labels must include `triage:flaky`")
    @test has(bad(t -> t["kind_labels"] = "triage:bug"), "triage.kind_labels must be a list of label names")
    @test has(bad(t -> push!(t["kind_labels"], "triage:bug")), "triage.kind_labels lists a label twice")
    @test has(bad(t -> delete!(t["deferred"][1], "files")), "triage.deferred[1] needs `files`")
    @test has(bad(t -> t["deferred"][1]["files"] = String[]), "triage.deferred[1] needs `files`")
    @test has(bad(t -> t["deferred"][2]["files"] = [1]), "triage.deferred[2] needs `files`")
    @test has(bad(t -> t["deferred"][1]["why"] = ""), "triage.deferred[1] needs a non-empty string `why`")
    @test has(bad(t -> t["deferred"][1]["extra"] = 1), "triage.deferred[1] has an unknown key `extra`")
    @test has(bad(t -> t["deferred"] = "gpu"), "triage.deferred must be a list of tables")
    @test has(bad(t -> t["deferred"] = [3]), "triage.deferred[1] must be a table")
    @test has(bad(t -> delete!(t["label_style"], "triage:bug")), "no entry for the label `triage:bug`")
    @test has(bad(t -> t["label_style"]["stray"] = Dict("color" => "ffffff", "description" => "x")), "the entry `stray`")
    @test has(bad(t -> t["label_style"]["triage:bug"]["color"] = "red"), "needs `color`, six hexadecimal digits")
    @test has(bad(t -> t["label_style"]["triage:bug"]["description"] = "x"^101), "1 to 100 characters")
    @test has(bad(t -> t["label_style"]["triage:bug"]["size"] = 1), "unknown key `size`")
    @test has(bad(t -> t["label_style"]["triage:bug"] = "red"), "must be a table with `color` and `description`")
    @test has(bad(t -> t["label_style"] = 3), "triage.label_style must be a table")
    # all the problems are listed together
    many = bad(t -> (t["stale_hours"] = "x"; delete!(t, "label"); t["bogus"] = 1; t["deferred"][1]["why"] = ""))
    @test length(many) >= 4 && has(many, "stale_hours") && has(many, "triage.label is missing") && has(many, "`bogus`") && has(many, "`why`")
    # a file with no [triage] table
    err = try
      T.parse_policy(Dict{String, Any}("events" => Dict{String, Any}()))
    catch e
      e
    end
    @test err isa T.PolicyError && err.problems == ["[triage] is missing"]
    @test occursin("[triage] is missing", sprint(showerror, err))
    @test occursin("has problems:\n  - triage has an unknown key `bogus`", sprint(showerror, T.PolicyError(["triage has an unknown key `bogus`"])))
  end

end

end # module
