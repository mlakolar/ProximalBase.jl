module PlatformFocusTest

# The helper of the on-demand focused platform workflow (scripts/platform_focus.jl and
# .github/workflows/platform-focused.yml): that the selector runs exactly one test file whatever
# its name, that every input is checked before a job on another system starts and that a rejected
# input cannot start a workflow command, and that the workflow reads what the script writes.

using Test
import ParallelTestRunner as PTR

module PF
include(joinpath(dirname(@__DIR__), "scripts", "platform_focus.jl"))
end

const ROOT = dirname(@__DIR__)
const WORKFLOW = joinpath(ROOT, ".github", "workflows", "platform-focused.yml")
const CI_WORKFLOW = joinpath(ROOT, ".github", "workflows", "CI.yml")
const RULES = joinpath(ROOT, ".github", "tier-rules.toml")
const TT = PF.TestTiers
const MANIFEST = TT.load(joinpath(ROOT, "test", "tiers.toml"))
const NAMES = sort!(collect(keys(MANIFEST.tier)))

# The same text, with LF endings (a Windows checkout may have CRLF).
lf(path) = replace(read(path, String), "\r\n" => "\n")

const SHA = "0123456789abcdef0123456789abcdef01234567"

planned(; os = "both", test_file = "three_factor_product", julia_version = "1", expected_sha = "",
        sha = SHA, head = sha, root = ROOT, manifest = MANIFEST) =
  PF.plan(Dict("os" => os, "test_file" => test_file, "julia_version" => julia_version, "expected_sha" => expected_sha);
          root, manifest, github_sha = sha, head)

# The files that the runner keeps for a selector, as test/runtests.jl applies it to the whole catalog.
function kept(names, selector)
  suite = Dict{String, Any}(n => nothing for n in names)
  PTR.filter_tests!(suite, PTR.parse_args(selector; custom = ["tier", "extra", "core"]))
  return sort!(collect(keys(suite)))
end

# The problems of one input; every message is a single line.
function problems(; kw...)
  p = planned(; kw...)
  @test p.result === nothing
  @test all(m -> !occursin(r"[\r\n]", m), p.problems)
  return p.problems
end
refused(phrase; kw...) = (p = problems(; kw...); length(p) == 1 && occursin(phrase, only(p)))

@testset "the selector selects exactly one file" begin
  # synthetic names, with prefixes within a directory too
  names = ["a", "ab", "abc", "a/b", "b", "b/c"]
  @test PF.selector(names, "a") == ["a", "!a/b", "!ab", "!abc"]
  @test PF.selector(names, "ab") == ["ab", "!abc"]
  @test PF.selector(names, "abc") == ["abc"]
  @test PF.selector(names, "a/b") == ["a/b"]
  @test PF.selector(names, "b") == ["b", "!b/c"]
  for name in names
    @test kept(names, PF.selector(names, name)) == [name]
  end

  # the whole catalog, with the real runner functions: exactly the named file, and nothing that
  # runtests.jl would reject or warn about
  @test length(NAMES) > 100
  prefix_cases = [n for n in NAMES if any(m -> m != n && startswith(m, n), NAMES)]
  @test !isempty(prefix_cases)
  wrong, unclean = String[], String[]
  for name in NAMES
    selector = PF.selector(NAMES, name)
    got = kept(NAMES, selector)
    got == [name] || push!(wrong, "$name selects $(join(got, " "))")
    errors, warnings = TT.check_selectors(MANIFEST, NAMES, NAMES, selector)
    isempty(errors) && isempty(warnings) || push!(unclean, "$name: $(join([errors; warnings], "; "))")
  end
  @test wrong == String[]
  @test unclean == String[]
  # without the exclusions the name alone selects more than one file: the property is not vacuous
  for name in prefix_cases
    @test length(kept(NAMES, [name])) > 1
  end
  # a name that is not a prefix of another needs no exclusion
  @test PF.selector(NAMES, "three_factor_product") == ["three_factor_product"]
  # the selector of the plan is the same list, as one string of words without spaces or globs
  p = planned(test_file = first(prefix_cases))
  @test split(p.result.selector, ' ') == PF.selector(NAMES, first(prefix_cases))
  @test !occursin(r"[*?\[\]{}$`\\\"';&|<>()]", p.result.selector)
end

@testset "test names: refusals say why" begin
  @test planned(test_file = "three_factor_product").result.test_file == "three_factor_product"
  @test planned(test_file = "conformance/tolerances").result.safe_name == "conformance-tolerances"
  @test planned(test_file = "conformance/tolerances").result.selector == "conformance/tolerances"
  @test refused("is empty"; test_file = "")
  # a prefix of tests, a directory, a helper, an environment, an unknown name
  @test refused("only the start of"; test_file = "three_factor")
  @test occursin("three_factor_product", only(problems(test_file = "three_factor")))
  @test refused("is a directory of"; test_file = "conformance")
  @test occursin("name one of them: conformance/", only(problems(test_file = "conformance")))
  @test refused("is a helper that test files include"; test_file = "testutils")
  @test occursin("tier_selection", only(problems(test_file = "tiers")))
  @test refused("environment of its own"; test_file = "cuda")
  @test refused("environment of its own"; test_file = "oracles/anything")
  @test refused("is not a test in test/tiers.toml"; test_file = "no_such_test")
  # a form that is not a name, whatever it holds
  for bad in (" ", "a b", "three_factor_product ", " three_factor_product", "three_factor_product\n", "..", "../x", "a/../b",
              "/etc/passwd", "C:\\x", "a//b", "a/", "/a", "a;b", "a&&b", "a|b", "\$(id)", "\${HOME}", "`id`", "a\nb",
              "a\r\nb", "a\n::warning::injected", "::error::x", "%0A", "a*", "a?", "é", "!three_factor_product",
              "!!x", "--tier=full", "-x", "--extra=gpu", "three_factor_product --tier=full")
    @test refused("is not a test name"; test_file = bad)
  end
  @test occursin("an option or an exclusion", only(problems(test_file = "--tier=full")))
  @test occursin("an option or an exclusion", only(problems(test_file = "!x")))
  @test !occursin("an option or an exclusion", only(problems(test_file = "a;b")))
  # too long: refused without being repeated
  long = repeat("a", PF.MAX_NAME + 1)
  @test refused("bytes long"; test_file = long)
  @test !occursin(long, only(problems(test_file = long)))
  @test PF.MAX_NAME == 100
  # a name at the limit has the right form (it is not in the catalog)
  @test refused("is not a test in test/tiers.toml"; test_file = repeat("a", PF.MAX_NAME))
end

# A repository of its own: a catalog with a prefix, a directory, a helper, an environment, and a
# listed name whose file is missing.
const FIXTURE = """
[tiers]
quick = ["alpha", "alpha_two", "gone", "dir/leaf"]
standard = ["std"]
thorough = ["thr"]
full = ["ful"]

[not_tests]
helpers = ["help"]
environments = ["env"]

[not_tests.run_through]
help = ["alpha"]
"""

function with_fixture(f; julia = "1.10")
  mktempdir() do dir
    mkpath(joinpath(dir, "test", "dir"))
    write(joinpath(dir, "Project.toml"), "name = \"Fixture\"\n\n[compat]\njulia = \"$julia\"\n")
    for name in ("alpha", "alpha_two", "dir/leaf", "std", "thr", "ful")
      write(joinpath(dir, "test", name * ".jl"), "# $name\n")
    end
    f(dir, TT.parse_manifest(FIXTURE))
  end
end

@testset "a declared name whose file is missing; a fixture catalog" begin
  with_fixture() do dir, manifest
    ok(name) = planned(; test_file = name, root = dir, manifest)
    @test ok("alpha").result.selector == "alpha !alpha_two"
    @test ok("alpha_two").result.selector == "alpha_two"
    @test ok("dir/leaf").result.safe_name == "dir-leaf"
    p = ok("gone")
    @test p.result === nothing && length(p.problems) == 1
    @test occursin("test/gone.jl does not exist", only(p.problems)) && occursin("listed in test/tiers.toml", only(p.problems))
    @test occursin("only the start of 2 test names", only(ok("alph").problems))
    @test occursin("is a directory of 1 tests", only(ok("dir").problems))
    @test occursin("it runs through alpha", only(ok("help").problems))
    @test occursin("environment of its own", only(ok("env").problems))
    @test occursin("environment of its own", only(ok("env/x").problems))
  end
  # a catalog that cannot be read is a problem, not a crash
  mktempdir() do dir
    mkpath(joinpath(dir, "test"))
    write(joinpath(dir, "Project.toml"), "[compat]\njulia = \"1.10\"\n")
    write(joinpath(dir, "test", "tiers.toml"), "[tiers]\nquick = []\n")
    p = planned(; root = dir, manifest = nothing)
    @test p.result === nothing && any(m -> startswith(m, "the test catalog could not be read"), p.problems)
  end
end

@testset "Julia versions" begin
  bound = PF.julia_bound(ROOT)
  @test bound >= v"1.6" && bound.major == 1 && bound.minor >= 1
  minor = "1.$(bound.minor)"
  below = "1.$(bound.minor - 1)"
  for version in ("1", "lts", "pre", minor, minor * ".0", minor * ".7", "1.$(bound.minor + 1)", "1.$(bound.minor + 1).2", "1.13.3", "1.10.12")
    v = match(r"^1\.(\d+)", version)
    (v === nothing || parse(Int, v[1]) >= bound.minor) || continue
    @test planned(julia_version = version).result.julia_version == version
  end
  # an empty version is the default
  @test planned(julia_version = "").result.julia_version == "1"
  # below the bound of Project.toml, in both forms
  for version in (below, below * ".9", "1.0", "1.0.0")
    @test refused("is below the Julia version that Project.toml allows"; julia_version = version)
  end
  # not the syntax of setup-julia that this workflow accepts
  for bad in ("0.7", "2", "2.0", "1.", "1.x", "v1.10", "1.10.", "1.10.1.2", "nightly", "1.010", "1.10.01", "1;ls",
              "1 ", " 1", "1\n", "1\n::warning::x", "1.1000", "1.10.99999", "1.10-rc1", "lts ", "Pre", "\$(id)", "1.*")
    @test refused("must be 1, lts, pre, 1.N or 1.N.P"; julia_version = bad)
  end
  # the bound is read from Project.toml, not fixed
  with_fixture(julia = "1.12") do dir, manifest
    ok(v) = planned(; test_file = "alpha", julia_version = v, root = dir, manifest).result !== nothing
    @test PF.julia_bound(dir) == v"1.12.0"
    @test ok("1.12") && ok("1.12.0") && ok("1.13") && ok("1") && ok("lts") && ok("pre")
    @test !ok("1.11") && !ok("1.11.9") && !ok("1.10") && !ok("1.10.12")
  end
  mktempdir() do dir
    compat(text) = (write(joinpath(dir, "Project.toml"), "[compat]\njulia = \"$text\"\n"); PF.julia_bound(dir))
    @test compat("1.10") == v"1.10.0"
    @test compat("1.6, 1.10") == v"1.6.0"
    @test compat("~1.8.2") == v"1.8.2"
    @test compat("^1.9") == v"1.9.0"
    @test_throws ErrorException compat("≥1.10")
    write(joinpath(dir, "Project.toml"), "name = \"X\"\n")
    @test_throws ErrorException PF.julia_bound(dir)
  end
end

@testset "systems" begin
  labels(os) = [r.label for r in (PF.RUNNERS[l] for l in PF.MODE_LABELS[os])]
  @test labels("windows") == ["windows"] && labels("macos") == ["macos"] && labels("linux") == ["linux"]
  @test labels("both") == ["windows", "macos"] && labels("all") == ["windows", "macos", "linux"]
  @test planned(os = "windows").result.matrix ==
        "{\"include\":[{\"os\":\"windows-latest\",\"arch\":\"x64\",\"label\":\"windows\"}]}"
  @test planned(os = "both").result.matrix ==
        "{\"include\":[{\"os\":\"windows-latest\",\"arch\":\"x64\",\"label\":\"windows\"}," *
        "{\"os\":\"macos-latest\",\"arch\":\"aarch64\",\"label\":\"macos\"}]}"
  @test planned(os = "linux").result.matrix ==
        "{\"include\":[{\"os\":\"ubuntu-latest\",\"arch\":\"x64\",\"label\":\"linux\"}]}"
  @test planned(os = "all").result.matrix ==
        "{\"include\":[{\"os\":\"windows-latest\",\"arch\":\"x64\",\"label\":\"windows\"}," *
        "{\"os\":\"macos-latest\",\"arch\":\"aarch64\",\"label\":\"macos\"}," *
        "{\"os\":\"ubuntu-latest\",\"arch\":\"x64\",\"label\":\"linux\"}]}"
  @test planned(os = "").result.matrix == planned(os = "both").result.matrix
  for bad in ("ubuntu", "Windows", "windows,macos", "both ", "both\n", "all\n::warning::x", "mac", "x\$(id)")
    @test refused("`os` must be one of windows, macos, linux, both, all"; os = bad)
  end
  # the fields of the matrix are the ones that the workflow reads
  @test Set(m[1] for m in eachmatch(r"\"(\w+)\":", planned(os = "all").result.matrix)) == Set(["include", "os", "arch", "label"])
end

@testset "the expected commit" begin
  @test planned().result.sha == SHA
  for expected in ("", SHA[1:7], SHA[1:12], uppercase(SHA[1:7]), SHA, uppercase(SHA))
    @test planned(expected_sha = expected).result !== nothing
  end
  for bad in (SHA[1:6], SHA * "0", "012345g", "abcdef0", "0123456\n", " 0123456", "0123456 ", "0x01234", "0123457")
    @test refused("`expected_sha`"; expected_sha = bad)
  end
  @test occursin("must be 7 to 40 hexadecimal digits", only(problems(expected_sha = "012345")))
  @test occursin("is not the start of the dispatched commit", only(problems(expected_sha = "abcdef0")))
  # the checked-out commit is the dispatched one; an upper-case GITHUB_SHA is the same commit
  @test refused("is not the dispatched commit"; head = repeat("f", 40))
  @test refused("could not be read"; head = nothing)
  @test refused("GITHUB_SHA is not a commit hash"; sha = "abc", head = "abc")
  @test refused("GITHUB_SHA is not a commit hash"; sha = "", head = "")
  r = planned(sha = uppercase(SHA), head = SHA).result
  @test r !== nothing && r.sha == SHA
  @test planned(sha = SHA, head = uppercase(SHA)).result !== nothing
end

@testset "every problem is reported at once" begin
  p = planned(os = "ubuntu", test_file = "nope", julia_version = "1.2", expected_sha = "zz")
  @test p.result === nothing && length(p.problems) == 4
  for key in ("`os`", "`test_file`", "`julia_version`", "`expected_sha`")
    @test count(m -> occursin(key, m), p.problems) == 1
  end
  p = planned(os = "ubuntu", test_file = "nope", julia_version = "1.2", expected_sha = "zz", head = repeat("f", 40))
  @test length(p.problems) == 5
  @test planned().problems == String[] && planned().result !== nothing
end

@testset "output keys and the workflow" begin
  r = planned(os = "all", test_file = "conformance/tolerances").result
  @test keys(r) == PF.OUTPUT_KEYS == (:matrix, :test_file, :safe_name, :selector, :julia_version, :sha)
  text = lf(WORKFLOW)
  step_outputs = Set(m[1] for m in eachmatch(r"steps\.plan\.outputs\.(\w+)", text))
  needs_outputs = Set(m[1] for m in eachmatch(r"needs\.plan\.outputs\.(\w+)", text))
  wanted = Set(String.(PF.OUTPUT_KEYS))
  # the plan job exports every key the script writes, and the other job reads only those
  @test step_outputs == wanted
  @test needs_outputs == wanted

  # GITHUB_OUTPUT has exactly those keys, one line each
  mktempdir() do dir
    out, summary = joinpath(dir, "output"), joinpath(dir, "summary")
    io = IOBuffer()
    @test PF.emit_plan(planned(os = "all", test_file = "conformance/tolerances"); io, output = out, summary) == 0
    lines = readlines(out)
    @test length(lines) == length(wanted)
    keys_written = [first(split(l, '=')) for l in lines]
    @test Set(keys_written) == wanted && length(unique(keys_written)) == length(wanted)
    values = Dict(first(split(l, '=')) => join(split(l, '=')[2:end], '=') for l in lines)
    @test values["test_file"] == "conformance/tolerances" && values["safe_name"] == "conformance-tolerances"
    @test values["matrix"] == r.matrix && values["selector"] == r.selector && values["sha"] == SHA
    @test occursin("conformance/tolerances", read(summary, String)) && occursin("macos-latest", read(summary, String))
    @test occursin("matrix=", String(take!(io)))
  end

  # the matrix fields read by the workflow are the fields of the matrix
  fields = Set(m[1] for m in eachmatch(r"matrix\.(\w+)", text))
  @test fields == Set(["os", "arch", "label"])
  @test occursin("matrix: \${{ fromJSON(needs.plan.outputs.matrix) }}", text)
end

# The text of the workflow, checked without a YAML reader.
function top_level_keys(text, block)
  lines = split(text, '\n')
  start = findfirst(==(block * ":"), lines)
  start === nothing && return nothing
  found = String[]
  for l in lines[start + 1:end]
    isempty(l) || startswith(l, ' ') || startswith(l, '#') || break
    m = match(r"^  ([A-Za-z_-]+):", l)
    m === nothing || push!(found, m[1])
  end
  return found
end

# The text of every `run:` step, as the lines under it or the rest of its line.
function run_blocks(text)
  lines = split(text, '\n')
  blocks = String[]
  i = 1
  while i <= length(lines)
    m = match(r"^(\s*)(- )?run:\s*(.*)$", lines[i])
    if m === nothing
      i += 1
      continue
    end
    indent = length(m[1]) + (m[2] === nothing ? 0 : 2)
    if m[3] in ("|", "|-", "|+", ">", ">-", ">+")
      j = i + 1
      body = String[]
      while j <= length(lines) && (isempty(strip(lines[j])) || length(lines[j]) - length(lstrip(lines[j])) > indent)
        push!(body, lines[j])
        j += 1
      end
      push!(blocks, join(body, "\n"))
      i = j
    else
      push!(blocks, m[3])
      i += 1
    end
  end
  return blocks
end

# The lines that name an input, other than where an input may be read: the title and group of the
# run, and an environment variable of the planning step.
inputs_outside_env(text) = [l for l in split(text, '\n')
                            if occursin(r"\binputs\.", l) && !occursin(r"^\s*(run-name:|group:|FOCUS_[A-Z_]+:)", l) && !startswith(strip(l), '#')]

@testset "the workflow text" begin
  text = lf(WORKFLOW)
  ci = lf(CI_WORKFLOW)
  # dispatch is the only trigger, and the token only reads
  @test top_level_keys(text, "on") == ["workflow_dispatch"]
  @test top_level_keys(text, "permissions") == ["contents"]
  @test occursin(r"(?m)^permissions:\n  contents: read\n", text)
  @test !occursin(r"(?m)^\s*[a-z-]+: write\s*$", text) && !occursin("pull_request_target", text) && !occursin("secrets.", text)
  @test top_level_keys(text, "on") !== nothing && top_level_keys(ci, "on") == ["pull_request", "push", "schedule", "workflow_dispatch"]
  # the check itself sees a second trigger
  @test top_level_keys(replace(text, "on:\n  workflow_dispatch:" => "on:\n  push:\n    branches: [main]\n  workflow_dispatch:"), "on") ==
        ["push", "workflow_dispatch"]
  # inputs reach the shell only through env: no expression at all in a run text
  blocks = run_blocks(text)
  @test length(blocks) >= 6
  @test all(b -> !occursin("\${{", b) && !occursin("github.event.inputs", b), blocks)
  @test !occursin("github.event.inputs", text)
  @test isempty(inputs_outside_env(text))
  inputs = ["os", "test_file", "julia_version", "expected_sha"]
  for input in inputs
    @test occursin("FOCUS_$(uppercase(input)): \${{ inputs.$input }}", text)
  end
  # the checks see a run text that reads an input
  bad = replace(text, "run: julia --startup-file=no scripts/platform_focus.jl check-log" =>
                "run: echo \${{ inputs.test_file }}; julia --startup-file=no scripts/platform_focus.jl check-log")
  @test any(b -> occursin("\${{", b), run_blocks(bad))
  @test !isempty(inputs_outside_env(replace(text, "echo \"Testing \$TEST_FILE" => "echo \"Testing \${{ inputs.test_file }}")))
  @test !isempty(inputs_outside_env(replace(text, "group: platform" => "key: \${{ inputs.os }}\n  group: platform")))
  # credentials are not kept, the shell is bash on every job (the default on Windows is not), and
  # the actions are the versions that CI.yml uses
  @test count("actions/checkout@", text) == count("persist-credentials: false", text) >= 2
  @test count("runs-on:", text) == count(r"defaults:\n      run:\n        shell: bash\n", text) == 2
  uses(t) = Set(m[1] for m in eachmatch(r"uses:\s*(\S+)", t))
  @test !isempty(uses(text)) && issubset(uses(text), uses(ci))
  @test uses(text) == Set(["actions/checkout@v7", "julia-actions/setup-julia@v3", "actions/upload-artifact@v7"])
  # a later dispatch replaces an earlier one; the jobs are bounded
  @test occursin("cancel-in-progress: true", text)
  @test occursin("timeout-minutes: 10", text) && occursin("timeout-minutes: 45", text)
  @test occursin("fail-fast: false", text)
  # the test step runs one job with the selector, after the plan; the selection is checked even
  # when the tests failed; the artifact does not depend on success
  runs = filter(b -> occursin("scripts/test.jl", b), blocks)
  @test length(runs) == 1
  only_run = only(runs)
  @test occursin("--verbose --jobs=1 \"\${args[@]}\"", only_run) && occursin("read -ra args <<< \"\$SELECTOR\"", only_run)
  @test !occursin("--tier", only_run) && !occursin("--extra", only_run) && occursin("|| status=\$?", only_run) && occursin("exit \$status", only_run)
  @test occursin("if: \${{ !cancelled() && steps.tests.outcome != 'skipped' }}", text)
  @test occursin("scripts/platform_focus.jl check-log", text)
  @test occursin("if: \${{ !cancelled() }}", text)
  @test occursin("actions/upload-artifact@v7", text) && occursin("retention-days: 14", text)
  @test occursin("name: focus-\${{ needs.plan.outputs.safe_name }}-\${{ matrix.label }}-\${{ needs.plan.outputs.julia_version }}-\${{ github.run_attempt }}", text)
  @test occursin("arch: \${{ matrix.arch }}", text) && occursin("runs-on: \${{ matrix.os }}", text)
  # the planner runs as in CI.yml: bounded, retried once, exit 0 or 1 is its verdict
  @test occursin("timeout 120 julia --startup-file=no scripts/platform_focus.jl plan", text)
  @test occursin("{ [ \$status -eq 0 ] || [ \$status -eq 1 ]; } && exit \$status", text)
end

@testset "reading a workflow with CRLF endings" begin
  mktempdir() do dir
    text = lf(WORKFLOW)
    path = joinpath(dir, "w.yml")
    write(path, replace(text, "\n" => "\r\n"))
    @test occursin("\r\n", read(path, String))
    @test lf(path) == text
    @test run_blocks(lf(path)) == run_blocks(text)
    @test top_level_keys(lf(path), "on") == ["workflow_dispatch"]
    # unnormalized, the text would not match its own checks
    @test top_level_keys(read(path, String), "on") === nothing
    write(path, "ProximalBase tests: tier quick: 1 files; Julia 1\r\n  a\r\n")
    @test PF.check_log(lf(path), "a") == String[]
  end
end

# What test/runtests.jl prints before any worker starts.
header(n; label = "every file (tier full), filtered by three_factor_product", tail = " on x86_64-linux-gnu (AMD EPYC 7763 64-Core Processor, 4 threads), commit abc") =
  "ProximalBase tests: $label: $n files; Julia 1.11.2$tail"
sample_log(n, names; before = "", after = "  ... run output\n") = before * header(n) * "\n  " * names * "\n" * after

@testset "check-log reads the runner's selection" begin
  name = "three_factor_product"
  @test PF.check_log(sample_log(1, name), name) == String[]
  # a log with other lines before (a warning) and after (the run), colors, CRLF endings
  noisy = sample_log(1, name; before = "Precompiling ProximalBase...\n┌ Warning: something\n└ @ Main x.jl:1\n", after = "Test Summary: |\n")
  @test PF.check_log(noisy, name) == String[]
  @test PF.check_log(replace(sample_log(1, name), "\n" => "\r\n"), name) == String[]
  @test PF.check_log("\e[32m" * header(1) * "\e[0m\n  " * name * "\n", name) == String[]
  @test PF.check_log(header(1), name) != String[]   # no names line
  # the label is not the selection: the same label and a different selection
  @test PF.check_log(sample_log(1, "blockchecks_a"), name) != String[]
  # two files, a different file, extra files, none, no header, two headers
  p = PF.check_log(sample_log(2, name * " " * name * "_x"), name)
  @test length(p) == 1 && occursin("selected 2 files", only(p))
  p = PF.check_log(sample_log(1, "other"), name)
  @test length(p) == 1 && occursin("\"other\"", only(p))
  p = PF.check_log(sample_log(1, name * " other"), name)
  @test length(p) == 1 && occursin("not only", only(p))
  p = PF.check_log(sample_log(0, ""), name)
  @test length(p) == 1 && occursin("selected 0 files", only(p))
  p = PF.check_log("no header here\n  " * name * "\n", name)
  @test length(p) == 1 && occursin("has no line", only(p))
  p = PF.check_log(sample_log(1, name) * sample_log(1, name), name)
  @test length(p) == 1 && occursin("2 lines", only(p))
  @test length(PF.check_log("", name)) == 1
  # an exact name only: a prefix of the name or the name with more is another file
  @test PF.check_log(sample_log(1, "three_factor"), name) != String[]
  @test PF.check_log(sample_log(1, name * "_x"), name) != String[]
  # a byte that is not UTF-8 in the log does not stop the check
  bytes = UInt8[codeunits("noise ")..., 0xff, 0xfe, codeunits("\n")..., codeunits(sample_log(1, name))...]
  @test PF.check_log(PF.ascii_text(bytes), name) == String[]
  @test PF.ascii_text(UInt8[0x61, 0xc3, 0xa9]) == "a??"
  # the same header the runner prints, from its source: the pattern is the one test/runtests.jl prints
  runtests = read(joinpath(ROOT, "test", "runtests.jl"), String)
  @test occursin("println(\"ProximalBase tests: \$label: \$(length(names)) files; Julia \$VERSION on", runtests)
  @test occursin("println(\"  \", join(names, \" \"))", runtests)
end

@testset "a rejected input cannot start a workflow command" begin
  @test PF.error_line("50%\r\nnext") == "::error::50%25%0D%0Anext"
  @test PF.escape_command("a%0Ab") == "a%250Ab"
  hostile = "x\n::warning::injected\r\n::set-output name=a::b%0A"
  planned_hostile = planned(test_file = hostile, julia_version = hostile, os = hostile, expected_sha = hostile)
  @test length(planned_hostile.problems) == 4
  mktempdir() do dir
    out, summary = joinpath(dir, "output"), joinpath(dir, "summary")
    io = IOBuffer()
    @test PF.emit_plan(planned_hostile; io, output = out, summary) == 1
    text = String(take!(io))
    lines = split(text, '\n'; keepempty = false)
    @test length(lines) == 4 && all(startswith("::error::"), lines)
    @test !occursin('\r', text) && !occursin("\n::warning", text) && !occursin("\n::set-output", text)
    @test occursin("%250A", text)          # the percent sign of the input is encoded
    @test !isfile(out)                      # nothing reaches GITHUB_OUTPUT
    s = read(summary, String)
    @test occursin("inputs refused", s) && count(==("```"), split(s, '\n')) == 2
    @test !occursin(r"(?m)^::", s)
  end
  # a valid name is the only thing that reaches the outputs, and every value is one line
  mktempdir() do dir
    out = joinpath(dir, "output")
    @test PF.emit_plan(planned(os = "all"); io = IOBuffer(), output = out) == 0
    @test length(readlines(out)) == 6
  end
end

@testset "the tier rule and the catalog entry" begin
  manifest = TT.load(joinpath(ROOT, "test", "tiers.toml"), RULES)
  @test get(manifest.tier, "platform_focus", "") == "quick" && isfile(joinpath(ROOT, "test", "platform_focus.jl"))
  for paths in ([".github/workflows/platform-focused.yml"], ["scripts/platform_focus.jl"],
                [".github/workflows/platform-focused.yml", "scripts/platform_focus.jl"])
    tier, focus, _ = TT.classify(manifest, paths)
    @test tier == "quick" && focus == ["platform_focus"]
  end
  # the rule is specific to these two paths
  @test TT.classify(manifest, [".github/workflows/CI.yml"])[1] == "full"
end

end
