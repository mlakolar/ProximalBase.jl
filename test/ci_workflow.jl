module CIWorkflowTest

# The text of .github/workflows/CI.yml, checked without a YAML library (the job is cut out of the
# file by its two-space key): who may write to issues, when each reporting job runs, that no pull
# request or manual run can start one, that no event field reaches a script line, which check names
# each kind of run uses, the test job's inventory step and its Codecov upload, and the release
# record's artifact. The decisions of the scripts are tested in their own files; this file pins
# the wiring between them, which no other test sees. Every setting is checked on the text without
# its full-line comments (`active`), so a commented-out setting does not count; the predicates for
# the Codecov policy, the inventory step and the release-record artifact are functions of the text,
# and a test below shows that each fails when its line is commented out.

using Test

include(joinpath(dirname(@__DIR__), "scripts", "ci_triage.jl"))
using .CITriage

const ROOT = dirname(@__DIR__)
const WORKFLOW = replace(read(joinpath(ROOT, ".github", "workflows", "CI.yml"), String), "\r\n" => "\n")

"The jobs of the workflow in file order: `name => text` (the text from the key line to the next job)."
function split_jobs(text)
  start = findfirst("\njobs:\n", text)
  start === nothing && return Pair{String, String}[]
  body = text[last(start):end]
  keys = collect(eachmatch(r"^  ([a-z][a-z0-9-]*):[ ]*$"m, body))
  out = Pair{String, String}[]
  for (i, m) in enumerate(keys)
    stop = i < length(keys) ? keys[i + 1].offset - 1 : lastindex(body)
    push!(out, String(m.captures[1]) => body[m.offset:stop])
  end
  return out
end

"The steps of a job: each from its `      - ` line to the next."
function split_steps(job)
  starts = [m.offset for m in eachmatch(r"^      - "m, job)]
  return [job[s:(i < length(starts) ? starts[i + 1] - 1 : lastindex(job))] for (i, s) in enumerate(starts)]
end

"The text of each `run:` of `text`: the rest of the line, or the indented block of `run: |` and `run: >-`."
function run_blocks(text)
  lines = split(text, '\n')
  out = String[]
  i = 1
  while i <= length(lines)
    m = match(r"^(\s*)(?:- )?run: ?(.*)$", lines[i])
    if m === nothing
      i += 1
      continue
    end
    indent = length(m.captures[1])
    rest = String(m.captures[2])
    if !isempty(rest) && rest[1] in ('|', '>')
      block = String[]
      i += 1
      while i <= length(lines) && (isempty(strip(lines[i])) || length(lines[i]) - length(lstrip(lines[i])) > indent)
        push!(block, lines[i])
        i += 1
      end
      while !isempty(block) && isempty(strip(last(block)))
        pop!(block)
      end
      push!(out, join(block, '\n'))
    else
      push!(out, rest)
      i += 1
    end
  end
  return out
end

nocomment(text) = join([l for l in split(text, '\n') if !startswith(strip(l), "#")], '\n')

# The workflow without its full-line comments: the text that GitHub acts on, apart from `#` after a value.
active(text) = nocomment(replace(text, "\r\n" => "\n"))
const ACTIVE = active(WORKFLOW)
const JOBS = split_jobs(ACTIVE)
job(name) = only(t for (n, t) in JOBS if n == name)
hasjob(name) = any(n == name for (n, _) in JOBS)
field(text, key) = (m = match(Regex("^    $key: (.*)\$", "m"), text); m === nothing ? nothing : String(m.captures[1]))
const REPORTING = ("alert", "triage", "stale")

jobtext(text, name) = only(t for (n, t) in split_jobs(active(text)) if n == name)
stepname(s) = (m = match(r"^      - name: (.*)$"m, s); m === nothing ? nothing : String(m.captures[1]))

# The Codecov step of the test job fails the job in the nightly and in a release run: one step, one
# active `fail_ci_if_error` line with the expression, and no other mention.
function codecov_ok(text)
  steps = split_steps(jobtext(text, "test"))
  c = [s for s in steps if occursin("uses: codecov/codecov-action@", s)]
  return length(c) == 1 && count("fail_ci_if_error", active(text)) == 1 &&
         occursin(r"^          fail_ci_if_error: \$\{\{ github\.event_name == 'schedule' \|\| needs\.plan\.outputs\.release == 'true' \}\}$"m, only(c))
end

# The inventory step of the test job: its active condition and its command.
function inventory_ok(text)
  steps = split_steps(jobtext(text, "test"))
  found = [s for s in steps if stepname(s) == "Check the test inventory"]
  length(found) == 1 || return false
  s = only(found)
  return occursin(r"^        if: \$\{\{ !cancelled\(\) && needs\.plan\.outputs\.inventory == 'true' && steps\.tests\.outcome != 'skipped' \}\}$"m, s) &&
         run_blocks(s) == ["julia --startup-file=no scripts/ci_result.jl check-inventory \"\$RUNNER_TEMP/test-output.log\" \"--tier=\$TIER\" \${FOCUS:+\"--extra=\$FOCUS\"}"]
end

# The upload of the release record: its active settings.
function artifact_ok(text)
  steps = split_steps(jobtext(text, "release-record"))
  ups = [s for s in steps if occursin("uses: actions/upload-artifact@v7", s)]
  length(ups) == 1 || return false
  u = only(ups)
  return occursin(r"^          name: release-record$"m, u) && occursin(r"^          path: \$\{\{ runner\.temp \}\}/release-record$"m, u) &&
         occursin(r"^          retention-days: 90$"m, u) && occursin(r"^          if-no-files-found: error$"m, u)
end

# The pathspecs of the release record's comparison with the default branch, and whether each is quoted.
function record_pathspecs(text)
  step = only(s for s in split_steps(jobtext(text, "release-record")) if stepname(s) == "Compare the workflow files with the default branch")
  m = match(r"git diff --quiet FETCH_HEAD -- (.*?) \|\| status=", step)
  m === nothing && return Tuple{String, Bool}[]
  return [(String(strip(t, '\'')), startswith(t, "'") && endswith(t, "'")) for t in split(m.captures[1])]
end

# A git pathspec glob (`*` stays within a directory) against a path.
globmatch(pattern, path) = occursin(Regex("^" * replace(replace(pattern, r"([.+()\[\]{}^\$|\\])" => s"\\\1"), "*" => "[^/]*") * "\$"), path)

@testset "the helpers cut the workflow" begin
  @test [n for (n, _) in JOBS][1:2] == ["plan", "test"]
  @test all(startswith(t, "  $n:\n") for (n, t) in JOBS)
  @test length(JOBS) == length(unique(first.(JOBS)))
  sample = "  a:\n    steps:\n      - run: echo one\n      - name: x\n        run: |\n          echo two\n\n          echo three\n        env:\n          K: v\n      - run: >-\n          echo four\n          five\n"
  @test run_blocks(sample) == ["echo one", "          echo two\n\n          echo three", "          echo four\n          five"]
  @test length(split_steps(sample)) == 3 && startswith(split_steps(sample)[2], "      - name: x")
  # the extractor finds the run block of every job of the real file
  for name in ("plan", "test", "verdict", "gate", REPORTING..., "release-record")
    @test !isempty(run_blocks(job(name)))
  end
end

@testset "the reporting jobs exist and the old report job is gone" begin
  @test !hasjob("report")
  @test !occursin("Nightly CI failing", WORKFLOW) && !occursin("Report a failing master run", WORKFLOW)
  for name in (REPORTING..., "release-record")
    @test hasjob(name)
  end
end

@testset "only the three reporting jobs write to issues" begin
  writes = [m.offset for m in eachmatch(r"^\s*issues:\s*write"m, nocomment(WORKFLOW))]
  @test length(writes) == 3
  for (name, text) in JOBS
    n = count(r"^\s*issues:\s*write"m, nocomment(text))
    @test n == (name in REPORTING ? 1 : 0)
  end
  # no write permission anywhere else in the file but the cache cleanup of the jobs that use the cache,
  # and no permission at the level of the workflow
  @test match(r"^permissions:"m, WORKFLOW) === nothing
  @test !occursin(r"^\s*(contents|pull-requests|id-token|packages|checks|statuses|deployments):\s*write"m, nocomment(WORKFLOW))
  for name in REPORTING
    perms = match(r"^    permissions:\n((?:      .*\n)+)"m, job(name) * "\n")
    @test perms !== nothing
    @test sort(strip.(replace.(split(strip(perms.captures[1]), '\n'), r"#.*" => ""))) == ["actions: read", "contents: read", "issues: write"]
  end
  # the test job and the jobs that run test code have no issue access
  test_perms = match(r"^    permissions:[^\n]*\n((?:      .*\n)+)"m, job("test") * "\n")
  @test test_perms !== nothing && !occursin("issues", test_perms.captures[1])
  @test !occursin("issues", job("test")) && !occursin("issues: write", job("verdict")) && !occursin("issues: write", job("gate"))
  @test !occursin("issues: write", job("release-record")) && !occursin("issues: write", job("plan"))
end

@testset "when each reporting job runs" begin
  cond(name) = field(job(name), "if")
  a, t, s, r = cond("alert"), cond("triage"), cond("stale"), cond("release-record")
  @test a !== nothing && t !== nothing && s !== nothing && r !== nothing
  # a failed verdict of a push to master
  @test a == "always() && needs.verdict.result == 'failure' && github.event_name == 'push' && github.ref == 'refs/heads/master'"
  @test field(job("alert"), "needs") == "verdict"
  # the nightly run, unless it was cancelled; it must run when the verdict passed (to resolve) and when the plan or
  # the verdict failed. `!cancelled()` does that and also stops a running job when the whole run is cancelled, which
  # `always()` does not (the verdict's result alone would not change): so `always()` is not used here.
  @test t == "\${{ !cancelled() && github.event_name == 'schedule' && needs.verdict.result != 'cancelled' }}"
  @test !occursin("always()", t) && occursin("!cancelled()", t) && occursin("needs.verdict.result != 'cancelled'", t)
  @test field(job("triage"), "needs") == "[plan, verdict]"
  @test !occursin("needs.verdict.result == 'failure'", t) && !occursin("needs.plan.result", t)
  # every push to master, whatever the rest of the run did
  @test s == "github.event_name == 'push' && github.ref == 'refs/heads/master'"
  @test field(job("stale"), "needs") === nothing
  # the release record: the plan's release output, also after a failed job
  @test r == "always() && needs.plan.outputs.release == 'true'"
  @test field(job("release-record"), "needs") == "[plan, test, verdict, gate]"
  # none of the four runs for a pull request, and the reporters not for a manual run or a tag
  for name in (REPORTING..., "release-record")
    @test !occursin("pull_request", nocomment(cond(name)))
  end
  for (name, c) in (("alert", a), ("triage", t), ("stale", s))
    @test !occursin("workflow_dispatch", c) && !occursin("ref_type", c) && !occursin("tags", c)
  end
  @test !occursin("schedule", a) && !occursin("schedule", s) && !occursin("push", t)
end

@testset "the reporting jobs run default-branch code with the least access" begin
  for name in REPORTING
    text = job(name)
    @test field(text, "runs-on") == "ubuntu-latest" && field(text, "timeout-minutes") == "15"
    steps = split_steps(text)
    @test startswith(steps[1], "      - uses: actions/checkout@v7")
    # a full checkout of the commit of the event: no ref, no sparse checkout, no pull request head
    @test !occursin("ref:", steps[1]) && !occursin("sparse-checkout", steps[1]) && occursin("persist-credentials: false", steps[1])
    @test startswith(steps[2], "      - uses: julia-actions/setup-julia@v3") && occursin("version: '1'", steps[2])
    @test length(steps) == 3 && !occursin("cache", text)
    # no code of a pull request, no event field, no input anywhere in the job (comments aside)
    code = nocomment(text)
    @test !occursin("github.event.", code) && !occursin("inputs.", code) && !occursin("head_ref", code) && !occursin("pull_request", code)
    @test !occursin("github.head", code) && !occursin("workflow_run", code)
    # no expression of any kind in a script line
    for block in run_blocks(text)
      @test !occursin("\${{", block)
      @test !occursin("\${{ github.event", block) && !occursin("\${{ inputs.", block)
    end
    @test run_blocks(text) == ["julia --startup-file=no scripts/ci_triage.jl $name"]
  end
  env_keys(name) = sort([String(m.captures[1]) for m in eachmatch(r"^          ([A-Z_]+): "m, job(name))])
  @test env_keys("alert") == ["FAILED", "GH_TOKEN", "LINUX_FAILED"]
  @test env_keys("triage") == ["ADVISORY", "GH_TOKEN", "REQUIRED"]
  @test env_keys("stale") == ["GH_TOKEN"]
  @test occursin("          FAILED: \${{ needs.verdict.outputs.failed }}\n", job("alert"))
  @test occursin("          LINUX_FAILED: \${{ needs.verdict.outputs.linux_failed }}\n", job("alert"))
  @test occursin("          REQUIRED: \${{ needs.plan.outputs.required }}\n", job("triage"))
  @test occursin("          ADVISORY: \${{ needs.plan.outputs.advisory }}\n", job("triage"))
  for name in REPORTING
    @test occursin("          GH_TOKEN: \${{ github.token }}\n", job(name))
  end
  # the verdict's outputs that the alert reads exist
  for key in ("failed", "linux_failed")
    @test occursin("      $key: \${{ steps.verdict.outputs.$key }}", job("verdict"))
  end
  # the plan's outputs that the triage reads exist
  for key in ("required", "advisory")
    @test occursin("      $key: \${{ steps.plan.outputs.$key }}", job("plan"))
  end
end

@testset "separate concurrency groups: a running report job is not cancelled by a newer run (a pending one can be replaced)" begin
  groups = String[]
  for (name, group) in (("alert", "ci-alert"), ("triage", "ci-triage"), ("stale", "ci-stale"))
    text = job(name)
    @test occursin("    concurrency:\n      group: $group\n      cancel-in-progress: false\n", text)
    push!(groups, group)
  end
  @test length(unique(groups)) == 3
  # the workflow's own group does not use these names, and only a pull request run cancels
  @test !any(occursin(g, ACTIVE[first(findfirst("\nconcurrency:\n", ACTIVE)):first(findfirst("\njobs:\n", ACTIVE))]) for g in groups)
  @test occursin("\n  cancel-in-progress: \${{ github.event_name == 'pull_request' }}\n", ACTIVE)
end

@testset "check names: the PR names belong to pull requests only" begin
  expressions = [strip(l) for l in split(WORKFLOW, '\n') if !startswith(strip(l), "#") && occursin(r"PR required jobs|PR merge gate|CI required jobs|CI result", l)]
  guarded = r"github\.event_name == 'pull_request' && '(PR required jobs|PR merge gate)' \|\| '(CI required jobs|CI result)'"
  @test length(expressions) == 4
  for l in expressions
    @test occursin(guarded, l)
    # a PR name appears once per line, in the first branch of the guard
    @test count("PR required jobs", l) + count("PR merge gate", l) == 1 && count("CI required jobs", l) + count("CI result", l) == 1
  end
  shown(event, l) = (m = match(guarded, l); event == "pull_request" ? m.captures[1] : m.captures[2])
  verdict_line = field(job("verdict"), "name")
  gate_line = field(job("gate"), "name")
  @test verdict_line !== nothing && gate_line !== nothing
  for event in ("push", "schedule", "workflow_dispatch")
    @test shown(event, verdict_line) == "CI required jobs" && shown(event, gate_line) == "CI result"
  end
  @test shown("pull_request", verdict_line) == "PR required jobs" && shown("pull_request", gate_line) == "PR merge gate"
  # the nightly's checks equal neither required name, and no other job is called by a required name
  required = ("PR required jobs", "PR merge gate")
  nightly = [shown("schedule", verdict_line), shown("schedule", gate_line)]
  @test isempty(intersect(nightly, required))
  other_names = String[]
  for (n, text) in JOBS
    n in ("verdict", "gate") && continue
    m = match(r"^    name: (.*)$"m, text)
    m !== nothing && push!(other_names, String(m.captures[1]))
  end
  @test isempty(intersect(other_names, required)) && isempty(intersect(other_names, nightly))
  for name in (REPORTING..., "release-record")
    @test field(job(name), "name") !== nothing && !(field(job(name), "name") in required)
  end
  # the gate reads the verdict under the same pair of names
  @test occursin("          GATE: \${{ github.event_name == 'pull_request' && 'PR merge gate' || 'CI result' }}\n", job("gate"))
  @test occursin("          VERDICT: \${{ github.event_name == 'pull_request' && 'PR required jobs' || 'CI required jobs' }}\n", job("gate"))
end

@testset "the test job checks the inventory after the tests" begin
  steps = split_steps(job("test"))
  stepnames = stepname.(steps)
  i = findfirst(==(CITriage.TEST_STEP), stepnames)
  @test i !== nothing && CITriage.TEST_STEP == "Run package tests"
  @test occursin("        id: tests\n", steps[i])
  @test count(==(CITriage.TEST_STEP), stepnames) == 1
  j = findfirst(==("Check the test inventory"), stepnames)
  @test j !== nothing && i < j
  @test stepnames[i + 1] == "Timing summary" && j == i + 2
  s = steps[j]
  @test inventory_ok(WORKFLOW)
  @test occursin("        shell: bash\n", s)
  @test occursin("          TIER: \${{ needs.plan.outputs.tier }}\n", s) && occursin("          FOCUS: \${{ needs.plan.outputs.focus }}\n", s)
  @test !occursin("continue-on-error", s)
  # the log the step reads is the one the test step writes
  @test occursin("tee \"\$RUNNER_TEMP/test-output.log\"", steps[i])
  # no step in front of the tests carries the id `tests`, and the plan's output exists
  @test count("id: tests", job("test")) == 1
  @test occursin("      inventory: \${{ steps.plan.outputs.inventory }}", job("plan"))
end

@testset "Codecov fails the nightly and a release run" begin
  @test codecov_ok(WORKFLOW)
  c = only(s for s in split_steps(job("test")) if occursin("uses: codecov/codecov-action@", s))
  @test !occursin("fail_ci_if_error: false", ACTIVE)
  # the step runs only when coverage is collected
  @test occursin("if: matrix.coverage && needs.plan.outputs.coverage == 'true'", c)
end

@testset "a setting that is only commented out does not count" begin
  # the same line with `# ` in front of it (at the start of the line, or after its indentation)
  commented(text, line) = (@assert count("\n" * line * "\n", text) == 1; replace(text, "\n" * line * "\n" => "\n# " * line * "\n"))
  indented(text, line) = (@assert count("\n" * line * "\n", text) == 1; replace(text, "\n" * line * "\n" => "\n" * match(r"^ *", line).match * "# " * lstrip(line) * "\n"))
  codecov_line = "          fail_ci_if_error: \${{ github.event_name == 'schedule' || needs.plan.outputs.release == 'true' }}"
  inventory_line = "        if: \${{ !cancelled() && needs.plan.outputs.inventory == 'true' && steps.tests.outcome != 'skipped' }}"
  retention_line = "          retention-days: 90"
  for (check, line) in ((codecov_ok, codecov_line), (inventory_ok, inventory_line), (artifact_ok, retention_line))
    @test check(WORKFLOW)
    @test !check(commented(WORKFLOW, line))
    @test !check(indented(WORKFLOW, line))
    # a CRLF copy of the file is the same file
    @test check(replace(WORKFLOW, "\n" => "\r\n"))
  end
  # other mutations of the same lines fail too
  @test !codecov_ok(replace(WORKFLOW, codecov_line => "          fail_ci_if_error: false"))
  @test !inventory_ok(replace(WORKFLOW, "needs.plan.outputs.inventory == 'true' && steps.tests" => "needs.plan.outputs.inventory != 'false' && steps.tests"))
  @test !artifact_ok(replace(WORKFLOW, retention_line => "          retention-days: 30"))
  # the extractors do not see a comment: a mention of the setting in a comment of the file does not change a count
  @test count("fail_ci_if_error", ACTIVE) == 1 && count("fail_ci_if_error", active(commented(WORKFLOW, codecov_line))) == 0
end

@testset "the release record" begin
  text = job("release-record")
  @test field(text, "runs-on") == "ubuntu-latest" && field(text, "timeout-minutes") == "10"
  perms = match(r"^    permissions:\n((?:      .*\n)+)"m, text * "\n")
  @test sort(strip.(replace.(split(strip(perms.captures[1]), '\n'), r"#.*" => ""))) == ["actions: read", "contents: read"]
  steps = split_steps(text)
  @test startswith(steps[1], "      - uses: actions/checkout@v7") && occursin("fetch-depth: 1", steps[1])
  fetch = [s for s in steps if occursin("git fetch --no-tags --depth=1 origin master", s)]
  @test length(fetch) == 1
  specs = record_pathspecs(WORKFLOW)
  @test first.(specs) == [".github/workflows/CI.yml", ".github/tier-rules.toml", "scripts/ci_*.jl", "scripts/release_record.jl",
                          "scripts/test.jl", "scripts/conformance_map.jl", "scripts/conformance_run.jl", "test/runtests.jl",
                          "test/tiers.jl", "test/tiers.toml"]
  # a glob is quoted, so that git expands it (the shell would match the checkout only, and drop a file that is new)
  @test all(quoted for (spec, quoted) in specs if occursin('*', spec))
  covered(path) = any(globmatch(spec, path) for (spec, _) in specs)
  # every check script of the repository is in the comparison, whatever its name, and so are the gate, the runner
  # and the test selection
  checks = ["scripts/$f" for f in readdir(joinpath(ROOT, "scripts")) if startswith(f, "ci_") && endswith(f, ".jl")]
  @test "scripts/ci_gate.jl" in checks && "scripts/ci_plan.jl" in checks && "scripts/ci_verdict.jl" in checks &&
        "scripts/ci_result.jl" in checks && "scripts/ci_triage.jl" in checks
  @test all(covered, checks)
  for path in ("scripts/ci_gate.jl", "scripts/test.jl", "test/tiers.jl", "test/tiers.toml", "test/runtests.jl", "scripts/release_record.jl",
               ".github/workflows/CI.yml", ".github/tier-rules.toml")
    @test isfile(joinpath(ROOT, path)) && covered(path)
  end
  @test !covered("README.md") && !covered("test/ci_workflow.jl")
  # the check itself works on the glob helper
  @test globmatch("scripts/ci_*.jl", "scripts/ci_gate.jl") && !globmatch("scripts/ci_*.jl", "scripts/ci/x.jl") && !globmatch("scripts/ci_*.jl", "scripts/ciXgate.jl")
  @test occursin("differs=unknown", only(fetch)) && occursin("differs=false", only(fetch)) && occursin("differs=true", only(fetch))
  record = [s for s in steps if occursin("scripts/release_record.jl \"\$RUNNER_TEMP/release-record/release-record.md\"", s)]
  @test length(record) == 1
  r = only(record)
  for (key, value) in ("PLAN" => "needs.plan.result", "TESTS" => "needs.test.result", "VERDICT" => "needs.verdict.result",
                       "GATE" => "needs.gate.result", "EXPECTED_SHA" => "inputs.expected_sha",
                       "WORKFLOW_DIFFERS" => "steps.differs.outputs.differs", "WORKFLOW_SHA" => "github.workflow_sha",
                       "GH_TOKEN" => "github.token")
    @test occursin("          $key: \${{ $value }}\n", r)
  end
  @test all(!occursin("\${{", b) for b in run_blocks(text))
  # the artifact of the record: the directory the script writes to, 90 days, an error without the file
  uploads = [s for s in steps if occursin("uses: actions/upload-artifact@v7", s)]
  @test length(uploads) == 1
  u = only(uploads)
  @test occursin("          name: release-record\n", u)
  @test occursin("          path: \${{ runner.temp }}/release-record\n", u)
  @test artifact_ok(WORKFLOW)
  @test occursin("          retention-days: 90\n", u) && occursin("          if-no-files-found: error\n", u)
  @test !occursin("continue-on-error", text) && !occursin("if: ", u)
  # the record is the last step and no step is allowed to fail silently
  @test occursin("uses: actions/upload-artifact@v7", steps[end])
  @test !occursin("continue-on-error", text)
  # the only other uploads are the test job's reports and the benchmark failures, not named release-record
  @test count("name: release-record", ACTIVE) == 1
end

@testset "pull requests are untouched by the new jobs" begin
  # the test job's inventory step needs the plan's flag, which a pull request run never sets
  @test occursin("needs.plan.outputs.inventory == 'true'", ACTIVE)
  # the verdict and the gate keep their jobs and needs
  @test field(job("verdict"), "needs") == "[plan, test, benchmark-check, analysis]" && field(job("verdict"), "if") == "always()"
  @test field(job("gate"), "needs") === nothing
  # none of the new jobs is a need of the verdict or the gate: they cannot change a check of a pull request
  for name in (REPORTING..., "release-record")
    @test !occursin(name, field(job("verdict"), "needs")) && field(job("gate"), "needs") === nothing
  end
end

end
