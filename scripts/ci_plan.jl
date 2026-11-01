# Choose the test tier of a CI run from the event, the changed files and the PR labels.
#
#   julia scripts/ci_plan.jl [--base=REF]     # your working tree against REF (origin/master)
#   julia scripts/ci_plan.jl --event=pull_request --base=A --head=B --labels=test:full,bug
#   julia scripts/ci_plan.jl --event=workflow_dispatch --tier=standard
#
# The rules are in test/tiers.toml; test/tiers.jl applies them. Writes `tier`, `focus`,
# `run_tests` and `timeout` to $GITHUB_OUTPUT and a decision table to $GITHUB_STEP_SUMMARY
# when those are set, and prints both. No packages are needed.

include(joinpath(dirname(@__DIR__), "test", "tiers.jl"))
using .TestTiers
using Dates

const ROOT = dirname(@__DIR__)

function options(args)
  opts = Dict{String, String}()
  for arg in args
    m = match(r"^--([a-z-]+)=(.*)$", arg)
    m === nothing && error("unknown argument `$arg`; expected --name=value")
    opts[m.captures[1]] = m.captures[2]
  end
  known = ("event", "base", "head", "labels", "tier", "sha", "repo")
  for k in keys(opts)
    k in known || error("unknown option --$k; the options are $(join(known, ", "))")
  end
  return opts
end

# Without `head`, the working tree including untracked files, as a local check before a push.
function changed_files(base, head)
  # --no-renames lists both paths of a rename, so a removed path still counts.
  revs = head === nothing ? (base,) : (base, head)
  out = read(Cmd(`git diff --name-only --no-renames $revs`; dir = ROOT), String)
  head === nothing && (out *= "\n" * read(Cmd(`git ls-files --others --exclude-standard`; dir = ROOT), String))
  return sort!(unique(filter(!isempty, split(out, '\n'))))
end

maxtier(a, b) = a == "none" ? b : b == "none" ? a : TestTiers.rank(a) >= TestTiers.rank(b) ? a : b

# A successful run of this workflow on `sha` in the last day whose test jobs ran the full tier.
function recent_full_run(repo, sha)
  cutoff = Dates.format(now(UTC) - Hour(24), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
  query = ".[] | select(.event != \"pull_request\" and .createdAt >= \"$cutoff\") | .databaseId"
  ids = readlines(`gh run list --repo $repo --workflow CI.yml --commit $sha --status success --limit 50 --json databaseId,event,createdAt --jq $query`)
  for id in ids
    names = readlines(`gh run view $id --repo $repo --json jobs --jq '.jobs[].name'`)
    any(endswith(" - full"), names) && return id
  end
  return nothing
end

function plan(opts)
  manifest = TestTiers.load(joinpath(ROOT, "test", "tiers.toml"))
  # The runner checks this too, but here an unclassified test file fails within seconds.
  TestTiers.check(manifest, TestTiers.discover(joinpath(ROOT, "test")))
  event = get(opts, "event", "pull_request")
  lines = String[]
  focus = String[]
  if event == "pull_request"
    paths = changed_files(get(opts, "base", "origin/master"), get(opts, "head", nothing))
    tier, focus, decisions = TestTiers.classify(manifest, paths)
    isempty(paths) && (tier = "quick"; push!(lines, "No changed files were found; the minimum is quick."))
    push!(lines, "Changed files ($(length(paths))), the rule that decides each, and its tier:", "",
          "| Path | Tier | Why |", "|---|---|---|")
    for (path, t, why) in decisions
      push!(lines, "| `$path` | $t | $why |")
    end
    labels = filter(!isempty, split(get(opts, "labels", ""), ','))
    for label in labels
      m = match(r"^test:(.+)$", label)
      m === nothing && continue
      t = String(m.captures[1])
      TestTiers.rank(t) > 0 || error("label `$label` names no tier; use test:$(join(TIERS, ", test:"))")
      push!(lines, "", "Label `$label` asks for at least $t.")
      tier = maxtier(tier, t)
    end
  elseif event == "workflow_dispatch"
    tier = get(opts, "tier", "full")
    TestTiers.rank(tier) > 0 || error("unknown tier `$tier`")
    push!(lines, "Manual run of tier $tier.")
  elseif event == "schedule"
    tier = "full"
    push!(lines, "Scheduled run of the full tier.")
    if haskey(opts, "sha") && haskey(opts, "repo")
      id = recent_full_run(opts["repo"], opts["sha"])
      if id !== nothing
        tier = "none"
        push!(lines, "Run $id already passed the full tier on this commit in the last 24 hours; nothing to repeat.")
      end
    end
  elseif event == "push"
    base, head = get(opts, "base", ""), get(opts, "head", "HEAD")
    tier = "full"
    if !isempty(base) && !all(==('0'), base) && success(Cmd(`git cat-file -e $(base * "^{commit}")`; dir = ROOT))
      paths = changed_files(base, head)
      t, _, _ = TestTiers.classify(manifest, paths)
      if t == "none" && !isempty(paths)
        tier = "none"
        push!(lines, "Only files that need no package tests changed ($(length(paths))).")
      end
    end
    tier == "full" && push!(lines, "A push runs the full tier.")
  else
    error("unknown event `$event`")
  end
  # Changed files the tier already runs need no separate mention.
  if tier != "none"
    intier = Set(TestTiers.select(manifest, tier))
    filter!(name -> !(name in intier), focus)
  else
    empty!(focus)
  end
  run_tests = tier != "none"
  timeout = run_tests ? manifest_timeout(tier) : 0
  summary = run_tests ? "Tests: tier **$tier**" * (isempty(focus) ? "" : " and the changed test files $(join(focus, ", "))") :
            "Tests: **none needed**"
  pushfirst!(lines, summary, "")
  return (; tier, focus = join(focus, ","), run_tests, timeout), lines
end

function manifest_timeout(tier)
  data = TestTiers.TOML.parsefile(joinpath(ROOT, "test", "tiers.toml"))
  return Int(data["timeouts"][tier])
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

main(ARGS)
