# Test tiers: which test files a tier runs, and which tier a pull request needs.
#
# test/runtests.jl uses this module to select the files of a tier, and scripts/ci_plan.jl uses
# it to choose the tier of a CI run. test/tier_selection.jl tests it. The tier lists are in
# test/tiers.toml and the CI rules in .github/tier-rules.toml.

module TestTiers

import TOML

export TIERS

# Each tier runs its own files and the files of every tier before it.
const TIERS = ("quick", "standard", "thorough", "full")

rank(tier::AbstractString) = something(findfirst(==(tier), TIERS), 0)

struct Rule
  patterns::Vector{String}
  regexes::Vector{Regex}
  tier::String
  focus::Vector{String}
  why::String
  narrow::String # the tier with the label test:iterate-narrow, or "" if the label does not apply
end

# The label that declares a change to one iterate implementation narrow (CONTRIBUTING.md).
const NARROW_LABEL = "test:iterate-narrow"

"""
    Manifest

The parsed tiers.toml: the lowest tier of each test file, the helper files that are not tests
and the directories with environments of their own; with tier-rules.toml also the rules that
map changed paths to tiers, the policy for package code without tests and the CI timeouts.
"""
struct Manifest
  tier::Dict{String, String}
  helpers::Vector{String}
  environments::Vector{String}
  rules::Vector{Rule}
  untested::String
  timeouts::Dict{String, Int}
end

struct ManifestError <: Exception
  problems::Vector{String}
end

function Base.showerror(io::IO, e::ManifestError)
  print(io, "test/tiers.toml or .github/tier-rules.toml has problems:")
  foreach(p -> print(io, "\n  - ", p), e.problems)
end

"""
    load(tiers_path, [rules_path])

Read the tier lists and, if given, the CI rules (two TOML files, read as one).
"""
function load(tiers_path::AbstractString, rules_path::Union{Nothing, AbstractString} = nothing)
  data = TOML.parsefile(tiers_path)
  if rules_path !== nothing
    rules = TOML.parsefile(rules_path)
    for k in keys(rules)
      haskey(data, k) && throw(ManifestError(["`$k` is in both $tiers_path and $rules_path"]))
    end
    merge!(data, rules)
  end
  return parse_manifest(data)
end
parse_manifest(text::AbstractString) = parse_manifest(TOML.parse(text))

function parse_manifest(data::AbstractDict)
  problems = String[]
  tier = Dict{String, String}()
  tables = get(data, "tiers", Dict{String, Any}())
  for name in keys(tables)
    name in TIERS || push!(problems, "unknown tier `$name` in [tiers]; the tiers are $(join(TIERS, ", "))")
  end
  for t in TIERS
    files = get(tables, t, String[])
    isempty(files) && push!(problems, "tier `$t` lists no test files")
    for file in files
      if haskey(tier, file)
        push!(problems, "`$file` is listed in both `$(tier[file])` and `$t`")
      else
        tier[file] = t
      end
    end
  end
  other = get(data, "not_tests", Dict{String, Any}())
  helpers = String[get(other, "helpers", String[])...]
  environments = String[get(other, "environments", String[])...]
  for file in helpers
    haskey(tier, file) && push!(problems, "`$file` is both a helper and in tier `$(tier[file])`")
  end
  rules = Rule[]
  for (i, rule) in enumerate(get(data, "risk", Any[]))
    t = get(rule, "tier", "")
    t == "none" || rank(t) > 0 || push!(problems, "risk rule $i has tier `$t`; use none or one of $(join(TIERS, ", "))")
    patterns = String[get(rule, "paths", String[])...]
    isempty(patterns) && push!(problems, "risk rule $i has no paths")
    focus = String[get(rule, "focus", String[])...]
    for name in focus
      haskey(tier, name) || push!(problems, "risk rule $i has focus `$name`, which is in no tier")
    end
    narrow = get(rule, "narrow", "")
    isempty(narrow) || 0 < rank(narrow) < rank(t) ||
      push!(problems, "risk rule $i has narrow tier `$narrow`; use a tier below its tier `$t`")
    push!(rules, Rule(patterns, map(glob_regex, patterns), t, focus, get(rule, "why", ""), narrow))
  end
  untested = get(get(data, "policy", Dict{String, Any}()), "source_without_tests", "standard")
  rank(untested) > 0 || push!(problems, "policy.source_without_tests must be a tier, not `$untested`")
  timeouts = Dict{String, Int}()
  for (t, minutes) in get(data, "timeouts", Dict{String, Any}())
    rank(t) > 0 && minutes isa Integer && minutes > 0 || push!(problems, "timeouts.$t = $minutes: use a tier and a positive number of minutes")
    timeouts[t] = minutes
  end
  isempty(problems) || throw(ManifestError(problems))
  return Manifest(tier, helpers, environments, rules, untested, timeouts)
end

isenvironment(m::Manifest, name) = any(dir -> startswith(name, dir * "/"), m.environments)

"""
    discover(testdir) -> Vector{String}

The names of the Julia files under `testdir` as ParallelTestRunner's `find_tests` gives them:
the path relative to `testdir` without `.jl`, with `/` separators, leaving out `runtests.jl`.
"""
function discover(testdir::AbstractString)
  names = String[]
  for (root, _, files) in walkdir(testdir), file in files
    endswith(file, ".jl") && file != "runtests.jl" || continue
    rel = relpath(joinpath(root, file[1:(end - 3)]), testdir)
    push!(names, replace(rel, '\\' => '/'))
  end
  return sort!(names)
end

"""
    check(manifest, discovered)

Check the manifest against the test file names found on disk (as `find_tests` names them):
every file has a tier, is a helper, or lies in an environment directory, and every name in
the manifest exists. Throws a `ManifestError` listing every problem.
"""
function check(m::Manifest, discovered)
  found = Set{String}(discovered)
  problems = String[]
  for name in sort!(collect(found))
    haskey(m.tier, name) || name in m.helpers || isenvironment(m, name) ||
      push!(problems, "`test/$name.jl` has no tier: add it to one tier in test/tiers.toml")
  end
  for name in sort!(collect(keys(m.tier)))
    name in found || push!(problems, "`$name` is in tier `$(m.tier[name])`, but test/$name.jl does not exist")
  end
  for name in m.helpers
    name in found || push!(problems, "helper `$name` does not exist")
  end
  isempty(problems) || throw(ManifestError(problems))
  return nothing
end

"""
    select(manifest, tier) -> Vector{String}

The test files that `tier` runs, sorted: every file whose lowest tier is `tier` or lower.
"""
function select(m::Manifest, tier::AbstractString)
  r = rank(tier)
  r > 0 || throw(ArgumentError("unknown test tier `$tier`; the tiers are $(join(TIERS, ", "))"))
  files = sort!([name for (name, t) in m.tier if rank(t) <= r])
  isempty(files) && throw(ArgumentError("test tier `$tier` selects no test files"))
  return files
end

# A glob over repository paths: `*` and `?` stay within a directory, `**` spans directories.
function glob_regex(pattern::AbstractString)
  out = IOBuffer()
  chars = collect(pattern)
  i = 1
  while i <= length(chars)
    c = chars[i]
    if c == '*' && i < length(chars) && chars[i + 1] == '*'
      if i + 1 < length(chars) && chars[i + 2] == '/'
        print(out, "(?:.*/)?")
        i += 3
      else
        print(out, ".*")
        i += 2
      end
      continue
    elseif c == '*'
      print(out, "[^/]*")
    elseif c == '?'
      print(out, "[^/]")
    elseif c in raw"\.^$+()[]{}|"
      print(out, '\\', c)
    else
      print(out, c)
    end
    i += 1
  end
  return Regex("^" * String(take!(out)) * "\$")
end

"""
    classify(manifest, paths; narrow = false, manifest_additions = nothing) -> (tier, focus, decisions)

The minimum tier for a change to `paths` (repository-relative, `/`-separated): the highest
tier of the first rule matching each path, or `full` for a path no rule matches. `"none"`
means no package tests. Changed test files and the `focus` files of matched rules are
returned in `focus`: CI runs them beside the tier. If package code changes and no test file
does, the tier is at least `policy.source_without_tests`. `decisions` gives each path's rule.

`narrow` is the label test:iterate-narrow. It lowers a rule with a `narrow` tier to that tier
only when that rule is the only one with a `narrow` tier among the matched rules (one iterate
implementation changed); such paths then do not count as package code without tests. It never
changes another rule, a path no rule matches, or the focus files.

`manifest_additions`, when given, are the names that test/tiers.toml adds, all of them test
files new in the change (see [`manifest_additions`](@ref)): the tier lists then decide quick,
and the new files run as changed test files.
"""
function classify(m::Manifest, paths; narrow::Bool = false, manifest_additions = nothing)
  decisions = Tuple{String, String, String}[]
  focus = Set{String}()
  tier = "none"
  raise(t) = (t != "none" && rank(t) > rank(tier)) && (tier = t)
  matched = [findfirst(rule -> any(r -> occursin(r, path), rule.regexes), m.rules) for path in paths]
  narrowable = unique(i for i in matched if i !== nothing && !isempty(m.rules[i].narrow))
  lower = narrow && length(narrowable) == 1
  if narrow && !lower
    push!(decisions, ("(label $NARROW_LABEL)", "-", isempty(narrowable) ?
          "not applied: no iterate implementation changed" :
          "not applied: more than one iterate implementation changed ($(join([m.rules[i].patterns[1] for i in narrowable], ", ")))"))
  end
  source = tests = false
  for (path, i) in zip(paths, matched)
    counts = true # package code that, without a test change, raises the tier to the policy minimum
    if path == "test/tiers.toml" && manifest_additions !== nothing
      push!(decisions, (path, "quick", isempty(manifest_additions) ? "no tier list changed (comments or order only)" :
                        "adds only test files new in this change ($(join(manifest_additions, ", "))), which run beside the tier"))
      raise("quick")
    elseif i === nothing
      push!(decisions, (path, "full", "no rule matches this path"))
      raise("full")
    else
      rule = m.rules[i]
      if lower && !isempty(rule.narrow)
        push!(decisions, (path, rule.narrow, "$(rule.why); narrow change ($NARROW_LABEL), so $(rule.narrow) instead of $(rule.tier)"))
        raise(rule.narrow)
        counts = false
      else
        push!(decisions, (path, rule.tier, rule.why))
        raise(rule.tier)
      end
      union!(focus, rule.focus)
    end
    if counts && (startswith(path, "src/") || startswith(path, "ext/"))
      source = true
    end
    m2 = match(r"^test/(.+)\.jl$", path)
    if m2 !== nothing && haskey(m.tier, m2.captures[1])
      tests = true
      push!(focus, m2.captures[1])
    end
  end
  if source && !tests && rank(tier) < rank(m.untested)
    push!(decisions, ("(policy)", m.untested, "package code changed and no test file did"))
    raise(m.untested)
  end
  # A focus file that has no tier no longer exists (deleted or renamed) or is a helper.
  filter!(name -> haskey(m.tier, name), focus)
  return tier, sort!(collect(focus)), decisions
end


"""
    pr_tier(manifest, paths, labels; manifest_additions = nothing) -> (tier, focus, decisions, notes)

The tier of a pull request that changes `paths` and carries `labels`: [`classify`](@ref), with
`narrow` set by the label test:iterate-narrow, then raised by test:standard, test:thorough or
test:full (labels never lower the tier). No changed paths give quick. `focus` leaves out the
files the tier runs anyway; `notes` describe the labels.
"""
function pr_tier(m::Manifest, paths, labels; manifest_additions = nothing)
  tier, focus, decisions = classify(m, paths; narrow = NARROW_LABEL in labels, manifest_additions)
  notes = String[]
  isempty(paths) && (tier = "quick"; push!(notes, "No changed files were found; the minimum is quick."))
  for label in labels
    label == NARROW_LABEL && continue # applied by classify, shown in the decisions
    mt = match(r"^test:(.+)$", label)
    mt === nothing && continue
    t = String(mt.captures[1])
    rank(t) > 0 || throw(ArgumentError("label `$label` names no tier; use test:$(join(TIERS, ", test:")) or $NARROW_LABEL"))
    push!(notes, "Label `$label` asks for at least $t.")
    (tier == "none" || rank(t) > rank(tier)) && (tier = t)
  end
  intier = tier == "none" ? Set(focus) : Set(select(m, tier))
  return tier, filter(name -> !(name in intier), focus), decisions, notes
end

"""
    manifest_additions(old_text, new_text, new_files) -> (additions, problems)

Compare two versions of test/tiers.toml, parsed. When the new one only adds test files that
are new in the change (`new_files`: names, as in the tier lists, of test files the change adds),
`additions` are their names (empty when nothing in the tier lists changed) and `problems` is
empty. Otherwise `additions` is `nothing` and `problems` says why: a file removed from the
lists or moved to another tier, an added name that is not a new file of the change, or a
change outside [tiers] (helpers, environments).
"""
function manifest_additions(old_text::AbstractString, new_text::AbstractString, new_files)
  old_data, new_data, old, new = try
    a, b = TOML.parse(old_text), TOML.parse(new_text)
    a, b, parse_manifest(a), parse_manifest(b)
  catch err
    return nothing, ["the tier lists could not be compared: " * first(sprint(showerror, err), 300)]
  end
  problems = String[]
  for name in sort!(collect(keys(old.tier)))
    t = old.tier[name]
    if !haskey(new.tier, name)
      push!(problems, "`$name` is removed from tier `$t`")
    elseif new.tier[name] != t
      push!(problems, "`$name` moves from `$t` to `$(new.tier[name])`")
    end
  end
  additions = sort!([name for name in keys(new.tier) if !haskey(old.tier, name)])
  for name in additions
    name in new_files || push!(problems, "`$name` is added to tier `$(new.tier[name])`, but test/$name.jl is not a new file of this change")
  end
  outside(d) = filter(kv -> kv.first != "tiers", d)
  outside(old_data) == outside(new_data) || push!(problems, "something outside [tiers] changed (helpers or environments)")
  return isempty(problems) ? (additions, problems) : (nothing, problems)
end

end
