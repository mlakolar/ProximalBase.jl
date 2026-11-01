using ProximalBase
using ParallelTestRunner

# Test files are found automatically, and test/tiers.toml gives each one a tier (see tiers.jl).
include(joinpath(@__DIR__, "tiers.jl"))
using .TestTiers

testsuite = find_tests(@__DIR__)
manifest = TestTiers.load(joinpath(@__DIR__, "tiers.toml"))
# An unclassified or missing file stops every run, whatever the arguments.
TestTiers.check(manifest, keys(testsuite))
# Drop the helpers and the environments of their own (cuda/, oracles/).
filter!(((name, _),) -> haskey(manifest.tier, name), testsuite)

args = parse_args(ARGS; custom = ["tier", "extra", "core"])

function option_value(flag, example)
  value = args.custom[flag]
  value === nothing && return nothing
  v = something(value)
  (v === nothing || isempty(v)) && error("--$flag needs a value, for example --$flag=$example")
  return String(v)
end

tier = option_value("tier", "quick")
tier === nothing || TestTiers.rank(tier) > 0 ||
  error("unknown test tier `$tier`; the tiers are $(join(TIERS, ", "))")
extra = filter(!isempty, String.(strip.(split(something(option_value("extra", "gpu,metric"), ""), ','))))
for name in extra
  haskey(testsuite, name) || error("--extra: no test file `$name`; `julia scripts/test.jl --list` shows the names")
end
core = args.custom["core"]
core === nothing || core == Some(nothing) || error("--core takes no value")
core = core !== nothing

# Deprecated: before tiers, --core ran every file except these (the extended tier).
legacy_extended = ["precision_mixed", "large_inputs", "spectral_overflow"]
if core
  tier === nothing || error("--core cannot be combined with --tier; use --tier alone")
  absent = filter(name -> !haskey(testsuite, name), legacy_extended)
  isempty(absent) || error("Missing extended test files: $(join(absent, ", "))")
  @warn "`--core` is deprecated; use `--tier=quick`, `standard`, `thorough` or `full`. It still runs every file except $(join(legacy_extended, ", ")) and conformance/*."
end

if tier !== nothing
  # The tier, the extra files, then the name filters among them.
  selected = Set(TestTiers.select(manifest, tier))
  union!(selected, extra)
  filter!(((name, _),) -> name in selected, testsuite)
elseif core && args.list === nothing
  # --list shows the whole catalog with --core, as before tiers.
  filter!(((name, _),) -> !(name in legacy_extended || startswith(name, "conformance/")), testsuite)
end
filter_tests!(testsuite, args)
isempty(testsuite) && error("no test file matches the selection: $(join(ARGS, " "))")

if args.list === nothing
  label = tier !== nothing ? "tier $tier" : core ? "--core (deprecated)" : "every file (tier full)"
  isempty(extra) || (label *= " with " * join(extra, ", "))
  isempty(args.positionals) || (label *= ", filtered by " * join(args.positionals, " "))
  names = sort!(collect(keys(testsuite)))
  commit = haskey(ENV, "GITHUB_SHA") ? ", commit $(ENV["GITHUB_SHA"])" : ""
  cpu = "$(strip(first(Sys.cpu_info()).model)), $(Sys.CPU_THREADS) threads"
  println("ProximalBase tests: $label: $(length(names)) files; Julia $VERSION on $(Sys.MACHINE) ($cpu)$commit")
  println("  ", join(names, " "))
end

init_code = quote
  include(joinpath(@__DIR__, "testutils.jl"))
  include(joinpath(@__DIR__, "conformance", "common.jl"))
end

# These run alone after the parallel batch; large_inputs has timing checks.
serial = ["allocations", "large_inputs"]
serial_position = :after

runtests(ProximalBase, args; testsuite, init_code, serial, serial_position)
