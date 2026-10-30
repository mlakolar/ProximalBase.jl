using ProximalBase
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
# Exclude helpers and suites with their own environments; new tests are picked up automatically.
filter!(testsuite) do (name, _)
  name ∉ ("testutils", "gpu_suite") && !startswith(name, "cuda/") && !startswith(name, "oracles/")
end

# Heavy catalog-wide files that PR CI skips; pushes to master run them.
extended = ["precision_mixed", "large_inputs", "spectral_overflow"]
absent = filter(name -> !haskey(testsuite, name), extended)
isempty(absent) || error("Missing extended test files: $(join(absent, ", "))")

args = parse_args(ARGS; custom = ["core"])
core = args.custom["core"]
if core !== nothing && core != Some(nothing)
  error("--core takes no value")
end
if args.list === nothing && core == Some(nothing)
  foreach(name -> delete!(testsuite, name), extended)
end

init_code = quote
  include(joinpath(@__DIR__, "testutils.jl"))
end

serial = ["allocations", "large_inputs"]
serial_position = :after

runtests(ProximalBase, args; testsuite, init_code, serial, serial_position)
