using ProximalBase
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
# Exclude helpers and suites with their own environments; new tests are picked up automatically.
filter!(testsuite) do (name, _)
  name ∉ ("testutils", "gpu_suite") && !startswith(name, "cuda/") && !startswith(name, "oracles/")
end

init_code = quote
  include(joinpath(@__DIR__, "testutils.jl"))
end

serial = ["allocations", "large_inputs"]
serial_position = :after

runtests(ProximalBase, ARGS; testsuite, init_code, serial, serial_position)
