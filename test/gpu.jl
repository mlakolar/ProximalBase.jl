module GPUTest

# The exhaustive GPU cases; the representative ones run in the Quick tier as gpu_smoke.jl.
# GPUArrays' reference backend runs on the CPU and forbids scalar indexing.
using JLArrays
const DeviceArray = JLArray
const test_types = (Float64, Float32)
host(x) = Array(x)
≃(a, b) = isapprox(a, b)
include("gpu_prelude.jl")
include("gpu_suite.jl")

end
