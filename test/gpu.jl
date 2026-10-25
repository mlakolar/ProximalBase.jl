module GPUTest

# GPUArrays' reference backend runs on the CPU and forbids scalar indexing.
using JLArrays
const DeviceArray = JLArray
const test_types = (Float64, Float32)
host(x) = Array(x)
≃(a, b) = isapprox(a, b)
include("gpu_suite.jl")

end
