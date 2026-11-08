module GPUSmokeTest

# Device compatibility in the Full tier (run beside a change to the GPU files, the GPU extension
# or the label test:gpu), on GPUArrays' reference backend with scalar indexing disallowed: the
# representative cases of gpu_suite_basics.jl (few function and element types per path), and the
# guard against ambiguities between package methods and the GPU extension, which the Quick tier
# also runs on its own (gpu_extension_ambiguities.jl). The exhaustive cases are in gpu.jl
# (gpu_suite.jl).

# GPUArrays' reference backend runs on the CPU and forbids scalar indexing.
using JLArrays
const DeviceArray = JLArray
const test_types = (Float64, Float32)
host(x) = Array(x)
≃(a, b) = isapprox(a, b)
include("gpu_prelude.jl")
include("gpu_suite_basics.jl")

end
