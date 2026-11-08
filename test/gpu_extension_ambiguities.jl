module GPUExtensionAmbiguitiesTest

# The guard against ambiguities between package methods and the GPU extension, in the Quick
# tier on GPUArrays' reference backend (the remaining device cases are in the Full tier,
# gpu_smoke.jl). The guard itself is gpu_ambiguities.jl, which gpu_suite_basics.jl also runs.

using Test, ProximalBase, JLArrays
using GPUArraysCore: allowscalar

const DeviceArray = JLArray

@testset "the GPU extension is loaded" begin
  # Without the extension the guard below would find nothing to be ambiguous with.
  @test Base.get_extension(ProximalBase, :ProximalBaseGPUArraysCoreExt) !== nothing
end

include(joinpath(@__DIR__, "gpu_ambiguities.jl"))

end
