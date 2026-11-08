# The guard against ambiguities between package methods and the GPU extension, included after
# the entry point defines DeviceArray and imports allowscalar, IndPattern, IndBanded and Test:
# at the end of gpu_suite_basics.jl (every GPU run: gpu_smoke in the full tier, and the physical
# CUDA run) and by gpu_extension_ambiguities.jl, which runs it in every tier.

@testset "no new ambiguities with the GPU extension" begin
  allowscalar(false)
  # These matrix-set intersections predate the host decomposition paths.
  D = typeof(DeviceArray(zeros(Float32, 1)))
  F = Nothing
  device_value = which(ProximalBase._value_dense, Tuple{F, D})
  device_prox = which(ProximalBase._prox_sep!, Tuple{D, F, D, Float64})
  known = (Set((device_value, which(ProximalBase._value_dense, Tuple{IndPattern, Vector{Float64}}))),
           Set((device_value, which(ProximalBase._value_dense, Tuple{IndBanded, Matrix{Float64}}))),
           Set((device_prox, which(ProximalBase._prox_sep!, Tuple{Matrix{Float64}, IndBanded, Matrix{Float64}, Float64}))))
  # Only methods the package defines: another test file on the same worker may have added
  # methods to ProximalBase functions (symmetric_parameters_trait's lla_weight does).
  ours(m) = Base.moduleroot(m.module) !== Main
  # Julia 1.10 prints the evaluated sides of ==, so a failure lists the pairs
  unexpected = filter(pair -> all(ours, pair) && !(Set(pair) in known), Test.detect_ambiguities(ProximalBase))
  @test unexpected == []
end
