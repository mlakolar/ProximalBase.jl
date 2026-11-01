# The shared prelude of the GPU cases, included before gpu_suite_basics.jl (gpu_smoke.jl),
# gpu_suite.jl (gpu.jl), or both (cuda/runtests.jl). The entry point supplies DeviceArray,
# test_types, host (synchronized copy), and ≃ (comparison).
using Test, ProximalBase, LinearAlgebra, Random
using GPUArraysCore: allowscalar
import Adapt

Random.seed!(1)

# a separable function of one's own, a bits type: Σᵢ c xᵢ² / 2
struct ScaledSquare{T} <: SeparableFunction
  c::T
end
ProximalBase.value_entry(f::ScaledSquare, v, i) = f.c * v^2 / 2
ProximalBase.prox_entry(f::ScaledSquare, v, i, γ) = v / (1 + γ * f.c)
ProximalBase.iszeropreserving(::ScaledSquare) = true

# the values agree, including Inf for a point outside a set
same(a, b) = isinf(b) ? a == b : a ≃ b
