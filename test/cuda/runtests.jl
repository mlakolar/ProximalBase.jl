module PhysicalGPUTest

using CUDA, Test, Pkg, ProximalBase, Dates, InteractiveUtils, Random

CUDA.functional() || error("A physical CUDA device is required; this harness never skips hardware validation")
CUDA.allowscalar(false)
const DeviceArray = CuArray
const test_types = (Float64, Float32) # CUDA 5 supported devices provide both.
host(x) = (CUDA.synchronize(); Array(x))

# Synchronize even when the result is a scalar. Array comparisons also call
# host above, before copying or checking device results.
function ≃(a, b)
  CUDA.synchronize()
  single = eltype(a) == Float32 || eltype(b) == Float32
  isapprox(a, b; rtol = single ? 5e-5 : 1e-10, atol = single ? 5e-6 : 1e-12)
end

root = normpath(joinpath(@__DIR__, "..", ".."))
println("Physical GPU validation at ", now(UTC), " UTC")
println("Source commit: ", readchomp(`git -C $root rev-parse HEAD`))
println("Source status: ", readchomp(`git -C $root status --porcelain --untracked-files=no`))
println("Invocation: julia --project=test/cuda test/cuda/runtests.jl")
println("Tolerances Float32: rtol=5e-5, atol=5e-6; Float64: rtol=1e-10, atol=1e-12")
println("Scalar indexing: disabled; seed: 1")
versioninfo()
CUDA.versioninfo()
Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)

# Every shared GPU case: the representative ones (test/gpu_smoke.jl) and the rest (test/gpu.jl).
@testset "Physical CUDA GPU" begin
  include(joinpath(@__DIR__, "..", "gpu_prelude.jl"))
  include(joinpath(@__DIR__, "..", "gpu_suite_basics.jl"))
  # The rest draws the random inputs test/gpu.jl gives it, which includes it after the prelude.
  Random.seed!(1)
  include(joinpath(@__DIR__, "..", "gpu_suite.jl"))
end

# Initialization and compilation are excluded. CUDA.@time synchronizes and
# prints CPU allocations and GPU allocations separately, including pool-backed
# device allocations; these are diagnostics, not zero-allocation assertions.
function allocation_diagnostics()
  for T in test_types
    x = CuArray(T.(range(-2, 2; length = 300)))
    y = similar(x)
    f = NormL1(T(0.3))
    X = CuArray(reshape(T.(sin.(1:280)), 40, 7))
    ls = LeastSquaresLoss(CuArray(ones(T, 40)), X)
    glm = LogisticLoss(CuArray(zeros(T, 40)), X)
    β, g = CuArray(ones(T, 7)), CuArray(zeros(T, 7))
    operations = ("prox!" => (() -> prox!(y, f, x, T(0.8))),
                  "value_and_prox!" => (() -> value_and_prox!(y, f, x, T(0.8))),
                  "least_squares_gradient!" => (() -> gradient!(g, ls, β)),
                  "glm_value_and_gradient!" => (() -> value_and_gradient!(g, glm, β)))
    for (label, operation) in operations
      operation(); operation(); CUDA.synchronize()
      println("Warmed allocation diagnostic: ", label, ", ", T)
      CUDA.@time operation()
    end
  end
end
allocation_diagnostics()
CUDA.synchronize()
println("PHYSICAL_GPU_VALIDATION_PASSED")

end
