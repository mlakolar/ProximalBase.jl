using Test, LinearAlgebra, ProximalBase
include("solver.jl")
using .GPUSolverBenchmark
const B = GPUSolverBenchmark
BLAS.set_num_threads(1)

@testset "Common-accuracy solver" begin
    for T in (Float32, Float64)
        data = B.problem(400, 40, 103, T)
        @test data.X == B.problem(400, 40, 103, T).X
        ref = B.reference(data)
        for frequency in (1, 10)
            w = B.workspace(data)
            result = B.solve!(w, data.gamma, data.scale; check_every = frequency)
            checked = B.verify(data, w.beta)
            @test result.status == "converged"
            @test checked.residual <= 1e-4
            @test B.assessment(result, checked, ref, 1e-4, T).status == "success"
            @test result.checks == result.iterations ÷ frequency + 1
        end
        w = B.workspace(data)
        capped = B.solve!(w, data.gamma, data.scale; maxiter = 1, check_every = 10)
        @test capped.status == "iteration_cap"
        @test capped.iterations == 1
        @test capped.checks == 2
        @test B.assessment(capped, B.verify(data, w.beta), ref, 1e-4, T).status == "iteration_cap"
        @test_throws ArgumentError B.solve!(w, data.gamma, data.scale; check_every = 0)
    end
    # Analytic soft-threshold solution for the identity design.
    data = (; X = Matrix{Float64}(I, 3, 3), y = [2.0, -1.0, 0.1],
              lambda = 0.2, gamma = 0.9, scale = 2.0)
    w = B.workspace(data)
    result = B.solve!(w, data.gamma, data.scale; tolerance = 1e-10)
    @test w.beta ≈ [1.8, -0.8, 0.0] atol=1e-9
    @test B.verify(data, w.beta).residual <= 1e-10
    @test result.status == "converged"
    bad = (; status = "converged")
    ref = B.reference(data)
    @test B.assessment(bad, (; residual = 1.0, objective = ref.objective), ref, 1e-4, Float64).status == "residual_failure"
    @test B.assessment(bad, (; residual = 0.0, objective = ref.objective + 1), ref, 1e-4, Float64).status == "objective_failure"
end
