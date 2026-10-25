using ProximalBase, CUDA, Adapt, LinearAlgebra, Random, Statistics, TOML, Dates, InteractiveUtils, Pkg
include("solver.jl")
using .GPUSolverBenchmark
const B = GPUSolverBenchmark
CUDA.functional() || error("A physical CUDA device is required")
CUDA.allowscalar(false)
BLAS.set_num_threads(parse(Int, get(ENV, "PB_BLAS_THREADS", "4")))
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = abspath(get(ENV, "PB_OUTPUT", "gpu-solver-results"))
ispath(OUT) && error("Refusing to overwrite $OUT")
mkpath(OUT)
const MODE = get(ENV, "PB_MODE", "small")
MODE in ("small", "full") || error("PB_MODE must be small or full")
const REPS = parse(Int, get(ENV, "PB_REPETITIONS", "5"))
REPS >= 5 || error("At least five timed repetitions are required")
const CAP = parse(Int, get(ENV, "PB_MAXITER", "20000"))
const SOURCE = readchomp(`git -C $ROOT rev-parse HEAD`)
const DIRTY = readchomp(`git -C $ROOT status --porcelain --untracked-files=no`)
manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
isfile(manifest) || error("Instantiate the CUDA environment first")
cp(manifest, joinpath(OUT, "Manifest.toml"))
cp(Base.active_project(), joinpath(OUT, "Project.toml"))
open(joinpath(OUT, "environment.txt"), "w") do io
    println(io, "UTC: ", now(UTC), "\nsource: ", SOURCE, "\ntracked dirty: ", DIRTY)
    println(io, "Loaded package: ", pathof(ProximalBase))
    realpath(dirname(dirname(pathof(ProximalBase)))) == realpath(ROOT) || error("Wrong package checkout")
    println(io, "CPU: ", Sys.cpu_info()[1].model, "\nCPU allocation: ", get(ENV, "SLURM_CPUS_PER_TASK", "unset"))
    println(io, "BLAS threads: ", BLAS.get_num_threads(), "\nBLAS: ", BLAS.get_config())
    println(io, "Julia threads: ", Threads.nthreads(), "\nmode: ", MODE, "\nrepetitions: ", REPS, "\ncap: ", CAP)
    versioninfo(io)
    CUDA.versioninfo(io)
    Pkg.status(; io, mode = Pkg.PKGMODE_MANIFEST)
end

sync(backend) = backend == "gpu" ? CUDA.synchronize() : nothing
make_workspace(data, backend) = B.workspace(data, backend == "gpu" ? CuArray : identity)

function timed_solve(data, backend, scenario, frequency, tolerance, resident)
    if scenario == "resident"
        fill!(resident.beta, 0)
        sync(backend)
    end
    # No forced GC: these are natural-GC timing samples. Compilation is warmed below.
    sync(backend)
    start = time_ns()
    w = scenario == "resident" ? resident : make_workspace(data, backend)
    result = B.solve!(w, data.gamma, data.scale; tolerance, check_every = frequency, maxiter = CAP)
    host_beta = scenario == "transfers" ? Array(w.beta) : nothing
    sync(backend)
    seconds = (time_ns() - start) / 1e9
    # Resident download and all independent verification are outside the clock.
    host_beta === nothing && (host_beta = Array(w.beta))
    return (; seconds, result, host_beta)
end

const columns = (:source, :n, :p, :precision, :seed, :backend, :scenario, :tolerance,
                 :check_every, :repetition, :order, :seconds, :iterations, :checks,
                 :solver_residual, :residual, :objective, :reference_residual,
                 :reference_objective, :objective_gap, :objective_limit, :status,
                 :gamma, :lambda, :scale, :maxiter)
const records = open(joinpath(OUT, "timings.csv"), "w")
println(records, join(string.(columns), ','))
function record(row)
    println(records, join((string(getproperty(row, key)) for key in columns), ','))
    flush(records)
end

function run_case(n, p, seed, T; initial = false, extra = false)
    key = "$(n)x$(p)-$(T)-$(seed)" * (extra ? "-tight" : "")
    setup = @elapsed data = B.problem(n, p, seed, T)
    reference_seconds = @elapsed ref = B.reference(data)
    work = Dict{String, Any}()
    warm = Dict{String, Any}()
    tolerances = extra ? (1e-6,) : T == Float64 && !initial ? (1e-4, 1e-6) : (1e-4,)
    for backend in ("cpu", "gpu")
        sync(backend)
        construction = @elapsed begin
            work[backend] = make_workspace(data, backend)
            sync(backend)
        end
        # First full call records compilation plus execution separately.
        first_call = @elapsed begin
            B.solve!(work[backend], data.gamma, data.scale; maxiter = CAP)
            sync(backend)
        end
        checked = B.verify(data, Array(work[backend].beta))
        warm[backend] = (; construction, first_call, objective = checked.objective,
                         residual = checked.residual)
        checked.residual <= 1e-4 || error("Initial verification failed: $key $backend $checked")
    end
    # Objective and residual agreement, not coefficient uniqueness, gates expansion.
    isapprox(warm["cpu"].objective, warm["gpu"].objective; rtol = 5e-4, atol = 1e-6) ||
        error("CPU/GPU objective disagreement: $key")
    open(joinpath(OUT, "setup-$key.toml"), "w") do io
        TOML.print(io, Dict("data_and_spectral_setup_seconds" => setup,
                           "reference_seconds" => reference_seconds,
                           "cpu" => Dict(string(k) => v for (k, v) in pairs(warm["cpu"])),
                           "gpu" => Dict(string(k) => v for (k, v) in pairs(warm["gpu"]))))
    end
    successes = true
    for tolerance in tolerances, frequency in (1, 10), scenario in ("resident", "transfers")
        # Warm both timing branches at the selected stopping parameters.
        for backend in ("cpu", "gpu")
            timed_solve(data, backend, scenario, frequency, tolerance, work[backend])
        end
        for repetition in 1:REPS
            # Paired runs alternate their first backend; seeds also flip the order.
            backends = isodd(repetition + seed) ? ("cpu", "gpu") : ("gpu", "cpu")
            for (order, backend) in enumerate(backends)
                sample = timed_solve(data, backend, scenario, frequency, tolerance, work[backend])
                independent = B.verify(data, sample.host_beta)
                assessed = B.assessment(sample.result, independent, ref, tolerance, T)
                successes &= assessed.status == "success"
                record((; source = SOURCE, n, p, precision = string(T), seed, backend, scenario,
                        tolerance, check_every = frequency, repetition, order,
                        seconds = sample.seconds, iterations = sample.result.iterations,
                        checks = sample.result.checks, solver_residual = sample.result.residual,
                        residual = independent.residual, objective = independent.objective,
                        reference_residual = ref.residual, reference_objective = ref.objective,
                        objective_gap = assessed.gap, objective_limit = assessed.objective_limit,
                        status = assessed.status, gamma = data.gamma, lambda = data.lambda,
                        scale = data.scale, maxiter = CAP))
            end
        end
    end
    println("Completed $key; successful=$successes")
    flush(stdout)
    return successes
end

try
    # Both precisions of the initial case must pass before expanding the design.
    initial_passed = true
    for T in (Float32, Float64)
        initial_passed &= run_case(400, 40, 103, T; initial = true)
    end
    initial_passed || error("Initial case failed; raw rows retained; expansion refused")
    if MODE == "full"
        for (n, p) in ((400, 40), (4000, 400), (10000, 1000)), seed in (103, 104, 105), T in (Float32, Float64)
            # Repeat initial case at Float64's additional 1e-6 target as well.
            if n == 400 && seed == 103
                T == Float64 && run_case(n, p, seed, T; extra = true)
                continue
            end
            run_case(n, p, seed, T)
        end
    end
finally
    close(records)
end
println("BENCHMARK_COMPLETE; numerical failures remain in timings.csv")
