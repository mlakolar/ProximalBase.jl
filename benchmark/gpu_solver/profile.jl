# Run separately from run.jl; stdout is the retained diagnostic receipt.
using CUDA, LinearAlgebra, Statistics, ProximalBase
include("solver.jl")
const B = GPUSolverBenchmark
CUDA.functional() || error("A physical CUDA device is required")
CUDA.allowscalar(false)
BLAS.set_num_threads(4)
n = parse(Int, get(ENV, "PB_N", "400"))
p = parse(Int, get(ENV, "PB_P", "40"))
T = get(ENV, "PB_PRECISION", "Float64") == "Float64" ? Float64 : Float32
seed = parse(Int, get(ENV, "PB_SEED", "103"))
println("Diagnostic case: ", (; n, p, T, seed))
CUDA.versioninfo()
data = B.problem(n, p, seed, T)
w = B.workspace(data, CuArray)
solve(frequency) = (fill!(w.beta, 0); B.solve!(w, data.gamma, data.scale; check_every = frequency))
operations = (
    "gradient!" => (() -> gradient!(w.gradient, w.loss, w.beta)),
    "value_and_gradient!" => (() -> value_and_gradient!(w.gradient, w.loss, w.beta)),
    "prox!" => (() -> prox!(w.candidate, w.penalty, w.temporary, data.gamma)),
    "value_and_prox!" => (() -> value_and_prox!(w.candidate, w.penalty, w.temporary, data.gamma)),
    "forward GEMV" => (() -> mul!(w.loss.tmp, w.loss.X, w.beta)),
    "transpose GEMV" => (() -> mul!(w.gradient, transpose(w.loss.X), w.loss.tmp)),
)
# Avoid uninitialized temporary inputs in the prox diagnostics.
fill!(w.temporary, 0)
for (_, operation) in operations
    operation(); operation(); CUDA.synchronize()
end
for frequency in (1, 10)
    solve(frequency); CUDA.synchronize()
end
for (label, operation) in operations
    println("OPERATION: ", label)
    host_bytes = @allocated begin
        operation(); CUDA.synchronize()
    end
    println("Host-only @allocated bytes: ", host_bytes)
    println("CUDA allocation diagnostic (host/device separate):")
    CUDA.@time operation()
    CUDA.@profile trace=true operation()
end
println("LIVE/POOL before repeated solves, after GC:")
GC.gc(true); CUDA.reclaim(); CUDA.synchronize(); CUDA.memory_status()
for _ in 1:20
    solve(10)
end
CUDA.synchronize()
println("LIVE/POOL after 20 solves, before GC:"); CUDA.memory_status()
GC.gc(true); CUDA.synchronize()
println("LIVE/POOL after 20 solves, after GC:"); CUDA.memory_status()
for frequency in (1, 10)
    println("FULL SOLVE PROFILE: check_every=", frequency)
    CUDA.@profile trace=true solve(frequency)
end
# Identical data/order; natural GC and explicit-GC samples remain separate.
# Julia @timed and CUDA @timed are also separate from plain wall measurements.
println("ANOMALY_COLUMNS,frequency,repetition,order,instrument,seconds,host_gc_seconds")
for repetition in 1:10
    frequencies = isodd(repetition) ? (1, 10) : (10, 1)
    for (order, frequency) in enumerate(frequencies)
        CUDA.synchronize()
        start = time_ns(); solve(frequency); CUDA.synchronize()
        println("ANOMALY,", join((frequency, repetition, order, "wall_natural_gc", (time_ns()-start)/1e9, "NA"), ','))
        timed = @timed begin
            solve(frequency); CUDA.synchronize()
        end
        println("ANOMALY,", join((frequency, repetition, order, "julia_timed", timed.time, timed.gctime), ','))
        println("CUDA_TIMED,frequency=", frequency, ",repetition=", repetition)
        display(CUDA.@timed solve(frequency))
        GC.gc(true); CUDA.synchronize()
        start = time_ns(); solve(frequency); CUDA.synchronize()
        println("ANOMALY,", join((frequency, repetition, order, "wall_explicit_gc", (time_ns()-start)/1e9, "NA"), ','))
    end
end
println("Final independent check: ", B.verify(data, Array(w.beta)))
