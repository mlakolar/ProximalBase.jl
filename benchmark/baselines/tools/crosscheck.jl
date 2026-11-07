# Independent loops (orchestrator's own tool; not the harness): three per-operation times for the
# cross-check of the baseline (plan section 4). Plain `time_ns` loops, the minimum and the median of
# many repetitions, one thread. Usage: julia --project=<env> crosscheck.jl [out.tsv]
using ProximalBase, LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with JULIA_NUM_THREADS=1")
const OUT = length(ARGS) >= 1 ? ARGS[1] : nothing
results = Tuple{String,String,Float64,Float64}[]   # name, shape, min ns/op, median ns/op
const SINK = Ref(0.0)   # keeps the timed calls from being removed as dead code

function fixture(T, n, s; seed)
    rng = Xoshiro(seed)
    ids = randperm(rng, n)[1:s]
    vals = [T((rand(rng) < 0.5 ? -1 : 1) * (1 + rand(rng))) for _ in 1:s]
    x = SparseIterate(T, n)
    for (i, v) in zip(ids, vals); x[i] = v; end
    x, ids, vals
end

# 1. first activation: 256 distinct absent coordinates per call, restore outside the timed region
function cross_activation(T, n, s; reps=400, nb=1)
    x, ids, _ = fixture(T, n, s; seed=11)
    active = Set(ids)
    rng = Xoshiro(5)
    batch = Int[]
    while length(batch) < 256 * nb
        k = rand(rng, 1:n)
        (k in active || k in batch) && continue
        push!(batch, k)
    end
    act!(x, batch, v) = (@inbounds for k in batch; x[k] = v; end; nothing)
    act!(x, batch, one(T))                      # compile
    for k in batch; x[k] = zero(T); end; dropzeros!(x)
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns(); act!(x, batch, one(T)); t1 = time_ns()
        push!(times, (t1 - t0) / length(batch))
        for k in batch; x[k] = zero(T); end; dropzeros!(x)
    end
    push!(results, (nb == 1 ? "first_activation_per_coordinate" : "first_activation_per_coordinate_x$nb", "$T n=$n s=$s", minimum(times), median(times)))
end

# 2. an active sweep through activecoordinates, ns per active coordinate
function sweep_ids(x)
    acc = 0
    for k in activecoordinates(x); acc += k; end
    acc
end
function cross_sweep(T, n, s; reps=2000)
    x, ids, _ = fixture(T, n, s; seed=12)
    sweep_ids(x) == sum(ids) || error("sweep sum")
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns(); r = sweep_ids(x); t1 = time_ns()
        SINK[] += r
        push!(times, (t1 - t0) / s)
    end
    push!(results, ("sweep_activecoordinates_per_active", "$T n=$n s=$s", minimum(times), median(times)))
end

# 3. dot(v, x), ns per active coordinate
function cross_dot(T, n, s; reps=2000)
    x, ids, vals = fixture(T, n, s; seed=13)
    v = randn(Xoshiro(14), T, n)
    expected = sum(v[i] * val for (i, val) in zip(ids, vals))
    isapprox(dot(v, x), expected; rtol=1e-3) || error("dot value")
    times = Float64[]
    for _ in 1:reps
        t0 = time_ns(); r = dot(v, x); t1 = time_ns()
        SINK[] += r
        push!(times, (t1 - t0) / s)
    end
    push!(results, ("dot_v_x_per_active", "$T n=$n s=$s", minimum(times), median(times)))
end

for T in (Float64, Float32)
    cross_activation(T, 1_000_000, 1000)
    cross_activation(T, 10_000, 100)
    cross_activation(T, 1_000_000, 1000; nb=16)
    cross_sweep(T, 1_000_000, 10_000)
    cross_sweep(T, 1_000_000, 1000)
    cross_dot(T, 1_000_000, 10_000)
    cross_dot(T, 1_000_000, 1000)
end
println(rpad("operation", 40), rpad("fixture", 28), "min ns/op   median ns/op")
for (name, shape, mn, md) in results
    println(rpad(name, 40), rpad(shape, 28), lpad(round(mn; digits=3), 10), lpad(round(md; digits=3), 14))
end
if OUT !== nothing
    open(OUT, "w") do io
        println(io, "operation\tfixture\tmin_ns_per_op\tmedian_ns_per_op")
        for (name, shape, mn, md) in results; println(io, join((name, shape, mn, md), '\t')); end
    end
end
