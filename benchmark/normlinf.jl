# Benchmark of NormLinf (issue #946): prox! into a distinct output, prox!(x, f, x, γ), value_and_prox! and the value,
# at n ∈ {1, 16, 1000, 100000} in Float32 and Float64, with γλ equal to 1%, 90% and 200% of ‖x‖₁, and a NormL2 prox at
# n = 1000 as a control row of the machine. It uses the public API only (NormLinf, prox!, value_and_prox! and the
# value), so that it runs against this package and against a checkout of the frozen full tree.
#
#   julia --project=<package root> benchmark/normlinf.jl <out.csv> <label>
#
# One BLAS thread. The data come from Xoshiro generators with fixed seeds. A sample is a loop of calls whose repetition
# count is calibrated to target at least 5 ms per sample; 31 samples per cell after a warm-up. The in-place operation restores x from a copy in each call and subtracts the
# time of the copies. The CSV has the columns label, julia, op, T, n, regime, median_ns, min_ns, allocs_bytes.
using ProximalBase, Random, LinearAlgebra, Statistics

BLAS.set_num_threads(1)
length(ARGS) == 2 || error("usage: julia --project=<package root> benchmark/normlinf.jl <out.csv> <label>")
const OUT, LABEL = ARGS[1], ARGS[2]
const SAMPLES = 31
const MIN_SECONDS = 5e-3
const HAS_L2 = isdefined(ProximalBase, :NormL2)

function loop_prox!(y, f, x, γ, reps)
  for _ in 1:reps
    prox!(y, f, x, γ)
  end
  nothing
end
function loop_inplace!(x, x0, f, γ, reps)
  for _ in 1:reps
    copyto!(x, x0)
    prox!(x, f, x, γ)
  end
  nothing
end
function loop_copy!(x, x0, reps)
  for _ in 1:reps
    copyto!(x, x0)
  end
  nothing
end
function loop_fused!(y, f, x, γ, reps)
  s = zero(eltype(y))
  for _ in 1:reps
    s += value_and_prox!(y, f, x, γ)
  end
  s
end
function loop_value(f, x, reps)
  s = zero(eltype(x))
  for _ in 1:reps
    s += f(x)
  end
  s
end

# the number of calls per sample: the first count whose loop took at least MIN_SECONDS (later samples may be shorter)
function calibrate(run)
  run(1)
  reps = 1
  while true
    t = @elapsed run(reps)
    t >= MIN_SECONDS && return reps
    reps *= 2
  end
end

function sample(run, reps)
  [(@elapsed run(reps)) / reps * 1e9 for _ in 1:SAMPLES]
end

# the bytes allocated by one call of a loop function with reps = 1, after a warm-up call
function bytes_allocated(loop::F, args::Vararg{Any, N}) where {F, N}
  loop(args..., 1)
  @allocated loop(args..., 1)
end

# one cell: a loop function and its arguments (without the repetition count, which is last)
function record(io, op, T, n, regime, loop, args...)
  run(reps) = loop(args..., reps)
  reps = calibrate(run)
  ts = sample(run, reps)
  allocs = bytes_allocated(loop, args...)
  println(io, join((LABEL, VERSION, op, T, n, regime, median(ts), minimum(ts), allocs), ","))
  flush(io)
end

# the in-place cell: the time of a loop that copies x back and calls prox!, minus that of the copies alone
function record_inplace(io, T, n, regime, x, x0, f, γ)
  full(reps) = loop_inplace!(x, x0, f, γ, reps)
  copies(reps) = loop_copy!(x, x0, reps)
  reps = calibrate(full)
  ts = [((@elapsed full(reps)) - (@elapsed copies(reps))) / reps * 1e9 for _ in 1:SAMPLES]
  allocs = bytes_allocated(loop_inplace!, x, x0, f, γ)
  println(io, join((LABEL, VERSION, "prox_inplace", T, n, regime, median(ts), minimum(ts), allocs), ","))
  flush(io)
end

open(OUT, "w") do io
  println(io, "label,julia,op,T,n,regime,median_ns,min_ns,allocs_bytes")
  for T in (Float32, Float64), n in (1, 16, 1000, 100000)
    rng = Xoshiro(946 + n)
    x0 = randn(rng, T, n)
    x = copy(x0)
    y = similar(x0)
    s1 = sum(abs, x0)
    f = NormLinf(one(T))
    for (regime, frac) in (("1pct", 0.01), ("90pct", 0.9), ("200pct", 2.0))
      γ = T(frac * s1)
      record(io, "prox", T, n, regime, loop_prox!, y, f, x0, γ)
      record_inplace(io, T, n, regime, x, x0, f, γ)
      record(io, "value_and_prox", T, n, regime, loop_fused!, y, f, x0, γ)
    end
    record(io, "value", T, n, "na", loop_value, f, x0)
    if HAS_L2 && n == 1000
      record(io, "control_NormL2_prox", T, n, "na", loop_prox!, y, ProximalBase.NormL2(one(T)), x0, T(0.3))
    end
  end
end
println("wrote ", OUT)
