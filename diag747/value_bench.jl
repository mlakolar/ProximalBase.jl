# Issue #747: whole NormL1Fused value calls (and value_and_prox!), Float32 and Float64.
# One line per case: BENCH <T> <case> n=<n> <call> alloc=<bytes after warm-up> ns_min=<min of 5> ns_max=<max of 5>
# The result of every call is accumulated into a printed sink, so no call can be dropped. The calls run in function barriers.
using ProximalBase, Random
const SINK = Ref(0.0)

function runvalue(f, x, N)
  s = 0.0
  for _ in 1:N
    s += Float64(f(x))
  end
  s
end
function runvp(f, y, x, γ, N)
  s = 0.0
  for _ in 1:N
    s += Float64(value_and_prox!(y, f, x, γ))
  end
  s
end
allocvalue(f, x) = @allocated f(x)
allocvp(f, y, x, γ) = @allocated value_and_prox!(y, f, x, γ)

# ns per call for each of 5 repetitions of N calls
function reps(g, N)
  g(2)                       # warm
  ts = Float64[]
  for _ in 1:5
    t0 = time_ns()
    SINK[] += g(N)
    push!(ts, (time_ns() - t0) / N)
  end
  ts
end
ncalls(n) = n <= 5 ? 200_000 : n <= 100 ? 20_000 : 300

function report(T, case, n, call, alloc, ts)
  println("BENCH $T $case n=$n $call alloc=$alloc ns_min=$(minimum(ts)) ns_max=$(maximum(ts))")
end

function vector_cases(T)
  M = floatmax(T)
  rng = MersenneTwister(747)
  for n in (5, 100, 10_000)
    xr = randn(rng, T, n)
    xe = fill(T(1.5), n)
    xo = M / 4 .* (T(0.5) .+ rand(rng, T, n)) .* T.(rand(rng, (-1, 1), n))
    xq = fill(M / 4, n)
    cases = (("ordinary-finite-λ2", NormL1Fused(T(0.5), T(0.25)), xr),
             ("inf-unequal", NormL1Fused(T(0.5), T(Inf)), xr),
             ("inf-equal", NormL1Fused(T(0.5), T(Inf)), xe),
             ("recovery-finite-λ2", NormL1Fused(T(1), T(0.5)), xo),
             ("recovery-inf-equal", NormL1Fused(T(1), T(Inf)), xq))
    for (name, f, x) in cases
      N = ncalls(n)
      runvalue(f, x, 2)
      report(T, name, n, "value", allocvalue(f, x), reps(k -> runvalue(f, x, k), N))
      y = similar(x)
      runvp(f, y, x, one(T), 2)
      report(T, name, n, "value_and_prox!", allocvp(f, y, x, one(T)), reps(k -> runvp(f, y, x, one(T), k), N))
    end
  end
end

function pair_cases(T)
  M = floatmax(T)
  cases = (("pair-ordinary-finite-λ2", NormL1Fused(T(0.5), T(0.25)), (T(1), T(2))),
           ("pair-inf-equal", NormL1Fused(T(0.5), T(Inf)), (T(1.5), T(1.5))),
           ("pair-recovery-finite-λ2", NormL1Fused(T(1), T(0.5)), (T(0.45) * M, -T(0.4) * M)),
           ("pair-recovery-inf-equal", NormL1Fused(T(1), T(Inf)), (M / 2, M / 2)))
  for (name, f, x) in cases
    runvalue(f, x, 2)
    report(T, name, 2, "value", allocvalue(f, x), reps(k -> runvalue(f, x, k), 200_000))
  end
end

function tuple_cases(T)
  M = floatmax(T)
  rng = MersenneTwister(747)
  n = 100
  a, b = randn(rng, T, n), randn(rng, T, n)
  c = M / 4 .* (T(0.5) .+ rand(rng, T, n))
  d = M / 4 .* (T(0.5) .+ rand(rng, T, n))
  cases = (("tuple-ordinary-finite-λ2", NormL1Fused(T(0.5), T(0.25)), (a, b)),
           ("tuple-inf-equal", NormL1Fused(T(0.5), T(Inf)), (a, copy(a))),
           ("tuple-recovery-finite-λ2", NormL1Fused(T(1), T(0.5)), (c, d)),
           ("tuple-recovery-inf-equal", NormL1Fused(T(1), T(Inf)), (c, copy(c))))
  for (name, f, x) in cases
    runvalue(f, x, 2)
    report(T, name, n, "value", allocvalue(f, x), reps(k -> runvalue(f, x, k), 20_000))
  end
end

println("julia ", VERSION, " ", Sys.MACHINE)
for T in (Float32, Float64)
  vector_cases(T)
  pair_cases(T)
  tuple_cases(T)
end
println("sink = ", SINK[])
println("== done")
