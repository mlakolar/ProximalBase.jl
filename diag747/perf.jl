# Issue #747: whole-call timings and allocations of NormL1Fused and of the entry points of other families, Float32 and Float64,
# and the LLVM IR of the helpers and entry points (between #IR-BEGIN / #IR-END markers).
# One line per case: BENCH <T> <case> n=<n> <call> alloc=<bytes after warm-up> ns_min=<min of 5> ns_max=<max of 5>
# The result of every call is accumulated into a printed sink, so no call can be dropped. The calls run in function barriers.
using ProximalBase, Random, InteractiveUtils
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

# the whole-call entry points of other families: IndSOC value and prox, IndBallL2 prox, TotalVariation1D prox, NormL1 prox!,
# Scaled(NormL1Fused) value
function runprox(f, y, x, γ, N)
  s = 0.0
  for _ in 1:N
    prox!(y, f, x, γ)
    s += Float64(y[1])
  end
  s
end
function entry_cases(T)
  rng = MersenneTwister(747)
  M = floatmax(T)
  for n in (5, 100, 10_000)
    x = randn(rng, T, n)
    xb = M / 4 .* (T(0.5) .+ rand(rng, T, n))
    N = ncalls(n)
    for (name, f, v) in (("IndSOC-value", IndSOC(), x), ("IndSOC-value-near-floatmax", IndSOC(), xb),
                         ("Scaled-NormL1Fused-value", Scaled(NormL1Fused(T(1), T(0.5)), T(0.5)), x),
                         ("Scaled-NormL1Fused-recovery", Scaled(NormL1Fused(T(1), T(0.5)), T(0.5)), xb))
      runvalue(f, v, 2)
      report(T, name, n, "value", allocvalue(f, v), reps(k -> runvalue(f, v, k), N))
    end
    for (name, f, v) in (("IndSOC-prox", IndSOC(), x), ("IndBallL2-prox", IndBallL2(T(1)), x), ("IndBallL2-prox-near-floatmax", IndBallL2(T(1)), xb),
                         ("TotalVariation1D-prox", TotalVariation1D(T(0.5)), x), ("NormL1-prox!", NormL1(T(0.5)), x))
      y = similar(v)
      runprox(f, y, v, one(T), 2)
      allocp(f, y, v, γ) = @allocated prox!(y, f, v, γ)
      report(T, name, n, "prox!", allocp(f, y, v, one(T)), reps(k -> runprox(f, y, v, one(T), k), N))
    end
  end
end

# LLVM IR of the helpers and of the entry points, between markers, for the comparison of base and head
function irblock(name, f, types)
  println("#IR-BEGIN ", name)
  println(sprint(io -> code_llvm(io, f, types; debuginfo = :none)))
  println("#IR-END ", name)
end
function ir_all()
  for T in (Float32, Float64)
    PB = ProximalBase
    irblock("_scaled_product2 $T", PB._scaled_product, Tuple{T, T, Int})
    irblock("_scaled_product3 $T", PB._scaled_product, Tuple{T, T, T, Int})
    irblock("_power_factors $T", PB._power_factors, Tuple{Type{T}, Int})
    x = zeros(T, 5); y = zeros(T, 5)
    irblock("NormL1Fused value $T", NormL1Fused(T(1), T(0.5)), Tuple{Vector{T}})
    irblock("NormL1Fused value (inf) $T", NormL1Fused(T(1), T(Inf)), Tuple{Vector{T}})
    irblock("IndSOC value $T", IndSOC(), Tuple{Vector{T}})
    irblock("IndSOC prox! $T", prox!, Tuple{Vector{T}, typeof(IndSOC()), Vector{T}, T})
    irblock("IndBallL2 prox! $T", prox!, Tuple{Vector{T}, typeof(IndBallL2(T(1))), Vector{T}, T})
    irblock("TotalVariation1D prox! $T", prox!, Tuple{Vector{T}, typeof(TotalVariation1D(T(0.5))), Vector{T}, T})
    irblock("NormL1 prox! $T", prox!, Tuple{Vector{T}, typeof(NormL1(T(0.5))), Vector{T}, T})
    irblock("Scaled(NormL1Fused) value $T", Scaled(NormL1Fused(T(1), T(0.5)), T(0.5)), Tuple{Vector{T}})
  end
end

println("julia ", VERSION, " ", Sys.MACHINE)
ir_all()
for T in (Float32, Float64)
  vector_cases(T)
  pair_cases(T)
  tuple_cases(T)
  entry_cases(T)
end
println("sink = ", SINK[])
println("== done")
