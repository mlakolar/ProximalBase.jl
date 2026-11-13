# PB #911 review r2 finding 6: paired timing of PB's existing least-squares methods, refactored copy
# (pkgs/pb-H) against PB master, with an A/A control. Three source copies are loaded as separate modules
# the same way (two of PB master: A, A2; the refactor: R), so load mechanism and code layout are alike.
# Paired, randomized order per round. Disposable. Run on an exclusive node.
include(joinpath(@__DIR__, "envsetup.jl"))
using Random, Printf
print_provenance()
function loadpb(name, dir)
  w = Module(Symbol(name))
  Base.include(w, joinpath(RUNDIR, "pkgs", dir, "src", "ProximalBase.jl"))
  Base.invokelatest(getfield, w, :ProximalBase)
end
const P = Dict("A" => loadpb("A", "pb-master"), "A2" => loadpb("A2", "pb-master"), "R" => loadpb("R", "pb-H"))
const NAMES = ["A", "A2", "R"]
const ROUNDS = parse(Int, argopt("ROUNDS", "60"))
med(v) = (s = sort(v); k = length(s); isodd(k) ? s[(k + 1) ÷ 2] : (s[k ÷ 2] + s[k ÷ 2 + 1]) / 2)
function boot_ci(r; B = 4000, rng = Xoshiro(7))
  n = length(r); m = Vector{Float64}(undef, B); buf = similar(r)
  for b in 1:B
    for i in 1:n; buf[i] = r[rand(rng, 1:n)]; end
    m[b] = med(buf)
  end
  sort!(m); (m[max(1, round(Int, 0.025B))], m[min(B, round(Int, 0.975B))])
end
println("CSV header: pbaa,julia,T,n,p,scale,op,variant,ratio_median_paired,ci_lo,ci_hi,median_ns")
for T in (Float64, Float32), (n, p) in ((2000, 500), (240, 160), (100, 32)), sc in (:raw, :scaled)
  rng = Xoshiro(5); X = randn(rng, T, n, p); Y = randn(rng, T, n); β = randn(rng, T, p) ./ T(10)
  fs = Dict(nm => (f = P[nm].LeastSquaresLoss(copy(Y), copy(X)); sc == :raw ? f : P[nm].Scaled(f, T(1 / n))) for nm in NAMES)
  gs = Dict(nm => similar(β) for nm in NAMES)
  for op in ("f(β)", "value_and_gradient!", "gradient!")
    th = Dict(nm => (f = fs[nm]; g = gs[nm]; M = P[nm];
                     op == "f(β)" ? (() -> f(β)) : op == "gradient!" ? (() -> M.gradient!(g, f, β)) : (() -> M.value_and_gradient!(g, f, β)))
              for nm in NAMES)
    for nm in NAMES, _ in 1:3; th[nm](); end
    t0 = time_ns(); th["A"](); t1 = time_ns()
    inner = clamp(ceil(Int, 2e6 / max(t1 - t0, 1)), 1, 100_000)
    times = Dict(nm => Float64[] for nm in NAMES)
    order = Xoshiro(hash((T, n, p, sc, op)) % 10_000)
    for _ in 1:ROUNDS, nm in shuffle(order, NAMES)
      f = th[nm]; t = time_ns()
      for _ in 1:inner; f(); end
      push!(times[nm], (time_ns() - t) / inner)
    end
    for nm in ("A2", "R")
      r = times[nm] ./ times["A"]; lo, hi = boot_ci(r)
      @printf "pbaa,%s,%s,%d,%d,%s,%s,%s,%.4f,%.4f,%.4f,%.1f\n" VERSION T n p sc op nm med(r) lo hi med(times[nm])
    end
    flush(stdout)
  end
end
