using Random
const P = "/path/to/pbopt"
load(name) = (m = Core.eval(Main, :(module $(Symbol("T_", name)) end)); Base.include(m, joinpath(P, name, "src", "ProximalBase.jl")); getfield(m, :ProximalBase))
const NAMES = split(ARGS[1], ','); const TREES = [load(String(n)) for n in NAMES]
@noinline call(op::F) where {F} = (op(); nothing)
batch(op::F, evals) where {F} = (t = time_ns(); for _ in 1:evals; call(op); end; (time_ns() - t) / evals)
function compare(ops; rounds = 60, evals = 30000)
  for o in ops; batch(o, 3000); end
  best = fill(Inf, length(ops))
  for r in 1:rounds, k in (isodd(r) ? eachindex(ops) : reverse(eachindex(ops))); best[k] = min(best[k], batch(ops[k], evals)); end
  best
end
mk(PB, y, x, γ, λ, w) = (f = w === nothing ? PB.NormL1(λ) : PB.NormL1(λ, w); p = PB.prox!; () -> p(y, f, x, γ))
for (T, weighted, data) in ((Float64, false, :const), (Float64, false, :rand), (Float32, true, :const), (Float32, true, :rand))
  print(VERSION, " ", T, weighted ? " weighted " : " unweighted ", data, ": ")
  for n in (1, 2, 4, 7, 8, 9, 12, 15, 16, 17, 24, 32)
    rng = MersenneTwister(n); x = data === :const ? fill(T(1.1), n) : randn(rng, T, n); y = similar(x); w = weighted ? fill(T(0.75), n) : nothing
    b = compare([mk(PB, y, x, T(0.5), T(0.3), w) for PB in TREES])
    print("n", n, " ", join([string(round(b[k] / b[1], digits = 2)) for k in 2:length(b)], "/"), " [", round(b[1], digits = 1), "]  ")
  end
  println()
end
