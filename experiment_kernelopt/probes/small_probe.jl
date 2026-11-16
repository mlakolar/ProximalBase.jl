# Small-input crossovers of B (step arrays), C (unweighted L1 value) and C2 (L2 / squared-L2 values) against master.
using Random
const P = "/path/to/pbopt"
load(name) = (m = Core.eval(Main, :(module $(Symbol("T_", name)) end)); Base.include(m, joinpath(P, name, "src", "ProximalBase.jl")); getfield(m, :ProximalBase))
const TREES = Dict(n => load(n) for n in ("base", "B", "C", "C2"))
@noinline call(op::F) where {F} = (op(); nothing)
function batch(op::F, evals) where {F}
  t = time_ns(); for _ in 1:evals; call(op); end; (time_ns() - t) / evals
end
function compare(ops::Vector; rounds = 30, evals = 20000)
  for o in ops; batch(o, 2000); end
  best = fill(Inf, length(ops))
  for r in 1:rounds, k in (isodd(r) ? eachindex(ops) : reverse(eachindex(ops)))
    best[k] = min(best[k], batch(ops[k], evals))
  end
  best
end
mk_prox(PB, y, x, γ, λ) = (f = PB.NormL1(λ); p = PB.prox!; () -> p(y, f, x, γ))
mk_val1(PB, r, x, λ) = (f = PB.NormL1(λ); () -> (r[] = f(x)))
mk_val2(PB, r, x, λ) = (f = PB.NormL2(λ); () -> (r[] = f(x)))
mk_valsq(PB, r, x, λ) = (f = PB.SqrNormL2(λ); () -> (r[] = f(x)))
for T in (Float64, Float32)
  println("== ", VERSION, " ", T, ": candidate/master (two same-tree columns are the A/A noise)")
  for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    rng = MersenneTwister(n); x = randn(rng, T, n); y = similar(x); γ = T(0.5) .+ rand(rng, T, n); λ = T(0.3); r = Ref(zero(T))
    b = compare([mk_prox(TREES["base"], y, x, γ, λ), mk_prox(TREES["B"], y, x, γ, λ), mk_prox(TREES["C"], y, x, γ, λ)])
    c = compare([mk_val1(TREES["base"], r, x, λ), mk_val1(TREES["C"], r, x, λ), mk_val1(TREES["B"], r, x, λ)])
    d = compare([mk_val2(TREES["base"], r, x, λ), mk_val2(TREES["C2"], r, x, λ), mk_val2(TREES["B"], r, x, λ)])
    e = compare([mk_valsq(TREES["base"], r, x, λ), mk_valsq(TREES["C2"], r, x, λ), mk_valsq(TREES["B"], r, x, λ)])
    f(v) = string(round(v[2] / v[1], digits = 2), " (", round(v[3] / v[1], digits = 2), ")")
    println("n=", lpad(n, 2), "  B step-array prox ", f(b), " [", round(b[1], digits = 1), " ns]  C l1 value ", f(c), " [", round(c[1], digits = 1), "]  C2 l2 value ", f(d), " [", round(d[1], digits = 1), "]  C2 sqr value ", f(e), " [", round(e[1], digits = 1), "]")
  end
end
