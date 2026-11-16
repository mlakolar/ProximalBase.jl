# The reviewer's blocker cases, revised trees against master (workstation; several orders, minimum of batches)
using Random
const P = "/path/to/pbopt"
load(name) = (m = Core.eval(Main, :(module $(Symbol("T_", name)) end)); Base.include(m, joinpath(P, name, "src", "ProximalBase.jl")); getfield(m, :ProximalBase))
const NAMES = split(ARGS[1], ',')
const TREES = [(String(n), load(String(n))) for n in NAMES]
@noinline call(op::F) where {F} = (op(); nothing)
batch(op::F, evals) where {F} = (t = time_ns(); for _ in 1:evals; call(op); end; (time_ns() - t) / evals)
function compare(ops; rounds = 40, target = 2e5)
  for o in ops; batch(o, 200); end
  ev = max(1, round(Int, target / max(batch(ops[1], 200), 0.5)))
  best = fill(Inf, length(ops))
  for r in 1:rounds, k in (isodd(r) ? eachindex(ops) : reverse(eachindex(ops))); best[k] = min(best[k], batch(ops[k], ev)); end
  best
end
show(label, b) = println(rpad(label, 62), join([string(NAMES[k], " ", round(b[k] / b[1], digits = 3)) for k in 2:length(b)], "  "), "   [", NAMES[1], " ", round(b[1], digits = 2), " ns]")
function run(::Type{T}) where {T}
  rng = MersenneTwister(1)
  for n in (1, 7, 8, 64, 1000, 16384)   # step arrays: all ordinary, one zero step in the middle, all zero steps, zero level
    x = randn(rng, T, n); y = similar(x); γ = fill(T(0.5), n); γz = copy(γ); γz[(n + 1) ÷ 2] = 0; γ0 = zeros(T, n)
    show("$T n=$n steps ordinary", compare([(f = PB.NormL1(T(0.3)); p = PB.prox!; () -> p(y, f, x, γ)) for (_, PB) in TREES]))
    n >= 8 && show("$T n=$n one zero step (middle)", compare([(f = PB.NormL1(T(0.3)); p = PB.prox!; () -> p(y, f, x, γz)) for (_, PB) in TREES]))
    n >= 8 && show("$T n=$n all steps zero", compare([(f = PB.NormL1(T(0.3)); p = PB.prox!; () -> p(y, f, x, γ0)) for (_, PB) in TREES]))
    n >= 8 && show("$T n=$n zero level", compare([(f = PB.NormL1(T(0)); p = PB.prox!; () -> p(y, f, x, γ)) for (_, PB) in TREES]))
  end
  r = Ref(zero(T))
  for n in (1, 2, 4, 15, 16, 1024, 1025)
    x = randn(rng, T, n)
    show("$T n=$n NormL1 value", compare([(f = PB.NormL1(T(0.3)); () -> (r[] = f(x))) for (_, PB) in TREES]))
    show("$T n=$n NormL2 value", compare([(f = PB.NormL2(T(0.3)); () -> (r[] = f(x))) for (_, PB) in TREES]))
    show("$T n=$n SqrNormL2 value", compare([(f = PB.SqrNormL2(T(0.3)); () -> (r[] = f(x))) for (_, PB) in TREES]))
  end
  for (m, K) in ((1, 1000), (5, 200), (1024, 4))   # zero groups: tiny normal threshold (the guard fails) and ordinary threshold
    x = zeros(T, m * K); y = similar(x); groups = [(k-1)*m+1:k*m for k in 1:K]; w = ones(T, K)
    show("$T zero groups m=$m, level floatmin", compare([(f = PB.GroupNormL2(floatmin(T), groups, w); p = PB.prox!; () -> p(y, f, x, one(T))) for (_, PB) in TREES]))
    show("$T zero groups m=$m, level 0.3", compare([(f = PB.GroupNormL2(T(0.3), groups, w); p = PB.prox!; () -> p(y, f, x, one(T))) for (_, PB) in TREES]))
    xr = randn(rng, T, m * K)
    show("$T dense groups m=$m, level 0.3", compare([(f = PB.GroupNormL2(T(0.3), groups, w); p = PB.prox!; () -> p(y, f, xr, one(T))) for (_, PB) in TREES]))
  end
end
run(Float64); run(Float32)
