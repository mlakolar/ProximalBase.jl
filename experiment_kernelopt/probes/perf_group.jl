using LinuxPerf, Random, LinearAlgebra
module T; Base.include(T, "/path/to/pbopt/base/src/ProximalBase.jl"); end
module K; include("/path/to/pbopt/harness/experiment_kernelopt/kernels/kernels.jl"); end
const PB = T.ProximalBase
@noinline call(op::F) where {F} = (op(); nothing)
function rep(op::F, n) where {F}
  for _ in 1:n; call(op); end
end
function counters(op::F, Kg; n = 20000) where {F}
  rep(op, 2000)
  b = LinuxPerf.make_bench(LinuxPerf.parse_groups("(cpu-cycles,instructions,branch-instructions,branch-misses)"))
  LinuxPerf.enable!(b); rep(op, n); LinuxPerf.disable!(b)
  c = LinuxPerf.counters(b)
  vals = [Float64(x.value) * (x.running > 0 ? x.enabled / x.running : NaN) for x in c.counters]
  close(b)
  vals ./ (n * Kg)
end
for m in (2, 5, 32, 1024), pattern in (:allshrink, :sparse)
  Kg = 4096 ÷ m
  rng = MersenneTwister(900 + m); x = randn(rng, Kg * m); y = similar(x)
  groups = [(k-1)*m+1:k*m for k in 1:Kg]
  pattern === :sparse && for (k, G) in enumerate(groups); k % 4 != 1 && (x[G] .= 0); end
  w = fill(sqrt(Float64(m)), Kg); λ = 0.6; γ = 0.8
  f = PB.GroupNormL2(λ, groups, w); r = Ref(0.0)
  ops = ["PB prox!" => () -> (PB.prox!(y, f, x, γ); nothing), "direct prox" => () -> (K.group_prox!(y, x, groups, w, γ * λ); nothing),
         "PB value_and_prox!" => () -> (r[] = PB.value_and_prox!(y, f, x, γ); nothing), "direct valprox" => () -> (r[] = K.group_valprox!(y, x, groups, w, γ, λ); nothing),
         "PB value" => () -> (r[] = f(x); nothing), "direct value" => () -> (r[] = K.group_value(x, groups, w, λ); nothing)]
  for (name, op) in ops
    c = counters(op, Kg)
    println(rpad("m=$m $pattern $name", 40), " per group: cycles ", round(c[1], digits = 1), "  instructions ", round(c[2], digits = 1), "  branches ", round(c[3], digits = 1), "  mispredicts ", round(c[4], digits = 3))
  end
end
