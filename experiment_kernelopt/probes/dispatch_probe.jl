# Does a probe written as `() -> PB.prox_block!(...)` with PB a captured module (as in benchmark/l2_norms.jl) pay a
# dynamic lookup and dispatch per call?
module T; Base.include(T, "/path/to/pbopt/base/src/ProximalBase.jl"); end
using BenchmarkTools
function probes(PB, m)
  f = PB.GroupNormL2(0.6, [1:m, m+1:2m]); u = randn(m); out = similar(u); γ = 0.8
  viamodule = () -> (PB.prox_block!(out, f, u, 1, γ); nothing)          # the form of l2_norms.jl
  pb! = PB.prox_block!
  viafunction = () -> (pb!(out, f, u, 1, γ); nothing)                   # the function captured as a value
  viamodule, viafunction
end
for m in (2, 32, 1024)
  a, b = probes(T.ProximalBase, m)
  ta = @belapsed $a() samples = 200 evals = 500
  tb = @belapsed $b() samples = 200 evals = 500
  println("m=$m: via captured module ", round(ta * 1e9, digits = 2), " ns, via captured function ", round(tb * 1e9, digits = 2), " ns, allocations ", (@allocated a()), " / ", (@allocated b()))
end
