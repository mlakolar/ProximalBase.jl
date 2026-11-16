# First-call (compilation) cost of the operations that the candidate trees touch, for one source tree in a fresh
# process: the time to include the source, and the time of the first call of each operation (inference and code
# generation; the operations run in a fixed order, so a later one reuses what an earlier one compiled).
#   julia --project=<env> bench/firstcall.jl <tree root> <name>    →  one CSV line per operation on stdout
using Random
root, name = ARGS[1], ARGS[2]
t0 = time_ns()
module Tree end
Base.include(Tree, joinpath(root, "src", "ProximalBase.jl"))
const PB = Tree.ProximalBase
tinc = (time_ns() - t0) / 1e9
println(name, ",include,", round(tinc, digits = 4))
function ops(::Type{T}) where {T}
  rng = MersenneTwister(1); n = 64
  x = randn(rng, T, n); y = similar(x); w = T(0.5) .+ rand(rng, T, n); γv = T(0.5) .+ rand(rng, T, n); γ = T(0.8); λ = T(0.6)
  groups = [(k-1)*4+1:4k for k in 1:n÷4]; gidx = [collect(G) for G in groups]
  u = x[1:4]; out = similar(u)
  f1, f1w, f2, fg, fgi = PB.NormL1(λ), PB.NormL1(λ, w), PB.NormL2(λ), PB.GroupNormL2(λ, groups), PB.GroupNormL2(λ, gidx)
  ["l1 prox" => () -> PB.prox!(y, f1, x, γ), "l1 prox weights" => () -> PB.prox!(y, f1w, x, γ),
   "l1 prox step array" => () -> PB.prox!(y, f1, x, γv), "l1 prox weights step array" => () -> PB.prox!(y, f1w, x, γv),
   "l1 value" => () -> f1(x), "l1 value weights" => () -> f1w(x),
   "l1 value_and_prox" => () -> PB.value_and_prox!(y, f1, x, γ), "l1 value_and_prox weights" => () -> PB.value_and_prox!(y, f1w, x, γ),
   "l1 prox_block scalar" => () -> PB.prox_block(f1, x[1], 1, γ),
   "l2 prox" => () -> PB.prox!(y, f2, x, γ), "l2 value" => () -> f2(x), "l2 value_and_prox" => () -> PB.value_and_prox!(y, f2, x, γ),
   "group prox" => () -> PB.prox!(y, fg, x, γ), "group value" => () -> fg(x), "group value_and_prox" => () -> PB.value_and_prox!(y, fg, x, γ),
   "group prox index vectors" => () -> PB.prox!(y, fgi, x, γ), "group prox_block" => () -> PB.prox_block!(out, fg, u, 1, γ)]
end
total = 0.0
for T in (Float64, Float32), (label, op) in Base.invokelatest(ops, T)
  t = @elapsed Base.invokelatest(op)
  global total += t
  println(name, ",", T, " ", label, ",", round(t, digits = 4))
end
println(name, ",total first calls,", round(total, digits = 4))
