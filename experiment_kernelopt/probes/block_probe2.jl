module T; Base.include(T, "/path/to/pbopt/base/src/ProximalBase.jl"); end
const PB = T.ProximalBase
using Random, InteractiveUtils
@noinline call(op::F) where {F} = (op(); nothing)
function bench(op::F; evals = 2000, reps = 25) where {F}
  call(op); best = Inf
  for _ in 1:reps
    t = time_ns(); for _ in 1:evals; call(op); end; best = min(best, (time_ns() - t) / evals)
  end
  best
end
@inline function pb_inl!(out, f, u, k, γ)
  m = PB._check_group_block(out, f, u, k, γ)
  PB._l2_shrink!(out, u, Base.OneTo(m), f.λ0, @inbounds(f.w[k]), γ, PB._l2_shrink_fixed!)
  out
end
@noinline pb_noinl!(out, f, u, k, γ) = pb_inl!(out, f, u, k, γ)
@noinline function pb_callfixed!(out, f, u, k, γ)     # the parameter exits out of line (the default of _l2_shrink!)
  m = PB._check_group_block(out, f, u, k, γ)
  PB._l2_shrink!(out, u, Base.OneTo(m), f.λ0, @inbounds(f.w[k]), γ)
  out
end
@noinline function pb_nocheck!(out, f, u, k, γ)
  PB._l2_shrink!(out, u, Base.OneTo(length(u)), f.λ0, @inbounds(f.w[k]), γ)
  out
end
sweep(g::G, outs, f, U, γ) where {G} = () -> (for k in eachindex(U); @inbounds g(outs[k], f, U[k], k, γ); end)
for m in (2, 5, 32)
  Kg = 4096 ÷ m
  rng = MersenneTwister(1); x = randn(rng, Kg * m)
  groups = [(k-1)*m+1:k*m for k in 1:Kg]
  f = PB.GroupNormL2(0.6, groups)
  U = [x[G] for G in groups]; outs = [similar(u) for u in U]; γ = 0.8
  r(g) = round(bench(sweep(g, outs, f, U, γ)) / Kg, digits = 2)
  println("m=$m ns/group: prox_block! ", r(PB.prox_block!), " | same body inlined into the sweep ", r(pb_inl!), " | same body noinline ", r(pb_noinl!),
          " | fixed exits out of line ", r(pb_callfixed!), " | no checks, noinline ", r(pb_nocheck!))
end
f = PB.GroupNormL2(0.6, [1:2, 3:4]); u = [1.0, 2.0]; o = similar(u)
io = IOBuffer(); code_native(io, PB.prox_block!, typeof((o, f, u, 1, 0.8)); debuginfo = :none); s = String(take!(io))
println("prox_block! native: ", count(==('\n'), s), " lines; calls: ", join(unique(m.match for m in eachmatch(r"call\w*\s+\S+", s)), " | "))
println("frame: ", join([strip(l) for l in split(s, '\n')[1:28] if occursin(r"push|sub\s+rsp|gcstack|fs:", l)], " ; "))
