# Records the method table of Base.ldexp BEFORE ProximalBase is loaded (the other packages ProximalBase loads are loaded first,
# so that only the effect of ProximalBase itself is compared). Usage: julia --project diag747/gate_before.jl <before.txt>
using LinearAlgebra, SparseArrays, PrecompileTools
sigs = sort!(String[string(m.sig) for m in methods(Base.ldexp)])
open(ARGS[1], "w") do io
  println(io, "n=", length(sigs))
  for s in sigs
    println(io, "sig=", s)
  end
  println(io, "which=", string(which(Base.ldexp, Tuple{BigFloat, Int64}).sig))
end
println("recorded ", length(sigs), " methods of Base.ldexp before ProximalBase; which(BigFloat, Int64) = ", which(Base.ldexp, Tuple{BigFloat, Int64}))
println("is ProximalBase loaded? ", any(k -> k.name == "ProximalBase", keys(Base.loaded_modules)))
