# PB #911 review r1 finding 10: the PB-side entry points themselves (not only the PS wrappers that
# call them) — inferred return type, non-inlined calls and dynamic calls in their optimized IR, and
# vector instructions in the shared kernel `_square_value_level` of H. Disposable.
include(joinpath(@__DIR__, "envsetup.jl"))
using Printf, InteractiveUtils
using ProximalBase: LeastSquaresLoss
print_provenance()
function loadvariant(name)
  dst = joinpath(mktempdir(), "src")
  cp(joinpath(@__DIR__, "variants", name, "src"), dst)
  w = Module(Symbol(name)); Base.include(w, joinpath(dst, "ProximalSolvers.jl"))
  Base.invokelatest(getfield, w, :ProximalSolvers)
end
function ir_report(f, argtypes)
  ci = Base.code_typed(f, argtypes; optimize = true)
  src, rt = ci[1]
  inv = String[]; dyn = String[]
  for st in src.code
    if Meta.isexpr(st, :invoke)
      mi = st.args[1]; mi isa Core.CodeInstance && (mi = mi.def)
      push!(inv, string(mi isa Core.MethodInstance ? mi.def.name : mi))
    elseif Meta.isexpr(st, :call)
      g = st.args[1]
      v = g isa GlobalRef ? (isdefined(g.mod, g.name) ? getfield(g.mod, g.name) : nothing) : g
      (v isa Core.IntrinsicFunction || v isa Core.Builtin) && continue
      push!(dyn, string(g))
    end
  end
  string(rt), join(unique(inv), ";"), join(unique(dyn), ";")
end
for nm in ("H", "B2")
  O = loadvariant(nm).PBOracle
  for T in (Float64, Float32), sc in (:raw, :scaled, :mixed)
    sc == :mixed && T === Float64 && continue
    base = LeastSquaresLoss(zeros(T, 4), zeros(T, 4, 2))
    f = sc == :raw ? base : (sc == :scaled ? T(0.5) * base : 0.5 * base)
    VT = Vector{T}; F = typeof(f)
    for (lab, fn, at) in (("value_at_image", O.value_at_image, (F, VT)),
                          ("value_and_gradient_at_image!", O.value_and_gradient_at_image!, (VT, VT, F, VT)),
                          ("value_at_residual", O.value_at_residual, (F, VT)))
      rt, inv, dyn = ir_report(fn, at)
      @printf "%-3s %-7s %-6s %-29s -> %-8s dynamic: %-6s non-inlined: %s\n" nm T sc lab rt (isempty(dyn) ? "none" : dyn) inv
    end
  end
  if nm == "H"
    for T in (Float64, Float32)
      W = T === Float64 ? "double" : "float"
      for (lab, at) in (("_square_value_level (center)", (T, T, Vector{T}, Vector{T}, Bool)), ("_square_value_level (no center)", (T, T, Vector{T}, Nothing, Bool)))
        ir = sprint(io -> code_llvm(io, O._square_value_level, at; debuginfo = :none))
        vec = unique([m.match for m in eachmatch(Regex("<\\d+ x $W>"), ir)])
        @printf "H   %-7s %-32s vector types: %s\n" T lab join(vec, " ")
      end
    end
  end
end
