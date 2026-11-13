# PB #911 review r1 finding 7: the Float32 "large α" PGD case (numerics.jl family largealpha, λ = 0)
# that throws "line search failed: nonfinite curvature estimate" for some variants. Copies of K, B3
# and H with an instrumented line search log every backtracking decision (mode, values, curvature),
# and the input hash is printed. Disposable.
include(joinpath(@__DIR__, "envsetup.jl"))
using Random, Printf
import SHA
using ProximalBase: LeastSquaresLoss, NormL1
print_provenance()
const HOOK_OLD = "    isfinite(state.L) || error(\"line search failed: nonfinite curvature estimate\")"
const HOOK_NEW = """    Main.lslog(state, local_l, xy_sq)
    isfinite(state.L) || error("line search failed: nonfinite curvature estimate")"""
const LOG = Ref{Vector{String}}(String[])
function lslog(state, local_l, xy_sq)
  push!(LOG[], @sprintf("simple=%-5s f_x=%-14s f_prev=%-14s L=%-12s local_l=%-12s xy_sq=%-12s gnorm2=%s",
        state.backtrack_simple, repr(state.f_x), repr(state.f_x_previous), repr(state.L), repr(local_l), repr(xy_sq),
        repr(sum(abs2, state.grad_f_x))))
end
function loadvariant(name)
  dst = joinpath(mktempdir(), "src")
  cp(joinpath(@__DIR__, "variants", name, "src"), dst)
  p = joinpath(dst, "line_searches.jl")
  s = read(p, String); occursin(HOOK_OLD, s) || error("hook anchor missing"); write(p, replace(s, HOOK_OLD => HOOK_NEW))
  w = Module(Symbol(name)); Base.include(w, joinpath(dst, "ProximalSolvers.jl"))
  Base.invokelatest(getfield, w, :ProximalSolvers)
end
const V = Dict(nm => loadvariant(nm) for nm in ("K", "B3", "H"))
# the same construction as numerics.jl solve_family("largealpha", Float32, Xoshiro(20911))
T = Float32; rng = Xoshiro(20911); n, p = 40, 5
X = randn(rng, T, n, p) * T.([1 0.5 0.3 0 0; 0 1 0.5 0 0; 0 0 1 0.2 0; 0 0 0 1 0.4; 0 0 0 0 1])
β = T[1, -2, 0.5, 0, 0]; y = X * β .+ T(0.1) .* randn(rng, T, n); α = 1f30
println("input hash ", string(hash((X, y)); base = 16), "  floatmax(Float32) = ", floatmax(T))
println("version-independent digest: SHA-256 of the bit patterns of X and y = ",
        bytes2hex(SHA.sha256(vcat(reinterpret(UInt8, vec(X)), reinterpret(UInt8, y)))))
println("(logged: the backtracking decisions that reject a trial and raise L; accepted trials return before the hook)")
for nm in ("K", "B3", "H")
  M = V[nm]; LOG[] = String[]
  x = zeros(T, p)
  r = try
    res = M.solve!(x, α * LeastSquaresLoss(y, X), NormL1(zero(T)), M.ProximalGradientDescent(),
                   M.ProximalOptions(maxiter = 20_000, ftol = 0, xtol = 1e-7, gradtol = 1e-7))
    "returned: iterations=$(res.iterations)"
  catch e
    "threw: " * sprint(showerror, e)
  end
  println("== $nm: $r; backtracking decisions logged: $(length(LOG[]))")
  for l in LOG[]
    println("   ", l)
  end
end
