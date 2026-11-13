# PB #911: end-to-end cost of the cache-aware oracle alternatives (disposable measurement script).
# Adapted from the PS #277 cost harness. Variants are source copies of PS master loaded as separate
# modules in one process (nothing is redefined in place); the PB-side oracle of each variant is its
# src/pb_oracle.jl.
#
# Run (cwd = worktree root):  julia --startup-file=no <this file> NAME=value ...
#   PB_ENV     master (default) | pin
#   MODE       comma list of count,diag,alloc,types,kern,time          (default diag,alloc,types)
#   CASES      lasso,B,Bwarm,C,D,big,rawB,mixB,B0,Blow,Bhigh           (default lasso,B,Bwarm,C,D,rawB)
#              B0/Blow/Bhigh: design B at λ = 0, 0.01 λmax, 0.5 λmax (harness λ = 0.15 λmax);
#              mixB: design B in Float32 with the Float64 scale 1/n as supplied (f32 only)
#   SOLVERS    cd,cds,pg,apg (cds = CD on a SparseIterate, design B only) (default cd,pg,apg)
#   TYPES      f64,f32
#   ROUNDS     timing rounds per configuration (default 60); TARGET_MS batch target (default 8)
#   BUDGET_S   stop starting new timing configurations after this many seconds (default 1e9)
#   VARIANTS   default K,K2,K3,B1,B2,B3,B3u,H,C
include(joinpath(@__DIR__, "envsetup.jl"))
using Random, Printf, SparseArrays, InteractiveUtils
using ProximalBase: LeastSquaresLoss, NormL1, SparseIterate
const T0_SCRIPT = time()
print_provenance()

envlist(name, default) = String.(split(argopt(name, default), ','; keepempty = false))
const MODES = envlist("MODE", "diag,alloc,types")
const CASES = envlist("CASES", "lasso,B,Bwarm,C,D,rawB")
const SOLVERS = envlist("SOLVERS", "cd,pg,apg")
const TYPES = envlist("TYPES", "f64,f32")
const ROUNDS = parse(Int, argopt("ROUNDS", "60"))
const TARGET_NS = 1e6 * parse(Float64, argopt("TARGET_MS", "8"))
const BUDGET_S = parse(Float64, argopt("BUDGET_S", "1e9"))
const VARIANT_NAMES = envlist("VARIANTS", "K,K2,K3,B1,B2,B3,B3u,H,C")
const COUNT_NAMES = filter(v -> !(v in ("K2", "K3", "B3u")), VARIANT_NAMES)
println("MODES=", MODES, " CASES=", CASES, " SOLVERS=", SOLVERS, " TYPES=", TYPES, " ROUNDS=", ROUNDS,
        " TARGET_MS=", TARGET_NS / 1e6, " VARIANTS=", VARIANT_NAMES)
include(joinpath(WTROOT, "benchmark", "problems.jl"))       # regression_problem, lambda_max
include(joinpath(WTROOT, "benchmark", "certificates.jl"))   # lasso_certificate, cert_loading_refit

# ---------------------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------------------
const VARIANT_DIR = joinpath(@__DIR__, "variants")
const PRELUDE = """
const _CNT = zeros(Int, 8)   # 1 X*v, 2 X'*v, 3 CD column-dot passes, 4 CD column-axpy passes
@inline function _cmul!(y, A, x)
  A isa Transpose ? (_CNT[2] += 1) : (_CNT[1] += 1)
  mul!(y, A, x)
end
"""
function countify!(dst)
  for (dir, _, names) in walkdir(dst), nm in names
    endswith(nm, ".jl") || continue
    path = joinpath(dir, nm)
    s = read(path, String)
    s = replace(s, "mul!(" => "_cmul!(")
    if nm == "pb_oracle.jl"
      s = replace(s, "using LinearAlgebra\n" => "using LinearAlgebra\nimport .._cmul!\n"; count = 1)
    end
    if nm == "leastsquares.jl" && basename(dir) == "cd"
      a = "  b = zero(T)\n  if isnan(a)"
      b = "  r, X = m.r, m.X\n"
      occursin(a, s) && occursin(b, s) || error("CD hooks do not match")
      s = replace(s, a => "  _CNT[3] += 1\n" * a)
      s = replace(s, b => "  _CNT[4] += 1\n" * b)
    end
    write(path, s)
  end
  main = joinpath(dst, "ProximalSolvers.jl")
  s = read(main, String)
  occursin("using LinearAlgebra\n", s) || error("prelude anchor missing")
  write(main, replace(s, "using LinearAlgebra\n" => "using LinearAlgebra\n" * PRELUDE; count = 1))
end
function loadvariant(name::AbstractString; count::Bool = false)
  tmp = mktempdir()
  dst = joinpath(tmp, "src")
  cp(joinpath(VARIANT_DIR, name, "src"), dst)
  count && countify!(dst)
  w = Module(Symbol(name, count ? "_count" : ""))
  t = @elapsed Base.include(w, joinpath(dst, "ProximalSolvers.jl"))
  @printf "loaded variant %s%s (include %.1f s)\n" name (count ? " [count build]" : "") t
  flush(stdout)
  Base.invokelatest(getfield, w, :ProximalSolvers)
end
const VMODS = Dict{String, Module}()
const CMODS = Dict{String, Module}()
"count" in MODES && for nm in COUNT_NAMES
  CMODS[nm] = loadvariant(nm; count = true)
end
any(m -> m in MODES, ("diag", "alloc", "types", "kern", "time")) && for nm in VARIANT_NAMES
  VMODS[nm] = loadvariant(nm)
end

# ---------------------------------------------------------------------------------------
# problems: the formal harness's current-core designs plus 2000×500
# ---------------------------------------------------------------------------------------
const DESIGNS = Dict(
  "lasso" => rep -> regression_problem(rep, "core/lasso"),
  "big" => rep -> regression_problem(rep, "extra/big"; n = 2000, p = 500),
  "B" => rep -> regression_problem(rep, "extended/design-B"; n = 240, p = 160, design = :B),
  "C" => rep -> regression_problem(rep, "extended/design-C"; n = 800, p = 120, design = :C),
  "D" => rep -> regression_problem(rep, "extended/design-D"; n = 180, p = 600, design = :D))
# case => (design, warm, raw, λ factor relative to λmax or nothing for the harness λ, mixed scale)
const CASEDEF = Dict(
  "lasso" => ("lasso", false, false, nothing, false), "B" => ("B", false, false, nothing, false),
  "Bwarm" => ("B", true, false, nothing, false), "C" => ("C", false, false, nothing, false),
  "D" => ("D", false, false, nothing, false), "big" => ("big", false, false, nothing, false),
  "rawB" => ("B", false, true, nothing, false), "mixB" => ("B", false, false, nothing, true),
  "B0" => ("B", false, false, 0.0, false), "Blow" => ("B", false, false, 0.01, false),
  "Bhigh" => ("B", false, false, 0.5, false))
const DRAWS = Dict{String, Any}()
draw(design) = get!(() -> DESIGNS[design](1), DRAWS, design)
function warm_start(a)
  refit = cert_loading_refit(a.X, a.y, 1.25a.λ, a.w, zeros(size(a.X, 2)); tol = 1e-12, maxpasses = 100_000)
  refit.converged || error("the warm start did not certify")
  refit.β
end
struct Prob{T, S}
  id::String
  raw::Bool
  X::Matrix{T}
  y::Vector{T}
  s::S              # the scale as supplied: T, or Float64 for the mixed case
  λ::T
  start::Vector{T}
  n::Int
  p::Int
  X64::Matrix{Float64}   # the rounded data, exactly, for the certificate
  y64::Vector{Float64}
end
const WARM = Dict{String, Vector{Float64}}()
function make_prob(caseid, ::Type{T}) where {T}
  design, warm, raw, lf, mixed = CASEDEF[caseid]
  a = draw(design)
  n, p = size(a.X)
  start = warm ? get!(() -> warm_start(a), WARM, design) : zeros(p)
  λ = lf === nothing ? a.λ : lf * lambda_max(a.X, a.y; s = a.s)
  X, y = T.(a.X), T.(a.y)
  s = mixed ? Float64(a.s) : T(a.s)
  # raw least squares: the same data with the loss n times larger, so the penalty level is n λ
  Prob{T, typeof(s)}(caseid, raw, X, y, s, T(raw ? λ * n : λ), T.(start), n, p, Float64.(X), Float64.(y))
end
valid_combo(c, s, tn) = (s != "cds" || c == "B") && (c != "mixB" || tn == "f32")
combos() = [(c, s, t) for c in CASES for s in SOLVERS for t in TYPES if valid_combo(c, s, t)]
function sparse_start(start::Vector{T}) where {T}
  x = SparseIterate(T, length(start))
  for j in eachindex(start)
    iszero(start[j]) || (x[j] = start[j])
  end
  x
end
struct Inst{F, G, M, O, X}
  name::String
  V::Module
  f::F
  g::G
  method::M
  options::O
  x::X
  start::X
end
# PGD/APG: objective-change stopping off (ftol = 0, #286), so that the stop is the iterate/gradient
# tests and the certificate status is not decided by an exactly flat computed objective (#281)
gradient_options(V, ::Type{Float64}) = V.ProximalOptions(maxiter = 10_000, ftol = 0, xtol = 1e-14, gradtol = 1e-11)
gradient_options(V, ::Type{Float32}) = V.ProximalOptions(maxiter = 5_000, ftol = 0, xtol = 1e-6, gradtol = 1e-6)
function make_inst(name, V::Module, pr::Prob{T}, solver) where {T}
  base = LeastSquaresLoss(pr.y, pr.X)
  f = pr.raw ? base : pr.s * base
  g = NormL1(pr.λ)
  if solver in ("cd", "cds")
    method = V.CDSolver(order = :cyclic, tol = T == Float32 ? 1e-6 : 1e-11, rng = Xoshiro(9))
    options = nothing
  else
    method = solver == "pg" ? V.ProximalGradientDescent() : V.AccProxGradDescent()
    options = gradient_options(V, T)
  end
  start = solver == "cds" ? sparse_start(pr.start) : copy(pr.start)
  Inst(name, V, f, g, method, options, copy(start), start)
end
runner(solve!, x, f, g, method, ::Nothing) = () -> solve!(x, f, g, method)
runner(solve!, x, f, g, method, options) = () -> solve!(x, f, g, method, options)
make_run(i::Inst) = runner(i.V.solve!, i.x, i.f, i.g, i.method, i.options)
make_reset(i::Inst) = (x = i.x; s = i.start; () -> copyto!(x, s))
function timebatch(run!, reset!, inner)
  t0 = time_ns()
  for _ in 1:inner
    reset!(); run!()
  end
  (time_ns() - t0) / inner
end
dense(x::AbstractVector) = collect(x)
bitsof(v::Vector{Float64}) = reinterpret(UInt64, v)
bitsof(v::Vector{Float32}) = reinterpret(UInt32, v)
med(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
function boot_ci(r; B = 4000, rng = Xoshiro(7))
  n = length(r)
  m = Vector{Float64}(undef, B)
  buf = Vector{Float64}(undef, n)
  for b in 1:B
    for i in 1:n
      buf[i] = r[rand(rng, 1:n)]
    end
    m[b] = med(buf)
  end
  sort!(m)
  (m[max(1, round(Int, 0.025B))], m[min(B, round(Int, 0.975B))])
end
const CSV = String[]
emit(kind, case, solver, T, variant, metric, idx, value) =
  push!(CSV, join(replace.(string.((kind, VERSION, case, solver, T, variant, metric, idx, value)), ',' => ';'), ","))
tname(::Type{Float64}) = "f64"
tname(::Type{Float32}) = "f32"
ttype(tn) = tn == "f64" ? Float64 : Float32
# the independent certificate (benchmark/certificates.jl) on the rounded data, at the scale as supplied
const CORE_TARGETS = Dict(Float64 => 1e-8, Float32 => 1e-5)
function certificate(pr::Prob, x)
  s = pr.raw ? 1.0 : Float64(pr.s)
  lasso_certificate(pr.X64, pr.y64, s, Float64(pr.λ), ones(pr.p), Float64.(x))
end

# ---------------------------------------------------------------------------------------
# one run with its counters (PGD/APG through the internals, to reach the state)
# ---------------------------------------------------------------------------------------
function diag_run(V::Module, i::Inst, solver)
  copyto!(i.x, i.start)
  if solver == "cd" || solver == "cds"
    r = V.solve!(i.x, i.f, i.g, i.method)
    return (; x = dense(i.x), obj = Float64(r.minimum), iters = r.passes, converged = r.converged,
            f_calls = -1, grad_calls = -1, prox_calls = -1, fallbacks = -1, restarts = -1, backtracks = -1)
  end
  V._check_keywords(NamedTuple())
  V._check_problem(i.x, i.f, i.g)
  if solver == "apg"
    V._check_method(i.method, eltype(i.x))
    V._check_bound_size(i.f, eltype(i.x))
  end
  state = V.initial_state(i.method, i.options, i.f, i.g, i.x)
  r = V._solve_gradient!(state, i.f, i.g, i.method, i.options)
  (; x = dense(i.x), obj = Float64(r.minimum), iters = r.iterations, converged = r.converged,
   f_calls = r.f_calls, grad_calls = r.grad_calls, prox_calls = r.prox_calls,
   fallbacks = solver == "apg" ? state.fallbacks : -1, restarts = solver == "apg" ? state.restarts : -1,
   backtracks = r.backtrack_steps)
end

# ---------------------------------------------------------------------------------------
# count: exact products from instrumented builds, against the PS #277 formulas
# ---------------------------------------------------------------------------------------
predicted_products(solver, d) = solver == "pg" ? (1 + d.prox_calls, d.grad_calls) :
                                solver == "apg" ? (1 + d.prox_calls + d.fallbacks, d.grad_calls) : nothing
function count_mode()
  println("\n==== count mode ====")
  for (c, s, tn) in combos()
    pr = make_prob(c, ttype(tn))
    for nm in COUNT_NAMES
      V = CMODS[nm]
      inst = make_inst(nm, V, pr, s)
      V._CNT .= 0
      d = diag_run(V, inst, s)
      cnt = copy(V._CNT)
      for (k, v) in zip(("fwd", "adj", "cd_dot_passes", "cd_axpy_passes"), cnt[1:4])
        emit("count", c, s, tn, nm, k, "", v)
      end
      pred = predicted_products(s, d)
      if pred !== nothing
        ok = pred == (cnt[1], cnt[2])
        emit("count", c, s, tn, nm, "formula_ok", "", ok)
        @printf "%-6s %-4s %-3s %-3s iters=%-5d fwd=%-6d adj=%-6d formula=(%d,%d) %s f/g/prox=%d/%d/%d fb=%d\n" c s tn nm d.iters cnt[1] cnt[2] pred[1] pred[2] (ok ? "OK" : "MISMATCH") d.f_calls d.grad_calls d.prox_calls d.fallbacks
      else
        @printf "%-6s %-4s %-3s %-3s passes=%-5d dot_passes=%-7d axpy_passes=%-7d fwd=%d adj=%d\n" c s tn nm d.iters cnt[3] cnt[4] cnt[1] cnt[2]
      end
      flush(stdout)
    end
  end
end

# ---------------------------------------------------------------------------------------
# diag: trajectories, flags, counts and the independent certificate, per variant
# ---------------------------------------------------------------------------------------
function diag_mode()
  println("\n==== diag mode (K is the reference for bitwise comparisons; the certificate is independent) ====")
  for (c, s, tn) in combos()
    T = ttype(tn)
    pr = make_prob(c, T)
    ref = nothing
    for nm in VARIANT_NAMES
      V = VMODS[nm]
      inst = make_inst(nm, V, pr, s)
      d = diag_run(V, inst, s)
      cert = certificate(pr, d.x)
      ref === nothing && (ref = d)
      same = bitsof(d.x) == bitsof(ref.x)
      mad = maximum(abs.(Float64.(d.x) .- Float64.(ref.x)))
      reldx = mad / max(maximum(abs, Float64.(ref.x)), floatmin(Float64))
      ok = cert.status == :ok && cert.value <= CORE_TARGETS[T]
      for (k, v) in pairs((; iterations = d.iters, converged = d.converged, f_calls = d.f_calls, grad_calls = d.grad_calls,
                            prox_calls = d.prox_calls, fallbacks = d.fallbacks, restarts = d.restarts,
                            backtracks = d.backtracks, bitwise_x_equal_K = same, rel_dx_vs_K = reldx,
                            reported_objective = repr(d.obj), cert_measure = cert.measure, cert_status = cert.status,
                            cert_value = cert.value, cert_primal = repr(cert.primal), meets_target = ok))
        emit("diag", c, s, tn, nm, String(k), "", v)
      end
      @printf "%-6s %-4s %-3s %-3s iters=%-5d conv=%-5s f/g/prox=%d/%d/%d fb=%d rs=%d bitwise_x=%-5s rel_dx=%.2e cert=%s %.3e %s\n" c s tn nm d.iters d.converged d.f_calls d.grad_calls d.prox_calls d.fallbacks d.restarts same reldx cert.measure cert.value (ok ? "MEETS" : "MISSES")
      flush(stdout)
    end
  end
end

# ---------------------------------------------------------------------------------------
# alloc: warmed solve, one warmed iteration (PGD/APG) or pass (CD)
# ---------------------------------------------------------------------------------------
alloc_of(run!) = (b = @allocated run!(); b)
function iter_alloc(upd!, f, g, state, method)
  for _ in 1:3
    upd!(f, g, state, method)
  end
  worst = 0
  for _ in 1:5
    worst = max(worst, @allocated upd!(f, g, state, method))
  end
  worst
end
function pass_alloc(pass!, work, m, g, st)
  pass!(work, m, g, st, true, true)
  pass!(work, m, g, st, true, true)
  worst = 0
  for full in (true, false, false)
    worst = max(worst, @allocated pass!(work, m, g, st, full, true))
  end
  worst
end
function alloc_mode()
  println("\n==== alloc mode ====")
  for (c, s, tn) in combos()
    pr = make_prob(c, ttype(tn))
    for nm in VARIANT_NAMES
      V = VMODS[nm]
      inst = make_inst(nm, V, pr, s)
      run! = make_run(inst); reset! = make_reset(inst)
      for _ in 1:3
        reset!(); run!()
      end
      reset!()
      full = alloc_of(run!)
      if s in ("pg", "apg")
        copyto!(inst.x, inst.start)
        state = V.initial_state(inst.method, inst.options, inst.f, inst.g, inst.x)
        iter = iter_alloc(V.update_state!, inst.f, inst.g, state, inst.method)
      else
        work = s == "cds" ? copy(inst.start) : SparseIterate(inst.start)
        m = V.cdmodel(inst.f, work)
        V.initialize!(m, work)
        st = V._CDState(eltype(work), work, inst.method)
        iter = pass_alloc(V._pass!, work, m, inst.g, st)
        # the once-per-solve reported value
        V.loss_value(m, work); emit("alloc", c, s, tn, nm, "loss_value_bytes", "", @allocated V.loss_value(m, work))
      end
      emit("alloc", c, s, tn, nm, "solve_bytes", "", full)
      emit("alloc", c, s, tn, nm, s in ("pg", "apg") ? "iteration_bytes" : "pass_bytes", "", iter)
      @printf "%-6s %-4s %-3s %-3s solve_bytes=%-9d %s=%d\n" c s tn nm full (s in ("pg", "apg") ? "iter_bytes" : "pass_bytes") iter
      flush(stdout)
    end
  end
end

# ---------------------------------------------------------------------------------------
# oracle-level probes shared by types and kern: the solver-facing calls of each variant
# ---------------------------------------------------------------------------------------
function oracle_thunks(V::Module, X::Matrix{T}, Y::Vector{T}, x::Vector{T}, α) where {T}
  z = X * x
  r = Y .- z
  G = zeros(T, size(X, 2))
  d = similar(Y)
  base = LeastSquaresLoss(Y, X)
  f = α === true ? base : α * base
  m = V._image_model(f)
  lm = V.cdmodel(f, SparseIterate(T, size(X, 2)))
  copyto!(lm.r, r)
  fv = V._image_value; fg = V._image_value_and_gradient!; lv = V.loss_value
  w = SparseIterate(T, size(X, 2))
  ["value" => (() -> fv(m, z)), "value_and_gradient" => (() -> fg(G, m, z, d)), "residual_value" => (() -> lv(lm, w))]
end
# non-inlined calls left in the optimized IR of a call, and whether the code is fully inferred
function ir_report(f, argtypes)
  ci = Base.code_typed(f, argtypes; optimize = true)
  length(ci) == 1 || return ("ambiguous", "")
  src, rt = ci[1]
  calls = String[]
  dyn = 0
  for st in src.code
    if Meta.isexpr(st, :invoke)
      mi = st.args[1]
      mi isa Core.CodeInstance && (mi = mi.def)
      push!(calls, string(mi isa Core.MethodInstance ? mi.def.name : mi))
    elseif Meta.isexpr(st, :call)
      g = st.args[1]
      v = g isa GlobalRef ? (isdefined(g.mod, g.name) ? getfield(g.mod, g.name) : nothing) : g
      (v isa Core.IntrinsicFunction || v isa Core.Builtin) && continue
      dyn += 1
      push!(calls, "dynamic:" * string(g))
    end
  end
  (string(rt), join(unique(calls), ";"))
end
function types_mode()
  println("\n==== types mode: inference, calls left after inlining, vectorized loops ====")
  for tn in TYPES
    T = ttype(tn)
    pr = make_prob("B", T)
    xo = T.(0.1 .* randn(Xoshiro(11), pr.p))
    for αk in ("scaled", "raw"), nm in VARIANT_NAMES
      nm in ("K2", "K3") && continue
      V = VMODS[nm]
      α = αk == "raw" ? true : pr.s
      base = LeastSquaresLoss(pr.y, pr.X); f = α === true ? base : α * base
      m = V._image_model(f); VT = Vector{T}
      lm = V.cdmodel(f, SparseIterate(T, pr.p))
      for (lab, fn, at) in (("value", V._image_value, (typeof(m), VT)),
                            ("value_and_gradient", V._image_value_and_gradient!, (VT, typeof(m), VT, VT)),
                            ("residual_value", V.loss_value, (typeof(lm), SparseIterate{T, 1})))
        rt, calls = ir_report(fn, at)
        emit("types", αk, lab, tn, nm, "return_type", "", rt)
        emit("types", αk, lab, tn, nm, "calls", "", calls)
        @printf "%-3s %-6s %-3s %-19s -> %-8s calls: %s\n" tn αk nm lab rt calls
      end
    end
    # vector instructions in the reduction loops (LLVM IR of the kernels themselves)
    W = T === Float64 ? "double" : "float"
    for (lab, fn, at) in (("PB._sumabs2_diff", ProximalBase._sumabs2_diff, (Vector{T}, Vector{T})),
                          ("B2._loss_sum", VMODS["B2"].PBOracle._loss_sum, (ProximalBase.Squared, Vector{T}, Vector{T})),
                          ("B2._loss_sum_deriv!", VMODS["B2"].PBOracle._loss_sum_deriv!, (Vector{T}, ProximalBase.Squared, Vector{T}, Vector{T})),
                          ("PB._ridge_gradient_dense!", ProximalBase._ridge_gradient_dense!, (Vector{T}, T, Vector{T}, Vector{T})))
      ir = sprint(io -> code_llvm(io, fn, at; debuginfo = :none))
      vec = [m.match for m in eachmatch(Regex("<\\d+ x $W>"), ir)]
      emit("types", "-", lab, tn, "-", "vector_types", "", join(unique(vec), ";"))
      @printf "%-3s %-26s vector types: %s\n" tn lab join(unique(vec), " ")
    end
    flush(stdout)
  end
end

# ---------------------------------------------------------------------------------------
# kern: the oracle calls alone, paired and alternating, on ordinary and recovery inputs
# ---------------------------------------------------------------------------------------
function kern_inputs(::Type{T}) where {T}
  a = draw("big"); b = draw("B")
  ins = Any[]
  for (lab, a1) in (("n2000p500", a), ("n240p160", b))
    X, y = T.(a1.X), T.(a1.y)
    push!(ins, (lab * "-ordinary", X, y, T.(0.05 .* randn(Xoshiro(3), size(X, 2)))))
  end
  # recovery: residual squares that underflow (finite residuals, sum below n floatmin/eps)
  tiny = T === Float64 ? 1e-160 : 1f-22
  X, y = T.(b.X), T.(b.y) .* T(tiny)
  push!(ins, ("n240p160-underflow", X, y, zeros(T, size(X, 2))))
  ins
end
function kern_mode()
  println("\n==== kern mode: oracle calls alone (rounds=$ROUNDS) ====")
  for tn in TYPES
    T = ttype(tn)
    for (lab, X, Y, x) in kern_inputs(T), αk in ("scaled", "raw")
      α = αk == "raw" ? true : T(1 / size(X, 1))
      th = Dict(nm => Dict(oracle_thunks(VMODS[nm], X, Y, x, α)) for nm in VARIANT_NAMES)
      for op in ("value", "value_and_gradient", "residual_value")
        for nm in VARIANT_NAMES
          t = th[nm][op]; t(); t()
          emit("kern", lab * "/" * αk, op, tn, nm, "bytes", "", @allocated t())
          emit("kern", lab * "/" * αk, op, tn, nm, "result", "", repr(t()))
        end
        tk = med([(t0 = time_ns(); th["K"][op](); Float64(time_ns() - t0)) for _ in 1:21])
        inner = clamp(ceil(Int, 2e6 / max(tk, 1.0)), 1, 100_000)
        times = Dict(nm => Float64[] for nm in VARIANT_NAMES)
        rng = Xoshiro(20261113 + hash((lab, αk, op, tn)) % 1000)
        for _ in 1:ROUNDS, nm in shuffle(rng, VARIANT_NAMES)
          t = th[nm][op]
          t0 = time_ns()
          for _ in 1:inner
            t()
          end
          push!(times[nm], (time_ns() - t0) / inner)
        end
        K = times["K"]
        for nm in VARIANT_NAMES
          rr = times[nm] ./ K
          lo, hi = nm == "K" ? (1.0, 1.0) : boot_ci(rr)
          emit("kern", lab * "/" * αk, op, tn, nm, "median_ns", "", med(times[nm]))
          emit("kern", lab * "/" * αk, op, tn, nm, "ratio_median_paired", "", med(rr))
          emit("kern", lab * "/" * αk, op, tn, nm, "ratio_ci_lo", "", lo)
          emit("kern", lab * "/" * αk, op, tn, nm, "ratio_ci_hi", "", hi)
          @printf "%-3s %-22s %-6s %-19s %-3s median %10.1f ns  ratio %.3f [%.3f, %.3f]\n" tn lab αk op nm med(times[nm]) med(rr) lo hi
        end
        flush(stdout)
      end
    end
  end
end

# ---------------------------------------------------------------------------------------
# time: paired, alternating, randomized order; A/A controls K2 and K3
# ---------------------------------------------------------------------------------------
function time_config(c, s, tn)
  pr = make_prob(c, ttype(tn))
  runs = Dict{String, Any}(); resets = Dict{String, Any}()
  for nm in VARIANT_NAMES
    inst = make_inst(nm, VMODS[nm], pr, s)
    runs[nm] = make_run(inst); resets[nm] = make_reset(inst)
  end
  for nm in VARIANT_NAMES, _ in 1:3
    resets[nm](); runs[nm]()
  end
  tk = med([(resets["K"](); t0 = time_ns(); runs["K"](); Float64(time_ns() - t0)) for _ in 1:5])
  inner = clamp(ceil(Int, TARGET_NS / tk), 1, 5000)
  times = Dict(nm => Float64[] for nm in VARIANT_NAMES)
  rng = Xoshiro(20261113 + hash((c, s, tn)) % 1000)
  for _ in 1:ROUNDS
    GC.gc(false)
    for nm in shuffle(rng, VARIANT_NAMES)
      GC.enable(false)
      push!(times[nm], timebatch(runs[nm], resets[nm], inner))
      GC.enable(true)
    end
  end
  (times, inner, tk)
end
function time_mode()
  println("\n==== time mode (rounds=$ROUNDS, target batch $(TARGET_NS/1e6) ms) ====")
  t_start = time()
  for (c, s, tn) in combos()
    if time() - t_start > BUDGET_S
      println("BUDGET: skipping $c $s $tn"); emit("time", c, s, tn, "-", "skipped_budget", "", true)
      continue
    end
    times, inner, tk = time_config(c, s, tn)
    K = times["K"]
    println("--- $c $s $tn: inner=$inner, K solve ≈ $(round(tk / 1e3; digits = 1)) µs")
    for nm in VARIANT_NAMES
      t = times[nm]
      rr = t ./ K
      lo, hi = nm == "K" ? (1.0, 1.0) : boot_ci(rr)
      emit("time", c, s, tn, nm, "inner", "", inner)
      emit("time", c, s, tn, nm, "min_ns", "", minimum(t))
      emit("time", c, s, tn, nm, "median_ns", "", med(t))
      emit("time", c, s, tn, nm, "ratio_median_paired", "", med(rr))
      emit("time", c, s, tn, nm, "ratio_ci_lo", "", lo)
      emit("time", c, s, tn, nm, "ratio_ci_hi", "", hi)
      for (r, v) in enumerate(t)
        emit("time", c, s, tn, nm, "round_ns", r, v)
      end
      @printf "  %-3s min %10.1f µs  median %10.1f µs  paired-median ratio %.4f [%.4f, %.4f]\n" nm minimum(t) / 1e3 med(t) / 1e3 med(rr) lo hi
    end
    flush(stdout)
  end
end

"count" in MODES && count_mode()
"diag" in MODES && diag_mode()
"alloc" in MODES && alloc_mode()
"types" in MODES && types_mode()
"kern" in MODES && kern_mode()
"time" in MODES && time_mode()
println("\nBEGIN CSV")
println("kind,julia,case,solver,T,variant,metric,idx,value")
foreach(println, CSV)
println("END CSV")
@printf "total script time %.1f s\n" time() - T0_SCRIPT
