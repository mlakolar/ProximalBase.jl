# PB #911 numerical gate (disposable). Every variant's solver-facing oracle and every solve are
# compared with independent 256-bit BigFloat references computed from the exact bits of the inputs
# (never with K or another solver).
#
# PRE-REGISTERED CRITERIA (fixed before the first run; T is the data type, u = eps(T)/2, n rows):
#  Oracle values (value at an image, value from value+gradient, value at a residual), reference
#  v* = α Σ (zᵢ - Yᵢ)²/2 (resp. α Σ rᵢ²/2) in BigFloat, α exactly as supplied:
#   PASS   v* is 0 and v == 0; or v* in [floatmin(T), floatmax(T)] and |v - v*| ≤ τ|v*|,
#          τ = (n + 8)·eps(T); or 0 < v* < floatmin(T) and |v - v*| ≤ τ|v*| + 2·nextfloat(zero(T));
#          or v* > floatmax(T) and v == Inf; or an input entry is NaN/±Inf and v is NaN/Inf.
#   MISS   v* is a finite representable number and v is ±Inf or NaN (no recovery: a weaker
#          contract, acceptable only where documented, never on ordinary data).
#   FAIL   v finite and outside the PASS bound (a silent wrong value: rejects the design).
#  Coefficient gradient g* = α Xᵀ(z - Y), componentwise bound
#          |gⱼ - g*ⱼ| ≤ τ·(α |X|ᵀ|z - Y|)ⱼ + τ·nextfloat(zero(T)); classes as above
#          (entries with |g*ⱼ| > floatmax(T) must be ±Inf; MISS for a non-finite entry where g*ⱼ is
#          representable; FAIL for a finite entry outside the bound).
#  Ordinary inputs (cases "ordinary", "cancellation", "nearfit", "exactfit", "zerocol") must PASS
#  for every quantity, type and scale, or the design is rejected.
#  Solves: relative KKT residual κ (256-bit, at the returned x, against the supplied α):
#          meets when κ ≤ 1e-6 (Float64) or κ ≤ 1e-3 (Float32); a variant that misses where K meets
#          is investigated and either explained (conditioning, a documented refusal) or a defect.
#          Reported objective: relative error of res.minimum against the 256-bit objective at the
#          returned x, recorded (no pass/fail for CD, whose residual is incrementally updated).
include(joinpath(@__DIR__, "envsetup.jl"))
using Random, Printf
using ProximalBase: LeastSquaresLoss, NormL1, SparseIterate
print_provenance()
setprecision(BigFloat, 256)
const VARIANTS = String.(split(argopt("VARIANTS", "K,B1,B2,B3,H,H1,C"), ','))
const PARTS = String.(split(argopt("PARTS", "oracle,solve"), ','))
function loadvariant(name)
  dst = joinpath(mktempdir(), "src")
  cp(joinpath(@__DIR__, "variants", name, "src"), dst)
  w = Module(Symbol(name))
  Base.include(w, joinpath(dst, "ProximalSolvers.jl"))
  Base.invokelatest(getfield, w, :ProximalSolvers)
end
const V = Dict(nm => loadvariant(nm) for nm in VARIANTS)
const CSV = String[]
emit(xs...) = push!(CSV, join(replace.(string.(xs), ',' => ';'), ","))

# ---------------------------------------------------------------------------------------
# oracle level
# ---------------------------------------------------------------------------------------
bigv(α, a, b) = big(α) * sum(abs2, big.(a) .- big.(b)) / 2
function classify_value(v, ref, ::Type{T}, n, nonfinite_input) where {T}
  τ = (n + 8) * eps(T)
  nonfinite_input && return (isnan(v) || isinf(v)) ? "PASS" : "FAIL"
  isfinite(ref) || return "FAIL"
  if iszero(ref)
    return iszero(v) ? "PASS" : (isfinite(v) ? "FAIL" : "MISS")
  elseif ref > floatmax(T)
    return v == Inf ? "PASS" : (isfinite(v) ? "FAIL" : "MISS")
  end
  isfinite(v) || return "MISS"
  err = abs(big(v) - ref)
  bound = ref >= floatmin(T) ? τ * ref : τ * ref + 2 * big(nextfloat(zero(T)))
  err <= bound ? "PASS" : "FAIL"
end
# underflow_abs = nothing: the pre-registered bound; underflow_abs = c: the same relative term plus
# c·nextfloat(zero(T)) absolute (POST-HOC CORRECTION, reported separately: the registered absolute
# term τ·nextfloat(0) is below the half-unit rounding of a single subnormal result)
function classify_grad(g, X, z, Y, α, ::Type{T}, nonfinite_input; underflow_abs = nothing) where {T}
  τ = (length(Y) + 8) * eps(T)
  nonfinite_input && return ("PASS-NA", 0.0)
  Xb = big.(X); r = big.(z) .- big.(Y)
  ref = big(α) .* (transpose(Xb) * r)
  ab = underflow_abs === nothing ? τ * big(nextfloat(zero(T))) : underflow_abs * big(nextfloat(zero(T)))
  bnd = τ .* big(α) .* (transpose(abs.(Xb)) * abs.(r)) .+ ab
  worst = "PASS"; maxrel = 0.0
  for j in eachindex(g)
    if abs(ref[j]) > floatmax(T)
      c = isinf(g[j]) && sign(g[j]) == sign(ref[j]) ? "PASS" : (isfinite(g[j]) ? "FAIL" : "MISS")
    elseif !isfinite(g[j])
      c = "MISS"
    else
      e = abs(big(g[j]) - ref[j])
      c = e <= bnd[j] ? "PASS" : "FAIL"
      maxrel = max(maxrel, Float64(e / max(bnd[j], big(floatmin(Float64)))))
    end
    worst = c == "FAIL" ? "FAIL" : (c == "MISS" && worst != "FAIL") ? "MISS" : worst
  end
  worst, maxrel
end
# images, responses and designs for the oracle cases (z need not be X β: the oracle takes any z)
function oracle_case(name, ::Type{T}, rng) where {T}
  n, p = 64, 6
  X = randn(rng, T, n, p); Y = randn(rng, T, n)
  z = X * randn(rng, T, p)
  if name == "cancellation"
    Y = T(1e3) .+ randn(rng, T, n); z = Y .+ T(1e-2) .* randn(rng, T, n)
  elseif name == "nearfit"
    z = Y .* (one(T) .+ 4eps(T) .* randn(rng, T, n))
  elseif name == "exactfit"
    z = copy(Y)
  elseif name == "zerocol"
    X[:, 3] .= 0
  elseif name == "underflow"
    t = T === Float64 ? 1e-160 : 1f-22
    Y = T(t) .* randn(rng, T, n); z = T(t) .* randn(rng, T, n)
  elseif name == "overflow"       # finite residuals whose squares overflow
    t = T === Float64 ? 1e160 : 1f20
    Y = T(t) .* randn(rng, T, n); z = T(t) .* randn(rng, T, n)
  elseif name == "diffoverflow"   # z - Y itself overflows in two entries
    z[1] = floatmax(T) * T(0.75); Y[1] = -floatmax(T) * T(0.75)
    z[2] = -floatmax(T) * T(0.6); Y[2] = floatmax(T) * T(0.6)
  elseif name == "inf"
    z[5] = T(Inf)
  elseif name == "nan"
    z[7] = T(NaN)
  end
  X, Y, z
end
const ORACLE_CASES = ("ordinary", "cancellation", "nearfit", "exactfit", "zerocol", "underflow", "overflow",
                      "diffoverflow", "inf", "nan")
# scales: as solve! receives them; a Float64 scale with Float32 data is kept as supplied
scales(::Type{Float64}, n) = (("raw", true), ("1/n", 1.0 / n), ("1e200", 1e200), ("1e-200", 1e-200),
                              ("2^-1030", 2.0^-1030), ("1e-300", 1e-300))
scales(::Type{Float32}, n) = (("raw", true), ("1/n", 1f0 / n), ("1e30", 1f30), ("1e-30", 1f-30),
                              ("f64:1/n", 1.0 / n), ("f64:1e-39", 1e-39), ("f64:1e-44", 1e-44), ("f64:1e-60", 1e-60))
function oracle_part()
  println("\n==== oracle level ====")
  tally = Dict{Tuple{String, String}, Dict{String, Int}}()
  for T in (Float64, Float32), cname in ORACLE_CASES, (sname, α) in scales(T, 64)
    X, Y, z = oracle_case(cname, T, Xoshiro(911))
    n = length(Y)
    r = Y .- z                                   # the residual as a CD cache holds it
    nf = !all(isfinite, z)
    vref = bigv(α, z, Y); rref = bigv(α, r, zero(T))
    base = LeastSquaresLoss(Y, X)
    f = α === true ? base : α * base
    for nm in VARIANTS
      M = V[nm]
      m = M._image_model(f)
      G = zeros(T, size(X, 2)); d = similar(Y)
      lm = M.cdmodel(f, SparseIterate(T, size(X, 2))); copyto!(lm.r, r)
      res = Dict{String, Any}()
      try
        v1 = convert(T, M._image_value(m, z))
        v2 = convert(T, M._image_value_and_gradient!(G, m, z, d))
        v3 = convert(T, M.loss_value(lm, SparseIterate(T, size(X, 2))))
        c1 = classify_value(v1, vref, T, n, nf); c2 = classify_value(v2, vref, T, n, nf)
        c3 = classify_value(v3, rref, T, n, !all(isfinite, r))
        cg, gm = classify_grad(G, X, z, Y, α, T, nf)
        cgu, gmu = classify_grad(G, X, z, Y, α, T, nf; underflow_abs = 2)
        for (q, c, v) in (("value", c1, v1), ("vg_value", c2, v2), ("residual_value", c3, v3), ("gradient", cg, gm),
                          ("gradient_uf", cgu, gmu))
          emit("oracle", VERSION, T, cname, sname, nm, q, c, repr(v))
          k = (nm, q); t = get!(tally, k, Dict{String, Int}()); t[c] = get(t, c, 0) + 1
        end
        @printf "%-7s %-12s %-9s %-3s value=%-5s vg=%-5s res=%-5s grad=%-7s | v=%-13s ref=%-13s vg=%-13s res=%-13s refr=%s\n" T cname sname nm c1 c2 c3 cg repr(v1) string(Float64(vref)) repr(v2) repr(v3) string(Float64(rref))
      catch e
        emit("oracle", VERSION, T, cname, sname, nm, "error", "ERROR", sprint(showerror, e))
        @printf "%-7s %-12s %-9s %-3s ERROR %s\n" T cname sname nm sprint(showerror, e)
      end
    end
  end
  println("\n---- oracle tally (variant, quantity) => class counts ----")
  for k in sort(collect(keys(tally)))
    println(k, " => ", sort(collect(tally[k])))
  end
end

# ---------------------------------------------------------------------------------------
# solve level
# ---------------------------------------------------------------------------------------
function kkt(X, y, α, λ, x)
  Xb, yb, xb = big.(X), big.(y), big.(x)
  G = big(α) .* (transpose(Xb) * (Xb * xb .- yb))
  v = [xi != 0 ? abs(G[i] + big(λ) * sign(xi)) : max(abs(G[i]) - big(λ), 0) for (i, xi) in enumerate(xb)]
  Float64(maximum(v) / max(big(λ), maximum(abs, big(α) .* (transpose(Xb) * yb))))
end
bigobj(X, y, α, λ, x) = big(α) * sum(abs2, big.(X) * big.(x) .- big.(y)) / 2 + big(λ) * sum(abs, big.(x))
function solve_family(name, ::Type{T}, rng) where {T}
  n, p = 40, 5
  X = randn(rng, T, n, p) * T.([1 0.5 0.3 0 0; 0 1 0.5 0 0; 0 0 1 0.2 0; 0 0 0 1 0.4; 0 0 0 0 1])
  β = T[1, -2, 0.5, 0, 0]
  y = X * β .+ T(0.1) .* randn(rng, T, n)
  α = one(T) / T(n)
  if name == "raw"
    α = true
  elseif name == "nearfit"
    y = X * β .+ T(1e-5) .* randn(rng, T, n)
  elseif name == "exactfit"
    y = X * β
  elseif name == "zerocol"
    X[:, 4] .= 0
  elseif name == "largealpha"
    α = T === Float64 ? 1e100 : 1f30
  elseif name == "smallalpha"
    α = T === Float64 ? 1e-100 : 1f-30
  elseif name == "f64alpha"          # Float32 data, Float64 scale 1/n as supplied
    α = 1.0 / n
  elseif name == "R3"                # Float32 data ×1e18 with α = 1e-39 (inner F32 sum overflows)
    X = X .* T(1e18); y = y .* T(1e18); α = 1e-39
  elseif name == "R4"                # α subnormal in Float32 (D3)
    α = 1e-44
  elseif name == "underflow"         # squares that underflow (R2)
    X = X .* T(T === Float64 ? 1e-160 : 1f-20); y = y .* T(T === Float64 ? 1e-160 : 1f-20)
  end
  X, y, α
end
families(::Type{Float64}) = ("ordinary", "raw", "nearfit", "exactfit", "zerocol", "largealpha", "smallalpha", "underflow")
families(::Type{Float32}) = ("ordinary", "raw", "nearfit", "exactfit", "zerocol", "largealpha", "smallalpha", "f64alpha", "R3", "R4", "underflow")
function solve_one(M, X::Matrix{T}, y, α, λ, solver) where {T}
  base = LeastSquaresLoss(y, X)
  f = α === true ? base : α * base
  x = zeros(T, size(X, 2))
  if solver == "cd"
    r = M.solve!(x, f, NormL1(λ), M.CDSolver(order = :cyclic, momentum = false, tol = T === Float64 ? 1e-13 : 1e-7, maxpasses = 100_000))
    return x, r.minimum, r.converged, r.passes
  end
  meth = solver == "pg" ? M.ProximalGradientDescent() : M.AccProxGradDescent()
  opt = M.ProximalOptions(maxiter = 20_000, ftol = 0, xtol = T === Float64 ? 1e-15 : 1e-7, gradtol = T === Float64 ? 1e-13 : 1e-7)
  r = M.solve!(x, f, NormL1(λ), meth, opt)
  x, r.minimum, r.converged, r.iterations
end
function solve_part()
  println("\n==== solve level ====")
  for T in (Float64, Float32), fam in families(T), lf in (0.0, 0.1), solver in ("cd", "pg", "apg")
    X, y, α = solve_family(fam, T, Xoshiro(20911))
    emit("input_hash", VERSION, T, fam, lf, solver, "-", "hash", string(hash((X, y)); base = 16))
    λ = T(lf * Float64(maximum(abs, big(α) .* (transpose(big.(X)) * big.(y)))))
    meet = T === Float64 ? 1e-6 : 1e-3
    for nm in VARIANTS
      try
        x, minimum, conv, it = solve_one(V[nm], X, y, α, λ, solver)
        κ = kkt(X, y, α, λ, x)
        ob = bigobj(X, y, α, λ, x)
        relobj = isfinite(minimum) ? Float64(abs(big(minimum) - ob) / max(abs(ob), big(floatmin(Float64)))) : Inf
        # POST-HOC (review r1), not pre-registered: the reported objective error relative to the size of
        # the computed terms, S = α (‖|X||x|‖ + ‖y‖)²/2 + λ‖x‖₁ (the magnitude that rounding in a fresh
        # image, an extrapolation or an incrementally updated residual scales with)
        Xb, xb, yb = big.(X), big.(x), big.(y)
        S = big(α) * (sqrt(sum(abs2, abs.(Xb) * abs.(xb))) + sqrt(sum(abs2, yb)))^2 / 2 + big(λ) * sum(abs, xb)
        # divided by S in BigFloat (review r2); an exactly zero S with an exact report gives 0
        errb = isfinite(minimum) ? abs(big(minimum) - ob) : big(Inf)
        scaleobj = iszero(S) ? (iszero(errb) ? 0.0 : Inf) : Float64(errb / S)
        emit("solve", VERSION, T, fam, lf, solver, nm, "scaled_reported_objective_error", scaleobj)
        absobj = isfinite(minimum) ? Float64(abs(big(minimum) - ob)) : Inf
        emit("solve", VERSION, T, fam, lf, solver, nm, "abs_reported_objective_error", absobj)
        # POST-HOC rule (review r1, applied first to this run): the reported objective is truthful when
        # |reported - F(x)| ≤ 4 eps(T) S + 2 nextfloat(zero(T)) (the absolute term: a subnormal total)
        emit("solve", VERSION, T, fam, lf, solver, nm, "reported_objective_ok",
             isfinite(minimum) && big(absobj) <= 4 * big(eps(T)) * S + 2 * big(nextfloat(zero(T))))
        emit("solve", VERSION, T, fam, lf, solver, nm, "kkt", κ)
        emit("solve", VERSION, T, fam, lf, solver, nm, "meets", κ <= meet)
        emit("solve", VERSION, T, fam, lf, solver, nm, "rel_reported_objective", relobj)
        emit("solve", VERSION, T, fam, lf, solver, nm, "iters", it)
        emit("solve", VERSION, T, fam, lf, solver, nm, "converged", conv)
        emit("solve", VERSION, T, fam, lf, solver, nm, "x", join(repr.(x), ' '))
        @printf "%-7s %-10s λ=%-4s %-3s %-3s κ=%.3e %-6s it=%-6d conv=%-5s relobj=%.2e scaled_obj_err=%.2e\n" T fam lf solver nm κ (κ <= meet ? "meets" : "MISSES") it conv relobj scaleobj
      catch e
        msg = sprint(showerror, e)
        emit("solve", VERSION, T, fam, lf, solver, nm, "error", msg)
        @printf "%-7s %-10s λ=%-4s %-3s %-3s threw: %s\n" T fam lf solver nm first(msg, 140)
      end
    end
    flush(stdout)
  end
end
"oracle" in PARTS && oracle_part()
"solve" in PARTS && solve_part()
println("\nBEGIN CSV"); foreach(println, CSV); println("END CSV")
