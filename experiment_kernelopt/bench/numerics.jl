# Numerical behavior of the implementations on ordinary and exceptional inputs, against BigFloat references that
# use no package code.
#   KO_TREES="base=<root>;A=<root>;..." julia --project=<env> bench/numerics.jl > notes/numerics_<version>.md
using LinearAlgebra, Random, Printf
module K; include(joinpath(@__DIR__, "..", "kernels", "kernels.jl")); end
import ProximalOperators
const PO = ProximalOperators
function loadtree(name, root)
  m = Core.eval(Main, :(module $(Symbol("Tree_", name)) end))
  Base.include(m, joinpath(root, "src", "ProximalBase.jl"))
  Base.invokelatest(getfield, m, :ProximalBase)   # the binding is defined in a later world than this call
end
specs = [split(s, '=') for s in split(ENV["KO_TREES"], ';') if !isempty(s)]
const TREES = [(String(n), loadtree(n, abspath(String(r)))) for (n, r) in specs]
setprecision(BigFloat, 1024)

# "ok" when the result is the correctly rounded exact value; otherwise the relative error in units of eps(T), or a
# label for a result of the wrong class
function ulps(got::T, exact::BigFloat) where {T}
  isnan(exact) && return isnan(got) ? "ok" : "not NaN ($got)"
  isnan(got) && return "NaN"
  e = T(exact)                                   # the correctly rounded value
  isinf(e) && return got == e ? "ok" : "$(got) (exact rounds to $(e))"
  isinf(got) && return "Inf"
  iszero(e) && return iszero(got) ? (signbit(got) ? "-0" : "ok") : "nonzero $(got)"     # an exact value that rounds to zero
  iszero(got) && return "0 (exact $(@sprintf("%.3g", Float64(exact))))"
  got == e && return "ok"
  d = abs(big(got) - exact) / max(abs(exact), big(floatmin(T)))
  @sprintf("%.3g eps", max(Float64(d / eps(T)), 0.5))
end
worst(labels) = begin
  bad = [l for l in labels if l != "ok"]
  isempty(bad) && return "ok"
  nums = [parse(Float64, split(l)[1]) for l in bad if endswith(l, "eps")]
  other = unique(l for l in bad if !endswith(l, "eps"))
  join(vcat(isempty(nums) ? String[] : [@sprintf("≤ %.3g eps", maximum(nums))], other), "; ")
end
safe(f) = try f() catch e; "throws $(nameof(typeof(e)))" end
function table(title, header, rows)
  println("\n### ", title, "\n")
  println("| ", join(header, " | "), " |"); println("|", repeat("---|", length(header)))
  for r in rows; println("| ", join(r, " | "), " |"); end
end

####################################
# A. soft thresholding on a grid of exceptional operands
####################################
for T in (Float64, Float32)
  vs = T[0, -0.0, nextfloat(zero(T)), -nextfloat(zero(T)), floatmin(T), 1, -1, prevfloat(one(T)), nextfloat(one(T)), 3, -3, floatmax(T), -floatmax(T), Inf, -Inf, NaN]
  cs = T[0, -0.0, nextfloat(zero(T)), floatmin(T), 1, 3, floatmax(T), Inf, NaN]
  ref(v, c) = (isnan(v) || isnan(c)) ? big(NaN) : (isinf(v) && isinf(c)) ? big(NaN) : sign(big(v)) * max(abs(big(v)) - big(c), 0)
  impls = Any[]
  for (n, PB) in TREES
    push!(impls, ("PB $n prox!", (v, c) -> (y = T[0]; PB.prox!(y, PB.NormL1(one(T)), T[v], c); y[1])))
  end
  push!(impls, ("ProximalOperators prox!", (v, c) -> (y = T[0]; PO.prox!(y, PO.NormL1(one(T)), T[v], c); y[1])))
  for st in (:st_maxmin, :st_selmaxmin, :st_clamp, :st_clampsel, :st_clampz, :st_copysign, :st_copysignsel, :st_ifelse, :st_po)
    push!(impls, ("kernel $st", getfield(K, st)))
  end
  rows = Vector{String}[]
  for (name, f) in impls
    nbad = 0; negzero = 0; nanlost = 0; ex = ""
    for v in vs, c in cs
      got = safe(() -> f(v, c)); r = ref(v, c)
      got isa String && (nbad += 1; ex = got; continue)
      if isnan(r)
        isnan(got) || (nanlost += 1; isempty(ex) && (ex = "f($v, $c) = $got, exact NaN"))
      elseif iszero(r)
        iszero(got) ? (signbit(got) && (negzero += 1)) : (nbad += 1; isempty(ex) && (ex = "f($v, $c) = $got, exact 0"))
      else
        got == T(r) || (nbad += 1; isempty(ex) && (ex = "f($v, $c) = $got, exact $(T(r))"))
      end
    end
    push!(rows, [name, string(nbad), string(nanlost), string(negzero), ex])
  end
  table("Soft thresholding, $T: $(length(vs)) values × $(length(cs)) thresholds", ["implementation", "wrong values", "NaN not propagated", "zero results returned as -0", "first difference"], rows)
end

####################################
# B. Euclidean norm (value with level 1)
####################################
for T in (Float64, Float32)
  rng = MersenneTwister(1)
  big1, tiny1 = sqrt(floatmax(T)) * 4, sqrt(floatmin(T)) / 4
  vecs = [("ordinary n=1000", randn(rng, T, 1000)), ("zero vector", zeros(T, 8)), ("entries near sqrt(floatmax)·4 (squares overflow)", fill(big1, 16)),
          ("entries floatmax/4", fill(floatmax(T) / 4, 4)), ("norm overflows (floatmax entries)", fill(floatmax(T), 4)),
          ("entries sqrt(floatmin)/4 (squares underflow)", fill(tiny1, 16)), ("subnormal entries", fill(nextfloat(zero(T)) * 3, 16)),
          ("one huge, many tiny", vcat(big1, fill(tiny1, 15))), ("mixed 1 and tiny", vcat(one(T), fill(tiny1, 15))),
          ("NaN entry", T[1, NaN, 2]), ("Inf entry", T[1, Inf, 2]), ("Inf and NaN", T[Inf, NaN])]
  impls = Any[]
  for (n, PB) in TREES; push!(impls, ("PB $n NormL2", x -> PB.NormL2(one(T))(x))); end
  push!(impls, ("ProximalOperators NormL2", x -> PO.NormL2(one(T))(x)))
  for nm in (:norm_simd, :norm_la, :norm_blas, :norm_dot, :norm_scaled, :norm_blue); push!(impls, ("kernel $nm", getfield(K, nm))); end
  refn(x) = any(isnan, x) ? big(NaN) : any(isinf, x) ? big(Inf) : sqrt(sum(abs2 ∘ big, x))
  rows = [vcat(name, [(g = safe(() -> f(x)); g isa String ? g : ulps(T(g), refn(x))) for (_, x) in vecs]) for (name, f) in impls]
  table("Euclidean norm, $T (error in eps relative to the exact norm; ok = correctly rounded)", vcat("implementation", first.(vecs)), rows)

  # the proximal operator with threshold c = norm/2 (shrink) and 2 norm (zero)
  pvecs = [v for v in vecs if !occursin(r"NaN|Inf|overflows", v[1])]
  pimpls = Any[]
  for (n, PB) in TREES; push!(pimpls, ("PB $n prox!", (x, c) -> PB.prox!(similar(x), PB.NormL2(one(T)), x, c))); end
  push!(pimpls, ("ProximalOperators prox!", (x, c) -> (y = similar(x); PO.prox!(y, PO.NormL2(one(T)), x, c); y)))
  push!(pimpls, ("kernel direct (plain sum of squares)", (x, c) -> (y = similar(x); K.l2prox!(y, x, c); y)))
  push!(pimpls, ("kernel norm-then-broadcast", (x, c) -> (y = similar(x); K.l2prox_norm!(y, x, c); y)))
  function proxerr(y, x, c)
    n = sqrt(sum(abs2 ∘ big, x))
    yb = n > c ? (1 - big(c) / n) .* big.(x) : zero(big.(x))
    worst([ulps(y[i], yb[i]) for i in eachindex(y)])
  end
  rows = Vector{String}[]
  for (name, f) in pimpls
    r = String[name]
    for (_, x) in pvecs
      n = T(sqrt(sum(abs2 ∘ big, x)))
      for c in (iszero(n) ? (one(T),) : (n / 2, min(2n, floatmax(T))))
        g = safe(() -> f(x, c)); push!(r, g isa String ? g : proxerr(g, x, c))
      end
    end
    push!(rows, r)
  end
  hdr = String["implementation"]
  for (nm, x) in pvecs; iszero(sum(abs, big.(x))) ? push!(hdr, nm) : append!(hdr, [nm * ", c = ‖x‖/2", nm * ", c = 2‖x‖"]); end
  table("ℓ2 proximal operator, $T (componentwise error in eps of the exact component)", hdr, rows)
end

####################################
# C. least squares: values at extreme residuals, and the proximal operator on ill-conditioned designs
####################################
for T in (Float64, Float32)
  rng = MersenneTwister(2)
  big1, tiny1 = sqrt(floatmax(T)) * 4, sqrt(floatmin(T)) / 4
  res = [("ordinary n=1000", randn(rng, T, 1000)), ("zero", zeros(T, 8)), ("one square overflows, ‖r‖²/2 finite", T[sqrt(floatmax(T)) * T(1.1)]), ("squares finite, their sum overflows", fill(sqrt(floatmax(T)) * T(0.9), 2)),
         ("squares overflow", fill(big1, 16)), ("squares underflow", fill(tiny1, 16)), ("subnormal", fill(nextfloat(zero(T)) * 3, 16))]
  impls = Any[]
  for (n, PB) in TREES
    isdefined(PB, :value_from_residual) && push!(impls, ("PB $n value_from_residual", r -> PB.value_from_residual(PB.LeastSquaresLoss(zeros(T, length(r)), zeros(T, length(r), 0)), r)))
  end
  for nm in (:halfsumsq_simd, :halfsumsq_base, :halfsumsq_dot, :halfsumsq_nrm2); push!(impls, ("kernel $nm", getfield(K, nm))); end
  rows = [vcat(name, [(g = safe(() -> f(r)); g isa String ? g : ulps(T(g), sum(abs2 ∘ big, r) / 2)) for (_, r) in res]) for (name, f) in impls]
  table("‖r‖²/2, $T", vcat("implementation", first.(res)), rows)

  n, p = 60, 12
  rows = Vector{String}[]
  conds = T === Float64 ? (1e2, 1e6, 1e10, 1e14) : (1f1, 1f3, 1f5, 1f7)
  γs = (T(1), T(1e6))
  hdr = String["implementation"]
  for κ in conds, γ in γs; push!(hdr, "cond $(@sprintf("%.0e", κ)), γ = $(@sprintf("%.0e", γ))"); end
  probs = Any[]
  for κ in conds
    U = Matrix(qr(randn(rng, n, p)).Q)[:, 1:p]; V = Matrix(qr(randn(rng, p, p)).Q)
    X = T.(U * Diagonal(exp10.(range(0, -log10(Float64(κ)), length = p))) * V')
    Y = T.(randn(rng, n)); v = T.(randn(rng, p))
    for γ in γs
      Xb, Yb, vb = big.(X), big.(Y), big.(v)
      push!(probs, (X, Y, v, γ, (Xb' * Xb + I / big(γ)) \ (Xb' * Yb .+ vb ./ big(γ))))
    end
  end
  relerr(β, βb) = @sprintf("%.1e", Float64(norm(big.(β) .- βb) / norm(βb)))
  pimpls = Any[]
  for (nm, PB) in TREES; push!(pimpls, ("PB $nm prox! (thin SVD)", (X, Y, v, γ) -> PB.prox!(zeros(T, length(v)), PB.LeastSquaresLoss(Y, X), v, γ))); end
  push!(pimpls, ("kernel Cholesky of X'X + I/γ", (X, Y, v, γ) -> K.chol_prox!(zeros(T, length(v)), K.CholLS(X, Y), v, γ)))
  push!(pimpls, ("ProximalOperators LeastSquares", (X, Y, v, γ) -> (β = zeros(T, length(v)); PO.prox!(β, PO.LeastSquares(X, Y), v, γ); β)))
  for (name, f) in pimpls
    push!(rows, vcat(name, [(g = safe(() -> f(X, Y, v, γ)); g isa String ? g : relerr(g, βb)) for (X, Y, v, γ, βb) in probs]))
  end
  table("Least-squares proximal operator, $T, $n × $p: relative error ‖β − β*‖/‖β*‖ (eps = $(@sprintf("%.1e", eps(T))))", hdr, rows)
end
