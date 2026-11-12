# Accuracy of the QuadraticFunction proximal operator against an independent BigFloat reference,
# baseline versus candidate.
#
#   julia --project=. docs/audit/numerical-kernels-file/spectral_accuracy.jl BASELINE_TREE CANDIDATE_TREE
#
# The reference solves (I + γA) z = x - γb in BigFloat with the stored A (no eigendecomposition). The
# error of a result y is ‖y - z‖ / (‖x‖ + γ‖b‖)/(1 + γλmin), the normwise error relative to the size
# of the operands the operator contracts, in units of eps(T): an eigendecomposition-based solver is
# accurate to a modest multiple of n eps in this measure. Each case is classified by whether the
# reference is representable and whether each result is finite and accurate (error <= 64 n).
# Results that differ between the trees are compared case by case, normwise and coordinate by
# coordinate (in eps units of the componentwise bound |Q| D |Q|'(|x| + γ|b|)), and every remaining
# inaccurate result is attributed: within n subnormal spacings of the reference, the BigFloat value of
# the factorization the function holds (an error of the eigendecomposition), or neither. Two further
# studies are coordinate-wise in ulps: diagonal A (an exact eigenbasis) over the whole exponent range,
# and the rank-one 2×2 forms whose zero eigenvalue LAPACK returns exactly, beside an eigenvalue whose
# product with γ overflows (projected coefficients of subnormal data that γ restores).
using LinearAlgebra, SparseArrays, Random, Printf

const TREES = abspath.(ARGS[1:2])
module Baseline end
module Candidate end
Base.include(Baseline, joinpath(TREES[1], "src", "ProximalBase.jl"))
Base.include(Candidate, joinpath(TREES[2], "src", "ProximalBase.jl"))
const PBS = (Baseline.ProximalBase, Candidate.ProximalBase)
setprecision(BigFloat, 1500)

signif(rng, T, e) = (rand(rng, Bool) ? -1 : 1) * ldexp(T(0.5) + T(rand(rng)) / 2, e)

function case(rng, ::Type{T}) where {T}
  E = exponent(floatmax(T))
  n = rand(rng, 1:5)
  eA = rand(rng, (0, E - 4, E ÷ 2, -E ÷ 2, rand(rng, -E÷2:E-4)))
  Bm = T[signif(rng, T, 0) for _ in 1:n, _ in 1:n]
  A = rand(rng) < 0.3 ? Matrix(Diagonal(T[abs(signif(rng, T, eA - rand(rng, 0:20))) for _ in 1:n])) :
                        Matrix(Symmetric(ldexp.(Bm' * Bm, eA)))
  x = T[signif(rng, T, rand(rng, (0, E - 2, E ÷ 2, rand(rng, -E:E-2)))) for _ in 1:n]
  b = rand(rng) < 0.3 ? nothing : T[signif(rng, T, rand(rng, (0, E - 2, E ÷ 2, rand(rng, -E:E-2)))) for _ in 1:n]
  γ = ldexp(one(T), rand(rng, (0, E - 2, E ÷ 2, -E ÷ 2, rand(rng, -E:E-2))))
  A, b, x, γ
end

# The coordinate-wise error of y, max over i of (|yᵢ - zᵢ| - n δ)⁺ / (eps cᵢ) with c = |Q| D |Q|' (|x| + γ|b|),
# D = Diagonal(1 ./ (1 .+ γΛ)): the componentwise bound of an eigendecomposition solver (plus n subnormal
# spacings), with the factorization the function holds; it is used only as the unit, the reference is z.
function coordinate_error(y, z, Q, Λ, x, b, γ, ::Type{T}) where {T}
  Qa = abs.(big.(Q))
  c = Qa * ((transpose(Qa) * (abs.(big.(x)) .+ (b === nothing ? 0 : big(γ) .* abs.(big.(b))))) ./ (1 .+ big(γ) .* big.(Λ)))
  d = max.(abs.(big.(y) .- z) .- length(x) * big(nextfloat(zero(T))), 0)
  maximum(i -> iszero(d[i]) ? 0.0 : iszero(c[i]) ? Inf : Float64(d[i] / (big(eps(T)) * c[i])), eachindex(d))
end

for T in (Float64, Float32)
  rng = Xoshiro(1112)
  counts = Dict{String, Int}()
  errs = ([Float64[] for _ in 1:2]...,)
  cerrs = ([Float64[] for _ in 1:2]...,)
  changed = better = worse = cbetter = cworse = 0
  worst_increase = worst_cincrease = 0.0
  subnormal_attributed = eig_attributed = operator_attributed = cw_eig = cw_other = 0
  N = 6000
  for _ in 1:N
    A, b, x, γ = case(rng, T)
    n = length(x)
    all(isfinite, A) || continue
    Ab, xb, γb = big.(A), big.(x), big(γ)
    rhs = b === nothing ? xb : xb .- γb .* big.(b)
    z = (I + γb * Ab) \ rhs
    λmin = max(minimum(eigvals(Symmetric(Float64.(A ./ maximum(abs, A))))) * Float64(maximum(abs, A)), 0.0)
    scale = (norm(xb) + (b === nothing ? 0 : γb * norm(big.(b)))) / (1 + γb * big(λmin))
    zT = T.(z)
    representable = all(isfinite, zT)
    fs = map(PB -> PB.QuadraticFunction(A, b), PBS)
    ys = map(PBS, fs) do PB, f
      try PB.prox(f, x, γ) catch; fill(T(NaN), n) end
    end
    nerr(y) = Float64(norm(big.(y) .- z) / scale / big(eps(T)))
    status = map(ys) do y
      all(isfinite, y) || return "non-finite"
      iszero(scale) && return all(iszero, y) ? "accurate" : "inaccurate"
      nerr(y) <= 64n ? "accurate" : "inaccurate"
    end
    key = (representable ? "representable" : "beyond range") * ": baseline $(status[1]), candidate $(status[2])"
    counts[key] = get(counts, key, 0) + 1
    (representable && !iszero(scale)) || continue
    Q, Λ = fs[2].Q, fs[2].Λ
    if all(s -> s == "accurate", status)
      for k in 1:2
        push!(errs[k], nerr(ys[k]))
        push!(cerrs[k], coordinate_error(ys[k], z, Q, Λ, x, b, γ, T))
      end
      # a coordinate beyond the componentwise bound: is it the BigFloat evaluation of the factorization held?
      if cerrs[2][end] > 64n
        Qb = big.(Q)
        r = Qb * ((transpose(Qb) * rhs) ./ (1 .+ γb .* big.(Λ)))
        coordinate_error(ys[2], r, Q, Λ, x, b, γ, T) <= 64n ? (cw_eig += 1) : (cw_other += 1)
      end
    end
    # every case whose result changed between the trees, compared one by one (not in aggregate)
    if !isequal(ys[1], ys[2]) && all(y -> all(isfinite, y), ys)
      changed += 1
      e1, e2 = nerr(ys[1]), nerr(ys[2])
      c1, c2 = (coordinate_error(y, z, Q, Λ, x, b, γ, T) for y in ys)
      better += e2 < e1; worse += e2 > e1; cbetter += c2 < c1; cworse += c2 > c1
      worst_increase = max(worst_increase, e2 - e1); worst_cincrease = max(worst_cincrease, c2 - c1)
    end
    # a remaining inaccurate candidate: is every coordinate within n subnormal spacings of the reference (a point
    # in or below the subnormal range, which the normwise measure does not allow for); does it reproduce the
    # BigFloat evaluation of the factorization that it holds (the error is then that of the eigendecomposition);
    # or neither (the error is the operator's)?
    if status[2] == "inaccurate"
      Qb = big.(Q)
      r = Qb * ((transpose(Qb) * rhs) ./ (1 .+ γb .* big.(Λ)))
      if maximum(abs.(big.(ys[2]) .- z)) <= n * big(nextfloat(zero(T)))
        subnormal_attributed += 1
      elseif norm(big.(ys[2]) .- r) / scale / big(eps(T)) <= 64n
        eig_attributed += 1
      else
        operator_attributed += 1
      end
    end
  end
  println("\n### $T ($N cases)\n\n| reference | baseline | candidate | cases |\n|---|---|---|---:|")
  for (k, v) in sort(collect(counts))
    r, rest = split(k, ": ")
    bs, cs = split(rest, ", ")
    println("| $r | $(replace(bs, "baseline " => "")) | $(replace(cs, "candidate " => "")) | $v |")
  end
  @printf("\nWhere both are accurate: max error %.2f vs %.2f n·eps-units (baseline vs candidate), mean %.3f vs %.3f\n",
          maximum(errs[1]), maximum(errs[2]), sum(errs[1]) / length(errs[1]), sum(errs[2]) / length(errs[2]))
  @printf("Coordinate-wise, in eps units of the componentwise bound: max %.2f vs %.2f, mean %.3f vs %.3f\n",
          maximum(cerrs[1]), maximum(cerrs[2]), sum(cerrs[1]) / length(cerrs[1]), sum(cerrs[2]) / length(cerrs[2]))
  @printf("Accurate candidate results with a coordinate beyond 64n: %d reproduce the factorization they hold coordinate by coordinate (eigendecomposition error), %d do not\n",
          cw_eig, cw_other)
  @printf("Finite results that differ between the trees: %d; normwise better/worse %d/%d (largest increase %.3f n·eps-units); coordinate-wise better/worse %d/%d (largest increase %.3f)\n",
          changed, better, worse, worst_increase, cbetter, cworse, worst_cincrease)
  @printf("Inaccurate candidate results: %d within n subnormal spacings (a point in or below the subnormal range), %d reproduce the BigFloat evaluation of the factorization they hold (eigendecomposition error), %d neither\n",
          subnormal_attributed, eig_attributed, operator_attributed)
end

# Coordinate-wise accuracy where the eigenbasis is exact (a diagonal A: Q is a permutation), with
# eigenvalues, data and steps spread over the whole exponent range, so that coordinates of very
# different sizes share one call (the normwise measure above would hide a lost small coordinate).
# Each coordinate is compared with its own BigFloat value Q ((Q'(x - γb)) ./ (1 .+ γΛ)), in ulps of the
# correctly rounded reference (subnormal spacing included; a reference beyond the range must be ±Inf),
# with the factorization (Λ, Q) that the function computed: LAPACK returns an eigenvalue below about
# eps ‖A‖ as 0 when the spectrum spans more than the range (it rescales A), an error of the
# eigendecomposition that both trees share and that is counted separately.
ulps(a::T, r::BigFloat) where {T} = (t = T(r); isinf(t) ? (a == t ? 0.0 : Inf) : Float64(abs(big(a) - r) / big(eps(max(abs(t), nextfloat(zero(T)))))))
for T in (Float64, Float32), family in ("whole range", "ends of the range")
  rng = Xoshiro(1113)
  E = exponent(floatmax(T))
  es = exponent(nextfloat(zero(T)))
  # "ends of the range": data entries near floatmax beside subnormal ones in the same call
  dataexp() = family == "whole range" ? rand(rng, -E:E-1) : rand(rng, (E - 1, E - 1, E - 2, es, es + 1, es + 3, rand(rng, -E:E-1)))
  worst = ([0.0, 0.0]...,)
  bad = [0, 0]; recovered = 0; lost = 0; skipped = 0; N = 6000
  for _ in 1:N
    n = rand(rng, 1:6)
    λ = T[rand(rng) < 0.15 ? zero(T) : abs(signif(rng, T, rand(rng, -E:E-2))) for _ in 1:n]
    x = T[rand(rng) < 0.1 ? zero(T) : signif(rng, T, dataexp()) for _ in 1:n]
    b = rand(rng) < 0.4 ? nothing : T[signif(rng, T, dataexp()) for _ in 1:n]
    γ = ldexp(one(T), rand(rng, -E:E-2))
    fs = map(PB -> PB.QuadraticFunction(Diagonal(λ), b), PBS)
    ys = map((PB, f) -> PB.prox(f, x, γ), PBS, fs)
    Q, Λ = big.(fs[2].Q), big.(fs[2].Λ)
    @assert fs[1].Λ == fs[2].Λ && fs[1].Q == fs[2].Q
    # a repeated eigenvalue may come with any orthonormal basis of its eigenspace: only a signed
    # permutation makes every coordinate exact
    all(q -> q == 0 || abs(q) == 1, Q) || (skipped += 1; continue)
    lost += sort(λ) != fs[2].Λ
    r = Q * ((transpose(Q) * (big.(x) .- (b === nothing ? 0 : big(γ) .* big.(b)))) ./ (1 .+ big(γ) .* Λ))
    recovered += ys[1] != ys[2]
    for k in 1:2
      e = maximum(ulps(ys[k][i], r[i]) for i in 1:n)
      worst = Base.setindex(worst, max(worst[k], e), k)
      bad[k] += e > 4
    end
  end
  @printf("\n%s diagonal, %s, coordinate-wise (%d cases, %d skipped for a basis that is not a signed permutation, %d differ between the trees, %d with an eigenvalue that LAPACK returned inexactly): worst coordinate %.2f vs %.2f ulp; cases with a coordinate beyond 4 ulp: %d vs %d (baseline vs candidate)\n",
          T, family, N, skipped, recovered, lost, worst[1], worst[2], bad[1], bad[2])
end

# Coordinate-wise accuracy where a projected coefficient underflows before γ restores it: the rank-one 2×2
# forms 2^p M (dense and sparse), whose zero eigenvalue LAPACK returns exactly, with a second eigenvalue whose
# product with γ overflows (the recomputation), and b, x with subnormal integer multiples of δ (projections
# that underflow and cancel) or entries near floatmax beside subnormal ones. Each coordinate against the BigFloat
# solution of (I + γA) z = x - γb with the stored A, in ulps of the correctly rounded reference; a case whose
# zero eigenvalue LAPACK did not return exactly is skipped (a limit of the eigendecomposition). γλ reaches
# 2^(3emax/2), so the reference needs more than the 1500 bits above to keep the 1 of I + γA.
setprecision(BigFloat, 2600) do
for T in (Float64, Float32)
  rng = Xoshiro(1114)
  E = exponent(floatmax(T))
  δ = nextfloat(zero(T))
  N = 4000
  worst = [0.0, 0.0]; cworst = [0.0, 0.0]; total = [0.0, 0.0]; coords = 0
  bad = [0, 0]; corrected = regressed = worsened = differ = skipped = 0
  for _ in 1:N
    M = rand(rng, (T[1 -2; -2 4], T[4 -2; -2 1], T[1 2; 2 4], T[4 6; 6 9]))
    p = rand(rng, E ÷ 4:E ÷ 2)
    A = ldexp.(M, p)
    A = rand(rng, Bool) ? A : sparse(A)
    γ = ldexp(one(T), rand(rng, E - p - 1:E - 1))
    sub() = T(rand(rng, -9:9)) * δ
    mix() = rand(rng, Bool) ? signif(rng, T, E - 2) : rand(rng, Bool) ? sub() : signif(rng, T, rand(rng, -E:0))
    b = rand(rng) < 0.2 ? nothing : T[rand(rng, Bool) ? sub() : mix() for _ in 1:2]
    x = rand(rng) < 0.4 ? zeros(T, 2) : T[rand(rng, Bool) ? sub() : mix() for _ in 1:2]
    fs = map(PB -> PB.QuadraticFunction(A, b), PBS)
    ys = map((PB, f) -> PB.prox(f, x, γ), PBS, fs)
    @assert fs[1].Λ == fs[2].Λ
    iszero(minimum(abs, fs[2].Λ)) || (skipped += 1; continue)
    z = (Matrix{BigFloat}(I, 2, 2) + big(γ) * big.(Matrix(A))) \ (big.(x) .- (b === nothing ? 0 : big(γ) .* big.(b)))
    all(isfinite, T.(z)) || (skipped += 1; continue)
    differ += !isequal(ys[1], ys[2])
    e = map(y -> maximum(ulps(y[i], z[i]) for i in 1:2), ys)
    c = map(y -> coordinate_error(y, z, fs[2].Q, fs[2].Λ, x, b, γ, T), ys)
    for k in 1:2
      worst[k] = max(worst[k], e[k]); cworst[k] = max(cworst[k], c[k]); bad[k] += c[k] > 128
      total[k] += sum(ulps(ys[k][i], z[i]) for i in 1:2)
    end
    coords += 2
    corrected += c[1] > 128 && c[2] <= 128; regressed += c[1] <= 128 && c[2] > 128
    worsened += isfinite(c[1]) && c[2] > c[1]
  end
  @printf("\n%s rank-one 2×2, γλ overflowing, coordinate-wise (%d cases, %d skipped for an inexact zero eigenvalue or a point beyond the range, %d differ between the trees): worst coordinate %.4g vs %.4g ulp, mean %.3g vs %.3g ulp; worst %.4g vs %.4g eps units of the componentwise bound, cases beyond 64n: %d vs %d; corrected %d, regressed %d, componentwise error increased in %d (baseline vs candidate)\n",
          T, N, skipped, differ, worst[1], worst[2], total[1] / coords, total[2] / coords, cworst[1], cworst[2], bad[1], bad[2], corrected, regressed, worsened)
end
end
