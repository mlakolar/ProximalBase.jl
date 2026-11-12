# Accuracy of the Euclidean-norm recovery against an independent BigFloat reference, baseline
# (hypot accumulation) versus candidate (squared residuals in power-of-two units).
#
#   julia --project=. docs/audit/numerical-kernels-file/norm_accuracy.jl BASELINE_TREE CANDIDATE_TREE
#
# For each case family it reports, per tree, the largest error and the mean error in units of the
# last place of the correctly rounded reference (subnormal spacing included), the share of
# correctly rounded results, and how many results differ bitwise between the trees.
using LinearAlgebra, Random, Printf

const TREES = abspath.(ARGS[1:2])
module Baseline end
module Candidate end
Base.include(Baseline, joinpath(TREES[1], "src", "ProximalBase.jl"))
Base.include(Candidate, joinpath(TREES[2], "src", "ProximalBase.jl"))
const PBS = (Baseline.ProximalBase, Candidate.ProximalBase)
setprecision(BigFloat, 2000)

signif(rng, T, e) = (rand(rng, Bool) ? -1 : 1) * ldexp(T(0.5) + T(rand(rng)) / 2, e)
ulps(a::T, r::BigFloat) where {T} = (t = T(r); isinf(t) ? (a == t ? 0.0 : Inf) : Float64(abs(big(a) - r) / big(eps(max(abs(t), nextfloat(zero(T)))))))

function sym(PB, T, vals, p)
  S = PB.SymmetricSparseIterate(T, p)
  k = 0
  for c in 1:p, r in c:p
    k += 1
    k <= length(vals) && (S[r, c] = vals[k])
  end
  S
end
symref(S, p) = sqrt(sum(big(S[r, c])^2 for r in 1:p, c in 1:p; init = big(0.0)))

function families(rng, ::Type{T}) where {T}
  E = exponent(floatmax(T)); e0 = exponent(floatmin(T)); es = exponent(nextfloat(zero(T)))
  (("ordinary", () -> rand(rng, -20:20)),
   ("overflowing squares", () -> rand(rng, E÷2+1:E-4)),
   ("underflowing squares", () -> rand(rng, e0÷2-20:e0÷2-1)),
   ("subnormal entries", () -> rand(rng, es+3:e0-1)),
   ("mixed: huge and tiny", () -> rand(rng, (E-6, E÷2+2, 0, e0÷2, e0+2))))
end

function run(::Type{T}) where {T}
  rng = Xoshiro(1111)
  println("\n### $T\n")
  println("| Operation | Inputs | baseline max ulp | baseline mean ulp | baseline correctly rounded | candidate max ulp | candidate mean ulp | candidate correctly rounded | bitwise differences |")
  println("|---|---|---:|---:|---:|---:|---:|---:|---:|")
  for (name, ex) in families(rng, T), op in (:norm_diff, :symmetric_norm)
    errs = ([Float64[] for _ in 1:2]...,)
    diff = 0
    N = 4000
    for _ in 1:N
      n = rand(rng, 1:40)
      if op === :norm_diff
        A = T[signif(rng, T, ex()) for _ in 1:n]; B = T[signif(rng, T, ex()) for _ in 1:n]
        rand(rng) < 0.2 && (B .= nextfloat.(A, rand(rng, -3:3)))       # cancellation
        r = sqrt(sum((big.(A) .- big.(B)) .^ 2))
        res = map(PB -> PB.norm_diff(A, B), PBS)
      else
        p = rand(rng, 1:8); vals = T[signif(rng, T, ex()) for _ in 1:rand(rng, 1:p*(p+1)÷2)]
        Ss = map(PB -> sym(PB, T, vals, p), PBS)
        r = symref(Ss[1], p)
        res = map(S -> norm(S), Ss)
      end
      for t in 1:2
        push!(errs[t], ulps(res[t], r))
      end
      diff += res[1] !== res[2]
    end
    cells = [(@sprintf("%.2f", maximum(e)), @sprintf("%.3f", sum(e) / length(e)), @sprintf("%.1f%%", 100count(<=(0.5), e) / length(e))) for e in errs]
    println("| `$op` | $name | $(join(cells[1], " | ")) | $(join(cells[2], " | ")) | $diff / $N |")
  end
end

# genuine NaN/Inf, exact and signed zeros: the trees must agree bit for bit
function specials()
  bad = 0
  for T in (Float64, Float32), a in (T(NaN), T(Inf), -T(Inf), zero(T), -zero(T), floatmax(T), nextfloat(zero(T))),
      b in (T(NaN), T(Inf), -T(Inf), zero(T), -zero(T), -floatmax(T), T(1))
    A, B = T[a, 1, 0], T[b, 1, 0]
    x = map(PB -> PB.norm_diff(A, B), PBS)
    y = map(PB -> norm(sym(PB, T, T[a, b, 0], 2)), PBS)
    bad += !isequal(x...) + !isequal(y...)
  end
  println("\nSpecial values (NaN, ±Inf, ±0, floatmax, the smallest subnormal): $bad differences")
end

run(Float64); run(Float32); specials()
