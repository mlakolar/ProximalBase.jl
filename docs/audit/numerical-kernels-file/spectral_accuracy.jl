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
using LinearAlgebra, Random, Printf

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

for T in (Float64, Float32)
  rng = Xoshiro(1112)
  counts = Dict{String, Int}()
  errs = ([Float64[] for _ in 1:2]...,)
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
    ys = map(PBS) do PB
      f = PB.QuadraticFunction(A, b)
      try PB.prox(f, x, γ) catch; fill(T(NaN), n) end
    end
    status = map(ys) do y
      all(isfinite, y) || return "non-finite"
      iszero(scale) && return all(iszero, y) ? "accurate" : "inaccurate"
      e = Float64(norm(big.(y) .- z) / scale / big(eps(T)))
      e <= 64n ? "accurate" : "inaccurate"
    end
    key = (representable ? "representable" : "beyond range") * ": baseline $(status[1]), candidate $(status[2])"
    counts[key] = get(counts, key, 0) + 1
    if representable && all(s -> s == "accurate", status) && !iszero(scale)
      for k in 1:2
        push!(errs[k], Float64(norm(big.(ys[k]) .- z) / scale / big(eps(T))))
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
for T in (Float64, Float32)
  rng = Xoshiro(1113)
  E = exponent(floatmax(T))
  worst = ([0.0, 0.0]...,)
  bad = [0, 0]; recovered = 0; lost = 0; skipped = 0; N = 6000
  for _ in 1:N
    n = rand(rng, 1:6)
    λ = T[rand(rng) < 0.15 ? zero(T) : abs(signif(rng, T, rand(rng, -E:E-2))) for _ in 1:n]
    x = T[rand(rng) < 0.1 ? zero(T) : signif(rng, T, rand(rng, -E:E-1)) for _ in 1:n]
    b = rand(rng) < 0.4 ? nothing : T[signif(rng, T, rand(rng, -E:E-1)) for _ in 1:n]
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
  @printf("\n%s diagonal, coordinate-wise (%d cases, %d skipped for a basis that is not a signed permutation, %d differ between the trees, %d with an eigenvalue that LAPACK returned inexactly): worst coordinate %.2f vs %.2f ulp; cases with a coordinate beyond 4 ulp: %d vs %d (baseline vs candidate)\n",
          T, N, skipped, recovered, lost, worst[1], worst[2], bad[1], bad[2])
end
