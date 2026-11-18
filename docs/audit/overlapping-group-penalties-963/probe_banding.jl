# Section 5 of docs/design/overlapping-group-penalties-963.md: the nested chain of the convex
# banding penalty of Bien, Bunea and Xiao, its general weights, the weights of Yu and Bien, and
# the exact evaluation by dual block coordinate descent on the band norms.
#
#   julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_banding.jl

include(joinpath(@__DIR__, "harness.jl"))
using ProximalBase

# The chain on K = p - 1 coordinates (band norms): group ℓ = 1:ℓ with weights W[ℓ, 1:ℓ].
general(K) = [m <= l ? sqrt(2l) / (l - m + 1) : 0.0 for l in 1:K, m in 1:K]        # BBX (8), arXiv v1
basic(K) = [m <= l ? sqrt(2l) : 0.0 for l in 1:K, m in 1:K]                         # BBX (7)
control(K) = [m == l ? sqrt(2l) : 0.0 for l in 1:K, m in 1:K]                        # BBX (6)
yubien(K) = [m <= l ? 1 / (l - m + 1)^2 : 0.0 for l in 1:K, m in 1:K]               # Yu and Bien (7)
twovalue(K, a) = [m < l ? a[l] : m == l ? sqrt(2l) : 0.0 for l in 1:K, m in 1:K]    # constant below the diagonal
chain(W) = [l2(1:l, W[l, 1:l]) for l in 1:size(W, 1)]

# --- The matrix problem reduces to the band norms ------------------------------------------

# s_m = {(i, j) : |i - j| = p - m}, both triangles; group g_ℓ = s_1 ∪ … ∪ s_ℓ.
band(p, m) = [LinearIndices((p, p))[i, j] for j in 1:p for i in 1:p if abs(i - j) == p - m]
matrix_groups(p, W) = [l2(reduce(vcat, [band(p, m) for m in 1:l]), reduce(vcat, [fill(W[l, m], 2m) for m in 1:l])) for l in 1:p-1]
band_norms(S) = (p = size(S, 1); [norm(S[band(p, m)]) for m in 1:p-1])

const S5 = [ 4.0   1.2  -0.7   0.5   0.9;
             1.2   3.0   1.1  -0.4   0.6;
            -0.7   1.1   2.5   0.8  -0.3;
             0.5  -0.4   0.8   3.5   1.4;
             0.9   0.6  -0.3   1.4   2.0]

let
  p = 5; λ = 0.25; W = general(p - 1)
  r = band_norms(S5)
  ρ, ξ, sweeps = dual_bcd(r, chain(W), 1:p-1, λ; tol = 1e-15)
  ρ1, _ = one_pass(r, chain(W), 1:p-1, λ)
  X = copy(S5); Ξ = [zeros(p * p) for _ in 1:p-1]
  for m in 1:p-1, i in band(p, m)
    X[i] = S5[i] * ρ[m] / r[m]
    for l in m:p-1
      Ξ[l][i] = ξ[l][m] * S5[i] / r[m]          # the dual block of group ℓ on band m
    end
  end
  gm = big_gap(vec(S5), vec(X), Ξ, matrix_groups(p, W), λ)
  gr = big_gap(r, ρ, ξ, chain(W), λ)
  record("B1 p = 5, general weights, λ = 0.25: the band-norm solution, scaled back, has a matrix duality gap at rounding level and keeps the diagonal",
         f64_ok(gm, vec(S5)) && all(X[i, i] == S5[i, i] for i in 1:p) && norm(ρ1 .- ρ) > 1e-3,
         "band norms $(fmtv(r)) -> $(fmtv(ρ)), matrix gap $(fmt(gm)), band-norm gap $(fmt(gr)), $sweeps sweeps; one pass differs by $(fmt(norm(ρ1 .- ρ)))")
end

let
  p = 5; λ = 0.6
  bands = [band(p, l) for l in 1:p-1]
  Xm = prox(GroupNormL2(1.0, bands, [sqrt(2l) for l in 1:p-1]), S5, λ)
  x, _ = one_pass(vec(S5), matrix_groups(p, control(p - 1)), 1:p-1, λ)
  d = maximum(abs.(vec(Xm) .- x))
  record("B2 the disjoint control (6) as nested groups with zero weights: one pass = prox(GroupNormL2) on master over the bands",
         d <= 1e-15, "largest difference $(fmt(d))")
end

# --- The recorded instance: p = 4 and the general weights -----------------------------------

let
  r = [3.3908789763847373, 2.723510817894522, 1.102998740113383]
  λ = 0.9768147919269893
  G = chain(general(3))
  x1, _ = one_pass(r, G, 1:3, λ)
  xr, ξr, sweeps, gapr = reference(r, G, 1:3, λ)
  dist, bound, refuted = refutation(x1, xr, gapr)
  g1, gr, excess = setprecision(BigFloat, 256) do
    B = BigFloat; GB = [convert_group(B, g) for g in G]
    (maximum(abs, objective_gradient(B.(r), B.(x1), GB, B(λ))), maximum(abs, objective_gradient(B.(r), B.(xr), GB, B(λ))),
     primal(B.(r), B.(x1), GB, B(λ)) - primal(B.(r), B.(xr), GB, B(λ)))
  end
  expected = [0.8806691284021112, 0.4398050685133424, 0.1577345228549693]
  record("B3 recorded instance p = 4, general weights: one pass (Float64) reproduces the posted point",
         maximum(abs.(x1 .- expected)) <= 1e-15, "one pass $(fmtv(x1))")
  record("B4 recorded instance: one pass refuted by the certified reference and by the gradient",
         refuted && g1 > 0.1 && all_norms_nonzero(x1, G) && all_norms_nonzero(xr, G),
         "reference $(fmtv(xr)) ($sweeps sweeps, BigFloat gap $(fmt(gapr))), distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound)), " *
         "largest gradient entry $(fmt(g1)) at the one-pass point and $(fmt(gr)) at the reference, objective excess $(repr(Float64(excess)))")
end

# --- Sweeps over weight families -------------------------------------------------------------

# For each instance: accepted (one pass in BigFloat has a gap at rounding level), refuted (its
# distance from a certified reference exceeds 2√(2 gap)), or neither. Also counts the instances
# whose number of zero bands differs between the one-pass point and the reference.
function family_sweep(rng, Wfun, K, N; scale)
  acc = 0; ref = 0; supp = 0
  for _ in 1:N
    W = Wfun(K)
    r = scale(K) .* 3 .* rand(rng, K)
    λ = 0.2 + rand(rng)
    G = chain(W)
    xb, _, gap = big_one_pass(r, G, 1:K, λ)
    if big_ok(gap, r)
      acc += 1
    else
      xr, _, _, gapr = reference(r, G, 1:K, λ)
      _, _, refuted = refutation(xb, xr, gapr)
      ref += refuted
      supp += count(iszero, Float64.(xb)) != count(iszero, xr)
    end
  end
  acc, ref, supp
end
bbx_scale(K) = [sqrt(2m) for m in 1:K]
unit_scale(K) = ones(K)

let rng = Xoshiro(6963)
  N = 100
  for (name, Wfun, Ks, scale, exact) in (
      ("basic weights (7)", basic, (3, 5, 9), bbx_scale, true),
      ("disjoint control (6)", control, (3, 5, 9), bbx_scale, true),
      ("general weights (8)", general, (2,), bbx_scale, true),
      ("general weights (8)", general, (3, 5, 9), bbx_scale, false),
      ("Yu-Bien weights 1/(ℓ-m+1)²", yubien, (2,), unit_scale, true),
      ("Yu-Bien weights 1/(ℓ-m+1)²", yubien, (3, 5, 9), unit_scale, false),
      ("two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ)", K -> twovalue(K, [sqrt(2l) * rand(rng) for l in 1:K]), (3, 5, 9), bbx_scale, true))
    for K in Ks
      acc, ref, supp = family_sweep(rng, Wfun, K, N; scale)
      ok = exact ? acc == N : acc + ref == N && ref > 0
      record("F $name, p = $(K + 1), $N random instances (seeded)", ok,
             exact ? "one pass exact in $acc of $N" :
                     "one pass exact in $acc, refuted in $ref, undecided in $(N - acc - ref); zero-band count differs in $supp")
    end
  end
end

let
  r = [2.0, 1.5, 1.0]; t = 0.5
  G = chain(yubien(3))
  x1, _ = one_pass(r, G, 1:3, t)
  xr, _, _, gapr = reference(r, G, 1:3, t)
  dist, bound, refuted = refutation(x1, xr, gapr)
  record("Y recorded instance, Yu-Bien weights, three coordinates, y = (2, 1.5, 1), step 0.5: one pass refuted", refuted,
         "one pass $(fmtv(x1)), reference $(fmtv(xr)), distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound))")
end

# --- λ_max of Appendix C (arXiv v1) and of lam.max.hierband: a diagonal result below it --------

let
  r = [2.0, 0.0]; λ = 1.0                     # p = 3: band norms of s₁ and s₂
  W = general(2); G = chain(W)
  lammax = maximum(r[l] / W[l, l] for l in 1:2)
  x, ξ, gap = big_one_pass(r, G, 1:2, λ)       # p = 3: one pass is exact (statement F above)
  record("L p = 3, general weights, band norms (2, 0), λ = 1 below max ‖S_sℓ‖/w_ℓℓ = $(fmt(lammax)): the proximal point is diagonal",
         λ < lammax && all(iszero, x) && big_ok(gap, r), "x = $(fmtv(x)), BigFloat gap $(fmt(gap))")
end

# --- Exact evaluation for the general weights: sweeps of dual BCD against p ------------------

# The largest spread, over the nonzero groups ℓ, of the factors s/(s + w_ℓm²) by which the step
# of group ℓ scales the coordinates m < ℓ that are nonzero, at the point x (s = ‖W_ℓ x‖/t). A zero
# spread for each group is the condition of Proposition 4; a large spread means that the later
# steps turn the residuals of the earlier groups.
function factor_spread(x, W, t)
  sp = 0.0
  for l in 2:size(W, 1)
    n = norm(W[l, 1:l] .* x[1:l]); iszero(n) && continue
    s = n / t
    f = [s / (s + W[l, m]^2) for m in 1:l-1 if !iszero(x[m])]
    length(f) >= 2 && (sp = max(sp, maximum(f) - minimum(f)))
  end
  sp
end

# An illustration, not a performance claim: cyclic dual BCD from zero (its first sweep is one
# pass) until the Float64 gap is at most 1e-12 (1 + ‖r‖²) or 100000 sweeps; the final gap is
# evaluated in BigFloat. Family A: signal in the b = K/4 bands next to the diagonal, λ = 1.
# Family B: signal in each band, λ = 1. Families C and D: as B with λ = 2 and λ = 2.5.
# "kept" is the largest one-pass band value at a band where the reference is zero, against
# √(2 gap), the largest distance of the reference from the proximal point.
let rng = Xoshiro(7963)
  N = 5; cap = 100_000
  for (fam, sig, λ) in (("A", K -> K - max(1, K ÷ 4), 1.0), ("B", K -> 0, 1.0), ("C", K -> 0, 2.0), ("D", K -> 0, 2.5))
    for K in (7, 15, 31, 63, 127)
      sw = Int[]; gaps = Float64[]; nz = Int[]; off = Float64[]; spread = Float64[]; kept = 0.0; bnd = 0.0
      for _ in 1:N
        r = [sqrt(2m) * (m > sig(K) ? 2 + abs(randn(rng)) : 0.5 * abs(randn(rng))) for m in 1:K]
        W = general(K); G = chain(W)
        x, ξ, s = dual_bcd(r, G, 1:K, λ; tol = 1e-12 * (1 + sum(abs2, r)), maxsweeps = cap)
        x1, _ = one_pass(r, G, 1:K, λ)
        g = big_gap(r, x, ξ, G, λ)
        push!(sw, s); push!(gaps, Float64(g / (1 + sum(abs2, r))))
        push!(nz, count(!iszero, x)); push!(off, norm(x1 .- x) / norm(r)); push!(spread, factor_spread(x, W, λ))
        z = [abs(x1[m]) for m in 1:K if iszero(x[m])]
        isempty(z) || (kept = max(kept, maximum(z)))
        bnd = max(bnd, Float64(sqrt(2g)))
      end
      reached = count(<=(1e-12), gaps)
      record("C$fam p = $(K + 1), general weights, λ = $λ, $N instances (seeded): sweeps of dual BCD", true,
             "sweeps $(sort(sw)) (gap reached in $reached of $N), nonzero bands $(extrema(nz)) of $K, largest factor spread $(fmt(maximum(spread))), " *
             "largest gap/(1+‖r‖²) $(fmt(maximum(gaps))), one pass off by ‖x₁ − x‖/‖r‖ up to $(fmt(maximum(off))), kept $(fmt(kept)) against √(2 gap) ≤ $(fmt(bnd))")
    end
  end
end

finish()
