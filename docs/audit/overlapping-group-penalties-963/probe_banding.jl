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
      size = startswith(name, "Yu") ? "$K penalized coordinates (row $(K + 1))" : "p = $(K + 1)"
      record("F $name, $size, $N random instances (seeded)", ok,
             exact ? "one pass accepted (BigFloat gap) in $acc of $N" :
                     "one pass accepted in $acc, refuted in $ref, undecided in $(N - acc - ref); zero-band count differs in $supp")
    end
  end
end

let
  r = [2.0, 1.5, 1.0]; t = 0.5
  G = chain(yubien(3))
  x1, _ = one_pass(r, G, 1:3, t)
  xr, _, _, gapr = reference(r, G, 1:3, t)
  dist, bound, refuted = refutation(x1, xr, gapr)
  record("Y recorded instance, Yu-Bien weights, three penalized coordinates, y = (2, 1.5, 1), step 0.5: one pass refuted", refuted,
         "one pass $(fmtv(x1)), reference $(fmtv(xr)), distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound))")
end

# --- Small steps: the one-pass gradient is −t² C(y) + O(t³) -----------------------------------

# Input supported on the two outermost bands, y = (2, 3, 0, …, 0). C(y) = Σ_ℓ H_ℓ(y) Σ_{j>ℓ} ∇N_j(y),
# with H_ℓ the Hessian of N_ℓ(z) = ‖W[ℓ,1:ℓ] ∘ z‖₂. Checked in BigFloat (512 bits) at t = 1e-40:
# ‖G/t² + C‖ ≤ 1e-30 ‖C‖ where C ≠ 0 (K ≥ 3), and ‖G‖/t² ≤ 1e-30 for K = 2.
function expansion(W, y, t)
  K = size(W, 1); G = chain(W)
  setprecision(BigFloat, 512) do
    B = BigFloat; yb = B.(y); tb = B(t); GB = [convert_group(B, g) for g in G]
    x, _ = one_pass(yb, GB, 1:K, tb)
    grad = objective_gradient(yb, x, GB, tb)
    C = zeros(B, K)
    for l in 1:K-1
      w2 = B.(W[l, 1:l]) .^ 2
      n = sqrt(sum(w2 .* yb[1:l] .^ 2)); a = w2 .* yb[1:l]
      v = zeros(B, l)
      for j in l+1:K
        wj = B.(W[j, 1:j]) .^ 2
        v .+= (wj .* yb[1:j] ./ sqrt(sum(wj .* yb[1:j] .^ 2)))[1:l]
      end
      C[1:l] .+= (w2 .* v .- a .* (dot(a, v) / n^2)) ./ n     # H_ℓ v
    end
    norm(grad ./ tb^2 .+ C), norm(C)
  end
end

let ok = true; det = String[]
  for (name, Wfun) in (("general", general), ("Yu-Bien", yubien)), K in (2, 3, 5, 10, 30)
    y = [2.0; 3.0; zeros(K - 2)]
    e, c = expansion(Wfun(K), y, 1e-40)
    ok &= K == 2 ? (c == 0 && e <= 1e-30) : (c > 0 && e <= 1e-30 * c)
    push!(det, "$name K=$K ‖C‖=$(fmt(c))")
  end
  record("E one-pass gradient = −t² C(y) + O(t³) at t = 1e-40, C = 0 for K = 2 and C ≠ 0 for K ≥ 3 (general and Yu-Bien weights)",
         ok, join(det, ", "))
end

# --- One pass can keep bands that the proximal point sets to zero -------------------------------

# p = 4, λ = 1, band norms (2.05, 3, 0), general weights. The proximal point is zero: dual block
# coordinate descent at 0.999λ gives blocks ξ₁, ξ₂; with ξ₃ = r − ξ₁ − ξ₂ (BigFloat) each block lies
# in the λ-ball of its dual norm, so r ∈ λ Σ_ℓ B°_ℓ and 0 satisfies the optimality condition.
let
  r = [2.05, 3.0, 0.0]; λ = 1.0
  W = general(3); G = chain(W)
  x1, _, _ = big_one_pass(r, G, 1:3, λ)
  xz, ξ, sweeps = dual_bcd(r, G, 1:3, 0.999λ; tol = 1e-20, maxsweeps = 100_000)
  norms = setprecision(BigFloat, 256) do
    B = BigFloat; ξb = [B.(z) for z in ξ]; ξb[3] = B.(r) .- ξb[1] .- ξb[2]
    [gdual(convert_group(B, G[l]), ξb[l]) for l in 1:3]
  end
  record("Z p = 4, λ = 1, band norms (2.05, 3, 0): one pass keeps bands 1 and 2, the proximal point is zero",
         maximum(norms) < λ && count(!iszero, x1) == 2,
         "one pass $(fmtv(x1)); BCD point at 0.999λ $(fmtv(xz)); dual norms of the decomposition r = ξ₁ + ξ₂ + ξ₃ $(fmtv(norms)) < λ ($sweeps sweeps at 0.999λ)")
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

# --- Certified evaluation for the general weights: dual BCD and a Newton-type method ----------

# The smallest nonzero group norm at x, relative to λ w_ℓℓ: a value near 0 is a group barely nonzero.
function smallest_group(x, W, λ)
  v = [norm(W[l, 1:l] .* x[1:l]) / (λ * W[l, l]) for l in 1:size(W, 1)]
  v = filter(!iszero, v)
  isempty(v) ? NaN : minimum(v)
end

# Newton's method with a backtracking (Armijo) line search on the bands L+1..K, the bands 1..L held
# at zero. The objective there is ½‖z − r‖² + λ Σ_{ℓ>L} ‖W[ℓ, L+1:ℓ] ∘ z‖₂; its Hessian is
# I + λ Σ_ℓ (diag(w²) − a aᵀ/n²)/n (a = w² ∘ z, n the group norm), formed densely and factored by
# Cholesky: O(K³) operations per iteration. Stops when the Newton decrement −gᵀd is at most
# 1e-14 (1 + ‖r‖²); a zero group norm, a stalled line search or 100 iterations stop it as a failure.
function newton_bands(r, W, λ, L, z0; maxit = 100)
  K = length(r); S = L+1:K; m = length(S)
  rs = r[S]; z = copy(z0)
  f(z) = sum(abs2, z .- rs) / 2 + λ * sum(norm(W[l, L+1:l] .* z[1:l-L]) for l in S)
  for it in 0:maxit
    g = z .- rs; H = Matrix{Float64}(I, m, m)
    for l in S
      k = l - L; w2 = W[l, L+1:l] .^ 2
      n = sqrt(sum(w2 .* z[1:k] .^ 2)); iszero(n) && return z, it, false
      a = w2 .* z[1:k]
      g[1:k] .+= λ .* a ./ n
      H[1:k, 1:k] .+= λ .* (Diagonal(w2) .- (a * a') ./ n^2) ./ n
    end
    d = -(cholesky(Symmetric(H)) \ g)
    gd = dot(g, d)
    -gd <= 1e-14 * (1 + sum(abs2, r)) && return z, it, true
    it == maxit && return z, it, false
    st = 1.0; f0 = f(z)
    while f(z .+ st .* d) > f0 + 1e-4 * st * gd && st > 1e-12
      st /= 2
    end
    st <= 1e-12 && return z, it, false
    z .+= st .* d
  end
  z, maxit, false
end

# The safeguarded method: n₀ dual sweeps (from zero) propose the zero prefix L (the leading zero bands of the
# iterate); Newton solves the remaining bands; dual BCD on the chain of the first L groups at r₁..L
# supplies their dual blocks; the result is accepted only if the duality gap of the whole problem,
# evaluated in BigFloat, is at most 1e-12 (1 + ‖r‖²). Otherwise the attempt is reported as failed
# (a wrong prefix or a Newton failure) and the proposal is repeated with ten times more sweeps
# (n₀ = 2, 20, 200, 2000).
function newton_method(r, W, λ)
  K = length(r); G = chain(W); tol = 1e-12 * (1 + sum(abs2, r))
  rejected = 0; newton_its = 0; cert_sweeps = 0; n0 = 0
  for n0_try in (2, 20, 200, 2000)
    n0 = n0_try
    x0, _, _ = dual_bcd(r, G, 1:K, λ; tol = 0.0, maxsweeps = n0)
    L = something(findfirst(!iszero, x0), K + 1) - 1
    ρ = zeros(K); ξ = [zeros(K) for _ in 1:K]; ok = true
    if L < K
      z, its, conv = newton_bands(r, W, λ, L, x0[L+1:K]); newton_its += its
      ok = conv
      ρ[L+1:K] .= z
      for l in L+1:K
        n = norm(W[l, 1:l] .* ρ[1:l]); iszero(n) && (ok = false; break)
        ξ[l][1:l] .= λ .* W[l, 1:l] .^ 2 .* ρ[1:l] ./ n
      end
    end
    if ok && L > 0
      xL, ξL, sw = dual_bcd(r[1:L], chain(W[1:L, 1:L]), 1:L, λ; tol = 1e-13 * (1 + sum(abs2, r)), maxsweeps = 100_000)
      cert_sweeps += sw
      ok = all(iszero, xL)
      for l in 1:L
        ξ[l][1:L] .= ξL[l]
      end
    end
    if ok && big_gap(r, ρ, ξ, G, λ) <= tol
      return true, n0, newton_its, cert_sweeps, rejected
    end
    rejected += 1
  end
  false, n0, newton_its, cert_sweeps, rejected
end

# An illustration, not a performance claim, on four families of band norms (five instances per
# size): A, signal in the b = K/4 bands next to the diagonal, λ = 1; B, signal in each band, λ = 1;
# C and D, as B with λ = 2 and λ = 2.5. Dual BCD runs from zero (its first sweep is one pass) until
# its Float64 gap is at most 1e-12 (1 + ‖r‖²) or 100000 sweeps; its final gap is evaluated in
# BigFloat. Each instance is listed as (sweeps, smallest nonzero group norm / λw_ℓℓ), sorted by sweeps.
# "kept" is the largest one-pass band value at a band where the BCD point is zero, against √(2 gap).
let rng = Xoshiro(7963)
  N = 5; cap = 100_000
  for (fam, sig, λ) in (("A", K -> K - max(1, K ÷ 4), 1.0), ("B", K -> 0, 1.0), ("C", K -> 0, 2.0), ("D", K -> 0, 2.5))
    for K in (7, 15, 31, 63, 127)
      runs = Tuple{Int, Float64}[]; gaps = Float64[]; nz = Int[]; off = Float64[]; kept = 0.0; bnd = 0.0
      nt = Tuple{Int, Int, Int, Int, Float64}[]; ncert = 0
      for _ in 1:N
        r = [sqrt(2m) * (m > sig(K) ? 2 + abs(randn(rng)) : 0.5 * abs(randn(rng))) for m in 1:K]
        W = general(K); G = chain(W)
        x, ξ, s = dual_bcd(r, G, 1:K, λ; tol = 1e-12 * (1 + sum(abs2, r)), maxsweeps = cap)
        x1, _ = one_pass(r, G, 1:K, λ)
        g = big_gap(r, x, ξ, G, λ)
        push!(runs, (s, smallest_group(x, W, λ))); push!(gaps, Float64(g / (1 + sum(abs2, r))))
        push!(nz, count(!iszero, x)); push!(off, norm(x1 .- x) / norm(r))
        z = [abs(x1[m]) for m in 1:K if iszero(x[m])]
        isempty(z) || (kept = max(kept, maximum(z)))
        bnd = max(bnd, Float64(sqrt(2g)))
        okn, n0, its, cs, rej = newton_method(r, W, λ)
        ncert += okn; push!(nt, (n0, its, cs, rej, smallest_group(x, W, λ)))
      end
      reached = count(<=(1e-12), gaps)
      sort!(runs)
      record("C$fam p = $(K + 1), general weights, λ = $λ, $N instances (seeded): dual BCD", true,
             "(sweeps, smallest group) $(join(("($a, $(fmt(b)))" for (a, b) in runs), " ")) (gap reached in $reached of $N), " *
             "nonzero bands $(extrema(nz)) of $K, largest gap/(1+‖r‖²) $(fmt(maximum(gaps))), " *
             "one pass off by ‖x₁ − x‖/‖r‖ up to $(fmt(maximum(off))), kept $(fmt(kept)) against √(2 gap) ≤ $(fmt(bnd))")
      record("N$fam p = $(K + 1), λ = $λ: Newton-type method", true,
             "certified in $ncert of $N; (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) " *
             join(("($a, $b, $c, $d, $(fmt(e)))" for (a, b, c, d, e) in nt), " "))
    end
  end
end

finish()
