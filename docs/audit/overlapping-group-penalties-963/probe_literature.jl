# Statements L1–L5 of docs/design/overlapping-group-penalties-963-literature.md: weighted ℓ∞
# groups on a laminar family (L1), a radial function on a maximal group (L2), weighted ℓ1 groups
# inside ℓ∞ groups (L3), the weights (W3) of Jenatton, Audibert and Bach on a chain, the latent
# penalty on a nested chain of Yan and Bien (L4), the smallest λ with a zero banding prox (L5),
# and dual block coordinate descent at a threshold (an observation).
#
#   julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_literature.jl

include(joinpath(@__DIR__, "harness.jl"))

# --- Group functions beyond the harness ------------------------------------------------------

struct WLinf{T}; idx::Vector{Int}; η::T; d::Vector{T}; end    # η ‖D z‖∞, D = diag(d) > 0
struct WL1{T}; idx::Vector{Int}; c::Vector{T}; end             # Σ cᵢ |zᵢ|, c ≥ 0
struct Radial{T}; idx::Vector{Int}; ω::T; μ::T; R::T; end      # h(‖z‖₂), h(r) = ωr + μr²/2 on [0, R]

conv(::Type{T}, g::Grp) where {T} = convert_group(T, g)
conv(::Type{T}, g::WLinf) where {T} = WLinf(g.idx, T(g.η), T.(g.d))
conv(::Type{T}, g::WL1) where {T} = WL1(g.idx, T.(g.c))
conv(::Type{T}, g::Radial) where {T} = Radial(g.idx, T(g.ω), T(g.μ), T(g.R))

val(g::Grp, x) = gvalue(g, x)
val(g::WLinf, x) = g.η * maximum(g.d[k] * abs(x[i]) for (k, i) in enumerate(g.idx))
val(g::WL1, x) = sum(g.c[k] * abs(x[i]) for (k, i) in enumerate(g.idx))
function val(g::Radial, x::AbstractVector{T}) where {T}        # the ball admits a rounding error
  r = sqrt(sum(abs2, x[g.idx])); r > g.R * (1 + 64eps(T)) ? T(Inf) : g.ω * r + g.μ * r^2 / 2
end

# The step of t N on x[g.idx], in place.
pprox!(x, g::Grp, t) = gprox!(x, g, t)
function pprox!(x::AbstractVector{T}, g::WL1, t) where {T}
  for (k, i) in enumerate(g.idx)
    x[i] = sign(x[i]) * max(abs(x[i]) - t * g.c[k], zero(T))
  end
  x
end
function pprox!(x::AbstractVector{T}, g::Radial, t) where {T}       # x ↦ (ρ/r) x, ρ the scalar step of t h
  v = x[g.idx]; r = sqrt(sum(abs2, v))
  iszero(r) && return x
  x[g.idx] .= v .* (clamp((r - t * g.ω) / (1 + t * g.μ), zero(T), g.R) / r)
  x
end
# v minus its projection onto {u : Σ |uᵢ|/dᵢ ≤ tη}: D x is D v clipped at a level θ (rational in v).
function pprox!(x::AbstractVector{T}, g::WLinf, t) where {T}
  τ = t * g.η; v = x[g.idx]; d = g.d
  iszero(τ) && return x
  if sum(abs(v[k]) / d[k] for k in eachindex(v)) <= τ
    x[g.idx] .= zero(T)
    return x
  end
  b = [d[k] * abs(v[k]) for k in eachindex(v)]                        # breakpoints of the level
  p = sortperm(b; rev = true); A = zero(T); B = zero(T); θ = zero(T)
  for (m, k) in enumerate(p)
    A += abs(v[k]) / d[k]; B += 1 / d[k]^2; θ = (A - τ) / B
    (m == length(p) || b[p[m+1]] <= θ) && break
  end
  for (k, i) in enumerate(g.idx)
    x[i] = sign(v[k]) * min(abs(v[k]), θ / d[k])
  end
  x
end

# The dual block ξ of t N enters the dual as c ξ with the conjugate value q: a sublinear function
# has its block scaled into the dual ball (q = 0); a radial function keeps it (c = 1, t > 0).
scl(dn, t) = dn <= t ? one(t) : (isfinite(dn) ? t / dn : zero(t))
ratio(a::T, w) where {T} = iszero(w) ? (iszero(a) ? zero(T) : T(Inf)) : a / w
dualpart(g::Grp, ξ, t) = (g.kind === :zero ? one(t) : scl(gdual(g, ξ), t), zero(t))
dualpart(g::WLinf, ξ, t) = (scl(ratio(sum(abs(ξ[i]) / g.d[k] for (k, i) in enumerate(g.idx)), g.η), t), zero(t))
dualpart(g::WL1, ξ, t) = (scl(maximum(ratio(abs(ξ[i]), g.c[k]) for (k, i) in enumerate(g.idx)), t), zero(t))
function dualpart(g::Radial, ξ, t)
  s = sqrt(sum(abs2, ξ[g.idx]))
  iszero(g.μ) && return (one(t), g.R * max(s - t * g.ω, zero(t)))  # needs R finite
  ρ = clamp((s - t * g.ω) / (t * g.μ), zero(t), g.R)
  (one(t), ρ * (s - t * g.ω) - t * g.μ * ρ^2 / 2)
end

function pass(y::Vector{T}, groups, order, t::T) where {T}
  x = copy(y); ξ = [zeros(T, length(y)) for _ in groups]
  for k in order
    idx = groups[k].idx; before = x[idx]
    pprox!(x, groups[k], t)
    ξ[k][idx] .= before .- x[idx]
  end
  x, ξ
end
# P(x) − D(ξ) ≥ ½‖x − x*‖², with D(ξ) = ½‖y‖² − ½‖y − Σ cₖξₖ‖² − Σ qₖ (Fenchel duality).
function gap(y::Vector{T}, x::Vector{T}, ξ, groups, t::T) where {T}
  s = copy(y); q = zero(T)
  for (k, g) in enumerate(groups)
    c, qk = dualpart(g, ξ[k], t); s .-= c .* ξ[k]; q += qk
  end
  sum(abs2, x .- y) / 2 + t * sum(val(g, x) for g in groups) - (sum(abs2, y) / 2 - sum(abs2, s) / 2 - q)
end
bigpass(y, groups, order, t) = setprecision(BigFloat, 256) do
  B = BigFloat; yb = B.(y); gb = [conv(B, g) for g in groups]
  x, ξ = pass(yb, gb, order, B(t))
  x, ξ, gap(yb, x, ξ, gb, B(t))
end
moved(x, y, groups) = any(g -> any(i -> !iszero(x[i]), g.idx) && any(i -> x[i] != y[i], g.idx), groups)
ischild(kids) = (c = falses(length(kids)); for ks in kids, k in ks; c[k] = true; end; c)

# One sweep: random laminar families (children before parents), groups from `make`, one pass
# accepted by its BigFloat gap.
function sweep(rng, N, make)
  ok = true; worst = big"0.0"; nz = 0
  for _ in 1:N
    n = rand(rng, 3:12)
    sets, kids = random_laminar(rng, n)
    groups = make(rng, n, sets, kids)
    y = 3 .* randn(rng, n)
    x, _, g = bigpass(y, groups, eachindex(groups), rand(rng) + 0.2)
    nz += moved(x, y, groups)
    worst = max(worst, g / (1 + sum(abs2, big.(y)))); ok &= big_ok(g, y)
  end
  ok, "$nz with a group moved and not zero, largest BigFloat gap/(1+‖y‖²) $(fmt(worst))"
end
w03(rng) = rand(rng) < 0.1 ? 0.0 : 0.2 + 3rand(rng)      # a weight, zero in about 10%

# --- L1: weighted ℓ∞ groups, D_j on g_k proportional to D_k ---------------------------------

let rng = Xoshiro(8963), N = 300
  ok, det = sweep(rng, N, (rng, n, sets, kids) -> begin
    # dyadic weights in [1/8, 3], so that the products below are exact and the weights proportional
    d = Vector{Vector{Float64}}(undef, length(sets)); G = Any[]; dy() = rand(rng, 1:24) / 8
    for j in eachindex(sets)
      dj = Dict(i => dy() for i in sets[j])                       # own coordinates: any positive weights
      for c in kids[j]                                            # on a child: a multiple of its weights
        a = dy()
        for (k, i) in enumerate(sets[c]); dj[i] = a * d[c][k]; end
      end
      d[j] = [dj[i] for i in sets[j]]
      push!(G, rand(rng) < 0.05 ? gzero(Float64, sets[j]) : WLinf(sets[j], w03(rng), d[j]))
    end
    G
  end)
  record("L1 $N random laminar families (seeded), η‖D z‖∞ with D_j on g_k proportional to D_k, η = 0 and ∞ included, one pass exact", ok, det)
end

let rng = Xoshiro(9963), N = 300
  ok, det = sweep(rng, N, (rng, n, sets, kids) -> begin
    D = 0.2 .+ 3 .* rand(rng, n)
    Any[WLinf(s, w03(rng), D[s]) for s in sets]
  end)
  record("L1 $N random laminar families (seeded), one weight vector D shared by all groups, one pass exact", ok, det)
end

# The recorded instance, in exact rational arithmetic: ‖z‖∞ on {1,2} inside ‖diag(1,2,1) z‖∞ on {1,2,3}.
let R = Rational{BigInt}
  y = R[3, 29//10, 0]; t = R(1)
  G = [WLinf([1, 2], R(1), R[1, 1]), WLinf([1, 2, 3], R(1), R[1, 2, 1])]
  x, _ = pass(y, G, [1, 2], t)
  δ = R[-1, 0, 0]                                   # direction −e₁
  dd = sum((x .- y) .* δ)                           # derivative of the smooth part, then of each max term
  for g in G
    m = val(g, x) / g.η
    dd += t * g.η * maximum(g.d[k] * sign(x[i]) * δ[i] for (k, i) in enumerate(g.idx) if g.d[k] * abs(x[i]) == m)
  end
  record("L1 ‖z‖∞ on {1,2} inside ‖diag(1,2,1) z‖∞ on {1,2,3}, η = t = 1, y = (3, 2.9, 0): one pass refuted (exact)",
         x == R[107//50, 107//100, 0] && dd == -7//50,
         "one pass $(join(string.(x), ", ")), directional derivative along −e₁ $(dd)")
end

# --- L2: a radial function on a maximal group --------------------------------------------------

let rng = Xoshiro(10963), N = 300
  ok, det = sweep(rng, N, (rng, n, sets, kids) -> begin
    top = .!ischild(kids); G = Any[]
    for j in eachindex(sets)
      s = sets[j]; u = rand(rng)
      if top[j]                                     # ωr + μr²/2, with a ball constraint in about 40%
        μ = u < 0.3 ? 0.0 : 2rand(rng); Rb = rand(rng) < 0.4 || iszero(μ) ? 0.5 + 3rand(rng) : Inf
        push!(G, Radial(s, w03(rng), μ, Rb))
      elseif !isempty(kids[j])
        push!(G, l2s(s, w03(rng)))                  # Proposition 2: scalar ℓ2 on a group that contains another
      else
        push!(G, u < 0.25 ? linf(s, w03(rng)) : u < 0.5 ? l1(s, w03(rng)) : u < 0.75 ? l2(s, [w03(rng) for _ in s]) :
                 u < 0.9 ? l2s(s, w03(rng)) : gzero(Float64, s))
      end
    end
    G
  end)
  record("L2 $N random laminar families (seeded), h(‖·‖₂) with h(r) = ωr + μr²/2 (and a ball) on each maximal group, Proposition 2 below it, one pass exact", ok, det)
end

function sgrad(y, x, groups, t)                     # gradient where every group norm is nonzero
  G = x .- y
  for g in groups
    r = sqrt(sum(abs2, x[g.idx]))
    if g isa Radial
      G[g.idx] .+= t .* (g.ω / r + g.μ) .* x[g.idx]
    else
      n = gvalue(g, x)
      for (k, i) in enumerate(g.idx); G[i] += t * g.d[k]^2 * x[i] / n; end
    end
  end
  G
end

let y = [2.0, 3.0, 1.5], t = 0.3
  xa, _, ga = bigpass(y, [l2s(1:2, 0.8), Radial([1, 2, 3], 0.5, 2.0, Inf)], [1, 2], t)
  record("L2 0.8‖z‖₂ on {1,2}, 0.5r + r² (r = ‖z‖₂) on {1,2,3}, y = (2, 3, 1.5), t = 0.3: one pass exact",
         big_ok(ga, y) && !iszero(xa), "x = $(fmtv(xa)), BigFloat gap $(fmt(ga))")
  G = [Radial([1, 2], 0.5, 2.0, Inf), l2s(1:3, 0.8)]
  xb, _, _ = bigpass(y, G, [1, 2], t)
  gr = setprecision(BigFloat, 256) do; maximum(abs, sgrad(big.(y), xb, [conv(BigFloat, g) for g in G], big(t))); end
  record("L2 the radial function on the contained group {1,2} and 0.8‖z‖₂ on {1,2,3}: one pass refuted by the gradient",
         gr > 1e-3 && all(g -> !iszero(sum(abs2, xb[g.idx])), G),
         "one pass $(fmtv(xb)), largest gradient entry $(fmt(gr)) (all group norms nonzero)")
end

# --- L3: weighted ℓ1 groups inside ℓ∞ groups -----------------------------------------------------

let rng = Xoshiro(11963), N = 300
  ok, det = sweep(rng, N, (rng, n, sets, kids) ->
    Any[!isempty(kids[j]) ? linf(sets[j], w03(rng)) : rand(rng) < 0.7 ? WL1(sets[j], [w03(rng) for _ in sets[j]]) : linf(sets[j], w03(rng))
        for j in eachindex(sets)])
  record("L3 $N random laminar families (seeded), ω‖·‖∞ on each group that contains another, weighted ℓ1 or ℓ∞ on the others, one pass exact", ok, det)
end

# --- The weights (W3) of Jenatton, Audibert and Bach on a chain: d^ℓ_j = ρ^(ℓ−j) ------------------

let y = [2.0, 1.5, 1.0], t = 0.5, ρ = 0.5
  G = [l2(1:l, [ρ^(l - j) for j in 1:l]) for l in 1:3]
  xr, _, sweeps, gapr = reference(y, G, 1:3, t)
  xh, _, _ = big_one_pass(y, G, 1:3, t)
  dist, bound, refuted = refutation(xh, xr, gapr)
  gr = setprecision(BigFloat, 256) do
    maximum(abs, objective_gradient(big.(y), xh, [convert_group(BigFloat, g) for g in G], big(t)))
  end
  record("W3 chain {1} ⊂ {1,2} ⊂ {1,2,3}, weights ρ^(ℓ−j) with ρ = 0.5, y = (2, 1.5, 1), t = 0.5: one pass refuted",
         refuted && gr > 1e-3 && all_norms_nonzero(xh, G),
         "one pass $(fmtv(xh)), reference $(fmtv(xr)) ($sweeps sweeps), distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound)), largest gradient entry $(fmt(gr))")
end

# --- L4: the latent penalty on a nested chain (Yan and Bien, Algorithm 3) ------------------------

# Increments s_1, …, s_D (`blocks`), groups g_ℓ = s_1 ∪ … ∪ s_ℓ, one weight w_ℓ > 0 per group.
cumZ(y::Vector{T}, blocks) where {T} = [zero(T); cumsum([sum(abs2, y[b]; init = zero(T)) for b in blocks])]
cat_blocks(blocks, a, b) = reduce(vcat, blocks[a:b]; init = Int[])

function alg3(y::Vector{T}, blocks, w, λ; tie = :first) where {T}     # as published, w strictly increasing
  D = length(blocks); Z = cumZ(y, blocks); W = [zero(T); T.(w) .^ 2]
  x = zeros(T, length(y)); k = 0; knots = Int[]
  while k < D
    best = -one(T); K = 0
    for j in k+1:D
      f = sqrt((Z[j+1] - Z[k+1]) / (W[j+1] - W[k+1]))
      if f > best || (tie === :last && f == best); best = f; K = j; end
    end
    best <= λ && break
    idx = cat_blocks(blocks, k + 1, K)
    x[idx] .= max(zero(T), 1 - λ * sqrt(W[K+1] - W[k+1]) / sqrt(sum(abs2, y[idx]))) .* y[idx]
    push!(knots, K); k = K
  end
  x, knots
end

# Vertices of the least concave majorant of (w_j², ‖y_{g_j}‖²), j = 0..D, by a monotone stack.
function hull(y::Vector{T}, blocks, w) where {T}
  Z = cumZ(y, blocks); W = [zero(T); T.(w) .^ 2]; st = [0]
  for j in 1:length(blocks)
    while length(st) >= 2 && (Z[st[end]+1] - Z[st[end-1]+1]) * (W[j+1] - W[st[end-1]+1]) <=
                             (Z[j+1] - Z[st[end-1]+1]) * (W[st[end]+1] - W[st[end-1]+1])
      pop!(st)
    end
    push!(st, j)
  end
  st, Z, W
end
function hull_prox(y::Vector{T}, blocks, w, λ) where {T}             # each segment: y (1 − λ/√slope)₊
  st, Z, W = hull(y, blocks, w); x = zeros(T, length(y))
  for m in 2:length(st)
    a, b = st[m-1], st[m]; σ = (Z[b+1] - Z[a+1]) / (W[b+1] - W[a+1]); idx = cat_blocks(blocks, a + 1, b)
    x[idx] .= (σ > λ^2 ? 1 - λ / sqrt(σ) : zero(T)) .* y[idx]
  end
  x
end
hull_value(y, blocks, w) = ((st, Z, W) = hull(y, blocks, w);
  sum((sqrt((W[st[m]+1] - W[st[m-1]+1]) * (Z[st[m]+1] - Z[st[m-1]+1])) for m in 2:length(st)); init = zero(eltype(y))))

# Groups k with w_k ≥ w_j for some j > k are removed and the increments merged.
function reduce_chain(blocks, w)
  D = length(w); keep = [l for l in 1:D if l == D || w[l] < minimum(w[l+1:D])]
  [cat_blocks(blocks, (r == 1 ? 1 : keep[r-1] + 1), keep[r]) for r in eachindex(keep)], w[keep], keep
end

# The decomposition of Appendix C on the knots k_j (a_{m+1} = λ): v^(k_j) = A_j (a_j − a_{j+1}),
# A_j = Σ_{i≤j} y_{seg_i}/a_i. Returned on the groups `gid` (indices of the original chain), with the
# rounding residual x − Σ v put into the largest group, so that Σ v = x holds.
function decomposition(y::Vector{T}, x, blocks, w, knots, λ, gid) where {T}
  Z = cumZ(y, blocks); W = [zero(T); T.(w) .^ 2]; k = [0; knots]
  a = [sqrt((Z[k[j+1]+1] - Z[k[j]+1]) / (W[k[j+1]+1] - W[k[j]+1])) for j in 1:length(knots)]
  v = Dict{Int,Vector{T}}(); A = zeros(T, length(y))
  for j in eachindex(knots)
    idx = cat_blocks(blocks, k[j] + 1, k[j+1]); A[idx] .= y[idx] ./ a[j]
    v[gid[knots[j]]] = A .* (a[j] - (j < length(knots) ? a[j+1] : λ))
  end
  top = gid[length(blocks)]
  v[top] = get(v, top, zeros(T, length(y))) .+ (x .- reduce(+, values(v); init = zeros(T, length(y))))
  v
end
groupnorms(u, blocks, ℓ) = sqrt(sum(abs2, u[cat_blocks(blocks, 1, ℓ)]; init = zero(eltype(u))))

# Duality gap of x for ½‖x − y‖² + λ Ω(x): u = y − x scaled into {‖u_{g_ℓ}‖ ≤ λ w_ℓ, all ℓ of the
# original chain}, and Ω(x) bounded above by the decomposition v.
function latent_gap(y, x, blocks, w, λ, v)
  u = y .- x; r = maximum(groupnorms(u, blocks, l) / (λ * w[l]) for l in eachindex(w))
  us = u ./ max(1, r)
  cost = sum(w[l] * sqrt(sum(abs2, vl)) for (l, vl) in v)
  sum(abs2, u) / 2 + λ * cost - (sum(abs2, y) / 2 - sum(abs2, y .- us) / 2), r
end

function random_chain(rng; increasing)
  D = rand(rng, 1:8); sizes = [rand(rng, 0:3) for _ in 1:D]; sizes[1] += 1; sizes[end] += 1
  starts = cumsum([1; sizes]); blocks = [collect(starts[i]:starts[i+1]-1) for i in 1:D]
  w = if increasing
    cumsum(0.1 .+ rand(rng, D))
  elseif rand(rng) < 0.5
    [sqrt(length(cat_blocks(blocks, 1, l))) for l in 1:D]    # √|g_ℓ|: equal weights where an increment is empty
  else
    [rand(rng, (0.5, 1.0, 1.5, 2.0)) for _ in 1:D]           # equal and decreasing weights
  end
  blocks, w, 3 .* randn(rng, starts[end] - 1), 0.1 + 2rand(rng)
end

let rng = Xoshiro(12963), N = 300
  worst = [big"0.0", big"0.0"]; dmax = big"0.0"; ok = [true, true]; knots_seen = [0, 0]; red = 0
  vworst = big"0.0"; vok = true
  for i in 1:2N
    s = i <= N ? 1 : 2
    blocks, w, y, λ = random_chain(rng; increasing = s == 1)
    setprecision(BigFloat, 256) do
      yb = big.(y); wb = big.(w); λb = big(λ)
      rb, rw, keep = reduce_chain(blocks, wb); red += length(keep) < length(w)
      x, knots = alg3(yb, rb, rw, λb)
      v = decomposition(yb, x, rb, rw, knots, λb, keep)
      g, r = latent_gap(yb, x, blocks, wb, λb, v)
      worst[s] = max(worst[s], g / (1 + sum(abs2, yb))); ok[s] &= big_ok(g, y) && r <= 1 + big"1e-60"
      knots_seen[s] += length(knots)
      dmax = max(dmax, maximum(abs.(hull_prox(yb, rb, rw, λb) .- x); init = big"0.0"))
      # the value formula on the reduced chain, between a lower bound (a point of the dual ball of
      # all original groups) and an upper bound (a decomposition on original groups)
      st, Z, W = hull(yb, rb, rw); u = zeros(BigFloat, length(y)); vk = Int[]
      for m in 2:length(st)
        σ = (Z[st[m]+1] - Z[st[m-1]+1]) / (W[st[m]+1] - W[st[m-1]+1]); idx = cat_blocks(rb, st[m-1] + 1, st[m])
        σ > 0 && (u[idx] .= yb[idx] ./ sqrt(σ); push!(vk, st[m]))
      end
      lower = sum(u .* yb) / max(1, maximum(groupnorms(u, blocks, l) / wb[l] for l in eachindex(wb)))
      upper = isempty(vk) ? big"0.0" : sum(wb[l] * sqrt(sum(abs2, vl)) for (l, vl) in decomposition(yb, yb, rb, rw, vk, big"0.0", keep))
      vf = hull_value(yb, rb, rw)
      vworst = max(vworst, (upper - lower) / (1 + vf)); vok &= lower <= vf * (1 + big"1e-60") && vf <= upper * (1 + big"1e-60") + big"1e-70"
    end
  end
  record("L4 Algorithm 3 as published, $N random nested chains (seeded), strictly increasing weights: certified",
         ok[1], "$(knots_seen[1]) knots in all, largest BigFloat gap/(1+‖y‖²) $(fmt(worst[1])), every ‖u_{g_ℓ}‖ ≤ λ w_ℓ")
  record("L4 $N random chains (seeded) with equal or decreasing weights ($red reduced): groups with w_k ≥ w_j for some j > k removed, Algorithm 3 on the rest, certified against all groups",
         ok[2], "largest BigFloat gap/(1+‖y‖²) $(fmt(worst[2]))")
  record("L4 the hull variant (each segment y (1 − λ/√slope)₊) equals Algorithm 3 on the $(2N) chains", dmax <= big"1e-60", "largest difference $(fmt(dmax))")
  record("L4 the value formula Σ √(Δw²)‖y_seg‖ on the hull lies between a dual lower bound and a decomposition upper bound on the $(2N) chains",
         vok && vworst <= big"1e-60", "largest (upper − lower)/(1 + value) $(fmt(vworst))")
end

# Exact ties in line 5: y = (1 | 1, 1, 1 | 1, 2), w = (1, 2, 3), λ = 1/2: every f(j, 0) equals 1.
let blocks = [[1], [2, 3, 4], [5, 6]]
  setprecision(BigFloat, 256) do
    y = big.([1.0, 1, 1, 1, 1, 2]); w = big.([1.0, 2, 3]); λ = big"0.5"
    res = map((:first, :last)) do tie
      x, knots = alg3(y, blocks, w, λ; tie)
      g, r = latent_gap(y, x, blocks, w, λ, decomposition(y, x, blocks, w, knots, λ, 1:3))
      x, knots, g, r
    end
    record("L4 exact ties in line 5 (y = (1 | 1, 1, 1 | 1, 2), w = (1, 2, 3), λ = 1/2): first and last maximizer both certified, same point",
           all(big_ok(z[3], y) && z[4] <= 1 for z in res) && res[1][1] == res[2][1] && res[1][1] == y ./ 2,
           "knots $(res[1][2]) and $(res[2][2]), x = y/2, BigFloat gaps $(fmt(res[1][3])) and $(fmt(res[2][3]))")
  end
end

# --- L5: the smallest λ with a zero banding prox is the dual norm Ω°(r) -------------------------

# p = 3, general weights W = [√2 0; 1 2], band norms r = (2, 0). Lower bound ⟨r, x⟩/Ω(x) at x = (1, 0);
# upper bound max dual norm of the decomposition r = (a, 0) + (2 − a, 0), a = 4 − 2√2.
let
  setprecision(BigFloat, 256) do
    s2 = sqrt(big(2)); a = 4 - 2s2
    lower = 2 / (s2 + 1)
    upper = max(a / s2, (2 - a) / 1)
    record("L5 p = 3, general weights, band norms (2, 0): λ_max = Ω°(r) between equal bounds, below max ‖S_sℓ‖/w_ℓℓ = 1.41",
           abs(upper - lower) <= big"1e-70" && abs(lower - (2s2 - 2)) <= big"1e-70",
           "lower $(fmt(lower)), upper $(fmt(upper)) (2√2 − 2 = $(fmt(2s2 - 2)))")
  end
end

# --- Observation: dual block coordinate descent with every group at its threshold ----------------

# p = 4, general weights, λ = 1, r = c (support point of B°₁ + B°₂ + B°₃ in the direction
# (cos π/3, sin π/3, 0)); for c ≤ 1 the proximal point is 0 (zero-band characterization), so ‖x_k‖ is
# the distance after sweep k; at c = 1 the dual blocks lie on their ellipsoids and the group norms are 0.
let K = 3, λ = 1.0
  W = [m <= l ? sqrt(2l) / (l - m + 1) : 0.0 for l in 1:K, m in 1:K]
  G = [l2(1:l, W[l, 1:l]) for l in 1:K]
  n = [cos(pi / 3), sin(pi / 3), 0.0]; pt = zeros(K)
  for l in 1:K
    d = W[l, 1:l]; pt[1:l] .+= d .^ 2 .* n[1:l] ./ norm(d .* n[1:l])
  end
  marks = (10, 100, 1000, 10000); res = Dict{Float64,Vector{Float64}}()
  for c in (1.0, 0.999)
    x = c * λ .* pt; ξ = [zeros(K) for _ in 1:K]; out = Float64[]
    for s in 1:last(marks)
      for k in 1:K
        idx = G[k].idx; x[idx] .+= ξ[k][idx]; before = x[idx]; gprox!(x, G[k], λ); ξ[k][idx] .= before .- x[idx]
      end
      s in marks && push!(out, norm(x))
    end
    res[c] = out
  end
  r1 = res[1.0][3] / res[1.0][4]
  record("O p = 4, general weights, every group at its threshold (c = 1) and inside it (c = 0.999): dual BCD distance after 10, 100, 1000, 10000 sweeps",
         5 <= r1 <= 20 && res[0.999][4] <= 1e-3 * res[1.0][4],
         "c = 1: $(join(fmt.(res[1.0]), ", ")); c = 0.999: $(join(fmt.(res[0.999]), ", "))")
end

finish()
