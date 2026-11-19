# Propositions 1-3 of docs/design/overlapping-group-penalties-963.md: disjoint groups, the
# sparse-group (two-level) penalty and laminar groups with one scalar weight per group.
#
#   julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_laminar.jl

include(joinpath(@__DIR__, "harness.jl"))
using ProximalBase

# --- Disjoint groups against prox(GroupNormL2) on master (a check of the harness) ---------

# GroupNormL2(λ0, groups, w) with step γ is the harness problem with t = γ and scalar weights
# λ0 wₖ; an infinite weight is the indicator of zero and a zero weight leaves the group unpenalized.
function harness_groups(λ0, groups, w)
  [isinf(w[k]) ? gzero(Float64, groups[k]) : l2s(groups[k], λ0 * w[k]) for k in eachindex(groups)]
end
function master_check(y, groups, w, λ0, γ)
  xm = prox(GroupNormL2(λ0, groups, w), y, γ)
  hg = harness_groups(λ0, groups, w)
  xh, ξh = one_pass(y, hg, eachindex(hg), γ)     # dual blocks rebuilt from the harness steps
  gap = big_gap(y, xm, ξh, hg, γ)                 # the gap of the master point with those blocks
  maximum(abs.(xm .- xh); init = 0.0), gap
end

let
  y = [3.0, -1.0, 4.0, 1.5, -2.0, 0.5, 2.5, -0.25, 1.0, 7.0]
  groups = [[3, 1], [4, 6, 5], [8], [9, 2], [10]]           # index 7 is in no group
  w = [1.0, 0.0, 2.0, Inf, 1.5]
  diff, gap = master_check(y, groups, w, 0.8, 1.25)
  record("D1 recorded disjoint instance, master prox(GroupNormL2) = one pass of the harness",
         diff <= 1e-15 && f64_ok(gap, y), "max difference $(fmt(diff)), BigFloat gap of the master point $(fmt(gap))")
end

let rng = Xoshiro(963)
  N = 200; worst = 0.0; worstgap = 0.0; ok = true
  for _ in 1:N
    n = rand(rng, 2:12)
    perm = shuffle(rng, collect(1:n))
    k = rand(rng, 1:n)
    cuts = sort(unique(rand(rng, 1:n, k - 1)))
    bounds = [0; cuts; n]
    groups = [perm[bounds[j]+1:bounds[j+1]] for j in 1:length(bounds)-1 if bounds[j+1] > bounds[j]]
    rand(rng) < 0.5 && length(groups) > 1 && pop!(groups)                  # coordinates in no group
    w = [rand(rng) < 0.1 ? 0.0 : rand(rng) < 0.1 ? Inf : 2rand(rng) for _ in groups]
    y = 3 .* randn(rng, n)
    diff, gap = master_check(y, groups, w, rand(rng) + 0.1, rand(rng) + 0.1)
    worst = max(worst, diff); worstgap = max(worstgap, Float64(gap / (1 + sum(abs2, y))))
    ok &= diff <= 1e-14 * (1 + maximum(abs, y)) && f64_ok(gap, y)
  end
  record("D2 $N random disjoint instances (seeded), master = harness", ok,
         "largest difference $(fmt(worst)), largest gap/(1+‖y‖²) $(fmt(worstgap))")
end

# --- Laminar groups with scalar weights: one pass in the inclusion order -------------------

# a forest with a duplicate group, a chain of five nested groups, a group at its
# threshold, zero and infinite weights, and coordinates in no group (11, 12)
const LAM_Y = [3, 4, 2.5, -1.5, 2, -3, 1, 5, -2, 0.75, 9, -9]
lam_groups(::Type{T}) where {T} = [
  l2s(1:2, T(5)),          # 1: ‖y[1:2]‖ = 5 = tω, at its threshold
  l2s([3], T(1) / 2),      # 2: a singleton (an ℓ1 term)
  l2s(1:4, T(1)),          # 3
  l2s(1:4, T(7) / 10),     # 4: a duplicate of 3
  l2s(1:6, T(3) / 10),     # 5
  l2s(1:7, T(1) / 5),      # 6
  gzero(T, [8]),           # 7: an infinite weight
  l2s([9], T(0)),          # 8: a zero weight
  l2s(8:10, T(2) / 5),     # 9: a second tree
]

let
  groups = lam_groups(Float64)
  order = inclusion_order(groups)
  x, ξ, gap = big_one_pass(LAM_Y, groups, order, 1)
  record("L1 recorded laminar instance (forest, duplicate, five nested groups, threshold, 0 and Inf weights), one pass exact",
         big_ok(gap, LAM_Y), "order $(order), BigFloat gap $(fmt(gap)), x[1:2] = $(fmtv(x[1:2])), x[11:12] = $(fmtv(x[11:12]))")
  order2 = [7, 8, 9, 2, 1, 4, 3, 5, 6]                       # another order that respects inclusion
  x2, _, gap2 = big_one_pass(LAM_Y, groups, order2, 1)
  record("L2 the same instance in another inclusion order gives the same point",
         big_ok(gap2, LAM_Y) && maximum(abs.(x2 .- x)) <= big"1e-60", "BigFloat gap $(fmt(gap2))")
end

let rng = Xoshiro(1963)
  N = 300; worst = big"0.0"; ok = true; nz = 0
  for _ in 1:N
    n = rand(rng, 3:14)
    sets, _ = random_laminar(rng, n)
    groups = [rand(rng) < 0.05 ? gzero(Float64, s) : l2s(s, rand(rng) < 0.1 ? 0.0 : 2rand(rng)) for s in sets]
    if rand(rng) < 0.3                       # a duplicate of a random group, right after it
      j = rand(rng, eachindex(groups)); insert!(groups, j + 1, l2s(groups[j].idx, rand(rng)))
    end
    y = 3 .* randn(rng, n)
    xs, _, gap = big_one_pass(y, groups, eachindex(groups), rand(rng) + 0.2)
    nz += nontrivial(xs, groups)
    worst = max(worst, gap / (1 + sum(abs2, big.(y)))); ok &= big_ok(gap, y)
  end
  record("L3 $N random laminar families (seeded; forests, duplicates, coordinates in no group, 0 and Inf weights), scalar ℓ2 weights, one pass exact",
         ok, "$nz with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) $(fmt(worst))")
end

# --- Wrong order: the sparse-group penalty |x₁|/2 + ‖x‖/2 at y = (1, 0.1) --------------------

let
  y = [1.0, 0.1]; t = 1.0
  groups = [l2s([1], 0.5), l2s([2], 0.5), l2s(1:2, 0.5)]
  xr, ξr, sweeps, gapr = reference(y, groups, [1, 2, 3], t)
  xc, _, gapc = big_one_pass(y, groups, [1, 2, 3], t)
  xw, _, _ = big_one_pass(y, groups, [3, 1, 2], t)
  dist, bound, refuted = refutation(xw, xr, gapr)
  record("W1 sparse-group, coordinates before the group: one pass exact", big_ok(gapc, y),
         "x = $(fmtv(xc)), BigFloat gap $(fmt(gapc))")
  record("W2 sparse-group, the group before the coordinates (wrong order): refuted", refuted,
         "x = $(fmtv(xw)), distance from the reference $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound)) (reference: $sweeps sweeps)")
end

# --- The ℓ∞ norm and mixed norms --------------------------------------------------------------

let rng = Xoshiro(2963)
  N = 300; worst = big"0.0"; ok = true; nz = 0
  for _ in 1:N
    n = rand(rng, 3:12)
    sets, _ = random_laminar(rng, n)
    groups = [linf(s, rand(rng) < 0.1 ? 0.0 : 2rand(rng)) for s in sets]
    y = 3 .* randn(rng, n)
    xs, _, gap = big_one_pass(y, groups, eachindex(groups), rand(rng) + 0.2)
    nz += nontrivial(xs, groups)
    worst = max(worst, gap / (1 + sum(abs2, big.(y)))); ok &= big_ok(gap, y)
  end
  record("M1 $N random laminar families (seeded), ℓ∞ with scalar weights on all groups, one pass exact",
         ok, "$nz with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) $(fmt(worst))")
end

let rng = Xoshiro(3963)
  N = 300; worst = big"0.0"; ok = true; nz = 0
  for _ in 1:N
    n = rand(rng, 3:12)
    sets, kids = random_laminar(rng, n)
    groups = map(eachindex(sets)) do j
      s = sets[j]
      contains_other = !isempty(kids[j])
      contains_other && return l2s(s, 2rand(rng))                  # a containing group: scalar ℓ2
      r = rand(rng)                                                  # a group containing no other: any of three norms
      r < 1/3 ? linf(s, 2rand(rng)) : r < 2/3 ? l2(s, 2 .* rand(rng, length(s))) : l2s(s, 2rand(rng))
    end
    y = 3 .* randn(rng, n)
    xs, _, gap = big_one_pass(y, groups, eachindex(groups), rand(rng) + 0.2)
    nz += nontrivial(xs, groups)
    worst = max(worst, gap / (1 + sum(abs2, big.(y)))); ok &= big_ok(gap, y)
  end
  record("M2 $N random laminar families (seeded), scalar ℓ2 on each containing group, ℓ∞ or diagonal-weight ℓ2 on the others, one pass exact",
         ok, "$nz with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) $(fmt(worst))")
end

let
  y = [3.0, 1.0, 0.0]; t = 1.0
  groups = [l2s(1:2, 1.0), linf(1:3, 1.0)]                    # ℓ2 group inside an ℓ∞ group
  xr, ξr, sweeps, gapr = reference(y, groups, [1, 2], t)
  xh, _, _ = big_one_pass(y, groups, [1, 2], t)
  dist, bound, refuted = refutation(xh, xr, gapr)
  record("M3 an ℓ2 group {1,2} inside an ℓ∞ group {1,2,3}, y = (3, 1, 0): one pass refuted", refuted,
         "one pass $(fmtv(xh)), reference $(fmtv(xr)), distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound))")
end

finish()
