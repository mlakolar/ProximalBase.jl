# Propositions 4 and 5 of docs/design/overlapping-group-penalties-963.md: laminar groups with
# diagonal weights inside a group, and a diagonal step.
#
#   julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_weighted.jl

include(joinpath(@__DIR__, "harness.jl"))

parents(kids) = (p = zeros(Int, length(kids)); for (j, ks) in enumerate(kids), c in ks; p[c] = j; end; p)

# --- Sufficiency: the weights of each group are constant on each group it contains ---------

let rng = Xoshiro(4963)
  N = 300; worst = big"0.0"; ok = true; nz = 0; nested = 0
  for _ in 1:N
    n = rand(rng, 3:14)
    sets, kids = random_laminar(rng, n)
    weight() = rand(rng) < 0.1 ? 0.0 : 2rand(rng)
    groups = map(eachindex(sets)) do j
      d = Dict(i => weight() for i in sets[j])            # its own coordinates: any weights
      for c in kids[j]                                    # constant on each child, so on each contained group
        a = weight()
        for i in sets[c]; d[i] = a; end
      end
      l2(sets[j], [d[i] for i in sets[j]])
    end
    if rand(rng) < 0.3            # a duplicate after group j, with a scalar weight (constant on group j)
      j = rand(rng, eachindex(groups)); insert!(groups, j + 1, l2s(groups[j].idx, 2rand(rng)))
    end
    nested += any(!isempty, kids)
    y = 3 .* randn(rng, n)
    xs, _, gap = big_one_pass(y, groups, eachindex(groups), rand(rng) + 0.2)
    nz += nontrivial(xs, groups)
    worst = max(worst, gap / (1 + sum(abs2, big.(y)))); ok &= big_ok(gap, y)
  end
  record("P1 $N random laminar families (seeded; $nested with nested groups), diagonal weights constant on each contained group, zero weights included, one pass exact",
         ok, "$nz with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) $(fmt(worst))")
end

# --- Necessity for two groups: weights of the larger group that differ on the smaller ------

let
  y = [2.0, 3.0, 1.0]; t = 0.3
  groups = [l2(1:2, [1.0, 2.0]), l2(1:3, [1.0, 3.0, 2.0])]   # group 2 has weights 1 and 3 on group 1
  xr, ξr, sweeps, gapr = reference(y, groups, [1, 2], t)
  xh, _, _ = big_one_pass(y, groups, [1, 2], t)
  dist, bound, refuted = refutation(xh, xr, gapr)
  gr = setprecision(BigFloat, 256) do
    B = BigFloat
    all_norms_nonzero(xh, groups) ? maximum(abs, objective_gradient(B.(y), B.(xh), [convert_group(B, g) for g in groups], B(t))) : B(NaN)
  end
  record("P2 two groups {1,2} ⊂ {1,2,3}, weights (1,2) and (1,3,2), y = (2,3,1), t = 0.3: one pass refuted",
         refuted && gr > 1e-3,
         "one pass $(fmtv(xh)), reference $(fmtv(xr)), distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound)), largest gradient entry at the one-pass point $(fmt(gr)) (all group norms nonzero)")
end

# --- The condition of (a) is sufficient, not necessary, for other contained functions -------

# An ℓ1 group inside an ℓ2 group whose weights differ on it: the step of the larger group is a
# positive diagonal scaling (or zero), which keeps the signs and zeros that ∂‖·‖₁ depends on.
let ok = true; worst = big"0.0"; det = String[]
  for (y, c, t) in (([3.0, -2.0, 1.0], 0.5, 0.4), ([5.0, 4.0, -2.0], 1.2, 0.7))
    groups = [l1(1:2, c), l2(1:3, [1.0, 3.0, 2.0])]
    x, _, gap = big_one_pass(y, groups, [1, 2], t)
    ok &= big_ok(gap, y) && nontrivial(x, groups); worst = max(worst, gap)
    push!(det, "y = $(fmtv(y)): x = $(fmtv(x))")
  end
  record("P5 an ℓ1 group {1,2} inside ‖diag(1,3,2) z‖₂ on {1,2,3}: one pass exact although the weights differ on {1,2}",
         ok, join(det, "; ") * ", largest BigFloat gap $(fmt(worst))")
end

# --- Duplicate groups with diagonal weights: the order between them matters -----------------

let
  y = [3.0, -2.0, 4.0]; t = 0.5
  groups = [l2(1:3, [1.0, 2.0, 3.0]), l2s(1:3, 1.5)]
  xr, _, _, gapr = reference(y, groups, [1, 2], t)
  x12, _, gap12 = big_one_pass(y, groups, [1, 2], t)
  x21, _, _ = big_one_pass(y, groups, [2, 1], t)
  dist, bound, refuted = refutation(x21, xr, gapr)
  record("P3 duplicates g₁ = g₂ = {1,2,3}, diagonal weights (1,2,3) then the scalar weight 1.5: one pass exact",
         big_ok(gap12, y) && !iszero(x12), "x = $(fmtv(x12)), BigFloat gap $(fmt(gap12))")
  record("P4 the same duplicates, the scalar-weight group first: one pass refuted", refuted,
         "distance $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound))")
end

# a later duplicate that is not a weighted ℓ2 norm: ‖z‖₂ then ‖z‖∞ on {1,2}, y = (3, 1), t = 1, whose
# proximal point is (2 − 2/√5, 1 − 1/√5): there y − x = (2, 1)/√5 + e₁
let
  y = [3.0, 1.0]; t = 1.0
  groups = [l2s(1:2, 1.0), linf(1:2, 1.0)]
  x, _, _ = big_one_pass(y, groups, [1, 2], t)
  xs = setprecision(BigFloat, 256) do; [2 - 2 / sqrt(big(5)), 1 - 1 / sqrt(big(5))]; end
  ξs = setprecision(BigFloat, 256) do; [[2 / sqrt(big(5)), 1 / sqrt(big(5))], [big(1), big(0)]]; end
  gs = big_gap(y, xs, ξs, groups, t)
  d = Float64(norm(big.(x) .- xs))
  record("P6 duplicates ‖z‖₂ then ‖z‖∞ on {1,2}, y = (3, 1), t = 1: the closed-form point is certified and one pass differs from it",
         big_ok(gs, y) && d > 1e-3, "one pass $(fmtv(x)), proximal point $(fmtv(xs)) (BigFloat gap $(fmt(gs))), distance $(fmt(d))")
end

# --- A diagonal step: ½ Σ (xᵢ − yᵢ)²/τᵢ + Ω(x) is the problem in z = x/√τ with weights d √τ ----

function step_groups(groups, τ)
  [g.kind === :l2 ? l2(g.idx, g.d .* sqrt.(τ[g.idx])) : g for g in groups]
end

let
  y = [3.5, -0.4, 4.0, 1.3, -1.0]; τ = [0.5, 1.0, 2.0, 4.0, 3.0]
  c = [0.3, 0.1, 0.5, 0.2]
  groups = [[l2s([i], c[i]) for i in 1:4]; l2s(1:4, 0.6)]     # sparse group; coordinate 5 in no group
  z, _, gap = big_one_pass(y ./ sqrt.(τ), step_groups(groups, τ), eachindex(groups), 1)
  x = z .* sqrt.(τ)
  record("S1 sparse-group penalty with the diagonal step τ = (0.5, 1, 2, 4, 3): one pass exact",
         big_ok(gap, y ./ sqrt.(τ)) && count(!iszero, x[1:4]) >= 2, "x = $(fmtv(x)), BigFloat gap $(fmt(gap))")
end

let rng = Xoshiro(5963)
  N = 300; worst = big"0.0"; ok = true; nz = 0
  for _ in 1:N
    n = rand(rng, 3:12)
    sets, kids = random_laminar(rng, n)
    par = parents(kids)
    τ = 0.2 .+ 3 .* rand(rng, n)
    for j in eachindex(sets)                       # τ constant on each group contained in another
      if par[j] != 0 && par[par[j]] == 0
        τ[sets[j]] .= 0.2 + 3rand(rng)
      end
    end
    groups = [l2s(s, 2rand(rng)) for s in sets]
    y = 3 .* randn(rng, n)
    z = y ./ sqrt.(τ)
    xs, _, gap = big_one_pass(z, step_groups(groups, τ), eachindex(groups), rand(rng) + 0.2)
    nz += nontrivial(xs, groups)
    worst = max(worst, gap / (1 + sum(abs2, big.(z)))); ok &= big_ok(gap, z)
  end
  record("S2 $N random laminar families (seeded), scalar weights, a diagonal step constant on each contained group, one pass exact",
         ok, "$nz with a penalized group left nonzero, largest BigFloat gap/(1+‖z‖²) $(fmt(worst))")
end

let
  y = [2.0, 1.0, 1.5]; τ = [1.0, 4.0, 1.0]; t = 0.4
  groups = step_groups([l2s(1:2, 1.0), l2s(1:3, 1.0)], τ)
  z = y ./ sqrt.(τ)
  xr, _, _, gapr = reference(z, groups, [1, 2], t)
  xh, _, _ = big_one_pass(z, groups, [1, 2], t)
  dist, bound, refuted = refutation(xh, xr, gapr)
  record("S3 scalar weights on {1,2} ⊂ {1,2,3} with the step τ = (1, 4, 1), not constant on {1,2}: one pass refuted",
         refuted, "distance (in z) $(fmt(dist)) > 2√(2 gap) = $(fmt(2bound))")
end

finish()
