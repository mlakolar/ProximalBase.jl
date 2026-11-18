# Shared code of the probes of docs/design/overlapping-group-penalties-963.md.
#
# The proximal problem of a sum of group functions with a scalar step t ≥ 0,
#
#     minimize_x  ½‖x − y‖² + t Σ_g N_g(x[g]),
#
# for three kinds of group function N_g:
#   :l2    the weighted ℓ2 norm ‖D z‖₂ with diagonal weights d ≥ 0 (zero allowed),
#   :linf  ω‖z‖∞ with a scalar weight ω ≥ 0,
#   :l1    ω‖z‖₁ with a scalar weight ω ≥ 0,
#   :zero  the indicator of {0} (an infinite weight).
# The code is generic in the element type, so that the same routines run in
# Float64 and in BigFloat. Nothing here is part of ProximalBase.

using LinearAlgebra, Printf, Random

struct Grp{T}
  kind::Symbol
  idx::Vector{Int}
  d::Vector{T}          # :l2: one weight per index; :linf, :l1: [ω]; :zero: empty
end
l2(idx, d) = Grp(:l2, collect(Int, idx), collect(d))
l2s(idx, ω) = Grp(:l2, collect(Int, idx), fill(ω, length(idx)))   # scalar weight
linf(idx, ω) = Grp(:linf, collect(Int, idx), [ω])
l1(idx, ω) = Grp(:l1, collect(Int, idx), [ω])
gzero(::Type{T}, idx) where {T} = Grp(:zero, collect(Int, idx), T[])
convert_group(::Type{T}, g::Grp) where {T} = Grp(g.kind, g.idx, T.(g.d))

# value of N_g at the point x
function gvalue(g::Grp, x::AbstractVector{T}) where {T}
  if g.kind === :l2
    return sqrt(sum((T(g.d[k]) * x[i])^2 for (k, i) in enumerate(g.idx)))
  elseif g.kind === :linf
    return T(g.d[1]) * maximum(abs(x[i]) for i in g.idx)
  elseif g.kind === :l1
    return T(g.d[1]) * sum(abs(x[i]) for i in g.idx)
  else
    return all(iszero(x[i]) for i in g.idx) ? zero(T) : T(Inf)
  end
end

# dual norm of N_g at the dual block u (a full-length vector supported on g.idx)
function gdual(g::Grp, u::AbstractVector{T}) where {T}
  if g.kind === :l2
    s = zero(T)
    for (k, i) in enumerate(g.idx)
      dk = T(g.d[k])
      if iszero(dk)
        iszero(u[i]) || return T(Inf)
      else
        s += (u[i] / dk)^2
      end
    end
    return sqrt(s)
  elseif g.kind === :linf
    ω = T(g.d[1])
    a = sum(abs(u[i]) for i in g.idx)
    return iszero(ω) ? (iszero(a) ? zero(T) : T(Inf)) : a / ω
  elseif g.kind === :l1
    ω = T(g.d[1])
    a = maximum(abs(u[i]) for i in g.idx)
    return iszero(ω) ? (iszero(a) ? zero(T) : T(Inf)) : a / ω
  else
    return zero(T)
  end
end

# Root s > 0 of h(s) = Σ d²v²/(s + d²)² = c², for h(0) > c². h is convex and decreasing,
# so Newton's method started left of the root increases monotonically to it.
function ellipsoid_root(v::Vector{T}, d::Vector{T}, c::T) where {T}
  nDv = sqrt(sum((d .* v) .^ 2))
  s = max(zero(T), nDv / c - maximum(d)^2)
  for _ in 1:2000
    h = zero(T); dh = zero(T)
    for k in eachindex(v)
      q = d[k]^2 + s
      a = (d[k] * v[k])^2
      h += a / q^2
      dh -= 2a / q^3
    end
    h <= c^2 && return s
    snew = s - (h - c^2) / dh
    snew <= s && return s
    s = snew
  end
  error("ellipsoid_root did not converge")
end

# The proximal operator of t N_g, applied to the entries g.idx of x in place.
function gprox!(x::AbstractVector{T}, g::Grp, t::T) where {T}
  if g.kind === :zero
    for i in g.idx
      x[i] = zero(T)
    end
  elseif g.kind === :l2
    P = [k for k in eachindex(g.idx) if !iszero(g.d[k])]       # entries of zero weight are unpenalized
    (isempty(P) || iszero(t)) && return x
    v = T[x[g.idx[k]] for k in P]
    d = T[T(g.d[k]) for k in P]
    if sum((v ./ d) .^ 2) <= t^2
      for k in P
        x[g.idx[k]] = zero(T)
      end
    else
      s = ellipsoid_root(v, d, t)
      for (j, k) in enumerate(P)
        x[g.idx[k]] = s * v[j] / (s + d[j]^2)
      end
    end
  elseif g.kind === :l1                                         # soft threshold at tω
    r = t * T(g.d[1])
    for i in g.idx
      x[i] = sign(x[i]) * max(abs(x[i]) - r, zero(T))
    end
  else                                                          # :linf, x minus its projection onto the ℓ1 ball
    r = t * T(g.d[1])
    iszero(r) && return x
    v = T[x[i] for i in g.idx]
    a = sort(abs.(v); rev = true)
    if sum(a) <= r
      for i in g.idx
        x[i] = zero(T)
      end
    else
      cs = zero(T); θ = zero(T)
      for k in eachindex(a)
        cs += a[k]
        θk = (cs - r) / k
        if k == length(a) || a[k+1] <= θk
          θ = θk
          break
        end
      end
      for (k, i) in enumerate(g.idx)
        x[i] = sign(v[k]) * min(abs(v[k]), θ)
      end
    end
  end
  x
end

# One pass of the group proximal steps in the given order. The dual block of each step is
# rebuilt from it: the entries before the step minus the entries after it.
function one_pass(y::Vector{T}, groups, order, t::T) where {T}
  x = copy(y)
  ξ = [zeros(T, length(y)) for _ in groups]
  for k in order
    g = groups[k]
    before = x[g.idx]
    gprox!(x, g, t)
    ξ[k][g.idx] .= before .- x[g.idx]
  end
  x, ξ
end

primal(y, x, groups, t) = sum(abs2, x .- y) / 2 + t * sum(gvalue(g, x) for g in groups)

# Duality gap of a primal point x and dual blocks ξ: each block is scaled into its dual ball
# (radius t) first, so that the dual value is that of a feasible point and the gap bounds
# P(x) − P* ≥ ½‖x − x*‖² from above.
function duality_gap(y::Vector{T}, x::Vector{T}, ξ, groups, t::T) where {T}
  s = copy(y)
  for (k, g) in enumerate(groups)
    dn = gdual(g, ξ[k])
    c = (g.kind === :zero || dn <= t) ? one(T) : (isfinite(dn) ? t / dn : zero(T))
    s .-= c .* ξ[k]
  end
  dual = sum(abs2, y) / 2 - sum(abs2, s) / 2
  primal(y, x, groups, t) - dual
end

# The gap evaluated in BigFloat (256 bits) from points of any type.
function big_gap(y, x, ξ, groups, t)
  setprecision(BigFloat, 256) do
    B = BigFloat
    duality_gap(B.(y), B.(x), [B.(z) for z in ξ], [convert_group(B, g) for g in groups], B(t))
  end
end

# Cyclic block coordinate descent on the dual, run until its duality gap is at most tol
# (or maxsweeps). Returns the primal point, the dual blocks and the number of sweeps.
function dual_bcd(y::Vector{T}, groups, order, t::T; tol, maxsweeps = 10^6) where {T}
  x = copy(y)
  ξ = [zeros(T, length(y)) for _ in groups]
  for sweep in 1:maxsweeps
    for k in order
      g = groups[k]
      x[g.idx] .+= ξ[k][g.idx]          # the residual without block k
      before = x[g.idx]
      gprox!(x, g, t)
      ξ[k][g.idx] .= before .- x[g.idx]
    end
    duality_gap(y, x, ξ, groups, t) <= tol && return x, ξ, sweep
  end
  x, ξ, maxsweeps
end

# Gradient of the objective; valid where each :l2 group norm is nonzero (and no other kind).
function objective_gradient(y::Vector{T}, x::Vector{T}, groups, t::T) where {T}
  G = x .- y
  for g in groups
    g.kind === :l2 || error("gradient needs :l2 groups")
    n = gvalue(g, x)
    for (k, i) in enumerate(g.idx)
      G[i] += t * T(g.d[k])^2 * x[i] / n
    end
  end
  G
end
all_norms_nonzero(x, groups) = all(g -> g.kind === :l2 && !iszero(gvalue(g, x)), groups)

# A certified reference: dual BCD in Float64 to a gap at rounding level, the gap then
# evaluated in BigFloat. A candidate is refuted when its distance from the reference
# exceeds √(2 gap), the largest distance of the reference from the proximal point.
function reference(y, groups, order, t)
  yf = Float64.(y)
  gf = [convert_group(Float64, g) for g in groups]
  x, ξ, sweeps = dual_bcd(yf, gf, order, Float64(t); tol = 1e-15 * (1 + sum(abs2, yf)), maxsweeps = 100_000)
  x, ξ, sweeps, big_gap(yf, x, ξ, gf, t)
end
function refutation(xhat, xref, gapref)
  setprecision(BigFloat, 256) do
    dist = norm(BigFloat.(xhat) .- BigFloat.(xref))
    bound = sqrt(2 * max(gapref, zero(gapref)))
    dist, bound, dist > 2 * bound + big"1e-12"
  end
end

# Recorded statements: one line each, and a nonzero exit status when one fails.
const FAILURES = Ref(0)
function record(name, ok::Bool, detail)
  ok || (FAILURES[] += 1)
  println(ok ? "PASS " : "FAIL ", name, ": ", detail)
  ok
end
finish() = (println(FAILURES[] == 0 ? "all statements hold" : "$(FAILURES[]) statement(s) failed"); exit(FAILURES[] == 0 ? 0 : 1))

fmt(x::Real) = @sprintf("%.3g", Float64(x))
fmtv(v) = "(" * join((repr(Float64(z)) for z in v), ", ") * ")"

# Tolerances for acceptance: a BigFloat candidate (256 bits) and a Float64 candidate.
big_ok(gap, y) = gap <= big"1e-50" * (1 + sum(abs2, big.(y)))
f64_ok(gap, y) = gap <= 1e-12 * (1 + sum(abs2, Float64.(y)))

# One pass in BigFloat, accepted by its BigFloat gap.
function big_one_pass(y, groups, order, t)
  setprecision(BigFloat, 256) do
    B = BigFloat
    yb = B.(y); gb = [convert_group(B, g) for g in groups]
    x, ξ = one_pass(yb, gb, order, B(t))
    x, ξ, duality_gap(yb, x, ξ, gb, B(t))
  end
end

# Order "a group after each group it contains", by size (ties keep the given order).
inclusion_order(groups) = sortperm([length(g.idx) for g in groups]; alg = Base.Sort.DEFAULT_STABLE)

# A random laminar family on 1:n: a forest built by recursive splitting, with some
# coordinates in no group. Returns the list of index sets in an order where each group
# comes after the groups it contains (children before parents), and for each group the
# list of its children.
function random_laminar(rng, n; maxdepth = 4)
  sets = Vector{Vector{Int}}()
  children = Vector{Vector{Int}}()
  coords = shuffle(rng, collect(1:n))
  nogroup = coords[1:rand(rng, 0:max(0, n ÷ 5))]
  rest = setdiff(coords, nogroup)
  function build(idxs, depth)
    local cuts, bounds, part               # not the variables of the enclosing function
    kids = Int[]
    own_n = rand(rng, 0:length(idxs))
    sub = idxs[own_n+1:end]
    if depth < maxdepth && length(sub) >= 1
      nparts = rand(rng, 1:min(3, length(sub)))
      cuts = sort(rand(rng, 1:length(sub), nparts - 1))
      bounds = [0; cuts; length(sub)]
      for p in 1:nparts
        part = sub[bounds[p]+1:bounds[p+1]]
        isempty(part) || push!(kids, build(part, depth + 1))
      end
    end
    push!(sets, sort(idxs)); push!(children, kids)
    length(sets)
  end
  # a forest: split the covered coordinates into one to three trees
  ntrees = rand(rng, 1:min(3, max(1, length(rest))))
  cuts = sort(rand(rng, 1:max(1, length(rest)), ntrees - 1))
  bounds = [0; cuts; length(rest)]
  for p in 1:ntrees
    part = rest[bounds[p]+1:bounds[p+1]]
    isempty(part) || build(part, 1)
  end
  sets, children
end

# whether some penalized group (positive weight) is left nonzero: the instance is not trivial
nontrivial(x, groups) = any(g -> g.kind !== :zero && any(!iszero, g.d) && any(i -> !iszero(x[i]), g.idx), groups)
