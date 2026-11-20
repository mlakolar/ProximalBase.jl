module GroupNormLinfTest
# GroupNormLinf (#956): f(x) = λ0 Σₖ wₖ max_{i∈Gₖ} |xᵢ| and its proximal operators, the clip of each group at the level τₖ
# with Σ_{i∈Gₖ}(|xᵢ| - τₖ)₊ = γλ0wₖ (and +0 when γλ0wₖ ≥ ‖x[Gₖ]‖₁), against an independent sort-based ℓ1-ball threshold
# written in this file in BigFloat, in Float32 and Float64. The established reference cases keep their tolerance of
# TOL_C n eps(T) max|xᵢ| plus two subnormal spacings. The documented bound allows (2n + 1) subnormal spacings to
# account for the shared threshold and the rounded radius. A group equal to the whole argument is also
# compared with NormLinf, and singleton groups with NormL1: the proximal outputs bit for bit, the values within a tolerance.
# kkt_violation: the margin of a zero group, the distance of a nonzero group to the subdifferential of the penalty, the full-point form
# (the maximum over the groups of that distance, zero groups included, and over the entries in no group of |vᵢ|) and the scores under
# Scaled, against BigFloat references written in this file, by sorting: the distances within the first-order bound of the kernel of
# NormLinf, the margins within the bound of a sum of magnitudes.
using Test, ProximalBase, LinearAlgebra, Random, OffsetArrays
using ..TestUtils

const PB = ProximalBase
const PREC = 256
const TOL_C = 8

# ---- independent references (BigFloat, from the definition) ----

# the sorted magnitudes and their cumulative sums
function ref_setup(u)
  setprecision(BigFloat, PREC) do
    p = sort!([abs(big(v)) for v in u]; rev = true)
    p, cumsum(p)
  end
end
# τ with Σ(pᵢ - τ)₊ = r: the first k with (S_k - r)/k ≥ p_{k+1} (or k = n), for 0 ≤ r < Σ pᵢ
function ref_level(p, cs, r)
  n = length(p)
  for k in 1:n
    τ = (cs[k] - r) / k
    (k == n || τ >= p[k + 1]) && return max(τ, zero(τ))
  end
  zero(r)
end
# the level of the group u for the radius γλ0w, formed in BigFloat from the stored factors; zero when the radius reaches ‖u‖₁
function ref_group_level(u, γ, λ0, w, setup = ref_setup(u))
  setprecision(BigFloat, PREC) do
    p, cs = setup
    r = big(γ) * big(λ0) * big(w)
    isempty(p) || r >= cs[end] ? big(0) : ref_level(p, cs, r)
  end
end
# the output of the proximal operator from the definition: each group clipped at its level (a zero weight copies the group,
# an infinite one gives zero), the entries in no group copied; also the levels
function ref_prox(x, groups, w, γ, λ0)
  setprecision(BigFloat, PREC) do
    y = big.(x)
    τs = Vector{BigFloat}(undef, length(groups))
    for k in eachindex(groups)
      G = groups[k]
      u = x[G]
      if iszero(w[k])
        τs[k] = maximum(abs, big.(u); init = big(0))
      elseif isinf(w[k])
        y[G] .= 0
        τs[k] = big(0)
      else
        τ = ref_group_level(u, γ, λ0, w[k])
        y[G] = [clamp(big(v), -τ, τ) for v in u]
        τs[k] = τ
      end
    end
    y, τs
  end
end
# the tolerance on an entry of the output of a group of n entries with largest magnitude m
tol(T, n, m) = TOL_C * n * big(eps(T)) * big(m) + 2 * big(nextfloat(zero(T)))
# every group within its tolerance, and the entries in no group equal to the argument
function close_prox(y, ref, x, groups)
  T = eltype(y)
  grouped = Int[]
  ok = true
  for G in groups
    append!(grouped, G)
    b = tol(T, length(G), maximum(abs, x[G]; init = zero(T)))
    ok &= all(i -> abs(big(y[i]) - ref[i]) <= b, G)
  end
  ok && all(i -> y[i] === x[i], setdiff(eachindex(x), grouped))
end
# a single group 1:n
function close_prox(y, ref, u)
  T = eltype(y)
  b = tol(T, length(u), maximum(abs, u; init = zero(T)))
  all(i -> abs(big(y[i]) - ref[i]) <= b, eachindex(y))
end
# the conditions of the clip of the group u with output y and radius r: ‖u - y‖₁ = r (tolerance TOL_C n eps ‖u‖₁), and
# ⟨u - y, y⟩ = r ‖y‖∞ (the same tolerance times ‖y‖∞); y = 0 needs ‖u‖₁ ≈ r or below
function kkt_ok(y, u, γ, λ0, w)
  setprecision(BigFloat, PREC) do
    T, n = eltype(y), length(u)
    r = big(γ) * big(λ0) * big(w)
    s1 = sum((abs(big(v)) for v in u); init = big(0))
    d = sum((abs(big(u[i]) - big(y[i])) for i in eachindex(u)); init = big(0))
    ym = big(maximum(abs, y; init = zero(T)))
    ip = sum(((big(u[i]) - big(y[i])) * big(y[i]) for i in eachindex(u)); init = big(0))
    b = TOL_C * n * big(eps(T)) * s1 + 2 * big(nextfloat(zero(T)))
    (r >= s1 ? d <= r + b : abs(d - r) <= b) && abs(ip - r * ym) <= b * ym
  end
end
# the value at the stored output against λ0 Σ wₖ τₖ over the penalized groups: the tolerances of the levels, and the
# rounding of a sum of K terms
function close_value(v, x, groups, w, λ0, τs)
  T = eltype(x)
  setprecision(BigFloat, PREC) do
    K = length(groups)
    pen = [k for k in 1:K if !iszero(w[k])]
    ref = sum((big(λ0) * big(w[k]) * τs[k] for k in pen); init = big(0))
    b = sum((big(λ0) * big(w[k]) * tol(T, length(groups[k]), maximum(abs, x[groups[k]]; init = zero(T))) for k in pen); init = big(0))
    b += (K + 8) * big(eps(T)) * abs(ref) + 2K * big(nextfloat(zero(T)))
    abs(big(v) - ref) <= b
  end
end
# a value within k eps of the reference (two subnormal spacings below floatmin); a reference beyond floatmax(T)
# by more than rounding must give Inf
function close_ref_value(v, ref; k = 8)
  T = typeof(v)
  abs(ref) > big(floatmax(T)) * (1 + big(eps(T))) && return v == copysign(T(Inf), ref)
  abs(big(v) - ref) <= k * big(eps(T)) * abs(ref) + 2 * big(nextfloat(zero(T)))
end

# seeded data: uniform, heavy-tailed and with many ties
function gen(rng, T, n, kind)
  kind == 1 && return T[2 * rand(rng, T) - 1 for _ in 1:n]
  kind == 2 && return T[(rand(rng, Bool) ? 1 : -1) * exp(3 * randn(rng, T)) for _ in 1:n]
  T[rand(rng, (T(-2), T(-1), T(0), T(1), T(2), T(3))) for _ in 1:n]
end

# groups of the given sizes: ranges with gaps between them, or random index vectors in which the indices left over stay in no
# group; the argument length is past the largest index of a group
function partition(rng, sizes, layout)
  if layout === :range
    groups = UnitRange{Int}[]
    start = 1
    for s in sizes
      start += rand(rng, 0:2)
      push!(groups, start:start + s - 1)
      start += s
    end
    return groups, start + 2
  end
  total = sum(sizes) + 5
  pool = randperm(rng, total)
  groups = Vector{Int}[]
  pos = 0
  for s in sizes
    push!(groups, pool[pos + 1:pos + s])
    pos += s
  end
  groups, total + 2
end

# the weights that give each group a radius γλ0wₖ that is a fraction of its ℓ1 norm
fraction_weights(rng, T, x, groups, γλ0, fracs) =
  T[T(rand(rng, fracs) * Float64(sum(abs, x[G])) / Float64(γλ0)) for G in groups]

# equal as bit patterns (the sign of a zero and the payload of a NaN included)
bits(v::Float32) = reinterpret(UInt32, v)
bits(v::Float64) = reinterpret(UInt64, v)
same(a, b) = length(a) == length(b) && all(i -> bits(a[i]) == bits(b[i]), eachindex(a))
# equal as bit patterns, a NaN being equal to any NaN
same_nan(a, b) = length(a) == length(b) && all(i -> (isnan(a[i]) && isnan(b[i])) || bits(a[i]) == bits(b[i]), eachindex(a))
same_value(a, b) = (isnan(a) && isnan(b)) || a === b

# the expected output of one group at the endpoint rules, in their order; nothing when the clip decides
function row_prox(u::Vector{T}, λ0, w, γ) where {T}
  iszero(w) && return copy(u)
  (isinf(w) || isinf(λ0)) && return zeros(T, length(u))
  (iszero(λ0) || iszero(γ)) && return copy(u)
  isinf(γ) && return zeros(T, length(u))
  any(isnan, u) && return fill(T(NaN), length(u))
  any(isinf, u) && return copy(u)
  nothing
end
# the expected value of one group at the endpoint rules, nothing when it is the product of the three factors
function row_value(u::Vector{T}, λ0, w) where {T}
  iszero(w) && return zero(T)
  (iszero(λ0) && !isinf(w)) && return zero(T)
  any(isnan, u) && return T(NaN)
  all(iszero, u) && return zero(T)
  (isinf(w) || isinf(λ0) || any(isinf, u)) && return T(Inf)
  nothing
end

# a sweep of the compact operator over the groups
function sweep!(outs, f, U, γ)
  for k in eachindex(U)
    prox_block!(outs[k], f, U[k], k, γ)
  end
  nothing
end

@testset "construction, traits and validation" begin
  for T in (Float32, Float64)
    f = GroupNormLinf(T(0.5), [1:2, [3, 5, 7]])
    @test f isa GroupNormLinf{T}
    @test f.w == ones(T, 2)                              # default weights 1
    @test f.maxlen == 3
    @test GroupNormLinf(T(0.5), [1:2], T[3]).w == T[3]
    # owned copies: changing the caller's arrays leaves f unchanged
    w = T[1, 2]; G = [[2, 1], [4, 3]]
    g = GroupNormLinf(one(T), G, w)
    w[1] = 5; G[1][1] = 9
    @test g.w == T[1, 2] && collect(g.groups[1]) == [2, 1]
    @test isproximable(g) && isconvex(g) && !issmooth(g) && !isseparable(g) && !iszeropreserving(g)
    @test GroupNormLinf(one(T), [1:2, 3:3]).groups isa Vector{UnitRange{Int}}
  end
  @test GroupNormLinf(1, [1:2]) isa GroupNormLinf{Float64}
  @test GroupNormLinf(1.0, Vector{Vector{Int}}()).maxlen == 0
  @test_throws DimensionMismatch GroupNormLinf(1.0, [1:2, 3:4], [1.0])
  @test_throws ArgumentError GroupNormLinf(-1.0, [1:2])
  @test_throws ArgumentError GroupNormLinf(NaN, [1:2])
  @test_throws ArgumentError GroupNormLinf(1.0, [1:2], [-1.0])
  @test_throws ArgumentError GroupNormLinf(1.0, [1:2], [NaN])
  @test_throws ArgumentError GroupNormLinf(1.0, [Int[]])
  @test_throws ArgumentError GroupNormLinf(1.0, [[0, 1]])
  @test_throws ArgumentError GroupNormLinf(1.0, [[1, 2], [2, 3]])   # overlapping
  @test_throws ArgumentError GroupNormLinf(1.0, [[1, 1]])           # repeated
  @test_throws ArgumentError GroupNormLinf(1.0, [1:2], [1.0 + 0im])
  @test_throws ArgumentError GroupNormLinf(1.0, [OffsetVector([1, 2], 0:1)])
end

@testset "analytic cases ($T)" for T in (Float32, Float64)
  f = GroupNormLinf(one(T), [1:2])
  @test f(T[3, -1]) === T(3) && f(T[3, 0, 5]) === T(3)             # the third entry is in no group
  @test GroupNormLinf(one(T), [1:4])(T[3 0; 0 -4]) === T(4)
  @test prox(f, T[3, -1], one(T)) == T[2, -1]                  # τ = 2
  @test value_and_prox!(zeros(T, 2), f, T[3, -1], one(T)) === T(2)
  @test prox(f, T[2, 1], T(2)) == T[0.5, 0.5]                  # τ = 0.5
  @test prox(f, T[3, -1], T(4)) == T[0, 0]                     # γλ0w = ‖x‖₁
  @test prox(f, T[3, -1], T(5)) == T[0, 0]
  @test prox(f, T[2, 2, 2], T(3)) == T[0.5, 0.5, 2]            # τ = 0.5; the third entry is in no group
  @test prox(f, T[2, 2, 2], T(4)) == T[0, 0, 2]
  h = GroupNormLinf(T(0.5), [1:3], T[2])
  @test prox(h, T[4, -4, 1], T(4)) == T[2, -2, 1]              # Σ(|xᵢ| - 2)₊ = 4 = γλ0w
  # two groups of weights 1 and 2, an entry in no group, a permuted group
  g = GroupNormLinf(T(0.5), [[3, 1], 4:5], T[1, 2])
  x = T[3, 9, 1, 4, -4, 7]
  @test g(x) === T(5.5)                                        # 0.5 ⋅ 1 ⋅ 3 + 0.5 ⋅ 2 ⋅ 4
  @test prox(g, x, one(T)) == T[2.5, 9, 1, 3.5, -3.5, 7]       # radii 0.5 (τ = 2.5) and 1 (τ = 3.5)
  # integer level, weights and step
  @test prox(GroupNormLinf(1, [1:2]), T[3, -1], 1) == T[2, -1]
  @test GroupNormLinf(2, [1:2], [1])(T[1, -3]) === T(6)
  @test prox(GroupNormLinf(1.0, [1:2]), [3, -1], 1) == [2.0, -1.0]
  @test GroupNormLinf(1.0, [1:2])([typemin(Int), 3]) == 2.0^63
  # an empty argument and an empty list of groups
  e = GroupNormLinf(one(T), Vector{Vector{Int}}())
  @test e(T[]) === zero(T) && e(T[1, 2]) === zero(T) && prox(e, T[1, 2], one(T)) == T[1, 2]
  @test value_and_prox!(T[], e, T[], one(T)) === zero(T)
  @test size(prox(e, zeros(T, 0, 3), one(T))) == (0, 3)
  # parameters in a wider type than the entries: the radius is formed in the wider type, the output is in the type of y
  d = GroupNormLinf(0.7, [1:3], [1.0])
  xd = T[4, -4, 1]
  yd = prox(d, xd, 1.0)
  refd, _ = ref_prox(xd, [1:3], [1.0], 1.0, 0.7)
  @test eltype(yd) == T && close_prox(yd, refd, xd)
end

@testset "one group equals NormLinf ($T)" for T in (Float32, Float64)
  rng = Xoshiro(956)
  λ = T(0.7)
  checked = 0
  # Condat's pass decides these (at most 1024 candidates), with a normal γλ and values in the ordinary range: the level,
  # the clip and the value agree bit for bit
  for n in (1, 2, 3, 5, 16, 17, 100, 300, 1000), kind in 1:3
    x = gen(rng, T, n, kind)
    s1 = sum(abs, x)
    for frac in (1e-9, 1e-6, 1e-3, 0.05, 0.3, 0.7, 0.95, 1.05, 1.5)
      γ = T(frac * Float64(s1) / Float64(λ))
      f, h = GroupNormLinf(λ, [1:n]), NormLinf(λ)
      y, yh = zeros(T, n), zeros(T, n)
      v, vh = value_and_prox!(y, f, x, γ), value_and_prox!(yh, h, x, γ)
      @test same(y, yh) && v === vh
      @test same(prox(f, x, γ), yh) && f(y) === h(yh) && f(x) === h(x)
      xi = copy(x)
      @test prox!(xi, f, xi, γ) === xi && same(xi, yh)
      checked += 1
    end
  end
  @test checked >= 200
  # larger groups, where Michelot's pass may decide: within the bound of the level
  for n in (5000, 20000)
    x = gen(rng, T, n, 1)
    s1 = sum(abs, x)
    for frac in (0.3, 0.9)
      γ = T(frac * Float64(s1) / Float64(λ))
      y = prox(GroupNormLinf(λ, [1:n]), x, γ)
      ref, _ = ref_prox(x, [1:n], [1], γ, λ)
      @test close_prox(y, ref, x)
    end
  end
  # the endpoint rows of NormLinf (zero and infinite levels and steps, NaN and infinite entries), with the weight 1
  inputs = (T[1, NaN, -Inf, -0.0, 3], T[-0.0, 2], T[Inf, 1], T[0, 0, 0], T[1, 2, 3], T[NaN, 1], T[1e30, -1e30])
  for λ1 in (zero(T), T(1e-30), one(T), T(1e30), floatmax(T), T(Inf)), γ in (zero(T), T(0.5), one(T), T(1e30), floatmax(T), T(Inf)), x in inputs
    f, h = GroupNormLinf(λ1, [1:length(x)]), NormLinf(λ1)
    y, yh = fill(T(7), length(x)), fill(T(7), length(x))
    v, vh = value_and_prox!(y, f, x, γ), value_and_prox!(yh, h, x, γ)
    @test same_nan(y, yh) && same_value(v, vh)
    @test same_nan(prox(f, x, γ), prox(h, x, γ)) && same_value(f(x), h(x))
  end
end

@testset "singleton groups equal NormL1 ($T)" for T in (Float32, Float64)
  rng = Xoshiro(957)
  for n in (1, 7, 64), kind in 1:3, λ in (T(0.3), T(2)), γ in (T(0.5), one(T), T(4), zero(T))
    x = gen(rng, T, n, kind)
    f = GroupNormLinf(λ, [[i] for i in 1:n])
    h = NormL1(λ)
    @test same(prox(f, x, γ), prox(h, x, γ))
    y, yh = zeros(T, n), zeros(T, n)
    v, vh = value_and_prox!(y, f, x, γ), value_and_prox!(yh, h, x, γ)
    @test same(y, yh) && abs(big(v) - big(vh)) <= 8 * n * big(eps(T)) * abs(big(vh)) + 2 * big(nextfloat(zero(T)))
    @test abs(big(f(x)) - big(h(x))) <= 8 * n * big(eps(T)) * abs(big(h(x))) + 2 * big(nextfloat(zero(T)))
  end
  # weights: the weighted NormL1, a zero weight included
  w = T[0.5, 2, 0, 1.5, 3]
  x = T[3, -4, 5, 0.1, -2]
  f, h = GroupNormLinf(T(0.8), [[i] for i in 1:5], w), NormL1(T(0.8), w)
  @test same(prox(f, x, T(0.7)), prox(h, x, T(0.7)))
end

@testset "irregular partitions against the BigFloat reference ($T, $layout)" for T in (Float32, Float64), layout in (:range, :perm)
  rng = Xoshiro(958)
  nclipped = nzero = ncopied = 0
  for trial in 1:60
    K = rand(rng, 2:9)
    sizes = [rand(rng, 1:64) for _ in 1:K]
    trial % 10 == 1 && (sizes[1] = 1)
    trial % 10 == 2 && (sizes[end] = 64)
    groups, p = partition(rng, sizes, layout)
    x = gen(rng, T, p, rand(rng, 1:3))
    λ0, γ = T(0.75), T(1.3)
    # the radius of each group is a fraction of its ℓ1 norm: clipped for the fractions below 1, and zero past it
    w = fraction_weights(rng, T, x, groups, γ * λ0, (0.01, 0.3, 0.7, 0.95, 1.05))
    trial % 7 == 3 && (w[1] = zero(T))
    trial % 7 == 4 && (w[end] = T(Inf))
    f = GroupNormLinf(λ0, groups, w)
    ref, τs = ref_prox(x, groups, w, γ, λ0)
    y = zeros(T, p)
    v = value_and_prox!(y, f, x, γ)
    @test close_prox(y, ref, x, groups)
    @test same_value(v, f(y))
    @test same(prox(f, x, γ), y) && same(prox!(similar(x), f, x, γ), y)
    xi = copy(x)
    @test prox!(xi, f, xi, γ) === xi && same(xi, y)
    @test value_and_prox!(copy(x), f, x, γ) === v
    for k in eachindex(groups)
      G = groups[k]
      isinf(w[k]) || @test kkt_ok(y[G], x[G], γ, λ0, w[k])
      nclipped += !iszero(w[k]) && !isinf(w[k]) && any(!iszero, y[G])
      nzero += all(iszero, y[G]) && !iszero(w[k])
      ncopied += iszero(w[k])
    end
    if !isinf(w[end])
      @test close_value(v, x, groups, w, λ0, τs)
      # the value of the argument
      refx = setprecision(BigFloat, PREC) do
        sum((big(λ0) * big(w[k]) * maximum(abs, big.(x[groups[k]])) for k in eachindex(groups) if !iszero(w[k])); init = big(0))
      end
      @test close_ref_value(f(x), refx; k = length(groups) + 8)
    end
    # a Matrix argument has the same groups as linear indices
    X = reshape(vcat(x, zeros(T, mod(-p, 4))), 4, :)
    Y = zeros(T, size(X))
    prox!(Y, f, X, γ)
    @test same(vec(Y)[1:p], y)
  end
  @test nclipped > 20 && nzero > 5 && ncopied > 3
  # a group of 5000 entries without ties: more than 1024 candidates, so that Michelot's pass decides
  n = 5000
  x = T[rand(rng, T) for _ in 1:n]
  s1 = sum(abs, x)
  λ0, γ = T(1), T(1)
  f = GroupNormLinf(λ0, [1:n, n+1:n+3], T[T(0.5) * s1, 1])
  xx = vcat(x, T[3, -1, 0.5])
  ref, τs = ref_prox(xx, f.groups, f.w, γ, λ0)
  y = prox(f, xx, γ)
  @test close_prox(y, ref, xx, f.groups)
  @test kkt_ok(y[1:n], xx[1:n], γ, λ0, f.w[1]) && kkt_ok(y[n+1:n+3], xx[n+1:n+3], γ, λ0, f.w[2])
  if Base.JLOptions().check_bounds == 1
    # the pass hands over at max(1024, n ÷ 8) candidates: a buffer of that length suffices for it, and the call returns
    # Michelot's level (checked where a write past the buffer throws)
    buf = Vector{T}(undef, 1024)
    xv = PB._IndexedValues(xx, f.groups[1], T)
    r = T(0.5) * s1
    τ = PB._l1ball_threshold!(buf, xv, r, PB._AbsRead{T}())
    @test τ == PB._l1ball_michelot(xv, r, PB._AbsRead{T}(), T)
  end
end

@testset "large ties through the shared threshold ($T)" for T in (Float32, Float64)
  # #987 puts the level just below a long tie. #1003 puts it above the tie, at the level of the head alone.
  # The final case reaches the handover with the sum in scaled units.
  M = floatmax(T)
  for (u, r, lower) in (([T(0.2); fill(T(0.1), 100_000)], T(0.3), T(0.09)),
                       ([fill(T(0.17933106), 100_000); one(T)], T(0.5), T(0.49)),
                       (fill(M / T(8), 4096), M, M / T(10)))
    n = length(u)
    f = GroupNormLinf(r, [1:n])
    ref, _ = ref_prox(u, f.groups, f.w, one(T), r)
    y = prox(f, u, one(T))
    @test close_prox(y, ref, u) && maximum(y) > lower
    ui = copy(u)
    @test same(prox!(ui, f, ui, one(T)), y)
    out = similar(u)
    @test close_prox(prox_block!(out, f, u, 1, one(T)), ref, u)
    @test same_value(value_and_prox!(out, f, u, one(T)), f(out)) && same(out, y)
  end
end

@testset "compact blocks equal the array operators ($T)" for T in (Float32, Float64)
  rng = Xoshiro(959)
  groups, p = partition(rng, [1, 2, 5, 17, 40], :perm)
  λ0, γ = T(0.9), T(1.1)
  x = gen(rng, T, p, 2)
  w = fraction_weights(rng, T, x, groups, γ * λ0, (0.1, 0.4, 0.8))
  w[2] = zero(T)
  f = GroupNormLinf(λ0, groups, w)
  y = prox(f, x, γ)
  for k in eachindex(groups)
    G = groups[k]
    u = x[G]
    out = prox_block(f, u, k, γ)
    @test same(out, y[G])
    out2 = fill(T(7), length(G))
    @test prox_block!(out2, f, u, k, γ) === out2 && same(out2, y[G])
    ui = copy(u)
    @test prox_block!(ui, f, ui, k, γ) === ui && same(ui, y[G])
    # a view of an array as the input, a view as the output
    big_u = vcat(u, u)
    @test same(prox_block!(fill(T(7), length(G)), f, view(big_u, 1:length(G)), k, γ), y[G])
    big_out = fill(T(7), 2length(G))
    ov = view(big_out, length(G)+1:2length(G))
    @test prox_block!(ov, f, u, k, γ) === ov && same(ov, y[G]) && all(==(T(7)), view(big_out, 1:length(G)))
  end
  # an equal view as the output and the input is the same buffer
  buf = T[3, 1, 7]
  v = view(buf, 1:2)
  f1 = GroupNormLinf(one(T), [1:2])
  expected = prox(f1, T[3, 1], T(0.5))
  @test prox_block!(v, f1, view(buf, 1:2), 1, T(0.5)) == expected && buf == T[expected; 7]
  # an output of another element type, and an integer input
  o64 = zeros(Float64, 2)
  @test prox_block!(o64, f1, T[3, -1], 1, one(T)) == [2.0, -1.0]
  @test prox_block(f1, [3, -1], 1, 1) == [2.0, -1.0]
end

@testset "invalid calls fail before writes ($T)" for T in (Float32, Float64)
  f = GroupNormLinf(T(0.5), [[3, 1], 4:6], T[1, 2])        # the groups reach index 6
  x = T[3, -1, 2, 5, -4, 1]
  sentinel(n = 6) = fill(T(7), n)
  for γ in (T(-1), T(-Inf), T(NaN))
    y = sentinel()
    @test_throws ArgumentError prox!(y, f, x, γ)
    @test_throws ArgumentError value_and_prox!(y, f, x, γ)
    @test_throws ArgumentError prox(f, x, γ)
    @test y == sentinel()
    xi = copy(x)
    @test_throws ArgumentError prox!(xi, f, xi, γ)
    @test xi == x
    for k in 1:2
      out = sentinel(length(f.groups[k]))
      @test_throws ArgumentError prox_block!(out, f, x[f.groups[k]], k, γ)
      @test out == sentinel(length(out))
    end
  end
  # sizes and group extent
  y5 = sentinel(5)
  @test_throws DimensionMismatch prox!(y5, f, x, one(T))
  @test_throws DimensionMismatch value_and_prox!(y5, f, x, one(T))
  @test_throws DimensionMismatch prox!(sentinel(), f, T[1, 2, 3, 4, 5, 6, 7], one(T))
  @test y5 == sentinel(5)
  x5 = T[1, 2, 3, 4, 5]
  @test_throws DimensionMismatch prox!(sentinel(5), f, x5, one(T))
  @test_throws DimensionMismatch value_and_prox!(sentinel(5), f, x5, one(T))
  @test_throws DimensionMismatch prox(f, x5, one(T))
  @test_throws DimensionMismatch f(x5)
  @test_throws DimensionMismatch prox!(x5, f, x5, one(T))
  @test x5 == T[1, 2, 3, 4, 5]
  # complex entries and offset axes
  xc, yc = Complex{T}[1, 2, 3, 4, 5, 6], fill(Complex{T}(7), 6)
  @test_throws ArgumentError prox!(sentinel(), f, xc, one(T))
  @test_throws ArgumentError prox!(yc, f, x, one(T))
  @test_throws ArgumentError f(xc)
  @test yc == fill(Complex{T}(7), 6)
  xo = OffsetVector(copy(x), 0:5)
  y = sentinel()
  @test_throws ArgumentError prox!(y, f, xo, one(T))
  @test_throws ArgumentError value_and_prox!(y, f, xo, one(T))
  @test_throws ArgumentError prox(f, xo, one(T))
  @test_throws ArgumentError f(xo)
  yo = OffsetVector(sentinel(), 0:5)
  @test_throws ArgumentError prox!(yo, f, x, one(T))
  @test_throws ArgumentError value_and_prox!(yo, f, x, one(T))
  @test y == sentinel() && parent(yo) == sentinel()
  # an array of steps is not supported
  @test_throws MethodError prox!(sentinel(), f, x, ones(T, 6))
  # a symmetric iterate as the argument or the output: nothing is written, directly and through Scaled
  S = SymmetricSparseIterate(T[5 6; 6 7])
  D = fill(T(7), 2, 2)
  g = GroupNormLinf(one(T), [1:2])
  for h in (g, 2 * g, 2 * (3 * g))
    @test_throws ArgumentError prox!(S, h, T[1 2; 2 1], one(T))
    @test_throws ArgumentError value_and_prox!(S, h, S, one(T))
    @test_throws ArgumentError prox!(D, h, S, one(T))
    @test_throws ArgumentError value_and_prox!(D, h, S, one(T))
    @test_throws ArgumentError prox(h, S, one(T))
  end
  @test Matrix(S) == T[5 6; 6 7] && D == fill(T(7), 2, 2)
  # compact blocks: the group, the lengths, the overlap, element types and axes
  out = sentinel(2)
  u = T[3, -1]
  for k in (0, 3, -1)
    @test_throws BoundsError prox_block!(out, f, u, k, one(T))
  end
  @test_throws BoundsError prox_block!(out, f, u, true, one(T))
  @test_throws DimensionMismatch prox_block!(sentinel(3), f, u, 1, one(T))
  @test_throws DimensionMismatch prox_block!(out, f, T[1, 2, 3], 1, one(T))
  @test_throws DimensionMismatch prox_block!(out, f, u, 2, one(T))                # group 2 has three entries
  v = T[1, 2, 3, 4]
  vv = copy(v)
  @test_throws ArgumentError prox_block!(view(v, 1:2), f, view(v, 2:3), 1, one(T))
  @test_throws ArgumentError prox_block!(view(v, 2:3), f, view(v, 1:2), 1, one(T))
  @test v == vv
  @test_throws ArgumentError prox_block!(Complex{T}[7, 7], f, u, 1, one(T))
  @test_throws ArgumentError prox_block!(out, f, Complex{T}[1, 2], 1, one(T))
  @test_throws ArgumentError prox_block!(OffsetVector(sentinel(2), 0:1), f, u, 1, one(T))
  @test_throws ArgumentError prox_block!(out, f, OffsetVector(u, 0:1), 1, one(T))
  @test out == sentinel(2)
end

@testset "endpoint rows: parameters, NaN and infinite entries ($T)" for T in (Float32, Float64)
  # group 1 holds the tested entries and has the tested weight, group 2 an ordinary pair of weight 1: the rules apply to each
  # group, the clip decides where no row does, and a NaN or infinite group leaves the other groups to their own rules
  M = floatmax(T)
  δ = nextfloat(zero(T))
  groupsets = (T[3, -1, 2], T[NaN, 1, 2], T[Inf, 1, 2], T[-Inf, NaN, 2], T[0, 0, 0], T[-0.0, 2, 0], T[1, 2, NaN])
  other = T[10, -20]
  nchecked = 0
  for λ0 in (zero(T), T(0.5), one(T), T(Inf), M, δ), w1 in (zero(T), one(T), T(2), T(Inf), M), γ in (zero(T), T(0.25), one(T), T(Inf)), u in groupsets
    f = GroupNormLinf(λ0, [1:3, 4:5], T[w1, 1])
    x = vcat(u, other)
    y = fill(T(7), 5)
    v = value_and_prox!(y, f, x, γ)
    # the output of each group
    for (k, uk) in ((1, u), (2, other))
      Gk = f.groups[k]
      expected = row_prox(uk, λ0, f.w[k], γ)
      if expected !== nothing
        @test same_nan(y[Gk], expected)
      else
        τ = ref_group_level(uk, γ, λ0, f.w[k])
        r = setprecision(() -> [clamp(big(t), -τ, τ) for t in uk], BigFloat, PREC)
        b = tol(T, length(uk), maximum(abs, uk; init = zero(T)))
        @test all(i -> abs(big(y[Gk[i]]) - r[i]) <= b, eachindex(uk))
      end
    end
    # the value at the stored output, which the fused call returns
    @test same_value(v, f(y))
    xi = copy(x)
    @test same_value(value_and_prox!(xi, f, xi, γ), v) && same_nan(xi, y)
    @test same_nan(prox(f, x, γ), y)
    nchecked += 1
  end
  @test nchecked == 6 * 5 * 4 * 7
  # the value of a group: where row_value gives a value, it is the same value; otherwise the exact product of the three
  # factors is the reference of close_ref_value(...; k = 2), which allows 2 eps(T) |reference| plus two subnormal spacings,
  # and requires Inf for a reference beyond floatmax(T) by more than rounding
  for λ0 in (zero(T), T(0.5), T(Inf), M, δ), w in (zero(T), one(T), T(3), T(Inf), M), u in groupsets
    f = GroupNormLinf(λ0, [1:3], T[w])
    r = row_value(u, λ0, w)
    if r !== nothing
      @test same_value(f(u), r)
    else
      m = maximum(abs, u)
      @test close_ref_value(f(u), setprecision(() -> big(λ0) * big(w) * big(m), BigFloat, PREC); k = 2)
    end
  end
  # a zero group of any level and weight has the value zero
  @test GroupNormLinf(T(Inf), [1:2], T[Inf])(zeros(T, 2)) === zero(T)
  @test GroupNormLinf(T(Inf), [1:2], T[Inf])(T[-0.0, 0.0]) === zero(T)
end

@testset "combined endpoint rows ($T)" for T in (Float32, Float64)
  M = floatmax(T)
  u = T[3, -1, 2]
  nan_group, inf_group = T[1, NaN, 2], T[Inf, 1, 2]
  # a zero weight copies the group also at an infinite level or step: NaN and Inf entries included
  for x in (u, nan_group, inf_group), (λ0, γ) in ((T(Inf), one(T)), (one(T), T(Inf)), (T(Inf), T(Inf)), (T(Inf), zero(T)))
    f = GroupNormLinf(λ0, [1:3], T[0])
    @test same_nan(prox(f, x, γ), x)
    @test value_and_prox!(similar(x), f, x, γ) === zero(T)
  end
  # an infinite weight, or an infinite level at a positive weight, gives +0 at the steps 0, 1 and Inf
  for x in (u, nan_group, inf_group, zeros(T, 3)), γ in (zero(T), one(T), T(Inf))
    for f in (GroupNormLinf(zero(T), [1:3], T[Inf]), GroupNormLinf(one(T), [1:3], T[Inf]), GroupNormLinf(T(Inf), [1:3], T[2]), GroupNormLinf(T(Inf), [1:3], T[Inf]))
      y = fill(T(7), 3)
      v = value_and_prox!(y, f, x, γ)
      @test all(iszero, y) && all(!signbit, y) && v === zero(T)
      # the value at the argument: zero at a zero group, NaN at a NaN entry, Inf otherwise (a zero level with an infinite weight included)
      fx = f(x)
      @test (all(iszero, x) ? fx === zero(T) : any(isnan, x) ? isnan(fx) : fx === T(Inf))
    end
  end
  # a zero level or a zero step is the identity (the weight finite and positive), at an infinite step too when the level is zero
  for x in (u, nan_group, inf_group)
    @test same_nan(prox(GroupNormLinf(zero(T), [1:3], T[2]), x, T(Inf)), x)
    @test same_nan(prox(GroupNormLinf(zero(T), [1:3], T[2]), x, one(T)), x)
    @test same_nan(prox(GroupNormLinf(one(T), [1:3], T[2]), x, zero(T)), x)
    @test GroupNormLinf(zero(T), [1:3], T[2])(x) === zero(T)
  end
  # a finite level and weight whose product overflows, at a zero step: the identity, and the value is Inf
  f = GroupNormLinf(M, [1:3], T[4])
  @test same(prox(f, u, zero(T)), u)
  @test value_and_prox!(similar(u), f, u, zero(T)) === T(Inf)
  # an infinite step with positive finite parameters gives +0
  y = prox(GroupNormLinf(one(T), [1:3], T[2]), u, T(Inf))
  @test all(iszero, y) && all(!signbit, y)
  # a NaN entry makes its group NaN and leaves the others alone
  g = GroupNormLinf(one(T), [1:3, 4:6, 7:9], T[1, 1, 1])
  one_group = GroupNormLinf(one(T), [1:3])
  xg = T[1, NaN, 2, 5, 1, 2, 3, -4, 6]
  y = prox(g, xg, one(T))
  @test all(isnan, y[1:3]) && same(y[4:6], prox(one_group, xg[4:6], one(T))) && same(y[7:9], prox(one_group, xg[7:9], one(T)))
  xi = T[1, 2, Inf, 5, 1, 2, 3, -4, 6]
  yi = prox(g, xi, one(T))
  @test same(yi[1:3], xi[1:3]) && same(yi[4:6], y[4:6]) && same(yi[7:9], y[7:9])
  # a NaN in one group and an infinite entry in another: independent
  xm = T[1, NaN, 2, Inf, 1, 2, 3, -4, 6]
  ym = prox(g, xm, one(T))
  @test all(isnan, ym[1:3]) && same(ym[4:6], xm[4:6]) && same(ym[7:9], y[7:9])
end

@testset "range: overflowing sums and radii, subnormal radii ($T)" for T in (Float32, Float64)
  M = floatmax(T)
  δ = nextfloat(zero(T))
  # the sum overflows and the product λ0 w alone does too: 3M/4 each (forming λ0 w first would give +0)
  f = GroupNormLinf(M, [1:2], T[2])
  y = prox(f, T[M, M], T(0.25))
  ref, _ = ref_prox(T[M, M], [1:2], T[2], T(0.25), M)
  @test close_prox(y, ref, T[M, M])
  @test abs(big(y[1]) - 3 * big(M) / 4) <= 8 * 2 * big(eps(T)) * big(M) && y[1] == y[2]
  @test value_and_prox!(zeros(T, 2), f, T[M, M], T(0.25)) === f(y)
  # the radius equals the sum although γλ0 alone underflows: +0 (the product (γλ0)w would be 0 and give the identity)
  a, b = T == Float64 ? (1000, 100) : (100, 30)
  g = GroupNormLinf(ldexp(one(T), -b), [1:2], T[ldexp(one(T), a)])
  x = T[ldexp(one(T), -b), 0]
  γ = ldexp(one(T), -a)
  @test prox(g, x, γ) == zeros(T, 2) && all(!signbit, prox(g, x, γ))
  @test value_and_prox!(zeros(T, 2), g, x, γ) === zero(T)
  # a radius of half the sum keeps half of the entry: τ = x₁ - radius
  @test prox(g, x, γ / 2) == T[ldexp(one(T), -b - 1), 0]
  # an overflowing radius with a finite sum gives +0, with an infinite entry the identity
  f = GroupNormLinf(M, [1:2], T[4])
  @test prox(f, T[3, 4], one(T)) == zeros(T, 2)
  @test prox(f, T[Inf, 1], one(T)) == T[Inf, 1]
  @test prox(GroupNormLinf(M, [1:2]), T[3, 4], M) == zeros(T, 2)
  # a sum at floatmax/2 and just beside it: the radius is a fraction of it, within the bound of the level
  for n in (2, 4, 16), d in (-3, -1, 0, 1, 3), frac in (0.2, 0.7)
    x = fill(M / (2n), n)
    for _ in 1:abs(d)
      x[1] = d < 0 ? prevfloat(x[1]) : nextfloat(x[1])
    end
    s = setprecision(() -> sum(big.(x)), BigFloat, PREC)
    gl = T(setprecision(() -> sqrt(big(frac) * s), BigFloat, PREC))
    ref, _ = ref_prox(x, [1:n], [1], gl, gl)
    @test close_prox(prox(GroupNormLinf(gl, [1:n]), x, gl), ref, x)
  end
  # entries of order floatmax with an overflowing sum, in two groups with weights
  rng = Xoshiro(960)
  for n in (2, 3, 10), kind in 1:3, frac in (0.001, 0.3, 0.9)
    x = T[(rand(rng, Bool) ? 1 : -1) * (T(0.3) + T(0.7) * rand(rng, T)) * M for _ in 1:2n]
    kind == 2 && (x[1] *= T(1e-3))
    kind == 3 && (x[n] = zero(T))
    groups = [1:n, n+1:2n]
    s1 = setprecision(() -> sum(abs.(big.(x[groups[1]]))), BigFloat, PREC)
    w = T[1, 2]
    gl = T(setprecision(() -> sqrt(big(frac) * s1), BigFloat, PREC))
    h = GroupNormLinf(gl, groups, w)
    ref, _ = ref_prox(x, groups, w, gl, gl)
    yv = zeros(T, 2n)
    v = value_and_prox!(yv, h, x, gl)
    @test close_prox(yv, ref, x, groups) && same_value(v, h(yv))
    @test same(prox!(copy(x), h, x, gl), yv)
  end
  # an overflowing group beside an ordinary one
  @test all(isfinite, prox(GroupNormLinf(one(T), [1:2, 3:4]), T[M, M, 1, 0], one(T)))
  # subnormal and tiny inputs, subnormal and tiny radii
  for x in (T[δ, 2δ, 3δ, -5δ], T[1000δ, 300δ, 20δ], T[floatmin(T), 2 * floatmin(T), 3δ], T[1, 2δ, floatmin(T)])
    for (gam, l) in ((one(T), 7δ), (T(0.5), 100δ), (one(T), one(T)), (T(2), floatmin(T)))
      hh = GroupNormLinf(l, [1:length(x)])
      ref, _ = ref_prox(x, [1:length(x)], [1], gam, l)
      @test close_prox(prox(hh, x, gam), ref, x)
    end
  end
  # A radius of one half spacing puts the exact result midway between zero and δ.
  y = prox(GroupNormLinf(δ, [1:1]), T[δ], T(0.5))
  @test abs(big(y[1]) - big(δ) / 2) == big(δ) / 2
  # a subnormal weight leaves entries of order one unchanged to rounding
  xs = T[1, 1]
  @test close_prox(prox(GroupNormLinf(one(T), [1:2], T[δ]), xs, one(T)), ref_prox(xs, [1:2], T[δ], one(T), one(T))[1], xs)
end

@testset "values: recomputed sums and the working type ($T)" for T in (Float32, Float64)
  M = floatmax(T)
  # many groups whose terms round into the subnormal range: each term (m0 + 0.4) δ rounds down by 0.4 δ, so the sum below
  # K floatmin loses about 0.4 K δ unless it is recomputed with the exponents
  m0, e0, e1 = T == Float64 ? (2^30, 1000, 74) : (2^12, 100, 49)
  for K in (32, 256)
    λ0 = ldexp(one(T), -e0)
    x = fill(ldexp(T(m0) + T(0.4), -e1), K)
    f = GroupNormLinf(λ0, [i:i for i in 1:K])
    ref = setprecision(() -> sum(big(λ0) * big(v) for v in x), BigFloat, PREC)
    @test close_ref_value(f(x), ref) && f(x) > 0
    # at the stored output too: a radius far below the entries leaves them
    y = zeros(T, K)
    @test same_value(value_and_prox!(y, f, x, T(1e-30)), f(y))
    @test close_ref_value(f(y), ref)
  end
  # a sum between floatmax/2 and floatmax, and past floatmax
  f = GroupNormLinf(one(T), [1:1, 2:2, 3:3])
  x = T[0.3, 0.3, 0.3] .* M
  @test close_ref_value(f(x), setprecision(() -> sum(big.(x)), BigFloat, PREC))
  x = T[0.6, 0.6, 0.6] .* M
  @test f(x) === T(Inf) && setprecision(() -> sum(big.(x)), BigFloat, PREC) > big(M)
  # a term whose product with the level overflows although the entry is finite
  @test GroupNormLinf(T(4), [1:1], T[M])(T[1]) === T(Inf)
  # a zero argument and zero terms
  @test GroupNormLinf(ldexp(one(T), -e0), [1:2, 3:4])(zeros(T, 4)) === zero(T)
  # a level of type Float64 with entries of type T: the product (λ0 w) m is formed in Float64 here (the weight is 1, so one
  # rounding) and converted to the type of the entries
  h = GroupNormLinf(0.7, [1:2], [1.0])
  xx = T[3, -1]
  @test h(xx) === T(0.7 * Float64(maximum(abs, xx)))
  # an integer argument
  @test GroupNormLinf(0.5, [1:2])([3, -4]) === 2.0
end

@testset "the fused value is the value at the stored output ($T)" for T in (Float32, Float64)
  rng = Xoshiro(961)
  for trial in 1:40
    K = rand(rng, 1:6)
    sizes = [rand(rng, 1:20) for _ in 1:K]
    groups, p = partition(rng, sizes, rand(rng, (:range, :perm)))
    x = gen(rng, T, p, rand(rng, 1:3))
    λ0 = rand(rng, (T(0.01), T(1), T(1e3), T(Inf), zero(T)))
    w = T[rand(rng, (zero(T), T(0.2), one(T), T(5))) for _ in 1:K]
    f = GroupNormLinf(λ0, groups, w)
    s1 = sum(abs, x)
    for γ in (zero(T), T(1e-4) * s1, T(0.1), one(T), T(2) * s1, T(Inf))
      y = zeros(T, p)
      v = value_and_prox!(y, f, x, γ)
      @test same_value(v, f(y))
      xi = copy(x)
      @test same_value(value_and_prox!(xi, f, xi, γ), v) && same_nan(xi, y)
    end
  end
  # an output of another element type: f(y) is recomputed after the conversions succeed (an integer output needs integral
  # entries; a fractional one throws an InexactError)
  f = GroupNormLinf(1.0, [1:2])
  y = zeros(Float32, 2)
  @test value_and_prox!(y, f, [3.0, -1.1], 1.0) === f(y)
  yi = zeros(Int, 2)
  @test value_and_prox!(yi, f, [3.0, -1.0], 1.0) === f(yi) && yi == [2, -1]
  @test_throws InexactError value_and_prox!(zeros(Int, 2), f, [3.0, -1.0], 0.5)
  # through Scaled, the value of the chain at the output
  g = GroupNormLinf(T(0.5), [[3, 1], 4:5], T[1, 2])
  x = T[3, 9, 1, 4, -4, 7]
  for s in (2 * g, 0.3 * g, 2 * (3 * g))
    y = zeros(T, 6)
    v = value_and_prox!(y, s, x, T(0.9))
    @test v ≈ s(y) && same_nan(y, prox(s, x, T(0.9)))
  end
end

@testset "iterates ($T)" for T in (Float32, Float64)
  f = GroupNormLinf(T(0.7), [[3, 1], [4, 2], 5:6], T[1, 2, 0.5])      # a partition of 1:6
  d = T[0, 3, 2, -4, 5, -1]
  γ = T(1.1)
  ref = prox(f, d, γ)
  B = BlockIterate(copy(d), f)
  @test B isa BlockIterate{T} && ncoordinates(B) == 3
  r = similar(B)
  @test prox!(r, f, B, γ) === r && Vector(r) == ref
  @test value_and_prox!(similar(B), f, B, γ) === value_and_prox!(similar(d), f, d, γ)
  @test f(B) === f(d)
  x = SparseIterate(copy(d))
  y = SparseIterate(T, 6)
  @test Vector(prox!(y, f, x, γ)) == ref
  @test value_and_prox!(SparseIterate(T, 6), f, x, γ) === value_and_prox!(similar(d), f, d, γ)
  @test f(x) === f(d) && f(SparseIterate(T, 6)) === zero(T)
  yd = zeros(T, 6)
  @test prox!(yd, f, x, γ) === yd && yd == ref
  @test Vector(prox!(SparseIterate(T, 6), f, d, γ)) == ref
  xs = SparseIterate(copy(d))
  @test Vector(prox!(xs, f, xs, γ)) == ref
  xb = BlockIterate(copy(d), f)
  @test Vector(prox!(xb, f, xb, γ)) == ref
  # a NaN group and a zero output through an iterate
  z = SparseIterate(T, 6)
  prox!(z, f, T[1, NaN, 2, 3, 4, 5], γ)
  @test isnan(z[4]) && isnan(z[2]) && !isnan(z[1]) && !isnan(z[5])
  @test Vector(prox!(z, f, T[1, 2, 3, 4, 5, 6], T(100))) == zeros(T, 6)
  # BlockIterate(x, f) and BlockIterate(p, f) require the groups to partition the entries
  @test BlockIterate(6, f) isa BlockIterate{Float64} && BlockIterate(T, 6, f) isa BlockIterate{T}
  part = GroupNormLinf(one(T), [1:2, 3:3])
  @test_throws ArgumentError BlockIterate(T[1, 2, 3, 4], part)
  @test_throws ArgumentError BlockIterate(4, part)
  @test_throws ArgumentError BlockIterate(2, part)
  @test_throws ArgumentError BlockIterate(3, GroupNormLinf(one(T), [1:2, 4:4]))
  @test BlockIterate(3, part) isa BlockIterate{Float64}
  # a trial block through the pure operator changes nothing; the commit activates the group
  zb = BlockIterate(6, f)
  out = prox_block(f, T[3, 4], 2, T(0.5))
  @test nactive(zb) == 0
  copyto!(blockview(zb, 2), out)
  @test collect(activecoordinates(zb)) == [2] && Vector(zb)[[4, 2]] == out
  @test allocs(f, B) == 0 && allocs(f, x) == 0
end

@testset "the buffer of the threshold ($T)" for T in (Float32, Float64)
  # parameter exits, NaN and infinite groups and a zero fill decided before the threshold do not touch the buffer of f; the
  # first group that reaches the threshold allocates it at the length of the longest group, and it is reused
  groups = [1:3, 4:4, 5:12]
  x = T[3, 1, 2, 5, 1, 2, 3, 4, 5, 6, 7, 8]
  f = GroupNormLinf(one(T), groups, T[1, 1, 1])
  @test f.maxlen == 8 && f.scratch.buf === nothing
  y = similar(x)
  prox!(y, f, x, zero(T)); prox!(y, f, x, T(Inf)); prox!(y, f, x, T(100))
  @test f.scratch.buf === nothing
  xn = copy(x); xn[1] = T(NaN); xn[5] = T(Inf)
  prox!(y, f, xn, T(6))                              # group 2 is zero-filled: its radius reaches its sum
  @test f.scratch.buf === nothing
  prox!(y, f, x, T(0.5))
  b = f.scratch.buf
  @test b isa Vector{T} && length(b) == 8
  prox!(y, f, x, T(1.5)); value_and_prox!(y, f, x, T(0.25))
  @test f.scratch.buf === b
  # a compact block: a distinct Array output of the output type, with an Array input, is the buffer, so the scratch stays unused
  g = GroupNormLinf(one(T), groups, T[1, 1, 1])
  expected = prox(GroupNormLinf(one(T), [1:3]), T[3, 1, 2], T(0.5))
  out = zeros(T, 3)
  prox_block!(out, g, T[3, 1, 2], 1, T(0.5))
  @test g.scratch.buf === nothing && out == expected
  # an input that is neither an Array nor a SubArray of an Array (an iterate) does not qualify: the scratch is used
  gs = GroupNormLinf(one(T), groups, T[1, 1, 1])
  prox_block!(zeros(T, 3), gs, SparseIterate(T[3, 1, 2]), 1, T(0.5))
  @test gs.scratch.buf isa Vector{T} && length(gs.scratch.buf) == 8
  # in place, the scratch of length maxlen serves the groups of different lengths
  u = T[3, 1, 2]
  prox_block!(u, g, u, 1, T(0.5))
  @test g.scratch.buf isa Vector{T} && length(g.scratch.buf) == 8 && u == expected
  # in place in another element type: the scratch has the element type of the output
  g2 = GroupNormLinf(one(T), groups, T[1, 1, 1])
  u64 = Float64[3, 1, 2]
  prox_block!(u64, g2, u64, 1, T(0.5))
  @test g2.scratch.buf isa Vector{Float64} && u64 == Float64.(expected)
end

@testset "Scaled ($T)" for T in (Float32, Float64)
  rng = Xoshiro(962)
  groups, p = partition(rng, [1, 3, 8, 20], :perm)
  x = gen(rng, T, p, 1)
  w = fraction_weights(rng, T, x, groups, one(T), (0.2, 0.5, 0.9))
  f = GroupNormLinf(one(T), groups, w)
  γ = T(0.8)
  # a power of two folds into the level without rounding
  @test same(prox(2 * f, x, γ), prox(GroupNormLinf(T(2), groups, w), x, γ))
  @test same(prox(f * 2, x, γ), prox(f, x, 2γ))
  @test (2 * f)(x) == 2 * f(x)
  # another factor: within the bound of the level for the product radius
  α = T(0.7)
  ref, _ = ref_prox(x, groups, w, α * γ, one(T))
  @test close_prox(prox(α * f, x, γ), ref, x, groups)
  @test (α * f)(x) ≈ α * f(x)
  @test value_and_prox!(zeros(T, p), α * f, x, γ) ≈ (α * f)(prox(α * f, x, γ))
  # a zero factor has no representation for this function
  @test_throws ArgumentError 0 * f
end

# ---- the score of a group: references and data ----
# The references are BigFloat evaluations of the definitions, by sorting, for the group G of the point (v, x) read in
# W = float(promote_type(eltype(v), eltype(x))) and the coefficient c = big(λ0) big(w), the exact product of the stored level and weight;
# none calls a kernel or a coefficient helper of the package. With J the set of the entries of x[G] of largest magnitude (equality
# after the conversion to W) and uᵢ = -sign(xᵢ) vᵢ:
#   x[G] = 0:  the margin ‖v[G]‖₁ - c (four-argument form), and the distance from v[G] to the ℓ1 ball of radius c (full-point form),
#              sqrt(Σ min(|vᵢ|, τ)²) with Σ (|vᵢ| - τ)₊ = c, and 0 when ‖v[G]‖₁ ≤ c;
#   x[G] ≠ 0:  the distance from -v[G] to the subdifferential of c‖·‖∞ at x[G], sqrt(Σ_{i∉J} vᵢ² + Σ_{i∈J} min(uᵢ, τ)²) with
#              Σ_{i∈J} (uᵢ - τ)₊ = c (τ can be negative);
#   c = 0:     ‖v[G]‖₂ in both cases.
# A value is accepted within (2n + √n + 5) u (‖v[G]‖₂ + c) + 4 n nextfloat(zero(W)) of its reference, u = eps(W)/2, the first-order bound
# of the kernel plus the subnormal spacing of W.
kk_sgn(x) = x > 0 ? one(x) : x < 0 ? -one(x) : zero(x)
kk_msg(f) = try f(); "" catch e; sprint(showerror, e) end
kk_close(a, b, ::Type{T}, s) where {T} = abs(a - b) <= 16 * eps(T) * s

# the level τ with Σ (uᵢ - τ)₊ = c, c > 0, by sorting
function kk_level(u::Vector{BigFloat}, c::BigFloat)
  s = sort(u; rev = true)
  acc = zero(BigFloat)
  τ = s[1]
  for j in eachindex(s)
    acc += s[j]
    t = (acc - c) / j
    s[j] > t || break
    τ = t
  end
  τ
end
# the distance from -V to the subdifferential at X of c max|xᵢ|, for BigFloat vectors of finite entries and a finite c ≥ 0
function kk_dist(c::BigFloat, V::Vector{BigFloat}, X::Vector{BigFloat})
  iszero(c) && return sqrt(sum(abs2, V))
  m = maximum(abs, X)
  if iszero(m)
    sum(abs, V) <= c && return zero(BigFloat)
    τ = kk_level(abs.(V), c)
    return sqrt(sum(abs2, [kk_sgn(t) * min(abs(t), τ) for t in V]))
  end
  J = abs.(X) .== m
  u = [-kk_sgn(X[i]) * V[i] for i in eachindex(V)]
  τ = kk_level(u[J], c)
  sqrt(sum(abs2, [J[i] ? min(u[i], τ) : V[i] for i in eachindex(V)]))
end
kk_linf_bound(n) = 2n + sqrt(n) + 5
function kk_bound(::Type{W}, V, c, n; slack = 1) where {W}
  slack * (big(kk_linf_bound(n)) * big(eps(W)) / 2 * (sqrt(sum(abs2, V)) + c) + 4n * big(nextfloat(zero(W))))
end
# the entries of the group read in W, and the reference of the distance with its bound
function kk_entries(v, x, G, ::Type{W}) where {W}
  [big(convert(W, v[i])) for i in G], [big(convert(W, x[i])) for i in G]
end
function kk_ref(λ0, w, v, x, G; slack = 1)
  W = float(promote_type(eltype(v), eltype(x)))
  setprecision(BigFloat, PREC) do
    c = BigFloat(λ0) * BigFloat(w)
    V, X = kk_entries(v, x, G, W)
    kk_dist(c, V, X), kk_bound(W, V, c, length(G); slack)
  end
end
# r is the distance for the group G: of the type W, within the bound of the reference where that is in the range of W, and Inf where the
# reference is above floatmax(W); the band of relative width 100 eps(W) around floatmax(W) is not decided. With far, the reference
# must also exceed 10 times the bound, so that a zero return fails.
function kk_ok(r, λ0, w, v, x, G; slack = 1, far = false)
  W = float(promote_type(eltype(v), eltype(x)))
  r isa W || return false
  setprecision(BigFloat, PREC) do
    ref, bound = kk_ref(λ0, w, v, x, G; slack)
    far && !(ref > 10 * bound) && return false
    lim, e = big(floatmax(W)), 100 * big(eps(W))
    ref > lim * (1 + e) && return r == W(Inf)
    ref > lim * (1 - e) && return true
    isfinite(r) && abs(big(r) - ref) <= bound
  end
end
# the margin of the zero group G: ‖v[G]‖₁ - c, with the bound of a sum of n magnitudes and a difference
function kk_margin_ok(r, λ0, w, v, G; factor = 1, extra = 0)
  W = eltype(v)
  r isa W || return false
  setprecision(BigFloat, PREC) do
    c = BigFloat(factor) * BigFloat(λ0) * BigFloat(w)
    V = [big(v[i]) for i in G]
    ref = sum(abs, V) - c
    isfinite(r) && abs(big(r) - ref) <= (length(G) + 6 + extra) * big(eps(W)) / 2 * (sum(abs, V) + c) + 4 * length(G) * big(nextfloat(zero(W)))
  end
end

# the families of nonzero groups: (name, far, generator (rng, m, s) -> (v, x, c0)) with v and x the m entries of the group, s the scale of
# v and c0 the coefficient. A far family has a distance of at least s (the entries of the face have uᵢ < 0, so that each face term of the
# distance is at least |uᵢ|, or the entry far below the level is large).
function kk_families(::Type{T}) where {T}
  t = T(1.5)
  aligned(rng, x, s, J) = [i in J ? kk_sgn(x[i]) * s * (1 + rand(rng, T)) : s * randn(rng, T) for i in eachindex(x)]
  coef(rng, s) = s * T(rand(rng, (0.01, 0.5, 3)))
  (("one largest entry", true, (rng, m, s) -> begin
      x = T(0.25) .* randn(rng, T, m); x[1] = rand(rng, Bool) ? T(3) : T(-3)
      (aligned(rng, x, s, 1:1), x, coef(rng, s))
    end),
   ("ties of the largest magnitude with mixed signs", true, (rng, m, s) -> begin
      x = T(0.25) .* randn(rng, T, m)
      for i in 1:m
        rand(rng) < 0.5 && (x[i] = rand(rng, Bool) ? t : -t)
      end
      x[1] = t
      (aligned(rng, x, s, findall(a -> abs(a) == t, x)), x, coef(rng, s))
    end),
   ("a face of all entries with an aligned gradient", true, (rng, m, s) -> begin
      x = T[rand(rng, Bool) ? 2 : -2 for _ in 1:m]
      (aligned(rng, x, s, 1:m), x, coef(rng, s))
    end),
   ("a face of all entries with a random gradient", false, (rng, m, s) -> begin
      x = T[rand(rng, Bool) ? 2 : -2 for _ in 1:m]
      v = s .* randn(rng, T, m)
      (v, x, sum(abs, v) * T(rand(rng, (0.01, 0.3, 0.8, 1.2))))
    end),
   ("a face entry far below the level", true, (rng, m, s) -> begin
      x = fill(T(3), m)
      v = -s .* rand(rng, T, m); v[1] = -50 * s
      (v, x, s / 2)
    end),
   ("zero gradients on the face", true, (rng, m, s) -> begin
      x = fill(T(-1), m)
      v = zeros(T, m); m > 1 && (v[end] = s)
      (v, x, s / 2)
    end),
   ("an entry one unit in the last place below the largest", false, (rng, m, s) -> begin
      x = T(0.25) .* randn(rng, T, m); x[1] = t; m > 1 && (x[2] = prevfloat(t))
      (s .* randn(rng, T, m), x, coef(rng, s))
    end),
   ("a stationary point", false, (rng, m, s) -> begin
      x = T(0.25) .* randn(rng, T, m)
      for i in 1:m
        rand(rng) < 0.5 && (x[i] = rand(rng, Bool) ? t : -t)
      end
      x[1] = t
      J = findall(a -> abs(a) == t, x)
      c0 = coef(rng, s)
      σ = rand(rng, T, length(J)) .+ T(0.1)
      σ .*= c0 / sum(σ)
      v = zeros(T, m)
      for (j, i) in enumerate(J)
        v[i] = -kk_sgn(x[i]) * σ[j]
      end
      (v, x, c0)
    end))
end

# the group G of m entries inside a point of p entries (the other entries random), as the first group of a function whose other groups
# are the single entries left; ranges or vectors of indices. λ0 w = c0 for a normal number c0 (w is a power of two).
function kk_instance(rng, ::Type{T}, gen, m, layout, s) where {T}
  p = m + 6 + m % 2
  if layout === :range
    a = rand(rng, 1:p - m + 1)
    G = a:a + m - 1
    groups = vcat([G], [i:i for i in 1:p if !(i in G)])
  else
    G = randperm(rng, p)[1:m]
    groups = vcat([G], [[i] for i in 1:p if !(i in G)])
  end
  v, x = s .* randn(rng, T, p), randn(rng, T, p)
  vg, xg, c0 = gen(rng, m, s)
  for (j, i) in enumerate(G)
    v[i], x[i] = vg[j], xg[j]
  end
  w = T(rand(rng, (0.5, 1, 2)))
  f = GroupNormLinf(c0 / w, groups, vcat(w, ones(T, length(groups) - 1)))
  f, v, x, G
end

# the full-point value against the reference: the maximum of zero, the distance of every group (a zero group included) with the coefficient
# α λ0 w, and |vᵢ| of the entries in no group; the bound is the largest bound of a group (the entries in no group are exact)
function kk_full_ok(r, f, v, x; α = 1, slack = 1)
  W = float(promote_type(eltype(v), eltype(x)))
  r isa W || return false
  setprecision(BigFloat, PREC) do
    best, b = big(0), big(0)
    for k in eachindex(f.groups)
      d, bd = kk_ref(big(α) * big(f.λ0), f.w[k], v, x, f.groups[k]; slack)
      best, b = max(best, d), max(b, bd)
    end
    for i in vcat(f.free, length(f.group_of) + 1:length(x))
      best = max(best, abs(big(convert(W, v[i]))))
    end
    isfinite(r) && abs(big(r) - best) <= b
  end
end

# a point whose face has several entries: entries of the largest magnitude, with their signs
function kk_tie_face(rng, ::Type{T}, n) where {T}
  x = randn(rng, T, n)
  m = maximum(abs, x)
  for i in 1:n
    rand(rng) < 0.5 && (x[i] = copysign(m, x[i]))
  end
  x
end

# chains of two wrappers whose product of factors is not a normal number, and a level that folds into it (tiny² lvl is a normal number)
kk_tiny(T) = ldexp(one(T), T === Float32 ? -100 : -600)
kk_lvl(T) = ldexp(one(T), T === Float32 ? 100 : 1000)
kk_chain(T, f) = (c = Scaled(Scaled(f, kk_tiny(T)), kk_tiny(T)); c.f isa Scaled || error("the wrappers merged"); c)
kk_raw(f, a) = Scaled{typeof(f), typeof(a)}(f, a)

@testset "optimality: the margin of a zero group and the examples of the other scores ($T)" for T in (Float32, Float64)
  f = GroupNormLinf(one(T), [1:2, 3:3], T[1, 0.5])
  z = zeros(T, 4)                                    # entry 4 is in no group
  g = T[5 // 8, 5 // 8, -2, 3]
  # the margin ‖v[G]‖₁ - λ0 w, by hand: 1.25 - 1 and 2 - 0.5
  @test kkt_violation(f, g, z, 1) === T(0.25) && kkt_violation(f, g, z, 2) === T(0.25)
  @test kkt_violation(f, g, z, 3) === T(1.5)
  # an entry in no group has the score |vᵢ|, also at a nonzero entry
  @test kkt_violation(f, g, z, 4) === T(3) && kkt_violation(f, g, T[0, 0, 0, 7], 4) === T(3)
  # g = (5/8, 5/8) with λ0 = w = 1 has the margin 0.25, where the Euclidean screen of GroupNormL2 is negative
  f1 = GroupNormLinf(one(T), [1:2])
  @test kkt_violation(f1, T[5 // 8, 5 // 8], zeros(T, 2), 1) === T(0.25)
  @test kkt_violation(GroupNormL2(one(T), [1:2], T[1]), T[5 // 8, 5 // 8], zeros(T, 2), 1) < 0
  # a zero group of either sign of zero is zero
  @test kkt_violation(f1, T[5 // 8, 5 // 8], T[-0.0, 0.0], 1) === T(0.25)
  # the conventions: NaN in v[G] gives NaN; Inf (and no NaN) gives Inf, also at an infinite coefficient; an infinite
  # coefficient with a finite computed sum of magnitudes gives -Inf; a zero weight gives ‖v[G]‖₁; an overflowing sum gives Inf
  M = floatmax(T)
  for (λ0, w) in ((one(T), one(T)), (T(Inf), one(T)), (one(T), T(Inf)), (zero(T), T(Inf)), (M, T(4)))
    h = GroupNormLinf(λ0, [1:2], T[w])
    @test isnan(kkt_violation(h, T[NaN, 1], zeros(T, 2), 1)) && isnan(kkt_violation(h, T[Inf, NaN], zeros(T, 2), 2))
    @test kkt_violation(h, T[Inf, 1], zeros(T, 2), 1) === T(Inf) && kkt_violation(h, T[1, -Inf], zeros(T, 2), 2) === T(Inf)
  end
  for (λ0, w) in ((T(Inf), one(T)), (one(T), T(Inf)), (zero(T), T(Inf)), (M, T(4)))
    @test kkt_violation(GroupNormLinf(λ0, [1:2], T[w]), T[1, 2], zeros(T, 2), 1) === T(-Inf)
  end
  for λ0 in (zero(T), one(T), T(Inf))
    h = GroupNormLinf(λ0, [1:2], T[0])
    @test kkt_violation(h, T[1.5, -2], zeros(T, 2), 1) === T(3.5)
  end
  @test kkt_violation(GroupNormLinf(one(T), [1:2]), T[M, M], zeros(T, 2), 1) === T(Inf)
  # a computed sum of magnitudes that overflows gives +Inf first, also at an infinite coefficient
  @test kkt_violation(GroupNormLinf(T(Inf), [1:2]), T[M, M], zeros(T, 2), 1) === T(Inf)
  # a level that is not of the type of the gradient: the coefficient is formed in the working type
  @test kkt_violation(GroupNormLinf(0.5, [1:2]), T[1, 1], zeros(T, 2), 1) === T(1.5)
  @test kkt_violation(GroupNormLinf(1, [1:2]), [1, 1], [0, 0], 1) === 1.0
  # a BlockIterate point: the zero group of the iterate, with the layout of f
  B = BlockIterate(T, 4, [1:2, 3:3, 4:4])
  fb = GroupNormLinf(one(T), [1:2, 3:3, 4:4], T[1, 1, 1])
  @test kkt_violation(fb, g, B, 1) === T(0.25) && kkt_violation(fb, g, B, 2) === T(1)
  copyto!(blockview(B, 2), T[1])
  @test kkt_violation(fb, g, B, 2) === T(1)                                     # a nonzero group: |v + c sign(x)| = |-2 + 1|
  @test kkt_violation(fb, g, B, 1) === T(0.25)
  @test_throws DimensionMismatch kkt_violation(fb, g, BlockIterate(T, 4, [1:1, 2:3, 4:4]), 1)   # a group differs
  @test_throws DimensionMismatch kkt_violation(fb, g, BlockIterate(T, 4, [1:3, 4:4]), 1)        # the group count differs
  @test_throws BoundsError kkt_violation(fb, g, B, 4)
  @test_throws DimensionMismatch kkt_violation(fb, T[1, 2], B, 1)                # the gradient length
  # an array point: the group of the entry, with checks of length and extent
  @test_throws DimensionMismatch kkt_violation(f, T[1, 2, 3], z, 1)
  @test_throws DimensionMismatch kkt_violation(f, T[1, 2], T[0, 0], 1)           # the groups reach index 3
  @test_throws BoundsError kkt_violation(f, g, z, 5)
  # a nonzero group: the entry selects its group, and the score is the distance from -v[G] to the subdifferential at x[G]. For group 1,
  # v = (5/8, 5/8) and c = 1, the face is the entry of x that is nonzero: sqrt((5/8)² + (1 + 5/8)²). Group 2 is the entry 3 with c = 1/2.
  d1 = sqrt(T(194)) / 8
  @test isapprox(kkt_violation(f, g, T[0, 1, 0, 0], 1), d1; rtol = 8 * eps(T))
  @test isapprox(kkt_violation(f, g, T[1e-30, 0, 0, 0], 1), d1; rtol = 8 * eps(T))
  @test kkt_violation(f, g, T[0, 0, 1, 0], 3) === T(1.5)                        # |-2 + 1/2|
  # NaN and infinite entries of x give NaN and Inf
  @test isnan(kkt_violation(f, g, T[NaN, 0, 0, 0], 2))
  @test kkt_violation(f, g, T[0, 0, Inf, 0], 3) === T(Inf)
  @test kkt_violation(f, g, T[0, 0, 1, 0], 1) === T(0.25)                       # group 1 is still zero
  # the full-point form: the maximum of zero, |vᵢ| of the entries in no group and the distance of every group, a zero group included
  # (here the entry 4 decides: group 1 has the distance sqrt(2)/8 to the ball, and group 2 has 3/2)
  @test kkt_violation(f, g, z) === T(3)
  @test kkt_violation(f, g, T[0, 1, 0, 0]) === T(3)
  @test isapprox(kkt_violation(fb, g, B), T(2); rtol = 8 * eps(T))              # the zero group 3 (entry 4, v = 3, c = 1) has the distance 2
  @test_throws ArgumentError kkt_violation(f, g, 1)                            # the three-argument form is not separable
  @test kkt_violation(GroupNormLinf(one(T), Vector{Vector{Int}}()), T[], T[]) === zero(T)
  # a symmetric iterate has no groups
  S = SymmetricSparseIterate(T, 3)
  @test_throws ArgumentError kkt_violation(GroupNormLinf(one(T), [1:2]), zeros(T, 9), S, 1)
  @test_throws ArgumentError kkt_violation(GroupNormLinf(one(T), [1:2]), zeros(T, 9), S)
  # under Scaled with a finite factor: α times the score at v/α. At the zero group 1 that is 2 (‖(5/16, 5/16)‖₁ - 1), and the full-point
  # value is 2 max(0, 0, 1/2, 3/2) with the zero groups at their distances
  @test kkt_violation(Scaled(f, T(2)), g, z, 1) === T(-0.75)
  @test kkt_violation(Scaled(f, T(2)), g, z) === T(3)
  @test allocs(kkt_violation, f, g, z, 1) == 0 && allocs(kkt_violation, f, g, z, 4) == 0
  @test allocs(kkt_violation, fb, g, B, 1) == 0
end

@testset "nonzero groups against the BigFloat reference ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1014)
  for (name, far, gen) in kk_families(T), layout in (:range, :perm), m in (1, 2, 3, 5, 17, 64, 257)
    bad = 0
    for rep in 1:(m <= 17 ? 8 : m <= 64 ? 3 : 2)
      s = T(2)^rand(rng, (-30, -3, 0, 0, 4, 20))
      f, v, x, G = kk_instance(rng, T, gen, m, layout, s)
      rep == 1 && m >= 2 && (x[last(G)] = zero(T))        # a zero entry inside the group, stored as a zero by the SparseIterate below
      r = kkt_violation(f, v, x, first(G))
      bad += !kk_ok(r, f.λ0, f.w[1], v, x, G; far)
      if rep == 1
        # the same group read from a Matrix, a SparseIterate with a stored zero and a BlockIterate (group 1)
        sx = SparseIterate(x)
        m >= 2 && (sx[last(G)] = 0)
        for (pt, k) in ((reshape(copy(x), 2, :), first(G)), (sx, first(G)), (BlockIterate(copy(x), f.groups), 1))
          bad += !kk_ok(kkt_violation(f, v, pt, k), f.λ0, f.w[1], v, x, G; far)
        end
        bad += !kk_ok(kkt_violation(f, reshape(copy(v), 2, :), x, first(G)), f.λ0, f.w[1], v, x, G; far)
      end
    end
    @test bad == 0
  end
end

# an entry one unit in the last place below the largest is not on the face, so that the value differs from that of the tie
@testset "an entry one unit in the last place below the largest is off the face ($T)" for T in (Float32, Float64)
  f = GroupNormLinf(one(T), [1:2])
  v = T[-0.4, 0.3]
  tied, broken = T[2, 2], T[2, prevfloat(T(2))]
  rt, rb = kkt_violation(f, v, tied, 1), kkt_violation(f, v, broken, 1)
  @test kk_ok(rt, one(T), one(T), v, tied, 1:2) && kk_ok(rb, one(T), one(T), v, broken, 1:2)
  @test isapprox(rt, T(0.45) * sqrt(T(2)); rtol = 32 * eps(T))             # both entries at -0.45
  @test isapprox(rb, sqrt(T(0.45)); rtol = 32 * eps(T))                    # the face is the first entry alone
  @test rt != rb
  # the reference of the other face is far outside the bound of this call
  @test !kk_ok(rb, one(T), one(T), v, tied, 1:2) && !kk_ok(rt, one(T), one(T), v, broken, 1:2)
  # the same pair inside a larger point and a group of vector indices
  fv = GroupNormLinf(one(T), [[5, 2], [1], [3], [4]])
  vv, xv = T[9, 0.3, 4, 5, -0.4], T[1, prevfloat(T(2)), 7, 8, 2]
  @test kk_ok(kkt_violation(fv, vv, xv, 5), one(T), one(T), vv, xv, [5, 2])
  @test !kk_ok(kkt_violation(fv, vv, xv, 5), one(T), one(T), vv, T[1, 2, 7, 8, 2], [5, 2])
end

# The branches of the kernel through a group, each by a case that reaches it (confirmed by calling the helpers of the kernel on the
# same inputs: the range test `_linf_in_range`, the first level τ0, and the sum of squares Q):
#   - the range test sends the face to the scaled path (the sum of magnitudes times n + 1 is above floatmax/4);
#   - the range test sends the zero point to the scaled path (the sum of magnitudes is above floatmax/2, or overflows);
#   - the range test passes and the first level τ0 of the face is -Inf, which sends it to the scaled path;
#   - the ordinary sum of squares Q is not normal (the squares underflow), which sends it to the recomputation.
# The entries around the group are NaN and Inf and do not reach the result of the four-argument form.
@testset "branches of the kernel through a group ($T)" for T in (Float32, Float64)
  M, δ = floatmax(T), nextfloat(zero(T))
  emax, emin = exponent(M), exponent(floatmin(T))
  function embed(λ0, w, v, x)
    p = length(v) + 3
    vv, xx = fill(T(NaN), p), fill(T(Inf), p)
    vv[2:end - 2] .= v
    xx[2:end - 2] .= x
    GroupNormLinf(λ0, [2:length(v) + 1], [w]), vv, xx, 2:length(v) + 1
  end
  check(λ0, w, v, x; slack = 1, far = false) = begin
    f, vv, xx, G = embed(λ0, w, v, x)
    r = kkt_violation(f, vv, xx, 2)
    @test kk_ok(r, f.λ0, f.w[1], vv, xx, G; slack, far)
    @test allocs(kkt_violation, f, vv, xx, 2) == 0
    r
  end
  # entries near floatmax: (n + 1) times the sum of magnitudes is above floatmax/4, so that the range test sends the face to the scaled
  # path (faces of three entries, of one entry, and of all four)
  v = T[0.9, -0.8, 0.7, 0.2] .* (M / 8)
  for x in (T[1, -1, 1, 0], T[0, 1, 0, 0], T[1, 1, 1, 1]), λ0 in (M / 64, M / 8, M / 3)
    check(λ0, one(T), v, x; slack = 4)
  end
  # the same for a face of 300 entries, with a level at floatmax: the range test sends it to the scaled path
  check(M, one(T), fill(M / (5 * 301), 300), ones(T, 300); slack = 4)
  # the range test passes ((n + 1) times the sum of magnitudes is at most floatmax/4) and the first level (S - c)/cnt is -Inf, because
  # S - c is below -floatmax: the exit of the ordinary path to the scaled path
  check(M * (1 - T(2)^-6), one(T), T[M / 32, M / 32], T[1, 1]; slack = 4)
  # the zero point, by the full-point form: the sum of magnitudes is above floatmax/2 (0.6 floatmax), and it overflows (4/3 floatmax, with
  # a distance of about 0.94 floatmax in each type): the range test sends the zero point to the scaled path
  for v0 in (T[0.3 * M, 0.3 * M], fill(M / T(1.5), 2))
    f0 = GroupNormLinf(one(T), [1:2])
    r0 = kkt_violation(f0, v0, zeros(T, 2))
    @test kk_ok(r0, one(T), one(T), v0, zeros(T, 2), 1:2; slack = 4)
    @test isfinite(r0) && allocs(kkt_violation, f0, v0, zeros(T, 2)) == 0
    @test kkt_violation(f0, v0, zeros(T, 2), 1) > M / 2                       # the margin of the zero group
  end
  # a distance above floatmax is Inf, and an entry at floatmax against a level at floatmax cancels
  @test check(M, one(T), T[M], T[1]) === T(Inf)
  @test check(M, one(T), T[-M], T[1]) === zero(T)
  # a residual far below the level, whose square underflows to zero: Q is not normal, and the value is the residual (the recomputation)
  small = ldexp(one(T), emin ÷ 2 - 30)
  r = check(one(T), one(T), T[-1, small], T[1, 0.5]; far = false)
  @test isapprox(r, small; rtol = 16 * eps(T))
  r = check(zero(T), one(T), fill(small, 7), T[1, 0, 0, 0, 0, 0, 0])
  @test isapprox(r, sqrt(T(7)) * small; rtol = 16 * eps(T))
  # subnormal entries and level
  check(T(2) * δ, one(T), T[3δ, -7δ, 2δ], T[δ, 3δ, -3δ])
  check(floatmin(T), one(T), T[1, 1] .* floatmin(T), T[1, 0])
  # a point at the bottom of the range with entries of the size of one
  check(one(T), one(T), T[1.5, -1, 0.25], T[δ, 0, 0])
  # the entries near floatmax with many of them: the squares and the sums are out of the ordinary range
  for n in (2, 10, 100)
    check(one(T), one(T), fill(M / T(n) * T(0.9), n), T[i == 1 ? 1 : 0.5 for i in 1:n]; slack = 4)
  end
end

@testset "full-point form ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1015)
  # entries 6, 8, 11 are in no group below the largest index of a group, and 17 and 18 are past it; group 4 has the weight zero
  groups = [[2, 7, 3], [9], [1, 5], [4, 10, 12], [13, 14, 15, 16]]
  f = GroupNormLinf(T(0.8), groups, T[1, 0.5, 2, 0, 1])
  free = [6, 8, 11, 17, 18]
  v = 2 .* randn(rng, T, 18)
  x = randn(rng, T, 18)
  x[[2, 7]] .= T(1.5); x[3] = T(0.5)        # a tie on the face of group 1
  x[[1, 5, 9, 13, 14, 15, 16]] .= 0        # groups 2, 3 and 5 are zero
  x[free] .= T[5, 0, 3, 0, -2]
  # the value against the reference: the maximum of zero, the distance of every group and |vᵢ| of the entries in no group
  full_ok(h, vv, xx) = kk_full_ok(kkt_violation(h, vv, xx), h, vv, xx)
  @test full_ok(f, v, x)
  R = kkt_violation(f, v, x)
  @test typeof(R) === T
  # one score per group: the entries of a group give the same value, and the full-point value agrees (within 16 eps of the scale) with
  # the maximum of zero, the four-argument scores of the nonzero groups and of the entries in no group, and the distances of the zero groups
  s = norm(v) + 8
  nonzero = [k for k in 1:5 if any(!iszero, x[groups[k]])]
  @test nonzero == [1, 4]
  locals = [kkt_violation(f, v, x, first(groups[k])) for k in nonzero]
  zero_dist = [kkt_violation(GroupNormLinf(f.λ0, [1:length(G)], [f.w[k]]), v[G], zeros(T, length(G))) for (k, G) in enumerate(groups) if !(k in nonzero)]
  uncovered = abs.(v[free])
  @test kk_close(R, max(zero(T), maximum(locals), maximum(zero_dist), maximum(uncovered)), T, s)
  for k in 1:5, i in groups[k]
    @test kkt_violation(f, v, x, i) === kkt_violation(f, v, x, first(groups[k]))
  end
  # the entries in no group decide when they are the largest, and x is not read there
  v2, x2 = copy(v), copy(x)
  v2[11], x2[11] = 100, T(NaN)
  @test kkt_violation(f, v2, x2) === T(100) && kkt_violation(f, v2, x2, 11) === T(100)
  v2[11] = -100
  @test kkt_violation(f, v2, x2) === T(100)
  # a zero point: every group is scored by its distance to the ℓ1 ball, so that the small gradient of a group with a positive
  # coefficient gives zero and that of the group with the weight zero gives its 2-norm
  z = zeros(T, 18)
  vs = T(0.01) .* v
  @test full_ok(f, vs, z)
  @test kk_close(kkt_violation(f, vs, z), max(norm(vs[groups[4]]), maximum(abs.(vs[free]))), T, 1)
  # a point at which the zero groups satisfy ‖v_G‖₁ ≤ c and the nonzero groups are stationary has the value zero
  f0 = GroupNormLinf(one(T), [1:3, 4:6], T[1, 1])
  @test kkt_violation(f0, T[0.1, -0.2, 0.05, -0.25, 0.75, 0, -0.0], T[0, 0, 0, 2, -2, 1, 5]) === zero(T)
  # a zero group with a positive margin contributes its distance and not its margin: (5/8, 5/8) at the level one
  f1 = GroupNormLinf(one(T), [1:2])
  g1 = T[5 // 8, 5 // 8]
  @test kkt_violation(f1, g1, zeros(T, 2), 1) === T(0.25)
  d = kkt_violation(f1, g1, zeros(T, 2))
  @test isapprox(d, sqrt(T(2)) / 8; rtol = 8 * eps(T)) && d < T(0.25)
  @test T(0.25) <= sqrt(T(2)) * d + 8 * eps(T)                         # distance ≤ margin ≤ √n distance in exact arithmetic, with equality on the right here (up to rounding)
  # and where the margin is positive in general: distance ≤ margin ≤ √n distance in exact arithmetic, within the bound of the calls
  # for the computed values
  for n in (1, 2, 5, 17, 64), _ in 1:6
    vg = randn(rng, T, n)
    c = T(rand(rng, (0.1, 0.3, 0.6))) * sum(abs, vg)
    h = GroupNormLinf(c, [1:n])
    mg, dist = kkt_violation(h, vg, zeros(T, n), 1), kkt_violation(h, vg, zeros(T, n))
    tolr = 16 * (n + 8) * eps(T) * (sum(abs, vg) + c)
    @test mg > 0 && dist <= mg + tolr && mg <= sqrt(T(n)) * dist + tolr
    @test kk_ok(dist, c, one(T), vg, zeros(T, n), 1:n)
  end
  # one group over all of x agrees with NormLinf with the level λ0 w, at the zero point and at another point: both calls are within
  # the bound of the same reference (they are not compared bit for bit)
  for n in (1, 2, 17, 64), x0 in (zeros(T, n), kk_tie_face(rng, T, n))
    vg = randn(rng, T, n)
    h = GroupNormLinf(T(0.7), [1:n], T[2])
    c = T(0.7) * T(2)
    a, b = kkt_violation(h, vg, x0), kkt_violation(NormLinf(c), vg, x0)
    @test kk_ok(a, T(0.7), T(2), vg, x0, 1:n) && kk_ok(b, c, one(T), vg, x0, 1:n)
  end
  # a NaN term wins over an Inf term, in either order of the groups and for an entry in no group; Inf alone gives Inf
  for gs in ([1:2, 3:4], [3:4, 1:2]), pt in (zeros(T, 4), T[1, 0, 1, 0], T[1, 1, 1, 1])
    h = GroupNormLinf(one(T), gs)
    @test isnan(kkt_violation(h, T[NaN, 1, Inf, 1], pt))
    @test kkt_violation(h, T[1, 1, Inf, 1], pt) === T(Inf)
    @test isnan(kkt_violation(h, T[Inf, 1, NaN, 1], pt))
  end
  hf = GroupNormLinf(one(T), [1:2])
  @test isnan(kkt_violation(hf, T[Inf, 1, NaN], T[1, 0, 0])) && isnan(kkt_violation(hf, T[NaN, 1, Inf], T[1, 0, 0]))
  @test kkt_violation(hf, T[1, 1, Inf], T[1, 0, 0]) === T(Inf)
  # an empty point without groups, and a function without groups: every entry is in no group
  @test kkt_violation(GroupNormLinf(one(T), Vector{Vector{Int}}()), T[], T[]) === zero(T)
  @test kkt_violation(GroupNormLinf(one(T), Vector{Vector{Int}}()), T[3, -7, 2], T[1, 1, 1]) === T(7)
  # a BlockIterate point: the whole layout is checked by the full-point form, the layout of the selected group by the four-argument form
  bg = [[2, 7, 3], [9], [1, 5], [4, 10, 12], [6, 8, 11, 13, 14, 15, 16, 17, 18]]
  fb = GroupNormLinf(T(0.8), bg, T[1, 0.5, 2, 0, 1])
  bx = BlockIterate(copy(x), bg)
  @test kk_ok(kkt_violation(fb, v, bx, 1), fb.λ0, fb.w[1], v, x, bg[1]) && kk_ok(kkt_violation(fb, v, bx, 4), fb.λ0, fb.w[4], v, x, bg[4])
  @test full_ok(fb, v, bx)
  mixed = BlockIterate(copy(x), [[2, 7, 3], [9], [1, 5], [10, 4, 12], [6, 8, 11, 13, 14, 15, 16, 17, 18]])
  @test kkt_violation(fb, v, mixed, 1) isa T
  @test_throws DimensionMismatch kkt_violation(fb, v, mixed, 4)
  @test_throws DimensionMismatch kkt_violation(fb, v, mixed)
  @test_throws DimensionMismatch kkt_violation(fb, v, BlockIterate(copy(x), [collect(1:18)]))
  @test_throws DimensionMismatch kkt_violation(fb, v, BlockIterate(copy(x), [[2, 7, 3], collect(setdiff(1:18, [2, 7, 3]))]), 1)
  # a SparseIterate point
  sx = SparseIterate(x)
  sx[1] = 0
  @test kk_close(kkt_violation(f, v, sx), R, T, s)
  # validation: complex entries, offset axes, lengths, groups that reach past the point, a symmetric iterate
  cv, cx = Complex{T}.(v), Complex{T}.(x)
  off(a) = OffsetArray(a, 0:length(a) - 1)
  @test_throws ArgumentError kkt_violation(f, cv, x)
  @test_throws ArgumentError kkt_violation(f, v, cx)
  @test occursin("offset arrays", kk_msg(() -> kkt_violation(f, off(v), x)))
  @test occursin("offset arrays", kk_msg(() -> kkt_violation(f, v, off(x))))
  @test_throws DimensionMismatch kkt_violation(f, v[1:17], x)
  @test_throws DimensionMismatch kkt_violation(f, v, x[1:17])
  @test_throws DimensionMismatch kkt_violation(f, v[1:15], x[1:15])
  @test_throws ArgumentError kkt_violation(f, ones(T, 9), SymmetricSparseIterate(T, 3))
  # a range, a view and a transposed matrix are points
  @test kk_close(kkt_violation(f, view(v, 1:18), view(x, 1:18)), R, T, s)
  fm = GroupNormLinf(T(0.8), [[1, 3], [2, 4]], T[1, 1])
  vm, xm = T[1, 2, 3, 4], T[0, 1, 0, 0]
  @test kk_close(kkt_violation(fm, reshape(vm, 2, 2), reshape(xm, 2, 2)), kkt_violation(fm, vm, xm), T, 10)
  # no write
  v0, x0 = copy(v), copy(x)
  kkt_violation(f, v, x); kkt_violation(f, v, x, 3); kkt_violation(f, v, x, 9)
  @test v == v0 && x == x0
end

@testset "exceptional values at a nonzero group ($T)" for T in (Float32, Float64)
  NaNs, Infs, M = T(NaN), T(Inf), floatmax(T)
  # a NaN in v[G] or x[G] gives NaN, also beside an Inf; otherwise an Inf entry gives Inf, also at the level zero and at an infinite level
  for n in (1, 3, 64, 130), pos in unique((1, max(1, n ÷ 2), n))
    for x0 in (ones(T, n), T[i == 1 ? 1 : 0 for i in 1:n]), λ in (zero(T), T(0.5), T(5), Infs)
      v = ones(T, n)
      f = GroupNormLinf(λ, [1:n])
      vn, vi, xn, xi = copy(v), copy(v), copy(x0), copy(x0)
      vn[pos] = NaNs; vi[pos] = Infs; xn[pos] = NaNs; xi[pos] = Infs
      for call in (kkt_violation(f, vn, x0, 1), kkt_violation(f, v, xn, 1), kkt_violation(f, vn, x0), kkt_violation(f, v, xn))
        @test isnan(call)
      end
      for call in (kkt_violation(f, vi, x0, 1), kkt_violation(f, v, xi, 1), kkt_violation(f, -vi, x0, 1), kkt_violation(f, v, -xi, 1),
                   kkt_violation(f, vi, x0), kkt_violation(f, v, xi))
        @test call === Infs
      end
      if n >= 2
        other = pos == 1 ? 2 : 1
        a, b = copy(vi), copy(xi)
        a[other] = NaNs
        @test isnan(kkt_violation(f, a, x0, 1)) && isnan(kkt_violation(f, a, x0))
        b[other] = NaNs
        @test isnan(kkt_violation(f, v, b, 1)) && isnan(kkt_violation(f, v, b))
        @test isnan(kkt_violation(f, vi, xn, 1)) && isnan(kkt_violation(f, vn, xi, 1))
      end
    end
  end
  # the coefficient zero (a zero weight, or a zero level with a finite weight, also with an infinite level or weight against the zero)
  # gives the 2-norm of v[G] at any point
  vg = T[3, -4, 12]
  for (λ, w) in ((zero(T), one(T)), (one(T), zero(T)), (zero(T), T(7)), (Infs, zero(T)))
    h = GroupNormLinf(λ, [1:3], T[w])
    for x0 in (T[1, 0, 0], T[1, -1, 0.5], T[0, 2, 0])
      @test isapprox(kkt_violation(h, vg, x0, 1), T(13); rtol = 16 * eps(T))
      @test isapprox(kkt_violation(h, vg, x0), T(13); rtol = 16 * eps(T))
    end
  end
  # an infinite coefficient (an infinite weight or level, a zero level with an infinite weight, a product that overflows in the type of
  # the arguments) gives Inf at a nonzero group; at a zero group the four-argument form keeps its margin -Inf and the full-point form,
  # where the distance to the ball of infinite radius is zero, gives zero
  for (λ, w) in ((one(T), Infs), (Infs, one(T)), (zero(T), Infs), (M, T(4)))
    h = GroupNormLinf(λ, [1:3], T[w])
    for x0 in (T[1, 0, 0], T[1, -1, 0.5])
      @test kkt_violation(h, vg, x0, 1) === Infs && kkt_violation(h, vg, x0) === Infs
    end
    @test kkt_violation(h, vg, zeros(T, 3), 1) === -Infs
    @test kkt_violation(h, vg, zeros(T, 3)) === zero(T)
  end
  # a level of a type that overflows the type of the entries is an infinite level there
  T === Float32 && @test kkt_violation(GroupNormLinf(1e300, [1:2]), Float32[1, 2], Float32[1, 0], 1) === Infs
  # a non-finite entry in another group does not reach the four-argument score of a group, and does reach the full-point value
  f2 = GroupNormLinf(T(0.5), [1:2, 3:4, 5:6])
  v, x = T[1, 2, 3, 4, 5, 6], T[1, 0, 1, 1, 0, 0]
  r1 = kkt_violation(f2, v, x, 1)
  @test kkt_violation(f2, T[1, 2, NaN, 4, Inf, 6], T[1, 0, 1, Inf, 0, NaN], 1) === r1
  @test isnan(kkt_violation(f2, T[1, 2, NaN, 4, Inf, 6], x)) && isnan(kkt_violation(f2, v, T[1, 0, 1, Inf, 0, NaN]))
  @test kkt_violation(f2, T[1, 2, 3, Inf, 5, 6], x) === Infs && kkt_violation(f2, v, T[1, 0, 1, Inf, 0, 0]) === Infs
  # an infinite entry beside the face and an entry at the level of floatmax
  @test kkt_violation(GroupNormLinf(M, [1:2]), T[M, M], T[1, 0], 1) === Infs
end

# a group of n entries needs n + 1 to be a finite number of the working type: Float16 from n = 65519. The test for a zero group comes first
# in the four-argument form, and the full-point form checks every group
@testset "a group whose size the working type refuses (Float16)" begin
  n = 65519
  f = GroupNormLinf(Float16(1), [1:n])
  vz, xz = zeros(Float16, n), zeros(Float16, n)
  x1 = copy(xz); x1[1] = 1
  @test kkt_violation(f, vz, xz, 1) === Float16(-1)                           # a zero group keeps its margin ‖v‖₁ - c
  @test_throws ArgumentError kkt_violation(f, vz, x1, 1)
  m = kk_msg(() -> kkt_violation(f, vz, x1, 1))
  @test occursin("65519", m) && occursin("Float16", m)
  @test_throws ArgumentError kkt_violation(f, vz, x1)
  @test_throws ArgumentError kkt_violation(f, vz, xz)                          # also a zero group
  @test_throws ArgumentError kkt_violation(Scaled(f, Float16(2)), vz, x1, 1)
  @test_throws ArgumentError kkt_violation(Scaled(f, Float16(2)), vz, x1)
  @test kkt_violation(Scaled(f, Float16(2)), vz, xz, 1) isa Float16
  # a group of 65518 entries is accepted
  fs = GroupNormLinf(Float16(1), [1:n - 1])
  r = kkt_violation(fs, vz[1:n - 1], x1[1:n - 1], 1)
  @test r isa Float16 && isapprox(r, Float16(1); rtol = 0.01)
  # a Float32 point of the same size is not refused
  @test kkt_violation(GroupNormLinf(1f0, [1:n]), zeros(Float32, n), Float32[i == 1 ? 1 : 0 for i in 1:n], 1) isa Float32
end

@testset "element types of the scores" begin
  # integer v and x: the working type is Float64
  f = GroupNormLinf(1.0, [1:3, 4:5], [1.0, 2.0])
  v, x = [3, 1, -2, 5, 5], [2, -2, 1, 0, 0]
  r = kkt_violation(f, v, x, 1)
  @test r isa Float64 && kk_ok(r, f.λ0, f.w[1], v, x, 1:3)
  @test kkt_violation(f, v, x, 4) === 8.0                                      # the margin 10 - 2 of a zero group
  @test kkt_violation(f, v, x) isa Float64
  @inferred kkt_violation(f, v, x, 1)
  @inferred kkt_violation(f, v, x)
  # a level that is not of the type of the entries: the coefficient is formed in the working type; Float32 v with a Float64 x is Float64
  # and uses the Float64 coefficient 0.1, which is not a Float32
  f1 = GroupNormLinf(0.1, [1:3])
  v32, x64 = Float32[3, 1, -2], [2.0, -2.0, 1.0]
  for (vv, xx) in ((v32, x64), (Float64.(v32), Float32.(x64)), (v32, Float32.(x64)))
    r = kkt_violation(f1, vv, xx, 1)
    W = float(promote_type(eltype(vv), eltype(xx)))
    @test r isa W && kk_ok(r, f1.λ0, 1.0, vv, xx, 1:3)
    @test kkt_violation(f1, vv, xx) isa W
    @inferred kkt_violation(f1, vv, xx, 1)
    @inferred kkt_violation(f1, vv, xx)
  end
  # Int8 and Int16 give Float64; an integer level
  @test kkt_violation(GroupNormLinf(1, [1:2]), Int8[3, 1], Int16[2, 2], 1) isa Float64
  @test kk_ok(kkt_violation(GroupNormLinf(1, [1:2]), Int8[3, 1], Int16[2, 2], 1), 1, 1, Int8[3, 1], Int16[2, 2], 1:2)
  # Float16: the bound of the type
  for (vv, xx, λ) in ((Float16[3, 1, -2], Float16[2, -2, 1], 1), (Float16[3, 1, -2], Float16[0, 0, 0.5], 1), (Float16[-3, 1, 2], Float16[2, 2, -2], 0.5))
    h = GroupNormLinf(Float16(λ), [1:3])
    r = kkt_violation(h, vv, xx, 1)
    @test r isa Float16 && kk_ok(r, Float16(λ), one(Float16), vv, xx, 1:3)
    @test kkt_violation(h, vv, xx) isa Float16
    @inferred kkt_violation(h, vv, xx, 1)
    @inferred kkt_violation(h, vv, xx)
  end
  # BigFloat arrays in the working precision, against a reference at four times the precision
  λb = setprecision(BigFloat, 256) do
    BigFloat(3) / 10
  end
  vb, xb = BigFloat[3, 1, 2], BigFloat[2, -2, 1]
  for x0 in (xb, zeros(BigFloat, 3))
    r, rf = setprecision(BigFloat, 256) do
      f = GroupNormLinf(λb, [1:3])
      kkt_violation(f, vb, x0, 1), kkt_violation(f, vb, x0)
    end
    @test r isa BigFloat && rf isa BigFloat
    setprecision(BigFloat, 1024) do
      ref = kk_dist(big(λb), vb, x0)
      allowed = 64 * big(2)^-256 * (sqrt(sum(abs2, vb)) + λb)
      @test abs(rf - ref) <= allowed
      # the four-argument form: the distance at a nonzero group and the margin at a zero group
      @test abs(r - (all(iszero, x0) ? sum(abs, vb) - λb : ref)) <= allowed
    end
  end
  @test kkt_violation(GroupNormLinf(1.0, [1:2]), [3.0, 1.0], BigFloat[1, 0], 1) isa BigFloat
end

@testset "a group of one entry has the score of NormL1 ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1018)
  p = 60
  v, x = 2 .* randn(rng, T, p), randn(rng, T, p)
  x[1:10] .= 0
  w = T(0.2) .+ rand(rng, T, p)
  fl, f1 = GroupNormLinf(T(0.7), [[i] for i in 1:p], w), NormL1(T(0.7), w)
  for i in 11:p
    @test kkt_violation(fl, v, x, i) === kkt_violation(f1, v, x, i)            # the nonzero entries
  end
  for i in 1:10
    @test kk_close(kkt_violation(fl, v, x, i), kkt_violation(f1, v, x, i), T, 4)   # the zero entries: the margin |vᵢ| - cᵢ
  end
  # the full-point values agree within a few ulps: the maximum of zero and the scores of the entries, where a zero entry has the positive
  # part of its margin
  @test kk_close(kkt_violation(fl, v, x), kkt_violation(f1, v, x), T, 4)
  @test kk_close(kkt_violation(fl, v, zeros(T, p)), kkt_violation(f1, v, zeros(T, p)), T, 4)
end

# ---- Scaled: one wrapper, an infinite factor, chains ----
@testset "Scaled with one factor ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1016)
  groups = [[2, 7, 3], [9], [1, 5], [4, 10, 12]]       # entries 6, 8 and 11 are in no group, and entry 13 is past the groups
  w = T[1, 0.5, 2, 0]
  f = GroupNormLinf(T(0.8), groups, w)
  gb = vcat(groups, [[6, 8, 11, 13]])
  fb = GroupNormLinf(T(0.8), gb, vcat(w, T(1)))
  v = 2 .* randn(rng, T, 13)
  x = randn(rng, T, 13)
  x[[2, 7]] .= T(1.5); x[3] = T(0.5)                   # a tie on the face of group 1
  x[[1, 5, 9]] .= 0                                    # groups 2 and 3 are zero; groups 1 and 4 are not
  x[[6, 13]] .= T[0.3, 0]
  sx = SparseIterate(x)
  bx = BlockIterate(copy(x), gb)
  sc(α) = norm(v) + 8 * α + 8
  for α in (T(0.5), T(2), T(3.7), T(1e-3), T(1e3))
    s, sb = Scaled(f, α), Scaled(fb, α)
    fa = GroupNormLinf(α * f.λ0, groups, w)
    # the score of each group against the definition with the coefficient α λ0 w: the margin at a zero group and the distance at a
    # nonzero one (the route multiplies the score at the lazy gradient v/α by α, so its bound has a few more roundings)
    for (k, G) in enumerate(groups)
      i = first(G)
      r = kkt_violation(s, v, x, i)
      @test r isa T
      if k in (2, 3)
        @test kk_margin_ok(r, f.λ0, w[k], v, G; factor = α)
      else
        @test kk_ok(r, big(α) * big(f.λ0), w[k], v, x, G; slack = 1 + 6 / kk_linf_bound(length(G)))
      end
      @test kk_close(r, α * kkt_violation(f, v ./ α, x, i), T, sc(α))
      @test kk_close(r, kkt_violation(fa, v, x, i), T, sc(α))
      @test kk_close(kkt_violation(s, v, sx, i), r, T, sc(α))
      @test kk_close(kkt_violation(sb, v, bx, k), r, T, sc(α))
    end
    # an entry in no group scores |vᵢ|, also where x is nonzero
    @test isapprox(kkt_violation(s, v, x, 6), abs(v[6]); rtol = 8 * eps(T))
    # the full-point value
    R = kkt_violation(s, v, x)
    @test R isa T && kk_full_ok(R, f, v, x; α, slack = 1 + 6 / kk_linf_bound(1))
    @test kk_close(R, α * kkt_violation(f, v ./ α, x), T, sc(α))
    @test kk_full_ok(kkt_violation(sb, v, bx), fb, v, x; α, slack = 1 + 6 / kk_linf_bound(1))
  end
  # factors that merge: the product decides
  for (α1, α2) in ((T(2), T(3)), (T(0.5), T(1e-3)), (T(1e3), T(7)))
    s = Scaled(Scaled(f, α1), α2)
    a = setprecision(BigFloat, PREC) do
      big(α1) * big(α2)
    end
    @test kk_ok(kkt_violation(s, v, x, 2), a * big(f.λ0), w[1], v, x, groups[1]; slack = 1 + 12 / kk_linf_bound(3))
    @test kk_margin_ok(kkt_violation(s, v, x, 9), f.λ0, w[2], v, [9]; factor = a, extra = 6)
  end
  # the three-argument form is refused, as for the function itself
  @test_throws ArgumentError kkt_violation(Scaled(f, T(2)), v, 1)
  @test_throws ArgumentError kkt_violation(kk_chain(T, f), v, 1)
  # a zero factor has no representation for this function
  @test_throws ArgumentError Scaled(f, zero(T))
  # the arguments are checked before the factor is applied
  @test_throws ArgumentError kkt_violation(Scaled(f, T(2)), Complex{T}.(v), x, 2)
  @test_throws DimensionMismatch kkt_violation(Scaled(f, T(2)), v[1:12], x, 2)
  @test_throws DimensionMismatch kkt_violation(Scaled(f, T(2)), v[1:12], x)
  @test_throws BoundsError kkt_violation(Scaled(f, T(2)), v, x, 14)
  @test_throws DimensionMismatch kkt_violation(Scaled(fb, T(2)), v, BlockIterate(copy(x), [collect(1:13)]), 1)
  # the class of a function with a score
  @test @inferred(PB._kkt_class(typeof(Scaled(f, T(2))))) isa PB._KktScore
end

@testset "Scaled with an infinite factor ($T)" for T in (Float32, Float64)
  inf, t = T(Inf), kk_tiny(T)
  f = GroupNormLinf(T(0.8), [1:2, 3:3], T[1, 0])
  v = T[0.25, 0.5, 0.75, 1]                            # entry 4 is in no group
  si = Scaled(f, inf)
  # at a zero group the formula Inf times the score at v/α = 0: -Inf for a positive coefficient, and NaN for the coefficient zero
  @test kkt_violation(si, v, zeros(T, 4), 1) === -inf
  @test isnan(kkt_violation(si, v, zeros(T, 4), 3)) && isnan(kkt_violation(si, v, zeros(T, 4), 4))
  # at a nonzero group (a zero entry inside it included), at a nonzero entry in no group and in the full-point form: refused
  for (pt, k) in ((T[1, 0, 0, 0], 1), (T[1, 0, 0, 0], 2), (T[0, 0, 1, 0], 3), (T[0, 0, 0, 1], 4), (SparseIterate(T[1, 0, 0, 0]), 2),
                  (BlockIterate(T[1, 0, 0], [[1, 2], [3]]), 1))
    @test_throws ArgumentError kkt_violation(si, pt isa BlockIterate ? v[1:3] : v, pt, k)
  end
  @test occursin("infinite factor", kk_msg(() -> kkt_violation(si, v, T[1, 0, 0, 0], 1)))
  @test_throws ArgumentError kkt_violation(si, v, zeros(T, 4))
  @test_throws ArgumentError kkt_violation(si, v, T[1, 0, 0, 0])
  @test occursin("infinite factor", kk_msg(() -> kkt_violation(si, v, T[1, 0, 0, 0])))
  # stacks that nest an infinite factor with other wrappers, built by the parameterized constructor, are refused in the same cases. The
  # stack decides the route, so the first stack is tried on every kind of point and the other two on arrays
  stacks = (kk_raw(kk_raw(Scaled(f, inf), t), t), kk_raw(kk_raw(Scaled(f, t), inf), t), kk_raw(kk_chain(T, f), inf))
  rows = ((T[1, 0, 0, 0], 1), (T[1, 0, 0, 0], 2), (T[0, 0, 0, 1], 4))
  allrows = (rows..., (SparseIterate(T[1, 0, 0, 0]), 2), (BlockIterate(T[1, 0, 0], [[1, 2], [3]]), 1))
  for (depth, st) in enumerate(stacks)
    @test st.f isa Scaled
    depth == 1 && @test PB._chained(st)
    depth == 1 && @test occursin("infinite factor", kk_msg(() -> kkt_violation(st, v, T[1, 0, 0, 0], 1)))
    @test_throws ArgumentError kkt_violation(st, v, zeros(T, 4))
    @test_throws ArgumentError kkt_violation(st, v, T[1, 0, 0, 0])
    for (pt, k) in (depth == 1 ? allrows : rows)
      @test_throws ArgumentError kkt_violation(st, pt isa BlockIterate ? v[1:3] : v, pt, k)
    end
  end
end

# a chain of two wrappers whose product of factors is not a normal number: where the level folds into a normal level of the function the
# chain has the score of that function, where it does not the zero groups have the margin with the coefficient a λ w, formed from the
# significands and the exponents of the three, and a nonzero group, a zero entry inside it, a nonzero entry in no group and the
# full-point form are refused
@testset "chains of Scaled around GroupNormLinf ($T)" for T in (Float32, Float64)
  groups = [[1], [2, 3], [4, 5, 6]]                    # sizes 1, 2 and 3; entry 7 is in no group
  base = T[0.3, -0.4, 0.5, -0.2, 0.7, 0.1, 0.9]
  tiny, lvl, folded = kk_tiny(T), kk_lvl(T), ldexp(one(T), T === Float32 ? -100 : -200)
  a = setprecision(BigFloat, PREC) do
    big(tiny)^2
  end
  # the level folds: tiny² lvl = folded
  w = T[1, 2, 0.5]
  h = kk_chain(T, GroupNormLinf(lvl, groups, w))
  fl, p = PB._scaled_chain(h)
  @test PB._chained(h) && PB._kkt_resolve(fl, p) !== nothing
  fo = GroupNormLinf(folded, groups, w)
  v = folded .* base
  x0 = zeros(T, 7)
  x1 = T[0, 1, 0, 0.5, -0.5, 0.5, 4]                   # group 1 is zero, group 2 has a zero entry, entry 7 is in no group
  bx = BlockIterate(copy(x1[1:6]), groups)
  for (k, G) in enumerate(groups), i in G
    @test kk_margin_ok(kkt_violation(h, v, x0, i), folded, w[k], v, G)
    @test kk_close(kkt_violation(h, v, x0, i), kkt_violation(fo, v, x0, i), T, 4 * folded)
    k == 1 || @test kk_ok(kkt_violation(h, v, x1, i), folded, w[k], v, x1, G; slack = 1)
    k == 1 || @test kk_close(kkt_violation(h, v, x1, i), kkt_violation(fo, v, x1, i), T, 8 * folded)
  end
  @test kkt_violation(h, v, x1, 7) === abs(v[7])
  @test kk_ok(kkt_violation(h, v[1:6], bx, 3), folded, w[3], v, x1, groups[3])
  @test kk_full_ok(kkt_violation(h, v, x1), fo, v, x1)
  @test kk_full_ok(kkt_violation(h, v, x0), fo, v, x0)
  @test kkt_violation(h, v, x0) isa T
  # the level does not fold: the coefficient is a λ w with a λ that underflows in the product tiny² λ
  λn, wn = T === Float32 ? (ldexp(one(T), -40), ldexp(one(T), 120)) : (ldexp(one(T), -300), ldexp(one(T), 1000))
  c1 = T === Float32 ? ldexp(one(T), -120) : ldexp(one(T), -500)      # tiny² λn wn = 2^-120 or 2^-500, the coefficient of group 1
  wk = T[wn, wn / 2, wn / 4]
  hn = kk_chain(T, GroupNormLinf(λn, groups, wk))
  fn, pn = PB._scaled_chain(hn)
  @test PB._chained(hn) && PB._kkt_resolve(fn, pn) === nothing
  vn = c1 .* base
  bz = BlockIterate(zeros(T, 6), groups)
  for (k, G) in enumerate(groups)
    ref(i) = kkt_violation(hn, vn, x0, i)
    for i in G
      @test kk_margin_ok(ref(i), λn, wk[k], vn, G; factor = a)
    end
    @test kk_margin_ok(kkt_violation(hn, vn[1:6], bz, k), λn, wk[k], vn, G; factor = a)
  end
  @test kkt_violation(hn, vn, x0, 7) === abs(vn[7])                       # an entry in no group keeps |vᵢ|
  # the refusals, after the validation of the arguments
  xz = copy(x0); xz[2] = 1                              # group 2 is nonzero and entry 3 is zero inside it
  xf = copy(x0); xf[7] = 1                              # a nonzero entry in no group
  @test_throws ArgumentError kkt_violation(hn, vn, xz, 2)
  @test_throws ArgumentError kkt_violation(hn, vn, xz, 3)
  @test occursin("Scaled wrappers around GroupNormLinf", kk_msg(() -> kkt_violation(hn, vn, xz, 2)))
  @test_throws ArgumentError kkt_violation(hn, vn, xf, 7)
  @test kkt_violation(hn, vn, xz, 1) isa T && kkt_violation(hn, vn, xf, 1) isa T       # the zero group 1 is scored
  @test_throws ArgumentError kkt_violation(hn, vn[1:6], BlockIterate(T[0, 1, 0, 0, 0, 0], groups), 2)
  @test_throws ArgumentError kkt_violation(hn, vn, SparseIterate(xz), 3)
  @test_throws ArgumentError kkt_violation(hn, vn, x0)
  @test_throws ArgumentError kkt_violation(hn, vn, xz)
  @test occursin("Scaled wrappers around GroupNormLinf", kk_msg(() -> kkt_violation(hn, vn, x0)))
  @test_throws DimensionMismatch kkt_violation(hn, vn[1:5], xz, 2)
  @test_throws BoundsError kkt_violation(hn, vn, xz, 8)
  @test_throws DimensionMismatch kkt_violation(hn, vn[1:6], BlockIterate(T[0, 1, 1, 0, 0, 0], [[1, 2], [3], [4, 5, 6]]), 1)
  # the three-argument form is refused for both chains
  @test_throws ArgumentError kkt_violation(h, v, 2)
  @test_throws ArgumentError kkt_violation(hn, vn, 2)
  # a symmetric iterate has no groups: the refusal names the function and does not advise a form that it lacks
  S = SymmetricSparseIterate(T[0 1; 1 0])
  for g in (Scaled(GroupNormLinf(T(0.8), groups, w), T(2)), h, hn)
    for call in (() -> kkt_violation(g, ones(T, 2, 2), S), () -> kkt_violation(g, ones(T, 2, 2), S, 1), () -> kkt_violation(g, ones(T, 2, 2), S, 2))
      @test_throws ArgumentError call()
      m = kk_msg(call)
      @test occursin("GroupNormLinf", m) && occursin("SymmetricSparseIterate", m) && !occursin("use the coordinate form", m)
    end
  end
end

# warmed allocations of the scores for Array and Matrix arguments, range and vector groups, a BlockIterate point and one Scaled
# wrapper; the arguments are not written
@testset "allocations and inference of the scores ($T, $layout)" for T in (Float32, Float64), layout in (:range, :perm)
  rng = Xoshiro(1017)
  sizes = [4, 33, 130, 1, 7]
  groups, p = partition(rng, sizes, layout)
  f = GroupNormLinf(T(0.3), groups, T[1, 0.5, 2, 1, 0.25])
  v, x = randn(rng, T, p), randn(rng, T, p)
  x[groups[2]] .= 0                                   # group 2 is zero
  zg, nz = first(groups[2]), first(groups[3])
  v0, x0 = copy(v), copy(x)
  q = p - p % 2
  Mv, Mx = reshape(v[1:q], 2, :), reshape(x[1:q], 2, :)
  parts = layout === :range ? [1:4, 5:37, 38:167, 168:168, 169:175] : [collect(1:4), collect(5:37), collect(38:167), [168], collect(169:175)]
  fp = GroupNormLinf(T(0.3), parts, T[1, 0.5, 2, 1, 0.25])
  xb = x[1:175]
  xb[parts[2]] .= 0
  bx = BlockIterate(xb, parts)
  for h in (f, Scaled(f, T(2.5)))
    for (g, pt) in ((v, x), (Mv, Mx))
      @test (@inferred kkt_violation(h, g, pt)) isa T
      @test (@inferred kkt_violation(h, g, pt, zg)) isa T
      @test (@inferred kkt_violation(h, g, pt, nz)) isa T
      @test allocs(kkt_violation, h, g, pt) == 0
      @test allocs(kkt_violation, h, g, pt, zg) == 0
      @test allocs(kkt_violation, h, g, pt, nz) == 0
    end
  end
  for h in (fp, Scaled(fp, T(2.5)))
    @test (@inferred kkt_violation(h, v[1:175], bx)) isa T
    @test (@inferred kkt_violation(h, v[1:175], bx, 2)) isa T
    @test (@inferred kkt_violation(h, v[1:175], bx, 3)) isa T
    @test allocs(kkt_violation, h, v[1:175], bx) == 0
    @test allocs(kkt_violation, h, v[1:175], bx, 2) == 0
    @test allocs(kkt_violation, h, v[1:175], bx, 3) == 0
  end
  # cold paths of the nonzero group nz (all x equal, so that the face is the whole group; and the zero point): a gradient whose squares
  # underflow in Float32, a gradient near floatmax, a NaN gradient, and a zero point
  for (vv, xx) in ((fill(T(1e-30), p), fill(T(2), p)), (fill(floatmax(T) / 8, p), ones(T, p)), (fill(T(NaN), p), ones(T, p)), (ones(T, p), zeros(T, p)))
    @test allocs(kkt_violation, f, vv, xx, nz) == 0
    @test allocs(kkt_violation, f, vv, xx) == 0
  end
  @test v == v0 && x == x0
end

@testset "allocations ($T)" for T in (Float32, Float64)
  rng = Xoshiro(963)
  groups, p = partition(rng, [1, 2, 4, 7, 16, 33, 64, 5, 12, 3], :perm)
  x = gen(rng, T, p, 1)
  λ0, γ = T(0.6), T(0.9)
  w = fraction_weights(rng, T, x, groups, γ * λ0, (0.05, 0.3, 0.9))
  f = GroupNormLinf(λ0, groups, w)
  y = zeros(T, p)
  @test allocs(prox!, y, f, x, γ) == 0
  @test allocs(value_and_prox!, y, f, x, γ) == 0
  @test allocs(f, x) == 0
  xi = copy(x)
  @test allocs(prox!, xi, f, xi, γ) == 0
  xi = copy(x)
  @test allocs(value_and_prox!, xi, f, xi, γ) == 0
  # ranges, the zero and identity exits, an infinite step, a zero step
  fr = GroupNormLinf(λ0, [1:3, 4:9, 10:10, 11:20], T[0.9, 0, 2, T(Inf)])
  xr = gen(rng, T, 22, 1)
  yr = zeros(T, 22)
  for s in (zero(T), T(0.05), T(0.5), T(5), T(Inf))
    @test allocs(prox!, yr, fr, xr, s) == 0
    @test allocs(value_and_prox!, yr, fr, xr, s) == 0
  end
  # exceptional branches: a NaN group, an infinite group, a sum above floatmax/2
  xn = copy(x); xn[groups[3][1]] = T(NaN)
  xf = copy(x); xf[groups[4][1]] = T(Inf)
  xb = copy(x)
  for i in 1:3
    xb[groups[6][i]] = floatmax(T) / 4
  end
  for z in (xn, xf, xb)
    @test allocs(prox!, y, f, z, γ) == 0
    @test allocs(value_and_prox!, y, f, z, γ) == 0
    zz = copy(z)
    @test allocs(prox!, zz, f, zz, γ) == 0
    @test allocs(f, z) == 0
  end
  # a Matrix argument, whose linear indices a view would reshape
  X = reshape(copy(x[1:p - mod(p, 4)]), 4, :)
  mg = [[1, 5, 9], 2:4, [10, 6, 7, 8], 11:11]
  fm = GroupNormLinf(λ0, mg, T[0.5, 0.2, 1.1, 3])
  Y = zeros(T, size(X))
  @test allocs(prox!, Y, fm, X, γ) == 0
  @test allocs(value_and_prox!, Y, fm, X, γ) == 0
  @test allocs(fm, X) == 0
  @test same(vec(prox(fm, X, γ)), prox(fm, vec(X), γ))
  # compact blocks: a sweep over groups of different sizes, with an output that is not the input, and in place
  U = [x[G] for G in groups]
  outs = [similar(u) for u in U]
  @test allocs(sweep!, outs, f, U, γ) == 0
  @test allocs(sweep!, U, f, U, T(1e-3) * γ) == 0
  @test allocs(prox_block!, outs[3], f, U[3], 3, γ) == 0
end

end
