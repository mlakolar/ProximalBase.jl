module GroupNormLinfTest
# GroupNormLinf (#956): f(x) = λ0 Σₖ wₖ max_{i∈Gₖ} |xᵢ| and its proximal operators, the clip of each group at the level τₖ
# with Σ_{i∈Gₖ}(|xᵢ| - τₖ)₊ = γλ0wₖ (and +0 when γλ0wₖ ≥ ‖x[Gₖ]‖₁), against an independent sort-based ℓ1-ball threshold
# written in this file in BigFloat, in Float32 and Float64. The established reference cases keep their tolerance of
# TOL_C n eps(T) max|xᵢ| plus two subnormal spacings. The documented bound allows (2n + 1) subnormal spacings to
# account for the shared threshold and the rounded radius. A group equal to the whole argument is also
# compared with NormLinf (bit for bit up to 8 entries, and above that both within the bound of the reference, since the passes
# of a handover sum in an order that depends on the layout), and singleton groups with NormL1: the proximal outputs bit for
# bit, the values within a tolerance.
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
  # Inputs with a normal γλ and values in the ordinary range. Up to 8 entries Condat's pass cannot hand over (its count is not tested
  # above 5), so the level comes from the entries read one by one into a dense buffer, whatever the layout of the argument: the
  # level, the clip and the value agree bit for bit. From 9 entries on the pass can hand over to Michelot's iteration (from
  # n = 16 on here), whose passes run over the argument, from the sum of the caller's pass: the order of those sums depends on
  # how the layout (an Array, or the group read through its indices) vectorizes, so both outputs and both values are compared
  # with the BigFloat reference, within the bound of the level that the file uses, and not with each other. The value at the
  # argument has no threshold and agrees bit for bit at each size
  for n in (1, 2, 3, 5, 16, 17, 100, 300, 1000), kind in 1:3
    x = gen(rng, T, n, kind)
    s1 = sum(abs, x)
    for frac in (1e-9, 1e-6, 1e-3, 0.05, 0.3, 0.7, 0.95, 1.05, 1.5)
      γ = T(frac * Float64(s1) / Float64(λ))
      f, h = GroupNormLinf(λ, [1:n]), NormLinf(λ)
      y, yh = zeros(T, n), zeros(T, n)
      v, vh = value_and_prox!(y, f, x, γ), value_and_prox!(yh, h, x, γ)
      yp = prox(f, x, γ)
      xi = copy(x)
      @test prox!(xi, f, xi, γ) === xi && f(x) === h(x)
      if n <= 8
        @test same(y, yh) && v === vh
        @test same(yp, yh) && f(y) === h(yh)
        @test same(xi, yh)
      else
        # the three outputs of the group (value_and_prox!, prox and prox! in place) read the same array through the same reader:
        # one layout and one path, so they agree bit for bit
        @test same(yp, y) && same(xi, y)
        ref, τs = ref_prox(x, [1:n], [one(T)], γ, λ)
        @test all(z -> close_prox(z, ref, x), (y, yh, yp, xi))
        @test close_value(v, x, [1:n], [one(T)], λ, τs) && close_value(vh, x, [1:n], [one(T)], λ, τs)
      end
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
  # a group of 5000 entries without ties: through the prox (with the sum of the group) its candidates pass the early count
  # max(8, n ÷ 8) = 625 while the level of Condat's pass is below the plain level of all the entries, so that Michelot's pass decides
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
    # without the caller's sum the pass hands over at a test with max(1024, n ÷ 8) = 1024 in force. For this input the bound
    # 1024 + 4 holds: the entries are below the radius, so they join the candidates and do not restart them, the count after g
    # groups of four is 1 + 4g, the test at the count 629 (above the early count 625) puts 1024 in force, and the next test finds
    # the count at most four above 1024 (here 1025). A buffer of 1024 + 4 entries therefore suffices here, and the call returns
    # Michelot's level (checked where a write past the buffer throws). The bound is not general: with an early count of 1021 to
    # 1023 the count can reach 1024 + 7 (test/normlinf.jl); a buffer of the length of the group suffices
    buf = Vector{T}(undef, 1024 + 4)
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
  ref, _ = ref_prox(x, groups, w, γ, λ0)
  for k in eachindex(groups)
    G = groups[k]
    m = length(G)
    u = x[G]
    # A group of at most 8 entries cannot hand over (after its one group of four the count is at most 5, not above the count 8
    # in force, so no test is made), and Condat's pass reads its entries one by one into a dense buffer: the compact block agrees
    # with the group of the array operator bit for bit. A larger group can hand over to Michelot's iteration, whose passes run
    # over the argument: the order of their sums, and of the caller's sum, depends on how the layout (the compact block, or the
    # group read through its indices in x) vectorizes, so both are compared with the BigFloat reference within the bound of the
    # level. The compact forms that read the vector u, or its copy, agree with one another bit for bit; a view of another array
    # as the input is held to the same rule as the array operator
    agrees(z) = m <= 8 ? same(z, y[G]) : close_prox(z, ref[G], u) && close_prox(y[G], ref[G], u)
    out = prox_block(f, u, k, γ)
    @test agrees(out)
    out2 = fill(T(7), m)
    @test prox_block!(out2, f, u, k, γ) === out2 && same(out2, out)
    ui = copy(u)
    @test prox_block!(ui, f, ui, k, γ) === ui && same(ui, out)
    # a view of an array as the input, a view as the output
    big_u = vcat(u, u)
    @test agrees(prox_block!(fill(T(7), m), f, view(big_u, 1:m), k, γ))
    big_out = fill(T(7), 2m)
    ov = view(big_out, m+1:2m)
    @test prox_block!(ov, f, u, k, γ) === ov && same(ov, out) && all(==(T(7)), view(big_out, 1:m))
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

@testset "optimality at a zero group ($T)" for T in (Float32, Float64)
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
  @test_throws ArgumentError kkt_violation(fb, g, B, 2)                         # a nonzero group
  @test kkt_violation(fb, g, B, 1) === T(0.25)
  @test_throws DimensionMismatch kkt_violation(fb, g, BlockIterate(T, 4, [1:1, 2:3, 4:4]), 1)   # a group differs
  @test_throws DimensionMismatch kkt_violation(fb, g, BlockIterate(T, 4, [1:3, 4:4]), 1)        # the group count differs
  @test_throws BoundsError kkt_violation(fb, g, B, 4)
  @test_throws DimensionMismatch kkt_violation(fb, T[1, 2], B, 1)                # the gradient length
  # an array point: the group of the entry, with checks of length and extent
  @test_throws DimensionMismatch kkt_violation(f, T[1, 2, 3], z, 1)
  @test_throws DimensionMismatch kkt_violation(f, T[1, 2], T[0, 0], 1)           # the groups reach index 3
  @test_throws BoundsError kkt_violation(f, g, z, 5)
  # a nonzero group (NaN and infinite entries included): the entry selects its group
  for (xnz, i) in ((T[0, 1, 0, 0], 1), (T[NaN, 0, 0, 0], 2), (T[0, 0, Inf, 0], 3), (T[1e-30, 0, 0, 0], 1), (T[0, 0, 1, 0], 3))
    @test_throws ArgumentError kkt_violation(f, g, xnz, i)
  end
  @test kkt_violation(f, g, T[0, 0, 1, 0], 1) === T(0.25)
  # the full-point and three-argument forms are not provided
  @test_throws ArgumentError kkt_violation(f, g, z)
  @test_throws ArgumentError kkt_violation(f, g, T[0, 1, 0, 0])
  @test_throws ArgumentError kkt_violation(fb, g, B)
  @test_throws ArgumentError kkt_violation(f, g, 1)
  @test_throws ArgumentError kkt_violation(GroupNormLinf(one(T), Vector{Vector{Int}}()), T[], T[])
  # a symmetric iterate has no groups
  S = SymmetricSparseIterate(T, 3)
  @test_throws ArgumentError kkt_violation(GroupNormLinf(one(T), [1:2]), zeros(T, 9), S, 1)
  @test_throws ArgumentError kkt_violation(GroupNormLinf(one(T), [1:2]), zeros(T, 9), S)
  # a function of this package without a score under Scaled has a MethodError
  @test_throws MethodError kkt_violation(Scaled(f, T(2)), g, z, 1)
  @test_throws MethodError kkt_violation(Scaled(f, T(2)), g, z)
  @test allocs(kkt_violation, f, g, z, 1) == 0 && allocs(kkt_violation, f, g, z, 4) == 0
  @test allocs(kkt_violation, fb, g, B, 1) == 0
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
