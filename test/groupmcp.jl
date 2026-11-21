module GroupMCPTest
# GroupMCP: values, array and compact block proximal operators, the zero-group KKT screen and the refused methods,
# against BigFloat evaluations, independent of the package, of
#   f(x) = Σₖ p(‖x[Gₖ]‖₂; λ0 wₖ, a),   prox_{γf}(u)[Gₖ] = (t*/r) u[Gₖ],  r = ‖u[Gₖ]‖₂,
# where t* minimizes p(t) + (t - r)²/(2γ) over t ≥ 0 (the scalar reference of mcp_reference.jl, which does not use the
# package: a candidate comparison at 256 bits, repeated at 4096 bits for near-ties), with r computed in 512-bit BigFloat,
# in Float32 and Float64.
using Test, ProximalBase, Random, OffsetArrays
using ..TestUtils

include("mcp_reference.jl")
using .MCPReferenceTest

const PREC = 512

# ---- references (BigFloat) ----
bignorm(u) = setprecision(() -> sqrt(sum(big.(u) .^ 2; init = big(0))), BigFloat, PREC)
biglevel(λ0, w) = setprecision(() -> big(λ0) * big(w), BigFloat, PREC)
# p(t; λ, a) for t ≥ 0, λ > 0 (a = Inf: λt)
bigp(t, λ, a) = isinf(a) ? λ * t : (t <= a * λ ? λ * t - t^2 / (2 * big(a)) : big(a) * λ^2 / 2)

# The proximal point of one group with finite entries at positive finite λ0, w and γ: the scalar reference at the
# norm computed in 512-bit BigFloat, along u.
function refgroup(u, λ0, w, a, γ)
  setprecision(BigFloat, PREC) do
    U = big.(u)
    r = bignorm(u)
    iszero(r) && return zero.(U)
    t = mcp_reference(biglevel(λ0, w), a, r, γ)
    (t / r) .* U
  end
end
# The value of one group with finite entries, positive finite λ0 and w.
refterm(u, λ0, w, a) = setprecision(() -> bigp(bignorm(u), biglevel(λ0, w), a), BigFloat, PREC)
# The group objective p(‖y‖) + ‖y - u‖²/(2γ), finite y and u, positive finite λ0, w and γ.
function refobjective(y, u, λ0, w, a, γ)
  setprecision(BigFloat, PREC) do
    Y, U = big.(y), big.(u)
    bigp(bignorm(y), biglevel(λ0, w), a) + sum((Y .- U) .^ 2; init = big(0)) / (2 * big(γ))
  end
end
# :zero, :keep or :firm, the reference decision at the norm r
function decision(r, λ0, w, a, γ)
  t = setprecision(() -> mcp_reference(biglevel(λ0, w), a, r, γ), BigFloat, PREC)
  iszero(t) ? :zero : t == r ? :keep : :firm
end

# The checks of one group y = prox at u (finite entries; positive finite λ0, w, γ):
#  * the objective at y is within tol of the reference minimum, with
#      tol = (n + 8) eps S (4 + (n + 8) eps κ) + n δ (2r/γ + λ),
#    S = r²/(2γ) + p(r), κ = a/(a - γ) for finite a and γ < a (0 otherwise), n = |G|, δ = nextfloat(zero(T)): the effect of an
#    error of (n + 2) eps in the computed norm on the decisions and on the firm point (whose slope is κ), and the
#    absolute rounding of subnormal outputs;
#  * when the reference decision is the same at r (1 ± (n + 8) eps), y is u (kept), +0 (zero), or within
#    ((n + 8) κ + 8) eps max|u| + 2δ of the reference in each entry (firm).
function check_group(y::AbstractVector{T}, u, λ0, w, a, γ) where {T}
  n = length(u)
  r = bignorm(u)
  λ = biglevel(λ0, w)
  ε, δ = big(eps(T)), big(nextfloat(zero(T)))
  κ = γ < a && isfinite(a) ? big(a) / (big(a) - big(γ)) : big(0)
  ref = refgroup(u, λ0, w, a, γ)
  F = refobjective(y, u, λ0, w, a, γ)
  Fmin = refobjective(ref, u, λ0, w, a, γ)
  S = setprecision(() -> r^2 / (2 * big(γ)) + bigp(r, λ, a), BigFloat, PREC)
  tol = (n + 8) * ε * S * (4 + (n + 8) * ε * κ) + n * δ * (2r / big(γ) + λ)
  okobj = F - Fmin <= tol
  ρ = (n + 8) * ε
  d = decision(r, λ0, w, a, γ)
  robust = d == decision(r * (1 - ρ), λ0, w, a, γ) == decision(r * (1 + ρ), λ0, w, a, γ)
  okdec = if !robust
    true
  elseif d == :keep
    isequal(y, T.(u))
  elseif d == :zero
    all(isequal(zero(T)), y)
  else
    m = maximum(abs, big.(u))
    all(i -> abs(big(y[i]) - ref[i]) <= ((n + 8) * κ + 8) * ε * m + 2δ, eachindex(y))
  end
  okobj && okdec
end

# A value against its reference, for K groups of at most n entries, with ρ = (n + K + 8) eps: NaN only for a NaN
# reference; beyond floatmax(T) (1 + ρ) the infinity of the sign of the reference; within ρ of floatmax(T) that infinity
# or a finite value within ρ; below, a finite value within ρ of the reference and 4K subnormal spacings.
function close_value(v::T, ref, n, K) where {T}
  isnan(ref) && return isnan(v)
  isnan(v) && return false
  ρ = (n + K + 8) * big(eps(T))
  M = big(floatmax(T))
  abs(ref) > M * (1 + ρ) && return v == copysign(T(Inf), ref)
  abs(ref) > M * (1 - ρ) && v == copysign(T(Inf), ref) && return true
  isfinite(v) && abs(big(v) - ref) <= ρ * abs(ref) + 4K * big(nextfloat(zero(T)))
end

# The fused value against the value at the stored output, for groups of at most n entries and K groups: the same terms,
# except that the sum of squares of a group scaled on the ordinary path is accumulated as its entries are stored and can
# associate differently, within (n + 4) eps of the value (the terms are nonnegative) and 4K subnormal spacings.
# Nonfinite values must be equal (isequal).
fused_close(v::T, w, n, K) where {T} = isequal(v, w) ||
  (isfinite(v) && isfinite(w) && abs(big(v) - big(w)) <= (n + 4) * big(eps(T)) * abs(big(w)) + 4K * big(nextfloat(zero(T))))

# a random vector of n entries and norm close to r (r a BigFloat), in T
function vector_with_norm(rng, ::Type{T}, n, r) where {T}
  d = randn(rng, n)
  T.(big.(d) .* (r / bignorm(d)))
end

dense(x) = Array(x)
snapshot(x::AbstractArray) = copy(Array(x))

@testset "construction and validation" begin
  for T in (Float32, Float64)
    f = GroupMCP(T(0.5), [1:2, [3, 5, 7]])
    @test f isa GroupMCP{T}
    @test f.a === T(3)
    @test f.w == T[sqrt(T(2)), sqrt(T(3))]             # default √|Gₖ| in T
    @test GroupMCP(T(0.5), [1:2], T[3]).w == T[3]
    @test GroupMCP(T(0.5), [1:2]; a = 2).a === T(2)
    @test GroupMCP(T(0.5), [1:2]; a = T(Inf)).a === T(Inf)
    w = T[1, 2]; G = [[2, 1], [4, 3]]
    g = GroupMCP(one(T), G, w)
    w[1] = 9; G[1][1] = 7
    @test g.w == T[1, 2] && g.groups == [[2, 1], [4, 3]]
    @test g.group_of == [1, 1, 2, 2] && g.free == Int[]
    h = GroupMCP(one(T), [[5, 2], 7:8])
    @test h.free == [1, 3, 4, 6]
  end
  @test GroupMCP(1f0, [1:2]).λ0 isa Float32               # the default a = 3 does not promote
  @test GroupMCP(1f0, [1:2]; a = 3.0).λ0 isa Float64       # a takes part in the promotion
  @test GroupMCP(1, [1:2]) isa GroupMCP{Float64}
  @test GroupMCP(1, [1:2], [2]) isa GroupMCP{Float64}
  @test GroupMCP(big(1), [1:2]) isa GroupMCP{BigFloat}
  for λ in (-1.0, NaN, -Inf, -big"1e-400")
    @test_throws ArgumentError GroupMCP(λ, [1:2])
  end
  for a in (0, -0.0, -1, NaN, -Inf)
    @test_throws ArgumentError GroupMCP(1.0, [1:2]; a)
  end
  for w in ([-1.0], [NaN], [-big"1e-400"])
    @test_throws ArgumentError GroupMCP(1.0, [1:2], w)
  end
  @test_throws ArgumentError GroupMCP(1.0, [1:2], [1.0 + 2im])
  @test_throws DimensionMismatch GroupMCP(1.0, [1:2], [1.0, 2.0])
  @test_throws ArgumentError GroupMCP(1.0, [Int[]])
  @test_throws ArgumentError GroupMCP(1.0, [[0, 1]])
  @test_throws ArgumentError GroupMCP(1.0, [1:2, 2:3])
  @test_throws ArgumentError GroupMCP{Float64, Vector{UnitRange{Int}}}(1.0, 0.0, [1:2], [1.0])
  @test_throws ArgumentError GroupMCP{Float64, Vector{UnitRange{Int}}}(-1.0, 3.0, [1:2], [1.0])
  f = GroupMCP(0.5, [1:2, 3:4])
  @test isproximable(f) && !isconvex(f) && !issmooth(f) && !isseparable(f)
  @test isproximable(GroupMCP) && !isconvex(GroupMCP)
  # BlockIterate constructors with the groups of f
  b = BlockIterate(4, f)
  @test b isa BlockIterate{Float64} && ncoordinates(b) == 2 && iszero(Vector(b))
  @test BlockIterate(Float32, 4, f) isa BlockIterate{Float32}
  c = BlockIterate([1.0, 0.0, 0.0, 0.0], f)
  @test nactive(c) == 1 && isactive(c, 1)
  @test_throws ArgumentError BlockIterate(5, f)                # the groups must partition 1:p
  # empty group set: zero value, identity proximal map
  e = GroupMCP(1.0, Vector{UnitRange{Int}}())
  @test e([1.0, NaN]) === 0.0 && e([1.0, -2.0]) === 0.0
  @test isequal(prox(e, [1.0, -0.0, NaN], 2.0), [1.0, -0.0, NaN])
end

@testset "worked examples ($T)" for T in (Float32, Float64)
  f = GroupMCP(one(T), [1:2], T[1])                        # λ = 1, a = 3
  pb(u, γ) = prox_block(f, T.(u), 1, T(γ))
  # γ = 1 < a: firm thresholding between γλ = 1 and aλ = 3
  @test pb([3, 4], 1) == T[3, 4]                           # r = 5 > aλ: kept
  @test pb([0.6, 0.8], 1) == T[0, 0]
  @test pb([1.2, 1.6], 1) ≈ T[0.9, 1.2]                    # r = 2: t* = 1.5
  @test check_group(pb([1.2, 1.6], 1), T[1.2, 1.6], 1, 1, 3, 1)
  # γ = a = 3: hard, kept when r² > aγλ² = 9
  @test pb([1.8, 2.4], 3) == T[0, 0] && pb([3, 4], 3) == T[3, 4]
  # γ = 12 > a: kept when r² > 36
  @test pb([3, 4], 12) == T[0, 0] && pb([6, 8], 12) == T[6, 8]
  # the value: the plateau aλ²/2 above aλ, λr - r²/(2a) below it
  @test f(T[3, 4]) == T(1.5)
  @test f(T[0.9, 1.2]) ≈ T(1.125)
  @test f(T[1.5, 0]) == T(1.125)
  # exactly representable ties give +0 (3-4-5 groups and powers of two)
  for s in (one(T), T(2)^-20, T(2)^20), (γ, a) in ((5, 8), (5, 5), (25, 1))
    g = GroupMCP(s, [1:2], T[1]; a = T(a))            # λ = s, r = 5s: r = γλ, or r² = aγλ²
    @test all(isequal(zero(T)), prox_block(g, s .* T[3, -4], 1, T(γ)))
    @test prox_block(g, s .* T[3, -4], 1, nextfloat(T(γ))) == T[0, 0]
  end
  g = GroupMCP(one(T), [1:2], T[1]; a = T(5))
  @test prox_block(g, T[3, -4], 1, one(T)) == T[3, -4]          # r = aλ: kept
  h = GroupMCP(T(0.5), [1:1], T[1]; a = T(4))
  @test prox_block(h, T[-2], 1, T(4)) == T[0]                   # r² = aγλ² = 4
  @test prox_block(h, T[-2], 1, prevfloat(T(4))) == T[-2]       # γ < a: r = aλ, kept
end

@testset "reference grid ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1069 + sizeof(T))
  sizes = (1, 2, 4, 32, 512)
  levels = T[0.5, 1, 3]
  shapes = T[1e-3, 0.5, prevfloat(T(1)), 1, nextfloat(T(1)), 3, 1e3, Inf]
  nok = ncase = 0
  for trial in 1:300
    n = sizes[rand(rng, 1:length(sizes))]
    λ0 = rand(rng, levels)
    w = rand(rng, (one(T), sqrt(T(n)), T(0.25)))
    a = rand(rng, shapes)
    base = isfinite(a) ? a : T(2)
    γ = rand(rng, (base / 10, prevfloat(base), base, nextfloat(base), 4base, one(T)))
    λ = biglevel(λ0, w)
    # norms around the thresholds of the step
    cuts = γ < a ? (big(γ) * λ, (isfinite(a) ? big(a) : 2big(γ)) * λ) : (λ * sqrt(big(a) * big(γ)),)
    c = rand(rng, cuts)
    r = c * rand(rng, (big"0.5", 1 - big(eps(T)), big(1), 1 + big(eps(T)), big"1.5", big(4)))
    u = vector_with_norm(rng, T, n, r)
    y = prox_block(GroupMCP(λ0, [1:n], [w]; a), u, 1, γ)
    ok = check_group(y, u, λ0, w, a, γ)
    ncase += 1
    nok += ok
    ok || @info "reference grid failure" T n λ0 w a γ
    # the value of the input and of the stored output
    g = GroupMCP(λ0, [1:n], [w]; a)
    @test close_value(g(u), refterm(u, λ0, w, a), n, 1)
    z = similar(u)
    v = value_and_prox!(z, g, u, γ)
    @test (z == y || check_group(z, u, λ0, w, a, γ)) && fused_close(v, g(z), n, 1)
  end
  @test nok == ncase
  # several groups in one array: permuted, noncontiguous, with entries in no group before, between and after
  groups = [[5, 2], 7:9, [1], [12, 10, 11]]
  w = T[1, 0.5, 2, 1]
  free = [3, 4, 6, 13, 14]
  for (a, γ) in ((T(3), T(1)), (T(3), T(3)), (T(2), T(5)), (T(Inf), T(1)))
    f = GroupMCP(T(0.7), groups, w; a)
    for trial in 1:20
      x = T(2) .* randn(rng, T, 14)
      trial % 4 == 0 && (x[groups[2]] .= 0)
      trial % 5 == 0 && (x[groups[4]] .*= T(0.05))
      y = prox(f, x, γ)
      @test y[free] == x[free]
      for k in eachindex(groups)
        @test check_group(y[groups[k]], x[groups[k]], T(0.7), w[k], a, γ)
        out = fill(T(NaN), length(groups[k]))
        @test prox_block!(out, f, x[groups[k]], k, γ) === out
        @test isapprox(out, y[groups[k]]; rtol = 0, atol = 64eps(T) * maximum(abs, x))
      end
      ref = setprecision(() -> sum(refterm(x[groups[k]], T(0.7), w[k], a) for k in eachindex(groups)), BigFloat, PREC)
      @test close_value(f(x), ref, 3, 4)
      for init in (zeros(T, 14), fill(T(NaN), 14))
        @test prox!(init, f, x, γ) === init && init == y
      end
      z = fill(T(NaN), 14)
      v = value_and_prox!(z, f, x, γ)
      @test z == y && fused_close(v, f(z), 3, 4)
      xi = copy(x)
      @test value_and_prox!(xi, f, xi, γ) === v && xi == y            # exact in-place use
    end
  end
end

@testset "scalar MCP agreement of a singleton group ($T)" for T in (Float32, Float64)
  rng = Xoshiro(17 + sizeof(T))
  δ, M = nextfloat(zero(T)), floatmax(T)
  params = [(T(0.5), T(2)), (T(1), T(3)), (T(2), T(Inf)), (T(1), T(0.25))]
  for (λ, a) in params, γ in T[0.1, 1, 2, 3, 8], trial in 1:20
    v = T(randn(rng)) * T(4) * λ
    s = prox_block(MCP(λ, a), v, 1, γ)
    g = prox_block(GroupMCP(λ, [1:1], T[1]; a), T[v], 1, γ)[1]
    if isfinite(a)
      # the decisions agree; the firm values within two ulps and a subnormal spacing (the group divides by the norm)
      @test iszero(s) == iszero(g) && (s == v) == (g == v)
      @test abs(g - s) <= 2eps(abs(s)) + δ
    else
      # GroupNormL2's shrinkage 1 - γλ/r: within 4 eps of the input
      @test abs(g - s) <= 4eps(abs(v))
    end
  end
  # extremes
  E = exponent(floatmax(T))
  for (λ, v, γ, a) in ((T(2)^(E - 23), T(2)^-30, T(2)^(-E - 36), T(3)),       # a subnormal step: firm
                       (T(2)^(-E + 30), T(2)^30, T(1.5) * T(2)^E, T(1.5) * T(2)^E),   # hard: zero
                       (M / 4, M / 2, T(0.5), T(3)), (T(4)δ, T(8)δ, T(1), T(3)), (one(T), M, T(3), T(3)))
    s = prox_block(MCP(λ, a), v, 1, γ)
    g = prox_block(GroupMCP(λ, [1:1], T[1]; a), T[v], 1, γ)[1]
    @test iszero(s) == iszero(g) && abs(g - s) <= 2eps(abs(s)) + δ
  end
end

@testset "cold-path counterexamples" begin
  # λ = 2^1000 is normal; in units of r = 2^-30 the level (2^1030) overflows, and the step is subnormal:
  # γλ = 2^-60 < r < aλ, firm
  g = GroupMCP(2.0^1000, [1:1], [1.0])
  y = prox_block(g, [2.0^-30], 1, 2.0^-1060)
  @test y[1] == prox_block(MCP(2.0^1000, 3.0), 2.0^-30, 1, 2.0^-1060) == 2.0^-30 - 2.0^-60
  # the same decision from the parts of a sum of squares that underflows (the scaled path)
  g2 = GroupMCP(2.0^490, [1:2], [1.0])
  @test prox_block(g2, [2.0^-540, 0.0], 1, 2.0^-1060) == [2.0^-540 - 2.0^-570, 0.0]
  @test check_group(prox_block(g2, [2.0^-540, 0.0], 1, 2.0^-1060), [2.0^-540, 0.0], 2.0^490, 1.0, 3.0, 2.0^-1060)
  # λ = 2^-993 is normal; in units of r = 2^30 the level (2^-1023) is subnormal, and aγ overflows:
  # r² = 2^60 ≤ aγλ² = 2.25 2^60, zero
  A = 1.5 * 2.0^1023
  h = GroupMCP(2.0^-993, [1:1], [1.0]; a = A)
  @test prox_block(h, [2.0^30], 1, A) == [0.0]
  @test prox_block(h, [2.0^30 * 1.5], 1, A) == [0.0]          # r² = aγλ², a tie
  @test prox_block(h, [nextfloat(2.0^30 * 1.5)], 1, A) == [nextfloat(2.0^30 * 1.5)]
  h2 = GroupMCP(2.0^-993, [1:2], [1.0]; a = A)                # the sum of squares underflows: the scaled path
  @test prox_block(h2, [2.0^-1000, 0.0], 1, A) == [0.0, 0.0]
  # an underflowed finite product λ0 w is not a stored zero: γλ ≈ 1e-300 > r, zero for γ < a
  u = [1e-310, 2e-310]
  for a in (1e200, Inf)
    @test prox_block(GroupMCP(1e-200, [1:2], [1e-200]; a), u, 1, 1e100) == [0.0, 0.0]
  end
  # at a = 3 the step is in the hard regime: kept, as r² > aγλ²
  @test prox_block(GroupMCP(1e-200, [1:2], [1e-200]), u, 1, 1e100) == u
  @test check_group(prox_block(GroupMCP(1e-200, [1:2], [1e-200]), u, 1, 1e100), u, 1e-200, 1e-200, 3.0, 1e100)
  # an overflowed finite product λ ≈ 2e308 is not a stored infinity: r² > aγλ² ≈ 4e296, kept, value aλ²/2 ≈ 2e296
  f = GroupMCP(2e154, [1:2], [1e154]; a = 1e-320)
  u2 = [3e300, 4e300]
  @test prox_block(f, u2, 1, 1.0) == u2 && check_group(prox_block(f, u2, 1, 1.0), u2, 2e154, 1e154, 1e-320, 1.0)
  @test prox_block(f, [3e147, 4e147], 1, 1.0) == [0.0, 0.0]
  @test isfinite(f(u2)) && close_value(f(u2), refterm(u2, 2e154, 1e154, 1e-320), 2, 1)
  @test value_and_prox!(similar(u2), f, u2, 1.0) === f(u2)
end

@testset "constructor ownership and default weights" begin
  # the outer constructors copy the groups and the weights
  G = [[2], [1]]
  w = [1.0, 2.0]
  weighted = GroupMCP(1.0, G, w)
  defaulted = GroupMCP(1.0, G)
  @test weighted.groups[1] !== G[1] && weighted.w !== w && defaulted.groups[1] !== G[1]
  G[1][1] = 3
  w[1] = 4
  @test weighted.groups == defaulted.groups == [[2], [1]] && weighted.w == [1.0, 2.0] && defaulted.w == [1.0, 1.0]
  @test weighted.group_of == defaulted.group_of == [2, 1]
  # the typed constructor validates and stores its arguments as given: nested vectors, ranges, an abstract element type,
  # view-backed groups and a list that is a view
  for groups in ([[1], [2]], [1:1, 2:2], AbstractVector{Int}[[1], 2:2], [view([1, 2, 3], 1:1), view([1, 2, 3], 2:2)],
                 view([[1], [2], [3]], 1:2))
    ws = [1.0, 1.0]
    f = GroupMCP{Float64, typeof(groups)}(1.0, 3.0, groups, ws)
    @test f.groups === groups && f.w === ws && f.group_of == [1, 2]
    @test prox(f, [3.0, 4.0], 1.0) == [3.0, 4.0]
  end
  @test_throws ArgumentError GroupMCP{Float64, Vector{Vector{Int}}}(1.0, 3.0, [[1], [1]], [1.0, 1.0])
  # the default weight √|G| without overflowing the group size in a narrow type
  @test GroupMCP(Float16(1), [1:65536]).w == Float16[256]
  @test GroupMCP(1f0, [1:2, 3:5]).w == Float32[sqrt(2f0), sqrt(3f0)]
end

@testset "tiny shapes ($T)" for T in (Float32, Float64)
  # a = 2^-(E+1) is subnormal and inv(a) overflows; λ = 2^E. At γ = a/2 the input 3/8 is firm (γλ = 1/4 < 3/8 < aλ = 1/2)
  # with output 1/4, whose value is r (λ - r/(2a)) = (3/16) 2^E
  E = exponent(floatmax(T))
  a = ldexp(one(T), -E - 1)
  λ = ldexp(one(T), E)
  f = GroupMCP(λ, [1:1], T[1]; a)
  u = T[3 / 8]
  y = similar(u)
  v = value_and_prox!(y, f, u, a / 2)
  @test y == T[1 / 4] && prox_block(f, u, 1, a / 2) == y
  @test f(y) == ldexp(T(3 / 16), E) && v == f(y)
  # with a second group whose plateau overflows, the value is Inf, not NaN
  g = GroupMCP(λ, [1:1, 2:2], T[1, 4]; a)
  @test g(T[1 / 4, 3]) === T(Inf)
  z = similar(u, 2)
  @test value_and_prox!(z, g, T[3 / 8, 3], a / 2) === T(Inf) && z == T[1 / 4, 3]
end

@testset "zero-group screen at nonfinite values" begin
  # an infinite weight: Inf at an infinite gradient entry, NaN at a NaN one, -Inf at finite entries
  g = GroupMCP(1.0, [1:2], [Inf])
  @test kkt_violation(g, [Inf, 1.0], zeros(2), 1) === Inf
  @test isnan(kkt_violation(g, [Inf, NaN], zeros(2), 1))
  @test kkt_violation(g, [3.0, 4.0], zeros(2), 1) === -Inf
end

@testset "scaling by powers of two ($T)" for T in (Float32, Float64)
  # prox(GroupMCP(2^j λ0), 2^j u, γ) = 2^j prox(GroupMCP(λ0), u, γ) across the ordinary and scaled paths; the sums
  # of squares of these groups are exact
  E = exponent(floatmax(T))
  for (u, γ, a) in ((T[3, 4], T(1), T(3)), (T[1.5, 2], T(1), T(3)), (T[1, 2, 2], T(3), T(3)), (T[1, 2, 2], T(4), T(2)),
                    (T[0.75, 1], T(1), T(2)), (T[3, 4], T(10), T(1)), (T[3, 4], T(30), T(1)), (T[6, 8], T(12), T(3)))
    f0 = GroupMCP(one(T), [1:length(u)], T[1]; a)
    y0 = prox_block(f0, u, 1, γ)
    for j in (-E + 8, -E ÷ 2 - 4, -E ÷ 2 + 4, -20, 20, E ÷ 2 - 4, E ÷ 2 + 4, E - 8)
      s = T(2)^j
      y = prox_block(GroupMCP(s, [1:length(u)], T[1]; a), s .* u, 1, γ)
      if T === Float64 || all(v -> iszero(v) || v in u, y0)
        @test y == s .* y0
      else
        # Float32 firm points: the scaled path forms the factor in Float64
        @test all(i -> abs(y[i] - s * y0[i]) <= 2eps(abs(s * y0[i])), eachindex(y))
      end
    end
  end
end

@testset "endpoint rows ($T)" for T in (Float32, Float64)
  G = [1:2]
  pb(f, u, γ) = prox_block(f, T.(u), 1, T(γ))
  z = zero(T)
  steps = (z, one(T), T(Inf))
  # w = 0: unchanged at the levels and steps below, value 0
  for λ in (z, one(T), T(Inf)), γ in steps, a in (T(3), T(Inf))
    f = GroupMCP(λ, G, T[0]; a)
    @test isequal(pb(f, [NaN, -0.0], γ), T[NaN, -0.0]) && isequal(pb(f, [Inf, -Inf], γ), T[Inf, -Inf])
    @test f(T[NaN, Inf]) === z
  end
  # stored infinities: +0 at the steps 0, 1 and Inf; value 0 at a zero group, NaN at a NaN entry, Inf otherwise
  for (λ, w) in ((one(T), T(Inf)), (T(Inf), one(T)), (z, T(Inf)), (T(Inf), T(Inf))), γ in steps, a in (T(3), T(Inf))
    f = GroupMCP(λ, G, T[w]; a)
    for u in ([NaN, 1], [Inf, -1], [-0.0, 0], [1, 2])
      @test all(isequal(z), pb(f, u, γ))
    end
    @test f(T[0, -0.0]) === z && isnan(f(T[NaN, Inf])) && f(T[1, 0]) === T(Inf) && f(T[Inf, 0]) === T(Inf)
  end
  # λ0 = 0 with finite w: identity, value 0 at nonfinite entries
  for γ in steps, w in (one(T), T(2)), a in (T(3), T(Inf))
    f = GroupMCP(z, G, T[w]; a)
    @test isequal(pb(f, [NaN, -0.0], γ), T[NaN, -0.0]) && pb(f, [Inf, 1], γ) == T[Inf, 1]
    @test f(T[NaN, Inf]) === z
  end
  for a in (T(3), T(Inf))
    f = GroupMCP(one(T), G, T[1]; a)
    # positive finite parameters, γ = 0: identity; γ = Inf: +0
    @test isequal(pb(f, [NaN, -0.0], z), T[NaN, -0.0]) && pb(f, [Inf, 3], z) == T[Inf, 3]
    for u in ([NaN, 1], [Inf, 1], [3, 4], [-0.0, 0])
      @test all(isequal(z), pb(f, u, Inf))
    end
    # finite steps: a NaN entry makes the group NaN, including where a threshold would zero or keep the group
    for γ in (T(0.5), T(3), T(12)), u in ([NaN, 1], [NaN, Inf], [Inf, NaN], [NaN, T(1e-30)], [NaN, T(1e30)])
      @test all(isnan, pb(f, u, γ))
    end
    @test isnan(f(T[NaN, Inf])) && isnan(f(T[1, NaN]))
    # an infinite entry and no NaN: unchanged, value the plateau aλ²/2 (Inf for a = Inf)
    for γ in (T(0.5), T(3), T(12))
      @test pb(f, [Inf, 1], γ) == T[Inf, 1] && pb(f, [-Inf, Inf], γ) == T[-Inf, Inf]
    end
    @test f(T[-Inf, 1]) === (isinf(a) ? T(Inf) : T(1.5))
    # a zero group: +0, value 0
    for γ in (T(0.5), T(3), T(12))
      @test all(isequal(z), pb(f, [-0.0, -0.0], γ))
    end
    @test f(T[-0.0, 0]) === z
  end
  # other groups are not affected by a NaN or infinite group
  f = GroupMCP(one(T), [1:2, 3:4, 5:6], T[1, 1, 1])
  x = T[NaN, 1, 1.2, 1.6, Inf, 2]
  y = prox(f, x, one(T))
  @test all(isnan, y[1:2]) && y[5:6] == T[Inf, 2]
  @test isapprox(y[3:4], prox_block(f, T[1.2, 1.6], 2, one(T)); rtol = 4eps(T))
  @test isnan(f(x)) && f(T[0, 0, 1.2, 1.6, Inf, 2]) ≈ GroupMCP(one(T), [1:2], T[1])(T[1.2, 1.6]) + T(1.5)
  # a = Inf: the group map and the value of GroupNormL2 at the level λ0 wₖ
  gw = T[0.5, 2, 1]
  fi = GroupMCP(T(0.8), [1:2, 3:4, 5:6], gw; a = T(Inf))
  gl = GroupNormL2(T(0.8), [1:2, 3:4, 5:6], gw)
  rng = Xoshiro(3)
  for trial in 1:10, γ in (z, T(0.3), one(T), T(Inf))
    x = randn(rng, T, 6)
    trial == 2 && (x[1] = T(NaN))
    @test isequal(prox(fi, x, γ), prox(gl, x, γ))
    v1, v2 = value_and_prox!(similar(x), fi, x, γ), value_and_prox!(similar(x), gl, x, γ)
    @test isequal(v1, v2) || isapprox(v1, v2; rtol = 4eps(T))
    @test isequal(fi(x), gl(x)) || isapprox(fi(x), gl(x); rtol = 4eps(T))
  end
end

@testset "representations ($T)" for T in (Float32, Float64)
  groups = [[3, 1], [4, 2], 5:6]
  f = GroupMCP(T(0.6), groups, T[1, 1, 0.5])
  x = T[1.5, -0.2, 0.9, 0.1, 4, -3]
  γ = T(1)
  y = prox(f, x, γ)
  # SparseIterate argument and output
  s = SparseIterate(T, 6)
  copyto!(s, x)
  # the sums of squares over an iterate may associate differently (SIMD)
  near(a, b) = isapprox(a, b; rtol = 0, atol = 8eps(T) * maximum(abs, x))
  ys = prox(f, s, γ)
  @test ys isa SparseIterate && near(Array(ys), y)
  out = SparseIterate(T, 6)
  @test prox!(out, f, x, γ) === out && Array(out) == y
  @test value_and_prox!(SparseIterate(T, 6), f, s, γ) ≈ value_and_prox!(similar(x), f, x, γ)
  @test f(s) ≈ f(x)
  # BlockIterate with the groups of f (a partition) and in place
  b = BlockIterate(x, f)
  yb = prox(f, b, γ)
  @test yb isa BlockIterate && near(Vector(yb), y)
  bc = copy(b)
  @test prox!(bc, f, bc, γ) === bc && near(Vector(bc), y)
  @test f(b) ≈ f(x) && value_and_prox!(similar(b), f, b, γ) ≈ value_and_prox!(similar(x), f, x, γ)
  # a matrix argument: groups of linear indices
  X = reshape(copy(x), 2, 3)
  Y = prox(f, X, γ)
  @test Y isa Matrix{T} && vec(Y) == y && f(X) == f(x)
  # compact blocks: out of place, in place, and the iterate is not changed
  for k in eachindex(groups)
    u = x[groups[k]]
    out = prox_block(f, u, k, γ)
    @test isapprox(out, y[groups[k]]; atol = 8eps(T) * maximum(abs, x))
    uu = copy(u)
    @test prox_block!(uu, f, uu, k, γ) === uu && uu == out
    bb = copy(b)
    before = (Vector(bb), collect(activecoordinates(bb)))
    prox_block!(zeros(T, length(u)), f, blockview(bb, k), k, γ)
    @test (Vector(bb), collect(activecoordinates(bb))) == before
  end
  @test prox_block(f, [1, 0], 1, 1) isa Vector{Float64}       # integer input, floating output
  # integer and mixed-precision inputs take the scaled path
  @test prox(GroupMCP(1.0, [1:2]), [3, 4], 1) == prox(GroupMCP(1.0, [1:2]), [3.0, 4.0], 1.0)
  @test check_group(prox(GroupMCP(1.0, [1:2]), Float32[1.2, 1.6], 0.1)[1:2], Float32[1.2, 1.6], 1.0, sqrt(2.0), 3.0, 0.1)
end

@testset "preflight refusals leave outputs unchanged ($T)" for T in (Float32, Float64)
  f = GroupMCP(one(T), [1:2, 3:4], T[1, 1])
  x = T[1, 2, 3, 4]
  sentinel() = fill(T(7), 4)
  for (op, args) in ((prox!, ()), (value_and_prox!, ()))
    y = sentinel()
    y5 = fill(T(7), 5)
    @test_throws DimensionMismatch op(y5, f, x, one(T))
    @test y5 == fill(T(7), 5)
    y3 = fill(T(7), 3)
    @test_throws DimensionMismatch op(y3, f, T[1, 2, 3], one(T))     # the groups reach index 4
    @test y3 == fill(T(7), 3)
    for γ in (-one(T), T(NaN), -T(Inf))
      @test_throws ArgumentError op(y, f, x, γ)
    end
    yi = fill(7, 4)
    @test_throws ArgumentError op(yi, f, x, one(T))                  # integer destination
    @test yi == fill(7, 4)
    yr = fill(7 // 1, 4)
    @test_throws ArgumentError op(yr, f, x, one(T))
    @test yr == fill(7 // 1, 4)
    @test_throws ArgumentError op(y, f, Complex{T}.(x), one(T))
    yo = OffsetVector(sentinel(), 0:3)
    @test_throws ArgumentError op(yo, f, x, one(T))
    @test parent(yo) == sentinel()
    @test y == sentinel()
    buf = T[1, 2, 3, 4, 5]
    @test_throws ArgumentError op(view(buf, 2:5), f, view(buf, 1:4), one(T))  # partial overlap
    @test buf == T[1, 2, 3, 4, 5]
    yi = zeros(Int, 4)
    @test_throws ArgumentError op(yi, f, [1, 0, 0, 1], one(T))
    @test yi == zeros(Int, 4)
  end
  out = fill(T(7), 2)
  for k in (0, 3, -1, true)
    @test_throws BoundsError prox_block!(out, f, T[1, 2], k, one(T))
  end
  @test_throws DimensionMismatch prox_block!(out, f, T[1, 2, 3], 1, one(T))    # input length
  o3 = fill(T(7), 3)
  @test_throws DimensionMismatch prox_block!(o3, f, T[1, 2], 1, one(T))       # output length
  @test o3 == fill(T(7), 3)
  oc = Complex{T}[7, 7]
  @test_throws ArgumentError prox_block!(oc, f, T[1, 2], 1, one(T))
  @test oc == Complex{T}[7, 7]
  for γ in (-one(T), T(NaN))
    @test_throws ArgumentError prox_block!(out, f, T[1, 2], 1, γ)
    @test_throws ArgumentError prox_block(f, T[1, 2], 1, γ)
  end
  oi = [7, 7]
  @test_throws ArgumentError prox_block!(oi, f, T[1.2, 1.6], 1, one(T))     # integer destination
  @test oi == [7, 7]
  buf = T[7, 1, 2]
  @test_throws ArgumentError prox_block!(view(buf, 1:2), f, view(buf, 2:3), 1, one(T))
  @test buf == T[7, 1, 2]
  @test out == T[7, 7]
end

@testset "zero allocations ($T)" for T in (Float32, Float64)
  rng = Xoshiro(5)
  groups = [1:4, 5:36, [40, 37, 39, 38], 41:552]
  f = GroupMCP(T(0.2), groups, ones(T, 4))
  x = randn(rng, T, 560)
  x[5:36] .*= T(0.01)                     # a zero group at moderate steps
  x[37:40] .= T(0.15)                     # r = 0.3: firm at γ = 0.5
  y = similar(x)
  for γ in (T(0.5), T(3), T(10))          # firm, hard at γ = a, hard above a
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
    xi = copy(x)
    @test allocs(prox!, xi, f, xi, γ) == 0
    for k in eachindex(groups)
      u = x[groups[k]]
      @test allocs(prox_block!, similar(u), f, u, k, γ) == 0
    end
  end
  @test allocs(f, x) == 0
  fi = GroupMCP(T(0.2), groups, ones(T, 4); a = T(Inf))
  @test allocs(prox!, y, fi, x, one(T)) == 0 && allocs(value_and_prox!, y, fi, x, one(T)) == 0 && allocs(fi, x) == 0
  @test allocs(kkt_violation, f, x, zeros(T, 560), 1) == 0
end

@testset "historical reproducers" begin
  # infinite weights (#250): a hard zero at the steps 0, 1 and Inf; other groups are independent
  f = GroupMCP(1.0, [1:2, 3:4], [Inf, 1.0])
  for γ in (0.0, 1.0, Inf)
    y = prox(f, [3.0, 4.0, 3.0, 4.0], γ)
    @test all(isequal(0.0), y[1:2])
  end
  @test prox(f, [3.0, 4.0, 3.0, 4.0], 0.0)[3:4] == [3.0, 4.0]
  @test f([0.0, -0.0, 3.0, 4.0]) == 1.5 && f([1.0, 0.0, 3.0, 4.0]) === Inf
  # norms beyond the input type (#534): the group norm is formed in the recovery type, not in the input type
  x32 = Float32[2f38, 3f38]                                                          # r ≈ 3.6e38
  @test prox!(zeros(Float64, 2), GroupMCP(1.0, [1:2]), x32, 1.0) == Float64.(x32)      # kept
  y = prox!(zeros(Float64, 2), GroupMCP(1.5e38, [1:2], [1.0]), x32, 1.0)              # firm: t* ≈ 3.2e38
  @test all(isfinite, y) && check_group(y, x32, 1.5e38, 1.0, 3.0, 1.0)
  u = Float32[3f38, 3f38]
  y32 = prox(GroupMCP(2f38, [1:2], [1f0]), u, 1f0)
  @test all(isfinite, y32) && check_group(y32, u, 2f38, 1f0, 3f0, 1f0)
  z64 = [1e300, 2e300, 2e300]
  @test check_group(prox(GroupMCP(1e300, [1:3], [1.0]), z64, 1.0), z64, 1e300, 1.0, 3.0, 1.0)
  @test close_value(GroupMCP(1e300, [1:3], [1.0])(z64), refterm(z64, 1e300, 1.0, 3.0), 3, 1)
end

@testset "range and extreme scales ($T)" for T in (Float32, Float64)
  rng = Xoshiro(2069 + sizeof(T))
  E = exponent(floatmax(T))
  δ = nextfloat(zero(T))
  nok = ncase = 0
  for trial in 1:400
    n = rand(rng, (1, 2, 3, 8, 33))
    eλ, ew, eg, er = rand(rng, -E-20:E-1), rand(rng, -40:40), rand(rng, -E-20:E-1), rand(rng, -E-20:E-1)
    λ0 = max(δ, ldexp(T(0.5) + rand(rng, T) / 2, eλ))
    w = ldexp(T(0.5) + rand(rng, T) / 2, ew)
    γ = max(δ, ldexp(T(0.5) + rand(rng, T) / 2, eg))
    a = rand(rng, (T(3), T(1), T(0.01), T(1000), ldexp(T(0.75), rand(rng, -E+2:E-2)), T(Inf)))
    λ = biglevel(λ0, w)
    # norms near the thresholds of the step, or at a random scale
    cut = γ < a ? big(γ) * λ * rand(rng, (big"0.5", big(1), big"1.0000001", big(2))) :
                  λ * sqrt(big(a) * big(γ)) * rand(rng, (big"0.5", big(1), big"1.0000001", big(2)))
    r = rand(rng, Bool) ? cut : ldexp(big(1.0), er)
    big(floatmax(T)) / 2 > r > 64 * big(δ) * n || continue
    u = vector_with_norm(rng, T, n, r)
    all(isfinite, u) && any(!iszero, u) || continue
    f = GroupMCP(λ0, [1:n], [w]; a)
    y = prox_block(f, u, 1, γ)
    ok = check_group(y, u, λ0, w, a, γ)
    v = f(u)
    okv = close_value(v, refterm(u, λ0, w, a), n, 1)
    z = similar(u)
    okf = fused_close(value_and_prox!(z, f, u, γ), f(z), n, 1) && check_group(z, u, λ0, w, a, γ)
    ncase += 1
    nok += ok & okv & okf
    (ok & okv & okf) || @info "extreme-scale failure" T n λ0 w a γ ok okv okf
  end
  @test ncase >= 200
  @test nok == ncase
end

@testset "fused value at the stored output ($T)" for T in (Float32, Float64)
  rng = Xoshiro(11)
  groups = [[2, 7, 4], [1], [5, 3, 6], 8:12]
  for (a, γ) in ((T(3), T(0.5)), (T(3), T(3)), (T(1.5), T(4)), (T(Inf), T(0.5)))
    f = GroupMCP(T(0.4), groups, T[1, 2, 0, 1]; a)
    for trial in 1:20
      x = randn(rng, T, 14) .* T(2)^rand(rng, -3:3)
      y = similar(x)
      v = value_and_prox!(y, f, x, γ)
      @test fused_close(v, f(y), 5, 4)
      # a Float64 destination: bitwise equality for the Float32 data of these cases, the tolerance above for Float64 data
      y64 = zeros(Float64, 14)
      v64 = value_and_prox!(y64, f, x, γ)
      @test T === Float32 ? v64 === f(y64) : fused_close(v64, f(y64), 5, 4)
    end
  end
end

@testset "Scaled ($T)" for T in (Float32, Float64)
  f = GroupMCP(T(0.5), [1:2, 3:4], T[1, 2])
  x = T[1.2, 1.6, -3, 0.5]
  for α in (T(0.5), T(2), T(3))
    s = Scaled(f, α)
    @test s(x) ≈ α * f(x)
    @test prox(s, x, T(0.7)) == prox(f, x, α * T(0.7))
    y = similar(x)
    @test value_and_prox!(y, s, x, T(0.7)) ≈ α * f(y)
  end
  @test_throws ArgumentError Scaled(f, zero(T))
  # an effective step that leaves the normal range is rounded: here αγ rounds to Inf and the group is +0, whereas the
  # exact step 2^1200 keeps it (r² = 1 > aγλ² = 3 2^-200)
  if T === Float64
    g = GroupMCP(2.0^-700, [1:2], [1.0])
    u = [0.6, 0.8]
    @test prox(Scaled(g, 2.0^600), u, 2.0^600) == prox(g, u, Inf) == [0.0, 0.0]
    @test prox_block(GroupMCP(2.0^-700, [1:2], [1.0]), u, 1, 2.0^1023) == u
    # an underflowing effective step rounds to zero: the identity at this positive finite level and finite weight
    @test prox(Scaled(GroupMCP(1.0, [1:2]), 2.0^-600), u, 2.0^-600) == u
    # a stored infinite level still gives +0 at that zero step
    @test isequal(prox(Scaled(GroupMCP(Inf, [1:1], [1.0]), 2.0^-600), [0.5], 2.0^-600), [0.0])
    # a nonzero subnormal effective step (2^-1025) is used as it is: γλ = 1/4, firm
    @test prox(Scaled(GroupMCP(2.0^1023, [1:1], [1.0]), 2.0^-512), [0.5], 2.0^-513) == [0.25]
    # a negative step whose product with the factor rounds to -0 reaches the function as a zero step
    @test prox(Scaled(GroupMCP(1.0, [1:1], [1.0]), nextfloat(0.0)), [2.0], -0.25) == [2.0]
  end
end

@testset "refused methods ($T)" for T in (Float32, Float64)
  f = GroupMCP(one(T), [1:2, 3:4])
  x = T[1, 2, 0, 0]
  S = SymmetricSparseIterate(T[1 2; 2 1])
  S2 = SymmetricSparseIterate(T[5 6; 6 7])
  @test_throws MethodError restrict(f, [1, 2])
  @test_throws MethodError prox(f, x, ones(T, 4))
  @test_throws MethodError prox!(similar(x), f, x, ones(T, 4))
  @test_throws MethodError value_and_prox!(similar(x), f, x, ones(T, 4))
  @test_throws MethodError prox_block(f, one(T), 1, one(T))
  @test_throws ArgumentError prox!(S2, f, S, one(T))
  @test_throws ArgumentError value_and_prox!(S2, f, S, one(T))
  D = fill(T(7), 2, 2)
  @test_throws ArgumentError prox!(D, f, S, one(T))
  @test_throws ArgumentError value_and_prox!(D, f, S, one(T))
  @test_throws ArgumentError prox(f, S, one(T))
  @test D == fill(T(7), 2, 2) && Matrix(S2) == T[5 6; 6 7]
  g = T[1, 1, 1, 1]
  @test_throws ArgumentError kkt_violation(f, g, 1)
  @test_throws ArgumentError kkt_violation(f, g, x, 1)              # a nonzero group
  @test_throws ArgumentError kkt_violation(f, g, x)                 # the full-point form
  @test_throws ArgumentError kkt_violation(f, g, zeros(T, 4))
  @test_throws ArgumentError kkt_violation(f, g, BlockIterate(x, f), 1)
  @test_throws MethodError kkt_violation(f, ones(T, 4), S, 1)
  @test_throws MethodError kkt_violation(f, ones(T, 4), S)
  s = Scaled(f, T(2))
  @test_throws MethodError kkt_violation(s, g, zeros(T, 4), 3)
  @test_throws MethodError kkt_violation(s, g, zeros(T, 4))
  @test_throws ArgumentError kkt_violation(s, g, 1)
  @test_throws MethodError prox_block!(zeros(T, 2), s, T[1, 2], 1, one(T))
end

@testset "zero-group KKT screen ($T)" for T in (Float32, Float64)
  groups = [[3, 1], 4:5, [2]]
  w = T[1, 2, 0.5]
  f = GroupMCP(T(0.75), groups, w)
  rng = Xoshiro(8)
  refscreen(v, G, k) = setprecision(() -> bignorm(v[G]) - big(T(0.75)) * big(w[k]), BigFloat, PREC)
  for trial in 1:20
    v = randn(rng, T, 7)
    x = zeros(T, 7)
    x[6] = T(trial)                                      # an entry in no group
    for k in eachindex(groups), i in groups[k]
      s = kkt_violation(f, v, x, i)
      @test s isa T && abs(big(s) - refscreen(v, groups[k], k)) <= 8eps(T) * (bignorm(v[groups[k]]) + big(w[k]))
    end
    @test kkt_violation(f, v, x, 6) === abs(v[6]) && kkt_violation(f, v, x, 7) === abs(v[7])
    b = BlockIterate(zeros(T, 5), groups)
    for k in eachindex(groups)
      @test kkt_violation(f, v[1:5], b, k) === kkt_violation(f, v, x, groups[k][1])
    end
  end
  # -0 counts as zero; NaN in the gradient gives NaN; an infinite weight gives -Inf; a zero weight gives ‖v_G‖₂
  @test kkt_violation(f, T[3, 0, 4, 0, 0, 0, 0], T[-0.0, 0, 0, 0, 0, 0, 0], 1) == T(5) - T(0.75)
  @test isnan(kkt_violation(f, T[NaN, 0, 4, 0, 0, 0, 0], zeros(T, 7), 3))
  @test kkt_violation(GroupMCP(one(T), [1:2], T[Inf]), T[3, 4], zeros(T, 2), 1) === -T(Inf)
  @test kkt_violation(GroupMCP(one(T), [1:2], T[0]), T[3, 4], zeros(T, 2), 1) === T(5)
  # an overflowed product λ0 w is recovered
  if T === Float64
    v = fill(floatmax(T), 4)
    g = GroupMCP(1.5 * 2.0^512, [1:4], [2.0^512])
    s = kkt_violation(g, v, zeros(4), 1)
    ref = setprecision(() -> bignorm(v) - big(1.5 * 2.0^512) * big(2.0^512), BigFloat, PREC)
    @test isfinite(s) && abs(big(s) - ref) <= 8eps(T) * abs(ref)
  end
  # layout and argument checks
  b = BlockIterate(zeros(T, 5), [1:2, 3:5])
  @test_throws DimensionMismatch kkt_violation(f, zeros(T, 5), b, 1)
  b3 = BlockIterate(zeros(T, 5), [[1, 3], 4:5, [2]])
  @test_throws DimensionMismatch kkt_violation(f, zeros(T, 5), b3, 1)
  @test_throws BoundsError kkt_violation(f, zeros(T, 5), BlockIterate(zeros(T, 5), groups), 4)
  @test_throws DimensionMismatch kkt_violation(f, zeros(T, 3), zeros(T, 3), 1)    # the groups reach index 5
  @test_throws DimensionMismatch kkt_violation(f, zeros(T, 6), zeros(T, 7), 1)
  @test_throws BoundsError kkt_violation(f, zeros(T, 7), zeros(T, 7), 8)
  @test_throws ArgumentError kkt_violation(f, Complex{T}.(zeros(7)), zeros(T, 7), 1)
end

end # module
