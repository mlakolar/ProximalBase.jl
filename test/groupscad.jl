module GroupSCADTest
# GroupSCAD: values, array and compact block proximal operators, the zero-group KKT screen and the refused methods,
# against BigFloat evaluations, independent of the package, of
#   f(x) = Σₖ p(‖x[Gₖ]‖₂; λ0 wₖ, a),   prox_{γf}(u)[Gₖ] = (t*/r) u[Gₖ],  r = ‖u[Gₖ]‖₂,
# where t* minimizes p(t) + (t - r)²/(2γ) over t ≥ 0 (the scalar reference of scad_reference.jl, which does not use the
# package: a candidate comparison at 512 bits, repeated at 4096 bits for near-ties), with r computed in 512-bit BigFloat,
# in Float32 and Float64.
using Test, ProximalBase, Random, OffsetArrays
using ..TestUtils

include("scad_reference.jl")
using .SCADReferenceTest

const PREC = 512

# ---- references (BigFloat) ----
bignorm(u) = setprecision(() -> sqrt(sum(big.(u) .^ 2; init = big(0))), BigFloat, PREC)
biglevel(λ0, w) = setprecision(() -> big(λ0) * big(w), BigFloat, PREC)
# p(t; λ, a) for t ≥ 0, λ > 0 (a = Inf: λt)
bigp(t, λ, a) = setprecision(() -> scad_penalty(big(λ), big(a), big(t)), BigFloat, PREC)
# the minimizer over t ≥ 0 of p(t; λ, a) + (t - r)²/(2γ) at r > 0 (a = Inf: soft thresholding, +0 at a tie)
function reftarget(λ, a, r, γ)
  setprecision(BigFloat, PREC) do
    isinf(a) ? max(big(r) - big(γ) * big(λ), big(0)) : first(scad_reference(big(λ), big(a), big(r), big(γ)))
  end
end

# The proximal point of one group with finite entries at positive finite λ0, w and γ: the scalar reference at the
# norm computed in 512-bit BigFloat, along u.
function refgroup(u, λ0, w, a, γ)
  setprecision(BigFloat, PREC) do
    U = big.(u)
    r = bignorm(u)
    iszero(r) && return zero.(U)
    (reftarget(biglevel(λ0, w), a, r, γ) / r) .* U
  end
end
# The value of one group with finite entries, positive finite λ0 and w.
refterm(u, λ0, w, a) = bigp(bignorm(u), biglevel(λ0, w), a)
# The group objective p(‖y‖) + ‖y - u‖²/(2γ), finite y and u, positive finite λ0, w and γ.
function refobjective(y, u, λ0, w, a, γ)
  setprecision(BigFloat, PREC) do
    Y, U = big.(y), big.(u)
    bigp(bignorm(y), biglevel(λ0, w), a) + sum((Y .- U) .^ 2; init = big(0)) / (2 * big(γ))
  end
end
# :zero, :keep or :firm, the reference decision at the norm r
function decision(r, λ0, w, a, γ)
  t = reftarget(biglevel(λ0, w), a, r, γ)
  iszero(t) ? :zero : t == r ? :keep : :firm
end

# The checks of one group y = prox at u (finite entries; positive finite λ0, w, γ):
#  * the objective at y is within tol of the reference minimum, with
#      tol = (n + 8) eps S (4 + (n + 8) eps κ) + n δ (2r/γ + λ),
#    S = r²/(2γ) + p(r), κ = max(1, (a - 1)/(a - 1 - γ)) (1 where γ ≥ a - 1 or a = Inf), n = |G|, δ = nextfloat(zero(T)):
#    the effect of an error of (n + 2) eps in the computed norm on the decisions and on the output point (whose slope is κ),
#    and the absolute rounding of subnormal outputs;
#  * when the reference decision is the same at r (1 ± (n + 8) eps), y is u (kept), +0 (zero), or within
#    ((n + 8) κ + 8) eps max|u| + 2δ of the reference in each entry (firm).
function check_group(y::AbstractVector{T}, u, λ0, w, a, γ) where {T}
  n = length(u)
  r = bignorm(u)
  λ = biglevel(λ0, w)
  ε, δ = big(eps(T)), big(nextfloat(zero(T)))
  κ = isfinite(a) && γ < a - 1 ? (big(a) - 1) / (big(a) - 1 - big(γ)) : big(1)
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
# reference; beyond floatmax(T) (1 + ρ) the infinity of the sign of the reference; above floatmax(T) (1 - ρ) and up to
# floatmax(T) (1 + ρ) that infinity or a finite value within ρ; below, a finite value within ρ of the reference and 4K
# subnormal spacings.
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

# candidate norms for the targets of the tests, for λ > 0: γλ, λ, (1 + γ)λ, aλ and the two cutoffs of the ties of the nonconvex
# case, (a + 1 + γ)λ/2 and λ√(γ(a + 1)); a = Inf has the first two. Which of them changes the decision depends on γ and a (the tie
# cutoffs apply for γ ≥ a - 1), and the tests use all of them as targets
function knots(λ, a, γ)
  setprecision(BigFloat, PREC) do
    isinf(a) && return [big(γ) * λ, λ]
    [big(γ) * λ, λ, (1 + big(γ)) * λ, big(a) * λ, (big(a) + 1 + big(γ)) * λ / 2, λ * sqrt(big(γ) * (big(a) + 1))]
  end
end

@testset "construction and validation" begin
  for T in (Float32, Float64)
    f = GroupSCAD(T(0.5), [1:2, [3, 5, 7]])
    @test f isa GroupSCAD{T}
    @test f.a === T(3.7)
    @test f.w == T[sqrt(T(2)), sqrt(T(3))]             # default √|Gₖ|: the root in at least Float64, converted to T (the same values here as the root in T)
    @test GroupSCAD(T(0.5), [1:2], T[3]).w == T[3]
    @test GroupSCAD(T(0.5), [1:2]; a = 2.5).a === T(2.5)
    @test GroupSCAD(T(0.5), [1:2]; a = T(Inf)).a === T(Inf)
    @test GroupSCAD(T(0.5), [1:2]; a = Inf).a === T(Inf)
    @test GroupSCAD(T(0.5), [1:2]; a = nextfloat(T(2))).a === nextfloat(T(2))
    w = T[1, 2]; G = [[2, 1], [4, 3]]
    g = GroupSCAD(one(T), G, w)
    w[1] = 9; G[1][1] = 7
    @test g.w == T[1, 2] && g.groups == [[2, 1], [4, 3]]
    @test g.group_of == [1, 1, 2, 2] && g.free == Int[]
    h = GroupSCAD(one(T), [[5, 2], 7:8])
    @test h.free == [1, 3, 4, 6]
  end
  # the parameter type: that of the level and the weights, and a is converted to it
  @test GroupSCAD(1f0, [1:2]) isa GroupSCAD{Float32}
  @test GroupSCAD(1f0, [1:2]).a === 3.7f0
  @test GroupSCAD(1f0, [1:2], Float32[1]; a = 3.0).a === 3f0
  @test GroupSCAD(1f0, [1:2], [1.0]) isa GroupSCAD{Float64}      # the weights are Float64
  @test GroupSCAD(1, [1:2]) isa GroupSCAD{Float64}
  @test GroupSCAD(1, [1:2], [2]) isa GroupSCAD{Float64}
  @test GroupSCAD(big(1), [1:2]) isa GroupSCAD{BigFloat}
  @test GroupSCAD(1f0, [1:2]; a = prevfloat(Inf32) |> Float64).a === prevfloat(Inf32)
  # a is checked before and after its conversion: a value above 2 that rounds to 2, and a finite value that overflows
  @test_throws ArgumentError GroupSCAD(1f0, [1:2]; a = 2.00000001)
  @test_throws ArgumentError GroupSCAD(1f0, [1:2]; a = 1e39)
  @test_throws ArgumentError GroupSCAD(1f0, [1:2], Float32[1]; a = 2.00000001)
  @test_throws ArgumentError GroupSCAD(Float16(1), [1:2]; a = 1e5)
  @test GroupSCAD(1.0, [1:2]; a = 2.00000001).a == 2.00000001
  try
    GroupSCAD(1.0, [1:2]; a = 2.0)
    @test false
  catch e
    @test e isa ArgumentError && occursin("a must exceed 2, got 2.0", e.msg)
  end
  for λ in (-1.0, NaN, -Inf, -big"1e-400")
    @test_throws ArgumentError GroupSCAD(λ, [1:2])
  end
  for a in (2, 2.0, 0, -0.0, -1, NaN, -Inf, 1.5)
    @test_throws ArgumentError GroupSCAD(1.0, [1:2]; a)
  end
  for w in ([-1.0], [NaN], [-big"1e-400"])
    @test_throws ArgumentError GroupSCAD(1.0, [1:2], w)
  end
  @test_throws ArgumentError GroupSCAD(1.0, [1:2], [1.0 + 2im])
  @test_throws DimensionMismatch GroupSCAD(1.0, [1:2], [1.0, 2.0])
  @test_throws ArgumentError GroupSCAD(1.0, [Int[]])
  @test_throws ArgumentError GroupSCAD(1.0, [[0, 1]])
  @test_throws ArgumentError GroupSCAD(1.0, [1:2, 2:3])
  @test_throws ArgumentError GroupSCAD{Float64, Vector{UnitRange{Int}}}(1.0, 2.0, [1:2], [1.0])
  @test_throws ArgumentError GroupSCAD{Float64, Vector{UnitRange{Int}}}(1.0, NaN, [1:2], [1.0])
  @test_throws ArgumentError GroupSCAD{Float64, Vector{UnitRange{Int}}}(-1.0, 3.0, [1:2], [1.0])
  f = GroupSCAD(0.5, [1:2, 3:4])
  @test isproximable(f) && !isconvex(f) && !issmooth(f) && !isseparable(f)
  @test isproximable(GroupSCAD) && !isconvex(GroupSCAD)
  # BlockIterate constructors with the groups of f
  b = BlockIterate(4, f)
  @test b isa BlockIterate{Float64} && ncoordinates(b) == 2 && iszero(Vector(b))
  @test BlockIterate(Float32, 4, f) isa BlockIterate{Float32}
  c = BlockIterate([1.0, 0.0, 0.0, 0.0], f)
  @test nactive(c) == 1 && isactive(c, 1)
  @test_throws ArgumentError BlockIterate(5, f)                # the groups must partition 1:p
  # empty group set: zero value, identity proximal map
  e = GroupSCAD(1.0, Vector{UnitRange{Int}}())
  @test e([1.0, NaN]) === 0.0 && e([1.0, -2.0]) === 0.0
  @test isequal(prox(e, [1.0, -0.0, NaN], 2.0), [1.0, -0.0, NaN])
end

# the norm of a soft point y = t u/r is t = r - γλ
norm_close(y, t, r) = isapprox(sqrt(sum(abs2, Float64.(y))), t; rtol = 1e-4)

@testset "worked examples ($T)" for T in (Float32, Float64)
  a = T(3.7)
  f = GroupSCAD(one(T), [1:2], T[1]; a)                        # λ = 1
  pb(f, u, γ) = prox_block(f, T.(u), 1, T(γ))
  # γ = 1 < a - 1 (a - 1 - γ > 0): the pieces of the proximal map
  @test pb(f, [3, 4], 1) == T[3, 4]                           # r = 5 ≥ aλ: kept
  @test all(isequal(zero(T)), pb(f, [0.6, 0.8], 1))           # r = γλ: +0
  @test pb(f, [0.9, 1.2], 1) ≈ T[0.3, 0.4]                    # r = 1.5 ≤ (1 + γ)λ: soft, t* = 0.5
  y = pb(f, [1.8, 2.4], 1)                                    # r = 3, middle: t* = ((a - 1) r - γ a λ)/(a - γ - 1)
  @test y ≈ T[1.552941176470588, 2.070588235294118] rtol = 8eps(T)
  for u in ([3, 4], [0.6, 0.8], [0.9, 1.2], [1.8, 2.4])
    @test check_group(pb(f, u, 1), T.(u), 1, 1, a, 1)
  end
  # γ = 4 ≥ a - 1 (a - 1 - γ ≤ 0): the left minimum (soft) or the right one (the point)
  for (r, expect) in ((3, :zero), (5, :keep), (4.5, :keep), (4.2, :soft))
    u = T[3, 4] .* T(r / 5)
    y = pb(f, u, 4)
    @test check_group(y, u, 1, 1, a, 4)
    expect == :zero && @test all(isequal(zero(T)), y)
    expect == :keep && @test y == u
    expect == :soft && @test norm_close(y, T(r - 4), T(r))
  end
  # the values: p(5) = (a + 1)/2, and the pieces at r = 3, 2, 1.5
  @test f(T[3, 4]) ≈ T(2.35)
  for (r, want) in ((3, 2.259259259259259), (2, 1.814814814814815), (1.5, 1.453703703703704))
    @test f(T[0.6, 0.8] .* T(r)) ≈ T(want) rtol = 64eps(T)
  end
  # a copied group inside the middle piece: the step is so small that t* rounds to r; the value is p(2), not the plateau
  g = GroupSCAD(one(T), [1:2], T[1]; a)
  u = T[1.2, 1.6]
  y = similar(u)
  v = value_and_prox!(y, g, u, T(1e-20))
  @test y == u && v == g(y)
  @test v ≈ T(1.814814814814815) rtol = 64eps(T)
  @test !(v ≈ T(2.35))
  @test prox_block(g, u, 1, T(1e-20)) == u
  # two ties with equal objectives in exact arithmetic at representable inputs give the smaller magnitude: a = γ = 9/2, r = 5 is the tie of the left minimum
  # (soft = 1/2) and the right one (r), and γ = 25/4, a = 3, r = 5 that of zero and r
  for s in (one(T), T(2)^-20, T(2)^20)
    g1 = GroupSCAD(s, [1:2], T[1]; a = T(9 / 2))
    y1 = prox_block(g1, s .* T[3, -4], 1, T(9 / 2))
    @test y1 ≈ s .* T[3, -4] ./ 10 rtol = 4eps(T)
    @test prox_block(g1, (1 + 4eps(T)) .* s .* T[3, -4], 1, T(9 / 2)) == (1 + 4eps(T)) .* s .* T[3, -4]
    g2 = GroupSCAD(s, [1:2], T[1]; a = T(3))
    @test all(isequal(zero(T)), prox_block(g2, s .* T[3, -4], 1, T(25 / 4)))
    @test all(isequal(zero(T)), prox_block(g2, s .* T[3, -4], 1, nextfloat(T(25 / 4))))
    @test prox_block(g2, s .* T[3, -4], 1, prevfloat(T(25 / 4), 4)) == s .* T[3, -4]   # the cutoff √(γ (a + 1)) is rounded
  end
end

@testset "tie objectives are equal ($T)" for T in (Float32, Float64)
  setprecision(BigFloat, 256) do
    # the two cutoffs of the nonconvex case, from the scalar reference: equal objectives at the left minimum and the right one
    L, A, G = big(1), big(T(9 / 2)), big(T(9 / 2))
    r = big(5)
    @test scad_objective(L, A, r, G, r - G * L) == scad_objective(L, A, r, G, r)
    A2, G2 = big(T(3)), big(T(25 / 4))
    @test scad_objective(L, A2, r, G2, big(0)) == scad_objective(L, A2, r, G2, r)
  end
end

@testset "reference grid ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1070 + sizeof(T))
  sizes = (1, 2, 4, 32, 512)
  levels = T[0.5, 1, 3]
  shapes = T[nextfloat(T(2)), 2 + 4eps(T), 2.5, 3.7, 10, 1e3, 1e10, Inf]
  nok = ncase = 0
  for trial in 1:300
    n = sizes[rand(rng, 1:length(sizes))]
    λ0 = rand(rng, levels)
    w = rand(rng, (one(T), sqrt(T(n)), T(0.25)))
    a = rand(rng, shapes)
    base = isfinite(a) ? a - 1 : T(2)
    γ = rand(rng, (base / 10, prevfloat(base), base, nextfloat(base), 4base, one(T)))
    λ = biglevel(λ0, w)
    c = rand(rng, knots(λ, a, γ))
    r = c * rand(rng, (big"0.5", 1 - big(eps(T)), big(1), 1 + big(eps(T)), big"1.5", big(4)))
    u = vector_with_norm(rng, T, n, r)
    y = prox_block(GroupSCAD(λ0, [1:n], [w]; a), u, 1, γ)
    ok = check_group(y, u, λ0, w, a, γ)
    ncase += 1
    nok += ok
    ok || @info "reference grid failure" T n λ0 w a γ
    # the value of the input and of the stored output
    g = GroupSCAD(λ0, [1:n], [w]; a)
    @test close_value(g(u), refterm(u, λ0, w, a), n, 1)
    z = similar(u)
    v = value_and_prox!(z, g, u, γ)
    @test (z == y || check_group(z, u, λ0, w, a, γ)) && fused_close(v, g(z), n, 1)
  end
  @test nok == ncase
  # a deterministic group of 4096 entries (default weight 64) at each candidate knot of three steps
  let n = 4096, λ0 = one(T), w = T(64), a = T(3.7), rng = Xoshiro(4096 + sizeof(T))
    for γ in (T(5), T(100), T(200)), c in knots(biglevel(λ0, w), a, γ)
      u = vector_with_norm(rng, T, n, c)
      f = GroupSCAD(λ0, [1:n], [w]; a)
      y = prox_block(f, u, 1, γ)
      @test check_group(y, u, λ0, w, a, γ)
      @test close_value(f(u), refterm(u, λ0, w, a), n, 1)
    end
  end
  # several groups in one array: permuted, noncontiguous, with entries in no group before, between and after
  groups = [[5, 2], 7:9, [1], [12, 10, 11]]
  w = T[1, 0.5, 2, 1]
  free = [3, 4, 6, 13, 14]
  for (a, γ) in ((T(3.7), T(1)), (T(3.7), T(3)), (T(2.5), T(5)), (T(Inf), T(1)))
    f = GroupSCAD(T(0.7), groups, w; a)
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

@testset "knots, plateau and shapes near 2 ($T)" for T in (Float32, Float64)
  rng = Xoshiro(5 + sizeof(T))
  # target norms 1 ± 4eps(T) times each candidate knot (the vectors are rounded after scaling), for a few shapes and group sizes
  for a in (nextfloat(T(2)), T(2) + 64eps(T), T(3), T(3.7), T(10)), λ0 in (one(T), T(0.6))
    crit = a - 1
    for γ in (T(0.3), one(T), prevfloat(crit), crit, nextfloat(crit), crit + 1, T(30)), n in (1, 2, 5)
      w = one(T)
      λ = biglevel(λ0, w)
      for c in knots(λ, a, γ), f in (1 - 4big(eps(T)), big(1), 1 + 4big(eps(T)))
        u = vector_with_norm(rng, T, n, c * f)
        y = prox_block(GroupSCAD(λ0, [1:n], [w]; a), u, 1, γ)
        @test check_group(y, u, λ0, w, a, γ)
      end
    end
  end
  # the plateau: the value does not change above aλ, and the group at r = 50 is kept at the three steps below a - 1 listed
  f = GroupSCAD(one(T), [1:2], T[1]; a = T(3))
  @test f(T[3, 4]) == f(T[30, 40]) == T(2)
  @test f(T[3e30, 4e30]) == T(2) && f(T[Inf, 0]) == T(2) && f(T[0, -Inf]) == T(2)
  for γ in (T(0.5), T(1.5), T(1.9))
    @test prox_block(f, T[30, 40], 1, γ) == T[30, 40]
  end
  # a near 2: the middle piece is a concave quadratic with slope λ at λ and 0 at aλ, here on a short interval
  g = GroupSCAD(one(T), [1:1], T[1]; a = nextfloat(T(2)))
  @test g(T[1]) == T(1)
  @test g(T[nextfloat(T(2))]) ≈ T(1.5) rtol = 4eps(T)
  @test prox_block(g, T[0.5], 1, T(0.25)) ≈ T[0.25] && prox_block(g, T[5], 1, T(1)) == T[5]
end

@testset "scalar SCAD agreement of a singleton group ($T)" for T in (Float32, Float64)
  rng = Xoshiro(17 + sizeof(T))
  δ = nextfloat(zero(T))
  # unit weight: the group is one entry. The decisions (zero, kept, or between) agree with the scalar SCAD, bit for bit
  # where the output is zero or the input; the points in between within (8κ + 8) ulps and a subnormal spacing, with κ = 1 for
  # γ ≥ a - 1 and (a - 1)/(a - 1 - γ) below (the group rounds t/r then (t/r) u, and the convex middle of the homogeneous
  # scalar uses its own root form)
  for (λ, a) in ((T(0.5), T(2.5)), (T(1), T(3.7)), (T(2), T(10)), (T(1), nextfloat(T(2))), (T(0.3), T(1e5))),
      γ in T[0.1, 1, 2, 2.7, 3, 5, 10, 40], trial in 1:20
    v = T(randn(rng)) * T(4) * λ
    s = prox_block(SCAD(λ, a), v, 1, γ)
    g = prox_block(GroupSCAD(λ, [1:1], T[1]; a), T[v], 1, γ)[1]
    κ = γ >= a - 1 ? one(T) : (a - 1) / (a - 1 - γ)
    @test iszero(s) == iszero(g) && (s == v) == (g == v)
    @test abs(g - s) <= (8κ + 8) * eps(abs(s)) + δ
    (iszero(s) || s == v) && @test isequal(g, s)
  end
  # the scalar values
  for (λ, a) in ((T(0.5), T(2.5)), (T(1), T(3.7)), (T(2), T(10)), (T(0.3), T(1e5))), trial in 1:20
    v = T(randn(rng)) * T(4) * λ
    @test GroupSCAD(λ, [1:1], T[1]; a)(T[v]) ≈ value_entry(SCAD(λ, a), v, 1) rtol = 16eps(T)
  end
  # extremes: the group of one entry follows the scalar decisions at the ends of the range
  E = exponent(floatmax(T))
  for (λ, v, γ, a) in ((T(2)^(E - 23), T(2)^-30, T(2)^(-E - 36), T(3.7)),       # Float64: a subnormal step, soft; Float32: the step rounds to 0, identity
                       (T(2)^(-E + 30), T(2)^30, T(1.5) * T(2)^E, T(1.5) * T(2)^E),
                       (floatmax(T) / 4, floatmax(T) / 2, T(0.5), T(3)), (T(4)δ, T(8)δ, T(1), T(3)), (one(T), floatmax(T), T(3), T(3)))
    s = prox_block(SCAD(λ, a), v, 1, γ)
    g = prox_block(GroupSCAD(λ, [1:1], T[1]; a), T[v], 1, γ)[1]
    @test iszero(s) == iszero(g) && abs(g - s) <= 4eps(abs(s)) + 2δ
  end
end

@testset "cold-path counterexamples" begin
  # λ = 2^1000 is normal; in units of r = 2^-30 the level (2^1030) overflows, and the step is subnormal:
  # γλ = 2^-60 < r < λ, the first piece
  g = GroupSCAD(2.0^1000, [1:1], [1.0])
  y = prox_block(g, [2.0^-30], 1, 2.0^-1060)
  @test y[1] == prox_block(SCAD(2.0^1000, 3.7), 2.0^-30, 1, 2.0^-1060) == 2.0^-30 - 2.0^-60
  @test check_group(y, [2.0^-30], 2.0^1000, 1.0, 3.7, 2.0^-1060)
  # the same decision from the parts of a sum of squares that underflows (the scaled path)
  g2 = GroupSCAD(2.0^490, [1:2], [1.0])
  @test prox_block(g2, [2.0^-540, 0.0], 1, 2.0^-1060) == [2.0^-540 - 2.0^-570, 0.0]
  @test check_group(prox_block(g2, [2.0^-540, 0.0], 1, 2.0^-1060), [2.0^-540, 0.0], 2.0^490, 1.0, 3.7, 2.0^-1060)
  # an underflowed finite product λ0 w is not a stored zero: r/λ ≈ 2.2e90 with a = 3.7 and γ = 1e100 (nonconvex case,
  # r/λ > √(γ (a + 1))): kept; a = 1e200 (convex, r < γλ): zero
  u = [1e-310, 2e-310]
  @test prox_block(GroupSCAD(1e-200, [1:2], [1e-200]), u, 1, 1e100) == u
  @test check_group(prox_block(GroupSCAD(1e-200, [1:2], [1e-200]), u, 1, 1e100), u, 1e-200, 1e-200, 3.7, 1e100)
  @test prox_block(GroupSCAD(1e-200, [1:2], [1e-200]; a = 1e200), u, 1, 1e100) == [0.0, 0.0]
  @test check_group(prox_block(GroupSCAD(1e-200, [1:2], [1e-200]; a = 1e200), u, 1, 1e100), u, 1e-200, 1e-200, 1e200, 1e100)
  @test prox_block(GroupSCAD(1e-200, [1:2], [1e-200]; a = Inf), u, 1, 1e100) == [0.0, 0.0]
  # an overflowed finite product λ ≈ 2e308 is not a stored infinity: the group is below the level, and γλ > r
  f = GroupSCAD(2e154, [1:2], [1e154])
  u2 = [3e300, 4e300]
  @test prox_block(f, u2, 1, 1.0) == [0.0, 0.0]
  @test prox_block(f, [3e147, 4e147], 1, 1.0) == [0.0, 0.0]
  # the norm floatmax with the level 1/2: r/λ is above floatmax, the exponent difference of the parts is 1024 (not above the bound of the
  # copy exit), and the group is unchanged at a = 3.7, γ = 1
  @test prox_block(GroupSCAD(0.5, [1:1], [1.0]), [floatmax(Float64)], 1, 1.0) == [floatmax(Float64)]
  # the same exponent difference with a = γ = floatmax: the group is below the cutoff and goes to +0
  g0 = GroupSCAD(0.75, [1:1], [1.0]; a = floatmax(Float64))
  y0 = prox_block(g0, [2.0^1023], 1, floatmax(Float64))
  @test y0 == [0.0] && check_group(y0, [2.0^1023], 0.75, 1.0, floatmax(Float64), floatmax(Float64))
  # the exponent difference of the parts above the range: kept
  f3 = GroupSCAD(2.0^-600, [1:2], [2.0^-600])
  u3 = [3.0 * 2.0^600, 4.0 * 2.0^600]
  @test prox_block(f3, u3, 1, 1.0) == u3
  @test check_group(prox_block(f3, u3, 1, 1.0), u3, 2.0^-600, 2.0^-600, 3.7, 1.0)
  # f(u2) overflows (r λ ≈ 1e609 in the first piece); f3(u3) is the plateau 2.35 λ² ≈ 2^-2399, which rounds to zero
  @test close_value(f(u2), refterm(u2, 2e154, 1e154, 3.7), 2, 1) && f(u2) === Inf
  @test f3(u3) === 0.0
  @test value_and_prox!(similar(u2), f, u2, 1.0) === 0.0
end

@testset "rounded decisions and stored entries" begin
  # the stored Float64 radius sqrt(20.0) is above the exact cutoff, and the rounded cutoff gives +0
  @test isequal(prox_block(GroupSCAD(1.0, [1:1], [1.0]; a = 3.0), [sqrt(20.0)], 1, 5.0), [0.0])
  # the stored entries are rounded: the stored direction can change
  @test prox_block(GroupSCAD(1.0, [1:2], [1.0]), [1.5, nextfloat(0.0)], 1, 1.0) == [0.5, 0.0]
end

@testset "constructor ownership and default weights" begin
  # the outer constructors copy the groups and the weights
  G = [[2], [1]]
  w = [1.0, 2.0]
  weighted = GroupSCAD(1.0, G, w)
  defaulted = GroupSCAD(1.0, G)
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
    f = GroupSCAD{Float64, typeof(groups)}(1.0, 3.7, groups, ws)
    @test f.groups === groups && f.w === ws && f.group_of == [1, 2]
    @test prox(f, [30.0, 40.0], 1.0) == [30.0, 40.0]
  end
  @test_throws ArgumentError GroupSCAD{Float64, Vector{Vector{Int}}}(1.0, 3.7, [[1], [1]], [1.0, 1.0])
  # the default weight √|G| without overflowing the group size in a narrow type
  @test GroupSCAD(Float16(1), [1:65536]).w == Float16[256]
  @test GroupSCAD(1f0, [1:2, 3:5]).w == Float32[sqrt(2f0), sqrt(3f0)]
end

@testset "tiny shapes and levels ($T)" for T in (Float32, Float64)
  # the value at a shape near floatmax: the middle piece at a = floatmax(T)/2, in the ordinary and the scaled path
  M = floatmax(T)
  a = M / 2
  f = GroupSCAD(one(T), [1:1], T[1]; a)
  for x in (T(0.9) * a, T(0.5) * a, T(1e5), T(1.5))
    @test close_value(f(T[x]), refterm(T[x], 1, 1, a), 1, 1)
  end
  # in Float64 a = 1e308: the middle value at x = 0.9e308 is about 4.95e307
  if T === Float64
    h = GroupSCAD(1.0, [1:1], [1.0]; a = 1e308)
    @test h([0.9e308]) ≈ 4.95e307 rtol = 1e-14
    @test close_value(h([0.9e308]), refterm([0.9e308], 1, 1, 1e308), 1, 1)
  end
  # a level 2^E below floatmax, with a second group whose linear term r λ overflows (both norms are below their levels): Inf, not NaN
  E = exponent(floatmax(T))
  λ = ldexp(one(T), E)
  g = GroupSCAD(λ, [1:1, 2:2], T[1, 4])
  @test g(T[1 / 4, 3e10]) === T(Inf)
  z = similar(T[1, 1])
  γ = ldexp(one(T), -E - 2)                 # γλ = 1/4 for the first group and 1 for the second: both are soft
  @test value_and_prox!(z, g, T[3 / 8, 3e10], γ) === T(Inf) && isfinite(z[1]) && z[2] ≈ T(3e10)
end

@testset "zero-group screen at nonfinite values" begin
  # an infinite weight: Inf at an infinite gradient entry, NaN at a NaN one, -Inf at finite entries
  g = GroupSCAD(1.0, [1:2], [Inf])
  @test kkt_violation(g, [Inf, 1.0], zeros(2), 1) === Inf
  @test isnan(kkt_violation(g, [Inf, NaN], zeros(2), 1))
  @test kkt_violation(g, [3.0, 4.0], zeros(2), 1) === -Inf
end

@testset "scaling by powers of two ($T)" for T in (Float32, Float64)
  # prox(GroupSCAD(2^j λ0), 2^j u, γ) = 2^j prox(GroupSCAD(λ0), u, γ) across the ordinary and scaled paths; the unit-scale
  # sums of squares of these groups are exact (those of the scaled inputs can leave the range) and the inputs are not
  # within rounding of the cutoffs of the decisions; bitwise (isequal, signed zeros included) in Float64, and in Float32 bitwise
  # for zero and kept groups, within two ulps for the points in between
  E = exponent(floatmax(T))
  for (u, γ, a) in ((T[3, 4], T(1), T(3.7)), (T[1.5, 2], T(1), T(3.7)), (T[1, 2, 2], T(3), T(3)), (T[1, 2, 2], T(4), T(2.5)),
                    (T[0.75, 1], T(1), T(3.7)), (T[3, 4], T(10), T(3)), (T[3, 4], T(30), T(3)), (T[6, 8], T(12), T(3)),
                    (T[1.5, 2], T(0.5), T(10)), (T[0.75, 1], T(2), T(2.5)), (T[3, 4], T(4), T(3.7)))
    f0 = GroupSCAD(one(T), [1:length(u)], T[1]; a)
    y0 = prox_block(f0, u, 1, γ)
    for j in (-E + 8, -E ÷ 2 - 4, -E ÷ 2 + 4, -20, 20, E ÷ 2 - 4, E ÷ 2 + 4, E - 8)
      s = T(2)^j
      y = prox_block(GroupSCAD(s, [1:length(u)], T[1]; a), s .* u, 1, γ)
      if T === Float64 || all(v -> iszero(v) || v in u, y0)
        @test isequal(y, s .* y0)
      else
        # Float32 points in between: the scaled path forms the factor in Float64
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
  # row 1, w = 0: unchanged at the levels and steps below, value 0
  for λ in (z, one(T), T(Inf)), γ in steps, a in (T(3), T(Inf))
    f = GroupSCAD(λ, G, T[0]; a)
    @test isequal(pb(f, [NaN, -0.0], γ), T[NaN, -0.0]) && isequal(pb(f, [Inf, -Inf], γ), T[Inf, -Inf])
    @test f(T[NaN, Inf]) === z
  end
  # row 2, stored infinities: +0 at the steps 0, 1 and Inf; value 0 at a zero group, NaN at a NaN entry, Inf otherwise
  for (λ, w) in ((one(T), T(Inf)), (T(Inf), one(T)), (z, T(Inf)), (T(Inf), T(Inf))), γ in steps, a in (T(3), T(Inf))
    f = GroupSCAD(λ, G, T[w]; a)
    for u in ([NaN, 1], [Inf, -1], [-0.0, 0], [1, 2])
      @test all(isequal(z), pb(f, u, γ))
    end
    @test f(T[0, -0.0]) === z && isnan(f(T[NaN, Inf])) && f(T[1, 0]) === T(Inf) && f(T[Inf, 0]) === T(Inf)
  end
  # row 3, λ0 = 0 with finite w: identity, value 0 at nonfinite entries
  for γ in steps, w in (one(T), T(2)), a in (T(3), T(Inf))
    f = GroupSCAD(z, G, T[w]; a)
    @test isequal(pb(f, [NaN, -0.0], γ), T[NaN, -0.0]) && pb(f, [Inf, 1], γ) == T[Inf, 1]
    @test f(T[NaN, Inf]) === z
  end
  for a in (T(3.7), T(Inf))
    f = GroupSCAD(one(T), G, T[1]; a)
    # rows 4 and 5, positive finite parameters, γ = 0: identity, signed zeros kept; γ = Inf: +0
    @test isequal(pb(f, [NaN, -0.0], z), T[NaN, -0.0]) && pb(f, [Inf, 3], z) == T[Inf, 3]
    @test isequal(pb(f, [-0.0, 0.0], z), T[-0.0, 0.0]) && all(signbit, pb(f, [-0.0, -0.0], z))
    @test isequal(prox!(zeros(T, 2), f, T[-0.0, 0.0], z), T[-0.0, 0.0])
    @test isequal(prox!(T[-0.0, 0.0], f, T[-0.0, 0.0], z), T[-0.0, 0.0])
    for u in ([NaN, 1], [Inf, 1], [3, 4], [-0.0, 0])
      @test all(isequal(z), pb(f, u, Inf))
    end
    # row 6, finite steps: a NaN entry makes the group NaN, including where a threshold would zero or keep the group
    for γ in (T(0.5), T(3), T(12)), u in ([NaN, 1], [NaN, Inf], [Inf, NaN], [NaN, T(1e-30)], [NaN, T(1e30)])
      @test all(isnan, pb(f, u, γ))
    end
    @test isnan(f(T[NaN, Inf])) && isnan(f(T[1, NaN]))
    # an infinite entry and no NaN: unchanged, value the plateau (a + 1)λ²/2 (Inf for a = Inf)
    for γ in (T(0.5), T(3), T(12))
      @test pb(f, [Inf, 1], γ) == T[Inf, 1] && pb(f, [-Inf, Inf], γ) == T[-Inf, Inf]
    end
    @test f(T[-Inf, 1]) === (isinf(a) ? T(Inf) : (T(3.7) + 1) / 2)
    # a zero group: +0, value 0
    for γ in (T(0.5), T(3), T(12))
      @test all(isequal(z), pb(f, [-0.0, -0.0], γ))
    end
    @test f(T[-0.0, 0]) === z
  end
  # other groups are not affected by a NaN or infinite group
  f = GroupSCAD(one(T), [1:2, 3:4, 5:6], T[1, 1, 1])
  x = T[NaN, 1, 1.2, 1.6, Inf, 2]
  y = prox(f, x, one(T))
  @test all(isnan, y[1:2]) && y[5:6] == T[Inf, 2]
  @test isapprox(y[3:4], prox_block(f, T[1.2, 1.6], 2, one(T)); rtol = 4eps(T))
  one_group = GroupSCAD(one(T), [1:2], T[1])
  @test isnan(f(x)) && f(T[0, 0, 1.2, 1.6, Inf, 2]) ≈ one_group(T[1.2, 1.6]) + one_group(T[Inf, 2])
  # a = Inf: the group map and the value of GroupNormL2 at the level λ0 wₖ
  gw = T[0.5, 2, 1]
  fi = GroupSCAD(T(0.8), [1:2, 3:4, 5:6], gw; a = T(Inf))
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
  f = GroupSCAD(T(0.6), groups, T[1, 1, 0.5])
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
  @test prox(GroupSCAD(1.0, [1:2]), [3, 4], 1) == prox(GroupSCAD(1.0, [1:2]), [3.0, 4.0], 1.0)
  @test check_group(prox(GroupSCAD(1.0, [1:2]), Float32[1.2, 1.6], 0.1)[1:2], Float32[1.2, 1.6], 1.0, sqrt(2.0), 3.7, 0.1)
end

@testset "signed zeros under the storage rule" begin
  f = GroupSCAD(1.0, [1:1], [1.0])
  # dense destinations keep a stored -0.0 at a zero step, and so does an active coordinate of a sparse destination
  @test isequal(prox(f, [-0.0], 0.0), [-0.0])
  s = SparseIterate(Float64, 1)
  activate!(s, 1)
  s[1] = -0.0
  @test signbit(s[1])
  act = SparseIterate(Float64, 1)
  activate!(act, 1)
  @test prox!(act, f, [-0.0], 0.0) === act && signbit(act[1])
  # a zero written to an absent coordinate leaves it absent: a fresh sparse destination reads +0
  fresh = prox(f, s, 0.0)
  @test fresh isa SparseIterate && isequal(fresh[1], 0.0) && !signbit(fresh[1])
  fresh2 = SparseIterate(Float64, 1)
  @test prox!(fresh2, f, [-0.0], 0.0) === fresh2 && isequal(fresh2[1], 0.0) && nactive(fresh2) == 0
end

@testset "BlockIterate with an inactive block ($T)" for T in (Float32, Float64)
  groups = [1:2, 3:4, 5:6]
  f = GroupSCAD(T(0.6), groups, T[1, 1, 0.5])
  x = T[1.5, -0.2, 0, 0, 4, -3]                 # the block 2 is inactive
  b = BlockIterate(x, f)
  @test !isactive(b, 2) && nactive(b) == 2
  for γ in (T(0.5), T(3), T(0))
    y = prox(f, x, γ)
    yb = prox(f, b, γ)
    @test yb isa BlockIterate && isapprox(Vector(yb), y; rtol = 0, atol = 8eps(T) * maximum(abs, x))
    @test isequal(Vector(yb)[3:4], y[3:4])
    @test f(b) ≈ f(x)
    z = similar(b)
    @test value_and_prox!(z, f, b, γ) ≈ f(Vector(z))
    @test isapprox(Vector(z), y; rtol = 0, atol = 8eps(T) * maximum(abs, x))
    bc = copy(b)
    @test prox!(bc, f, bc, γ) === bc && isapprox(Vector(bc), y; rtol = 0, atol = 8eps(T) * maximum(abs, x))
  end
  # a compact call on the inactive block, from the iterate's own (zero) block
  out = fill(T(7), 2)
  @test prox_block!(out, f, blockview(b, 2), 2, one(T)) === out && all(isequal(zero(T)), out)
  @test !isactive(b, 2)
end

@testset "preflight refusals leave outputs unchanged ($T)" for T in (Float32, Float64)
  f = GroupSCAD(one(T), [1:2, 3:4], T[1, 1])
  hard = GroupSCAD(one(T), [1:2, 3:4], T[Inf, 1])
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
      @test_throws ArgumentError op(y, hard, x, γ)                   # also with a hard group present
    end
    @test y == sentinel()
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
  # an abstract floating-point element type is refused before the entries in no group are written
  fa = GroupSCAD(one(T), [2:2], T[1])
  for op in (prox!, value_and_prox!), step in (T(0.5), one(T), T(0))
    ya = AbstractFloat[99.0, 99.0]
    @test_throws ArgumentError op(ya, fa, T[1, 2], step)
    @test ya == AbstractFloat[99.0, 99.0]
    yb = AbstractFloat[99.0, 99.0]
    @test_throws ArgumentError op(yb, fa, T[1, 2], T(Inf))
    @test yb == AbstractFloat[99.0, 99.0]
  end
  oa = AbstractFloat[99.0]
  @test_throws ArgumentError prox_block!(oa, fa, T[2], 1, one(T))
  @test oa == AbstractFloat[99.0]
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
    @test_throws ArgumentError prox_block!(out, hard, T[1, 2], 1, γ)
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
  f = GroupSCAD(T(0.2), groups, ones(T, 4))
  x = randn(rng, T, 560)
  x[5:36] .*= T(0.01)                     # a zero group at moderate steps
  x[37:40] .= T(0.15)                     # r = 0.3: soft at γ = 0.5
  y = similar(x)
  for γ in (T(0.5), T(1.5), T(3), T(10))  # soft and middle, then the nonconvex steps
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
    xi = copy(x)
    @test allocs(prox!, xi, f, xi, γ) == 0
    @test allocs(value_and_prox!, xi, f, xi, γ) == 0
    for k in eachindex(groups)
      u = x[groups[k]]
      @test allocs(prox_block!, similar(u), f, u, k, γ) == 0
      uu = copy(u)
      @test allocs(prox_block!, uu, f, uu, k, γ) == 0
    end
  end
  @test allocs(f, x) == 0
  fi = GroupSCAD(T(0.2), groups, ones(T, 4); a = T(Inf))
  @test allocs(prox!, y, fi, x, one(T)) == 0 && allocs(value_and_prox!, y, fi, x, one(T)) == 0 && allocs(fi, x) == 0
  @test allocs(kkt_violation, f, x, zeros(T, 560), 1) == 0
end

@testset "historical reproducers" begin
  # infinite weights (#250): a hard zero at the steps 0, 1 and Inf; other groups are independent
  f = GroupSCAD(1.0, [1:2, 3:4], [Inf, 1.0])
  for γ in (0.0, 1.0, Inf)
    y = prox(f, [3.0, 4.0, 3.0, 4.0], γ)
    @test all(isequal(0.0), y[1:2])
  end
  @test prox(f, [3.0, 4.0, 3.0, 4.0], 0.0)[3:4] == [3.0, 4.0]
  @test f([0.0, -0.0, 3.0, 4.0]) ≈ 2.35 && f([1.0, 0.0, 3.0, 4.0]) === Inf
  # norms beyond the range of the input type's sum of squares (#534): the scaled norm extends the range
  x32 = Float32[2f38, 3f38]                                                          # r ≈ 3.6e38
  @test prox!(zeros(Float64, 2), GroupSCAD(1.0, [1:2]), x32, 1.0) == Float64.(x32)      # kept
  y = prox!(zeros(Float64, 2), GroupSCAD(1.5e38, [1:2], [1.0]), x32, 1.0)              # between λ and aλ
  @test all(isfinite, y) && check_group(y, x32, 1.5e38, 1.0, 3.7, 1.0)
  u = Float32[3f38, 3f38]
  y32 = prox(GroupSCAD(2f38, [1:2], [1f0]), u, 1f0)
  @test all(isfinite, y32) && check_group(y32, u, 2f38, 1f0, 3.7f0, 1f0)
  z64 = [1e300, 2e300, 2e300]
  @test check_group(prox(GroupSCAD(1e300, [1:3], [1.0]), z64, 1.0), z64, 1e300, 1.0, 3.7, 1.0)
  @test close_value(GroupSCAD(1e300, [1:3], [1.0])(z64), refterm(z64, 1e300, 1.0, 3.7), 3, 1)
end

@testset "range and extreme scales ($T)" for T in (Float32, Float64)
  rng = Xoshiro(2070 + sizeof(T))
  E = exponent(floatmax(T))
  δ = nextfloat(zero(T))
  nok = ncase = 0
  for trial in 1:400
    n = rand(rng, (1, 2, 3, 8, 33))
    eλ, ew, eg, er = rand(rng, -E-20:E-1), rand(rng, -40:40), rand(rng, -E-20:E-1), rand(rng, -E-20:E-1)
    λ0 = max(δ, ldexp(T(0.5) + rand(rng, T) / 2, eλ))
    w = ldexp(T(0.5) + rand(rng, T) / 2, ew)
    γ = max(δ, ldexp(T(0.5) + rand(rng, T) / 2, eg))
    a = rand(rng, (T(3.7), nextfloat(T(2)), T(2.5), T(1000), ldexp(T(0.75), rand(rng, 2:E-2)), ldexp(T(1.5), E - 1), T(Inf)))
    λ = biglevel(λ0, w)
    # norms near the thresholds of the step, or at a random scale
    c = rand(rng, knots(λ, a, γ))
    cut = c * rand(rng, (big"0.5", big(1), big"1.0000001", big(2)))
    r = rand(rng, Bool) ? cut : ldexp(big(1.0), er)
    big(floatmax(T)) / 2 > r > 64 * big(δ) * n || continue
    u = vector_with_norm(rng, T, n, r)
    all(isfinite, u) && any(!iszero, u) || continue
    f = GroupSCAD(λ0, [1:n], [w]; a)
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
  for (a, γ) in ((T(3.7), T(0.5)), (T(3.7), T(1.5)), (T(3.7), T(3.5)), (T(2.5), T(4)), (T(Inf), T(0.5)))
    f = GroupSCAD(T(0.4), groups, T[1, 2, 0, 1]; a)
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
  # a copied group takes its term at the norm it was copied at: here inside the middle piece, and on the plateau
  g = GroupSCAD(one(T), [1:2, 3:4], T[1, 1]; a = T(3.7))
  x = T[1.2, 1.6, 30, 40]
  y = similar(x)
  v = value_and_prox!(y, g, x, T(1e-20))
  @test y == x && fused_close(v, g(y), 2, 2) && close_value(v, refterm(x[1:2], 1, 1, T(3.7)) + refterm(x[3:4], 1, 1, T(3.7)), 2, 2)
end

@testset "Scaled ($T)" for T in (Float32, Float64)
  f = GroupSCAD(T(0.5), [1:2, 3:4], T[1, 2])
  x = T[1.2, 1.6, -3, 0.5]
  for α in (T(0.5), T(2), T(3))
    s = Scaled(f, α)
    @test s(x) ≈ α * f(x)
    @test prox(s, x, T(0.7)) == prox(f, x, α * T(0.7))
    y = similar(x)
    @test value_and_prox!(y, s, x, T(0.7)) ≈ α * f(y)
  end
  @test_throws ArgumentError Scaled(f, zero(T))
  # a factor that makes the effective step overflow gives the step Inf: both groups are +0
  @test prox(Scaled(f, floatmax(T)), x, T(2)) == zeros(T, 4)
  # an effective step that leaves the normal range is rounded: here αγ rounds to Inf and the group is +0, whereas the
  # exact step 2^1200 keeps it (r/λ = 2^700 exceeds √(γ(a + 1)))
  if T === Float64
    g = GroupSCAD(2.0^-700, [1:2], [1.0])
    u = [0.6, 0.8]
    @test prox(Scaled(g, 2.0^600), u, 2.0^600) == prox(g, u, Inf) == [0.0, 0.0]
    @test prox_block(GroupSCAD(2.0^-700, [1:2], [1.0]), u, 1, 2.0^1023) == u
    # an underflowing effective step rounds to zero: the identity at this positive finite level and finite weight
    @test prox(Scaled(GroupSCAD(1.0, [1:2]), 2.0^-600), u, 2.0^-600) == u
    # a stored infinite level still gives +0 at that zero step
    @test isequal(prox(Scaled(GroupSCAD(Inf, [1:1], [1.0]), 2.0^-600), [0.5], 2.0^-600), [0.0])
    # a nonzero subnormal effective step (2^-1025) is used as it is: γλ = 1/4, soft
    @test prox(Scaled(GroupSCAD(2.0^1023, [1:1], [1.0]), 2.0^-512), [0.5], 2.0^-513) == [0.25]
    # a negative step whose product with the factor rounds to -0 reaches the function as a zero step
    @test prox(Scaled(GroupSCAD(1.0, [1:1], [1.0]), nextfloat(0.0)), [2.0], -0.25) == [2.0]
  end
end

@testset "refused methods ($T)" for T in (Float32, Float64)
  f = GroupSCAD(one(T), [1:2, 3:4])
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

@testset "working type of the zero-group screen" begin
  f = GroupSCAD(1.0, [[1]], [1.0])
  big128 = typemax(UInt128)
  # the working type is W = float(promote_type(eltype(v), eltype(x))): a finite stored gradient entry that overflows on
  # conversion to W is infinite there
  @test kkt_violation(f, UInt128[big128], Float32[0], 1) === Inf32
  @test kkt_violation(f, [10^6], Float16[0], 1) === Inf16
  # the entry is converted before the subtraction: a residual formed in a wider type would be 0 here
  @test kkt_violation(GroupSCAD(1e6, [1:1], [1.0]), [10^6], Float16[0], 1) === Inf16
  # an entry in no group is converted in the same way
  @test kkt_violation(f, UInt128[0, big128], Float32[0, 0], 2) === Inf32
  # entries that fit are scored as finite entries in W (here Float64: the Float64 gradient is not narrowed)
  @test kkt_violation(f, [3.0], Float32[0], 1) === 2.0
  @test kkt_violation(f, [1e300], Float32[0], 1) === 1e300 - 1.0
  @test kkt_violation(f, Float32[3], Float32[0], 1) === 2f0
end

@testset "zero-group KKT screen ($T)" for T in (Float32, Float64)
  groups = [[3, 1], 4:5, [2]]
  w = T[1, 2, 0.5]
  f = GroupSCAD(T(0.75), groups, w)
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
  # -0 counts as zero; NaN in the gradient gives NaN; an infinite weight gives -Inf at finite gradient entries; a zero
  # weight gives ‖v_G‖₂
  @test kkt_violation(f, T[3, 0, 4, 0, 0, 0, 0], T[-0.0, 0, 0, 0, 0, 0, 0], 1) == T(5) - T(0.75)
  @test isnan(kkt_violation(f, T[NaN, 0, 4, 0, 0, 0, 0], zeros(T, 7), 3))
  @test kkt_violation(GroupSCAD(one(T), [1:2], T[Inf]), T[3, 4], zeros(T, 2), 1) === -T(Inf)
  @test kkt_violation(GroupSCAD(one(T), [1:2], T[0]), T[3, 4], zeros(T, 2), 1) === T(5)
  @test kkt_violation(GroupSCAD(T(Inf), [1:2], T[0]), T[3, 4], zeros(T, 2), 1) === T(5)
  # an overflowed product λ0 w is recovered
  if T === Float64
    v = fill(floatmax(T), 4)
    g = GroupSCAD(1.5 * 2.0^512, [1:4], [2.0^512])
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
