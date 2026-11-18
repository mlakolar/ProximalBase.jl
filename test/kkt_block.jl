module KktBlockTest
# The point-aware forms of kkt_violation (#970, stage B) for GroupNormL2, NormL2 and Scaled:
#   kkt_violation(f, v, x, k)   the score of the group of entry k of an array point, or of group k of a BlockIterate;
#   kkt_violation(f, v, x)      max(0, max over the groups of the block scores, over the entries in no group of |vᵢ|);
#   NormL2: the whole vector is one block, and the full-point form is the one that it has;
#   Scaled: the score of a·f, with a the product of the factors.
# A block of the penalty c‖·‖₂ scores
#   ‖v_G‖₂ - c                    at x_G = 0   (the signed screen; its positive part is the distance from -v_G to the
#                                              subdifferential at 0, the ball of radius c),
#   ‖v_G + c x_G/‖x_G‖₂‖₂        at x_G ≠ 0   (the distance from -v_G to the subdifferential {c x_G/‖x_G‖₂}).
# Each reference is a BigFloat evaluation of that definition, for c the exact product of the level and the weight (and,
# under Scaled, of the factors); no reference calls a ProximalBase kernel or the formula α score(v/α, x) of the route.
# A value is accepted within (n + 8) u (‖v_G‖₂ + c) + 4 n nextfloat(zero(W)) of its reference, u = eps(W)/2: the bound of the
# ordinary path derived in the PR plus the subnormal spacing of W.
# The excluded band is (1 - 100 eps(W)) floatmax(W) < |ref| <= (1 + 100 eps(W)) floatmax(W); references beyond it must
# overflow with the correct sign (blockok).
#
# Which tests fail at the base (2f787fd8): the nonzero blocks of GroupNormL2 (an ArgumentError there), the full-point
# forms of GroupNormL2, NormL2 and Scaled (MethodError), NormL2 with a local form (MethodError, not ArgumentError),
# Scaled at a nonzero coordinate or group, the three forms of a chain around LinearFunction and a centered SqrNormL2 at
# v of order one, chains at nonzero points, mixed element types (Float32 result at the base for a Float32 v and a
# Float64 x). The malformed-parameter test and the zero-point values pass at the base.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random, OffsetArrays
using ..TestUtils
const PB = ProximalBase

const TYPES = (Float32, Float64)
const PREC = 1024
bigprec(f) = setprecision(f, BigFloat, PREC)

msg(f) = try f(); "" catch e; sprint(showerror, e) end
etype(f) = try f(); nothing catch e; typeof(e) end
same(a, b) = isequal(a, b) && typeof(a) === typeof(b)
wt(w, i) = w === nothing ? 1 : w[i]
center(y, i) = y === nothing ? 0 : y[i]
sgn(x) = x > 0 ? one(x) : x < 0 ? -one(x) : zero(x)

# ---- references --------------------------------------------------------------------------------------------------
# the score of a block from the definition; c is the exact coefficient, v and x are BigFloat vectors of finite entries
function refblock(c, v, x)
  nv = sqrt(sum(abs2, v))
  all(iszero, x) && return nv - c
  nx = sqrt(sum(abs2, x))
  sqrt(sum(abs2, v .+ (c / nx) .* x))
end
# the accuracy bound of the ordinary path, plus the subnormal spacing of the type (a term of the cold path may be rounded
# there)
blocktol(n, ::Type{W}, v, c) where {W} =
  (n + 8) * big(eps(W)) / 2 * (sqrt(sum(abs2, v)) + c) + 4 * n * big(nextfloat(zero(W)))
# r is the score of the block: within the bound where the reference is in the range of W, and an overflow to Inf where it
# is above it; the narrow band at floatmax is not decided
function blockok(r, ::Type{W}, λ, w, v, x) where {W}
  bigprec() do
    c = (iszero(λ) || iszero(w)) ? big(0) : big(λ) * big(w)
    V, X = big.(v), big.(x)
    ref = refblock(c, V, X)
    lim = big(floatmax(W))
    ref > lim * (1 + 100 * big(eps(W))) && return r == W(Inf)
    ref < -lim * (1 + 100 * big(eps(W))) && return r == W(-Inf)
    abs(ref) > lim * (1 - 100 * big(eps(W))) && return true
    isfinite(r) && abs(big(r) - ref) <= blocktol(length(v), W, V, c)
  end
end

# the entry scores of a·f from the definition, with a the BigFloat product of the factors, and the size of their terms
function refentry(f::NormL1, a, v, x, i)
  c = a * big(f.λ0) * big(wt(f.w, i))
  V, X = big(v), big(x)
  (iszero(X) ? abs(V) - c : abs(V + c * sgn(X)), abs(V) + c)
end
function refentry(f::ElasticNet, a, v, x, i)
  c = a * big(f.λ1) * big(wt(f.w, i))
  V, X, r = big(v), big(x), a * big(f.λ2) * big(x)
  (iszero(X) ? abs(V) - c : abs(V + r + c * sgn(X)), abs(V) + abs(r) + c)
end
refentry(::ZeroFunction, a, v, x, i) = (abs(big(v)), abs(big(v)))
refentry(f::LinearFunction, a, v, x, i) = (abs(big(v) + a * big(f.c[i])), abs(big(v)) + abs(a * big(f.c[i])))
function refentry(f::SqrNormL2, a, v, x, i)
  d = a * big(f.λ) * (big(x) - big(center(f.y, i)))
  (abs(big(v) + d), abs(big(v)) + abs(d))
end
# a coordinate of a function with groups: the block of its group (the entry alone if it is in no group)
function refentry(f::GroupNormL2, a, v, x, i)
  k = i <= length(f.group_of) ? f.group_of[i] : 0
  k == 0 && return (abs(big(v[i])), abs(big(v[i])))
  G = f.groups[k]
  c = a * big(f.λ0) * big(f.w[k])
  V = big.(v[G])
  (refblock(c, V, big.(x[G])), sqrt(sum(abs2, V)) + c)
end
# the entry or block reference of a leaf at the arrays v and x
entryref(f, a, v, x, i) = f isa GroupNormL2 ? refentry(f, a, v, x, i) : refentry(f, a, v[i], x[i], i)
# two values of one score that two kernels computed in an order of their own (a block is summed in a vectorized loop for an
# Array and in a plain one for another storage): equal up to the size of the terms
close(a, b, ::Type{T}, s) where {T} = abs(a - b) <= 16 * eps(T) * s
# within the bound of the route α score(v/α, x): a few more roundings than the kernel, n for the sums of a block
within(r, ref, scale, n, ::Type{W}) where {W} =
  isfinite(r) && abs(big(r) - ref) <= (n + 12) * big(eps(W)) / 2 * scale + 4 * n * big(nextfloat(zero(W)))

# ---- data ----------------------------------------------------------------------------------------------------------
# exponents that put the entries of a block at the edges of the range of T: squares that overflow and underflow, entries near
# floatmax, subnormal entries
function exps(::Type{T}) where {T}
  emax, emin, p = exponent(floatmax(T)), exponent(floatmin(T)), precision(T)
  (0, emax ÷ 2 + 2, emax - 7, emin ÷ 2 - 3, emin + 3, emin - p ÷ 2, emin - p + 2)
end
# (level, weight) pairs: ordinary, a zero level, a zero weight, tiny and subnormal and large coefficients
function params(::Type{T}) where {T}
  emax, emin = exponent(floatmax(T)), exponent(floatmin(T))
  [(T(0.37), T(0.61)), (T(0), T(1)), (T(1), T(0)), (ldexp(T(1), emin + 3), T(1)), (nextfloat(zero(T)) * 3, T(1)),
   (ldexp(T(1.3), emax ÷ 2), T(0.75)), (ldexp(T(1.3), emax - 4), T(1)), (floatmax(T), T(1)), (floatmax(T), T(1.5)), (T(Inf), T(0))]
end
function blockdata(rng, ::Type{T}, n, ev, ex; zero_x = false) where {T}
  v = [ldexp(randn(rng, T), ev) for _ in 1:n]
  x = [ldexp(randn(rng, T), ex) for _ in 1:n]
  x[rand(rng, n) .< 0.3] .= zero(T)
  zero_x ? fill!(x, zero(T)) : (x[1] = ldexp(T(1.5), ex))
  v, x
end
fgroup(T, λ, w, n) = GroupNormL2(λ, [collect(1:n)], T[w])

# chains of two wrappers whose product is not a normal number, and a level that folds into it
tiny(T) = ldexp(one(T), T === Float32 ? -100 : -600)           # tiny(T)^2 is not a normal number
lvl(T) = ldexp(one(T), T === Float32 ? 100 : 1000)             # tiny(T)^2 * lvl(T) is
chain(T, f) = (c = Scaled(Scaled(f, tiny(T)), tiny(T)); c.f isa Scaled || error("merged"); c)
chain_product(T) = bigprec(() -> big(tiny(T))^2)
folded_level(T) = ldexp(one(T), T === Float32 ? -100 : -200)           # tiny(T)^2 lvl(T)

# a point with zeros, groups and entries that no group holds
function groupdata(rng, ::Type{T}) where {T}
  groups = [[2, 7, 3], [9], [1, 5], [4]]             # entries 6, 8, 10 and 11 are in no group
  w = T[1, 0.5, 2, 0]
  f = GroupNormL2(T(0.8), groups, w)
  v = 2 .* randn(rng, T, 11)
  x = randn(rng, T, 11)
  x[[1, 5]] .= 0                                      # group 3 is zero; group 4 has weight 0
  x[[6, 10]] .= 0
  f, v, x
end

# ---- 1. the Stage 0 example and the zero block -----------------------------------------------------------------------
# fails at the base: the nonzero block (an ArgumentError there)
@testset "$T the group example of the contract" for T in TYPES
  g = GroupNormL2(T(1), [[1, 2], [3, 4]], T[1, 1])
  x = T[3, 4, 0, 0]
  v = T[-0.6, -0.8, 0.3, 0.4]
  tol = 20 * eps(T)                               # (n + 8) u (‖v_G‖ + c) for n = 2, ‖v_G‖ = c = 1
  # the nonzero block is stationary to the bound; the two-pass kernel gives 2^-53, not 0, in Float64
  r1, r2 = kkt_violation(g, v, x, 1), kkt_violation(g, v, x, 2)
  @test r1 === r2 && typeof(r1) === T
  @test 0 <= r1 <= tol
  @test blockok(r1, T, T(1), T(1), v[1:2], x[1:2])
  # the zero block: the signed margin 0.5 - 1
  for k in (3, 4)
    r = kkt_violation(g, v, x, k)
    @test r ≈ -T(0.5) atol = 4 * eps(T)
    @test blockok(r, T, T(1), T(1), v[3:4], x[3:4])
    @test r === PB._l2_screen(T(1), T(1), v, 3:4)         # the value of the screen kernel, bit for bit
  end
  # the full point: the positive part of the largest score
  R = kkt_violation(g, v, x)
  @test R === max(zero(T), r1) && 0 <= R <= tol
  # a zero point has the signed margin of each block, and the full point drops the negative ones
  z = zeros(T, 4)
  @test kkt_violation(g, v, z, 1) ≈ norm(v[1:2]) - 1 atol = 4 * eps(T)
  vz = T[0.3, 0.4, -0.3, 0.1]
  @test kkt_violation(g, vz, z, 1) < 0 && kkt_violation(g, vz, z, 3) < 0 && kkt_violation(g, vz, z) === zero(T)
  @test kkt_violation(g, 3 * v, z) ≈ norm(3 * v[1:2]) - 1 atol = 8 * eps(T)
  # signed zeros are zeros; a subnormal entry is not
  @test kkt_violation(g, v, T[-0.0, 0.0, 0, 0], 1) === kkt_violation(g, v, z, 1)
  @test kkt_violation(g, v, T[nextfloat(zero(T)), 0, 0, 0], 1) ≈ norm(v[1:2] + [T(1), 0]) atol = 20 * eps(T)
  # an empty point
  @test kkt_violation(GroupNormL2(T(1), Vector{Int}[], T[]), T[], T[]) === zero(T)
  @test kkt_violation(NormL2(T(1)), T[], T[]) === zero(T)
end

# ---- 2. nonzero and zero blocks over the range ------------------------------------------------------------------------
# fails at the base: the nonzero blocks
@testset "$T blocks against the definition over magnitudes, sizes and coefficients" for T in TYPES
  rng = MersenneTwister(970)
  bad = Tuple[]
  for n in (1, 2, 3, 7, 8, 16, 33, 64), ev in exps(T), ex in exps(T), (λ, w) in params(T), zero_x in (false, true)
    ev == ex || rand(rng) < 0.25 || continue         # the pairs of different magnitudes are sampled
    v, x = blockdata(rng, T, n, ev, ex; zero_x)
    r = kkt_violation(fgroup(T, λ, w, n), v, x, 1)
    typeof(r) === T || push!(bad, (:type, n, ev, ex, λ, w, typeof(r)))
    blockok(r, T, λ, w, v, x) || push!(bad, (n, ev, ex, λ, w, zero_x, r))
  end
  isempty(bad) || println(first(bad, 5))
  @test isempty(bad)
  # c = floatmax: the entry c x/‖x‖ may round to an infinity, and the residual is then formed at half the scale
  fm = fgroup(T, floatmax(T), T(1), 1)
  for ex in (0, exponent(floatmax(T)) ÷ 2 + 2, exponent(floatmin(T)) ÷ 2 - 3)
    x = T[ldexp(T(1.5), ex)]
    @test kkt_violation(fm, T[-floatmax(T) / 4], x, 1) ≈ 3 * (floatmax(T) / 4) rtol = 8 * eps(T)
    @test kkt_violation(fm, T[floatmax(T) / 4], x, 1) === T(Inf)               # the true value is 1.25 floatmax
  end
end

@testset "$T a proximal point is stationary" for T in TYPES
  rng = MersenneTwister(971)
  groups = [[2, 7, 3], [1, 5], [4], [8, 6]]
  for γ in (T(0.3), T(2)), λ in (T(0.8), T(5)), trial in 1:20
    f = GroupNormL2(λ, groups, T[1, 0.5, 2, 1.5])
    z = 3 .* randn(rng, T, 8)
    y = prox(f, z, γ)
    v = (y .- z) ./ γ                                   # 0 ∈ v + ∂f(y)
    scale = norm(v) + λ * 2
    @test kkt_violation(f, v, y) <= 40 * eps(T) * scale
    for k in 1:4                                        # a group that the step sets to zero has a nonpositive margin
      G = groups[k]
      iszero(y[G]) && @test kkt_violation(f, v, y, first(G)) <= 40 * eps(T) * scale
    end
  end
end

# ---- 3. the full-point form ----------------------------------------------------------------------------------------
# fails at the base: the full-point form (MethodError)
@testset "$T full-point form" for T in TYPES
  rng = MersenneTwister(972)
  f, v, x = groupdata(rng, T)
  s = norm(v) + 8
  # the maximum of the local scores and |vᵢ| of the entries in no group, and 0
  locals = [kkt_violation(f, v, x, first(G)) for G in f.groups]
  uncovered = [abs(v[i]) for i in (6, 8, 10, 11)]
  R = kkt_violation(f, v, x)
  @test typeof(R) === T && close(R, max(zero(T), maximum(locals), maximum(uncovered)), T, s)
  for G in f.groups, i in G
    @test kkt_violation(f, v, x, i) === kkt_violation(f, v, x, first(G))     # one score for the whole group
  end
  # the definition
  ref = bigprec() do
    sc = [refentry(f, big(1), v, x, first(G))[1] for G in f.groups]
    max(big(0), maximum(sc), maximum(abs.(big.(v[[6, 8, 10, 11]]))))
  end
  @test abs(big(R) - ref) <= 16 * big(eps(T)) * s
  # the entries that no group holds decide when they are the largest, and x is not read there
  v2 = copy(v); v2[10] = 100
  x2 = copy(x); x2[10] = 5
  @test kkt_violation(f, v2, x2) === T(100) && kkt_violation(f, v2, x2, 10) === T(100)
  v2[10] = -100
  @test kkt_violation(f, v2, x2) === T(100)
  # a zero point drops the negative margins: the weight zero of group 4 and the entries in no group are left
  z = zeros(T, 11)
  vs = T(0.01) .* v
  @test kkt_violation(f, vs, z) === max(zero(T), maximum(abs.(vs[[6, 8, 10, 11, 4]])))
  # SparseIterate and BlockIterate points
  sx = SparseIterate(x)
  sx[5] = 0                                              # a stored zero
  @test close(kkt_violation(f, v, sx), R, T, s)
  for i in 1:11
    @test close(kkt_violation(f, v, sx, i), kkt_violation(f, v, x, i), T, s)
  end
  bgroups = [[2, 7, 3], [9], [1, 5], [4], [6, 8, 10, 11]]
  fb = GroupNormL2(T(0.8), bgroups, T[1, 0.5, 2, 0, 0])
  bx = BlockIterate(copy(x), bgroups)
  for k in 1:4
    @test close(kkt_violation(fb, v, bx, k), kkt_violation(f, v, x, first(bgroups[k])), T, s)
  end
  @test close(kkt_violation(fb, v, bx, 5), norm(v[[6, 8, 10, 11]]), T, s)             # the weight zero: ‖v_G‖
  @test close(kkt_violation(fb, v, bx), max(zero(T), maximum(locals[1:4]), norm(v[[6, 8, 10, 11]])), T, s)
  # an empty point; groups reaching past the point
  @test kkt_violation(GroupNormL2(T(1), Vector{Int}[], T[]), T[], T[]) === zero(T)
  @test_throws DimensionMismatch kkt_violation(f, v[1:8], x[1:8])
  # a point longer than the groups: the extra entries are in no group
  @test kkt_violation(f, [v; T(500)], [x; T(1)]) === T(500)
  # no mutation
  v0, x0 = copy(v), copy(x)
  kkt_violation(f, v, x); kkt_violation(f, v, x, 3)
  @test v == v0 && x == x0
end

# ---- 4. exceptional values and their precedence -----------------------------------------------------------------------
# fails at the base: the rows at a nonzero block, and the Inf of an infinite v with an infinite coefficient (NaN there)
@testset "$T exceptional values" for T in TYPES
  nan, inf = T(NaN), T(Inf)
  g(λ, w) = GroupNormL2(T(λ), [[1, 2, 3]], T[w])
  x0 = zeros(T, 3)
  xn = T[0, 2, 0]
  v = T[0.5, -1, 2]
  # a NaN in v or x gives NaN, at a zero and at a nonzero block, and before an infinity
  for (vv, xx) in ((T[nan, 1, 2], x0), (T[nan, 1, 2], xn), (v, T[0, nan, 0]), (T[inf, 1, 2], T[0, nan, 0]),
                   (T[nan, 1, 2], T[inf, 0, 0]), (T[nan, inf, 2], xn))
    for (λ, w) in ((0.3, 1), (0, 1), (Inf, 1), (1, 0))
      r = kkt_violation(g(λ, w), vv, xx, 1)
      @test isnan(r) && typeof(r) === T
      @test isnan(kkt_violation(g(λ, w), vv, xx))
    end
  end
  # otherwise an infinite entry gives +Inf: at a nonzero block, and at a zero block with any coefficient (c = Inf included)
  for (vv, xx) in ((T[inf, 1, 2], x0), (T[-inf, 1, 2], x0), (T[inf, 1, 2], xn), (v, T[0, inf, 0]), (v, T[0, -inf, 0]))
    for (λ, w) in ((0.3, 1), (0, 1), (Inf, 1), (1, 0), (Inf, 0))
      @test kkt_violation(g(λ, w), vv, xx, 1) === inf
      @test kkt_violation(g(λ, w), vv, xx) === inf
    end
  end
  # a NaN in the full point wins over an Inf in another group
  f2 = GroupNormL2(T(0.3), [[1, 2], [3]], T[1, 1])
  @test isnan(kkt_violation(f2, T[inf, 0, nan], zeros(T, 3))) && isnan(kkt_violation(f2, T[nan, 0, inf], zeros(T, 3)))
  @test isnan(kkt_violation(f2, T[inf, 0, 1], T[0, 0, nan])) && kkt_violation(f2, T[inf, 0, 1], T[0, 0, 2]) === inf
  # an infinite coefficient: a zero block does not violate it (-Inf), and a nonzero one does (Inf)
  for (λ, w) in ((Inf, 1), (1, Inf), (Inf, Inf))
    @test kkt_violation(g(λ, w), v, x0, 1) === -inf
    @test kkt_violation(g(λ, w), v, xn, 1) === inf
    @test kkt_violation(g(λ, w), v, x0) === zero(T) && kkt_violation(g(λ, w), v, xn) === inf
  end
  # a zero coefficient (a zero level or weight, an infinite level with a zero weight): ‖v_G‖ at a nonzero block and at a zero one
  nv = norm(v)
  for (λ, w) in ((0, 1), (1, 0), (Inf, 0))
    @test kkt_violation(g(λ, w), v, xn, 1) ≈ nv atol = 4 * eps(T)
    @test kkt_violation(g(λ, w), v, x0, 1) ≈ nv atol = 4 * eps(T)
  end
  # an infinite weight at a zero level has the coefficient Inf, as the screen of NormL1 has
  @test kkt_violation(g(0, Inf), v, x0, 1) === -inf && kkt_violation(g(0, Inf), v, xn, 1) === inf
  # a coefficient that overflows in T is infinite
  @test kkt_violation(g(floatmax(T), 2), v, xn, 1) === inf
  @test kkt_violation(g(floatmax(T), 2), v, x0, 1) === -inf
  # an entry in no group: |vᵢ|, with the same rules
  f3 = GroupNormL2(T(0.3), [[1, 2]], T[1])
  @test kkt_violation(f3, T[1, 2, -7], zeros(T, 3), 3) === T(7)
  @test isnan(kkt_violation(f3, T[1, 2, nan], zeros(T, 3), 3)) && kkt_violation(f3, T[1, 2, -inf], zeros(T, 3), 3) === inf
  @test kkt_violation(f3, T[1, 2, 5], T[0, 0, 9], 3) === T(5)
  # -0.0 and a subnormal: the zero test of the shrinkage kernel
  @test kkt_violation(g(0.3, 1), v, T[-0.0, 0.0, -0.0], 1) === kkt_violation(g(0.3, 1), v, x0, 1)
  @test kkt_violation(g(0.3, 1), v, T[0, nextfloat(zero(T)), 0], 1) ≈ norm(v + T(0.3) * T[0, 1, 0]) atol = 20 * eps(T)
  @test kkt_violation(g(0.3, 1), v, T[0, -nextfloat(zero(T)), 0], 1) ≈ norm(v - T(0.3) * T[0, 1, 0]) atol = 20 * eps(T)
  # NormL2 with the same rules
  h(λ) = NormL2(T(λ))
  @test isnan(kkt_violation(h(1), T[nan, 1], T[0, 2])) && isnan(kkt_violation(h(Inf), T[nan, 1], T[0, 0]))
  @test kkt_violation(h(1), T[inf, 1], T[0, 2]) === inf && kkt_violation(h(Inf), T[inf, 1], T[0, 0]) === inf
  @test kkt_violation(h(Inf), T[1, 1], T[0, 0]) === zero(T) && kkt_violation(h(Inf), T[1, 1], T[0, 2]) === inf
  @test kkt_violation(h(0), T[3, 4], T[0, 2]) ≈ 5 && kkt_violation(h(0), T[3, 4], T[0, 0]) ≈ 5
end

# ---- 5. validation ------------------------------------------------------------------------------------------------------
# fails at the base: the full-point and nonzero-block checks reached through GroupNormL2, NormL2 and Scaled
@testset "$T validation" for T in TYPES
  groups = [[1, 2], [3, 4]]
  f = GroupNormL2(T(0.5), groups, T[1, 1])
  v = T[1, 2, 3, 4]
  x = T[0, 1, 0, 0]
  bx = BlockIterate(copy(x), groups)
  nl = NormL2(T(0.5))
  fs = (f, Scaled(f, T(2)), Scaled(Scaled(f, T(2)), T(3)))
  cx = Complex{T}.(x)
  cv = Complex{T}.(v)
  off(a) = OffsetArray(a, 0:length(a)-1)
  for h in (f, nl, Scaled(f, T(2)), Scaled(nl, T(2)))
    @test_throws ArgumentError kkt_violation(h, cv, x)
    @test_throws ArgumentError kkt_violation(h, v, cx)
    @test occursin("offset arrays", msg(() -> kkt_violation(h, off(v), x)))
    @test occursin("offset arrays", msg(() -> kkt_violation(h, v, off(x))))
    @test_throws DimensionMismatch kkt_violation(h, v[1:3], x)
    @test_throws DimensionMismatch kkt_violation(h, v, x[1:3])
    @test_throws ArgumentError kkt_violation(h, v, SymmetricSparseIterate(T[1 2; 2 1]))
  end
  for h in fs
    @test_throws ArgumentError kkt_violation(h, cv, x, 2)
    @test_throws ArgumentError kkt_violation(h, v, cx, 2)
    @test occursin("offset arrays", msg(() -> kkt_violation(h, off(v), x, 2)))
    @test occursin("offset arrays", msg(() -> kkt_violation(h, v, off(x), 2)))
    @test_throws DimensionMismatch kkt_violation(h, v[1:3], x, 2)
    @test_throws DimensionMismatch kkt_violation(h, v, x[1:3], 2)
    @test_throws BoundsError kkt_violation(h, v, x, 5)
    @test_throws BoundsError kkt_violation(h, v, x, 0)
    @test_throws ArgumentError kkt_violation(h, ones(T, 2, 2), SymmetricSparseIterate(T[1 2; 2 1]), 1)
  end
  # the groups reach past the point; the layout of a BlockIterate
  @test_throws DimensionMismatch kkt_violation(f, v[1:3], x[1:3])
  @test_throws DimensionMismatch kkt_violation(f, v, x[1:3], 1)
  for layout in ([[1, 3], [2, 4]], [[1], [2], [3], [4]], [[1, 2, 3, 4]], [[2, 1], [3, 4]])
    bad = BlockIterate(T, 4, layout)
    bad[2] = 1
    @test_throws DimensionMismatch kkt_violation(f, v, bad, 1)
    @test_throws DimensionMismatch kkt_violation(f, v, bad)
    @test_throws DimensionMismatch kkt_violation(Scaled(f, T(2)), v, bad)
  end
  @test_throws BoundsError kkt_violation(f, v, bx, 3)
  # the full-point layout check covers all the groups; the local form compares the group that it reads
  mixed = BlockIterate(T, 4, [[1, 2], [4, 3]])
  @test kkt_violation(f, v, mixed, 1) isa T
  @test_throws DimensionMismatch kkt_violation(f, v, mixed, 2)
  @test_throws DimensionMismatch kkt_violation(f, v, mixed)
  # the three-argument form is the one that GroupNormL2 refuses
  @test_throws ArgumentError kkt_violation(f, v, 1)
  @test_throws ArgumentError kkt_violation(f, cv, 1)
  @test_throws ArgumentError kkt_violation(nl, v, 1)                     # NormL2 has no screen of an entry
  @test occursin("full-point", msg(() -> kkt_violation(nl, v, 1)))
  @test_throws ArgumentError kkt_violation(nl, cv, 1)
  @test_throws ArgumentError kkt_violation(Scaled(nl, T(2)), v, 1)
  @test_throws ArgumentError kkt_violation(Scaled(Scaled(nl, tiny(T)), tiny(T)), v, 1)
  # a range, a view and a transposed matrix are points
  @test close(kkt_violation(f, view(v, 1:4), view(x, 1:4)), kkt_violation(f, v, x), T, 10)
  @test close(kkt_violation(f, range(T(1), T(4), length = 4), x), kkt_violation(f, v, x), T, 10)
  M = reshape(copy(x), 2, 2)
  fm = GroupNormL2(T(0.5), [[1, 3], [2, 4]], T[1, 1])
  @test close(kkt_violation(fm, reshape(v, 2, 2), M), kkt_violation(fm, v, x), T, 10)
  @test close(kkt_violation(nl, reshape(v, 2, 2), M), kkt_violation(nl, v, x), T, 10)
  @test close(kkt_violation(f, v, x), kkt_violation(f, v, SparseIterate(x)), T, 10)
end

# ---- 6. NormL2 -------------------------------------------------------------------------------------------------------------
# fails at the base: all of it (NormL2 has no kkt_violation method)
@testset "$T NormL2" for T in TYPES
  rng = MersenneTwister(973)
  bad = Tuple[]
  for n in (1, 2, 5, 17, 64), ev in exps(T), ex in exps(T), λ in (T(0.37), T(0), ldexp(T(1.3), exponent(floatmax(T)) ÷ 2)), zero_x in (false, true)
    ev == ex || rand(rng) < 0.15 || continue
    v, x = blockdata(rng, T, n, ev, ex; zero_x)
    r = kkt_violation(NormL2(λ), v, x)
    typeof(r) === T || push!(bad, (:type, n))
    # the full-point value is the positive part of the block score
    ok = bigprec() do
      ref = max(big(0), refblock(big(λ), big.(v), big.(x)))
      lim = big(floatmax(T))
      ref > lim * (1 + 100 * big(eps(T))) ? r == T(Inf) :
        abs(ref) > lim * (1 - 100 * big(eps(T))) ? true :
        isfinite(r) && abs(big(r) - ref) <= blocktol(n, T, big.(v), big(λ))
    end
    ok || push!(bad, (n, ev, ex, λ, zero_x, r))
  end
  isempty(bad) || println(first(bad, 5))
  @test isempty(bad)
  # a stationary point, a zero vector inside and outside the ball, the unit examples
  nl = NormL2(T(1))
  @test kkt_violation(nl, T[0.3, 0.4], T[0, 0]) === zero(T)               # ‖v‖ - λ < 0 is dropped
  @test kkt_violation(nl, T[3, 4], T[0, 0]) ≈ 4 atol = 4 * eps(T)
  @test kkt_violation(nl, T[-0.6, -0.8], T[3, 4]) <= 20 * eps(T)
  @test kkt_violation(nl, T[0, 0], T[3, 4]) ≈ 1 atol = 4 * eps(T)
  # points that are read as one vector: SparseIterate, BlockIterate, a matrix
  x = T[0, 1.5, 0, -2]
  v = T[1, -2, 0.5, 3]
  r = kkt_violation(nl, v, x)
  @test close(kkt_violation(nl, v, SparseIterate(x)), r, T, 10)
  @test close(kkt_violation(nl, v, BlockIterate(copy(x), [[1, 4], [2, 3]])), r, T, 10)
  @test close(kkt_violation(nl, v, BlockIterate(copy(x), [[1, 2, 3, 4]])), r, T, 10)
  @test close(kkt_violation(nl, reshape(v, 2, 2), reshape(x, 2, 2)), r, T, 10)
  # the other forms of NormL2 are refused, and name the full-point form
  for pt in (x, SparseIterate(x), BlockIterate(copy(x), [[1, 4], [2, 3]]))
    @test_throws ArgumentError kkt_violation(nl, v, pt, 1)
    @test occursin("full-point", msg(() -> kkt_violation(nl, v, pt, 2)))
  end
  S = SymmetricSparseIterate(T[1 2; 2 1])
  @test_throws ArgumentError kkt_violation(nl, ones(T, 2, 2), S, 1)
  @test_throws ArgumentError kkt_violation(nl, ones(T, 2, 2), S)
  @test occursin("SymmetricSparseIterate", msg(() -> kkt_violation(nl, ones(T, 2, 2), S)))
  # the group of one group is NormL2 with the weight one, bit for bit
  @test close(kkt_violation(GroupNormL2(T(0.7), [collect(1:4)], T[1]), v, x), kkt_violation(NormL2(T(0.7)), v, x), T, 10)
  @test applicable(kkt_violation, nl, v, x)
end

# ---- 7. Scaled, one wrapper ----------------------------------------------------------------------------------------------
# fails at the base: a nonzero coordinate or group, the full-point form
function scaled_leaves(rng, ::Type{T}, n) where {T}
  w = T(0.2) .+ rand(rng, T, n)
  w[2] = 0
  y, c = randn(rng, T, n), randn(rng, T, n)
  [NormL1(T(0.3)), NormL1(T(0.3), w), ElasticNet(T(0.3), T(0.7)), ElasticNet(T(0.3), T(0.7), w), ZeroFunction(),
   LinearFunction(c), SqrNormL2(T(0.7)), SqrNormL2(T(0.7), y), GroupNormL2(T(0.4), [[1, 5, 3], [4, 2]], T[1, 0.6])]
end
@testset "$T Scaled with one factor against the definition" for T in TYPES
  rng = MersenneTwister(974)
  n = 6                                                   # entry 6 is in no group
  blocks = [[1, 5, 3], [4, 2], [6]]
  for leaf in scaled_leaves(rng, T, n), α in (T(0.5), T(2), T(3.7), T(1e-3), T(1e3), T(1))
    s = Scaled(leaf, α)
    x, v = randn(rng, T, n), 2 .* randn(rng, T, n)
    x[[1, 4]] .= 0
    sc = norm(v) + 10
    a = bigprec(() -> big(α))
    # all coordinates, for array, SparseIterate and BlockIterate points
    bx = BlockIterate(copy(x), blocks)
    sx = SparseIterate(x)
    for i in 1:n
      r = kkt_violation(s, v, x, i)
      @test typeof(r) === T
      bigprec() do
        ref, scale = entryref(leaf, a, v, x, i)
        @test within(r, ref, scale, leaf isa GroupNormL2 ? 3 : 1, T)
      end
      @test close(kkt_violation(s, v, sx, i), r, T, sc)
    end
    if !(leaf isa GroupNormL2)
      for k in 1:3
        r = kkt_violation(s, v, bx, k)
        bigprec() do
          refs = [entryref(leaf, a, v, x, i) for i in blocks[k]]
          @test within(r, maximum(first.(refs)), maximum(last.(refs)), 1, T)
        end
      end
    end
    # the full point: max(0, max scores)
    R = kkt_violation(s, v, x)
    bigprec() do
      refs = [entryref(leaf, a, v, x, i) for i in 1:n]
      ref = leaf isa GroupNormL2 ? max(big(0), maximum(first(refentry(leaf, a, v, x, first(G))) for G in leaf.groups), abs(big(v[6]))) :
                                   max(big(0), maximum(first.(refs)))
      @test within(R, ref, maximum(last.(refs)), 3, T)
    end
    # a zero point has the value of the screen of the three-argument form at each coordinate
    if !(leaf isa GroupNormL2)
      z = zeros(T, n)
      for i in 1:n
        @test kkt_violation(s, v, z, i) === kkt_violation(s, v, i)
      end
    end
  end
  # factors that merge: the product decides, in BigFloat
  leaf = NormL1(T(0.3), T[1, 2, 3, 4, 5, 6])
  x, v = T[0, 1, 0, -2, 3, 0], T[2, 1, 0.5, -3, 4, 1]
  for (α1, α2) in ((T(2), T(3)), (T(0.5), T(1e-3)), (T(1e3), T(7)))
    s = Scaled(Scaled(leaf, α1), α2)
    a = bigprec(() -> big(α1) * big(α2))
    for i in 1:6
      bigprec() do
        ref, scale = refentry(leaf, a, v[i], x[i], i)
        @test within(kkt_violation(s, v, x, i), ref, scale, 1, T)
      end
    end
  end
  # an integer factor: the type widens to Float64 (the factor 2 is inverted in Float64)
  r = kkt_violation(2 * NormL1(T(0.5)), T[1, 2], T[0, 3], 2)
  @test r isa Float64 && r ≈ 3                              # 2 NormL1(0.5) = NormL1(1): |2 + 1|
  @test kkt_violation(2 * NormL1(T(0.5)), T[1, 2], T[0, 3]) isa Float64
end

@testset "Scaled, the stated values outside the domain of the factor" begin
  # v/α overflows: the value is the +Inf of the formula, where the true score of 1e-300 * SqrNormL2(1) at (1e10, 1) is 1e10
  @test kkt_violation(Scaled(SqrNormL2(1.0), 1e-300), [1e10], [1.0], 1) === Inf
  # 1/α overflows for a subnormal α: the formula has Inf * 0 at v = 0, and the zero-point value of the base stays
  @test isnan(kkt_violation(Scaled(NormL1(1e300), 1e-310), [0.0], [0.0], 1))
  @test isnan(kkt_violation(Scaled(NormL1(1e300), 1e-310), [0.0], 1))
  # a Float32 pair
  @test kkt_violation(Scaled(SqrNormL2(1f0), 1f-30), [1f10], [1f0], 1) === Inf32
  # inside the domain, the same functions have the value of the definition
  @test kkt_violation(Scaled(SqrNormL2(1.0), 1e-30), [1e10], [1.0], 1) ≈ 1e10
  @test kkt_violation(Scaled(SqrNormL2(1.0), 1e-300), [1e-10], [1.0], 1) ≈ 1e-10
end

# ---- 8. Scaled, wrappers that stay nested ------------------------------------------------------------------------------
# fails at the base: the three-argument form around LinearFunction and a centered SqrNormL2 at v of order one; the
# nonzero points and the full-point form

@testset "$T the rows of the table: chains at v of order one" for T in TYPES
  t = tiny(T)
  # LinearFunction([1]) under t, t at v = 0.25 (true value 0.25; the base returns 1.25 in the three-argument form)
  s = chain(T, LinearFunction(T[1]))
  @test kkt_violation(s, T[0.25], 1) === T(0.25)
  for xx in (T[0], T[0.7])
    @test kkt_violation(s, T[0.25], xx, 1) === T(0.25)
    @test kkt_violation(s, T[0.25], xx) === T(0.25)
  end
  # NormL1(1) under t, t: the level does not fold to a normal number. The values of the base stay where they are defined, and a
  # nonzero point and the full-point form are refused
  s = chain(T, NormL1(T(1)))
  @test_throws ArgumentError kkt_violation(s, T[0.25], T[0.7], 1)
  @test_throws ArgumentError kkt_violation(s, T[0.25], T[0.7])
  @test_throws ArgumentError kkt_violation(s, T[0.25], T[0], )
  @test kkt_violation(s, T[0.25], 1) === T(Inf)                         # the value of the base, not the screen 0.25
  @test kkt_violation(s, T[0.25], T[0], 1) === T(Inf)
  # a GroupNormL2 under the same factors at a zero block
  s = chain(T, GroupNormL2(T(1), [[1, 2]], T[1]))
  @test kkt_violation(s, T[0.3, 0.4], T[0, 0], 1) === T(Inf)
  @test_throws ArgumentError kkt_violation(s, T[0.3, 0.4], T[0, 1], 1)
  @test_throws ArgumentError kkt_violation(s, T[0.3, 0.4], T[0, 1])
  @test_throws ArgumentError kkt_violation(chain(T, NormL2(T(1))), T[0.3, 0.4], T[0, 1])
  # the level folds: NormL1(lvl) under t, t has the level tiny^2 lvl and the score of the unscaled function
  s = chain(T, NormL1(lvl(T)))
  folded = NormL1(folded_level(T))
  @test kkt_violation(s, T[0.25], 1) === kkt_violation(folded, T[0.25], 1)
  @test kkt_violation(s, T[0.25], 1) === T(0.25)
  @test kkt_violation(s, T[0.25], T[0.5], 1) === kkt_violation(folded, T[0.25], T[0.5], 1)
  @test kkt_violation(s, T[0.25], T[0.5]) === T(0.25)
end

@testset "$T chains of every leaf against the definition" for T in TYPES
  rng = MersenneTwister(975)
  n = 6
  a = chain_product(T)
  w = T(0.5) .+ rand(rng, T, n)
  y, c = randn(rng, T, n), randn(rng, T, n)
  groups = [[1, 5, 3], [4, 2]]
  # at v of order one, and at v of the size of the folded levels and of the products of the factors with the parameters
  small = ldexp(one(T), T === Float32 ? -90 : -190)
  for (label, leaf) in (("NormL1", NormL1(lvl(T), w)), ("NormL1 unweighted", NormL1(lvl(T))),
                        ("ElasticNet", ElasticNet(lvl(T), lvl(T) / 2, w)), ("ZeroFunction", ZeroFunction()),
                        ("LinearFunction", LinearFunction(c)), ("SqrNormL2", SqrNormL2(T(0.7), y)),
                        ("SqrNormL2 uncentered", SqrNormL2(T(0.7))),
                        ("GroupNormL2", GroupNormL2(lvl(T), groups, T[1, 0.6]))),
      scale_v in (one(T), small)
    s = chain(T, leaf)
    x, v = randn(rng, T, n), scale_v .* (2 .* randn(rng, T, n))
    x[[1, 4]] .= 0
    # the three-argument screen (not for GroupNormL2), four-argument scores at each coordinate, the full point
    for i in 1:n
      bigprec() do
        ref, scale = entryref(leaf, a, v, x, i)
        r = kkt_violation(s, v, x, i)
        @test within(r, ref, scale, leaf isa GroupNormL2 ? 3 : 1, T) && typeof(r) === T
        if !(leaf isa GroupNormL2)
          ref0, scale0 = refentry(leaf, a, v[i], zero(T), i)
          @test within(kkt_violation(s, v, i), ref0, scale0, 1, T)
          @test within(kkt_violation(s, v, zeros(T, n), i), ref0, scale0, 1, T)
        end
      end
    end
    bigprec() do
      refs = [entryref(leaf, a, v, x, i) for i in 1:n]
      ref = leaf isa GroupNormL2 ? max(big(0), maximum(first(refentry(leaf, a, v, x, first(G))) for G in groups), abs(big(v[6]))) :
                                   max(big(0), maximum(first.(refs)))
      @test within(kkt_violation(s, v, x), ref, maximum(last.(refs)), 3, T)
    end
  end
  # NormL2: the full point
  v, x = T[0.3, -0.4, 0.5], T[1, 0.5, -2]
  s = chain(T, NormL2(lvl(T)))
  ref = bigprec() do
    refblock(a * big(lvl(T)), big.(v), big.(x))
  end
  @test within(kkt_violation(s, v, x), ref, norm(big.(v)) + 1, 3, T)
  @test_throws ArgumentError kkt_violation(s, v, x, 1)
  @test_throws ArgumentError kkt_violation(s, v, zeros(T, 3), 1)
  # the gradient entry is multiplied by the product, so v = -a c gives 0 where the unscaled gradient would not
  cc = T[lvl(T), -lvl(T), 2]
  sl = chain(T, LinearFunction(cc))
  vv = T[-folded_level(T), folded_level(T), 0]
  @test kkt_violation(sl, vv, 1) === zero(T) && kkt_violation(sl, vv, 2) === zero(T)
  @test kkt_violation(sl, vv, T[1, 2, 3], 1) === zero(T) && kkt_violation(sl, vv, T[1, 2, 3], 2) === zero(T)
  @test kkt_violation(sl, vv, T[1, 2, 3]) === zero(T)
  # block and symmetric points
  s = chain(T, NormL1(lvl(T), w))
  bx = BlockIterate(T[0, 1, 0, 2, 0, -1], [[1, 2], [3, 4], [5, 6]])
  v = T[0.5, 0.25, -1, 2, 0.1, 0.3]
  for k in 1:3
    G = [[1, 2], [3, 4], [5, 6]][k]
    bigprec() do
      refs = [refentry(NormL1(lvl(T), w), a, v[i], bx[i], i) for i in G]
      @test within(kkt_violation(s, v, bx, k), maximum(first.(refs)), maximum(last.(refs)), 1, T)
    end
  end
  S = SymmetricSparseIterate(T[0 1; 1 0])
  GS = T[1 2; 3 4]
  sl = chain(T, NormL1(lvl(T), T[1 2; 2 3]))
  @test kkt_violation(sl, GS, S, 2) ≈ T(2.5) atol = 4 * eps(T)             # the symmetric pair (2 + 3)/2, level tiny^2 lvl
  @test kkt_violation(sl, GS, S, 1) === kkt_violation(sl, GS, 1)
  # a chain around a function with no score: a MethodError for the call on the chain
  sqr = SqrtLeastSquaresLoss(T[1, 2, 3], T[1 0; 0 1; 1 1])
  @test_throws MethodError kkt_violation(chain(T, sqr), T[1, 2], T[0, 1], 1)
  @test_throws MethodError kkt_violation(chain(T, sqr), T[1, 2], T[0, 1])
end

# ---- 9. an infinite factor, functions without a score, functions defined outside the package -------------------------------
struct OnlyScreen <: AbstractFunction end
PB.kkt_violation(::OnlyScreen, g::AbstractArray, i::Integer) = abs(g[i]) - 1
struct NoScreen <: AbstractFunction end
# a function with a screen, four-argument methods and a full-point method of its own
struct OwnAll <: AbstractFunction end
PB.kkt_violation(::OwnAll, g::AbstractArray, i::Integer) = abs(g[i]) - 1
PB.kkt_violation(::OwnAll, g::AbstractArray, x::AbstractArray, k::Integer) = 10 * g[k] + x[k]
PB.kkt_violation(::OwnAll, g::AbstractArray, x::BlockIterate, k::Integer) = sum(g[blockindices(x, k)])
PB.kkt_violation(::OwnAll, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) = 0.0
PB.kkt_violation(::OwnAll, g::AbstractArray, x::AbstractArray) = sum(abs, g) + sum(abs, x)
# a function whose methods for the scaled function are its own
struct OwnScaled <: AbstractFunction end
PB.kkt_violation(::OwnScaled, g::AbstractArray, i::Integer) = abs(g[i])
PB.kkt_violation(s::Scaled{<:OwnScaled}, g::AbstractArray, x::AbstractArray, k::Integer) = -s.α
PB.kkt_violation(s::Scaled{<:OwnScaled}, g::AbstractArray, x::BlockIterate, k::Integer) = -3 * s.α   # as the guide says, with the one above
PB.kkt_violation(s::Scaled{<:OwnScaled}, g::AbstractArray, x::AbstractArray) = 7 * s.α

@testset "$T infinite factors, functions without a score, custom functions" for T in TYPES
  inf = T(Inf)
  # Scaled(f, Inf): the values at a zero point stay, and a nonzero point and the full point are refused
  s = Scaled(NormL1(T(1)), inf)
  @test s isa Scaled
  @test kkt_violation(s, T[3], T[0], 1) === -inf                          # the base value
  @test kkt_violation(s, T[3], 1) === -inf
  @test_throws ArgumentError kkt_violation(s, T[3], T[1], 1)
  @test_throws ArgumentError kkt_violation(s, T[3], T[0])
  @test_throws ArgumentError kkt_violation(s, T[3], T[1])
  @test occursin("infinite factor", msg(() -> kkt_violation(s, T[3], T[1], 1)))
  # the same for a group, a block and a symmetric point
  sg = Scaled(GroupNormL2(T(1), [[1, 2]], T[1]), inf)
  @test kkt_violation(sg, T[0.3, 0.4], T[0, 0], 1) === -inf
  @test_throws ArgumentError kkt_violation(sg, T[0.3, 0.4], T[0, 1], 2)
  @test_throws ArgumentError kkt_violation(sg, T[0.3, 0.4], BlockIterate(T[0, 1], [[1, 2]]), 1)
  @test kkt_violation(sg, T[0.3, 0.4], BlockIterate(T, 2, [[1, 2]]), 1) === -inf
  @test_throws ArgumentError kkt_violation(Scaled(NormL1(T(1)), inf), T[1 2; 3 4], SymmetricSparseIterate(T[0 1; 1 0]), 2)
  @test kkt_violation(Scaled(NormL1(T(1)), inf), T[1 2; 3 4], SymmetricSparseIterate(T[0 1; 1 0]), 1) === -inf
  # an entry in no group: at a zero entry the formula gives Inf * 0, as the base does; a nonzero entry is a nonzero coordinate
  sgi = Scaled(GroupNormL2(T(1), [[1, 2]], T[1]), inf)
  @test isnan(kkt_violation(sgi, T[0.3, 0.4, 2], T[0, 0, 0], 3))
  @test_throws ArgumentError kkt_violation(sgi, T[0.3, 0.4, 2], T[0, 0, 5], 3)
  @test_throws ArgumentError kkt_violation(sgi, T[0.3, 0.4, 2], T[0, 0, 5], 4 - 1)
  @test kkt_violation(sgi, T[0.3, 0.4, 2], T[0, 0, 5], 1) === -inf              # the group of entry 1 is zero
  # an unfoldable chain around a GroupNormL2: a nonzero entry in no group is a nonzero coordinate, a zero one keeps its value
  cg = chain(T, GroupNormL2(T(1), [[1, 2]], T[1]))
  @test_throws ArgumentError kkt_violation(cg, T[0.3, 0.4, 2], T[0, 0, 5], 3)
  @test occursin("Scaled wrappers around GroupNormL2", msg(() -> kkt_violation(cg, T[0.3, 0.4, 2], T[0, 0, 5], 3)))
  @test kkt_violation(cg, T[0.3, 0.4, 2], T[0, 0, 0], 3) isa T
  # a function of the package without a score: a MethodError for the call on the Scaled function
  for f in (SqrtLeastSquaresLoss(randn(T, 3), randn(T, 3, 2)),)
    for s in (Scaled(f, T(2)), Scaled(Scaled(f, T(2)), T(3)))
      e = try kkt_violation(s, zeros(T, 2), zeros(T, 2), 1); nothing catch e; e end
      @test e isa MethodError && e.f === kkt_violation && e.args[1] === s
      e = try kkt_violation(s, zeros(T, 2), zeros(T, 2)); nothing catch e; e end
      @test e isa MethodError && e.f === kkt_violation && e.args[1] === s
    end
  end
  # a function defined outside the package: its screen at a zero coordinate (scaled), an ArgumentError at a nonzero one, no
  # full-point form
  o = Scaled(OnlyScreen(), T(2))
  @test kkt_violation(o, T[3, 4], T[0, 1], 1) === T(2) * (abs(T(3) / 2) - 1)
  @test kkt_violation(o, T[3, 4], T[0, 1], 1) === kkt_violation(o, T[3, 4], 1)
  @test_throws ArgumentError kkt_violation(o, T[3, 4], T[0, 1], 2)
  @test occursin("Scaled{<:OnlyScreen}", msg(() -> kkt_violation(o, T[3, 4], T[0, 1], 2)))
  @test_throws MethodError kkt_violation(o, T[3, 4], T[0, 1])
  @test_throws MethodError kkt_violation(Scaled(NoScreen(), T(2)), T[3, 4], T[0, 1], 1)
  # the functions that Scaled(f, 0) returns have their own forms
  z = Scaled(NormL1(T(1)), zero(T))
  @test z isa ZeroFunction
  @test kkt_violation(z, T[3, -4], T[0, 1], 2) === T(4) && kkt_violation(z, T[3, -4], T[0, 1]) === T(4)
  zi = Scaled(NormL1(T(1), T[Inf, 1]), zero(T))
  @test kkt_violation(zi, T[3, -4], T[0, 1], 1) === -inf
  @test kkt_violation(zi, T[3, -4], T[0, 1], 2) === T(4)
end

# the plan's semantics of a function defined outside the package under Scaled, also when it has methods of its own
@testset "custom functions under Scaled" begin
  g, x = [3.0, -2.0, 0.5, 4.0], [0.0, 1.0, 0.0, -1.0]
  for o in (OwnAll(), OnlyScreen())
    s = Scaled(o, 2.0)
    # the scaled screen at a zero coordinate, not a method of its own; an ArgumentError at a nonzero one that says what to define
    @test kkt_violation(s, g, x, 1) === 2.0 * (abs(0.5 * 3.0) - 1)
    @test kkt_violation(s, g, x, 1) === kkt_violation(s, g, 1)
    @test_throws ArgumentError kkt_violation(s, g, x, 2)
    @test occursin("Scaled{<:$(nameof(typeof(o)))}", msg(() -> kkt_violation(s, g, x, 2)))
    @test_throws MethodError kkt_violation(s, g, x)
    b = BlockIterate(copy(x), [[1, 3], [2, 4]])
    @test kkt_violation(s, g, b, 1) === 2.0 * max(abs(0.5 * 3.0) - 1, abs(0.5 * 0.5) - 1)
    @test_throws ArgumentError kkt_violation(s, g, b, 2)
    S = SymmetricSparseIterate([0.0 1.0; 1.0 0.0])
    GS = [1.0 2.0; 3.0 4.0]
    @test kkt_violation(s, GS, S, 1) === kkt_violation(s, GS, 1)
    @test_throws ArgumentError kkt_violation(s, GS, S, 2)
    # a chain of wrappers
    c = Scaled(Scaled(o, 1e-200), 1e-200)
    @test c.f isa Scaled
    @test kkt_violation(c, g, x, 1) === kkt_violation(c, g, 1)
    @test_throws ArgumentError kkt_violation(c, g, x, 2)
    @test_throws MethodError kkt_violation(c, g, x)
  end
  # the own methods of OwnAll are used unscaled
  @test kkt_violation(OwnAll(), g, x, 2) === 10 * g[2] + x[2] && kkt_violation(OwnAll(), g, x) === sum(abs, g) + sum(abs, x)
  # the method for Scaled{<:OwnScaled} is the one that runs, in both forms
  so = Scaled(OwnScaled(), 2.0)
  @test kkt_violation(so, g, x, 1) === -2.0 && kkt_violation(so, g, x, 2) === -2.0
  @test kkt_violation(so, g, x) === 14.0
  @test kkt_violation(so, g, BlockIterate(copy(x), [[1, 3], [2, 4]]), 1) === -6.0
  @test kkt_violation(so, g, 1) === 2.0 * abs(0.5 * g[1])                 # the three-argument screen is that of the wrapper
  @test isempty(Test.detect_ambiguities(ProximalBase))
end

# an out-of-range index raises the error of the unscaled function, before any read of x that decides a refusal
@testset "$T out-of-range indices through the refusing routes" for T in TYPES
  routes(f) = (Scaled(f, T(Inf)), chain(T, f))
  pts(n) = (zeros(T, n), SparseIterate(T, n), BlockIterate(T, n, [collect(1:n)]))
  for (leaf, n) in ((NormL1(T(1)), 2), (GroupNormL2(T(1), [[1, 2]], T[1]), 2))
    for pt in pts(n), k in (0, n + 3)
      e0 = etype(() -> kkt_violation(leaf, zeros(T, n), pt, k))
      e0 === nothing && continue
      for s in routes(leaf)
        @test etype(() -> kkt_violation(s, zeros(T, n), pt, k)) === e0
      end
    end
  end
  # the same with a nonzero point elsewhere, which would be refused at an index in range
  for (leaf, n) in ((NormL1(T(1)), 2), (GroupNormL2(T(1), [[1, 2]], T[1]), 3))
    x = zeros(T, n); x[1] = 1
    for k in (0, n + 3)
      e0 = etype(() -> kkt_violation(leaf, zeros(T, n), x, k))
      @test e0 === BoundsError
      for s in routes(leaf)
        @test etype(() -> kkt_violation(s, zeros(T, n), x, k)) === BoundsError
      end
    end
  end
end

# ---- 10. mixed element types and the result type ---------------------------------------------------------------------------
# fails at the base: the Float64 result of a Float32 v with a Float64 x (a Float32 result at the base), nonzero blocks
@testset "mixed element types" begin
  groups = [[1, 2], [3, 4, 5]]
  f = GroupNormL2(0.5, groups, [1.0, 2.0])
  v32 = Float32[1, -2, 3, -4, 5]
  x64 = [0.0, 0.0, 1.0, 0.5, 0.0]
  # W = float(promote_type(eltype(v), eltype(x))): the result is a Float64 at a zero and at a nonzero block
  r0, r1 = kkt_violation(f, v32, x64, 1), kkt_violation(f, v32, x64, 3)
  @test r0 isa Float64 && r1 isa Float64
  @test r0 === PB._l2_screen(0.5, 1.0, Float64.(v32), 1:2)                 # the screen of v read in Float64
  @test blockok(r1, Float64, 0.5, 2.0, Float64.(v32[3:5]), x64[3:5])
  @test kkt_violation(f, v32, x64) isa Float64
  @test @inferred(kkt_violation(f, v32, x64, 1)) isa Float64 && @inferred(kkt_violation(f, v32, x64, 3)) isa Float64
  @test @inferred(kkt_violation(f, v32, x64)) isa Float64
  # a Float64 v with a Float32 x, an integer v, a Float32 gradient with the default BlockIterate of Float64
  @test kkt_violation(f, Float64.(v32), Float32.(x64), 3) isa Float64
  @test kkt_violation(f, Int[1, -2, 3, -4, 5], x64, 3) isa Float64 && kkt_violation(f, Int[1, -2, 3, -4, 5], x64, 1) isa Float64
  @test kkt_violation(f, Int[1, -2, 3, -4, 5], x64, 3) ≈ kkt_violation(f, Float64.([1, -2, 3, -4, 5]), x64, 3)
  bx = BlockIterate(5, f)
  bx[3] = 1.0
  @test bx isa BlockIterate{Float64}
  @test kkt_violation(f, v32, bx, 1) isa Float64 && kkt_violation(f, v32, bx, 2) isa Float64
  @test kkt_violation(f, v32, bx) isa Float64
  # an entry in no group
  f1 = GroupNormL2(0.5, [[1, 2]], [1.0])
  @test kkt_violation(f1, Float32[1, 2, 3], zeros(3), 3) === 3.0
  # Float32 level and weights with Float64 data: the coefficient is formed in Float64
  f32 = GroupNormL2(0.5f0, groups, Float32[1, 2])
  @test kkt_violation(f32, Float64.(v32), x64, 3) isa Float64
  # NormL2 and Scaled
  @test kkt_violation(NormL2(0.5), v32, x64) isa Float64
  @test kkt_violation(Scaled(f, 2.0), v32, x64, 3) isa Float64 && kkt_violation(Scaled(f, 2.0), v32, x64) isa Float64
  @test kkt_violation(Scaled(f, 2f0), v32, x64, 3) isa Float64
  @test kkt_violation(Scaled(f32, 2f0), v32, Float32.(x64), 3) isa Float32
  @test kkt_violation(Scaled(NormL1(0.5f0), 2f0), v32, Float32.(x64), 3) isa Float32
end

# ---- 11. inference, allocations ---------------------------------------------------------------------------------------------
@testset "$T inference and allocations" for T in TYPES
  rng = MersenneTwister(976)
  n = 120
  groups = [collect(i:6:n) for i in 1:6]                            # six interleaved groups of twenty
  f = GroupNormL2(T(0.3), groups, T[1, 2, 0.5, 1, 3, 1])
  fg = GroupNormL2(T(0.3), [collect(1:50), collect(60:100)], T[1, 2])   # entries 51:59 and 101:120 are in no group
  x = randn(rng, T, n)
  x[groups[2]] .= 0
  v = randn(rng, T, n)
  sx = SparseIterate(x)
  bx = BlockIterate(copy(x), groups)
  nl = NormL2(T(0.3))
  α = T(2.5)
  sf = Scaled(f, α)
  for (h, pt) in ((f, x), (fg, x), (sf, x), (Scaled(fg, α), x))
    @inferred kkt_violation(h, v, pt)
    @inferred kkt_violation(h, v, pt, 3)
    @inferred kkt_violation(h, v, pt, 2)
    @test allocs(kkt_violation, h, v, pt) == 0
    @test allocs(kkt_violation, h, v, pt, 3) == 0
    @test allocs(kkt_violation, h, v, pt, 2) == 0                     # a zero group
  end
  @inferred kkt_violation(nl, v, x)
  @test allocs(kkt_violation, nl, v, x) == 0
  @test allocs(kkt_violation, nl, v, zeros(T, n)) == 0
  @test allocs(kkt_violation, Scaled(nl, α), v, x) == 0
  @inferred kkt_violation(Scaled(nl, α), v, x)
  for (h, pt) in ((f, sx), (f, bx), (sf, bx))
    @inferred kkt_violation(h, v, pt, 1)
    @inferred kkt_violation(h, v, pt, 2)
  end
  @inferred kkt_violation(f, v, sx)
  @inferred kkt_violation(f, v, bx)
  @inferred kkt_violation(sf, v, bx)
  # nested chains: the leaf's own methods and the folded levels
  for h in (chain(T, NormL1(lvl(T))), chain(T, LinearFunction(randn(rng, T, n))), chain(T, GroupNormL2(lvl(T), groups, T[1, 2, 0.5, 1, 3, 1])))
    @test kkt_violation(h, v, x, 3) isa T
    @test kkt_violation(h, v, x) isa T
  end
  # the argument is not changed
  v0, x0 = copy(v), copy(x)
  kkt_violation(f, v, x); kkt_violation(sf, v, x); kkt_violation(nl, v, x); kkt_violation(f, v, x, 2); kkt_violation(f, v, x, 3)
  @test v == v0 && x == x0
  # the kernels of the block run on a view
  @test close(kkt_violation(f, view(v, 1:n), view(x, 1:n)), kkt_violation(f, v, x), T, norm(v) + 1)
  @test close(kkt_violation(nl, view(v, 1:n), view(x, 1:n)), kkt_violation(nl, v, x), T, norm(v) + 1)
end

# the cold paths of the block kernel (squares that overflow or underflow, c near floatmax, c = Inf, NaN, an infinite v at a zero
# block) allocate nothing after warm-up, and the class of a Scaled is folded by inference
@testset "$T cold paths: allocations and class" for T in TYPES
  emax, emin = exponent(floatmax(T)), exponent(floatmin(T))
  n = 8
  f = GroupNormL2(T(0.3), [collect(1:n)], T[1])
  cases = [(ldexp(T(1), emax ÷ 2 + 2), ldexp(T(1), emax ÷ 2 + 2), f), (ldexp(T(1), emin ÷ 2 - 3), ldexp(T(1), emin ÷ 2 - 3), f),
           (T(1), ldexp(T(1), emin ÷ 2 - 3), f), (T(-1), ldexp(T(1), emax ÷ 2 + 2), GroupNormL2(floatmax(T), [collect(1:n)], T[1])),
           (T(1), T(1), GroupNormL2(T(Inf), [collect(1:n)], T[1])), (T(NaN), T(1), f), (T(Inf), T(0), f)]
  for (sv, sx, g) in cases
    v, x = fill(sv, n), fill(sx, n)
    @test allocs(kkt_violation, g, v, x, 1) == 0
    @test allocs(kkt_violation, g, v, x) == 0
    @test allocs(kkt_violation, NormL2(T(0.3)), v, x) == 0
    @test allocs(kkt_violation, Scaled(g, T(2)), v, x, 1) == 0
  end
  sf = Scaled(f, T(2))
  @test @inferred(PB._kkt_class(typeof(sf))) isa PB._KktScore
  @test @inferred(PB._kkt_class(typeof(Scaled(NormLinf(T(1)), T(2))))) isa PB._KktScore
  @test @inferred(PB._kkt_class(typeof(Scaled(SqrtLeastSquaresLoss(T[1, 2], T[1 0; 0 1]), T(2))))) isa PB._KktNone
  @test @inferred(PB._kkt_class(typeof(Scaled(OwnAll(), T(2))))) isa PB._KktCustom
end

# ---- entries that are converted before they are classified or measured -----------------------------------------------------
# fails at the base: the nonzero blocks; the entries in no group with integer extremes fail since stage A of the function
@testset "entries are converted to the working type before classification and magnitude" begin
  # the magnitude of an entry in no group is taken after the conversion: abs(typemin(Int)) overflows in Int
  imin = abs(Float64(typemin(Int)))                          # 2^63 where Int is Int64
  f = GroupNormL2(1.0, [[2]], [1.0])
  @test kkt_violation(f, [typemin(Int), 0], [0, 0], 1) === imin
  @test kkt_violation(f, [typemin(Int), 0], [0, 0]) === imin
  @test kkt_violation(f, Int8[typemin(Int8), 0], Int8[0, 0], 1) === 128.0
  @test kkt_violation(f, Int8[typemin(Int8), 0], Int8[0, 0]) === 128.0
  ft = GroupNormL2(1.0, [[1]], [1.0])                       # the entry 2 is past the groups
  @test kkt_violation(ft, [0, typemin(Int)], [0, 0], 2) === imin
  @test kkt_violation(ft, [0, typemin(Int)], [0, 0]) === imin
  @test kkt_violation(f, UInt8[255, 0], UInt8[0, 0], 1) === 255.0 && kkt_violation(f, UInt8[255, 0], UInt8[0, 0]) === 255.0
  @test kkt_violation(f, [true, false], [false, false], 1) === 1.0 && kkt_violation(f, [true, false], [false, false]) === 1.0
  # the same through Scaled (v/α is a lazy Float64 array, so the entries are converted there)
  @test kkt_violation(Scaled(f, 2.0), [typemin(Int), 0], [0, 0], 1) === imin
  # an entry that does not fit the working type is infinite there, on the ordinary and on the cold path
  big128 = typemax(UInt128)
  @test kkt_violation(NormL2(1.0), Float32[0], UInt128[big128]) === Inf32
  @test kkt_violation(GroupNormL2(1.0, [[1]], [1.0]), Float32[0], UInt128[big128], 1) === Inf32
  @test kkt_violation(GroupNormL2(1.0, [[1]], [1.0]), Float32[0], UInt128[big128]) === Inf32
  @test kkt_violation(NormL2(1.0), Float16[0], [10^6]) === Inf16
  @test kkt_violation(GroupNormL2(1.0, [[1]], [1.0]), Float16[0], [10^6], 1) === Inf16
  @test kkt_violation(NormL2(1.0), Float32[Inf], UInt128[0]) === Inf32          # a zero block with an infinite v
  @test isnan(kkt_violation(NormL2(1.0), Float32[NaN], UInt128[big128]))        # NaN takes precedence over the infinity
  # a point of Rational or BigInt entries that converts to finite values of the working type has its documented score
  # Float64 gradient with a Rational{BigInt} point: the working type is BigFloat, and the squares (about 1e-600) are ordinary
  # there, so this is the ordinary path; the reference is built from the entries converted to BigFloat
  @test float(promote_type(Float64, Rational{BigInt})) === BigFloat
  xr = Rational{BigInt}[1 // big(10)^300, 2 // big(10)^300]
  xrw = BigFloat.(xr)
  vr = [0.3e-300, -0.4e-300]
  u = big(eps(BigFloat)) / 2                               # the unit roundoff of the working type at the default precision
  for (λ, w) in ((0.37, 1.0), (1e-300, 1.0))
    for r in (kkt_violation(GroupNormL2(λ, [[1, 2]], [w]), vr, xr, 1), kkt_violation(NormL2(λ), vr, xr))
      @test r isa BigFloat
      ok = bigprec() do
        c = big(λ) * big(w)
        ref = refblock(c, big.(vr), big.(xrw))
        abs(big(r) - ref) <= (2 + 8) * u * (sqrt(sum(abs2, big.(vr))) + c)
      end
      @test ok
    end
  end
  # Float32 gradient with a Rational{Int128} point of denominator 10^30: the working type is Float32 and the converted
  # entries (about 1e-30) have squares that underflow in Float32, so the sum of squares fails the test of the ordinary path
  # and the score is computed by the cold path
  @test float(promote_type(Float32, Rational{Int128})) === Float32
  xi = Rational{Int128}[1 // Int128(10)^30, 2 // Int128(10)^30]
  xiw = Float32.(xi)
  vi = Float32[0.3f-30, -0.4f-30]
  sxi = PB._indexed_sumabs2(xi, [1, 2], Float32)
  @test !(isfinite(sxi) & PB._squares_normal(sxi, 2, Float32))                    # the ordinary-path test fails on these entries
  for (λ, w) in ((0.37, 1.0), (1e-30, 1.0))
    r1 = kkt_violation(GroupNormL2(λ, [[1, 2]], [w]), vi, xi, 1)
    r2 = kkt_violation(NormL2(λ), vi, xi)
    @test r1 isa Float32 && r2 isa Float32
    @test blockok(r1, Float32, λ, w, vi, xiw)
    @test blockok(r2, Float32, λ, 1.0, vi, xiw)
  end
  xq = Rational{Int}[3, 4]
  @test kkt_violation(GroupNormL2(1.0, [[1, 2]], [1.0]), [-0.6, -0.8], xq, 1) <= 1e-15
  xb = BigInt[3, 4]                                       # the working type is BigFloat
  @test kkt_violation(NormL2(1.0), [-0.6, -0.8], xb) isa BigFloat
  @test abs(kkt_violation(GroupNormL2(1.0, [[1, 2]], [1.0]), [-0.6, -0.8], xb, 1)) <= 1e-15
end

# A stack that holds an infinite factor under or over another wrapper is built by a direct call of the parameterized
# constructor Scaled{F, T}(f, α): Scaled(f, α) and α * f merge an infinite factor with the wrapper next to it (first testset). For such a
# stack the four-argument form at a nonzero coordinate or group and the full-point form throw an ArgumentError (second
# testset). Its three-argument and zero-point values are outside the contract (the scaling guide) and are not tested.
@testset "$T the constructor merges an infinite factor" for T in TYPES
  t, inf = tiny(T), T(Inf)
  leaves = (ZeroFunction(), LinearFunction(T[1]), NormL1(T(1)), GroupNormL2(T(1), [[1]], T[1]), NormL1(zero(T)))
  for leaf in leaves
    stacks = (Scaled(Scaled(leaf, t), inf), Scaled(Scaled(leaf, inf), t), Scaled(chain(T, leaf), inf), inf * (t * (t * leaf)))
    for s in stacks
      @test s isa Scaled && !(s.f isa Scaled) && isinf(s.α)
    end
  end
end

@testset "$T stacks with an infinite factor are refused at nonzero points and in the full-point form" for T in TYPES
  raw(f, a) = Scaled{typeof(f), typeof(a)}(f, a)
  t, inf = tiny(T), T(Inf)
  leaves = (ZeroFunction(), LinearFunction(T[1]), NormL1(T(1)), GroupNormL2(T(1), [[1]], T[1]), NormL1(zero(T)))
  stacks(leaf) = (raw(raw(Scaled(leaf, inf), t), t),       # the infinite factor innermost, under a retained chain
                  raw(raw(Scaled(leaf, t), inf), t),       # in the middle
                  raw(chain(T, leaf), inf))                # outermost, around a retained chain
  for leaf in leaves
    for (depth, s) in enumerate(stacks(leaf))
      @test s.f isa Scaled
      depth == 1 && @test PB._chained(s)
      depth == 3 && @test !PB._chained(s)
      depth == 1 && @test occursin("infinite factor", msg(() -> kkt_violation(s, T[0.25], T[1], 1)))
      @test_throws ArgumentError kkt_violation(s, T[0.25], T[0])               # the full-point form, also at a zero point
      for pt in (T[1], SparseIterate(T[1]), BlockIterate(T[1], [[1]]))
        @test_throws ArgumentError kkt_violation(s, T[0.25], pt, 1)            # a nonzero coordinate or group
      end
      for pt in (T[1], SparseIterate(T[1]), BlockIterate(T[1], [[1]]))
        @test_throws ArgumentError kkt_violation(s, T[0.25], pt)
      end
    end
  end
  # a SymmetricSparseIterate at a nonzero coordinate, for the leaves that have a symmetric form
  xs = SymmetricSparseIterate(T, 2)
  xs[2, 1] = 1
  G = ones(T, 2, 2)
  for leaf in (ZeroFunction(), NormL1(T(1)), LinearFunction(ones(T, 2, 2)))
    for s in stacks(leaf)
      @test_throws ArgumentError kkt_violation(s, G, xs, 2)
    end
  end
  # a chain with a zero-level leaf and finite factors is scored by the folded zero level
  z = chain(T, NormL1(zero(T)))
  @test kkt_violation(z, T[0.25], T[0.5], 1) === T(0.25) && kkt_violation(z, T[0.25], T[0.5]) === T(0.25)
end

# the helper of the block tests rejects values that are wrong
@testset "blockok rejects wrong values" begin
  for W in TYPES
    # a zero block with an overflowing coefficient: the reference -1.5 floatmax is below the range, the value is -Inf
    args = (W, floatmax(W), W(1.5), W[0], W[0])
    @test blockok(-W(Inf), args...)
    for bad in (W(NaN), zero(W), W(Inf), -floatmax(W), one(W))
      @test !blockok(bad, args...)
    end
    # a nonzero block with an overflowing reference: +Inf is the right value and the others are not
    big_args = (W, floatmax(W), W(1.5), W[1], W[1])
    @test blockok(W(Inf), big_args...) && !blockok(W(NaN), big_args...) && !blockok(floatmax(W), big_args...) && !blockok(-W(Inf), big_args...)
    # a bounded block: the value within the bound is accepted, and a value off by more than it is not
    v, x = W[0.3, -0.4], W[1, 2]
    ref = bigprec() do
      refblock(big(W(0.37)) * big(W(0.61)), big.(v), big.(x))
    end
    r = W(ref)
    @test blockok(r, W, W(0.37), W(0.61), v, x)
    for bad in (W(NaN), zero(W), W(Inf), -W(Inf), r + 100 * eps(W), r - 100 * eps(W))
      @test !blockok(bad, W, W(0.37), W(0.61), v, x)
    end
  end
end

# ---- 12. a malformed parameter at the selected coordinate of a SymmetricSparseIterate under Scaled -------------------------
# passes at the base for one wrapper (the check runs in the public method before the Scaled method); it is a regression test
@testset "$T malformed parameters of a symmetric model under Scaled" for T in TYPES
  p = 3
  x = SymmetricSparseIterate(T, p)
  x[2, 1] = 1
  x[3, 3] = -2
  G = randn(MersenneTwister(977), T, p, p)
  bad = (NormL1(T(0.5), T[]), NormL1(T(0.5), T[1, 2, 3]), ElasticNet(T(0.5), T(1), T[1, 2, 3]), LinearFunction(T[1, 2, 3]),
         SqrNormL2(T(1), T[]), SqrNormL2(T(1), T[1, 2, 3]))
  t = tiny(T)
  for leaf in bad
    for k in (1, 2, 6)                                   # coordinate 1 is zero; 2 and 6 are nonzero
      e0 = etype(() -> kkt_violation(leaf, G, x, k))
      @test e0 !== ArgumentError && e0 !== MethodError
      for s in (Scaled(leaf, T(2)), Scaled(Scaled(leaf, T(2)), T(3)), chain(T, leaf))
        @test etype(() -> kkt_violation(s, G, x, k)) === e0
      end
    end
    # the last coordinate reads the parameter at the entry 9, past the end of the short ones: an error of its own
    @test etype(() -> kkt_violation(leaf, G, x, 6)) !== nothing
  end
  # a well-formed model does not throw under the same wrappers
  good = LinearFunction(zeros(T, p, p) .+ T(1))
  for s in (Scaled(good, T(2)), chain(T, good))
    @test kkt_violation(s, G, x, 2) isa T && kkt_violation(s, G, x, 1) isa T
  end
end

end
