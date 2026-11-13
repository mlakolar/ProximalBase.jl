module GroupNormL2Test
# GroupNormL2 (#900): values, array and compact block proximal operators, KKT screens and
# BlockIterate interoperability, against independent BigFloat evaluations of the formulas
#   g(x) = λ Σₖ wₖ ‖x[Gₖ]‖₂,   prox_{γgₖ}(u) = max(1 - γλwₖ/‖u‖₂, 0) u,
# in Float32 and Float64, with the maintainer decisions D1–D3 of #900 (infinite level at a zero step,
# invalid steps, ±Inf inputs with an overflowing finite threshold).
using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils

const PREC = 512

# ---- independent references (BigFloat, from the formulas only) ----
bignorm(u) = setprecision(() -> sqrt(sum(big.(u) .^ 2; init = big(0))), BigFloat, PREC)
function refprox(u, γ, λ, w)
  setprecision(BigFloat, PREC) do
    U = big.(u)
    n = bignorm(u)
    c = big(γ) * big(λ) * big(w)
    n > c ? (1 - c / n) .* U : zero.(U)
  end
end
refvalue(x, λ, groups, w) =
  setprecision(() -> sum((big(λ) * big(w[k]) * bignorm(x[groups[k]]) for k in eachindex(groups)); init = big(0)), BigFloat, PREC)

# Componentwise error within k eps max|u|, the accuracy of a backward stable shrinkage (1 - c/n) u,
# plus two subnormal spacings for results in the subnormal range.
function close_prox(y, ref, u; k = 8)
  T = eltype(y)
  m = big(maximum(abs, u; init = zero(T)))
  bound = k * big(eps(T)) * m + 2 * big(nextfloat(zero(T)))
  all(i -> abs(big(y[i]) - ref[i]) <= bound, eachindex(y))
end
# A value within k eps of the reference (two subnormal spacings below floatmin); a reference beyond
# floatmax(T) by more than rounding must give Inf
function close_value(v, ref; k = 8)
  T = typeof(v)
  abs(ref) > big(floatmax(T)) * (1 + big(eps(T))) && return v == copysign(T(Inf), ref)
  abs(big(v) - ref) <= k * big(eps(T)) * abs(ref) + 2 * big(nextfloat(zero(T)))
end

# the dense result of prox! from the compact references, entries in no group copied
function refprox_full(x, γ, λ, groups, w)
  y = big.(x)
  for k in eachindex(groups)
    y[groups[k]] = refprox(x[groups[k]], γ, λ, w[k])
  end
  y
end

@testset "construction and validation" begin
  for T in (Float32, Float64)
    f = GroupNormL2(T(0.5), [1:2, [3, 5, 7]])
    @test f isa GroupNormL2{T}
    @test f.w == T[sqrt(T(2)), sqrt(T(3))]             # historical default √|Gₖ|
    @test GroupNormL2(T(0.5), [1:2], T[3]).w == T[3]
    # owned copies: changing the caller's arrays leaves f unchanged
    w = T[1, 2]; G = [[2, 1], [4, 3]]
    g = GroupNormL2(one(T), G, w)
    w[1] = 5; G[1][1] = 9
    @test g.w == T[1, 2] && collect(g.groups[1]) == [2, 1]
    @test isproximable(g) && isconvex(g) && !issmooth(g) && !isseparable(g) && !iszeropreserving(g)
  end
  @test GroupNormL2(1, [1:2]) isa GroupNormL2{Float64}
  @test_throws DimensionMismatch GroupNormL2(1.0, [1:2, 3:4], [1.0])
  @test_throws ArgumentError GroupNormL2(-1.0, [1:2])
  @test_throws ArgumentError GroupNormL2(NaN, [1:2])
  @test_throws ArgumentError GroupNormL2(1.0, [1:2], [-1.0])
  @test_throws ArgumentError GroupNormL2(1.0, [1:2], [NaN])
  @test_throws ArgumentError GroupNormL2(1.0, [Int[]])
  @test_throws ArgumentError GroupNormL2(1.0, [[0, 1]])
  @test_throws ArgumentError GroupNormL2(1.0, [[1, 2], [2, 3]])   # overlapping
  @test_throws ArgumentError GroupNormL2(1.0, [[1, 1]])           # repeated
  @test_throws ArgumentError GroupNormL2(1.0, [1:2], [1.0 + 0im])
end

@testset "analytical group soft thresholding ($T)" for T in (Float32, Float64)
  f = GroupNormL2(one(T), [1:2], T[1])
  @test prox_block(f, T[3, 4], 1, one(T)) ≈ T[2.4, 3.2]          # (1 - 1/5) [3, 4]
  @test prox_block(f, T[3, 4], 1, T(5)) == T[0, 0]               # norm exactly at the threshold
  @test prox_block(f, T[3, 4], 1, T(6)) == T[0, 0]
  @test prox_block(f, T[3, 4], 1, nextfloat(T(5))) == T[0, 0]
  y = prox_block(f, T[3, 4], 1, prevfloat(T(5)))                 # just below: tiny nonzero
  @test all(>(0), y) && close_prox(y, refprox(T[3, 4], prevfloat(T(5)), 1, 1), T[3, 4])
  @test prox_block(f, T[-6, 8], 1, T(5)) == T[-3, 4]             # (1 - 5/10)
  @test f(T[3, 4]) == 5
  @test GroupNormL2(T(2), [[2], [1, 3]], T[1, 3])(T[3, 1, 4]) == 2 * (1 + 3 * 5)
end

@testset "singleton groups agree with weighted NormL1 ($T)" for T in (Float32, Float64)
  rng = Xoshiro(900)
  n = 40
  x = T(3) .* randn(rng, T, n)
  w = T(0.5) .+ rand(rng, T, n)
  w[1:7:end] .= 0                                   # unpenalized entries
  singles = [[i] for i in 1:n]
  for (λ, γ) in ((T(0.7), T(1.3)), (T(0.7), zero(T)), (zero(T), T(1.3)))
    g, l = GroupNormL2(λ, singles, w), NormL1(λ, w)
    @test isapprox(prox(g, x, γ), prox(l, x, γ); rtol = 0, atol = 4eps(T) * maximum(abs, x))
    @test all(i -> isapprox(prox_block(g, x[i:i], i, γ)[1], prox_block(l, x[i], i, γ); atol = 4eps(T) * abs(x[i])), 1:n)
    @test g(x) ≈ l(x) rtol = 4eps(T)
    @test GroupNormL2(λ, singles)(x) ≈ NormL1(λ)(x) rtol = 4eps(T)   # default weights √1 = 1
  end
  # infinite weights fix finite trials at zero at every level and step, as for NormL1 (#853)
  wi = copy(w); wi[2:5:end] .= T(Inf)
  for λ in (zero(T), T(0.7)), γ in (zero(T), T(1.3))
    pg, pl = prox(GroupNormL2(λ, singles, wi), x, γ), prox(NormL1(λ, wi), x, γ)
    @test all(i -> pg[i] === pl[i] === zero(T), findall(isinf, wi))
    @test isapprox(pg, pl; rtol = 0, atol = 4eps(T) * maximum(abs, x))
  end
  # D1: an infinite level projects positive-weight entries to +0 at every nonnegative step, γ = 0 included
  # (#665), also at NaN and ±Inf, and keeps zero-weight entries unchanged
  xe = copy(x); xe[3] = NaN; xe[4] = T(Inf); xe[8] = T(-Inf)
  for γ in (zero(T), T(1.3))
    pg, pl = prox(GroupNormL2(T(Inf), singles, w), xe, γ), prox(NormL1(T(Inf), w), xe, γ)
    @test isequal(pg, pl)
    @test all(i -> iszero(w[i]) ? isequal(pg[i], xe[i]) : (pg[i] === zero(T)), 1:n)
    @test all(i -> isequal(prox_block(GroupNormL2(T(Inf), singles, w), xe[i:i], i, γ)[1],
                           prox_block(NormL1(T(Inf), w), xe[i], i, γ)), 1:n)
  end
end

@testset "multiple groups, weights, layouts and initialization ($T)" for T in (Float32, Float64)
  rng = Xoshiro(901)
  # noncontiguous, arbitrarily ordered groups; entries 4 and 11 are in no group
  groups = [[9, 1, 5], [2, 10], [7], [12, 3, 6, 8]]
  w = T[0.4, 2.5, 1.0, 0.8]
  λ, γ = T(0.6), T(0.9)
  f = GroupNormL2(λ, groups, w)
  for trial in 1:20
    x = T(2) .* randn(rng, T, 12)
    trial % 4 == 0 && (x[groups[2]] .= 0)                    # a zero group
    trial % 5 == 0 && (x[groups[4]] .*= T(0.05))             # a group below its threshold
    ref = refprox_full(x, γ, λ, groups, w)
    for init in (zeros(T, 12), fill(T(NaN), 12), copy(x))      # every output entry is written
      y = init
      @test prox!(y, f, x, γ) === y
      @test close_prox(y, ref, x)
    end
    y = prox(f, x, γ)
    @test y[[4, 11]] == x[[4, 11]]
    for k in eachindex(groups)
      out = fill(T(NaN), length(groups[k]))
      @test prox_block!(out, f, x[groups[k]], k, γ) === out
      # the same formula as the array prox; the sums of squares may associate differently (SIMD)
      @test isapprox(out, y[groups[k]]; rtol = 0, atol = 4eps(T) * maximum(abs, x))
    end
    @test close_value(f(x), refvalue(x, λ, groups, w))
    z = similar(x)
    v = value_and_prox!(z, f, x, γ)
    @test z == y
    @test close_value(v, refvalue(z, λ, groups, w)) && isapprox(v, f(z); rtol = 4eps(T))
  end
  @test prox(f, zeros(T, 12), γ) == zeros(T, 12)
  @test all(isequal(zero(T)), prox(f, fill(-zero(T), 12), γ)[vcat(groups...)])   # +0 for a zero group
  @test f(zeros(T, 12)) === zero(T)
  # matrix arguments: groups of linear indices, such as the rows of a 3 × 2 coefficient matrix
  rows = GroupNormL2(λ, [i:3:6 for i in 1:3])
  X = T(2) .* randn(rng, T, 3, 2)
  Y = prox(rows, X, γ)
  @test size(Y) == (3, 2)
  @test all(i -> close_prox(Y[i, :], refprox(X[i, :], γ, λ, sqrt(T(2))), X), 1:3)
end

@testset "in-place and aliasing ($T)" for T in (Float32, Float64)
  f = GroupNormL2(T(0.5), [[3, 1], [2, 4]], T[1, 2])
  x = T[1, -2, 3, 4]
  y = prox(f, x, one(T))
  z = copy(x)
  @test prox!(z, f, z, one(T)) === z && z == y
  z = copy(x)
  @test value_and_prox!(z, f, z, one(T)) ≈ f(y) && z == y
  u = T[3, 1]
  out = prox_block(f, u, 1, one(T))
  @test prox_block!(u, f, u, 1, one(T)) === u && u == out       # exact aliasing
  buf = T[3, 1, 7]
  v = view(buf, 1:2)
  @test prox_block!(v, f, view(buf, 1:2), 1, one(T)) == out     # an equal view is the same buffer
  @test buf == T[out; 7]
  # other overlap is rejected before any write
  buf = T[3, 1, 7]
  @test_throws ArgumentError prox_block!(view(buf, 2:3), f, view(buf, 1:2), 1, one(T))
  @test buf == T[3, 1, 7]
  @test allocs(prox_block!, out, f, u, 1, one(T)) == 0
  @test allocs(prox!, z, f, x, one(T)) == 0
  @test allocs(value_and_prox!, z, f, x, one(T)) == 0
  @test allocs(f, x) == 0
end

@testset "invalid calls fail before writes ($T)" for T in (Float32, Float64)
  f = GroupNormL2(T(0.5), [[3, 1], [2, 4]], T[1, 2])
  x = T[1, -2, 3, 4]
  sentinel() = fill(T(7), 4)
  # D2: negative and NaN steps in all three proximal interfaces; zero and infinite steps are accepted
  for γ in (-one(T), T(NaN), -T(Inf), -floatmin(T))
    y = sentinel()
    @test_throws ArgumentError prox!(y, f, x, γ)
    @test y == sentinel()
    @test_throws ArgumentError value_and_prox!(y, f, x, γ)
    @test y == sentinel()
    z = copy(x)
    @test_throws ArgumentError prox!(z, f, z, γ)
    @test z == x
    out = fill(T(7), 2)
    @test_throws ArgumentError prox_block!(out, f, T[1, 2], 1, γ)
    @test out == T[7, 7]
    u = T[1, 2]
    @test_throws ArgumentError prox_block!(u, f, u, 1, γ)
    @test u == T[1, 2]
    @test_throws ArgumentError prox(f, x, γ)
    @test_throws ArgumentError prox_block(f, T[1, 2], 1, γ)
  end
  @test prox(f, x, -zero(T)) == x            # -0.0 is a zero step: the identity
  @test prox(f, x, T(Inf)) == zeros(T, 4)
  # shapes, extents, group ids and buffers
  y = sentinel()
  @test_throws DimensionMismatch prox!(y, f, T[1, 2, 3], one(T))
  @test_throws DimensionMismatch prox!(zeros(T, 3), f, T[1, 2, 3], one(T))      # groups reach index 4
  @test_throws DimensionMismatch value_and_prox!(zeros(T, 3), f, T[1, 2, 3], one(T))
  @test_throws DimensionMismatch f(T[1, 2, 3])
  @test_throws ArgumentError prox!(zeros(Complex{T}, 4), f, x, one(T))
  @test_throws ArgumentError f(Complex{T}[1, 2, 3, 4])
  @test y == sentinel()
  out = fill(T(7), 2)
  for k in (0, 3, -1, true)
    @test_throws BoundsError prox_block!(out, f, T[1, 2], k, one(T))
  end
  @test_throws DimensionMismatch prox_block!(out, f, T[1, 2, 3], 1, one(T))
  @test_throws DimensionMismatch prox_block!(fill(T(7), 3), f, T[1, 2], 1, one(T))
  @test_throws ArgumentError prox_block!(Complex{T}[7, 7], f, T[1, 2], 1, one(T))
  @test out == T[7, 7]
  # traits do not create symmetric-iterate or scalar-entry methods
  @test_throws MethodError prox_block(f, one(T), 1, one(T))
  # no proximal operator for a symmetric iterate (#900 NR2): a clear error, nothing written
  S = SymmetricSparseIterate(T[1 2; 2 1])
  S2 = SymmetricSparseIterate(T[5 6; 6 7])
  g2 = GroupNormL2(T(0.5), [1:2, 3:4])
  @test_throws ArgumentError prox!(S2, g2, S, one(T))
  @test_throws ArgumentError value_and_prox!(S2, g2, S, one(T))
  @test Matrix(S2) == T[5 6; 6 7]
  D = fill(T(7), 2, 2)
  @test_throws ArgumentError prox!(D, g2, S, one(T))
  @test_throws ArgumentError value_and_prox!(D, g2, S, one(T))
  @test D == fill(T(7), 2, 2)
  @test_throws ArgumentError prox(g2, S, one(T))
end

@testset "exceptional parameters and values ($T)" for T in (Float32, Float64)
  G = [1:2]
  pb(f, u, γ) = prox_block(f, u, 1, γ)
  one_ = one(T)
  # zero weight: unpenalized, NaN and ±Inf included, at every level and step
  for λ in (zero(T), one_, T(Inf)), γ in (zero(T), one_, T(Inf))
    @test isequal(pb(GroupNormL2(λ, G, T[0]), T[NaN, 4], γ), T[NaN, 4])
    @test GroupNormL2(λ, G, T[0])(T[NaN, Inf]) === zero(T)
  end
  # infinite weight: fixed at +0 at every level and step, also for NaN and ±Inf (E4 before E2)
  for λ in (zero(T), one_), γ in (zero(T), one_, T(Inf)), u in (T[3, 4], T[NaN, 1], T[Inf, 1])
    @test all(isequal(zero(T)), pb(GroupNormL2(λ, G, T[Inf]), u, γ))
  end
  # D1: infinite level with a positive weight, every nonnegative step
  for γ in (zero(T), one_, T(Inf)), u in (T[3, 4], T[NaN, 1], T[-Inf, 1], T[-0.0, 0])
    @test all(isequal(zero(T)), pb(GroupNormL2(T(Inf), G, T[1]), u, γ))
  end
  f = GroupNormL2(T(Inf), [[1, 2], [3]], T[1, 0])
  @test isequal(prox(f, T[3, 4, NaN], zero(T)), T[0, 0, NaN])  # zero-weight group preserved
  # zero level or zero step with finite parameters: the identity
  @test isequal(pb(GroupNormL2(zero(T), G, T[1]), T[NaN, 1], one_), T[NaN, 1])
  @test isequal(pb(GroupNormL2(one_, G, T[1]), T[NaN, 1], zero(T)), T[NaN, 1])
  @test pb(GroupNormL2(zero(T), G, T[1]), T[3, 4], T(Inf)) == T[3, 4]
  # infinite step with a positive finite level: fixed at +0
  @test all(isequal(zero(T)), pb(GroupNormL2(one_, G, T[1]), T[Inf, 1], T(Inf)))
  # positive finite parameters: NaN block, ±Inf unchanged, zero group +0
  @test all(isnan, pb(GroupNormL2(one_, G, T[1]), T[NaN, 1], one_))
  @test pb(GroupNormL2(one_, G, T[1]), T[Inf, 1], one_) == T[Inf, 1]
  @test pb(GroupNormL2(one_, G, T[1]), T[-Inf, Inf], one_) == T[-Inf, Inf]
  # D3: the threshold of finite factors is finite also when its floating-point product overflows
  big_ = floatmax(T) / 4
  for (λ, w, γ) in ((big_, T(8), one_), (one_, T(8), big_), (big_, one_, T(8)))
    f = GroupNormL2(λ, G, T[w])
    @test pb(f, T[Inf, 1], γ) == T[Inf, 1]
    @test pb(f, T[1, -Inf], γ) == T[1, -Inf]
    @test pb(f, T[3, 4], γ) == T[0, 0]            # finite inputs: far below the threshold
    @test all(isnan, pb(f, T[NaN, 1], γ))
  end
  # values
  @test GroupNormL2(T(Inf), G)(T[0, 0]) === zero(T)
  @test GroupNormL2(T(Inf), G)(T[0, 1]) == T(Inf)
  @test isnan(GroupNormL2(T(Inf), G)(T[NaN, 1]))
  @test GroupNormL2(zero(T), G, T[Inf])(T[0, 1]) == T(Inf)
  @test GroupNormL2(zero(T), G, T[Inf])(T[0, 0]) === zero(T)
  @test GroupNormL2(zero(T), G, T[1])(T[NaN, 1]) === zero(T)
  @test isnan(GroupNormL2(one_, G)(T[NaN, 1])) && GroupNormL2(one_, G)(T[Inf, 1]) == T(Inf)
  # value_and_prox! returns the value at the stored output also on exceptional paths
  for (f, x, γ) in ((GroupNormL2(T(Inf), [[1, 2], [3]], T[1, 0]), T[3, 4, 5], zero(T)),
                    (GroupNormL2(one_, G, T[Inf]), T[3, 4], one_),
                    (GroupNormL2(one_, G), T[Inf, 1], one_),
                    (GroupNormL2(one_, G), T[NaN, 1], one_),
                    (GroupNormL2(one_, G), T[NaN, 1], zero(T)))
    y = similar(x)
    v = value_and_prox!(y, f, x, γ)
    @test isequal(y, prox(f, x, γ)) && isequal(v, f(y))
  end
end

@testset "overflow and underflow of norms, thresholds and values ($T)" for T in (Float32, Float64)
  G = [1:3]
  emax, emin = exponent(floatmax(T)), exponent(floatmin(T))
  s(e) = T(2)^e
  cases = [
    # (u, γ, λ, w): sums of squares that overflow
    (s(emax - 1) .* T[1, 0.75, -0.5], one(T), s(emax - 2), one(T)),
    (s(emax) .* T[1.5, 1.5, 1], one(T), floatmax(T) / 2, T(3)),     # γλw overflows, result representable
    (s(emax - 3) .* T[1, -1, 1], s(emax - 10), s(5), T(1.5)),
    # sums of squares that underflow, with thresholds at the same scale
    (s(emin - 5) .* T[1, 0.75, -0.5], one(T), s(emin - 6), one(T)),
    (s(emin + 1) .* T[1, -1, 0.25], s(emin + 2), s(-3), T(0.5)),    # γλ and γλw subnormal
    (s(emin - 20) .* T[1, 3, 2], s(-100), s(emin + 80), T(2)^-3),    # subnormal entries
    (s(emin ÷ 2 - 3) .* T[1, 1, 1], one(T), s(emin ÷ 2 - 2), one(T)), # squares below floatmin/eps
    # ordinary data with thresholds out of range
    (T[3, -4, 12], s(emax - 1), T(8), T(6.5) * s(-emax - 2)),        # γλ overflows, γλw = 6.5
    (T[3, -4, 12], s(emin), T(0.5), T(1.625) * s(-emin)),             # γλ subnormal, γλw = 0.8125
    (T[3, -4, 12], one(T), s(emin - 20), one(T)),                     # subnormal threshold
  ]
  for (u, γ, λ, w) in cases
    f = GroupNormL2(λ, G, T[w])
    ref = refprox(u, γ, λ, w)
    y = prox_block(f, u, 1, γ)
    @test close_prox(y, ref, u)
    z = similar(u)
    v = value_and_prox!(z, f, u, γ)
    @test z == y
    @test close_value(v, refvalue(z, λ, G, T[w]))
    @test close_value(f(u), refvalue(u, λ, G, T[w]))
  end
  # a value whose terms round into the subnormal range (#373): 512 groups whose terms are
  # 2^(emin-9) plus half a smallest-subnormal spacing (2^-135 + 2^-150 in Float32), each rounded to
  # 2^(emin-9); the ordinary sum floatmin loses 256 smallest-subnormal spacings
  K = 512
  e0 = emin - 9
  groups = [[2k - 1, 2k] for k in 1:K]
  x = repeat(s(e0 ÷ 2) .* T[1 + T(2)^-(precision(T) - 9), 0], K)
  λ = s(e0 - e0 ÷ 2)
  f = GroupNormL2(λ, groups, ones(T, K))
  ref = refvalue(x, λ, groups, ones(T, K))
  @test sum(fill(λ * x[1], K)) == floatmin(T) != T(ref)   # the fixture defeats the ordinary sum
  @test close_value(f(x), ref; k = 2)
  @test f(x) == NormL1(λ)(x)                 # the singleton-equivalent NormL1 value, also recomputed
  y = similar(x)
  @test close_value(value_and_prox!(y, f, x, zero(T)), ref; k = 2) && y == x
end

@testset "BlockIterate storage separation ($T)" for T in (Float32, Float64)
  groups = [[5, 1], [2, 7, 3], [4], [8, 6]]
  w = T[1, 0.5, 2, 1]
  f = GroupNormL2(T(0.8), groups, w)
  x = BlockIterate(T, 8, f)
  @test collect(blockindices(x, 2)) == [2, 7, 3] && ncoordinates(x) == 4 && nactive(x) == 0
  @test BlockIterate(8, f) isa BlockIterate{Float64}
  @test ProximalBase.samegroups(x, BlockIterate(zeros(T, 8), groups))
  @test_throws ArgumentError BlockIterate(9, f)                 # the groups must partition 1:p
  @test_throws ArgumentError BlockIterate(zeros(8), GroupNormL2(1.0, [[1, 2]]))
  x[7] = T(3); x[2] = T(-4)                                       # group 2 active
  activate!(x, 3)                                                 # group 3 active with a stored zero
  x[6] = T(0.5)                                                   # group 4 active
  before, order, n = Vector(x), collect(activecoordinates(x)), nactive(x)
  γ = T(1.5)
  outs = Dict{Int, Vector{T}}()
  for k in 1:4
    u = Vector(blockview(x, k)) .+ T(0.25)                        # a trial built from the iterate
    out = similar(u)
    prox_block!(out, f, u, k, γ)
    @test close_prox(out, refprox(u, γ, T(0.8), w[k]), u)
    outs[k] = out
    # the pure oracle changes no value, activity, order or stored zero
    @test Vector(x) == before && collect(activecoordinates(x)) == order && nactive(x) == n
    @test isactive(x, 3) && !isactive(x, 1)
  end
  # a block view as the input is read, not written
  y = copy(x)
  out = zeros(T, 3)
  prox_block!(out, f, blockview(y, 2), 2, γ)
  @test Vector(y) == before && collect(activecoordinates(y)) == order
  # explicit writes are a separate operation, with the activation rule of the iterate
  for k in 1:4
    copyto!(blockview(x, k), outs[k])
  end
  @test Vector(x)[vcat(groups...)] == vcat((outs[k] for k in 1:4)...)
  @test isactive(x, 1) == any(!iszero, outs[1])
  @test all(k -> isactive(x, k), 2:4)                             # zeros never deactivate a group
  # a zero result for an inactive group leaves it inactive
  z = BlockIterate(T, 8, f)
  copyto!(blockview(z, 1), prox_block(f, T[0.1, 0.1], 1, γ))
  @test !isactive(z, 1) && nactive(z) == 0
  # array oracles on block iterates: dense values, writes follow the same rule
  d = Vector(x)
  @test f(x) == f(d)
  r = similar(x)
  @test prox!(r, f, x, γ) === r && Vector(r) == prox(f, d, γ)
  @test value_and_prox!(similar(x), f, x, γ) == value_and_prox!(similar(d), f, d, γ)
end

@testset "group KKT screens ($T)" for T in (Float32, Float64)
  groups = [[5, 1], [2, 7, 3], [4], [8, 6]]
  w = T[1, 0.5, 2, 0]
  λ = T(0.8)
  f = GroupNormL2(λ, groups, w)
  x = BlockIterate(T, 8, f)
  rng = Xoshiro(902)
  g = randn(rng, T, 8)
  for k in 1:4
    ref = bignorm(g[groups[k]]) - big(λ) * big(w[k])
    @test close_value(kkt_violation(f, g, x, k), ref)
  end
  @test kkt_violation(f, g, x, 4) ≈ norm(g[[8, 6]])               # zero weight
  # array points: the group screen at a zero group, |g[i]| at a nonzero group and outside the groups
  a = zeros(T, 9); a[2] = 1
  @test kkt_violation(f, [g; T(5)], a, 1) == kkt_violation(f, g, x, 1)
  @test kkt_violation(f, [g; T(5)], a, 7) == abs(g[7])
  @test kkt_violation(f, [g; T(-5)], a, 9) == 5
  # infinite weight or level: never violated
  @test kkt_violation(GroupNormL2(λ, groups, T[Inf, 1, 1, 1]), g, x, 1) == -Inf
  @test kkt_violation(GroupNormL2(T(Inf), groups, w), g, x, 1) == -Inf
  @test kkt_violation(GroupNormL2(zero(T), groups, T[Inf, 1, 1, 1]), g, x, 1) == -Inf
  @test isnan(kkt_violation(f, T[NaN, 0, 0, 0, 1, 0, 0, 0], x, 1))
  # norms and coefficients out of range
  big_ = floatmax(T)
  gg = zeros(T, 8); gg[[5, 1]] .= big_                             # ‖g_G‖ = √2 floatmax overflows
  for (λb, wb) in ((big_ / 2, T(2)), (big_ / 2, T(4)), (big_, T(2)), (one(T), one(T)))
    fb = GroupNormL2(λb, groups, T[wb, 1, 1, 1])
    @test close_value(kkt_violation(fb, gg, x, 1), bignorm(gg[[5, 1]]) - big(λb) * big(wb))
  end
  tiny = zeros(T, 8); tiny[[5, 1]] .= nextfloat(zero(T)) * 3
  ft = GroupNormL2(nextfloat(zero(T)), groups, T[2, 1, 1, 1])
  @test close_value(kkt_violation(ft, tiny, x, 1), bignorm(tiny[[5, 1]]) - 2 * big(nextfloat(zero(T))))
  # layout, extent and argument checks
  @test_throws DimensionMismatch kkt_violation(f, g, BlockIterate(T, 8, [[1, 5], [2, 7, 3], [4], [8, 6]]), 1)
  @test kkt_violation(f, g, BlockIterate(T, 8, [[5, 1], [2, 7, 3], [4], [8, 6]]), 1) == kkt_violation(f, g, x, 1)
  @test_throws DimensionMismatch kkt_violation(f, g, BlockIterate(T, 8, 4), 1)
  @test_throws DimensionMismatch kkt_violation(f, g, BlockIterate(T, 8, 2), 1)
  @test_throws BoundsError kkt_violation(f, g, x, 5)
  @test_throws DimensionMismatch kkt_violation(f, g[1:7], x, 1)
  @test_throws DimensionMismatch kkt_violation(f, g[1:7], zeros(T, 8), 1)
  @test_throws ArgumentError kkt_violation(f, g, 1)
  @test_throws ArgumentError kkt_violation(f, Complex{T}.(g), x, 1)
  @test_throws ArgumentError kkt_violation(f, ones(T, 2, 2), SymmetricSparseIterate(T[1 2; 2 1]), 1)
  # a scaled penalty scales its screen; its array prox forwards the step
  @test kkt_violation(2 * f, g, x, 2) ≈ 2 * kkt_violation(f, g / 2, x, 2)
  @test prox(2 * f, g, T(0.5)) == prox(f, g, one(T))
  @test allocs(kkt_violation, f, g, x, 2) == 0
end

@testset "values at the overflow threshold and subnormal values ($T)" for T in (Float32, Float64)
  # Outside a band of 32 eps around the threshold Ω = floatmax + ulp/2, where rounding to T overflows, a
  # value has the class of its correctly rounded BigFloat value and is within 8 eps of it (two subnormal
  # spacings in the subnormal range). Within the band it may be Inf or such a finite value (#904).
  Ω = big(floatmax(T)) * (1 + big(eps(T)) / 4)
  same_class(v, ref) = abs(ref / Ω - 1) <= 32 * big(eps(T)) ?
    (isinf(v) || abs(big(v) - ref) <= 8 * big(eps(T)) * abs(ref)) :
    (isinf(v) == isinf(T(ref)) && (isinf(v) || abs(big(v) - ref) <= 8 * big(eps(T)) * abs(ref) + 2 * big(nextfloat(zero(T)))))
  # reproducers of standalone values whose ordinary or recovered sum overflowed although the value rounds
  # to floatmax; they lie within the band
  fixtures = T === Float32 ?
    [(Float32[-2.5213273, 0.27024558, 0.64377713, 0.8137605, -0.49109128, -1.100171, 1.5615983, 0.70515937], [1:4, 5:8], Float32[0.8435895, 0.73866713], 8.818969f37),
     (Float32[1.2626572, 0.79796016, -1.6117067], [1:3], Float32[1.2702762], 1.2190703f38)] :
    [([-0.07312823706023215, 1.3229738772397932, 0.4408404901829929, -0.5521009045153256, -0.45097332420421443], [1:5], [0.6830546515360696], 1.67863692812149e308),
     ([0.38404986280275943, -1.9135391084491589, 0.024367670945626523, -1.2807864420007569, 0.1981528027756723, -0.1403750996117117], [1:3, 4:6], [1.4948884409876482, 1.1576985411815683], 4.060767109142168e307)]
  for (x, groups, w, λ) in fixtures
    f = GroupNormL2(λ, groups, w)
    ref = refvalue(x, λ, groups, w)
    @test ref < big(floatmax(T)) && same_class(f(x), ref)
  end
  # sweeps across floatmax: direct values on arrays, sparse and block iterates, and fused values
  rng = Xoshiro(903)
  for K in (1, 3), n in (1, 4)
    groups = [((k - 1) * n + 1):(k * n) for k in 1:K]
    x = randn(rng, T, n * K)
    w = T(0.5) .+ rand(rng, T, K)
    base = refvalue(x, 1, groups, w)
    for k in -6:6
      λ = T(setprecision(() -> big(floatmax(T)) / base * (1 + k * big(eps(T)) / 2), BigFloat, PREC))
      isfinite(λ) || continue
      f = GroupNormL2(λ, groups, w)
      ref = refvalue(x, λ, groups, w)
      @test same_class(f(x), ref)
      @test same_class(f(SparseIterate(copy(x))), ref)
      @test same_class(f(BlockIterate(copy(x), f)), ref)
      y = similar(x)
      v = value_and_prox!(y, f, x, nextfloat(zero(T)) * 3)
      @test same_class(v, refvalue(y, λ, groups, w))
      z = similar(x)
      @test same_class(value_and_prox!(z, f, x, zero(T)), ref) && z == x
      # a NaN or an infinite entry keeps the ordinary NaN or Inf near the threshold too
      xe = copy(x); xe[1] = T(NaN)
      @test isnan(f(xe))
      xe[1] = T(Inf)
      @test f(xe) == Inf
    end
  end
  # near floatmax, a zero-weight group contributes zero also at a NaN or ±Inf entry, and an infinite
  # weight contributes zero at an exactly zero group (analytical expectations), direct and fused
  for (w, x) in ((T[1, 0], T[1, NaN]), (T[1, 0], T[1, -Inf]), (T[1, Inf], T[1, 0]), (T[1, Inf], T[1, -0.0]))
    f = GroupNormL2(floatmax(T), [1:1, 2:2], w)
    @test f(x) == floatmax(T)
    y = similar(x)
    @test value_and_prox!(y, f, x, zero(T)) == floatmax(T) && isequal(y, w[2] == 0 ? x : T[1, 0])
  end
  # subnormal values and terms, direct and fused, against BigFloat (two subnormal spacings)
  emin = exponent(floatmin(T))
  for (x, λ) in ((T[3, 4, 0, 1], T(2)^(emin - 8)), (T(2)^(emin + 2) .* T[3, 4, 1, 1], T(2)^-6))
    f = GroupNormL2(λ, [1:2, 3:4], T[1, 2])
    ref = refvalue(x, λ, [1:2, 3:4], T[1, 2])
    @test ref < big(floatmin(T)) && same_class(f(x), ref)
    y = similar(x)
    @test same_class(value_and_prox!(y, f, x, zero(T)), ref)
  end
end

@testset "no ambiguities" begin
  @test isempty(Test.detect_ambiguities(ProximalBase; recursive = true))
end

end
