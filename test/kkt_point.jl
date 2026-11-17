module KktPointTest
# The point-aware forms of kkt_violation (#970) for NormL1, ElasticNet, ZeroFunction, LinearFunction and
# SqrNormL2:
#   kkt_violation(f, v, i)      the screen of coordinate i assumed zero (unchanged),
#   kkt_violation(f, v, x, k)   the score of the actual coordinate or group k of x,
#   kkt_violation(f, v, x)      max(0, max_k score_k).
# References are written here from the definitions of the scores: in BigFloat (finite arguments), or as the scalar
# formula in the working type for the claims of bit equality. No reference calls a ProximalBase kernel.
# Accuracy is measured against k eps(W) (|v| + |λ2 x| + c), the size of the terms that cancel, not against the result.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random, OffsetArrays
using ..TestUtils
const PB = ProximalBase

const TYPES = (Float32, Float64)
const PREC = 1024

# ---- references ----------------------------------------------------------------------------------------------
bigprec(f) = setprecision(f, BigFloat, PREC)
wt(w, i) = w === nothing ? 1 : w[i]
center(y, i) = y === nothing ? 0 : y[i]
sgn(x) = x > 0 ? one(x) : x < 0 ? -one(x) : zero(x)

# the score of entry i from the definition, and the size of the terms that it adds (finite arguments)
function refscore(f::NormL1, v, x, i)
  V, X, c = big(v), big(x), big(f.λ0) * big(wt(f.w, i))
  iszero(X) ? abs(V) - c : abs(V + c * sgn(X))
end
refscale(f::NormL1, v, x, i) = abs(big(v)) + big(f.λ0) * big(wt(f.w, i))
function refscore(f::ElasticNet, v, x, i)
  V, X, c = big(v), big(x), big(f.λ1) * big(wt(f.w, i))
  iszero(X) ? abs(V) - c : abs(V + big(f.λ2) * X + c * sgn(X))
end
refscale(f::ElasticNet, v, x, i) = abs(big(v)) + abs(big(f.λ2) * big(x)) + big(f.λ1) * big(wt(f.w, i))
refscore(::ZeroFunction, v, x, i) = abs(big(v))
refscale(::ZeroFunction, v, x, i) = abs(big(v))
refscore(f::LinearFunction, v, x, i) = abs(big(v) + big(f.c[i]))
refscale(f::LinearFunction, v, x, i) = abs(big(v)) + abs(big(f.c[i]))
refscore(f::SqrNormL2, v, x, i) = abs(big(v) + big(f.λ) * (big(x) - big(center(f.y, i))))
refscale(f::SqrNormL2, v, x, i) = abs(big(v)) + abs(big(f.λ) * (big(x) - big(center(f.y, i))))

# the result r of type W is within k eps(W) of the scale of the terms, plus a few subnormal spacings
function within(r, f, v, x, i; k = 4)
  W = typeof(r)
  isfinite(r) || return false
  bigprec() do
    abs(big(r) - refscore(f, v, x, i)) <= k * big(eps(W)) * refscale(f, v, x, i) + 4 * big(nextfloat(zero(W)))
  end
end
function within_point(r, f, v, x; k = 4)
  W = typeof(r)
  isfinite(r) || return false
  bigprec() do
    n = length(x)
    ref = n == 0 ? big(0) : max(big(0), maximum(refscore(f, v[i], x[i], i) for i in 1:n))
    scale = n == 0 ? big(0) : maximum(refscale(f, v[i], x[i], i) for i in 1:n)
    abs(big(r) - ref) <= k * big(eps(W)) * scale + 4 * big(nextfloat(zero(W)))
  end
end

# the scalar formulas in the working type, as the contract writes them
l1_formula(c, v, x) = iszero(x) ? abs(v) - c : abs(v + c * sign(x))
function en_formula(c, l2, v, x)
  iszero(x) && return abs(v) - c
  iszero(l2) ? abs(v + c * sign(x)) : abs((v + l2 * x) + c * sign(x))
end

# the two loops of ProximalSolvers that unweighted NormL1 and ElasticNet reproduce bit for bit
function ps_l1(grad, x, λ::T) where {T}
  m = zero(T)
  for i in eachindex(grad, x)
    xi = x[i]
    r = iszero(xi) ? max(abs(grad[i]) - λ, zero(T)) : abs(grad[i] + λ * sign(xi))
    m = max(m, r)
  end
  m
end
function ps_en(grad, x, λ1::T, λ2::T) where {T}
  m = zero(T)
  for i in eachindex(grad, x)
    xi = x[i]
    r = iszero(xi) ? max(abs(grad[i]) - λ1, zero(T)) : abs(grad[i] + λ2 * xi + λ1 * sign(xi))
    m = max(m, r)
  end
  m
end

# the maximum that keeps a NaN, and max(0, ·)
nanmax(a, b) = (isnan(a) || isnan(b)) ? oftype(a, NaN) : max(a, b)

msg(f) = try f(); "" catch e; sprint(showerror, e) end
same(a, b) = isequal(a, b) && typeof(a) === typeof(b)

# random data of one type: the point has about 40% of exact zeros
function data(rng, ::Type{T}, n) where {T}
  x = randn(rng, T, n)
  x[rand(rng, n) .< 0.4] .= zero(T)
  v = 2 .* randn(rng, T, n)
  x, v
end
weights(rng, ::Type{T}, n) where {T} = (w = T(0.2) .+ rand(rng, T, n); w[rand(rng, n) .< 0.2] .= zero(T); w)

# the functions of the family, with a mix of weights, centers and coefficients
function functions(rng, ::Type{T}, n) where {T}
  w, y, c = weights(rng, T, n), randn(rng, T, n), randn(rng, T, n)
  [NormL1(T(0.3)), NormL1(T(0.3), w), NormL1(T(0), w), ElasticNet(T(0.3), T(0.7)), ElasticNet(T(0.3), T(0.7), w),
   ElasticNet(T(0.3), T(0), w), ElasticNet(T(0), T(0.7)), ZeroFunction(), LinearFunction(c), SqrNormL2(T(0.7)),
   SqrNormL2(T(0.7), y), SqrNormL2(T(0), y)]
end

# a function defined outside the package: only the screen of a coordinate assumed zero; one without it; and one
# with its own four-argument and full-point methods
struct OnlyScreen <: AbstractFunction end
PB.kkt_violation(::OnlyScreen, g::AbstractArray, i::Integer) = abs(g[i]) - 1
struct NoScreen <: AbstractFunction end
struct OwnScores <: AbstractFunction end
PB.kkt_violation(::OwnScores, g::AbstractArray, x::AbstractArray, k::Integer) = 10 * g[k] + x[k]
PB.kkt_violation(::OwnScores, g::AbstractArray, x::BlockIterate, k::Integer) = sum(g[blockindices(x, k)])
PB.kkt_violation(::OwnScores, g::AbstractArray, x::SymmetricSparseIterate, k::Integer) = 0.0
PB.kkt_violation(::OwnScores, g::AbstractArray, x::AbstractArray) = sum(abs, g) + sum(abs, x)

# ---- 1. the worked example and the local forms ---------------------------------------------------------------
@testset "$T the worked example of the contract" for T in TYPES
  x = T[0, 0, 2, -1]
  v = T[0.8, 0.2, 0, 0.5]
  c = T(0.5)
  g = NormL1(c)
  screen = [abs(v[i]) - c for i in 1:4]
  point = [abs(v[1]) - c, abs(v[2]) - c, abs(v[3] + c * sign(x[3])), abs(v[4] + c * sign(x[4]))]
  for i in 1:4
    @test kkt_violation(g, v, i) === screen[i]          # the screen is unchanged: -0.5 at the nonzero entry 3
    @test kkt_violation(g, v, x, i) === point[i]
  end
  @test screen[3] == -T(0.5) && point[3] == T(0.5) && point[4] == 0
  @test point[1] > 0 && point[2] < 0                    # signed at zero entries, negative slack kept
  @test kkt_violation(g, v, x) === T(0.5)
  # λ2 = 3: entries 3 and 4 are |(0 + 3·2) + 0.5| = 6.5 and |(0.5 + 3·(-1)) + 0.5·(-1)| = 3
  f = ElasticNet(c, T(3))
  @test kkt_violation(f, v, x, 3) === T(6.5) && kkt_violation(f, v, x, 4) === T(3)
  @test kkt_violation(f, v, x, 1) === screen[1] && kkt_violation(f, v, x, 2) === screen[2]
  @test kkt_violation(f, v, x) === T(6.5)
  for i in 1:4
    @test kkt_violation(f, v, i) === screen[i]
  end
  # the smooth functions and ZeroFunction: the same value at zero and nonzero entries
  @test [kkt_violation(ZeroFunction(), v, x, i) for i in 1:4] == abs.(v)
  @test kkt_violation(ZeroFunction(), v, x) === maximum(abs, v)
  cc = T[1, -1, 2, 0]
  h = LinearFunction(cc)
  @test [kkt_violation(h, v, x, i) for i in 1:4] == [abs(v[i] + cc[i]) for i in 1:4]
  @test kkt_violation(h, v, x) === maximum(abs(v[i] + cc[i]) for i in 1:4)
  q, y = T(2), T[1, 2, 3, 4]
  s0, s1 = SqrNormL2(q), SqrNormL2(q, y)
  @test [kkt_violation(s0, v, x, i) for i in 1:4] == [abs(v[i] + q * x[i]) for i in 1:4]
  @test [kkt_violation(s1, v, x, i) for i in 1:4] == [abs(v[i] + q * (x[i] - y[i])) for i in 1:4]
  @test kkt_violation(s1, v, x) === maximum(abs(v[i] + q * (x[i] - y[i])) for i in 1:4)
end

@testset "$T zero entries equal the screen, nonzero entries the formula, for the five functions" for T in TYPES
  rng = MersenneTwister(970)
  for n in (1, 5, 17), rep in 1:4
    x, v = data(rng, T, n)
    for f in functions(rng, T, n)
      for i in 1:n
        r = kkt_violation(f, v, x, i)
        @test r isa T
        iszero(x[i]) && @test r === kkt_violation(f, v, i)      # finite v and parameters of one type: bit for bit
        @test within(r, f, v[i], x[i], i)
      end
    end
    # the formulas in the working type
    w = weights(rng, T, n)
    l1 = NormL1(T(0.3), w)
    en = ElasticNet(T(0.3), T(0.7), w)
    @test all(kkt_violation(l1, v, x, i) === l1_formula(T(0.3) * w[i], v[i], x[i]) for i in 1:n)
    @test all(kkt_violation(en, v, x, i) === en_formula(T(0.3) * w[i], T(0.7), v[i], x[i]) for i in 1:n)
  end
end

@testset "$T negative margins are not clipped, and the full point value is" for T in TYPES
  f = NormL1(T(2), T[1, 2, 3])
  x, v = zeros(T, 3), T[1, -1, 0.5]
  @test [kkt_violation(f, v, x, i) for i in 1:3] == [T(1) - 2, T(1) - 4, T(0.5) - 6]
  @test all(<(0), [kkt_violation(f, v, x, i) for i in 1:3])
  @test kkt_violation(f, v, x) === zero(T)
  @test kkt_violation(ElasticNet(T(2), T(5), T[1, 2, 3]), v, x) === zero(T)
  # the signed value of the screen of a zero entry that is positive is the full point value
  v[2] = 5
  @test kkt_violation(f, v, x, 2) === T(5) - 4 && kkt_violation(f, v, x) === T(5) - 4
end

@testset "$T ElasticNet(λ1, 0) is NormL1(λ1) bit for bit" for T in TYPES
  rng = MersenneTwister(971)
  for n in (1, 4, 9, 33), w in (nothing, weights(rng, T, n))
    x, v = data(rng, T, n)
    f, g = ElasticNet(T(0.37), zero(T), w), NormL1(T(0.37), w)
    @test all(same(kkt_violation(f, v, x, i), kkt_violation(g, v, x, i)) for i in 1:n)
    @test same(kkt_violation(f, v, x), kkt_violation(g, v, x))
    @test all(same(kkt_violation(f, v, i), kkt_violation(g, v, i)) for i in 1:n)
  end
  # also with an integer λ2 and a mixed type of the data
  x, v = Float64[0, 2, -3], T[1, 2, 3]
  @test same(kkt_violation(ElasticNet(0.5, 0), v, x), kkt_violation(NormL1(0.5), v, x))
end

# ---- 2. mixed types against BigFloat ---------------------------------------------------------------------------
@testset "mixed element types and parameters against BigFloat" begin
  rng = MersenneTwister(972)
  n = 25
  for (V, X, P) in ((Float32, Float64, Float64), (Float64, Float32, Float32), (Float32, Float32, Float64),
                    (Float64, Float64, Float32), (Float32, Int, Float64), (Int, Float32, Float32))
    x = X <: Integer ? rand(rng, -2:2, n) : (xx = randn(rng, X, n); xx[1:3:end] .= 0; xx)
    v = V <: Integer ? rand(rng, -3:3, n) : randn(rng, V, n)
    w = P.(weights(rng, Float64, n))
    y, c = randn(rng, P, n), randn(rng, P, n)
    W = float(promote_type(V, X))
    for f in (NormL1(P(0.3), w), NormL1(P(0.3)), ElasticNet(P(0.3), P(0.7), w), ElasticNet(P(0.3), P(0.7)),
              ZeroFunction(), LinearFunction(c), SqrNormL2(P(0.7), y), SqrNormL2(P(0.7)))
      Wf = f isa LinearFunction ? float(promote_type(W, P)) : f isa SqrNormL2 && f.y !== nothing ? float(promote_type(W, P)) : W
      r = [kkt_violation(f, v, x, i) for i in 1:n]
      @test all(ri -> ri isa Wf, r)
      @test all(within(r[i], f, v[i], x[i], i) for i in 1:n)
      R = kkt_violation(f, v, x)
      @test R isa Wf && within_point(R, f, v, x)
    end
  end
end

# ---- 3. the full point form -------------------------------------------------------------------------------------
@testset "$T the full point value" for T in TYPES
  rng = MersenneTwister(973)
  for n in (1, 2, 3, 7, 8, 9, 31, 64, 100, 257), rep in 1:3
    x, v = data(rng, T, n)
    for f in functions(rng, T, n)
      R = kkt_violation(f, v, x)
      @test R isa T
      @test R === max(zero(T), maximum(kkt_violation(f, v, x, i) for i in 1:n))
      @test within_point(R, f, v, x)
    end
    # unweighted NormL1 and ElasticNet reproduce the two loops of ProximalSolvers bit for bit
    λ = 0.3
    @test kkt_violation(NormL1(λ), v, x) === ps_l1(v, x, T(λ))
    @test kkt_violation(ElasticNet(λ, 0.7), v, x) === ps_en(v, x, T(λ), T(0.7))
    @test kkt_violation(ElasticNet(λ, 0.0), v, x) === ps_l1(v, x, T(λ))
  end
end

@testset "$T a point from a proximal step is stationary within rounding" for T in TYPES
  rng = MersenneTwister(974)
  n, γ = 200, T(0.5)
  w = weights(rng, T, n)
  for f in (NormL1(T(2)), NormL1(T(2), w), ElasticNet(T(2), T(0.9)), ElasticNet(T(2), T(0.9), w))
    u = 3 .* randn(rng, T, n)
    x = PB.prox!(similar(u), f, u, γ)
    @test any(iszero, x) && any(!iszero, x)
    v = (x .- u) ./ γ        # the negative of a subgradient of f at x: 0 ∈ v + ∂f(x)
    λ1, λ2 = f isa ElasticNet ? (f.λ1, f.λ2) : (f.λ0, zero(T))
    # the rounding of the step and of the division is measured against |u|/γ as well
    scale = maximum(abs, u) / γ + maximum(abs, v) + λ2 * maximum(abs, x) + λ1 * maximum(wt(f.w, i) for i in 1:n)
    @test kkt_violation(f, v, x) <= 16 * eps(T) * scale
  end
end

@testset "the cancellation case" begin
  f, x, v = ElasticNet(0.5, 0.1), [3.0], [-0.8]
  r = kkt_violation(f, v, x)
  # 0.1·3 = 0.30000000000000004 and -0.8 + that = -0.5 exactly: the computed score is exactly 0, while the
  # BigFloat value of the same rounded parameters is about 2.8e-17, within the bound of the size of the terms
  @test r === 0.0
  @test within_point(r, f, v, x)
  @test bigprec(() -> refscore(f, v[1], x[1], 1)) > 1e-17
  @test kkt_violation(f, v, x, 1) === 0.0
end

@testset "$T empty arrays, matrices, SparseIterate and BlockIterate" for T in TYPES
  for f in (NormL1(T(0.3)), NormL1(T(0.3), T[]), ElasticNet(T(0.3), T(0.2)), ZeroFunction(), LinearFunction(T[]),
            SqrNormL2(T(0.5)), SqrNormL2(T(0.5), T[]))
    @test same(kkt_violation(f, T[], T[]), zero(T))
  end
  @test same(kkt_violation(NormL1(0.3), Float32[], Float64[]), 0.0)
  @test same(kkt_violation(NormL1(0.3), Int[], Int[]), 0.0)
  rng = MersenneTwister(975)
  # a matrix point with a flat vector, and the other way around
  X = randn(rng, T, 3, 4)
  X[2, :] .= 0
  vf = randn(rng, T, 12)
  W = reshape(weights(rng, T, 12), 3, 4)
  for f in (NormL1(T(0.3), W), ElasticNet(T(0.3), T(0.5), W), LinearFunction(randn(rng, T, 3, 4)),
            SqrNormL2(T(0.5), randn(rng, T, 3, 4)), NormL1(T(0.3)))
    @test kkt_violation(f, vf, X) === kkt_violation(f, reshape(vf, 3, 4), X)
    @test within_point(kkt_violation(f, vf, X), f, vf, X)
    @test within_point(kkt_violation(f, reshape(vf, 3, 4), X), f, vf, X)
    @test all(within(kkt_violation(f, vf, X, i), f, vf[i], X[i], i) for i in 1:12)
  end
  # SparseIterate (a vector and a matrix) and BlockIterate, by entry
  for n in (6, 50)
    x, v = data(rng, T, n)
    sx = SparseIterate(T, n)
    for i in 1:n
      iszero(x[i]) || (sx[i] = x[i])
    end
    sx[2] = 1; sx[2] = 0                                       # a stored zero
    x[2] = 0
    groups = [collect(1:2:n), collect(2:2:n)]
    bx = BlockIterate(copy(x), groups)
    for f in functions(rng, T, n)
      @test kkt_violation(f, v, sx) === kkt_violation(f, v, x)
      @test kkt_violation(f, v, bx) === kkt_violation(f, v, x)
    end
  end
  sm = SparseIterate(T, 2, 3)
  sm[2, 1] = T(2); sm[1, 3] = T(-1)
  Xm = Matrix(sm)
  vm = randn(rng, T, 6)
  f = NormL1(T(0.3), reshape(weights(rng, T, 6), 2, 3))
  @test kkt_violation(f, vm, sm) === kkt_violation(f, vm, Xm)
end

# ---- 4. the smooth functions -----------------------------------------------------------------------------------
@testset "$T the smooth functions" for T in TYPES
  rng = MersenneTwister(976)
  for n in (1, 7, 40)
    x, v = data(rng, T, n)
    y, c = randn(rng, T, n), randn(rng, T, n)
    for f in (SqrNormL2(T(0.7)), SqrNormL2(T(0.7), y), SqrNormL2(T(1.5), y))
      @test kkt_violation(f, v, x) == norm(v .+ gradient(f, x), Inf)
      @test all(kkt_violation(f, v, x, i) >= 0 for i in 1:n)       # a point subdifferential: no negative slack
    end
    @test kkt_violation(ZeroFunction(), v, x) == norm(v, Inf)
    @test kkt_violation(LinearFunction(c), v, x) == norm(v .+ c, Inf)
    @test all(kkt_violation(ZeroFunction(), v, x, i) >= 0 for i in 1:n)
  end
  # the ridge gradient recovers a finite entry where xᵢ - yᵢ overflows; the score takes it from the same helpers
  f = SqrNormL2(T(0.25), T[-floatmax(T), 0])
  x, v = T[floatmax(T), 1], T[1, 1]
  @test isfinite(gradient(f, x)[1])
  @test kkt_violation(f, v, x) == norm(v .+ gradient(f, x), Inf)
  @test kkt_violation(f, v, x, 1) == abs(v[1] + gradient(f, x)[1])
  # a level that does not fit the type is used in the wider type, as the gradient does
  if T === Float32
    f = SqrNormL2(1e-40, T[1, 2])
    x, v = T[3, 4], zeros(T, 2)
    @test kkt_violation(f, v, x) == norm(gradient(f, x), Inf) > 0
    @test kkt_violation(f, v, x, 2) == abs(gradient(f, x)[2])
  end
  # LinearFunction: the coefficients have the size of x
  @test_throws DimensionMismatch kkt_violation(LinearFunction(ones(T, 3)), ones(T, 4), ones(T, 4))
  @test_throws DimensionMismatch kkt_violation(LinearFunction(ones(T, 6)), ones(T, 6), ones(T, 2, 3))
  @test kkt_violation(LinearFunction(ones(T, 2, 3)), ones(T, 6), ones(T, 2, 3)) == 2
end

# ---- 5. exceptional values -------------------------------------------------------------------------------------
# the contract: a NaN in the entries of v and of x that the formula reads gives NaN, otherwise an infinite one gives Inf
function rule(f, v, x)
  usesx = !(f isa Union{ZeroFunction, LinearFunction})
  (isnan(v) || (usesx && isnan(x))) && return NaN
  (isinf(v) || (usesx && isinf(x))) && return Inf
  nothing
end

@testset "$T NaN and infinite entries of v and x" for T in TYPES
  specials = T[NaN, Inf, -Inf, 0, -0.0, 1, -1, floatmax(T), nextfloat(zero(T))]
  n = 17
  w = fill(T(0.5), n)
  fs = [NormL1(T(0.3)), NormL1(T(0.3), w), ElasticNet(T(0.3), T(0.7)), ElasticNet(T(0.3), T(0.7), w),
        ElasticNet(T(0.3), T(0)), ZeroFunction(), LinearFunction(fill(T(0.25), n)), SqrNormL2(T(0.7)),
        SqrNormL2(T(0.7), fill(T(0.5), n))]
  fm = bigprec(() -> big(floatmax(T)))
  for f in fs, a in specials, b in specials
    r = kkt_violation(f, [a], [b], 1)
    k = rule(f, a, b)
    if k !== nothing
      @test isequal(r, T(k))
    elseif isfinite(a) && isfinite(b)
      ref = bigprec(() -> refscore(f, a, b, 1))
      if ref > fm * (1 + 8 * eps(T))
        @test r == Inf
      elseif ref < fm * (1 - 8 * eps(T))
        @test within(r, f, a, b, 1)
      end
    end
    # one entry among 17: the loop over the natural scores and the pass by entry agree with the local forms
    for pos in (1, 9, 17)
      v, x = fill(T(0.75), n), fill(T(-0.5), n)
      v[pos], x[pos] = a, b
      R = kkt_violation(f, v, x)
      @test isequal(R, foldl(nanmax, (kkt_violation(f, v, x, i) for i in 1:n); init = zero(T)))
      k === nothing || @test isequal(R, T(k))
    end
  end
  # NaN takes precedence over Inf across entries, in v and in x
  for f in fs
    v, x = fill(T(0.75), n), fill(T(-0.5), n)
    v[1], v[2] = Inf, NaN
    @test isnan(kkt_violation(f, v, x))
    v[1], v[2] = NaN, Inf
    @test isnan(kkt_violation(f, v, x))
    f isa Union{ZeroFunction, LinearFunction} && continue
    v = fill(T(0.75), n)
    x[1], x[2] = Inf, NaN
    @test isnan(kkt_violation(f, v, x))
    x[1], x[2] = NaN, -Inf
    @test isnan(kkt_violation(f, v, x))
  end
  # ZeroFunction and LinearFunction do not read x
  x = fill(T(NaN), n)
  @test kkt_violation(ZeroFunction(), fill(T(1), n), x) === T(1)
  @test kkt_violation(LinearFunction(fill(T(0.25), n)), fill(T(1), n), x) === T(1.25)
  @test kkt_violation(ZeroFunction(), fill(T(1), n), x, 3) === T(1)
end

@testset "$T the overriding rules of the contract" for T in TYPES
  I = T(Inf)
  l1, en = NormL1(T(0.5)), ElasticNet(T(0.5), T(2))
  # x = ±Inf with λ2 = 0: Inf, not a finite score
  for x in (I, -I), v in (T(1), -T(1), T(0))
    @test kkt_violation(l1, [v], [x], 1) === I && kkt_violation(l1, [v], [x]) === I
    @test kkt_violation(ElasticNet(T(0.5), T(0)), [v], [x], 1) === I
  end
  # x = ±Inf, λ2 > 0 and v the opposite infinity: Inf, not NaN
  @test kkt_violation(en, [-I], [I], 1) === I && kkt_violation(en, [I], [-I], 1) === I
  @test kkt_violation(en, [-I], [I]) === I && kkt_violation(en, [I], [-I]) === I
  @test kkt_violation(SqrNormL2(T(2)), [-I], [I], 1) === I
  # an infinite v with an infinite coefficient: Inf, not NaN, at a zero and at a nonzero entry
  for f in (NormL1(I), NormL1(T(1), [I]), NormL1(T(0), [I]), ElasticNet(I, T(1)), ElasticNet(T(1), T(1), [I]))
    for v in (I, -I), x in (T(0), T(1), -T(1))
      @test kkt_violation(f, [v], [x], 1) === I
      @test kkt_violation(f, [v], [x]) === I
    end
  end
  # an infinite ridge level: -Inf at a zero entry with a finite v, which the screen returns for any v, and the
  # point-aware forms do not for a NaN or infinite v
  e = ElasticNet(T(1), I)
  @test kkt_violation(e, [T(3)], 1) === -I && kkt_violation(e, [T(3)], [T(0)], 1) === -I
  @test kkt_violation(e, [T(NaN)], 1) === -I && kkt_violation(e, [I], 1) === -I        # the screen, unchanged
  @test isnan(kkt_violation(e, [T(NaN)], [T(0)], 1)) && kkt_violation(e, [I], [T(0)], 1) === I
  @test isnan(kkt_violation(e, [T(NaN)], [T(0)])) && kkt_violation(e, [-I], [T(0)]) === I
  @test kkt_violation(e, [T(3)], [T(2)], 1) === I && kkt_violation(e, [T(3)], [-T(2)], 1) === I
  @test kkt_violation(e, [T(3)], [T(0)]) === zero(T) && kkt_violation(e, [T(3)], [T(2)]) === I
  @test kkt_violation(ElasticNet(T(1), I, [T(0)]), [T(3)], [T(2)], 1) === I
  # SqrNormL2 with x and y both infinite
  @test kkt_violation(SqrNormL2(T(1), [I]), [T(0)], [I], 1) === I
  @test kkt_violation(SqrNormL2(T(1), [-I]), [T(0)], [-I]) === I
  # finite v and x: an infinite coefficient fixes the zero coordinate at zero
  for f in (NormL1(I), NormL1(T(1), [I]), NormL1(T(0), [I]))
    @test kkt_violation(f, [T(2)], [T(0)], 1) === -I && kkt_violation(f, [T(2)], [-T(0)], 1) === -I
    @test kkt_violation(f, [T(2)], [T(0)]) === zero(T)
    @test kkt_violation(f, [T(2)], [T(1)], 1) === I && kkt_violation(f, [T(2)], [-T(1)]) === I
  end
  # a zero weight at an infinite level leaves the entry unpenalized
  z = NormL1(I, [T(0), T(1)])
  @test kkt_violation(z, T[3, 3], T[0, 0], 1) === T(3) && kkt_violation(z, T[3, 3], T[1, 1], 1) === T(3)
  @test kkt_violation(z, T[3, 3], T[1, 1]) === I
  # a NaN parameter entry
  @test isnan(kkt_violation(LinearFunction([T(NaN)]), [T(1)], [T(1)], 1))
  @test isnan(kkt_violation(LinearFunction([T(NaN)]), [T(1)], [T(1)]))
  @test isnan(kkt_violation(SqrNormL2(T(1), [T(NaN)]), [T(1)], [T(1)], 1))
  @test isnan(kkt_violation(SqrNormL2(T(1), [T(NaN)]), [T(1)], [T(0)]))
  @test isnan(kkt_violation(LinearFunction(T[1, NaN, 3]), T[1, 1, 1], T[1, 1, 1]))
  # an infinite level of SqrNormL2 follows the gradient: NaN at a zero residual, Inf elsewhere
  s = SqrNormL2(I)
  @test isnan(kkt_violation(s, [T(1)], [T(0)], 1)) && kkt_violation(s, [T(1)], [T(2)], 1) === I
  @test isnan(kkt_violation(s, [T(1)], [T(0)])) && kkt_violation(s, [T(1)], [-T(2)]) === I
  # λ2·xᵢ overflows
  big_ = ElasticNet(T(1), floatmax(T))
  @test kkt_violation(big_, [T(1)], [T(2)], 1) === I && kkt_violation(big_, [T(1)], [-T(2)], 1) === I
  @test kkt_violation(big_, [T(1)], [T(2)]) === I
  # signed zeros, subnormal entries
  for f in (NormL1(T(0.5)), ElasticNet(T(0.5), T(2)), NormL1(T(0.5), [T(2)]), SqrNormL2(T(2), [T(1)]), LinearFunction([T(1)]))
    for v in (T(0.75), -T(0.75), T(0), -T(0))
      @test same(kkt_violation(f, [v], [T(0)], 1), kkt_violation(f, [v], [-T(0)], 1))
      @test same(kkt_violation(f, [v], [T(0)]), kkt_violation(f, [v], [-T(0)]))
    end
  end
  d = nextfloat(zero(T))
  for f in (NormL1(T(0.5)), ElasticNet(T(0.5), T(2)))
    @test kkt_violation(f, [T(0.75)], [d], 1) === abs(T(0.75) + T(0.5))
    @test kkt_violation(f, [T(0.75)], [-d], 1) === abs(T(0.75) - T(0.5))
    @test kkt_violation(f, [T(0.25)], [-d]) === T(0.25)
  end
  @test kkt_violation(NormL1(T(0.5)), [T(0)], [d], 1) === T(0.5)
  @test kkt_violation(NormL1(T(0.5)), [T(0)], [T(0)], 1) === -T(0.5)
end

# ---- 6. validation --------------------------------------------------------------------------------------------
@testset "$T validation" for T in TYPES
  x, v = T[0, 1, 2, 0], T[1, 2, 3, 4]
  fs = (NormL1(T(0.5)), NormL1(T(0.5), T[1, 2, 3, 4]), ElasticNet(T(0.5), T(1)), ZeroFunction(), LinearFunction(T[1, 2, 3, 4]),
        SqrNormL2(T(1)), SqrNormL2(T(1), T[1, 2, 3, 4]))
  for f in fs
    @test_throws DimensionMismatch kkt_violation(f, v[1:3], x)
    @test_throws DimensionMismatch kkt_violation(f, v, x[1:3])
    @test_throws DimensionMismatch kkt_violation(f, v[1:3], x, 1)
    @test_throws DimensionMismatch kkt_violation(f, v, x[1:3], 1)
    @test_throws BoundsError kkt_violation(f, v, x, 5)
    @test_throws BoundsError kkt_violation(f, v, x, 0)
    @test_throws ArgumentError kkt_violation(f, Complex{T}.(v), x)
    @test_throws ArgumentError kkt_violation(f, v, Complex{T}.(x))
    @test_throws ArgumentError kkt_violation(f, Complex{T}.(v), x, 2)
    @test_throws ArgumentError kkt_violation(f, v, Complex{T}.(x), 2)
    @test occursin("offset arrays", msg(() -> kkt_violation(f, OffsetArray(v, 0:3), x)))
    @test occursin("offset arrays", msg(() -> kkt_violation(f, v, OffsetArray(x, 0:3))))
    @test occursin("offset arrays", msg(() -> kkt_violation(f, OffsetArray(v, 0:3), x, 2)))
    @test occursin("offset arrays", msg(() -> kkt_violation(f, v, OffsetArray(x, 0:3), 2)))
  end
  # the extent of the parameters
  for f in (NormL1(T(0.5), T[1, 2, 3]), ElasticNet(T(0.5), T(1), T[1, 2, 3]), LinearFunction(T[1, 2, 3]),
            SqrNormL2(T(1), T[1, 2, 3]))
    @test_throws DimensionMismatch kkt_violation(f, v, x)
  end
  @test_throws BoundsError kkt_violation(NormL1(T(0.5), T[1, 2, 3]), v, x, 4)
  # a SparseIterate with a short gradient and an invalid coordinate
  sx = SparseIterate(T, 4)
  sx[2] = 1
  @test_throws DimensionMismatch kkt_violation(NormL1(T(0.5)), v[1:3], sx, 1)
  @test_throws BoundsError kkt_violation(NormL1(T(0.5)), v, sx, 5)
  # a BlockIterate with a short gradient
  bx = BlockIterate(T, 4, [[1, 2], [3, 4]])
  @test_throws DimensionMismatch kkt_violation(NormL1(T(0.5)), v[1:3], bx, 1)
  @test_throws BoundsError kkt_violation(NormL1(T(0.5)), v, bx, 3)
  # an error of validation comes before the refusal at a nonzero point (the message tells which)
  nz = T[1, 2, 3, 4]
  s = Scaled(NormL1(T(0.5)), T(2))
  for f in (s, GroupNormL2(T(1), [[1, 2], [3, 4]], T[1, 1]))
    @test occursin("requires real entries", msg(() -> kkt_violation(f, Complex{T}.(v), nz, 1)))
    @test occursin("requires real entries", msg(() -> kkt_violation(f, v, Complex{T}.(nz), 1)))
    @test occursin("offset arrays", msg(() -> kkt_violation(f, OffsetArray(v, 0:3), nz, 1)))
    @test occursin("offset arrays", msg(() -> kkt_violation(f, v, OffsetArray(nz, 0:3), 1)))
    @test_throws DimensionMismatch kkt_violation(f, v[1:3], nz, 1)
    @test_throws DimensionMismatch kkt_violation(f, v, nz[1:3], 1)
  end
  gl = GroupNormL2(T(1), [[1, 2], [3, 4]], T[1, 1])
  for layout in ([[1, 3], [2, 4]], [[1], [2], [3], [4]])
    bn = BlockIterate(T, 4, layout)
    bn[1] = 1
    @test_throws DimensionMismatch kkt_violation(gl, v, bn, 1)
  end
end

# ---- 7. storage owners ----------------------------------------------------------------------------------------
@testset "$T SparseIterate" for T in TYPES
  rng = MersenneTwister(977)
  n = 9
  w = weights(rng, T, n) .+ T(0.1)
  fs = (NormL1(T(0.3), w), ElasticNet(T(0.3), T(0.5), w), ZeroFunction(), LinearFunction(randn(rng, T, n)),
        SqrNormL2(T(0.5), randn(rng, T, n)))
  x = SparseIterate(T, n)
  x[2] = T(1.5); x[5] = T(-0.5); x[7] = T(1); x[7] = T(0)            # stored zero at 7, unstored zeros elsewhere
  v = randn(rng, T, n)
  v[2] = T(1)                                       # the mismatch |1 + c| differs from the screen |1| - c at the nonzero entry 2
  d = Vector(x)
  for f in fs, i in 1:n
    @test same(kkt_violation(f, v, x, i), kkt_violation(f, v, d, i))
  end
  @test isactive(x, 7) && !isactive(x, 1)
  @test kkt_violation(NormL1(T(0.3), w), v, x, 7) === kkt_violation(NormL1(T(0.3), w), v, 7)     # stored zero: the screen
  @test kkt_violation(NormL1(T(0.3), w), v, x, 2) !== kkt_violation(NormL1(T(0.3), w), v, 2)     # nonzero: the mismatch
  # the use of the historical active-set method: coordinates that are inactive give the screen
  for f in fs, i in setdiff(1:n, [2, 5])
    @test kkt_violation(f, v, x, i) === kkt_violation(f, v, i)
  end
end

@testset "$T BlockIterate" for T in TYPES
  rng = MersenneTwister(978)
  groups = [[1, 3, 5], [2, 4], [6]]
  w = weights(rng, T, 6) .+ T(0.1)
  fs = (NormL1(T(0.3), w), ElasticNet(T(0.3), T(0.5), w), ZeroFunction(), LinearFunction(randn(rng, T, 6)),
        SqrNormL2(T(0.5), randn(rng, T, 6)))
  x = BlockIterate(T, 6, groups)
  x[4] = T(-2)                                    # group 2 is mixed, group 1 and group 3 are zero
  v = randn(rng, T, 6)
  d = Vector(x)
  for f in fs, k in 1:3
    local_ = [kkt_violation(f, v, d, i) for i in groups[k]]
    @test same(kkt_violation(f, v, x, k), maximum(local_))
  end
  f = NormL1(T(0.3), w)
  zero_group = maximum(abs(v[i]) - T(0.3) * w[i] for i in groups[1])
  @test kkt_violation(f, v, x, 1) === zero_group
  @test kkt_violation(f, v, x, 3) === abs(v[6]) - T(0.3) * w[6]            # signed: the group is zero
  # a mixed group: the largest entry score, the nonzero entry's score being nonnegative
  @test kkt_violation(f, v, x, 2) === max(abs(v[2]) - T(0.3) * w[2], abs(v[4] - T(0.3) * w[4]))
  @test kkt_violation(f, v, x, 2) >= 0
  @test same(kkt_violation(f, v, x), kkt_violation(f, v, d))
  @test_throws BoundsError kkt_violation(f, v, x, 4)
end

@testset "$T SymmetricSparseIterate" for T in TYPES
  rng = MersenneTwister(979)
  p = 3
  M = T[0 1.5 0; 1.5 0 0; 0 0.25 -2]
  M = (M + M') / 2
  x = SymmetricSparseIterate(M)
  G = randn(rng, T, p, p)
  W = randn(rng, T, p, p) .^ 2
  W = (W + W') / 2
  Y = randn(rng, T, p, p)
  Y = (Y + Y') / 2
  C = randn(rng, T, p, p)
  C = (C + C') / 2
  fs = (NormL1(T(0.3), W), NormL1(T(0.3)), ElasticNet(T(0.3), T(0.5), W), ZeroFunction(), LinearFunction(C),
        SqrNormL2(T(0.5), Y), SqrNormL2(T(0.5)))
  for f in fs, k in 1:ncoordinates(x)
    r, c = tril_cartesian(p, k)
    j = (c - 1) * p + r
    pair = (G[r, c] + G[c, r]) / 2
    for gradient_ in (G, vec(G))
      s = kkt_violation(f, gradient_, x, k)
      @test s isa T
      # the stored value of coordinate k, the symmetric gradient pair, the parameters at the lower position
      @test within(s, f, pair, x[r, c], j; k = 8)
    end
  end
  # the refusal of the full point form
  for f in fs
    @test_throws ArgumentError kkt_violation(f, G, x)
    @test_throws ArgumentError kkt_violation(f, vec(G), x)
  end
  @test_throws ArgumentError kkt_violation(NormL1(T(0.3)), G, x)
  @test occursin("SymmetricSparseIterate", msg(() -> kkt_violation(NormL1(T(0.3)), G, x)))
  @test_throws MethodError kkt_violation(NormLinf(T(1)), G, x)
end

# ---- 8. extension by a function defined outside the package ----------------------------------------------------
@testset "functions defined outside the package" begin
  g = [3.0, -2.0, 0.5, 4.0]
  x = [0.0, 1.0, 0.0, -1.0]
  f = OnlyScreen()
  # only the three-argument method: its value at a zero coordinate, an error that names the method at a nonzero one
  @test kkt_violation(f, g, x, 1) === kkt_violation(f, g, 1)
  @test kkt_violation(f, g, x, 3) === kkt_violation(f, g, 3)
  @test_throws ArgumentError kkt_violation(f, g, x, 2)
  @test occursin("kkt_violation(f::OnlyScreen", msg(() -> kkt_violation(f, g, x, 4)))
  @test_throws MethodError kkt_violation(f, g, x)
  b = BlockIterate(4, [[1, 3], [2, 4]])
  b[2] = 1.0
  @test kkt_violation(f, g, BlockIterate(4, [[1, 3], [2, 4]]), 1) === max(kkt_violation(f, g, 1), kkt_violation(f, g, 3))
  @test_throws ArgumentError kkt_violation(f, g, b, 2)
  @test occursin("kkt_violation(f::OnlyScreen", msg(() -> kkt_violation(f, g, b, 2)))
  sx = SparseIterate(4)
  sx[2] = 1.0
  @test kkt_violation(f, g, sx, 1) === kkt_violation(f, g, 1)
  @test_throws ArgumentError kkt_violation(f, g, sx, 2)
  S = SymmetricSparseIterate([0.0 1.0; 1.0 0.0])
  GS = [1.0 2.0; 3.0 4.0]
  @test_throws ArgumentError kkt_violation(f, GS, S, 2)                # coordinate (2, 1), stored nonzero
  @test kkt_violation(f, GS, S, 1) === kkt_violation(f, GS, 1)         # coordinate (1, 1), unstored zero
  # no three-argument method: a MethodError at a zero and at a nonzero coordinate
  @test_throws MethodError kkt_violation(NoScreen(), g, x, 1)
  @test_throws MethodError kkt_violation(NoScreen(), g, x, 2)
  @test_throws MethodError kkt_violation(NoScreen(), g, b, 2)
  # its own methods, with the documented signatures, are used as defined, on arrays and on a BlockIterate
  o = OwnScores()
  @test kkt_violation(o, g, x, 2) === 10 * g[2] + x[2]
  @test kkt_violation(o, g, x) === sum(abs, g) + sum(abs, x)
  @test kkt_violation(o, g, b, 2) === g[2] + g[4]
  # a function of the package without a score
  @test_throws MethodError kkt_violation(NormLinf(1.0), g, x, 1)
  @test_throws MethodError kkt_violation(NormL2(1.0), g, x, 1)
  @test_throws MethodError kkt_violation(NormLinf(1.0), g, x)
  @test_throws MethodError kkt_violation(GroupNormL2(1.0, [[1, 2], [3, 4]], [1.0, 1.0]), g, x)
  @test_throws MethodError kkt_violation(Scaled(NormL1(1.0), 2.0), g, x)
  # no ambiguity inside the package for the new methods
  @test isempty(Test.detect_ambiguities(ProximalBase))
end

# ---- 9. the guards of GroupNormL2 and Scaled -------------------------------------------------------------------
@testset "$T GroupNormL2 and Scaled at nonzero points" for T in TYPES
  rng = MersenneTwister(980)
  groups = [[2, 7, 3], [1, 5], [4]]                       # entries 6 and 8 are in no group
  gl = GroupNormL2(T(0.5), groups, T[1, 2, 3])
  v = randn(rng, T, 8)
  x = zeros(T, 8)
  # a zero group: the group screen; an entry in no group: |vᵢ|, also when it is nonzero
  @test kkt_violation(gl, v, x, 7) === kkt_violation(gl, v, x, 2)
  @test kkt_violation(gl, v, x, 7) ≈ norm(v[[2, 7, 3]]) - T(0.5)
  x[6] = 3
  @test kkt_violation(gl, v, x, 6) === abs(v[6]) && kkt_violation(gl, v, x, 8) === abs(v[8])
  # a group with a nonzero entry
  x[2] = 1
  for i in (2, 7, 3)
    @test_throws ArgumentError kkt_violation(gl, v, x, i)
  end
  @test kkt_violation(gl, v, x, 1) ≈ norm(v[[1, 5]]) - 2 * T(0.5)
  @test occursin("nonzero", msg(() -> kkt_violation(gl, v, x, 7)))
  bx = BlockIterate(T, 8, [[2, 7, 3], [1, 5], [4], [6, 8]])
  gb = GroupNormL2(T(0.5), [[2, 7, 3], [1, 5], [4], [6, 8]], T[1, 2, 3, 4])
  @test kkt_violation(gb, v, bx, 2) ≈ norm(v[[1, 5]]) - 2 * T(0.5)
  bx[3] = 1
  @test_throws ArgumentError kkt_violation(gb, v, bx, 1)
  @test kkt_violation(gb, v, bx, 2) ≈ norm(v[[1, 5]]) - 2 * T(0.5)
  @test_throws ArgumentError kkt_violation(Scaled(gb, T(2)), v, bx, 1)
  @test_throws ArgumentError kkt_violation(Scaled(gl, T(2)), v, x, 7)
  @test kkt_violation(Scaled(gl, T(2)), v, x, 1) === kkt_violation(Scaled(gl, T(2)), v, zeros(T, 8), 1)

  # Scaled over the five functions: the screen at a zero coordinate, an ArgumentError at a nonzero one
  n = 6
  x, v = data(rng, T, n)
  x[1], x[2] = 0, 1
  w = weights(rng, T, n)
  leaves = (NormL1(T(0.3), w), ElasticNet(T(0.3), T(0.5), w), ZeroFunction(), LinearFunction(randn(rng, T, n)),
            SqrNormL2(T(0.5), randn(rng, T, n)), SqrNormL2(T(0.5)))
  for leaf in leaves
    for s in (Scaled(leaf, T(2)), Scaled(Scaled(leaf, T(2)), T(3)), Scaled(Scaled(leaf, T(0.5)), T(1e-3)))
      @test kkt_violation(s, v, x, 1) === kkt_violation(s, v, 1)
      @test_throws ArgumentError kkt_violation(s, v, x, 2)
      bx = BlockIterate(copy(x), [[1, 3], [2, 4], [5, 6]])
      @test_throws ArgumentError kkt_violation(s, v, bx, 2)
      sx = SparseIterate(T, n)
      sx[2] = 1
      @test_throws ArgumentError kkt_violation(s, v, sx, 2)
      @test kkt_violation(s, v, sx, 1) === kkt_violation(s, v, 1)
    end
  end
  S = SymmetricSparseIterate(T[0 1; 1 0])
  GS = T[1 2; 3 4]
  for s in (Scaled(NormL1(T(0.3)), T(2)), Scaled(Scaled(SqrNormL2(T(0.5)), T(2)), T(3)))
    @test_throws ArgumentError kkt_violation(s, GS, S, 2)
    @test kkt_violation(s, GS, S, 1) === kkt_violation(s, GS, 1)
  end
  # a function without a score stays a MethodError under a scale, at a nonzero point too
  X, Y, β = randn(rng, T, 5, 3), randn(rng, T, 5), randn(rng, T, 3)
  sq = SqrtLeastSquaresLoss(Y, X)
  @test_throws MethodError kkt_violation(Scaled(sq, T(0.5)), zeros(T, 3), β, 1)
  @test_throws MethodError kkt_violation(sq, zeros(T, 3), β, 1)
end

# ---- 10. no mutation, read-only inputs, aliasing, inference and allocation ---------------------------------------
@testset "$T no mutation, read-only inputs and aliasing" for T in TYPES
  rng = MersenneTwister(981)
  n = 40
  x, v = data(rng, T, n)
  for f in functions(rng, T, n)
    v0, x0 = copy(v), copy(x)
    pars = [copy(getfield(f, k)) for k in fieldnames(typeof(f)) if getfield(f, k) isa AbstractArray]
    kkt_violation(f, v, x); kkt_violation(f, v, x, 3)
    @test v == v0 && x == x0
    @test pars == [copy(getfield(f, k)) for k in fieldnames(typeof(f)) if getfield(f, k) isa AbstractArray]
    # a read-only point and gradient: ranges
    xr, vr = range(-one(T), 2one(T), length = n), range(T(0.5), T(1.5), length = n)
    @test eltype(xr) === T && eltype(vr) === T
    @test kkt_violation(f, vr, xr) === kkt_violation(f, collect(vr), collect(xr))
    @test kkt_violation(f, vr, xr, 4) === kkt_violation(f, collect(vr), collect(xr), 4)
    # a view, a transposed matrix
    @test kkt_violation(f, view(v, 1:n), view(x, 1:n)) === kkt_violation(f, v, x)
    # v and x are the same array
    @test kkt_violation(f, x, x) === max(zero(T), maximum(kkt_violation(f, x, x, i) for i in 1:n))
  end
end

@testset "$T inference and allocations" for T in TYPES
  rng = MersenneTwister(982)
  n = 100
  w = weights(rng, T, n)
  x, v = data(rng, T, n)
  vn, xn = copy(v), copy(x)
  vn[40] = NaN                      # the pass by entry
  xi = copy(x)
  xi[60] = Inf
  for f in (NormL1(T(0.3)), NormL1(T(0.3), w), ElasticNet(T(0.3), T(0.5)), ElasticNet(T(0.3), T(0.5), w), ZeroFunction(),
            LinearFunction(randn(rng, T, n)), SqrNormL2(T(0.5)), SqrNormL2(T(0.5), randn(rng, T, n)),
            NormL1(T(0.3), fill(T(Inf), n)), SqrNormL2(T(Inf)))
    @inferred kkt_violation(f, v, x)
    @inferred kkt_violation(f, v, x, 5)
    @inferred kkt_violation(f, v, x, 1)
    @inferred kkt_violation(f, v, 5)
    for (vv, xx) in ((v, x), (vn, xn), (v, xi))
      @test allocs(kkt_violation, f, vv, xx) == 0
      @test allocs(kkt_violation, f, vv, xx, 5) == 0
    end
    @test allocs(kkt_violation, f, v, view(x, 1:n)) == 0
  end
  sx = SparseIterate(T, n)
  sx[3] = 1
  bx = BlockIterate(copy(x), [collect(1:2:n), collect(2:2:n)])
  f = NormL1(T(0.3))
  @test allocs(kkt_violation, f, v, sx) == 0
  @test allocs(kkt_violation, f, v, bx) == 0
  @test allocs(kkt_violation, f, v, sx, 3) == 0
  @test allocs(kkt_violation, f, v, bx, 2) == 0
  @test allocs(kkt_violation, f, Float64.(v), x) == 0
end

end
