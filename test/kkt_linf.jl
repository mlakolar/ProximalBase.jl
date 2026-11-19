module KktLinfTest
# The full-point form of kkt_violation (#970) for NormLinf, and for Scaled over it:
#   kkt_violation(f, v, x) = the Euclidean distance from -v to the subdifferential of f = λ max|xᵢ| at x,
# in W = float(promote_type(eltype(v), eltype(x))). With J = {i : |xᵢ| == max|x|} (equality after conversion to W) and
# uᵢ = -sign(xᵢ) vᵢ,
#   x = 0:   0 when Σ|vᵢ| ≤ λ, and otherwise sqrt(Σ min(|vᵢ|, τ)²) with Σ (|vᵢ| - τ)₊ = λ (the norm of prox(NormLinf(λ), v, 1));
#   x ≠ 0:   sqrt(Σ_{i∉J} vᵢ² + Σ_{i∈J} min(uᵢ, τ)²) with Σ_{i∈J} (uᵢ - τ)₊ = λ; τ can be negative;
#   λ = 0:   ‖v‖₂;   λ = Inf: 0 at x = 0 and Inf at x ≠ 0.
# The forms for an entry or a group throw an ArgumentError.
# The references are BigFloat evaluations of these definitions (sort-based levels of the simplex and of the ball); none calls
# a ProximalBase kernel. Besides them, the zero point is compared with the norm of the proximal point, stationary points
# are built from the proximal operator, and the distance is checked as the steepest descent rate of v + ∂f at x.
# A value is accepted within (2n + √n + 5) u (‖v‖₂ + λ) + 4 n nextfloat(zero(W)) of its reference (linf_bound), u = eps(W)/2, where the reference
# is in the range of W. This test tolerance uses the ordinary first-order estimate and an additive subnormal term. A reference above floatmax(W) must
# overflow to Inf, and the band of relative width 100 eps(W) around floatmax(W) is not decided.
#
# Which tests fail at the base (13891ccf): the checks of valid inputs, of the new NormLinf methods and of the refusals that they
# throw, which are MethodErrors at the base. Checks of argument validation that the generic public methods perform before they
# reach a NormLinf method (complex entries, offset axes, lengths) can pass at the base. The tests of Scaled over
# SqrtLeastSquaresLoss and the other functions pass there.
using Test, ProximalBase, LinearAlgebra, SparseArrays, Random, OffsetArrays
using ..TestUtils
const PB = ProximalBase

const TYPES = (Float32, Float64)
const PREC = 1024
bigprec(f) = setprecision(f, BigFloat, PREC)

msg(f) = try f(); "" catch e; sprint(showerror, e) end
sgn(x) = x > 0 ? one(x) : x < 0 ? -one(x) : zero(x)

# ---- references --------------------------------------------------------------------------------------------------
# the level τ with Σ (uᵢ - τ)₊ = c for c ≥ 0, by sorting
function level(u::Vector{BigFloat}, c::BigFloat)
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
# the distance from -V to the subdifferential at X of c max|xᵢ|, and the nearest subgradient (V, X are BigFloat vectors of
# finite entries, c ≥ 0 finite)
function refsub(c::BigFloat, V::Vector{BigFloat}, X::Vector{BigFloat})
  m = maximum(abs, X)
  if iszero(c)
    return sqrt(sum(abs2, V)), zero(V)
  elseif iszero(m)
    sum(abs, V) <= c && return zero(BigFloat), -V
    τ = level(abs.(V), c)
    s = [-sgn(t) * max(abs(t) - τ, 0) for t in V]
    return sqrt(sum(abs2, -V - s)), s
  end
  J = abs.(X) .== m
  u = [-sgn(X[i]) * V[i] for i in eachindex(V)]
  τ = level(u[J], c)
  s = [J[i] ? sgn(X[i]) * max(u[i] - τ, 0) : zero(BigFloat) for i in eachindex(V)]
  sqrt(sum(abs2, -V - s)), s
end
refbig(c, V, X) = first(refsub(c, V, X))
# the value for the arrays as the kernel reads them (converted to W) and the level λ
function refvalue(λ, v, x)
  W = float(promote_type(eltype(v), eltype(x)))
  bigprec() do
    refbig(BigFloat(λ), [big(W(a)) for a in vec(v)], [big(W(a)) for a in vec(x)])
  end
end
# which branch of the face the inputs take: :zero for x = 0, :closed when the entries of J are all active, :threshold otherwise
function branch(λ, v, x)
  W = float(promote_type(eltype(v), eltype(x)))
  bigprec() do
    X = [big(W(a)) for a in x]
    V = [big(W(a)) for a in v]
    m = maximum(abs, X)
    iszero(m) && return :zero
    u = [-sgn(X[i]) * V[i] for i in eachindex(V) if abs(X[i]) == m]
    (sum(u) - big(λ)) / length(u) <= minimum(u) ? :closed : :threshold
  end
end
# the constant of the accuracy bound of the ordinary path, in units of u (‖v‖₂ + λ)
linf_bound(n) = 2n + sqrt(n) + 5
# the accuracy bound of the ordinary path, plus the subnormal spacing of the type
function tol(n, ::Type{W}, v, λ; slack = 1) where {W}
  u = eps(W) / 2
  bigprec() do
    slack * (big(linf_bound(n)) * big(u) * (sqrt(sum(abs2, [big(W(a)) for a in v])) + big(λ)) + 4n * big(nextfloat(zero(W))))
  end
end
# r is the value for the level λ at v and x: of the type W, within the bound of the reference where that is in range, and Inf
# where the reference is above floatmax(W)
function ok(r, λ, v, x; slack = 1)
  W = float(promote_type(eltype(v), eltype(x)))
  r isa W || return false
  bound = tol(length(v), W, v, λ; slack)
  ref = refvalue(λ, v, x)
  W === BigFloat && return abs(r - ref) <= bound
  bigprec() do
    lim, e = big(floatmax(W)), 100 * big(eps(W))
    ref > lim * (1 + e) && return r == W(Inf)
    ref > lim * (1 - e) && return true
    isfinite(r) && abs(big(r) - ref) <= bound
  end
end

# ---- data --------------------------------------------------------------------------------------------------------
# the entries of x are replaced by the largest magnitude, with their sign, with probability frac
function tie!(rng, x, frac)
  m = maximum(abs, x)
  for i in eachindex(x)
    rand(rng) < frac && (x[i] = copysign(m, x[i]))
  end
  x
end
# a point x whose face is the first k entries, at magnitude one with the signs sg, and v with uᵢ = u[i] there; the other entries
# have magnitudes below one, and v there is `rest` (zero without it)
function facepoint(::Type{T}, u, sg, n; rest = true) where {T}
  k = length(u)
  x = T[i <= k ? sg[i] : T(0.25 * (i % 4)) for i in 1:n]
  v = T[i <= k ? -sg[i] * u[i] : (rest ? T(0.1 * (i - 3)) : zero(T)) for i in 1:n]
  v, x
end
# the six families of the accuracy sweep: (name, generator (rng, n) -> (λ, v, x))
function families(::Type{T}) where {T}
  (("random, ties, zero points", (rng, n) -> begin
      v = randn(rng, T, n)
      λ = T(exp(4 * randn(rng)))
      rand(rng) < 0.2 && (λ = sum(abs, v) * T(rand(rng)))
      x = rand(rng) < 0.3 ? zeros(T, n) : tie!(rng, randn(rng, T, n), rand(rng))
      (λ, v, x)
    end),
   ("cancellation on the face", (rng, n) -> begin
      v = randn(rng, T, n)
      for i in 2:2:n
        v[i] = -v[i - 1] * (1 + T(1e-3) * randn(rng, T))
      end
      (T(exp(3 * randn(rng))) * T(1e-3), v, ones(T, n))
    end),
   ("λ far above ‖v‖₁, face", (rng, n) -> begin
      v = randn(rng, T, n)
      (T(1e6) * sum(abs, v), v, tie!(rng, randn(rng, T, n), 0.5))
    end),
   ("huge entries off the face", (rng, n) -> begin
      v = randn(rng, T, n)
      x = tie!(rng, randn(rng, T, n), 0.2)
      m = maximum(abs, x)
      for i in 1:n
        abs(x[i]) == m || (v[i] *= T(1e12))
      end
      (T(exp(randn(rng))), v, x)
    end),
   ("one entry on the face", (rng, n) -> begin
      v = randn(rng, T, n)
      (T(exp(2 * randn(rng))), v, randn(rng, T, n))
    end),
   ("all on the face, tiny λ", (rng, n) -> begin
      v = randn(rng, T, n)
      (sum(abs, v) * T(1e-6), v, T[rand(rng) < 0.5 ? 1 : -1 for _ in 1:n])
    end))
end

# ---- 1. the worked examples ----------------------------------------------------------------------------------------
@testset "$T worked examples" for T in TYPES
  f = NormLinf(one(T))
  near(r, a; k = 16) = r isa T && isapprox(r, T(a); rtol = k * eps(T))
  # the zero point inside the ball, and outside it
  @test kkt_violation(f, T[0.3, -0.5], T[0, 0]) === zero(T)
  r = kkt_violation(f, T[3, 1], T[0, 0])
  @test near(r, sqrt(T(5)))
  @test isapprox(r, norm(prox(f, T[3, 1], 1)); rtol = 16 * eps(T))
  # one entry on the face: |v₁ + λ sign(x₁)| there, v₂ off it
  @test near(kkt_violation(f, T[-0.4, 0.3], T[2, 1]), sqrt(T(0.45)))
  @test near(kkt_violation(f, T[0.4, 0.3], T[-2, 1]), sqrt(T(0.45)); k = 32)          # x₁ < 0: s₁ = -1, |0.4 - 1|
  # a stationary point on a face of two: the subgradient (0.25, -0.75, 0) cancels v
  @test kkt_violation(f, T[-0.25, 0.75, 0], T[2, -2, 1]) === zero(T)
  # a face of two, the closed form (all entries active) and the threshold branch
  @test branch(1, T[-3, 1], T[2, -2]) === :threshold
  @test near(kkt_violation(f, T[-3, 1], T[2, -2]), sqrt(T(5)))
  @test branch(10, T[-3, 1], T[2, -2]) === :closed
  @test near(kkt_violation(NormLinf(T(10)), T[-3, 1], T[2, -2]), sqrt(T(18)))
  # λ = 0 gives the 2-norm of v at any x; λ = Inf gives 0 at x = 0 and Inf elsewhere
  for x in (T[0, 0, 0], T[1, -1, 0.5], T[0, 2, 0])
    @test near(kkt_violation(NormLinf(zero(T)), T[3, -4, 12], x), T(13))
    @test kkt_violation(NormLinf(T(Inf)), T[3, -4, 12], x) === (all(iszero, x) ? zero(T) : T(Inf))
  end
  @test near(kkt_violation(NormLinf(0), T[3, -4, 12], T[1, 0, 0]), T(13))                     # an integer level
  # a tie broken by one ulp: the face of one entry gives a different value than the face of two
  tied = kkt_violation(f, T[-0.4, 0.3], T[2, 2])
  broken = kkt_violation(f, T[-0.4, 0.3], T[2, prevfloat(T(2))])
  @test ok(tied, 1, T[-0.4, 0.3], T[2, 2]) && ok(broken, 1, T[-0.4, 0.3], T[2, prevfloat(T(2))])
  @test branch(1, T[-0.4, 0.3], T[2, 2]) === :closed
  @test near(tied, T(0.45) * sqrt(T(2)); k = 32)                                               # τ = (0.1 - 1)/2: both entries at -0.45
  @test tied != broken
  @test near(broken, sqrt(T(0.45)); k = 32)
  # empty arguments
  @test kkt_violation(f, T[], T[]) === zero(T)
end

# ---- 2. the references, on the six families ------------------------------------------------------------------------------
@testset "$T the six families against BigFloat" for T in TYPES
  rng = MersenneTwister(970)
  for (name, gen) in families(T), n in (1, 2, 3, 5, 17, 64, 200)
    bad = 0
    for _ in 1:(n <= 17 ? 60 : n <= 64 ? 25 : 6)
      λ, v, x = gen(rng, n)
      bad += !ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
    end
    @test bad == 0
  end
end

# ---- 3. the zero point against the proximal point, stationary points from the proximal operator ------------------------------
@testset "$T the proximal operator" for T in TYPES
  rng = MersenneTwister(971)
  # at x = 0 the score is the norm of the proximal point of NormLinf(λ) at v with the step 1
  for n in (1, 2, 7, 64, 1000), frac in (0.001, 0.3, 0.9, 1.5)
    v = randn(rng, T, n)
    λ = T(frac) * sum(abs, v)
    r = kkt_violation(NormLinf(λ), v, zeros(T, n))
    @test abs(r - norm(prox(NormLinf(λ), v, one(T)))) <= 32 * (n + 8) * eps(T) * (norm(v) + λ)
    frac >= 1 && @test r === zero(T)
  end
  # a stationary point y = prox(z) with v = (y - z)/γ: the score at y is of the size of the rounding of the prox, whose level has
  # an absolute error of order n eps max|zᵢ| and enters v by the division by γ (here γ = 1 or 1/2). The bound in units of
  # eps (‖v‖₂ + λ) max(1, max|zᵢ|) is 16 (n + 8)
  for n in (2, 5, 40, 300), γ in (T(1), T(0.5)), kind in (:one, :several, :all)
    for _ in 1:5
      z = randn(rng, T, n)
      λ = T(0.3) * sum(abs, z)
      if kind === :one
        z[1] = T(50)
        λ = T(0.5)
      elseif kind === :several
        z[1:min(3, n)] .= T(30) .+ T(0.001) .* randn(rng, T, min(3, n))
        λ = T(2)
      else
        z .= sign.(z) .* (T(1) .+ T(0.01) .* abs.(randn(rng, T, n)))
        λ = 2 * sum(abs(a) - minimum(abs, z) for a in z) + T(0.5)
      end
      f = NormLinf(λ)
      y = prox(f, z, γ)
      v = (y .- z) ./ γ
      r = kkt_violation(f, v, y)
      @test r <= 16 * (n + 8) * eps(T) * (norm(v) + λ) * max(1, maximum(abs, z))
      kind === :all && @test count(==(maximum(abs, y)), abs.(y)) == n                         # all entries tie, as the clip stores one level
    end
  end
end

# ---- 4. the steepest descent rate ---------------------------------------------------------------------------------------
# The distance from -v to the subdifferential is the largest rate of descent of v·d + f over ‖d‖₂ ≤ 1:
# -(⟨v, d⟩ + λ max_{i∈J} sign(xᵢ) dᵢ), with J the face (and λ max|dᵢ| at x = 0). No sampled unit direction exceeds the score by
# more than the bound; when the distance is positive, the unit direction (-v - s)/dist from the nearest subgradient s attains it.
@testset "$T steepest descent" for T in TYPES
  rng = MersenneTwister(972)
  rate(c, V, X, d) = begin
    m = maximum(abs, X)
    fd = iszero(m) ? c * maximum(abs, d) : c * maximum(sgn(X[i]) * d[i] for i in eachindex(X) if abs(X[i]) == m)
    -(dot(V, d) + fd)
  end
  for (name, gen) in families(T), n in (2, 5, 13)
    for _ in 1:4
      λ, v, x = gen(rng, n)
      r = kkt_violation(NormLinf(λ), v, x)
      bigprec() do
        V, X, c = big.(v), big.(x), big(λ)
        T0 = tol(n, T, v, λ)
        worst = -big(Inf)
        for _ in 1:100
          d = randn(rng, n)
          d ./= norm(d)
          worst = max(worst, rate(c, V, X, big.(d)))
        end
        @test worst <= big(r) + T0
        D, s = refsub(c, V, X)
        if D > 0
          d = (-V - s) / D
          @test abs(rate(c, V, X, d) - D) <= 1e-60 * (D + c)                              # the identity, at the nearest subgradient
          @test abs(big(r) - rate(c, V, X, d)) <= T0
        end
      end
    end
  end
end

# ---- 5. faces, both branches ------------------------------------------------------------------------------------------------
@testset "$T faces" for T in TYPES
  rng = MersenneTwister(973)
  # one entry, several, all; mixed signs; the intended branch is checked on the inputs, and the value against the reference
  cases = [
    # (u on the face, signs, λ, n, branch)
    (T[0.7], T[1], T(0.2), 4, :closed),
    (T[0.7], T[-1], T(3), 4, :closed),
    (T[3, 3.2, 3.1], T[1, -1, 1], T(1), 5, :closed),                           # (S - λ)/3 = 2.77 ≤ μ = 3
    (T[3, 4, 5], T[1, 1, -1], T(1), 5, :threshold),                            # (S - λ)/3 = 3.67 > μ = 3
    (T[3, 4, 5], T[1, 1, -1], T(100), 5, :closed),
    (T[-5, 1, 1, 1, 1, 1, 1], T[1, 1, -1, 1, -1, 1, 1], T(0.3), 9, :threshold),  # the minimum entry is not active
    (T[2, -1, 0.5, 3, -4], T[1, -1, 1, -1, 1], T(2.5), 5, :threshold),
    (T[2, -1, 0.5, 3, -4], T[1, -1, 1, -1, 1], T(40), 5, :closed),
  ]
  for (u, sg, λ, n, br) in cases, rest in (true, false)
    v, x = facepoint(T, u, sg, n; rest)
    @test branch(λ, v, x) === br
    @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
  end
  # random faces of several sizes at levels on both sides of the branch
  for n in (3, 8, 33), frac in (1e-6, 0.01, 0.3, 0.8, 1.2, 10.0)
    for _ in 1:10
      x = tie!(rng, randn(rng, T, n), 0.6)
      v = randn(rng, T, n)
      λ = T(frac) * sum(abs, v)
      @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
    end
  end
  # a point with mixed signs on the face, and the sign convention: the score at x and at -x with v and -v is the same
  x = T[2, -2, 2, -2, 1]
  v = T[0.3, 0.5, -0.1, 0.9, 0.4]
  λ = T(0.7)
  @test kkt_violation(NormLinf(λ), v, x) === kkt_violation(NormLinf(λ), -v, -x)
  @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
  # an entry that is the face of one, tied with an entry one ulp below
  x = T[3, prevfloat(T(3)), 1]
  v = T[-1, -0.9, 0.25]
  @test branch(T(0.5), v, x) === :closed
  @test ok(kkt_violation(NormLinf(T(0.5)), v, x), T(0.5), v, x)
  @test kkt_violation(NormLinf(T(0.5)), v, x) !== kkt_violation(NormLinf(T(0.5)), v, T[3, 3, 1])
end

# ---- 6. x = 0, λ = 0, λ = Inf, signed zeros, subnormals, integers, mixed and BigFloat types -----------------------------------
@testset "$T the zero point and the special levels" for T in TYPES
  rng = MersenneTwister(974)
  v = T[0.25, -0.5, 0.25]
  z = zeros(T, 3)
  @test kkt_violation(NormLinf(one(T)), v, z) === zero(T)                          # on the ball: Σ|vᵢ| = λ, the sum is exact here
  @test kkt_violation(NormLinf(nextfloat(one(T))), v, z) === zero(T)                 # inside
  r = kkt_violation(NormLinf(prevfloat(one(T))), v, z)                               # outside by one ulp
  @test r isa T && r >= 0 && ok(r, prevfloat(one(T)), v, z)
  @test ok(kkt_violation(NormLinf(T(0.5)), v, z), T(0.5), v, z)
  @test kkt_violation(NormLinf(T(0.3)), v, z) ≈ norm(prox(NormLinf(T(0.3)), v, 1)) rtol = 32 * eps(T)
  for n in (1, 4, 50)
    w = randn(rng, T, n)
    @test kkt_violation(NormLinf(zero(T)), w, zeros(T, n)) ≈ norm(w) rtol = 8 * n * eps(T)
    @test kkt_violation(NormLinf(zero(T)), w, randn(rng, T, n)) ≈ norm(w) rtol = 8 * n * eps(T)
    @test kkt_violation(NormLinf(T(Inf)), w, zeros(T, n)) === zero(T)
    @test kkt_violation(NormLinf(T(Inf)), w, ones(T, n)) === T(Inf)
    @test kkt_violation(NormLinf(T(Inf)), w, T[i == 1 ? T(1e-30) : 0 for i in 1:n]) === T(Inf)
  end
  # signed zeros
  @test kkt_violation(NormLinf(one(T)), T[-0.0, 0.0, -0.0], T[-0.0, 0.0, 0.0]) === zero(T)
  @test kkt_violation(NormLinf(one(T)), T[0.2, -0.0], T[-0.0, 0.0]) === zero(T)
  @test ok(kkt_violation(NormLinf(one(T)), T[0.5, -0.0, 0.25], T[-0.0, 2, -0.0]), 1, T[0.5, -0.0, 0.25], T[-0.0, 2, -0.0])
  @test ok(kkt_violation(NormLinf(one(T)), T[3, -0.0, 1], T[-0.0, 0.0, 0.0]), 1, T[3, -0.0, 1], T[-0.0, 0.0, 0.0])
  @test kkt_violation(NormLinf(zero(T)), T[-0.0, 0.0], T[1, 2]) === zero(T)
  # subnormal entries: v and λ, x
  δ = nextfloat(zero(T))
  vs = T[3δ, -7δ, 2δ]
  @test ok(kkt_violation(NormLinf(T(5) * δ), vs, zeros(T, 3)), T(5) * δ, vs, zeros(T, 3))
  @test kkt_violation(NormLinf(T(20) * δ), vs, zeros(T, 3)) === zero(T)
  @test ok(kkt_violation(NormLinf(T(2) * δ), vs, T[δ, 3δ, -3δ]), T(2) * δ, vs, T[δ, 3δ, -3δ])
  @test ok(kkt_violation(NormLinf(T(0.5)), T[1, -1, 0.25], T[δ, 0, 0]), T(0.5), T[1, -1, 0.25], T[δ, 0, 0])   # the face of x is a subnormal entry
  @test ok(kkt_violation(NormLinf(floatmin(T)), T[1, 1] .* floatmin(T), T[0, 0]), floatmin(T), T[1, 1] .* floatmin(T), T[0, 0])
end

@testset "integer, mixed and BigFloat types" begin
  f = NormLinf(1.0)
  # an integer x with typemin(Int): the face is that entry, read as -2^63
  x = [typemin(Int), 5, 7]
  v = [1.0, 2.0, 3.0]
  r = kkt_violation(f, v, x)
  @test r isa Float64 && ok(r, 1.0, v, x)
  @test branch(1.0, v, x) === :closed
  @test r ≈ sqrt(13.0)                                                              # |1 - 1| on the face, then (2, 3) off it
  @test kkt_violation(f, [1, 2, 3], [typemin(Int), 5, 7]) isa Float64
  @test kkt_violation(f, [typemin(Int), 2], [0, 0]) isa Float64
  @test ok(kkt_violation(f, [typemin(Int), 2], [0, 0]), 1.0, [typemin(Int), 2], [0, 0])
  @test ok(kkt_violation(f, [3, 1], [0, 0]), 1.0, [3, 1], [0, 0])
  @test kkt_violation(f, [3, 1], [0, 0]) ≈ sqrt(5.0)
  # the working type of mixed element types
  for (V, X, W) in ((Float32, Float64, Float64), (Float64, Float32, Float64), (Float32, Float32, Float32), (Int, Float32, Float32),
                    (Float32, Int, Float32), (Int8, Int16, Float64), (Int, Int, Float64), (Float64, Float64, Float64))
    v = V[2, -1, 3]
    x = X[1, 1, -1]
    r = kkt_violation(NormLinf(0.5), v, x)
    @test float(promote_type(V, X)) === W
    @test r isa W
    @test ok(r, 0.5, v, x)
    @inferred kkt_violation(NormLinf(0.5), v, x)
  end
  # a level of another type than the entries (Float64 level, Float32 entries; BigFloat level, Float64 entries)
  @test kkt_violation(NormLinf(0.1), Float32[3, 1], Float32[0, 0]) isa Float32
  @test ok(kkt_violation(NormLinf(0.1), Float32[3, 1], Float32[0, 0]), Float32(0.1), Float32[3, 1], Float32[0, 0])
  @test kkt_violation(NormLinf(big"1.5"), [3.0, 1.0], [0.0, 0.0]) isa Float64
  @test ok(kkt_violation(NormLinf(big"1.5"), [3.0, 1.0], [0.0, 0.0]), 1.5, [3.0, 1.0], [0.0, 0.0])
  @test kkt_violation(NormLinf(1), [3.0, 1.0], [0.0, 0.0]) ≈ sqrt(5.0)
  # BigFloat arrays, in the working precision
  setprecision(BigFloat, 256) do
    v = BigFloat[3, 1, 2]
    for x in (zeros(BigFloat, 3), BigFloat[2, -2, 1], BigFloat[0, 0, 5])
      for λ in (BigFloat(1), BigFloat("0.3"), BigFloat(10))
        r = kkt_violation(NormLinf(λ), v, x)
        @test r isa BigFloat
        @test ok(r, λ, v, x)
        @inferred kkt_violation(NormLinf(λ), v, x)
      end
    end
    @test kkt_violation(NormLinf(BigFloat(1)), BigFloat[3, 1], zeros(BigFloat, 2)) ≈ sqrt(big(5)) rtol = big(2)^-250
    @test kkt_violation(NormLinf(BigFloat(0)), BigFloat[3, 4], BigFloat[1, 0]) ≈ 5
    @test kkt_violation(NormLinf(BigFloat(Inf)), BigFloat[3, 4], BigFloat[1, 0]) == Inf
    @test isnan(kkt_violation(NormLinf(BigFloat(1)), BigFloat[3, NaN], BigFloat[1, 0]))
    @test kkt_violation(NormLinf(BigFloat(1)), BigFloat[3, 4], BigFloat[Inf, 0]) == Inf
    @test kkt_violation(NormLinf(1.0), [3.0, 1.0], BigFloat[0, 0]) isa BigFloat
  end
end

# Float16 represents the integers up to 2048, so a sum of ones is not rounded below that: small sizes are accepted and their value is within the bound of the type; a
# length n with n + 1 not finite in Float16 (n + 1 ≥ 65520) is refused before any pass
@testset "Float16" begin
  for (v, x, λ, br) in ((Float16[3, 1], zeros(Float16, 2), 1, :zero), (Float16[0.3, -0.5], zeros(Float16, 2), 1, :zero),
                        (Float16[-3, 1], Float16[2, -2], 1, :threshold), (Float16[-3, 1], Float16[2, -2], 10, :closed),
                        (Float16[3, -1, 0.5, 2], Float16[2, 2, -2, 1], 0.5, :threshold),
                        (Float16[5, -1, -1, -1, -1, -1, -1], ones(Float16, 7), 0.3, :threshold))
    @test branch(λ, v, x) === br
    r = kkt_violation(NormLinf(Float16(λ)), v, x)
    @test r isa Float16
    @test ok(r, Float16(λ), v, x)
    @inferred kkt_violation(NormLinf(Float16(λ)), v, x)
  end
  n = 65519
  for (v, x) in ((ones(Float16, n), zeros(Float16, n)), (ones(Float16, n), ones(Float16, n)))
    f = NormLinf(Float16(1))
    @test_throws ArgumentError kkt_violation(f, v, x)
    m = msg(() -> kkt_violation(f, v, x))
    @test occursin("65519", m) && occursin("Float16", m)
    @test_throws ArgumentError kkt_violation(Scaled(f, Float16(2)), v, x)
  end
  @test kkt_violation(NormLinf(Float16(1)), Float16[], Float16[]) === zero(Float16)
end

# ---- 7. NaN and infinite entries ---------------------------------------------------------------------------------------------
# rule 1: a NaN anywhere gives NaN, also with an Inf present; otherwise an Inf gives Inf (also where λ = 0 or the level is Inf)
@testset "$T NaN and Inf" for T in TYPES
  NaNs, Infs = T(NaN), T(Inf)
  for n in (1, 3, 64, 130), pos in unique((1, max(1, n ÷ 2), n))
    for x0 in (zeros(T, n), ones(T, n)), λ in (T(0), T(0.5), T(5), Infs)
      v = ones(T, n)
      f = NormLinf(λ)
      vn, vi, xn, xi = copy(v), copy(v), copy(x0), copy(x0)
      vn[pos] = NaNs; vi[pos] = Infs; xn[pos] = NaNs; xi[pos] = Infs
      @test isnan(kkt_violation(f, vn, x0))
      @test isnan(kkt_violation(f, v, xn))
      @test kkt_violation(f, vi, x0) === Infs
      @test kkt_violation(f, v, xi) === Infs
      @test kkt_violation(f, -vi, x0) === Infs
      @test kkt_violation(f, v, -xi) === Infs
      if n >= 2
        other = pos == 1 ? 2 : 1
        a, b = copy(vi), copy(xi)
        a[other] = NaNs
        @test isnan(kkt_violation(f, a, x0))                                              # a NaN in v beside an Inf in v
        b[other] = NaNs
        @test isnan(kkt_violation(f, v, b))
        @test isnan(kkt_violation(f, vi, xn))                                             # an Inf in v and a NaN in x
        @test isnan(kkt_violation(f, vn, xi))                                             # a NaN in v and an Inf in x
      end
    end
  end
  # λ = Inf at x = 0 with an infinite v gives Inf, not 0, and a finite v gives 0
  @test kkt_violation(NormLinf(Infs), T[1, Infs], zeros(T, 2)) === Infs
  @test kkt_violation(NormLinf(Infs), T[1, 2], zeros(T, 2)) === zero(T)
  @test isnan(kkt_violation(NormLinf(Infs), T[1, NaNs], zeros(T, 2)))
  # a level that overflows the type of the entries is an infinite level there
  if T === Float32
    @test kkt_violation(NormLinf(1e300), Float32[1, 2], Float32[0, 0]) === 0f0
    @test kkt_violation(NormLinf(1e300), Float32[1, 2], Float32[1, 0]) === Inf32
  end
end

# ---- 8. range ---------------------------------------------------------------------------------------------------------------
@testset "$T range" for T in TYPES
  near(r, a; k = 16) = r isa T && isapprox(r, T(a); rtol = k * eps(T))
  emax, emin = exponent(floatmax(T)), exponent(floatmin(T))
  small = ldexp(one(T), emin ÷ 2 - 30)                                       # its square underflows to zero
  # a residual far below the level: the squares underflow, and the value is the residual, not zero
  @test near(kkt_violation(NormLinf(zero(T)), [small], [zero(T)]), small)
  @test near(kkt_violation(NormLinf(one(T)), [-one(T), small], T[1, 0.5]), small)
  @test ok(kkt_violation(NormLinf(one(T)), [-one(T), small], T[1, 0.5]), one(T), [-one(T), small], T[1, 0.5])
  w = fill(small, 7)
  @test near(kkt_violation(NormLinf(zero(T)), w, zeros(T, 7)), sqrt(T(7)) * small)
  @test ok(kkt_violation(NormLinf(small / 4), w, zeros(T, 7)), small / 4, w, zeros(T, 7))
  # a level near floatmax with small entries: the negative closed-form face threshold gives a large residual whose squares
  # require the cold path
  for λ in (floatmax(T), floatmax(T) / 2, floatmax(T) / 4, ldexp(one(T), emax - 5))
    v, x = T[0.5, 0.25], T[1, 1]
    @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
    v, x = T[0.5, 0.25, -3], T[1, -1, 0.5]
    @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
  end
  # a level that makes S - λ overflow although the sum of magnitudes is far below the range limit
  let v = fill(floatmax(T) / (5 * 301), 300), x = ones(T, 300)
    @test ok(kkt_violation(NormLinf(floatmax(T)), v, x), floatmax(T), v, x; slack = 4)
  end
  # a distance above floatmax is Inf: the level and the entry add up
  @test kkt_violation(NormLinf(floatmax(T)), [floatmax(T)], [one(T)]) === T(Inf)
  @test kkt_violation(NormLinf(floatmax(T)), [-floatmax(T)], [one(T)]) === zero(T)
  # v near floatmax, with a sum of magnitudes that overflows: at the zero point and on a face
  big_ = floatmax(T) * T(0.4)
  for (λ, v) in ((T(0.1) * floatmax(T), T[1, 1, 1] * big_), (T(1), T[1, -1, 1] * big_), (T(3), T[1, 1, 1] * big_ * T(0.5) + T[1, 2, 3]),
                 (floatmax(T) * T(0.3), T[1, 0.5, -0.75] * big_))
    for x in (zeros(T, 3), T[1, 1, 1], T[1, -1, 0.5], T[0, 2, 2])
      @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x; slack = 4)
    end
  end
  # entries of the scale of floatmax with a level of the scale of the entries
  v = T[0.9, -0.8, 0.7, 0.2] .* floatmax(T) ./ 8
  for x in (zeros(T, 4), T[1, 1, 1, 1], T[1, -1, 1, 0], T[0, 0, 0, 1])
    for λ in (floatmax(T) / 64, floatmax(T) / 8, floatmax(T) / 3)
      @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x; slack = 4)
    end
  end
  # the same entries, many of them: the squares and the sums are out of range for n above one
  for n in (2, 10, 100)
    v = fill(floatmax(T) / T(n) * T(0.9), n)
    x = tie!(MersenneTwister(n), randn(MersenneTwister(n + 1), T, n), 1.0)
    for λ in (T(1), floatmax(T) / 100)
      @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x; slack = 4)
      @test ok(kkt_violation(NormLinf(λ), v, zeros(T, n)), λ, v, zeros(T, n); slack = 4)
    end
  end
  # entries and level at the bottom of the range
  v = T[1, 2, 3] .* floatmin(T)
  for x in (zeros(T, 3), T[1, 1, 0], T[0, 1, -1])
    for λ in (floatmin(T), floatmin(T) * 3, floatmin(T) / 2)
      @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
    end
  end
  # stationary points give zero, also where the residual is zero and the sum of squares is not normal
  @test kkt_violation(NormLinf(one(T)), T[-0.25, 0.75, 0], T[2, -2, 1]) === zero(T)
  @test kkt_violation(NormLinf(one(T)), T[-1], T[3]) === zero(T)
  @test kkt_violation(NormLinf(T(4)), T[2, -2], T[0, 0]) === zero(T)
  @test kkt_violation(NormLinf(one(T)), zeros(T, 4), T[1, 0, 2, 2]) !== zero(T)
  # the subnormal spacing: an entry of x at the bottom of the range with v of the scale of one
  @test ok(kkt_violation(NormLinf(one(T)), T[1.5, -1, 0.25], T[nextfloat(zero(T)), 0, 0]), one(T), T[1.5, -1, 0.25], T[nextfloat(zero(T)), 0, 0])
end

# ---- 9. other shapes of the arguments --------------------------------------------------------------------------------------------
@testset "$T matrices and iterates" for T in TYPES
  rng = MersenneTwister(975)
  f = NormLinf(T(0.8))
  V, X = randn(rng, T, 3, 4), tie!(rng, randn(rng, T, 3, 4), 0.5)
  @test ok(kkt_violation(f, V, X), T(0.8), V, X)
  @test isapprox(kkt_violation(f, V, X), kkt_violation(f, vec(V), vec(X)); rtol = 16 * eps(T))
  @test isapprox(kkt_violation(f, V, zeros(T, 3, 4)), kkt_violation(f, vec(V), zeros(T, 12)); rtol = 16 * eps(T))
  @test kkt_violation(f, view(V, :, 1:2), view(X, :, 1:2)) isa T
  # a SparseIterate and a BlockIterate are read as one vector
  sx = SparseIterate(T, 8)
  sx[2] = T(1.5); sx[5] = T(-1.5); sx[7] = T(0.5)
  dv = randn(rng, T, 8)
  @test ok(kkt_violation(f, dv, sx), T(0.8), dv, collect(sx))
  @test ok(kkt_violation(f, dv, SparseIterate(T, 8)), T(0.8), dv, zeros(T, 8))
  bx = BlockIterate(T[0, 1.5, 0, -1.5, 0.5, 0], [[1, 2], [3, 4], [5, 6]])
  @test ok(kkt_violation(f, dv[1:6], bx), T(0.8), dv[1:6], collect(bx))
  @test kkt_violation(f, dv[1:6], bx) ≈ kkt_violation(f, dv[1:6], collect(bx)) rtol = 8 * eps(T)
  @test kkt_violation(f, T[], SparseIterate(T, 0)) === zero(T)
  # a SymmetricSparseIterate has no full-point form
  S = SymmetricSparseIterate(T[0 1; 1 0])
  @test_throws ArgumentError kkt_violation(f, ones(T, 2, 2), S)
  @test_throws ArgumentError kkt_violation(f, ones(T, 4), S)
  @test occursin("SymmetricSparseIterate", msg(() -> kkt_violation(f, ones(T, 2, 2), S)))
  @test !occursin("kkt_violation(f, v, x, k)", msg(() -> kkt_violation(f, ones(T, 2, 2), S)))   # the form that NormLinf refuses is not advised
end

# ---- 10. the other forms are refused ----------------------------------------------------------------------------------------------
@testset "$T refusals" for T in TYPES
  f = NormLinf(T(0.5))
  v = T[1, 2, 3, 4]
  x = T[0, 1, 0, -1]
  bx = BlockIterate(copy(x), [[1, 2], [3, 4]])
  S = SymmetricSparseIterate(T[1 2; 2 1])
  GS = T[1 2; 3 4]
  sx = SparseIterate(T, 4)
  sx[2] = T(1)
  forms = (() -> kkt_violation(f, v, 2), () -> kkt_violation(f, v, x, 2), () -> kkt_violation(f, v, bx, 1), () -> kkt_violation(f, v, sx, 2),
           () -> kkt_violation(f, GS, S, 2), () -> kkt_violation(f, v, zeros(T, 4), 3), () -> kkt_violation(f, v, zeros(T, 4), 1))
  for form in forms
    @test_throws ArgumentError form()
    @test occursin("kkt_violation(f, v, x)", msg(form))
    @test occursin("not separable", msg(form))
  end
  @test_throws ArgumentError PB._level_score(f, T(0.5), v, x, 1)
  @test_throws ArgumentError PB._kkt_violation(f, v, x, 1)
  @test_throws ArgumentError PB._kkt_violation(f, v, bx, 1)
  # the arguments are checked as for the other functions: offset axes, complex entries, a length mismatch
  off(a) = OffsetArray(a, 0:length(a)-1)
  cx, cv = Complex{T}.(x), Complex{T}.(v)
  @test_throws ArgumentError kkt_violation(f, cv, x)
  @test_throws ArgumentError kkt_violation(f, v, cx)
  @test occursin("offset arrays", msg(() -> kkt_violation(f, off(v), x)))
  @test occursin("offset arrays", msg(() -> kkt_violation(f, v, off(x))))
  @test_throws DimensionMismatch kkt_violation(f, v[1:3], x)
  @test_throws DimensionMismatch kkt_violation(f, v, x[1:3])
  @test_throws ArgumentError kkt_violation(f, cv, 1)
  @test_throws ArgumentError kkt_violation(f, cv, x, 2)
  @test_throws ArgumentError kkt_violation(f, v, cx, 2)
  @test_throws ArgumentError kkt_violation(f, cv, S)
  # nothing is evaluated for a refused call, and the error of an invalid argument precedes the NaN rule
  @test_throws ArgumentError kkt_violation(f, T[NaN, 1], Complex{T}[1, 2])
end

# ---- 11. nothing is written, nothing is allocated ----------------------------------------------------------------------------------
# the allocation of one call, behind a function barrier, with no warm-up of this function
alloc_first(f, v, x) = @allocated kkt_violation(f, v, x)

@testset "$T no mutation, no allocation, inference" for T in TYPES
  rng = MersenneTwister(976)
  n = 300
  v, x = randn(rng, T, n), tie!(rng, randn(rng, T, n), 0.3)
  λ = T(0.2) * sum(abs, v)
  for (name, w, p, lvl) in (("face, threshold branch", v, x, λ), ("face, closed form", v, x, T(1e4) * λ),
                            ("zero point, threshold", v, zeros(T, n), λ), ("zero point, inside", v, zeros(T, n), T(2) * sum(abs, v)),
                            ("λ = 0", v, x, zero(T)), ("λ = Inf", v, x, T(Inf)),
                            ("NaN", [T(NaN); v[2:end]], x, λ), ("Inf in x", v, [T(Inf); x[2:end]], λ),
                            ("sum overflows", fill(floatmax(T) * T(0.4), n), zeros(T, n), floatmax(T) / 8),
                            ("sum overflows, face", fill(floatmax(T) * T(0.4), n), x, floatmax(T) / 8),
                            ("squares underflow", fill(ldexp(one(T), exponent(floatmin(T)) ÷ 2 - 30), n), x, zero(T)),
                            ("S - λ overflows", fill(floatmax(T) / (5 * (n + 1)), n), ones(T, n), floatmax(T)))
    f = NormLinf(lvl)
    w0, p0 = copy(w), copy(p)
    r = kkt_violation(f, w, p)
    @test r isa T
    @test isequal(w, w0) && isequal(p, p0)                                                   # the call does not write v or x
    @test f.scratch.buf === nothing                                                           # no buffer of the object is written
    @test allocs(kkt_violation, f, w, p) == 0
    @test (@inferred kkt_violation(f, w, p)) isa T
    # a fresh object, a first call on arguments of the same types
    g = NormLinf(lvl)
    @test alloc_first(g, w, p) == 0
    @test g.scratch.buf === nothing
  end
  # the call does not change the object: equal calls give equal values, and the prox of the object still works
  f = NormLinf(λ)
  r1, r2 = kkt_violation(f, v, x), kkt_violation(f, v, x)
  @test r1 === r2
  y = prox(f, v, one(T))
  @test y == prox(NormLinf(λ), v, one(T))
  # a buffer populated by a proximal call that uses it is not written by the score
  let f = NormLinf(T(0.2) * sum(abs, v)), z = copy(v)
    prox!(z, f, z, one(T))
    buf = f.scratch.buf
    @test buf isa Vector{T}
    saved = copy(buf)
    kkt_violation(f, v, x)
    kkt_violation(f, v, zeros(T, n))
    @test f.scratch.buf === buf
    @test isequal(buf, saved)
  end
  # the lazy v/α of the Scaled route
  @test allocs(kkt_violation, Scaled(NormLinf(λ), T(2)), v, x) == 0
  @test allocs(kkt_violation, Scaled(NormLinf(λ), T(2)), v, zeros(T, n)) == 0
end

# ---- 11b. a face with entries far below the level ------------------------------------------------------------------------------------
# One face entry lies far below the level and the others (a bulk of a = n - 1 equal entries) are above it: x = ones(n), u = -v =
# (-1, w, ..., w) with w = fl(1/√a + λ/a). The level is w - λ/a and the distance sqrt(a (w - λ/a)² + 1), from the stored w and λ. The
# level of the face is found on the face values shifted by the first level of the face, a lower bound of the level in exact arithmetic
# (the mean of the face entries minus λ/n); a shift by the smallest face entry instead gives each shifted value the size of |u₁| and a
# level whose error grows with the number of entries. Measured in units of u (‖v‖₂ + λ) with default compilation, --check-bounds=yes and
# optimization level 0 (scalar loops: the LLVM IR of the reductions has no vector instructions at level 0 and has 4 and 8 lanes
# otherwise), on Julia 1.10.13 and 1.13.4, on znver2 and znver3 nodes (and, for n = 200001, on a skylake-avx512 node), for a variant of
# the search with the shift by the smallest entry (μ-shift) and for the search of this file (first-level shift):
#   Float32, n = 2000001, λ = 3:    μ-shift 1.56e6 to 1.57e6,  first-level shift at most 6.3e4 (guard 3e5)
#   Float32, n = 2000001, λ = 0.3:  μ-shift 4.0e6 to 4.1e6,    first-level shift at most 1.6e5 (guard 8e5)
#   Float64, n = 200001, λ = 0.3:   μ-shift 2.8e5 to 1.9e6,    first-level shift at most 1.3e4 (guard 6e4)
#   Float64, n = 200001, λ = 3:     μ-shift 8.4e4 to 3.1e6,    first-level shift at most 4.8e3 (guard 2e4)
#   Float64, n = 2000001, λ = 3:    μ-shift 8.7e6 to 2.2e7,    first-level shift at most 1.0e5 (guard 1e6)
# Each guard is at least 4.2 times the largest error of the first-level shift and at least 4.2 times below the smallest error of the
# μ-shift. The stated bound (about 4e5 for n = 200001 and 4e6 for n = 2000001) does not separate the two in all of these runs, which
# is why the guards are given.
function bulk_error(::Type{W}, a, λ) where {W}
  w = one(W) / sqrt(W(a)) + λ / a
  v = fill(-w, a + 1)
  v[1] = one(W)
  x = ones(W, a + 1)
  r = kkt_violation(NormLinf(λ), v, x)
  bigprec() do
    τ = big(w) - big(λ) / a
    ref = sqrt(a * τ^2 + 1)
    unit = (big(eps(W)) / 2) * (sqrt(a * big(w)^2 + 1) + big(λ))
    r isa W, Float64(abs(big(r) - ref) / unit), linf_bound(a + 1)
  end
end
@testset "a face entry far below the level, equal bulk" begin
  for (W, a, λ, guard) in ((Float32, 2_000_000, 3.0, 3e5), (Float32, 2_000_000, 0.3, 8e5), (Float64, 200_000, 0.3, 6e4),
                           (Float64, 200_000, 3.0, 2e4), (Float64, 2_000_000, 3.0, 1e6))
    isW, units, bound = bulk_error(W, a, W(λ))
    @test isW
    @test units <= bound
    @test units <= guard
  end
end

@testset "$T faces with entries far below the level, randomized" for T in TYPES
  rng = MersenneTwister(978)
  for n in (50, 500, 2000), nlow in (1, 8), ρ in (0.1, 1.0, 10.0), cf in (0.01, 0.3, 3.0), spread in (0.0, 0.3)
    M = T(exp(randn(rng)))
    λ = T(cf) * M
    a = n - nlow
    τt = T(ρ) * M / sqrt(T(a))
    u = T[τt + λ / a * T(1 + spread * randn(rng)) for _ in 1:n]
    for i in eachindex(u)
      u[i] > τt || (u[i] = τt + λ / a)
    end
    for k in 1:nlow
      u[k] = -M * T(0.5 + rand(rng))
    end
    x = T[rand(rng) < 0.5 ? 1 : -1 for _ in 1:n]
    v = T[x[i] > 0 ? -u[i] : u[i] for i in 1:n]
    @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
  end
end

# ---- 11c. a large tie and a level near it ---------------------------------------------------------------------------------------------
# The inputs are a head entry y1 above m equal entries y0 (the tie), with a radius c near the breakpoint y1 - y0, where the exact level
# is at or just below y0 and the first level of all the entries, (Σ y - c)/(m + 1), lies within the summation error of the sum
# of the entries from y0. The sum of the ties is rounded in an order that depends on the compilation mode (and on the loop that the
# compiler chooses), so the first level is above the tie in some modes and below it in others: each input is a check of accuracy in
# either case. A search that returns a level far below the tie gives a negative level and a zero score at the zero point on some
# inputs, and a positive level and an inaccurate score on others; the comparisons with the exact distance cover both. The reference
# is the exact distance in BigFloat from the stored entries and level (the levels of a head and a tie have a closed form; no kernel
# of the package is called). The optional off entries lie off the face (x = 1/2 there, v = off), and enter the distance as they are.
# the exact distance for the head y1 and the tie (y0, m): at the zero point the distance to the ball of radius c (0 inside), on a
# face (all entries of the cluster on the face, u = (y1, y0, ..., y0)) the distance to the simplex, whose level can be negative
function cluster_dist(c, y1, y0, m; face = false, off = ())
  bigprec() do
    C, Y1, Y0 = big(c), big(y1), big(y0)
    !face && Y1 + m * Y0 <= C && return zero(BigFloat)
    τ = C <= Y1 - Y0 ? Y1 - C : (Y1 + m * Y0 - C) / (m + 1)
    sqrt(min(Y1, τ)^2 + m * min(Y0, τ)^2 + sum(a -> big(a)^2, off; init = zero(BigFloat)))
  end
end
# the cluster as arrays: the head first (or last), at the zero point or on the face x = ones (v = -u there), then the off entries
function cluster_arrays(::Type{W}, y1, y0, m; face = false, pos = :first, off = ()) where {W}
  u = fill(W(y0), m + 1)
  u[pos === :first ? 1 : end] = W(y1)
  face || return u, zeros(W, m + 1)
  vo = W[a for a in off]
  vcat(-u, vo), vcat(ones(W, m + 1), fill(W(0.5), length(vo)))
end
# whether r is within slack × the bound (and the subnormal term) of the reference
function cluster_ok(r, c, y1, y0, m, ::Type{W}; face = false, slack = 1, off = ()) where {W}
  r isa W || return false
  ref = cluster_dist(c, y1, y0, m; face, off)
  bigprec() do
    unit = (big(eps(W)) / 2) * (sqrt(big(y1)^2 + m * big(y0)^2 + sum(a -> big(W(a))^2, off; init = zero(BigFloat))) + big(c))
    n = m + 1 + length(off)
    isfinite(r) && abs(big(r) - ref) <= slack * big(linf_bound(n)) * unit + 4 * n * big(nextfloat(zero(W)))
  end
end
const TIE_CASES = ((Float32, 0.2, 0.1, 10^5, 0.3), (Float32, 1.0, 0.1, 10^5, 1.2), (Float32, 0.2, 0.1, 10^4, 0.102),
                   (Float64, 1.8, 0.9, 10^6, 0.9000008), (Float32, 0.2, 0.1, 10^6, 1.0))

@testset "two entries whose first level rounds above the second ($T)" for T in TYPES
  ε = eps(T)
  # y = (3, 1 + 3ε), radius 2 - 2ε: the sum 4 + 3ε rounds up to 4 + 4ε, so the first level 1 + 3ε is above the second entry; the
  # exact level is 1 + 2.5ε
  y = T[3, 1 + 3ε]
  r = T(2) - 2ε
  @test cluster_ok(kkt_violation(NormLinf(r), y, zeros(T, 2)), r, y[1], y[2], 1, T)
  @test cluster_ok(kkt_violation(NormLinf(r), -y, ones(T, 2)), r, y[1], y[2], 1, T; face = true)
end

const OFF_ENTRIES = (0.3, -0.2, 0.7)

@testset "large ties against the exact distance" begin
  for (W, y1, y0, m, λ) in TIE_CASES
    c = W(λ)
    for pos in (:first, :last), (face, off) in ((false, ()), (true, ()), (true, OFF_ENTRIES))
      v, x = cluster_arrays(W, y1, y0, m; face, pos, off)
      r = kkt_violation(NormLinf(c), v, x)
      @test cluster_ok(r, c, W(y1), W(y0), m, W; face, off)
      # through Scaled with a power of two: the level c/2 under the factor 2 is the level c
      slack = 1 + 6 / linf_bound(m + 1 + length(off))
      @test cluster_ok(kkt_violation(Scaled(NormLinf(c / 2), W(2)), v, x), c, W(y1), W(y0), m, W; face, off, slack)
    end
  end
end

@testset "$W the head above 10^5 equal entries, at the zero point and on the face of all entries" for W in TYPES
  # a head 1 above 10^5 entries of 0.009486833 and the radius 0.1 (100001 entries): the distance is 3.1320918539449983 for the
  # stored Float32 entries and radius, and 3.1320918... for Float64 within the bound
  m = 10^5
  c = W(0.1)
  for pos in (:first, :last), face in (false, true)
    v, x = cluster_arrays(W, 1.0, 0.009486833, m; face, pos)
    r = kkt_violation(NormLinf(c), v, x)
    @test cluster_ok(r, c, W(1.0), W(0.009486833), m, W; face)
    @test isapprox(r, 3.13209; atol = 1e-3)
  end
end

@testset "$W large ties out of the ordinary range (the scaled path)" for W in TYPES
  # the cluster (0.2, 0.1 × m) with the radius 0.102, scaled so that the sum of the ties is above floatmax/2 and (m + 1) times the
  # sum is above floatmax/4: the zero point and the face take the scaled path
  m = 10^4
  sc = W(0.8) * floatmax(W) / (m * W(0.1))
  y0, y1, c = W(0.1) * sc, W(0.2) * sc, W(0.102) * sc
  for face in (false, true), pos in (:first, :last)
    v, x = cluster_arrays(W, y1, y0, m; face, pos)
    @test cluster_ok(kkt_violation(NormLinf(c), v, x), c, y1, y0, m, W; face, slack = 4)
  end
  # the head 1 above 10^5 entries of 0.009486833 with the radius 0.1, scaled to entries near floatmax, on the zero point, on the face of
  # all entries and on a face with entries off it
  m = 10^5
  sc = W(0.8) * floatmax(W) / (1 + m * W(0.009486833))
  y0, y1, c = W(0.009486833) * sc, sc, W(0.1) * sc
  for pos in (:first, :last), (face, off) in ((false, ()), (true, ()), (true, ntuple(i -> OFF_ENTRIES[i] * Float64(sc), 3)))
    v, x = cluster_arrays(W, y1, y0, m; face, pos, off)
    @test cluster_ok(kkt_violation(NormLinf(c), v, x), c, y1, y0, m, W; face, off, slack = 4)
  end
end

@testset "$W a grid of clusters around the breakpoint against the exact distance" for W in TYPES
  # the radius is the breakpoint y1 - y0 plus θ times the rounding error of the sum of the entries, read from the sum that the
  # kernel forms
  bad = 0
  for m in (1000, 3000, 10_000, 100_000), y0f in (0.1, 0.3, 1 / 3, 0.9), mult in (2, 10), pos in (:first, :last), θ in (0.05, 0.6, 1.5)
    y0, y1 = W(y0f), W(mult) * W(y0f)
    v, _ = cluster_arrays(W, y1, y0, m; pos)
    s, _ = PB._linf_classify(v, PB._AbsRead{W}(), W)
    serr = bigprec() do
      Float64(big(s) - (big(y1) + m * big(y0)))
    end
    c = bigprec() do
      W(big(y1) - big(y0) + θ * abs(serr))
    end
    for (face, off) in ((false, ()), (true, ()), (true, OFF_ENTRIES))
      u, x = cluster_arrays(W, y1, y0, m; face, pos, off)
      bad += !cluster_ok(kkt_violation(NormLinf(c), u, x), c, y1, y0, m, W; face, off)
    end
  end
  @test bad == 0
end

# ---- 11d. the closed form of a face, certified or not ----------------------------------------------------------------------------
# The closed form of the level of a face is returned when the computed first level plus a bound on its error is at most the
# smallest face value; otherwise the level search continues from the first level. The tests assert values and accuracy, not the
# branch taken.
@testset "$W the closed form of a face inside its range, and faces at its boundary" for W in TYPES
  # the first level -1 is far below the smallest face value 1: the level is -1 and the residual is -1 on the three entries
  @test kkt_violation(NormLinf(W(9)), -W[3, 2, 1], ones(W, 3)) === sqrt(W(3))
  @test kkt_violation(NormLinf(W(9)), W[3, 2, 1], -ones(W, 3)) === sqrt(W(3))
  # three equal entries and radii so small that the first level, formed in W, is at or above the entries
  v, x = fill(-W(0.1), 3), ones(W, 3)
  for λ in (ldexp(one(W), W === Float32 ? -30 : -60), ldexp(one(W), W === Float32 ? -22 : -50), W(1e-3), W(0.2), W(0.3), W(0.31))
    @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
  end
  # the head 1 above 10^5 entries of 0.009486833 with the radius 0.1, on the face of all entries
  m = 10^5
  v, x = cluster_arrays(W, 1.0, 0.009486833, m; face = true)
  @test cluster_ok(kkt_violation(NormLinf(W(0.1)), v, x), W(0.1), W(1.0), W(0.009486833), m, W; face = true)
end

@testset "$T faces of nearly equal entries" for T in TYPES
  rng = MersenneTwister(981)
  for n in (3, 17, 200, 2000), spread in (0, 1, 4, 64), cf in (1e-8, 1e-4, 1e-2, 0.5, 5.0), sg in (1, -1)
    y = T(0.1 + rand(rng))
    u = T[y + spread * eps(y) * T(rand(rng) - 0.5) for _ in 1:n]
    x = fill(T(sg), n)
    v = -sg .* u
    λ = T(cf) * y
    @test ok(kkt_violation(NormLinf(λ), v, x), λ, v, x)
    @test ok(kkt_violation(NormLinf(λ * n), v, x), λ * n, v, x)
  end
end

# ---- 12. Scaled -----------------------------------------------------------------------------------------------------------------------
tiny(T) = ldexp(one(T), T === Float32 ? -100 : -600)                    # tiny(T)^2 is not a normal number
lvl(T) = ldexp(one(T), T === Float32 ? 100 : 1000)                      # tiny(T)^2 * lvl(T) is
folded(T) = ldexp(one(T), T === Float32 ? -100 : -200)                   # tiny(T)^2 lvl(T)
chain(T, f) = (c = Scaled(Scaled(f, tiny(T)), tiny(T)); c.f isa Scaled || error("merged"); c)

@testset "$W the clusters through a chain that folds" for W in TYPES
  # the cluster and the radius times folded(W) (a power of two), and the chain around NormLinf(λ lvl(W)), which folds to the radius
  # λ folded(W): the score is that of the cluster at that radius
  f = folded(W)
  for (W0, y1, y0, m, λ) in TIE_CASES
    W0 === W || continue
    h = chain(W, NormLinf(lvl(W) * W(λ)))
    for pos in (:first, :last), (face, off) in ((false, ()), (true, ()), (true, OFF_ENTRIES .* Float64(f)))
      v, x = cluster_arrays(W, W(y1) * f, W(y0) * f, m; face, pos, off)
      slack = 1 + 6 / linf_bound(m + 1 + length(off))
      @test cluster_ok(kkt_violation(h, v, x), W(λ) * f, W(y1) * f, W(y0) * f, m, W; face, off, slack)
    end
  end
end

@testset "$T Scaled" for T in TYPES
  rng = MersenneTwister(977)
  n = 9
  v, x = randn(rng, T, n), tie!(rng, randn(rng, T, n), 0.4)
  for λ in (T(0.3), T(2), T(30))
    f = NormLinf(λ)
    # one finite factor: α times the score at v/α, which is the score of the function with the level αλ at v. For a power of two
    # the division is exact and the two are equal; for another α the division rounds, so the value is compared with the
    # reference of the level αλ within the bound that includes the rounding of v/α
    for α in (T(2), T(0.5), T(3), T(0.7))
      s = Scaled(f, α)
      for p in (x, zeros(T, n))
        r = kkt_violation(s, v, p)
        @test r isa T
        bigprec() do
          ref = refbig(big(α) * big(λ), big.(v), big.(p))
          b = (linf_bound(n) + 6) * big(eps(T)) / 2 * (sqrt(sum(abs2, big.(v))) + big(α) * big(λ))
          @test abs(big(r) - ref) <= b
        end
        # the same loops read the lazy v/α and the materialized v ./ α, but their sums can be formed in another order: close, not
        # compared bit for bit
        @test isapprox(r, α * kkt_violation(f, v ./ α, p); rtol = 16 * n * eps(T))
      end
    end
    # the full point of a chain of Scaled (merged: one factor)
    @test kkt_violation(Scaled(Scaled(f, T(2)), T(2)), v, x) === kkt_violation(Scaled(f, T(4)), v, x)
  end
  # the four-argument and three-argument forms give the ArgumentError of the function
  s = Scaled(NormLinf(T(0.3)), T(2))
  for s2 in (s, Scaled(s, T(3)), Scaled(NormLinf(T(0.3)), T(0.25)))
    for form in (() -> kkt_violation(s2, v, x, 2), () -> kkt_violation(s2, v, 2), () -> kkt_violation(s2, v, BlockIterate(copy(x), [collect(1:n)]), 1),
                 () -> kkt_violation(s2, v, zeros(T, n), 2))
      @test_throws ArgumentError form()
      @test occursin("kkt_violation(f, v, x)", msg(form))
    end
  end
  # an infinite factor: refused in the full-point form
  inf = Scaled(NormLinf(T(0.3)), T(Inf))
  @test_throws ArgumentError kkt_violation(inf, v, x)
  @test_throws ArgumentError kkt_violation(inf, v, zeros(T, n))
  @test occursin("infinite factor", msg(() -> kkt_violation(inf, v, x)))
  # a retained chain: the product of the factors is not a normal number. Where it folds into the level, the value is that of the
  # folded level; where it does not, an ArgumentError
  c = chain(T, NormLinf(lvl(T)))
  for p in (x, zeros(T, n))
    r = kkt_violation(c, v, p)
    @test r isa T
    @test ok(r, folded(T), v, p)
    @test r === kkt_violation(NormLinf(folded(T)), v, p)
  end
  @test_throws ArgumentError kkt_violation(c, v, 2)
  @test_throws ArgumentError kkt_violation(c, v, x, 2)
  nf = chain(T, NormLinf(T(1)))
  @test_throws ArgumentError kkt_violation(nf, v, x)
  @test occursin("Scaled wrappers around NormLinf", msg(() -> kkt_violation(nf, v, x)))
  @test_throws ArgumentError kkt_violation(nf, v, x, 2)
  @test_throws ArgumentError kkt_violation(chain(T, NormLinf(lvl(T))), v, SymmetricSparseIterate(T[0 1; 1 0]))
  # a SymmetricSparseIterate point
  for h in (s, Scaled(s, T(3)), chain(T, NormLinf(lvl(T))), chain(T, NormLinf(T(1))))
    @test_throws ArgumentError kkt_violation(h, ones(T, 2, 2), SymmetricSparseIterate(T[0 1; 1 0]))
    m = msg(() -> kkt_violation(h, ones(T, 2, 2), SymmetricSparseIterate(T[0 1; 1 0])))
    @test occursin("NormLinf", m) && occursin("SymmetricSparseIterate", m)
    @test !occursin("kkt_violation(f, v, x, k)", m)                       # the form that NormLinf refuses is not advised
  end
  # arguments are checked before the factor is applied
  @test_throws ArgumentError kkt_violation(s, Complex{T}.(v), x)
  @test_throws DimensionMismatch kkt_violation(s, v[1:3], x)
  # the class of a function with a score
  @test @inferred(PB._kkt_class(typeof(s))) isa PB._KktScore
  @test (@inferred kkt_violation(s, v, x)) isa T
  # a zero factor is the zero function, whose score is the largest magnitude of v
  @test kkt_violation(Scaled(NormLinf(T(0.3)), zero(T)), v, x) === maximum(abs, v)
end

@testset "no ambiguity" begin
  @test isempty(Test.detect_ambiguities(ProximalBase))
end

end
