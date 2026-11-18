module NormLinfTest
# NormLinf (#946): f(x) = λ max|xᵢ| and its proximal operator, the clip yᵢ = clamp(xᵢ, -τ, τ) with
# Σ(|xᵢ| - τ)₊ = γλ (and +0 when γλ ≥ ‖x‖₁), against an independent sort-based ℓ1-ball threshold written in this
# file in BigFloat, in Float32 and Float64. The tolerance of an output entry is TOL_C n eps(T) max|xᵢ| plus two subnormal
# spacings: the absolute accuracy O(n eps(T) max|xᵢ| + δ) of the threshold that the docstring states, with the constants
# TOL_C and 2 (δ = nextfloat(zero(T))).
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
# the reference level and clip for the radius r = γλ, formed in BigFloat from the stored γ and λ
function ref_prox(u, γ, λ, setup = ref_setup(u))
  setprecision(BigFloat, PREC) do
    p, cs = setup
    r = big(γ) * big(λ)
    τ = isempty(p) || r >= cs[end] ? big(0) : ref_level(p, cs, r)
    [clamp(big(v), -τ, τ) for v in u], τ
  end
end
# the tolerance on an entry of the output
tol(T, n, m) = TOL_C * n * big(eps(T)) * big(m) + 2 * big(nextfloat(zero(T)))
function close_prox(y, ref, u)
  T = eltype(y)
  b = tol(T, length(u), maximum(abs, u; init = zero(T)))
  all(i -> abs(big(y[i]) - ref[i]) <= b, eachindex(y))
end
# the conditions of the clip, from the data and the output: ‖x - y‖₁ = γλ (tolerance TOL_C n eps ‖x‖₁), and
# ⟨x - y, y⟩ = γλ ‖y‖∞ (the same tolerance times ‖y‖∞); y = 0 needs ‖x‖₁ ≈ γλ or below
function kkt_ok(y, u, γ, λ)
  setprecision(BigFloat, PREC) do
    T, n = eltype(y), length(u)
    r = big(γ) * big(λ)
    s1 = sum((abs(big(v)) for v in u); init = big(0))
    d = sum((abs(big(u[i]) - big(y[i])) for i in eachindex(u)); init = big(0))
    ym = big(maximum(abs, y; init = zero(T)))
    ip = sum(((big(u[i]) - big(y[i])) * big(y[i]) for i in eachindex(u)); init = big(0))
    b = TOL_C * n * big(eps(T)) * s1 + 2 * big(nextfloat(zero(T)))
    (r >= s1 ? d <= r + b : abs(d - r) <= b) && abs(ip - r * ym) <= b * ym
  end
end
# the value at the stored output y, against λ times the reference level
function close_value(v, y, u, λ, τref)
  T = eltype(y)
  b = tol(T, length(u), maximum(abs, u; init = zero(T)))
  abs(big(v) - big(λ) * τref) <= big(λ) * b + big(eps(T)) * abs(big(v)) + 2 * big(nextfloat(zero(T)))
end

# seeded data: uniform, heavy-tailed and with many ties
function gen(rng, T, n, kind)
  kind == 1 && return T[2 * rand(rng, T) - 1 for _ in 1:n]
  kind == 2 && return T[(rand(rng, Bool) ? 1 : -1) * exp(3 * randn(rng, T)) for _ in 1:n]
  T[rand(rng, (T(-2), T(-1), T(0), T(1), T(2), T(3))) for _ in 1:n]
end

# equal as bit patterns (the sign of a zero and the payload of a NaN included)
bits(v::Float32) = reinterpret(UInt32, v)
bits(v::Float64) = reinterpret(UInt64, v)
same(a, b) = length(a) == length(b) && all(i -> bits(a[i]) == bits(b[i]), eachindex(a))

@testset "construction, traits, equality and display" begin
  @test NormLinf().λ === 1 && NormLinf(0.5f0).λ === 0.5f0
  @test_throws ArgumentError NormLinf(-1.0)
  @test_throws ArgumentError NormLinf(NaN)
  f = NormLinf(2.0)
  @test isproximable(f) && isconvex(f) && !issmooth(f) && !isseparable(f) && !iszeropreserving(f)
  # the buffer is not part of the function
  @test NormLinf(1.0) == NormLinf(1.0) && hash(NormLinf(1.0)) == hash(NormLinf(1.0))
  @test NormLinf(1.0) != NormLinf(2.0)
  @test NormLinf(1) == NormLinf(1.0) && hash(NormLinf(1)) == hash(NormLinf(1.0))
  g = NormLinf(1.0); prox!(zeros(3), g, [1.0, 2.0, 3.0], 1.0)
  @test g == NormLinf(1.0)
  @test repr(NormLinf(2.5)) == "NormLinf(2.5)" && repr(NormLinf(0.5f0)) == "NormLinf(0.5)"
end

@testset "analytic cases ($T)" for T in (Float32, Float64)
  f = NormLinf(one(T))
  @test f(T[3, -1]) == 3 && f(T[3 0; 0 -4]) == 4
  @test prox(f, T[3, -1], one(T)) == T[2, -1]                   # τ = 2
  @test value_and_prox!(zeros(T, 2), f, T[3, -1], one(T)) === T(2)
  @test prox(f, T[2, 1], T(2)) == T[0.5, 0.5]                    # τ = 0.5
  @test prox(f, T[3, -1], T(4)) == T[0, 0]                       # γλ = ‖x‖₁
  @test prox(f, T[3, -1], T(5)) == T[0, 0]
  @test prox(f, T[2, 2, 2], T(3)) == T[1, 1, 1]                  # ties
  @test prox(NormLinf(T(0.5)), T[4, -4, 1], T(4)) == T[3, -3, 1]  # Σ(abs(xᵢ) - 3)₊ = 2 = γλ
  # integer level and step
  @test prox(NormLinf(), T[3, -1], 1) == T[2, -1]
  @test NormLinf(2)(T[1, -3]) === T(6)
end

@testset "random vectors against the sort-based ℓ1-ball reference ($T)" for T in (Float32, Float64)
  rng = Xoshiro(946)
  λ = T(0.7)
  checked = 0
  for n in (1, 2, 3, 5, 16, 17, 100, 1000, 5000, 10^4, 10^5), kind in 1:3
    n >= 5000 && kind == 3 && continue
    x = gen(rng, T, n, kind)
    setup = ref_setup(x)
    s1 = setup[2][end]
    fracs = n >= 5000 ? (1e-6, 0.3, 0.9, 0.999999) : (1e-12, 1e-6, 1e-3, 0.05, 0.3, 0.7, 0.95, 0.999999)
    for frac in fracs
      γ = T(frac * Float64(s1) / Float64(λ))
      f = NormLinf(λ)
      y = zeros(T, n)
      v = value_and_prox!(y, f, x, γ)
      ref, τref = ref_prox(x, γ, λ, setup)
      @test close_prox(y, ref, x)
      @test kkt_ok(y, x, γ, λ)
      @test v === f(y)
      @test close_value(v, y, x, λ, τref)
      # prox! writes the same output, with the output as the buffer or with the scratch of f
      @test same(prox!(zeros(T, n), f, x, γ), y)
      xi = copy(x)
      @test prox!(xi, f, xi, γ) === xi && same(xi, y)
      @test value_and_prox!(copy(x), f, x, γ) === v
      checked += 1
    end
  end
  @test checked >= 200
end

@testset "Michelot fallback and its stopping rule ($T)" for T in (Float32, Float64)
  δ = nextfloat(zero(T))
  # a level that rounds onto the entries: ones with a radius below the spacing of the sum. At n = 2000 and 2001 the
  # candidate set passes max(1024, n ÷ 8) and Michelot's algorithm takes over; the level (n - r)/n rounds to 1
  r0 = T == Float64 ? T(8e-14) : T(4.5e-5)
  for n in (2000, 2001)
    f = NormLinf(r0)
    x = ones(T, n)
    @test prox(f, x, one(T)) == x
    y = zeros(T, n)
    @test value_and_prox!(y, f, x, one(T)) === r0 && y == x
    @test value_and_prox!(copy(x), f, x, one(T)) === r0
  end
  # at a larger n the level may land one spacing below 1, within the accuracy of the threshold
  x = ones(T, 20001)
  @test close_prox(prox(NormLinf(r0), x, one(T)), ref_prox(x, one(T), r0)[1], x)
  # the loop of the legacy kernel took a time quadratic in n here: 4 10^10 operations, against two passes
  if T == Float64
    x = ones(T, 200001); y = similar(x); f = NormLinf(1e-11)
    prox!(y, f, x, one(T))
    @test (@elapsed prox!(y, f, x, one(T))) < 2.0 && y == x
    # the same size with a radius that keeps half of the sum: the level 1/2 is found in two passes
    f = NormLinf(100000.5)
    @test (@elapsed prox!(y, f, x, one(T))) < 2.0 && y == fill(T(0.5), 200001)
    # subnormal analogue: 2001 entries of 1000δ and the radius 600δ; the level rounds to 1000δ
    x = fill(1000 * δ, 2001)
    @test prox(NormLinf(600 * δ), x, one(T)) == x
  end
  # ties with a tiny radius, and with a few distinct values, at a large n
  rng = Xoshiro(95)
  for (n, r) in ((5000, 1e-9), (20000, 1e-3), (20000, 5.0))
    x = T[rand(rng, (T(1), T(2), T(3))) for _ in 1:n]
    f = NormLinf(one(T))
    y = prox(f, x, T(r))
    ref, _ = ref_prox(x, T(r), one(T))
    @test close_prox(y, ref, x) && kkt_ok(y, x, T(r), one(T))
  end
  # the branch is taken: Condat's pass hands over when more than max(1024, n ÷ 8) entries are candidates, so a buffer of
  # that length suffices for it and the call returns Michelot's level (checked where a write past the buffer throws)
  if Base.JLOptions().check_bounds == 1
    x = ones(T, 2001)
    buf = Vector{T}(undef, 1024)
    τ = PB._l1ball_threshold!(buf, x, r0, PB._AbsRead{T}())
    @test τ == 1 && τ == PB._l1ball_michelot(x, r0, PB._AbsRead{T}(), T)
  end
end

@testset "the rounding return ($T)" for T in (Float32, Float64)
  f = NormLinf(T(1e-17))
  x = T[1, 0.5]
  y = zeros(T, 2)
  @test value_and_prox!(y, f, x, one(T)) === T(1e-17) && y == x
  @test prox(NormLinf(one(T)), T[1], T(1e-20)) == T[1]
end

@testset "structural cases ($T)" for T in (Float32, Float64)
  rng = Xoshiro(947)
  λ, γ = T(0.9), T(1.3)
  f = NormLinf(λ)
  # zero entries and signed zeros: -0 is unchanged below τ, an all-zero argument gives +0
  y = prox(NormLinf(one(T)), T[-0.0, 3, 1], one(T))
  @test y == T[0, 2, 1] && signbit(y[1])
  @test all(!signbit, prox(f, T[-0.0, 0.0, -0.0], γ)) && prox(f, zeros(T, 4), γ) == zeros(T, 4)
  # sign and permutation equivariance
  for n in (5, 40, 300)
    x = gen(rng, T, n, 2)
    s = T[rand(rng, Bool) ? 1 : -1 for _ in 1:n]
    r = T(0.4) * sum(abs, x)
    y = prox(f, x, r)
    @test same(prox(f, s .* x, r), s .* y)
    p = randperm(rng, n)
    @test close_prox(prox(f, x[p], r), big.(y[p]), x)
  end
  # n = 1 agrees with NormL1 (the exceptions are in the table: -0 at γ = 0 and an infinite entry with an overflowing γλ)
  for x in T[5, -5, 0.3, -0.3, 1e-3, 3e5], γ1 in T[0.5, 2, 7, 100], λ1 in T[0.5, 1, 3]
    @test prox(NormLinf(λ1), [x], γ1) == prox(NormL1(λ1), [x], γ1)
    @test value_and_prox!(zeros(T, 1), NormLinf(λ1), [x], γ1) == value_and_prox!(zeros(T, 1), NormL1(λ1), [x], γ1)
  end
  # matrices, non-contiguous views, x === y, an output that overlaps the argument through a view
  A = gen(rng, T, 24, 1); A = reshape(A, 4, 6)
  yA = prox(f, A, γ)
  @test size(yA) == (4, 6) && same(vec(yA), prox(f, vec(A), γ)) && f(A) == f(vec(A))
  xv = view(A, :, 2:2:6)
  @test same(prox(f, xv, γ), prox(f, Array(xv), γ))
  Y = zeros(T, 4, 6); yv = view(Y, :, 1:2:5)
  @test prox!(yv, f, xv, γ) === yv && same(Array(yv), prox(f, Array(xv), γ))
  B = copy(A)
  @test prox!(B, f, B, γ) === B && same(B, yA)
  C = copy(A)
  @test prox!(C, f, reshape(view(C, :), 4, 6), γ) === C && same(C, yA)
  D = copy(A)
  @test prox!(D, f, view(D, :, :), γ) === D && same(D, yA)
  v = copy(vec(A))
  @test prox!(v, f, view(v, :), γ) === v && same(v, vec(yA))
  @test_throws MethodError prox!(zeros(T, 4, 6), f, A, ones(T, 4, 6))
end

@testset "endpoint rows ($T)" for T in (Float32, Float64)
  rng = Xoshiro(948)
  f = NormLinf(one(T))
  γ = one(T)
  # a NaN entry makes all the outputs NaN, in each position tested (#677: the middle and the last positions failed in legacy)
  for n in (2, 3, 5, 16, 17, 40), pos in unique((1, 2, n ÷ 2 + 1, n - 1, n))
    x = T[1 + i for i in 1:n]
    x[pos] = T(NaN)
    y = zeros(T, n)
    @test isnan(value_and_prox!(y, f, x, γ)) && all(isnan, y)
    @test all(isnan, prox(f, x, γ)) && isnan(f(x))
    xi = copy(x)
    @test all(isnan, prox!(xi, f, xi, γ))
    x[1 + pos % n] = T(Inf)                       # a NaN and an infinite entry
    @test all(isnan, prox(f, x, γ))
  end
  @test all(isnan, prox(f, T[1, NaN, 5], γ)) && all(isnan, prox(f, T[1, 2, 5, NaN], T(1e-3)))
  # an infinite entry without NaN leaves the argument unchanged, also when γλ overflows
  for (g, l) in ((one(T), one(T)), (floatmax(T), floatmax(T)), (T(1e30), T(1e30)), (T(0.5), T(1e-30)))
    for x in (T[Inf, 1], T[1, -Inf], T[1, 2, -Inf, 4, 5, 6, 7, Inf], T[Inf])
      h = NormLinf(l)
      y = zeros(T, length(x))
      v = value_and_prox!(y, h, x, g)
      @test y == x && v === T(Inf) && v === h(y)
    end
  end
  # λ ∈ {0, Inf} and γ ∈ {0, 1, Inf}
  inputs = (T[1, NaN, -Inf, -0.0, 3], T[-0.0, 2], T[Inf, 1], T[0, 0, 0], T[])
  for g in (zero(T), one(T), T(Inf)), x in inputs
    h = NormLinf(T(Inf))
    y = fill(T(7), length(x))
    v = value_and_prox!(y, h, x, g)
    @test all(iszero, y) && all(!signbit, y) && v === zero(T)
    @test prox(h, x, g) == zeros(T, length(x))
    h0 = NormLinf(zero(T))
    y = fill(T(7), length(x))
    v = value_and_prox!(y, h0, x, g)
    @test same(y, x) && v === zero(T) && h0(x) === zero(T)
  end
  # finite λ > 0: γ = 0 is the identity (-0 kept), γ = Inf gives +0 for the inputs listed
  y = prox(f, T[-0.0, 2], zero(T))
  @test same(y, T[-0.0, 2]) && value_and_prox!(zeros(T, 2), f, T[-0.0, 2], zero(T)) === T(2)
  @test same(prox(f, T[NaN, Inf, 1], zero(T)), T[NaN, Inf, 1])
  for x in (T[NaN, Inf, 1], T[1, 2, 3], T[-0.0, 4], T[])
    y = fill(T(7), length(x))
    @test value_and_prox!(y, f, x, T(Inf)) === zero(T) && all(iszero, y) && all(!signbit, y)
  end
  # values
  @test NormLinf(T(Inf))(zeros(T, 3)) === zero(T) && NormLinf(T(Inf))(T[]) === zero(T)
  @test NormLinf(T(Inf))(T[0, -0.0]) === zero(T) && NormLinf(T(Inf))(T[0, 1]) === T(Inf)
  @test isnan(NormLinf(T(Inf))(T[NaN, 0])) && NormLinf(zero(T))(T[NaN, Inf]) === zero(T)
  @test f(T[]) === zero(T) && prox(f, T[], γ) == T[] && value_and_prox!(T[], f, T[], γ) === zero(T)
  @test isnan(f(T[1, NaN])) && f(T[-3, Inf]) === T(Inf)
  # an empty matrix
  @test size(prox(f, zeros(T, 0, 3), γ)) == (0, 3)
  # zero output when γλ reaches the sum, and an overflowing γλ with a finite sum
  @test prox(f, T[1, 2, 3], T(6)) == zeros(T, 3) && prox(f, T[1, 2, 3], T(7)) == zeros(T, 3)
  @test prox(NormLinf(T(1e30)), T[1e10, 1e10], T(1e30)) == zeros(T, 2)           # γλ = Inf in T
  @test prox(NormLinf(floatmax(T)), T[1, 1], floatmax(T)) == zeros(T, 2)
  # finite γ, λ with γλ above the sum, a subnormal γλ, an underflowing γλ
  @test prox(NormLinf(nextfloat(zero(T))), T[1, 2, 3], T(0.5)) == T[1, 2, 3]
  @test prox(NormLinf(T(1e-30)), T[1, 2, 3], T(1e-30)) == T[1, 2, 3]
end

@testset "range: overflowing sums, overflowing and subnormal γλ ($T)" for T in (Float32, Float64)
  M = floatmax(T)
  rng = Xoshiro(949)
  # entries near floatmax/n with a sum between floatmax/2 and floatmax, and entries of order floatmax whose sum
  # overflows; γ = λ = √r reaches a radius r beyond floatmax
  for n in (1, 2, 3, 10, 1000), fam in (:near, :over), kind in 1:3, frac in (0.001, 0.3, 0.9)
    (fam == :over && n == 1) && continue
    x = if fam == :near
      T[(rand(rng, Bool) ? 1 : -1) * (T(0.5) + rand(rng, T) / 2) * (M / n) for _ in 1:n]
    else
      T[(rand(rng, Bool) ? 1 : -1) * (T(0.3) + T(0.7) * rand(rng, T)) * M for _ in 1:n]
    end
    kind == 2 && (x[1] *= T(1e-3))
    kind == 3 && (x[end] = zero(T))
    setup = ref_setup(x)
    g = T(sqrt(big(frac) * setup[2][end]))
    f = NormLinf(g)
    ref, τref = ref_prox(x, g, g, setup)
    y = zeros(T, n)
    v = value_and_prox!(y, f, x, g)
    @test close_prox(y, ref, x) && v === f(y)
    @test same(prox!(copy(x), f, x, g), y)
    @test f(x) == T(big(g) * maximum(abs, big.(x)))                # one rounding of the exact product
  end
  # the example whose threshold is far below the entries: x = [floatmax, 2^(e-54)] with γλ = floatmax has τ = 2^(e-55)
  e = exponent(M) + 1
  x = T[M, ldexp(one(T), e - 54)]
  y = prox(NormLinf(M), x, one(T))
  ref, τref = ref_prox(x, one(T), M)
  @test τref == big(2)^(e - 55) && close_prox(y, ref, x)
  # the sum sits at floatmax/2 and just above it, and at the overflow, with the radius a fraction of it
  for n in (2, 4, 16), d in (-3, -1, 0, 1, 3)
    x = fill(M / (2n), n)
    for _ in 1:abs(d)
      x[1] = d < 0 ? prevfloat(x[1]) : nextfloat(x[1])
    end
    setup = ref_setup(x)
    for frac in (0.2, 0.7)
      g = T(sqrt(big(frac) * setup[2][end]))
      ref, _ = ref_prox(x, g, g, setup)
      @test close_prox(prox(NormLinf(g), x, g), ref, x)
    end
  end
  # a subnormal threshold, subnormal and tiny inputs
  δ = nextfloat(zero(T))
  for x in (T[δ, 2δ, 3δ, -5δ], T[1000δ, 300δ, 20δ], T[floatmin(T), 2 * floatmin(T), 3δ], T[1, 2δ, floatmin(T)])
    for (g, l) in ((one(T), 7δ), (T(0.5), 100δ), (one(T), one(T)), (T(2), floatmin(T)))
      ref, _ = ref_prox(x, g, l)
      y = prox(NormLinf(l), x, g)
      @test close_prox(y, ref, x)
    end
  end
  # γλ overflowing from finite factors at a finite sum gives +0
  @test prox(NormLinf(floatmax(T) / 2), T[3, 4], T(4)) == zeros(T, 2)
end

@testset "validation before writes ($T)" for T in (Float32, Float64)
  f = NormLinf(one(T))
  x = T[3, -1, 2]
  sentinel() = fill(T(7), 3)
  for γ in (T(-1), T(-Inf), T(NaN))
    y = sentinel()
    @test_throws ArgumentError prox!(y, f, x, γ)
    @test_throws ArgumentError value_and_prox!(y, f, x, γ)
    @test_throws ArgumentError prox(f, x, γ)
    @test y == sentinel()
    xi = copy(x)
    @test_throws ArgumentError prox!(xi, f, xi, γ)
    @test xi == x
  end
  y = fill(T(7), 4)
  @test_throws DimensionMismatch prox!(y, f, x, one(T))
  @test_throws DimensionMismatch value_and_prox!(y, f, x, one(T))
  @test y == fill(T(7), 4)
  # complex entries
  xc, yc = Complex{T}[1, 2, 3], fill(Complex{T}(7), 3)
  @test_throws ArgumentError prox!(sentinel(), f, xc, one(T))
  @test_throws ArgumentError prox!(yc, f, x, one(T))
  @test_throws ArgumentError f(xc)
  @test yc == fill(Complex{T}(7), 3)
  # offset axes: the input, the output and the value
  xo = OffsetVector(copy(x), 0:2)
  y = sentinel()
  @test_throws ArgumentError prox!(y, f, xo, one(T))
  @test_throws ArgumentError value_and_prox!(y, f, xo, one(T))
  @test_throws ArgumentError prox(f, xo, one(T))
  @test_throws ArgumentError f(xo)
  yo = OffsetVector(sentinel(), 0:2)
  @test_throws ArgumentError prox!(yo, f, x, one(T))
  @test_throws ArgumentError value_and_prox!(yo, f, x, one(T))
  @test y == sentinel() && parent(yo) == sentinel()
  # integer inputs, typemin(Int) among them (abs of the integer first would be negative)
  @test prox(f, [3, -1], 1) == [2, -1]
  @test prox(NormLinf(1.0), [3, -1], 1) == [2.0, -1.0]
  xi = [typemin(Int), 3]
  @test NormLinf(1.0)(xi) == 2.0^63
  yi = prox(NormLinf(1.0), xi, 1.0)
  @test yi == [-2.0^63, 3.0]
  @test prox(NormLinf(1.0), [typemin(Int), typemax(Int)], 2.0) == [-2.0^63, 2.0^63]
  # an array of steps is not supported
  @test_throws MethodError prox!(zeros(T, 3), f, x, ones(T, 3))
end

@testset "fused value is the value at the stored output" begin
  rng = Xoshiro(950)
  for T in (Float32, Float64), n in (1, 2, 7, 64, 3000), kind in 1:3, λ in (T(0.01), T(1), T(1e3), T(Inf), zero(T))
    x = gen(rng, T, n, kind)
    s1 = sum(abs, x)
    for γ in (T(0), T(1e-6) * s1 / max(λ, T(1e-3)), T(0.5) * s1 / max(λ, T(1e-3)), T(2) * s1 / max(λ, T(1e-3)), T(Inf))
      isnan(γ) && continue
      f = NormLinf(λ)
      y = zeros(T, n)
      v = value_and_prox!(y, f, x, γ)
      @test v === f(y)
      @test v === value_and_prox!(copy(x), f, x, γ)
    end
  end
  # mixed output types: the value is the value at the stored output
  f = NormLinf(1.0)
  x = Float32[3, -1, 0.1]
  y = zeros(Float32, 3)
  @test value_and_prox!(y, f, x, 1.0) === f(y)
end

@testset "iterates ($T)" for T in (Float32, Float64)
  f = NormLinf(T(0.7))
  d = T[0, 3, 0, -4, 0, 0]
  γ = T(1.5)
  ref = prox(f, d, γ)
  # BlockIterate and SparseIterate through their entries, as arrays; the values
  B = BlockIterate(copy(d), 3)
  r = similar(B)
  @test prox!(r, f, B, γ) === r && Vector(r) == ref
  @test value_and_prox!(similar(B), f, B, γ) == value_and_prox!(similar(d), f, d, γ) && f(B) == f(d)
  x = SparseIterate(copy(d))
  y = SparseIterate(T, 6)
  @test Vector(prox!(y, f, x, γ)) == ref && nactive(y) == 2
  @test value_and_prox!(SparseIterate(T, 6), f, x, γ) === value_and_prox!(similar(d), f, d, γ)
  @test f(x) == f(d) && f(SparseIterate(T, 4)) === zero(T)
  yd = zeros(T, 6)
  @test prox!(yd, f, x, γ) === yd && yd == ref
  @test Vector(prox!(SparseIterate(T, 6), f, d, γ)) == ref
  xs = SparseIterate(copy(d))
  @test Vector(prox!(xs, f, xs, γ)) == ref
  # a NaN and zero output through a SparseIterate
  z = SparseIterate(T, 3)
  @test all(isnan, prox!(z, f, T[1, NaN, 2], γ))
  @test Vector(prox!(z, f, T[1, 2, 3], T(100))) == zeros(T, 3)
  # the value of a SymmetricSparseIterate is the largest stored magnitude
  S = SymmetricSparseIterate(T[1 -5; -5 2])
  @test f(S) == f(Matrix(S)) == T(0.7) * 5
  @test NormLinf(T(Inf))(SymmetricSparseIterate(T, 3)) === zero(T)
  @test NormLinf(zero(T))(S) === zero(T)
  S2 = SymmetricSparseIterate(T, 3); S2[2, 1] = T(NaN)
  @test isnan(f(S2))
  @test allocs(f, S) == 0 && allocs(f, x) == 0
  # no proximal operator for a symmetric iterate, nothing written, directly and through Scaled
  S3 = SymmetricSparseIterate(T[5 6; 6 7])
  D = fill(T(7), 2, 2)
  for g in (f, 2 * f, 2 * (3 * f))
    @test_throws ArgumentError prox!(S3, g, S, one(T))
    @test_throws ArgumentError value_and_prox!(S3, g, S, one(T))
    @test_throws ArgumentError prox!(S3, g, T[1 2; 2 1], one(T))
    @test_throws ArgumentError prox!(D, g, S, one(T))
    @test_throws ArgumentError value_and_prox!(D, g, S, one(T))
    @test_throws ArgumentError prox(g, S, one(T))
  end
  @test Matrix(S3) == T[5 6; 6 7] && D == fill(T(7), 2, 2)
end

@testset "Scaled ($T)" for T in (Float32, Float64)
  f = NormLinf(T(0.5))
  x = T[3, -4, 1, 0.5]
  γ = T(1.5)
  # a factor folded into the level, against the unscaled function (a power of two keeps the products identical)
  @test prox(2 * f, x, γ) == prox(NormLinf(one(T)), x, γ)
  @test prox(f * 2, x, γ) == prox(f, x, 2γ)
  @test same(prox(4 * (2 * f), x, γ), prox(NormLinf(T(4)), x, γ))
  @test (3 * f)(x) == 3 * f(x) && (2 * NormLinf(T(2)))(x) == T(16)
  @test value_and_prox!(zeros(T, 4), 2 * f, x, γ) ≈ (2 * f)(prox(2 * f, x, γ))
  # the zero factor, an infinite factor at a zero point, and infinite levels
  @test (0 * f) isa ZeroFunction
  @test_throws ArgumentError 0 * NormLinf(T(Inf))
  @test (T(Inf) * NormLinf(T(2)))(zeros(T, 3)) == 0
  @test (T(Inf) * NormLinf(T(2)))(T[0, 1]) == T(Inf) && (T(Inf) * NormLinf(zero(T)))(T[1, 2]) == 0
  @test prox(2 * NormLinf(T(Inf)), T[1, NaN], zero(T)) == zeros(T, 2)            # the zero map at γ = 0
  @test_throws ArgumentError prox(T(Inf) * NormLinf(T(1)), T[1, 2], zero(T))     # the step Inf⋅0 is NaN
  @test prox(2 * NormLinf(T(Inf)), T[1, NaN], one(T)) == zeros(T, 2)
end

@testset "Scaled with an overflowing step (Float64)" begin
  # r = 1e-300 ⋅ 1e200 ⋅ 1e200 = 1e100, while the step αγ = 1e400 overflows
  s = Scaled(NormLinf(1e-300), 1e200)
  x = [1e150, 0.0]
  y = prox(s, x, 1e200)
  setprecision(BigFloat, PREC) do
    r = big(1e-300) * big(1e200) * big(1e200)
    @test abs(big(y[1]) - (big(1e150) - r)) <= 8 * 2 * big(eps(Float64)) * big(1e150)
  end
  @test y[1] != 0 && y[2] == 0
  # a radius that clips the largest entry alone: τ = 2e100 for r = 1e100 (the radius is rounded in the reference)
  x2 = [3e100, 1e100]
  y2 = prox(s, x2, 1e200)
  ref2, _ = ref_prox(x2, 1e200, 1e-300 * 1e200)
  @test close_prox(y2, ref2, x2) && y2[1] < 2.5e100 && y2[2] == 1e100
  # the value of the chain at the output
  v = value_and_prox!(zeros(2), s, x, 1e200)
  @test isfinite(v) && v ≈ s(y)
end

@testset "Scaled validates before writing, in prox! and in the fused hook ($T)" for T in (Float32, Float64)
  x = T[3, -1]
  for λ in (zero(T), one(T), T(Inf)), g in (2 * NormLinf(λ), 2 * (3 * NormLinf(λ))), op in (prox!, value_and_prox!)
    for γ in (T(-1), T(NaN))
      y = fill(T(7), 2)
      @test_throws ArgumentError op(y, g, x, γ)
      @test y == fill(T(7), 2)
    end
    for m in (3, 1)
      y = fill(T(7), m)
      @test_throws DimensionMismatch op(y, g, x, one(T))
      @test y == fill(T(7), m)
    end
    # an infinite factor at the zero step forms the step Inf⋅0 = NaN
    y = fill(T(7), 2)
    @test_throws ArgumentError op(y, T(Inf) * NormLinf(λ), x, zero(T))
    @test y == fill(T(7), 2)
  end
end

@testset "Scaled at the top binade: the folded level needs exponent headroom ($T)" for (T, e) in ((Float64, 1000), (Float32, 100))
  M = floatmax(T)
  s = Scaled(NormLinf(ldexp(one(T), -e)), ldexp(one(T), e))
  γ = T(0.75) * M                  # the radius γ ⋅ 2^-e ⋅ 2^e = γ is representable, the step α γ is not
  y = prox(s, T[M], γ)
  @test y == T[M - γ]
  v = value_and_prox!(zeros(T, 1), s, T[M], γ)
  @test v ≈ s(y) && v ≈ M - γ
  # two entries at the maximum
  x2 = T[M, M]
  ref, _ = ref_prox(x2, γ, one(T))
  y2 = prox(s, x2, γ)
  @test close_prox(y2, ref, x2) && all(!iszero, y2)
  v2 = value_and_prox!(zeros(T, 2), s, x2, γ)
  @test v2 ≈ s(y2)
  # two factors whose product is normal are combined by the constructor (a simplification, not a retained chain)
  s2 = Scaled(Scaled(NormLinf(ldexp(one(T), -e)), ldexp(one(T), e ÷ 2)), ldexp(one(T), e - e ÷ 2))
  @test !(s2.f isa Scaled) && prox(s2, T[M], γ) == T[M - γ]
  # a retained nested chain: the factors 2^e and 2^e have no normal product, and the step is γ 2^-e
  s3 = Scaled(Scaled(NormLinf(ldexp(one(T), -e)), ldexp(one(T), e)), ldexp(one(T), e))
  @test s3.f isa Scaled
  γ3 = ldexp(γ, -e)
  y3 = prox(s3, x2, γ3)
  @test close_prox(y3, ref, x2) && all(!iszero, y3)
  @test prox(s3, T[M], γ3) == T[M - γ]
  v3 = value_and_prox!(zeros(T, 2), s3, x2, γ3)
  @test v3 ≈ s3(y3)
end

@testset "the documented limit of the Scaled recovery (Float64)" begin
  # the radius 2 ⋅ 0.75 floatmax is beyond the range of the folded level even at half the scale: the wrapper passes an
  # infinite step, and the output is +0 where the exact result is about floatmax/4 (documented, no further recovery)
  M = floatmax(Float64)
  s = Scaled(NormLinf(2.0 * 2.0^-1000), 2.0^1000)
  @test prox(s, [M, M], 0.75M) == [0.0, 0.0]
  ref, _ = ref_prox([M, M], 0.75M, 2.0)
  @test abs(ref[1] - M / 4) <= big(M) / 100
end

@testset "equality, hash and signed-zero levels ($T)" for T in (Float32, Float64)
  f, g = NormLinf(zero(T)), NormLinf(-zero(T))
  @test f == g && isequal(f, g) && hash(f) == hash(g)
  @test get(Dict(f => 42), g, :missing) == 42
  @test length(Set([f, g])) == 1
  @test NormLinf(one(T)) != NormLinf(T(2)) && length(Set([NormLinf(one(T)), NormLinf(T(2))])) == 2
end

@testset "subnormal outputs: the absolute accuracy has the additive term δ ($T)" for T in (Float32, Float64)
  δ = nextfloat(zero(T))
  y = prox(NormLinf(δ), T[δ], T(0.5))
  @test y == T[δ]                                  # the exact point is δ/2
  @test abs(big(y[1]) - big(δ) / 2) <= tol(T, 1, δ)
end

@testset "allocations ($T)" for T in (Float32, Float64)
  rng = Xoshiro(951)
  n = 1000
  x = gen(rng, T, n, 1)
  s1 = sum(abs, x)
  y = zeros(T, n)
  for λ in (T(0.7), T(Inf), zero(T)), γ in (T(0.01) * s1, T(0.9) * s1, T(2) * s1, T(Inf), zero(T))
    f = NormLinf(λ)
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
    xi = copy(x)
    @test allocs(prox!, xi, f, xi, γ) == 0
    xi = copy(x)
    @test allocs(value_and_prox!, xi, f, xi, γ) == 0
  end
  f = NormLinf(T(0.7))
  @test allocs(f, x) == 0
  # exceptional branches and the Michelot fallback
  xn = copy(x); xn[n ÷ 2] = T(NaN)
  xf = copy(x); xf[3] = T(Inf)
  xb = fill(floatmax(T) / 4, n)
  xo = ones(T, 5000)
  for z in (xn, xf, xb, xo)
    yy = similar(z)
    @test allocs(prox!, yy, f, z, T(1)) == 0
    @test allocs(value_and_prox!, yy, f, z, T(1)) == 0
    zz = copy(z)
    @test allocs(prox!, zz, f, zz, T(1)) == 0
  end
  xs = view(reshape(copy(x), 10, 100), :, 1:2:99)
  ys = zeros(T, 10, 50)
  @test allocs(prox!, ys, f, xs, T(0.3)) == 0
end

# ---- the threshold of the ℓ1 ball: the level equation and the bounds of its derivation (#1003, #987) ---------------------------
# The tests below check the level that the threshold routines return against the exact level and against the bounds proved in the
# header of `_l1ball_threshold!` (src/functions/normlinf.jl): for n entries of magnitude at most M in the type T, with u =
# eps(T)/2, δ = nextfloat(zero(T)) and the hypothesis n u ≤ 1/4, the returned level τ̂ satisfies τ̂ - τ* ≤ the overshoot bound,
# Σ (yᵢ - τ̂)₊ - a ≤ the excess bound of the exits, and |τ̂ - τ*| ≤ 4.5 n u M + 2 n δ (the exit-by-exit statements are checked
# with the sums in sequential order, below). The references are BigFloat (no kernel of the package): the exact level from the
# magnitudes sorted and grouped by value, with prefix sums, and the excess Σ (yᵢ - τ)₊ - a at a float level τ. Each testset says
# which exit or path it reaches.
#
# ---- helpers: references, bounds and inputs ----
# the magnitudes of x sorted decreasingly and grouped by value: the distinct values (BigFloat) and their counts
function ref_groups(x)
  p = sort!(abs.(x); rev = true)
  vals, cnt = BigFloat[], Int[]
  setprecision(BigFloat, PREC) do
    i = 1
    while i <= length(p)
      j = i
      while j < length(p) && p[j + 1] == p[i]
        j += 1
      end
      push!(vals, big(p[i]))
      push!(cnt, j - i + 1)
      i = j + 1
    end
  end
  vals, cnt
end
# the exact level of the groups at the radius a: τ with Σ (vᵢ - τ)₊ = a, 0 when a is at least the sum
function ref_level_groups((vals, cnt), a)
  setprecision(BigFloat, PREC) do
    A = big(a)
    S = big(0)
    K = 0
    for i in eachindex(vals)
      S += vals[i] * cnt[i]
      K += cnt[i]
      τ = (S - A) / K
      (i == length(vals) || τ >= vals[i + 1]) && return max(τ, big(0))
    end
    big(0)
  end
end
# Σ (vᵢ - τ)₊ - a at a float level τ
ref_excess_groups((vals, cnt), τ, a) = setprecision(BigFloat, PREC) do
  t = big(τ)
  s = big(0)
  for i in eachindex(vals)
    vals[i] > t && (s += (vals[i] - t) * cnt[i])
  end
  s - big(a)
end

# the arithmetic of a type: u = eps/2 and δ = nextfloat(zero) as BigFloat
arith(::Type{T}) where {T} = setprecision(BigFloat, PREC) do
  (u = big(eps(T)) / 2, δ = big(nextfloat(zero(T))))
end
γb(k, u) = k * u / (1 - k * u)
# the bounds of the derivation as BigFloat: the overshoot bound (the larger of the bound of a start and of a step), and the
# level bound 4.5 n u M + 2 n δ
function tbounds(ar, n, M)
  setprecision(BigFloat, PREC) do
    u, δ, Mb = ar.u, ar.δ, big(M)
    ζ = 2 * (1 + u)^2 * γb(n - 1, u) * Mb + δ
    εstart = Mb * ((2u + u^2) + (1 + u)^2 * γb(n - 1, u)) + δ / 2
    Γ = max(γb(n, u) + (2u + u^2) / (1 - n * u), (2 + 2u + u^2) * γb(n, u))
    εstep = (Γ * (Mb + ζ) + u * Mb + δ / 2) / (1 - u)
    (over = max(εstart, εstep), level = 9 * n * u * Mb / 2 + 2 * n * δ)
  end
end
# the bound of the excess Σ (yᵢ - τ̂)₊ - a over the exits (k ≤ n entries above the level), at the level τ̂
function excess_bound(ar, n, M, a, τ̂)
  setprecision(BigFloat, PREC) do
    u, δ, A, t = ar.u, ar.δ, big(a), abs(big(τ̂))
    cert = (3 * γb(n, u) / (2 * (1 - n * u)) + u / 2) * A + n * u * t + n * δ
    vanish = (1 + 2 * n * u) / (1 - n * u) * n * (u * t + δ / 2) / (1 - u)^2
    max(cert, vanish, γb(n, u) * A)
  end
end
# nothing when the level τ̂ of the n entries with the groups (largest entry M) is within the bounds, a message otherwise
function level_msg(ar, groups, n, M, a, τ̂)
  setprecision(BigFloat, PREC) do
    τx = ref_level_groups(groups, a)
    t = big(τ̂)
    b = tbounds(ar, n, M)
    t - τx > b.over && return "level $τ̂ is above the exact level $(Float64(τx)) by more than the overshoot bound $(Float64(b.over))"
    abs(t - τx) > b.level && return "level $τ̂ differs from the exact level $(Float64(τx)) by more than $(Float64(b.level))"
    g = ref_excess_groups(groups, τ̂, a)
    eb = excess_bound(ar, n, M, a, τ̂)
    g > eb && return "the excess $(Float64(g)) at the level $τ̂ is above its bound $(Float64(eb))"
    nothing
  end
end

# the output y of a prox of x at the radius a: the clip at its largest magnitude, and the level it implies against the bounds
function output_msg(::Type{T}, x, groups, y, a) where {T}
  n = length(x)
  M = maximum(abs, x)
  τ̂ = maximum(abs, y)
  if iszero(τ̂)
    (all(iszero, y) && all(!signbit, y)) || return "a zero level with a nonzero output"
  else
    all(i -> y[i] == clamp(x[i], -τ̂, τ̂), eachindex(x)) || return "the output is not the clip of the argument at its largest magnitude $τ̂"
  end
  level_msg(arith(T), groups, n, M, a, τ̂)
end
# the output of each entry point at the radius a = λ (step one) and of the wrapper with a power of two: the first message or nothing
function prox_msg(::Type{T}, x, groups, λ; entry_points = true, scaled = true) where {T}
  f = NormLinf(λ)
  o = one(T)
  outs = Pair{String, Any}["prox" => prox(f, x, o)]
  if entry_points
    y = similar(x); prox!(y, f, x, o); push!(outs, "prox!" => y)
    y = copy(x); prox!(y, f, y, o); push!(outs, "prox! in place" => y)
    y = similar(x); v = value_and_prox!(y, f, x, o); push!(outs, "value_and_prox!" => y)
    v === f(y) || return "value_and_prox! returned a value that is not that of its output"
    y = copy(x); value_and_prox!(y, f, y, o); push!(outs, "value_and_prox! in place" => y)
  end
  scaled && push!(outs, "Scaled" => prox(2 * NormLinf(λ / 2), x, o))
  for (name, y) in outs
    msg = output_msg(T, x, groups, y, λ)
    msg === nothing || return "$name: $msg"
  end
  nothing
end

# a head of h entries H above m ties y0, in an entry order: heads first, last, in the middle, or last followed by zeros (the padding
# keeps the number of candidates of Condat's pass below the handover, so that the pass runs alone)
function place_input(ties::Vector{T}, heads::Vector{T}, order) where {T}
  m = length(ties)
  order === :first && return vcat(heads, ties)
  order === :last && return vcat(ties, heads)
  order === :middle && return vcat(ties[1:(m ÷ 2)], heads, ties[(m ÷ 2 + 1):end])
  vcat(ties, heads, zeros(T, 8 * (m + length(heads)) + 16))              # :padded
end
tie_input(::Type{T}, y0, m, H, h, order) where {T} = place_input(fill(T(y0), m), fill(T(H), h), order)
const ORDERS = (:first, :last, :middle, :padded)

# the inputs of #1003, measured at the base: levels 0.17933007, 0.29999998, 0.8999997002396832 and 0.0027803732 where the exact
# ones are 0.5, 0.93, 0.9 and 0.5. Input 1: Condat's pass hands over to Michelot's iteration (10^5 candidates); input 2: the pass
# alone (the zeros keep the candidates below the handover); inputs 3 and 4: Michelot's iteration after the handover.
function issue1003_inputs()
  y0, y1, y2 = 0.17933106f0, 0.8999997002398832, 0.0027806521f0
  (("1", Float32, [fill(y0, 100_000); 1f0], 0.5f0, 0.5),
   ("2", Float32, [fill(0.3f0, 100_000); fill(1.2f0, 10); zeros(Float32, 800_096)], 2.7f0, 0.93),
   ("3", Float64, [fill(y1, 1_000_000); 1.0], 0.1, 0.9),
   ("4", Float32, [fill(y2, 1_000_000); 1f0], 0.5f0, 0.5))
end
# the cases of #987 (large ties) as (type, x, radius)
function large_tie_cases()
  cases = ((Float32, 0.2, 0.1, 10^5, 0.3), (Float32, 1.0, 0.1, 10^5, 1.2), (Float32, 0.2, 0.1, 10^4, 0.102),
           (Float64, 1.8, 0.9, 10^6, 0.9000008))
  [(T, [T(y1); fill(T(y0), m)], T(λ)) for (T, y1, y0, m, λ) in cases]
end
# one head 1 above m ties y0 = (1 - c)/(1 + ρ m² u) at the radius c: the exact level 1 - c lies above the ties by ρ m² u y0, and the
# plain level of all the entries within ρ times the rounding error of the sum of the ties; (m, c, ρ, order, x)
function head_ties_cases(::Type{T}) where {T}
  u = Float64(eps(T)) / 2
  [(m, cf, ρ, order, tie_input(T, (1 - cf) / (1 + ρ * Float64(m)^2 * u), m, 1, 1, order))
   for m in (T == Float32 ? (10, 1000, 10^4, 10^5) : (10, 1000, 10^5)), cf in (0.1, 0.5, 0.9), ρ in (0.003, 0.03, 0.25, 1.0), order in ORDERS]
end
# ---- end of the helpers ----

# a reader that counts its calls: a pass over n entries makes n calls
# (an atomic counter: a plain counter in a vectorized loop could lose increments)
struct CountRead{T}
  calls::Threads.Atomic{Int}
end
CountRead{T}() where {T} = CountRead{T}(Threads.Atomic{Int}(0))
@inline (r::CountRead{T})(v) where {T} = (Threads.atomic_add!(r.calls, 1); abs(convert(T, v)))

# the pass and the iteration of the threshold with the sums taken in sequential order (a copy of the iteration, to check the
# bounds for an order of summation that the vectorized pass does not produce); the exit taken and the number of passes are returned
function seq_pass(y::AbstractVector{T}, τ::T) where {T}
  r = zero(T)
  k = 0
  hi = T(Inf)
  for v in y
    if v > τ
      r += v - τ
      k += 1
      hi = min(hi, v)
    end
  end
  r, k, hi
end
function seq_iterate(y::AbstractVector{T}, a::T, τ::T; certify_any_step = false) where {T}
  passes = 0
  while true
    passes += 1
    r, k, hi = seq_pass(y, τ)
    k == 0 && return maximum(y), :top, k, passes
    r <= a && return τ, :level, k, passes
    t = τ + (r - a) / k
    ((certify_any_step || r - a <= a / 2) && t < hi) && return t, :cert, k, passes
    t > τ || return τ, :vanish, k, passes
    τ = t
  end
end
seq_michelot(y::AbstractVector{T}, a::T; kw...) where {T} = seq_iterate(y, a, (foldl(+, y) - a) / length(y); kw...)
# the inequality of the derivation for the exit taken, with k the count of the last pass: nothing or a message
function exit_msg(ar, groups, n, M, a, τ̂, exit, k)
  msg = level_msg(ar, groups, n, M, a, τ̂)
  msg === nothing || return "$exit: $msg"
  setprecision(BigFloat, PREC) do
    u, δ, A, t = ar.u, ar.δ, big(a), abs(big(τ̂))
    g = ref_excess_groups(groups, τ̂, a)
    if exit === :cert
      abs(g) <= (3 * γb(k, u) / (2 * (1 - k * u)) + u / 2) * A + k * u * t + k * δ || return "the certified step: |excess| $(Float64(abs(g))) is above its bound"
    elseif exit === :level
      g <= γb(k, u) * A || return "the level of the pass: the excess $(Float64(g)) is above γ_k a"
    elseif exit === :vanish
      (0 < g <= (1 + 2 * k * u) / (1 - k * u) * k * (u * t + δ / 2) / (1 - u)^2) || return "the step vanished: the excess $(Float64(g)) is not in (0, its bound]"
    else
      g == -big(a) || return "no entry above the level: the excess is not -a"
    end
    nothing
  end
end

@testset "the four inputs of #1003 through every entry point" begin
  # Entry points: prox, prox! (distinct output and in place), value_and_prox! (both) and the wrapper Scaled with a power of two.
  # Paths: Condat's pass with the handover to Michelot's iteration (input 1), Condat's pass alone (input 2), and Michelot's iteration
  # after the handover (inputs 3 and 4); the returns reached depend on the order of the vectorized sums, and the bounds hold for each.
  for (name, T, x, λ, exact) in issue1003_inputs()
    groups = ref_groups(x)
    @test abs(ref_level_groups(groups, λ) - big(exact)) < 1e-6      # the exact level of the issue
    @test prox_msg(T, x, groups, λ) === nothing
    # the head first (inputs 1, 3 and 4), where the first level is exact in the issue
    if name != "2"
      xf = [x[end]; x[1:(end - 1)]]
      @test prox_msg(T, xf, groups, λ; scaled = false) === nothing
    end
  end
end

@testset "the threshold search when the count rises: two entries ($T)" for T in (Float32, Float64)
  # #987. y = (3, 1 + 3ε), radius 2 - 2ε. The sum 4 + 3ε rounds to 4 + 4ε, the plain level is 1 + 3ε (the second entry, not above
  # it), and the pass at that level gives the excess 2 - 3ε ≤ r: the level of the pass, at the first pass. The exact level is
  # 1 + 2.5ε, midway between the floats 1 + 2ε and 1 + 3ε; the routine returns the upper one (n = 2 fixes the order of the sum).
  ε = eps(T)
  y = T[3, 1 + 3ε]
  r = T(2) - 2ε
  τ = PB._l1ball_michelot(y, r, PB._AbsRead{T}(), T)
  groups = ref_groups(y)
  @test ref_level_groups(groups, r) == big(1) + 2.5 * big(ε)
  @test level_msg(arith(T), groups, 2, 3, r, τ) === nothing
  @test τ == 1 + 3ε
  @test prox_msg(T, y, groups, r) === nothing
end

@testset "large ties: the threshold of the prox is not a false zero" begin
  # the radius puts the level just below the tie y0 (or at it) with m large (the handover to Michelot's iteration, and the
  # return of the level of the pass or of a certified step); each entry point and the wrapper return a level within the bounds
  # and clip the ties at the level, not at zero
  for (T, v, λ) in large_tie_cases()
    @test prox_msg(T, v, ref_groups(v), λ) === nothing
    p = prox(NormLinf(λ), v, one(T))
    @test all(==(p[2]), view(p, 2:length(v)))
    @test p[2] > v[2] / 2
  end
end

@testset "a grid of clusters around the breakpoint: the prox against the exact level ($T)" for T in (Float32, Float64)
  # one head entry above m equal entries, the radius just above the breakpoint y1 - y0 by a multiple of the rounding error of
  # the sum of the entries (read from the sum that the kernel forms): Condat's pass hands over for m ≥ 1024, and the returns of
  # Michelot's iteration are those near a tie
  bad = 0
  for m in (1000, 3000, 10_000), y0f in (0.1, 0.3, 1 / 3, 0.9), mult in (2, 10), pos in (:first, :last), θ in (0.05, 0.6, 1.5)
    y0 = T(y0f)
    y1 = T(mult) * y0
    v = fill(y0, m + 1)
    v[pos === :first ? 1 : end] = y1
    s, _ = PB._linf_classify(v, PB._AbsRead{T}(), T)
    serr = setprecision(BigFloat, PREC) do
      Float64(big(s) - (big(y1) + m * big(y0)))
    end
    c = setprecision(BigFloat, PREC) do
      T(big(y1) - big(y0) + θ * abs(serr))
    end
    bad += prox_msg(T, v, ref_groups(v), c; entry_points = false, scaled = false) !== nothing
  end
  @test bad == 0
end

@testset "head and ties where the first level lies within the rounding of the sum ($T)" for T in (Float32, Float64)
  # heads first, last, in the middle and padded with zeros (Condat's pass alone); the inputs of #1003 are of this form (Float32,
  # m ≥ 10^4). All entry points on the small inputs and on one radius of the large ones
  bad = 0
  ncase = 0
  for (m, cf, ρ, order, x) in head_ties_cases(T)
    msg = prox_msg(T, x, ref_groups(x), T(cf); entry_points = m <= 1000 || (ρ == 0.003 && cf == 0.5), scaled = m <= 10^4)
    ncase += 1
    if msg !== nothing
      bad += 1
      @info "head and ties" T m cf ρ order msg
    end
  end
  @test ncase >= 100 && bad == 0
end

@testset "the four exits of the iteration, each reached by a constructed input ($T)" for T in (Float32, Float64)
  # `_l1ball_michelot` reads the entries once for the sum and once per pass (and once more for the largest entry when no entry is
  # above the level); a counting reader tells which exit was taken
  ar = arith(T)
  # the level of the pass, at the first pass: y = (2, 2, 2), a = 3. The plain level (6 - 3)/3 = 1 is exact, the pass gives the
  # excess 3 ≤ a
  c = CountRead{T}()
  @test PB._l1ball_michelot(T[2, 2, 2], T(3), c, T) == 1
  @test c.calls[] == 2 * 3
  # the level of the pass after a large step: y = (1, 0, 0, 0), a = 1/4. The plain level is 3/16, the pass gives the excess 13/16
  # with one entry above, which is 9/16 beyond a and above a/2: the step to 3/4 is taken and not returned (although no entry lies
  # between the levels), and the next pass gives the excess 1/4 ≤ a
  c = CountRead{T}()
  @test PB._l1ball_michelot(T[1, 0, 0, 0], T(1) / 4, c, T) == T(3) / 4
  @test c.calls[] == 3 * 4
  # the certified step: ones(3) with a radius whose plain level rounds below the exact level, so that the pass gives an excess
  # just above a and the step is returned without a second pass (two readings of the entries, a level other than the plain one)
  found = false
  for a in T.(0.02:0.001:0.98)
    c = CountRead{T}()
    τ = PB._l1ball_michelot(ones(T, 3), a, c, T)
    if c.calls[] == 2 * 3 && τ != (T(3) - a) / 3
      found = true
      @test exit_msg(ar, ref_groups(ones(T, 3)), 3, 1, a, τ, :cert, 3) === nothing
      break
    end
  end
  @test found
  # the step vanished: ones(4) with a = 1.25 eps = 2.5u. The sum less a, 4 - 2.5u, rounds to 4 - 4u, so the plain level is
  # 1 - u = prevfloat(1); the pass gives the excess 4u, which is 1.5u beyond a and above a/2 = 1.25u, and the step 1.5u/4 is below
  # half the spacing u of the floats below 1: the level does not move and is returned (the exact level is 1 - 0.625u)
  a = T(1.25) * eps(T)
  c = CountRead{T}()
  τ = PB._l1ball_michelot(ones(T, 4), a, c, T)
  @test τ == prevfloat(one(T)) && c.calls[] == 2 * 4
  groups = ref_groups(ones(T, 4))
  @test ref_excess_groups(groups, τ, a) > 0
  @test exit_msg(ar, groups, 4, 1, a, τ, :vanish, 4) === nothing
  # no entry above the level: ones(2001) with a below half the spacing of the sum, so that the sum less a rounds to 2001 and the
  # plain level is 1: no entry is above it and the largest entry is returned (readings: the sum, the pass and the largest entry)
  a0 = T == Float64 ? T(8e-14) : T(4.5e-5)
  c = CountRead{T}()
  @test PB._l1ball_michelot(ones(T, 2001), a0, c, T) == 1
  @test c.calls[] == 3 * 2001
end

@testset "Condat's pass alone, with a restart and with the handover ($T)" for T in (Float32, Float64)
  rd = PB._AbsRead{T}()
  # the pass alone reads each entry once (the buffer is read through `identity`); the exact level of these four inputs is 3/2
  # (the entries 3 and 2 are above it: 3/2 + 1/2 = 2 = a) and the pass returns it
  for x in (T[3, 2, 1, 0.5], T[1, 3, 2, 0.5], T[1, 3, 2, 0.5, 0.25, 0.125, 0, 0])
    # the second input restarts the candidates at the entry 3 (the test of the join, 1 > (3 - 2)·1, fails), and the third has
    # entries that are discarded below the level of the candidates
    c = CountRead{T}()
    @test PB._l1ball_threshold!(Vector{T}(undef, length(x)), x, T(2), c) == T(3) / 2
    @test c.calls[] == length(x)
    length(x) == 4 && @test PB._l1ball_michelot(x, T(2), rd, T) == T(3) / 2    # the plain level 9/8 and one step: exact in binary
    @test level_msg(arith(T), ref_groups(x), length(x), 3, T(2), PB._l1ball_michelot(x, T(2), rd, T)) === nothing
  end
  # the handover: more than max(1024, n ÷ 8) candidates (2001 equal entries) go to Michelot's iteration, which reads the argument
  # again; the level is the entries' value, as the existing test of the Michelot fallback has it
  a0 = T == Float64 ? T(8e-14) : T(4.5e-5)
  c = CountRead{T}()
  x = ones(T, 2001)
  @test PB._l1ball_threshold!(Vector{T}(undef, 2001), x, a0, c) == 1
  @test c.calls[] > 2001
end

@testset "the discard level of Condat's pass does not decrease (Float32)" begin
  # the head 1.25 with the radius 1 puts the level at prevfloat(1/4) and the 4000 entries prevfloat(1/4) are discarded; the entry
  # 1/4 then joins the candidates, whose plain level 1/4 would lower the discard level below the 60 entries 1/4 - 15 2^-26, which
  # must not be stored: the level is 1/4 (the excess of the head, 1.25 - 1/4 = 1 = a, is exact)
  x = Float32[1.25; fill(prevfloat(0.25f0), 4000); 0.25f0; fill(0.25f0 - 15 * 2.0f0^-26, 60)]
  c = CountRead{Float32}()
  @test PB._l1ball_threshold!(Vector{Float32}(undef, length(x)), x, 1f0, c) == 0.25f0
  @test c.calls[] == length(x)                                    # no handover
  @test maximum(prox(NormLinf(1f0), x, 1f0)) == 0.25f0
  @test prox_msg(Float32, x, ref_groups(x), 1f0) === nothing
end

@testset "a certified step is not accepted after a large excess ($T)" for T in (Float32, Float64)
  # a head 1, m ties y0 = 3/10 and 8m zeros, radius 0.4: the exact level is 0.6 (the head alone). The plain level is far below the
  # ties, the pass there gives an excess of about 2667 and the step lands just below the ties; the sums of the pass in sequential
  # order make it land at 0.29999757 in Float32. A rule that returned that step (nothing between the levels, so the set is the
  # same) would return it, 0.3 where the level is 0.6; the condition on the excess of the pass sends the iteration on to the next passes
  m = 10^4
  y0, a = T(0.3), T(0.4)
  x = [T(1); fill(y0, m); zeros(T, 8m)]
  groups = ref_groups(x)
  τ, exit, k, passes = seq_michelot(x, a)
  @test exit_msg(arith(T), groups, length(x), 1, a, τ, exit, k) === nothing
  @test passes >= 3                                                # the large step is followed by passes
  @test abs(big(τ) - big(0.6)) < 1e-6
  if T == Float32
    τ0, exit0, _, _ = seq_michelot(x, a; certify_any_step = true)
    @test exit0 === :cert && abs(τ0 - 0.6f0) > 0.1                # the rule without the condition returns about 0.2999976
  end
  @test prox_msg(T, x, groups, a) === nothing
end

@testset "the exits and their bounds with the sums in sequential order ($T)" for T in (Float32, Float64)
  # the iteration with the sums of its passes (and of its start) taken in the order of the loop, which the vectorized passes do
  # not produce: the inequality of the derivation for the exit taken, and each exit reached at least once
  rng = Xoshiro(1006)
  ar = arith(T)
  seen = Dict{Symbol, Int}()
  bad = 0
  function check_seq(x, a)
    a < foldl(+, x) || return
    τ, exit, k, _ = seq_michelot(x, a)
    seen[exit] = get(seen, exit, 0) + 1
    msg = exit_msg(ar, ref_groups(x), length(x), maximum(x), a, τ, exit, k)
    msg === nothing || (bad += 1; @info "sequential order" T length(x) a exit msg)
  end
  for (m, cf, ρ, order, x) in head_ties_cases(T)
    m <= 10^4 && check_seq(x, T(cf))
  end
  for m in (10, 1000, 3 * 10^4), h in (1, 2, 10), order in ORDERS, f in (0.2, 0.95, 1.0, 1.05, 3.0)
    check_seq(tie_input(T, 0.3, m, 1.0, h, order), T(f * h * 0.7))
  end
  for n in (10, 1000, 10^4), ord in (:up, :down, :shuffled), frac in (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999)
    base = T.(range(1, 2; length = n))
    x = ord === :up ? base : ord === :down ? reverse(base) : shuffle(rng, base)
    check_seq(x, T(frac * sum(Float64, x)))
  end
  for n in (100, 10^4), f in (0.2, 0.999, 1.001, 2.0), order in ORDERS
    y0 = T(0.7)
    x = place_input(T[y0 + rand(rng, 0:8) * eps(y0) for _ in 1:(n - 3)], fill(T(1.2), 3), order)
    check_seq(x, T(f * 3 * 0.5))
  end
  # the inputs of the testset of the exits
  check_seq(T[2, 2, 2], T(3))
  check_seq(T[1, 0, 0, 0], T(1) / 4)
  check_seq(ones(T, 4), T(1.25) * eps(T))
  check_seq(ones(T, 2001), T == Float64 ? T(8e-14) : T(4.5e-5))
  for a in T.(0.02:0.001:0.98)
    check_seq(ones(T, 3), a)
    check_seq(ones(T, 7), a)
  end
  @test bad == 0
  @test all(haskey(seen, e) for e in (:level, :cert, :vanish, :top))
end

@testset "deterministic families against the exact level, in four entry orders ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1003)
  bad = 0
  ncase = 0
  report(tag, msg) = (ncase += 1; msg === nothing || (bad += 1; @info "family" T tag msg))
  # equal entries with 1, 2 and 10 larger ones, n from 2 to 10^5, the radius around the breakpoint h (H - y0) where the level
  # reaches the ties (H = y0 + 0.7 up to a rounding). Paths: Condat's pass alone when the ties are not candidates, or when the zeros
  # of the padded order keep the candidates below the handover, and the handover to Michelot's iteration otherwise (m ≥ 1024)
  for m in (1, 2, 9, 1000, 10^4), h in (1, 2, 10), y0f in (0.1, 0.3, 0.9), order in ORDERS, f in (0.2, 0.95, 1.0, 1.05, 3.0)
    x = tie_input(T, y0f, m, y0f + 0.7, h, order)
    a = T(f * h * 0.7)
    report((m, h, y0f, order, f), prox_msg(T, x, ref_groups(x), a; entry_points = m <= 9, scaled = m <= 9))
  end
  # nearly equal entries, 0 to 8 spacings apart, with h larger entries placed in the four orders
  for n in (2, 3, 10, 100, 1000, 10^4, 10^5), y0f in (0.1, 1.0, 0.7), h in (1, 3), f in (0.2, 0.9, 0.999, 1.001, 2.0, 50.0), order in ORDERS
    n > h || continue
    (n >= 10^4 && (f in (0.9, 1.001) || y0f == 0.7)) && continue
    y0 = T(y0f)
    ties = T[y0 + rand(rng, 0:8) * eps(y0) for _ in 1:(n - h)]
    x = place_input(ties, fill(T(y0f + 0.5), h), order)
    report((n, y0f, h, f, order), prox_msg(T, x, ref_groups(x), T(f * h * 0.5); entry_points = false, scaled = false))
  end
  # ramps, up, down and shuffled with a fixed seed, and shuffled with zeros appended; the radius is a fraction of the sum
  for n in (2, 3, 10, 100, 1000, 10^4), ord in (:up, :down, :shuffled, :padded), frac in (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999)
    base = T.(range(1, 2; length = n))
    x = ord === :up ? base : ord === :down ? reverse(base) : shuffle(rng, base)
    ord === :padded && (x = vcat(x, zeros(T, 8n)))
    report((n, ord, frac), prox_msg(T, x, ref_groups(x), T(frac * sum(Float64, x)); entry_points = false, scaled = false))
  end
  x = shuffle(rng, T.(range(1, 2; length = 10^5)))
  for frac in (1e-3, 0.5, 0.999)
    report((10^5, :shuffled, frac), prox_msg(T, x, ref_groups(x), T(frac * sum(Float64, x)); entry_points = false, scaled = false))
  end
  @test ncase >= 1500 && bad == 0
end

@testset "a radius below the rounding of the sum, and a radius near the full sum ($T)" for T in (Float32, Float64)
  rd = PB._AbsRead{T}()
  bad = 0
  # equal entries and a radius below the spacing of their sum: the plain level rounds onto the entries (Michelot's iteration after
  # the handover for n > 2000, Condat's pass alone below)
  for n in (4, 100, 2001, 20_001), y0 in T[1, 0.3, 0.7], a in (eps(y0) / 4, eps(y0), 3 * eps(y0), T(1.25) * eps(T), eps(T) * n * y0 / 1000)
    x = fill(y0, n)
    bad += prox_msg(T, x, ref_groups(x), a; entry_points = n < 1000, scaled = n < 1000) !== nothing
  end
  @test bad == 0
  # a radius within a few spacings of the sum (the level is near zero): through the public API the output is zero when the radius
  # reaches the sum that the kernel forms, and a level within the bounds otherwise
  for x in (fill(T(0.3), 1000), T.(range(1, 2; length = 1000)), [T(1); fill(T(0.1), 5000)])
    groups = ref_groups(x)
    s = PB._linf_sum(x, rd, T)
    for d in (0, 1, 2, 4, 8, 64, 1000)
      a = s
      for _ in 1:d
        a = prevfloat(a)
      end
      @test prox_msg(T, x, groups, a; entry_points = false, scaled = false) === nothing
      d == 0 && @test all(iszero, prox(NormLinf(a), x, one(T)))        # a radius that reaches the sum gives the zero output
    end
    @test all(iszero, prox(NormLinf(2 * s), x, one(T)))
  end
end

@testset "the scaled path: entries near floatmax ($T)" for T in (Float32, Float64)
  # the sum of the magnitudes is 0.9 floatmax, above floatmax/2: the entries are read in units of a power of two; the level of
  # the exceptional path is within the same bounds (the units of the subnormal losses are below these at these sizes)
  bad = 0
  u = Float64(eps(T)) / 2
  for m in (10, 1000, 10^4), cf in (0.1, 0.5, 0.9), ρ in (0.06, 0.25, 1.0), order in (:first, :last, :padded)
    y0 = (1 - cf) / (1 + ρ * Float64(m)^2 * u)
    c = T(0.9) * floatmax(T) / T(1 + m * y0)
    x = c .* tie_input(T, y0, m, 1, 1, order)
    bad += prox_msg(T, x, ref_groups(x), T(cf) * c; entry_points = false, scaled = false) !== nothing
  end
  @test bad == 0
end

@testset "BigFloat and Float16 arguments reach the plain-loop pass" begin
  # the pass for types other than Float32 and Float64 is a plain loop; the routines are called directly (the array methods of the
  # package support Float32 and Float64). BigFloat at 64 bits of precision (the arithmetic of a BigFloat is at the global
  # precision), Float16 with n eps ≤ 1/2
  rng = Xoshiro(55)
  for (T, n, bits) in ((BigFloat, 60, 64), (Float16, 100, 0), (Float16, 400, 0))
    xs = [exp(2 * randn(rng)) for _ in 1:n]
    ar = T === BigFloat ? (u = big(2.0)^-bits, δ = big(nextfloat(zero(BigFloat)))) : arith(T)
    ok = true
    run() = begin
      x = T.(xs)
      groups = ref_groups(x)
      M = maximum(x)
      for f in (1e-4, 0.01, 0.3, 0.9, 0.99)
        a = T(f * Float64(sum(Float64, x)))
        τ = PB._l1ball_threshold!(Vector{T}(undef, n), x, a, PB._AbsRead{T}())
        τm = PB._l1ball_michelot(x, a, PB._AbsRead{T}(), T)
        ok &= level_msg(ar, groups, n, M, a, τ) === nothing && level_msg(ar, groups, n, M, a, τm) === nothing
      end
    end
    T === BigFloat ? setprecision(run, BigFloat, bits) : run()
    @test ok
  end
end

@testset "the routines end, and are bounded by the largest entry, outside their preconditions ($T)" for T in (Float32, Float64)
  rd = PB._AbsRead{T}()
  δ = nextfloat(zero(T))
  x = T[0.5, 1, 2, 3]
  s = sum(x)
  # a at or above the sum: the level is (s - a)/n or below, at most the largest entry
  for a in (s, nextfloat(s), 2s, 10s, floatmax(T) / 4)
    τ = PB._l1ball_michelot(x, a, rd, T)
    τc = PB._l1ball_threshold!(similar(x), x, a, rd)
    @test isfinite(τ) && τ <= maximum(x) && isfinite(τc) && τc <= maximum(x)
  end
  # a tiny radius (the smallest subnormal) and a zero radius (a radius that underflowed): the level is the largest entry within the bounds
  groups = ref_groups(x)
  ar = arith(T)
  for a in (δ, T(1e-30), zero(T))
    τ = PB._l1ball_michelot(x, a, rd, T)
    τc = PB._l1ball_threshold!(similar(x), x, a, rd)
    @test level_msg(ar, groups, 4, 3, a, τ) === nothing && level_msg(ar, groups, 4, 3, a, τc) === nothing
  end
  @test prox(NormLinf(floatmin(T)), x, floatmin(T)) == x           # γλ underflows to zero: the clip changes nothing
  # all entries equal, all zero, and n = 1
  @test PB._l1ball_michelot(zeros(T, 5), zero(T), rd, T) == 0 && PB._l1ball_threshold!(zeros(T, 5), zeros(T, 5), zero(T), rd) == 0
  for y in T[5, 0.3, 1e-3, 3e5], f in (0.1, 0.5, 0.999999)
    a = T(f) * y
    g1 = ref_groups([y])
    @test level_msg(ar, g1, 1, y, a, PB._l1ball_michelot([y], a, rd, T)) === nothing
    @test level_msg(ar, g1, 1, y, a, PB._l1ball_threshold!(T[0], [y], a, rd)) === nothing
  end
  # subnormal entries and radius: the arithmetic is exact except for the division, and the bounds carry the terms δ
  for n in (3, 50), j in (1, n ÷ 2, 3n)
    x = δ * T.(1:n)
    a = j * δ
    a < sum(x) || continue
    g = ref_groups(x)
    @test level_msg(ar, g, n, maximum(x), a, PB._l1ball_michelot(x, a, rd, T)) === nothing
    @test level_msg(ar, g, n, maximum(x), a, PB._l1ball_threshold!(similar(x), x, a, rd)) === nothing
  end
end

@testset "the threshold routines do not allocate ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1005)
  rd = PB._AbsRead{T}()
  # Condat's pass alone (random entries and ties with a head), the handover, and a start far below the level (several passes)
  x1 = gen(rng, T, 1000, 1)
  inputs = (("random", x1, T(0.3) * sum(abs, x1)),
            ("heads and ties", tie_input(T, 0.3, 10^4, 1.0, 1, :padded), T(0.4)),
            ("handover", ones(T, 5000), T(1.5)),
            ("ramp", T.(range(1, 2; length = 3000)), T(100)))
  for (name, x, a) in inputs
    buf = Vector{T}(undef, length(x))
    @test allocs(PB._l1ball_threshold!, buf, x, a, rd) == 0
    @test allocs(PB._l1ball_michelot, x, a, rd, T) == 0
  end
end

end
