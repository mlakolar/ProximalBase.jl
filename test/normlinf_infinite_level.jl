module NormLinfInfiniteLevelTest

# NormLinf with an infinite level is the indicator of {0} (decision 49, E2 after E4): at a finite
# positive step its proximal map is the zero vector at every argument, a NaN or infinite entry
# included, and value_and_prox! returns the value at the stored output (O2), which is 0. The
# standalone value is 0 at the zero vector, Inf at a nonzero finite argument and NaN at a NaN
# entry. A zero, infinite, negative or NaN step keeps its results, and a finite level or step that
# merely overflows the working precision keeps the finite-level computation (R3).
# Outputs are compared with isequal, so a stored zero is +0.0 and NaN equals NaN.

using Test, ProximalBase, SparseArrays, Random
using JLArrays: JLArray

const PB = ProximalBase

# prox, then prox! and value_and_prox! into a fresh output, an output holding a sentinel, and the
# argument itself (a copy); every entry with isequal, and the value against the standalone value at
# the stored output (stored = false leaves that comparison out, for rows whose standalone value at
# the zero output is covered by its own tests)
function check(f, x, γ, expected = zeros(float(eltype(x)), size(x)); value = zero(float(eltype(x))), stored = true)
  T = float(eltype(x))
  @test isequal(prox(f, x, γ), expected)
  for op in (prox!, value_and_prox!)
    fresh = similar(x, T)
    v1 = op(fresh, f, x, γ)
    sentinel = fill(T(-7), size(x))
    op(sentinel, f, x, γ)
    alias = copy(x)
    v3 = op(alias, f, alias, γ)
    @test isequal(fresh, expected) && isequal(sentinel, expected) && isequal(alias, expected)
    if op === value_and_prox!
      @test v1 === value && v3 === value
      stored && @test isequal(v1, f(fresh)) && isequal(v3, f(alias))
    end
  end
end

####################################
#  the proximal map is zero at every entry
####################################

@testset "NormLinf(Inf): rows that fail without the exceptional route" begin
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    xs = (T[1, NaN], T[NaN, 1], T[-Inf, NaN], T[NaN, NaN], T[NaN], T[-3, NaN, 2], T[1, -2, NaN, 0.5],
          T[NaN, -Inf, Inf], T[NaN, floatmax(T), -floatmax(T)])
    for x in xs, γ in (T(1), T(0.5), 1.0, 2, floatmin(T), T(1e30), 1e300)
      check(f, x, γ)
    end
    # a long vector, past the blocks of the reductions, with a NaN in the middle and at the end
    for n in (8 * 1024 + 7, 20_000), k in (n ÷ 2, n)
      x = T.(sin.(1:n))
      x[k] = NaN
      check(f, x, T(1))
    end
    # multidimensional arguments and views
    check(f, T[1 NaN; 2 3], T(1))
    A = reshape(T.(1:24), 2, 3, 4)
    A[2, 1, 3] = NaN
    check(f, A, T(1))
    check(f, view(T[5, 1, NaN, 2, 7], 2:4), T(1))
    check(f, view(T[5, 1, NaN, 2, 7], 1:2:5), T(1))
    check(f, view(A, :, 2:3, 3), T(1))
    check(f, view(A, 2, :, :), T(1))
    # the standalone value at the zero vector, and the value at the stored zero output
    @test f(zeros(T, 3)) === zero(T) && f(T[-0.0, 0.0]) === zero(T)
    @test f(view(zeros(T, 4), 1:2:4)) === zero(T) && f(zeros(T, 2, 2)) === zero(T)
  end
  # the level and the step in different precisions
  check(NormLinf(Inf32), [1.0, NaN], 1.0)
  check(NormLinf(Inf32), Float32[1, NaN], 1.0)
  check(NormLinf(Inf), Float32[1, NaN], 1.0)
  check(NormLinf(Inf), Float32[NaN, 1], 1f0)
  check(NormLinf(Inf32), [NaN, 2.0], 1f0)
  check(NormLinf(Inf), [1.0, NaN], 1f0)
  @test NormLinf(Inf)(zeros(3)) == 0 && NormLinf(Inf)(zeros(Float32, 3)) === 0f0
  @test NormLinf(Inf32)(zeros(3)) === 0.0 && NormLinf(Inf32)(zeros(Float32, 3)) === 0f0
  @test NormLinf(Inf)(Float32[0, 0, 0]) == 0
end

@testset "NormLinf(Inf): regression guards, rows that already pass" begin
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    xs = (T[1, -2, 0.5], T[Inf, 1], T[-Inf, 1], T[Inf, -Inf], zeros(T, 3), T[-0.0, 0.0], T[], T[3], T[-3], T[0],
          T[1 -2; 0.5 4], T[floatmax(T), -floatmax(T)], T[floatmin(T) / 4, 1])
    for x in xs, γ in (T(1), T(0.5), 1.0, floatmin(T), T(1e30), 1e300)
      check(f, x, γ; stored = false)
    end
    x = T.(sin.(1:20_000))
    check(f, x, T(1); stored = false)
    check(f, view(x, 3:2:9999), T(1); stored = false)
  end
end

@testset "NormLinf(Inf): standalone values" begin
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    for n in 0:3
      @test f(zeros(T, n)) === zero(T)
    end
    @test f(T[-0.0, 0.0]) === zero(T) && f(zeros(T, 2, 3)) === zero(T)
    @test f(T[0, 1]) === T(Inf) && f(T[1, 2]) === T(Inf) && f(T[-3]) === T(Inf)
    @test f(T[0, floatmin(T) / 4]) === T(Inf) && f(T[0, nextfloat(zero(T))]) === T(Inf)
    @test f(T[Inf, 1]) === T(Inf) && f(T[-Inf, 0]) === T(Inf) && f(T[floatmax(T), 0]) === T(Inf)
    @test isnan(f(T[0, NaN])) && isnan(f(T[NaN, 1])) && isnan(f(T[Inf, NaN])) && isnan(f(T[NaN, -Inf]))
    @test isnan(f(T[NaN])) && isnan(f(T[NaN, NaN]))
    x = T.(sin.(1:20_000))
    @test f(x) === T(Inf)
    x[9_001] = NaN
    @test isnan(f(x))
    @test f(zeros(T, 20_000)) === zero(T)
  end
  @test NormLinf(Inf)(Float32[0, 1]) === Inf32 && NormLinf(Inf32)([0.0, 1.0]) === Inf
  @test isnan(NormLinf(Inf)(Float32[0, NaN]))
end

@testset "NormLinf(Inf): the steps that keep their results" begin
  # zero, infinite, negative and NaN steps are not covered by the rule (no endpoint is documented
  # for them); the results at a finite argument are the earlier ones
  nan = NaN
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    x = T[1, 2]
    @test isequal(prox(f, x, T(0)), T[nan, nan])
    @test isequal(prox(f, x, T(Inf)), zeros(T, 2))
    @test isequal(prox(f, x, T(-1)), T[1, 2])
    @test isequal(prox(f, x, T(nan)), T[nan, nan])
    for γ in (T(0), T(Inf), T(-1), T(nan))
      y = similar(x)
      @test isequal(prox!(y, f, x, γ), prox(f, x, γ))
    end
  end
end

####################################
#  finite levels, and finite parameters that overflow (R3)
####################################

@testset "NormLinf: finite level or step whose product overflows is not infinite" begin
  # the finite-level computation is unchanged: the NaN entry stays NaN, and the output is not the
  # zero vector of the infinite level
  for (f, x, γ) in ((NormLinf(floatmax(Float64)), [1.0, NaN], floatmax(Float64)),
                    (NormLinf(1e300), Float32[1, NaN], 1f0),
                    (NormLinf(1e10), [1.0, NaN], 1e300),
                    (NormLinf(1e300), Float32[NaN, 1], 1f0))
    y = prox(f, x, γ)
    @test any(isnan, y) && any(!iszero, y)
    @test isnan(y[findfirst(isnan, x)])
  end
  # at a finite argument the level is far above the ℓ1 norm: the result is zero, as before
  @test isequal(prox(NormLinf(floatmax(Float64)), [1.0, 2.0], floatmax(Float64)), [0.0, 0.0])
  @test isequal(prox(NormLinf(1e300), Float32[1, 2], 1f0), Float32[0, 0])
  @test isequal(prox(NormLinf(1e10), [1.0, 2.0], 1e300), [0.0, 0.0])
  @test isequal(prox(NormLinf(1.0), [3.0, -1.0], 1.0), [2.0, -1.0])
  @test NormLinf(floatmax(Float64))(zeros(3)) === 0.0 && NormLinf(1e300)(zeros(Float32, 3)) === 0f0
  @test NormLinf(1e300)(Float32[0, 1]) === Inf32
  @test NormLinf(1.0)(zeros(3)) === 0.0 && isnan(NormLinf(0.0)([1.0, NaN]))
end

# the thresholds of the unchanged finite-level path against an independent sort-based reference:
# x minus the Euclidean projection onto the ℓ1 ball of radius γλ
function l1_projection(x::AbstractVector{<:Real}, r::Real)
  v = Float64.(x)
  sum(abs, v) <= r && return v
  u = sort(abs.(v); rev = true)
  cs = cumsum(u)
  ρ = maximum(j for j in eachindex(u) if u[j] - (cs[j] - r) / j > 0)
  θ = (cs[ρ] - r) / ρ
  return sign.(v) .* max.(abs.(v) .- θ, 0)
end

@testset "NormLinf: finite levels against the ℓ1-ball projection (seeded)" begin
  rng = MersenneTwister(20261105)
  for T in (Float32, Float64), λ in (0.3, 1.0, 1e-3, 1e5), n in (1, 2, 3, 10, 100, 1000), _ in 1:6
    tol = T === Float32 ? 1e-5 : 1e-12
    x = T.(randn(rng, n) .* exp.(randn(rng, n)))
    γ = T(exp(2 * randn(rng)))
    f = NormLinf(T(λ))
    r = Float64(γ) * Float64(T(λ))
    expected = Float64.(x) .- l1_projection(x, r)
    scale = max(1.0, maximum(abs, Float64.(x)), r)
    y = prox(f, x, γ)
    @test eltype(y) === T
    @test maximum(abs, Float64.(y) .- expected) <= tol * scale
    z = similar(x)
    v = value_and_prox!(z, f, x, γ)
    @test z == y
    @test isapprox(Float64(v), Float64(T(λ)) * maximum(abs, Float64.(z)); rtol = 10 * tol)
    w = copy(x)
    @test prox!(w, f, w, γ) === w && w == y
  end
end

####################################
#  wrappers and iterates
####################################

@testset "NormLinf(Inf): wrappers and iterates" begin
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    nan = T(NaN)
    # a translation
    check(Translate(f, T[2, 3]), T[1, NaN], T(1), T[2, 3])
    # a constraint classified by the infinite level
    for x in (T[1, NaN, 2], T[Inf, NaN, 0.5], T[1, 2, 3], T[NaN, NaN, NaN])
      check(Constrained(f, IndBallL2(T(1))), x, T(1))
    end
    # a Moreau envelope of the indicator of {0}: the quadratic ‖x‖²/(2γ) at a finite argument, and
    # the finite entry keeps its value beside a NaN entry
    me = MoreauEnvelope(f, T(1))
    @test prox(me, T[1, 2], T(1)) ≈ T[0.5, 1]
    @test isequal(prox(me, T[1, NaN], T(1)), T[0.5, nan])
    # iterates
    x = SparseIterate(sparsevec(T[1, NaN, 0]))
    @test isequal(value_and_prox!(x, f, x, T(1)), zero(T)) && isequal(Vector(x), zeros(T, 3))
    @test isequal(Vector(prox(f, SparseIterate(sparsevec(T[1, NaN, 0])), T(1))), zeros(T, 3))
    @test isequal(Matrix(prox(f, SymmetricSparseIterate(T[1 NaN; NaN 2]), T(1))), zeros(T, 2, 2))
    s = SymmetricSparseIterate(T[1 NaN; NaN 2])
    @test isequal(value_and_prox!(s, f, s, T(1)), zero(T)) && isequal(Matrix(s), zeros(T, 2, 2))
    # fibers: the off-diagonal fibers are zero, the diagonal ones are the identity here
    fb = FiberSum(f; diagonal = nothing)
    X = ones(T, 2, 2, 3)
    X[1, 2, 1] = NaN
    X[2, 1, 2] = Inf
    X[1, 1, 3] = 5
    expected = copy(X)
    expected[1, 2, :] .= 0
    expected[2, 1, :] .= 0
    Y = prox(fb, X, T(1))
    @test isequal(Y, expected)
    Z = similar(X)
    v = value_and_prox!(Z, fb, X, T(1))
    @test isequal(Z, expected) && v === zero(T) && isequal(v, fb(Z))
    W = copy(X)
    @test value_and_prox!(W, fb, W, T(1)) === zero(T) && isequal(W, expected)
  end
end

@testset "NormLinf(Inf): the value at the zero vector reaches the callers of the reduction" begin
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    @test FiberSum(f; diagonal = nothing)(zeros(T, 3, 3, 2)) === zero(T)
    @test FiberSum(f)(zeros(T, 3, 3, 2)) === zero(T)
    @test Constrained(f, IndBallL2(T(1)))(zeros(T, 3)) === zero(T)
    @test Spectral(f)(zeros(T, 3, 3)) === zero(T)
    @test SingularValue(f)(zeros(T, 3, 2)) === zero(T)
    @test Scaled(f, T(2))(zeros(T, 3)) === zero(T)
    @test isinf(Spectral(f)(T[1 0; 0 0])) && isinf(SingularValue(f)(T[1 0; 0 0]))
    @test isinf(Constrained(f, IndBallL2(T(1)))(T[0.5, 0]))
    # the conjugate of the indicator of an ℓ1 ball of infinite radius shares the reduction
    c = Conjugate(IndBallL1(T(Inf)))
    @test c(zeros(T, 3)) === zero(T)
    @test c(T[1, 0]) === T(Inf)
  end
end

####################################
#  device arrays
####################################

@testset "NormLinf(Inf): device arrays agree with the CPU" begin
  for T in (Float32, Float64)
    f = NormLinf(T(Inf))
    for x in (T[1, NaN], T[NaN, 1], T[-Inf, NaN], T[1, 2, -3], T[Inf, 1], zeros(T, 3)), γ in (T(1), 1.0, T(1e30))
      @test isequal(Array(prox(f, JLArray(x), γ)), zeros(T, length(x)))
      y = JLArray(fill(T(-7), length(x)))
      v = value_and_prox!(y, f, JLArray(x), γ)
      @test v === zero(T) && isequal(Array(y), zeros(T, length(x)))
      @test f(y) === zero(T)
      a = JLArray(copy(x))
      @test value_and_prox!(a, f, a, γ) === zero(T) && isequal(Array(a), zeros(T, length(x)))
      b = JLArray(copy(x))
      @test prox!(b, f, b, γ) === b && isequal(Array(b), zeros(T, length(x)))
    end
    # the standalone value
    @test f(JLArray(zeros(T, 3))) === zero(T)
    @test f(JLArray(T[0, 1])) === T(Inf) && f(JLArray(T[Inf, 1])) === T(Inf)
    @test isnan(f(JLArray(T[0, NaN]))) && isnan(f(JLArray(T[Inf, NaN])))
    # steps that keep their results, and the finite-level path
    @test isequal(Array(prox(f, JLArray(T[1, 2]), T(Inf))), zeros(T, 2))
    @test isequal(Array(prox(NormLinf(T(1)), JLArray(T[3, 1]), T(1))), T[2, 1])
    @test NormLinf(T(1))(JLArray(zeros(T, 3))) === zero(T)
  end
  @test isequal(Array(prox(NormLinf(Inf32), JLArray([1.0, NaN]), 1.0)), zeros(2))
  @test isequal(Array(prox(NormLinf(Inf), JLArray(Float32[1, NaN]), 1.0)), zeros(Float32, 2))
  @test isequal(Array(prox(NormLinf(1e300), JLArray(Float32[1, 2]), 1f0)), zeros(Float32, 2))
  @test any(isnan, Array(prox(NormLinf(1e300), JLArray(Float32[1, NaN]), 1f0)))
end

end
