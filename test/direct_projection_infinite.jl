module DirectProjectionInfiniteTest

# #469, Stage 2 (decision 48, E2/E3/E5/O2/M2/R1-R3): the direct projections IndBallL1, IndBallL2,
# IndSphereL2 and IndSimplex at NaN and infinite arguments. The expected points are derived by
# hand and checked against finite BigFloat sequences x(M), M → ∞, projected by a sort-based
# reference that shares no code with the package. A path-dependent literal Inf is asserted to
# give the all-NaN block, witnessed by two sequences with different limits.

using Test
using ProximalBase
using LinearAlgebra

# --- a reference projection that shares nothing with the package ---------------------------

const BF = BigFloat

# projection of x onto {z ≥ 0, Σ z = r} by sorting, in BigFloat
function ref_simplex_eq(x::AbstractVector{BF}, r)
  u = sort(x; rev = true)
  c = cumsum(u)
  ρ = 0
  for k in eachindex(u)
    u[k] - (c[k] - r) / k > 0 && (ρ = k)
  end
  τ = (c[ρ] - r) / ρ
  max.(x .- τ, 0)
end
ref_simplex(x, r, eq) = (eq || sum(v -> max(v, 0), x) > r) ? ref_simplex_eq(x, r) : max.(x, 0)
ref_l1(x, r) = sum(abs, x) <= r ? copy(x) : sign.(x) .* ref_simplex_eq(abs.(x), r)
ref_l2(x, r) = (n = norm(x); n <= r ? copy(x) : x .* (r / n))
ref_sphere(x, r) = x .* (r / norm(x))

const families = (
  (:l1, r -> IndBallL1(r), (x, r) -> ref_l1(x, r)),
  (:l2, r -> IndBallL2(r), (x, r) -> ref_l2(x, r)),
  (:sphere, r -> IndSphereL2(r), (x, r) -> ref_sphere(x, r)),
  (:simplex, r -> IndSimplex(r), (x, r) -> ref_simplex(x, r, true)),
  (:simplex_le, r -> IndSimplex(r; equality = false), (x, r) -> ref_simplex(x, r, false)))

# the sequence point of an extended-real vector: ±Inf entries become ±M (a multiple k·M when a
# pair is given as (k, sign)), finite entries stay
point(x, M) = BF[isinf(v) ? sign(v) * M : BF(v) for v in x]

# equal up to rounding; the NaN entries must coincide
function close_to(a, b; rtol)
  size(a) == size(b) || return false
  all(eachindex(a)) do i
    isnan(a[i]) || isnan(b[i]) ? (isnan(a[i]) && isnan(b[i])) : isapprox(a[i], b[i]; rtol, atol = rtol)
  end
end

# every route (prox, prox!, value_and_prox!, separate and aliased outputs, Float32 and Float64
# in each role, steps 0, 1.5 and Inf) returns `want`; the value is NaN exactly for a NaN block
function check_routes(f, x, want; label = "")
  for S in (Float64, Float32), T in (Float64, Float32), γ in (0.0, 1.5, Inf)
    rtol = 16 * eps(Float32) * (S == Float32 || T == Float32)  + 16 * eps(Float64)
    xs = S.(x)
    wv = any(isnan, want) ? NaN : 0.0
    y = fill(T(7), size(x))
    @test prox!(y, f, xs, γ) === y
    @test close_to(y, want; rtol)
    y = fill(T(7), size(x))
    v = value_and_prox!(y, f, xs, γ)
    @test close_to(y, want; rtol)
    @test isnan(v) == isnan(wv) && (isnan(v) || v == 0)
    @test isequal(xs, S.(x))                            # the source is not written
    if S == T
      z = copy(xs)
      @test prox!(z, f, z, γ) === z
      @test close_to(z, want; rtol)
      z = copy(xs)
      v = value_and_prox!(z, f, z, γ)
      @test close_to(z, want; rtol)
      @test isnan(v) == isnan(wv) && (isnan(v) || v == 0)
    end
  end
  @test close_to(prox(f, x), want; rtol = 16 * eps())
end

nans(n) = fill(NaN, n)

# 2400 bits resolve 1 next to 2e308 (the sequences use M up to 1e100, the overflow cases 1e308)
setprecision(BF, 2400) do
@testset "direct projections at NaN and infinite arguments" begin

@testset "task table (radius 1)" begin
  r2 = 1 / sqrt(2)
  for f in (IndBallL1(1), IndBallL2(1), IndSphereL2(1))
    check_routes(f, [Inf, 2.0], [1.0, 0.0])
    check_routes(f, [-Inf, 2.0], [-1.0, 0.0])
    check_routes(f, [Inf, Inf], nans(2))
  end
  check_routes(IndBallL1(0), [Inf, 2.0], [0.0, 0.0])
  check_routes(IndBallL1(0), [NaN, 2.0], [0.0, 0.0])
  check_routes(IndBallL2(0), [Inf, 2.0], [0.0, 0.0])
  check_routes(IndBallL2(0), [NaN, 2.0], [0.0, 0.0])
  check_routes(IndSimplex(1), [Inf, -Inf], [1.0, 0.0])
  check_routes(IndSimplex(1), [-Inf, 0.0], [0.0, 1.0])
  check_routes(IndSimplex(1), [-Inf, -Inf], nans(2))
  check_routes(IndSimplex(1), [Inf, Inf], nans(2))
  check_routes(IndSimplex(1; equality = false), [-Inf, -Inf], [0.0, 0.0])
  check_routes(IndSimplex(1; equality = false), [-Inf, 0.0], [0.0, 0.0])
  check_routes(IndSimplex(1), [-Inf], [1.0])
  check_routes(IndSimplex(1), [NaN], [1.0])
  check_routes(IndSimplex(0), [NaN, Inf], [0.0, 0.0])
  check_routes(IndSimplex(0; equality = false), [NaN, -Inf], [0.0, 0.0])
  check_routes(IndSimplex(1; equality = false), [Inf, 0.0], [1.0, 0.0])
  check_routes(IndSimplex(1; equality = false), [Inf], [1.0])
  check_routes(IndSimplex(1; equality = false), [-Inf], [0.0])
  check_routes(IndSimplex(1; equality = false), [NaN], [NaN])    # a clamp of NaN depends on it
  check_routes(IndSimplex(1), [Inf], [1.0])
  # radius other than one
  check_routes(IndBallL1(2.5), [3.0, -Inf, 1.0], [0.0, -2.5, 0.0])
  check_routes(IndBallL2(0.25), [Inf, 1.0, 1.0], [0.25, 0.0, 0.0])
  check_routes(IndSphereL2(4), [0.0, 1.0, Inf], [0.0, 0.0, 4.0])
  check_routes(IndSimplex(3), [0.0, Inf, -Inf], [0.0, 3.0, 0.0])
  @test r2 > 0
end

@testset "single infinity: every position and sign, with finite remainders" begin
  rest = [0.3, -0.7, 2.0]
  for k in 1:4, s in (1.0, -1.0)
    x = insert!(copy(rest), k, s * Inf)
    e = zeros(4); e[k] = s
    for f in (IndBallL1(1), IndBallL2(1), IndSphereL2(1), IndBallL1(3), IndBallL2(0.5), IndSphereL2(2))
      check_routes(f, x, f.r .* e)
    end
    if s > 0
      for f in (IndSimplex(1), IndSimplex(1; equality = false), IndSimplex(2))
        check_routes(f, x, f.r .* e)
      end
    end
  end
end

@testset "simplex with negative infinities and finite remainders" begin
  # reduced simplex on the finite entries
  check_routes(IndSimplex(1), [-Inf, 0.3, 0.9], [0.0, 0.2, 0.8])
  check_routes(IndSimplex(1; equality = false), [-Inf, 0.3, 0.9], [0.0, 0.2, 0.8])
  check_routes(IndSimplex(1), [-Inf, 0.2, 0.3], [0.0, 0.45, 0.55])
  check_routes(IndSimplex(1; equality = false), [-Inf, 0.2, 0.3], [0.0, 0.2, 0.3])
  check_routes(IndSimplex(1), [0.2, -Inf, -Inf, 0.3], [0.45, 0.0, 0.0, 0.55])
  check_routes(IndSimplex(1), [-Inf, -Inf, 0.4], [0.0, 0.0, 1.0])
  check_routes(IndSimplex(1; equality = false), [-Inf, -Inf, -0.4], [0.0, 0.0, 0.0])
  # +Inf next to -Inf
  check_routes(IndSimplex(1), [-Inf, Inf, 0.5, -Inf], [0.0, 1.0, 0.0, 0.0])
  # equality with every entry -Inf: path dependent for two or more, the singleton for one
  check_routes(IndSimplex(1), [-Inf, -Inf, -Inf], nans(3))
  check_routes(IndSimplex(1; equality = false), [-Inf, -Inf, -Inf], zeros(3))
  check_routes(IndSimplex(2), [Inf, Inf, -Inf], nans(3))
  check_routes(IndSimplex(2; equality = false), [0.1, Inf, Inf], nans(3))
end

@testset "several infinities and NaN give the NaN block" begin
  for f in (IndBallL1(1), IndBallL2(1), IndSphereL2(1), IndBallL1(2), IndSphereL2(3))
    check_routes(f, [Inf, Inf, 3.0], nans(3))
    check_routes(f, [Inf, -Inf, 3.0], nans(3))
    check_routes(f, [-Inf, -Inf], nans(2))
    check_routes(f, [NaN, 0.0], nans(2))
    check_routes(f, [NaN, Inf], nans(2))
    check_routes(f, [-Inf, 2.0, NaN], nans(3))
    check_routes(f, [0.5, 0.25, NaN], nans(3))
  end
  for f in (IndSimplex(1), IndSimplex(1; equality = false))
    check_routes(f, [NaN, 0.0], nans(2))
    check_routes(f, [0.3, NaN, 0.9], nans(3))                  # a NaN that Condat's pass would skip
    check_routes(f, [0.3, 0.9, NaN], nans(3))
    check_routes(f, [0.3, NaN, 0.9, Inf], nans(4))
    check_routes(f, [NaN, -Inf], nans(2))
    check_routes(f, [Inf, NaN, 0.9], nans(3))
  end
  # zero radius and the length-one equality simplex do not depend on the input
  check_routes(IndBallL1(0), [NaN, NaN], zeros(2))
  check_routes(IndBallL2(0), [NaN, -Inf, 1.0], zeros(3))
  check_routes(IndSimplex(0), [NaN, NaN], zeros(2))
  check_routes(IndSimplex(2.5), [NaN], [2.5])
  check_routes(IndSimplex(2.5), [-Inf], [2.5])
end

@testset "NaN is not an infinity: finite entries keep the ordinary projection" begin
  for (f, x) in ((IndBallL1(1), [3.0, 4.0]), (IndBallL2(1), [3.0, 4.0]), (IndSphereL2(1), [3.0, 4.0]),
                 (IndSimplex(1), [3.0, 0.0]), (IndSimplex(1; equality = false), [-1.0, 0.5]))
    @test all(isfinite, prox(f, x))
    @test value_and_prox!(similar(x), f, x, 1.0) === 0.0
  end
end

@testset "independent sequences: unique limits, and two limits where the direction is free" begin
  M = BF(10)^40
  a = [0.3, -0.7, 2.0]
  for (name, mk, ref) in families, k in 1:4
    f = mk(1)
    for s in (1.0, -1.0)
      name in (:simplex, :simplex_le) && s < 0 && continue
      x = insert!(copy(a), k, s * Inf)
      limit = ref(point(x, M), BF(1))
      @test close_to(prox(f, x), Float64.(limit); rtol = 1e-12)
    end
  end
  # negative infinity of the simplex, finite remainder: the limit of the reduced problem
  for eq in (true, false), k in 1:4
    f = IndSimplex(1; equality = eq)
    x = insert!([0.3, 0.9, 0.1], k, -Inf)
    @test close_to(prox(f, x), Float64.(ref_simplex(point(x, M), BF(1), eq)); rtol = 1e-12)
  end
  # two infinite entries: (M, M) and (M, 2M) have different limits, so the literal Inf is NaN
  witnesses = (
    (:l1, [Inf, Inf, 0.3], BF[M, M, 3//10], BF[M, 2M, 3//10]),
    (:l1, [Inf, -Inf, 0.3], BF[M, -M, 3//10], BF[M, -2M, 3//10]),
    (:l1, [-Inf, -Inf, 0.3], BF[-M, -M, 3//10], BF[-M, -2M, 3//10]),
    (:l2, [Inf, Inf, 0.3], BF[M, M, 3//10], BF[M, 2M, 3//10]),
    (:l2, [Inf, -Inf, 0.3], BF[M, -M, 3//10], BF[M, -2M, 3//10]),
    (:sphere, [Inf, Inf, 0.3], BF[M, M, 3//10], BF[M, 2M, 3//10]),
    (:sphere, [-Inf, Inf, 0.3], BF[-M, M, 3//10], BF[-M, 2M, 3//10]),
    (:simplex, [Inf, Inf, 0.3], BF[M, M, 3//10], BF[M, 2M, 3//10]),
    (:simplex, [Inf, Inf, -Inf], BF[M, M, -M], BF[M, 2M, -M]),
    (:simplex, [-Inf, -Inf], BF[-M, -M], BF[-M, -2M]),
    (:simplex_le, [Inf, Inf, 0.3], BF[M, M, 3//10], BF[M, 2M, 3//10]))
  for (name, x, p, q) in witnesses
    _, mk, ref = first(filter(t -> t[1] == name, collect(families)))
    a1, a2 = Float64.(ref(p, BF(1))), Float64.(ref(q, BF(1)))
    @test norm(a1 - a2, Inf) > 0.05                          # two finite paths, two limits
    @test all(isnan, prox(mk(1), x))                         # so the literal Inf is the NaN block
  end
  # at least the classic witnesses are present: ℓ1 and ℓ2 ball, sphere, equality simplex
  @test norm(Float64.(ref_l1(BF[M, M], BF(1)) - ref_l1(BF[M, 2M], BF(1))), Inf) > 0.4
  @test norm(Float64.(ref_l2(BF[M, M], BF(1)) - ref_l2(BF[M, 2M], BF(1))), Inf) > 0.2
  @test norm(Float64.(ref_sphere(BF[M, M], BF(1)) - ref_sphere(BF[M, 2M], BF(1))), Inf) > 0.2
  @test norm(Float64.(ref_simplex(BF[M, M], BF(1), true) - ref_simplex(BF[M, 2M], BF(1), true)), Inf) > 0.4
  @test norm(Float64.(ref_simplex(BF[-M, -M], BF(1), true) - ref_simplex(BF[-M, -2M], BF(1), true)), Inf) > 0.4
  # the inequality simplex with every entry -Inf is determined: zero
  @test all(iszero, ref_simplex(BF[-M, -M], BF(1), false))
end

@testset "finite data that overflows or narrows is not infinite input (T3, E5, R2)" begin
  M = BF(10)^100
  # Float64 argument, Float32 destination: both the argument and the answer are representable
  for (name, mk, ref) in families, v in (1e100, -1e100), γ in (1.0, 0.0, Inf)
    name in (:simplex, :simplex_le) && v < 0 && continue
    f = mk(1)
    x = [v, 2.0]
    want = Float32.(ref(BF.(x), BF(1)))
    y = zeros(Float32, 2)
    @test close_to(prox!(y, f, x, γ), want; rtol = 4 * eps(Float32))
    y .= 7
    val = value_and_prox!(y, f, x, γ)
    @test close_to(y, want; rtol = 4 * eps(Float32)) && val === 0f0
    z = prox(f, x)
    @test close_to(z, Float64.(ref(BF.(x), BF(1))); rtol = 1e-12)
  end
  # the equality simplex at -1e100: the finite entry takes the whole mass
  for eq in (true, false)
    f = IndSimplex(1; equality = eq)
    y = zeros(Float32, 2)
    @test close_to(prox!(y, f, [-1e100, 2.0], 1.0), Float32[0, 1]; rtol = 4 * eps(Float32))
  end
  # large pairs, enormous radii and values near floatmax
  for (name, mk, ref) in families
    for x in ([1e308, 1e308, 2.0], [-1e308, 1e308, 0.5], [1e154, 1e154, 3.0], [1e300, 1.0, 1e300])
      f = mk(1)
      want = ref(BF.(x), BF(1))
      @test close_to(prox(f, x), Float64.(want); rtol = 1e-12)
      @test close_to(prox(f, Float32.(clamp.(x, -3e38, 3e38))),
                     Float32.(ref(BF.(Float32.(clamp.(x, -3e38, 3e38))), BF(1))); rtol = 8 * eps(Float32))
    end
    f = mk(1e200)
    x = [1e308, 1e308, 2.0]
    @test close_to(prox(f, x), Float64.(ref(BF.(x), BF(1e200))); rtol = 1e-12)
  end
  # Float32 near floatmax
  for (name, mk, ref) in families
    f = mk(1)
    x = Float32[3e38, 3e38, 2]
    @test close_to(prox(f, x), Float32.(ref(BF.(x), BF(1))); rtol = 8 * eps(Float32))
  end
  # ordinary finite results are the ones of the base: spot checks against the reference
  for (name, mk, ref) in families
    f = mk(1)
    for x in ([3.0, 0.0, 4.0], [0.2, 0.3, 0.1], [-1.0, 0.5, 2.0], [1e-3, 2e-3, 5e-4])
      @test close_to(prox(f, x), Float64.(ref(BF.(x), BF(1))); rtol = 1e-12)
    end
  end
end

@testset "extremes of the finite path, and NaN next to a large Float32 argument" begin
  # a norm that underflows in a Float32 destination is not the zero point of the sphere
  y = zeros(Float32, 3)
  @test close_to(prox!(y, IndSphereL2(1.0), [1e-300, -1e-300, 0.0], 1.0), Float32[1/sqrt(2), -1/sqrt(2), 0]; rtol = 1e-6)
  # the equality simplex skips a NaN that is not the first entry: it must still reach every output
  for f in (IndSimplex(1f0), IndSimplex(1f0; equality = false)), x in (Float32[1e4, 2e4, 1e4, NaN], Float32[1e30, NaN, 2])
    @test all(isnan, prox(f, x))
    @test isnan(value_and_prox!(similar(x), f, x, 1f0))
  end
  # ordinary Float32 and Float64 arguments are projected by the plain threshold
  for T in (Float32, Float64), n in (3, 50), scale in (1, 1e3)
    x = T.(scale .* cos.(1.0:n) .+ scale)
    for (name, mk, ref) in families
      @test close_to(prox(mk(one(T)), x), T.(ref(BF.(x), BF(1))); rtol = 1000 * eps(T) * scale)
    end
  end
end

@testset "SymmetricSparseIterate: a mirrored pair is one coordinate counted twice (M2)" begin
  M = BF(10)^40
  p = 3
  base = [0.1 0.5 -0.2; 0.5 0.2 0.4; -0.2 0.4 0.3]
  put(i, j, v) = (X = copy(base); X[i, j] = v; X[j, i] = v; X)
  full(X, M) = reshape(BF[isinf(v) ? sign(v) * M : BF(v) for v in X], size(X))
  mat(y) = Matrix(y)
  r = 1.0
  for (name, mk, ref) in families, (i, j) in ((2, 1), (3, 1), (3, 2), (1, 1), (2, 2), (3, 3)), s in (1.0, -1.0)
    name in (:simplex, :simplex_le) && s < 0 && continue
    f = mk(r)
    X = put(i, j, s * Inf)
    y = SymmetricSparseIterate(p)
    v = value_and_prox!(y, f, X, 1.0)
    want = reshape(Float64.(ref(vec(full(X, M)), BF(r))), p, p)
    @test close_to(mat(y), want; rtol = 1e-12)
    @test v == 0
    # hand-derived: ±r on the diagonal, ±r/2 (ℓ1, simplex) or ±r/√2 (ℓ2) on each mirrored entry
    w = i == j ? 1.0 : name in (:l1, :simplex, :simplex_le) ? 0.5 : 1 / sqrt(2)
    @test mat(y)[i, j] ≈ s * w * r && mat(y)[j, i] ≈ s * w * r
    @test count(!iszero, mat(y)) == (i == j ? 1 : 2)
    # aliasing a symmetric matrix input, and a SymmetricSparseIterate source
    ys = SymmetricSparseIterate(p)
    prox!(ys, f, SymmetricSparseIterate(X), 1.0)
    @test mat(ys) == mat(y)
  end
  # two independent coordinates: a diagonal and a pair, or two pairs
  for (name, mk, ref) in families
    f = mk(1)
    X = put(2, 1, Inf); X[3, 3] = Inf
    y = SymmetricSparseIterate(p)
    v = value_and_prox!(y, f, X, 1.0)
    @test all(isnan, mat(y)) && isnan(v)
    X = put(2, 1, Inf); X[3, 2] = X[2, 3] = Inf
    y = SymmetricSparseIterate(p)
    @test all(isnan, mat(prox!(y, f, X, 1.0)))
    # an asymmetric pair (+Inf, -Inf) has the symmetric part Inf - Inf = NaN (E2)
    X = copy(base); X[2, 1] = Inf; X[1, 2] = -Inf
    y = SymmetricSparseIterate(p)
    v = value_and_prox!(y, f, X, 1.0)
    @test all(isnan, mat(y)) && isnan(v)
    # an asymmetric pair (+Inf, 3) has the symmetric part +Inf
    X = copy(base); X[2, 1] = Inf; X[1, 2] = 3.0
    y = SymmetricSparseIterate(p)
    prox!(y, f, X, 1.0)
    w = name in (:l1, :simplex, :simplex_le) ? 0.5 : 1 / sqrt(2)
    @test mat(y)[2, 1] ≈ w && mat(y)[1, 2] ≈ w
    # a NaN in one entry of a pair
    X = copy(base); X[2, 1] = NaN
    y = SymmetricSparseIterate(p)
    @test all(isnan, mat(prox!(y, f, X, 1.0)))
  end
  # a dense matrix output has no mirrored storage: an infinite symmetric pair is two coordinates
  for (name, mk, ref) in families
    X = put(2, 1, Inf)
    @test all(isnan, prox(mk(1), X))
    X = copy(base); X[2, 2] = Inf
    @test prox(mk(1), X)[2, 2] == 1
  end
  # the 1×1 case: one coordinate, weight one
  y = SymmetricSparseIterate(1)
  @test mat(prox!(y, IndSimplex(2), reshape([NaN], 1, 1), 1.0)) == reshape([2.0], 1, 1)
  y = SymmetricSparseIterate(1)
  @test mat(prox!(y, IndBallL2(2), reshape([-Inf], 1, 1), 1.0)) == reshape([-2.0], 1, 1)
  y = SymmetricSparseIterate(2)
  @test mat(prox!(y, IndBallL2(0), [NaN 1.0; 1.0 NaN], 1.0)) == zeros(2, 2)
  # the negative infinity of a mirrored pair in the simplex: the reduced finite problem
  for eq in (true, false)
    f = IndSimplex(1; equality = eq)
    X = put(2, 1, -Inf)
    y = SymmetricSparseIterate(p)
    prox!(y, f, X, 1.0)
    want = reshape(Float64.(ref_simplex(vec(full(X, M)), BF(1), eq)), p, p)
    @test close_to(mat(y), want; rtol = 1e-12)
  end
end

@testset "SparseIterate output" begin
  for (name, mk, ref) in families
    f = mk(1)
    y = SparseIterate(3)
    v = value_and_prox!(y, f, [0.2, Inf, 0.5], 1.0)
    @test collect(y) == [0.0, 1.0, 0.0] && v == 0
    y = SparseIterate(3)
    v = value_and_prox!(y, f, [0.2, Inf, NaN], 1.0)
    @test all(isnan, collect(y)) && isnan(v)
  end
end

@testset "invalid steps and shapes write nothing, also at exceptional arguments" begin
  for f in (IndBallL1(1), IndBallL2(1), IndSphereL2(1), IndSimplex(1), IndSimplex(1; equality = false)),
      x in ([Inf, 0.0], [NaN, 1.0], [-Inf, -Inf]), γ in (-1.0, NaN, -Inf)
    for op in (prox!, value_and_prox!)
      y = fill(7.0, 2)
      @test_throws ArgumentError op(y, f, x, γ)
      @test y == fill(7.0, 2)
      z = copy(x)
      @test_throws ArgumentError op(z, f, z, γ)
      @test isequal(z, x)
    end
  end
  for f in (IndBallL1(1), IndBallL2(1), IndSphereL2(1), IndSimplex(1))
    y = fill(7.0, 3)
    @test_throws DimensionMismatch prox!(y, f, [Inf, 0.0], 1.0)
    @test y == fill(7.0, 3)
  end
end

@testset "functions are unchanged by the call" begin
  for (name, mk, ref) in families
    f = mk(1)
    for x in ([Inf, 1.0], [NaN, 1.0], [Inf, Inf], [0.3, 0.9])
      prox(f, x)
      prox(f, x)
      @test f.r == 1
    end
  end
end

end # outer testset

end # setprecision

end # module
