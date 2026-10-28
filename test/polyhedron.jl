module PolyhedronTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random
using ..TestUtils

const PB = ProximalBase

# Independent small-QP oracle: eliminate equalities by SVD, then enumerate
# independent inequality subsets in the null space. Check original constraints.
function reference(A, l, u, lower, upper, x)
  n = length(x)
  E, B = zeros(0, n), zeros(0, n)
  e, d = Float64[], Float64[]
  for i in 1:(size(A, 1) + n)
    a = i <= size(A, 1) ? Vector(A[i, :]) : Float64.(1:n .== i - size(A, 1))
    lo, hi = i <= size(A, 1) ? (l[i], u[i]) : (lower[i - size(A, 1)], upper[i - size(A, 1)])
    if lo == hi
      E, e = [E; a'], [e; lo]
    else
      if isfinite(hi)
        B, d = [B; a'], [d; hi]
      end
      if isfinite(lo)
        B, d = [B; -a'], [d; -lo]
      end
    end
  end
  if isempty(e)
    z0, N = copy(x), Matrix(1.0I, n, n)
  else
    F = svd(E; full = true)
    rank = count(>(max(size(E)...) * eps() * maximum(F.S)), F.S)
    z0 = x - pinv(E) * (E * x - e)
    N = F.V[:, (rank + 1):n]
  end
  G, h = B * N, d - B * z0
  best, objective = nothing, Inf
  p, q = size(G)
  @assert p <= 8 && n <= 3
  for mask in 0:((1 << p) - 1)
    count_ones(mask) <= q || continue
    active = [i for i in 1:p if !iszero(mask & (1 << (i - 1)))]
    if isempty(active)
      v, multipliers = zeros(q), Float64[]
    else
      C = G[active, :]
      F = svd(C)
      minimum(F.S) > 100 * eps() * maximum(F.S) || continue
      v = pinv(C) * h[active]
      multipliers = -pinv(C') * v
    end
    z = z0 + N * v
    all(B * z .<= d .+ 2e-11 * max(1, norm(z), norm(d))) || continue
    norm(E * z - e) <= 2e-11 * max(1, norm(e), norm(z)) || continue
    all(multipliers .>= -2e-11 * max(1, norm(x))) || continue
    norm(v + G[active, :]' * multipliers) <= 2e-11 * max(1, norm(v)) || continue
    value = sum(abs2, z - x)
    if value < objective
      best, objective = z, value
    end
  end
  best === nothing && error("enumeration did not find a feasible KKT point")
  best
end

# Convert scaled working multipliers back to the user's row and bound units.
function multipliers(f)
  T = eltype(f.point)
  λ, μ = zeros(T, f.m), zeros(T, f.n)
  for j in 1:f.count
    row = PB._poly_row(f, f.active[j])
    i = abs(row)
    v = row > 0 ? f.multipliers[j] : -f.multipliers[j]
    if i <= f.m
      λ[i] += ldexp(v, f.scale[i])
    else
      μ[i - f.m] += v
    end
  end
  λ, μ
end

function check_kkt(f, A, l, u, lower, upper, x, z; tol = 1e-9)
  λ, μ = multipliers(f)
  az = A * z
  for (v, lo, hi, w, scale) in
      ((az, l, u, λ, max(1, norm(A) * norm(z))), (z, lower, upper, μ, max(1, norm(z))))
    for i in eachindex(v)
      @test v[i] >= lo[i] - tol * max(scale, isfinite(lo[i]) ? abs(lo[i]) : 0)
      @test v[i] <= hi[i] + tol * max(scale, isfinite(hi[i]) ? abs(hi[i]) : 0)
      if lo[i] != hi[i]
        if w[i] > 0
          @test isfinite(hi[i])
          @test abs(w[i] * (v[i] - hi[i])) <= tol * max(1, abs(w[i]) * scale)
        elseif w[i] < 0
          @test isfinite(lo[i])
          @test abs(w[i] * (v[i] - lo[i])) <= tol * max(1, abs(w[i]) * scale)
        end
      end
    end
  end
  @test norm(x - z - A' * λ - μ) <= tol * max(1, norm(x), norm(z), norm(A' * λ + μ))
  @test all(j -> abs(f.active[j]) <= f.equalities || f.multipliers[j] >= 0, 1:f.count)
end

@testset "IndPolyhedron: enumeration reference" begin
  rng = Xoshiro(172)
  for trial in 1:200
    n, m = rand(rng, 1:3), rand(rng, 0:2)
    A = randn(rng, m, n)
    if m == 2 && trial % 4 == 0
      A[2, :] = (trial % 8 == 0 ? -1 : 1) * A[1, :]
    end
    planted = randn(rng, n)
    l, u = fill(-Inf, m), A * planted + rand(rng, m)
    if m > 0 && trial % 3 == 0
      l[1] = u[1] = dot(A[1, :], planted)
    end
    lower, upper = planted - rand(rng, n), planted + rand(rng, n)
    if trial % 7 == 0
      lower[1] = upper[1] = planted[1]
    end
    x = 3randn(rng, n)
    f = IndPolyhedron(iseven(trial) ? sparse(A) : A, l, u, lower, upper)
    z = prox(f, x)
    @test z ≈ reference(A, l, u, lower, upper, x) rtol = 1e-10 atol = 1e-10
    check_kkt(f, A, l, u, lower, upper, x, z)
  end
end

@testset "Special cases and constraints" begin
  rng = Xoshiro(173)
  x = randn(rng, 7)
  for A in (zeros(0, 7), spzeros(0, 7), zeros(3, 7), spzeros(3, 7))
    f = IndPolyhedron(A, -Inf, Inf, -0.2, 0.6)
    @test prox(f, x) ≈ prox(IndBox(-0.2, 0.6), x) atol = 1e-10
    check_traits(f; proximable = true, convex = true)
  end
  a, b = randn(rng, 7), 0.3
  for A in (reshape(a, 1, :), sparse(reshape(a, 1, :)))
    @test prox(IndPolyhedron(A, -Inf, b), x) ≈ prox(IndHalfspace(a, b), x) atol = 1e-10
    @test prox(IndPolyhedron(A, b, b), x) ≈ prox(IndHyperplane(a, b), x) atol = 1e-10
  end
  A = randn(rng, 3, 7)
  b = randn(rng, 3)
  @test prox(IndPolyhedron(A, b, b), x) ≈ prox(IndAffine(A, b), x) atol = 1e-10
  A, b = [A; A[1:1, :]], [b; b[1]]
  @test prox(IndPolyhedron(A, b, b), x) ≈ prox(IndAffine(A, b), x) atol = 1e-10
  chain = zeros(6, 7)
  for i in 1:6
    chain[i, i], chain[i, i + 1] = 1, -1
  end
  @test prox(IndPolyhedron(sparse(chain), -Inf, 0), x) ≈ prox(IndIsotonic(), x) atol = 1e-10
  @test prox(IndPolyhedron(ones(1, 7), 1, 1, 0, Inf), x) ≈ prox(IndSimplex(1), x) atol = 1e-10
  # A branching order pools incomparable children only when constraints require it.
  A = [1 -1 0; 1 0 -1]
  @test prox(IndPolyhedron(A, -Inf, 0), [3., 0, 1]) ≈ fill(4 / 3, 3) atol = 1e-10
  # Parallel nested rows force a dual-only drop for one of the row orders.
  for A in ([1. 0; 2 0; -1 0], [2. 0; -1 0; 1 0])
    @test prox(IndPolyhedron(A, [-Inf, -Inf, -Inf], [0., 0, 1]), [3., 2]) ≈ [0., 2] atol = 1e-10
  end
  # Both coordinate sides enter first. The third row is dependent and drops
  # one side in a dual-only step, including a Givens deletion in the second case.
  for (row, expected) in (([-1., 1], [0., -1]), ([1., -1], [-1., 0]))
    f = IndPolyhedron([1. 0; 0 1; row'], -Inf, [0., 0, -1])
    @test prox(f, [10., 10]) ≈ expected atol = 1e-10
    check_kkt(f, [1. 0; 0 1; row'], fill(-Inf, 3), [0., 0, -1],
      fill(-Inf, 2), fill(Inf, 2), [10., 10], expected)
  end
  A = [1. 0; 1 0; -1 0; 3 0]
  f = IndPolyhedron(A, -Inf, [2., 1, 0, 6])
  @test prox(f, [4., 3]) ≈ [1., 3] atol = 1e-10
  @test prox(f, [-4., 3]) ≈ [0., 3] atol = 1e-10
  @test prox(f, x[1:2], 0.125) ≈ prox(f, x[1:2], 7.0) atol = 1e-10
  @test prox(f, x[1:2]) + prox(Conjugate(f), x[1:2]) ≈ x[1:2] atol = 1e-10
  # Tiny independent rows must survive normalization rather than SVD truncation.
  @test prox(IndPolyhedron([1. 0; 0 1e-20], [0., 1e-20], [0., 1e-20]), zeros(2)) ≈ [0., 1] atol = 1e-10
  for δ in (1e-4, 1e-8, 1e-12, 1e-16, 1e-20)
    f = IndPolyhedron([1. 0; 1 δ], [-Inf, 1.], [0., Inf])
    @test prox(f, zeros(2)) ≈ [0., 1 / δ] rtol = 1e-10 atol = 1e-9
  end
  f = IndPolyhedron(Float32[1 0; 1 1e-8], Float32[-Inf, 1], Float32[0, Inf])
  @test prox(f, zeros(Float32, 2)) ≈ Float32[0, 1e8] rtol = 4eps(Float32) atol = 4eps(Float32)
  for k in (-900, 900)
    a = ldexp(1., k)
    @test prox(IndPolyhedron([a a / 2], a, a), [0., 0]) ≈ [0.8, 0.4] atol = 1e-10
  end
  a = nextfloat(0.)
  @test prox(IndPolyhedron(reshape([a, 0.], 1, :), a, a), zeros(2)) == [1., 0]
  # Representation changes preserve the projection.
  A, b = randn(rng, 5, 7), rand(rng, 5)
  z = prox(IndPolyhedron(A, -Inf, b), x)
  p, s = randperm(rng, 5), exp2.([40., -30, 10, -15, 5])
  @test prox(IndPolyhedron(A[p, :] .* s, -Inf, b[p] .* s), x) ≈ z atol = 1e-10
end

@testset "Planted KKT, dense and sparse" begin
  rng = Xoshiro(174)
  for T in (Float64, Float32), sparse_data in (false, true)
    n, m = 16, 24
    A, zstar = randn(rng, T, m, n), randn(rng, T, n)
    l, u = fill(T(-Inf), m), A * zstar
    l[1:3] = u[1:3]
    u[9:end] .+= one(T) .+ rand(rng, T, m - 8)
    λ = zeros(T, m)
    λ[1:3] = randn(rng, T, 3)
    λ[4:8] = one(T) .+ rand(rng, T, 5)
    lower, upper, μ = fill(T(-Inf), n), fill(T(Inf), n), zeros(T, n)
    lower[1], upper[2], μ[1], μ[2] = zstar[1], zstar[2], T(-0.7), T(0.6)
    x = zstar + A' * λ + μ
    f = IndPolyhedron(sparse_data ? sparse(A) : A, l, u, lower, upper)
    z = prox(f, x)
    tol = T === Float64 ? 1e-9 : 5e-5
    @test z ≈ zstar rtol = tol atol = tol
    check_kkt(f, A, l, u, lower, upper, x, z; tol)
  end
end

@testset "Membership, precision and empty sets" begin
  for T in (Float32, Float64)
    f = IndPolyhedron(ones(T, 1, 1), -Inf, one(T))
    @test eltype(f.point) === T
    @test f(T[1]) === zero(T)
    @test f(T[2]) === T(Inf)
    @test f(T[1 + sqrt(eps(T)) / 2]) === zero(T)
    @test prox(f, T[1 + (T === Float64 ? 1e-9 : 1e-4)]) == T[1]
    eq = IndPolyhedron(ones(T, 1, 1), zero(T), zero(T))
    @test eq(T[1]) === T(Inf)
    @test eq(T[-1]) === T(Inf)
    huge = floatmax(T)
    g = IndPolyhedron(ones(T, 1, 2), -Inf, zero(T))
    @test g(T[huge, huge]) === T(Inf)
    @test g(T[huge, -huge]) === zero(T)
    @test g(T[-huge, -huge]) === zero(T)
    for A in (zeros(T, 0, 0), spzeros(T, 0, 0))
      empty = IndPolyhedron(A, T[], T[])
      @test prox(empty, T[]) == T[]
      @test empty(T[]) === zero(T)
    end
    unbounded = IndPolyhedron(zeros(T, 0, 2), -Inf, Inf)
    @test prox(unbounded, T[huge, huge]) == T[huge, huge]
    @test_throws ArgumentError IndPolyhedron(zeros(T, 1, 0), one(T), T(Inf))
    for (A, l, u, lower, upper) in
        ((ones(T, 2, 1), T[-Inf, 1e-3], T[0, Inf], -Inf, Inf),
         (ones(T, 2, 1), T[0, 1], T[0, 1], -Inf, Inf),
         (ones(T, 1, 1), one(T), T(Inf), -Inf, zero(T)))
      empty = IndPolyhedron(A, l, u, lower, upper)
      @test_throws ArgumentError prox(empty, zeros(T, 1))
      @test !empty.valid
      @test empty.iterations < 20
    end
  end
  empty = IndPolyhedron(ones(2, 1), [-Inf, 1e-10], [0., Inf])
  @test_throws ArgumentError prox(empty, [0.])
  # Redundant normals below full dimension still yield a checked certificate.
  empty = IndPolyhedron([1. 1 0; 1 1 0], [0., 1], [0., 1])
  @test_throws ArgumentError prox(empty, zeros(3))
  f = IndPolyhedron(Float32[1 1], Float32(0), Float32(1))
  @test eltype(f.point) === Float32
  @test f([0., 0]) === 0.0
  @test f([2, 2]) === Inf
  # Integer arguments choose a Float64 output, but the solve still uses Float32 data.
  @test prox(f, [2, 2]) ≈ [0.5, 0.5] rtol = 0 atol = 4eps(Float32)
  for x in (Float32[2, 2], [2., 2], Float32[0, 0], [2., 2])
    y = zeros(Float32, 2)
    @test value_and_prox!(y, f, x, 1) === zero(float(eltype(x)))
    @test y == Float32.(prox(f, x))
    @test eltype(f.origin) === Float32
  end
  mixed = IndPolyhedron(Float32[1 1], 0.0, 1.0)
  @test eltype(mixed.point) === Float64
  y = zeros(Float32, 2)
  @test value_and_prox!(y, mixed, [2., 2], 1) === 0.0
  @test y == Float32[0.5, 0.5]
  @test prox(IndPolyhedron([typemin(Int) 0], -Inf, 0), [-1., 2]) == [0., 2]
  @test_throws ArgumentError IndPolyhedron([NaN;;], -Inf, 1)
  @test_throws ArgumentError IndPolyhedron([Inf;;], -Inf, 1)
  @test_throws ArgumentError IndPolyhedron(ones(1, 1), NaN, 1)
  @test_throws ArgumentError IndPolyhedron(ones(1, 1), 2, 1)
  @test_throws ArgumentError IndPolyhedron(ones(1, 1), Inf, Inf)
  @test_throws ArgumentError IndPolyhedron(ones(1, 1), -Inf, -Inf)
  @test_throws ArgumentError IndPolyhedron(ones(1, 1), 0, 1; maxiter = 0)
  @test_throws DimensionMismatch IndPolyhedron(ones(2, 2), [0.], 1)
  limited = IndPolyhedron(Matrix(1.0I, 2, 2), 0, 0; maxiter = 1)
  @test_throws r"iteration limit exhausted" prox(limited, ones(2))
  @test !limited.valid
  # The point is representable, but its dual multiplier is outside Float64.
  numerical = IndPolyhedron([1. 0; 1 1e-200], [-Inf, 1.], [0., Inf])
  out = fill(37., 2)
  @test_throws r"numerical failure" prox!(out, numerical, zeros(2), 1)
  @test out == fill(37., 2)
  @test !numerical.valid
  @test prox(numerical, [0., 1e200]) == [0., 1e200]
end

@testset "Warm starts, ownership and preflight" begin
  rng = Xoshiro(175)
  A, l, u = randn(rng, 8, 5), fill(-1., 8), ones(8)
  lower, upper = fill(-2., 5), fill(2., 5)
  f = IndPolyhedron(A, l, u, lower, upper)
  x = randn(rng, 5)
  for trial in 1:25
    x .+= 0.15randn(rng, 5)
    @test prox(f, x) ≈ prox(IndPolyhedron(A, l, u, lower, upper), x) rtol = 1e-10 atol = 1e-10
  end
  z = prox(f, x)
  A .= 0
  l .= -Inf
  u .= Inf
  lower .= 10
  upper .= 20
  @test prox(f, x) ≈ z atol = 1e-10
  @test reset_cache!(f) === f
  @test !f.valid && f.count == 0
  @test prox(f, x) ≈ z atol = 1e-10
  alias = copy(x)
  @test prox!(alias, f, alias, 0.5) === alias
  @test alias ≈ z atol = 1e-10
  storage = fill(37., 2length(x))
  xv, yv = view(storage, 1:2:length(storage)), view(storage, 2:2:length(storage))
  xv .= x
  @test prox!(yv, f, xv, 1) === yv
  @test yv ≈ z atol = 1e-10
  for op in (prox!, value_and_prox!)
    for bad in (ones(4), ones(5, 1), fill(Inf, 5), fill(NaN, 5))
      out = fill(37., size(bad))
      E = all(isfinite, bad) ? DimensionMismatch : ArgumentError
      @test_throws E op(out, f, bad, 1)
      @test all(==(37.), out)
      @test !f.valid
    end
    out = fill(37., 4)
    @test_throws DimensionMismatch op(out, f, x, 1)
    @test out == fill(37., 4)
    broken = IndPolyhedron(ones(1, 3), -Inf, 1)
    fs = SeparableSum(SqrNormL2(2), broken)
    inputs, outputs = ([3.], [1., 2]), ([37.], [37., 37])
    @test_throws DimensionMismatch op(outputs, fs, inputs, 1)
    @test outputs == ([37.], [37., 37])
    @test_throws DimensionMismatch op(inputs, fs, inputs, 1)
    @test inputs == ([3.], [1., 2])
  end
  @test prox(f, x) ≈ z atol = 1e-10
  @test_throws MethodError prox!(similar(x), f, x, ones(5))
  A, b = [1. 1; 2 2; -1 -1], [1., 2, -1]
  f = IndPolyhedron(A, b, b, 0, Inf)
  for x in ([3., 2], [-2., 4], [1., -3], [0.5, 0.5])
    @test prox(f, x) ≈ prox(IndPolyhedron(A, b, b, 0, Inf), x) atol = 1e-10
  end
  A = sparse([1. 1])
  f = IndPolyhedron(A, -Inf, 1)
  z = prox(f, [2., 2])
  nonzeros(A) .= 0
  @test prox(f, [2., 2]) ≈ z atol = 1e-10
  malformed = IndPolyhedron(ones(2, 2), -Inf, 1)
  pop!(malformed.l)
  out = fill(37., 2)
  @test_throws DimensionMismatch prox!(out, malformed, ones(2), 1)
  @test out == fill(37., 2)
end

@testset "Cancellation, skipped rows and wrapper preflight" begin
  for T in (Float64, Float32), sparse_data in (false, true)
    A = ones(T, 1, 1)
    f = IndPolyhedron(sparse_data ? sparse(A) : A, -Inf, T(0.1))
    @test prox(f, T[1024]) ≈ T[0.1] rtol = 0 atol = 4eps(T)
    @test prox(f, T[1]) ≈ T[0.1] rtol = 0 atol = 4eps(T)
    @test prox(f, T[-1024]) == T[-1024]
    # An unrelated large coordinate must not mask the first row's violation.
    A = T[1 0]
    f = IndPolyhedron(sparse_data ? sparse(A) : A, -Inf, T(0.1))
    z = prox(f, T[1, 1e16])
    @test z[1] ≈ T(0.1) rtol = 0 atol = 4eps(T)
    @test z[2] == T(1e16)
    A = T[1 0; 1 0]
    empty = IndPolyhedron(sparse_data ? sparse(A) : A, T[0, 1], T[0, 1], T[-Inf, 0], T[Inf, 0])
    for x in (T[0, 1e16], T[0, 0])
      out = fill(T(37), 2)
      err = try
        prox!(out, empty, x, one(T))
      catch err
        err
      end
      @test err isa ArgumentError
      @test occursin("empty", sprint(showerror, err))
      @test out == fill(T(37), 2)
      @test !empty.valid
    end
    # This independent equality is satisfied, but below the rank threshold,
    # on the first call. The warm call must add it when the input changes.
    δ = T === Float64 ? T(1e-16) : T(1e-8)
    A = T[1 0; 1 δ]
    f = IndPolyhedron(sparse_data ? sparse(A) : A, T[0, 1000], T[0, 1000])
    expected = T[0, 1000 / δ]
    @test prox(f, expected) ≈ expected rtol = 8eps(T) atol = 8eps(T)
    @test !f.used[2]
    @test prox(f, zeros(T, 2)) ≈ expected rtol = 8eps(T) atol = 8eps(T)
  end
  for T in (Float64, Float32), op in (prox!, value_and_prox!)
    f = IndPolyhedron(ones(T, 1, 1), -Inf, zero(T))
    for (wrapper, x, expected) in
        ((Translate(f, T[0]), T[1], T[0]), (Translate(f, T[2]), T[3], T[2]),
         (Tilt(f, T[0]), T[1], T[0]), (Tilt(f, T[2]), T[3], T[0]))
      out = T[NaN]
      value = op(out, wrapper, x, one(T))
      @test out == expected
      op === value_and_prox! && @test value == zero(T)
    end
  end
end

@testset "Finite solver tolerances near floatmax" begin
  for T in (Float64, Float32), sparse_data in (false, true)
    A = ones(T, 1, 2)
    f = IndPolyhedron(sparse_data ? sparse(A) : A, -Inf, zero(T))
    a = T === Float64 ? T(1.2e308) : T(2.4e38)
    x = T[a, -a / 2]
    z = prox(f, x)
    @test all(isfinite, z)
    @test z ./ a ≈ T[0.75, -0.75] rtol = 8eps(T) atol = 0
    @test f(z) == 0
    @test prox(f, x) ./ a ≈ T[0.75, -0.75] rtol = 8eps(T) atol = 0
  end
end

# The sequence forces warm-start drops as well as additions inside a call.
function sequence!(op, y, f, points, γ)
  for x in points
    op(y, f, x, γ)
  end
  nothing
end

@testset "Zero allocations" begin
  for T in (Float32, Float64)
    A = T[1 1 0; 0 1 1; 1 0 -1]
    f = IndPolyhedron(A, -one(T), one(T), -one(T), one(T))
    points = [T[3, 2, -1], T[-2, 0, 4], T[0, 0, 0], T[1, -3, 2]]
    y = similar(points[1])
    @test allocs(f, points[1]) == 0
    for op in (prox!, value_and_prox!)
      @test allocs(sequence!, op, y, f, points, one(T)) == 0
    end
  end
end

end
