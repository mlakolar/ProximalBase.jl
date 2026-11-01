module HyperslabTest
# CI scenario (do not merge): a comment in a test file outside the quick tier.

using Test
using ProximalBase
using LinearAlgebra
using Random
using ..TestUtils

const PB = ProximalBase
Random.seed!(173)

normal_entries(a::Real, x) = fill(a, size(x))
normal_entries(a::AbstractArray, x) = a
steps(γ::Real, x) = fill(γ, size(x))
steps(γ::AbstractArray, x) = γ

function projection_ref(a, l, u, x, γ)
  A, X = BigFloat.(normal_entries(a, x)), BigFloat.(x)
  G = BigFloat.(steps(γ, x))
  s = dot(A, X)
  b = clamp(s, BigFloat(l), BigFloat(u))
  s == b && return X
  X .- ((s - b) / sum(G .* A.^2)) .* G .* A
end

@testset "projection precision and translated group steps" begin
  g = GroupNormL2(1.0, [1:2, 3:4], ones(2))
  γ = [1.0, 2]
  for b in (zeros(4), [0.5, -0.25, 0.75, 1.0])
    f = Translate(g, b)
    x = b .+ [3.0, 4, 0, 5]
    ref = b .+ [2.4, 3.2, 0, 3]
    @test prox(f, x, γ) ≈ ref rtol = 1e-12
    y = similar(x)
    @test prox!(y, f, x, γ) === y
    @test y ≈ ref rtol = 1e-12
    @test value_and_prox!(y, f, x, γ) ≈ 7 rtol = 1e-12
    @test y ≈ ref rtol = 1e-12
    y .= x
    @test prox!(y, f, y, γ) === y
    @test y ≈ ref rtol = 1e-12
    y .= x
    @test value_and_prox!(y, f, y, γ) ≈ 7 rtol = 1e-12
    @test y ≈ ref rtol = 1e-12
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
  end
  for op in (prox!, value_and_prox!)
    y = fill(19.0, 4)
    @test_throws DimensionMismatch op(y, Tilt(g, ones(4)), [3.0, 4, 0, 5], γ)
    @test y == fill(19.0, 4)
  end

  @test prox(IndHyperslab(2, 0, 2), [3.0], [typemax(Int)]) == [1.0]
  for a in (2, [2], -2, [-2]), n in (1, 2)
    A = a isa Real ? a : fill(only(a), n)
    x = fill(3.0, n)
    γ = fill(typemax(Int), n)
    l, u = first(a) > 0 ? (0, 2) : (-2, 0)
    f = IndHyperslab(A, l, u)
    ref = projection_ref(A, l, u, x, γ)
    @test prox(f, x, γ) ≈ ref rtol = 1e-12
    y = similar(x)
    @test value_and_prox!(y, f, x, γ) === 0.0
    @test y ≈ ref rtol = 1e-12
    @test allocs(prox!, y, f, x, γ) == 0
  end

  @testset "normal $c, $(typeof(a)), $(typeof(γ))" for c in (2.0^100, -2.0^100, 2.0^300), a in (c, [c]), γ in (1.0, [1.0], Float32[1], [typemax(Int)])
    f = IndHyperslab(a, 0, 0)
    x, y = Float32[2.0^-100], zeros(Float32, 1)
    @test prox(f, x, γ) == zeros(Float32, 1)
    @test value_and_prox!(y, f, x, γ) === 0.0f0
    @test y == zeros(Float32, 1)
    y .= x
    @test prox!(y, f, y, γ) === y
    @test y == zeros(Float32, 1)
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
  end

  for c in (2.0, 1e200, 1e-200), a in (c, [c]), F in (IndHyperplane, IndHalfspace), γ in (0.7, [0.5])
    f = F(a, c)
    x, y = Float32[3], zeros(Float32, 1)
    @test prox(f, x, γ) ≈ Float32[1] rtol = 1e-5
    @test value_and_prox!(y, f, x, γ) === 0.0f0
    @test y ≈ Float32[1] rtol = 1e-5
  end
end

@testset "hyperslab projection and traits" begin
  for T in (Float32, Float64), a in (T(2), T[1, -2, 0, 0.5]), dims in ((4,), (2, 2))
    A = a isa Real ? a : reshape(a, dims)
    x0 = reshape(T[0.5, -0.25, 1, 0.75], dims)
    n = normal_entries(A, x0)
    f = IndHyperslab(A, T(-1), T(1))
    check_traits(f; proximable = true, separable = false, convex = true)
    rtol = T === Float64 ? 1e-12 : 1e-5
    for x in (x0, x0 .- 4 .* n, x0 .+ 4 .* n), γ in (T(0.7), reshape(T[0.5, 1, 1.5, 2], dims))
      ref = projection_ref(A, f.l, f.u, x, γ)
      y = prox(f, x, γ)
      @test y ≈ ref rtol = rtol
      @test eltype(y) === T
      @test f(y) === zero(T)
      z = similar(x)
      @test value_and_prox!(z, f, x, γ) === zero(T)
      @test z ≈ ref rtol = rtol
      z .= x
      @test prox!(z, f, z, γ) === z
      @test z ≈ ref rtol = rtol
    end
    x = zero.(x0)
    @test prox(f, x) == x
    @test prox(f, x0, fill(T(0.7), dims)) ≈ prox(f, x0, T(0.7)) rtol = rtol
    dims == (4,) && check_prox(f, x0, T(0.7))
    a isa AbstractArray && @test f.a === A
  end

  a, x = [1.0, -2, 0, 0.5], [2.0, -1, 3, 0.7]
  for γ in (0.6, [0.4, 0.7, 1.2, 2.0]), b in (-1.0, 1.0)
    @test prox(IndHyperslab(a, b, b), x, γ) ≈ prox(IndHyperplane(a, b), x, γ) rtol = 1e-12
    @test prox(IndHyperslab(a, -Inf, b), x, γ) ≈ prox(IndHalfspace(a, b), x, γ) rtol = 1e-12
    @test prox(IndHyperslab(a, b, Inf), x, γ) ≈ prox(IndHalfspace(-a, -b), x, γ) rtol = 1e-12
    @test prox(IndHyperslab(a, -Inf, Inf), x, γ) == x
  end
  @test prox(IndHyperslab(1, 1, 2), [1 - 1e-9]) == [1.0]
  @test prox(IndHyperslab(1, 1, 2), [2 + 1e-9]) == [2.0]
  @test prox(IndHyperslab(1, -1, 1), [3, -1]) == [2.5, -1.5]
  @test prox(IndHyperslab([0, 1], 1, 2), [3.0, 0]) == [3.0, 1]
  for a in (typemax(Int), [typemax(Int), -typemax(Int)]), γ in (0.7, [0.5, 1.2])
    x = [2, 0]
    f = IndHyperslab(a, -typemax(Int), typemax(Int))
    @test prox(f, x, γ) ≈ projection_ref(a, f.l, f.u, x, γ) rtol = 1e-12
    @test f(prox(f, x, γ)) == 0
    @test allocs(prox!, zeros(2), f, x, γ) == 0
  end

  s = Spectral(IndHyperslab(1, 0, 1))
  @test prox(s, Matrix(Diagonal([3.0, 0]))) ≈ Diagonal([2.0, -1]) rtol = 1e-12
  @test s(Matrix(Diagonal([0.2, 0.3]))) == 0
  # the check before the decomposition reads no eigenvalues
  fill!(s.eig.w, NaN)
  @test prox(s, Matrix(Diagonal([3.0, 0]))) ≈ Diagonal([2.0, -1]) rtol = 1e-12
  @test_throws ArgumentError Spectral(IndHyperslab([1, 2], 0, 1))
end

@testset "hyperslab membership and exceptional scales" begin
  for T in (Float32, Float64)
    tol = sqrt(eps(T))
    f = IndHyperslab(one(T), one(T), T(2))
    @test f(T[1 - tol / 2]) === zero(T)
    @test f(T[1 - 4tol]) === T(Inf)
    @test f(T[2 + tol]) === zero(T)
    @test f(T[2 + 8tol]) === T(Inf)
    @test IndHyperslab(one(T), 0, T(1e30))(T[-1]) == Inf
    @test IndHyperslab(one(T), -T(1e30), 0)(T[1]) == Inf
  end
  @test IndHyperslab(1, 1, 3)([1, 2]) === 0.0
  @test IndHyperslab([1, 2], -3, 3)([2, 1]) === Inf
  @test IndHyperslab(1, 0, 1e300)([-1.0]) == Inf
  @test IndHyperslab(1, -1e300, 0)([1.0]) == Inf

  setprecision(BigFloat, 256) do
    for T in (Float32, Float64)
      e = round(Int, 0.6 * exponent(floatmax(T)))
      for c in (ldexp(one(T), -e), one(T), ldexp(one(T), e), nextfloat(zero(T)))
        for a in (c, c .* T[1, -0.5, 0, 0.25])
          l, u = -c, c
          f = IndHyperslab(a, l, u)
          for x in (T[2, -0.75, 0.5, 2], T[-2, 0.75, -0.5, -2]), γ in (T(0.7), T[0.5, 1, 1.5, 2])
            ref = projection_ref(a, l, u, x, γ)
            y = prox(f, x, γ)
            bound = 16length(x) * BigFloat(eps(T)) * max(norm(BigFloat.(x)), norm(ref)) + BigFloat(floatmin(T))
            @test norm(BigFloat.(y) - ref) <= bound
            @test f(y) === zero(T)
          end
        end
      end
    end
    for (T, U) in ((Float32, Float64), (Float64, Float32)), c in (U(1), U(1e-20), U(1e20))
      f = IndHyperslab(c .* U[1, -0.5, 0], -c, c)
      x = T[2, -1, 0.5]
      S = eps(T) > eps(U) ? T : U
      for γ in (T(0.7), T[0.5, 1, 2])
        y = similar(x)
        ref = projection_ref(f.a, f.l, f.u, x, γ)
        @test value_and_prox!(y, f, x, γ) === zero(T)
        bound = 16length(x) * BigFloat(eps(S)) * max(norm(BigFloat.(x)), norm(ref)) + BigFloat(floatmin(S))
        @test norm(BigFloat.(y) - ref) <= bound
        @test f(y) === zero(T)
      end
    end
    for a in (2, typemin(Int), [typemin(Int), 0, 1, -2])
      x = [1.25, -0.75, 0.5, 2.0]
      f = IndHyperslab(a, -1, 1)
      @test prox(f, x) ≈ projection_ref(a, -1, 1, x, 1) rtol = 1e-12
    end
  end
  for (a, l, u, x) in (([1e200], 1e200, 2e200, [0.0]),
                        ([1e-170], 1e-170, 2e-170, [0.0]),
                        (1e154, -1e154, 1e154, ones(1000)),
                        ([1.0, -1], -1e300, 1e300, [1e300, -1e300]))
    @test prox(IndHyperslab(a, l, u), x) ≈ projection_ref(a, l, u, x, 1) rtol = 1e-12
  end
  for (a, l, u, x) in (([1.0, -1], -1, 1, [1e308, 1e308]),
                        (1e200, -1e200, 1e200, [1e200, -1e200]),
                        (1e-200, -1, 1, [1e308, -1e308]))
    f = IndHyperslab(a, l, u)
    @test f(x) == 0
    @test f([1e308, -1e308]) == (a isa AbstractArray ? Inf : 0)
  end
end

# Enumerate each entry at a box bound, at zero, or free with either sign.
# Include the inactive candidate and both finite active bounds independently.
function brute_slab(x, γ, a, l, u, λ, bl, bu)
  n = length(x)
  obj(z) = λ * norm(z, 1) + sum((z .- x).^2 ./ (2 .* γ))
  feasible(z) = all(bl - 1e-12 .<= z .<= bu + 1e-12) && l - 1e-10 <= dot(a, z) <= u + 1e-10
  best, zbest, νbest = Inf, fill(NaN, n), NaN
  function consider(z, ν)
    if feasible(z) && obj(z) < best
      best, zbest, νbest = obj(z), copy(z), ν
    end
  end
  consider(clamp.(sign.(x) .* max.(abs.(x) .- γ .* λ, 0), bl, bu), 0.0)
  for (b, side) in ((l, -1), (u, 1))
    isfinite(b) || continue
    for states in Iterators.product(ntuple(_ -> 1:5, n)...)
      z = zeros(n)
      free = [s >= 4 for s in states]
      σ = [s == 4 ? 1.0 : s == 5 ? -1.0 : 0.0 for s in states]
      for i in 1:n
        states[i] == 1 && (z[i] = bl)
        states[i] == 2 && (z[i] = bu)
      end
      all(isfinite, z) || continue
      w = sum(γ[free] .* a[free].^2; init = 0.0)
      w > 0 || continue
      ν = (dot(a[.!free], z[.!free]) + dot(a[free], x[free] .- γ[free] .* λ .* σ[free]) - b) / w
      side * ν >= -1e-12 || continue
      z[free] .= x[free] .- γ[free] .* (λ .* σ[free] .+ ν .* a[free])
      all(sign.(z[free]) .== σ[free] .|| λ == 0) || continue
      consider(z, ν)
    end
  end
  zbest, νbest, best
end

function check_kkt(z, ν, x, γ, a, l, u, λ, bl, bu)
  @test l - 1e-10 <= dot(a, z) <= u + 1e-10
  @test all(bl - 1e-10 .<= z .<= bu + 1e-10)
  ν < -1e-10 && @test dot(a, z) ≈ l atol = 1e-10
  ν > 1e-10 && @test dot(a, z) ≈ u atol = 1e-10
  for i in eachindex(z)
    d = (z[i] - x[i]) / γ[i] + ν * a[i]
    if abs(z[i]) > 1e-10
      d += λ * sign(z[i])
      @test z[i] <= bl + 1e-10 ? d >= -1e-9 : (z[i] >= bu - 1e-10 ? d <= 1e-9 : abs(d) <= 1e-9)
    else
      @test z[i] <= bl + 1e-10 ? d + λ >= -1e-9 : (z[i] >= bu - 1e-10 ? d - λ <= 1e-9 : abs(d) <= λ + 1e-9)
    end
  end
end

@testset "constrained hyperslab against independent KKT references" begin
  rng = Xoshiro(173)
  for trial in 1:8
    x = 2 .* randn(rng, 3)
    a = trial % 2 == 0 ? ones(3) : [1.0, -0.7, 0]
    for γ in (fill(0.7, 3), 0.2 .+ rand(rng, 3)), (l, u) in ((-0.2, 0.4), (0.3, 0.3), (-Inf, 0.4), (-0.2, Inf), (-Inf, Inf))
      for (f, λ, bl, bu) in ((NormL1(0.3), 0.3, -Inf, Inf),
                              (IndBox(-0.8, 0.9), 0.0, -0.8, 0.9),
                              (Constrained(NormL1(0.3), IndBox(-0.8, 0.9)), 0.3, -0.8, 0.9))
        c = Constrained(f, IndHyperslab(a, l, u))
        ref, ν, obj = brute_slab(x, γ, a, l, u, λ, bl, bu)
        y = prox(c, x, γ)
        @test y ≈ ref rtol = 1e-8 atol = 1e-10
        @test λ * norm(y, 1) + sum((y .- x).^2 ./ (2 .* γ)) ≈ obj rtol = 1e-10 atol = 1e-12
        @test c.last[] ≈ ν atol = 1e-9
        check_kkt(y, c.last[], x, γ, a, l, u, λ, bl, bu)
        if all(==(γ[1]), γ)
          @test prox(c, x, γ[1]) ≈ ref rtol = 1e-8 atol = 1e-10
          trial == 1 && check_prox(c, x, γ[1])
        end
        p = prox(f, x, γ)
        b = clamp(dot(a, p), l, u)
        extra = b == dot(a, p) ? p : prox(Constrained(f, IndHyperplane(a, b)), x, γ)
        @test y ≈ extra rtol = 1e-8 atol = 1e-10
        check_traits(c; proximable = true, separable = false, convex = true)
      end

      λ = 0.8
      c = Constrained(SqrNormL2(λ), IndHyperslab(a, l, u))
      D = Diagonal(1 ./ γ .+ λ)
      p = D \ (x ./ γ)
      b = clamp(dot(a, p), l, u)
      if dot(a, p) == b
        ref, ν = p, 0.0
      else
        sol = [D a; a' 0] \ [x ./ γ; b]
        ref, ν = sol[1:3], sol[4]
      end
      y = prox(c, x, γ)
      @test y ≈ ref rtol = 1e-8 atol = 1e-10
      @test c.last[] ≈ ν atol = 1e-10
      @test D * y + c.last[] .* a ≈ x ./ γ rtol = 1e-12 atol = 1e-12
      @test l - 1e-10 <= dot(a, y) <= u + 1e-10
      @test c.f(y) + sum((y .- x).^2 ./ (2 .* γ)) ≈ c.f(ref) + sum((ref .- x).^2 ./ (2 .* γ)) rtol = 1e-12
      ν < 0 && @test dot(a, y) ≈ l atol = 1e-10
      ν > 0 && @test dot(a, y) ≈ u atol = 1e-10
      if trial == 1 && all(==(γ[1]), γ)
        @test prox(c, x, γ[1]) ≈ ref rtol = 1e-8 atol = 1e-10
        check_prox(c, x, γ[1])
      end
    end
  end
end

struct NoSlope{F} <: SeparableFunction
  f::F
end
PB.prox_entry(g::NoSlope, v, i, γ) = prox_entry(g.f, v, i, γ)
PB.value_entry(g::NoSlope, v, i) = value_entry(g.f, v, i)
PB.isconvex(::Type{<:NoSlope}) = true

cold!(y, c, x, γ) = (c.last[] = NaN; prox!(y, c, x, γ))

@testset "hyperslab warm starts and variable fixing" begin
  for T in (Float32, Float64), f in (NormL1(T(0.2)), SqrNormL2(T(0.8)), IndBox(T(-1), T(1)),
                                    Constrained(NormL1(T(0.2)), IndBox(T(-1), T(1))), NoSlope(NormL1(T(0.2))))
    c = Constrained(f, IndHyperslab(one(T), T(-0.5), T(0.5)))
    for (x, γ) in ((fill(T(-2), 5), T(0.7)), (fill(T(2), 5), T(1.1)),
                   (zeros(T, 5), T[0.5, 0.7, 1, 1.5, 2]), (fill(T(-3), 5), T[0.7, 1.2, 1.4, 0.4, 2]))
      y = prox(c, x, γ)
      @test y ≈ prox(Constrained(f, c.set), x, γ) rtol = (T === Float64 ? 1e-12 : 1e-5)
      @test sign(c.last[]) == sign(sum(x))
      @test eltype(y) === T
      z = copy(x)
      @test prox!(z, c, z, γ) === z
      @test z ≈ y rtol = (T === Float64 ? 1e-12 : 1e-5)
    end
  end
  for normal in (1.0, [1.0, -0.5, 0]), γ in (0.7, [0.5, 1, 2]), (l, u) in ((0.2, 0.2), (-Inf, 0.3), (-0.3, Inf), (-Inf, Inf))
    x = [2.0, -1, 3]
    f = NormL1(0.2)
    c = Constrained(f, IndHyperslab(normal, l, u))
    p = prox(f, x, γ)
    a = normal_entries(normal, x)
    b = clamp(dot(a, p), l, u)
    ref = b == dot(a, p) ? p : prox(Constrained(f, IndHyperplane(normal, b)), x, γ)
    @test prox(c, x, γ) ≈ ref rtol = 1e-12 atol = 1e-12
  end
  for cscale in (1e-200, 1e200), side in (-1, 1)
    c = Constrained(NormL1(0.1), IndHyperslab(cscale, -cscale, cscale))
    @test prox(c, fill(side * 2.0, 3), [0.5, 1, 2]) ≈
      prox(Constrained(NormL1(0.1), IndHyperslab(1, -1, 1)), fill(side * 2.0, 3), [0.5, 1, 2]) rtol = 1e-12
  end

  rng = Xoshiro(174)
  n = 5000
  x = 3 .* randn(rng, n)
  γv = 0.3 .+ rand(rng, n)
  # A small total puts the root in the tail, where most entries can be fixed.
  for side in (-1, 1), γ in (0.7, γv), normal in (Float64(side), fill(Float64(side), n))
    l, u = side > 0 ? (1.0, 2.0) : (-2.0, -1.0)
    c = Constrained(IndBox(0, 1), IndHyperslab(normal, l, u))
    y = prox(c, x, γ)
    ref = prox(Constrained(IndBox(0, 1), IndHyperplane(normal, side * 2.0)), x, γ)
    @test y ≈ ref rtol = 1e-10 atol = 1e-12
    @test c.live[6].buf[].compact
    @test sign(c.last[]) == side
    @test allocs(cold!, y, c, x, γ) == 0
  end
end

@testset "hyperslab errors and preflight" begin
  for a in (0, zeros(3), Float64[], [NaN], [Inf], -Inf)
    @test_throws ArgumentError IndHyperslab(a, -1, 1)
  end
  for (l, u) in ((2, 1), (NaN, 1), (0, NaN), (Inf, Inf), (-Inf, -Inf))
    @test_throws ArgumentError IndHyperslab(1, l, u)
  end
  @test_throws ArgumentError Constrained(NormL0(), IndHyperslab(1, -1, 1))
  @test_throws ArgumentError Constrained(NormL2(), IndHyperslab(1, -1, 1))

  for (l, u) in ((-1, 1), (0, 0), (-Inf, Inf))
    f = IndHyperslab(1, l, u)
    @test f(Float64[]) === 0.0
    @test isempty(prox(f, Float64[]))
    @test isempty(prox(f, Float64[], Float64[]))
    @test isempty(prox(Constrained(NormL1(0.2), f), Float64[]))
  end
  for (l, u) in ((1, 2), (-2, -1))
    f = IndHyperslab(1, l, u)
    @test f(Float64[]) == Inf
    @test_throws ArgumentError prox(f, Float64[])
    @test_throws ArgumentError prox(f, Float64[], Float64[])
    @test_throws ArgumentError prox(Constrained(NormL1(0.2), f), Float64[])
  end
  for (l, u) in ((-1, 1), (-Inf, Inf)), v in (NaN, Inf, -Inf)
    f = IndHyperslab(1, l, u)
    @test f([v]) == Inf
    for γ in (0.7, [0.5]), op in (prox!, value_and_prox!)
      y = [19.0]
      op(y, f, [v], γ)
      @test isequal(y, l == -Inf ? [v] : [NaN])
      op(y, Constrained(SqrNormL2(), f), [v], γ)
      @test isequal(y, [v])
    end
  end
  for slab in (IndHyperslab(1, 4, 5), IndHyperslab(1, -5, -4)), γ in (0.7, [0.5, 1, 2])
    @test_throws ArgumentError prox(Constrained(IndBox(0, 1), slab), zeros(3), γ)
  end
  for (l, u, expected) in ((3, 4, ones(3)), (-1, 0, zeros(3)))
    c = Constrained(IndBox(0, 1), IndHyperslab(1, l, u))
    @test prox(c, [1.5, -0.5, 0.3]) ≈ expected atol = 1e-12
  end

  x = [2.0, -1, 0.5, 1]
  badsets = (IndHyperslab(ones(3), -1, 1), IndHyperslab(ones(2, 2), -1, 1))
  badfuncs = (badsets..., (Constrained(NormL1(0.2), f) for f in badsets)...,
              Constrained(NormL1(0.2, ones(3)), IndHyperslab(1, -1, 1)),
              Constrained(SqrNormL2(0.2, ones(3)), IndHyperslab(1, -1, 1)))
  for f in badfuncs, wrap in (identity, h -> Tilt(h, ones(4)), h -> Translate(h, ones(4))),
      op in (prox!, value_and_prox!), γ in (0.7, ones(4))
    h = wrap(f)
    y = fill(19.0, 4)
    @test_throws DimensionMismatch op(y, h, x, γ)
    @test y == fill(19.0, 4)
    z = copy(x)
    @test_throws DimensionMismatch op(z, h, z, γ)
    @test z == x
    s = SeparableSum(NormL1(0.2), h)
    out = (fill(19.0, 4), fill(19.0, 4))
    @test_throws DimensionMismatch op(out, s, (x, x), 0.7)
    @test all(y -> y == fill(19.0, 4), out)
  end
  for f in (IndHyperslab(1, -1, 1), Constrained(NormL1(0.2), IndHyperslab(1, -1, 1))),
      op in (prox!, value_and_prox!)
    y = fill(19.0, 3)
    @test_throws DimensionMismatch op(y, f, x, 0.7)
    @test y == fill(19.0, 3)
    for h in (f, Tilt(f, ones(4)), Translate(f, ones(4)))
      y = fill(19.0, 4)
      @test_throws DimensionMismatch op(y, h, x, ones(3))
      # Translate delegates steps to its inner function after shifting the output.
      h isa Translate || @test y == fill(19.0, 4)
      y .= 19.0
      if h isa Translate && f isa PB.LinearConstrained
        ref = prox(h, x, ones(4))
        if op === value_and_prox!
          @test op(y, h, x, ones(2, 2)) ≈ h(ref) rtol = 1e-12
        else
          @test op(y, h, x, ones(2, 2)) === y
        end
        @test y ≈ ref rtol = 1e-12
      elseif !(h === f) || f isa IndHyperslab
        @test_throws DimensionMismatch op(y, h, x, ones(2, 2))
        h isa Translate || @test y == fill(19.0, 4)
      end
    end
  end
  @test !PB._has_conjugate_value(Constrained(IndBox(0, 1), IndHyperslab(1, 1, 2)))
  X = reshape(x, 2, 2)
  for H in (IndHyperplane([1.0, -2, 0, 0.5], 0.2), IndHalfspace([1.0, -2, 0, 0.5], 0.2))
    c = Constrained(NormL1(0.2), H)
    @test vec(prox(c, X)) ≈ prox(c, x) rtol = 1e-12
  end
end

@testset "hyperslab precision and allocations" begin
  for T in (Float32, Float64), U in (Float32, Float64)
    f = IndHyperslab(U[1, -2, 0], U(-0.5), U(0.5))
    x, y = T[2, -1, 0.5], zeros(T, 3)
    for γ in (U(0.7), U[0.5, 1, 2])
      @test f(x) isa T
      @test value_and_prox!(y, f, x, γ) isa T
      @test eltype(y) === T
      c = Constrained(NormL1(U(0.2)), f)
      @test c(x) isa T
      @test value_and_prox!(y, c, x, γ) isa T
      @test eltype(y) === T
      out = zeros(U, 3)
      @test value_and_prox!(out, f, x, γ) isa U
      @test out ≈ projection_ref(f.a, f.l, f.u, x, γ) rtol = (T === Float64 && U === Float64 ? 1e-12 : 1e-5)
      @test value_and_prox!(out, c, x, γ) isa U
    end
  end
  f = IndHyperslab(1e60, -Inf, Inf)
  x, y = Float32[1, 2], zeros(Float32, 2)
  @test prox!(y, f, x) == x
  @test prox!(y, Constrained(NormL1(0.2f0), f), x) == prox(NormL1(0.2f0), x)
  for x in ([2.0, 0], [-2.0, 0], [0.0, 0]), a in (1.0, [1.0, -0.5]), γ in (0.7, [0.5, 1.2])
    f = IndHyperslab(a, -0.5, 0.5)
    y = similar(x)
    @test allocs(f, x) == 0
    @test allocs(prox!, y, f, x, γ) == 0
    @test allocs(value_and_prox!, y, f, x, γ) == 0
    c = Constrained(NormL1(0.2), f)
    @test allocs(prox!, y, c, x, γ) == 0
    @test allocs(value_and_prox!, y, c, x, γ) == 0
    @test allocs(cold!, y, c, x, γ) == 0
  end
  for f in (IndHyperslab(1, -1, 1), IndHyperslab(1e200, -1e200, 1e200)), x in ([1e308, -1e308], [1.0, -1.0])
    @test allocs(f, x) == 0
  end
end

end
