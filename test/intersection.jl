module IntersectionTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(4)

# the projection onto {l ≤ x ≤ u, ⟨a, x⟩ = b}: x(ν) = clamp(v - νa, l, u), with
# ⟨a, x(ν)⟩ nonincreasing in ν, so the multiplier ν is found by bisection
function box_hyperplane_proj(v, a, b, l, u)
  φ(ν) = dot(a, clamp.(v .- ν .* a, l, u)) - b
  lo, hi = -1.0, 1.0
  while φ(lo) < 0; lo *= 2; end
  while φ(hi) > 0; hi *= 2; end
  for _ in 1:200
    mid = (lo + hi) / 2
    φ(mid) > 0 ? (lo = mid) : (hi = mid)
  end
  clamp.(v .- (lo + hi) / 2 .* a, l, u)
end

# Dykstra's algorithm as in its statement, with fresh arrays, for a fixed number
# of rounds
function dykstra_ref(sets, v; rounds = 20_000)
  y = copy(v)
  P = [zero(v) for _ in sets]
  for _ in 1:rounds, (k, C) in enumerate(sets)
    w = y + P[k]
    y = prox(C, w)
    P[k] = w - y
  end
  y
end

# the constraints of a correlation matrix: PSD, and a unit diagonal as a box
function unit_diagonal(p)
  L, U = fill(-Inf, p, p), fill(Inf, p, p)
  L[diagind(L)] .= 1
  U[diagind(U)] .= 1
  IndBox(L, U)
end
# a band, as a box: zero outside the band, free inside
function band_box(p, k)
  B = [abs(i - j) <= k ? Inf : 0.0 for i in 1:p, j in 1:p]
  IndBox(-B, B)
end
function random_correlation(p)
  B = randn(3p, p)
  C = B'B
  d = sqrt.(diag(C))
  Symmetric(C ./ (d * d'))
end

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)

@testset "IndIntersection" begin
  @testset "a box and a hyperplane" begin
    n = 30
    a, b = randn(n), 0.5
    l, u = -ones(n), ones(n)
    f = IndIntersection(IndBox(l, u), IndHyperplane(a, b))
    check_traits(f; proximable = true, convex = true)
    for _ in 1:5
      v = 2 .* randn(n)
      z = prox(f, v)
      @test z ≈ box_hyperplane_proj(v, a, b, l, u) atol = 1e-7
      @test f(z) == 0
      @test f.converged && f.iterations > 1
    end
    v = 2 .* randn(n)
    check_prox(f, v, 1.; rtol = 1e-6, atol = 1e-8)
    @test value_and_prox!(similar(v), f, v, 1.) == 0
    @test f(v) == Inf
    # a point in the intersection is its own projection
    z = prox(f, v)
    @test prox(f, z) ≈ z atol = 1e-9
  end

  @testset "the nearest correlation matrix" begin
    p = 8
    R = random_correlation(p)
    E = 0.3 * randn(p, p)
    R̃ = Matrix(R + Symmetric(E + E') / 2)
    R̃[diagind(R̃)] .= 1
    D = unit_diagonal(p)
    f = IndIntersection(IndPSD(), D)
    C = prox(f, R̃)
    @test C ≈ dykstra_ref((IndPSD(), D), R̃) atol = 1e-7
    @test all(diag(C) .== 1)                        # exactly in the last set
    @test eigmin(Symmetric(C)) > -1e-8
    @test f(C) == 0 && f(R̃) == (eigmin(Symmetric(R̃)) >= -1e-8 ? 0 : Inf)
    # the projection onto a convex set: ⟨R̃ - C, Z - C⟩ ≤ 0 for correlation matrices Z
    for _ in 1:10
      Z = Matrix(random_correlation(p))
      @test dot(R̃ - C, Z - C) <= 1e-7
    end
    # a Symmetric input, a Float32 input
    @test prox(f, Symmetric(R̃)) ≈ C atol = 1e-7
    f32 = IndIntersection(IndPSD(), unit_diagonal(p))
    C32 = prox(f32, Float32.(R̃))
    @test C32 isa Matrix{Float32}
    @test C32 ≈ C atol = 1e-4
    @test f32.converged
  end

  @testset "PSD and banded matrices" begin
    p = 10
    A = randn(p, p); A = A + A'
    sets = (IndPSD(), band_box(p, 2))
    f = IndIntersection(sets...; maxiter = 100_000)          # slow: up to about 10⁴ rounds
    Z = prox(f, A)
    @test f.converged
    @test Z ≈ dykstra_ref(sets, A; rounds = 50_000) atol = 1e-7
    @test all(Z[i, j] == 0 for i in 1:p, j in 1:p if abs(i - j) > 2)
    @test eigmin(Symmetric(Z)) > -1e-7
  end

  @testset "warm starts" begin
    p = 8
    R̃ = Matrix(random_correlation(p) + Symmetric(0.3 * randn(p, p)))
    R̃[diagind(R̃)] .= 1
    cold = IndIntersection(IndPSD(), unit_diagonal(p); warmstart = false)
    warm = IndIntersection(IndPSD(), unit_diagonal(p))
    prox(cold, R̃); prox(warm, R̃)
    first_rounds = warm.iterations
    # a nearby input, as in the next iteration of a solver
    R2 = R̃ + Symmetric(1e-4 * randn(p, p))
    C_cold, C_warm = prox(cold, R2), prox(warm, R2)
    @test C_warm ≈ C_cold atol = 1e-7
    @test warm.iterations < cold.iterations
    @test cold.iterations >= first_rounds ÷ 4                # cold starts redo the work
  end

  @testset "nonconvex sets and errors" begin
    f = IndIntersection(IndNonnegative(), IndBallL0(3))
    @test !isconvex(f)
    z = prox(f, randn(20))
    @test f.converged && f(z) == 0 && count(!iszero, z) <= 3
    @test_throws ArgumentError IndIntersection()
    @test_throws ArgumentError IndIntersection(LeastSquaresLoss(randn(3), randn(3, 2)), IndNonnegative())
    @test_throws ArgumentError IndIntersection(IndNonnegative(); tol = 0)
    @test_throws ArgumentError IndIntersection(IndNonnegative(); maxiter = 0)
    # no convergence: a warning, and the iterate so far
    g = IndIntersection(IndPSD(), unit_diagonal(8); maxiter = 2)
    R̃ = Matrix(random_correlation(8) + Symmetric(0.5 * randn(8, 8)))
    @test_logs (:warn, r"did not converge") prox(g, R̃)
    @test !g.converged && g.iterations == 2
  end

  @testset "allocations" begin
    p = 8
    R̃ = Matrix(random_correlation(p) + Symmetric(0.3 * randn(p, p)))
    f = IndIntersection(IndPSD(), unit_diagonal(p))
    Y = similar(R̃)
    @test allocs(a_value, f, R̃) == 0
    @test allocs(a_prox!, Y, f, R̃, 1.) == 0
    @test allocs(a_vprox!, Y, f, R̃, 1.) == 0
    Z = copy(R̃)
    @test allocs(a_prox!, Z, f, Z, 1.) == 0                  # in place
    n = 30
    g = IndIntersection(IndBox(-1., 1.), IndHyperplane(randn(n), 0.5))
    v = randn(n)
    @test allocs(a_prox!, similar(v), g, v, 1.) == 0
    @test allocs(a_prox!, Float32.(v), g, Float32.(v), 1.) == 0
  end
end

end
