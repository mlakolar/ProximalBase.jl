module LossesTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

const PB = ProximalBase

ρ(τ, u) = max(τ * u, (τ - 1) * u)

# the scalar terms of the three losses, written from their definitions
check_term(τ, w) = u -> w * ρ(τ, u)
hinge_term(y, w) = u -> w * max(0, 1 - y * u)
eps_term(ε, w) = u -> w * max(0, abs(u) - ε)

# the proximal operator of a convex scalar term is unique: compare with brute force
function test_scalar_prox(f, term, i)
  for v in (-3., -1.2, -0.4, -0.05, 0., 0.03, 0.3, 0.9, 1., 1.4, 3.5), γ in (0.1, 0.6, 1., 2.5)
    t, ht = brute_force_prox(term, v, γ)
    z = prox_entry(f, v, i, γ)
    @test term(z) + (z - v)^2 / (2γ) <= ht + 1e-9
    @test z ≈ t atol = 1e-5
  end
end

# the one-sided slopes at zero of t ↦ g t + fᵢ(t): kkt_violation is the larger of
# minus the two, positive when moving away from zero decreases the objective
function test_kkt(f, term, i)
  for g in (-2., -0.6, -0.1, 0., 0.2, 0.5, 1.7)
    h = 1e-7
    φ(t) = g * t + term(t)
    up, down = (φ(h) - φ(0)) / h, (φ(-h) - φ(0)) / h
    gv = zeros(i); gv[i] = g
    @test kkt_violation(f, gv, i) ≈ max(-up, -down) atol = 1e-5
  end
end

# Fenchel–Young holds with equality at p = prox_{γf}(x) and s = (x - p)/γ ∈ ∂f(p),
# which checks the value of the conjugate on its domain
function test_conjugate(f, x, γ)
  c = Conjugate(f)
  p = prox(f, x, γ)
  s = (x .- p) ./ γ
  @test f(p) + c(s) ≈ dot(p, s) rtol = 1e-10 atol = 1e-10
  check_prox(c, x, γ)
  # Moreau's decomposition
  @test prox(f, x, γ) .+ γ .* prox(c, x ./ γ, inv(γ)) ≈ x
end

@testset "CheckLoss" begin
  τ = 0.3
  x = 2 .* randn(12)
  w = rand(12) .+ 0.2
  f = CheckLoss(τ)
  @test f(x) ≈ sum(u -> ρ(τ, u), x)
  @test CheckLoss(τ, w)(x) ≈ sum(w .* ρ.(τ, x))
  @test prox(f, [2., 0.1, -0.1, -2.], 1.) ≈ [2 - τ, 0., 0., -2 + (1 - τ)]
  check_traits(f; proximable = true, separable = true, convex = true)
  @test iszeropreserving(f)
  for g in (CheckLoss(τ), CheckLoss(0.8, w), CheckLoss(0.5, w))
    wi = g.w === nothing ? 1. : g.w[3]
    test_scalar_prox(g, check_term(g.τ, wi), 3)
    test_kkt(g, check_term(g.τ, wi), 3)
    check_prox(g, x, 0.7)
    test_conjugate(g, x, 0.7)
  end
  # the conjugate is the indicator of the box [wᵢ(τ - 1), wᵢτ]
  c = Conjugate(CheckLoss(τ, w))
  @test c(w .* τ) == 0 && c(w .* (τ - 1)) == 0
  @test c(w .* τ .+ 1e-3) == Inf && c(w .* (τ - 1) .- 1e-3) == Inf
  # the loss of predictions z of y: Σ ρ_τ(yᵢ - zᵢ)
  y, z = randn(12), randn(12)
  @test Translate(CheckLoss(1 - τ), y)(z) ≈ sum(ρ.(τ, y .- z))
  r = restrict(CheckLoss(τ, w), [2, 5])
  @test r.τ == τ && r.w == w[[2, 5]]
  @test_throws ArgumentError CheckLoss(0.)
  @test_throws ArgumentError CheckLoss(1.)
  @test_throws ArgumentError CheckLoss(0.5, -w)
  @test_throws DimensionMismatch CheckLoss(0.5, w)(randn(5))
end

@testset "HingeLoss" begin
  x = 2 .* randn(12)
  yl = rand([-1., 1.], 12)
  yg = [2., -0.5, 1., -1., 3., -2., 0.7, 1., -1., 1., -4., 0.25]       # labels that are not ±1
  w = rand(12) .+ 0.2
  @test HingeLoss()(x) ≈ sum(max.(0, 1 .- x))
  @test HingeLoss(yl, w)(x) ≈ sum(w .* max.(0, 1 .- yl .* x))
  @test prox(HingeLoss(), [2., 0.9, 0.5, -1.], 0.3) ≈ [2., 1., 0.8, -0.7]
  f = HingeLoss(yl)
  check_traits(f; proximable = true, separable = true, convex = true)
  @test !iszeropreserving(f)
  for (g, i) in ((HingeLoss(), 4), (HingeLoss(yl), 4), (HingeLoss(yl, w), 4), (HingeLoss(yg, w), 4), (HingeLoss(yg, w), 1),
                 (HingeLoss(yg, w), 12))
    yi = g.y === nothing ? 1. : g.y[i]
    wi = g.w === nothing ? 1. : g.w[i]
    test_scalar_prox(g, hinge_term(yi, wi), i)
    test_kkt(g, hinge_term(yi, wi), i)
    check_prox(g, x, 0.7)
    test_conjugate(g, x, 0.7)
  end
  # a label scales the margin: the loss of yᵢxᵢ, with the step γwᵢyᵢ²
  @test prox(HingeLoss(yg), x, 0.4) ≈ [prox_entry(HingeLoss(), yg[i] * x[i], i, 0.4 * yg[i]^2) / yg[i] for i in 1:12]
  # the conjugate: Σ sᵢ/yᵢ on -wᵢ ≤ sᵢ/yᵢ ≤ 0
  c = Conjugate(HingeLoss(yg, w))
  s = -yg .* w ./ 2
  @test c(s) ≈ -sum(w) / 2
  @test c(-s) == Inf && c(-2.1 .* yg .* w) == Inf
  r = restrict(HingeLoss(yl, w), [1, 3])
  @test r.y == yl[[1, 3]] && r.w == w[[1, 3]]
  @test_throws ArgumentError HingeLoss([1., 0., -1.])
  @test_throws ArgumentError HingeLoss(yl, -w)
  @test_throws DimensionMismatch HingeLoss(yl)(randn(5))
end

@testset "EpsInsensitiveLoss" begin
  x = 2 .* randn(12)
  w = rand(12) .+ 0.2
  f = EpsInsensitiveLoss(0.4)
  @test f(x) ≈ sum(max.(0, abs.(x) .- 0.4))
  @test EpsInsensitiveLoss(0.4, w)(x) ≈ sum(w .* max.(0, abs.(x) .- 0.4))
  @test prox(f, [0.3, -0.6, 1., -2.], 0.5) ≈ [0.3, -0.4, 0.5, -1.5]
  check_traits(f; proximable = true, separable = true, convex = true)
  @test iszeropreserving(f)
  for g in (EpsInsensitiveLoss(0.4), EpsInsensitiveLoss(0.4, w), EpsInsensitiveLoss(0., w))
    wi = g.w === nothing ? 1. : g.w[2]
    test_scalar_prox(g, eps_term(g.ε, wi), 2)
    test_kkt(g, eps_term(g.ε, wi), 2)
    check_prox(g, x, 0.7)
    test_conjugate(g, x, 0.7)
  end
  # with ε = 0 it is the weighted ℓ1 norm
  @test prox(EpsInsensitiveLoss(0., w), x, 0.6) ≈ prox(NormL1(1., w), x, 0.6)
  c = Conjugate(EpsInsensitiveLoss(0.4, w))
  @test c(w ./ 2) ≈ 0.2 * sum(w)
  @test c(w .+ 1e-3) == Inf
  @test_throws ArgumentError EpsInsensitiveLoss(-0.1)
end

@testset "iterates and coordinate steps" begin
  for f in (CheckLoss(0.3), CheckLoss(0.7, rand(40)), EpsInsensitiveLoss(0.2), HingeLoss(), HingeLoss(rand([-1., 1.], 40)))
    xs = SparseIterate(sprandn(40, 0.3))
    x = Vector(xs)
    @test f(xs) ≈ f(x)
    @test Vector(prox(f, xs, 0.6)) ≈ prox(f, x, 0.6)
    zs = copy(xs)
    prox!(zs, f, zs, 0.6)
    @test Vector(zs) ≈ prox(f, x, 0.6)
    # a coordinate step: the scalar proximal operator of the coordinate
    zs = copy(xs)
    @test prox_coordinate!(zs, f, 7, 0.6) ≈ prox_entry(f, x[7], 7, 0.6)
    xa = AtomIterate(randn(40), 8)
    @test f(xa) ≈ f(Vector(xa))
  end
  # the zero-preserving losses touch only the stored entries (an entry set to zero
  # is not stored in a new iterate)
  xs = SparseIterate(sprandn(40, 0.3))
  y = SparseIterate(40)
  prox!(y, CheckLoss(0.3), xs, 0.6)
  @test 0 < nnz(y) <= nnz(xs) && all(k -> isactive(xs, k), activecoordinates(y))
  prox!(y, HingeLoss(), xs, 0.6)
  @test nnz(y) == 40                        # the hinge moves zeros
  # on a SymmetricSparseIterate: the dense symmetric matrix
  p = 5
  D = randn(p, p); D = D + D'
  for f in (CheckLoss(0.3), EpsInsensitiveLoss(0.5), HingeLoss())
    s = SymmetricSparseIterate(D)
    @test f(s) ≈ f(D)
    @test Matrix(prox(f, s, 0.5)) ≈ prox(f, D, 0.5)
  end
end

@testset "Float32" begin
  x = randn(Float32, 10)
  for f in (CheckLoss(0.3), CheckLoss(0.3f0, rand(Float32, 10)), HingeLoss(), HingeLoss(rand([-1f0, 1f0], 10)),
            EpsInsensitiveLoss(0.2))
    @test f(x) isa Float32
    @test prox(f, x, 0.3) isa Vector{Float32}
    @test @inferred(prox_entry(f, x[2], 2, 0.3f0)) isa Float32
  end
end

@testset "smooth scalar losses" begin
  losses = (PB.SquaredHinge(), PB.SmoothCheck(0.3, 0.5), PB.SmoothCheck(0.8, 0.1), PB.ConvolutionCheck(0.3, 0.5),
            PB.ConvolutionCheck(0.8, 0.2; kernel = :uniform))
  # the points where the second derivative jumps, for each loss, as functions of (η, y)
  kinks(::PB.SquaredHinge, η, y) = (y * η - 1,)
  kinks(ℓ::PB.SmoothCheck, η, y) = ((y - η) - ℓ.h * ℓ.τ, (y - η) + ℓ.h * (1 - ℓ.τ))
  kinks(ℓ::PB.ConvolutionCheck, η, y) = ((y - η) - ℓ.h, (y - η) + ℓ.h)
  for ℓ in losses, η in (-2.3, -0.7, -0.05, 0.1, 0.35, 0.8, 1.6), y in (-1., 1.)
    h = 1e-6
    @test PB.deriv(ℓ, η, y) ≈ (ℓ(η + h, y) - ℓ(η - h, y)) / 2h rtol = 1e-6 atol = 1e-8
    if all(k -> abs(k) > 1e-3, kinks(ℓ, η, y))
      @test PB.deriv2(ℓ, η, y) ≈ (PB.deriv(ℓ, η + h, y) - PB.deriv(ℓ, η - h, y)) / 2h rtol = 1e-5 atol = 1e-7
    end
    @test 0 <= PB.deriv2(ℓ, η, y) <= PB.curvature(ℓ) + 1e-12
    v, d = PB.value_and_deriv(ℓ, η, y)
    @test v ≈ ℓ(η, y) && d ≈ PB.deriv(ℓ, η, y)
  end
  # the curvature bounds are attained
  @test PB.deriv2(PB.SmoothCheck(0.3, 0.5), 0., 0.) == PB.curvature(PB.SmoothCheck(0.3, 0.5)) == 2
  @test PB.deriv2(PB.ConvolutionCheck(0.3, 0.5), 0., 0.) ≈ PB.curvature(PB.ConvolutionCheck(0.3, 0.5)) ≈ 1.5
  @test PB.curvature(PB.ConvolutionCheck(0.3, 0.5; kernel = :uniform)) == 1
  @test PB.curvature(PB.SquaredHinge()) == 2
  # SmoothCheck is the Moreau envelope of the check loss: from below, within h/2
  for τ in (0.2, 0.5, 0.9), u in (-2., -0.3, -0.01, 0., 0.04, 0.5, 3.)
    for h in (1., 0.1, 1e-3)
      e = PB.SmoothCheck(τ, h)(0., u)
      @test e <= ρ(τ, u) + 1e-15
      @test e >= ρ(τ, u) - h / 2
      # the envelope by brute force: min_s ρ_τ(s) + (s - u)²/(2h)
      _, eb = brute_force_prox(s -> ρ(τ, s), u, h)
      @test e ≈ eb atol = 1e-8
    end
  end
  # ConvolutionCheck is the convolution of ρ_τ with the kernel, by quadrature
  K(::Val{:uniform}, t) = abs(t) <= 1 ? 1 / 2 : 0.
  K(::Val{:epanechnikov}, t) = abs(t) <= 1 ? 3 * (1 - t^2) / 4 : 0.
  for kernel in (:uniform, :epanechnikov), τ in (0.25, 0.6), u in (-1.2, -0.3, 0., 0.1, 0.45, 2.)
    h = 0.5
    ℓ = PB.ConvolutionCheck(τ, h; kernel)
    n = 200_000
    ts = range(-h, h; length = n + 1)
    q = sum(ρ(τ, u - (ts[j] + ts[j+1]) / 2) * K(Val(kernel), (ts[j] + ts[j+1]) / (2h)) / h for j in 1:n) * (2h / n)
    @test ℓ(0., u) ≈ q atol = 1e-8
    abs(u) >= h && @test ℓ(0., u) ≈ ρ(τ, u)
  end
  # in Float32, and without dynamic dispatch
  for ℓ in losses
    @test ℓ(0.3f0, 1f0) isa Float32
    @test @inferred(PB.value_and_deriv(ℓ, 0.3f0, 1f0)) isa Tuple{Float32, Float32}
    @test @inferred(PB.deriv2(ℓ, 0.3f0, 1f0)) isa Float32
  end
  @test_throws ArgumentError PB.SmoothCheck(0., 1.)
  @test_throws ArgumentError PB.SmoothCheck(0.5, 0.)
  @test_throws ArgumentError PB.ConvolutionCheck(0.5, 1.; kernel = :gaussian)
  @test_throws ArgumentError PB.ConvolutionCheck(1.5, 1.)
end

@testset "GLMLoss with the smooth losses" begin
  n, p = 60, 5
  X = randn(n, p)
  β = 0.3 .* randn(p)
  yb = rand([-1., 1.], n)
  yr = X * β .+ randn(n)
  for f in (GLMLoss(PB.SquaredHinge(), yb, X), GLMLoss(PB.SmoothCheck(0.3, 0.2), yr, X),
            GLMLoss(PB.ConvolutionCheck(0.7, 0.4), yr, X; weights = rand(n)))
    @testset "$(nameof(typeof(f.loss)))" begin
      check_gradient(f, β)
      @test isfinite(lipschitz(f))
    end
  end
  # the squared hinge is what it says
  η = X * β
  @test GLMLoss(PB.SquaredHinge(), yb, X)(β) ≈ sum(max.(0, 1 .- yb .* η) .^ 2)
  @test GLMLoss(PB.SmoothCheck(0.3, 1e-8), yr, X)(β) ≈ sum(ρ.(0.3, yr .- η)) rtol = 1e-6
end

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_vgrad!(g, f, x) = value_and_gradient!(g, f, x)
a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
a_kkt(f, g, k) = kkt_violation(f, g, k)

@testset "zero allocations" begin
  n, p = 50, 20
  x = randn(p)
  xs = SparseIterate(sprandn(p, 0.3))
  w = rand(p)
  for f in (CheckLoss(0.3), CheckLoss(0.3, w), HingeLoss(), HingeLoss(rand([-1., 1.], p), w), EpsInsensitiveLoss(0.2),
            EpsInsensitiveLoss(0.2, w))
    @testset "$(nameof(typeof(f)))" begin
      for z in (x, Float32.(x), xs)
        y = similar(z)
        @test allocs(a_value, f, z) == 0
        @test allocs(a_prox!, y, f, z, 0.7) == 0
        @test allocs(a_vprox!, y, f, z, 0.7) == 0
      end
      @test allocs(a_kkt, f, x, 4) == 0
      @test allocs(a_cd!, copy(xs), f, 3, 0.7) == 0
    end
  end
  # the conjugates and the loss of predictions, on dense arrays
  for f in (Conjugate(CheckLoss(0.3, w)), Conjugate(HingeLoss(rand([-1., 1.], p))), Conjugate(EpsInsensitiveLoss(0.2)),
            Translate(CheckLoss(0.4), randn(p)))
    @testset "$(nameof(typeof(f)))" begin
      y = similar(x)
      @test allocs(a_value, f, x) == 0
      @test allocs(a_prox!, y, f, x, 0.7) == 0
      @test allocs(a_vprox!, y, f, x, 0.7) == 0
      @test allocs(a_cd!, copy(xs), f, 3, 0.7) == 0
    end
  end
  X = randn(n, p)
  for f in (GLMLoss(PB.SquaredHinge(), rand([-1., 1.], n), X), GLMLoss(PB.SmoothCheck(0.3, 0.2), randn(n), X),
            GLMLoss(PB.ConvolutionCheck(0.3, 0.2), randn(n), X))
    g = similar(x)
    @test allocs(a_value, f, x) == 0
    @test allocs(a_vgrad!, g, f, x) == 0
  end
end

end
