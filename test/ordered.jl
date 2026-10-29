module OrderedTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

# total-variation denoising by projected gradient on the dual,
# min ‖x - D'u‖²/2 over ‖u‖∞ ≤ λ with z = x - D'u, which converges linearly
function tv_reference(x, λ; iters = 50_000)
  n = length(x)
  n == 1 && return copy(x)
  u = zeros(n - 1)
  z = similar(x)
  for _ in 1:iters
    z .= x
    z[1:n-1] .+= u
    z[2:n] .-= u                                 # z = x - D'u, (Dz)ᵢ = zᵢ₊₁ - zᵢ
    u .= clamp.(u .+ (z[2:n] .- z[1:n-1]) ./ 4, -λ, λ)
  end
  z .= x
  z[1:n-1] .+= u
  z[2:n] .-= u
  z
end

# the proximal operator of SLOPE from an explicit sort and a plain stack of
# blocks, pooled while their means increase, then clipped at zero
function slope_reference(x, λ, γ)
  p = sortperm(abs.(x); rev = true)
  s, c = Float64[], Int[]
  for v in abs.(x[p]) .- γ .* λ
    push!(s, v); push!(c, 1)
    while length(s) > 1 && s[end-1] / c[end-1] < s[end] / c[end]
      s[end-1] += s[end]; c[end-1] += c[end]; pop!(s); pop!(c)
    end
  end
  z = similar(x)
  z[p] = sign.(x[p]) .* max.(reduce(vcat, [fill(s[b] / c[b], c[b]) for b in eachindex(s)]), 0)
  z
end

# isotonic regression by the min-max formula, zᵢ = max_{j ≤ i} min_{k ≥ i} mean(x[j:k])
isotonic_reference(x) = [maximum(minimum(sum(x[j:k]) / (k - j + 1) for k in i:length(x)) for j in 1:i) for i in eachindex(x)]

@testset "total variation" begin
  for n in (1, 2, 5, 12), λ in (0.0, 0.2, 1.0, 5.0)
    x = cumsum(randn(n)) .+ 0.3 .* randn(n)
    @test prox(TotalVariation1D(λ), x) ≈ tv_reference(x, λ) atol = 1e-9
  end
  x = cumsum(randn(30))
  for γ in (0.3, 2.0)
    check_prox(TotalVariation1D(0.4), x, γ)
    check_prox(FusedLasso(0.2, 0.4), x, γ)
  end
  @test prox(TotalVariation1D(1e3), x) ≈ fill(sum(x) / 30, 30)             # λ large: the mean
  @test prox(TotalVariation1D(0.7), fill(2.0, 9)) ≈ fill(2.0, 9)
  @test prox(TotalVariation1D(0.0), x) == x
  @test TotalVariation1D(0.5)([1.0, 3.0, 2.0]) ≈ 0.5 * 3
  # the fused lasso soft-thresholds the denoised vector
  @test prox(FusedLasso(0.3, 0.5), x, 0.8) ≈ shrink.(prox(TotalVariation1D(0.5), x, 0.8), 0.24)
  @test FusedLasso(0.3, 0.5)([1.0, -3.0]) ≈ 0.3 * 4 + 0.5 * 4
  z = copy(x)
  @test prox!(z, TotalVariation1D(0.4), z, 1.0) == prox(TotalVariation1D(0.4), x)
  @test eltype(prox(TotalVariation1D(0.4), Float32.(x))) == Float32
  @test prox(TotalVariation1D(0.4), Float32.(x)) ≈ prox(TotalVariation1D(0.4), x) atol = 1e-4
  check_traits(TotalVariation1D(1.0); proximable = true, convex = true)
  check_traits(FusedLasso(1.0, 1.0); proximable = true, convex = true)
  @test_throws ArgumentError TotalVariation1D(-1.0)
end

@testset "total variation values" begin
  for T in (Float32, Float64)
    n = 100_000
    x = randn(T, n)
    tv = zero(T)
    for i in 1:n-1
      tv += abs(x[i + 1] - x[i])
    end
    @test TotalVariation1D(one(T))(x) ≈ tv rtol = n * eps(T)
    f = FusedLasso(T(0.2), T(0.4))
    @test f(x) ≈ T(0.2) * sum(abs, x) + T(0.4) * tv rtol = n * eps(T)
    for z in (T[], T[2])
      @test TotalVariation1D(one(T))(z) === zero(T)
      @test f(z) == T(0.2) * sum(abs, z; init = zero(T))
    end
  end
end

@testset "isotonic regression" begin
  for n in (1, 2, 7, 15)
    x = randn(n) .+ 0.2 .* (1:n)
    z = prox(IndIsotonic(), x)
    @test z ≈ isotonic_reference(x)
    @test prox(IndIsotonic(decreasing = true), x) ≈ -isotonic_reference(-x)
    @test IndIsotonic()(z) == 0
  end
  x = randn(20)
  check_prox(IndIsotonic(), x, 1.0)
  check_prox(IndIsotonic(decreasing = true), x, 1.0)
  @test IndIsotonic()([1.0, 2.0, 2.0]) == 0 && IndIsotonic()([1.0, 0.5]) == Inf
  @test IndIsotonic(decreasing = true)([2.0, 1.0]) == 0
  z = copy(x)
  @test prox!(z, IndIsotonic(), z, 1.0) == prox(IndIsotonic(), x)
  @test prox(IndIsotonic(), sort(x)) == sort(x)                            # already monotone
  check_traits(IndIsotonic(); proximable = true, convex = true)
end

# Weighted isotonic regression by the max-min formula in higher precision.
function weighted_isotonic_reference(x, γ)
  v = BigFloat.(x)
  w = inv.(BigFloat.(γ))
  [maximum(minimum(sum(w[j:k] .* v[j:k]) / sum(w[j:k]) for k in i:length(x)) for j in 1:i) for i in eachindex(x)]
end

function check_isotonic_kkt(x, γ, θ, σ)
  w = 1 ./ γ
  tol = 1e-10 * sum(w .* abs.(x))
  λ = 0.0
  for k in eachindex(x)
    λ += w[k] * (x[k] - θ[k])
    @test σ * λ ≥ -tol
    if k < length(x)
      @test σ * (θ[k + 1] - θ[k]) ≥ 0
    end
    if k == length(x) || θ[k + 1] != θ[k]
      @test abs(λ) ≤ tol
    end
  end
end

struct OffsetSteps{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(a::OffsetSteps) = size(a.data)
Base.axes(a::OffsetSteps) = (0:length(a.data)-1,)
Base.getindex(a::OffsetSteps, i::Int) = a.data[i + 1]

@testset "weighted isotonic regression" begin
  for decreasing in (false, true)
    f = IndIsotonic(; decreasing)
    σ = decreasing ? -1 : 1
    reference(x, γ) = σ .* weighted_isotonic_reference(σ .* x, γ)
    for n in (0, 1, 2, 7, 15)
      x = randn(n)
      γ = 10 .^ (4 .* rand(n) .- 2)
      θ = prox(f, x, γ)
      @test isapprox(θ, reference(x, γ); rtol = 1e-10)
      check_isotonic_kkt(x, γ, θ, σ)
      @test prox(f, x, 5 .* γ) ≈ θ
      @test reinterpret(UInt64, prox(f, x, fill(0.3, n))) == reinterpret(UInt64, prox(f, x, 1.0))
      z = copy(x)
      @test prox!(z, f, z, γ) === z
      @test z ≈ θ
      y = similar(x)
      @test value_and_prox!(y, f, x, γ) === 0.0
      @test y ≈ θ
    end
    # Normalized weights preserve the projection across scales.
    x = σ .* [2.0, 1.0]
    for s in (1e-200, 1.0, 1e200)
      @test prox(f, x, fill(s, 2)) == σ .* [1.5, 1.5]
      @test isapprox(prox(f, x, [s, 2s]), reference(x, [s, 2s]); rtol = 1e-10)
    end
    for x in (σ .* [1.0, 2.0, 3.0], fill(1.3, 3), σ .* [1.0, 1.0, 2.0])
      @test prox(f, x, [1/3, 2.0, 0.7]) == x
    end
    x = σ .* [3.0, 1.0, 1.0, 2.0]
    γ = [0.3, 2.0, 0.5, 1.0]
    θ = prox(f, x, γ)
    @test isapprox(θ, reference(x, γ); rtol = 1e-10)
    x2 = repeat(x; inner = 2)
    @test prox(f, view(x2, 1:2:length(x2)), view(γ, :)) ≈ θ
    @test prox(f, x2[1:2:end], γ) ≈ θ
    y2 = similar(x2)
    @test prox!(view(y2, 1:2:length(y2)), f, view(x2, 1:2:length(x2)), γ) ≈ θ
    @test prox(f, x, reshape(γ, 2, 2)) ≈ θ
    z32 = prox(f, Float32.(x), γ)
    @test eltype(z32) == Float32
    @test isapprox(z32, θ; rtol = 1e-5)
    @test isapprox(prox(f, x, Float32.(γ)), θ; rtol = 1e-5)
    @test prox(f, σ .* [2.0, 0.0], [1, 3]) ≈ σ .* [1.5, 1.5]

    # Wrappers pass the diagonal metric to the underlying projection.
    b, c = randn(4), randn(4)
    @test prox(Scaled(f, 2.0), x, γ) ≈ θ
    @test prox(Translate(f, b), x, γ) ≈ b .+ prox(f, x .- b, γ)
    @test prox(Tilt(f, c), x, γ) ≈ prox(f, x .- γ .* c, γ)
    y = prox(Conjugate(f), x, γ)
    @test y ≈ x .- γ .* prox(f, x ./ γ, 1 ./ γ)
    # The polar cone has zero sum and signed nonpositive tail sums.
    tol = 1e-10 * sum(abs, x)
    @test abs(sum(y)) ≤ tol
    for k in eachindex(y)
      @test σ * sum(y[k:end]) ≤ tol
    end
    r = (x .- y) ./ γ
    @test all(σ .* diff(r) .≥ -tol)
    @test abs(dot(r, y)) ≤ tol
  end
  @test prox(IndIsotonic(), [2.0, 0.0], [1.0, 3.0]) ≈ [1.5, 1.5]
  check_isotonic_kkt([2.0, 0.0], [1.0, 3.0], [1.5, 1.5], 1)
  check_isotonic_kkt([0.0, 2.0], [3.0, 1.0], [1.5, 1.5], -1)
  @test prox(Conjugate(IndIsotonic()), [1.0, -1.0], ones(2)) ≈ [1.0, -1.0]
  @test prox(Conjugate(IndIsotonic()), [-1.0, 1.0], ones(2)) ≈ zeros(2)
  @test prox(IndIsotonic(), [2.0, 0.0], [1e10, 1e-10]) ≈ Float64.(weighted_isotonic_reference([2.0, 0.0], [1e10, 1e-10])) atol = 1e-15

  f = IndIsotonic()
  x = [2.0, 0.0]
  y = fill(7.0, 2)
  @test_throws DimensionMismatch prox!(y, f, x, ones(3))
  @test_throws DimensionMismatch prox!(zeros(3), f, x, ones(2))
  @test_throws MethodError prox!(zeros(2, 1), f, reshape(x, 2, 1), ones(2))
  @test_throws ArgumentError prox!(y, f, x, OffsetSteps(ones(2)))
  @test_throws ArgumentError prox!(OffsetSteps(copy(y)), f, x, ones(2))
  @test_throws ArgumentError prox!(y, f, OffsetSteps(copy(x)), ones(2))
  for bad in (0.0, -1.0, Inf, NaN), γ in ([bad, 1.0], [1.0, bad])
    @test_throws ArgumentError("the steps of IndIsotonic must be positive and finite") prox!(y, f, x, γ)
    @test y == fill(7.0, 2)
    z = copy(x)
    @test_throws ArgumentError("the steps of IndIsotonic must be positive and finite") prox!(z, f, z, γ)
    @test z == x
  end
  @test_throws ArgumentError("the ratio of the largest to the smallest step of IndIsotonic is too large to represent") prox!(y, f, x, [floatmax(Float64), 1e-300])
  @test y == fill(7.0, 2)
end

@testset "SLOPE" begin
  n = 10
  x = 2 .* randn(n)
  λ = sort(rand(n); rev = true)
  f = SLOPE(λ)
  for γ in (0.4, 1.5)
    check_prox(f, x, γ)
  end
  @test f(x) ≈ dot(λ, sort(abs.(x); rev = true))
  # short and long vectors take different sorts; ties
  for m in (50, 5000)
    xm = randn(m)
    xm[1:3:end] .= round.(xm[1:3:end], digits = 1)
    λm = sort(rand(m); rev = true)
    @test prox(SLOPE(λm), xm, 0.7) ≈ slope_reference(xm, λm, 0.7)
    @test SLOPE(λm)(xm) ≈ dot(λm, sort(abs.(xm); rev = true))
  end
  # constant weights give the ℓ1 norm, and one weight the ℓ∞ norm
  @test prox(SLOPE(fill(0.7, n)), x, 0.6) ≈ prox(NormL1(0.7), x, 0.6)
  @test prox(SLOPE([0.7; zeros(n - 1)]), x, 0.6) ≈ prox(NormLinf(0.7), x, 0.6)
  @test SLOPE([0.7; zeros(n - 1)])(x) ≈ NormLinf(0.7)(x)
  # equal magnitudes are pooled into one value
  z = prox(SLOPE([3.0, 1.0, 0.0]), [2.0, -2.1, 0.5], 1.0)
  @test abs(z[1]) ≈ abs(z[2]) && sign.(z) == [1.0, -1.0, 1.0]
  w = similar(x)
  @test value_and_prox!(w, f, x, 0.5) ≈ f(w)
  z = copy(x)
  @test prox!(z, f, z, 0.5) == prox(f, x, 0.5)
  @test eltype(prox(f, Float32.(x))) == Float32
  @test prox(f, Float32.(x)) ≈ prox(f, x) atol = 1e-5
  check_traits(f; proximable = true, convex = true)
  @test_throws ArgumentError SLOPE([0.1, 0.5])
  @test_throws ArgumentError SLOPE([0.5, -0.1])
  @test_throws DimensionMismatch prox(f, randn(n + 1))
end

end
