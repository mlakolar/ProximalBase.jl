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
