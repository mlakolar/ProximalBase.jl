module CubeNormTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random
using ..TestUtils

# Build references from the machine inputs, including their rounded parameters.
function cube_reference(x, λ, γ)
  xb = BigFloat.(Array(x))
  n = norm(xb)
  level, step = BigFloat(λ), BigFloat(γ)
  t = 2 / (1 + sqrt(1 + 4 * step * level * n))
  level * n^3 / 3, (level * n) .* xb, t .* xb
end

function check_big(got::T, reference::BigFloat) where {T<:AbstractFloat}
  expected = T(reference)
  if isinf(expected)
    @test got == expected
  elseif abs(reference) >= BigFloat(floatmin(T))
    @test isfinite(got)
    @test abs(BigFloat(got) - reference) <= 8 * BigFloat(eps(expected))
  else
    @test abs(BigFloat(got) - reference) <= BigFloat(nextfloat(zero(T)))
  end
end

function check_big_prox(x, λ, γ; output_type = eltype(x))
  f = CubeNormL2(λ)
  _, _, reference = cube_reference(x, λ, γ)
  y = similar(Array(x), output_type)
  @test prox!(y, f, x, γ) === y
  for i in eachindex(y)
    check_big(y[i], reference[i])
  end
  v = value_and_prox!(y, f, x, γ)
  @test v === f(y)
  n = norm(reference)
  r = output_type(n)
  if isfinite(r) && r >= floatmin(output_type)
    @test norm(y) ≈ r rtol = (output_type === Float32 ? 1e-5 : 1e-12)
  end
  if output_type === eltype(x)
    alias = copy(x)
    value_and_prox!(alias, f, alias, γ)
    @test alias == y
  end
end

function check_big_gradient(x, λ)
  f = CubeNormL2(λ)
  value, reference, _ = cube_reference(x, λ, one(λ))
  check_big(f(x), value)
  g = gradient(f, x)
  for i in eachindex(g)
    check_big(g[i], reference[i])
  end
  alias = copy(x)
  v = value_and_gradient!(alias, f, alias)
  check_big(v, value)
  @test alias == g
end

@testset "CubeNormL2" begin
  rng = MersenneTwister(183)

  @testset "constructors and traits" begin
    @test CubeNormL2().λ == 1
    @test CubeNormL2(0.5) isa CubeNormL2{Float64}
    @test CubeNormL2(0.5f0) isa CubeNormL2{Float32}
    for λ in (-1, -Inf, Inf, NaN)
      @test_throws ArgumentError CubeNormL2(λ)
      @test_throws ArgumentError CubeNormL2{Float64}(Float64(λ))
    end
    for T in (Float64, Float32), λ in (zero(T), T(0.7))
      f = CubeNormL2(λ)
      check_traits(f; smooth = true, proximable = true, convex = true)
      L = iszero(λ) ? zero(T) : T(Inf)
      @test lipschitz(f) === L
      @test coordinate_lipschitz(f, 2) === L
      constants = zeros(T, 2, 3)
      @test coordinate_lipschitz!(constants, f) === constants
      @test all(==(L), constants)
      @test Scaled(f, 0) isa ZeroFunction
      @test 0 * f isa ZeroFunction
    end
  end

  @testset "values, gradients and radial proximal operators" begin
    for T in (Float64, Float32), dims in ((7,), (2, 3))
      x = randn(rng, T, dims...)
      λ, γ = T(0.7), T(0.9)
      f = CubeNormL2(λ)
      rtol = T === Float32 ? 1e-5 : 1e-12
      @test f(x) ≈ λ * norm(x)^3 / 3 rtol = rtol
      g = check_gradient(f, x; rng = rng)
      @test g ≈ (λ * norm(x)) .* x rtol = rtol
      p = check_prox(f, x, γ; rtol = rtol, atol = 0, rng = rng)
      @test p ≈ (2 / (1 + sqrt(1 + 4 * γ * λ * norm(x)))) .* x rtol = rtol
      n, r = norm(x), norm(p)
      @test γ * λ * r^2 + r ≈ n rtol = rtol
      # The scalar search only locates the radial minimum; BigFloat checks
      # below establish accuracy without cancellation in the objective.
      radial(t) = λ * t^3 / 3
      reference, _ = brute_force_prox(radial, n, γ; lo = zero(T), hi = n, n = 2001)
      @test r ≈ reference rtol = (T === Float32 ? 1e-5 : 1e-7)

      alias = copy(x)
      @test gradient!(alias, f, alias) === alias
      @test alias ≈ g rtol = rtol
      alias = copy(x)
      @test value_and_gradient!(alias, f, alias) ≈ f(x) rtol = rtol
      @test alias ≈ g rtol = rtol
      alias = copy(x)
      @test value_and_prox!(alias, f, alias, γ) ≈ f(p) rtol = rtol
      @test alias ≈ p rtol = rtol
    end
  end

  @testset "zeros, empty arrays and shapes" begin
    for T in (Float64, Float32), dims in ((4,), (2, 3), (2, 1, 3), (0,), (0, 3))
      x = zeros(T, dims...)
      for λ in (zero(T), T(0.8))
        f = CubeNormL2(λ)
        g, y = similar(x), similar(x)
        @test f(x) === zero(T)
        @test value_and_gradient!(g, f, x) === zero(T)
        @test g == x
        @test value_and_prox!(y, f, x, T(0.9)) === zero(T)
        @test y == x
      end
      x = randn(rng, T, dims...)
      f = CubeNormL2(zero(T))
      @test f(x) === zero(T)
      @test gradient(f, x) == zero(x)
      @test prox(f, x) == x
      f = CubeNormL2(T(0.8))
      @test gradient(f, x) ≈ (T(0.8) * norm(x)) .* x
      @test prox(f, x) ≈ (2 / (1 + sqrt(1 + 4 * T(0.8) * norm(x)))) .* x
    end
    for λ in (0.0, 0.7), x in ([1.0, -2.0], Float64[])
      f = CubeNormL2(λ)
      for dims in ((length(x) + 1,), (1, length(x)))
        y = fill(9.0, dims...)
        before = copy(y)
        for op in (prox!, value_and_prox!)
          @test_throws DimensionMismatch op(y, f, x, 0.8)
          @test y == before
        end
        for op in (gradient!, value_and_gradient!)
          @test_throws DimensionMismatch op(y, f, x)
          @test y == before
        end
      end
    end
  end

  @testset "SparseIterate" begin
    for T in (Float64, Float32), dims in ((5,), (1, 5))
      dense = reshape(T[3, 0, -4, 0, 0], dims...)
      x = SparseIterate(Array(dense))
      x[2] = one(T)
      x[2] = zero(T)                         # both stored and unstored zeros
      @test nnz(x) == 3
      for λ in (zero(T), T(0.7))
        f = CubeNormL2(λ)
        @test f(x) ≈ f(dense)
        @test gradient(f, x) isa Array{T}
        @test gradient(f, x) ≈ gradient(f, dense)
        @test Array(prox(f, x, T(0.9))) ≈ prox(f, dense, T(0.9))
        g = similar(dense)
        @test value_and_gradient!(g, f, x) ≈ f(dense)
        @test g ≈ gradient(f, dense)
        alias = copy(x)
        @test value_and_prox!(alias, f, alias, T(0.9)) ≈ f(prox(f, dense, T(0.9)))
        @test Array(alias) ≈ prox(f, dense, T(0.9))
      end
    end
  end

  @testset "precision" begin
    for (x, T) in ((Float32[1, -2, 0.5], Float32), ([1.0, -2.0, 0.5], Float32))
      f, γ = CubeNormL2(0.7), 0.9
      y, g = similar(x, T), similar(x, T)
      @test f(x) isa float(eltype(x))
      @test value_and_prox!(y, f, x, γ) isa T
      @test value_and_gradient!(g, f, x) isa float(eltype(x))
      @test g ≈ T.((f.λ * norm(x)) .* x) rtol = 1e-5
      @test y ≈ T.((2 / (1 + sqrt(1 + 4 * γ * f.λ * norm(x)))) .* x) rtol = 1e-5
      @test eltype(prox(f, x, γ)) === float(eltype(x))
      @test eltype(gradient(f, x)) === float(eltype(x))
      v, g = value_and_gradient(f, x)
      @test v isa float(eltype(x))
      @test eltype(g) === float(eltype(x))
    end
  end

  @testset "extreme scales" begin
    setprecision(BigFloat, 256) do
      for T in (Float64, Float32)
        norms = T === Float64 ? T[1e-300, 1e-5, 1, 1e5, 1e300] : T[1e-30, 1e-5, 1, 1e5, 1e30]
        scales = T === Float64 ? T[1e-300, 1e-8, 1, 1e8, 1e300] : T[1e-30, 1e-8, 1, 1e8, 1e30]
        @testset "n = $n, γλ = $c" for n in norms, c in scales
          check_big_prox(T[n], one(T), c)
        end
        M, tiny = floatmax(T), nextfloat(zero(T))
        small, large = T === Float64 ? (T(1e-200), T(1e200)) : (T(1e-20), T(1e20))
        for (x, λ, γ) in ((T[small], large, large),
                          (T[last(norms)], last(scales), last(scales)),
                          (fill(M, 2), one(T), one(T)),
                          (T[M, M, T === Float64 ? 1e-100 : 1e-5], one(T), one(T)),
                          (T[M, -M, tiny, 0], M, M),
                          (T[large], tiny, tiny),
                          (T[tiny, 2tiny, 0], M, one(T)))
          @testset "x = $x, λ = $λ, γ = $γ" begin
            check_big_prox(x, λ, γ)
            check_big_gradient(x, λ)
          end
        end
        for (x, λ) in ((T[large], first(norms)),
                       (T[small], last(scales)),
                       (T[T === Float64 ? 2.0^400 : 2.0^50], tiny),
                       (T[T === Float64 ? 1e10 : 1e5, T === Float64 ? 1e-10 : 1e-5, 0], last(scales)))
          @testset "gradient: x = $x, λ = $λ" begin
            check_big_gradient(x, λ)
          end
        end
      end
      check_big_prox(Float32[3, -4], 1e100, 1e-100)
      check_big_prox([1e300, 1e-100, 0], 1.0, 1e300; output_type = Float32)
    end
  end

  @testset "nonfinite entries" begin
    f = CubeNormL2(0.7)
    for x in ([NaN, 1.0, 0.0], [Inf, NaN, 0.0])
      @test isnan(f(x))
      @test all(isnan, gradient(f, x))
      @test all(isnan, prox(f, x))
    end
    x = [Inf, -Inf, 1.0, 0.0]
    @test f(x) == Inf
    @test isequal(gradient(f, x), [Inf, -Inf, Inf, NaN])
    @test isequal(prox(f, x), [NaN, NaN, 0.0, 0.0])
    @test isnan(value_and_prox!(similar(x), f, x))
    f = CubeNormL2(0.0)
    @test f(x) === 0.0
    @test gradient(f, x) == zeros(4)
    @test isequal(prox(f, x), x)
  end

  @testset "Spectral and Conjugate" begin
    for T in (Float64, Float32)
      f = CubeNormL2(T(0.7))
      X = T[2 0.5 -0.3; 0.5 -1 0.4; -0.3 0.4 0.8]
      spectral = Spectral(f)
      @test spectral(X) ≈ f(X)
      @test prox(spectral, X, T(0.9)) ≈ prox(f, X, T(0.9))
      Y = similar(X)
      @test value_and_prox!(Y, spectral, X, T(0.9)) ≈ f(Y)
      y, x = zeros(T, 3), T[1, -2, 0.5]
      v = value_and_prox!(y, Conjugate(f), x, T(0.9))
      @test v ≈ 2 * norm(y)^(T(3) / 2) / (3 * sqrt(f.λ)) rtol = (T === Float32 ? 1e-5 : 1e-12)
    end
  end

  @testset "allocations" begin
    for T in (Float64, Float32)
      for x in (randn(rng, T, 16), T[floatmax(T), floatmax(T), T === Float64 ? 1e-100 : 1e-5, 0])
        f, γ = CubeNormL2(one(T)), one(T)
        y, g = similar(x), similar(x)
        @test allocs(f, x) == 0
        @test allocs(prox!, y, f, x, γ) == 0
        @test allocs(value_and_prox!, y, f, x, γ) == 0
        @test allocs(gradient!, g, f, x) == 0
        @test allocs(value_and_gradient!, g, f, x) == 0
      end
    end
  end
end

end
