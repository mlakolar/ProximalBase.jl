using Test, LinearAlgebra, Random, ProximalBase

# Constrained(f, IndSphereL2(r)) on a SymmetricSparseIterate: the sphere is the
# Frobenius one, so an off-diagonal pair takes r/√2 in each of its entries.

_sym_sphere_objective(c, y, x, γ) = dot(Matrix(y), Matrix(x)) - γ * c.f(y)

# the maximum of ⟨z, x⟩ - γ f(z) over symmetric 2×2 z = [a b; b d], a² + 2b² + d² = r²
function _sym_sphere_grid(c, X, γ, r; N = 400)
  best = -Inf
  for φ in range(0, π, length = N), ψ in range(0, 2π, length = 2N)
    a = r * sin(φ) * cos(ψ); d = r * sin(φ) * sin(ψ); b = r * cos(φ) / sqrt(2)
    best = max(best, dot([a b; b d], X) - γ * c.f([a b; b d]))
  end
  best
end

@testset "SphereConstrained{NormL1} on a SymmetricSparseIterate" begin
  @testset "2×2 against a grid" begin
    for (X, w, γ, λ0, r) in (
        ([0.3 0.9; 0.9 0.1], 1.0, 1.0, 1.0, 1.0),                     # the pair wins
        ([0.5 0.4; 0.4 0.5], 1.0, 1.0, 1.0, 1.0),                     # the diagonal wins
        ([-0.3 -0.9; -0.9 -0.1], 1.0, 1.0, 1.0, 1.0),                 # negative signs
        ([0.3 -0.9; -0.9 0.1], [1.0 2.0; 2.0 0.5], 0.7, 1.3, 2.0),    # weights, r ≠ 1
        ([0.2 0.6; 0.6 -0.4], [3.0 0.5; 0.5 1.0], 1.0, 0.5, 1.5),
        ([0.0 0.0; 0.0 0.0], [1.0 2.0; 2.0 3.0], 1.0, 1.0, 1.0),
        ([0.1 0.3; 0.3 0.2], [1.0 1.0; 1.0 1.0], 2.0, 1.0, 0.5))
      c = Constrained(NormL1(λ0, w isa Real ? nothing : w), IndSphereL2(r))
      y = SymmetricSparseIterate(2)
      v = value_and_prox!(y, c, SymmetricSparseIterate(X), γ)
      @test norm(y) ≈ r rtol = 1e-14
      @test v ≈ c.f(y) rtol = 1e-14
      @test c(y) ≈ v rtol = 1e-14
      obj = _sym_sphere_objective(c, y, X, γ)
      grid = _sym_sphere_grid(c, X, γ, r)
      @test obj ≥ grid - 1e-12
      @test obj ≤ grid + 2e-2 * (1 + abs(grid))   # the grid misses a kink by its spacing
      @test prox(c, SymmetricSparseIterate(X), γ) == y
      y2 = SymmetricSparseIterate(2)
      value_and_prox!(y2, c, X, γ)                               # a dense symmetric argument
      @test y2 == y
    end
  end

  @testset "against enumeration" begin
    rng = MersenneTwister(336)
    for p in (3, 5), trial in 1:20
      A = randn(rng, p, p); X = (A + A') / 20
      W = abs.(randn(rng, p, p)) .+ 1.0; W = (W + W') / 2
      λ0, γ, r = 0.5 + rand(rng), 0.5 + rand(rng), 0.5 + 2rand(rng)
      c = Constrained(NormL1(λ0, W), IndSphereL2(r))
      y = SymmetricSparseIterate(p)
      v = value_and_prox!(y, c, X, γ)
      @test norm(y) ≈ r rtol = 1e-13
      @test v ≈ c.f(y) rtol = 1e-13
      if all(abs.(X) .≤ γ * λ0 .* W)                              # the fallback case
        best = maximum(r * (i == j ? 1 : sqrt(2)) * (abs(X[i, j]) - γ * λ0 * W[i, j]) for j in 1:p for i in j:p)
        @test _sym_sphere_objective(c, y, X, γ) ≈ best rtol = 1e-12
      else
        @test y ≈ prox(Constrained(NormL1(λ0, W), IndSphereL2(r)), Matrix(X), γ) rtol = 1e-12
      end
    end
  end

  @testset "soft thresholding branch" begin
    c = Constrained(NormL1(0.1), IndSphereL2(1.0))
    X = [1.0 0.5; 0.5 -0.3]
    y = SymmetricSparseIterate(2)
    v = value_and_prox!(y, c, X, 1.0)
    @test norm(y) ≈ 1
    @test v ≈ c.f(y)
    @test Matrix(y) ≈ prox(c, X, 1.0)
  end

  @testset "boundaries" begin
    # equal scaled margins: the pair's √2(0.5 - 1) = -√2/2 against a diagonal at -0.7071…
    c = Constrained(NormL1(1.0), IndSphereL2(1.0))
    for d in (0.2928, 0.2929, 0.2930)
      X = [d 0.5; 0.5 0.0]
      y = SymmetricSparseIterate(2)
      v = value_and_prox!(y, c, X, 1.0)
      @test norm(y) ≈ 1
      @test v ≈ c.f(y)
      @test _sym_sphere_objective(c, y, X, 1.0) ≈ max(d - 1, sqrt(2) * (0.5 - 1)) atol = 1e-12
    end
    # infinite weights: those entries are never chosen
    W = [Inf 1.0; 1.0 Inf]
    c = Constrained(NormL1(1.0, W), IndSphereL2(1.0))
    y = SymmetricSparseIterate(2)
    v = value_and_prox!(y, c, [0.1 0.1; 0.1 0.1], 1.0)
    @test y[1, 2] == y[2, 1] ≈ 1 / sqrt(2)
    @test v ≈ c.f(y)
    W = fill(Inf, 2, 2)
    @test_throws ArgumentError prox(Constrained(NormL1(1.0, W), IndSphereL2(1.0)), SymmetricSparseIterate(zeros(2, 2)), 1.0)
  end

  @testset "comparison without rounding errors" begin
    S2(X) = SymmetricSparseIterate(X)
    obj(c, y, X, γ) = dot(Matrix(y), X) - γ * c.f(y)
    # the margins 0 (second diagonal), -1 (first diagonal) and -√2 (pair): the second diagonal
    # wins, although B - 0 and B - 1 round to the same number
    B = ldexp(1.0, 60)
    X = [0.0 0.0; 0.0 B]; W = [1.0 1.0; 1.0 B]
    c = Constrained(NormL1(1.0, W), IndSphereL2(1.0))
    y = SymmetricSparseIterate(2)
    v = value_and_prox!(y, c, S2(X), 1.0)
    @test Matrix(y) == [0.0 0.0; 0.0 1.0]
    @test v == c.f(y) == B
    @test obj(c, y, X, 1.0) == 0
    # a step so small that the margins are subnormal
    W = [4.0 1.0; 1.0 1.0]
    c = Constrained(NormL1(1.0, W), IndSphereL2(1.0))
    y = SymmetricSparseIterate(2)
    v = value_and_prox!(y, c, S2(zeros(2, 2)), nextfloat(0.0))
    @test Matrix(y) == [0.0 0.0; 0.0 1.0]            # the second diagonal costs 1, the pair √2
    @test v == c.f(y) == 1
    # a Float32 argument and a Float64 destination: the thresholds are Float64
    c = Constrained(NormL1(2.0^160, W), IndSphereL2(1.0))
    y = SymmetricSparseIterate(2)
    v = value_and_prox!(y, c, S2(zeros(Float32, 2, 2)), 1.0)
    @test Matrix(y) == [0.0 0.0; 0.0 1.0]
    @test v == c.f(y) == 2.0^160
    # margins that differ by one ulp
    c = Constrained(NormL1(1.0, [1.0 2.0; 2.0 1.0]), IndSphereL2(1.0))
    for X in ([0.0 0.0; 0.0 0.0], [0.5 0.0; 0.0 nextfloat(0.5)], [nextfloat(0.5) 0.0; 0.0 0.5])
      y = SymmetricSparseIterate(2)
      value_and_prox!(y, c, S2(X), 1.0)
      @test obj(c, y, X, 1.0) ≥ max(X[1, 1] - 1, X[2, 2] - 1) - eps()
    end
  end

  @testset "Float32" begin
    c = Constrained(NormL1(1f0), IndSphereL2(1f0))
    y = SymmetricSparseIterate(Float32, 2)
    v = value_and_prox!(y, c, Float32[0.3 0.9; 0.9 0.1], 1f0)
    @test eltype(y) == Float32 && v isa Float32
    @test norm(y) ≈ 1f0 rtol = 1f-6
    @test v ≈ c.f(y) rtol = 1f-6
  end

  @testset "other outputs keep their results" begin
    c = Constrained(NormL1(1.0), IndSphereL2(1.0))
    X = [0.3 0.9; 0.9 0.1]
    @test prox(c, X, 1.0) == [0.0 0.0; 1.0 0.0]                  # a dense output: one entry, as before
    @test Vector(prox(c, [0.3, 0.9, 0.1], 1.0)) == [0.0, 1.0, 0.0]
  end
end

@testset "SphereConstrained{GroupNormL2} on a SymmetricSparseIterate: fallbacks" begin
  S(X) = SymmetricSparseIterate(X)
  z = zeros(2, 2)
  # an ungrouped index, the first of them in the upper triangle
  c = Constrained(GroupNormL2(1.0, [[1], [4]]), IndSphereL2(1.0))
  y = SymmetricSparseIterate(2)
  v = value_and_prox!(y, c, S(z), 1.0)
  @test norm(y) ≈ 1 && v ≈ c.f(y) == 0
  # a group over every index with an off-diagonal first index
  c = Constrained(GroupNormL2(1.0, [[3, 2, 1, 4]]), IndSphereL2(1.0))
  y = SymmetricSparseIterate(2)
  v = value_and_prox!(y, c, S(z), 1.0)
  @test norm(y) ≈ 1 && v ≈ c.f(y)
  # the pair is its own group with the lowest weight, the diagonal entries are groups of their own
  c = Constrained(GroupNormL2(1.0, [[1], [4], [3, 2]], [3.0, 3.0, 1.0]), IndSphereL2(2.0))
  y = SymmetricSparseIterate(2)
  v = value_and_prox!(y, c, S(z), 1.0)
  @test y[1, 1] == 0 && y[2, 2] == 0
  @test y[1, 2] == y[2, 1] ≈ 2 / sqrt(2)
  @test v ≈ c.f(y) ≈ 2
  # a diagonal index that is not first
  c = Constrained(GroupNormL2(1.0, [[2, 3, 4, 1]]), IndSphereL2(3.0))
  y = SymmetricSparseIterate(2)
  v = value_and_prox!(y, c, S(z), 1.0)
  @test norm(y) ≈ 3 && v ≈ c.f(y)
end
