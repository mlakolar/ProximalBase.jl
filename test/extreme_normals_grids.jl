module ExtremeNormalsGridsTest

# The Float32 passes of "overflowing dot products, residuals and corrections" that extreme_normals.jl
# leaves to the Float64 pass: scalar normals, rescaled normals, tiny entries keep their bits, the halfspace is
# judged by the residual, iterates, the neighbourhood of the thresholds, and Float32 data projected into a
# Float64 output (narrow and wide destinations). extreme_normals.jl keeps the other Float32 blocks, array
# normals and the steps whose weighted squared norm or product overflows. The helpers are copied from it.

using Test
using ProximalBase
using LinearAlgebra

normal_entries(a::Real, x) = fill(BigFloat(a), length(x))
normal_entries(a::AbstractArray, x) = BigFloat.(a)

function projection_ref(a, b, x, γ; half = false)
  A, X = normal_entries(a, x), BigFloat.(x)
  G = γ isa Real ? ones(BigFloat, length(x)) : BigFloat.(γ)
  g = dot(A, X) - BigFloat(b)
  half && (g = max(g, zero(g)))
  X .- (g / sum(G .* A.^2)) .* G .* A
end

const OVERFLOW_PRECISION = 2400

function overflow_precision(a, b, D)
  types = (D, typeof(b), float(eltype(a)))
  reduce((p, q) -> eps(p) >= eps(q) ? p : q, types)
end

function overflow_bound(a, b, x, γ, D)
  S = overflow_precision(a, b, D)
  A, X = normal_entries(a, x), BigFloat.(x)
  G = γ isa Real ? ones(BigFloat, length(x)) : BigFloat.(γ)
  scale = sum(abs.(A .* X)) + abs(BigFloat(b))
  den = sum(G .* A .^ 2)
  [16length(x) * BigFloat(eps(S)) * (abs(X[i]) + abs(G[i] * A[i]) * scale / den) + BigFloat(floatmin(S))
   for i in eachindex(X)]
end

# y, the dense output of a projection of the dense x, against the reference, and its residual.
# Feasibility is the residual of the constraint in the scale of its terms, 32n·eps(S) (Σ|aᵢxᵢ| + |b|),
# not H(y) == 0, whose tolerance is relative to ‖y‖ and fails for a y that is small next to x.
function check_against_reference(H, x, y, γ, D = eltype(y))
  a = H.a isa AbstractArray ? vec(H.a) : H.a
  setprecision(BigFloat, OVERFLOW_PRECISION) do
    ref = projection_ref(a, H.b, x, γ; half = H isa IndHalfspace)
    @test all(abs.(ref) .<= floatmax(D))      # the arguments of the test have a projection
    @test all(isfinite, y)
    @test all(abs.(BigFloat.(vec(y)) .- ref) .<= overflow_bound(a, H.b, x, γ, D))
    A = normal_entries(a, x)
    gap = dot(A, BigFloat.(vec(y))) - BigFloat(H.b)
    scale = sum(abs.(A .* BigFloat.(x))) + abs(BigFloat(H.b))
    @test (H isa IndHalfspace ? gap : abs(gap)) <= 32length(x) * BigFloat(eps(overflow_precision(a, H.b, D))) * scale
  end
end

# prox, prox!, value_and_prox! and the aliased prox!, each against the reference.
function check_overflow(H, x, γ, D = float(eltype(x)))
  out = fill(D(NaN), size(x))
  @test prox!(out, H, x, γ) === out
  check_against_reference(H, x, out, γ, D)
  out = fill(D(NaN), size(x))
  @test value_and_prox!(out, H, x, γ) === zero(D)
  check_against_reference(H, x, out, γ, D)
  if D === eltype(x)
    y = prox(H, x, γ)
    @test y isa typeof(x)
    check_against_reference(H, x, y, γ, D)
    z = copy(x)
    @test prox!(z, H, z, γ) === z
    check_against_reference(H, x, z, γ, D)
  end
end

# A scalar step, and the given vectors of steps.
function check_overflow_steps(H, x, steps, D = float(eltype(x)))
  check_overflow(H, x, one(D), D)
  for γ in steps
    check_overflow(H, x, γ, D)
  end
end

function plain_formula(H, x, γ)
  a = H.a
  g = (a isa Real ? a * sum(x) : dot(a, x)) - H.b
  H isa IndHalfspace && (g = max(g, zero(g)))
  if γ isa Real
    t = g / (a isa Real ? H.q * length(x) : H.q)
    x .- t .* a
  else
    t = g / ProximalBase._metric_sqnorm(H, x, γ)
    x .- t .* γ .* a
  end
end

@testset "overflowing dot products, residuals and corrections" begin
  setprecision(BigFloat, OVERFLOW_PRECISION) do  # scoped: test files share worker processes
  # extreme_normals.jl loops over (Float64, Float32); this is the Float32 pass
  for T in (Float32,)
    M = floatmax(T)
    em = exponent(M)
    e4 = em ÷ 4
    up, down = ldexp(one(T), e4 + 45), ldexp(one(T), -e4 - 45)   # scale the normal out of range
    steps(n) = (T[0.5 + i / 4 for i in 1:n], T[ldexp(one(T), i % 4) for i in 1:n])
    @testset "$T" begin
      @testset "scalar normals" begin
        for F in (IndHyperplane, IndHalfspace)
          check_overflow_steps(F(T(1), 0), T[M, M], steps(2))
          check_overflow_steps(F(T(-3), 0), T[M / 4, M / 4], steps(2))
          check_overflow_steps(F(T(2), T(0.5) * M), T[M / 2, M / 4, M / 8], steps(3))
          check_overflow_steps(F(1, 0), T[M, M], steps(2))
        end
      end

      @testset "rescaled normals" begin
        for F in (IndHyperplane, IndHalfspace), c in (up, down)
          check_overflow_steps(F(c .* T[1, 1.5], 0), T[M, M], steps(2))
          check_overflow_steps(F(c .* T[1, -0.5, 0.25], 0), T[M / 2, M / 4, M / 2], steps(3))
          check_overflow_steps(F(c, 0), T[M, M], steps(2))
        end
        # The dot product and the rescaled offset b·2^300 both overflow (Inf - Inf), or only the offset does.
        for F in (IndHyperplane, IndHalfspace)
          a = T(1.5) * down .* T[1, 1]
          check_overflow_steps(F(a, T(2.5) * (M * down)), T[M, M], steps(2))
          check_overflow_steps(F(a, T(1.2) * (M * down)), T[M / 6, M / 6], steps(2))
        end
      end

      @testset "tiny entries keep their bits" begin
        # y₃ = x₃ - t a₃ with a₃ the smallest subnormal: scaling x by 2^-k would drop x₃
        for F in (IndHyperplane, IndHalfspace)
          check_overflow_steps(F(T[1, 1, nextfloat(zero(T))], 0), T[M, M, ldexp(one(T), -em ÷ 17)], steps(3))
        end
      end

      @testset "the halfspace is judged by the residual, not by an overflowed dot product" begin
        # A sequential sum of x reaches -Inf, although the sum is 0 and the residual is 2^k > 0.
        k = em - 4
        x = T[-M / 2, -M / 2, -M / 2, M / 2, M / 2, M / 2]
        check_overflow_steps(IndHalfspace(one(T), -ldexp(one(T), k)), x, steps(6))
        check_overflow_steps(IndHalfspace(ones(T, 6), -ldexp(one(T), k)), x, steps(6))
        # a normal that is scaled: the residual is 2^k in the scaled units
        check_overflow_steps(IndHalfspace(fill(down, 6), -ldexp(one(T), k - e4 - 45)), x, steps(6))
        check_overflow_steps(IndHalfspace(fill(up, 6), -ldexp(one(T), k + e4 + 45 - em)), x, steps(6))
        # the other side stays inactive, and is returned unchanged
        for H in (IndHalfspace(one(T), ldexp(one(T), k)), IndHalfspace(ones(T, 6), ldexp(one(T), k)))
          @test prox(H, x, one(T)) == x
          @test prox(H, x, steps(6)[1]) == x
        end
      end
    end
  end

  @testset "narrow and wide destinations" begin
    M32 = floatmax(Float32)
    # Float32 data whose Float32 dot product overflows, projected into a Float64 output
    for F in (IndHyperplane, IndHalfspace), γ in (1.0, [1.0, 2.0])
      check_overflow(F(Float32[1, 1], 0f0), Float32[M32, M32], γ, Float64)
      check_overflow(F(Float32[1, 1], 2f0^100), Float32[M32, M32], γ, Float64)
      check_overflow(F(Float32[1, 1.5], 0f0), Float32[M32, M32 / 2], γ, Float64)
      check_overflow(F(2f0^40 .* Float32[1, 1], 0f0), Float32[2f0^100, 2f0^99], γ, Float64)
    end
  end

  @testset "iterates" begin
    for T in (Float32,), F in (IndHyperplane, IndHalfspace)
      M = floatmax(T)
      up = ldexp(one(T), exponent(M) ÷ 4 + 45)
      # 1.8 floatmax in the sum, 0.6 floatmax in the correction
      x = T[M, 0.4M, 0.4M]
      for a in (T[1, 1, 1], T(1), up .* T[1, 1, 1]), γ in (T(1), T[1, 2, 3])
        H = F(a, 0)
        make = (SparseIterate(x), AtomIterate(x, [1:1, 2:3]))
        blank = (() -> SparseIterate(T, 3), () -> AtomIterate(T, 3, [1:1, 2:3]))
        for (xi, new) in zip(make, blank)
          y = new()
          @test prox!(y, H, xi, γ) === y
          check_against_reference(H, x, Array(y), γ)
          y = new()
          @test value_and_prox!(y, H, xi, γ) === zero(T)
          check_against_reference(H, x, Array(y), γ)
          z = deepcopy(xi)
          @test prox!(z, H, z, γ) === z
          check_against_reference(H, x, Array(z), γ)
        end
      end
      # a symmetric matrix: the off-diagonal entries count twice in ⟨a, X⟩, which is 1.8 floatmax
      X = T[1 0.4 0; 0.4 0 0; 0 0 0] .* M
      steps = T[1 2 3; 2 1 2; 3 2 1]
      for a in (ones(T, 3, 3), T(1), up .* ones(T, 3, 3)), γ in (T(1),)
        H = F(a, 0)
        y = SymmetricSparseIterate(T, 3)
        @test prox!(y, H, SymmetricSparseIterate(X), γ) === y
        check_against_reference(H, vec(X), vec(Array(y)), γ)
        y = SymmetricSparseIterate(T, 3)
        @test value_and_prox!(y, H, SymmetricSparseIterate(X), γ) === zero(T)
        check_against_reference(H, vec(X), vec(Array(y)), γ)
        z = SymmetricSparseIterate(X)
        @test prox!(z, H, z, γ) === z
        check_against_reference(H, vec(X), vec(Array(z)), γ)
        # an owner output takes a scalar step: an array of steps is refused before anything is written
        y = SymmetricSparseIterate(T, 3)
        @test_throws ArgumentError prox!(y, H, SymmetricSparseIterate(X), steps)
        @test_throws ArgumentError value_and_prox!(y, H, SymmetricSparseIterate(X), steps)
        @test nactive(y) == 0 && Matrix(y) == zeros(T, 3, 3)
      end
    end
  end

  @testset "the neighbourhood of the thresholds" begin
    for T in (Float32,)
      M = floatmax(T)
      # u runs from far below to the largest arguments, around where a sum overflows
      for F in (IndHyperplane, IndHalfspace), u in T.(M .* 2.0 .^ range(-5, 0; length = 21))
        check_overflow_steps(F(T[1, 1], 0), T[u, u], (T[1, 2],))
        check_overflow_steps(F(one(T), 0), T[u, u], (T[1, 2],))
        check_overflow_steps(F(T[1, 1], 0), T[u, u / 2], (T[1, 2],))
        check_overflow_steps(F(T[1, 1, 1, 1], 0), T[u, -u / 2, u / 2, u], (T[1, 2, 3, 4],))
      end
      # well below the thresholds the result is that of the formula, bit for bit
      for F in (IndHyperplane, IndHalfspace), u in T.(M .* (1 / 64, 1 / 32, 1 / 16))
        x = T[u, u / 2, -u / 4]
        for a in (T[1, 2, -1], T(3)), γ in (T(1), T[1, 2, 4])
          H = F(a, T(0.5))
          @test prox(H, x, γ) == plain_formula(H, x, γ)
        end
      end
    end
  end
  end
end

end
