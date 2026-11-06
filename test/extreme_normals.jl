module ExtremeNormalsTest

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

function check_projection(a, b, x)
  T = eltype(x)
  γv = T[0.5 + i / 4 for i in eachindex(x)]
  for F in (IndHyperplane, IndHalfspace)
    H = F(a, b)
    S = eps(typeof(H.b)) > eps(T) ? typeof(H.b) : T
    for γ in (T(0.7), γv)
      ref = projection_ref(a, H.b, x, γ; half = F === IndHalfspace)
      bound = 16length(x) * BigFloat(eps(S)) * max(norm(BigFloat.(x)), norm(ref)) + BigFloat(floatmin(S))
      y = prox(H, x, γ)
      @test norm(BigFloat.(y) - ref) <= bound
      @test H(y) == 0
      out = similar(x)
      @test value_and_prox!(out, H, x, γ) == 0
      @test norm(BigFloat.(out) - ref) <= bound
      @test H(out) == 0
    end

    # Move outward from the boundary, beyond the slack in the original units.
    A = normal_entries(a, x)
    boundary = projection_ref(a, H.b, x, one(T))
    tol = sqrt(BigFloat(eps(S)))
    slack = tol * max(1, abs(BigFloat(H.b)), norm(A) * norm(boundary))
    outside = T.(boundary .+ (16slack / sum(abs2, A)) .* A)
    # For a subnormal normal the absolute floor can cover every finite point.
    if all(isfinite, outside)
      residual = dot(A, BigFloat.(outside)) - BigFloat(H.b)
      exact_slack = tol * max(1, abs(BigFloat(H.b)), norm(A) * norm(BigFloat.(outside)))
      @test residual > 4exact_slack
      @test H(outside) == Inf
    end
  end
end

function check_residual(H, y)
  A = normal_entries(H.a, y)
  m = maximum(abs, A)
  s = ldexp(one(BigFloat), -exponent(m))
  A .*= s
  b = BigFloat(H.b) * s
  gap = dot(A, BigFloat.(y)) - b
  T = eltype(y)
  tol = T === Float32 ? 16BigFloat(eps(T)) : big"1e-10"
  @test (H isa IndHalfspace ? max(gap, 0) : abs(gap)) <= tol * max(1, abs(b))
end

@testset "projections with extreme normals" begin
  setprecision(BigFloat, 256) do
    for T in (Float32, Float64)
      e = round(Int, 0.6 * exponent(floatmax(T)))
      x, x0 = T[1.25, -0.75, 0.5, 2], T[0.5, 0.25, -0.5, 0.75]
      for c in (ldexp(one(T), -e), one(T), ldexp(one(T), e))
        a = c .* T[1, -0.5, 0, 0.25]
        b = T(dot(BigFloat.(a), BigFloat.(x0)))
        check_projection(a, b, x)
        check_projection(c, T(BigFloat(c) * sum(BigFloat.(x0))), x)
        check_projection([c], c, T[2])
      end
      tiny = nextfloat(zero(T))
      check_projection([tiny], tiny, T[2])
      check_projection(tiny, zero(T), T[1.25, -0.75, 0.5])
    end
    for a in ([1, -2, 0, 3], 2, typemin(Int), [typemin(Int), 0, 1, -2])
      check_projection(a, 0, [1.25, -0.75, 0.5, 2.0])
    end
    for (T, U) in ((Float32, Float64), (Float64, Float32))
      e = round(Int, 0.6 * exponent(floatmax(U)))
      for c in (ldexp(one(U), -e), one(U), ldexp(one(U), e))
        a = c .* U[1, -0.375, 0.125]
        x0 = U[0.5, 0.25, -0.5]
        b = U(dot(BigFloat.(a), BigFloat.(x0)))
        check_projection(a, b, T[1.25, -0.75, 2])
        check_projection(c, c, T[1.25, -0.75, 2])
      end
    end
  end

  @test prox(IndHyperplane([1e200], 1e200), [0.0]) ≈ [1.0]
  @test prox(IndHalfspace([1e200], 1e200), [2.0]) ≈ [1.0]
  @test prox(IndHyperplane([1e-170], 1e-170), [0.0]) ≈ [1.0]
  @test prox(IndHyperplane(1e200, 0), [1.0, 0.0]) ≈ [0.5, -0.5]
  @test IndHyperplane(1e200, 0)([1.0, 1.0]) == Inf
  @test prox(IndHyperplane([1e200], 1e200), [0.0], [2.0]) ≈ [1.0]
  x = ones(1000)
  @test prox(IndHyperplane(1e154, 0), x) == zeros(1000)
end

@testset "scale invariance of linear constraints" begin
  for T in (Float32, Float64)
    e = round(Int, 0.6 * exponent(floatmax(T)))
    rtol, atol = T === Float64 ? (1e-10, 1e-12) : (1e-4, 1e-6)
    for a in (one(T), T[1, -0.5, 0, 0.25])
      b = a isa Real ? T(1.5) : T(0.5)
      x0 = a isa Real ? fill(T(0.375), 4) : T[0.5, 0.25, 0.75, 0.5]
      direction = a isa Real ? ones(T, 4) : a
      xs = (x0 .+ T(0.4) .* direction, x0 .- T(0.3) .* direction, x0 .+ T(0.3) .* direction)
      for F in (IndHyperplane, IndHalfspace), c in (ldexp(one(T), -e), ldexp(one(T), e))
        ac, bc = c .* a, c * b
        @test ac ./ c == a
        @test bc / c == b
        for f in (nothing, NormL1(T(0.1)), IndBox(zero(T), one(T))), γ in (T(0.7), T[0.5, 1, 1.5, 2])
          H, Hc = F(a, b), F(ac, bc)
          G = f === nothing ? H : Constrained(f, H)
          Gc = f === nothing ? Hc : Constrained(f, Hc)
          for (j, x) in enumerate(xs)
            y, ref = prox(Gc, x, γ), prox(G, x, γ)
            @test isapprox(y, ref; rtol, atol = atol * max(1, norm(x)))
            check_residual(Hc, y)
            if F === IndHalfspace && j == 2
              free = f === nothing ? x : prox(f, x, γ)
              @test isapprox(y, free; rtol, atol = atol * max(1, norm(x)))
            elseif F === IndHalfspace
              check_residual(IndHyperplane(ac, bc), y)
            end
          end
        end
      end
    end
  end
  C = Constrained(NormL1(0.1), IndHyperplane([1e200, 1e200], 1e200))
  @test prox(C, [0.0, 0.0]) ≈ [0.5, 0.5]
end

@testset "intersections with extreme normals" begin
  x = [0.0, 2.0, -1.0]
  for c in (1e200, 1e-200)
    H = IndHyperplane(c .* [1.0, -0.5, 0.25], c * 0.5)
    G = IndIntersection(IndBox(-5, 5), H)
    ref = prox(IndIntersection(IndBox(-5, 5), IndHyperplane([1.0, -0.5, 0.25], 0.5)), x)
    y = prox(G, x)
    @test G.converged
    @test y ≈ ref rtol = 1e-10 atol = 1e-12
    check_residual(H, y)
  end
  G = IndIntersection(IndBox(-5, 5), IndHyperplane([1e200], 1e200))
  @test prox(G, [0.0]) ≈ [1.0]
  @test G.converged
end

@testset "support functions with extreme scalar normals" begin
  for T in (Float32, Float64)
    y = T[1.25, -0.75, 2, 0.5]
    for c in (1e200, -1e200, 1e-200, -1e-200), k in (1, 2, 2.5)
      s = sort(y; rev = true)
      j = floor(Int, k)
      ref = sum(s[1:j]) + T(k - j) * s[j + 1]
      H = IndHyperplane(c, c * k)
      S = Conjugate(Constrained(IndBox(0, 1), H))
      @test S(y) isa T
      @test S(y) ≈ ref
      # Reversing the box and the objective also reverses a lower sum bound.
      box = c > 0 ? IndBox(0, 1) : IndBox(-1, 0)
      Hhalf = IndHalfspace(c, abs(c) * k)
      Shalf = Conjugate(Constrained(box, Hhalf))
      v = c > 0 ? y : -y
      @test Shalf(v) ≈ sum(sort(max.(y, 0); rev = true)[1:j]) + T(k - j) * sort(max.(y, 0); rev = true)[j + 1]
    end
  end
end

@testset "spectral functions with extreme scalar normals" begin
  A = [2.0 0.25 0; 0.25 -1 0.5; 0 0.5 1]
  B = [2.0 0.5; -0.25 1; 0.75 -0.5]
  for c in (1e200, 1e-200), (Wrapper, X) in ((Spectral, A), (SingularValue, B))
    f = Constrained(IndBox(0, 1), IndHyperplane(c, c))
    f0 = Constrained(IndBox(0, 1), IndHyperplane(1, 1))
    G, G0 = Wrapper(f), Wrapper(f0)
    @test prox(G, X) ≈ prox(G0, X) rtol = 1e-10 atol = 1e-12
    S, S0 = Wrapper(Conjugate(f)), Wrapper(Conjugate(f0))
    @test S(X) ≈ S0(X)
    @test prox(S, X) ≈ prox(S0, X) rtol = 1e-10 atol = 1e-12
  end
end

@testset "finite nonzero normals" begin
  for F in (IndHyperplane, IndHalfspace)
    for T in (Float32, Float64), a in (nextfloat(zero(T)), floatmin(T), one(T), floatmax(T))
      @test F(a, zero(T)) isa F
      @test F([a], zero(T)) isa F
    end
    for a in (1, typemin(Int), [1, -2, 0], [typemin(Int), 0])
      @test F(a, 0) isa F
    end
    for T in (Float32, Float64), a in (zero(T), T(Inf), T(-Inf), T(NaN))
      @test_throws ArgumentError F(a, zero(T))
      @test_throws ArgumentError F([a], zero(T))
      iszero(a) || @test_throws ArgumentError F(T[1, a], zero(T))
    end
    @test_throws ArgumentError F(zeros(Int, 3), 0)
    for T in (Float32, Float64), b in (T(Inf), T(-Inf), T(NaN))
      @test_throws ArgumentError F(one(T), b)
      @test_throws ArgumentError F(T[1, -2], b)
    end
  end
end


# The projection coefficient g / d, with g = ⟨a, x⟩ - b and d = ‖a‖² or Σ γᵢaᵢ², is not rounded
# below the type of g / d or the output's: rounding it to a narrow output first loses a
# representable correction (R2, R3). Inside that type it can still underflow (#631).
function check_exact_projection(H, x, γ, expected)
  D = eltype(expected)
  if eltype(x) == D
    y = prox(H, x, γ)
    @test y isa typeof(expected)
    @test y == expected
    z = copy(x)
    @test prox!(z, H, z, γ) === z
    @test z == expected
  end
  y = fill!(similar(expected), NaN)
  @test prox!(y, H, x, γ) === y
  @test y == expected
  y = fill!(similar(expected), NaN)
  @test value_and_prox!(y, H, x, γ) === zero(D)
  @test y == expected
end

@testset "projection coefficient precision" begin
  @testset "coefficient below a Float32 output's range" begin
    for F in (IndHyperplane, IndHalfspace), σ in (-1, 1),
        a in (σ * 2.0^100, [σ * 2.0^100]), γ in (1.0, 1f0, [1.0], Float32[1])
      H = F(a, 0)
      # ⟨a, x⟩ = 1, so the coefficient is 2^-200 and the correction 2^-200 · a = x exactly;
      # this side violates the halfspace for either sign of the normal.
      x = Float32[σ * 2.0^-100]
      check_exact_projection(H, x, γ, Float32[0])
      # The other side is feasible and is returned unchanged.
      F === IndHalfspace && check_exact_projection(H, -x, γ, -x)
    end
  end

  @testset "scaled normal with a subnormal output" begin
    c = 1.5 * 2.0^300
    @test Float32[2.0^-149] == [nextfloat(0f0)]
    for F in (IndHyperplane, IndHalfspace), a in (c, -c), γ in (1.0, [1.0], Float32[1])
      b = a * 2.0^-149
      # From 0 the projection onto ⟨a, x⟩ = b is b / a = 2^-149 exactly, a Float32 subnormal.
      # The halfspace contains 0 when b > 0, so only the negative normal is active.
      @test b / a == 2.0^-149
      expected = F === IndHalfspace && a > 0 ? Float32[0] : Float32[2.0^-149]
      for normal in (a, [a])
        check_exact_projection(F(normal, b), Float32[0], γ, expected)
      end
    end
  end

  @testset "coefficient above a Float32 output's range" begin
    for F in (IndHyperplane, IndHalfspace), a in (2.0^-200, [2.0^-200]), γ in (1.0, [1.0], Float32[1])
      # The coefficient is 2^-200 / 2^-400 = 2^200, beyond Float32, and the correction
      # 2^200 · 2^-200 = 1 = x exactly; x violates the halfspace.
      check_exact_projection(F(a, 0), Float32[1], γ, Float32[0])
    end
  end

  @testset "scaled normal in a diagonal metric" begin
    # The normal is scaled by 2^-40 to (1, 2^-80). The second correction is
    # t γ₂ â₂ = 2^80 · 2^-80 · 2^-80. The Float64 argument makes the coefficient Float64,
    # and γ₂ â₂ = 2^-160 is formed in that type; in Float32 it underflows (below).
    a = Float32[2.0^40, 2.0^-40]
    x = Float64[1, 0]
    γ = Float32[2.0^-80, 2.0^-80]
    for D in (Float32, Float64), F in (IndHyperplane, IndHalfspace)
      expected = setprecision(BigFloat, 256) do
        # x violates the halfspace, so both sets have the same projection.
        D.(projection_ref(a, 0, x, γ))
      end
      @test expected[2] == -D(2.0^-80)
      ys = [prox!(fill(D(NaN), 2), F(a, 0f0), x, γ)]
      y = fill(D(NaN), 2)
      @test value_and_prox!(y, F(a, 0f0), x, γ) === zero(D)
      push!(ys, y)
      if D === Float64
        push!(ys, prox(F(a, 0f0), x, γ))
        z = copy(x)
        @test prox!(z, F(a, 0f0), z, γ) === z
        push!(ys, z)
      end
      for y in ys
        @test y isa Vector{D}
        @test y[2] == expected[2]
        # The exact first entry is about 2^-160: zero in Float32. In Float64 it is
        # within the normwise accuracy, since Σ γᵢaᵢ² is formed in the normal's type.
        D === Float32 && @test y[1] == expected[1] == 0
        @test abs(y[1] - expected[1]) <= eps(Float64) * norm(x)
      end
    end
    # With a Float32 argument the coefficient and γ₂ â₂ are Float32, and γ₂ â₂ = 2^-160
    # underflows although the correction -2^-80 is representable (#631).
    @test_broken prox!(zeros(Float32, 2), IndHyperplane(a, 0f0), Float32[1, 0], γ) == Float32[0, -2.0^-80]
  end

  @testset "mixed argument and output precision" begin
    for F in (IndHyperplane, IndHalfspace), a in (2f0^30, [2f0^30]), γ in (1.0, 1f0, [1.0], Float32[1])
      # The Float64 argument raises the model precision: the coefficient 2^-95 / 2^60 = 2^-155
      # is below Float32's range, and the correction 2^-155 · 2^30 = 2^-125 = x exactly.
      H = F(a, 0f0)
      x = Float64[2.0^-125]
      check_exact_projection(H, x, γ, Float32[0])
      check_exact_projection(H, x, γ, [0.0])
    end
  end

  @testset "integer normal" begin
    for F in (IndHyperplane, IndHalfspace), γ in (1.0, 1f0, Float32[1], [1.0])
      # The normal 2^62 is stored as an Int and the offset as Float64: ⟨a, x⟩ = 2^-38 > 0,
      # which violates the halfspace, and ‖a‖² = 2^124. The coefficient 2^-162 is below
      # Float32's range, and the correction 2^-162 · 2^62 = 2^-100 = x exactly.
      check_exact_projection(F(1 << 62, 0), Float32[2.0^-100], γ, Float32[0])
    end
  end

  @testset "Float64 steps with Float32 data" begin
    for F in (IndHyperplane, IndHalfspace)
      # ⟨a, x⟩ = 2^30 · 2^-130 = 2^-100 > 0 violates the halfspace. With the Float64 step,
      # Σ γᵢaᵢ² = 2^60 is Float64 and the coefficient 2^-160 is below Float32's range
      # but not Float64's; the correction 2^-160 · 2^30 = 2^-130 = x exactly (a Float32
      # subnormal), so the projection is 0.
      check_exact_projection(F(Float32[2.0^30], 0f0), Float32[2.0^-130], [1.0], Float32[0])
    end
  end

  @testset "sparse arguments and wrappers" begin
    for F in (IndHyperplane, IndHalfspace), a in (2.0^100, [2.0^100, 2.0^100]),
        γ in (1.0, [1.0, 1.0], Float32[1, 1])
      H = F(a, 0)
      # ⟨a, x⟩ = 1 and ‖a‖² = 2^201: the correction 2^-201 · 2^100 = 2^-101 per entry.
      x = Float32[2.0^-100, 0]
      expected = Float32[2.0^-101, -2.0^-101]
      check_exact_projection(H, x, γ, expected)
      y = SparseIterate(Float32, 2)
      @test prox!(y, H, SparseIterate(x), γ) === y
      @test Array(y) == expected
      y = SparseIterate(Float32, 2)
      @test value_and_prox!(y, H, SparseIterate(x), γ) === 0f0
      @test Array(y) == expected
      z = SparseIterate(x)
      @test prox!(z, H, z, γ) === z
      @test Array(z) == expected
    end
    for F in (IndHyperplane, IndHalfspace), γ in (1.0, [1.0], Float32[1])
      # x - b = 2^-101 projects to 0, so the result is b: the set is {u : 2^100 (u - b) = 0}.
      T = Translate(F(2.0^100, 0), Float32[2.0^-101])
      check_exact_projection(T, Float32[2.0^-100], γ, Float32[2.0^-101])
    end
  end

  @testset "argument checks run before writing" begin
    for F in (IndHyperplane, IndHalfspace), a in ([2.0^100, 1.0], 2.0^100, [1.5 * 2.0^300, 2.0^300])
      H = F(a, 0)
      x = Float32[0.25, 0.5]
      for op in (prox!, value_and_prox!)
        function rejects(E, y, x, γ)
          y0, x0 = copy(y), copy(x)
          @test_throws E op(y, H, x, γ)
          @test isequal(y, y0)
          @test isequal(x, x0)
        end
        message = ArgumentError("the steps of $(nameof(F)) must be positive and finite")
        for bad in (-1f0, NaN32, 0f0, Inf32)
          rejects(message, fill(NaN32, 2), x, Float32[1, bad])
          alias = copy(x)
          rejects(message, alias, alias, Float32[bad, 1])
        end
        rejects(DimensionMismatch, fill(NaN32, 2), x, ones(Float32, 3))
        for γ in (1f0, ones(Float32, 2))
          rejects(DimensionMismatch, fill(NaN32, 3), x, γ)
          rejects(ArgumentError, fill(NaN32, 2), ComplexF32.(x), γ)
          # An integer output takes only integer values; this projection has none.
          @test_throws InexactError op(zeros(Int, 2), H, x, γ)
        end
        if a isa AbstractArray
          for γ in (1f0, ones(Float32, 3))
            rejects(DimensionMismatch, fill(NaN32, 3), Float32[0.25, 0.5, 1], γ)
          end
        end
      end
    end
  end
end


# Projections whose dot product, residual or correction overflows although the projection is
# representable (#563). The reference is a BigFloat formed from the original data, wide enough to
# add the products of any two doubles exactly. The bound is componentwise and in the error scale of
# the formula x - ((⟨a, x⟩ - b) / Σ γᵢaᵢ²) γ .* a: each entry has the relative error of a few
# operations on its own size, n·eps(S) times |xᵢ| and times the scale of its correction,
# |γᵢaᵢ| (Σ|aⱼxⱼ| + |b|) / Σγa². S has the largest eps of the output, the offset and the normal,
# and the factor 16 is that of check_projection above. floatmin(S) covers the underflow of the
# tiniest corrections, which is not what these tests are about (#631).
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

# The formula of the ordinary path, for a normal that is not scaled and arguments that are finite.
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
  setprecision(BigFloat, OVERFLOW_PRECISION)
  @testset "the two examples of the issue" begin
    M = floatmax(Float64)
    @test prox(IndHalfspace([1.0, 1.0], 0.0), [M, M], 1.0) == [0.0, 0.0]
    @test prox(IndHyperplane([1.0, 1.0], 1e308), [M, M], 1.0) == [5e307, 5e307]
    @test IndHalfspace([1.0, 1.0], 0.0)([0.0, 0.0]) == 0
    @test IndHyperplane([1.0, 1.0], 1e308)([5e307, 5e307]) == 0
  end

  for T in (Float64, Float32)
    M = floatmax(T)
    em = exponent(M)
    e4 = em ÷ 4
    up, down = ldexp(one(T), e4 + 45), ldexp(one(T), -e4 - 45)   # scale the normal out of range
    steps(n) = (T[0.5 + i / 4 for i in 1:n], T[ldexp(one(T), i % 4) for i in 1:n])
    @testset "$T" begin
      @testset "array normals" begin
        for F in (IndHyperplane, IndHalfspace)
          # the sum of the entries overflows; the projection is zero, or half the offset
          check_overflow_steps(F(T[1, 1], 0), T[M, M], steps(2))
          check_overflow_steps(F(T[1, 1], T(0.9) * M), T[M, M], steps(2))
          # unequal and signed normals, offsets of either sign
          check_overflow_steps(F(T[1, 2], 0), T[M / 2, M / 2], steps(2))
          check_overflow_steps(F(T[-1, 3, 0.5], T(0.25) * M), T[M / 4, -M / 8, M / 2], steps(3))
          check_overflow_steps(F(T[2, -1, 0.5, 4], -T(0.125) * M), T[-M / 2, M / 4, M / 2, M / 8], steps(4))
          # a partial sum overflows although the dot product does not
          check_overflow_steps(F(T[1, 1, 1, 1], ldexp(one(T), em - 3)), T[M / 2, M / 2, M / 2, -M / 2], steps(4))
          check_overflow_steps(F(T[1, 1], T(0.5) * M), T[M, M / 2], steps(2))
          # the exact correction is 1.4 floatmax, the projection is not
          check_overflow_steps(F(T[1], -T(0.5) * M), T[T(0.9) * M], (T[0.75],))
          # the halfspace's inactive side
          check_overflow_steps(F(T[1, 1], 0), T[-M, -M], steps(2))
        end
        @test prox(IndHalfspace(T[1, 1], 0), T[-M, -M]) == T[-M, -M]
      end

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

      @testset "steps whose weighted squared norm or product overflows" begin
        for F in (IndHyperplane, IndHalfspace)
          # Σ γᵢaᵢ² overflows; the formula returned the argument for d = Inf
          a = ldexp(one(T), e4 - 5)
          check_overflow(F(T[a, a], 0), T[1, 0], fill(ldexp(one(T), em - 2 * (e4 - 5) + 2), 2))
          # the same with a normal that is scaled
          check_overflow(F(up .* T[1, 1.5], 0), T[1, 0], fill(M / 2, 2))
          # a scalar normal with Σ γᵢ overflowing
          check_overflow(F(ldexp(one(T), e4 ÷ 2), 0), T[1, 0], T[M, M])
          # t γᵢ overflows although t γᵢ aᵢ does not
          pa, pγ, px = T === Float64 ? (-600, 1000, 100) : (-60, 100, 40)
          check_overflow(F(T[1, ldexp(one(T), pa)], 0), T[ldexp(one(T), px), 0], T[1, ldexp(one(T), pγ)])
          # integer steps, and steps of the other precision
          check_overflow(F(T[1, 1], 0), T[M, M], [1, 2])
          check_overflow(F(T[1, 1], 0), T[M, M], Float32[1, 3])
          check_overflow(F(T[1, 1], 0), T[M, M], Float64[1, 3])
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
    # A Float64 model with a Float32 output: the Float64 dot product 2^900 (x₁ + x₂) overflows, the
    # projection is formed in Float64 and rounded once. A well-conditioned projection that fits
    # Float32 has an argument of that size too.
    for F in (IndHyperplane, IndHalfspace), γ in (1.0, [1.0, 2.0], [3.0, 3.0])
      check_overflow(F(2.0^900 .* [1.0, 1.0], 0.0), [2.0^127, 2.0^124], γ, Float32)
      check_overflow(F(2.0^900, 0.0), [2.0^127, 2.0^124], γ, Float32)
      check_overflow(F(2.0^900 .* [1.0, -0.5, 0.25], 2.0^1000), [2.0^127, -2.0^124, 2.0^125], [1.0, 2.0, 3.0] .* (γ isa Real ? γ : γ[1]), Float32)
    end
    # Float32 data whose Float32 dot product overflows, projected into a Float64 output
    for F in (IndHyperplane, IndHalfspace), γ in (1.0, [1.0, 2.0])
      check_overflow(F(Float32[1, 1], 0f0), Float32[M32, M32], γ, Float64)
      check_overflow(F(Float32[1, 1], 2f0^100), Float32[M32, M32], γ, Float64)
      check_overflow(F(Float32[1, 1.5], 0f0), Float32[M32, M32 / 2], γ, Float64)
      check_overflow(F(2f0^40 .* Float32[1, 1], 0f0), Float32[2f0^100, 2f0^99], γ, Float64)
    end
  end

  @testset "iterates" begin
    for T in (Float64, Float32), F in (IndHyperplane, IndHalfspace)
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
      for a in (ones(T, 3, 3), T(1), up .* ones(T, 3, 3)), γ in (T(1), steps)
        H = F(a, 0)
        y = SymmetricSparseIterate(T, 3)
        @test prox!(y, H, SymmetricSparseIterate(X), γ) === y
        γv = γ isa AbstractArray ? vec(γ) : γ
        check_against_reference(H, vec(X), vec(Array(y)), γv)
        y = SymmetricSparseIterate(T, 3)
        @test value_and_prox!(y, H, SymmetricSparseIterate(X), γ) === zero(T)
        check_against_reference(H, vec(X), vec(Array(y)), γv)
        z = SymmetricSparseIterate(X)
        @test prox!(z, H, z, γ) === z
        check_against_reference(H, vec(X), vec(Array(z)), γv)
      end
    end
  end

  @testset "the neighbourhood of the thresholds" begin
    for T in (Float64, Float32)
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

  @testset "projections that do not fit, and arguments that are not finite" begin
    M = floatmax(Float64)
    # the exact projection of 0 is 2M: it overflows, and is not clamped
    @test prox(IndHyperplane([0.5], M), [0.0]) == [Inf]
    @test prox(IndHyperplane([0.5], -M), [0.0]) == [-Inf]
    @test prox(IndHyperplane([0.5], M), [0.0], [2.0]) == [Inf]
    y = zeros(1)
    value_and_prox!(y, IndHyperplane([0.5], M), [0.0], 1.0)
    @test y == [Inf]
    @test prox!(zeros(Float32, 2), IndHyperplane([1.0, 1.0], 2.0^130), [0.0, 0.0], 1.0) == fill(Inf32, 2)
    @test prox!(zeros(Float32, 1), IndHyperplane([0.5], 2.0^253), [0.0], 1.0) == Float32[Inf]
    # a NaN or an infinite entry keeps the result of the formula, in IEEE arithmetic
    for x in ([NaN, M], [Inf, 1.0], [Inf, -Inf], [M, -Inf], [1.0, NaN]), F in (IndHyperplane, IndHalfspace)
      for a in ([1.0, 1.0], 1.0), γ in (1.0, [1.0, 2.0])
        H = F(a, 0.0)
        y = prox(H, x, γ)
        @test isequal(y, plain_formula(H, x, γ))
        @test !all(isfinite, y)
      end
      for a in ([1e300, 1e300], 1e300), γ in (1.0, [1.0, 2.0])
        @test !any(isfinite, prox(F(a, 0.0), [NaN, M], γ))
      end
    end
    # a step below floatmin: the coefficient overflows or underflows (#631), and the result is the formula's
    H = IndHyperplane([1.0], 0.0)
    for γ in (2.0^-1074, 3 * 2.0^-1074)
      @test isequal(prox(H, [1.0], [γ]), plain_formula(H, [1.0], [γ]))
      @test isequal(prox(H, [1.0], [γ]), [-Inf])
    end
  end

  @testset "a model type narrower than the data" begin
    # The squares of the normal entries underflow in Float32, as do the products of Float32 steps
    # with them, but not in the Float64 model of the projection; the entries 1 and 2 of the exact
    # projection are beyond floatmax and the others are within the bound.
    a = Float32[-5.713f-40, -2.5967962f-13, 2.7628436f-11, -4.1102825f-13, -4.8f-44]
    x = [1.2911956633587734e307, -1.593686991591948e307, 5.510386459094743e307, -3.19143653438417e307, 1.8894918353094168e307]
    γ = [7.760465636490048e303, 2.5402850971441077e203, 1.2194554501014753e190, 2.1865531330082964e194, 1.8768120776652858e185]
    for F in (IndHyperplane, IndHalfspace)
      H = F(a, 1.0431324f38)
      setprecision(BigFloat, OVERFLOW_PRECISION) do
        ref = projection_ref(a, H.b, x, γ; half = F === IndHalfspace)
        bound = overflow_bound(a, H.b, x, γ, Float64)
        y = prox(H, x, γ)
        for i in eachindex(x)
          if abs(ref[i]) > floatmax(Float64)
            @test y[i] == sign(ref[i]) * Inf
          else
            @test abs(BigFloat(y[i]) - ref[i]) <= bound[i]
          end
        end
      end
    end
  end

  @testset "preflight at the arguments of the overflowing projections" begin
    M = floatmax(Float64)
    for F in (IndHyperplane, IndHalfspace), op in (prox!, value_and_prox!)
      H = F([1.0, 1.0], 0.0)
      x = [M, M]
      y = [1.0, 2.0]
      for bad in (-1.0, NaN, 0.0, Inf)
        @test_throws ArgumentError op(y, H, x, [1.0, bad])
        @test y == [1.0, 2.0]
        @test x == [M, M]
      end
      @test_throws DimensionMismatch op(y, H, x, ones(3))
      @test_throws DimensionMismatch op([1.0, 2.0, 3.0], H, x, 1.0)
      @test y == [1.0, 2.0]
    end
  end

  @testset "the conjugate of a hyperplane or halfspace" begin
    # x / γ = 2^1023 is finite, and its projection is 1/2 in each entry
    for F in (IndHyperplane, IndHalfspace)
      y = prox(Conjugate(F([1.0, 1.0], 1.0)), [2.0^1000, 2.0^1000], 2.0^-23)
      @test y ≈ [2.0^1000, 2.0^1000]
    end
  end
end

end
