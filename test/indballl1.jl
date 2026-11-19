module IndBallL1Test

using Test, ProximalBase, OffsetArrays, Random

# The full Float64 exponent span, plus room for the sums and divisions in
# these cases. The threshold comes from sorted prefix sums.
function reference_projection(x, r)
  setprecision(BigFloat, 2304) do
    magnitudes = sort!(abs.(BigFloat.(vec(x))); rev = true)
    radius = BigFloat(r)
    sum(magnitudes) <= radius && return BigFloat.(x)
    total = zero(radius)
    for k in eachindex(magnitudes)
      total += magnitudes[k]
      threshold = (total - radius) / k
      if k == length(magnitudes) || threshold >= magnitudes[k + 1]
        return map(v -> copysign(max(abs(BigFloat(v)) - threshold, zero(radius)), v), x)
      end
    end
    error("reference threshold was not found")
  end
end

function check_kkt(x, y, r, tol)
  setprecision(BigFloat, 2304) do
    point, projected, radius = BigFloat.(x), BigFloat.(y), BigFloat(r)
    residual = point .- projected
    norm1 = sum(abs, projected)
    norminf_residual = maximum(abs, residual)
    inner_product = sum(projected .* residual)
    @test norm1 <= radius + length(y) * tol
    @test abs(radius * norminf_residual - inner_product) <=
      length(y) * tol * (norminf_residual + maximum(abs, point))
  end
end

@testset "constructor and traits" begin
  @test IndBallL1().r == 1
  @test_throws ArgumentError IndBallL1(-1)
  @test_throws ArgumentError IndBallL1(-Inf)
  @test_throws ArgumentError IndBallL1(NaN)
  f = IndBallL1()
  @test isconvex(f) && isproximable(f)
  @test !issmooth(f) && !isseparable(f)
  @test Scaled(f, 0) === f
  @test IndBallL1(0.0) == IndBallL1(-0.0)
  @test hash(IndBallL1(0.0)) == hash(IndBallL1(-0.0))
  @test repr(IndBallL1(2.5)) == "IndBallL1(2.5)"
end

@testset "analytic projection and exact identity ($T)" for T in (Float32, Float64)
  for γ in (zero(T), T(0.3), T(Inf))
    for (x, r, expected) in ((T[3, -1], T(2), T[2, 0]),
                              (T[3, -1], T(3), T[2.5, -0.5]),
                              (T[-3, -1], T(3), T[-2.5, -0.5]),
                              (T[-3], T(2), T[-2]),
                              (T[3 0; -1 0], T(3), T[2.5 0; -0.5 0]))
      f = IndBallL1(r)
      @test f(x) === T(Inf)
      @test prox(f, x, γ) ≈ expected
      y = copy(x)
      @test isequal(value_and_prox!(y, f, y, γ), f(y))
      @test y ≈ expected
    end
    x = T[3, -1, -0.0]
    for r in (T(4), nextfloat(T(4)), T(5), T(Inf))
      f = IndBallL1(r)
      @test f(x) === zero(T)
      @test isequal(prox(f, x, γ), x)
    end
    for r in (zero(T), one(T), T(Inf)), x in (T[], zeros(T, 0, 2))
      f = IndBallL1(r)
      @test f(x) === zero(T)
      @test size(prox(f, x, γ)) == size(x)
      @test value_and_prox!(x, f, x, γ) === zero(T)
    end
  end
  x = view(T[3, 9, -1, 9], 1:2:4)
  destination = zeros(T, 4)
  y = view(destination, 2:2:4)
  @test prox!(y, IndBallL1(T(3)), x) === y
  @test destination ≈ T[0, 2.5, 0, -0.5]
  parent_x = T[3, 9]
  x = view(parent_x, [1, 1])
  f = IndBallL1(one(T))
  @test prox!(x, f, x) === x
  @test collect(x) ≈ fill(T(0.5), 2)
  @test parent_x[2] == T(9)
  snapshot = f.scratch.buf
  parent_x[1] = T(3)
  @test prox!(x, f, x) === x
  @test f.scratch.buf === snapshot
  parent_x[1] = floatmax(T)
  @test prox!(x, f, x) === x
  @test collect(x) == fill(T(0.5), 2)
  @test f.scratch.buf === snapshot

  r = prevfloat(T(4))
  @test IndBallL1(r)(T[3, -1]) === T(Inf)
  @test prox(IndBallL1(r), T[3, -1]) ≈ T.(reference_projection(T[3, -1], r))
end

@testset "computed feasibility at a rounded tie ($T)" for T in (Float32, Float64)
  for (r, tail) in ((one(T), eps(T)/4), (ldexp(one(T), exponent(floatmax(T))), one(T)))
    x = T[r, tail]
    f = IndBallL1(r)
    @test T.(reference_projection(x, r)) == T[r, tail/2]
    @test f(x) === zero(T)
    @test isequal(prox(f, x), x)
  end
end

@testset "sorted high-precision reference and KKT ($T)" for T in (Float32, Float64)
  rng = Xoshiro(955)
  for n in (2, 5, 16, 100), x in (randn(rng, T, n),
      rand(rng, T[-3, -1, 0, 1, 3], n),
      randn(rng, T, n) .* T.(exp2.(rand(rng, -25:25, n))))
    total = sum(abs, x)
    for ratio in T[0.01, 0.4, 0.9, 1.3]
      r = ratio * total
      f = IndBallL1(r)
      y = prox(f, x)
      ref = reference_projection(x, r)
      tol = 8n * eps(T) * maximum(abs, x) + 2nextfloat(zero(T))
      @test maximum(abs, BigFloat.(y) .- ref) <= tol
      check_kkt(x, y, r, tol)
      @test maximum(abs, prox(f, y) .- y) <= 2tol
      perm = randperm(rng, n)
      signs = rand(rng, T[-1, 1], n)
      @test maximum(abs, prox(f, signs .* x[perm]) .- signs .* y[perm]) <= 2tol
    end
  end
end

@testset "finite range and stored value ($T)" for T in (Float32, Float64)
  M, δ = floatmax(T), nextfloat(zero(T))
  cases = ((T[M, -M], one(T)),
           (T[M, prevfloat(M)], one(T)),
           (T[M, -M, prevfloat(M)], 3(M - prevfloat(M))),
           (T[M, -M], 3δ),
           (T[M, -M/2], δ),
           (T[M, -M], M),
           (T[M/2, -M/2, M/4], M))
  for (x, r) in cases
    f = IndBallL1(r)
    reference = T.(reference_projection(x, r))
    @test f(x) === T(Inf)
    for inplace in (false, true)
      y = inplace ? copy(x) : similar(x)
      source = inplace ? y : x
      value = value_and_prox!(y, f, source)
      @test isequal(value, f(y))
      @test all(i -> isapprox(y[i], reference[i]; rtol = 16eps(T)), eachindex(y))
    end
  end
  # The total radius also needs recovery when it is spread across many ties.
  x = fill(one(T), 2048)
  r = 32eps(T)
  @test all(==(r / length(x)), prox(IndBallL1(r), x))

  # Losing the tied coordinates can remove mass even when the largest
  # coordinate retains a nonzero projection.
  x[1] += 512eps(T)
  r = 576eps(T)
  ref = T.(reference_projection(x, r))
  @test all(isapprox.(prox(IndBallL1(r), x), ref; rtol = 16eps(T)))

  # The radius is small relative to the whole sum, but the first coordinate
  # holds it exactly. An eligible dense output suffices as the level buffer.
  x = vcat(one(T), fill(T(0.5), 2048))
  f = IndBallL1(128eps(T))
  y = prox(f, x)
  @test y[1] ≈ f.r && all(iszero, @view y[2:end])
  @test f.scratch.buf === nothing

  f = IndBallL1(3δ)
  y = similar(T[M, -M])
  @test value_and_prox!(y, f, T[M, -M]) === T(Inf)
  @test y == T[2δ, -2δ]
end

@testset "cancellation close to the threshold ($T)" for T in (Float32, Float64)
  e, δ, m = eps(T), nextfloat(zero(T)), floatmin(T)
  cases = ((T[1, -1], 17 * e / 2),
           (T[1 + 9 * e, -1], 37 * e / 4),
           (T[δ, δ], δ),
           (T[m, prevfloat(m)], 2δ))
  for (x, r) in cases
    f = IndBallL1(r)
    ref = T.(reference_projection(x, r))
    @test prox(f, x) == ref
    y = copy(x)
    @test isequal(value_and_prox!(y, f, y), f(y))
    @test y == ref
    # Shift the two active entries apart to exercise implicit sparse zeros.
    sparse_x = SparseIterate(T[x[1], 0, x[2], 0])
    @test prox!(sparse_x, f, sparse_x) === sparse_x
    @test Array(sparse_x) == T[ref[1], 0, ref[2], 0]
  end
end

@testset "cancellation beyond the original screen ($T)" for T in (Float32, Float64)
  e = eps(T)
  b = one(T) + 16 * e
  scale = ldexp(one(T), exponent(floatmax(T)) - 1)
  for k in (one(T), scale), (x, r, exact) in (
      (T[b, -b], 17 * e, T[8.5, -8.5] .* e),
      (T[1 + 50 * e, -(1 + 33 * e)], 35 * e / 2, T[69/4, -1/4] .* e))
    arg, radius, expected = k .* x, k * r, k .* exact
    @test T.(reference_projection(arg, radius)) == expected
    @test prox(IndBallL1(radius), arg) == expected
    y = copy(arg)
    @test prox!(y, IndBallL1(radius), y) === y
    @test y == expected
  end
end

@testset "nonfinite inputs and Scaled ($T)" for T in (Float32, Float64)
  δ = nextfloat(zero(T))
  f = IndBallL1(T(2))
  small = Scaled(Scaled(f, δ), δ)
  large = Scaled(Scaled(f, floatmax(T)), floatmax(T))
  for γ in (zero(T), T(0.3), T(Inf)), h in (f, Scaled(f, zero(T)),
      Scaled(f, T(0.5)), Scaled(f, T(Inf)), small, large)
    @test prox(h, T[3, -1], γ) ≈ T[2, 0]
    @test prox(h, T[Inf, 3], γ) == T[2, 0]
    @test prox(h, T[3, -Inf], γ) == T[0, -2]
    @test h(T[Inf, 0]) === T(Inf)
    @test h(T[-Inf, Inf]) === T(Inf)
    @test all(isnan, prox(h, T[-Inf, Inf], γ))
    for x in (T[NaN, 1, Inf], T[1, NaN, Inf], T[1, Inf, NaN])
      @test isnan(h(x))
      y = copy(x)
      @test isnan(value_and_prox!(y, h, y, γ))
      @test all(isnan, y)
    end
    y = similar(T[3, 0])
    @test value_and_prox!(y, h, T[1, 0], γ) === h(y) === zero(T)
  end
  for γ in (zero(T), one(T), T(Inf)), x in (T[NaN, Inf], T[-0.0, 0], T[-Inf, 1])
    zero_ball, full_ball = IndBallL1(zero(T)), IndBallL1(T(Inf))
    @test isequal(prox(zero_ball, x, γ), zeros(T, length(x)))
    @test isequal(prox(full_ball, x, γ), x)
    @test isequal(full_ball(x), any(isnan, x) ? T(NaN) : zero(T))
    @test isequal(zero_ball(x), any(isnan, x) ? T(NaN) : all(iszero, x) ? zero(T) : T(Inf))
  end
end

@testset "iterate storage and validation before writes ($T)" for T in (Float32, Float64)
  f = IndBallL1(T(3))
  for dense in (T[3, 0, -1, 0], T[3 -1; 0 0])
    x = SparseIterate(dense)
    y = SparseIterate(fill(T(9), size(x)))
    ref = T.(reference_projection(dense, T(3)))
    @test f(x) === T(Inf)
    @test Array(prox(f, x)) ≈ ref
    @test isequal(value_and_prox!(y, f, x, zero(T)), f(y))
    @test Array(y) ≈ ref
    @test prox!(x, f, x) === x
    @test Array(x) ≈ ref
  end
  x = SparseIterate(T[0, 3, 0])
  x[2] = T(-Inf)
  @test value_and_prox!(x, f, x) === zero(T)
  @test Array(x) == T[0, -3, 0]
  x[2] = T(NaN)
  @test isnan(value_and_prox!(x, f, x))
  @test all(isnan, x) && nactive(x) == length(x)

  x = BlockIterate(T[3, 0, -1, 0], [[3, 1], [4, 2]])
  y = similar(x)
  @test isequal(value_and_prox!(y, f, x), f(y))
  @test collect(y) ≈ T[2.5, 0, -0.5, 0]
  @test nactive(y) == 1
  @test prox!(x, f, x) === x
  @test collect(x) ≈ collect(y)
  @test prox!(x, IndBallL1(zero(T)), x) === x
  @test isequal(collect(x), zeros(T, 4))

  x = T[3, -1]
  y = fill(T(9), length(x))
  δ = nextfloat(zero(T))
  sx = SymmetricSparseIterate(T[3 0; 0 -1])
  sy = SymmetricSparseIterate(T[8 1; 1 8])
  for r in (zero(T), T(3), T(Inf))
    ball = IndBallL1(r)
    inner = Scaled(ball, T(0.5))
    # The explicit type constructor preserves a nested wrapper with a normal
    # product, so it exercises recursive validation outside the chain path.
    nested = Scaled{typeof(inner), T}(inner, T(0.5))
    for h in (ball, Scaled(ball, δ), Scaled(Scaled(ball, δ), δ), nested)
      for op in (prox!, value_and_prox!)
        for γ in (T(-1), -δ, T(NaN))
          @test_throws ArgumentError op(y, h, x, γ)
          @test y == fill(T(9), length(x))
          @test_throws ArgumentError prox(h, x, γ)
        end
        @test_throws DimensionMismatch op(y, h, reshape(x, 1, 2), one(T))
        @test_throws ArgumentError op(y, h, complex.(x), one(T))
        complex_y = complex.(y)
        @test_throws ArgumentError op(complex_y, h, x, one(T))
        @test complex_y == complex.(y)
        @test_throws ArgumentError op(y, h, OffsetArray(x, 0:1), one(T))
        @test_throws ArgumentError op(OffsetArray(y, 0:1), h, x, one(T))
        @test y == fill(T(9), length(x))
        dense = fill(T(9), 2, 2)
        @test_throws ArgumentError op(dense, h, sx, one(T))
        @test dense == fill(T(9), 2, 2)
        @test_throws ArgumentError op(sy, h, dense, one(T))
        @test Matrix(sy) == T[8 1; 1 8]
      end
      @test_throws ArgumentError h(sx)
      @test_throws ArgumentError prox(h, sx)
    end
  end
end

end
