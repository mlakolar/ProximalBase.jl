module IndBallL2Test

using Test, ProximalBase, OffsetArrays, LinearAlgebra, SparseArrays

# ---- Scaled returns the indicator (the same checks for each built-in indicator) -----------------------------
const CANON_FACTORS = Any[2, 0, 1, 0.5, 2.0, 2f0, 3f0, 5e-324, 1e-300, floatmax(Float64), floatmax(Float32), nextfloat(0f0),
                          Inf, Inf32, 0.0, -0.0, big(2.0), big(0), big(Inf)]
const CANON_INVALID = Any[-1, -1.0, NaN, -Inf, -1f0, NaN32, -5e-324, -Inf32, big(-1.0), big(NaN)]
scaled_allocs(f, α) = (Scaled(f, α); @allocated Scaled(f, α))
times_allocs(f, α) = (α * f; @allocated α * f)

# The listed valid factors, nested chains whose product overflows, underflows or passes through Inf, and the
# invalid factors; construction is inferred and does not allocate.
function check_canonical(f)
  for α in CANON_FACTORS
    @test Scaled(f, α) === f
    @test α * f === f
    @test f * α === f
    @test Scaled{AbstractFunction, typeof(α)}(f, α) === f
    @test Scaled{typeof(f), typeof(α)}(f, α) === f
  end
  @test Scaled(Scaled(f, 1e300), 1e100) === f
  @test Scaled(Scaled(f, 1e-300), 1e-100) === f
  @test Scaled(Scaled(f, 1f30), 1f30) === f
  @test Scaled(Scaled(f, 5e-324), 5e-324) === f
  @test 0 * (Inf * f) === f
  @test Scaled(Scaled(f, Inf), 0) === f
  @test Inf * (0 * f) === f
  @test (2 * f) * 3 === f
  for α in CANON_INVALID
    @test_throws ArgumentError Scaled(f, α)
    @test_throws ArgumentError α * f
    @test_throws ArgumentError f * α
    @test_throws ArgumentError Scaled{AbstractFunction, typeof(α)}(f, α)
  end
  @test_throws ArgumentError Scaled(Scaled(f, 2), -1)
  @test_throws ArgumentError Scaled(Scaled(f, Inf), NaN)
  for α in (2, 2.0, 2f0, 0, Inf, -0.0, 1e-300)
    @test @inferred(Scaled(f, α)) === f
    @test @inferred(α * f) === f
    @test scaled_allocs(f, α) == 0
    @test times_allocs(f, α) == 0
  end
  @test @inferred(Scaled(Scaled(f, 1e300), 1e100)) === f
  @test !(Scaled(f, 2) isa Scaled) && Scaled(f, 2) isa typeof(f)
  @test Scaled(NormL1(1.0), 2.0) isa Scaled
end

# A negative, NaN or tiny negative scalar step through a scaled indicator throws before anything is written, and
# the other steps give the result of the indicator at the points x (the vectors of one element type).
function check_scaled_steps(f, points)
  for α in (2, Inf, 1e-300), x in points
    T = eltype(x)
    s = α * f
    for γ in (-one(T), T(NaN), -nextfloat(zero(T)), -1)
      y = fill(T(7), length(x))
      @test_throws ArgumentError prox!(y, s, x, γ)
      @test all(==(T(7)), y)
      @test_throws ArgumentError value_and_prox!(y, s, x, γ)
      @test all(==(T(7)), y)
      @test_throws ArgumentError prox(s, x, γ)
    end
    for γ in (zero(T), one(T), T(0.3), T(Inf), 2, 0)
      y1, y2 = similar(x), similar(x)
      @test isequal(prox!(y1, s, x, γ), prox!(y2, f, x, γ)) && isequal(y1, y2)
      @test isequal(s(x), f(x))
      v1 = value_and_prox!(y1, s, x, γ)
      v2 = value_and_prox!(y2, f, x, γ)
      @test isequal(v1, v2) && isequal(y1, y2) && isequal(v1, s(y1))
    end
  end
end


@testset "IndBallL2 constructor and tools" begin
  @test IndBallL2().r == 1
  @test_throws ArgumentError IndBallL2(-1)
  @test_throws ArgumentError IndBallL2(NaN)
  @test isconvex(IndBallL2()) && isproximable(IndBallL2())
  @test !isseparable(IndBallL2()) && !iszeropreserving(IndBallL2())
  @test Scaled(IndBallL2(), 0) === IndBallL2()
end

@testset "Scaled returns the ball" begin
  for f in (IndBallL2(), IndBallL2(2.5), IndBallL2(0), IndBallL2(Inf), IndBallL2(2f0), IndBallL2(floatmax(Float64)),
            IndBallL2(big(2.0)))
    check_canonical(f)
  end
  for f in (IndBallL2(2.0), IndBallL2(0.0), IndBallL2(Inf), IndBallL2(1.0))
    check_scaled_steps(f, (Float64[3, -1], Float64[0.1, 0.2], Float64[NaN, 1], Float64[Inf, 3], Float64[], Float64[0, -0.0]))
  end
  for f in (IndBallL2(2f0), IndBallL2(1f0))
    check_scaled_steps(f, (Float32[3, -1], Float32[0.1, 0.2], Float32[NaN, 1], Float32[Inf, 3]))
  end
end

@testset "IndBallL2 analytic projections" begin
  for T in (Float32, Float64)
    for (x, p) in ((T[3, -4], T[1.5, -2]), (T[3 0; 0 -4], T[1.5 0; 0 -2]))
      @test IndBallL2(T(5))(x) === zero(T)
      @test IndBallL2(T(6))(x) === zero(T)
      @test IndBallL2(T(2.5))(x) === T(Inf)
      for γ in (zero(T), T(0.3), T(Inf))
        @test prox(IndBallL2(T(5)), x, γ) == x
        @test prox(IndBallL2(T(6)), x, γ) == x
        f = IndBallL2(T(2.5))
        @test prox(f, x, γ) == p
        y = copy(x)
        @test value_and_prox!(y, f, y, γ) === f(y) === zero(T)
        @test y == p
      end
    end
    for x in (T[], zeros(T, 0, 3)), r in (zero(T), one(T), T(Inf))
      f = IndBallL2(r)
      @test f(x) === zero(T)
      @test size(prox(f, x)) == size(x)
      @test value_and_prox!(x, f, x, zero(T)) === zero(T)
    end
    for r in (nextfloat(zero(T)), one(T), prevfloat(floatmax(T)))
      f = IndBallL2(r)
      @test prox(f, T[-floatmax(T)]) == [-r]
      @test f(T[r]) === zero(T)
      @test f(T[nextfloat(r)]) === T(Inf)
      @test isequal(prox(f, T[-0.0]), T[-0.0])
    end
    x = view(T[3, 99, -4, 99], 1:2:4)
    out = zeros(T, 4)
    y = view(out, 2:2:4)
    @test prox!(y, IndBallL2(T(2.5)), x) === y
    @test out == T[0, 1.5, 0, -2]
    @test prox(IndBallL2(T(2.5)), Symmetric(T[3 0; 0 -4])) == T[1.5 0; 0 -2]
  end
end

@testset "IndBallL2 stored feasibility for four dense coefficients" begin
  for (T, a) in ((Float32, 1), (Float64, 3))
    x = T[a, 11, 0, 0]
    f = IndBallL2(one(T))
    for (input, r) in ((x, one(T)), (-T(8) .* x, T(0.125)))
      g = IndBallL2(r)
      saved = copy(input)
      expected = prox(g, input)
      setprecision(BigFloat, 256) do
        w = BigFloat.(input)
        ref = w .* (BigFloat(r) / sqrt(sum(abs2, w)))
        @test sum(abs2, BigFloat.(expected)) <= BigFloat(r)^2
        @test all(isapprox(expected[i], T(ref[i]); rtol = 8eps(T)) for i in eachindex(ref))
      end
      @test g(expected) === zero(T)
      @test signbit(expected[4]) == signbit(input[4])
      @test Scaled(g, T(0.5)) === g
      for h in (g, Scaled(g, T(0.5))), γ in (zero(T), one(T), T(Inf)), alias in (false, true)
        source = copy(input)
        y = alias ? source : similar(input)
        @test prox!(y, h, source, γ) === y
        @test isequal(y, expected)
        @test alias || isequal(source, saved)
        copyto!(source, saved)
        value = value_and_prox!(y, h, source, γ)
        @test value === h(y) === zero(T)
        @test isequal(y, expected)
        @test alias || isequal(source, saved)
      end
      @test isequal(input, saved)
    end
    for b in (T(4), T(-4))
      control = T[3, b, 0, -0.0]
      for (r, expected) in ((T(2.5), control .* T(0.5)), (T(5), control), (T(6), control))
        y = copy(control)
        g = IndBallL2(r)
        value = value_and_prox!(y, g, y)
        @test value === g(y) === zero(T)
        @test isequal(y, expected)
        @test isequal(prox(g, control), expected)
      end
    end
    @test f(T[0, nextfloat(one(T)), 0, 0]) === T(Inf)
    for γ in (T(-1), T(NaN))
      @test_throws ArgumentError prox(f, x, γ)
    end
    for op in (prox!, value_and_prox!), γ in (T(-1), T(NaN))
      y = copy(x)
      @test_throws ArgumentError op(y, f, y, γ)
      @test isequal(y, x)
    end
    for n in (2, 3, 5, 32)
      input = zeros(T, n)
      input[1], input[2] = T(a), T(11)
      expected = input .* (one(T) / sqrt(sum(abs2, input)))
      @test isequal(prox(f, input), expected)
      for alias in (false, true)
        source = copy(input)
        y = alias ? source : similar(input)
        value = value_and_prox!(y, f, source, one(T))
        @test isequal(y, expected)
        @test value === f(y)
      end
    end
  end
end

@testset "IndBallL2 stable finite range and stored value" begin
  for T in (Float32, Float64)
    δ = nextfloat(zero(T))
    M = floatmax(T)
    @test IndBallL2(M)(fill(M, 2)) === T(Inf)
    @test IndBallL2(δ)(fill(δ, 2)) === T(Inf)
    @test IndBallL2(zero(T))(T[δ, 0]) === T(Inf)
    for (x, r) in ((T[M, -M], T(1)), (T[M, -M/2], 2δ),
                   (T[4δ, -3δ], 2δ), (T[M, -M/2, -0.0], δ),
                   (T[floatmin(T), -2floatmin(T)], floatmin(T)))
      f = IndBallL2(r)
      ref = setprecision(BigFloat, 256) do
        w = BigFloat.(x)
        T.((BigFloat(r) ./ sqrt(sum(abs2, w))) .* w)
      end
      y = similar(x)
      @test isequal(value_and_prox!(y, f, x, T(Inf)), f(y))
      if r < floatmin(T)
        @test isequal(y, ref)
      else
        @test all(i -> isapprox(y[i], ref[i]; rtol = 8eps(T)), eachindex(y))
      end
    end
    x = fill(M, 2)
    f = IndBallL2(δ)
    @test Scaled(f, T(0.5)) === f && Scaled(Scaled(f, δ), δ) === f
    for h in (f, Scaled(f, T(0.5)), Scaled(f, T(Inf)),
              Scaled(Scaled(f, δ), δ))
      y = similar(x)
      @test value_and_prox!(y, h, x, zero(T)) === T(Inf)
      @test y == fill(δ, 2)
      @test h(y) === T(Inf)
    end
  end
end

@testset "IndBallL2 nonfinite endpoints" begin
  for T in (Float32, Float64), γ in (zero(T), one(T), T(Inf))
    z, ball, full = IndBallL2(zero(T)), IndBallL2(T(2)), IndBallL2(T(Inf))
    for x in (T[NaN, Inf], T[Inf, -Inf], T[3, -Inf], T[0, -0.0])
      @test isequal(prox(z, x, γ), zeros(T, 2))
      @test isequal(prox(full, x, γ), x)
      @test isequal(full(x), any(isnan, x) ? T(NaN) : zero(T))
      @test isequal(z(x), any(isnan, x) ? T(NaN) : any(!iszero, x) ? T(Inf) : zero(T))
      y = copy(x)
      @test isequal(value_and_prox!(y, full, y, γ), full(y))
    end
    @test isequal(prox(ball, T[3, -Inf], γ), T[0, -2])
    @test isequal(prox(ball, T[Inf, -3], γ), T[2, 0])
    @test ball(T[3, -Inf]) === T(Inf)
    @test isnan(ball(T[Inf, NaN]))
    for x in (T[Inf, -Inf], T[Inf, NaN], T[NaN, 1])
      @test isnan(value_and_prox!(x, ball, x, γ))
      @test all(isnan, x)
    end
    y = fill(T(9), 2)
    @test value_and_prox!(y, Scaled(ball, T(Inf)), T[1, 0], γ) === zero(T)
  end
end

@testset "IndBallL2 supported iterates and preflight" begin
  for T in (Float32, Float64)
    f = IndBallL2(T(2.5))
    for dense in (T[3, 0, -4, 0], T[3 -4; 0 0])
      x = SparseIterate(dense)
      y = SparseIterate(fill(T(7), size(dense)))
      expected = dense .* T(0.5)
      @test f(x) === T(Inf)
      @test Array(prox(f, x, zero(T))) == expected
      @test value_and_prox!(y, f, x, zero(T)) === zero(T)
      @test Array(y) == expected
      @test Array(prox!(x, f, x)) == expected
      @test Array(prox!(similar(dense), f, y)) == expected
    end
    x = SparseIterate(T[3, 0, -4, 0])
    activate!(x, 4)
    @test prox!(x, f, x) === x
    @test isactive(x, 4)
    x[3] = T(-Inf)
    @test value_and_prox!(x, f, x) === zero(T)
    @test Array(x) == T[0, 0, -2.5, 0]
    x[3] = T(NaN)
    @test isnan(value_and_prox!(x, f, x))
    @test all(isnan, x) && nactive(x) == length(x)

    x = BlockIterate(T[3, 0, -4, 0], [[3, 1], [4, 2]])
    y = similar(x)
    @test value_and_prox!(y, f, x, zero(T)) === zero(T)
    @test collect(y) == T[1.5, 0, -2, 0]
    @test nactive(y) == 1
    @test prox!(x, f, x) === x
    @test collect(x) == collect(y)

    y = fill(T(9), 2)
    x = T[3, -4]
    for op in (prox!, value_and_prox!), h in (f, Scaled(f, nextfloat(zero(T))))
      for γ in (T(-1), -nextfloat(zero(T)), T(NaN))
        @test_throws ArgumentError op(y, h, x, γ)
        @test y == T[9, 9]
        @test_throws ArgumentError prox(h, x, γ)
      end
      @test_throws DimensionMismatch op(y, h, reshape(x, 1, 2), zero(T))
      @test_throws ArgumentError op(y, h, OffsetArray(x, 0:1), zero(T))
      @test_throws ArgumentError op(y, h, complex.(x), zero(T))
      @test y == T[9, 9]
    end

    sx = SymmetricSparseIterate(T[3 0; 0 -4])
    sy = SymmetricSparseIterate(T[8 1; 1 8])
    for op in (prox!, value_and_prox!), h in (f, Scaled(f, T(0.5)))
      dense = fill(T(9), 2, 2)
      @test_throws ArgumentError op(dense, h, sx, zero(T))
      @test_throws ArgumentError op(sy, h, dense, zero(T))
      @test dense == fill(T(9), 2, 2)
      @test Matrix(sy) == T[8 1; 1 8]
      @test_throws ArgumentError prox(h, sx, zero(T))
      @test_throws ArgumentError h(sx)
    end
  end
end

@testset "IndBallL2 alias and storage edge cases" begin
  for T in (Float32, Float64)
    δ = nextfloat(zero(T))
    f = IndBallL2(one(T))
    inner = Scaled(f, T(0.5))
    nested = Scaled{typeof(inner), T}(inner, T(0.5))
    @test inner === f && nested === f
    x = T[3, 4]
    y = fill(T(9), 2)
    @test_throws ArgumentError prox(nested, x, -δ)
    for op in (prox!, value_and_prox!)
      @test_throws ArgumentError op(y, nested, x, -δ)
      @test y == fill(T(9), 2)
    end
    @test prox(nested, x, zero(T)) == prox(f, x, zero(T))

    z = BlockIterate(T[-0.0, 0, -0.0], 3)
    @test nactive(z) == 0
    @test isequal(collect(prox(IndBallL2(T(Inf)), z)), collect(z))
    @test isequal(collect(prox(f, z)), collect(z))
    @test prox!(z, IndBallL2(zero(T)), z) === z
    @test isequal(collect(z), zeros(T, 3)) && nactive(z) == 0

    for a in (T(4), floatmax(T))
      z = view(T[a, 9], [1, 1])
      expected = prox(f, z)
      @test prox!(z, f, z) === z
      @test collect(z) == expected
      @test parent(z)[2] == T(9)
    end

    x = SparseIterate(T[floatmax(T), -floatmax(T)/2])
    f = IndBallL2(δ)
    z = prox(f, x)
    @test isequal(Array(z), T[δ, -0.0])
    @test isactive(z, 2)
    @test isequal(Array(prox!(x, f, x)), Array(z))

    for (x, f) in ((T[floatmax(T), -floatmax(T)/2], IndBallL2(δ)),
                   (T[3, -δ], IndBallL2(one(T))),
                   (T[0, -0.0], IndBallL2(T(Inf))))
      z = SparseIterate(T, 2)
      @test isequal(value_and_prox!(z, f, x), f(z))
      @test isequal(Array(z), prox(f, x))
      @test isactive(z, 2)
    end

    for x in (SparseVector(4, [2, 3], T[-0.0, 3]),
              SparseMatrixCSC(4, 1, [1, 3], [2, 3], T[-0.0, 3])), r in (T(Inf), one(T))
      z = SparseIterate(T, size(x)...)
      z[1] = T(7)
      @test prox!(z, IndBallL2(r), x) === z
      @test isequal(z[1], zero(T))
      @test isequal(z[2], T(-0.0)) && isactive(z, 2)
      @test z[3] == min(T(3), r)
    end
    x = spzeros(T, 0, 128)
    z = SparseIterate(T, size(x)...)
    @test prox!(z, IndBallL2(T(Inf)), x) === z
    @test value_and_prox!(z, IndBallL2(one(T)), x) === zero(T)
  end
end

end
