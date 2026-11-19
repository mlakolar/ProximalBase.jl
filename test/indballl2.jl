module IndBallL2Test

using Test, ProximalBase, OffsetArrays, LinearAlgebra

@testset "IndBallL2 constructor and tools" begin
  @test IndBallL2().r == 1
  @test_throws ArgumentError IndBallL2(-1)
  @test_throws ArgumentError IndBallL2(NaN)
  @test isconvex(IndBallL2()) && isproximable(IndBallL2())
  @test !isseparable(IndBallL2()) && !iszeropreserving(IndBallL2())
  @test Scaled(IndBallL2(), 0) === IndBallL2()
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
  end
end

end
