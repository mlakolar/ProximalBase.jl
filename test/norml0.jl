module NormL0Test

using Test, ProximalBase, OffsetArrays, Random

function reference_prox(v::T, λ, γ) where {T}
  iszero(λ) && return v
  (isinf(λ) || isinf(γ)) && return zero(T)
  (iszero(γ) || !isfinite(v)) && return v
  setprecision(BigFloat, 256) do
    zero_cost = abs2(BigFloat(v)) / 2
    keep_cost = BigFloat(γ) * BigFloat(λ)
    keep_cost < zero_cost ? v : zero(T)
  end
end

function reference_value(x, λ::T) where {T}
  iszero(λ) && return zero(T)
  any(isnan, x) && return T(NaN)
  n = count(!iszero, x)
  iszero(n) ? zero(T) : n * λ
end

@testset "NormL0 constructor and traits" begin
  f = NormL0()
  @test f.λ == 1
  @test_throws ArgumentError NormL0(-1)
  @test_throws ArgumentError NormL0(NaN)
  @test_throws ArgumentError NormL0(-Inf)
  @test isseparable(f) && isproximable(f) && iszeropreserving(f)
  @test !isconvex(f) && !issmooth(f)
  @test restrict(f, [2, 1]) === f
end

@testset "NormL0, $T" for T in (Float32, Float64)
  z, u = zero(T), one(T)
  @testset "exact objectives" begin
    edge = T[nextfloat(z), floatmin(T), T(0.5), u, T(2), floatmax(T)]
    for λ in edge, γ in edge, v in (edge..., -edge..., z, -z)
      @test prox_block(NormL0(λ), v, 1, γ) === reference_prox(v, λ, γ)
    end

    exponents = T === Float32 ? (-100, -70, 0, 70, 120) : (-1000, -600, 0, 600, 1000)
    for e in exponents
      for m in T[1, 1.5, 1.2345678]
        v = ldexp(m, e)
        γ, λ = v, v / 2
        f = NormL0(λ)
        for sign in (-u, u)
          @test prox(f, [sign * v], γ) == T[0]
          @test prox(f, [sign * prevfloat(v)], γ) == T[0]
          @test prox(f, [sign * nextfloat(v)], γ) == [sign * nextfloat(v)]
        end
      end
      # The rounded products agree in the ordinary case at e = 0.
      a = ldexp(u, e)
      v, λ = prevfloat(a), a / 2
      γ = prevfloat(v)
      @test reference_prox(v, λ, γ) === v
      @test prox(NormL0(λ), T[v, -v], γ) == T[v, -v]
    end

    # Both sides of near ties at ordinary, overflow and underflow scales.
    rng = MersenneTwister(951)
    for e in exponents, _ in 1:120
      v = ldexp(u + rand(rng, T), e)
      γ = ldexp(u + rand(rng, T), e + rand(rng, -2:2))
      λ = setprecision(BigFloat, 256) do
        T(abs2(BigFloat(v)) / (2 * BigFloat(γ)))
      end
      v = rand(rng, Bool) ? v : -v
      for level in (prevfloat(λ), λ, nextfloat(λ))
        @test prox_block(NormL0(level), v, 1, γ) === reference_prox(v, level, γ)
      end
    end
  end

  @testset "values, endpoints and fused arrays" begin
    x = T[NaN, -Inf, Inf, -2, 0.5, -0.0, 0]
    for λ in (z, u, T(Inf)), γ in (z, u, T(Inf))
      f = NormL0(λ)
      expected = reference_prox.(x, λ, γ)
      @test f(T[]) === z
      @test f(x) === reference_value(x, λ)
      @test f(x[2:end]) === reference_value(x[2:end], λ)
      @test value_entry(f, -z, 1) === z
      @test value_entry(f, x[1], 1) === reference_value(x[1:1], λ)
      @test isequal(map(v -> prox_block(f, v, 1, γ), x), expected)
      @test isequal(prox(f, x, γ), expected)
      @test isequal(prox(f, x, fill(γ, length(x))), expected)
      y = fill(T(7), length(x))
      @test prox!(y, f, x, γ) === y
      @test isequal(y, expected)
      for step in (γ, fill(γ, length(x)))
        y .= x
        @test value_and_prox!(y, f, y, step) === reference_value(expected, λ)
        @test isequal(y, expected)
      end
    end
    f = NormL0(T(0.5))
    @test NormL0(T(0.1))(ones(T, 23)) ≈ T(2.3)
    xmat = T[2 -3; 3 -3]
    steps = T[1, 9, 16, 0]
    @test prox(f, xmat, steps) == T[2 0; 0 -3]
    parent = T[7, 7]
    repeated = view(parent, [1, 1, 2])
    @test value_and_prox!(repeated, f, T[2, 0.25, 3], u) === T(0.5)
    @test parent == T[0, 3]
    narrowed = zeros(Float32, 2)
    @test value_and_prox!(narrowed, f, [nextfloat(0.0), 2.0], z) === 0.5f0
    @test narrowed == Float32[0, 2]
  end

  @testset "prepared scalar step" begin
    for (λ, γ) in ((T(0.5), u), (T(0.1), u), (T(0.3), u), (T(0.5), prevfloat(prevfloat(u))),
                   (floatmax(T), floatmax(T)), (nextfloat(z), floatmin(T)),
                   (z, T(Inf)), (T(Inf), z), (u, T(Inf)))
      f = NormL0(λ)
      root = sqrt((2γ) * λ)
      # Special-value and near-threshold entries mixed with both signs.
      v = isfinite(root) ? root : u
      cases = T[NaN, Inf, -Inf, -z, z, floatmax(T), nextfloat(z),
                v, -v, prevfloat(v), -prevfloat(v), nextfloat(v), -nextfloat(v)]
      x = repeat(cases, 16)
      want = reference_prox.(x, λ, γ)
      @test isequal(prox(f, x, γ), want)
      y = copy(x)
      @test value_and_prox!(y, f, y, γ) === f(want)
      @test isequal(y, want)
      sparse = SparseIterate(x)
      sparse_want = reference_prox.(Array(sparse), λ, γ)
      @test value_and_prox!(sparse, f, sparse, γ) === f(sparse_want)
      @test isequal(Array(sparse), sparse_want)
    end
    f = NormL0(T(0.5))
    parent = T[7, 7]
    y = view(parent, [ones(Int, 64); 2])
    x = [fill(T(2), 63); T(0.25); T(3)]
    @test value_and_prox!(y, f, x, u) === T(0.5)
    @test parent == T[0, 3]
  end

  @testset "iterate cardinality and stored values" begin
    f = NormL0(T(0.5))
    sp = SparseIterate(T, 6)
    sp[2], sp[3], sp[5] = T(-2), T(0.5), T(3)
    sp[5] = z
    blocks = BlockIterate(T[0, -2, 0.5, 0, 0, 0], [[1, 2], [3, 4], [5, 6]])
    activate!(blocks, 3)
    symmetric = SymmetricSparseIterate(T, 3)
    symmetric[1, 1], symmetric[2, 1], symmetric[3, 1], symmetric[3, 3] =
      T(0.5), T(-2), T(7), T(3)
    symmetric[3, 1] = z
    @test ProximalBase.check_symmetric_parameters(f, symmetric) === true
    @test f(sp) === f(blocks) === u
    @test f(symmetric) === T(2)
    for x in (sp, blocks, symmetric)
      dense = x isa SymmetricSparseIterate ? Matrix : Array
      expected = reference_prox.(dense(x), f.λ, u)
      allocated = prox(f, x, u)
      @test allocated isa typeof(x)
      @test dense(allocated) == expected
      @test f(x) === reference_value(dense(x), f.λ)
      y = copy(x)
      if y isa SymmetricSparseIterate
        y[2, 2] = T(7)
      else
        y[1] = T(7)
      end
      @test value_and_prox!(y, f, x, u) === reference_value(expected, f.λ)
      @test dense(y) == expected
      @test value_and_prox!(x, f, x, u) === reference_value(expected, f.λ)
      @test dense(x) == expected
    end
    @test f(symmetric) === T(1.5)
    y = similar(symmetric)
    a = T[0 -2; 6 0]
    small = SymmetricSparseIterate(T, 2)
    @test value_and_prox!(small, f, a, u) === u
    @test Matrix(small) == T[0 2; 2 0]
    @test Matrix(prox!(y, f, symmetric, u)) == Matrix(symmetric)
  end

  @testset "pre-write validation" begin
    x = T[2, 0.5, -3]
    s = SymmetricSparseIterate(T[2 0; 0 1])
    for f in (NormL0(z), NormL0(u), NormL0(T(Inf))), oracle in (prox!, value_and_prox!)
      for bad in (-u, T(NaN))
        for point in (x, SparseIterate(x), BlockIterate(x, [[1, 2], [3]]), s)
          y = copy(point)
          dense = y isa SymmetricSparseIterate ? Matrix : Array
          ids = y isa Vector ? nothing : collect(activecoordinates(y))
          @test_throws ArgumentError oracle(y, f, point, bad)
          @test dense(y) == dense(point)
          if ids !== nothing
            @test collect(activecoordinates(y)) == ids
          end
        end
        y = copy(x)
        @test_throws ArgumentError oracle(y, f, x, T[u, u, bad])
        @test y == x
        # Invalid values in inactive coordinates are still checked.
        sparse = SparseIterate(T, 3)
        sparse[1] = T(2)
        out = copy(sparse)
        @test_throws ArgumentError oracle(out, f, sparse, T[u, u, bad])
        @test Array(out) == Array(sparse)
      end
      y = copy(x)
      @test_throws DimensionMismatch oracle(y, f, x, T[u, u])
      @test_throws DimensionMismatch oracle(y, f, x[1:2], u)
      @test_throws ArgumentError oracle(y, f, x, OffsetArray(fill(u, 3), 0:2))
      @test_throws ArgumentError oracle(y, f, x, complex.(fill(u, 3)))
      @test y == x
      sy = copy(s)
      @test_throws ArgumentError oracle(sy, f, s, ones(T, 2, 2))
      @test Matrix(sy) == Matrix(s)
    end
    @test_throws ArgumentError prox(NormL0(u), x, T(NaN))
    @test_throws ArgumentError prox(NormL0(u), s, ones(T, 2, 2))
  end
end

end
