module IndBallLinfTest

using Test, ProximalBase

# Independent magnitude and clipping references, with either sign of zero feasible.
refvalue(x, r) = any(isnan, x) ? eltype(x)(NaN) : any(v -> abs(v) > r, x) ? eltype(x)(Inf) : zero(eltype(x))
refproject(x, r) = map(v -> isnan(v) ? v : v > r ? oftype(v, r) : v < -r ? oftype(v, -r) : v, x)
same(a, b) = size(a) == size(b) && all(i -> isnan(a[i]) ? isnan(b[i]) : a[i] == b[i], eachindex(a))

@testset "IndBallLinf constructor" begin
  for r in (-1, -0.5f0, NaN, -Inf, NaN32)
    @test_throws ArgumentError IndBallLinf(r)
  end
  for r in (UInt(1), 0x80)
    @test_throws ArgumentError IndBallLinf(r)
  end
  @test_throws MethodError IndBallLinf()
  @test IndBallLinf(2) === IndBox(-2, 2)
  @test IndBallLinf(0x00) === IndBox(0x00, 0x00)
  @test IndBallLinf(Inf).l === -Inf && IndBallLinf(Inf).u === Inf
  @test prox(IndBallLinf(2), [-3, 1, 5]) == [-2, 1, 2]
  @test prox(IndBallLinf(0x00), [-3.0, 0.0]) == [0.0, 0.0]
end

@testset "IndBallLinf separable interface" begin
  for T in (Float32, Float64)
    f = IndBallLinf(T(1.5))
    @test f === IndBox(T(-1.5), T(1.5))
    @test isconvex(f) && isseparable(f) && isproximable(f) && iszeropreserving(f)
    @test value_entry(f, T(1.5), 7) === zero(T)
    @test value_entry(f, T(2), 7) === T(Inf)
    @test isnan(value_entry(f, T(NaN), 7))
    @test prox_block(f, T(-3), 7, T(NaN)) === T(-1.5)
    @test restrict(f, [3, 1]) === f
    @test Scaled(f, zero(T)) === f
    g = Scaled(f, T(2))
    x = T[2, 1, -3]
    @test g(x) === T(Inf)
    for γ in T[0, 1, Inf, -1, NaN]
      y = copy(x)
      @test value_and_prox!(y, g, y, γ) === zero(T)
      @test y == T[1.5, 1, -1.5]
    end
  end
end

@testset "IndBallLinf value and projection" begin
  for T in (Float32, Float64)
    values = T[-Inf, -floatmax(T), -2, -1, -0.5, -0.0, 0, 0.5, 1, 2, floatmax(T), Inf, NaN]
    for r in (zero(T), nextfloat(zero(T)), one(T), T(2), floatmax(T), T(Inf))
      f = IndBallLinf(r)
      for v in values
        @test isequal(f(T[v]), refvalue(T[v], r))
        @test same(prox(f, T[v], zero(T)), refproject(T[v], r))
      end
      for γ in (zero(T), T(0.7), T(Inf), fill(zero(T), length(values)))
        y = copy(values)
        @test isequal(value_and_prox!(y, f, y, γ), f(y))
        @test same(y, refproject(values, r))
      end
      @test f(T[]) === zero(T) && f(zeros(T, 0, 3)) === zero(T)
      @test isempty(prox(f, T[]))
    end
    f = IndBallLinf(one(T))
    @test f(T[-1, 1, 0]) === zero(T) && f(T[-1, nextfloat(one(T))]) === T(Inf)
    @test isnan(f(T[2, NaN]))
    long = zeros(T, 600)
    @test f(long) === zero(T)
    long[300] = -nextfloat(one(T))
    @test f(long) === T(Inf)
    X = reshape(T[-2, 0.5, 3, -0.25, 1, -1], 2, 3)
    for A in (X, view(X, :, 2:3), view(vec(X), 1:2:6))
      @test f(A) === refvalue(A, one(T))
      @test same(prox(f, A), refproject(A, one(T)))
      out = zeros(T, size(A))
      @test value_and_prox!(out, f, A, T(Inf)) === f(out)
      @test f(out) === zero(T)
      @test same(out, refproject(A, one(T)))
    end
    x = T[4, -5, 0.5]
    for op in (prox!, value_and_prox!)
      y = fill(T(9), 3)
      for γ in T[-1, -Inf, NaN]
        @test_throws ArgumentError prox(f, x, γ)
        @test_throws ArgumentError op(y, f, x, γ)
        @test y == fill(T(9), 3)
      end
      for γ in (zero(T), T(3), T(Inf))
        op(y, f, x, γ)
        @test y == T[1, -1, 0.5]
      end
      op(y, f, x, T[-1, NaN, 2])
      @test y == T[1, -1, 0.5]
    end
  end
  # Float32 rounds the endpoints away from zero.
  f = IndBallLinf(0.1)
  y = zeros(Float32, 2)
  @test value_and_prox!(y, f, [0.5, -0.5], 0.0) === f(y)
  @test f(y) === Inf32
  @test y == Float32[0.1, -0.1]
end

@testset "IndBallLinf supported iterates" begin
  for T in (Float32, Float64), r in T[0, 1, Inf]
    f = IndBallLinf(r)
    dense = T[3, 0, -4, 0.5, 0, 0]
    s = SparseIterate(dense)
    @test f(s) === refvalue(dense, r)
    @test same(Array(prox(f, s, zero(T))), refproject(dense, r))
    y = SparseIterate(T, length(dense))
    @test value_and_prox!(y, f, s, T(Inf)) === zero(T)
    @test same(Array(y), refproject(dense, r))
    @test prox!(s, f, s) === s && same(Array(s), refproject(dense, r))

    # An explicitly stored zero stays stored.
    z = SparseIterate(T, 4)
    z[3] = T(-4)
    activate!(z, 1)
    n = nactive(z)
    @test value_and_prox!(z, f, z, zero(T)) === zero(T)
    @test isactive(z, 1) && nactive(z) == n && z[3] == max(-r, T(-4))

    b = BlockIterate(dense, [[3, 1], [4, 2], [5, 6]])
    @test f(b) === refvalue(dense, r)
    c = similar(b)
    @test value_and_prox!(c, f, b, zero(T)) === zero(T)
    @test same(collect(c), refproject(dense, r))
    @test prox!(b, f, b) === b && same(collect(b), collect(c))

    M = T[3 0.5 0; 0.5 -4 0; 0 0 0]
    sm = SymmetricSparseIterate(M)
    @test f(sm) === refvalue(M, r)
    @test same(Matrix(prox(f, sm, zero(T))), refproject(M, r))
    @test same(prox!(similar(M), f, sm, zero(T)), refproject(M, r))
    k = tril_index(3, 3, 2)
    @test !isactive(sm, k)
    activate!(sm, k)
    n = nactive(sm)
    @test value_and_prox!(sm, f, sm, T(Inf)) === zero(T)
    @test same(Matrix(sm), refproject(M, r)) && isactive(sm, k) && nactive(sm) == n
    @test_throws ArgumentError prox!(sm, f, sm, zeros(T, size(M)))
  end
  x = SparseIterate(Float64[0.0, 5.0])
  @test_throws ArgumentError prox!(x, IndBallLinf(1.0), x, -1.0)
  @test x[2] == 5.0
end

end
