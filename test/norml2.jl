module NormL2Test
# NormL2 (#900 amendment): values on arrays and the three iterates (#808: stored entries only, symmetric
# off-diagonal coordinates twice), proximal operators on arrays, the one-group equivalence with
# GroupNormL2 (rulings D1–D3 and NR1), and the unsupported symmetric-iterate proximal operators (NR2),
# against independent BigFloat evaluations of λ‖x‖₂ and max(1 - γλ/‖x‖₂, 0) x in Float32 and Float64.
using Test, ProximalBase, LinearAlgebra, Random
using ..TestUtils

const PREC = 512
bignorm(u) = setprecision(() -> sqrt(sum(big.(u) .^ 2; init = big(0))), BigFloat, PREC)
function refprox(u, γ, λ)
  setprecision(BigFloat, PREC) do
    U = big.(u)
    n = bignorm(u)
    c = big(γ) * big(λ)
    n > c ? (1 - c / n) .* U : zero.(U)
  end
end
refvalue(u, λ) = setprecision(() -> big(λ) * bignorm(u), BigFloat, PREC)
function close_prox(y, ref, u; k = 8)
  T = eltype(y)
  m = big(maximum(abs, u; init = zero(T)))
  bound = k * big(eps(T)) * m + 2 * big(nextfloat(zero(T)))
  all(i -> abs(big(y[i]) - ref[i]) <= bound, eachindex(y))
end
function close_value(v, ref; k = 8)
  T = typeof(v)
  abs(ref) > big(floatmax(T)) * (1 + big(eps(T))) && return v == copysign(T(Inf), ref)
  abs(big(v) - ref) <= k * big(eps(T)) * abs(ref) + 2 * big(nextfloat(zero(T)))
end
# equal, or finite and within 4 eps: exceptional and zero values must agree exactly
same_value(a, b) = isequal(a, b) || (isfinite(a) && isfinite(b) && !iszero(a) && abs(a - b) <= 4eps(typeof(a)) * max(abs(a), abs(b)))
# the matrix a symmetric iterate represents, for the references
symmetric_dense(S) = Matrix(S)

@testset "construction and traits" begin
  @test NormL2().λ === 1 && NormL2(0.5f0).λ === 0.5f0
  @test_throws ArgumentError NormL2(-1.0)
  @test_throws ArgumentError NormL2(NaN)
  f = NormL2(2.0)
  @test isproximable(f) && isconvex(f) && !issmooth(f) && !isseparable(f) && !iszeropreserving(f)
end

@testset "analytical values and shrinkage ($T)" for T in (Float32, Float64)
  f = NormL2(one(T))
  @test f(T[3, 4]) == 5 && f(T[3 0; 0 4]) == 5
  @test prox(f, T[3, 4], one(T)) ≈ T[2.4, 3.2]
  @test prox(f, T[3, 4], T(5)) == T[0, 0]                       # norm exactly at the threshold
  @test prox(f, T[3 4; 0 0], T(2.5)) == T[1.5 2; 0 0]           # Frobenius norm of a matrix
  @test prox(f, zeros(T, 3), one(T)) == zeros(T, 3)
  rng = Xoshiro(910)
  for λ in (T(0.3), T(2)), γ in (T(0.5), T(4)), n in (1, 7, 100)
    x = T(2) .* randn(rng, T, n)
    y = prox(NormL2(λ), x, γ)
    @test close_prox(y, refprox(x, γ, λ), x)
    @test close_value(NormL2(λ)(x), refvalue(x, λ))
    z = similar(x)
    v = value_and_prox!(z, NormL2(λ), x, γ)
    @test z == y && close_value(v, refvalue(z, λ)) && isapprox(v, NormL2(λ)(z); rtol = 4eps(T))
  end
end

@testset "one group of GroupNormL2 ($T)" for T in (Float32, Float64)
  # NormL2(λ) and GroupNormL2(λ, [1:p], [1]) are the same function, with the same rules (NR1)
  rng = Xoshiro(911)
  emax, emin = exponent(floatmax(T)), exponent(floatmin(T))
  s(e) = T(2)^e
  big_ = floatmax(T) / 4
  inputs = [T(2) .* randn(rng, T, 6), zeros(T, 6), T[3, 4, 0, 0, 0, 0], T[7],
            s(emax - 1) .* T[1, 0.75, -0.5, 0, 0, 0], s(emin - 5) .* T[1, 0.75, -0.5, 0, 0, 0],
            T[NaN, 1, 0, 0, 0, 0], T[Inf, 1, 0, 0, 0, 0], T[-Inf, Inf, 1, 0, 0, 0]]
  params = [(T(0.6), T(0.9)), (T(5), one(T)),                            # ordinary, at the threshold
            (T(Inf), zero(T)), (T(Inf), one(T)),                         # D1 (#665)
            (zero(T), one(T)), (one(T), zero(T)), (one(T), T(Inf)),
            (big_, T(8)), (T(8), big_),                                  # D3: finite γλ overflows
            (s(emin), s(-2)), (s(emax - 2), s(emin + 1))]
  for x in inputs, (λ, γ) in params
    n2 = NormL2(λ)
    g1 = GroupNormL2(λ, [1:length(x)], T[1])
    @test isequal(prox(n2, x, γ), prox(g1, x, γ))
    @test isequal(prox(n2, x, γ), prox_block(g1, x, 1, γ))
    y1, y2 = similar(x), similar(x)
    v1, v2 = value_and_prox!(y1, n2, x, γ), value_and_prox!(y2, g1, x, γ)
    @test isequal(y1, y2)
    # the values may sum the squares in another order (Base's sum for NormL2)
    @test same_value(v1, v2) && same_value(n2(x), g1(x))
  end
  # D1: an infinite level projects to +0 at every nonnegative step, γ = 0 included, for every input
  for γ in (zero(T), one(T)), x in (T[3, 4], T[NaN, 1], T[-Inf, 2])
    @test all(isequal(zero(T)), prox(NormL2(T(Inf)), x, γ))
  end
  # D3: ±Inf with an overflowing finite threshold leaves the argument unchanged
  @test prox(NormL2(big_), T[Inf, 1], T(8)) == T[Inf, 1]
  @test prox(NormL2(T(8)), T[1, -Inf], big_) == T[1, -Inf]
  @test all(isnan, prox(NormL2(one(T)), T[NaN, 1], one(T)))
  @test isequal(prox(NormL2(one(T)), T[NaN, 1], zero(T)), T[NaN, 1])
  @test all(isequal(zero(T)), prox(NormL2(one(T)), T[Inf, 1], T(Inf)))
  # a representable result whose threshold and norm both overflow
  u = s(emax) .* T[1.5, 1.5]
  @test close_prox(prox(NormL2(floatmax(T) / 2), u, T(3)), refprox(u, T(3), floatmax(T) / 2), u)
end

@testset "invalid steps and shapes fail before writes ($T)" for T in (Float32, Float64)
  f = NormL2(T(0.5))
  x = T[1, -2, 3]
  for γ in (-one(T), T(NaN), -T(Inf))      # D2
    y = fill(T(7), 3)
    @test_throws ArgumentError prox!(y, f, x, γ)
    @test_throws ArgumentError value_and_prox!(y, f, x, γ)
    @test y == fill(T(7), 3)
    z = copy(x)
    @test_throws ArgumentError prox!(z, f, z, γ)
    @test z == x
    @test_throws ArgumentError prox(f, x, γ)
  end
  y = fill(T(7), 2)
  @test_throws DimensionMismatch prox!(y, f, x, one(T))
  @test_throws ArgumentError prox!(zeros(Complex{T}, 3), f, x, one(T))
  @test y == fill(T(7), 2)
  z = copy(x)
  @test prox!(z, f, z, one(T)) === z && z == prox(f, x, one(T))
  @test allocs(prox!, y, f, T[3, 4], one(T)) == 0
  out = zeros(T, 2)
  @test allocs(value_and_prox!, out, f, T[3, 4], one(T)) == 0
  @test allocs(f, x) == 0
end

@testset "values on the iterates (#808) ($T)" for T in (Float32, Float64)
  emax, emin = exponent(floatmax(T)), exponent(floatmin(T))
  s(e) = T(2)^e
  # stored entries: huge, subnormal, ordinary, with stored signed zeros
  for data in ([(1, 1, 3.0), (2, 1, -4.0), (3, 3, 0.0)],
               [(1, 1, s(emax - 1)), (2, 1, s(emax - 1)), (4, 3, -0.0)],
               [(1, 1, s(emin - 10)), (2, 1, -s(emin - 12)), (3, 2, 0.0)],
               Tuple{Int, Int, Float64}[])
    for λ in (one(T), T(0.25), s(-emax ÷ 2), s(emax ÷ 2), zero(T))
      f = NormL2(λ)
      S = SymmetricSparseIterate(T, 5)
      M = SparseIterate(T, 5, 5)
      for (r, c, v) in data
        S[r, c] = one(T); S[r, c] = T(v)          # keeps a stored zero
        M[r, c] = one(T); M[r, c] = T(v)
      end
      @test nactive(S) == length(data)
      @test close_value(f(S), refvalue(symmetric_dense(S), λ))   # off-diagonal entries twice
      @test close_value(f(M), refvalue(Array(M), λ))
      @test f(S) == f(symmetric_dense(S)) || close_value(f(S), refvalue(symmetric_dense(S), λ); k = 2)
      @test f(M) == f(Array(M)) || close_value(f(M), refvalue(Array(M), λ); k = 2)
      @test allocs(f, S) == 0 && allocs(f, M) == 0
      v = SparseIterate(vec(Array(M)))
      @test close_value(f(v), refvalue(vec(Array(M)), λ))
      B = BlockIterate(vec(Array(M)), 5)
      @test f(B) == f(vec(Array(M)))
    end
  end
  # exceptional values
  S = SymmetricSparseIterate(T, 3); S[2, 1] = T(NaN)
  @test isnan(NormL2(one(T))(S)) && isnan(NormL2(T(Inf))(S)) && NormL2(zero(T))(S) === zero(T)
  S = SymmetricSparseIterate(T, 3); S[2, 1] = T(Inf)
  @test NormL2(one(T))(S) == Inf && NormL2(T(Inf))(S) == Inf
  @test NormL2(T(Inf))(SymmetricSparseIterate(T, 3)) === zero(T)
  @test NormL2(T(Inf))(SparseIterate(T, 4)) === zero(T)
  @test NormL2(T(Inf))(T[0, -0.0]) === zero(T)
  @test NormL2(T(Inf))(T[0, 1]) == Inf
  @test NormL2(T(Inf))(T[]) === zero(T)
end

@testset "proximal operators on iterates (NR2) ($T)" for T in (Float32, Float64)
  f = NormL2(T(0.7))
  d = T[0, 3, 0, -4, 0, 0]
  # BlockIterate and SparseIterate through their entries, as arrays
  B = BlockIterate(copy(d), 3)
  r = similar(B)
  @test prox!(r, f, B, one(T)) === r && Vector(r) == prox(f, d, one(T))
  @test value_and_prox!(similar(B), f, B, one(T)) == value_and_prox!(similar(d), f, d, one(T))
  x = SparseIterate(copy(d))
  y = SparseIterate(T, 6)
  @test Vector(prox!(y, f, x, one(T))) == prox(f, d, one(T)) && nactive(y) == 2
  # no proximal operator for a symmetric iterate: a clear error, nothing written
  S = SymmetricSparseIterate(T[1 2; 2 1])
  S2 = SymmetricSparseIterate(T[5 6; 6 7])
  @test_throws ArgumentError prox!(S2, f, S, one(T))
  @test_throws ArgumentError value_and_prox!(S2, f, S, one(T))
  @test_throws ArgumentError prox!(S2, f, T[1 2; 2 1], one(T))
  @test Matrix(S2) == T[5 6; 6 7]
  D = fill(T(7), 2, 2)
  @test_throws ArgumentError prox!(D, f, S, one(T))
  @test_throws ArgumentError value_and_prox!(D, f, S, one(T))
  @test D == fill(T(7), 2, 2)
  @test_throws ArgumentError prox(f, S, one(T))
  # a positive finite scale forwards the step
  @test prox(2 * f, d, T(0.5)) == prox(f, d, one(T))
  @test (2 * f)(d) ≈ 2 * f(d)
end

end
