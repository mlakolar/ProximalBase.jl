module LinearValueRecoveryTest
using Test, ProximalBase, LinearAlgebra, Random
const PB = ProximalBase

# The value of f = LinearFunction(c, d) when the ordinary dot(c, x) + d is not finite (#903), on a dense vector, a
# dense matrix, a view, a vector SparseIterate and a SymmetricSparseIterate. When d, c and x are finite, the value is
# recomputed from exponent-separated products. The reference is a BigFloat sum computed here, without the package.
# With B = 4 n eps(T) (Σ|cᵢxᵢ| + |d|), n the number of nonzero products (an off-diagonal coordinate of a symmetric
# iterate counts at both positions), a value with |ref| + B <= floatmax(T) is asserted finite and within B of the
# reference (R1); outside that region nothing is asserted about the value (#907). Nonfinite d, c or x give the
# ordinary IEEE value.

pow2(T, e) = ldexp(one(T), e)
emax(T) = exponent(floatmax(T))
esub(T) = exponent(nextfloat(zero(T)))     # the exponent of the smallest subnormal

# Float64 products lie between 2^-2148 and 2^2048: at this precision the products and sums in `reference` and
# `within` are exact
const PREC = 4400

# the cases are symmetric, so that the symmetric route represents the same matrix as the dense ones (#374)
function reference(T, C, X, d)
  setprecision(BigFloat, PREC) do
    terms = BigFloat[big(C[i]) * big(X[i]) for i in eachindex(C, X) if !iszero(C[i]) && !iszero(X[i])]
    r = sum(terms; init = big(0)) + big(d)
    B = 4 * length(terms) * big(eps(T)) * (sum(abs, terms; init = big(0)) + abs(big(d)))
    (value = r, bound = B, asserted = abs(r) + B <= big(floatmax(T)))
  end
end
within(v, ref) = setprecision(() -> isfinite(v) && abs(big(v) - ref.value) <= ref.bound, BigFloat, PREC)

function sparse_vector(v::AbstractVector{T}) where {T}
  x = SparseIterate(T, length(v))
  for i in eachindex(v)
    iszero(v[i]) || (x[i] = v[i])
  end
  x
end
function symmetric_iterate(X::AbstractMatrix{T}) where {T}
  p = size(X, 1)
  x = SymmetricSparseIterate(T, p)
  for j in 1:p, i in j:p
    iszero(X[i, j]) || (x[i, j] = X[i, j])
  end
  x
end

# (route, value, ordinary value) on each route, the ordinary value being the sum that the value starts from
function route_values(C, X, d)
  c, x = vec(C), vec(X)
  xs, xo = sparse_vector(x), symmetric_iterate(X)
  fv, fm = LinearFunction(c, d), LinearFunction(C, d)
  [("dense vector", fv(x), dot(c, x) + d),
   ("dense matrix", fm(X), dot(C, X) + d),
   ("view", fv(view(x, :)), dot(c, view(x, :)) + d),
   ("sparse vector", fv(xs), dot(c, xs) + d),
   ("symmetric", fm(xo), PB._symmetric_matrix_dot(C, xo) + d)]
end

# The R1 assertion on every route; overflow = true also asserts that the ordinary value is not finite there, so that
# the recovery runs. The route name is part of each comparison, so that a failure names it.
function check_case(T, C, X, d; overflow = true)
  @test issymmetric(C) && issymmetric(X)
  ref = reference(T, C, X, d)
  @test ref.asserted
  for (route, v, ordinary) in route_values(C, X, T(d))
    overflow && @test (route, isfinite(ordinary)) == (route, false)
    @test (route, v isa T, within(v, ref)) == (route, true, true)
  end
end

# a value with a random significand in [2^e, 2^(e+1)) (rounded to a subnormal below floatmin) and a random sign
function rand_value(rng, ::Type{T}, e) where {T}
  m = rand(rng, 2^(precision(T) - 1):(2^precision(T) - 1))
  (rand(rng, Bool) ? -one(T) : one(T)) * ldexp(T(m), e - precision(T) + 1)
end
# c and x with a product of about 2^ep, the exponent split at random between them (either can be subnormal)
function rand_factors(rng, ::Type{T}, ep) where {T}
  E, Es = emax(T), esub(T)
  ec = rand(rng, max(Es, ep - E):min(E, ep - Es))
  rand_value(rng, T, ec), rand_value(rng, T, ep - ec)
end

# A symmetric case whose ordinary dot is not finite on every route: it holds products a b and -a b2, with b2 = b or b
# moved by at most 1000 units in its last place, and a b >= 2^(emax + 2), which overflows alone and when fused with
# any finite partial sum (check_case asserts the overflow); the products cancel when b2 = b. The other coordinates
# have random product exponents over the whole range (some near the top, so that partial sums can overflow), with
# zeros; d is zero, random, or -1, -1/2 or 1 times the dot. Cases outside the asserted region are drawn again.
function random_case(rng, ::Type{T}) where {T}
  E, Es = emax(T), esub(T)
  while true
    p = rand(rng, 2:4)
    C, X = zeros(T, p, p), zeros(T, p, p)
    a, b = rand_factors(rng, T, E + 2 + rand(rng, 0:5))
    a, b = abs(a), abs(b)
    b2 = rand(rng, Bool) ? b : nextfloat(b, rand(rng, -1000:1000))
    isfinite(b2) || continue
    if rand(rng, Bool)        # two diagonal products
      C[1, 1], X[1, 1], C[2, 2], X[2, 2] = a, b, a, -b2
      pair = ((1, 1), (2, 2))
    else                      # an off-diagonal coordinate (both positions) against two diagonal products
      C[2, 1] = C[1, 2] = a; X[2, 1] = X[1, 2] = b
      C[1, 1], X[1, 1], C[2, 2], X[2, 2] = a, -b2, a, -b
      pair = ((1, 1), (2, 2), (2, 1))
    end
    for j in 1:p, i in j:p
      ((i, j) in pair || rand(rng) < 0.25) && continue
      ep = rand(rng) < 0.3 ? rand(rng, E - 6:E - 1) : rand(rng, 2Es + 2:E - 3)
      ci, xi = rand_factors(rng, T, ep)
      C[i, j] = C[j, i] = ci; X[i, j] = X[j, i] = xi
    end
    k = rand(rng, 1:4)
    d = k == 1 ? zero(T) : k == 2 ? rand_value(rng, T, rand(rng, Es:E - 2)) :
        T(-sum(big.(C) .* big.(X)) * rand(rng, (1, 0.5, -1)))
    isfinite(d) || continue
    reference(T, C, X, d).asserted && return C, X, d
  end
end

diagm3(T, a, b, c) = Matrix(Diagonal(T[a, b, c]))

function test_recovery(::Type{T}) where {T}
  E, Es = emax(T), esub(T)
  h, M = pow2(T, E), floatmax(T)
  @testset "LinearFunction value recovery, $T" begin
    @testset "the reproducer of #903" begin
      # c = diag(1, 1, floatmax), x = diag(2^emax, 2^emax, small): the dot 2^(emax+1) - floatmax small is in range;
      # the dense Float32 dot can be finite already (its order of summation), so overflow is not asserted
      sm = T == Float64 ? -pow2(T, -46) : -pow2(T, -17)
      C, X = diagm3(T, 1, 1, M), diagm3(T, h, h, sm)
      for d in (zero(T), -one(T))
        check_case(T, C, X, d; overflow = false)
      end
      # the small term off the diagonal: an off-diagonal coordinate counts twice
      C = T[1 0 0; 0 0 M; 0 M 1]
      X = T[h 0 0; 0 0 sm/2; 0 sm/2 h]
      check_case(T, C, X, zero(T); overflow = false)
    end
    @testset "widely separated exponents beside an overflow" begin
      t = pow2(T, -(E - 1))
      C = T[4 0 t; 0 4 0; t 0 M]
      X = T[h 0 3pow2(T, E - 3); 0 -h 0; 3pow2(T, E - 3) 0 T(1.5) * pow2(T, -(E - 2))]
      for d in (zero(T), T(-0.5), pow2(T, E - 1))
        check_case(T, C, X, d)        # small arguments times large coefficients
        check_case(T, X, C, d)        # and the reverse
      end
    end
    @testset "subnormal factors with normal products beside an overflow" begin
      σ, φ = pow2(T, Es + 2), pow2(T, -(E - 1) - 2)
      @test issubnormal(σ) && issubnormal(φ)
      C = T[4 φ 0; φ 4 0; 0 0 σ]
      X = T[h 3pow2(T, E - 1) 0; 3pow2(T, E - 1) -h 0; 0 0 M]
      for d in (zero(T), T(0.25))
        check_case(T, C, X, d)        # subnormal coefficients
        check_case(T, X, C, d)        # subnormal arguments
      end
    end
    @testset "overflow cancellation and the constant" begin
      check_case(T, diagm3(T, 4, 4, 1), diagm3(T, h, -h, 0.75), zero(T))          # a small remainder
      check_case(T, diagm3(T, 4, 2, 2), diagm3(T, h, -h, -h / 2), zero(T))        # a normal remainder, 2^emax
      check_case(T, T[4 0; 0 4], T[h 0; 0 -prevfloat(h)], zero(T))                # a few units of the last place
      for d in (pow2(T, -30), -pow2(T, -30))                                      # a small d after a cancelling dot
        check_case(T, T[4 4; 4 4], T[h -h; -h h], d)
      end
      for d in (h / 2, -h, -T(0.75) * h)                                          # d of the size of the dot, negative d
        check_case(T, diagm3(T, 4, 2, 2), diagm3(T, h, -h, -h / 2), d)
      end
    end
    @testset "random exponents over the whole range" begin
      rng = Xoshiro(903)
      for _ in 1:300
        C, X, d = random_case(rng, T)
        check_case(T, C, X, d)
      end
    end
    @testset "nonfinite data, coefficients or constant keep the ordinary value" begin
      # (C, X, d, the expected class or nothing): the value is the ordinary value of the route, and of that class
      I2 = T[1 0; 0 1]
      cases = [(I2, T[Inf 0; 0 1], zero(T), :inf),
               (I2, T[Inf 0; 0 -Inf], zero(T), :nan),
               (T[NaN 0; 0 1], I2, zero(T), :nan),
               (I2, I2, T(Inf), :inf),
               # the finite products overflow as well: the nonfinite entry or coefficient, not the overflow, decides
               (ones(T, 2, 2), T[h h; h Inf], zero(T), :inf),
               (T[4 0; 0 NaN], T[h 0; 0 1], zero(T), :nan),
               # 4 2^emax overflows to +Inf against -Inf (NaN), unless a fused product meets -Inf first (-Inf)
               (T[4 0; 0 1], T[h 0; 0 -Inf], zero(T), nothing)]
      for (C, X, d, class) in cases, (route, v, ordinary) in route_values(C, X, d)
        @test (route, isequal(v, ordinary), isfinite(v)) == (route, true, false)
        class === :inf && @test (route, v === T(Inf)) == (route, true)
        class === :nan && @test (route, isnan(v)) == (route, true)
      end
    end
    if T == Float64
      @testset "cancellation far beyond the range: no assertion on the value (#907)" begin
        # products of ±2^1500 cancel in real arithmetic beside d = 0.25: B exceeds floatmax, and the test checks that
        # the call returns a Float64
        a = pow2(T, 750)
        @test !reference(T, T[a 0; 0 a], T[a 0; 0 -a], T(0.25)).asserted
        for (route, v, _) in route_values(T[a 0; 0 a], T[a 0; 0 -a], T(0.25))
          @test (route, v isa T) == (route, true)
        end
      end
    end
  end
end

for T in (Float32, Float64)
  test_recovery(T)
end

end
