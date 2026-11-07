module ThreeFactorProductTest

# The direct strong-zero product A * X * B of two Symmetric matrices and a matrix SparseIterate X
# with Float32 or Float64 entries (#419): C[i, j] is the sum of A[i, k] * X[k, l] * B[l, j] over the
# coordinates where X is mathematically nonzero, a term is the exact product of its three factors
# rounded to the element type, and the terms are added in that precision in the storage order of X.

using Test, ProximalBase, LinearAlgebra, SparseArrays, Random

const SI = ProximalBase

####################################
#
#  The reference. It calls no helper of the package: the contribution set comes from Matrix(X), the
#  exact products and their exact sum are BigFloat, and only + - * and conversions are used (no
#  frexp or ldexp on a BigFloat).
#
####################################

# A BigFloat of this many bits holds the exact sum of any number of products of three Float64 values.
# Such a product is a multiple of 2^-3222 and is below 2^3072, so every sum of them is a multiple of
# 2^-3222 and below 2^(3072 + 64), which needs fewer than 6400 bits.
const REFPREC = 6600

# The class of one term, from the rule written out here: a NaN factor gives NaN, a zero factor
# against an infinite one gives NaN, a zero factor gives zero, an infinite factor gives an infinity,
# and otherwise the term is finite.
function term_class(a, x, b)
  (isnan(a) || isnan(x) || isnan(b)) && return :nan
  anyinf = isinf(a) || isinf(x) || isinf(b)
  (iszero(a) || iszero(x) || iszero(b)) && return anyinf ? :nan : :zero
  anyinf ? :inf : :finite
end

# the entries of C that disagree with the reference, as (i, j, computed, expected) tuples.
# exact: finite entries must equal the correctly rounded exact sum (the powers of two of the range
# tests); otherwise abs(C - exact) <= (s + 8) * eps(T) * sum(abs(t)), plus (s + 2) subnormal spacings
# when a term or the sum is below floatmin. An entry whose finite terms sum above floatmax in
# magnitude, or that has a term within rounding of the overflow threshold, is skipped unless a NaN
# term decides it: that sum may overflow before a cancellation, the limitation of the method.
function mismatches(C::AbstractMatrix{T}, A, X, B; exact::Bool = false) where {T}
  M, Am, Bm = Matrix(X), Matrix(A), Matrix(B)
  m, n = size(M)
  nzs = [(k, l) for l in 1:n for k in 1:m if !iszero(M[k, l])]
  bad = Tuple[]
  size(C) == (m, n) || return [(0, 0, size(C), (m, n))]
  setprecision(BigFloat, REFPREC) do
    for j in 1:n, i in 1:m
      nan = false; pinf = false; minf = false
      total = BigFloat(0); mag = BigFloat(0); s = 0; small = false; edge = false
      for (k, l) in nzs
        a, x, b = Am[i, k], M[k, l], Bm[l, j]
        cls = term_class(a, x, b)
        cls === :zero && continue
        if cls === :nan
          nan = true
        elseif cls === :inf
          (signbit(a) ⊻ signbit(x) ⊻ signbit(b)) ? (minf = true) : (pinf = true)
        else
          t = BigFloat(a) * BigFloat(x) * BigFloat(b)
          if isinf(T(t))                # the product alone is beyond the range of T
            signbit(t) ? (minf = true) : (pinf = true)
            # within a few units of rounding of the overflow threshold, the two roundings of a
            # computed product may land on either side: that entry is not asserted
            abs(t) <= BigFloat(floatmax(T)) * (1 + 4 * BigFloat(eps(T))) && (edge = true)
          else
            total += t; mag += abs(t); s += 1
            small |= abs(t) < BigFloat(floatmin(T))
          end
        end
      end
      c = C[i, j]
      # finite terms whose magnitudes sum beyond floatmax may overflow, with either sign, before
      # a later term: the class of such an entry is not asserted unless a NaN term decides it
      !nan && (edge || mag > BigFloat(floatmax(T))) && continue
      if nan || (pinf && minf)
        isnan(c) || push!(bad, (i, j, c, T(NaN)))
      elseif pinf || minf
        want = pinf ? T(Inf) : -T(Inf)
        c === want || push!(bad, (i, j, c, want))
      else
        want = T(total)
        if exact
          isequal(c, want) || push!(bad, (i, j, c, want))
        else
          (isfinite(c) && !isnan(c)) || (push!(bad, (i, j, c, want)); continue)
          small |= abs(total) < BigFloat(floatmin(T))
          tol = (s + 8) * BigFloat(eps(T)) * mag
          small && (tol += (s + 2) * BigFloat(nextfloat(zero(T))))
          abs(BigFloat(c) - total) <= tol || push!(bad, (i, j, c, want))
        end
      end
    end
  end
  bad
end

####################################
#
#  Fixtures
#
####################################

# a matrix iterate with the given entries stored in the given order
function iterate_of(::Type{T}, m, n, entries) where {T}
  X = SparseIterate(T, m, n)
  for ((i, j), v) in entries
    X[i, j] = v
  end
  X
end

emax(::Type{T}) where {T} = exponent(floatmax(T))
emin(::Type{T}) where {T} = exponent(floatmin(T))
esub(::Type{T}) where {T} = exponent(floatmin(T)) - precision(T) + 1
pow2(::Type{T}, e::Integer) where {T} = ldexp(one(T), e)     # exact, also for subnormal results

# the product of three 1 × 1 operands
function scalar_product(a::T, x::T, b::T) where {T}
  Symmetric(fill(a, 1, 1)) * iterate_of(T, 1, 1, [((1, 1), x)]) * Symmetric(fill(b, 1, 1))
end

# a random symmetric operand and a random iterate drawn from a pool that holds every class
# mode :all holds every class and magnitude, :moderate the special values and small magnitudes
# (no overflow of a sum), :finite ordinary finite values
function pool(::Type{T}, mode::Symbol) where {T}
  big = pow2(T, emax(T) ÷ 3)
  tiny = pow2(T, emin(T) ÷ 3)
  mode === :finite && return T[0, 1, -1, 2, 0.5, 3.25]
  mode === :moderate && return T[0, -0.0, 1, -1, 2, -0.5, 3.25, tiny, -tiny, nextfloat(zero(T)),
                                 -nextfloat(zero(T)), Inf, -Inf, NaN]
  T[0, -0.0, 1, -1, 2, -0.5, 3.25, tiny, -tiny, big, -big, floatmax(T), -floatmax(T), floatmin(T),
    nextfloat(zero(T)), -nextfloat(zero(T)), Inf, -Inf, NaN]
end

function random_case(rng, ::Type{T}, m, n, dens; mode = :all) where {T}
  P = pool(T, mode)
  Am = rand(rng, P, m, m); Bm = rand(rng, P, n, n)
  X = SparseIterate(T, m, n)
  for l in 1:n, k in 1:m
    rand(rng) < dens && (X[k, l] = rand(rng, P))
  end
  uplo = rand(rng, (:U, :L))
  Symmetric(Am, uplo), X, Symmetric(Bm, uplo === :U ? :L : :U)
end

####################################

@testset "three-factor product A * X * B" begin

  @testset "the reference agrees with the dense product on finite data" begin
    for T in (Float32, Float64), trial in 1:20
      rng = MersenneTwister(trial)
      A, X, B = random_case(rng, T, rand(rng, 1:7), rand(rng, 1:7), 0.4; mode = :finite)
      C = A * X * B
      @test isempty(mismatches(C, A, X, B))
      @test isempty(mismatches(Matrix(A) * Matrix(X) * Matrix(B), A, X, B))
      @test isapprox(C, Matrix(A) * Matrix(X) * Matrix(B); rtol = 1000 * eps(T), atol = 1000 * eps(T))
    end
  end

  @testset "the scalar rule, written separately" begin
    for T in (Float32, Float64)
      vals = T[0, -0.0, 1, -3, 0.5, floatmax(T), nextfloat(zero(T)), Inf, -Inf, NaN]
      for a in vals, x in vals, b in vals
        iszero(x) && continue
        cls = term_class(a, x, b)
        t = scalar_product(a, x, b)[1, 1]
        if cls === :nan
          @test isnan(t)
        elseif cls === :zero
          @test t === zero(T)
        elseif cls === :inf
          @test t === ((signbit(a) ⊻ signbit(x) ⊻ signbit(b)) ? -T(Inf) : T(Inf))
        else
          @test isempty(mismatches(fill(t, 1, 1), Symmetric(fill(a, 1, 1)),
                                   iterate_of(T, 1, 1, [((1, 1), x)]), Symmetric(fill(b, 1, 1))))
        end
      end
    end
  end

  @testset "FT11 fixtures" begin
    for T in (Float32, Float64)
      one_x(m, n, coords) = iterate_of(T, m, n, [(c, one(T)) for c in coords])
      @testset "p1 ($T)" begin
        Am = Matrix{T}(I, 4, 4); Am[1, 2] = Am[2, 1] = Inf
        A, B = Symmetric(Am), Symmetric(ones(T, 4, 4) + 3I)
        X = one_x(4, 4, [(1, 1), (2, 2), (3, 1)])
        C = A * X * B
        @test C isa Matrix{T}
        @test isequal(C, T[Inf Inf Inf Inf; Inf Inf Inf Inf; 4 1 1 1; 0 0 0 0])
        @test isempty(mismatches(C, A, X, B))
      end
      @testset "p4 ($T)" begin
        A = Symmetric(Matrix{T}(I, 4, 4) .+ 1)
        Bm = Matrix{T}(I, 4, 4) .+ 1; Bm[4, 1] = Bm[1, 4] = NaN
        B = Symmetric(Bm)
        X = one_x(4, 4, [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3)])
        C = A * X * B
        @test isequal(C, T[9 10 9 NaN; 8 10 10 NaN; 7 8 9 NaN; 6 7 7 NaN])
        @test isempty(mismatches(C, A, X, B))
      end
      @testset "p6 ($T)" begin
        A, B = Symmetric(Matrix(Diagonal(T[Inf, 1, 1, 1]))), Symmetric(Matrix{T}(I, 4, 4))
        X = one_x(4, 4, [(1, 1), (1, 2), (2, 1), (3, 1)])
        C = A * X * B
        @test isequal(C, T[NaN NaN NaN NaN; 1 0 0 0; 1 0 0 0; 0 0 0 0])
        @test isempty(mismatches(C, A, X, B))
      end
      @testset "R5 ($T)" begin
        A, B = Symmetric(T[NaN 0 0; 0 1 0; 0 0 1]), Symmetric(Matrix{T}(I, 3, 3))
        X1 = one_x(3, 3, [(2, 2)])
        X2 = one_x(3, 3, [(2, 2), (2, 3), (3, 2), (3, 3)])
        @test isequal(A * X1 * B, T[0 0 0; 0 1 0; 0 0 0])
        @test isequal(A * X2 * B, T[0 0 0; 0 1 1; 0 1 1])
        @test isempty(mismatches(A * X1 * B, A, X1, B))
        @test isempty(mismatches(A * X2 * B, A, X2, B))
      end
      @testset "mixed range ($T)" begin
        u = pow2(T, cld(3 * emax(T), 5))
        A, B = Symmetric(T[Inf 0; 0 u]), Symmetric(T[1 0; 0 1/u])
        X = iterate_of(T, 2, 2, [((1, 1), one(T)), ((2, 2), u)])
        C = A * X * B
        @test isequal(C, T[Inf NaN; 0 u])
        @test C[2, 2] == u
        @test isempty(mismatches(C, A, X, B))
      end
    end
  end

  @testset "range: the first partial product overflows or underflows" begin
    permuted(v, p) = (v[p[1]], v[p[2]], v[p[3]])
    perms = ((1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2))
    for T in (Float32, Float64)
      hi = (emax(T) ÷ 2 + 1, 3 * emax(T) ÷ 4, emax(T) - 1, emax(T))
      lo = (-esub(T) ÷ 2 + 1, -emin(T) - 1, -emin(T), emax(T))     # 2^e must be representable
      for e in hi, sgn in (1, -1), p in perms        # 2^e * 2^e * 2^-e: the first product overflows
        C = scalar_product(permuted((pow2(T, e), sgn * pow2(T, e), pow2(T, -e)), p)...)
        @test C[1, 1] === sgn * pow2(T, e)
      end
      for e in lo, sgn in (1, -1), p in perms        # 2^-e * 2^-e * 2^e: the first product underflows
        C = scalar_product(permuted((pow2(T, -e), sgn * pow2(T, -e), pow2(T, e)), p)...)
        @test C[1, 1] === sgn * pow2(T, -e)
      end
      big = pow2(T, emax(T))
      for sgn in (1, -1)
        # an overflow that is legitimate, on both signs
        @test scalar_product(big, sgn * big, big)[1, 1] === sgn * T(Inf)
        @test scalar_product(floatmax(T), sgn * T(2), T(1))[1, 1] === sgn * T(Inf)
        # a large product followed by a tiny factor, and a subnormal factor followed by large ones
        @test scalar_product(big, sgn * big, pow2(T, emin(T) - 1))[1, 1] === sgn * pow2(T, 2emax(T) + emin(T) - 1)
        @test scalar_product(nextfloat(zero(T)), sgn * big, big)[1, 1] === sgn * pow2(T, esub(T) + 2emax(T))
        @test scalar_product(nextfloat(zero(T)), sgn * big, pow2(T, -esub(T) - emax(T)))[1, 1] === T(sgn)
        # subnormal results from normal operands
        @test scalar_product(floatmin(T), sgn * T(0.5), T(1))[1, 1] === sgn * floatmin(T) / 2
        @test scalar_product(floatmin(T), sgn * T(0.5), T(0.5))[1, 1] === sgn * pow2(T, emin(T) - 2)
      end
      # overflow or underflow of the first product followed by a finite zero, a zero of A or of B
      @test scalar_product(big, big, zero(T))[1, 1] === zero(T)
      @test scalar_product(zero(T), big, big)[1, 1] === zero(T)
      @test scalar_product(pow2(T, esub(T)), pow2(T, esub(T)), zero(T))[1, 1] === zero(T)
      # a stored zero of X forms no term at all
      @test scalar_product(big, zero(T), big)[1, 1] === zero(T)
      # a result below half the smallest subnormal rounds to zero
      @test scalar_product(nextfloat(zero(T)), T(0.25), T(1))[1, 1] === zero(T)
    end
  end

  @testset "random exponents against the high-precision reference" begin
    rng = MersenneTwister(419)
    for T in (Float32, Float64)
      lo, hi = esub(T), emax(T)
      for trial in 1:1500
        ea, ex = rand(rng, lo:hi), rand(rng, lo:hi)
        # half of the draws put the final exponent near the range of T, whatever the first product does
        eb = isodd(trial) ? rand(rng, lo:hi) : clamp(rand(rng, lo-3:hi+3) - ea - ex, lo, hi)
        mk(e) = (rand(rng, Bool) ? 1 : -1) * (1 + rand(rng, T)) * pow2(T, e)
        a, x, b = mk(ea), mk(ex), mk(eb)
        A, X, B = Symmetric(fill(a, 1, 1)), iterate_of(T, 1, 1, [((1, 1), x)]), Symmetric(fill(b, 1, 1))
        @test isempty(mismatches(A * X * B, A, X, B))
      end
    end
  end

  @testset "random operands with every class" begin
    for T in (Float32, Float64)
      rng = MersenneTwister(7)
      for trial in 1:300
        m, n = rand(rng, 1:6), rand(rng, 1:6)
        A, X, B = random_case(rng, T, m, n, rand(rng, (0.1, 0.3, 0.7, 1.0)))
        C = A * X * B
        @test C isa Matrix{T}
        @test size(C) == (m, n)
        @test isempty(mismatches(C, A, X, B))
      end
    end
  end

  @testset "NaN and Inf in each operand, participating or meeting a zero of X" begin
    for T in (Float32, Float64), special in (T(NaN), T(Inf), -T(Inf)), place in (:A, :X, :B)
      Am, Bm = T[1 2 3; 2 4 5; 3 5 6], T[2 1 0; 1 3 1; 0 1 4]
      X = iterate_of(T, 3, 3, [((1, 2), T(2)), ((3, 3), T(-1))])
      place === :A && (Am[1, 1] = special)      # A[1, 1] meets the coefficient X[1, 2]
      place === :B && (Bm[2, 2] = special)      # B[2, 2] meets the coefficient X[1, 2]
      place === :X && (X = iterate_of(T, 3, 3, [((1, 2), special), ((3, 3), T(-1))]))
      A, B = Symmetric(Am), Symmetric(Bm)
      C = A * X * B
      @test isempty(mismatches(C, A, X, B))
      @test !all(isfinite, C)
      # the same operands with X[1, 2] zero: no term meets the special value
      Xz = iterate_of(T, 3, 3, [((1, 2), zero(T)), ((2, 1), -zero(T)), ((3, 3), T(-1))])
      Cz = A * Xz * B
      @test isempty(mismatches(Cz, A, Xz, B))
      @test all(isfinite, Cz)
      # stored zeros of X and an empty X against NaN or Inf everywhere in A and B: no term
      Az, Bz = Symmetric(fill(special, 3, 3)), Symmetric(fill(special, 3, 3))
      Xs = iterate_of(T, 3, 3, [((2, 1), zero(T)), ((1, 3), -zero(T))])
      @test all(c -> c === zero(T), Az * Xs * Bz)
      @test all(c -> c === zero(T), Az * SparseIterate(T, 3, 3) * Bz)
    end
    for T in (Float32, Float64)
      # the genuine 0 * Inf of a nonzero coefficient is NaN, in every entry that it reaches
      A, B = Symmetric(T[0 0; 0 1]), Symmetric(T[1 0; 0 1])
      X = iterate_of(T, 2, 2, [((1, 1), T(Inf))])
      C = A * X * B
      @test all(isnan, C)
      @test isempty(mismatches(C, A, X, B))
      # a zero of B against a finite coefficient
      Xf = iterate_of(T, 2, 2, [((2, 2), T(3))])
      @test isequal(Symmetric(T[1 1; 1 1]) * Xf * Symmetric(T[0 0; 0 2]), T[0 6; 0 6])
    end
  end

  @testset "storage order, stored zeros and density do not change the product" begin
    for T in (Float32, Float64)
      rng = MersenneTwister(11)
      for trial in 1:40
        m, n = rand(rng, 2:8), rand(rng, 2:8)
        A, Xr, B = random_case(rng, T, m, n, 0.3; mode = :moderate)
        entries = [((k, l), Xr[k, l]) for l in 1:n for k in 1:m if !iszero(Xr[k, l])]
        results = Matrix{T}[]
        for ord in 1:4
          ord == 2 && (entries = reverse(entries))
          ord == 3 && shuffle!(rng, entries)
          X = iterate_of(T, m, n, entries)
          if ord == 4                    # explicit stored zeros, some then dropped
            for (k, l) in ((1, 1), (m, n), (1, n))
              iszero(Xr[k, l]) && (X[k, l] = zero(T))
            end
          end
          C = A * X * B
          @test isempty(mismatches(C, A, X, B))
          push!(results, C)
          ord == 4 && (dropzeros!(X); @test isequal(A * X * B, C))
        end
        for C in results[2:end]            # the classes agree; the finite values are checked above
          @test isequal(isnan.(C), isnan.(results[1]))
          @test isequal(ifelse.(isinf.(C), sign.(C), 0), ifelse.(isinf.(results[1]), sign.(results[1]), 0))
        end
      end
      for dens in (0.02, 0.2, 0.6, 1.0)
        A, X, B = random_case(MersenneTwister(5), T, 12, 9, dens; mode = :finite)
        @test isempty(mismatches(A * X * B, A, X, B))
      end
    end
  end

  @testset "shapes" begin
    for T in (Float32, Float64)
      A0, B0 = Symmetric(zeros(T, 0, 0)), Symmetric(zeros(T, 3, 3))
      @test size(A0 * SparseIterate(T, 0, 3) * B0) == (0, 3)
      A3, B0 = Symmetric(ones(T, 3, 3)), Symmetric(zeros(T, 0, 0))
      @test size(A3 * SparseIterate(T, 3, 0) * B0) == (3, 0)
      @test size(A0 * SparseIterate(T, 0, 0) * B0) == (0, 0)
      @test isequal(Symmetric(fill(T(NaN), 1, 1)) * SparseIterate(T, 1, 1) * Symmetric(fill(T(Inf), 1, 1)), fill(zero(T), 1, 1))
      @test scalar_product(T(2), T(3), T(-5)) == fill(T(-30), 1, 1)
      # all stored values zero
      X = iterate_of(T, 4, 4, [((1, 1), zero(T)), ((2, 3), -zero(T))])
      Aq, Bq = Symmetric(fill(T(NaN), 4, 4)), Symmetric(fill(T(Inf), 4, 4))
      @test all(iszero, Aq * X * Bq)
      # rectangular and square iterates
      for (m, n) in ((6, 2), (2, 6), (5, 5), (1, 7), (7, 1))
        A, X, B = random_case(MersenneTwister(m * 10 + n), T, m, n, 0.5; mode = :finite)
        @test isempty(mismatches(A * X * B, A, X, B))
      end
      @test_throws DimensionMismatch Symmetric(ones(T, 3, 3)) * SparseIterate(T, 4, 2) * Symmetric(ones(T, 2, 2))
      @test_throws DimensionMismatch Symmetric(ones(T, 4, 4)) * SparseIterate(T, 4, 2) * Symmetric(ones(T, 3, 3))
    end
  end

  @testset "the ignored triangle is not read, inputs are unchanged" begin
    for T in (Float32, Float64), uplo in (:U, :L)
      rng = MersenneTwister(3)
      Am, Bm = randn(rng, T, 5, 5), randn(rng, T, 4, 4)
      X = iterate_of(T, 5, 4, [((2, 3), T(1.5)), ((5, 1), T(-2)), ((1, 4), T(0.25))])
      ignored(M) = uplo === :U ? tril(trues(size(M)), -1) : triu(trues(size(M)), 1)
      Cok = Symmetric(Am, uplo) * X * Symmetric(Bm, uplo)
      Ap, Bp = copy(Am), copy(Bm)
      Ap[ignored(Ap)] .= T(NaN); Bp[ignored(Bp)] .= T(Inf)
      A, B = Symmetric(Ap, uplo), Symmetric(Bp, uplo)
      before = (copy(Ap), copy(Bp), Matrix(X), X.nzval[1:nnz(X)], nnz(X), copy(activecoordinates(X)))
      C = A * X * B
      @test isequal(C, Cok)
      @test all(isfinite, C)
      @test isequal(before[1], Ap) && isequal(before[2], Bp) && isequal(before[3], Matrix(X)) &&
            before[4] == X.nzval[1:nnz(X)] && before[5] == nnz(X) && before[6] == activecoordinates(X)
      @test C isa Matrix{T}
    end
  end

  @testset "an aggregate zero is +0.0" begin
    for T in (Float32, Float64)
      A, B = Symmetric(ones(T, 3, 3)), Symmetric(ones(T, 3, 3))
      X = iterate_of(T, 3, 3, [((1, 1), T(2)), ((2, 2), -T(2)), ((3, 1), T(-0.0))])
      C = A * X * B
      @test all(c -> c === zero(T), C)
      Xn = iterate_of(T, 3, 3, [((1, 1), -T(1))])
      @test all(c -> c === -T(1), Symmetric(ones(T, 3, 3)) * Xn * B)
      C = Symmetric(Matrix(Diagonal(T[0, 1, 1]))) * iterate_of(T, 3, 3, [((1, 1), -T(1))]) * B
      @test all(c -> c === zero(T), C[1, :])
      @test all(!signbit, C[1, :])
    end
  end

  @testset "dispatch" begin
    new_sig = Tuple{typeof(*), Symmetric{T, <:StridedMatrix{T}}, SparseIterate{T, 2},
                    Symmetric{T, <:StridedMatrix{T}}} where {T<:Union{Float32, Float64}}
    fwd(A, X, B) = which(*, Tuple{typeof(A), typeof(X), typeof(B)})
    for T in (Float32, Float64)
      A, B = Symmetric(randn(T, 3, 3)), Symmetric(randn(T, 2, 2))
      X = iterate_of(T, 3, 2, [((1, 1), one(T))])
      @test fwd(A, X, B).sig <: new_sig
      @test A * X * B isa Matrix{T}
      Av, Bv = Symmetric(view(randn(T, 6, 6), 1:3, 1:3)), Symmetric(view(randn(T, 4, 4), 1:2, 1:2))
      @test fwd(Av, X, Bv).sig <: new_sig
    end
    # a mixed call, and other element types, reach the old methods without a MethodError
    A32, B64 = Symmetric(randn(Float32, 3, 3)), Symmetric(randn(2, 2))
    X64 = iterate_of(Float64, 3, 2, [((1, 1), 1.0), ((3, 2), -2.0)])
    @test !(fwd(A32, X64, B64).sig <: new_sig)
    @test A32 * X64 * B64 ≈ Matrix(A32) * Matrix(X64) * Matrix(B64)
    for T in (Float16, BigFloat)
      A, B = Symmetric(T.(randn(3, 3))), Symmetric(T.(randn(2, 2)))
      X = iterate_of(T, 3, 2, [((1, 1), one(T)), ((3, 2), T(-2))])
      @test !(fwd(A, X, B).sig <: new_sig)
      @test A * X * B ≈ Matrix(A) * Matrix(X) * Matrix(B) rtol = (T === Float16 ? 1e-2 : 1e-12)
    end
    # the owner keeps its own method
    Ao, Bo = Symmetric(randn(4, 4)), Symmetric(randn(4, 4))
    Xo = SymmetricSparseIterate(4); Xo[3, 1] = 2.0
    @test !(fwd(Ao, Xo, Bo).sig <: new_sig)
    @test Ao * Xo * Bo ≈ Ao * Matrix(Xo) * Bo
    # no pair of methods of * on these types is ambiguous
    amb = filter(Test.detect_ambiguities(ProximalBase)) do (m1, m2)
      m1.name === :* && m2.name === :* && endswith(string(m1.file), "sparse_iterate.jl") &&
        endswith(string(m2.file), "sparse_iterate.jl")
    end
    @test isempty(amb)
  end

  @testset "allocation does not grow with the number of nonzeros" begin
    @noinline measure(A, X, B) = @allocated A * X * B
    for T in (Float32, Float64)
      m, n = 30, 20
      A, B = Symmetric(randn(T, m, m)), Symmetric(randn(T, n, n))
      X1 = iterate_of(T, m, n, [((i, 2i), one(T)) for i in 1:2])
      X2 = iterate_of(T, m, n, [((mod1(7i, m), mod1(3i, n)), T(i)) for i in 1:120])
      Xe = SparseIterate(T, m, n)
      for X in (X1, X2, Xe)
        A * X * B                         # compiled before measuring
      end
      a1, a2, a0 = measure(A, X1, B), measure(A, X2, B), measure(A, Xe, B)
      @test nnz(X2) > 10 * nnz(X1)
      @test a1 == a2 == a0
      # the output, two buffers of length m and one of length n, and the array headers
      @test a1 <= sizeof(T) * (m * n + 2m + n) + 1024
    end
    f(a, x, b) = @allocated SI._triple(a, x, b)
    for T in (Float32, Float64)
      args = (T(3), T(-5), T(0.25))
      SI._triple(args...)
      @test f(args...) == 0
      big = floatmax(T)
      SI._triple(big, big, T(1) / big)
      @test f(big, big, T(1) / big) == 0
      @test f(T(Inf), T(0), T(1)) == 0
    end
  end

  @testset "the scalar kernel against the high-precision reference" begin
    for T in (Float32, Float64)
      rng = MersenneTwister(2)
      exps = unique(sort!([esub(T), esub(T) + 1, emin(T) - 1, emin(T), emin(T) + 1, -1, 0, 1,
                           emax(T) - 1, emax(T), -emax(T), emin(T) ÷ 2, emax(T) ÷ 2]))
      mants = T[1, nextfloat(T(1)), prevfloat(T(2)), T(1.5), T(1.25)]
      vals = T[]
      for e in exps, mant in mants, s in (1, -1)
        v = s * mant * pow2(T, e)
        isfinite(v) && !iszero(v) && push!(vals, v)
      end
      push!(vals, nextfloat(zero(T)), floatmin(T), prevfloat(floatmin(T)), floatmax(T))
      for trial in 1:3000
        a, x, b = rand(rng, vals), rand(rng, vals), rand(rng, vals)
        t = SI._triple(a, x, b)
        A, X, B = Symmetric(fill(a, 1, 1)), iterate_of(T, 1, 1, [((1, 1), x)]), Symmetric(fill(b, 1, 1))
        @test isempty(mismatches(fill(t, 1, 1), A, X, B))
      end
      # the class rule on the scalar kernel itself
      for a in T[0, 1, -2, Inf, -Inf, NaN], x in T[0, 1, -2, Inf, -Inf, NaN], b in T[0, 1, -2, Inf, -Inf, NaN]
        t = SI._triple(a, x, b)
        cls = term_class(a, x, b)
        cls === :nan && @test isnan(t)
        cls === :zero && @test t === zero(T)
        cls === :inf && @test t === ((signbit(a) ⊻ signbit(x) ⊻ signbit(b)) ? -T(Inf) : T(Inf))
        cls === :finite && @test t == a * x * b
      end
    end
  end

  @testset "sums that overflow before a cancellation (documented limitation)" begin
    # Three terms of magnitude near floatmax with signs +, +, -: the high-precision value of the
    # entry is finite, while the sum in the element type overflows at the second term and stays
    # non-finite. This is the behavior of the method, not a promise of the contract.
    for T in (Float32, Float64)
      h = T(0.9) * floatmax(T)
      A, B = Symmetric(Matrix{T}(I, 3, 3)), Symmetric(ones(T, 3, 3))
      X = iterate_of(T, 3, 3, [((1, 1), h), ((1, 2), h), ((1, 3), -h)])
      C = A * X * B
      exact = setprecision(BigFloat, REFPREC) do
        T(BigFloat(h) + BigFloat(h) - BigFloat(h))
      end
      @test isfinite(exact)
      @test !isfinite(C[1, 1])
      @test all(iszero, C[2:3, :])
    end
  end

  @testset "the deprecated forwarder reaches the product method" begin
    for T in (Float32, Float64)
      A, B = Symmetric(randn(T, 5, 5)), Symmetric(randn(T, 5, 5))
      D = randn(T, 5, 5) .* (rand(5, 5) .< 0.3)
      C = @test_deprecated A_mul_X_mul_B(A, SparseIterate(vec(D)), B)
      @test C isa Matrix{T}
      @test isapprox(C, Matrix(A) * D * Matrix(B); rtol = 1000 * eps(T), atol = 1000 * eps(T))
      @test isempty(mismatches(C, A, SparseIterate(D), B))
    end
  end
end

end
