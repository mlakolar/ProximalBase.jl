module L0SymmetricFullEntryTest

# IndBallL0 on a symmetric matrix counts full entries (#798): a nonzero diagonal entry counts one, a nonzero
# off-diagonal pair counts two, and the projection is the exact Frobenius projection onto that set. The oracle
# below enumerates every support within the budget (p ≤ 4) with exact rational arithmetic and does not call
# any selection code of the package.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

const PB = ProximalBase

rat(x) = Rational{BigInt}(x)

# the full count of a symmetric matrix as the set sees it, the diagonal not counted when it is free
function fullcount(X, diagonal)
  n = 0
  for c in 1:size(X, 1), r in c:size(X, 1)
    iszero(X[r, c]) && continue
    r == c ? (diagonal && (n += 1)) : (n += 2)
  end
  n
end

# the exact minimum of the squared Frobenius error over the supports within the budget: a nonzero diagonal
# entry costs one unit and saves x², a nonzero pair costs two and saves 2x², a free diagonal is kept
function oracle_error(X, s, diagonal)
  p = size(X, 1)
  items = [(r == c ? 1 : 2, rat(X[r, c])^2) for c in 1:p for r in c:p if !iszero(X[r, c]) && (diagonal || r != c)]
  n = length(items)
  total = sum(w * g for (w, g) in items; init = 0 // 1)
  best = 0 // 1
  for mask in 0:(2^n - 1)
    cost, saving = 0, 0 // 1
    for j in 1:n
      if (mask >> (j - 1)) & 1 == 1
        cost += items[j][1]
        saving += items[j][1] * items[j][2]
      end
    end
    cost <= s && (best = max(best, saving))
  end
  total - best
end

prod_error(X, Y) = sum((rat(X[r, c]) - rat(Y[r, c]))^2 for r in 1:size(X, 1), c in 1:size(X, 1); init = 0 // 1)

value_at(X, s, diagonal) = fullcount(X, diagonal) <= s ? 0 : Inf

# The output is feasible, symmetric, made of kept and zeroed entries, a projection of the least error: exactly
# (mode :exact, inputs whose scaled squares are exact), within a relative tolerance of the optimal error
# (:relative), or within that tolerance of the total energy (:range, entries of very different magnitude,
# where the comparison of the savings rounds before the small terms that fix the optimal error are seen).
function verify(f, X::Matrix{T}, Y, mode) where {T}
  s, diagonal = f.s, f.diagonal
  @test Y == Y'
  @test all(i -> Y[i] == X[i] || iszero(Y[i]), CartesianIndices(X))
  diagonal || @test diag(Y) == diag(X)
  @test fullcount(Y, diagonal) <= s
  @test f(Symmetric(Y)) == 0
  e, b = prod_error(X, Y), oracle_error(X, s, diagonal)
  tol = rat(T == Float32 ? 1e-5 : 1e-12)
  total = sum(rat(X[r, c])^2 * (r == c ? 1 : 2) for c in 1:size(X, 1) for r in c:size(X, 1); init = 0 // 1)
  if mode == :exact
    @test e == b
  elseif mode == :relative
    @test abs(e - b) <= tol * b
  else
    @test abs(e - b) <= tol * total
  end
end

# a matrix whose triangle other than uplo holds garbage
function with_garbage(X::Matrix{T}, uplo) where {T}
  A = copy(X)
  for c in 1:size(X, 1), r in 1:size(X, 1)
    (uplo == :L ? r < c : r > c) && (A[r, c] = T(99))
  end
  A
end

# every kind of input and output for one matrix X (symmetric, finite) and one set
function run_case(X::Matrix{T}, s, diagonal, mode) where {T}
  p = size(X, 1)
  f = IndBallL0(s; diagonal)
  expected = value_at(X, s, diagonal)
  Yref = zeros(T, p, p)
  for uplo in (:L, :U)
    A = with_garbage(X, uplo)
    S = Symmetric(A, uplo)
    @test f(S) == expected
    Y = fill(T(7), p, p)
    prox!(Y, f, S, 1)
    uplo == :L && (Yref = copy(Y))
    @test Y == Yref
    verify(f, X, Y, mode)
    Y2 = fill(T(7), p, p)
    v = value_and_prox!(Y2, f, S, 1)
    @test Y2 == Y && v == 0
    B = copy(A)
    prox!(B, f, Symmetric(B, uplo), 1)                       # in place
    @test B == Y
  end
  O = SymmetricSparseIterate(X)
  @test f(O) == expected
  Yo = SymmetricSparseIterate(T, p)
  prox!(Yo, f, O, 1)
  @test Matrix(Yo) == Yref
  Yo = SymmetricSparseIterate(T, p)
  v = value_and_prox!(Yo, f, O, 1)
  @test Matrix(Yo) == Yref && v == 0
  O2 = SymmetricSparseIterate(X)
  prox!(O2, f, O2, 1)                                        # in place
  @test Matrix(O2) == Yref
  Yd = fill(T(7), p, p)
  prox!(Yd, f, O, 1)                                         # an owner argument, a dense output
  @test Yd == Yref
  @test Matrix(O) == X                                       # the argument is not written
  # stored zeros, signed zeros included, are zeros
  Z = SymmetricSparseIterate(X)
  for c in 1:p, r in c:p
    if iszero(X[r, c])
      k = PB.tril_index(p, r, c)
      PB.activate!(Z, k)
      iseven(r + c) && (Z.nzval[Z.ind2nzval[k]] = -zero(T))
    end
  end
  @test f(Z) == expected
  Zo = SymmetricSparseIterate(T, p)
  prox!(Zo, f, Z, 1)
  @test Matrix(Zo) == Yref
  prox!(Z, f, Z, 1)
  @test Matrix(Z) == Yref
  nothing
end

budgets(p, diagonal) = (m = diagonal ? p^2 : p * (p - 1); sort!(unique(filter(>=(0), [0, 1, 2, 3, m - 1, m, m + 1]))))

# symmetric matrices of integers with zeros and ties, and with all magnitudes equal
function integer_matrices(::Type{T}, p) where {T}
  out = Matrix{T}[]
  push!(out, ones(T, p, p))
  push!(out, T[(-1)^(r + c) for r in 1:p, c in 1:p])
  push!(out, T[r == c ? 2 : 1 for r in 1:p, c in 1:p])
  push!(out, T[r == c ? 1 : 2 for r in 1:p, c in 1:p])
  push!(out, T[r == c ? 0 : 1 for r in 1:p, c in 1:p])
  for _ in 1:3
    A = rand(-3:3, p, p)
    push!(out, T.(tril(A) + tril(A, -1)'))
  end
  out
end

function real_matrices(::Type{T}, p) where {T}
  out = Matrix{T}[]
  for _ in 1:3
    A = randn(T, p, p)
    push!(out, tril(A) + tril(A, -1)')
  end
  out
end

@testset "the oracle at the worked examples" begin
  X = [0.0 4.0; 4.0 0.0]
  @test oracle_error(X, 1, true) == 32 && oracle_error(X, 2, true) == 0
  X = [1.0 0.8; 0.8 0.1]
  @test oracle_error(X, 2, true) == rat(1.0)^2 + 2 * rat(0.8)^2 + rat(0.1)^2 - 2 * rat(0.8)^2
end

@testset "worked examples" begin
  Z = prox(IndBallL0(1), Symmetric([0.0 4.0; 4.0 0.0]))
  @test Z == zeros(2, 2) && IndBallL0(1)(Symmetric([0.0 4.0; 4.0 0.0])) == Inf   # a pair does not fit in one unit
  X = [0.0 4.0; 4.0 0.0]
  @test prox(IndBallL0(2), Symmetric(X)) == X && IndBallL0(2)(Symmetric(X)) == 0
  X = [1.0 0.8; 0.8 0.1]
  @test prox(IndBallL0(2), Symmetric(X)) == [0.0 0.8; 0.8 0.0]                    # the pair saves 1.28, both diagonals 1.01
  @test prox(IndBallL0(1), Symmetric(X)) == [1.0 0.0; 0.0 0.0]
  @test prox(IndBallL0(3), Symmetric(X)) == [1.0 0.8; 0.8 0.0]
  @test prox(IndBallL0(3; diagonal = false), Symmetric(X)) == X                   # fld(3, 2) = 1 pair, the diagonal is kept
  W = [1.0 3.0 2.0; 3.0 1.0 2.5; 2.0 2.5 1.0]
  Z = prox(IndBallL0(3; diagonal = false), Symmetric(W))
  @test Z == [1.0 3.0 0.0; 3.0 1.0 0.0; 0.0 0.0 1.0] && count(!iszero, Z) == 5
  @test prox(IndBallL0(1; diagonal = false), Symmetric(W)) == Diagonal(diag(W))
  # ties: a pair and two diagonal entries save 2 each, and the smaller number of pairs wins; within a class the
  # entry earlier in the lower triangle by columns wins
  T2 = [1.0 1.0; 1.0 1.0]
  @test prox(IndBallL0(2), Symmetric(T2)) == [1.0 0.0; 0.0 1.0]                   # e = 0 (two diagonals) over e = 1
  @test prox(IndBallL0(1), Symmetric(T2)) == [1.0 0.0; 0.0 0.0]                   # ties in a class: the earlier position
  @test prox(IndBallL0(3), Symmetric(T2)) == [1.0 1.0; 1.0 0.0]
end

@testset "against the exact oracle, $T" for T in (Float32, Float64)
  Random.seed!(798)
  for p in 0:4, diagonal in (true, false), s in budgets(p, diagonal)
    for X in integer_matrices(T, p)
      run_case(X, s, diagonal, :exact)
    end
    for X in real_matrices(T, p)
      run_case(X, s, diagonal, :relative)
    end
  end
end

@testset "a larger matrix against the dynamic programme, $T" for T in (Float32, Float64)
  Random.seed!(7981)
  p = 9
  A = randn(T, p, p)
  X = tril(A) + tril(A, -1)'
  items = [(r == c ? 1 : 2, Float64(X[r, c])^2) for c in 1:p for r in c:p]
  for diagonal in (true, false), s in (0, 1, 2, 5, 11, 20, 40, 81)
    f = IndBallL0(s; diagonal)
    Y = prox(f, Symmetric(X))
    its = [(w, w * g) for ((w, g), (r, c)) in zip(items, [(r, c) for c in 1:p for r in c:p]) if diagonal || r != c]
    best = zeros(s + 1)
    for (w, g) in its, b in s:-1:w
      best[b + 1] = max(best[b + 1], best[b - w + 1] + g)
    end
    saved = sum(abs2, Float64.(X)) - sum(abs2, Float64.(X) - Float64.(Y))
    free = diagonal ? 0.0 : sum(abs2, Float64.(diag(X)))
    tol = T == Float32 ? 1e-5 : 1e-12
    @test saved - free ≈ best[s + 1] rtol = tol atol = tol * sum(abs2, Float64.(X))
    @test fullcount(Y, diagonal) <= s && f(Symmetric(Y)) == 0
    O = SymmetricSparseIterate(X)
    @test Matrix(prox(f, O)) == Y
  end
end

@testset "NaN, $T" for T in (Float32, Float64)
  X = T[1 2 0; 2 3 0.5; 0 0.5 4]
  Xn = copy(X); Xn[2, 1] = Xn[1, 2] = NaN                     # a NaN in a pair
  Xd = copy(X); Xd[2, 2] = NaN                                # a NaN on the diagonal
  for diagonal in (true, false)
    m = diagonal ? 9 : 6
    for (Xa, counted) in ((Xn, true), (Xd, diagonal)), s in 0:(m + 2)
      f = IndBallL0(s; diagonal)
      owner() = SymmetricSparseIterate(Xa)
      X0 = copy(Xa); counted || (X0[2, 2] = 0)
      expected_value = counted ? (s < m ? NaN : 0.0) : value_at(X0, s, diagonal)
      @test isequal(f(Symmetric(Xa, :L)), expected_value) && isequal(f(owner()), expected_value)
      if counted && 0 < s < m                                  # E2: every counted output is NaN, a free diagonal is copied
        Yref = fill(T(NaN), 3, 3)
        diagonal || (Yref[diagind(Yref)] .= diag(Xa))
        fused = T(NaN)
      elseif counted && s == 0                                 # the zero map
        Yref = zeros(T, 3, 3)
        diagonal || (Yref[diagind(Yref)] .= diag(Xa))
        fused = T(0)
      elseif counted                                           # s ≥ m: the identity, NaN included
        Yref = copy(Xa)
        fused = T(0)
      else                                                     # a free NaN does not count and is kept
        Yref = prox(f, Symmetric(X0)); Yref[2, 2] = NaN
        fused = T(0)
      end
      Y = fill(T(7), 3, 3)
      prox!(Y, f, Symmetric(Xa, :L), 1)
      @test isequal(Y, Yref)
      Y = fill(T(7), 3, 3)
      v = value_and_prox!(Y, f, Symmetric(Xa, :L), 1)
      @test isequal(Y, Yref) && isequal(v, fused)
      O = owner(); Yo = SymmetricSparseIterate(T, 3)
      v = value_and_prox!(Yo, f, O, 1)
      @test isequal(Matrix(Yo), Yref) && isequal(v, fused)
      prox!(O, f, O, 1)
      @test isequal(Matrix(O), Yref)
    end
  end
  # an output of integer type refuses the NaN block before writing
  yi = fill(7, 3, 3)
  @test_throws InexactError prox!(yi, IndBallL0(2), Symmetric(Xn, :L), 1)
  @test yi == fill(7, 3, 3)
end

@testset "infinite entries, $T" for T in (Float32, Float64)
  Inf_ = T(Inf)
  for sg in (1, -1)
    iv = sg * Inf_
    # an infinite diagonal entry has weight one: it is kept and wins every finite saving
    @test prox(IndBallL0(1), Symmetric(T[iv 5; 5 3])) == T[iv 0; 0 0]
    @test prox(IndBallL0(2), Symmetric(T[iv 5; 5 3])) == T[iv 0; 0 3]               # over the pair, which saves 50
    @test prox(IndBallL0(3), Symmetric(T[iv 5; 5 3])) == T[iv 5; 5 0]
    @test prox(IndBallL0(4), Symmetric(T[iv 5; 5 3])) == T[iv 5; 5 3]
    # an infinite pair has weight two
    @test prox(IndBallL0(2), Symmetric(T[1 iv; iv 3])) == T[0 iv; iv 0]
    @test prox(IndBallL0(1), Symmetric(T[1 iv; iv 3])) == T[0 0; 0 3]
    @test prox(IndBallL0(2), Symmetric(T[iv iv; iv 3])) == T[0 iv; iv 0]               # weight two over weight one
    @test prox(IndBallL0(3), Symmetric(T[iv iv; iv 3])) == T[iv iv; iv 0]
    @test prox(IndBallL0(2; diagonal = false), Symmetric(T[iv iv; iv 3])) == T[iv iv; iv 3]
    O = SymmetricSparseIterate(T[iv 5; 5 3])
    @test Matrix(prox(IndBallL0(2), O)) == T[iv 0; 0 3]
    @test IndBallL0(3)(O) == Inf && IndBallL0(4)(O) == 0
    Y = zeros(T, 2, 2)
    @test value_and_prox!(Y, IndBallL0(2), Symmetric(T[iv 5; 5 3]), 1) == 0 && Y == T[iv 0; 0 3]
  end
end

@testset "a large dynamic range, $T" for T in (Float32, Float64)
  Random.seed!(7982)
  fm, fn, ε = floatmax(T), floatmin(T), nextfloat(zero(T))
  # powers of two, so that the scaled squares are exact although the squares overflow
  b2 = ldexp(one(T), exponent(fm))
  Xbig = T[b2 b2/2 0; b2/2 b2/4 0; 0 0 b2/8]
  Xsmall = ε * T[3 2 0; 2 1 0; 0 0 2]                          # subnormal, a square underflows
  Xnear = T[fm fn 0; fn ε 0; 0 0 fm]
  for (X, mode) in ((Xbig, :exact), (Xsmall, :exact), (Xnear, :range)), diagonal in (true, false), s in 0:10
    run_case(X, s, diagonal, mode)
  end
  # random entries of mixed magnitude: the support is optimal up to the rounding of the comparison
  for _ in 1:4, p in 2:4
    A = [rand(Bool) ? randn(T) * fm / 8 : randn(T) * fn for _ in 1:p, _ in 1:p]
    X = tril(A) + tril(A, -1)'
    for diagonal in (true, false), s in budgets(p, diagonal)
      run_case(X, s, diagonal, :range)
    end
    B = randn(T, p, p) * (fm / 4)
    XB = tril(B) + tril(B, -1)'
    @test all(isfinite, XB)
    for diagonal in (true, false), s in budgets(p, diagonal)
      run_case(XB, s, diagonal, :relative)
    end
  end
end

@testset "the whole space is decided without p² overflowing" begin
  huge = typemax(Int)
  @test PB._ncounted_symmetric(IndBallL0(0), huge) == Int128(huge)^2
  @test PB._ncounted_symmetric(IndBallL0(0; diagonal = false), huge) == Int128(huge) * (Int128(huge) - 1)
  @test typemax(Int) < PB._ncounted_symmetric(IndBallL0(typemax(Int)), 3_037_000_500)
  @test PB._ncounted_symmetric(IndBallL0(1), 3) == 9 && PB._ncounted_symmetric(IndBallL0(1; diagonal = false), 3) == 6
  @test PB._ncounted_symmetric(IndBallL0(1), 0) == 0 && PB._ncounted_symmetric(IndBallL0(1; diagonal = false), 0) == 0
  @test PB._ncounted_symmetric(IndBallL0(1; diagonal = false), 1) == 0
end

@testset "NormL0 counts the mirrored entries of an owner twice" begin
  for T in (Float32, Float64)
    S = SymmetricSparseIterate(T[1 2 0; 2 0 0; 0 0 3])         # one pair and two diagonal entries
    @test NormL0(T(1))(S) == 4 && NormL0(T(0.5))(S) == 2
    @test NormL0(T(0.5))(S) == NormL0(T(0.5))(Matrix(S))
    @test IndBallL0(4)(S) == 0 && IndBallL0(3)(S) == Inf       # the set counts the same entries
  end
end

@testset "steady-state allocation of the projection" begin
  Random.seed!(7983)
  alloc(y, f, x) = @allocated prox!(y, f, x, 1.0)
  for T in (Float32, Float64), diagonal in (true, false), (p, s) in ((6, 7), (30, 40), (30, 3))
    A = randn(T, p, p)
    X = tril(A) + tril(A, -1)'
    f = IndBallL0(s; diagonal)
    Sx = Symmetric(X, :L)
    y = zeros(T, p, p)
    alloc(y, f, Sx); alloc(y, f, Sx)
    @test alloc(y, f, Sx) == 0                                   # a dense output
    o = SymmetricSparseIterate(T, p)
    alloc(o, f, Sx); alloc(o, f, Sx)
    @test alloc(o, f, Sx) == 0                                   # an owner output
    z = SymmetricSparseIterate(X)
    w = SymmetricSparseIterate(T, p)
    alloc(w, f, z); alloc(w, f, z)
    @test alloc(w, f, z) == 0                                    # an owner argument and output
  end
end

end
