module IndBallL0Test

using Test, ProximalBase, LinearAlgebra, Random, SparseArrays

####################################
#  A reference independent of the implementation
####################################

const Rat = Rational{BigInt}

# the strict lower pairs in column-major order
lower_pairs(p) = [(i, j) for j in 1:p for i in j+1:p]

# The squared error ‖Y - X‖² over all entries, in exact rational arithmetic. X may be a Symmetric.
function exact_error(Y, X)
  err = zero(Rat)
  for j in axes(X, 2), i in axes(X, 1)
    err += (Rat(Y[i, j]) - Rat(X[i, j]))^2
  end
  err
end

# For every subset of the pairs with at most k members, the squared error of
# keeping that subset (and the diagonal). Returns the minimum error, the
# expected output, and E. Among subsets with the minimum error and k members the
# one with the lexicographically smallest sorted indices is the one with the
# earliest tied entries.
function brute(X, s)
  p = size(X, 1)
  T = eltype(X)
  pairs = lower_pairs(p)
  n = length(pairs)
  k = min(s ÷ 2, n)
  v = [X[i, j] for (i, j) in pairs]
  energy = [2 * Rat(x)^2 for x in v]
  total = sum(energy; init = zero(Rat))
  best, chosen = nothing, Int[]
  for mask in 0:(1 << n) - 1
    count_ones(mask) <= k || continue
    kept = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
    loss = total - sum(energy[kept]; init = zero(Rat))
    if best === nothing || loss < best || (loss == best && isless(kept, chosen))
      best, chosen = loss, kept
    end
  end
  E = count(!iszero, v)
  Y = Matrix{T}(undef, p, p)
  for c in 1:p
    Y[c, c] = X[c, c]
  end
  for (idx, (i, j)) in enumerate(pairs)
    w = (E <= k || idx in chosen) ? v[idx] : zero(T)
    Y[i, j] = w
    Y[j, i] = w
  end
  (best, Y, E)
end

# A symmetric matrix and the two parents of a Symmetric(…, :L) and a Symmetric(…, :U):
# the first matrix is symmetric in the sense of the symmetry check (a pair of zeros
# may have opposite signs), and each parent holds 77 in the triangle that its view
# does not read. The logical lower entry of the pair (i, j) is the same in all three.
function make_input(rng, T, p, density, pool)
  Xm = zeros(T, p, p)
  PL = fill(T(77), p, p)
  PU = fill(T(77), p, p)
  signed(m) = T(m * rand(rng, (-1, 1)))
  signed_zero() = rand(rng, Bool) ? zero(T) : -zero(T)
  for c in 1:p
    d = rand(rng) < 0.3 ? signed_zero() : signed(rand(rng, pool))
    Xm[c, c] = PL[c, c] = PU[c, c] = d
    for r in c+1:p
      v = rand(rng) < density ? signed(rand(rng, pool)) : signed_zero()
      Xm[r, c] = v
      PL[r, c] = v
      PU[c, r] = v
      Xm[c, r] = (iszero(v) && rand(rng, Bool)) ? -v : v
    end
  end
  (Xm, PL, PU)
end

function budgets(p)
  n = p * (p - 1) ÷ 2
  b = Int[0, 1, 2, 3, p * (p - 1) - 1, p * (p - 1), p * (p - 1) + 1, typemax(Int)]
  append!(b, 2 .* (1:n))
  sort!(unique!(filter!(>=(0), b)))
end

mirrored(Y) = all(isequal(Y[i, j], Y[j, i]) for i in axes(Y, 1), j in axes(Y, 2))
offdiagonal_count(Y) = count(!iszero, Y[i, j] for i in axes(Y, 1), j in axes(Y, 2) if i != j)

# the checks that one input, one form and one budget pass
function check_case(f, s, X, best, Yexp, E, T, parent_of)
  p = size(X, 1)
  Y = fill(T(42), p, p)
  @test prox!(Y, f, X, 1.0) === Y
  @test isequal(Y, Yexp)
  @test exact_error(Y, X) == best
  @test offdiagonal_count(Y) <= s
  @test mirrored(Y)
  @test all(i -> isequal(Y[i, i], X[i, i]), 1:p)
  @test f(X) === (2 * E <= s ? zero(T) : T(Inf))
  @test f(Y) === zero(T)
  # the fused call returns the value of the stored output
  Y2 = fill(T(42), p, p)
  @test value_and_prox!(Y2, f, X, 0.5) === zero(T)
  @test isequal(Y2, Yexp)
  # the new output of prox
  @test isequal(prox(f, X), Yexp)
  # the input is the output
  P = parent_of()
  Z = P isa Symmetric ? parent(P) : P
  @test prox!(Z, f, P, 1.0) === Z
  @test isequal(Z, Yexp)
end

@testset "brute force reference, small matrices" begin
  rng = Xoshiro(20261119)
  for T in (Float32, Float64), p in 1:4, trial in 1:14
    density = (0.0, 0.3, 0.7, 1.0)[mod1(trial, 4)]
    pool = trial <= 8 ? T[1, 2, 3] : T.(round.(abs.(randn(rng, 6)); digits = 2) .+ 1)
    Xm, PL, PU = make_input(rng, T, p, density, pool)
    for s in budgets(p)
      f = IndBallL0(s; diagonal = false)
      @test f.s == s
      best, Yexp, E = brute(Xm, s)
      SL = Symmetric(PL, :L)
      check_case(f, s, SL, best, Yexp, E, T, () -> Symmetric(copy(PL), :L))
      SU = Symmetric(PU, :U)
      check_case(f, s, SU, best, Yexp, E, T, () -> Symmetric(copy(PU), :U))
    end
  end
end

@testset "worked examples and the tie rule" begin
  X = [5.0 1 -3; 1 7 2; -3 2 9]
  Y1 = [5.0 0 -3; 0 7 0; -3 0 9]
  for s in (2, 3)
    f = IndBallL0(s; diagonal = false)
    Y = prox(f, Symmetric(X, :L))
    @test isequal(Y, Y1)
    @test all(x -> x === 0.0, Y[[2, 4, 6, 8]])
    @test f(X) == Inf
    @test f(Y) == 0
  end
  for s in (6, 7, typemax(Int))
    f = IndBallL0(s; diagonal = false)
    @test isequal(prox(f, Symmetric(X, :L)), X)
    @test f(X) === 0.0
  end
  @test isequal(prox(IndBallL0(0; diagonal = false), Symmetric(X, :L)), [5.0 0 0; 0 7 0; 0 0 9])
  @test IndBallL0(1; diagonal = false)(X) == Inf

  Xt = [0.0 2 2; 2 0 -2; 2 -2 0]
  @test isequal(prox(IndBallL0(2; diagonal = false), Symmetric(Xt, :L)), [0.0 2 0; 2 0 0; 0 0 0])
  @test isequal(prox(IndBallL0(4; diagonal = false), Symmetric(Xt, :L)), [0.0 2 2; 2 0 0; 2 0 0])
  @test isequal(prox(IndBallL0(6; diagonal = false), Symmetric(Xt, :L)), Xt)

  # Symmetric(A, :U) reads the upper triangle of A
  A = [1.0 2 3; 99 4 5; 98 97 6]
  @test isequal(prox(IndBallL0(2; diagonal = false), Symmetric(A, :U)), [1.0 0 0; 0 4 5; 0 5 6])
  @test isequal(prox(IndBallL0(2; diagonal = false), Symmetric(A, :L)), [1.0 99 0; 99 4 0; 0 0 6])

  # The same raw matrix can retain just one position of a mirrored pair.
  @test prox(IndBallL0(1; diagonal = false), X) == [5.0 0 0; 0 7 0; -3 0 9]
  @test prox(IndBallL0(1; diagonal = false), Symmetric(X)) == Diagonal(diag(X))
  # a larger budget than the number of pairs, and a one-by-one matrix
  @test isequal(prox(IndBallL0(10^6; diagonal = false), Symmetric(X, :L)), X)
  @test isequal(prox(IndBallL0(0; diagonal = false), [3.0;;]), [3.0;;])
  @test IndBallL0(0; diagonal = false)([3.0;;]) === 0.0
  @test IndBallL0(0; diagonal = false)(Float32[3;;]) === 0.0f0
end

@testset "signed zeros" begin
  f0 = IndBallL0(0; diagonal = false)
  # Both output slots take the signed zero from the selected lower triangle.
  X = [1.0 -0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
  Y = prox(f0, Symmetric(X, :L))
  @test Y[2, 1] === 0.0 && Y[1, 2] === 0.0
  @test f0(X) === 0.0
  # a kept -0.0 entry is read from the lower entry
  X = [1.0 0.0 0.0; -0.0 1.0 0.0; 0.0 0.0 1.0]
  Y = prox(f0, Symmetric(X, :L))
  @test Y[2, 1] === -0.0 && Y[1, 2] === -0.0
  # an entry that is dropped becomes +0.0
  f2 = IndBallL0(2; diagonal = false)
  X = [1.0 -0.0 3.0; -0.0 1.0 2.0; 3.0 2.0 1.0]
  Y = prox(f2, Symmetric(X, :L))
  @test isequal(Y, [1.0 0.0 3.0; 0.0 1.0 0.0; 3.0 0.0 1.0])
  @test Y[2, 1] === 0.0 && Y[1, 2] === 0.0
  # a diagonal -0.0 is copied
  X = [-0.0 3.0; 3.0 -0.0]
  @test isequal(prox(f0, Symmetric(X, :L)), [-0.0 0.0; 0.0 -0.0])
  @test isequal(prox(f2, Symmetric(X, :L)), X)
end

@testset "value" begin
  f = IndBallL0(2; diagonal = false)
  for T in (Float32, Float64)
    @test f(zeros(T, 3, 3)) === zero(T)
    @test f(T[1 0; 0 1]) === zero(T)
    @test f(T[1 2; 2 1]) === zero(T)
    @test f(T[1 2 0; 2 1 0; 0 0 1]) === zero(T)
    @test f(T[1 2 3; 2 1 0; 3 0 1]) === T(Inf)
    @test IndBallL0(1; diagonal = false)(T[1 2; 2 1]) === T(Inf)
    @test IndBallL0(0; diagonal = false)(T[1 0; 0 1]) === zero(T)
    @test IndBallL0(0; diagonal = false)(T[1 0; 0 NaN]) === zero(T)    # the diagonal is not read
    @test isnan(f(T[1 NaN; NaN 1]))
    @test isnan(f(T[1 NaN 3; NaN 1 4; 3 4 1]))                            # NaN wins over an infeasible count
    @test isnan(IndBallL0(typemax(Int); diagonal = false)(T[1 NaN; NaN 1]))
    @test f(T[1 Inf; Inf 1]) === zero(T)                                  # infinity counts as nonzero
    @test IndBallL0(0; diagonal = false)(T[1 -Inf; -Inf 1]) === T(Inf)
    @test f(T[NaN 0; 0 NaN]) === zero(T)
    @test f(Symmetric(T[1 2 3; 99 1 4; 98 97 1], :U)) === T(Inf)
    @test f(Symmetric(T[1 2 0; 99 1 0; 98 97 1], :U)) === zero(T)
    @test f(T[1 2; 3 4]) === zero(T)
    @test IndBallL0(1; diagonal = false)(T[1 0; 1e-30 1]) === zero(T)
    @test f(zeros(T, 2, 3)) === zero(T)
    @test f(zeros(T, 0, 0)) === zero(T)
  end
  @test isnan(f([1.0 NaN; NaN 1.0]))
  @test isnan(f([1.0 NaN; 0.0 1.0]))
end

@testset "errors precede writes" begin
  f = IndBallL0(2; diagonal = false)
  X = [5.0 1 -3; 1 7 2; -3 2 9]
  function refused(ErrorType, Y, f, X, γ = 1.0)
    fill!(Y, 42)
    @test_throws ErrorType prox!(Y, f, X, γ)
    @test all(==(42), Y)
    fill!(Y, 42)
    @test_throws ErrorType value_and_prox!(Y, f, X, γ)
    @test all(==(42), Y)
  end
  message(g) = try; g(); "no error"; catch e; sprint(showerror, e); end

  Y = zeros(3, 3)
  for γ in (-1.0, -0.5, NaN, Inf, -Inf, prevfloat(0.0), -1, NaN32, Inf32, -1//2)
    refused(ArgumentError, Y, f, X, γ)
    refused(ArgumentError, Y, f, Symmetric(X, :L), γ)
    @test_throws ArgumentError prox(f, X, γ)
  end
  @test occursin("finite and nonnegative", message(() -> prox!(Y, f, X, Inf)))
  # step zero, negative zero and a rational step project
  for γ in (0.0, -0.0, 0, 1//2, big"2.5", 1e300, floatmax(), Float16(2), UInt8(3))
    fill!(Y, 42)
    @test isequal(prox!(Y, f, X, γ), [5.0 0 -3; 0 7 0; -3 0 9])
  end

  # Nonfinite counted entries are classified during the support count.
  for budget in (0, 2, 4, typemax(Int)), bad in (NaN, Inf, -Inf), (i, j) in lower_pairs(3)
    g = IndBallL0(budget; diagonal = false)
    Xb = copy(X)
    Xb[i, j] = Xb[j, i] = bad
    refused(ArgumentError, Y, g, Xb)
    refused(ArgumentError, Y, g, Symmetric(Xb, :L))
    refused(ArgumentError, Y, g, Symmetric(Xb, :U))
    @test_throws ArgumentError prox(g, Xb)
  end
  # an entry of the unread triangle of a Symmetric is not read
  Xu = copy(X)
  Xu[1, 3] = NaN
  fill!(Y, 42)
  @test isequal(prox!(Y, f, Symmetric(Xu, :L), 1.0), [5.0 0 -3; 0 7 0; -3 0 9])
  # shapes
  @test_throws DimensionMismatch prox!(zeros(2, 2), f, X, 1.0)
  @test_throws DimensionMismatch prox!(zeros(3, 4), f, X, 1.0)
  @test_throws DimensionMismatch prox!(zeros(3, 3), f, zeros(3, 2), 1.0)
  @test_throws DimensionMismatch value_and_prox!(zeros(2, 2), f, X, 1.0)
  fill!(Y, 42)
  @test_throws DimensionMismatch prox!(Y, f, zeros(2, 2), 1.0)
  @test all(==(42), Y)

  # unsupported arguments name the supported types
  Yf = fill(Float32(42), 3, 3)
  refused(ArgumentError, Yf, f, X)                            # Float32 output for Float64 input
  Yd = fill(42.0, 3, 3)
  refused(ArgumentError, Yd, f, Float32.(X))                  # Float64 output for Float32 input
  refused(ArgumentError, Yd, f, Hermitian(X))
  refused(ArgumentError, Yd, f, view(X, :, :))
  refused(ArgumentError, Yd, f, sparse(X))
  refused(ArgumentError, Yd, f, round.(Int, X))
  refused(ArgumentError, Yd, f, BigFloat.(X))
  refused(ArgumentError, Yd, f, Symmetric(round.(Int, X)))
  refused(ArgumentError, Yd, f, ComplexF64.(X))
  refused(ArgumentError, Yd, f, X, ones(3, 3))                # an array step
  refused(ArgumentError, Yd, f, X, fill(-1.0, 3, 3))
  refused(ArgumentError, Yd, f, X, ones(3))
  refused(ArgumentError, Yd, f, X, 1:3)
  Yv = fill(42.0, 9)
  @test_throws ArgumentError prox!(Yv, f, X, 1.0)
  @test all(==(42), Yv)
  @test_throws ArgumentError prox!(Yd, f, vec(X), 1.0)
  @test_throws ArgumentError prox!(view(Yd, :, :), f, X, 1.0)
  @test_throws ArgumentError prox!(Symmetric(Yd), f, X, 1.0)
  @test_throws ArgumentError prox!(Yd, f, X, "1")
  @test_throws ArgumentError prox(f, X, ones(3, 3))
  @test_throws ArgumentError prox(f, Hermitian(X))
  @test_throws ArgumentError prox(f, round.(Int, X))
  @test_throws ArgumentError prox(f, BigFloat.(X))
  @test_throws ArgumentError prox(f, (X,))
  @test_throws ArgumentError f(Hermitian(X))
  @test_throws ArgumentError f(round.(Int, X))
  @test_throws ArgumentError f(BigFloat.(X))
  @test_throws ArgumentError f(view(X, :, :))
  @test_throws ArgumentError f(sparse(X))
  @test_throws ArgumentError f(1.0)
  @test_throws ArgumentError f(zeros(2, 2, 2))
  @test all(==(42), Yd)
  for T in (Float32, Float64)
    msg = message(() -> prox!(zeros(T, 3), f, zeros(T, 3, 1), one(T)))
    @test occursin("Float32", msg) && occursin("Float64", msg)
  end

  # SymmetricSparseIterate: an input, an output, an array step
  ssi = SymmetricSparseIterate(Float64[8 1 0; 1 8 0; 0 0 8])
  before = collect(matrixview(ssi))
  @test_throws ArgumentError prox(f, ssi, ones(3, 3))
  @test_throws ArgumentError prox!(zeros(3, 3), f, ssi, 1.0)
  @test_throws ArgumentError prox!(ssi, f, X, 1.0)
  @test_throws ArgumentError prox!(ssi, f, X, ones(3, 3))
  @test_throws ArgumentError prox!(ssi, f, X)
  @test_throws ArgumentError value_and_prox!(zeros(3, 3), f, ssi, 1.0)
  @test_throws ArgumentError value_and_prox!(ssi, f, X, 1.0)
  @test_throws ArgumentError value_and_prox!(ssi, f, X, ones(3, 3))
  @test_throws ArgumentError value_and_prox!(ssi, f, X)
  @test collect(matrixview(ssi)) == before

  # constructor
  @test_throws ArgumentError IndBallL0(-1; diagonal = false)
  @test_throws ArgumentError IndBallL0(-big(2)^70; diagonal = false)
  @test_throws MethodError IndBallL0(2.5; diagonal = false)
  @test_throws MethodError IndBallL0(; diagonal = false)
end

# brute force only for the sizes it can enumerate
brute_or_nothing(A, s) = size(A, 1) <= 4 ? brute(A, s) : (nothing, nothing, nothing)

@testset "sizes and budgets" begin
  X = [5.0 1 -3; 1 7 2; -3 2 9]
  @test IndBallL0(big(2)^70; diagonal = false).s == typemax(Int)
  @test IndBallL0(typemax(Int); diagonal = false).s == typemax(Int)
  @test IndBallL0(UInt8(3); diagonal = false).s == 3
  @test IndBallL0(UInt64(typemax(UInt64)); diagonal = false).s == typemax(Int)
  for s in (big(2)^70, typemax(Int), typemax(Int) - 1, 6, 7)
    f = IndBallL0(s; diagonal = false)
    @test isequal(prox(f, X), X)
    @test f(X) === 0.0
  end
  @test isequal(prox(IndBallL0(4; diagonal = false), Symmetric(X, :L)), [5.0 0 -3; 0 7 2; -3 2 9])
  @test isequal(prox(IndBallL0(5; diagonal = false), Symmetric(X, :L)), [5.0 0 -3; 0 7 2; -3 2 9])
  # one object serves different sizes and element types
  f = IndBallL0(4; diagonal = false)
  for p in (3, 5, 2, 7), T in (Float32, Float64, Float32)
    A = T[min(i, j) * (-1)^(i + j) for i in 1:p, j in 1:p]
    best, Yexp, E = brute_or_nothing(A, 4)
    Y = prox(f, Symmetric(A, :L))
    @test Y isa Matrix{T}
    best === nothing || @test isequal(Y, Yexp)
    @test offdiagonal_count(Y) <= 4
  end
end

@testset "magnitudes without squaring" begin
  rng = Xoshiro(1057)
  for T in (Float32, Float64)
    pool = T[floatmax(T), prevfloat(floatmax(T)), floatmax(T) / 2, floatmax(T) / 4, one(T),
             floatmin(T), prevfloat(floatmin(T)), 2 * nextfloat(zero(T)), nextfloat(zero(T)), zero(T)]
    for trial in 1:60, p in (3, 4)
      Xm = zeros(T, p, p)
      for c in 1:p
        Xm[c, c] = rand(rng, pool) * rand(rng, (-1, 1))
        for r in c+1:p
          v = rand(rng, pool) * rand(rng, (-1, 1))
          Xm[r, c] = v
          Xm[c, r] = v
        end
      end
      for s in budgets(p)
        best, Yexp, E = brute(Xm, s)
        f = IndBallL0(s; diagonal = false)
        Y = prox(f, Symmetric(Xm, :L))
        @test isequal(Y, Yexp)
        @test exact_error(Y, Xm) == best
        @test f(Y) === zero(T)
      end
    end
    # the squares of these overflow or underflow
    huge = floatmax(T)
    tiny = nextfloat(zero(T))
    X = T[0 huge tiny; huge 0 2*tiny; tiny 2*tiny 0]
    @test isequal(prox(IndBallL0(2; diagonal = false), Symmetric(X, :L)), T[0 huge 0; huge 0 0; 0 0 0])
    X = T[0 tiny tiny; tiny 0 2*tiny; tiny 2*tiny 0]
    @test isequal(prox(IndBallL0(2; diagonal = false), Symmetric(X, :L)), T[0 0 0; 0 0 2*tiny; 0 2*tiny 0])
    @test isequal(prox(IndBallL0(4; diagonal = false), Symmetric(X, :L)), T[0 tiny 0; tiny 0 2*tiny; 0 2*tiny 0])
  end
end

# The expected output from a sort of the strict lower magnitudes by (-|x|, index):
# every pair when E ≤ k, otherwise the first k of the sorted order.
function sort_reference(A::Matrix{T}, s) where {T}
  p = size(A, 1)
  pairs = lower_pairs(p)
  mags = [abs(A[i, j]) for (i, j) in pairs]
  k = min(s ÷ 2, length(pairs))
  order = sort(eachindex(mags); by = i -> (-mags[i], i))
  keep = count(!iszero, mags) <= k ? collect(1:length(pairs)) : sort(order[1:k])
  Yexp = zeros(T, p, p)
  for c in 1:p
    Yexp[c, c] = A[c, c]
  end
  for idx in keep
    i, j = pairs[idx]
    Yexp[i, j] = A[i, j]
    Yexp[j, i] = A[i, j]
  end
  Yexp
end

# the matrix whose strict lower entries in column-major order are g(idx, n)
function from_lower(::Type{T}, p, g) where {T}
  n = p * (p - 1) ÷ 2
  A = zeros(T, p, p)
  idx = 0
  for c in 1:p, r in c+1:p
    idx += 1
    A[r, c] = A[c, r] = g(idx, n)
  end
  for c in 1:p
    A[c, c] = T(c)
  end
  A
end

@testset "selection on larger matrices" begin
  rng = Xoshiro(7)
  for T in (Float32, Float64), p in (6, 11, 25), pool in (T[1, 2], T[1, 2, 3, 4, 5, 6], T.(1:200))
    A = zeros(T, p, p)
    for c in 1:p, r in c:p
      v = r == c ? T(rand(rng, 0:3)) : rand(rng, pool) * rand(rng, (-1, 1, 0))
      A[r, c] = v
      A[c, r] = v
    end
    for s in (0, 1, 2, 5, 10, 40, 100, p * (p - 1) - 2, p * (p - 1))
      @test isequal(prox(IndBallL0(s; diagonal = false), Symmetric(A, :L)), sort_reference(A, s))
    end
  end
end

@testset "selection on structured orderings" begin
  for T in (Float32, Float64), p in (30, 47, 60)
    inputs = (
      "organ pipe" => from_lower(T, p, (i, n) -> T(min(i, n - i + 1))),
      "ascending" => from_lower(T, p, (i, n) -> T(i)),
      "descending" => from_lower(T, p, (i, n) -> T(n - i + 1)),
      "alternating signs" => from_lower(T, p, (i, n) -> T(iseven(i) ? min(i, n - i + 1) : -min(i, n - i + 1))),
      "Toeplitz" => T[0.5^abs(i - j) for i in 1:p, j in 1:p],
      "distance" => T[abs(i - j) for i in 1:p, j in 1:p],
      "distance with ties and zeros" => T[abs(i - j) % 4 for i in 1:p, j in 1:p])
    for (name, A) in inputs, s in (0, 2, 10, 2p, 20p, p * (p - 1) - 2, p * (p - 1))
      Y = prox(IndBallL0(s; diagonal = false), Symmetric(A, :L))
      @test isequal(Y, sort_reference(A, s))
    end
  end
end

@testset "traits, equality and printing" begin
  f = IndBallL0(4; diagonal = false)
  @test isproximable(f) && isproximable(typeof(f))
  @test !isconvex(f) && !isconvex(typeof(f))
  @test ProximalBase._isindicator(f)
  @test !issmooth(f) && !isseparable(f)
  @test sprint(show, f) == "IndBallL0(4; diagonal = false)"
  @test f == IndBallL0(4; diagonal = false)
  @test f != IndBallL0(4)
  @test IndBallL0(4) == IndBallL0(4; diagonal = true)
  @test sprint(show, IndBallL0(4)) == "IndBallL0(4; diagonal = true)"
  @test isempty(Test.detect_ambiguities(ProximalBase; recursive = true))
  @test f != IndBallL0(2; diagonal = false)
  @test hash(f) == hash(IndBallL0(4; diagonal = false))
  @test hash(f) != hash(IndBallL0(2; diagonal = false))
  @test IndBallL0(big(2)^70; diagonal = false) == IndBallL0(typemax(Int); diagonal = false)
  @test Scaled(f, 0) === f
  @test_throws MethodError kkt_violation(f, zeros(3, 3), zeros(3, 3), 1)
  @test_throws MethodError prox_block(f, zeros(3), 1, 1.0)
  @test_throws MethodError restrict(f, [1])
  @test_throws ArgumentError lipschitz(f)
end

@testset "three-argument calls and returns" begin
  f = IndBallL0(2; diagonal = false)
  X = [5.0 1 -3; 1 7 2; -3 2 9]
  Y1 = [5.0 0 -3; 0 7 0; -3 0 9]
  Y = zeros(3, 3)
  @test prox!(Y, f, X) === Y
  @test isequal(Y, Y1)
  @test isequal(prox(f, X), Y1)
  @test isequal(prox(f, X, 2), Y1)
  @test isequal(prox(f, Symmetric(X)), Y1)
  @test prox(f, Symmetric(X)) isa Matrix{Float64}
  @test prox(f, Float32.(X)) isa Matrix{Float32}
  Y .= 0
  @test value_and_prox!(Y, f, X) === 0.0
  @test isequal(Y, Y1)
  @test value_and_prox!(Y, f, X, 1) === 0.0
  Yf = zeros(Float32, 3, 3)
  @test value_and_prox!(Yf, f, Float32.(X)) === 0.0f0
  # in place on a Matrix, and on the parent of a Symmetric of either triangle
  Z = copy(X)
  @test prox!(Z, f, Z) === Z
  @test isequal(Z, prox(f, X))
  for uplo in (:L, :U)
    P = copy(X)
    S = Symmetric(P, uplo)
    @test prox!(P, f, S) === P
    @test isequal(P, Y1)
    P = [5.0 1 -3; 1 7 2; -3 2 9]
    P[uplo == :L ? 1 : 3, uplo == :L ? 3 : 1] = 1000     # the triangle that is not read
    S = Symmetric(P, uplo)
    @test value_and_prox!(P, f, S, 1.0) === 0.0
    @test isequal(P, Y1)
  end
  # an aliased input that is already feasible
  Z = copy(Y1)
  @test prox!(Z, f, Z) === Z
  @test isequal(Z, Y1)
end

@testset "Scaled" begin
  f = IndBallL0(2; diagonal = false)
  X = [5.0 1 -3; 1 7 2; -3 2 9]
  Y1 = [5.0 0 -3; 0 7 0; -3 0 9]
  Y = zeros(3, 3)
  h = Scaled(f, 3.0)
  @test isequal(prox(h, X, 0.5), Y1)
  @test isequal(prox!(fill!(Y, 42), h, X, 1.0), Y1)
  @test h(X) == Inf
  @test h(Y1) == 0
  @test value_and_prox!(fill!(Y, 42), h, X, 1.0) == 0
  @test isequal(Y, Y1)
  @test Scaled(f, 0) === f
  @test 0 * f === f
  # an infinite step is refused, also as the product of the scale and the step
  Yc = fill(42.0, 3, 3)
  hinf = Scaled(f, Inf)
  @test_throws ArgumentError prox!(Yc, hinf, X, 1.0)
  @test all(==(42), Yc)
  @test_throws ArgumentError value_and_prox!(Yc, hinf, X, 1.0)
  @test all(==(42), Yc)
  hbig = Scaled(f, floatmax())
  @test_throws ArgumentError prox!(Yc, hbig, X, 2.0)
  @test all(==(42), Yc)
  @test_throws ArgumentError value_and_prox!(Yc, hbig, X, 2.0)
  @test all(==(42), Yc)
  @test_throws ArgumentError prox!(Yc, hbig, X, Inf)
  @test all(==(42), Yc)
  # a zero step stays zero for an indicator
  @test isequal(prox!(fill!(Yc, 42), hinf, X, 0.0), Y1)
  @test isequal(prox!(fill!(Yc, 42), hbig, X, 0.0), Y1)
  # a negative or NaN step
  @test_throws ArgumentError prox!(Yc, h, X, -1.0)
  @test_throws ArgumentError prox!(Yc, h, X, NaN)
  # two factors with a normal product merge into one wrapper
  @test isequal(prox(Scaled(Scaled(f, 2.0), 3.0), X), Y1)
  # a chain whose factor product overflows passes the step γ and projects
  chain = Scaled(Scaled(f, floatmax()), 4.0)
  @test isequal(prox!(fill!(Yc, 42), chain, X, 1.0), Y1)
  @test value_and_prox!(fill!(Yc, 42), chain, X, 1.0) === 0.0
  @test isequal(Yc, Y1)
  @test isequal(prox(Scaled(f, 2.0), Symmetric(X)), Y1)
end

# Each helper calls once before it measures, so that the buffer of the object and the compiled methods exist.
function alloc_prox(Y, f, X)
  prox!(Y, f, X, 1.0)
  @allocated prox!(Y, f, X, 1.0)
end
function alloc_inplace(P, P0, f, uplo)
  S = Symmetric(P, uplo)
  copyto!(P, P0)
  prox!(P, f, S, 1.0)
  copyto!(P, P0)
  @allocated prox!(P, f, S, 1.0)
end
function alloc_matrix_inplace(P, P0, f)
  copyto!(P, P0)
  prox!(P, f, P, 1.0)
  copyto!(P, P0)
  @allocated prox!(P, f, P, 1.0)
end
function alloc_value(f, X)
  f(X)
  @allocated f(X)
end

@testset "no allocation in repeated calls" begin
  rng = Xoshiro(99)
  for T in (Float32, Float64), p in (3, 8, 30)
    A = zeros(T, p, p)
    for c in 1:p, r in c:p
      A[r, c] = A[c, r] = T(rand(rng, -3:3))
    end
    PL = copy(A)
    PU = copy(A)
    for c in 1:p, r in c+1:p
      PL[c, r] = T(77)
      PU[r, c] = T(77)
    end
    n = p * (p - 1) ÷ 2
    for s in (0, 2, p, 2 * (n ÷ 2), typemax(Int))     # projections that drop pairs and that keep all
      f = IndBallL0(s; diagonal = false)
      Y = zeros(T, p, p)
      for X in (A, Symmetric(PL, :L), Symmetric(PU, :U))
        @test alloc_prox(Y, f, X) == 0
        @test alloc_value(f, X) == 0
      end
      @test alloc_matrix_inplace(copy(A), A, f) == 0
      @test alloc_inplace(copy(PL), PL, f, :L) == 0
      @test alloc_inplace(copy(PU), PU, f, :U) == 0
    end
  end
end

end
