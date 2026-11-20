module IndBallL0Test

using Test, ProximalBase, LinearAlgebra, Random, SparseArrays

# ---- Scaled returns the indicator (the same checks for each built-in indicator) -----------------------------
const CANON_FACTORS = Any[2, 0, 1, 0.5, 2.0, 2f0, 3f0, 5e-324, 1e-300, floatmax(Float64), floatmax(Float32), nextfloat(0f0),
                          Inf, Inf32, 0.0, -0.0, big(2.0), big(0), big(Inf)]
const CANON_INVALID = Any[-1, -1.0, NaN, -Inf, -1f0, NaN32, -5e-324, -Inf32, big(-1.0), big(NaN)]
scaled_allocs(f, α) = (Scaled(f, α); @allocated Scaled(f, α))
times_allocs(f, α) = (α * f; @allocated α * f)

# Every valid factor, nested chains whatever their product, and the invalid factors; construction is inferred
# and does not allocate.
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
  for γ in (-1.0, -0.5, NaN, -Inf, prevfloat(0.0), -1, NaN32, -Inf32, -1//2, -nextfloat(0f0))
    refused(ArgumentError, Y, f, X, γ)
    refused(ArgumentError, Y, f, Symmetric(X, :L), γ)
    @test_throws ArgumentError prox(f, X, γ)
  end
  for γ in (-1.0, NaN, -Inf)
    msg = message(() -> prox!(Y, f, X, γ))
    @test occursin("must be nonnegative", msg) && !occursin("finite", msg)
  end
  # step zero, negative zero, an infinite step and a rational step project
  for γ in (0.0, -0.0, 0, 1//2, big"2.5", 1e300, floatmax(), Float16(2), UInt8(3), Inf, Inf32, big(Inf), Float16(Inf))
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
  for mixed in (Union{Float32, Float64}[1.0f0, 2.0], Union{Float32, Float64}[1.0f0 2.0])
    y = copy(mixed)
    whole = IndBallL0(2)
    @test_throws ArgumentError whole(mixed)
    @test_throws ArgumentError prox(whole, mixed)
    @test_throws ArgumentError prox!(y, whole, mixed, 1.0)
    @test_throws ArgumentError value_and_prox!(y, whole, mixed, 1.0)
    @test isequal(y, mixed)
  end
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

@testset "Scaled returns the indicator" begin
  for f in (IndBallL0(2; diagonal = false), IndBallL0(0), IndBallL0(3), IndBallL0(typemax(Int); diagonal = false))
    check_canonical(f)
  end
  f = IndBallL0(2; diagonal = false)
  X = [5.0 1 -3; 1 7 2; -3 2 9]
  Y1 = [5.0 0 -3; 0 7 0; -3 0 9]
  Y = zeros(3, 3)
  h = Scaled(f, 3.0)
  @test h === f && 3.0 * f === f && f * 3.0 === f
  @test isequal(prox(h, X, 0.5), Y1)
  @test isequal(prox!(fill!(Y, 42), h, X, 1.0), Y1)
  @test h(X) == Inf
  @test h(Y1) == 0
  @test value_and_prox!(fill!(Y, 42), h, X, 1.0) == 0
  @test isequal(Y, Y1)
  @test Scaled(f, 0) === f
  @test 0 * f === f
  # an infinite or overflowing scale returns the indicator, and the step is the caller's
  Yc = fill(42.0, 3, 3)
  hinf = Scaled(f, Inf)
  hbig = Scaled(f, floatmax())
  @test hinf === f && hbig === f
  for hh in (hinf, hbig), γ in (0.0, 1.0, 2.0, Inf)
    @test isequal(prox!(fill!(Yc, 42), hh, X, γ), Y1)
    @test value_and_prox!(fill!(Yc, 42), hh, X, γ) === 0.0
    @test isequal(Yc, Y1)
  end
  # a negative or NaN step is refused before writing, whatever the scale
  for α in (3.0, Inf, floatmax(), 1e-300, 0.5), γ in (-1.0, NaN, -Inf, -nextfloat(0.0), -1e-300)
    fill!(Yc, 42)
    @test_throws ArgumentError prox!(Yc, α * f, X, γ)
    @test all(==(42), Yc)
    @test_throws ArgumentError value_and_prox!(Yc, α * f, X, γ)
    @test all(==(42), Yc)
  end
  # nested factors, whatever their product, return the indicator
  @test Scaled(Scaled(f, 2.0), 3.0) === f
  @test isequal(prox(Scaled(Scaled(f, 2.0), 3.0), X), Y1)
  chain = Scaled(Scaled(f, floatmax()), 4.0)
  @test chain === f && 0 * (Inf * f) === f
  @test isequal(prox!(fill!(Yc, 42), chain, X, 1.0), Y1)
  @test value_and_prox!(fill!(Yc, 42), chain, X, 1.0) === 0.0
  @test isequal(Yc, Y1)
  @test isequal(prox(Scaled(f, 2.0), Symmetric(X)), Y1)
end

@testset "an infinite step projects" begin
  @test prox(IndBallL0(1), [3.0, -1.0], Inf) == [3.0, 0.0]
  @test prox(IndBallL0(1), Float32[3, -1], Inf32) == Float32[3, 0]
  rng = Xoshiro(1080)
  for T in (Float32, Float64), diagonal in (true, false), s in (0, 1, 2, 5, typemax(Int))
    f = IndBallL0(s; diagonal)
    A = T.(rand(rng, -3:3, 4, 4))
    S = Symmetric(Matrix(Symmetric(A, :L)), :L)
    sp = SparseIterate(T, 12)
    for k in (1, 3, 4, 9, 12)
      activate!(sp, k); coordinates(sp)[k] = T(k) * (isodd(k) ? 1 : -1)
    end
    spm = SparseIterate(T, 4, 4)
    for k in (1, 6, 7, 11, 16)
      activate!(spm, k); coordinates(spm)[k] = T(k) * (isodd(k) ? 1 : -1)
    end
    ss = SymmetricSparseIterate(T, 4)
    for k in (1, 2, 5, 7, 10)
      activate!(ss, k); coordinates(ss)[k] = T(k) * (isodd(k) ? 1 : -1)
    end
    for x in (T.(rand(rng, -3:3, 9)), A, S, Symmetric(A, :U), sp, spm, ss)
      y1 = x isa Symmetric ? similar(Matrix(x)) : similar(x)
      y2 = copy(y1)
      @test prox!(y1, f, x, one(T)) === y1
      @test prox!(y2, f, x, T(Inf)) === y2
      @test Array(y2) == Array(y1)
      @test typeof(prox(f, x, T(Inf))) === typeof(prox(f, x, one(T)))
      @test Array(prox(f, x, T(Inf))) == Array(prox(f, x, one(T)))
      if x isa Union{SparseIterate, SymmetricSparseIterate}
        @test collect(activecoordinates(y2)) == collect(activecoordinates(y1))
      end
      v1 = value_and_prox!(y1, f, x, one(T))
      v2 = value_and_prox!(y2, f, x, T(Inf))
      @test v1 === zero(T) && v2 === zero(T)
      @test Array(y2) == Array(y1)
      # the infinite step passes through a scale unchanged
      y3 = copy(y1)
      @test value_and_prox!(y3, 2 * f, x, T(Inf)) === zero(T) && Array(y3) == Array(y1)
    end
    # errors precede writes at an invalid step
    for γ in (-one(T), T(NaN), -nextfloat(zero(T)), T(-Inf))
      y = fill(T(42), 4, 4)
      @test_throws ArgumentError prox!(y, f, A, γ)
      @test all(==(T(42)), y)
      @test_throws ArgumentError value_and_prox!(y, f, A, γ)
      @test all(==(T(42)), y)
      @test_throws ArgumentError prox!(y, f, S, γ)
      @test all(==(T(42)), y)
      ys = SparseIterate(T, 4, 4)
      activate!(ys, 2); coordinates(ys)[2] = T(42)
      @test_throws ArgumentError prox!(ys, f, spm, γ)
      @test_throws ArgumentError value_and_prox!(ys, f, spm, γ)
      @test coordinates(ys)[2] == T(42) && nactive(ys) == 1
      yss = SymmetricSparseIterate(T, 4)
      activate!(yss, 2); coordinates(yss)[2] = T(42)
      @test_throws ArgumentError prox!(yss, f, ss, γ)
      @test_throws ArgumentError value_and_prox!(yss, f, ss, γ)
      @test coordinates(yss)[2] == T(42) && nactive(yss) == 1
    end
  end
  # a zero step and negative zero project, as before
  @test prox(IndBallL0(1), [3.0, -1.0], 0.0) == [3.0, 0.0] && prox(IndBallL0(1), [3.0, -1.0], -0.0) == [3.0, 0.0]
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
