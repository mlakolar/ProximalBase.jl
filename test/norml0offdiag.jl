module NormL0DiagonalFreeTest

using Test, ProximalBase, LinearAlgebra, Random, SparseArrays, OffsetArrays

# The exact value of a finite float or an integer as a rational number.
function exact(x::AbstractFloat)
  iszero(x) && return zero(Rational{BigInt})
  m, e = frexp(x)
  n = BigInt(ldexp(m, precision(x)))
  k = e - precision(x)
  n * (k >= 0 ? Rational{BigInt}(big(2)^k) : 1 // big(2)^(-k))
end
exact(x::Integer) = Rational{BigInt}(x)

# g(Z) + ‖Z - V‖²_F / (2γ) over the whole matrix, in exact arithmetic.
function objective(Z, V, L, G)
  p = size(V, 1)
  squares = zero(L)
  count = 0
  for j in 1:p, i in 1:p
    d = Z[i, j] - V[i, j]
    squares += d * d
    count += i != j && !iszero(Z[i, j])
  end
  L * count + squares / (2G)
end

# Minimum of the objective over the symmetric matrices whose diagonal is that of V and
# whose pair (i, j) holds either V[i, j] or zero in both positions.
function best_objective(V, L, G)
  p = size(V, 1)
  pairs = [(i, j) for j in 1:p for i in 1:j - 1]
  best = nothing
  for mask in 0:2^length(pairs) - 1
    Z = copy(V)
    for (b, (i, j)) in enumerate(pairs)
      if (mask >> (b - 1)) & 1 == 0
        Z[i, j] = zero(L)
        Z[j, i] = zero(L)
      end
    end
    h = objective(Z, V, L, G)
    best = best === nothing ? h : min(best, h)
  end
  best
end

nonzero_offdiagonal(Y) = count(!iszero, Y) - count(!iszero, diag(Y))

function expected_value(::Type{T}, λ, k) where {T}
  (k == 0 || iszero(λ)) && return zero(T)
  R = promote_type(T, float(typeof(λ)))
  convert(T, R(λ) * R(k))
end

# Returns nothing when the oracles agree with the exact minimization over the keep or drop
# patterns of the pairs, and a description of the first failure otherwise.
function problem(V::Matrix{T}, λ, γ) where {T}
  p = size(V, 1)
  f = NormL0(λ; diagonal = false)
  X = copy(V)
  Y = fill(T(NaN), p, p)
  v = value_and_prox!(Y, f, X, γ)
  isequal(X, V) || return "input changed"
  all(isfinite, Y) || return "output not written"
  Yo = prox(f, V, γ)
  Yi = copy(V)
  prox!(Yi, f, Yi, γ)
  Yp = fill(T(NaN), p, p)
  prox!(Yp, f, V, γ) === Yp || return "prox! did not return its output"
  (isequal(Y, Yo) && isequal(Y, Yi) && isequal(Y, Yp)) || return "in-place or allocating result differs"
  all(i -> isequal(Y[i, i], V[i, i]), 1:p) || return "diagonal changed"
  Y == transpose(Y) || return "output not symmetric"
  for j in 1:p, i in 1:p
    i != j && !(iszero(Y[i, j]) || Y[i, j] == V[i, j]) && return "entry ($i, $j) is neither zero nor the input"
  end
  v isa T && v === f(Y) || return "fused value differs from the value at the output"
  L, G = exact(λ), exact(γ)
  Vq, Yq = exact.(V), exact.(Y)
  best = best_objective(Vq, L, G)
  objective(Yq, Vq, L, G) == best || return "not a minimizer"
  for j in 1:p, i in 1:j - 1
    iszero(Y[i, j]) && continue
    Z = copy(Yq)
    Z[i, j] = zero(L)
    Z[j, i] = zero(L)
    objective(Z, Vq, L, G) > best || return "pair ($i, $j) kept although dropping it does not cost more"
  end
  e = count(!iszero, Y[i, j] for j in 1:p for i in 1:j - 1)
  v == convert(T, 2 * L * e) || return "value is not 2λ times the number of kept pairs"
  nothing
end

function build(::Type{T}, diagonal, pairs) where {T}
  p = length(diagonal)
  X = zeros(T, p, p)
  for i in 1:p
    X[i, i] = diagonal[i]
  end
  k = 0
  for j in 1:p, i in 1:j - 1
    k += 1
    u, v = pairs[k]
    X[i, j] = u
    X[j, i] = v
  end
  X
end

# Values of a symmetric pair around the threshold: below, at and above it, with either sign, and the
# four ways to store a zero pair.
function pair_values(::Type{T}, tie) where {T}
  below, above = prevfloat(tie), nextfloat(tie)
  z = zero(T)
  [[(a, a) for a in T[below, -below, tie, -tie, above, -above]]; [(z, z), (z, -z), (-z, z), (-z, -z)]]
end

@testset "constructor, traits and types" begin
  f = NormL0(; diagonal = false)
  @test f.λ == 1 && f isa ProximalBase._NormL0OffDiag{Int} && f isa AbstractFunction
  @test NormL0(0.5f0; diagonal = false).λ === 0.5f0
  @test NormL0(0; diagonal = false).λ == 0
  for bad in (-1, -1.0, -0.5f0, NaN, -NaN, Inf, -Inf, Float32(Inf), Float32(NaN))
    @test_throws ArgumentError NormL0(bad; diagonal = false)
  end
  @test isproximable(f) && isproximable(typeof(f))
  @test !isconvex(f) && !issmooth(f) && !isseparable(f)
  @test !isconvex(typeof(f)) && !issmooth(typeof(f)) && !isseparable(typeof(f))
  @test !(f isa SeparableFunction)
end

@testset "documented examples, $T" for T in (Float32, Float64)
  # λ = 2, γ = 1/2: the threshold of the squared entry is 2.
  f = NormL0(T(2); diagonal = false)
  V = T[0.01 1.5 1.4 -1.5; 1.5 3 0 1.5; 1.4 0 0.5 0; -1.5 1.5 0 0]
  Y = prox(f, V, T(0.5))
  @test Y[1, 2] === T(1.5) && Y[2, 1] === T(1.5)
  @test isequal(Y[1, 3], zero(T)) && isequal(Y[3, 1], zero(T))
  @test Y[1, 4] === T(-1.5) && Y[4, 1] === T(-1.5)
  @test Y[2, 4] === T(1.5) && Y[4, 2] === T(1.5)
  @test Y[1, 1] === T(0.01) && Y[2, 2] === T(3) && Y[3, 3] === T(0.5) && isequal(Y[4, 4], zero(T))
  @test prox(NormL0(T(2)), V, T(0.5))[1, 1] === zero(T)

  # λ = 2, γ = 1: the threshold of the squared entry is 4.
  g = NormL0(T(2); diagonal = false)
  tie = T(2)
  for s in (1, -1), (x, kept) in ((tie, false), (nextfloat(tie), true), (prevfloat(tie), false))
    for p in (2, 9)
      M = Matrix{T}(I, p, p)
      M[1, 2] = M[2, 1] = s * x
      Z = prox(g, M, one(T))
      @test isequal(Z[1, 2], kept ? s * x : zero(T)) && isequal(Z[2, 1], Z[1, 2])
      @test Z[1, 1] === one(T)
    end
  end
end

@testset "brute force over keep and drop patterns, $T, λ = $λ, γ = $γ" for T in (Float32, Float64),
    (λ, γ, tie) in ((T(2), T(1), T(2)), (T(8), T(4), T(8)), (2, 1, T(2)), (T(2), 1, T(2)), (0, 1, T(2)), (T(0), T(1), T(2)))
  vals = pair_values(T, tie)
  for p in (2, 3), diagonal in (T[0.01, -2, 3], T[-0.0, 0.0, 1])
    d = diagonal[1:p]
    failures = String[]
    for pairs in Iterators.product(ntuple(_ -> vals, p * (p - 1) ÷ 2)...)
      msg = problem(build(T, d, pairs), λ, γ)
      msg === nothing || push!(failures, string(msg, ": ", d, " ", pairs))
    end
    @test isempty(failures)
  end
  rng = Xoshiro(20261119)
  failures = String[]
  for _ in 1:30
    d = rand(rng, T[0.01, -0.0, 0.0, -2, 3, 1e-3], 4)
    pairs = [rand(rng, vals) for _ in 1:6]
    msg = problem(build(T, d, pairs), λ, γ)
    msg === nothing || push!(failures, string(msg, ": ", d, " ", pairs))
  end
  @test isempty(failures)
end

@testset "diagonal has no penalty, $T" for T in (Float32, Float64)
  f = NormL0(T(5); diagonal = false)
  for d in T[0, -0.0, 1e-3, 0.01, -2, 7], γ in T[0.1, 1, 10]
    Y = prox(f, fill(d, 1, 1), γ)
    @test isequal(Y[1, 1], d)
    G = exact(γ)
    cost(z) = (exact(z) - exact(d))^2 / (2G)
    @test all(z -> cost(z) >= cost(d), (zero(T), d / 2, 2d, -d, nextfloat(d), prevfloat(d)))
    @test f(Y) === zero(T)
  end
end

# A random matrix that is symmetric under ==, and entries planted at and next to the threshold t.
function symmetric_random(rng, ::Type{T}, p) where {T}
  A = randn(rng, T, p, p)
  A .+ A'
end
function plant!(rng, V::Matrix{T}, t) where {T}
  p = size(V, 1)
  for _ in 1:max(p, 4)
    i, j = rand(rng, 1:p), rand(rng, 1:p)
    i == j && continue
    x = rand(rng, (prevfloat(t), t, nextfloat(t))) * (rand(rng, Bool) ? 1 : -1)
    V[i, j] = V[j, i] = x
  end
  V
end

@testset "agreement with NormL0 off the diagonal, $T" for T in (Float32, Float64)
  rng = Xoshiro(1058)
  settings = ((T(0.3), T(1)), (T(0.5), 1), (1, T(0.4)), (0.3, T(1)), (Float32(0.3), 1.0), (2, 1), (T(0), T(1)),
              (T(0.3), T(2)), (T(1e-3), T(1)), (T(40), T(0.5)))
  for p in (3, 4, 8, 32, 64), (λ, γ) in settings
    V = symmetric_random(rng, T, p)
    plant!(rng, V, T(sqrt(2 * Float64(γ) * Float64(λ))))
    for i in 1:min(p, 3)
      V[i, i] = (T(0.01), T(-0.0), T(0.2))[i]
    end
    f = NormL0(λ; diagonal = false)
    W = prox(NormL0(λ), V, γ)
    Y = prox(f, V, γ)
    offdiag = [(i, j) for j in 1:p for i in 1:p if i != j]
    @test all(ij -> isequal(Y[ij...], W[ij...]), offdiag)
    @test all(i -> isequal(Y[i, i], V[i, i]), 1:p)
    @test Y == transpose(Y)
    Z = fill(T(NaN), p, p)
    v = value_and_prox!(Z, f, V, γ)
    @test isequal(Z, Y)
    @test v === f(Y) && v === expected_value(T, λ, nonzero_offdiagonal(Y))
    @test f(V) === expected_value(T, λ, nonzero_offdiagonal(V))
    U = copy(V)
    @test prox!(U, f, U, γ) === U && isequal(U, Y)
    U = copy(V)
    @test value_and_prox!(U, f, U, γ) === v && isequal(U, Y)
  end
  # About half of the entries are above the threshold.
  V = symmetric_random(rng, T, 64)
  Y = prox(NormL0(T(0.3); diagonal = false), V, T(1))
  @test 0.3 < nonzero_offdiagonal(Y) / (64 * 63) < 0.7
end

@testset "diagonal contrast with the all-entry penalty, $T" for T in (Float32, Float64)
  V = T[0.4 0.7 0.05; 0.7 2 0.3; 0.05 0.3 1.5]
  @test isposdef(V)
  λ, γ = T(0.1), T(1)
  W = prox(NormL0(λ), V, γ)
  Y = prox(NormL0(λ; diagonal = false), V, γ)
  @test isequal(W[1, 1], zero(T)) && W[2, 2] === T(2)
  @test Y[1, 1] === T(0.4) && Y[2, 2] === T(2) && Y[3, 3] === T(1.5)
  @test Y == T[0.4 0.7 0; 0.7 2 0; 0 0 1.5]
  @test isequal(Y[1, 3], zero(T)) && isequal(Y[2, 3], zero(T))
end

# A sentinel output, to see that a refused call does not write.
function refuses(E, f, Y, X, γ = 1; watch = (Y, X), value = false)
  before = map(copy, watch)
  @test_throws E prox!(Y, f, X, γ)
  @test_throws E value_and_prox!(Y, f, X, γ)
  value && @test_throws E f(X)
  @test all(map(isequal, before, watch))
end

@testset "refusals, $T" for T in (Float32, Float64)
  f = NormL0(T(1); diagonal = false)
  p = 5
  rng = Xoshiro(5)
  S = symmetric_random(rng, T, p)
  sentinel(n = p, m = n) = fill(T(7.5), n, m)

  @testset "asymmetric input gives the entrywise result" begin
    checks(X, γ) = begin
      n = size(X, 1)
      Xc = copy(X)
      W = prox(NormL0(f.λ), X, γ)
      Y = fill(T(NaN), n, n)
      v = value_and_prox!(Y, f, X, γ)
      k = nonzero_offdiagonal(Y)
      ok = isequal(X, Xc) && v === f(Y) && v === expected_value(T, f.λ, k)
      ok &= all(i -> isequal(Y[i, i], X[i, i]), 1:n)
      ok &= all(ij -> ij[1] == ij[2] || isequal(Y[ij...], W[ij...]), Iterators.product(1:n, 1:n))
      ok &= isequal(prox(f, X, γ), Y) && f(X) === expected_value(T, f.λ, nonzero_offdiagonal(X))
      Z = copy(X)
      ok &= prox!(Z, f, Z, γ) === Z && isequal(Z, Y)
      Z = copy(X)
      ok &= value_and_prox!(Z, f, Z, γ) === v && isequal(Z, Y)
      ok
    end
    for γ in (T(1), 1, T(0.4))
      X = copy(S)
      X[2, 4] = T(5)
      X[4, 2] = T(0.1)
      @test checks(X, γ)
      Y = prox(f, X, γ)
      @test Y[2, 4] === T(5) && isequal(Y[4, 2], zero(T)) && Y != transpose(Y)
      X = copy(S)
      X[5, 1] = -X[5, 1]
      @test checks(X, γ)
      X = copy(S)
      X[1, 5] = zero(T)
      @test checks(X, γ)
      # an asymmetric pair in a block away from the first one, at a size on the cutoff route
      q = 70
      X = symmetric_random(rng, T, q)
      X[q, 3] = X[3, q] + one(T)
      X[40, 36] = nextfloat(X[36, 40])
      @test checks(X, γ)
    end
    X = copy(S)
    X[3, 3] = T(-0.0)
    X[1, 2] = T(0.0)
    X[2, 1] = T(-0.0)
    @test prox(f, X) isa Matrix{T}
    # the value counts every nonzero off-diagonal entry of the given matrix
    @test NormL0(3; diagonal = false)(T[1 2 0; 0 1 0; 0 0 1]) === T(3)
    @test NormL0(3; diagonal = false)(T[1 2 0; 4 1 0; 0 0 1]) === T(6)
  end

  @testset "non-finite input" begin
    for bad in (T(NaN), T(Inf), -T(Inf))
      for (i, j) in ((1, 1), (3, 3), (1, 2), (2, 1), (5, 4), (4, 5))
        X = copy(S)
        X[i, j] = bad
        i == j || (X[j, i] = bad)
        refuses(ArgumentError, f, sentinel(), X; value = true)
        # the mirror entry is finite
        X = copy(S)
        X[i, j] = bad
        refuses(ArgumentError, f, sentinel(), X; value = true)
      end
      q = 70
      X = symmetric_random(rng, T, q)
      X[65, 66] = X[66, 65] = bad
      refuses(ArgumentError, f, sentinel(q), X; value = true)
    end
    X = copy(S)
    X[1, 2] = X[2, 1] = T(NaN)
    @test_throws ArgumentError f(X)
    # a strided view, which is scanned column by column, and a Matrix, which is scanned in one pass
    for bad in (T(NaN), T(Inf), -T(Inf)), q in (3, 9, 70), (i, j) in ((1, 1), (q, q), (2, 1), (q, q - 1), (1, q))
      buf = fill(T(1), q + 1, q)
      buf[1:q, :] .= symmetric_random(rng, T, q)
      buf[i, j] = bad
      V = view(buf, 1:q, :)
      Ys = fill(T(7.5), q, q)
      refuses(ArgumentError, f, Ys, V; watch = (buf, Ys), value = true)
      M = Matrix(V)
      refuses(ArgumentError, f, fill(T(7.5), q, q), M; value = true)
      Z = copy(M)
      @test_throws ArgumentError prox!(Z, f, Z, 1)
      @test isequal(Z, M)
    end
    # the largest and smallest finite magnitudes and both zeros are finite entries for both scans
    for q in (4, 70)
      M = symmetric_random(rng, T, q)
      M[1, 1], M[2, 2], M[3, 3], M[4, 4] = floatmax(T), -floatmax(T), nextfloat(zero(T)), -zero(T)
      M[1, 2] = M[2, 1] = -floatmax(T)
      M[2, 3] = M[3, 2] = prevfloat(zero(T))
      buf = zeros(T, q + 1, q)
      buf[1:q, :] .= M
      for X in (M, view(buf, 1:q, :))
        Y = prox(f, X, 1)
        @test isequal(Y[1, 2], -floatmax(T)) && isequal(Y[2, 3], zero(T)) && isequal(Y[4, 4], -zero(T))
        @test f(X) === f(M)
      end
    end
  end

  @testset "shape" begin
    refuses(DimensionMismatch, f, sentinel(3, 4), randn(rng, T, 3, 4); value = true)
    refuses(DimensionMismatch, f, sentinel(4, 4), S)
    refuses(DimensionMismatch, f, sentinel(p, p + 1), S)
    refuses(DimensionMismatch, f, sentinel(p - 1, p - 1), S)
    refuses(DimensionMismatch, f, sentinel(p, p - 1), S)
    # the one-based check comes first, then storage, element type and access, then the shape
    X = OffsetArray(zeros(T, 2, 3), 0:1, 0:2)
    refuses(ArgumentError, f, sentinel(2, 3), X; value = true)
    for A in (sprand(rng, T, 2, 3, 0.5), spzeros(T, 2, 3), Symmetric(S)[1:2, 1:3]', zeros(Int, 2, 3), zeros(Float16, 3, 2))
      refuses(ArgumentError, f, sentinel(2, 3), A; value = true)
      @test_throws ArgumentError prox(f, A)
      @test_throws ArgumentError prox(f, A, T(1))
    end
    refuses(ArgumentError, f, sparse(sentinel(2, 3)), S)
    refuses(ArgumentError, f, zeros(Int, 2, 3), S)
    # a strided non-square matrix reaches the shape check, also through the allocating oracle
    @test_throws DimensionMismatch prox(f, randn(rng, T, 3, 4))
  end

  @testset "reinterpreted storage that Base cannot read" begin
    # (Int32, Int64) has four bytes of padding: a Float64 reinterpretation is writable and not readable
    padded() = fill((Int32(1), Int64(1)), 1, 2)
    if T === Float64
      X = T[1 2; 2 1]
      buf = padded()
      Y = reinterpret(Float64, buf)
      @test Y isa StridedMatrix{Float64} && size(Y) == (2, 2)
      old = copy(buf)
      Xc = copy(X)
      @test_throws ArgumentError value_and_prox!(Y, f, X, 0.5)
      @test_throws ArgumentError prox!(Y, f, X, 0.5)
      @test isequal(buf, old) && isequal(X, Xc)
      wrappers = (B -> reinterpret(Float64, B), B -> view(reinterpret(Float64, B), :, 1:2),
                  B -> view(reinterpret(Float64, B), 1:2, :))
      for wrap in wrappers
        B = padded()
        Bold = copy(B)
        Z = wrap(B)
        @test Z isa StridedMatrix{Float64}
        @test_throws ArgumentError f(Z)
        @test_throws ArgumentError prox(f, Z)
        @test_throws ArgumentError prox!(Z, f, Z, 0.5)
        @test_throws ArgumentError value_and_prox!(Z, f, Z, 0.5)
        Ys = fill(7.5, 2, 2)
        @test_throws ArgumentError prox!(Ys, f, Z, 0.5)
        @test all(==(7.5), Ys)
        @test_throws ArgumentError value_and_prox!(Z, f, X, 0.5)
        @test isequal(B, Bold)
      end
    end
    # a reinterpretation without padding is accepted
    R = reinterpret(T, zeros(Complex{T}, 2, 4))
    R .= symmetric_random(rng, T, 4)
    @test isequal(prox(f, R, T(1)), prox(f, Matrix(R), T(1)))
  end

  @testset "unsupported matrix types" begin
    refuses(ArgumentError, f, sentinel(), Symmetric(S); value = true)
    refuses(ArgumentError, f, sentinel(), Hermitian(S); value = true)
    refuses(ArgumentError, f, sentinel(), Symmetric(S, :L); value = true)
    refuses(ArgumentError, f, Symmetric(sentinel()), S)
    refuses(ArgumentError, f, sentinel(), sparse(S); value = true)
    refuses(ArgumentError, f, sparse(sentinel()), S)
    refuses(ArgumentError, f, sentinel(), S'; value = true)
    refuses(ArgumentError, f, sentinel(), transpose(S); value = true)
    refuses(ArgumentError, f, sentinel()', S)
    refuses(ArgumentError, f, sentinel(), view(S, [1, 2, 3, 4, 5], :); value = true)
    refuses(ArgumentError, f, view(sentinel(), [1, 2, 3, 4, 5], :), S)
    Y = sentinel()
    refuses(ArgumentError, f, Y, matrixview(SymmetricSparseIterate(S)); watch = (Y,), value = true)
    refuses(ArgumentError, f, Y, SparseIterate(S); watch = (Y,), value = true)
    refuses(ArgumentError, f, sentinel(), Diagonal(diag(S)); value = true)
    refuses(ArgumentError, f, sentinel(), OffsetArray(S, 0:p - 1, 0:p - 1); value = true)
    refuses(ArgumentError, f, OffsetArray(sentinel(), 0:p - 1, 0:p - 1), S)
  end

  @testset "element types" begin
    U = T === Float32 ? Float64 : Float32
    refuses(ArgumentError, f, sentinel(), U.(S))
    refuses(ArgumentError, f, fill(U(7.5), p, p), S)
    for E in (Float16, BigFloat, Int, Int32, Complex{T}, Rational{Int})
      X = zeros(E, p, p)
      refuses(ArgumentError, f, X, X, 1; value = true)
      refuses(ArgumentError, f, sentinel(), X; value = true)
      refuses(ArgumentError, f, fill(one(E), p, p), S)
    end
  end

  @testset "partial aliases" begin
    buf = fill(T(7.5), 2p, p)
    A = view(buf, 1:p, :)
    A .= S
    # shifted views of one buffer overlap
    refuses(ArgumentError, f, view(buf, 2:p + 1, :), A; watch = (buf,))
    refuses(ArgumentError, f, A, view(buf, 2:p + 1, :); watch = (buf,))
    # shifted reshaped views of one vector
    v = zeros(T, p * p + 1)
    v[1:p * p] .= vec(S)
    refuses(ArgumentError, f, reshape(view(v, 2:p * p + 1), p, p), reshape(view(v, 1:p * p), p, p); watch = (v,))
    # Y === X is supported
    B = copy(S)
    @test prox!(B, f, B, one(T)) === B
  end

  @testset "steps" begin
    for γ in (0, -1, 0.0, -0.0, -1.0, T(-1), NaN, Inf, -Inf, T(NaN), T(Inf), T(0), 0//1, -1//2)
      refuses(ArgumentError, f, sentinel(), S, γ)
    end
    # array steps give an ArgumentError, also through the allocating oracle
    for γ in (fill(T(1), p, p), T[1, 2])
      Y, X = sentinel(), copy(S)
      @test_throws ArgumentError prox!(Y, f, X, γ)
      @test_throws ArgumentError value_and_prox!(Y, f, X, γ)
      @test_throws ArgumentError prox(f, X, γ)
      @test Y == sentinel() && X == S
    end
    # a step that is neither Real nor an array: prox has no method for it
    for γ in ((1.0,), 1im, "1", nothing)
      Y, X = sentinel(), copy(S)
      @test_throws ArgumentError prox!(Y, f, X, γ)
      @test_throws ArgumentError value_and_prox!(Y, f, X, γ)
      @test_throws MethodError prox(f, X, γ)
      @test Y == sentinel() && X == S
    end
  end

  @testset "arguments that are not matrices" begin
    v = T[1, 2, 3]
    w = T[4, 5, 6]
    Y = sentinel()
    X = copy(S)
    xs = SparseIterate(T, 3)
    xs[2] = T(1)
    xb = BlockIterate(T, 4, [1:2, 3:4])
    xb[1] = T(1)
    xss = SymmetricSparseIterate(S)
    snapshot() = map(copy, (v, w, Y, X, xs, xb, Matrix(matrixview(xss))))
    before = snapshot()
    @test_throws ArgumentError f(v)
    @test_throws ArgumentError f((S,))
    @test_throws ArgumentError f(zeros(T, 2, 2, 2))
    @test_throws ArgumentError f(xss)
    @test_throws ArgumentError f(xs)
    @test_throws ArgumentError f(xb)
    @test_throws ArgumentError f(1.0)
    for (out, inp) in ((w, v), (v, v), (Y, v), (vec(Y), X), (xs, xs), (xb, xb), (Y, xss), (xss, X), (xss, xss),
                       (copy(xs), xs), (Y, xs), (Y, xb))
      @test_throws ArgumentError prox!(out, f, inp, 1)
      @test_throws ArgumentError prox!(out, f, inp)
      @test_throws ArgumentError value_and_prox!(out, f, inp, 1)
      @test_throws ArgumentError value_and_prox!(out, f, inp)
    end
    @test_throws ArgumentError prox!((Y,), f, (S,), 1)
    @test_throws ArgumentError value_and_prox!((Y,), f, (S,), 1)
    @test_throws ArgumentError prox(f, v)
    @test_throws ArgumentError prox(f, v, T(1))
    @test_throws ArgumentError prox(f, (S,))
    @test_throws ArgumentError prox(f, xss)
    @test_throws ArgumentError prox(f, xss, T(1))
    @test_throws ArgumentError prox(f, xs)
    @test_throws ArgumentError prox(f, xb)
    # a symmetric sparse iterate with an array step
    @test_throws ArgumentError prox!(xss, f, xss, fill(T(1), p, p))
    @test_throws ArgumentError value_and_prox!(xss, f, xss, fill(T(1), p, p))
    @test_throws ArgumentError prox(f, xss, fill(T(1), p, p))
    @test all(map(isequal, before, snapshot()))
  end

  @testset "operations that are not provided" begin
    g = zeros(T, p, p)
    xb = BlockIterate(T, 4, [1:2, 3:4])
    xss = SymmetricSparseIterate(S)
    out = fill(T(7.5), 2)
    @test_throws ArgumentError prox_block(f, T(1), 1, T(1))
    @test_throws ArgumentError prox_block(f, T[1, 2], 1, T(1))
    @test_throws ArgumentError prox_block!(out, f, T[1, 2], 1, T(1))
    @test out == fill(T(7.5), 2)
    @test_throws ArgumentError prox_block(Scaled(f, T(2)), T(1), 1, T(1))
    @test_throws ArgumentError restrict(f, [1, 2])
    @test_throws ArgumentError kkt_violation(f, g, S, 1)
    @test_throws ArgumentError kkt_violation(f, g, S)
    @test_throws ArgumentError kkt_violation(f, g, 1)
    @test_throws ArgumentError kkt_violation(f, zeros(T, 4), xb, 1)
    @test_throws ArgumentError kkt_violation(f, g, xss, 1)
    @test_throws ArgumentError kkt_violation(f, g, xss)
  end

  @testset "prox of a Symmetric matrix is refused" begin
    @test_throws ArgumentError prox(f, Symmetric(S))
    @test_throws ArgumentError prox(f, Hermitian(S), 1)
    @test_throws ArgumentError prox(f, Symmetric(S, :L), T(1))
  end

  @testset "unsupported element types through the allocating oracle" begin
    @test_throws ArgumentError prox(f, ones(Int, 3, 3))
    @test_throws ArgumentError prox(f, Matrix{Float16}(I, 3, 3))
    @test_throws ArgumentError prox(f, Matrix{BigFloat}(I, 3, 3))
    @test_throws ArgumentError prox(f, Matrix{Complex{T}}(I, 3, 3))
    # element types without a float type, and tuples with any step, are refused before allocation
    @test_throws ArgumentError prox(f, ["a", "b"])
    @test_throws ArgumentError prox(f, fill("a", 2, 2))
    @test_throws ArgumentError prox(f, fill("a", 2, 2), T(1))
    @test_throws ArgumentError prox(f, fill("a", 2, 2), ones(T, 2, 2))
    @test_throws ArgumentError prox(f, (S,), ones(T, p, p))
    @test_throws ArgumentError prox(f, (S, 0))
    @test_throws ArgumentError prox(f, (S,), T(1))
    # a number as X has no method
    @test_throws MethodError prox(f, T(1))
  end

  @testset "refusals through Scaled leave the arguments unchanged" begin
    xss = SymmetricSparseIterate(S)
    before_ss = Matrix(matrixview(xss))
    steps = fill(T(1), p, p)
    for α in (T(2), 1e100, T(0.5))
      s = Scaled(f, α)
      Y, X = sentinel(), copy(S)
      @test_throws ArgumentError prox!(Y, s, X, steps)
      @test_throws ArgumentError value_and_prox!(Y, s, X, steps)
      @test_throws ArgumentError prox(s, X, steps)
      @test Y == sentinel() && X == S && steps == fill(T(1), p, p)
      @test_throws ArgumentError prox!(xss, s, X, 1)
      @test_throws ArgumentError value_and_prox!(xss, s, X, 1)
      @test_throws ArgumentError prox!(xss, s, xss, 1)
      @test_throws ArgumentError prox(s, xss)
      @test Matrix(matrixview(xss)) == before_ss && X == S
      v = T[1, 2, 3]
      @test_throws ArgumentError prox!(copy(v), s, v, 1)
      @test_throws ArgumentError s(v)
      @test_throws ArgumentError prox!(Y, s, sparse(S), 1)
      @test_throws ArgumentError prox!(Y, s, Symmetric(S), 1)
      @test Y == sentinel()
      # operations that are not provided
      @test_throws ArgumentError prox_block(s, T(1), 1, T(1))
      @test_throws ArgumentError restrict(s, [1, 2])
      @test_throws ArgumentError kkt_violation(s, zeros(T, p * p), 1)
      # these routes of Scaled have no method for this function
      out = fill(T(7.5), 2)
      @test_throws MethodError prox_block(s, T[1, 2], 1, T(1))
      @test_throws MethodError prox_block!(out, s, T[1, 2], 1, T(1))
      @test out == fill(T(7.5), 2)
      @test_throws MethodError kkt_violation(s, zeros(T, p, p), S, 1)
      @test_throws MethodError kkt_violation(s, zeros(T, p, p), S)
      @test_throws MethodError kkt_violation(s, zeros(T, p, p), xss)
      @test_throws MethodError kkt_violation(s, zeros(T, p, p), xss, 1)
      @test_throws MethodError kkt_violation(s, zeros(T, 4), BlockIterate(T, 4, [1:2, 3:4]), 1)
    end
  end
end

@testset "small matrices, $T" for T in (Float32, Float64)
  f = NormL0(T(2); diagonal = false)
  X0 = zeros(T, 0, 0)
  Y0 = zeros(T, 0, 0)
  @test prox!(Y0, f, X0, 1) === Y0
  @test value_and_prox!(Y0, f, X0, 1) === zero(T)
  @test f(X0) === zero(T)
  @test size(prox(f, X0)) == (0, 0) && prox(f, X0) isa Matrix{T}
  X1 = fill(T(0.01), 1, 1)
  Y1 = fill(T(NaN), 1, 1)
  @test value_and_prox!(Y1, f, X1, 100) === zero(T) && Y1[1, 1] === T(0.01)
  @test f(X1) === zero(T)
  @test isequal(prox(f, fill(T(-0.0), 1, 1)), fill(T(-0.0), 1, 1))
  @test prox(NormL0(T(0); diagonal = false), X1) == X1
end

@testset "views and reshapes, $T" for T in (Float32, Float64)
  rng = Xoshiro(77)
  p = 6
  S = symmetric_random(rng, T, p)
  S[1, 2] = S[2, 1] = T(0.001)
  f = NormL0(T(0.3); diagonal = false)
  γ = T(1)
  expected = prox(f, S, γ)
  expected_value = f(expected)
  stored(n, m) = fill(T(9.5), n, m)

  rows = stored(2p, p)
  rows[1:2:2p, :] .= S
  rowsY = stored(2p, p)
  X, Y = view(rows, 1:2:2p, :), view(rowsY, 1:2:2p, :)
  @test value_and_prox!(Y, f, X, γ) === expected_value
  @test isequal(Matrix(Y), expected)
  @test all(==(T(9.5)), rowsY[2:2:2p, :])
  @test f(X) === f(S)

  cols = stored(p, 3p)
  cols[:, 1:3:3p] .= S
  colsY = stored(p, 3p)
  X, Y = view(cols, :, 1:3:3p), view(colsY, :, 1:3:3p)
  @test prox!(Y, f, X, γ) === Y && isequal(Matrix(Y), expected)
  @test all(==(T(9.5)), colsY[:, 2:3:3p]) && all(==(T(9.5)), colsY[:, 3:3:3p])

  big = stored(p + 3, p + 2)
  big[2:p + 1, 2:p + 1] .= S
  bigY = stored(p + 3, p + 2)
  X, Y = view(big, 2:p + 1, 2:p + 1), view(bigY, 2:p + 1, 2:p + 1)
  @test prox!(Y, f, X, γ) === Y && isequal(Matrix(Y), expected)
  @test all(==(T(9.5)), bigY[1, :]) && all(==(T(9.5)), bigY[:, 1]) && all(==(T(9.5)), bigY[p + 2:end, :])
  # a view is supported in place
  @test prox!(X, f, X, γ) === X && isequal(Matrix(X), expected)

  v = vec(copy(S))
  Xr = reshape(view(v, 1:p * p), p, p)
  w = fill(T(9.5), p * p)
  Yr = reshape(view(w, 1:p * p), p, p)
  @test value_and_prox!(Yr, f, Xr, γ) === expected_value && isequal(Matrix(Yr), expected)
  @test prox!(reshape(w, p, p), f, reshape(v, p, p), γ) isa Matrix{T}
  @test isequal(reshape(w, p, p), expected)
  @test isequal(prox(f, Xr, γ), expected) && isequal(prox(f, view(rows, 1:2:2p, :), γ), expected)
  # a mixed dense and strided pair
  Z = stored(2p, p)
  @test prox!(view(Z, 1:2:2p, :), f, copy(S), γ) isa SubArray
  @test isequal(Z[1:2:2p, :], expected)
end

@testset "default step, $T" for T in (Float32, Float64)
  rng = Xoshiro(3)
  S = symmetric_random(rng, T, 6)
  f = NormL0(T(0.7); diagonal = false)
  Y = fill(T(NaN), 6, 6)
  @test prox!(Y, f, S) === Y
  @test isequal(Y, prox(f, S, 1)) && isequal(Y, prox(f, S)) && isequal(Y, prox(f, S, T(1)))
  Z = fill(T(NaN), 6, 6)
  v = value_and_prox!(Z, f, S)
  @test v === value_and_prox!(fill(T(NaN), 6, 6), f, S, 1) && v === f(Y)
  @test isequal(Z, Y)
end

@testset "value, $T" for T in (Float32, Float64)
  S = T[1 5; 5 1]
  @test NormL0(T(0); diagonal = false)(S) === zero(T)
  @test NormL0(T(3); diagonal = false)(Matrix{T}(I, 4, 4)) === zero(T)
  @test NormL0(3; diagonal = false)(S) === T(6)
  @test NormL0(T(3); diagonal = false)(S) === T(6)
  @test NormL0(T(3); diagonal = false)(T[0 -0.0 0; 0.0 0 0; 0 0 1]) === zero(T)
  # a (+0, -0) pair has no nonzero entry
  M = T[1 0.0 2; -0.0 1 0; 2 0 1]
  @test NormL0(2; diagonal = false)(M) === T(4)
  @test NormL0(floatmax(T); diagonal = false)(S) === T(Inf)
  @test NormL0(floatmax(T); diagonal = false)(Matrix{T}(I, 3, 3)) === zero(T)
  @test NormL0(floatmax(T); diagonal = false)(T[1 0; 0 1]) === zero(T)
  @test NormL0(floatmax(T) / 2; diagonal = false)(S) === floatmax(T)
  @test NormL0(1 // 3; diagonal = false)(S) === convert(T, (1 / 3) * 2.0)
  X = T[1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1]
  four = T[1 1 0 0; 1 1 0 0; 0 0 1 0; 0 0 0 1]
  @test NormL0(0.1; diagonal = false)(four) === convert(T, 0.1 * 2.0)
  # the level of a wider type is not rounded to T before the product
  @test NormL0(0.1; diagonal = false)(X) === convert(T, 0.1 * 12.0)
  @test NormL0(Float64(nextfloat(T(0.1))); diagonal = false)(X) === convert(T, Float64(nextfloat(T(0.1))) * 12.0)
  @test NormL0(1.0e300; diagonal = false)(S) === (T === Float64 ? 2.0e300 : T(Inf))
end

@testset "Float32 matrix with a Float64 level" begin
  f = NormL0(0.1; diagonal = false)
  X = Float32[1 1 1 1; 1 1 1 0; 1 1 1 0; 1 0 0 1]
  @test f(X) === Float32(0.1 * 8)
  Y = similar(X)
  @test value_and_prox!(Y, f, X, 0.5) === Float32(0.1 * 8)
  @test Y == X
  @test f(Float32[0 1 1; 1 0 0; 1 0 0]) === 0.4f0
  @test f(Float32[1 1; 1 1]) === Float32(0.1 * 2)
end

@testset "signed zero pairs, $T" for T in (Float32, Float64)
  z = zero(T)
  M = T[1 0 2; 0 1 3; 2 3 1]
  M[1, 2] = z
  M[2, 1] = -z
  @test M == transpose(M) && signbit(M[2, 1]) && !signbit(M[1, 2])
  # an identity map keeps the pair as it is
  Y = prox(NormL0(T(0); diagonal = false), M, 1)
  @test isequal(Y, M)
  # at a positive level both entries are positive zero
  Y = prox(NormL0(T(1); diagonal = false), M, 1)
  @test isequal(Y[1, 2], z) && isequal(Y[2, 1], z)
  @test Y[1, 3] === T(2) && Y[3, 1] === T(2) && Y[2, 3] === T(3)
  N = copy(M)
  N[1, 2] = N[2, 1] = -z
  @test isequal(prox(NormL0(T(1); diagonal = false), N, 1)[1, 2], z) && isequal(prox(NormL0(T(1); diagonal = false), N, 1)[2, 1], z)
  @test isequal(prox(NormL0(T(0); diagonal = false), N, 1), N)
  # a negative zero on the diagonal is copied
  D = Matrix{T}(I, 3, 3)
  D[2, 2] = -z
  @test isequal(prox(NormL0(T(1); diagonal = false), D, 1), D)
end

@testset "scaling, $T" for T in (Float32, Float64)
  rng = Xoshiro(11)
  S = symmetric_random(rng, T, 9)
  f = NormL0(T(0.4); diagonal = false)
  for α in T[0.5, 1, 2, 3], γ in T[0.3, 1, 2]
    s = Scaled(f, α)
    @test isequal(prox(s, S, γ), prox(f, S, α * γ))
    @test s(S) === α * f(S)
    Y = fill(T(NaN), 9, 9)
    v = value_and_prox!(Y, s, S, γ)
    @test isequal(Y, prox(f, S, α * γ)) && v === α * f(Y)
    Z = copy(S)
    @test prox!(Z, s, Z, γ) === Z && isequal(Z, Y)
  end
  @test (T(2) * f)(S) === T(2) * f(S) && (f * T(2))(S) === T(2) * f(S)
  @test isproximable(Scaled(f, T(2))) && !isconvex(Scaled(f, T(2)))
  @test_throws ArgumentError prox(Scaled(f, T(2)), Symmetric(S))
  A = copy(S)
  A[1, 2] += 1
  @test isequal(prox(Scaled(f, T(2)), A, T(0.5)), prox(f, A, T(1)))
  @test Scaled(f, T(2))(A) === T(2) * f(A)
end

# A Float32 matrix stored in a reinterpreted Float64 or ComplexF32 array.
function reinterpreted(::Type{P}, A::Matrix{Float32}) where {P}
  p = size(A, 1)
  R = reinterpret(Float32, Matrix{P}(undef, p ÷ 2, p))
  for j in 1:p, i in 1:p
    R[i, j] = A[i, j]
  end
  R
end

@testset "reinterpreted matrices, $P parent, p = $p" for P in (Float64, ComplexF32), p in (4, 8, 10)
  rng = Xoshiro(900 + p)
  S = symmetric_random(rng, Float32, p)
  γ = 1f0
  for λ in (0.5f0, 0.5)
    f = NormL0(λ; diagonal = false)
    expected = prox(f, S, γ)
    ev = f(expected)
    nans() = fill(NaN32, p, p)
    # as the input, with a Matrix output
    X = reinterpreted(P, S)
    @test X isa StridedMatrix{Float32} && isequal(Matrix(X), S)
    Y = nans()
    @test prox!(Y, f, X, γ) === Y && isequal(Y, expected)
    Y = nans()
    @test value_and_prox!(Y, f, X, γ) === ev && isequal(Y, expected)
    @test isequal(prox(f, X, γ), expected)
    @test f(X) === f(S)
    @test isequal(Matrix(X), S)
    # as the output, with a Matrix input
    Y = reinterpreted(P, nans())
    @test prox!(Y, f, S, γ) === Y && isequal(Matrix(Y), expected)
    Y = reinterpreted(P, nans())
    @test value_and_prox!(Y, f, S, γ) === ev && isequal(Matrix(Y), expected)
    # as the input and the output, with another matrix and in place
    Y = reinterpreted(P, nans())
    @test prox!(Y, f, reinterpreted(P, S), γ) === Y && isequal(Matrix(Y), expected)
    Z = reinterpreted(P, S)
    @test prox!(Z, f, Z, γ) === Z && isequal(Matrix(Z), expected)
    Z = reinterpreted(P, S)
    @test value_and_prox!(Z, f, Z, γ) === ev && isequal(Matrix(Z), expected)
  end
end

# Bytes allocated by one call after warm-up calls, measured inside a function.
function bytes_prox!(Y, f, X, γ)
  prox!(Y, f, X, γ)
  prox!(Y, f, X, γ)
  @allocated prox!(Y, f, X, γ)
end
function bytes_fused!(Y, f, X, γ)
  value_and_prox!(Y, f, X, γ)
  value_and_prox!(Y, f, X, γ)
  @allocated value_and_prox!(Y, f, X, γ)
end
function bytes_value(f, X)
  f(X)
  f(X)
  @allocated f(X)
end
function bytes_inplace!(X, f, γ)
  prox!(X, f, X, γ)
  prox!(X, f, X, γ)
  @allocated prox!(X, f, X, γ)
end

@testset "allocations, $T, p = $p" for T in (Float32, Float64), p in (4, 16, 70)
  rng = Xoshiro(100 + p)
  X = symmetric_random(rng, T, p)
  Y = similar(X)
  steps = (T(1), 1, T(0.5), 0.5)
  levels = (T(0.3), 1, 0.3, Float32(0.3))
  for λ in levels, γ in steps
    f = NormL0(λ; diagonal = false)
    @test bytes_prox!(Y, f, X, γ) == 0
    @test bytes_fused!(Y, f, X, γ) == 0
    @test bytes_value(f, X) == 0
    @test bytes_inplace!(copy(X), f, γ) == 0
  end
  V = view(zeros(T, 2p, p), 1:2:2p, :)
  V .= X
  W = view(zeros(T, 2p, p), 1:2:2p, :)
  f = NormL0(T(0.3); diagonal = false)
  @test bytes_prox!(W, f, V, T(1)) == 0 && bytes_fused!(W, f, V, T(1)) == 0 && bytes_value(f, V) == 0
end


# The result of op(f), or the type of the exception it throws, in a form that isequal compares bit for bit.
function outcome(op, f)
  try
    r = op(f)
    r isa SymmetricSparseIterate ? (:ok, Matrix(matrixview(r))) : (:ok, r)
  catch e
    (:error, typeof(e))
  end
end
# A supported route: both calls succeed and agree bit for bit.
same(op, f, g) = (r = outcome(op, f); first(r) === :ok && isequal(r, outcome(op, g)))
# A route that NormL0 does not provide: both calls throw E.
same_error(op, f, g, E) = outcome(op, f) == (:error, E) && outcome(op, g) == (:error, E)
with_diagonal(λ) = NormL0(λ; diagonal = true)
without_diagonal(λ) = NormL0(λ; diagonal = false)

@testset "NormL0 diagonal keyword" begin
  OD = ProximalBase._NormL0OffDiag

  @testset "diagonal = true is NormL0(λ)" begin
    for λ in (0, 1, 2.5, 0.5f0, Inf, 2, 1 // 3)
      f = NormL0(λ)
      g = NormL0(λ; diagonal = true)
      @test f === g && typeof(g) === NormL0{typeof(λ)} && g.λ === λ && hash(f) == hash(g)
    end
    @test @inferred(NormL0(2.0)) === NormL0{Float64}(2.0)
    @test @inferred(NormL0(0.5f0)) === NormL0{Float32}(0.5f0)
    @test @inferred(NormL0()) === NormL0{Int}(1)
    # @inferred infers a keyword call from the keyword types, not their values, so a literal keyword is
    # checked inside a function
    @test @inferred(with_diagonal(2.0)) === NormL0{Float64}(2.0)
    @test @inferred(with_diagonal(0.5f0)) === NormL0{Float32}(0.5f0)
    @test @inferred(with_diagonal(2)) === NormL0{Int}(2)
    @test NormL0{Float32}(1f0) === NormL0(1f0) === NormL0(1f0; diagonal = true)
    @test NormL0(2) isa NormL0{Int} && NormL0(2).λ === 2
    @test NormL0(Inf) isa NormL0{Float64} && NormL0(Inf; diagonal = true) === NormL0(Inf)
    for bad in (-1, NaN, -Inf)
      @test_throws ArgumentError NormL0(bad)
      @test_throws ArgumentError NormL0(bad; diagonal = true)
    end
    # one constructor method for each positional signature
    @test length(methods(NormL0, Tuple{})) == 1
    @test length(methods(NormL0, Tuple{Float64})) == 1
    @test allunique([m.sig for m in methods(NormL0)])
  end

  @testset "diagonal = true outputs, λ = $λ" for λ in (0, 1, 2.5, 0.5f0, Inf)
    rng = Xoshiro(10581)
    γ = 0.7
    f = NormL0(λ)
    g = NormL0(λ; diagonal = true)
    v = randn(rng, 9)
    v[3] = 0.0
    v[4] = -0.0
    v[5] = 0.5
    A = randn(rng, 4, 4)
    B = randn(rng, 5, 5)
    sp = sparse(B .* (abs.(B) .> 0.6))
    xss = SymmetricSparseIterate(B .+ B')
    for x in (v, Float32.(v), A, Float32.(A), sp)
      @test same(h -> h(x), f, g)
      @test same(h -> prox(h, x, γ), f, g)
      @test same(h -> prox(h, x), f, g)
      @test same(h -> prox!(copy(x), h, x, γ), f, g)
      @test same(h -> (y = copy(x); prox!(y, h, y, γ)), f, g)
      @test same(h -> (y = copy(x); (value_and_prox!(y, h, x, γ), y)), f, g)
      @test same(h -> (y = copy(x); (value_and_prox!(y, h, y, γ), y)), f, g)
      steps = fill(γ, size(x))
      steps[1] = 2.0
      @test same(h -> prox(h, x, steps), f, g)
      @test same(h -> prox!(copy(x), h, x, steps), f, g)
      @test same(h -> (y = copy(x); (value_and_prox!(y, h, x, steps), y)), f, g)
    end
    @test same(h -> h(xss), f, g)
    @test same(h -> prox(h, xss, γ), f, g)
    @test same(h -> prox!(copy(xss), h, xss, γ), f, g)
    @test same(h -> (y = copy(xss); (value_and_prox!(y, h, xss, γ), Matrix(matrixview(y)))), f, g)
    @test same(h -> restrict(h, [1, 3]), f, g)
    for u in (0.5, -1.3, 0.0, -0.0, 0.5f0)
      @test same(h -> prox_block(h, u, 1, γ), f, g)
      @test same(h -> value_entry(h, u, 1), f, g)
    end
    grad = randn(rng, 9)
    @test same_error(h -> kkt_violation(h, grad, 3), f, g, MethodError)
    @test same_error(h -> kkt_violation(h, grad, v, 5), f, g, MethodError)
    @test same_error(h -> kkt_violation(h, grad, v), f, g, MethodError)
  end

  @testset "diagonal = false" begin
    for λ in (0, 1, 2.5, 0.5f0, 1 // 3)
      h = NormL0(λ; diagonal = false)
      @test typeof(h) === OD{typeof(λ)} && h.λ === λ && h isa AbstractFunction
      @test h != NormL0(λ) && !isequal(h, NormL0(λ))
      @test h == NormL0(λ; diagonal = false) && hash(h) == hash(NormL0(λ; diagonal = false))
    end
    @test @inferred(without_diagonal(2.0)) === NormL0(2.0; diagonal = false)
    @test @inferred(without_diagonal(0.5f0)) isa OD{Float32}
    @test NormL0(2; diagonal = false) isa OD{Int}
    @test NormL0(; diagonal = false) isa OD{Int} && NormL0(; diagonal = false).λ === 1
    diagonal = false
    @test NormL0(2.0; diagonal) isa OD{Float64}
    @test sprint(show, NormL0(2.5; diagonal = false)) == "NormL0(2.5; diagonal = false)"
    @test sprint(show, NormL0(0.5f0; diagonal = false)) == "NormL0(0.5f0; diagonal = false)"
    @test sprint(show, NormL0(; diagonal = false)) == "NormL0(1; diagonal = false)"
    @test repr(NormL0(1 // 3; diagonal = false)) == "NormL0(1//3; diagonal = false)"
    for bad in (Inf, -Inf, NaN, Inf32, -1, -0.5f0, -1 // 2)
      @test_throws ArgumentError NormL0(bad; diagonal = false)
    end
    @test_throws TypeError NormL0(1.0; diagonal = 1)
  end

  @testset "Scaled diagonal-free penalty, $T" for T in (Float32, Float64)
    rng = Xoshiro(10582)
    S = symmetric_random(rng, T, 12)
    for λ in (T(0.4), 1, 0.25)
      f = NormL0(λ; diagonal = false)
      for α in T[0.5, 1, 4], γ in T[0.25, 1]
        s = Scaled(f, α)
        @test s.f === f && s.α === α
        @test s(S) === α * f(S)
        @test isequal(prox(s, S, γ), prox(f, S, α * γ))
        Y = fill(T(NaN), 12, 12)
        @test value_and_prox!(Y, s, S, γ) === α * f(Y) && isequal(Y, prox(f, S, α * γ))
      end
    end
    # scale, step and level of other types than the matrix: the value is converted to the element type
    for λ in (T(0.4), 1, 0.25, 0.3f0), α in (0.1, 0.5f0, 2, T(3)), γ in (1, 0.5, T(0.25), 1 // 2)
      f = NormL0(λ; diagonal = false)
      s = Scaled(f, α)
      @test s(S) === convert(T, α * f(S))
      @test isequal(prox(s, S, γ), prox(f, S, α * γ))
      Y = fill(T(NaN), 12, 12)
      v = value_and_prox!(Y, s, S, γ)
      @test v === s(Y) && isequal(Y, prox(f, S, α * γ))
    end
  end

  @testset "Scaled step and value rounding" begin
    # Scaled computes the step by its own rule. Here it rounds to zero, which is refused, although the native
    # product is the smallest subnormal and the direct call at that step is accepted.
    f = NormL0(1.0; diagonal = false)
    X = [-0.0 2.0; 2.0 1.0]
    η = nextfloat(0.0)
    α, γ = nextfloat(1 / 6), 3η
    @test α * γ === η
    @test isequal(prox(f, X, α * γ), [-0.0 2.0; 2.0 1.0])
    s = Scaled(f, α)
    Y = fill(7.0, 2, 2)
    @test_throws ArgumentError prox(s, X, γ)
    @test_throws ArgumentError prox!(Y, s, X, γ)
    @test_throws ArgumentError value_and_prox!(Y, s, X, γ)
    @test all(==(7.0), Y) && isequal(X, [-0.0 2.0; 2.0 1.0])
    # the value of the function is rounded before the scale: Inf32 although α g(V) is representable
    V = Float32[1 2; 2 1]
    huge = NormL0(floatmax(Float32); diagonal = false)
    @test huge(V) === Inf32
    @test Scaled(huge, 0.25f0)(V) === Inf32
  end
end

end
