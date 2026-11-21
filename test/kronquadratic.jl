using Test, ProximalBase, LinearAlgebra, Random, OffsetArrays

# References written from the formulas, independent of the implementation.

# q(X) = vec(X)' (B ⊗ A) vec(X) / 2 + vec(C)' vec(X) and its gradient (B ⊗ A) vec(X) + vec(C)
kron_value(A, B, C, X) = (x = vec(X); dot(x, kron(B, A) * x) / 2 + (C === nothing ? zero(eltype(x)) : dot(vec(C), x)))
kron_gradient(A, B, C, X) = reshape(kron(B, A) * vec(X), size(X)) .+ (C === nothing ? 0 : C)
# the same with absolute values: the scale of the rounding errors
kron_scale(A, B, C, X) = kron_value(abs.(A), abs.(B), C === nothing ? nothing : abs.(C), abs.(X))
kron_gradient_scale(A, B, C, X) = kron_gradient(abs.(A), abs.(B), C === nothing ? nothing : abs.(C), abs.(X))
# (A X B + B X A)/2 + C, the derivative over symmetric matrices, by explicit loops
function symmetric_reference(A, B, C, X)
  p = size(X, 1)
  W = [sum(A[i, k] * X[k, l] * B[l, j] for k in 1:p, l in 1:p) for i in 1:p, j in 1:p]
  (W + W') / 2 .+ (C === nothing ? 0 : C)
end

random_symmetric(rng, ::Type{T}, p) where {T} = (M = randn(rng, p, p); T.(M + M'))
function random_iterate(rng, ::Type{T}, p, coordinates) where {T}
  X = SymmetricSparseIterate(T, p)
  for k in coordinates
    r, c = tril_cartesian(p, k)
    X[r, c] = T(randn(rng))
  end
  X
end
# s[i] s[j] ρ^|i - j|, exactly symmetric
ar1(::Type{T}, p, ρ, s = ones(p)) where {T} = T[s[i] * s[j] * ρ^abs(i - j) for i in 1:p, j in 1:p]

value_allocs(f::F, X::A) where {F, A} = (f(X); @allocated f(X))
gradient_allocs(G::M, f::F, X::A) where {M, F, A} = (gradient!(G, f, X); @allocated gradient!(G, f, X))
value_gradient_allocs(G::M, f::F, X::A) where {M, F, A} =
  (value_and_gradient!(G, f, X); @allocated value_and_gradient!(G, f, X))

@testset "KronQuadratic" begin

@testset "construction, traits and data ($T)" for T in (Float32, Float64)
  A = T[2 1; 1 3]
  B = T[1 0 1; 0 2 0; 1 0 3]
  C = T[1 0 0; 0 0 -1]
  f = KronQuadratic(A, B, C)
  @test f isa KronQuadratic{T}
  @test issmooth(f) && !isconvex(f) && !isproximable(f) && !isseparable(f)
  @test issmooth(KronQuadratic) && !isconvex(KronQuadratic)
  @test size(f) == (2, 3)
  @test lipschitz(f) isa T
  @test lipschitz(f) ≈ opnorm(Float64.(A)) * opnorm(Float64.(B)) rtol = 10eps(T)
  d = ProximalBase.kronquadratic_data(f)
  @test keys(d) == (:A, :B, :C)
  @test d.A == A && d.B == B && d.C == C
  @test d.A isa Matrix{T} && d.C isa Matrix{T}
  @test ProximalBase.kronquadratic_data(KronQuadratic(A, B)).C === nothing
  @test ProximalBase.entry_parameters(f) == (:C => d.C,)

  # snapshots: later edits of the caller's arrays do not reach f
  X = T[1 0 2; -1 1 0]
  v = f(X)
  A[1, 1] = 100; B[2, 2] = 100; C[1, 1] = 100
  @test f(X) == v
  @test d.A[1, 1] == 2 && d.B[2, 2] == 2 && d.C[1, 1] == 1

  # structured wrappers of the data type are copied densely
  S = T[2 1; 1 3]
  g = KronQuadratic(Symmetric(T[2 1; 7 3], :U), Diagonal(T[1, 2, 3]), Matrix{T}(undef, 2, 3) .= 0)
  @test ProximalBase.kronquadratic_data(g).A == S
  @test ProximalBase.kronquadratic_data(g).B == Diagonal(T[1, 2, 3])
  @test ProximalBase.kronquadratic_data(g).A isa Matrix{T}
  @test ProximalBase.kronquadratic_data(KronQuadratic(Hermitian(T[2 7; 1 3], :L), T[1;;])).A == T[2 1; 1 3]
end

@testset "analytic examples ($T)" for T in (Float32, Float64)
  # 1 × 1: q = 5 * 2 * 5 * 3 / 2 + 5 = 80, gradient 2 * 5 * 3 + 1 = 31
  f = KronQuadratic(T[2;;], T[3;;], T[1;;])
  @test f(T[5;;]) === T(80)
  @test gradient(f, T[5;;]) == T[31;;]
  X1 = SymmetricSparseIterate(T, 1); X1[1, 1] = 5
  @test f(X1) === T(80)
  @test gradient(f, X1) == T[31;;]
  @test value_and_gradient(f, X1) == (T(80), T[31;;])

  # 2 × 2, A = I, B = diag(1, 2), X = [0 1; 1 0]: A X B = [0 2; 1 0]
  f = KronQuadratic(Matrix{T}(I, 2, 2), Diagonal(T[1, 2]))
  X = T[0 1; 1 0]
  Xs = SymmetricSparseIterate(X)
  @test gradient(f, X) == T[0 2; 1 0]
  @test gradient(f, Xs) == T[0 1.5; 1.5 0]
  @test f(X) === f(Xs) === T(1.5)
  # one variable moves both entries of the off-diagonal pair: the derivative along E = [0 1; 1 0] is
  # ⟨E, A X B⟩ = 3, the multiplicity 2 times the symmetric gradient entry 1.5
  h = T(1 / 4)
  Xp, Xm = copy(Xs), copy(Xs)
  Xp[1, 2] += h; Xm[1, 2] -= h
  @test (f(Xp) - f(Xm)) / (2h) ≈ 3 rtol = 8eps(T)
  @test 2 * gradient(f, Xs)[1, 2] == 3

  # rectangular 2 × 3: A X B = [5 2 13; 0 6 4], ⟨X, AXB⟩/2 = 37/2, ⟨C, X⟩ = 1
  f = KronQuadratic(T[2 1; 1 3], T[1 0 1; 0 2 0; 1 0 3], T[1 0 0; 0 0 -1])
  X = T[1 0 2; -1 1 0]
  @test f(X) === T(39 / 2)
  @test gradient(f, X) == T[6 2 13; 0 6 3]
  v, G = value_and_gradient(f, X)
  @test v === T(39 / 2) && G == T[6 2 13; 0 6 3] && G isa Matrix{T}
  @test KronQuadratic(T[2 1; 1 3], T[1 0 1; 0 2 0; 1 0 3])(X) === T(37 / 2)

  # noncommuting A and B (A B ≠ B A) with a symmetric X and C: A X B = [4 8 -3; 5 2 -3; 2 -4 3]
  A = T[2 1 0; 1 2 1; 0 1 2]
  B = T[1 0 0; 0 2 0; 0 0 3]
  C = T[0 1 0; 1 0 0; 0 0 -2]
  @test A * B != B * A
  f = KronQuadratic(A, B, C)
  X = T[1 2 0; 2 0 -1; 0 -1 1]
  Xs = SymmetricSparseIterate(X)
  @test f(X) === f(Xs) === T(22)
  @test gradient(f, X) == T[4 9 -3; 6 2 -3; 2 -4 1]
  @test gradient(f, Xs) == T[4 7.5 -0.5; 7.5 2 -3.5; -0.5 -3.5 1]
  # matrixview(X) is a matrix argument: the gradient A X B + C
  @test gradient(f, matrixview(Xs)) == T[4 9 -3; 6 2 -3; 2 -4 1]
  G = fill(T(NaN), 3, 3)
  @test gradient!(G, f, matrixview(Xs)) === G
  @test G == T[4 9 -3; 6 2 -3; 2 -4 1]
  @test gradient!(G, f, Xs) === G
  @test G == T[4 7.5 -0.5; 7.5 2 -3.5; -0.5 -3.5 1]
end

@testset "Kronecker reference ($T)" for T in (Float32, Float64)
  rng = Xoshiro(1059)
  for (p, q) in ((1, 1), (2, 3), (4, 4), (5, 2), (3, 7)), linear in (false, true)
    A, B = random_symmetric(rng, T, p), random_symmetric(rng, T, q)
    C = linear ? T.(randn(rng, p, q)) : nothing
    X = T.(randn(rng, p, q))
    f = KronQuadratic(A, B, C)
    # Float64 references of the same T data. Tolerances for these moderate seeded cases, with u = eps(T):
    # (pq + p + q + 2) u times the sum of absolute terms for the value, and (p + q + 1) u times the same
    # with absolute values for each gradient entry
    A64, B64, C64, X64 = Float64.(A), Float64.(B), C === nothing ? nothing : Float64.(C), Float64.(X)
    vtol = (p * q + p + q + 2) * eps(T) * kron_scale(A64, B64, C64, X64)
    Gtol = (p + q + 1) * eps(T) .* kron_gradient_scale(A64, B64, C64, X64)
    v, G = value_and_gradient(f, X)
    @test v isa T && G isa Matrix{T}
    @test abs(v - kron_value(A64, B64, C64, X64)) <= vtol
    @test all(abs.(G .- kron_gradient(A64, B64, C64, X64)) .<= Gtol)
    @test f(X) == v
    @test gradient(f, X) == G
  end
end

@testset "argument representations ($T)" for T in (Float32, Float64)
  rng = Xoshiro(7)
  p, q = 4, 3
  A, B, C = random_symmetric(rng, T, p), random_symmetric(rng, T, q), T.(randn(rng, p, q))
  f = KronQuadratic(A, B, C)
  X = T.(randn(rng, p, q))
  X[2, 3] = 0
  v, G = value_and_gradient(f, X)
  scale = kron_scale(Float64.(A), Float64.(B), Float64.(C), Float64.(X))
  Gscale = maximum(kron_gradient_scale(Float64.(A), Float64.(B), Float64.(C), Float64.(X)))
  # contiguous columns of a larger matrix, every other column, every other row, a SparseIterate
  larger = zeros(T, 2p, 2q)
  larger[1:p, 1:q] = X
  cols = zeros(T, p, 2q); cols[:, 1:2:end] = X
  rows = zeros(T, 2p, q); rows[1:2:end, :] = X
  for Y in (view(larger, 1:p, 1:q), view(cols, :, 1:2:2q), view(rows, 1:2:2p, :), SparseIterate(X))
    @test f(Y) ≈ v atol = 4 * (p * q + p + q) * eps(T) * scale
    @test gradient(f, Y) ≈ G atol = 4 * (p + q) * eps(T) * Gscale
    @test gradient(f, Y) isa Matrix{T}
  end
  # destinations with contiguous and with strided columns; rows not written stay as they were
  for (backing, out) in ((fill(T(-7), 2p, q), (b -> view(b, 1:2:2p, :))), (fill(T(-7), p, 2q), (b -> view(b, :, 2:2:2q))))
    D = out(backing)
    @test value_and_gradient!(D, f, X) ≈ v atol = 4 * (p * q + p + q) * eps(T) * scale
    @test D ≈ G atol = 4 * (p + q) * eps(T) * Gscale
    @test count(==(T(-7)), backing) == length(backing) - p * q
  end
  # a Symmetric wrapper is a matrix argument: the gradient A X B + C, not its symmetric part
  S = random_symmetric(rng, T, p)
  g = KronQuadratic(A, random_symmetric(rng, T, p), random_symmetric(rng, T, p))
  @test gradient(g, Symmetric(S)) ≈ gradient(g, S)
  @test gradient(g, Symmetric(S)) ≉ gradient(g, SymmetricSparseIterate(S))
  # a NaN entry of an accepted argument propagates by IEEE arithmetic
  Y = copy(X); Y[1, 1] = T(NaN)
  @test isnan(f(Y))
  @test any(isnan, gradient(f, Y))
end

@testset "symmetric iterates ($T)" for T in (Float32, Float64)
  rng = Xoshiro(21)
  p = 6
  n = p * (p + 1) ÷ 2
  A, B = random_symmetric(rng, T, p), random_symmetric(rng, T, p)
  for linear in (false, true)
    C = linear ? random_symmetric(rng, T, p) : nothing
    f = KronQuadratic(A, B, C)
    A64, B64, C64 = Float64.(A), Float64.(B), C === nothing ? nothing : Float64.(C)
    # empty; an off-diagonal and a diagonal coordinate (2n = 4 for n = 2 nonzero values, at most p²/8 = 4.5,
    # the accumulating route); every coordinate (2n = 42, the dense route); and active coordinates holding
    # explicit zeros (n = 1 and n = 14)
    empty = SymmetricSparseIterate(T, p)
    sparse = random_iterate(rng, T, p, [tril_index(p, 4, 2), tril_index(p, 3, 3)])
    dense = random_iterate(rng, T, p, 1:n)
    zeros_active = random_iterate(rng, T, p, [tril_index(p, 5, 2), tril_index(p, 2, 2), tril_index(p, 6, 6)])
    ids, vals = activecoordinates(zeros_active), activevalues(zeros_active)
    for a in eachindex(vals)
      ids[a] in (tril_index(p, 5, 2), tril_index(p, 2, 2)) && (vals[a] = 0)
    end
    @test nactive(zeros_active) == 3 && count(iszero, activevalues(zeros_active)) == 2
    zeros_dense = random_iterate(rng, T, p, 1:n)
    activevalues(zeros_dense)[1:3:n] .= 0
    @test nactive(zeros_dense) == n && count(iszero, activevalues(zeros_dense)) == 7
    # the route counts nonzero values, not active coordinates (zeros_active would cross 2n > p²/8 with n = 3)
    @test !ProximalBase._kron_dense_route(zeros_active, p) && !ProximalBase._kron_dense_route(sparse, p)
    @test ProximalBase._kron_dense_route(dense, p) && ProximalBase._kron_dense_route(zeros_dense, p)
    for X in (empty, sparse, dense, zeros_active, zeros_dense)
      Xm = Float64.(Matrix(X))
      scale = kron_scale(A64, B64, C64, Xm)
      Gscale = maximum(kron_gradient_scale(A64, B64, C64, Xm))
      v, G = value_and_gradient(f, X)
      @test v isa T && G isa Matrix{T}
      @test issymmetric(G)
      # a common tolerance chosen for both routes and both value oracles in these cases, (p² + 2p + 4) u times the
      # sum of absolute terms; p² is the term count of the transposed sum of the accumulating value-only call
      @test abs(f(X) - kron_value(A64, B64, C64, Xm)) <= (p^2 + 2p + 4) * eps(T) * scale
      @test abs(v - kron_value(A64, B64, C64, Xm)) <= (p^2 + 2p + 4) * eps(T) * scale
      @test abs(f(X) - f(Matrix(X))) <= 2 * (p^2 + 2p + 4) * eps(T) * scale
      @test maximum(abs.(G .- symmetric_reference(A64, B64, C64, Xm))) <= (2p + 2) * eps(T) * Gscale
      @test gradient(f, X) == G
      Gd = gradient(f, matrixview(X))
      @test maximum(abs.(Gd .- kron_gradient(A64, B64, C64, Xm))) <= (2p + 2) * eps(T) * Gscale
      if nactive(X) > 0 && any(!iszero, activevalues(X))
        # for these data, the derivative over symmetric matrices differs from A X B + C
        @test maximum(abs.(G .- Gd)) > 1000 * (2p + 2) * eps(T) * Gscale
      end
      @test G ≈ (Gd + Gd') / 2 atol = 4 * (2p + 2) * eps(T) * Gscale
    end
    @test f(empty) == 0
    @test gradient(f, empty) == (C === nothing ? zeros(T, p, p) : C)

    # the central difference along coordinate k moves both entries of an off-diagonal pair:
    # its derivative is m G[r, c], m = 1 on the diagonal and 2 off it
    X = random_iterate(rng, T, p, 1:2:n)
    G = gradient(f, X)
    h = T(1 / 8)
    scale = kron_scale(A64, B64, C64, Float64.(Matrix(X)) .+ h)
    for k in 1:n
      r, c = tril_cartesian(p, k)
      Xp, Xm = copy(X), copy(X)
      Xp[r, c] += h; Xm[r, c] -= h
      m = r == c ? 1 : 2
      @test abs((f(Xp) - f(Xm)) / (2h) - m * G[r, c]) <= 4 * (p^2 + 2p + 4) * eps(T) * scale / h
    end
  end
end

@testset "finite differences and curvature ($T)" for T in (Float32, Float64)
  rng = Xoshiro(3)
  p, q = 4, 3
  A, B, C = random_symmetric(rng, T, p), random_symmetric(rng, T, q), T.(randn(rng, p, q))
  f = KronQuadratic(A, B, C)
  X = T.(randn(rng, p, q))
  G = gradient(f, X)
  h = T(1 / 8)
  for _ in 1:5
    V = T.(randn(rng, p, q))
    scale = kron_scale(Float64.(A), Float64.(B), Float64.(C), abs.(Float64.(X)) .+ h .* abs.(Float64.(V)))
    @test abs((f(X + h * V) - f(X - h * V)) / (2h) - dot(G, V)) <= 8 * (p * q + p + q + 2) * eps(T) * scale / h
  end
  # H[V] = ∇q(V) - ∇q(0) = A V B: ⟨U, H[V]⟩ = ⟨V, H[U]⟩
  H(V) = gradient(f, V) - gradient(f, zeros(T, p, q))
  U, V = T.(randn(rng, p, q)), T.(randn(rng, p, q))
  bound = 4 * (p + q + 2) * eps(T) * sum(abs.(U) .* kron_gradient_scale(Float64.(A), Float64.(B), Float64.(C), abs.(V)))
  @test abs(dot(U, H(V)) - dot(V, H(U))) <= 2bound
  # for positive semidefinite factors the quadratic form is nonnegative
  M, N = T.(randn(rng, p, 2)), T.(randn(rng, q, 2))
  P = KronQuadratic(Symmetric(M * M'), Symmetric(N * N'))
  for _ in 1:5
    V = T.(randn(rng, p, q))
    s = kron_scale(Float64.(M * M'), Float64.(N * N'), nothing, Float64.(V))
    @test dot(V, gradient(P, V)) >= -4 * (p * q + p + q) * eps(T) * s
  end
  # a rank-one direction v u' with A v = 0 gives a zero form up to rounding
  v, u = T.(nullspace(Float64.(M'))[:, 1]), T.(randn(rng, q))
  @test abs(P(v * u')) <= 100 * (p + q) * eps(T) * opnorm(Float64.(M))^2 * opnorm(Float64.(N))^2 * norm(u)^2
end

@testset "zero factors and extreme scales ($T)" for T in (Float32, Float64)
  C = T[1 -2; 3 0.5]
  X = T[2 1; -1 4]
  for f in (KronQuadratic(zeros(T, 2, 2), T[1 2; 2 1], C), KronQuadratic(T[1 2; 2 1], zeros(T, 2, 2), C))
    @test f(X) == dot(C, X)
    @test gradient(f, X) == C
    @test lipschitz(f) == 0
  end
  # scalar cases with entries near sqrt(floatmax(T)): finite results below, IEEE overflow at X = 4
  e = (exponent(floatmax(T)) - 1) ÷ 2
  s = ldexp(one(T), e)
  f = KronQuadratic(T[s;;], T[s;;])
  @test f(T[1;;]) == ldexp(one(T), 2e - 1)
  @test gradient(f, T[1;;]) == T[ldexp(one(T), 2e);;]
  @test f(T[4;;]) == T(Inf)
  @test gradient(f, T[4;;]) == T[Inf;;]
  # scalar cases with entries near sqrt(floatmin(T)): a subnormal gradient, and a value whose exact result is
  # below half the smallest subnormal and rounds to zero (no range recovery)
  e = exponent(floatmin(T)) ÷ 2
  s = ldexp(one(T), e)
  f = KronQuadratic(T[s;;], T[s;;])
  @test f(T[1;;]) == ldexp(one(T), 2e - 1)
  k = (precision(T) - 1) ÷ 2 + 1
  x = ldexp(one(T), -k)
  @test gradient(f, T[x;;]) == T[ldexp(one(T), 2e - k);;]
  @test f(T[x;;]) == 0
end

@testset "D-trace loss ($T)" for T in (Float32, Float64)
  # q(D) = tr(Σx D Σy D)/2 - tr((Σy - Σx) D) is KronQuadratic(Σx, Σy, Σx - Σy); its symmetric gradient
  # (Σx D Σy + Σy D Σx)/2 + Σx - Σy vanishes at D* = inv(Σx) - inv(Σy)
  # analytic: Σx = [2 1; 1 2], Σy = I, D* = [2 -1; -1 2]/3 - I = -[1 1; 1 1]/3, q(D*) = ⟨C, D*⟩/2 = -2/3
  Σx, Σy = T[2 1; 1 2], Matrix{T}(I, 2, 2)
  f = KronQuadratic(Σx, Σy, Σx - Σy)
  D = SymmetricSparseIterate(fill(T(-1 / 3), 2, 2))
  @test maximum(abs, gradient(f, D)) <= 8eps(T)
  @test f(D) ≈ T(-2 / 3) rtol = 8eps(T)
  W = SymmetricSparseIterate(fill(T(1 / 3), 2, 2))     # the opposite sign is not stationary
  @test gradient(f, W) ≈ T[2 2; 2 2] rtol = 8eps(T)
  for E in (T[0.1 0; 0 0], T[0 0.1; 0.1 0], T[0 0; 0 -0.1])
    @test f(SymmetricSparseIterate(fill(T(-1 / 3), 2, 2) + E)) > f(D)
  end

  # SPD Σx ≠ Σy: residual of the gradient at D* rounded from a BigFloat solution, both arguments.
  # Tolerance for these two moderate cases: ‖G‖max ≤ c u ‖Σx‖F ‖Σy‖F ‖D*‖F with c = 4p + 8 and
  # u = eps(T), not a uniform accuracy bound for arbitrary SPD inputs and scales. A dense D* (5 × 5) takes the
  # dense route of the symmetric iterate; a diagonal D* (16 × 16, 2n = 32 ≤ p²/8) the accumulating route.
  scales = [1, 1.2, 0.9, 1.1, 1.3]
  for (Σx, Σy) in ((ar1(T, 5, 0.4), ar1(T, 5, 0.2, scales)),
                   (Matrix(Diagonal(T.(1:16))), Matrix(Diagonal(T.(mod.(7 .* (1:16), 17))))))
    p = size(Σx, 1)
    @test Σx != Σy
    f = KronQuadratic(Σx, Σy, Σx - Σy)
    Dbig = inv(big.(Σx)) - inv(big.(Σy))
    Dstar = Matrix(Symmetric(T.(Dbig), :L))
    bound = (4p + 8) * eps(T) * norm(Σx) * norm(Σy) * norm(Dstar)
    @test maximum(abs, gradient(f, SymmetricSparseIterate(Dstar))) <= bound
    @test maximum(abs, gradient(f, Dstar)) <= bound
    @test norm(Dstar) > 0.1
  end
end

# A proximal-gradient loop on a symmetric iterate: X ← prox_{t g}(X - t ∇q(X)), t = 1/lipschitz(f).
function symmetric_pgd(f, g, p, ::Type{T}, iterations) where {T}
  X = SymmetricSparseIterate(T, p)
  G = zeros(T, p, p)
  t = inv(lipschitz(f))
  for _ in 1:iterations
    gradient!(G, f, X)
    prox!(X, g, Matrix(X) - t * G, t)
  end
  X
end
function dense_pgd(f, g, p, ::Type{T}, iterations) where {T}
  X = zeros(T, p, p)
  G = zeros(T, p, p)
  t = inv(lipschitz(f))
  for _ in 1:iterations
    gradient!(G, f, X)
    prox!(X, g, X - t * G, t)
  end
  X
end

@testset "D-trace with an off-diagonal ℓ1 penalty ($T)" for T in (Float32, Float64)
  p = 5
  Σx = ar1(T, p, 0.4)
  Σy = ar1(T, p, 0.2, [1, 1.2, 0.9, 1.1, 1.3])
  C = Σx - Σy
  f = KronQuadratic(Σx, Σy, C)
  λ = T(0.05)
  Wt = ones(T, p, p) - I
  g = NormL1(λ, Wt)
  X = symmetric_pgd(f, g, p, T, 2000)
  Xm = Matrix(X)
  @test Xm == Xm'
  # the gradient recomputed in Float64 by the formula; tolerance 100 p u (L max|X| + λ + max|C|)
  Gref = symmetric_reference(Float64.(Σx), Float64.(Σy), Float64.(C), Float64.(Xm))
  L = opnorm(Float64.(Σx)) * opnorm(Float64.(Σy))
  tol = 100 * p * eps(T) * (L * maximum(abs, Xm) + λ + maximum(abs, C))
  scores = [kkt_violation(g, Gref, X, k) for k in 1:ncoordinates(X)]
  @test maximum(scores) <= tol
  @test count(iszero, Xm) > 0 && count(!iszero, Xm - Diagonal(Xm)) > 0
  # the same loop on dense matrices targets the objective over all matrices, with the gradient A X B + C, a different
  # feasible set; the checks below establish approximate stationarity of its result and, for these data and λ,
  # an observed difference from the symmetric result
  Xd = dense_pgd(f, g, p, T, 2000)
  Gd = Float64.(Σx) * Float64.(Xd) * Float64.(Σy) + Float64.(C)
  @test maximum(kkt_violation(g, Gd, Xd, i) for i in 1:p^2) <= tol
  @test norm(Xd - Xd') > 10tol
  @test norm(Xd - Xm) > 10tol
end

@testset "coordinate-descent direction model ($T)" for T in (Float32, Float64)
  # around Θ with W = inv(Θ), tr(S(Θ + D)) - logdet(Θ + D) has the second-order model
  # tr((S - W) D) + tr(W D W D)/2 in the direction D: KronQuadratic(W, W, S - W)
  p = 4
  Θ = ar1(Float64, p, 0.3) + 0.5I
  S = ar1(Float64, p, 0.5)
  W = Matrix(Symmetric(T.(inv(Θ)), :L))
  S, Θ = T.(S), T.(Θ)
  f = KronQuadratic(W, W, S - W)
  zero_iterate = SymmetricSparseIterate(T, p)
  @test gradient(f, zero_iterate) == S - W
  rng = Xoshiro(11)
  D = random_iterate(rng, T, p, 1:3:p * (p + 1) ÷ 2)
  Dm = Float64.(Matrix(D))
  HD = gradient(f, D) - gradient(f, zero_iterate)
  @test isapprox(HD, Float64.(W) * Dm * Float64.(W); rtol = 50p * eps(T), atol = 50p * eps(T) * norm(S - W))
  @test HD ≈ HD'
  # the curvature of LogDetLoss along D, by central differences of its gradient
  h = 1e-3
  ld = LogDetLoss(Float64.(S))
  fd = (gradient(ld, Float64.(Θ) + h * Dm) - gradient(ld, Float64.(Θ) - h * Dm)) / (2h)
  @test fd ≈ Float64.(W) * Dm * Float64.(W) rtol = max(1e-5, 100p * eps(T))
  # a weighted loss w (tr(SΘ) - logdet(Θ)) has the model KronQuadratic(w W, W, w (S - W))
  w = T(0.75)
  fw = KronQuadratic(w * W, W, w * (S - W))
  @test gradient(fw, zero_iterate) ≈ gradient(LogDetLoss(S, w), Θ) rtol = 50p * eps(T)
  @test gradient(fw, D) ≈ w * (gradient(f, D)) rtol = 50p * eps(T)
  # the model of the new point Z = Θ + D is a different function: KronQuadratic(W, W, S - 2W) at Z,
  # whose gradient there equals the direction model's gradient at D (W Θ W = W)
  fz = KronQuadratic(W, W, S - 2W)
  Z = SymmetricSparseIterate(Matrix(Symmetric(Θ + Matrix(D), :L)))
  @test gradient(fz, Z) ≈ gradient(f, D) rtol = 100p * eps(T) atol = 100p * eps(T)
  @test gradient(f, Z) ≉ gradient(f, D)
end

@testset "Scaled and symmetric parameters ($T)" for T in (Float32, Float64)
  rng = Xoshiro(5)
  p = 4
  A, B, C = random_symmetric(rng, T, p), random_symmetric(rng, T, p), random_symmetric(rng, T, p)
  f = KronQuadratic(A, B, C)
  α = T(0.5)
  s = Scaled(f, α)
  X = T.(randn(rng, p, p))
  Xs = random_iterate(rng, T, p, 1:2:10)
  for Y in (X, Xs)
    @test s(Y) ≈ α * f(Y)
    @test gradient(s, Y) ≈ α * gradient(f, Y)
    v, G = value_and_gradient(s, Y)
    @test v ≈ α * f(Y) && G ≈ α * gradient(f, Y)
    H = zeros(T, p, p)
    @test gradient!(H, s, Y) === H
    @test H ≈ α * gradient(f, Y)
  end
  @test lipschitz(s) ≈ α * lipschitz(f)
  H = fill(T(3), p, p)
  @test_throws DimensionMismatch gradient!(H, s, zeros(T, p, p + 1))
  @test_throws DimensionMismatch gradient!(H, s, SymmetricSparseIterate(T, p + 1))
  @test all(==(3), H)

  @test ProximalBase.check_symmetric_parameters(f, Xs)
  @test ProximalBase.check_symmetric_parameters(KronQuadratic(A, B), Xs)
  @test ProximalBase.check_symmetric_parameters(Scaled(f, α), Xs)
  Ca = copy(C); Ca[1, 2] += 1
  fa = KronQuadratic(A, B, Ca)
  @test_throws ArgumentError ProximalBase.check_symmetric_parameters(fa, Xs)
  @test_throws ArgumentError ProximalBase.check_symmetric_parameters(Scaled(fa, α), Xs)
  @test ProximalBase.check_symmetric_parameters(fa, X)
end

@testset "failures before writes ($T)" for T in (Float32, Float64)
  A, B = T[2 1; 1 3], T[1 0 1; 0 2 0; 1 0 3]
  C = T[1 0 0; 0 0 -1]
  # construction
  @test_throws DimensionMismatch KronQuadratic(A, B, T[1 0; 0 1])
  @test_throws DimensionMismatch KronQuadratic(T[1 2 3; 2 1 0], B)
  @test_throws DimensionMismatch KronQuadratic(A, T[1 2], C)
  @test_throws ArgumentError KronQuadratic(zeros(T, 0, 0), B)
  @test_throws ArgumentError KronQuadratic(A, zeros(T, 0, 0))
  @test_throws "the KronQuadratic matrix A must be symmetric" KronQuadratic(T[2 1; 0 3], B)
  @test_throws "the KronQuadratic matrix B must be symmetric" KronQuadratic(A, T[1 0 1; 0 2 0; 2 0 3])
  @test_throws "finite entries in A" KronQuadratic(T[NaN 0; 0 1], B)
  @test_throws "finite entries in B" KronQuadratic(A, T[Inf 0 0; 0 1 0; 0 0 1])
  @test_throws "finite entries in C" KronQuadratic(A, B, T[1 0 0; 0 -Inf 0])
  @test_throws ArgumentError KronQuadratic(OffsetArray(A, -1, -1), B)
  @test_throws ArgumentError KronQuadratic(A, B, OffsetArray(C, 0, -1))
  other = T === Float64 ? Float32 : Float64
  @test_throws "$T.(C)" KronQuadratic(A, B, Int[1 0 0; 0 0 -1])
  @test_throws "$T.(C)" KronQuadratic(A, B, Bool[1 0 0; 0 0 1])
  @test_throws "$T.(B)" KronQuadratic(A, other.(B), C)
  @test_throws "$T.(B)" KronQuadratic(A, other.(B))
  @test_throws "Float64.(A) and Float64.(B)" KronQuadratic([2 1; 1 3], [1;;])
  U = Union{Float32, Float64}
  @test_throws "Float64.(A) and Float64.(B)" KronQuadratic(U[1.0;;], U[1.0;;])
  @test_throws ArgumentError KronQuadratic(U[1.0;;], U[1.0;;], U[1.0;;])
  for U in (Int, BigFloat, Float16, Complex{T})
    @test_throws ArgumentError KronQuadratic(U.(A), U.(B))
    @test_throws ArgumentError KronQuadratic(U.(A), U.(B), U.(C))
  end

  # oracles: each preflight failure below leaves the destination and the data unchanged
  f = KronQuadratic(A, B, C)
  d = ProximalBase.kronquadratic_data(f)
  X = T[1 0 2; -1 1 0]
  G = fill(T(-7), 2, 3)
  function unchanged(op, out, args...)
    before = copy(out)
    data = (copy(d.A), copy(d.B), copy(d.C))
    try
      op(out, args...)
    catch err
      return out == before && (d.A, d.B, d.C) == data ? typeof(err) : :changed
    end
    :no_error
  end
  for op in (gradient!, value_and_gradient!)
    @test unchanged(op, G, f, zeros(T, 3, 2)) === DimensionMismatch
    @test unchanged(op, fill(T(-7), 3, 2), f, X) === DimensionMismatch
    @test unchanged(op, G, f, other.(X)) === ArgumentError
    @test unchanged(op, fill(other(-7), 2, 3), f, X) === ArgumentError
    @test unchanged(op, G, f, OffsetArray(X, 1, 0)) === ArgumentError
    @test unchanged(op, OffsetArray(G, 0, 1), f, X) === ArgumentError
    @test unchanged(op, X, f, X) === ArgumentError
    backing = zeros(T, 2, 6)
    backing[:, 1:3] = X
    @test unchanged(op, view(backing, :, 1:3), f, view(backing, :, 1:3)) === ArgumentError
    @test unchanged(op, view(backing, :, 2:4), f, view(backing, :, 1:3)) === ArgumentError
    @test unchanged(op, d.C, f, X) === ArgumentError
    @test unchanged(op, reshape(view(d.B, 1:6), 2, 3), f, X) === ArgumentError
    @test unchanged(op, zeros(T, 6), f, vec(X)) in (MethodError, ArgumentError)
  end
  @test_throws MethodError f(vec(X))
  @test_throws DimensionMismatch f(zeros(T, 3, 2))
  @test_throws ArgumentError f(other.(X))
  @test_throws ArgumentError f(OffsetArray(X, 1, 1))

  # square functions and symmetric iterates
  Cs = T[1 2; 2 0]
  f = KronQuadratic(A, T[3 1; 1 2], Cs)
  d = ProximalBase.kronquadratic_data(f)
  Xs = SymmetricSparseIterate(T[1 2; 2 -1])
  G = fill(T(-7), 2, 2)
  for op in (gradient!, value_and_gradient!)
    @test unchanged(op, G, f, SymmetricSparseIterate(T, 3)) === DimensionMismatch
    @test unchanged(op, fill(T(-7), 3, 3), f, Xs) === DimensionMismatch
    @test unchanged(op, G, f, SymmetricSparseIterate(other[1 2; 2 -1])) === ArgumentError
    @test unchanged(op, fill(other(-7), 2, 2), f, Xs) === ArgumentError
    @test unchanged(op, matrixview(SymmetricSparseIterate(T, 2)), f, Xs) === ArgumentError
    @test unchanged(op, Symmetric(G), f, Xs) === ArgumentError
    @test unchanged(op, SymmetricSparseIterate(T, 2), f, Xs) === ArgumentError
    @test unchanged(op, d.A, f, Xs) === ArgumentError
    @test unchanged(op, OffsetArray(G, 1, 1), f, Xs) === ArgumentError
    @test unchanged(op, matrixview(SymmetricSparseIterate(T, 2)), f, X[:, 1:2]) === ArgumentError
    @test unchanged(op, Symmetric(G), f, X[:, 1:2]) === ArgumentError
  end
  @test_throws DimensionMismatch f(SymmetricSparseIterate(T, 3))
  # an asymmetric C is refused at symmetric iterates, and accepted for matrix arguments
  fa = KronQuadratic(A, T[3 1; 1 2], T[1 2; 0 0])
  for op in (gradient!, value_and_gradient!)
    @test unchanged(op, G, fa, Xs) === ArgumentError
  end
  @test_throws "exactly symmetric C" fa(Xs)
  @test_throws ArgumentError gradient(fa, Xs)
  @test_throws ArgumentError value_and_gradient(fa, Xs)
  @test fa(Matrix(Xs)) isa T
  # symmetric iterates need a square function
  f = KronQuadratic(A, B, C)
  G = fill(T(-7), 2, 2)
  @test unchanged(gradient!, G, f, SymmetricSparseIterate(T, 2)) === DimensionMismatch
  @test_throws DimensionMismatch f(SymmetricSparseIterate(T, 2))
  @test_throws DimensionMismatch gradient(f, SymmetricSparseIterate(T, 2))
  @test all(==(-7), G)
end

@testset "aliases that the checks detect ($T)" for T in (Float32, Float64)
  # each case returns the error type and checks that the destination and the argument are unchanged
  function refused(op, G, g, X)
    Gb, Xb = copy(G), X isa SymmetricSparseIterate ? copy(X) : copy(Matrix(X))
    try
      op(G, g, X)
    catch err
      same = X isa SymmetricSparseIterate ? isequal(X, Xb) : isequal(Matrix(X), Xb)
      return isequal(G, Gb) && same ? typeof(err) : :changed
    end
    :no_error
  end
  # the storage of a symmetric iterate: its active values, reshaped to the 1 × 1 destination
  h = KronQuadratic(T[2;;], T[3;;], T[1;;])
  for g in (h, Scaled(h, T(2))), op in (gradient!, value_and_gradient!)
    S = SymmetricSparseIterate(T, 1); S[1, 1] = 5
    G = reshape(activevalues(S), 1, 1)
    @test G isa StridedMatrix
    @test refused(op, G, g, S) === ArgumentError
    @test refused(op, G, g, matrixview(S)) === ArgumentError
    @test S[1, 1] == 5
  end
  # two slabs of one array are disjoint for Base.mightalias: accepted, with the values of independent arrays
  f = KronQuadratic(T[2 1; 1 3], T[1 0; 0 2], T[1 2; 2 1])
  M = zeros(T, 2, 2, 2)
  M[:, :, 1] = T[1 2; 3 4]
  X, G = view(M, :, :, 1), view(M, :, :, 2)
  for g in (f, Scaled(f, T(2)))
    v, H = value_and_gradient(g, T[1 2; 3 4])
    @test value_and_gradient!(G, g, X) == v
    @test G == H
    @test X == T[1 2; 3 4]
    # overlapping slabs of one array are refused
    N = ones(T, 2, 3)
    @test refused(value_and_gradient!, view(N, :, 1:2), g, view(N, :, 2:3)) === ArgumentError
  end
end

@testset "route predicates at the boundaries 2n = p²/8 and, for the value, 2n = p²/16 ($T)" for T in (Float32, Float64)
  p = 8                                         # p²/8 = 8: n = 4 accumulates, n = 5 takes the dense copy
  X = SymmetricSparseIterate(T, p)
  for k in 1:4
    X[k + 1, k] = T(k)
  end
  @test !ProximalBase._kron_dense_route(X, p)
  X[8, 1] = 1
  @test ProximalBase._kron_dense_route(X, p)
  # the value-only cutoff 2n = p²/16 = 4: n = 2 accumulates, n = 3 takes the dense copy
  Yv = SymmetricSparseIterate(T, p)
  Yv[2, 1] = 1; Yv[3, 3] = 2
  @test !ProximalBase._kron_dense_route(Yv, p, 32)
  Yv[5, 4] = 3
  @test ProximalBase._kron_dense_route(Yv, p, 32)
  # both routes agree with the formula on each side of both boundaries (n = 2, 3 for the value; n = 4, 5 for the gradient)
  rng = Xoshiro(17)
  f = KronQuadratic(random_symmetric(rng, T, p), random_symmetric(rng, T, p), random_symmetric(rng, T, p))
  d = ProximalBase.kronquadratic_data(f)
  for n in (2, 3, 4, 5)
    Y = SymmetricSparseIterate(T, p)
    for k in 1:n
      Y[k + 1, k] = T(k) / 4
    end
    Ym = Float64.(Matrix(Y))
    Gscale = maximum(kron_gradient_scale(Float64.(d.A), Float64.(d.B), Float64.(d.C), Ym))
    scale = kron_scale(Float64.(d.A), Float64.(d.B), Float64.(d.C), Ym)
    @test maximum(abs.(gradient(f, Y) .- symmetric_reference(Float64.(d.A), Float64.(d.B), Float64.(d.C), Ym))) <=
          (2p + 2) * eps(T) * Gscale
    @test abs(f(Y) - kron_value(Float64.(d.A), Float64.(d.B), Float64.(d.C), Ym)) <= (p^2 + 2p + 4) * eps(T) * scale
  end
end

@testset "reversed columns and allocating oracles ($T)" for T in (Float32, Float64)
  rng = Xoshiro(23)
  p, q = 35, 12
  f = KronQuadratic(random_symmetric(rng, T, p), random_symmetric(rng, T, q), T.(randn(rng, p, q)))
  X = T.(randn(rng, p, q))
  v, G = value_and_gradient(f, X)
  d = ProximalBase.kronquadratic_data(f)
  scale = kron_scale(Float64.(d.A), Float64.(d.B), Float64.(d.C), Float64.(X))
  Gscale = maximum(kron_gradient_scale(Float64.(d.A), Float64.(d.B), Float64.(d.C), Float64.(X)))
  M = X[:, end:-1:1]
  Xr = view(M, :, q:-1:1)                       # the same matrix through a negative column stride
  @test Xr == X && stride(Xr, 2) < 0
  Gr = view(zeros(T, p, q), :, q:-1:1)
  for (Y, D) in ((Xr, zeros(T, p, q)), (X, Gr), (Xr, Gr))
    @test value_and_gradient!(D, f, Y) ≈ v atol = 4 * (p * q + p + q) * eps(T) * scale
    @test D ≈ G atol = 4 * (p + q) * eps(T) * Gscale
    @test value_allocs(f, Y) == 0
    @test gradient_allocs(D, f, Y) == 0
    @test value_gradient_allocs(D, f, Y) == 0
  end
  # the generic allocating oracles return a Matrix{T} for matrices and strided views, directly and through Scaled
  g = KronQuadratic(T[2 1; 1 3], T[1 0; 0 2])
  Y = zeros(T, 2, 4); Y[:, 1:2:4] = T[1 2; 3 4]
  for h in (g, Scaled(g, T(2)))
    @test gradient(h, view(Y, :, 1:2:4)) isa Matrix{T}
    @test gradient(h, view(Y, :, 1:2:4)) == gradient(h, T[1 2; 3 4])
    w, H = value_and_gradient(h, view(Y, :, 1:2:4))
    @test H isa Matrix{T} && w == h(T[1 2; 3 4]) && H == gradient(h, T[1 2; 3 4])
    @test_throws DimensionMismatch gradient(h, zeros(T, 2, 3))
  end
end

@testset "allocations and inference ($T)" for T in (Float32, Float64)
  rng = Xoshiro(8)
  p, q = 6, 4
  f = KronQuadratic(random_symmetric(rng, T, p), random_symmetric(rng, T, q), T.(randn(rng, p, q)))
  X = T.(randn(rng, p, q))
  rows = zeros(T, 2p, q); rows[1:2:end, :] = X
  Xr = view(rows, 1:2:2p, :)
  G = zeros(T, p, q)
  Gr = view(zeros(T, 2p, q), 1:2:2p, :)
  for Y in (X, Xr), D in (G, Gr)
    @test value_allocs(f, Y) == 0
    @test gradient_allocs(D, f, Y) == 0
    @test value_gradient_allocs(D, f, Y) == 0
    @test @inferred(f(Y)) isa T
    @test @inferred(value_and_gradient!(D, f, Y)) isa T
    @test @inferred(gradient!(D, f, Y)) === D
  end
  n = p * (p + 1) ÷ 2
  for C in (random_symmetric(rng, T, p), nothing)
    g = KronQuadratic(random_symmetric(rng, T, p), random_symmetric(rng, T, p), C)
    G = zeros(T, p, p)
    Gr = view(zeros(T, 2p, p), 1:2:2p, :)
    for X in (random_iterate(rng, T, p, [2, 9]), random_iterate(rng, T, p, 1:n)), D in (G, Gr)
      @test value_allocs(g, X) == 0
      @test gradient_allocs(D, g, X) == 0
      @test value_gradient_allocs(D, g, X) == 0
      @test @inferred(g(X)) isa T
      @test @inferred(value_and_gradient!(D, g, X)) isa T
    end
  end
end

end
