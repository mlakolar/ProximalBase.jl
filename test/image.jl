module ImageTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random
using ..TestUtils

const PB = ProximalBase
Random.seed!(180)

# Exercise the operator interface without inheriting matrix indexing.
struct ImageMap{M}
  A::M
end
Base.size(A::ImageMap, d::Integer) = size(A.A, d)
Base.eltype(A::ImageMap) = eltype(A.A)
Base.adjoint(A::ImageMap) = ImageMap(A.A')
LinearAlgebra.mul!(y, A::ImageMap, x) = mul!(y, A.A, x)

@testset "LinearImage quadratic" begin
  for T in (Float64, Float32)
    rtol, atol = T === Float64 ? (1e-10, 1e-11) : (3e-5, 3e-6)
    n, m = 5, 3
    A = randn(T, n, n)
    Q = A' * A + T(2) * I
    q, c = randn(T, n), T(0.7)
    L, y = randn(T, m, n), randn(T, m)
    f = QuadraticFunction(Q, q, c)
    g = LinearImage(f, L)
    S = L * (Q \ L')
    H = inv(S)
    y0 = -L * (Q \ q)
    minimum_value = c - dot(q, Q \ q) / 2
    reference(y) = dot(y - y0, H * (y - y0)) / 2 + minimum_value
    check_traits(g; proximable = true, convex = true)
    @test g.cache.valuecache === nothing
    for γ in T.((0.4, 0.9, 0.4))
      p = prox(g, y, γ)
      @test p ≈ (I + γ * H) \ (y + γ * H * y0) rtol = rtol atol = atol
      @test g.cache.valuecache === nothing
      z = similar(y)
      @test value_and_prox!(z, g, y, γ) ≈ reference(p) rtol = rtol atol = atol
      @test z ≈ p rtol = rtol atol = atol
      latent = copy(g.cache.latent)
      @test L * latent ≈ p rtol = rtol atol = atol
      @test Q * latent + q + L' * (p - y) / γ ≈ zeros(T, n) atol = 20atol
      # Fenchel equality tests g*(u) = f*(L'u) independently of the prox formula.
      u = (y - p) / γ
      a = L' * u - q
      @test dot(u, p) - reference(p) ≈ dot(a, Q \ a) / 2 - c rtol = rtol atol = atol
    end
    @test g(y) ≈ reference(y) rtol = rtol atol = atol
    K = [Q L'; L zeros(T, m, m)]
    latent = (K \ [-q; y])[1:n]
    @test g(y) ≈ f(latent) rtol = rtol atol = atol
    @test g(y) isa T
    @test value_and_prox!(similar(y), g, y, T(0.4)) isa T
    check_prox(g, y, T(0.4); rtol = rtol, atol = atol)
    @test prox(g, zeros(T, m), T(0.4)) ≈
      L * ((T(0.4) * Q + L' * L) \ (-T(0.4) * q)) rtol = rtol atol = atol
  end
end

@testset "profiling and the Frisch–Waugh–Lovell identity" begin
  Z = randn(12, 2)
  X = [randn(12, 2) Z[:, 1]]
  A, r = [X Z], randn(12)
  f = QuadraticFunction(A' * A, -A' * r, dot(r, r) / 2)
  L = [Matrix{Float64}(I, 3, 3) zeros(3, 2)]
  g = LinearImage(f, L)
  U = Matrix(qr(Z).Q)[:, 1:2]
  M = I - U * U'
  θ = randn(3)
  @test rank(A) < size(A, 2)
  @test g(θ) ≈ norm(M * (r - X * θ))^2 / 2 rtol = 1e-10
  H, b = X' * M * X, -X' * M * r
  @test prox(g, θ, 0.6) ≈ (I + 0.6H) \ (θ - 0.6b) rtol = 1e-10
  check_prox(g, θ, 0.6)
end

@testset "roughness and structured data" begin
  n = 8
  D = zeros(n - 2, n)
  for i in axes(D, 1)
    D[i, i:(i + 2)] = [1, -2, 1]
  end
  L = Matrix{Float64}(I, n, n)[[1, 4, 8], :]
  y, q = randn(3), randn(n)
  for ε in (0.0, 0.1)
    Q = D' * D + ε * I
    dense = LinearImage(QuadraticFunction(Q, q, 0.3), L)
    sparse_image = LinearImage(QuadraticFunction(sparse(Q), q, 0.3), sparse(L))
    for γ in (0.3, 0.8, 0.3)
      @test prox(dense, y, γ) ≈ prox(sparse_image, y, γ) rtol = 1e-10
      @test value_and_prox!(similar(y), dense, y, γ) ≈
        value_and_prox!(similar(y), sparse_image, y, γ) rtol = 1e-10
    end
    @test dense(y) ≈ sparse_image(y) rtol = 1e-10
    K = [Q L'; L zeros(3, 3)]
    latent = (K \ [-q; y])[1:n]
    @test dense(y) ≈ dot(latent, Q * latent) / 2 + dot(q, latent) + 0.3 rtol = 1e-10
  end
  for T in (Float32, Float64), map in (Matrix{T}(I, n, n)[1:1, :], sparse(Matrix{T}(I, n, n)[1:1, :]))
    # A single grid value leaves the affine slope unobserved.
    g = LinearImage(QuadraticFunction(T.(D' * D)), map)
    @test_throws ArgumentError prox(g, T[0.4], T(0.6))
    @test_throws ArgumentError g(T[0.4])
  end
  for Q in (Diagonal(fill(2.0, n)), SymTridiagonal(fill(3.0, n), fill(-1.0, n - 1)),
            Tridiagonal(fill(-1.0, n - 1), fill(3.0, n), fill(-1.0, n - 1)),
            Bidiagonal(fill(2.0, n), zeros(n - 1), :U))
    g = LinearImage(QuadraticFunction(Q), L)
    @test issparse(g.cache.gram)
    @test prox(g, y, 0.4) ≈ prox(LinearImage(QuadraticFunction(Matrix(Q)), L), y, 0.4) rtol = 1e-10
  end
  for T in (Float32, Float64)
    g = LinearImage(QuadraticFunction(sparse(T[2 0; 0 3]), T[0.2, -0.3], T(0.4)), sparse(T[1 0]))
    @test prox(g, T[1], T(0.5)) ≈ T[0.45] rtol = (T === Float32 ? 1e-6 : 1e-12)
    @test g(T[1]) isa T
    @test value_and_prox!(T[0], g, T[1], T(0.5)) isa T
  end
end

@testset "numerical positive definiteness" begin
  for T in (Float32, Float64), map in (T[1 0], sparse(T[1 0]))
    # These strictly positive pivots pass the old one-rounding-level cutoff.
    # The enlarged cutoff must reject them in both the prox and reduced solve.
    δ = T(32 * 2) * eps(T)
    near_singular = LinearImage(QuadraticFunction(T[1 0; 0 δ]), map)
    @test_throws ArgumentError prox(near_singular, T[1], one(T))
    @test_throws ArgumentError near_singular(T[1])

    # A resolved small eigenvalue remains valid: the proximal system has
    # condition number about 4e12 in Float64 and 8e3 in Float32.
    δ = T(1024 * 2) * eps(T)
    g = LinearImage(QuadraticFunction(T[1 0; 0 δ], T[-1, δ]), map)
    @test prox(g, T[2], one(T)) ≈ T[1.5] rtol = 8eps(T)
    @test g.cache.latent ≈ T[1.5, -1] rtol = 8eps(T)
    @test g(T[1]) ≈ -one(T) / 2 - δ / 2 rtol = 8eps(T)
    @test value_and_prox!(T[0], g, T[2], one(T)) ≈ -T(0.375) - δ / 2 rtol = 8eps(T)
  end
end

@testset "rank-deficient images and empty dimensions" begin
  for matrix in ([1.0 0; 2 0], [1.0 0; 0 1; 1 1], zeros(3, 2),
                 zeros(0, 2), zeros(3, 0), zeros(0, 0)), L in (matrix, sparse(matrix))
    m, n = size(L)
    Q, q, c = Matrix{Float64}(I, n, n), randn(n), 0.8
    f = QuadraticFunction(Q, q, c)
    g = LinearImage(f, L)
    latent, y = randn(n), randn(m)
    feasible = L * latent
    shift = -L * q
    reference = c - dot(q, q) / 2 + norm(pinv(matrix) * (feasible - shift))^2 / 2
    @test g(feasible) ≈ reference rtol = 1e-10 atol = 1e-12
    p = prox(g, y, 0.6)
    @test p ≈ L * ((0.6Q + L' * L) \ (L' * y - 0.6q)) rtol = 1e-10 atol = 1e-12
    z = similar(y)
    @test value_and_prox!(z, g, y, 0.6) ≈ g(p) rtol = 1e-10 atol = 1e-12
    if m > rank(matrix)
      outside = Matrix(svd(matrix; full = true).U)[:, end]
      @test g(outside) == Inf
    end
    isempty(y) || check_prox(g, y, 0.6; local_points = false)
  end
  Q = [2.0 0.3; 0.3 3.0]
  q = [0.4, -0.5]
  g = LinearImage(QuadraticFunction(Q, q, 0.7), zeros(2, 2))
  @test prox(g, ones(2), 0.3) == zeros(2)
  @test g(zeros(2)) ≈ 0.7 - dot(q, Q \ q) / 2
  @test g([1.0, 0.0]) == Inf
  # A latent null direction remains unobserved and unpenalized.
  bad = LinearImage(QuadraticFunction(Diagonal([1.0, 0.0, 0.0])), [1.0 0.0 0.0])
  @test_throws ArgumentError prox(bad, ones(1), 0.4)
  @test_throws ArgumentError bad(ones(1))
end

@testset "orthogonal columns" begin
  μ = 2.3
  L = sqrt(μ) .* Matrix(qr(randn(6, 3)).Q)[:, 1:3]
  y = randn(6)
  for f in (NormL1(0.4), SqrNormL2(0.7, randn(3)), IndBox(-0.5, 0.8))
    for map in (L, sparse(L), ImageMap(L))
      g = LinearImage(f, map, μ)
      check_traits(g; proximable = true, convex = true)
      p = check_prox(g, y, 0.6; local_points = false)
      z = prox(f, L' * y / μ, 0.6 / μ)
      @test p ≈ L * z rtol = 1e-12 atol = 1e-13
      @test g(p) ≈ f(z) rtol = 1e-12 atol = 1e-13
      @test g(y) == Inf
      @test g(p) ≈ Precompose(f, L' / μ, inv(μ))(p) rtol = 1e-12 atol = 1e-13
      @test value_and_prox!(similar(y), g, y, 0.6) ≈ f(z) rtol = 1e-12 atol = 1e-13
    end
  end
  indicator = LinearImage(IndBox(-1, 1), L, μ)
  @test PB._isindicator(indicator)
  f = SqrNormL2(0.7)
  g = LinearImage(f, L, μ)
  p = prox(g, y, 0.6)
  u = (y - p) / 0.6
  @test dot(u, p) - g(p) ≈ Conjugate(f)(L' * u) rtol = 1e-12
  inner = Conjugate(ElasticNet(0.3, 0.7))
  h = LinearImage(inner, L, μ)
  z = similar(y)
  latent = prox(inner, L' * y / μ, 0.6 / μ)
  inner_value = value_and_prox!(similar(latent), inner, L' * y / μ, 0.6 / μ)
  @test value_and_prox!(z, h, y, 0.6) ≈ inner_value rtol = 1e-12
  @test z ≈ L * latent rtol = 1e-12
  @test_throws ArgumentError h(z)
  @test prox(LinearImage(NormL1(), zeros(2, 0), 1), ones(2), 0.5) == zeros(2)
  @test LinearImage(NormL1(), zeros(2, 0), 1)(zeros(2)) == 0
  @test LinearImage(NormL1(), zeros(2, 0), 1)(ones(2)) == Inf
  @test isempty(prox(LinearImage(NormL1(), zeros(0, 0), 1), Float64[], 0.5))
end

@testset "precision, scales and nonfinite arguments" begin
  for T in (Float32, Float64)
    L = reshape(T[1, 0, 0], 3, 1)
    g = LinearImage(NormL1(1.0), L, 1.0)
    large = floatmax(T) / T(2)
    @test g(T[0, large, large]) == T(Inf)
    @test g(T[1, sqrt(eps(T)) / 4, 0]) ≈ one(T)
    @test g(T[1, 2sqrt(eps(T)), 0]) == T(Inf)
    @test g(zeros(T, 3)) === zero(T)
    @test value_and_prox!(zeros(T, 3), g, ones(T, 3), 0.5) isa T
    scale, argument = T === Float64 ? (T(1e150), T(1e200)) : (T(1e15), T(1e30))
    μ = Float64(scale)^2
    huge = LinearImage(NormL1(one(T)), reshape([scale], 1, 1), μ)
    @test huge([argument]) ≈ argument / scale rtol = 4eps(T)
    @test prox(huge, [argument], one(T)) ≈ [argument] rtol = 4eps(T)
    qimage = LinearImage(QuadraticFunction(T[2 0; 0 3], T[0.2, -0.1], T(0.3)), T[1 0; 0 1])
    for h in (g, qimage)
      n = size(h.L, 1)
      for bad in (T(NaN), T(Inf), T(-Inf))
        x = fill(one(T), n)
        x[end] = bad
        @test h(x) === T(Inf)
        @test all(isnan, prox(h, x, one(T)))
        @test isnan(value_and_prox!(similar(x), h, x, one(T)))
      end
    end
    for S in (Float32, Float64)
      x, z = ones(S, 2), zeros(S, 2)
      @test eltype(prox(qimage, x, 0.4)) === S
      @test typeof(qimage(x)) === typeof(value_and_prox!(z, qimage, x, 0.4)) === S
      oimage = LinearImage(SqrNormL2(1.0), Matrix{T}(I, 2, 2), 1.0)
      @test typeof(oimage(x)) === typeof(value_and_prox!(z, oimage, x, 0.4)) === S
    end
  end
  # Only Q and L determine the precision of a quadratic image's buffers; the
  # value has the argument's precision.
  mixed = LinearImage(QuadraticFunction(Float32[2 0; 0 3], [0.2, 0.3], 0.4), Float32[1 0])
  @test mixed([1.0]) isa Float64
  @test value_and_prox!([0.0], mixed, [1.0], 0.5) isa Float64
end

@testset "cache reset through wrappers" begin
  Q, q, L = [2.0 0.3; 0.3 3.0], [0.4, -0.1], [1.0 0.2]
  f = QuadraticFunction(Q, q, 0.5)
  g = LinearImage(f, L)
  outer = Scaled(Translate(g, [0.2]), 1.3)
  x = [0.8]
  prox(outer, x, 0.4)
  outer(x)
  @test g.cache.factor !== nothing && g.cache.valuecache !== nothing
  Q[1, 1] = 4
  L[1, 2] = -0.4
  q[2] = 0.7
  f.c = 1.2
  @test reset_cache!(outer) === outer
  @test g.cache.factor === nothing && g.cache.valuecache === nothing
  fresh = Scaled(Translate(LinearImage(QuadraticFunction(Q, q, f.c), L), [0.2]), 1.3)
  @test outer(x) ≈ fresh(x) rtol = 1e-12
  @test prox(outer, x, 0.4) ≈ prox(fresh, x, 0.4) rtol = 1e-12
  @test value_and_prox!(similar(x), outer, x, 0.4) ≈
    value_and_prox!(similar(x), fresh, x, 0.4) rtol = 1e-12
  inner = QuadraticFunction(copy(Q), copy(q))
  orthogonal = LinearImage(inner, Matrix{Float64}(I, 2, 2), 1.0)
  prox(orthogonal, ones(2), 0.5)
  @test !isempty(inner.Λ)
  reset_cache!(orthogonal)
  @test isempty(inner.Λ)
end

function unchanged(f, x)
  for op in (prox!, value_and_prox!), alias in (false, true)
    input = deepcopy(x)
    output = alias ? input : deepcopy(x)
    before = deepcopy(output)
    @test_throws DimensionMismatch op(output, f, input, 0.5)
    @test output == before
    @test input == x
  end
end

@testset "validation before writes" begin
  L = Matrix{Float64}(I, 2, 2)
  for g in (LinearImage(SqrNormL2(1.0, ones(3)), L, 1),
            LinearImage(NormL1(1.0, ones(3)), L, 1))
    x = [2.0, -1.0]
    for wrapper in (g, Tilt(g, ones(2)), Translate(g, ones(2)), Scaled(g, 2), Conjugate(g))
      unchanged(wrapper, x)
    end
    unchanged(SeparableSum(ZeroFunction(), g), (copy(x), copy(x)))
  end
  for g in (LinearImage(NormL1(), L, 1), LinearImage(QuadraticFunction(L), L))
    unchanged(g, ones(3))
    unchanged(g, ones(2, 1))
    @test_throws DimensionMismatch g(ones(3))
    @test_throws DimensionMismatch prox!(ones(1), g, ones(2), 0.5)
    for γ in (0, -1, Inf, NaN)
      output = ones(2)
      @test_throws ArgumentError prox!(output, g, zeros(2), γ)
      @test output == ones(2)
    end
  end
  q = zeros(2)
  g = LinearImage(QuadraticFunction(copy(L), q), L)
  resize!(q, 3)
  for wrapper in (g, Tilt(g, ones(2)), Translate(g, ones(2)), Scaled(g, 2), Conjugate(g))
    unchanged(wrapper, ones(2))
  end
  unchanged(SeparableSum(ZeroFunction(), g), (ones(2), ones(2)))
  @test_throws DimensionMismatch LinearImage(QuadraticFunction(ones(3, 3)), L)
  @test_throws ArgumentError LinearImage(QuadraticFunction([1.0 1; 0 1]), L)
  @test_throws ArgumentError LinearImage(QuadraticFunction([Inf 0; 0 1]), L)
  @test_throws ArgumentError LinearImage(QuadraticFunction(L, [NaN, 0]), L)
  @test_throws ArgumentError LinearImage(QuadraticFunction(L, nothing, Inf), L)
  for μ in (0, -1, Inf, NaN)
    @test_throws ArgumentError LinearImage(NormL1(), L, μ; check = false)
  end
  @test_throws ArgumentError LinearImage(NormL1(), [1.0 1; 0 1], 1)
  @test LinearImage(NormL1(), [1.0 1; 0 1], 1; check = false) isa LinearImage
  @test_throws ArgumentError LinearImage(NormL1(), [Inf 0; 0 1], 1; check = false)
  @test_throws ArgumentError LinearImage(NormL1(), complex.(L), 1; check = false)
  @test_throws ArgumentError LinearImage(NormL0(), L, 1)
  @test_throws ArgumentError LinearImage(NormL1(), L)
  @test_throws ArgumentError LinearImage(LeastSquaresLoss(ones(2), L), L)
  @test_throws ArgumentError LinearImage(QuadraticFunction(L), ImageMap(L))
  try
    LinearImage(NormL1(), L)
  catch e
    @test occursin("QuadraticFunction", sprint(showerror, e))
    @test occursin("μI", sprint(showerror, e))
  end
end

@testset "hot dense oracles" begin
  for T in (Float32, Float64)
    Q = T[3 0.5; 0.5 2]
    L = T[1 0; 0 1; 0 0]
    x = T[0.6, -0.3, 0]
    y = similar(x)
    functions = (LinearImage(QuadraticFunction(Q, T[0.2, -0.1], T(0.4)), L),
      LinearImage(NormL1(T(0.3)), L, 1.0),
      LinearImage(SqrNormL2(T(0.7)), L, 1.0),
      LinearImage(IndBox(T(-1), T(1)), L, 1.0))
    for g in functions
      @test allocs(g, x) == 0
      @test allocs(prox!, y, g, x, T(0.4)) == 0
      @test allocs(value_and_prox!, y, g, x, T(0.4)) == 0
      alias = copy(x)
      @test allocs(prox!, alias, g, alias, T(0.4)) == 0
      @test allocs(value_and_prox!, alias, g, alias, T(0.4)) == 0
    end
  end
end

@testset "tall value decomposition" begin
  L = randn(2000, 3)
  Q = Matrix{Float64}(I, 3, 3)
  warm = LinearImage(QuadraticFunction(Q), L)
  point = randn(3)
  y = L * point
  @test warm(y) ≈ dot(point, point) / 2 rtol = 1e-12
  g = LinearImage(QuadraticFunction(Q), L)
  # Include construction of the value cache: a full left factor exceeds 32 MB.
  bytes = @allocated g(y)
  @test bytes < 2_000_000
  @test size(g.cache.valuecache.U) == (2000, 3)
  @test size(g.cache.valuecache.V) == (3, 3)
  @test size(g.cache.valuecache.N) == (3, 0)
  L[:, 3] .= L[:, 1]
  reset_cache!(g)
  @test g(y) == Inf
  @test size(g.cache.valuecache.U) == (2000, 2)
  @test size(g.cache.valuecache.N) == (3, 1)
end

@testset "orthogonal image scales retain their range" begin
  s = Float32(1e30)
  L = reshape([s], 1, 1)
  g = LinearImage(NormL1(s), L, Float64(s)^2)
  y = Float32[1]
  @test g(y) ≈ 1f0 rtol = 4eps(Float32)
  @test prox(g, y, 1f0) == Float32[0]
  z = similar(y)
  @test value_and_prox!(z, g, y, 1f0) == 0f0
  @test z == Float32[0]
  quadratic = LinearImage(QuadraticFunction(zeros(Float32, 1, 1), [s], 0f0), L, Float64(s)^2)
  @test prox(quadratic, y, 1f0) ≈ Float32[0] atol = eps(Float32)
  @test value_and_prox!(z, quadratic, y, 1f0) ≈ 0f0 atol = eps(Float32)
  for map in (reshape(Float32[1e-40], 1, 1), ImageMap(reshape(Float32[1e-40], 1, 1)))
    s = Float32(1e-40)
    g = LinearImage(ZeroFunction(), map, Float64(s)^2)
    @test g([s]) == 0f0
    @test prox(g, [s], 1f0) == [s]
    @test value_and_prox!(Float32[0], g, [s], 1f0) == 0f0
  end
  # Float64 can also need an exponent range beyond a machine scalar's.
  s, γ = 1e150, 1e-300
  g = LinearImage(NormL1(s, [s]), reshape([s], 1, 1), s^2)
  y = [2e-150]
  @test prox(g, y, γ) ≈ [1e-150] rtol = 4eps(Float64)
  @test value_and_prox!(zeros(1), g, y, γ) ≈ 1.0 rtol = 4eps(Float64)
end

@testset "small coordinates retain finite value contributions" begin
  g = LinearImage(LinearFunction([0.0, 1e20]), Matrix{Float64}(I, 2, 2), 1.0)
  @test g([1e308, 1e-20]) ≈ 1.0 rtol = 1e-15
  mixed = LinearImage(ZeroFunction(), Matrix{Float32}(I, 2, 2), 1.0)
  @test mixed([1e200, 1e-200]) == 0.0
  L = 1e200 * Matrix{Float64}(I, 2, 2)
  g = LinearImage(QuadraticFunction(zeros(2, 2), [0.0, 1e200], 0.0), L)
  @test g([1e200, 1.0]) ≈ 1.0 rtol = 1e-15
  @test g.cache.latent[2] ≈ 1e-200 rtol = 1e-15
end

@testset "scaled quadratic right-hand side and values" begin
  for map in (reshape([1e150], 1, 1), sparse(reshape([1e150], 1, 1)))
    g = LinearImage(QuadraticFunction(ones(1, 1)), map)
    y = [1e200]
    p = prox(g, y, 1.0)
    @test p ≈ y rtol = 1e-14
    @test g.cache.latent ≈ [1e50] rtol = 1e-14
    output = similar(y)
    @test value_and_prox!(output, g, y, 1.0) ≈ 5e99 rtol = 1e-14
    @test output ≈ y rtol = 1e-14
  end
  g = LinearImage(QuadraticFunction(reshape([2.0], 1, 1)), ones(1, 1))
  @test g([1e154]) ≈ 1e308 rtol = 1e-15
  output = zeros(1)
  @test value_and_prox!(output, g, [2e154], 0.5) ≈ 1e308 rtol = 1e-15
  @test output ≈ [1e154] rtol = 1e-15
end

end
