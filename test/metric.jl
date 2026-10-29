module MetricTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# a smooth function that does not know its per-coordinate constants
struct SmoothWithoutConstants <: AbstractFunction end
ProximalBase.issmooth(::Type{SmoothWithoutConstants}) = true

# the subgradient inequality in the metric Diagonal(1 ./ γ): p = prox_{γf}(x) when
# f(z) ≥ f(p) + ⟨(x - p) ./ γ, z - p⟩ for every z
function check_metric_optimality(f, x, γ, p; points = 5)
  fp = f(p)
  for _ in 1:points, δ in (1e-3, 1e-1, 1.)
    z = p .+ δ .* randn(size(p)...)
    fz = f(z)
    isfinite(fz) && @test fz >= fp + dot((x .- p) ./ γ, z .- p) - 1e-9 * max(1, abs(fp))
  end
end

@testset "a step per entry for separable functions" begin
  n = 12
  w = rand(n) .+ 0.2
  x = 2 .* randn(n)
  γ = rand(n) .+ 0.1
  for f in (NormL1(0.3), NormL1(0.3, w), NormL1Nonneg(0.2), ElasticNet(0.3, 0.4, w), IndBox(-0.5, 0.5),
            NormL0(0.05), MCP(0.3), SCAD(0.3))
    p = prox(f, x, γ)
    @test p == [prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
    @test prox(f, x, fill(0.7, n)) == prox(f, x, 0.7)        # equal steps give the scalar step
    z = copy(x)
    prox!(z, f, z, γ)
    @test z == p                                               # in place
    isconvex(f) && check_metric_optimality(f, x, γ, p)
    # the iterates: a SparseIterate, and a SymmetricSparseIterate with symmetric steps
    xs = SparseIterate(sprandn(n, 0.4))
    @test Vector(prox(f, xs, γ)) ≈ prox(f, Vector(xs), γ)
    if !(hasproperty(f, :w) && f.w isa AbstractArray)        # weights for a vector of length n
      D = randn(4, 4); D = D + D'
      G = rand(4, 4) .+ 0.1; G = G + G'
      S = SymmetricSparseIterate(D)
      @test Matrix(prox(f, S, G)) ≈ prox(f, D, G)
    end
    @test_throws DimensionMismatch prox(f, x, rand(n - 1))
  end
  # into a SymmetricSparseIterate that holds entries: from another iterate, from a
  # dense matrix, and with a function that does not preserve zeros
  D = randn(5, 5); D = D + D'
  G = rand(5, 5) .+ 0.1; G = G + G'
  for f in (NormL1(0.3), MCP(0.3), IndBox(0.1, 1.))
    out = SymmetricSparseIterate(randn(5, 5) |> A -> A + A')
    prox!(out, f, SymmetricSparseIterate(D), G)
    @test Matrix(out) ≈ prox(f, D, G)
    out = SymmetricSparseIterate(randn(5, 5) |> A -> A + A')
    prox!(out, f, D, G)
    @test Matrix(out) ≈ prox(f, D, G)
  end
end

@testset "shared scalar and diagonal-step kernels" begin
  for T in (Float32, Float64), dims in ((12,), (3, 4))
    x = randn(T, dims)
    γ = T(0.7)
    steps = reshape(view(fill(γ, 2length(x)), 1:2:2length(x)), dims)
    for f in (NormL1(T(0.3)), NormL1(T(0.3), rand(T, 12)), MCP(T(0.3)),
              SCAD(T(0.3)), IndBox(T(0.1), one(T)))
      for input in (x, SparseIterate(x))
        out = similar(input)
        @test Array(prox!(out, f, input, steps)) == Array(prox(f, input, γ))
        @test value_and_prox!(out, f, input, steps) ≈ f(out)
        @test allocs(prox!, out, f, input, γ) == 0
        @test allocs(prox!, out, f, input, steps) == 0
        alias = copy(input)
        @test Array(prox!(alias, f, alias, steps)) == Array(prox(f, input, γ))
        @test_throws DimensionMismatch prox!(out, f, input, zeros(T, 11))
      end
    end
  end
end

@testset "a step per group for group norms" begin
  groups = [1:3, 4:5, [6, 8], 9:12]               # x[7] is in no group
  x = 2 .* randn(12)
  γ = [0.2, 0.9, 0.5, 1.4]
  for F in (GroupNormL2, GroupNormLinf)
    f = F(0.5, groups)
    p = prox(f, x, γ)
    @test p[7] == x[7]
    for k in eachindex(groups)
      G = groups[k]
      @test p[G] ≈ prox(F(0.5, [1:length(G)], [f.w[k]]), x[G], γ[k])
    end
    @test prox(f, x, fill(0.6, 4)) ≈ prox(f, x, 0.6)
    @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
    # the metric is constant on each group
    γe = zeros(12); γe[7] = 1.0
    for k in eachindex(groups)
      γe[groups[k]] .= γ[k]
    end
    check_metric_optimality(f, x, γe, p)
    # an AtomIterate with the groups of f
    fa = F(0.5, [groups; [[7]]], [f.w; 0.])
    xa = AtomIterate(x, fa)
    ya = similar(xa)
    γa = [γ; 1.0]
    @test value_and_prox!(ya, fa, xa, γa) ≈ fa(prox(fa, x, γa))
    @test Vector(ya) ≈ prox(fa, x, γa)
    @test_throws DimensionMismatch prox(f, x, rand(3))
  end
end

@testset "a step per group for MCP and SCAD" begin
  groups = [1:2, [4, 6]]                    # internal and trailing free entries
  for T in (Float32, Float64), F in (GroupMCP, GroupSCAD)
    f = F(T(0.5), groups, T[1, 0.7])
    x = T[3, 4, 7, -1, 8, 2, 9, 10]
    steps = T[0.2, 4]                      # includes a nonconvex scalar subproblem
    p = prox(f, x, steps)
    @test eltype(p) == T
    @test p[[3, 5, 7, 8]] == x[[3, 5, 7, 8]]
    for k in eachindex(groups)
      G = groups[k]
      @test p[G] ≈ prox(F(f.λ0, [1:length(G)], [f.w[k]]; a=f.a), x[G], steps[k])
    end
    @test prox(f, x, fill(T(0.6), 2)) ≈ prox(f, x, T(0.6))
    # A strided step vector and a sparse iterate use the same per-group metric.
    @test prox(f, x, view(T[steps[1], 0, steps[2], 0], 1:2:4)) ≈ p
    @test Vector(prox(f, SparseIterate(x), steps)) ≈ p
    for input in (x, SparseIterate(x))
      out = similar(input)
      @test value_and_prox!(out, f, input, steps) ≈ f(p)
      @test Vector(out) ≈ p
      alias = copy(input)
      @test Vector(prox!(alias, f, alias, steps)) ≈ p
    end
    @test allocs(value_and_prox!, similar(x), f, x, steps) == 0
    @inferred value_and_prox!(similar(x), f, x, steps)
    @test_throws DimensionMismatch prox(f, x, steps[1:1])
    @test_throws DimensionMismatch prox(f, x, [steps; one(T)])

    # Shared groups retain the active-group path, including zero and free groups.
    fa = F(T(0.5), [1:2, 3:4, 5:6], T[1, 0.7, 0])
    xa = AtomIterate(T[3, 4, 0, 0, 2, -1], fa)
    γa = T[0.2, 4, 0.5]
    reference = prox(fa, Vector(xa), γa)
    ya = similar(xa)
    @test value_and_prox!(ya, fa, xa, γa) ≈ fa(reference)
    @test Vector(ya) ≈ reference
    alias = copy(xa)
    @test Vector(prox!(alias, fa, alias, γa)) ≈ reference
    @test allocs(value_and_prox!, ya, fa, xa, γa) == 0
    @inferred value_and_prox!(ya, fa, xa, γa)
    @test_throws DimensionMismatch prox(fa, xa, γa[1:2])
    # Equal but independently owned groups exercise the ordinary-array fallback.
    for layout in ([1:2, 3:4, 5:6], [1:3, 4:6])
      independent = AtomIterate(Vector(xa), layout)
      out = similar(independent)
      @test value_and_prox!(out, fa, independent, γa) ≈ fa(reference)
      @test Vector(out) ≈ reference
    end
    # A regularized group penalty still updates one atom through the shared API.
    regularized = Regularize(fa, T(0.3))
    alias = copy(xa)
    prox_coordinate!(alias, regularized, 1, T(0.2))
    @test Vector(alias)[1:2] ≈ prox(regularized, Vector(xa), T(0.2))[1:2]
  end
end

@testset "a step per group for the sparse group lasso" begin
  groups = [1:3, 4:5, [6, 8], 9:12]               # x[7] is in no group
  x = 2 .* randn(12)
  γ = [0.2, 0.9, 0.5, 1.4]
  f = SparseGroupNormL2(0.3, 0.5, groups)
  p = prox(f, x, γ)
  @test p[7] == x[7]
  for k in eachindex(groups)
    G = groups[k]
    @test p[G] ≈ prox(SparseGroupNormL2(0.3, 0.5, [1:length(G)], [f.w[k]]), x[G], γ[k])
  end
  @test prox(f, x, fill(0.6, 4)) ≈ prox(f, x, 0.6)
  @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
  @test prox!(similar(x), f, x, 0.6) ≈ prox(f, x, 0.6)   # prox! goes through value_and_prox!
  # an AtomIterate with the groups of f
  fa = SparseGroupNormL2(0.3, 0.5, [groups; [[7]]], [f.w; 0.])
  xa = AtomIterate(x, fa)
  ya = similar(xa)
  γa = [γ; 1.0]
  @test value_and_prox!(ya, fa, xa, γa) ≈ fa(prox(fa, x, γa))
  @test Vector(ya) ≈ prox(fa, x, γa)
  @test_throws DimensionMismatch prox(f, x, rand(3))
end

@testset "functions built from functions pass the steps on" begin
  n = 10
  x = 2 .* randn(n)
  γ = rand(n) .+ 0.1
  c, b = randn(n), randn(n)
  for f in (Scaled(NormL1(0.3, rand(n)), 2.), Tilt(NormL1(0.3), c), Translate(NormL1(0.3), b),
            Conjugate(NormL1(0.3)), Conjugate(IndBox(-1., 1.)), 0.5 * Translate(ElasticNet(0.2, 0.3), b))
    p = prox(f, x, γ)
    @test p ≈ [prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    @test prox(f, x, fill(0.4, n)) ≈ prox(f, x, 0.4)
    f isa Conjugate || @test value_and_prox!(similar(x), f, x, γ) ≈ f(p)
  end
  # Moreau's decomposition in each entry: x = prox_{γf}(x) + γ ⊙ prox_{f*/γ}(x ./ γ)
  f = NormL1(0.3)
  @test prox(f, x, γ) .+ γ .* prox(Conjugate(f), x ./ γ, 1 ./ γ) ≈ x
  # a scaled group norm with a step per group
  groups = [1:4, 5:10]
  g = GroupNormL2(0.4, groups)
  @test prox(2.0 * g, x, [0.3, 0.8]) ≈ prox(g, x, [0.6, 1.6])
end

@testset "per-coordinate Lipschitz constants" begin
  n, p = 30, 8
  X = randn(n, p)
  y = randn(n)
  H = X' * X
  # least squares, with a vector and with a matrix response
  f = LeastSquaresLoss(y, X)
  @test coordinate_lipschitz!(zeros(p), f) ≈ diag(H)
  @test [coordinate_lipschitz(f, k) for k in 1:p] ≈ diag(H)
  fm = LeastSquaresLoss(randn(n, 3), X)
  @test coordinate_lipschitz!(zeros(p, 3), fm) ≈ repeat(diag(H), 1, 3)
  @test coordinate_lipschitz(fm, p + 2) ≈ H[2, 2]
  @test_throws BoundsError coordinate_lipschitz(f, p + 1)
  @test_throws DimensionMismatch coordinate_lipschitz!(zeros(p + 1), f)
  # a sparse design reads its stored entries
  Xs = sprandn(n, p, 0.3)
  @test coordinate_lipschitz!(zeros(p), LeastSquaresLoss(y, Xs)) ≈ vec(sum(abs2, Matrix(Xs); dims = 1))
  # the diagonal of the Hessian, against finite differences of the gradient
  β = randn(p)
  for g in (f, QuadraticFunction(Symmetric(H) + I, randn(p)), (1 / n) * f, Sum(f, SqrNormL2(2.)),
            Tilt(f, randn(p)), Translate(f, randn(p)))
    for k in (1, 4, p)
      e = zeros(p); e[k] = 1e-4
      fd = (gradient(g, β .+ e)[k] - gradient(g, β .- e)[k]) / 2e-4
      @test coordinate_lipschitz(g, k) ≈ fd rtol = 1e-6
    end
  end
  # GLMs: the curvature bound times the weighted squared column norms, which bounds
  # the diagonal of the Hessian everywhere
  wobs = rand(n) .+ 0.5
  fl = LogisticLoss(Float64.(rand(n) .< 0.5), X; weights = wobs)
  L = coordinate_lipschitz!(zeros(p), fl)
  @test L ≈ 0.25 .* vec(sum(wobs .* X .^ 2; dims = 1))
  η = X * β
  hdiag = vec(sum(wobs .* (exp.(η) ./ (1 .+ exp.(η)) .^ 2) .* X .^ 2; dims = 1))
  @test all(hdiag .<= L)
  @test coordinate_lipschitz(HuberLoss(y, X, 1.0), 3) ≈ H[3, 3]
  Xsp = sprandn(n, p, 0.4)
  @test coordinate_lipschitz!(zeros(p), LogisticLoss(Float64.(rand(n) .< 0.5), Xsp; weights = wobs)) ≈
        0.25 .* vec(sum(wobs .* Matrix(Xsp) .^ 2; dims = 1))       # sparse, weighted
  @test coordinate_lipschitz(PoissonLoss(rand(0:3, n), X), 2) == Inf
  # Kronecker quadratics: A[r, r] B[c, c] for entry (r, c) of Θ
  A = randn(5, 5); A = A * A'
  B = randn(4, 4); B = B * B'
  fk = KronQuadratic(A, B)
  @test vec(coordinate_lipschitz!(zeros(5, 4), fk)) ≈ kron(diag(B), diag(A))
  @test coordinate_lipschitz(SqrNormL2(2.5), 3) == 2.5
  @test coordinate_lipschitz(ZeroFunction(), 1) == 0
  @test coordinate_lipschitz(LinearFunction(randn(p)), 2) == 0
  # the wrappers keep the constants of the loss, also all at once
  @test coordinate_lipschitz!(zeros(p), Tilt(f, randn(p))) ≈ diag(H)
  @test coordinate_lipschitz!(zeros(p), Translate(f, randn(p))) ≈ diag(H)
  @test coordinate_lipschitz!(zeros(p), Sum(f, SqrNormL2(2.))) ≈ diag(H) .+ 2
  @test coordinate_lipschitz!(zeros(p), LeastSquaresLoss(y, sprandn(n, p, 0.3))) isa Vector{Float64}
  @test coordinate_lipschitz(LogDetLoss(Matrix(1.0I, 3, 3)), 2) == Inf
  @test_throws ArgumentError coordinate_lipschitz(NormL1(1.), 1)
  @test_throws ArgumentError coordinate_lipschitz(SmoothWithoutConstants(), 1)
end

# The minimizer of λ‖z‖₁ + Σᵢ (zᵢ - xᵢ)²/(2γᵢ) on the box [l, u] with ⟨a, z⟩ = b
# (or ≤ b), by brute force: each entry is at a bound, at zero, or free with a
# sign, zᵢ = xᵢ - γᵢ(λσᵢ + νaᵢ), and ν makes the constraint hold. The best
# feasible candidate is the minimizer, which is one of them.
function brute_linear(x, γ, a, b, λ, l, u; halfspace = false)
  n = length(x)
  obj(z) = λ * norm(z, 1) + sum((z .- x) .^ 2 ./ (2 .* γ))
  feasible(z) = all(l .- 1e-12 .<= z .<= u .+ 1e-12) &&
                (halfspace ? dot(a, z) <= b + 1e-10 : abs(dot(a, z) - b) <= 1e-10)
  best, zbest = Inf, fill(NaN, n)
  consider(z) = (feasible(z) && obj(z) < best) && ((best, zbest) = (obj(z), z))
  halfspace && consider(clamp.(sign.(x) .* max.(abs.(x) .- γ .* λ, 0), l, u))   # inactive
  for states in Iterators.product(ntuple(_ -> 1:5, n)...)
    z = zeros(n)
    free = [s >= 4 for s in states]
    σ = [s == 4 ? 1.0 : s == 5 ? -1.0 : 0.0 for s in states]
    for i in 1:n
      states[i] == 1 && (z[i] = l)
      states[i] == 2 && (z[i] = u)
      states[i] == 3 && (z[i] = 0.0)
    end
    w = sum(γ[free] .* a[free] .^ 2; init = 0.0)
    w > 0 || continue
    rest = dot(a[.!free], z[.!free])
    ν = (rest + dot(a[free], x[free] .- γ[free] .* λ .* σ[free]) - b) / w
    halfspace && ν < 0 && continue
    z[free] .= x[free] .- γ[free] .* (λ .* σ[free] .+ ν .* a[free])
    all(sign.(z[free]) .== σ[free] .|| λ == 0) || continue
    consider(z)
  end
  zbest
end

# the projection onto the ℓ1 ball in the metric, by brute force over the signs
function brute_ballL1(x, γ, r)
  norm(x, 1) <= r && return copy(x)
  n = length(x)
  best, zbest = Inf, fill(NaN, n)
  for σ in Iterators.product(ntuple(_ -> -1:1, n)...)
    F = [s != 0 for s in σ]
    any(F) || continue
    ν = (sum(σ[i] * x[i] for i in 1:n if F[i]) - r) / sum(γ[F])
    z = [F[i] ? x[i] - γ[i] * ν * σ[i] : 0.0 for i in 1:n]
    all(sign(z[i]) == σ[i] for i in 1:n if F[i]) || continue
    o = sum((z .- x) .^ 2 ./ (2 .* γ))
    o < best && ((best, zbest) = (o, z))
  end
  zbest
end

@testset "sets in a diagonal metric" begin
  rng = Xoshiro(13)
  n = 4
  for trial in 1:20
    x = 1.5 .* randn(rng, n)
    γ = 0.2 .+ 3 .* rand(rng, n)
    a = randn(rng, n)
    b = randn(rng)
    # the hyperplane and the halfspace in closed form, against the KKT system
    # of the metric, Diagonal(1 ./ γ) z + νa = x ./ γ, ⟨a, z⟩ = b
    K = [Diagonal(1 ./ γ) a; a' 0]
    zh = (K \ [x ./ γ; b])[1:n]
    @test prox(IndHyperplane(a, b), x, γ) ≈ zh
    @test prox(IndHalfspace(a, b), x, γ) ≈ (dot(a, x) > b ? zh : x)
    @test prox(IndHyperplane(1, b), x, γ) ≈ prox(IndHyperplane(ones(n), b), x, γ)
    # Constrained: the multiplier search with a step per entry
    # b from a point of the box, so that the set is not empty
    for (f, λ, l, u, b) in ((NormL1(0.4), 0.4, -Inf, Inf, b), (IndBox(0, 1), 0.0, 0.0, 1.0, dot(a, rand(rng, n))),
                            (Constrained(NormL1(0.3), IndBox(-1, 2)), 0.3, -1.0, 2.0, dot(a, 3 .* rand(rng, n) .- 1)))
      c = Constrained(f, IndHyperplane(a, b))
      @test prox(c, x, γ) ≈ brute_linear(x, γ, a, b, λ, l, u) atol = 1e-9
      ch = Constrained(f, IndHalfspace(a, b))
      @test prox(ch, x, γ) ≈ brute_linear(x, γ, a, b, λ, l, u; halfspace = true) atol = 1e-9
    end
    # the simplex and the ℓ1 ball, alone and through Constrained
    zs = brute_linear(x, γ, ones(n), 1.0, 0.0, 0.0, Inf)
    @test prox(IndSimplex(1), x, γ) ≈ zs
    @test prox(Constrained(ZeroFunction(), IndSimplex(1)), x, γ) ≈ zs atol = 1e-9
    @test prox(IndSimplex(1; equality = false), x, γ) ≈ brute_linear(x, γ, ones(n), 1.0, 0.0, 0.0, Inf; halfspace = true)
    @test prox(IndBallL1(1), x, γ) ≈ brute_ballL1(x, γ, 1.0)
    @test prox(IndBallL1(10), x, γ) == x
  end
  # equal steps give the scalar step
  x = randn(rng, 50)
  a = randn(rng, 50)
  for f in (IndHyperplane(a, 0.3), IndHalfspace(a, -0.3), IndSimplex(2), IndSimplex(2; equality = false),
            IndBallL1(2), Constrained(NormL1(0.2), IndHyperplane(1, 0)), Constrained(IndBox(0, 1), IndHyperplane(1, 3)),
            Constrained(SqrNormL2(0.5), IndSimplex(1)), Constrained(NormL1(0.2), IndBox(-0.3, 0.4)))
    @test prox(f, x, fill(0.7, 50)) ≈ prox(f, x, 0.7)
    @test value_and_prox!(similar(x), f, x, fill(0.7, 50)) ≈ value_and_prox!(similar(x), f, x, 0.7)
  end
  # two algorithms for the simplex and the ℓ1 ball in the metric: the level by
  # Newton's method and the multiplier search of Constrained, at n = 1000
  x = randn(rng, 1000)
  γ = 0.1 .+ rand(rng, 1000)
  @test prox(IndSimplex(3), x, γ) ≈ prox(Constrained(ZeroFunction(), IndSimplex(3)), x, γ)
  zb = prox(Constrained(ZeroFunction(), IndSimplex(5; equality = false)), abs.(x), γ)
  @test prox(IndBallL1(5), x, γ) ≈ sign.(x) .* zb
  @test sum(prox(IndSimplex(3), x, γ)) ≈ 3
  # the warm start: a sequence of nearby points gives what a new function gives
  c = Constrained(NormL1(0.1), IndHyperplane(1, 0))
  for k in 1:5
    xk = x .+ 0.01k .* sin.(1:1000)
    @test prox(c, xk, γ) ≈ prox(Constrained(NormL1(0.1), IndHyperplane(1, 0)), xk, γ) rtol = 1e-10
  end
  # in place, Float32, and the size of the steps
  z = copy(x)
  prox!(z, IndSimplex(3), z, γ)
  @test z ≈ prox(IndSimplex(3), x, γ)
  z = copy(x)
  prox!(z, c, z, γ)
  @test z ≈ prox(c, x, γ)
  x32, γ32 = Float32.(x), Float32.(γ)
  for f in (IndSimplex(3), IndBallL1(5), IndHyperplane(1, 2), c)
    @test prox(f, x32, γ32) isa Vector{Float32}
    @test prox(f, x32, γ32) ≈ prox(f, x, γ) rtol = 1e-4
  end
  for f in (IndSimplex(3), IndBallL1(5), IndHyperplane(1, 2), c, Constrained(NormL1(0.2), IndBox(-1, 1)))
    @test_throws DimensionMismatch prox(f, x, γ[1:999])
  end
end

@testset "array steps for scalar compositions and regularization" begin
  n = 3
  x = [1.0, 2.0, -0.5]
  γ = rand(n) .+ 0.1
  c = Conjugate(ElasticNet(1.0, 0.5))
  cvec = randn(n)
  for f in (Precompose(NormL1(0.3), 2.0), Precompose(c, -1.5),
            Regularize(NormL1(0.3), 0.7), Regularize(NormL1(0.3), 0.7, cvec),
            Regularize(c, 0.3))
    p = prox(f, x, γ)
    @test p ≈ [prox_entry(f, x[i], i, γ[i]) for i in 1:n]
    @test prox(f, x, fill(0.4, n)) ≈ prox(f, x, 0.4)
    @test value_and_prox!(similar(x), f, x, fill(0.4, n)) ≈
          value_and_prox!(similar(x), f, x, 0.4)
    y = similar(x)
    v = value_and_prox!(y, f, x, γ)
    @test y ≈ p
    z = copy(x)
    @test value_and_prox!(z, f, z, γ) ≈ v
    @test z ≈ y
    z = copy(x)
    @test prox!(z, f, z, γ) ≈ p
    for op in (prox!, value_and_prox!)
      fill!(y, 17.0)
      @test_throws DimensionMismatch op(y, f, x, γ[1:2])
      @test y == fill(17.0, n)
    end
  end

  # The inner conjugate supplies its value only through the proximal oracle.
  h = Precompose(c, 2.0)
  y = zeros(1)
  v = value_and_prox!(y, Precompose(h, [1.0]), [2.0], 0.5)
  @test y ≈ [0.8]
  @test v ≈ 0.36
  z = similar(y)
  @test value_and_prox!(z, h, [2.0], 0.5) ≈ v
  @test z ≈ y

  a = [2.0, -0.5, 1.5]
  for inner in (h, Regularize(c, 0.3))
    f = Precompose(inner, a)
    y = similar(x)
    @test isfinite(value_and_prox!(y, f, x, 0.5))
    @test y ≈ prox(f, x, 0.5)
    for s in (2.0, -1.5)
      fa, fs = Precompose(inner, fill(s, n)), Precompose(inner, s)
      ya, ys = similar(x), similar(x)
      @test value_and_prox!(ya, fa, x, 0.5) ≈ value_and_prox!(ys, fs, x, 0.5)
      @test ya ≈ ys
      @test ya ≈ prox(fa, x, 0.5)
    end
  end

  for (f, message) in ((Precompose(NormL2(1.0), 2.0),
                       "array steps for a scalar scale require a separable function"),
                      (Regularize(NormL2(1.0), 0.3),
                       "array steps for Regularize require a separable function"))
    for op in (prox!, value_and_prox!)
      y = fill(17.0, n)
      @test_throws ArgumentError(message) op(y, f, x, γ)
      @test y == fill(17.0, n)
    end
  end
end

@testset "zero allocations" begin
  a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
  a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  a_cl!(L, f) = coordinate_lipschitz!(L, f)
  a_cl(f, k) = coordinate_lipschitz(f, k)
  n = 20
  x = randn(n)
  γ = rand(n) .+ 0.1
  xs = SparseIterate(sprandn(n, 0.3))
  for f in (NormL1(0.3), NormL1(0.3, rand(n)), MCP(0.3), IndBox(-0.5, 0.5), Scaled(NormL1(0.3), 2.),
            Tilt(NormL1(0.3), randn(n)), Translate(NormL1(0.3), randn(n)), Conjugate(NormL1(0.3)))
    y = similar(x)
    @test allocs(a_prox!, y, f, x, γ) == 0
    f isa Conjugate || @test allocs(a_vprox!, y, f, x, γ) == 0
  end
  for f in (IndHyperplane(randn(n), 0.3), IndHalfspace(1, -0.3), IndSimplex(1), IndSimplex(1; equality = false),
            IndBallL1(1), Constrained(NormL1(0.1), IndHyperplane(1, 0)), Constrained(IndBox(0, 1), IndHyperplane(1, 3)),
            Constrained(NormL1(0.1), IndHalfspace(randn(n), -0.5)), Constrained(NormL1(0.2), IndBox(-0.3, 0.4)))
    y = similar(x)
    @test allocs(a_prox!, y, f, x, γ) == 0
    @test allocs(a_vprox!, y, f, x, γ) == 0
  end
  ys = similar(xs)
  @test allocs(a_prox!, ys, NormL1(0.3), xs, γ) == 0
  groups = [1:5, 6:10, 11:15, 16:20]
  g = GroupNormL2(0.4, groups)
  γg = rand(4) .+ 0.1
  @test allocs(a_prox!, similar(x), g, x, γg) == 0
  xa = AtomIterate(randn(n), g)
  @test allocs(a_vprox!, similar(xa), g, xa, γg) == 0
  X = randn(30, n)
  for f in (LeastSquaresLoss(randn(30), X), LogisticLoss(Float64.(rand(30) .< 0.5), X), QuadraticFunction(X' * X),
            (1 / 30) * LeastSquaresLoss(randn(30), X))
    L = zeros(n)
    @test allocs(a_cl!, L, f) == 0
    @test allocs(a_cl, f, 3) == 0
  end
end

@testset "coordinate bounds with no observations" begin
  for T in (Float32, Float64)
    f = LogisticLoss(T[], zeros(T, 0, 3))
    @test [coordinate_lipschitz(f, k) for k in 1:3] == zeros(T, 3)
    @test coordinate_lipschitz!(ones(T, 3), f) == zeros(T, 3)
    @test_throws BoundsError coordinate_lipschitz(f, 0)
    @test_throws BoundsError coordinate_lipschitz(f, 4)
  end
end

end
