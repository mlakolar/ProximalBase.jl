# prox! and value_and_prox! into a SymmetricSparseIterate from an argument that is not
# symmetric: the proximal point over symmetric matrices at V is the proximal point at
# the symmetric part (V + V')/2, for every function.
using ProximalBase, LinearAlgebra, Random, Test
using ProximalBase: _SymmetricPart, _symmetric_argument

const _SSI = SymmetricSparseIterate{Float64}

# a group of linear indices of a p x p matrix that is closed under transposition
function _symmetric_groups(p)
  seen = falses(p, p)
  groups = Vector{Int}[]
  for c in 1:p, r in c:p
    seen[r, c] && continue
    g = r == c ? [r + (c - 1) * p] : [r + (c - 1) * p, c + (r - 1) * p]
    seen[r, c] = true
    push!(groups, g)
  end
  # merge neighbours so that the groups are not all of size one or two
  merged = Vector{Int}[]
  for k in 1:3:length(groups)
    push!(merged, sort(reduce(vcat, groups[k:min(k + 2, length(groups))])))
  end
  merged
end

function _catalog(p, Sx)
  rng = Xoshiro(11)
  sym(A) = (A + A') / 2
  c = sym(randn(rng, p, p)); b = sym(randn(rng, p, p)); a = sym(randn(rng, p, p) .+ 1)
  # a cut of the sorted magnitudes that falls between two distinct values, so that the
  # kept set is closed under transposition
  mags = sort(unique(abs.(Sx)); rev = true)
  keep = count(>=(mags[3]), abs.(Sx))
  groups = _symmetric_groups(p)
  box = IndBox(-0.4, 0.4)
  [
    "ZeroFunction" => ZeroFunction(),
    "SqrNormL2" => SqrNormL2(0.6),
    "LinearFunction" => LinearFunction(c),
    "NormL1" => NormL1(0.3),
    "NormL1 weights" => NormL1(0.3, sym(abs.(randn(rng, p, p)))),
    "NormL0" => NormL0(0.3),
    "MCP" => MCP(0.3, 2.0),
    "SCAD" => SCAD(0.3, 3.0),
    "IndNonnegative" => IndNonnegative(),
    "IndBox" => box,
    "IndBallL0" => IndBallL0(keep),
    "IndBallL1" => IndBallL1(1.5),
    "IndBallL2" => IndBallL2(0.8),
    "IndSphereL2" => IndSphereL2(0.8),
    "IndSimplex" => IndSimplex(2.0; equality = false),
    "NormL2" => NormL2(0.3),
    "CubeNormL2" => CubeNormL2(0.3),
    "NormLinf" => NormLinf(0.3),
    "SumLargest" => SumLargest(2, 0.3),
    "IndHyperplane" => IndHyperplane(a, 0.5),
    "IndHalfspace" => IndHalfspace(a, 0.5),
    "IndHyperslab" => IndHyperslab(a, -0.5, 0.5),
    "Scaled" => Scaled(NormL1(0.3), 1.7),
    "Tilt" => Tilt(NormL1(0.3), c, 0.2),
    "Translate" => Translate(NormL1(0.3), b),
    "Precompose scalar" => Precompose(NormL1(0.3), 2.0),
    "Precompose array" => Precompose(NormL1(0.3), a),
    "Regularize" => Regularize(NormL1(0.3), 0.5),
    "Conjugate" => Conjugate(NormL1(0.3)),
    "Conjugate of a conjugate" => Conjugate(Conjugate(NormL1(0.3))),
    "Conjugate of SumLargest" => Conjugate(SumLargest(2, 0.3)),
    "MoreauEnvelope" => MoreauEnvelope(NormL1(0.3), 0.5),
    "DistL2" => DistL2(box, 0.7),
    "SqrDistL2" => SqrDistL2(box, 0.7),
    "BoxConstrained" => Constrained(NormL1(0.3), box),
    "BallConstrained" => Constrained(NormL1(0.3), IndBallL2(0.6)),
    "SeparableSum single" => NormL1(0.3),
    "NuclearNorm" => NuclearNorm(0.4),
    "SingularValue" => SingularValue(NormL1(0.4)),
    "IndBallRank" => IndBallRank(2),
    "IndPSD" => IndPSD(),
    "IndPSD rank" => IndPSD(rank = 2),
    "TracePSD" => TracePSD(0.4),
    "Spectral" => Spectral(NormL1(0.4)),
    "GroupNormL2" => GroupNormL2(0.3, groups),
        "SparseGroupNormL2" => SparseGroupNormL2(0.2, 0.3, groups),
    "GroupMCP" => GroupMCP(0.3, groups; a = 2.0),
    "GroupSCAD" => GroupSCAD(0.3, groups; a = 3.7),
    "IndBallGroupL0" => IndBallGroupL0(2, groups),
  ]
end

# the reference: the proximal point at the symmetric part, over dense matrices
function _check(name, f, V, Sx, γ)
  ref = prox(f, Matrix(Sx), γ)
  # singular value functions are symmetric up to rounding
  norm(ref - ref') <= 1e-10 * (1 + norm(ref)) || return :nosymmetric
  ref = (ref + ref') / 2
  y = SymmetricSparseIterate(Float64, size(V, 1))
  prox!(y, f, V, γ)
  @test Matrix(y) ≈ ref atol = 1e-10
  y2 = SymmetricSparseIterate(Float64, size(V, 1))
  v = value_and_prox!(y2, f, V, γ)
  @test Matrix(y2) ≈ ref atol = 1e-10
  # the value of a conjugate of a conjugate is not implemented: compare when it is
  fy = try f(Matrix(y2)) catch e; e isa ArgumentError ? nothing : rethrow() end
  fy === nothing || @test isapprox(v, fy; atol = 1e-8, rtol = 1e-8) || isinf(v)
  :ok
end

@testset "prox! into a SymmetricSparseIterate reads the symmetric part" begin
  p = 4
  rng = Xoshiro(7)
  V = randn(rng, p, p)
  Sx = (V + V') / 2
  @test !issymmetric(V)
  γ = 0.7
  entries = _catalog(p, Sx)
  statuses = Dict{String, Symbol}()
  for (name, f) in entries
    f === nothing && continue
    @testset "$name" begin
      statuses[name] = _check(name, f, V, Sx, γ)
    end
  end
  # every function of the catalog gave a symmetric reference, so was compared
  @test all(==(:ok), values(statuses))
end

@testset "the contract example: the lower triangle alone gives another value" begin
  V = [0.0 1.0; 3.0 0.0]
  y = SymmetricSparseIterate(Float64, 2)
  prox!(y, NormL1(0.5), V, 1.0)
  @test y[2, 1] == 1.5 && y[1, 2] == 1.5     # the symmetric part has 2; the entry (2, 1) has 3
  prox!(y, NormL1(0.5), Symmetric(V, :L), 1.0)   # an argument symmetric by type is used as is
  @test y[2, 1] == 2.5
end

@testset "an argument symmetric by type or exactly symmetric gives today's result" begin
  p = 5
  rng = Xoshiro(2)
  S = (S0 = randn(rng, p, p); (S0 + S0') / 2)
  f = NormL1(0.3)
  y1 = SymmetricSparseIterate(Float64, p); prox!(y1, f, S, 0.5)
  y2 = SymmetricSparseIterate(Float64, p); prox!(y2, f, Symmetric(S, :L), 0.5)
  y3 = SymmetricSparseIterate(Float64, p); prox!(y3, f, copy(y1), 0.5)
  @test Matrix(y1) == Matrix(y2) == prox(f, S, 0.5)
  @test _symmetric_argument(y1, S) === S                    # exactly symmetric: used as it is
  V = copy(S); V[1, 2] += 1e-8                               # not symmetric
  @test _symmetric_argument(y1, V) isa _SymmetricPart
  @test _symmetric_argument(y1, Symmetric(S)) isa Symmetric
  @test _symmetric_argument(y1, y2) === y2
  @test _symmetric_argument(zeros(p, p), S) === S           # a dense output is untouched
  # a size that does not match the output is left to the shape checks
  @test_throws DimensionMismatch prox!(y1, f, randn(rng, p + 1, p + 1), 0.5)
end

@testset "an argument symmetric by type, or exactly symmetric, allocates nothing" begin
  p = 20
  rng = Xoshiro(1)
  S = (V = randn(rng, p, p); (V + V') / 2)
  y = SymmetricSparseIterate(Float64, p)
  xs = SymmetricSparseIterate(Float64, p); prox!(xs, NormL1(0.1), S, 0.1)
  f = NormL1(0.3)
  allocated(y, f, x) = @allocated prox!(y, f, x, 0.7)
  for x in (xs, S, Symmetric(S, :L))
    allocated(y, f, x)
    @test allocated(y, f, x) == 0
  end
end

@testset "the symmetric part: exact, finite, and NaN for opposite infinities" begin
  A = [1.0 2.0 floatmax(); 4.0 5.0 6.0; floatmax() 8.0 9.0]
  S = _SymmetricPart(A)
  @test issymmetric(S)
  @test S[1, 3] == floatmax() && S[3, 1] == floatmax()       # no overflow in the average
  @test S[1, 2] == 3.0 && S[2, 1] == 3.0
  B = [0.0 Inf; -Inf 0.0]
  @test isnan(_SymmetricPart(B)[1, 2]) && isnan(_SymmetricPart(B)[2, 1])
  @test _SymmetricPart([1 2; 3 4]) == [1.0 2.5; 2.5 4.0]     # integers give a floating point part
end

@testset "a block sum checks every block before it writes any" begin
  p = 4
  rng = Xoshiro(5)
  V = randn(rng, p, p); Sx = (V + V') / 2
  f = SeparableSum(SqrNormL2(0.6), NormL1(0.3))
  y1 = SymmetricSparseIterate(Float64, p); y2 = SymmetricSparseIterate(Float64, p)
  prox!((y1, y2), f, (V, V), 0.7)
  @test Matrix(y1) ≈ prox(SqrNormL2(0.6), Matrix(Sx), 0.7)
  @test Matrix(y2) ≈ prox(NormL1(0.3), Matrix(Sx), 0.7)
  # weights that are not symmetric still throw before the first block is written
  W = abs.(randn(rng, p, p))
  z1 = SymmetricSparseIterate(Float64, p); z2 = SymmetricSparseIterate(Float64, p)
  z1[1, 1] = 5.0
  g = SeparableSum(SqrNormL2(0.6), NormL1(0.3, W))
  @test_throws ArgumentError prox!((z1, z2), g, (V, V), 0.7)
  @test z1[1, 1] == 5.0 && z1[2, 1] == 0.0
end

@testset "spectral functions use the symmetric part for a dense output too" begin
  V = [1.0 2.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 3.0]
  S = (V + V') / 2
  for f in (IndPSD(), TracePSD(0.5), Spectral(NormL1(0.2)))
    y = similar(V)
    prox!(y, f, V, 1.0)
    @test y ≈ prox(f, S, 1.0)
    @test y ≈ prox(f, Symmetric(S), 1.0)
  end
end

@testset "gradient! and hessian_vec! into a SymmetricSparseIterate write the symmetric part" begin
  p = 4
  rng = Xoshiro(5)
  A = randn(rng, p, p); Σ = A * A' + I
  C = (B = randn(rng, p, p); (B + B') / 2)
  f = KronQuadratic(Σ, Σ, C)
  Θ = randn(rng, p, p)
  G = SymmetricSparseIterate(Float64, p)
  gradient!(G, f, Θ)
  ref = gradient(f, Θ)
  @test Matrix(G) ≈ (ref + ref') / 2
  G2 = SymmetricSparseIterate(Float64, p)
  v = value_and_gradient!(G2, f, Θ)
  @test Matrix(G2) ≈ (ref + ref') / 2
  @test v ≈ f(Θ)
  V = (B = randn(rng, p, p); (B + B') / 2)
  Hv = SymmetricSparseIterate(Float64, p)
  hessian_vec!(Hv, f, Θ, V)
  H = similar(Θ); hessian_vec!(H, f, Θ, V)
  @test Matrix(Hv) ≈ (H + H') / 2
end

@testset "every method that accepts a symmetric output is exercised" begin
  p = 4
  V = randn(Xoshiro(7), p, p)
  entries = [f for (_, f) in _catalog(p, (V + V') / 2) if f !== nothing]
  catalog_methods = Set{Method}()
  for f in entries, op in (prox!, value_and_prox!)
    push!(catalog_methods, which(op, Tuple{_SSI, typeof(f), Matrix{Float64}, Float64}))
  end
  accepting = Method[]
  for op in (prox!, value_and_prox!), m in methods(op)
    m.module === ProximalBase || continue      # not the methods that other test files define
    sig = Base.unwrap_unionall(m.sig).parameters
    length(sig) == 5 || continue
    Base.typeintersect(sig[2], _SSI) !== Union{} &&
      Base.typeintersect(sig[4], Matrix{Float64}) !== Union{} &&
      Base.typeintersect(sig[5], Float64) !== Union{} && push!(accepting, m)
  end
  missing_ = [m for m in accepting if m ∉ catalog_methods]
  # functions whose proximal point is not a symmetric matrix for a symmetric argument,
  # or that do not take a matrix argument; they keep their own checks
  skipped = ("IndSOC", "IndRotatedSOC", "_ExpCone", "IndExpDual", "IndExpPrimal", "IndGraph",
             "IndPolyhedron", "LeastSquaresLoss", "IndToeplitz", "IndStiefel", "LogDetLoss",
             "LinearConstrained", "LinearImage", "SliceSum", "FiberSum", "PointwiseMinimum",
             "SphereConstrained", "IndIntersection", "SeparableSum",
             "SingularValue")   # Conjugate{<:SingularValue}: its sum is not exactly symmetric, even for a
                                # symmetric argument (a separate limitation, not the argument's symmetric part)
  unexplained = [m for m in missing_ if !any(s -> occursin(s, string(Base.unwrap_unionall(m.sig).parameters[3])), skipped)]
  @test isempty(unexplained)
  isempty(unexplained) || foreach(m -> println("not covered: ", m), unexplained)
end
