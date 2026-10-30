module SymmetricParametersMergeTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils: allocs

const PB = ProximalBase
state(x) = map(n -> deepcopy(getfield(x, n)), fieldnames(typeof(x)))

@testset "Master type and shape errors precede parameter symmetry" begin
  W = ones(3, 3); W[2, 1] = 2.
  x = fill!(SymmetricSparseIterate(3), .5)
  for f in (NormL1(.1, W), Scaled(NormL1(.1, W), 2.), Tilt(NormL1(.1, W), zeros(3, 3)))
    before = state(x)
    e = try prox!(x, f, complex.(Matrix(x)), 1.); nothing catch e; e end
    @test e isa ArgumentError
    @test sprint(showerror, e) == "ArgumentError: $(nameof(typeof(f))) requires real entries"
    @test isequal(state(x), before)
  end
  @test_throws DimensionMismatch gradient!(copy(x), LinearFunction(vec(W)), x)
  @test_throws DimensionMismatch hessian_vec!(zeros(2, 2), LinearFunction(W), x, zeros(3, 3))
end

@testset "Regularize group steps stay in the group domain" begin
  groups = [[1, 2], [3, 4]]
  f = GroupNormL2(.2, groups, ones(2)); c = [.1 .3; .3 .2]
  r = Regularize(f, .4, c); steps = [.2, .5]; X = [.5 .4; .4 .8]
  expected = similar(X)
  for k in eachindex(groups)
    G = groups[k]; γ = steps[k]
    z = (X[G] + γ * r.ρ * c[G]) / (1 + γ * r.ρ)
    threshold = γ * f.λ0 / (1 + γ * r.ρ)
    expected[G] = max(1 - threshold / norm(z), 0) * z
  end
  y = similar(X)
  @test prox!(y, r, X, steps) ≈ expected
  reference = .2sum(norm(expected[G]) for G in groups) + .4sum(abs2, expected - c) / 2
  @test value_and_prox!(y, r, X, steps) ≈ reference
  for h in (r, Scaled(r, 2.)), op in (prox!, value_and_prox!)
    x = SymmetricSparseIterate(X); before = state(x)
    e = try op(x, h, x, steps); nothing catch e; e end
    @test e isa ArgumentError
    @test sprint(showerror, e) == "ArgumentError: per-group steps for Regularize do not support SymmetricSparseIterate outputs"
    @test isequal(state(x), before)
  end
end

@testset "New spectral non-finite values survive unchecked forwarding" begin
  X = fill(Inf, 2, 2)
  for f in (IndPSD(), IndBallRank(1), IndStiefel(), Conjugate(SingularValue(NormL1(.2))))
    y = zeros(2, 2)
    @test value_and_prox!(y, Scaled(f, 2.), X, 1.) == Inf
    @test all(isnan, y)
  end
end


@testset "Mixed-precision quadratic forwarding" begin
  A = [2. .5; .5 1.]; x = [.4, .8]
  f = QuadraticFunction(A, [.1, -.2])
  for h in (f, Scaled(f, 2.), Tilt(f, zeros(2)), Sum(f, f), Precompose(f, 2.))
    scale = h isa Sum || h isa Scaled ? 2. : 1.
    input = h isa Precompose ? 2x : x
    reference = scale * (dot(input, A * input) / 2 + dot(f.b, input))
    expected = scale * (A * input + f.b) * (h isa Precompose ? 2. : 1.)
    g = zeros(Float32, 2)
    @test value_and_gradient!(g, h, x) ≈ reference
    @test g ≈ expected rtol = 4eps(Float32)
    @test gradient!(g, h, x) ≈ expected rtol = 4eps(Float32)
  end
end


struct CountedValueArray{T} <: AbstractMatrix{T}
  data::Matrix{T}
  reads::Base.RefValue{Int}
end
CountedValueArray(A::Matrix) = CountedValueArray(A, Ref(0))
Base.size(a::CountedValueArray) = size(a.data)
Base.IndexStyle(::Type{<:CountedValueArray}) = IndexLinear()
Base.getindex(a::CountedValueArray, i::Int) = (a.reads[] += 1; a.data[i])

@testset "Value domains follow sparse parameter reads" begin
  p = 1000
  x = SymmetricSparseIterate(p); x[1, 1] = 1.; x[2, 1] = .5
  cases = ((a -> IndHyperplane(a, 0.), Inf, 4, :a, :IndHyperplane),
           (a -> IndHalfspace(a, 10.), 0., 4, :a, :IndHalfspace),
           (a -> IndHyperslab(a, -10., 10.), 0., 6, :a, :IndHyperslab),
           (a -> LinearFunction(a), 2., 4, :c, :LinearFunction),
           (a -> Tilt(ZeroFunction(), a), 2., 4, :c, :Tilt),
           (a -> Tilt(NormL1(.1, a), ones(p, p)), 2.2, 5, :w, :NormL1),
           (a -> Scaled(IndHalfspace(a, 10.), 2.), 0., 4, :a, :IndHalfspace),
           (a -> Constrained(NormL1(.1, a), IndSphereL2(sqrt(1.5))), .2, 5, :w, :NormL1),
           (a -> Scaled(Tilt(ZeroFunction(), a), 2.), 4., 4, :c, :Tilt),
           (a -> Sum(LinearFunction(a), ZeroFunction()), 2., 4, :c, :LinearFunction),
           (a -> PointwiseMinimum(LinearFunction(a), LinearFunction(ones(p, p))), 2., 4, :c, :LinearFunction),
           (a -> Constrained(LinearFunction(a), IndBox(-2., 2.)), 2., 4, :c, :LinearFunction),
           (a -> Constrained(NormL1(.1, a), IndBallL2(2.)), .2, 5, :w, :NormL1),
           (a -> Constrained(NormL1(.1), IndHalfspace(a, 10.)), .2, 4, :a, :IndHalfspace),
           (a -> Regularize(LinearFunction(a), .2), 2.15, 4, :c, :LinearFunction),
           (a -> IndIntersection(IndHalfspace(a, 10.), IndBox(-2., 2.)), 0., 4, :a, :IndHalfspace))
  for (make, expected, bound, name, owner) in cases
    a = CountedValueArray(ones(p, p)); a.data[3, 2] = 2.
    f = make(a); a.reads[] = 0
    @test f(x) ≈ expected
    @test 0 < a.reads[] <= bound
    a.data[2, 1] = 2.
    e = try f(x); nothing catch e; e end
    @test e isa ArgumentError
    @test sprint(showerror, e) == "ArgumentError: the array $name of $owner must be symmetric on a SymmetricSparseIterate, but entry (2, 1) is 2.0 and entry (1, 2) is 1.0; use Symmetric(A, :L) to keep the lower triangle of A, or (A + A')/2 for its symmetric part"
    a.data[2, 1] = 1.
    f(x)
    @test allocs(f, x) == 0
  end
  # A slab can evaluate its normal twice: two validation reads plus two pairs
  # of reduction reads give six. The ball penalty uses five: two validation reads
  # plus three entrywise value reads; the one-pass dot oracles use four.
  for c in (1., 1e200), s in (-.5, .5)
    a = CountedValueArray(fill(c, p, p)); a.data[3, 2] = 2c
    y = copy(x); y[2, 1] = s
    f = IndHyperplane(a, 0.); a.reads[] = 0
    @test f(y) == (s < 0 ? 0. : Inf)
    @test a.reads[] <= 2nnz(y)
  end
  for make in (a -> SqrNormL2(1., a), a -> Regularize(ZeroFunction(), .2, a),
               a -> Translate(NormL1(), a), a -> Precompose(NormL1(.1, a), 2.),
               a -> Tilt(SqrNormL2(1., zeros(p, p)), a))
    a = ones(p, p); a[3, 2] = 2.
    @test_throws ArgumentError make(a)(x)
  end
  # A full-reading inner part makes the union full, including a tilt's array.
  a = ones(p, p); a[3, 2] = 2.
  @test_throws ArgumentError Sum(LinearFunction(a), SqrNormL2(1., zeros(p, p)))(x)
  for make in (a -> LinearFunction(a), a -> Tilt(ZeroFunction(), a), a -> IndHyperplane(a, 0.))
    a = CountedValueArray(ones(p, p)); f = make(a)
    f(x)
    @test allocs(f, x) == 0
  end
  a = CountedValueArray(ones(p, p)); a.data[3, 2] = 2.
  f = SeparableSum(LinearFunction(a), NormL1(.1))
  a.reads[] = 0
  @test f((x, x)) ≈ 2.2
  @test a.reads[] <= 4
end

@testset "Value proof does not leak into mixed writes" begin
  X = [.5 .2; .2 .8]; x = SymmetricSparseIterate(X)
  c = [1. 2.; 3. 4.]; f = LinearFunction(c); g = zeros(2, 2)
  @test value_and_gradient!(g, f, x) ≈ sum(c .* X)
  @test g == c
  h = Tilt(LinearFunction(c), c)
  @test value_and_gradient!(g, h, x) ≈ 2sum(c .* X)
  @test g == 2c
  f = IndHalfspace(c, 0.)
  @test prox!(g, f, x, 1.) ≈ X - sum(c .* X) / sum(abs2, c) * c
end

end
