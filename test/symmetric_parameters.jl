module SymmetricParametersTest
using Test, ProximalBase, LinearAlgebra, Random
const PB = ProximalBase

# Symmetry of per-entry function parameters is a model precondition. The ordinary
# oracles assume it and never compare mirrored parameter entries; the explicit
# checker validates it on demand. Everything else an oracle verifies (shapes,
# element types, aliasing, supported operations) is unchanged.

# A user extension with explicit parameter metadata.
struct WeightedSquare{A} <: SeparableFunction
  w::A
end
PB.value_entry(f::WeightedSquare, v, i) = f.w[i] * v^2 / 2
PB.prox_block(f::WeightedSquare, v::Real, i::Integer, γ::Real) = v / (1 + γ * f.w[i])
PB.iszeropreserving(::WeightedSquare) = true
PB.isconvex(::Type{<:WeightedSquare}) = true
PB.entry_parameters(f::WeightedSquare) = (:w => f.w,)
PB.check_parameters(f::WeightedSquare, x) =
  length(f.w) == prod(size(x)) ? nothing : throw(DimensionMismatch("weights"))
# a separable function that does not declare its per-entry arrays, and one that declares them wrongly
struct Undeclared{A} <: SeparableFunction
  w::A
end
struct Malformed{A} <: SeparableFunction
  w::A
end
PB.entry_parameters(f::Malformed) = (f.w,)
struct NotAnArray{A} <: SeparableFunction
  w::A
end
PB.entry_parameters(f::NotAnArray) = (:w => "weights",)

# A parameter matrix that counts the reads of each of its entries.
struct ReadCounter{T} <: AbstractMatrix{T}
  A::Matrix{T}
  n::Matrix{Int}
end
ReadCounter(A::AbstractMatrix) = ReadCounter(Matrix(A), zeros(Int, size(A)))
Base.size(a::ReadCounter) = size(a.A)
Base.IndexStyle(::Type{<:ReadCounter}) = IndexLinear()
Base.getindex(a::ReadCounter, i::Int) = (a.n[i] += 1; a.A[i])
reads(a::ReadCounter) = sum(a.n)
reset!(a::ReadCounter) = (fill!(a.n, 0); a)

symm(A) = (A + A') / 2
state(x::SymmetricSparseIterate) = (Matrix(x), collect(activecoordinates(x)), nactive(x))
state(x::AbstractArray) = copy(x)
same(a::Tuple, b::Tuple) = length(a) == length(b) && all(same(u, v) for (u, v) in zip(a, b))
same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(same(u, v) for (u, v) in zip(a, b))
same(a, b) = typeof(a) == typeof(b) && isequal(a, b)    # isequal separates the bits of 0.0 and -0.0, and equates NaN
samestate(a, b) = state(a) == state(b)

function sparse_iterate(T, p)
  x = SymmetricSparseIterate(T, p)
  x[2, 1] = T(1.5); x[p, p] = T(-2); x[3, 2] = T(0.25)
  activate!(x, tril_index(p, 4, 1))      # an active zero
  x
end
function full_iterate(T, p, rng)
  x = SymmetricSparseIterate(T, p)
  for c in 1:p, r in c:p
    x[r, c] = randn(rng, T)
  end
  x
end
function output_iterate(T, p)
  y = SymmetricSparseIterate(T, p)
  y[3, 1] = T(0.75); y[2, 2] = T(-1)
  y
end

# every per-entry array exactly symmetric
function fixtures(T, p, rng)
  W = T.(symm(rand(rng, p, p) .+ 0.1)); W[1, 1] = 0; W[2, 1] = W[1, 2] = 0
  W, T.(symm(randn(rng, p, p))), T.(symm(randn(rng, p, p)))
end
function models(T, W, Y, C)
  l1, sq, lin, ws = NormL1(T(0.3), W), SqrNormL2(T(0.7), Y), LinearFunction(C), WeightedSquare(W)
  ["NormL1" => l1, "SqrNormL2(y)" => sq, "SqrNormL2" => SqrNormL2(T(0.7)), "LinearFunction" => lin,
   "WeightedSquare" => ws, "Scaled(NormL1)" => T(2) * l1, "Scaled(SqrNormL2)" => T(0.5) * sq,
   "Scaled(LinearFunction)" => T(3) * lin, "Scaled(WeightedSquare)" => T(2) * ws,
   "nested Scaled(NormL1)" => T(2) * (T(0.5) * l1), "nested Scaled(LinearFunction)" => T(2) * (T(1.5) * lin)]
end
# the same models with their arrays read through the lower triangle
function controls(T, W, Y, C)
  l1, sq, lin, ws = NormL1(T(0.3), Symmetric(W, :L)), SqrNormL2(T(0.7), Symmetric(Y, :L)),
                    LinearFunction(Symmetric(C, :L)), WeightedSquare(Symmetric(W, :L))
  ["NormL1" => l1, "SqrNormL2(y)" => sq, "LinearFunction" => lin, "WeightedSquare" => ws,
   "Scaled(NormL1)" => T(2) * l1, "Scaled(SqrNormL2)" => T(0.5) * sq, "Scaled(LinearFunction)" => T(3) * lin]
end
# the screen of a coordinate exists for the built-in functions only
haskkt(f) = hasmethod(kkt_violation, Tuple{typeof(f), AbstractArray, Int})
haskkt(f::Scaled) = haskkt(f.f)

for T in (Float32, Float64)
  @testset "$T valid symmetric models agree with dense and structurally symmetric references" begin
    p = 5; rng = MersenneTwister(374)
    γ = T(0.4)
    for (name, f) in models(T, fixtures(T, p, rng)...), x in (sparse_iterate(T, p), full_iterate(T, p, rng))
      @testset "$name, $(nactive(x)) active" begin
        D = Matrix(x)
        @test isapprox(f(x), f(D); rtol = 20eps(T))
        # prox: the dense computation is independent of the iterate routes
        ref = prox(f, D, γ)
        y = output_iterate(T, p); prox!(y, f, x, γ)
        @test Matrix(y) == ref
        z = zeros(T, p, p); prox!(z, f, x, γ)
        @test z == ref
        y2 = output_iterate(T, p); v2 = value_and_prox!(y2, f, x, γ)
        @test Matrix(y2) == ref
        @test isapprox(v2, f(ref); rtol = 20eps(T))
        yi = copy(x); prox!(yi, f, yi, γ)
        @test Matrix(yi) == ref
        @test Matrix(prox(f, x, γ)) == ref
        # an argument that is not symmetric is read through its symmetric part
        X = randn(rng, T, p, p)
        y3 = output_iterate(T, p); prox!(y3, f, X, γ)
        @test Matrix(y3) == prox(f, T.(symm(X)), γ)
        if issmooth(f)
          g = gradient(f, x)
          @test Matrix(g) == gradient(f, D)
          gs = output_iterate(T, p); gradient!(gs, f, x)
          @test Matrix(gs) == gradient(f, D)
          gv = output_iterate(T, p); vg = value_and_gradient!(gv, f, x)
          @test Matrix(gv) == gradient(f, D) && isapprox(vg, f(D); rtol = 20eps(T))
          gd = zeros(T, p, p); gradient!(gd, f, x)
          @test gd == gradient(f, D)
          # a gradient that is computed at a non-symmetric point is symmetrized
          g4 = output_iterate(T, p); gradient!(g4, f, X)
          @test Matrix(g4) == T.(symm(gradient(f, X)))
        end
        # the screens of a coordinate at zero, from the symmetric part of a full gradient
        G = randn(rng, T, p, p); Gs = T.(symm(G))
        if haskkt(f)
          for k in 1:ncoordinates(x)
            r, c = tril_cartesian(p, k)
            @test kkt_violation(f, G, x, k) == kkt_violation(f, Gs, (c - 1) * p + r)
            @test kkt_violation(f, vec(G), x, k) == kkt_violation(f, Gs, (c - 1) * p + r)
          end
        end
      end
    end
  end

  @testset "$T a raw symmetric parameter array and its lower-triangle control give identical results" begin
    p = 5
    W, Y, C = fixtures(T, p, MersenneTwister(1))
    raw, ctl = models(T, W, Y, C), controls(T, W, Y, C)
    γ = T(0.4)
    for (name, c) in ctl
      f = Dict(raw)[name]
      for x in (sparse_iterate(T, p), full_iterate(T, p, MersenneTwister(2)))
        @test same(f(x), c(x))
        a, b = output_iterate(T, p), output_iterate(T, p)
        prox!(a, f, x, γ); prox!(b, c, x, γ)
        @test samestate(a, b)
        a, b = zeros(T, p, p), zeros(T, p, p)
        va, vb = value_and_prox!(a, f, x, γ), value_and_prox!(b, c, x, γ)
        @test same(a, b) && isapprox(va, vb; rtol = 20eps(T))   # the dense reduction of the value may differ in order
        if issmooth(f)
          a, b = output_iterate(T, p), output_iterate(T, p)
          va, vb = value_and_gradient!(a, f, x), value_and_gradient!(b, c, x)
          @test samestate(a, b) && same(va, vb)
        end
        G = reshape(T.(1:p^2), p, p) .- T(7)
        haskkt(f) && for k in 1:ncoordinates(x)
          @test same(kkt_violation(f, G, x, k), kkt_violation(c, G, x, k))
        end
      end
    end
  end
end

@testset "the explicit checker validates every declared mirrored pair" begin
  p = 4
  check = PB.check_symmetric_parameters
  x = SymmetricSparseIterate(Float64, p); x[2, 1] = 1.0                # (4, 3) is not stored
  W = symm(reshape(1.0:p^2, p, p) .+ 1)
  bad = copy(W); bad[4, 3] += 1                                        # an unstored pair
  badstored = copy(W); badstored[2, 1] += 1                            # a stored pair
  for (label, mk) in ("NormL1" => W -> NormL1(1.0, W), "SqrNormL2" => W -> SqrNormL2(1.0, W),
                      "LinearFunction" => W -> LinearFunction(W), "WeightedSquare" => WeightedSquare,
                      "Scaled" => W -> 2.0 * NormL1(1.0, W), "nested Scaled" => W -> 2.0 * (3.0 * LinearFunction(W)))
    @test check(mk(W), x) === true
    @test_throws ArgumentError check(mk(bad), x)
    @test_throws ArgumentError check(mk(badstored), x)
    @test check(mk(Symmetric(bad, :L)), x) === true                    # symmetric by construction
    @test check(mk(Symmetric(bad, :U)), x) === true
  end
  full = full_iterate(Float64, p, MersenneTwister(3))
  @test_throws ArgumentError check(NormL1(1.0, bad), full)
  @test check(NormL1(1.0, W), full) === true
  # the checker neither averages nor repairs the array it rejects
  keep = copy(bad)
  @test_throws ArgumentError check(NormL1(1.0, bad), x)
  @test bad == keep
  # equality is exact: paired NaNs and signed zeros match, an off-by-one-ulp pair does not
  Wn = copy(W); Wn[1, 2] = Wn[2, 1] = NaN
  @test check(WeightedSquare(Wn), x) === true
  Wz = copy(W); Wz[1, 2] = 0.0; Wz[2, 1] = -0.0
  @test check(NormL1(1.0, Wz), x) === true
  Wu = copy(W); Wu[3, 1] = nextfloat(Wu[1, 3])
  @test_throws ArgumentError check(NormL1(1.0, Wu), x)
  Wi = copy(W); Wi[3, 1] = NaN
  @test_throws ArgumentError check(WeightedSquare(Wi), x)
  # scalars, missing arrays and non-iterates have nothing to compare
  @test check(NormL1(2.0), x) === true
  @test check(SqrNormL2(2.0), x) === true
  @test check(ZeroFunction(), x) === true
  @test check(NormL1(1.0, bad), [1.0 2; 3 4]) === true
  @test check(NormL1(1.0, bad), SparseIterate(Float64, 4)) === true
  # the parameter metadata of a custom function
  @test check(WeightedSquare(1.0), x) === true
  @test_throws ArgumentError check(Undeclared(W), x)
  @test_throws ArgumentError check(Malformed(W), x)
  @test_throws ArgumentError check(NotAnArray(W), x)
  @test_throws ArgumentError check(2.0 * Undeclared(W), x)
  @test check(Undeclared(W), [1.0 2; 3 4]) === true                    # no iterate, no comparison
  # a parameter of the wrong extent is a dimension error of the checker too
  @test_throws DimensionMismatch check(NormL1(1.0, ones(p + 1, p)), x)
  @test_throws DimensionMismatch check(WeightedSquare(ones(p + 1, p)), x)
end

@testset "structural validation of the ordinary oracles is unchanged" begin
  for T in (Float32, Float64)
    p = 4; γ = T(0.4)
    x = sparse_iterate(T, p)
    badlen = [NormL1(T(0.3), ones(T, p + 1, p)), NormL1(T(0.3), ones(T, p)), SqrNormL2(T(0.7), ones(T, p + 1, p)),
              LinearFunction(ones(T, p + 1, p)), WeightedSquare(ones(T, p + 1, p)),
              T(2) * NormL1(T(0.3), ones(T, p + 1, p)), T(2) * (T(3) * LinearFunction(ones(T, p + 1, p)))]
    for f in badlen
      @test_throws DimensionMismatch f(x)
      haskkt(f) && @test_throws DimensionMismatch kkt_violation(f, ones(T, p, p), x, 1)
      for out in (output_iterate(T, p), zeros(T, p, p))
        before = state(out)
        @test_throws DimensionMismatch prox!(out, f, x, γ)
        @test_throws DimensionMismatch value_and_prox!(out, f, x, γ)
        @test_throws DimensionMismatch prox!(out, f, zeros(T, p, p), γ)
        issmooth(f) && @test_throws DimensionMismatch gradient!(out, f, x)
        issmooth(f) && @test_throws DimensionMismatch value_and_gradient!(out, f, x)
        @test state(out) == before
      end
    end
    # a parameter vector has the extent p², but a write of it needs the axes of the matrix
    for f in (LinearFunction(ones(T, p * p)), SqrNormL2(T(0.7), ones(T, p * p)))
      out = output_iterate(T, p); before = state(out)
      @test_throws DimensionMismatch prox!(out, f, x, γ)
      @test_throws DimensionMismatch gradient!(out, f, x)
      @test state(out) == before
    end
    f = NormL1(T(0.3), ones(T, p, p))
    out = output_iterate(T, p); before = state(out)
    # an array step into a symmetric iterate, and a step of the wrong extent
    @test_throws ArgumentError prox!(out, f, x, fill(one(T), p, p))
    @test_throws ArgumentError value_and_prox!(out, f, x, fill(one(T), p, p))
    @test_throws ArgumentError prox!(out, ZeroFunction(), x, fill(one(T), p, p))
    @test_throws ArgumentError prox(f, x, fill(one(T), p, p))
    @test state(out) == before
    # sizes and element types
    @test_throws DimensionMismatch prox!(SymmetricSparseIterate(T, p + 1), f, x, γ)
    @test_throws DimensionMismatch prox!(zeros(T, p + 1, p + 1), f, x, γ)
    @test_throws DimensionMismatch prox!(zeros(T, p * p), f, x, γ)
    for h in (f, LinearFunction(ones(T, p, p)), SqrNormL2(T(0.7), ones(T, p, p)))
      @test_throws ArgumentError prox!(out, h, ComplexF64.(ones(p, p)), γ)
      @test_throws ArgumentError prox!(zeros(Complex{T}, p, p), h, x, γ)
      @test_throws ArgumentError h(ComplexF64.(ones(p, p)))
      @test state(out) == before
    end
    # aliased gradient outputs
    for h in (LinearFunction(ones(T, p, p)), SqrNormL2(T(0.7), ones(T, p, p)))
      xs = copy(x); snap = state(xs)
      @test_throws ArgumentError gradient!(xs, h, xs)
      @test_throws ArgumentError gradient!(xs, h, xs)
      @test state(xs) == snap
      @test_throws ArgumentError value_and_gradient!(xs, h, xs)
      @test state(xs) == snap
      @test_throws DimensionMismatch gradient!(zeros(T, p + 1, p + 1), h, x)
      @test_throws DimensionMismatch gradient!(SymmetricSparseIterate(T, p + 1), h, x)
    end
    # coordinates and gradients of the screen
    for k in (0, -1, ncoordinates(x) + 1)
      @test_throws BoundsError kkt_violation(f, ones(T, p, p), x, k)
    end
    @test_throws DimensionMismatch kkt_violation(f, ones(T, p + 1, p), x, 1)
    @test_throws DimensionMismatch kkt_violation(f, ones(T, p), x, 1)
    @test_throws ArgumentError kkt_violation(f, ones(ComplexF64, p, p), x, 1)
    # opposing infinities have no symmetric part; two equal infinities do
    G = ones(T, p, p); G[2, 1] = T(Inf); G[1, 2] = T(-Inf)
    k = tril_index(p, 2, 1)
    @test_throws ArgumentError kkt_violation(f, G, x, k)
    G[1, 2] = T(Inf)
    @test kkt_violation(f, G, x, k) == T(Inf)
    # matrix imports validate the iterate's own data
    @test_throws ArgumentError SymmetricSparseIterate(T[1 2; 3 1])
    y = SymmetricSparseIterate(T, 2); before = state(y)
    @test_throws ArgumentError copyto!(y, T[1 2; 3 1])
    @test state(y) == before
    @test Matrix(copyto!(y, T[1 2; 2 1])) == T[1 2; 2 1]
    @test_throws ArgumentError copyto!(y, matrixview(y))
    # constructor and domain errors
    @test_throws ArgumentError NormL1(T(0.3), -ones(T, 2, 2))
    @test_throws ArgumentError NormL1(T(0.3), ones(Complex{T}, 2, 2))
    @test_throws ArgumentError NormL1(-T(0.3))
  end
end

@testset "a custom extension is validated explicitly" begin
  for T in (Float32, Float64)
    f = WeightedSquare(T[1 2; 2 3]); s = SymmetricSparseIterate(T[1 2; 2 0]); snapshot = state(s)
    @test PB.check_symmetric_parameters(f, s) === true
    @test f(s) ≈ sum(f.w .* Matrix(s) .^ 2) / 2
    @test Matrix(prox(f, s, T(0.5))) ≈ Matrix(s) ./ (1 .+ T(0.5) .* f.w)
    bad = WeightedSquare(T[1 2; 3 1])
    @test_throws ArgumentError PB.check_symmetric_parameters(bad, s)
    @test state(s) == snapshot
  end
end
end
