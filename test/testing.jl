module TestingTest

# The public checks of ProximalBase.Testing pass on correct functions and fail
# on broken ones, so that none passes vacuously.

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

const PB = ProximalBase
const PT = ProximalBase.Testing

Random.seed!(1)

# a test set that records its results without printing them
struct Collect <: Test.AbstractTestSet
  description::String
  results::Vector{Any}
end
Collect(description; kwargs...) = Collect(description, Any[])
Test.record(ts::Collect, r) = (push!(ts.results, r); r)
Test.finish(ts::Collect) = ts

# the number of failed or erroring tests of a check
function failures(check)
  ts = @testset Collect "probe" begin
    check()
  end
  count(r -> r isa Test.Fail || r isa Test.Error, ts.results)
end
passes(check) = failures(check) == 0
# the numbers of failed and of erroring tests of a check: a broken function must fail an assertion, not throw
function outcomes(check)
  ts = @testset Collect "probe" begin
    check()
  end
  (count(r -> r isa Test.Fail, ts.results), count(r -> r isa Test.Error, ts.results))
end

# broken functions, each wrong in one way
struct IdentityProx <: SeparableFunction               # λ|x| with the identity as its prox
  λ::Float64
end
PB.value_entry(f::IdentityProx, v, i) = f.λ * abs(v)
PB.prox_block(::IdentityProx, v, i, γ) = float(v)
PB.isconvex(::Type{IdentityProx}) = true

struct WrongValue <: AbstractFunction                  # value_and_prox! reports twice the value
  λ::Float64
end
PB.isproximable(::Type{WrongValue}) = true
(f::WrongValue)(x) = f.λ * norm(x, 1)
PB.prox!(y::AbstractArray, f::WrongValue, x::AbstractArray, γ::Real) = (y .= shrink.(x, γ * f.λ))
PB.value_and_prox!(y::AbstractArray, f::WrongValue, x::AbstractArray, γ::Real) = (prox!(y, f, x, γ); 2 * f(y))

struct InPlaceValue <: AbstractFunction                # value_and_prox! reads x after writing y
  λ::Float64
end
PB.isproximable(::Type{InPlaceValue}) = true
(f::InPlaceValue)(x) = f.λ * norm(x, 1)
PB.prox!(y::AbstractArray, f::InPlaceValue, x::AbstractArray, γ::Real) = (y .= shrink.(x, γ * f.λ))
PB.value_and_prox!(y::AbstractArray, f::InPlaceValue, x::AbstractArray, γ::Real) =
  (prox!(y, f, x, γ); f.λ * sum(v -> abs(shrink(v, γ * f.λ)), x))

struct WrongGradient <: AbstractFunction               # ‖x‖²/2 with the gradient 2x
end
PB.issmooth(::Type{WrongGradient}) = true
(::WrongGradient)(x) = sum(abs2, x) / 2
PB.value_and_gradient!(g::AbstractArray, f::WrongGradient, x::AbstractArray) = (g .= 2 .* x; f(x))

struct BiasedGradient{F<:AbstractFunction} <: AbstractFunction
  f::F
end
PB.issmooth(::Type{<:BiasedGradient}) = true
(f::BiasedGradient)(x) = f.f(x)
function PB.value_and_gradient!(g::AbstractArray, f::BiasedGradient, x::AbstractArray)
  v = value_and_gradient!(g, f.f, x)
  g .*= 1.05
  v
end

struct FalseZero <: SeparableFunction                  # declares that it preserves zero, but shifts it
end
PB.value_entry(::FalseZero, v, i) = abs2(v - 0.1) / 2
PB.prox_block(::FalseZero, v, i, γ) = (v + 0.1γ) / (1 + γ)
PB.iszeropreserving(::FalseZero) = true

struct FalseZeroWrapper{F<:AbstractFunction} <: AbstractFunction
  f::F
end
PB.isseparable(::Type{<:FalseZeroWrapper}) = true
PB.iszeropreserving(::FalseZeroWrapper) = true
PB.prox_block(f::FalseZeroWrapper, v::Real, i::Integer, γ::Real) = prox_block(f.f, v, i, γ)
PB.prox!(y::AbstractArray, f::FalseZeroWrapper, x::AbstractArray, γ::Real) = prox!(y, f.f, x, γ)

# a separable wrapper with correct delegated prox, whose value is shifted by 1 (so f(0) = 1)
struct OffsetWrapper{F<:AbstractFunction} <: AbstractFunction
  f::F
end
PB.isseparable(::Type{<:OffsetWrapper}) = true
PB.iszeropreserving(::OffsetWrapper) = true
PB.prox_block(f::OffsetWrapper, v::Real, i::Integer, γ::Real) = prox_block(f.f, v, i, γ)
PB.prox!(y::AbstractArray, f::OffsetWrapper, x::AbstractArray, γ::Real) = prox!(y, f.f, x, γ)
(f::OffsetWrapper)(x::AbstractArray) = f.f(x) + 1

# the same wrapper with a correct value, and without any standalone value
struct ValueWrapper{F<:AbstractFunction, B} <: AbstractFunction
  f::F
end
ValueWrapper(f::F; value::Bool = true) where {F<:AbstractFunction} = ValueWrapper{F, value}(f)
PB.isseparable(::Type{<:ValueWrapper}) = true
PB.iszeropreserving(::ValueWrapper) = true
PB.prox_block(f::ValueWrapper, v::Real, i::Integer, γ::Real) = prox_block(f.f, v, i, γ)
PB.prox!(y::AbstractArray, f::ValueWrapper, x::AbstractArray, γ::Real) = prox!(y, f.f, x, γ)
(f::ValueWrapper{<:Any, true})(x::AbstractArray) = f.f(x)

# value methods that accept dense vectors only (tested at zero alone, and reported as skipped on the
# sparse paths), and ones that are wrong only on a SparseIterate or only in value_and_prox!
struct VectorValue{F<:AbstractFunction} <: AbstractFunction
  f::F
end
struct WrongFusedOutput{F<:AbstractFunction} <: AbstractFunction   # right value, but never writes y
  f::F
end
struct WrongSparseValue{F<:AbstractFunction} <: AbstractFunction
  f::F
end
struct WrongFusedValue{F<:AbstractFunction} <: AbstractFunction
  f::F
end
for W in (:VectorValue, :WrongSparseValue, :WrongFusedValue, :WrongFusedOutput)
  @eval begin
    PB.isseparable(::Type{<:$W}) = true
    PB.iszeropreserving(::$W) = true
    PB.prox_block(f::$W, v::Real, i::Integer, γ::Real) = prox_block(f.f, v, i, γ)
    PB.prox!(y::AbstractArray, f::$W, x::AbstractArray, γ::Real) = prox!(y, f.f, x, γ)
  end
end
(f::VectorValue)(x::Vector) = f.f(x)
(f::WrongSparseValue)(x::AbstractArray) = f.f(x)
(f::WrongSparseValue)(x::SparseIterate) = f.f(x) + 1
(f::WrongFusedValue)(x::AbstractArray) = f.f(x)
(f::WrongFusedOutput)(x::AbstractArray) = f.f(x)
PB.value_and_prox!(y::AbstractArray, f::WrongFusedOutput, x::AbstractArray, γ::Real) =
  f.f(prox(f.f, x, γ))                                          # y is left as it was
PB.value_and_prox!(y::AbstractArray, f::WrongFusedValue, x::AbstractArray, γ::Real) =
  (prox!(y, f.f, x, γ); f.f(y) + 1)

# a vector-only value that is wrong at zero (a vacuous pass on a matrix x without the report), and a
# callable whose only call method takes two arguments (no value method for any array)
struct OffsetVectorValue{F<:AbstractFunction} <: AbstractFunction
  f::F
end
struct TwoArgCall{F<:AbstractFunction} <: AbstractFunction
  f::F
end
for W in (:OffsetVectorValue, :TwoArgCall)
  @eval begin
    PB.isseparable(::Type{<:$W}) = true
    PB.iszeropreserving(::$W) = true
    PB.prox_block(f::$W, v::Real, i::Integer, γ::Real) = prox_block(f.f, v, i, γ)
    PB.prox!(y::AbstractArray, f::$W, x::AbstractArray, γ::Real) = prox!(y, f.f, x, γ)
  end
end
(f::OffsetVectorValue)(x::Vector) = f.f(x) + 1
(f::TwoArgCall)(x::AbstractArray, y::AbstractArray) = f.f(x) + f.f(y)

struct WrongEntries <: SeparableFunction               # prox! overridden, disagreeing with prox_block
  λ::Float64
end
PB.value_entry(f::WrongEntries, v, i) = f.λ * abs(v)
PB.prox_block(f::WrongEntries, v, i, γ) = shrink(v, γ * f.λ)
PB.prox!(y::AbstractArray, ::WrongEntries, x::AbstractArray, γ::Real) = (y .= x)

struct WrongKKT <: SeparableFunction                   # NormL1 with twice the threshold in kkt_violation
  λ::Float64
end
PB.value_entry(f::WrongKKT, v, i) = f.λ * abs(v)
PB.prox_block(f::WrongKKT, v, i, γ) = shrink(v, γ * f.λ)
PB.kkt_violation(f::WrongKKT, g::AbstractArray, i::Integer) = abs(g[i]) - 2f.λ

struct UserSquare{D} <: SeparableFunction end
PB.entry_parameters(::UserSquare{true}) = ()
PB.value_entry(::UserSquare, v, i) = abs2(v) / 2
PB.prox_block(::UserSquare, v, i, γ) = v / (1 + γ)
PB.iszeropreserving(::UserSquare) = true

# tuple-valued arguments: functions that fail the gradient check, and one that records its points
struct BlockBiased{F<:AbstractFunction} <: AbstractFunction   # f with the gradient wrong by 1 + ε in the block at path
  f::F
  path::Tuple{Vararg{Int}}
  ε::Float64
end
PB.issmooth(::Type{<:BlockBiased}) = true
(w::BlockBiased)(x) = w.f(x)
function PB.value_and_gradient!(g::Tuple, w::BlockBiased, x::Tuple)
  v = value_and_gradient!(g, w.f, x)
  block = foldl((t, i) -> t[i], w.path; init = g)
  block .*= 1 + w.ε
  v
end

struct InconsistentGradients <: AbstractFunction       # gradient! writes 2x in the last block, value_and_gradient! x
end
PB.issmooth(::Type{InconsistentGradients}) = true
(::InconsistentGradients)(x::Tuple) = sum(b -> sum(abs2, b), x) / 2
PB.value_and_gradient!(g::Tuple, f::InconsistentGradients, x::Tuple) = (foreach(copyto!, g, x); f(x))
function PB.gradient!(g::Tuple, ::InconsistentGradients, x::Tuple)
  foreach(copyto!, g, x)
  g[end] .*= 2
  g
end

struct ReturnsCopy <: AbstractFunction                 # gradient! returns a copy of its output, not the output
end
PB.issmooth(::Type{ReturnsCopy}) = true
(::ReturnsCopy)(x::Tuple) = sum(b -> sum(abs2, b), x) / 2
PB.value_and_gradient!(g::Tuple, f::ReturnsCopy, x::Tuple) = (foreach(copyto!, g, x); f(x))
PB.gradient!(g::Tuple, ::ReturnsCopy, x::Tuple) = (foreach(copyto!, g, x); map(copy, g))

struct WrongStructure <: AbstractFunction              # its gradient has one block less than x
end
PB.issmooth(::Type{WrongStructure}) = true
(::WrongStructure)(x::Tuple) = sum(b -> sum(abs2, b), x) / 2
PB.gradient(::WrongStructure, x::Tuple) = Base.front(x)

struct WrongReturnedValue{F<:AbstractFunction} <: AbstractFunction   # f; value_and_gradient! returns f(x) + δ
  f::F
  δ::Float64
end
PB.issmooth(::Type{<:WrongReturnedValue}) = true
(w::WrongReturnedValue)(x) = w.f(x)
PB.value_and_gradient!(g::Tuple, w::WrongReturnedValue, x::Tuple) = value_and_gradient!(g, w.f, x) + w.δ

struct Coupled{T<:Real} <: AbstractFunction            # ‖A x₁ + B x₂ - y‖²/2: the two blocks share one residual
  A::Matrix{T}
  B::Matrix{T}
  y::Vector{T}
end
PB.issmooth(::Type{<:Coupled}) = true
_residual(f::Coupled, x::Tuple) = f.A * x[1] .+ f.B * x[2] .- f.y
(f::Coupled)(x::Tuple) = sum(abs2, _residual(f, x)) / 2
function PB.value_and_gradient!(g::Tuple, f::Coupled, x::Tuple)
  r = _residual(f, x)
  g[1] .= f.A' * r
  g[2] .= f.B' * r
  sum(abs2, r) / 2
end

struct Recorder{F<:AbstractFunction} <: AbstractFunction  # f, with the points where f and its gradient were evaluated
  f::F
  points::Vector{Any}
  gradients::Base.RefValue{Int}
end
Recorder(f) = Recorder(f, Any[], Ref(0))
PB.issmooth(::Type{<:Recorder}) = true
(r::Recorder)(x) = (push!(r.points, deepcopy(x)); r.f(x))
PB.value_and_gradient!(g, r::Recorder, x) = (r.gradients[] += 1; value_and_gradient!(g, r.f, x))

@testset "ProximalBase.Testing" begin
  x = 2 .* randn(12)
  w = rand(12) .+ 0.5
  γ = 0.7

  @testset "public helpers" begin
    if VERSION >= v"1.11"
      for name in (:test_prox, :test_gradient, :test_separable, :test_conjugate, :test_kkt, :test_iterate, :test_traits,
                   :allocations, :brute_force_prox)
        @test Base.ispublic(PT, name)
        @test !Base.isexported(PT, name)
      end
    end
  end

  @testset "symmetric declarations" begin
    X = [1. .2; .2 2.]
    @test passes(() -> PT.test_separable(UserSquare{true}(), X, γ; symmetric = true))
    @test failures(() -> PT.test_separable(UserSquare{false}(), X, γ; symmetric = true)) == 1
    @test passes(() -> PT.test_separable(UserSquare{false}(), X, γ))
  end

  @testset "iterate bookkeeping" begin
    groups = [1:2, 3:4, 5:6]
    large_atom = AtomIterate(ones(65_536), [i:i+3 for i in 1:4:65_536])
    large_sparse = SparseIterate(65_536)
    for i in 1:16:65_536
      large_sparse[i] = 1
    end
    state(x) = deepcopy(ntuple(i -> getfield(x, i), fieldcount(typeof(x))))
    for x in (AtomIterate(6, groups), AtomIterate([1., 0, 0, 0, 2, 0], groups),
              SparseIterate(4), SparseIterate([1., 0, 2, 0]),
              SparseIterate(2, 2), SparseIterate([1. 0; 0 2]),
              SymmetricSparseIterate(2), SymmetricSparseIterate([1. 2; 2 0]),
              large_atom, large_sparse)
      c = x isa SymmetricSparseIterate ? coordinates(x) : x
      c[1] = 1
      c[1] = 0
      before = state(x)
      @test passes(() -> @test PT.test_iterate(x) === x)
      @test isequal(state(x), before)
    end

    corruptions = Any[]
    a = AtomIterate(6, groups)
    a.storage[3] = 1
    push!(corruptions, a)
    for mutate! in (a -> (a.active[2] = a.active[1]),
                    a -> (a.active[1] = 4),
                    a -> (a.nactive = -1),
                    a -> (a.nactive = 4),
                    a -> (a.group_of[1] = 2),
                    a -> resize!(a.group_of, 5),
                    a -> resize!(a.position, 2))
      a = AtomIterate(ones(6), groups)
      mutate!(a)
      push!(corruptions, a)
    end
    a = AtomIterate(6, [[1, 2], [3, 4], [5, 6]])
    a.groups[3][2] = 7
    push!(corruptions, a)
    a = AtomIterate(4, [[1, 2], [3, 4]])
    a.groups[1][2] = 1
    a.storage[2] = 1
    push!(corruptions, a)
    a = AtomIterate(4, [[1, 2], [3, 4]])
    a.groups[2][1] = 2
    push!(corruptions, a)
    for make in (() -> SparseIterate([1., 0, 2, 0]),
                 () -> SparseIterate([1. 2; 0 0]),
                 () -> SymmetricSparseIterate([1. 2; 2 0]))
      for mutate! in (a -> (a.ind2nzval[a.nzval2ind[1]] = 0),
                      a -> (a.ind2nzval[end] = 1),
                      a -> (a.nzval2ind[1] = length(a.ind2nzval)),
                      a -> (a.nzval2ind[2] = a.nzval2ind[1]),
                      a -> (a.nzval2ind[1] = length(a.ind2nzval) + 1),
                      a -> (a.ind2nzval[end] = a.nnz + 1),
                      a -> (a.nnz = -1),
                      a -> (a.nnz = length(a.ind2nzval) + 1),
                      a -> resize!(a.nzval, 1))
        a = make()
        mutate!(a)
        push!(corruptions, a)
      end
    end
    a = SymmetricSparseIterate(2)
    resize!(a.ind2nzval, 2)
    push!(corruptions, a)
    a = SymmetricSparseIterate(2)
    resize!(a.nzval, 2)
    resize!(a.nzval2ind, 2)
    resize!(a.ind2nzval, 2)
    push!(corruptions, a)
    for x in corruptions
      before = state(x)
      ts = @testset Collect "corrupted iterate" begin
        @test PT.test_iterate(x) === x
      end
      @test count(r -> r isa Test.Fail, ts.results) > 0
      @test count(r -> r isa Test.Error, ts.results) == 0
      @test isequal(state(x), before)
    end
  end

  @testset "correct functions pass" begin
    @test passes(() -> PT.test_traits(NormL1(0.5); proximable = true, separable = true, convex = true))
    @test passes(() -> PT.test_prox(NormL1(0.5, w), x, γ))
    @test passes(() -> PT.test_prox(IndPSD(), Matrix(Symmetric(randn(5, 5))), γ; symmetric = true))
    X, y = randn(30, 12), randn(30)
    @test passes(() -> PT.test_gradient(LeastSquaresLoss(y, X), x))
    @test passes(() -> PT.test_gradient(LogisticLoss(rand(0:1, 30), X), x ./ 5; rng = Xoshiro(3)))
    groups = [1:3, 4:8, 9:12]
    for f in (NormL1(0.5, w), ElasticNet(0.3, 0.6), MCP(0.4), IndBox(-1.0, 2.0), NormL0(0.2))
      @test passes(() -> PT.test_separable(f, x, γ; groups))
    end
    @test passes(() -> PT.test_separable(SqrHingeLoss(), x, 0.4; groups = 2))
    for f in (Scaled(NormL1(1.0), 2.0), Conjugate(NormL1(1.0)), Conjugate(ElasticNet(0.3, 0.6)))
      @test passes(() -> PT.test_separable(f, x, γ; rng = Xoshiro(2)))
    end
    Ws = Matrix(Symmetric(rand(5, 5)))
    @test passes(() -> PT.test_separable(NormL1(0.5, Ws), randn(5, 5), γ; symmetric = true))
    # an integer x: the iterates hold floats, so the prox results can be stored
    @test passes(() -> PT.test_separable(NormL1(0.5), [3, -2, 0, 1, -4, 2], γ; rng = Xoshiro(2), groups = 2))
    @test passes(() -> PT.test_separable(NormL1(0.5), [3 -2 1; -2 0 4; 1 4 -3], γ; rng = Xoshiro(2), symmetric = true))
    # zero-preserving functions with and without a standalone value
    # (a function whose value checks all run, or that has no value method for an array, logs nothing)
    for f in (NormL1(0.5), Scaled(NormL1(0.5), 2.0), Conjugate(NormL1(0.5)), ValueWrapper(NormL1(0.5)),
              ValueWrapper(Scaled(NormL1(0.5), 2.0)))
      @test_logs @test passes(() -> PT.test_separable(f, x, γ; rng = Xoshiro(2)))
    end
    for f in (Conjugate(ElasticNet(0.3, 0.6)), Scaled(Conjugate(ElasticNet(0.3, 0.6)), 2.0),
              ValueWrapper(NormL1(0.5); value = false), TwoArgCall(NormL1(0.5)))
      @test !PB._has_value(f) || !applicable(f, zeros(length(x)))
      @test_logs @test passes(() -> PT.test_separable(f, x, γ; rng = Xoshiro(2)))
    end
    # a value method that rejects the array: valid, and the skipped checks are reported once
    f = VectorValue(NormL1(0.5))
    @test_logs (:info, r"skipped the value checks on a SparseIterate.*VectorValue.*SparseIterate") begin
      @test passes(() -> PT.test_separable(f, x, γ; rng = Xoshiro(2)))
    end
    @test_logs (:info, r"skipped the value checks at zero and on a SparseIterate.*VectorValue.*Matrix") begin
      @test passes(() -> PT.test_separable(f, randn(4, 3), γ; rng = Xoshiro(2)))
    end
    @test_logs (:info, r"skipped the value checks at zero and on a SparseIterate.*VectorValue.*Matrix") begin
      @test passes(() -> PT.test_separable(f, Matrix(Symmetric(randn(4, 4))), γ; rng = Xoshiro(2), symmetric = true))
    end
    @test passes(() -> PT.test_conjugate(NormL1(0.5), x, γ; conjugate = IndBallLinf(0.5)))
    @test passes(() -> PT.test_conjugate(NormL1(0.5), x, γ))
    @test passes(() -> PT.test_conjugate(SqrNormL2(2.0), x, γ))
    @test passes(() -> PT.test_conjugate(NormL2(1.0), x, γ))
    # the value of its conjugate is not known: Moreau's decomposition alone
    @test passes(() -> PT.test_conjugate(ElasticNet(0.3, 0.6), x, γ))
    for f in (NormL1(0.5, w), ElasticNet(0.3, 0.6), MCP(0.4), SCAD(0.4), NormL1Nonneg(0.3), IndBox(-1.0, 2.0))
      @test passes(() -> PT.test_kkt(f, x, 0.5))
    end
    gn = GroupNormL2(0.5, groups)
    @test passes(() -> PT.test_kkt(gn, AtomIterate(12, gn), γ))
    @test passes(() -> PT.test_kkt(NormL1(0.5), AtomIterate(12, groups), γ))
  end

  @testset "gradient precision" begin
    for seed in 1:20
      T = Float32
      Random.seed!(seed)
      X = randn(T, 50, 5)
      β = randn(T, 5) ./ 10
      fs = (LogisticLoss(T.(rand(50) .< 0.5), X), LeastSquaresLoss(randn(T, 50), X),
            PoissonLoss(T.(rand(0:3, 50)), X ./ 3))
      for f in fs
        @test passes(() -> PT.test_gradient(f, β))
        @test passes(() -> PT.test_gradient(f, β; atol = 0.0))
      end
    end
    for T in (Float32, Float64), seed in 1:20
      Random.seed!(seed)
      X = randn(T, 50, 5)
      β = randn(T, 5) ./ 10
      fs = (LeastSquaresLoss(randn(T, 50), X), LogisticLoss(T.(rand(50) .< 0.5), X),
            PoissonLoss(T.(rand(0:3, 50)), X ./ 3))
      for f in fs
        ts = @testset Collect "biased gradient" begin
          PT.test_gradient(BiasedGradient(f), β)
        end
        @test count(r -> r isa Test.Fail, ts.results) > 0
        @test count(r -> r isa Test.Error, ts.results) == 0
      end
    end
    for T in (Float64,)
      @test passes(() -> PT.test_gradient(SqrNormL2(), T[1, 2]; rng = Xoshiro(4)))
      @test passes(() -> PT.test_gradient(SqrNormL2(), T[1, 2]; h = 1e-6, rtol = 1e-5,
                                         atol = 1e-6, rng = Xoshiro(4)))
    end
    f = BiasedGradient(SqrNormL2())
    @test passes(() -> PT.test_gradient(f, Float32[1, 2]; h = 0.01, rtol = 0.1, atol = 0.0,
                                       rng = Xoshiro(4)))
    @test passes(() -> PT.test_gradient(f, Float32[1, 2]; h = 0.01, rtol = 0.0, atol = 1.0,
                                       rng = Xoshiro(4)))
  end

  @testset "broken functions fail" begin
    @test failures(() -> PT.test_traits(NormL1(0.5); smooth = true, proximable = true, separable = true,
                                        convex = true)) > 0
    @test failures(() -> PT.test_prox(IdentityProx(0.5), x, γ)) > 0
    @test failures(() -> PT.test_prox(WrongValue(0.5), x, γ)) > 0
    # right with a separate output, and its point is right in place: only the value is wrong in place (O7)
    ip = InPlaceValue(0.5)
    y = similar(x)
    @test value_and_prox!(y, ip, x, γ) ≈ ip(y)
    z = copy(x)
    @test value_and_prox!(z, ip, z, γ) < ip(z)
    @test z ≈ y
    @test failures(() -> PT.test_prox(ip, x, γ)) > 0
    @test failures(() -> PT.test_gradient(WrongGradient(), x)) > 0
    @test failures(() -> PT.test_separable(FalseZero(), x, γ)) > 0
    f = FalseZeroWrapper(Translate(SqrNormL2(), fill(0.1, length(x))))
    ts = @testset Collect "false zero wrapper" begin
      PT.test_separable(f, x, γ; rng = Xoshiro(2))
    end
    @test count(r -> r isa Test.Fail, ts.results) > 0
    @test count(r -> r isa Test.Error, ts.results) == 0
    f = OffsetWrapper(NormL1(1.0))
    @test f(zeros(length(x))) == 1 && prox(f, x, γ) == prox(NormL1(1.0), x, γ)
    ts = @testset Collect "offset wrapper" begin
      PT.test_separable(f, x, γ; rng = Xoshiro(2))
    end
    @test count(r -> r isa Test.Fail, ts.results) == 1          # only the value at zero
    @test count(r -> r isa Test.Error, ts.results) == 0
    @test only(filter(r -> r isa Test.Fail, ts.results)).orig_expr == "iszero(f(z))"
    f = OffsetVectorValue(NormL1(1.0))                          # f(0) = 1, which a matrix x cannot test
    @test f(zeros(length(x))) == 1
    ts = @test_logs (:info, r"skipped the value checks on a SparseIterate.*OffsetVectorValue") begin
      @testset Collect "offset vector value" begin
        PT.test_separable(f, x, γ; rng = Xoshiro(2))
      end
    end
    @test count(r -> r isa Test.Fail, ts.results) == 1          # the value at zero
    @test count(r -> r isa Test.Error, ts.results) == 0
    ts = @test_logs (:info, r"skipped the value checks at zero.*OffsetVectorValue.*Matrix") begin
      @testset Collect "offset vector value, matrix" begin
        PT.test_separable(f, randn(4, 3), γ; rng = Xoshiro(2))
      end
    end
    @test !any(r -> r isa Test.Fail || r isa Test.Error || r isa Test.Broken, ts.results)
    f = WrongSparseValue(NormL1(1.0))                           # f(SparseIterate) is off by 1
    ts = @testset Collect "wrong sparse value" begin
      PT.test_separable(f, x, γ; rng = Xoshiro(2))
    end
    @test count(r -> r isa Test.Fail, ts.results) >= 1          # the sparse value, and with it value_and_prox!
    @test count(r -> r isa Test.Error, ts.results) == 0
    f = WrongFusedValue(NormL1(1.0))                            # only value_and_prox! is off by 1
    ts = @testset Collect "wrong fused value" begin
      PT.test_separable(f, x, γ; rng = Xoshiro(2))
    end
    @test count(r -> r isa Test.Fail, ts.results) == 1
    @test count(r -> r isa Test.Error, ts.results) == 0
    @test only(filter(r -> r isa Test.Fail, ts.results)).orig_expr == "value_and_prox!(yv, f, xs, γ) ≈ f(pz)"
    f = WrongFusedOutput(NormL1(1.0))                           # x has entries above the threshold, so prox(x) != x
    ts = @testset Collect "wrong fused output" begin
      PT.test_separable(f, x, γ; rng = Xoshiro(2))
    end
    @test count(r -> r isa Test.Fail, ts.results) == 1
    @test count(r -> r isa Test.Error, ts.results) == 0
    @test only(filter(r -> r isa Test.Fail, ts.results)).orig_expr == "_dense(yv) ≈ _dense(pz)"
    @test failures(() -> PT.test_separable(WrongEntries(0.5), x, γ)) > 0
    @test failures(() -> PT.test_conjugate(NormL1(0.5), x, γ; conjugate = IndBallLinf(1.0))) > 0
    @test failures(() -> PT.test_kkt(WrongKKT(0.5), x, γ)) > 0
  end

  @testset "tuple gradients" begin
    shapes(b::Tuple) = map(shapes, b)
    shapes(b::AbstractArray) = size(b)
    s = SeparableSum(SqrNormL2(), SqrNormL2(2))
    xt = ([1.0, 2.0], [3.0])
    g = PT.test_gradient(s, xt)                          # a MethodError before #159
    @test g == ([1.0, 2.0], [6.0])
    @test g isa Tuple{Vector{Float64}, Vector{Float64}}
    @test passes(() -> PT.test_gradient(s, xt))

    @testset "block structure" begin
      s1 = SeparableSum(SqrNormL2(3))
      @test PT.test_gradient(s1, ([1.0, -2.0, 4.0],)) == ([3.0, -6.0, 12.0],)
      sm = SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3))
      xm = ([1.0, 2.0], [3.0 4.0 5.0; 6.0 7.0 8.0], [9.0])
      gm = PT.test_gradient(sm, xm)
      @test shapes(gm) == shapes(xm)
      @test gm[2] == 2 .* xm[2] && gm[3] == 3 .* xm[3]
      n = SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2)), SqrNormL2(3))
      xn = (([1.0, 2.0], [3.0]), [4.0 5.0; 6.0 7.0])
      gn = PT.test_gradient(n, xn)
      @test shapes(gn) == shapes(xn) && gn[1][2] == [6.0] && gn[2] == 3 .* xn[2]
      d = SeparableSum(n, SeparableSum(SqrNormL2(), SeparableSum(SqrNormL2(2), SqrNormL2(3))))
      xd = (xn, ([1.0], ([2.0, 3.0], [4.0 5.0])))
      @test shapes(PT.test_gradient(d, xd)) == shapes(xd)
      @test passes(() -> PT.test_gradient(d, xd; rng = Xoshiro(5)))
      # many blocks: the traversal is generated, not recursive in the tuple length
      many = SeparableSum(ntuple(_ -> SqrNormL2(), 40)...)
      xmany = ntuple(i -> fill(Float64(i), 2), 40)
      @test passes(() -> PT.test_gradient(many, xmany))
      @test PT.test_gradient(many, xmany) == xmany
    end

    @testset "precisions" begin
      g32 = PT.test_gradient(s, (Float32[1, 2], Float32[3]))
      @test g32 isa Tuple{Vector{Float32}, Vector{Float32}} && g32 == (Float32[1, 2], Float32[6])
      # smooth losses that are not quadratic, over seeds: no spurious failures
      for T in (Float32, Float64), seed in 1:20
        Random.seed!(seed)
        X1, X2 = randn(T, 50, 5), randn(T, 40, 4)
        f = SeparableSum(LogisticLoss(T.(rand(50) .< 0.5), X1), LeastSquaresLoss(randn(T, 40), X2))
        β = (randn(T, 5) ./ 10, randn(T, 4) ./ 10)
        @test passes(() -> PT.test_gradient(f, β))
        @test passes(() -> PT.test_gradient(f, β; atol = 0.0))
      end
      # model data of another precision than the variables is supported: Float64 data, Float32 blocks
      for seed in 1:20
        Random.seed!(seed)
        X1, X2 = randn(50, 5), randn(40, 4)
        f = SeparableSum(LogisticLoss(Float64.(rand(50) .< 0.5), X1), LeastSquaresLoss(randn(40), X2))
        β = (randn(Float32, 5) ./ 10, randn(Float32, 4) ./ 10)
        @test passes(() -> PT.test_gradient(f, β))
        gβ = PT.test_gradient(f, β)
        @test gβ isa Tuple{Vector{Float32}, Vector{Float32}}
        c = Coupled(randn(30, 5), randn(30, 4), randn(30))
        @test passes(() -> PT.test_gradient(c, β))
        @test passes(() -> PT.test_gradient(c, map(b -> Float64.(b), β)))
      end
    end

    @testset "coupled blocks" begin
      for T in (Float32, Float64)
        Random.seed!(2)
        c = Coupled(randn(T, 30, 5), randn(T, 30, 4), randn(T, 30))
        xc = (randn(T, 5), randn(T, 4))
        @test passes(() -> PT.test_gradient(c, xc))
        gc = PT.test_gradient(c, xc)
        r = c.A * xc[1] .+ c.B * xc[2] .- c.y             # the analytic gradient
        @test gc[1] ≈ c.A' * r && gc[2] ≈ c.B' * r
        @test gc isa Tuple{Vector{T}, Vector{T}}
      end
    end

    @testset "directions and steps" begin
      # the points of the finite differences are x ± h d, d drawn block by block from rng
      r = Recorder(SeparableSum(SqrNormL2(), SqrNormL2(2), SqrNormL2(3)))
      xb = ([1.0, 2.0], [3.0 4.0; 5.0 6.0], [7.0])
      PT.test_gradient(r, xb; h = 1e-3, directions = 4, rng = Xoshiro(3))
      moved = filter(!=(xb), r.points)
      @test length(moved) == 2 * 4
      rng = Xoshiro(3)
      for k in 1:4
        d = map(b -> randn(rng, Float64, size(b)...), xb)
        @test all(map((p, b, e) -> p ≈ b .+ 1e-3 .* e, moved[2k-1], xb, d))
        @test all(map((m, b, e) -> m ≈ b .- 1e-3 .* e, moved[2k], xb, d))
        @test all(map((p, b) -> all(p .!= b), moved[2k-1], xb))     # every block moves
      end
      # an array keeps its direction stream
      ra = Recorder(SqrNormL2())
      PT.test_gradient(ra, [1.0, 2.0, 3.0]; h = 1e-3, directions = 2, rng = Xoshiro(3))
      moved = filter(!=([1.0, 2.0, 3.0]), ra.points)
      rng = Xoshiro(3)
      for k in 1:2
        @test moved[2k-1] ≈ [1.0, 2.0, 3.0] .+ 1e-3 .* randn(rng, Float64, 3)
      end
      # the same seed gives the same points and gradient; another seed other points
      points(seed) = (rr = Recorder(s); gg = PT.test_gradient(rr, xt; rng = Xoshiro(seed)); (rr.points, gg))
      @test points(7) == points(7)
      @test points(7)[1] != points(8)[1]
      # a Float64 tuple keeps the Float64 scale
      r64 = Recorder(s)
      PT.test_gradient(r64, xt; rng = Xoshiro(1))
      moved = filter(!=(xt), r64.points)
      @test length(moved) == 2 * 3
      rng = Xoshiro(1)
      d = (randn(rng, 2), randn(rng, 1))
      @test (moved[1][2] .- xt[2]) ./ d[2] ≈ [1e-6] rtol = 1e-6
      # a Float32 tuple takes the scale of a Float32 array: one check per direction
      h32 = 1e-6 * cbrt(eps(Float32) / eps(Float64))
      x32 = (Float32[1, 2], Float32[3])
      r32 = Recorder(s)
      PT.test_gradient(r32, x32; rng = Xoshiro(1))
      moved = filter(!=(x32), r32.points)
      @test length(moved) == 2 * 3
      rng = Xoshiro(1)
      d = (randn(rng, Float32, 2), randn(rng, Float32, 1))
      @test all(map((p, b, e) -> isapprox((p .- b) ./ e, fill(Float32(h32), length(b)); rtol = 1e-2), moved[1], x32, d))
    end

    @testset "keywords" begin
      f = BlockBiased(s, (2,), 0.05)
      @test failures(() -> PT.test_gradient(f, xt; rng = Xoshiro(2))) > 0
      @test passes(() -> PT.test_gradient(f, xt; rtol = 0.0, atol = 1.0, rng = Xoshiro(2)))
      @test passes(() -> PT.test_gradient(f, xt; rtol = 0.5, atol = 0.0, directions = 1, rng = Xoshiro(2)))
      fl = SeparableSum(LogisticLoss([1, 0, 1, 1, 0], randn(Xoshiro(2), 5, 2)), SqrNormL2())
      xl = ([0.3, -0.2], [1.0])
      @test passes(() -> PT.test_gradient(fl, xl; rng = Xoshiro(2)))
      @test failures(() -> PT.test_gradient(fl, xl; h = 1.0, rng = Xoshiro(2))) > 0     # the step is used
      @test passes(() -> PT.test_gradient(fl, xl; h = 1.0, rtol = 1.0, atol = 1.0, rng = Xoshiro(2)))
    end

    @testset "unsupported arguments" begin
      @test_throws ArgumentError PT.test_gradient(s, ([1.0], 2.0))
      @test_throws ArgumentError PT.test_gradient(s, ())
      @test_throws ArgumentError PT.test_gradient(s, ((), [1.0]))
      @test_throws MethodError PT.test_gradient(s, (a = [1.0], b = [2.0]))
    end

    @testset "blocks of different precisions are rejected before f is called" begin
      n = SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2)), SqrNormL2(3))
      cases = [(s, (Float32[1, 2], Float64[3])), (s, (Float64[1, 2], Float32[3])),
               (s, (Float16[1, 2], Float32[3])), (s, (Int[1, 2], Float32[3])),
               (n, ((Float64[1, 2], Float32[3]), Float64[4])), (n, ((Float64[1, 2], Float64[3]), Float32[4])),
               (n, ((Float32[1, 2], Float32[3]), Float64[4]))]
      for (f, x) in cases
        r = Recorder(f)
        @test_throws ArgumentError PT.test_gradient(r, x)
        @test isempty(r.points) && r.gradients[] == 0
      end
      err = try
        PT.test_gradient(s, (Float32[1, 2], Float64[3]))
      catch e
        e
      end
      @test err isa ArgumentError
      @test err.msg == "test_gradient requires tuple blocks with a common floating-point type; got Float32 and Float64"
      # the type is float(eltype), as for an array: integer and Float64 blocks share Float64
      @test passes(() -> PT.test_gradient(s, ([1, 2], [3.0])))
      @test passes(() -> PT.test_gradient(s, ([1, 2], [3])))
    end
  end

  @testset "incorrect tuple gradients fail" begin
    s = SeparableSum(SqrNormL2(), SqrNormL2(2))
    xt = ([1.0, 2.0], [3.0])
    # each case fails an assertion and nothing throws, whichever block is wrong
    detected(check) = ((nf, ne) = outcomes(check); nf > 0 && ne == 0)
    @testset "Float64 blocks" begin
      n = SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2)), SqrNormL2(3))
      xn = (([1.0, 2.0], [3.0]), [4.0 5.0; 6.0 7.0])
      for seed in 1:20
        for path in ((1,), (2,))
          @test detected(() -> PT.test_gradient(BlockBiased(s, path, 0.01), xt; rng = Xoshiro(seed)))
        end
        @test detected(() -> PT.test_gradient(BlockBiased(SeparableSum(SqrNormL2()), (1,), 0.01), ([1.0, 2.0],);
                                              rng = Xoshiro(seed)))
        for path in ((1, 1), (1, 2), (2,))
          @test detected(() -> PT.test_gradient(BlockBiased(n, path, 0.01), xn; rng = Xoshiro(seed)))
        end
      end
    end
    @testset "Float32 blocks" begin
      s32 = SeparableSum(SqrNormL2(), SqrNormL2(2))
      for seed in 1:20, path in ((1,), (2,))
        @test detected(() -> PT.test_gradient(BlockBiased(s32, path, 0.2), (Float32[1, 2], Float32[3]);
                                              directions = 10, rng = Xoshiro(seed)))
      end
    end
    @testset "wrong returned value" begin
      # a correct gradient with a returned value that is 1e-4 off: the value of s at these points is 0.5
      sv = SeparableSum(SqrNormL2(), SqrNormL2())
      for x in (([1.0], [0.0]), ([0.0], [1.0]), ([sqrt(0.5), sqrt(0.5)], [0.0]), ([0.0], [0.0, 1.0]))
        @test sv(x) ≈ 0.5
        @test passes(() -> PT.test_gradient(WrongReturnedValue(sv, 0.0), x))
        @test detected(() -> PT.test_gradient(WrongReturnedValue(sv, 1e-4), x))
        @test detected(() -> PT.test_gradient(WrongReturnedValue(sv, -1e-4), x))
      end
      # in Float32 the tolerance is that of a Float32 array: sqrt(eps(Float32)) ≈ 3.4e-4
      x32 = (Float32[1], Float32[0])
      @test passes(() -> PT.test_gradient(WrongReturnedValue(sv, 0.0), x32))
      @test detected(() -> PT.test_gradient(WrongReturnedValue(sv, 1e-2), x32))
    end
    @testset "coupled blocks" begin
      for seed in 1:20, (T, ε, directions) in ((Float64, 0.01, 3), (Float32, 0.2, 10)), k in 1:2
        Random.seed!(seed)
        c = Coupled(randn(T, 30, 5), randn(T, 30, 4), randn(T, 30))
        xc = (randn(T, 5), randn(T, 4))
        @test detected(() -> PT.test_gradient(BlockBiased(c, (k,), ε), xc; directions, rng = Xoshiro(seed)))
      end
    end
    @testset "interfaces and structure" begin
      @test detected(() -> PT.test_gradient(InconsistentGradients(), xt))
      @test detected(() -> PT.test_gradient(ReturnsCopy(), xt))
      @test detected(() -> PT.test_gradient(WrongStructure(), xt))
    end
  end

  @testset "allocations and the scalar reference" begin
    a_sum(x) = sum(copy(x))
    @test PT.allocations(a_sum, x) > 0
    @test PT.allocations(prox!, similar(x), NormL1(0.5), x, γ) == 0
    for v in (-2.0, -0.2, 0.1, 1.5)
      t, h = PT.brute_force_prox(u -> 0.5 * abs(u), v, γ)
      @test t ≈ shrink(v, 0.5γ) atol = 1e-6
      @test h ≈ 0.5 * abs(t) + (t - v)^2 / (2γ)
    end
  end
end

end
