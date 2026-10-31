module SymmetricParametersTraitTest

using Test, ProximalBase, LinearAlgebra
using ..TestUtils: allocs

const PB = ProximalBase

struct MyL1{D, W} <: SeparableFunction
  w::W
end
MyL1{D}(w = nothing) where {D} = MyL1{D, typeof(w)}(w)
weight(f, i) = f.w === nothing ? 1.0 : f.w[i]
PB.entry_parameters(f::MyL1{true}) = (:w => f.w,)
PB.value_entry(f::MyL1, v, i) = weight(f, i) * abs(v)
PB.prox_entry(f::MyL1, v, i, γ) = shrink(v, γ * weight(f, i))
PB.kkt_violation(f::MyL1, g::AbstractArray, i::Integer) = abs(g[i]) - weight(f, i)
PB.lla_weight(f::MyL1, v, i) = weight(f, i)
PB.iszeropreserving(::MyL1) = true
PB.isconvex(::Type{<:MyL1}) = true

struct FreeL1 <: SeparableFunction end
PB.entry_parameters(::FreeL1) = ()
PB.value_entry(::FreeL1, v, i) = abs(v)
PB.prox_entry(::FreeL1, v, i, γ) = shrink(v, γ)
PB.iszeropreserving(::FreeL1) = true
PB.isconvex(::Type{FreeL1}) = true

struct Declared{P} <: SeparableFunction
  parameters::P
end
PB.entry_parameters(f::Declared) = f.parameters

struct CheckedOverride{W} <: SeparableFunction
  w::W
end
PB.entry_parameters(f::CheckedOverride) = (:w => f.w,)
function PB.prox!(y::AbstractArray, f::CheckedOverride, x::AbstractArray, γ::Real)
  PB.check_symmetric_parameters(f, x)
  y .= x
end

struct UncheckedOverride{W} <: SeparableFunction
  w::W
end
PB.entry_parameters(f::UncheckedOverride) = (:w => f.w,)
const override_calls = Ref(0)
PB.prox!(y::AbstractArray, f::UncheckedOverride, x::AbstractArray, γ::Real) =
  (override_calls[] += 1; y .= x)

struct CountingMatrix <: AbstractMatrix{Float64}
  data::Matrix{Float64}
  reads::Base.RefValue{Int}
end
Base.size(a::CountingMatrix) = size(a.data)
Base.IndexStyle(::Type{CountingMatrix}) = IndexLinear()
Base.getindex(a::CountingMatrix, i::Int) = (a.reads[] += 1; a.data[i])

state(x::SymmetricSparseIterate) = map(n -> deepcopy(getfield(x, n)), fieldnames(typeof(x)))
state(x::Tuple) = map(state, x)
state(x::AbstractArray) = copy(x)
function rejects(op, y, message)
  before = state(y)
  err = try op(y); nothing catch e; e end
  @test err isa ArgumentError
  @test err isa ArgumentError && err.msg == message
  @test isequal(state(y), before)
end
undeclared(f) = "$(nameof(typeof(f))) does not declare its per-entry parameters; define ProximalBase.entry_parameters(f::$(nameof(typeof(f)))) to return them, or () when it has none, to use it on a SymmetricSparseIterate"
asymmetric(name, F = "MyL1") = "the array $name of $F must be symmetric on a SymmetricSparseIterate, but entry (2, 1) is 3.0 and entry (1, 2) is 1.0; use Symmetric(A, :L) to keep the lower triangle of A, or (A + A')/2 for its symmetric part"
wrappers(f) = (f, Scaled(f, 2.), Tilt(f, zeros(2, 2)),
               Precompose(f, 2.), Translate(f, zeros(2, 2)))

@testset "Declarations at public boundaries" begin
  X = [0. .5; .5 0.]
  x = SymmetricSparseIterate(X)
  W = [0. 1.; 3. 0.]
  for (f, message) in ((MyL1{false}(), undeclared(MyL1{false}())),
                        (MyL1{false}(W), undeclared(MyL1{false}(W))),
                        (MyL1{true}(W), asymmetric(:w)))
    @test PB.entry_parameters(f) === (f isa MyL1{false} ? nothing : (:w => W,))
    for h in wrappers(f)
      rejects(y -> h(y), copy(x), message)
      for γ in (.1, fill(.1, 2, 2))
        rejects(y -> prox!(y, h, x, γ), copy(x), message)
        rejects(y -> value_and_prox!(y, h, x, γ), copy(x), message)
      end
      rejects(y -> prox_coordinate!(y, h, 2, .1), copy(x), message)
      rejects(y -> kkt_violation(h, ones(2, 2), y, 2), copy(x), message)
      rejects(y -> PB.check_symmetric_parameters(h, y), copy(x), message)
      for fs in ((h, NormL1()), (NormL1(), h))
        sumf = SeparableSum(fs...)
        rejects(y -> sumf(y), (copy(x), copy(x)), message)
        rejects(y -> prox!(y, sumf, (x, x), .1), (copy(x), copy(x)), message)
        rejects(y -> value_and_prox!(y, sumf, (x, x), .1), (copy(x), copy(x)), message)
      end
    end
    rejects(y -> lla_weights!(y, f, x), fill(-7., 2, 2), message)
    rejects(y -> lla_weights!(y, f, x), copy(x), message)
    # Dense output retains the input's ordinary entrywise semantics.
    @test prox!(zeros(2, 2), f, x, .1) ≈ prox(NormL1(1., f.w), X, .1)
    for z in (X, SparseIterate(X))
      @test f(z) ≈ NormL1(1., f.w)(z)
      @test Matrix(prox(f, z, .1)) ≈ Matrix(prox(NormL1(1., f.w), z, .1))
      @test PB.check_symmetric_parameters(f, z)
    end
  end
  rejects(y -> lla_weights!(y, NormL1(1., W), x), fill(-7., 2, 2), asymmetric(:w, "NormL1"))
end

@testset "Declared functions retain results" begin
  X = [.7 .5; .5 .9]
  for f in (FreeL1(), MyL1{true}(), MyL1{true}([1. 2.; 2. 3.]),
            MyL1{true}(Symmetric([1. 9.; 2. 3.], :L)))
    base = f isa FreeL1 ? NormL1() : NormL1(1., f.w)
    for (h, b) in zip(wrappers(f), wrappers(base))
      x = SymmetricSparseIterate(X)
      @test PB.check_symmetric_parameters(h, x)
      @test h(x) ≈ b(x)
      for γ in (.1, fill(.1, 2, 2))
        y = copy(x)
        @test Matrix(prox!(y, h, x, γ)) ≈ Matrix(prox(b, x, γ))
        @test value_and_prox!(y, h, x, γ) ≈ b(prox(b, x, γ))
      end
      y = copy(x); z = copy(x)
      @test prox_coordinate!(y, h, 2, .1) ≈ prox_coordinate!(z, b, 2, .1)
    end
  end
end

@testset "All pairs, local pairs and malformed declarations" begin
  W = ones(3, 3); W[3, 2] = 3.
  x = SymmetricSparseIterate(3)
  f = MyL1{true}(W)
  @test prox_coordinate!(x, f, 2, .1) == 0
  @test_throws ArgumentError PB.check_symmetric_parameters(f, x)
  @test_throws ArgumentError Scaled(f, 2.)(x)
  @test_throws ArgumentError prox!(copy(x), Scaled(f, 2.), x, .1)
  good = MyL1{true}(ones(3, 3))
  blocks = SeparableSum(good, SeparableSum(good, Scaled(f, 2.)))
  @test PB.check_symmetric_parameters(blocks, (x, (x, zeros(3, 3))))
  @test_throws ArgumentError PB.check_symmetric_parameters(blocks, (zeros(3, 3), (x, x)))
  @test PB.check_symmetric_parameters(SeparableSum(good, SeparableSum(good, good)), (x, (zeros(3, 3), x)))
  bad = [0. 1.; 3. 0.]
  rejects(y -> PB.check_symmetric_parameters(Declared((:first => ones(2, 2), :second => bad)), y),
    SymmetricSparseIterate(2), asymmetric(:second, "Declared"))
  @test PB.check_symmetric_parameters(Declared((:none => nothing, :scalar => 2.)), x)
  for declaration in ((:bad => "not an array",), (:bad => (1, 2),), ((:bad, ones(3, 3)),), [:bad => ones(3, 3)])
    err = try PB.check_symmetric_parameters(Declared(declaration), x); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("Declared", sprint(showerror, err))
    @test occursin("bad", sprint(showerror, err)) || occursin("entry_parameters", sprint(showerror, err))
  end
  f = CheckedOverride(bad)
  for h in (f, Scaled(f, 2.))
    rejects(y -> prox!(y, h, SymmetricSparseIterate(2), .1), SymmetricSparseIterate(2), asymmetric(:w, "CheckedOverride"))
  end
  override_calls[] = 0
  rejects(y -> prox!(y, Scaled(UncheckedOverride(bad), 2.), SymmetricSparseIterate(2), .1),
    SymmetricSparseIterate(2), asymmetric(:w, "UncheckedOverride"))
  @test override_calls[] == 0
  p = 1000
  a = CountingMatrix(ones(p, p), Ref(0))
  f = MyL1{true}(a); x = SymmetricSparseIterate(p)
  for h in (f, Scaled(f, 2.), Tilt(f, zeros(p, p)), Precompose(f, 2.))
    a.reads[] = 0
    prox_coordinate!(x, h, 2, .1)
    @test 2 <= a.reads[] <= 4
    a.reads[] = 0
    kkt_violation(h, zeros(p, p), x, 2)
    @test 2 <= a.reads[] <= 4
  end
  a.reads[] = 0
  @test Scaled(f, 2.)(x) == 0
  @test a.reads[] == 2p * (p - 1)
end

@testset "Built-in separable declaration inventory" begin
  # Walk the concrete families, including any future non-exported ones.
  for name in names(PB; all = true)
    F = getfield(PB, name)
    F isa Type && F <: SeparableFunction || continue
    isabstracttype(Base.unwrap_unionall(F)) && continue
    method = which(PB.entry_parameters, (F,))
    @test method !== which(PB.entry_parameters, (SeparableFunction,))
  end
  for f in (NormL1(), NormL1Nonneg(), ElasticNet(.1, .2), CappedL1(.1, 1.),
            LogPenalty(.1, 1.), NormLq(.1, .5), Berhu(.1, 1.), CheckLoss(.3),
            HingeLoss(), SqrHingeLoss(), EpsInsensitiveLoss(.2), BernoulliLoss(.5),
            IndBox(-1., 1.), IndBinary(0., 1.), IndPattern(trues(2, 2)), IndBanded(1, 2),
            KLDivergence(1.), IndNonnegative(), NormL0(), MCP(.1), SCAD(.1), NegEntropy(), NegLog())
    @test PB.entry_parameters(f) isa Tuple
    @test PB.check_symmetric_parameters(f, SymmetricSparseIterate(2))
  end
end

@testset "Built-in checked hot calls allocate nothing" begin
  x = SymmetricSparseIterate([0. .5; .5 0.]); g = ones(2, 2)
  for f in (NormL1(), NormL1(.1, ones(2, 2)), NormL1(.1, Symmetric(ones(2, 2))),
            Scaled(Tilt(NormL1(.1), zeros(2, 2)), 2.))
    for domain in (PB._AllPairs(), PB._StoredPairs(x, x), PB._CoordinatePair(2))
      @test allocs(PB._check_entry_arrays, f, x, domain) == 0
    end
    @test allocs(prox!, copy(x), f, x, .1) == 0
    @test allocs(kkt_violation, f, g, x, 2) == 0
  end
end

end
