module SeparableBlocksTest

using Test, ProximalBase
using ..TestUtils

const PB = ProximalBase
@noinline call_oracle(op::F, args::Vararg{Any, N}) where {F, N} = op(args...)
blocks_match(a::AbstractArray, b::AbstractArray) = a ≈ b
blocks_match(a::Tuple, b::Tuple) = all(map(blocks_match, a, b))

function check_allocations(op, args...)
  @nospecialize op args
  @test allocs(op, args...) == 0
  @test allocs(call_oracle, op, args...) == 0
end

function check_proximal_blocks(fs, x, y, γ)
  @nospecialize fs x y
  s = SeparableSum(fs)
  reference = map((f, xk) -> prox(f, xk, γ), fs, x)
  @test prox!(y, s, x, γ) === y
  @test blocks_match(y, reference)
  expected = sum(map((f, xk) -> f(xk), fs, x))
  @test isequal(s(x), expected)
  @test typeof(s(x)) === typeof(expected)
  check_allocations(s, x)
  foreach(yk -> fill!(yk, 19), y)
  expected = sum(map((f, yk) -> f(yk), fs, reference))
  @test isequal(value_and_prox!(y, s, x, γ), expected)
  @test blocks_match(y, reference)
  for op in (prox!, value_and_prox!)
    check_allocations(op, y, s, x, γ)
  end
  check_conjugate_blocks(fs, x, y, γ)
  if length(x) in (33, 64)
    @inferred prox(s, x, γ)
    @inferred prox!(y, s, x, γ)
    @test (@inferred value_and_prox!(y, s, x, γ)) isa Float64
  end
end

function check_conjugate_blocks(fs, x, y, γ)
  @nospecialize fs x y
  c = Conjugate(SeparableSum(fs))
  cr = map(Conjugate, fs)
  reference = map((f, xk) -> prox(f, xk, γ), cr, x)
  @test prox!(y, c, x, γ) === y
  @test blocks_match(y, reference)
  expected = sum(map((f, xk) -> value_and_prox!(similar(xk), f, xk, γ), cr, x))
  foreach(yk -> fill!(yk, 19), y)
  @test value_and_prox!(y, c, x, γ) ≈ expected
  @test blocks_match(y, reference)
  for op in (prox!, value_and_prox!)
    check_allocations(op, y, c, x, γ)
  end
end

function check_blocks(::Val{N}) where {N}
  x = ntuple(_ -> [0.5, -2.0], N)
  y = map(similar, x)
  γ = 0.5
  fs = ntuple(k -> isodd(k) ? NormL1(0.5) : SqrNormL2(2.0), N)
  check_proximal_blocks(fs, x, y, γ)
  fallback = ntuple(k -> isodd(k) ? ElasticNet(0.1, 0.2) : ElasticNet(0.3, 0.1), N)
  check_conjugate_blocks(fallback, x, y, γ)
  fs = ntuple(_ -> SqrNormL2(2.0), N)
  check_smooth_blocks(fs, x, y)
end

function check_smooth_blocks(fs, x, y)
  @nospecialize fs x y
  s = SeparableSum(fs)
  expected = sum(map((f, xk) -> f(xk), fs, x))
  @test isequal(value_and_gradient!(y, s, x), expected)
  @test blocks_match(y, map(gradient, fs, x))
  @test gradient!(y, s, x) === y
  @test blocks_match(y, map(gradient, fs, x))
  @test hessian_vec!(y, s, x, x) === y
  @test blocks_match(y, map((f, xk) -> hessian_vec!(similar(xk), f, xk, xk), fs, x))
  q = quadratic_model(s, x)
  @test quadratic_model!(q, s, x) === q
  @test q.fs == fs
  for (op, args) in ((value_and_gradient!, (y, s, x)), (gradient!, (y, s, x)),
      (hessian_vec!, (y, s, x, x)), (quadratic_model!, (q, s, x)))
    check_allocations(op, args...)
  end
  if length(x) in (33, 64)
    @inferred gradient(s, x)
    @inferred quadratic_model(s, x)
  end
end

struct P <: AbstractFloat end
struct Q <: AbstractFloat end
struct R <: AbstractFloat end
Base.promote_rule(::Type{P}, ::Type{Q}) = P
Base.promote_rule(::Type{Q}, ::Type{R}) = Q
Base.promote_rule(::Type{P}, ::Type{R}) = R

struct ConstantValue{T} <: AbstractFunction
  value::T
end
(f::ConstantValue)(x) = f.value
ProximalBase.prox!(y, ::ConstantValue, x, γ::Real) = (copyto!(y, x); y)
ProximalBase.value_and_gradient!(y, f::ConstantValue, x) = (fill!(y, 0); f.value)

const events = Int[]
struct OrderedValue <: Number
  value::Int
end
Base.:+(a::OrderedValue, b::OrderedValue) = (push!(events, -b.value); OrderedValue(a.value + b.value))
struct OrderedFunction <: AbstractFunction
  id::Int
end
(f::OrderedFunction)(x) = (push!(events, f.id); OrderedValue(f.id))
ProximalBase.prox!(y, f::OrderedFunction, x, γ::Real) = (push!(events, 10 + f.id); copyto!(y, x); y)
ProximalBase.value_and_prox!(y, f::OrderedFunction, x, γ::Real) = (prox!(y, f, x, γ); f(x))
ProximalBase.value_and_gradient!(y, f::OrderedFunction, x) = (fill!(y, 0); f(x))

struct MutableModel <: AbstractFunction
  calls::Vector{Int}
end
ProximalBase.quadratic_model!(q::MutableModel, f::MutableModel, x) = (push!(q.calls, length(x)); q)

struct VisitProbe{I} <: AbstractFunction end
ProximalBase._same_buffer_sizes(::VisitProbe{I}, x) where {I} = (push!(events, I); I != 1)
ProximalBase._moreau_nonfinite(::VisitProbe{I}, x, z, γ, always::Bool) where {I} = (push!(events, I); I == 2)
ProximalBase._has_conjugate_value(::VisitProbe{I}) where {I} = (push!(events, I); I != 2)
ProximalBase._conjugate_isindicator(::Type{VisitProbe{I}}) where {I} = (push!(events, I); I != 2)

struct FailingProx <: AbstractFunction end
ProximalBase.prox!(y, ::FailingProx, x, γ::Real) = throw(ArgumentError("block computation failed"))

@testset "straight-line block traversals" begin
  @testset "promotion grouping" begin
    @test PB._valtype(()) === Union{}
    @test PB._valtype((P[],)) === promote_type(P)
    @test PB._valtype((P[], Q[])) === promote_type(P, Q)
    @test PB._valtype((P[], Q[], R[])) === promote_type(P, Q, R)
    @test PB._valtype((P[], Q[], R[])) === (VERSION < v"1.11" ? P : R)
    types = ntuple(k -> (P, Q, R)[mod1(k, 3)], 33)
    @test PB._valtype(map(T -> T[], types)) === promote_type(types...)
  end
  @testset "count $n" for n in (1, 2, 31, 32, 33, 64)
    check_blocks(Val(n))
    for value in (Int8(100), true)
      fs = ntuple(_ -> ConstantValue(value), n)
      x = ntuple(_ -> [1.0], n)
      expected = sum(ntuple(_ -> value, n))
      @test isequal(PB._sum_values(fs, x), expected)
      @test typeof(PB._sum_values(fs, x)) === typeof(expected)
      s = SeparableSum(fs)
      @test isequal(s(x), Float64(expected))
    end
  end

  @testset "nested and mixed blocks" begin
    for T in (Float32, Float64)
      leaf = T[0.5, -2]
      for (s, x) in (
          (SeparableSum(SeparableSum(NormL1(), SqrNormL2()), NormL1()), ((copy(leaf), copy(leaf)), copy(leaf))),
          (SeparableSum(NormL1(), SeparableSum(SqrNormL2(), NormL1())), (copy(leaf), (copy(leaf), copy(leaf)))),
          (SeparableSum(SeparableSum(SqrNormL2(), SqrNormL2(2.0)), SqrNormL2()), ((copy(leaf), copy(leaf)), copy(leaf))),
          (SeparableSum(SqrNormL2(), SeparableSum(SqrNormL2(2.0), SqrNormL2())), (copy(leaf), (copy(leaf), copy(leaf)))))
        y = PB._gradient_buffer(x)
        check_allocations(s, x)
        for f in (s, Conjugate(s)), op in (prox!, value_and_prox!)
          check_allocations(op, y, f, x, T(0.5))
        end
        if issmooth(s)
          for op in (gradient!, value_and_gradient!)
            check_allocations(op, y, s, x)
          end
          check_allocations(hessian_vec!, y, s, x, x)
          check_allocations(quadratic_model!, quadratic_model(s, x), s, x)
        end
      end
    end
    fs = (SqrNormL2(0.5f0), SqrNormL2(2.0))
    x = (Float32[0.5, -2], Float64[0.25, 3])
    s = SeparableSum(fs)
    @test isequal(s(x), sum(map((f, xk) -> f(xk), fs, x)))
    @test typeof(s(x)) === Float64
  end

  @testset "evaluation order and redefinition" begin
    fs = (OrderedFunction(1), OrderedFunction(2), OrderedFunction(3))
    x = ([1.0], [2.0], [3.0])
    y = map(similar, x)
    for (op, args, expected) in (
        (PB._sum_values, (fs, x), [1, 2, 3, -2, -3]),
        (PB._value_and_prox_blocks!, (fs, y, x, 1.0), [11, 1, 12, 2, 13, 3, -2, -3]),
        (PB._value_and_gradient_blocks!, (fs, y, x), [1, 2, 3, -2, -3]))
      empty!(events)
      @test op(args...).value == 6
      @test events == expected
    end
    @eval (f::OrderedFunction)(x) = (push!(events, 100 + f.id); OrderedValue(f.id))
    empty!(events)
    @test Base.invokelatest(PB._sum_values, fs, x).value == 6
    @test events == [101, 102, 103, -2, -3]
  end

  @testset "eager comparisons and short circuits" begin
    probes = (VisitProbe{1}(), VisitProbe{2}(), VisitProbe{3}())
    empty!(events)
    @test !PB._same_buffer_sizes(probes, probes)
    @test events == [1, 2, 3]
    empty!(events)
    @test PB._moreau_nonfinite(probes, probes, probes, 1.0, false)
    @test events == [1, 2]
    empty!(events)
    @test !PB._has_conjugate_value(SeparableSum(probes))
    @test events == [1, 2]
    empty!(events)
    @test !PB._conjugate_isindicator(typeof(SeparableSum(probes)))
    @test events == [1, 2]
  end

  @testset "lengths, exceptions and untouched outputs" begin
    fs = ntuple(_ -> MutableModel(Int[]), 3)
    s = SeparableSum(fs)
    x = ([1.0], [1.0, 2.0], [1.0, 2.0, 3.0])
    for n in (2, 4)
      qs = ntuple(_ -> MutableModel(Int[]), n)
      q = SeparableSum(qs)
      @test quadratic_model!(q, s, x) === q
      @test map(f -> f.calls, qs) == ntuple(k -> k <= 3 ? [k] : Int[], n)
    end
    @test_throws DimensionMismatch("3 functions for 2 blocks") s(x[1:2])
    @test_throws DimensionMismatch("the output has 2 blocks and the input 3") prox!(x[1:2], s, x)
    @test PB._checkgradient_output((), x) === nothing
    @test PB._checkgradient_output(map(copy, x), ()) === nothing
    @test_throws ArgumentError("the gradient must not overwrite x") PB._checkgradient_output((x[1],), x)
    for alias in (false, true), op in (prox!, value_and_prox!)
      s = SeparableSum(SqrNormL2(), FailingProx(), SqrNormL2())
      point = ([3.0], [4.0], [5.0])
      out = alias ? point : ([19.0], [19.0], [19.0])
      before = deepcopy(out)
      @test_throws ArgumentError("block computation failed") op(out, s, point, 1.0)
      @test out[1] == [1.5]
      @test out[2:3] == before[2:3]
    end
    for k in (31, 32, 33), alias in (false, true), op in (prox!, value_and_prox!)
      s = SeparableSum(ntuple(i -> i == k ? NormL1(1.0, ones(2)) : SqrNormL2(), 33))
      point = ntuple(_ -> [3.0], 33)
      out = alias ? point : ntuple(_ -> [19.0], 33)
      before = deepcopy(out)
      @test_throws DimensionMismatch("a parameter array has length 2 and the argument has length 1") op(out, s, point, 0.5)
      @test out == before
    end
  end

  @testset "empty blocks" begin
    s = SeparableSum(())
    @test prox!((), s, (), 1.0) === ()
    error = try
      sum(())
    catch e
      e
    end
    @test error isa ArgumentError
    for (op, args) in ((s, ((),)), (value_and_prox!, ((), s, (), 1.0)),
                       (value_and_gradient!, ((), s, ())))
      @test_throws error op(args...)
    end
    @test PB._valtype(()) === Union{}
  end
end

end
