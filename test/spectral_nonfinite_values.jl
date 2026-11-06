module SpectralNonfiniteValuesTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using JLArrays: JLArray
using ..TestUtils

function cases(T)
  fs = Any[SingularValue(NormL1(one(T))), SingularValue(NormL1(zero(T))),
           SingularValue(IndBallL0(1)), Spectral(NormL1(one(T))),
           Spectral(IndBox(-one(T), one(T))), Spectral(IndNonnegative()),
           IndStiefel(), LogDetLoss(Matrix{T}(I, 2, 2), one(T)),
           Conjugate(SingularValue(NormL1(one(T))))]
  for partial in (false, true)
    append!(fs, (NuclearNorm(one(T); partial), NuclearNorm(zero(T); partial),
                 TracePSD(one(T); partial), TracePSD(zero(T); partial),
                 IndPSD(; partial), IndPSD(; rank = 1, partial),
                 IndPSD(; rank = 2, partial), IndBallRank(1; partial),
                 IndBallRank(2; partial)))
  end
  fs
end

identity_output(f) = f isa NuclearNorm && iszero(f.λ) || f isa IndBallRank && f.r >= 2
# The fused value at a matrix with non-finite entries.
# - The counting functions give the value at their output (O2, #614): IndBallRank with
#   r ≥ minimum(size) copies its input and gives the value of the copy (0 unless an entry is
#   infinite); otherwise the output is all NaN, where the value is NaN, or 0 for SingularValue of
#   IndBallL0 with room for every singular value.
# - A function of the singular values that is the zero function is 0 at every input (E4; ruling 2b
#   of PR #384).
# - The fused value of a conjugate is decided by the conjugate (unchanged), and every other
#   function is NaN at a NaN entry and Inf otherwise (E2).
function fused_nonfinite(f, X, ::Type{T}) where {T}
  p = minimum(size(X))
  f isa IndBallRank && return f.r >= p ? (any(isinf, X) ? T(Inf) : zero(T)) : T(NaN)
  f isa SingularValue && f.f isa IndBallL0 && return f.f.s >= p ? zero(T) : T(NaN)
  zerofun = f isa NuclearNorm && iszero(f.λ) || f isa SingularValue && f.f isa NormL1{<:Real, Nothing} && iszero(f.f.λ0)
  zerofun ? zero(T) : f isa Conjugate || !any(isnan, X) ? T(Inf) : T(NaN)
end

@testset "Input domain and unchanged outputs" begin
  for T in (Float32, Float64), f in cases(T), v in (T(Inf), -T(Inf), T(NaN)),
      index in (CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(2, 1)),
      alias in (false, true)
    input = T[2 0; 0 1]
    input[index] = v
    x = copy(input)
    y = alias ? x : similar(x)
    @test isequal(value_and_prox!(y, f, x, one(T)), fused_nonfinite(f, input, T))
    @test isequal(y, identity_output(f) ? input : fill(T(NaN), 2, 2))
    alias || @test isequal(x, input)
  end
end

@testset "Array types and omitted step" begin
  x = [Inf 0; 0 1]
  @test value_and_prox!(similar(x), NuclearNorm(), x) === Inf
  for T in (Float32, Float64), f in cases(T)
    input = T[Inf 0; 0 1]
    for x in (sparse(input), SymmetricSparseIterate(input), JLArray(input))
      y = x isa JLArray ? similar(x) : zeros(T, 2, 2)
      @test isequal(value_and_prox!(y, f, x, one(T)), fused_nonfinite(f, input, T))
      @test isequal(Array(y), identity_output(f) ? input : fill(T(NaN), 2, 2))
      @test isequal(Matrix(x), input)
    end
    # Device outputs and aliasing use the same input-domain decision.
    x = JLArray(input)
    @test isequal(value_and_prox!(x, f, x, one(T)), fused_nonfinite(f, input, T))
    @test isequal(Array(x), identity_output(f) ? input : fill(T(NaN), 2, 2))
  end
end

@testset "Finite values, history and allocations" begin
  for T in (Float32, Float64), f in cases(T)
    x = T[3 0; 0 1]
    y = similar(x)
    value = value_and_prox!(y, f, x, one(T))
    expected = copy(y)
    # Baseline values for diag(3, 1) with unit step and unit weights.
    reference = if f isa LogDetLoss
      a = one(T) + sqrt(T(2))
      a + one(T) - log(a)
    elseif f isa Union{NuclearNorm, TracePSD}
      iszero(f.λ) ? zero(T) : T(2)
    elseif f isa Union{SingularValue, Spectral} && f.f isa NormL1
      iszero(f.f.λ0) ? zero(T) : T(2)
    else
      zero(T)
    end
    @test value ≈ reference atol = 20eps(T)
    @test value ≈ f(y) atol = 20eps(T)
    @test allocs(value_and_prox!, y, f, x, one(T)) == 0
    bad = T[Inf 0; 0 1]
    @test isequal(value_and_prox!(y, f, bad, one(T)), fused_nonfinite(f, bad, T))
    @test allocs(value_and_prox!, y, f, bad, one(T)) == 0
    @test value_and_prox!(y, f, x, one(T)) === value
    @test isequal(y, expected)
  end
end

@testset "Partial decomposition history" begin
  for T in (Float32, Float64), f in (NuclearNorm(one(T); partial = true),
      TracePSD(one(T); partial = true), IndPSD(; partial = true),
      IndPSD(; rank = 1, partial = true), IndBallRank(1; partial = true))
    x = Matrix(Diagonal(vcat(T(3), fill(-one(T), 63))))
    y = similar(x)
    value_and_prox!(y, f, x, one(T))
    value = value_and_prox!(y, f, x, one(T))
    expected = copy(y)
    history = f.pws
    @test history !== nothing
    basis = hasproperty(history, :Q) ? history.Q : history.Z
    warm, rank, vectors = history.warm, history.rank, copy(basis)
    for v in (T(Inf), -T(Inf), T(NaN)), alias in (false, true)
      bad = copy(x)
      bad[2, 1] = v
      output = alias ? bad : y
      @test isequal(value_and_prox!(output, f, bad, one(T)), fused_nonfinite(f, bad, T))
      @test all(isnan, output)
      @test f.pws === history
      @test (history.warm, history.rank) == (warm, rank)
      @test isequal(basis, vectors)
    end
    @test value_and_prox!(y, f, x, one(T)) ≈ value atol = 100eps(T)
    @test y ≈ expected atol = 100eps(T)
  end
end

@testset "Wrappers and blocks" begin
  for T in (Float32, Float64), f in (NuclearNorm(one(T)), IndPSD(), SingularValue(NormL1(one(T))))
    x = T[Inf 0; 0 1]
    y = similar(x)
    for g in (Scaled(f, T(2)), Translate(f, ones(T, 2, 2)), Precompose(f, T(2)))
      @test value_and_prox!(y, g, x, one(T)) === T(Inf)
      @test all(isnan, y)
    end
    # A finite outer input can overflow before reaching the inner oracle.
    @test value_and_prox!(y, Precompose(f, T(2)), fill(floatmax(T), 2, 2), one(T)) === T(Inf)
    @test all(isnan, y)
    point = (T[3], x)
    output = (zeros(T, 1), similar(x))
    @test value_and_prox!(output, SeparableSum(SqrNormL2(T(2)), f), point, one(T)) === T(Inf)
    @test output[1] == T[1]
    @test all(isnan, output[2])
    slices = cat(Matrix{T}(I, 2, 2), x; dims = 3)
    result = similar(slices)
    @test value_and_prox!(result, SliceSum(SqrNormL2(T(2)), f), slices, one(T)) === T(Inf)
    @test result[:, :, 1] ≈ Matrix{T}(I, 2, 2) / T(3)
    @test all(isnan, result[:, :, 2])
    envelope = MoreauEnvelope(f, T(0.5))
    @test isnan(envelope(x))
    @test isnan(value_and_gradient!(y, envelope, x))
    @test all(isnan, y)
    @test isnan(value_and_prox!(y, envelope, x, one(T)))
    @test all(isnan, y)
    @test isnan(value_and_prox!(y, PointwiseMinimum(f, deepcopy(f)), x, one(T)))
    @test all(isnan, y)
  end
  # The mixed-precision indicator recomputation remains outside this rule.
  x = [Inf 0; 0 1]
  @test value_and_prox!(zeros(Float32, 2, 2), Precompose(IndPSD(), 2), x, 1f0) === 0f0
end

@testset "Conjugate failure boundary" begin
  for T in (Float32, Float64), inner in (IndBallL1(one(T)), ElasticNet(one(T), one(T))),
      v in (T(Inf), -T(Inf), T(NaN)), alias in (false, true)
    f = Conjugate(SingularValue(inner))
    x = T[v 0; 0 1]
    y = alias ? x : fill(T(7), 2, 2)
    before = copy(y)
    @test_throws ArgumentError value_and_prox!(y, f, x, one(T))
    @test isequal(y, before)
  end
end

end
