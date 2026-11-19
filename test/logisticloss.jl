using Test, ProximalBase, LinearAlgebra, SparseArrays, OffsetArrays
using .TestUtils: allocations

const PBLogistic = ProximalBase
const logistic_value = PBLogistic.value_from_predictor
const logistic_fused! = PBLogistic.value_and_gradient_from_predictor!

# Enough bits to resolve log1p(exp(t)) - t and the sigmoid minus one even
# for the positive predictors in the weighted-underflow cases.
function logistic_reference(X, η, y, w)
  setprecision(BigFloat, 4096) do
    p = big.(η)
    Y, W = big.(y), big.(w)
    v = sum(W .* (log1p.(exp.(p)) .- Y .* p))
    d = W .* (inv.(1 .+ exp.(-p)) .- Y)
    v, d, transpose(big.(Matrix(X))) * d
  end
end

@testset "Logistic regression ($T)" for T in (Float32, Float64)
  X = T[1 2; 1 0; 1 -1; 1 3]
  y, w, offset = T[0, 1, 0.3, 1], T[0.5, 2, 1.25, -0.0], T[0.125, -0.5, 1, 0.25]
  β = T[0.75, -0.25]
  η = X * β + offset
  for design in (X, sparse(X), view(hcat(X, X), :, 1:2)),
      (weights, off) in ((nothing, nothing), (w, nothing), (nothing, offset), (w, offset))
    f = LogisticLoss(y, design; weights, offset=off)
    @test f.y === y && f.X === design && f.weights === weights && f.offset === off
    @test issmooth(f) && isconvex(f) && !isproximable(f) && !isseparable(f)
    z = off === nothing ? X * β : η
    vref, dref, gref = logistic_reference(design, z, y, weights === nothing ? ones(T, 4) : weights)
    @test (@inferred f(β)) ≈ T(vref) rtol=8eps(T)
    @test (@inferred gradient(f, β)) ≈ T.(gref) rtol=8eps(T)
    v, g = @inferred value_and_gradient(f, β)
    @test v == f(β)
    @test g == gradient(f, β)

    d = fill(T(NaN), 4)
    zcopy = copy(z)
    saved = copy(f.tmp)
    @test (@inferred logistic_value(f, z)) == v
    @test (@inferred logistic_fused!(g, d, f, z)) == v
    @test d ≈ T.(dref) rtol=8eps(T)
    @test g ≈ T.(gref) rtol=8eps(T)
    @test isequal(f.tmp, saved) && isequal(z, zcopy)

    h = cbrt(eps(T))
    for j in eachindex(β)
      step = zeros(T, 2)
      step[j] = h
      @test g[j] ≈ (f(β + step) - f(β - step)) / (2h) atol=12h^2
    end

    s = T(1.75) * f
    @test s(β) == T(1.75) * v
    @test gradient(s, β) == T(1.75) .* g
    @test value_and_gradient(s, β) == (T(1.75) * v, T(1.75) .* g)
    @test_throws MethodError logistic_value(s, z)
    @test_throws MethodError logistic_fused!(g, d, s, z)

    @test allocations(f, β) == 0
    @test allocations(gradient!, g, f, β) == 0
    @test allocations(value_and_gradient!, g, f, β) == 0
    @test allocations(logistic_value, f, z) == 0
    @test allocations(logistic_fused!, g, d, f, z) == 0
  end

  @testset "Borrowed data and strided arguments" begin
    Xcopy, ycopy, wcopy, ocopy = copy(X), copy(y), copy(w), copy(offset)
    f = LogisticLoss(ycopy, Xcopy; weights=wcopy, offset=ocopy)
    Xcopy[1] += one(T)
    ycopy[1] = T(0.125)
    wcopy[2] = T(0.5)
    ocopy[3] = T(2)
    z = Xcopy * β + ocopy
    ηview = view(repeat(z; inner=2), 1:2:8)
    βview = view(repeat(β; inner=2), 1:2:4)
    g = view(fill(T(NaN), 4), 1:2:4)
    d = view(fill(T(NaN), 8), 1:2:8)
    v, dr, gr = logistic_reference(Xcopy, z, ycopy, wcopy)
    @test f(βview) ≈ T(v) rtol=8eps(T)
    @test logistic_fused!(g, d, f, ηview) ≈ T(v) rtol=8eps(T)
    @test d ≈ T.(dr) rtol=8eps(T)
    @test g ≈ T.(gr) rtol=8eps(T)
  end

  @testset "Tail, overflow and IEEE cases" begin
    for a in (T === Float32 ? (T(20), T(95), T(120), T(180)) : (T(40), T(720), T(800), T(1000))),
        sign in (-one(T), one(T))
      z = [sign * a]
      ytail = [sign > 0 ? one(T) : zero(T)]
      wt = [a <= T(40) ? one(T) : floatmax(T)]
      M = ones(T, 1, 1)
      f = LogisticLoss(ytail, M; weights=wt)
      vr, dr, _ = logistic_reference(M, z, ytail, wt)
      v, g = @inferred value_and_gradient(f, z)
      @test v > 0
      @test v ≈ T(vr) rtol=8eps(T)
      @test g[1] ≈ T(only(dr)) rtol=8eps(T)
      @test gradient(f, z) == g
      @test logistic_value(f, z) == v
    end
    for (z, response, weight, value, slope) in (
        (Inf, 1, 1, 0, 0), (-Inf, 0, 1, 0, 0),
        (Inf, 0.25, 2, Inf, 1.5), (-Inf, 0.25, 2, Inf, -0.5),
        (NaN, 0.25, 1, NaN, NaN), (NaN, 0.25, -0.0, 0, 0),
        (Inf, 0, 0, 0, 0), (-Inf, 1, 0, 0, 0))
      f = LogisticLoss(T[response], ones(T, 1, 1); weights=T[weight])
      g, d = zeros(T, 1), zeros(T, 1)
      @test isequal(logistic_fused!(g, d, f, T[z]), T(value))
      @test abs(d[1]) == abs(T(slope)) || (isnan(d[1]) && isnan(slope))
      @test iszero(slope) || isequal(d[1], T(slope))
      @test iszero(slope) ? iszero(g[1]) : isequal(g[1], T(slope))
      @test isequal(f(T[z]), T(value))
    end
    M = floatmax(T)
    f = LogisticLoss(T[0.25], ones(T, 1, 1))
    @test f(T[M]) == T(0.75) * M
    f = LogisticLoss(T[0], ones(T, 1, 1); weights=T[2])
    @test value_and_gradient(f, T[M]) == (T(Inf), T[2])
    f = LogisticLoss(T[1], fill(M, 1, 1))
    @test value_and_gradient(f, T[2]) == (zero(T), T[0])
    @test value_and_gradient(f, T[M]) == (zero(T), T[0])
    f = LogisticLoss(T[0], ones(T, 1, 1))
    @test value_and_gradient(f, T[-2000]) == (zero(T), T[0])

    # Masking applies before weighting, not inside the transposed product.
    f = LogisticLoss(T[1, 0], T[Inf; 1;;]; weights=T[0, 1])
    g, d = zeros(T, 1), zeros(T, 2)
    @test logistic_fused!(g, d, f, T[Inf, 0]) == log(T(2))
    @test d == T[0, 0.5]
    @test isnan(g[1])
  end

  @testset "Empty shapes" for (n, p) in ((0, 3), (2, 0), (0, 0)), design in (zeros(T, n, p), spzeros(T, n, p))
    f = LogisticLoss(zeros(T, n), design; offset=ones(T, n))
    empty_β, eta = zeros(T, p), ones(T, n)
    v = f(empty_β)
    @test v ≈ T(n * log1p(exp(1)))
    @test value_and_gradient(f, empty_β) == (v, zeros(T, p))
    d = zeros(T, n)
    g = ones(T, p)
    @test logistic_fused!(g, d, f, eta) == v
    @test iszero(g)
    @test d ≈ fill(T(inv(1 + exp(-1))), n)
  end

  @testset "Validation before writes" begin
    @test_throws DimensionMismatch LogisticLoss(ones(T, 3), X)
    @test_throws DimensionMismatch LogisticLoss(y, X; weights=T[1])
    @test_throws DimensionMismatch LogisticLoss(y, X; offset=T[1])
    @test_throws ArgumentError LogisticLoss(y, X; weights=ones(Int, 4))
    @test_throws ArgumentError LogisticLoss(y, X; offset=ones(Int, 4))
    @test_throws ArgumentError LogisticLoss(y, X; weights=T[1, -1, 0, 1])
    @test_throws ArgumentError LogisticLoss(y, X; weights=T[1, Inf, 0, 1])
    @test_throws ArgumentError LogisticLoss(y, X; weights=T[1, NaN, 0, 1])
    @test_throws ArgumentError LogisticLoss(y, X; offset=T[1, Inf, 0, 1])
    @test_throws ArgumentError LogisticLoss(y, X; offset=T[1, NaN, 0, 1])
    @test_throws MethodError LogisticLoss(complex.(y), X)
    @test_throws MethodError LogisticLoss(ones(Int, 4), X)
    @test_throws MethodError LogisticLoss(y, Matrix{Int}(X))
    @test_throws Union{ArgumentError,MethodError} LogisticLoss(OffsetArray(y, 0:3), X)
    @test_throws Union{ArgumentError,MethodError} LogisticLoss(y, OffsetArray(X, 0:3, 1:2))
    for bad in (NaN, -Inf, Inf, -0.25, 1.25)
      @test_throws ArgumentError LogisticLoss(T[1, 0, 0.5, bad], X; weights=w)
    end

    f = LogisticLoss(y, X; weights=w, offset)
    g, d, scratch = fill(T(2), 2), fill(T(3), 4), fill(T(7), 4)
    copyto!(f.tmp, scratch)
    @test_throws DimensionMismatch f(T[1])
    @test_throws DimensionMismatch (T(2)*f)(T[1])
    @test_throws ArgumentError (T(2)*f)(view(f.tmp, 1:2))
    @test_throws DimensionMismatch gradient!(g, f, T[1])
    @test_throws DimensionMismatch value_and_gradient!(d, f, β)
    @test_throws MethodError value_and_gradient!(Int[1, 2], f, β)
    @test_throws DimensionMismatch logistic_value(f, β)
    @test_throws DimensionMismatch logistic_fused!(d, d, f, η)
    @test_throws DimensionMismatch logistic_fused!(g, g, f, η)
    @test_throws MethodError logistic_fused!(g, d, f, Int[1, 2, 3, 4])
    @test_throws MethodError logistic_fused!(Int[1, 2], d, f, η)
    @test_throws ArgumentError logistic_value(f, f.tmp)
    @test_throws ArgumentError logistic_fused!(g, d, f, f.tmp)
    @test_throws ArgumentError logistic_fused!(g, η, f, η)
    @test_throws ArgumentError logistic_fused!(view(d, 2:3), d, f, η)
    @test_throws ArgumentError logistic_fused!(view(η, 1:2), d, f, η)
    @test_throws ArgumentError logistic_fused!(g, view(X, :, 1), f, η)
    @test_throws ArgumentError logistic_fused!(g, f.tmp, f, η)
    @test_throws ArgumentError logistic_fused!(g, y, f, η)
    @test_throws ArgumentError logistic_fused!(g, w, f, η)
    @test_throws ArgumentError logistic_fused!(g, offset, f, η)
    @test_throws ArgumentError gradient!(β, f, β)
    @test_throws ArgumentError gradient!(view(y, 1:2), f, β)
    @test_throws ArgumentError gradient!(view(f.tmp, 1:2), f, β)
    @test_throws ArgumentError value_and_gradient!(g, f, view(f.tmp, 1:2))
    @test_throws ArgumentError value_and_gradient!(view(offset, 1:2), T(2)*f, β)
    @test g == fill(T(2), 2) && d == fill(T(3), 4) && f.tmp == scratch
    @test y == T[0, 1, 0.3, 1] && w == T[0.5, 2, 1.25, -0.0]
    @test X == T[1 2; 1 0; 1 -1; 1 3] && η == X * β + offset
    @test offset == T[0.125, -0.5, 1, 0.25] && β == T[0.75, -0.25]

    S = sparse(X)
    fs = LogisticLoss(y, S)
    oldvalues = copy(nonzeros(S))
    @test_throws ArgumentError logistic_fused!(view(nonzeros(S), 1:2), d, fs, η)
    @test_throws ArgumentError logistic_fused!(g, view(nonzeros(S), 1:4), fs, η)
    @test nonzeros(S) == oldvalues && g == fill(T(2), 2) && d == fill(T(3), 4)
  end
end

# Include the same implementation under a local product counter. This inspects
# dense and CSC products without extending LinearAlgebra.mul! on foreign types.
module CountLogisticProducts
using LinearAlgebra: transpose
using SparseArrays
using ProximalBase: AbstractFunction, _check_one_based, _ls_checklength, _ls_overlap, _ls_alias_error
import ProximalBase: issmooth, isconvex
const forward = Ref(0)
const backward = Ref(0)
function mul!(dest, X, src)
  count = X isa LinearAlgebra.Transpose ? backward : forward
  count[] += 1
  LinearAlgebra.mul!(dest, X, src)
end
import LinearAlgebra
function gradient! end
function value_and_gradient! end
include(joinpath(@__DIR__, "..", "src", "functions", "logisticloss.jl"))
end

@testset "Logistic predictor product counts" begin
  C = CountLogisticProducts
  for X in ([1.0 2.0; 3.0 4.0], sparse([1.0 2.0; 3.0 4.0]))
    C.forward[] = C.backward[] = 0
    f = C.LogisticLoss([1.0, 0.0], X; offset=[0.5, -1.0])
    β = [0.25, 0.5]
    z = X * β + f.offset
    g, d = zeros(2), zeros(2)
    fill!(f.tmp, NaN)
    C.value_from_predictor(f, z)
    @test (C.forward[], C.backward[]) == (0, 0)
    v = C.value_and_gradient_from_predictor!(g, d, f, z)
    @test (C.forward[], C.backward[]) == (0, 1)
    @test all(isnan, f.tmp)
    @test C.value_and_gradient!(g, f, β) == v
    @test (C.forward[], C.backward[]) == (1, 2)
  end
end
