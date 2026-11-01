module ConformanceFailuresStorageTest

using Test
using ProximalBase
using ..Conformance

# O6-O7/M1-M3: failure atomicity, aliasing, and storage semantics.

# #371 uses an ordinary self-parented AbstractVector with generic storage.
struct Unknown{T} <: AbstractVector{T}
  data::Vector{T}
end
Base.size(x::Unknown) = size(x.data)
Base.IndexStyle(::Type{<:Unknown}) = IndexLinear()
Base.getindex(x::Unknown, i::Int) = x.data[i]
Base.setindex!(x::Unknown, v, i::Int) = (x.data[i] = v)

@row begin
  id = "failures_storage.dimensions.prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 E1"
  problem = "Decision 41 size validation before any caller-owned block is written."
  expected = Throws(:DimensionMismatch)
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; prox!(y, NormL1(1.0, ones(3)), x, 1.0) end
end

@row begin
  id = "failures_storage.dimensions.prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "A late size failure leaves input and every output block unchanged."
  expected = ([2.0, 3.0], [7.0, 8.0])
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; try prox!(y, NormL1(1.0, ones(3)), x, 1.0) catch end; (x, y) end
end

@row begin
  id = "failures_storage.dimensions.value_and_prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 E1"
  problem = "Decision 41 size validation before any caller-owned block is written."
  expected = Throws(:DimensionMismatch)
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; value_and_prox!(y, NormL1(1.0, ones(3)), x, 1.0) end
end

@row begin
  id = "failures_storage.dimensions.value_and_prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "A late size failure leaves input and every output block unchanged."
  expected = ([2.0, 3.0], [7.0, 8.0])
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; try value_and_prox!(y, NormL1(1.0, ones(3)), x, 1.0) catch end; (x, y) end
end

@row begin
  id = "failures_storage.tuple-conjugate.value_and_prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 E1"
  problem = "Decision 41 size validation before any caller-owned block is written."
  expected = Throws(:DimensionMismatch)
  observed = let x = ([2.0, 3.0], [4.0, 5.0]), y = ([7.0, 8.0], [9.0, 10.0]); value_and_prox!(y, Conjugate(SeparableSum(NormL1(1.0), LinearFunction(ones(1, 2)))), x, 1.0) end
end

@row begin
  id = "failures_storage.tuple-conjugate.value_and_prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "A late size failure leaves input and every output block unchanged."
  expected = (([2.0, 3.0], [4.0, 5.0]), ([7.0, 8.0], [9.0, 10.0]))
  observed = let x = ([2.0, 3.0], [4.0, 5.0]), y = ([7.0, 8.0], [9.0, 10.0]); try value_and_prox!(y, Conjugate(SeparableSum(NormL1(1.0), LinearFunction(ones(1, 2)))), x, 1.0) catch end; (x, y) end
end

@row begin
  id = "failures_storage.regularize.prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 E1"
  problem = "Decision 41 size validation before any caller-owned block is written."
  expected = Throws(:DimensionMismatch)
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; prox!(y, Regularize(NormL1(1.0, ones(3)), 1.0), x, 1.0) end
end

@row begin
  id = "failures_storage.regularize.prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "A late size failure leaves input and every output block unchanged."
  expected = ([2.0, 3.0], [7.0, 8.0])
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; try prox!(y, Regularize(NormL1(1.0, ones(3)), 1.0), x, 1.0) catch end; (x, y) end
end

@row begin
  id = "failures_storage.regularize.value_and_prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 E1"
  problem = "Decision 41 size validation before any caller-owned block is written."
  expected = Throws(:DimensionMismatch)
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; value_and_prox!(y, Regularize(NormL1(1.0, ones(3)), 1.0), x, 1.0) end
end

@row begin
  id = "failures_storage.regularize.value_and_prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "A late size failure leaves input and every output block unchanged."
  expected = ([2.0, 3.0], [7.0, 8.0])
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; try value_and_prox!(y, Regularize(NormL1(1.0, ones(3)), 1.0), x, 1.0) catch end; (x, y) end
end

@row begin
  id = "failures_storage.regularize-step.False.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 P1"
  problem = "Dense scalar-step Regularize checks before transformation, also in place."
  expected = Throws(:ArgumentError)
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; prox!(y, Regularize(MoreauEnvelope(NormL1(), 1.0), 0.5), x, -2.0) end
end

@row begin
  id = "failures_storage.regularize-step.False.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6 P1"
  problem = "Dense scalar-step Regularize checks before transformation, also in place."
  expected = [7.0, 8.0]
  observed = let x = [2.0, 3.0], y = [7.0, 8.0]; try prox!(y, Regularize(MoreauEnvelope(NormL1(), 1.0), 0.5), x, -2.0) catch end; y end
end

@row begin
  id = "failures_storage.regularize-step.True.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6 P1"
  problem = "Dense scalar-step Regularize checks before transformation, also in place."
  expected = Throws(:ArgumentError)
  observed = let x = [2.0, 3.0], y = x; prox!(y, Regularize(MoreauEnvelope(NormL1(), 1.0), 0.5), x, -2.0) end
end

@row begin
  id = "failures_storage.regularize-step.True.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6 P1"
  problem = "Dense scalar-step Regularize checks before transformation, also in place."
  expected = [2.0, 3.0]
  observed = let x = [2.0, 3.0], y = x; try prox!(y, Regularize(MoreauEnvelope(NormL1(), 1.0), 0.5), x, -2.0) catch end; y end
end

@row begin
  id = "failures_storage.late-composed.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "Later composed block fails validation before earlier block writes."
  expected = Throws(:DimensionMismatch)
  observed = let x = ([2.0], [3.0]), y = ([7.0], [8.0]); value_and_prox!(y, SeparableSum(NormL1(1.0), Translate(NormL1(1.0), ones(3))), x, 1.0) end
end

@row begin
  id = "failures_storage.late-composed.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "Later composed block fails validation before earlier block writes."
  expected = ([7.0], [8.0])
  observed = let x = ([2.0], [3.0]), y = ([7.0], [8.0]); try value_and_prox!(y, SeparableSum(NormL1(1.0), Translate(NormL1(1.0), ones(3))), x, 1.0) catch end; y end
end

@row begin
  id = "failures_storage.sparse-failure.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "Iterate storage and activation maps survive validation failure."
  expected = Throws(:DimensionMismatch)
  observed = let x = SparseIterate([2.0, 0.0]), y = SparseIterate([0.0, 7.0]); prox!(y, NormL1(1.0, ones(3)), x, 1.0) end
end

@row begin
  id = "failures_storage.sparse-failure.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "Iterate storage and activation maps survive validation failure."
  expected = true
  observed = let x = SparseIterate([2.0, 0.0]), y = SparseIterate([0.0, 7.0]); before = deepcopy(y); input = deepcopy(x); try prox!(y, NormL1(1.0, ones(3)), x, 1.0) catch end; y.nnz == before.nnz && x.nnz == input.nnz && isequal(y.nzval, before.nzval) && isequal(y.nzval2ind, before.nzval2ind) && isequal(y.ind2nzval, before.ind2nzval) && isequal(x.nzval, input.nzval) && isequal(x.nzval2ind, input.nzval2ind) && isequal(x.ind2nzval, input.ind2nzval) end
end

@row begin
  id = "failures_storage.alias.l1.False"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [1.0, 2.0]
  observed = let x = [2.0, 3.0]; prox!(similar(x), NormL1(1.0), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.l1.True"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [1.0, 2.0]
  observed = let x = [2.0, 3.0]; prox!(x, NormL1(1.0), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.ridge.False"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [1.0, 2.0]
  observed = let x = [2.0, 4.0]; prox!(similar(x), SqrNormL2(1.0), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.ridge.True"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [1.0, 2.0]
  observed = let x = [2.0, 4.0]; prox!(x, SqrNormL2(1.0), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.centered-ridge.False"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [1.0, 2.0]
  observed = let x = [0.0, 4.0]; prox!(similar(x), SqrNormL2(1.0, [2.0, 0.0]), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.centered-ridge.True"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [1.0, 2.0]
  observed = let x = [0.0, 4.0]; prox!(x, SqrNormL2(1.0, [2.0, 0.0]), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.weighted-recovery.False"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [pow2(Float64, -54)]
  observed = let x = [pow2(Float64, -52)]; prox!(similar(x), NormL1(pow2(Float64, -538), [pow2(Float64, 1022)]), x, 3 * pow2(Float64, -538)) end
end

@row begin
  id = "failures_storage.alias.weighted-recovery.True"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [pow2(Float64, -54)]
  observed = let x = [pow2(Float64, -52)]; prox!(x, NormL1(pow2(Float64, -538), [pow2(Float64, 1022)]), x, 3 * pow2(Float64, -538)) end
end

@row begin
  id = "failures_storage.alias.ridge-recovery.False"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [pow2(Float64, -700)]
  observed = let x = [pow2(Float64, 900)]; prox!(similar(x), SqrNormL2(pow2(Float64, 800)), x, pow2(Float64, 800)) end
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "failures_storage.alias.ridge-recovery.True"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [pow2(Float64, -700)]
  observed = let x = [pow2(Float64, 900)]; prox!(x, SqrNormL2(pow2(Float64, 800)), x, pow2(Float64, 800)) end
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "failures_storage.alias.translate.False"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [2.0, -2.0]
  observed = let x = [3.0, -3.0]; prox!(similar(x), Translate(NormL1(1.0), [1.0, -1.0]), x, 1.0) end
end

@row begin
  id = "failures_storage.alias.translate.True"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "O7 R2 R3"
  problem = "Aliased and separate outputs checked against the same independent exact reference."
  expected = [2.0, -2.0]
  observed = let x = [3.0, -3.0]; prox!(x, Translate(NormL1(1.0), [1.0, -1.0]), x, 1.0) end
end

@row begin
  id = "failures_storage.gradient-alias.gradient!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O7 O6"
  problem = "Gradient output must be disjoint from input; rejection preserves input."
  expected = Throws(:ArgumentError, r"gradient.*overwrite")
  observed = let x = [2.0, 3.0]; gradient!(x, SqrNormL2(), x) end
end

@row begin
  id = "failures_storage.gradient-alias.gradient!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O7 O6"
  problem = "Gradient output must be disjoint from input; rejection preserves input."
  expected = [2.0, 3.0]
  observed = let x = [2.0, 3.0]; try gradient!(x, SqrNormL2(), x) catch end; x end
end

@row begin
  id = "failures_storage.gradient-alias.value_and_gradient!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O7 O6"
  problem = "Gradient output must be disjoint from input; rejection preserves input."
  expected = Throws(:ArgumentError, r"gradient.*overwrite")
  observed = let x = [2.0, 3.0]; value_and_gradient!(x, SqrNormL2(), x) end
end

@row begin
  id = "failures_storage.gradient-alias.value_and_gradient!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O7 O6"
  problem = "Gradient output must be disjoint from input; rejection preserves input."
  expected = [2.0, 3.0]
  observed = let x = [2.0, 3.0]; try value_and_gradient!(x, SqrNormL2(), x) catch end; x end
end

@row begin
  id = "failures_storage.implicit-zero.Float32.ridge"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "M1"
  problem = "Centered ridge moves unstored zero to a nonzero result."
  expected = Float32[1, 2]
  observed = let x = SparseIterate(Float32[0, 4]); Vector(prox!(similar(x), SqrNormL2(one(Float32), Float32[2, 0]), x, one(Float32))) end
end

@row begin
  id = "failures_storage.implicit-zero.Float32.NegLog"; case = ""; family = :failures_storage; oracle = :value
  types = "Float32"; backend = :cpu; rules = "M1 E6"
  problem = "Unstored zero is outside logarithm domain."
  expected = Float32(Inf)
  observed = NegLog()(SparseIterate(Float32[0, 1]))
end

@row begin
  id = "failures_storage.symmetric.Float32.value"; case = ""; family = :failures_storage; oracle = :value
  types = "Float32"; backend = :cpu; rules = "M2 A6"
  problem = "Off-diagonal square counts twice; A6 reduction budget 8eps(T)*4."
  expected = Float32(4)
  observed = SqrNormL2(one(Float32))(SymmetricSparseIterate(Float32[0 2; 2 0]))
  compare = (a, b) -> typed_bound(a, b, 8eps(Float32) * 4)
end

@row begin
  id = "failures_storage.symmetric.Float32.prox"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "M2 O7"
  problem = "Each off-diagonal stored coordinate is shrunk once."
  expected = Float32[0 1; 1 0]
  observed = let x = SymmetricSparseIterate(Float32[0 2; 2 0]); Matrix(prox!(x, SqrNormL2(one(Float32)), x, one(Float32))) end
end

@row begin
  id = "failures_storage.cache.Float32.fresh"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float32 output; Float64 data"; backend = :cpu; rules = "M3 T8 A1"
  problem = "Cache history independent scalar quadratic solution (2+1)/(1+2); A1 component budget 16eps(D)."
  expected = Float32[1]
  observed = let f = QuadraticFunction(reshape([2.0], 1, 1), [-1.0]);  prox!(zeros(Float32, 1), f, Float32[2], 1.0) end
  compare = (a, b) -> length(a) == 1 && typed_bound(a[1], b[1], 16eps(Float32))
end

@row begin
  id = "failures_storage.cache.Float32.reset"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float32 output; Float64 data"; backend = :cpu; rules = "M3 T8 A1"
  problem = "Cache history independent scalar quadratic solution (2+1)/(1+2); A1 component budget 16eps(D)."
  expected = Float32[1]
  observed = let f = QuadraticFunction(reshape([2.0], 1, 1), [-1.0]); prox!(zeros(1), f, [9.0], 2.0); reset_cache!(f);  prox!(zeros(Float32, 1), f, Float32[2], 1.0) end
  compare = (a, b) -> length(a) == 1 && typed_bound(a[1], b[1], 16eps(Float32))
end

@row begin
  id = "failures_storage.cache.Float32.other-precision"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float32 output; Float64 data"; backend = :cpu; rules = "M3 T8 A1"
  problem = "Cache history independent scalar quadratic solution (2+1)/(1+2); A1 component budget 16eps(D)."
  expected = Float32[1]
  observed = let f = QuadraticFunction(reshape([2.0], 1, 1), [-1.0]); prox!(zeros(Float64, 1), f, [9.0], 2.0);  prox!(zeros(Float32, 1), f, Float32[2], 1.0) end
  compare = (a, b) -> length(a) == 1 && typed_bound(a[1], b[1], 16eps(Float32))
end

@row begin
  id = "failures_storage.implicit-zero.Float64.ridge"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "M1"
  problem = "Centered ridge moves unstored zero to a nonzero result."
  expected = Float64[1, 2]
  observed = let x = SparseIterate(Float64[0, 4]); Vector(prox!(similar(x), SqrNormL2(one(Float64), Float64[2, 0]), x, one(Float64))) end
end

@row begin
  id = "failures_storage.implicit-zero.Float64.NegLog"; case = ""; family = :failures_storage; oracle = :value
  types = "Float64"; backend = :cpu; rules = "M1 E6"
  problem = "Unstored zero is outside logarithm domain."
  expected = Float64(Inf)
  observed = NegLog()(SparseIterate(Float64[0, 1]))
end

@row begin
  id = "failures_storage.symmetric.Float64.value"; case = ""; family = :failures_storage; oracle = :value
  types = "Float64"; backend = :cpu; rules = "M2 A6"
  problem = "Off-diagonal square counts twice; A6 reduction budget 8eps(T)*4."
  expected = Float64(4)
  observed = SqrNormL2(one(Float64))(SymmetricSparseIterate(Float64[0 2; 2 0]))
  compare = (a, b) -> typed_bound(a, b, 8eps(Float64) * 4)
end

@row begin
  id = "failures_storage.symmetric.Float64.prox"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "M2 O7"
  problem = "Each off-diagonal stored coordinate is shrunk once."
  expected = Float64[0 1; 1 0]
  observed = let x = SymmetricSparseIterate(Float64[0 2; 2 0]); Matrix(prox!(x, SqrNormL2(one(Float64)), x, one(Float64))) end
end

@row begin
  id = "failures_storage.cache.Float64.fresh"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64 output; Float64 data"; backend = :cpu; rules = "M3 T8 A1"
  problem = "Cache history independent scalar quadratic solution (2+1)/(1+2); A1 component budget 16eps(D)."
  expected = Float64[1]
  observed = let f = QuadraticFunction(reshape([2.0], 1, 1), [-1.0]);  prox!(zeros(Float64, 1), f, Float64[2], 1.0) end
  compare = (a, b) -> length(a) == 1 && typed_bound(a[1], b[1], 16eps(Float64))
end

@row begin
  id = "failures_storage.cache.Float64.reset"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64 output; Float64 data"; backend = :cpu; rules = "M3 T8 A1"
  problem = "Cache history independent scalar quadratic solution (2+1)/(1+2); A1 component budget 16eps(D)."
  expected = Float64[1]
  observed = let f = QuadraticFunction(reshape([2.0], 1, 1), [-1.0]); prox!(zeros(1), f, [9.0], 2.0); reset_cache!(f);  prox!(zeros(Float64, 1), f, Float64[2], 1.0) end
  compare = (a, b) -> length(a) == 1 && typed_bound(a[1], b[1], 16eps(Float64))
end

@row begin
  id = "failures_storage.cache.Float64.other-precision"; case = ""; family = :failures_storage; oracle = :prox
  types = "Float64 output; Float64 data"; backend = :cpu; rules = "M3 T8 A1"
  problem = "Cache history independent scalar quadratic solution (2+1)/(1+2); A1 component budget 16eps(D)."
  expected = Float64[1]
  observed = let f = QuadraticFunction(reshape([2.0], 1, 1), [-1.0]); prox!(zeros(Float32, 1), f, [9.0], 2.0);  prox!(zeros(Float64, 1), f, Float64[2], 1.0) end
  compare = (a, b) -> length(a) == 1 && typed_bound(a[1], b[1], 16eps(Float64))
end

@row begin
  id = "failures_storage.tuple-conversion.prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64 arguments/first output; Int second output"; backend = :cpu; rules = "T1 O6"
  problem = "#371 fractional integer destination is rejected with ArgumentError before committing any block."
  expected = Throws(:ArgumentError)
  observed = let x = ([2.0], [1.0, 1.0]), y = ([7.0], Unknown(Int[7, 7])); prox!(y, Conjugate(SeparableSum(NormLinf(), NormLinf())), x, 1.0) end
  broken = (group = :G12, baseline = Threw(:InexactError, Regex("InexactError:\\ Int64\\(0\\.5\\)")))
end

@row begin
  id = "failures_storage.tuple-conversion.prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64 first output; Int generic second output"; backend = :cpu; rules = "O6"
  problem = "#371 a later conversion failure must preserve the first tuple block."
  expected = ([7.0], Int[7, 7])
  observed = let x = ([2.0], [1.0, 1.0]), y = ([7.0], Unknown(Int[7, 7])); try prox!(y, Conjugate(SeparableSum(NormLinf(), NormLinf())), x, 1.0) catch end; (y[1], y[2].data) end
  broken = (group = :G12, baseline = Returned(v -> same(v, ([1.0], [7, 7]))))
end

@row begin
  id = "failures_storage.tuple-conversion.value_and_prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64 arguments/first output; Int second output"; backend = :cpu; rules = "T1 O6"
  problem = "#371 fractional integer destination is rejected with ArgumentError before committing any block."
  expected = Throws(:ArgumentError)
  observed = let x = ([2.0], [1.0, 1.0]), y = ([7.0], Unknown(Int[7, 7])); value_and_prox!(y, Conjugate(SeparableSum(NormLinf(), NormLinf())), x, 1.0) end
  broken = (group = :G12, baseline = Threw(:InexactError, Regex("InexactError:\\ Int64\\(0\\.5\\)")))
end

@row begin
  id = "failures_storage.tuple-conversion.value_and_prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64 first output; Int generic second output"; backend = :cpu; rules = "O6"
  problem = "#371 a later conversion failure must preserve the first tuple block."
  expected = ([7.0], Int[7, 7])
  observed = let x = ([2.0], [1.0, 1.0]), y = ([7.0], Unknown(Int[7, 7])); try value_and_prox!(y, Conjugate(SeparableSum(NormLinf(), NormLinf())), x, 1.0) catch end; (y[1], y[2].data) end
  broken = (group = :G12, baseline = Returned(v -> same(v, ([1.0], [7, 7]))))
end

@row begin
  id = "failures_storage.tuple-fallback.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64 generic first argument; Int second output"; backend = :cpu; rules = "T1 O6"
  problem = "#371 tuple Fenchel value fallback rejects fractional integer destination before writes."
  expected = Throws(:ArgumentError)
  observed = let x = (Unknown([3.0, 2.0]), [3.0]), y = ([7.0, 7.0], Int[7]); value_and_prox!(y, Conjugate(SeparableSum(Translate(GroupNormLinf(1.0, [1:2]), zeros(2)), SqrNormL2())), x, 1.0) end
  broken = (group = :G12, baseline = Threw(:InexactError, Regex("InexactError:\\ Int64\\(1\\.5\\)")))
end

@row begin
  id = "failures_storage.tuple-fallback.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64 first output; Int second output"; backend = :cpu; rules = "O6"
  problem = "#371 tuple Fenchel fallback must stage every block before commit."
  expected = ([7.0, 7.0], Int[7])
  observed = let x = (Unknown([3.0, 2.0]), [3.0]), y = ([7.0, 7.0], Int[7]); try value_and_prox!(y, Conjugate(SeparableSum(Translate(GroupNormLinf(1.0, [1:2]), zeros(2)), SqrNormL2())), x, 1.0) catch end; y end
  broken = (group = :G12, baseline = Returned(v -> same(v, ([1.0, 0.0], [7]))))
end

@row begin
  id = "failures_storage.late-regularize.prox!.False.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355/#323 later capped-simplex block has empty feasible set; infeasibility must be named."
  expected = Throws(:ArgumentError, r"[Ii]nfeasible|[Ee]mpty|no point|attain|intersect|misses the domain")
  observed = let x = [2.0 4.0; 3.0 5.0], y = [7.0 9.0; 8.0 10.0]; prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) end
end

@row begin
  id = "failures_storage.late-regularize.prox!.False.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355 dense scalar-step Regularize must not commit an earlier slice when a later search fails."
  expected = [7.0 9.0; 8.0 10.0]
  observed = let x = [2.0 4.0; 3.0 5.0], y = [7.0 9.0; 8.0 10.0]; try prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) catch end; y end
end

@row begin
  id = "failures_storage.late-regularize.value_and_prox!.False.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355/#323 later capped-simplex block has empty feasible set; infeasibility must be named."
  expected = Throws(:ArgumentError, r"[Ii]nfeasible|[Ee]mpty|no point|attain|intersect|misses the domain")
  observed = let x = [2.0 4.0; 3.0 5.0], y = [7.0 9.0; 8.0 10.0]; value_and_prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) end
end

@row begin
  id = "failures_storage.late-regularize.value_and_prox!.False.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355 dense scalar-step Regularize must not commit an earlier slice when a later search fails."
  expected = [7.0 9.0; 8.0 10.0]
  observed = let x = [2.0 4.0; 3.0 5.0], y = [7.0 9.0; 8.0 10.0]; try value_and_prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) catch end; y end
end

@row begin
  id = "failures_storage.late-regularize.prox!.True.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355/#323 later capped-simplex block has empty feasible set; infeasibility must be named."
  expected = Throws(:ArgumentError, r"[Ii]nfeasible|[Ee]mpty|no point|attain|intersect|misses the domain")
  observed = let x = [2.0 4.0; 3.0 5.0], y = x; prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) end
end

@row begin
  id = "failures_storage.late-regularize.prox!.True.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355 dense scalar-step Regularize must not commit an earlier slice when a later search fails."
  expected = [2.0 4.0; 3.0 5.0]
  observed = let x = [2.0 4.0; 3.0 5.0], y = x; try prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) catch end; y end
end

@row begin
  id = "failures_storage.late-regularize.value_and_prox!.True.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355/#323 later capped-simplex block has empty feasible set; infeasibility must be named."
  expected = Throws(:ArgumentError, r"[Ii]nfeasible|[Ee]mpty|no point|attain|intersect|misses the domain")
  observed = let x = [2.0 4.0; 3.0 5.0], y = x; value_and_prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) end
end

@row begin
  id = "failures_storage.late-regularize.value_and_prox!.True.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "O6"
  problem = "#355 dense scalar-step Regularize must not commit an earlier slice when a later search fails."
  expected = [2.0 4.0; 3.0 5.0]
  observed = let x = [2.0 4.0; 3.0 5.0], y = x; try value_and_prox!(y, Regularize(SliceSum(NormL1(), Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 3.0))), 0.5), x, 1.0) catch end; y end
end

@row begin
  id = "failures_storage.dense-regularize.prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64 argument; Int destination"; backend = :cpu; rules = "T1 O6"
  problem = "#355 dense Regularize: exact first entry 1, fractional second entry 1/2; reject destination before writing."
  expected = Throws(:ArgumentError)
  observed = let y = Int[7, 7]; prox!(y, Regularize(NormL1(), 1.0), [3.0, 2.0], 1.0) end
  broken = (group = :G9, baseline = Threw(:InexactError, Regex("InexactError:\\ Int64\\(1\\.5\\)")))
end

@row begin
  id = "failures_storage.dense-regularize.prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64 argument; Int destination"; backend = :cpu; rules = "T1 O6"
  problem = "#355 dense Regularize: exact first entry 1, fractional second entry 1/2; reject destination before writing."
  expected = Int[7, 7]
  observed = let y = Int[7, 7]; try prox!(y, Regularize(NormL1(), 1.0), [3.0, 2.0], 1.0) catch end; y end
end

@row begin
  id = "failures_storage.dense-regularize.value_and_prox!.exception"; case = ""; family = :failures_storage; oracle = :validation
  types = "Float64 argument; Int destination"; backend = :cpu; rules = "T1 O6"
  problem = "#355 dense Regularize: exact first entry 1, fractional second entry 1/2; reject destination before writing."
  expected = Throws(:ArgumentError)
  observed = let y = Int[7, 7]; value_and_prox!(y, Regularize(NormL1(), 1.0), [3.0, 2.0], 1.0) end
  broken = (group = :G9, baseline = Threw(:InexactError, Regex("InexactError:\\ Int64\\(1\\.5\\)")))
end

@row begin
  id = "failures_storage.dense-regularize.value_and_prox!.unchanged"; case = ""; family = :failures_storage; oracle = :storage
  types = "Float64 argument; Int destination"; backend = :cpu; rules = "T1 O6"
  problem = "#355 dense Regularize: exact first entry 1, fractional second entry 1/2; reject destination before writing."
  expected = Int[7, 7]
  observed = let y = Int[7, 7]; try value_and_prox!(y, Regularize(NormL1(), 1.0), [3.0, 2.0], 1.0) catch end; y end
end

end
