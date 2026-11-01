module ConformanceStepsTest

using Test
using ProximalBase
using ..Conformance

# P1-P3/O6: endpoints, original step types, and rejection before writes.

@row begin
  id = "S7.exception"; case = "S7"; family = :steps; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Array metric of a coupling set must be positive and finite."
  expected = Throws(:ArgumentError)
  observed = prox!(fill(7.0, 2), IndSimplex(1.0), [2.0, 3.0], zeros(2))
end

@row begin
  id = "S7.unchanged"; case = "S7"; family = :steps; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Array metric of a coupling set must be positive and finite."
  expected = [7.0, 7.0]
  observed = let y = fill(7.0, 2); try prox!(y, IndSimplex(1.0), [2.0, 3.0], zeros(2)) catch end; y end
end

@row begin
  id = "S8.separate"; case = "S8"; family = :steps; oracle = :prox
  types = "Float32 argument/destination; Float64 level/step"; backend = :cpu; rules = "T3 A2 O7"
  problem = "Converted threshold equals one; both routes select zero independently."
  expected = Float32[0]
  observed = prox!(zeros(Float32, 1), NormL1(1 - pow2(Float64, -30)), Float32[1], 1.0)
end

@row begin
  id = "S8.aliased"; case = "S8"; family = :steps; oracle = :prox
  types = "Float32 argument/destination; Float64 level/step"; backend = :cpu; rules = "T3 A2 O7"
  problem = "Converted threshold equals one; both routes select zero independently."
  expected = Float32[0]
  observed = let x = Float32[1]; prox!(x, NormL1(1 - pow2(Float64, -30)), x, 1.0) end
end

@row begin
  id = "S10.negative.exception"; case = "S10"; family = :steps; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Negative or NaN steps fail before output writes."
  expected = Throws(:ArgumentError)
  observed = prox!(fill(7.0, 2), NormL1(1.0), [2.0, 3.0], -1.0)
  broken = (group = :G3, baseline = Returned(v -> same(v, [3.0, 4.0])))
end

@row begin
  id = "S10.negative.unchanged"; case = "S10"; family = :steps; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Negative or NaN steps fail before output writes."
  expected = [7.0, 7.0]
  observed = let y = fill(7.0, 2); try prox!(y, NormL1(1.0), [2.0, 3.0], -1.0) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, [3.0, 4.0])))
end

@row begin
  id = "S10.nan.exception"; case = "S10"; family = :steps; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Negative or NaN steps fail before output writes."
  expected = Throws(:ArgumentError)
  observed = prox!(fill(7.0, 2), NormL1(1.0), [2.0, 3.0], NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, [NaN, NaN])))
end

@row begin
  id = "S10.nan.unchanged"; case = "S10"; family = :steps; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Negative or NaN steps fail before output writes."
  expected = [7.0, 7.0]
  observed = let y = fill(7.0, 2); try prox!(y, NormL1(1.0), [2.0, 3.0], NaN) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, [NaN, NaN])))
end

@row begin
  id = "steps.NormL1.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[-2, 3]
  observed = prox(NormL1(one(Float32)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.NormL1.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[0, 0]
  observed = prox(NormL1(one(Float32)), Float32[-2, 3], Float32(Inf))
end

@row begin
  id = "steps.ElasticNet.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[-2, 3]
  observed = prox(ElasticNet(one(Float32), one(Float32)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.ElasticNet.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[0, 0]
  observed = prox(ElasticNet(one(Float32), one(Float32)), Float32[-2, 3], Float32(Inf))
end

@row begin
  id = "steps.Berhu.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[-2, 3]
  observed = prox(Berhu(one(Float32), one(Float32)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.Berhu.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[0, 0]
  observed = prox(Berhu(one(Float32), one(Float32)), Float32[-2, 3], Float32(Inf))
end

@row begin
  id = "steps.ridge.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[-2, 3]
  observed = prox(SqrNormL2(one(Float32), Float32[1, -1]), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.ridge.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[1, -1]
  observed = prox(SqrNormL2(one(Float32), Float32[1, -1]), Float32[-2, 3], Float32(Inf))
end

@row begin
  id = "steps.HingeLoss.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[-2, 3]
  observed = prox(HingeLoss(), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.HingeLoss.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float32[1, 3]
  observed = prox(HingeLoss(), Float32[-2, 3], Float32(Inf))
end

@row begin
  id = "steps.MCP.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Full-domain nonconvex zero-step endpoint is identity."
  expected = Float32[-2, 3]
  observed = prox(MCP(one(Float32), Float32(3)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.SCAD.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Full-domain nonconvex zero-step endpoint is identity."
  expected = Float32[-2, 3]
  observed = prox(SCAD(one(Float32), Float32(3)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.NormL0.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Full-domain nonconvex zero-step endpoint is identity."
  expected = Float32[-2, 3]
  observed = prox(NormL0(one(Float32)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.NegLog.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Zero step projects onto closure of positive half-line."
  expected = Float32[0, 2]
  observed = prox(NegLog(), Float32[-1, 2], zero(Float32))
end

@row begin
  id = "steps.NegLog.Float32.infinite"; case = ""; family = :steps; oracle = :validation
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "NegLog has no finite minimizer; infinite step rejected."
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Float32, 1), NegLog(), Float32[0], Float32(Inf))
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[Inf])))
end

@row begin
  id = "steps.NegLog.Float32.infinite-unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float32"; backend = :cpu; rules = "P1 O6"
  problem = "Unsupported endpoint leaves output unchanged."
  expected = Float32[7]
  observed = let y = Float32[7]; try prox!(y, NegLog(), Float32[0], Float32(Inf)) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[Inf])))
end

@row begin
  id = "steps.IndBox.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float32[0, 1]
  observed = prox(IndBox(zero(Float32), one(Float32)), Float32[-2, 3], zero(Float32))
end

@row begin
  id = "steps.IndBox.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float32[0, 1]
  observed = prox(IndBox(zero(Float32), one(Float32)), Float32[-2, 3], Float32(Inf))
end

@row begin
  id = "steps.IndSimplex.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float32[1, 0]
  observed = prox(IndSimplex(one(Float32)), Float32[2, 0], zero(Float32))
end

@row begin
  id = "steps.IndSimplex.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float32[1, 0]
  observed = prox(IndSimplex(one(Float32)), Float32[2, 0], Float32(Inf))
end

@row begin
  id = "steps.IndHyperplane.Float32.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float32[1, 0]
  observed = prox(IndHyperplane(Float32[1, 1], one(Float32)), Float32[2, 1], zero(Float32))
end

@row begin
  id = "steps.IndHyperplane.Float32.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float32[1, 0]
  observed = prox(IndHyperplane(Float32[1, 1], one(Float32)), Float32[2, 1], Float32(Inf))
end

@row begin
  id = "steps.routes.Float32.negative.prox"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Float32, 1), NormL1(one(Float32)), Float32[2], -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[2.0])))
end

@row begin
  id = "steps.routes.Float32.negative.joint"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = value_and_prox!(zeros(Float32, 1), NormL1(one(Float32)), Float32[2], -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, 2.0f0)))
end

@row begin
  id = "steps.routes.Float32.negative.entry"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox_entry(NormL1(one(Float32)), Float32(2), 1, -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, 2.0f0)))
end

@row begin
  id = "steps.routes.Float32.negative.block"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!((Float32[7], Float32[7]), SeparableSum(NormL1(one(Float32)), NormL1(one(Float32))), (Float32[2], Float32[3]), -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, (Float32[2.0], Float32[3.0]))))
end

@row begin
  id = "steps.routes.Float32.nan.prox"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Float32, 1), NormL1(one(Float32)), Float32[2], NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[NaN])))
end

@row begin
  id = "steps.routes.Float32.nan.joint"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = value_and_prox!(zeros(Float32, 1), NormL1(one(Float32)), Float32[2], NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, NaN32)))
end

@row begin
  id = "steps.routes.Float32.nan.entry"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox_entry(NormL1(one(Float32)), Float32(2), 1, NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, NaN32)))
end

@row begin
  id = "steps.routes.Float32.nan.block"; case = ""; family = :steps; oracle = :validation
  types = "Float32 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!((Float32[7], Float32[7]), SeparableSum(NormL1(one(Float32)), NormL1(one(Float32))), (Float32[2], Float32[3]), NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, (Float32[NaN], Float32[NaN]))))
end

@row begin
  id = "steps.NormL1.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[-2, 3]
  observed = prox(NormL1(one(Float64)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.NormL1.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[0, 0]
  observed = prox(NormL1(one(Float64)), Float64[-2, 3], Float64(Inf))
end

@row begin
  id = "steps.ElasticNet.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[-2, 3]
  observed = prox(ElasticNet(one(Float64), one(Float64)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.ElasticNet.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[0, 0]
  observed = prox(ElasticNet(one(Float64), one(Float64)), Float64[-2, 3], Float64(Inf))
end

@row begin
  id = "steps.Berhu.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[-2, 3]
  observed = prox(Berhu(one(Float64), one(Float64)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.Berhu.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[0, 0]
  observed = prox(Berhu(one(Float64), one(Float64)), Float64[-2, 3], Float64(Inf))
end

@row begin
  id = "steps.ridge.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[-2, 3]
  observed = prox(SqrNormL2(one(Float64), Float64[1, -1]), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.ridge.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[1, -1]
  observed = prox(SqrNormL2(one(Float64), Float64[1, -1]), Float64[-2, 3], Float64(Inf))
end

@row begin
  id = "steps.HingeLoss.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[-2, 3]
  observed = prox(HingeLoss(), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.HingeLoss.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Named separable endpoint: domain closure at zero; minimizer set at infinity."
  expected = Float64[1, 3]
  observed = prox(HingeLoss(), Float64[-2, 3], Float64(Inf))
end

@row begin
  id = "steps.MCP.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Full-domain nonconvex zero-step endpoint is identity."
  expected = Float64[-2, 3]
  observed = prox(MCP(one(Float64), Float64(3)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.SCAD.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Full-domain nonconvex zero-step endpoint is identity."
  expected = Float64[-2, 3]
  observed = prox(SCAD(one(Float64), Float64(3)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.NormL0.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Full-domain nonconvex zero-step endpoint is identity."
  expected = Float64[-2, 3]
  observed = prox(NormL0(one(Float64)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.NegLog.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Zero step projects onto closure of positive half-line."
  expected = Float64[0, 2]
  observed = prox(NegLog(), Float64[-1, 2], zero(Float64))
end

@row begin
  id = "steps.NegLog.Float64.infinite"; case = ""; family = :steps; oracle = :validation
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "NegLog has no finite minimizer; infinite step rejected."
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Float64, 1), NegLog(), Float64[0], Float64(Inf))
  broken = (group = :G3, baseline = Returned(v -> same(v, [Inf])))
end

@row begin
  id = "steps.NegLog.Float64.infinite-unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float64"; backend = :cpu; rules = "P1 O6"
  problem = "Unsupported endpoint leaves output unchanged."
  expected = Float64[7]
  observed = let y = Float64[7]; try prox!(y, NegLog(), Float64[0], Float64(Inf)) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, [Inf])))
end

@row begin
  id = "steps.IndBox.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float64[0, 1]
  observed = prox(IndBox(zero(Float64), one(Float64)), Float64[-2, 3], zero(Float64))
end

@row begin
  id = "steps.IndBox.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float64[0, 1]
  observed = prox(IndBox(zero(Float64), one(Float64)), Float64[-2, 3], Float64(Inf))
end

@row begin
  id = "steps.IndSimplex.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float64[1, 0]
  observed = prox(IndSimplex(one(Float64)), Float64[2, 0], zero(Float64))
end

@row begin
  id = "steps.IndSimplex.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float64[1, 0]
  observed = prox(IndSimplex(one(Float64)), Float64[2, 0], Float64(Inf))
end

@row begin
  id = "steps.IndHyperplane.Float64.zero"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float64[1, 0]
  observed = prox(IndHyperplane(Float64[1, 1], one(Float64)), Float64[2, 1], zero(Float64))
end

@row begin
  id = "steps.IndHyperplane.Float64.infinite"; case = ""; family = :steps; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "P1"
  problem = "Scalar indicator endpoints retain projection; exact dyadic fixture."
  expected = Float64[1, 0]
  observed = prox(IndHyperplane(Float64[1, 1], one(Float64)), Float64[2, 1], Float64(Inf))
end

@row begin
  id = "steps.routes.Float64.negative.prox"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Float64, 1), NormL1(one(Float64)), Float64[2], -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, [2.0])))
end

@row begin
  id = "steps.routes.Float64.negative.joint"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = value_and_prox!(zeros(Float64, 1), NormL1(one(Float64)), Float64[2], -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, 2.0)))
end

@row begin
  id = "steps.routes.Float64.negative.entry"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox_entry(NormL1(one(Float64)), Float64(2), 1, -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, 2.0)))
end

@row begin
  id = "steps.routes.Float64.negative.block"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!((Float64[7], Float64[7]), SeparableSum(NormL1(one(Float64)), NormL1(one(Float64))), (Float64[2], Float64[3]), -pow2(Float64, -160))
  broken = (group = :G3, baseline = Returned(v -> same(v, ([2.0], [3.0]))))
end

@row begin
  id = "steps.routes.Float64.nan.prox"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Float64, 1), NormL1(one(Float64)), Float64[2], NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, [NaN])))
end

@row begin
  id = "steps.routes.Float64.nan.joint"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = value_and_prox!(zeros(Float64, 1), NormL1(one(Float64)), Float64[2], NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, NaN)))
end

@row begin
  id = "steps.routes.Float64.nan.entry"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox_entry(NormL1(one(Float64)), Float64(2), 1, NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, NaN)))
end

@row begin
  id = "steps.routes.Float64.nan.block"; case = ""; family = :steps; oracle = :validation
  types = "Float64 argument/destination/level; Float64 step"; backend = :cpu; rules = "P1 P2"
  problem = "Validate negative/NaN step in its given type on every route."
  expected = Throws(:ArgumentError)
  observed = prox!((Float64[7], Float64[7]), SeparableSum(NormL1(one(Float64)), NormL1(one(Float64))), (Float64[2], Float64[3]), NaN)
  broken = (group = :G3, baseline = Returned(v -> same(v, ([NaN], [NaN]))))
end

@row begin
  id = "steps.routes.positive.prox"; case = ""; family = :steps; oracle = :prox
  types = "Float32 argument/destination; Float64 step/level"; backend = :cpu; rules = "P2 R3"
  problem = "Tiny positive original step remains valid; output rounds to two."
  expected = Float32[2]
  observed = prox!(zeros(Float32, 1), NormL1(1.0), Float32[2], pow2(Float64, -160))
end

@row begin
  id = "steps.routes.positive.joint"; case = ""; family = :steps; oracle = :prox
  types = "Float32 argument/destination; Float64 step/level"; backend = :cpu; rules = "P2 R3"
  problem = "Tiny positive original step remains valid; output rounds to two."
  expected = Float32[2]
  observed = let y = zeros(Float32, 1); value_and_prox!(y, NormL1(1.0), Float32[2], pow2(Float64, -160)); y end
end

@row begin
  id = "steps.routes.positive.entry"; case = ""; family = :steps; oracle = :prox
  types = "Float32 argument/destination; Float64 step/level"; backend = :cpu; rules = "P2 R3"
  problem = "Tiny positive original step remains valid; output rounds to two."
  expected = Float32[2]
  observed = [prox_entry(NormL1(1.0), 2f0, 1, pow2(Float64, -160))]
end

@row begin
  id = "steps.routes.positive.block"; case = ""; family = :steps; oracle = :prox
  types = "Float32 argument/destination; Float64 step/level"; backend = :cpu; rules = "P2 R3"
  problem = "Tiny positive original step remains valid; output rounds to two."
  expected = Float32[2]
  observed = first(prox!((Float32[7], Float32[7]), SeparableSum(NormL1(1.0), NormL1(1.0)), (Float32[2], Float32[3]), pow2(Float64, -160)))
end

@row begin
  id = "steps.metric.tiny"; case = ""; family = :steps; oracle = :prox
  types = "Float32 argument/destination/geometry; Float64 steps"; backend = :cpu; rules = "P3 A1"
  problem = "Scale-free diagonal projection onto y1+y2=1; P3/A1 component budget 16eps(Float32)."
  expected = Float32[1, 0]
  observed = prox!(zeros(Float32, 2), IndHyperplane(Float32[1, 1], 1f0), Float32[2, 2], [1e-50, 2e-50])
  compare = (a, b) -> length(a) == 2 && all(within_bound(a[i], b[i], 16eps(Float32)) for i in eachindex(b))
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[-Inf, -Inf])))
end

@row begin
  id = "steps.metric.normalized"; case = ""; family = :steps; oracle = :prox
  types = "Float32 argument/destination/geometry; Float64 steps"; backend = :cpu; rules = "P3 A1"
  problem = "Scale-free diagonal projection onto y1+y2=1; P3/A1 component budget 16eps(Float32)."
  expected = Float32[1, 0]
  observed = prox!(zeros(Float32, 2), IndHyperplane(Float32[1, 1], 1f0), Float32[2, 2], [1.0, 2.0])
  compare = (a, b) -> length(a) == 2 && all(within_bound(a[i], b[i], 16eps(Float32)) for i in eachindex(b))
end

@row begin
  id = "steps.metric.ratio-underflow.exception"; case = ""; family = :steps; oracle = :validation
  types = "Float32 output; Float64 steps"; backend = :cpu; rules = "P3 O6"
  problem = "Metric ratio underflows Float32; reject before writing."
  expected = Throws(:ArgumentError)
  observed = prox!(Float32[7, 7], IndHyperplane(Float32[1, 1], 1f0), Float32[2, 2], [floatmin(Float64), floatmax(Float64)])
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[2.0, 2.0])))
end

@row begin
  id = "steps.metric.ratio-underflow.unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float32 output; Float64 steps"; backend = :cpu; rules = "P3 O6"
  problem = "Metric ratio underflows Float32; reject before writing."
  expected = Float32[7, 7]
  observed = let y = Float32[7, 7]; try prox!(y, IndHyperplane(Float32[1, 1], 1f0), Float32[2, 2], [floatmin(Float64), floatmax(Float64)]) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, Float32[2.0, 2.0])))
end

@row begin
  id = "steps.routes.Float32.negative.block-unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float32 blocks; Float64 step"; backend = :cpu; rules = "P1 P2 O6"
  problem = "Block validation of the original step must precede every block write."
  expected = (Float32[7], Float32[7])
  observed = let y = (Float32[7], Float32[7]); try prox!(y, SeparableSum(NormL1(one(Float32)), NormL1(one(Float32))), (Float32[2], Float32[3]), -pow2(Float64, -160)) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, (Float32[2.0], Float32[3.0]))))
end

@row begin
  id = "steps.routes.Float32.nan.block-unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float32 blocks; Float64 step"; backend = :cpu; rules = "P1 P2 O6"
  problem = "Block validation of the original step must precede every block write."
  expected = (Float32[7], Float32[7])
  observed = let y = (Float32[7], Float32[7]); try prox!(y, SeparableSum(NormL1(one(Float32)), NormL1(one(Float32))), (Float32[2], Float32[3]), NaN) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, (Float32[NaN], Float32[NaN]))))
end

@row begin
  id = "steps.routes.Float64.negative.block-unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float64 blocks; Float64 step"; backend = :cpu; rules = "P1 P2 O6"
  problem = "Block validation of the original step must precede every block write."
  expected = (Float64[7], Float64[7])
  observed = let y = (Float64[7], Float64[7]); try prox!(y, SeparableSum(NormL1(one(Float64)), NormL1(one(Float64))), (Float64[2], Float64[3]), -pow2(Float64, -160)) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, ([2.0], [3.0]))))
end

@row begin
  id = "steps.routes.Float64.nan.block-unchanged"; case = ""; family = :steps; oracle = :storage
  types = "Float64 blocks; Float64 step"; backend = :cpu; rules = "P1 P2 O6"
  problem = "Block validation of the original step must precede every block write."
  expected = (Float64[7], Float64[7])
  observed = let y = (Float64[7], Float64[7]); try prox!(y, SeparableSum(NormL1(one(Float64)), NormL1(one(Float64))), (Float64[2], Float64[3]), NaN) catch end; y end
  broken = (group = :G3, baseline = Returned(v -> same(v, ([NaN], [NaN]))))
end

end
