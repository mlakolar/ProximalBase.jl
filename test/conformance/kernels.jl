module ConformanceKernelsTest

using Test
using ProximalBase
using ..Conformance

# T3/R1-R3/A2: converted models, range recovery, and exact ties.

@row begin
  id = "V1"; case = "V1"; family = :kernels; oracle = :value
  types = "Float64"; backend = :cpu; rules = "R1"
  problem = "Scaled square; R1 budget 2 ULP."
  expected = pow2(Float64, 199)
  observed = SqrNormL2(pow2(Float64, -1000))([pow2(Float64, 600)])
  compare = (a, b) -> within_ulps(a, b, 2)
end

@row begin
  id = "V2"; case = "V2"; family = :kernels; oracle = :value
  types = "Float64"; backend = :cpu; rules = "R1 A6"
  problem = "Underflowed square terms restored before scaling; R1 budget 2 ULP."
  expected = pow2(Float64, -23) + pow2(Float64, -72)
  observed = SqrNormL2(pow2(Float64, 1000))([pow2(Float64, -511); fill(pow2(Float64, -538), 32)])
  compare = (a, b) -> within_ulps(a, b, 2)
end

@row begin
  id = "V3"; case = "V3"; family = :kernels; oracle = :prox
  types = "Float32 destination; arguments and factors as written"; backend = :cpu; rules = "T3 T4 R2 R3 A2"
  problem = "Combined parameters rounded once; range recovery retains unrounded factors."
  expected = Float32[0.5]
  observed = prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160)), Float32[1], pow2(Float64, -160))
end

@row begin
  id = "V4"; case = "V4"; family = :kernels; oracle = :prox
  types = "Float32 destination; arguments and factors as written"; backend = :cpu; rules = "T3 T4 R2 R3 A2"
  problem = "Combined parameters rounded once; range recovery retains unrounded factors."
  expected = [pow2(Float32, -60)]
  observed = prox!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160)), [pow2(Float32, 100)], 1.0)
end

@row begin
  id = "V6"; case = "V6"; family = :kernels; oracle = :prox
  types = "Float64 destination; arguments and factors as written"; backend = :cpu; rules = "T3 T4 R2 R3 A2"
  problem = "Combined parameters rounded once; range recovery retains unrounded factors."
  expected = [pow2(Float64, -700)]
  observed = prox!(zeros(Float64, 1), SqrNormL2(pow2(Float64, 800)), [pow2(Float64, 900)], pow2(Float64, 800))
end

@row begin
  id = "V8"; case = "V8"; family = :kernels; oracle = :prox
  types = "Float32 destination; arguments and factors as written"; backend = :cpu; rules = "T3 T4 R2 R3 A2"
  problem = "Combined parameters rounded once; range recovery retains unrounded factors."
  expected = Float32[0]
  observed = prox!(zeros(Float32, 1), NormL1(1 - pow2(Float64, -30)), Float32[1], 1.0)
end

@row begin
  id = "V10"; case = "V10"; family = :kernels; oracle = :prox
  types = "Float64 destination; arguments and factors as written"; backend = :cpu; rules = "T3 T4 R2 R3 A2"
  problem = "Combined parameters rounded once; range recovery retains unrounded factors."
  expected = [pow2(Float64, -46)]
  observed = prox!(zeros(Float64, 1), NormL1(1f0 + pow2(Float32, -23)), Float32[1], 1f0 - pow2(Float32, -23))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "V13"; case = "V13"; family = :kernels; oracle = :prox
  types = "Float64 destination; arguments and factors as written"; backend = :cpu; rules = "T3 T4 R2 R3 A2"
  problem = "Combined parameters rounded once; range recovery retains unrounded factors."
  expected = [pow2(Float64, -54)]
  observed = prox!(zeros(Float64, 1), NormL1(pow2(Float64, -538), [pow2(Float64, 1022)]), [pow2(Float64, -52)], 3 * pow2(Float64, -538))
end

@row begin
  id = "V5.value"; case = "V5"; family = :kernels; oracle = :value
  types = "Float32 argument/destination; Float64 level"; backend = :cpu; rules = "R1 R2"
  problem = "Finite ridge result despite level outside Float32 range."
  expected = 0.5f0
  observed = SqrNormL2(pow2(Float64, 160))([pow2(Float32, -80)])
end

@row begin
  id = "V5.gradient"; case = "V5"; family = :kernels; oracle = :gradient
  types = "Float32 argument/destination; Float64 level"; backend = :cpu; rules = "R1 R2"
  problem = "Finite ridge result despite level outside Float32 range."
  expected = [pow2(Float32, 80)]
  observed = gradient!(zeros(Float32, 1), SqrNormL2(pow2(Float64, 160)), [pow2(Float32, -80)])
end

@row begin
  id = "V7.value"; case = "V7"; family = :kernels; oracle = :value
  types = "Float64"; backend = :cpu; rules = "R2 O1"
  problem = "Overflowing centered difference; independent gradient remains finite."
  expected = Inf
  observed = value_and_gradient!(zeros(1), SqrNormL2(pow2(Float64, -1022), [-pow2(Float64, 1023)]), [pow2(Float64, 1023)])
end

@row begin
  id = "V7.gradient"; case = "V7"; family = :kernels; oracle = :gradient
  types = "Float64"; backend = :cpu; rules = "R2 O1"
  problem = "Overflowing centered difference; independent gradient remains finite."
  expected = [4.0]
  observed = let g = zeros(1); value_and_gradient!(g, SqrNormL2(pow2(Float64, -1022), [-pow2(Float64, 1023)]), [pow2(Float64, 1023)]); g end
end

# O2: one fused invocation from a sentinel destination; the scalar and a copy of
# the buffer it wrote are asserted in separate rows. A fused call that returns
# the right scalar but leaves the sentinel fails V9.prox.
fused9 = once() do
  y = fill(7f0, 1)
  value = value_and_prox!(y, SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1], 1.0)
  (value, copy(y))
end

@row begin
  id = "V9.value"; case = "V9"; family = :kernels; oracle = :value
  types = "Float32 argument/destination; Float64 center/level"; backend = :cpu; rules = "T3 O2"
  problem = "Center converts to one in stored-output problem; the scalar of the fused value_and_prox! call."
  expected = 0f0
  observed = fused9()[1]
end

@row begin
  id = "V9.prox"; case = "V9"; family = :kernels; oracle = :prox
  types = "Float32 argument/destination; Float64 center/level"; backend = :cpu; rules = "T3 O2"
  problem = "Center converts to one in stored-output problem; the destination written by the same fused value_and_prox! call, started from the sentinel 7."
  expected = Float32[1]
  observed = fused9()[2]
end

@row begin
  id = "V9.prox.standalone"; case = "V9"; family = :kernels; oracle = :prox
  types = "Float32 argument/destination; Float64 center/level"; backend = :cpu; rules = "T3 O2"
  problem = "Center converts to one in stored-output problem; standalone prox! from the sentinel 7."
  expected = Float32[1]
  observed = prox!(fill(7f0, 1), SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1], 1.0)
end

@row begin
  id = "V11.value"; case = "V11"; family = :kernels; oracle = :value
  types = "Float32 argument/value; Float64 gradient/center"; backend = :cpu; rules = "T3 O1"
  problem = "Value and gradient have separate model precisions."
  expected = 0f0
  observed = value_and_gradient!(zeros(1), SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1])
end

@row begin
  id = "V11.gradient"; case = "V11"; family = :kernels; oracle = :gradient
  types = "Float32 argument/value; Float64 gradient/center"; backend = :cpu; rules = "T3 O1"
  problem = "Value and gradient have separate model precisions."
  expected = [-pow2(Float64, -28)]
  observed = let g = zeros(1); value_and_gradient!(g, SqrNormL2(1.0, [1 + pow2(Float64, -28)]), Float32[1]); g end
end

@row begin
  id = "kernels.NormL1.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float32(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(Float32(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL1.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * Float32(0.25)
  observed = NormL1(Float32(0.5))(Float32[0.25])
end

@row begin
  id = "kernels.NormL1.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float64(0.5), Float64(0.5)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(Float64(0.5)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.NormL1.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * Float64(0.25)
  observed = NormL1(Float64(0.5))(Float64[0.25])
end

@row begin
  id = "kernels.NormL1.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64))]
  observed = prox!(zeros(Float32, 1), NormL1(Float32(0.5)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL1.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * Float64(0.25)
  observed = NormL1(Float32(0.5))(Float64[0.25])
end

@row begin
  id = "kernels.NormL1.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(Float32(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL1.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float64(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(Float64(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL1.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * Float32(0.25)
  observed = NormL1(Float64(0.5))(Float32[0.25])
end

@row begin
  id = "kernels.NormL1.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float64(0.5), Float32(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(Float32(0.5)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.WeightedNormL1.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) - combine(Float32, Float32(0.5), Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float32, 1), NormL1(Float32(0.5), Float32[0.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.WeightedNormL1.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5), Float32(0.5)) * Float32(0.25)
  observed = NormL1(Float32(0.5), Float32[0.5])(Float32[0.25])
end

@row begin
  id = "kernels.WeightedNormL1.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[Float64(0.25) - combine(Float64, Float64(0.5), Float64(0.5), Float64(0.5))]
  observed = prox!(zeros(Float64, 1), NormL1(Float64(0.5), Float64[0.5]), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.WeightedNormL1.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5), Float64(0.5)) * Float64(0.25)
  observed = NormL1(Float64(0.5), Float64[0.5])(Float64[0.25])
end

@row begin
  id = "kernels.WeightedNormL1.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float32, 1), NormL1(Float32(0.5), Float32[0.5]), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.WeightedNormL1.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5), Float32(0.5)) * Float64(0.25)
  observed = NormL1(Float32(0.5), Float32[0.5])(Float64[0.25])
end

@row begin
  id = "kernels.WeightedNormL1.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float64, 1), NormL1(Float32(0.5), Float32[0.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.WeightedNormL1.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) - combine(Float32, Float32(0.5), Float64(0.5), Float64(0.5))]
  observed = prox!(zeros(Float32, 1), NormL1(Float64(0.5), Float64[0.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.WeightedNormL1.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5), Float64(0.5)) * Float32(0.25)
  observed = NormL1(Float64(0.5), Float64[0.5])(Float32[0.25])
end

@row begin
  id = "kernels.WeightedNormL1.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) - combine(Float32, Float64(0.5), Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float32, 1), NormL1(Float32(0.5), Float32[0.5]), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.SqrNormL2.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) / (one(Float32) + combine(Float32, Float32(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float32(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SqrNormL2.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * abs2(Float32(0.25)) / Float32(2)
  observed = SqrNormL2(Float32(0.5))(Float32[0.25])
end

@row begin
  id = "kernels.SqrNormL2.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[Float64(0.25) / (one(Float64) + combine(Float64, Float64(0.5), Float64(0.5)))]
  observed = prox!(zeros(Float64, 1), SqrNormL2(Float64(0.5)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.SqrNormL2.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * abs2(Float64(0.25)) / Float64(2)
  observed = SqrNormL2(Float64(0.5))(Float64[0.25])
end

@row begin
  id = "kernels.SqrNormL2.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float64(0.25) / (one(Float64) + combine(Float64, Float32(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float32(0.5)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SqrNormL2.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * abs2(Float64(0.25)) / Float64(2)
  observed = SqrNormL2(Float32(0.5))(Float64[0.25])
end

@row begin
  id = "kernels.SqrNormL2.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[Float64(0.25) / (one(Float64) + combine(Float64, Float32(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float64, 1), SqrNormL2(Float32(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SqrNormL2.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) / (one(Float32) + combine(Float32, Float32(0.5), Float64(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float64(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SqrNormL2.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * abs2(Float32(0.25)) / Float32(2)
  observed = SqrNormL2(Float64(0.5))(Float32[0.25])
end

@row begin
  id = "kernels.SqrNormL2.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) / (one(Float32) + combine(Float32, Float64(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float32(0.5)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.CenteredSqrNormL2.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[(Float32(0.25) + combine(Float32, Float32(0.5), Float32(0.5)) * Float32(1.5)) / (one(Float32) + combine(Float32, Float32(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float32(0.5), Float32[1.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.CenteredSqrNormL2.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * abs2(Float32(0.25) - Float32(Float32(1.5))) / Float32(2)
  observed = SqrNormL2(Float32(0.5), Float32[1.5])(Float32[0.25])
end

@row begin
  id = "kernels.CenteredSqrNormL2.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[(Float64(0.25) + combine(Float64, Float64(0.5), Float64(0.5)) * Float64(1.5)) / (one(Float64) + combine(Float64, Float64(0.5), Float64(0.5)))]
  observed = prox!(zeros(Float64, 1), SqrNormL2(Float64(0.5), Float64[1.5]), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.CenteredSqrNormL2.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * abs2(Float64(0.25) - Float64(Float64(1.5))) / Float64(2)
  observed = SqrNormL2(Float64(0.5), Float64[1.5])(Float64[0.25])
end

@row begin
  id = "kernels.CenteredSqrNormL2.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[(Float64(0.25) + combine(Float64, Float32(0.5), Float32(0.5)) * Float64(1.5)) / (one(Float64) + combine(Float64, Float32(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float32(0.5), Float32[1.5]), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.CenteredSqrNormL2.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * abs2(Float64(0.25) - Float64(Float32(1.5))) / Float64(2)
  observed = SqrNormL2(Float32(0.5), Float32[1.5])(Float64[0.25])
end

@row begin
  id = "kernels.CenteredSqrNormL2.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[(Float64(0.25) + combine(Float64, Float32(0.5), Float32(0.5)) * Float64(1.5)) / (one(Float64) + combine(Float64, Float32(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float64, 1), SqrNormL2(Float32(0.5), Float32[1.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.CenteredSqrNormL2.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[(Float32(0.25) + combine(Float32, Float32(0.5), Float64(0.5)) * Float32(1.5)) / (one(Float32) + combine(Float32, Float32(0.5), Float64(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float64(0.5), Float64[1.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.CenteredSqrNormL2.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * abs2(Float32(0.25) - Float32(Float64(1.5))) / Float32(2)
  observed = SqrNormL2(Float64(0.5), Float64[1.5])(Float32[0.25])
end

@row begin
  id = "kernels.CenteredSqrNormL2.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[(Float32(0.25) + combine(Float32, Float64(0.5), Float32(0.5)) * Float32(1.5)) / (one(Float32) + combine(Float32, Float64(0.5), Float32(0.5)))]
  observed = prox!(zeros(Float32, 1), SqrNormL2(Float32(0.5), Float32[1.5]), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.ElasticNet.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float32(0.5)), zero(Float32)) / (one(Float32) + combine(Float32, Float32(0.5), Float32(2)))]
  observed = prox!(zeros(Float32, 1), ElasticNet(Float32(0.5), Float32(2)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.ElasticNet.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * Float32(0.25) + Float32(Float32(2)) * abs2(Float32(0.25)) / Float32(2)
  observed = ElasticNet(Float32(0.5), Float32(2))(Float32[0.25])
end

@row begin
  id = "kernels.ElasticNet.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float64(0.5), Float64(0.5)), zero(Float64)) / (one(Float64) + combine(Float64, Float64(0.5), Float64(2)))]
  observed = prox!(zeros(Float64, 1), ElasticNet(Float64(0.5), Float64(2)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.ElasticNet.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * Float64(0.25) + Float64(Float64(2)) * abs2(Float64(0.25)) / Float64(2)
  observed = ElasticNet(Float64(0.5), Float64(2))(Float64[0.25])
end

@row begin
  id = "kernels.ElasticNet.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64)) / (one(Float64) + combine(Float64, Float32(0.5), Float32(2)))]
  observed = prox!(zeros(Float32, 1), ElasticNet(Float32(0.5), Float32(2)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.ElasticNet.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * Float64(0.25) + Float64(Float32(2)) * abs2(Float64(0.25)) / Float64(2)
  observed = ElasticNet(Float32(0.5), Float32(2))(Float64[0.25])
end

@row begin
  id = "kernels.ElasticNet.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64)) / (one(Float64) + combine(Float64, Float32(0.5), Float32(2)))]
  observed = prox!(zeros(Float64, 1), ElasticNet(Float32(0.5), Float32(2)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.ElasticNet.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float64(0.5)), zero(Float32)) / (one(Float32) + combine(Float32, Float32(0.5), Float64(2)))]
  observed = prox!(zeros(Float32, 1), ElasticNet(Float64(0.5), Float64(2)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.ElasticNet.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * Float32(0.25) + Float32(Float64(2)) * abs2(Float32(0.25)) / Float32(2)
  observed = ElasticNet(Float64(0.5), Float64(2))(Float32[0.25])
end

@row begin
  id = "kernels.ElasticNet.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float64(0.5), Float32(0.5)), zero(Float32)) / (one(Float32) + combine(Float32, Float64(0.5), Float32(2)))]
  observed = prox!(zeros(Float32, 1), ElasticNet(Float32(0.5), Float32(2)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.Berhu.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float32(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), Berhu(Float32(0.5), Float32(1)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.Berhu.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * Float32(0.25)
  observed = Berhu(Float32(0.5), Float32(1))(Float32[0.25])
end

@row begin
  id = "kernels.Berhu.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float64(0.5), Float64(0.5)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), Berhu(Float64(0.5), Float64(1)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.Berhu.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * Float64(0.25)
  observed = Berhu(Float64(0.5), Float64(1))(Float64[0.25])
end

@row begin
  id = "kernels.Berhu.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64))]
  observed = prox!(zeros(Float32, 1), Berhu(Float32(0.5), Float32(1)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.Berhu.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * Float64(0.25)
  observed = Berhu(Float32(0.5), Float32(1))(Float64[0.25])
end

@row begin
  id = "kernels.Berhu.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), Berhu(Float32(0.5), Float32(1)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.Berhu.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float64(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), Berhu(Float64(0.5), Float64(1)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.Berhu.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * Float32(0.25)
  observed = Berhu(Float64(0.5), Float64(1))(Float32[0.25])
end

@row begin
  id = "kernels.Berhu.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float64(0.5), Float32(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), Berhu(Float32(0.5), Float32(1)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.HingeLoss.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; labels Float32; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) + combine(Float32, Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float32, 1), HingeLoss(Float32[1], Float32[0.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.HingeLoss.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; labels Float32; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * (Float32(1) - Float32(0.25))
  observed = HingeLoss(Float32[1], Float32[0.5])(Float32[0.25])
end

@row begin
  id = "kernels.HingeLoss.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; labels Float64; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[Float64(0.25) + combine(Float64, Float64(0.5), Float64(0.5))]
  observed = prox!(zeros(Float64, 1), HingeLoss(Float64[1], Float64[0.5]), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.HingeLoss.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; labels Float64; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * (Float64(1) - Float64(0.25))
  observed = HingeLoss(Float64[1], Float64[0.5])(Float64[0.25])
end

@row begin
  id = "kernels.HingeLoss.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; labels Float32; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float64(0.25) + combine(Float64, Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float32, 1), HingeLoss(Float32[1], Float32[0.5]), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.HingeLoss.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; labels Float32; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * (Float64(1) - Float64(0.25))
  observed = HingeLoss(Float32[1], Float32[0.5])(Float64[0.25])
end

@row begin
  id = "kernels.HingeLoss.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; labels Float32; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[Float64(0.25) + combine(Float64, Float32(0.5), Float32(0.5))]
  observed = prox!(zeros(Float64, 1), HingeLoss(Float32[1], Float32[0.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.HingeLoss.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; labels Float64; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) + combine(Float32, Float32(0.5), Float64(0.5))]
  observed = prox!(zeros(Float32, 1), HingeLoss(Float64[1], Float64[0.5]), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.HingeLoss.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; labels Float64; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * (Float32(1) - Float32(0.25))
  observed = HingeLoss(Float64[1], Float64[0.5])(Float32[0.25])
end

@row begin
  id = "kernels.HingeLoss.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; labels Float32; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[Float32(0.25) + combine(Float32, Float64(0.5), Float32(0.5))]
  observed = prox!(zeros(Float32, 1), HingeLoss(Float32[1], Float32[0.5]), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.NormL0.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[abs2(Float32(0.25)) > 2 * combine(Float32, Float32(0.5), Float32(0.5)) ? Float32(0.25) : zero(Float32)]
  observed = prox!(zeros(Float32, 1), NormL0(Float32(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL0.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5))
  observed = NormL0(Float32(0.5))(Float32[0.25])
end

@row begin
  id = "kernels.NormL0.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[abs2(Float64(0.25)) > 2 * combine(Float64, Float64(0.5), Float64(0.5)) ? Float64(0.25) : zero(Float64)]
  observed = prox!(zeros(Float64, 1), NormL0(Float64(0.5)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.NormL0.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5))
  observed = NormL0(Float64(0.5))(Float64[0.25])
end

@row begin
  id = "kernels.NormL0.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[abs2(Float64(0.25)) > 2 * combine(Float64, Float32(0.5), Float32(0.5)) ? Float64(0.25) : zero(Float64)]
  observed = prox!(zeros(Float32, 1), NormL0(Float32(0.5)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL0.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5))
  observed = NormL0(Float32(0.5))(Float64[0.25])
end

@row begin
  id = "kernels.NormL0.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[abs2(Float64(0.25)) > 2 * combine(Float64, Float32(0.5), Float32(0.5)) ? Float64(0.25) : zero(Float64)]
  observed = prox!(zeros(Float64, 1), NormL0(Float32(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL0.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[abs2(Float32(0.25)) > 2 * combine(Float32, Float32(0.5), Float64(0.5)) ? Float32(0.25) : zero(Float32)]
  observed = prox!(zeros(Float32, 1), NormL0(Float64(0.5)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.NormL0.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5))
  observed = NormL0(Float64(0.5))(Float32[0.25])
end

@row begin
  id = "kernels.NormL0.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[abs2(Float32(0.25)) > 2 * combine(Float32, Float64(0.5), Float32(0.5)) ? Float32(0.25) : zero(Float32)]
  observed = prox!(zeros(Float32, 1), NormL0(Float32(0.5)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.MCP.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float32(0.5)), zero(Float32)) / (one(Float32) - Float32(Float32(0.5)) / Float32(Float32(2)))]
  observed = prox!(zeros(Float32, 1), MCP(Float32(0.5), Float32(2)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.MCP.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * Float32(0.25) - abs2(Float32(0.25)) / (2 * Float32(Float32(2)))
  observed = MCP(Float32(0.5), Float32(2))(Float32[0.25])
end

@row begin
  id = "kernels.MCP.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float64(0.5), Float64(0.5)), zero(Float64)) / (one(Float64) - Float64(Float64(0.5)) / Float64(Float64(2)))]
  observed = prox!(zeros(Float64, 1), MCP(Float64(0.5), Float64(2)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.MCP.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * Float64(0.25) - abs2(Float64(0.25)) / (2 * Float64(Float64(2)))
  observed = MCP(Float64(0.5), Float64(2))(Float64[0.25])
end

@row begin
  id = "kernels.MCP.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64)) / (one(Float64) - Float64(Float32(0.5)) / Float64(Float32(2)))]
  observed = prox!(zeros(Float32, 1), MCP(Float32(0.5), Float32(2)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.MCP.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * Float64(0.25) - abs2(Float64(0.25)) / (2 * Float64(Float32(2)))
  observed = MCP(Float32(0.5), Float32(2))(Float64[0.25])
end

@row begin
  id = "kernels.MCP.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64)) / (one(Float64) - Float64(Float32(0.5)) / Float64(Float32(2)))]
  observed = prox!(zeros(Float64, 1), MCP(Float32(0.5), Float32(2)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.MCP.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float64(0.5)), zero(Float32)) / (one(Float32) - Float32(Float32(0.5)) / Float32(Float64(2)))]
  observed = prox!(zeros(Float32, 1), MCP(Float64(0.5), Float64(2)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.MCP.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * Float32(0.25) - abs2(Float32(0.25)) / (2 * Float32(Float64(2)))
  observed = MCP(Float64(0.5), Float64(2))(Float32[0.25])
end

@row begin
  id = "kernels.MCP.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float64(0.5), Float32(0.5)), zero(Float32)) / (one(Float32) - Float32(Float64(0.5)) / Float32(Float32(2)))]
  observed = prox!(zeros(Float32, 1), MCP(Float32(0.5), Float32(2)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.SCAD.all32.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float32(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), SCAD(Float32(0.5), Float32(3)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SCAD.all32.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float32(0.5)) * Float32(0.25)
  observed = SCAD(Float32(0.5), Float32(3))(Float32[0.25])
end

@row begin
  id = "kernels.SCAD.all64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float64(0.5), Float64(0.5)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), SCAD(Float64(0.5), Float64(3)), Float64[0.25], Float64(0.5))
end

@row begin
  id = "kernels.SCAD.all64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float64 destination; Float64 parameters; Float64 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float64(0.5)) * Float64(0.25)
  observed = SCAD(Float64(0.5), Float64(3))(Float64[0.25])
end

@row begin
  id = "kernels.SCAD.argument64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64))]
  observed = prox!(zeros(Float32, 1), SCAD(Float32(0.5), Float32(3)), Float64[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SCAD.argument64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float64 argument; Float32 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float64, Float32(0.5)) * Float64(0.25)
  observed = SCAD(Float32(0.5), Float32(3))(Float64[0.25])
end

@row begin
  id = "kernels.SCAD.destination64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float64[max(Float64(0.25) - combine(Float64, Float32(0.5), Float32(0.5)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), SCAD(Float32(0.5), Float32(3)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SCAD.parameter64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float32(0.5), Float64(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), SCAD(Float64(0.5), Float64(3)), Float32[0.25], Float32(0.5))
end

@row begin
  id = "kernels.SCAD.parameter64.value"; case = ""; family = :kernels; oracle = :value
  types = "Float32 argument; Float32 destination; Float64 parameters; Float32 step"; backend = :cpu; rules = "T3"
  problem = "One-coordinate dyadic value after parameter conversion; exact representable answer."
  expected = combine(Float32, Float64(0.5)) * Float32(0.25)
  observed = SCAD(Float64(0.5), Float64(3))(Float32[0.25])
end

@row begin
  id = "kernels.SCAD.step64.prox"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float32 parameters; Float64 step"; backend = :cpu; rules = "T3 A2"
  problem = "Dyadic converted problem; combined threshold gamma*level*weight, left to right; equality selects zero."
  expected = Float32[max(Float32(0.25) - combine(Float32, Float64(0.5), Float32(0.5)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), SCAD(Float32(0.5), Float32(3)), Float32[0.25], Float64(0.5))
end

@row begin
  id = "kernels.rounded.Float32.False"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameter; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "T3 threshold rounds once after forming product in promoted factor precision."
  expected = Float32[max(Float32(nextfloat(0.9f0)) - combine(Float32, Float32(9), Float64(0.1)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(Float64(0.1)), [nextfloat(0.9f0)], Float32(9))
end

@row begin
  id = "kernels.rounded.Float32.True"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float32 destination; Float64 parameter; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "T3 threshold rounds once after forming product in promoted factor precision."
  expected = Float32[max(Float32(nextfloat(0.9f0)) - combine(Float32, Float32(9), Float64(0.1), Float64(1)), zero(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(Float64(0.1), Float64[1]), [nextfloat(0.9f0)], Float32(9))
end

@row begin
  id = "kernels.rounded.Float64.False"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameter; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "T3 threshold rounds once after forming product in promoted factor precision."
  expected = Float64[max(Float64(nextfloat(0.9f0)) - combine(Float64, Float32(9), Float32(0.1)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(Float32(0.1)), [nextfloat(0.9f0)], Float32(9))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.rounded.Float64.True"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument; Float64 destination; Float32 parameter; Float32 step"; backend = :cpu; rules = "T3 A2"
  problem = "T3 threshold rounds once after forming product in promoted factor precision."
  expected = Float64[max(Float64(nextfloat(0.9f0)) - combine(Float64, Float32(9), Float32(0.1), Float32(1)), zero(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(Float32(0.1), Float32[1]), [nextfloat(0.9f0)], Float32(9))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.identity.Float32.subnormal"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [nextfloat(zero(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(zero(Float32)), [nextfloat(zero(Float32))], one(Float32))
end

@row begin
  id = "kernels.identity.Float32.below_floatmin"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [prevfloat(floatmin(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(zero(Float32)), [prevfloat(floatmin(Float32))], one(Float32))
end

@row begin
  id = "kernels.identity.Float32.floatmin"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [floatmin(Float32)]
  observed = prox!(zeros(Float32, 1), NormL1(zero(Float32)), [floatmin(Float32)], one(Float32))
end

@row begin
  id = "kernels.identity.Float32.below_floatmax"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [prevfloat(floatmax(Float32))]
  observed = prox!(zeros(Float32, 1), NormL1(zero(Float32)), [prevfloat(floatmax(Float32))], one(Float32))
end

@row begin
  id = "kernels.identity.Float64.subnormal"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [nextfloat(zero(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(zero(Float64)), [nextfloat(zero(Float64))], one(Float64))
end

@row begin
  id = "kernels.identity.Float64.below_floatmin"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [prevfloat(floatmin(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(zero(Float64)), [prevfloat(floatmin(Float64))], one(Float64))
end

@row begin
  id = "kernels.identity.Float64.floatmin"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [floatmin(Float64)]
  observed = prox!(zeros(Float64, 1), NormL1(zero(Float64)), [floatmin(Float64)], one(Float64))
end

@row begin
  id = "kernels.identity.Float64.below_floatmax"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R3 E4"
  problem = "Zero level copies finite boundary and subnormal entries exactly."
  expected = [prevfloat(floatmax(Float64))]
  observed = prox!(zeros(Float64, 1), NormL1(zero(Float64)), [prevfloat(floatmax(Float64))], one(Float64))
end

@row begin
  id = "kernels.NormL1.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1 - combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))]
  observed = prox!(zeros(1), NormL1((1f0 + pow2(Float32, -23))), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.WeightedNormL1.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1 - combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))]
  observed = prox!(zeros(1), NormL1((1f0 + pow2(Float32, -23)), Float32[1]), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.SqrNormL2.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1 / (1 + combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23))))]
  observed = prox!(zeros(1), SqrNormL2((1f0 + pow2(Float32, -23))), Float32[1], (1f0 - pow2(Float32, -23)))
end

@row begin
  id = "kernels.CenteredSqrNormL2.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1 / (1 + combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23))))]
  observed = prox!(zeros(1), SqrNormL2((1f0 + pow2(Float32, -23)), Float32[0]), Float32[1], (1f0 - pow2(Float32, -23)))
end

@row begin
  id = "kernels.ElasticNet.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [(1 - combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))) / (1 + Float64((1f0 - pow2(Float32, -23))))]
  observed = prox!(zeros(1), ElasticNet((1f0 + pow2(Float32, -23)), 1f0), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.Berhu.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1 - combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))]
  observed = prox!(zeros(1), Berhu((1f0 + pow2(Float32, -23)), 2f0), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.HingeLoss.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [-1 + combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))]
  observed = prox!(zeros(1), HingeLoss(Float32[1], [(1f0 + pow2(Float32, -23))]), Float32[-1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.NormL0.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1.0]
  observed = prox!(zeros(1), NormL0(0.5f0 * (1f0 + pow2(Float32, -23))), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.MCP.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [(1 - combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))) / (1 - Float64((1f0 - pow2(Float32, -23))) / 3)]
  observed = prox!(zeros(1), MCP((1f0 + pow2(Float32, -23)), 3f0), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.SCAD.wider-product"; case = ""; family = :kernels; oracle = :prox
  types = "Float32 argument/parameters/step; Float64 destination"; backend = :cpu; rules = "T3 T4 A2"
  problem = "Product of neighboring Float32 factors recomputed in Float64 model; NormL0 compares x squared to twice the combined level-step."
  expected = [1 - combine(Float64, (1f0 - pow2(Float32, -23)), (1f0 + pow2(Float32, -23)))]
  observed = prox!(zeros(1), SCAD((1f0 + pow2(Float32, -23)), 3f0), Float32[1], (1f0 - pow2(Float32, -23)))
  broken = (group = :G1, baseline = Returned(v -> same(v, [0.0])))
end

@row begin
  id = "kernels.NormL1.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[0.25]; prox!(x, NormL1(Float32(0.5)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.WeightedNormL1.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[0.25]; prox!(x, NormL1(Float32(1), Float32[0.5]), x, Float32(0.5)) end
end

@row begin
  id = "kernels.SqrNormL2.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[1]
  observed = let x = Float32[2]; prox!(x, SqrNormL2(Float32(2)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.CenteredSqrNormL2.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[1]
  observed = let x = Float32[0]; prox!(x, SqrNormL2(Float32(2), Float32[2]), x, Float32(0.5)) end
end

@row begin
  id = "kernels.ElasticNet.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[0.25]; prox!(x, ElasticNet(Float32(0.5), one(Float32)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.Berhu.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[0.25]; prox!(x, Berhu(Float32(0.5), one(Float32)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.HingeLoss.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[1]
  observed = let x = Float32[0.75]; prox!(x, HingeLoss(nothing, Float32[0.5]), x, Float32(0.5)) end
end

@row begin
  id = "kernels.NormL0.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[1]; prox!(x, NormL0(one(Float32)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.MCP.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[0.25]; prox!(x, MCP(Float32(0.5), Float32(2)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.SCAD.Float32.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float32[0]
  observed = let x = Float32[0.25]; prox!(x, SCAD(Float32(0.5), Float32(3)), x, Float32(0.5)) end
end

@row begin
  id = "kernels.threshold.Float32.subnormal"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R3 A2"
  problem = "Threshold equality at subnormal or minimum normal coordinate returns zero exactly."
  expected = Float32[0]
  observed = prox!(zeros(Float32, 1), NormL1(nextfloat(zero(Float32))), [nextfloat(zero(Float32))], one(Float32))
end

@row begin
  id = "kernels.threshold.Float32.floatmin"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R3 A2"
  problem = "Threshold equality at subnormal or minimum normal coordinate returns zero exactly."
  expected = Float32[0]
  observed = prox!(zeros(Float32, 1), NormL1(floatmin(Float32)), [floatmin(Float32)], one(Float32))
end

@row begin
  id = "kernels.ridge.Float32.near-floatmax"; case = ""; family = :kernels; oracle = :prox
  types = "Float32"; backend = :cpu; rules = "R2"
  problem = "Largest finite neighbor shrinks without an overflowing product."
  expected = [prevfloat(floatmax(Float32)) / Float32(2)]
  observed = prox!(zeros(Float32, 1), SqrNormL2(one(Float32)), [prevfloat(floatmax(Float32))], one(Float32))
end

@row begin
  id = "kernels.NormL1.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[0.25]; prox!(x, NormL1(Float64(0.5)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.WeightedNormL1.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[0.25]; prox!(x, NormL1(Float64(1), Float64[0.5]), x, Float64(0.5)) end
end

@row begin
  id = "kernels.SqrNormL2.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[1]
  observed = let x = Float64[2]; prox!(x, SqrNormL2(Float64(2)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.CenteredSqrNormL2.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[1]
  observed = let x = Float64[0]; prox!(x, SqrNormL2(Float64(2), Float64[2]), x, Float64(0.5)) end
end

@row begin
  id = "kernels.ElasticNet.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[0.25]; prox!(x, ElasticNet(Float64(0.5), one(Float64)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.Berhu.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[0.25]; prox!(x, Berhu(Float64(0.5), one(Float64)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.HingeLoss.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[1]
  observed = let x = Float64[0.75]; prox!(x, HingeLoss(nothing, Float64[0.5]), x, Float64(0.5)) end
end

@row begin
  id = "kernels.NormL0.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[1]; prox!(x, NormL0(one(Float64)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.MCP.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[0.25]; prox!(x, MCP(Float64(0.5), Float64(2)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.SCAD.Float64.aliased-tie"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "A2 O7"
  problem = "Aliased branch against exact dyadic reference; threshold equality selects zero (hinge clamps to one)."
  expected = Float64[0]
  observed = let x = Float64[0.25]; prox!(x, SCAD(Float64(0.5), Float64(3)), x, Float64(0.5)) end
end

@row begin
  id = "kernels.threshold.Float64.subnormal"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R3 A2"
  problem = "Threshold equality at subnormal or minimum normal coordinate returns zero exactly."
  expected = Float64[0]
  observed = prox!(zeros(Float64, 1), NormL1(nextfloat(zero(Float64))), [nextfloat(zero(Float64))], one(Float64))
end

@row begin
  id = "kernels.threshold.Float64.floatmin"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R3 A2"
  problem = "Threshold equality at subnormal or minimum normal coordinate returns zero exactly."
  expected = Float64[0]
  observed = prox!(zeros(Float64, 1), NormL1(floatmin(Float64)), [floatmin(Float64)], one(Float64))
end

@row begin
  id = "kernels.ridge.Float64.near-floatmax"; case = ""; family = :kernels; oracle = :prox
  types = "Float64"; backend = :cpu; rules = "R2"
  problem = "Largest finite neighbor shrinks without an overflowing product."
  expected = [prevfloat(floatmax(Float64)) / Float64(2)]
  observed = prox!(zeros(Float64, 1), SqrNormL2(one(Float64)), [prevfloat(floatmax(Float64))], one(Float64))
end

end
