module ConformanceGPUArraysTest

using Test
using ProximalBase
using JLArrays
using GPUArraysCore: allowscalar
using ..Conformance

allowscalar(false)
# G1: entrywise decisions include an exact tie and both neighbors.
for T in (Float32, Float64)
  x = T[prevfloat(T(1)), 1, nextfloat(T(1))]
  @row begin
    id = "gpu_arrays.threshold.$T"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "argument/destination/parameter/step=$T"; backend = :jlarray
    rules = "G1 G2 A2"; problem = "unit soft threshold: below and at threshold choose zero; nextfloat(1) leaves eps(T)"
    expected = T[0, 0, eps(T)]
    observed = Array(prox(NormL1(T(1)), JLArray(x), T(1)))
  end
  @row begin
    id = "gpu_arrays.output-type.$T"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "argument/destination=$T"; backend = :jlarray
    rules = "G1 G5 T1"; problem = "allocating proximal output retains the device layout and argument element type"
    expected = typeof(JLArray(T[1]))
    observed = typeof(prox(NormL1(T(1)), JLArray(T[2]), T(1)))
  end
  @row begin
    id = "gpu_arrays.value-type.$T"; case = ""; family = :gpu_arrays; oracle = :value
    types = "argument=$T Float64 parameter"; backend = :jlarray
    rules = "G1 T1"; problem = "scalar value type follows argument"
    expected = T
    observed = typeof(NormL1(1.0)(JLArray(T[2])))
  end
  @row begin
    id = "gpu_arrays.inf.$T"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "argument/destination=$T"; backend = :jlarray
    rules = "G1 E3"; problem = "soft threshold of [-Inf,4] is [-Inf,3]"
    expected = T[-Inf, 3]
    observed = Array(prox(NormL1(T(1)), JLArray(T[-Inf, 4]), T(1)))
  end
  @row begin
    id = "gpu_arrays.inf-value.$T"; case = ""; family = :gpu_arrays; oracle = :value
    types = "argument=$T"; backend = :jlarray
    rules = "G1 E3"; problem = "l1 norm of [-Inf,4] has exact value +Inf in argument return type"
    expected = T(Inf)
    observed = NormL1(T(1))(JLArray(T[-Inf, 4]))
  end
end
# G5 / #322: combine risky host level and step with device weights.
for k in (-160, 160)
  weight = pow2(Float32, k > 0 ? -120 : 120)
  step = pow2(Float64, k > 0 ? -40 : 40)
  @row begin
    id = "gpu_arrays.host-level.$k"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "Float32 device argument/destination/weights Float64 host level/step"; backend = :jlarray
    rules = "G1 G5 T3 R3"; problem = "#322: combined threshold=1 although level 2^$k does not fit Float32"
    expected = Float32[1]
    observed = Array(prox(NormL1(pow2(Float64, k), Float32[weight]), JLArray(Float32[2]), step))
  end
end
# G1 / #372: l1-ball threshold and adjacent radii. Only the zero decision
# is exact; nonzero entries of this reduction receive an A6 error budget.
for (label, radius) in ((:below, prevfloat(1f0)), (:tie, 1f0), (:above, nextfloat(1f0)))
  @row begin
    id = "gpu_arrays.l1-zero.$label"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "Float32 argument/destination/radius"; backend = :jlarray
    rules = "G1 A2 A6"; problem = "#372: x=[2,1], radius adjacent to 1; second coordinate is zero iff radius<=1"
    expected = label != :above
    observed = iszero(Array(prox(IndBallL1(radius), JLArray(Float32[2, 1])))[2])
  end
  @row begin
    id = "gpu_arrays.l1-value.$label"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "Float32 argument/destination/radius"; backend = :jlarray
    rules = "G1 G2 A6"; problem = "l1 projection: first coordinate min(radius,(1+radius)/2); 8eps(Float32) absolute budget, bound pending"
    expected = bigref(() -> min(BigFloat(radius), (1 + BigFloat(radius))/2), Float32)
    observed = Array(prox(IndBallL1(radius), JLArray(Float32[2, 1])))[1]
    compare = (a, b) -> within_bound(a, b, 8eps(Float32))
  end
end
# E2 / G4: NaN must survive comparisons in separable and coupled kernels.
for (name, f, want, baseline) in (
    (:l1, NormL1(1f0), Float32[NaN, 3], nothing),
    (:l2, NormL2(1f0), Float32[NaN, NaN], (group = :G11, baseline = Returned(v -> same(v, Float32[0, 0])))),
    (:group, GroupNormL2(1f0, [1:1, 2:2]), Float32[NaN, 3], (group = :G11, baseline = Returned(v -> same(v, Float32[0, 3])))),
    (:box, IndBox(-1f0, 1f0), Float32[NaN, 1], nothing),
    (:ball, IndBallL1(1f0), Float32[NaN, NaN], (group = :G11, baseline = Returned(v -> same(v, Float32[NaN, 1])))))
  @row begin
    id = "gpu_arrays.nan.$name.prox"; case = ""; family = :gpu_arrays; oracle = :prox
    types = "Float32 device argument/destination"; backend = :jlarray
    rules = "G1 G4 E2"; problem = "x=[NaN,4]: NaN affects only its separable coordinate or its entire coupled block"
    expected = want
    observed = Array(prox(f, JLArray(Float32[NaN, 4]), 1f0))
    broken = baseline
  end
  @row begin
    id = "gpu_arrays.nan.$name.value"; case = ""; family = :gpu_arrays; oracle = :value
    types = "Float32 device argument"; backend = :jlarray
    rules = "G1 G4 E2"; problem = "value depends on NaN and must return NaN, including indicators"
    expected = Float32(NaN)
    observed = f(JLArray(Float32[NaN, 4]))
    broken = name in (:box, :ball) ? (group = :G11, baseline = Returned(v -> same(v, Float32(Inf)))) : nothing
  end
end
@notcovered :gpu_arrays :environment "Float32/Float64" :jlarray "G3 G6: JLArrays checks dispatch and layout only; real hardware flush-to-zero and subnormal modes cannot be tested here."
@notcovered :gpu_arrays :performance "Float32/Float64" :jlarray "G6: JLArrays cannot establish real-device throughput or synchronization behavior."

end
