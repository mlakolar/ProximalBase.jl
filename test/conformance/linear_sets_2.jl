module ConformanceLinearSets2Test

using Test
using ProximalBase
using ..Conformance

const baselines = Dict{String,Any}(
  "linear_sets.conjugate.hyperplane.-1040" => (group = :G5, baseline = Threw(:ArgumentError, r"proximal point of the conjugate is not finite at a finite input")),
  "linear_sets.conjugate.halfspace.-1040" => (group = :G5, baseline = Threw(:ArgumentError, r"proximal point of the conjugate is not finite at a finite input")),
  "linear_sets.conjugate.hyperslab.-1040" => (group = :G5, baseline = Threw(:ArgumentError, r"proximal point of the conjugate is not finite at a finite input")),
  "linear_sets.subnormal.Float32.false.false" => (group = :G5, baseline = Returned(v -> isequal(v, Float32[0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float32.false.true" => (group = :G5, baseline = Returned(v -> isequal(v, Float32[0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float32.true.false" => (group = :G5, baseline = Returned(v -> isequal(v, Float32[0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float32.true.true" => (group = :G5, baseline = Returned(v -> isequal(v, Float32[0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float64.false.false" => (group = :G5, baseline = Returned(v -> isequal(v, [0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float64.false.true" => (group = :G5, baseline = Returned(v -> isequal(v, [0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float64.true.false" => (group = :G5, baseline = Returned(v -> isequal(v, [0.25, -1.0, 1.0]))),
  "linear_sets.subnormal.Float64.true.true" => (group = :G5, baseline = Returned(v -> isequal(v, [0.25, -1.0, 1.0]))),
)

@testset "Dilated linear sets and subnormal searches" begin
  # #255/#376: Moreau's projection is onto gamma*C, without forming x/gamma.
  for name in ("hyperplane", "halfspace", "hyperslab"), k in (-1040, 0, 900)
    gamma = pow2(Float64, k)
    f = name == "hyperplane" ? IndHyperplane([1.0, 1.0], 1.0) :
      name == "halfspace" ? IndHalfspace([1.0, 1.0], 1.0) : IndHyperslab([1.0, 1.0], 0.0, 1.0)
    x = [0.0, 1.0]
    # All three select the upper/equality boundary at these steps, except
    # when gamma >= 1 and the inequality projections are inactive.
    ref = bigref() do
      g = BigFloat(gamma)
      t = name == "hyperplane" ? (1 - g) / 2 : max((1 - g) / 2, BigFloat(0))
      [t, t]
    end
    id = "linear_sets.conjugate.$name.$k"
    @row begin
      id = id; case = ""; family = :linear_sets; oracle = :prox
      types = "Float64"; backend = :cpu; rules = "R2 R3 T3 A1 A6"
      problem = "conjugate of sum constraint, step 2^$k: subtract projection onto gamma*C; A1/C1 pending budget 16 output ULPs"
      expected = ref
      observed = prox(Conjugate(f), x, gamma)
      compare = (v, r) -> eltype(v) == eltype(r) && all(within_ulps(v[i], r[i], 16) for i in eachindex(r))
      broken = haskey(baselines, id) ? baselines[id] : nothing
    end
  end
  # #257: project onto the capped plane, retaining the metric scale.
  for T in (Float32, Float64), diagonal in (false, true), boxed in (false, true)
    gamma = pow2(T, T === Float32 ? -140 : -1040)
    inner = boxed ? IndBox(-one(T), one(T)) : Conjugate(NormL1(one(T)))
    c = Constrained(inner, IndHyperplane(one(T), one(T)))
    x = T[0.25, -1.25, 2]
    id = "linear_sets.subnormal.$T.$diagonal.$boxed"
    @row begin
      id = id; case = ""; family = :linear_sets; oracle = :prox
      types = "$T"; backend = :cpu; rules = "P3 R2 R3 A3"
      problem = "box [-1,1] intersect sum=1 at (1/4,-5/4,2); metric has equal subnormal steps; exact projection (3/4,-3/4,1), C1 pending budget 16 output ULPs"
      expected = T[0.75, -0.75, 1]
      observed = prox(c, x, diagonal ? fill(gamma, 3) : gamma)
      compare = (v, r) -> eltype(v) == eltype(r) && all(within_ulps(v[i], r[i], 16) for i in eachindex(r))
      broken = haskey(baselines, id) ? baselines[id] : nothing
    end
  end
end

end
