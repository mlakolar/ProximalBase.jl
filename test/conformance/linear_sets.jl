module ConformanceLinearSetsTest

using Test
using ProximalBase
using ..Conformance

const baselines = Dict{String,Any}(
  "linear_sets.wider.ProximalBase.IndHyperplane.scalar" => [0.6666666567325592, -0.3333333432674408, -0.3333333432674408],
  "linear_sets.wider.ProximalBase.IndHyperplane.array" => [0.6666666567325592, -0.3333333432674408, -0.3333333432674408],
  "linear_sets.wider.ProximalBase.IndHalfspace.scalar" => [0.6666666567325592, -0.3333333432674408, -0.3333333432674408],
  "linear_sets.wider.ProximalBase.IndHalfspace.array" => [0.6666666567325592, -0.3333333432674408, -0.3333333432674408],
  "linear_sets.wider.ProximalBase.IndHyperslab.scalar" => [0.6666666269302368, -0.3333333432674408, -0.3333333432674408],
  "linear_sets.wider.ProximalBase.IndHyperslab.array" => [0.6666666269302368, -0.3333333432674408, -0.3333333432674408],
  "linear_sets.coefficient.ProximalBase.IndHyperplane.false" => Float32[7.888609f-31],
  "linear_sets.coefficient.ProximalBase.IndHyperplane.true" => Float32[7.888609f-31],
  "linear_sets.coefficient.ProximalBase.IndHalfspace.false" => Float32[7.888609f-31],
  "linear_sets.coefficient.ProximalBase.IndHalfspace.true" => Float32[7.888609f-31],
)
components(v, r, n) = eltype(v) == eltype(r) && all(within_ulps(v[i], r[i], n) for i in eachindex(r))


# A3: certify the stored point as well as the face decisions. For unit normals,
# products and cached normals are exact, and fixed zero entries add no error.
# gamma_4 bounds the sum and offset subtraction; u*sum(abs, zstar) accounts
# for final Float32 storage. C1: this is the stated test budget, not a general
# termination bound for every search.
function certified_decision(v, expected, b)
  v === :precision_error && return true
  v isa Tuple{AbstractVector{Float32}, NTuple{3, Bool}} || return false
  z, decisions = v
  axes(z) == axes(expected[1]) && all(isfinite, z) || return false
  all(t -> 0f0 <= t <= 1f0, z) && same(decisions, expected[2]) || return false
  u = pow2(Float64, -24)
  gamma4 = nextfloat(4u / (1 - 4u))
  Z = BigFloat.(z)
  bound = BigFloat(gamma4) * (sum(abs, Z) + abs(BigFloat(b))) +
    BigFloat(u) * sum(abs, BigFloat.(expected[1]))
  # BigFloat evaluates the certificate on stored entries without Float32 loss.
  abs(sum(Z) - BigFloat(b)) <= bound
end

@testset "Linear geometry and searches" begin
  # T2/decision 46(b): normals and offsets are data, including in a one-row affine set.
  a = [1.0, 1 + pow2(Float64, -25), 1.0]
  x = Float32[pow2(Float32, 25), -pow2(Float32, 25), 0]
  ref = bigref(; T = Float32) do
    A, X = BigFloat.(a), BigFloat.(x)
    t = sum(A .* X) / sum(abs2, A)
    X .- t .* A
  end
  for name in ("hyperplane", "halfspace", "hyperslab", "affine", "constrained"), diagonal in (false, true)
    name == "affine" && diagonal && continue  # IndAffine admits scalar steps.
    f = name == "hyperplane" ? IndHyperplane(a, 0.0) :
      name == "halfspace" ? IndHalfspace(-a, 0.0) :
      name == "hyperslab" ? IndHyperslab(a, 0.0, 1.0) :
      name == "affine" ? IndAffine(reshape(a, 1, 3), [0.0]) :
      Constrained(NormL1(0f0), IndHyperplane(a, 0.0))
    step = diagonal ? ones(Float32, 3) : 1f0
    id = "linear_sets.normal_data.$name.$(diagonal ? "array" : "scalar")"
    @row begin
      id = id; case = ""; family = :linear_sets; oracle = :prox
      types = "Float32 argument/destination/Float64 normal and offset"; backend = :cpu
      rules = "T2 T3 A1 A6"
      problem = "46(b): residual -1 with stored normal (1,1+2^-25,1); third coordinate 1/(3+2^-24+2^-50); A1/C1 pending budget 16 output ULPs per component"
      expected = ref
      observed = prox!(zeros(Float32, 3), f, x, step)
      compare = (v, r) -> components(v, r, 16)
      broken = name == "constrained" ? (group = :G5, baseline = Returned(v -> same(v, x))) : nothing
    end
  end
  # T4: recompute the coefficient in D, rather than widening a Float32 coefficient.
  for F in (IndHyperplane, IndHalfspace, IndHyperslab), diagonal in (false, true)
    f = F === IndHyperslab ? F(Float32[1, 1, 1], -1f0, 0f0) : F(Float32[1, 1, 1], 0f0)
    id = "linear_sets.wider.$F.$(diagonal ? "array" : "scalar")"
    @row begin
      id = id; case = ""; family = :linear_sets; oracle = :prox
      types = "Float32 argument/data/Float64 destination"; backend = :cpu
      rules = "T3 T4 A1 A6"
      problem = "project (1,0,0) onto sum=0 (upper slab bound); A1/C1 pending budget 8 Float64 ULPs"
      expected = bigref(() -> BigFloat[2//3, -1//3, -1//3])
      observed = prox!(zeros(3), f, Float32[1, 0, 0], diagonal ? ones(Float32, 3) : 1f0)
      compare = (v, r) -> components(v, r, 8)
      # A6: Float32 normal reductions can move a few Float32 ULPs in Float64 output.
      broken = haskey(baselines, id) ? (group = :G5, baseline = Returned(near(baselines[id], 16 << 29)), volatile = true) : nothing
    end
  end
  # #218: the coefficient 2^-200 underflows in Float32, its correction does not.
  for F in (IndHyperplane, IndHalfspace), diagonal in (false, true)
    id = "linear_sets.coefficient.$F.$diagonal"
    @row begin
      id = id; case = ""; family = :linear_sets; oracle = :prox
      types = "Float32 argument/destination/Float64 normal"; backend = :cpu
      rules = "T2 T3 R2 R3"
      problem = "normal 2^100, offset 0, argument 2^-100; coefficient 2^-200 times normal is 2^-100"
      expected = Float32[0]
      observed = prox!(zeros(Float32, 1), F(pow2(Float64, 100), 0.0), Float32[pow2(Float32, -100)], diagonal ? Float32[1] : 1f0)
      broken = (group = :G5, baseline = Returned(v -> same(v, baselines[id])))
    end
  end
  # #363: integer input sums must not wrap before conversion.
  @row begin
    id = "linear_sets.integer_sum"; case = ""; family = :linear_sets; oracle = :prox
    types = "Int64 argument/Float64 destination/Int64 normal"; backend = :cpu
    rules = "T2 T3 R3 A1 A6"
    problem = "equal typemax(Int) coordinates on sum=0; A1/C1 pending budget 8 eps(Float64)*typemax(Int) per coordinate"
    expected = [0.0, 0.0]
    observed = prox(IndHyperplane(1, 0), [typemax(Int), typemax(Int)], 1)
    compare = (v, r) -> all(typed_bound(v[i], r[i], 8BigFloat(eps(Float64)) * BigFloat(typemax(Int))) for i in eachindex(r))
    broken = (group = :G5, baseline = Returned(v -> same(v, fill(Float64(typemax(Int)), 2))))
  end
  # A3: near a box face, refine the decision or raise PrecisionError.
  # The exact solution has two free coordinates 1-2^-23 and one fixed zero.
  for F in (IndHyperplane, IndHalfspace, IndHyperslab), diagonal in (false, true)
    b = 2f0 - pow2(Float32, -22)
    set = F === IndHyperslab ? F(1f0, 0f0, b) : F(1f0, b)
    c = Constrained(IndBox(0f0, 1f0), set)
    @row begin
      id = "linear_sets.certify.$F.$diagonal"; case = ""; family = :linear_sets; oracle = :decision
      types = "Float32"; backend = :cpu; rules = "A3 O6"
      problem = "sum bound 2-2^-22 at (1,1,-1): finite Float32 point in [0,1], two free entries, one fixed; A3/C1 certificate |sum(z)-b| <= gamma4*(sum|z|+|b|)+u*sum|zstar|, u=2^-24, gamma4=4u/(1-4u); refine or raise PrecisionError"
      expected = (Float32[1 - pow2(Float32, -23), 1 - pow2(Float32, -23), 0], (false, false, true))
      observed = try
        z = prox(c, Float32[1, 1, -1], diagonal ? ones(Float32, 3) : 1f0)
        (z, (z[1] == 1f0, z[2] == 1f0, z[3] == 0f0))
      catch e
        nameof(typeof(e)) == :PrecisionError || rethrow()
        :precision_error
      end
      compare = (v, r) -> certified_decision(v, r, b)
      broken = (group = :G5, baseline = Returned(v -> same(v, (Float32[1, 1, 0], (true, true, true)))))
    end
  end
  # A3: a well-separated active bound admits the independently derived solution.
  for T in (Float32, Float64), F in (IndHyperplane, IndHalfspace, IndHyperslab), diagonal in (false, true)
    set = F === IndHyperslab ? F(one(T), -one(T), zero(T)) : F(one(T), zero(T))
    @row begin
      id = "linear_sets.certified_far.$T.$F.$diagonal"; case = ""; family = :linear_sets; oracle = :prox
      types = "$T"; backend = :cpu; rules = "A3 A1 A6"
      problem = "ridge level 1 at (3,1), sum bound 0, unit metric gives (1/2,-1/2); A1/C1 pending budget 16 output ULPs"
      expected = T[0.5, -0.5]
      observed = prox(Constrained(SqrNormL2(one(T)), set), T[3, 1], diagonal ? ones(T, 2) : one(T))
      compare = (v, r) -> components(v, r, 16)
    end
  end
  # #323/O6: an empty intersection discovered during a later block's search.
  function infeasible()
    c = Constrained(Constrained(SqrNormL2(), IndBox(0, 1)), IndHyperplane(1, 5))
    SeparableSum(SqrNormL2(2.0), c)
  end
  @row begin
    id = "linear_sets.infeasible.exception"; case = ""; family = :linear_sets; oracle = :prox
    types = "Float64 tuple blocks"; backend = :cpu; rules = "O6"
    problem = "later constrained block has two entries in [0,1] but sum=5"
    expected = Throws(:ArgumentError, r"misses the domain")
    observed = prox!(([19.0], [19.0, 19.0]), infeasible(), ([3.0], [0.5, 0.5]), 1.0)
  end
  @row begin
    id = "linear_sets.infeasible.unchanged"; case = ""; family = :linear_sets; oracle = :prox
    types = "Float64 tuple blocks"; backend = :cpu; rules = "O6"
    problem = "infeasibility during search must leave every earlier block unchanged"
    expected = ([19.0], [19.0, 19.0])
    observed = begin
      y = ([19.0], [19.0, 19.0])
      try
        prox!(y, infeasible(), ([3.0], [0.5, 0.5]), 1.0)
      catch e
        e isa ArgumentError && occursin("misses the domain", sprint(showerror, e)) || rethrow()
      end
      y
    end
    broken = (group = :G5, baseline = Returned(v -> same(v, ([1.0], [19.0, 19.0]))))
  end
end

end
