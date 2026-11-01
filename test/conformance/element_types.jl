module ConformanceElementTypesTest

using Test
using ProximalBase
using ..Conformance

# T6 and O6: rejection and preservation are distinct promises.
for (caseid, A, D) in (("F16a", Float16, Float64), ("F16b", Float32, Float16))
  @row begin
    id = caseid; case = caseid; family = :element_types; oracle = :prox
    types = "argument=$A destination=$D parameter=Float64 step=Float64"; backend = :cpu
    rules = "T6 O6"; problem = "NormL1(1), x=[2], step=1: reject Float16 role"
    expected = Throws(:ArgumentError)
    observed = prox!(fill(D(17), 1), NormL1(1.0), A[2], 1.0)
    broken = (group = :G9, baseline = Returned(v -> same(v, D[1])))
  end
  @row begin
    id = caseid * ".unchanged"; case = caseid; family = :element_types; oracle = :prox
    types = "argument=$A destination=$D"; backend = :cpu
    rules = "T6 O6"; problem = "invalid Float16 role leaves sentinel output unchanged"
    expected = D[17]
    observed = let y = D[17]
      try prox!(y, NormL1(1.0), A[2], 1.0) catch e; e isa ArgumentError || rethrow(); end
      y
    end
    broken = (group = :G9, baseline = Returned(v -> same(v, D[1])))
  end
end
for (part, level, step) in (("level", Float16(1), 1f0), ("step", 1f0, Float16(1)),
                            ("weights", 1f0, 1f0), ("array-step", 1f0, Float16[1]))
  f = part == "weights" ? NormL1(level, Float16[1]) : NormL1(level)
  @row begin
    id = "F16c." * part; case = "F16c"; family = :element_types; oracle = :prox
    types = "Float32 argument/destination; Float16 $part"; backend = :cpu
    rules = "T3 T6"; problem = "exactly widened unit level, weight and step; threshold=1"
    expected = Float32[1]
    observed = prox!(zeros(Float32, 1), f, Float32[2], step)
  end
end
@row begin
  id = "element_types.direction"; case = ""; family = :element_types; oracle = :hessian_vec
  types = "Float64 argument/destination Float16 direction"; backend = :cpu
  rules = "T6 O6"; problem = "ridge Hessian-vector product rejects Float16 direction"
  expected = Throws(:ArgumentError)
  observed = hessian_vec!(zeros(1), SqrNormL2(1.0), [2.0], Float16[1])
  broken = (group = :G9, baseline = Returned(v -> same(v, [1.0])))
end
@row begin
  id = "element_types.direction-unchanged"; case = ""; family = :element_types; oracle = :hessian_vec
  types = "Float64 argument/destination Float16 direction"; backend = :cpu
  rules = "T6 O6"; problem = "rejected direction preserves sentinel"
  expected = [17.0]
  observed = let y = [17.0]
    try hessian_vec!(y, SqrNormL2(1.0), [2.0], Float16[1]) catch e; e isa ArgumentError || rethrow(); end
    y
  end
  broken = (group = :G9, baseline = Returned(v -> same(v, [1.0])))
end
@row begin
  id = "element_types.data"; case = ""; family = :element_types; oracle = :constructor
  types = "Float16 data"; backend = :cpu
  rules = "T6"; problem = "least-squares constructor rejects Float16 response and design"
  expected = Throws(:ArgumentError)
  observed = let f = LeastSquaresLoss(Float16[0], reshape(Float16[1], 1, 1)); (eltype(f.Y), eltype(f.X)); end
  broken = (group = :G9, baseline = Returned(v -> v === (Float16, Float16)))
end
@row begin
  id = "element_types.integer-destination"; case = ""; family = :element_types; oracle = :prox
  types = "Int destination Float64 argument"; backend = :cpu
  rules = "T1 O6"; problem = "fractional soft threshold rejects integer destination before work"
  expected = Throws(:ArgumentError)
  observed = prox!(zeros(Int, 1), NormL1(0.5), [1.0], 1.0)
  broken = (group = :G9, baseline = Threw(:InexactError, r"Int64\(0.5\)|Int32\(0.5\)"))
end
for A in (Int, BigInt)
  @row begin
    id = "element_types.integer-value.$A"; case = ""; family = :element_types; oracle = :value
    types = "argument=$A parameter=Float64"; backend = :cpu
    rules = "T1"; problem = "integer argument returns float(eltype(argument))"
    expected = float(A)
    observed = typeof(NormL1(0.5)(A[2]))
  end
end
# T5: validate before a common-type conversion, and retain parameter range.
@row begin
  id = "element_types.bounds-order"; case = ""; family = :element_types; oracle = :constructor
  types = "Float64 lower Float32 upper"; backend = :cpu
  rules = "T5"; problem = "lower=1+2^-30 exceeds upper=1 before rounding (#313)"
  expected = Throws(:ArgumentError, r"lower bounds")
  observed = IndBox(1.0 + pow2(Float64, -30), 1f0)
end
for k in (-600, 600)
  @row begin
    id = "element_types.parameter-range.$k"; case = ""; family = :element_types; oracle = :constructor
    types = "Float64 parameter Float32 weight"; backend = :cpu
    rules = "T5"; problem = "constructor retains finite positive level 2^$k without narrowing"
    expected = pow2(Float64, k)
    observed = NormL1(pow2(Float64, k), Float32[1]).λ0
  end
end
@row begin
  id = "element_types.nested-scaled"; case = ""; family = :element_types; oracle = :constructor
  types = "Float64 scales"; backend = :cpu
  rules = "T5 R3"; problem = "positive exact scale 2^-1200 must not acquire zero-function meaning (#333)"
  expected = false
  observed = Scaled(Scaled(NormL1(1.0), pow2(Float64, -600)), pow2(Float64, -600)) isa ZeroFunction
  broken = (group = :G9, baseline = Returned(v -> v === true))
end
# T7: rebuilding cached numbers in place is sufficient; identity is immaterial.
f = IndIntersection(IndBox(0.0, 1.0))
setprecision(BigFloat, 64) do
  prox(f, BigFloat[2], BigFloat(1))
end
@row begin
  id = "element_types.bigfloat-cache"; case = ""; family = :element_types; oracle = :prox
  types = "BigFloat precision 64 to 128"; backend = :cpu
  rules = "T7 M3"; problem = "after a 128-bit call every cached iterate, work and correction entry has precision 128; rebuilding in place is allowed"
  expected = true
  observed = setprecision(BigFloat, 128) do
    prox(f, BigFloat[2], BigFloat(1))
    all(a -> all(v -> precision(v) == 128, a),
        (f.buffers.y, f.buffers.w, f.buffers.corrections...))
  end
end
@row begin
  id = "element_types.bigfloat-cache.result-precision"; case = ""; family = :element_types; oracle = :prox
  types = "BigFloat precision 64 to 128"; backend = :cpu
  rules = "T7"; problem = "a result computed after the precision change has 128-bit entries"
  expected = 128
  observed = setprecision(BigFloat, 128) do
    precision(only(prox(f, BigFloat[2], BigFloat(1))))
  end
end
@row begin
  id = "element_types.bigfloat-cache.result"; case = ""; family = :element_types; oracle = :prox
  types = "BigFloat precision 64 to 128"; backend = :cpu
  rules = "T7 M3 A1"; problem = "projection of [2] onto [0,1] after the precision change; 128-bit rational reference and 32eps(BigFloat at 128 bits) budget, bound pending"
  expected = setprecision(BigFloat, 128) do
    BigFloat[1 // 1]
  end
  observed = setprecision(BigFloat, 128) do
    prox(f, BigFloat[2], BigFloat(1))
  end
  compare = (a, b) -> eltype(a) == BigFloat && length(a) == 1 && typed_bound(a[1], b[1], 32eps(b[1]))
end
# T8: a multiplier from either precision remains only a seed.
c = Constrained(NormL1(1.0), IndHyperplane(1.0, 1.0))
for (visit, T) in enumerate((Float32, Float64, Float32))
  @row begin
    id = "element_types.warm.$visit"
    case = ""; family = :element_types; oracle = :prox
    types = "argument/destination=$T shared Float64 multiplier"; backend = :cpu
    rules = "T8 A3"; problem = "constrained soft threshold of [3,2], sum=1; budget 32eps(T), bound pending"
    expected = T[1, 0]
    observed = prox(c, T[3, 2], T(1))
    compare = (a, b) -> eltype(a) == T && all(typed_bound(a[i], b[i], 32eps(T)) for i in eachindex(b))
  end
end


# T1--T4: vary each admitted role independently around a Float32 call.
for familyname in (:separable, :group, :ball, :matrix, :data_loss)
  roles = familyname == :data_loss ? (:base, :argument, :destination, :step, :data) :
                                    (:base, :argument, :destination, :step, :parameter)
  for role in roles
    A = role == :argument ? Float64 : Float32
    D = role == :destination ? Float64 : Float32
    P = role == :parameter ? Float64 : Float32
    S = role == :step ? Float64 : Float32
    B = role == :data ? Float64 : Float32
    f = familyname == :separable ? NormL1(P(1)) :
        familyname == :group ? GroupNormL2(P(1), [1:1]) :
        familyname == :ball ? IndBallL2(P(1)) :
        familyname == :matrix ? NuclearNorm(P(1)) :
        LeastSquaresLoss(B[0], reshape(B[1], 1, 1))
    x = familyname == :matrix ? reshape(A[2], 1, 1) : A[2]
    y = similar(x, D)
    @row begin
      id = "element_types.$familyname.$role.prox"; case = ""; family = :element_types; oracle = :prox
      types = familyname == :data_loss ? "argument=$A destination=$D step=$S data=$B" : "argument=$A destination=$D parameter=$P step=$S"; backend = :cpu
      rules = "T1 T2 T3 T4 A1 A5 A6"
      problem = "unit level/radius/design, x=[2], step=1; exact minimizer=[1]; 32eps(D) budget, bounds pending"
      expected = fill(D(1), size(x))
      observed = prox!(y, f, x, S(1))
      compare = (a, b) -> eltype(a) == D && size(a) == size(b) && all(typed_bound(a[i], b[i], 32eps(D)) for i in eachindex(b))
    end
    @row begin
      id = "element_types.$familyname.$role.value-type"; case = ""; family = :element_types; oracle = :value
      types = familyname == :data_loss ? "argument=$A data=$B" : "argument=$A parameter=$P"; backend = :cpu
      rules = "T1 T2"; problem = "standalone scalar return type follows argument, including wider data"
      expected = A
      observed = typeof(f(x))
    end
    @row begin
      id = "element_types.$familyname.$role.joint-type"; case = ""; family = :element_types; oracle = :value_and_prox
      types = familyname == :data_loss ? "argument=$A destination=$D step=$S data=$B" : "argument=$A destination=$D parameter=$P step=$S"; backend = :cpu
      rules = "T1"; problem = "joint scalar return type follows destination"
      expected = D
      observed = typeof(value_and_prox!(y, f, x, S(1)))
    end
  end
end
# T7: the generic representatives admit BigFloat; LAPACK does not.
setprecision(BigFloat, 128) do
  for (name, f) in ((:separable, NormL1(BigFloat(1))),
                    (:group, GroupNormL2(BigFloat(1), [1:1])),
                    (:ball, IndBallL2(BigFloat(1))))
    @row begin
      id = "element_types.$name.bigfloat"; case = ""; family = :element_types; oracle = :prox
      types = "BigFloat all admitted roles precision=128"; backend = :cpu
      rules = "T1 T7 A1 A6"; problem = "unit level/radius and step, x=[2]; 32eps(BigFloat) budget, bound pending"
      expected = BigFloat[1]
      observed = prox(f, BigFloat[2], BigFloat(1))
      compare = (a, b) -> eltype(a) == BigFloat && isfinite(a[1]) && abs(a[1] - b[1]) <= 32eps(BigFloat)
    end
  end
end
@row begin
  id = "element_types.matrix.bigfloat"; case = ""; family = :element_types; oracle = :value
  types = "BigFloat matrix"; backend = :cpu
  rules = "T7"; problem = "LAPACK family rejects BigFloat"
  expected = Throws(:ArgumentError)
  observed = NuclearNorm(1.0)(reshape(BigFloat[2], 1, 1))
end


# T6 applies by role in each representative family, even with a wider output.
for (name, f) in ((:group, GroupNormL2(1.0, [1:1])), (:ball, IndBallL2(1.0)),
                  (:matrix, NuclearNorm(1.0)),
                  (:data_loss, LeastSquaresLoss([0.0], reshape([1.0], 1, 1))))
  for (role, A, D) in ((:argument, Float16, Float64), (:destination, Float32, Float16))
    x = name == :matrix ? reshape(A[2], 1, 1) : A[2]
    before = fill(D(17), size(x))
    old = fill(D(1), size(x))
    spec = name == :matrix && role == :destination ? nothing :
           (group = :G9, baseline = Returned(v -> same(v, old)))
    @row begin
      id = "element_types.$name.float16-$role"; case = ""; family = :element_types; oracle = :prox
      types = "argument=$A destination=$D Float64 data/parameter/step"; backend = :cpu
      rules = "T6 O6"; problem = "Float16 $role must raise ArgumentError before writing"
      expected = Throws(:ArgumentError)
      observed = prox!(copy(before), f, x, 1.0)
      broken = spec
    end
    @row begin
      id = "element_types.$name.float16-$role-unchanged"; case = ""; family = :element_types; oracle = :prox
      types = "argument=$A destination=$D"; backend = :cpu
      rules = "T6 O6"; problem = "rejected Float16 $role preserves destination sentinel"
      expected = before
      observed = let y = copy(before)
        try prox!(y, f, x, 1.0) catch e; e isa ArgumentError || rethrow(); end
        y
      end
      broken = spec
    end
  end
end
for name in (:group, :ball, :matrix), role in (:parameter, :step)
  P = role == :parameter ? Float16 : Float32
  S = role == :step ? Float16 : Float32
  f = name == :group ? GroupNormL2(P(1), [1:1]) : name == :ball ? IndBallL2(P(1)) : NuclearNorm(P(1))
  x = name == :matrix ? reshape(Float32[2], 1, 1) : Float32[2]
  @row begin
    id = "element_types.$name.float16-$role-accepted"; case = ""; family = :element_types; oracle = :prox
    types = "Float32 argument/destination parameter=$P step=$S"; backend = :cpu
    rules = "T3 T6 A1 A5 A6"; problem = "unit parameter and step widen exactly; minimizer=[1]; 32eps(Float32) budget, bound pending"
    expected = fill(1f0, size(x))
    observed = prox!(similar(x), f, x, S(1))
    compare = (a, b) -> eltype(a) == Float32 && all(typed_bound(a[i], b[i], 32eps(Float32)) for i in eachindex(b))
  end
end
@row begin
  id = "element_types.hyperslab-bounds-order"; case = ""; family = :element_types; oracle = :constructor
  types = "Float32 normal Int64 bounds"; backend = :cpu
  rules = "T5"; problem = "#313: original integer lower=2^53+1 exceeds upper=2^53 before Float32 promotion"
  expected = Throws(:ArgumentError)
  observed = let f = IndHyperslab(1f0, Int64(2)^53 + 1, Int64(2)^53); (f.l, f.u); end
  broken = (group = :G9, baseline = Returned(v -> same(v, (pow2(Float32, 53), pow2(Float32, 53)))))
end
for T in (Float32, Float64)
  @row begin
    id = "element_types.gradient-return.$T"; case = ""; family = :element_types; oracle = :gradient
    types = "argument/destination=$T Float64 level"; backend = :cpu
    rules = "T1"; problem = "gradient! returns its caller-owned destination"
    expected = true
    observed = let g = zeros(T, 1); gradient!(g, SqrNormL2(1.0), T[2]) === g; end
  end
  @row begin
    id = "element_types.joint-gradient-type.$T"; case = ""; family = :element_types; oracle = :value_and_gradient
    types = "argument=$T destination=$(T === Float32 ? Float64 : Float32)"; backend = :cpu
    rules = "T1 O1"; problem = "joint value return type follows argument despite different gradient precision"
    expected = T
    observed = typeof(value_and_gradient!(zeros(T === Float32 ? Float64 : Float32, 1), SqrNormL2(1.0), T[2]))
  end
  @row begin
    id = "element_types.allocating-gradient.$T"; case = ""; family = :element_types; oracle = :gradient
    types = "argument=$T Float64 level"; backend = :cpu
    rules = "T1"; problem = "allocating gradient keeps float(eltype(argument))"
    expected = T
    observed = eltype(gradient(SqrNormL2(1.0), T[2]))
  end
end
setprecision(BigFloat, 128) do
  @row begin
    id = "element_types.data_loss.bigfloat"; case = ""; family = :element_types; oracle = :gradient
    types = "BigFloat data/argument/destination precision=128"; backend = :cpu
    rules = "T7 A1"; problem = "generic least-squares gradient for design=[1], response=[0], x=[2]; 32eps(BigFloat) budget"
    expected = BigFloat[2]
    observed = gradient(LeastSquaresLoss(BigFloat[0], reshape(BigFloat[1], 1, 1)), BigFloat[2])
    compare = (a, b) -> eltype(a) == BigFloat && isfinite(a[1]) && abs(a[1] - b[1]) <= 32eps(BigFloat)
  end
end

end
