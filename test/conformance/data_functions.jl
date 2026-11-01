module ConformanceDataFunctionsTest

using Test
using ProximalBase
using ..Conformance

# Baselines below are measured at d92b8d8; never used to construct references.
const baselines = Dict{String,Any}(
  "data_functions.least_squares.Float32.Float32.Float64.gradient" => 0.0012212395668029785,
  "data_functions.least_squares.Float32.Float64.Float32.value" => 2.9802322387695312e-8,
  "data_functions.least_squares.Float32.Float64.Float32.gradient" => 0.00024420023f0,
  "data_functions.least_squares.Float32.Float64.Float64.value" => 2.9802322387695312e-8,
  "data_functions.least_squares.Float32.Float64.Float64.gradient" => 0.0002442002296447754,
  "data_functions.glm.Float32.Float32.Float64.gradient" => 0.0012212395668029785,
  "data_functions.glm.Float32.Float64.Float32.value" => 2.9802322387695312e-8,
  "data_functions.glm.Float32.Float64.Float32.gradient" => 0.00024420023f0,
  "data_functions.glm.Float32.Float64.Float64.value" => 2.9802322387695312e-8,
  "data_functions.glm.Float32.Float64.Float64.gradient" => 0.0002442002296447754,
  "data_functions.logistic.Float32.Float32.Float64.gradient" => -0.26876702904701233,
  "data_functions.logistic.Float32.Float64.Float32.value" => 0.3131960332393646,
  "data_functions.logistic.Float32.Float64.Float64.value" => 0.3131960332393646,
  "data_functions.logistic.Float32.Float64.Float64.gradient" => -0.268959105014801,
  "data_functions.poisson.Float32.Float32.Float64.gradient" => 1.7220230102539062,
  "data_functions.poisson.Float32.Float64.Float32.value" => 1.7187013626098633,
  "data_functions.poisson.Float32.Float64.Float64.value" => 1.7187013626098633,
  "data_functions.poisson.Float32.Float64.Float64.gradient" => 1.719365119934082,
  "data_functions.huber.Float32.Float32.Float64.gradient" => 0.0012212395668029785,
  "data_functions.huber.Float32.Float64.Float32.value" => 2.9802322387695312e-8,
  "data_functions.huber.Float32.Float64.Float32.gradient" => 0.00024420023f0,
  "data_functions.huber.Float32.Float64.Float64.value" => 2.9802322387695312e-8,
  "data_functions.huber.Float32.Float64.Float64.gradient" => 0.0002442002296447754,
  "data_functions.quadratic.Float32.Float64.Float32.value" => -0.4998779371362616,
  "data_functions.quadratic.Float32.Float64.Float32.gradient" => 0.00024414062f0,
  "data_functions.quadratic.Float32.Float64.Float64.value" => -0.4998779371362616,
  "data_functions.quadratic.Float64.Float32.Float32.gradient" => 0.0009765625f0,
  "data_functions.quadratic.Float64.Float64.Float32.gradient" => 0.0f0,
  "data_functions.kron.Float32.Float64.Float32.value" => -0.4998779296875,
  "data_functions.kron.Float32.Float64.Float32.gradient" => 0.00024414062f0,
  "data_functions.kron.Float32.Float64.Float64.value" => -0.4998779296875,
  "data_functions.kron.Float32.Float64.Float64.gradient" => 0.000244140625,
  "data_functions.logdet.Float32.Float32.Float64.gradient" => 0.0012197494506835938,
  "data_functions.logdet.Float32.Float64.Float32.value" => 1.0002441555297992,
  "data_functions.logdet.Float32.Float64.Float32.gradient" => 0.00024414062f0,
  "data_functions.logdet.Float32.Float64.Float64.value" => 1.0002441555297992,
  "data_functions.logdet.Float32.Float64.Float64.gradient" => 0.000244140625,
)

function smooth_rows(name, U, X, D)
  a = U(1) + pow2(U, U === Float32 ? -12 : -25)
  t = X(1) + pow2(X, X === Float32 ? -10 : -26)
  A = reshape(U[a], 1, 1)
  f = if name == "least_squares"
    LeastSquaresLoss(U[1], A)
  elseif name == "glm"
    GLMLoss(ProximalBase.Squared(), U[1], A)
  elseif name == "logistic"
    LogisticLoss(U[1], A)
  elseif name == "poisson"
    PoissonLoss(U[1], A)
  elseif name == "huber"
    HuberLoss(U[1], A, U(2))
  elseif name == "quadratic"
    QuadraticFunction(A, U[-1])
  elseif name == "kron"
    KronQuadratic(A, ones(U, 1, 1), fill(U(-1), 1, 1))
  else
    LogDetLoss(A)
  end
  x = name in ("kron", "logdet") ? reshape(X[t], 1, 1) : X[t]
  g = zeros(D, size(x))
  # T2/T3: exact stored data, argument widened to W, Huber level converted to W.
  reference(part) = bigref(; T = part == :value ? X : D) do
    b, z = BigFloat(a), BigFloat(t)
    r = b * z - 1
    if name in ("least_squares", "glm", "huber")
      part == :value ? r^2 / 2 : b * r
    elseif name == "logistic"
      part == :value ? log1p(exp(-b * z)) : -b / (1 + exp(b * z))
    elseif name == "poisson"
      part == :value ? exp(b * z) - b * z : b * (exp(b * z) - 1)
    elseif name in ("quadratic", "kron")
      part == :value ? b * z^2 / 2 - z : r
    else
      part == :value ? b * z - log(z) : b - inv(z)
    end
  end
  for part in (:value, :gradient)
    id = "data_functions.$name.$U.$X.$D.$part"
    W = promote_type(U, X, part == :value ? X : D)
    # A1/C1 pending bound: 16 output ULPs for ordinary smooth outputs;
    # cancellation in affine gradients adds 8 eps(W) times operand scale.
    cancellation = part == :gradient && name in ("least_squares", "glm", "huber", "quadratic", "kron", "logdet")
    cmp = cancellation ? ((v, r) -> within_bound(v, r,
      8BigFloat(eps(W)) * (abs(BigFloat(a) * BigFloat(t)) + 1) + 2BigFloat(eps(D)) * abs(BigFloat(r)))) :
      ((v, r) -> within_ulps(v, r, 16))
    # A6: retain Float32-product rounding latitude even when stored in Float64.
    # near counts output ULPs, so 16 data-precision ULPs need 2^(p_out-p_data).
    broken_spec = nothing
    if haskey(baselines, id)
      old = baselines[id]
      if iszero(old)
        broken_spec = (group = :G4, baseline = Returned(v -> same(v, old)))
      else
        n = 16 << max(precision(typeof(old)) - precision(U), 0)
        broken_spec = (group = :G4, baseline = Returned(near(old, n)), volatile = true)
      end
    end
    @row begin
      id = id; case = ""; family = :data_functions; oracle = part
      types = "$X argument/$D destination/$U data"; backend = :cpu
      rules = "T2 T3 A1 R1"
      problem = "$name: a=$a, x=$t, response/linear term=1; C1 pending budget 16 output ULPs, or A1 cancellation budget 8 eps(W)(|ax|+1) + 2 eps(D)|reference|"
      expected = reference(part)
      observed = part == :value ? f(x) : only(gradient!(g, f, x))
      compare = cmp
      broken = broken_spec
    end
  end
end

@testset "Data model precision" begin
  @row begin
    id = "V12"; case = "V12"; family = :data_functions; oracle = :gradient
    types = "Float64 argument/Float32 destination/Float32 data"; backend = :cpu
    rules = "T3"
    problem = "A=Float32[1], b=Float32[-1], x=1+2^-28; subtract before storing gradient"
    expected = pow2(Float32, -28)
    observed = only(gradient!(zeros(Float32, 1), QuadraticFunction(Float32[1;;], Float32[-1]), [1 + pow2(Float64, -28)]))
    broken = (group = :G4, baseline = Returned(v -> same(v, 0f0)))
  end
  for name in ("least_squares", "glm", "logistic", "poisson", "huber", "quadratic", "kron", "logdet"),
      U in (Float32, Float64), X in (Float32, Float64), D in (Float32, Float64)
    smooth_rows(name, U, X, D)
  end
end

end
