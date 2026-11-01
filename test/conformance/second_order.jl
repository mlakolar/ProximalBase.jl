module ConformanceSecondOrderTest

using Test
using ProximalBase
using ..Conformance

const baselines = Dict{String,Any}(
  "second_order.hvp.Float32.Float32.Float64.Float32" => Inf32,
  "second_order.hvp.Float32.Float32.Float64.Float64" => Inf32,
  "second_order.hvp.Float64.Float32.Float64.Float32" => Inf32,
  "second_order.hvp.Float64.Float32.Float64.Float64" => Inf32,
  "second_order.lla.Float64" => 0.0,
  "second_order.model_precision.weight" => 2.7182817f0,
  "second_order.model_precision.response" => 0.0f0,
)

@testset "Second-order numerical contract" begin
  # S2/O5: restriction solves over the support, with the complement fixed at zero.
  f = QuadraticFunction([2.0 1; 1 2])
  r = restrict(f, [1])
  @row begin
    id = "Q1"; case = "Q1"; family = :second_order; oracle = :prox
    types = "Float64"; backend = :cpu; rules = "S2 O5 A1"
    problem = "restricted quadratic z^2, unit step at 1; C1 pending budget 8 ULPs"
    expected = bigref(() -> BigFloat(1) / 3)
    observed = only(prox(r, [1.0], 1.0))
    compare = (v, w) -> within_ulps(v, w, 8)
  end
  for (part, want) in ((:value, 4.0), (:gradient, 4.0))
    @row begin
      id = "second_order.restrict.$part"; case = ""; family = :second_order; oracle = part
      types = "Float64"; backend = :cpu; rules = "S2 O5 A1"
      problem = "support {1}, complement zero: r(z)=z^2, r'(z)=2z at z=2; A1/C1 pending budget 8 ULPs"
      expected = want
      observed = part == :value ? r([2.0]) : only(gradient!(zeros(1), r, [2.0]))
      compare = (v, w) -> within_ulps(v, w, 8)
    end
  end
  # O5: a centered ridge may omit the unsupported coordinate's constant 9/2.
  @row begin
    id = "second_order.restrict.constant"; case = ""; family = :second_order; oracle = :value
    types = "Float64"; backend = :cpu; rules = "O5"
    problem = "restrict ridge centered at (1,3) to {1}; removed constant is 9/2"
    expected = 4.5
    observed = begin
      h = SqrNormL2(1.0, [1.0, 3.0])
      h([2.0, 0.0]) - restrict(h, [1])([2.0])
    end
  end
  @row begin
    id = "P2"; case = "P2"; family = :second_order; oracle = :quadratic_model
    types = "Float64"; backend = :cpu; rules = "S2 O6"
    problem = "Poisson at eta=710 has unrepresentable effective coefficient exp(710)"
    expected = Throws(:PrecisionError)
    observed = quadratic_model(PoissonLoss([0.0], ones(1, 1)), [710.0])
    broken = (group = :G8, baseline = Returned(q -> isequal(q.weights, [Inf]) && isequal(q.y, [NaN])))
  end
  # P2m's later observation overflows; inspect exception and output separately.
  function update_model()
    loss = PoissonLoss(zeros(2), reshape([1.0, 710.0], 2, 1))
    q = quadratic_model(loss, [0.0])
    before = deepcopy(q)
    q, before, loss
  end
  @row begin
    id = "P2m.exception"; case = "P2m"; family = :second_order; oracle = :quadratic_model!
    types = "Float64"; backend = :cpu; rules = "S2 O6"
    problem = "second observation overflows at beta=1; PrecisionError before model writes"
    expected = Throws(:PrecisionError)
    observed = begin
      q, before, loss = update_model()
      quadratic_model!(q, loss, [1.0])
    end
    broken = (group = :G8, baseline = Returned(q -> isequal(q.weights, [exp(1.0), Inf]) && isequal(q.y, [0.0, NaN])))
  end
  @row begin
    id = "P2m.unchanged"; case = "P2m"; family = :second_order; oracle = :quadratic_model!
    types = "Float64"; backend = :cpu; rules = "O6"
    problem = "all fields of caller-owned q equal their deep copy after failed update"
    expected = true
    observed = begin
      q, before, loss = update_model()
      try
        quadratic_model!(q, loss, [1.0])
      catch e
        nameof(typeof(e)) == :PrecisionError || rethrow()
      end
      all(isequal(getfield(q, k), getfield(before, k)) for k in fieldnames(typeof(q)))
    end
    broken = (group = :G8, baseline = Returned(v -> isequal(v, false)))
  end
  for part in (:weight, :response)
    @row begin
      id = "P3.$part"; case = "P3"; family = :second_order; oracle = :quadratic_model
      types = "Float64"; backend = :cpu; rules = "S2 R2 R3 A1"
      problem = "Poisson eta=710, weight=1/8; effective coefficient exp(710)/8, response 709; A1/C1 pending budget 16 ULPs"
      expected = part == :weight ? bigref(() -> exp(BigFloat(710)) / 8) : 709.0
      observed = begin
        q = quadratic_model(PoissonLoss([0.0], ones(1, 1); weights = [1/8]), [710.0])
        part == :weight ? only(q.weights) : only(q.y)
      end
      compare = part == :weight ? ((v, w) -> within_ulps(v, w, 16)) : same
      broken = (group = :G8, baseline = Returned(v -> isequal(v, part == :weight ? Inf : NaN)))
    end
  end
  @row begin
    id = "second_order.kkt.opposing_infinities"; case = ""; family = :second_order; oracle = :kkt_violation
    types = "Float64 symmetric iterate"; backend = :cpu; rules = "S2 E3 M2"
    problem = "off-diagonal symmetric gradient is (Inf + -Inf)/2, hence NaN"
    expected = NaN
    observed = kkt_violation(NormL1(1.0), [0.0 Inf; -Inf 0.0], SymmetricSparseIterate(2), 2)
    broken = (group = :G8, baseline = Threw(:ArgumentError, r"gradient g has no symmetric part"))
  end
  # T3: direction precision participates even with a narrow destination.
  for X in (Float32, Float64), D in (Float32, Float64), V in (Float32, Float64), P in (Float32, Float64)
    k = V === Float32 ? 100 : 150
    v = pow2(V, k)
    id = "second_order.hvp.$X.$D.$V.$P"
    @row begin
      id = id; case = ""; family = :second_order; oracle = :hessian_vec!
      types = "$X argument/$D destination/$V direction/$P level"; backend = :cpu
      rules = "T3 R2 A1"
      problem = "ridge Hessian product 2^-100 * 2^k, exact power of two; direction is not narrowed before multiplication"
      expected = pow2(D, k - 100)
      observed = only(hessian_vec!(zeros(D, 1), SqrNormL2(pow2(P, -100)), zeros(X, 1), V[v]))
      compare = same
      broken = haskey(baselines, id) ? (group = :G8, baseline = Returned(z -> isequal(z, baselines[id]))) : nothing
    end
  end
  for D in (Float32, Float64)
    id = "second_order.lla.$D"
    @row begin
      id = id; case = ""; family = :second_order; oracle = :lla_weights!
      types = "Float32 argument/$D destination/Float64 level"; backend = :cpu
      rules = "T3 R2 A1"
      problem = "MCP level 1+2^-28, a=2, x=2: slope max(level-1,0); A1/C1 pending budget 2 ULPs"
      expected = D === Float32 ? 0f0 : pow2(D, -28)
      observed = only(lla_weights!(zeros(D, 1), MCP(1 + pow2(Float64, -28), 2.0), Float32[2]))
      compare = same
      broken = haskey(baselines, id) ? (group = :G8, baseline = Returned(z -> isequal(z, baselines[id]))) : nothing
    end
  end
  for part in (:weight, :response)
    id = "second_order.model_precision.$part"
    @row begin
      id = id; case = ""; family = :second_order; oracle = :quadratic_model
      types = "Float64 argument/Float32 data"; backend = :cpu; rules = "T2 T3 A1"
      problem = "Poisson eta=1+2^-28 with unit data; A1/C1 pending budget 16 Float64 ULPs on model coefficient and response"
      expected = part == :weight ? bigref(() -> exp(BigFloat(1) + BigFloat(pow2(Float64, -28)))) : pow2(Float64, -28)
      observed = begin
        q = quadratic_model(PoissonLoss(Float32[0], ones(Float32, 1, 1)), [1 + pow2(Float64, -28)])
        part == :weight ? only(q.weights) : only(q.y)
      end
      compare = (z, r) -> typeof(z) == typeof(r) && within_ulps(z, r, 16)
      broken = haskey(baselines, id) ? (group = :G8, baseline = Returned(z -> isequal(z, baselines[id]))) : nothing
    end
  end
end

end
