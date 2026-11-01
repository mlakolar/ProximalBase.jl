module ConformanceDataFunctions2Test

using Test
using ProximalBase
using ..Conformance

const baselines = Dict{String,Any}(
  "data_functions.affine.Float32.Float32.Float64.prox" => [0.40064454078674316, -0.20037102699279785],
  "data_functions.affine.Float32.Float64.Float64.prox" => [0.3998633101582527, -0.19998037815093994],
  "data_functions.graph.Float32.Float32.Float64.prox" => [0.00036612158874049783, 0.00036621096660383046],
  "data_functions.graph.Float32.Float64.Float32.prox" => Float32[-0.00012204053, -0.00012207033],
  "data_functions.graph.Float32.Float64.Float64.prox" => [-0.00012204053200548515, -0.00012207032705191523],
  "data_functions.polyhedron.Float32.Float32.Float64.prox" => [0.4006444811820984, -0.20037102699279785],
  "data_functions.polyhedron.Float32.Float64.Float64.prox" => [0.3998633027076721, -0.19998037815093994],
  "data_functions.image.Float32.Float32.Float64.prox" => [0.33376747369766235],
  "data_functions.image.Float32.Float64.Float64.prox" => [0.3334418535232544],
  "data_functions.image.parameter_Float32.step_Float32" => 0.800000011920929,
  "data_functions.image.parameter_Float32.step_Float64" => 0.800000011920929,
  "data_functions.image.parameter_Float64.step_Float32" => 0.800000011920929,
  "data_functions.image.parameter_Float64.step_Float64" => 0.800000011920929,
)

@testset "Data set and image precision" begin
  # T2/T3: all four data-bearing projection families, every data/argument/output type.
  for name in ("affine", "graph", "polyhedron", "image"), U in (Float32, Float64),
      X in (Float32, Float64), D in (Float32, Float64)
    a = U(1) + pow2(U, U === Float32 ? -12 : -28)
    A = reshape(U[a, 2], 1, 2)
    t = X(1) + pow2(X, X === Float32 ? -10 : -27)
    f, x, y = if name == "affine"
      (IndAffine(A, U[0]), X[t, 1], zeros(D, 2))
    elseif name == "polyhedron"
      (IndPolyhedron(A, U[0], U[0]), X[t, 1], zeros(D, 2))
    elseif name == "graph"
      (IndGraph(reshape(U[a], 1, 1)), (X[t], X[-1]), (zeros(D, 1), zeros(D, 1)))
    else
      (LinearImage(QuadraticFunction(reshape(U[2], 1, 1)), reshape(U[a], 1, 1)), X[t], zeros(D, 1))
    end
    ref = bigref(; T = D) do
      b, z = BigFloat(a), BigFloat(t)
      if name in ("affine", "polyhedron")
        r = (b * z + 2) / (b^2 + 4)
        [z - b * r, 1 - 2r]
      elseif name == "graph"
        r = (z - b) / (1 + b^2)
        [r, b * r]
      else
        [z * b^2 / (b^2 + 2)]
      end
    end
    id = "data_functions.$name.$U.$X.$D.prox"
    @row begin
      id = id; case = ""; family = :data_functions; oracle = :prox
      types = "$X argument/$D destination/$U data/Float64 step"; backend = :cpu
      rules = "T2 T3 T4 A1 A6"
      problem = "$name with stored a=$a and x=$t; A1/C1 pending budget 64 output ULPs per component (one-row/direct solve)"
      expected = ref
      observed = begin
        prox!(y, f, x, 1.0)
        name == "graph" ? [only(y[1]), only(y[2])] : copy(y)
      end
      compare = (v, r) -> eltype(v) == eltype(r) && all(within_ulps(v[i], r[i], 64) for i in eachindex(r))
      broken = haskey(baselines, id) ? (group = :G4, baseline = Returned(v -> isequal(v, baselines[id]))) : nothing
    end
  end
  # T3 type x role matrix: LinearImage admits scalar steps and an inner level.
  for P in (Float32, Float64), S in (Float32, Float64)
    id = "data_functions.image.parameter_$P.step_$S"
    @row begin
      id = id; case = ""; family = :data_functions; oracle = :prox
      types = "Float32 argument/Float64 destination/Float32 data/$P parameter/$S step"; backend = :cpu
      rules = "T3 T4 A1"
      problem = "image of ridge under identity, level 3, step 1/2; A1/C1 pending budget 16 output ULPs"
      expected = bigref(() -> BigFloat(2) / (1 + BigFloat(3) / 2))
      observed = only(prox!(zeros(1), LinearImage(SqrNormL2(P(3)), ones(Float32, 1, 1), 1f0), Float32[2], S(0.5)))
      compare = (v, r) -> within_ulps(v, r, 16)
      broken = (group = :G4, baseline = Returned(v -> isequal(v, baselines[id])))
    end
  end
end

end
