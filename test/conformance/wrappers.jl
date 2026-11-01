module ConformanceWrappersTest

using Test
using ProximalBase
using LinearAlgebra
using ..Conformance

@testset "Wrapper contract" begin
  M = floatmax(Float64)
  @row begin
    id = "W1"; case = "W1"; family = :wrappers; oracle = :value
    types = "Float64"; backend = :cpu; rules = "R1"
    problem = "Translate norm recovers x-b=2M through homogeneity; reduction bound 16eps(R)*reference"
    expected = M / 2; observed = Translate(NormL1(0.25), [-M])([M])
    compare = (a, b) -> within_bound(a, b, 16eps(Float64) * b)
    broken = (group = :G7, baseline = Returned(v -> isequal(v, Inf)))
  end
  @row begin
    id = "wrappers.precompose-homogeneity"; case = ""; family = :wrappers; oracle = :value
    types = "Float64"; backend = :cpu; rules = "R1"
    problem = "Precompose norm recovers 2M through homogeneity; reduction bound 16eps(R)*reference"
    expected = M / 2; observed = Precompose(NormL1(0.25), 2.0)([M])
    compare = (a, b) -> within_bound(a, b, 16eps(Float64) * b)
    broken = (group = :G7, baseline = Returned(v -> isequal(v, Inf)))
  end

  t = Translate(NormL1(1.0), [1e8])
  @row begin
    id = "wrappers.staged-l1.prox"; case = ""; family = :wrappers; oracle = :prox
    types = "Float64 arg/parameter/step, Float32 destination"; backend = :cpu
    rules = "O4"; problem = "Shift, inner map and backshift round in Float32"
    expected = Float32[0]; observed = prox!(zeros(Float32, 1), t, [0.5], 1.0)
  end
  @row begin
    id = "wrappers.staged-l1.entry"; case = ""; family = :wrappers; oracle = :prox_entry
    types = "Float64"; backend = :cpu; rules = "O4"
    problem = "Unstaged scalar map: b + ((x-b)+1) = 1.5"
    expected = 1.5; observed = prox_entry(t, 0.5, 1, 1.0)
  end
  @row begin
    id = "wrappers.staged-l1.bound"; case = ""; family = :wrappers; oracle = :prox
    types = "Float64 arg/parameter/step, Float32 destination"; backend = :cpu
    rules = "O4"; problem = "O4 bound eps(Float32)*1e8; stage amplification one"
    expected = 1.5
    observed = Float64(only(prox!(zeros(Float32, 1), t, [0.5], 1.0)))
    compare = (a, b) -> isfinite(a) && abs(a - b) <= eps(Float32) * 1e8
  end
  for D in (Float32, Float64)
    @row begin
      id = "wrappers.staged-l0-" * string(D); case = ""; family = :wrappers; oracle = :prox
      types = "Float64 arg/parameter/step, $D destination"; backend = :cpu
      rules = "O4,A2"; problem = "Discontinuous inner map: staged Float32 tie selects zero; Float64 stays above tie"
      expected = D === Float32 ? D[-pow2(Float64, -26)] : D[1]
      observed = prox!(zeros(D, 1), Translate(NormL0(0.5), [-pow2(Float64, -26)]), [1.0], 1.0)
    end
  end

  # R1: an inner value can overflow at a representable argument.
  @row begin
    id = "wrappers.scaled-inner-overflow"; case = ""; family = :wrappers; oracle = :value
    types = "Float64"; backend = :cpu; rules = "R1"
    problem = "Scaled translated ridge: 2^-1000*(2^600)^2/2 = 2^199; 16 ULP R1 budget"
    expected = pow2(Float64, 199)
    observed = Scaled(Translate(SqrNormL2(1.0), [0.0]), pow2(Float64, -1000))([pow2(Float64, 600)])
    compare = (a, b) -> within_ulps(a, b, 16)
    broken = (group = :G7, baseline = Returned(v -> isequal(v, Inf)))
  end
  @row begin
    id = "wrappers.scaled-no-level"; case = ""; family = :wrappers; oracle = :value
    types = "Float64"; backend = :cpu; rules = "R1"
    problem = "Scaled MoreauEnvelope has no level to fold; envelope at (M,M) is 2M-1; 32eps(R) reduction budget"
    expected = bigref(() -> (2BigFloat(floatmax(Float64)) - 1) / 4)
    observed = Scaled(MoreauEnvelope(NormL1(1.0), 1.0), 0.25)([M, M])
    compare = (a, b) -> within_bound(a, b, 32eps(Float64) * b)
    broken = (group = :G7, baseline = Returned(v -> isequal(v, Inf)))
  end

  for (name, f, value, point) in (
      ("tilt", Tilt(SqrNormL2(0.5), [0.25], 0.125), 41 / 8, [5 / 2]),
      ("regularize", Regularize(NormL1(0.25), 0.5, [1.0]), 13 / 4, [17 / 6]),
      ("linearimage", LinearImage(NormL1(0.25), reshape([2.0], 1, 1), 4.0), 1 / 2, [31 / 8]))
    @row begin
      id = "wrappers." * name * ".value"; case = ""; family = :wrappers; oracle = :value
      types = "Float64"; backend = :cpu; rules = "T3,R1"
      problem = "Analytic wrapper at x=4; R1 reduction bound 32eps(R)*reference"
      expected = value; observed = f([4.0])
      compare = (a, b) -> within_bound(a, b, 32eps(Float64) * b)
    end
    @row begin
      id = "wrappers." * name * ".prox"; case = ""; family = :wrappers; oracle = :prox
      types = "Float64"; backend = :cpu; rules = "T3,R2,O4"
      problem = "Unit-step scalar minimizer; R2/O4 bound 32eps(D)*max(1,|point|)"
      expected = point; observed = prox!(zeros(1), f, [4.0], 1.0)
      compare = (a, b) -> eltype(a) == Float64 && all(within_ulps(x, y, 32) for (x, y) in zip(a, b))
    end
  end

  # E5: membership recomputation must retain an exact Inf or NaN.
  for q in (Inf, NaN)
    @row begin
      id = "wrappers.precompose-exact-" * (isnan(q) ? "nan" : "inf")
      case = ""; family = :wrappers; oracle = :value_and_prox
      types = "Float64 arg/parameter, Float32 destination/step"; backend = :cpu
      rules = "E5,E2"; problem = "#350: keep exact non-finite inner value in mixed-precision Precompose"
      expected = Float32(q)
      observed = value_and_prox!(zeros(Float32, 2, 2), Precompose(IndPSD(), 2.0), [q 0.0; 0.0 1.0], 1f0)
      broken = (group = :G2, baseline = Returned(v -> isequal(v, 0f0)))
    end
  end

  # T3: LinearImage holds data L; its inner level and step are parameters.
  for A in (Float32, Float64), D in (Float32, Float64), P in (Float32, Float64),
      S in (Float32, Float64), B in (Float32, Float64)
    W = promote_type(A, D, B)
    @row begin
      id = "wrappers.roles-" * join(string.((A, D, P, S, B)), "-")
      case = ""; family = :wrappers; oracle = :prox
      types = "arg=$A destination=$D parameter=$P step=$S data=$B"; backend = :cpu
      rules = "T3,R2,A1"; problem = "L=2, inner level=1/4, x=4, step=1: soft threshold 1/8; A1 bound 32eps(W)*4"
      expected = D[31 / 8]
      observed = prox!(zeros(D, 1), LinearImage(NormL1(P(1 / 4)), reshape(B[2], 1, 1), B(4)), A[4], S(1))
      compare = (a, b) -> eltype(a) == D && all(isfinite, a) && norm(Float64.(a) - b) <= 32eps(W) * 4
    end
  end

  @notcovered :wrappers :value "Float32/Float64" :cpu "R1: overflowing transformed arguments of non-homogeneous inner functions in Translate have no supported transformation"
  @notcovered :wrappers :value "Float32/Float64" :cpu "R1: overflowing transformed arguments of non-homogeneous inner functions in Precompose have no supported transformation"
  @notcovered :wrappers :value "Float32/Float64" :cpu "R1: overflowing transformed arguments in LinearImage have no supported transformation"
end

end
