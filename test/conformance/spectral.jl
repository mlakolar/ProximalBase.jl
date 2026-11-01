module ConformanceSpectralTest

using Test
using ProximalBase
using LinearAlgebra
using ..Conformance

# R1, decision 45: custom functions work while their own value stays in range.
struct QuarterMaximum <: AbstractFunction end
(::QuarterMaximum)(x) = maximum(abs, x) / 4
ProximalBase.isproximable(::Type{QuarterMaximum}) = true

# A5: provisional c = 64; the contract's constant is pending.
const c = 64

@testset "Spectral contract" begin
  f = LogDetLoss(zeros(2, 2))
  X = [0.0 pow2(Float64, 15); pow2(Float64, 15) 0.0]
  Y = zeros(Float32, 2, 2)
  @row begin
    id = "L2.value"; case = "L2"; family = :spectral; oracle = :value_and_prox
    types = "Float64 arg/data/step, Float32 destination"; backend = :cpu
    rules = "O2,O3"; problem = "Decision 39: stored output is singular"
    expected = Inf32; observed = value_and_prox!(Y, f, X, 1.0)
    broken = (group = :G6, baseline = Returned(v -> isequal(v, 0f0)))
  end
  # A5: exact roots of z^2 - t*z - 1 = 0 give diagonal sqrt(t^2+4)/2
  # and off-diagonal t/2. Keep the storage error separate from W=Float64.
  exact2, bound2 = setprecision(BigFloat, 256) do
    t = BigFloat(pow2(Float64, 15))
    d = sqrt(t^2 + 4) / 2
    point = [d t / 2; t / 2 d]
    storage = norm(BigFloat.(Float32.(point)) - point)
    point, c * 2eps(Float64) * (norm(BigFloat.(X)) + norm(point)) + storage
  end
  compare2 = (a, b) -> eltype(a) == Float32 && axes(a) == axes(b) &&
    all(isfinite, a) && norm(BigFloat.(a) - b) <= bound2
  @row begin
    id = "L2.prox"; case = "L2"; family = :spectral; oracle = :prox
    types = "Float64 arg/data/step, Float32 destination"; backend = :cpu
    rules = "A5"; problem = "c=64 pending in contract; W=Float64, bound c*p*eps(W)*(norm(X)+norm(Y_W))+norm(Float32(Y_W)-Y_W), exact closed-form Y_W"
    expected = exact2
    observed = prox!(Y, f, X, 1.0)
    compare = compare2
  end

  # A5 rank residual includes the storage error, even at an exact rank-one input.
  X3 = [1 + pow2(Float64, -28) 0; 0 0]
  @row begin
    id = "L3.prox"; case = "L3"; family = :spectral; oracle = :prox
    types = "Float64 arg, Float32 destination"; backend = :cpu
    rules = "A5"; problem = "c=64 pending in contract; best rank-one residual zero, storage norm 2^-28"
    expected = Float32[1 0; 0 0]
    observed = prox!(zeros(Float32, 2, 2), IndBallRank(1), X3, 1.0)
    compare = (a, b) -> eltype(a) == Float32 && axes(a) == axes(b) && norm(Float64.(a) - X3) <= pow2(Float64, -28) + c * 2eps(Float64) * norm(X3)
  end
  @row begin
    id = "L4.gram"; case = "L4"; family = :spectral; oracle = :prox
    types = "Float64 arg, Float32 destination"; backend = :cpu
    rules = "A5"; problem = "c=64 pending in contract; Gram bound 2e+e^2, e=c*2eps(Float64)+eps(Float32) includes storage"
    expected = 0.0
    observed = let y = prox!(zeros(Float32, 2, 1), IndStiefel(), ones(2, 1), 1.0)
      abs(sum(abs2, Float64.(y)) - 1)
    end
    compare = (a, b) -> isfinite(a) && a <= 2(c * 2eps(Float64) + eps(Float32)) + (c * 2eps(Float64) + eps(Float32))^2
  end

  polar4 = bigref(() -> fill(inv(sqrt(BigFloat(2))), 2, 1), Float32)
  compare4 = (a, b) -> eltype(a) == eltype(b) && axes(a) == axes(b) &&
    all(isfinite, a) && norm(Float64.(a) - Float64.(b)) <= 8eps(Float32)
  @row begin
    id = "L4.projection"; case = "L4"; family = :spectral; oracle = :prox
    types = "Float64 arg, Float32 destination"; backend = :cpu
    rules = "A5"; problem = "Single-column polar factor (1,1)/sqrt(2); sigma_min=norm(X)=sqrt(2), so conditioning is benign; bound 8eps(Float32) against rounded exact factor"
    expected = polar4
    observed = prox!(zeros(Float32, 2, 1), IndStiefel(), ones(2, 1), 1.0)
    compare = compare4
  end

  h = Float64(nextfloat(0f0))
  v = [1.0, 5 / 8]
  X5 = h .* v * transpose(v)
  Y5 = zeros(Float32, 2, 2)
  f5 = IndBallRank(1)
  fused5 = value_and_prox!(Y5, f5, X5, 1.0)
  @row begin
    id = "L5.value"; case = "L5"; family = :spectral; oracle = :value_and_prox
    types = "Float64 arg, Float32 destination"; backend = :cpu
    rules = "A4,A5,O2"; problem = "Independent closed-form singular values of stored 2x2 matrix; rank threshold sqrt(eps(Float32))*sigma1 in Float32 scaled units"
    expected = setprecision(BigFloat, 256) do
      scale = maximum(abs, Y5)
      if iszero(scale)
        0f0
      else
        # Eigenvalues of the symmetric Gram matrix, without LAPACK or f5.
        a, b, cc, d = BigFloat.(Y5) ./ BigFloat(scale)
        trace = a^2 + b^2 + cc^2 + d^2
        determinant = abs(a * d - b * cc)
        sigma1 = sqrt((trace + sqrt(max(zero(trace), trace^2 - 4determinant^2))) / 2)
        sigma2 = determinant / sigma1
        cutoff = sqrt(eps(Float32)) * Float32(sigma1)
        sigma2 > BigFloat(cutoff) ? Inf32 : 0f0
      end
    end
    observed = fused5
    broken = (group = :G6, baseline = Returned(v -> isequal(v, 0f0)))
  end
  @row begin
    id = "L5.residual"; case = "L5"; family = :spectral; oracle = :prox
    types = "Float64 arg, Float32 destination"; backend = :cpu
    rules = "A5,T3"; problem = "c=64 pending in contract; optimal residual zero, Float32 storage <=h (four entries each <=h/2), W=Float64"
    expected = 0.0
    observed = norm(Float64.(prox!(Y5, f5, X5, 1.0)) - X5)
    compare = (a, b) -> isfinite(a) && a <= h + c * 2eps(Float64) * norm(X5)
    broken = (group = :G6, baseline = Returned(v -> isequal(v, 1.801411492657615e-45)))
  end

  for (name, f) in (("psd", IndPSD()), ("nuclear", NuclearNorm(0.25)),
      ("trace", TracePSD(0.25)), ("singular", SingularValue(NormL1(0.25))),
      ("eigen", Spectral(NormL1(0.25))), ("rank", IndBallRank(1)),
      ("stiefel", IndStiefel()), ("logdet", LogDetLoss(zeros(2, 2))))
    Xnan = [NaN 0.0; 0.0 1.0]
    @row begin
      id = "spectral.nan-" * name * ".value"; case = ""; family = :spectral; oracle = :value
      types = "Float64"; backend = :cpu; rules = "E2,A5"
      problem = "NaN detected before LAPACK; spectral value is NaN (reverses #343)"
      expected = NaN; observed = f(Xnan)
      broken = (group = :G6, baseline = Returned(v -> isequal(v, Inf)))
    end
    @row begin
      id = "spectral.nan-" * name * ".fused"; case = ""; family = :spectral; oracle = :value_and_prox
      types = "Float64"; backend = :cpu; rules = "E2,O2,A5"
      problem = "NaN value for coupled NaN output"
      expected = NaN; observed = value_and_prox!(zeros(2, 2), f, Xnan, 1.0)
      broken = (group = :G6, baseline = Returned(v -> isequal(v, Inf)))
    end
    @row begin
      id = "spectral.nan-" * name * ".prox"; case = ""; family = :spectral; oracle = :prox
      types = "Float64"; backend = :cpu; rules = "E2,A5"
      problem = "Every entry of the coupled output depends on the NaN"
      expected = fill(NaN, 2, 2); observed = prox!(zeros(2, 2), f, Xnan, 1.0)
    end
  end

  for T in (Float32, Float64), (name, f) in (("nuclear", NuclearNorm(T(1 / 4))),
      ("trace", TracePSD(T(1 / 4))), ("singular", SingularValue(NormL1(T(1 / 4)))),
      ("eigen", Spectral(NormL1(T(1 / 4)))))
    M = floatmax(T)
    @row begin
      id = "spectral.overflow-" * name * "-" * string(T); case = ""; family = :spectral; oracle = :value
      types = string(T); backend = :cpu; rules = "R1,A5"
      problem = "Spectrum of M*ones(2,2) is (2M,0); power-of-two recovery, bound c=64 pending in contract"
      expected = M / T(2); observed = f(fill(M, 2, 2))
      compare = (a, b) -> within_bound(a, b, c * eps(T) * b)
      broken = name in ("singular", "eigen") ? (group = :G6, baseline = Returned(v -> isequal(v, T(Inf)))) : nothing
    end
  end

  # A5: analytic maps and independent Frobenius references.
  for (name, f, X, ref) in (("psd", IndPSD(), [2.0 1; 1 2], [2.0 1; 1 2]),
      ("nuclear", NuclearNorm(0.5), [2.0 1; 1 2], [1.5 1; 1 1.5]),
      ("logdet-zero", LogDetLoss(zeros(2, 2)), zeros(2, 2),
       bigref(() -> [sqrt(BigFloat(2)) BigFloat(0); BigFloat(0) sqrt(BigFloat(2))])))
    step = name == "logdet-zero" ? 2.0 : 1.0
    @row begin
      id = "spectral.nonexpansive-" * name; case = ""; family = :spectral; oracle = :prox
      types = "Float64"; backend = :cpu; rules = "A5"
      problem = "c=64 pending in contract; ||error||F <= c*p*eps(W)*(||X||F+||Y||F), D=W"
      expected = ref; observed = prox!(zeros(2, 2), f, X, step)
      compare = (a, b) -> eltype(a) == eltype(b) && axes(a) == axes(b) && all(isfinite, a) && norm(a - b) <= c * 2eps(Float64) * (norm(X) + norm(b))
    end
  end

  # T3: full argument/destination/parameter/step/data role matrix.
  for A in (Float32, Float64), D in (Float32, Float64), P in (Float32, Float64),
      S in (Float32, Float64), B in (Float32, Float64)
    W = promote_type(A, D, B)
    @row begin
      id = "spectral.roles-" * join(string.((A, D, P, S, B)), "-")
      case = ""; family = :spectral; oracle = :prox
      types = "arg=$A destination=$D parameter=$P step=$S data=$B"; backend = :cpu
      rules = "T3,A5"; problem = "LogDetLoss at zero, S=0, gamma*w=1; c=64 pending in contract, storage exact"
      expected = D[1 0; 0 1]
      observed = prox!(zeros(D, 2, 2), LogDetLoss(zeros(B, 2, 2), P(1)), zeros(A, 2, 2), S(1))
      compare = (a, b) -> eltype(a) == eltype(b) && axes(a) == axes(b) && all(isfinite, a) && norm(Float64.(a) - b) <= c * 2eps(W) * sqrt(2.0)
    end
  end
end

@testset "Spectral types and reconstruction" begin
  @row begin
    id = "spectral.custom-inner"; case = ""; family = :spectral; oracle = :value
    types = "Float64"; backend = :cpu; rules = "R1,A5"
    problem = "User-defined inner function at spectrum (4,2); A5 bound 64eps(R)*reference, constant pending"
    expected = 1.0; observed = SingularValue(QuarterMaximum())([4.0 0; 0 2])
    compare = (a, b) -> within_bound(a, b, 64eps(Float64))
  end

  for T in (Float32, Float64)
    M = floatmax(T)
    @row begin
      id = "spectral.logdet-saturation-" * string(T); case = ""; family = :spectral; oracle = :prox
      types = string(T); backend = :cpu; rules = "A5,R4"
      problem = "Decision 39: reconstruction may saturate at floatmax within normwise bound; c=64 pending in contract, normalized units"
      expected = ones(T, 2, 2)
      observed = prox!(zeros(T, 2, 2), LogDetLoss(zeros(T, 2, 2), one(T)), fill(M, 2, 2), one(T)) ./ M
      compare = (a, b) -> eltype(a) == eltype(b) && axes(a) == axes(b) && all(isfinite, a) && norm(a - b) <= 64 * 2eps(T) * 4
    end
  end

  # A5 requires checking both operands, even with a supported output type.
  for (name, make) in (("nuclear", () -> NuclearNorm(1.0)),
      ("psd", () -> IndPSD()), ("rank", () -> IndBallRank(1)),
      ("stiefel", () -> IndStiefel()), ("logdet", () -> LogDetLoss(zeros(2, 2))),
      ("singular", () -> SingularValue(NormL1(1.0))),
      ("eigen", () -> Spectral(NormL1(1.0)))), T in (Float16, BigFloat)
    @row begin
      id = "spectral.reject-argument-" * name * "-" * string(T)
      case = ""; family = :spectral; oracle = :prox
      types = "$T arg, Float64 destination"; backend = :cpu; rules = "A5,T6,T7"
      problem = "LAPACK functions reject unsupported argument element types before conversion"
      expected = Throws(:ArgumentError)
      observed = typeof(prox!(zeros(2, 2), make(), T[2 0; 0 1], 1.0))
      broken = (group = :G6, baseline = Returned(v -> v === Matrix{Float64}))
    end
    @row begin
      id = "spectral.reject-destination-" * name * "-" * string(T)
      case = ""; family = :spectral; oracle = :prox
      types = "Float64 arg, $T destination"; backend = :cpu; rules = "A5,T6,T7"
      problem = "LAPACK functions reject unsupported destination element types"
      expected = Throws(:ArgumentError)
      observed = typeof(prox!(zeros(T, 2, 2), make(), [2.0 0; 0 1], 1.0))
      broken = name == "logdet" ? (group = :G6, baseline = Returned(v -> v === Matrix{T})) : nothing
    end
  end
end

end
