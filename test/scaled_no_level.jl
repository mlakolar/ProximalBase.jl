module ScaledNoLevelTest

# Scaled(f, α) for functions without a level to fold the scale into (R1, "Wrappers"; #283):
# when f(x) leaves the range while α f(x) is a normal number, the value comes from the value of
# f as a significand and a power of two, with α applied last. References are BigFloat
# evaluations of the functions; the bounds are those registered on #608: 4 ulps of R for a
# closed form, (n + 4) eps(R) Σ|terms| for a sum of n terms, one more ulp of Float32 for a
# Float32 result with a Float64 factor, and an absolute floatmin(R) for a subnormal reference.

using Test, ProximalBase, SparseArrays, LinearAlgebra

pow2(T, k) = ldexp(one(T), k)
const H = pow2(Float64, 600)
const M = floatmax(Float64)

bigval(f, x::AbstractArray) = setprecision(() -> f(big.(x)), BigFloat, 512)
ref(f, α, x) = setprecision(() -> big(α) * f(big.(x)), BigFloat, 512)

# v (of type R) agrees with the reference r within the registered bound; S is the scale Σ|terms|
function agrees(v, r, n, S = abs(r); closed = false, wide = false)
  R = typeof(v)
  isfinite(v) || return false
  a = abs(r)
  a < floatmin(R) && return abs(big(v) - r) <= floatmin(R)
  bound = closed ? 4 * eps(R(a)) : (n + 4) * eps(R) * S
  wide && (bound += eps(R(a)))
  abs(big(v) - r) <= bound
end

# (name, f, x, number of terms, closed form, Σ|terms| as a function of f and x or nothing)
const F64 = [
  ("SqrNormL2", SqrNormL2(1.0), [H, -H / 3], 2, false, nothing),
  ("SqrNormL2 y", SqrNormL2(1.0, [1.0, 2.0]), [H, -H / 3], 2, false, nothing),
  ("SqrNormL2 big level", SqrNormL2(1e300), [1e10, -3e10], 2, false, nothing),
  ("CubeNormL2", CubeNormL2(1.0), [pow2(Float64, 400), pow2(Float64, 400)], 2, true, nothing),
  ("SqrDistL2", SqrDistL2(IndBox(-1.0, 1.0)), [H, -H / 3], 2, false, nothing),
  ("DistL2", DistL2(IndBox(-1.0, 1.0), 1e308), [10.0, 10.0], 2, true, nothing),
  ("NormL1Fused", NormL1Fused(1.0, 1.0), [1e308, -1e308, 3.0], 6, false, nothing),
  ("NormL1Fused pair", NormL1Fused(1.0, 1.0), [1e308, -1e308], 4, false, nothing),
  ("TotalVariation1D", TotalVariation1D(1.0), [1e308, -1e308], 2, false, nothing),
  ("FusedLasso", FusedLasso(1.0, 1.0), [1e308, -1e308], 4, false, nothing),
  ("SLOPE", SLOPE([2.0, 1.0]), [1e308, 1e308], 2, false, nothing),
  ("SumLargest", SumLargest(2, 1.0), [1e308, 1e308], 2, false, nothing),
  ("NegEntropy", NegEntropy(), [1e308, 1e308], 2, false, nothing),
  ("KLDivergence", KLDivergence([1.0, 1.0]), [1e308, 1e308], 6, false,
    (f, x) -> sum(t -> abs(t * log(t)) + t + 1, big.(x))),
  ("NegLog", NegLog(1e308), [1e-10, 1e-10], 2, false, nothing),
  ("NegLog negative", NegLog(1e308), [1e10], 1, false, nothing),
  ("Berhu", Berhu(1.0, 1.0), [H, -H / 3], 4, false, nothing),
  ("HingeLoss", HingeLoss(), [-1e308, -1e308], 2, false, nothing),
  ("HingeLoss y w", HingeLoss([2.0, -2.0], [3.0, 1.0]), [-1e308, 1e308], 2, false, nothing),
  ("SqrHingeLoss", SqrHingeLoss(), [-H, -H], 2, false, nothing),
  ("EpsInsensitiveLoss", EpsInsensitiveLoss(0.1), [1e308, 1e308], 2, false, nothing),
  ("CheckLoss", CheckLoss(0.9), [M, M], 2, false, nothing),
  ("NormL0", NormL0(1e308), [1.0, 1.0], 2, true, nothing),
  ("CappedL1", CappedL1(1e308, 1.0), [5.0, 5.0], 2, false, nothing),
  ("LogPenalty", LogPenalty(1e308, 1.0), [1e300, 1e300], 2, false, nothing),
  ("NormLq", NormLq(1e300, 0.5), [1e308, 1e308], 2, false, nothing),
  ("MCP", MCP(1e200, 3.0), [1e300, 1e300], 2, false, nothing),
  ("MCP middle", MCP(1e300, 3.0), [1e10], 2, false, (f, x) -> big(1e300) * big(1e10) + big(1e10)^2 / 6),
  ("SCAD", SCAD(1e200, 3.7), [1e300, 1e300], 2, false, nothing),
  ("SCAD middle", SCAD(1e150, 1e10), [1e154], 3, false,
    (f, x) -> (2 * big(1e10) * big(1e150) * big(1e154) + big(1e154)^2 + big(1e150)^2) / (2 * (big(1e10) - 1))),
  ("GroupMCP", GroupMCP(1e200, [[1, 2], [3]]; a = 3.0), [1e300, 1e300, 1e300], 4, false, nothing),
  ("GroupSCAD", GroupSCAD(1e200, [[1, 2], [3]]), [1e300, 1e300, 1e300], 6, false, nothing),
  ("Translate SqrNormL2", Translate(SqrNormL2(1.0), [1.0, 2.0]), [H, -H / 3], 2, false, nothing),
  ("Translate HingeLoss", Translate(HingeLoss(), [1.0, 1.0]), [-1e308, -1e308], 2, false, nothing),
]

const F32 = [
  ("SqrNormL2", SqrNormL2(1f0), Float32[pow2(Float32, 70), -pow2(Float32, 70) / 3], 2),
  ("HingeLoss", HingeLoss(), Float32[-3f38, -3f38], 2),
  ("TotalVariation1D", TotalVariation1D(1f0), Float32[3f38, -3f38], 2),
  ("SLOPE", SLOPE(Float32[2, 1]), Float32[3f38, 3f38], 2),
  ("NegLog", NegLog(1f38), Float32[1f-10, 1f-10], 2),
  ("NormL0", NormL0(3f38), Float32[1, 1], 2),
  ("KLDivergence", KLDivergence(Float32[1, 1]), Float32[3f38, 3f38], 6),
  ("MCP", MCP(1f20, 3f0), Float32[1f30, 1f30], 2),
  ("GroupSCAD", GroupSCAD(1f20, [[1, 2]]), Float32[1f30, 1f30], 3),
  ("CheckLoss", CheckLoss(0.9f0), Float32[3f38, 3f38], 2),
]

@testset "overflow: Float64" begin
  for (name, f, x, n, closed, terms) in F64, α in (pow2(Float64, -600), 0.3 * pow2(Float64, -600), pow2(Float64, -900))
    @testset "$name α = $α" begin
      v0 = f(x)
      @test isinf(v0)
      r = ref(f, α, x)
      @test floatmin(Float64) <= abs(r) <= M
      S = terms === nothing ? abs(r) : big(α) * terms(f, x)
      v = Scaled(f, α)(x)
      @test v isa Float64
      @test agrees(v, r, n, S; closed)
      @test signbit(v) == signbit(v0)
    end
  end
end

@testset "overflow: Float32" begin
  for (name, f, x, n) in F32, α in (pow2(Float32, -100), 0.3f0 * pow2(Float32, -100), pow2(Float64, -100))
    @testset "$name α = $α" begin
      @test isinf(f(x))
      r = ref(f, α, x)
      @test floatmin(Float32) <= abs(r) <= floatmax(Float32)
      v = Scaled(f, α)(x)
      @test v isa Float32
      @test agrees(v, r, n; wide = α isa Float64)
    end
  end
  # a Float32 argument with a Float64 level: the level is not rounded below the result's range
  f, x, α = SqrNormL2(1.0), Float32[pow2(Float32, 100)], pow2(Float64, -150)
  @test Scaled(f, α)(x) === Float32(big(α) * big(pow2(Float64, 200)) / 2)
end

@testset "underflow" begin
  α = pow2(Float64, 1000)
  t = pow2(Float64, -600)
  for (name, f, x, n) in (
      ("SqrNormL2", SqrNormL2(1.0), [t], 1),
      ("SqrNormL2 two", SqrNormL2(1.0), [t, 3t], 2),
      ("CubeNormL2", CubeNormL2(1.0), [pow2(Float64, -400)], 1),
      ("SqrDistL2", SqrDistL2(IndBox(0.0, 0.0)), [t, -t], 2),
      ("DistL2", DistL2(IndBox(0.0, 0.0), t), [t], 1),
      ("CheckLoss", CheckLoss(0.5), [nextfloat(0.0), 3 * nextfloat(0.0)], 2),
      ("SqrHingeLoss", SqrHingeLoss(nothing, [pow2(Float64, -1070)]), [0.5], 1),
      ("HingeLoss", HingeLoss(nothing, [pow2(Float64, -1070), pow2(Float64, -1070)]), [0.5, 0.25], 2),
      ("EpsInsensitiveLoss", EpsInsensitiveLoss(0.0, [t]), [t], 1),
      ("CappedL1", CappedL1(1e-300, 1.0), [1e-300, 2e-300], 2),
      ("NormLq", NormLq(1e-300, 0.5), [1e-300], 1),
      ("LogPenalty", LogPenalty(t, 1.0), [t], 1),
      ("NormL0", NormL0(1e-310), [1.0, 2.0], 2),
      ("MCP", MCP(t, 3.0), [t], 2),
      ("SCAD", SCAD(t, 3.7), [2t], 3),
      ("GroupMCP", GroupMCP(t, [[1, 2]]; a = 3.0), [t, t], 2),
      ("NegLog", NegLog(pow2(Float64, -1070)), [0.5, 0.25], 2),
      ("KLDivergence", KLDivergence([1.0]; λ = pow2(Float64, -1070)), [2.0], 3),
      ("NegEntropy", NegEntropy(pow2(Float64, -1070)), [3.0], 1),
      ("Berhu", Berhu(t, 1.0), [t], 1),
      ("TotalVariation1D", TotalVariation1D(1.0), [pow2(Float64, -1070), pow2(Float64, -1060)], 1),
      ("FusedLasso", FusedLasso(1.0, 1.0), [pow2(Float64, -1070), pow2(Float64, -1060)], 3),
      ("NormL1Fused", NormL1Fused(1.0, 1.0), [pow2(Float64, -1070), pow2(Float64, -1060), 0.0], 6),
      ("SLOPE", SLOPE([1.0, 0.5]), [pow2(Float64, -1070), pow2(Float64, -1072)], 2),
      ("SumLargest", SumLargest(1, t), [t, 0.0], 1),
      ("Translate SqrNormL2", Translate(SqrNormL2(1.0), [t]), [3t], 1))
    @testset "$name" begin
      r = ref(f, α, x)
      @test floatmin(Float64) <= abs(r) <= M
      @test !(floatmin(Float64) <= abs(f(x)))
      @test agrees(Scaled(f, α)(x), r, n)
      if issmooth(typeof(f))
        g = zeros(length(x))
        @test agrees(value_and_gradient!(g, Scaled(f, α), x), r, n)
      end
    end
  end
end

@testset "oracles" begin
  for (f, x) in ((SqrNormL2(1.0), [H, -H / 3]), (HingeLoss(), [-1e308, -1e308]), (SqrHingeLoss(), [-H, -H]),
                 (TotalVariation1D(1.0), [1e308, -1e308]), (NegLog(1e308), [1e-10, 1e-10]),
                 (GroupMCP(1e200, [[1, 2]]; a = 3.0), [1e300, 1e300]))
    α = pow2(Float64, -600)
    s = Scaled(f, α)
    @testset "$(nameof(typeof(f)))" begin
      if issmooth(typeof(f))
        g = zeros(length(x))
        @test agrees(value_and_gradient!(g, s, x), ref(f, α, x), 4)
      end
      # value_and_prox! with a scalar step: the value at the output
      y = zeros(length(x))
      v = value_and_prox!(y, s, x, 1.0)
      r = ref(f, α, y)
      floatmin(Float64) <= abs(r) <= M && @test agrees(v, r, 4)
      # an array step outside the band (the separable functions take one)
      if f isa ProximalBase.SeparableFunction
        y = zeros(length(x))
        v = value_and_prox!(y, s, x, ones(length(x)))
        r = ref(f, α, y)
        floatmin(Float64) <= abs(r) <= M && @test agrees(v, r, 4)
      end
      # a chain of two factors whose product is not a normal number
      c = Scaled(Scaled(f, pow2(Float64, -600)), pow2(Float64, -500))
      xc = 2 .* x
      r = ref(f, pow2(Float64, -1100), xc)
      floatmin(Float64) <= abs(r) <= M && @test agrees(c(xc), r, 4)
    end
  end
  # an array step inside the band: overflow of α f at the output is recovered (#359 keeps underflow)
  h = HingeLoss()
  for γ in (1.0, [1.0, 1.0])
    y = zeros(2)
    v = value_and_prox!(y, Scaled(h, 0.25), [-M, -M], γ)
    @test agrees(v, ref(h, 0.25, y), 2)
    @test isapprox(v, M / 2; rtol = 1e-15)
  end
  # a folded function on the array step inside the band: Inf before, now M/2 (the levels fold first)
  y = zeros(2)
  v = value_and_prox!(y, Scaled(NormL1(1.0), 0.25), [M, M], [1.0, 1.0])
  @test agrees(v, ref(NormL1(1.0), 0.25, y), 2)
  # nested chains, and a Scaled inside a Translate inside a Scaled
  @test agrees(Scaled(Scaled(SqrNormL2(1e300), 1e-200), 1e-200)([1e10]), big(1e300) * big(1e-200) * big(1e-200) * big(1e10)^2 / 2, 1)
  @test isapprox(Scaled(Scaled(SqrNormL2(1e300), 1e-200), 1e-200)([1e10]), 5e-81; rtol = 1e-14)
  @test gradient(Scaled(Scaled(SqrNormL2(1e300), 1e-200), 1e-200), [1e10]) == [Inf]   # #610
  w = Scaled(Translate(Scaled(SqrNormL2(1.0), 0.5), [0.0]), pow2(Float64, -1000))
  @test w([pow2(Float64, 600)]) === pow2(Float64, 198)
  w = Scaled(Translate(Scaled(HingeLoss(), 0.5), [0.0, 0.0]), pow2(Float64, -600))
  @test agrees(w([-1e308, -1e308]), ref(HingeLoss(), 0.5 * pow2(Float64, -600), [-1e308, -1e308]), 2)
end

@testset "argument kinds" begin
  α = pow2(Float64, -600)
  @test agrees(Scaled(SqrNormL2(1e300), 1e-300)([10^9, 10^9]), big(1e300) * big(1e-300) * 10^18, 2)
  h = HingeLoss(nothing, [1e300, 1e300])
  @test agrees(Scaled(h, 1e-300)([-10^9, -10^9]), ref(h, 1e-300, [-10^9, -10^9]), 2)
  xs = SparseIterate([H, 0.0, -H / 3])
  @test agrees(Scaled(SqrNormL2(1.0), α)(xs), ref(SqrNormL2(1.0), α, [H, 0.0, -H / 3]), 3)
  xs = SparseIterate([-1e308, 0.0, -1e308])
  @test agrees(Scaled(HingeLoss(), α)(xs), ref(HingeLoss(), α, [-1e308, 0.0, -1e308]), 3)
  A = [H 1.0; 1.0 -H]
  xs = SymmetricSparseIterate(sparse(A))
  @test agrees(Scaled(SqrNormL2(1.0), α)(xs), ref(SqrNormL2(1.0), α, A), 4)
  A = [-1e308 1.0; 1.0 -1e308]
  xs = SymmetricSparseIterate(sparse(A))
  @test agrees(Scaled(HingeLoss(), α)(xs), ref(HingeLoss(), α, A), 4)
  # Tuple arguments: no recovery, the outcome of the base
  @test Scaled(SeparableSum((SqrNormL2(1.0), HingeLoss())), α)(([H], [-1e308])) === Inf
  @test Scaled(NormL1Fused(1.0, 1.0), α)(([1e308], [-1e308])) === Inf
  @test Scaled(Sum(SqrNormL2(1.0), SqrNormL2(1.0)), α)([H]) === Inf
  # other array types keep the ordinary result
  @test Scaled(SqrNormL2(1.0), α)(transpose([H, -H / 3])) === Inf
end

@testset "unchanged outcomes" begin
  @testset "the trigger does not fire" begin
    for (name, f, x, _, _, _) in F64, α in (0.5, 2.0, 1e-3)
      for z in (x ./ 1e300, [0.25 * (-1)^i for i in eachindex(x)] .+ 1.5)
        v = f(z)
        (isfinite(v) && !iszero(v) && floatmin(Float64) <= abs(α * v) <= M) || continue
        @test Scaled(f, α)(z) === α * v
      end
    end
  end
  @testset "infinite and NaN values" begin
    @test Scaled(NegLog(1.0), 0.5)([-1.0, 2.0]) === Inf
    @test Scaled(NegLog(1.0), 0.5)([NaN, 2.0]) === Inf
    @test Scaled(NegEntropy(), 0.5)([-1.0, 2.0]) === Inf
    @test Scaled(HingeLoss(nothing, [Inf, 1.0]), 0.5)([0.0, 0.0]) === Inf
    @test Scaled(TotalVariation1D(1.0), 0.5)([Inf, 1.0]) === Inf
    @test Scaled(CappedL1(1.0, 1.0, [Inf, 1.0]), 0.5)([1.0, 1.0]) === Inf
    @test isnan(Scaled(SqrNormL2(1.0), pow2(Float64, -600))([NaN, H]))
    @test isnan(Scaled(MCP(1e200, 1e200), 1e-300)([1e308, 1e308]))
    @test isnan(Scaled(NegLog(1e308), 1e-300)([1e-10, 1e10]))
    # a value infinite in extended range stays infinite
    @test Scaled(SqrNormL2(1.0), 0.5)([M, M]) === Inf
  end
  @testset "exact zeros and folded families" begin
    @test Scaled(SqrNormL2(1.0), 2.0)(zeros(5)) === 0.0
    # an ordinary value zero by underflow: recomputed when α can lift it into the normal range
    x = [pow2(Float64, -540)]
    @test SqrNormL2(1.0)(x) === 0.0
    @test Scaled(SqrNormL2(1.0), pow2(Float64, 60))(x) === pow2(Float64, -1021)
    @test abs(Scaled(SqrNormL2(1.0), 2.0)(x) - pow2(Float64, -1080)) <= floatmin(Float64)
    @test Scaled(HingeLoss(), 10.0)(fill(2.0, 5)) === 0.0
    # unweighted hinge-type losses: a small value is exact by the type, and stays as computed
    @test ProximalBase._zero_value_exact(HingeLoss()) && ProximalBase._zero_value_exact(SqrHingeLoss()) &&
      ProximalBase._zero_value_exact(EpsInsensitiveLoss(0.1))
    @test !ProximalBase._zero_value_exact(HingeLoss(nothing, [1.0])) && !ProximalBase._zero_value_exact(SqrNormL2(1.0))
    @test Scaled(HingeLoss([2.0, -1.0]), pow2(Float64, 1000))([1.0, -3.0]) === 0.0
    @test Scaled(EpsInsensitiveLoss(0.5), pow2(Float64, 1000))([0.25, -0.5]) === 0.0
    @test Scaled(EpsInsensitiveLoss(0.0), pow2(Float64, 100))([nextfloat(0.0)]) === pow2(Float64, -974)
    # with weights, an underflowed value is still recovered
    w = pow2(Float64, -1070)
    @test agrees(Scaled(HingeLoss(nothing, [w]), pow2(Float64, 1000))([0.5]), big(w) / 2 * big(2.0)^1000, 1)
    @test agrees(Scaled(SqrHingeLoss(nothing, [w]), pow2(Float64, 1000))([0.5]), big(w) / 4 * big(2.0)^1000, 1)
    g = zeros(5)
    @test value_and_gradient!(g, Scaled(SqrHingeLoss(), 10.0), fill(2.0, 5)) === 0.0
    @test agrees(Scaled(NormL1(1.0), pow2(Float64, -600))([1e308, 1e308]), ref(NormL1(1.0), pow2(Float64, -600), [1e308, 1e308]), 2)
  end
  @testset "ordinary recoveries keep their bits" begin
    # values recovered by the functions themselves, pinned at the base
    @test TotalVariation1D(0.5)([M, -M]) === 1.7976931348623157e308
    @test FusedLasso(0.25, 0.25)([M, -M]) === 1.7976931348623157e308
    @test NormL1Fused(0.125, 0.125)([M, -M, 0.5 * M]) === 1.4606256720756315e308
    @test NormL1Fused(0.25, 0.125)([M, -M]) === 1.3482698511467367e308
    @test SLOPE([0.5, 0.25])([M, M]) === 1.3482698511467367e308
    @test SLOPE([pow2(Float64, -1000), 1.0e-310])([M, M]) === 1.677721601797693e7
    @test SumLargest(2, 0.5)([M, M, 1.0]) === 1.7976931348623157e308
    @test SumLargest(2, 0.5)([M, -M, 1.0]) === 8.988465674311579e307
    @test TotalVariation1D(pow2(Float64, -1000))([M, -M, 3.0]) === 5.033164799999999e7
  end
  @testset "a gradient output over a parameter" begin
    x = [pow2(Float64, -600), 3.0]
    c = [0.0, 3.0]
    @test value_and_gradient!(c, Scaled(SqrNormL2(1.0, c), pow2(Float64, 1000)), x) === 0.0
    @test c == [pow2(Float64, 400), 0.0]
    c = [0.0, 3.0]
    g = zeros(2)
    @test agrees(value_and_gradient!(g, Scaled(SqrNormL2(1.0, c), pow2(Float64, 1000)), x),
                 ref(SqrNormL2(1.0, [0.0, 3.0]), pow2(Float64, 1000), x), 2)
  end
end

@testset "exact-zero protection" begin
  t = pow2(Float64, -600)
  # an ordinary value that is zero by underflow, which the factor brings into the normal range
  for (name, f, x, α, n) in (
      ("SqrNormL2 level", SqrNormL2(pow2(Float64, -900)), [pow2(Float64, -200)], pow2(Float64, 600), 1),
      ("SqrNormL2 squares", SqrNormL2(1.0), [pow2(Float64, -540)], pow2(Float64, 60), 1),
      ("SqrNormL2 sum", SqrNormL2(1.0), fill(pow2(Float64, -538), 4), pow2(Float64, 100), 4),
      ("SqrNormL2 level above one", SqrNormL2(pow2(Float64, 100)), [t], pow2(Float64, 200), 1),
      ("SqrNormL2 level below one", SqrNormL2(pow2(Float64, -1000)), [pow2(Float64, -40)], pow2(Float64, 100), 1),
      ("SqrNormL2 center", SqrNormL2(pow2(Float64, -1000), [1.0]), [1.0 + pow2(Float64, -40)], pow2(Float64, 100), 1),
      ("SqrNormL2 center level above one", SqrNormL2(pow2(Float64, 100), [t]), [2t], pow2(Float64, 200), 1),
      ("CubeNormL2", CubeNormL2(pow2(Float64, -1000)), [pow2(Float64, -30)], pow2(Float64, 100), 1),
      ("DistL2", DistL2(IndBox(0.0, 0.0), pow2(Float64, -1070)), [pow2(Float64, -10)], pow2(Float64, 100), 1),
      ("SqrDistL2", SqrDistL2(IndBox(0.0, 0.0), pow2(Float64, -1000)), [pow2(Float64, -40)], pow2(Float64, 100), 1),
      ("NormL1Fused", NormL1Fused(pow2(Float64, -1070), pow2(Float64, -1070)), [pow2(Float64, -10), 0.0, 0.0], pow2(Float64, 100), 6),
      ("TotalVariation1D", TotalVariation1D(pow2(Float64, -1070)), [0.0, pow2(Float64, -10)], pow2(Float64, 100), 1),
      ("FusedLasso", FusedLasso(pow2(Float64, -1070), pow2(Float64, -1070)), [0.0, pow2(Float64, -10)], pow2(Float64, 100), 3),
      ("SLOPE", SLOPE([pow2(Float64, -1070), pow2(Float64, -1070)]), [pow2(Float64, -10), 0.0], pow2(Float64, 100), 2),
      ("SumLargest", SumLargest(1, pow2(Float64, -1070)), [pow2(Float64, -10), 0.0], pow2(Float64, 100), 1),
      ("NegEntropy", NegEntropy(pow2(Float64, -1000)), [pow2(Float64, -80)], pow2(Float64, 100), 1),
      ("KLDivergence level product", KLDivergence([1e-300]; λ = pow2(Float64, -800)), [0.0], pow2(Float64, 1000), 1),
      ("NegLog", NegLog(pow2(Float64, -1060)), [1.0 - pow2(Float64, -40)], pow2(Float64, 100), 1),
      ("Berhu", Berhu(pow2(Float64, -1070), 1.0), [pow2(Float64, -10)], pow2(Float64, 100), 1),
      ("HingeLoss w", HingeLoss(nothing, [pow2(Float64, -1070)]), [1.0 - pow2(Float64, -20)], pow2(Float64, 100), 1),
      ("SqrHingeLoss w", SqrHingeLoss(nothing, [pow2(Float64, -1000)]), [1.0 - pow2(Float64, -40)], pow2(Float64, 100), 1),
      ("EpsInsensitiveLoss w", EpsInsensitiveLoss(0.25, [pow2(Float64, -1070)]), [0.25 + pow2(Float64, -20)], pow2(Float64, 100), 1),
      ("CheckLoss w", CheckLoss(0.5, [pow2(Float64, -1070)]), [pow2(Float64, -20)], pow2(Float64, 100), 1),
      ("CheckLoss", CheckLoss(0.5), [nextfloat(0.0)], pow2(Float64, 100), 1),
      ("CappedL1", CappedL1(pow2(Float64, -1070), 1.0), [pow2(Float64, -10)], pow2(Float64, 100), 1),
      ("CappedL1 w", CappedL1(pow2(Float64, -600), 1.0, [pow2(Float64, -600)]), [0.5], pow2(Float64, 200), 1),
      ("LogPenalty", LogPenalty(pow2(Float64, -1070), 1.0), [pow2(Float64, -10)], pow2(Float64, 100), 1),
      ("LogPenalty w", LogPenalty(1.0, 1.0, [pow2(Float64, -1070)]), [pow2(Float64, -10)], pow2(Float64, 100), 1),
      ("NormLq", NormLq(pow2(Float64, -1070), 0.5), [pow2(Float64, -40)], pow2(Float64, 100), 1),
      ("NormLq w", NormLq(pow2(Float64, -600), 0.5, [pow2(Float64, -600)]), [4.0], pow2(Float64, 200), 1),
      ("MCP", MCP(t, 3.0), [t], pow2(Float64, 1000), 2),
      ("SCAD", SCAD(t, 3.7), [t / 2], pow2(Float64, 1000), 1),
      ("GroupMCP", GroupMCP(t, [[1, 2]]; a = 3.0), [t, t], pow2(Float64, 1000), 2),
      ("GroupSCAD", GroupSCAD(t, [[1, 2]]), [t, 0.0], pow2(Float64, 1000), 2),
      ("CubeNormL2 cube", CubeNormL2(1.0), [pow2(Float64, -400)], pow2(Float64, 1000), 1),
      ("NegEntropy last product", NegEntropy(pow2(Float64, -1022)), [prevfloat(1.0)], pow2(Float64, 100), 1),
      ("KLDivergence t log(t/q)", KLDivergence([2 * nextfloat(0.0)]; λ = pow2(Float64, 100)), [nextfloat(0.0)], 2.0, 3),
      ("Berhu quadratic piece", Berhu(pow2(Float64, -300), pow2(Float64, -500)), [pow2(Float64, -400)], 2.0, 2),
      ("Berhu square of M", Berhu(1.0, pow2(Float64, -600)), [pow2(Float64, -590)], 2.0, 2),
      ("LogPenalty ratio", LogPenalty(1.0, pow2(Float64, 60)), [pow2(Float64, -1020)], pow2(Float64, 100), 1),
      ("LogPenalty coefficient", LogPenalty(pow2(Float64, -600), 1.0, [pow2(Float64, -600)]), [1.0], pow2(Float64, 1000), 1),
      ("MCP far piece", MCP(t, 3.0), [1.0], pow2(Float64, 1000), 1),
      ("SCAD middle piece", SCAD(t, 3.7), [2t], pow2(Float64, 1000), 3),
      ("SCAD far piece", SCAD(t, 3.7), [1.0], pow2(Float64, 1000), 1),
      ("GroupMCP far piece", GroupMCP(t, [[1, 2]]; a = 3.0), [1.0, 1.0], pow2(Float64, 1000), 1),
      ("Translate", Translate(SqrNormL2(pow2(Float64, -1000)), [1.0]), [1.0 + pow2(Float64, -40)], pow2(Float64, 600), 1),
      ("Scaled∘Translate∘Scaled", Translate(Scaled(SqrNormL2(pow2(Float64, -900)), 2.0), [0.0]), [pow2(Float64, -200)], pow2(Float64, 600), 1))
    @testset "hidden: $name" begin
      @test iszero(f(x))
      r = ref(f, α, x)
      @test floatmin(Float64) <= abs(r) <= M
      @test agrees(Scaled(f, α)(x), r, n)
    end
  end
  # a level that does not keep its range in Float32 is rounded to zero by the ordinary value
  @test iszero(NormL0(1e-50)(Float32[1, 0]))
  @test agrees(Scaled(NormL0(1e-50), 1e30)(Float32[1, 0]), big(1e-50) * big(1e30), 1; wide = true)
  # an exact zero stays zero, whether the factor lets the bound decide or the value is recomputed
  for (name, f, x) in (
      ("SqrNormL2", SqrNormL2(1.0), zeros(3)), ("SqrNormL2 y", SqrNormL2(1.0, [1.0, 2.0]), [1.0, 2.0]),
      ("CubeNormL2", CubeNormL2(1.0), zeros(3)), ("DistL2", DistL2(IndBox(-1.0, 1.0), 2.0), [0.5, -0.5]),
      ("SqrDistL2", SqrDistL2(IndBox(-1.0, 1.0)), [0.5, -0.5]), ("NormL1Fused", NormL1Fused(1.0, 1.0), zeros(3)),
      ("TotalVariation1D", TotalVariation1D(1.0), fill(3.0, 4)), ("FusedLasso", FusedLasso(1.0, 1.0), zeros(3)),
      ("SLOPE", SLOPE([2.0, 1.0]), zeros(2)), ("SumLargest", SumLargest(2, 1.0), zeros(3)),
      ("NegEntropy", NegEntropy(), ones(3)), ("KLDivergence", KLDivergence([2.0, 3.0]), [2.0, 3.0]),
      ("NegLog", NegLog(1.0), ones(3)), ("Berhu", Berhu(1.0, 1.0), zeros(3)), ("Berhu w", Berhu(1.0, 1.0, [1.0, 2.0]), zeros(2)),
      ("HingeLoss", HingeLoss(), fill(2.0, 3)), ("HingeLoss w", HingeLoss(nothing, [1.0, 2.0]), fill(2.0, 2)),
      ("SqrHingeLoss", SqrHingeLoss(), fill(2.0, 3)), ("EpsInsensitiveLoss", EpsInsensitiveLoss(0.5), [0.25, -0.5]),
      ("CheckLoss", CheckLoss(0.3), zeros(3)), ("CheckLoss w", CheckLoss(0.3, [1.0, 2.0]), zeros(2)),
      ("NormL0", NormL0(1.0), zeros(3)), ("CappedL1", CappedL1(1.0, 0.5), zeros(3)), ("LogPenalty", LogPenalty(1.0, 0.5), zeros(3)),
      ("LogPenalty w", LogPenalty(1.0, 0.5, [1.0, 2.0]), zeros(2)), ("NormLq", NormLq(1.0, 0.5), zeros(3)),
      ("NormLq w", NormLq(1.0, 0.5, [1.0, 2.0]), zeros(2)), ("MCP", MCP(1.0, 3.0), zeros(3)), ("SCAD", SCAD(1.0, 3.7), zeros(3)),
      ("GroupMCP", GroupMCP(1.0, [[1, 2], [3]]; a = 3.0), zeros(3)), ("GroupSCAD", GroupSCAD(1.0, [[1, 2], [3]]), zeros(3)),
      ("Translate", Translate(SqrNormL2(1.0), [1.0, 2.0]), [1.0, 2.0]))
    @testset "exact: $name" begin
      @test iszero(f(x))
      for α in (2.0, pow2(Float64, 1000), 2f0)
        v = Scaled(f, α)(x)
        @test iszero(v) && v isa Float64
      end
    end
  end
end

@testset "weights written in place between calls" begin
  # the bounds read the weights at every call: a hidden zero is recovered after the weights change,
  # and an exact zero stays zero after they change back
  tiny = pow2(Float64, -1070)
  for (name, mk, xhid, whid, α) in (
      ("CheckLoss", w -> CheckLoss(0.5, w), [pow2(Float64, -20)], tiny, pow2(Float64, 100)),
      ("LogPenalty", w -> LogPenalty(1.0, 1.0, w), [pow2(Float64, -10)], tiny, pow2(Float64, 100)),
      ("NormLq", w -> NormLq(pow2(Float64, -600), 0.5, w), [4.0], pow2(Float64, -600), pow2(Float64, 200)),
      ("Berhu", w -> Berhu(pow2(Float64, -600), 1.0, w), [0.5], pow2(Float64, -600), pow2(Float64, 1000)),
      ("Berhu bound", w -> Berhu(pow2(Float64, -1000), 1.0, w), [pow2(Float64, -80)], 1.0, pow2(Float64, 100)))
    @testset "$name" begin
      w = [1.0]
      f = mk(w)
      s = Scaled(f, α)
      @test s([0.0]) === 0.0
      w .= whid
      @test iszero(f(xhid))
      @test agrees(s(xhid), ref(f, α, xhid), 1)
      w .= 1.0
      @test s([0.0]) === 0.0
      @test Scaled(f, 2.0)([0.0]) === 0.0
      w .= whid
      @test agrees(s(xhid), ref(f, α, xhid), 1)
    end
  end
end

@testset "conformance row" begin
  @test Scaled(Translate(SqrNormL2(1.0), [0.0]), pow2(Float64, -1000))([pow2(Float64, 600)]) === pow2(Float64, 199)
end

end
