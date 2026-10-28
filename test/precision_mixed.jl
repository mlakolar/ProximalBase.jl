module MixedPrecisionTest

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase

vector(T) = T[1, 2, -0.5]
matrix(T) = T[2 0.25 0; 0.25 1 0.1; 0 0.1 3]
stack(T) = cat(matrix(T), 2matrix(T); dims = 3)
other(T) = T === Float32 ? Float64 : Float32
buffer(x::AbstractArray) = similar(x)
buffer(x::Tuple) = map(buffer, x)

case(name, f, input = vector; output = input, metric = :none) =
  (; name, f, input, output, metric)

function rule_cases(P)
  b = P[0.25, -0.5, 0.1]
  A = P[1 0.5 0; -0.5 1 0.2]
  X, response = P[1 0.5 0; -0.5 1 0.2; 1 1 1; 0 1 -1], P[0, 1, 0, 1]
  groups = [1:2, 3:3]
  cases = Any[
    case("ZeroFunction", ZeroFunction(); metric = :entry),
    case("NormL1", NormL1(P(1)); metric = :entry),
    case("NormL1 weighted", NormL1(P(1), P[1, 2, 3]); metric = :entry),
    case("NormL1Nonneg", NormL1Nonneg(P(1)); metric = :entry),
    case("ElasticNet", ElasticNet(P(1), P(0.5)); metric = :entry),
    case("ElasticNet weighted", ElasticNet(P(1), P(0.5), P[1, 2, 3]); metric = :entry),
    case("IndBox", IndBox(-P(1), P(1)); metric = :entry),
    case("IndBox array", IndBox(-ones(P, 3), ones(P, 3)); metric = :entry),
    case("NormL2", NormL2(P(1))),
    case("IndBallL2", IndBallL2(P(1))),
    case("NormLinf", NormLinf(P(1))),
    case("IndBallL1", IndBallL1(P(1)); metric = :entry),
    case("IndSimplex", IndSimplex(P(1)); metric = :entry),
    case("IndNonnegative", IndNonnegative(); metric = :entry),
    case("SqrNormL2", SqrNormL2(P(1))),
    case("SqrNormL2 centered", SqrNormL2(P(1), b)),
    case("LinearFunction", LinearFunction(b)),
    case("QuadraticFunction", QuadraticFunction(Matrix(P(1)I, 3, 3), b)),
    case("QuadraticFunction uncentered", QuadraticFunction(Matrix(P(1)I, 3, 3))),
    case("GroupNormL2", GroupNormL2(P(1), groups); metric = :group),
    case("GroupNormLinf", GroupNormLinf(P(1), groups); metric = :group),
    case("SparseGroupNormL2", SparseGroupNormL2(P(1), P(0.5), groups); metric = :group),
    case("GroupMCP", GroupMCP(P(1), groups); metric = :group),
    case("GroupSCAD", GroupSCAD(P(1), groups); metric = :group),
    case("NormL0", NormL0(P(1)); metric = :entry),
    case("MCP", MCP(P(1), P(3)); metric = :entry),
    case("SCAD", SCAD(P(1), P(3.7)); metric = :entry),
    case("CheckLoss", CheckLoss(P(0.3)); metric = :entry),
    case("HingeLoss", HingeLoss(); metric = :entry),
    case("EpsInsensitiveLoss", EpsInsensitiveLoss(P(0.1)); metric = :entry),
    case("TotalVariation1D", TotalVariation1D(P(1))),
    case("FusedLasso", FusedLasso(P(1), P(0.5))),
    case("NormL1Fused pair", NormL1Fused(P(1), P(0.5)), T -> T[1, -2]),
    case("NormL1Fused vector", NormL1Fused(P(1), P(0.5))),
    case("NormL1Fused arrays", NormL1Fused(P(1), P(0.5)), T -> (vector(T), -vector(T))),
    case("SLOPE", SLOPE(P[3, 2, 1])),
    case("NegEntropy", NegEntropy(P(1)), T -> T[0.2, 0.5, 1.5]; metric = :entry),
    case("NormLq", NormLq(P(1), P(0.5)); metric = :entry),
    case("CappedL1", CappedL1(P(1), P(1)); metric = :entry),
    case("LogPenalty", LogPenalty(P(1), P(1)); metric = :entry),
    case("Berhu", Berhu(P(1), P(1)); metric = :entry),
    case("IndHyperplane", IndHyperplane(b, P(0.5)); metric = :entry),
    case("IndHalfspace", IndHalfspace(b, P(0.5)); metric = :entry),
    case("IndAffine", IndAffine(A, P[1, 0])),
    case("IndBallL0", IndBallL0(2)),
    case("IndIsotonic", IndIsotonic()),
    case("LeastSquaresLoss", LeastSquaresLoss(response, X)),
    case("LogisticLoss", LogisticLoss(response, X)),
    case("GLMLoss", GLMLoss(PB.Huber(P(0.5)), response, X)),
    case("NuclearNorm", NuclearNorm(P(1)), matrix),
    case("IndPSD", IndPSD(), matrix),
    case("TracePSD", TracePSD(P(1)), matrix),
    case("IndBallRank", IndBallRank(1), matrix),
    case("LogDetLoss", LogDetLoss(Matrix(P(1)I, 3, 3), P(1)), matrix),
    case("Spectral(NormL1)", Spectral(NormL1(P(1))), matrix),
    case("SingularValue(NormL1)", SingularValue(NormL1(P(1))), matrix),
  ]
  # The rectangular frame has orthogonal rows of squared norm nine.
  frame = P[1 2 2; 2 1 -2]
  for (name, inner) in (("NormL1", NormL1(P(1))), ("SqrNormL2", SqrNormL2(P(1))))
    metric = inner isa SeparableFunction ? :entry : :none
    push!(cases, case("Scaled($name)", Scaled(inner, P(2)); metric))
    push!(cases, case("Tilt($name)", Tilt(inner, b); metric))
    push!(cases, case("Translate($name)", Translate(inner, b); metric))
    push!(cases, case("Translate nested($name)", Translate(Translate(inner, b), b); metric))
    push!(cases, case("Regularize($name)", Regularize(inner, P(0.25))))
    push!(cases, case("Regularize centered($name)", Regularize(inner, P(0.25), b)))
    push!(cases, case("Conjugate($name)", Conjugate(inner); metric))
    push!(cases, case("Precompose scalar($name)", Precompose(inner, P(2))))
    push!(cases, case("Precompose matrix($name)", Precompose(inner, frame, P(9))))
    push!(cases, case("MoreauEnvelope($name)", MoreauEnvelope(inner, P(0.5))))
    push!(cases, case("Constrained($name)", Constrained(inner, IndBox(-P(1), P(1)))))
    push!(cases, case("Spectral(Scaled($name))", Spectral(Scaled(inner, P(2))), matrix))
    push!(cases, case("Precompose(Translate($name))", Precompose(Translate(inner, b), P(-2))))
  end
  for inner in (IndBox(-P(1), P(1)), IndBallL2(P(1)), IndAffine(P[1 0], P[0]))
    name = nameof(typeof(inner))
    push!(cases, case("Precompose indicator scalar($name)", Precompose(inner, P(2)), T -> T[1, 2]))
    push!(cases, case("Precompose indicator matrix($name)", Precompose(inner, frame, P(9))))
  end
  push!(cases, case("Sum", Sum(SqrNormL2(P(1)), LinearFunction(b))))
  push!(cases, case("IndIntersection", IndIntersection(IndBox(-P(1), P(1)), IndBallL2(P(1)))))
  push!(cases, case("SliceSum", SliceSum(NormL1(P(1)), SqrNormL2(P(1))), stack))
  push!(cases, case("SliceSum smooth", SliceSum(SqrNormL2(P(1)), SqrNormL2(P(1))), stack))
  push!(cases, case("SliceSum vector", SliceSum([NormL1(P(1)), NormL1(P(2))]), stack))
  for (name, inner) in (("NormL2", NormL2(P(1))), ("NormL1Fused", NormL1Fused(P(1), P(0.5))))
    for symmetric in (false, true), diagonal in (inner, nothing, NormL1(P(0.25)))
      push!(cases, case("FiberSum($name,$symmetric,$(typeof(diagonal)))", FiberSum(inner; symmetric, diagonal), stack))
    end
  end
  blocks(T) = (T[1, -2], T[0.5, 3])
  heterogeneous(T) = (T[1, -2], other(T)[0.5, 3])
  nested(T) = (blocks(T), T[1, -2])
  for (name, mk) in (("blocks", blocks), ("heterogeneous", heterogeneous))
    push!(cases, case("SeparableSum $name", SeparableSum(NormL1(P(1)), SqrNormL2(P(1))), mk))
    push!(cases, case("SeparableSum smooth $name", SeparableSum(SqrNormL2(P(1)), SqrNormL2(P(1), P[0.1, -0.2])), mk))
    push!(cases, case("Conjugate tuple $name", Conjugate(SeparableSum(NormL1(P(1)), SqrNormL2(P(1)))), mk))
  end
  push!(cases, case("SeparableSum nested", SeparableSum(SeparableSum(NormL1(P(1)), SqrNormL2(P(1))), NormL2(P(1))), nested))
  push!(cases, case("SeparableSum nested smooth", SeparableSum(SeparableSum(SqrNormL2(P(1)), MoreauEnvelope(NormL1(P(1)), P(0.5))), SqrNormL2(P(1))), nested))
  push!(cases, case("Conjugate nested tuple", Conjugate(SeparableSum(SeparableSum(NormL1(P(1)), SqrNormL2(P(1))), NormL2(P(1)))), nested))
  # These conjugates have a proximal value but no standalone value.
  push!(cases, case("Conjugate unsupported", Conjugate(ElasticNet(P(1), P(0.5)))))
  push!(cases, case("Conjugate tuple unsupported", Conjugate(SeparableSum(ElasticNet(P(1), P(0.5)), NormL1(P(1)))), blocks))
  for f in (NormL1(P(1)), ElasticNet(P(1), P(0.5)), SqrNormL2(P(1)), MCP(P(1)), SCAD(P(1)), IndBox(-P(1), P(1)))
    name = nameof(typeof(f))
    metric = f isa SeparableFunction ? :entry : :none
    push!(cases, case("SparseIterate $name", f, T -> SparseIterate(T[1, 0, -0.5]); metric))
    push!(cases, case("SymmetricSparseIterate $name", f, T -> SymmetricSparseIterate(T[1 0.5; 0.5 0]); metric))
  end
  for f in (GroupNormL2(P(1), groups), GroupNormLinf(P(1), groups), SparseGroupNormL2(P(1), P(0.5), groups), GroupMCP(P(1), groups), GroupSCAD(P(1), groups))
    same(T) = AtomIterate(vector(T), groups)
    different(T) = AtomIterate(vector(T), [1:1, 2:3])
    for (layout, mkx, mky) in (("same", same, same), ("input", different, same), ("output", same, different), ("both", different, different))
      push!(cases, case("AtomIterate $(nameof(typeof(f))) $layout", f, mkx; output = mky, metric = :group))
    end
  end
  cases
end

const unsupported_values = ("Conjugate unsupported", "Conjugate tuple unsupported")

function check_rule(c, Tx, Ty, S)
  @nospecialize c
  f, x, y = c.f, c.input(Tx), buffer(c.output(Ty))
  if isproximable(f)
    if c.name in unsupported_values
      @test value_and_prox!(y, f, x, S(0.5)) isa Real
      @test_throws ArgumentError f(y)
    else
      @test typeof(value_and_prox!(y, f, x, S(0.5))) === typeof(f(y))
    end
    if c.metric != :none
      steps = c.metric == :group ? fill(S(0.5), length(f.groups)) : fill(S(0.5), size(x))
      @test typeof(value_and_prox!(y, f, x, steps)) === typeof(f(y))
    end
  end
  if issmooth(f)
    @test typeof(value_and_gradient!(y, f, x)) === typeof(f(x))
  end
end

function changed_cases(T)
  x = T[0.3, 1.7, -2.2]
  frame = T[1 2 2; 2 1 -2]
  groups = [1:2, 3:3]
  [
    ("Precompose scalar", Precompose(NormL1(one(T)), T(2)), x),
    ("Precompose smooth", Precompose(SqrNormL2(one(T)), T(-0.5)), x),
    ("Precompose matrix", Precompose(NormL1(one(T)), frame, T(9)), x),
    ("Precompose indicator", Precompose(IndBox(-one(T), one(T)), T(2)), x),
    ("Precompose matrix indicator", Precompose(IndBallL2(one(T)), frame, T(9)), x),
    ("MoreauEnvelope norm", MoreauEnvelope(NormL1(one(T)), T(0.5)), x),
    ("MoreauEnvelope indicator", MoreauEnvelope(IndBox(-one(T), one(T)), T(0.5)), x),
    ("QuadraticFunction", QuadraticFunction(T[2 0.5 0; 0.5 1 0; 0 0 3], T[0.1, -0.2, 0.3], T(0.5)), x),
    ("GroupNormL2", GroupNormL2(one(T), groups, T[1, 2]), x),
    ("GroupNormLinf", GroupNormLinf(one(T), groups), x),
    ("SparseGroupNormL2", SparseGroupNormL2(one(T), T(0.5), groups), x),
    ("GroupMCP", GroupMCP(one(T), groups), x),
    ("GroupSCAD", GroupSCAD(one(T), groups), x),
    ("IndAffine", IndAffine(T[1 0.5 0; -0.5 1 0.2], T[1, 0]), x),
    ("NormL1Fused pair", NormL1Fused(one(T), T(0.5)), x[1:2]),
    ("NormL1Fused vector", NormL1Fused(one(T), T(0.5)), x),
    ("NormL1Fused zero", NormL1Fused(zero(T), zero(T)), x[1:2]),
  ]
end
function oracle_bits(f, x, T)
  values = [f(x)]
  if isproximable(f)
    y = similar(x)
    push!(values, value_and_prox!(y, f, x, T(0.4)))
    append!(values, y)
    append!(values, prox!(y, f, x, T(0.4)))
  end
  if issmooth(f)
    g = similar(x)
    push!(values, value_and_gradient!(g, f, x))
    append!(values, g)
    append!(values, gradient!(g, f, x))
  end
  reinterpret(T === Float32 ? UInt32 : UInt64, values)
end

# Fixed inputs recorded before the precision fixes, including signed zeros.
const same_precision_bits = Dict(
  Float32 => Dict(
    "Precompose scalar" => UInt32[0x41066666, 0x40933334, 0x00000000, 0x3f666667, 0xbfb33334, 0x00000000, 0x3f666667, 0xbfb33334],
    "Precompose smooth" => UInt32[0x3f7a3d71, 0x3f4ecf56, 0x3e8ba2e9, 0x3fc5d174, 0xc0000000, 0x3e8ba2e9, 0x3fc5d174, 0xc0000000, 0x3f7a3d71, 0x3d99999a, 0x3ed9999a, 0xbf0ccccd, 0x3d99999a, 0x3ed9999a, 0xbf0ccccd],
    "Precompose matrix" => UInt32[0x40ecccce, 0x40466667, 0xbed82d83, 0x3fba4fa5, 0xbf9f49f5, 0xbed82d83, 0x3fba4fa5, 0xbf9f49f5],
    "Precompose indicator" => UInt32[0x7f800000, 0x00000000, 0x3e99999a, 0x3f000000, 0xbf000000, 0x3e99999a, 0x3f000000, 0xbf000000],
    "Precompose matrix indicator" => UInt32[0x7f800000, 0x00000000, 0xbf66d1ba, 0x3f996967, 0xbf4cb6ee, 0xbf66d1ba, 0x3f996967, 0xbf4cb6ee],
    "MoreauEnvelope norm" => UInt32[0x405f5c2a, 0x40282d84, 0x3e2aaaaa, 0x3fa66667, 0xbfe66667, 0x3e2aaaaa, 0x3fa66667, 0xbfe66667, 0x405f5c2a, 0x3f19999a, 0x3f800000, 0xbf800000, 0x3f19999a, 0x3f800000, 0xbf800000],
    "MoreauEnvelope indicator" => UInt32[0x3ff70a41, 0x3f187e6b, 0x3e99999a, 0x3fb1c71c, 0xbfd55555, 0x3e99999a, 0x3fb1c71c, 0xbfd55555, 0x3ff70a41, 0x00000000, 0x3fb33334, 0xc019999a, 0x00000000, 0x3fb33334, 0xc019999a],
    "QuadraticFunction" => UInt32[0x410947ae, 0x401a1632, 0x3b536719, 0x3fa2af13, 0xbf86fb59, 0x3b536719, 0x3fa2af13, 0xbf86fb59, 0x410947ae, 0x3fc66667, 0x3fd33333, 0xc0c9999a, 0x3fc66667, 0x3fd33333, 0xc0c9999a],
    "GroupNormL2" => UInt32[0x40c40a63, 0x40840a63, 0x3e6c047d, 0x3fa72dd9, 0xbfb33333, 0x3e6c047d, 0x3fa72dd9, 0xbfb33333],
    "GroupNormLinf" => UInt32[0x4079999a, 0x40466667, 0x3e99999a, 0x3fa66667, 0xbfe66667, 0x3e99999a, 0x3fa66667, 0xbfe66667],
    "SparseGroupNormL2" => UInt32[0x40d0a936, 0x40845d5c, 0x00000000, 0x3f823236, 0xbfcccccd, 0x00000000, 0x3f823236, 0xbfcccccd],
    "GroupMCP" => UInt32[0x4055a16e, 0x403cfcca, 0x3e6e4e9c, 0x3fa8cd04, 0xc004ec4f, 0x3e6e4e9c, 0x3fa8cd04, 0xc004ec4f],
    "GroupSCAD" => UInt32[0x408b695c, 0x405ab202, 0x3e4e8865, 0x3f924b48, 0xbff8356e, 0x3e4e8865, 0x3f924b48, 0xbff8356e],
    "IndAffine" => UInt32[0x7f800000, 0x00000000, 0x3f1c3832, 0x3f478f99, 0xc017d05f, 0x3f1c3832, 0x3f478f99, 0xc017d05f],
    "NormL1Fused pair" => UInt32[0x402ccccd, 0x3fd9999a, 0x3dcccccc, 0x3f8ccccd, 0x3dcccccc, 0x3f8ccccd],
    "NormL1Fused vector" => UInt32[0x4101999a, 0x40933334, 0x00000000, 0x3f666668, 0xbfb33334, 0x00000000, 0x3f666668, 0xbfb33334],
    "NormL1Fused zero" => UInt32[0x00000000, 0x00000000, 0x3e99999a, 0x3fd9999a, 0x3e99999a, 0x3fd9999a],
  ),
  Float64 => Dict(
    "Precompose scalar" => UInt64[0x4020cccccccccccd, 0x4012666666666666, 0x0000000000000000, 0x3feccccccccccccc, 0xbff6666666666667, 0x0000000000000000, 0x3feccccccccccccc, 0xbff6666666666667],
    "Precompose smooth" => UInt64[0x3fef47ae147ae148, 0x3fe9d9ead7cd3920, 0x3fd1745d1745d174, 0x3ff8ba2e8ba2e8ba, 0xc000000000000000, 0x3fd1745d1745d174, 0x3ff8ba2e8ba2e8ba, 0xc000000000000000, 0x3fef47ae147ae148, 0x3fb3333333333333, 0x3fdb333333333333, 0xbfe199999999999a, 0x3fb3333333333333, 0x3fdb333333333333, 0xbfe199999999999a],
    "Precompose matrix" => UInt64[0x401d99999999999a, 0x4008cccccccccccd, 0xbfdb05b05b05b05a, 0x3ff749f49f49f4a0, 0xbff3e93e93e93e94, 0xbfdb05b05b05b05a, 0x3ff749f49f49f4a0, 0xbff3e93e93e93e94],
    "Precompose indicator" => UInt64[0x7ff0000000000000, 0x0000000000000000, 0x3fd3333333333333, 0x3fe0000000000000, 0xbfe0000000000000, 0x3fd3333333333333, 0x3fe0000000000000, 0xbfe0000000000000],
    "Precompose matrix indicator" => UInt64[0x7ff0000000000000, 0x0000000000000000, 0xbfecda373e8a79ad, 0x3ff32d2ce9e116fd, 0xbfe996dddb66b0b2, 0xbfecda373e8a79ad, 0x3ff32d2ce9e116fd, 0xbfe996dddb66b0b2],
    "MoreauEnvelope norm" => UInt64[0x400beb851eb851ec, 0x400505b05b05b05a, 0x3fc5555555555555, 0x3ff4cccccccccccc, 0xbffcccccccccccce, 0x3fc5555555555555, 0x3ff4cccccccccccc, 0xbffcccccccccccce, 0x400beb851eb851ec, 0x3fe3333333333333, 0x3ff0000000000000, 0xbff0000000000000, 0x3fe3333333333333, 0x3ff0000000000000, 0xbff0000000000000],
    "MoreauEnvelope indicator" => UInt64[0x3ffee147ae147adf, 0x3fe30fcd6e9e0650, 0x3fd3333333333333, 0x3ff638e38e38e38e, 0xbffaaaaaaaaaaaab, 0x3fd3333333333333, 0x3ff638e38e38e38e, 0xbffaaaaaaaaaaaab, 0x3ffee147ae147adf, 0x0000000000000000, 0x3ff6666666666666, 0xc003333333333334, 0x0000000000000000, 0x3ff6666666666666, 0xc003333333333334],
    "QuadraticFunction" => UInt64[0x402128f5c28f5c29, 0x400342c61bae4234, 0x3f6a6d01a6d01b30, 0x3ff455e2455e2456, 0xbff0df6b0df6b0e0, 0x3f6a6d01a6d01b30, 0x3ff455e2455e2456, 0xbff0df6b0df6b0e0, 0x402128f5c28f5c29, 0x3ff8cccccccccccd, 0x3ffa666666666666, 0xc019333333333334, 0x3ff8cccccccccccd, 0x3ffa666666666666, 0xc019333333333334],
    "GroupNormL2" => UInt64[0x4018814c4e8ff998, 0x4010814c4e8ff998, 0x3fcd808f954469d8, 0x3ff4e5bb09bb204f, 0xbff6666666666667, 0x3fcd808f954469d8, 0x3ff4e5bb09bb204f, 0xbff6666666666667],
    "GroupNormLinf" => UInt64[0x400f333333333334, 0x4008cccccccccccd, 0x3fd3333333333333, 0x3ff4cccccccccccc, 0xbffcccccccccccce, 0x3fd3333333333333, 0x3ff4cccccccccccc, 0xbffcccccccccccce],
    "SparseGroupNormL2" => UInt64[0x401a1526bb45c0a8, 0x40108bab6b87e9ee, 0x0000000000000000, 0x3ff04646b84cf409, 0xbff999999999999a, 0x0000000000000000, 0x3ff04646b84cf409, 0xbff999999999999a],
    "GroupMCP" => UInt64[0x400ab42db3102eff, 0x40079f993eae8a4e, 0x3fcdc9d36679aa40, 0x3ff519a068962dee, 0xc0009d89d89d89d9, 0x3fcdc9d36679aa40, 0x3ff519a068962dee, 0xc0009d89d89d89d9],
    "GroupSCAD" => UInt64[0x40116d2b9347b4ff, 0x400b564023f7bf8c, 0x3fc9d10c8c030b05, 0x3ff24968e32cd279, 0xbfff06ada2811cf1, 0x3fc9d10c8c030b05, 0x3ff24968e32cd279, 0xbfff06ada2811cf1],
    "IndAffine" => UInt64[0x7ff0000000000000, 0x0000000000000000, 0x3fe387066e053d34, 0x3fe8f1f323f58599, 0xc002fa0be82fa0bf, 0x3fe387066e053d34, 0x3fe8f1f323f58599, 0xc002fa0be82fa0bf],
    "NormL1Fused pair" => UInt64[0x400599999999999a, 0x3ffb333333333334, 0x3fb9999999999998, 0x3ff199999999999a, 0x3fb9999999999998, 0x3ff199999999999a],
    "NormL1Fused vector" => UInt64[0x4020333333333334, 0x4012666666666667, 0x0000000000000000, 0x3feccccccccccccb, 0xbff6666666666668, 0x0000000000000000, 0x3feccccccccccccb, 0xbff6666666666668],
    "NormL1Fused zero" => UInt64[0x0000000000000000, 0x0000000000000000, 0x3fd3333333333333, 0x3ffb333333333333, 0x3fd3333333333333, 0x3ffb333333333333],
  ),
)

const selection = Regex(get(ENV, "PROXIMALBASE_PRECISION_FILTER", ""))

if occursin(selection, "bitwise")
  @testset "Same precision bits" begin
    for T in (Float32, Float64), (name, f, x) in changed_cases(T)
      @testset "$T/$name" begin
        @test oracle_bits(f, x, T) == same_precision_bits[T][name]
        if f isa PB.AbstractGroupPenalty
          for groups in (f.groups, [1:1, 2:3])
            @test oracle_bits(f, AtomIterate(x, groups), T) == same_precision_bits[T][name]
          end
          y, steps = similar(x), fill(T(0.4), length(f.groups))
          v = value_and_prox!(y, f, x, steps)
          U = T === Float32 ? UInt32 : UInt64
          @test reinterpret(U, [v; y]) == same_precision_bits[T][name][2:5]
        end
      end
    end
  end
end

if occursin(selection, "regressions")
  @testset "Mixed precision branches" begin
    for n in (0, 2, 3)
      f = NormL1Fused(1e-50, 0.0)
      x, y = fill(1f0, n), zeros(Float32, n)
      v = value_and_prox!(y, f, x, 0.5f0)
      @test typeof(v) === typeof(f(y))
      @test iszero(v)
      @test y == x
    end
    affine = IndAffine([1.0 0.0], [0.0])
    @test affine(Float32[1e-5, 0]) === Inf32
    @test affine(Float32[0, 0]) === 0f0
    for P in (Float32, Float64), T in (Int32, Int64)
      integer = IndAffine(P[1 0], P[0])
      @test integer(T[1, 0]) === Inf
      @test integer(T[0, 0]) === 0.0
    end
    for T in (Float32, Float64)
      scaled = IndAffine([1.0 1.0], [0.0])
      @test scaled(T[floatmax(T), -floatmax(T)]) === zero(T)
      @test scaled(T[Inf, 0]) === T(Inf)
    end
    q, x, g = QuadraticFunction(ones(1, 1)), [1e39], zeros(Float32, 1)
    v = value_and_gradient!(g, q, x)
    @test isfinite(v)
    @test v === q(x)
    @test g == Float32[Inf]
    for f in (Precompose(NormL1(1.0), 1.0), Precompose(NormL1(1.0), ones(1, 1), 1.0))
      y = zeros(Float32, 1)
      v = value_and_prox!(y, f, x, 1e39)
      @test !any(isnan, y)
      @test !isnan(v)
      @test v === f(y)
      @test eltype(f.image.buf) === Float64
    end
    for Tx in (Float32, Float64), Ty in (Float32, Float64)
      Tx === Ty && continue
      x, g = Tx[0.3, 1.7, -2.2], zeros(Ty, 3)
      e = MoreauEnvelope(NormL1(Tx(1)), Tx(0.5))
      v = e(x)
      @test value_and_gradient!(g, e, x) === v
      @test eltype(e.scratch.buf) === Tx
      expected = Ty.(gradient(e, x))
      @test g == expected
      @test gradient!(g, e, x) == expected
    end
    translated = Translate(NormL1(1.0), [1e8])
    x, y = [0.5], zeros(Float32, 1)
    @test prox!(y, translated, x, 1.0) == Float32[0]
    @test prox_entry(translated, x[1], 1, 1.0) === 1.5
  end
end

if occursin(selection, "allocations")
  @testset "Warmed precision allocations" begin
    for Tx in (Float32, Float64), Ty in (Float32, Float64), (name, f, x) in changed_cases(Tx)
      @testset "$name, x=$Tx, output=$Ty" begin
        y = similar(x, Ty)
        if isproximable(f)
          @test PB.Testing.allocations(prox!, y, f, x, Tx(0.4)) == 0
          @test PB.Testing.allocations(value_and_prox!, y, f, x, Tx(0.4)) == 0
        end
        if issmooth(f)
          @test PB.Testing.allocations(gradient!, y, f, x) == 0
          @test PB.Testing.allocations(value_and_gradient!, y, f, x) == 0
        end
      end
    end
    f = NormL1Fused(1e-50, 0.0)
    @test PB.Testing.allocations(value_and_prox!, zeros(Float32, 2), f, Float32[1, 2], 0.5f0) == 0
  end
end

if occursin(selection, "group allocations")
  @testset "Warmed group precision allocations" begin
    groups = [1:2, 3:3]
    for Tx in (Float32, Float64), Ty in (Float32, Float64)
      for f in (GroupNormL2(Tx(1), groups), GroupNormLinf(Tx(1), groups),
                SparseGroupNormL2(Tx(1), Tx(0.5), groups), GroupMCP(Tx(1), groups), GroupSCAD(Tx(1), groups))
        for xgroups in (groups, [1:1, 2:3]), ygroups in (groups, [1:1, 2:3])
          x, y = AtomIterate(vector(Tx), xgroups), AtomIterate(zeros(Ty, 3), ygroups)
          for step in (Tx(0.5), fill(Tx(0.5), length(groups)))
            @test PB.Testing.allocations(value_and_prox!, y, f, x, step) == 0
          end
        end
      end
    end
  end
end

if occursin(selection, "value allocations")
  @testset "Mixed precision value workspace" begin
    # The inner SVD workspace changes precision between the two oracles.
    h = Precompose(NuclearNorm(0.5f0), 2f0)
    x, y = Float32[2 0.25; -0.5 1], zeros(Float64, 2, 2)
    v = value_and_prox!(y, h, x, 0.4f0)
    @test eltype(h.image.buf) === Float32
    @test eltype(h.f.ws.A) === Float64
    @test typeof(v) === typeof(h(y))
  end
end

@testset "Mixed precision value types" begin
  for P in (Float32, Float64), c in rule_cases(P), Tx in (Float32, Float64), Ty in (Float32, Float64), S in (Float32, Float64)
    occursin(selection, "$P/$(c.name)") || continue
    @testset "$(c.name), parameters=$P, x=$Tx, output=$Ty, step=$S" begin
      check_rule(c, Tx, Ty, S)
    end
  end
end

end
