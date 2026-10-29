# Shared reference-backend and physical-device cases. The entry point supplies
# DeviceArray, test_types, host (synchronized copy), and ≃ (comparison).
using Test, ProximalBase, LinearAlgebra, Random
using GPUArraysCore: allowscalar
import Adapt

Random.seed!(1)

# a separable function of one's own, a bits type: Σᵢ c xᵢ² / 2
struct ScaledSquare{T} <: SeparableFunction
  c::T
end
ProximalBase.value_entry(f::ScaledSquare, v, i) = f.c * v^2 / 2
ProximalBase.prox_entry(f::ScaledSquare, v, i, γ) = v / (1 + γ * f.c)
ProximalBase.iszeropreserving(::ScaledSquare) = true

# the values agree, including Inf for a point outside a set
same(a, b) = isinf(b) ? a == b : a ≃ b

@testset "separable functions on GPU arrays" begin
  allowscalar(false)
  n = 300
  x, w, l, u = 2 .* randn(n), 0.5 .+ rand(n), -rand(n), rand(n)
  for T in test_types
    xc = T.(x)
    xg = DeviceArray(xc)
    for f in (NormL1(0.3), NormL1(0.3, T.(w)), NormL1Nonneg(0.3), NormL1Nonneg(0.3, T.(w)),
              ElasticNet(0.3, 0.5), ElasticNet(0.3, 0.5, T.(w)), IndBox(-0.5, 0.7), IndBox(T.(l), T.(u)),
              IndBallLinf(0.4), IndNonnegative(), NormL0(0.2), MCP(0.3, 3.0), SCAD(0.3, 3.7), ScaledSquare(T(0.6)),
              SqrNormL2(0.7), SqrNormL2(0.7, T.(w)))
      @testset "$(nameof(typeof(f))), $T" begin
        fg = Adapt.adapt(DeviceArray, f)                       # its parameter arrays on the device
        yc = prox(f, xc, 0.8)
        yg = similar(xg)
        @test prox!(yg, fg, xg, 0.8) === yg
        @test eltype(yg) == T
        @test host(yg) ≃ yc
        @test same(fg(xg), f(xc))
        @test same(fg(yg), f(yc))
        vc = value_and_prox!(similar(xc), f, xc, 0.8)
        @test same(value_and_prox!(yg, fg, xg, 0.8), vc)
        @test host(yg) ≃ yc
        zg = copy(xg)
        @test host(prox!(zg, fg, zg, 0.8)) ≃ yc        # in place
      end
    end
  end
  # matrices, and a point on each side of a box
  X = randn(6, 5)
  fw = NormL1(0.2, rand(6, 5))
  @test Adapt.adapt(DeviceArray, fw)(DeviceArray(X)) ≃ fw(X)
  @test host(prox(Adapt.adapt(DeviceArray, IndBox(-0.1, 0.1)), DeviceArray(X))) ≃ clamp.(X, -0.1, 0.1)
  @test IndBox(0.0, 1.0)(DeviceArray([0.5, 1.0])) == 0
  @test IndBox(0.0, 1.0)(DeviceArray([0.5, 1.5])) == Inf
  # adapt converts the parameter arrays and keeps the constructors' checks
  @test Adapt.adapt(DeviceArray, NormL1(0.3, w)).w isa DeviceArray
  @test Adapt.adapt(DeviceArray, IndBox(l, u)).u isa DeviceArray
  @test Adapt.adapt(DeviceArray, NormL1(0.3)) == NormL1(0.3)
  @test_throws ArgumentError Adapt.adapt(DeviceArray, NormL1(0.3, -w))
end

@testset "compensated separable L1 coefficients" begin
  allowscalar(false)
  for T in test_types, F in (NormL1, NormL1Nonneg)
    large = floatmax(T)/2
    f = F(large, T[inv(large), 0])
    x = T[7, 3]
    fg = Adapt.adapt(DeviceArray, f)
    y = similar(DeviceArray(x))
    value = value_and_prox!(y, fg, DeviceArray(x), T(4))
    @test host(y) ≃ T[3, 3]
    @test value ≃ one(T)*3
    @test ProximalBase._prox_entry_slope(f, x[1], 1, T(4))[1] == prox_entry(f, x[1], 1, T(4))
    small = T === Float32 ? T(1e-30) : T(1e-200)
    f = F(small, T[small])
    x = T[inv(small)]
    expected = T(BigFloat(small)^2*BigFloat(x[1]))
    @test f(x) ≈ expected
    @test Adapt.adapt(DeviceArray, f)(DeviceArray(x)) ≈ expected
    tiny = nextfloat(zero(T))
    final_subnormal = F(T(1.5), T[2])
    @test Adapt.adapt(DeviceArray, final_subnormal)(DeviceArray(T[tiny])) == 3tiny
    factor = T === Float32 ? T(4096.5) : T(100000000.5)
    threshold = T(BigFloat(tiny)*BigFloat(factor)^2)
    f = F(factor, T[factor])
    x = T[4threshold]
    expected = x .- threshold
    @test prox(f, x, tiny) ≈ expected rtol=8eps(T) atol=zero(T)
    @test host(prox(Adapt.adapt(DeviceArray, f), DeviceArray(x), tiny)) ≈ expected rtol=8eps(T) atol=zero(T)
  end
end

@testset "norms, balls and the simplex on GPU arrays" begin
  allowscalar(false)
  n = 300
  x = 2 .* randn(n)
  for T in test_types
    xc = T.(x)
    xg = DeviceArray(xc)
    # each function also with a level or radius at which the result is zero or x
    for f in (NormL2(0.3), NormL2(100.), NormLinf(0.3), NormLinf(1e4), IndBallL2(1.), IndBallL2(1e3), IndBallL1(2.),
              IndBallL1(1e4), IndBallL1(0.), IndSimplex(1.), IndSimplex(3., equality = false), IndSimplex(1e4, equality = false),
              IndSimplex(1e4), IndSimplex(0.))
      @testset "$(nameof(typeof(f))), $T" begin
        yc = prox(f, xc, 0.8)
        yg = similar(xg)
        @test prox!(yg, f, xg, 0.8) === yg
        @test eltype(yg) == T
        @test host(yg) ≃ yc
        @test same(f(xg), f(xc))
        @test same(f(yg), f(yc))
        vc = value_and_prox!(similar(xc), f, xc, 0.8)
        @test same(value_and_prox!(yg, f, xg, 0.8), vc)
        @test host(yg) ≃ yc
        zg = copy(xg)
        @test host(prox!(zg, f, zg, 0.8)) ≃ yc       # in place
      end
    end
  end
  # a matrix, and entries of one sign, where every entry stays active
  X = randn(6, 5)
  @test host(prox(IndBallL1(1.), DeviceArray(X))) ≃ prox(IndBallL1(1.), X)
  @test host(prox(IndSimplex(1.), DeviceArray(fill(0.5, 4)))) ≃ fill(0.25, 4)
  @test host(prox(NormLinf(0.5), DeviceArray([3., -3., 3.]))) ≃ fill(3 - 0.5 / 3, 3) .* [1, -1, 1]
end

@testset "NormLinf precision on GPU arrays" begin
  allowscalar(false)
  ext = Base.get_extension(ProximalBase, :ProximalBaseGPUArraysCoreExt)
  for P in test_types, Tx in test_types, Ty in test_types, S in test_types, λ in (0, 0.3, 100)
    f = NormLinf(P(λ))
    x = DeviceArray(Tx[0.3, 1.7, -2.2])
    y = DeviceArray(zeros(Ty, 3))
    v = value_and_prox!(y, f, x, S(0.4))
    @test typeof(v) === typeof(f(y))
    @test isapprox(host(y), prox(f, Tx[0.3, 1.7, -2.2], S(0.4)); rtol = 8sqrt(max(eps(Tx), eps(Ty))))
    if P === Tx === Ty === S && λ == 0.3
      # The level and product used by the original same-precision kernel.
      τ = ext._level(x, S(0.4) * f.λ, abs)
      expected = f.λ * τ
      expected_y = sign.(host(x)) .* min.(abs.(host(x)), τ)
      U = Tx === Float32 ? UInt32 : UInt64
      @test reinterpret(U, v) == reinterpret(U, expected)
      @test reinterpret(U, host(y)) == reinterpret(U, expected_y)
      @test reinterpret(U, host(prox!(y, f, x, S(0.4)))) == reinterpret(U, expected_y)
    end
  end
end

@testset "group norms on GPU arrays" begin
  allowscalar(false)
  # groups of sizes 1 to 26 in six size classes, out of order and not all
  # contiguous; x[2] and x[58:60] are in no group
  groups = [[1, 5, 9], 3:4, [6], 10:17, [8, 7], 18:30, [31], 32:57]
  n = 60
  x = 2 .* randn(n)
  x[3:4] .= 0                                         # a group at zero
  x[10:17] .*= 0.05                                   # groups that shrink to zero
  x[[8, 7]] .*= 0.05
  w = 0.5 .+ rand(length(groups))
  for T in test_types
    xc = T.(x)
    xg = DeviceArray(xc)
    # each function also with a level at which nothing or everything shrinks to zero
    for f in (GroupNormL2(0.3, groups), GroupNormL2(0.3, groups, w), GroupNormL2(0., groups), GroupNormL2(50., groups),
              GroupNormLinf(0.3, groups), GroupNormLinf(0.3, groups, w), GroupNormLinf(0., groups), GroupNormLinf(100., groups),
              SparseGroupNormL2(0.2, 0.3, groups), SparseGroupNormL2(0., 0.3, groups), SparseGroupNormL2(0.5, 1.5, groups, w),
              SparseGroupNormL2(0.2, 0., groups), IndBallGroupL0(3, groups), IndBallGroupL0(0, groups), IndBallGroupL0(20, groups))
      # with its groups on the device, and left on the host
      for (where, fg) in (("device", Adapt.adapt(DeviceArray, f)), ("host", f))
        @testset "$(nameof(typeof(f))) on the $where, $T" begin
          yc = prox(f, xc, 0.8)
          yg = similar(xg)
          @test prox!(yg, fg, xg, 0.8) === yg
          @test eltype(yg) == T
          @test host(yg) ≃ yc
          @test same(fg(xg), f(xc))
          @test same(fg(yg), f(yc))
          vc = value_and_prox!(similar(xc), f, xc, 0.8)
          @test same(value_and_prox!(yg, fg, xg, 0.8), vc)
          @test host(yg) ≃ yc
          zg = copy(xg)
          @test host(prox!(zg, fg, zg, 0.8)) ≃ yc      # in place
        end
      end
    end
  end
  # the rows of a matrix: the multi-task group lasso
  X = randn(6, 5)
  rows = [i:6:30 for i in 1:6]
  for f in (GroupNormL2(0.4, rows), GroupNormLinf(0.4, rows), SparseGroupNormL2(0.2, 0.4, rows))
    fg = Adapt.adapt(DeviceArray, f)
    @test host(prox(fg, DeviceArray(X), 0.7)) ≃ prox(f, X, 0.7)
    @test fg(DeviceArray(X)) ≃ f(X)
  end
  # the earlier groups first among ties, and at most k entries in a group
  f = IndBallGroupL0(2, [1:2, 3:4, 5:6])
  @test host(prox(Adapt.adapt(DeviceArray, f), DeviceArray([1., 1., 1., 1., 2., 0.]))) == [1., 1., 0., 0., 2., 0.]
  fk = IndBallGroupL0(3, groups; k = 2)
  @test Adapt.adapt(DeviceArray, fk)(DeviceArray(x)) == fk(x) == Inf
  @test IndBallGroupL0(6, groups; k = 30)(DeviceArray(x)) == Inf          # seven nonzero groups
  @test IndBallGroupL0(7, groups; k = 30)(DeviceArray(x)) == 0
  @test_throws ArgumentError prox(Adapt.adapt(DeviceArray, fk), DeviceArray(x))
  # no groups, and groups that do not fit in x
  f0 = GroupNormL2(0.3, UnitRange{Int}[])
  @test f0(DeviceArray(x)) == 0
  @test host(prox(f0, DeviceArray(x))) == x
  @test_throws BoundsError prox(GroupNormL2(0.3, groups), DeviceArray(randn(20)))
  @test_throws DimensionMismatch prox!(DeviceArray(randn(59)), GroupNormL2(0.3, groups), DeviceArray(x), 1.0)
  # adapt to the device and back; on host arrays the adapted function is the function
  f = SparseGroupNormL2(0.2, 0.3, groups)
  fg = Adapt.adapt(DeviceArray, f)
  @test Adapt.adapt(Array, fg) === f
  @test Adapt.adapt(Array, f) === f
  @test host(prox(Adapt.adapt(DeviceArray, fg), DeviceArray(x))) ≃ prox(f, x)
  @test isproximable(fg) && isconvex(fg) && !ProximalBase._isindicator(fg)
  fl0 = Adapt.adapt(DeviceArray, IndBallGroupL0(2, groups))
  @test isproximable(fl0) && !isconvex(fl0) && ProximalBase._isindicator(fl0)
  @test fg(x) == f(x)
  @test prox(fg, x, 0.8) == prox(f, x, 0.8)
  @test value_and_prox!(similar(x), fg, x, 0.8) == value_and_prox!(similar(x), f, x, 0.8)
  @test value_and_prox!(DeviceArray(similar(x)), fl0, DeviceArray(x), 1.0) == 0
end

@testset "adapted group penalties on host AtomIterates" begin
  allowscalar(false)
  ranges = [1:2, 3:5, 6:6, 7:9]
  layouts = (ranges, collect.(ranges))
  for T in test_types, gs in layouts
    v = T[3, -4, 0, 0, 0, 0, 2, -1, 0.5]
    for f in (GroupNormL2(T(0.4), gs), GroupNormLinf(T(0.4), gs),
              SparseGroupNormL2(T(0.2), T(0.4), gs), IndBallGroupL0(1, gs; k = 1))
      fg = Adapt.adapt(DeviceArray, f)
      reference = prox(f, v, T(0.7))
      for xgroups in layouts, ygroups in layouts
        xa = AtomIterate(v, xgroups)
        activate!(xa, 3)
        ya = AtomIterate(ones(T, 9), ygroups)
        @test same(fg(xa), f(Vector(xa)))
        @test prox!(ya, fg, xa, T(0.7)) === ya
        @test Vector(ya) ≃ reference
        @test same(value_and_prox!(ya, fg, xa, T(0.7)), f(reference))
        @test Vector(ya) ≃ reference
        @test xa.layout_cache === ya.layout_cache === f.groups
        alias = copy(xa)
        @test prox!(alias, fg, alias, T(0.7)) === alias
        @test Vector(alias) ≃ reference
        @test Vector(xa) == v
      end
    end
  end
end

@testset "representable group norms outside the squared range" begin
  allowscalar(false)
  groups = [1:2, 3:4]
  for T in test_types, scale in (floatmin(T), floatmax(T) / T(16))
    @test norm_diff(DeviceArray(scale .* T[3, 4]), DeviceArray(zeros(T, 2))) / scale ≃ T(5)
    cx = DeviceArray(complex.(scale .* T[3, 4], scale .* T[3, 4]))
    cy = DeviceArray(zeros(Complex{T}, 2))
    @test norm_diff(cx, cy) isa T
    @test norm_diff(cx, cy) / scale ≃ sqrt(T(50))
    # The shared Euclidean routines fall back to GPUArrays' scaled norm.
    xv = scale .* T[3, 4]
    @test NormL2(one(T))(DeviceArray(xv)) / scale ≃ T(5)
    for f in (NormL2(scale * T(2.5)), IndBallL2(scale * T(2.5)))
      @test host(prox(f, DeviceArray(xv))) ./ scale ≃ T[1.5, 2]
      if f isa NormL2 && scale > one(T)
        xg = DeviceArray(xv)
        yg = similar(xg)
        @test value_and_prox!(yg, f, xg, one(T)) === T(Inf)
        @test host(yg) ./ scale ≃ T[1.5, 2]
      end
    end
    x = scale .* T[1, 0, 2, 0]
    for f in (GroupNormL2(scale / T(2), groups, ones(T, 2)),
              SparseGroupNormL2(scale / T(4), scale / T(4), groups, ones(T, 2)))
      fg = Adapt.adapt(DeviceArray, f)
      expected = T[0.5, 0, 1.5, 0]
      @test host(prox(fg, DeviceArray(x))) ./ scale ≃ expected
      @test host(prox(f, DeviceArray(x))) ./ scale ≃ expected
    end
    f = Adapt.adapt(DeviceArray, GroupNormL2(one(T), groups, ones(T, 2)))
    @test f(DeviceArray(x)) / scale ≃ T(3)
    # Norm-based selection must distinguish a zero group and finite nonzero
    # groups even when every nonzero squared norm overflows or underflows.
    h = Adapt.adapt(DeviceArray, IndBallGroupL0(1, [1:1, 2:2, 3:3]))
    @test host(prox(h, DeviceArray(scale .* T[0, 1, 2]))) ./ scale == T[0, 0, 2]
  end
end

@testset "weighted norms at device overflow" begin
  allowscalar(false)
  for T in test_types
    M = floatmax(T)
    x = DeviceArray(T[M, M])
    f = NormL2(T(1) / 4)
    reference = sqrt(big(2)) * big(M) / 4
    tolerance = 2big(eps(T(reference)))
    v = f(x)
    @test isfinite(v) && abs(big(v) - reference) <= tolerance
    y = similar(x)
    v = value_and_prox!(y, f, x, T(1) / 8)
    @test isfinite(v) && abs(big(v) - reference) <= tolerance
    expected = host(prox(f, x, T(1) / 8))
    @test expected ≈ prox(f, T[M, M], T(1) / 8)       # the device shrink agrees with the host one
    @test reinterpret(UInt8, host(y)) == reinterpret(UInt8, expected)
    z = copy(x)
    v = value_and_prox!(z, f, z, T(1) / 8)
    @test isfinite(v) && abs(big(v) - reference) <= tolerance
    @test reinterpret(UInt8, host(z)) == reinterpret(UInt8, expected)
    v = NormL1(T(1) / 4)(x)
    @test isfinite(v) && abs(big(v) - big(M) / 2) <= 2big(eps(M / 2))
    u = nextfloat(zero(T))
    reference = big(u) * sqrt(big(2)) * big(M)
    v = NormL2(u)(x)
    @test isfinite(v) && abs(big(v) - reference) <= 2big(eps(T(reference)))
    # the spectral functions decompose on the host; the overflow fallback too
    v = NuclearNorm(T(1) / 4)(DeviceArray(Matrix{T}(M * I, 2, 2)))
    @test isfinite(v) && abs(big(v) - big(M) / 2) <= 2big(eps(M / 2))
  end
end

@testset "zero penalty levels before device reductions" begin
  allowscalar(false)
  groups = [1:2, 3:4]
  for T in test_types
    x = fill(floatmax(T), 4)
    functions = (NormL1(zero(T)), NormL1Nonneg(zero(T)), ElasticNet(zero(T), zero(T)),
                 SqrNormL2(zero(T)), NormL2(zero(T)), NormLinf(zero(T)),
                 GroupNormL2(zero(T), groups), GroupNormLinf(zero(T), groups),
                 SparseGroupNormL2(zero(T), zero(T), groups),
                 GroupNormL2(one(T), groups, zeros(T, 2)))
    for f in functions, fg in (f, Adapt.adapt(DeviceArray, f))
      input = DeviceArray(x)
      output = similar(input)
      @test fg(input) === zero(T)
      @test value_and_prox!(output, fg, input, one(T)) === zero(T)
      @test host(output) == x
      @test value_and_prox!(input, fg, input, one(T)) === zero(T)
      @test host(input) == x
    end
    # Zero coefficient preserves the nonnegative domain.
    f = Adapt.adapt(DeviceArray, NormL1Nonneg(zero(T)))
    @test f(DeviceArray(T[-1, 1])) == T(Inf)
    # An absent elastic-net quadratic term must not multiply an overflowing
    # sum of squares by zero when the remaining l1 value is representable.
    z = DeviceArray(T[floatmax(T) / 4])
    @test Adapt.adapt(DeviceArray, ElasticNet(one(T), zero(T)))(z) == floatmax(T) / 4
  end
end

@testset "losses with a design matrix on GPU arrays" begin
  allowscalar(false)
  n, p, k = 40, 7, 3
  X, y, Y, b, B = randn(n, p), randn(n), randn(n, k), randn(p), randn(p, k)
  yb = Float64.(rand(n) .> 0.5)
  w, o = 0.5 .+ rand(n), 0.1 .* randn(n)
  for T in test_types
    Xc = T.(X)
    for f in (LeastSquaresLoss(T.(y), Xc), LeastSquaresLoss(T.(Y), Xc), GLMLoss(ProximalBase.Squared(), T.(y), Xc),
              LogisticLoss(T.(yb), Xc), PoissonLoss(T.(abs.(y)), Xc; offset = T.(o)), HuberLoss(T.(y), Xc, 0.5; weights = T.(w)),
              GLMLoss(ProximalBase.SmoothCheck(0.3, 0.2), T.(y), Xc; weights = T.(w), offset = T.(o)))
      @testset "$(nameof(typeof(f))) $(f isa GLMLoss ? nameof(typeof(f.loss)) : ndims(f.Y)), $T" begin
        fg = Adapt.adapt(DeviceArray, f)                      # X, y and the buffers on the device
        @test fg.X isa DeviceArray && (fg isa GLMLoss ? fg.η isa DeviceArray && fg.d isa DeviceArray : fg.tmp isa DeviceArray)
        βc = T.(f isa LeastSquaresLoss && ndims(f.Y) == 2 ? B : b)
        βg = DeviceArray(βc)
        @test fg(βg) ≃ f(βc)
        gc = similar(βc)
        vc = value_and_gradient!(gc, f, βc)
        gg = similar(βg)
        @test value_and_gradient!(gg, fg, βg) ≃ vc
        @test eltype(gg) == T
        @test host(gg) ≃ gc
        fill!(gg, 0)
        @test gradient!(gg, fg, βg) === gg
        @test host(gg) ≃ gc
        @test gradient(fg, βg) isa DeviceArray
        @test host(gradient(fg, βg)) ≃ gc
        # Reuse initialized buffers at a different point, then revisit the first.
        for βnew in (T(0.5) .* βc, βc)
          copyto!(βg, βnew)
          vc = value_and_gradient!(gc, f, βnew)
          @test value_and_gradient!(gg, fg, βg) ≃ vc
          @test host(gg) ≃ gc
        end
      end
    end
  end
  # the Hessian of least squares is products; its proximal operator runs on the CPU
  for T in test_types
    # Match the CPU's zero test after conversion to the working precision,
    # including higher-precision weights that underflow when converted.
    weight_sets = T == Float32 ? (T[0, 1], Float64[floatmin(Float64), 1]) : (T[0, 1],)
    for weights in weight_sets
      fzero = PoissonLoss(zeros(T, 2), ones(T, 2, 1); weights, offset=T[1000, 0])
      @test fzero(T[0]) ≃ one(T)
      @test gradient(fzero, T[0]) ≃ T[1]
      device = Adapt.adapt(DeviceArray, fzero)
      β = DeviceArray(T[0])
      g = similar(β)
      @test device(β) ≃ one(T)
      @test value_and_gradient!(g, device, β) ≃ one(T)
      @test host(g) ≃ T[1]
      @test host(gradient(device, β)) ≃ T[1]
    end
  end
  f = LeastSquaresLoss(y, X)
  fg = Adapt.adapt(DeviceArray, f)
  v = randn(p)
  @test host(hessian_vec!(DeviceArray(similar(b)), fg, DeviceArray(b), DeviceArray(v))) ≃ hessian_vec!(similar(b), f, b, v)
  @test_throws ArgumentError prox(fg, DeviceArray(b), 0.5)
  # back to the host, with the buffers there
  fh = Adapt.adapt(Array, fg)
  @test fh.tmp isa Vector{Float64} && prox(fh, b, 0.5) ≃ prox(f, b, 0.5)
  gh = Adapt.adapt(Array, Adapt.adapt(DeviceArray, LogisticLoss(yb, X; weights = w)))
  @test gh.η isa Vector{Float64} && gh.weights isa Vector{Float64} && gh(b) ≃ LogisticLoss(yb, X; weights = w)(b)
  # a negative weight is caught on the device, as on the host
  @test_throws ArgumentError GLMLoss(ProximalBase.Squared(), DeviceArray(y), DeviceArray(X); weights = DeviceArray(-w))
end

@testset "compensated group coefficients on GPU arrays" begin
  allowscalar(false)
  for T in test_types
    large, step = floatmax(T) / 2, T(4)
    groups, x = [1:1, 2:2], T[7, -8]
    for (λ, w) in ((large, inv(large)), (inv(large), large))
      coefficient = BigFloat(λ) * BigFloat(w)
      functions = (GroupNormL2(λ, groups, fill(w, 2)),
                   GroupNormLinf(λ, groups, fill(w, 2)),
                   SparseGroupNormL2(T(0.25), λ, groups, fill(w, 2)))
      for f in functions
        λ1 = f isa SparseGroupNormL2 ? BigFloat(f.λ1) : BigFloat(0)
        expected = T[sign(v) * max(abs(BigFloat(v)) - BigFloat(step) * (coefficient + λ1), 0) for v in x]
        value = T((coefficient + λ1) * sum(abs, BigFloat.(expected)))
        fg, xg = Adapt.adapt(DeviceArray, f), DeviceArray(x)
        @test fg(xg) ≃ T((coefficient + λ1) * sum(abs, BigFloat.(x)))
        yg = similar(xg)
        @test prox!(yg, fg, xg, step) === yg
        @test host(yg) ≃ expected
        @test value_and_prox!(yg, fg, xg, step) ≃ value
        @test host(yg) ≃ expected
        zg = copy(xg)
        @test value_and_prox!(zg, fg, zg, step) ≃ value
        @test host(zg) ≃ expected
      end
    end
    # A vanished group coefficient must remain an identity even when γλ overflows.
    for f in (GroupNormL2(large, groups, T[0, 1]),
              GroupNormLinf(large, groups, T[0, 1]),
              SparseGroupNormL2(T(0.25), large, groups, T[0, 1]))
      fg, xg = Adapt.adapt(DeviceArray, f), DeviceArray(x)
      expected = f isa SparseGroupNormL2 ? T[6, 0] : T[7, 0]
      value = f isa SparseGroupNormL2 ? T(1.5) : zero(T)
      @test host(prox(fg, xg, step)) ≃ expected
      yg = similar(xg)
      @test value_and_prox!(yg, fg, xg, step) ≃ value
      @test host(yg) ≃ expected
    end
    # Scaled norms also identify zero groups for exceptional inputs; no separate
    # nonzero reduction is needed to preserve these unchanged/zero projections.
    xg = DeviceArray(T[Inf, 0, NaN, 0])
    for s in (0, 2)
      fg = Adapt.adapt(DeviceArray, IndBallGroupL0(s, [1:2, 3:4]))
      expected = s == 0 ? zeros(T, 4) : T[Inf, 0, NaN, 0]
      @test isequal(host(prox(fg, xg)), expected)
    end
  end
end

@testset "infinite and zero weights on GPU arrays" begin
  allowscalar(false)
  for T in test_types, λ in (0, 1)
    w = T[1, Inf, 0, 2]
    for f in (NormL1(T(λ), w), NormL1Nonneg(T(λ), w), ElasticNet(T(λ), T(0.5), w))
      fg = Adapt.adapt(DeviceArray, f)
      for x in (T[0.5, 0, -0.3, 1], T[0.5, 0.2, -0.3, 1])
        xg = DeviceArray(x)
        # the value: finite when entry 2 is zero (and, for NormL1Nonneg, x ≥ 0),
        # Inf otherwise
        vc, vg = f(x), fg(xg)
        @test typeof(vg) == typeof(vc)
        if x[2] == 0 && !(f isa NormL1Nonneg)
          @test isfinite(vc) && vg ≈ vc
        else
          @test vc === T(Inf) && vg === T(Inf)
        end
        # the proximal operator at step one fixes entry 2 at zero
        yc = prox(f, x, T(1))
        yg = similar(xg)
        @test prox!(yg, fg, xg, T(1)) === yg
        @test eltype(yg) == T
        @test Array(yg) ≈ yc
        @test Array(yg)[2] === T(0)
        # the value and the proximal operator together
        yv = similar(xg)
        valg = value_and_prox!(yv, fg, xg, T(1))
        valc = value_and_prox!(similar(x), f, x, T(1))
        @test typeof(valg) == T
        @test eltype(yv) == T
        @test isfinite(valg) && valg ≈ valc
        @test Array(yv) ≈ yc
        @test Array(yv)[2] === T(0)
      end
    end
  end

  for T in test_types
    f = ElasticNet(T(Inf), T(0.5), T[0, 1, Inf, 0])
    for x in (T[0.5, 0, 0, 1], T[0.5, 0, 0.2, 1])
      vc, vg = f(x), Adapt.adapt(DeviceArray, f)(DeviceArray(x))
      @test typeof(vg) == T
      if x[3] == 0
        @test isfinite(vc) && vg ≈ vc
      else
        @test vc === T(Inf) && vg === T(Inf)
      end
    end
    # zero scaling before and after adaptation preserves the fixed entry
    w = T[1, Inf, 0, 2]
    for f in (NormL1(one(T), w), NormL1Nonneg(one(T), w), ElasticNet(one(T), one(T), w))
      z = Scaled(f, zero(T))
      for zg in (Adapt.adapt(DeviceArray, z), Scaled(Adapt.adapt(DeviceArray, f), zero(T)))
        for x in (T[0.5, 0, 3, 1], T[0.5, 2, 3, 1], T[-0.5, 0, 3, 1])
          xg = DeviceArray(x)
          @test zg(xg) == z(x)
          @test Array(prox(zg, xg, one(T))) ≈ prox(z, x, one(T))
        end
      end
    end
    # weights of another precision
    x = T[0.5, 0, -0.3, 1]
    for f in (ElasticNet(0.3, 0.5, Float64[1, Inf, 0, 2]), NormL1(0.3, Float64[1, Inf, 0, 2]))
      vg = Adapt.adapt(DeviceArray, f)(DeviceArray(x))
      @test typeof(vg) == T
      @test vg ≈ f(x)
    end
  end
end
@testset "weighted values at overflow on GPU arrays" begin
  allowscalar(false)
  # The same two-ulp reference check as spectral_overflow.jl.
  function check_rounded(v::T, reference) where {T}
    r = big(reference)
    k = iszero(r) ? exponent(floatmin(T)) : max(exponent(abs(r)), exponent(floatmin(T)))
    δ = 2ldexp(one(r), k - precision(T) + 1)
    lo, hi = r - δ, r + δ
    if isfinite(T(hi))
      @test isfinite(v) && lo <= big(v) <= hi
    elseif T(lo) === T(Inf)
      @test v === T(Inf)
    else
      @test v === T(Inf) || (isfinite(v) && lo <= big(v) <= hi)
    end
  end
  for T in test_types
    M = floatmax(T)
    x = DeviceArray(T[M, M])
    for (f, ref) in ((ElasticNet(T(1)/4, zero(T)), big(M)/2),
                     (NormL1(T(1)/4), big(M)/2),
                     (GroupNormL2(T(1)/4, [1:2]), big(M)/2),
                     (SparseGroupNormL2(T(1)/4, zero(T), [1:2]), big(M)/2),
                     (SparseGroupNormL2(zero(T), T(1)/4, [1:2]), big(M)/2),
                     (Scaled(NormL1(one(T)), T(1)/4), big(M)/2),
                     (Scaled(NormL2(one(T)), T(1)/4), sqrt(big(2))*big(M)/4),
                     (Scaled(GroupNormL2(one(T), [1:2]), T(1)/4), big(M)/2),
                     (Conjugate(IndBallL1(T(1)/4)), big(M)/4),
                     (Conjugate(IndBallL2(T(1)/4)), sqrt(big(2))*big(M)/4))
      for fg in (f, Adapt.adapt(DeviceArray, f))
        v = fg(x)
        @test v isa T
        @test isfinite(v)
        check_rounded(v, ref)
        @test v ≃ f(host(x))
        y = similar(x)
        if f isa Union{ElasticNet, NormL1, GroupNormL2, SparseGroupNormL2}
          v = value_and_prox!(y, fg, x, T(1)/8)
          @test isfinite(v)
          check_rounded(v, ref)
          @test v ≃ f(host(y))
        end
      end
    end
    for a in (T(Inf), T(NaN)), f in (ElasticNet(T(1)/4, zero(T)), ElasticNet(zero(T), zero(T)))
      z = DeviceArray(T[a])
      @test isequal(f(z), isnan(a) ? T(NaN) : iszero(f.λ1) ? zero(T) : T(Inf))
      @test isequal(f(z), f(host(z)))
    end
    small = T === Float32 ? T(1e-20) : T(1e-200)
    large = T === Float32 ? T(1e30) : T(1e200)
    for (λ, a) in ((T === Float32 ? 1e-50 : 1e-300, large),
                   (T === Float32 ? 1e30 : 1e300, small))
      f = ElasticNet(0, λ)
      z = DeviceArray(T[a])
      ref = big(λ)*big(a)^2/2
      @test isfinite(f(z))
      check_rounded(f(z), ref)
      @test f(z) ≃ f(host(z))
    end
    for f in (GroupNormL2(T(1)/4, [1:2]), SparseGroupNormL2(zero(T), T(1)/4, [1:2]))
      y = similar(x)
      v = value_and_prox!(y, f, x, M)
      @test host(y) ≃ prox(f, host(x), M)
      @test v ≃ f(host(y))
      @test isfinite(v)
    end
    for f in (NuclearNorm(T(1)/4),)
      X = DeviceArray(Matrix{T}(M * I, 2, 2))
      v = f(X)
      @test isfinite(v)
      check_rounded(v, big(M)/2)
      @test v ≃ f(host(X))
    end
    # Legacy reductions on exact binary inputs retain their ordinary bits.
    z = DeviceArray(T[1, 2, 3, 4])
    ext = Base.get_extension(ProximalBase, :ProximalBaseGPUArraysCoreExt)
    for f in (NormL1(T(1)/4), ElasticNet(T(1)/4, T(1)/2), NormLinf(T(1)/4),
              GroupNormL2(T(1)/4, [1:2, 3:4]), GroupNormLinf(T(1)/4, [1:2, 3:4]),
              SparseGroupNormL2(T(1)/4, T(1)/4, [1:2, 3:4]))
      old = if f isa NormL1
        ext._sum(T, ext._WeightValue(T(1)/4), abs.(z), true)
      elseif f isa ElasticNet
        T(1)/4 * ext._sum(T, ext._EnetValue(one(T)), abs.(z), true) + T(1)/2 * ext._sum(T, abs2, z)/2
      elseif f isa NormLinf
        T(1)/4 * mapreduce(abs, max, z; init=zero(T))
      else
        d = ext._on_device(f, z)
        v = zero(T)
        for b in d.layout.blocks
          X = ext._gather(z, b)
          n = f isa GroupNormLinf ? ext._segreduce(abs, max, zero(T), X, b) : ext._group_norms(X, b, T)
          v += ext._sum(T, ext._Weighted(T(1)/4), n, b.w)
        end
        f isa SparseGroupNormL2 ? v + T(1)/4 * ext._grouped_l1(T, z, d.layout) : v
      end
      @test isequal(f(z), old)
    end
  end
  if Float32 in test_types
    T = Float32
    for f in (NormL1(1f-10), NormLinf(1f-10), GroupNormL2(1f-10, [1:1]),
              GroupNormLinf(1f-10, [1:1]), SparseGroupNormL2(0f0, 1f-10, [1:1]))
      x = DeviceArray(T[2f30])
      y = similar(x)
      v = value_and_prox!(y, f, x, 1e40)
      @test host(y) ≃ prox(f, host(x), 1e40)
      @test v ≃ f(host(y))
    end
    for F in (GroupNormL2, GroupNormLinf)
      f = F(1f-10, [1:1], T[1f-30])
      x = DeviceArray(T[2f20])
      y = similar(x)
      v = value_and_prox!(y, f, x, 1e60)
      @test host(y) ≃ prox(f, host(x), 1e60)
      @test v ≃ f(host(y))
    end
    for (λ, a) in ((1e40, 1f-20), (1e-50, 1f30))
      x = DeviceArray(T[a])
      ref = big(λ)*big(a)
      for f in (NormL1(λ), NormL2(λ), NormLinf(λ), ElasticNet(λ, 0),
                GroupNormL2(λ, [1:1]), GroupNormLinf(λ, [1:1]),
                SparseGroupNormL2(λ, 0, [1:1]), SparseGroupNormL2(0, λ, [1:1]),
                Conjugate(IndBallL1(λ)), Conjugate(IndBallL2(λ)))
        for fg in (f, Adapt.adapt(DeviceArray, f))
          v = fg(x)
          @test v isa T
          @test isfinite(v)
          check_rounded(v, ref)
          @test v ≃ f(host(x))
          if f isa Union{ElasticNet, NormL1, NormL2, NormLinf, GroupNormL2, GroupNormLinf, SparseGroupNormL2}
            γ = λ > 1 ? 1e-60 : 1e40
            y = similar(x)
            v = value_and_prox!(y, fg, x, γ)
            @test host(y) ≃ prox(f, host(x), γ)
            @test v ≃ f(host(y))
          end
        end
      end
      X = DeviceArray(reshape(T[a], 1, 1))
      for f in (NuclearNorm(λ), TracePSD(λ))
        v = f(X)
        @test isfinite(v)
        check_rounded(v, ref)
        @test v ≃ f(host(X))
      end
    end
  end
end

@testset "infinite group weights on GPU arrays" begin
  allowscalar(false)
  for T in test_types, F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2),
      groups in ([[1, 2], [3]], [[1], [2, 3]]), λ in (0, 1, Inf),
      w in (T[0, Inf], T[Inf, 0], T[1, Inf], T[Inf, 1])
    for λ1 in (F === SparseGroupNormL2 ? (T(0), T(0.25)) : (T(0),))
      f = F === SparseGroupNormL2 ? F(λ1, T(λ), groups, w) : F(T(λ), groups, w)
      fixed = findfirst(isinf, w)
      other = 3 - fixed
      x = T[3, 4, 2]
      x0 = copy(x)
      x0[groups[fixed]] .= 0
      G = groups[other]
      n = F === GroupNormLinf ? maximum(abs, x0[G]) : norm(x0[G])
      value = iszero(w[other]) || iszero(λ) ? λ1 * sum(abs, x0) : T(λ) * w[other] * n + λ1 * sum(abs, x0)
      for d in (f, Adapt.adapt(DeviceArray, f))
        @test d(DeviceArray(x)) == T(Inf)
        @test same(d(DeviceArray(x0)), value)
        for γ in (zero(T), one(T))
          expected = copy(x)
          expected[groups[fixed]] .= 0
          z = max.(x[G] .- γ * λ1, 0)
          c = iszero(γ) || iszero(w[other]) || iszero(λ) ? zero(T) : T(λ) * w[other]
          if isinf(c)
            z .= 0
          elseif F === GroupNormLinf
            # These groups have at most two positive entries.
            if sum(z) <= c
              z .= 0
            else
              τ = length(z) == 1 ? z[1] - c : max((sum(z) - c)/2, maximum(z) - c)
              z = min.(z, τ)
            end
          elseif !iszero(norm(z))
            z *= max(1 - c/norm(z), 0)
          end
          expected[G] = z
          xd = DeviceArray(x)
          @test host(prox(d, xd, γ)) ≃ expected
          yd = copy(xd)
          @test same(value_and_prox!(yd, d, yd, γ), f(expected))
          @test host(yd) ≃ expected
        end
      end
      if isfinite(λ)
        z = Scaled(f, zero(T))
        for d in (z, Adapt.adapt(DeviceArray, z))
          @test d(DeviceArray(x)) == T(Inf)
          @test d(DeviceArray(x0)) == zero(T)
          @test host(prox(d, DeviceArray(x), zero(T))) == x0
        end
      end
    end
  end
end

@testset "stored group weights before device narrowing" begin
  allowscalar(false)
  for F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2)
    for λ1 in (F === SparseGroupNormL2 ? (0.0, 0.25) : (0.0,))
      for (λ, w, x, γ, expected, value) in (
          (1.0, [1e100], Float32[2], 0f0, Float32[2], Inf32),
          (0.0, [Inf, 1e100], Float32[2, 2], 0f0, Float32[0, 2], Float32(2λ1)),
          (0.0, [Inf, 1e100], Float32[2, 2], 1f0, Float32[0, 2-λ1], Float32(λ1*(2-λ1))),
          (1e-20, [1e40], Float32[2], 0f0, Float32[2], 2f20),
          (1e-20, [1e40], Float32[2], 1f-20, Float32[1], 1f20))
        groups = [[i] for i in eachindex(x)]
        f = F === SparseGroupNormL2 ? F(λ1, λ, groups, w) : F(λ, groups, w)
        for d in (f, Adapt.adapt(DeviceArray, f))
          @test same(d(DeviceArray(expected)), value)
          for inplace in (false, true)
            xd = DeviceArray(x)
            yd = inplace ? xd : similar(xd)
            v = value_and_prox!(yd, d, xd, γ)
            @test host(yd) ≃ expected
            @test same(v, value)
            @test same(v, d(yd))
            xd = DeviceArray(x)
            yd = inplace ? xd : similar(xd)
            prox!(yd, d, xd, γ)
            @test host(yd) ≃ expected
          end
        end
      end
    end
  end
end

@testset "infinite inputs in fixed device groups" begin
  allowscalar(false)
  for T in test_types, F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2),
      λ in (zero(T), one(T)), γ in (zero(T), one(T)),
      input in (T[Inf, -Inf], T[-Inf, Inf])
    f = F === SparseGroupNormL2 ? F(T(0.25), λ, [[1, 2]], T[Inf]) : F(λ, [[1, 2]], T[Inf])
    for d in (f, Adapt.adapt(DeviceArray, f)), inplace in (false, true)
      xd = DeviceArray(input)
      yd = inplace ? xd : similar(xd)
      v = value_and_prox!(yd, d, xd, γ)
      @test host(yd) == zeros(T, 2)
      @test v == zero(T)
      @test v == d(yd)
      xd = DeviceArray(input)
      yd = inplace ? xd : similar(xd)
      prox!(yd, d, xd, γ)
      @test host(yd) == zeros(T, 2)
    end
  end
end

@testset "zero scaling before and after group adaptation" begin
  allowscalar(false)
  for T in test_types, F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2),
      w in (T[1, 2], T[Inf, 1], T[1, Inf])
    groups = [[1, 2], [3]]
    f = F === SparseGroupNormL2 ? F(T(0.25), one(T), groups, w) : F(one(T), groups, w)
    d = Adapt.adapt(DeviceArray, f)
    x = T[3, -4, 2]
    expected = copy(x)
    for k in eachindex(w)
      isinf(w[k]) && (expected[groups[k]] .= 0)
    end
    zeroed = Scaled(d, zero(T))
    if all(isfinite, w)
      @test zeroed isa ZeroFunction
    else
      @test typeof(zeroed) == typeof(d)
      @test zeroed.layout === d.layout
      @test iszero(zeroed.f.λ0)
      if F === SparseGroupNormL2
        @test iszero(zeroed.f.λ1)
      end
    end
    for z in (zeroed, Adapt.adapt(DeviceArray, Scaled(f, zero(T))),
              Scaled{typeof(d), T}(d, zero(T)), Scaled(Scaled(d, T(2)), zero(T)),
              Scaled(Adapt.adapt(DeviceArray, Scaled(f, T(2))), zero(T)),
              Adapt.adapt(DeviceArray, Scaled(Scaled(f, T(2)), zero(T))))
      @test z(DeviceArray(x)) == (all(isfinite, w) ? zero(T) : T(Inf))
      @test z(DeviceArray(expected)) == zero(T)
      for γ in (zero(T), one(T)), inplace in (false, true)
        xd = DeviceArray(x)
        yd = inplace ? xd : similar(xd)
        v = value_and_prox!(yd, z, xd, γ)
        @test host(yd) == expected
        @test v == zero(T)
        @test v == z(yd)
        xd = DeviceArray(x)
        yd = inplace ? xd : similar(xd)
        prox!(yd, z, xd, γ)
        @test host(yd) == expected
      end
    end
  end
end
