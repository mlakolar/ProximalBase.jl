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

@testset "real entries on GPU arrays" begin
  allowscalar(false)
  groups = [1:2, 3:4]
  functions = (ScaledSquare(0.6), NormL1(0.3), NormL1(0), NormL1Nonneg(0.3),
               ElasticNet(0.3, 0.5), IndBox(-1, 1), IndNonnegative(),
               SqrNormL2(0.7, ones(4)), NormL2(0.3), NormLinf(0.3), NormLinf(0),
               IndBallL1(1), IndBallL1(0), IndSimplex(1), IndSimplex(0),
               GroupNormL2(0.3, groups), GroupNormL2(0, groups),
               GroupNormLinf(0.3, groups), SparseGroupNormL2(0.2, 0.3, groups),
               IndBallGroupL0(1, groups), GroupNormL2(0.3, UnitRange{Int}[]),
               GroupNormL2(0, groups, [Inf, 1.0]),
               GroupNormLinf(0, groups, [Inf, 1.0]),
               SparseGroupNormL2(0, 0, groups, [Inf, 1.0]))
  for f in functions, fg in (f, Adapt.adapt(DeviceArray, f))
    error = ArgumentError("$(nameof(typeof(fg))) requires real entries")
    for imaginary in (0, 1)
      x = DeviceArray(fill(0.5, 4))
      z = DeviceArray(fill(complex(0.5, imaginary), 4))
      @test_throws error fg(z)
      @test_throws error prox(fg, z, 0.8)
      for op in (prox!, value_and_prox!)
        for (input, output) in ((z, DeviceArray(fill(17.0, 4))),
                                (x, DeviceArray(fill(17.0 + 2im, 4))),
                                (z, DeviceArray(fill(17.0 + 2im, 4))), (z, z))
          before = host(output)
          @test_throws error op(output, fg, input, 0.8)
          @test host(output) == before
        end
      end
    end
    # Adapted group functions also accept host arrays.
    if fg !== f
      x, z = fill(0.5, 4), fill(0.5 + 0im, 4)
      @test_throws error fg(z)
      for op in (prox!, value_and_prox!), (input, output) in ((z, fill(17.0, 4)), (x, fill(17.0 + 2im, 4)))
        before = copy(output)
        @test_throws error op(output, fg, input, 0.8)
        @test output == before
      end
    end
  end
  # Diagonal and group step arrays are checked before the CPU fallback writes.
  for f in (NormL1(0.3), GroupNormL2(0.3, groups)), op in (prox!, value_and_prox!)
    x, y = DeviceArray(fill(0.5, 4)), DeviceArray(fill(17.0, 4))
    step = DeviceArray(fill(0.8 + 0im, f isa GroupNormL2 ? 2 : 4))
    @test_throws ArgumentError("$(nameof(typeof(f))) requires real entries") op(y, f, x, step)
    @test host(y) == fill(17.0, 4)
  end
  for f in (LeastSquaresLoss(ones(3), ones(3, 4)), LogisticLoss(ones(3), ones(3, 4)))
    fg = Adapt.adapt(DeviceArray, f)
    error = ArgumentError("$(nameof(typeof(fg))) requires real entries")
    # Initialize every cache so preservation does not depend on uninitialized data.
    fg(DeviceArray(fill(0.5, 4)))
    buffers = fg isa GLMLoss ? (fg.η, fg.d) : (fg.tmp,)
    foreach(b -> fill!(b, 13), buffers)
    saved = map(host, buffers)
    for imaginary in (0, 1)
      x, z = DeviceArray(fill(0.5, 4)), DeviceArray(fill(complex(0.5, imaginary), 4))
      @test_throws error fg(z)
      for op in (gradient!, value_and_gradient!),
          (input, output) in ((z, DeviceArray(fill(17.0, 4))),
                              (x, DeviceArray(fill(17.0 + 2im, 4))),
                              (z, DeviceArray(fill(17.0 + 2im, 4))), (z, z))
        before = host(output)
        @test_throws error op(output, fg, input)
        @test host(output) == before
        @test map(host, buffers) == saved
      end
      if fg isa LeastSquaresLoss
        for op in (prox!, value_and_prox!),
            (input, output) in ((z, DeviceArray(fill(17.0, 4))), (x, DeviceArray(fill(17.0 + 2im, 4))))
          before = host(output)
          @test_throws error op(output, fg, input, 0.8)
          @test host(output) == before
          @test map(host, buffers) == saved
        end
      end
    end
  end
end

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
  if Float32 in test_types
    for (w, x) in (([1 - 2.0^-26], Float32[1]), ([1e40], Float32[Inf]))
      f = ElasticNet(1, 0, w)
      fg = Adapt.adapt(DeviceArray, f)
      # The broadcast retains the weights' precision until writing Float32.
      expected = Float32.(prox(f, Float64.(x), 1.0))
      xg = DeviceArray(x)
      @test host(prox(fg, xg, 1f0)) == expected
      @test host(prox!(xg, fg, xg, 1f0)) == expected
    end
  end

end

@testset "linear separable parameters on GPU arrays" begin
  allowscalar(false)
  for T in test_types
    x = T[5, -5]
    w = T[1 2]
    l, u = T[0 2], T[1, 3]
    xg = DeviceArray(x)
    for (f, direct) in ((IndBox(l, u), IndBox(DeviceArray(l), DeviceArray(u))),
                        (NormL1(T(0.3), w), NormL1(T(0.3), DeviceArray(w))),
                        (NormL1Nonneg(T(0.3), w), NormL1Nonneg(T(0.3), DeviceArray(w))),
                        (ElasticNet(T(0.3), T(0.5), w), ElasticNet(T(0.3), T(0.5), DeviceArray(w))))
      for fg in (Adapt.adapt(DeviceArray, f), direct, Adapt.adapt(DeviceArray, direct))
        @test same(fg(xg), f(x))
        yc = prox(f, x, T(0.8))
        yg = similar(xg)
        @test prox!(yg, fg, xg, T(0.8)) === yg
        @test host(yg) ≃ yc
        @test same(value_and_prox!(yg, fg, xg, T(0.8)), f(yc))
        @test host(yg) ≃ yc
        for op in (prox!, value_and_prox!)
          z = copy(xg)
          op(z, fg, z, T(0.8))
          @test host(z) ≃ yc
        end
      end
    end
    @test_throws DimensionMismatch IndBox(DeviceArray(T[0 0]), DeviceArray(T[1]))
  end
end

@testset "nonconvex box intersections on GPU arrays" begin
  allowscalar(false)
  for f in (MCP(1.), SCAD(1.), NormL0(1.))
    for (a, b) in ((IndBox([0 10], 20), IndBox(-1, [5, 15])),
                   (IndBox([100_000 100_001], Float16(Inf)), IndBox(Float16(0), Float16[Inf, Inf])),
                   (IndBox(Float16(-Inf), [-100_000 -100_001]), IndBox(Float16[-Inf, -Inf], Float16(0))),
                   (IndBox(zeros(0, 2), 3), IndBox(0, Int[])))
      for (first, second) in ((a, b), (b, a))
        c = Constrained(Constrained(f, first), second)
        before = Adapt.adapt(DeviceArray, c)
        after = Constrained(Adapt.adapt(DeviceArray, Constrained(f, first)), Adapt.adapt(DeviceArray, second))
        for cg in (before, after)
          @test isequal(host(cg.box.l), c.box.l)
          @test isequal(host(cg.box.u), c.box.u)
          @test size(cg.box.l) == size(cg.box.u) == size(c.box.l)
          @test eltype(cg.box.l) == eltype(c.box.l)
          @test eltype(cg.box.u) == eltype(c.box.u)
        end
      end
    end
    @test_throws DimensionMismatch Constrained(
      Constrained(f, IndBox(DeviceArray([0, 1]), 3)), IndBox(0, DeviceArray([3, 4, 5])))
    k = 2^53
    @test_throws ArgumentError Constrained(
      Constrained(f, IndBox(DeviceArray([k + 1]), k + 1)), IndBox(k, DeviceArray([k])))
  end
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
  # and so are infinite weights and offsets (#251)
  @test_throws ArgumentError GLMLoss(ProximalBase.Squared(), DeviceArray(y), DeviceArray(X); weights = DeviceArray([w[1:end-1]; Inf]))
  @test_throws ArgumentError GLMLoss(ProximalBase.Squared(), DeviceArray(y), DeviceArray(X); offset = DeviceArray([w[1:end-1]; NaN]))
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

@testset "host spectral oracles on device matrices" begin
  allowscalar(false)
  square = [3.0 1 0; 1 4 1; 0 1 5]
  factories = (() -> NuclearNorm(0.5), () -> NuclearNorm(0),
               () -> IndBallRank(1), () -> IndBallRank(3), () -> IndStiefel(),
               () -> TracePSD(0.5), () -> TracePSD(0), () -> IndPSD(),
               () -> IndPSD(rank = 1), () -> IndPSD(rank = 3),
               () -> Spectral(NormL1(0.5)), () -> SingularValue(NormL1(0.5)),
               () -> LogDetLoss(Matrix{Float64}(I, 3, 3)))
  for T in test_types, make in factories
    X = T.(square)
    f, fd = make(), make()
    D = DeviceArray(X)
    @test same(fd(D), f(X))
    for R in (T, Float32)
      Y, DY = R.(X), DeviceArray(R.(X))
      vc = value_and_prox!(Y, f, X, 0.8)
      @test same(value_and_prox!(DY, fd, D, 0.8), vc)
      @test host(DY) ≃ Y
      @test prox!(DY, fd, D, 0.8) === DY
      @test host(DY) ≃ Y
      @test host(D) == X
    end
    @test host(prox(fd, D, 0.8)) ≃ prox(f, X, 0.8)
    Y = prox(f, X, 0.8)
    @test prox!(D, fd, D, 0.8) === D
    @test host(D) ≃ Y
    D = DeviceArray(X)
    @test same(value_and_prox!(D, fd, D, 0.8), value_and_prox!(Y, f, X, 0.8))
    @test host(D) ≃ Y
    # Shape errors must not write even part of the output.
    bad = DeviceArray(fill(T(17), 2, 3))
    for op in (prox!, value_and_prox!)
      @test_throws DimensionMismatch op(bad, fd, DeviceArray(X), 0.8)
      @test host(bad) == fill(T(17), 2, 3)
    end
  end
  for T in test_types, X in (T[2 1; -1 3], T[-2 0; 0 1],
                              fill(T(2), 1, 1), zeros(T, 2, 2), T[-2 0; 0 -1], T[Inf 0; 0 1], T[NaN 0; 0 1])
    n = size(X, 1)
    for f in (NuclearNorm(0.5), IndBallRank(1), IndStiefel(), TracePSD(0.5),
              IndPSD(), IndPSD(rank = 1), Spectral(NormL1(0.5)),
              SingularValue(NormL1(0.5)), LogDetLoss(Matrix{T}(I, n, n)))
      @test same(f(DeviceArray(X)), f(X))
      if !all(isfinite, X)
        Y = fill(T(17), size(X))
        DY = DeviceArray(Y)
        cpu = try
          (:value, value_and_prox!(Y, f, X, 0.8))
        catch e
          (:error, typeof(e), sprint(showerror, e))
        end
        device = try
          (:value, value_and_prox!(DY, f, DeviceArray(X), 0.8))
        catch e
          (:error, typeof(e), sprint(showerror, e))
        end
        @test isequal(device, cpu) || (device[1] === cpu[1] === :value && same(device[2], cpu[2]))
        @test isequal(host(DY), Y) || host(DY) ≃ Y
      else
        @test host(prox(f, DeviceArray(X), 0.8)) ≃ prox(f, X, 0.8)
      end
    end
  end
  for T in test_types, dims in ((3, 2), (2, 3))
    X = reshape(T.(1:prod(dims)), dims)
    for f in (NuclearNorm(0.5), IndBallRank(1), IndBallRank(3), SingularValue(NormL1(0.5)))
      @test same(f(DeviceArray(X)), f(X))
      Y, D, DY = similar(X), DeviceArray(X), DeviceArray(zeros(T, dims))
      @test same(value_and_prox!(DY, f, D, 0.8), value_and_prox!(Y, f, X, 0.8))
      @test host(DY) ≃ Y
      @test prox!(DY, f, D, 0.8) === DY
    end
  end
  for T in test_types
    X = T.(square)
    f = LogDetLoss(Matrix{T}(I, 3, 3))
    for R in (T, Float32), A in (X, -X)
      G, D = zeros(R, 3, 3), DeviceArray(A)
      DG = DeviceArray(G)
      @test same(value_and_gradient!(DG, f, D), value_and_gradient!(G, f, A))
      @test isequal(host(DG), G) || host(DG) ≃ G
      @test gradient!(DG, f, D) === DG
      @test isequal(host(DG), G) || host(DG) ≃ G
      if R === T
        before = copy(A)
        @test_throws ArgumentError("the gradient must not overwrite x") value_and_gradient!(A, f, A)
        @test_throws ArgumentError("the gradient must not overwrite x") value_and_gradient!(D, f, D)
        @test host(D) == before
      end
    end
    f = LogDetLoss(Matrix{Float32}(I, 3, 3))
    A = Float64.(X)
    D, DY = DeviceArray(A), DeviceArray(zeros(Float32, 3, 3))
    Y = zeros(Float32, 3, 3)
    @test same(value_and_prox!(DY, f, D, 0.8), value_and_prox!(Y, f, A, 0.8))
    @test host(DY) ≃ Y
    input_buffer, output_buffer = f.out.buf[1].buf, f.out.buf[2].buf
    value_and_prox!(DY, f, D, 0.8)
    @test f.out.buf[1].buf === input_buffer
    @test f.out.buf[2].buf === output_buffer
    @test same(value_and_gradient!(DY, f, D), value_and_gradient!(Y, f, A))
    @test host(DY) ≃ Y
    M = floatmax(T)
    # Wrapper proximal operators also recover an overflowing spectrum on the host.
    for make in (() -> Spectral(NormL1(T(1)/4)), () -> SingularValue(NormL1(T(1)/4)),
                 () -> Spectral(IndNonnegative()))
      Xoverflow = fill(M, 2, 2)
      Yoverflow = similar(Xoverflow)
      vc = value_and_prox!(Yoverflow, make(), Xoverflow, one(T))
      Doverflow = DeviceArray(Xoverflow)
      DYoverflow = similar(Doverflow)
      vd = value_and_prox!(DYoverflow, make(), Doverflow, one(T))
      @test vd === vc
      @test host(DYoverflow) == Yoverflow
    end
    # Mapped roots and the half spectrum can overflow at finite inputs.
    for n in (2, 3), (w, γ) in ((one(T), one(T)), (T(1)/4, T(4)))
      Xlarge = fill(M, n, n)
      Slarge = Matrix{T}(I, n, n)
      Ylarge = similar(Xlarge)
      vc = value_and_prox!(Ylarge, LogDetLoss(Slarge, w), Xlarge, γ)
      Dlarge = DeviceArray(Xlarge)
      DYlarge = similar(Dlarge)
      vd = value_and_prox!(DYlarge, LogDetLoss(Slarge, w), Dlarge, γ)
      @test vd === vc
      @test host(DYlarge) == Ylarge
      if n == 2
        @test isone(w) ? vc === T(Inf) : isapprox(vc, M / 2)
      end
    end
    f = TracePSD(T(0.25))
    Xlarge = Matrix{T}(M * I, 2, 2)
    @test f(DeviceArray(Xlarge)) == f(Xlarge) == M / 2
    Xoverflow = fill(M / 2, 3, 3)
    for f in (NuclearNorm(T(0.25)), IndBallRank(1), TracePSD(T(0.25)),
              IndPSD(), IndPSD(rank = 1), Spectral(IndBox(-M, M)),
              SingularValue(IndBallLinf(M)))
      @test same(f(DeviceArray(Xoverflow)), f(Xoverflow))
      Y, D = similar(Xoverflow), DeviceArray(Xoverflow)
      DY = similar(D)
      @test same(value_and_prox!(DY, f, D, 0.8), value_and_prox!(Y, f, Xoverflow, 0.8))
      @test host(DY) ≃ Y
    end
    # Partial objects start independently, with the same deterministic seed.
    Xpartial = Matrix(Diagonal(vcat(T[5, 3], fill(T(-1), 94))))
    for make in (() -> NuclearNorm(2; partial = true), () -> IndBallRank(1; partial = true),
                 () -> TracePSD(2; partial = true), () -> IndPSD(partial = true),
                 () -> IndPSD(rank = 1, partial = true))
      fc, fd = make(), make()
      for _ in 1:2
        Y, D = similar(Xpartial), DeviceArray(Xpartial)
        DY = similar(D)
        @test same(value_and_prox!(DY, fd, D, 1), value_and_prox!(Y, fc, Xpartial, 1))
        @test host(DY) ≃ Y
      end
    end
    D = DeviceArray(X)
    c = fill(T(0.1), 3, 3)
    for make in (a -> Scaled(TracePSD(0.5), 2), a -> Tilt(TracePSD(0.5), a),
                 a -> Translate(TracePSD(0.5), a), a -> Regularize(TracePSD(0.5), 0.3, a))
      fc, fd = make(c), make(DeviceArray(c))
      Y, DY = similar(X), similar(D)
      @test same(value_and_prox!(DY, fd, D, 0.8), value_and_prox!(Y, fc, X, 0.8))
      @test host(DY) ≃ Y
      @test prox!(DY, fd, D, 0.8) === DY
    end
    adapted = Adapt.adapt(DeviceArray, Regularize(TracePSD(0.5), 0.3))
    @test host(prox(adapted, D, 0.8)) ≃ prox(Regularize(TracePSD(0.5), 0.3), X, 0.8)
    for f in (Spectral(NormL1()), IndStiefel())
      A = DeviceArray(zeros(T, 2, 3))
      Y = DeviceArray(fill(T(17), 2, 3))
      @test_throws DimensionMismatch value_and_prox!(Y, f, A, 1)
      @test host(Y) == fill(T(17), 2, 3)
    end
    A = DeviceArray(T[Inf 0; 0 1])
    Y = DeviceArray(fill(T(17), 2, 2))
    @test prox!(Y, IndStiefel(), A, 1) === Y
    @test all(isnan, host(Y))
    @test isequal(host(Y), prox(IndStiefel(), host(A), 1))
    Y = DeviceArray(fill(T(17), 3, 3))
    @test_throws DomainError prox!(Y, LogDetLoss(Matrix{T}(I, 3, 3)), D, -1)
    @test host(Y) == fill(T(17), 3, 3)
  end
  # These matrix-set intersections predate the host decomposition paths.
  D = typeof(DeviceArray(zeros(Float32, 1)))
  F = Nothing
  device_value = which(ProximalBase._value_dense, Tuple{F, D})
  device_prox = which(ProximalBase._prox_sep!, Tuple{D, F, D, Float64})
  known = (Set((device_value, which(ProximalBase._value_dense, Tuple{IndPattern, Vector{Float64}}))),
           Set((device_value, which(ProximalBase._value_dense, Tuple{IndBanded, Matrix{Float64}}))),
           Set((device_prox, which(ProximalBase._prox_sep!, Tuple{Matrix{Float64}, IndBanded, Matrix{Float64}, Float64}))))
  @test all(pair -> Set(pair) in known, Test.detect_ambiguities(ProximalBase))
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

@testset "conjugate real entries on GPU arrays" begin
  allowscalar(false)
  error = ArgumentError("Conjugate requires real entries")
  for wrap in (identity, a -> view(a, 2:4))
    for f in (NormL1(1, ones(3)), IndNonnegative(), IndBox(-ones(3), ones(3)),
              ElasticNet(1, 1, ones(3)), Berhu(1, 1))
      c = Conjugate(Adapt.adapt(DeviceArray, f))
      for imaginary in (0, 1)
        x = wrap(DeviceArray(fill(0.5, wrap === identity ? 3 : 5)))
        z = wrap(DeviceArray(fill(complex(0.5, imaginary), wrap === identity ? 3 : 5)))
        # A rejected value must not initialize the working buffer.
        @test_throws error c(z)
        @test c.scratch.buf === nothing
        for step in (0.5, wrap(DeviceArray(fill(0.5, wrap === identity ? 3 : 5)))),
            op in (prox!, value_and_prox!)
          for (input, output) in ((z, wrap(DeviceArray(fill(17.0, wrap === identity ? 3 : 5)))),
                                  (x, wrap(DeviceArray(fill(17.0 + 2im, wrap === identity ? 3 : 5)))),
                                  (z, wrap(DeviceArray(fill(17.0 + 2im, wrap === identity ? 3 : 5)))), (z, z))
            storage = wrap === identity ? output : parent(output)
            before = host(storage)
            @test_throws error op(output, c, input, step)
            @test host(storage) == before
            @test c.scratch.buf === nothing
          end
          output = wrap(DeviceArray(fill(17.0, wrap === identity ? 3 : 5)))
          storage = wrap === identity ? output : parent(output)
          before = host(storage)
          complexstep = wrap(DeviceArray(fill(0.5 + 0im, wrap === identity ? 3 : 5)))
          @test_throws error op(output, c, x, complexstep)
          @test host(storage) == before
          @test c.scratch.buf === nothing
        end
      end
    end
  end
  # A late complex block must leave every earlier block and cache unchanged.
  c = Conjugate(SeparableSum(NormL1(), SeparableSum(SqrNormL2(), NormL1())))
  for imaginary in (0, 1)
    x = (DeviceArray(fill(0.5, 3)), (DeviceArray(fill(0.5, 3)), DeviceArray(fill(0.5, 3))))
    z = (x[1], (x[2][1], DeviceArray(fill(complex(0.5, imaginary), 3))))
    @test_throws error c(z)
    for op in (prox!, value_and_prox!)
      for (input, output) in ((z, (DeviceArray(fill(17.0, 3)), (DeviceArray(fill(17.0, 3)), DeviceArray(fill(17.0, 3))))),
                              (x, (DeviceArray(fill(17.0, 3)), (DeviceArray(fill(17.0, 3)), DeviceArray(fill(17.0 + 2im, 3))))),
                              (z, (DeviceArray(fill(17.0 + 2im, 3)), (DeviceArray(fill(17.0, 3)), DeviceArray(fill(17.0 + 2im, 3))))), (z, z))
        blocks = (output[1], output[2]...)
        before = map(host, blocks)
        @test_throws error op(output, c, input, 0.5)
        @test map(host, blocks) == before
        @test c.scratch.buf === nothing
      end
    end
  end
end

@testset "separable array-step real entries on GPU arrays" begin
  allowscalar(false)
  for f in (Berhu(1, 1), ElasticNet(1, 1, ones(3))),
      wrap in (identity, a -> view(a, 2:4))
    fg = Adapt.adapt(DeviceArray, f)
    error = ArgumentError("$(nameof(typeof(fg))) requires real entries")
    device(v) = wrap(DeviceArray(fill(v, wrap === identity ? 3 : 5)))
    x, step = device(0.5), device(0.5)
    for imaginary in (0, 1)
      z = device(complex(0.5, imaginary))
      for op in (prox!, value_and_prox!, ProximalBase._prox_sep!)
        for (input, output, γ) in ((z, device(17.0), step), (x, device(17.0 + 2im), step),
                                   (z, device(17.0 + 2im), step), (z, z, step),
                                   (x, device(17.0), device(complex(0.5, imaginary))))
          storage = wrap === identity ? output : parent(output)
          before = host(storage)
          @test_throws error op(output, fg, input, γ)
          @test host(storage) == before
        end
      end
    end
  end
end

@testset "conjugate scalar steps on GPU arrays" begin
  allowscalar(false)
  for T in test_types, dims in ((3,), (3, 2))
    x = reshape(repeat(T[0.5, -2, 1.5], prod(dims) ÷ 3), dims)
    xg = DeviceArray(x)
    for f in (NormL1(), NormL1(T(0.7), fill(T(0.8), dims)), IndNonnegative(),
              IndBox(-1, 1), IndBallL2(1), SqrNormL2(), ElasticNet(1, 1))
      c = Conjugate(f)
      cg = Conjugate(Adapt.adapt(DeviceArray, f))
      expected = prox(c, x, T(0.5))
      y = similar(xg)
      @test prox!(y, cg, xg, T(0.5)) === y
      @test host(y) ≃ expected
      buffer = cg.scratch.buf
      @test same(value_and_prox!(y, cg, xg, T(0.5)), value_and_prox!(similar(x), c, x, T(0.5)))
      @test cg.scratch.buf === buffer
      @test host(y) ≃ expected
      alias = copy(xg)
      @test host(prox!(alias, cg, alias, T(0.5))) ≃ expected
      alias = copy(xg)
      @test same(value_and_prox!(alias, cg, alias, T(0.5)), value_and_prox!(similar(x), c, x, T(0.5)))
      @test host(alias) ≃ expected
    end
    # A single cache can move between array kinds and sizes.
    c = Conjugate(SqrNormL2())
    @test prox(c, x, T(0.5)) ≃ x ./ T(1.5)
    @test host(prox(c, xg, T(0.5))) ≃ x ./ T(1.5)
    @test prox(c, x, T(0.5)) ≃ x ./ T(1.5)
    small = DeviceArray(T[1, 2])
    @test host(prox(c, small, T(0.5))) ≃ T[1, 2] ./ T(1.5)
    v = view(DeviceArray(T[0.5, -2, 1.5, 9]), 1:3)
    for f in (NormL1(), SqrNormL2(), IndBox(-1, 1))
      @test host(prox(Conjugate(f), v, T(0.5))) ≃ prox(Conjugate(f), T[0.5, -2, 1.5], T(0.5))
    end
  end
  # Parameters with the right length keep the CPU's linear pairing.
  for T in test_types, dims in ((2,), (2, 2))
    x = reshape(repeat(T[0.5, 2], prod(dims) ÷ 2), dims)
    w = ndims(x) == 1 ? reshape(T[1, 3], 1, 2) : T[1, 3, 2, 4]
    xg = DeviceArray(x)
    for f in (NormL1(1, w), NormL1Nonneg(1, w), ElasticNet(1, 1, w), IndBox(zero.(w), w))
      fg = Adapt.adapt(DeviceArray, f)
      # The CPU box Moreau helper requires bounds shaped like x.
      reference = f isa IndBox ? IndBox(reshape(f.l, size(x)), reshape(f.u, size(x))) : f
      c, cg = Conjugate(reference), Conjugate(fg)
      expected = prox(c, x, T(0.5))
      y = similar(xg)
      @test host(prox!(y, cg, xg, T(0.5))) ≃ expected
      @test same(value_and_prox!(y, cg, xg, T(0.5)), value_and_prox!(similar(x), c, x, T(0.5)))
      @test host(y) ≃ expected
      @test same(fg(xg), f(x))
    end
  end

end

@testset "conjugate values on GPU arrays" begin
  allowscalar(false)
  for T in test_types, dims in ((3,), (3, 2))
    for f in (NormL1(), NormL1(T(0.7), fill(T(0.8), dims)), IndNonnegative(),
              IndBox(-1, 1), IndBox(fill(T(-2), dims), fill(T(3), dims)), IndBox(-Inf, Inf))
      c, cg = Conjugate(f), Conjugate(Adapt.adapt(DeviceArray, f))
      for v in (T(-0.2), T(0), T(2))
        x = fill(v, dims)
        @test same(cg(DeviceArray(x)), c(x))
      end
    end
  end
  for T in test_types, dims in ((2,), (2, 2))
    x = reshape(repeat(T[0.5, 2], prod(dims) ÷ 2), dims)
    w = ndims(x) == 1 ? reshape(T[1, 3], 1, 2) : T[1, 3, 2, 4]
    for f in (NormL1(1, w), IndBox(zero.(w), w))
      @test same(Conjugate(Adapt.adapt(DeviceArray, f))(DeviceArray(x)), Conjugate(f)(x))
    end
  end
  if Float32 in test_types
    x = Float32[0.5, 2]
    w = reshape(Float32[0, Inf], 1, 2)
    for λ in (BigFloat(0), BigFloat(1), BigFloat(Inf)), weights in (nothing, w)
      f = NormL1(λ, weights)
      @test same(Conjugate(Adapt.adapt(DeviceArray, f))(DeviceArray(x)), Conjugate(f)(x))
    end
  end
  if Float32 in test_types
    for λ in (1e40, BigFloat(1e40)), x in (Float32[1e20], Float32[1e9])
      f = NormL1(λ, Float32[1e-30])
      @test same(Conjugate(Adapt.adapt(DeviceArray, f))(DeviceArray(x)), Conjugate(f)(x))
    end
  end

end

@testset "conjugate array steps on GPU arrays" begin
  allowscalar(false)
  for T in test_types, dims in ((3,), (3, 2))
    x = reshape(repeat(T[0.5, -2, 1.5], prod(dims) ÷ 3), dims)
    γ = reshape(repeat(T[0.5, 1, 2], prod(dims) ÷ 3), dims)
    xg, γg = DeviceArray(x), DeviceArray(γ)
    @test host(prox(Berhu(1, 1), xg, γg)) ≃ prox(Berhu(1, 1), x, γ)
    for f in (NormL1(), IndNonnegative(), IndBox(-1, 1), ElasticNet(1, 1), Berhu(1, 1),
              ElasticNet(1, 1, fill(T(0.8), dims)))
      c, cg = Conjugate(f), Conjugate(Adapt.adapt(DeviceArray, f))
      expected = prox(c, x, γ)
      y = similar(xg)
      @test host(prox!(y, cg, xg, γg)) ≃ expected
      buffer = cg.scratch.buf
      @test same(value_and_prox!(y, cg, xg, γg), value_and_prox!(similar(x), c, x, γ))
      @test host(y) ≃ expected
      @test cg.scratch.buf === buffer
      alias = copy(xg)
      @test host(prox!(alias, cg, alias, γg)) ≃ expected
      alias = copy(xg)
      @test same(value_and_prox!(alias, cg, alias, γg), value_and_prox!(similar(x), c, x, γ))
      @test host(alias) ≃ expected
    end
    for f in (SqrNormL2(), IndBallL2(1))
      @test_throws MethodError prox(Conjugate(f), x, γ)
      @test_throws MethodError prox(Conjugate(f), xg, γg)
    end
  end
  for T in test_types, dims in ((2,), (2, 2))
    x = reshape(repeat(T[0.5, 2], prod(dims) ÷ 2), dims)
    w = ndims(x) == 1 ? reshape(T[1, 3], 1, 2) : T[1, 3, 2, 4]
    γ = reshape(repeat(T[0.5, 1], prod(dims) ÷ 2), dims)
    xg, γg = DeviceArray(x), DeviceArray(γ)
    for f in (NormL1(1, w), IndBox(zero.(w), w), ElasticNet(1, 1, w), Berhu(1, 1))
      reference = f isa IndBox ? IndBox(reshape(f.l, size(x)), reshape(f.u, size(x))) : f
      c, cg = Conjugate(reference), Conjugate(Adapt.adapt(DeviceArray, f))
      expected = prox(c, x, γ)
      y = similar(xg)
      @test host(prox!(y, cg, xg, γg)) ≃ expected
      @test same(value_and_prox!(y, cg, xg, γg), value_and_prox!(similar(x), c, x, γ))
      @test host(y) ≃ expected
    end
    # Direct separable proxes also accept steps with another shape.
    γother = reshape(γ, size(w))
    @test host(prox(Berhu(1, 1), xg, DeviceArray(γother))) ≃ prox(Berhu(1, 1), x, γother)
    f = ElasticNet(1, 1, w)
    @test host(prox(Adapt.adapt(DeviceArray, f), xg, DeviceArray(γother))) ≃ prox(f, x, γother)
  end
  if Float32 in test_types
    x, γ = Float32[0.5, 2], Float32[0.5, 1]
    for λ in (BigFloat(0), BigFloat(1), BigFloat(Inf)), weights in (nothing, reshape(Float32[0, Inf], 1, 2))
      f = ElasticNet(λ, BigFloat(1), weights)
      c, cg = Conjugate(f), Conjugate(Adapt.adapt(DeviceArray, f))
      xg, γg = DeviceArray(x), DeviceArray(γ)
      @test host(prox(cg, xg, γg)) ≃ prox(c, x, γ)
      y = similar(xg)
      @test same(value_and_prox!(y, cg, xg, γg), value_and_prox!(similar(x), c, x, γ))
      @test host(y) ≃ prox(c, x, γ)
    end
  end
  if Float32 in test_types
    for λ in (1e40, BigFloat(1e40))
      x, γ = Float32[2e10], Float32[1]
      f = ElasticNet(λ, 1, Float32[1e-30])
      c, cg = Conjugate(f), Conjugate(Adapt.adapt(DeviceArray, f))
      xg, γg = DeviceArray(x), DeviceArray(γ)
      y = similar(xg)
      @test host(prox!(y, cg, xg, γg)) ≃ prox(c, x, γ)
      @test same(value_and_prox!(y, cg, xg, γg), value_and_prox!(similar(x), c, x, γ))
      @test host(y) ≃ prox(c, x, γ)
    end
  end

end

conjugate_blocks_match(a::AbstractArray, b::AbstractArray) = host(a) ≃ b
conjugate_blocks_match(a::Tuple, b::Tuple) = all(map(conjugate_blocks_match, a, b))

@testset "conjugate checks and tuples on GPU arrays" begin
  allowscalar(false)
  for T in test_types
    x = T[0.5, -2, 1.5]
    xg = DeviceArray(x)
    for γ in (nextfloat(zero(T)), DeviceArray(fill(nextfloat(zero(T)), 3))),
        op in (prox!, value_and_prox!), alias in (false, true)
      y = alias ? copy(xg) : DeviceArray(fill(T(19), 3))
      before = host(y)
      c = Conjugate(ElasticNet(1, 1))
      @test_throws ArgumentError op(y, c, alias ? y : xg, γ)
      @test host(y) == before
    end
    for (f, blocks, deviceblocks) in (
        (SeparableSum(NormL1(), SqrNormL2()), (x, copy(x)), (xg, copy(xg))),
        (SeparableSum(NormL1(), SeparableSum(SqrNormL2(), NormL1())),
         (x, (copy(x), copy(x))), (xg, (copy(xg), copy(xg)))))
      c = Conjugate(f)
      expected = prox(Conjugate(f), blocks, T(0.5))
      y = map(ProximalBase._gradient_buffer, deviceblocks)
      for _ in 1:2
        @test prox!(y, c, deviceblocks, T(0.5)) === y
        @test ProximalBase._tuple_buffer_type(typeof(y)) === typeof(c.scratch.buf[3])
        buffer = c.scratch.buf
        @test same(value_and_prox!(y, c, deviceblocks, T(0.5)), value_and_prox!(map(ProximalBase._gradient_buffer, blocks), Conjugate(f), blocks, T(0.5)))
        @test c.scratch.buf === buffer
        @test conjugate_blocks_match(y, expected)
      end
    end
  end
end
