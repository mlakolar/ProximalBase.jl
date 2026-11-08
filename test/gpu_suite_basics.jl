# The representative GPU cases, included after gpu_prelude.jl: one or two functions for each
# device path (separable, nonseparable, group layouts on the host and the device, adapted
# parameters, losses, array steps, overflow, infinite weights), each testset calling
# allowscalar(false). gpu_suite.jl keeps the rest of each split loop and the other cases.
# At the end: _maximum_abs of a Diagonal of a device vector (dispatch, values, a LeastSquaresLoss
# with a Diagonal design) and the guard against ambiguities with the GPU extension
# (gpu_ambiguities.jl), which every GPU run executes (gpu_smoke in the Full tier, and the
# physical CUDA run); the Quick tier runs the guard alone (gpu_extension_ambiguities.jl).
# The entry point supplies DeviceArray, test_types, host, and ≃.

@testset "separable functions on GPU arrays" begin
  allowscalar(false)
  n = 300
  x, w, l, u = 2 .* randn(n), 0.5 .+ rand(n), -rand(n), rand(n)
  for T in test_types
    xc = T.(x)
    xg = DeviceArray(xc)
    # the remaining functions are in gpu_suite.jl
    for f in (NormL1(0.3), NormL1(0.3, T.(w)), ScaledSquare(T(0.6)))
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
end

@testset "norms, balls and the simplex on GPU arrays" begin
  allowscalar(false)
  n = 300
  x = 2 .* randn(n)
  for T in test_types
    xc = T.(x)
    xg = DeviceArray(xc)
    # the remaining functions are in gpu_suite.jl
    for f in (NormL2(0.3), IndSimplex(1.))
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
    # the remaining functions are in gpu_suite.jl
    for f in (GroupNormL2(0.3, groups),)
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

@testset "losses with a design matrix on GPU arrays" begin
  allowscalar(false)
  n, p, k = 40, 7, 3
  X, y, Y, b, B = randn(n, p), randn(n), randn(n, k), randn(p), randn(p, k)
  yb = Float64.(rand(n) .> 0.5)
  w, o = 0.5 .+ rand(n), 0.1 .* randn(n)
  for T in test_types
    Xc = T.(X)
    # the remaining losses are in gpu_suite.jl
    for f in (LeastSquaresLoss(T.(y), Xc), LogisticLoss(T.(yb), Xc))
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
end

@testset "GLM weights and predictors that leave the range on GPU arrays" begin
  allowscalar(false)
  # the remaining cases are in value_overflow_glm.jl
  for T in test_types
    M = floatmax(T)
    lo, hi, ηa, ηb = T === Float32 ? (1f-35, 1f35, 100f0, 120f0) : (1e-300, 1e300, 1000.0, 800.0)
    for f in (PoissonLoss(T[0], ones(T, 1, 1); weights = T[lo]),                       # exp(η) overflows before w
              LogisticLoss(T[1], ones(T, 1, 1); weights = T[hi]),                       # the tail underflows before w
              GLMLoss(ProximalBase.Squared(), T[0.5], reshape(T[M, -M, 1], 1, 3); weights = T[1.5]))   # X β cancels
      @testset "$(nameof(typeof(f.loss))), $T" begin
        βc = length(f.y) == 1 && size(f.X, 2) == 1 ? T[f isa PoissonLoss ? ηa : ηb] : T[2, 2, 3]
        fg = Adapt.adapt(DeviceArray, f)
        βg = DeviceArray(βc)
        vc = f(βc)
        @test isfinite(vc) && vc > floatmin(T)
        @test fg(βg) ≃ vc
        gc = similar(βc)
        @test value_and_gradient!(gc, f, βc) ≃ vc
        gg = similar(βg)
        @test value_and_gradient!(gg, fg, βg) ≃ vc
        @test host(gg) ≃ gc
        fill!(gg, 0)
        gradient!(gg, fg, βg)
        @test host(gg) ≃ gc
      end
    end
  end
end

@testset "SmoothCheck without intermediates out of range on GPU arrays" begin
  allowscalar(false)
  # the review's witnesses (PR #539): the value against the formula in BigFloat
  setprecision(BigFloat, 512) do
    for T in test_types
      F32 = T === Float32
      for (τ, h, u) in ((T(0.5), F32 ? T(1e-15) : T(1e-150), F32 ? T(1e-22) : T(1e-160)),
                        (F32 ? T(1e-30) : T(1e-200), F32 ? T(1e30) : T(1e300), F32 ? T(2) : T(2e100)))
        tb, hb, ub = big(τ), big(h), big(u)
        vr = ub >= hb * tb ? tb * ub - hb * tb^2 / 2 : ub^2 / (2hb)
        dr = ub >= hb * tb ? -tb : -ub / hb
        for w in (T(0.5), T(2))
          f = GLMLoss(ProximalBase.SmoothCheck(τ, h), DeviceArray(T[0]), DeviceArray(ones(T, 1, 1)); weights = DeviceArray(T[w]))
          β, g = DeviceArray(T[-u]), DeviceArray(zeros(T, 1))
          @test abs(big(f(β)) - big(w) * vr) <= 3eps(T) * big(w) * abs(vr)
          @test abs(big(value_and_gradient!(g, f, β)) - big(w) * vr) <= 3eps(T) * big(w) * abs(vr)
          @test abs(big(host(g)[1]) - big(w) * dr) <= 3eps(T) * big(w) * abs(dr)
        end
      end
    end
  end
end

@testset "inactive observations with non-finite rows on GPU arrays" begin
  allowscalar(false)
  # an observation with weight zero takes no part: the oracles equal those of the
  # problem without it, also when the active ones need the recovery (PR #539)
  for T in test_types
    F32 = T === Float32
    tiny, bb = F32 ? T(1e-30) : T(1e-300), F32 ? T(100) : T(1000)
    Xa, ya, wa = T[1 0.5; 0.25 1], T[0, 1], T[tiny, 2]
    β = T[bb, -1]
    fa = GLMLoss(ProximalBase.Poisson(), DeviceArray(ya), DeviceArray(Xa); weights = DeviceArray(wa))
    ga = DeviceArray(zeros(T, 2))
    va = value_and_gradient!(ga, fa, DeviceArray(β))
    @test isfinite(va) && all(isfinite, host(ga))
    for (xrow, yi) in ((T[0 0], T(NaN)), (T[NaN 1], T(0)), (T[Inf -Inf], T(Inf)), (T[NaN NaN], T(-Inf)))
      X = vcat(Xa[1:1, :], xrow, Xa[2:2, :]); y = T[ya[1], yi, ya[2]]; w = T[wa[1], 0, wa[2]]
      f = GLMLoss(ProximalBase.Poisson(), DeviceArray(y), DeviceArray(X); weights = DeviceArray(w))
      g1, g2 = DeviceArray(zeros(T, 2)), DeviceArray(zeros(T, 2))
      @test f(DeviceArray(β)) ≈ va rtol = 4eps(T)
      @test value_and_gradient!(g1, f, DeviceArray(β)) ≈ va rtol = 4eps(T)
      @test host(g1) ≈ host(ga) rtol = 4eps(T)
      gradient!(g2, f, DeviceArray(β))
      @test host(g2) ≈ host(ga) rtol = 4eps(T)
    end
    # a mutated offset of the inactive observation
    f = GLMLoss(ProximalBase.Poisson(), DeviceArray(T[0, 0, 1]), DeviceArray(vcat(Xa[1:1, :], T[0 0], Xa[2:2, :]));
                weights = DeviceArray(T[tiny, 0, 2]), offset = DeviceArray(zeros(T, 3)))
    copyto!(f.offset, DeviceArray(T[0, NaN, 0]))
    g1 = DeviceArray(zeros(T, 2))
    @test value_and_gradient!(g1, f, DeviceArray(β)) ≈ va rtol = 4eps(T)
    @test host(g1) ≈ host(ga) rtol = 4eps(T)
  end
end

@testset "GLMLoss weight and offset checks on GPU arrays" begin
  allowscalar(false)
  # the checks run as a device reduction, also when Adapt rebuilds the loss (#579)
  wmsg, omsg = "the weights must be finite and nonnegative", "the offset must be finite"
  for T in test_types
    yc, Xc, w, o = T[1, 0, 1], T[1 0; 0 1; 1 1], T[0.5, 0, 2], T[0.1, -0.2, 0]
    y, X, β = DeviceArray(yc), DeviceArray(Xc), T[0.3, -0.7]
    fc = LogisticLoss(yc, Xc; weights = w, offset = o)
    f = LogisticLoss(y, X; weights = DeviceArray(w), offset = DeviceArray(o))
    @test f.weights isa DeviceArray && f.offset isa DeviceArray
    @test f(DeviceArray(β)) ≃ fc(β)
    @test Adapt.adapt(DeviceArray, fc)(DeviceArray(β)) ≃ fc(β)
    for bad in (T[1, Inf, 1], T[1, NaN, 1], T[1, -1, 1])
      @test_throws ArgumentError(wmsg) LogisticLoss(y, X; weights = DeviceArray(bad))
    end
    for bad in (T[0, Inf, 0], T[0, NaN, 0], T[-Inf, 0, 0])
      @test_throws ArgumentError(omsg) LogisticLoss(y, X; offset = DeviceArray(bad))
    end
  end
  if Float64 in test_types && Float32 in test_types
    # in the working precision: finite in Float64, it overflows Float32 data
    y32, X32 = DeviceArray(Float32[1, 0]), DeviceArray(Float32[1 0; 0 1])
    @test_throws ArgumentError(wmsg) LogisticLoss(y32, X32; weights = DeviceArray([1.0, 1e39]))
    @test_throws ArgumentError(omsg) LogisticLoss(y32, X32; offset = DeviceArray([0.0, 1e39]))
    @test LogisticLoss(y32, X32; weights = DeviceArray([1.0, 1e38]), offset = DeviceArray([0.0, -1e38])) isa GLMLoss
  end
end

@testset "infinite and zero weights on GPU arrays" begin
  allowscalar(false)
  for T in test_types, λ in (0, 1)
    w = T[1, Inf, 0, 2]
    # the remaining functions and cases are in gpu_suite.jl
    for f in (NormL1(T(λ), w),)
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
end

@testset "conjugate array steps on GPU arrays" begin
  allowscalar(false)
  for T in test_types, dims in ((3,), (3, 2))
    x = reshape(repeat(T[0.5, -2, 1.5], prod(dims) ÷ 3), dims)
    γ = reshape(repeat(T[0.5, 1, 2], prod(dims) ÷ 3), dims)
    xg, γg = DeviceArray(x), DeviceArray(γ)
    # the remaining functions and cases are in gpu_suite.jl
    for f in (ElasticNet(1, 1, fill(T(0.8), dims)),)
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
  end
end

using GPUArraysCore: AnyGPUArray

@testset "maximum_abs of a Diagonal of a device vector: dispatch" begin
  allowscalar(false)
  ext = Base.get_extension(ProximalBase, :ProximalBaseGPUArraysCoreExt)
  for T in (Float32, Float64)
    D = typeof(DeviceArray(T[]))
    pick(X) = which(ProximalBase._maximum_abs, Tuple{X})
    new = pick(Diagonal{T, D})
    generic, typed, host_diagonal, device = pick(Vector{Int}), pick(Vector{T}), pick(Diagonal{T, Vector{T}}), pick(D)
    # each of the four methods is the one intended, and none is the new method
    @test generic.module === ProximalBase && generic.sig == Tuple{typeof(ProximalBase._maximum_abs), AbstractArray}
    @test typed.module === ProximalBase && typed.sig == (Tuple{typeof(ProximalBase._maximum_abs), AbstractArray{S}} where S<:Union{Float32, Float64})
    @test host_diagonal.module === ProximalBase && host_diagonal.sig == (Tuple{typeof(ProximalBase._maximum_abs), Diagonal{S}} where S<:Union{Float32, Float64})
    @test device.module === ext && device.sig == Tuple{typeof(ProximalBase._maximum_abs), AnyGPUArray}
    @test new.module === ext
    for m in (generic, typed, host_diagonal, device)
      @test new !== m
      @test new.sig <: m.sig
      @test !(m.sig <: new.sig)
      @test !Base.isambiguous(new, m)
    end
  end
end

@testset "maximum_abs of a Diagonal of a device vector: values" begin
  allowscalar(false)
  for T in test_types
    cases = ((T[1, -3, 2], T(3)), (T[1, NaN, 2], nothing), (T[-Inf, 1], T(Inf)), (T[-0.0], zero(T)), (T[], zero(T)))
    for (v, expected) in cases
      r = ProximalBase._maximum_abs(Diagonal(DeviceArray(v)))
      @test r isa T
      @test isequal(r, ProximalBase._maximum_abs(DeviceArray(v)))
      if expected !== nothing                              # no NaN: the result does not depend on the backend's max
        @test isequal(r, expected)
        @test isequal(r, ProximalBase._maximum_abs(Diagonal(v)))
      end
    end
  end
end

@testset "LeastSquaresLoss with a Diagonal design on GPU arrays" begin
  allowscalar(false)
  for T in test_types
    y = T[1, 2]
    # Xβ overflows: the value is formed again from the largest entries of X, β and Y, which reads the design
    fh = LeastSquaresLoss(y, Diagonal(T[floatmax(T) / 2, 1]))
    fd = Adapt.adapt(DeviceArray, fh)
    @test !(fd.X.diag isa Array)
    β = T[4, 1]
    @test same(fd(DeviceArray(β)), fh(β))
    gh = zeros(T, 2)
    vh = value_and_gradient!(gh, fh, β)
    gd = DeviceArray(zeros(T, 2))
    @test same(value_and_gradient!(gd, fd, DeviceArray(β)), vh)
    @test isequal(host(gd), gh)
    # nothing overflows
    fh = LeastSquaresLoss(y, Diagonal(T[3, 1]))
    fd = Adapt.adapt(DeviceArray, fh)
    β = T[1, 1]
    @test same(fd(DeviceArray(β)), fh(β))
    gh = zeros(T, 2)
    vh = value_and_gradient!(gh, fh, β)
    gd = DeviceArray(zeros(T, 2))
    @test same(value_and_gradient!(gd, fd, DeviceArray(β)), vh)
    @test host(gd) ≃ gh
  end
end

# The ambiguity guard is in its own file so that the Quick tier runs it without these cases.
include(joinpath(@__DIR__, "gpu_ambiguities.jl"))
