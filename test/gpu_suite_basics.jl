# The representative GPU cases, included after gpu_prelude.jl: one or two functions for each
# device path (separable, nonseparable, group layouts on the host and the device, adapted
# parameters, losses, array steps, overflow, infinite weights), each testset calling
# allowscalar(false). gpu_suite.jl keeps the rest of each split loop and the other cases.
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
