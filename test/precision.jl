module PrecisionTest

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase

function check_precision(f, x::Array{T, N}, V) where {T, N}
  @nospecialize f
  @test typeof(f(x)) === V
  y = similar(x)
  if isproximable(f)
    @test typeof(prox(f, x, T(0.5))) === Array{T, N}
    @test prox!(y, f, x, T(0.5)) === y
    @test typeof(value_and_prox!(y, f, x, T(0.5))) === V
  end
  if issmooth(f)
    @test typeof(gradient(f, x)) === Array{T, N}
    @test gradient!(y, f, x) === y
    @test typeof(value_and_gradient!(y, f, x)) === V
    @test typeof(value_and_gradient(f, x)) === Tuple{V, Array{T, N}}
  end
  hashessian(f) && @test typeof(hessian_vec(f, x, x)) === Array{T, N}
end

@testset "Precision" begin
  @testset "All Float32 audit cases" begin
    x, w, b = Float32[1, 2], Float32[1, 2], Float32[0.25, -0.5]
    X, y = Float32[1 0.5; -0.5 1; 1 1], Float32[0, 1, 0]
    weights, offset = Float32[1, 2, 1], Float32[0.1, 0, -0.1]
    cases = [
      ("NormL1", NormL1(0.5f0), false),
      ("NormL1_weighted", NormL1(0.5f0, w), false),
      ("ElasticNet", ElasticNet(0.5f0, 0.25f0), false),
      ("ElasticNet_weighted", ElasticNet(0.5f0, 0.25f0, w), false),
      ("IndBox_scalar", IndBox(-1f0, 1f0), false),
      ("IndBox_array", IndBox(-w, w), false),
      ("NormL2", NormL2(0.5f0), false),
      ("IndBallL2", IndBallL2(1f0), false),
      ("SqrNormL2", SqrNormL2(0.5f0), false),
      ("SqrNormL2_centered", SqrNormL2(0.5f0, b), false),
      ("GroupNormL2_default", GroupNormL2(0.5f0, [1:1, 2:2]), false),
      ("GroupNormL2_weighted", GroupNormL2(0.5f0, [1:1, 2:2], w), false),
      ("LeastSquaresLoss", LeastSquaresLoss(y, X), false),
      ("LogisticLoss", LogisticLoss(y, X), false),
      ("LogisticLoss_weighted", LogisticLoss(y, X; weights, offset), false),
      ("GLMLoss_Huber", GLMLoss(PB.Huber(0.5f0), y, X; weights, offset), false),
      ("LeastSquaresLoss_mixed", LeastSquaresLoss(copy(y), X), false),
    ]
    for (name, f) in (("NormL1", NormL1(0.5f0)), ("SqrNormL2", SqrNormL2(0.5f0)))
      push!(cases, ("Scaled_$name", Scaled(f, 2f0), false))
      push!(cases, ("Tilt_$name", Tilt(f, b, 0f0), false))
      push!(cases, ("Translate_$name", Translate(f, b), false))
      push!(cases, ("Regularize_$name", Regularize(f, 0.25f0), false))
      push!(cases, ("Regularize_$(name)_centered", Regularize(f, 0.25f0, b), false))
      push!(cases, ("Conjugate_$name", Conjugate(f), false))
    end
    push!(cases, ("NuclearNorm", NuclearNorm(0.5f0), true))
    @test length(cases) == 30
    for (name, f, matrix) in cases
      @testset "$name" begin
        check_precision(f, matrix ? Float32[1 0.25; -0.5 2] : x, Float32)
      end
    end
  end

  @testset "Value types agree across oracles" begin
    for T in (Float32, Float64)
      S = T === Float32 ? Float64 : Float32
      x, b = T[1, 2], Float64[0.25, -0.5]
      X = T[1 0.25; 0.25 2]
      stacked = cat(X, 2X; dims = 3)
      translated = Translate(NormL1(one(T)), b)
      smooth = Translate(SqrNormL2(one(T)), b)
      ls = Translate(LeastSquaresLoss(S[0, 1, 0], S[1 0.5; -0.5 1; 1 1]), b)
      scaled = Scaled(NormL1(one(T)), 2.0)
      cases = [
        ("Translate_NormL1", translated, x, true),
        ("Translate_SqrNormL2", smooth, x, false),
        ("Translate_IndBallL2", Translate(IndBallL2(one(T)), b), x, false),
        ("Translate_NuclearNorm", Translate(NuclearNorm(one(T)), Float64[0.25 0.5; -0.5 0.25]), X, false),
        ("Translate_LeastSquaresLoss", ls, x, false),
        ("Translate_Scaled", Translate(scaled, b), x, true),
        ("Translate_integer", Translate(NormL1(one(T)), [1, -1]), x, true),
        ("Translate_nested", Translate(translated, T[0.5, -0.25]), x, true),
        ("Scaled_Translate_NormL1", Scaled(translated, 2.0), x, true),
        ("Scaled_Translate_SqrNormL2", Scaled(smooth, 2.0), x, false),
        ("Tilt_Translate_NormL1", Tilt(translated, T[0.5, -0.25]), x, true),
        ("Tilt_Translate_SqrNormL2", Tilt(smooth, T[0.5, -0.25]), x, false),
        ("Regularize_Translate_NormL1", Regularize(translated, T(0.25)), x, false),
        ("Regularize_Translate_SqrNormL2", Regularize(smooth, T(0.25)), x, false),
        ("Precompose_Translate_NormL1", Precompose(translated, T(-2)), x, false),
        ("Precompose_Translate_SqrNormL2", Precompose(smooth, T(-2)), x, false),
        ("Conjugate_Translate_NormL1", Conjugate(translated), x, true),
        ("Conjugate_Translate_SqrNormL2", Conjugate(smooth), x, false),
        ("SeparableSum_Translate_NormL1", SeparableSum(translated, NormL1(one(T))), (x, x), false),
        ("SeparableSum_Translate_SqrNormL2", SeparableSum(smooth, SqrNormL2(one(T))), (x, x), false),
        ("SliceSum_Translate_NormL1", SliceSum(translated, NormL1(one(T))), X, false),
        ("SliceSum_Translate_SqrNormL2", SliceSum(smooth, SqrNormL2(one(T))), X, false),
        ("SliceSum_Scaled", SliceSum(scaled, scaled), X, false),
        ("SingularValue_Scaled", SingularValue(scaled), X, false),
        ("Spectral_Scaled", Spectral(scaled), X, false),
      ]
      for (name, inner) in (("Translate", translated), ("Scaled", scaled))
        for (rule, diagonal) in (("default", inner), ("free", nothing), ("own", Scaled(NormL1(one(T)), 3.0))),
            symmetric in (false, true)
          push!(cases, ("FiberSum_$(name)_$(rule)_$symmetric", FiberSum(inner; diagonal, symmetric), stacked, false))
        end
      end
      fused = NormL1Fused(1.0, 0.25)
      for (rule, diagonal) in (("default", fused), ("free", nothing), ("own", NormL1Fused(0.0, 0.5))),
          symmetric in (false, true)
        push!(cases, ("FiberSum_NormL1Fused_$(rule)_$symmetric", FiberSum(fused; diagonal, symmetric), stacked, false))
      end
      @test length(cases) == 43
      for (name, f, x, metric) in cases
        @testset "$T/$name" begin
          y = x isa Tuple ? map(similar, x) : similar(x)
          if isproximable(f)
            @test typeof(value_and_prox!(y, f, x, T(0.5))) === typeof(f(x))
            if metric
              @test isseparable(f)
              γ = fill(T(0.5), size(x))
              @test typeof(value_and_prox!(y, f, x, γ)) === typeof(f(x))
            end
            if x isa Tuple
              alias = map(copy, x)
              V = typeof(f(alias))
              @test typeof(value_and_prox!(alias, f, alias, T(0.5))) === V
            end
          end
          if issmooth(f)
            @test typeof(value_and_gradient!(y, f, x)) === typeof(f(x))
          end
          @test typeof(f(x)) === T
        end
      end
    end
  end

  @testset "Destination precision and caches" begin
    x, y32 = [1.0, 2.0], zeros(Float32, 2)
    @test prox!(y32, NormL1(1.0), x, 0.5) === y32
    c = Conjugate(NormL1(1.0))
    @test prox!(y32, c, x, 0.5) === y32
    @test typeof(c.scratch.buf) === Vector{Float32}

    f = NuclearNorm(0.5)
    A, B = [1.0 0.25; -0.5 2.0], zeros(Float32, 2, 2)
    @test typeof(f(A)) === Float64
    @test typeof(f.ws.A) === Matrix{Float64}
    @test prox!(B, f, A, 0.5) === B
    @test typeof(f.ws.A) === Matrix{Float32}

    X, y = [1.0 0.5; -0.5 1.0; 1.0 1.0], [0.0, 1.0, 0.0]
    ls = LeastSquaresLoss(y, X)
    @test prox!(y32, ls, x, 0.5) === y32
    @test gradient!(y32, ls, x) === y32
    @test typeof(value_and_gradient!(y32, ls, x)) === Float64
    @test typeof(y32) === Vector{Float32}

    t = Translate(SqrNormL2(1f0), [0.25, -0.5])
    @test typeof(t(Float32[1, 2])) === Float32
    @test typeof(t.scratch.buf) === Vector{Float32}
    @test gradient!(y32, t, Float32[1, 2]) === y32
    @test typeof(t.scratch.buf) === Vector{Float32}
  end

  @testset "Translate precision and second-order oracles" begin
    x, v = Float32[1, 2], Float32[0.5, -1]
    t = Translate(SqrNormL2(1f0), Float64[0.25, -0.5])
    Hv = hessian_vec(t, x, v)
    @test eltype(Hv) === Float32
    @test Hv == v
    q = quadratic_model(t, x)
    @test q(x) == t(x)
    @test quadratic_model!(q, t, x) === q
    @test gradient(q, x) == gradient(t, x)
    t(x)
    @test PB.Testing.allocations(t, x) == 0
    @test typeof(t.scratch.buf) === Vector{Float32}

    x64, b = [1.25, -2.5], [0.25, -0.5]
    @test Translate(NormL1(1.0), b)(x64) === NormL1(1.0)(x64 .- b)
  end

  @testset "Parameters and data precision" begin
    w = [1.0, 2.0]
    f = NormL1(1f0, w)
    @test typeof(f.λ0) === Float32
    @test f.w === w
    g = GroupNormL2(1f0, [1:1, 2:2], w)
    @test typeof(g.λ0) === Float64
    @test typeof(g.w) === Vector{Float64}

    X, y = Float32[1 0.5; -0.5 1; 1 1], Float32[0, 1, 0]
    ls = LeastSquaresLoss(y, X)
    check_precision(ls, [1.0, 2.0], Float64)
    @test typeof(ls.tmp) === Vector{Float32}
    @test typeof(ls.V) === Matrix{Float32}
    @test typeof(ls.s2) === Vector{Float32}
    for loss in (LogisticLoss(y, X), GLMLoss(PB.Huber(0.5f0), y, X))
      check_precision(loss, [1.0, 2.0], Float64)
      @test typeof(loss.η) === Vector{Float32}
      @test typeof(loss.d) === Vector{Float32}
    end
    mixed = LeastSquaresLoss(Float64.(y), X)
    @test typeof(mixed.tmp) === Vector{Float64}
    for loss in (LogisticLoss(Float64.(y), X), GLMLoss(PB.Huber(0.5f0), Float64.(y), X))
      @test typeof(loss.η) === Vector{Float64}
      @test typeof(loss.d) === Vector{Float64}
    end
  end

  @testset "Documented examples and allocating forms" begin
    x = Float32[1, 2]
    @test typeof(NormL1(1.0)(x)) === Float32
    @test typeof(SqrNormL2(1.0)(x)) === Float32
    @test typeof(Scaled(NormL1(1f0), 2.0)(x)) === Float32
    @test typeof(SqrNormL2(1f0, Float64[0, 1])(x)) === Float32
    f = Translate(NormL1(1f0), Float64[0.25, -0.5])
    @test typeof(f(x)) === Float32
    @test typeof(value_and_prox!(similar(x), f, x, 0.5)) === Float32
    for T in (Int, Float32, Float64)
      z = T[1, 2]
      R = float(T)
      q = SqrNormL2(1.0)
      @test typeof(prox(q, z, 0.5)) === Vector{R}
      @test typeof(gradient(q, z)) === Vector{R}
      @test typeof(last(value_and_gradient(q, z))) === Vector{R}
      @test typeof(hessian_vec(q, z, z)) === Vector{R}
    end
    for γ in (1, 1f0, 1.0, 1 // 2)
      @test typeof(prox(NormL1(1f0), x, γ)) === Vector{Float32}
    end
  end
end

end
