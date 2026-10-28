module PrecisionTest

using Test
using ProximalBase
using LinearAlgebra

const PB = ProximalBase

function check_precision(f, x::Array{T, N}, V; prox_value_type = V) where {T, N}
  @test typeof(f(x)) === V
  y = similar(x)
  if isproximable(f)
    @test typeof(prox(f, x, T(0.5))) === Array{T, N}
    @test typeof(value_and_prox!(y, f, x, T(0.5))) === prox_value_type
    @test eltype(y) === T
    y32 = similar(x, Float32)
    @test prox!(y32, f, x, 0.5) === y32
    @test eltype(y32) === Float32
    @test all(v -> typeof(v) === Float32, y32)
  end
  if issmooth(f)
    @test typeof(gradient(f, x)) === Array{T, N}
    @test typeof(value_and_gradient!(y, f, x)) === V
    @test eltype(y) === T
    @test typeof(value_and_gradient(f, x)) === Tuple{V, Array{T, N}}
    g32 = similar(x, Float32)
    @test gradient!(g32, f, x) === g32
    @test eltype(g32) === Float32
  end
  hashessian(f) && @test typeof(hessian_vec(f, x, x)) === Array{T, N}
end

@testset "Precision" begin
  @testset "Penalties and parameters" begin
    for T in (Float32, Float64), P in (Float32, Float64), D in (Float32, Float64)
      x, w, b = T[1, 2], D[1, 2], D[0.25, -0.5]
      for f in (NormL1(P(0.5)), NormL1(P(0.5), w),
                ElasticNet(P(0.5), P(0.25)), ElasticNet(P(0.5), P(0.25), w),
                IndBox(-P(1), P(1)), IndBox(-w, w), NormL2(P(0.5)),
                IndBallL2(P(1)), SqrNormL2(P(0.5)),
                GroupNormL2(P(0.5), [1:1, 2:2]),
                GroupNormL2(P(0.5), [1:1, 2:2], w))
        check_precision(f, x, T)
      end
      check_precision(SqrNormL2(P(0.5), b), x, promote_type(T, D))

      f = NormL1(P(0.5), w)
      @test typeof(f.λ0) === P
      @test f.w === w
      g = GroupNormL2(P(0.5), [1:1, 2:2], w)
      @test typeof(g.λ0) === promote_type(P, D)
      @test typeof(g.w) === Vector{promote_type(P, D)}
    end
  end

  @testset "Wrappers" begin
    for T in (Float32, Float64), P in (Float32, Float64), D in (Float32, Float64)
      x, b = T[1, 2], D[0.25, -0.5]
      for f in (NormL1(P(0.5)), SqrNormL2(P(0.5)))
        check_precision(Scaled(f, D(2)), x, promote_type(T, D))
        check_precision(Tilt(f, b, P(0)), x, promote_type(T, P, D))
        t = Translate(f, b)
        check_precision(t, x, promote_type(T, D); prox_value_type = T)
        @test typeof(t.scratch.buf) === Vector{promote_type(T, D)}
        check_precision(Regularize(f, D(0.25)), x, T)
        check_precision(Regularize(f, P(0.25), b), x, promote_type(T, D))
        c = Conjugate(f)
        check_precision(c, x, T)
        @test typeof(c.scratch.buf) === Vector{Float32}
      end
    end
  end

  @testset "Data-carrying losses" begin
    for T in (Float32, Float64), P in (Float32, Float64), D in (Float32, Float64)
      x = T[1, 2]
      X, y = D[1 0.5; -0.5 1; 1 1], D[0, 1, 0]
      f = LeastSquaresLoss(y, X)
      check_precision(f, x, D)
      @test typeof(f.tmp) === Vector{D}
      @test typeof(f.V) === Matrix{D}
      @test typeof(f.s2) === Vector{D}
      for g in (LogisticLoss(y, X),
                LogisticLoss(y, X; weights = P[1, 2, 1], offset = P[0.1, 0, -0.1]),
                GLMLoss(PB.Huber(P(0.5)), y, X; weights = P[1, 2, 1], offset = P[0.1, 0, -0.1]))
        check_precision(g, x, D)
        @test typeof(g.η) === Vector{D}
        @test typeof(g.d) === Vector{D}
      end
      mixed = LeastSquaresLoss(P.(y), X)
      R = promote_type(P, D)
      check_precision(mixed, x, R)
      @test typeof(mixed.tmp) === Vector{R}
    end
  end

  @testset "Spectral workspaces" begin
    for T in (Float32, Float64), P in (Float32, Float64)
      x = T[1 0.25; -0.5 2]
      f = NuclearNorm(P(0.5))
      @test typeof(f(x)) === T
      @test typeof(f.ws.A) === Matrix{T}
      check_precision(f, x, T)
      @test typeof(f.ws.A) === Matrix{Float32}
    end
  end

  @testset "Documented examples and allocating forms" begin
    x = Float32[1, 2]
    @test typeof(NormL1(1.0)(x)) === Float32
    @test typeof(SqrNormL2(1.0)(x)) === Float32
    @test typeof(Scaled(NormL1(1f0), 2.0)(x)) === Float64
    @test typeof(SqrNormL2(1f0, Float64[0, 1])(x)) === Float64
    f = Translate(NormL1(1f0), Float64[0.25, -0.5])
    @test typeof(f(x)) === Float64
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
