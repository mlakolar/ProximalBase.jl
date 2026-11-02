module LLATest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# the slope of t ↦ fᵢ(t) at t > 0, by central differences, and from the right
fd_slope(f, t, i; h = 1e-6) = (value_entry(f, t + h, i) - value_entry(f, t - h, i)) / 2h
fd_right(f, t, i; h = 1e-7) = (value_entry(f, t + h, i) - value_entry(f, t, i)) / h

const W = [0.5, 1.0, 2.0, 0.0, 1.3]

@testset "the slopes of the separable penalties" begin
  # (f, the kinks on t > 0, the slope at zero of entry i)
  cases = [
    (CappedL1(0.7, 1.5), (1.5,), i -> 0.7),
    (CappedL1(0.7, 1.5, W), (1.5,), i -> 0.7 * W[i]),
    (LogPenalty(0.8, 0.3), (), i -> 0.8 / 0.3),
    (LogPenalty(0.8, 0.3, W), (), i -> 0.8 * W[i] / 0.3),
    (MCP(0.5, 2.5), (1.25,), i -> 0.5),
    (MCP(0.5, 0.6), (0.3,), i -> 0.5),            # a < 1: the slope is still that of the definition
    (SCAD(0.5, 3.7), (0.5, 1.85), i -> 0.5),
    (NormL1(0.4, W), (), i -> 0.4 * W[i]),
  ]
  ts = (0.01, 0.2, 0.45, 0.9, 1.3, 1.6, 2.2, 5.0)
  for (f, kinks, slope0) in cases, i in eachindex(W)
    # away from the kinks, the derivative of the value, and the same for -t
    for t in ts
      any(k -> abs(t - k) < 1e-3, kinks) && continue
      @test lla_weight(f, t, i) ≈ fd_slope(f, t, i) atol = 1e-6
      @test lla_weight(f, -t, i) == lla_weight(f, t, i)
    end
    # at zero and at the kinks, the slope from the right
    @test lla_weight(f, 0.0, i) ≈ slope0(i)
    @test lla_weight(f, -0.0, i) == lla_weight(f, 0.0, i)
    @test lla_weight(f, 0.0, i) ≈ fd_right(f, 0.0, i) atol = 1e-6
    for k in kinks
      @test lla_weight(f, k, i) ≈ fd_right(f, k, i) atol = 1e-6
    end
    # the tangent from the right majorizes the penalty: p(|s|) ≤ p(|t|) + w(|s| - |t|)
    for t in (0.0, ts..., kinks...), s in -3:0.25:3
      @test value_entry(f, s, i) <= value_entry(f, t, i) + lla_weight(f, t, i) * (abs(s) - t) + 1e-12
    end
  end
end

@testset "the conventions at the kinks" begin
  # capped ℓ1, Zhang (2010): λwᵢ below the cap, 0 from the cap on
  f = CappedL1(0.7, 1.5, W)
  @test lla_weight(f, prevfloat(1.5), 2) == 0.7
  @test lla_weight(f, 1.5, 2) == 0
  @test lla_weight(f, -1.5, 2) == 0
  @test lla_weight(f, 9.0, 2) == 0
  # MCP: λ at zero, decreasing linearly to 0 at aλ
  m = MCP(0.5, 2.5)
  @test lla_weight(m, 0.0, 1) == 0.5
  @test lla_weight(m, 0.5, 1) ≈ 0.3
  @test lla_weight(m, 1.25, 1) == 0
  @test lla_weight(m, 3.0, 1) == 0
  # SCAD: λ up to λ, then (aλ - t)/(a - 1), 0 from aλ on
  s = SCAD(0.5, 3.7)
  @test lla_weight(s, 0.0, 1) == 0.5
  @test lla_weight(s, 0.5, 1) == 0.5
  @test lla_weight(s, 1.0, 1) ≈ (1.85 - 1.0) / 2.7
  @test lla_weight(s, 1.85, 1) == 0
  @test lla_weight(s, 4.0, 1) == 0
  # log: the reweighted ℓ1 weights of Candès, Wakin and Boyd, λ/(ε + |t|)
  @test lla_weight(LogPenalty(1.0, 0.1), 0.4, 1) ≈ 1 / 0.5
  # a zero level or a zero entry weight gives zero weights
  @test lla_weight(MCP(0.0, 3.0), 0.0, 1) == 0
  @test lla_weight(SCAD(0.0, 3.7), 0.2, 1) == 0
  @test lla_weight(CappedL1(0.7, 1.5, W), 0.2, 4) == 0
  # the truncated ℓ1 penalty λ min(|t|/τ, 1) of Shen, Pan and Zhu is CappedL1(λ/τ, τ)
  λ, τ = 0.6, 0.25
  tlp = CappedL1(λ / τ, τ)
  @test value_entry(tlp, 0.1, 1) ≈ λ * 0.1 / τ
  @test value_entry(tlp, 0.4, 1) ≈ λ
  @test lla_weight(tlp, 0.1, 1) ≈ λ / τ
end

@testset "lla_weights! on arrays and iterates" begin
  for f in (CappedL1(0.3, 0.8), CappedL1(0.3, 0.8, rand(40)), LogPenalty(0.3, 0.5), LogPenalty(0.3, 0.5, rand(40)),
            MCP(0.4, 2.0), SCAD(0.4, 3.7), NormL1(0.2, rand(40)))
    xs = SparseIterate(sprandn(40, 0.3))
    x = Vector(xs)
    ref = [lla_weight(f, x[i], i) for i in 1:40]
    w = zeros(40)
    @test lla_weights!(w, f, x) === w
    @test w == ref
    @test lla_weights(f, x) == ref
    @test lla_weights(f, xs) == ref
    # the weights are those of the next stage: its proximal operator is the
    # weighted soft thresholding
    @test prox(NormL1(1, w), x, 0.7) ≈ shrink.(x, 0.7 .* w)
    # a matrix and a SymmetricSparseIterate: one weight per entry, at its linear index
    f isa Union{MCP, SCAD} || continue
    p = 6
    D = randn(p, p); D = D + D'
    S = SymmetricSparseIterate(D)
    @test vec(lla_weights(f, S)) == [lla_weight(f, D[i], i) for i in 1:p^2]
    @test lla_weights(f, D) == reshape([lla_weight(f, D[i], i) for i in 1:p^2], p, p)
  end
  # on an AtomIterate, entry by entry
  groups = [1:4, 5:8]
  xa = AtomIterate([1.0, 0.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0], groups)
  @test lla_weights(MCP(0.5, 3.0), xa) == lla_weights(MCP(0.5, 3.0), Vector(xa))
end

@testset "a stage of LLA decreases the objective" begin
  # with h(x) = ‖x - y‖²/2, a stage is weighted soft thresholding of y, exact,
  # so each stage decreases h + f (majorization-minimization)
  y = 2 .* randn(50)
  for f in (CappedL1(0.6, 1.0), LogPenalty(0.4, 0.5), MCP(0.8, 2.0), SCAD(0.8, 3.7))
    obj(x) = sum(abs2, x .- y) / 2 + f(x)
    x = zeros(50)
    w = similar(x)
    prev = obj(x)
    for stage in 1:6
      lla_weights!(w, f, x)
      stage == 1 && @test all(w .== lla_weight(f, 0.0, 1))     # from zero, the first stage is the lasso
      x = prox(NormL1(1, w), y)
      @test obj(x) <= prev + 1e-12
      prev = obj(x)
    end
  end
end

@testset "group weights" begin
  groups = [1:3, 4:5, [6, 8], 9:12]                 # x[7] is in no group
  x = 2 .* randn(12)
  x[4:5] .= 0                                       # a zero group
  for (F, P, a) in ((GroupMCP, MCP, 2.5), (GroupSCAD, SCAD, 3.7))
    λ = 0.5
    f = F(λ, groups; a)
    w = lla_weights(f, x)
    @test length(w) == 4
    for k in eachindex(groups)
      pk = P(λ * f.w[k], a)
      nk = norm(x[groups[k]])
      @test w[k] ≈ lla_weight(pk, nk, 1)
      @test lla_weight(f, nk, k) ≈ w[k]
      # the slope of the value along the ray of the group
      if nk > 0
        u = x[groups[k]] / nk
        z(t) = (y = copy(x); y[groups[k]] .= t .* u; f(y))
        @test w[k] ≈ (z(nk + 1e-6) - z(nk - 1e-6)) / 2e-6 atol = 1e-6
      end
    end
    @test w[2] ≈ λ * f.w[2]                         # a zero group: the group lasso's slope
    # the next stage is the weighted group lasso, majorizing f up to a constant
    gl = GroupNormL2(1, groups, w)
    for _ in 1:20
      v = 3 .* randn(12)
      @test f(v) <= f(x) + gl(v) - gl(x) + 1e-10
    end
    # an AtomIterate with the groups of f
    fa = F(λ, [groups; [[7]]], [f.w; 0.0]; a)
    xa = AtomIterate(x, fa)
    @test lla_weights(fa, xa) ≈ [w; 0.0]
    # groups of one entry with unit weights give the scalar weights
    fs = F(λ, [[i] for i in 1:12], ones(12); a)
    @test lla_weights(fs, x) ≈ lla_weights(P(λ, a), x)
    @test lla_weights(F(0.4f0, groups), randn(Float32, 12)) isa Vector{Float32}
    @test_throws DimensionMismatch lla_weights!(zeros(3), f, x)
    @test_throws BoundsError lla_weights(f, zeros(7))
  end
  gl = GroupNormL2(0.3, groups)
  @test lla_weights(gl, x) ≈ 0.3 .* gl.w
end

@testset "element types, inference and errors" begin
  x32 = randn(Float32, 10)
  for f in (CappedL1(0.3, 0.8), LogPenalty(0.3, 0.5, rand(10)), MCP(0.4, 2.0), SCAD(0.4, 3.7))
    @test lla_weights(f, x32) isa Vector{Float32}
    @test lla_weights(f, x32) ≈ lla_weights(f, Float64.(x32))
    @inferred lla_weight(f, 0.7, 2)
    @inferred lla_weight(f, 0.7f0, 2)
    w = zeros(10)
    @inferred lla_weights!(w, f, randn(10))
  end
  groups = [1:5, 6:10]
  for f in (GroupMCP(0.4, groups), GroupSCAD(0.4, groups), GroupNormL2(0.4, groups))
    @inferred lla_weights!(zeros(2), f, randn(10))
    @inferred lla_weight(f, 0.7, 2)
  end
  # NormLq has an infinite slope at zero; other functions have no weights
  for f in (NormLq(0.3, 0.5), ElasticNet(0.3, 0.1), IndBox(-1, 1), NormL2(1.0), NuclearNorm(1.0))
    @test_throws ArgumentError lla_weights(f, randn(4, 4))
    @test_throws ArgumentError lla_weight(f, 0.5, 1)
  end
  @test_throws "infinite slope" lla_weights(NormLq(0.3, 0.5), randn(4))
  @test_throws DimensionMismatch lla_weights!(zeros(3), MCP(0.4), randn(4))
  @test_throws DimensionMismatch lla_weights(CappedL1(0.3, 0.8, rand(3)), randn(4))
end

@testset "zero allocations" begin
  a_lla!(w, f, x) = lla_weights!(w, f, x)
  x = randn(20)
  xs = SparseIterate(sprandn(20, 0.3))
  D = randn(6, 6); D = D + D'
  S = SymmetricSparseIterate(D)
  for f in (CappedL1(0.3, 1.0), CappedL1(0.3, 1.0, rand(20)), LogPenalty(0.3, 0.5), MCP(0.4, 2.0), SCAD(0.4, 3.7))
    for v in (x, Float32.(x), xs)
      @test allocs(a_lla!, similar(Array(v)), f, v) == 0
    end
  end
  for f in (CappedL1(0.3, 1.0, Symmetric(rand(6, 6), :L)), LogPenalty(0.3, 0.5), MCP(0.4, 2.0))
    @test allocs(a_lla!, zeros(6, 6), f, S) == 0
  end
  groups = [1:5, 6:10, 11:15, 16:20]
  xa = AtomIterate(randn(20), groups)
  for f in (GroupMCP(0.4, groups), GroupSCAD(0.4, groups), GroupNormL2(0.4, groups))
    @test allocs(a_lla!, zeros(4), f, x) == 0
    @test allocs(a_lla!, zeros(4), f, xa) == 0
  end
end

end

# scenario A
