module NonconvexTest

using Test
using ProximalBase
using LinearAlgebra
using SparseArrays
using Random

using ..TestUtils

Random.seed!(1)

# the objective of a scalar proximal problem
hobj(pen, z, v, γ) = pen(z) + (z - v)^2 / (2γ)

# the proximal operator of a separable function is never worse than the
# brute-force minimizer of each scalar problem
function check_scalar_prox(f, pen; vs = (-4., -2.1, -1., -0.6, -0.1, 0., 0.05, 0.3, 0.8, 1.2, 2.5, 6.),
                           γs = (0.1, 0.5, 1., 2., 5.))
  for v in vs, γ in γs
    _, hb = brute_force_prox(pen, v, γ)
    z = prox_entry(f, v, 1, γ)
    @test hobj(pen, z, v, γ) <= hb + 1e-9 * max(1, abs(hb))
  end
end

@testset "CappedL1" begin
  λ, a = 0.7, 1.5
  f = CappedL1(λ, a)
  pen(t) = λ * min(abs(t), a)
  x = 3 .* randn(30)
  @test f(x) ≈ sum(pen, x)
  check_scalar_prox(f, pen)
  # soft thresholding below the cap, the identity well above it
  @test prox_entry(f, 1.0, 1, 1.) ≈ 0.3
  @test prox_entry(f, -0.5, 1, 1.) == 0
  @test prox_entry(f, 4.0, 1, 1.) == 4.0
  # weights scale the level entry by entry; a zero weight leaves the entry alone
  w = rand(30); w[3] = 0
  fw = CappedL1(λ, a, w)
  @test fw(x) ≈ sum(i -> λ * w[i] * min(abs(x[i]), a), 1:30)
  for i in (1, 5, 9), γ in (0.4, 2.)
    _, hb = brute_force_prox(t -> λ * w[i] * min(abs(t), a), x[i], γ)
    @test hobj(t -> λ * w[i] * min(abs(t), a), prox_entry(fw, x[i], i, γ), x[i], γ) <= hb + 1e-9
  end
  @test prox_entry(fw, x[3], 3, 1.) == x[3]
  check_prox(f, x, 0.8)
  check_prox(fw, x, 0.8)
  check_traits(f; proximable = true, separable = true)
  @test iszeropreserving(f)
  @test kkt_violation(fw, x, x, 5) ≈ abs(x[5]) - λ * w[5]
  @test restrict(fw, [2, 5]).w == w[[2, 5]]
  @test_throws ArgumentError CappedL1(-1., 1.)
  @test_throws ArgumentError CappedL1(1., 0.)
  @test_throws ArgumentError CappedL1(1., 1., [-1., 1.])
end

@testset "LogPenalty" begin
  λ, ε = 0.8, 0.3
  f = LogPenalty(λ, ε)
  pen(t) = λ * log(1 + abs(t) / ε)
  x = 3 .* randn(30)
  @test f(x) ≈ sum(pen, x)
  # steps on both sides of γλ = ε², where the scalar problem stops being convex
  check_scalar_prox(f, pen; γs = (0.01, 0.05, 0.11, 0.2, 1., 4.))
  for ε2 in (1e-3, 0.05, 2.), λ2 in (0.01, 1., 10.)
    g = LogPenalty(λ2, ε2)
    check_scalar_prox(g, t -> λ2 * log(1 + abs(t) / ε2))
  end
  w = rand(30) .+ 0.1
  fw = LogPenalty(λ, ε, w)
  @test fw(x) ≈ sum(i -> λ * w[i] * log1p(abs(x[i]) / ε), 1:30)
  check_prox(f, x, 0.5)
  check_prox(fw, x, 0.5)
  check_traits(f; proximable = true, separable = true)
  @test kkt_violation(fw, x, x, 4) ≈ abs(x[4]) - λ * w[4] / ε
  @test restrict(fw, [1, 3]).w == w[[1, 3]]
  @test_throws ArgumentError LogPenalty(1., 0.)
  @test_throws ArgumentError LogPenalty(-1., 1.)
end

# the root of z + μq z^(q-1) = t by Newton's method in BigFloat, the reference
# for the closed forms
function lq_root_reference(t, μ, q)
  t, μ, q = big(t), big(μ), big(q)
  z = t
  for _ in 1:200
    Δ = (z + μ * q * z^(q - 1) - t) / (1 + μ * q * (q - 1) * z^(q - 2))
    z -= Δ
    abs(Δ) < big(10)^-60 && break
  end
  Float64(z)
end

@testset "NormLq" begin
  for q in (0.3, 0.5, 2/3, 0.9)
    λ = 0.6
    f = NormLq(λ, q)
    pen(t) = λ * abs(t)^q
    x = 3 .* randn(30)
    @test f(x) ≈ sum(pen, x)
    check_scalar_prox(f, pen)
    check_prox(f, x, 0.7)
    # the threshold: zero up to it, the root beyond, which is at least β
    for γ in (0.2, 1., 3.)
      μ = γ * λ
      β = (2μ * (1 - q))^(1 / (2 - q))
      τ = (2 - q) / (2 - 2q) * β
      @test prox_entry(f, τ * (1 - 1e-9), 1, γ) == 0
      @test prox_entry(f, -τ * (1 + 1e-9), 1, γ) <= -β * (1 - 1e-6)
      # zero and β tie at the threshold
      @test hobj(pen, β, τ, γ) ≈ hobj(pen, 0., τ, γ) atol = 1e-12
    end
  end
  # the closed forms of q = 1/2 and 2/3 against the root in BigFloat, from just
  # above the threshold to far beyond it
  for q in (1/2, 2/3), μ in (0.01, 0.4, 3.), s in (1 + 1e-12, 1 + 1e-6, 1.01, 1.5, 3., 30.)
    τ = (2 - q) / (2 - 2q) * (2μ * (1 - q))^(1 / (2 - q))
    t = τ * s
    @test ProximalBase._lq_root(t, μ, q) ≈ lq_root_reference(t, μ, q) rtol = 1e-12
  end
  # and Newton's method for any other q
  for q in (0.1, 0.45, 0.99), μ in (0.05, 2.)
    τ = (2 - q) / (2 - 2q) * (2μ * (1 - q))^(1 / (2 - q))
    @test ProximalBase._lq_root(1.3τ, μ, q) ≈ lq_root_reference(1.3τ, μ, q) rtol = 1e-12
  end
  # weights, Float32, and zero as a local minimizer everywhere
  w = rand(10) .+ 0.5
  fw = NormLq(0.4, 0.5, w)
  xw = randn(10)
  @test fw(xw) ≈ sum(i -> 0.4 * w[i] * sqrt(abs(xw[i])), 1:10)
  check_prox(fw, xw, 0.9)
  @test kkt_violation(NormLq(0.4, 0.5), randn(5), 2) == -Inf
  @test prox(NormLq(0.4f0, 0.5f0), randn(Float32, 6), 0.5f0) isa Vector{Float32}
  @test prox(NormLq(0.4, 0.5), Float32[1.5, -0.1], 1.) ≈ prox(NormLq(0.4, 0.5), [1.5, -0.1], 1.) rtol = 1e-5
  @test NormLq(1, 1//2) isa NormLq{Float64}
  @test_throws ArgumentError NormLq(1., 1.)
  @test_throws ArgumentError NormLq(1., 0.)
end

@testset "separable functions on the iterates" begin
  for f in (CappedL1(0.3, 1.), LogPenalty(0.3, 0.5), NormLq(0.3, 0.5), NormLq(0.3, 0.7, rand(50)))
    xs = SparseIterate(sprandn(50, 0.2))
    x = Vector(xs)
    @test f(xs) ≈ f(x)
    @test Vector(prox(f, xs, 0.6)) ≈ prox(f, x, 0.6)
    zs = copy(xs)
    prox!(zs, f, zs, 0.6)
    @test Vector(zs) ≈ prox(f, x, 0.6)
    k = xs.nzval2ind[1]
    xc = copy(xs)
    @test prox_coordinate!(xc, f, k, 0.6) ≈ prox_entry(f, x[k], k, 0.6)
    @inferred prox_entry(f, 0.7, 2, 0.6)
    @inferred f(x)
  end
  p = 6
  D = randn(p, p); D = D + D'
  for f in (CappedL1(0.2, 1.), LogPenalty(0.2, 0.4), NormLq(0.2, 0.5))
    s = SymmetricSparseIterate(D)
    @test f(s) ≈ f(D)
    @test Matrix(prox(f, s, 0.5)) ≈ prox(f, D, 0.5)
  end
end

@testset "half thresholding across floating-point scales" begin
  # Independent Newton roots in higher precision, including scales at which
  # a reciprocal power would overflow although the proximal result is finite.
  setprecision(BigFloat, 256) do
    for T in (Float32, Float64)
      levels = T == Float32 ? T[1e-35,1e-10,1,1e10,1e35] : T[1e-300,1e-100,1,1e100,1e300]
      append!(levels, (nextfloat(zero(T)), floatmin(T), floatmax(T)/16))
      for μ in levels
        bμ = BigFloat(μ)
        threshold = BigFloat(3)/2 * bμ^(BigFloat(2)/3)
        f = NormLq(μ, T(0.5))
        for scale in (0.9,1.0001,2,1e10), sgn in (-1,1)
          t = T(threshold * BigFloat(scale))
          isfinite(t) || continue
          actual = prox_entry(f, sgn*t, 1, one(T))
          @test isfinite(actual)
          if BigFloat(t) <= threshold
            @test actual == 0
          else
            z = BigFloat(t)
            for _ in 1:100
              Δ = (z + bμ/(2sqrt(z)) - BigFloat(t)) /
                  (1 - bμ/(4z*sqrt(z)))
              z -= Δ
              abs(Δ) <= 8eps(one(z))*abs(z) && break
            end
            @test actual ≈ sgn*T(z) rtol = 32eps(T)
          end
        end
      end
    end
  end
end

# the minimizer of p(‖z‖) + ‖z - v‖²/(2γ) along the ray of v, by brute force
function radial_reference(p, v, γ)
  nv = norm(v)
  r, _ = brute_force_prox(t -> p(abs(t)), nv, γ; lo = 0.0, hi = 2nv + 1)
  r / nv .* v
end

@testset "GroupMCP and GroupSCAD" begin
  groups = [1:3, 4:5, [6, 8], 9:12]          # x[7] is in no group
  x = 2 .* randn(12)
  for (F, P, a) in ((GroupMCP, MCP, 2.5), (GroupSCAD, SCAD, 3.7))
    λ = 0.5
    f = F(λ, groups; a)
    w = sqrt.(length.(groups))
    @test f.w ≈ w
    pk(k) = P(λ * w[k], a)
    pv(k, t) = value_entry(pk(k), t, 1)
    @test f(x) ≈ sum(k -> pv(k, norm(x[groups[k]])), eachindex(groups))
    check_traits(f; proximable = true)
    for γ in (0.3, 1., 4.)                    # steps on both sides of the convexity limit
      y = prox(f, x, γ)
      @test y[7] == x[7]
      for k in eachindex(groups)
        G = groups[k]
        ref = radial_reference(t -> pv(k, t), x[G], γ)
        @test hobj(t -> pv(k, t), norm(y[G]), norm(x[G]), γ) <= hobj(t -> pv(k, t), norm(ref), norm(x[G]), γ) + 1e-9
        @test norm(y[G]) <= norm(x[G]) + 1e-12
        iszero(norm(y[G])) || @test dot(y[G], x[G]) ≈ norm(y[G]) * norm(x[G])      # along x[G]
      end
      check_prox(f, x, γ)
    end
    # with groups of one entry and unit weights it is the scalar penalty
    fs = F(λ, [[i] for i in 1:12], ones(12); a)
    @test prox(fs, x, 0.8) ≈ prox(P(λ, a), x, 0.8)
    @test fs(x) ≈ P(λ, a)(x)
    # zero groups: the optimality test of the group lasso
    g = randn(12)
    gl = GroupNormL2(λ, groups)
    z = zeros(12)
    @test kkt_violation(f, g, z, 2) ≈ kkt_violation(gl, g, z, 2)
    fa = F(λ, [groups; [[7]]], [w; 0.]; a)
    xa2 = AtomIterate(12, fa)
    @test kkt_violation(fa, g, xa2, 3) ≈ norm(g[groups[3]]) - λ * w[3]
    # on an AtomIterate with its groups: the block step and the active groups
    xv = 2 .* randn(12)
    xa = AtomIterate(xv, fa)
    @test fa(xa) ≈ fa(xv)
    ya = similar(xa)
    @test value_and_prox!(ya, fa, xa, 0.7) ≈ fa(prox(fa, xv, 0.7))
    @test Vector(ya) ≈ prox(fa, xv, 0.7)
    xc = copy(xa)
    prox_coordinate!(xc, fa, 2, 0.7)
    @test Vector(xc)[groups[2]] ≈ prox(fa, xv, 0.7)[groups[2]]
    xz = AtomIterate(12, fa)
    xz.storage[4:5] .= [3., 4.]
    prox_coordinate!(xz, fa, 2, 0.1)
    @test isactive(xz, 2)
    # restriction to some coordinates keeps a and the weights
    r = restrict(f, [1, 2, 4, 9])
    @test r isa F && r.a == f.a && r.groups == [[1, 2], [3], [4]] && r.w ≈ w[[1, 2, 4]]
    @test F(0.4f0, groups) isa F{Float32}
    @test prox(F(0.4f0, groups), randn(Float32, 12), 0.5f0) isa Vector{Float32}
    @test_throws DimensionMismatch F(1., groups, [1., 2.])
  end
  @test_throws ArgumentError GroupMCP(1., [1:2]; a = 0.)
  @test_throws ArgumentError GroupSCAD(1., [1:2]; a = 2.)
  @test_throws ArgumentError GroupMCP(-1., [1:2])
end

@testset "IndBallGroupL0" begin
  groups = [1:2, 3:5, [6], 7:8]               # x[9] is free
  x = [0.1, -0.2, 3., 1., 0., -2.5, 0.5, 0.5, 9.]
  norms = [norm(x[G]) for G in groups]
  f = IndBallGroupL0(2, groups)
  check_traits(f; proximable = true)
  @test f(x) == Inf
  y = prox(f, x, 1.)
  # the two groups of largest norm and the free index stay; the others are zeroed
  keep = sortperm(norms; rev = true)[1:2]
  for k in eachindex(groups)
    @test y[groups[k]] == (k in keep ? x[groups[k]] : zeros(length(groups[k])))
  end
  @test y[9] == 9.
  @test f(y) == 0
  # it is a projection: no other choice of s groups is closer
  for (i, j) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    z = zeros(9); z[9] = x[9]
    for k in (i, j)
      z[groups[k]] = x[groups[k]]
    end
    @test norm(y - x) <= norm(z - x) + 1e-12
  end
  # ties keep exactly s groups, the earlier first
  xt = [1., 0., 0., 1., 0., 1., 0., 0., 0.]
  @test prox(IndBallGroupL0(2, groups), xt, 1.) == [1., 0., 0., 1., 0., 0., 0., 0., 0.]
  @test prox(IndBallGroupL0(0, groups), x, 1.) == [zeros(8); 9.]
  @test prox(IndBallGroupL0(4, groups), x, 1.) == x
  # in place, and on an AtomIterate with its groups: the dropped groups are deactivated
  z = copy(x)
  prox!(z, f, z, 1.)
  @test z == y
  ga = [groups; [[9]]]
  fa = IndBallGroupL0(3, ga)
  xa = AtomIterate(x, ga)
  ya = similar(xa)
  prox!(ya, fa, xa, 1.)
  @test Vector(ya) == prox(fa, x, 1.)
  @test nactive(ya) == 3
  @test fa(ya) == 0 && fa(xa) == Inf
  @test_throws ArgumentError IndBallGroupL0(-1, groups)
end

# the subsets of 1:n with min(k, n) elements
subsets(n, k) = k >= n ? [collect(1:n)] : [findall(b -> b == '1', reverse(string(m; base = 2, pad = n))) for m in 0:2^n-1
                                          if count_ones(m) == k]

@testset "IndBallGroupL0 with at most k entries per group" begin
  groups = [1:3, 4:7, 8:9, 10:12]               # x[13] is free
  for trial in 1:20, s in 0:4, k in 0:4
    x = randn(13)
    f = IndBallGroupL0(s, groups; k)
    y = prox(f, x, 1.)
    @test f(y) == 0
    @test all(y[i] == 0 || y[i] == x[i] for i in 1:13)
    @test y[13] == x[13]
    # the projection maximizes the kept sum of squares over the supports with at
    # most s groups and k entries in each: by brute force over all of them
    best = [maximum(sum(abs2, x[G[S]]) for S in subsets(length(G), k)) for G in groups]
    kept = s == 0 ? 0.0 : sum(sort(best; rev = true)[1:s])
    @test sum(abs2, y[1:12]) ≈ kept atol = 1e-12
    @test value_and_prox!(similar(x), f, x, 1.) == 0
  end
  # ties keep exactly k entries and s groups, the earlier first
  xt = [1., -1., 1., 0., 2., 0., 0., 1., 1., 0., 0., 0., 5.]
  @test prox(IndBallGroupL0(2, groups; k = 2), xt, 1.) == [1., -1., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 5.]
  @test prox(IndBallGroupL0(1, groups; k = 1), xt, 1.) == [0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 5.]
  # within the bound on the groups, the groups are still cut to k entries
  @test prox(IndBallGroupL0(4, groups; k = 1), xt, 1.) == [1., 0., 0., 0., 2., 0., 0., 1., 0., 0., 0., 0., 5.]
  f = IndBallGroupL0(2, groups; k = 2)
  @test f(xt) == Inf
  @test f([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.]) == 0
  @test f([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.]) == Inf     # three entries in a group
  @test IndBallGroupL0(2, groups; k = 3)([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5.]) == 0
  # on an AtomIterate with its groups
  x = randn(13)
  ga = [groups; [[13]]]
  fa = IndBallGroupL0(3, ga; k = 2)
  xa = AtomIterate(x, ga)
  ya = similar(xa)
  prox!(ya, fa, xa, 1.)
  @test Vector(ya) == prox(fa, x, 1.)
  @test nactive(ya) == 3
  @test fa(ya) == 0 && fa(xa) == Inf
  @test prox(f, Float32.(x), 1.) == Float32.(prox(f, x, 1.))
  @test_throws ArgumentError IndBallGroupL0(1, groups; k = -1)
end

@testset "zero allocations" begin
  a_value(f, x) = f(x)
  a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
  a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
  a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
  x = randn(20)
  xs = SparseIterate(sprandn(20, 0.3))
  groups = [1:5, 6:10, 11:15, 16:20]
  xa = AtomIterate(randn(20), groups)
  for f in (CappedL1(0.3, 1.), CappedL1(0.3, 1., rand(20)), LogPenalty(0.3, 0.5), NormLq(0.3, 0.5),
            NormLq(0.3, 2/3), NormLq(0.3, 0.4))
    for v in (x, Float32.(x), xs)
      y = similar(v)
      @test allocs(a_value, f, v) == 0
      @test allocs(a_prox!, y, f, v, 0.7) == 0
      @test allocs(a_vprox!, y, f, v, 0.7) == 0
    end
    xc = copy(xs)
    @test allocs(a_cd!, xc, f, 3, 0.7) == 0
  end
  for f in (GroupMCP(0.4, groups), GroupSCAD(0.4, groups), IndBallGroupL0(2, groups), IndBallGroupL0(2, groups; k = 3))
    y = similar(x)
    @test allocs(a_value, f, x) == 0
    @test allocs(a_prox!, y, f, x, 0.7) == 0
    @test allocs(a_vprox!, y, f, x, 0.7) == 0
    ya = similar(xa)
    @test allocs(a_prox!, ya, f, xa, 0.7) == 0
    f isa IndBallGroupL0 && continue
    xc = copy(xa)
    @test allocs(a_cd!, xc, f, 2, 0.7) == 0
  end
end

end
