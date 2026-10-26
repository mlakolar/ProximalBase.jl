module SparseGroupTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(5)

# prox of f + g from the proximal operators of f and g alone, by the Dykstra-like
# proximal algorithm (Bauschke and Combettes), which converges to prox_{f+g}(v):
# a reference independent of the closed form
function prox_of_sum(f, g, v, γ; iters = 20_000)
  x = copy(v)
  p, q = zero(v), zero(v)
  for _ in 1:iters
    y = prox(g, x + p, γ)
    p = x + p - y
    xn = prox(f, y + q, γ)
    q = y + q - xn
    x = xn
  end
  x
end

value_ref(λ1, λ2, groups, w, x) = λ1 * sum(abs, x[reduce(vcat, collect.(groups))]) + λ2 * sum(w[k] * norm(x[groups[k]]) for k in eachindex(groups))

a_value(f, x) = f(x)
a_prox!(y, f, x, γ) = prox!(y, f, x, γ)
a_vprox!(y, f, x, γ) = value_and_prox!(y, f, x, γ)
a_cd!(x, f, k, γ) = prox_coordinate!(x, f, k, γ)
a_kkt(f, g, x, k) = kkt_violation(f, g, x, k)

@testset "SparseGroupNormL2" begin
  p = 20
  groups = [1:5, 6:10, 11:15, 16:20]
  λ1, λ2 = 0.3, 0.5
  f = SparseGroupNormL2(λ1, λ2, groups)
  w = sqrt.(length.(groups))
  check_traits(f; proximable = true, convex = true)

  @testset "value and proximal operator" begin
    for _ in 1:5
      x = 2 .* randn(p)
      γ = 0.5 + rand()
      @test f(x) ≈ value_ref(λ1, λ2, groups, w, x)
      z = prox(f, x, γ)
      @test z ≈ prox_of_sum(NormL1(λ1), GroupNormL2(λ2, groups), x, γ) atol = 1e-8
      # the closed form: the composition of the two proximal operators
      @test z ≈ prox(GroupNormL2(λ2, groups), prox(NormL1(λ1), x, γ), γ)
      @test value_and_prox!(similar(x), f, x, γ) ≈ f(z)
      check_prox(f, x, γ)
    end
    # some groups vanish, and some entries of the others
    x = [0.1, -0.2, 0.05, 0.3, -0.1, 3.0, -0.2, 2.0, 0.1, -4.0, zeros(10)...]
    z = prox(f, x, 1.)
    @test all(iszero, z[1:5]) && all(iszero, z[11:20])
    @test iszero(z[7]) && iszero(z[9]) && all(!iszero, z[[6, 8, 10]])
    # the extreme levels: the group lasso and the ℓ1 norm
    x = randn(p)
    @test prox(SparseGroupNormL2(0, λ2, groups), x) ≈ prox(GroupNormL2(λ2, groups), x)
    @test prox(SparseGroupNormL2(λ1, 0, groups), x) ≈ prox(NormL1(λ1), x)
    @test SparseGroupNormL2(λ1, 0, groups)(x) ≈ NormL1(λ1)(x)
    # weights of one's own, and indices in no group, which are not penalized
    fw = SparseGroupNormL2(λ1, λ2, [1:3, [5, 8, 9]], [1.0, 2.0])
    x = randn(10)
    z = prox(fw, x, 1.)
    @test z[[4, 6, 7, 10]] == x[[4, 6, 7, 10]]
    @test fw(x) ≈ value_ref(λ1, λ2, [1:3, [5, 8, 9]], [1.0, 2.0], x)
    @test z ≈ prox_of_sum(NormL1(λ1, [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]), GroupNormL2(λ2, [1:3, [5, 8, 9]], [1.0, 2.0]), x, 1.) atol = 1e-8
    # matrices: groups of linear indices, here the rows
    M = randn(4, 3)
    fr = SparseGroupNormL2(λ1, λ2, [i:4:12 for i in 1:4])
    @test vec(prox(fr, M)) ≈ prox(fr, vec(M))
    # Float32 and in place
    x = randn(p)
    @test prox(f, Float32.(x)) isa Vector{Float32}
    @test prox(f, Float32.(x)) ≈ Float32.(prox(f, x)) rtol = 1e-5
    y = copy(x)
    prox!(y, f, y, 0.7)
    @test y ≈ prox(f, x, 0.7)
    @inferred f(x)
    @inferred value_and_prox!(similar(x), f, x, 0.7)
    # errors
    @test_throws ArgumentError SparseGroupNormL2(-1, λ2, groups)
    @test_throws ArgumentError SparseGroupNormL2(λ1, -1, groups)
    @test_throws DimensionMismatch SparseGroupNormL2(λ1, λ2, groups, [1.0])
    @test_throws ArgumentError SparseGroupNormL2(λ1, λ2, [1:3, 3:5])
  end

  @testset "one group: the sparse group lasso of a vector" begin
    K = 6
    fk = SparseGroupNormL2(λ1, λ2, [1:K], [1.0])
    for _ in 1:5
      v = randn(K)
      s = shrink.(v, λ1)
      @test prox(fk, v) ≈ max(1 - λ2 / norm(s), 0) .* s
      @test fk(v) ≈ λ1 * norm(v, 1) + λ2 * norm(v)
    end
    @test allocs(a_prox!, zeros(K), fk, randn(K), 0.7) == 0
    # as the fiber function of a FiberSum: the group graphical lasso penalty
    q = 5
    x = randn(q, q, K)
    fb = FiberSum(fk)
    z = prox(fb, x, 0.7)
    for j in 1:q, i in 1:q
      @test z[i, j, :] ≈ prox(fk, x[i, j, :], 0.7)
    end
    @test fb(x) ≈ λ1 * norm(x, 1) + λ2 * sum(norm(x[i, j, :]) for i in 1:q, j in 1:q)
    @test value_and_prox!(similar(x), fb, x, 0.7) ≈ fb(z)
    @test allocs(a_prox!, similar(x), fb, x, 0.7) == 0
    # symmetric slices with a free diagonal, as for precision matrices
    s = x .+ permutedims(x, (2, 1, 3))
    fs = FiberSum(fk; diagonal = nothing, symmetric = true)
    zs = prox(fs, s, 0.7)
    @test all(zs[:, :, k] ≈ zs[:, :, k]' for k in 1:K)
    @test all(zs[i, i, :] == s[i, i, :] for i in 1:q)
    @test zs[3, 1, :] ≈ prox(fk, s[3, 1, :], 0.7)
  end

  @testset "AtomIterate, coordinate steps and optimality" begin
    x = randn(p)
    xa = AtomIterate(x, f)
    @test f(xa) ≈ f(x)
    γ = 0.8
    @test Vector(prox(f, xa, γ)) ≈ prox(f, x, γ)
    ya = copy(xa)
    @test value_and_prox!(ya, f, xa, γ) ≈ f(prox(f, x, γ))
    # a block step on group 2 is the proximal operator on that group
    xc = AtomIterate(copy(x), f)
    prox_coordinate!(xc, f, 2, γ)
    @test Vector(xc)[6:10] ≈ prox(f, x, γ)[6:10]
    @test Vector(xc)[[1:5; 11:20]] == x[[1:5; 11:20]]
    # a group at zero: the coordinate step from zero keeps it at zero exactly when
    # kkt_violation ≤ 0, that is ‖soft(g[G], λ1)‖ ≤ λ2 w
    for _ in 1:20
      g = 0.6 .* randn(p)
      z0 = AtomIterate(zeros(p), f)
      viol = kkt_violation(f, g, z0, 3)
      @test viol ≈ norm(shrink.(g[11:15], λ1)) - λ2 * w[3]
      step = AtomIterate(-g, f)                       # the gradient step from zero, γ = 1
      prox_coordinate!(step, f, 3, 1.)
      @test all(iszero, Vector(step)[11:15]) == (viol <= 0)
    end
    # on an array: an index of a zero group, of a nonzero group, and in no group
    g = randn(p)
    xz = copy(x); xz[6:10] .= 0; xz[2] = 0
    @test kkt_violation(f, g, xz, 7) ≈ norm(shrink.(g[6:10], λ1)) - λ2 * w[2]
    @test kkt_violation(f, g, xz, 2) ≈ abs(g[2]) - λ1
    fw = SparseGroupNormL2(λ1, λ2, [1:3])
    @test kkt_violation(fw, g, zeros(p), 5) == abs(g[5])
    @test_throws ArgumentError kkt_violation(f, g, SymmetricSparseIterate(5), 1)
    # restriction to S: the value on the vectors supported on S
    S = [2, 3, 7, 8, 9, 16]
    xs = zeros(p); xs[S] .= randn(length(S))
    @test restrict(f, S)(xs[S]) ≈ f(xs)
    @test restrict(f, S) isa SparseGroupNormL2
  end

  @testset "allocations" begin
    x = randn(p)
    zero_value = allocs(a_value, f, x)
    @test zero_value == 0
    @test allocs(a_prox!, similar(x), f, x, 0.7) == 0
    @test allocs(a_vprox!, similar(x), f, x, 0.7) == 0
    z = copy(x)
    @test allocs(a_prox!, z, f, z, 0.7) == 0
    @test allocs(a_prox!, similar(Float32.(x)), f, Float32.(x), 0.7) == 0
    xa = AtomIterate(x, f)
    @test allocs(a_value, f, xa) == 0
    @test allocs(a_vprox!, copy(xa), f, xa, 0.7) == 0
    @test allocs(a_cd!, copy(xa), f, 2, 0.7) == 0
    @test allocs(a_kkt, f, x, xa, 3) == 0
    @test allocs(a_kkt, f, x, x, 3) == 0
  end
end

@testset "sparse group norm at extreme magnitudes" begin
  for T in (Float32, Float64)
    large = T == Float32 ? T(1e30) : T(1e200)
    for scale in (large, inv(large))
      x = T[0, scale]
      f = SparseGroupNormL2(scale/4, scale/4, [1:2], T[1])
      @test prox(f, x) ≈ T[0, scale/2] rtol=16eps(T)
      @test f(x) ≈ scale*(scale/2) rtol=16eps(T)
      a = AtomIterate(x, f)
      prox_coordinate!(a, f, 1, one(T))
      @test Vector(a) ≈ T[0, scale/2] rtol=16eps(T)
    end
  end
end


end
