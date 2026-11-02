module BallConstrainedTest

using Test
using ProximalBase
using LinearAlgebra
using Random

using ..TestUtils

Random.seed!(1)

# the objective of the proximal problem of f on a set, γf(z) + ‖z - v‖²/2, Inf
# outside the set
obj(c, z, v, γ) = γ * c(z) + sum(abs2, z .- v) / 2

# random points on the sphere ‖z‖ = r, and points of the sphere near z
sphere_points(z, r, m) = [r .* w ./ norm(w) for w in (randn(size(z)...) for _ in 1:m)]
near_points(z, r, m) = [r .* w ./ norm(w) for w in (z .+ δ .* randn(size(z)...) for δ in (1e-6, 1e-3, 1e-1) for _ in 1:m)]

# the best point of the ball (or the sphere) supported on at most s entries, over
# every support: the projection of v onto the ball on the support
function l0_reference(v, s, r; sphere = false)
  n = length(v)
  best, bz = Inf, zero(v)
  for mask in 0:(2^n - 1)
    count_ones(mask) <= s || continue
    z = [(mask >> (i - 1)) & 1 == 1 ? v[i] : 0.0 for i in 1:n]
    nz = norm(z)
    if sphere
      iszero(nz) && continue
      z .*= r / nz
    elseif nz > r
      z .*= r / nz
    end
    d = sum(abs2, z .- v)
    d < best && ((best, bz) = (d, z))
  end
  bz
end

@testset "IndSphereL2" begin
  f = IndSphereL2(2.)
  @test prox(f, [3., 4.]) ≈ [1.2, 1.6]
  @test prox(f, zeros(3)) == [2., 0., 0.]
  @test f([1.2, 1.6]) == 0 && f([1., 1.]) == Inf && f(zeros(2)) == Inf
  @test f([1.2, 1.6 + 1e-12]) == 0                    # the tolerance of rounding
  M = randn(3, 4)
  @test prox(f, M) ≈ 2 .* M ./ norm(M)
  z = randn(5)
  w = copy(z)
  @test prox!(w, f, w, 1.) ≈ prox(f, z)               # in place
  @test value_and_prox!(similar(z), f, z, 1.) == 0
  check_traits(f; proximable = true)
  @test_throws ArgumentError IndSphereL2(0.)
  @test_throws ArgumentError IndSphereL2(-1.)
  @test_throws ArgumentError prox(f, Float64[])
  @test eltype(prox(IndSphereL2(1f0), Float32[3, 4])) == Float32
  @test allocs((y, x) -> prox!(y, f, x, 1.), similar(z), z) == 0
end

@testset "a norm on an ℓ2 ball" begin
  n = 8
  v = 2 .* randn(n)
  fs = (NormL1(0.3), NormL1(0.3, rand(n)), NormL1Nonneg(0.4), NormL2(0.5), NormLinf(0.2),
        GroupNormL2(0.4, [1:3, 4:5, [6, 8]]), GroupNormLinf(0.3, [1:4, 5:8]),
        SparseGroupNormL2(0.2, 0.3, [1:3, 4:5, [6, 8]]), Scaled(NormL1(1.), 0.3),
        ZeroFunction(), IndNonnegative())
  for f in fs, r in (0.5, 2., 50.), γ in (0.4, 1.5)
    @testset "$(nameof(typeof(f))), r = $r" begin
      c = Constrained(f, IndBallL2(r))
      check_traits(c; proximable = true, convex = true)
      p = check_prox(c, v, γ)
      # the proximal point of f, scaled onto the ball
      q = prox(f, v, γ)
      @test p ≈ (norm(q) > r ? (r / norm(q)) .* q : q)
      @test norm(p) <= r * (1 + 1e-12)
      # no point of the ball does better
      for w in sphere_points(p, r, 50), t in (0.3, 1.)
        @test obj(c, t .* w, v, γ) >= obj(c, p, v, γ) - 1e-10
      end
    end
  end

  # matrices: the nuclear norm and the trace on the Frobenius ball
  M = randn(6, 4)
  for (f, X) in ((NuclearNorm(0.4), M), (TracePSD(0.3), Matrix(Symmetric(randn(5, 5)))), (IndPSD(), Matrix(Symmetric(randn(5, 5)))))
    c = Constrained(f, IndBallL2(1.))
    p = check_prox(c, X, 0.7; symmetric = !(f isa NuclearNorm))
    q = prox(f, X, 0.7)
    @test p ≈ (norm(q) > 1 ? q ./ norm(q) : q)
    @test value_and_prox!(similar(X), c, X, 0.7) ≈ c(p)
  end

  # the nonconvex cones: the s largest entries, or the rank-r truncation, scaled
  for s in 0:4, _ in 1:10
    v6 = randn(6)
    c = Constrained(IndBallL0(s), IndBallL2(0.8))
    check_traits(c; proximable = true)
    @test prox(c, v6) ≈ l0_reference(v6, s, 0.8)
  end
  c = Constrained(IndBallRank(2), IndBallL2(1.))
  p = prox(c, M)
  @test rank(p) <= 2 && norm(p) <= 1 + 1e-12
  for _ in 1:200
    W = randn(6, 2) * randn(2, 4)
    W .*= rand() / norm(W)
    @test sum(abs2, W .- M) >= sum(abs2, p .- M) - 1e-10
  end

  # the value scales with the point
  c = Constrained(NormL1(0.5), IndBallL2(1.))
  p = prox(c, v)
  @test value_and_prox!(similar(v), c, v, 1.) ≈ 0.5 * norm(p, 1)
  @test c(p) ≈ 0.5 * norm(p, 1) && c(2 .* p) == Inf

  # an indicator stays an indicator, and its value is 0
  for C in (IndBallL2(1.), IndSphereL2(1.))
    @test ProximalBase._isindicator(typeof(Constrained(IndBallL0(2), C)))
    @test !ProximalBase._isindicator(typeof(Constrained(NormL1(1.), C)))
  end
  @test_throws ArgumentError Constrained(SqrNormL2(1.), IndBallL2(1.))
  @test_throws ArgumentError Constrained(MCP(1.), IndBallL2(1.))
  # MCP and SCAD share the grouping interface, but are not homogeneous norms.
  for F in (GroupMCP, GroupSCAD)
    f = F(1.0, [1:1, 2:2])
    @test f isa ProximalBase.AbstractGroupPenalty
    @test !(f isa ProximalBase.HomogeneousGroupNorm)
    @test f isa ProximalBase.AbstractGroupNorm   # preserve solvers' grouped-iterate selection
    @test_throws ArgumentError Constrained(f, IndBallL2(1.0))
    @test_throws ArgumentError Constrained(0.5 * f, IndBallL2(1.0))
  end
  @test_throws ArgumentError Constrained(IndBox(-1, 1), IndBallL2(1.))
end

@testset "a norm on an ℓ2 sphere" begin
  n = 7
  @testset "NormL1" begin
    w = rand(n)
    for f in (NormL1(0.4), NormL1(0.4, w)), r in (0.5, 3.), γ in (0.5, 2.), scale in (0.1, 3.)
      v = scale .* randn(n)
      c = Constrained(f, IndSphereL2(r))
      check_traits(c; proximable = true)
      p = check_prox(c, v, γ)
      @test norm(p) ≈ r
      cands = [sphere_points(p, r, 200); near_points(p, r, 20)]
      for i in 1:n, σ in (-1, 1)                # the points on one axis
        e = zeros(n); e[i] = σ * r
        push!(cands, e)
      end
      @test all(z -> obj(c, z, v, γ) >= obj(c, p, v, γ) - 1e-9, cands)
    end
    # every entry within its threshold: all the mass on the entry of largest margin
    c = Constrained(NormL1(1., [1., 2., 1., 4.]), IndSphereL2(2.))
    v = [0.5, -1.9, -0.8, 3.]
    @test prox(c, v) == [0., -2., 0., 0.]                 # margins -0.5, -0.1, -0.2, -1
    @test prox(c, v) == prox!(copy(v), c, copy(v), 1.)    # in place
    @test value_and_prox!(similar(v), c, v, 1.) ≈ 4.
    @test prox(c, zeros(4)) == [2., 0., 0., 0.]
    @test_throws ArgumentError prox(Constrained(NormL1(1., fill(Inf, 3)), IndSphereL2(1.)), ones(3))
  end

  @testset "GroupNormL2" begin
    groups = [1:3, 4:5, [6, 7]]
    for f in (GroupNormL2(0.5, groups), GroupNormL2(0.5, [1:3, [6, 7]])), γ in (0.5, 2.), scale in (0.2, 3.)
      v = scale .* randn(n)
      c = Constrained(f, IndSphereL2(1.5))
      p = check_prox(c, v, γ)
      @test norm(p) ≈ 1.5
      cands = [sphere_points(p, 1.5, 300); near_points(p, 1.5, 20)]
      for G in groups                            # the points on one group
        z = zeros(n); z[G] .= v[G]; push!(cands, 1.5 .* z ./ norm(z))
      end
      @test all(z -> obj(c, z, v, γ) >= obj(c, p, v, γ) - 1e-9, cands)
    end
    # every group within its threshold: all the mass on the group of largest margin
    f = GroupNormL2(1., [1:2, 3:4], [1., 0.6])
    c = Constrained(f, IndSphereL2(2.))
    v = [0.3, 0.4, -0.3, 0.]                      # margins 0.5 - 1 and 0.3 - 0.6
    @test prox(c, v) ≈ [0., 0., -2., 0.]
    @test prox!(copy(v), c, copy(v), 1.) ≈ prox(c, v)
    @test value_and_prox!(similar(v), c, v, 1.) ≈ 1.2
    @test prox(c, zeros(4)) == [0., 0., 2., 0.]   # the group of the smaller threshold
    # an index in no group is free on the sphere
    c = Constrained(GroupNormL2(1., [1:2, [4]]), IndSphereL2(2.))
    @test prox(c, [0.1, 0.2, 0., -0.3]) == [0., 0., 2., 0.]
    @test value_and_prox!(zeros(4), c, [0.1, 0.2, 0., -0.3], 1.) == 0
  end

  @testset "IndBallL0" begin
    for s in 1:4, _ in 1:10
      v = randn(6)
      c = Constrained(IndBallL0(s), IndSphereL2(1.3))
      p = prox(c, v)
      @test p ≈ l0_reference(v, s, 1.3; sphere = true)
      @test prox!(copy(v), c, copy(v), 1.) ≈ p
    end
    c = Constrained(IndBallL0(2), IndSphereL2(1.))
    @test prox(c, zeros(3)) == [1., 0., 0.]
    @test value_and_prox!(zeros(3), c, [1., 2., 3.], 1.) == 0
    # with a free diagonal, the diagonal stays and does not count
    M = randn(4, 4)
    P = prox(Constrained(IndBallL0(2; diagonal = false), IndSphereL2(1.)), M)
    Q = prox(IndBallL0(2; diagonal = false), M)
    @test P ≈ Q ./ norm(Q)
    @test prox(Constrained(IndBallL0(0; diagonal = false), IndSphereL2(1.)), M) ≈ Diagonal(diag(M)) ./ norm(diag(M))
    @test_throws ArgumentError prox(Constrained(IndBallL0(0), IndSphereL2(1.)), randn(3))
  end

  @test_throws ArgumentError Constrained(NormL2(1.), IndSphereL2(1.))
  @test_throws ArgumentError Constrained(GroupNormLinf(1., [1:2]), IndSphereL2(1.))
  @test_throws ArgumentError Constrained(SqrNormL2(1.), IndSphereL2(1.))
end

@testset "allocations, inference and Float32" begin
  n = 50
  v = randn(n)
  y = similar(v)
  cs = (Constrained(NormL1(0.3), IndBallL2(1.)), Constrained(GroupNormL2(0.3, [1:10, 11:30]), IndBallL2(1.)),
        Constrained(IndBallL0(5), IndBallL2(1.)), Constrained(NormL1(0.3), IndSphereL2(1.)),
        Constrained(NormL1(10.), IndSphereL2(1.)), Constrained(GroupNormL2(0.3, [1:10, 11:30]), IndSphereL2(1.)),
        Constrained(GroupNormL2(10., [1:25, 26:50]), IndSphereL2(1.)), Constrained(IndBallL0(5), IndSphereL2(1.)))
  for c in cs
    @test allocs((y, x) -> value_and_prox!(y, c, x, 0.7), y, v) == 0
    @inferred value_and_prox!(y, c, v, 0.7)
    @inferred prox!(y, c, v, 0.7)
    z = prox(c, Float32.(v), 0.7f0)
    @test eltype(z) == Float32
    @test z ≈ prox(c, v, 0.7) atol = 1e-5
  end
  M = randn(8, 5)
  c = Constrained(NuclearNorm(0.3), IndBallL2(1.))
  @test allocs((Y, X) -> value_and_prox!(Y, c, X, 1.), similar(M), M) == 0
end

@testset "normalization beyond the scale-factor range" begin
  for T in (Float32, Float64)
    for x in (T[0, nextfloat(zero(T))], T[floatmax(T), floatmax(T)])
      expected = T.(BigFloat.(x) ./ norm(BigFloat.(x)))
      for f in (NormL1(zero(T)), GroupNormL2(zero(T), [1:2]), IndBallL0(2))
        c = Constrained(f, IndSphereL2(one(T)))
        y = copy(x)
        @test value_and_prox!(y, c, y, one(T)) == zero(T)
        @test y ≈ expected
        @test prox(IndSphereL2(one(T)), x) ≈ expected
        if maximum(abs, x) > 1
          @test prox(Constrained(f, IndBallL2(one(T))), x) ≈ expected
          @test prox(IndBallL2(one(T)), x) ≈ expected
        end
      end
    end
    tiny = nextfloat(zero(T))
    for f in (NormL1(one(T)), GroupNormL2(one(T), [[1], [2]], ones(T, 2)))
      @test prox(Constrained(f, IndSphereL2(one(T))), T[0, tiny]) == T[0, 1]
    end
    r = T === Float32 ? T(1e-30) : T(1e-300)
    # The normal scale factor can still be inaccurate when the input norm
    # itself is subnormal; evaluate the value at the normalized point.
    for f in (NormL1(zero(T)), NormL1(one(T)))
      c = Constrained(f, IndSphereL2(r))
      y = zeros(T, 2)
      value = value_and_prox!(y, c, T[tiny, tiny], one(T))
      @test value ≈ f(y)
      @test norm(y)/r ≈ one(T)
    end
    large = floatmax(T)/2
    f = NormL1(large, fill(inv(large), 2))
    @test prox(f, T[7, 1], T(4)) ≈ T[3, 0]
    @test prox(Constrained(f, IndSphereL2(one(T))), T[7, 1], T(4)) ≈ T[1, 0]
    @test prox(Constrained(f, IndSphereL2(one(T))), T[1, 2], T(4)) ≈ T[0, 1]
  end
end

@testset "constrained steps with unrepresentable inner norms" begin
  for T in (Float32, Float64)
    large = floatmax(T)
    x = T[large, large, large, 0]
    groups = [1:2, 3:4]
    families = ((NormL2(large/2), NormL2(T(0.5))),
                (GroupNormL2(large/2, groups, ones(T,2)), GroupNormL2(T(0.5), groups, ones(T,2))),
                (GroupNormLinf(large/2, groups, ones(T,2)), GroupNormLinf(T(0.5), groups, ones(T,2))),
                (SparseGroupNormL2(large/10, large/2, groups, ones(T,2)),
                 SparseGroupNormL2(T(0.1), T(0.5), groups, ones(T,2))))
    for (f, scaled) in families
      # Homogeneity: prox_f(x)/large = prox_{f/large}(x/large).
      q = prox(scaled, x/large)
      expected = q/norm(q)
      c = Constrained(f, IndBallL2(one(T)))
      y = copy(x)
      value = value_and_prox!(y, c, y, one(T))
      @test y ≈ expected rtol=32eps(T)
      @test value ≈ f(y) rtol=32eps(T)
    end
    # Independent group-sphere formula in higher precision.
    norms = [norm(BigFloat.(x[G])) for G in groups]
    margins = norms .- BigFloat(large/2)
    expected = zeros(T,4)
    for (G, n, m) in zip(groups, norms, margins)
      expected[G] .= T.((m/norm(margins)) .* BigFloat.(x[G]) ./ n)
    end
    f = GroupNormL2(large/2, groups, ones(T,2))
    @test prox(Constrained(f, IndSphereL2(one(T))), x) ≈ expected rtol=32eps(T)
  end
end

@testset "balls and spheres at extreme scales" begin
  for T in (Float32, Float64)
    tiny, huge = T === Float32 ? (T(1e-30), T(1e30)) : (T(1e-200), T(1e200))
    # Squared norms underflow or overflow here, while the norms themselves
    # and the results are representable.
    @test prox(Constrained(NormL1(one(T)), IndBallL2(one(T))), T[huge, 0]) ≈ T[1, 0]
    @test prox(Constrained(NormL1(tiny / 10), IndSphereL2(one(T))), T[0, tiny]) ≈ T[0, 1]
    @test prox(Constrained(IndBallL0(1), IndSphereL2(one(T))), T[0, tiny]) ≈ T[0, 1]
    groups = [1:2, 3:4]
    c = Constrained(GroupNormL2(tiny / 10, groups), IndSphereL2(one(T)))
    @test prox(c, T[0, tiny, 0, 0]) ≈ T[0, 1, 0, 0]
    # A large level and a compensating small weight give the threshold 4 at
    # the step 4; forming γλ0 first overflows.
    large = floatmax(T) / 2
    c = Constrained(GroupNormL2(large, groups, fill(inv(large), 2)), IndSphereL2(one(T)))
    y = zeros(T, 4)
    @test value_and_prox!(y, c, T[7, 0, 0, -1], T(4)) ≈ one(T)
    @test y ≈ T[1, 0, 0, 0]
    @test value_and_prox!(y, c, T[1, 0, 0, -2], T(4)) ≈ one(T)     # every group within its threshold
    @test y ≈ T[0, 0, 0, -1]
  end
end

@testset "sphere prox of the ℓ1 norm: exact margins and thresholds in the working precision" begin
  B = 2.0^60
  # The margins are -1, -1 and 0: the entry (2,2) wins, though B - 0 and B - 1 round alike.
  c = Constrained(NormL1(1.0, [1.0 1.0; 1.0 B]), IndSphereL2(1.0))
  y = ones(2, 2)
  v = value_and_prox!(y, c, [0.0 0.0; 0.0 B], 1.0)
  @test y == [0.0 0.0; 0.0 1.0]
  @test v == 2.0^60
  @test prox(c, [0.0 0.0; 0.0 B], 1.0) == [0.0 0.0; 0.0 1.0]

  # thresholds finite in y's type but not in x's
  c = Constrained(NormL1(2.0^160, ones(2, 2)), IndSphereL2(1.0))
  y = zeros(2, 2)
  v = value_and_prox!(y, c, Float32[1 0; 0 2], 1.0)
  @test y == [0.0 0.0; 0.0 1.0]
  @test v == 2.0^160
  # still an error when no threshold is finite in the working type
  c = Constrained(NormL1(big(2.0)^2000, ones(2, 2)), IndSphereL2(1.0))
  @test_throws ArgumentError value_and_prox!(zeros(2, 2), c, ones(2, 2), 1.0)

  # the product γλ0w overflows Float32 but not Float64
  c = Constrained(NormL1(2f0^80, ones(Float32, 2, 2)), IndSphereL2(1f0))
  for y in (zeros(2, 2), SymmetricSparseIterate(2))
    v = value_and_prox!(y, c, zeros(2, 2), 2f0^80)
    @test v == 2.0^80                    # λ0 ⋅ w ⋅ r; the threshold is 2^160
    @test y[1, 1] == 1.0 && sum(abs2, y) == 1.0
  end
  @test_throws ArgumentError value_and_prox!(zeros(Float32, 2, 2), c, zeros(Float32, 2, 2), 2f0^80)

  # an unweighted penalty and the same type everywhere give the usual values
  for T in (Float32, Float64)
    c = Constrained(NormL1(T(0.3)), IndSphereL2(T(2)))
    x = T[1, -2, 0.1, 0.5]
    z = copy(x)
    m = max.(abs.(x) .- T(0.3) * T(0.7), 0)
    @test prox(c, x, T(0.7)) ≈ copysign.(T(2) .* m ./ norm(m), x) rtol=4eps(T)
    v = value_and_prox!(z, c, z, T(0.7))                          # y === x
    @test z ≈ copysign.(T(2) .* m ./ norm(m), x) rtol=4eps(T)
    @test v ≈ T(0.3) * norm(z, 1) rtol=4eps(T)
    z = copy(x)
    value_and_prox!(z, Constrained(NormL1(T(5)), IndSphereL2(T(2))), z, T(1))    # fallback, y === x
    @test z == T[0, -2, 0, 0]
  end

  # a wide scale range, also into a narrower output: nothing is narrowed before the norm
  for (x, γ) in (([1e300, 2e300, 1e299], 1.0), ([1e-300, 2e-300, 1e-299], 1e-310))
    c = Constrained(NormL1(0.5), IndSphereL2(1.0))
    mb = setprecision(BigFloat, 400) do
      max.(abs.(big.(x)) .- big(γ) * big(0.5), 0)
    end
    for T in (Float32, Float64)
      y = zeros(T, 3)
      value_and_prox!(y, c, x, γ)
      @test y ≈ T.(mb ./ norm(mb)) rtol=4eps(T)
    end
  end

  # the winner of the fallback, against the exact margins in BigFloat
  Random.seed!(418)
  for trial in 1:400
    m = rand(2:6)
    if iseven(trial)      # entries of very different size: B - 0 and B - 1 round alike
      b = 2.0^rand(55:70)
      big_entry = rand(Bool, m)
      w = [big_entry[i] ? b : 1 + rand() for i in 1:m]
      x = [big_entry[i] ? b * (1 - rand(0:3) * 2.0^-53) : rand() / 2 for i in 1:m] .* rand((-1, 1), m)
      γ, λ0 = 1.0, 1.0
    else
      x = [rand((-1, 1)) * 2.0^rand(-80:80) * (1 + rand()) for _ in 1:m]
      w = [2.0^rand(-80:80) * (1 + rand()) for _ in 1:m]
      γ, λ0 = 2.0^rand(-20:20), 2.0^rand(-20:20)
    end
    margins = setprecision(BigFloat, 600) do
      [abs(big(x[i])) - big(γ) * big(λ0) * big(w[i]) for i in 1:m]
    end
    all(<=(0), margins) || continue
    kbest = argmax(margins)               # the first of equal margins
    c = Constrained(NormL1(λ0, w), IndSphereL2(1.0))
    y = zeros(m)
    value_and_prox!(y, c, x, γ)
    @test findall(!iszero, y) == [kbest]
    @test y[kbest] == sign(x[kbest])
  end

  # infinite weights: on the first entry, on the best entry
  c = Constrained(NormL1(1.0, [Inf, 1.0, 1.0]), IndSphereL2(1.0))
  y = zeros(3)
  value_and_prox!(y, c, [5.0, 0.0, 0.5], 1.0)
  @test y == [0.0, 0.0, 1.0]
  c = Constrained(NormL1(1.0, [1.0, Inf, 1.0]), IndSphereL2(1.0))
  value_and_prox!(y, c, [0.0, 5.0, 0.5], 1.0)
  @test y == [0.0, 0.0, 1.0]
  # signed zeros and ties: the first entry wins with its own sign
  c = Constrained(NormL1(1.0), IndSphereL2(1.0))
  value_and_prox!(y, c, [-0.0, 0.0, 0.0], 1.0)
  @test y == [-1.0, 0.0, 0.0]
  value_and_prox!(y, c, [0.0, -0.0, 0.0], 1.0)
  @test y == [1.0, 0.0, 0.0]

  # a sparse output
  y = SparseIterate(4)
  c = Constrained(NormL1(0.1), IndSphereL2(1.0))
  value_and_prox!(y, c, [3.0, 0.0, -4.0, 0.0], 1.0)
  @test norm(y) ≈ 1
  @test Vector(y) ≈ [0.6, 0.0, -0.8, 0.0] rtol=0.02
end

end
