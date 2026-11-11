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
    @test y[1, 1] == 1.0 && sum(abs2, Matrix(y)) == 1.0
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

@testset "sphere prox of the ℓ1 norm: a diagonal entry against an off-diagonal pair, exactly (A2, #473)" begin
  # In the fallback a diagonal entry with margin D reaches r⋅D and a pair with margin P reaches
  # √2⋅r⋅P, so the diagonal wins when D > √2 P, decided exactly on the margins n - t; the tie is
  # only D = P = 0 (√2 is irrational), where the earlier candidate stays.
  ref(nd, td, np, tp) = setprecision(BigFloat, 10_000) do
    Int(sign((big(nd) - big(td)) - sqrt(big(2)) * (big(np) - big(tp))))
  end
  sgn = ProximalBase._diagonal_pair_sign

  # a decision the rounded ratio got wrong: the pair is better by 6e-8 of the objective
  c = Constrained(NormL1(1f0), IndSphereL2(1f0))
  X = Float32[0.36858305 0.5535208; 0.5535208 0]
  @test ref(X[1, 1], 1f0, X[2, 1], 1f0) == -1
  y = SymmetricSparseIterate(Float32, 2)
  value_and_prox!(y, c, X, 1f0)
  @test Matrix(y) == [0 1; 1 0] .* (1f0 / sqrt(2f0))

  # the sign cases, zeros, subnormals and the ends of the range
  for T in (Float16, Float32, Float64)
    tiny, big_ = nextfloat(zero(T)), floatmax(T)
    cases = [(T(0), T(0), T(0), T(0)), (T(0), T(0), T(0), T(1)), (T(0), T(1), T(0), T(0)),
             (T(1), T(0), T(0), T(1)), (T(0), T(1), T(1), T(0)), (T(3), T(1), T(2), T(1)),
             (T(1), T(3), T(1), T(2)), (tiny, T(0), tiny, T(0)), (2tiny, T(0), tiny, T(0)),
             (T(0), 2tiny, T(0), tiny), (big_, T(0), big_, T(0)), (big_, T(0), prevfloat(big_ / sqrt(T(2))), T(0)),
             (big_, tiny, big_ / T(1.5), tiny), (T(0), big_, T(0), big_ / sqrt(T(2))),
             (T(Inf), T(1), T(1), T(0)), (T(NaN), T(1), T(1), T(0))]
    for q in cases
      @test all(isfinite, q) ? sgn(q...) == ref(q...) : sgn(q...) isa Int    # non-finite: the block is NaN (E2)
    end
  end

  # near ties against the exact sign, with margins whose subtraction rounds and wide exponents
  rng = Xoshiro(473)
  for T in (Float16, Float32, Float64), _ in 1:2000
    e = rand(rng, -12:12)
    p = abs(T(randn(rng))) * T(2)^e
    tp = rand(rng, Bool) ? zero(T) : abs(T(randn(rng))) * T(2)^(e + rand(rng, -40:4))
    td = rand(rng, Bool) ? zero(T) : abs(T(randn(rng))) * T(2)^(e + rand(rng, -40:4))
    np = tp + p
    D = setprecision(() -> sqrt(big(2)) * (big(np) - big(tp)), BigFloat, 600)
    nd = max(zero(T), nextfloat(T(big(td) + D), rand(rng, -3:3)))
    q = rand(rng, Bool) ? (nd, td, np, tp) : (td, nd, tp, np)        # both positive or both negative
    all(isfinite, q) || continue
    @test sgn(q...) == ref(q...)
  end

  # through value_and_prox!, with the diagonal before the pair and after it, and in a wider U
  for (S, T) in ((Float32, Float32), (Float32, Float64), (Float64, Float64)), first_diag in (true, false), _ in 1:200
    b = S(0.5) + S(0.45) * rand(rng, S)
    a = nextfloat(S(1 - sqrt(big(2)) * (1 - big(b))), rand(rng, -2:2))
    U = promote_type(S, T)
    s = ref(U(a), one(U), U(b), one(U))
    X = first_diag ? S[a b; b 0] : S[0 b; b a]
    W = first_diag ? T[1 1; 1 1] : T[4 1; 1 1]
    c = Constrained(NormL1(one(T), W), IndSphereL2(one(T)))
    y = SymmetricSparseIterate(T, 2)
    value_and_prox!(y, c, X, one(T))
    d = first_diag ? (1, 1) : (2, 2)
    @test s != 0
    @test Matrix(y) == (s > 0 ? T[i == j == d[1] for i in 1:2, j in 1:2] : T[0 1; 1 0] .* (one(T) / sqrt(T(2))))
  end

  # the exact tie D = P = 0: the earlier candidate stays
  c = Constrained(NormL1(1.0, [4.0 1.0; 1.0 1.0]), IndSphereL2(1.0))
  @test Matrix(prox(Constrained(NormL1(1.0), IndSphereL2(1.0)), SymmetricSparseIterate([1.0 1.0; 1.0 0.0]), 1.0)) == [1 0; 0 0]
  y = SymmetricSparseIterate(2)
  value_and_prox!(y, c, [0.0 1.0; 1.0 1.0], 1.0)
  @test Matrix(y) == [0 1; 1 0] ./ sqrt(2.0)

  # margins that cancel as far as possible: D = a⋅2^k, P = b⋅2^k with a² - 2b² = ±1 (Pell), each
  # written as n - t with a rounded n and a small t, so the tails of the margins decide; scaled
  # into the subnormals and to the top of the range
  pell = [(big(1), big(1))]
  for _ in 1:200
    a, b = pell[end]
    push!(pell, (a + 2b, a + b))
  end
  function as_margin(::Type{T}, m, k) where {T}
    n = T(m, RoundUp)
    t = T(big(n) - m)
    big(n) - big(t) == m || return nothing
    n, t = ldexp(n, k), ldexp(t, k)
    isfinite(n) && setprecision(() -> ldexp(big(n), -k) - ldexp(big(t), -k) == m, BigFloat, 4000) ? (n, t) : nothing
  end
  for T in (Float16, Float32, Float64)
    q = precision(T)
    lo, hi = exponent(nextfloat(zero(T))), exponent(floatmax(T))
    for (a, b) in pell
      bits = ndigits(a, base = 2)
      q - 2 <= bits <= q + 30 || continue
      for k in (lo, lo + q, -bits, 0, hi - bits - 1)
        x, y = as_margin(T, a, k), as_margin(T, b, k)
        (x === nothing || y === nothing) && continue
        for qq in ((x..., y...), (x[2], x[1], y[2], y[1]))     # both positive, both negative
          @test sgn(qq...) == ref(qq...)
        end
      end
    end
  end

  # within the band in the subnormals: a pair margin of m smallest subnormals, 2^(q-2) ≤ m < 2^(q-1)
  for T in (Float32, Float64)
    for _ in 1:60
      k = precision(T) - 1
      m = rand(rng, 2^(k - 1):2^k - 1)
      bh = m * nextfloat(zero(T))
      ah = T(sqrt(big(2)) * big(bh))
      for qq in ((ah, zero(T), bh, zero(T)), (nextfloat(ah), zero(T), bh, zero(T)), (zero(T), ah, zero(T), bh))
        @test abs(abs(qq[1] - qq[2]) / abs(qq[3] - qq[4]) - sqrt(T(2))) <= 8eps(T)    # the exact path
        @test sgn(qq...) == ref(qq...)
      end
    end
  end

  # BigFloat at the current precision (T7), with exponents far apart: the exact comparison stays
  # the size of the precision
  for prec in (24, 64, 256)
    setprecision(BigFloat, prec) do
      for (a, b) in pell[40:20:end], k in (-10^6, 0, 10^6), gap in (0, 10^7)
        n1, n2 = BigFloat(a, RoundUp), BigFloat(b, RoundUp)
        qq = (ldexp(n1, k), ldexp(n1 - a, k) + ldexp(BigFloat(1), k - gap), ldexp(n2, k), ldexp(n2 - b, k))
        @test sgn(qq...) == ref(qq...)
      end
      qq = (sqrt(BigFloat(2)), ldexp(BigFloat(1), -10^7), BigFloat(1), BigFloat(0))
      @test sgn(qq...) == ref(qq...)
      @test @allocated(sgn(qq...)) < 10^5
    end
  end

  # a structured input in which every pair is within the band of the diagonal entry: exact, and
  # nothing allocates
  for (wd, p) in ((prevfloat(sqrt(2.0)), 16), (sqrt(2.0), 16), (nextfloat(sqrt(2.0)), 16))
    W = ones(p, p)
    W[diagind(W)] .= wd
    c = Constrained(NormL1(1.0, W), IndSphereL2(1.0))
    y, X = SymmetricSparseIterate(p), zeros(p, p)
    value_and_prox!(y, c, X, 1.0)
    s = ref(0.0, wd, 0.0, 1.0)
    @test Matrix(y) == (s > 0 ? [i == j == 1 for i in 1:p, j in 1:p] : [(i, j) in ((2, 1), (1, 2)) for i in 1:p, j in 1:p] ./ sqrt(2.0))
    allocs(y, c, X) = @allocated value_and_prox!(y, c, X, 1.0)
    allocs(y, c, X)
    @test allocs(y, c, X) == 0
  end

  # no allocation and a stable type, outside the band of the tie and within it
  f(q) = sgn(q...)
  for q in ((2.0, 0.5, 1.0, 0.2), (2f0, 0.5f0, 1f0, 0.2f0), (0.1, 0.5, 0.1, 0.2),
            (sqrt(2.0), 0.0, 1.0, 0.0), (0.0, sqrt(2f0), 0.0, 1f0), (Float16(1.414), Float16(0), Float16(1), Float16(0)),
            (3.0, 3.0 - sqrt(2.0) + 2.0^-60, 2.0, 1.0))
    f(q)
    @test @allocated(f(q)) == 0
    @test @inferred(sgn(q...)) isa Int
    @test sgn(q...) == ref(q...)
  end
end

@testset "sphere prox of the ℓ1 norm: a NaN entry, or two infinite ones, gives NaN in the whole free block (E2, E3)" begin
  # The sphere couples every free entry, so a NaN among them is NaN in every free output entry and in
  # the value, whichever entry would have won the selection, on either branch and at any weight.
  # Two infinite free entries have no limit (E3, #469). One infinite free entry has the limit ±r there,
  # and a hard-fixed (infinite) weight outputs zero: both are in constrained_norml1_infinite.jl.
  # Not [1, 0] with the value 1 (before #441), and not NaN beside a zero (#441).
  PT = ProximalBase.Testing
  allnan(y) = all(isnan, collect(y))
  n = 4
  bad = ([NaN, 0, 0, 0], [0, NaN, 0, 0], [0, 0, 0, NaN],       # first, middle and last entry
         [5, NaN, 0, 0], [0.1, NaN, 0, 0.2],                    # a NaN that loses: shrink, fallback
         [NaN, 3, NaN, 0.5], [NaN, NaN, NaN, NaN],              # several
         [Inf, Inf, 0, 0], [Inf, -Inf, 0, 1], [Inf, NaN, 0, 0], [NaN, -Inf, 0, 0])
  weights = (nothing, [1, 1, 1, 1], [0, 1, 1, 1])
  for S in (Float32, Float64), T in (Float32, Float64), w in weights, λ in (1, 1e30), γ in (1, 0)
    f = w === nothing ? NormL1(T(λ)) : NormL1(T(λ), T.(w))
    c = Constrained(f, IndSphereL2(T(1)))
    for xs in bad
      x = S.(xs)
      for y in (zeros(T, n), SparseIterate(T, n), BlockIterate(T, n, [[1, 2], [3], [4]]))
        v = value_and_prox!(y, c, x, T(γ))
        @test allnan(y) && isnan(v) && v isa T
        y isa Array || PT.test_iterate(y)
        fill!(y, zero(T))
        @test prox!(y, c, x, T(γ)) === y && allnan(y)
      end
      @test allnan(prox(c, x, T(γ)))
      # y === x, in a dense array and in a sparse iterate that holds the argument
      for make in (copy, SparseIterate), op in (value_and_prox!, prox!)
        z = make(T.(xs))
        r = op(z, c, z, T(γ))
        @test allnan(z) && (op === prox! ? r === z : isnan(r) && r isa T)
        z isa Array || PT.test_iterate(z)
      end
    end
  end
  # a symmetric output: the argument's symmetric part, with the entry on the diagonal or in an
  # off-diagonal pair, on the fallback branch (zeros) and with a diagonal entry that wins (5)
  for S in (Float32, Float64), T in (Float32, Float64)
    c = Constrained(NormL1(T(1), ones(T, 3, 3)), IndSphereL2(T(1)))
    for (i, j) in ((1, 1), (3, 3), (2, 1), (3, 2)), b in (NaN,), top in (0, 5)
      X = zeros(S, 3, 3)
      X[2, 2] = top
      X[i, j] = X[j, i] = b                       # both entries of a pair, as a symmetric matrix has them
      for op in (value_and_prox!, prox!)
        y = SymmetricSparseIterate(T, 3)
        r = op(y, c, X, T(1))
        @test allnan(Matrix(y)) && (op === prox! ? r === y : isnan(r) && r isa T)
        PT.test_iterate(y)
        z = SymmetricSparseIterate(T.(X))         # y === x
        r = op(z, c, z, T(1))
        @test allnan(Matrix(z)) && (op === prox! ? r === z : isnan(r) && r isa T)
        PT.test_iterate(z)
      end
    end
    # an argument that is not symmetric: only its upper or lower entry is not finite, or the two
    # entries of the pair differ. Its symmetric part (a + b)/2 is not finite in each case, and
    # Inf - Inf is NaN; the output is NaN, also in a dense matrix, which reads the entries as given.
    # A single infinite entry (0.5, Inf) and the like are in constrained_norml1_infinite.jl.
    for (a, b) in ((NaN, 0), (0, NaN), (Inf, -Inf), (-Inf, Inf),
                   (NaN, NaN), (NaN, Inf)), top in (0, 5)
      X = zeros(S, 3, 3)
      X[2, 2] = top
      X[1, 3], X[3, 1] = a, b                     # upper, lower
      @test isequal(ProximalBase._SymmetricPart(X)[1, 3], S(a + b) / 2)
      for y in (SymmetricSparseIterate(T, 3), zeros(T, 3, 3))
        v = value_and_prox!(y, c, X, T(1))
        @test allnan(Matrix(y)) && isnan(v) && v isa T
        y isa Array || PT.test_iterate(y)
        @test prox!(y, c, X, T(1)) === y && allnan(Matrix(y))
      end
    end
  end
  # a finite argument is unaffected, on both branches
  for S in (Float32, Float64), T in (Float32, Float64)
    c = Constrained(NormL1(T(1)), IndSphereL2(T(1)))
    for y in (zeros(T, n), SparseIterate(T, n))
      v = value_and_prox!(y, c, S[3, 0, -4, 0], T(1))     # shrink: (2, 0, -3, 0)/√13
      @test collect(y) ≈ T[2, 0, -3, 0] / sqrt(T(13)) && v ≈ 5 / sqrt(T(13)) && v isa T
      v = value_and_prox!(y, c, S[0.5, 0, -0.7, 0.2], T(1)) # fallback: the largest margin, -0.3
      @test collect(y) == T[0, 0, -1, 0] && v == 1 && v isa T
    end
  end
  # validation comes first (E1) and leaves the output unchanged (O6), NaN or not
  for T in (Float32, Float64)
    sphere = IndSphereL2(T(1))
    for (c, γ) in ((Constrained(NormL1(T(1), fill(T(Inf), n)), sphere), T(1)),  # no finite threshold
                   (Constrained(NormL1(floatmax(T)), sphere), T(4)),            # the threshold overflows
                   (Constrained(NormL1(T(1)), sphere), T(Inf)),                 # infinite step: the same
                   (Constrained(NormL1(T(1)), sphere), T(NaN))),                # NaN step (negative steps: #311)
        x in (T[NaN, 1, 0, 0], T[Inf, 1, 0, 0], T[1, 2, 0, 0])
      for y in (fill(T(7), n), SparseIterate(fill(T(7), n)))
        @test_throws ArgumentError value_and_prox!(y, c, x, γ)
        @test_throws ArgumentError prox!(y, c, x, γ)
        @test collect(y) == fill(T(7), n)
        y isa Array || PT.test_iterate(y)
      end
    end
  end
end

@testset "sphere prox of the ℓ1 norm: the margin comparison by one subtraction is exact (#459)" begin
  # _margin_better(n, t, bn, bt) decides n - t > bn - bt from the rounded margins where they
  # differ (rounding is monotone) and by _margin_greater where they are equal. Fixed cases where
  # the rounded margins tie but the exact ones do not, exact ties (false: the first entry stays),
  # subnormal and near-floatmax operands, and non-finite n or bn; each against the exact order.
  setprecision(BigFloat, 2300) do                 # exact for any four Float64 operands
    for T in (Float32, Float64)
      tiny, huge, e = nextfloat(zero(T)), floatmax(T), eps(T)
      cases = [(T(1), e / 4, T(1), zero(T)),       # rounded margins 1 and 1, exact 1 - e/4 < 1
               (T(1) + e, e / 4, T(1), zero(T)),   # 1 + e - e/4 > 1, rounded 1 + e > 1
               (T(3), e, T(3), e / 2),             # rounded tie, exact 3 - e < 3 - e/2
               (T(2), T(1), T(5), T(4)),           # exact tie
               (huge, huge / 2, huge / 2, zero(T)),# exact tie near floatmax
               (huge, prevfloat(huge), huge, huge - eps(huge) / 2),
               (3tiny, tiny, 2tiny, zero(T)),      # subnormal tie
               (3tiny, tiny, tiny, zero(T)), (tiny, zero(T), zero(T), tiny),
               (zero(T), zero(T), zero(T), zero(T)), (zero(T), huge, zero(T), prevfloat(huge))]
      rng = Random.Xoshiro(459)
      for _ in 1:500                               # margins equal after rounding, exact ones not
        bn, bt, t = T(abs(randn(rng))), T(abs(randn(rng))) * e, T(abs(randn(rng)))
        n = nextfloat(t + (bn - bt), rand(rng, -1:1))
        n >= 0 && push!(cases, (n, t, bn, bt))
      end
      for (n, t, bn, bt) in cases
        exact = big(n) - big(t) > big(bn) - big(bt)
        @test ProximalBase._margin_better(n, t, bn, bt) == exact == ProximalBase._margin_greater(n, t, bn, bt)
      end
      for n in (T(NaN), T(Inf), T(1)), bn in (T(NaN), T(Inf), T(2)), t in (zero(T), huge), bt in (zero(T), T(3))
        @test ProximalBase._margin_better(n, t, bn, bt) == ProximalBase._margin_greater(n, t, bn, bt)
      end
    end
  end
end

end
