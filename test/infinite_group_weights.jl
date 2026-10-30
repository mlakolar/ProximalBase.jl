module InfiniteGroupWeightsTest

using Test, ProximalBase, LinearAlgebra, SparseArrays
const PB = ProximalBase

families(T, λ, groups, w) = (GroupNormL2(T(λ), groups, w),
  GroupNormLinf(T(λ), groups, w), SparseGroupNormL2(zero(T), T(λ), groups, w),
  SparseGroupNormL2(T(0.25), T(λ), groups, w),
  GroupMCP(T(λ), groups, w), GroupSCAD(T(λ), groups, w))

@testset "infinite group weights" begin
  for T in (Float32, Float64), groups in ([[1, 2], [3]], [[1], [2, 3]]),
      w in (T[0, Inf], T[Inf, 0], T[1, Inf], T[Inf, 1]), λ in (0, 1, Inf)
    fixed = findfirst(isinf, w)
    other = 3 - fixed
    x = T[3, 4, 2]
    x0 = copy(x)
    x0[groups[fixed]] .= 0
    for f in families(T, λ, groups, w)
      l1 = f isa SparseGroupNormL2 ? f.λ1 : zero(T)
      # Isolate the ordinary group as a reference for its unchanged arithmetic.
      ref = PB.restrict(f, groups[other])
      expected_value = ref(x0[groups[other]])
      @test f(x) == T(Inf)
      @test f(x0) ≈ expected_value
      @test f(SparseIterate(sparsevec(x))) == T(Inf)
      @test f(SparseIterate(sparsevec(x0))) ≈ expected_value
      @test PB.kkt_violation(f, ones(T, 3), x0, first(groups[fixed])) == T(-Inf)
      if f isa Union{GroupNormL2, GroupMCP, GroupSCAD}
        @test PB.lla_weight(f, zero(T), fixed) == T(Inf)
        @test PB.lla_weight(f, one(T), fixed) == T(Inf)
        @test PB.lla_weights(f, x0)[fixed] == T(Inf)
      end
      for γ in (zero(T), one(T), T[0, 1], T[1, 0])
        expected = copy(x)
        expected[groups[fixed]] .= 0
        step = γ isa Real ? γ : γ[other]
        expected[groups[other]] = prox(ref, x[groups[other]], step)
        if iszero(w[other])
          @test expected[groups[other]] ≈ sign.(x[groups[other]]) .* max.(abs.(x[groups[other]]) .- step*l1, 0)
        elseif iszero(λ)
          @test expected[groups[other]] ≈ max.(x[groups[other]] .- step*l1, 0)
        end
        @test prox(f, x, γ) ≈ expected
        y = copy(x)
        @test value_and_prox!(y, f, y, γ) ≈ f(expected)
        @test y ≈ expected
        for input in (x, x0, zeros(T, 3))
          a = AtomIterate(input, f)
          input === x0 && PB.activate!(a, fixed)
          target = prox(f, input, γ)
          @test f(a) ≈ f(input)
          @test PB.kkt_violation(f, ones(T, 3), a, fixed) == T(-Inf)
          b = AtomIterate(T, 3, f)
          @test value_and_prox!(b, f, a, γ) ≈ f(target)
          @test Vector(b) ≈ target
          @test value_and_prox!(a, f, a, γ) ≈ f(target)
          @test Vector(a) ≈ target
        end
      end
      a = AtomIterate(x, f)
      PB.prox_coordinate!(a, f, fixed, zero(T))
      @test all(iszero, Vector(a)[groups[fixed]])
      # A different atom partition uses the generic group path.
      a = AtomIterate(x, [[1], [2], [3]])
      @test f(a) == T(Inf)
      @test Vector(prox(f, a, one(T))) ≈ prox(f, x, one(T))
      for S in ([3, 1], [2], [3, 2, 1], groups[other])
        r = PB.restrict(f, S)
        @test r(x[S]) ≈ f(T[i in S ? x[i] : 0 for i in eachindex(x)])
        z = prox(r, x[S], zero(T))
        for (j, i) in enumerate(S)
          @test z[j] ≈ (i in groups[fixed] ? zero(T) : x[i])
        end
      end
      if isfinite(λ)
        for z in (Scaled(f, zero(T)), Scaled(Scaled(f, T(2)), zero(T)))
          @test typeof(z) == typeof(f)
          @test z(x0) == zero(T)
          @test z(x) == T(Inf)
          @test prox(z, x, zero(T)) == x0
        end
        @test prox(Scaled(Scaled(f, T(2)), T(0.5)), x, one(T)) ≈ prox(f, x, one(T))
      end
      r = Regularize(f, one(T), ones(T, 3))
      @test prox(r, x, one(T)) ≈ prox(f, (x .+ one(T)) ./ 2, T(0.5))
    end
  end
end

@testset "group reproduction and zero scaling" begin
  for T in (Float32, Float64)
    groups = [[1], [2]]
    for f in families(T, 0, groups, T[1, Inf])
      f isa SparseGroupNormL2 && !iszero(f.λ1) && continue
      @test f(T[0.5, 2]) == T(Inf)
      @test f(T[0.5, 0]) == zero(T)
      @test prox(f, T[0.5, 2], one(T)) == T[0.5, 0]
    end
    for F in (GroupNormL2, GroupNormLinf, SparseGroupNormL2, GroupMCP, GroupSCAD)
      f = F === SparseGroupNormL2 ? F(one(T), one(T), groups, T[1, Inf]) : F(one(T), groups, T[1, Inf])
      v = F === GroupMCP ? T(0.5 - 0.25/6) : F === SparseGroupNormL2 ? one(T) : T(0.5)
      @test f(T[0.5, 0]) ≈ v
      @test f(T[0.5, 2]) == T(Inf)
      @test prox(f, T[0.5, 2], one(T)) == T[0, 0]
    end
    @test GroupMCP(one(T), groups, T[1, Inf])(T[0.5, 0]) ≈ T(0.5 - 0.25/6)
    @test GroupSCAD(one(T), groups, T[1, Inf])(T[0.5, 0]) == T(0.5)
    for f in families(T, 1, groups, T[1, 0])
      @test Scaled(f, zero(T)) isa ZeroFunction
      @test prox(f, T[0.5, 2], one(T))[2] == (f isa SparseGroupNormL2 ? 2 - f.λ1 : 2)
    end
  end
end

@testset "sphere with fixed groups" begin
  for T in (Float32, Float64), w in (T[Inf, 0], T[0, Inf]), λ in (0, 1)
    f = GroupNormL2(T(λ), [[1, 2], [3]], w)
    fixed = findfirst(isinf, w)
    other = 3 - fixed
    for x in (T[3, 4, 2], zeros(T, 3)), γ in (zero(T), one(T))
      c = Constrained(f, IndSphereL2(one(T)))
      y = similar(x)
      @test value_and_prox!(y, c, x, γ) == zero(T)
      @test all(iszero, y[f.groups[fixed]])
      @test norm(y) ≈ one(T)
      expected = zeros(T, 3)
      G = f.groups[other]
      if iszero(norm(x[G]))
        expected[first(G)] = one(T)
      else
        expected[G] = x[G] / norm(x[G])
      end
      @test y ≈ expected
    end
    f = GroupNormL2(T(λ), [[1], [2]], T[Inf, Inf])
    @test_throws ArgumentError prox(Constrained(f, IndSphereL2(one(T))), T[1, 2], one(T))
    c = Constrained(f, IndSphereL2(one(T)))
    @test prox(c, T[0, 0, 0], zero(T)) == T[0, 0, 1]
  end
end

@testset "sphere with an overflowing fixed group" begin
  for T in (Float32, Float64), γ in (zero(T), one(T))
    M = floatmax(T)
    f = GroupNormL2(zero(T), [[1, 2]], T[Inf])
    c = Constrained(f, IndSphereL2(one(T)))
    x = T[M, M, 0]
    y = similar(x)
    @test value_and_prox!(y, c, x, γ) == zero(T)
    @test y == T[0, 0, 1]
    @test c(y) == zero(T)
    # A finite-threshold group can also replace the overflowing fixed group.
    f = GroupNormL2(zero(T), [[1, 2], [3]], T[Inf, 1])
    @test prox(Constrained(f, IndSphereL2(one(T))), x, γ) == T[0, 0, 1]
  end
end

@testset "fixed radial groups skip unused scalar validation" begin
  for (F, a) in ((GroupSCAD, nextfloat(2.0)), (GroupMCP, 1e-100)),
      groups in ([[1]], [[1, 2], [3]]), λ in (0.0, 1.0, Inf)
    f = F(λ, groups, fill(Inf, length(groups)); a)
    x = Float32.(1:maximum(last, groups))
    expected = zeros(Float32, length(x))
    for γ in (0f0, 1f0, zeros(Float32, length(groups)), ones(Float32, length(groups)))
      @test prox(f, x, γ) == expected
      for input in (x, AtomIterate(x, f)), inplace in (false, true)
        source = copy(input)
        y = inplace ? source : copy(input)
        @test value_and_prox!(y, f, source, γ) == 0f0
        @test Vector(y) == expected
        @test f(y) == 0f0
        source = copy(input)
        y = inplace ? source : copy(input)
        prox!(y, f, source, γ)
        @test Vector(y) == expected
      end
    end
    for γ in (0f0, 1f0)
      xi = AtomIterate(x, f)
      for k in eachindex(groups)
        PB.prox_coordinate!(xi, f, k, γ)
        @test all(iszero, Vector(xi)[groups[k]])
      end
      @test Vector(xi) == expected
    end
  end
  for (F, a) in ((GroupSCAD, nextfloat(2.0)), (GroupMCP, 1e-100))
    # Infinite levels also bypass the scalar when the stored weight is finite.
    f = F(Inf, [[1]], [1.0]; a)
    @test prox(f, Float32[1], 1f0) == Float32[0]
    @test prox(f, Float32[1], 0f0) == Float32[1]
    # A later ordinary group must still fail before an earlier group is written.
    f = F(0.0, [[1], [2]], [Inf, 1.0]; a)
    for op in (prox!, value_and_prox!), γ in (0f0, Float32[0, 1]),
        atom in (false, true)
      x = atom ? AtomIterate(Float32[1, 2], f) : Float32[1, 2]
      y = atom ? AtomIterate(Float32[9, 9], f) : Float32[9, 9]
      @test_throws ArgumentError op(y, f, x, γ)
      @test Vector(y) == Float32[9, 9]
    end
    xi = AtomIterate(Float32[1, 2], f)
    PB.prox_coordinate!(xi, f, 1, 0f0)
    @test Vector(xi) == Float32[0, 2]
    @test_throws ArgumentError PB.prox_coordinate!(xi, f, 2, 0f0)
    @test Vector(xi) == Float32[0, 2]
    for k in (0, 3)
      @test_throws BoundsError PB.prox_coordinate!(xi, f, k, 0f0)
      @test Vector(xi) == Float32[0, 2]
    end
    other = AtomIterate(Float32[1, 2], [[1, 2]])
    @test_throws DimensionMismatch PB.prox_coordinate!(other, f, 1, 0f0)
    @test Vector(other) == Float32[1, 2]
  end
end

@testset "ordinary radial validation before writes" begin
  for F in (GroupMCP, GroupSCAD), bad in (NaN, -1.0),
      op in (prox!, value_and_prox!), γ in (0f0, Float32[0, 1]), atom in (false, true)
    f = F(0.3, [[1], [2]], [1.0, 1.0])
    f.w[2] = bad
    x = atom ? AtomIterate(Float32[1, 2], f) : Float32[1, 2]
    y = atom ? AtomIterate(Float32[9, 9], f) : Float32[9, 9]
    @test_throws ArgumentError op(y, f, x, γ)
    @test Vector(y) == Float32[9, 9]
  end
end

@testset "overflowing radial coefficients" begin
  for F in (GroupMCP, GroupSCAD), (T, λ, w) in (
      (Float32, floatmax(Float32), Float32[2, 0]),
      (Float64, floatmax(Float64), Float64[2, 0]),
      (Float32, Float64(floatmax(Float32)), [2.0, 0.0]))
    f = F(λ, [[1, 2], [3]], w)
    x = T[3, 4, 2]
    x0 = T[0, 0, 2]
    for input in (x, AtomIterate(x, f)), zeroed in (x0, AtomIterate(x0, f))
      @test f(input) == T(Inf)
      @test f(zeroed) == zero(T)
    end
    for γ in (zero(T), one(T), T[0, 1], T[1, 0])
      step = γ isa Real ? γ : γ[1]
      expected = iszero(step) ? x : x0
      expected_value = iszero(step) ? T(Inf) : zero(T)
      for input in (x, AtomIterate(x, f)), inplace in (false, true)
        source = copy(input)
        y = inplace ? source : copy(input)
        @test value_and_prox!(y, f, source, γ) == expected_value
        @test Vector(y) == expected
        @test f(y) == expected_value
        source = copy(input)
        y = inplace ? source : copy(input)
        prox!(y, f, source, γ)
        @test Vector(y) == expected
      end
    end
    xi = AtomIterate(x, f)
    PB.prox_coordinate!(xi, f, 1, one(T))
    @test Vector(xi) == x0
  end
  for F in (GroupMCP, GroupSCAD), T in (Float32, Float64)
    @test F(floatmax(T), [[1]], T[2])(T[0]) == zero(T)
  end
end

@testset "mixed sparse group weights preserve NaN" begin
  for T in (Float32, Float64), λ1 in (zero(T), T(0.25)),
      groups in ([[1], [2]], [[1, 2], [3]]), w in (T[1, Inf], T[Inf, 1])
    f = SparseGroupNormL2(λ1, zero(T), groups, w)
    fixed = findfirst(isinf, w)
    free = 3 - fixed
    x = fill(T(3), maximum(last, groups))
    x[first(groups[free])] = T(NaN)
    for γ in (zero(T), one(T), T[0, 1], T[1, 0])
      step = γ isa Real ? γ : γ[free]
      expected = copy(x)
      expected[groups[fixed]] .= 0
      for j in groups[free]
        isnan(x[j]) || (expected[j] = max(x[j] - step * λ1, 0))
      end
      for input in (x, AtomIterate(x, f)), inplace in (false, true)
        source = copy(input)
        y = inplace ? source : copy(input)
        v = value_and_prox!(y, f, source, γ)
        @test isequal(Vector(y), expected)
        @test isequal(v, f(y))
        @test iszero(λ1) ? iszero(v) : isnan(v)
        source = copy(input)
        y = inplace ? source : copy(input)
        prox!(y, f, source, γ)
        @test isequal(Vector(y), expected)
      end
    end
  end
end

@testset "radial classification visits active groups" begin
  for F in (GroupMCP, GroupSCAD), T in (Float32, Float64),
      w in (T[1, Inf], T[1, floatmax(T)])
    f = F(T(2), [[1, 2], [3, 4]], w)
    x = AtomIterate(T[3, 4, 0, 0], f)
    @test PB._ordinary_radial_groups(f, T, x.active, x.nactive)
    @test !PB._ordinary_radial_groups(f, T)
    @test f(x) == f(Vector(x))
    for γ in (zero(T), one(T), T[0, 1]), inplace in (false, true)
      source = copy(x)
      y = inplace ? source : copy(x)
      expected = prox(f, Vector(x), γ)
      v = value_and_prox!(y, f, source, γ)
      @test Vector(y) == expected
      @test v ≈ f(y)
      source = copy(x)
      y = inplace ? source : copy(x)
      prox!(y, f, source, γ)
      @test Vector(y) == expected
    end
    PB.atom(x, 2) .= T[3, 4]
    PB.activate!(x, 2)
    @test !PB._ordinary_radial_groups(f, T, x.active, x.nactive)
    @test f(x) == T(Inf)
    @test Vector(prox(f, x, one(T)))[3:4] == zeros(T, 2)
  end
end

@testset "finite nonzero group levels retain base values" begin
  for T in (Float32, Float64), w in (T[1, Inf], T[Inf, 1])
    fixed = findfirst(isinf, w)
    free = 3 - fixed
    x0 = zeros(T, 2)
    x0[free] = T(0.5)
    x = copy(x0)
    x[fixed] = T(2)
    for f in (GroupNormL2(one(T), [[1], [2]], w),
        GroupNormLinf(one(T), [[1], [2]], w),
        SparseGroupNormL2(one(T), one(T), [[1], [2]], w))
      expected = f isa SparseGroupNormL2 ? one(T) : T(0.5)
      for input in (x0, AtomIterate(x0, f), AtomIterate(x0, [[1, 2]]))
        @test f(input) == expected
      end
      for input in (x, AtomIterate(x, f), AtomIterate(x, [[1, 2]]))
        @test f(input) == T(Inf)
      end
    end
  end
  for f in (GroupNormL2(1e300, [[1]], [1e10]),
      GroupNormLinf(1e300, [[1]], [1e10]),
      SparseGroupNormL2(1.0, 1e300, [[1]], [1e10]))
    x = [1e-300]
    @test f(x) ≈ 1e10
    @test f(AtomIterate(x, f)) ≈ 1e10
  end
end

end
