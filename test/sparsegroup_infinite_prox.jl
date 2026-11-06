module SparseGroupInfiniteProxTest

# The proximal map of SparseGroupNormL2 with an infinite ℓ1 level (#620). The stored level fixes
# every grouped entry at zero at every step, γ = 0 and γ = Inf included, whatever the entry holds
# and whatever the group level and weight are (zero included); indices in no group are copied, a
# NaN there stays and reaches no group. The value is 0, the value at the stored output (O2). The
# step is not validated (#311): a negative or NaN step gives the same output. A finite level, also
# one that does not fit the data type (R3), keeps the ordinary map. A comparison with NaN selects no
# branch, so these tests never compare NaN with ==.

using Test, ProximalBase, SparseArrays, JLArrays, Adapt

# the same type, the same class (NaN, ±Inf, finite) and the same value
agree(a::Real, b::Real) = typeof(a) == typeof(b) && (isnan(a) || isnan(b) ? isnan(a) && isnan(b) : a == b)
agree(a::AbstractArray, b::AbstractArray) =
  size(a) == size(b) && all(i -> isnan(a[i]) || isnan(b[i]) ? isnan(a[i]) && isnan(b[i]) : a[i] == b[i], eachindex(a))

# the output of the infinite level: grouped entries zero, the others copied
function want(f::SparseGroupNormL2, x, ::Type{T} = float(eltype(x))) where {T}
  y = T.(collect(x))
  for G in f.groups, i in G
    y[i] = zero(T)
  end
  y
end

active_set(x::AtomIterate) = Set(k for k in 1:length(x.groups) if isactive(x, k))

# prox!, value_and_prox! into a separate output (filled with a sentinel), in place, and prox
function check(f, x, γ, wanty)
  T = eltype(wanty)
  y = fill(T(-7), size(x))
  v = value_and_prox!(y, f, copy(x), γ)
  @test agree(y, wanty)
  @test v === zero(T)
  @test agree(v, f(y))
  y = fill(T(-7), size(x))
  @test prox!(y, f, copy(x), γ) === y
  @test agree(y, wanty)
  if eltype(x) === T
    z = copy(x)
    u = value_and_prox!(z, f, z, γ)
    @test agree(z, wanty)
    @test u === zero(T)
    z = copy(x)
    prox!(z, f, z, γ)
    @test agree(z, wanty)
    @test agree(prox(f, x, γ), wanty)
  end
end

@testset "vectors and matrices, $T" for T in (Float32, Float64)
  nan = T(NaN)
  inf = T(Inf)
  # groups 1:2 and 4:5; index 3 is in no group and index 6 is beyond the last group
  groups = [1:2, 4:5]
  xs = (T[3, 4, 9, 1, 2, 5], T[nan, 4, 9, 1, 2, 5], T[3, inf, 9, -inf, 2, 5], T[nan, nan, nan, nan, nan, nan],
        T[0, -0.0, nan, 0, 0, inf], T[-inf, nan, -inf, inf, 1, nan], zeros(T, 6))
  steps = (zero(T), one(T), T(0.5), inf, -one(T), nan, T[0, 1], T[inf, 0.5], T[-1, nan])
  for λ2 in (zero(T), one(T), inf), w1 in (zero(T), one(T), inf), w2 in (zero(T), one(T), inf)
    f = SparseGroupNormL2(inf, λ2, groups, T[w1, w2])
    for x in xs, γ in steps
      check(f, x, γ, want(f, x))
    end
  end
  # the default weights; the example of the decision
  check(SparseGroupNormL2(inf, zero(T), groups), xs[2], zero(T), want(SparseGroupNormL2(inf, zero(T), groups), xs[2]))
  @test agree(prox(SparseGroupNormL2(Inf, 0.0, [1:2], [0.0]), T[3, 4, 9], 0.0), T[0, 0, 9])
  # groups that cover every index: the whole output is zero
  f = SparseGroupNormL2(inf, zero(T), [1:2, 3:3], T[1, 0])
  for γ in (zero(T), one(T), inf)
    check(f, T[nan, inf, 9], γ, zeros(T, 3))
  end
  # a matrix argument with linear-index groups
  f = SparseGroupNormL2(inf, zero(T), [[1, 2], [5]], T[1, 0])
  X = T[nan 3 inf; 4 nan 6]
  for γ in (zero(T), one(T), inf)
    check(f, X, γ, want(f, X))
  end
end

@testset "mixed precision" begin
  # Float64 parameters on Float32 data; a wider argument with a narrower output
  f = SparseGroupNormL2(Inf, 0.0, [1:2], [1.0])
  for γ in (0.0, 1.0, Inf, 0.0f0)
    check(f, Float32[3, NaN, 9], γ, Float32[0, 0, 9])
    y = fill(-7.0f0, 3)
    v = value_and_prox!(y, f, [3.0, NaN, 9.0], γ)
    @test agree(y, Float32[0, 0, 9])
    @test v === 0.0f0
    y = fill(-7.0, 3)
    v = value_and_prox!(y, SparseGroupNormL2(Inf32, 0.0f0, [1:2], [0.0f0]), Float32[Inf, 4, 9], γ)
    @test agree(y, [0.0, 0.0, 9.0])
    @test v === 0.0
  end
end

@testset "AtomIterate, $T" for T in (Float32, Float64)
  nan = T(NaN)
  inf = T(Inf)
  groups = [1:2, 3:3, 4:5]              # a partition of 1:5, as AtomIterate(x, f) requires
  xs = (T[3, 4, 9, 1, 2], T[nan, 4, 0, inf, 2], T[0, 0, nan, 0, 0], T[inf, -inf, 0, 0, 0], zeros(T, 5))
  for λ2 in (zero(T), one(T), inf), w in (T[0, 1, inf], T[1, 1, 1], T[0, 0, 0], T[inf, 0, 1])
    f = SparseGroupNormL2(inf, λ2, groups, w)
    for x in xs
      # the oracles: every entry zero, the activity of the argument, value 0
      for γ in (zero(T), one(T), inf, -one(T), T[0, 1, inf])
        xa = AtomIterate(copy(x), f)
        ya = AtomIterate(fill(T(-7), 5), f)
        v = value_and_prox!(ya, f, xa, γ)
        @test agree(collect(ya), zeros(T, 5))
        @test v === zero(T)
        @test agree(v, f(ya))
        @test active_set(ya) == active_set(xa)
        za = AtomIterate(copy(x), f)
        before = active_set(za)
        u = value_and_prox!(za, f, za, γ)
        @test agree(collect(za), zeros(T, 5))
        @test u === zero(T)
        @test active_set(za) == before
        # an AtomIterate with other groups takes the array path
        xb = AtomIterate(copy(x), [1:5])
        yb = AtomIterate(fill(T(-7), 5), [1:5])
        @test value_and_prox!(yb, f, xb, γ) === zero(T)
        @test agree(collect(yb), zeros(T, 5))
      end
      # the coordinate step on each group: that group zero, the others and their activity unchanged
      for k in 1:3, γ in (zero(T), one(T), inf, -one(T), nan)
        xa = AtomIterate(copy(x), f)
        before = active_set(xa)
        r = prox_coordinate!(xa, f, k, γ)
        wanty = copy(x)
        wanty[groups[k]] .= 0
        @test agree(collect(xa), wanty)
        @test agree(collect(r), zeros(T, length(groups[k])))
        @test active_set(xa) == before
      end
      # the block step with NaN and infinite staged values, from a strided and a non-strided vector
      for k in (1, 3), γ in (zero(T), one(T), inf)
        u = T[nan, inf, -inf][1:length(groups[k])]
        for staged in (u, view([u; u], collect(1:length(u))))
          xa = AtomIterate(copy(x), f)
          before = active_set(xa)
          prox_coordinate!(xa, f, k, staged, γ)
          wanty = copy(x)
          wanty[groups[k]] .= 0
          @test agree(collect(xa), wanty)
          @test active_set(xa) == before
        end
      end
    end
  end
end

@testset "SymmetricSparseIterate outputs, $T" for T in (Float32, Float64)
  nan = T(NaN)
  inf = T(Inf)
  # the symmetric pairs (1, 2) and (2, 3) are the groups; the diagonal and the pair (1, 3) are free
  groups = [[2, 4], [6, 8]]
  As = (T[1 2 3; 2 4 5; 3 5 6], T[1 nan 3; nan 4 5; 3 5 6], T[1 2 nan; 2 inf inf; nan inf 6])
  for λ2 in (zero(T), one(T), inf), w in (T[0, 1], T[1, 1], T[inf, 0])
    f = SparseGroupNormL2(inf, λ2, groups, w)
    for A in As, γ in (zero(T), one(T), inf, T[0, inf])
      wanty = want(f, A)
      for source in (Matrix, SparseIterate, SymmetricSparseIterate)
        y = SymmetricSparseIterate(T, 3)
        fill!(y, -1)
        v = value_and_prox!(y, f, source(copy(A)), γ)
        @test agree(Matrix(y), wanty)
        @test v === zero(T)
        y = SymmetricSparseIterate(T, 3)
        @test prox!(y, f, source(copy(A)), γ) === y
        @test agree(Matrix(y), wanty)
      end
      x = SymmetricSparseIterate(copy(A))
      @test value_and_prox!(x, f, x, γ) === zero(T)
      @test agree(Matrix(x), wanty)
    end
  end
end

@testset "device arrays (JLArrays), $T" for T in (Float32, Float64)
  nan = T(NaN)
  inf = T(Inf)
  groups = [1:2, 4:5]
  xs = (T[3, 4, 9, 1, 2, 5], T[nan, 4, 9, 1, 2, 5], T[inf, -inf, nan, 1, nan, 5], zeros(T, 6))
  for λ2 in (zero(T), one(T), inf), w in (T[0, 1], T[1, 1], T[1, inf], T[0, inf], T[0, 0])
    f = SparseGroupNormL2(inf, λ2, groups, w)
    # the host function on device arrays, its device copy, and the device copy on host arrays
    for g in (f, adapt(JLArray, f)), x in xs, γ in (zero(T), one(T), T(0.5), inf)
      wanty = want(f, x)
      y = JLArray(fill(T(-7), 6))
      v = value_and_prox!(y, g, JLArray(x), γ)
      @test agree(Array(y), wanty)
      @test v === zero(T)
      @test agree(v, g(y))
      y = JLArray(fill(T(-7), 6))
      prox!(y, g, JLArray(x), γ)
      @test agree(Array(y), wanty)
      z = JLArray(copy(x))
      @test value_and_prox!(z, g, z, γ) === zero(T)
      @test agree(Array(z), wanty)
      z = JLArray(copy(x))
      prox!(z, g, z, γ)
      @test agree(Array(z), wanty)
      g === f || check(g, x, γ, wanty)
    end
  end
end

@testset "wrappers, $T" for T in (Float32, Float64)
  nan = T(NaN)
  inf = T(Inf)
  for λ2 in (zero(T), one(T)), w in (T[1], T[0])
    f = SparseGroupNormL2(inf, λ2, [1:2], w)
    for x in (T[3, 4, 9], T[nan, 4, 9], T[inf, 4, 9]), γ in (zero(T), one(T))
      # a regularization: grouped entries zero, the free entry shrunk by the ridge term alone
      r = Regularize(f, T(1))
      y = fill(T(-7), 3)
      v = value_and_prox!(y, r, copy(x), γ)
      @test agree(y, T[0, 0, 9 / (1 + γ)])
      @test agree(v, r(y))
      # a scaling of the argument, a scaled function (an infinite level is never folded)
      for h in (Precompose(f, T(2)), Scaled(f, T(2)))
        check(h, x, γ, T[0, 0, 9])
      end
      # the conjugate (Moreau): the grouped entries pass, an infinite one included, the free entry is zero
      y = fill(T(-7), 3)
      value_and_prox!(y, Conjugate(f), copy(x), one(T))
      @test agree(y, [x[1], x[2], zero(T)])
      # with a ball constraint: the projection of the fixed output onto the ball
      c = Constrained(f, IndBallL2(one(T)))
      y = fill(T(-7), 3)
      v = value_and_prox!(y, c, copy(x), γ)
      @test agree(y, T[0, 0, 1])
      @test v === zero(T)
    end
    # per-group steps of a regularization over groups that cover every index, and its coordinate step
    g = SparseGroupNormL2(inf, λ2, [1:2, 3:3], [w; one(T)])
    r = Regularize(g, T(1))
    for γ in (T[0, 1], T[1, 0])
      y = fill(T(-7), 3)
      v = value_and_prox!(y, r, T[nan, inf, 9], γ)
      @test agree(y, zeros(T, 3))
      @test v === zero(T)
    end
    for x in (T[nan, 4, 9], T[0, 0, 9]), γ in (zero(T), one(T))
      xa = AtomIterate(copy(x), g)
      before = active_set(xa)
      prox_coordinate!(xa, r, 1, γ)
      @test agree(collect(xa), T[0, 0, 9])
      @test active_set(xa) == before
    end
    # the fiber function of a FiberSum: the first two entries of each fiber are zero
    fb = FiberSum(f)
    X = reshape(T[3, nan, 4, inf, 5, 6, 9, 8, 7, nan, 2, -inf], 2, 2, 3)
    for γ in (zero(T), one(T), inf)
      Y = fill(T(-7), 2, 2, 3)
      @test value_and_prox!(Y, fb, copy(X), γ) === zero(T)
      wanty = copy(X)
      wanty[:, :, 1:2] .= 0
      @test agree(Y, wanty)
    end
  end
end

# Controls: these pass before the change too.
@testset "finite levels keep the ordinary map, $T" for T in (Float32, Float64)
  nan = T(NaN)
  # the step 0 is the identity for a finite level, also with a zero group level
  f = SparseGroupNormL2(T(2), zero(T), [1:2], T[1])
  y = fill(T(-7), 3)
  @test value_and_prox!(y, f, T[3, 4, 9], zero(T)) === T(14)
  @test y == T[3, 4, 9]
  # near the range limits
  f = SparseGroupNormL2(floatmax(T), zero(T), [1:2], T[1])
  y = fill(T(-7), 3)
  @test value_and_prox!(y, f, T[1, 0, 9], one(T)) === zero(T)
  @test y == T[0, 0, 9]
  y = fill(T(-7), 3)
  @test value_and_prox!(y, f, T[1, 0, 9], zero(T)) === floatmax(T)
  @test y == T[1, 0, 9]
  y = fill(T(-7), 3)
  @test value_and_prox!(y, f, T[2, 0, 9], zero(T)) === T(Inf)
  @test y == T[2, 0, 9]
  # a positive value that underflows in a naive product stays positive
  f = SparseGroupNormL2(floatmin(T), zero(T), [1:2], T[1])
  y = fill(T(-7), 3)
  v = value_and_prox!(y, f, T[ldexp(one(T), -10), 0, 9], zero(T))
  @test y == T[ldexp(one(T), -10), 0, 9]
  @test v > 0
  @test agree(v, f(y))
  # a free NaN stays and reaches no group, for an infinite and a finite level
  f = SparseGroupNormL2(T(Inf), one(T), [1:2, 4:5], T[1, 1])
  y = fill(T(-7), 5)
  @test value_and_prox!(y, f, T[3, 4, nan, 1, 2], one(T)) === zero(T)
  @test agree(y, T[0, 0, nan, 0, 0])
  f = SparseGroupNormL2(T(0.5), one(T), [1:2, 4:5], T[1, 1])
  y = fill(T(-7), 5)
  value_and_prox!(y, f, T[3, 4, nan, 1, 2], one(T))
  @test agree(y, prox(f, T[3, 4, 0, 1, 2], one(T)) .+ T[0, 0, nan, 0, 0])
end

@testset "a finite level that does not fit the data type (R3), $T" for T in (Float32, Float64)
  # the stored level 2^128 is finite; on Float32 data it does not fit the data type
  f = SparseGroupNormL2(ldexp(1.0, 128), 0.0, [1:2], [1.0])
  x = T[ldexp(one(T), -100), 0, 9]
  @test f(x) === T(ldexp(1.0, 28))
  for g in (f, adapt(JLArray, f)), dev in (identity, JLArray)
    g === f || dev === JLArray || continue
    y = dev(fill(T(-7), 3))
    @test value_and_prox!(y, g, dev(copy(x)), 0.0) === T(ldexp(1.0, 28))
    @test Array(y) == x
    y = dev(fill(T(-7), 3))
    @test value_and_prox!(y, g, dev(copy(x)), 1.0) === zero(T)
    @test Array(y) == T[0, 0, 9]
  end
end

alloc_vprox(y, f, x, γ) = (value_and_prox!(y, f, x, γ); @allocated value_and_prox!(y, f, x, γ))
alloc_coordinate(x, f, k, γ) = (prox_coordinate!(x, f, k, γ); @allocated prox_coordinate!(x, f, k, γ))

@testset "inference and allocations, $T" for T in (Float32, Float64)
  x = T[3, 4, 9, 1, 2]
  y = similar(x)
  for f in (SparseGroupNormL2(T(Inf), zero(T), [1:2, 4:5], T[0, 1]), SparseGroupNormL2(T(0.5), one(T), [1:2, 4:5], T[1, 1]),
            SparseGroupNormL2(T(0.5), one(T), [1:2, 4:5], T[1, Inf]), SparseGroupNormL2(T(Inf), one(T), [1:2, 4:5], T[1, Inf]))
    for γ in (one(T), zero(T), T[1, 2])
      @test (@inferred value_and_prox!(y, f, x, γ)) isa T
      @test alloc_vprox(y, f, x, γ) == 0
    end
    g = SparseGroupNormL2(f.λ1, f.λ0, [1:2, 3:3, 4:5], [f.w[1], one(T), f.w[2]])
    ya = AtomIterate(zeros(T, 5), g)
    xa = AtomIterate(T[3, 4, 9, 1, 2], g)
    @test (@inferred value_and_prox!(ya, g, xa, one(T))) isa T
    @test alloc_vprox(ya, g, xa, one(T)) == 0
    @test alloc_coordinate(xa, g, 1, one(T)) == 0
  end
end

end
